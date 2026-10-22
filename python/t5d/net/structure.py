"""The attention structure of stage N2 (spec section 6): masks and biases, built per query block.

Everything here is computed from the input contract's sparse lists - the typed edges
``(src_token, dst_token, relation)`` of ``state.py`` - and from small per-board tables,
and is turned into dense rows only for one query block at a time (``rows``,
``role_rows``, ``bias_rows``), so no ``(L, L)`` tensor per head is ever needed (B14,
``attention.py``, which under autograd rebuilds a block's rows in the backward pass
instead of keeping them). The rules engine will define the relations; the network only
consumes the lists.

**Head roles** (``NetConfig.head_roles``). Without the routing table, B9 (committed):
heads ``0..k-1`` carry the move-graph mask (``move_graph_heads``, default 2), the rest are
dense. With the routing table, B10 (prototype): section 6's table, one role per head
(``rope_head_roles``; under B5 the same entry sets the head's rotation - see ``rope.py``).

**The key set of a query on a masked head** - exactly, for valid query ``i`` and valid
key ``j`` (padding keys are never allowed):

* ``i`` itself, always - so no valid token's row is ever empty;
* a square query, by role:

  - ``move_graph``: ``j`` is a square joined to ``i`` by an edge of relation ``move``, in
    the direction ``move_graph_direction`` selects: ``forward`` - an edge ``(i, j)``;
    ``reverse`` - an edge ``(j, i)``; ``symmetric`` - either;
  - ``relations``: the same over the edges whose relation is in ``relations_head_types``,
    in the direction ``relations_head_direction`` selects (``move_graph_direction``'s by
    default);
  - ``same_board``: ``j`` is any token of ``i``'s board - its squares and its per-board
    registers;
  - ``same_square``: ``j`` is a square with ``i``'s (file, rank), on any board;

* a per-board register query on a ``same_board`` head: any token of its own board, as
  for the board's squares - a per-board register is on its board (``same_board_registers=
  "own_board"``, which stands as v1 - M20; ``"registers"``: as the next item);
* every other register query - a per-board register on the ``move_graph``, ``relations``
  and ``same_square`` heads, whose relations join squares only, and a global register,
  which holds no board, on every masked head: every valid key
  (``masked_head_registers="read"``) or only itself (``"self"``);
* a padding query: every valid key (its output is discarded; this keeps every row of
  the structure non-empty).

Dense heads allow every valid key. Edges join squares of one position; a position's
lists never reach another position of the batch.

**Block sparsity** (B14, section 6.1; ``key_selection``, ``block_plan``). In board-major
order a board's tokens are contiguous, and board-to-board reachability is a block-level
relation: for a query block and a masked role, the key boards that any of the block's
queries can reach are a short list of board slices. The block-sparse attention path hands
the kernel those boards only. A query whose key set is every valid key (a register under
``"read"`` - ``free_rows``) would select every board, so it is computed apart, against
every key; a padding query selects nothing. The masks depend on the position, never on
the layer, so each block's plan - its selection and its mask on it - is built once per
structure and reused by every layer; a block that would select more than a set fraction
of the keys is computed against every key instead, so the plans of a role hold at most
that fraction of an ``(L, L)`` bool mask in all (and, for a move graph whose boards reach
a bounded number of boards, an amount that grows linearly in N).

**Typed edge biases** (B11, provisional, ``typed_edge_bias``): a learned scalar per head
per relation type (``TYPED_RELATIONS``: attack, defence, pin, x-ray, check - travel-threat
folded into attack at CHECKPOINT 6), shared by all layers, added on every head to the logits
of the related pairs: for an edge ``(s, d, type)``, query ``s`` / key ``d`` gets the type's scalar for
"query is the source" and query ``d`` / key ``s`` the scalar for "query is the target" -
one scalar for both unless ``typed_edge_directed``. A pair related several ways gets the
sum; an edge repeated in a list counts once (a relation holds or it does not). On a
masked head the bias reaches only allowed pairs, so a type that head's mask never opens
keeps a scalar no gradient reaches - defence on a move-graph head, for one (own pieces
are never move targets); "per head per type" is kept literally, every head has every
type's scalar. Cross-board edges are included (a travel threat is an attack edge across
boards). The network takes whatever the rules engine labels each type: it builds attack,
defence and x-ray today; pin and check are its ABI round 1's, and until then no position
from the engine carries them.

**Board-pair biases** (``pair_table`` ``(B, H, Nb + 1, Nb + 1)``; row and column ``Nb``
stand for "no board" and are zero): B12's smolgen (``smolgen.py``) generates a table per
layer from that layer's input. A token pair's bias is the table at (query's board, key's
board); global registers and padding have no board and get none.

**B15's displacement bias** (provisional, the v1 default; ``displacement.py``): one learned
scalar per head per layer, indexed by the pair's displacement class. Its rows come from a
per-batch ``DisplacementIndex`` - each token's slot and code, each board pair's tile - and a
static class map: entry = ``class_map[tile(query board, key board), code(query), code(key)]``,
then the layer's table at that entry. The table is per layer, so the structure must be made
for its layer (``for_layer(h, layer)``) before any bias row is built. Section 15.2's B15
row compares these rows with B14's score function (``flex.py``), which computes the same
entries by arithmetic on the coordinates, and with the bias materialised by hand.

**Per layer.** ``for_layer(h, layer)`` adds what one layer's structure needs: B12's table,
generated from the layer's attention input ``h``, B15's table of layer ``layer``, and B16's
when it is per layer. Masks never depend on the layer; ``mask_rows`` builds them without it.

**Under the block-causal scope** (stage N3, ``scope.py``) the edge-derived masks and
biases (B9's move graph, B10's relations head, B11's typed biases) and B12's smolgen
biases open and bias live query rows only (``build_structure(live=...)``): they are
generated per position, and a frozen token that read through them would read the
present. ``scope.py`` documents the reading. The geometric roles (same board, same
square) and B15's displacement bias - a function of the two tokens' coordinates alone -
apply to every row, frozen ones included.

**B16's status key biases** (committed with the cached configuration; ``status.py``) exist
under the block-causal scope only: on a live query row, a frozen board's token as key gets
its head's entry for that board's status - Delta-t-to-present bin and active flag, from the
batch's own present - and nothing else gets anything (``status_rows``). A frozen board's rows
carry neither B12 nor B16 (``for_frozen_rows``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F
from torch import nn

from .config import TYPED_RELATIONS, NetConfig
from .displacement import DisplacementBias, DisplacementIndex, displacement_index, lookup
from .smolgen import BoardPairSmolgen, SmolgenInputs, smolgen_inputs
from .state import (
    RELATION_IDS,
    RELATION_TYPES,
    TOKEN_BOARD_REGISTER,
    TOKEN_GLOBAL_REGISTER,
    TOKEN_PAD,
    TOKEN_SQUARE,
    Batch,
)
from .status import StatusIndex, StatusKeyBias, status_index

#: The roles that carry a mask (every role but "dense").
MASK_ROLES: tuple[str, ...] = ("move_graph", "same_board", "same_square", "relations")

#: B11's index (``TYPED_RELATIONS``) of each relation id of ``state.RELATION_TYPES``; -1
#: for ``move``, which is B9's mask and carries no typed bias.
TYPED_INDEX: tuple[int, ...] = tuple(TYPED_RELATIONS.index(name) if name in TYPED_RELATIONS else -1 for name in RELATION_TYPES)


class TypedEdgeBias(nn.Module):
    """B11's learned scalars: ``table`` ``(n_heads, n_types, orientations)``."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.directed = cfg.typed_edge_directed
        self.table = nn.Parameter(torch.empty(cfg.n_heads, len(TYPED_RELATIONS), 2 if self.directed else 1))


@dataclass
class Pairs:
    """Directed token pairs ``(position b, query q, key k)``, sorted by query, with an
    optional typed-relation index ``t`` and orientation ``o`` (0: the query is the
    relation's source, 1: its target) per pair."""

    b: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    t: torch.Tensor | None = None
    o: torch.Tensor | None = None

    @classmethod
    def sorted(cls, b, q, k, t=None, o=None) -> Pairs:
        order = torch.argsort(q, stable=True)
        pick = lambda x: None if x is None else x[order]  # noqa: E731
        return cls(b[order], q[order], k[order], pick(t), pick(o))

    def span(self, start: int, stop: int) -> slice:
        """The pairs whose query lies in ``[start, stop)``."""
        bounds = torch.tensor([start, stop], dtype=self.q.dtype, device=self.q.device)
        lo, hi = torch.searchsorted(self.q, bounds).tolist()
        return slice(lo, hi)

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def live_queries(self, live: torch.Tensor | None) -> Pairs:
        """The pairs whose query is a live token (``live`` ``(B, L)`` bool; None: all)."""
        if live is None:
            return self
        keep = live[self.b, self.q]
        pick = lambda x: None if x is None else x[keep]  # noqa: E731
        return Pairs(self.b[keep], self.q[keep], self.k[keep], pick(self.t), pick(self.o))


def _endpoints(batch: Batch, relation_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(b, src, dst, relation)`` of the valid edges with the given relations."""
    rel = batch.edges[..., 2]
    keep = batch.edge_valid & torch.isin(rel, torch.tensor(relation_ids, device=rel.device))
    b, e = torch.nonzero(keep, as_tuple=True)
    return b, batch.edges[b, e, 0], batch.edges[b, e, 1], rel[b, e]


def mask_pairs(batch: Batch, relation_ids: list[int], direction: str) -> Pairs:
    """The (query, key) pairs the edges of these relations open, by ``direction``."""
    b, src, dst, _ = _endpoints(batch, relation_ids)
    if direction == "forward":
        return Pairs.sorted(b, src, dst)
    if direction == "reverse":
        return Pairs.sorted(b, dst, src)
    return Pairs.sorted(torch.cat([b, b]), torch.cat([src, dst]), torch.cat([dst, src]))


def typed_pairs(batch: Batch, directed: bool) -> Pairs:
    """B11's biased pairs: both orientations of every distinct typed edge."""
    ids = [i for i, t in enumerate(TYPED_INDEX) if t >= 0]
    b, src, dst, rel = _endpoints(batch, ids)
    length, n_rel = batch.n_tokens, len(RELATION_TYPES)
    key = torch.unique(((b * length + src) * length + dst) * n_rel + rel)  # a relation holds once
    rel, rest = key % n_rel, key // n_rel
    dst, rest = rest % length, rest // length
    src, b = rest % length, rest // length
    t = torch.tensor(TYPED_INDEX, device=rel.device)[rel]
    o_rev = torch.ones_like(t) if directed else torch.zeros_like(t)
    return Pairs.sorted(
        torch.cat([b, b]), torch.cat([src, dst]), torch.cat([dst, src]), torch.cat([t, t]), torch.cat([torch.zeros_like(t), o_rev])
    )


def _positions(index: torch.Tensor, real: torch.Tensor | None, length: int) -> torch.Tensor:
    """``(B, length)`` long: where each token sits in ``index`` ``(B, n)`` (its real entries
    only, ``real`` ``(B, n)``; all when None), -1 for a token not in it."""
    bsz, n = index.shape
    pos = torch.full((bsz, length), -1, dtype=torch.long, device=index.device)
    slot = torch.arange(n, device=index.device).expand(bsz, n)
    b = torch.arange(bsz, device=index.device)[:, None].expand(bsz, n)
    if real is None:
        pos[b, index] = slot
    else:
        pos[b[real], index[real]] = slot[real]
    return pos


@dataclass
class BlockPlan:
    """One query block's block-sparse plan on a masked role (``AttnStructure.block_plan``);
    every field None for a dense block (computed against every key)."""

    index: torch.Tensor | None  # (B, K) long: the selected kernel keys, each position its own
    valid: torch.Tensor | None  # (B, K) bool: real entries of ``index``
    allowed: torch.Tensor | None  # (B, c, K) bool: the role's mask on the selection


@dataclass
class AttnStructure:
    """Section 6's masks and biases for one batch, rows built on demand (``rows``)."""

    head_roles: tuple[str, ...]
    registers: str  # masked_head_registers
    kind: torch.Tensor  # (B, L) long
    board_idx: torch.Tensor  # (B, L) long: board index, n_boards where the token has none
    square_id: torch.Tensor  # (B, L) long: a (file, rank) key for squares, -1 otherwise
    n_boards: int = 0  # boards of the batch's largest position: the "no board" slot
    same_board_registers: str = "own_board"
    move: Pairs | None = None  # B9 / the move-graph role
    rel: Pairs | None = None  # B10's relations role
    typed: Pairs | None = None  # B11
    typed_table: torch.Tensor | None = None  # (H, n_types, orientations)
    smolgen: BoardPairSmolgen | None = None  # B12
    smolgen_in: SmolgenInputs | None = None
    pair_table: torch.Tensor | None = None  # this layer's board-pair table (B12; set by ``for_layer``)
    #: (B, Nb) bool: the query boards whose rows carry B12's biases (None: every board) -
    #: the live boards under the block-causal scope.
    smolgen_rows: torch.Tensor | None = None
    disp: DisplacementBias | None = None  # B15
    disp_index: DisplacementIndex | None = None  # B15's per-batch slots, codes and tiles
    disp_table: torch.Tensor | None = None  # (H, entries + 1): this layer's B15 table (set by ``for_layer``)
    status: StatusKeyBias | None = None  # B16 (the block-causal scope only)
    status_index: StatusIndex | None = None  # B16's per-token status entries and live rows
    #: (H, statuses + 1): B16's table - shared by every layer (set when built) or this layer's
    #: (set by ``for_layer``).
    status_table: torch.Tensor | None = None
    #: Block-sparse plans by (role, block, kernel keys, fraction), shared by every layer's
    #: copy of the structure (``for_layer`` copies the reference).
    plans: dict = field(default_factory=dict, repr=False)

    # -- what the attention paths ask --

    @property
    def structured(self) -> bool:
        return any(role != "dense" for role in self.head_roles)

    @property
    def has_bias(self) -> bool:
        return self.typed is not None or self.disp is not None or self.smolgen is not None or self.status is not None

    @property
    def layer_dependent(self) -> bool:
        """Whether the bias rows differ by layer: B12's table (from each layer's input), B15's
        table (one per layer) and B16's when it is per layer."""
        return self.smolgen is not None or self.disp is not None or (self.status is not None and self.status.per_layer)

    def head_groups(self) -> list[tuple[str, torch.Tensor]]:
        """Heads grouped by role, in order of first appearance: ``[(role, head indices)]``."""
        roles = list(dict.fromkeys(self.head_roles))
        device = self.kind.device
        return [(r, torch.tensor([h for h, x in enumerate(self.head_roles) if x == r], device=device)) for r in roles]

    def for_layer(self, h: torch.Tensor | None, layer: int) -> AttnStructure:
        """This structure for layer ``layer``, whose attention input is ``h`` ``(B, L, d)``:
        B12's board-pair table generated from ``h`` (which only B12 needs), B15's table of
        that layer and B16's when it is per layer."""
        changes: dict = {}
        if self.smolgen is not None:
            if h is None:
                raise ValueError("board-pair smolgen (B12) generates each layer's table from that layer's attention input")
            table = self.smolgen(h, self.smolgen_in).float()
            if self.smolgen_rows is not None:
                table = table * self.smolgen_rows[:, None, :, None].to(table.dtype)
            changes["pair_table"] = F.pad(table, (0, 1, 0, 1))
        if self.disp is not None:
            changes["disp_table"] = self.disp.layer_table(layer)
        if self.status is not None and self.status.per_layer:
            changes["status_table"] = self.status.layer_table(layer)
        return replace(self, **changes) if changes else self

    def for_frozen_rows(self) -> AttnStructure:
        """This structure with B12 and B16 left out - what a frozen board's rows carry under the
        block-causal scope (``cache.py``), where both bias live rows only."""
        return replace(
            self, smolgen=None, smolgen_in=None, pair_table=None, smolgen_rows=None, status=None, status_index=None, status_table=None
        )

    # -- which queries a masked role's relation covers --

    def uncovered_registers(self, role: str) -> torch.Tensor:
        """``(B, L)`` bool: the register tokens that a masked role's relation does not cover
        as queries - they follow ``masked_head_registers``. The same-board relation covers
        per-board registers (they are on their board) under ``same_board_registers=
        "own_board"``; no relation covers a global register."""
        if role == "same_board" and self.same_board_registers == "own_board":
            return self.kind == TOKEN_GLOBAL_REGISTER
        return (self.kind == TOKEN_BOARD_REGISTER) | (self.kind == TOKEN_GLOBAL_REGISTER)

    def free_rows(self, role: str) -> torch.Tensor | None:
        """``(B, L)`` bool: the queries that read every valid key on a masked role's heads -
        uncovered registers under ``masked_head_registers="read"`` - or None when there
        are none. (Padding queries read every valid key too; their output is discarded.)"""
        if self.registers != "read":
            return None
        free = self.uncovered_registers(role)
        return free if bool(free.any()) else None

    # -- rows --

    def role_rows(self, role: str, start: int, stop: int) -> torch.Tensor:
        """Allowed keys ``(B, stop - start, L)`` of the queries ``[start, stop)`` on a head
        of a masked role (padding keys still to be removed by the caller)."""
        bsz, length = self.kind.shape
        c = stop - start
        device = self.kind.device
        ident = (torch.arange(start, stop, device=device)[:, None] == torch.arange(length, device=device)[None, :]).expand(bsz, c, length)
        if role in ("move_graph", "relations"):
            pairs = self.move if role == "move_graph" else self.rel
            rel = ident.clone()
            sl = pairs.span(start, stop)
            rel[pairs.b[sl], pairs.q[sl] - start, pairs.k[sl]] = True
        elif role == "same_board":
            on_board = (self.kind == TOKEN_SQUARE) | (self.kind == TOKEN_BOARD_REGISTER)
            rel = (self.board_idx[:, start:stop, None] == self.board_idx[:, None, :]) & on_board[:, None, :] | ident
        elif role == "same_square":
            key_sq = self.square_id[:, None, :]
            rel = (self.square_id[:, start:stop, None] == key_sq) & (key_sq >= 0) | ident
        else:
            raise ValueError(f"role {role!r} carries no mask")
        # The relation above covers squares (and, on the same-board head, per-board
        # registers - they are on their board: their rows are already right). The rows it
        # does not cover are fixed in place: padding reads every key; an uncovered register
        # every key under "read", only itself under "self".
        kind_q = self.kind[:, start:stop]
        pad = kind_q == TOKEN_PAD
        if bool(pad.any()):
            rel[pad] = True
        uncovered = self.uncovered_registers(role)[:, start:stop]
        if bool(uncovered.any()):
            if self.registers == "read":
                rel[uncovered] = True
            else:
                b_idx, r_idx = torch.nonzero(uncovered, as_tuple=True)
                rel[b_idx, r_idx] = False
                rel[b_idx, r_idx, start + r_idx] = True
        return rel

    def key_selection(self, role: str, start: int, stop: int, allowed: torch.Tensor, n_keys: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Block sparsity at board granularity (B14, section 6.1): the kernel keys among
        ``[0, n_keys)`` that the queries ``[start, stop)`` read on a masked role's heads, as
        ``(index, valid)`` ``(B, K)`` - each position's keys in token order, padded to the
        batch's longest selection (``valid`` False on the padding entries).

        ``allowed`` ``(B, stop - start, L)`` is ``role_rows`` with padding keys removed.
        Every token of each board that a selecting query reaches is selected - whole
        boards are section 6.1's blocks - and each reached token that has no board on its
        own (none is a kernel key since M20 left the global registers position-free - B8 -
        to the merge; kept so that the selection stays lossless for any key set). The
        selecting queries are the valid ones that are not free
        (``free_rows``): a free query reads every key and is computed apart, a padding
        query's output is discarded. The selection is lossless: every key a selecting
        query may read is selected."""
        bsz = allowed.shape[0]
        selecting = self.selecting_rows(role, start, stop)
        reach = (allowed[..., :n_keys] & selecting[..., None]).any(dim=1)  # (B, n_keys)
        slot = self.n_boards  # "no board"
        board = self.board_idx[:, :n_keys]
        boards = torch.zeros(bsz, slot + 1, dtype=torch.bool, device=board.device)
        boards.scatter_(1, torch.where(reach, board, slot), True)
        boards[:, slot] = False
        cols = boards.gather(1, board) | (reach & (board == slot))
        # Compaction in token order: each selected key's place is the count of selected
        # keys before it; unselected keys are written to a dump column that is dropped.
        counts = cols.sum(dim=1)
        k = int(counts.max())
        place = torch.where(cols, torch.cumsum(cols, dim=1) - 1, torch.full_like(board, k))
        index = torch.zeros(bsz, k + 1, dtype=torch.long, device=board.device)
        index.scatter_(1, place, torch.arange(n_keys, device=board.device).expand(bsz, n_keys))
        valid = torch.arange(k, device=board.device)[None, :] < counts[:, None]
        return index[:, :k], valid

    def selecting_rows(self, role: str, start: int, stop: int) -> torch.Tensor:
        """``(B, stop - start)`` bool: the queries whose keys choose their block's key
        selection on a masked role's heads - valid and not free (``free_rows``)."""
        selecting = self.kind[:, start:stop] != TOKEN_PAD
        free = self.free_rows(role)
        return selecting if free is None else selecting & ~free[:, start:stop]

    def block_plan(self, role: str, start: int, stop: int, n_keys: int, max_fraction: float) -> BlockPlan:
        """The block-sparse plan of the queries ``[start, stop)`` on a masked role's heads,
        over the kernel keys ``[0, n_keys)`` (``key_selection``): built once per structure
        and reused by every layer - the masks depend on the position, never on the layer.
        A block whose selection exceeds ``max_fraction`` of the kernel keys gets a dense
        plan (``index`` None): it is computed against every key, its rows rebuilt per
        layer, so that no plan holds more than ``max_fraction`` of a block's (rows, L). So
        does a block where a selecting query may read a key past ``n_keys`` (a
        position-free global register, which the kernel keys leave to the global merge):
        none does under the documented key sets - no relation reaches a global register -
        and the dense path merges exactly if one ever did."""
        key = (role, start, stop, n_keys, max_fraction)
        plan = self.plans.get(key)
        if plan is not None:
            return plan
        valid_keys = self.kind != TOKEN_PAD
        rel = self.role_rows(role, start, stop) & valid_keys[:, None, :]
        index, valid = self.key_selection(role, start, stop, rel, n_keys)
        selecting = self.selecting_rows(role, start, stop)
        if index.shape[1] > max_fraction * n_keys or bool((rel[..., n_keys:] & selecting[..., None]).any()):
            plan = BlockPlan(None, None, None)
        else:
            allowed = rel[..., :n_keys].gather(2, index[:, None, :].expand(-1, stop - start, -1)) & valid[:, None, :]
            plan = BlockPlan(index, valid, allowed)
        self.plans[key] = plan
        return plan

    def rows(self, start: int, stop: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Allowed keys ``(B, H, c, L)`` (None: every head dense) and additive bias
        ``(B, H, c, L)`` (None: no bias) of the queries ``[start, stop)``."""
        return self.mask_rows(start, stop), self.bias_rows(start, stop)

    def mask_rows(self, start: int, stop: int) -> torch.Tensor | None:
        """Allowed keys ``(B, H, c, L)`` of the queries ``[start, stop)`` (None: every head
        dense) - the same at every layer."""
        if not self.structured:
            return None
        cache: dict[str, torch.Tensor] = {}
        for role in self.head_roles:
            if role != "dense" and role not in cache:
                cache[role] = self.role_rows(role, start, stop)
        any_rows = next(iter(cache.values()))
        dense = torch.ones_like(any_rows)
        return torch.stack([cache.get(role, dense) for role in self.head_roles], dim=1)

    def bias_rows(
        self,
        start: int,
        stop: int,
        heads: torch.Tensor | None = None,
        cols: torch.Tensor | None = None,
        cols_valid: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The additive bias ``(B, Hg, c, Lk)`` of the queries ``[start, stop)`` on the heads
        ``heads`` (every head when None), over every key (``Lk = L``) or over the keys
        ``cols`` ``(B, K)`` (``Lk = K``; ``cols_valid`` marks the real entries); None when
        nothing adds a bias. Over a key selection, a typed pair whose key is not selected
        is left out - its key is masked for its query."""
        bsz = self.kind.shape[0]
        rows = torch.arange(start, stop, device=self.kind.device).expand(bsz, stop - start)
        return self._bias(rows, None, (start, stop), heads, cols, cols_valid)

    def bias_rows_at(
        self, rows: torch.Tensor, rows_real: torch.Tensor | None = None, heads: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """The additive bias ``(B, Hg, n, L)`` of the queries ``rows`` ``(B, n)`` (each
        position's own rows; ``rows_real`` marks the real entries, the others are computed
        and meant to be discarded)."""
        return self._bias(rows, rows_real, None, heads, None, None)

    def _bias(self, rows, rows_real, span, heads, cols, cols_valid) -> torch.Tensor | None:
        if self.smolgen is not None and self.pair_table is None:
            raise RuntimeError("this structure's board-pair smolgen needs the layer input: use AttnContext.for_layer")
        if self.disp is not None and self.disp_table is None:
            raise RuntimeError("B15's displacement table is per layer: use AttnContext.for_layer to choose the layer")
        if self.status is not None and self.status_table is None:
            raise RuntimeError("B16's status table is per layer here: use AttnContext.for_layer to choose the layer")
        bsz, length = self.kind.shape
        n_heads = len(self.head_roles) if heads is None else int(heads.shape[0])
        n_rows = rows.shape[1]
        key_board = self.board_idx if cols is None else self.board_idx.gather(1, cols)
        lk = key_board.shape[1]
        bias = None
        if self.pair_table is not None:
            table = self.pair_table if heads is None else self.pair_table[:, heads]
            slots = table.shape[-1]
            q_board = self.board_idx.gather(1, rows)
            picked = table.gather(2, q_board[:, None, :, None].expand(bsz, n_heads, n_rows, slots))
            bias = picked.gather(3, key_board[:, None, None, :].expand(bsz, n_heads, n_rows, lk))
        if self.disp is not None:
            disp = self.disp_rows(rows, cols, heads)
            bias = disp if bias is None else bias + disp
        if self.status is not None and self.status_index.any_frozen:
            status = self.status_rows(rows, cols, heads)
            bias = status if bias is None else bias + status
        if self.typed is not None:
            typed = self.typed
            if span is not None:
                sl = typed.span(*span)
                b, k, t, o = typed.b[sl], typed.k[sl], typed.t[sl], typed.o[sl]
                row = typed.q[sl] - span[0]
            else:
                row = _positions(rows, rows_real, length)[typed.b, typed.q]
                keep = row >= 0
                b, k, t, o, row = typed.b[keep], typed.k[keep], typed.t[keep], typed.o[keep], row[keep]
            if cols is None:
                col = k
            else:
                col = _positions(cols, cols_valid, length)[b, k]
                keep = col >= 0
                b, t, o, row, col = b[keep], t[keep], o[keep], row[keep], col[keep]
            table = self.typed_table if heads is None else self.typed_table[heads]
            values = table[:, t, o].float()  # (Hg, n)
            head = torch.arange(n_heads, device=b.device)[:, None]
            index = ((b[None, :] * n_heads + head) * n_rows + row[None, :]) * lk + col[None, :]
            bias = values.new_zeros(bsz, n_heads, n_rows, lk) if bias is None else bias.contiguous()
            # Accumulated in place into these fresh rows (no gradient needs their previous
            # values); a pair related several ways sums its types' scalars.
            bias.view(-1).index_put_((index.reshape(-1),), values.reshape(-1), accumulate=True)
        return bias

    def disp_entries(self, rows: torch.Tensor, cols: torch.Tensor | None = None) -> torch.Tensor:
        """B15's entry ``(B, n, Lk)`` of every pair of the queries ``rows`` ``(B, n)`` and the
        keys ``cols`` ``(B, K)`` (every key when None): the static class map at (the pair's
        board tile, the query's code, the key's code) - the same at every layer."""
        return lookup(self.disp_index, self.disp.class_map, rows, cols)

    def disp_rows(self, rows: torch.Tensor, cols: torch.Tensor | None = None, heads: torch.Tensor | None = None) -> torch.Tensor:
        """B15's bias ``(B, Hg, n, Lk)`` of the queries ``rows`` over the keys ``cols`` (every
        key when None) on the heads ``heads`` (every head when None), from this layer's table."""
        table = self.disp_table if heads is None else self.disp_table[heads]
        return table[:, self.disp_entries(rows, cols)].transpose(0, 1)

    def status_rows(self, rows: torch.Tensor, cols: torch.Tensor | None = None, heads: torch.Tensor | None = None) -> torch.Tensor:
        """B16's bias ``(B, Hg, n, Lk)`` of the queries ``rows`` ``(B, n)`` over the keys ``cols``
        (every key when None) on the heads ``heads`` (every head when None): the table's entry for
        the key's board's status where the query is live and the key a frozen board's token, zero
        everywhere else (the table's appended zero, or a frozen query row)."""
        idx = self.status_index
        table = self.status_table if heads is None else self.status_table[heads]  # (Hg, statuses + 1)
        key = idx.key if cols is None else idx.key.gather(1, cols)  # (B, Lk)
        values = table[:, key].transpose(0, 1)[:, :, None, :]  # (B, Hg, 1, Lk)
        live = idx.live.gather(1, rows)[:, None, :, None]  # (B, 1, n, 1)
        return torch.where(live, values, values.new_zeros(()))

    def dense_mask(self) -> torch.Tensor | None:
        """Every row's allowed keys ``(B, H, L, L)`` (small inputs; tests)."""
        return self.mask_rows(0, self.kind.shape[1])

    def dense_bias(self) -> torch.Tensor | None:
        """Every row's bias ``(B, H, L, L)`` (small inputs; tests)."""
        return self.bias_rows(0, self.kind.shape[1])


def board_coordinates(batch: Batch, coords: torch.Tensor) -> torch.Tensor:
    """Per-board ``(p, h)`` ``(B, Nb, 2)`` from per-token coordinates ``(B, L, 4)`` (every
    token of a board carries its board's timeline and time)."""
    bsz = coords.shape[0]
    n_max = batch.board_valid.shape[1]
    has = batch.valid & (batch.board >= 0)
    slot = torch.where(has, batch.board, torch.full_like(batch.board, n_max))
    out = coords.new_zeros(bsz, n_max + 1, 2)
    out.scatter_(1, slot[..., None].expand(bsz, slot.shape[1], 2), coords[..., 2:4])
    return out[:, :n_max]


def build_structure(
    batch: Batch,
    cfg: NetConfig,
    coords: torch.Tensor,
    typed_edges: TypedEdgeBias | None = None,
    displacement: DisplacementBias | None = None,
    smolgen: BoardPairSmolgen | None = None,
    live: torch.Tensor | None = None,
    live_boards: torch.Tensor | None = None,
    status: StatusKeyBias | None = None,
) -> AttnStructure | None:
    """The batch's attention structure, or None when every head is dense and nothing adds
    a bias (then attention needs only the padding mask). ``live`` ``(B, L)`` /
    ``live_boards`` ``(B, Nb)``: under the block-causal scope, the tokens and boards whose
    rows the edge-derived masks and biases and B12's smolgen may open or bias (None:
    every row - the dense scope). B15's displacement bias applies to every row (its
    entries come from the batch's raw coordinates, ``displacement.displacement_index``).
    ``status``: B16's table, under the block-causal scope only (it needs ``live``): the live
    rows' biases on frozen boards' tokens, from the batch's own Delta-t and active flags."""
    roles = cfg.head_roles
    structured = any(role != "dense" for role in roles)
    if not (structured or typed_edges is not None or displacement is not None or smolgen is not None or status is not None):
        return None
    if status is not None and live is None:
        raise ValueError("B16's status biases exist under the block-causal scope only: they need its live rows")
    n_max = batch.board_valid.shape[1]
    board_idx = torch.where(batch.valid & (batch.board >= 0), batch.board, torch.full_like(batch.board, n_max))
    xy = batch.coords[..., :2].round().long()
    square_id = torch.where(batch.square_mask, xy[..., 1] * 4096 + xy[..., 0], torch.full_like(xy[..., 0], -1))
    move = rel = typed = None
    if "move_graph" in roles:
        move = mask_pairs(batch, [RELATION_IDS["move"]], cfg.move_graph_direction).live_queries(live)
    if "relations" in roles:
        rel = mask_pairs(batch, [RELATION_IDS[name] for name in cfg.relations_head_types], cfg.relations_direction).live_queries(live)
    table = None
    if typed_edges is not None:
        typed = typed_pairs(batch, cfg.typed_edge_directed).live_queries(live)
        table = typed_edges.table
    ph = board_coordinates(batch, coords).round() if smolgen is not None else None
    return AttnStructure(
        head_roles=roles,
        registers=cfg.masked_head_registers,
        kind=batch.kind,
        board_idx=board_idx,
        square_id=square_id,
        n_boards=n_max,
        same_board_registers=cfg.same_board_registers,
        move=move,
        rel=rel,
        typed=typed,
        typed_table=table,
        smolgen=smolgen,
        smolgen_in=None if smolgen is None else smolgen_inputs(batch, cfg, ph),
        smolgen_rows=live_boards if smolgen is not None else None,
        disp=displacement,
        disp_index=None if displacement is None else displacement_index(batch, displacement.geom),
        status=status,
        status_index=None if status is None else status_index(batch, status, live),
        status_table=None if status is None or status.per_layer else status.layer_table(0),
    )
