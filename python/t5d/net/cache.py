"""The cached evaluation path (stage N3; spec sections 7.1-7.3, 7.9, 8.5, 8.7, 10.2, 10.3;
register C2, C3, C3a rung 1, C15, C17, C19, C20, C21, C24, G1).

A network trained under the block-causal scope (``NetConfig.attn_scope="block_causal"``,
``scope.py``) computes the same function whether it runs one full pass over a position
or evaluates it incrementally, the way a search would: the K/V of every frozen
(superseded) board at every layer is computed once, stored, and reused by every node
that holds the board; a node computes only its live set - the head boards and the global
registers - against the stored K/V (section 7.1). This module is that incremental path,
built for section 15.2's equivalence suite and section 15.3's cost figures (C2 is
deferred; the build rules allow it exactly for those).

**What a node evaluation computes** (``CachedEvaluator.evaluate``): (1) every frozen
board of the position that has no entry yet is computed - once, one board at a time,
under the state-defined mask: its causal past, with the past's K/V taken from the store
(C20: "recompute the board once under the state-defined mask ... and share it down the
subtree", never snapshotted - C21). A search root's own frozen boards are computed the
same way - from both colours' sides (register A7: each perspective its own entry) - so an
entry is one function of its digest wherever it is computed - which
root a store started from never changes a value, not even in the last bit; (2) the live pass: the heads and the global registers,
every layer, against the stored K/V of all frozen boards and their own (C19: the heads
read each other both ways and are recomputed at every node); (3) the heads of the network
over the assembled final trunk output (frozen boards' outputs are stored with their K/V,
for the policy's destinations and the per-board heads).

**Prefix sharing (section 7.2).** Entries are content-addressed: an entry's key is its
epoch's key and a dependency digest - a hash of the board itself (``board_ident``:
coordinates, contents with the has-moved markers, size; under ``frozen_features="current"``
its state-dependent flags; and the acting player, since register A7 encodes every board from
the acting player's side - owners, side to move, mirrored ranks and negated timelines - so a
frozen board's K/V differ by perspective: spec issue 303) and of the digests of the boards
it reads (so two boards that differ only in their branch origin get two entries). A frozen
board has up to two entries per epoch, one per acting colour; the acting colour's own
embedding sits on the live global registers (D8, M20) - with none, on the live boards' tokens
(spec issue 225) - which no frozen board's input carries. Two nodes
whose board has the same digest share the entry; a node computes only boards with novel
digests. The digest of a board covers its whole causal past, so the store is section
7.2's radix structure with the board as the unit: under the lineage reading its entries
form the board-derivation tree of each epoch, and a new node reuses its longest cached
prefix - every board it inherited - and computes its novel suffix. Eviction is out of
scope (the structure can share; nothing here frees an entry but ``drop_epoch``).

**Epochs (C3, C3a; section 7.9).** Every evaluation names its epoch (``Epoch``):

* ``"causal"`` - no refresh: every frozen board is computed under the block-causal
  mask. Under ``supersession_scope="causal_past"`` this epoch is shared by every search
  and carries across game moves (section 8.7: "KV cache, frozen boards: across moves");
  under ``"past_and_epoch_history"`` it is anchored at its search root, whose frozen
  boards are its history.
* ``"root"`` (``CacheConfig.root_refresh``, C3 provisional): once per game move, one
  dense bidirectional pass over the root (the ordinary dense model: every board with its
  full features, everything reading everything); the K/V of the root's frozen boards
  become the epoch's history - "every node in that search tree then attends into history
  representations that already know what is currently relevant".
* ``"node"`` (``CacheConfig.node_refresh``, C3a rung 1, prototype): the same dense pass
  at an internal node (``refresh``) - travel-triggered and visit-gated, decided by the
  search (``tree.py``) - forks a new epoch for that node's subtree, copy-on-write: new
  entries under the new epoch's key, the parent epoch's entries untouched, so siblings
  keep the parent epoch ("fork, never overwrite - except at the root"). C24's complexity
  trigger (``CacheConfig.complexity_refresh``, prototype; ``complexity.py``) forks the same
  way, at more nodes: it changes when a node refreshes, never how.

Within a refreshed epoch, the history boards keep the refreshed K/V; every board frozen
after the anchor is recomputed under the state-defined mask, reading its past's K/V from
the epoch - "recomputing against the root epoch inside a refreshed subtree would silently
undo the refresh". Evaluations are a function of (state, epoch) (section 7.9). A board
belongs to an epoch's history by its intrinsic identity (coordinates and contents), so
under ``frozen_features="current"`` a history board stays one when the present moves, and it
belongs whoever acts. Its refreshed K/V are per perspective (register A7; spec issue 303): a
refresh runs the dense pass over the anchor once from each colour's side - the anchor as it
is, and the same boards with the other colour acting and no moves or edges (``viewed_by``:
the anchor's are its acting player's; a placeholder, spec issue 229) - so a node of either
acting player reads history encoded in its own frame; about twice one refresh's compute and
history memory, a cost of the deferred cached configuration (C2).

**Storage precision (C15, G1).** The cache stores frozen K/V in the network's own format,
``NetConfig.frozen_kv_format`` (``kvformat.py``): ``"match"`` (default) stores them exactly
as the live computation produces them - BF16 under the BF16 trunk, FP32 under the FP32
trunk: G1's "cache and live computation share precision", and lossless, since the kernel
reads those very values. ``"fp8"`` stores ``torch.float8_e4m3fn`` with one FP32 scale per
board, layer, head and K-or-V, for a network whose ordinary forward rounds every read of a
frozen board's K/V the same way - the function it is trained on; serving a ``"match"``
network from an FP8 cache would compute a function no training path computes, so the
evaluator never does. Not built: the varlen/ragged kernels C15 also adopts - PyTorch on CPU
has no ragged attention kernel, so each block here runs the stock kernel over its own key
set.

**References.** ``reference_output`` is the from-scratch computation the cached path must
equal: one full pass of the ordinary model over the position under the block-causal
scope, through any attention implementation (with the network's frozen K/V format applied
by the model itself) - and for a refreshed epoch, the dense pass over the anchor first,
whose history K/V are injected into that full pass (the two-level mask of section 7.9), the
history boards' outputs taken from the anchor pass.

**B15's displacement bias** (the v1 default) is geometry - a function of the two tokens'
coordinates, "computed at score time from coordinates, so nothing cached goes stale" (section
4.1) - so it biases every row, a frozen board's included: ``compute_board`` builds a frozen
board's bias rows over its key set once per layer, from that layer's table, and the live pass
gives each layer its own (``AttnContext.for_layer``).

**B17's key-side Shaw term** (behind its flag; ``shaw.py``) is geometry and the query's own
content - a frozen board's rows get it from the board's own queries, so ``compute_board``
adds it to them, layer by layer, and the live pass to the live rows (``_FastPath``).

**B16's status key biases** (committed with this configuration; ``status.py``) are the
current status of the frozen boards - their Delta-t-to-present bin and active flag - as seen
by the live rows that read them: the live pass adds them from the node's own present, every
evaluation, so a frozen entry never goes stale when the present moves or a timeline is
activated (its K/V carry intrinsic features only, and its digest no status). A frozen board's
own rows never carry them (``AttnStructure.for_frozen_rows``). A refreshed epoch's history
boards are biased like any other at every node below the anchor; at the anchor itself the
evaluation is the refresh pass (section 7.9), so there they are left out - in ``prepare`` and
in ``reference_output`` alike. That is one rule in both, which comparing them cannot test:
``test_cache.py`` checks it against an oracle of its own (B16's table changed, the anchor's
evaluation stays bit for bit and every node below it moves).

**Not supported** (``CachedEvaluator`` refuses them): B12's ``board_bilinear`` form, which
reads every board's per-board register states at every layer - frozen boards' hidden
states are not cached; and B12 with no global registers (its context is then the mean of
every square, frozen ones included). The ``relative_cells`` default reads only the global
registers, which are live.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import struct
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .attention import AttnContext, _block_mask, _block_rows, _FastPath, _scores, _softmax_rows
from .kvformat import FROZEN_KV_FORMATS, dequantise_fp8, fake_quantise, quantise_fp8
from .model import NetOutput, T5DNet
from .rope import rope_angles, rope_cos_sin
from .scope import board_scope
from .state import TOKEN_BOARD_REGISTER, Batch, PositionState, collate

KV_FORMATS = FROZEN_KV_FORMATS  # the storage formats (``kvformat.py``)


# --- configuration, epochs, the store -------------------------------------------------------------


@dataclass(frozen=True)
class CacheConfig:
    """The cached path's switches (the evaluator's; the network's are in ``NetConfig`` -
    the storage format among them: ``NetConfig.frozen_kv_format``)."""

    #: C3 (provisional): one dense refresh pass over the root per game move; its K/V
    #: become the frozen-history cache of the whole tree (two-level mask).
    root_refresh: bool = False
    #: C3a rung 1 (prototype): travel-triggered, visit-gated node refreshes that fork the
    #: cache copy-on-write for the node's subtree (the search decides when - ``tree.py``).
    node_refresh: bool = False
    #: Rung 1's visit gate V (section 7.9's table: "V ~ 50").
    refresh_visit_threshold: int = 50

    # --- C24: the complexity trigger (prototype; default off; ``complexity.py``) ---
    #: C24 (prototype, gated like rung 1 on 6c's existence test): node refreshes also where
    #: the position is complex - add-only: never removes or delays a refresh the same
    #: configuration makes with this off (the search decides - ``tree.py``). Every number
    #: below is a PLACEHOLDER, not a decision: section 7.9 names the signals and no
    #: threshold, weight or combination rule (spec issue).
    complexity_refresh: bool = False
    #: The score at which a node fires: ``sum(weight * signal) >= complexity_threshold``.
    complexity_threshold: float = 1.0
    #: Per board of the node's position: 1/128 - 128 boards alone reach the threshold.
    complexity_board_weight: float = 1.0 / 128
    #: Per active timeline of the node's position: 1/8 - 8 alone reach it.
    complexity_timeline_weight: float = 1.0 / 8
    #: Per travel move since the node's last refresh: 1/2 - 2 alone reach it.
    complexity_travel_weight: float = 1.0 / 2
    #: Per effort bin of the parent's predicted difficulty (log2 visits): 1/12 - a predicted
    #: 4,096 visits alone reach it; the censored "did not converge" bin (15 of 16) is over.
    complexity_effort_weight: float = 1.0 / 12
    #: Which quantile of the parent's effort distribution is its predicted difficulty (the
    #: median; section 8.6's allocator plans to "an upper quantile (tunable, 6v)").
    complexity_effort_quantile: float = 0.5
    #: C24's visit gate: rung 1's V by default (one whole-scope refresh ~ ten node
    #: evaluations, so a subtree must consume well over ten visits to repay one).
    complexity_visit_threshold: int = 50
    #: C24's refreshes per search at most (section 7.9: "epochs must be bounded per
    #: search"); rung 1's are not counted against it and never blocked by it.
    complexity_refresh_budget: int = 8

    def validate(self) -> None:
        if self.refresh_visit_threshold < 1:
            raise ValueError("refresh_visit_threshold must be positive")
        if self.complexity_visit_threshold < 1:
            raise ValueError("complexity_visit_threshold must be positive")
        if self.complexity_refresh_budget < 0:
            raise ValueError("complexity_refresh_budget must not be negative")
        weights = (
            self.complexity_board_weight,
            self.complexity_timeline_weight,
            self.complexity_travel_weight,
            self.complexity_effort_weight,
        )
        if not all(math.isfinite(w) and w >= 0 for w in weights):
            raise ValueError("the complexity weights must be finite and non-negative (more complexity never means fewer refreshes)")
        if not (math.isfinite(self.complexity_threshold) and self.complexity_threshold > 0):
            raise ValueError("complexity_threshold must be finite and positive")
        if not 0 < self.complexity_effort_quantile <= 1:
            raise ValueError("complexity_effort_quantile must be in (0, 1]")


@dataclass(frozen=True, eq=False)
class Epoch:
    """A refresh epoch (section 7.9): which conditioning a node's frozen history carries."""

    key: str  # the cache-key component of every entry of this epoch
    kind: str  # "causal", "root" or "node"
    parent: Epoch | None = None  # the epoch this one forked from (node refreshes)
    anchor: str | None = None  # digest of the anchor state (root or refreshed node)
    #: Intrinsic idents (``board_ident``, no perspective) of the anchor's frozen boards: the
    #: epoch's history.
    #: Empty for the shared causal epoch (C20's first reading needs none).
    history: frozenset[bytes] = frozenset()
    #: True: the history's K/V come from a dense pass over the anchor (a refresh).
    refreshed: bool = False

    def lineage(self) -> list[Epoch]:
        """This epoch and its ancestors, newest first."""
        out, e = [], self
        while e is not None:
            out.append(e)
            e = e.parent
        return out


@dataclass(eq=False)
class BoardKV:
    """One frozen board's entry: its K/V at every layer and its final trunk output."""

    kv: torch.Tensor  # (n_layers, 2, H, T_b, d_head): "match" - the compute dtype; "fp8" - FP8 data
    scale: torch.Tensor | None  # (n_layers, 2, H, 1, 1) FP32 scales ("fp8"), else None
    out: torch.Tensor  # (T_b, d_model) FP32: the board's final-normed trunk output
    deps: tuple[tuple[str, bytes], ...] = ()  # the keys of the entries it read (its prefix)
    source: str = ""  # "history" (a search root's), "supersession" (in a search) or "refresh"

    def layer(self, layer: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V ``(H, T_b, d_head)`` of one layer, in ``dtype``."""
        kv = self.kv[layer].to(dtype) if self.scale is None else dequantise_fp8(self.kv[layer], self.scale[layer], dtype)
        return kv[0], kv[1]

    @property
    def kv_bytes(self) -> int:
        """Bytes of the K/V (and their scales) - section 15.3's "KV memory"."""
        return self.kv.nbytes + (0 if self.scale is None else self.scale.nbytes)

    @property
    def out_bytes(self) -> int:
        return self.out.nbytes


@dataclass
class KVStore:
    """Content-addressed per-board K/V entries, keyed ``(epoch key, dependency digest)``."""

    entries: dict[tuple[str, bytes], BoardKV] = field(default_factory=dict)
    #: Boards computed, by source; and entries found already stored, by evaluations.
    computed: dict[str, int] = field(default_factory=lambda: {"history": 0, "supersession": 0, "refresh": 0})
    hits: int = 0

    def __contains__(self, key) -> bool:
        return key in self.entries

    def __getitem__(self, key) -> BoardKV:
        return self.entries[key]

    def put(self, key, entry: BoardKV) -> None:
        if key in self.entries:
            raise RuntimeError("an entry is never overwritten (copy-on-write: a refresh forks a new epoch)")
        self.entries[key] = entry
        self.computed[entry.source] = self.computed.get(entry.source, 0) + 1

    def epoch_keys(self, epoch_key: str) -> list[tuple[str, bytes]]:
        return [k for k in self.entries if k[0] == epoch_key]

    def drop_epoch(self, epoch_key: str) -> int:
        """Forget an epoch's entries (garbage collection after the next root refresh)."""
        keys = self.epoch_keys(epoch_key)
        for k in keys:
            del self.entries[k]
        return len(keys)

    def kv_bytes(self, epoch_key: str | None = None) -> int:
        return sum(e.kv_bytes for k, e in self.entries.items() if epoch_key is None or k[0] == epoch_key)

    def out_bytes(self, epoch_key: str | None = None) -> int:
        return sum(e.out_bytes for k, e in self.entries.items() if epoch_key is None or k[0] == epoch_key)


# --- identities and digests ------------------------------------------------------------------------


def board_ident(state: PositionState, board: int, frozen_features: str = "intrinsic", acting: bool = False) -> bytes:
    """A frozen board's identity: what its own input is - size, register count, coordinates,
    side to move, contents, and under ``frozen_features="current"`` its Delta-t-to-present
    and active flag (state-dependent inputs of that reading). ``acting`` adds the acting
    player: under register A7 every board's input is encoded from the acting player's side
    (``state.py``), so the cached path keys every frozen board's K/V with it (spec issue 303);
    without it the identity is the board's intrinsic one - an epoch's history membership,
    which does not depend on who acts."""
    h = hashlib.blake2b(digest_size=16)
    head = (int(state.width), int(state.height), int(state.regs_per_board), int(state.timeline[board]), int(state.half_turn[board]))
    h.update(struct.pack("<5q", *head))
    h.update(state.piece[board].astype(np.int8).tobytes())
    h.update(state.colour[board].astype(np.int8).tobytes())
    h.update(state.moved[board].astype(np.int8).tobytes())
    if frozen_features == "current":
        h.update(struct.pack("<2q", int(state.half_turn[board]) - state.h_present, int(bool(state.active[board]))))
    if acting:
        h.update(struct.pack("<q", int(state.acting_player)))
    return h.digest()


def state_digest(state: PositionState) -> str:
    """A digest of a whole position (an epoch's anchor), as hex: every board with its flags,
    and the acting player (D8)."""
    h = hashlib.blake2b(digest_size=16)
    for b in range(state.n_boards):
        h.update(board_ident(state, b))
        h.update(struct.pack("<2q", int(state.is_head[b]), int(state.active[b])))
    h.update(struct.pack("<q", int(state.acting_player)))
    return h.hexdigest()


def topological_order(frozen: np.ndarray, deps: dict[int, list[int]], state: PositionState) -> list[int]:
    """The frozen boards with every board after the boards it reads; ties in ``(p, h)``
    order (canonical, so that every path computes a node's boards in one order)."""
    order: list[int] = []
    pending = {int(b): {j for j in deps[int(b)]} for b in frozen}
    key = lambda b: (int(state.timeline[b]), int(state.half_turn[b]))  # noqa: E731
    ready = sorted((b for b, d in pending.items() if not d), key=key)
    readers: dict[int, list[int]] = {}
    for b, d in pending.items():
        for j in d:
            readers.setdefault(j, []).append(b)
    while ready:
        b = ready.pop(0)
        order.append(b)
        newly = []
        for r in readers.get(b, ()):
            pending[r].discard(b)
            if not pending[r]:
                newly.append(r)
        if newly:
            ready = sorted(ready + newly, key=key)
    if len(order) != len(pending):
        raise ValueError("the frozen boards' dependencies have a cycle")
    return order


@dataclass
class _Plan:
    """What one evaluation needs to know about its position's frozen boards."""

    scope: np.ndarray  # (N, N) bool: board_scope
    frozen: np.ndarray  # frozen board indices
    order: list[int]  # topological
    deps: dict[int, list[int]]  # the frozen boards each frozen board reads (itself excluded)
    ident: dict[int, bytes]
    digest: dict[int, bytes]
    history: np.ndarray | None  # (N,) bool: boards in the epoch's history (None: none)

    def key(self, epoch: Epoch, board: int) -> tuple[str, bytes]:
        return (epoch.key, self.digest[board])


@dataclass(eq=False)
class NodeWork:
    """One node evaluation's inputs: the position, its plan, its attention context (the
    block-causal scope and the structure) and its token embeddings."""

    state: PositionState
    epoch: Epoch
    batch: Batch
    plan: _Plan
    ctx: AttnContext
    emb: torch.Tensor
    coords: torch.Tensor | None = None  # RoPE coordinates, computed when a board needs them


# --- the evaluator ---------------------------------------------------------------------------------


class CachedEvaluator:
    """Evaluates positions incrementally against a per-board K/V store (see the module
    docstring). One model, one store; every evaluation names its epoch."""

    def __init__(self, model: T5DNet, config: CacheConfig | None = None, store: KVStore | None = None) -> None:
        cfg = model.cfg
        if cfg.attn_scope != "block_causal":
            raise ValueError("the cached path evaluates a network of the block-causal scope (NetConfig.attn_scope)")
        if model.smolgen is not None and (cfg.smolgen_form != "relative_cells" or cfg.global_regs == 0):
            raise NotImplementedError(
                "B12's board_bilinear form (and smolgen without global registers) reads frozen boards' hidden states, "
                "which the cache does not keep"
            )
        self.model = model
        self.config = CacheConfig() if config is None else config
        self.config.validate()
        self.store = KVStore() if store is None else store
        self._epochs = itertools.count()

    # -- epochs --

    @property
    def root_dependent(self) -> bool:
        """Whether C20's reading (``supersession_scope``) makes even the no-refresh epoch
        depend on its search root."""
        return self.model.cfg.supersession_scope == "past_and_epoch_history"

    def root_epoch(self, root: PositionState) -> Epoch:
        """The epoch of a new search at ``root``: a root refresh (C3) or the causal epoch,
        whose missing frozen boards are computed here, one board at a time like every other
        (so a value never depends on which root computed it) - from each colour's side
        (register A7; spec issue 303): the root's acting player's, and the other's, whose
        nodes the search reaches after the root's action is submitted."""
        if self.config.root_refresh:
            return self._refresh(root, "root", None)
        anchor = state_digest(root)
        if self.root_dependent:
            history = frozenset(board_ident(root, int(b)) for b in np.flatnonzero(~root.is_head))
            epoch = Epoch(f"causal@{anchor}", "causal", None, anchor, history, False)
        else:
            epoch = Epoch("causal", "causal", None, anchor, frozenset(), False)
        for view in (root, viewed_by(root, 1 - int(root.acting_player))):
            work = self.prepare(view, epoch)
            for b in self.missing(work):
                self.compute_board(work, b, source="history")
        return epoch

    def refresh(self, state: PositionState, parent: Epoch) -> Epoch:
        """A node refresh (C3a rung 1): fork a new epoch for this node's subtree, copy-on-write."""
        return self._refresh(state, "node", parent)

    def _refresh(self, state: PositionState, kind: str, parent: Epoch | None) -> Epoch:
        """A refresh epoch anchored at ``state``: the dense pass over it once per perspective
        (register A7; the module docstring) - the anchor's own acting player first - with its
        frozen boards' K/V stored as the epoch's history under each."""
        model = self.model
        frozen = np.flatnonzero(~state.is_head)
        idents = {int(b): board_ident(state, int(b)) for b in frozen}  # intrinsic: history membership
        anchor = state_digest(state)
        epoch = Epoch(f"{kind}:{next(self._epochs)}:{anchor[:12]}", kind, parent, anchor, frozenset(idents.values()), True)
        tpb = state.layout.tokens_per_board
        acting = int(state.acting_player)
        for view in (state, viewed_by(state, 1 - acting)):
            captured: list[tuple[torch.Tensor, torch.Tensor]] = []
            out = model(collate([view]), scope="dense", kv_hook=_capture_into(captured), return_trunk=True)
            for b in frozen:
                sl = slice(int(b) * tpb, (int(b) + 1) * tpb)
                kv = torch.stack([torch.stack([k[0, :, sl], v[0, :, sl]]) for k, v in captured])
                entry = self._entry(kv, out.trunk[0, sl], (), "refresh")
                self.store.put((epoch.key, _history_digest(board_ident(view, int(b), acting=True))), entry)
        return epoch

    # -- planning --

    def _plan(self, state: PositionState, epoch: Epoch) -> _Plan:
        cfg = self.model.cfg
        frozen = np.flatnonzero(~state.is_head)
        # Register A7: a frozen board's input is the acting player's view of it, so its K/V are
        # keyed by the acting player (spec issue 303).
        ident = {int(b): board_ident(state, int(b), cfg.frozen_features, acting=True) for b in frozen}
        # History membership by the intrinsic identity: under frozen_features="current" a
        # board's input changes with the present, and under A7 with the acting player; its
        # belonging to the anchor does neither. A refreshed history board's K/V are its
        # perspective's (``_refresh``).
        intrinsic = {int(b): board_ident(state, int(b)) for b in frozen}
        viewed = {int(b): board_ident(state, int(b), acting=True) for b in frozen}
        history = None
        if epoch.history:
            history = np.zeros(state.n_boards, dtype=bool)
            for b in frozen:
                history[b] = intrinsic[int(b)] in epoch.history
        scope = board_scope(state.parent, state.creation, state.is_head, cfg.causal_past, cfg.supersession_scope, history, state.source)
        deps: dict[int, list[int]] = {}
        for b in frozen.tolist():
            if epoch.refreshed and history is not None and history[b]:
                deps[b] = []  # the refreshed K/V: fixed, read nothing at this node
            else:
                deps[b] = [int(j) for j in np.flatnonzero(scope[b]) if j != b]
        order = topological_order(frozen, deps, state)
        digest: dict[int, bytes] = {}
        for b in order:
            if epoch.refreshed and history is not None and history[b]:
                digest[b] = _history_digest(viewed[b])
            else:
                h = hashlib.blake2b(ident[b], digest_size=16)
                for d in sorted(digest[j] for j in deps[b]):
                    h.update(d)
                digest[b] = h.digest()
        return _Plan(scope, frozen, order, deps, ident, digest, history)

    def _context(self, batch: Batch, plan: _Plan, at_anchor: bool = False) -> AttnContext:
        """The node's block-causal context; ``at_anchor``: the node is its refreshed epoch's
        anchor, whose evaluation is the refresh pass - no B16 (``status.py``)."""
        history = None
        if plan.history is not None and self.root_dependent:
            history = torch.from_numpy(plan.history)[None]
        return self.model.make_context(batch, scope="block_causal", history=history, status_bias=not at_anchor)

    # -- the frozen boards --

    def _entry(self, kv: torch.Tensor, out: torch.Tensor, deps, source: str) -> BoardKV:
        """An entry from a board's K/V ``(n_layers, 2, H, T_b, dh)`` as the live
        computation produced them (before any quantisation) and its final output."""
        if self.model.cfg.frozen_kv_format == "fp8":
            data, scale = quantise_fp8(kv)
            return BoardKV(data, scale, out.detach().float().clone(), tuple(deps), source)
        return BoardKV(kv.detach().clone(), None, out.detach().float().clone(), tuple(deps), source)

    def compute_board(self, work: NodeWork, b: int, source: str = "supersession") -> None:
        """One frozen board, once, under the state-defined mask (C20): its tokens against
        the stored K/V of the boards it reads and its own, in a frame centred on itself -
        so the computation is the same wherever and whenever it runs. ``source`` labels the
        entry for the store's counts."""
        model, cfg = self.model, self.model.cfg
        fmt = cfg.frozen_kv_format
        state, batch, ctx, emb, plan, epoch = work.state, work.batch, work.ctx, work.emb, work.plan, work.epoch
        if work.coords is None:
            work.coords = model.coordinates(batch)
        coords_all = work.coords
        tpb = state.layout.tokens_per_board
        key_boards = sorted(plan.deps[b], key=lambda j: (int(state.timeline[j]), int(state.half_turn[j])))
        own = torch.arange(b * tpb, (b + 1) * tpb)
        keys = torch.cat([torch.arange(j * tpb, (j + 1) * tpb) for j in key_boards] + [own])
        n_keys, c = keys.shape[0], tpb
        entries = [self.store[plan.key(epoch, j)] for j in key_boards]
        # Rows: the scope and the structure for this board's queries, on its key set - the
        # mask once, the bias per layer (B15's table is per layer). The structure's smolgen
        # and B16's status biases are live rows' only, so a frozen board's rows have neither;
        # B15 is geometry and biases them (``scope.py``).
        st = None if ctx.structure is None else ctx.structure.for_frozen_rows()
        frozen_ctx = replace(ctx, structure=st)
        allowed = _block_mask(frozen_ctx, b * tpb, (b + 1) * tpb)[..., keys]
        caller_bias = None  # a caller's dense bias (the cache's own contexts have none)
        if ctx.bias is not None:
            caller_bias = _block_rows(replace(frozen_ctx, structure=None), b * tpb, (b + 1) * tpb)[1][..., keys]
        # Positions relative to the board itself (integer offsets: exact in any frame).
        coords = coords_all[:, keys].clone()
        coords[..., 2:4] -= coords_all[:, b * tpb, 2:4][:, None, :]
        cos, sin = rope_cos_sin(rope_angles(coords, model.rope_axis, model.rope_freq))
        spatial_free = batch.kind[:, keys] == TOKEN_BOARD_REGISTER  # B8: no file or rank (M20)
        compact = AttnContext(
            cos=cos,
            sin=sin,
            body_len=n_keys,
            key_valid=torch.ones(1, n_keys, dtype=torch.bool),
            compute_dtype=ctx.compute_dtype,
            spatial_free=spatial_free,
            spatial_dims=model.rope_spatial,
        )
        x = emb[:, b * tpb : (b + 1) * tpb]
        layers: list[torch.Tensor] = []
        with model._autocast(x.device):
            # The trunk's own block steps around this attention (``trunk.Block``): the same
            # function under either norm placement (F5).
            for layer, block in enumerate(model.trunk.blocks):
                h = block.attn_input(x)
                q, k, v = block.attn.qkv(h)
                layers.append(torch.stack([k[0], v[0]]))
                k, v = fake_quantise(k, fmt), fake_quantise(v, fmt)
                past = [e.layer(layer, k.dtype) for e in entries]
                kk = torch.cat([p[0][None] for p in past] + [k], dim=2)
                vv = torch.cat([p[1][None] for p in past] + [v], dim=2)
                qq = torch.cat([q.new_zeros(1, q.shape[1], n_keys - c, q.shape[-1]), q], dim=2)
                bias = caller_bias
                if st is not None:
                    st_l = st.for_layer(None, layer) if st.layer_dependent else st
                    s_bias = st_l.bias_rows(b * tpb, (b + 1) * tpb, cols=keys[None])
                    if s_bias is not None:
                        bias = s_bias if bias is None else bias + s_bias
                if ctx.shaw is not None:
                    # B17: the board's own queries against its key set's classes (geometry).
                    shaw = ctx.shaw.for_layer(layer).rows(q, own[None], cols=keys[None])
                    bias = shaw if bias is None else bias + shaw
                fast = _FastPath(qq, kk, vv, compact)
                o = fast.full((n_keys - c, n_keys), None, fast.keys, allowed, bias)
                x = block.after_ffn(block.after_attn(x, block.attn.output(o)))
            out = model.trunk.output(x)
        deps = tuple(plan.key(epoch, j) for j in key_boards)
        self.store.put(plan.key(epoch, b), self._entry(torch.stack(layers), out[0], deps, source))

    # -- a node --

    def prepare(self, state: PositionState, epoch: Epoch) -> NodeWork:
        """A node's inputs, before anything is computed."""
        batch = collate([state])
        plan = self._plan(state, epoch)
        at_anchor = epoch.refreshed and state_digest(state) == epoch.anchor
        ctx = self._context(batch, plan, at_anchor)
        return NodeWork(state, epoch, batch, plan, ctx, self.model.embed_tokens(batch, "block_causal"))

    def missing(self, work: NodeWork) -> list[int]:
        """The node's frozen boards with no entry yet, in the order they must be computed."""
        out = []
        for b in work.plan.order:
            if work.plan.key(work.epoch, b) in self.store:
                self.store.hits += 1
            elif work.epoch.refreshed and work.plan.history is not None and work.plan.history[b]:
                raise RuntimeError("a history board of a refreshed epoch has no entry: was the state evaluated in another epoch?")
            else:
                out.append(b)
        return out

    def evaluate(self, state: PositionState, epoch: Epoch) -> NetOutput:
        """The network's output for ``state`` in ``epoch``, the trunk output included: every
        missing frozen board computed (supersession), then the live pass."""
        work = self.prepare(state, epoch)
        for b in self.missing(work):
            self.compute_board(work, b)
        return self.model.heads(self.live_pass(work), work.batch, return_trunk=True)

    def live_pass(self, work: NodeWork, query_boards=None) -> torch.Tensor:
        """The live set - every head (``query_boards``: a subset, for section 15.3's
        changed-boards-only cost probe; its output is then meaningless) and the global
        registers - against every frozen board's stored K/V; returns the final trunk output
        ``(1, L, d)`` with the frozen boards' stored outputs in their rows."""
        model, cfg = self.model, self.model.cfg
        state, ctx, emb, plan, epoch = work.state, work.ctx, work.emb, work.plan, work.epoch
        lay = state.layout
        tpb, length, lb = lay.tokens_per_board, lay.n_tokens, lay.n_body
        heads_all = [int(b) for b in np.flatnonzero(state.is_head)]
        # The cost probe's other heads keep zero K/V: same key count, meaningless values.
        query = heads_all if query_boards is None else sorted(int(b) for b in query_boards)
        frozen = [int(b) for b in plan.frozen]
        entries = [self.store[plan.key(epoch, b)] for b in frozen]
        frozen_tok = torch.cat([torch.arange(b * tpb, (b + 1) * tpb) for b in frozen]) if frozen else torch.zeros(0, dtype=torch.long)
        live_tok = torch.cat([torch.arange(b * tpb, (b + 1) * tpb) for b in query] + [torch.arange(lb, length)])
        blocks = _runs(query, cfg.attn_block_boards, tpb)
        global_free = lb < length  # the position-free global registers' rows (B8), apart
        kernel_rows = torch.cat([torch.arange(s, e) for s, e in blocks])[None] if blocks else None
        x = emb[:, live_tok]
        d_head = cfg.d_head
        with model._autocast(x.device):
            for layer, block in enumerate(model.trunk.blocks):  # the trunk's block steps (``trunk.Block``)
                h = block.attn_input(x)
                q, k, v = block.attn.qkv(h)
                n_heads = q.shape[1]
                k_full = k.new_zeros(1, n_heads, length, d_head)
                v_full = v.new_zeros(1, n_heads, length, d_head)
                if frozen:
                    past = [e.layer(layer, k.dtype) for e in entries]
                    k_full[0, :, frozen_tok] = torch.cat([p[0] for p in past], dim=1)
                    v_full[0, :, frozen_tok] = torch.cat([p[1] for p in past], dim=1)
                k_full[:, :, live_tok] = k
                v_full[:, :, live_tok] = v
                q_full = q.new_zeros(1, n_heads, length, d_head)
                q_full[:, :, live_tok] = q
                # A live row reads every valid key: the scope adds nothing to the live set's rows.
                ctx_l = replace(ctx, scope=None)
                if ctx.layer_dependent:
                    h_full = None
                    if ctx.structure is not None and ctx.structure.smolgen is not None:  # B12 reads the layer's input (B15 does not)
                        h_full = h.new_zeros(1, length, h.shape[-1])
                        h_full[:, live_tok] = h
                    ctx_l = ctx_l.for_layer(h_full, layer)
                fast = _FastPath(q_full, k_full, v_full, ctx_l)
                outs = [_live_rows(fast, ctx_l, blocks, kernel_rows)] if blocks else []
                if global_free:
                    allowed_g, bias_g = _block_rows(ctx_l, lb, length)
                    s_g = _scores(q_full[:, :, lb:], k_full, fast.scale, fast.cd)
                    if bias_g is not None:
                        s_g = s_g + bias_g.float()
                    s_g = s_g.masked_fill(~allowed_g, float("-inf"))
                    outs.append(torch.matmul(_softmax_rows(s_g).to(fast.cd), v_full.to(fast.cd)))
                o = torch.cat([t.to(fast.cd) for t in outs], dim=2)
                x = block.after_ffn(block.after_attn(x, block.attn.output(o)))
            out_live = model.trunk.output(x)
        h_all = out_live.new_zeros(1, length, out_live.shape[-1])
        h_all[:, live_tok] = out_live
        for b, e in zip(frozen, entries, strict=True):
            h_all[0, b * tpb : (b + 1) * tpb] = e.out
        return h_all


def _live_rows(fast: _FastPath, ctx: AttnContext, blocks: list[tuple[int, int]], rows: torch.Tensor) -> torch.Tensor:
    """Every head's output ``(1, H, n, dv)`` on the live set's kernel rows ``rows`` ``(1, n)``
    (the concatenation of ``blocks``). With no bias anywhere the heads are grouped by mask
    role, as in the block-sparse path: the unmasked heads in one call over every live row with
    no per-row mask (the CPU fused kernel runs about 1.6x slower with one, and better on many
    rows than on one board's 66), each masked role block by block with its own rows. With a
    bias, every head runs per block with its rows and biases."""
    st = ctx.structure
    if (st is not None and st.has_bias) or ctx.shaw is not None:
        return torch.cat([fast.all_heads_block(s, e) for s, e in blocks], dim=2)
    groups = [("dense", torch.arange(fast.q.shape[1]))] if st is None else st.head_groups()
    parts, order = [], []
    for role, heads in groups:
        if role == "dense":
            parts.append(fast.full(rows, heads, fast.keys.heads(heads), ctx.key_valid[:, None, None, :], None))
        else:
            parts.append(torch.cat([fast.group_block(role, heads, s, e, sparse=False) for s, e in blocks], dim=2))
        order.append(heads)
    return torch.cat(parts, dim=1)[:, torch.argsort(torch.cat(order))]


def viewed_by(state: PositionState, acting: int) -> PositionState:
    """``state``'s boards with ``acting`` (0 white, 1 black) the acting player: ``state``
    itself when that is its own; otherwise the same boards read from the other colour's side
    (register A7), with no submit action offered (the submit flags reach no K/V), no moves and
    no edges. The view a refresh's dense pass over its anchor takes for each acting colour
    (spec issue 303).

    The other colour's view drops ``moves``, ``opp_moves`` and every edge - the move graph
    and the typed relations - because all of them are the anchor's acting player's: the
    engine builds the move graph and the typed relations "only for the player to move's
    pieces on the boards its moves start from" (``engine/convert.py``), and a strict
    conversion refuses a state whose moves are not the acting player's. Kept, they would
    make every move edge start on "their" pieces, and under the move-graph heads' mask (B9,
    ``move_graph_direction``) frozen squares would attend along them in the dense pass -
    an input no conversion produces, encoded into the history K/V of every node the other
    colour acts at. The other colour's own move graph on these boards would be the sound
    input, but no expansion supplies it: the anchor's boards with the other colour acting
    are no game state (the action is the anchor's acting player's, T5), and the network
    computes no rule. A position with no moves and no edges is one conversion does produce
    (a player with no move). COMMITTED-CLASS SILENCE (C2, C3: what the dense pass over a
    refresh's anchor reads from the other colour's side; spec issue 229) - a placeholder,
    fixed in code: no second arm can be built until something supplies the other colour's
    move graph."""
    if int(acting) == int(state.acting_player):
        return state
    none = np.zeros((0, 3), dtype=np.int64)
    return replace(
        state,
        acting_player=int(acting),
        submit_possible=False,
        submit_legal=False,
        moves=none,
        opp_moves=none.copy(),
        edges=none.copy(),
    )


def _capture_into(captured: list) -> object:
    """A ``kv_hook`` that records every layer's K and V into ``captured``, unchanged."""

    def capture(layer, k, v):
        captured.append((k, v))
        return k, v

    return capture


def _history_digest(ident: bytes) -> bytes:
    """The digest of a refreshed history board: its K/V are the epoch's, whatever it reads."""
    return hashlib.blake2b(b"H" + ident, digest_size=16).digest()


def _runs(boards: list[int], max_boards: int, tpb: int) -> list[tuple[int, int]]:
    """Token ranges of runs of consecutive boards, at most ``max_boards`` boards each."""
    runs: list[tuple[int, int]] = []
    for b in boards:
        if runs and runs[-1][1] == b * tpb and (runs[-1][1] - runs[-1][0]) < max_boards * tpb:
            runs[-1] = (runs[-1][0], (b + 1) * tpb)
        else:
            runs.append((b * tpb, (b + 1) * tpb))
    return runs


# --- the reference ---------------------------------------------------------------------------------


def reference_output(
    model: T5DNet,
    state: PositionState,
    *,
    anchor: PositionState | None = None,
    history_from: PositionState | None = None,
) -> NetOutput:
    """The from-scratch evaluation the cached path must equal: one full pass of the model over
    ``state`` under the block-causal scope (any ``attn_impl``).

    ``anchor``: a refreshed epoch's anchor (section 7.9's two-level mask) - first the dense
    pass over the anchor, from ``state``'s acting player's side (register A7: the anchor's
    boards with that player acting, ``viewed_by``; spec issue 303); its frozen boards (which ``state`` must hold, frozen) keep the K/V
    of that pass, injected into the full pass at every layer, and its outputs. Every other
    board is computed by the full pass. B16 biases every frozen key in the full pass, the
    injected ones included, unless ``state`` is the anchor itself (``status.py``).
    ``history_from``: under ``supersession_scope=
    "past_and_epoch_history"`` without a refresh, the search root whose frozen boards are the
    epoch's history. The network's frozen K/V format applies as in any forward: the model
    rounds every frozen board's K/V - injected ones included - after the injection."""
    cfg = model.cfg
    batch = collate([state])
    tpb = state.layout.tokens_per_board
    ident = {b: board_ident(state, b) for b in range(state.n_boards)}  # intrinsic, as the cache's history
    by_ident = {v: b for b, v in ident.items()}
    injected: dict[int, int] = {}  # state board -> anchor board
    anchor_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    anchor_out = None
    history_state = anchor if anchor is not None else history_from
    history = None
    if history_state is not None:
        history = np.zeros(state.n_boards, dtype=bool)
        for a in np.flatnonzero(~history_state.is_head):
            b = by_ident.get(board_ident(history_state, int(a)))
            if b is None or state.is_head[b]:
                raise ValueError("the position does not hold the anchor's frozen boards, frozen")
            history[b] = True
            if anchor is not None:
                injected[b] = int(a)
    if anchor is not None:

        def capture(layer, k, v):
            anchor_kv.append((k, v))
            return k, v

        view = viewed_by(anchor, int(state.acting_player))
        anchor_out = model(collate([view]), scope="dense", kv_hook=capture, return_trunk=True).trunk

    def hook(layer, k, v):
        if not injected:
            return k, v
        k, v = k.clone(), v.clone()
        for b, a in injected.items():
            sl, src = slice(b * tpb, (b + 1) * tpb), slice(a * tpb, (a + 1) * tpb)
            k[:, :, sl], v[:, :, sl] = anchor_kv[layer][0][:, :, src], anchor_kv[layer][1][:, :, src]
        return k, v

    hist_t = None
    if history is not None and cfg.supersession_scope == "past_and_epoch_history":
        hist_t = torch.from_numpy(history)[None]
    # B16 biases every frozen key, history boards' included - except at the anchor itself,
    # whose evaluation is the refresh pass (section 7.9; ``status.py``).
    at_anchor = anchor is not None and state_digest(state) == state_digest(anchor)
    out = model(batch, scope="block_causal", history=hist_t, kv_hook=hook, return_trunk=True, status_bias=not at_anchor)
    if not injected:
        return out
    h = out.trunk.clone()
    for b, a in injected.items():
        h[0, b * tpb : (b + 1) * tpb] = anchor_out[0, a * tpb : (a + 1) * tpb]
    return model.heads(h, batch, return_trunk=True)


__all__ = [
    "KV_FORMATS",
    "BoardKV",
    "CacheConfig",
    "CachedEvaluator",
    "Epoch",
    "KVStore",
    "board_ident",
    "dequantise_fp8",
    "fake_quantise",
    "quantise_fp8",
    "reference_output",
    "state_digest",
    "viewed_by",
]
