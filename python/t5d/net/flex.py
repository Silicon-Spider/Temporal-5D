"""B14's score function: attention through FlexAttention, every bias and mask computed inside it.

Register B14 (committed engineering): block-sparse mask kernels via board-major ordering and
FlexAttention's ``score_mod``. Section 10.2: "every attention bias - typed edges, the
displacement bias, the masks - is a FlexAttention score function computed inside the
kernel"; section 15.2's B15 row: "Displacement bias through the score function = the same
bias materialised, on small positions". This module is that score function, built for
torch's ``flex_attention`` (``NetConfig.attn_impl="flex"``):

* the **score function** (``score_mod``) adds, per (position, head, query, key): B15's
  displacement bias - the pair's entry computed by integer arithmetic from the two tokens'
  kinds and raw coordinates (``displacement.pair_entry``, "at score time from coordinates"),
  then the layer's compact table; B16's status key bias under the block-causal scope - on a
  live query row, the head's entry for the key's frozen board's status, read from a
  per-token index (``status.py``); B11's typed biases - the pair's typed relations, counted
  per type and orientation, times the head's scalars; B12's board-pair bias - the layer's
  table at the two tokens' boards; and a caller's dense bias;
* the **mask function** (``mask_mod``) allows a key by padding, the caller's mask, the
  block-causal scope (a slot table) and the head's role (section 6's structure): dense;
  move-graph and relations heads by the pair's edge membership (the generator's sparse
  edges, as one bit per pair); same board and same square from the tokens' boards and
  squares; the register rows by ``masked_head_registers`` and ``same_board_registers``;
* the **block mask** (``board_block_mask``) is built from the board-major order: a query
  block of ``block_size`` consecutive tokens reaches a key block when some board (or the
  global-register or padding slot) it holds may read one the key block holds - read off
  board-level tables (the scope's slots, each role's board-to-board reach) without
  evaluating the mask at every pair; lossless (a superset of every allowed pair's block),
  with no block marked full, so the mask function runs on every computed block.

The positional pair rule (``attention.py``: an axis is rotated iff both tokens have it) is
kept by feature widening, as the fast path keeps it: rows that have every axis score
``[q_rot | q_fr | q] . [k_rot' | k_fr' | k_g]`` - rotated against tokens with every axis,
time-only rotated against keys without file and rank, unrotated against position-free global
keys - and two more calls cover the rows without file and rank and the position-free global
rows. Three calls at most, each over every row; each row keeps its own call's output.

**Eager, on CPU.** PyTorch 2.13 runs ``flex_attention`` uncompiled on CPU by materialising
every score and evaluating the score and mask functions under ``vmap``, and has no CPU
backward: this path is for inference and for section 15.2's equivalence on small positions,
not for speed, and under autograd on CPU it refuses. The eager path ignores the block mask's
sparsity (it masks with the mask function), so the tests check the block mask's
losslessness directly. A compiled kernel - on a GPU - and its speed are G9's and M19's: the
per-pair inputs here (edge bits, typed counts) are materialised for the eager reference, and
a kernel would read them in a compact form. Under v1's register rule
(``masked_head_registers="read"``) every board-major query block holds register rows that
read every key, so the masked heads' block masks have no empty block (spec issue 209); the
CPU block-sparse path computes those rows apart.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .attention import AttnContext, _as4d, _spatial_free, _time_rotated, _widened_k, _widened_q
from .displacement import pair_entry
from .rope import apply_rope
from .state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_PAD, TOKEN_SQUARE
from .structure import MASK_ROLES

#: Role codes of the mask function: 0 dense, then ``structure.MASK_ROLES`` in order.
ROLE_CODES: dict[str, int] = {"dense": 0, **{role: i + 1 for i, role in enumerate(MASK_ROLES)}}
#: FlexAttention's default block size (compiled kernels tile by it).
BLOCK_SIZE = 128


def _pair_bits(bsz: int, length: int, b: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(bsz, length, length, dtype=torch.bool, device=q.device)
    out[b, q, k] = True
    return out


@dataclass
class ScoreInputs:
    """What the score and mask functions read, per batch and layer: per-token integers,
    small tables, and - for the eager reference - one bit (or count) per pair for the
    generator's sparse edges."""

    key_valid: torch.Tensor  # (B, L) bool
    mask: torch.Tensor | None = None  # the caller's, expanded to (B, H, L, L)
    bias: torch.Tensor | None = None  # the caller's, expanded to (B, H, L, L)
    scope_slot: torch.Tensor | None = None  # (B, L) long
    scope_allowed: torch.Tensor | None = None  # (B, S, S) bool
    # section 6's masks
    roles: torch.Tensor | None = None  # (H,) long: ROLE_CODES
    kind: torch.Tensor | None = None  # (B, L) long
    board: torch.Tensor | None = None  # (B, L) long: board index, Nb where none
    square: torch.Tensor | None = None  # (B, L) long: a (file, rank) key for squares, -1 otherwise
    move: torch.Tensor | None = None  # (B, L, L) bool: the move-graph role's edge pairs
    rel: torch.Tensor | None = None  # (B, L, L) bool: the relations role's edge pairs
    registers_read: bool = True  # masked_head_registers == "read"
    own_board: bool = True  # same_board_registers == "own_board"
    # B11
    typed: torch.Tensor | None = None  # (B, L, L, types x orientations) float: relation counts
    typed_table: torch.Tensor | None = None  # (H, types x orientations)
    # B12
    pair_table: torch.Tensor | None = None  # (B, H, Nb + 1, Nb + 1)
    # B15
    disp_kind: torch.Tensor | None = None  # (B, L) long, padding marked
    disp_coords: torch.Tensor | None = None  # (B, L, 4) long, raw
    disp_table: torch.Tensor | None = None  # (H, entries + 1): the layer's
    disp_geom: object = None
    # B16
    status_key: torch.Tensor | None = None  # (B, L) long: a frozen board's token's status entry (statuses: none)
    status_live: torch.Tensor | None = None  # (B, L) bool: the live query rows
    status_table: torch.Tensor | None = None  # (H, statuses + 1): the layer's

    @classmethod
    def from_context(cls, ctx: AttnContext, n_heads: int) -> ScoreInputs:
        bsz, length = ctx.key_valid.shape
        out = cls(key_valid=ctx.key_valid)
        shape = (bsz, n_heads, length, length)
        if ctx.mask is not None:
            out.mask = _as4d(ctx.mask).expand(shape)
        if ctx.bias is not None:
            out.bias = _as4d(ctx.bias).float().expand(shape)
        if ctx.scope is not None:
            out.scope_slot, out.scope_allowed = ctx.scope.slot, ctx.scope.allowed
        st = ctx.structure
        if st is None:
            return out
        if st.structured:
            out.roles = torch.tensor([ROLE_CODES[r] for r in st.head_roles], device=st.kind.device)
            out.kind, out.board, out.square = st.kind, st.board_idx, st.square_id
            out.registers_read = st.registers == "read"
            out.own_board = st.same_board_registers == "own_board"
            if st.move is not None:
                out.move = _pair_bits(bsz, length, st.move.b, st.move.q, st.move.k)
            if st.rel is not None:
                out.rel = _pair_bits(bsz, length, st.rel.b, st.rel.q, st.rel.k)
        if st.typed is not None:
            n_types, n_orient = st.typed_table.shape[1:]
            counts = torch.zeros(bsz, length, length, n_types * n_orient, dtype=torch.float32, device=st.kind.device)
            t = st.typed
            counts.index_put_((t.b, t.q, t.k, t.t * n_orient + t.o), torch.ones_like(t.b, dtype=torch.float32), accumulate=True)
            out.typed = counts
            out.typed_table = st.typed_table.float().reshape(st.typed_table.shape[0], -1)
            out.board = st.board_idx
        if st.smolgen is not None:
            if st.pair_table is None:
                raise RuntimeError("this structure's board-pair smolgen needs the layer input: use AttnContext.for_layer")
            out.pair_table = st.pair_table
            out.board = st.board_idx
        if st.disp is not None:
            if st.disp_table is None:
                raise RuntimeError("B15's displacement table is per layer: use AttnContext.for_layer to choose the layer")
            out.disp_kind, out.disp_coords = st.disp_index.kind, st.disp_index.coords
            out.disp_table, out.disp_geom = st.disp_table, st.disp.geom
        if st.status is not None and st.status_index.any_frozen:
            if st.status_table is None:
                raise RuntimeError("B16's status table is per layer here: use AttnContext.for_layer to choose the layer")
            out.status_key, out.status_live, out.status_table = st.status_index.key, st.status_index.live, st.status_table
        return out


def make_score_mod(inp: ScoreInputs):
    """The score function: ``score + bias(b, h, q, k)``, every bias computed per pair here."""

    def score_mod(score, b, h, q, k):
        s = score
        if inp.disp_table is not None:
            cq, ck = inp.disp_coords[b, q], inp.disp_coords[b, k]
            e = pair_entry(
                inp.disp_kind[b, q], inp.disp_kind[b, k], ck[0] - cq[0], ck[1] - cq[1], ck[2] - cq[2], ck[3] - cq[3], inp.disp_geom
            )
            s = s + inp.disp_table[h, e]
        if inp.status_table is not None:
            # B16: a live query's key on a frozen board - the head's entry for that board's
            # status (its last, zero entry for any other key); nothing on a frozen query row.
            status = inp.status_table[h, inp.status_key[b, k]]
            s = s + torch.where(inp.status_live[b, q], status, torch.zeros_like(status))
        if inp.typed is not None:
            s = s + (inp.typed[b, q, k] * inp.typed_table[h]).sum()
        if inp.pair_table is not None:
            s = s + inp.pair_table[b, h, inp.board[b, q], inp.board[b, k]]
        if inp.bias is not None:
            s = s + inp.bias[b, h, q, k]
        return s

    return score_mod


def make_mask_mod(inp: ScoreInputs):
    """The mask function: padding, the caller's mask, the scope and the head's role (the key
    sets of ``structure.py``, pair by pair)."""

    def mask_mod(b, h, q, k):
        ok = inp.key_valid[b, k]
        if inp.mask is not None:
            ok = ok & inp.mask[b, h, q, k]
        if inp.scope_slot is not None:
            ok = ok & inp.scope_allowed[b, inp.scope_slot[b, q], inp.scope_slot[b, k]]
        if inp.roles is not None:
            ok = ok & _role_allows(inp, b, h, q, k)
        return ok

    return mask_mod


def _role_allows(inp: ScoreInputs, b, h, q, k):
    role = inp.roles[h]
    kind_q, kind_k = inp.kind[b, q], inp.kind[b, k]
    itself = q == k
    false = itself & ~itself
    move = inp.move[b, q, k] if inp.move is not None else false
    rel = inp.rel[b, q, k] if inp.rel is not None else false
    on_board_k = (kind_k == TOKEN_SQUARE) | (kind_k == TOKEN_BOARD_REGISTER)
    same_board = (inp.board[b, q] == inp.board[b, k]) & on_board_k
    sq_k = inp.square[b, k]
    same_square = (inp.square[b, q] == sq_k) & (sq_k >= 0)
    relation = torch.where(
        role == ROLE_CODES["move_graph"],
        move,
        torch.where(role == ROLE_CODES["relations"], rel, torch.where(role == ROLE_CODES["same_board"], same_board, same_square)),
    )
    relation = relation | itself
    # The rows the relation does not cover: padding reads every key; an uncovered register
    # every key under "read", itself under "self" (``AttnStructure.role_rows``).
    is_global = kind_q == TOKEN_GLOBAL_REGISTER
    board_register = kind_q == TOKEN_BOARD_REGISTER
    own_board_row = (role == ROLE_CODES["same_board"]) & inp.own_board
    uncovered = is_global | (board_register & ~own_board_row)
    register_rule = (itself | True) if inp.registers_read else itself
    allowed = torch.where(uncovered, register_rule, relation)
    allowed = torch.where(kind_q == TOKEN_PAD, itself | True, allowed)
    return torch.where(role == ROLE_CODES["dense"], itself | True, allowed)


def _over_pairs(fn, prefix: int):
    """``fn`` vmapped over (b, h, q, k) - with ``prefix`` leading arguments (the score) batched
    along with all four - as FlexAttention's eager path evaluates its functions."""
    for dim in (4, 3, 2, 1):  # k innermost, b outermost
        in_dims = tuple([0] * prefix + [0 if d == dim else None for d in (1, 2, 3, 4)])
        fn = torch.func.vmap(fn, in_dims=in_dims)
    return fn


def materialise_score(score_mod, bsz: int, n_heads: int, n_queries: int, n_keys: int, device=None) -> torch.Tensor:
    """The score function's additive term at every (position, head, query, key): ``score_mod``
    evaluated on zero scores - ``(B, H, Lq, Lk)``, FP32 (tests: section 15.2's B15 row)."""
    idx = [torch.arange(n, device=device) for n in (bsz, n_heads, n_queries, n_keys)]
    return _over_pairs(score_mod, 1)(torch.zeros(bsz, n_heads, n_queries, n_keys, device=device), *idx)


def materialise_mask(mask_mod, bsz: int, n_heads: int, n_queries: int, n_keys: int, device=None) -> torch.Tensor:
    """The mask function at every (position, head, query, key): ``(B, H, Lq, Lk)`` bool."""
    idx = [torch.arange(n, device=device) for n in (bsz, n_heads, n_queries, n_keys)]
    return _over_pairs(mask_mod, 0)(*idx)


# --- the block mask, from the board-major order -------------------------------------------------


def _slot_reach(ctx: AttnContext, n_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(slot (B, L), reach (B, H, S, S))``: each token's slot - its board, ``Nb`` for a
    global register, ``Nb + 1`` for padding - and which slots a slot's queries may read on
    each head: a superset of every allowed pair's, from board-level tables only."""
    st = ctx.structure
    valid = ctx.key_valid
    bsz, length = valid.shape
    device = valid.device
    if st is not None:
        n_boards = st.n_boards
        board = st.board_idx
        kind = st.kind
    elif ctx.scope is not None:
        n_boards = ctx.scope.allowed.shape[-1] - 2
        board = torch.where(ctx.scope.slot < n_boards, ctx.scope.slot, torch.full_like(ctx.scope.slot, n_boards))
        kind = None
    else:
        n_boards, board, kind = 0, torch.zeros_like(valid, dtype=torch.long), None
    pad_slot = n_boards + 1
    is_global = valid & (board == n_boards) if kind is None else valid & (kind == TOKEN_GLOBAL_REGISTER)
    slot = torch.where(valid & ~is_global & (board < n_boards), board, torch.where(is_global, n_boards, pad_slot))
    n_slots = n_boards + 2
    # Which slots hold a valid key (padding never does).
    holds = torch.zeros(bsz, n_slots, dtype=torch.bool, device=device)
    holds.scatter_(1, torch.where(valid, slot, torch.full_like(slot, pad_slot)), valid)
    holds[:, pad_slot] = False
    reach = holds[:, None, None, :].expand(bsz, n_heads, n_slots, n_slots).clone()
    if ctx.scope is not None:
        sc = ctx.scope.allowed  # (B, S, S), the same slots
        reach &= sc[:, None]
        reach[:, :, pad_slot] = holds[:, None, :]  # a padding query reads every valid key
    if st is not None and st.structured:
        eye = torch.eye(n_slots, dtype=torch.bool, device=device)
        for h, role in enumerate(st.head_roles):
            if role == "dense":
                continue
            role_reach = eye[None].expand(bsz, n_slots, n_slots).clone()  # a token reads itself
            if role in ("move_graph", "relations"):
                pairs = st.move if role == "move_graph" else st.rel
                role_reach[pairs.b, slot[pairs.b, pairs.q], slot[pairs.b, pairs.k]] = True
            elif role == "same_board":
                pass  # its own board's tokens: its own slot
            else:  # same_square: a square on any board
                role_reach[:, :n_boards, :n_boards] = True
            uncovered = st.uncovered_registers(role) & valid
            if st.registers == "read":
                rows = torch.zeros(bsz, n_slots, dtype=torch.bool, device=device)
                rows.scatter_(1, torch.where(uncovered, slot, torch.full_like(slot, pad_slot)), uncovered)
                role_reach |= rows[:, :, None]
            role_reach[:, pad_slot] = True
            reach[:, h] &= role_reach
    return slot, reach


def board_block_mask(ctx: AttnContext, n_heads: int, block_size: int = BLOCK_SIZE, mask_mod=None) -> BlockMask:
    """A lossless block mask over ``block_size`` consecutive tokens in board-major order (the
    module docstring): a (query block, key block) is computed when some slot of the query
    block may read some slot of the key block. No block is marked full."""
    bsz, length = ctx.key_valid.shape
    slot, reach = _slot_reach(ctx, n_heads)
    n_slots = reach.shape[-1]
    n_blocks = -(-length // block_size)
    block = torch.arange(length, device=slot.device) // block_size
    member = torch.zeros(bsz, n_blocks, n_slots, dtype=torch.float32, device=slot.device)
    member[torch.arange(bsz)[:, None], block[None, :], slot] = 1.0
    blocks = torch.einsum("bqs,bhst,bkt->bhqk", member, reach.float(), member) > 0  # (B, H, QB, KB)
    kv_num = blocks.sum(-1).to(torch.int32)
    kv_idx = torch.argsort((~blocks).to(torch.uint8), dim=-1, stable=True).to(torch.int32)
    return BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=block_size, mask_mod=mask_mod, seq_lengths=(length, length))


# --- the attention ---------------------------------------------------------------------------------


def attention_flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Attention through ``flex_attention`` with the context's every bias and mask computed
    inside its score and mask functions; ``q``, ``k`` unrotated ``(B, H, L, dh)``, ``v``
    ``(B, H, L, dv)``. Forward only on CPU (the module docstring)."""
    if q.device.type == "cpu" and torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError(
            "B14's score function runs forward-only on CPU (PyTorch's FlexAttention has no CPU backward): "
            "train through attn_impl 'block_sparse' or 'chunked'"
        )
    bsz, n_heads, length, dh = q.shape
    cd = ctx.compute_dtype
    inp = ScoreInputs.from_context(ctx, n_heads)
    score_mod, mask_mod = make_score_mod(inp), make_mask_mod(inp)
    block_mask = board_block_mask(ctx, n_heads, block_size, mask_mod)
    qr, kr = apply_rope(q, ctx.cos, ctx.sin), apply_rope(k, ctx.cos, ctx.sin)
    sf = _spatial_free(ctx)
    lb = ctx.body_len
    global_free = ctx.global_free and lb < length

    def with_global_keys(qf: torch.Tensor, kf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the unrotated dims that score a position-free global key."""
        if not global_free:
            return qf, kf
        g = (torch.arange(length, device=q.device) >= lb)[None, None, :, None]
        return torch.cat([qf, q.float()], dim=-1), torch.cat([torch.where(g, 0.0, kf), torch.where(g, k.float(), 0.0)], dim=-1)

    def run(qf: torch.Tensor, kf: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings(), torch.autocast(device_type=q.device.type, enabled=False):
            warnings.filterwarnings("ignore", message="flex_attention called without torch.compile")
            return flex_attention(qf.to(cd), kf.to(cd), v.to(cd), score_mod=score_mod, block_mask=block_mask, scale=dh**-0.5)

    # Rows with every axis: rotated keys, time-only rotated keys without file and rank, raw
    # position-free global keys.
    q_feat = qr if sf is None else _widened_q(ctx, q, qr)
    k_feat = kr if sf is None else _widened_k(ctx, sf, k, kr)
    out = run(*with_global_keys(q_feat, k_feat))
    if sf is not None:
        # Rows without file and rank: time-only rotation on both sides of every pair.
        qt, kt = _time_rotated(ctx, q, k, qr, kr)
        out = torch.where(sf[:, None, :, None], run(*with_global_keys(qt, kt)), out)
    if global_free:
        # Position-free global rows: unrotated against every key.
        out = torch.cat([out[:, :, :lb], run(q.float(), k.float())[:, :, lb:]], dim=2)
    return out
