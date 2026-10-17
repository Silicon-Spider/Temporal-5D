"""B15, the 4D displacement bias (spec section 4.1, register B15), and section 15.2's B15 row.

Section 15.2: "Displacement bias through the score function = the same bias materialised, on
small positions | The score-function implementation is exact | B15". Its protocol: random
weights at realistic initialisation scale - never zeros, which make distinct paths agree
trivially - several seeds, tolerances matched to the arithmetic, adversarial constructions
(duplicated boards, deep travel, many timelines, every variant geometry).

* **The classes against their definition.** ``Reference`` writes section 4.1's classes
  out pair by pair in plain Python - the exact box, the far bins, the line test in whole
  turns, the register and global entries, padding - following the layout documented in
  ``displacement.py``, without its code. Every displacement of a grid that reaches every far
  bin and clamp, every token kind, and every pair of every constructed position, under the
  placeholder default and two other placeholder settings, gets the same entry from the
  module's arithmetic (the score function's) and from its lookup (the CPU paths'): integers,
  compared exactly.
* **Section 15.2's row.** The bias materialised by hand from ``Reference`` and the
  layer's table is the bias the score function computes inside B14's FlexAttention path
  (``flex.py``), bit for bit, and the attention through the score function equals the
  attention with that bias materialised (FP32 roundoff: measured below 1e-6, asserted 1e-5).
  The CPU block paths, which build the same rows from the lookup, equal it too - outputs and
  gradients, the table's included (FlexAttention has no CPU backward).
* **The table**: its size (section 4.1: ~1.6M), one per layer, every layer's read by that
  layer - in the cached path too - and the no-table arm of ablation 2, which is the network
  without the table.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, permute_boards, synthetic_state, translate
from t5d.net.attention import attention_block_sparse, attention_chunked, attention_explicit, attention_sdpa
from t5d.net.cache import CachedEvaluator, reference_output
from t5d.net.displacement import N_CODES, REGISTER_CODE, DisplacementGeometry, class_map, pair_entry, tile_index
from t5d.net.flex import (
    ScoreInputs,
    attention_flex,
    board_block_mask,
    make_mask_mod,
    make_score_mod,
    materialise_mask,
    materialise_score,
)
from t5d.net.state import (
    TOKEN_BOARD_REGISTER,
    TOKEN_GLOBAL_REGISTER,
    TOKEN_PAD,
    TOKEN_SQUARE,
    PositionState,
)
from t5d.net.tree import Multiverse, SearchTree

SQ, REG, GLOB, PAD = TOKEN_SQUARE, TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_PAD

# --- section 4.1, written out -------------------------------------------------------------------


class Reference:
    """Section 4.1's classes, pair by pair, in the layout ``displacement.py`` documents:
    the board-distance classes' 15 x 15 slices (exact classes ``(dp + 2) 9 + (dh + 4)``, then
    the far classes ``45 + (sign(dh) + 1) bins + bin``), the line entries (the directions that
    also move on the board, then those along the board axes only, each in
    ``itertools.product`` order of (file, rank, timeline, turn) steps), the register entries
    by class and kind, the global entry; padding last, outside the table."""

    def __init__(self, far_edges=(3, 5, 9), line_max=8, register_pairs="by_kind") -> None:
        self.far_edges, self.line_max = tuple(far_edges), line_max
        self.kinds = 3 if register_pairs == "by_kind" else 1
        units = [u for u in itertools.product((-1, 0, 1), repeat=4) if u[2] or u[3]]
        self.spatial = {u: i for i, u in enumerate(u for u in units if u[0] or u[1])}
        self.board = {u: i for i, u in enumerate(u for u in units if not (u[0] or u[1]))}
        self.n_board = 45 + 3 * len(self.far_edges)
        self.lines = self.n_board * 225
        self.board_lines = self.lines + len(self.spatial) * 5  # k = 3 .. 7
        self.registers = self.board_lines + len(self.board) * (line_max - 2)  # k = 3 .. line_max
        self.glob = self.registers + self.n_board * self.kinds
        self.zero = self.glob + 1

    def board_class(self, dp: int, dh: int) -> int:
        if abs(dp) <= 2 and abs(dh) <= 4:  # "exact up to two timelines and two turns apart"
            return (dp + 2) * 9 + (dh + 4)
        distance = max(abs(dp), math.ceil(abs(dh) / 2))  # timelines and whole turns
        far_bin = max(i for i, edge in enumerate(self.far_edges) if distance >= edge)  # the last clamps
        sign = (dh > 0) - (dh < 0)
        return 45 + (sign + 1) * len(self.far_edges) + far_bin

    def __call__(self, kind_q: int, kind_k: int, dx: int, dy: int, dp: int, dh: int) -> int:
        if PAD in (kind_q, kind_k):
            return self.zero
        if GLOB in (kind_q, kind_k):  # "global registers one entry per head"
            return self.glob
        bc = self.board_class(dp, dh)
        if kind_q != SQ or kind_k != SQ:  # "per-board registers by their board-distance class"
            kind = 0 if kind_q == SQ else (1 if kind_k == SQ else 2)
            return self.registers + bc * self.kinds + (kind if self.kinds == 3 else 0)
        if bc >= 45 and dh % 2 == 0:  # beyond the exact classes, in whole turns
            step = (dx, dy, dp, dh // 2)
            sizes = {abs(c) for c in step if c}
            if len(sizes) == 1:  # equal steps along one, two, three or four axes
                k = sizes.pop()
                u = tuple((c > 0) - (c < 0) for c in step)
                if u in self.spatial:
                    return self.lines + self.spatial[u] * 5 + (k - 3)
                return self.board_lines + self.board[u] * (self.line_max - 2) + (min(k, self.line_max) - 3)
        return bc * 225 + (dx + 7) * 15 + (dy + 7)


def _reference_for(cfg: NetConfig) -> Reference:
    return Reference(cfg.disp_far_edges, cfg.disp_line_max, cfg.disp_register_pairs)


def reference_entries(batch, cfg: NetConfig) -> torch.Tensor:
    """Every pair's entry ``(B, L, L)`` from ``Reference``, from the batch's raw coordinates
    (each distinct (kinds, displacement) evaluated once)."""
    ref = _reference_for(cfg)
    kind = torch.where(batch.valid, batch.kind, torch.full_like(batch.kind, PAD)).numpy().astype(np.int64)
    c = batch.coords.round().long().numpy()
    d = [c[:, None, :, a] - c[:, :, None, a] for a in range(4)]  # key minus query
    # One integer per (query kind, key kind, displacement): 4 x 4 kinds, |dx|, |dy| <= 7,
    # |dp| < 2^10, |dh| < 2^14.
    key = ((kind[:, :, None] * 4 + kind[:, None, :]) * 16 + (d[0] + 7)) * 16 + (d[1] + 7)
    key = (key * 2**11 + (d[2] + 2**10)) * 2**15 + (d[3] + 2**14)
    distinct, inverse = np.unique(key.reshape(-1), return_inverse=True)

    def decode(k: int) -> tuple[int, ...]:
        k, dh = divmod(k, 2**15)
        k, dp = divmod(k, 2**11)
        k, dy = divmod(k, 16)
        k, dx = divmod(k, 16)
        kq, kk = divmod(k, 4)
        return kq, kk, dx - 7, dy - 7, dp - 2**10, dh - 2**14

    values = np.array([ref(*decode(int(k))) for k in distinct], dtype=np.int64)
    return torch.from_numpy(values[inverse.reshape(-1)].reshape(kind.shape[0], kind.shape[1], kind.shape[1]))


def reference_bias(model, batch, layer: int, entries: torch.Tensor | None = None) -> torch.Tensor:
    """B15's bias at layer ``layer`` materialised by hand: ``(B, H, L, L)``, the layer's table
    at each pair's reference entry (zero for padding) - differentiable in the table."""
    entries = reference_entries(batch, model.cfg) if entries is None else entries
    table = model.displacement.table[layer].float()
    table = torch.cat([table, table.new_zeros(table.shape[0], 1)], dim=1)
    return table[:, entries].transpose(0, 1)


# --- constructions -------------------------------------------------------------------------------


def far_state(seed: int, size: int = 8, regs: int = 2, globals_: int = 16) -> PositionState:
    """A lattice with boards far apart: every far bin, both signs of Delta-time, odd Delta-times,
    lines of every board direction with k = 3 .. 9 and beyond the clamp, and timelines with
    gaps - (timeline, half-turn) of each board below; contents, edges and candidate moves from a
    synthetic state of the same layout (plausible as lists of tokens, not as geometry)."""
    boards = [(0, 0), (0, 6), (0, 14), (3, 0), (3, 6), (-4, 8), (7, 1), (-9, 18), (12, 30), (5, 9)]
    timeline = np.array([p for p, _ in boards])
    half_turn = np.array([h for _, h in boards])
    is_head = np.array([h == half_turn[timeline == p].max() for p, h in boards])
    src = synthetic_state(seed, len(boards), width=size, height=size, regs_per_board=regs, global_regs=globals_)
    return PositionState(
        width=size,
        height=size,
        timeline=timeline,
        half_turn=half_turn,
        is_head=is_head,
        active=np.ones(len(boards), dtype=bool),
        side_to_move=half_turn & 1,
        piece=src.piece,
        colour=src.colour,
        moved=src.moved,
        edges=src.edges,
        moves=src.moves,
        opp_moves=src.opp_moves,
        regs_per_board=regs,
        global_regs=globals_,
    )


#: The adversarial constructions of section 15.2's protocol: each a list of positions, one batch.
CONSTRUCTIONS = {
    "far": lambda s: [far_state(s)],
    "many_timelines": lambda s: [synthetic_state(s, 9, n_timelines=8)],
    "deep_travel": lambda s: [synthetic_state(s, 9, n_timelines=4, deep_travel_prob=1.0)],
    "duplicated": lambda s: [synthetic_state(s, 9, n_timelines=3, duplicate_prob=0.8)],
    "even_start": lambda s: [synthetic_state(s, 8, n_timelines=4, even_start=True)],
    "small_boards_mixed": lambda s: [synthetic_state(s, 5, width=5, height=5), far_state(s, size=5), synthetic_state(s + 1, 2)],
    "padded_1_2_5": lambda s: [synthetic_state(s, 5), synthetic_state(s + 1, 1), synthetic_state(s + 2, 2)],
    "registers_4_0": lambda s: [synthetic_state(s, 5, regs_per_board=4, global_regs=0), far_state(s, regs=4, globals_=0)],
    "registers_0_16": lambda s: [synthetic_state(s, 5, regs_per_board=0, global_regs=16)],
    "registers_2_0": lambda s: [synthetic_state(s, 5, regs_per_board=2, global_regs=0)],
}

#: The placeholder default and two other settings of the unstated details (spec issues 205-207).
GEOMETRIES = {
    "default": dict(),
    "four_bins_short_lines_shared": dict(disp_far_edges=(3, 4, 7, 13), disp_line_max=5, disp_register_pairs="shared"),
    "one_bin_long_lines": dict(disp_far_edges=(3,), disp_line_max=12),
}


def _model(seed=0, layers=2, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=layers, **flags), seed=seed)


def _qkv(batch, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch.batch_size, 8, batch.n_tokens, 32)
    q, k, v = (torch.randn(*shape, generator=gen) for _ in range(3))
    return q * scale, k * scale, v


# --- the classes ---------------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_entries_match_their_definition_on_a_grid(geometry):
    """Every displacement with |Delta-timeline| <= 14 and |Delta-time| <= 28 half-turns - every
    exact class, every far bin and the clamp, every line family and step and the line clamp -
    for every pair of token kinds: the arithmetic (``pair_entry``, the score function's) and the
    lookup (``class_map`` at ``tile_index``, the CPU paths') both give ``Reference``'s."""
    cfg = NetConfig(**GEOMETRIES[geometry])
    geom = DisplacementGeometry.from_config(cfg)
    ref = _reference_for(cfg)
    assert geom.n_entries == ref.zero, "the same layout size"
    cmap = class_map(geom)
    dps, dhs, steps = range(-14, 15), range(-28, 29), range(-7, 8)
    dp = torch.tensor(dps)[:, None, None, None]
    dh = torch.tensor(dhs)[None, :, None, None]
    dx = torch.tensor(steps)[None, None, :, None]
    dy = torch.tensor(steps)[None, None, None, :]
    squares = pair_entry(torch.tensor(SQ), torch.tensor(SQ), dx, dy, dp, dh, geom)
    expected = torch.tensor([[[[ref(SQ, SQ, a, b, p, h) for b in steps] for a in steps] for h in dhs] for p in dps])
    assert torch.equal(squares, expected)
    for kq, kk in ((a, b) for a in (SQ, REG, GLOB, PAD) for b in (SQ, REG, GLOB, PAD) if (a, b) != (SQ, SQ)):
        got = pair_entry(torch.tensor(kq), torch.tensor(kk), torch.tensor(0), torch.tensor(0), dp[..., 0, 0], dh[..., 0, 0], geom)
        assert torch.equal(got, torch.tensor([[ref(kq, kk, 0, 0, p, h) for h in dhs] for p in dps])), (kq, kk)
    # The lookup: every (query code, key code) of the tile of every board displacement.
    codes = torch.arange(N_CODES)
    code_kind = torch.where(codes == REGISTER_CODE, REG, SQ)
    cx, cy = codes % 8, codes // 8
    tiles = tile_index(dp[..., 0, 0], dh[..., 0, 0], geom)  # (dp, dh)
    by_code = pair_entry(code_kind[:, None], code_kind[None, :], cx[None, :] - cx[:, None], cy[None, :] - cy[:, None], dp, dh, geom)
    assert torch.equal(cmap[tiles], by_code)
    assert (cmap[geom.global_tile] == ref.glob).all() and (cmap[geom.zero_tile] == ref.zero).all()


def test_spot_checks():
    """Section 4.1's examples and the placeholders' edges, one by one."""
    ref, geom = Reference(), DisplacementGeometry()

    def e(dx, dy, dp, dh, kq=SQ, kk=SQ):
        got = int(pair_entry(*(torch.tensor(v) for v in (kq, kk, dx, dy, dp, dh)), geom))
        assert got == ref(kq, kk, dx, dy, dp, dh)
        return got

    # "Knight and king relations fall inside the exact classes": one timeline over, a 2-file
    # offset (a knight jump: 2 + 1 steps) and the within-board knight offset each have their
    # own entry, distinct from the same offsets on one board.
    assert e(2, 0, 1, 0) != e(2, 0, 0, 0) and e(2, 1, 1, 0) != e(2, 1, 0, 0)
    assert e(0, 0, 1, 2) != e(0, 0, -1, 2), "signed: a timeline over either way"
    assert e(1, 1, 1, 2) < 45 * 225 and e(0, 0, 2, 4) < 45 * 225, "a king step diagonal in time, two boards away: exact"
    # Odd Delta-time is never a line: a far pair 3 timelines over and an odd number of
    # half-turns apart gets its far slice cell, never a line entry.
    assert all(e(0, 0, 3, dh) < ref.lines for dh in (-7, -5, 5, 7, 9))
    # The four line families beyond the exact classes: rook (one axis), bishop (two), unicorn
    # (three), dragon (four) - each direction and step its own entry.
    rook, bishop, unicorn, dragon = e(0, 0, 3, 0), e(3, 0, 3, 0), e(3, 3, 3, 0), e(3, 3, 3, 6)
    assert len({rook, bishop, unicorn, dragon}) == 4 and min(rook, bishop, unicorn, dragon) >= ref.lines
    assert e(4, 4, 4, 8) != e(3, 3, 3, 6), "step 4 is not step 3"
    # A line along the board axes longer than the clamp shares the clamp's entry; within it,
    # every step has its own.
    assert e(0, 0, 0, 2 * 8) == e(0, 0, 0, 2 * 9) == e(0, 0, 0, 2 * 40)
    assert len({e(0, 0, 0, 2 * k) for k in range(3, 9)}) == 6
    assert e(0, 0, -9, 18) == e(0, 0, -30, 60), "a board diagonal (timelines and turns), clamped"
    # The far bins, clamped, by the sign of Delta-time.
    assert e(1, 2, 3, 1) == e(1, 2, 4, 1) != e(1, 2, 5, 1), "bins [3, 4] and [5, 8]"
    assert e(1, 2, 9, 1) == e(1, 2, 100, 1), "the last bin holds every larger distance"
    assert e(1, 2, 5, 1) != e(1, 2, 5, -1) != e(1, 2, 5, 0), "past, same time, future"
    # Registers: by class and kind, never by (file, rank); globals one entry either side.
    assert e(3, 4, 0, 0, SQ, REG) == e(0, 0, 0, 0, SQ, REG) != e(0, 0, 0, 0, REG, SQ) != e(0, 0, 0, 0, REG, REG)
    assert e(0, 0, 1, 0, REG, REG) != e(0, 0, 0, 0, REG, REG)
    assert e(1, 1, 0, 0, GLOB, SQ) == e(0, 0, 5, 9, SQ, GLOB) == e(0, 0, 0, 0, GLOB, GLOB) == ref.glob
    assert e(0, 0, 0, 0, PAD, SQ) == e(0, 0, 0, 0, GLOB, PAD) == ref.zero


def test_the_tables_size():
    """Section 4.1: "about 1.6M parameters - ~7% of the network": 54 classes x 225 + 368 line
    entries + 162 register entries + 1 global = 12,681 per head per layer; x 8 x 16 = 1,623,168,
    7.07% of the v1's 22,961,802. Three of each head's entries are never indexed: the (0, 0)
    cell of the same-time far classes, whose displacements are all board-axis lines."""
    geom = DisplacementGeometry()
    assert (geom.n_board_classes, geom.lines_offset, geom.n_entries) == (54, 54 * 225, 12_681)
    assert geom.registers_offset - geom.lines_offset == 64 * 5 + 8 * 6
    counts = parameter_breakdown(build_model(NetConfig()))
    assert counts["attn.displacement"] == 16 * 8 * 12_681 == 1_623_168
    assert abs(counts["attn.displacement"] / counts["total"] - 0.07) < 0.005
    f2 = parameter_breakdown(build_model(NetConfig.preset("L24-ff12")))
    assert f2["attn.displacement"] == 24 * 8 * 12_681
    shared = DisplacementGeometry.from_config(NetConfig(disp_register_pairs="shared"))
    assert shared.n_entries == 12_681 - 2 * 54
    # Every entry but those three is reached by some displacement and token kinds.
    reached = set()
    ref = Reference()
    for dp, dh in itertools.product(range(-10, 11), range(-21, 22)):
        for dx, dy in itertools.product(range(-7, 8), repeat=2):
            reached.add(ref(SQ, SQ, dx, dy, dp, dh))
        reached.update(ref(kq, kk, 0, 0, dp, dh) for kq, kk in ((SQ, REG), (REG, SQ), (REG, REG), (GLOB, SQ)))
    never = set(range(geom.n_entries)) - reached
    assert never == {(45 + 3 * 1 + b) * 225 + 7 * 15 + 7 for b in range(3)}, sorted(never)


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_every_pair_of_the_constructions(construction):
    """The structure's rows - every row at once, query blocks of any span over every key or
    over a key selection, and arbitrary rows - are the bias materialised from
    ``Reference``, bit for bit, at two layers, under each placeholder setting."""
    batch = collate(CONSTRUCTIONS[construction](3))
    length = batch.n_tokens
    gen = torch.Generator().manual_seed(3)
    for name, flags in GEOMETRIES.items():
        model = _model(seed=3, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, **flags)
        ctx = model.make_context(batch)
        entries = reference_entries(batch, model.cfg)
        assert torch.equal(ctx.structure.disp_entries(torch.arange(length).expand(batch.batch_size, length)), entries)
        for layer in (0, 1):
            st = ctx.for_layer(None, layer).structure
            ref = reference_bias(model, batch, layer, entries)
            assert torch.equal(st.dense_bias(), ref), (name, layer)
            for start, stop in ((0, 7), (60, 200), (length - 20, length)):
                assert torch.equal(st.bias_rows(start, stop), ref[:, :, start:stop])
            cols = torch.stack([torch.randperm(length, generator=gen)[:40] for _ in range(batch.batch_size)])
            heads = torch.tensor([1, 5, 6])
            picked = st.bias_rows(10, 30, heads=heads, cols=cols, cols_valid=torch.ones_like(cols, dtype=torch.bool))
            assert torch.equal(picked, ref[:, heads, 10:30].gather(3, cols[:, None, None, :].expand(-1, 3, 20, -1)))
            rows = torch.stack([torch.randperm(length, generator=gen)[:25] for _ in range(batch.batch_size)])
            assert torch.equal(st.bias_rows_at(rows), ref.gather(2, rows[:, None, :, None].expand(-1, 8, -1, length)))


def test_translation_and_board_order_leave_the_bias_alone():
    """Only displacements enter: translating every timeline by +k and every board by +2j
    half-turns leaves the bias bitwise as it was, and storing the boards in another order
    permutes it with them."""
    model = _model(seed=4, layers=1)
    state = synthetic_state(44, 9, n_timelines=4)
    base = model.make_context(collate([state])).for_layer(None, 0).structure.dense_bias()
    for dp, dh in ((5, 12), (-13, 300)):
        moved = model.make_context(collate([translate(state, dp=dp, dh=dh)])).for_layer(None, 0).structure.dense_bias()
        assert torch.equal(moved, base), (dp, dh)
    order = np.random.default_rng(4).permutation(state.n_boards)
    perm = torch.from_numpy(np.concatenate([np.arange(66) + 66 * int(o) for o in order] + [66 * state.n_boards + np.arange(16)]))
    permuted = model.make_context(collate([permute_boards(state, order)])).for_layer(None, 0).structure.dense_bias()
    assert torch.equal(permuted, base[:, :, perm][:, :, :, perm])


def test_boards_wider_than_eight_are_refused():
    """The 15 x 15 slice spans boards up to 8 x 8 - every variant (section 4.1); a wider board
    is refused with B15 on, and the no-table arm takes it."""
    state = synthetic_state(5, 3, width=9, height=9)
    with pytest.raises(ValueError, match="8 x 8"):
        _model(layers=1)(collate([state]))
    assert torch.isfinite(_model(layers=1, displacement_bias=False)(collate([state])).value_logits).all()


def test_the_unstated_details_are_validated():
    for bad in (dict(disp_far_edges=()), dict(disp_far_edges=(4, 6)), dict(disp_far_edges=(3, 3)), dict(disp_line_max=2)):
        with pytest.raises(ValueError, match="disp_"):
            NetConfig(**bad).validate()
    with pytest.raises(ValueError, match="disp_register_pairs"):
        NetConfig(disp_register_pairs="by_slot").validate()


# --- section 15.2's row: the score function = the bias materialised --------------------------------


def _b15_only(ctx, **changes):
    """The context with B15 taken out of its structure (``changes`` on top)."""
    return replace(ctx, structure=replace(ctx.structure, disp=None, disp_index=None, disp_table=None), **changes)


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_score_function_equals_the_bias_materialised(construction):
    """Section 15.2's B15 row. Three seeds - the table at its initialisation scale (0.1) and
    twice at unit scale (where the bias dominates the scores), layers 0 and 1, B9's masks and the
    routing table's: (a) the score function's own term - evaluated at every pair, as FlexAttention's
    eager path evaluates it - is the hand-materialised bias, bit for bit; (b) attention through
    the score function (B14's ``flex_attention`` path) equals attention with that bias
    materialised and passed as a dense bias - measured under 1e-6, asserted 1e-5."""
    settings = [(0.1, dict(), 0), (1.0, dict(routing_table=True), 1), (1.0, dict(), 1)]
    for seed, (std, flags, layer) in enumerate(settings):
        batch = collate(CONSTRUCTIONS[construction](10 + seed))
        q, k, v = _qkv(batch, seed)
        valid = batch.valid[:, None, :, None]
        model = _model(seed=seed, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, disp_init_std=std, **flags)
        ctx = model.make_context(batch).for_layer(None, layer)
        ref = reference_bias(model, batch, layer)
        inp = ScoreInputs.from_context(ctx, 8)
        term = materialise_score(make_score_mod(inp), batch.batch_size, 8, batch.n_tokens, batch.n_tokens)
        assert torch.equal(term, ref), (seed, std, layer)
        through = attention_flex(q, k, v, ctx)
        materialised = attention_explicit(q, k, v, _b15_only(ctx, bias=ref))
        assert torch.allclose(through * valid, materialised * valid, atol=1e-5), float(((through - materialised) * valid).abs().max())


def _outputs_and_grads(attend, model, batch, qkv, w, materialised: bool):
    """One path's output and the gradients of Q, K, V and B15's table - B15 from the structure,
    or (``materialised``) taken out of it and passed as the hand-materialised dense bias."""
    model.zero_grad(set_to_none=True)
    ctx = model.make_context(batch).for_layer(None, 0)
    if materialised:
        ctx = _b15_only(ctx, bias=reference_bias(model, batch, 0))
    leaves = [t.clone().requires_grad_(True) for t in qkv]
    out = attend(*leaves, ctx)
    (out * w).sum().backward()
    return out.detach(), [t.grad for t in leaves] + [model.displacement.table.grad]


#: The CPU paths, as test_chunked.py runs them.
CPU_PATHS = {
    "block_sparse": lambda q, k, v, ctx: attention_block_sparse(q, k, v, replace(ctx, block_rows=66, sparse_max_fraction=1.0)),
    "block_sparse_mixed": lambda q, k, v, ctx: attention_block_sparse(
        q, k, v, replace(ctx, block_rows=2 * 66 + 5, sparse_max_fraction=0.3)
    ),
    "chunked": lambda q, k, v, ctx: attention_chunked(q, k, v, replace(ctx, block_rows=66)),
    "sdpa": attention_sdpa,
    "explicit": lambda q, k, v, ctx: attention_explicit(q, k, v, replace(ctx, chunk=37)),
}


@pytest.mark.grad
@pytest.mark.parametrize(
    "flags",
    [dict(), dict(routing_table=True), dict(board_register_spatial="position_free"), dict(global_register_pairs="anchored")],
    ids=["b9", "routing", "regs-free", "anchored"],
)
@pytest.mark.parametrize("construction", ["far", "small_boards_mixed", "registers_4_0"])
def test_cpu_paths_equal_the_bias_materialised(construction, flags):
    """The CPU paths build B15's rows from the lookup, one query block (or key selection, or
    set of free rows) at a time: their outputs and gradients - Q, K, V and the table - equal
    attention with the bias materialised by hand (test_chunked.py's tolerances); two seeds."""
    for seed in (0, 1):
        batch = collate(CONSTRUCTIONS[construction](20 + seed))
        model = _model(seed=seed, layers=1, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, disp_init_std=1.0, **flags)
        qkv = _qkv(batch, seed)
        w = (
            torch.randn(batch.batch_size, 8, batch.n_tokens, 32, generator=torch.Generator().manual_seed(seed))
            * batch.valid[:, None, :, None]
        )

        out_r, grads_r = _outputs_and_grads(attention_explicit, model, batch, qkv, w, materialised=True)
        valid = batch.valid[:, None, :, None]
        for name, attend in CPU_PATHS.items():
            out, grads = _outputs_and_grads(attend, model, batch, qkv, w, materialised=False)
            assert torch.allclose(out * valid, out_r * valid, atol=1e-5), (name, float(((out - out_r) * valid).abs().max()))
            for i, (ga, gb) in enumerate(zip(grads_r, grads, strict=True)):
                tol = 1e-4 + 1e-5 * float(ga.abs().max())
                assert torch.allclose(ga, gb, atol=tol, rtol=0.0), (name, i, float((ga - gb).abs().max()))
            assert grads[-1].abs().sum() > 0, "the table receives gradient"


@pytest.mark.parametrize(
    "flags",
    [
        dict(),
        dict(routing_table=True, typed_edge_bias=True, typed_edge_directed=True, board_pair_smolgen=True),
        dict(attn_scope="block_causal", typed_edge_bias=True, board_pair_smolgen=True),
        dict(board_register_spatial="position_free", masked_head_registers="self", rope_per_head=True),
        dict(global_register_pairs="anchored", move_graph_heads=4),
        dict(displacement_bias=False),
    ],
    ids=["v1", "routing-typed-smolgen", "block-causal", "regs-free-self-b5", "anchored-mg4", "no-table"],
)
def test_the_network_through_the_score_function(flags):
    """The whole network (two layers) with every attention through the score function equals
    the dense reference's - B11's typed biases and B12's smolgen in the score function too,
    and the block-causal scope in the mask function."""
    batch = collate([far_state(7, size=5), synthetic_state(8, 6, n_timelines=3), synthetic_state(9, 1)])
    outs = {
        impl: build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, **flags), seed=5).eval()(batch, return_trunk=True)
        for impl in ("dense", "flex")
    }
    a, b = outs["dense"], outs["flex"]
    valid = batch.valid[..., None]
    assert torch.allclose(a.trunk * valid, b.trunk * valid, atol=2e-5), float(((a.trunk - b.trunk) * valid).abs().max())
    assert torch.allclose(a.value_logits, b.value_logits, atol=1e-4)
    assert torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=1e-4)


@pytest.mark.grad
def test_the_score_function_refuses_a_cpu_backward():
    model = _model(layers=1)
    batch = collate([synthetic_state(1, 2)])
    q, k, v = (t.requires_grad_(True) for t in _qkv(batch, 1))
    with pytest.raises(NotImplementedError, match="forward-only on CPU"):
        attention_flex(q, k, v, model.make_context(batch).for_layer(None, 0))


@pytest.mark.parametrize(
    "flags",
    [
        dict(),
        dict(routing_table=True, masked_head_registers="self"),
        dict(routing_table=True, same_board_registers="registers", attn_scope="block_causal"),
        dict(move_graph_heads=4, masked_head_registers="self", attn_scope="block_causal"),
    ],
    ids=["b9", "routing-self", "routing-registers-causal", "mg4-self-causal"],
)
@pytest.mark.parametrize("block_size", [66, 128])
def test_the_block_mask_is_lossless(flags, block_size):
    """The block mask from the board-major order covers the block of every allowed pair - the
    mask function evaluated everywhere, which equals the structure's dense mask - on every
    head; and where no register row reads every key (``masked_head_registers="self"``) the
    masked heads skip blocks, while under v1's "read" they cannot (spec issue 209)."""
    batch = collate([synthetic_state(31, 14, n_timelines=5), synthetic_state(32, 3)])
    model = _model(seed=6, layers=1, **flags)
    ctx = model.make_context(batch).for_layer(None, 0)
    inp = ScoreInputs.from_context(ctx, 8)
    allowed = materialise_mask(make_mask_mod(inp), batch.batch_size, 8, batch.n_tokens, batch.n_tokens)
    from t5d.net.attention import _block_mask

    assert torch.equal(allowed, _block_mask(ctx, 0, batch.n_tokens).expand_as(allowed))
    dense_blocks = board_block_mask(ctx, 8, block_size).to_dense().bool()  # (B, H, QB, KB)
    n = -(-batch.n_tokens // block_size)
    padded = torch.zeros(batch.batch_size, 8, n * block_size, n * block_size, dtype=torch.bool)
    padded[..., : batch.n_tokens, : batch.n_tokens] = allowed
    needed = padded.reshape(batch.batch_size, 8, n, block_size, n, block_size).any(dim=(3, 5))
    assert torch.all(dense_blocks | ~needed), "every allowed pair's block is computed"
    masked = [h for h, role in enumerate(ctx.structure.head_roles) if role != "dense"]
    skipped = (~dense_blocks[:, masked]).float().mean()
    if model.cfg.masked_head_registers == "self":
        assert skipped > 0.1, float(skipped)
    elif model.cfg.attn_scope == "dense":
        assert not (~dense_blocks[0, masked]).any(), "under 'read' every query block holds a register row that reads every key"


# --- the table in the network -------------------------------------------------------------------


def test_the_table_reaches_the_outputs():
    """Every layer's scalars reach the board outputs. The value head pools the global
    registers, whose rows get the one global entry on every key - shift-invariant under the
    softmax - so the last layer's table cannot reach it, the first layer's can (through the
    board tokens the later layers' global rows read)."""
    model = build_model(NetConfig(trunk_dtype="fp32"), seed=4).eval()
    batch = collate([synthetic_state(10, 6)])
    base = model(batch)
    table = model.displacement.table
    saved = table.detach().clone()
    for layer in (0, 15):
        table.data[layer] += torch.randn(table.shape[1:], generator=torch.Generator().manual_seed(layer))
        out = model(batch)
        table.data.copy_(saved)
        assert not torch.allclose(out.board_logits, base.board_logits, atol=1e-4), layer
        assert not torch.allclose(out.policy_logits[out.move_valid], base.policy_logits[base.move_valid], atol=1e-4), layer
        reaches_value = not torch.allclose(out.value_logits, base.value_logits, atol=1e-5)
        assert reaches_value == (layer == 0), layer


def test_each_layer_reads_its_own_table():
    """The table is per layer: layer 1's scalars change layer 1's attention and not layer 0's,
    and the bias rows of two layers differ."""
    model = _model(seed=8, layers=2).eval()
    batch = collate([synthetic_state(12, 5)])
    ctx = model.make_context(batch)
    assert not torch.equal(ctx.for_layer(None, 0).structure.dense_bias(), ctx.for_layer(None, 1).structure.dense_bias())
    p0, p1 = (model.attention_probabilities(batch, layer=i) for i in (0, 1))
    model.displacement.table.data[1] += 1.0 + torch.randn(model.displacement.table.shape[1:], generator=torch.Generator().manual_seed(8))
    assert torch.equal(model.attention_probabilities(batch, layer=0), p0)
    assert not torch.allclose(model.attention_probabilities(batch, layer=1), p1, atol=1e-4)


def test_the_cached_path_reads_each_layers_table():
    """Section 15.2's row 1 with B15 on and its layers' tables far apart: a frozen board is
    computed once for every layer (``CachedEvaluator.compute_board``), and the live set layer
    by layer, each with its own layer's table - cached = one full pass. Not vacuous: the same
    pass with the two layers' tables swapped is far from it."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", disp_init_std=2.0), seed=9).eval()
    root = Multiverse.from_state(synthetic_state(1009, 12, n_timelines=4), seed=9)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.root
    ps = sorted(root.heads())
    for move in [None, ("physical", ps[0]), ("physical", ps[1])]:
        if move is not None:
            node = tree.child(node, move)
        cached, ref = tree.evaluate(node), reference_output(model, node.state)
        assert torch.allclose(cached.trunk, ref.trunk, atol=2e-5), float((cached.trunk - ref.trunk).abs().max())
    assert tree.evaluator.store.computed["supersession"] >= 1
    with torch.no_grad():
        model.displacement.table.data = model.displacement.table.data.flip(0)
    swapped = reference_output(model, node.state)
    assert not torch.allclose(swapped.trunk, cached.trunk, atol=1e-2)


# --- ablation 2's no-table arm --------------------------------------------------------------------


def test_the_no_table_arm_is_the_network_without_the_table():
    """``displacement_bias=False`` (ablation 2's no-table arm): no table, every other weight the
    v1's bit for bit (B15 draws from a generator of its own), no bias in the structure under
    B9's default - and the v1 with its table at zero computes the same function."""
    on, off = _model(seed=11, layers=3).eval(), _model(seed=11, layers=3, displacement_bias=False).eval()
    p_on, p_off = dict(on.named_parameters()), dict(off.named_parameters())
    assert set(p_on) - set(p_off) == {"displacement.table"}
    assert all(torch.equal(p, p_on[name]) for name, p in p_off.items())
    batch = collate([far_state(11), synthetic_state(12, 4)])
    st = off.make_context(batch).structure
    assert st.disp is None and not st.has_bias and not st.layer_dependent
    with torch.no_grad():
        on.displacement.table.zero_()
    a, b = on(batch, return_trunk=True), off(batch, return_trunk=True)
    assert torch.allclose(a.trunk, b.trunk, atol=1e-6) and torch.allclose(a.value_logits, b.value_logits, atol=1e-6)
    assert counts_differ_by_the_table(on, off)


def counts_differ_by_the_table(on, off) -> bool:
    a, b = parameter_breakdown(on), parameter_breakdown(off)
    return a["total"] - b["total"] == a["attn.displacement"] == on.displacement.table.numel() and b["attn.displacement"] == 0


def test_no_flag_changes_the_table():
    """B15's table is drawn from its own generator, after everything else: the flags of other
    items never change it, and the seed does."""
    base = _model(seed=13, layers=1).displacement.table
    for flags in (
        dict(typed_edge_bias=True, board_pair_smolgen=True, routing_table=True),
        dict(norm_placement="post"),
        dict(qkv_bias=True),
    ):
        assert torch.equal(_model(seed=13, layers=1, **flags).displacement.table, base), flags
    assert not torch.equal(_model(seed=14, layers=1).displacement.table, base)
    assert abs(float(base.std()) - 0.1) < 0.005, "N(0, disp_init_std)"
