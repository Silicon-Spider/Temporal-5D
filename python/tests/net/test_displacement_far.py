"""B15's two per-axis far tables, and its layer-shared arm (register B6 and B15; spec sections 4.1
and 10.2; the 2026-11-15 review, decision 1).

Section 4.1: "Beyond the exact classes, a coarse slice plus B6's per-axis terms. A far pair takes
its file-rank slice from a class binned by board distance and clamped ... and adds B6's two clamped
log-bucket scalars, one for Delta-time and one for Delta-timeline ... The slice alone cannot tell a
board's own deep past from a present board far across the timelines: both are simply 'far'. The
per-axis terms can, for 3,456 parameters, and they keep B6's extrapolation guard on each axis
separately." And: "Ablation 2 keeps a no-table arm, and gains a layer-shared arm: one table per head
for all sixteen layers, a sixteenth of the parameters and a bias that can be assembled once per
forward (section 10.2)."

``test_displacement.py`` holds B15's classes and section 15.2's B15 row; its hand-materialised
reference (``reference_bias``, ``FarReference``) carries the per-axis terms, so every row there -
the structure's rows bit for bit, the score function bit for bit, the CPU paths' outputs and
gradients - runs with them, with their buckets moved, with them off and under the layer-shared arm.
Held here:

* **the buckets are B6's** - the integer buckets against ``buckets.py`` (which B6 used) and written
  out; the far cells against their definition on a grid and on every construction;
* **what they are for** - pairs one far class merges are told apart; and B6 is B15's special case
  that ignores file and rank (section 4.1, register B6);
* **the default changed, and the off switch gives the old network back** - ``disp_far_axis=False`` has
  the parameters recorded before the review (``test_params.py``'s fingerprints, unfiltered) and, as a
  function, is the default network with both far tables at zero, bit for bit; the tables' own draw
  is pinned; they act only on boards beyond the exact classes;
* **every attention path** - the six implementations agree with them (tolerances as
  ``test_displacement.py``'s: trunk 2e-5, heads 1e-4), gradients reach both tables on the training
  paths, and the cached path equals one full pass with them (section 15.2's row 1; trunk 2e-5),
  frozen rows included - ``attention.py`` and ``cache.py`` take B15 from ``AttnStructure.bias_rows``
  alone, which this checks rather than assumes;
* **the layer-shared arm** - its counts, that it is the per-layer network with equal tables, and
  that its structure needs no layer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, synthetic_state
from t5d.net.buckets import log_bucket, signed_log_bucket
from t5d.net.cache import CachedEvaluator, reference_output
from t5d.net.displacement import (
    DisplacementGeometry,
    FarAxes,
    displacement_index,
    far_cell,
    far_time_bucket,
    far_timeline_bucket,
    pair_entry,
    pair_far_cell,
)
from t5d.net.tree import Multiverse, SearchTree
from tests.net.test_displacement import CONSTRUCTIONS, GLOB, PAD, REG, SQ, FarReference, far_state, reference_bias, reference_far
from tests.net.test_params import _LEAVE_SEED_7, _PRE_D2, _SMALL, _fingerprint, _same

#: B6's buckets moved (the settings exist for this): 11 Delta-time and 5 Delta-timeline buckets.
MOVED = dict(disp_far_time_exact=2, disp_far_time_max=16, disp_far_timeline_exact=1, disp_far_timeline_max=8)

#: The two per-axis far tables' own draw at seed 7 in ``test_params._SMALL`` (``_fingerprint`` of
#: ``displacement.far_time`` and ``displacement.far_timeline``; recorded 2026-11-22 on the build
#: that added them): they draw from a generator of their own (``model.b15_far_seed``).
_FAR_SEED_7 = (0.1397849341764128, 0.17859133551741765, 9.452122048001911)

IMPLS = ("dense", "block_sparse", "chunked", "sdpa", "explicit", "flex")


def _model(seed: int = 0, layers: int = 2, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=layers, **flags), seed=seed).eval()


def _far_tables(model) -> tuple[torch.Tensor, torch.Tensor]:
    return model.displacement.far_time, model.displacement.far_timeline


def _board_pairs_beyond(state) -> int:
    """How many ordered pairs of the position's boards lie outside B15's exact classes."""
    p, h = np.asarray(state.timeline), np.asarray(state.half_turn)
    return int(((np.abs(p[None, :] - p[:, None]) > 2) | (np.abs(h[None, :] - h[:, None]) > 4)).sum())


# --- the buckets and the cells ----------------------------------------------------------------------------


def test_far_buckets_are_b6s():
    """The two tables' buckets are B6's as it was built (build-log issue 205): signed Delta-time
    exact below 4 half-turns, then 4-7, 8-15, ..., 64-127, 128+ - 19; |Delta-timeline| exact below
    4, then 4-7, 8-15, 16-31, 32+ - 8. Against ``buckets.py``'s float form (B6's own, which B12 and
    B16 still use), against the lists written out, and with the settings moved."""
    far = FarAxes()
    assert FarAxes.from_config(NetConfig()) == far
    assert (far.n_time, far.n_timeline, far.n_cells, far.zero_cell) == (19, 8, 152, 152)
    assert (far.time_edges, far.timeline_edges) == ((8, 16, 32, 64, 128), (8, 16, 32))
    dh, dp = torch.arange(-600, 601), torch.arange(-300, 301)
    assert torch.equal(far_time_bucket(dh, far), signed_log_bucket(dh, 4, 128))
    assert torch.equal(far_timeline_bucket(dp, far), log_bucket(dp, 4, 32))
    sizes = [0, 1, 2, 3] + [4] * 4 + [5] * 8 + [6] * 16 + [7] * 32 + [8] * 64 + [9] * 100  # |Delta-time| 0 .. 227
    assert far_time_bucket(torch.arange(len(sizes)), far).tolist() == sizes, "the future, and the same time"
    assert far_time_bucket(-torch.arange(1, len(sizes)), far).tolist() == [9 + b for b in sizes[1:]], "the past: its own nine buckets"
    widths = [0, 1, 2, 3] + [4] * 4 + [5] * 8 + [6] * 16 + [7] * 40  # |Delta-timeline| 0 .. 71
    for sign in (1, -1):
        assert far_timeline_bucket(sign * torch.arange(len(widths)), far).tolist() == widths, "Delta-timeline's sign is dropped"
    moved = FarAxes.from_config(NetConfig(**MOVED))
    assert (moved.n_time, moved.n_timeline, moved.time_edges, moved.timeline_edges) == (11, 5, (4, 8, 16), (2, 4, 8))
    assert torch.equal(far_time_bucket(dh, moved), signed_log_bucket(dh, 2, 16))
    assert torch.equal(far_timeline_bucket(dp, moved), log_bucket(dp, 1, 8))
    # Edges that are no power of two times the exact count: the last bucket starts at the last edge reached.
    odd = FarAxes(time_exact=3, time_max=100, timeline_exact=5, timeline_max=5)
    assert (odd.time_edges, odd.timeline_edges, odd.n_time, odd.n_timeline) == ((6, 12, 24, 48, 96), (), 17, 6)
    assert torch.equal(far_time_bucket(dh, odd), signed_log_bucket(dh, 3, 100))
    assert torch.equal(far_timeline_bucket(dp, odd), log_bucket(dp, 5, 5))


def _reference_cell(ref: FarReference, far: FarAxes, kind_q: int, kind_k: int, dp: int, dh: int) -> int:
    buckets = ref(kind_q, kind_k, dp, dh)
    return far.zero_cell if buckets is None else buckets[0] * ref.n_timeline + buckets[1]


@pytest.mark.parametrize("setting", ["default", "moved"])
def test_far_cells_match_their_definition_on_a_grid(setting):
    """Every board displacement with |Delta-timeline| <= 40 and |Delta-time| <= 300 half-turns -
    every bucket of both axes and both clamps: the arithmetic (``far_cell``, the score function's)
    gives ``FarReference``'s buckets - written out in plain Python - and the zero cell exactly inside
    the exact classes. By token kinds (``pair_far_cell``): the boards' cell for squares and per-board
    registers in any pairing, the zero cell for any pair with a global register or padding."""
    flags = {} if setting == "default" else MOVED
    far = FarAxes.from_config(NetConfig(**flags))
    ref = FarReference((far.time_exact, far.time_max), (far.timeline_exact, far.timeline_max))
    assert (far.n_time, far.n_timeline) == (ref.n_time, ref.n_timeline)
    dps, dhs = range(-40, 41), range(-300, 301)
    dp, dh = torch.tensor(dps)[:, None], torch.tensor(dhs)[None, :]
    cells = far_cell(dp, dh, far)
    assert torch.equal(cells, torch.tensor([[_reference_cell(ref, far, SQ, SQ, p, h) for h in dhs] for p in dps]))
    inside = (dp.abs() <= 2) & (dh.abs() <= 4)
    assert torch.equal(cells == far.zero_cell, inside.expand_as(cells)), "no per-axis term inside the exact classes, one everywhere beyond"
    assert set(cells[~inside.expand_as(cells)].tolist()) <= set(range(far.n_cells))
    for kq in (SQ, REG, GLOB, PAD):
        for kk in (SQ, REG, GLOB, PAD):
            got = pair_far_cell(torch.tensor(kq), torch.tensor(kk), dp, dh, far)
            on_boards = kq in (SQ, REG) and kk in (SQ, REG)
            assert torch.equal(got, cells if on_boards else torch.full_like(cells, far.zero_cell)), (kq, kk)


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_the_index_holds_every_pairs_far_cell(construction):
    """The CPU paths' per-batch matrix of far cells by board slot (``DisplacementIndex.far_cells``)
    gives every pair of tokens the cell the arithmetic gives it from its kinds and coordinates, and
    ``FarReference``'s buckets - integers, compared exactly - on every construction, padded batches
    and Global Warming's 1 x 1 boards included; without the per-axis tables it is not built."""
    batch = collate(CONSTRUCTIONS[construction](5))
    b = torch.arange(batch.batch_size)[:, None, None]
    for flags in ({}, MOVED):
        cfg = NetConfig(**flags)
        geom, far = DisplacementGeometry.from_config(cfg), FarAxes.from_config(cfg)
        index = displacement_index(batch, geom, far)
        by_slot = index.far_cells[b, index.slot[:, :, None], index.slot[:, None, :]]
        c = index.coords
        dp, dh = (c[:, None, :, a] - c[:, :, None, a] for a in (2, 3))
        assert torch.equal(by_slot, pair_far_cell(index.kind[:, :, None], index.kind[:, None, :], dp, dh, far))
        takes, t, tl = reference_far(batch, cfg)
        assert torch.equal(by_slot, torch.where(takes, t * far.n_timeline + tl, torch.full_like(t, far.zero_cell)))
    assert displacement_index(batch, DisplacementGeometry()).far_cells is None


def test_the_far_buckets_are_validated():
    cfg = NetConfig()
    assert cfg.disp_far_axis is True and cfg.disp_layer_shared is False, "the 2026-11-15 ruling's defaults"
    assert (cfg.disp_far_time_exact, cfg.disp_far_time_max, cfg.disp_far_timeline_exact, cfg.disp_far_timeline_max) == (4, 128, 4, 32)
    for bad in (
        dict(disp_far_time_exact=0),
        dict(disp_far_timeline_exact=0),
        dict(disp_far_time_max=3),
        dict(disp_far_timeline_exact=3, disp_far_timeline_max=2),
    ):
        with pytest.raises(ValueError, match="disp_far_"):
            NetConfig(**bad).validate()
    for good in (MOVED, dict(disp_far_axis=False, disp_layer_shared=True), dict(disp_far_time_max=4, disp_far_timeline_max=4)):
        NetConfig(**good).validate()


# --- what the tables are for ---------------------------------------------------------------------------------


def test_the_per_axis_tables_tell_apart_what_one_distance_merges():
    """Issue 205's collisions: B15's far class is one distance ``D = max(|Delta-timeline|,
    ceil(|Delta-time| / 2))`` crossed with Delta-time's sign, so a board 20 half-turns back on the
    query's own timeline and one 2 half-turns back 10 timelines over share a class - and, off the
    lines, every entry. The per-axis cells differ. In a position with such boards the two pairs'
    biases are equal on every head without the far tables and differ on every head with them, by
    exactly the two scalars of their buckets."""
    geom, far = DisplacementGeometry(), FarAxes()

    def scalars(*values):
        return tuple(torch.tensor(v) for v in values)

    for a, b in (((0, -20), (10, -2)), ((0, 20), (0, 100)), ((10, 0), (30, 0)), ((3, 20), (9, 20))):
        entries = [int(pair_entry(*scalars(SQ, SQ, 1, 2, dp, dh), geom)) for dp, dh in (a, b)]
        cells = [int(far_cell(*scalars(dp, dh), far)) for dp, dh in (a, b)]
        assert entries[0] == entries[1] and cells[0] != cells[1], (a, b)
    assert int(pair_entry(*scalars(SQ, SQ, 1, 2, 0, -20), geom)) == 47 * 225 + 8 * 15 + 9
    # Section 4.1's own example - "a board's own deep past" against "a present board far across the
    # timelines" - is told apart by Delta-time's sign already (a spec issue: the report proposes the
    # wording); it is once the far-across board is also in the past that the slice merges them.
    assert int(pair_entry(*scalars(SQ, SQ, 1, 2, 0, -20), geom)) != int(pair_entry(*scalars(SQ, SQ, 1, 2, 10, 0), geom))

    state = far_state(61, boards=[(0, 20), (0, 0), (10, 18)])
    batch = collate([state])
    lay = state.layout
    q, own_past, across = lay.square_token(0, 2, 3), lay.square_token(1, 3, 5), lay.square_token(2, 3, 5)
    c = batch.coords[0].round().long()
    assert (c[own_past] - c[q]).tolist() == [1, 2, 0, -20] and (c[across] - c[q]).tolist() == [1, 2, 10, -2]
    on, off = _model(seed=61), _model(seed=61, disp_far_axis=False)
    bias_on, bias_off = (m.make_context(batch).for_layer(None, 0).structure.dense_bias()[0] for m in (on, off))
    assert torch.equal(bias_off[:, q, own_past], bias_off[:, q, across]), "without the per-axis tables: one entry, one bias"
    assert bool((bias_on[:, q, own_past] != bias_on[:, q, across]).all()), "with them: apart on every head"
    far_time, far_timeline = (t.detach()[0] for t in _far_tables(on))
    # By hand: Delta-time -20 is the past's 16-31 bucket (9 + 6) and -2 its exact bucket 2 (9 + 2);
    # |Delta-timeline| 0 is bucket 0 and 10 the 8-15 bucket (5).
    assert torch.equal(bias_on[:, q, own_past], bias_off[:, q, own_past] + (far_time[:, 15] + far_timeline[:, 0]))
    assert torch.equal(bias_on[:, q, across], bias_off[:, q, across] + (far_time[:, 11] + far_timeline[:, 5]))


def test_b6_is_the_special_case_that_ignores_file_and_rank():
    """Section 4.1 and register B6: B15 "keeps these two per-axis tables for pairs beyond its exact
    classes - so B6 is exactly its special case that ignores file and rank". Given any B6 - a table
    ``f`` over the 19 Delta-time buckets and ``g`` over the 8 Delta-timeline buckets, per head - B15
    with its far tables set to ``f`` and ``g``, each exact class's slice and register entries set to
    the constant ``f + g`` of the class's own displacement, and everything else at zero (the far
    slices, the lines, the far register entries, the global entry) is that B6: its bias on every
    pair of board tokens is ``f[bucket(Delta-time)] + g[bucket(|Delta-timeline|)]`` whatever their
    squares, and zero for global registers and padding."""
    model = _model(seed=3, layers=1)
    disp = model.displacement
    geom = disp.geom
    gen = torch.Generator().manual_seed(3)
    f, g = torch.randn(8, 19, generator=gen), torch.randn(8, 8, generator=gen)
    with torch.no_grad():
        disp.far_time[0].copy_(f)
        disp.far_timeline[0].copy_(g)
        disp.table.zero_()
        for bc in range(45):  # the exact classes: (dp + 2) 9 + (dh + 4)
            dp, dh = torch.tensor(bc // 9 - 2), torch.tensor(bc % 9 - 4)
            b6 = f[:, int(signed_log_bucket(dh, 4, 128))] + g[:, int(log_bucket(dp, 4, 32))]
            disp.table[0, :, bc * 225 : (bc + 1) * 225] = b6[:, None]
            registers = geom.registers_offset + bc * geom.register_kinds
            disp.table[0, :, registers : registers + geom.register_kinds] = b6[:, None]
    batch = collate([far_state(3), synthetic_state(4, 5, n_timelines=3)])
    bias = model.make_context(batch).for_layer(None, 0).structure.dense_bias()
    kind = torch.where(batch.valid, batch.kind, torch.full_like(batch.kind, PAD))
    on_board = (kind == SQ) | (kind == REG)
    both = on_board[:, :, None] & on_board[:, None, :]
    c = batch.coords.round().long()
    dp, dh = (c[:, None, :, a] - c[:, :, None, a] for a in (2, 3))
    b6 = (f[:, signed_log_bucket(dh, 4, 128)] + g[:, log_bucket(dp, 4, 32)]).transpose(0, 1)  # (B, H, L, L)
    expected = torch.where(both[:, None], b6, torch.zeros(()))
    assert torch.allclose(bias, expected, atol=1e-6), float((bias - expected).abs().max())
    beyond = both & ((dp.abs() > 2) | (dh.abs() > 4))
    assert beyond.any() and (both & ~beyond).any() and (~both & batch.valid[:, :, None] & batch.valid[:, None, :]).any()
    assert bool((kind == REG).any()) and bool((kind == GLOB).any()) and bool((kind == PAD).any())


# --- the default changed; the off switch is the network as built before the review ---------------------------


def test_the_far_off_arms_parameters_are_those_recorded_before_the_review():
    """``disp_far_axis=False``: every parameter of the network is the one drawn before the per-axis
    tables existed - ``test_params.py``'s fingerprints of the whole parameter set, recorded on an
    earlier commit's build and taken here with no tensor left out. The default network has those
    same parameters and two more, the far tables, whose own draw is pinned (their own generator)."""
    off = dict(build_model(NetConfig(**_SMALL, disp_far_axis=False), seed=7).named_parameters())
    count, expected = _PRE_D2["default"]
    kept = {n: p for n, p in off.items() if not n.startswith("leave.")}
    assert len(kept) == count and len(off) == count + 2
    _same(_fingerprint(kept), expected, count)
    _same(_fingerprint({n: p for n, p in off.items() if n.startswith("leave.")}), _LEAVE_SEED_7, 2)
    on = dict(build_model(NetConfig(**_SMALL), seed=7).named_parameters())
    assert set(on) - set(off) == {"displacement.far_time", "displacement.far_timeline"} and set(off) <= set(on)
    assert [n for n in on if n in off] == list(off), "and in the same order"
    assert all(torch.equal(p, on[n]) for n, p in off.items())
    far = {n: p for n, p in on.items() if n not in off}
    _same(_fingerprint(far), _FAR_SEED_7, 2, values=sum(p.numel() for p in far.values()))  # 108 values: a small set's bound


def test_the_far_off_arm_is_the_default_network_without_the_far_terms():
    """As a function: ``disp_far_axis=False`` builds no far table, no far cell and no far row, and the
    default network with both far tables at zero computes it bit for bit (the term is then an added
    zero); with the tables as drawn the two differ. The tables are ``N(0, disp_init_std)``, a
    function of the seed alone - no other flag moves them, the main table's geometry included."""
    on, off = _model(seed=21, layers=4), _model(seed=21, layers=4, disp_far_axis=False)
    assert set(on.state_dict()) - set(off.state_dict()) == {"displacement.far_time", "displacement.far_timeline"}
    assert off.displacement.far is None and off.displacement.far_time is None and off.displacement.far_timeline is None
    assert off.displacement.layer_far_table(0) is None
    batch = collate([far_state(21), synthetic_state(22, 4)])
    st = off.make_context(batch).for_layer(None, 1).structure
    assert st.disp_far_table is None and st.disp_far_pairs is None and st.disp_index.far_cells is None
    a, b = on(batch, return_trunk=True), off(batch, return_trunk=True)
    assert not torch.allclose(a.trunk, b.trunk, atol=1e-3), "as drawn, the per-axis terms are live"
    assert not torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=1e-3)
    far_time, far_timeline = _far_tables(on)
    for table in (far_time, far_timeline):
        assert abs(float(table.std()) - 0.1) < 0.02, "N(0, disp_init_std)"
    for flags in (
        dict(typed_edge_bias=True, board_pair_smolgen=True, routing_table=True),
        dict(norm_placement="post"),
        dict(disp_far_edges=(3,), disp_line_max=12, disp_register_pairs="shared"),
        dict(attn_scope="block_causal"),
    ):
        other = _model(seed=21, layers=4, **flags)
        assert all(torch.equal(x, y) for x, y in zip(_far_tables(other), (far_time, far_timeline), strict=True)), flags
    assert not torch.equal(_model(seed=22, layers=4).displacement.far_time, far_time)
    assert math.isclose(float(_model(seed=21, layers=4, disp_init_std=1.0).displacement.far_time.std()), 1.0, abs_tol=0.1)
    with torch.no_grad():
        far_time.zero_()
        far_timeline.zero_()
    a = on(batch, return_trunk=True)
    for name in ("trunk", "value_logits", "wdl_logits", "board_logits", "square_logits", "policy_logits", "leave_logits"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_the_far_tables_act_only_beyond_the_exact_classes():
    """A position whose boards all lie within two timelines and four half-turns of each other reads
    neither table - changing them changes no output, bit for bit - and a position with boards far
    apart reads both: each alone moves the outputs."""
    model = _model(seed=10)
    near_state, far_position = synthetic_state(10, 6), far_state(10)
    assert _board_pairs_beyond(near_state) == 0 and _board_pairs_beyond(far_position) > 0
    near, far = collate([near_state]), collate([far_position])
    names = ("trunk", "value_logits", "wdl_logits", "board_logits", "square_logits", "policy_logits")
    base_near, base_far = model(near, return_trunk=True), model(far, return_trunk=True)
    gen = torch.Generator().manual_seed(10)
    for table in _far_tables(model):
        saved = table.detach().clone()
        with torch.no_grad():
            table.add_(torch.randn(table.shape, generator=gen))
        out_near, out_far = model(near, return_trunk=True), model(far, return_trunk=True)
        with torch.no_grad():
            table.copy_(saved)
        for name in names:
            assert torch.equal(getattr(out_near, name), getattr(base_near, name)), name
        assert not torch.allclose(out_far.board_logits, base_far.board_logits, atol=1e-4)
        assert not torch.allclose(out_far.policy_logits[out_far.move_valid], base_far.policy_logits[base_far.move_valid], atol=1e-4)


# --- every attention path ---------------------------------------------------------------------------------------

#: The default (the per-axis tables on), their buckets moved, the tables off, ablation 2's
#: layer-shared arm, and the cached configuration's scope (B16 beside them, in the score function).
PATH_ARMS = {
    "default": dict(),
    "moved": MOVED,
    "far_axis_off": dict(disp_far_axis=False),
    "layer_shared": dict(disp_layer_shared=True),
    "block_causal": dict(attn_scope="block_causal"),
    "block_causal_layer_shared": dict(attn_scope="block_causal", disp_layer_shared=True, typed_edge_bias=True, board_pair_smolgen=True),
}


@pytest.mark.parametrize("arm", list(PATH_ARMS))
def test_every_implementation_is_one_function_with_the_far_tables(arm):
    """The whole network (two layers; B15's tables at unit scale, where the bias dominates the
    scores) through the six attention implementations - the dense reference, the block-sparse and
    chunked kernels, the fused path, the explicit one and B14's score function, which computes the
    far cell inside the kernel's score function - on a batch with boards far apart: one function."""
    batch = collate([far_state(7, size=5), synthetic_state(8, 6, n_timelines=3)])
    assert _board_pairs_beyond(far_state(7, size=5)) > 0
    outs = {impl: _model(seed=5, attn_impl=impl, disp_init_std=1.0, **PATH_ARMS[arm])(batch, return_trunk=True) for impl in IMPLS}
    ref = outs["dense"]
    valid = batch.valid[..., None]
    for impl in IMPLS[1:]:
        out = outs[impl]
        assert torch.allclose(out.trunk * valid, ref.trunk * valid, atol=2e-5), (impl, float(((out.trunk - ref.trunk) * valid).abs().max()))
        assert torch.allclose(out.value_logits, ref.value_logits, atol=1e-4), impl
        assert torch.allclose(out.policy_logits[out.move_valid], ref.policy_logits[ref.move_valid], atol=1e-4), impl
        assert torch.allclose(out.board_logits, ref.board_logits, atol=1e-4), impl


@pytest.mark.grad
@pytest.mark.parametrize("flags", [dict(), dict(disp_layer_shared=True)], ids=["per-layer", "layer-shared"])
def test_gradients_reach_every_b15_table_on_the_training_paths(flags):
    """The block-sparse and chunked paths (whose blocks are recomputed in the backward pass) give
    B15's three tensors - the table and the two per-axis far tables - the dense reference's
    gradients, none of them zero, per layer and under the layer-shared arm (where every layer's
    blocks read the one table set when the structure was built)."""
    batch = collate([far_state(51, size=5), synthetic_state(52, 4)])
    grads = {}
    for impl in ("dense", "chunked", "block_sparse"):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, attn_block_boards=1, **flags), seed=51)
        out = model(batch)
        gen = torch.Generator().manual_seed(51)
        loss = (out.value_logits * torch.randn(out.value_logits.shape, generator=gen)).sum()
        loss = loss + (out.policy_logits[out.move_valid] * torch.randn(int(out.move_valid.sum()), generator=gen)).sum()
        loss = loss + (out.board_logits * torch.randn(out.board_logits.shape, generator=gen)).sum()
        loss.backward()
        grads[impl] = {n: p.grad for n, p in model.displacement.named_parameters()}
    assert set(grads["dense"]) == {"table", "far_time", "far_timeline"}
    for name, g in grads["dense"].items():
        assert g is not None and g.abs().sum() > 0, name
        assert g.shape[0] == (1 if flags else 2), "one table for every layer under the arm"
        for impl in ("chunked", "block_sparse"):
            other = grads[impl][name]
            assert torch.allclose(other, g, atol=2e-4, rtol=1e-3), (impl, name, float((other - g).abs().max()))


def _far_lineage_boards(state) -> list[int]:
    """The frozen boards whose causal past (C23: the lineage, ``PositionState.parent``) holds a
    board beyond the exact classes - their rows carry a per-axis term."""
    p, h, parent = (np.asarray(x) for x in (state.timeline, state.half_turn, state.parent))
    out = []
    for b in range(state.n_boards):
        a = int(parent[b])
        while a >= 0 and not state.is_head[b]:
            if abs(int(p[a] - p[b])) > 2 or abs(int(h[a] - h[b])) > 4:
                out.append(b)
                break
            a = int(parent[a])
    return out


@pytest.mark.parametrize("shared", [False, True], ids=["per-layer", "layer-shared"])
def test_the_cached_path_carries_the_far_terms(shared):
    """Section 15.2's row 1 (C2, C17) with the per-axis tables far from zero: along a search path
    every node's cached evaluation equals one full block-causal pass - the live pass, and the frozen
    boards computed once with their rows' own terms (B15 is geometry: it biases frozen rows too) -
    with a supersession computed inside the search. Not vacuous: with the far tables changed the
    full pass moves away from the cached outputs, on the rows of frozen boards that read a far
    board of their causal past too, and a cache built then equals the full pass again. (The cached
    path takes B15 from ``AttnStructure.bias_rows``; ``cache.py`` needed no change.)"""
    flags = dict(n_layers=2, attn_scope="block_causal", disp_init_std=2.0, disp_layer_shared=shared)
    model = build_model(NetConfig(trunk_dtype="fp32", **flags), seed=9).eval()
    root = Multiverse.from_state(synthetic_state(1009, 12, n_timelines=4), seed=9)
    ps = sorted(root.heads())
    path = [None, ("physical", ps[0]), ("physical", ps[1])]

    def cached_along_the_path():
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.root
        for move in path:
            if move is not None:
                node = tree.child(node, move)
            cached, ref = tree.evaluate(node), reference_output(model, node.state)
            assert torch.allclose(cached.trunk, ref.trunk, atol=2e-5), (move, float((cached.trunk - ref.trunk).abs().max()))
            assert torch.allclose(cached.value_logits, ref.value_logits, atol=1e-4), move
        assert tree.evaluator.store.computed["supersession"] >= 1
        return node.state, cached

    state, cached = cached_along_the_path()
    frozen_far = _far_lineage_boards(state)
    assert _board_pairs_beyond(state) > 0 and frozen_far, "the position holds far pairs, and frozen boards that read one"
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for table in _far_tables(model):
            table.add_(2.0 * torch.randn(table.shape, generator=gen))
    moved = reference_output(model, state)
    assert not torch.allclose(moved.trunk, cached.trunk, atol=1e-2), "the per-axis terms act on this position"
    tpb = state.layout.tokens_per_board
    rows = torch.cat([torch.arange(b * tpb, (b + 1) * tpb) for b in frozen_far])
    assert not torch.allclose(moved.trunk[0, rows], cached.trunk[0, rows], atol=1e-2), "and on its frozen boards' own rows"
    cached_along_the_path()


# --- the counts, and ablation 2's layer-shared arm -----------------------------------------------------------------


def test_parameter_counts_of_the_far_tables_and_the_shared_arm():
    """Section 4.1: "1,623,168 parameters ... before the per-axis terms", which are 3,456 - (19 + 8)
    x 8 heads x 16 layers - reported under their own key; the v1 is 22,969,139 with them and
    22,965,683 without (``disp_far_axis=False``, the network before the review). F2's 24 layers:
    5,184, 46,900,275. The layer-shared arm: "one table per head for all sixteen layers, a sixteenth
    of the parameters" - 8 x 12,681 = 101,448 and 8 x 27 = 216, 101,664 of 1,626,624; 21,444,179."""
    v1 = parameter_breakdown(build_model(NetConfig()))
    assert (v1["attn.displacement"], v1["attn.displacement_far"], v1["total"]) == (1_623_168, 16 * 8 * 27, 22_969_139)
    assert v1["attn.displacement_far"] == 3_456
    f2 = parameter_breakdown(build_model(NetConfig.preset("L24-ff12")))
    assert (f2["attn.displacement"], f2["attn.displacement_far"], f2["total"]) == (2_434_752, 5_184, 46_900_275)
    shared = parameter_breakdown(build_model(NetConfig(disp_layer_shared=True)))
    assert (shared["attn.displacement"], shared["attn.displacement_far"], shared["total"]) == (8 * 12_681, 8 * 27, 21_444_179)
    assert 16 * (shared["attn.displacement"] + shared["attn.displacement_far"]) == v1["attn.displacement"] + v1["attn.displacement_far"]
    assert shared["attn.displacement"] + shared["attn.displacement_far"] == 101_664
    off = parameter_breakdown(build_model(NetConfig(disp_far_axis=False)))
    assert (off["attn.displacement"], off["attn.displacement_far"], off["total"]) == (1_623_168, 0, 22_965_683)
    shared_off = parameter_breakdown(build_model(NetConfig(disp_layer_shared=True, disp_far_axis=False)))
    assert (shared_off["attn.displacement"], shared_off["attn.displacement_far"]) == (101_448, 0)
    no_table = parameter_breakdown(build_model(NetConfig(displacement_bias=False)))
    assert no_table["attn.displacement"] == no_table["attn.displacement_far"] == 0 and no_table["total"] == 22_969_139 - 1_626_624
    rest = lambda counts: {k: v for k, v in counts.items() if k not in ("attn.displacement", "attn.displacement_far", "total")}  # noqa: E731
    assert rest(v1) == rest(shared) == rest(off) == rest(no_table), "the arms differ in B15 alone"
    moved = build_model(NetConfig(n_layers=2, **MOVED)).displacement
    assert (tuple(moved.far_time.shape), tuple(moved.far_timeline.shape)) == ((2, 8, 11), (2, 8, 5))


def test_the_layer_shared_arm_equals_the_per_layer_network_with_equal_tables():
    """Ablation 2's arm is the same function family with the layers' tables tied: every parameter
    outside B15 is the per-layer network's, bit for bit (B15 draws from its own generators); B15
    has one table per head - the main table and both far tables - read by every layer; and the
    per-layer network with the shared tables copied into each of its layers computes the arm's
    function exactly."""
    shared, per_layer = _model(seed=31, layers=3, disp_layer_shared=True), _model(seed=31, layers=3)
    ps, pl = dict(shared.named_parameters()), dict(per_layer.named_parameters())
    assert list(ps) == list(pl)
    assert all(torch.equal(p, pl[n]) for n, p in ps.items() if not n.startswith("displacement."))
    names = ("table", "far_time", "far_timeline")
    assert [tuple(ps[f"displacement.{n}"].shape) for n in names] == [(1, 8, 12_681), (1, 8, 19), (1, 8, 8)]
    assert [tuple(pl[f"displacement.{n}"].shape) for n in names] == [(3, 8, 12_681), (3, 8, 19), (3, 8, 8)]
    assert shared.displacement.per_layer is False and per_layer.displacement.per_layer is True
    assert torch.equal(shared.displacement.layer_table(2), shared.displacement.layer_table(0))
    assert torch.equal(shared.displacement.layer_far_table(2), shared.displacement.layer_far_table(0))
    batch = collate([far_state(31, size=6), synthetic_state(32, 5, n_timelines=3)])
    a, b = shared(batch, return_trunk=True), per_layer(batch, return_trunk=True)
    assert not torch.allclose(a.trunk, b.trunk, atol=1e-3), "three layers' tables against one"
    with torch.no_grad():
        for name in names:
            getattr(per_layer.displacement, name).copy_(getattr(shared.displacement, name).expand(3, -1, -1))
    b = per_layer(batch, return_trunk=True)
    for name in ("trunk", "value_logits", "board_logits", "policy_logits"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_the_layer_shared_structure_needs_no_layer():
    """Section 4.1: the arm's bias "can be assembled once per forward". Under it the structure
    carries B15's tables from the moment it is built: it is not layer-dependent, ``for_layer``
    returns it as it is, and its rows are the hand-materialised bias with no layer chosen - where
    the per-layer default refuses to build a row before ``for_layer``. (Register G11 materialises
    the (head, T, T) bias once per forward at inference from this function: tests/net/
    test_once_per_forward.py.) With B12 on
    the structure still needs each layer's input, and B15's tables stay the ones set when built."""
    batch = collate([far_state(41, size=4), synthetic_state(42, 3)])
    shared = _model(seed=41, disp_layer_shared=True)
    ctx = shared.make_context(batch)
    assert not ctx.structure.layer_dependent and not ctx.layer_dependent
    assert ctx.for_layer(None, 1) is ctx and ctx.structure.for_layer(None, 1) is ctx.structure
    bias = ctx.structure.dense_bias()
    assert torch.equal(bias, reference_bias(shared, batch, 0)) and torch.equal(bias, reference_bias(shared, batch, 1))
    assert ctx.structure.disp_far_pairs.shape == (2, 8, batch.board_valid.shape[1] + 2, batch.board_valid.shape[1] + 2)
    per_layer = _model(seed=41)
    ctx = per_layer.make_context(batch)
    assert ctx.structure.layer_dependent and ctx.structure.disp_table is None and ctx.structure.disp_far_pairs is None
    with pytest.raises(RuntimeError, match="for_layer"):
        ctx.structure.dense_bias()
    with_b12 = _model(seed=41, disp_layer_shared=True, board_pair_smolgen=True)
    st = with_b12.make_context(batch).structure
    assert st.layer_dependent and st.disp_table is not None
    h = torch.randn(batch.batch_size, batch.n_tokens, 256, generator=torch.Generator().manual_seed(41))
    layered = st.for_layer(h, 1)
    assert layered.disp_table is st.disp_table and layered.disp_far_pairs is st.disp_far_pairs and layered.pair_table is not None


@pytest.mark.grad
def test_a_smoke_step_moves_the_layer_shared_tables():
    """One optimiser step on synthetic rows (no training run) under the layer-shared arm: every
    tensor moves, B15's three among them (the smoke batch's 9-board position holds far boards)."""
    from t5d.train.smoke import smoke_train

    report = smoke_train(NetConfig(trunk_dtype="fp32", n_layers=2, disp_layer_shared=True), steps=1, seed=0, board_counts=(1, 3, 6, 9))
    assert all(math.isfinite(v) for v in report.losses[0].values())
    assert report.unchanged == [] and report.changed_fraction == 1.0, report.unchanged
