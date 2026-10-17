"""B16, the score-time status key biases on frozen boards (spec sections 4.1, 7.3; register B16,
A3, M20), in every attention path.

* **The bias against its definition.** ``status_reference`` writes section 4.1's term out from
  the positions themselves: for a live query row (a head board's token or a global register)
  and a key on a frozen board - its squares and its per-board registers - the head's entry for
  that board's status, ``bin(Delta-t-to-present) * 2 + active``, the bin written out by hand;
  zero on every other pair. The structure's rows - every row, any block, key selection, row set
  or head subset - are that, bit for bit, on positions that reach every bin, past and future,
  on active and inactive timelines, and on section 15.2's adversarial constructions; under the
  placeholder default and other settings of the bins and the layer sharing.
* **Where it is not.** The dense scope (the v1, and every refresh pass) has none, nor
  ``frozen_features="current"``; a frozen row never gets it, so a frozen board's outputs and
  K/V do not move when its status does - the bias does.
* **Every path computes it.** B14's score function computes the hand-materialised bias bit for
  bit, and attention through it equals attention with the bias materialised; the CPU paths'
  outputs and gradients - the table's included - equal it too; the whole network agrees across
  every implementation. Section 15.2's row 1 (cached = dense) runs with B16 on in
  ``test_cache.py`` (every case with intrinsic frozen features, the default; its
  ``frozen_features="current"`` test arm has no B16), which also checks a moving present and,
  against an oracle of its own, B16 below a refreshed epoch's anchor and none at the anchor.
* **The table**: 304 parameters at the placeholders, from a generator of its own (switching it
  changes no other weight), reaching the outputs; translation and board order leave it alone.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, PositionState, build_model, collate, parameter_breakdown, permute_boards, synthetic_state, translate
from t5d.net.attention import attention_block_sparse, attention_chunked, attention_explicit, attention_sdpa
from t5d.net.flex import ScoreInputs, attention_flex, make_score_mod, materialise_score
from tests.net.test_displacement import CONSTRUCTIONS, reference_bias

#: A position whose frozen boards reach every Delta-t bin (exact below 4, then one per doubling
#: to 128, clamped beyond), past and future of the present (h 200), on active and inactive
#: timelines: ``timeline: (half-turns of its boards, active)``.
STATUS_SPEC = {
    0: ([0, 1, 2, 3, 4, 70, 130, 190, 196, 197, 198, 199, 200], True),
    1: ([150, 200, 201, 202, 203, 204, 205, 209, 218, 233, 265, 329, 460], True),
    -1: ([10, 61, 180, 199, 200, 201], False),
    2: ([300, 301], False),
}


def status_state(seed: int, size: int = 3) -> PositionState:
    """``STATUS_SPEC`` as a position (3 x 3 boards keep it small); contents, edges and candidate
    moves from a synthetic state of the same layout (tokens, not geometry)."""
    boards = [(p, h, a) for p, (hs, a) in STATUS_SPEC.items() for h in hs]
    timeline = np.array([p for p, _, _ in boards])
    half_turn = np.array([h for _, h, _ in boards])
    src = synthetic_state(seed, len(boards), width=size, height=size)
    return PositionState(
        width=size,
        height=size,
        timeline=timeline,
        half_turn=half_turn,
        is_head=np.array([h == half_turn[timeline == p].max() for p, h, _ in boards]),
        active=np.array([a for _, _, a in boards]),
        side_to_move=half_turn & 1,
        piece=src.piece,
        colour=src.colour,
        moved=src.moved,
        edges=src.edges,
        moves=src.moves,
        opp_moves=src.opp_moves,
    )


def delta_t_bin(dt: int, exact: int, maximum: int) -> int:
    """Section 4.1's "Delta-t-to-present bin" as built (a placeholder, spec issue 213), written out:
    ``|dt|`` exact below ``exact``, then one bin per doubling up to ``maximum``, the last shared by
    every larger offset; the future's bins first, then the past's, sharing bin 0."""
    size = abs(dt)
    doublings = int(math.floor(math.log2(maximum / exact)))
    magnitude = size if size < exact else exact + min(int(math.floor(math.log2(size / exact))), doublings)
    return magnitude if dt >= 0 else exact + doublings + magnitude


def status_keys(model, states) -> tuple[torch.Tensor, torch.Tensor]:
    """By hand, from the positions: each token's status entry ``(B, L)`` - ``bin(h - h_present) *
    2 + active`` for a frozen board's tokens, -1 for every other token - and the live query rows
    ``(B, L)`` - the head boards' tokens and the global registers."""
    cfg = model.cfg
    batch = collate(states)
    key = np.full((len(states), batch.n_tokens), -1)
    live = np.zeros((len(states), batch.n_tokens), dtype=bool)
    for b, state in enumerate(states):
        tpb = state.layout.tokens_per_board
        for board in range(state.n_boards):
            tokens = slice(board * tpb, (board + 1) * tpb)
            if state.is_head[board]:
                live[b, tokens] = True
            else:
                dt = int(state.half_turn[board]) - state.h_present
                key[b, tokens] = delta_t_bin(dt, cfg.status_dt_exact, cfg.status_dt_max) * 2 + int(state.active[board])
        live[b, batch.body_len : batch.body_len + state.global_regs] = True
    return torch.from_numpy(key), torch.from_numpy(live)


def status_reference(model, states, layer: int = 0) -> torch.Tensor:
    """B16's bias of layer ``layer`` materialised from ``status_keys`` ``(B, H, L, L)``: the table at
    a frozen board's token's status entry, on the live rows; zero elsewhere - differentiable in the
    table."""
    key, live = status_keys(model, states)
    table = model.status.table[layer if model.cfg.status_bias_per_layer else 0].float()
    padded = torch.cat([table, table.new_zeros(table.shape[0], 1)], dim=1)  # the last entry: no status
    values = padded[:, torch.where(key >= 0, key, table.shape[1])].transpose(0, 1)[:, :, None, :]  # (B, H, 1, L)
    return torch.where(live[:, None, :, None], values, values.new_zeros(()))


def _model(seed=0, layers=2, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=layers, attn_scope="block_causal", **flags), seed=seed)


#: Positions for the bias tests: every bin, and section 15.2's adversarial constructions.
POSITIONS = {
    "every_bin": lambda s: [status_state(s)],
    "every_bin_padded": lambda s: [synthetic_state(s, 4), status_state(s), synthetic_state(s + 1, 1)],
    **{name: CONSTRUCTIONS[name] for name in ("far", "many_timelines", "deep_travel", "duplicated", "even_start", "registers_4_0")},
}

#: The placeholder default and other settings of B16's unstated details (spec issues 213-214).
SETTINGS = {
    "default": dict(),
    "per_layer": dict(status_bias_per_layer=True),
    "coarse_bins": dict(status_dt_exact=2, status_dt_max=16),
    "exact_bins": dict(status_dt_exact=64, status_dt_max=64),
}


# --- the bias against its definition --------------------------------------------------------------


def test_the_bins_written_out():
    """The default bins: 19 - exact |Delta-t| 0..3, then 4-7, 8-15, 16-31, 32-63, 64-127, 128+,
    the future's (and the present's 0) first, then the past's."""
    got = [delta_t_bin(dt, 4, 128) for dt in (0, 1, 3, 4, 7, 8, 127, 128, 9000, -1, -3, -4, -127, -128, -9000)]
    assert got == [0, 1, 3, 4, 4, 5, 8, 9, 9, 10, 12, 13, 17, 18, 18]
    model = _model(layers=1)
    assert model.status.n_bins == 19 and model.status.n_status == 38
    dts = torch.arange(-300, 301)
    for flags in SETTINGS.values():
        status = _model(layers=1, **flags).status
        cfg = NetConfig(**flags)
        for active in (0, 1):
            want = torch.tensor([delta_t_bin(int(d), cfg.status_dt_exact, cfg.status_dt_max) * 2 + active for d in dts])
            assert torch.equal(status.status(dts.float(), torch.full_like(dts, active)), want), flags


@pytest.mark.parametrize("position", list(POSITIONS))
def test_the_rows_are_the_definition(position):
    """B16 alone (no B15, no masks): the structure's bias at every row - over every key, any
    block, a key selection on a head subset, arbitrary rows - is ``status_reference``, bit for
    bit, at two layers, under each setting of the placeholders."""
    states = POSITIONS[position](5)
    batch = collate(states)
    length = batch.n_tokens
    gen = torch.Generator().manual_seed(5)
    for name, flags in SETTINGS.items():
        model = _model(
            seed=5,
            displacement_bias=False,
            move_graph_heads=0,
            regs_per_board=batch.regs_per_board,
            global_regs=batch.global_regs,
            status_bias_init_std=1.0,
            **flags,
        )
        ctx = model.make_context(batch)
        for layer in (0, 1):
            st = ctx.for_layer(None, layer).structure
            ref = status_reference(model, states, layer)
            assert torch.equal(st.dense_bias(), ref), (name, layer)
            for start, stop in ((0, 7), (40, 200), (length - 20, length)):
                assert torch.equal(st.bias_rows(start, stop), ref[:, :, start:stop])
            cols = torch.stack([torch.randperm(length, generator=gen)[:40] for _ in range(batch.batch_size)])
            heads = torch.tensor([0, 3, 6])
            picked = st.bias_rows(10, 30, heads=heads, cols=cols, cols_valid=torch.ones_like(cols, dtype=torch.bool))
            assert torch.equal(picked, ref[:, heads, 10:30].gather(3, cols[:, None, None, :].expand(-1, 3, 20, -1)))
            rows = torch.stack([torch.randperm(length, generator=gen)[:25] for _ in range(batch.batch_size)])
            assert torch.equal(st.bias_rows_at(rows), ref.gather(2, rows[:, None, :, None].expand(-1, 8, -1, length)))
        if not flags.get("status_bias_per_layer"):
            assert torch.equal(ctx.structure.dense_bias(), status_reference(model, states)), "shared: no layer to choose"
    assert bool((ref != 0).any()), "the positions hold live rows reading frozen keys"


def test_the_rows_add_to_b15s():
    """With B15 (the v1 default): the bias is B15's, materialised by hand (test_displacement.py),
    plus B16's - every pair, bit for bit, the two added in that order."""
    states = [status_state(6), synthetic_state(7, 5)]
    batch = collate(states)
    model = _model(seed=6)
    for layer in (0, 1):
        bias = model.make_context(batch).for_layer(None, layer).structure.dense_bias()
        assert torch.equal(bias, reference_bias(model, batch, layer) + status_reference(model, states, layer)), layer


def test_where_there_is_none():
    """No table and no bias outside the cached configuration: the dense scope (the v1 - and every
    refresh pass of a block-causal network), ``frozen_features="current"`` (frozen boards carry
    their status as inputs), the switch off; and none for the one forward that leaves it out."""
    states = [status_state(8)]
    batch = collate(states)
    for flags in (
        dict(attn_scope="dense"),
        dict(attn_scope="block_causal", frozen_features="current"),
        dict(attn_scope="block_causal", status_key_bias=False),
    ):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, **flags), seed=8)
        assert model.status is None and parameter_breakdown(model)["attn.status"] == 0, flags
        assert model.make_context(batch, scope="block_causal", status_bias=False).structure.status is None, flags
        if flags["attn_scope"] == "block_causal":  # the dense scope's override: refused (below)
            assert model.make_context(batch, scope="block_causal").structure.status is None, flags
    model = _model(seed=8, layers=1)
    assert model.status is not None
    assert model.make_context(batch, scope="dense").structure.status is None, "a refresh pass is a dense pass"
    assert model.make_context(batch, status_bias=False).structure.status is None
    b15 = reference_bias(model, batch, 0)
    assert torch.equal(model.make_context(batch, status_bias=False).for_layer(None, 0).structure.dense_bias(), b15)


def test_a_dense_network_is_not_forwarded_block_causally_without_b16():
    """A network of the dense scope (the v1) has no B16 table, and a block-causal forward strips its
    frozen boards' status from their inputs (intrinsic frozen features): with B16 switched on that
    forward would carry the status nowhere - section 4.1's "current status enters as score-time key
    biases" (register A3: "instead (B16)") broken, silently. So it is refused - through the model,
    its attention probabilities and the dense path's evaluator alike - and allowed where it means
    what it says: with ``status_bias=False``, ``status_key_bias=False`` or
    ``frozen_features="current"`` (the status then stays in the inputs: shifting it moves the
    outputs)."""
    from t5d.engine.evaluator import NetworkEvaluator

    state = status_state(10)
    batch = collate([state])
    v1 = build_model(NetConfig(trunk_dtype="fp32", n_layers=1), seed=10).eval()
    assert v1.status is None
    for call in (
        lambda: v1(batch, scope="block_causal"),
        lambda: v1.attention_probabilities(batch, scope="block_causal"),
        lambda: NetworkEvaluator(v1, scope="block_causal")([state]),
    ):
        with pytest.raises(ValueError, match="no B16 table"):
            call()
    v1(batch, scope="block_causal", status_bias=False)  # says it means the network without B16
    v1(batch)  # its own scope, dense: every board carries its status as inputs
    frozen = (batch.is_head == 0) & (batch.board >= 0)
    shifted = replace(batch, dt=torch.where(frozen, batch.dt - 37.0, batch.dt), active=torch.where(frozen, 1 - batch.active, batch.active))
    for flags, moves in ((dict(status_key_bias=False), False), (dict(frozen_features="current"), True)):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, **flags), seed=10).eval()
        a, b = (model(x, scope="block_causal", return_trunk=True).trunk for x in (batch, shifted))
        assert (not torch.equal(a, b)) == moves, flags


def test_a_status_change_moves_the_bias_not_the_frozen_boards():
    """Section 4.1: frozen tokens carry intrinsic features only, and their status reaches the
    live rows at score time. Change only the frozen boards' Delta-t and active flags: every
    frozen token's K/V (every layer) and output stay bit for bit, the live tokens' outputs move -
    and with B16 off they do not (the live tokens' own inputs did not change)."""
    state = status_state(9)
    batch = collate([state])
    frozen = (batch.is_head == 0) & (batch.board >= 0)
    shifted = replace(batch, dt=torch.where(frozen, batch.dt - 37.0, batch.dt), active=torch.where(frozen, 1 - batch.active, batch.active))
    live = ~frozen & batch.valid
    for on in (True, False):
        model = _model(seed=9, status_key_bias=on).eval()
        captured = {}
        for name, b in (("base", batch), ("shifted", shifted)):
            kv = []
            out = model(b, return_trunk=True, kv_hook=lambda layer, k, v, kv=kv: kv.append((k, v)) or (k, v))
            captured[name] = (out.trunk[0], kv)
        (t0, kv0), (t1, kv1) = captured["base"], captured["shifted"]
        assert torch.equal(t0[frozen[0]], t1[frozen[0]]), "frozen outputs"
        for (k0, v0), (k1, v1) in zip(kv0, kv1, strict=True):
            assert torch.equal(k0[0][:, frozen[0]], k1[0][:, frozen[0]]) and torch.equal(v0[0][:, frozen[0]], v1[0][:, frozen[0]])
        assert (not torch.allclose(t0[live[0]], t1[live[0]], atol=1e-4)) == on, on


# --- every path -------------------------------------------------------------------------------------


def _b16_only(ctx, **changes):
    """The context with B16 taken out of its structure (``changes`` on top)."""
    return replace(ctx, structure=replace(ctx.structure, status=None, status_index=None, status_table=None), **changes)


def _qkv(batch, seed):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch.batch_size, 8, batch.n_tokens, 32)
    return tuple(torch.randn(*shape, generator=gen) for _ in range(3))


@pytest.mark.parametrize("position", list(POSITIONS))
def test_score_function_equals_the_bias_materialised(position):
    """B14's score function (``flex.py``) with B16 inside it: its term at every pair is B15's and
    B16's biases materialised by hand, bit for bit (B16 at its initial scale and at unit scale,
    per layer and shared), and attention through it equals attention with that bias passed as a
    dense one (measured under 1e-6, asserted 1e-5) - the block-causal scope in its mask."""
    settings = [(0.1, dict(), 0), (1.0, dict(status_bias_per_layer=True), 1), (1.0, dict(routing_table=True), 1)]
    for seed, (std, flags, layer) in enumerate(settings):
        states = POSITIONS[position](20 + seed)
        batch = collate(states)
        model = _model(seed=seed, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, status_bias_init_std=std, **flags)
        ctx = model.make_context(batch).for_layer(None, layer)
        ref = reference_bias(model, batch, layer) + status_reference(model, states, layer)
        term = materialise_score(make_score_mod(ScoreInputs.from_context(ctx, 8)), batch.batch_size, 8, batch.n_tokens, batch.n_tokens)
        assert torch.equal(term, ref), (seed, std, layer)
        q, k, v = _qkv(batch, seed)
        valid = batch.valid[:, None, :, None]
        through = attention_flex(q, k, v, ctx)
        bare = replace(ctx.structure, disp=None, disp_index=None, disp_table=None, status=None, status_index=None, status_table=None)
        materialised = attention_explicit(q, k, v, replace(ctx, structure=bare, bias=ref))
        assert torch.allclose(through * valid, materialised * valid, atol=1e-5), float(((through - materialised) * valid).abs().max())


CPU_PATHS = {
    "block_sparse": lambda q, k, v, ctx: attention_block_sparse(q, k, v, replace(ctx, block_rows=11, sparse_max_fraction=1.0)),
    "block_sparse_mixed": lambda q, k, v, ctx: attention_block_sparse(
        q, k, v, replace(ctx, block_rows=2 * 11 + 3, sparse_max_fraction=0.3)
    ),
    "chunked": lambda q, k, v, ctx: attention_chunked(q, k, v, replace(ctx, block_rows=11)),
    "sdpa": attention_sdpa,
    "explicit": lambda q, k, v, ctx: attention_explicit(q, k, v, replace(ctx, chunk=37)),
}


@pytest.mark.grad
@pytest.mark.parametrize(
    "flags",
    [dict(), dict(routing_table=True), dict(board_register_spatial="position_free"), dict(status_bias_per_layer=True)],
    ids=["b9", "routing", "regs-free", "per-layer"],
)
def test_cpu_paths_equal_the_bias_materialised(flags):
    """The CPU paths build B16's rows block by block (key selections, free rows): their outputs
    and gradients - Q, K, V and B16's table - equal attention with B16's bias materialised by hand
    (differentiable in the table) and passed as a dense one; two seeds, the table at unit scale."""
    for seed in (0, 1):
        states = [status_state(30 + seed), synthetic_state(32 + seed, 3, width=3, height=3)]
        batch = collate(states)
        model = _model(seed=seed, layers=1, status_bias_init_std=1.0, **flags)
        qkv = _qkv(batch, seed)
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(batch.batch_size, 8, batch.n_tokens, 32, generator=gen) * batch.valid[:, None, :, None]
        valid = batch.valid[:, None, :, None]

        def run(attend, materialised: bool, states=states, batch=batch, model=model, qkv=qkv, w=w):
            model.zero_grad(set_to_none=True)
            ctx = model.make_context(batch).for_layer(None, 0)
            if materialised:
                ctx = _b16_only(ctx, bias=status_reference(model, states, 0))
            leaves = [t.clone().requires_grad_(True) for t in qkv]
            out = attend(*leaves, ctx)
            (out * w).sum().backward()
            return out.detach(), [t.grad for t in leaves] + [model.status.table.grad.clone()]

        out_r, grads_r = run(attention_explicit, materialised=True)
        assert grads_r[-1].abs().sum() > 0, "the table receives gradient through the materialised bias"
        for name, attend in CPU_PATHS.items():
            out, grads = run(attend, materialised=False)
            assert torch.allclose(out * valid, out_r * valid, atol=1e-5), (name, float(((out - out_r) * valid).abs().max()))
            for i, (ga, gb) in enumerate(zip(grads_r, grads, strict=True)):
                tol = 1e-4 + 1e-5 * float(ga.abs().max())
                assert torch.allclose(ga, gb, atol=tol, rtol=0.0), (name, i, float((ga - gb).abs().max()))


@pytest.mark.parametrize(
    "flags",
    [
        dict(),
        dict(typed_edge_bias=True, board_pair_smolgen=True, routing_table=True),
        dict(status_bias_per_layer=True, status_bias_init_std=1.0),
    ],
    ids=["v1-causal", "n2-all-causal", "per-layer-unit"],
)
def test_every_implementation_is_one_function(flags):
    """The whole block-causal network with B16, through every attention implementation - the
    score function among them - against the dense reference."""
    batch = collate([status_state(40), synthetic_state(41, 6, n_timelines=3)])
    outs = {
        impl: _model(seed=7, attn_impl=impl, **flags).eval()(batch, return_trunk=True)
        for impl in ("dense", "explicit", "sdpa", "chunked", "block_sparse", "flex")
    }
    ref = outs["dense"]
    valid = batch.valid[..., None]
    for impl, out in outs.items():
        assert torch.allclose(out.trunk * valid, ref.trunk * valid, atol=2e-5), (impl, float(((out.trunk - ref.trunk) * valid).abs().max()))
        assert torch.allclose(out.value_logits, ref.value_logits, atol=1e-4), impl


# --- the table ------------------------------------------------------------------------------------


def test_the_tables_size_and_its_own_generator():
    """304 parameters at the placeholders (8 heads x 19 bins x 2 statuses; 4,864 with one table per
    layer, 16 x 304), added only to a network of the cached configuration; drawn from B16's own
    generator, so the network without it has every other weight bit for bit, no flag changes it,
    the seed does."""
    on, off = _model(seed=11, layers=3), _model(seed=11, layers=3, status_key_bias=False)
    a, b = parameter_breakdown(on), parameter_breakdown(off)
    assert a["attn.status"] == 8 * 19 * 2 == 304 and a["total"] - b["total"] == 304 and b["attn.status"] == 0
    assert parameter_breakdown(_model(seed=11, layers=3, status_bias_per_layer=True))["attn.status"] == 3 * 304
    p_on, p_off = dict(on.named_parameters()), dict(off.named_parameters())
    assert set(p_on) - set(p_off) == {"status.table"}
    assert all(torch.equal(p, p_on[name]) for name, p in p_off.items())
    base = on.status.table
    for flags in (dict(typed_edge_bias=True, board_pair_smolgen=True), dict(displacement_bias=False), dict(norm_placement="post")):
        assert torch.equal(_model(seed=11, layers=3, **flags).status.table, base), flags
    assert not torch.equal(_model(seed=12, layers=3).status.table, base)
    assert abs(float(build_model(NetConfig(attn_scope="block_causal"), seed=3).status.table.std()) - 0.1) < 0.02, "N(0, 0.1)"


def test_the_table_reaches_the_outputs():
    """Perturbing B16's table moves the live tokens' outputs and the value head (it pools the
    global registers, which are live rows), and never a frozen token's output."""
    model = _model(seed=12).eval()
    batch = collate([status_state(12)])
    frozen = ((batch.is_head == 0) & (batch.board >= 0))[0]
    base = model(batch, return_trunk=True)
    with torch.no_grad():
        model.status.table.add_(torch.randn(model.status.table.shape, generator=torch.Generator().manual_seed(12)))
    out = model(batch, return_trunk=True)
    assert torch.equal(out.trunk[0][frozen], base.trunk[0][frozen])
    assert not torch.allclose(out.trunk[0][~frozen & batch.valid[0]], base.trunk[0][~frozen & batch.valid[0]], atol=1e-4)
    assert not torch.allclose(out.value_logits, base.value_logits, atol=1e-5)


def test_the_bin_placeholders_regroup_the_statuses():
    """Each bin setting groups the frozen boards' Delta-t differently on ``status_state`` (which
    pairs share an entry), and the table's size follows: a placeholder that changed nothing would
    be an inert switch."""
    batch = collate([status_state(13)])

    def keys(**flags):
        return _model(seed=13, layers=1, **flags).make_context(batch).structure.status_index.key

    def same(a, b):
        joint = torch.unique(torch.stack([a.reshape(-1), b.reshape(-1)]), dim=1).shape[1]
        return joint == torch.unique(a).numel() == torch.unique(b).numel()

    default = keys()
    for flags in (dict(status_dt_exact=2), dict(status_dt_max=16), dict(status_dt_exact=64, status_dt_max=64)):
        assert not same(default, keys(**flags)), flags
    with pytest.raises(ValueError, match="status_dt"):
        NetConfig(status_dt_exact=0).validate()
    with pytest.raises(ValueError, match="status_dt"):
        NetConfig(status_dt_exact=8, status_dt_max=4).validate()
    with pytest.raises(ValueError, match="status_bias_init_std"):
        NetConfig(status_bias_init_std=float("nan")).validate()


def test_translation_and_board_order():
    """Delta-t-to-present is relative to the position's own present: translating every timeline
    and every board by +2j half-turns (the present with them) leaves the outputs as they were, and
    storing the boards in another order permutes them."""
    model = _model(seed=14).eval()
    state = synthetic_state(14, 12, n_timelines=4)
    a = model(collate([state]), return_trunk=True)
    for dp, dh in ((3, 8), (-7, 100)):
        b = model(collate([translate(state, dp=dp, dh=dh)]), return_trunk=True)
        assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.board_logits, b.board_logits), (dp, dh)
    order = np.random.default_rng(14).permutation(state.n_boards)
    perm = torch.from_numpy(np.concatenate([np.arange(66) + 66 * int(o) for o in order] + [66 * state.n_boards + np.arange(16)]))
    c = model(collate([permute_boards(state, order)]), return_trunk=True)
    assert torch.allclose(c.trunk[0], a.trunk[0][perm], atol=2e-5) and torch.allclose(c.value_logits, a.value_logits, atol=1e-5)


def test_the_status_index():
    """The per-token index: a frozen board's tokens - squares and per-board registers - carry its
    status; every other token (head boards', global registers, padding) the table's last, zero
    entry; the live rows are the head boards' tokens and the global registers."""
    states = [status_state(15), synthetic_state(16, 1, width=3, height=3)]
    batch = collate(states)
    model = _model(seed=15, layers=1)
    idx = model.make_context(batch).structure.status_index
    none = model.status.n_status
    for b, state in enumerate(states):
        tpb = state.layout.tokens_per_board
        for board, t in itertools.product(range(state.n_boards), range(tpb)):
            token = board * tpb + t
            if state.is_head[board]:
                assert int(idx.key[b, token]) == none and bool(idx.live[b, token])
            else:
                dt = int(state.half_turn[board]) - state.h_present
                assert int(idx.key[b, token]) == delta_t_bin(dt, 4, 128) * 2 + int(state.active[board])
                assert not bool(idx.live[b, token])
        glob = slice(batch.body_len, batch.body_len + state.global_regs)
        assert bool(idx.live[b, glob].all()) and bool((idx.key[b, glob] == none).all())
    pad = ~batch.valid
    assert bool((idx.key[pad] == none).all()) and not bool(idx.live[pad].any())
    assert idx.any_frozen
    alone = collate([synthetic_state(17, 1)])
    assert not model.make_context(alone).structure.status_index.any_frozen, "every board a head: nothing to bias"
