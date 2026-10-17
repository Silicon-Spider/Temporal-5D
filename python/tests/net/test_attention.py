"""Attention core: the pair rule against a brute-force reference, the SDPA path equal to
the explicit one in every pair-rule mode, masks and biases, and the B5/B15 wiring."""

from __future__ import annotations

import math

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net import attention as attention_module
from t5d.net.attention import AttnContext, attention_explicit, attention_sdpa
from t5d.net.rope import AXIS_FILE, AXIS_RANK, head_allocation, rope_angles, rope_cos_sin
from t5d.net.state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER

#: The open committed-class switches that change which pairs are rotated (NetConfig):
#: (global_register_pairs, board_register_spatial).
PAIR_MODES = {
    "default": ("position_free", "learned_angle"),
    "anchored": ("anchored", "learned_angle"),
    "regs_free": ("position_free", "position_free"),
    "regs_free_anchored": ("anchored", "position_free"),
    "zero_angle": ("position_free", "zero_angle"),
}


def _config(mode: str, **extra) -> NetConfig:
    pairs, spatial = PAIR_MODES[mode]
    return NetConfig(trunk_dtype="fp32", global_register_pairs=pairs, board_register_spatial=spatial, **extra)


def _setup(mode="default", n_boards=(4, 2), seed=0, **extra):
    model = build_model(_config(mode, **extra), seed=0)
    batch = collate([synthetic_state(seed + i, n) for i, n in enumerate(n_boards)])
    gen = torch.Generator().manual_seed(seed)
    shape = (batch.batch_size, 8, batch.n_tokens, 32)
    q, k, v = (torch.randn(*shape, generator=gen) for _ in range(3))
    with torch.no_grad():
        coords = model.coordinates(batch)
    return model, batch, coords, q, k, v, gen


def _ctx(model, batch, coords, **kw) -> AttnContext:
    cfg = model.cfg
    cos, sin = rope_cos_sin(rope_angles(coords, model.rope_axis, model.rope_freq))
    spatial_free = batch.valid & (batch.kind == TOKEN_BOARD_REGISTER) if cfg.board_register_spatial == "position_free" else None
    return AttnContext(
        cos=cos,
        sin=sin,
        body_len=batch.body_len,
        key_valid=batch.valid,
        global_free=cfg.global_register_pairs == "position_free",
        spatial_free=spatial_free,
        spatial_dims=model.rope_spatial,
        **kw,
    )


def _has_axes(cfg: NetConfig, batch) -> torch.Tensor:
    """(B, L, 4) bool: which axes each token has a coordinate on, by the config."""
    has = torch.ones(*batch.kind.shape, 4, dtype=torch.bool)
    if cfg.global_register_pairs == "position_free":
        has[batch.kind == TOKEN_GLOBAL_REGISTER] = False
    if cfg.board_register_spatial == "position_free":
        breg = batch.kind == TOKEN_BOARD_REGISTER
        has[..., AXIS_FILE] &= ~breg
        has[..., AXIS_RANK] &= ~breg
    return has


def _reference_probs(q, k, coords, has, axis, freq, allowed, bias=None) -> torch.Tensor:
    """Brute force in FP64, pair by pair: rotary pair m of a head turns by the coordinate
    difference on its axis when *both* tokens have that axis, and by nothing otherwise."""
    q, k, coords = q.double(), k.double(), coords.double()
    bsz, length = coords.shape[:2]
    hr, n_pairs = axis.shape
    idx = axis[None, :, None, :].expand(bsz, hr, length, n_pairs)
    theta = coords[:, None].expand(bsz, hr, length, 4).gather(-1, idx) * freq.double()[None, :, None, :]
    has_m = has[:, None].expand(bsz, hr, length, 4).gather(-1, idx)
    delta = theta[:, :, None, :, :] - theta[:, :, :, None, :]  # key minus query
    delta = torch.where(has_m[:, :, :, None, :] & has_m[:, :, None, :, :], delta, torch.zeros_like(delta))
    qa, qb = q[..., :n_pairs][:, :, :, None, :], q[..., n_pairs:][:, :, :, None, :]
    ka, kb = k[..., :n_pairs][:, :, None, :, :], k[..., n_pairs:][:, :, None, :, :]
    scores = (torch.cos(delta) * (qa * ka + qb * kb) + torch.sin(delta) * (qb * ka - qa * kb)).sum(-1) / math.sqrt(q.shape[-1])
    if bias is not None:
        scores = scores + bias.double()
    scores = scores.masked_fill(~allowed, float("-inf"))
    has_any = allowed.any(-1, keepdim=True)
    probs = torch.softmax(torch.where(has_any, scores, torch.zeros_like(scores)), dim=-1)
    return torch.where(has_any, probs, torch.zeros_like(probs))


def _close_on_valid(a, b, batch, atol):
    valid = batch.valid[:, None, :, None]
    assert torch.allclose(a * valid, b * valid, atol=atol), float(((a - b) * valid).abs().max())


def _adversarial_mask(batch, gen, density=0.3):
    """A random mask with the hard rows: nothing allowed, only a global key, no global
    key, a per-board register with nothing allowed, a global query with nothing allowed."""
    big_l = batch.n_tokens
    mask = torch.rand(batch.batch_size, 8, big_l, big_l, generator=gen) < density
    mask[:, :, 5, :] = False
    mask[:, :, 7, :] = False
    mask[:, :, 7, batch.body_len + 3] = True
    mask[:, :, 9, batch.body_len :] = False
    mask[:, :, 64, :] = False
    mask[:, :, -1, :] = False
    return mask


@pytest.mark.parametrize("mode", list(PAIR_MODES))
def test_pair_rule_matches_brute_force(mode):
    """The explicit path computes the pair rule: an axis is rotated iff both tokens have it."""
    model, batch, coords, q, k, v, gen = _setup(mode, n_boards=(2, 1))
    mask = _adversarial_mask(batch, gen)
    bias = torch.randn(batch.batch_size, 8, batch.n_tokens, batch.n_tokens, generator=gen)
    ctx = _ctx(model, batch, coords, mask=mask, bias=bias)
    _, probs = attention_explicit(q, k, v, ctx, return_probs=True)
    allowed = batch.valid[:, None, None, :] & mask
    ref = _reference_probs(q, k, coords, _has_axes(model.cfg, batch), model.rope_axis, model.rope_freq, allowed, bias)
    assert torch.allclose(probs.double(), ref, atol=1e-5), float((probs.double() - ref).abs().max())


@pytest.mark.parametrize("mode", list(PAIR_MODES))
def test_sdpa_equals_explicit_padding_only(mode):
    model, batch, coords, q, k, v, _ = _setup(mode)
    ctx = _ctx(model, batch, coords)
    _close_on_valid(attention_sdpa(q, k, v, ctx), attention_explicit(q, k, v, ctx), batch, 1e-5)


@pytest.mark.parametrize("mode", list(PAIR_MODES))
def test_sdpa_equals_explicit_with_mask_bias_and_empty_rows(mode):
    model, batch, coords, q, k, v, gen = _setup(mode)
    mask = _adversarial_mask(batch, gen)
    bias = torch.randn(batch.batch_size, 8, batch.n_tokens, batch.n_tokens, generator=gen)
    for kw in (dict(mask=mask), dict(bias=bias), dict(mask=mask, bias=bias)):
        ctx = _ctx(model, batch, coords, **kw)
        a, b = attention_sdpa(q, k, v, ctx), attention_explicit(q, k, v, ctx)
        _close_on_valid(a, b, batch, 1e-5)
        if "mask" in kw:
            for row in (5, 64, -1):
                assert torch.all(a[:, :, row] == 0), f"row {row}: no allowed key gives zeros"
            assert torch.allclose(a[:, :, 7], v[:, :, batch.body_len + 3], atol=1e-6)


@pytest.mark.parametrize("mode", ["default", "regs_free"])
def test_sdpa_merge_is_stable_at_large_logits(mode):
    model, batch, coords, q, k, v, _ = _setup(mode, n_boards=(3,))
    ctx = _ctx(model, batch, coords)
    q, k = q * 12.0, k * 12.0  # logits of order +-100
    a, b = attention_sdpa(q, k, v, ctx), attention_explicit(q, k, v, ctx)
    assert torch.isfinite(a).all()
    _close_on_valid(a, b, batch, 1e-4)


@pytest.mark.parametrize("mode", ["default", "regs_free"])
def test_explicit_chunking_is_exact(mode):
    model, batch, coords, q, k, v, _ = _setup(mode, n_boards=(3, 1))
    whole = attention_explicit(q, k, v, _ctx(model, batch, coords, chunk=10_000))
    for chunk in (1, 37, 64):
        part = attention_explicit(q, k, v, _ctx(model, batch, coords, chunk=chunk))
        assert torch.allclose(whole, part, atol=1e-6)


@pytest.mark.parametrize("mode", list(PAIR_MODES))
def test_padded_keys_receive_no_attention(mode):
    model, batch, coords, q, k, v, _ = _setup(mode, n_boards=(1, 3))
    ctx = _ctx(model, batch, coords)
    v2 = v.clone()
    pad = ~batch.valid
    v2[pad[:, None, :, None].expand_as(v2)] = 1e6  # poison the padding values
    for attend in (attention_sdpa, attention_explicit):
        _close_on_valid(attend(q, k, v, ctx), attend(q, k, v2, ctx), batch, 1e-4)


def test_pair_modes_change_the_function():
    """Each switch changes what attention computes (none of them is inert)."""
    outs = {}
    for mode in ("default", "anchored", "regs_free", "zero_angle"):
        model, batch, coords, q, k, v, _ = _setup(mode, n_boards=(3,))
        outs[mode] = attention_sdpa(q, k, v, _ctx(model, batch, coords))
    for mode in ("anchored", "regs_free", "zero_angle"):
        assert not torch.allclose(outs[mode], outs["default"], atol=1e-3), mode


@pytest.mark.parametrize(
    "flags",
    [
        dict(),
        dict(rope_per_head=True),
        dict(policy_promotion="offset", board_register_spatial="zero_angle"),
        dict(global_register_pairs="anchored"),
        dict(board_register_spatial="position_free"),
        dict(global_register_pairs="anchored", board_register_spatial="position_free", rope_per_head=True),
        dict(displacement_bias=False, move_graph_heads=0),
    ],
    ids=["default", "b5", "offset_zero_angle", "anchored", "regs_free", "anchored_regs_free_b5", "no_table_n1"],
)
def test_model_paths_agree(flags):
    """The whole network gives the same outputs on either attention path."""
    batch = collate([synthetic_state(3, 5), synthetic_state(4, 2)])
    outs = []
    for impl in ("sdpa", "explicit"):
        model = build_model(NetConfig(trunk_dtype="fp32", attn_impl=impl, **flags), seed=7).eval()
        outs.append(model(batch))
    a, b = outs
    assert torch.allclose(a.value_logits, b.value_logits, atol=1e-4)
    assert torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=1e-4)
    assert torch.allclose(a.board_logits, b.board_logits, atol=1e-4)
    assert torch.allclose(a.wdl_logits, b.wdl_logits, atol=1e-4)


@pytest.mark.grad
@pytest.mark.parametrize("mode", list(PAIR_MODES))
def test_gradients_agree_and_stay_finite_with_empty_rows(mode):
    """Training goes through every fast-path piece: same gradients as the reference, no
    NaN even when a query has no allowed global key or no allowed key at all."""
    model, batch, coords, q, k, v, gen = _setup(mode, n_boards=(2, 1))
    mask = _adversarial_mask(batch, gen, density=0.4)
    bias = torch.randn(batch.batch_size, 8, batch.n_tokens, batch.n_tokens, generator=gen)
    w = torch.randn(batch.batch_size, 8, batch.n_tokens, 32, generator=gen) * batch.valid[:, None, :, None]
    with torch.no_grad():
        ctxs = [_ctx(model, batch, coords, mask=mask, bias=bias), _ctx(model, batch, coords)]
    for ctx in ctxs:
        grads = []
        for attend in (attention_sdpa, attention_explicit):
            leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
            out = attend(*leaves, ctx)
            (out * w).sum().backward()
            grads.append([t.grad for t in leaves])
        for ga, gb in zip(*grads, strict=True):
            assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
            assert torch.allclose(ga, gb, atol=1e-4), float((ga - gb).abs().max())


def _kernel_with_nan_rows(q, k, v, attn_mask=None, scale=None):
    """Scaled dot-product attention as kernels before PyTorch 2.5 computed it: a fully
    masked row comes out NaN (PyTorch 2.13's CPU kernels return zeros instead)."""
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask.float()
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


@pytest.mark.grad
@pytest.mark.parametrize("mode", ["default", "anchored", "regs_free"])
def test_empty_rows_do_not_rely_on_the_kernel(mode, monkeypatch):
    """The fast path never hands the kernel a fully masked row, so it stays exact and
    NaN-free on kernels that return NaN there (the dependency pins no torch version)."""
    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", _kernel_with_nan_rows)
    model, batch, coords, q, k, v, gen = _setup(mode, n_boards=(2, 1))
    mask = _adversarial_mask(batch, gen)
    with torch.no_grad():
        ctx = _ctx(model, batch, coords, mask=mask)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = attention_sdpa(*leaves, ctx)
    out.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in leaves)
    _close_on_valid(out.detach(), attention_explicit(q, k, v, ctx), batch, 1e-5)


# --- B5 and B15 reach the computation (register B5 provisional, B15 the v1 default) -------------


def _layer0_qkv(model, batch):
    block = model.trunk.blocks[0]
    return block.attn.qkv(block.norm_attn(model.embed(batch)))


@pytest.mark.parametrize("mode", ["default", "regs_free_anchored"])
def test_per_head_roles_and_the_displacement_bias_reach_attention(mode):
    """B5: every head rotates by its own role's allocation (section 6's routing table);
    B15: layer 0's displacement table, materialised by hand (test_displacement.py), is added to
    the scores. Checked against layer 0's probabilities, rebuilt head by head from
    ``head_allocation``. (B9's move-graph mask is off here so that every head's reference is
    the unmasked one; the masks, and B5 with them, are checked in test_structure.py.)"""
    from tests.net.test_displacement import reference_bias

    cfg = _config(mode, rope_per_head=True, move_graph_heads=0)
    model = build_model(cfg, seed=3).eval()
    batch = collate([synthetic_state(9, 2)])
    ctx = model.make_context(batch)
    assert ctx.cos.shape[1] == cfg.n_heads, "B5: one rotation table per head"
    coords = model.coordinates(batch)
    bias = reference_bias(model, batch, 0)
    assert torch.equal(ctx.for_layer(None, 0).structure.dense_bias(), bias), "B15 wired into the context (its structure, per layer)"
    probs = model.attention_probabilities(batch, layer=0)
    q, k, _ = _layer0_qkv(model, batch)
    allowed = batch.valid[:, None, None, :].expand(1, 1, batch.n_tokens, batch.n_tokens)
    has = _has_axes(cfg, batch)
    for head, role in enumerate(cfg.rope_head_roles):
        alloc = head_allocation(role, cfg)
        axis = torch.tensor([[a for a, _ in alloc]])
        freq = torch.tensor([[f for _, f in alloc]], dtype=torch.float32)
        ref = _reference_probs(q[:, head : head + 1], k[:, head : head + 1], coords, has, axis, freq, allowed, bias[:, head : head + 1])
        assert torch.allclose(probs[:, head : head + 1].double(), ref, atol=1e-5), (head, role)
    # Not vacuous: without B5 (the dense split everywhere) the routed heads differ, and so do
    # the heads without B15's bias.
    dense = head_allocation("dense", cfg)
    axis = torch.tensor([[a for a, _ in dense]])
    freq = torch.tensor([[f for _, f in dense]], dtype=torch.float32)
    for head in (cfg.rope_head_roles.index("same_board"), cfg.rope_head_roles.index("move_graph")):
        ref = _reference_probs(q[:, head : head + 1], k[:, head : head + 1], coords, has, axis, freq, allowed, bias[:, head : head + 1])
        assert not torch.allclose(probs[:, head : head + 1].double(), ref, atol=1e-3), head
    head = cfg.rope_head_roles.index("dense")
    alloc = head_allocation("dense", cfg)
    axis = torch.tensor([[a for a, _ in alloc]])
    freq = torch.tensor([[f for _, f in alloc]], dtype=torch.float32)
    unbiased = _reference_probs(q[:, head : head + 1], k[:, head : head + 1], coords, has, axis, freq, allowed)
    assert not torch.allclose(probs[:, head : head + 1].double(), unbiased, atol=1e-3)
