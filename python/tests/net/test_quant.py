"""G2's FP8 E4M3 emulation (``quant.py``), C15's straight-through fix (``kvformat.py``) and the FP8
trunk (``NetConfig.trunk_quant``; spec section 10.2, register G2, G8, C15, G1).

The element format is checked against a decode of all 256 E4M3 codes written here from OCP's
definition (1 sign, 4 exponent bits at bias 7, 3 mantissa bits; exponent 0 subnormal; S.1111.111
NaN; no infinities), and the rounding against a nearest-even search over that grid: at every
midpoint between neighbours, just either side of it, beyond the range, and on random values.

Network tolerances, measured (CPU, torch 2.13, 2026-10-23) and the bounds set 3-4x above:

* the FP8 network against the same weights unquantised (F1, 16 layers, seeds 0, 5, 11,
  ``synthetic_state(30 + N, N)`` for N = 2, 17; BF16 and FP32 trunks; attention quantised or
  not): trunk RMS 2.7-3.2e-2 (max 0.16), value logits 5.0e-2, policy logits 0.14, value mean
  6.4e-3 (W - L) - about 12x the BF16-vs-FP32 gap (``test_precision.py``). Bounds 0.1, 0.6, 0.2, 0.5,
  0.025, and a floor (trunk RMS above 1e-2: it rounded).
* the attention implementations against the dense reference under the FP8 trunk (FP32, 2 layers,
  seeds 0-5, the v1, B17 on, the routing table on): explicit and flex bitwise; sdpa, chunked and
  block-sparse trunk RMS 2.1e-3 (max 4.7e-2), value 2.0e-3, policy 1.5e-2. Not FP32's 1e-5: an
  FP32 summation-order difference that tips a value across an FP8 midpoint moves it a whole FP8
  step (2^-3 relative), so the paths agree up to rare rounding flips. A fast path that left its
  operands unquantised (a mutant) gives RMS 1.9e-2, value 1.7e-2, policy 9.2e-2. Bounds 6e-3,
  8e-3, 5e-2.
* the cached path against the full pass with the projections alone in FP8 (FP32, 2 layers, model
  seeds 0-2, ``test_cache.py``'s standard trees, seeds 0-3, five moves): trunk RMS 1.9e-3 (max
  5.6e-2), value 2.2e-3, policy 1.8e-2 - flips again (unquantised, 1.8e-7). With attention
  quantised too (the combination the cache refuses, forced through) the gap is RMS 1.6e-2, value
  6.0e-3, policy 5.3e-2: post-RoPE rounding in the frozen board's own frame. Bounds 6e-3, 8e-3,
  6e-2.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net.attention import AttnContext
from t5d.net.cache import CachedEvaluator, reference_output
from t5d.net.kvformat import dequantise_fp8, fake_quantise, quantise_fp8
from t5d.net.quant import (
    E4M3,
    E4M3_MAX,
    AttnQuant,
    QuantLinear,
    fake_quant,
    quant_linear,
    quantise,
    round_e4m3,
    scale_of,
)
from t5d.net.tree import SearchTree
from tests.net.test_cache import _random_moves, _root

# --- the element format: an independent decode, from OCP's definition -------------------------


def ocp_e4m3(code: int) -> float:
    """The value of E4M3 code ``code`` (0..255), from the OCP 8-bit floating point
    specification's E4M3: sign, exponent (bias 7), mantissa; exponent 0 subnormal; S.1111.111
    NaN; no infinities."""
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0xF, code & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return math.nan
    if exponent == 0:
        return sign * (mantissa / 8) * 2.0**-6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


#: The non-negative finite values in code order (codes 0..126: increasing), and their codes.
GRID = np.array([ocp_e4m3(c) for c in range(127)])


def ref_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even onto GRID by search, saturating at 448, NaN kept (float64)."""
    out = np.empty_like(x)
    a = np.abs(x)
    for i, (v, m) in enumerate(zip(x, a, strict=True)):
        if math.isnan(v):
            out[i] = math.nan
            continue
        if m >= E4M3_MAX:
            r = E4M3_MAX
        else:
            hi = int(np.searchsorted(GRID, m, side="right"))  # GRID[hi - 1] <= m < GRID[hi]
            lo_v, hi_v = GRID[hi - 1], GRID[hi]
            if m - lo_v < hi_v - m:
                r = lo_v
            elif m - lo_v > hi_v - m:
                r = hi_v
            else:
                r = lo_v if (hi - 1) % 2 == 0 else hi_v  # a tie: the even code
        out[i] = math.copysign(r, v)
    return out


def test_the_256_codes_decode_as_ocp_defines_them():
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    torch_values = codes.view(E4M3).float()
    for c in range(256):
        expected, got = ocp_e4m3(c), float(torch_values[c])
        if math.isnan(expected):
            assert math.isnan(got), c
        else:
            assert got == expected and math.copysign(1, got) == math.copysign(1, expected), (c, got, expected)
    assert GRID[-1] == E4M3_MAX == 448 and GRID[1] == 2.0**-9 and GRID[8] == 2.0**-6
    assert np.all(np.diff(GRID) > 0), "codes 0..126 are increasing"
    assert sum(math.isnan(ocp_e4m3(c)) for c in range(256)) == 2  # 0x7F and 0xFF


def test_every_code_is_a_fixed_point():
    values = torch.tensor([ocp_e4m3(c) for c in range(256)], dtype=torch.float32)
    rounded = round_e4m3(values)
    finite = ~torch.isnan(values)
    assert torch.equal(rounded[finite], values[finite])
    assert torch.isnan(rounded[~finite]).all()
    # Signed zero kept (code 0x80 is -0).
    assert math.copysign(1, float(round_e4m3(torch.tensor([-0.0]))[0])) == -1


def test_rounding_at_every_midpoint_ties_to_even():
    lo, hi = GRID[:-1], GRID[1:]
    mid = (lo + hi) / 2  # exact in float64 and in FP32 (at most 5 significant bits)
    even = np.where(np.arange(126) % 2 == 0, lo, hi)
    mid32 = torch.tensor(mid, dtype=torch.float32)
    assert np.array_equal(mid32.double().numpy(), mid), "midpoints are FP32 values"
    for sign in (1.0, -1.0):
        got = round_e4m3(sign * mid32).double().numpy()
        assert np.array_equal(got, sign * even), "ties go to the even code"
        below = torch.nextafter(sign * mid32, torch.zeros_like(mid32))
        above = torch.nextafter(sign * mid32, torch.full_like(mid32, sign * math.inf))
        assert np.array_equal(round_e4m3(below).double().numpy(), sign * lo), "just inside a midpoint: the lower"
        assert np.array_equal(round_e4m3(above).double().numpy(), sign * hi), "just past a midpoint: the upper"
    # The subnormal edge: half the smallest subnormal ties to zero, anything above it rounds up.
    tiny = torch.tensor([2.0**-10, 2.0**-10 * 1.0001, 2.0**-11, 1e-30])
    assert round_e4m3(tiny).tolist() == [0.0, 2.0**-9, 0.0, 0.0]


def test_saturation_and_nan():
    x = torch.tensor([448.0, 448.5, 463.9, 464.0, 480.0, 500.0, 1e6, math.inf, 3.4e38])
    assert torch.equal(round_e4m3(x), torch.full_like(x, 448.0))
    assert torch.equal(round_e4m3(-x), torch.full_like(x, -448.0))
    assert torch.isnan(round_e4m3(torch.tensor([math.nan])))[0]


def test_random_values_round_as_the_reference():
    gen = np.random.default_rng(0)
    n = 20_000
    mag = np.exp2(gen.uniform(-14, 10, size=n))
    x = (mag * gen.choice([-1.0, 1.0], size=n)).astype(np.float32)
    got = round_e4m3(torch.from_numpy(x)).double().numpy()
    assert np.array_equal(got, ref_round(x.astype(np.float64)))


# --- scales and the straight-through estimator -------------------------------------------------


def test_scales():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(5, 7, 33, generator=gen) * torch.logspace(-6, 3, 7)[None, :, None]
    amax = x.abs().amax(-1, keepdim=True)
    s = scale_of(x, -1, "pow2")
    assert s.shape == (5, 7, 1)
    assert torch.equal(torch.exp2(torch.log2(s).round()), s), "powers of two"
    assert (s >= amax / E4M3_MAX).all() and (s < 2 * amax / E4M3_MAX).all()
    assert torch.equal(scale_of(x, -1, "absmax"), amax / E4M3_MAX)
    for rule in ("pow2", "absmax"):
        y, s = fake_quant(x, -1, rule), scale_of(x, -1, rule)
        on_grid = round_e4m3(x / s)
        assert torch.equal(y, on_grid * s), "every value is a code times its scale"
        if rule == "pow2":
            assert torch.equal(y / s, on_grid), "exactly: the scale is a power of two"
        assert (on_grid.abs() <= E4M3_MAX).all(), "no slice overflows its scale"
        if rule == "absmax":
            assert torch.equal(on_grid.abs().amax(-1), torch.full((5, 7), E4M3_MAX)), "the absmax lands on 448"
    assert torch.equal(fake_quant(torch.zeros(3, 4)), torch.zeros(3, 4)), "a zero slice gives zeros"
    with pytest.raises(ValueError, match="scale rule"):
        scale_of(x, -1, "per_batch")


def test_pow2_values_are_exact_in_bf16():
    """Under the "pow2" rule every dequantised value is a code times a power of two - exact in
    BF16 - so a BF16 kernel reads exactly what FP8 holds."""
    x = torch.randn(64, 256, generator=torch.Generator().manual_seed(1)) * 3
    y = fake_quant(x, -1, "pow2")
    assert torch.equal(y.bfloat16().float(), y)
    yb = fake_quant(x.bfloat16(), -1, "pow2")
    assert yb.dtype == torch.bfloat16 and torch.equal(yb.float(), fake_quant(x.bfloat16().float(), -1, "pow2"))


def test_per_token_scales_make_a_row_independent_of_its_tensor():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(3, 10, 16, generator=gen)
    other = x.clone()
    other[1:] *= 1000.0  # other positions, other tokens of the batch
    other[0, 5:] *= 1e-3
    assert torch.equal(fake_quant(x)[0, :5], fake_quant(other)[0, :5])


@pytest.mark.grad
@pytest.mark.parametrize("rule", ["pow2", "absmax"])
def test_the_gradient_goes_straight_through(rule):
    """The STE: the incoming gradient unchanged, however small - autograd through the cast
    rounded it onto the unscaled E4M3 grid (1e-4 became 0)."""
    x = (torch.randn(4, 32, generator=torch.Generator().manual_seed(3)) * 0.1).requires_grad_()
    upstream = torch.tensor([1e-4, 0.3, 1000.0, -7e-9]).repeat_interleave(8).repeat(4, 1)
    fake_quant(x, -1, rule).backward(upstream)
    assert torch.equal(x.grad, upstream)


@pytest.mark.grad
def test_c15_frozen_kv_rounding_is_trainable():
    """C15's FP8 K/V rounding (``kvformat.fake_quantise``): the forward is the cache's values
    bit for bit, and the gradient is the incoming one - the defect (BUILD_LOG C15 row: with |x|
    about 0.1, 63 of 64 gradients were zero) would fail here."""
    x = (torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)) * 0.1).requires_grad_()
    y = fake_quantise(x, "fp8")
    data, scale = quantise_fp8(x.detach())
    assert data.dtype == E4M3 and scale.shape == (2, 3, 1, 1)
    assert torch.equal(y.detach(), dequantise_fp8(data, scale, x.dtype))
    upstream = torch.full_like(x, 1e-3)
    y.backward(upstream)
    assert torch.equal(x.grad, upstream)
    assert fake_quantise(x, "match") is x
    xb = x.detach().bfloat16()
    assert torch.equal(fake_quantise(xb, "fp8"), dequantise_fp8(*quantise_fp8(xb), torch.bfloat16))


# --- the linear layer ---------------------------------------------------------------------------


def test_quant_linear():
    gen = torch.Generator().manual_seed(4)
    lin = QuantLinear(48, 24, fmt="fp8_e4m3")
    plain = nn.Linear(48, 24)
    plain.load_state_dict(lin.state_dict())
    assert isinstance(lin, nn.Linear) and lin.state_dict().keys() == plain.state_dict().keys()
    x = torch.randn(2, 5, 48, generator=gen)
    expected = torch.nn.functional.linear(fake_quant(x), fake_quant(lin.weight, -1), lin.bias)
    assert torch.equal(lin(x), expected)
    per_tensor = quant_linear(x, lin.weight, lin.bias, weight_scale="tensor")
    assert torch.equal(per_tensor, torch.nn.functional.linear(fake_quant(x), fake_quant(lin.weight, (0, 1)), lin.bias))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y16 = lin(x)
    assert y16.dtype == torch.bfloat16 and torch.equal(y16, expected.bfloat16()), "FP32 accumulation, the trunk's dtype out"
    lin.set_format("none")
    assert torch.equal(lin(x), plain(x)), "off: nn.Linear bit for bit"
    nobias = QuantLinear(48, 24, bias=False, fmt="fp8_e4m3")
    assert nobias(x).shape == (2, 5, 24)
    with pytest.raises(ValueError):
        QuantLinear(4, 4, fmt="int8")


# --- the FP8 trunk -----------------------------------------------------------------------------

TRUNK_LINEARS = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "ffn.lin_in", "ffn.lin_out")


def _as_plain_linears(model):
    """The model with every QuantLinear replaced by an ``nn.Linear`` holding the same tensors."""
    for block in model.trunk.blocks:
        for name in TRUNK_LINEARS:
            owner, attr = name.split(".")
            q = getattr(getattr(block, owner), attr)
            plain = nn.Linear(q.in_features, q.out_features, bias=q.bias is not None)
            plain.weight, plain.bias = q.weight, q.bias
            setattr(getattr(block, owner), attr, plain)
    return model


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_the_flag_off_is_the_network_bit_for_bit(dtype):
    """``trunk_quant="none"`` (the default) is the network before the flag existed, bitwise: the
    same outputs as the model whose trunk projections are plain ``nn.Linear`` layers. (Also checked
    once against the tree before the change - F1 and five other configurations, outputs and
    gradients - in the change's notes.)"""
    cfg = NetConfig(trunk_dtype=dtype, n_layers=3)
    assert NetConfig().trunk_quant == "none"
    batch = collate([synthetic_state(31, 3), synthetic_state(32, 5)])
    a = build_model(cfg, seed=3).eval()(batch, return_trunk=True)
    b = _as_plain_linears(build_model(cfg, seed=3).eval())(batch, return_trunk=True)
    for name in ("trunk", "policy_logits", "wdl_logits", "value_logits", "submit_raw"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_only_the_trunk_projections_are_quantised():
    """The six projections of every block carry the format - no head, embedding, norm, smolgen,
    B15/B16/B17 table or RoPE module does (section 10.2: "keep the policy and value heads and
    RoPE at FP32"; section 8.3)."""
    cfg = NetConfig(n_layers=2, trunk_quant="fp8_e4m3", board_pair_smolgen=True, shaw_key_term=True, opp_reply=True)
    model = build_model(cfg, seed=0)
    quantised = {name for name, m in model.named_modules() if isinstance(m, QuantLinear)}
    assert quantised == {f"trunk.blocks.{i}.{n}" for i in range(2) for n in TRUNK_LINEARS}
    assert all(m.fmt == "fp8_e4m3" for n, m in model.named_modules() if n in quantised)
    assert all(m.fmt == "none" for m in build_model(NetConfig(n_layers=2), seed=0).modules() if isinstance(m, QuantLinear))


def test_heads_and_rope_are_untouched():
    """With the flag on, RoPE's tables and every head's function are bit for bit those of the
    network without it: the same cos/sin, and the same head outputs from the same trunk output."""
    base = build_model(NetConfig(n_layers=2, opp_reply=True, optimistic_policy=True), seed=1).eval()
    fp8 = base.with_trunk_quant("fp8_e4m3")
    batch = collate([synthetic_state(40, 4), synthetic_state(41, 7, n_timelines=3)])
    c0, c1 = base.make_context(batch), fp8.make_context(batch)
    assert torch.equal(c0.cos, c1.cos) and torch.equal(c0.sin, c1.sin)
    assert c0.quant is None and c1.quant == AttnQuant("pow2")
    h = fp8.trunk_output(batch)
    a, b = base.heads(h, batch), fp8.heads(h, batch)
    for name in ("policy_logits", "wdl_logits", "value_logits", "short_term", "value_error", "moves_left_log", "board_logits"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for name in ("square_logits", "effort_logits", "submit_raw", "soft_policy_logits", "opp_reply_logits", "optimistic_logits"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not torch.equal(h, base.trunk_output(batch)), "the trunk did round"


def test_the_fp8_twin_has_the_same_weights_and_the_flag_validates():
    base = build_model(NetConfig(n_layers=2), seed=2)
    base.t5d_optimiser_steps = 2
    twin = base.with_trunk_quant("fp8_e4m3", trunk_quant_attention=False)
    assert twin.cfg.trunk_quant == "fp8_e4m3" and not twin.cfg.trunk_quant_attention and base.cfg.trunk_quant == "none"
    assert twin.t5d_optimiser_steps == 2 and twin.embed.cfg is twin.cfg
    loaded = build_model(twin.cfg, seed=9)
    loaded.load_state_dict(base.state_dict(), strict=True)
    sd = twin.state_dict()
    assert all(torch.equal(v, sd[k]) for k, v in base.state_dict().items())
    assert all(p is not q for p, q in zip(base.parameters(), twin.parameters(), strict=True)), "a copy"
    for bad in (dict(trunk_quant="fp8"), dict(trunk_quant_scale="dynamic"), dict(trunk_quant_weight_scale="row"), dict(ffn_ratio=0)):
        with pytest.raises(ValueError):
            NetConfig(**bad).validate()
    with pytest.raises(ValueError, match="trunk_quant setting"):
        base.with_trunk_quant("fp8_e4m3", trunk_dtype="fp32")


def _gaps(a, b, valid=None):
    d = a.trunk - b.trunk
    d = d if valid is None else d[valid]
    va, vb = torch.softmax(a.wdl_logits, -1), torch.softmax(b.wdl_logits, -1)
    return {
        "trunk_rms": float(d.pow(2).mean().sqrt()),
        "trunk_max": float(d.abs().max()),
        "value": float((a.value_logits - b.value_logits).abs().max()),
        "policy": float((a.policy_logits[a.move_valid] - b.policy_logits[b.move_valid]).abs().max()),
        "value_mean": float(((va[:, 0] - va[:, 2]) - (vb[:, 0] - vb[:, 2])).abs().max()),
    }


@pytest.mark.parametrize("attention", [True, False])
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_the_fp8_network_is_a_bounded_approximation(dtype, attention):
    """G2 measured, not claimed: the FP8 trunk computes a different function from the same
    weights unquantised - bounded (the module docstring), and not zero."""
    for seed in (0, 5):
        base = build_model(NetConfig(trunk_dtype=dtype), seed=seed).eval()
        fp8 = base.with_trunk_quant("fp8_e4m3", trunk_quant_attention=attention)
        for n in (2, 17):
            batch = collate([synthetic_state(30 + n, n)])
            g = _gaps(fp8(batch, return_trunk=True), base(batch, return_trunk=True))
            assert g["trunk_rms"] < 0.1 and g["trunk_max"] < 0.6 and g["value"] < 0.2 and g["policy"] < 0.5, g
            assert g["value_mean"] < 0.025, g
            assert g["trunk_rms"] > 1e-2, "the FP8 network did not round"


@pytest.mark.parametrize("flags", [dict(), dict(shaw_key_term=True), dict(routing_table=True)], ids=["v1", "b17", "routing"])
def test_the_attention_paths_agree_under_fp8(flags):
    """N2's implementation agreement under the FP8 trunk (FP32 arithmetic), up to rare FP8
    rounding flips (the module docstring): every path takes its operands quantised once per layer
    (``attention.quantised_operands``), so each computes the same function."""
    for seed in (0, 1):
        cfg = NetConfig(n_layers=2, trunk_dtype="fp32", trunk_quant="fp8_e4m3", attn_block_boards=2, **flags)
        batch = collate([synthetic_state(200 + seed, 9, n_timelines=3), synthetic_state(300 + seed, 4)])
        ref = build_model(replace(cfg, attn_impl="dense"), seed=seed).eval()(batch, return_trunk=True)
        for impl in ("block_sparse", "chunked", "sdpa", "explicit", "flex"):
            out = build_model(replace(cfg, attn_impl=impl), seed=seed).eval()(batch, return_trunk=True)
            g = _gaps(out, ref, batch.valid)
            assert g["trunk_rms"] < 6e-3 and g["value"] < 8e-3 and g["policy"] < 5e-2, (impl, g)


@pytest.mark.parametrize("rule", ["pow2", "absmax"])
def test_an_evaluation_does_not_depend_on_its_batch(rule):
    """C19-C21's purity under the FP8 trunk: per-token (per token and head in attention) scales
    make a position's evaluation the same alone and in a padded batch with another, up to the
    flips of a different FP32 summation order. Measured (FP32, 2 layers, seeds 0-5), trunk RMS
    over the position's tokens: "pow2" 5.2e-7, "absmax" 3.1e-3. Mutants with one scale per tensor
    give, under "absmax", 2.6e-2 (the projections' inputs) and 2.2e-2 (the attention operands) -
    and under "pow2" 2.1e-4 and 5.2e-7: a power-of-two scale moves only the subnormal floor of a
    floating-point format, so "pow2" barely depends on the granularity. Bound 1e-2."""
    cfg = NetConfig(n_layers=2, trunk_dtype="fp32", trunk_quant="fp8_e4m3", trunk_quant_scale=rule)
    for seed in (0, 1, 2):
        model = build_model(cfg, seed=seed).eval()
        s1, s2 = synthetic_state(400 + seed, 5), synthetic_state(500 + seed, 11, n_timelines=3)
        b1, b2 = collate([s1]), collate([s2, s1])
        alone, mixed = model(b1, return_trunk=True), model(b2, return_trunk=True)
        lb1, lb2 = b1.body_len, b2.body_len  # the global registers follow the batch's longest body
        d = torch.cat([alone.trunk[0, :lb1] - mixed.trunk[1, :lb1], alone.trunk[0, lb1:] - mixed.trunk[1, lb2:]])
        assert float(d.pow(2).mean().sqrt()) < 1e-2, (seed, float(d.pow(2).mean().sqrt()))
        valid = alone.move_valid[0]
        assert float((alone.value_logits[0] - mixed.value_logits[1]).abs().max()) < 3e-2
        assert float((alone.policy_logits[0][valid] - mixed.policy_logits[1][: valid.shape[0]][valid]).abs().max()) < 6e-2


def test_the_cache_takes_linear_only_fp8_and_refuses_quantised_attention():
    """The cached path (C2, deferred): with the projections alone in FP8 it equals the full pass
    along a search path - per-token scales are the same wherever a token is computed - up to the
    rare flips of a different FP32 summation order (the module docstring); with attention quantised
    it refuses: post-RoPE rounding depends on the frame, and the cache computes a frozen board in a
    frame of its own (spec issue 238)."""
    base = build_model(NetConfig(n_layers=2, trunk_dtype="fp32", attn_scope="block_causal"), seed=0).eval()
    with pytest.raises(NotImplementedError, match="trunk_quant_attention"):
        CachedEvaluator(base.with_trunk_quant("fp8_e4m3"))
    for seed in (0, 1):
        model = base.with_trunk_quant("fp8_e4m3", trunk_quant_attention=False)
        root = _root("standard", seed)
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.root
        for move in [None, *_random_moves(root, np.random.default_rng(seed), 4)]:
            if move is not None:
                node = tree.child(node, move)
            g = _gaps(tree.evaluate(node), reference_output(model, node.state))
            assert g["trunk_rms"] < 6e-3 and g["value"] < 8e-3 and g["policy"] < 6e-2, g


def test_the_context_carries_the_format():
    assert "quant" in AttnContext.__dataclass_fields__
    model = build_model(NetConfig(n_layers=1, trunk_quant="fp8_e4m3", trunk_quant_attention=False), seed=0)
    assert model.make_context(collate([synthetic_state(1, 2)])).quant is None
    assert quantise(torch.tensor([1.0, 500.0]), torch.tensor(1.0)).float().tolist() == [1.0, 448.0]
