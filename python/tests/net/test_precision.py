"""Precision (G1): BF16 trunk close to FP32; RoPE and every head in FP32.

Tolerances. BF16 keeps an 8-bit significand (relative rounding ~2^-9 = 0.2%). With the
residual stream and norms in FP32 and only matmuls and attention in BF16, the measured
gap after 16 layers (default configuration - F5's pre-norm RMSNorm - model seeds 0, 5 and
11, ``synthetic_state(30 + N, N)`` for N = 2, 17, 60; re-measured after F5) is at most
0.23% RMS on the unit-RMS final-normed trunk output (max 1.3%), 0.0037 on value logits,
0.0088 on policy logits and 1.9e-4 on the value mean. The bounds below sit 3-11x above
those measurements, so they catch a precision bug (e.g. RoPE computed in BF16, where an
angle of 400 rad carries ~1.5 rad of error) without flaking. F5's post-norm arm (ablation
6y) is held to the same bounds; its measured gap is smaller (0.06% RMS, max 0.7%; 8.3e-4
on value logits, 2.5e-3 on policy logits, 1.9e-5 on the value mean), since every
sublayer's output is normed in FP32.
"""

from __future__ import annotations

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state, translate
from t5d.net.rope import apply_rope, rope_angles, rope_cos_sin, rope_tables

TRUNK_RMS = 0.01
TRUNK_MAX = 0.06
VALUE_LOGITS_MAX = 0.02
POLICY_LOGITS_MAX = 0.03
VALUE_MEAN_MAX = 2e-3


@pytest.fixture(scope="module", params=["pre", "post"], ids=["v1", "post-norm"])
def pair(request):
    return (
        build_model(NetConfig(trunk_dtype="fp32", norm_placement=request.param), seed=5).eval(),
        build_model(NetConfig(trunk_dtype="bf16", norm_placement=request.param), seed=5).eval(),
    )


@pytest.mark.parametrize("n", [2, 17])
def test_bf16_trunk_close_to_fp32(pair, n):
    m32, m16 = pair
    batch = collate([synthetic_state(30 + n, n)])
    a, b = m32(batch, return_trunk=True), m16(batch, return_trunk=True)
    diff = (a.trunk - b.trunk)[batch.valid]
    assert diff.pow(2).mean().sqrt() < TRUNK_RMS
    assert diff.abs().max() < TRUNK_MAX
    assert (a.value_logits - b.value_logits).abs().max() < VALUE_LOGITS_MAX
    assert (a.policy_logits - b.policy_logits)[batch.move_valid].abs().max() < POLICY_LOGITS_MAX
    assert (a.value_mean() - b.value_mean()).abs().max() < VALUE_MEAN_MAX
    # And BF16 really is in use: the outputs are not bitwise FP32.
    assert not torch.equal(a.trunk, b.trunk)


def test_rope_runs_in_fp32_under_autocast():
    """G1: RoPE in FP32 even inside the BF16 trunk's autocast. The rotation of a BF16 Q/K
    at large angles (h ~ 400, the raw coordinates of a long game) must match an FP64
    rotation of the same BF16 values. Measured: 1.4e-5 (FP32 angle rounding at ~400 rad);
    rotating in BF16 instead gives 1.9e-2, and BF16 angles 1.7."""
    axis, freq = rope_tables(NetConfig())
    gen = torch.Generator().manual_seed(7)
    coords = torch.tensor([[[3.0, 5.0, -7.0, 401.0], [0.0, 7.0, 12.0, 398.0], [6.0, 1.0, -3.0, 2.0]]])
    x = torch.randn(1, 8, 3, 32, generator=gen).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cos, sin = rope_cos_sin(rope_angles(coords, axis, freq))
        out = apply_rope(x, cos, sin)
    assert out.dtype == torch.float32
    picked = coords.double()[:, None].expand(1, 1, 3, 4).gather(-1, axis[None, :, None, :].expand(1, 1, 3, 16))
    angles = picked * freq.double()[None, :, None, :]
    c64, s64 = torch.cat([angles.cos(), angles.cos()], -1), torch.cat([angles.sin(), angles.sin()], -1)
    x64 = x.double()
    ref = x64 * c64 + torch.cat([-x64[..., 16:], x64[..., :16]], -1) * s64
    assert (out.double() - ref).abs().max() < 1e-4, float((out.double() - ref).abs().max())


def test_bf16_large_times_stay_accurate():
    """With FP32 angles a late-game time coordinate costs no accuracy in BF16, even
    without recentring (raw coordinates up to h ~ 610)."""
    late = translate(synthetic_state(40, 8), dh=600)
    raw32 = build_model(NetConfig(trunk_dtype="fp32", rope_recenter=False), seed=5).eval()
    raw16 = build_model(NetConfig(trunk_dtype="bf16", rope_recenter=False), seed=5).eval()
    a, b = raw32(collate([late])), raw16(collate([late]))
    assert (a.value_logits - b.value_logits).abs().max() < VALUE_LOGITS_MAX
