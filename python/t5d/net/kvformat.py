"""The storage format of frozen boards' keys and values (stage N3; register C15, G1).

``NetConfig.frozen_kv_format`` names how a frozen (superseded) board's K/V exist under the
block-causal scope, and the cached path (``cache.py``) stores them that way:

* ``"match"`` (default): in the trunk's own precision - BF16 under the BF16 trunk, FP32
  under the FP32 trunk. G1's "cache and live computation share precision": the cache keeps
  exactly the values the live computation produces, so storing them changes nothing.
* ``"fp8"``: through ``torch.float8_e4m3fn`` with one FP32 scale per board, layer, head and
  K-or-V (the absmax over the board's tokens and the head's dims, over 448 - ``quant.py``'s
  ``"absmax"`` rule), dequantised to the compute dtype wherever they are read - by the board
  itself, by later frozen boards, by the live set. The ordinary forward applies the same rounding
  (``model.py``), so this is a function a network can be trained on; a cache can then serve it at
  half the bytes. C15 adopts FP8 K/V on the grounds that it "obeys section 10.2", which presumes
  an FP8 trunk (G2, ``NetConfig.trunk_quant``) or a network trained with FP8 frozen K/V; a
  BF16-trunk network served from an FP8 cache computes a different function unless it is trained
  this way (build log, the C15-vs-G1 spec issue 34, settled upstream).

The element format, its rounding and saturation are ``quant.py``'s - one definition for G2, G8 and
C15 ("the simulated format is pinned with the inference path, K/V cache included"). The forward's
rounding (``fake_quantise``) passes the gradient straight through (``quant.fake_quant_scaled``):
until 2026-10-23 it cast inside autograd, which rounded the gradient onto the unscaled E4M3 grid -
with ``|x|`` about 0.1, 63 of 64 gradients were zero - so the "fp8" network could not be trained
(BUILD_LOG C15 row).
"""

from __future__ import annotations

import torch

from .quant import E4M3, E4M3_MAX, SCALE_FLOOR, dequantise, fake_quant_scaled, quantise, scale_of

FROZEN_KV_FORMATS: tuple[str, ...] = ("match", "fp8")
FP8_DTYPE = E4M3
FP8_MAX = E4M3_MAX  # 448
_SCALE_FLOOR = SCALE_FLOOR


def _board_scale(x: torch.Tensor) -> torch.Tensor:
    """One FP32 scale per leading index of ``x`` ``(..., T, dh)``: the absmax over the last two
    dims (a board's tokens and a head's dims), over 448."""
    return scale_of(x, (-2, -1), "absmax")


def quantise_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 (e4m3fn) data and FP32 scales of ``x`` ``(..., T, dh)``: one scale per leading
    index - the absmax over the last two dims (a board's tokens and a head's dims), over 448."""
    scale = _board_scale(x)
    return quantise(x, scale), scale


def dequantise_fp8(data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return dequantise(data, scale, dtype)


def fake_quantise(x: torch.Tensor, kv_format: str) -> torch.Tensor:
    """``x`` as the format holds it: unchanged under "match", through FP8 and back under "fp8"
    (the values ``dequantise_fp8(*quantise_fp8(x), x.dtype)`` - the cache's - with the gradient
    straight through)."""
    if kv_format == "match":
        return x
    if kv_format != "fp8":
        raise ValueError(f"unknown frozen K/V format {kv_format!r}")
    return fake_quant_scaled(x, _board_scale(x))
