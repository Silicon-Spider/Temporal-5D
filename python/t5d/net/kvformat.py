"""The storage format of frozen boards' keys and values (stage N3; register C15, G1).

``NetConfig.frozen_kv_format`` names how a frozen (superseded) board's K/V exist under the
block-causal scope, and the cached path (``cache.py``) stores them that way:

* ``"match"`` (default): in the trunk's own precision - BF16 under the BF16 trunk, FP32
  under the FP32 trunk. G1's "cache and live computation share precision": the cache keeps
  exactly the values the live computation produces, so storing them changes nothing.
* ``"fp8"``: through ``torch.float8_e4m3fn`` with one FP32 scale per board, layer, head and
  K-or-V (the absmax over the board's tokens and the head's dims, over 448), dequantised to
  the compute dtype wherever they are read - by the board itself, by later frozen boards,
  by the live set. The ordinary forward applies the same rounding (``model.py``), so this
  is a function a network can be trained on; a cache can then serve it at half the bytes.
  C15 adopts FP8 K/V on the grounds that it "obeys section 10.2", which presumes an FP8
  trunk (G2, not built); a BF16-trunk network served from an FP8 cache computes a different
  function unless it is trained this way (build log, the C15-vs-G1 spec issue).
"""

from __future__ import annotations

import torch

FROZEN_KV_FORMATS: tuple[str, ...] = ("match", "fp8")
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448
_SCALE_FLOOR = 1e-30


def quantise_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 (e4m3fn) data and FP32 scales of ``x`` ``(..., T, dh)``: one scale per leading
    index - the absmax over the last two dims (a board's tokens and a head's dims), over 448."""
    xf = x.float()
    scale = (xf.abs().amax(dim=(-2, -1), keepdim=True) / FP8_MAX).clamp(min=_SCALE_FLOOR)
    return (xf / scale).to(FP8_DTYPE), scale


def dequantise_fp8(data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (data.float() * scale).to(dtype)


def fake_quantise(x: torch.Tensor, kv_format: str) -> torch.Tensor:
    """``x`` as the format holds it: unchanged under "match", through FP8 and back under "fp8"."""
    if kv_format == "match":
        return x
    if kv_format != "fp8":
        raise ValueError(f"unknown frozen K/V format {kv_format!r}")
    data, scale = quantise_fp8(x)
    return dequantise_fp8(data, scale, x.dtype)
