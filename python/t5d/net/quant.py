"""FP8 E4M3 emulation: the one definition of the trunk's 8-bit format (register G2, G8, C15;
spec section 10.2).

Section 10.2: "Quantise the trunk's matrix multiplications - weights and activations, attention
included - and keep the policy and value heads and RoPE at FP32"; "FP8 before INT8"; QAT is
"prepared, not adopted ... the exact simulated format is pinned once the inference path is
chosen", and "QAT simulates exactly the deployed format, K/V cache included". This module is the
format, emulated on CPU in PyTorch - numerics only, never speed (FP8 speed needs a GPU, M19). G2's
network switch (``NetConfig.trunk_quant``), G8's QAT hook (``train.trainer.apply_qat``) and C15's
frozen-K/V format (``kvformat.py``) all round through it.

**The element format** is OCP's 8-bit floating point E4M3 ("OCP 8-bit Floating Point
Specification (OFP8)", rev. 1.0, 2023; Micikevicius et al. 2022, "FP8 Formats for Deep
Learning"): 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits; exponent field 0 holds the
subnormals ``(-1)^s 2^-6 m/8``, fields 1-15 the normals ``(-1)^s 2^(e-7) (1 + m/8)``, and the one
code ``S.1111.111`` is NaN - no infinities, so the largest finite value is 448 and the smallest
subnormal 2^-9. It is PyTorch's ``torch.float8_e4m3fn``; ``round_e4m3`` takes a value to it by
round-to-nearest-even (ties to the even code, subnormals included) with **saturation**: every
value beyond +-448, infinities included, becomes +-448 (OCP's saturating conversion, clamped here
before the cast so that nothing depends on how a backend converts out-of-range values - on this
backend, PyTorch 2.13's CPU cast, the clamp is not observable: the cast saturates by itself, so no
test here can tell it is there); NaN stays NaN; zero keeps its sign. ``tests/net/test_quant.py``
checks it against an independent decode of all 256 codes, at every midpoint between neighbours and
beyond the range. **The input precision is FP32** (BF16 and FP16 widen to it exactly): a float64
tensor is refused, since narrowing it to FP32 first would round it twice - a value just off an
E4M3 midpoint could land on it and then tie to even the wrong way.

**Scales.** A tensor is held as E4M3 codes times FP32 scales, one scale per slice of the tensor:
``fake_quant(x, dims)`` takes one scale over the dims ``dims`` (kept), ``s = absmax / 448``, so the
slice's largest magnitude lands on the format's largest value. ``scale_rule`` (a PLACEHOLDER of a
committed-class silence - the spec says only "FP8", spec issue 238) chooses:

* ``"pow2"`` (the default): ``s`` rounded up to a power of two (as OCP's MX formats' E8M0 scales
  are). Dividing by it is exact, so the only rounding is the cast; and every dequantised value - an
  E4M3 code, 4 significant bits, times a power of two - is exact in BF16 as in FP32, so the trunk's
  BF16 kernels compute an FP8 GEMM's products exactly (with FP32 accumulation). It gives up at
  most one binade at the top of the range.
* ``"absmax"``: ``s = absmax / 448`` exactly (the scale of C15's frozen K/V, ``kvformat.py``). Its
  dequantised values are rounded again wherever a BF16 kernel reads them, so under the BF16 trunk
  it emulates the FP8 GEMM to BF16 precision only (an FP8 kernel applies the scales to its FP32
  accumulator instead).

A scale is computed from the tensor itself (dynamic; no calibration - a placeholder, issue 238)
under no gradient, floored at ``SCALE_FLOOR`` (a zero slice gives zeros). A slice holding an
infinity or a NaN gets a non-finite scale under either rule, so the whole slice comes out NaN and
the trainer's non-finite guards fire (rounding such a scale "up to a power of two" would give a
finite, platform-dependent one: ``frexp`` of inf or NaN is unspecified). Which slices share a
scale is the caller's: the trunk takes one per token (activations), per token and head (the
attention operands) and per output channel (weights) - never per tensor or per batch for
activations, whose scales would then depend on what else is in the batch, breaking C19-C21's
purity ("same state + epoch -> same evaluation") and C2/C17's cached = dense (``NetConfig``).

**Gradients: the straight-through estimator** (Bengio, Leonard & Courville 2013; the minimal
reading of section 10.2's "fake-quantisation hooks" - learned scales, e.g. LSQ, are not built, a
placeholder, issue 238). ``fake_quant``'s forward is the rounded value; its backward passes the
incoming gradient through unchanged, and the scale takes none. Autograd through the cast itself
would round the gradient onto the unscaled E4M3 grid - a gradient of 1e-4 becomes 0 - which is
the C15 defect this replaces (``kvformat.fake_quantise``; BUILD_LOG C15 row).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: The trunk's formats (``NetConfig.trunk_quant``): "none" (the network as G1 has it) or FP8 E4M3.
TRUNK_QUANT_FORMATS: tuple[str, ...] = ("none", "fp8_e4m3")
#: The ``NetConfig`` fields that make up a network's trunk format (``trunk_format``).
TRUNK_QUANT_FIELDS: tuple[str, ...] = ("trunk_quant", "trunk_quant_attention", "trunk_quant_scale", "trunk_quant_weight_scale")
#: How a slice's scale is chosen (the module docstring; a placeholder, spec issue 238).
SCALE_RULES: tuple[str, ...] = ("pow2", "absmax")
#: Which slices of a weight share a scale: one per output channel, or one per tensor (both are
#: fixed by the weights alone, so neither depends on the batch; a placeholder, issue 238).
WEIGHT_SCALES: tuple[str, ...] = ("channel", "tensor")

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0  # (1 + 6/8) * 2^8, the largest finite value (S.1111.110)
E4M3_MIN_NORMAL = 2.0**-6
E4M3_MIN_SUBNORMAL = 2.0**-9
#: The smallest scale: a slice whose absmax is below ``SCALE_FLOOR * 448`` keeps this scale, so a
#: zero slice gives zeros and no scale is an FP32 subnormal.
SCALE_FLOOR = 1e-30


def _refuse_float64(*xs: torch.Tensor) -> None:
    """The format's input precision is FP32 (the module docstring): float64 is refused - narrowing
    it first would round it twice."""
    if any(x.dtype == torch.float64 for x in xs):
        raise ValueError("FP8 emulation takes FP32 (or BF16/FP16) input: float64 would be rounded twice (to FP32, then E4M3)")


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """``x`` rounded to E4M3 - round-to-nearest-even, saturating at +-448 (infinities too), NaN
    kept - as FP32 values (exact: every E4M3 value is an FP32 value)."""
    _refuse_float64(x)
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).float()


def _pow2_ceil(s: torch.Tensor) -> torch.Tensor:
    """The least power of two not below each (positive, finite) ``s`` - for a non-finite one,
    ``frexp`` is unspecified (a finite exponent here), so ``scale_of`` keeps it out."""
    mantissa, exponent = torch.frexp(s)  # s = mantissa * 2^exponent, mantissa in [0.5, 1)
    return torch.ldexp(torch.ones_like(s), exponent - (mantissa == 0.5).to(exponent.dtype))


def scale_of(x: torch.Tensor, dims: int | tuple[int, ...], rule: str = "pow2") -> torch.Tensor:
    """The FP32 scales of ``x``: one per slice over ``dims`` (kept as size-1 dims), from the
    slice's absmax by ``rule`` (``SCALE_RULES``). No gradient."""
    if rule not in SCALE_RULES:
        raise ValueError(f"scale rule must be one of {SCALE_RULES}")
    _refuse_float64(x)
    with torch.no_grad():
        amax = x.detach().float().abs().amax(dim=dims, keepdim=True)
        s = (amax / E4M3_MAX).clamp(min=SCALE_FLOOR)
        # A slice holding inf or NaN keeps its non-finite scale under either rule (the module docstring).
        return torch.where(torch.isfinite(s), _pow2_ceil(s), s) if rule == "pow2" else s


def quantise(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The E4M3 codes of ``x`` under ``scale`` (broadcastable): ``round_e4m3(x / scale)`` as
    ``torch.float8_e4m3fn``. Storage; no gradient flows through a cast to it usefully."""
    _refuse_float64(x)
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def dequantise(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Codes times their scales, in ``dtype``."""
    return (codes.float() * scale).to(dtype)


def trunk_format(cfg) -> tuple:
    """A network configuration's trunk format, as it decides the function computed: ``("none",)``
    when ``cfg.trunk_quant`` is "none" (the settings then do nothing), else the format and its
    settings (``TRUNK_QUANT_FIELDS``). What a record's network condition, a network's identity
    (``train.records.network_id``) and the cache's format check compare."""
    if cfg.trunk_quant == "none":
        return ("none",)
    return tuple(getattr(cfg, f) for f in TRUNK_QUANT_FIELDS)


class _StraightThrough(torch.autograd.Function):
    """Forward: ``round_e4m3(x / scale) * scale`` in ``x``'s dtype. Backward: the identity on
    ``x`` (the straight-through estimator); the scale takes no gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        _refuse_float64(x)
        return (round_e4m3(x.float() / scale) * scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def fake_quant_scaled(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x`` through E4M3 under the given ``scale`` and back (the forward), with a
    straight-through gradient (the backward)."""
    return _StraightThrough.apply(x, scale)


def fake_quant(x: torch.Tensor, dims: int | tuple[int, ...] = -1, rule: str = "pow2") -> torch.Tensor:
    """``x`` through E4M3 and back with one scale per slice over ``dims`` (``scale_of``), in
    ``x``'s dtype; the gradient straight through (the module docstring)."""
    return fake_quant_scaled(x, scale_of(x, dims, rule))


def quant_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, rule: str = "pow2", weight_scale: str = "channel"
) -> torch.Tensor:
    """An FP8 linear layer, emulated: the input fake-quantised with one scale per token (its last
    dim), the weight with one per output channel (or one per tensor, ``weight_scale``), multiplied
    and accumulated in FP32 with the bias added in FP32 (an FP8 GEMM's FP32 accumulator), and the
    result returned in the dtype the layer's ``nn.Linear`` returns there - the autocast dtype under
    autocast (BF16 under the BF16 trunk), else the input's."""
    if weight_scale not in WEIGHT_SCALES:
        raise ValueError(f"weight scale must be one of {WEIGHT_SCALES}")
    _refuse_float64(x, weight)
    device = x.device.type
    out_dtype = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else x.dtype
    with torch.autocast(device_type=device, enabled=False):
        xq = fake_quant(x.float(), -1, rule)
        wq = fake_quant(weight.float(), -1 if weight_scale == "channel" else (0, 1), rule)
        y = torch.nn.functional.linear(xq, wq, None if bias is None else bias.float())
    return y.to(out_dtype)


class QuantLinear(torch.nn.Linear):
    """``nn.Linear`` with G2's switch: under ``fmt="fp8_e4m3"`` its forward is ``quant_linear``;
    under ``"none"`` it is ``nn.Linear``'s, bit for bit. Same parameters, same ``state_dict``
    keys, still an ``nn.Linear`` - so a network's weights load into its FP8 twin unchanged, and
    its initialisation draws them as before (``model._init_weights``). The trunk's six
    projections per block are these (``attention.SelfAttention``, ``trunk.FeedForward``); no
    head, embedding or RoPE module is one."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        fmt: str = "none",
        rule: str = "pow2",
        weight_scale: str = "channel",
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.set_format(fmt, rule, weight_scale)

    def set_format(self, fmt: str, rule: str = "pow2", weight_scale: str = "channel") -> None:
        if fmt not in TRUNK_QUANT_FORMATS or rule not in SCALE_RULES or weight_scale not in WEIGHT_SCALES:
            raise ValueError(f"format {fmt!r}, scale rule {rule!r}, weight scale {weight_scale!r}: unknown")
        self.fmt, self.rule, self.weight_scale = fmt, rule, weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fmt == "none":
            return super().forward(x)
        return quant_linear(x, self.weight, self.bias, self.rule, self.weight_scale)

    def extra_repr(self) -> str:
        extra = "" if self.fmt == "none" else f", fmt={self.fmt}, rule={self.rule}, weight_scale={self.weight_scale}"
        return super().extra_repr() + extra


@dataclass(frozen=True)
class AttnQuant:
    """The attention operands' format (``AttnContext.quant``; None: not quantised). The operands
    of the score and value matmuls - the unrotated Q and K the position-free pairs read, the
    rotated Q and K (after FP32 RoPE: G1), and V - each fake-quantised once per layer with one
    scale per token and head (``rule``), before any implementation selects from them: every
    operand the pair rule builds (rotated, time-only rotated, unrotated, widened) is then a
    selection of quantised values, so every implementation computes the same function. The
    probabilities P stay in the compute dtype (a placeholder: the fused CPU kernel cannot round
    them; issue 238); the global merge's appended log-sum-exp and marker columns are never
    quantised."""

    rule: str = "pow2"

    def operands(self, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Each ``(B, H, L, d)`` operand fake-quantised per token and head."""
        return tuple(fake_quant(x, -1, self.rule) for x in xs)


__all__ = [
    "E4M3",
    "E4M3_MAX",
    "E4M3_MIN_NORMAL",
    "E4M3_MIN_SUBNORMAL",
    "SCALE_FLOOR",
    "SCALE_RULES",
    "TRUNK_QUANT_FIELDS",
    "TRUNK_QUANT_FORMATS",
    "WEIGHT_SCALES",
    "AttnQuant",
    "QuantLinear",
    "dequantise",
    "fake_quant",
    "fake_quant_scaled",
    "quant_linear",
    "quantise",
    "round_e4m3",
    "scale_of",
    "trunk_format",
]
