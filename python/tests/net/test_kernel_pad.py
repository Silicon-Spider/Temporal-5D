"""Register G11's kernel-friendly widths (``NetConfig.attn_pad_multiple``; spec section 10.2).

Section 10.2, "Kernel-friendly widths" (2026-11-15 review, decision 15, settling BUILD_LOG issue
521): "Fused attention kernels want head widths in multiples of 8. On the AMD card a masked call
at width 45 or 33 - the v1's widened query and key features - fell back to the unfused path ...
Zero-padding Q and K is exact - a zero feature adds zero to every score - so the v1 is to pad
them". Held here:

* the as-built widths are 45 and 33, queries, keys and values alike, on every fast path
  ("block_sparse", "chunked", "sdpa"), and 76, 64 and 32 (queries and keys) through B14's score
  function ("flex"); at multiple 1 - the default in this build - nothing is padded;
* at 8 every kernel call is at a multiple of 8 (48 and 40; 80, 64 and 32), the padded features
  are the unpadded ones with zero columns appended, and the network computes the same function;
* ``cost.count_flops`` counts the padded width, by exactly what the widths predict;
* the field is validated.

**Tolerances (section 15.2's protocol: matched to the arithmetic).** A zero column is exact in
arithmetic: it adds a zero product to a score. Whether the *floating-point* result is the same
depends on the kernel alone - a product summed over the feature width in order gives the same
bits with zeros appended, one reduced in another grouping need not. Measured on the development
machine (PyTorch 2.13.0+cpu, Windows 11, Ryzen 9 9900X; 2026-11-22), padded against unpadded:

* **the forward: bit for bit** - every output of every case of this module, FP32 and BF16, on
  all four padded implementations (38 comparisons), and the same on the configurations and
  sizes recorded in ``docs/MEASUREMENTS.md`` ("G11 - the kernel's widths, where a layer's operators come from");
* **gradients: not bit for bit** - under autograd a learned bias sends the CPU kernel down its
  math path, whose backward products are taken at the padded width: every parameter's gradient
  within 7.2e-7 of the unpadded one (FP32, 2 layers, the three fast paths).

The assertions do not demand the bitwise result of every platform's kernel (BUILD_LOG issue 602:
last bits differ between CPUs): they hold the padded network to the bounds this suite already
gives two paths of one function whose sums run in another order - FP32: 2e-5 on the unit-scale
trunk output and 1e-4 on every head (``test_cache.py``'s); BF16, at 2 layers: 0.06 on the trunk
(``test_precision.py``'s largest element) and 0.03 on every head; gradients 1e-4 plus 1e-5
relative (``test_chunked.py``'s). What the padding *is* - zero columns, appended, the merge's
extra feature left in its column - is pinned exactly by the structural tests, so the numerical
comparison has only the kernel's rounding to bound (it cannot tell a constant pad from a zero
one: a constant added to every score of a row leaves its softmax alone).

Protocol: random weights at ``build_model``'s scale, model seeds 0-2, one position and padded
batches, 3 to 12 boards, 2 layers (the comparison is per kernel call; 16 layers and the large
sizes are the cross-build comparison's), every padded implementation, both precisions, and the
register-count arms that change the widths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
import torch

from t5d.net import CachedEvaluator, NetConfig, build_model, collate, reference_output, synthetic_state
from t5d.net import attention as attention_module
from t5d.net import flex as flex_module
from t5d.net.attention import _KernelKeys, padded_width
from t5d.net.cost import count_flops, kernel_widths

#: The implementations that hand widened features to a kernel: the fast paths (stock fused
#: attention) and B14's score function. "explicit" and "dense" call none.
FAST = ("block_sparse", "chunked", "sdpa")
PADDED = (*FAST, "flex")
#: The suite's bounds for two paths whose sums run in another order (the module docstring).
BOUNDS = {"fp32": (2e-5, 1e-4), "bf16": (0.06, 0.03)}


def _model(impl: str, dtype: str, pad: int, seed: int = 0, layers: int = 2, **flags):
    return build_model(NetConfig(trunk_dtype=dtype, n_layers=layers, attn_impl=impl, attn_pad_multiple=pad, **flags), seed=seed).eval()


def _batch(impl: str, seed: int, padded: bool):
    """One position, or a larger one beside a smaller (padding); small for the eager score function."""
    n = 3 if impl == "flex" else 9
    states = [synthetic_state(100 * seed + n, n)]
    if padded:
        states.append(synthetic_state(100 * seed + n + 1, max(1, n // 3)))
    return collate(states)


class _Kernel:
    """Records the widths (and shapes) of every kernel call: stock fused attention on the fast
    paths, ``flex_attention`` through the score function."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[tuple] = []  # (query shape, key shape, value shape)
        fused, flex = attention_module.F.scaled_dot_product_attention, flex_module.flex_attention

        def spy_fused(q, k, v, attn_mask=None, scale=None):
            self.calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
            return fused(q, k, v, attn_mask=attn_mask, scale=scale)

        def spy_flex(q, k, v, **kwargs):
            self.calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
            return flex(q, k, v, **kwargs)

        monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy_fused)
        monkeypatch.setattr(flex_module, "flex_attention", spy_flex)

    def widths(self) -> Counter:
        """``(query width, key width, value width) -> calls``."""
        return Counter((q[-1], k[-1], v[-1]) for q, k, v in self.calls)


def _tensors(out) -> dict[str, torch.Tensor]:
    return {name: value for name, value in vars(out).items() if isinstance(value, torch.Tensor) and value.is_floating_point()}


def _gap(a, b) -> dict[str, float]:
    """The largest difference of each output over its finite entries (-inf masks must coincide)."""
    gaps = {}
    for name, x in _tensors(a).items():
        y = getattr(b, name)
        finite = torch.isfinite(x)
        assert torch.equal(finite, torch.isfinite(y)), name
        assert torch.equal(x[~finite], y[~finite]), name
        gaps[name] = float((x[finite] - y[finite]).abs().max()) if bool(finite.any()) else 0.0
    return gaps


# --- the default: the widths as built ---------------------------------------------------------------


def test_the_default_pads_nothing():
    """Off equals what was there before: 1 is the default (the user's ruling of 2026-11-23 - 8 is
    for the compiled GPU kernel path), it reaches the context, and at 1 every width is left as it
    is."""
    cfg = NetConfig()
    assert cfg.attn_pad_multiple == 1
    model = build_model(replace(cfg, n_layers=1), seed=0)
    assert model.make_context(collate([synthetic_state(0, 2)])).pad_multiple == 1
    assert attention_module.AttnContext.__dataclass_fields__["pad_multiple"].default == 1, "a context made without the field pads nothing"
    assert all(padded_width(w, 1) == w for w in range(1, 200))
    assert all(v["as_built"] == v["padded"] for v in kernel_widths(cfg).values())


@pytest.mark.parametrize("impl", FAST)
def test_the_as_built_widths_are_45_and_33(monkeypatch, impl):
    """BUILD_LOG issue 521's figures, verified: the fast paths hand the kernel queries, keys and
    values 45 wide - 32 rotated dims, the 12 unrotated file/rank dims that score a key without
    them (B8), one for the global merge's extra key - and 33 wide on the rows without file and rank
    (32 dims rotated on timeline and time, and the merge's one). Neither is a multiple of 8."""
    spy = _Kernel(monkeypatch)
    _model(impl, "bf16", 1)(_batch(impl, 0, padded=True))
    assert set(spy.widths()) == {(45, 45, 45), (33, 33, 33)}, spy.widths()
    widths = kernel_widths(NetConfig())
    assert (widths["all_axes"]["as_built"], widths["time_only"]["as_built"]) == (45, 33)


def test_the_score_function_s_as_built_widths(monkeypatch):
    """Through B14's score function the queries and keys are 76 wide (the all-axes features and
    the unrotated dims that score a global key), 64 (the rows without file and rank) and 32 (the
    global registers' rows); the values are never widened."""
    spy = _Kernel(monkeypatch)
    _model("flex", "fp32", 1, layers=1)(_batch("flex", 0, padded=False))
    assert set(spy.widths()) == {(76, 76, 32), (64, 64, 32), (32, 32, 32)}, spy.widths()
    widths = kernel_widths(NetConfig())
    assert [widths[call]["as_built"] for call in ("flex_all_axes", "flex_time_only", "flex_global")] == [76, 64, 32]


# --- the arm ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("impl", FAST)
def test_the_arm_pads_every_kernel_call_to_a_multiple_of_8(monkeypatch, impl, dtype):
    """At 8 every kernel call is 48 or 40 wide - queries, keys and values - and there are exactly
    as many calls, over the same rows and keys, as unpadded."""
    batch = _batch(impl, 1, padded=True)
    spy = _Kernel(monkeypatch)
    _model(impl, dtype, 1)(batch)
    plain = list(spy.calls)
    spy.calls.clear()
    _model(impl, dtype, 8)(batch)
    assert set(spy.widths()) == {(48, 48, 48), (40, 40, 40)}, spy.widths()
    assert [(q[:-1], k[:-1], v[:-1]) for q, k, v in spy.calls] == [(q[:-1], k[:-1], v[:-1]) for q, k, v in plain]
    assert [q[-1] for q, _, _ in spy.calls] == [padded_width(q[-1], 8) for q, _, _ in plain]


def test_the_arm_pads_the_score_function_s_queries_and_keys(monkeypatch):
    spy = _Kernel(monkeypatch)
    _model("flex", "fp32", 8, layers=1)(_batch("flex", 1, padded=False))
    assert set(spy.widths()) == {(80, 80, 32), (64, 64, 32), (32, 32, 32)}, spy.widths()


@pytest.mark.parametrize(
    ("flags", "multiple", "want"),
    [
        (dict(), 8, {(48, 48, 48), (40, 40, 40)}),
        (dict(), 16, {(48, 48, 48)}),
        (dict(), 5, {(45, 45, 45), (35, 35, 35)}),
        (dict(regs_per_board=0), 8, {(40, 40, 40)}),  # no row lacks file and rank: 33 wide, one kind of call
        (dict(global_regs=0), 8, {(48, 48, 48), (32, 32, 32)}),  # no merge: 44 and 32 wide
        (dict(rope_per_head=True, routing_table=True), 8, {(72, 72, 72), (40, 40, 40)}),  # B5: every dim is some head's file/rank: 65
    ],
    ids=["v1-8", "v1-16", "v1-5", "no-board-registers", "no-global-registers", "b5-routing"],
)
def test_the_widths_follow_the_configuration(monkeypatch, flags, multiple, want):
    """``cost.kernel_widths`` states what the kernel is handed - as built and padded - for the arms
    that change the features: register counts (ablation 3) and B5's per-head allocation."""
    spy = _Kernel(monkeypatch)
    model = _model("block_sparse", "fp32", multiple, layers=1, **flags)
    model(collate([synthetic_state(3, 5, **{k: v for k, v in flags.items() if k in ("regs_per_board", "global_regs")})]))
    assert set(spy.widths()) == want, spy.widths()
    stated = kernel_widths(model.cfg)
    assert {(w["padded"],) * 3 for call, w in stated.items() if not call.startswith("flex")} == want
    assert all(w["padded"] == padded_width(w["as_built"], multiple) and w["padded"] % multiple == 0 for w in stated.values())


@pytest.mark.parametrize("merged", [False, True])
def test_the_padded_features_are_the_unpadded_ones_with_zero_columns(merged):
    """The one place the fast paths' widths are set (``_KernelKeys.build``): padded keys and
    values are the unpadded ones, column for column, then zeros - the merge's extra feature and its
    value marker stay in their columns."""
    gen = torch.Generator().manual_seed(4)
    kk, vv = torch.randn(2, 3, 7, 44, generator=gen), torch.randn(2, 3, 7, 32, generator=gen)
    for cd in (torch.float32, torch.bfloat16):
        plain, padded = _KernelKeys.build(kk, vv, merged, cd), _KernelKeys.build(kk, vv, merged, cd, 8)
        width = 44 + merged
        assert plain.k.shape[-1] == plain.v.shape[-1] == width and padded.k.shape[-1] == padded.v.shape[-1] == 48
        assert (padded.lk, padded.dv, padded.merged) == (plain.lk, plain.dv, plain.merged)
        for a, b in ((plain.k, padded.k), (plain.v, padded.v)):
            assert torch.equal(b[..., :width], a) and not bool(b[..., width:].any()), "zero columns, appended"
        assert torch.equal(_KernelKeys.build(kk, vv, merged, cd, 1).k, plain.k), "multiple 1 is the unpadded build"


# --- the same function ------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("impl", PADDED)
def test_padding_is_measured_bit_for_bit_or_within_the_bound(impl, dtype):
    """The padded network against the unpadded one, the same weights: every output within the
    bound of its arithmetic (the module docstring) - measured bit for bit on the development
    machine; a platform whose kernel sums the width in another order shows its gap here."""
    trunk_bound, head_bound = BOUNDS[dtype]
    for seed in (0, 1, 2):
        for padded in (False, True):
            if impl == "flex" and (padded or seed):
                continue  # the eager score function materialises every score: one small case
            batch = _batch(impl, seed, padded)
            a = _model(impl, dtype, 1, seed)(batch, return_trunk=True)
            b = _model(impl, dtype, 8, seed)(batch, return_trunk=True)
            gaps = _gap(a, b)
            assert gaps.pop("trunk") <= trunk_bound, (impl, dtype, seed, padded)
            assert max(gaps.values()) <= head_bound, (impl, dtype, seed, padded, {k: v for k, v in gaps.items() if v > 0})


@pytest.mark.grad
@pytest.mark.parametrize("impl", FAST)
def test_padding_leaves_the_gradients(impl):
    """Under autograd (where a learned bias sends the CPU kernel down its math path): the padded
    columns carry no gradient into any parameter - every gradient within ``test_chunked.py``'s
    bound of the unpadded one (measured: within 7.2e-7, not bit for bit - the module docstring)."""
    batch = _batch(impl, 2, padded=True)
    grads = []
    for pad in (1, 8):
        model = _model(impl, "fp32", pad, seed=2).train()
        out = model(batch)
        (out.value_logits.sum() + out.policy_logits[out.move_valid].sum() + out.wdl_logits.square().sum()).backward()
        grads.append({name: p.grad for name, p in model.named_parameters()})
    assert grads[0].keys() == grads[1].keys()
    for name, g in grads[0].items():
        assert (g is None) == (grads[1][name] is None), name
        if g is not None:
            tol = 1e-4 + 1e-5 * float(g.abs().max())
            assert torch.allclose(g, grads[1][name], atol=tol, rtol=0.0), (name, float((g - grads[1][name]).abs().max()))


# --- the counted cost --------------------------------------------------------------------------------


@pytest.mark.parametrize("impl", FAST)
def test_the_flop_counter_counts_the_padded_width(monkeypatch, impl):
    """``cost.count_flops`` at 8 exceeds the count at 1 by exactly what the widths predict: every
    kernel call's ``4 B H L_q L_k`` times the columns it gained (queries, keys and values are one
    width: 2 FLOPs a multiply-add in each of the two products). Nothing else in the forward moves."""
    batch = collate([synthetic_state(7, 5)])
    counted, calls = {}, {}
    for pad in (1, 8):
        model = _model(impl, "fp32", pad)
        _, counted[pad] = count_flops(lambda model=model: model(batch))
        spy = _Kernel(monkeypatch)
        model(batch)
        calls[pad] = list(spy.calls)
    assert len(calls[1]) == len(calls[8])
    predicted = 0
    for (q1, k1, _), (q8, _, _) in zip(calls[1], calls[8], strict=True):
        bsz, n_heads, lq, _ = q1
        predicted += 4 * bsz * n_heads * lq * k1[2] * (q8[3] - q1[3])
    assert predicted > 0 and counted[8] - counted[1] == predicted, (counted, predicted)


def test_the_one_block_path_s_counted_difference_in_closed_form():
    """On the one-block path each layer makes one call over the T - G body rows at 45 -> 48 and
    one over the 2 N register rows at 33 -> 40, each against the body keys and the merge's extra
    key: ``4 H (T - G + 1) (3 (T - G) + 7 x 2 N)`` FLOPs a layer more."""
    n, layers = 6, 3
    batch = collate([synthetic_state(8, n)])
    body, cfg = batch.body_len, NetConfig()
    counted = {}
    for pad in (1, 8):
        model = _model("sdpa", "fp32", pad, layers=layers)
        _, counted[pad] = count_flops(lambda model=model: model(batch))
    assert body == batch.n_tokens - cfg.global_regs == n * (64 + cfg.regs_per_board)
    assert counted[8] - counted[1] == layers * 4 * cfg.n_heads * (body + 1) * (3 * body + 7 * cfg.regs_per_board * n)


# --- the cached path ---------------------------------------------------------------------------------


def test_the_cached_path_pads_its_live_pass_and_computes_the_same_function(monkeypatch):
    """Under the arm the cached evaluation still equals the full pass (``test_cache.py``'s FP32
    bound, 2e-5 on the trunk): its live pass is padded like any forward. Its frozen-board passes
    build a context of their own (``cache.CachedEvaluator.compute_board``) that does not carry the
    field, so those calls stay at their as-built widths - 44 and 32: a frozen board reads no
    global register, so its calls have no merge - the same function, unpadded. Replace the last
    assertion (48 and 32) when ``cache.py`` passes ``pad_multiple`` on."""
    model = _model("block_sparse", "fp32", 8, attn_scope="block_causal")
    state = synthetic_state(5, 7)
    evaluator = CachedEvaluator(model)
    epoch = evaluator.root_epoch(state)
    spy = _Kernel(monkeypatch)
    cached = evaluator.evaluate(state, epoch)  # every frozen board is stored: the live pass alone
    assert set(spy.widths()) == {(48, 48, 48), (40, 40, 40)}, spy.widths()
    reference = reference_output(model, state)
    assert float((cached.trunk - reference.trunk).abs().max()) <= 2e-5
    spy.calls.clear()
    CachedEvaluator(model).root_epoch(state)  # the frozen boards alone
    assert set(spy.widths()) == {(44, 44, 44), (32, 32, 32)}, "not yet padded: cache.py's own context (see the docstring)"


# --- validation --------------------------------------------------------------------------------------


def test_the_multiple_is_validated():
    for good in (1, 2, 8, 16, 64):
        NetConfig(attn_pad_multiple=good).validate()
    for bad in (0, -8, 1.5, 8.0, "8", None, True):
        with pytest.raises(ValueError, match="attn_pad_multiple"):
            NetConfig(attn_pad_multiple=bad).validate()
    with pytest.raises(ValueError, match="attn_pad_multiple"):
        build_model(NetConfig(attn_pad_multiple=0))
