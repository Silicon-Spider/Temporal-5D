"""G8's QAT hook in M21's loop (``train.trainer.apply_qat``; register G8, M21, G2; spec section
10.2): off by default; on, the trainer's network is G2's FP8 trunk - its forward the served
format's full pass bit for bit (the matching rule; the frozen K/V rounded as the full pass rounds
them), its gradients straight through the rounding to every quantised weight, its records naming
the format - smoke-tested with a few optimiser steps on synthetic rows only (the user's rule;
issue 237(a)).

Measured for the gradient bound (2026-10-23, CPU, torch 2.13): over model seeds 0-2 on
``_synthetic_rows(0)``, the QAT network's gradient on each trunk projection weight against the
unquantised network's (same weights): relative L2 distance at most 0.116 under the dense scope
(FP32) and 0.117 under the block-causal scope with FP8 frozen K/V, norm ratios 0.97-1.03. With the
estimator the STE replaced - autograd through the cast, C15's defect - every one of those gradients
is zero (distance 1, ratio 0). Bound 0.5 on the distance, 0.5-2 on the ratio (MEASUREMENTS, "G2's
FP8 emulation").
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model
from t5d.net.quant import QuantLinear, trunk_format
from t5d.train import trainer as trainer_module
from t5d.train.losses import compute_losses
from t5d.train.records import network_id
from t5d.train.selfplay import GeneratorConfig, generate_game, network_info
from t5d.train.smoke import optimiser_steps
from t5d.train.targets import collate_rows
from t5d.train.trainer import Trainer, TrainerConfig, apply_qat
from tests.engine.helpers import requires_engine
from tests.train.test_trainer import GEN, _config, _synthetic_rows

pytestmark = pytest.mark.grad

NET = NetConfig(**SMALL)


def _projection_weights(model) -> dict[str, torch.nn.Parameter]:
    return {name: m.weight for name, m in model.named_modules() if isinstance(m, QuantLinear)}


def test_qat_is_off_by_default_and_switches_the_network_in_place():
    assert TrainerConfig().qat is False and TrainerConfig().qat_format == "fp8_e4m3"
    model = build_model(NET, seed=0)
    plain = Trainer(model, NET, _config())
    assert plain.model is model and model.cfg.trunk_quant == "none"
    assert all(m.fmt == "none" for m in model.modules() if isinstance(m, QuantLinear))
    trainer = Trainer(model, NET, _config(qat=True))
    assert trainer.model is model, "in place: the network's step count stays with it (issue 237(a))"
    assert model.cfg.trunk_quant == "fp8_e4m3" and all(m.fmt == "fp8_e4m3" for m in model.modules() if isinstance(m, QuantLinear))
    assert len(_projection_weights(model)) == 6 * NET.n_layers
    other = build_model(NET, seed=0)
    assert apply_qat(other, TrainerConfig(qat=True)) is other and other.cfg.trunk_quant == "fp8_e4m3"
    assert apply_qat(other, TrainerConfig(qat=True)) is other, "already in the format: unchanged"
    for bad in ("none", "int8"):
        with pytest.raises(ValueError, match="qat_format"):
            TrainerConfig(qat=True, qat_format=bad).validate()


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_qat_smoke(dtype):
    """Three optimiser steps on synthetic rows with QAT on: every loss finite, a gradient on every
    quantised weight - dense, not flushed - and every trunk parameter moved (the heads whose
    targets the synthetic records leave out - the masked per-square labels, a submit never offered
    - move under neither format)."""
    _, _, rows = _synthetic_rows(0)
    net = replace(NET, trunk_dtype=dtype)
    model = build_model(net, seed=1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainer = Trainer(model, net, _config(qat=True))
    for _ in range(3):
        losses = trainer.step_on(rows)
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
        for name, w in _projection_weights(model).items():
            assert w.grad is not None and torch.isfinite(w.grad).all(), name
            assert float((w.grad != 0).float().mean()) > 0.99, (name, "the straight-through gradient reaches every weight")
    assert optimiser_steps(model) == 3
    unchanged = [n for n, p in model.named_parameters() if n.startswith(("trunk.", "embed.")) and torch.equal(p.detach(), before[n])]
    assert not unchanged, unchanged
    with pytest.raises(RuntimeError, match="no training runs"):
        trainer.step_on(rows)


SETTINGS = dict(trunk_quant_attention=False, trunk_quant_scale="absmax", trunk_quant_weight_scale="tensor")


@pytest.mark.parametrize(
    "flags",
    [
        dict(),
        dict(attn_scope="block_causal", frozen_kv_format="fp8"),
        dict(trunk_dtype="bf16", attn_impl="chunked"),
        dict(attn_scope="block_causal", frozen_kv_format="fp8", **SETTINGS),
    ],
    ids=["dense", "block_causal_fp8_kv", "bf16_chunked", "other_settings"],
)
def test_qat_forward_is_the_inference_format(flags):
    """The matching rule (section 10.2: "QAT simulates exactly the deployed format, K/V cache
    included"): the QAT network in training mode and G2's network in evaluation mode, the same
    weights, give the same outputs bit for bit - under the block-causal scope with FP8 frozen K/V
    too (C15's format, kept by the hook: the full pass's rounding of the frozen K/V, which is the
    cache's; the cache itself serves only the projections-only case, issue 239(a)), and with the
    network's own quantisation settings, which the hook keeps. Both under autograd: the CPU
    attention kernel differs with and without it, and FP8 turns its last-bit differences into rare
    whole-step flips."""
    _, _, rows = _synthetic_rows(1)
    net = replace(NET, **flags)
    model = build_model(net, seed=2)
    served = model.with_trunk_quant("fp8_e4m3").eval()
    trainer = Trainer(model, net, _config(qat=True))
    assert model.cfg == served.cfg and model.cfg.frozen_kv_format == net.frozen_kv_format
    for name in SETTINGS:
        assert getattr(model.cfg, name) == getattr(net, name), (name, "the network's setting kept")
    assert trunk_format(trainer.net_cfg) == trunk_format(model.cfg) == ("fp8_e4m3", *(getattr(net, n) for n in SETTINGS))
    batch, _ = collate_rows(rows, net)
    a = model.train()(batch, return_trunk=True)
    b = served(batch, return_trunk=True)
    for name in ("trunk", "policy_logits", "wdl_logits", "value_logits", "submit_raw"):
        assert torch.equal(getattr(a, name).detach(), getattr(b, name).detach()), name


@pytest.mark.parametrize("flags", [dict(), dict(attn_scope="block_causal", frozen_kv_format="fp8")], ids=["dense", "fp8_kv"])
def test_qat_gradients_follow_the_unquantised_ones(flags):
    """The straight-through estimator end to end: each projection's gradient under QAT is the
    unquantised network's up to FP8-level differences (the module docstring) - including the K and
    V projections of the frozen boards read through C15's FP8 format, whose gradient the flushing
    cast zeroed."""
    _, _, rows = _synthetic_rows(0)
    net = replace(NET, **flags)
    batch, targets = collate_rows(rows, net)
    grads = []
    for qat in (False, True):
        model = build_model(net, seed=0)
        if qat:
            apply_qat(model, TrainerConfig(qat=True))
        compute_losses(model.train()(batch), targets)["total"].backward()
        grads.append({n: w.grad for n, w in _projection_weights(model).items()})
    for name, g0 in grads[0].items():
        g1 = grads[1][name]
        distance = float((g1 - g0).norm() / g0.norm())
        ratio = float(g1.norm() / g0.norm())
        assert distance < 0.5 and 0.5 < ratio < 2, (name, distance, ratio)


@pytest.mark.parametrize("ffn_ratio", [2, 12])
def test_qat_with_other_ffn_ratios(ffn_ratio):
    """F1's FFN ratio is a parameter (ablation 5 tests 2x; F2's arm is 12x): QAT takes any."""
    _, _, rows = _synthetic_rows(2)
    net = replace(NET, ffn_ratio=ffn_ratio)
    model = build_model(net, seed=3)
    trainer = Trainer(model, net, _config(qat=True))
    assert model.trunk.blocks[0].ffn.lin_in.out_features == ffn_ratio * net.d_model
    losses = trainer.step_on(rows)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    w = model.trunk.blocks[0].ffn.lin_in.weight
    assert w.grad is not None and float((w.grad != 0).float().mean()) > 0.99


def test_qat_refuses_a_network_served_in_another_format(monkeypatch):
    """QAT simulates exactly the format the network is served in: a network already in another
    8-bit format is refused, not re-quantised. The hook validates its configuration first, so a
    format G2 does not build is refused there; the refusal proper is checked with a second 8-bit
    format made known to the configuration (only E4M3 is built)."""
    model = build_model(NET, seed=0).set_trunk_quant("fp8_e4m3")
    with pytest.raises(ValueError, match="qat_format"):
        apply_qat(model, TrainerConfig(qat=True, qat_format="fp8_e5m2"))
    monkeypatch.setattr(trainer_module, "TRUNK_QUANT_FORMATS", ("none", "fp8_e4m3", "fp8_e5m2"))
    TrainerConfig(qat=True, qat_format="fp8_e5m2").validate()
    with pytest.raises(ValueError, match="exactly the deployed format"):
        apply_qat(model, TrainerConfig(qat=True, qat_format="fp8_e5m2"))
    assert model.cfg.trunk_quant == "fp8_e4m3"


def test_the_network_id_names_the_format():
    """The same weights in FP8 compute another function: another network id - and a network in
    "none" hashes its weights alone, so switching back gives the old id (existing ids unchanged)."""
    model = build_model(NET, seed=0)
    plain = network_id(model)
    twin = model.with_trunk_quant("fp8_e4m3")
    assert network_id(twin) != plain
    assert network_id(model.with_trunk_quant("fp8_e4m3", trunk_quant_scale="absmax")) not in (plain, network_id(twin))
    assert network_id(twin.set_trunk_quant("none")) == plain


@requires_engine
def test_the_records_of_a_qat_network_name_its_format():
    """The miniature loop under QAT: steps on synthetic rows, SWA's network published (in QAT's
    format), self-play with it - and its records carry the format: the configuration's
    ``trunk_quant*`` the network's own, its id not the unquantised network's. A configuration that
    contradicts the evaluated network's format is refused, not recorded."""
    torch.manual_seed(0)
    model = build_model(NET, seed=0)
    trainer = Trainer(model, NET, _config(qat=True))
    assert NET.trunk_quant == "none" and trainer.net_cfg.trunk_quant == "fp8_e4m3", "the trainer's configuration follows"
    _, _, rows = _synthetic_rows(6)
    trainer.step_on(rows)
    published = trainer.publish()
    assert published.cfg.trunk_quant == "fp8_e4m3" and published.swa_snapshots == 1
    evaluator = NetworkEvaluator(published, max_batch=8)
    with torch.no_grad():
        g = generate_game(evaluator, GeneratorConfig(seed=35, **GEN), net_cfg=trainer.net_cfg)
    network = g.record.header["network"]
    assert network["config"]["trunk_quant"] == "fp8_e4m3"
    assert all(network["config"][f] == getattr(published.cfg, f) for f in SETTINGS)
    assert network["id"] == network_id(published) != network_id(published.with_trunk_quant("none"))
    with pytest.raises(ValueError, match="trunk format"):
        network_info(evaluator, NET)
