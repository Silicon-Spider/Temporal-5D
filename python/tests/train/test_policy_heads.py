"""Register D4's soft policy and opponent-reply prototype (spec section 12.4, CHECKPOINT 7) - and,
behind its flag, D9's optimistic policy (section 8.1) - and their losses on synthetic targets
(a few optimiser steps, never a training run)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_states
from t5d.train.losses import LossConfig, compute_losses, optimistic_weights, soft_policy_target
from t5d.train.smoke import smoke_train, synthetic_targets, with_actions

pytestmark = pytest.mark.grad


def _setup(seed: int = 3, **flags):
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, **flags)
    model = build_model(cfg, seed=seed)
    batch = collate(with_actions(synthetic_states(seed, [1, 3, 2, 4])))
    return cfg, model, batch, synthetic_targets(batch, cfg, torch.Generator().manual_seed(seed))


def test_the_soft_target_is_the_policy_target_to_the_quarter_power():
    """Section 12.4: "the policy target raised to the power 1/4 and renormalised" - the submit
    action's mass included (one distribution over the actions, D8); zeros stay zero."""
    t = torch.tensor([[0.81, 0.16, 0.0, 0.03], [0.0, 0.0, 0.0, 0.0]])
    soft = soft_policy_target(t, 0.25)
    expected = t[0] ** 0.25 / (t[0] ** 0.25).sum()
    assert torch.allclose(soft[0], expected) and float(soft[0, 2]) == 0.0
    assert torch.equal(soft[1], torch.zeros(4)), "no target, no soft target"
    assert float(soft[0, 0] / soft[0, 1]) == pytest.approx((0.81 / 0.16) ** 0.25), "flattened towards uniform"


def test_the_soft_loss_is_its_own_heads_cross_entropy_at_eight_times_the_policys_weight():
    """D4: the soft policy's loss is its own head's cross-entropy (moves and its own submit) against
    the flattened target over the full searches, weighted 8 against the policy's 1; it reaches the
    soft head's projections and submit readout and never the policy's own."""
    cfg, model, batch, tgt = _setup()
    weights = LossConfig().weights
    assert weights["soft_policy"] == 8.0 * weights["policy"] and LossConfig().soft_policy_power == 0.25
    out = model(batch)
    losses = compute_losses(out, tgt)
    full = tgt.full_search
    target = soft_policy_target(torch.cat([tgt.policy, tgt.submit[:, None]], -1))
    logp = torch.log_softmax(out.soft_action_logits().masked_fill(~out.action_valid(), float("-inf")), -1)
    per = -(target * logp.masked_fill(~out.action_valid(), 0.0)).sum(-1)
    assert torch.allclose(losses["soft_policy"], per[full].mean(), rtol=1e-5)
    total = sum(weights[k] * v for k, v in losses.items() if k != "total")
    assert torch.allclose(losses["total"], total, rtol=1e-6)
    losses["soft_policy"].backward()
    soft = model.aux.soft_policy
    for p in (soft.pointer.src.weight, soft.pointer.dst.weight, soft.submit.out.weight):
        assert float(p.grad.abs().sum()) > 0
    assert model.policy.src.weight.grad is None and model.submit.out.weight.grad is None
    model.zero_grad(set_to_none=True)
    cheap = compute_losses(model(batch), replace(tgt, full_search=torch.zeros(4, dtype=torch.bool)))
    assert float(cheap["soft_policy"].detach()) == 0.0, "cheap searches emit no policy targets (section 12.5)"


def test_the_reply_loss_exists_only_under_its_flag_at_weight_015():
    """D4's prototype: off, there is no reply head and no reply loss; on, its loss is the
    cross-entropy against the mapped turn's distribution, weighted 0.15 (KataGo's), and a position
    without a target (all zero) is skipped."""
    _, model, batch, tgt = _setup()
    assert "opp_reply" not in compute_losses(model(batch), tgt)
    assert LossConfig().weights["opp_reply"] == 0.15
    _, model, batch, tgt = _setup(opp_reply=True)
    out = model(batch)
    losses = compute_losses(out, tgt)
    has = tgt.opp_reply.sum(-1) > 0
    logp = torch.log_softmax(out.opp_reply_logits.masked_fill(~out.opp_move_valid, float("-inf")), -1).masked_fill(~out.opp_move_valid, 0)
    per = -(tgt.opp_reply * logp).sum(-1)
    assert torch.allclose(losses["opp_reply"], per[has].mean(), rtol=1e-5)
    none = tgt.opp_reply.clone()
    none[0] = 0.0
    partial = compute_losses(out, replace(tgt, opp_reply=none))
    keep = has.clone()
    keep[0] = False
    assert torch.allclose(partial["opp_reply"], per[keep].mean(), rtol=1e-5)


def test_a_few_optimiser_steps_reduce_the_soft_and_reply_losses():
    """At most three optimiser steps on synthetic targets: each of D4's losses, optimised alone,
    falls. Not training."""
    for name in ("soft_policy", "opp_reply"):
        cfg = NetConfig(trunk_dtype="fp32", n_layers=2, opp_reply=True)
        report = smoke_train(cfg, steps=3, seed=6, loss_cfg=LossConfig(weights={name: 1.0}))
        first, last = report.losses[0][name], report.losses[-1][name]
        assert last < first, (name, first, last)


# --- register D9: the optimistic policy (a prototype, behind its flag) ---------------------------------


def test_the_optimistic_weight():
    """Section 8.1 / D9: each sample weighted clamp(0, 1, sigmoid((z - 1.5) x 3)), z how far the
    mover's short-term outcome beat the network's value in units of its predicted error: 0.5 at
    z = 1.5, towards 1 above, towards 0 below; NaN without a short-term target; no gradient."""
    v = torch.zeros(5, requires_grad=True)
    err = torch.full((5,), 0.25, requires_grad=True)  # sqrt = 0.5
    st = torch.tensor([0.75, 3.0, -1.0, 0.0, math.nan])  # z = 1.5, 6, -2, 0, nan
    w = optimistic_weights(st, v, err, eps=0.0)
    assert float(w[0]) == pytest.approx(0.5) and float(w[1]) > 0.999 and float(w[2]) < 1e-4
    assert 0.0 <= float(w[3]) <= 0.02 and math.isnan(float(w[4]))
    assert not w.requires_grad


def test_the_optimistic_loss_exists_only_under_its_flag_and_trains_only_its_head():
    """D9: off, there is no optimistic head and no loss; on, the loss is the weighted cross-entropy
    of its own actions against the normal policy target over the full searches (the weighted sum
    over their number), reaching its own projections and submit readout - never the value, the
    short-term or the error heads through the weight."""
    _, model, batch, tgt = _setup()
    assert model.optimistic is None and "optimistic" not in compute_losses(model(batch), tgt)
    cfg, model, batch, tgt = _setup(optimistic_policy=True)
    out = model(batch)
    losses = compute_losses(out, tgt)
    lc = LossConfig()
    w = optimistic_weights(tgt.short_term[:, lc.error_horizon], out.search_value(), out.value_error)
    target = torch.cat([tgt.policy, tgt.submit[:, None]], -1)
    valid = out.action_valid()
    logp = torch.log_softmax(out.optimistic_action_logits().masked_fill(~valid, float("-inf")), -1).masked_fill(~valid, 0.0)
    per = -(target * logp).sum(-1)
    keep = tgt.full_search & torch.isfinite(w)
    assert torch.allclose(losses["optimistic"], (w[keep] * per[keep]).sum() / keep.sum(), rtol=1e-5)
    losses["optimistic"].backward()
    head = model.optimistic
    assert float(head.pointer.src.weight.grad.abs().sum()) > 0 and float(head.submit.out.weight.grad.abs().sum()) > 0
    for m in (model.value.wdl, model.value.short_term, model.value.error, model.value.categorical, model.policy):
        assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in m.parameters()), m


def test_a_few_optimiser_steps_reduce_the_optimistic_loss():
    """At most three optimiser steps on synthetic targets (short-term targets above the network's
    value, so the samples weigh): the loss, optimised alone, falls. Not training."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, optimistic_policy=True)
    report = smoke_train(cfg, steps=3, seed=6, loss_cfg=LossConfig(weights={"optimistic": 1.0}, optimistic_centre=-10.0))
    first, last = report.losses[0]["optimistic"], report.losses[-1]["optimistic"]
    assert last < first, (first, last)
