"""Register D3's value heads (spec section 8.3, as corrected at CHECKPOINT 7) and their losses:
HL-Gauss targets, the WDL target and its q_ratio blend, the categorical on full searches only,
the error head's stop-gradient target, and a few optimiser steps reducing each loss on synthetic
targets (never a training run)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from t5d.net import NetConfig, build_model, collate, synthetic_states
from t5d.net.heads import categorical_mean, hl_gauss, value_bin_centres
from t5d.train.losses import LossConfig, compute_losses, error_target, wdl_target
from t5d.train.smoke import smoke_train, synthetic_targets, with_actions

pytestmark = pytest.mark.grad

SIGMA = 0.75 * 2 / 50  # the placeholder: three quarters of a bin width (Farebrother et al.)


def test_hl_gauss_is_a_distribution_centred_on_the_value():
    v = torch.tensor([-0.6, -0.37, 0.0, 0.123, 0.5])
    target = hl_gauss(v, 51, SIGMA)
    assert target.shape == (5, 51) and target.dtype == torch.float32
    assert torch.allclose(target.sum(-1), torch.ones(5), atol=1e-6)
    assert torch.allclose(categorical_mean(target), v, atol=1e-4), "away from the edges its mean is the value"
    centres = value_bin_centres(51)
    nearest = (centres[None] - v[:, None]).abs().argmin(-1)
    assert torch.equal(target.argmax(-1), nearest), "its mode is the nearest centre"
    assert int((target > 1e-3).sum(-1).max()) <= 6, "smoothed into neighbouring bins only"


def test_hl_gauss_limits_and_edges():
    # A tiny sigma puts (almost) everything in the value's bin; a value on a centre is one-hot.
    sharp = hl_gauss(torch.tensor([0.0, 0.52]), 51, 1e-4)
    assert float(sharp[0, 25]) == pytest.approx(1.0, abs=1e-6) and float(sharp[1, 38]) == pytest.approx(1.0, abs=1e-6)
    # At +-1 the mass outside the support is dropped and the rest renormalised.
    edge = hl_gauss(torch.tensor([1.0, -1.0]), 51, SIGMA)
    assert torch.allclose(edge.sum(-1), torch.ones(2), atol=1e-6)
    assert float(edge[0, -1]) > 0.5 and float(edge[1, 0]) > 0.5
    m = categorical_mean(edge)
    assert 0.98 < float(m[0]) < 1.0 and -1.0 < float(m[1]) < -0.98, "pulled inside by the truncation"
    assert torch.isnan(hl_gauss(torch.tensor([math.nan]), 51, SIGMA)).all(), "no value, no target"


def test_the_wdl_target_is_the_result_at_q_ratio_zero():
    """D3: WDL on the game result; Lc0's q_ratio blend towards the search's own WDL is "a tuned
    knob, default zero" (6z) - at zero the target is exactly the result."""
    cfg = NetConfig(trunk_dtype="fp32")
    batch = collate(synthetic_states(2, [1, 2, 3]))
    tgt = synthetic_targets(batch, cfg, torch.Generator().manual_seed(2))
    assert LossConfig().q_ratio == 0.0
    exact = wdl_target(tgt, 0.0)
    assert torch.equal(exact, torch.nn.functional.one_hot(tgt.wdl, 3).float())
    blended = wdl_target(tgt, 0.25)
    assert torch.allclose(blended, 0.75 * exact + 0.25 * tgt.root_wdl)
    assert torch.allclose(blended.sum(-1), torch.ones(3))
    distribution = torch.tensor([[0.5, 0.25, 0.25]]).expand(3, 3)  # I13's deblundered targets are distributions
    assert torch.equal(wdl_target(replace(tgt, wdl=distribution), 0.0), distribution)
    with pytest.raises(ValueError, match="root_wdl"):
        wdl_target(replace(tgt, root_wdl=None), 0.5)


def _setup(seed: int = 3):
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2)
    model = build_model(cfg, seed=seed)
    batch = collate(with_actions(synthetic_states(seed, [1, 3, 2, 4])))
    return cfg, model, batch, synthetic_targets(batch, cfg, torch.Generator().manual_seed(seed))


def test_the_categorical_trains_on_full_searches_only():
    """D3: the categorical is "over the root search value of full searches" - a cheap search
    (section 12.5) gives it no target and no gradient; the loss is the cross-entropy against the
    HL-Gauss target at the placeholder sigma."""
    cfg, model, batch, tgt = _setup()
    out = model(batch)
    losses = compute_losses(out, tgt)
    keep = tgt.full_search & torch.isfinite(tgt.root_value)
    assert keep.tolist() == [True, True, False, True]
    target = hl_gauss(tgt.root_value[keep], 51, SIGMA)
    expected = -(target * torch.log_softmax(out.value_logits[keep], -1)).sum(-1).mean()
    assert torch.allclose(losses["categorical"], expected, rtol=1e-5)
    cheap = replace(tgt, full_search=torch.zeros(4, dtype=torch.bool))
    losses = compute_losses(model(batch), cheap)
    assert float(losses["categorical"].detach()) == 0.0 and float(losses["policy"].detach()) == 0.0
    model.zero_grad(set_to_none=True)
    losses["categorical"].backward()
    grad = model.value.categorical.weight.grad
    assert grad is None or float(grad.abs().sum()) == 0.0


def test_no_gradient_flows_from_the_error_target_into_the_value_or_short_term_heads():
    """D3's error head predicts the squared gap between the network's short-term value and its
    target (register D9's "predicted short-term value" and KataGo's form; spec issue 231 records
    the W - L reading as the alternative); its target takes the network's short-term output with
    no gradient - the error loss moves the error head and the shared layer, never the WDL or
    short-term outputs."""
    cfg, model, batch, tgt = _setup(4)
    out = model(batch)
    err = error_target(out.short_term, tgt.short_term, LossConfig().error_horizon)
    assert torch.allclose(err, (out.short_term[:, 0].detach() - tgt.short_term[:, 0]) ** 2)
    assert not torch.allclose(err, (tgt.short_term[:, 0] - out.search_value().detach()) ** 2), "not the W - L reading"
    assert not err.requires_grad
    compute_losses(out, tgt)["value_error"].backward()
    for head in (model.value.wdl, model.value.short_term, model.value.categorical):
        assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in head.parameters()), head
    assert float(model.value.error.weight.grad.abs().sum()) > 0 and float(model.value.hidden.weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match="error_horizon"):
        error_target(out.short_term, tgt.short_term, 3)


def test_the_error_loss_is_huber_against_the_built_target_over_the_rows_that_have_one():
    """The error head's loss is the Huber loss (the placeholder delta) of its output against
    ``(short-term output - short-term target)^2`` at the error head's horizon - any horizon -
    averaged over the positions whose target there is finite."""
    cfg, model, batch, tgt = _setup(7)
    st = tgt.short_term.clone()
    st[2, 0] = math.nan  # (the synthetic targets already leave the last position's last horizon out)
    tgt = replace(tgt, short_term=st)
    out = model(batch)
    for h in (0, 2):
        lc = LossConfig(error_horizon=h)
        built = (out.short_term[:, h].detach() - st[:, h]) ** 2
        has = torch.isfinite(built)
        assert int(has.sum()) == 3, h
        expected = F.huber_loss(out.value_error[has], built[has], delta=lc.error_huber_delta)
        assert torch.allclose(compute_losses(out, tgt, lc)["value_error"], expected, rtol=1e-6), h


def test_missing_short_term_targets_are_skipped():
    """A missing short-term target (NaN) contributes nothing: the short-term loss is the mean
    squared error over the finite entries only, a position with none gets no gradient, and the
    error loss is zero where its horizon has a target nowhere."""
    cfg, model, batch, tgt = _setup(5)
    st = tgt.short_term.clone()
    st[:, 0] = math.nan
    st[1, :] = math.nan
    out = model(batch)
    losses = compute_losses(out, replace(tgt, short_term=st))
    finite = torch.isfinite(st)
    expected = ((out.short_term - st) ** 2)[finite].mean()
    assert torch.allclose(losses["short_term"], expected, rtol=1e-6) and torch.isfinite(losses["total"])
    (grad,) = torch.autograd.grad(losses["short_term"], out.short_term, retain_graph=True)
    assert float(grad[~finite].abs().sum()) == 0.0 and float(grad[1].abs().sum()) == 0.0, "no target, no gradient"
    assert float(grad[finite].abs().min()) > 0
    assert float(losses["value_error"].detach()) == 0.0, "the error head's horizon has no target anywhere"


def test_a_few_optimiser_steps_reduce_each_value_loss():
    """Section 15's build rule: at most three optimiser steps on synthetic targets - each of D3's
    losses, optimised alone, falls (AdamW). Not training. The error head's target moves with the
    network's own value, so its loss is stepped on the error head's parameters alone (its target
    then fixed)."""
    for name in ("wdl", "categorical", "short_term"):
        report = smoke_train(
            NetConfig(trunk_dtype="fp32", n_layers=2), steps=3, seed=6, loss_cfg=LossConfig(weights={name: 1.0, "flat_policy": 0.0})
        )
        first, last = report.losses[0][name], report.losses[-1][name]
        assert last < first, (name, first, last)
    cfg, model, batch, tgt = _setup(6)
    opt = torch.optim.AdamW(model.value.error.parameters(), lr=1e-2, weight_decay=0.0)
    seen = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = compute_losses(model(batch), tgt)["value_error"]
        seen.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert seen[-1] < seen[0], seen


def test_the_effort_loss_reads_full_searches_only():
    """Section 12.4's search-effort target is "full-search moves only": a cheap search's row is
    skipped even where its effort target is given."""
    cfg, model, batch, tgt = _setup(8)
    out = model(batch)
    keep = (tgt.effort >= 0) & tgt.full_search
    assert keep.tolist() == [True, True, False, True] and bool((tgt.effort >= 0).all())
    expected = F.cross_entropy(out.effort_logits[keep], tgt.effort[keep])
    assert torch.allclose(compute_losses(out, tgt)["effort"], expected, rtol=1e-6)
    cheap = compute_losses(out, replace(tgt, full_search=torch.zeros(4, dtype=torch.bool)))
    assert float(cheap["effort"].detach()) == 0.0
