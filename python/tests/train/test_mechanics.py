"""Section 12.8's training-data mechanics and section 12.5's generation scheme (register M21):
I12's sampler with 6u's arms, I16's replay window, I17's SWA, and in the generator I5's playout
cap randomisation, I20's KL-adaptive full searches, I18's resignation with the false-resign check
and I19's temperature schedules. Tiny games with random or synthetic evaluators - nothing is
trained."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import Evaluation, NetworkEvaluator, UniformEvaluator, move_count, state_of
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model
from t5d.train.sampler import BucketTracker, SamplerConfig, copies, row_weights
from t5d.train.selfplay import (
    GeneratorConfig,
    KLAdaptiveConfig,
    PCRConfig,
    ResignConfig,
    false_resign_report,
    generate_game,
    is_full_search,
    move_temperature,
    resign_enabled,
    root_temperature,
)
from t5d.train.swa import SWA, SWAConfig
from t5d.train.targets import TargetConfig, build_targets
from t5d.train.window import ReplayWindow, WindowConfig, window_rows
from tests.engine.helpers import requires_engine
from tests.train.test_targets import _children, _decision, _game, _line

NET = NetConfig(**SMALL)

# --- I12 and 6u ------------------------------------------------------------------------------------------


def _surprise_game(kls):
    """A game whose full-search rows have the given KL divergences (set directly on the targets)."""
    g = _line(len(kls), end="checkmate", winner=0)
    t = build_targets(g, TargetConfig(deblunder=False))
    t.kl = np.array(kls, dtype=np.float64)
    return g, t


def test_surprise_weights_have_mean_one_and_follow_the_kl():
    g, t = _surprise_game([0.1, 0.1, 0.1, 0.1])
    w = row_weights(g, t, SamplerConfig(value_share=0.0))
    assert w == pytest.approx(np.ones(4))  # equal surprises: uniform
    g, t = _surprise_game([0.0, 0.0, 0.0, 1.0])
    w = row_weights(g, t, SamplerConfig(value_share=0.0))
    assert w.mean() == pytest.approx(1.0)
    assert w.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5 + 0.5 * 4])  # half uniform, half in proportion to the KL
    g, t = _surprise_game([0.0, 0.0, 0.0, 0.0])
    assert row_weights(g, t).tolist() == pytest.approx([1.0] * 4)  # no surprise anywhere: the share spreads uniformly
    w = row_weights(g, t, SamplerConfig())
    assert w.mean() == pytest.approx(1.0)


def test_value_surprise_and_cheap_rows():
    ds = [_decision(i, _children([10, 5], [0.0, -0.1], [0.5, 0.5]), full=i != 1) for i in range(3)]
    ds[2] = replace(ds[2], raw_wdl_logits=np.array([-4.0, 0.0, 4.0]))  # the network was sure it lost; it won
    g = _game(ds, end="checkmate", winner=0)
    t = build_targets(g, TargetConfig(deblunder=False))
    w = row_weights(g, t, SamplerConfig(policy_share=0.0, value_share=0.5, cheap_weight=0.25))
    assert w[1] == 0.25  # a cheap search's row (I5)
    assert w[2] > w[0] and (w[0] + w[2]) / 2 == pytest.approx(1.0)


def test_pruned_targets_keep_steering_out_of_the_surprise():
    """I12: the divergence is measured against the pruned target (I2), so visits a steering source
    added to a child do not read as surprise."""
    plain = _decision(0, _children([40, 30, 30], [0.0, 0.0, 0.0], [0.4, 0.3, 0.3]), raw_logits=np.log([0.4, 0.3, 0.3]))
    steered = replace(plain, children=_children([40, 30, 70], [0.0, 0.0, 0.0], [0.4, 0.3, 0.3], steered=[0, 0, 40]))
    cfg = TargetConfig(deblunder=False, forced_k=0.0)
    a = build_targets(_game([plain]), cfg).kl[0]
    b = build_targets(_game([steered]), cfg).kl[0]
    assert a == b
    assert build_targets(_game([steered]), replace(cfg, policy_mode="provenance", provenance_weight=1.0)).kl[0] > a


def test_copies_are_seeded_and_unbiased():
    w = np.array([0.0, 1.0, 1.5, 2.25, 0.3] * 200)
    a, b = copies(w, 7), copies(w, 7)
    assert np.array_equal(a, b) and not np.array_equal(a, copies(w, 8))
    assert ((a == np.floor(w)) | (a == np.floor(w) + 1)).all()
    assert a.mean() == pytest.approx(w.mean(), abs=0.05)


def test_the_6u_arms():
    g = _line(4, end="checkmate", winner=1)
    t = build_targets(g, TargetConfig(deblunder=False))
    assert row_weights(g, t, SamplerConfig(arm="uniform")).tolist() == [1.0] * 4
    g2 = _game([replace(d, raw_value_logits=np.linspace(-1, 1, 51) * (i + 1)) for i, d in enumerate(g.decisions)])
    w = row_weights(g2, t, SamplerConfig(arm="raw_variance"))
    assert w.mean() == pytest.approx(1.0) and not np.allclose(w, 1.0)
    with pytest.raises(ValueError, match="BucketTracker"):
        row_weights(g, t, SamplerConfig(arm="persistent_bucket"))
    cfg = SamplerConfig(arm="persistent_bucket", persist_checkpoints=2, bucket_threshold=0.1)
    tracker = BucketTracker(cfg)
    sure = _game([replace(d, raw_wdl_logits=np.array([5.0, 0.0, -5.0])) for d in g.decisions], end="checkmate", winner=1)
    st = build_targets(sure, TargetConfig(deblunder=False))
    tracker.add_checkpoint([(sure, st)])
    assert row_weights(sure, st, cfg, tracker).tolist() == [1.0] * 4  # one checkpoint: not persistent yet
    tracker.add_checkpoint([(sure, st)])
    w = row_weights(sure, st, cfg, tracker)
    assert w.mean() == pytest.approx(1.0)  # every bucket triggered: still mean 1
    with pytest.raises(ValueError):
        SamplerConfig(arm="realised_error").validate()  # I6's realised-error arm is superseded by I12


# --- I16 ----------------------------------------------------------------------------------------------------


def test_the_window_grows_sublinearly():
    cfg = WindowConfig(min_rows=1000, exponent=0.75, expand=0.4)
    assert [window_rows(n, cfg) for n in (0, 10, 1000)] == [0, 10, 1000]
    sizes = [window_rows(n, cfg) for n in (2000, 10_000, 100_000, 1_000_000)]
    assert sizes == sorted(sizes) and all(s <= n for s, n in zip(sizes, (2000, 10_000, 100_000, 1_000_000), strict=True))
    ratios = [s / n for s, n in zip(sizes, (2000, 10_000, 100_000, 1_000_000), strict=True)]
    assert ratios == sorted(ratios, reverse=True) and ratios[-1] < 0.1
    # KataGo's constants: a 250k-row window grows to about 22M rows by about 225M rows written.
    assert 21.5e6 < window_rows(224_750_000) < 22.5e6
    with pytest.raises(ValueError):
        WindowConfig(exponent=1.0).validate()


def test_the_window_keeps_the_latest_rows():
    w = ReplayWindow(WindowConfig(min_rows=10, exponent=0.5, expand=0.1))
    for i in range(100):
        w.add(i, copies=2 if i % 10 == 0 else 1)
    assert len(w) == window_rows(w.written, w.cfg) and w.written == 110
    assert w.rows()[-1] == 99 and min(w.rows()) > 80
    drawn = w.sample(50, np.random.default_rng(0))
    assert set(drawn) <= set(w.rows())


# --- I17 -----------------------------------------------------------------------------------------------------


def test_swa_is_the_capped_running_mean():
    model = torch.nn.Linear(3, 2)
    swa = SWA(model, SWAConfig(every_steps=2, max_snapshots=3))
    seen = []
    for step in range(1, 13):
        with torch.no_grad():
            model.weight.fill_(float(step))
        if swa.update(model, step):
            seen.append(float(step))
    assert seen == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] and swa.snapshots == 6
    avg = 0.0
    for k, x in enumerate(seen, start=1):
        avg = x if k == 1 else avg + (x - avg) / min(k, 3)
    published = swa.publish(model)
    assert torch.allclose(published.weight, torch.full((2, 3), avg)) and published is not model
    assert float(model.weight[0, 0]) == 12.0  # the raw weights untouched
    off = SWA(model, SWAConfig(enabled=False))
    assert not off.update(model, 100) and off.publish(model) is model
    with pytest.raises(ValueError, match="BatchNorm"):
        SWA(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)))


def test_swa_first_three_is_the_plain_mean():
    model = torch.nn.Linear(2, 1, bias=False)
    swa = SWA(model, SWAConfig(every_steps=1, max_snapshots=10))
    for step, x in enumerate((1.0, 2.0, 6.0), start=1):
        with torch.no_grad():
            model.weight.fill_(x)
        swa.update(model, step)
    assert torch.allclose(swa.publish(model).weight, torch.full((1, 2), 3.0))


# --- I19 -------------------------------------------------------------------------------------------------------


def test_the_temperature_schedules():
    cfg = GeneratorConfig()
    assert move_temperature(cfg, 0) == 0.9 and move_temperature(cfg, 30) == pytest.approx(0.675)
    assert move_temperature(cfg, 59) == pytest.approx(0.9 - 0.45 * 59 / 60) and move_temperature(cfg, 60) == 0.0
    assert (
        root_temperature(cfg, 0) == 1.25
        and root_temperature(cfg, 60) == pytest.approx(1.1)
        and root_temperature(cfg, 600) == pytest.approx(1.1)
    )


# --- I5, I20, I18 in the generator ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluator():
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


def test_pcr_draws_are_seeded_and_near_the_fraction():
    cfg = GeneratorConfig(seed=3, pcr=PCRConfig(full_fraction=0.25))
    draws = [is_full_search(cfg, d) for d in range(4000)]
    assert draws == [is_full_search(cfg, d) for d in range(4000)]
    assert abs(sum(draws) / 4000 - 0.25) < 0.03
    assert all(is_full_search(replace(cfg, pcr=PCRConfig(enabled=False)), d) for d in range(50))
    with pytest.raises(ValueError, match="mixture"):
        GeneratorConfig(pcr=PCRConfig(difficulty_mix=1.0)).validate()
    with pytest.raises(NotImplementedError, match="236"):
        GeneratorConfig(pcr=PCRConfig(difficulty_mix=0.5)).validate()


@requires_engine
def test_pcr_in_a_game(evaluator):
    cfg = GeneratorConfig(
        playouts=12, batch_size=4, cap_turns=3, cap_boards=20, seed=21, pcr=PCRConfig(full_fraction=0.5, cheap_playouts=3)
    )
    with torch.no_grad():
        g = generate_game(evaluator, cfg, net_cfg=NET)
    ds = g.record.decisions
    assert any(d.full_search for d in ds) and any(not d.full_search for d in ds)
    for d in ds:
        assert d.full_search == is_full_search(cfg, d.decision)  # re-derived from the seed
        assert d.playout_cap == (12 if d.full_search else 3) and d.root_noise == d.full_search  # cheap: no noise
        assert d.playouts <= d.playout_cap or d.verified_illegal > 0
    assert set(g.record.header["search"]) == {"full", "cheap"}
    off = replace(cfg, pcr=PCRConfig(enabled=False))
    with torch.no_grad():
        g2 = generate_game(evaluator, off, net_cfg=NET)
    assert all(d.full_search and d.playout_cap == 12 for d in g2.record.decisions)


@requires_engine
def test_kl_adaptive_full_searches_stop_on_a_quiet_root():
    """I20 (a prototype, off by default): a full search stops once the root's visit distribution
    stops moving - with the uniform evaluator, well before its cap; cheap searches are not chunked."""
    ev = UniformEvaluator()
    base = GeneratorConfig(
        playouts=400, batch_size=8, cap_turns=1, cap_boards=10, seed=5, pcr=PCRConfig(full_fraction=0.5, cheap_playouts=4)
    )
    cfg = replace(base, kl=KLAdaptiveConfig(enabled=True, interval=16, min_playouts=32, min_gain=1e-3))
    g = generate_game(ev, cfg, net_cfg=NET)
    full = [d for d in g.record.decisions if d.full_search and d.move_count > 1]
    assert full and all(d.stop == "kl" and d.playouts < 400 for d in full)
    assert all(d.stop != "kl" for d in g.record.decisions if not d.full_search)
    t = build_targets(g.record, TargetConfig(short_term_horizons=NET.short_term_horizons))
    assert all(t.effort[d.decision] < 15 for d in full)  # I20's stop is convergence (D4's effort label)
    g0 = generate_game(ev, base, net_cfg=NET)
    assert all(d.stop != "kl" and d.playouts == d.playout_cap for d in g0.record.decisions if d.full_search and d.move_count > 1)


class _Doomed:
    """Every position looks lost to White and won by Black (synthetic values for I18's tests),
    from the acting player's view."""

    def __call__(self, positions):
        out = []
        for p in positions:
            wdl = [-6.0, 0.0, 6.0] if state_of(p).acting_player == 0 else [6.0, 0.0, -6.0]
            out.append(Evaluation(np.zeros(move_count(p), dtype=np.float32), np.array(wdl, dtype=np.float32)))
        return out


@requires_engine
def test_resignation_and_its_check():
    resign = ResignConfig(threshold=0.5, consecutive=2, disabled_fraction=0.0)
    cfg = GeneratorConfig(playouts=8, batch_size=4, cap_turns=4, cap_boards=30, seed=2, pcr=PCRConfig(enabled=False), resign=resign)
    assert resign_enabled(cfg)
    g = generate_game(_Doomed(), cfg, net_cfg=NET)
    rec = g.record
    last = rec.decisions[-1]
    assert rec.end == "resign" and last.played_child == -1 and last.choice == "resigned" and last.moves_made == 0
    assert rec.winner == 1 - last.acting_player and rec.header["result"]["result"] == ("1-0" if rec.winner == 0 else "0-1")
    with Game.from_pgn(rec.header["pgn"]) as replayed:
        assert replayed.replay_all() == rec.n_actions and replayed.snapshot().fingerprint == rec.header["result"]["final_fingerprint"]
    t = build_targets(rec, TargetConfig(short_term_horizons=NET.short_term_horizons, deblunder=False))
    assert t.wdl[-1].tolist() == [0.0, 0.0, 1.0] and np.isnan(t.moves_left).all()  # the resigner lost; moves-left censored
    # Resignation disabled (I18's 10%, here every game): the game goes on, the would-resign events kept.
    off = replace(cfg, resign=replace(resign, disabled_fraction=1.0))
    assert not resign_enabled(off)
    g2 = generate_game(_Doomed(), off, net_cfg=NET)
    h = g2.record.header
    assert g2.record.end != "resign" and not h["resign"]["enabled"] and h["resign"]["events"]
    assert all(g2.record.decisions[e["decision"]].moves_made == 0 for e in h["resign"]["events"])  # only at turn starts


def test_false_resign_report_per_range():
    def rec(end, winner, events, enabled=False):
        return SimpleNamespace(header={"resign": {"enabled": enabled, "events": events}, "result": {"end": end, "winner": winner}})

    games = [
        rec("checkmate", 1, [{"player": 0, "n_boards": 10, "decision": 3}, {"player": 0, "n_boards": 12, "decision": 5}]),  # true
        rec("checkmate", 0, [{"player": 0, "n_boards": 20, "decision": 3}]),  # false: White would have resigned and won
        rec("turn_cap", None, [{"player": 1, "n_boards": 60, "decision": 3}]),  # false: a draw
        rec("unknown", None, [{"player": 1, "n_boards": 60, "decision": 3}]),  # unscored: left out
        rec("resign", 0, [], enabled=True),  # resignation on: not part of the check
    ]
    report = false_resign_report(games, ranges=(50, 100))
    assert report == {"0-49": {"would_resign": 2, "false": 1, "rate": 0.5}, "50-99": {"would_resign": 1, "false": 1, "rate": 1.0}}
    assert math.isclose(sum(r["would_resign"] for r in report.values()), 3)
