"""Register M24's target arms for I21 (``train.target_arms``; spec sections 12.2, 13.1): the arms'
configurations, the record-level comparison of the pruned-visit and completed-Q targets by hand,
the held-out yardstick against the losses it restates, the ordering, the command line - and two
arms through at most three optimiser steps each on synthetic rows. Nothing is trained and no arm
is ranked: the ranking waits for the pilot run's data."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model, synthetic_states
from t5d.train.losses import LossConfig, compute_losses
from t5d.train.records import GameRecord, read_game, write_game
from t5d.train.selfplay import generate_game
from t5d.train.smoke import MAX_STEPS, optimiser_steps, with_actions
from t5d.train.target_arms import (
    ARMS,
    BASELINE,
    COMPARISON_COUNTS,
    COMPARISON_MEANS,
    SEARCH_CLASSES,
    arm_configs,
    arm_trainer_config,
    compare_records,
    evaluate_rows,
    format_comparison,
    main,
    rank_arms,
)
from t5d.train.targets import Row, TargetConfig, build_targets, collate_rows
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import WindowConfig
from tests.engine.helpers import requires_engine
from tests.train.test_completed_q import LOGITS, _children, _decision, _softmax
from tests.train.test_targets import GAMES

pytestmark = pytest.mark.grad

NET = NetConfig(**SMALL)  # two layers at d = 64, FP32
ALL = TargetConfig(short_term_horizons=NET.short_term_horizons, completed_q="all")


def _record(seed: int, counts=(1, 3, 2, 4, 2, 5)):
    states = with_actions(synthetic_states(seed, list(counts)))
    return synthetic_record(seed, states), states


def _rows(seed: int, cfg: TargetConfig = ALL):
    record, states = _record(seed)
    targets = build_targets(record, cfg)
    return [Row(s, targets, i, synthetic=True) for i, s in enumerate(states)]


def _entropy(p) -> float:
    return -sum(float(x) * math.log(float(x)) for x in p if x > 0)


def _kl(t, p) -> float:
    return sum(float(a) * math.log(float(a) / float(b)) for a, b in zip(t, p, strict=True) if a > 0)


# --- the arms ------------------------------------------------------------------------------------------


def test_the_arms_and_their_configurations():
    """An arm is the builder's switch and whether the main policy head trains on the completed-Q
    target; the baseline is the as-built target and loss, and nothing else differs between arms."""
    assert set(ARMS) == {"visits_full", "completed_q_full", "completed_q_cheap", "completed_q_all"}
    assert BASELINE == "visits_full" and ARMS[BASELINE] == ("off", False)
    target, loss = arm_configs(BASELINE)
    assert target == TargetConfig() and loss.weights == LossConfig().weights
    for arm, searches in (("completed_q_full", "full"), ("completed_q_cheap", "cheap"), ("completed_q_all", "all")):
        target, loss = arm_configs(arm)
        assert target == TargetConfig(completed_q=searches)
        assert loss.weights["policy"] == 0.0 and loss.weights["completed_q"] == 1.0
        assert {k: v for k, v in loss.weights.items() if k not in ("policy", "completed_q")} == {
            k: v for k, v in LossConfig().weights.items() if k not in ("policy", "completed_q")
        }
    # the caller's settings are kept: only the switch and the two weights move
    given = TargetConfig(forced_k=0.0, deblunder=False)
    target, loss = arm_configs("completed_q_all", given, LossConfig(q_ratio=0.25))
    assert target == replace(given, completed_q="all") and loss.q_ratio == 0.25
    base = TrainerConfig(batch_size=8, lr=1e-3)
    for arm in ARMS:
        cfg = arm_trainer_config(arm, NET, base)
        assert cfg.target.short_term_horizons == tuple(NET.short_term_horizons)  # the network's, as the trainer reads it
        assert cfg.target.completed_q == ARMS[arm][0] and (cfg.loss.weights["completed_q"] != 0.0) == ARMS[arm][1]
        assert (cfg.batch_size, cfg.lr) == (8, 1e-3)
        Trainer(build_model(NET, seed=0), NET, cfg)  # each arm's settings pass the trainer's guard
    assert arm_trainer_config("completed_q_full", NET, replace(base, target=given)).target == replace(given, completed_q="full")
    with pytest.raises(ValueError, match="arm must be one of"):
        arm_configs("visits_cheap")  # not built: the builder writes no pruned-visit target for a cheap search (I5)


# --- the two targets on the records ----------------------------------------------------------------------


def _hand_record() -> GameRecord:
    """Two decisions over three moves with the raw prior 1/4, 1/2, 1/4 and a level raw WDL: a full
    search (visits 4, 0, 1; values 0.2, -, -0.4) and a cheap one (visits 2, 1, 0; values -0.3, 0.5, -)
    whose once-visited move drew a lucky value."""
    full = _decision(_children([4, 0, 1], [0.2, 9.9, -0.4]), LOGITS)
    cheap = replace(_decision(_children([2, 1, 0], [-0.3, 0.5, 9.9]), LOGITS), full_search=False)
    ds = [replace(d, decision=i, action=i) for i, d in enumerate([full, cheap])]
    header = {"result": {"end": "checkmate", "winner": 0, "actions": 2, "final_active_timelines": [0]}, "search": {}}
    return GameRecord(header=header, decisions=ds)


def test_the_comparison_by_hand():
    """Every statistic of ``compare_records`` from the formulas, on two constructed decisions (no
    pruning, so the visit distribution is the visits'): the paper's target is softmax(logit +
    (c_visit + max N) (q^ + 1) / 2), an unvisited move's q^ the mixed value."""
    stats = compare_records([_hand_record()], TargetConfig(forced_k=0.0, prune_single_visit=False))
    assert tuple(stats) == SEARCH_CLASSES and all(set(s) == set(COMPARISON_COUNTS) | set(COMPARISON_MEANS) for s in stats.values())
    stored = np.asarray(LOGITS, dtype=np.float32).astype(np.float64)  # the record keeps raw logits as float32
    prior = _softmax(stored)
    # the full search: the visited moves' prior-weighted mean is (0.2 - 0.4) / 2 = -0.1 over 5 visits
    v_full = np.array([0.8, 0.0, 0.2])
    c_full = _softmax(stored + 54.0 * (np.array([0.2, (0.0 + 5 * -0.1) / 6, -0.4]) + 1.0) / 2.0)
    # the cheap search: the mean is (1/4 x -0.3 + 1/2 x 0.5) / (3/4) over 3 visits (the prior as stored)
    v_cheap = np.array([2 / 3, 1 / 3, 0.0])
    mix = (0.0 + 3 * ((prior[0] * -0.3 + prior[1] * 0.5) / (prior[0] + prior[1]))) / 4
    c_cheap = _softmax(stored + 52.0 * (np.array([-0.3, 0.5, mix]) + 1.0) / 2.0)
    assert int(np.argmax(c_full)) == 0 and int(np.argmax(c_cheap)) == 1  # the cheap one's top move was visited once
    want = {
        "full": dict(v=v_full, c=c_full, visits=[4, 0, 1], playouts=5, visited=2, agree=1.0, top_visits=4.0, single=0.0),
        "cheap": dict(v=v_cheap, c=c_cheap, visits=[2, 1, 0], playouts=3, visited=2, agree=0.0, top_visits=1.0, single=1.0),
    }
    for name, w in want.items():
        got = stats[name]
        assert (got["decisions"], got["flat"], got["visit_targets"], got["completed_q_targets"], got["visit_kl_infinite"]) == (
            1,
            0,
            1,
            1,
            0,
        )
        assert (got["actions"], got["playouts"], got["visited"]) == (3, w["playouts"], w["visited"])
        assert got["visit_entropy"] == pytest.approx(_entropy(w["v"]), abs=1e-12)
        assert got["completed_q_entropy"] == pytest.approx(_entropy(w["c"]), abs=1e-12)
        assert got["visit_kl"] == pytest.approx(_kl(w["v"], prior), abs=1e-12)
        assert got["completed_q_kl"] == pytest.approx(_kl(w["c"], prior), abs=1e-12)
        assert got["top1_agree"] == w["agree"]
        assert got["total_variation"] == pytest.approx(0.5 * float(np.abs(w["v"] - w["c"]).sum()), abs=1e-12)
        unvisited = sum(float(c) for c, n in zip(w["c"], w["visits"], strict=True) if n == 0)
        assert got["completed_q_unvisited_mass"] == pytest.approx(unvisited, abs=1e-12) and unvisited > 0
        assert got["completed_q_top1_visits"] == w["top_visits"] and got["completed_q_top1_single_visit"] == w["single"]
    # sigma's scale under the paper's constants: both targets are close to one-hot where the visits are not
    assert stats["full"]["completed_q_entropy"] < 0.05 < stats["full"]["visit_entropy"]
    # the settings are the caller's: a flatter sigma brings the target back towards the prior
    flatter = compare_records([_hand_record()], TargetConfig(forced_k=0.0, prune_single_visit=False, completed_q_c_visit=0.0))
    assert flatter["full"]["completed_q_entropy"] > stats["full"]["completed_q_entropy"]
    # two records are one tally: the means stay, the counts add
    twice = compare_records([_hand_record(), _hand_record()], TargetConfig(forced_k=0.0, prune_single_visit=False))
    assert twice["full"]["decisions"] == 2 and twice["full"]["total_variation"] == pytest.approx(
        stats["full"]["total_variation"], abs=1e-12
    )


def test_a_flat_record_is_counted_and_skipped():
    """D2's flat form has no completed-Q target: its decisions are counted under ``flat`` and enter
    nothing else."""
    record = synthetic_record(4, synthetic_states(4, [3, 5], n_timelines=3), full_every=2, flat=True)
    assert [d.full_search for d in record.decisions] == [True, False]
    stats = compare_records([record])
    for name in SEARCH_CLASSES:
        assert stats[name]["flat"] == 1
        assert all(stats[name][k] == 0 for k in COMPARISON_COUNTS if k != "flat")
        assert all(math.isnan(stats[name][k]) for k in COMPARISON_MEANS)
    text = format_comparison(stats)
    assert " -" in text and "nan" not in text


def test_the_comparison_reads_only_the_records(tmp_path, capsys):
    """No network, no engine, no positions: the comparison over records read back from disk is the
    comparison over the live records, and the command line prints it."""
    live = [_record(seed)[0] for seed in (1, 7)]
    paths = [write_game(tmp_path / f"{i}.npz", r) for i, r in enumerate(live)]
    a, b = compare_records(live), compare_records(read_game(p) for p in paths)
    for name in SEARCH_CLASSES:
        assert a[name]["decisions"] == 6 and a[name]["completed_q_targets"] == 6  # six decisions a record, every second cheap
        for key, value in a[name].items():
            assert b[name][key] == value or (math.isnan(value) and math.isnan(b[name][key])), (name, key)
    assert main([str(p) for p in paths]) == 0
    out = capsys.readouterr().out
    assert out.rstrip("\n") == format_comparison(a)
    lines = out.splitlines()
    assert lines[0].split() == ["statistic", *SEARCH_CLASSES]
    assert [line.split()[0] for line in lines[1:]] == [*COMPARISON_COUNTS, *COMPARISON_MEANS]
    # the placeholders from the command line: the target under other constants
    assert main([str(paths[0]), "--c-visit", "0", "--c-scale", "0.5", "--value-range", "signed"]) == 0
    assert capsys.readouterr().out.rstrip("\n") == format_comparison(
        compare_records([live[0]], TargetConfig(completed_q_c_visit=0.0, completed_q_c_scale=0.5, completed_q_value_range="signed"))
    )


# --- the held-out yardstick ----------------------------------------------------------------------------


def test_the_yardstick_restates_the_losses_and_is_the_same_for_every_arm():
    """``evaluate_rows`` reads the main policy head's raw distribution against fixed references,
    whatever the network trained on: its cross-entropies are the losses' own on the same rows (the
    policy loss; the completed-Q loss as the count-weighted mean of its two classes; the WDL loss),
    its counts the rows', forward only."""
    rows = _rows(4)
    model = build_model(NET, seed=3)
    model.train()
    got = evaluate_rows(model, rows, NET)
    assert model.training and optimiser_steps(model) == 0  # the mode it was given in; no step
    batch, tgt = collate_rows(rows, NET)
    model.eval()
    with torch.no_grad():
        losses = compute_losses(model(batch), tgt)
    full = tgt.full_search
    n_full, n_cheap = int(full.sum()), int((~full).sum())
    assert n_full > 0 and n_cheap > 0
    assert (got["n_visits"], got["n_completed_q_full"], got["n_completed_q_cheap"], got["n_wdl"]) == (n_full, n_full, n_cheap, len(rows))
    assert got["ce_visits"] == pytest.approx(float(losses["policy"]), rel=1e-5)
    mixed = (n_full * got["ce_completed_q_full"] + n_cheap * got["ce_completed_q_cheap"]) / (n_full + n_cheap)
    assert mixed == pytest.approx(float(losses["completed_q"]), rel=1e-5)
    assert got["wdl"] == pytest.approx(float(losses["wdl"]), rel=1e-5)
    # the figures do not depend on how the rows are batched (float32 rounding of the forward pass aside)
    for batch_size in (1, 4):
        other = evaluate_rows(model, rows, NET, batch_size=batch_size)
        assert other.keys() == got.keys()
        for key, value in got.items():
            assert other[key] == pytest.approx(value, rel=1e-5, abs=1e-6), (batch_size, key)
    # rows built with the switch off: no completed-Q reference, the visit reference as it was
    plain = evaluate_rows(model, _rows(4, replace(ALL, completed_q="off")), NET)
    assert plain["ce_visits"] == got["ce_visits"] and plain["top1_visits"] == got["top1_visits"] and plain["wdl"] == got["wdl"]
    for name in ("completed_q_full", "completed_q_cheap"):
        assert plain[f"n_{name}"] == 0 and math.isnan(plain[f"ce_{name}"]) and math.isnan(plain[f"top1_{name}"])
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_rows(model, rows, NET, batch_size=0)


def test_top1_agreement_by_hand():
    """The primary figure: the share of held-out full searches whose most visited move is the raw
    policy's top action - checked on targets written around the network's own top actions: a hit
    where the top action holds the most visits (alone or tied), a miss where another move does."""
    rows = _rows(6)
    targets = rows[0].targets
    model = build_model(NET, seed=2).eval()
    batch, tgt = collate_rows(rows, NET)
    with torch.no_grad():
        out = model(batch)
    logp = out.action_log_probs()
    m_max = batch.moves.shape[1]
    moves = [int(r.state.moves.shape[0]) for r in rows]
    # each row's top action among its own (its moves, then submit), and another legal one: a move
    top = [moves[i] if column == m_max else column for i, column in enumerate(logp.argmax(dim=-1).tolist())]
    alt = [0 if a != 0 else 1 for a in top]

    def log_prob(i: int, action: int) -> float:  # an action of row i's own, in the batch's columns (submit last)
        return float(logp[i, m_max if action == moves[i] else action])

    full = [i for i in range(len(rows)) if bool(tgt.full_search[i])]
    cheap = [i for i in range(len(rows)) if i not in full]
    assert len(full) == len(cheap) == 3
    policy, completed = list(targets.policy), list(targets.completed_q)
    shares = {"hit": (0.75, 0.25), "tie": (0.5, 0.5), "miss": (0.25, 0.75)}  # the visits' shares of the top action and the other
    for (on_top, on_alt), i in zip(shares.values(), full, strict=True):
        policy[i] = np.zeros(moves[i] + 1)
        policy[i][top[i]], policy[i][alt[i]] = on_top, on_alt
    for i in range(len(rows)):  # the completed-Q references: one-hot, on the top action in the first row of each class only
        completed[i] = np.zeros(moves[i] + 1)
        completed[i][top[i] if i in (full[0], cheap[0]) else alt[i]] = 1.0
    written = replace(targets, policy=policy, completed_q=completed)
    got = evaluate_rows(model, [replace(r, targets=written) for r in rows], NET)
    assert got["n_visits"] == 3 and got["top1_visits"] == pytest.approx(2 / 3)  # the hit and the tie
    ce = [-(a * log_prob(i, top[i]) + b * log_prob(i, alt[i])) for (a, b), i in zip(shares.values(), full, strict=True)]
    assert got["ce_visits"] == pytest.approx(sum(ce) / 3, rel=1e-5)
    for name, members in (("full", full), ("cheap", cheap)):
        assert got[f"n_completed_q_{name}"] == 3 and got[f"top1_completed_q_{name}"] == pytest.approx(1 / 3)
        # a one-hot reference: its cross-entropy is that action's own log-probability
        ce = [-log_prob(i, int(np.argmax(completed[i]))) for i in members]
        assert got[f"ce_completed_q_{name}"] == pytest.approx(sum(ce) / 3, rel=1e-5)


@requires_engine
def test_the_comparison_and_the_yardstick_on_a_generated_game(tmp_path):
    """Both tools on a self-play record read back from disk (PCR's mix of full and cheap searches):
    the comparison's counts are the record's, and the yardstick reads its rows forward only."""
    with torch.no_grad():
        g = generate_game(NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8), GAMES["ply"], net_cfg=NET, keep_positions=True)
    record = read_game(write_game(tmp_path / "game.npz", g.record))
    ds = record.decisions
    n_full = sum(bool(d.full_search) for d in ds)
    assert 0 < n_full < len(ds)
    targets = build_targets(record, ALL)
    stats = compare_records([record], ALL)
    assert (stats["full"]["decisions"], stats["cheap"]["decisions"]) == (n_full, len(ds) - n_full)
    assert stats["full"]["flat"] == stats["cheap"]["flat"] == 0
    assert stats["full"]["visit_targets"] == sum(p is not None for p in targets.policy)  # the builder's, full searches only
    assert stats["full"]["completed_q_targets"] + stats["cheap"]["completed_q_targets"] == len(ds)
    assert all(math.isfinite(stats[name][key]) for name in SEARCH_CLASSES for key in ("completed_q_entropy", "completed_q_kl", "playouts"))
    assert stats["full"]["playouts"] > stats["cheap"]["playouts"]
    assert main([str(tmp_path / "game.npz")]) == 0
    model = build_model(NET, seed=1)
    rows = [Row(p.state, targets, i) for i, p in enumerate(g.positions)]  # record rows: not synthetic
    got = evaluate_rows(model, rows, NET, batch_size=4)
    assert got["n_visits"] == stats["full"]["visit_targets"] > 0
    assert got["n_completed_q_full"] + got["n_completed_q_cheap"] == len(ds) and got["n_completed_q_cheap"] > 0
    assert all(math.isfinite(v) for key, v in got.items() if key != "wdl") and optimiser_steps(model) == 0


def test_rank_arms_orders_by_the_key():
    results = {
        "b": {"top1_visits": 0.4, "ce_visits": 2.0},
        "a": {"top1_visits": 0.4, "ce_visits": 3.0},
        "c": {"top1_visits": 0.7, "ce_visits": 2.5},
        "d": {"top1_visits": math.nan, "ce_visits": 1.0},
        "e": {"ce_visits": math.nan},
    }
    assert rank_arms(results) == ["c", "a", "b", "d", "e"]  # the highest first; ties by name; no figure: last
    assert rank_arms(results, "ce_visits", higher_is_better=False) == ["d", "b", "c", "a", "e"]
    assert rank_arms(results, "ce_visits") == ["a", "c", "b", "d", "e"]
    assert rank_arms({}) == []


# --- two arms, smoke-tested ----------------------------------------------------------------------------


def test_two_arms_smoke_ranked_on_synthetic_rows():
    """The path the ranking will take, end to end, on synthetic rows: per arm a fresh network from
    one seed, the arm's trainer, at most three optimiser steps, then the same held-out yardstick.
    Not a ranking - three steps on random targets say nothing about either target - so nothing is
    asserted about which arm comes first."""
    record, states = _record(5)
    held_out = _rows(9)
    results = {}
    for arm in (BASELINE, "completed_q_all"):
        model = build_model(NET, seed=5)
        cfg = arm_trainer_config(arm, NET, TrainerConfig(batch_size=8, window=WindowConfig(min_rows=1000)))
        trainer = Trainer(model, NET, cfg)
        trainer.add_game(record, positions=states)
        for _ in range(MAX_STEPS):
            values = trainer.step()
            assert all(math.isfinite(v) for v in values.values())
            assert ("completed_q" in values) == (arm != BASELINE)
        assert optimiser_steps(model) == MAX_STEPS
        with pytest.raises(RuntimeError, match="no training runs"):
            trainer.step()
        results[arm] = evaluate_rows(model, held_out, NET)
        assert optimiser_steps(model) == MAX_STEPS  # the yardstick takes no step
        assert all(math.isfinite(v) for v in results[arm].values())
        assert results[arm]["n_visits"] == 3 and results[arm]["n_completed_q_full"] == 3 and results[arm]["n_completed_q_cheap"] == 3
    assert results[BASELINE] != results["completed_q_all"]  # the arms trained different networks
    for key, better in (("top1_visits", True), ("ce_visits", False), ("ce_completed_q_full", False), ("wdl", False)):
        assert sorted(rank_arms(results, key, higher_is_better=better)) == sorted(results)
