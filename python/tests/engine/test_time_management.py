"""Stage E11: time management through the C API (``t5d/timeman.hpp``; ``search.TimeManager``,
``evaluator.run_timed_search``, ``netplay``'s match clock).

The mirror of ``t5d_time_params`` and its companions (laid out as the library's), the
parameters' defaults and refusals - the proof hook refused from Python, as the proof channel is
(build-log spec issue 180) -, a timed decision driven from Python on the model clock building the
engine's own ``run_timed`` tree (H15's digest) with the same plan and stop, the wall clock's hard
bound held (a generous correctness bound), and match play on a clock end to end: the games
replay through the gate, a flag ends a game, self-play takes no clock.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from t5d.engine import Game, NetworkEvaluator, UniformEvaluator
from t5d.engine.capi import (
    MIRRORED_SEARCH_STRUCTS,
    T5D_EVALUATOR_UNIFORM,
    T5D_SPLIT_BOARD_MOVE,
    T5D_SPLIT_FLAT,
    T5D_STOP_EVALUATIONS,
    T5D_STOP_PLAYOUTS,
    T5D_TIME_STOP_HARD_BOUND,
    T5D_TIME_STOP_NONE,
    T5D_TIME_STOP_SEARCH,
    T5D_TIME_STOP_SOFT_BUDGET,
    CTimeParams,
    layout_mismatches,
    load_library,
)
from t5d.engine.evaluator import run_timed_search
from t5d.engine.netplay import PlayConfig, check_replay, network_config, play_game, replay_with_gate
from t5d.engine.search import ALLOCATORS, Search, TimeManager, search_params, time_params
from t5d.net import build_model
from t5d.net.config import NetConfig
from t5d.net.cost import dense_flops
from t5d.net.heads import effort_censored_prob, effort_quantile
from tests.engine.helpers import RECORDS, requires_engine
from tests.engine.test_netplay import GATE, gate_required, requires_gate  # noqa: F401 - gate_required: the fixture requires_gate names

pytestmark = requires_engine


def test_the_time_structs_are_mirrored_as_the_library_lays_them_out():
    for name in ("t5d_time_params", "t5d_time_context", "t5d_time_plan", "t5d_time_verdict"):
        assert name in MIRRORED_SEARCH_STRUCTS
    assert layout_mismatches(load_library()) == []
    names = [name for name, _ in CTimeParams._fields_]
    # E8's fields appended after E11's (appended to, never reordered)
    assert names.index("difficulty_weighting") == names.index("extend_instability") + 1
    assert names[-1] == "difficulty_reference_weight"


def test_the_cost_model_is_cost_pys_dense_flops():
    """The engine's cost shape (``t5d/timeman.hpp``) is ``net/cost.py``'s dense model: with no fixed
    share and the prior's scale (0.01 s at 100 boards), a plan's seconds per evaluation at N boards
    is the prior x dense_flops(66 N + 16) / dense_flops(66 x 100 + 16) at the F1 width (d 256, FFN
    ratio 8), at positions of different board counts."""
    cfg = NetConfig()
    assert (cfg.d_model, cfg.ffn_ratio) == (256, 8)
    reference = dense_flops(66 * 100 + 16, cfg)["total"]
    seen = set()
    for text in (None, RECORDS["branching"], RECORDS["cross_timeline"]):
        with Game.variant("Standard") if text is None else Game.from_pgn(text) as g:
            if text is not None:
                g.replay_all()
            with Search(g, params=search_params(hunt=0, settling=0)) as s, TimeManager(time_params(cost_fixed_fraction=0.0)) as tm:
                plan = tm.begin(s, remaining=60.0)
                seen.add(plan.boards)
                expected = 0.01 * dense_flops(66 * plan.boards + 16, cfg)["total"] / reference
                assert plan.seconds_per_evaluation == pytest.approx(expected, rel=1e-12)
    assert len(seen) == 3


def test_time_params_defaults_names_and_refusals():
    p = time_params()
    assert p.allocator == ALLOCATORS["integral"] and p.signals == 1 and p.proof_volatility == 0
    assert list(p.boards_per_action) == [1.05, 1.25, 1.77]  # M22's moves per turn by range
    q = time_params(allocator="flat_wall_clock", signals=0, boards_per_action=(1, 2, 3))
    assert q.allocator == ALLOCATORS["flat_wall_clock"] and q.signals == 0 and list(q.boards_per_action) == [1, 2, 3]
    with pytest.raises(ValueError, match="issue 180"):
        time_params(proof_volatility=1)
    with pytest.raises(ValueError, match="allocator"):
        time_params(allocator="sideways")
    with pytest.raises(ValueError, match="no field"):
        time_params(reserve=1.0)  # no root-scan reserve (issue 306)
    with pytest.raises(Exception, match="hard_fraction"):
        TimeManager(time_params(hard_fraction=2.0))


def test_the_allocator_names_are_the_headers_constants():
    """``ALLOCATORS`` maps each of 6o's arms to the constant ``t5d/capi_search.h`` defines for it
    (review M7: a swap of the two flat arms ran the other arm silently)."""
    header = (Path(__file__).resolve().parents[3] / "engine" / "include" / "t5d" / "capi_search.h").read_text(encoding="utf-8")
    defined = {name.lower(): int(value) for name, value in re.findall(r"#define T5D_ALLOCATOR_(\w+) (\d+)", header)}
    assert defined == {"integral": 0, "flat_evals": 1, "flat_wall_clock": 2}
    assert defined == ALLOCATORS


def test_the_time_manager_refuses_a_proof_hook_set_by_hand():
    """``TimeManager``'s own refusal, past ``time_params``' (a field set afterwards)."""
    p = time_params()
    p.proof_volatility = 1
    with pytest.raises(ValueError, match="issue 180"):
        TimeManager(p)


def test_the_flat_arms_execute_as_named_from_python():
    """After one batch, a check at 1.5x the plan's soft time (inside the hard bound): the
    wall-clock arm has spent its time, the evaluations arm has not done its work."""
    for name, expected in (("flat_evals", T5D_TIME_STOP_NONE), ("flat_wall_clock", T5D_TIME_STOP_SOFT_BUDGET)):
        params = time_params(allocator=name, signals=0)
        with Game.variant("Standard") as g, Search(g, params=search_params(hunt=0, settling=0)) as s, TimeManager(params) as tm:
            plan = tm.begin(s, remaining=600.0)
            s.run(T5D_EVALUATOR_UNIFORM, 8)
            assert 1.5 * plan.soft_seconds < plan.hard_seconds
            assert tm.check(s, 1.5 * plan.soft_seconds).stop == expected
            tm.end(s)


def _engine_timed(search, tm, *, remaining, increment, clock):
    """``run_timed``'s loop through the C API with the engine's own uniform evaluator - a check,
    then a slice of ``batch_size`` playouts (one when synchronous) - on the driver's clock."""
    batch = 1 if search.params.synchronous else max(int(search.params.batch_size), 1)
    plan = tm.begin(search, remaining=remaining, increment=increment)
    t0 = clock()
    while True:
        verdict = tm.check(search, clock() - t0)
        if verdict.stop != T5D_TIME_STOP_NONE:
            break
        if search.run_limits(T5D_EVALUATOR_UNIFORM, playouts=batch) not in (T5D_STOP_PLAYOUTS, T5D_STOP_EVALUATIONS):
            verdict = replace(verdict, stop=T5D_TIME_STOP_SEARCH)
            break
    tm.end(search)
    return plan, verdict, clock() - t0


@pytest.mark.parametrize(
    "allocator,signals",
    [("integral", 1), ("integral", 0), ("flat_evals", 1), ("flat_wall_clock", 1)],
)
@pytest.mark.parametrize("split,synchronous", [(T5D_SPLIT_BOARD_MOVE, 0), (T5D_SPLIT_FLAT, 0), (T5D_SPLIT_BOARD_MOVE, 1)])
def test_a_timed_decision_from_python_is_the_engines_on_the_drivers_clock(split, synchronous, allocator, signals):
    """Under each of 6o's switches, on an elapsed time the driver supplies (review M7): a clock
    that reads the search's playouts at 2 ms each - about 3.6x the prior's price at these boards,
    so the flat arms differ - ``run_timed_search`` (Python's loop and evaluator) builds the tree
    the engine's own slices build, decision after decision, with the same plans and stops."""
    price = 2e-3
    with Game.variant("Standard - Two Timelines") as g:
        params = search_params(seed=3, split=split, batch_size=4, synchronous=synchronous, hunt=0, settling=0, asset_map=0)
        a, b = Search(g, params=params), Search(g, params=params)
        ta = TimeManager(time_params(allocator=allocator, signals=signals))
        tb = TimeManager(time_params(allocator=allocator, signals=signals))
        remaining = 6.0

        def clock_a():
            return a.info().playouts * price

        def clock_b():
            return b.info().playouts * price

        try:
            for _ in range(3):
                run = run_timed_search(a, UniformEvaluator(), ta, remaining=remaining, increment=0.05, clock=clock_a)
                plan, verdict, seconds = _engine_timed(b, tb, remaining=remaining, increment=0.05, clock=clock_b)
                assert run.plan == plan
                assert replace(run.verdict, kld_gain=0.0) == replace(verdict, kld_gain=0.0)
                kld_a, kld_b = run.verdict.kld_gain, verdict.kld_gain
                assert kld_a == kld_b or (math.isnan(kld_a) and math.isnan(kld_b))
                assert verdict.stop != T5D_TIME_STOP_NONE
                assert run.seconds == seconds
                assert signals or verdict.extensions == 0
                assert a.digest() == b.digest()
                pick = a.choose()[0]
                assert pick == b.choose()[0]
                a.advance(pick)
                b.advance(pick)
                remaining -= run.seconds
        finally:
            for x in (a, b, ta, tb):
                x.close()


@pytest.mark.parametrize("split,synchronous", [(T5D_SPLIT_BOARD_MOVE, 0), (T5D_SPLIT_FLAT, 0), (T5D_SPLIT_BOARD_MOVE, 1)])
def test_a_timed_decision_from_python_is_the_engines_on_the_model_clock(split, synchronous):
    """H15: the model clock makes a timed search node-limited - Python's loop (check, then a
    slice of batch_size playouts through CStepDriver; one in the synchronous mode) builds
    run_timed's tree, decision after decision, with the same plan and stop."""
    with Game.variant("Standard - Two Timelines") as g:
        params = search_params(seed=3, split=split, batch_size=4, synchronous=synchronous, hunt=0, settling=0, asset_map=0)
        a, b = Search(g, params=params), Search(g, params=params)
        ta, tb = TimeManager(), TimeManager()
        remaining = 2.0
        try:
            for _ in range(3):
                run = run_timed_search(a, UniformEvaluator(), ta, remaining=remaining, increment=0.05, model_clock=True)
                plan, verdict = b.run_timed(T5D_EVALUATOR_UNIFORM, tb, remaining=remaining, increment=0.05)
                assert run.plan == plan
                assert run.verdict.stop == verdict.stop != T5D_TIME_STOP_NONE
                assert run.verdict.target_evaluations == verdict.target_evaluations
                assert a.digest() == b.digest()
                pick = a.choose()[0]
                assert pick == b.choose()[0]
                a.advance(pick)
                b.advance(pick)
                remaining -= run.seconds
        finally:
            for x in (a, b, ta, tb):
                x.close()


def test_the_hard_bound_holds_on_the_wall_clock():
    """A correctness bound, generous: an evaluator taking 3 ms a batch, each decision within its
    hard bound plus half a second; the search's deadline cleared after the run."""

    class Slow(UniformEvaluator):
        def __call__(self, positions):
            time.sleep(0.003)
            return super().__call__(positions)

    with Game.variant("Standard") as g, Search(g, params=search_params(hunt=0, settling=0)) as s, TimeManager() as tm:
        for _ in range(2):
            t0 = time.perf_counter()
            run = run_timed_search(s, Slow(), tm, remaining=3.0)
            spent = time.perf_counter() - t0
            assert spent <= run.plan.hard_seconds + 0.5
            assert run.plan.hard_seconds <= 0.25 * (3.0 - 0.05) + 1e-9
            assert run.verdict.stop != T5D_TIME_STOP_NONE
            s.advance(s.choose()[0])
        # A clock with hardly anything left: the hard bound, not the plan, ends it (one batch runs).
        run = run_timed_search(s, Slow(), tm, remaining=0.06)
        assert run.verdict.stop in (T5D_TIME_STOP_HARD_BOUND,) or run.plan.soft_seconds <= run.plan.hard_seconds


@pytest.fixture(scope="module")
def nets():
    cfg = network_config("small")
    return cfg, [NetworkEvaluator(build_model(cfg, seed=s).eval(), max_batch=8) for s in (1, 2)]


@requires_gate
@pytest.mark.parametrize("split,allocator,model", [("ply", "integral", True), ("flat", "flat_evals", True), ("ply", "integral", False)])
def test_match_play_on_a_clock_replays_legal(nets, split, allocator, model):
    cfg, evaluators = nets
    play = PlayConfig(
        variant="Standard - Two Timelines",
        split=split,
        mode="match",
        batch_size=4,
        cap_turns=2,
        cap_boards=30,
        seed=5,
        hunt=False,
        settling=False,
        time=4.0,
        increment=0.05,
        allocator=allocator,
        model_clock=model,
        time_overrides=(("cost_seconds_per_evaluation", 1.0),) if model else (),  # the model clock's price: a few batches a decision
    )
    with torch.no_grad():
        res = play_game(evaluators, play, net_cfg=cfg)
    assert res.end in ("checkmate", "stalemate", "turn_cap", "flag")
    assert res.timed_decisions == res.decisions and res.clock_remaining is not None
    replay = replay_with_gate(res.record, gate=GATE)
    assert check_replay(res, replay) == [], replay.lines[-3:]
    assert "clock 4 s + 0.05 s per action" in res.record
    if res.end != "flag":
        assert all(x > 0 for x in res.clock_remaining)


@requires_gate
def test_a_flag_ends_the_game_at_the_actions_start(nets):
    cfg, evaluators = nets
    play = PlayConfig(variant="Standard", mode="match", batch_size=4, time=1e-6, model_clock=True, hunt=False, settling=False)
    with torch.no_grad():
        res = play_game(evaluators, play, net_cfg=cfg)
    assert res.end == "flag" and res.winner == 1 and res.actions == [] and res.result == "*"
    assert check_replay(res, replay_with_gate(res.record, gate=GATE)) == []


def test_self_play_takes_no_clock():
    with pytest.raises(ValueError, match="match play"):
        PlayConfig(mode="self", time=5.0).validate()
    with pytest.raises(ValueError, match="allocator"):
        PlayConfig(mode="match", time=5.0, allocator="sideways").validate()
    assert math.isfinite(PlayConfig(mode="match", time=5.0).time)


def test_e8_the_effort_quantile_is_heads_through_the_c_api():
    """E8's bin (``t5d_time_plan.effort_bin``) is ``heads.effort_quantile``'s at every quantile, on
    random distributions; its censored mass is the last bin's share."""
    rng = np.random.default_rng(7)
    with Game.variant("Standard") as g, Search(g, params=search_params(hunt=0, settling=0)) as s:
        for q in (0.5, 0.8, 0.95):
            with TimeManager(time_params(difficulty_weighting=1, effort_quantile=q)) as tm:
                for _ in range(25):
                    probs = rng.dirichlet(np.full(16, 0.3)).astype(np.float32)
                    plan = tm.begin(s, remaining=60.0, effort=probs)
                    assert plan.difficulty_applied
                    assert plan.effort_bin == int(effort_quantile(torch.from_numpy(probs), q))
                    assert abs(plan.censored_mass - float(effort_censored_prob(torch.from_numpy(probs))) / float(probs.sum())) < 1e-6


def test_e8_off_leaves_the_plan_alone():
    with Game.variant("Standard") as g, Search(g, params=search_params(hunt=0, settling=0)) as s, TimeManager() as a, TimeManager() as b:
        given = a.begin(s, remaining=30.0, effort=[1 / 16] * 16)
        none = b.begin(s, remaining=30.0)
        assert given == none and not given.difficulty_applied and given.effort_bin == -1
        with pytest.raises(ValueError, match="16"):
            a.begin(s, remaining=30.0, effort=[1.0] * 3)
