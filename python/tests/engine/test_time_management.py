"""Stage E11: time management through the C API (``t5d/timeman.hpp``; ``search.TimeManager``,
``evaluator.run_timed_search``, ``netplay``'s match clock).

The mirror of ``t5d_time_params`` and its companions (laid out as the library's), the
parameters' defaults and refusals - the proof hook taken with the proof channel off, extend-only
(decision 12, build-log spec issue 180) -, a timed decision driven from Python on the model clock building the
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
    T5D_LEGALITY_ILLEGAL,
    T5D_LEGALITY_LEGAL,
    T5D_LEGALITY_UNKNOWN,
    T5D_NO_DEADLINE,
    T5D_SPLIT_BOARD_MOVE,
    T5D_SPLIT_FLAT,
    T5D_STOP_EVALUATIONS,
    T5D_STOP_PLAYOUTS,
    T5D_TIME_STOP_HARD_BOUND,
    T5D_TIME_STOP_NONE,
    T5D_TIME_STOP_SEARCH,
    T5D_TIME_STOP_SOFT_BUDGET,
    T5D_TIME_STOP_SOLVED,
    CTimeParams,
    layout_mismatches,
    load_library,
)
from t5d.engine.evaluator import ServeStats, run_timed_search
from t5d.engine.netplay import GameResult, PlayConfig, check_replay, network_config, play_game, replay_with_gate
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
    assert time_params(proof_volatility=1).proof_volatility == 1  # decision 12: extend-only, the channel off
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


def test_an_implausible_moves_left_is_refused():
    """A moves-left above the plan's bound (2000 turns, build-log spec issue 476) is refused: the
    integral summed one term an action, and 2e8 turns held the plan for half a second."""
    with Game.variant("Standard") as g, Search(g, params=search_params(hunt=0, settling=0)) as s, TimeManager() as tm:
        with pytest.raises(Exception, match="issue 476"):
            tm.begin(s, remaining=60.0, moves_left=2e8)
        assert tm.begin(s, remaining=60.0, moves_left=2000.0).our_actions == 1000.0


def test_the_proof_hook_is_taken_with_the_proof_channel_off():
    """Decision 12 (build-log spec issue 180, ruled 2026-11-23): before the sign-off a proof may
    extend a decision, never stop it - so the hook is taken by a search whose proof channel is off
    (every Python search's), and its plan is the plan without it. A Hunt proof of the root's mate,
    withheld from the tree, leaves the root unsolved, nothing injected, and the decision is not
    stopped as solved."""
    hook, quiet = time_params(proof_volatility=1), search_params(hunt=0, settling=0)
    with Game.variant("Standard") as g, Search(g, params=quiet) as s, TimeManager(hook) as hooked, TimeManager() as plain:
        assert hooked.begin(s, remaining=60.0) == plain.begin(s, remaining=60.0)
    mate = ["[k4/4Q/1K3/5/5:0:1:w]"]  # White mates in one on a 5 x 5 board (test_v3_support.hpp's)
    with Game.from_fen(mate) as g:
        params = search_params(synchronous=1, hunt=1, settling=0, asset_map=0)
        with Search(g, params=params) as s, TimeManager(time_params(proof_volatility=1)) as tm:
            run = run_timed_search(s, UniformEvaluator(), tm, remaining=60.0, model_clock=True)
            info = s.info()
            assert info.hunt_mates >= 1 and info.proofs_withheld >= 1 and info.proofs_injected == 0
            assert not info.root_solved
            assert run.verdict.stop != T5D_TIME_STOP_SOLVED


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
        # A clock with hardly anything left - its hard bound a quarter of 0.01 s - and a measured
        # batch of 3 ms or more: the hard bound ends the decision at its first check (review L8(a)).
        # No batch need run: the root carried from the last decision has visited children.
        run = run_timed_search(s, Slow(), tm, remaining=0.06)
        assert run.verdict.stop in (T5D_TIME_STOP_HARD_BOUND, T5D_TIME_STOP_SEARCH)
        assert run.slices == 0


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
    # The flagged decision's work is counted (review L6), as the engine's play_game counts it; it
    # overran its hard bound (0 s: a quarter of nothing left).
    assert res.decisions == 1 and res.playouts > 0 and res.evaluations > 0
    assert res.hard_overruns == 1


@pytest.mark.parametrize("model", [False, True])
def test_a_clocked_turn_start_verdict_is_uncharged_within_the_node_budget(nets, model):
    """H1 part 5 (decision 13's budget, H22): on a match clock the turn start's verdict is the
    referee's - within ``terminal_nodes`` and charged to neither side. The mated start within one
    node (its proof needs 15) is unsettled: the game ends unscored, both clocks untouched (on the
    wall clock a charge would show); exactly, it is checkmate, the clocks untouched too."""
    cfg, evaluators = nets
    base = PlayConfig(
        start_fen=("[8/8/8/8/8/5k2/6q1/7K:0:1:w]",),
        mode="match",
        batch_size=4,
        cap_turns=0,
        cap_boards=0,
        hunt=False,
        settling=False,
        time=60.0,
        model_clock=model,
    )
    with torch.no_grad():
        tiny = play_game(evaluators, replace(base, terminal_nodes=1), net_cfg=cfg)
        exact = play_game(evaluators, base, net_cfg=cfg)
    assert (tiny.end, tiny.unsettled, tiny.result, tiny.actions) == ("unknown", "verdict", "*", [])
    assert (exact.end, exact.winner, exact.actions) == ("checkmate", 1, [])
    for res in (tiny, exact):
        assert res.timed_decisions == 0 and res.decisions == 0
        assert res.clock_remaining == [60.0, 60.0]


@pytest.mark.parametrize("model", [True, False])
def test_netplay_a_search_that_cannot_move_with_time_left_ends_unscored(model):
    """The E11 review's S1 (its spec pass's setup, SS-F2): the flat form, a generator stopped before
    its first turn and no fallback listing - a root that can get no visit - on a 60 s clock: the
    game ends unscored ("unknown", its question "search", "*"), never raising, both clocks untouched
    - never a flag scored as a loss (before S1)."""
    from t5d.engine.netplay import UNSETTLED_NOTES
    from t5d.net.config import NetConfig

    over = (("generator_nodes", 1), ("flat_fallback", 0))
    cfg = PlayConfig(
        split="flat",
        mode="match",
        hunt=False,
        settling=False,
        asset_map=False,
        max_actions=2,
        search_overrides=over,
        time=60.0,
        model_clock=model,
    )
    res = play_game(UniformEvaluator(), cfg, net_cfg=NetConfig())
    assert (res.end, res.unsettled, res.winner, res.result, res.actions) == ("unknown", "search", None, "*", [])
    assert res.clock_remaining == [60.0, 60.0] and res.decisions == 0 and res.timed_decisions == 1
    assert f'[Termination "unknown{UNSETTLED_NOTES["search"]}"]' in res.record


def _fake_no_visit_search(monkeypatch, limit: int):
    """A search whose batches add playouts but never a root child's visit (the mutation pass's
    probe, adopted): ``run_search`` stubbed, each call counted."""
    import types

    import t5d.engine.netplay as netplay

    class FakeSearch:
        def __init__(self):
            self.p = 0

        def children(self):
            return [types.SimpleNamespace(visits=0, legality=T5D_LEGALITY_LEGAL, index=0)]

        def info(self):
            return types.SimpleNamespace(playouts=self.p)

    calls: list[int] = []

    def fake_run_search(search, evaluator, *, playouts, batch_size, self_play, **regs):
        calls.append(playouts)
        if len(calls) > limit:
            raise RuntimeError("no end: the loop ran on")
        search.p += playouts
        return ServeStats(), None

    monkeypatch.setattr(netplay, "run_search", fake_run_search)
    return FakeSearch(), calls


def test_netplay_the_no_visit_cap_holds_on_a_clock_that_does_not_advance_only(monkeypatch):
    """S1 (and M1): on the model clock, which no idle loop advances, a decision whose batches add
    playouts but never a visit cannot move after ``_NO_VISIT_ROUNDS`` batches - unscored
    (``_NoMove``), never a flag, never a hang; on the wall clock the cap does not apply - the
    side's clock ends the loop, and that is its flag."""
    import t5d.engine.netplay as netplay
    from t5d.engine.netplay import _choose, _DecisionClock, _NoMove, _OutOfClock

    cfg = PlayConfig(mode="match", batch_size=4)
    res = GameResult(config=cfg, record="", actions=[], end="", winner=None)
    fake, calls = _fake_no_visit_search(monkeypatch, 100)
    model = _DecisionClock(d0=time.perf_counter(), hard=1e9, remaining=1e9, model=True)
    with pytest.raises(_NoMove):
        _choose(fake, cfg, res, ServeStats(), None, {}, None, clock=model)
    assert len(calls) == 1 + netplay._NO_VISIT_ROUNDS
    assert netplay._NO_VISIT_ROUNDS == 16  # the placeholder (issue 480), the engine's kNoVisitRounds
    fake, calls = _fake_no_visit_search(monkeypatch, 10**7)
    wall = _DecisionClock(d0=time.perf_counter(), hard=0.01, remaining=0.05, model=False)
    with pytest.raises(_OutOfClock):
        _choose(fake, cfg, res, ServeStats(), None, {}, None, clock=wall)
    assert len(calls) > 1 + netplay._NO_VISIT_ROUNDS


def test_netplay_an_out_of_clock_decision_is_the_flag(monkeypatch):
    """S1 (the mutation pass's probe, adopted): a decision whose side runs out of clock before it
    has a move (``_OutOfClock``) flags the side even where the decision's charge leaves its clock
    positive (the model clock); one whose search cannot move (``_NoMove``) ends the game unscored,
    uncharged."""
    import t5d.engine.netplay as netplay
    from t5d.net.config import NetConfig

    cfg = PlayConfig(
        mode="match", playouts=8, batch_size=4, cap_turns=2, cap_boards=30, hunt=False, settling=False, time=600.0, model_clock=True
    )
    for exc, end in ((netplay._OutOfClock, "flag"), (netplay._NoMove, "unknown")):

        def stub(*args, exc=exc, **kwargs):
            raise exc

        monkeypatch.setattr(netplay, "_choose", stub)
        res = play_game(UniformEvaluator(), cfg, net_cfg=NetConfig())
        assert res.end == end and res.actions == []
        if end == "flag":
            assert res.winner == 1 and res.clock_remaining[0] > 0  # the charge alone would not have flagged
        else:
            assert res.winner is None and res.unsettled == "search" and res.clock_remaining == [600.0, 600.0]


def _clocked_game(nets, **kw):
    """A match game on the model clock (the reviewers' probe configuration)."""
    cfg, evaluators = nets
    play = PlayConfig(mode="match", batch_size=4, hunt=False, settling=False, model_clock=True, **kw)
    with torch.no_grad():
        return play_game(evaluators, play, net_cfg=cfg)


def test_netplay_credits_the_increment(nets):
    """At 1 s and 10 s an action, both clocks end above the base (an action's charge is below
    the base, or its side would have flagged)."""
    price = (("cost_seconds_per_evaluation", 1.0),)
    res = _clocked_game(nets, variant="Standard", cap_turns=2, time=1.0, increment=10.0, time_overrides=price)
    assert res.end != "flag" and len(res.actions) >= 2
    assert all(x > 1.0 for x in res.clock_remaining)


def test_netplay_on_the_model_clock_is_reproducible_and_charges_the_models_price(nets):
    """The model clock (H15; the module's claim): the same game twice is the same game, clocks
    included; and it charges the model's price, not the wall time - at 1e4 s an evaluation the
    first decision's forced batch alone exhausts a 5 s clock."""
    kw = dict(
        variant="Standard - Two Timelines",
        cap_turns=2,
        cap_boards=30,
        seed=5,
        time=4.0,
        increment=0.05,
        time_overrides=(("cost_seconds_per_evaluation", 1.0),),
    )
    a, b = _clocked_game(nets, **kw), _clocked_game(nets, **kw)
    assert a.clock_remaining == b.clock_remaining
    assert a.record == b.record
    c = _clocked_game(nets, variant="Standard", cap_turns=2, time=5.0, time_overrides=(("cost_seconds_per_evaluation", 1e4),))
    assert c.end == "flag" and c.decisions == 1


def test_netplay_a_flag_mid_action_leaves_the_game_at_the_actions_start(nets):
    """A flag after the action's first decision: the game handle goes back to the action's start
    (its fingerprint and board count the start's). The bases are scanned for such a flag."""
    with Game.variant("Standard") as g:
        start = g.snapshot()
    found = 0
    for i in range(1, 61):
        res = _clocked_game(nets, variant="Standard", cap_turns=2, time=0.05 * i, time_overrides=(("cost_seconds_per_evaluation", 1.0),))
        if res.end == "flag" and res.actions == [] and res.decisions >= 2:
            found += 1
            assert res.final_fingerprint == start.fingerprint
            assert res.final_boards == len(start.boards)
            break
    assert found  # the premise


def test_netplay_refuses_bad_time_overrides():
    """A clear refusal, never a TypeError from the keyword clash (review L6)."""
    for overrides, match in (
        ((("allocator", 1),), "PlayConfig.allocator"),
        ((("signals", 0),), "PlayConfig.signals"),
        ((("reserve", 1.0),), "no field 'reserve'"),
        ((("cost_seconds_per_evaluation",),), "pairs"),
    ):
        with pytest.raises(ValueError, match=match):
            PlayConfig(mode="match", time=5.0, time_overrides=overrides).validate()
    PlayConfig(mode="match", time=5.0, time_overrides=(("cost_seconds_per_evaluation", 1.0),)).validate()


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


def test_run_timed_search_sets_the_bound_lifts_it_for_the_forced_slice_and_restores_the_callers(monkeypatch):
    """The review's M3 (test 4) and L3, with M1's forced slice and S2: on the wall clock the hard
    bound is the search's deadline - a 0 s bound (a clock at the overhead) a deadline already
    passed (1e-9 s), never 0, which means none - the caller's deadline (``set_deadline_at``) for a
    slice before any root child has a visit and the bound's same time point again after it (what
    is left of it, not the whole bound again - the mutation pass's Y03), and the caller's restored,
    exactly, when the run ends."""
    calls: list[tuple[str, float]] = []
    real_set, real_at = Search.set_deadline, Search.set_deadline_at

    def spy_set(self, seconds):
        calls.append(("set", float(seconds)))
        real_set(self, seconds)

    def spy_at(self, steady_ns):
        calls.append(("at", int(steady_ns)))
        real_at(self, steady_ns)

    monkeypatch.setattr(Search, "set_deadline", spy_set)
    monkeypatch.setattr(Search, "set_deadline_at", spy_at)
    quiet = search_params(hunt=0, settling=0)
    with Game.variant("Standard") as g, Search(g, params=quiet) as s, TimeManager() as tm:
        s.run(T5D_EVALUATOR_UNIFORM, 32)  # visited: no forced slice
        run = run_timed_search(s, UniformEvaluator(), tm, remaining=3.0)
        assert calls == [("set", run.plan.hard_seconds), ("at", T5D_NO_DEADLINE)]
        assert s.deadline()[0] == T5D_NO_DEADLINE
    calls.clear()
    with Game.variant("Standard") as g, Search(g, params=quiet) as s, TimeManager() as tm:
        run = run_timed_search(s, UniformEvaluator(), tm, remaining=3.0)  # a fresh root
        assert calls[0] == ("set", run.plan.hard_seconds)
        assert calls[1] == ("at", T5D_NO_DEADLINE)  # the forced slice: the caller's (none)
        bound = calls[2]
        assert bound[0] == "at" and bound[1] != T5D_NO_DEADLINE  # the bound's time point again
        assert calls[-1] == ("at", T5D_NO_DEADLINE)  # restored
    calls.clear()
    with Game.variant("Standard") as g, Search(g, params=quiet) as s, TimeManager() as tm:
        s.run(T5D_EVALUATOR_UNIFORM, 32)
        run = run_timed_search(s, UniformEvaluator(), tm, remaining=time_params().overhead_seconds)
        assert run.plan.hard_seconds == 0.0
        assert calls == [("set", 1e-9), ("at", T5D_NO_DEADLINE)]
    calls.clear()
    with Game.variant("Standard") as g, Search(g, params=quiet) as s, TimeManager() as tm:
        real_set(s, 100.0)  # the caller's own deadline, later than the bound
        held, _ = s.deadline()
        s.run(T5D_EVALUATOR_UNIFORM, 32)
        run_timed_search(s, UniformEvaluator(), tm, remaining=3.0)
        assert s.deadline()[0] == held  # restored, the same time point (S2: never cleared)
        real_set(s, 1e-3)  # one earlier than the bound: kept, never replaced
        held, _ = s.deadline()
        calls.clear()
        run_timed_search(s, UniformEvaluator(), tm, remaining=600.0)
        assert not [c for c in calls if c[0] == "set"] and s.deadline()[0] == held


def test_run_timed_search_ends_the_decision_when_a_slice_raises(monkeypatch):
    """S2: ``tm.end`` however the run returns - an evaluator that raises still ends the decision,
    and the caller's deadline is restored."""
    ended: list[bool] = []
    real_end = TimeManager.end

    def spy_end(self, search):
        ended.append(True)
        real_end(self, search)

    monkeypatch.setattr(TimeManager, "end", spy_end)

    class Raising(UniformEvaluator):
        def __call__(self, positions):
            raise RuntimeError("the evaluator failed")

    with Game.variant("Standard") as g, Search(g, params=search_params(hunt=0, settling=0)) as s, TimeManager() as tm:
        s.set_deadline(100.0)
        held, _ = s.deadline()
        with pytest.raises(RuntimeError, match="the evaluator failed"):
            run_timed_search(s, Raising(), tm, remaining=3.0)
        assert ended == [True] and s.deadline()[0] == held


def test_run_timed_search_returns_on_a_carried_root_waiting_to_be_relisted():
    """S2 (the correctness pass's probe_py1, adopted): a carried flat root whose listing a passed
    deadline cut, the hard bound already spent (the clock at the overhead) - Python used its own
    forced condition (``root_visits < 2``) where the engine's is "no root child visited", kept the
    bound inside the slices, never re-listed, and ran on; it returns, as the engine does."""
    import threading

    def params():
        return search_params(split=T5D_SPLIT_FLAT, generator_nodes=1, hunt=0, settling=0, asset_map=0)

    def prepared(game):
        s = Search(game, params=params())
        s.run(T5D_EVALUATOR_UNIFORM, 1)  # the root listed, its children made
        s.set_deadline(1e-9)  # a deadline already passed
        time.sleep(0.01)
        s.run(T5D_EVALUATOR_UNIFORM, 24)  # the children's listings cut
        s.set_deadline(0.0)
        return s

    with Game.variant("Standard") as g:
        emptied = None
        probe = prepared(g)
        for c in probe.children():
            if c.visits == 0:
                continue
            with prepared(g) as x:
                x.advance(c.index)
                if not x.children():
                    emptied = c.index
                    break
        probe.close()
        assert emptied is not None, "the premise: a visited child left with no turns"
        search = prepared(g)
        search.advance(emptied)
        tp = time_params()
        out: list = []
        with TimeManager(tp) as tm:
            worker = threading.Thread(
                target=lambda: out.append(run_timed_search(search, UniformEvaluator(), tm, remaining=float(tp.overhead_seconds))),
                daemon=True,
            )
            worker.start()
            worker.join(60.0)
            assert not worker.is_alive(), "run_timed_search did not return within 60 s (a hang)"
        assert out and out[0].verdict.stop != T5D_TIME_STOP_NONE
        search.close()


@pytest.mark.parametrize("model", [False, True])
def test_run_timed_search_returns_once_the_callers_deadline_has_passed(model):
    """S3 in Python (the engine's p1 setup): the caller's deadline already passed - the run stops
    between slices (``TIME_STOP_SEARCH``, "deadline") before running one under it, and restores it."""
    params = search_params(split=T5D_SPLIT_FLAT, generator_nodes=1, hunt=0, settling=0, asset_map=0)
    with Game.variant("Standard") as g, Search(g, params=params) as s, TimeManager() as tm:
        s.set_deadline(1e-9)
        held, _ = s.deadline()
        time.sleep(0.01)
        run = run_timed_search(s, UniformEvaluator(), tm, remaining=5.0, model_clock=model)
        assert run.verdict.stop == T5D_TIME_STOP_SEARCH and run.search_stop == "deadline" and run.slices == 0
        assert s.deadline()[0] == held


def test_run_timed_search_gives_up_on_slices_that_make_no_progress(monkeypatch):
    """S3's no-progress guard in Python (the engine's kNoProgressSlices): forced slices that leave no
    root child visited - a root with no edge that gains none at once, or ``NO_PROGRESS_SLICES`` in
    a row - end the run (``TIME_STOP_SEARCH``, "exhausted") rather than holding the exception open.
    The slices are stubbed: no real root stays unvisited under them."""
    import types

    import t5d.engine.evaluator as evaluator_module
    from t5d.engine.evaluator import NO_PROGRESS_SLICES

    def stub_serve(driver, evaluator, *, batch_size):
        return ServeStats()

    class StubDriver:
        def __init__(self, search, **kw):
            self.stop = "playouts"

    monkeypatch.setattr(evaluator_module, "serve", stub_serve)
    monkeypatch.setattr(evaluator_module, "CStepDriver", StubDriver)
    monkeypatch.setattr(evaluator_module, "start_epoch", lambda *a, **k: None)
    for children, slices in (([], 1), ([types.SimpleNamespace(visits=0)], NO_PROGRESS_SLICES)):
        quiet = search_params(hunt=0, settling=0)
        with Game.variant("Standard") as g, Search(g, params=quiet) as s, TimeManager() as tm, monkeypatch.context() as m:
            m.setattr(Search, "children", lambda self, children=children: list(children))
            run = run_timed_search(s, UniformEvaluator(), tm, remaining=600.0, model_clock=True)
        assert run.verdict.stop == T5D_TIME_STOP_SEARCH and run.search_stop == "exhausted" and run.slices == slices
    assert NO_PROGRESS_SLICES == 16  # the placeholder (issue 482), the engine's kNoProgressSlices


def _check_position_search(**kw):
    """White in check (a rook on e2), the expansion settling nothing beyond its cheap tiers and no
    lazy verification: root children left unknown."""
    g = Game.from_fen(["[4k3/8/8/8/8/8/4r3/4K3:0:1:w]"])
    params = search_params(hunt=0, settling=0, legality_checks=0, legality_nodes=0, verify_unknown=0, **kw)
    s = Search(g, params=params)
    s.run(T5D_EVALUATOR_UNIFORM, 32)
    return g, s


def test_verify_takes_a_deadline():
    """The review's H1 (part 4): ``Search.verify(index, seconds)`` - past the deadline an unknown
    child stays unknown and stays; with none, it is decided."""
    g, s = _check_position_search()
    try:
        unknown = [c for c in s.children() if c.legality != T5D_LEGALITY_LEGAL]
        assert unknown  # the premise
        n = len(s.children())
        assert s.verify(unknown[0].index, 1e-9) == T5D_LEGALITY_UNKNOWN
        assert len(s.children()) == n
        assert s.verify(unknown[0].index) in (T5D_LEGALITY_LEGAL, T5D_LEGALITY_ILLEGAL)
    finally:
        s.close()
        g.close()


def test_netplay_choice_follows_the_clocked_policy():
    """The review's H1 in netplay (parts 1, 2 and 4), on a decision clock the test sets: the hard
    bound spent, the pick's verification is cut; with no child known legal the children are
    verified in visit order against the side's remaining clock and one shown legal is returned;
    with that clock spent too the side's flag falls (``_OutOfClock``) - never the old throw."""
    from t5d.engine.netplay import PlayConfig, _choose, _DecisionClock, _OutOfClock

    cfg = PlayConfig(mode="match", batch_size=4)
    res = GameResult(config=cfg, record="", actions=[], end="", winner=None)
    g, s = _check_position_search()
    try:
        assert not any(c.legality == T5D_LEGALITY_LEGAL for c in s.children())  # the premise
        spent = _DecisionClock(d0=time.perf_counter() - 10.0, hard=1.0, remaining=600.0, model=False)
        pick = _choose(s, cfg, res, ServeStats(), UniformEvaluator(), {}, None, clock=spent)
        assert s.child(pick).legality == T5D_LEGALITY_LEGAL
        assert res.verified_choices >= 2  # the cut verification, then the search
    finally:
        s.close()
        g.close()
    g, s = _check_position_search()
    try:
        out = _DecisionClock(d0=time.perf_counter() - 10.0, hard=1.0, remaining=5.0, model=False)
        with pytest.raises(_OutOfClock):
            _choose(s, cfg, res, ServeStats(), UniformEvaluator(), {}, None, clock=out)
    finally:
        s.close()
        g.close()
