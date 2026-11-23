"""Register G12's pool of concurrent games behind one evaluator (``t5d/engine/pool.py``; spec section
7.8): the batch planner under its padded-token budget, the single-caller ``TokenBudgetEvaluator``,
the ``SharedEvaluator`` that merges concurrent callers' requests, and ``GamePool``.

Engine-free: the positions are synthetic states, and - except in the one test that holds spec
section 15.2's padded-batch row at the evaluator's level - the evaluator is ``DigestEvaluator``, an
exact function of each position, so that every routing claim is checked bit for bit. Threads are
interleaved by events and barriers (``pool_helpers``), never by sleeping: a batch's composition
follows timing, so no test asserts one that timing could change.

**The tolerance** of the padded-batch test is the network's batch-equivalence tolerance on an FP32
trunk, 1e-5 (``tests/net/test_invariance.py``'s ``ATOL``): a merged forward gives each position's
evaluation alone to that tolerance, NOT bit for bit - a forward's last bits move with the batch's
size and padding (BUILD_LOG issue 262). Measured on that test's six positions (2026-11-22; CPU,
torch 2.13.0, netplay's small FP32 network, seeds 0 to 3): the largest difference of any head from
the position alone was 4.2e-7, 3.9e-7, 1.0e-6 and 7.2e-7 - never zero, and a tenth of the tolerance
at most. Bit for bit holds with merging off, which the same test checks. The pooled games' records
are held in ``tests/train/test_selfplay_pool.py``.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import CachedNetworkEvaluator, NetworkEvaluator
from t5d.engine.netplay import SMALL
from t5d.engine.pool import (
    BOARD_RANGES,
    GAME_THREAD,
    GAMES,
    MAX_PADDING,
    MAX_POSITIONS,
    SERVICE_THREAD,
    TOKEN_BUDGET,
    BatchConfig,
    EvaluatorFailed,
    GameFailed,
    GamePool,
    PoolClosed,
    PooledEvaluator,
    PoolStats,
    SharedEvaluator,
    TokenBudgetEvaluator,
    plan_batch,
    plan_batches,
    token_count,
)
from t5d.net import NetConfig, build_model, synthetic_state
from tests.engine.pool_helpers import (
    PATIENCE,
    DigestEvaluator,
    FailingEvaluator,
    GatedEvaluator,
    RecordingEvaluator,
    pool_threads,
    same_evaluation,
    wait_for,
)

NET = NetConfig(**SMALL)
ATOL = 1e-5  # FP32 trunk: the network's batch-equivalence tolerance (module docstring)


def _states(boards, seed: int = 100):
    """Distinct synthetic positions of the given board counts (148 tokens at 2 boards, 214 at 3 ...)."""
    return [synthetic_state(seed + i, n) for i, n in enumerate(boards)]


def _offering_submit(state):
    """The same boards past the shift, a legal submit offered: the evaluation has a submit logit."""
    return replace(state, acting_player=1 - state.player_to_move, submit_possible=True, submit_legal=True)


def _within(sizes, cfg: BatchConfig) -> bool:
    """Whether a forward of ``sizes`` keeps the budget, the position cap and the padding ceiling."""
    padded = len(sizes) * max(sizes)
    return len(sizes) == 1 or (
        padded <= cfg.token_budget and len(sizes) <= cfg.max_positions and padded - sum(sizes) <= cfg.max_padding * padded
    )


def _run(target, count: int) -> list[threading.Thread]:
    threads = [threading.Thread(target=target, args=(k,), daemon=True) for k in range(count)]
    for t in threads:
        t.start()
    return threads


def _join(threads) -> None:
    for t in threads:
        t.join(PATIENCE)
    assert not any(t.is_alive() for t in threads), "a caller never returned"


# --- the planner ------------------------------------------------------------------------------------


def test_the_placeholders_are_the_named_constants():
    """Every number of the pool is a named PLACEHOLDER (G12; Gate 1 step 2 sets them): the defaults
    are those constants, and the budget is 22 positions at 44 boards and 4 at 240."""
    cfg = BatchConfig()
    assert (cfg.token_budget, cfg.max_positions, cfg.max_padding) == (TOKEN_BUDGET, MAX_POSITIONS, MAX_PADDING) == (65_536, 64, 0.25)
    assert cfg.merge is True and cfg.max_wait_ms == 0.0 and GAMES == 32 and BOARD_RANGES == (50, 100, 200)
    assert TOKEN_BUDGET // (66 * 44 + 16) == 22 and TOKEN_BUDGET // (66 * 240 + 16) == 4
    assert [token_count(s) for s in _states((1, 2, 3))] == [82, 148, 214]  # 66 N + 16


def test_plan_batch_fills_to_the_token_budget():
    """The budget counts padded tokens - members x the longest (section 7.8: "a padded batch pays
    for its longest member"): 3 x 280 fits 1000, 4 x 280 does not."""
    cfg = BatchConfig(token_budget=1000)
    assert plan_batch([280] * 10, cfg) == [0, 1, 2]
    assert [len(b) for b in plan_batches([280] * 10, cfg)] == [3, 3, 3, 1]
    rng = np.random.default_rng(0)
    for _ in range(300):
        sizes = [int(s) for s in rng.integers(82, 1700, size=int(rng.integers(1, 40)))]
        cfg = BatchConfig(
            token_budget=int(rng.integers(100, 6000)), max_positions=int(rng.integers(1, 20)), max_padding=float(rng.uniform(0.0, 0.9))
        )
        for batch in plan_batches(sizes, cfg):
            assert _within([sizes[i] for i in batch], cfg), (sizes, cfg, batch)


def test_plan_batch_keeps_a_large_position_alone_and_buckets_by_size():
    """Section 7.8: "small positions many to a batch and a 240-board position alone", and "a batch
    holds positions of similar size"."""
    at_240, at_44 = 66 * 240 + 16, 66 * 44 + 16
    assert plan_batches([at_240, at_44, at_44], BatchConfig(token_budget=16_000)) == [[0], [1, 2]]
    # A position larger than the budget still goes - alone.
    assert plan_batches([at_240, at_44], BatchConfig(token_budget=4000)) == [[0], [1]]
    # 44 and 45 boards share a batch; the 2- and 3-board positions would put 32% of it in padding,
    # and form their own (15% padding).
    sizes = [2920, 148, 2986, 214]
    cfg = BatchConfig(token_budget=65_536, max_padding=0.25)
    assert plan_batch(sizes, cfg) == [0, 2]
    assert plan_batches(sizes, cfg) == [[0, 2], [1, 3]]
    assert plan_batches(sizes, replace(cfg, max_padding=0.5)) == [[0, 1, 2, 3]]
    # A position that does not fit is passed over, not waited for: the later equal one joins.
    assert plan_batch([148, 2920, 148], cfg) == [0, 2]


def test_plan_batch_always_serves_the_oldest_and_caps_the_count():
    rng = np.random.default_rng(1)
    for _ in range(300):
        sizes = [int(s) for s in rng.integers(82, 3000, size=int(rng.integers(1, 30)))]
        cfg = BatchConfig(
            token_budget=int(rng.integers(82, 9000)), max_positions=int(rng.integers(1, 9)), max_padding=float(rng.uniform(0, 0.6))
        )
        batch = plan_batch(sizes, cfg)
        assert batch[0] == 0, "the oldest waiting position is always served (no starvation)"
        assert batch == sorted(set(batch)) and len(batch) <= cfg.max_positions
        assert batch == plan_batch(list(sizes), cfg), "the planner is a function of the sizes"
        batches = plan_batches(sizes, cfg)
        assert sorted(i for b in batches for i in b) == list(range(len(sizes))), "every position in exactly one batch"
        assert all(b == sorted(b) for b in batches)
    assert plan_batch([], BatchConfig()) == [] and plan_batches([], BatchConfig()) == []
    assert plan_batch([148] * 5, BatchConfig(max_positions=2)) == [0, 1]
    for bad in (
        dict(token_budget=0),
        dict(max_positions=0),
        dict(max_padding=1.0),
        dict(max_padding=-0.1),
        dict(max_padding=float("nan")),
        dict(max_wait_ms=-1.0),
        dict(max_wait_ms=float("inf")),
        dict(max_wait_ms=float("nan")),
    ):
        with pytest.raises(ValueError):
            BatchConfig(**bad).validate()
        with pytest.raises(ValueError):
            SharedEvaluator(DigestEvaluator(), BatchConfig(**bad))


def test_pool_stats_count_tokens_padding_and_ranges():
    stats = PoolStats()
    stats.record([148, 214, 214], [2, 3, 3], 2, 0.5)
    stats.record([3250], [49], 1, 0.25)
    stats.record([3316, 6616], [50, 100], 1, 0.25)
    stats.record([13216, 13216], [200, 199], 3, 1.0)
    assert (stats.forwards, stats.positions, stats.largest_batch, stats.cross_caller_forwards) == (4, 8, 3, 2)
    assert stats.tokens == 148 + 214 + 214 + 3250 + 3316 + 6616 + 2 * 13216
    assert stats.padded_tokens == 3 * 214 + 3250 + 2 * 6616 + 2 * 13216
    assert stats.batch_histogram == {3: 1, 1: 1, 2: 2} and stats.mean_batch == 2.0
    assert stats.padding_share == pytest.approx(1 - stats.tokens / stats.padded_tokens)
    assert stats.fill(13216 * 2) == pytest.approx(stats.padded_tokens / (4 * 26432))
    d = stats.to_dict()
    assert list(d["by_range"]) == ["0-49", "100-199", "200+"], "a forward counts in its largest member's range"
    assert d["by_range"]["0-49"] == {"forwards": 2, "positions": 4, "tokens": 3826, "padded_tokens": 3892, "seconds": 0.75}
    assert d["by_range"]["100-199"]["positions"] == 2 and d["by_range"]["200+"]["forwards"] == 1
    assert json.loads(json.dumps(d)) == d, "the statistics are JSON as they stand"
    assert PoolStats().mean_batch == 0.0 and PoolStats().padding_share == 0.0 and PoolStats().fill(1) == 0.0


# --- one caller ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_a_token_budget_batch_evaluates_each_position_as_alone(seed):
    """Spec section 15.2's row "a position evaluated in a padded batch = the same position evaluated
    alone", at the evaluator's level and with the planner's buckets: every head of every evaluation
    equals the position's alone to the FP32 tolerance - not bit for bit (module docstring) - in the
    caller's order, though the planner reorders; with merging off the evaluations are the inner
    evaluator's own, bit for bit."""
    model = build_model(NET, seed=seed).eval()
    states = _states((12, 2, 3, 7, 3), seed=40)
    states[2] = _offering_submit(states[2])
    states.append(states[1])  # the same position twice in one request
    alone = NetworkEvaluator(model)
    want = [alone([s])[0] for s in states]
    inner = RecordingEvaluator(NetworkEvaluator(model, max_batch=64))
    budget = TokenBudgetEvaluator(inner, BatchConfig())
    got = budget(states)
    assert len(got) == len(states)
    for g, w, s in zip(got, want, states, strict=True):
        assert g.policy_logits.shape == (s.moves.shape[0],) and g.board_logits.shape == (s.n_boards,)
        for name in ("policy_logits", "wdl_logits", "value_logits", "board_logits"):
            a, b = getattr(g, name), getattr(w, name)
            assert np.allclose(a, b, rtol=0, atol=ATOL), (name, float(np.abs(a - b).max()))
        for name in ("submit_logit", "error", "moves_left"):
            assert np.allclose(getattr(g, name), getattr(w, name), rtol=0, atol=ATOL, equal_nan=True), name
    assert np.isfinite(got[2].submit_logit) and np.isnan(got[0].submit_logit)
    planned = [[token_count(p) for p in positions] for positions, _ in inner.calls]
    assert planned == [[808, 478], [148, 214, 214, 148]], "the planner's buckets, each one forward"
    assert budget.stats.forwards == 2 and budget.stats.requests == 1 and budget.stats.positions == 6
    assert budget.stats.padded_tokens == 2 * 808 + 4 * 214 and budget.model is model and budget.policy == "normal"
    # Merging off: the inner evaluator's own call, bit for bit.
    plain = NetworkEvaluator(model, max_batch=64)(states)
    off = TokenBudgetEvaluator(NetworkEvaluator(model, max_batch=64), BatchConfig(merge=False))(states)
    assert all(same_evaluation(a, b) for a, b in zip(off, plain, strict=True))


def test_a_token_budget_evaluator_with_an_exact_evaluator_is_exact():
    exact = DigestEvaluator()
    states = _states((5, 1, 9, 2, 2, 4, 1), seed=60)
    budget = TokenBudgetEvaluator(RecordingEvaluator(exact), BatchConfig(token_budget=700, max_positions=3))
    assert all(same_evaluation(a, b) for a, b in zip(budget(states), exact(states), strict=True))
    assert budget.stats.forwards == len(budget.inner.calls) > 1 and budget([]) == []
    assert all(_within([token_count(p) for p in positions], budget.cfg) for positions, _ in budget.inner.calls)
    with pytest.raises(RuntimeError, match="evaluations"):
        TokenBudgetEvaluator(FailingEvaluator(exact, 1, short=True))(states)


# --- many callers -------------------------------------------------------------------------------------


def test_shared_evaluator_returns_each_callers_evaluations_in_order():
    """Six callers, each making several requests of mixed sizes at once: every request gets exactly
    its own positions' evaluations, bit for bit and in its order, whatever shared its forwards."""
    exact = DigestEvaluator()
    stock = _states((1, 2, 2, 3, 5, 8, 3, 1, 6, 2), seed=200)
    stock[3] = _offering_submit(stock[3])
    rng = np.random.default_rng(5)
    plans = [[[stock[int(i)] for i in rng.integers(0, len(stock), size=int(rng.integers(1, 5)))] for _ in range(6)] for _ in range(6)]
    cfg = BatchConfig(token_budget=1200, max_positions=6, max_padding=0.3)
    barrier = threading.Barrier(len(plans))
    got: list = [None] * len(plans)
    errors: list[BaseException] = []
    recorder = RecordingEvaluator(exact)
    with SharedEvaluator(recorder, cfg) as shared:

        def caller(k: int) -> None:
            try:
                with shared.client() as evaluator:
                    barrier.wait(PATIENCE)
                    got[k] = [evaluator(request) for request in plans[k]]
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        _join(_run(caller, len(plans)))
        assert shared.clients == 0 and shared.waiting == 0
    assert not errors, errors
    for plan, results in zip(plans, got, strict=True):
        for request, result in zip(plan, results, strict=True):
            want = exact(request)
            assert len(result) == len(want) and all(same_evaluation(a, b) for a, b in zip(result, want, strict=True))
    stats = shared.stats
    assert stats.requests == 36 and stats.positions == sum(len(r) for plan in plans for r in plan)
    assert stats.forwards == len(recorder.calls) and sum(stats.batch_histogram.values()) == stats.forwards
    assert all(_within([token_count(p) for p in positions], cfg) for positions, _ in recorder.calls)
    assert pool_threads() == []


def _gated_callers(inner, cfg: BatchConfig, requests):
    """Run one caller per request behind a gate: caller 0's request alone enters the first forward,
    and that forward is held until every other caller's request is waiting behind it - so what the
    later forwards hold does not depend on timing. Returns the closed pool and each caller's result."""
    gated = GatedEvaluator(inner)
    results: list = [None] * len(requests)
    errors: list[BaseException] = []
    with SharedEvaluator(gated, cfg) as shared:
        clients = [shared.client() for _ in requests]

        def caller(k: int) -> None:
            try:
                results[k] = clients[k](requests[k])
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=caller, args=(k,), daemon=True) for k in range(len(requests))]
        threads[0].start()
        assert gated.entered.wait(PATIENCE), "the first forward never began"
        for t in threads[1:]:
            t.start()
        wait_for(lambda: shared.waiting == len(requests), "every caller to be waiting")
        gated.gate.set()
        _join(threads)
    assert not errors, errors
    return shared, results


def test_requests_from_different_callers_share_a_forward_within_the_budget():
    """With every caller's request waiting, the next forward holds positions of two or more callers
    - batches fill ACROSS games (section 7.8) - and every forward keeps the budget, the position cap
    and the padding ceiling."""
    exact = DigestEvaluator()
    recorder = RecordingEvaluator(exact)
    cfg = BatchConfig(token_budget=1500, max_positions=8, max_padding=0.25)
    requests = [_states(boards, seed=300 + 10 * k) for k, boards in enumerate(((2,), (2, 3), (3, 2), (2, 2, 3), (9,), (3,), (7, 2)))]
    shared, results = _gated_callers(recorder, cfg, requests)
    for request, result in zip(requests, results, strict=True):
        assert all(same_evaluation(a, b) for a, b in zip(result, exact(request), strict=True))
    owner = {id(p): k for k, request in enumerate(requests) for p in request}
    callers = [{owner[id(p)] for p in positions} for positions, _ in recorder.calls]
    assert callers[0] == {0} and len(callers[1]) >= 2, callers
    assert shared.stats.cross_caller_forwards == sum(len(c) >= 2 for c in callers) >= 1
    assert all(_within([token_count(p) for p in positions], cfg) for positions, _ in recorder.calls)
    assert shared.stats.largest_batch == max(len(positions) for positions, _ in recorder.calls) <= cfg.max_positions
    assert pool_threads() == []


def test_merge_off_forwards_each_request_whole():
    """``merge`` off: with every caller's request waiting at once, each call the evaluator sees is
    still exactly one caller's request - the same positions in the same order, the call the game
    alone makes - and never a mixture."""
    exact = DigestEvaluator()
    recorder = RecordingEvaluator(exact)
    requests = [_states(boards, seed=400 + 10 * k) for k, boards in enumerate(((2,), (2, 3), (3, 2, 2, 5), (9, 1), (3,)))]
    shared, results = _gated_callers(recorder, BatchConfig(merge=False, token_budget=100, max_positions=1), requests)
    for request, result in zip(requests, results, strict=True):
        assert all(same_evaluation(a, b) for a, b in zip(result, exact(request), strict=True))
    seen = sorted([id(p) for p in positions] for positions, _ in recorder.calls)
    assert seen == sorted([id(p) for p in request] for request in requests), "one call per request, whole and in order"
    assert shared.stats.cross_caller_forwards == 0 and shared.stats.forwards == len(requests)


def test_the_service_thread_holds_a_batch_for_the_callers_still_to_come():
    """``max_wait_ms``: the oldest request is held while a registered client has yet to ask - so
    three callers' positions go in ONE forward - unless the planned batch is already full, and never
    past the wait; 0 (the default) is greedy."""
    exact = DigestEvaluator()
    long_wait = BatchConfig(max_wait_ms=10 * 60 * 1000.0)  # only the release under test can end the hold
    positions = _states((2, 2, 2), seed=500)

    def ask(shared: SharedEvaluator, clients, count: int):
        results: list = [None] * count

        def caller(k: int) -> None:
            results[k] = clients[k]([positions[k]])

        _join(_run(caller, count))
        return results

    # Every client waiting: one forward of all three.
    recorder = RecordingEvaluator(exact)
    with SharedEvaluator(recorder, long_wait) as shared:
        clients = [shared.client() for _ in range(3)]
        results = ask(shared, clients, 3)
    assert [len(p) for p, _ in recorder.calls] == [3] and shared.stats.cross_caller_forwards == 1
    assert all(same_evaluation(r[0], exact.evaluate(p)) for r, p in zip(results, positions, strict=True))
    # A full batch goes without the client that never asks (2 x 148 fills a budget of 300).
    recorder = RecordingEvaluator(exact)
    with SharedEvaluator(recorder, replace(long_wait, token_budget=300)) as shared:
        clients = [shared.client() for _ in range(3)]
        ask(shared, clients, 2)
    assert [len(p) for p, _ in recorder.calls] == [2]
    # The wait ends: one caller of two is served after it, alone.
    with SharedEvaluator(exact, BatchConfig(max_wait_ms=30.0)) as shared:
        clients = [shared.client() for _ in range(2)]
        t0 = time.perf_counter()
        (only,) = clients[0]([positions[0]])
        assert time.perf_counter() - t0 >= 0.030 and same_evaluation(only, exact.evaluate(positions[0]))
    # A client that closes no longer holds the others' batch.
    with SharedEvaluator(exact, long_wait) as shared:
        clients = [shared.client() for _ in range(2)]
        clients[1].close()
        assert same_evaluation(clients[0]([positions[1]])[0], exact.evaluate(positions[1]))
    assert pool_threads() == []


def test_the_pools_threads_run_without_autograd_at_the_makers_thread_count(monkeypatch):
    """Grad mode is thread-local and ON in a new thread, and torch's thread count is not promised to
    carry over to one: the service thread - where every forward runs - and each game worker run
    under ``torch.no_grad()``, and the service thread takes the count of the thread that made the
    pool where its own differs."""
    seen: dict = {}

    class Probe(DigestEvaluator):
        def __call__(self, positions):
            seen.update(grad=torch.is_grad_enabled(), thread=threading.current_thread().name)
            return super().__call__(positions)

    maker = threading.current_thread()
    taken: list[int] = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 3 if threading.current_thread() is maker else 5)
    monkeypatch.setattr(torch, "set_num_threads", taken.append)
    with torch.enable_grad(), SharedEvaluator(Probe()) as shared, shared.client() as client:
        client(_states((2,), seed=550))
    assert seen == {"grad": False, "thread": SERVICE_THREAD} and taken == [3]

    def play(evaluator, config):
        return torch.is_grad_enabled(), threading.current_thread().name

    with GamePool(DigestEvaluator(), games=1) as pool:
        (result,) = pool.run(play, [0])
    assert result.value == (False, f"{GAME_THREAD}-0") and taken == [3, 3]
    assert pool_threads() == []


def test_an_evaluator_failure_reaches_every_waiting_caller_and_closes_the_pool():
    """The evaluator raises on its second call, with three callers waiting - two in that forward and
    one behind it: each raises ``EvaluatorFailed`` chained from the original exception, the caller
    served before it keeps its evaluations, the pool is closed and its thread has ended. An evaluator
    that returns the wrong number of evaluations fails the same way."""
    exact = DigestEvaluator()
    boom = ArithmeticError("the network fell over")
    gated = GatedEvaluator(exact)
    failing = FailingEvaluator(gated, 2, error=boom)
    requests = [[s] for s in _states((2, 2, 2, 2), seed=600)]
    outcomes: list = [None] * len(requests)
    shared = SharedEvaluator(failing, BatchConfig(token_budget=300))  # two 148-token positions a forward
    clients = [shared.client() for _ in requests]

    def caller(k: int) -> None:
        try:
            outcomes[k] = clients[k](requests[k])
        except BaseException as exc:  # noqa: BLE001 - the outcome under test
            outcomes[k] = exc

    threads = [threading.Thread(target=caller, args=(k,), daemon=True) for k in range(len(requests))]
    threads[0].start()
    assert gated.entered.wait(PATIENCE)
    for t in threads[1:]:
        t.start()
    wait_for(lambda: shared.waiting == len(requests), "every caller to be waiting")
    gated.gate.set()
    _join(threads)
    assert same_evaluation(outcomes[0][0], exact.evaluate(requests[0][0]))
    for outcome in outcomes[1:]:
        assert isinstance(outcome, EvaluatorFailed) and outcome.__cause__ is boom
    assert len({id(o) for o in outcomes[1:]}) == 3, "each caller raises its own exception"
    wait_for(lambda: not shared.running, "the service thread to end")
    assert shared.closed and shared.failure is boom and failing.calls == 2
    for call in (lambda: clients[0](requests[0]), shared.client):
        with pytest.raises(PoolClosed) as later:
            call()
        assert later.value.__cause__ is boom
    shared.close()
    # The wrong number of evaluations.
    with (
        SharedEvaluator(FailingEvaluator(exact, 1, short=True)) as short,
        pytest.raises(EvaluatorFailed, match="1 evaluations for 2 positions"),
    ):
        short.client()(_states((2, 3), seed=610))
    assert short.closed and isinstance(short.failure, RuntimeError)
    assert pool_threads() == []


class _Rooted:
    """An evaluator shaped as the cached path's: ``start(root)`` opens a search's epoch."""

    policy = "optimistic"
    model = object()

    def start(self, root):
        return None

    def __call__(self, positions):
        return DigestEvaluator()(positions)


class _Counted(DigestEvaluator):
    """An evaluator that splits what it is given by a count, as ``NetworkEvaluator`` does."""

    max_batch = 8
    policy = "optimistic"
    model = object()


def test_close_releases_waiting_callers_and_the_pool_refuses_what_it_cannot_share():
    exact = DigestEvaluator()
    gated = GatedEvaluator(exact)
    requests = [[s] for s in _states((2, 3, 4), seed=700)]
    outcomes: list = [None] * len(requests)
    shared = SharedEvaluator(gated)
    clients = [shared.client() for _ in requests]

    def caller(k: int) -> None:
        try:
            outcomes[k] = clients[k](requests[k])
        except BaseException as exc:  # noqa: BLE001 - the outcome under test
            outcomes[k] = exc

    threads = [threading.Thread(target=caller, args=(k,), daemon=True) for k in range(len(requests))]
    threads[0].start()
    assert gated.entered.wait(PATIENCE)
    for t in threads[1:]:
        t.start()
    wait_for(lambda: shared.waiting == len(requests), "every caller to be waiting")
    # A client serves one search at a time: a second call while one waits is refused.
    with pytest.raises(RuntimeError, match="one search at a time"):
        clients[0](requests[0])
    closer = threading.Thread(target=shared.close, daemon=True)
    closer.start()
    _join(threads)  # released at once, though the forward in flight is still held
    assert all(type(o) is PoolClosed for o in outcomes), outcomes
    assert shared.running, "the forward in flight cannot be interrupted"
    gated.gate.set()
    closer.join(PATIENCE)
    assert not closer.is_alive() and not shared.running and shared.closed and shared.failure is None
    shared.close()  # idempotent
    with pytest.raises(PoolClosed):
        shared.client()
    with pytest.raises(PoolClosed):
        clients[1](requests[1])
    assert pool_threads() == []
    # What the pool cannot share: an evaluator whose evaluations depend on one search's root ...
    assert callable(CachedNetworkEvaluator.start), "the cached path's evaluator is the one refused"
    for make in (SharedEvaluator, TokenBudgetEvaluator, lambda inner: GamePool(inner, games=2)):
        with pytest.raises(ValueError, match="start"):
            make(_Rooted())
    # ... and, when merging, one that would split a planned batch again by its own count.
    with pytest.raises(ValueError, match="max_batch"):
        SharedEvaluator(_Counted(), BatchConfig(max_positions=9))
    with pytest.raises(ValueError, match="size"):
        TokenBudgetEvaluator(exact, size_of=lambda p: 0)(requests[0])
    for cfg in (BatchConfig(max_positions=8), BatchConfig(max_positions=9, merge=False)):
        with SharedEvaluator(_Counted(), cfg) as ok, ok.client() as client:
            # The client shows the real evaluator's prior and network, and is no rooted evaluator.
            assert isinstance(client, PooledEvaluator) and client.policy == "optimistic" and client.model is _Counted.model
            assert not hasattr(client, "start") and not hasattr(client, "max_batch")
            assert same_evaluation(client(requests[2])[0], exact.evaluate(requests[2][0])) and client([]) == []
            assert ok.clients == 1
            client.close()
            client.close()  # idempotent
            assert ok.clients == 0
            with pytest.raises(RuntimeError, match="closed"):
                client(requests[2])
    assert pool_threads() == []


# --- the pool of games ------------------------------------------------------------------------------


_STOCK: list = []


def _game(k: int):
    """Game ``k``'s requests: three rounds of one to three positions, as a search's rounds are -
    drawn (seeded) from a stock of 24 distinct positions of 1 to 6 boards."""
    if not _STOCK:
        _STOCK.extend(_states([1 + i % 6 for i in range(24)], seed=1000))
    rng = np.random.default_rng(900 + k)
    return [[_STOCK[int(i)] for i in rng.choice(len(_STOCK), size=int(rng.integers(1, 4)), replace=False)] for _ in range(3)]


def _play(evaluator, config):
    return [evaluator(request) for request in config]


def test_game_pool_runs_every_config_once():
    """Twenty games on four workers: every config is played exactly once, each game's evaluations
    are the ones it gets alone (bit for bit), and no thread outlives the pool."""
    exact = DigestEvaluator()
    configs = [_game(k) for k in range(20)]
    with GamePool(exact, games=4, batch=BatchConfig(token_budget=900, max_positions=5)) as pool:
        assert pool.evaluator is exact and pool.batch.token_budget == 900
        results = list(pool.run(_play, iter(configs)))
        assert pool.alive == 0
    assert sorted(r.index for r in results) == list(range(20))
    for r in results:
        assert r.error is None and r.config is configs[r.index] and r.seconds >= 0.0
        for got, request in zip(r.value, configs[r.index], strict=True):
            assert all(same_evaluation(a, b) for a, b in zip(got, exact(request), strict=True))
    assert pool.stats.requests == 60 and pool.stats.positions == sum(len(q) for c in configs for q in c)
    assert pool_threads() == []
    with pytest.raises(RuntimeError, match="once"):
        pool.run(_play, configs)
    for bad in (dict(games=0), dict(on_error="ignore"), dict(results_ahead=0)):
        with pytest.raises(ValueError):
            GamePool(exact, **bad)
    closed = GamePool(exact, games=2)
    closed.close()
    with pytest.raises(PoolClosed):
        closed.run(_play, configs)


def test_a_worker_failure_is_raised_or_collected():
    exact = DigestEvaluator()
    configs = [_game(k) for k in range(8)]
    broken = KeyError("game 3's own fault")

    def play(evaluator, config):
        if config is configs[3]:
            evaluator(config[0])
            raise broken
        return _play(evaluator, config)

    # "raise": the first failure ends the run, naming the game; the other workers are unwound.
    with GamePool(exact, games=3) as pool, pytest.raises(GameFailed) as failed:
        list(pool.run(play, configs))
    assert failed.value.index == 3 and failed.value.config is configs[3] and failed.value.__cause__ is broken
    assert pool_threads() == []
    # "collect": the failure is that game's result, and every other game completes.
    with GamePool(exact, games=3, on_error="collect") as pool:
        results = {r.index: r for r in pool.run(play, configs)}
    assert sorted(results) == list(range(8)) and results[3].error is broken and results[3].value is None
    assert all(r.error is None for k, r in results.items() if k != 3)

    # The configs' own iterable failing is reported the same two ways (index -1).
    def feed():
        yield from configs[:2]
        raise broken

    with GamePool(exact, games=2, on_error="collect") as pool:
        results = {r.index: r for r in pool.run(_play, feed())}
    assert sorted(results) == [-1, 0, 1] and results[-1].error is broken
    with GamePool(exact, games=2) as pool, pytest.raises(GameFailed) as failed:
        list(pool.run(_play, feed()))
    assert failed.value.index == -1 and failed.value.__cause__ is broken
    # The evaluator failing ends the run with EvaluatorFailed in both modes - never silently short.
    for on_error in ("raise", "collect"):
        boom = ArithmeticError("the network fell over")
        with GamePool(FailingEvaluator(exact, 4, error=boom), games=3, on_error=on_error) as pool, pytest.raises(EvaluatorFailed) as failed:
            list(pool.run(_play, configs))
        assert failed.value.__cause__ is boom and pool.shared.failure is boom
    assert pool_threads() == []


def test_results_back_pressure_and_early_exit():
    """A consumer that stops reading stalls the workers: with ``results_ahead`` 1 and three workers,
    the games started are those read plus one waiting plus one finished in each worker's hands - and
    configs are pulled only then, so an endless iterable is safe. Leaving early cancels the games
    waiting on the evaluator (``PoolClosed``) and joins every thread."""
    exact = DigestEvaluator()
    started: list[int] = []
    lock = threading.Lock()

    def count(evaluator, config):
        with lock:
            started.append(config)
        return config

    with GamePool(exact, games=3, results_ahead=1) as pool:
        results = pool.run(count, itertools.count())
        for read in (1, 2, 3):
            next(results)
            wait_for(lambda read=read: len(started) == read + 1 + 3, "the workers to stall")
            time.sleep(0.05)  # nothing more can start: every worker is blocked handing over its game
            assert len(started) == read + 1 + 3
    assert pool_threads() == [] and len(started) <= 3 + 1 + 3

    # Leaving early: game 0 completes, games 1 and 2 wait on a held forward.
    gated = GatedEvaluator(exact)
    request = _states((2, 3), seed=1100)
    cancelled: list[int] = []

    def play(evaluator, config):
        if config == 0:
            return "done"
        try:
            return evaluator(request)
        except PoolClosed:
            cancelled.append(config)
            gated.gate.set()  # the forward in flight may end now: close() is waiting for it
            raise

    with GamePool(gated, games=3) as pool:
        for result in pool.run(play, range(3)):
            assert (result.index, result.value) == (0, "done")
            wait_for(lambda: pool.shared.waiting == 2, "both other games to wait on the evaluator")
            break
    assert sorted(cancelled) == [1, 2] and pool.alive == 0 and pool_threads() == []
