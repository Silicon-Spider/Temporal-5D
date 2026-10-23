"""Stage N4: the evaluator side of the search's external-evaluator protocol (``t5d/engine/evaluator.py``).

The C step API (engine stage E6a) is not settled, so the protocol is tested through fake
drivers: one hands out a fixed pool of leaves in batches; one plays a one-ply search over
real engine positions, choosing children by the delivered policy - which checks that logit
``i`` is the engine's move ``i``. Every evaluation is checked against its leaf by ``serve``,
and against a direct forward of the model. The evaluator tests on synthetic states run
without the engine (CI's network job); the rest skip when it is not built.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import types

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import (
    ERROR_HEAD_FIELD,
    CachedNetworkEvaluator,
    CStepDriver,
    Evaluation,
    Evaluator,
    Leaf,
    NetworkEvaluator,
    SearchDriver,
    _split,
    check_evaluation,
    serve,
)
from t5d.net import NetConfig, build_model, collate, relayout, synthetic_state
from t5d.net.cache import CachedEvaluator, reference_output
from t5d.net.model import NetOutput
from tests.engine.helpers import RECORDS, requires_engine


class PoolDriver:
    """Leaves from a fixed pool, at most ``per_round`` per collect; records deliveries."""

    def __init__(self, positions, per_round: int | None = None) -> None:
        self.pending = list(enumerate(positions))
        self.per_round = per_round
        self.delivered: dict[int, Evaluation] = {}
        self.out: set[int] = set()

    def collect(self, max_leaves: int):
        k = max_leaves if self.per_round is None else min(max_leaves, self.per_round)
        batch, self.pending = self.pending[:k], self.pending[k:]
        self.out |= {i for i, _ in batch}
        return [Leaf(i, p) for i, p in batch]

    def deliver(self, results):
        for i, ev in results:
            assert i in self.out and i not in self.delivered
            self.delivered[i] = ev

    def finished(self) -> bool:
        return not self.pending


@pytest.fixture(scope="module")
def small_model():
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=3).eval()


def _direct(model, state):
    out = model(collate([state]))
    m = int(state.moves.shape[0])
    return out.policy_logits[0, :m].numpy(), out.value_logits[0].numpy(), out.wdl_logits[0].numpy(), out.moves_left_log[0].item(), out


def _error(out) -> float:
    return float(getattr(out, ERROR_HEAD_FIELD)[0])


def _moves_left_says(model, turns: float):
    """A copy of ``model`` whose moves-left head outputs ``log1p(turns)`` for every position."""
    biased = copy.deepcopy(model)
    with torch.no_grad():
        biased.aux.moves_left.weight.zero_()
        biased.aux.moves_left.bias.fill_(math.log1p(turns))
    return biased


# --- without the engine (synthetic states) --------------------------------------------------------


def test_protocols_are_satisfied(small_model):
    assert isinstance(NetworkEvaluator(small_model), Evaluator)
    assert isinstance(PoolDriver([]), SearchDriver)


def test_serve_evaluates_every_leaf_in_batches(small_model):
    states = [synthetic_state(100 + i, n) for i, n in enumerate((1, 3, 7, 2, 12, 5, 4))]
    driver = PoolDriver(states)
    stats = serve(driver, NetworkEvaluator(small_model, max_batch=2), batch_size=3)
    assert stats.leaves == len(states) and stats.batch_sizes == [3, 3, 1] and stats.rounds == 3
    for i, state in enumerate(states):
        ev = driver.delivered[i]
        pol, val, wdl, moves_left_log, out = _direct(small_model, state)
        np.testing.assert_allclose(ev.policy_logits, pol, atol=1e-4)
        np.testing.assert_allclose(ev.value_logits, val, atol=1e-4)
        np.testing.assert_allclose(ev.wdl_logits, wdl, atol=1e-4)  # D3: the WDL is the search's value
        assert ev.moves_left == pytest.approx(math.expm1(max(moves_left_log, 0.0)), rel=1e-4, abs=1e-5)
        assert ev.error == pytest.approx(_error(out), rel=1e-4, abs=1e-6) and ev.error >= 0  # D3's error head
        assert ev.policy_logits.dtype == np.float32 and ev.value_probs.shape == (51,)
        assert abs(float(ev.value_probs.sum()) - 1.0) < 1e-5 and -1.0 <= ev.value_mean <= 1.0
        p = ev.wdl_probs.astype(np.float64)
        assert ev.value_mean == pytest.approx(p[0] - p[2], abs=1e-6)


def test_moves_left_is_delivered_in_turns(small_model):
    """D4 / M20: the head predicts ``log1p`` of the turns left; the evaluation carries turns
    (``expm1``), floored at 0 - shown with a head that says 37 turns, and one that says less
    than none."""
    states = [synthetic_state(120 + i, n) for i, n in enumerate((1, 3, 2))]
    for ev in NetworkEvaluator(_moves_left_says(small_model, 37.0))(states):
        assert ev.moves_left == pytest.approx(37.0, rel=1e-5)
    for ev in NetworkEvaluator(_moves_left_says(small_model, -0.5))(states):
        assert ev.moves_left == 0.0


def test_evaluation_helpers():
    ev = Evaluation(np.array([0.0, np.log(3.0)], np.float32), np.log(np.array([0.5, 0.25, 0.25], np.float32)), np.zeros(51, np.float32))
    np.testing.assert_allclose(ev.policy_probs(), [0.25, 0.75], atol=1e-6)
    assert ev.value_mean == pytest.approx(0.25, abs=1e-6)  # D3: W - L, not the categorical's mean
    assert abs(ev.categorical_mean) < 1e-9  # a uniform distribution over symmetric bins
    assert ev.categorical_variance == pytest.approx(float(np.mean(((2.0 * np.arange(51) - 50) / 50) ** 2)), rel=1e-5)
    peaked = Evaluation(np.zeros(0, np.float32), np.zeros(3, np.float32), np.r_[np.full(50, -1e4), 0.0].astype(np.float32))
    assert peaked.categorical_mean == pytest.approx(1.0) and peaked.policy_probs().shape == (0,)
    assert peaked.value_mean == pytest.approx(0.0) and peaked.categorical_variance == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="no categorical"):
        _ = Evaluation(np.zeros(0, np.float32), np.zeros(3, np.float32)).categorical_mean


def test_check_evaluation_refuses_what_breaks_the_contract():
    leaf = Leaf(7, synthetic_state(5, 3))
    m = int(leaf.position.moves.shape[0])
    w = np.zeros(3, np.float32)
    good = Evaluation(np.zeros(m, np.float32), w, np.zeros(51, np.float32))
    check_evaluation(leaf, good, 51)
    check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), w, error=0.5, moves_left=12.0), 51)  # the categorical optional
    with pytest.raises(ValueError, match="policy logits for"):
        check_evaluation(leaf, Evaluation(np.zeros(m + 1, np.float32), w, np.zeros(51, np.float32)), 51)
    with pytest.raises(ValueError, match="not finite"):
        check_evaluation(leaf, Evaluation(np.full(m, np.nan, np.float32), w, np.zeros(51, np.float32)), 51)
    with pytest.raises(ValueError, match="value logits"):
        check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), w, np.zeros(50, np.float32)), 51)
    with pytest.raises(ValueError, match="WDL logits of shape"):
        check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), np.zeros(51, np.float32)), 51)
    with pytest.raises(ValueError, match="WDL logit is not finite"):
        check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), np.array([0, np.inf, 0], np.float32)), 51)
    with pytest.raises(ValueError, match="error output"):
        check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), w, error=-0.1), 51)
    with pytest.raises(ValueError, match="moves-left output"):
        check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), w, moves_left=math.inf), 51)


def test_serve_refuses_a_broken_evaluator_or_driver(small_model):
    states = [synthetic_state(1, 2), synthetic_state(2, 3)]
    with pytest.raises(RuntimeError, match="evaluations for"):
        serve(PoolDriver(states), lambda positions: [], batch_size=2)

    class Stuck(PoolDriver):
        def collect(self, max_leaves):
            return []

    with pytest.raises(RuntimeError, match="no leaves but is not finished"):
        serve(Stuck(states), NetworkEvaluator(small_model))
    stats = serve(PoolDriver(states, per_round=1), NetworkEvaluator(small_model), batch_size=8, max_rounds=1)
    assert stats.rounds == 1 and stats.leaves == 1


def test_other_register_counts_are_relaid_out(small_model):
    state = synthetic_state(9, 4)
    other = relayout(state, 4, 0)
    a = NetworkEvaluator(small_model)([state])[0]
    b = NetworkEvaluator(small_model)([other])[0]
    np.testing.assert_allclose(a.policy_logits, b.policy_logits, atol=1e-5)
    np.testing.assert_allclose(a.value_logits, b.value_logits, atol=1e-5)


def test_the_cached_evaluator_needs_a_search():
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, attn_scope="block_causal"), seed=0).eval()
    ev = CachedNetworkEvaluator(CachedEvaluator(model))
    with pytest.raises(RuntimeError, match="start"):
        ev([synthetic_state(0, 3)])


def test_the_c_step_driver_is_built():
    """Stage N5 implemented the adapter point: the driver needs a search (tests/engine/test_search_api.py)."""
    assert isinstance(CStepDriver, type) and hasattr(CStepDriver, "abandon")
    with pytest.raises(TypeError):
        CStepDriver()


def _shift(state, legal):
    from dataclasses import replace

    return replace(state, acting_player=1 - state.player_to_move, submit_possible=legal is not None, submit_legal=bool(legal))


def test_the_submit_logit_crosses_the_protocol(small_model):
    """D8: the dense evaluator returns the submit logit where the leaf offers a legal submit and
    NaN elsewhere - not offered, or masked as illegal; the evaluation's probabilities are one
    distribution over the moves and submit."""
    s = synthetic_state(40, 6)
    states = [s, _shift(s, None), _shift(s, False), _shift(s, True)]
    evs = NetworkEvaluator(small_model)(states)
    assert [np.isnan(ev.submit_logit) for ev in evs] == [True, True, True, False]
    out = small_model(collate([states[3]]))
    assert evs[3].submit_logit == pytest.approx(float(out.submit_logits[0]), abs=1e-5)
    for ev, state in zip(evs, states, strict=True):
        check_evaluation(Leaf(0, state), ev, 51)
        assert abs(float(ev.policy_probs().sum()) + ev.submit_prob - 1.0) < 1e-5
    assert evs[3].submit_prob > 0 and evs[2].submit_prob == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_network_submit_logit_is_refused(bad):
    """Where submitting is legal a non-finite submit logit from the network is an error - NaN
    would otherwise read as "no logit given" (the search's placeholder share) - as a non-finite
    move logit is; where submit is masked or absent the readout is not used."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1), seed=2).eval()
    with torch.no_grad():
        model.submit.out.bias.fill_(bad)
    s = synthetic_state(42, 5)
    with pytest.raises(ValueError, match="submit logit"):
        NetworkEvaluator(model)([_shift(s, True)])
    for ok in (s, _shift(s, False), _shift(s, None)):
        assert np.isnan(NetworkEvaluator(model)([ok])[0].submit_logit)


def test_the_error_head_is_read_by_its_pinned_name(small_model):
    """D3 / 6m: the evaluator reads the error head's output from ``NetOutput``'s
    ``ERROR_HEAD_FIELD`` - no other field of ``NetOutput`` may name an error head (one added under
    another name would go unread, every evaluation's ``error`` NaN, and a 6m arm on the error
    head refused by the search). The network's error head is that field (D3, the network lane's
    heads): each evaluation carries its row; an output without the field gives NaN (none)."""
    assert ERROR_HEAD_FIELD == "value_error"
    names = [f.name for f in dataclasses.fields(NetOutput)]
    assert [n for n in names if "error" in n] == [ERROR_HEAD_FIELD]
    s = synthetic_state(43, 4)
    (plain,) = NetworkEvaluator(small_model)([s])
    out = small_model(collate([s]))
    assert plain.error == pytest.approx(_error(out), rel=1e-5) and plain.error >= 0
    fields = {n: getattr(out, n) for n in names if n != ERROR_HEAD_FIELD}
    (without,) = _split(types.SimpleNamespace(**fields), [s])
    assert math.isnan(without.error)
    (read,) = _split(types.SimpleNamespace(**fields, **{ERROR_HEAD_FIELD: torch.tensor([0.375])}), [s])
    assert read.error == 0.375


def test_check_evaluation_refuses_a_wrong_submit_logit():
    s = synthetic_state(41, 3)
    m = int(s.moves.shape[0])
    with pytest.raises(ValueError, match="offers no legal submit"):
        check_evaluation(Leaf(1, s), Evaluation(np.zeros(m, np.float32), np.zeros(3, np.float32), submit_logit=0.5), 51)
    with pytest.raises(ValueError, match="infinite"):
        check_evaluation(Leaf(2, _shift(s, True)), Evaluation(np.zeros(m, np.float32), np.zeros(3, np.float32), submit_logit=np.inf), 51)
    check_evaluation(Leaf(3, _shift(s, True)), Evaluation(np.zeros(m, np.float32), np.zeros(3, np.float32)), 51)  # NaN: none given


# --- with the engine (real positions) ---------------------------------------------------------------


class OnePlyDriver:
    """A one-ply search over real positions: the root first, then the children of its
    ``width`` highest-policy moves, each made by ``apply(i)`` - the engine's move ``i`` - on a
    copy of the root, and submitted when it shifts the present."""

    def __init__(self, game, width: int = 4) -> None:
        self.root = game
        self.width = width
        self.queue = [(0, game.position(), None)]
        self.results: dict[int, Evaluation] = {}
        self.children: dict[int, tuple[int, str]] = {}  # leaf id -> (root move index, its text)
        self.next_id = 1

    def collect(self, max_leaves):
        batch, self.queue = self.queue[:max_leaves], self.queue[max_leaves:]
        return [Leaf(i, p) for i, p, _ in batch]

    def deliver(self, results):
        for i, ev in results:
            self.results[i] = ev
            if i == 0:
                self._expand_root(ev)

    def _expand_root(self, ev):
        x = self.root.expand(square_features="none")
        texts = x.move_texts()
        for move in np.argsort(-ev.policy_logits, kind="stable")[: self.width].tolist():
            child = self.root.copy()
            child.expand(square_features="none")
            child.apply(move)  # the policy's index is the engine's move index
            if child.info().obligations_met:
                child.submit()
            self.queue.append((self.next_id, child.position(), None))
            self.children[self.next_id] = (move, texts[move])
            self.next_id += 1
            child.close()

    def finished(self):
        return not self.queue


@requires_engine
def test_a_one_ply_search_over_real_positions(small_model):
    from t5d.engine import Game

    with Game.from_pgn(RECORDS["branching"]) as g:
        for _ in range(4):
            g.replay_next()
        root_moves = g.expand(square_features="none").move_texts()
        driver = OnePlyDriver(g, width=5)
        stats = serve(driver, NetworkEvaluator(small_model, max_batch=4), batch_size=4)
    assert stats.leaves == 6 and stats.rounds == 3  # the root alone, then 4 + 1 children
    root = driver.results[0]
    assert root.policy_logits.shape == (len(root_moves),)
    best = np.argsort(-root.policy_logits, kind="stable")[:5].tolist()
    assert [driver.children[i][0] for i in sorted(driver.children)] == best
    assert [driver.children[i][1] for i in sorted(driver.children)] == [root_moves[i] for i in best]
    for i in driver.children:
        assert driver.results[i].policy_logits.ndim == 1 and np.isfinite(driver.results[i].value_logits).all()


@requires_engine
def test_real_leaves_through_both_evaluators():
    """Dense and cached evaluators on the same real leaves: each equals its reference."""
    from t5d.engine import Game

    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal"), seed=6).eval()
    with Game.from_pgn(RECORDS["cross_timeline"]) as g:
        positions = [g.position()] + [g.position() for _ in g.replay()]
    dense = PoolDriver(positions)
    serve(dense, NetworkEvaluator(model, scope="block_causal"), batch_size=3)
    cached_ev = CachedNetworkEvaluator(CachedEvaluator(model))
    cached_ev.start(positions[0])
    cached = PoolDriver(positions, per_round=2)
    serve(cached, cached_ev, batch_size=2)
    for i, pos in enumerate(positions):
        ref = reference_output(model, pos.state)
        m = pos.move_count
        for ev in (dense.delivered[i], cached.delivered[i]):
            np.testing.assert_allclose(ev.policy_logits, ref.policy_logits[0, :m].numpy(), atol=1e-4)
            np.testing.assert_allclose(ev.value_logits, ref.value_logits[0].numpy(), atol=1e-4)
    assert torch.is_tensor(ref.value_logits)


class Recording:
    """An evaluator that records every evaluation it returns, call by call."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[list[Evaluation]] = []

    def __call__(self, positions):
        evs = self.inner(positions)
        self.calls.append(evs)
        return evs


@requires_engine
def test_the_network_delivers_every_value_head_to_the_search(small_model):
    """D3 through the C step API (engine ABI round 2): the network's WDL, categorical, error head
    and moves-left reach the search - the search's record of its root's own evaluation (I15) is
    the network's evaluation of the root leaf, the error head's output included - and the
    search's default uncertainty signal (6m: the categorical's variance) gets its signal at every
    leaf, as does the error head's under ``uncertainty_source``: with E2 reading it, no delivery is
    refused and no node lacks it."""
    from t5d.engine import Game
    from t5d.engine.capi import T5D_UNCERTAINTY_CATEGORICAL, T5D_UNCERTAINTY_ERROR_HEAD
    from t5d.engine.evaluator import run_search
    from t5d.engine.search import Search, search_params

    for source in (T5D_UNCERTAINTY_CATEGORICAL, T5D_UNCERTAINTY_ERROR_HEAD):
        with Game.variant("Standard") as g:
            s = Search(g, params=search_params(seed=1, variance_scaled_cpuct=1, uncertainty_source=source))
            rec = Recording(NetworkEvaluator(small_model, max_batch=4))
            run_search(s, rec, playouts=24, batch_size=4)
            info = s.info()
            s.close()
        root = rec.calls[0][0]  # the first gather of a fresh search is its root
        assert len(rec.calls[0]) == 1 and info.root_evaluated
        np.testing.assert_allclose(info.root_wdl, root.wdl_probs, atol=1e-5)
        assert info.root_error == pytest.approx(root.error, rel=1e-5) and root.error >= 0
        assert info.root_moves_left == pytest.approx(root.moves_left, rel=1e-5)
        assert all(ev.value_logits is not None and math.isfinite(ev.error) for evs in rec.calls for ev in evs)
        assert info.evaluations == sum(len(evs) for evs in rec.calls) and info.uncertainty_missing == 0, source
    # Moves-left's unit across the boundary: a head that says log1p(37) everywhere reaches the
    # search as 37 turns (the random head's outputs are clamped to 0 turns, so they cannot show it).
    with Game.variant("Standard") as g:
        s = Search(g, params=search_params(seed=1))
        run_search(s, NetworkEvaluator(_moves_left_says(small_model, 37.0), max_batch=4), playouts=8, batch_size=4)
        info = s.info()
        s.close()
    assert info.root_evaluated and info.root_moves_left == pytest.approx(37.0, rel=1e-5)


def test_an_optimistic_evaluator_delivers_the_optimistic_prior_with_the_same_values():
    """D9 (match play only): an evaluator built with ``policy="optimistic"`` delivers the optimistic
    head's move and submit logits as the prior, and the same value heads as the normal evaluator -
    on the dense path and the cached one; a network without the head, or an unknown source, is
    refused."""
    from t5d.engine.evaluator import POLICY_SOURCES, policy_source
    from t5d.train.smoke import with_actions

    assert POLICY_SOURCES == ("normal", "optimistic")
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, optimistic_policy=True, attn_scope="block_causal"), seed=8).eval()
    states = with_actions([synthetic_state(50 + i, n) for i, n in enumerate((2, 3, 1, 4))])
    normal = NetworkEvaluator(model, scope="block_causal")
    optimistic = NetworkEvaluator(model, scope="block_causal", policy="optimistic")
    assert policy_source(normal) == "normal" and policy_source(optimistic) == "optimistic"
    out = model(collate(states), scope="block_causal")
    cached = CachedNetworkEvaluator(CachedEvaluator(model), policy="optimistic")
    cached.start(states[0])
    for i, (a, b) in enumerate(zip(normal(states), optimistic(states), strict=True)):
        m = int(states[i].moves.shape[0])
        np.testing.assert_allclose(b.policy_logits, out.optimistic_logits[i, :m].numpy(), atol=1e-5)
        assert not np.allclose(a.policy_logits, b.policy_logits, atol=1e-3)
        if bool(out.submit_valid[i]):
            assert b.submit_logit == pytest.approx(float(out.optimistic_submit_logits[i]), abs=1e-5) and b.submit_logit != a.submit_logit
        else:
            assert math.isnan(b.submit_logit)
        np.testing.assert_array_equal(a.wdl_logits, b.wdl_logits)
        np.testing.assert_array_equal(a.value_logits, b.value_logits)
        assert (a.error, a.moves_left) == (b.error, b.moves_left)
    (c,) = cached([states[0]])
    np.testing.assert_allclose(c.policy_logits, out.optimistic_logits[0, : int(states[0].moves.shape[0])].numpy(), atol=1e-4)
    # Each evaluation names its prior, so a wrapper that hides the evaluator's cannot carry the
    # optimistic prior into self-play: serve (and run_search) refuse it there.
    assert c.policy == "optimistic" and {ev.policy for ev in optimistic(states)} == {"optimistic"}
    assert {ev.policy for ev in normal(states)} == {"normal"}
    hidden = Recording(optimistic)
    assert policy_source(hidden) == "normal", "the wrapper names no prior"
    with pytest.raises(ValueError, match="self-play"):
        serve(PoolDriver(states), hidden, batch_size=2, self_play=True)
    assert serve(PoolDriver(states), Recording(normal), batch_size=2, self_play=True).leaves == len(states)
    assert serve(PoolDriver(states), hidden, batch_size=2).leaves == len(states), "match play takes it"
    with pytest.raises(ValueError, match="prior"):
        check_evaluation(Leaf(0, states[2]), dataclasses.replace(c, policy="greedy"))
    with pytest.raises(ValueError, match="D9"):
        NetworkEvaluator(build_model(NetConfig(trunk_dtype="fp32", n_layers=1), seed=0), policy="optimistic")
    with pytest.raises(ValueError, match="policy must be"):
        NetworkEvaluator(model, policy="greedy")
