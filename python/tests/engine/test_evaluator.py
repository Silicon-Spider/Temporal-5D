"""Stage N4: the evaluator side of the search's external-evaluator protocol (``t5d/engine/evaluator.py``).

The C step API (engine stage E6a) is not settled, so the protocol is tested through fake
drivers: one hands out a fixed pool of leaves in batches; one plays a one-ply search over
real engine positions, choosing children by the delivered policy - which checks that logit
``i`` is the engine's move ``i``. Every evaluation is checked against its leaf by ``serve``,
and against a direct forward of the model. The evaluator tests on synthetic states run
without the engine (CI's network job); the rest skip when it is not built.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import (
    CachedNetworkEvaluator,
    CStepDriver,
    Evaluation,
    Evaluator,
    Leaf,
    NetworkEvaluator,
    SearchDriver,
    check_evaluation,
    serve,
)
from t5d.net import NetConfig, build_model, collate, relayout, synthetic_state
from t5d.net.cache import CachedEvaluator, reference_output
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
    return out.policy_logits[0, :m].numpy(), out.value_logits[0].numpy()


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
        pol, val = _direct(small_model, state)
        np.testing.assert_allclose(ev.policy_logits, pol, atol=1e-4)
        np.testing.assert_allclose(ev.value_logits, val, atol=1e-4)
        assert ev.policy_logits.dtype == np.float32 and ev.value_probs.shape == (51,)
        assert abs(float(ev.value_probs.sum()) - 1.0) < 1e-5 and -1.0 <= ev.value_mean <= 1.0


def test_evaluation_helpers():
    ev = Evaluation(policy_logits=np.array([0.0, np.log(3.0)], np.float32), value_logits=np.zeros(51, np.float32))
    np.testing.assert_allclose(ev.policy_probs(), [0.25, 0.75], atol=1e-6)
    assert abs(ev.value_mean) < 1e-9  # a uniform distribution over symmetric bins
    peaked = Evaluation(policy_logits=np.zeros(0, np.float32), value_logits=np.r_[np.full(50, -1e4), 0.0].astype(np.float32))
    assert peaked.value_mean == pytest.approx(1.0) and peaked.policy_probs().shape == (0,)


def test_check_evaluation_refuses_what_breaks_the_contract():
    leaf = Leaf(7, synthetic_state(5, 3))
    m = int(leaf.position.moves.shape[0])
    good = Evaluation(np.zeros(m, np.float32), np.zeros(51, np.float32))
    check_evaluation(leaf, good, 51)
    with pytest.raises(ValueError, match="policy logits for"):
        check_evaluation(leaf, Evaluation(np.zeros(m + 1, np.float32), np.zeros(51, np.float32)), 51)
    with pytest.raises(ValueError, match="not finite"):
        check_evaluation(leaf, Evaluation(np.full(m, np.nan, np.float32), np.zeros(51, np.float32)), 51)
    with pytest.raises(ValueError, match="value logits"):
        check_evaluation(leaf, Evaluation(np.zeros(m, np.float32), np.zeros(50, np.float32)), 51)


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


def test_check_evaluation_refuses_a_wrong_submit_logit():
    s = synthetic_state(41, 3)
    m = int(s.moves.shape[0])
    with pytest.raises(ValueError, match="offers no legal submit"):
        check_evaluation(Leaf(1, s), Evaluation(np.zeros(m, np.float32), np.zeros(51, np.float32), submit_logit=0.5), 51)
    with pytest.raises(ValueError, match="infinite"):
        check_evaluation(Leaf(2, _shift(s, True)), Evaluation(np.zeros(m, np.float32), np.zeros(51, np.float32), submit_logit=np.inf), 51)
    check_evaluation(Leaf(3, _shift(s, True)), Evaluation(np.zeros(m, np.float32), np.zeros(51, np.float32)), 51)  # NaN: none given


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
