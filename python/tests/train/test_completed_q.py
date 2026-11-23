"""Register I21's completed-Q policy target (spec section 12.2): hand-computed answers on
constructed records; the properties the spec claims for it - it is the raw prior where nothing was
visited, it reads which children have values and not how the visits were allocated, steering's
fields cannot move it, it does not zero what the search never reached - and its round trip through
a stored record. Nothing is trained."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from t5d.engine.capi import MOVE_DTYPE
from t5d.net import synthetic_states
from t5d.train.completed_q import (
    VALUE_RANGES,
    CompletedQConfig,
    action_statistics,
    completed_q_policy,
    completed_values,
    mixed_value,
    root_network_value,
    sigma,
)
from t5d.train.records import Children, DecisionRecord, read_game, write_game
from t5d.train.targets import TargetConfig, policy_target, pruned_visits, raw_prior
from t5d.train.trainer import synthetic_record

# --- constructed records ------------------------------------------------------------------------------


def _children(visits, q, *, candidate=None, prior=None, noise_free=None, boost=None, steered=None, forced=None):
    n = len(visits)
    uniform = [1.0 / n] * n if n else []
    return Children(
        candidate=list(range(n)) if candidate is None else candidate,
        legality=[0] * n,
        visits=visits,
        prior=uniform if prior is None else prior,
        prior_noise_free=uniform if noise_free is None else noise_free,
        q=q,
        d=[0.1] * n,
        value_variance=[0.0] * n,
        boost=[0.0] * n if boost is None else boost,
        exploration_scale=[1.0] * n,
        forced_visits=[0] * n if forced is None else forced,
        steered_visits=[0] * n if steered is None else steered,
        child_key=list(range(1, n + 1)),
        evaluated=[True] * n,
        network_wdl=np.tile([0.4, 0.2, 0.4], (n, 1)),
        network_moves_left=[10.0] * n,
        network_error=[math.nan] * n,
        lo=[-1] * n,
        hi=[1] * n,
    )


def _decision(children, logits, *, wdl=(0.0, 0.0, 0.0), submit_logit=math.nan, submit=False, root_q=0.0):
    m = len(logits)
    moves = np.zeros(m, dtype=MOVE_DTYPE)
    moves["from_square"] = np.arange(m) % 64
    moves["to_square"] = (np.arange(m) + 8) % 64
    return DecisionRecord(
        decision=0,
        action=0,
        move_in_action=0,
        acting_player=0,
        moves_made=0,
        n_boards=1,
        present=2,
        submit_possible=submit,
        submit_legal=submit,
        full_search=True,
        playout_cap=100,
        playouts=int(np.sum(children.visits)),
        stop="playouts",
        move_temperature=1.0,
        root_policy_temperature=1.0,
        root_noise=True,
        search_seed=0,
        root_key=0,
        epoch=0,
        raw_error=math.nan,
        raw_moves_left=10.0,
        raw_submit_logit=submit_logit,
        root_evaluated=True,
        root_visits=int(np.sum(children.visits)) + 1,
        root_q=root_q,
        root_d=0.0,
        root_cpuct=1.5,
        root_best_since=1,
        best_child=int(np.argmax(children.visits)) if len(children) else -1,
        played_child=0 if len(children) else -1,
        choice="sampled",
        verified_illegal=0,
        raw_wdl_logits=np.array(wdl, dtype=np.float64),
        raw_value_logits=np.zeros(51),
        root_wdl=np.full(3, 1 / 3),
        boards=np.array([(0, 2)]),
        candidates=moves,
        raw_logits=np.array(logits, dtype=np.float64),
        children=children,
    )


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


LOGITS = [0.0, math.log(2.0), 0.0]  # the raw prior 1/4, 1/2, 1/4

# --- the paper's formulas, by hand --------------------------------------------------------------------


def test_the_target_is_the_softmax_of_the_logits_plus_sigma_by_hand():
    """Three moves, the middle one unvisited. v_net = 0 (a level raw WDL); the visited moves carry
    prior 1/4 each and values 0.2 and -0.4, so their prior-weighted mean is -0.1 and, with 5
    visits, v_mix = (0 + 5 x -0.1) / 6. sigma's scale is c_visit + the largest count = 54, on
    values mapped onto [0, 1]."""
    d = _decision(_children([4, 0, 1], [0.2, 9.9, -0.4]), LOGITS)
    v_mix = (0.0 + 5 * -0.1) / 6
    completed = np.array([0.2, v_mix, -0.4])
    stored = np.asarray(LOGITS, dtype=np.float32).astype(np.float64)  # the record keeps raw logits as float32
    want = _softmax(stored + 54.0 * (completed + 1.0) / 2.0)
    got = completed_q_policy(d)
    np.testing.assert_allclose(got[:3], want, rtol=0, atol=1e-15)
    assert got[3] == 0.0  # no submit offered
    assert math.isclose(float(got.sum()), 1.0, abs_tol=1e-15)
    values, mix = completed_values(d)
    assert math.isclose(mix, v_mix, abs_tol=1e-15)
    np.testing.assert_allclose(values[:3], completed, rtol=0, atol=1e-15)
    # the unvisited move's recorded q (9.9, garbage) is never read
    assert math.isnan(action_statistics(d)[1][1])


def test_the_mixed_value_is_the_papers_interpolation():
    prior = np.array([0.1, 0.6, 0.3])
    q = np.array([0.5, math.nan, -0.25])
    visits = np.array([3.0, 0.0, 9.0])
    mean = (0.1 * 0.5 + 0.3 * -0.25) / 0.4
    assert math.isclose(mixed_value(prior, q, visits, 0.3), (0.3 + 12 * mean) / 13, abs_tol=1e-15)
    # nothing visited: the network's own value
    assert mixed_value(prior, q, np.zeros(3), 0.3) == 0.3
    # visited actions with no prior mass cannot weight a mean
    assert mixed_value(np.array([0.0, 1.0, 0.0]), q, visits, 0.3) == 0.3


def test_sigma_scales_with_the_largest_count_and_the_value_range():
    q = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(sigma(q, 10.0, CompletedQConfig()), [0.0, 30.0, 60.0])
    np.testing.assert_allclose(sigma(q, 10.0, CompletedQConfig(value_range="signed")), [-60.0, 0.0, 60.0])
    np.testing.assert_allclose(sigma(q, 0.0, CompletedQConfig(c_visit=4.0, c_scale=0.5)), [0.0, 1.0, 2.0])


def test_the_roots_own_value_is_w_minus_l_of_its_raw_wdl():
    d = _decision(_children([1, 1, 1], [0.0, 0.0, 0.0]), LOGITS, wdl=(math.log(6.0), 0.0, math.log(3.0)), root_q=-0.7)
    assert math.isclose(root_network_value(d), 0.6 - 0.3, abs_tol=1e-7)  # the record keeps the WDL logits as float32
    # no raw WDL in the record: the recorded root search value stands in
    assert root_network_value(replace(d, raw_wdl_logits=np.full(3, math.nan))) == -0.7


# --- the properties section 12.2 claims ---------------------------------------------------------------


def test_with_nothing_visited_the_target_is_the_raw_prior():
    d = _decision(_children([0, 0, 0], [0.9, -0.9, 0.3]), LOGITS, wdl=(1.0, 0.0, -2.0))
    np.testing.assert_allclose(completed_q_policy(d), raw_prior(d), rtol=0, atol=1e-15)
    # and with no root children at all
    d = _decision(_children([], []), LOGITS)
    np.testing.assert_allclose(completed_q_policy(d), raw_prior(d), rtol=0, atol=1e-15)


def test_it_reads_which_children_have_values_not_how_the_visits_were_allocated():
    """Every move visited, so no mixed value enters: two allocations with one largest count and the
    same values give one target - where the visit distribution's targets differ."""
    q = [0.3, -0.2, 0.1]
    a = _decision(_children([40, 5, 5], q), LOGITS)
    b = _decision(_children([40, 39, 1], q), LOGITS)
    np.testing.assert_array_equal(completed_q_policy(a), completed_q_policy(b))
    cfg = TargetConfig(prune_single_visit=False, forced_k=0.0)
    va, vb = policy_target(a, pruned_visits(a, cfg)), policy_target(b, pruned_visits(b, cfg))
    assert abs(float(va[1]) - float(vb[1])) > 0.3


def test_steerings_fields_cannot_move_it():
    """The search's prior (noised, boosted), the boosts and the steered and forced counts are not
    read: I2's exclusion and provenance have nothing to act on."""
    visits, q = [10, 3, 0], [0.1, 0.4, 0.0]
    plain = _decision(_children(visits, q), LOGITS)
    steered = _decision(
        _children(visits, q, prior=[0.1, 0.8, 0.1], noise_free=[0.2, 0.7, 0.1], boost=[0.0, 0.5, 0.0], steered=[0, 3, 0], forced=[2, 0, 0]),
        LOGITS,
    )
    np.testing.assert_array_equal(completed_q_policy(plain), completed_q_policy(steered))


def test_a_better_value_raises_a_visited_moves_share():
    low = completed_q_policy(_decision(_children([6, 6, 6], [0.0, 0.0, 0.0]), LOGITS))
    high = completed_q_policy(_decision(_children([6, 6, 6], [0.0, 0.0, 0.05]), LOGITS))
    assert high[2] > low[2] and high[0] < low[0] and high[1] < low[1]


def test_it_does_not_zero_what_the_search_never_reached():
    """Five moves, two visited. The visit target gives the other three nothing; this one leaves
    them the prior's shape among themselves."""
    logits = [0.0, 1.0, 2.0, 0.5, -1.0]
    d = _decision(_children([7, 2, 0, 0, 0], [0.1, 0.0, 0.0, 0.0, 0.0]), logits)
    got = completed_q_policy(d)
    assert (got[:5] > 0).all()
    prior = raw_prior(d)
    np.testing.assert_allclose(got[2:5] / got[2:5].sum(), prior[2:5] / prior[2:5].sum(), rtol=1e-12)
    visit = policy_target(d, pruned_visits(d, TargetConfig(prune_single_visit=False, forced_k=0.0)))
    assert (visit[2:5] == 0).all()


def test_the_signed_range_is_the_unit_range_at_twice_the_scale():
    d = _decision(_children([9, 4, 0], [0.3, -0.6, 0.0]), LOGITS, wdl=(0.5, 0.0, 0.1))
    unit = completed_q_policy(d, CompletedQConfig(c_scale=1.0))
    signed = completed_q_policy(d, CompletedQConfig(c_scale=0.5, value_range="signed"))
    np.testing.assert_allclose(signed, unit, rtol=1e-12, atol=0)


# --- submit, the flat form, the configuration ---------------------------------------------------------


def test_submit_is_an_action_like_any_other():
    """A legal submit with a raw logit takes the last entry (D8); its child is found by candidate -1."""
    children = _children([5, 1, 6], [0.0, 0.0, 0.2], candidate=[0, 1, -1])
    d = _decision(children, [0.0, 0.0], submit=True, submit_logit=0.0)
    got = completed_q_policy(d)
    assert got.shape == (3,) and math.isclose(float(got.sum()), 1.0, abs_tol=1e-15)
    want = _softmax(np.log([1 / 3] * 3) + 56.0 * (np.array([0.0, 0.0, 0.2]) + 1.0) / 2.0)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
    # not offered: no mass, whatever the record's children say
    d = _decision(_children([5, 1], [0.0, 0.0]), [0.0, 0.0], submit=False, submit_logit=3.0)
    assert completed_q_policy(d)[2] == 0.0
    # offered with no logit from the network: the search's placeholder share, as raw_prior reads it
    d = _decision(children, [0.0, 0.0], submit=True)
    assert completed_q_policy(d)[2] == 0.0
    assert completed_q_policy(d, CompletedQConfig(submit_share=0.25))[2] > 0.0


def test_a_flat_decision_has_no_target_yet():
    """D2's flat form: the root children are whole turns, and the reading over them is open."""
    rec = synthetic_record(4, synthetic_states(4, [3, 5], n_timelines=3), full_every=4, flat=True)
    assert all(d.flat for d in rec.decisions)
    assert all(completed_q_policy(d) is None and completed_values(d) is None for d in rec.decisions)


def test_the_configuration_is_validated():
    assert VALUE_RANGES == ("unit", "signed")
    for bad in (
        CompletedQConfig(c_visit=-1.0),
        CompletedQConfig(c_scale=math.inf),
        CompletedQConfig(value_range="half"),
        CompletedQConfig(submit_share=1.5),
    ):
        with pytest.raises(ValueError):
            completed_q_policy(_decision(_children([1], [0.0]), [0.0]), bad)
    assert CompletedQConfig().to_dict() == {"c_visit": 50.0, "c_scale": 1.0, "value_range": "unit", "submit_share": 0.0}


# --- section 12.8: built from the records as they are -------------------------------------------------


@pytest.mark.parametrize("seed", [1, 7])
def test_the_target_is_rebuilt_exactly_from_a_stored_record(seed, tmp_path):
    """Section 15.2's I15 row, for this arm: the target from a record read back from disk is the
    target from the record that was written - it needs nothing the format does not keep."""
    rec = synthetic_record(seed, synthetic_states(seed, [2, 4, 3, 5], n_timelines=3), full_every=2)
    back = read_game(write_game(tmp_path / "game.npz", rec))
    assert len(back.decisions) == len(rec.decisions) > 0
    for cfg in (CompletedQConfig(), CompletedQConfig(c_visit=10.0, c_scale=0.25, value_range="signed", submit_share=0.1)):
        seen = 0
        for a, b in zip(rec.decisions, back.decisions, strict=True):
            ta, tb = completed_q_policy(a, cfg), completed_q_policy(b, cfg)
            assert (ta is None) == (tb is None)
            if ta is None:
                continue
            seen += 1
            np.testing.assert_array_equal(ta, tb)
            assert math.isclose(float(ta.sum()), 1.0, abs_tol=1e-12)
            assert ((ta > 0) == (raw_prior(a, cfg.submit_share) > 0)).all()
        assert seen > 0
