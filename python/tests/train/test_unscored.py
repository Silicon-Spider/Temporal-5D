"""Review decision 13's target side (register H22, I15; 2026-11-15): a game that ended unscored has
no outcome-derived target - its WDL, moves-left and short-term values stay masked, and no deblunder
blend gives it one. ``TargetConfig.unscored_whole_blend`` is the reading built before (issue
235(g)): a whole blend there gives that decision and every earlier one the best child's WDL and
moves-left. A scored game is the same under either."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from t5d.train.targets import TargetConfig, build_targets
from tests.train.test_targets import _line

# Turn 2's player (White) plays child 1 at q 0.3 below the best's: a whole blend at the defaults
# (Lc0's threshold 0.1, band 0.05) - the game of test_targets' deblundering test.
VALUES = [0.0, 0.0, 0.5, 0.0]
Q_PLAYED = [-0.05, -0.05, 0.2, -0.05]
PLAYED = [0, 0, 1, 0]
ARM = TargetConfig(unscored_whole_blend=True)


def _game(end, winner=None):
    return _line(4, values=VALUES, played=PLAYED, q_played=Q_PLAYED, end=end, winner=winner)


@pytest.mark.parametrize("end", ["unknown", "action_limit"])
def test_an_unscored_game_has_no_outcome_target_and_takes_no_blend(end):
    t = build_targets(_game(end), TargetConfig())
    assert np.isnan(t.wdl).all() and np.isnan(t.moves_left).all() and np.isnan(t.short_term).all()
    assert t.deblunder_weight.tolist() == [0.0, 0.0, 0.0, 0.0]  # nothing blended


@pytest.mark.parametrize("end", ["unknown", "action_limit"])
def test_the_arm_keeps_the_whole_blend_built_before(end):
    """Under the arm the blunder and every earlier decision take the best child's WDL, in their own
    acting player's view, and its moves-left; the decision after the blunder keeps nothing."""
    t = build_targets(_game(end), ARM)
    assert t.deblunder_weight.tolist() == [0.0, 0.0, 1.0, 0.0]
    best_white = np.array([(1 - 0.2 + 0.5) / 2, 0.2, (1 - 0.2 - 0.5) / 2])  # the best child's (q, d), White's view
    for i in range(3):
        mine = best_white if i % 2 == 0 else best_white[::-1]
        assert t.wdl[i] == pytest.approx(mine, abs=1e-12)
        assert np.isfinite(t.moves_left[i])
    assert np.isnan(t.wdl[3]).all() and np.isnan(t.moves_left[3])
    assert np.isnan(t.short_term).all()  # the arm blends WDL and moves-left only, as before


@pytest.mark.parametrize("end,winner", [("checkmate", 1), ("turn_cap", None), ("resign", 0)])
def test_a_scored_game_is_the_same_under_either_reading(end, winner):
    off, on = build_targets(_game(end, winner), TargetConfig()), build_targets(_game(end, winner), ARM)
    for f in fields(off):
        a, b = getattr(off, f.name), getattr(on, f.name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True), f.name
    assert off.deblunder_weight[2] == 1.0


def test_policy_and_search_targets_survive_an_unscored_ending():
    """Decision 13 masks what reads the result, nothing else: the pruned visits, the policy target
    and the search values are the scored game's."""
    unscored, scored = build_targets(_game("unknown"), TargetConfig()), build_targets(_game("checkmate", 1), TargetConfig())
    assert np.array_equal(unscored.search_value, scored.search_value)
    assert np.array_equal(unscored.categorical, scored.categorical)
    assert all(np.array_equal(a, b) for a, b in zip(unscored.pruned, scored.pruned, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(unscored.policy, scored.policy, strict=True))


def test_the_switch_is_in_the_header_dictionary():
    assert TargetConfig().to_dict()["unscored_whole_blend"] is False
    assert replace(TargetConfig(), unscored_whole_blend=True).to_dict()["unscored_whole_blend"] is True
