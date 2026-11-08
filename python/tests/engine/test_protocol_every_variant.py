"""Section 15.2's protocol names "every variant" among its adversarial constructions. Two rows
held it on a few starts only; this module runs their own checks on random play from every named
variant the engine sets up (``helpers.ENGINE_VARIANTS``), with no change to the code under test
(the gate lane's done-means audit, 2026-10-24).

* **A6** - "Turn-state counts = a brute-force recount from the rules engine's obligations | The
  new inputs are exact". ``test_turn_state_engine.py`` checks the converted counts against the
  expansion, the handle and a recount from the rulebook's definitions on eight plays from five
  starts; here the same check (``_play``) runs from every variant's start under both of Q4's
  readings (moves after the shift allowed - R8, the default - and the Guide-literal reading).
* **D4** - "Each mapped reply square's descendant is the reply's square, on constructed turns
  with travel; ambiguous mappings masked | The reply head's move space is exact".
  ``tests/train/test_reply_map.py`` holds the row on constructed turns (a branching jump, two
  branches into one board) with its checker ``_check`` - written apart from ``reply_map.py``
  but for its one-line ``board_keys``; the moves compared exactly, the target to a tolerance:
  every mapped square's descendant is the reply square it came from (same cell, lineage from
  the mapped board), the
  moving piece stands on that cell of the mapped board, every candidate and played move appears
  mapped exactly when it touches no ambiguous board, and the target is uniform over the played
  moves that survive. Here ``_check`` runs at every decision point of every action of seeded,
  travel-heavy random play from every variant's start, against the opponent's next action as
  played - the random-play smoke sweep the row's constructed turns leave out (non-branching
  jumps, moves between two present boards, rewinding branches, inactive timelines, the T0
  board, every board size).

Random play is for exercising shapes, never evidence about real games (``play.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from t5d.engine import Game, to_position
from t5d.engine.capi import T5D_MOVE_TRAVEL, T5D_RULES_NO_MOVES_AFTER_SHIFT
from t5d.engine.play import ROYAL_IDS
from t5d.train.reply_map import map_reply
from tests.engine.helpers import ENGINE_VARIANTS, requires_engine
from tests.engine.test_turn_state_engine import _play
from tests.train.test_reply_map import _check

pytestmark = requires_engine

A6_CASES = (
    "turn starts",
    "mid-action",
    "past the shift, moves allowed",
    "past the shift, no move",
    "optional boards",
    "a jump between two present boards (-2)",
    "a rewinding branch releasing a board (-> 0)",
    "a rise (T12)",
)


@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_turn_state_counts_equal_a_recount_on_every_variant(variant):
    """A6's row from every variant's start: at every state of two seeded plays per reading (travel
    favoured), the converted counts are the expansion's and the handle's, and the mandatory and
    optional counts equal ``test_turn_state_engine.recount``'s, written from the rulebook."""
    tally = dict.fromkeys(A6_CASES, 0)
    seed = 100 * ENGINE_VARIANTS.index(variant)
    for rules in (0, T5D_RULES_NO_MOVES_AFTER_SHIFT):
        for travel in (0.5, 0.8):
            _play(variant, seed, travel, rules, tally, states=80, max_boards=60)
            seed += 1
    assert tally["turn starts"] > 2 and tally["mid-action"] + tally["past the shift, moves allowed"] > 0, (variant, tally)
    assert tally["past the shift, no move"] > 0, (variant, tally)  # the Guide-literal reading was met


# --- D4: the reply map on random play ------------------------------------------------------------


def _playable(game: Game, x) -> np.ndarray:
    """The moves random play may make: never a royal capture."""
    if not x.move_count:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS))


def _random_play(variant: str, seed: int, max_states: int = 70, boards: int = 18) -> list[tuple]:
    """Seeded pseudo-legal play from ``variant``'s start, travel drawn with probability 0.7 when
    both kinds exist: every state as ``(PositionState, index played)`` - the index into the
    state's own ``moves`` (its expansion's order), -1 for a submission. A move after which the
    action could neither go on nor be submitted (Timeline Tactician's first action, T13) is passed
    over for another; play ends at a turn start with ``boards`` boards, or stuck."""
    rng = np.random.default_rng(seed)
    out: list[tuple] = []
    game = Game.variant(variant)
    game.set_legality(False)
    try:
        for _ in range(max_states):
            info = game.info()
            x = game.expand()
            state = to_position(x, created_timelines=game.created_timelines()).state
            assert len(state.moves) == x.move_count, (variant, "the converted moves are the expansion's, in order")
            if info.boards >= boards and info.moves_made == 0:
                break
            ok = _playable(game, x) if info.may_move else np.zeros(0, dtype=np.int64)
            if info.obligations_met and (not ok.size or rng.random() < 0.4):
                out.append((state, -1))
                game.submit()
                continue
            travel = (x.moves["flags"][ok] & T5D_MOVE_TRAVEL) != 0 if ok.size else np.zeros(0, dtype=bool)
            if travel.any() and (~travel).any():
                first = ok[travel] if rng.random() < 0.7 else ok[~travel]
                order = np.concatenate([rng.permutation(first), rng.permutation(np.setdiff1d(ok, first))])
            else:
                order = rng.permutation(ok)
            chosen = None
            for move in order.tolist():
                child = game.copy()
                child.set_legality(False)
                cx = child.expand(square_features="none")
                assert cx.move_count == x.move_count
                child.apply(int(move))
                ci = child.info()
                fine = ci.obligations_met or (ci.may_move and _playable(child, child.expand(square_features="none")).size > 0)
                child.close()
                if fine:
                    chosen = int(move)
                    break
            if chosen is None:
                break  # stuck: no move lets the action go on
            out.append((state, chosen))
            game.expand(square_features="none")  # the index is into the current state's expansion
            game.apply(chosen)
    finally:
        game.close()
    return out


def _actions(play: list[tuple]) -> list[list[tuple]]:
    """The play cut into actions: each a run of states ending with its submission."""
    actions, current = [], []
    for state, played in play:
        current.append((state, played))
        if played < 0:
            actions.append(current)
            current = []
    return actions


@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_reply_map_holds_on_random_play_from_every_variant(variant):
    """D4's row on random play: at every decision point of an action (each state before one of
    its moves, and the state it is submitted from) the opponent's next action, as played (its
    move points; its submission has no mapped move), maps as ``test_reply_map._check`` requires."""
    facts = {"decision points": 0, "descended": 0, "ambiguous": 0, "masked": 0, "hits": 0}
    for seed in (0, 1):
        actions = _actions(_random_play(variant, 1000 * (ENGINE_VARIANTS.index(variant) + 1) + seed))
        for ours, theirs in zip(actions, actions[1:], strict=False):
            assert int(ours[0][0].acting_player) != int(theirs[0][0].acting_player), variant
            points = [s for s, i in theirs if i >= 0]
            played = [i for _, i in theirs if i >= 0]
            if not points:
                continue
            for decision, _ in ours:
                m = map_reply(decision, points, played)
                got = _check(decision, points, played, m)
                facts["decision points"] += 1
                facts["descended"] += any(d > 0 for d in got["depths"])
                facts["ambiguous"] += got["ambiguous"] > 0
                facts["masked"] += got["masked"]
                facts["hits"] += got["hits"]
    # What the play met when written: 20-34 decision points per variant, reply squares descending
    # from a board of the decision point at 12-20 of them, ambiguous boards in 12 of the 16
    # variants (up to 533 moves masked) - the floors here are far below that.
    assert facts["decision points"] >= 6, (variant, facts)
    assert facts["descended"] > 0, (variant, facts)  # some reply squares map through a board's lineage
    assert facts["hits"] > 0, (variant, facts)  # played replies were mapped into targets
