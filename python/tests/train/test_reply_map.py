"""Spec section 15.2's D4 row: "Each mapped reply square's descendant is the reply's square, on
constructed turns with travel; ambiguous mappings masked | The reply head's move space is exact".

The turns are this project's own, played through the engine from handwritten 5dpgn (never the
oracle corpus): a branching jump (White and Black acting), two branches into one past board in
one turn (Two Timelines), each mapped at the turn's start and at the decision point inside it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate
from t5d.net.state import PositionState
from t5d.train.losses import distribution_ce
from t5d.train.reply_map import ReplyMap, board_keys, map_reply
from tests.engine.helpers import requires_engine

pytestmark = requires_engine

BRANCHING = """[Board "Standard"]
[Mode "5D"]

1. Nf3 / Nf6
"""

TWO_TIMELINES = """[Board "Standard - Two Timelines"]
[Mode "5D"]

1. (-0T1)e4 (+0T1)d4 / (-0T1)e5 (+0T1)d5
2. (-0T2)Ng1f3 (+0T2)Ng1f3 / (-0T2)Nb8c6 (+0T2)Ng8f6
"""


def _find(game, text: str):
    for m in game.moves():
        if m.text(game.info().even) == text:
            return m
    raise AssertionError(f"{text} is not among {[m.text(game.info().even) for m in game.moves()]}")


def _state(game) -> PositionState:
    return game.position().state


def _play_turn(game, first: list[str]) -> list[PositionState]:
    """Our turn: the moves ``first`` (texts), recording the decision state before each; then
    submit. Returns the decision states."""
    points = []
    for text in first:
        points.append(_state(game))
        game.apply(_find(game, text))
    assert game.info().obligations_met, "the turn's moves meet its obligations"
    game.submit()
    return points


def _play_reply(game, prefer_new: set[tuple[int, int]]) -> tuple[list[PositionState], list[int]]:
    """The opponent's turn, played pseudo-legally: at each decision point a move from a board in
    ``prefer_new`` (keys) if there is one, else the first move, until its obligations are met.
    Returns its decision states and the index played at each."""
    points, played = [], []
    for _ in range(8):
        if game.info().obligations_met:
            break
        state = _state(game)
        moves = game.moves()
        keys = [(m.from_timeline, m.from_half_turn) for m in moves]
        pick = next((i for i, k in enumerate(keys) if k in prefer_new), 0)
        points.append(state)
        played.append(pick)
        game.apply(moves[pick])
    assert game.info().obligations_met
    return points, played


def _walk(later: PositionState, board: int, keys: dict) -> tuple[int, int]:
    """The decision board ``board`` descends from (independently of ``reply_map``), and how many
    lineage steps it took."""
    steps = 0
    while (int(later.timeline[board]), int(later.half_turn[board])) not in keys:
        board = int(later.parent[board])
        assert board >= 0, "every reply board exists now or descends from one that does"
        steps += 1
    return keys[(int(later.timeline[board]), int(later.half_turn[board]))], steps


def _check(decision: PositionState, points: list[PositionState], played: list[int], m: ReplyMap) -> dict:
    """Section 15.2's D4 row on one decision point: every mapped square's descendant is the reply
    square it came from (same cell, lineage from the mapped board), every candidate and played
    move appears mapped exactly when it touches no ambiguous board, and the target is uniform
    over the played moves that survive. Returns counts for the caller's case-specific checks."""
    keys = board_keys(decision)
    tpb = decision.layout.tokens_per_board
    seen = set()
    depths = []
    reached: dict[int, set] = {}
    for x in points:
        lay = x.layout
        for src, dst, _promo in x.moves.tolist():
            for tok in (src, dst):
                b, cell = divmod(int(tok), lay.tokens_per_board)
                db, _ = _walk(x, b, keys)
                reached.setdefault(db, set()).add((int(x.timeline[b]), int(x.half_turn[b])))
    ambiguous = {b for b, boards in reached.items() if len(boards) > 1}
    assert set(m.ambiguous_boards) == ambiguous
    expected_moves = []
    for x in points:
        lay = x.layout
        for src, dst, promo in x.moves.tolist():
            ends = []
            for tok in (src, dst):
                b, cell = divmod(int(tok), lay.tokens_per_board)
                db, steps = _walk(x, b, keys)
                ends.append((db, cell, steps))
            if any(db in ambiguous for db, _, _ in ends):
                continue
            # The content, independently of any lineage walk: the moving piece stands on the same
            # cell of the decision board it maps to (our turn moved none of the opponent's pieces,
            # and its own moves this turn left the boards it moves on untouched).
            sb, scell = divmod(int(src), lay.tokens_per_board)
            assert int(x.piece[sb, scell]) == int(decision.piece[ends[0][0], scell]) != 0, (src, dst)
            assert int(x.colour[sb, scell]) == int(decision.colour[ends[0][0], scell]), (src, dst)
            t = (ends[0][0] * tpb + ends[0][1], ends[1][0] * tpb + ends[1][1], promo)
            if t not in seen:
                seen.add(t)
                expected_moves.append(t)
                depths.extend(steps for _, _, steps in ends)
    assert [tuple(r) for r in m.moves.tolist()] == expected_moves
    for src, dst, _ in m.moves.tolist():  # the mapped squares lie on the decision state's squares
        assert decision.layout.is_square_token(np.array([src, dst])).all()
    hits = []
    for x, i in zip(points, played, strict=True):
        src, dst, promo = x.moves[i].tolist()
        xtpb = x.layout.tokens_per_board
        ends = [_walk(x, int(t) // xtpb, keys)[0] for t in (src, dst)]
        if not any(db in ambiguous for db in ends):
            cells = [int(t) % xtpb for t in (src, dst)]
            hits.append(expected_moves.index((ends[0] * tpb + cells[0], ends[1] * tpb + cells[1], promo)))
    expected = np.zeros(len(expected_moves))
    for i in hits:
        expected[i] += 1.0 / len(hits)
    np.testing.assert_allclose(m.target, expected)
    return {"depths": depths, "hits": len(hits), "ambiguous": len(ambiguous), "masked": m.masked}


@pytest.mark.parametrize("black", [False, True], ids=["white-acting", "black-acting"])
def test_a_branching_jumps_reply_maps_through_the_new_timeline(black):
    """A knight jumps back a turn and branches; the opponent replies on the new timeline's board,
    which exists only after our turn - its squares map to the history board it branched from.
    With Black acting (register A7: the network reads Black's view) the map is the same: it
    moves no token."""
    from t5d.engine import Game

    with Game.from_pgn(BRANCHING) as g:
        g.set_legality(False)
        g.replay_all()
        if black:
            g.apply(_find(g, "(0T2)Pe2(0T2)e3"))
            g.submit()
            ours = "(0T2)Nf6(0T1)f4"
        else:
            ours = "(0T2)Nf3(0T1)f5"
        (decision,) = _play_turn(g, [ours])
        assert decision.acting_player == int(black)
        start = _state(g)
        new = {k for k in board_keys(start) if k not in board_keys(decision)}
        assert len(new) == 2, "the jump's source board's successor and the branch's first board"
        points, played = _play_reply(g, {k for k in new if k[0] != 0})
    m = map_reply(decision, points, played)
    facts = _check(decision, points, played, m)
    assert facts["ambiguous"] == 0 and facts["masked"] == 0 and facts["hits"] == len(played) >= 1
    assert max(facts["depths"]) == 1 and min(facts["depths"]) == 0, "some reply squares exist now, some descend"
    assert m.target.sum() == pytest.approx(1.0)


def test_two_branches_into_one_board_are_masked_and_the_map_is_per_decision_point():
    """Two branches into the same past board in one turn make two new boards descend from one:
    at the turn's start every reply touching either is masked (here the whole reply - no
    target); at the decision point between the two jumps the first branch's board exists and
    maps to itself, so nothing is ambiguous and the reply has its target."""
    from t5d.engine import Game

    with Game.from_pgn(TWO_TIMELINES) as g:
        g.set_legality(False)
        g.replay_all()
        decisions = _play_turn(g, ["(+0T3)Nf3(-0T1)f3", "(-0T3)Nf3(-0T1)g3"])
        start = _state(g)
        branched = [k for k in board_keys(start) if k not in board_keys(decisions[0]) and k[0] not in (-1, 0)]
        assert len(branched) == 2, "two new timelines"
        points, played = _play_reply(g, set(branched))
    at_start = map_reply(decisions[0], points, played)
    facts = _check(decisions[0], points, played, at_start)
    target_board = board_keys(decisions[0])[(-1, 2)]  # (-0T1), White to move: both jumps' destination
    assert at_start.ambiguous_boards == (target_board,) and facts["masked"] > 0
    assert facts["hits"] == 0 and at_start.target.sum() == 0.0, "every played reply touched a branch: no target"
    whole_turn = map_reply(decisions[0], points, played, ambiguous="turn")
    assert whole_turn.target.sum() == 0.0
    mid = map_reply(decisions[1], points, played)
    facts = _check(decisions[1], points, played, mid)
    assert mid.ambiguous_boards == () and facts["masked"] == 0 and facts["hits"] == len(played)
    assert mid.target.sum() == pytest.approx(1.0) and len(mid.moves) > len(at_start.moves)


def test_the_turn_arm_drops_a_turn_with_one_masked_played_move():
    """A reply with one played move that maps cleanly (an optional move on the +0 timeline's
    present board) and one that touches the ambiguous board: ``ambiguous="move"`` keeps the clean one as
    the whole target; ``"turn"`` drops the turn's target - the candidates are the same under both."""
    from t5d.engine import Game

    with Game.from_pgn(TWO_TIMELINES) as g:
        g.set_legality(False)
        g.replay_all()
        decisions = _play_turn(g, ["(+0T3)Nf3(-0T1)f3", "(-0T3)Nf3(-0T1)g3"])
        start = _state(g)
        branched = {k for k in board_keys(start) if k not in board_keys(decisions[0]) and k[0] not in (-1, 0)}
        moves = g.moves()
        clean = next(i for i, m in enumerate(moves) if (m.from_timeline, m.from_half_turn) == (m.to_timeline, m.to_half_turn) == (0, 7))
        g.apply(moves[clean])  # an optional move on the +0 timeline's present board
        points, played = _play_reply(g, branched)
        points, played = [start, *points], [clean, *played]
    by_move = map_reply(decisions[0], points, played)
    facts = _check(decisions[0], points, played, by_move)
    assert facts["ambiguous"] == 1 and facts["hits"] == 1 < len(played), "one played move kept, the rest masked"
    assert by_move.target.sum() == pytest.approx(1.0)
    by_turn = map_reply(decisions[0], points, played, ambiguous="turn")
    assert by_turn.target.sum() == 0.0, "the turn arm: a masked played move drops the turn"
    np.testing.assert_array_equal(by_turn.moves, by_move.moves)
    assert by_turn.masked == by_move.masked


def test_the_mapped_space_feeds_the_reply_head():
    """The map is the head's move space: put on the decision state as ``opp_moves``, the reply
    head (behind its flag) scores exactly those moves and its loss reads the target."""
    from t5d.engine import Game

    with Game.from_pgn(BRANCHING) as g:
        g.set_legality(False)
        g.replay_all()
        (decision,) = _play_turn(g, ["(0T2)Nf3(0T1)f5"])
        start = _state(g)
        new = {k for k in board_keys(start) if k not in board_keys(decision)}
        points, played = _play_reply(g, {k for k in new if k[0] != 0})
    m = map_reply(decision, points, played)
    state = replace(decision, opp_moves=m.moves)
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, opp_reply=True), seed=0).eval()
    out = model(collate([state]))
    assert out.opp_reply_logits.shape == (1, len(m.moves)) and torch.isfinite(out.opp_reply_logits).all()
    target = torch.from_numpy(m.target).float()[None]
    loss = distribution_ce(out.opp_reply_logits, out.opp_move_valid, target)
    assert torch.isfinite(loss) and float(loss) > 0
    with pytest.raises(ValueError, match="played move"):
        map_reply(decision, points, [len(points[0].moves)] + played[1:])
