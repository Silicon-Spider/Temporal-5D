"""The opponent-reply prototype's move space (register D4, spec section 12.4; CHECKPOINT 7).

"Every board the opponent can move on next turn either exists now or descends from one that
does, so each square of a reply maps to the same square on that board, and the target - attached
to each of our decision points in the turn - is the opponent's next turn as a distribution over
its mapped moves. Two new boards descending from one board (two branches into the same past
board in one turn) make the mapping ambiguous, and those replies are masked" (section 12.4).

``map_reply(decision, reply_points, played)`` maps the opponent's next turn onto one of our
decision points:

* **Board identity** is ``(timeline, half_turn)``: boards are immutable (A4), so a board of a
  later state with the key of a board of the decision state is that board.
* **Descent**: a board new since the decision point is mapped to its nearest ancestor that
  exists there, walking the later state's lineage (``PositionState.parent`` - the engine's: a
  timeline's previous board, a branch's origin; C23). A board with no such ancestor (an unknown
  origin) is unmappable, and its moves are masked like ambiguous ones.
* **Squares** map to the same square (cell) of the ancestor; tokens are the decision state's
  (register A7's canonical view moves no token, so the map holds under either acting player).
* **Ambiguity** (a reading - spec issue 232): the map is ambiguous at a decision-point board
  when two distinct boards that a reply move touches (as its source or destination) map onto
  it - two new boards from one (two branches into one past board), or a board and its own new
  descendant. Every candidate or played move that touches an ambiguous board is masked: never
  offered to the head, never in the target.
* **Candidates** (the head's move space, ``PositionState.opp_moves``): the mapped candidate moves
  of every reply decision point, turn start first, duplicates dropped - a reading (issue 232):
  §12.4 names no candidate set, and the turn's later moves need not be candidates at its start.
* **The target**: uniform over the played moves' mapped entries (the game record's turn; the
  opponent's submit has no mapped move). ``ambiguous="move"`` (the default, a placeholder) drops
  only the masked moves; ``"turn"`` drops the whole turn's target when any played move is masked.

The map is computed per decision point of our turn - each in its own frame (its boards, its
layout), so the same reply maps differently before and after one of our moves creates a board
(a board our earlier move made exists at the later point and maps to itself). Inference never
needs it: the head is a training target's (``NetConfig.opp_reply``, off - a prototype).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..net.state import PositionState

#: How a turn with an ambiguous played move is masked (``map_reply``).
AMBIGUOUS_MODES: tuple[str, ...] = ("move", "turn")


@dataclass(frozen=True)
class ReplyMap:
    """The opponent's next turn mapped onto one decision point."""

    moves: np.ndarray  # (M2, 3) int64: mapped candidate moves (src token, dst token, promotion), the decision state's tokens
    target: np.ndarray  # (M2,) float64: uniform over the played moves' entries; all zero when the turn has no target
    ambiguous_boards: tuple[int, ...]  # decision-state boards two distinct reply boards map onto
    masked: int  # candidate and played moves dropped (ambiguous or unmappable), counted once per distinct move


def board_keys(state: PositionState) -> dict[tuple[int, int], int]:
    """``(timeline, half_turn)`` -> board index."""
    return {(int(p), int(h)): i for i, (p, h) in enumerate(zip(state.timeline.tolist(), state.half_turn.tolist(), strict=True))}


def ancestor_in(decision_keys: dict[tuple[int, int], int], later: PositionState, board: int) -> int:
    """The decision-state board that ``later``'s board ``board`` is or descends from (walking
    ``later.parent``), or -1 if its lineage reaches none."""
    seen = 0
    b = int(board)
    while b >= 0 and seen <= later.n_boards:
        key = (int(later.timeline[b]), int(later.half_turn[b]))
        if key in decision_keys:
            return decision_keys[key]
        b = int(later.parent[b])
        seen += 1
    return -1


def map_reply(
    decision: PositionState, reply_points: Sequence[PositionState], played: Sequence[int], *, ambiguous: str = "move"
) -> ReplyMap:
    """Map the opponent's next turn onto ``decision`` (one of our decision points; module
    docstring). ``reply_points``: the opponent's decision states in its next turn, its turn start
    first, each with its candidate moves (``moves``); ``played[i]``: the index into
    ``reply_points[i].moves`` of the move played there (-1: none - a submit)."""
    if ambiguous not in AMBIGUOUS_MODES:
        raise ValueError(f"ambiguous must be one of {AMBIGUOUS_MODES}")
    if len(played) != len(reply_points):
        raise ValueError("one played move (or -1) per reply decision point")
    keys = board_keys(decision)
    tpb = decision.layout.tokens_per_board

    # Per reply point, each move as (src ancestor, src cell, dst ancestor, dst cell, promotion,
    # src key, dst key): the decision-state board each end maps to (-1: none) and its own key.
    mapped: list[list[tuple[int, int, int, int, int, tuple, tuple]]] = []
    touched: dict[int, set[tuple[int, int]]] = {}  # decision board -> the distinct reply boards (keys) mapping onto it
    for point in reply_points:
        if (point.width, point.height) != (decision.width, decision.height):
            raise ValueError("the reply's boards are not the decision state's size")
        ptpb = point.layout.tokens_per_board
        rows = []
        for src, dst, promo in point.moves.tolist():
            (sb, sc), (db, dc) = divmod(int(src), ptpb), divmod(int(dst), ptpb)
            ends = []
            for b in (sb, db):
                key = (int(point.timeline[b]), int(point.half_turn[b]))
                anc = ancestor_in(keys, point, b)
                if anc >= 0:
                    touched.setdefault(anc, set()).add(key)
                ends.append((anc, key))
            (sa, skey), (da, dkey) = ends
            rows.append((sa, sc, da, dc, int(promo), skey, dkey))
        mapped.append(rows)
    ambiguous_boards = tuple(sorted(b for b, reply_boards in touched.items() if len(reply_boards) > 1))
    bad = set(ambiguous_boards)

    def triple(row) -> tuple[int, int, int] | None:
        sa, sc, da, dc, promo, _, _ = row
        if sa < 0 or da < 0 or sa in bad or da in bad:
            return None
        return (sa * tpb + sc, da * tpb + dc, promo)

    candidates: dict[tuple[int, int, int], int] = {}
    dropped: set[tuple] = set()
    for rows in mapped:
        for row in rows:
            t = triple(row)
            if t is None:
                dropped.add((row[5], row[1], row[6], row[3], row[4]))
            elif t not in candidates:
                candidates[t] = len(candidates)
    target = np.zeros(len(candidates), dtype=np.float64)
    hits: list[int] = []
    any_masked = False
    for rows, index in zip(mapped, played, strict=True):
        if index < 0:
            continue
        if not 0 <= index < len(rows):
            raise ValueError(f"played move {index} is not one of the reply point's {len(rows)} candidate moves")
        t = triple(rows[index])
        if t is None:
            any_masked = True
        else:
            hits.append(candidates[t])
    if hits and not (ambiguous == "turn" and any_masked):
        for i in hits:
            target[i] += 1.0 / len(hits)
    moves = np.array(list(candidates), dtype=np.int64).reshape(-1, 3)
    return ReplyMap(moves=moves, target=target, ambiguous_boards=ambiguous_boards, masked=len(dropped))


__all__ = ["AMBIGUOUS_MODES", "ReplyMap", "ancestor_in", "board_keys", "map_reply"]
