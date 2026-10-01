"""The synthetic multiverse generator: seeded, exact in N, structurally plausible."""

from __future__ import annotations

import numpy as np
import pytest

from t5d.net import synthetic_state
from t5d.net.state import PIECE_IDS


def _same(a, b) -> bool:
    fields = ("timeline", "half_turn", "is_head", "active", "piece", "colour", "moved", "edges", "moves", "opp_moves")
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in fields)


def test_seeded_and_deterministic():
    assert _same(synthetic_state(11, 40), synthetic_state(11, 40))
    assert not _same(synthetic_state(11, 40), synthetic_state(12, 40))


@pytest.mark.parametrize("n", [1, 2, 5, 17, 60, 120, 240])
def test_exact_board_count_and_valid(n):
    state = synthetic_state(n, n)
    assert state.n_boards == n
    state.validate()  # also run at construction


def test_timelines_branch_from_earlier_boards():
    state = synthetic_state(0, 240, n_timelines=20)
    timelines = np.unique(state.timeline)
    assert len(timelines) == 20
    all_h = set(state.half_turn.tolist())
    for p in timelines:
        hs = np.sort(state.half_turn[state.timeline == p])
        assert (np.diff(hs) == 1).all(), "a timeline is a contiguous run of half-turns"
        if p != 0:
            assert hs[0] >= 1 and (hs[0] - 1) in all_h, "a branch starts one half-turn after an existing board"
            # White-created timelines (positive) start on black-to-move boards and vice versa.
            assert (hs[0] % 2 == 1) == (p > 0)


def test_heads_at_varying_times_and_inactive_timelines():
    state = synthetic_state(5, 120, n_timelines=30)
    head_h = state.half_turn[state.is_head]
    assert len(set(head_h.tolist())) > 2
    assert not state.active.all(), "an unbalanced timeline count leaves some timelines inactive"
    p = state.timeline
    n_white, n_black = int(p.max()), int(-p.min())
    for label in np.unique(p):
        on = p == label
        expected = label == 0 or (label > 0 and label <= n_black + 1) or (label < 0 and -label <= n_white + 1)
        assert state.active[on].all() == expected


def test_duplicated_boards_and_deep_travel():
    state = synthetic_state(0, 240)
    flat = np.concatenate([state.piece, state.colour, state.moved.astype(np.int64)], axis=1)
    assert len(np.unique(flat, axis=0)) < state.n_boards, "some boards are exact duplicates"
    starts = [state.half_turn[state.timeline == p].min() for p in np.unique(state.timeline) if p != 0]
    assert min(starts) <= 2, "at least one deep travel branch"


def test_even_start_uses_the_pm0_pair():
    state = synthetic_state(3, 30, even_start=True)
    origins = state.half_turn[np.isin(state.timeline, (-1, 0))]
    assert 0 in state.half_turn[state.timeline == -1] and 0 in state.half_turn[state.timeline == 0]
    assert origins.min() == 0
    assert np.all(np.diff(np.unique(state.timeline)) == 1), "timeline coordinates stay adjacent"


def test_moves_and_edges_are_plausible():
    state = synthetic_state(1, 60)
    lay = state.layout
    player = state.player_to_move
    colour = 1 + player
    b, sq = lay.split_square_tokens(state.moves[:, 0])
    assert (state.colour[b, sq] == colour).all(), "the player to move moves its own pieces"
    assert (state.is_head[b]).all() and ((state.half_turn[b] & 1) == player).all()
    db, dsq = lay.split_square_tokens(state.moves[:, 1])
    assert ((state.half_turn[db] & 1) == player).all(), "destinations are boards of the mover's colour"
    promo = state.moves[:, 2] > 0
    assert (state.piece[b[promo], sq[promo]] == PIECE_IDS["P"]).all()
    assert len({tuple(m) for m in state.moves.tolist()}) == len(state.moves)
    ob, osq = lay.split_square_tokens(state.opp_moves[:, 0])
    assert (state.colour[ob, osq] == 2 - player).all()
    assert lay.is_square_token(state.edges[:, :2]).all()
    # Stage N2: the edges are a geometric move graph plus typed relations (reach_graph),
    # distinct, and every candidate move is a move-graph edge (test_structure.py has more).
    assert len(np.unique(state.edges, axis=0)) == len(state.edges)
    graph = {(int(a), int(b)) for a, b, r in state.edges.tolist() if r == 0}
    assert all((int(a), int(b)) in graph for a, b in np.concatenate([state.moves, state.opp_moves])[:, :2].tolist())
