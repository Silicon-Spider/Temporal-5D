"""The synthetic move graph and typed relations (``synthetic.reach_graph``): plausible 4D
geometry with cross-board edges, relations that mean what they say, candidate moves
drawn from the graph. Synthetic, not the rules - see the module docstring."""

from __future__ import annotations

import numpy as np
import pytest

from t5d.net import PIECE_IDS, PositionState, synthetic_state
from t5d.net.state import COLOUR_BLACK, COLOUR_WHITE, RELATION_IDS, RELATION_TYPES
from t5d.net.synthetic import reach_graph

ROYALS = (PIECE_IDS["K"], PIECE_IDS["Y"])


def _split(state, tokens):
    return state.layout.split_square_tokens(tokens)


@pytest.mark.parametrize("n", [5, 17, 60])
def test_edges_are_plausible_relations(n):
    state = synthetic_state(n + 100, n)
    e = state.edges
    sb, ss = _split(state, e[:, 0])
    db, ds = _split(state, e[:, 1])
    src_piece, src_colour = state.piece[sb, ss], state.colour[sb, ss]
    dst_piece, dst_colour = state.piece[db, ds], state.colour[db, ds]
    rel = e[:, 2]
    assert (src_piece != 0).all(), "every relation starts at a piece"
    assert ((state.half_turn[sb] - state.half_turn[db]) % 2 == 0).all(), "time steps are whole turns: same colour to move"
    move, attack, defence = rel == RELATION_IDS["move"], rel == RELATION_IDS["attack"], rel == RELATION_IDS["defence"]
    assert ((dst_piece == 0) | (dst_colour != src_colour))[move].all(), "moves land on empty or enemy squares"
    assert ((dst_piece != 0) & (dst_colour != src_colour))[attack].all(), "attacks land on enemies"
    assert ((dst_piece != 0) & (dst_colour == src_colour))[defence].all(), "defence lands on own pieces"
    check = rel == RELATION_IDS["check"]
    assert (np.isin(dst_piece, ROYALS) & (dst_colour != src_colour))[check].all(), "check lands on an enemy royal"
    # B11 (CHECKPOINT 6): travel-threat is folded into attack - a travel threat is an attack
    # edge onto another board, and no relation of its own (its C id, 6, stays reserved).
    assert (attack & (sb != db)).any(), "cross-board attacks - the travel threats - exist"
    assert set(np.unique(rel).tolist()) <= set(range(len(RELATION_TYPES))) and "travel_threat" not in RELATION_IDS
    sliders = np.isin(src_piece, [PIECE_IDS[c] for c in "RBUDQSY"])
    for name in ("xray", "pin"):
        assert sliders[rel == RELATION_IDS[name]].all(), f"{name} relations come from sliders"
    assert (move & (sb != db)).any(), "cross-board moves exist"
    assert (move & (sb == db)).any(), "same-board moves exist"
    assert len(np.unique(e, axis=0)) == len(e)


def test_relations_switch_and_candidate_moves():
    state = synthetic_state(7, 12)
    only_moves = synthetic_state(7, 12, relations=False)
    assert (only_moves.edges[:, 2] == RELATION_IDS["move"]).all()
    assert np.array_equal(only_moves.edges, state.edges[state.edges[:, 2] == RELATION_IDS["move"]])
    graph = {tuple(x) for x in state.edges[state.edges[:, 2] == RELATION_IDS["move"], :2].tolist()}
    for moves, player in ((state.moves, state.player_to_move), (state.opp_moves, 1 - state.player_to_move)):
        assert all(tuple(m) in graph for m in moves[:, :2].tolist()), "candidate moves are move-graph edges"
        b, s = _split(state, moves[:, 0])
        assert (state.colour[b, s] == (COLOUR_WHITE if player == 0 else COLOUR_BLACK)).all()


def _line_state() -> PositionState:
    """One timeline, three boards (h = 0, 1, 2): a white rook on a1 throughout, a black
    rook on a8 at h = 2, a black king on h8 and a white king on h1 everywhere."""
    n = 3
    piece = np.zeros((n, 64), dtype=np.int64)
    colour = np.zeros((n, 64), dtype=np.int64)
    piece[:, 0], colour[:, 0] = PIECE_IDS["R"], COLOUR_WHITE  # a1
    piece[:, 7], colour[:, 7] = PIECE_IDS["K"], COLOUR_WHITE  # h1
    piece[:, 63], colour[:, 63] = PIECE_IDS["K"], COLOUR_BLACK  # h8
    piece[2, 56], colour[2, 56] = PIECE_IDS["R"], COLOUR_BLACK  # a8 at h = 2
    return PositionState(
        width=8,
        height=8,
        timeline=[0, 0, 0],
        half_turn=[0, 1, 2],
        is_head=[False, False, True],
        active=[True, True, True],
        side_to_move=[0, 1, 0],
        piece=piece,
        colour=colour,
        moved=np.zeros((n, 64), dtype=bool),
    )


def test_a_rook_reaches_along_its_axes_and_is_blocked_by_its_past_self():
    """Section 10.4.2: "a long-stationary piece blocks its own pure-time ray with its past
    selves" - the a1 rook at h = 2 defends its own past self at h = 0 and goes no further."""
    state = _line_state()
    edges = {tuple(x) for x in reach_graph(state).tolist()}
    a1_now = 2 * 66 + 0
    rook_moves = {d for s, d, r in edges if s == a1_now and r == RELATION_IDS["move"]}
    same_board = {2 * 66 + sq for sq in [8 * k for k in range(1, 7)] + list(range(1, 7))}
    assert same_board | {2 * 66 + 56} <= rook_moves, "a2..a7, b1..g1 and the capture on a8"
    assert (a1_now, 2 * 66 + 56, RELATION_IDS["attack"]) in edges
    assert (a1_now, 0 * 66 + 0, RELATION_IDS["defence"]) in edges, "its past self on the h = 0 board"
    assert not any(d // 66 == 0 for d in rook_moves), "no move back in time past its past self"
    assert (a1_now, 2 * 66 + 7, RELATION_IDS["defence"]) in edges, "the king on h1"
    assert (2 * 66 + 56, a1_now, RELATION_IDS["attack"]) in edges, "the black rook attacks back"
    # x-ray through the black rook on a8? Nothing behind it on the a-file: no x-ray from a1.
    assert not any(s == a1_now and r == RELATION_IDS["xray"] and d // 66 == 2 and d % 8 == 0 for s, d, r in edges)


def test_xray_reaches_the_second_piece_only_and_pins_through_to_a_royal():
    """A white rook on a1 with a black pawn on a3, a black king on a5 and a white knight on
    a7: the rook captures on a3, x-rays the king (the second piece) and pins the pawn to
    it - and sees nothing past the second piece."""
    piece = np.zeros((1, 64), dtype=np.int64)
    colour = np.zeros((1, 64), dtype=np.int64)
    for square, name, side in (
        (0, "R", COLOUR_WHITE),
        (16, "P", COLOUR_BLACK),
        (32, "K", COLOUR_BLACK),
        (48, "N", COLOUR_WHITE),
        (7, "K", COLOUR_WHITE),
    ):
        piece[0, square], colour[0, square] = PIECE_IDS[name], side
    state = PositionState(
        width=8,
        height=8,
        timeline=[0],
        half_turn=[0],
        is_head=[True],
        active=[True],
        side_to_move=[0],
        piece=piece,
        colour=colour,
        moved=np.zeros((1, 64), dtype=bool),
    )
    edges = {tuple(x) for x in reach_graph(state).tolist()}
    from_rook = {(d, r) for s, d, r in edges if s == 0}
    assert {(16, RELATION_IDS["move"]), (16, RELATION_IDS["attack"]), (32, RELATION_IDS["xray"]), (16, RELATION_IDS["pin"])} <= from_rook
    assert (48, RELATION_IDS["xray"]) not in from_rook, "nothing past the second piece"
    assert not any(d in (24, 32, 40) and r == RELATION_IDS["move"] for d, r in from_rook), "no move past the first piece"
