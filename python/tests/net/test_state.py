"""The input contract: token layout (A1, B14), features (A2, A3), validation, batching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.net import (
    PIECE_IDS,
    Batch,
    PositionState,
    TokenLayout,
    collate,
    permute_boards,
    relayout,
    synthetic_state,
    timeline_index,
    translate,
)
from t5d.net.state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_PAD, TOKEN_SQUARE


def _two_board_state(**overrides) -> PositionState:
    """Timeline 0 at h = 0, 1 (head at h = 1); a white king on e1 at h = 0 and on e2 at h = 1."""
    piece = np.zeros((2, 64), dtype=np.int64)
    colour = np.zeros((2, 64), dtype=np.int64)
    moved = np.zeros((2, 64), dtype=bool)
    piece[0, 4], colour[0, 4] = PIECE_IDS["K"], 1  # e1
    piece[1, 12], colour[1, 12], moved[1, 12] = PIECE_IDS["K"], 1, True  # e2
    piece[:, 60], colour[:, 60] = PIECE_IDS["K"], 2  # e8
    kwargs = dict(
        width=8,
        height=8,
        timeline=[0, 0],
        half_turn=[0, 1],
        is_head=[False, True],
        active=[True, True],
        side_to_move=[0, 1],
        piece=piece,
        colour=colour,
        moved=moved,
        moves=[[66 + 60, 66 + 59, 0]],  # black king e8 -> d8 on the head board
    )
    kwargs.update(overrides)
    return PositionState(**kwargs)


@pytest.mark.parametrize("n", [1, 2, 17, 60, 240])
def test_token_count_is_66n_plus_16(n):
    state = synthetic_state(n, n)
    assert state.n_tokens == 66 * n + 16
    assert state.layout.n_tokens == n * (8 * 8 + 2) + 16


def test_token_count_small_board():
    lay = TokenLayout(n_boards=3, width=5, height=5)
    assert lay.n_tokens == 3 * (25 + 2) + 16


def test_board_major_order_and_square_order():
    lay = TokenLayout(n_boards=4, width=8, height=8)
    # a1..h8 within a board: s = y*W + x.
    assert lay.square_token(0, 0, 0) == 0  # a1
    assert lay.square_token(0, 7, 0) == 7  # h1
    assert lay.square_token(0, 0, 1) == 8  # a2
    assert lay.square_token(0, 7, 7) == 63  # h8
    # Board-major: board b's squares, then its two registers, then the next board.
    assert lay.board_register_token(0, 0) == 64
    assert lay.board_register_token(0, 1) == 65
    assert lay.square_token(1, 0, 0) == 66
    assert lay.square_token(3, 4, 3) == 3 * 66 + 3 * 8 + 4
    # Global registers last.
    assert lay.global_register_token(0) == 4 * 66
    assert lay.global_register_token(15) == 4 * 66 + 15 == lay.n_tokens - 1
    kinds = lay.kinds()
    assert (kinds[:64] == TOKEN_SQUARE).all() and (kinds[64:66] == TOKEN_BOARD_REGISTER).all()
    assert (kinds[-16:] == TOKEN_GLOBAL_REGISTER).all()


def test_token_arrays_carry_board_features():
    state = synthetic_state(3, 9)
    tok = state.token_arrays()
    lay = state.layout
    for b in range(state.n_boards):
        sl = slice(b * lay.tokens_per_board, (b + 1) * lay.tokens_per_board)
        assert (tok["board"][sl] == b).all()
        assert (tok["p"][sl] == state.timeline[b]).all() and (tok["h"][sl] == state.half_turn[b]).all()
        assert (tok["dt"][sl] == state.half_turn[b] - state.h_present).all()
        squares = np.arange(64)
        assert (tok["x"][sl][:64] == squares % 8).all() and (tok["y"][sl][:64] == squares // 8).all()
        assert (tok["x"][sl][64:] == 0).all() and (tok["reg_index"][sl][64:] == [0, 1]).all()
        assert (tok["piece"][sl][:64] == state.piece[b]).all()
    # Global registers belong to no board and carry no board features.
    assert (tok["board"][-16:] == -1).all() and (tok["reg_index"][-16:] == np.arange(16)).all()
    assert (tok["is_head"][-16:] == 0).all() and (tok["dt"][-16:] == 0).all()


def test_timeline_label_mapping():
    # Odd start: labels are their own coordinates.
    assert [timeline_index(x) for x in ("-2", "-1", "0", "+1", "1", "+2")] == [-2, -1, 0, 1, 1, 2]
    # Even start: -0 -> -1 and +0 -> 0 (section 12.6), and adjacency is kept.
    labels = ["-2", "-1", "-0", "+0", "+1", "+2"]
    coords = [timeline_index(x, even_start=True) for x in labels]
    assert coords == [-3, -2, -1, 0, 1, 2]
    assert np.all(np.diff(coords) == 1)
    with pytest.raises(ValueError):
        timeline_index("-0")  # an even-start label in an odd-start game
    with pytest.raises(ValueError):
        timeline_index("0", even_start=True)  # ambiguous: which of the pair?
    with pytest.raises(ValueError):
        timeline_index("x1")


def test_present_and_delta_t():
    state = _two_board_state()
    assert state.h_present == 1 and state.player_to_move == 1
    assert state.dt_present.tolist() == [-1.0, 0.0]


def _three_timeline_state(branch_active: bool) -> PositionState:
    """Timeline 0: boards at h = 4, 5 (head 5); timeline -1 (black-created, so it starts
    on a white-to-move board): one board at h = 2; timeline +1 (white-created): one board
    at h = 7. Timeline -1's head is the earliest board of all."""
    n = 4
    piece = np.zeros((n, 64), dtype=np.int64)
    colour = np.zeros((n, 64), dtype=np.int64)
    piece[:, 4], colour[:, 4] = PIECE_IDS["K"], 1
    piece[:, 60], colour[:, 60] = PIECE_IDS["K"], 2
    half_turn = np.array([4, 5, 2, 7])
    return PositionState(
        width=8,
        height=8,
        timeline=[0, 0, -1, 1],
        half_turn=half_turn,
        is_head=[False, True, True, True],
        active=[True, True, branch_active, True],
        side_to_move=half_turn & 1,
        piece=piece,
        colour=colour,
        moved=np.zeros((n, 64), dtype=bool),
    )


def test_inactive_timelines_do_not_define_the_present():
    """Section 2.3: the present is the minimum time among *active* timelines' heads. An
    inactive timeline whose head is the earliest board of all does not move it, nor the
    player to move (the value/WDL perspective, D3/D4), nor Delta-t (A3)."""
    state = _three_timeline_state(branch_active=False)
    assert int(state.half_turn[state.is_head].min()) == 2, "the inactive head is the earliest"
    assert state.h_present == 5 and state.player_to_move == 1
    assert state.dt_present.tolist() == [-1.0, 0.0, -3.0, 2.0]
    tok = state.token_arrays()
    assert (tok["dt"][: 4 * 66].reshape(4, 66) == np.array([-1.0, 0.0, -3.0, 2.0])[:, None]).all()
    batch = collate([state])
    assert batch.coord_ref[0].tolist() == [-1.0, 5.0], "(p_ref, h_present) for recentring"
    # The same boards with that timeline active: now it defines the present.
    active = _three_timeline_state(branch_active=True)
    assert active.h_present == 2 and active.player_to_move == 0
    assert active.dt_present.tolist() == [2.0, 3.0, 0.0, 5.0]


def test_validation_rejects_register_endpoints():
    lay = _two_board_state().layout
    reg = lay.board_register_token(1, 0)
    with pytest.raises(ValueError, match="square token"):
        _two_board_state(moves=[[66 + 60, reg, 0]])
    with pytest.raises(ValueError, match="square token"):
        _two_board_state(moves=[[lay.global_register_token(3), 66 + 59, 0]])
    with pytest.raises(ValueError, match="square token"):
        _two_board_state(edges=[[66 + 60, reg, 1]])
    with pytest.raises(ValueError, match="square token"):
        _two_board_state(opp_moves=[[lay.n_tokens + 5, 0, 0]])


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(side_to_move=[0, 0]), "h & 1"),
        (dict(is_head=[True, True]), "exactly one head"),
        (dict(is_head=[True, False]), "latest board"),
        (dict(active=[True, False]), "disagree"),
        (dict(half_turn=[1, 1], side_to_move=[1, 1]), "two boards"),
        (dict(edges=[[0, 1, 7]]), "relation ids"),
        (dict(edges=[[0, 1, 6]]), "relation ids"),  # the C ABI's reserved travel-threat id (B11)
        (dict(moves=[[66 + 60, 66 + 59, 13]]), "promotion ids"),
    ],
)
def test_validation_rejects_inconsistent_states(overrides, message):
    with pytest.raises(ValueError, match=message):
        _two_board_state(**overrides)


def test_validation_rejects_bad_squares():
    state = _two_board_state()
    piece = state.piece.copy()
    piece[0, 0] = 13
    with pytest.raises(ValueError, match="piece ids"):
        _two_board_state(piece=piece)
    colour = state.colour.copy()
    colour[0, 0] = 1  # a colour on an empty square
    with pytest.raises(ValueError, match="colour exactly"):
        _two_board_state(colour=colour)


def test_collate_layout_and_padding():
    states = [synthetic_state(1, 1), synthetic_state(2, 3), synthetic_state(3, 2)]
    batch = collate(states)
    assert batch.body_len == 3 * 66 and batch.n_tokens == 3 * 66 + 16
    assert batch.n_body.tolist() == [66, 198, 132]
    for i, state in enumerate(states):
        nb = state.layout.n_body
        assert batch.valid[i, :nb].all() and not batch.valid[i, nb : batch.body_len].any()
        assert (batch.kind[i, nb : batch.body_len] == TOKEN_PAD).all()
        assert (batch.kind[i, batch.body_len :] == TOKEN_GLOBAL_REGISTER).all()
        assert batch.board_valid[i].sum() == state.n_boards
        # Moves pass through unchanged: square tokens coincide in both layouts.
        m = len(state.moves)
        assert torch.equal(batch.moves[i, :m], torch.from_numpy(state.moves))
        assert batch.move_valid[i, :m].all() and not batch.move_valid[i, m:].any()
        assert batch.batch_token(i, state.layout.global_register_token(2)) == batch.body_len + 2
        assert torch.equal(batch.coord_ref[i], torch.tensor([state.p_ref, state.h_present], dtype=torch.float32))


def test_batch_to_keeps_every_field():
    batch = collate([synthetic_state(1, 2), synthetic_state(2, 3)])
    moved = batch.to("cpu")
    assert isinstance(moved, Batch) and moved is not batch
    for name, value in vars(batch).items():
        other = getattr(moved, name)
        if isinstance(value, torch.Tensor):
            assert other.device.type == "cpu" and other.dtype == value.dtype and torch.equal(other, value), name
        else:
            assert other == value, name


def test_collate_rejects_mixed_register_counts():
    with pytest.raises(ValueError):
        collate([synthetic_state(1, 2), synthetic_state(2, 2, regs_per_board=4)])


def test_relayout_round_trip():
    state = synthetic_state(4, 7)
    four = relayout(state, regs_per_board=4, global_regs=8)
    assert four.n_tokens == 7 * 68 + 8
    lay4 = four.layout
    b, sq = lay4.split_square_tokens(four.moves[:, 0])
    assert (four.piece[b, sq] == state.piece[b, sq]).all()
    back = relayout(four, 2, 16)
    assert (back.moves == state.moves).all() and (back.edges == state.edges).all()


def test_permute_boards_moves_follow_their_boards():
    state = synthetic_state(5, 12)
    order = np.random.default_rng(0).permutation(state.n_boards)
    perm = permute_boards(state, order)
    lay = state.layout
    old_b, old_sq = lay.split_square_tokens(state.moves[:, 0])
    new_b, new_sq = lay.split_square_tokens(perm.moves[:, 0])
    assert (old_sq == new_sq).all() and (order[new_b] == old_b).all()
    assert (perm.piece[new_b, new_sq] == state.piece[old_b, old_sq]).all()
    assert perm.h_present == state.h_present


def test_translate():
    state = synthetic_state(6, 10)
    moved = translate(state, dp=3, dh=8)
    assert (moved.timeline == state.timeline + 3).all() and (moved.half_turn == state.half_turn + 8).all()
    assert (moved.dt_present == state.dt_present).all()
    with pytest.raises(ValueError):
        translate(state, dh=1)
