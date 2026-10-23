"""Register A5, piece-centric tokenisation (spec section 14.4; a prototype behind
``NetConfig.piece_centric``, off; dense scope only).

A5's tokens are A1's minus the empty squares no candidate move lands on (``state.to_piece_centric``):
the pieces, the candidate moves' empty destination cells, the registers. The readings behind that -
a capture points at the captured piece's token, a destination token is an empty-square token, cell
order within a board, destinations from the candidate moves - are placeholders (BUILD_LOG issues
244-247). What is tested here:

* **The tokens and the layout (A5.1)**: the count ``T = pieces + D + N R + G``; each board's tokens
  contiguous, cells ascending, its registers after them, the global registers last; every move and
  edge endpoint a token - synthetic positions drop exactly the move edges onto cells that are no
  token; the round trip back to A1 reproduces the moves and edges in their order, and every per-token
  feature is A1's at the same (board, cell); the transforms (board order, translation, the register
  counts, the colour-swapped mirror) commute with the conversion, which is idempotent; section 15.2's
  adversarial constructions; injected mutants of the conversion are caught.
* **The network (A5.2)** - the proposed section 15.2 row (BUILD_LOG proposed edit 190): "piece-centric
  forward = square-centric forward with the non-token squares removed as keys". The reference is the
  same-seed A1 network on the square-centric batch with a caller's ``attn_mask`` that removes the
  dropped squares as keys; its heads read the kept tokens (the per-board and global square pools over
  the square tokens A5 has - BUILD_LOG issue 250). Several seeds, positions from 1 to 40 boards beside
  smaller ones (padding), every attention implementation, stage N2's switches and the attention flags,
  the heads' and inputs' flags that compose (A8, G2, D9, D2's pools, D8's readouts), FP32 at 1e-5 and
  BF16 no further than its own precision, gradients. Mutants of the A5 batch are caught. The policy is
  one logit per candidate move in the engine's order, promotions sharing one destination token apart
  by D1's offset alone; no parameter and no draw changes; the invariance suite and the colour-swapped
  mirror (A7) hold under A5; block-sparse attention stays lossless, sparse and one block at a time.
* **The refusals (A5.3)**: the block-causal scope (config and forward), the cached path, the cache
  bench, D4's reply prototype, a batch of the other tokenisation, a mixed batch (issues 251, 252).
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, mirror, parameter_breakdown, permute_boards, relayout, synthetic_state, translate
from t5d.net.cache import CachedEvaluator
from t5d.net.state import (
    COLOUR_BLACK,
    COLOUR_WHITE,
    NO_PROMOTION,
    PIECE_IDS,
    RELATION_IDS,
    TOKEN_BOARD_REGISTER,
    TOKEN_GLOBAL_REGISTER,
    TOKEN_PAD,
    TOKEN_SQUARE,
    PieceCentric,
    PositionState,
    TokenLayout,
    mirror_tokens,
    piece_centric_cells,
    square_centric_tokens,
    to_piece_centric,
    to_square_centric,
)
from tests.net.test_chunked import N2_COMBOS, _attention_shaped, _combo_id, _Spy
from tests.net.test_d8 import shifted
from tests.net.test_invariance import FLAG_SETS as INVARIANCE_FLAGS
from tests.net.test_invariance import _assert_same
from tests.net.test_perspective import CONSTRUCTIONS, _acting, assert_mirrored, tie

ATOL = 1e-5  # FP32: removing keys changes only the order and length of floating-point sums
TRUNK_ATOL = 3e-5  # the trunk's states (larger than the heads' outputs)

# --- positions -------------------------------------------------------------------------------------------


def a5(state: PositionState) -> PositionState:
    """A synthetic position's A5 view (synthetic data: the dropped edges are dropped)."""
    return to_piece_centric(state, strict=False)


def _empty_board(width: int = 8, height: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(width * height, dtype=np.int64), np.zeros(width * height, dtype=np.int64)


def tiny_state(boards: list[dict], moves=(), *, width: int = 8, height: int = 8, regs_per_board: int = 2, global_regs: int = 16, **kw):
    """A hand-made position: ``boards`` as ``{"p": timeline, "h": half-turn, "head": bool, "cells":
    {cell: (piece letter, colour)}}`` (one timeline's boards in time order), ``moves`` as
    ``(board, cell, board, cell, promotion letter or "")`` in the A1 layout's tokens."""
    n = len(boards)
    piece = np.zeros((n, width * height), dtype=np.int64)
    colour = np.zeros((n, width * height), dtype=np.int64)
    for b, spec in enumerate(boards):
        for cell, (letter, col) in spec.get("cells", {}).items():
            piece[b, cell] = PIECE_IDS[letter]
            colour[b, cell] = col
    h = np.array([s["h"] for s in boards])
    lay = TokenLayout(n, width, height, regs_per_board, global_regs)
    rows = [
        (lay.square_token_from_index(sb, sc), lay.square_token_from_index(db, dc), PIECE_IDS[pr] if pr else NO_PROMOTION)
        for sb, sc, db, dc, pr in moves
    ]
    return PositionState(
        width=width,
        height=height,
        timeline=np.array([s.get("p", 0) for s in boards]),
        half_turn=h,
        is_head=np.array([s["head"] for s in boards]),
        active=np.ones(n, dtype=bool),
        side_to_move=h & 1,
        piece=piece,
        colour=colour,
        moved=np.zeros_like(piece, dtype=bool),
        moves=np.array(rows, dtype=np.int64).reshape(-1, 3),
        regs_per_board=regs_per_board,
        global_regs=global_regs,
        **kw,
    )


W, B = COLOUR_WHITE, COLOUR_BLACK
KINGS = {4: ("K", W), 60: ("K", B)}  # e1, e8


def promotion_state() -> PositionState:
    """A white pawn on a7 promoting on a8 to each of N, B, R, Q - four rows, one destination - beside a
    king move and a capture (the pawn on b7 takes nothing; the rook on h1 takes the knight on h5)."""
    cells = KINGS | {48: ("P", W), 7: ("R", W), 39: ("N", B)}
    moves = [(0, 48, 0, 56, p) for p in "NBRQ"] + [(0, 4, 0, 12, ""), (0, 7, 0, 39, ""), (0, 7, 0, 15, "")]
    return tiny_state([{"h": 0, "head": True, "cells": cells}], moves)


def empty_board_state(regs_per_board: int = 2) -> PositionState:
    """Two boards of one timeline, the first empty and no move landing on it: under A5 it has its
    registers only - or no token at all with ``regs_per_board = 0``."""
    return tiny_state(
        [{"h": 0, "head": False, "cells": {}}, {"h": 1, "head": True, "cells": KINGS | {52: ("P", B)}}],
        [(1, 60, 1, 59, ""), (1, 52, 1, 44, "")],
        regs_per_board=regs_per_board,
    )


#: Section 15.2's adversarial constructions under A5 - the A7 suite's and A5's own.
A5_CONSTRUCTIONS = {
    "standard": lambda s: synthetic_state(s, 12, n_timelines=3),
    "one_board": lambda s: synthetic_state(s, 1),
    "many_timelines": lambda s: synthetic_state(s, 22, n_timelines=11),
    "deep_travel": lambda s: synthetic_state(s, 16, n_timelines=6, deep_travel_prob=1.0),
    "duplicated": lambda s: synthetic_state(s, 14, n_timelines=4, duplicate_prob=0.8),
    "even_start": lambda s: synthetic_state(s, 14, n_timelines=5, even_start=True),
    "fairy": lambda s: synthetic_state(s, 12, n_timelines=3, fairy_prob=0.7),
    "size_5": lambda s: synthetic_state(s, 10, n_timelines=4, width=5, height=5, deep_travel_prob=1.0),
    "size_3": lambda s: synthetic_state(s, 8, n_timelines=3, width=3, height=3),
    "registers_4_0": lambda s: synthetic_state(s, 10, n_timelines=3, regs_per_board=4, global_regs=0),
    "registers_0_16": lambda s: synthetic_state(s, 10, n_timelines=3, regs_per_board=0),
    "submit_only": lambda s: replace(shifted(synthetic_state(s, 9, n_timelines=3), legal=True), moves=np.zeros((0, 3), dtype=np.int64)),
    "promotions": lambda s: promotion_state(),
    "empty_board": lambda s: empty_board_state(),
    "empty_board_no_registers": lambda s: empty_board_state(regs_per_board=0),
}


def _empty_destinations(state: PositionState) -> set[tuple[int, int]]:
    """The distinct (board, cell) pairs an A1 state's moves land on that hold no piece - recounted
    from the move rows, independently of ``to_piece_centric``."""
    tpb = state.layout.tokens_per_board
    out = set()
    for dst in state.moves[:, 1].tolist():
        b, c = divmod(int(dst), tpb)
        if state.piece[b, c] == 0:
            out.add((b, c))
    return out


# --- (a)-(g): the tokens and the layout (A5.1) ----------------------------------------------------------------


@pytest.mark.parametrize("construction", list(A5_CONSTRUCTIONS))
def test_the_token_count(construction):
    """(a) ``T = sum of the pieces + D + N R + G``, D the distinct empty destination cells -
    the layout's and a single position's batch's alike."""
    for seed in (0, 1, 2):
        s = A5_CONSTRUCTIONS[construction](100 + seed)
        p = a5(s)
        d = len(_empty_destinations(s))
        expected = int((s.piece != 0).sum()) + d + s.n_boards * s.regs_per_board + s.global_regs
        assert p.n_tokens == p.layout.n_tokens == expected
        assert collate([p]).n_tokens == expected and p.n_tokens <= s.n_tokens


@pytest.mark.parametrize("construction", list(A5_CONSTRUCTIONS))
def test_the_layout_is_board_major(construction):
    """(b) each board's tokens contiguous - its kept cells ascending, then its registers - and the
    global registers last; ``board_offsets`` where each board starts; the layout's maps agree with
    the per-token arrays the network reads."""
    s = a5(A5_CONSTRUCTIONS[construction](7))
    lay = s.layout
    tok = s.token_arrays()
    kinds = lay.kinds()
    assert np.array_equal(kinds, tok["kind"])
    board, cell, slot = lay.body_maps()
    for b in range(s.n_boards):
        start, stop = int(lay.board_offsets[b]), int(lay.board_offsets[b + 1])
        assert np.all(board[start:stop] == b)
        k = int(s.token_cells[b].sum())
        assert np.array_equal(cell[start : start + k], np.flatnonzero(s.token_cells[b])), "the kept cells, ascending"
        assert np.all(kinds[start : start + k] == TOKEN_SQUARE) and np.all(kinds[start + k : stop] == TOKEN_BOARD_REGISTER)
        assert np.array_equal(slot[start + k : stop], np.arange(s.regs_per_board))
        for r in range(s.regs_per_board):
            assert lay.board_register_token(b, r) == start + k + r
        for c in np.flatnonzero(s.token_cells[b]).tolist():
            t = lay.square_token_from_index(b, c)
            assert board[t] == b and cell[t] == c
    assert lay.n_body == int(lay.board_offsets[-1]) and np.all(kinds[lay.n_body :] == TOKEN_GLOBAL_REGISTER)
    assert np.array_equal(tok["x"][: lay.n_body][cell >= 0], cell[cell >= 0] % s.width)
    assert lay.max_tokens_per_board == int((lay.board_offsets[1:] - lay.board_offsets[:-1]).max())
    with pytest.raises(ValueError, match="different lengths"):
        _ = lay.tokens_per_board  # every caller assuming uniform boards surfaces
    off = np.flatnonzero(~s.token_cells[0])
    if off.size:
        with pytest.raises(ValueError, match="not a token"):
            lay.square_token_from_index(0, int(off[0]))


@pytest.mark.parametrize(
    "construction", [c for c in A5_CONSTRUCTIONS if c not in ("promotions", "empty_board", "empty_board_no_registers")]
)
def test_every_endpoint_is_a_token_and_the_dropped_edges_are_exactly_those_off_it(construction):
    """(c) every move and edge endpoint of the A5 position is one of its square tokens; a synthetic
    position drops exactly its edges with an endpoint that is no token - move edges onto empty cells
    no candidate move lands on (every other relation ends on a piece) - and strict conversion refuses
    them; the opponent's moves likewise (D4's reply has no A5 reading - BUILD_LOG issue 251)."""
    s = A5_CONSTRUCTIONS[construction](11)
    r = to_piece_centric(s, strict=False, return_dropped=True)
    assert isinstance(r, PieceCentric)
    p = r.state
    lay = p.layout
    for name in ("moves", "edges", "opp_moves"):
        arr = getattr(p, name)
        assert lay.is_square_token(arr[:, :2]).all() if arr.size else True
    tpb = s.layout.tokens_per_board
    cells = piece_centric_cells(s)

    def is_token(t):
        b, c = divmod(int(t), tpb)
        return bool(cells[b, c])

    off = [e for e in s.edges.tolist() if not (is_token(e[0]) and is_token(e[1]))]
    assert r.dropped_edges == len(off) == s.edges.shape[0] - p.edges.shape[0]
    assert all(e[2] == RELATION_IDS["move"] for e in off), "only move edges end on an empty cell"
    assert r.dropped_opp_moves == sum(1 for m in s.opp_moves.tolist() if not (is_token(m[0]) and is_token(m[1])))
    if r.dropped_edges or r.dropped_opp_moves:
        with pytest.raises(ValueError, match="strict"):
            to_piece_centric(s)
    assert p.moves.shape == s.moves.shape, "no move is ever dropped"


@pytest.mark.parametrize("construction", list(A5_CONSTRUCTIONS))
def test_the_round_trip_and_every_feature_at_its_cell(construction):
    """(d) back in A1's layout the moves are the original moves, row for row, and the kept edges
    the original edges in their order; every per-token array the network reads (piece, owner,
    moved, x, y, p, h, head, active, side to move, Delta-t, kind, board, register slot) is A1's at the
    same (board, cell), register or global register."""
    for seed in (0, 3):
        s = A5_CONSTRUCTIONS[construction](200 + seed)
        assert_conversion(s, a5(s))


def assert_conversion(s: PositionState, p: PositionState) -> None:
    """``p`` is ``s``'s A5 conversion: the token set, the round trip and every feature."""
    expected_cells = (s.piece != 0).copy()
    for b, c in _empty_destinations(s):
        expected_cells[b, c] = True
    assert np.array_equal(p.token_cells, expected_cells), "the pieces and the empty destinations, no more"
    back = to_square_centric(p)
    assert np.array_equal(back.moves, s.moves), "moves, row for row"
    tpb = s.layout.tokens_per_board

    def kept(e):
        return all(expected_cells[divmod(int(t), tpb)] for t in e[:2])

    assert back.edges.tolist() == [e for e in s.edges.tolist() if kept(e)], "the kept edges, in their order"
    image = square_centric_tokens(p.layout)
    ta, tb = s.token_arrays(), p.token_arrays()
    for name, values in tb.items():
        assert np.array_equal(values, ta[name][image]), name


@pytest.mark.parametrize("construction", ["standard", "deep_travel", "even_start", "size_5", "registers_4_0", "submit_only"])
def test_the_transforms_commute_and_the_conversion_is_idempotent(construction):
    """(e) board order, translation, the register counts and the colour-swapped mirror commute
    with the conversion; converting twice is converting once."""
    s = A5_CONSTRUCTIONS[construction](13)
    p = a5(s)
    order = np.random.default_rng(5).permutation(s.n_boards)
    pairs = [
        (permute_boards(p, order), a5(permute_boards(s, order))),
        (translate(p, dp=3, dh=6), a5(translate(s, dp=3, dh=6))),
        (relayout(p, 4, 0), a5(relayout(s, 4, 0))),
        (relayout(p, 0, 16), a5(relayout(s, 0, 16))),
        (mirror(p), a5(mirror(s))),
        (to_piece_centric(p, strict=True), p),
        (a5(p), p),
    ]
    for x, y in pairs:
        assert np.array_equal(x.token_cells, y.token_cells)
        for name in ("moves", "edges", "opp_moves", "timeline", "half_turn", "piece", "colour"):
            assert np.array_equal(getattr(x, name), getattr(y, name)), name
        assert x.acting_player == y.acting_player and x.regs_per_board == y.regs_per_board
    assert np.array_equal(mirror(mirror(p)).token_cells, p.token_cells)


def test_promotions_captures_and_empty_boards():
    """(f) hand-made: four promotions share their destination's one token; a capture points at the
    captured piece's own token (no token added); a board with no piece and no destination keeps its
    registers - or has no token at all without them - and still collates and runs."""
    s = promotion_state()
    p = a5(s)
    lay = p.layout
    dst = p.moves[:4, 1]
    assert len(set(dst.tolist())) == 1 and lay.split_square_tokens(dst[:1])[1][0] == 56
    capture = p.moves[5]
    assert lay.split_square_tokens(capture[1:2])[1][0] == 39 and p.piece[0, 39] == PIECE_IDS["N"]
    assert p.n_tokens == 5 + 3 + 2 + 16, "five pieces, three empty destinations (a8, e2, h2), registers"
    for regs, expected in ((2, 2), (0, 0)):
        e = a5(empty_board_state(regs))
        assert int(e.layout.board_offsets[1]) == expected, "the empty board's tokens: its registers"
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=True, regs_per_board=regs), seed=1).eval()
        out = model(collate([e, a5(synthetic_state(3, 4, regs_per_board=regs))]))
        assert torch.isfinite(out.board_logits[0]).all() and torch.isfinite(out.policy_logits[0, :2]).all()


def test_travel_lands_a_destination_token_on_a_frozen_board():
    """(f) a cross-board move onto a superseded board (branching travel) puts its landing square's
    token on that board, with the board's own (p, h) and flags."""
    for seed in range(40):
        s = synthetic_state(seed, 16, n_timelines=5, deep_travel_prob=1.0)
        tpb = s.layout.tokens_per_board
        frozen = [m for m in s.moves.tolist() if not s.is_head[m[1] // tpb] and s.piece[m[1] // tpb, m[1] % tpb] == 0]
        if frozen:
            break
    assert frozen, "no synthetic position with travel onto a frozen board"
    p = a5(s)
    image = square_centric_tokens(p.layout)
    row = s.moves.tolist().index(frozen[0])
    t5 = int(p.moves[row, 1])
    assert image[t5] == frozen[0][1]
    tok = p.token_arrays()
    b = frozen[0][1] // tpb
    assert tok["board"][t5] == b and tok["is_head"][t5] == 0 and tok["piece"][t5] == 0
    assert tok["h"][t5] == s.half_turn[b] and tok["p"][t5] == s.canonical_timeline[b]


def _mutants(s: PositionState, p: PositionState) -> dict[str, PositionState]:
    """(g) wrong conversions: a destination dropped (its move then refused), a cell misplaced, a
    destination remapped one token off, a promotion row forgotten."""
    out = {}
    tpb = s.layout.tokens_per_board
    dst = next(m for m in s.moves.tolist() if s.piece[m[1] // tpb, m[1] % tpb] == 0)
    b, c = divmod(dst[1], tpb)
    cells = p.token_cells.copy()
    cells[b, c] = False
    try:
        out["drop_destination"] = replace(
            p, token_cells=cells, moves=p.moves[[i for i, m in enumerate(s.moves.tolist()) if m[1] != dst[1]]]
        )
    except ValueError:
        out["drop_destination"] = None
    spare = np.flatnonzero(~p.token_cells[b])
    if spare.size:
        cells = p.token_cells.copy()
        cells[b, c], cells[b, spare[0]] = False, True  # the same count, the wrong cell
        out["misplace_cell"] = SimpleNamespace(token_cells=cells, moves=p.moves, edges=p.edges, layout=replace(p.layout, cells=cells))
    moves = p.moves.copy()
    moves[0, 1] = moves[0, 1] + 1 if moves[0, 1] + 1 < p.layout.n_body else moves[0, 1] - 1
    out["off_by_one"] = replace(p, moves=moves) if p.layout.is_square_token(moves[:, :2]).all() else None
    return out


def test_mutants_of_the_conversion_are_caught():
    """(g) each wrong conversion fails ``assert_conversion`` or the state's own validation."""
    s = synthetic_state(31, 10, n_timelines=3)
    p = a5(s)
    caught = 0
    for name, m in _mutants(s, p).items():
        if m is None:
            caught += 1  # refused by validation: a move onto a cell that is no token
            continue
        with pytest.raises((AssertionError, ValueError, IndexError)):
            if isinstance(m, SimpleNamespace):
                image = square_centric_tokens(m.layout)
                back = image[m.moves[:, :2]]
                assert np.array_equal(back, s.moves[:, :2]), name
            else:
                assert_conversion(s, m)
        caught += 1
    q = promotion_state()
    forgot = replace(a5(q), moves=a5(q).moves[1:])  # a promotion row forgotten
    with pytest.raises(AssertionError):
        assert_conversion(q, forgot)
    assert caught == 3


def test_validation_refuses_a_piece_off_the_tokens():
    """A piece is always a token (A5): a mask leaving one out is refused."""
    s = synthetic_state(4, 3)
    cells = piece_centric_cells(s)
    b, c = np.argwhere(s.piece != 0)[0]
    cells[b, c] = False
    with pytest.raises(ValueError, match="every occupied cell"):
        PositionState(
            **{
                f: getattr(s, f)
                for f in ("width", "height", "timeline", "half_turn", "is_head", "active", "side_to_move", "piece", "colour", "moved")
            },
            token_cells=cells,
        )


# --- the network: the equivalence (A5.2, test 1) ---------------------------------------------------------------


def _batch_index(states: list[PositionState], batch) -> list[np.ndarray]:
    """Each position's own token -> its batch token."""
    out = []
    for s in states:
        nb = s.layout.n_body
        own = np.arange(s.n_tokens)
        out.append(np.where(own < nb, own, batch.body_len + own - nb))
    return out


def reference(model1, states1: list[PositionState], states5: list[PositionState], batch5=None):
    """The A1 reference of an A5 forward and the kept-token map: the square-centric batch with a
    caller's mask removing every square that is no A5 token as a key (padding is masked already),
    its trunk, and its heads over the kept tokens (``valid`` restricted to them - the pools then read
    the square tokens A5 has). Returns (A5 batch, reference trunk, reference outputs, map (B, L5) ->
    the A1 batch token, -1 at A5 padding)."""
    batch1 = collate(states1)
    batch5 = collate(states5) if batch5 is None else batch5
    bsz, l1, l5 = batch1.batch_size, batch1.n_tokens, batch5.n_tokens
    index = torch.full((bsz, l5), -1, dtype=torch.long)
    keep = torch.zeros(bsz, l1, dtype=torch.bool)
    for i, (s5, own1, own5) in enumerate(zip(states5, _batch_index(states1, batch1), _batch_index(states5, batch5), strict=True)):
        image = square_centric_tokens(s5.layout)  # A5 own token -> A1 own token
        index[i, torch.from_numpy(own5)] = torch.from_numpy(own1[image])
        keep[i, torch.from_numpy(own1[image])] = True
    mask = keep[:, None, None, :].expand(bsz, 1, l1, l1)
    h1 = model1.trunk_output(batch1, attn_mask=mask)
    out1 = model1.heads(h1, replace(batch1, valid=keep & batch1.valid))
    return batch5, h1, out1, index


def assert_equivalent(model5, model1, states1, states5, atol: float = ATOL, trunk_atol: float = TRUNK_ATOL, batch5=None) -> float:
    """A5's forward equals the masked A1 reference at every kept token and every head. Returns the
    largest difference of the heads' outputs."""
    batch5, h1, ref, index = reference(model1, states1, states5, batch5)
    out = model5(batch5, return_trunk=True)
    worst = 0.0

    def close(x, y, what, tol=atol):
        nonlocal worst
        d = float((x - y).abs().max()) if x.numel() else 0.0
        if tol == atol:
            worst = max(worst, d)
        assert d <= tol, (what, d)

    valid5 = index >= 0
    b_idx, t_idx = torch.nonzero(valid5, as_tuple=True)
    close(out.trunk[b_idx, t_idx], h1[b_idx, index[b_idx, t_idx]], "trunk", trunk_atol)
    close(out.square_logits[b_idx, t_idx], ref.square_logits[b_idx, index[b_idx, t_idx]], "square")
    assert torch.equal(out.move_valid, ref.move_valid) and torch.equal(out.board_valid, ref.board_valid)
    for name in ("value_logits", "wdl_logits", "short_term", "value_error", "moves_left_log", "effort_logits", "submit_raw"):
        close(getattr(out, name), getattr(ref, name), name)
    for name in ("policy_logits", "soft_policy_logits", "optimistic_logits"):
        if getattr(out, name) is not None:
            close(getattr(out, name)[out.move_valid], getattr(ref, name)[ref.move_valid], name)
    for name in ("submit_logits", "soft_submit_logits", "optimistic_submit_logits"):
        x, y = getattr(out, name), getattr(ref, name)
        if x is not None:
            assert torch.equal(torch.isinf(x), torch.isinf(y)), name
            close(x[torch.isfinite(x)], y[torch.isfinite(y)], name)
    close(out.board_logits[out.board_valid], ref.board_logits[ref.board_valid], "board")
    close(out.leave_logits[out.board_valid], ref.leave_logits[ref.board_valid], "leave")
    return worst


def _models(seed: int, **flags):
    """The same-seed A5 and A1 networks (identical weights - A5 adds no parameter)."""
    base = dict(trunk_dtype="fp32", n_layers=2) | flags
    m5 = build_model(NetConfig(piece_centric=True, **base), seed=seed).eval()
    m1 = build_model(NetConfig(**base), seed=seed).eval()
    return m5, m1


def _positions(sizes, seed: int, **kw):
    states1 = [synthetic_state(1000 * seed + 37 * i + n, n, **kw) for i, n in enumerate(sizes)]
    return states1, [a5(s) for s in states1]


IMPLS = {
    "dense": dict(attn_impl="dense"),
    "explicit": dict(attn_impl="explicit"),
    "sdpa": dict(attn_impl="sdpa"),
    "chunked": dict(attn_impl="chunked"),
    "block_sparse_1.0": dict(attn_impl="block_sparse", attn_sparse_max_fraction=1.0),
    "block_sparse_0.3": dict(attn_impl="block_sparse", attn_sparse_max_fraction=0.3),
}


@pytest.mark.parametrize("sizes", [(1, 5), (5, 17), (17, 3), (40, 5)], ids=lambda s: "-".join(map(str, s)))
@pytest.mark.parametrize("impl", list(IMPLS))
def test_the_forward_equals_the_masked_square_centric_forward(impl, sizes):
    """Section 15.2's proposed A5 row, on every attention path, two seeds, positions of 1 to 40
    boards beside a smaller or larger one (padding of both kinds)."""
    for seed in (0, 1):
        m5, m1 = _models(seed, **IMPLS[impl])
        states1, states5 = _positions(sizes, seed)
        with torch.no_grad():
            assert_equivalent(m5, m1, states1, states5)


def test_the_forward_equals_the_masked_square_centric_forward_through_flex():
    """The same through B14's FlexAttention score function (eager on CPU: small positions)."""
    m5, m1 = _models(3, attn_impl="flex")
    states1, states5 = _positions((5, 2), 3)
    with torch.no_grad():
        assert_equivalent(m5, m1, states1, states5)


@pytest.mark.parametrize("flags", N2_COMBOS, ids=[_combo_id(f) for f in N2_COMBOS])
def test_the_equivalence_under_stage_n2s_switches(flags):
    """B9 on 0, 2 or 4 heads or B10's routing table, x B11's typed biases x B12's smolgen."""
    m5, m1 = _models(4, **flags)
    states1, states5 = _positions((9, 4), 4)
    with torch.no_grad():
        assert_equivalent(m5, m1, states1, states5)


#: The other flags that compose with A5: the invariance suite's attention sets (B5, no B15 table,
#: routing with typed biases and the pointer submit readout, B17), B12's other form, A8 (its layout
#: holds only occupied cells, all of them tokens - BUILD_LOG issue 254), G2's 8-bit trunk (per-token
#: scales), D9, D2's register pool, D4's soft policy off, the per-board register count.
COMPOSED = {f"inv_{k}": v for k, v in INVARIANCE_FLAGS.items()} | {
    "b12_bilinear": dict(board_pair_smolgen=True, smolgen_form="board_bilinear"),
    "a8": dict(board_layout_embedding=True),
    "a8_centre": dict(board_layout_embedding=True, layout_padding="centre"),
    "g2_fp8": dict(trunk_quant="fp8_e4m3"),
    "d9_optimistic": dict(optimistic_policy=True),
    "d2_leave_registers": dict(leave_pool="registers"),
    "no_soft_policy": dict(soft_policy=False, submit_head="pointer"),
    "post_norm_squared_relu": dict(norm_placement="post", ffn_activation="squared_relu"),
}


@pytest.mark.parametrize("flags", list(COMPOSED))
def test_the_equivalence_under_every_flag_that_composes(flags):
    m5, m1 = _models(5, **COMPOSED[flags])
    states1, states5 = _positions((12, 3), 5)
    states1 = [states1[0], shifted(states1[1], legal=True)]  # submit offered and legal in one
    states5 = [a5(s) for s in states1]
    with torch.no_grad():
        assert_equivalent(m5, m1, states1, states5)


@pytest.mark.parametrize("construction", list(A5_CONSTRUCTIONS))
def test_the_equivalence_on_every_construction(construction):
    """Section 15.2's adversarial constructions (the register arms with networks to match)."""
    s = A5_CONSTRUCTIONS[construction](41)
    m5, m1 = _models(6, regs_per_board=s.regs_per_board, global_regs=s.global_regs)
    with torch.no_grad():
        assert_equivalent(m5, m1, [s], [a5(s)])


def test_the_equivalence_in_bf16():
    """The shipped precision: A5's BF16 forward no further from the masked reference's BF16 forward
    than that is from its own FP32 computation (twice, for the two computations' rounding)."""
    states1, states5 = _positions((12, 4), 7)
    m5, m1 = _models(7, trunk_dtype="bf16")
    f5, f1 = _models(7)
    with torch.no_grad():
        precision = assert_equivalent(f5, m1, states1, states5, atol=float("inf"), trunk_atol=float("inf"))
        assert precision > 1e-4, "BF16 rounds"
        assert_equivalent(m5, m1, states1, states5, atol=2 * precision, trunk_atol=float("inf"))


@pytest.mark.grad
def test_the_gradients_equal_the_masked_reference():
    """Every parameter's gradient - B15's table, B11's typed biases, B12's smolgen among them - is
    the masked reference's."""
    flags = dict(typed_edge_bias=True, board_pair_smolgen=True)
    m5, m1 = _models(8, **flags)
    states1, states5 = _positions((7, 3), 8)
    batch5, h1, ref, _ = reference(m1, states1, states5)
    out = m5(batch5)
    gen = torch.Generator().manual_seed(8)
    weights = {name: torch.randn(getattr(out, name).shape, generator=gen) for name in ("value_logits", "wdl_logits", "board_logits")}

    def loss(o):
        total = sum((getattr(o, n) * w).sum() for n, w in weights.items())
        return total + o.policy_logits[o.move_valid].sum() + o.leave_logits[o.board_valid].sum()

    loss(out).backward()
    loss(ref).backward()
    grads1 = dict(m1.named_parameters())
    checked = 0
    for name, p in m5.named_parameters():
        q = grads1[name]
        if p.grad is None:
            assert q.grad is None or not q.grad.abs().any(), name
            continue
        assert torch.allclose(p.grad, q.grad, atol=2e-4, rtol=1e-3), (name, float((p.grad - q.grad).abs().max()))
        checked += 1
    assert {"displacement.table", "typed_edges.table"} <= {n for n, p in m5.named_parameters() if p.grad is not None}
    assert any(n.startswith("smolgen.") and p.grad is not None for n, p in m5.named_parameters()) and checked > 50


def test_mutants_of_the_a5_batch_are_caught():
    """Test 1 catches a stale coordinate, a wrong board flag on a destination token and a dropped
    piece token."""
    m5, m1 = _models(9)
    states1, states5 = _positions((8,), 9)
    batch = collate(states5)
    dest = int(next(t for t in states5[0].moves[:, 1].tolist() if states5[0].token_arrays()["piece"][t] == 0))
    piece = int(np.flatnonzero(states5[0].token_arrays()["piece"] > 0)[0])
    stale = batch.coords.clone()
    stale[0, dest, 0] += 1.0
    head = batch.is_head.clone()
    head[0, dest] = 1 - head[0, dest]
    dropped = batch.valid.clone()
    dropped[0, piece] = False
    with torch.no_grad():
        assert_equivalent(m5, m1, states1, states5)
        for mutant in (replace(batch, coords=stale), replace(batch, is_head=head), replace(batch, valid=dropped)):
            with pytest.raises(AssertionError):
                assert_equivalent(m5, m1, states1, states5, batch5=mutant)


# --- the policy, the parameters (A5.2, tests 2 and 5) -------------------------------------------------------------


def test_the_policy_is_one_logit_per_candidate_move():
    """``policy_logits`` is ``(B, M)``, row ``i`` move ``i``, the same M as A1's; ``action_logits``
    ``(B, M + 1)``; promotions to different pieces through one destination token differ by D1's
    per-piece offset alone."""
    s = promotion_state()
    m5, m1 = _models(10)
    with torch.no_grad():
        out = m5(collate([a5(s)]), return_trunk=True)
        out1 = m1(collate([s]))
    m = s.moves.shape[0]
    assert out.policy_logits.shape == out1.policy_logits.shape == (1, m) and out.action_logits().shape == (1, m + 1)
    dst = int(a5(s).moves[0, 1])
    offsets = m5.policy.promo(out.trunk[0, dst])
    ids = s.moves[:4, 2]
    logits = out.policy_logits[0, :4]
    assert torch.allclose(logits - logits[0], offsets[ids] - offsets[ids[0]], atol=1e-5)
    assert not torch.allclose(logits, logits[0].expand(4))


def test_no_parameter_and_no_draw_changes():
    """A5 is a pure input change: the same parameter count by component and the same draws, bit for
    bit, with A5 on and off - the v1 and the small network with flags."""
    for flags in (
        dict(),
        dict(n_layers=2, typed_edge_bias=True, board_pair_smolgen=True, board_layout_embedding=True, optimistic_policy=True),
    ):
        on, off = build_model(NetConfig(piece_centric=True, **flags), seed=3), build_model(NetConfig(**flags), seed=3)
        assert parameter_breakdown(on) == parameter_breakdown(off)
        b = dict(off.named_parameters())
        assert len(b) == len(dict(on.named_parameters()))
        for name, p in on.named_parameters():
            assert torch.equal(p, b[name]), name


# --- the invariance suite and the mirror under A5 (A5.2, test 3; section 15.2's A7 row) ---------------------------


def _a5_token_perm(old: PositionState, new: PositionState, order: np.ndarray) -> torch.Tensor:
    """The old token of each token of ``new`` = ``permute_boards(old, order)`` (A5 layouts)."""
    lo, ln = old.layout, new.layout
    board, cell, slot = ln.body_maps()
    out = np.empty(ln.n_tokens, dtype=np.int64)
    sq = cell >= 0
    out[: ln.n_body][sq] = lo.square_tokens(order[board[sq]], cell[sq])
    out[: ln.n_body][~sq] = lo.board_offsets[order[board[~sq]]] + lo.cell_counts[order[board[~sq]]] + slot[~sq]
    out[ln.n_body :] = lo.n_body + np.arange(ln.global_regs)
    return torch.from_numpy(out)


@pytest.mark.parametrize("flags", list(INVARIANCE_FLAGS))
def test_board_storage_order_does_not_matter(flags):
    model = build_model(NetConfig(trunk_dtype="fp32", piece_centric=True, **INVARIANCE_FLAGS[flags]), seed=1).eval()
    base = synthetic_state(21, 24, n_timelines=6)
    for state in (a5(base), a5(shifted(base, legal=None))):
        order = np.random.default_rng(3).permutation(state.n_boards)
        other = permute_boards(state, order)
        with torch.no_grad():
            a, b = model(collate([state])), model(collate([other]))
        _assert_same(a, b, board_perm=torch.from_numpy(order), token_perm=_a5_token_perm(state, other, order))


@pytest.mark.parametrize("recenter", [True, False])
@pytest.mark.parametrize("flags", ["v1", "b17_shaw"])
def test_translation_invariance(flags, recenter):
    model = build_model(NetConfig(trunk_dtype="fp32", piece_centric=True, rope_recenter=recenter, **INVARIANCE_FLAGS[flags]), seed=2).eval()
    state = a5(synthetic_state(22, 20))
    with torch.no_grad():
        a = model(collate([state]))
        for dp, dh in ((7, 30), (-11, 400)):
            b = model(collate([translate(state, dp=dp, dh=dh)]))
            if recenter:
                assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.policy_logits, b.policy_logits)
            else:
                _assert_same(a, b)


def test_batch_composition_does_not_matter():
    """Positions of very different per-board token counts in one batch: each as alone."""
    model = build_model(NetConfig(trunk_dtype="fp32", piece_centric=True), seed=0).eval()
    small = a5(synthetic_state(24, 3, n_moves=2))
    big = a5(synthetic_state(25, 11, fairy_prob=0.0))
    sparse = a5(synthetic_state(26, 6, width=5, height=5))
    with torch.no_grad():
        alone = model(collate([small]))
        together = model(collate([big, small, sparse]))
    assert len({lay.max_tokens_per_board for lay in collate([big, small, sparse]).layouts}) > 1
    assert torch.allclose(alone.value_logits[0], together.value_logits[1], atol=ATOL)
    m = int(alone.move_valid[0].sum())
    assert torch.allclose(alone.policy_logits[0, :m], together.policy_logits[1, :m], atol=ATOL)
    assert torch.allclose(alone.board_logits[0], together.board_logits[1, :3], atol=ATOL)
    nb = small.layout.n_body
    assert torch.allclose(alone.square_logits[0, :nb], together.square_logits[1, :nb], atol=ATOL)


@pytest.mark.parametrize("construction", [c for c in CONSTRUCTIONS if c != "far_size_1"])
def test_the_mirror_under_a5(construction):
    """Section 15.2's A7 row on A5's tokens: a position and its colour-swapped mirror give the same
    outputs, mapped back (the mirror's tokens of a board keep its cell order, so the map is
    ``mirror_tokens`` into the mirror's layout), the acting colour equalised - three seeds."""
    for seed in (0, 1, 2):
        s = CONSTRUCTIONS[construction](1000 * (seed + 1) + 17)
        p = a5(s)
        m = mirror(p)
        assert np.array_equal(m.token_cells, a5(mirror(s)).token_cells)
        cfg = NetConfig(trunk_dtype="fp32", n_layers=2, piece_centric=True, regs_per_board=s.regs_per_board, global_regs=s.global_regs)
        model = tie(build_model(cfg, seed=seed).eval())
        with torch.no_grad():
            assert_mirrored(model(collate([p])), model(collate([m])), torch.from_numpy(mirror_tokens(p.layout)), _acting(p, m))


# --- block-sparse attention under A5 (A5.2, test 4) -------------------------------------------------------------------


def _selected_fractions(states, piece_centric: bool) -> list[float]:
    """Per query block of the first position (40 boards): the share of its key boards the
    move-graph heads' selection reads - checked lossless and of whole boards on the way."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=piece_centric), seed=7)
    batch = collate(states)
    ctx = model.make_context(batch)
    structure, block, kl = ctx.structure, ctx.block_rows, batch.body_len
    lay = batch.layouts[0]
    assert block == 4 * max(-(-x.n_body // x.n_boards) for x in batch.layouts), "4 boards of the mean length"
    assert not piece_centric or block < 4 * lay.max_tokens_per_board
    board = structure.board_idx
    fractions = []
    for start in range(0, kl, block):
        stop = min(kl, start + block)
        rel = structure.role_rows("move_graph", start, stop) & batch.valid[:, None, :]
        idx, valid = structure.key_selection("move_graph", start, stop, rel, kl)
        for b in range(batch.batch_size):
            chosen = set(idx[b][valid[b]].tolist())
            free = structure.free_rows("move_graph")[b, start:stop]
            selecting = (batch.kind[b, start:stop] != TOKEN_PAD) & ~free
            needed = set(torch.nonzero(rel[b][selecting].any(dim=0)[:kl]).flatten().tolist())
            assert needed <= chosen, "lossless"
            boards = {int(board[b, t]) for t in chosen}
            assert chosen == {t for t in range(kl) if int(board[b, t]) in boards and bool(batch.valid[b, t])}, "whole boards"
            if b == 0 and start < int(batch.n_body[0]):
                fractions.append(len(boards) / 40)
    return fractions


def test_block_sparse_is_lossless_and_sparse_under_a5():
    """The key selection over A5's boards of different lengths: lossless, whole boards, and still
    skipping boards - at 40 boards a block's move-graph rows read about as few key boards as A1's
    (0.77 and 0.77 of them on average as built; blocks of the longest board's length read 0.86 -
    BUILD_LOG issue 249)."""
    states = [synthetic_state(70, 40), synthetic_state(71, 9)]
    a1 = _selected_fractions(states, False)
    pc = _selected_fractions([a5(s) for s in states], True)
    assert max(pc) < 1.0 and sum(pc) / len(pc) <= sum(a1) / len(a1) + 0.02, (pc, a1)


@pytest.mark.parametrize("impl", ["block_sparse", "chunked"])
def test_the_network_attends_one_block_at_a_time_under_a5(monkeypatch, impl):
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, piece_centric=True)
    model = build_model(cfg, seed=6).eval()
    batch = collate([a5(synthetic_state(60, 12))])
    block = model.make_context(batch).block_rows
    spy = _Spy(monkeypatch)
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out.value_logits).all()
    assert spy.spans and max(spy.spans) <= block
    masks = spy.per_row_masks()
    assert masks and max(m[-2] for m in masks) <= block


@pytest.mark.grad
def test_no_quadratic_tensor_is_kept_for_backward_under_a5():
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, piece_centric=True), seed=12)
    batch = collate([a5(synthetic_state(95, 24))])
    saved: list[torch.Tensor] = []

    def pack(t: torch.Tensor):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(batch)
    out.value_logits.sum().backward()
    assert saved and not _attention_shaped(saved), _attention_shaped(saved)


# --- the refusals (A5.3) and the tokenisation check ------------------------------------------------------------------


def test_the_block_causal_scope_and_the_cached_path_are_refused():
    """A5 is built for the dense scope only (BUILD_LOG issue 252): the configuration, a forward that
    overrides the scope, the cached evaluator and the cache bench each refuse it, naming the issue; so
    is D4's reply prototype (issue 251). (The engine's cached evaluator:
    ``tests/engine/test_piece_centric_engine.py``.)"""
    with pytest.raises(NotImplementedError, match="issue 252"):
        NetConfig(piece_centric=True, attn_scope="block_causal").validate()
    with pytest.raises(NotImplementedError, match="issue 251"):
        NetConfig(piece_centric=True, opp_reply=True).validate()
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=True), seed=0).eval()
    batch = collate([a5(synthetic_state(1, 4))])
    with pytest.raises(NotImplementedError, match="issue 252"):
        model(batch, scope="block_causal")
    with pytest.raises(NotImplementedError, match="issue 252"):
        CachedEvaluator(model)
    from t5d.net.cache_bench import _setup

    with pytest.raises(NotImplementedError, match="issue 252"):
        _setup(4, None, "fp32", "dense", 0, piece_centric=True)


def test_a_batch_of_the_other_tokenisation_is_refused():
    """In both directions, and a batch that mixes the two."""
    s = synthetic_state(2, 3)
    m5, m1 = _models(0, n_layers=1)
    with pytest.raises(ValueError, match="square-centric"):
        m5(collate([s]))
    with pytest.raises(ValueError, match="piece-centric"):
        m1(collate([a5(s)]))
    with pytest.raises(ValueError, match="mixed"):
        collate([s, a5(s)])
    assert collate([a5(s)]).piece_centric and not collate([s]).piece_centric


def test_the_default_is_square_centric():
    """A5 is off by default: the v1 reads A1's tokens (``T = 66 N + 16``)."""
    assert NetConfig().piece_centric is False
    s = synthetic_state(5, 7)
    assert not s.piece_centric and s.n_tokens == 66 * 7 + 16


def test_relayout_keeps_the_a5_tokens_and_the_evaluator_style_conversion():
    """``relayout`` of an A5 state changes only the registers; an A1 state relaid then converted is
    the A5 state relaid (the engine evaluator's order)."""
    s = synthetic_state(18, 6)
    for regs, glob in itertools.product((0, 2, 4), (0, 16)):
        x = relayout(a5(s), regs, glob)
        assert x.n_tokens == int(x.token_cells.sum()) + regs * s.n_boards + glob
        assert np.array_equal(x.moves, a5(relayout(s, regs, glob)).moves)
