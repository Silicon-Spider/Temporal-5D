"""Stage N4: an engine expansion as the network's ``PositionState`` (``t5d/engine/convert.py``).

Two kinds of check. Small 5DFEN positions whose every feature, token, move, edge and
lineage entry is computed here by hand. And real positions - every variant's start, seeded
random play from each, every position of the records in ``helpers.RECORDS`` - on which each
converted state validates, every board's square features equal an independent decoding of
the engine's own canonical 5DFEN board string, the candidate moves are exactly the engine's
move list in its order, the edges are the move graph and the relations the engine built,
and the W/E scope agrees with the engine's at several k.
"""

from __future__ import annotations

import numpy as np
import pytest

from t5d.engine import Game, PresentMismatch, relations_for, to_position, to_position_state
from t5d.engine.capi import T5D_NO_WINDOW, T5D_RULES_NO_MOVES_AFTER_SHIFT
from t5d.engine.play import random_game
from t5d.net import NetConfig
from t5d.net.state import PIECE_IDS, RELATION_IDS, TokenLayout
from tests.engine.helpers import RECORDS, VARIANTS, check_position, requires_engine

pytestmark = requires_engine

ALL_RELATIONS = ("attack", "defence", "xray")
K_VALUES = (0, 1, 2, 8, 64, T5D_NO_WINDOW)


def _tpb(state) -> int:
    return state.width * state.height + state.regs_per_board


# --- hand-computed positions --------------------------------------------------------------------


def test_one_small_board_by_hand():
    """4x4, odd, one board at T1 with White to move: R a1 (moved), K d1 (unmoved), k b4."""
    with Game.from_fen(["[1k2/4/4/R2K*:0:1:w]"]) as g:
        pos = g.position(relations=ALL_RELATIONS)
    s = pos.state
    assert (s.width, s.height, s.n_boards, s.n_tokens) == (4, 4, 1, 18 + 16)
    assert s.timeline.tolist() == [0] and s.half_turn.tolist() == [2]
    assert s.is_head.tolist() == [True] and s.active.tolist() == [True] and s.side_to_move.tolist() == [0]
    piece, colour, moved = np.zeros(16, int), np.zeros(16, int), np.zeros(16, bool)
    piece[0], colour[0], moved[0] = PIECE_IDS["R"], 1, True  # a1: a rook without '*' has moved
    piece[3], colour[3], moved[3] = PIECE_IDS["K"], 1, False  # d1: K* unmoved
    piece[13], colour[13], moved[13] = PIECE_IDS["K"], 2, True  # b4 = rank 3 * 4 + file 1
    assert s.piece[0].tolist() == piece.tolist() and s.colour[0].tolist() == colour.tolist() and s.moved[0].tolist() == moved.tolist()
    assert s.dt_present.tolist() == [0.0] and s.h_present == 2 and s.player_to_move == 0
    # Rook a1: a2 a3 a4 b1 c1 (d1 own king); king d1: c1 d2 c2 - the engine's order.
    expected = [(0, 4), (0, 8), (0, 12), (0, 1), (0, 2), (3, 2), (3, 7), (3, 6)]
    assert s.moves.tolist() == [[a, b, 0] for a, b in expected]
    # Move graph (relation 0), then the one relation: the rook defends its king along rank 1.
    assert s.edges.tolist() == [[a, b, 0] for a, b in expected] + [[0, 3, RELATION_IDS["defence"]]]
    assert s.parent.tolist() == [-1] and s.creation is None and s.source.tolist() == [-1]
    assert s.opp_moves.shape == (0, 3)
    assert pos.relations == ALL_RELATIONS and pos.origins_unknown_all == 0 and pos.lineage_complete


def test_two_boards_black_to_move_by_hand():
    with Game.from_fen(["[1k2/4/4/R2K*:0:1:w]", "[1k2/4/4/1R1K*:0:1:b]"]) as g:
        pos = g.position()
    s = pos.state
    assert s.half_turn.tolist() == [2, 3] and s.is_head.tolist() == [False, True] and s.side_to_move.tolist() == [0, 1]
    assert s.parent.tolist() == [-1, 0]
    assert s.dt_present.tolist() == [-1.0, 0.0] and pos.present == 3 and pos.to_move == 1
    head = 18  # board 1's first token
    # The black king b4 (cell 13): a4, b3, c4, a3, c3.
    assert s.moves.tolist() == [[head + 13, head + c, 0] for c in (12, 9, 14, 8, 10)]


def test_a_branched_timeline_from_5dfen_has_no_origin():
    """5DFEN records no branch origins (spec issue 25): +1's first board has no parent."""
    boards = ["[1k2/4/4/R2K*:0:1:w]", "[1k2/4/4/1R1K*:0:1:b]", "[2k1/4/4/1R1K*:0:2:w]", "[1k2/4/R3/3K*:+1:1:b]"]
    with Game.from_fen(boards) as g:
        pos = g.position()
    s = pos.state
    assert s.timeline.tolist() == [0, 0, 0, 1] and s.half_turn.tolist() == [2, 3, 4, 3]
    assert s.parent.tolist() == [-1, 0, 1, -1]
    assert pos.origins_unknown == 1 and pos.origins_unknown_all == 1 and pos.lineage_complete is False
    assert s.h_present == 3 and s.dt_present.tolist() == [-1.0, 0.0, 1.0, 0.0] and pos.to_move == 1


def test_an_even_game_maps_minus_and_plus_zero_and_crosses_timelines():
    with Game.from_fen(["[1k2/4/4/R2K*:-0:1:w]", "[1k2/4/4/R2K*:+0:1:w]"]) as g:
        x = g.expand()
        pos = to_position(x)
    s = pos.state
    assert s.timeline.tolist() == [-1, 0]
    texts = x.move_texts()
    i = texts.index("(-0T1)Kd1(+0T1)c1")  # a king step along the timeline axis
    assert s.moves[i].tolist() == [3, 18 + 2, 0]
    assert x.moves["flags"][i] & 64  # travel: another board


def test_register_counts_shift_the_tokens():
    with Game.from_fen(["[1k2/4/4/R2K*:0:1:w]", "[1k2/4/4/1R1K*:0:1:b]"]) as g:
        pos = g.position(regs_per_board=4, global_regs=0)
    s = pos.state
    assert s.n_tokens == 2 * (16 + 4)
    assert s.moves[0].tolist() == [20 + 13, 20 + 12, 0]


# --- real positions ---------------------------------------------------------------------------------


def _real_games():
    """(label, Game) for every variant's start, random play from each, and every position of
    the records - each a separate handle."""
    out = []
    for name in VARIANTS:
        out.append((f"{name} start", Game.variant(name)))
        for seed in (0, 1):
            for i, game in enumerate(random_game(name, seed, target_boards=70, max_actions=40)):
                if i % 4 == 3:
                    out.append((f"{name} seed {seed} action {i + 1}", game.copy()))
    for key, text in RECORDS.items():
        game = Game.from_pgn(text)
        out.append((f"{key} position 0", game.copy()))
        for n in game.replay():
            out.append((f"{key} position {n}", game.copy()))
    return out


@pytest.fixture(scope="module")
def real_games():
    games = _real_games()
    yield games
    for _, g in games:
        g.close()


def test_real_positions_cover_branching_and_inactive_timelines(real_games):
    timelines = [g.info().timelines for _, g in real_games]
    inactive = [g.info().timelines - g.info().active_timelines for _, g in real_games]
    assert len(real_games) > 60 and max(timelines) >= 8 and max(inactive) >= 1


def test_every_real_position_converts_and_matches_the_engine(real_games):
    checked_boards = 0
    for label, game in real_games:
        pos = check_position(game, label, relations=ALL_RELATIONS)
        assert pos.origins_unknown_all == 0 or "Battleground" in label, label  # its outer initial timelines (engine reading)
        checked_boards += pos.state.n_boards
    assert checked_boards > 1000


def test_the_scope_agrees_with_the_engine_at_every_k(real_games):
    for label, game in real_games[::3]:
        pos = game.position()
        lay = pos.state.layout
        for k in K_VALUES:
            x = game.expand(window_k=k, square_features="none")
            for bands_only in (False,):
                sc = pos.scope.at(k, bands_only=bands_only)
                assert sc.window.tolist() == x.window.tolist(), (label, k)
                assert sc.e_boards.tolist() == x.e_boards.tolist(), (label, k)
                assert sc.src_extra.tolist() == x.src_extra.tolist(), (label, k)
        sc = pos.scope
        boards = sc.boards()
        tokens = sc.tokens(lay)
        assert tokens.size == boards.size * lay.tokens_per_board + lay.global_regs
        board_of, square = lay.split_square_tokens(tokens[: boards.size * lay.tokens_per_board])
        assert set(board_of.tolist()) == set(boards.tolist())
        squares_only = sc.tokens(lay, registers=False, global_registers=False)
        assert lay.is_square_token(squares_only).all() and squares_only.size == boards.size * lay.squares_per_board
        # every candidate move's source is in W + E + src_extra (the pointer's sources)
        src_boards = lay.split_square_tokens(pos.state.moves[:, 0])[0]
        assert set(src_boards.tolist()) <= set(boards.tolist()), label


def test_bands_only_drops_the_origins():
    with Game.from_pgn(RECORDS["branching"]) as g:
        g.replay_all()
        pos = g.position()
    sc = pos.scope
    origin = int(np.flatnonzero(sc.is_origin)[0])
    assert origin in sc.at(0).window.tolist()
    assert origin not in sc.at(0, bands_only=True).window.tolist()


# --- the present, and what cannot be converted -----------------------------------------------------


def test_a_shifted_unsubmitted_action_is_refused():
    """After 1. e4 the present is Black's but the action is White's until submitted - under both
    readings of Q4 (by default, R8, White may still move, but the one board has been played)."""
    for rules in (0, T5D_RULES_NO_MOVES_AFTER_SHIFT):
        with Game.variant("Standard", rules=rules) as g:
            e4 = next(m for m in g.moves() if m.text() == "(0T1)Pe2(0T1)e4")
            g.apply(e4)
            x = g.expand()
            with pytest.raises(PresentMismatch, match="submit the action first"):
                to_position(x)
            pos = to_position(x, strict=False)
            assert not pos.present_matches and pos.move_count == 0 and pos.may_move == (rules == 0)
            assert (x.acting_player, x.present & 1, x.submit_possible, x.submit_legal) == (0, 1, True, True)
            g.submit()
            assert g.position().present_matches


def test_mid_action_before_the_shift_converts():
    with Game.variant("Standard - Two Timelines") as g:
        e4 = next(m for m in g.moves() if m.text(even=True) == "(-0T1)Pe2(-0T1)e4")
        g.apply(e4)
        pos = g.position()
    s = pos.state
    assert pos.to_move == 0 and pos.present == 2 and s.h_present == 2
    assert s.is_head.tolist() == [False, True, True]  # -0 has its new board; +0 is still to play
    assert set(s.layout.split_square_tokens(s.moves[:, 0])[0].tolist()) == {2}  # only +0's board is playable


def test_features_must_cover_every_board():
    with Game.variant("Standard") as g, pytest.raises(ValueError, match="square_features='all'"):
        to_position_state(g.expand(square_features="scope"))


def test_relations_follow_the_network_flags():
    assert relations_for(NetConfig()) == ()
    assert relations_for(NetConfig(typed_edge_bias=True)) == ALL_RELATIONS
    assert relations_for(NetConfig(routing_table=True, relations_head_types=("attack", "defence"))) == ("attack", "defence")


def test_scope_tokens_on_a_small_layout():
    from t5d.engine.convert import EngineScope

    sc = EngineScope(
        k=0,
        bands_only=False,
        window=np.array([1]),
        e_boards=np.array([0]),
        src_extra=np.zeros(0, int),
        window_depth=np.array([T5D_NO_WINDOW, 0]),
        band_depth=np.array([T5D_NO_WINDOW, 0]),
        is_origin=np.array([False, False]),
        travel_destinations=np.array([0]),
        source_boards=np.array([1]),
    )
    lay = TokenLayout(2, 2, 2, regs_per_board=1, global_regs=2)
    assert sc.tokens(lay).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert sc.tokens(lay, registers=False, global_registers=False).tolist() == [0, 1, 2, 3, 5, 6, 7, 8]


def test_moves_after_the_shift_cannot_be_represented():
    """Under the default rules (Q4, ruled R8) a move may follow the shift: the engine's acting
    player is not the one the boards' present names, and the network's contract has no feature
    to say so until register D8's acting-player input (stage N5) - refused. Under Q4's
    Guide-literal reading (T5D_RULES_NO_MOVES_AFTER_SHIFT) the same random play never meets a
    move after the shift."""
    from t5d.engine.play import ROYAL_IDS

    def first_move_after_the_shift(rules: int):
        """Random play until a state past the shift has a move: the game, that state's expansion
        (None if 400 steps found none) and how many shifted states were met."""
        rng = np.random.default_rng(0)
        g = Game.variant("Standard", rules=rules)
        g.set_legality(False)
        shifted = 0
        for _ in range(400):
            x = g.expand(square_features="none")
            if g.info().obligations_met:
                shifted += 1
                assert x.submit_possible and x.acting_player == g.info().to_move != x.present & 1
                if x.move_count:  # a move after the shift
                    return g, x, shifted
                g.submit()
                continue
            assert not x.submit_possible
            g.apply(int(rng.choice(np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS)))))
        return g, None, shifted

    g, found, _ = first_move_after_the_shift(0)
    with g:
        if found is None:
            pytest.fail("random play found no move after the shift")
        x = g.expand()
        assert x.may_move and x.move_count > 0
        assert x.acting_player == x.to_move and x.acting_player != x.present & 1 and x.submit_possible
        with pytest.raises(PresentMismatch):
            to_position(x)
        assert not to_position(x, strict=False).present_matches
    g, found, shifted = first_move_after_the_shift(T5D_RULES_NO_MOVES_AFTER_SHIFT)
    with g:
        assert found is None and shifted > 20  # many shifts (48), never a move after one
        assert not g.info().may_move or not g.info().obligations_met


def test_promotions_are_one_candidate_each_and_one_edge():
    """A pawn on a3 of a 4x4 board promotes on a4; the record's Promotions header offers Q and R
    (S8): two candidate moves with the same tokens, told apart by the promotion id; one merged
    move-graph edge."""
    record = '[Board "custom"]\n[Size "4x4"]\n[Promotions "Q,R"]\n[4/P3/4/K2k:0:1:w]\n'
    with Game.from_pgn(record) as g:
        pos = g.position()
    s = pos.state
    a3, a4 = 8, 12
    promos = s.moves[s.moves[:, 0] == a3]
    assert promos.tolist() == [[a3, a4, PIECE_IDS["Q"]], [a3, a4, PIECE_IDS["R"]]]
    assert s.edges[s.edges[:, 0] == a3].tolist() == [[a3, a4, RELATION_IDS["move"]]]
    with Game.from_fen(["[4/P3/4/K2k:0:1:w]"]) as g:  # the standard set: the queen alone
        assert g.position().state.moves[-1].tolist() == [a3, a4, PIECE_IDS["Q"]]
