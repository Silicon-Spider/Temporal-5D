"""Stage N4: the ctypes binding to the engine's C API (``t5d/engine``).

Finding the library (these run without it), making games from variants, 5DFEN and
records, replaying, moving and submitting, the move list, errors carrying the engine's
messages, and the pointer-lifetime rule: an ``Expansion`` is a copy that outlives the
handle's next call.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from t5d.engine import (
    ENV_VAR,
    EngineArgumentError,
    EngineError,
    EngineNotFoundError,
    Expansion,
    Game,
    IllegalError,
    NotationError,
    UnsupportedError,
    find_library,
    load_library,
)
from t5d.engine.capi import MOVE_DTYPE, MoveRecord
from t5d.engine.errors import error_for
from t5d.engine.expansion import Move, timeline_label
from t5d.engine.game import relation_bits
from tests.engine.helpers import RECORDS, VARIANTS, requires_engine

# --- finding the library (no engine needed) ---------------------------------------------------


def test_an_explicit_path_that_does_not_exist_is_a_clear_error(tmp_path):
    with pytest.raises(EngineNotFoundError, match="no engine library at"):
        find_library(tmp_path / "missing.dll")


def test_an_environment_variable_that_names_nothing_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "nowhere"))
    with pytest.raises(EngineNotFoundError, match=ENV_VAR):
        find_library()


def test_an_empty_directory_holds_no_library(tmp_path):
    with pytest.raises(EngineNotFoundError, match="expected a file, or a directory"):
        find_library(tmp_path)


def test_a_file_that_is_not_a_library_fails_to_load(tmp_path):
    bogus = tmp_path / "t5d.dll"
    bogus.write_bytes(b"not a library")
    with pytest.raises(EngineNotFoundError, match="cannot load"):
        load_library(bogus)


def test_statuses_map_to_exceptions_that_are_also_builtins():
    assert isinstance(error_for(6, "no"), IllegalError) and isinstance(error_for(6, "no"), ValueError)
    assert isinstance(error_for(5), UnsupportedError) and isinstance(error_for(5), NotImplementedError)
    assert isinstance(error_for(2), NotationError)
    assert isinstance(error_for(1), EngineArgumentError)
    assert isinstance(error_for(3), MemoryError)
    unknown = error_for(99, "", "f")
    assert isinstance(unknown, EngineError) and unknown.status == 99


def test_move_layout_is_the_c_record():
    assert MOVE_DTYPE.itemsize == 32
    rec = MoveRecord(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 64)
    as_np = np.frombuffer(bytes(rec), dtype=MOVE_DTYPE)[0]
    assert Move.from_record(rec) == Move.from_record(as_np)
    assert Move.from_record(rec).to_c().flags == 64


@pytest.mark.parametrize(
    ("p", "even", "label"), [(0, False, "0"), (2, False, "+2"), (-3, False, "-3"), (-1, True, "-0"), (0, True, "+0"), (-2, True, "-1")]
)
def test_timeline_labels_invert_the_index_mapping(p, even, label):
    from t5d.net.state import timeline_index

    assert timeline_label(p, even) == label
    assert timeline_index(label, even_start=even) == p


def test_relation_names_and_ids():
    assert relation_bits(["attack", "defence", "xray"]) == (1 << 1) | (1 << 2) | (1 << 4)
    assert relation_bits([3]) == 1 << 3
    with pytest.raises(ValueError, match="unknown relation"):
        relation_bits(["threat"])


# --- games ---------------------------------------------------------------------------------------


@requires_engine
def test_standard_start():
    with Game.variant("Standard") as g:
        info = g.info()
        assert (info.boards, info.width, info.height, info.even, info.to_move, info.present) == (1, 8, 8, False, 0, 2)
        assert info.may_move and not info.may_submit and not info.obligations_met
        snap = g.snapshot()
        assert snap.fingerprint == "d574889fd9da3f2bc65249ff27249b00"  # the notation's published example (H30)
        assert snap.boards == ((0, 2, "[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"),)
        moves = g.moves()
        assert len(moves) == 20
        assert {m.text() for m in moves} >= {"(0T1)Pe2(0T1)e4", "(0T1)Ng1(0T1)f3"}
    assert g.closed
    with pytest.raises(EngineArgumentError, match="closed"):
        g.info()


@requires_engine
def test_two_timelines_start_maps_minus_and_plus_zero():
    with Game.variant("Standard - Two Timelines") as g:
        assert g.snapshot().fingerprint == "3672761404ffcd15ae644c75401812be"  # the notation's second example
        x = g.expand()
        assert x.even and x.timeline.tolist() == [-1, 0] and x.half_turn.tolist() == [2, 2]


@requires_engine
@pytest.mark.parametrize("name", VARIANTS)
def test_every_variant_starts_with_white_to_move_and_moves(name):
    with Game.variant(name) as g:
        info = g.info()
        assert info.to_move == 0 and info.may_move
        assert len(g.moves()) > 0


@requires_engine
def test_errors_carry_the_engines_messages():
    with pytest.raises(UnsupportedError, match="not one the engine sets up") as exc:
        Game.variant("No Such Variant")
    assert exc.value.function == "t5d_game_new_variant" and exc.value.status == 5
    with pytest.raises(NotationError):
        Game.from_fen(["[this is not 5dfen]"])
    with Game.from_pgn("1. e4 / (0T1)Zz9\n") as g:  # the set-up never reads the move text
        assert g.record_actions().truncated
        g.replay_next()
        with pytest.raises(NotationError, match="cannot read the move"):
            g.replay_next()
    with Game.from_pgn("1. e4 / e5\n2. Ke3 / e6\n") as g:
        g.replay_next()
        g.replay_next()
        with pytest.raises(IllegalError, match="no pseudo-legal move fits"):
            g.replay_next()
        assert g.record_actions().played == 2  # a failed action leaves the game where it was
    with Game.variant("Standard") as g:
        with pytest.raises(UnsupportedError, match="pin and check are not built"):
            g.expand(relations=["pin"])
        with pytest.raises(IllegalError):
            g.submit()  # the present has not shifted (T5)
        with pytest.raises(EngineArgumentError):
            g.replay_next()  # not a record
        with pytest.raises(EngineArgumentError):
            g.apply(10_000)  # no such index in the last expansion
    with pytest.raises(ValueError, match="square_features"):
        Game.variant("Standard").expand(square_features="most")


@requires_engine
def test_apply_by_record_and_by_index_and_submit():
    with Game.variant("Standard") as g:
        x = g.expand(square_features="none")
        e4 = next(m for m in x.move_list() if m.text() == "(0T1)Pe2(0T1)e4")
        g.apply(e4)
        assert g.info().obligations_met and not g.info().may_move  # Q4: no move after the shift
        assert g.moves() == []
        g.submit()
        assert g.info().to_move == 1
        x = g.expand(square_features="none")
        i = x.move_texts().index("(0T1)Pe7(0T1)e5")
        g.apply(i)
        g.submit()
        assert g.info().present == 4
        assert g.snapshot().boards[-1][2] == "[r*nbqk*bnr*/p*p*p*p*1p*p*p*/8/4p3/4P3/8/P*P*P*P*1P*P*P*/R*NBQK*BNR*:0:2:w]"


@requires_engine
def test_a_move_the_rules_refuse_is_illegal():
    with Game.variant("Standard") as g:
        e4 = next(m for m in g.moves() if m.text() == "(0T1)Pe2(0T1)e4")
        bogus = Move(**{**e4.__dict__, "to_square": 36, "to_cell": 36})  # e2-e5
        with pytest.raises(IllegalError):
            g.apply(bogus)


@requires_engine
def test_copy_is_independent():
    with Game.variant("Standard") as g:
        h = g.copy()
        g.apply(g.moves()[0])
        g.submit()
        assert h.info().boards == 1 and g.info().boards == 2
        h.close()


@requires_engine
def test_an_expansion_outlives_the_handles_next_call():
    """t5d/capi.h: the arrays belong to the handle until it next changes; the binding copies."""
    with Game.from_pgn(RECORDS["branching"]) as g:
        g.replay_next()
        g.replay_next()
        x = g.expand(relations=["attack", "defence", "xray"])
        frozen = {name: getattr(x, name).copy() for name in ("timeline", "half_turn", "piece", "moves", "edges", "relations", "window")}
        g.replay_all()  # changes the handle many times over
        g.expand()
        g.close()
        gc.collect()
    for name, before in frozen.items():
        assert np.array_equal(getattr(x, name), before), name
    assert isinstance(x, Expansion) and x.board_count == 3


@requires_engine
def test_records_replay_action_by_action():
    with Game.from_pgn(RECORDS["branching"]) as g:
        progress = g.record_actions()
        assert (progress.total, progress.played, progress.truncated) == (6, 0, False)
        counts = [g.info().boards for _ in g.replay()]
        assert counts == [2, 3, 5, 6, 7, 9]
        assert g.record_actions().remaining == 0
        x = g.expand()
        assert x.timeline.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
        # the branch's first board (+1 at T1 black) descends from the T1 white board it targets
        first = x.board_index(1, 3)
        assert x.parent[first] == x.board_index(0, 2)
        with pytest.raises(EngineArgumentError, match="every action"):
            g.replay_next()


@requires_engine
def test_legality_switch():
    with Game.variant("Standard") as g:
        g.set_legality(False)
        g.set_legality(True)


@requires_engine
def test_from_fen_parity_and_player_to_move():
    boards = ["[1k2/4/4/R2K*:0:1:w]"]
    with Game.from_fen(boards, width=4, height=4, parity="odd", to_move="white") as g:
        assert g.info().to_move == 0
    with Game.from_fen(" ".join(boards)) as g:  # one string, white space between boards
        assert g.info().width == 4
    with pytest.raises(ValueError, match="parity"):
        Game.from_fen(boards, parity="sideways")
