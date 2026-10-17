"""Stage N4: the ctypes binding to the engine's C API (``t5d/engine``).

Finding the library (these run without it), making games from variants, 5DFEN and
records, replaying, moving and submitting, the move list, errors carrying the engine's
messages, and the pointer-lifetime rule: an ``Expansion`` is a copy that outlives the
handle's next call.
"""

from __future__ import annotations

import ctypes
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
from t5d.engine.capi import (
    MIRRORED_STRUCTS,
    MOVE_DTYPE,
    T5D_RULES_MOVES_AFTER_SHIFT,
    T5D_RULES_NO_MOVES_AFTER_SHIFT,
    CExpansion,
    GameSummary,
    MoveRecord,
    layout_mismatches,
)
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


def test_a_refused_library_fails_the_engine_tests_and_a_missing_one_skips_them(tmp_path, monkeypatch):
    """A library that is found but refused - not a library, or a path or T5D_ENGINE_LIB that
    names nothing - fails every test that needs the engine; one that is not built skips them,
    unless T5D_REQUIRE_ENGINE=1 (CI) makes that a failure too."""
    from t5d.engine import capi, library_status
    from tests.engine.helpers import engine_verdict

    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / capi._library_names()[0]).write_bytes(b"not a library")
    assert library_status(bogus, fresh=True)[0] == "refused"
    assert library_status(tmp_path / "nowhere", fresh=True)[0] == "refused"
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "nowhere"))
    assert library_status(fresh=True)[0] == "refused"
    monkeypatch.delenv(ENV_VAR)
    monkeypatch.setattr(capi, "default_build_dir", lambda: tmp_path / "empty")
    status, why = library_status(fresh=True)
    assert status == "missing" and "not built" in why
    assert engine_verdict("available", "", True) == ("run", "")
    assert engine_verdict("refused", "layouts differ", False)[0] == "fail"
    assert engine_verdict("missing", "not built", True)[0] == "fail"
    assert engine_verdict("missing", "not built", False)[0] == "skip"


@requires_engine
def test_a_binding_laid_out_otherwise_is_a_refused_library(monkeypatch):
    """The review's M3b: t5d_leaf's submit_possible and submit_legal swapped in the mirror - the
    library is refused, which fails the engine tests (it used to skip them)."""
    from t5d.engine import capi, library_status

    fields = list(capi.CLeaf._fields_)
    i, j = (next(k for k, f in enumerate(fields) if f[0] == name) for name in ("submit_possible", "submit_legal"))
    fields[i], fields[j] = fields[j], fields[i]

    class SwappedLeaf(ctypes.Structure):
        _fields_ = fields

    monkeypatch.setitem(capi.MIRRORED_SEARCH_STRUCTS, "t5d_leaf", SwappedLeaf)
    status, why = library_status(fresh=True)
    assert status == "refused" and "t5d_leaf.submit_possible: offset" in why


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
        # Q4, ruled R8 (the default): moves may follow the shift until submission - but the one
        # board has been played (T3), so there is none, and submitting is possible and legal (D8)
        info = g.info()
        assert info.obligations_met and info.may_move and info.may_submit
        assert g.moves() == []
        after = g.expand(square_features="none")
        assert (after.acting_player, after.present & 1, after.submit_possible, after.submit_legal) == (0, 1, True, True)
        g.submit()
        assert g.info().to_move == 1
        x = g.expand(square_features="none")
        i = x.move_texts().index("(0T1)Pe7(0T1)e5")
        g.apply(i)
        g.submit()
        assert g.info().present == 4
        assert g.snapshot().boards[-1][2] == "[r*nbqk*bnr*/p*p*p*p*1p*p*p*/8/4p3/4P3/8/P*P*P*P*1P*P*P*/R*NBQK*BNR*:0:2:w]"


@requires_engine
def test_q4_guide_literal_reading_is_bit_3_and_refuses_moves_after_the_shift():
    """Engine stage E7: bit 3 of the ruleset encoding marks Q4's Guide-literal reading (no move
    once the present has shifted); the old name for moves after the shift is now 0, the default."""
    assert (T5D_RULES_NO_MOVES_AFTER_SHIFT, T5D_RULES_MOVES_AFTER_SHIFT) == (1 << 3, 0)
    with Game.variant("Standard", rules=T5D_RULES_MOVES_AFTER_SHIFT) as g:
        assert (g.info().rules, g.info().ruleset) == (0, 0)
    late = '[Board "custom"]\n[5/5/5/5/R4:0:1:w]\n[5/5/5/5/R4:+1:2:w]\n'
    for rules, moves_after in ((0, 8), (T5D_RULES_NO_MOVES_AFTER_SHIFT, 0)):
        with Game.from_pgn(late, rules) as g:
            assert g.info().rules == rules
            g.apply(next(m for m in g.moves() if m.text() == "(0T1)Ra1(0T1)a2"))
            info = g.info()
            assert info.obligations_met and info.may_submit
            assert info.may_move == (rules == 0)
            x = g.expand(square_features="none")
            assert x.move_count == moves_after  # the rook on +1: a2-a5, b1-e1, under R8 only
            assert (x.acting_player, x.present & 1, x.submit_possible, x.submit_legal) == (0, 1, True, True)
            g.submit()
            assert g.info().to_move == 1


@requires_engine
def test_the_binding_mirrors_the_librarys_struct_layouts():
    """Every mirrored struct's size and field offsets equal the library's own (t5d_layout); a
    struct that differs - a binding older than its library - is named."""
    lib = load_library()
    assert layout_mismatches(lib) == []
    # D8's three int32 come after every older field (engine stage E7), A6's three after them
    # (engine ABI round 1): appended, so no older offset moved
    tail = (CExpansion.acting_player.offset, CExpansion.submit_possible.offset, CExpansion.submit_legal.offset)
    assert tail[0] >= CExpansion.src_extra.offset + ctypes.sizeof(ctypes.c_void_p)
    assert tail == (tail[0], tail[0] + 4, tail[0] + 8)
    a6 = (CExpansion.mandatory_remaining.offset, CExpansion.optional_playable.offset, CExpansion.moves_made.offset)
    assert a6 == (tail[2] + 4, tail[2] + 8, tail[2] + 12)
    assert GameSummary.moves_made.offset >= GameSummary.rules.offset + 8
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert tail == (272, 276, 280) and a6 == (284, 288, 292) and ctypes.sizeof(CExpansion) == 296
        assert GameSummary.moves_made.offset == 64 and ctypes.sizeof(GameSummary) == 72

    class OldExpansion(ctypes.Structure):  # before engine ABI round 1: no turn state
        _fields_ = CExpansion._fields_[:-3]

    class OlderExpansion(ctypes.Structure):  # before stage E7: no acting player, no submit action either
        _fields_ = CExpansion._fields_[:-6]

    for old in (OldExpansion, OlderExpansion):
        try:
            MIRRORED_STRUCTS["t5d_expansion"] = old
            found = layout_mismatches(lib)
        finally:
            MIRRORED_STRUCTS["t5d_expansion"] = CExpansion
        assert any(m.startswith("t5d_expansion: size") for m in found)


@requires_engine
@pytest.mark.parametrize("rules", [0, T5D_RULES_NO_MOVES_AFTER_SHIFT], ids=["R8", "guide-literal"])
def test_the_expansion_carries_the_turn_state_a6(rules):
    """Register A6: the turn state the engine exports - the mandatory boards still to move, the
    optional boards still playable, the moves made in the action (the handle's count: no state
    records it) - through apply, copy, submit and a record's replay. White's 0 at T1 is owed;
    its +1 at T2, ahead of the present, is optional."""
    with Game.from_fen(["[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:2:w]"], rules=rules) as g:
        x = g.expand(square_features="none")
        assert (x.mandatory_remaining, x.optional_playable, x.moves_made) == (1, 1, 0)
        assert g.info().moves_made == 0 and g.info().turn_start
        on_zero = next(m for m in x.move_list() if m.text() == "(0T1)Ka1(0T1)a2")
        g.apply(on_zero)
        x = g.expand(square_features="none")
        r8 = rules == 0
        assert (x.mandatory_remaining, x.optional_playable, x.moves_made) == (0, 1 if r8 else 0, 1)
        assert x.submit_possible and x.mandatory_remaining == 0
        assert g.info().moves_made == 1 and not g.info().turn_start
        if r8:  # moves after the shift (R8): the optional board, the count going on - in a copy
            h = g.copy()
            assert h.info().moves_made == 1
            h.apply(next(m for m in h.moves() if m.text() == "(+1T2)Ka1(+1T2)b1"))
            y = h.expand(square_features="none")
            assert (y.mandatory_remaining, y.optional_playable, y.moves_made) == (0, 0, 2)
            assert g.info().moves_made == 1  # the original keeps its own count
            h.close()
        g.submit()
        assert g.info().moves_made == 0
        x = g.expand(square_features="none")
        assert x.moves_made == 0 and x.mandatory_remaining > 0  # Black's turn start owes a board (T5)
    # A record's replay: whole actions, so 0 after each.
    with Game.from_pgn("1. e4 / e5 2. Nf3\n") as g:
        for _ in g.replay():
            assert g.info().moves_made == 0 and g.expand(square_features="none").moves_made == 0
        g.apply(g.moves()[0])
        assert g.info().moves_made == 1


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
