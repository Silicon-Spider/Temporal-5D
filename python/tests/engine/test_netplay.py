"""Stage N5: games end to end through the C step API with the network (random weights) as the
evaluator (``t5d/engine/netplay.py``).

Short games - self-play and match play, both arms of D2, Standard and Two Timelines, under a
turn cap and a board cap - each written as a 5dpgn record and replayed by the engine's own
tools: its reader (in process) and its gate adapter in full mode (``t5d-gate``, a
subprocess): every action legal, one position per action and the start, the last position's
verdict the game's end, the final boards the game's, no warning from the reader. Self-play
is deterministic in its seed. The record's move form and the turn-start verdicts are checked
on their own, and the capture mark (H51: written strictly) against the boards.
"""

from __future__ import annotations

import os
import random

import pytest
import torch

from t5d.engine import Game, NetworkEvaluator
from t5d.engine.capi import T5D_MOVE_BRANCHING, T5D_MOVE_CAPTURE, T5D_MOVE_TRAVEL
from t5d.engine.expansion import Move
from t5d.engine.netplay import (
    GameResult,
    PlayConfig,
    check_replay,
    gate_executable,
    network_config,
    pgn_move,
    play_game,
    replay_with_gate,
    turn_start_verdict,
    write_record,
)
from t5d.engine.play import ROYAL_IDS
from t5d.net import build_model
from t5d.net.state import COLOUR_BLACK, COLOUR_WHITE, PIECE_IDS
from tests.engine.helpers import ENGINE, REQUIRE_VAR, decode_board, requires_engine

pytestmark = requires_engine

GATE = gate_executable() if ENGINE else None


@pytest.fixture
def gate_required():
    """The gate adapter beside the library: skip without it - or fail, under T5D_REQUIRE_ENGINE=1
    (CI builds t5d-gate with the library)."""
    if GATE is None or not GATE.is_file():
        why = f"t5d-gate is not built beside the engine library (looked for {GATE})"
        if os.environ.get(REQUIRE_VAR, "").strip() == "1":
            pytest.fail(f"{REQUIRE_VAR}=1: {why}", pytrace=False)
        pytest.skip(why)


requires_gate = pytest.mark.usefixtures("gate_required")


@pytest.fixture(scope="module")
def nets():
    cfg = network_config("small")
    return cfg, [NetworkEvaluator(build_model(cfg, seed=s).eval(), max_batch=8) for s in (1, 2)]


@requires_gate
@pytest.mark.parametrize(
    "variant,split,mode",
    [
        ("Standard", "ply", "self"),
        ("Standard - Two Timelines", "ply", "match"),
        ("Standard", "turn", "match"),
        ("Standard - Two Timelines", "turn", "self"),
        ("Standard - Two Timelines", "tree", "self"),
        ("Misc - Timeline Tactician", "tree", "match"),
    ],
)
def test_a_game_replays_legal(nets, variant, split, mode):
    cfg, evaluators = nets
    play = PlayConfig(variant=variant, split=split, mode=mode, playouts=8, batch_size=4, cap_turns=3, cap_boards=30, seed=3)
    with torch.no_grad():
        res = play_game(evaluators if mode == "match" else evaluators[0], play, net_cfg=cfg)
    assert res.end in ("checkmate", "stalemate", "turn_cap")
    assert res.actions and all(res.actions)  # every action holds a move
    assert res.decisions >= len(res.actions) and res.evaluations > 0 and res.playouts >= 8 * res.decisions - 8
    if split == "turn":
        assert res.decisions == len(res.actions)  # whole turns: one decision per action
    else:
        assert res.decisions == sum(len(a) for a in res.actions) + len(res.actions)  # moves, then a submit
    replay = replay_with_gate(res.record, gate=GATE)
    assert check_replay(res, replay) == [], replay.lines[-3:]
    assert '[Termination "turn_cap (self-play' in res.record or res.end != "turn_cap"
    assert res.reroot_fallbacks == 0  # match play re-rooted every carried search on the reply


MATED_WHITE = "[8/8/8/8/8/5k2/6q1/7K:0:1:w]"  # Kh1, Qg2 protected by Kf3
MATED_BLACK = "[8/8/8/8/8/5K2/6Q1/7k:0:1:b]"
STALEMATED_WHITE = "[k7/8/8/8/8/6q1/8/7K:0:1:w]"  # Kh1 boxed in by Qg3, not in check


@requires_gate
@pytest.mark.parametrize(
    "fen,end,winner,result",
    [(MATED_WHITE, "checkmate", 1, "0-1"), (MATED_BLACK, "checkmate", 0, "1-0"), (STALEMATED_WHITE, "stalemate", None, "1/2-1/2")],
)
def test_the_result_is_the_final_positions(nets, fen, end, winner, result):
    """A game from a finished custom start: the game's winner and result, the record's Result
    header and closing token, are the ones the final position calls for - judged by the gate
    (the side to move, its verdict) and the engine's reader. A wrong winner (the review's M9: the
    mated side recorded as the winner), a wrong end or a wrong header is reported."""
    from dataclasses import replace

    cfg, evaluators = nets
    with torch.no_grad():
        res = play_game(evaluators[0], PlayConfig(start_fen=(fen,), playouts=4, cap_turns=0, cap_boards=0), net_cfg=cfg)
    assert (res.end, res.winner, res.result, res.actions) == (end, winner, result, [])
    assert '[Board "custom"]' in res.record and fen in res.record and f'[Result "{result}"]' in res.record
    replay = replay_with_gate(res.record, gate=GATE)
    assert check_replay(res, replay) == []
    wrong = []
    if end == "checkmate":
        wrong.append(replace(res, winner=1 - winner))
    wrong.append(replace(res, end="turn_cap", winner=None))
    wrong.append(replace(res, record=res.record.replace(f'[Result "{result}"]', '[Result "*"]')))
    for w in wrong:
        assert check_replay(w, replay) != [], (w.end, w.winner)


@requires_gate
def test_a_game_that_ends_in_checkmate_replays_with_its_result(nets):
    """Focused - Just Kings under whole turns: with these seeds the game ends in checkmate on the
    development machine (the review's case); whatever it ends in, its result is the final
    position's."""
    cfg, evaluators = nets
    play = PlayConfig(variant="Focused - Just Kings", split="turn", playouts=8, batch_size=4, cap_turns=4, cap_boards=40, seed=1)
    with torch.no_grad():
        res = play_game(evaluators[0], play, net_cfg=cfg)
    replay = replay_with_gate(res.record, gate=GATE)
    assert check_replay(res, replay) == [], (res.end, res.result, replay.last_side, replay.last_terminal)


def test_a_reroot_that_fails_is_counted_not_hidden():
    """Match play: a carried search that cannot re-root on the reply (here, a reply that is not
    a legal action of the position) is dropped and the fallback reported."""
    from t5d.engine.netplay import _reroot
    from t5d.engine.search import Search

    with Game.variant("Standard") as g:
        s = Search(g)
        e4 = next(m for m in g.moves() if m.text() == "(0T1)Pe2(0T1)e4")
        carried = [None, s]
        assert _reroot(carried, 1, [e4, e4], g) is False and carried[1] is None
        s2 = Search(g)
        g.apply(e4)
        g.submit()
        carried = [None, s2]
        assert _reroot(carried, 1, [e4], g) is True and carried[1] is s2
        s2.close()


def test_self_play_is_deterministic_in_its_seed(nets):
    cfg, evaluators = nets
    play = PlayConfig(variant="Standard - Two Timelines", playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=5)
    with torch.no_grad():
        a = play_game(evaluators[0], play, net_cfg=cfg)
        b = play_game(evaluators[0], play, net_cfg=cfg)
        c = play_game(evaluators[0], PlayConfig(**{**play.__dict__, "seed": 6}), net_cfg=cfg)
    assert a.record == b.record and a.final_fingerprint == b.final_fingerprint
    assert a.record != c.record


def test_the_record_is_read_back_by_the_engines_reader(nets):
    cfg, evaluators = nets
    with torch.no_grad():
        res = play_game(evaluators, PlayConfig(mode="match", playouts=6, batch_size=3, cap_turns=3, seed=7), net_cfg=cfg)
    with Game.from_pgn(res.record) as g:
        assert g.replay_all() == len(res.actions)
        assert g.snapshot().fingerprint == res.final_fingerprint
    assert res.reroot_fallbacks == 0


def _move(**kw) -> Move:
    base = dict(
        from_timeline=0, from_half_turn=2, to_timeline=0, to_half_turn=2, from_board=0, to_board=0, from_square=12, to_square=28,
        from_cell=12, to_cell=28, piece=PIECE_IDS["P"], captured=0, promotion=0, flags=0,
    )  # fmt: skip
    return Move(**{**base, **kw})


def test_the_record_writes_the_readers_long_form():
    assert pgn_move(_move(), even=False) == "(0T1)Pe2e4"
    assert pgn_move(_move(piece=PIECE_IDS["N"], from_square=6, to_square=21, flags=T5D_MOVE_CAPTURE), even=False) == "(0T1)Ng1xf3"
    jump = _move(
        piece=PIECE_IDS["N"], from_half_turn=4, to_half_turn=2, from_square=21, to_square=37, flags=T5D_MOVE_TRAVEL | T5D_MOVE_BRANCHING
    )
    assert pgn_move(jump, even=False) == "(0T2)Nf3>>(0T1)f5"
    across = _move(
        from_timeline=-1, to_timeline=0, piece=PIECE_IDS["N"], from_square=6, to_square=22, flags=T5D_MOVE_TRAVEL | T5D_MOVE_CAPTURE
    )
    assert pgn_move(across, even=True) == "(-0T1)Ng1>x(+0T1)g3"
    promo = _move(from_square=52, to_square=60, promotion=PIECE_IDS["Q"])
    assert pgn_move(promo, even=False) == "(0T1)Pe7e8=Q"


def _marks_judged_by_the_boards(g: Game, tally: dict) -> list[Move]:
    """Every move offered now, written by ``pgn_move`` and judged from the boards - never from
    its flags: the opponent's pieces on its target board against those on the board it makes
    there (the source board's successor for a move on its board; for a jump, the other board it
    makes). Its kind is read from the boards too: a capture onto an empty square is en passant,
    a jump that makes a timeline a branch. Returns the moves."""
    snap = g.snapshot()
    before = {(p, h): fen for p, h, fen in snap.boards}
    timelines = {p for p, _ in before}
    opponent = COLOUR_BLACK if snap.to_move == 0 else COLOUR_WHITE

    def theirs(fen: str) -> tuple[int, dict]:
        board = decode_board(fen, snap.width, snap.height)
        return int(((board["colour"] == opponent) & (board["piece"] != 0)).sum()), board

    moves = g.moves()
    for m in moves:
        with g.copy() as after:
            after.apply(m)
            made = {(p, h): fen for p, h, fen in after.snapshot().boards if (p, h) not in before}
        successor = (m.from_timeline, m.from_half_turn + 1)
        others = [k for k in made if k != successor]
        assert successor in made and len(others) <= 1, (m.text(snap.even), sorted(made))
        count_before, target = theirs(before[(m.to_timeline, m.to_half_turn)])
        count_after, _ = theirs(made[others[0]] if others else made[successor])
        takes = count_after < count_before
        if not others:
            kind = "en passant" if takes and target["piece"][m.to_cell] == 0 else "on its board"
        else:
            kind = "jump" if others[0][0] in timelines else "branch"
        written = pgn_move(m, snap.even)
        assert ("x" in written) == takes, (written, kind, "takes a piece" if takes else "takes none")
        tally[kind, takes] = tally.get((kind, takes), 0) + 1
    return moves


#: Two en passant captures on offer: White's exd6 and Black's exd3.
EN_PASSANT = (
    '[Mode "5D"]\n[Board "Standard"]\n\n1. e4 / a6\n2. e5 / d5\n',
    '[Mode "5D"]\n[Board "Standard"]\n\n1. a3 / e5\n2. a4 / e4\n3. d4\n',
)


def test_the_writer_marks_exactly_the_captures():
    """H51 (rulebook R11): the engine writes the capture mark strictly - ``x`` on every move that
    takes a piece and on no other. This writer (``pgn_move``, which N5's self-play and match
    records use) beside the C++ one (``synthetic::pgn_move``, the engine's test
    ``synthetic_moves_mark_exactly_the_captures_h51``): every move offered at every position of
    random pseudo-legal games under five variants - their actions' optional moves included - and
    at two en passant positions is written and judged from the boards; each kind of capture
    (on its board, en passant, by a jump, by a branch) and of quiet move must occur."""
    tally: dict = {}
    for variant in ("Standard", "Standard - Two Timelines", "Standard - Turn Zero", "Very Small - Open", "Focused - Just Brawns"):
        for seed in (1, 2):
            rng = random.Random(f"{variant}/{seed}")
            with Game.variant(variant) as g:
                g.set_legality(False)  # as t5d.engine.play: royals may be left capturable
                for _ in range(10):  # actions
                    for _ in range(4):  # moves in an action, at most: after the shift, optional ones
                        moves = _marks_judged_by_the_boards(g, tally)
                        if g.info().obligations_met and rng.random() < 0.6:
                            break
                        pool = [m for m in moves if m.captured not in ROYAL_IDS]  # never a royal taken
                        if not pool:
                            break
                        g.apply(rng.choice(pool))
                    if not g.info().obligations_met:
                        break  # stuck
                    g.submit()
    for record in EN_PASSANT:
        with Game.from_pgn(record) as g:
            g.replay_all()
            _marks_judged_by_the_boards(g, tally)
    kinds = [("on its board", True), ("on its board", False), ("en passant", True)]
    kinds += [("jump", True), ("jump", False), ("branch", True), ("branch", False)]
    assert all(tally.get(k, 0) > 0 for k in kinds), tally


def _played(variant: str, texts: list[list[str]]) -> tuple[list[list[Move]], bool]:
    """The moves named (in ``Move.text`` form) played as actions from ``variant``'s start."""
    actions = []
    with Game.variant(variant) as g:
        even = g.snapshot().even
        for action in texts:
            moves = []
            for text in action:
                m = next(m for m in g.moves() if m.text(even) == text)
                g.apply(m)
                moves.append(m)
            g.submit()
            actions.append(moves)
    return actions, even


@requires_gate
def test_a_capture_mark_that_disagrees_is_a_problem():
    """Since H51 the gate's reader reads a capture mark that disagrees with its move - ``x`` on a
    move that takes nothing, a capture written without ``x`` - as the squares define it, with a
    warning on standard error, where before it the record was refused (superfluous ``x``). The
    replay keeps the warnings and ``check_replay`` reports each: a record the engine writes must
    replay with none."""
    from dataclasses import replace

    texts = [["(0T1)Pe2(0T1)e4"], ["(0T1)Pd7(0T1)d5"], ["(0T2)Pe4(0T2)d5"], ["(0T2)Ng8(0T2)f6"]]
    actions, even = _played("Standard", texts)
    record = write_record("Standard", actions, even=even, first_turn=1, result="*", termination="unknown")
    assert "(0T2)Pe4xd5" in record and "(0T1)Pe2e4" in record
    res = GameResult(config=PlayConfig(), record=record, actions=actions, end="unknown", winner=None)
    replay = replay_with_gate(record, gate=GATE)
    assert replay.warnings == () and check_replay(res, replay) == []
    slips = [
        (record.replace("(0T1)Pe2e4", "(0T1)Pe2xe4"), "takes no piece"),  # superfluous: refused before H51
        (record.replace("Pe4xd5", "Pe4d5"), "written without 'x'"),  # missing: read with no warning before H51
    ]
    for wrong, slip in slips:
        slipped = replay_with_gate(wrong, gate=GATE)
        assert slipped.status == "ok" and slipped.actions_accepted == len(actions), slipped.lines[-1:]
        assert slipped.final_boards == replay.final_boards
        assert len(slipped.warnings) == 1 and slip in slipped.warnings[0], slipped.warnings
        problems = check_replay(replace(res, record=wrong), slipped)
        assert problems == [f"the gate's reader warned: {slipped.warnings[0]}"], problems


@pytest.mark.parametrize(
    "fen,verdict",
    [
        ("[8/8/8/8/8/5k2/6q1/7K:0:1:w]", "checkmate"),  # Kh1, Qg2 protected by Kf3
        ("[k7/8/8/8/8/6q1/8/7K:0:1:w]", "stalemate"),  # Kh1 boxed in by Qg3, not in check
        ("[k7/8/8/8/8/8/8/7K:0:1:w]", None),
    ],
)
def test_turn_start_verdicts(fen, verdict):
    with Game.from_fen([fen]) as g:
        assert turn_start_verdict(g, PlayConfig(cap_turns=0, cap_boards=0)) == verdict
    with Game.from_fen([fen]) as g:  # the cap comes after the rules' verdict
        assert turn_start_verdict(g, PlayConfig(cap_turns=0, cap_boards=1)) == (verdict or "turn_cap")
