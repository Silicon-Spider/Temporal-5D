"""Stage N5: games end to end through the C step API with the network (random weights) as the
evaluator (``t5d/engine/netplay.py``).

Short games - self-play and match play, both arms of D2, Standard and Two Timelines, under a
turn cap and a board cap - each written as a 5dpgn record and replayed by the engine's own
tools: its reader (in process) and its gate adapter in full mode (``t5d-gate``, a
subprocess): every action legal, one position per action and the start, the last position's
verdict the game's end, the final boards the game's. Self-play is deterministic in its seed.
The record's move form and the turn-start verdicts are checked on their own.
"""

from __future__ import annotations

import os

import pytest
import torch

from t5d.engine import Game, NetworkEvaluator
from t5d.engine.capi import T5D_MOVE_BRANCHING, T5D_MOVE_CAPTURE, T5D_MOVE_TRAVEL
from t5d.engine.expansion import Move
from t5d.engine.netplay import (
    PlayConfig,
    check_replay,
    gate_executable,
    network_config,
    pgn_move,
    play_game,
    replay_with_gate,
    turn_start_verdict,
)
from t5d.net import build_model
from t5d.net.state import PIECE_IDS
from tests.engine.helpers import ENGINE, REQUIRE_VAR, requires_engine

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
