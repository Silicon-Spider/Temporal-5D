"""Register H22 in the training generator (section 11.0; review decision 13, 2026-11-15): each turn
start's verdict and each chosen move's verification run within ``GeneratorConfig.terminal_nodes``,
and a question the budget leaves unsettled ends the game unscored - "unknown", never a mate. The
record keeps format version 2: the budget enters it through ``generator_config``; the end is the
"unknown" the target builder already masks (register I15; ``tests/train/test_unscored.py`` holds the
targets); an unsettled choice drops the action under way - its decisions, their positions and their
resignation events - so the record ends at that action's turn start.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import t5d.train.selfplay as selfplay
from t5d.engine import netplay
from t5d.engine.capi import T5D_LEGALITY_UNKNOWN, T5D_NO_LIMIT
from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL, TERMINAL_NODES, UNSETTLED_NOTES
from t5d.engine.search import Search
from t5d.net import NetConfig, build_model
from t5d.train.records import RECORD_VERSION, read_game, record_differences, write_game
from t5d.train.selfplay import GeneratorConfig, generate_game, replay_positions
from t5d.train.targets import target_differences
from tests.engine.helpers import requires_engine

pytestmark = requires_engine

NET = NetConfig(**SMALL)
BASE = GeneratorConfig(playouts=6, batch_size=3, cap_turns=2, cap_boards=20, seed=21)
RESULT_KEYS = {"end", "winner", "result", "actions", "final_boards", "final_fingerprint", "final_active_timelines"}
#: Bare kings on three timelines: every action is at least two decisions (no single move lets White submit).
THREE_BOARDS = ("[k7/8/8/8/8/8/8/7K:-1:1:w]", "[k7/8/8/8/8/8/8/7K:0:1:w]", "[k7/8/8/8/8/8/8/7K:+1:1:w]")


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(0)
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


def _generate(evaluator, cfg, **kwargs):
    with torch.no_grad():
        return generate_game(evaluator, cfg, net_cfg=NET, **kwargs)


def test_the_generator_records_its_budget_in_format_version_2(evaluator):
    """The budget's default is netplay's; 0 and beyond T5D_NO_LIMIT are refused. A game names its budget
    in ``generator_config`` and nowhere else: within the default budget the game is the exact one, its
    record the same but for that key; the version stays 2 and the result keeps its seven keys."""
    assert GeneratorConfig().terminal_nodes == TERMINAL_NODES
    for bad in (0, T5D_NO_LIMIT + 1):
        with pytest.raises(ValueError, match="terminal_nodes"):
            replace(BASE, terminal_nodes=bad).validate()
    budgeted = _generate(evaluator, BASE).record
    exact = _generate(evaluator, replace(BASE, terminal_nodes=T5D_NO_LIMIT)).record
    assert budgeted.header["generator_config"]["terminal_nodes"] == TERMINAL_NODES
    assert exact.header["generator_config"]["terminal_nodes"] == T5D_NO_LIMIT
    assert record_differences(budgeted, exact) == ["header['generator_config']"]
    assert budgeted.header["record_version"] == RECORD_VERSION == 2
    assert set(budgeted.header["result"]) == RESULT_KEYS
    assert budgeted.end == "turn_cap" and budgeted.n_actions > 0


def test_an_unsettled_turn_start_ends_the_record_unknown(evaluator, monkeypatch, tmp_path):
    """A turn start the budget does not settle (the third here: the verdict made to answer unknown there)
    ends the record "unknown" - no winner, "*", the two actions before it kept, the Termination naming
    the verdict - and the record writes, reads back, replays and gives the same targets again."""
    real = selfplay.turn_start_verdict
    calls = []

    def verdict(game, cfg, **kwargs):
        calls.append(cfg.terminal_nodes)
        return real(game, cfg, **kwargs) if len(calls) <= 2 else "unknown"

    monkeypatch.setattr(selfplay, "turn_start_verdict", verdict)
    cfg = replace(BASE, cap_turns=0, cap_boards=0)
    first = _generate(evaluator, cfg, keep_positions=True)
    calls.clear()
    again = _generate(evaluator, cfg, keep_positions=True)
    rec = first.record
    assert (rec.end, rec.winner, rec.header["result"]["result"], rec.n_actions) == ("unknown", None, "*", 2)
    assert set(calls) == {TERMINAL_NODES}
    assert f'[Termination "unknown{UNSETTLED_NOTES["verdict"]}"]' in rec.header["pgn"]
    back = read_game(write_game(tmp_path / "unknown.npz", rec))
    assert record_differences(rec, back) == []
    assert len(replay_positions(back)) == len(rec.decisions) == len(first.positions) > 0
    assert record_differences(rec, again.record) == []
    assert target_differences(first.targets, again.targets) == []


def test_a_game_unsettled_at_its_first_turn_start_has_no_decision(evaluator, tmp_path):
    """Within one node the start's verdict is unknown: the record has no action and no decision, writes
    and reads back, replays to no input, and its targets are empty."""
    g = _generate(evaluator, replace(BASE, terminal_nodes=1), keep_positions=True)
    rec = g.record
    assert (rec.end, rec.n_actions, rec.decisions, g.positions, g.actions) == ("unknown", 0, [], [], [])
    assert f'[Termination "unknown{UNSETTLED_NOTES["verdict"]}"]' in rec.header["pgn"]
    back = read_game(write_game(tmp_path / "first.npz", rec))
    assert record_differences(rec, back) == [] and replay_positions(back) == []
    assert len(g.targets.kl) == 0


def test_a_choice_that_cannot_be_settled_drops_its_action(evaluator, monkeypatch, tmp_path):
    """No child shown legal within the budget at the action's second decision: the game ends "unknown",
    the action under way dropped - its first decision too, so the record has none, nor their positions
    or resignation events - and the game back at its turn start (the start's fingerprint). The search
    classifies nothing itself and the verification answers unknown after its first, real, call (a small
    test cannot make tier 2 fail by configuration)."""
    real = Search.verify
    calls = []

    def verify(self, index, **kwargs):
        calls.append(kwargs)
        return real(self, index, **kwargs) if len(calls) == 1 else T5D_LEGALITY_UNKNOWN

    monkeypatch.setattr(Search, "verify", verify)
    monkeypatch.setattr(netplay.TurnWitness, "moves", lambda self: [])  # X2's witness withheld: tested apart
    overrides = (("legality_checks", 0), ("legality_nodes", 0), ("verify_unknown", 0), ("hunt", 0), ("settling", 0))
    cfg = replace(BASE, start_fen=THREE_BOARDS, cap_turns=0, cap_boards=0, search_overrides=overrides)
    with Game.from_fen(list(THREE_BOARDS)) as start:
        fingerprint = start.snapshot().fingerprint
    g = _generate(evaluator, cfg, keep_positions=True)
    rec = g.record
    assert len(calls) == 2 and all(c == {"nodes": TERMINAL_NODES} for c in calls)
    assert (rec.end, rec.winner, rec.n_actions, rec.decisions, g.positions, g.actions) == ("unknown", None, 0, [], [], [])
    assert rec.header["resign"]["events"] == []
    assert rec.header["result"]["final_fingerprint"] == fingerprint
    assert f'[Termination "unknown{UNSETTLED_NOTES["choice"]}"]' in rec.header["pgn"]
    back = read_game(write_game(tmp_path / "choice.npz", rec))
    assert record_differences(rec, back) == [] and replay_positions(back) == []
    with Game.from_pgn(rec.header["pgn"]) as replay:
        assert replay.replay_all() == 0


# --- review decision 13's X2: the turn-start verdict's witness ----------------------------------------

UNVERIFIED_OVERRIDES = (("legality_checks", 0), ("legality_nodes", 0), ("verify_unknown", 0), ("hunt", 0), ("settling", 0))


def _coordinates(m):
    return (m.from_timeline, m.from_half_turn, m.to_timeline, m.to_half_turn, m.from_square, m.to_square, m.piece, m.promotion)


def test_the_generator_plays_the_witness_where_no_child_is_known_legal(evaluator, monkeypatch, tmp_path):
    """X2 in the generator: every verification unknown and nothing classified by the search, so each decision
    of the first action plays the turn-start verdict's witness's next step - a "verified_fallback", the move
    known legal (a step the root had not instantiated is recorded as a further child, unvisited) - and the
    action is the witness. The record writes, reads back and replays; its targets build."""
    monkeypatch.setattr(Search, "verify", lambda self, index, **kwargs: T5D_LEGALITY_UNKNOWN)
    cfg = replace(BASE, start_fen=THREE_BOARDS, cap_turns=0, cap_boards=0, max_actions=1, search_overrides=UNVERIFIED_OVERRIDES)
    with Game.from_fen(list(THREE_BOARDS)) as start:
        witness = start.verdict(TERMINAL_NODES)[1]
    g = _generate(evaluator, cfg, keep_positions=True)
    rec = g.record
    assert (rec.end, rec.n_actions) == ("action_limit", 1)
    assert [_coordinates(m) for m in g.actions[0]] == [_coordinates(m) for m in witness]
    moves = [d for d in rec.decisions if d.choice == "verified_fallback"]
    assert len(moves) >= len(witness)
    assert all(d.children.legality[d.played_child] == 0 for d in rec.decisions)  # every played child known legal
    back = read_game(write_game(tmp_path / "witness.npz", rec))
    assert record_differences(rec, back) == []
    assert len(replay_positions(back)) == len(rec.decisions)
    assert len(g.targets.kl) == len(rec.decisions)


@pytest.mark.parametrize("split", ["ply", "tree"])
@pytest.mark.parametrize(
    "start,nodes,actions",
    [
        (dict(start_fen=THREE_BOARDS), 32, 1),
        (dict(variant="Focused - Just Kings"), 64, 5),
    ],
)
def test_the_reviews_four_choice_endings_now_play_on(evaluator, split, start, nodes, actions):
    """The four real unsettled choices the review of decision 13 found (its spec pass; no patching): each at
    an action's first decision, right after the turn start's verdict had found a legal action within the
    same budget. With X2 the witness is played instead: the game goes on and ends later, at a turn start the
    budget cannot settle - never on the choice."""
    cfg = GeneratorConfig(
        playouts=6,
        batch_size=3,
        cap_turns=4,
        cap_boards=30,
        seed=3,
        split=split,
        terminal_nodes=nodes,
        search_overrides=UNVERIFIED_OVERRIDES,
        max_actions=12,
        **start,
    )
    rec = _generate(evaluator, cfg).record
    assert (rec.end, rec.n_actions) == ("unknown", actions)
    assert f'[Termination "unknown{UNSETTLED_NOTES["verdict"]}"]' in rec.header["pgn"]
    assert any(d.choice == "verified_fallback" for d in rec.decisions)
