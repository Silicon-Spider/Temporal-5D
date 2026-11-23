"""Register H22 (section 11.0; review decision 13, 2026-11-15, settling build-log issue 408): the game
loop's terminal test takes a node budget, never a time limit, and a turn start it cannot settle ends
the game unscored - never a guessed mate. The same budget covers the loop's other exact question,
whether a chosen move is legal: with no child known legal within it, the game ends unscored too.

Here: the binding's budgeted verification (``Search.verify``'s ``checks`` and ``nodes``, the C API's
``t5d_search_verify_budgeted``), the turn-start verdict under a node budget (unknown or exact, never
another verdict; deterministic), ``netplay.play_game``'s two unscored endings - the action under way
dropped and the game put back at its turn start - and self-play's determinism under a budget.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from t5d.engine import Game, NetworkEvaluator, netplay
from t5d.engine.capi import T5D_EVALUATOR_UNIFORM, T5D_LEGALITY_LEGAL, T5D_LEGALITY_UNKNOWN, T5D_NO_LIMIT
from t5d.engine.netplay import TERMINAL_NODES, UNSETTLED_NOTES, PlayConfig, network_config, play_game, turn_start_verdict
from t5d.engine.search import Search, search_params
from t5d.net import build_model
from tests.engine.helpers import requires_engine

pytestmark = requires_engine

BARE_KINGS = "[k7/8/8/8/8/8/8/7K:0:1:w]"  # every White move is the royal's: the fast classification settles none
#: test_netplay's three verdict positions: Kh1 mated (Qg2 protected by Kf3), stalemated, ongoing.
VERDICTS = [
    ("[8/8/8/8/8/5k2/6q1/7K:0:1:w]", "checkmate"),
    ("[k7/8/8/8/8/6q1/8/7K:0:1:w]", "stalemate"),
    ("[k7/8/8/8/8/8/8/7K:0:1:w]", None),
]
MATED = VERDICTS[0][0]
#: Bare kings on three timelines: no single move - a jump touches two boards at most - lets White submit,
#: so every action is at least two decisions (on two adjacent timelines a jump would do).
THREE_BOARDS = ("[k7/8/8/8/8/8/8/7K:-1:1:w]", "[k7/8/8/8/8/8/8/7K:0:1:w]", "[k7/8/8/8/8/8/8/7K:+1:1:w]")
QUIET = dict(hunt=False, settling=False, asset_map=False)
UNVERIFIED = dict(legality_checks=0, legality_nodes=0, verify_unknown=0)


@pytest.fixture(scope="module")
def net():
    cfg = network_config("small")
    return cfg, NetworkEvaluator(build_model(cfg, seed=1).eval(), max_batch=8)


def _searched(game: Game) -> Search:
    s = Search(game, params=search_params(game.lib, **QUIET, **UNVERIFIED))
    s.run(T5D_EVALUATOR_UNIFORM, 32)
    return s


def test_the_binding_verifies_within_a_budget():
    """Within no budget a chosen unknown move stays unknown and stays a child; within none at all
    (the default) it is classified exactly - the C API's t5d_search_verify."""
    with Game.from_fen([BARE_KINGS]) as g, _searched(g) as s:
        n = len(s.children())
        assert n > 0 and all(c.legality == T5D_LEGALITY_UNKNOWN for c in s.children())
        assert s.verify(0, checks=0, nodes=0) == T5D_LEGALITY_UNKNOWN
        assert len(s.children()) == n and s.child(0).legality == T5D_LEGALITY_UNKNOWN
        assert s.verify(0) == T5D_LEGALITY_LEGAL
        assert s.child(0).legality == T5D_LEGALITY_LEGAL


def test_the_default_budget_is_the_searchs_own():
    """TERMINAL_NODES, a placeholder, is the search's own mate_nodes default (the budget its root check
    runs under; the engine's kTerminalNodes is pinned to it too); 1 to T5D_NO_LIMIT is accepted."""
    assert TERMINAL_NODES == 2_000_000 == search_params().mate_nodes
    assert PlayConfig().terminal_nodes == TERMINAL_NODES
    for bad in (0, -1, T5D_NO_LIMIT + 1):
        with pytest.raises(ValueError, match="terminal_nodes"):
            PlayConfig(terminal_nodes=bad).validate()
    PlayConfig(terminal_nodes=1).validate()
    PlayConfig(terminal_nodes=T5D_NO_LIMIT).validate()


def _verdicts(fen: str, budgets) -> list[str | None]:
    cfg = PlayConfig(cap_turns=0, cap_boards=0)
    out = []
    for n in budgets:
        with Game.from_fen([fen]) as g:
            out.append(turn_start_verdict(g, cfg, nodes=n))
    return out


@pytest.mark.parametrize("fen,exact", VERDICTS)
def test_a_turn_start_verdict_is_unknown_or_exact_never_another(fen, exact):
    """H37: a budget that stops the detector answers "unknown", never a verdict it did not prove. Below
    the fewest nodes that settle the turn start every budget is unknown, from them on every one is the
    exact verdict; the same sequence again on fresh handles (no clock is read)."""
    budgets = list(range(1, 257))
    seq = _verdicts(fen, budgets)
    assert set(seq) <= {"unknown", exact}
    settled = [n for n, v in zip(budgets, seq, strict=True) if v != "unknown"]
    assert settled, "256 nodes settle each of these one-board positions"
    least = settled[0]
    assert all(v == "unknown" for v in seq[: least - 1]) and all(v == exact for v in seq[least - 1 :])
    assert _verdicts(fen, [least, least + 1, 2 * least, TERMINAL_NODES, T5D_NO_LIMIT]) == [exact] * 5
    assert _verdicts(fen, budgets) == seq
    if exact == "checkmate":
        assert least > 1  # proving a mate takes more than the first node


def _least(fen: str) -> int:
    seq = _verdicts(fen, range(1, 257))
    return 1 + next(i for i, v in enumerate(seq) if v != "unknown")


def test_an_unsettled_turn_start_ends_the_game_unscored_never_as_a_mate(net):
    """The mated start within one node fewer than its proof: "unknown" at once - no winner, "*", no
    action, the verdict named the question, the Termination saying so; within the default budget the
    same start is checkmate. The rules come before the cap: capped, it is still "unknown"."""
    cfg, ev = net
    small = _least(MATED) - 1
    base = PlayConfig(start_fen=(MATED,), playouts=2, cap_turns=0, cap_boards=0, **QUIET)
    with torch.no_grad():
        res = play_game(ev, replace(base, terminal_nodes=small), net_cfg=cfg)
        mate = play_game(ev, base, net_cfg=cfg)
    assert (res.end, res.winner, res.result, res.actions, res.unsettled) == ("unknown", None, "*", [], "verdict")
    assert '[Result "*"]' in res.record
    assert f'[Termination "unknown{UNSETTLED_NOTES["verdict"]}"]' in res.record
    assert res.summary()["unsettled"] == "verdict"
    assert (mate.end, mate.winner, mate.result, mate.unsettled) == ("checkmate", 1, "0-1", "")
    with Game.from_fen([MATED]) as g:
        assert turn_start_verdict(g, PlayConfig(cap_turns=0, cap_boards=1), nodes=small) == "unknown"


def test_a_choice_no_budget_settles_ends_the_game_at_its_turn_start(net, monkeypatch):
    """No child shown legal within the budget, in the middle of an action: the game ends "unknown",
    its question the choice, the action under way not recorded and the game put back at that
    action's turn start (the final fingerprint the start's; the record replays to no action). A small
    test cannot make tier 2 fail by configuration, so the verification is made to answer unknown
    after its first, real, call - and the search classifies nothing itself."""
    cfg, ev = net
    real_params = netplay._decision_params

    def unverified(*args, **kwargs):
        p = real_params(*args, **kwargs)
        for name, value in UNVERIFIED.items():
            setattr(p, name, value)
        return p

    real_verify = Search.verify
    calls = []

    def verify(self, index, **kwargs):
        calls.append(kwargs)
        return real_verify(self, index, **kwargs) if len(calls) == 1 else T5D_LEGALITY_UNKNOWN

    monkeypatch.setattr(netplay, "_decision_params", unverified)
    monkeypatch.setattr(Search, "verify", verify)
    with Game.from_fen(list(THREE_BOARDS)) as start:
        fingerprint = start.snapshot().fingerprint
        assert len(start.moves()) > 0
    play = PlayConfig(start_fen=THREE_BOARDS, playouts=4, cap_turns=0, cap_boards=0, **QUIET)
    with torch.no_grad():
        res = play_game(ev, play, net_cfg=cfg)
    assert len(calls) == 2 and all(c == {"nodes": TERMINAL_NODES} for c in calls)
    assert (res.end, res.unsettled, res.winner, res.actions) == ("unknown", "choice", None, [])
    assert res.decisions == 1  # the first move was played, and the game put back before it
    assert res.final_fingerprint == fingerprint
    assert f'[Termination "unknown{UNSETTLED_NOTES["choice"]}"]' in res.record
    with Game.from_pgn(res.record) as replay:
        assert replay.replay_all() == 0
        assert replay.snapshot().fingerprint == fingerprint


def test_self_play_under_a_budget_is_deterministic(net):
    """A node budget reproduces: at these sizes the default budget settles every question and plays
    the game exactly as no budget does; two games within one node both end "unknown" alike."""
    cfg, ev = net
    base = PlayConfig(playouts=4, cap_turns=2, cap_boards=20, seed=5, **QUIET)
    with torch.no_grad():
        games = [play_game(ev, replace(base, terminal_nodes=n), net_cfg=cfg) for n in (T5D_NO_LIMIT,) * 2]
        games += [play_game(ev, base, net_cfg=cfg) for _ in range(2)]
        tiny = [play_game(ev, replace(base, terminal_nodes=1), net_cfg=cfg) for _ in range(2)]
    first = games[0]
    assert first.end == "turn_cap" and first.actions
    for g in games[1:]:
        assert (g.record, g.end, g.actions, g.final_fingerprint) == (first.record, first.end, first.actions, first.final_fingerprint)
    assert tiny[0].end == "unknown" and tiny[0].unsettled == "verdict" and tiny[0].actions == []
    assert tiny[1].record == tiny[0].record
