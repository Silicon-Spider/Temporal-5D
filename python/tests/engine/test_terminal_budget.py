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

import os
import re
from dataclasses import replace

import pytest
import torch

from t5d.engine import Game, NetworkEvaluator, netplay
from t5d.engine.capi import T5D_EVALUATOR_UNIFORM, T5D_LEGALITY_LEGAL, T5D_LEGALITY_UNKNOWN, T5D_NO_LIMIT
from t5d.engine.errors import EngineArgumentError
from t5d.engine.netplay import (
    TERMINAL_NODES,
    UNSETTLED_NOTES,
    PlayConfig,
    gate_executable,
    network_config,
    play_game,
    replay_problems,
    replay_with_gate,
    turn_start_verdict,
)
from t5d.engine.search import Search, search_params
from t5d.net import build_model
from tests.engine.helpers import REQUIRE_VAR, requires_engine

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
    after its first, real, call - and the search classifies nothing itself. The verdict's witness is
    withheld (review decision 13's X2 would play it while the moves are its prefix; tested apart)."""
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
    monkeypatch.setattr(netplay.TurnWitness, "moves", lambda self: [])
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
    unnamed = re.compile(r'^\[TerminalNodes "[0-9]+"\]\n', re.MULTILINE)  # the exact games name their budget (L5)
    for g in games[1:]:
        assert (unnamed.sub("", g.record), g.end, g.actions, g.final_fingerprint) == (
            unnamed.sub("", first.record),
            first.end,
            first.actions,
            first.final_fingerprint,
        )
    assert tiny[0].end == "unknown" and tiny[0].unsettled == "verdict" and tiny[0].actions == []
    assert tiny[1].record == tiny[0].record


# --- review decision 13's X2: the turn-start verdict's witness ----------------------------------------


def _coordinates(m):
    return (m.from_timeline, m.from_half_turn, m.to_timeline, m.to_half_turn, m.from_square, m.to_square, m.piece, m.promotion)


@pytest.mark.parametrize("fen,exact", VERDICTS)
def test_the_game_verdict_answers_with_a_legal_witness(fen, exact):
    """``Game.verdict`` (``t5d_game_verdict``): E4's verdict at a turn start within a budget - the exact one
    with none, "unknown" within too few nodes, never another (H37) - and for an ongoing one its witness, a
    legal action: played move by move, its submission is legal. Deterministic; out of range refused."""
    with Game.from_fen([fen]) as g:
        verdict, witness = g.verdict()
        assert verdict == (exact or "ongoing")
        assert (len(witness) > 0) == (verdict == "ongoing")
        assert g.verdict() == (verdict, witness)
        assert g.verdict(1)[0] in ("unknown", verdict)
        for bad in (-1, T5D_NO_LIMIT + 1):
            with pytest.raises(ValueError, match="nodes"):
                g.verdict(bad)
        if witness:
            with g.copy() as c:
                for m in witness:
                    c.apply(m)
                c.submit()


def test_the_witness_of_three_boards_is_a_whole_action_recorded_state_by_state():
    """On three timelines the witness is at least two moves (no single move lets White submit); each is
    recorded in the state it is made in, so it equals the move the game offers there. A handle in
    mid-action is refused."""
    with Game.from_fen(list(THREE_BOARDS)) as g:
        verdict, witness = g.verdict(TERMINAL_NODES)
        assert verdict == "ongoing" and len(witness) >= 2
        with g.copy() as c:
            for m in witness:
                assert m in c.moves()
                c.apply(m)
            with pytest.raises(EngineArgumentError):
                c.verdict()
            c.submit()


def _unverified_netplay(monkeypatch):
    real_params = netplay._decision_params

    def unverified(*args, **kwargs):
        p = real_params(*args, **kwargs)
        for name, value in UNVERIFIED.items():
            setattr(p, name, value)
        return p

    monkeypatch.setattr(netplay, "_decision_params", unverified)
    monkeypatch.setattr(Search, "verify", lambda self, index, **kwargs: T5D_LEGALITY_UNKNOWN)


@pytest.mark.parametrize("mode", ["self", "match"])
def test_with_no_child_known_legal_netplay_plays_the_witness(net, monkeypatch, mode):
    """X2: the turn start's verdict found a legal action within the budget, so a decision with no child known
    legal plays its witness's next step - here every verification answers unknown and the search classifies
    nothing, so the first action is the witness, move for move, and the game goes on (before X2 it ended
    unscored at once, its question the choice). Under match play a step the tree did not hold drops the
    carried search (the next decision makes a fresh one). The record replays."""
    cfg, ev = net
    _unverified_netplay(monkeypatch)
    with Game.from_fen(list(THREE_BOARDS)) as start:
        witness = start.verdict(TERMINAL_NODES)[1]
    play = PlayConfig(start_fen=THREE_BOARDS, mode=mode, playouts=4, cap_turns=0, cap_boards=0, max_actions=1, **QUIET)
    with torch.no_grad():
        res = play_game(ev, play, net_cfg=cfg)
    assert (res.end, res.unsettled) == ("action_limit", "")
    assert [_coordinates(m) for m in res.actions[0]] == [_coordinates(m) for m in witness]
    assert res.witness_moves >= len(witness)
    with Game.from_pgn(res.record) as replay:
        assert replay.replay_all() == 1
        assert replay.snapshot().fingerprint == res.final_fingerprint


# --- the review of decision 13: its mutation pass's candidate tests, adopted (X1, X6, L10) -------------

#: Bare kings on five timelines: a move touches two boards at most, so no two moves complete White's action.
FIVE_BOARDS = tuple(f"[k7/8/8/8/8/8/8/7K:{t}:1:w]" for t in ("-2", "-1", "0", "+1", "+2"))


def _verify_real_for(n: int, monkeypatch) -> list:
    """Search.verify answers for real for its first n calls, then unknown; the verdict's witness withheld
    (X2 would play it while the moves are its prefix - tested apart)."""
    real = Search.verify
    calls: list = []

    def verify(self, index, **kwargs):
        calls.append(kwargs)
        return real(self, index, **kwargs) if len(calls) <= n else T5D_LEGALITY_UNKNOWN

    monkeypatch.setattr(Search, "verify", verify)
    monkeypatch.setattr(netplay.TurnWitness, "moves", lambda self: [])
    return calls


def _unverified_params(monkeypatch):
    real_params = netplay._decision_params

    def unverified(*args, **kwargs):
        p = real_params(*args, **kwargs)
        for name, value in UNVERIFIED.items():
            setattr(p, name, value)
        return p

    monkeypatch.setattr(netplay, "_decision_params", unverified)


def test_checks_and_nodes_reach_their_own_tiers():
    """X1 (M24, M28): the binding hands checks to tier 2 and nodes to tier 3, never the other way. Bare kings:
    tier 3 needs 4 search nodes (2 leave the move unknown), one tier-2 check settles it."""
    with Game.from_fen([BARE_KINGS]) as g, _searched(g) as s:
        assert s.verify(0, checks=0, nodes=2) == T5D_LEGALITY_UNKNOWN
        assert s.child(0).legality == T5D_LEGALITY_UNKNOWN
    with Game.from_fen([BARE_KINGS]) as g, _searched(g) as s:
        assert s.verify(0, checks=1, nodes=0) == T5D_LEGALITY_LEGAL


def test_the_bindings_default_verification_is_exact(monkeypatch):
    """M31: ``Search.verify``'s default is exact - T5D_NO_LIMIT for both - not the game loop's budget."""
    with Game.from_fen([BARE_KINGS]) as g, _searched(g) as s:
        seen = []
        real_call = s._lib.call

        def call(name, *args, **kwargs):
            if name == "t5d_search_verify_budgeted":
                seen.append((args[2].value, args[3].value))
            return real_call(name, *args, **kwargs)

        monkeypatch.setattr(s._lib, "call", call)
        assert s.verify(0) == T5D_LEGALITY_LEGAL
        assert seen == [(T5D_NO_LIMIT, T5D_NO_LIMIT)]


def test_the_budgeted_verification_keeps_the_memory_ceiling():
    """X7 (M26): tier 3 alone (no checks, nodes unlimited) settles a child of bare kings on three timelines;
    under a search whose mate_bytes is below one held position it stops at that ceiling - unknown, the child
    kept."""
    for mate_bytes, want in ((None, T5D_LEGALITY_LEGAL), (1, T5D_LEGALITY_UNKNOWN)):
        extra = {} if mate_bytes is None else {"mate_bytes": mate_bytes}
        with Game.from_fen(list(THREE_BOARDS)) as g:
            s = Search(g, params=search_params(g.lib, **QUIET, **UNVERIFIED, **extra))
            s.run(T5D_EVALUATOR_UNIFORM, 32)
            n = len(s.children())
            assert s.verify(0, checks=0, nodes=T5D_NO_LIMIT) == want
            assert len(s.children()) == n
            s.close()


def test_netplay_rolls_back_the_whole_action_after_two_moves(net, monkeypatch):
    """X6 (M39): two moves played in the action, the third choice unsettled - the game is put back before both,
    not only the last; the record replays to no action."""
    cfg, ev = net
    _unverified_params(monkeypatch)
    calls = _verify_real_for(2, monkeypatch)
    play = PlayConfig(start_fen=FIVE_BOARDS, playouts=4, cap_turns=0, cap_boards=0, **QUIET)
    with Game.from_fen(list(FIVE_BOARDS)) as start:
        fingerprint = start.snapshot().fingerprint
    with torch.no_grad():
        res = play_game(ev, play, net_cfg=cfg)
    assert len(calls) == 3
    assert (res.end, res.unsettled, res.actions) == ("unknown", "choice", [])
    assert res.decisions == 2  # two moves played in the action, then the game put back before both
    assert res.final_fingerprint == fingerprint
    with Game.from_pgn(res.record) as replay:
        assert replay.replay_all() == 0


def test_netplay_decision_log_survives_an_unsettled_choice(net, monkeypatch):
    """L10 (M47): the unsettled decision is not logged (no move was chosen) - the log holds the decisions made."""
    cfg, ev = net
    _unverified_params(monkeypatch)
    _verify_real_for(1, monkeypatch)
    play = PlayConfig(start_fen=THREE_BOARDS, playouts=4, cap_turns=0, cap_boards=0, decision_log=True, **QUIET)
    with torch.no_grad():
        res = play_game(ev, play, net_cfg=cfg)
    assert (res.end, res.unsettled) == ("unknown", "choice")
    assert len(res.decision_log) == res.decisions == 1


def test_netplay_cli_takes_terminal_nodes(monkeypatch, capsys):
    """L10 (M42): ``--terminal-nodes`` reaches the games - within one node the start's verdict is unknown."""
    import json

    seen = []
    real = netplay.play_game

    def spy(evs, cfg, **kwargs):
        seen.append(cfg.terminal_nodes)
        return real(evs, cfg, **kwargs)

    monkeypatch.setattr(netplay, "play_game", spy)
    argv = [
        *("--net", "small", "--threads", str(torch.get_num_threads()), "--no-replay", "--variant", "Standard"),
        *("--split", "ply", "--mode", "self", "--no-hunt", "--no-settling", "--no-assets", "--terminal-nodes", "1"),
    ]
    assert netplay.main(argv) == 0
    assert seen == [1]
    games = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"unsettled"' in line]
    assert len(games) == 1 and games[0]["end"] == "unknown" and games[0]["unsettled"] == "verdict"


# --- the review of decision 13: L3, L5 and X5 -----------------------------------------------------------


def test_the_binding_refuses_a_budget_out_of_range():
    """L3: a budget outside 0 to T5D_NO_LIMIT is refused, never wrapped into the C type (-1 read as exact,
    2**64 as 0); the child stays as it was."""
    with Game.from_fen([BARE_KINGS]) as g, _searched(g) as s:
        for kwargs in ({"nodes": -1}, {"nodes": T5D_NO_LIMIT + 1}, {"checks": -1}, {"checks": T5D_NO_LIMIT + 1}):
            with pytest.raises(ValueError, match="T5D_NO_LIMIT"):
                s.verify(0, **kwargs)
        assert s.child(0).legality == T5D_LEGALITY_UNKNOWN


def test_a_record_names_a_budget_other_than_the_default(net):
    """L5: a netplay record names the game loop's budget in a TerminalNodes header where it is not the
    default - so two records that differ only by it say so - and is unchanged at the default."""
    cfg, ev = net
    base = PlayConfig(playouts=2, cap_turns=1, cap_boards=20, **QUIET)
    with torch.no_grad():
        small = play_game(ev, replace(base, terminal_nodes=1), net_cfg=cfg)
        default = play_game(ev, base, net_cfg=cfg)
    assert '[TerminalNodes "1"]' in small.record
    assert "TerminalNodes" not in default.record


@pytest.fixture
def gate():
    path = gate_executable()
    if not path.is_file():
        why = f"t5d-gate is not built beside the engine library (looked for {path})"
        if os.environ.get(REQUIRE_VAR, "").strip() == "1":
            pytest.fail(f"{REQUIRE_VAR}=1: {why}", pytrace=False)
        pytest.skip(why)
    return path


def test_an_unsettled_verdicts_game_replays_without_its_final_verdict(net, gate):
    """X5: a game ended on a verdict its budget left unsettled replays through the gate parse-only - full
    mode would ask the gate that very verdict with no budget (minutes at large positions; here, at the mated
    start, a mate it settles at once) - with the states and the final boards still checked; a replay past
    its timeout is a note, never a failure."""
    cfg, ev = net
    play = PlayConfig(start_fen=(MATED,), playouts=2, cap_turns=0, cap_boards=0, terminal_nodes=_least(MATED) - 1, **QUIET)
    with torch.no_grad():
        res = play_game(ev, play, net_cfg=cfg)
    assert (res.end, res.unsettled) == ("unknown", "verdict")
    problems, note = replay_problems(res, gate=gate)
    assert problems == [] and note.startswith("parse-only")
    assert replay_with_gate(res.record, gate=gate).last_terminal == "checkmate"  # what full mode would have asked
    problems, note = replay_problems(res, gate=gate, timeout=1e-6)
    assert problems == [] and note.startswith("not replayed")
