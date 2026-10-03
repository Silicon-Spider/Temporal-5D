"""Games through the engine's C step API with the network as the evaluator (stage N5).

The search is the engine's (``search.py``, ``t5d/capi_search.h``): it selects, expands,
backs up, judges terminals and chooses; the network - random weights, nothing is trained -
evaluates its leaves through ``evaluator.CStepDriver``. This module plays whole games so,
writes each as a 5dpgn record, and replays the record through the engine's gate adapter
(``t5d-gate``, full mode - the engine's own tool, never an oracle) to show it is legal. It
mirrors the engine's ``play_game`` (``src/selfplay.cpp``; ``t5d-play``) step for step:

* **A turn start is judged first** - by a probe search whose root classification is E4's
  exhaustive detector with no node budget (H37): checkmate, stalemate, or a verdict the
  detector could not reach (the game stops unjudged); then I11's turn cap (turns or
  boards), scored as a draw; then the safety stop on the number of actions.
* **Self-play** (``mode="self"``): a fresh search per decision (C14), each sharing the
  previous one's evaluation cache (``t5d_search_new_sharing``), root Dirichlet noise on
  (E1), the move sampled from the root's visits at ``temperature`` for the first
  ``temperature_decisions`` decisions, then the most visited.
* **Match play** (``mode="match"``): two networks, White's and Black's (different seeds);
  each side carries its own search across its decisions (C12: ``advance``) and re-roots it
  on the opponent's reply (``advance_action``); final selection by the search's rule (LCB),
  noise off. The two searches never share a cache - each holds one network's evaluations.
* **D2's two arms** (``split``): the board-move split - an action is its moves, each its
  own decision, then a submit decision (D8: never automatic); or whole turns - a decision is
  an action, submitted with its last move.
* **A move is played only once known legal**: a chosen child the legality budget left
  unknown is classified exactly first (``Search.verify``) - illegal, it is removed and the
  choice made again; still unknown, the most visited child known legal is played. Every
  move goes through the game handle (``Game.apply`` matches it against the generated moves)
  and every submission through ``Game.submit`` (legal: C4, R1).

The record is the engine reader's long form (``engine/tools/play_record.cpp`` writes the
same): headers Mode, Board, Result, Termination (I11's cap named a convention, not a rule),
then the actions numbered by turn - each move as its source board, piece letter and
squares, ``>``/``>>`` for jumps. ``python -m t5d.engine.netplay`` plays games from the
command line and reports throughput (information, not a verdict: no figure here is a
section 15.3 measurement).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..net.config import NetConfig
from ..net.model import build_model
from ..net.state import PIECE_LETTERS
from .capi import (
    T5D_LEGALITY_ILLEGAL,
    T5D_LEGALITY_LEGAL,
    T5D_MOVE_BRANCHING,
    T5D_MOVE_CAPTURE,
    T5D_MOVE_TRAVEL,
    T5D_NO_LIMIT,
    T5D_SPLIT_BOARD_MOVE,
    T5D_SPLIT_WHOLE_TURN,
    find_library,
)
from .convert import relations_for
from .evaluator import Evaluator, NetworkEvaluator, run_search
from .expansion import Move, board_label, square_name
from .game import Game, relation_bits
from .search import Search, search_params

SPLITS = {"ply": T5D_SPLIT_BOARD_MOVE, "turn": T5D_SPLIT_WHOLE_TURN}
MODES = ("self", "match")

#: A small network for end-to-end play on a CPU (random weights; nothing is trained): two
#: layers at d = 64 - the F1 architecture's shape, scaled down so that a game takes seconds.
SMALL = dict(trunk_dtype="fp32", n_layers=2, d_model=64, n_heads=2, ffn_ratio=4, policy_dim=64, value_hidden=64, opp_policy_dim=32)


def network_config(name: str) -> NetConfig:
    """``"small"`` (``SMALL``) or ``"F1"`` (the committed v1 network as shipped, BF16 trunk)."""
    if name == "small":
        return NetConfig(**SMALL)
    if name == "F1":
        return NetConfig()
    raise ValueError(f"unknown network {name!r}: 'small' or 'F1'")


@dataclass(frozen=True)
class PlayConfig:
    """One game's settings (see the module docstring). Every search constant not named here is
    the engine's default (``t5d_search_params_default``); the proof channel is off (H41)."""

    variant: str = "Standard"
    split: str = "ply"  # D2: "ply" (board-move split) or "turn" (whole turns)
    mode: str = "self"  # "self" or "match"
    playouts: int = 16  # per decision
    batch_size: int = 8  # leaves per network call
    cap_turns: int = 6  # I11: a present past this turn, at a turn start, ends the game (0: none)
    cap_boards: int = 80  # ... or this many boards (0: none)
    temperature: float = 1.0
    temperature_decisions: int = 16
    seed: int = 1
    max_actions: int = 200  # a safety stop, not a rule
    max_listing: int = 256  # whole turns: actions listed per node (the engine's default)

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.playouts < 1 or self.batch_size < 1 or self.max_actions < 1:
            raise ValueError("playouts, batch_size and max_actions must be positive")


@dataclass
class GameResult:
    """One game: its record, how it ended, and where the time went."""

    config: PlayConfig
    record: str
    actions: list[list[Move]]
    end: str  # "checkmate", "stalemate", "turn_cap", "unknown", "action_limit"
    winner: int | None  # 0 white, 1 black (checkmate only)
    decisions: int = 0
    playouts: int = 0
    evaluations: int = 0  # leaves the network evaluated (the cache's hits not counted)
    cache_hits: int = 0
    verified_choices: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    seconds: float = 0.0
    network_seconds: float = 0.0  # the network's forward passes (serve's evaluate phase)
    collect_seconds: float = 0.0  # gather + leaf conversion
    deliver_seconds: float = 0.0  # back-ups in the engine
    verdict_seconds: float = 0.0  # turn-start verdicts (probe searches)
    final_boards: int = 0
    final_fingerprint: str = ""

    @property
    def result(self) -> str:
        """The 5dpgn result: ``1-0``, ``0-1``, ``1/2-1/2`` (stalemate) or ``*`` (not ended by
        the rules - the turn cap included, which self-play scores as a draw)."""
        if self.end == "checkmate":
            return "1-0" if self.winner == 0 else "0-1"
        if self.end == "stalemate":
            return "1/2-1/2"
        return "*"

    def summary(self) -> dict:
        s = self.seconds or math.nan
        batches = self.batch_sizes or [0]
        return {
            "variant": self.config.variant,
            "split": self.config.split,
            "mode": self.config.mode,
            "seed": self.config.seed,
            "actions": len(self.actions),
            "end": self.end,
            "result": self.result,
            "final_boards": self.final_boards,
            "decisions": self.decisions,
            "playouts": self.playouts,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "verified_choices": self.verified_choices,
            "seconds": round(self.seconds, 3),
            "decisions_per_s": round(self.decisions / s, 2),
            "evaluations_per_s": round(self.evaluations / s, 1),
            "network_seconds": round(self.network_seconds, 3),
            "collect_seconds": round(self.collect_seconds, 3),
            "deliver_seconds": round(self.deliver_seconds, 3),
            "verdict_seconds": round(self.verdict_seconds, 3),
            "batches": len(self.batch_sizes),
            "mean_batch": round(float(np.mean(batches)), 2),
            "max_batch": int(max(batches)),
        }


# --- the record ---------------------------------------------------------------------------------


def pgn_move(m: Move, even: bool) -> str:
    """A move in the engine reader's long form: source board, piece letter, source square; then
    ``x`` and the destination square for a move on its board, or ``>``/``>>`` (a jump that does
    not branch / that branches), ``x`` for a capture, the destination board and square; ``=X``
    for a promotion."""
    out = board_label(m.from_timeline, m.from_half_turn, even) + PIECE_LETTERS[m.piece] + square_name(m.from_square)
    capture = "x" if m.flags & T5D_MOVE_CAPTURE else ""
    if not m.flags & T5D_MOVE_TRAVEL:
        out += capture + square_name(m.to_square)
    else:
        out += (">>" if m.flags & T5D_MOVE_BRANCHING else ">") + capture
        out += board_label(m.to_timeline, m.to_half_turn, even) + square_name(m.to_square)
    if m.promotion:
        out += "=" + PIECE_LETTERS[m.promotion]
    return out


def _header(name: str, value: str) -> str:
    return f'[{name} "' + value.replace("\\", "\\\\").replace('"', '\\"') + '"]\n'


def write_record(
    variant: str, actions: Sequence[Sequence[Move]], *, even: bool, first_turn: int, result: str, termination: str, extra=()
) -> str:
    """The 5dpgn record of a game from a named variant's start (White to move first)."""
    out = _header("Mode", "5D") + _header("Board", variant) + _header("Result", result) + _header("Termination", termination)
    for name, value in extra:
        out += _header(name, value)
    out += "\n"
    turn = max(1, first_turn)
    for a, action in enumerate(actions):
        if a % 2 == 0:
            out += ("\n" if a else "") + f"{turn}."
        else:
            out += " /"
        out += "".join(" " + pgn_move(m, even) for m in action)
        if a % 2:
            turn += 1
    return out + (" " if actions else "") + result + "\n"


# --- playing ------------------------------------------------------------------------------------


def turn_start_verdict(game: Game, cfg: PlayConfig) -> str | None:
    """A turn start's verdict: the probe search's root classification - E4's exhaustive
    detector, no node budget (H37) - then I11's cap."""
    params = search_params(
        game.lib, mate_nodes=T5D_NO_LIMIT, turn_cap_turns=cfg.cap_turns, turn_cap_boards=cfg.cap_boards, eval_cache_entries=0
    )
    with Search(game, turn_start=True, params=params) as probe:
        info = probe.info()
    if info.checkmates:
        return "checkmate"
    if info.stalemates:
        return "stalemate"
    if info.unknown_verdicts:
        return "unknown"
    if info.turn_caps:
        return "turn_cap"
    return None


def _decision_params(game: Game, cfg: PlayConfig, relations: int, *, seed: int, noise: bool):
    return search_params(
        game.lib,
        split=SPLITS[cfg.split],
        seed=seed,
        root_noise=int(noise),
        batch_size=cfg.batch_size,
        turn_cap_turns=cfg.cap_turns,
        turn_cap_boards=cfg.cap_boards,
        max_actions=cfg.max_listing,
        expand=(8, "all", relations),
    )


def _mix(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def play_game(evaluators: Evaluator | Sequence[Evaluator], cfg: PlayConfig, *, net_cfg: NetConfig | None = None) -> GameResult:
    """Play one game (see the module docstring). ``evaluators``: one (self-play) or White's and
    Black's (match play; one is used for both sides if a single one is given). ``net_cfg``: the
    network's configuration - its register counts and the typed relations it reads."""
    cfg.validate()
    net_cfg = NetConfig() if net_cfg is None else net_cfg
    evs = list(evaluators) if isinstance(evaluators, (list, tuple)) else [evaluators]
    if cfg.mode == "match" and len(evs) == 1:
        evs = evs * 2
    relations = relation_bits(relations_for(net_cfg))
    regs = dict(regs_per_board=net_cfg.regs_per_board, global_regs=net_cfg.global_regs)
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    game = Game.variant(cfg.variant)
    start = game.info()
    res = GameResult(config=cfg, record="", actions=[], end="", winner=None)
    shared: Search | None = None  # self-play: the previous decision's search, the cache's donor (C14)
    carried: list[Search | None] = [None, None]  # match play: each side's search
    try:
        while True:
            v0 = time.perf_counter()
            verdict = turn_start_verdict(game, cfg)
            res.verdict_seconds += time.perf_counter() - v0
            if verdict is not None:
                res.end = verdict
                if verdict == "checkmate":
                    res.winner = 1 - game.info().to_move
                break
            if len(res.actions) >= cfg.max_actions:
                res.end = "action_limit"
                break
            mover = game.info().to_move
            action: list[Move] = []
            turn_start = True
            if cfg.mode == "match" and carried[mover] is not None and res.actions:
                _reroot(carried, mover, res.actions[-1], game)
            while True:
                if cfg.mode == "self":
                    seed = _mix(cfg.seed + 0x9E3779B97F4A7C15 * (res.decisions + 1))
                    params = _decision_params(game, cfg, relations, seed=seed, noise=True)
                    search = Search(game, turn_start=turn_start, params=params, share=shared)
                    if shared is not None:
                        shared.close()  # the cache lives on in the new search
                    shared = search
                else:
                    if carried[mover] is None:
                        params = _decision_params(game, cfg, relations, seed=_mix(cfg.seed + 17 * mover), noise=False)
                        carried[mover] = Search(game, turn_start=turn_start, params=params)
                    search = carried[mover]
                before = search.info()
                evaluator = evs[mover] if cfg.mode == "match" else evs[0]
                stats, _ = run_search(search, evaluator, playouts=cfg.playouts, batch_size=cfg.batch_size, **regs)
                pick = _choose(search, cfg, res, stats, evaluator, regs, rng)
                after = search.info()
                res.decisions += 1
                res.playouts += after.playouts - before.playouts
                res.evaluations += after.evaluations - before.evaluations
                res.cache_hits += after.cache_hits - before.cache_hits
                res.batch_sizes.extend(stats.batch_sizes)
                res.network_seconds += stats.evaluate_seconds
                res.collect_seconds += stats.collect_seconds
                res.deliver_seconds += stats.deliver_seconds
                child = search.child(pick)
                submitted = False
                if cfg.split == "turn":
                    for m in child.moves:
                        game.apply(m)
                        action.append(m)
                    game.submit()
                    submitted = True
                elif child.submit:
                    game.submit()
                    submitted = True
                else:
                    game.apply(child.moves[0])
                    action.append(child.moves[0])
                if cfg.mode == "match":
                    search.advance(pick)
                turn_start = False
                if submitted:
                    break
            res.actions.append(action)
        snap = game.snapshot()
        res.final_boards = len(snap.boards)
        res.final_fingerprint = snap.fingerprint
        termination = res.end + (" (self-play's convention, I11: scored as a draw; not a game rule)" if res.end == "turn_cap" else "")
        extra = [
            ("White", "t5d netplay"),
            ("Black", "t5d netplay"),
            (
                "Engine",
                f"t5d search through the C step API: network evaluator (random weights), {cfg.playouts} playouts per decision, "
                f"{'board-move split' if cfg.split == 'ply' else 'whole turns'}, "
                f"{'self-play, root noise' if cfg.mode == 'self' else 'match play, two networks'}, seed {cfg.seed}",
            ),
            ("TurnCap", f"turns {cfg.cap_turns}, boards {cfg.cap_boards}"),
        ]
        res.record = write_record(
            cfg.variant,
            res.actions,
            even=start.even,
            first_turn=start.present // 2,
            result=res.result,
            termination=termination,
            extra=extra,
        )
    finally:
        for s in [shared, *carried]:
            if s is not None:
                s.close()
        game.close()
    res.seconds = time.perf_counter() - t0
    return res


def _choose(search: Search, cfg: PlayConfig, res: GameResult, stats, evaluator, regs, rng) -> int:
    """The decision's child: the final rule (match) or visit sampling (self-play), then made
    known legal (a chosen unknown child is classified exactly first)."""

    def visited() -> bool:
        return any(c.visits > 0 for c in search.children())

    def choose_now() -> int:
        if not visited():  # children were removed: search again
            more, _ = run_search(search, evaluator, playouts=cfg.playouts, batch_size=cfg.batch_size, **regs)
            stats.batch_sizes.extend(more.batch_sizes)
            stats.evaluate_seconds += more.evaluate_seconds
        if cfg.mode == "match":
            return search.choose()[0]
        t = cfg.temperature if res.decisions < cfg.temperature_decisions else 0.0
        return search.choose_by_visits(t, float(rng.random()))

    pick = choose_now()
    while search.child(pick).legality != T5D_LEGALITY_LEGAL:
        verdict = search.verify(pick)
        res.verified_choices += 1
        if verdict == T5D_LEGALITY_LEGAL:
            break
        if verdict == T5D_LEGALITY_ILLEGAL:
            pick = choose_now()
            continue
        legal = [c for c in search.children() if c.legality == T5D_LEGALITY_LEGAL]
        if not legal:
            raise RuntimeError("no root child could be shown legal (the memory ceiling)")
        pick = max(legal, key=lambda c: c.visits).index
    return pick


def _reroot(carried: list[Search | None], mover: int, reply: Sequence[Move], game: Game) -> None:
    """Match play: the mover's carried search, rooted at the opponent's turn start since its own
    last submission, re-roots on the opponent's reply; a search that cannot is dropped (the next
    decision makes a fresh one)."""
    s = carried[mover]
    try:
        s.advance_action(list(reply))
        with s.root_game() as root:
            same = root.snapshot().fingerprint == game.snapshot().fingerprint
    except Exception:  # noqa: BLE001 - any refusal: start afresh
        same = False
    if not same:
        s.close()
        carried[mover] = None


# --- replaying through the engine's gate adapter ------------------------------------------------------


def gate_executable() -> Path:
    """``t5d-gate`` beside the engine library (``py engine/build.py`` builds both), or the
    ``T5D_GATE`` environment variable."""
    env = os.environ.get("T5D_GATE", "").strip()
    if env:
        return Path(env)
    name = "t5d-gate.exe" if sys.platform in ("win32", "cygwin") else "t5d-gate"
    return find_library().parent / name


@dataclass(frozen=True)
class Replay:
    """What ``t5d-gate`` (full mode) said of a record."""

    status: str  # the end event's: "ok", "rejected", "unsupported", "error"
    actions_accepted: int
    positions: int
    last_terminal: str | None  # the last position's "terminal"
    final_boards: tuple[str, ...]  # the last state event's boards, "<L>:<T>:<c>:<placement>"
    lines: tuple[str, ...]


def replay_with_gate(record: str, *, gate: Path | None = None, timeout: float = 600.0) -> Replay:
    """Replay ``record`` through the engine's gate adapter in full mode (a subprocess: the
    engine's own tool - legal replay, check status, legal actions and terminal at every
    position)."""
    gate = gate_executable() if gate is None else gate
    proc = subprocess.run([str(gate)], input=record.encode("utf-8"), capture_output=True, timeout=timeout, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"t5d-gate exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')}")
    lines = tuple(line for line in proc.stdout.decode("utf-8").splitlines() if line.strip())
    events = [json.loads(line) for line in lines]
    end = events[-1]
    positions = [e for e in events if e.get("event") == "position"]
    states = [e for e in events if e.get("event") == "state"]
    return Replay(
        status=end.get("status", "?"),
        actions_accepted=int(end.get("actions_accepted", -1)),
        positions=len(positions),
        last_terminal=positions[-1].get("terminal") if positions else None,
        final_boards=tuple(states[-1]["boards"]) if states else (),
        lines=lines,
    )


def gate_boards(game_snapshot) -> tuple[str, ...]:
    """A snapshot's boards in the gate's state-event form, ``<L>:<T>:<c>:<placement>``."""
    out = []
    for _, _, fen in game_snapshot.boards:
        placement, label, turn, colour = fen[1:-1].split(":")
        out.append(f"{label}:{turn}:{colour}:{placement}")
    return tuple(out)


def check_replay(res: GameResult, replay: Replay) -> list[str]:
    """Where the gate's replay disagrees with the game: every action accepted, one position more
    than actions, the last position's verdict the game's end, the final boards the game's."""
    problems = []
    if replay.status != "ok":
        problems.append(f"status {replay.status}")
    if replay.actions_accepted != len(res.actions):
        problems.append(f"{replay.actions_accepted} actions accepted of {len(res.actions)}")
    if replay.positions != len(res.actions) + 1:
        problems.append(f"{replay.positions} positions for {len(res.actions)} actions")
    expected = {"checkmate": "checkmate", "stalemate": "stalemate"}.get(res.end, "none")
    if res.end in ("checkmate", "stalemate", "turn_cap", "action_limit") and replay.last_terminal != expected:
        problems.append(f"the last position is {replay.last_terminal!r}, the game ended {res.end}")
    with Game.from_pgn(res.record) as g:
        g.replay_all()
        boards = gate_boards(g.snapshot())
    if replay.final_boards != boards:
        problems.append("the final boards differ from the record's replay by the engine's reader")
    return problems


# --- the command line ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Games through the C step API with the network (random weights) as the evaluator.")
    parser.add_argument("--variant", action="append", help="a named variant (repeatable; default Standard and Two Timelines)")
    parser.add_argument("--split", action="append", choices=sorted(SPLITS), help="D2's arm (repeatable; default both)")
    parser.add_argument("--mode", action="append", choices=MODES, help="self-play or match play (repeatable; default both)")
    parser.add_argument("--games", type=int, default=1, help="games per (variant, split, mode)")
    parser.add_argument("--net", default="small", choices=["small", "F1"])
    parser.add_argument("--playouts", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cap-turns", type=int, default=6)
    parser.add_argument("--cap-boards", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (the machine is shared)")
    parser.add_argument("--out", type=Path, help="write each record here (a directory)")
    parser.add_argument("--no-replay", action="store_true", help="do not replay the records through t5d-gate")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    net_cfg = network_config(args.net)
    nets = [build_model(net_cfg, seed=args.seed + k).eval() for k in (0, 1)]
    evaluators = [NetworkEvaluator(n, max_batch=args.batch) for n in nets]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    failed = 0
    for variant in args.variant or ["Standard", "Standard - Two Timelines"]:
        for split in args.split or ["ply", "turn"]:
            for mode in args.mode or ["self", "match"]:
                for g in range(args.games):
                    cfg = PlayConfig(
                        variant=variant,
                        split=split,
                        mode=mode,
                        playouts=args.playouts,
                        batch_size=args.batch,
                        cap_turns=args.cap_turns,
                        cap_boards=args.cap_boards,
                        seed=args.seed + 101 * g,
                    )
                    with torch.no_grad():
                        res = play_game(evaluators if mode == "match" else evaluators[0], cfg, net_cfg=net_cfg)
                    s = res.summary()
                    if not args.no_replay:
                        problems = check_replay(res, replay_with_gate(res.record))
                        s["replay"] = "ok" if not problems else "; ".join(problems)
                        failed += bool(problems)
                    if args.out:
                        name = f"{variant.replace(' ', '').replace('-', '_')}_{split}_{mode}_{cfg.seed}.5dpgn"
                        (args.out / name).write_text(res.record, encoding="utf-8")
                        s["record"] = name
                    summaries.append(s)
                    print(json.dumps(s), flush=True)
    conditions = {
        "net": args.net,
        "net_config": {k: v for k, v in asdict(net_cfg).items() if k in ("d_model", "n_layers", "n_heads", "ffn_ratio", "trunk_dtype")},
        "torch_threads": args.threads,
        "torch": torch.__version__,
        "playouts": args.playouts,
        "batch": args.batch,
        "cap_turns": args.cap_turns,
        "cap_boards": args.cap_boards,
    }
    total_s = sum(s["seconds"] for s in summaries) or math.nan
    totals = {
        "games": len(summaries),
        "decisions_per_s": round(sum(s["decisions"] for s in summaries) / total_s, 2),
        "evaluations_per_s": round(sum(s["evaluations"] for s in summaries) / total_s, 1),
        "replays_failed": failed,
    }
    print(json.dumps({"conditions": conditions, "totals": totals}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GameResult",
    "PlayConfig",
    "Replay",
    "check_replay",
    "gate_boards",
    "gate_executable",
    "network_config",
    "pgn_move",
    "play_game",
    "replay_with_gate",
    "turn_start_verdict",
    "write_record",
]
