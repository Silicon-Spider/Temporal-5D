"""Games through the engine's C step API with the network as the evaluator (stage N5).

The search is the engine's (``search.py``, ``t5d/capi_search.h``): it selects, expands,
backs up, judges terminals and chooses; the network - random weights, nothing is trained -
evaluates its leaves through ``evaluator.CStepDriver``. This module plays whole games so,
writes each as a 5dpgn record, and replays the record through the engine's gate adapter
(``t5d-gate``, full mode - the engine's own tool, never an oracle) to show it is legal and
written strictly (the adapter's reader lets nothing pass: no warning). It
mirrors the engine's ``play_game`` (``src/selfplay.cpp``; ``t5d-play``) step for step:

* **A turn start is judged first** - by a probe search whose root classification is E4's
  exhaustive detector within the game loop's node budget (``PlayConfig.terminal_nodes``;
  register H22, section 11.0: a node budget, never a time limit): checkmate, stalemate, or a
  verdict the budget did not reach - the game ends *unscored* ("unknown", never a mate; H37);
  then I11's turn cap (turns or boards), scored as a draw; then the safety stop on the number
  of actions.
* **Self-play** (``mode="self"``): a fresh search per decision (C14), each sharing the
  previous one's evaluation cache (``t5d_search_new_sharing``), root Dirichlet noise on
  (E1), the move sampled from the root's visits at ``temperature`` for the first
  ``temperature_decisions`` decisions, then the most visited.
* **Match play** (``mode="match"``): two networks, White's and Black's (different seeds);
  each side carries its own search across its decisions (C12: ``advance``) and re-roots it
  on the opponent's reply (``advance_action``); final selection by the search's rule (LCB),
  noise off. The two searches never share a cache - each holds one network's evaluations.
  Only match play may search with D9's optimistic policy (section 8.1: "searched with for both
  sides at match play only ... kept out of self-play"): ``PlayConfig.optimistic`` and evaluators
  built with ``policy="optimistic"``; ``play_game`` refuses an optimistic evaluator in self-play,
  and its self-play searches refuse every evaluation that carries the optimistic prior
  (``run_search``'s ``self_play``), however the evaluator is wrapped. The record's Engine header
  names the optimistic prior where it was searched with; no I15 training record is written here.
  The flat form refuses the optimistic prior (``PlayConfig.validate``, and ``run_search`` and the
  step driver whatever the config): D9's head has no leave logit (issue 242(h)).
* **D2's arms** (``split``): the board-move split - an action is its moves, each its own
  decision, then a submit decision (D8: never automatic); the tree form (``"tree"``) - the
  same decisions, each search inheriting the turn-start priors inside the turn and calling
  the network again only where an interaction gate fires (the engine's default gates); or
  the flat form (``"flat"``) - a decision is a whole turn, generated best first by the engine and
  submitted with its last move (it replaced the enumerating whole-turn arm, ``"turn"``, retired).
* **Match play's clock** (stage E11; ``PlayConfig.time``, off): each side a clock of ``time``
  seconds plus ``increment`` after each of its actions (Fischer's, a reading - build-log spec
  issue 456), each side's decisions run by its own ``TimeManager`` (``run_timed_search``) instead
  of fixed playouts - 6o's arms by ``allocator`` and ``signals``; a decision charged its whole wall
  time (search, choice, verification), or on ``model_clock`` its work at the cost model's price
  (deterministic). A side whose clock runs out loses on time (``end`` "flag", result ``*``: the
  clock is the match's convention, not a rule), as the engine's ``play_game`` does - the flagged
  decision's work counted in the totals. The steps after the search follow the engine's policy
  (the E11 review's H1): the search re-run a batch at a time, the chosen move verified within the
  hard bound (``Search.verify(index, seconds)``), the most visited child known legal played where
  it is cut; with none known legal, the children verified in visit order against the side's
  remaining clock - the flag falls if it runs out, never a throw (M1). Self-play takes no clock.
* **6e's steering switches** (H13, H43; spec section 11.2: "an on/off test of its own at match
  play"): ``asset_map`` (on) and ``hunt_suggest`` (off) for every search; ``search_overrides``,
  further ``t5d_search_params`` fields by name; ``side_overrides``, White's and Black's own (match
  play only), so a match can play two settings against each other with one network
  (``switch_match.py`` pairs the colours). Fields the game loop sets itself (``RESERVED_OVERRIDES``)
  are refused. ``GameResult`` counts, per side, the root edges each switch raised and the turn-start
  roots the carried tree had already visited (where Hunt's suggestions do not reach - build-log spec
  issue 477); the record's ``Steering`` header names each side's settings when any is not the
  default.
* **A move is played only once known legal**: a chosen child the legality budget left
  unknown is classified first within the same node budget (``Search.verify``; H22: "the same
  budget covers the loop's other exact question") - illegal, it is removed and the choice made
  again; still unknown, the most visited child known legal is played; with none, the next step of
  the turn-start verdict's witness while the action's moves are its prefix (``TurnWitness``;
  review decision 13's X2); and otherwise the game ends unscored at the action's turn start (the
  action under way is not recorded). Every
  move goes through the game handle (``Game.apply`` matches it against the generated moves)
  and every submission through ``Game.submit`` (legal: C4, R1). ``GameResult.unsettled`` and
  the record's Termination header say which question went unanswered.

The record is the engine reader's long form (``engine/tools/play_record.cpp`` writes the
same): headers Mode, Board, Result, Termination (I11's cap named a convention, not a rule; an
unscored end its question, H22), TerminalNodes where the game loop's budget is not the default,
then the actions numbered by turn - each move as its source board, piece letter and
squares, ``>``/``>>`` for jumps. ``python -m t5d.engine.netplay`` plays games from the
command line and reports throughput (information, not a verdict: no figure here is a
section 15.3 measurement).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
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
    T5D_SPLIT_FLAT,
    T5D_SPLIT_TREE,
    CTimeParams,
    find_library,
)
from .convert import relations_for
from .evaluator import FLAT_OPTIMISTIC, Evaluator, NetworkEvaluator, policy_source, run_search, run_timed_search
from .expansion import Move, board_label, square_name
from .game import Game, relation_bits
from .search import ALLOCATORS, RootChild, Search, TimeManager, search_params, time_params

SPLITS = {"ply": T5D_SPLIT_BOARD_MOVE, "flat": T5D_SPLIT_FLAT, "tree": T5D_SPLIT_TREE}
SPLIT_NAMES = {"ply": "board-move split", "flat": "flat form", "tree": "tree form"}
MODES = ("self", "match")

#: Register H22 (section 11.0; review decision 13, 2026-11-15): the game loop's node budget for
#: its two exact questions - a turn start's verdict and the verification of a chosen move - in
#: search nodes (deterministic: never a time limit). A PLACEHOLDER (build-log issue 413): the spec gives no figure; it
#: is the search's own ``mate_nodes`` default (``t5d/mcts.hpp``), the budget each fresh search's
#: root check already runs under, so the loop and its searches cannot disagree (a test pins the
#: two together), and the engine's ``kTerminalNodes`` (``t5d/selfplay.hpp``). ``T5D_NO_LIMIT``:
#: exact, the behaviour before the ruling.
TERMINAL_NODES = 2_000_000
#: The record's Termination note for each question a game ended unscored on (``GameResult.unsettled``;
#: the engine's ``tools/play_record.cpp`` writes the same).
UNSETTLED_NOTES = {
    "verdict": " (the turn start's verdict stayed unknown within the game loop's budget, H22: unscored; never a mate)",
    "choice": " (the chosen move was not shown legal within the game loop's budget and no other is known legal, H22: unscored)",
}

#: A small network for end-to-end play on a CPU (random weights; nothing is trained): two
#: layers at d = 64 - the F1 architecture's shape, scaled down so that a game takes seconds.
SMALL = dict(
    trunk_dtype="fp32",
    n_layers=2,
    d_model=64,
    n_heads=2,
    ffn_ratio=4,
    policy_dim=64,
    value_hidden=64,
    opp_policy_dim=32,
    soft_policy_dim=32,
)


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
    split: str = "ply"  # D2: "ply" (board-move split), "tree" (the tree form) or "flat" (the flat form)
    mode: str = "self"  # "self" or "match"
    playouts: int = 16  # per decision
    batch_size: int = 8  # leaves per network call
    cap_turns: int = 6  # I11: a present past this turn, at a turn start, ends the game (0: none)
    cap_boards: int = 80  # ... or this many boards (0: none)
    temperature: float = 1.0
    temperature_decisions: int = 16
    seed: int = 1
    max_actions: int = 200  # a safety stop, not a rule
    #: H22: the node budget of the turn-start verdict and of a chosen move's verification
    #: (``TERMINAL_NODES``, a PLACEHOLDER - build-log issue 413; 1 to ``T5D_NO_LIMIT``, which is exact).
    terminal_nodes: int = TERMINAL_NODES
    #: A custom start instead of ``variant``'s: 5DFEN board strings (the player to move inferred
    #: from the present, Q9); the record then names the board "custom" and writes them.
    start_fen: tuple[str, ...] | None = None
    #: D9 (a prototype): both sides search with the optimistic policy - match play only, so the
    #: evaluators must deliver it (``policy="optimistic"``); self-play refuses it, and so does the
    #: flat form (D9's head has no leave logit; issue 242(h)).
    optimistic: bool = False
    #: Stage E10's verification ladder (the engine's, through ``t5d_search_params``; on by default
    #: in 6b''s ranges - Hunt under 100 boards, the settling scans under 50 - and the asset map's
    #: boosts); a game's fresh searches share it as they share the cache. Its proofs never change
    #: play from Python: the proof channel stays off (H41).
    hunt: bool = True
    settling: bool = True
    asset_map: bool = True
    #: H43's flag: Hunt's suggested lines raise the root edges they start (off; 6e's match test).
    hunt_suggest: bool = False
    #: Further ``t5d_search_params`` fields by name for every search (as ``GeneratorConfig``'s), and
    #: White's and Black's own, applied after them - match play only (6e's two settings).
    search_overrides: tuple[tuple[str, object], ...] = ()
    side_overrides: tuple[tuple[tuple[str, object], ...], tuple[tuple[str, object], ...]] = ((), ())
    hunt_ms: float = 50.0
    settling_ms: float = 50.0
    verification_threads: int = 5
    #: Stage E10, H44: record each decision for the offline audit lane (``GameResult.decision_log``,
    #: written by ``decision_log_lines`` as the engine's t5d-audit reads it).
    decision_log: bool = False
    #: Stage E11: match play's clock - each side ``time`` seconds (0: no clock - fixed ``playouts``
    #: per decision) plus ``increment`` after each of its actions; 6o's arms (``allocator``:
    #: ``"integral"``, ``"flat_evals"`` or ``"flat_wall_clock"``; ``signals``: the stop/extend table);
    #: ``model_clock``: the cost model's deterministic clock instead of the wall clock.
    time: float = 0.0
    increment: float = 0.0
    allocator: str = "integral"
    signals: bool = True
    model_clock: bool = False
    #: Any other ``t5d_time_params`` field by name (``search.time_params``): e.g. the model clock's
    #: price, ``cost_seconds_per_evaluation``.
    time_overrides: tuple[tuple[str, float], ...] = ()

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.optimistic and self.mode != "match":
            raise ValueError("the optimistic policy is searched with at match play only, never in self-play (register D9)")
        if self.optimistic and self.split == "flat":
            raise ValueError(FLAT_OPTIMISTIC)
        if self.playouts < 1 or self.batch_size < 1 or self.max_actions < 1:
            raise ValueError("playouts, batch_size and max_actions must be positive")
        if not 1 <= self.terminal_nodes <= T5D_NO_LIMIT:
            raise ValueError("terminal_nodes: 1 to T5D_NO_LIMIT (exact) - the game loop's node budget (H22)")
        if not (self.hunt_ms > 0 and self.settling_ms > 0 and 1 <= self.verification_threads <= 256):
            raise ValueError("the ladder's budgets positive, its threads 1 to 256")
        if not (math.isfinite(self.time) and self.time >= 0 and math.isfinite(self.increment) and self.increment >= 0):
            raise ValueError("the clock's time and increment: finite, 0 or more")
        if self.time > 0 and self.mode != "match":
            raise ValueError("the clock is match play's: self-play's budgets are PCR's (section 12.5), never a clock's")
        if self.allocator not in ALLOCATORS:
            raise ValueError(f"allocator must be one of {sorted(ALLOCATORS)}")
        fields = {name for name, _ in CTimeParams._fields_}
        for entry in self.time_overrides:
            if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)):
                raise ValueError(f"time_overrides: (field name, value) pairs, not {entry!r}")
            if entry[0] in ("allocator", "signals"):
                raise ValueError(f"time_overrides: {entry[0]!r} is PlayConfig.{entry[0]}'s, not an override")
            if entry[0] not in fields:
                raise ValueError(f"time_overrides: t5d_time_params has no field {entry[0]!r}")
        if not (isinstance(self.side_overrides, tuple) and len(self.side_overrides) == 2):
            raise ValueError("side_overrides: a pair - White's overrides, Black's")
        if any(self.side_overrides) and self.mode != "match":
            raise ValueError("side_overrides: match play only (a self-play search serves both sides)")
        for overrides in (self.search_overrides, *self.side_overrides):
            for entry in overrides:
                if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)):
                    raise ValueError(f"search overrides: (field name, value) pairs, not {entry!r}")
                if entry[0] in RESERVED_OVERRIDES:
                    raise ValueError(f"search overrides: {entry[0]!r} is the game loop's (RESERVED_OVERRIDES)")


@dataclass
class GameResult:
    """One game: its record, how it ended, and where the time went."""

    config: PlayConfig
    record: str
    actions: list[list[Move]]
    end: str  # "checkmate", "stalemate", "turn_cap", "unknown", "action_limit", "flag" (E11: lost on time)
    winner: int | None  # 0 white, 1 black (checkmate; or a flag: the side whose clock still ran)
    #: An "unknown" end's question (H22): "verdict" (a turn start the budget could not settle) or
    #: "choice" (no move shown legal within it); "" for every other end.
    unsettled: str = ""
    decisions: int = 0
    #: Moves played as steps of the turn-start verdict's witness where no root child was known legal
    #: (H22; review decision 13's X2).
    witness_moves: int = 0
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
    #: Match play: carried searches that could not be re-rooted on the opponent's reply and were
    #: replaced by a fresh search (0 in a sound run - the tests require it).
    reroot_fallbacks: int = 0
    final_boards: int = 0
    final_fingerprint: str = ""
    #: Stage E10, H44 (``PlayConfig.decision_log``): each decision - its action and index, the
    #: action's moves before it, the root's board count, visits and value, its children and the
    #: child chosen (``decision_log_lines``).
    decision_log: list[dict] = field(default_factory=list)
    #: Stage E11 (``PlayConfig.time``): each side's clock at the end (White, Black; None: no clock),
    #: the timed decisions, why each stopped (by ``T5D_TIME_STOP_*``), the extensions they took, and
    #: the decisions charged more than their hard bound.
    clock_remaining: list[float] | None = None
    timed_decisions: int = 0
    time_stops: dict[int, int] = field(default_factory=dict)
    extensions: int = 0
    hard_overruns: int = 0
    #: 6e's switches (H13, H43), per side (White, Black): the root edges the asset map and Hunt's
    #: suggestions raised, the side's turn-start decisions, and how many of those roots the carried
    #: tree had already visited (Hunt's suggestions do not reach them - issue 477); the proofs the
    #: game's searches withheld and injected (0 while the proof channel is off - H41).
    asset_boosted_edges: list[int] = field(default_factory=lambda: [0, 0])
    suggestion_edges: list[int] = field(default_factory=lambda: [0, 0])
    turn_start_decisions: list[int] = field(default_factory=lambda: [0, 0])
    carried_turn_starts: list[int] = field(default_factory=lambda: [0, 0])
    proofs_withheld: int = 0
    proofs_injected: int = 0

    @property
    def result(self) -> str:
        """The 5dpgn result: ``1-0``, ``0-1``, ``1/2-1/2`` (stalemate) or ``*`` (not ended by
        the rules - the turn cap included, which self-play scores as a draw, and a flag, which is
        the match's convention, scored for the other side)."""
        if self.end == "checkmate":
            return "1-0" if self.winner == 0 else "0-1"
        if self.end == "stalemate":
            return "1/2-1/2"
        return "*"

    def summary(self) -> dict:
        s = self.seconds or math.nan
        batches = self.batch_sizes or [0]
        return {
            "variant": self.config.variant if self.config.start_fen is None else "custom",
            "split": self.config.split,
            "mode": self.config.mode,
            "seed": self.config.seed,
            "actions": len(self.actions),
            "end": self.end,
            "unsettled": self.unsettled,
            "witness_moves": self.witness_moves,
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
            "reroot_fallbacks": self.reroot_fallbacks,
            "batches": len(self.batch_sizes),
            "mean_batch": round(float(np.mean(batches)), 2),
            "max_batch": int(max(batches)),
            **(
                {}
                if self.clock_remaining is None
                else {
                    "clock_remaining": [round(x, 4) for x in self.clock_remaining],
                    "timed_decisions": self.timed_decisions,
                    "time_stops": {str(k): v for k, v in sorted(self.time_stops.items())},
                    "extensions": self.extensions,
                    "hard_overruns": self.hard_overruns,
                }
            ),
        }


# --- the record ---------------------------------------------------------------------------------


def pgn_move(m: Move, even: bool) -> str:
    """A move in the engine reader's long form: source board, piece letter, source square; then
    ``x`` and the destination square for a move on its board, or ``>``/``>>`` (a jump that does
    not branch / that branches), ``x`` for a capture, the destination board and square; ``=X``
    for a promotion. The capture mark is written strictly (H51, rulebook R11): on every move
    that takes a piece - en passant and jumps included - and on no other."""
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
    board: str,
    actions: Sequence[Sequence[Move]],
    *,
    even: bool,
    first_turn: int,
    result: str,
    termination: str,
    extra=(),
    first_player: int = 0,
    size: tuple[int, int] | None = None,
    boards: Sequence[str] = (),
) -> str:
    """The 5dpgn record of a game (the engine's ``tools/play_record.cpp`` layout): headers, for a
    ``"custom"`` board its ``Size`` and the start's 5DFEN board strings, then the actions
    numbered by turn - ``N.`` before White's, `` /`` before Black's, ``Nb.`` when Black moves
    first - then the result."""
    out = _header("Mode", "5D") + _header("Board", board)
    if board == "custom":
        if size is None:
            raise ValueError("a custom board's record needs its size")
        out += _header("Size", f"{size[0]}x{size[1]}")
    out += _header("Result", result) + _header("Termination", termination)
    for name, value in extra:
        out += _header(name, value)
    if board == "custom":
        out += "".join(b + "\n" for b in boards)
    out += "\n"
    colour, turn = int(first_player), max(1, first_turn)
    for a, action in enumerate(actions):
        if colour == 0:
            out += ("\n" if a else "") + f"{turn}."
        else:
            out += f"{turn}b." if a == 0 else " /"
        out += "".join(" " + pgn_move(m, even) for m in action)
        if colour == 1:
            turn += 1
        colour = 1 - colour
    return out + (" " if actions else "") + result + "\n"


# --- playing ------------------------------------------------------------------------------------


def verdict_params(game: Game, cfg: PlayConfig, *, nodes: int | None = None):
    """The turn-start probe's ``t5d_search_params``: E4's detector within ``nodes`` (None:
    ``cfg.terminal_nodes``; register H22) and the memory ceiling the loop's verifications use - a
    ``GeneratorConfig``'s ``search_overrides`` name it (``mate_bytes``), as the C++ loop takes both
    questions' from its search parameters (the review of decision 13, L8) - and I11's caps."""
    overrides = dict(getattr(cfg, "search_overrides", ()))
    ceiling = {"mate_bytes": overrides["mate_bytes"]} if "mate_bytes" in overrides else {}
    return search_params(
        game.lib,
        mate_nodes=int(cfg.terminal_nodes if nodes is None else nodes),
        turn_cap_turns=cfg.cap_turns,
        turn_cap_boards=cfg.cap_boards,
        eval_cache_entries=0,
        **ceiling,
    )


def turn_start_verdict(game: Game, cfg: PlayConfig, *, nodes: int | None = None) -> str | None:
    """A turn start's verdict: the probe search's root classification - E4's exhaustive
    detector within ``nodes`` (None: ``cfg.terminal_nodes``; register H22; ``verdict_params``) -
    then I11's cap. "unknown" where the budget (or the memory ceiling) stopped the detector: never
    a mate (H37); the rules come before the cap, so an unsettled capped turn start is "unknown"
    too. ``cfg``: a ``PlayConfig`` or the training generator's ``GeneratorConfig`` (its caps,
    budget and memory ceiling)."""
    params = verdict_params(game, cfg, nodes=nodes)
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


#: The search fields the game loop sets from the game's own settings, and the proof channel:
#: never overridden (``PlayConfig.search_overrides``, ``side_overrides``).
RESERVED_OVERRIDES = frozenset(
    {"split", "seed", "root_noise", "batch_size", "turn_cap_turns", "turn_cap_boards", "expand", "proof_channel"}
)


def _search_settings(cfg: PlayConfig, side: int | None) -> dict:
    """The steering switches and overrides a search takes: ``cfg``'s, then ``side``'s own."""
    settings = {"asset_map": int(cfg.asset_map), "hunt_suggest": int(cfg.hunt_suggest)}
    settings.update(dict(cfg.search_overrides))
    if side is not None:
        settings.update(dict(cfg.side_overrides[side]))
    return settings


def _decision_params(game: Game, cfg: PlayConfig, relations: int, *, seed: int, noise: bool, side: int | None = None):
    params = dict(
        split=SPLITS[cfg.split],
        seed=seed,
        root_noise=int(noise),
        batch_size=cfg.batch_size,
        turn_cap_turns=cfg.cap_turns,
        turn_cap_boards=cfg.cap_boards,
        expand=(8, "all", relations),
        hunt=int(cfg.hunt),
        settling=int(cfg.settling),
        hunt_ms=float(cfg.hunt_ms),
        settling_ms=float(cfg.settling_ms),
        verification_threads=int(cfg.verification_threads),
    )
    params.update(_search_settings(cfg, side))
    return search_params(game.lib, **params)


def steering_header(cfg: PlayConfig) -> str | None:
    """The record's ``Steering`` header - each side's steering switches and overrides as it
    searched - or None where every search took the defaults (default records stay as they were)."""
    if not (cfg.hunt_suggest or cfg.search_overrides or any(cfg.side_overrides)):
        return None

    def side(k: int) -> str:
        settings = _search_settings(cfg, k if cfg.mode == "match" else None)
        return ", ".join(f"{name}={value}" for name, value in settings.items())

    return f"white: {side(0)}; black: {side(1)}"


def _mix(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _coordinates(m: Move) -> tuple[int, ...]:
    """A move by its coordinates, not by the board indices a record takes from the state it was
    recorded in: two records of one move compare equal whichever state recorded them."""
    return (m.from_timeline, m.from_half_turn, m.to_timeline, m.to_half_turn, m.from_square, m.to_square, m.piece, m.promotion)


class TurnWitness:
    """Register H22 (review decision 13's X2): the turn-start verdict's witness - the legal action E4
    found within the game loop's budget - fetched (``Game.verdict``, the probe's budget and ceiling:
    deterministic, the verdict's own) only when a decision finds no root child known legal.
    ``start``: a handle at the action's turn start that stays there while the action is played - its
    copy under the board-move split and the tree form, the game itself under the flat form (whose
    single decision comes before any move)."""

    def __init__(self, start: Game, cfg: PlayConfig) -> None:
        self._start, self._cfg, self._moves = start, cfg, None

    def moves(self) -> list[Move]:
        if self._moves is None:
            p = verdict_params(self._start, self._cfg)
            verdict, witness = self._start.verdict(int(p.mate_nodes), bytes=int(p.mate_bytes))
            self._moves = witness if verdict == "ongoing" else []
        return self._moves

    def step(self, search: Search, played: Sequence[Move], flat: bool) -> int | RootChild | None:
        """The root child that continues the witness: its next move while ``played`` (the action's
        moves so far) is its prefix, the submission once they are all of it; under the flat form the
        turn that is the whole witness. A step the root has not instantiated (progressive widening
        lists only those) comes back as a child of its own (``off_tree_child``: known legal, unvisited,
        its index past the search's children) - played outside the tree. None: the moves played have
        left the witness (or there is none)."""
        witness = self.moves()
        w = [_coordinates(m) for m in witness]
        if not w:
            return None
        children = search.children()
        if flat:
            if played:
                return None
            found = next((c.index for c in children if [_coordinates(m) for m in c.moves] == w), None)
            return found if found is not None else off_tree_child(len(children), tuple(witness), submit=False)
        k = len(played)
        if k > len(w) or [_coordinates(m) for m in played] != w[:k]:
            return None
        for c in children:
            if k == len(w) and c.submit:
                return c.index
            if k < len(w) and not c.submit and len(c.moves) == 1 and _coordinates(c.moves[0]) == w[k]:
                return c.index
        return off_tree_child(len(children), () if k == len(w) else (witness[k],), submit=k == len(w))


def off_tree_child(index: int, moves: tuple[Move, ...], *, submit: bool) -> RootChild:
    """A root child the search did not instantiate (review decision 13's X2: a step of the verdict's
    witness): known legal, unvisited, no statistics - its index past the search's own children."""
    nan = float("nan")
    return RootChild(
        index=index,
        visits=0,
        submit=submit,
        legality=T5D_LEGALITY_LEGAL,
        lo=-1,
        hi=1,
        prior=0.0,
        q=0.0,
        moves=moves,
        d=0.0,
        evaluated=False,
        network_wdl=(nan, nan, nan),
        network_moves_left=nan,
        network_error=nan,
        child_key=0,
        prior_noise_free=0.0,
        value_variance=0.0,
        boost=0.0,
        exploration_scale=1.0,
        forced_visits=0,
        steered_visits=0,
    )


def play_game(evaluators: Evaluator | Sequence[Evaluator], cfg: PlayConfig, *, net_cfg: NetConfig | None = None) -> GameResult:
    """Play one game (see the module docstring). ``evaluators``: one (self-play) or White's and
    Black's (match play; one is used for both sides if a single one is given). ``net_cfg``: the
    network's configuration - its register counts and the typed relations it reads."""
    cfg.validate()
    net_cfg = NetConfig() if net_cfg is None else net_cfg
    evs = list(evaluators) if isinstance(evaluators, (list, tuple)) else [evaluators]
    if cfg.mode == "match" and len(evs) == 1:
        evs = evs * 2
    sources = {policy_source(ev) for ev in evs}
    if "optimistic" in sources and cfg.mode != "match":
        raise ValueError("an optimistic evaluator searches at match play only, never in self-play (register D9, section 8.1)")
    if cfg.mode == "match" and cfg.optimistic != (sources == {"optimistic"}):
        raise ValueError("PlayConfig.optimistic: both sides' evaluators deliver the optimistic policy, or neither does (D9)")
    relations = relation_bits(relations_for(net_cfg))
    regs = dict(regs_per_board=net_cfg.regs_per_board, global_regs=net_cfg.global_regs)
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    game = Game.variant(cfg.variant) if cfg.start_fen is None else Game.from_fen(list(cfg.start_fen))
    start = game.info()
    start_boards = [fen for _, _, fen in game.snapshot().boards] if cfg.start_fen is not None else []
    res = GameResult(config=cfg, record="", actions=[], end="", winner=None)
    shared: Search | None = None  # self-play: the previous decision's search, the cache's donor (C14)
    carried: list[Search | None] = [None, None]  # match play: each side's search
    clocks: list[TimeManager] = []
    if cfg.time > 0:  # E11: each side's clock and time manager
        tp = time_params(allocator=cfg.allocator, signals=int(cfg.signals), **dict(cfg.time_overrides))
        clocks = [TimeManager(tp), TimeManager(tp)]
        res.clock_remaining = [float(cfg.time), float(cfg.time)]
    turn_start: Game | None = None  # the action's turn start, while its moves are played one by one
    try:
        while True:
            v0 = time.perf_counter()
            verdict = turn_start_verdict(game, cfg)
            res.verdict_seconds += time.perf_counter() - v0
            if verdict is not None:
                res.end = verdict
                if verdict == "checkmate":
                    res.winner = 1 - game.info().to_move
                if verdict == "unknown":
                    res.unsettled = "verdict"  # H22: unscored, never a mate
                break
            if len(res.actions) >= cfg.max_actions:
                res.end = "action_limit"
                break
            mover = game.info().to_move
            action: list[Move] = []
            flagged = False
            if cfg.split != "flat":  # a flag, or a choice left unsettled, mid-action puts the game back here (H22, E11)
                turn_start = game.copy()
            witness = TurnWitness(turn_start if turn_start is not None else game, cfg)
            if cfg.mode == "match" and carried[mover] is not None and res.actions and not _reroot(carried, mover, res.actions[-1], game):
                res.reroot_fallbacks += 1
            while True:
                if cfg.mode == "self":
                    seed = _mix(cfg.seed + 0x9E3779B97F4A7C15 * (res.decisions + 1))
                    params = _decision_params(game, cfg, relations, seed=seed, noise=True)
                    search = Search(game, params=params, share=shared)  # its root: the game's moves made (A6)
                    if shared is not None:
                        shared.close()  # the cache lives on in the new search
                    shared = search
                else:
                    if carried[mover] is None:
                        params = _decision_params(game, cfg, relations, seed=_mix(cfg.seed + 17 * mover), noise=False, side=mover)
                        carried[mover] = Search(game, params=params)
                    search = carried[mover]
                before = search.info()
                evaluator = evs[mover] if cfg.mode == "match" else evs[0]
                d0 = time.perf_counter()
                timed = None
                if clocks:
                    # E11: the side's time manager runs the decision on its clock.
                    timed = run_timed_search(
                        search,
                        evaluator,
                        clocks[mover],
                        remaining=res.clock_remaining[mover],
                        increment=cfg.increment,
                        model_clock=cfg.model_clock,
                        **regs,
                    )
                    stats = timed.stats
                    res.timed_decisions += 1
                    res.time_stops[timed.verdict.stop] = res.time_stops.get(timed.verdict.stop, 0) + 1
                    res.extensions += bin(timed.verdict.extensions).count("1")
                else:
                    stats, _ = run_search(
                        search, evaluator, playouts=cfg.playouts, batch_size=cfg.batch_size, self_play=cfg.mode != "match", **regs
                    )
                dclock = None if timed is None else _DecisionClock(d0, timed.plan.hard_seconds, res.clock_remaining[mover], cfg.model_clock)
                step = partial(witness.step, played=action, flat=cfg.split == "flat")
                out_of_clock = False
                try:
                    pick = _choose(search, cfg, res, stats, evaluator, regs, rng, step, dclock)
                except _OutOfClock:
                    pick, out_of_clock = None, True  # the side's flag falls (charged below)
                if cfg.decision_log and pick is not None:
                    res.decision_log.append(_decision_entry(game, search, pick, len(res.actions), res.decisions, action, start.even))
                after = search.info()
                if pick is not None or out_of_clock:  # a decision is counted where a move is played, or its flag falls
                    res.decisions += 1
                # The decision's work, counted before the clock is charged: a flagged or unsettled
                # decision's too (as the engine's play_game counts it).
                res.playouts += after.playouts - before.playouts
                res.evaluations += after.evaluations - before.evaluations
                res.cache_hits += after.cache_hits - before.cache_hits
                res.batch_sizes.extend(stats.batch_sizes)
                res.network_seconds += stats.evaluate_seconds
                res.collect_seconds += stats.collect_seconds
                res.deliver_seconds += stats.deliver_seconds
                res.asset_boosted_edges[mover] += after.asset_boosted_edges - before.asset_boosted_edges
                res.suggestion_edges[mover] += after.suggestion_edges - before.suggestion_edges
                res.proofs_withheld += after.proofs_withheld - before.proofs_withheld
                res.proofs_injected += after.proofs_injected - before.proofs_injected
                if before.root_turn_start:
                    res.turn_start_decisions[mover] += 1
                    res.carried_turn_starts[mover] += int(before.root_visits > 0)
                if timed is not None and (pick is not None or out_of_clock):
                    # The decision charged to the side's clock: its whole wall time, or its model time
                    # (an unsettled choice ends the game unscored, uncharged, as play_game's does).
                    spent = timed.seconds if cfg.model_clock else time.perf_counter() - d0
                    if spent > timed.plan.hard_seconds:
                        res.hard_overruns += 1
                    res.clock_remaining[mover] -= spent
                    if res.clock_remaining[mover] <= 0 or out_of_clock:
                        flagged = True  # the side's flag fell during its action: not recorded
                        break
                if pick is None:  # H22: no move shown legal within the budget - the game ends unscored
                    res.end, res.unsettled = "unknown", "choice"
                    break
                child = pick if isinstance(pick, RootChild) else search.child(pick)
                submitted = False
                if cfg.split == "flat":
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
                if cfg.mode == "match" and isinstance(pick, RootChild):
                    # A witness step the tree did not hold (X2): the C API re-roots on a child or a
                    # whole action only, so the carried search goes and the side's next decision
                    # makes a fresh one.
                    search.close()
                    carried[mover] = None
                elif cfg.mode == "match":
                    search.advance(pick)
                if submitted:
                    break
            if flagged or res.end:
                # A flag, or an unsettled choice: the action under way is not recorded - the game
                # handle goes back to its turn start.
                if turn_start is not None:
                    game.close()
                    game, turn_start = turn_start, None
                if flagged:
                    res.end = "flag"
                    res.winner = 1 - mover
                break
            if clocks:
                res.clock_remaining[mover] += cfg.increment
            res.actions.append(action)
            if turn_start is not None:
                turn_start.close()
                turn_start = None
        snap = game.snapshot()
        res.final_boards = len(snap.boards)
        res.final_fingerprint = snap.fingerprint
        budget = (
            f"clock {cfg.time:g} s + {cfg.increment:g} s per action ({cfg.allocator} allocator, "
            f"signals {'on' if cfg.signals else 'off'}, {'model' if cfg.model_clock else 'wall'} clock)"
            if cfg.time > 0
            else f"{cfg.playouts} playouts per decision"
        )
        termination = res.end + (" (self-play's convention, I11: scored as a draw; not a game rule)" if res.end == "turn_cap" else "")
        termination += UNSETTLED_NOTES.get(res.unsettled, "")
        extra = [
            ("White", "t5d netplay"),
            ("Black", "t5d netplay"),
            (
                "Engine",
                f"t5d search through the C step API: network evaluator (random weights), {budget}, "
                f"{SPLIT_NAMES[cfg.split]}, "
                f"{'self-play, root noise' if cfg.mode == 'self' else 'match play, two networks'}"
                f"{', both sides with the optimistic prior (D9)' if cfg.optimistic else ''}, seed {cfg.seed}",
            ),
            ("TurnCap", f"turns {cfg.cap_turns}, boards {cfg.cap_boards}"),
        ]
        steering = steering_header(cfg)
        if steering is not None:
            extra.append(("Steering", steering))
        if cfg.terminal_nodes != TERMINAL_NODES:  # H22: a non-default budget is named (the review of decision 13, L5)
            extra.append(("TerminalNodes", str(cfg.terminal_nodes)))
        res.record = write_record(
            cfg.variant if cfg.start_fen is None else "custom",
            res.actions,
            even=start.even,
            first_turn=start.present // 2,
            result=res.result,
            termination=termination,
            extra=extra,
            first_player=start.to_move,
            size=(start.width, start.height),
            boards=start_boards,
        )
    finally:
        for s in [shared, *carried]:
            if s is not None:
                s.close()
        for tm in clocks:
            tm.close()
        if turn_start is not None:
            turn_start.close()
        game.close()
    res.seconds = time.perf_counter() - t0
    return res


#: The audit lane's log format (the engine's ``tools/audit.hpp``): JSONL "t5d-decisions" version 1.
DECISIONS_FORMAT, DECISIONS_VERSION = "t5d-decisions", 1


def _decision_entry(game: Game, search: Search, pick: int | RootChild, action: int, decision: int, prefix, even: bool) -> dict:
    info = search.info()
    children = []
    played = [pick] if isinstance(pick, RootChild) else []  # a witness step the tree did not hold (X2)
    for c in [*search.children(), *played]:
        children.append(
            {
                "step": {"moves": [m.text(even) for m in c.moves], "submit": bool(c.submit)},
                "visits": int(c.visits),
                "q": float(c.q),
                "was_pv": bool(c.was_pv),
                "peak_share": float(c.peak_share),
            }
        )
    return {
        "action": int(action),
        "decision": int(decision),
        "prefix": [m.text(even) for m in prefix],
        "boards": len(game.snapshot().boards),
        "root_visits": int(info.root_visits),
        "root_q": float(info.root_q),
        "children": children,
        "chosen": int(pick.index if isinstance(pick, RootChild) else pick),
    }


def decision_log_lines(res: GameResult, *, demotion_share: float = 0.3) -> list[str]:
    """The game's decisions as the engine's offline audit lane reads them (``t5d-audit``; register
    H44): a "game" line - its start (the variant, or the custom start's 5DFEN boards) and actions -
    then an "event" line per final move (the action played, at the decision completing it) and per
    demotion (a child that was the most visited, or held ``demotion_share`` of the root's visits -
    the placeholder the engine's writer uses - and was not chosen). PV changes are not written: the
    C API exposes a root's PV-change list to no binding (the engine's own writer, t5d-play's
    ``--decision-log``, writes them; build-log spec issue 195). The game's id: the record's
    SHA-256, its first 16 hex digits."""
    cfg = res.config
    if not res.record:
        raise ValueError("the game has no record yet")
    record = hashlib.sha256(res.record.encode("utf-8")).hexdigest()[:16]
    game = Game.variant(cfg.variant) if cfg.start_fen is None else Game.from_fen(list(cfg.start_fen))
    try:
        start = game.info()
    finally:
        game.close()
    head = {"format": DECISIONS_FORMAT, "version": DECISIONS_VERSION}
    actions = [[m.text(start.even) for m in a] for a in res.actions]
    lines = [
        json.dumps(
            head
            | {
                "kind": "game",
                "record": record,
                "variant": cfg.variant if cfg.start_fen is None else "custom",
                "start": [] if cfg.start_fen is None else list(cfg.start_fen),
                "width": 0 if cfg.start_fen is None else int(start.width),
                "height": 0 if cfg.start_fen is None else int(start.height),
                "rules": int(start.rules),
                "split": cfg.split,
                "actions": actions,
            }
        )
    ]
    none = {"moves": [], "submit": False}
    for d in res.decision_log:
        if d["action"] >= len(actions):
            continue  # the game stopped during the action
        base = {
            "kind": "event",
            "record": record,
            "action": d["action"],
            "decision": d["decision"],
            "prefix": d["prefix"],
            "from": none,
            "boards": d["boards"],
            "from_visits": 0,
            "from_q": 0.0,
            "root_visits": d["root_visits"],
            "root_q": d["root_q"],
        }
        for i, c in enumerate(d["children"]):
            if i == d["chosen"] or not (c["was_pv"] or c["peak_share"] >= demotion_share):
                continue
            lines.append(
                json.dumps(
                    head
                    | base
                    | {"category": "demotion", "step": c["step"], "visits": c["visits"], "q": c["q"], "peak_share": c["peak_share"]}
                )
            )
        chosen = d["children"][d["chosen"]]
        if cfg.split == "flat" or chosen["step"]["submit"]:
            lines.append(
                json.dumps(
                    head
                    | base
                    | {
                        "category": "final",
                        "step": {"moves": actions[d["action"]], "submit": False},
                        "visits": chosen["visits"],
                        "q": chosen["q"],
                        "peak_share": chosen["peak_share"],
                    }
                )
            )
    return lines


class _OutOfClock(Exception):
    """The side's clock ran out before it had a move (stage E11, the review's H1 and M1)."""


@dataclass
class _DecisionClock:
    """A clocked decision's time after its search (stage E11, the review's H1): its start, its
    hard bound and the side's clock at its start, all on the wall clock (the model clock has no
    wall time: ``model``)."""

    d0: float
    hard: float
    remaining: float
    model: bool

    def left(self, budget: float) -> float:
        """Seconds of ``budget`` from the decision's start still unspent (a tiny positive value
        once spent: a deadline already passed)."""
        return max(budget - (time.perf_counter() - self.d0), 1e-9)

    def out(self) -> bool:
        return not self.model and time.perf_counter() - self.d0 >= self.remaining


#: A clocked decision's batches without a visit before its flag falls, where its clock does not
#: advance (the model clock): the engine's play_game's kNoVisitRounds.
_NO_VISIT_ROUNDS = 16


def _choose(
    search: Search, cfg: PlayConfig, res: GameResult, stats, evaluator, regs, rng, witness=None, clock: _DecisionClock | None = None
) -> int | RootChild | None:
    """The decision's child: the final rule (match) or visit sampling (self-play), then made
    known legal (a chosen unknown child is classified first, within ``cfg.terminal_nodes``;
    H22) - still unknown, the most visited child known legal, and with none the next step of the
    turn-start verdict's witness (``witness(search)``, ``TurnWitness.step``; review decision 13's
    X2) - a child index, or a ``RootChild`` of its own where the root has not instantiated that step.
    None: the chosen move was not shown legal, no other is known legal and the witness does not
    continue - the game ends unscored. A clocked decision (``clock``; the engine's play_game's
    policy, the E11 review's H1): the search re-run one batch at a time; the verification within
    the hard bound, then the most visited child known legal, then the witness; with none of them,
    the children verified in visit order against the side's remaining clock, within the same node
    budget; ``_OutOfClock`` when the clock runs out first - never a throw for want of a move."""

    def visited() -> bool:
        return any(c.visits > 0 for c in search.children())

    def search_more(playouts: int) -> None:
        more, _ = run_search(search, evaluator, playouts=playouts, batch_size=cfg.batch_size, self_play=cfg.mode != "match", **regs)
        stats.batch_sizes.extend(more.batch_sizes)
        stats.evaluate_seconds += more.evaluate_seconds

    def choose_now() -> int:
        if not visited():  # children were removed: search again (a clocked decision: one batch)
            search_more(cfg.batch_size if clock is not None else cfg.playouts)
            rounds = 0
            while clock is not None and not visited():
                # M1: never a throw for want of a visit - on, a batch at a time, until the clock runs out
                before = search.info().playouts
                if clock.out() or rounds >= _NO_VISIT_ROUNDS:
                    raise _OutOfClock
                search_more(cfg.batch_size)
                if not visited() and search.info().playouts == before:
                    raise _OutOfClock
                rounds += 1
        if cfg.mode == "match":
            return search.choose()[0]
        t = cfg.temperature if res.decisions < cfg.temperature_decisions else 0.0
        return search.choose_by_visits(t, float(rng.random()))

    def within(budget: float) -> dict:
        """The verification's deadline: ``budget`` seconds from the decision's start - on a clock
        with wall time only (none unclocked or on the model clock)."""
        return {} if clock is None or clock.model else {"seconds": clock.left(budget)}

    pick = choose_now()
    while search.child(pick).legality != T5D_LEGALITY_LEGAL:
        verdict = search.verify(pick, nodes=cfg.terminal_nodes, **within(clock.hard if clock is not None else 0.0))
        res.verified_choices += 1
        if verdict == T5D_LEGALITY_LEGAL:
            break
        if verdict == T5D_LEGALITY_ILLEGAL:
            pick = choose_now()
            continue
        legal = [c for c in search.children() if c.legality == T5D_LEGALITY_LEGAL]
        if legal:
            pick = max(legal, key=lambda c: c.visits).index
            continue
        step = None if witness is None else witness(search)
        if step is not None:
            res.witness_moves += 1  # a step of the verdict's legal action: known legal
            return step
        if clock is not None and not clock.model:
            return _legal_against_the_clock(search, cfg, res, clock)
        return None  # H22: the game ends unscored
    return pick


def _legal_against_the_clock(search: Search, cfg: PlayConfig, res: GameResult, clock: _DecisionClock) -> int | None:
    """H1's second exception: no child known legal after the bound's verification and the
    witness does not continue - the children verified in visit order against the side's remaining
    clock, within the game loop's node budget (H22), the first shown legal returned;
    ``_OutOfClock`` when the clock runs out first; None (an unsettled choice: the game ends
    unscored) when every verification stopped with time left."""
    tried: set[tuple] = set()
    while True:
        children = sorted(search.children(), key=lambda c: -c.visits)
        removed = False
        for c in children:
            key = (tuple(c.moves), c.submit)
            if key in tried:
                continue
            verdict = search.verify(c.index, seconds=clock.left(clock.remaining), nodes=cfg.terminal_nodes)
            res.verified_choices += 1
            if verdict == T5D_LEGALITY_LEGAL:
                return c.index
            if verdict == T5D_LEGALITY_ILLEGAL:
                removed = True
                break
            if clock.out():
                raise _OutOfClock
            tried.add(key)
        if not removed:
            return None


def _reroot(carried: list[Search | None], mover: int, reply: Sequence[Move], game: Game) -> bool:
    """Match play: the mover's carried search, rooted at the opponent's turn start since its own
    last submission, re-roots on the opponent's reply. Whether it did: a search that cannot (the
    engine refuses the action, or the new root is not the game's position) is dropped - the next
    decision makes a fresh one, and the game counts the fallback (``reroot_fallbacks``)."""
    s = carried[mover]
    try:
        s.advance_action(list(reply))
        with s.root_game() as root:
            same = root.snapshot().fingerprint == game.snapshot().fingerprint and root.info().to_move == game.info().to_move
    except Exception:  # noqa: BLE001 - any refusal: start afresh (and count it)
        same = False
    if not same:
        s.close()
        carried[mover] = None
    return same


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
    last_side: str | None  # the last position's "side": the player to move there, "white" or "black"
    final_boards: tuple[str, ...]  # the last state event's boards, "<L>:<T>:<c>:<placement>"
    lines: tuple[str, ...]
    #: What the gate's reader let pass, as it prints it on standard error (``t5d-gate: warning:
    #: ...``) - a capture mark that disagrees with its move (H51: read as its squares define it,
    #: with a warning), an annotation or a turn number. Never part of the protocol: a record
    #: read with a warning replays to the same events as the record without the slip.
    warnings: tuple[str, ...] = ()

    @property
    def result(self) -> str | None:
        """The 5dpgn result the last position calls for: the side to move checkmated - the other
        side wins; stalemate - ``1/2-1/2``; no game end - ``*`` (None without a position)."""
        if self.last_terminal is None:
            return None
        if self.last_terminal == "checkmate":
            return "0-1" if self.last_side == "white" else "1-0"
        if self.last_terminal == "stalemate":
            return "1/2-1/2"
        return "*"


#: How ``t5d-gate`` starts a line of its reader's warnings on standard error.
_GATE_WARNING = "t5d-gate: warning: "


def replay_with_gate(record: str, *, gate: Path | None = None, timeout: float = 600.0, parse_only: bool = False) -> Replay:
    """Replay ``record`` through the engine's gate adapter in full mode (a subprocess: the
    engine's own tool - legal replay, check status, legal actions and terminal at every
    position), keeping the warnings its reader prints on standard error. ``parse_only``: the
    gate's ``--parse-only`` (the states, no verdict - ``Replay.last_terminal`` None)."""
    gate = gate_executable() if gate is None else gate
    args = [str(gate), "--parse-only"] if parse_only else [str(gate)]
    proc = subprocess.run(args, input=record.encode("utf-8"), capture_output=True, timeout=timeout, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"t5d-gate exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')}")
    warnings = tuple(
        line[len(_GATE_WARNING) :] for line in proc.stderr.decode("utf-8", "replace").splitlines() if line.startswith(_GATE_WARNING)
    )
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
        last_side=positions[-1].get("side") if positions else None,
        final_boards=tuple(states[-1]["boards"]) if states else (),
        lines=lines,
        warnings=warnings,
    )


def replay_problems(res: GameResult, *, gate: Path | None = None, timeout: float = 600.0) -> tuple[list[str], str]:
    """``check_replay`` of the game against the gate's replay of its record, and a note. A game that
    ended on a verdict the loop's budget left unsettled (H22: "unknown", ``unsettled`` "verdict") is
    replayed ``parse_only``: full mode would ask the gate the very question the game gave up on,
    with no budget - minutes at large positions; the states and the final boards are still checked
    (the review of decision 13, X5). A replay past ``timeout`` is a note, never a failure or an
    abort."""
    parse_only = res.end == "unknown" and res.unsettled == "verdict"
    try:
        replay = replay_with_gate(res.record, gate=gate, timeout=timeout, parse_only=parse_only)
    except subprocess.TimeoutExpired:
        return [], f"not replayed: t5d-gate ran past {timeout:g} s"
    return check_replay(res, replay), "parse-only: the final verdict is the one the game left unsettled" if parse_only else ""


def gate_boards(game_snapshot) -> tuple[str, ...]:
    """A snapshot's boards in the gate's state-event form, ``<L>:<T>:<c>:<placement>``."""
    out = []
    for _, _, fen in game_snapshot.boards:
        placement, label, turn, colour = fen[1:-1].split(":")
        out.append(f"{label}:{turn}:{colour}:{placement}")
    return tuple(out)


_RESULT = re.compile(r'^\[Result "([^"]*)"\]', re.MULTILINE)


def check_replay(res: GameResult, replay: Replay) -> list[str]:
    """Where the gate's replay disagrees with the game: every action accepted, one position more
    than actions, the last position's verdict the game's end; the game's result - the record's
    Result header, its closing token and ``GameResult.result``/``winner`` - the one the final
    position calls for (the side to move there checkmated: the other side wins; stalemate a
    draw; otherwise ``*``), judged by the gate and, for the side to move, by the engine's reader
    replaying the record; the final boards the reader's; and nothing let pass - the record is
    written strictly (H51: ``x`` on every move that takes a piece and on no other), so any
    warning of the gate's reader is a problem: since H51 a capture mark that disagrees with its
    move is read with a warning (before it, a superfluous ``x`` made the record unreadable and a
    missing one passed unremarked)."""
    problems = []
    if replay.status != "ok":
        problems.append(f"status {replay.status}")
    problems.extend(f"the gate's reader warned: {w}" for w in replay.warnings)
    if replay.actions_accepted != len(res.actions):
        problems.append(f"{replay.actions_accepted} actions accepted of {len(res.actions)}")
    if replay.positions != len(res.actions) + 1:
        problems.append(f"{replay.positions} positions for {len(res.actions)} actions")
    expected = {"checkmate": "checkmate", "stalemate": "stalemate"}.get(res.end, "none")
    if res.end in ("checkmate", "stalemate", "turn_cap", "action_limit", "flag") and replay.last_terminal != expected:
        problems.append(f"the last position is {replay.last_terminal!r}, the game ended {res.end}")
    with Game.from_pgn(res.record) as g:
        g.replay_all()
        boards = gate_boards(g.snapshot())
        side = ("white", "black")[g.info().to_move]
    if replay.final_boards != boards:
        problems.append("the final boards differ from the record's replay by the engine's reader")
    if replay.last_side is not None and replay.last_side != side:
        problems.append(f"the gate has {replay.last_side} to move at the end, the engine's reader {side}")
    header = _RESULT.search(res.record)
    stated = header.group(1) if header else None
    closing = res.record.split()[-1] if res.record.split() else None
    if stated != res.result or closing != res.result:
        problems.append(f"the record's result (header {stated}, closing {closing}) is not {res.result}")
    if res.end != "unknown" and replay.result is not None and res.result != replay.result:
        problems.append(
            f"the result is {res.result}, the final position ({side} to move, {replay.last_terminal}) calls for {replay.result}"
        )
    if res.end == "checkmate" and res.winner != 1 - ("white", "black").index(side):
        problems.append(f"the winner is {res.winner}, but {side} - to move - is the side checkmated")
    return problems


# --- the command line ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Games through the C step API with the network (random weights) as the evaluator.")
    parser.add_argument("--variant", action="append", help="a named variant (repeatable; default Standard and Two Timelines)")
    parser.add_argument(
        "--split", action="append", choices=sorted(SPLITS), help="D2's arm (repeatable; default ply and flat - ply alone with --optimistic)"
    )
    parser.add_argument("--mode", action="append", choices=MODES, help="self-play or match play (repeatable; default both)")
    parser.add_argument("--games", type=int, default=1, help="games per (variant, split, mode)")
    parser.add_argument("--net", default="small", choices=["small", "F1"])
    parser.add_argument("--playouts", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cap-turns", type=int, default=6)
    parser.add_argument("--cap-boards", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--terminal-nodes", type=int, default=TERMINAL_NODES, help="H22: the game loop's node budget for its verdicts and verifications"
    )
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (the machine is shared)")
    parser.add_argument("--out", type=Path, help="write each record here (a directory)")
    parser.add_argument("--no-replay", action="store_true", help="do not replay the records through t5d-gate")
    parser.add_argument("--optimistic", action="store_true", help="match play: both sides search with D9's optimistic policy")
    parser.add_argument(
        "--piece-centric", action="store_true", help="register A5's piece-centric tokens (a prototype; the evaluator converts each leaf)"
    )
    parser.add_argument("--no-hunt", action="store_true", help="stage E10: no Hunt at the roots (on under 100 boards)")
    parser.add_argument("--no-settling", action="store_true", help="stage E10: no settling scans (on under 50 boards)")
    parser.add_argument("--no-assets", action="store_true", help="stage E10: no asset-map boosts")
    parser.add_argument("--hunt-suggest", action="store_true", help="H43: Hunt's suggested lines raise their root edges (6e)")
    parser.add_argument("--hunt-ms", type=float, default=50.0, help="Hunt's budget per question (6b'': 50 ms)")
    parser.add_argument("--settling-ms", type=float, default=50.0, help="each settling scan's budget (50 ms)")
    parser.add_argument("--verification-threads", type=int, default=5, help="the provers' threads (M13: 5)")
    parser.add_argument("--decision-log", type=Path, help="stage E10, H44: write every game's decisions here for t5d-audit")
    parser.add_argument("--time", type=float, default=0.0, help="stage E11, match play only: each side's clock in seconds (0: none)")
    parser.add_argument("--increment", type=float, default=0.0, help="... and seconds added after each of its actions")
    parser.add_argument("--allocator", choices=sorted(ALLOCATORS), default="integral", help="6o's arm (with --time)")
    parser.add_argument("--no-signals", action="store_true", help="6o: the stop/extend table off (with --time)")
    parser.add_argument("--model-clock", action="store_true", help="the cost model's deterministic clock (with --time)")
    args = parser.parse_args(argv)
    if args.time > 0 and (args.mode or ["self", "match"]) != ["match"]:
        parser.error("--time is match play's clock: add --mode match alone (self-play's budgets are PCR's)")
    if args.optimistic and (args.mode or ["self", "match"]) != ["match"]:
        parser.error("--optimistic is match play only (register D9): add --mode match alone")
    splits = args.split or (["ply"] if args.optimistic else ["ply", "flat"])
    if args.optimistic and "flat" in splits:
        parser.error("--optimistic is refused under the flat form (D9's head has no leave logit; issue 242(h))")
    torch.set_num_threads(args.threads)
    net_cfg = network_config(args.net)
    if args.optimistic:
        net_cfg = replace(net_cfg, optimistic_policy=True)
    if args.piece_centric:
        net_cfg = replace(net_cfg, piece_centric=True)
    nets = [build_model(net_cfg, seed=args.seed + k).eval() for k in (0, 1)]
    policy = "optimistic" if args.optimistic else "normal"
    evaluators = [NetworkEvaluator(n, max_batch=args.batch, policy=policy) for n in nets]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    failed = 0
    for variant in args.variant or ["Standard", "Standard - Two Timelines"]:
        for split in splits:
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
                        terminal_nodes=args.terminal_nodes,
                        optimistic=args.optimistic,
                        hunt=not args.no_hunt,
                        settling=not args.no_settling,
                        asset_map=not args.no_assets,
                        hunt_suggest=args.hunt_suggest,
                        hunt_ms=args.hunt_ms,
                        settling_ms=args.settling_ms,
                        verification_threads=args.verification_threads,
                        decision_log=args.decision_log is not None,
                        time=args.time,
                        increment=args.increment,
                        allocator=args.allocator,
                        signals=not args.no_signals,
                        model_clock=args.model_clock,
                    )
                    with torch.no_grad():
                        res = play_game(evaluators if mode == "match" else evaluators[0], cfg, net_cfg=net_cfg)
                    s = res.summary()
                    if not args.no_replay:
                        problems, note = replay_problems(res)
                        s["replay"] = ("ok" if not problems else "; ".join(problems)) + (f" ({note})" if note else "")
                        failed += bool(problems)
                    if args.out:
                        name = f"{variant.replace(' ', '').replace('-', '_')}_{split}_{mode}_{cfg.seed}.5dpgn"
                        (args.out / name).write_text(res.record, encoding="utf-8")
                        s["record"] = name
                    summaries.append(s)
                    if args.decision_log is not None:
                        with args.decision_log.open("a", encoding="utf-8", newline="\n") as f:
                            f.writelines(line + "\n" for line in decision_log_lines(res))
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
        "terminal_nodes": args.terminal_nodes,
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
    "RESERVED_OVERRIDES",
    "TERMINAL_NODES",
    "UNSETTLED_NOTES",
    "GameResult",
    "PlayConfig",
    "Replay",
    "check_replay",
    "gate_boards",
    "gate_executable",
    "network_config",
    "pgn_move",
    "play_game",
    "replay_problems",
    "replay_with_gate",
    "steering_header",
    "turn_start_verdict",
    "write_record",
]
