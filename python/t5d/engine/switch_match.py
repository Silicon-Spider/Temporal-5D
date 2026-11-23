"""6e's on/off match test's plumbing: one steering switch against itself (H13, H43).

Spec section 11.2 gives each hand-written steering term "an on/off test of its own at match play
(6e)"; section 13.1's 6e: "each steering term on and off at match play - the asset map, Hunt's
suggested lines". ``play_switch_match`` plays that match: for each start, two games with one
network on both sides - the switch ON for White and OFF for Black, then the colours swapped - every
other setting the base config's for both sides (``PlayConfig.side_overrides``). It tallies from the
ON side's view and reports, per arm, the root edges the switch raised (``GameResult``'s counters),
so a verdict can be told from an arm that never acted.

What is not here, by the plan's recommended answer (the handover of 2026-11-15, open question 4):
the match's openings, its size and its pass rule. Those are pre-registered in section 15.6's style
before a trained network exists; with random weights a run of this module is plumbing, not a
section 15.3 or 15.6 figure. I11's turn cap is scored as a draw (self-play's convention) and
counted separately; a game ended otherwise than by checkmate, stalemate, a flag or the cap is
excluded and counted. At match play the tree is carried (C12), and Hunt's suggestions do not reach
a turn-start root the carried tree had already expanded (build-log spec issue 477):
``carried_on`` reports how many of the ON side's turn starts that was.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from ..net.config import NetConfig
from ..net.model import build_model
from .evaluator import Evaluator, NetworkEvaluator
from .netplay import GameResult, PlayConfig, check_replay, network_config, play_game, replay_with_gate

#: 6e's terms: the switch's name, its ``t5d_search_params`` field and the ``GameResult`` counter
#: that shows the arm acted.
SWITCHES = {
    "asset_map": ("asset_map", "asset_boosted_edges"),
    "hunt_suggest": ("hunt_suggest", "suggestion_edges"),
}

#: A start: a variant's name, or a tuple of 5DFEN board strings.
Start = str | tuple[str, ...]


@dataclass
class SwitchMatch:
    """A switch match's games and its tally from the ON side's view."""

    switch: str
    games: list[GameResult] = field(default_factory=list)
    on_side: list[int] = field(default_factory=list)  # per game: 0 White, 1 Black
    wins: int = 0
    draws: int = 0  # stalemates
    capped: int = 0  # I11's turn cap: scored as a draw, counted apart
    losses: int = 0
    unscored: int = 0  # any other end: excluded, counted
    edges_on: int = 0  # the root edges the switch raised on the ON side
    edges_off: int = 0  # ... on the OFF side (0: the switch off is off)
    turn_starts_on: int = 0
    carried_on: int = 0  # the ON side's turn-start roots the carried tree had already visited

    @property
    def score(self) -> float:
        scored = self.wins + self.draws + self.capped + self.losses
        return (self.wins + 0.5 * (self.draws + self.capped)) / scored if scored else math.nan

    def summary(self) -> dict:
        return {
            "switch": self.switch,
            "games": len(self.games),
            "on_side": list(self.on_side),
            "wins": self.wins,
            "draws": self.draws,
            "capped": self.capped,
            "losses": self.losses,
            "unscored": self.unscored,
            "score": self.score,
            "edges_on": self.edges_on,
            "edges_off": self.edges_off,
            "turn_starts_on": self.turn_starts_on,
            "carried_on": self.carried_on,
        }


def _tally(match: SwitchMatch, res: GameResult, on: int, counter: str) -> None:
    if res.winner is not None:  # checkmate, or a flag (the match clock)
        if res.winner == on:
            match.wins += 1
        else:
            match.losses += 1
    elif res.end == "stalemate":
        match.draws += 1
    elif res.end == "turn_cap":
        match.capped += 1
    else:
        match.unscored += 1
    match.edges_on += getattr(res, counter)[on]
    match.edges_off += getattr(res, counter)[1 - on]
    match.turn_starts_on += res.turn_start_decisions[on]
    match.carried_on += res.carried_turn_starts[on]


def play_switch_match(
    evaluator: Evaluator,
    base: PlayConfig,
    switch: str,
    *,
    starts: Sequence[Start] = ("Standard",),
    pairs: int = 1,
    net_cfg: NetConfig | None = None,
) -> SwitchMatch:
    """Each start ``pairs`` times, two games a time: ``switch`` ON for White and OFF for Black, then
    the colours swapped; one ``evaluator`` for both sides (``play_game`` duplicates it), so the
    switch is the only difference. Game p of a start is seeded ``base.seed + 101 * p``."""
    if switch not in SWITCHES:
        raise ValueError(f"switch must be one of {sorted(SWITCHES)}")
    name, counter = SWITCHES[switch]
    if any(base.side_overrides):
        raise ValueError("play_switch_match sets the side overrides itself: give a base without them")
    if any(entry[0] == name for entry in base.search_overrides):
        raise ValueError(f"the base's search_overrides name {name!r}: the switch is the match's")
    match = SwitchMatch(switch=switch)
    for start in starts:
        for p in range(pairs):
            for on in (0, 1):
                sides = [((name, 0),), ((name, 0),)]
                sides[on] = ((name, 1),)
                where = {"variant": start, "start_fen": None} if isinstance(start, str) else {"start_fen": tuple(start)}
                cfg = replace(base, mode="match", seed=base.seed + 101 * p, side_overrides=(sides[0], sides[1]), **where)
                res = play_game(evaluator, cfg, net_cfg=net_cfg)
                match.games.append(res)
                match.on_side.append(on)
                _tally(match, res, on, counter)
    return match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="6e's on/off match test: one steering switch against itself (plumbing)")
    parser.add_argument("--switch", required=True, choices=sorted(SWITCHES))
    parser.add_argument("--variant", action="append", help="a start (repeatable; default Standard and Standard - Two Timelines)")
    parser.add_argument("--pairs", type=int, default=1, help="games per start and colour")
    parser.add_argument("--net", choices=["small", "F1"], default="small")
    parser.add_argument("--playouts", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cap-turns", type=int, default=6)
    parser.add_argument("--cap-boards", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (the machine is shared)")
    parser.add_argument("--split", choices=["ply", "tree", "flat"], default="ply")
    parser.add_argument("--no-hunt", action="store_true")
    parser.add_argument("--no-settling", action="store_true")
    parser.add_argument("--no-assets", action="store_true", help="the asset map off on both sides (the other switch's match)")
    parser.add_argument("--hunt-ms", type=float, default=50.0)
    parser.add_argument("--settling-ms", type=float, default=50.0)
    parser.add_argument("--verification-threads", type=int, default=5)
    parser.add_argument("--out", type=Path, help="write each record here (a directory)")
    parser.add_argument("--no-replay", action="store_true", help="do not replay the records through t5d-gate")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    net_cfg = network_config(args.net)
    evaluator = NetworkEvaluator(build_model(net_cfg, seed=args.seed).eval(), max_batch=args.batch)
    base = PlayConfig(
        split=args.split,
        mode="match",
        playouts=args.playouts,
        batch_size=args.batch,
        cap_turns=args.cap_turns,
        cap_boards=args.cap_boards,
        seed=args.seed,
        hunt=not args.no_hunt,
        settling=not args.no_settling,
        asset_map=not args.no_assets,
        hunt_ms=args.hunt_ms,
        settling_ms=args.settling_ms,
        verification_threads=args.verification_threads,
    )
    starts = args.variant or ["Standard", "Standard - Two Timelines"]
    with torch.no_grad():
        match = play_switch_match(evaluator, base, args.switch, starts=starts, pairs=args.pairs, net_cfg=net_cfg)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    name, counter = SWITCHES[args.switch]
    for k, (res, on) in enumerate(zip(match.games, match.on_side, strict=True)):
        line = {"game": k, "variant": res.config.variant, "seed": res.config.seed, "on_side": on, "end": res.end, "winner": res.winner}
        line[counter] = getattr(res, counter)
        if not args.no_replay:
            problems = check_replay(res, replay_with_gate(res.record))
            line["replay"] = "ok" if not problems else "; ".join(problems)
            failed += bool(problems)
        if args.out:
            record = f"{args.switch}_{k}_{res.config.variant.replace(' ', '').replace('-', '_')}.5dpgn"
            (args.out / record).write_text(res.record, encoding="utf-8")
            line["record"] = record
        print(json.dumps(line), flush=True)
    conditions = {
        "net": args.net,
        "net_config": {k: v for k, v in asdict(net_cfg).items() if k in ("d_model", "n_layers", "n_heads", "ffn_ratio")},
        "weights": "random (plumbing: not a section 15.3 or 15.6 figure)",
        "starts": starts,
        "pairs": args.pairs,
        "playouts": args.playouts,
        "split": args.split,
        "cap_turns": args.cap_turns,
        "cap_boards": args.cap_boards,
        "torch_threads": args.threads,
    }
    print(json.dumps({"conditions": conditions, "totals": match.summary(), "replays_failed": failed}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SWITCHES", "SwitchMatch", "main", "play_switch_match"]
