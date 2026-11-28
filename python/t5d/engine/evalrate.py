"""F1's CPU evaluation rate: self-play games through the engine with the v1 network at random weights.

Section 15.6 cites "6.9 evaluations/s for F1" - the only evaluation rate the build has. It comes
from stage N5's end-to-end run (CHANGELOG, 2026-10-04: 20 games, F1 in BF16, 0.60 decisions/s and
6.9 evaluations/s), whose command, games, variants and threads were not recorded. This module is
that figure's reproducible command, for the quiet-machine block: a fixed number of **self-play**
games (``netplay.play_game``: the engine's search through the C step API, the network evaluating
its leaves - nothing is trained), with fixed seeds, variants, splits and threads, every one of
them printed before the first game.

**The provers are off by default** - no Hunt at the roots, no settling scans, no asset map
(``PlayConfig.hunt``, ``settling``, ``asset_map``; stage E10's verification ladder, which did not
exist at N5) - so that the figure is the network's own rate within the search; ``--provers`` turns
all three on (their own defaults: Hunt under 100 boards, the settling scans under 50, 50 ms each,
5 threads). The game loop's own exact questions (a turn start's verdict, a chosen move's
verification: register H22's node budget) stay: they are the game's, not a prover's.

Reported, per game (``GameResult.summary``, and the evaluator's counts) and in total:

* ``decisions_per_s`` and ``evaluations_per_s``: the search's decisions and the leaves the network
  evaluated (the cache's hits not counted) over the games' whole wall time - N5's two figures;
* ``forwards`` (the network's forward passes: ``NetworkEvaluator`` runs one per batch of at most
  ``--batch`` positions), ``positions`` (the positions they evaluated) and
  ``positions_per_forward``; ``forwards_per_s``;
* ``network_seconds`` (the evaluator's calls: the fit, the collation, the forward, the split,
  and the serve loop's check of each evaluation), ``network_share`` of the wall time, and
  ``network_positions_per_s`` - positions over the network's own seconds, the rate with the
  search and the rules engine taken out;
* **where an evaluation's time goes** (``assembly_breakdown``; section 15.3's row "Input assembly
  and host-to-device transfer per evaluation", its CPU half): the seconds of each part -
  ``convert_seconds`` (the step driver's: each leaf's expansion copied out of the engine and
  converted to the network's input, ``CStepDriver.convert_seconds``; it is inside the game's
  ``collect_seconds``, not its ``network_seconds``), ``fit_seconds``, ``collate_seconds``,
  ``transfer_seconds`` (nothing on the CPU), ``forward_seconds`` and ``split_seconds``
  (``NetworkEvaluator``'s counters) - then ``assembly_seconds`` (conversion, fit and collation),
  ``assembly_ms_per_position`` and ``assembly_share`` of the six parts' sum, and the forward's
  ``forward_ms_per_position`` and ``forward_share``;
* ``by_range`` (totals only): the network's calls by board-count range - 6b''s ranges, < 50,
  50-99, 100-199, 200+ - each call counted in the range of its largest position (the padded
  batch's size sets its cost): forwards, positions, seconds, positions per second and per
  forward, and the same breakdown per range (a batch's conversion is filed with its call).
  Section 15.6 reads an evaluation rate per board-count range: the pooled figure is
  the games' mix of sizes (a 2-turn Standard game stays under 16 boards; the caps allow 80).

**Assembly alone, without a forward**: ``python -m t5d.engine.bench --assembly`` times the
conversion and the collation of positions in each board-count range in minutes (this module's
games take an hour or more with F1 on the CPU).

Information, never a verdict: a CPU figure (section 15.6's Phase-3 re-check waits for a GPU rate).
The first line is the configuration (``{"conditions": ...}``: the network's every field and its
parameter count, the game settings, the seeds, PyTorch's threads, the machine, the engine
library's file name and the commit), then one line per game, then ``{"totals": ...}``.

Run from ``python/``: ``python -m t5d.engine.evalrate`` (the defaults: F1, one game for each of
Standard and Standard - Two Timelines under the ply split, 16 playouts and batches of 8 a
decision, I11's caps at 6 turns or 80 boards - netplay's settings, N5's - seeds 1 + 101 g, 4
PyTorch threads, the provers off). The F1 forward is slow on the CPU at the caps' sizes (on a
shared machine a 2-turn Standard game took 70 s at 2.5 evaluations/s; a 6-turn game ran past 50
minutes and was stopped), so ``--games`` stays small by default and ``--cap-turns`` bounds a run.
Never through ``oracles/``: the engine is the only rules source (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ..net.config import NetConfig
from ..net.model import build_model
from .capi import find_library
from .evaluator import NetworkEvaluator, PositionLike, state_of
from .netplay import SPLITS, TERMINAL_NODES, GameResult, PlayConfig, network_config, play_game

#: The defaults (stated in the output with every other setting): netplay's variants, D2's default
#: arm, one game each, netplay's seed scheme, its search and cap settings, and its 4 threads.
VARIANTS: tuple[str, ...] = ("Standard", "Standard - Two Timelines")
SPLIT_NAMES: tuple[str, ...] = ("ply",)
GAMES = 1
SEED = 1
THREADS = 4
#: The board-count ranges a rate is read by (6b''s: section 15.6 reads a rate per range).
RANGES: tuple[tuple[str, int], ...] = (("< 50", 50), ("50-99", 100), ("100-199", 200), ("200+", 1 << 30))


def range_of(boards: int) -> str:
    """The name of the range ``boards`` falls in."""
    return next(name for name, below in RANGES if boards < below)


#: The parts of an evaluation's time, in the order they happen (``assembly_breakdown``): the
#: driver's conversion, then the evaluator's own five (``NetworkEvaluator``'s counters).
PARTS: tuple[str, ...] = ("convert", "fit", "collate", "transfer", "forward", "split")
#: Input assembly (section 15.3's row, its CPU half): everything that makes the network's input
#: from the engine's leaf - the conversion, the fit to the network's layout and the collation.
ASSEMBLY: tuple[str, ...] = ("convert", "fit", "collate")


def assembly_breakdown(seconds: dict[str, float], positions: int) -> dict:
    """Where an evaluation's time goes, from the summed seconds of each part (``PARTS``) and the
    positions evaluated: the parts' seconds; ``assembly_seconds`` (conversion, fit and collation
    - section 15.3's "input assembly"), its milliseconds per position and its share of the parts'
    sum; the forward's the same; and the transfer's seconds (nothing on the CPU: the batch is
    already on the evaluator's device - the row's host-to-device half waits for a GPU). Rates of
    the sums, never means of per-call rates."""
    total = sum(seconds[p] for p in PARTS)
    assembly = sum(seconds[p] for p in ASSEMBLY)
    return {
        "convert_seconds": round(seconds["convert"], 4),
        "fit_seconds": round(seconds["fit"], 4),
        "collate_seconds": round(seconds["collate"], 4),
        "transfer_seconds": round(seconds["transfer"], 4),
        "forward_seconds": round(seconds["forward"], 4),
        "split_seconds": round(seconds["split"], 4),
        "assembly_seconds": round(assembly, 4),
        "assembly_ms_per_position": _ratio(1e3 * assembly, positions, 4),
        "assembly_share": _ratio(assembly, total, 4),
        "forward_ms_per_position": _ratio(1e3 * seconds["forward"], positions, 4),
        "forward_share": _ratio(seconds["forward"], total, 4),
    }


class RangedEvaluator(NetworkEvaluator):
    """``NetworkEvaluator`` that also times each call and files it by board-count range - the
    range of its largest position (the padded batch's size sets the call's cost) - with the call's
    parts (``PARTS``): the evaluator's own fit, collation, transfer, forward and split, and the
    conversion of the batch's positions, which the serve loop reports before the call
    (``note_conversion``; ``evaluator.serve``). ``convert_seconds`` is the conversions' sum over
    the evaluator's life, beside ``NetworkEvaluator``'s five."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convert_seconds = 0.0
        self.ranges: dict[str, dict] = {
            name: {"forwards": 0, "positions": 0, "seconds": 0.0, "parts": dict.fromkeys(PARTS, 0.0)} for name, _ in RANGES
        }

    def parts(self) -> dict[str, float]:
        """The seconds of each part so far (``PARTS``)."""
        return {
            "convert": self.convert_seconds,
            "fit": self.fit_seconds,
            "collate": self.collate_seconds,
            "transfer": self.transfer_seconds,
            "forward": self.forward_seconds,
            "split": self.split_seconds,
        }

    def note_conversion(self, positions: Sequence[PositionLike], seconds: float) -> None:
        """The seconds the driver spent converting ``positions`` (the batch about to be evaluated),
        filed in the range of its largest position, as the call will be."""
        boards = max((int(state_of(p).n_boards) for p in positions), default=0)
        self.convert_seconds += seconds
        self.ranges[range_of(boards)]["parts"]["convert"] += seconds

    def __call__(self, positions: Sequence[PositionLike]) -> list:
        boards = max((int(state_of(p).n_boards) for p in positions), default=0)
        f0 = self.batches
        before = self.parts()
        start = time.perf_counter()
        out = super().__call__(positions)
        r = self.ranges[range_of(boards)]
        r["seconds"] += time.perf_counter() - start
        r["forwards"] += self.batches - f0
        r["positions"] += len(positions)
        after = self.parts()
        for part in PARTS[1:]:  # the conversion is filed when it is reported
            r["parts"][part] += after[part] - before[part]
        return out

    def by_range(self) -> dict:
        """The ranges with any call: their counts, seconds and rates, and where the time went
        (``assembly_breakdown``; ``seconds`` is the evaluator's calls, the conversion not in it)."""
        return {
            name: {
                "forwards": r["forwards"],
                "positions": r["positions"],
                "seconds": round(r["seconds"], 3),
                "positions_per_s": _ratio(r["positions"], r["seconds"], 2),
                "positions_per_forward": _ratio(r["positions"], r["forwards"], 2),
            }
            | assembly_breakdown(r["parts"], r["positions"])
            for name, r in self.ranges.items()
            if r["forwards"]
        }


def game_seeds(seed: int, games: int) -> list[int]:
    """The games' seeds: ``seed + 101 g`` for game g (``netplay``'s scheme)."""
    return [seed + 101 * g for g in range(games)]


def play_config(variant: str, split: str, seed: int, *, provers: bool = False, **settings) -> PlayConfig:
    """A self-play game's settings: ``settings`` (``PlayConfig``'s fields) with the provers - Hunt,
    the settling scans, the asset map - all off, or all at their defaults with ``provers``."""
    return PlayConfig(variant=variant, split=split, mode="self", seed=seed, hunt=provers, settling=provers, asset_map=provers, **settings)


@dataclass
class GameFigures:
    """One game's figures: its result and what the evaluator counted during it - its forwards
    and positions, and the seconds of each part of its evaluations (``PARTS``; all 0 for an
    evaluator that keeps none)."""

    result: GameResult
    forwards: int
    positions: int
    parts: dict[str, float] = field(default_factory=lambda: dict.fromkeys(PARTS, 0.0))

    def row(self) -> dict:
        r = self.result.summary()
        r |= {"forwards": self.forwards, "positions": self.positions, "positions_per_forward": _ratio(self.positions, self.forwards, 2)}
        r |= assembly_breakdown(self.parts, self.positions)
        return r


def _ratio(a: float, b: float, digits: int) -> float:
    return round(a / b, digits) if b else 0.0


def _parts(evaluator: NetworkEvaluator) -> dict[str, float]:
    """The evaluator's seconds by part so far (``PARTS``): ``RangedEvaluator.parts``, or the
    plain evaluator's five with no conversion (it is told of none)."""
    parts = getattr(evaluator, "parts", None)
    if callable(parts):
        return dict(parts())
    return {
        "convert": 0.0,
        "fit": evaluator.fit_seconds,
        "collate": evaluator.collate_seconds,
        "transfer": evaluator.transfer_seconds,
        "forward": evaluator.forward_seconds,
        "split": evaluator.split_seconds,
    }


def totals(games: Sequence[GameFigures]) -> dict:
    """The figures over every game: counts summed, rates of the sums (never a mean of the games'
    rates, which would weigh a short game like a long one)."""
    seconds = sum(g.result.seconds for g in games)
    network = sum(g.result.network_seconds for g in games)
    decisions = sum(g.result.decisions for g in games)
    evaluations = sum(g.result.evaluations for g in games)
    forwards = sum(g.forwards for g in games)
    positions = sum(g.positions for g in games)
    parts = {p: sum(g.parts[p] for g in games) for p in PARTS}
    return {
        "games": len(games),
        "ends": dict(Counter(g.result.end for g in games)),
        "decisions": decisions,
        "evaluations": evaluations,
        "forwards": forwards,
        "positions": positions,
        "positions_per_forward": _ratio(positions, forwards, 2),
        "seconds": round(seconds, 3),
        "network_seconds": round(network, 3),
        "decisions_per_s": _ratio(decisions, seconds, 3),
        "evaluations_per_s": _ratio(evaluations, seconds, 2),
        "forwards_per_s": _ratio(forwards, seconds, 2),
        "network_positions_per_s": _ratio(positions, network, 2),
        "network_share": _ratio(network, seconds, 3),
    } | assembly_breakdown(parts, positions)


def play(
    evaluator: NetworkEvaluator,
    net_cfg: NetConfig,
    *,
    variants: Sequence[str] = VARIANTS,
    splits: Sequence[str] = SPLIT_NAMES,
    games: int = GAMES,
    seed: int = SEED,
    provers: bool = False,
    on_game: Callable[[GameFigures], None] | None = None,
    **settings,
) -> list[GameFigures]:
    """Every game, in order: for each variant, each split, each seed (``game_seeds``)."""
    out = []
    for variant in variants:
        for split in splits:
            for s in game_seeds(seed, games):
                cfg = play_config(variant, split, s, provers=provers, **settings)
                f0, p0, parts0 = evaluator.batches, evaluator.positions, _parts(evaluator)
                with torch.no_grad():
                    result = play_game(evaluator, cfg, net_cfg=net_cfg)
                parts1 = _parts(evaluator)
                game = GameFigures(result, evaluator.batches - f0, evaluator.positions - p0, {p: parts1[p] - parts0[p] for p in PARTS})
                out.append(game)
                if on_game is not None:
                    on_game(game)
    return out


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=30)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--net", default="F1", choices=["F1", "small"], help="the network (F1: the v1 as shipped, BF16 trunk)")
    parser.add_argument("--variant", action="append", help=f"a named variant (repeatable; default {' and '.join(VARIANTS)})")
    parser.add_argument("--split", action="append", choices=sorted(SPLITS), help="D2's arm (repeatable; default ply)")
    parser.add_argument("--games", type=int, default=GAMES, help=f"games per (variant, split) ({GAMES})")
    parser.add_argument("--seed", type=int, default=SEED, help="game g's seed is SEED + 101 g; the network's is SEED")
    parser.add_argument("--playouts", type=int, default=16, help="per decision (16)")
    parser.add_argument("--batch", type=int, default=8, help="leaves per network call (8)")
    parser.add_argument("--cap-turns", type=int, default=6, help="I11's turn cap (6)")
    parser.add_argument("--cap-boards", type=int, default=80, help="I11's board cap (80)")
    parser.add_argument("--terminal-nodes", type=int, default=TERMINAL_NODES, help="H22: the game loop's node budget")
    parser.add_argument("--threads", type=int, default=THREADS, help=f"PyTorch's CPU threads ({THREADS})")
    parser.add_argument("--provers", action="store_true", help="Hunt, the settling scans and the asset map on (off by default)")
    args = parser.parse_args(argv)
    if args.games < 1 or args.threads < 1:
        parser.error("--games and --threads must be at least 1")
    variants = args.variant or list(VARIANTS)
    splits = args.split or list(SPLIT_NAMES)
    settings = dict(
        playouts=args.playouts,
        batch_size=args.batch,
        cap_turns=args.cap_turns,
        cap_boards=args.cap_boards,
        terminal_nodes=args.terminal_nodes,
    )
    template = play_config(variants[0], splits[0], args.seed, provers=args.provers, **settings)
    template.validate()

    torch.set_num_threads(args.threads)
    net_cfg = network_config(args.net)
    model = build_model(net_cfg, seed=args.seed).eval()
    evaluator = RangedEvaluator(model, max_batch=args.batch)
    conditions = {
        "tool": "t5d.engine.evalrate",
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "net": args.net,
        "net_seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "net_config": asdict(net_cfg),
        "provers": "on: Hunt, the settling scans and the asset map at their defaults"
        if args.provers
        else "off: no Hunt, no settling scans, no asset map",
        "variants": variants,
        "splits": splits,
        "games_per_variant_split": args.games,
        "seeds": game_seeds(args.seed, args.games),
        "play": {k: v for k, v in asdict(template).items() if k not in ("variant", "split", "seed")},
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_processors": os.cpu_count(),
        "engine_library": find_library().name,
        "commit": _commit(),
    }
    print(json.dumps({"conditions": conditions}), flush=True)
    games = play(
        evaluator,
        net_cfg,
        variants=variants,
        splits=splits,
        games=args.games,
        seed=args.seed,
        provers=args.provers,
        on_game=lambda g: print(json.dumps(g.row()), flush=True),
        **settings,
    )
    print(json.dumps({"totals": totals(games) | {"by_range": evaluator.by_range()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
