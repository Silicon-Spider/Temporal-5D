"""Conversion and forward-pass timings on real engine positions (stage N4).

Two figures, each with its conditions printed alongside:

* **Conversion, per position**: ``expand`` (the C call with every board's features, and
  the copy out of the engine's buffers), ``to_position`` (the ``PositionState``, validated,
  and the W/E scope), and ``Game.position`` end to end (both, plus the creator count) - the
  median and the 90th percentile over every position and repeat of a size class.
* **Network forward, per batch**: ``NetworkEvaluator`` over a batch of B positions of a
  size class - ``collate``, one dense forward of the model, the split into per-position
  policy logits and value logits - and the share of it spent in ``collate``; the median over
  repeats after one warm-up batch. With ``--piece-centric`` (register A5, a prototype) the
  network reads piece-centric tokens: the evaluator converts each position at the network's
  edge (``state.to_piece_centric``), and the conversion's own time is reported beside
  ``collate``'s (the tokens column is then A5's).

Positions are seeded random pseudo-legal play from the Standard start through the binding
(``play.py``; travel moves drawn with probability ``--travel-prob``), taken when the game
first reaches each board count - real engine positions, but random play: for costs only.

Run from ``python/``: ``py -m t5d.engine.bench`` (defaults: conversion at 1-240 boards,
forward at 1-60 boards, batches of 1 and 4, F1 as shipped - BF16 trunk, dense). These are
development-machine CPU figures, not section 15.3's cost measurements.

**Input assembly alone, by board-count range** (``--assembly``; section 15.3's row "Input
assembly and host-to-device transfer per evaluation", its CPU half - no model is built and no
forward runs, so it takes minutes): for positions at ``--assembly-boards`` board counts, filed in
6b''s ranges (< 50, 50-99, 100-199, 200+), what making the network's input from an engine
position costs per position -

* ``expand``: the engine's expansion and its copy out of the engine's buffers (the search pays
  the expansion itself in its gather; the copy is the part the step driver's conversion adds -
  the two cannot be told apart from here, so the figure is given beside the assembly, not in it);
* ``to_position``: the copy made the network's input, as the step driver makes it
  (``CStepDriver``: the network's register counts, no W/E scope);
* ``fit``: the evaluator's ``_fit`` - register A5's conversion with ``--piece-centric``, nothing
  otherwise (the positions are made with the network's register counts);
* ``collate``: the padded batch's tensors, per position, for each ``--batch`` size (a batch is
  positions of one board count: the mixed batches of a search pad more);
* **assembly** = ``to_position`` + ``fit`` + ``collate``, per position, for each batch size.

Per range: the median and the 95th percentile over every (position, repeat) of the range's board
counts. The host-to-device transfer is not here: on the CPU there is none to time (the row's
other half waits for a GPU). ``evalrate`` reports the same parts inside real games.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ..net import NetConfig, build_model, collate, to_piece_centric
from .capi import load_library
from .convert import to_position
from .evalrate import RANGES, range_of
from .evaluator import NetworkEvaluator
from .game import Game
from .play import random_game

#: ``--assembly``'s default board counts: two or three in each of 6b''s ranges.
ASSEMBLY_BOARDS: tuple[int, ...] = (10, 30, 44, 60, 80, 120, 160, 240)


def _positions(boards: list[int], seeds: int, travel_prob: float) -> dict[int, list]:
    """Game copies at the first position of each seed's random play with at least N boards
    (N = 1: the Standard start itself)."""
    out: dict[int, list] = {n: [] for n in boards}
    for seed in range(seeds):
        wanted = sorted(n for n in boards if n > 1)
        if 1 in out:
            out[1].append(Game.variant("Standard"))
        if not wanted:
            continue
        for game in random_game("Standard", seed, target_boards=max(wanted), max_actions=5000, travel_prob=travel_prob):
            n = game.info().boards
            while wanted and n >= wanted[0]:
                out[wanted.pop(0)].append(game.copy())
    return out


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def conversion(games: dict[int, list], repeat: int) -> list[dict]:
    rows = []
    for n, gs in games.items():
        expand_t, convert_t, total_t = [], [], []
        shapes = []
        for g in gs:
            created = g.created_timelines()
            for _ in range(repeat):
                t0 = time.perf_counter()
                x = g.expand()
                t1 = time.perf_counter()
                pos = to_position(x, created_timelines=created)
                t2 = time.perf_counter()
                g.position()
                t3 = time.perf_counter()
                expand_t.append(t1 - t0)
                convert_t.append(t2 - t1)
                total_t.append(t3 - t2)
            info = g.info()
            shapes.append((info.boards, info.timelines, info.active_timelines, pos.move_count, int(pos.state.edges.shape[0])))
        rows.append(
            {
                "boards": n,
                "positions": len(gs),
                "actual boards": "/".join(str(s[0]) for s in shapes),
                "timelines (active)": "/".join(f"{s[1]}({s[2]})" for s in shapes),
                "moves": "/".join(str(s[3]) for s in shapes),
                "expand ms": 1e3 * statistics.median(expand_t),
                "to_position ms": 1e3 * statistics.median(convert_t),
                "position ms": 1e3 * statistics.median(total_t),
                "position p90 ms": 1e3 * _percentile(total_t, 90),
            }
        )
    return rows


def forward(games: dict[int, list], batches: list[int], repeat: int, cfg: NetConfig) -> list[dict]:
    model = build_model(cfg, seed=0).eval()
    evaluator = NetworkEvaluator(model, max_batch=max(batches))
    rows = []
    for n, gs in games.items():
        positions = [g.position(regs_per_board=cfg.regs_per_board, global_regs=cfg.global_regs) for g in gs]
        for b in batches:
            batch = [positions[i % len(positions)] for i in range(b)]
            evaluator(batch)  # warm-up
            times, collate_t, convert_t = [], [], []
            for _ in range(repeat):
                t0 = time.perf_counter()
                states = [to_piece_centric(p.state) for p in batch] if cfg.piece_centric else [p.state for p in batch]
                tc = time.perf_counter()
                collate(states)
                t1 = time.perf_counter()
                evaluator(batch)
                t2 = time.perf_counter()
                convert_t.append(tc - t0)
                collate_t.append(t1 - tc)
                times.append(t2 - t1)
            tokens = sum(s.n_tokens for s in states)
            med = statistics.median(times)
            row = {
                "boards": n,
                "batch": b,
                "tokens": tokens,
                "batch s": med,
                "per position s": med / b,
                "collate ms": 1e3 * statistics.median(collate_t),
            }
            if cfg.piece_centric:  # A5: the conversion at the network's edge, inside the evaluator's time
                row["A5 convert ms"] = 1e3 * statistics.median(convert_t)
            row["moves scored"] = sum(p.move_count for p in batch)
            rows.append(row)
    return rows


@dataclass
class AssemblySample:
    """One timing of one position's input assembly (seconds per position): the parts the module
    docstring names. ``collate`` maps a batch size to the batch's collation time divided by it."""

    boards: int
    expand: float
    convert: float
    fit: float
    collate: dict[int, float] = field(default_factory=dict)

    def assembly(self, batch: int) -> float:
        """``to_position`` + ``fit`` + ``collate`` at this batch size, per position."""
        return self.convert + self.fit + self.collate[batch]


def assembly_samples(games: dict[int, list], batches: Sequence[int], repeat: int, cfg: NetConfig) -> list[AssemblySample]:
    """Every (position, repeat)'s timings, with the engine: ``expand``, ``to_position`` as the step
    driver calls it, the evaluator's fit (A5's conversion for a piece-centric ``cfg``) and
    ``collate`` over ``b`` copies of the fitted state for each batch size - after one untimed pass
    per position (the first call pays imports and allocator warm-up)."""
    out = []
    for n, gs in games.items():
        for g in gs:
            boards = g.info().boards
            for r in range(repeat + 1):
                t0 = time.perf_counter()
                x = g.expand()
                t1 = time.perf_counter()
                pos = to_position(x, regs_per_board=cfg.regs_per_board, global_regs=cfg.global_regs, scope=False)
                t2 = time.perf_counter()
                state = to_piece_centric(pos.state, strict=True) if cfg.piece_centric else pos.state
                t3 = time.perf_counter()
                sample = AssemblySample(boards=boards, expand=t1 - t0, convert=t2 - t1, fit=t3 - t2)
                for b in batches:
                    states = [state] * b
                    c0 = time.perf_counter()
                    collate(states)
                    sample.collate[b] = (time.perf_counter() - c0) / b
                if r > 0:  # the first pass is the warm-up
                    out.append(sample)
            if boards < n:
                raise RuntimeError(f"a position taken for {n} boards has {boards}")
    return out


def assembly_rows(samples: Sequence[AssemblySample], batches: Sequence[int]) -> list[dict]:
    """The samples by board-count range (6b''s; a range with none gives no row): how many, the
    board counts met, each part's median in milliseconds per position, and the assembly's median
    and 95th percentile for each batch size. Each figure is over the range's samples; the
    assembly's is the median of the samples' sums, not the sum of the parts' medians."""
    rows = []
    for name, _ in RANGES:
        mine = [s for s in samples if range_of(s.boards) == name]
        if not mine:
            continue
        row: dict = {
            "range": name,
            "samples": len(mine),
            "boards": "/".join(str(n) for n in sorted({s.boards for s in mine})),
            "expand ms": 1e3 * statistics.median(s.expand for s in mine),
            "to_position ms": 1e3 * statistics.median(s.convert for s in mine),
            "fit ms": 1e3 * statistics.median(s.fit for s in mine),
        }
        for b in batches:
            row[f"collate ms/pos (B={b})"] = 1e3 * statistics.median(s.collate[b] for s in mine)
        for b in batches:
            sums = [s.assembly(b) for s in mine]
            row[f"assembly ms/pos (B={b})"] = 1e3 * statistics.median(sums)
            row[f"assembly p95 (B={b})"] = 1e3 * _percentile(sums, 95)
        rows.append(row)
    return rows


def _table(rows: list[dict]) -> str:
    if not rows:
        return "(none)"
    keys = list(rows[0])

    def cell(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)

    widths = [max(len(k), *(len(cell(r[k])) for r in rows)) for k in keys]
    lines = ["  ".join(k.rjust(w) for k, w in zip(keys, widths, strict=True))]
    lines += ["  ".join(cell(r[k]).rjust(w) for k, w in zip(keys, widths, strict=True)) for r in rows]
    return "\n".join(lines)


def conditions(cfg: NetConfig | None) -> str:
    lib = load_library()
    parts = [
        f"machine: {platform.platform()}, {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs",
        f"python {sys.version.split()[0]}, torch {torch.__version__} ({torch.get_num_threads()} threads), numpy {np.__version__}",
        f"engine {lib.version} ({lib.path.name}), one thread, positions copied out per call",
    ]
    if cfg is not None:
        parts.append(
            f"network: d={cfg.d_model} L={cfg.n_layers} heads={cfg.n_heads}, trunk {cfg.trunk_dtype}, scope {cfg.attn_scope}, "
            f"attn {cfg.attn_impl}, tokens {'piece-centric (A5)' if cfg.piece_centric else 'one per square (A1)'}, "
            "random weights, no grad, CPU"
        )
    parts.append("load: not controlled - other processes may be running; figures are indicative")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--convert-boards", type=int, nargs="+", default=[1, 20, 60, 120, 240])
    ap.add_argument("--forward-boards", type=int, nargs="+", default=[1, 10, 30, 60])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seeds", type=int, default=3, help="positions per size class (one per seed)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--forward-repeat", type=int, default=3)
    ap.add_argument("--travel-prob", type=float, default=0.1)
    ap.add_argument("--layers", type=int, default=None, help="trunk depth (default: F1's 16)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default=None, help="trunk dtype (default: F1's bf16)")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--piece-centric", action="store_true", help="register A5's piece-centric tokens (a prototype)")
    ap.add_argument("--assembly", action="store_true", help="input assembly alone, by board-count range: no model, no forward")
    ap.add_argument("--assembly-boards", type=int, nargs="+", default=list(ASSEMBLY_BOARDS), help="its board counts")
    args = ap.parse_args(argv)

    overrides = {}
    if args.piece_centric:
        overrides["piece_centric"] = True
    if args.layers is not None:
        overrides["n_layers"] = args.layers
    if args.dtype is not None:
        overrides["trunk_dtype"] = args.dtype
    if args.assembly:
        # The network's input layout alone decides the assembly: its register counts and tokens.
        layout = NetConfig(**overrides)
        print(conditions(None))
        print(
            f"input layout: {layout.regs_per_board} registers a board, {layout.global_regs} global, tokens "
            f"{'piece-centric (A5)' if layout.piece_centric else 'one per square (A1)'}; no model built, no forward; host-to-device "
            "transfer: none on the CPU"
        )
        print(f"positions: seeded random pseudo-legal play from Standard (travel probability {args.travel_prob}), {args.seeds} seeds")
        torch.set_grad_enabled(False)
        boards = sorted(set(args.assembly_boards))
        games = _positions(boards, args.seeds, args.travel_prob)
        samples = assembly_samples(games, args.batch, args.repeat, layout)
        print(f"\ninput assembly per position, by board-count range (median over positions x {args.repeat} repeats after a warm-up):")
        print(_table(assembly_rows(samples, args.batch)))
        for gs in games.values():
            for g in gs:
                g.close()
        return 0
    cfg = None if args.skip_forward else NetConfig(**overrides)
    print(conditions(cfg))
    print(f"positions: seeded random pseudo-legal play from Standard (travel probability {args.travel_prob}), {args.seeds} seeds")
    torch.set_grad_enabled(False)

    sizes = sorted(set(args.convert_boards) | (set() if args.skip_forward else set(args.forward_boards)))
    games = _positions(sizes, args.seeds, args.travel_prob)
    print("\nconversion per position (median over positions x repeats):")
    print(_table(conversion({n: games[n] for n in sorted(args.convert_boards)}, args.repeat)))
    if not args.skip_forward:
        what = f"NetworkEvaluator: collate, forward, split; median of {args.forward_repeat} after a warm-up"
        print(f"\nnetwork forward per batch ({what}):")
        print(_table(forward({n: games[n] for n in sorted(args.forward_boards)}, args.batch, args.forward_repeat, cfg)))
    for gs in games.values():
        for g in gs:
            g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
