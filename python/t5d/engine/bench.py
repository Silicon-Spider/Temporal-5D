"""Conversion and forward-pass timings on real engine positions (stage N4).

Two figures, each with its conditions printed alongside:

* **Conversion, per position**: ``expand`` (the C call with every board's features, and
  the copy out of the engine's buffers), ``to_position`` (the ``PositionState``, validated,
  and the W/E scope), and ``Game.position`` end to end (both, plus the creator count) - the
  median and the 90th percentile over every position and repeat of a size class.
* **Network forward, per batch**: ``NetworkEvaluator`` over a batch of B positions of a
  size class - ``collate``, one dense forward of the model, the split into per-position
  policy logits and value logits - and the share of it spent in ``collate``; the median over
  repeats after one warm-up batch.

Positions are seeded random pseudo-legal play from the Standard start through the binding
(``play.py``; travel moves drawn with probability ``--travel-prob``), taken when the game
first reaches each board count - real engine positions, but random play: for costs only.

Run from ``python/``: ``py -m t5d.engine.bench`` (defaults: conversion at 1-240 boards,
forward at 1-60 boards, batches of 1 and 4, F1 as shipped - BF16 trunk, dense). These are
development-machine CPU figures, not section 15.3's cost measurements.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

from ..net import NetConfig, build_model, collate
from .capi import load_library
from .convert import to_position
from .evaluator import NetworkEvaluator
from .game import Game
from .play import random_game


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
            times, collate_t = [], []
            for _ in range(repeat):
                t0 = time.perf_counter()
                collate([p.state for p in batch])
                t1 = time.perf_counter()
                evaluator(batch)
                t2 = time.perf_counter()
                collate_t.append(t1 - t0)
                times.append(t2 - t1)
            tokens = sum(p.state.n_tokens for p in batch)
            med = statistics.median(times)
            rows.append(
                {
                    "boards": n,
                    "batch": b,
                    "tokens": tokens,
                    "batch s": med,
                    "per position s": med / b,
                    "collate ms": 1e3 * statistics.median(collate_t),
                    "moves scored": sum(p.move_count for p in batch),
                }
            )
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
            f"attn {cfg.attn_impl}, random weights, no grad, CPU"
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
    args = ap.parse_args(argv)

    overrides = {}
    if args.layers is not None:
        overrides["n_layers"] = args.layers
    if args.dtype is not None:
        overrides["trunk_dtype"] = args.dtype
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
