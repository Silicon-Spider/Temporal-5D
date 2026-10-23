"""Register A5's token counts on 5dpgn records - section 14.4's piece-centric figures measured
where M18 says compute goes (the human games now), through the engine binding only.

For every record, replayed as the gate adapter's parse mode replays (``Game.from_pgn(...,
legal=False)``: the record's actions, submitted without the royal-safety test), at every action
start: the board count N, the pieces, and the distinct empty cells the candidate moves land on
(D) - for the **legal** moves (``Game.position(legal_moves=True)``, exact: no budget) and the
**pseudo-legal** ones (the expansion's default) - and from them A1's and A5's token counts
(``net.cost.token_counts``: ``T_A1 = 66 N + 16``, ``T_A5 = pieces + D + 2 N + 16``, the network's
own construction, ``state.to_piece_centric``) and section 9.1's analytic costs, A1 against A5
(``net.cost.piece_centric_costs``: the dense pass; the cached node, A5's destination tokens live;
the frozen K/V). Positions are distinct by fingerprint (H30) and side to move, as M7's are
(``t5d-measure``'s summary), and only records that replay to the end count (M7's 716 of the 761);
the action starts of every such record, repeats included, weight the buckets (M18: "where compute
goes" - the decisions spent there).

Per N bucket (M7's: 1-49, 50-99, 100-159, 160-239) and per move set: the distinct positions, the
median N, the mean pieces per board, D (median, p95, max), the mean T_A1 and T_A5 and their ratio
(of the means, and the median of the per-position ratios), and the mean analytic costs' ratios -
then every bucket pooled under M18's weighting, dense alone and M16's blend (0.9 cached, 0.1 dense).

Run from ``python/``: ``py -m t5d.engine.token_counts RECORD ... [--jsonl FILE]`` (``--jsonl``:
one line per distinct position, for re-computation). Indicative CPU wall times are not reported:
the counts do not depend on the machine.

**Counted FLOPs** (``--flops``): one dense forward of the F1 network (FP32 trunk, random weights),
A1 against A5, on B14's block-sparse path and on the every-key (chunked) path - counted by
``net.cost.count_flops`` (PyTorch's FlopCounterMode, which executes the forward) at 1 and at 2
layers and extrapolated to the network's 16: the layers are identical, so the difference of the two
counts is one layer's, exactly (``--flops-check`` counts the full depth too, on the first position,
to show it). Positions: the first action start of ``--flops-record`` with at least
``--flops-boards`` boards (legal moves, exact), and ``synthetic_state`` at 240 boards (8 and 24
timelines) and 600 (12 and 24).

Never through ``oracles/``: the engine's own expansion is the only rules source (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..net.config import NetConfig
from ..net.cost import piece_centric_costs, token_counts
from .errors import EngineError
from .game import Game

#: M7's board-count buckets (``t5d-measure --summary``'s first four; the games stop at 186).
BUCKETS: tuple[tuple[int, int], ...] = ((1, 49), (50, 99), (100, 159), (160, 239), (240, 10**9))
#: The move sets measured: the legal moves (exact) and the pseudo-legal ones.
MOVE_SETS = ("legal", "pseudo")
#: M16's blend of the cached and dense configurations' per-node costs.
CACHED_SHARE = 0.9


def bucket_of(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    raise ValueError(f"no bucket holds {n} boards")


def position_counts(game: Game) -> dict:
    """The current position's token counts for both move sets (``cost.token_counts``) and its key."""
    snap = game.snapshot()
    out: dict = {"key": snap.fingerprint + ("w" if snap.to_move == 0 else "b"), "boards": game.info().boards}
    for name in MOVE_SETS:
        pos = game.position(legal_moves=name == "legal")
        out[name] = token_counts(pos.state) | {"moves": int(pos.state.moves.shape[0])}
    return out


def record_positions(path: str | Path) -> tuple[list[dict], str]:
    """Every action start of a record (``position_counts``) and how it ended: ``"ok"`` (replayed to
    the end) or ``"refused: ..."`` (its positions then do not count)."""
    out = []
    try:
        with Game.from_pgn_file(path, legal=False) as g:
            out.append(position_counts(g))
            for _ in g.replay():
                out.append(position_counts(g))
            if g.record_actions().truncated:
                return out, "refused: truncated"
    except EngineError as e:
        return out, f"refused: {type(e).__name__}: {e}"
    return out, "ok"


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if len(values) else math.nan


def summarise(positions: dict[str, dict], weights: dict[str, int], cfg: NetConfig | None = None) -> dict:
    """The per-bucket summary of distinct positions (``positions``: key -> ``position_counts``),
    weighted by ``weights`` (key -> action starts) for the pooled figures (the module docstring)."""
    cfg = NetConfig() if cfg is None else cfg
    out: dict = {"buckets": {}, "pooled": {}}
    for move_set in MOVE_SETS:
        rows: dict[str, list[tuple[dict, dict, int]]] = {}
        for key, p in positions.items():
            c = p[move_set]
            rows.setdefault(bucket_of(p["boards"]), []).append((c, piece_centric_costs(c, cfg), weights.get(key, 1)))
        pooled = {"dense_a1": 0.0, "dense_a5": 0.0, "node_a1": 0.0, "node_a5": 0.0, "action_starts": 0}
        for bucket, items in sorted(rows.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+"))):
            counts = [c for c, _, _ in items]
            costs = [k for _, k, _ in items]
            n = [c["boards"] for c in counts]
            a1 = [c["a1"] for c in counts]
            a5 = [c["a5"] for c in counts]
            d = [c["destinations"] for c in counts]
            summary = {
                "positions": len(items),
                "action_starts": int(sum(w for _, _, w in items)),
                "median_boards": _quantile(n, 0.5),
                "pieces_per_board": float(np.mean([c["pieces"] / c["boards"] for c in counts])),
                "d_median": _quantile(d, 0.5),
                "d_p95": _quantile(d, 0.95),
                "d_max": int(max(d)),
                "d_per_board": float(np.mean([c["destinations"] / c["boards"] for c in counts])),
                "head_d_share": float(sum(c["head_destinations"] for c in counts) / max(1, sum(d))),
                "t_a1_mean": float(np.mean(a1)),
                "t_a5_mean": float(np.mean(a5)),
                "token_ratio_of_means": float(np.mean(a1) / np.mean(a5)),
                "token_ratio_median": _quantile([x / y for x, y in zip(a1, a5, strict=True)], 0.5),
                "token_ratio_max": float(max(x / y for x, y in zip(a1, a5, strict=True))),
            }
            for what in ("dense", "dense_attention", "node", "kv"):
                summary[f"{what}_ratio_of_means"] = float(
                    np.mean([k[f"{what}_a1"] for k in costs]) / np.mean([k[f"{what}_a5"] for k in costs])
                )
            out["buckets"].setdefault(move_set, {})[bucket] = summary
            for k, (_, cost, w) in zip(counts, items, strict=True):
                del k
                for what in ("dense", "node"):
                    for arm in ("a1", "a5"):
                        pooled[f"{what}_{arm}"] += w * cost[f"{what}_{arm}"]
                pooled["action_starts"] += w
        blend = {arm: CACHED_SHARE * pooled[f"node_{arm}"] + (1 - CACHED_SHARE) * pooled[f"dense_{arm}"] for arm in ("a1", "a5")}
        out["pooled"][move_set] = {
            "action_starts": pooled["action_starts"],
            "dense_ratio": pooled["dense_a1"] / pooled["dense_a5"],
            "node_ratio": pooled["node_a1"] / pooled["node_a5"],
            "blend_ratio": blend["a1"] / blend["a5"],
            "dense_share_by_bucket_a1": {
                b: float(sum(w * k["dense_a1"] for _, k, w in rows[b]) / pooled["dense_a1"])
                for b in sorted(rows, key=lambda x: int(x.split("-")[0].rstrip("+")))
            },
        }
    return out


def measure(paths: Iterable[str | Path], *, jsonl: Path | None = None, progress: bool = False) -> dict:
    """The summary over ``paths`` (``summarise``), with the records' fates."""
    positions: dict[str, dict] = {}
    weights: dict[str, int] = {}
    fates: dict[str, int] = {}
    refused: list[str] = []
    sink = jsonl.open("w", encoding="utf-8") if jsonl else None
    try:
        paths = list(paths)
        for i, path in enumerate(paths):
            found, fate = record_positions(path)
            fates[fate.split(":")[0]] = fates.get(fate.split(":")[0], 0) + 1
            if fate != "ok":
                refused.append(f"{Path(path).name}: {fate}")
                continue
            for p in found:
                weights[p["key"]] = weights.get(p["key"], 0) + 1
                if p["key"] not in positions:
                    positions[p["key"]] = p
                    if sink:
                        sink.write(json.dumps({"record": Path(path).name} | p) + "\n")
            if progress and (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(paths)} records, {len(positions)} distinct positions", file=sys.stderr, flush=True)
    finally:
        if sink:
            sink.close()
    summary = summarise(positions, weights)
    summary["records"] = fates
    summary["refused"] = refused
    summary["distinct_positions"] = len(positions)
    return summary


def counted_forward(state, *, piece_centric: bool, impl: str, layers: int, cfg: NetConfig | None = None) -> int:
    """The FLOPs ``net.cost.count_flops`` counts for one forward of ``state`` (square-centric) through
    a network of ``cfg`` (default F1, FP32 trunk) at ``layers`` layers on ``impl``, A5's tokens or A1's."""
    import torch

    from ..net.cost import count_flops
    from ..net.model import build_model
    from ..net.state import collate, to_piece_centric

    base = NetConfig(trunk_dtype="fp32") if cfg is None else cfg
    net_cfg = replace(base, n_layers=layers, attn_impl=impl, piece_centric=piece_centric)
    model = build_model(net_cfg, seed=0).eval()
    x = to_piece_centric(state, strict=False) if piece_centric else state
    batch = collate([x])
    with torch.no_grad():
        _, flops = count_flops(lambda: model(batch))
    return flops


def flops_table(states: dict[str, object], *, check: bool = False) -> list[dict]:
    """Per position and path: A1's and A5's counted forward at 16 layers (from 1 and 2 - the module
    docstring), their ratio, the token counts and section 9.1's analytic figures beside them."""
    cfg = NetConfig()
    rows = []
    for name, state in states.items():
        counts = token_counts(state)
        analytic = piece_centric_costs(counts, cfg)
        for impl in ("block_sparse", "chunked"):
            row: dict = {"position": name, "path": impl, "boards": counts["boards"], "t_a1": counts["a1"], "t_a5": counts["a5"]}
            for arm, pc in (("a1", False), ("a5", True)):
                one = counted_forward(state, piece_centric=pc, impl=impl, layers=1)
                two = counted_forward(state, piece_centric=pc, impl=impl, layers=2)
                row[f"counted_{arm}"] = one + (cfg.n_layers - 1) * (two - one)
                row[f"per_layer_{arm}"] = two - one
                row[f"analytic_{arm}"] = analytic[f"dense_{arm}"]
                if check and not rows:
                    row[f"counted_full_{arm}"] = counted_forward(state, piece_centric=pc, impl=impl, layers=cfg.n_layers)
            row["counted_ratio"] = row["counted_a1"] / row["counted_a5"]
            row["analytic_ratio"] = analytic["dense_ratio"]
            rows.append(row)
            print(json.dumps(row), file=sys.stderr, flush=True)
    return rows


def first_position(path: str | Path, boards: int):
    """The first action start of a record with at least ``boards`` boards (legal moves, exact)."""
    with Game.from_pgn_file(path, legal=False) as g:
        if g.info().boards >= boards:
            return g.position(legal_moves=True).state
        for _ in g.replay():
            if g.info().boards >= boards:
                return g.position(legal_moves=True).state
    raise ValueError(f"{path} never reaches {boards} boards")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="*", help="5dpgn files, or directories of them")
    ap.add_argument("--jsonl", type=Path, help="one line per distinct position")
    ap.add_argument("--out", type=Path, help="write the summary here (JSON) as well")
    ap.add_argument("--flops", action="store_true", help="count A1's and A5's dense forward instead (the module docstring)")
    ap.add_argument("--flops-record", type=Path, help="the record whose position the counted forward starts with")
    ap.add_argument("--flops-boards", type=int, default=44)
    ap.add_argument("--flops-check", action="store_true", help="count the full depth too, on the first position")
    ap.add_argument("--no-synthetic", action="store_true", help="--flops: the record's position alone")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (the machine is shared)")
    args = ap.parse_args(argv)
    if args.flops:
        import torch

        from ..net.synthetic import synthetic_state

        torch.set_num_threads(args.threads)
        states: dict[str, object] = {}
        if args.flops_record:
            states[f"{args.flops_record.name}@{args.flops_boards}"] = first_position(args.flops_record, args.flops_boards)
        if not args.no_synthetic:
            for n, k in ((240, 8), (240, 24), (600, 12), (600, 24)):
                states[f"synthetic {n} boards, {k} timelines"] = synthetic_state(0, n, n_timelines=k)
        rows = flops_table(states, check=args.flops_check)
        text = json.dumps(rows, indent=1)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if not args.records:
        ap.error("records are needed (or --flops)")
    paths: list[Path] = []
    for r in args.records:
        p = Path(r)
        paths.extend(sorted(p.glob("*.5dpgn")) if p.is_dir() else [p])
    summary = measure(paths, jsonl=args.jsonl, progress=True)
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
