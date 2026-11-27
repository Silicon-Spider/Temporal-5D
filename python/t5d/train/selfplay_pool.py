"""The training generator as a pool of concurrent games behind one evaluator (register G12).

Spec section 7.8 (2026-11-15 review): "the generator that does this - a pool of concurrent games
behind one evaluator - is part of the pilot run's build"; section 15.7's Gate 2 names it as the
pilot's configuration, "with batches assembled to a token budget". ``train.selfplay.generate_game``
plays one game at a time and is unchanged; here it runs on ``engine.pool.GamePool``'s worker
threads, each game behind its own client of the one shared evaluator, which forms the batches
(``engine.pool``: the planner, the placeholders, the threading model).

* ``game_seed`` / ``game_configs`` - a run's games: game ``i``'s seed derives from the run's seed
  and ``i`` alone (``decision_seed`` under ``SALT_GAME``), so a game does not depend on how many
  games ran beside it, in which order they finished, or on which worker.
* ``generate_games`` - ``generate_game`` for every config, on a pool; the results as they complete.
  Each game's config is its own: a caller may hand any ``GeneratorConfig`` per game.
* ``main`` - ``python -m t5d.train.selfplay_pool`` (from ``python/``): plays games with a network of
  random weights (nothing is trained), writes each record (``records.write_game``) and the run's
  ``pool.json``.

**The record format is unchanged** (``RECORD_VERSION`` stays as it is): a record is a function of
its game - its config, its seed, the network - and the pool adds no field to it: no slot, no batch
statistics, no clock. Those are the RUN's conditions and go to ``pool.json`` (``POOL_FORMAT``
version ``POOL_VERSION``: the batching configuration, the games' seeds and files,
``engine.pool.PoolStats``) - information, never a section 15.3 figure.

**What "the same record" holds for.** With an evaluator that is an exact function of each position,
or with merging off (``BatchConfig.merge`` False: every request is the call the game alone makes),
a game played in the pool writes the record of the same game played alone, byte for byte -
``tests/train/test_selfplay_pool.py`` holds both, across concurrent games whose positions share
forwards. With MERGED forwards of a real network it does not hold bit for bit, and is not claimed:
a forward's last bits move with the batch's size and padding (BUILD_LOG issue 262), so the pooled
game's evaluations equal the alone game's only to the network's batch-equivalence tolerance (spec
section 15.2's padded-batch row; 1e-5 on an FP32 trunk), a search can then choose otherwise and
the games diverge; two pooled runs need not agree with each other either, since which positions
share a batch follows thread timing. Measured on three tiny games (docs/MEASUREMENTS.md,
2026-11-22): no merged-forward record equalled the alone game's in eight runs - the raw outputs
apart by up to 4.5e-7 - though the moves played were the same. Every such record is still a valid
record of the game that was played (it reads back and rebuilds its inputs).

**The ladder's threads** (BUILD_LOG issue 451, the user's to rule): each game makes its own
verification ladder - the engine's default five prover threads (M13) - so a pool asks for
games x that many. The library passes each game's ``verification_threads`` through untouched, so
that a pooled game is the game alone; the command line can lower it and warns when the product
exceeds the machine's logical processors. With the proof channel off (H23, H41) the ladder's proofs
are withheld and the engine leaves the tree untouched (``t5d/mcts.hpp``), so the ladder's timing -
which a loaded pool changes - is outside the record; the pooled-against-alone tests run with the
ladder on.

**Not built here**: match play's batching (G12's cap on the batch as a share of the simulations so
far, and speculative prefetch); worker processes; any measurement of speed - the batching gain is
a GPU figure (Gate 1 step 2), and ``CONCURRENT`` and ``LEAVES`` are PLACEHOLDERS the pilot's owner
fixes.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ..engine.capi import load_library
from ..engine.evaluator import NetworkEvaluator, policy_source
from ..engine.netplay import network_config
from ..engine.pool import (
    MAX_PADDING,
    MAX_POSITIONS,
    ON_ERROR,
    TOKEN_BUDGET,
    BatchConfig,
    EvaluatorFailed,
    GameFailed,
    GamePool,
    PoolResult,
)
from ..engine.search import search_params
from ..net.config import NetConfig
from ..net.model import build_model
from .records import write_game
from .selfplay import SPLITS, GeneratorConfig, decision_seed, generate_game, network_info
from .targets import TargetConfig

#: The salt of a game's seed in a run (``decision_seed``): apart from the per-decision draws'
#: salts, ``train.selfplay``'s and the sampler's small integers. "GAME" in ASCII.
SALT_GAME = 0x47414D45

#: PLACEHOLDER (G12; the pilot's owner fixes it): the command line's games at once. The library's
#: ``engine.pool.GAMES`` (32) is ``GamePool``'s own default.
CONCURRENT = 8
#: PLACEHOLDER (G12): the command line's leaves per game per round - the game's own
#: ``GeneratorConfig.batch_size``, its tree's virtual loss (section 7.8: "each contributes a leaf or
#: two"). The library's ``GeneratorConfig`` default stays 8, as built.
LEAVES = 2

#: The run manifest (``pool.json``).
POOL_FORMAT, POOL_VERSION = "t5d-selfplay-pool", 1
MANIFEST = "pool.json"


def game_seed(run_seed: int, index: int) -> int:
    """The seed of game ``index`` of the run seeded ``run_seed`` - a function of the two alone."""
    return decision_seed(run_seed, index, SALT_GAME)


def game_configs(base: GeneratorConfig, run_seed: int, count: int | None = None) -> Iterator[GeneratorConfig]:
    """A run's configs: ``base`` with each game's own seed (``game_seed``); ``count`` None: endless."""
    for index in itertools.count() if count is None else range(count):
        yield replace(base, seed=game_seed(run_seed, index))


def generate_games(
    pool: GamePool,
    configs: Iterable[GeneratorConfig],
    *,
    net_cfg: NetConfig,
    network: dict | None = None,
    keep_positions: bool = False,
    target_config: TargetConfig | None = None,
    **game_options,
) -> Iterator[PoolResult]:
    """``generate_game`` for every config, on ``pool``: a ``PoolResult`` per game as it completes,
    its ``value`` the ``GeneratedGame`` (``GamePool.run``: completion order, each result naming its
    config's index; ``GameFailed`` or a result carrying the error, by the pool's ``on_error``).
    ``network``: the records' network condition - None: ``network_info`` of the pool's evaluator,
    computed ONCE (it hashes the weights) and copied into each game's header. ``game_options``:
    further keyword arguments of ``generate_game``, passed to every game unchanged (the generator's
    later options need nothing here; whatever they hold is shared by the game threads and must be
    safe to read from several). D9's optimistic prior is refused up front, as the generator refuses
    it (section 8.1: "kept out of self-play")."""
    if policy_source(pool.evaluator) != "normal":
        raise ValueError("an optimistic evaluator searches at match play only, never in self-play (register D9, section 8.1)")
    if network is None:
        network = network_info(pool.evaluator, net_cfg)

    def play(evaluator, cfg: GeneratorConfig):
        return generate_game(
            evaluator,
            cfg,
            net_cfg=net_cfg,
            network=copy.deepcopy(network),
            keep_positions=keep_positions,
            target_config=target_config,
            **game_options,
        )

    return pool.run(play, configs)


def _summary(net_cfg: NetConfig) -> dict:
    return {k: v for k, v in asdict(net_cfg).items() if k in ("d_model", "n_layers", "n_heads", "ffn_ratio", "trunk_dtype")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-play as a pool of concurrent games behind one evaluator (register G12): records and the run's pool.json. "
        "Random weights - nothing is trained; the figures printed are information, not section 15.3 measurements."
    )
    parser.add_argument("--games", type=int, default=CONCURRENT, help="games to play in all")
    parser.add_argument("--concurrent", type=int, default=CONCURRENT, help=f"games at once (a placeholder: {CONCURRENT})")
    parser.add_argument("--net", default="small", choices=["small", "F1"])
    parser.add_argument("--variant", default="Standard")
    parser.add_argument("--split", default="ply", choices=sorted(SPLITS), help="D2's arm")
    parser.add_argument("--playouts", type=int, help="the full search's cap (default: the generator's)")
    parser.add_argument(
        "--leaves", type=int, default=LEAVES, help=f"leaves per game per round, the game's own batch size (a placeholder: {LEAVES})"
    )
    parser.add_argument("--cap-turns", type=int, help="I11's turn cap (default: the generator's)")
    parser.add_argument("--cap-boards", type=int, help="I11's board cap (default: the generator's)")
    parser.add_argument("--terminal-nodes", type=int, help="H22: the game loop's node budget (default: the generator's)")
    parser.add_argument("--seed", type=int, default=1, help="the run's seed: every game's seed, and the network's weights")
    parser.add_argument(
        "--token-budget", type=int, default=TOKEN_BUDGET, help="padded tokens per forward: positions x the longest (a placeholder)"
    )
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS, help="positions per forward (a placeholder)")
    parser.add_argument("--max-padding", type=float, default=MAX_PADDING, help="the largest padded share of a forward (a placeholder)")
    parser.add_argument(
        "--no-merge", action="store_true", help="every request its own forward: each game exactly as alone (the baseline arm)"
    )
    parser.add_argument("--max-wait-ms", type=float, default=0.0, help="hold a batch this long for more positions (0: greedy)")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (the machine is shared)")
    parser.add_argument("--device", help="the network's device (default: the CPU)")
    parser.add_argument("--verification-threads", type=int, help="each game's prover threads (default: the engine's 5, M13; issue 451)")
    parser.add_argument("--no-hunt", action="store_true", help="stage E10: no Hunt at the roots")
    parser.add_argument("--no-settling", action="store_true", help="stage E10: no settling scans")
    parser.add_argument(
        "--on-error", default="raise", choices=ON_ERROR, help="a failed game ends the run, or is recorded and the rest go on"
    )
    parser.add_argument("--out", type=Path, help=f"write each record and {MANIFEST} here (a directory)")
    args = parser.parse_args(argv)
    if args.games < 1 or args.concurrent < 1:
        parser.error("--games and --concurrent must be positive")

    overrides: list[tuple[str, object]] = []
    if args.verification_threads is not None:
        overrides.append(("verification_threads", args.verification_threads))
    if args.no_hunt:
        overrides.append(("hunt", 0))
    if args.no_settling:
        overrides.append(("settling", 0))
    fields: dict = dict(variant=args.variant, split=args.split, batch_size=args.leaves, search_overrides=tuple(overrides))
    for name in ("playouts", "cap_turns", "cap_boards", "terminal_nodes"):
        if getattr(args, name) is not None:
            fields[name] = getattr(args, name)
    base = GeneratorConfig(**fields)
    batch = BatchConfig(
        token_budget=args.token_budget,
        max_positions=args.max_positions,
        max_padding=args.max_padding,
        merge=not args.no_merge,
        max_wait_ms=args.max_wait_ms,
    )
    try:
        base.validate()
        batch.validate()
    except ValueError as exc:
        parser.error(str(exc))

    lib = load_library()
    params = search_params(lib, **dict(overrides))
    cpus = os.cpu_count() or 1  # logical processors
    ladder_threads = int(params.verification_threads) if (params.hunt or params.settling) else 0
    if args.concurrent * ladder_threads > cpus:
        print(
            f"warning: {args.concurrent} games x {ladder_threads} verification threads exceed this machine's {cpus} logical processors "
            "(BUILD_LOG issue 451): lower --verification-threads, or pass --no-hunt --no-settling",
            file=sys.stderr,
        )

    torch.set_num_threads(args.threads)
    net_cfg = network_config(args.net)
    model = build_model(net_cfg, seed=args.seed).eval()
    if args.device:
        model = model.to(args.device)
    evaluator = NetworkEvaluator(model, max_batch=max(args.max_positions, args.leaves))
    network = network_info(evaluator, net_cfg, seed=args.seed)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    games: list[dict] = []
    failure: BaseException | None = None
    t0 = time.perf_counter()
    with GamePool(evaluator, games=args.concurrent, batch=batch, on_error=args.on_error) as pool:
        try:
            for res in generate_games(pool, game_configs(base, args.seed, args.games), net_cfg=net_cfg, network=network):
                entry: dict = {
                    "index": res.index,
                    "seed": None if res.config is None else res.config.seed,
                    "seconds": round(res.seconds, 3),
                }
                if res.error is not None:
                    entry["error"] = repr(res.error)
                else:
                    record = res.value.record
                    result = record.header["result"]
                    entry.update(
                        end=result["end"],
                        unsettled=record.header.get("unsettled", ""),  # H22: an "unknown" end's question
                        winner=result["winner"],
                        actions=result["actions"],
                        decisions=len(record.decisions),
                        final_boards=result["final_boards"],
                    )
                    if args.out:
                        entry["file"] = f"game_{res.index:06d}.npz"
                        write_game(args.out / entry["file"], record)
                games.append(entry)
                print(json.dumps(entry), flush=True)
        except (GameFailed, EvaluatorFailed) as exc:
            failure = exc
            print(f"error: {exc}", file=sys.stderr)
    seconds = time.perf_counter() - t0
    stats = pool.stats

    games.sort(key=lambda g: g["index"])
    played = [g for g in games if "error" not in g]
    conditions = {
        "net": args.net,
        "net_config": _summary(net_cfg),
        "network": {"id": network["id"], "seed": network["seed"]},
        "engine_version": lib.version,
        "torch": torch.__version__,
        "torch_threads": args.threads,
        "device": str(evaluator.device),
        "cpus": cpus,
        "run_seed": args.seed,
        "games": args.games,
        "concurrent": args.concurrent,
        "batch": batch.to_dict(),
        "generator_config": base.to_dict(),  # the seed here is the base's: each game's own is in "games"
        "on_error": args.on_error,
    }
    totals = {
        "games": len(played),
        "failed": len(games) - len(played) + int(failure is not None),
        "decisions": sum(g["decisions"] for g in played),
        "seconds": round(seconds, 3),
        "decisions_per_s": round(sum(g["decisions"] for g in played) / seconds, 2) if seconds > 0 else 0.0,
        "positions_per_s": round(stats.positions / seconds, 1) if seconds > 0 else 0.0,
        "mean_batch": round(stats.mean_batch, 2),
        "padding_share": round(stats.padding_share, 4),
        "budget_fill": round(stats.fill(batch.token_budget), 4),
    }
    manifest = {
        "format": POOL_FORMAT,
        "version": POOL_VERSION,
        "note": "information, not a section 15.3 measurement: random weights, a shared machine",
        "conditions": conditions,
        "games": games,
        "stats": stats.to_dict(),
        "totals": totals,
        "error": None if failure is None else repr(failure),
    }
    if args.out:
        (args.out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"conditions": conditions, "totals": totals}), flush=True)
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONCURRENT",
    "LEAVES",
    "MANIFEST",
    "POOL_FORMAT",
    "POOL_VERSION",
    "SALT_GAME",
    "game_configs",
    "game_seed",
    "generate_games",
    "main",
]
