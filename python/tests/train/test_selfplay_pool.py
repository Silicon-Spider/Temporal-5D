"""Register G12: the training generator as a pool of concurrent games behind one evaluator
(``t5d/train/selfplay_pool.py`` on ``t5d/engine/pool.py``; spec section 7.8). Tiny games through the
engine, random weights - nothing is trained.

**What the record-equality tests hold, and what they do not.** A record is a function of its game
(its config, its seed, the network); the pool must add nothing to it.

* ``test_a_pooled_game_writes_the_record_of_the_game_alone``: with an evaluator that is an EXACT
  function of each position (``pool_helpers.DigestEvaluator``), eight games played concurrently
  each write the record of the same game played alone - every array of the file equal byte for
  byte, the header's JSON included - under three batchings: forwards merged across games under a
  token budget small enough to split a game's round, the shipped placeholders with greedy batches
  on fewer workers than games, and merging off. That holds the pool's own part: every evaluation
  reaches the search that asked, in order, and no draw, cache or search state leaks between games
  or depends on the thread, on timing or on what shared a forward. It is also the first test to run
  several search handles at once: a race between them would show here as a differing record or a
  crash - a test can show one, not rule one out. (The ``.npz`` container is not compared: a zip
  member carries its write time.)
* ``test_merge_off_with_the_network_is_the_game_alone_bit_for_bit``: with a real network and
  ``BatchConfig.merge`` off - every request the very forward the game alone makes - the same.
* It does NOT hold, and no test claims, that a game played with MERGED forwards of a real network
  writes the alone game's record: a forward's last bits move with the batch's size and padding
  (BUILD_LOG issue 262), so equality there is only up to the network's batch-equivalence tolerance
  (spec section 15.2's padded-batch row; 1e-5 on an FP32 trunk), after which a search may choose
  otherwise and the games diverge; nor need two pooled runs agree. Measured (docs/MEASUREMENTS.md,
  2026-11-22): in eight pooled runs of this file's three network games no record equalled the alone
  game's - the raw outputs apart by up to 4.5e-7 - though the moves played were the same.
  ``test_merged_forwards_give_the_alone_evaluations_to_tolerance_and_valid_records`` holds what is
  true there: the delivered evaluations are the positions' alone to that tolerance, and every
  record written is a valid record of the game that was played.

Each equality test first plays games alone twice: were the generator itself not deterministic, the
comparison with the pool would mean nothing. The ladder (Hunt, the settling scans) runs as built in
every game, on its time budgets: its proofs are withheld (H41) and the engine states that the tree
is then untouched, so its timing should be outside the record - and here is where it would show if
it were not.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.netplay import SMALL, PlayConfig, play_game
from t5d.engine.pool import BatchConfig, GameFailed, GamePool, SharedEvaluator, token_count
from t5d.net import NetConfig, build_model
from t5d.net.cache import state_digest
from t5d.train.records import RECORD_VERSION, network_id, params_from_dict, read_game, record_differences, to_arrays, write_game
from t5d.train.selfplay import SALT_SEARCH, GeneratorConfig, decision_seed, generate_game, replay_positions
from t5d.train.selfplay_pool import (
    CONCURRENT,
    LEAVES,
    MANIFEST,
    POOL_FORMAT,
    POOL_VERSION,
    SALT_GAME,
    game_configs,
    game_seed,
    generate_games,
    main,
)
from tests.engine.helpers import requires_engine
from tests.engine.pool_helpers import DigestEvaluator, RecordingEvaluator, pool_threads

pytestmark = requires_engine

NET = NetConfig(**SMALL)
ATOL = 1e-5  # FP32 trunk: the network's batch-equivalence tolerance (tests/net/test_invariance.py)

#: Eight tiny games (``tests/train/test_records.py``'s scale): the ply split at two seeds, the tree
#: form, Two Timelines at two seeds, D2's flat form at two seeds, and a tight legality budget with
#: another history window and no root noise (unknown moves among the candidates: a chosen child is
#: classified exactly before it is played).
CONFIGS = [
    GeneratorConfig(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=3),
    GeneratorConfig(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=11),
    GeneratorConfig(split="tree", playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=4),
    GeneratorConfig(variant="Standard - Two Timelines", playouts=6, batch_size=3, cap_turns=2, cap_boards=20, seed=5),
    GeneratorConfig(variant="Standard - Two Timelines", split="flat", playouts=10, batch_size=4, cap_turns=3, cap_boards=24, seed=9),
    GeneratorConfig(variant="Standard - Two Timelines", playouts=6, batch_size=2, cap_turns=2, cap_boards=20, seed=21),
    GeneratorConfig(variant="Standard - Two Timelines", split="flat", playouts=8, batch_size=1, cap_turns=2, cap_boards=16, seed=22),
    GeneratorConfig(
        playouts=8,
        batch_size=4,
        cap_turns=3,
        cap_boards=20,
        seed=6,
        window_k=2,
        root_noise=False,
        search_overrides=(("legality_checks", 0), ("legality_nodes", 0)),
    ),
]
#: The games the network plays here (every forward costs): the ply split, the tree form, a small
#: flat game.
NETWORK_CONFIGS = [CONFIGS[0], CONFIGS[2], CONFIGS[6]]


def _differences(a, b) -> list[str]:
    """Where two generated games differ: the records field by field (``record_differences``), the
    file's arrays byte for byte, the actions played and the targets' prior-to-target divergence."""
    out = record_differences(a.record, b.record)
    xa, xb = to_arrays(a.record), to_arrays(b.record)
    if sorted(xa) != sorted(xb):
        return [*out, "the files' arrays are not the same set"]
    out += [f"array {name}" for name in xa if xa[name].dtype != xb[name].dtype or xa[name].tobytes() != xb[name].tobytes()]
    if a.actions != b.actions:
        out.append("actions")
    if not np.array_equal(a.targets.kl, b.targets.kl, equal_nan=True):
        out.append("targets.kl")
    return out


def _alone(evaluator, configs, *, twice: int) -> list:
    """Each game alone; the first ``twice`` of them again - the baseline: the generator is
    deterministic in its config, or a comparison with the pool would mean nothing."""
    alone = [generate_game(evaluator, cfg, net_cfg=NET) for cfg in configs]
    for cfg, game in zip(configs[:twice], alone[:twice], strict=True):
        assert _differences(game, generate_game(evaluator, cfg, net_cfg=NET)) == [], "the game alone, twice, is not one record"
    return alone


_EXACT_ALONE: list = []


def _exact_alone() -> list:
    """``CONFIGS`` alone with the exact evaluator, each game played twice (once for the module)."""
    if not _EXACT_ALONE:
        _EXACT_ALONE.extend(_alone(DigestEvaluator(), CONFIGS, twice=len(CONFIGS)))
    return _EXACT_ALONE


def _within(sizes, cfg: BatchConfig) -> bool:
    padded = len(sizes) * max(sizes)
    return len(sizes) == 1 or (
        padded <= cfg.token_budget and len(sizes) <= cfg.max_positions and padded - sum(sizes) <= cfg.max_padding * padded
    )


#: The batchings the equality is held under, each with its games at once. "split_and_held": a budget
#: of three 2-board positions a forward, so a game's round of four leaves cannot go in one - requests
#: are split as well as merged - and the service thread holds a batch while a game has yet to ask, so
#: games do share forwards (both asserted). "greedy_defaults": the placeholders as shipped, on fewer
#: workers than games (a worker plays several games in turn); what shares a forward follows timing
#: there, and the records must not. "merge_off": every request its own call.
ARMS = {
    "split_and_held": (BatchConfig(token_budget=512, max_positions=8, max_wait_ms=2000.0), len(CONFIGS)),
    "greedy_defaults": (BatchConfig(), 3),
    "merge_off": (BatchConfig(merge=False), 4),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_a_pooled_game_writes_the_record_of_the_game_alone(arm):
    """The test that matters (module docstring): an evaluator exact per position, eight games played
    concurrently behind it - each record the alone game's, byte for byte, however the batches were
    formed."""
    batch, games = ARMS[arm]
    alone = _exact_alone()
    assert all(len(g.record.decisions) >= 2 for g in alone)
    recorder = RecordingEvaluator(DigestEvaluator())
    with GamePool(recorder, games=games, batch=batch) as pool:
        pooled = {r.index: r for r in generate_games(pool, CONFIGS, net_cfg=NET)}
    assert sorted(pooled) == list(range(len(CONFIGS))) and pool_threads() == []
    for i, game in enumerate(alone):
        assert pooled[i].error is None and pooled[i].config is CONFIGS[i]
        assert _differences(game, pooled[i].value) == [], f"game {i} in the pool is not the game alone"
        assert pooled[i].value.record.header["record_version"] == RECORD_VERSION
        assert set(pooled[i].value.record.header) == set(game.record.header), "the pool adds no field to a record"
    stats = pool.stats
    assert stats.forwards == len(recorder.calls) and stats.positions == sum(len(p) for p, _ in recorder.calls)
    if arm == "split_and_held":
        assert stats.cross_caller_forwards > 0, "no forward held two games' positions: the pooled run tested nothing"
        assert stats.split_requests > 0, "no request was split between forwards"
    if arm == "merge_off":
        assert stats.cross_caller_forwards == 0 and stats.split_requests == 0 and stats.forwards == stats.requests
    else:
        assert all(_within([token_count(p) for p in positions], batch) for positions, _ in recorder.calls)


def test_merge_off_with_the_network_is_the_game_alone_bit_for_bit():
    """A real network, merging off: every request is its own call of the evaluator - the forward
    the game alone makes, on another thread - so each pooled record is the alone game's, the header's
    network condition included."""
    evaluator = NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)
    alone = _alone(evaluator, NETWORK_CONFIGS, twice=1)
    with GamePool(evaluator, games=len(NETWORK_CONFIGS), batch=BatchConfig(merge=False)) as pool:
        pooled = {r.index: r.value for r in generate_games(pool, NETWORK_CONFIGS, net_cfg=NET)}
    for i, game in enumerate(alone):
        assert _differences(game, pooled[i]) == [], f"game {i} in the pool (merging off) is not the game alone"
        assert pooled[i].record.header["network"]["id"] == network_id(evaluator.model) is not None
    assert pool.stats.cross_caller_forwards == 0 and pool.stats.forwards == pool.stats.requests > 0


def test_merged_forwards_give_the_alone_evaluations_to_tolerance_and_valid_records(tmp_path):
    """A real network, merging on: what holds is the tolerance, not the bits (module docstring).
    Every evaluation a merged forward delivered equals the network's on that position alone to
    1e-5; every record reads back bit for bit and rebuilds its inputs and its final state. That a
    pooled record equals the alone game's is deliberately NOT asserted here: it need not."""
    model = build_model(NET, seed=0).eval()
    recorder = RecordingEvaluator(NetworkEvaluator(model, max_batch=16))
    configs = NETWORK_CONFIGS
    batch = BatchConfig(token_budget=4096, max_positions=16, max_wait_ms=2000.0)
    with GamePool(recorder, games=len(configs), batch=batch) as pool:
        pooled = {r.index: r.value for r in generate_games(pool, configs, net_cfg=NET, keep_positions=True)}
    assert sorted(pooled) == list(range(len(configs))) and pool.stats.cross_caller_forwards > 0
    alone = NetworkEvaluator(model)
    merged = [
        (p, e) for positions, evaluations in recorder.calls if len(positions) > 1 for p, e in zip(positions, evaluations, strict=True)
    ]
    assert len(merged) >= 8
    step = max(1, len(merged) // 24)  # a sample across the run
    for position, got in merged[::step]:
        want = alone([position])[0]
        for name in ("policy_logits", "wdl_logits", "value_logits", "board_logits"):
            a, b = getattr(got, name), getattr(want, name)
            assert a.shape == b.shape and np.allclose(a, b, rtol=0, atol=ATOL), (name, float(np.abs(a - b).max()))
        for name in ("submit_logit", "error", "moves_left"):
            assert np.allclose(getattr(got, name), getattr(want, name), rtol=0, atol=ATOL, equal_nan=True), name
    for i, game in pooled.items():
        record = game.record
        assert record.decisions and record.header["seed"] == configs[i].seed
        assert record.header["network"]["id"] == network_id(model)
        assert record_differences(record, read_game(write_game(tmp_path / f"game_{i}.npz", record))) == []
        rebuilt = replay_positions(record)  # the candidates and the final fingerprint checked inside
        assert [state_digest(p.state) for p in rebuilt] == [state_digest(p.state) for p in game.positions]


def test_a_pooled_netplay_game_writes_the_same_5dpgn():
    """The pool is not the generator's alone: ``netplay.play_game`` (self-play) on a ``GamePool``
    with the exact evaluator writes each game's 5dpgn, end and final state as the game alone."""
    exact = DigestEvaluator()
    configs = [
        PlayConfig(variant="Standard - Two Timelines", playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=5),
        PlayConfig(variant="Standard", split="tree", playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=6),
        PlayConfig(variant="Standard - Two Timelines", split="flat", playouts=8, batch_size=4, cap_turns=3, cap_boards=24, seed=7),
    ]

    def play(evaluator, cfg: PlayConfig):
        return play_game(evaluator, cfg, net_cfg=NET)

    def essence(res):
        return (res.record, res.end, res.winner, res.final_fingerprint, res.decisions, res.playouts, res.evaluations, res.cache_hits)

    alone = [essence(play(exact, cfg)) for cfg in configs]
    assert alone == [essence(play(exact, cfg)) for cfg in configs], "the game alone, twice, is not one record"
    assert all(a[4] >= 2 for a in alone)
    with GamePool(exact, games=len(configs), batch=BatchConfig(token_budget=512, max_positions=8, max_wait_ms=2000.0)) as pool:
        pooled = {r.index: essence(r.value) for r in pool.run(play, configs)}
    assert [pooled[i] for i in range(len(configs))] == alone
    assert pool.stats.cross_caller_forwards > 0 and pool.stats.split_requests > 0 and pool_threads() == []


class _Optimistic(DigestEvaluator):
    """An evaluator that says it delivers D9's optimistic prior."""

    policy = "optimistic"


class _HiddenOptimistic(DigestEvaluator):
    """An evaluator that does not say so - but each evaluation names its prior."""

    def evaluate(self, position):
        return replace(super().evaluate(position), policy="optimistic")


def test_generation_errors_and_the_self_play_guards():
    exact = DigestEvaluator()
    # A config the generator refuses is that game's failure; the other games finish.
    configs = [CONFIGS[3], replace(CONFIGS[0], split="turn"), replace(CONFIGS[3], seed=6)]
    with GamePool(exact, games=2, on_error="collect") as pool:
        results = {r.index: r for r in generate_games(pool, configs, net_cfg=NET)}
    assert sorted(results) == [0, 1, 2] and isinstance(results[1].error, ValueError) and "retired" in str(results[1].error)
    assert results[0].error is None and results[2].error is None and results[1].value is None
    assert results[0].value.record.header["seed"] == 5 and results[2].value.record.header["seed"] == 6
    with GamePool(exact, games=2) as pool, pytest.raises(GameFailed) as failed:
        list(generate_games(pool, configs[1:2], net_cfg=NET))
    assert failed.value.index == 0 and isinstance(failed.value.__cause__, ValueError)
    # D9's optimistic prior is kept out of self-play (section 8.1) through the pool: up front ...
    with GamePool(_Optimistic(), games=2) as pool, pytest.raises(ValueError, match="D9"):
        generate_games(pool, configs, net_cfg=NET)
    # ... by the generator, which sees the real evaluator's prior through its client ...
    with SharedEvaluator(_Optimistic()) as shared, shared.client() as client, pytest.raises(ValueError, match="D9"):
        generate_game(client, CONFIGS[3], net_cfg=NET)
    # ... and by the search's own loop, for an evaluator that hides it: each evaluation names its prior.
    with GamePool(_HiddenOptimistic(), games=1) as pool, pytest.raises(GameFailed) as failed:
        list(generate_games(pool, [CONFIGS[3]], net_cfg=NET))
    assert isinstance(failed.value.__cause__, ValueError) and "D9" in str(failed.value.__cause__)
    assert pool_threads() == []
    # The generator's further options reach every game unchanged (here its per-decision hook, called
    # on the game's own worker thread) - the pool names none of them.
    seen: list[tuple[int, int, str]] = []
    games = [CONFIGS[3], replace(CONFIGS[3], seed=6)]
    with GamePool(exact, games=2) as pool:
        played = {
            r.index: r.value
            for r in generate_games(
                pool,
                games,
                net_cfg=NET,
                on_decision=lambda search, d, position: seen.append((d.search_seed, d.decision, threading.current_thread().name)),
            )
        }
    for i, cfg in enumerate(games):
        mine = [(seed, d) for seed, d, _ in seen if seed == decision_seed(cfg.seed, d, SALT_SEARCH)]
        assert [d for _, d in mine] == list(range(len(played[i].record.decisions)))
    assert len(seen) == sum(len(g.record.decisions) for g in played.values()) and all(name.startswith("t5d-game-") for _, _, name in seen)
    # A run's seeds: a function of the run's seed and the game's index alone.
    seeds = [game_seed(7, i) for i in range(64)]
    assert seeds == [game_seed(7, i) for i in range(64)] and len(set(seeds)) == 64
    assert all(0 <= s < 2**64 for s in seeds) and game_seed(8, 0) != seeds[0]
    assert all(game_seed(7, i) != decision_seed(7, i, SALT_SEARCH) for i in range(64)) and SALT_GAME == 0x47414D45
    base = CONFIGS[2]
    assert list(game_configs(base, 7, 5)) == [replace(base, seed=s) for s in seeds[:5]]
    assert list(itertools.islice(game_configs(base, 7), 7)) == [replace(base, seed=s) for s in seeds[:7]]


def _lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_the_command_line_writes_records_and_a_manifest(tmp_path, capsys, monkeypatch):
    """``python -m t5d.train.selfplay_pool``: the records, and ``pool.json`` with the run's batching
    conditions - none of which is in a record."""
    monkeypatch.setattr(os, "cpu_count", lambda: 1)  # 2 games x the ladder's 5 threads exceed it: the warning
    out = tmp_path / "run"
    argv = ["--games", "3", "--concurrent", "2", "--net", "small", "--playouts", "6", "--cap-turns", "2", "--cap-boards", "20"]
    # --threads: this process's own count, so that the run leaves torch's global setting as it found it.
    argv += ["--threads", str(torch.get_num_threads())]
    assert main([*argv, "--seed", "4", "--token-budget", "2048", "--max-positions", "8", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "issue 451" in captured.err and "2 games x 5 verification threads" in captured.err
    lines = _lines(captured.out)
    assert sorted(line["index"] for line in lines[:-1]) == [0, 1, 2] and set(lines[-1]) == {"conditions", "totals"}
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    assert (manifest["format"], manifest["version"], manifest["error"]) == (POOL_FORMAT, POOL_VERSION, None)
    conditions, games = manifest["conditions"], manifest["games"]
    assert conditions["batch"] == BatchConfig(token_budget=2048, max_positions=8).to_dict()
    assert (conditions["concurrent"], conditions["games"], conditions["run_seed"], conditions["cpus"]) == (2, 3, 4, 1)
    assert conditions["generator_config"]["batch_size"] == LEAVES == 2 and conditions["generator_config"]["playouts"] == 6
    assert [g["index"] for g in games] == [0, 1, 2] and [g["seed"] for g in games] == [game_seed(4, i) for i in range(3)]
    stats = manifest["stats"]
    assert stats["positions"] >= stats["forwards"] > 0 and stats["requests"] > 0 and stats["padded_tokens"] >= stats["tokens"] > 0
    assert set(stats["by_range"]) == {"0-49"} and stats["largest_batch"] <= 8
    assert manifest["totals"]["games"] == 3 and manifest["totals"]["failed"] == 0
    for g in games:
        record = read_game(out / g["file"])
        assert g["file"] == f"game_{g['index']:06d}.npz" and record.header["seed"] == g["seed"]
        assert (g["end"], g["decisions"], g["actions"]) == (record.end, len(record.decisions), record.n_actions)
        assert record.header["network"]["id"] == conditions["network"]["id"] and record.header["network"]["seed"] == 4
        assert record.header["generator_config"]["batch_size"] == 2
        assert len(replay_positions(record)) == len(record.decisions)
    # No absolute path in the manifest: the files are named beside it.
    text = (out / MANIFEST).read_text(encoding="utf-8")
    assert tmp_path.name not in text and not any(os.path.isabs(g["file"]) for g in games)
    assert sorted(p.name for p in out.iterdir()) == ["game_000000.npz", "game_000001.npz", "game_000002.npz", MANIFEST]

    # The baseline arm and the ladder's switches: merging off, no Hunt, no settling scans - no warning.
    out = tmp_path / "plain"
    assert main([*argv, "--games", "2", "--no-merge", "--no-hunt", "--no-settling", "--on-error", "collect", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "issue 451" not in captured.err
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["conditions"]["batch"]["merge"] is False and manifest["stats"]["cross_caller_forwards"] == 0
    assert manifest["conditions"]["generator_config"]["search_overrides"] == [["hunt", 0], ["settling", 0]]
    full = params_from_dict(read_game(out / "game_000000.npz").header["search"]["full"])
    assert (full.hunt, full.settling, full.batch_size) == (0, 0, 2)
    assert CONCURRENT == 8 and pool_threads() == []
    with pytest.raises(SystemExit):
        main(["--games", "0"])
    with pytest.raises(SystemExit):
        main(["--max-padding", "1.5"])
