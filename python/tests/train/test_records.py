"""Register I15's training records (spec section 12.8): written by the generator, read back bit
for bit, their network input rebuilt from the record alone, their raw root outputs the root's
evaluation the search used, their conditions complete. Tiny games with random weights - nothing
is trained."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from t5d.engine.capi import T5D_LEGALITY_LEGAL, T5D_NO_LIMIT, CSearchParams
from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL
from t5d.engine.play import random_position_game
from t5d.engine.search import search_params
from t5d.net import NetConfig, build_model
from t5d.net.cache import state_digest
from t5d.train.records import (
    DECISION_FIELDS,
    RECORD_VERSION,
    RecordVersionError,
    from_arrays,
    network_id,
    params_dict,
    params_from_dict,
    read_game,
    record_differences,
    to_arrays,
    write_game,
)
from t5d.train.selfplay import GeneratorConfig, generate_game, replay_positions
from tests.engine.helpers import requires_engine

NET = NetConfig(**SMALL)


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(0)
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


def _travel_start(seed: int) -> tuple[str, ...]:
    """A start with timelines branched by travel: seeded random play (travel preferred)."""
    with random_position_game("Standard", seed, boards=10, travel_prob=1.0) as g:
        return tuple(fen for _, _, fen in g.snapshot().boards)


#: Section 15.2's configurations: both of D2's generator arms, two variants, a custom start with
#: branched timelines, a tight legality budget (unknown moves among the candidates), the root
#: temperature and noise off.
CONFIGS = {
    "ply": GeneratorConfig(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=3),
    "tree": GeneratorConfig(split="tree", playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=4),
    "two_timelines": GeneratorConfig(variant="Standard - Two Timelines", playouts=6, batch_size=3, cap_turns=2, cap_boards=20, seed=5),
    "budget": GeneratorConfig(
        playouts=8,
        batch_size=4,
        cap_turns=3,
        cap_boards=20,
        seed=6,
        root_noise=False,
        search_overrides=(("legality_checks", 0), ("legality_nodes", 0)),
    ),
}


def _generate(evaluator, cfg: GeneratorConfig, keep_positions: bool = True):
    with torch.no_grad():
        return generate_game(evaluator, cfg, net_cfg=NET, keep_positions=keep_positions)


@requires_engine
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_a_record_reads_back_bit_for_bit(evaluator, name, tmp_path):
    g = _generate(evaluator, CONFIGS[name])
    rec = g.record
    assert rec.decisions, "the game made decisions"
    path = write_game(tmp_path / f"{name}.npz", rec)
    back = read_game(path)
    assert record_differences(rec, back) == []
    assert back.header["record_version"] == RECORD_VERSION
    # The arrays' layout: every decision field present, offsets consistent.
    arrays = to_arrays(rec)
    for name_ in DECISION_FIELDS:
        assert arrays[f"d_{name_}"].shape == (len(rec.decisions),)
    assert arrays["c_offset"][-1] == sum(d.move_count for d in rec.decisions)
    assert arrays["e_offset"][-1] == sum(len(d.children) for d in rec.decisions)


@requires_engine
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_record_rebuilds_its_network_input(evaluator, name, tmp_path):
    """I15: "later architectures retrained on the archive" - each decision's input rebuilt from the
    stored record (the start, the played children, the recorded expand options and legality
    budget) has the record's candidate moves and the generation-time state, digest for digest."""
    g = _generate(evaluator, CONFIGS[name])
    back = read_game(write_game(tmp_path / "g.npz", g.record))
    rebuilt = replay_positions(back)
    assert len(rebuilt) == len(g.positions) == len(back.decisions)
    for live, again, d in zip(g.positions, rebuilt, back.decisions, strict=True):
        assert state_digest(live.state) == state_digest(again.state)
        assert again.moves.tobytes() == d.candidates.tobytes()
        keys = np.stack([again.state.timeline, again.state.half_turn], axis=1)
        assert np.array_equal(keys, d.boards)


@requires_engine
def test_a_custom_start_with_travel(evaluator, tmp_path):
    """A custom 5DFEN start with branched timelines (seeded random play preferring travel) plays,
    records and rebuilds like a variant's."""
    for seed in (1, 2, 3):
        try:
            fens = _travel_start(seed)
        except ValueError:
            continue
        cfg = GeneratorConfig(start_fen=fens, playouts=6, batch_size=3, cap_turns=0, cap_boards=16, max_actions=3, seed=seed)
        g = _generate(evaluator, cfg)
        back = read_game(write_game(tmp_path / f"c{seed}.npz", g.record))
        assert record_differences(g.record, back) == []
        assert back.header["variant"] == "custom" and back.header["start_fen"] == list(fens)
        if back.decisions:
            assert len(replay_positions(back)) == len(back.decisions)
        return
    pytest.skip("no seeded travel start reached 10 boards")


@requires_engine
def test_the_raw_root_outputs_are_the_searchs_root_evaluation(evaluator):
    """The raw outputs are the network's evaluation of the decision's input: its WDL is the
    engine's normalised root WDL (the evaluation the search used, often a cache hit), and its
    logits give the search's noise-free prior - a softmax at the root's temperature over the
    candidates and a legal submit (``set_priors``) - wherever every child is known legal."""
    g = _generate(evaluator, CONFIGS["ply"])
    compared = 0
    for d in g.record.decisions:
        assert d.root_evaluated
        p = np.exp(d.raw_wdl_logits.astype(np.float64) - d.raw_wdl_logits.max())
        assert np.allclose(p / p.sum(), d.root_wdl, atol=1e-5)
        c = d.children
        legal = bool((c.legality == T5D_LEGALITY_LEGAL).all())
        offered = d.submit_possible and d.submit_legal
        if not legal or len(c) != d.move_count + int(offered) or (offered and math.isnan(d.raw_submit_logit)):
            continue
        z = np.append(d.raw_logits.astype(np.float64), d.raw_submit_logit if offered else -np.inf) / d.root_policy_temperature
        prior = np.exp(z - z.max())
        prior /= prior.sum()
        expected = np.where(c.candidate >= 0, prior[np.maximum(c.candidate, 0)], prior[-1])
        assert np.allclose(c.prior_noise_free, expected, atol=1e-5)
        compared += 1
    assert compared > 0


@requires_engine
def test_the_records_conditions(evaluator, tmp_path):
    """The conditions I15 names: ruleset, board count, network, search configuration (every
    t5d_search_params field, the legality budget among them), temperature, the PCR flag; no
    proof (the channel is off - H23, H41); the visits' forced and steered counts (0 today)."""
    cfg = CONFIGS["budget"]
    g = _generate(evaluator, cfg, keep_positions=False)
    h = read_game(write_game(tmp_path / "g.npz", g.record)).header
    for key in ("ruleset", "rules", "engine_version", "network", "search", "expand", "generator_config", "pgn", "result", "value_source"):
        assert key in h, key
    assert h["network"]["id"] == network_id(evaluator.model) and h["network"]["config"]["d_model"] == NET.d_model
    full = params_from_dict(h["search"]["full"])
    assert (full.legality_checks, full.legality_nodes, full.root_noise) == (0, 0, 0)
    assert h["proof_refs"] == [] and all(d.proof_ref == -1 for d in g.record.decisions)
    assert h["value_source"] == "wdl"
    for d in g.record.decisions:
        assert d.n_boards == d.boards.shape[0] and d.full_search and d.playout_cap == cfg.playouts
        assert (d.children.forced_visits == 0).all() and (d.children.steered_visits == 0).all()
        assert d.move_temperature in (0.0, cfg.temperature)
    # The tight budget leaves moves unknown at some root.
    assert any((d.children.legality != T5D_LEGALITY_LEGAL).any() for d in g.record.decisions)


def test_search_params_round_trip():
    p = search_params(legality_checks=7, legality_nodes=T5D_NO_LIMIT, cpuct=1.75, expand=(6, "all", 3))
    d = params_dict(p)
    assert json.loads(json.dumps(d)) == d
    q = params_from_dict(d)
    assert bytes(q) == bytes(p) and isinstance(q, CSearchParams)


def test_another_version_is_refused(tmp_path):
    from t5d.train.records import GameRecord

    rec = GameRecord(header={"record_version": RECORD_VERSION + 1, "result": {"end": "turn_cap", "winner": None, "actions": 0}})
    arrays = to_arrays(rec)
    with pytest.raises(RecordVersionError):
        from_arrays(arrays)
    np.savez(tmp_path / "x.npz", **arrays)
    with pytest.raises(RecordVersionError):
        read_game(tmp_path / "x.npz")


def test_the_network_id_follows_the_weights():
    a, b = build_model(NET, seed=0), build_model(NET, seed=0)
    assert network_id(a) == network_id(b)
    with torch.no_grad():
        next(b.parameters()).add_(1e-3)
    assert network_id(a) != network_id(b)


@requires_engine
def test_the_generator_refuses_whole_turns_and_the_optimistic_policy():
    with pytest.raises(ValueError, match="234"):
        GeneratorConfig(split="turn").validate()
    cfg = NetConfig(**{**SMALL, "optimistic_policy": True})
    ev = NetworkEvaluator(build_model(cfg, seed=0).eval(), policy="optimistic")
    with pytest.raises(ValueError, match="D9"):
        generate_game(ev, CONFIGS["ply"], net_cfg=cfg)
    with Game.variant("Standard") as g:
        assert g.info().boards == 1
