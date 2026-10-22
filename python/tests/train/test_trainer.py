"""Register M21's loop end to end, smoke-tested only (no training runs): self-play records from a
random network -> the sampler -> targets -> losses -> at most three optimiser steps -> SWA's
published network back into self-play; the same on synthetic records; the losses falling over a
few steps on a fixed synthetic batch; the step cap and G8's seam refusing."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model, synthetic_states
from t5d.train.records import network_id, read_game, write_game
from t5d.train.selfplay import GeneratorConfig, PCRConfig, generate_game
from t5d.train.smoke import MAX_STEPS, with_actions
from t5d.train.swa import SWAConfig
from t5d.train.targets import Row, build_targets
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import WindowConfig
from tests.engine.helpers import requires_engine

pytestmark = pytest.mark.grad

NET = NetConfig(**SMALL)
SMALL_WINDOW = WindowConfig(min_rows=1000)


def _config(**kw) -> TrainerConfig:
    return TrainerConfig(batch_size=8, swa=SWAConfig(every_steps=1, max_snapshots=10), window=SMALL_WINDOW, **kw)


@requires_engine
def test_self_play_records_to_optimiser_steps_to_the_published_network(tmp_path):
    torch.manual_seed(0)
    model = build_model(NET, seed=0)
    ev = NetworkEvaluator(model.eval(), max_batch=8)
    gen = dict(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, pcr=PCRConfig(full_fraction=0.5, cheap_playouts=2))
    paths = []
    with torch.no_grad():
        for seed, variant in ((31, "Standard"), (32, "Standard - Two Timelines")):
            g = generate_game(ev, GeneratorConfig(variant=variant, seed=seed, **gen), net_cfg=NET)
            paths.append(write_game(tmp_path / f"{seed}.npz", g.record))
    trainer = Trainer(model, NET, _config())
    written = sum(trainer.add_game(read_game(p)) for p in paths)  # inputs rebuilt from the stored records
    assert written == trainer.window.written > 0
    raw_id = network_id(model)
    for _ in range(MAX_STEPS):
        losses = trainer.step()
        assert all(math.isfinite(v) for v in losses.values())
        for head in ("policy", "wdl", "categorical", "short_term", "value_error", "soft_policy", "moves_left", "board", "effort"):
            assert head in losses
    with pytest.raises(RuntimeError, match="no training runs"):
        trainer.step()
    published = trainer.publish()
    assert published is not model and network_id(published) not in (raw_id, network_id(model))
    with torch.no_grad():  # the published network drives self-play again, and the record names it
        g = generate_game(NetworkEvaluator(published, max_batch=8), GeneratorConfig(seed=33, **gen), net_cfg=NET)
    assert g.record.header["network"]["id"] == network_id(published)


def _synthetic_rows(seed: int, counts=(1, 3, 2, 4, 2, 5)):
    states = with_actions(synthetic_states(seed, list(counts)))
    record = synthetic_record(seed, states)
    targets = build_targets(record)
    return record, states, [Row(s, targets, i) for i, s in enumerate(states)]


@pytest.mark.parametrize("optimiser", ["adamw", "nadam"])
def test_synthetic_records_through_the_loop(optimiser):
    record, states, _ = _synthetic_rows(4)
    model = build_model(replace(NET, trunk_dtype="fp32"), seed=4)
    trainer = Trainer(model, NET, _config(optimiser=optimiser))
    trainer.add_game(record, positions=states)
    for _ in range(MAX_STEPS):
        assert all(math.isfinite(v) for v in trainer.step().values())
    assert trainer.swa.snapshots == MAX_STEPS


def test_the_losses_fall_on_a_fixed_synthetic_batch():
    """A fixed batch, three steps at a raised rate: every step's total loss below the last."""
    _, _, rows = _synthetic_rows(5)
    model = build_model(replace(NET, trunk_dtype="fp32"), seed=5)
    trainer = Trainer(model, NET, _config(lr=1e-3))
    totals = [trainer.step_on(rows)["total"] for _ in range(MAX_STEPS)]
    assert all(math.isfinite(t) for t in totals)
    assert totals[2] < totals[1] < totals[0], totals


def test_the_step_cap_and_the_qat_seam():
    model = build_model(NET, seed=0)
    with pytest.raises(ValueError, match="never a training run"):
        TrainerConfig(max_steps=MAX_STEPS + 1).validate()
    with pytest.raises(NotImplementedError, match="G8"):
        Trainer(model, NET, TrainerConfig(qat=True))
    with pytest.raises(ValueError, match="horizons"):
        from t5d.train.targets import TargetConfig

        Trainer(model, NET, TrainerConfig(target=TargetConfig(short_term_horizons=(3.0,))))
