"""Register M21's loop end to end, smoke-tested only - the user's rule to the letter ("a few
optimiser steps on synthetic data - nothing more"; BUILD_LOG issue 237(a)): self-play records from
a random network -> rebuilt under the trainer's own network configuration -> the sampler ->
targets -> every loss, forward only; optimiser steps on synthetic rows alone, at most three per
model whatever the trainer -> SWA's published network back into self-play, its snapshot count in
the record; the losses falling over a few steps on a fixed synthetic batch; the step cap, the
synthetic-rows rule refusing (G8's QAT hook: ``test_qat.py``)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.expansion import Move
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model, synthetic_states
from t5d.train.records import network_id, params_from_dict, read_game, write_game
from t5d.train.selfplay import GeneratorConfig, PCRConfig, generate_game, replay_positions, root_position
from t5d.train.smoke import MAX_STEPS, optimiser_steps, smoke_train, with_actions
from t5d.train.swa import SWAConfig
from t5d.train.targets import Row, build_targets
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import WindowConfig
from tests.engine.helpers import requires_engine
from tests.train.test_records import same_input

pytestmark = pytest.mark.grad

NET = NetConfig(**SMALL)
SMALL_WINDOW = WindowConfig(min_rows=1000)
GEN = dict(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, pcr=PCRConfig(full_fraction=0.5, cheap_playouts=2))


def _config(**kw) -> TrainerConfig:
    return TrainerConfig(**{**dict(batch_size=8, swa=SWAConfig(every_steps=1, max_snapshots=10), window=SMALL_WINDOW), **kw})


def _synthetic_rows(seed: int, counts=(1, 3, 2, 4, 2, 5)):
    states = with_actions(synthetic_states(seed, list(counts)))
    record = synthetic_record(seed, states)
    targets = build_targets(record)
    return record, states, [Row(s, targets, i, synthetic=True) for i, s in enumerate(states)]


@requires_engine
def test_self_play_records_forward_only_then_synthetic_steps_then_the_published_network(tmp_path):
    """The miniature loop: generate -> records read back -> rows (inputs rebuilt from the stored
    records) -> every head's loss, forward only, and a step on them refused -> three steps on
    synthetic rows -> SWA's network published -> self-play again, its hash and snapshot count in
    the record."""
    torch.manual_seed(0)
    model = build_model(NET, seed=0)
    ev = NetworkEvaluator(model.eval(), max_batch=8)
    paths = []
    with torch.no_grad():
        for seed, variant in ((31, "Standard"), (32, "Standard - Two Timelines")):
            g = generate_game(ev, GeneratorConfig(variant=variant, seed=seed, **GEN), net_cfg=NET)
            paths.append(write_game(tmp_path / f"{seed}.npz", g.record))
    trainer = Trainer(model, NET, _config())
    written = sum(trainer.add_game(read_game(p)) for p in paths)
    assert written == trainer.window.written > 0
    raw_id = network_id(model)
    losses = trainer.losses_on(trainer.window.rows())  # the record -> batch path, forward only
    assert all(math.isfinite(v) for v in losses.values())
    for head in ("policy", "wdl", "categorical", "short_term", "value_error", "soft_policy", "moves_left", "board", "square", "effort"):
        assert head in losses
    # D4's per-square labels: re-derived from each record's replayed states by the engine's generator.
    assert all(r.targets.square and r.targets.square[r.index].shape[0] == r.state.n_boards for r in trainer.window.rows())
    assert losses["square"] > 0
    assert network_id(model) == raw_id and optimiser_steps(model) == 0
    with pytest.raises(RuntimeError, match="synthetic rows only"):
        trainer.step()
    assert network_id(model) == raw_id and trainer.steps == 0
    _, _, rows = _synthetic_rows(6)
    for _ in range(MAX_STEPS):
        assert all(math.isfinite(v) for v in trainer.step_on(rows).values())
    with pytest.raises(RuntimeError, match="no training runs"):
        trainer.step_on(rows)
    published = trainer.publish()
    assert published is not model and network_id(published) not in (raw_id, network_id(model))
    assert published.swa_snapshots == trainer.swa.snapshots == MAX_STEPS
    with torch.no_grad():  # the published network drives self-play again, and the record names it
        g = generate_game(NetworkEvaluator(published, max_batch=8), GeneratorConfig(seed=33, **GEN), net_cfg=NET)
    assert g.record.header["network"]["id"] == network_id(published)
    assert g.record.header["network"]["swa_snapshots"] == MAX_STEPS
    with torch.no_grad():  # the raw weights published (SWA off): no snapshot
        raw = Trainer(build_model(NET, seed=1), NET, _config(swa=SWAConfig(enabled=False))).publish()
        g = generate_game(NetworkEvaluator(raw, max_batch=8), GeneratorConfig(seed=34, **GEN), net_cfg=NET)
    assert g.record.header["network"]["swa_snapshots"] == 0


def _expected_inputs(record, net_cfg: NetConfig, window_k: int):
    """Each decision's input under ``net_cfg`` - the generator's own ``root_position`` on the
    record's game replayed by hand, with its search parameters."""
    h = record.header
    out = []
    with Game.variant(h["variant"]) as game:
        for d in record.decisions:
            params = params_from_dict(h["search"]["full" if d.full_search or "cheap" not in h["search"] else "cheap"])
            out.append(root_position(game, params, net_cfg, window_k).state)
            k = int(d.children.candidate[d.played_child])
            if k < 0:
                game.submit()
            else:
                game.apply(Move.from_record(d.candidates[k]))
    return out


@requires_engine
@pytest.mark.parametrize(
    ("kw", "window_k"),
    [
        (dict(regs_per_board=1, global_regs=4), None),
        (dict(regs_per_board=3, global_regs=8, typed_edge_bias=True), None),
        (dict(typed_edge_bias=True), 3),
    ],
    ids=["registers", "registers_and_relations", "relations_and_window"],
)
def test_a_trainer_rebuilds_inputs_under_its_own_configuration(kw, window_k):
    """A record generated by one network configuration, read by a trainer of another (register
    counts, typed relations, the history window): each row's input is the generator's input
    under the trainer's configuration, not the record's."""
    with torch.no_grad():
        g = generate_game(NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8), GeneratorConfig(seed=35, **GEN), net_cfg=NET)
    net2 = replace(NET, **kw)
    trainer = Trainer(build_model(net2, seed=1), net2, _config(window_k=window_k))
    trainer.add_game(g.record)
    wk = g.record.header["expand"]["window_k"] if window_k is None else window_k
    want = _expected_inputs(g.record, net2, wk)
    record_cfg = _expected_inputs(g.record, NET, g.record.header["expand"]["window_k"])
    rows = trainer.window.rows()
    assert {r.index for r in rows} == set(range(len(want)))
    for r in rows:
        assert same_input(r.state, want[r.index]), r.index
    assert not all(same_input(want[i], record_cfg[i]) for i in range(len(want)))  # the configurations differ
    assert (rows[0].state.regs_per_board, rows[0].state.global_regs) == (net2.regs_per_board, net2.global_regs)
    assert all(math.isfinite(v) for v in trainer.losses_on(rows).values())  # forward only
    # The window sets the rebuilt positions' W/E scope (the cached configuration's; a row keeps the state alone).
    assert all(p.scope.k == wk for p in replay_positions(g.record, net2, window_k=window_k))


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


def test_the_step_cap_is_per_model():
    """MAX_STEPS bounds a model, not a trainer: a second trainer, a smoke run or the published
    copy of the same weights adds no step past it."""
    _, _, rows = _synthetic_rows(7)
    model = build_model(replace(NET, trunk_dtype="fp32"), seed=7)
    first = Trainer(model, NET, _config())
    first.step_on(rows)
    first.step_on(rows)
    second = Trainer(model, NET, _config())
    second.step_on(rows)
    assert optimiser_steps(model) == MAX_STEPS
    for trainer in (second, Trainer(model, NET, _config())):
        with pytest.raises(RuntimeError, match="no training runs"):
            trainer.step_on(rows)
    with pytest.raises(RuntimeError, match="no training runs"):
        smoke_train(NET, steps=1, model=model)
    published = first.publish()
    assert optimiser_steps(published) == MAX_STEPS
    with pytest.raises(RuntimeError, match="no training runs"):
        Trainer(published, NET, _config()).step_on(rows)


def test_steps_on_synthetic_rows_only():
    _, _, rows = _synthetic_rows(8)
    model = build_model(replace(NET, trunk_dtype="fp32"), seed=8)
    trainer = Trainer(model, NET, _config())
    with pytest.raises(RuntimeError, match="synthetic rows only"):
        trainer.step_on([replace(r, synthetic=False) for r in rows])
    assert optimiser_steps(model) == 0 and trainer.steps == 0
    trainer.step_on(rows)
    assert optimiser_steps(model) == 1


def test_the_step_cap_and_the_horizons():
    model = build_model(NET, seed=0)
    with pytest.raises(ValueError, match="never a training run"):
        TrainerConfig(max_steps=MAX_STEPS + 1).validate()
    with pytest.raises(ValueError, match="horizons"):
        from t5d.train.targets import TargetConfig

        Trainer(model, NET, TrainerConfig(target=TargetConfig(short_term_horizons=(3.0,))))
