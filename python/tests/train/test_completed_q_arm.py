"""Register I21's completed-Q policy target in its place (spec sections 12.2, 12.5; M24's target
arm): the builder's switch (``TargetConfig.completed_q``, off by default - the targets, their file
and the batch then as they were), the target stored and batched beside the pruned-visit one, the
loss arm (weight 0 by default; ``losses.completed_q_arm``), the smoke path and the trainer under
the arm - at most three optimiser steps on synthetic rows, record rows forward only. Nothing is
trained, and nothing is ranked: the ranking waits for the pilot run's data."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model, collate, synthetic_states
from t5d.train.completed_q import CompletedQConfig, completed_q_policy
from t5d.train.losses import LossConfig, Targets, completed_q_arm, compute_losses
from t5d.train.records import GameRecord, read_game, write_game
from t5d.train.selfplay import GeneratorConfig, PCRConfig, generate_game, replay_positions
from t5d.train.smoke import MAX_STEPS, main, optimiser_steps, smoke_train, synthetic_targets, with_actions
from t5d.train.swa import SWAConfig
from t5d.train.targets import (
    COMPLETED_Q_SEARCHES,
    TARGET_SOURCES,
    Row,
    TargetConfig,
    build_targets,
    collate_rows,
    emits_completed_q,
    raw_prior,
    read_targets,
    submit_share_of,
    target_differences,
    write_targets,
)
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import WindowConfig
from tests.engine.helpers import requires_engine
from tests.train.test_completed_q import LOGITS, _children, _decision
from tests.train.test_targets import GAMES

pytestmark = pytest.mark.grad

NET = NetConfig(**SMALL)  # two layers at d = 64, FP32
BASE = TargetConfig(short_term_horizons=NET.short_term_horizons)
ALL = replace(BASE, completed_q="all")
CQ_FIELDS = ("completed_q", "completed_q_submit")


def _record(seed: int, counts=(1, 3, 2, 4, 2, 5), **kw):
    """A synthetic record whose states offer submit on some rows (legal on rows 1 and 5); every
    second decision a cheap search."""
    states = with_actions(synthetic_states(seed, list(counts)))
    return synthetic_record(seed, states, **kw), states


def _rows(seed: int, cfg: TargetConfig = ALL, **kw):
    record, states = _record(seed, **kw)
    targets = build_targets(record, cfg)
    return record, states, targets, [Row(s, targets, i, synthetic=True) for i, s in enumerate(states)]


def _game(decisions, *, share: float | None = None) -> GameRecord:
    """A record of constructed decisions (``test_completed_q``'s), one per turn; ``share``: the full
    search's ``submit_share`` in the header."""
    ds = [replace(d, decision=i, action=i) for i, d in enumerate(decisions)]
    search = {} if share is None else {"full": {"submit_share": share}}
    header = {"result": {"end": "checkmate", "winner": 0, "actions": len(ds), "final_active_timelines": [0]}, "search": search}
    return GameRecord(header=header, decisions=ds)


def _same(a, b) -> bool:
    """Two optional tensors equal bit for bit (NaN equal to NaN)."""
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        return bool(((a == b) | (a.isnan() & b.isnan())).all())
    return bool(torch.equal(a, b))


def _config(**kw) -> TrainerConfig:
    return TrainerConfig(**{**dict(batch_size=8, swa=SWAConfig(every_steps=1, max_snapshots=10), window=WindowConfig(min_rows=1000)), **kw})


# --- the builder's switch -----------------------------------------------------------------------------


def test_the_default_builder_writes_no_completed_q_target_and_the_arm_changes_nothing_else():
    """Off (the default) nothing is written. On, the completed-Q target is written beside the
    others and every other target - the policy, the soft policy, the pruned visits, the divergence,
    ``full_search``, every value-type target - is the default build's, bit for bit (section 12.2:
    "Value-type targets are unchanged")."""
    record, _ = _record(4)
    assert TargetConfig().completed_q == "off"
    default = build_targets(record)
    assert default.completed_q == []
    assert target_differences(default, build_targets(record, TargetConfig(completed_q="off"))) == []
    for mode in ("full", "cheap", "all"):
        arm = build_targets(record, TargetConfig(completed_q=mode))
        differences = target_differences(default, arm)
        assert differences and all(name.startswith("completed_q") for name in differences), differences
        assert target_differences(default, replace(arm, completed_q=[])) == []


@pytest.mark.parametrize("seed", [1, 7])
def test_the_switch_names_which_searches_emit(seed):
    """ "full": the full searches (like for like with the visit target); "cheap": the cheap ones;
    "all": every one (section 12.5: under I21 cheap searches may emit policy targets) - each entry
    exactly the function's, a distribution."""
    record, _ = _record(seed, full_every=2)
    wanted = {"full": lambda d: d.full_search, "cheap": lambda d: not d.full_search, "all": lambda d: True}
    for mode, want in wanted.items():
        cfg = TargetConfig(completed_q=mode)
        t = build_targets(record, cfg)
        assert len(t.completed_q) == len(record.decisions)
        for d, c in zip(record.decisions, t.completed_q, strict=True):
            assert (c is not None) == want(d) == emits_completed_q(d, cfg)
            if c is not None:
                assert np.array_equal(c, completed_q_policy(d, CompletedQConfig(submit_share=0.0)))
                assert math.isclose(float(c.sum()), 1.0, abs_tol=1e-12)
    assert not any(emits_completed_q(d, TargetConfig()) for d in record.decisions)
    t = build_targets(record, TargetConfig(completed_q="all"))
    cheap = [i for i, d in enumerate(record.decisions) if not d.full_search]
    assert cheap and all(t.policy[i] is None and t.completed_q[i] is not None for i in cheap)
    assert sum(c is not None for c in t.completed_q) > sum(p is not None for p in t.policy)


def test_the_builders_settings_and_the_records_submit_share_reach_the_target():
    """The builder passes its three settings and the record's submit share (the full search's, as
    the recorded divergence reads it) to the function, and nothing else."""
    cfg = TargetConfig(completed_q="all", completed_q_c_visit=10.0, completed_q_c_scale=0.25, completed_q_value_range="signed")
    assert cfg.completed_q_settings(0.25) == CompletedQConfig(10.0, 0.25, "signed", 0.25)
    plain = _decision(_children([4, 0, 1], [0.2, 9.9, -0.4]), LOGITS)
    # submit offered, the network gave it no logit: its prior is the search's placeholder share
    sub = _decision(_children([5, 1, 6], [0.0, 0.0, 0.2], candidate=[0, 1, -1]), [0.0, 0.0], submit=True)
    for share in (None, 0.25):
        record = _game([plain, sub], share=share)
        recorded = 0.0 if share is None else share
        assert submit_share_of(record) == recorded
        t = build_targets(record, cfg)
        for d, c in zip(record.decisions, t.completed_q, strict=True):
            assert np.array_equal(c, completed_q_policy(d, CompletedQConfig(10.0, 0.25, "signed", recorded)))
        assert (t.completed_q[1][2] > 0) == (share is not None)
    defaults = build_targets(record, TargetConfig(completed_q="all"))
    assert not np.array_equal(defaults.completed_q[0], t.completed_q[0])  # the settings are read
    assert np.array_equal(defaults.completed_q[0], completed_q_policy(record.decisions[0], CompletedQConfig(submit_share=0.25)))


def test_a_search_that_visited_nothing_emits_the_raw_prior():
    """The reading pinned: which decisions emit is a function of the search class alone, so a full
    search that visited no child has the function's target - the raw prior - while it has no visit
    target and is no full search to the visit-type targets."""
    visited = _decision(_children([4, 0, 1], [0.2, 9.9, -0.4]), LOGITS)
    nothing = _decision(_children([0, 0, 0], [0.9, -0.9, 0.3]), LOGITS, wdl=(1.0, 0.0, -2.0))
    record = _game([visited, nothing])
    for mode in ("full", "all"):
        t = build_targets(record, TargetConfig(completed_q=mode))
        np.testing.assert_allclose(t.completed_q[1], raw_prior(record.decisions[1]), rtol=0, atol=1e-15)
        assert t.policy[1] is None and not t.full_search[1] and math.isnan(t.kl[1])
        assert t.policy[0] is not None and t.full_search[0]
    assert build_targets(record, TargetConfig(completed_q="cheap")).completed_q == [None, None]  # both are full searches


def test_a_flat_decision_has_no_completed_q_target_and_keeps_its_per_board_one():
    """D2's flat form: the function returns None, so the builder writes none (one entry per decision,
    each None), the per-board pruned-visit target is as it was, and the loss skips the rows."""
    states = synthetic_states(4, [3, 5], n_timelines=3)
    record = synthetic_record(4, states, full_every=4, flat=True)
    default = build_targets(record, BASE)
    t = build_targets(record, ALL)
    assert len(t.completed_q) == len(record.decisions) == 2 and all(c is None for c in t.completed_q)
    assert any(f is not None for f in t.flat)
    assert target_differences(default, t) == ["completed_q: 0 entries against 2"]
    batch, tgt = collate_rows([Row(s, t, i, synthetic=True) for i, s in enumerate(states)], NET)
    assert tgt.completed_q is not None and float(tgt.completed_q.abs().sum()) == 0.0 and float(tgt.completed_q_submit.abs().sum()) == 0.0
    losses = compute_losses(build_model(NET, seed=2)(batch), tgt, completed_q_arm())
    assert float(losses["completed_q"].detach()) == 0.0 and float(losses["flat_policy"].detach()) > 0.0


def test_the_configuration_is_validated():
    assert COMPLETED_Q_SEARCHES == ("off", "full", "cheap", "all")
    for mode in COMPLETED_Q_SEARCHES:
        TargetConfig(completed_q=mode).validate()
    with pytest.raises(ValueError, match="completed_q"):
        TargetConfig(completed_q="half").validate()
    for bad in (
        TargetConfig(completed_q_c_visit=-1.0),
        TargetConfig(completed_q_c_scale=math.inf),
        TargetConfig(completed_q_value_range="half"),
    ):
        with pytest.raises(ValueError, match="completed_q"):
            bad.validate()
        with pytest.raises(ValueError, match="completed_q"):
            build_targets(_game([_decision(_children([1], [0.0]), [0.0])]), bad)
    made = TargetConfig().to_dict()
    assert {k: made[k] for k in made if k.startswith("completed_q")} == {
        "completed_q": "off",
        "completed_q_c_visit": 50.0,
        "completed_q_c_scale": 1.0,
        "completed_q_value_range": "unit",
    }
    assert TargetConfig().completed_q_settings() == CompletedQConfig()  # the paper's values, the function's defaults


def test_the_arm_round_trips_through_a_file_and_default_files_are_unchanged(tmp_path):
    record, _ = _record(4)
    for mode in ("full", "cheap", "all"):
        t = build_targets(record, TargetConfig(completed_q=mode))
        assert target_differences(t, read_targets(write_targets(tmp_path / f"{mode}.npz", t))) == []
    default = build_targets(record)
    path = write_targets(tmp_path / "default.npz", default)
    with np.load(path) as z:
        assert not [name for name in z.files if name.startswith("completed_q")]
    back = read_targets(path)
    assert back.completed_q == [] and target_differences(default, back) == []


def test_the_source_is_the_search_record():
    """D5 (section 12.4's source rule): the target is the engine's own search's - the raw logits and
    the root children's visits and backed-up values - never a learned verifier's estimate."""
    assert "search record" in TARGET_SOURCES["completed_q"] and "verifier" not in TARGET_SOURCES["completed_q"]


# --- the batch ----------------------------------------------------------------------------------------


def test_collate_carries_the_target_beside_the_policy():
    """The batch holds the completed-Q target next to the policy target, padded with zeros - and is
    otherwise the default batch, field for field; off, the two fields are None."""
    record, states = _record(4)
    built = {mode: build_targets(record, replace(BASE, completed_q=mode)) for mode in COMPLETED_Q_SEARCHES}
    batches = {mode: collate_rows([Row(s, t, i, synthetic=True) for i, s in enumerate(states)], NET)[1] for mode, t in built.items()}
    off, tgt = batches["off"], batches["all"]
    assert off.completed_q is None and off.completed_q_submit is None
    assert tgt.completed_q.shape == tgt.policy.shape and tgt.completed_q_submit.shape == (len(states),)
    assert tgt.completed_q.dtype == tgt.completed_q_submit.dtype == torch.float32
    for i, st in enumerate(states):
        m = int(st.moves.shape[0])
        want = built["all"].completed_q[i].astype(np.float32)
        assert np.array_equal(tgt.completed_q[i, :m].numpy(), want[:m]) and float(tgt.completed_q_submit[i]) == float(want[m])
        assert float(tgt.completed_q[i, m:].abs().sum()) == 0.0  # padding
    assert torch.allclose(tgt.completed_q.sum(-1) + tgt.completed_q_submit, torch.ones(len(states)), atol=1e-6)
    assert (
        float(tgt.completed_q_submit[1]) > 0.0 and float(tgt.completed_q_submit[0]) == 0.0
    )  # submit: legal on row 1, not offered on row 0
    for mode in ("full", "cheap", "all"):
        for f in fields(Targets):
            if f.name not in CQ_FIELDS:
                assert _same(getattr(batches[mode], f.name), getattr(off, f.name)), (mode, f.name)
    full = torch.tensor([bool(d.full_search) for d in record.decisions])
    assert 0 < int(full.sum()) < len(states)
    for mode, emits in (("full", full), ("cheap", ~full)):
        mass = batches[mode].completed_q.sum(-1) + batches[mode].completed_q_submit
        assert torch.allclose(mass, emits.float(), atol=1e-6), mode
    # rows of a game built with the switch off, beside rows of one built with it on: no target, all zero
    mixed = [Row(states[0], built["off"], 0, synthetic=True), Row(states[1], built["all"], 1, synthetic=True)]
    _, tgt = collate_rows(mixed, NET)
    assert float(tgt.completed_q[0].sum() + tgt.completed_q_submit[0]) == 0.0 and float(tgt.completed_q[1].sum()) > 0.0


def test_a_target_over_the_wrong_move_count_is_refused():
    _, states, targets, _ = _rows(4)
    assert targets.policy[1] is None and targets.completed_q[1] is not None  # a cheap search: only the completed-Q target
    other = next(s for s in states if s.moves.shape[0] != states[1].moves.shape[0])
    with pytest.raises(ValueError, match="completed-Q target"):
        collate_rows([Row(other, targets, 1, synthetic=True)], NET)


# --- the loss arm -------------------------------------------------------------------------------------


def _forward(seed: int = 4, model_seed: int = 3):
    _, _, _, rows = _rows(seed)
    model = build_model(NET, seed=model_seed)
    batch, tgt = collate_rows(rows, NET)
    return model, batch, tgt


def test_the_loss_is_the_policy_heads_cross_entropy_over_every_row_that_carries_a_target():
    """L = the mean, over the rows whose completed-Q target has mass, of -sum_a pi'(a) log softmax(z)(a),
    z the main policy head's action logits (moves and submit, D8) - cheap rows included, each row
    weighing the same. At its default weight, 0, the total is the total without it, bit for bit."""
    model, batch, tgt = _forward()
    out = model(batch)
    losses = compute_losses(out, tgt)
    target = torch.cat([tgt.completed_q, tgt.completed_q_submit[:, None]], dim=-1)
    valid = out.action_valid()
    logp = torch.log_softmax(out.action_logits().masked_fill(~valid, float("-inf")), dim=-1).masked_fill(~valid, 0.0)
    per = -(target * logp).sum(dim=-1)
    has = target.sum(dim=-1) > 0
    assert bool(has.all()) and 0 < int(tgt.full_search.sum()) < len(has)  # rows of cheap searches among them
    assert torch.allclose(losses["completed_q"], per[has].mean(), rtol=1e-5)
    # a row with no target (all zero) is skipped; the rest weigh the same
    moves, submit = tgt.completed_q.clone(), tgt.completed_q_submit.clone()
    moves[0], submit[0] = 0.0, 0.0
    keep = has.clone()
    keep[0] = False
    partial = compute_losses(out, replace(tgt, completed_q=moves, completed_q_submit=submit))
    assert torch.allclose(partial["completed_q"], per[keep].mean(), rtol=1e-5)
    # no submit mass given: the moves alone
    moves_only = compute_losses(out, replace(tgt, completed_q_submit=None))
    assert torch.allclose(moves_only["completed_q"], -(tgt.completed_q * logp[:, :-1]).sum(dim=-1).mean(), rtol=1e-5)
    # weight 0 by default: every other loss and the total are those of the batch without the target
    assert LossConfig().weights["completed_q"] == 0.0
    without = compute_losses(out, replace(tgt, completed_q=None, completed_q_submit=None))
    assert "completed_q" not in without and set(losses) == set(without) | {"completed_q"}
    for name, value in without.items():
        assert torch.equal(losses[name], value), name
    # under the arm: the policy's term leaves the total and the completed-Q term enters at its weight
    arm = compute_losses(out, tgt, completed_q_arm())
    assert torch.allclose(arm["total"], without["total"] - without["policy"] + arm["completed_q"], rtol=1e-6)
    assert torch.equal(arm["policy"], without["policy"])  # still reported, at weight 0


def test_without_the_target_there_is_no_loss_key():
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2)
    model = build_model(cfg, seed=3)
    batch = collate(with_actions(synthetic_states(3, [1, 3, 2, 4])))
    tgt = synthetic_targets(batch, cfg, torch.Generator().manual_seed(3))
    assert tgt.completed_q is None and tgt.completed_q_submit is None
    assert "completed_q" not in compute_losses(model(batch), tgt)
    assert "completed_q" not in smoke_train(cfg, steps=1, seed=0).losses[0]
    # the arm's weight on a batch with no target is refused - never the policy head trained on nothing
    with pytest.raises(ValueError, match="completed_q"):
        compute_losses(model(batch), tgt, completed_q_arm())
    fresh = build_model(cfg, seed=3)
    with pytest.raises(ValueError, match="completed_q"):
        smoke_train(cfg, steps=1, seed=0, loss_cfg=completed_q_arm(), model=fresh)
    assert optimiser_steps(fresh) == 0


def test_the_arm_moves_the_policys_weight_and_trains_the_policy_head_alone():
    """``completed_q_arm``: the policy's weight on the completed-Q loss, the policy's own 0, every
    other weight and setting as given (the soft policy, D9's optimistic policy and the flat form
    stay on the pruned-visit target). The loss reaches the main policy head and its submit readout,
    never the soft policy head."""
    base, arm = LossConfig(), completed_q_arm()
    assert arm.weights["policy"] == 0.0 and arm.weights["completed_q"] == base.weights["policy"] == 1.0
    rest = [name for name in base.weights if name not in ("policy", "completed_q")]
    assert set(arm.weights) == set(base.weights) and all(arm.weights[name] == base.weights[name] for name in rest)
    assert base.weights["completed_q"] == 0.0  # the configuration given is not modified
    assert completed_q_arm(arm).weights == arm.weights  # applying the arm twice is applying it once
    custom = completed_q_arm(LossConfig(weights={"policy": 0.5, "soft_policy": 2.0}, q_ratio=0.25, board_reduction="sum"))
    assert custom.weights == {"policy": 0.0, "completed_q": 0.5, "soft_policy": 2.0}
    assert (custom.q_ratio, custom.board_reduction) == (0.25, "sum")
    model, batch, tgt = _forward()
    assert bool(batch.submit_legal.any())
    compute_losses(model(batch), tgt)["completed_q"].backward()
    assert float(model.policy.src.weight.grad.abs().sum()) > 0 and float(model.policy.dst.weight.grad.abs().sum()) > 0
    assert float(model.submit.out.weight.grad.abs().sum()) > 0
    soft = model.aux.soft_policy
    assert soft.pointer.src.weight.grad is None and soft.submit.out.weight.grad is None


def test_cheap_rows_train_under_the_arm_only():
    """Section 12.5: only full searches emit the visit-type policy targets; under I21 a cheap search
    emits the completed-Q one. A batch of cheap searches alone has no policy and no soft-policy loss,
    and a completed-Q loss."""
    model, batch, tgt = _forward()
    cheap = replace(tgt, full_search=torch.zeros(tgt.policy.shape[0], dtype=torch.bool))
    losses = compute_losses(model(batch), cheap, completed_q_arm())
    assert float(losses["policy"].detach()) == 0.0 and float(losses["soft_policy"].detach()) == 0.0
    assert float(losses["completed_q"].detach()) > 0.0 and math.isfinite(float(losses["total"].detach()))


# --- the smoke path -----------------------------------------------------------------------------------


@pytest.mark.parametrize("flat_row", [None, 3])
def test_the_synthetic_arm_targets_are_drawn_last(flat_row):
    """The smoke test's synthetic completed-Q targets are drawn after every other target, so every
    seeded target is the same with and without the arm; they are a distribution, positive on every
    legal action, on every position but the flat one."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2)
    batch = collate(with_actions(synthetic_states(1, [1, 3, 6, 3])))
    plain = synthetic_targets(batch, cfg, torch.Generator().manual_seed(1), flat_row=flat_row)
    arm = synthetic_targets(batch, cfg, torch.Generator().manual_seed(1), flat_row=flat_row, completed_q=True)
    assert plain.completed_q is None and plain.completed_q_submit is None
    for f in fields(Targets):
        if f.name not in CQ_FIELDS:
            assert _same(getattr(plain, f.name), getattr(arm, f.name)), f.name
    _, submit_legal = batch.submit_mask()
    valid = torch.cat([batch.move_valid, submit_legal[:, None]], dim=-1)
    target = torch.cat([arm.completed_q, arm.completed_q_submit[:, None]], dim=-1)
    carries = torch.ones(valid.shape[0], dtype=torch.bool)
    if flat_row is not None:
        assert bool(arm.flat_rows[flat_row])
        carries[flat_row] = False
        assert float(target[flat_row].sum()) == 0.0  # a flat decision has no completed-Q target
    assert bool(submit_legal.any()) and float(target[~valid].abs().sum()) == 0.0  # nothing on padding or an illegal submit
    assert bool((target[carries][valid[carries]] > 0).all())  # positive on every legal action
    assert torch.allclose(target[carries].sum(dim=-1), torch.ones(int(carries.sum())), atol=1e-6)
    assert not bool(arm.full_search.all())  # cheap searches carry one too


def test_a_few_optimiser_steps_reduce_the_completed_q_loss():
    """At most three optimiser steps on synthetic targets: the completed-Q loss, optimised alone,
    falls (the last below the first). Not training."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2)
    report = smoke_train(cfg, steps=3, seed=6, loss_cfg=LossConfig(weights={"completed_q": 1.0, "flat_policy": 0.0}), completed_q=True)
    assert all(math.isfinite(v) for step in report.losses for v in step.values())
    first, last = report.losses[0]["completed_q"], report.losses[-1]["completed_q"]
    assert last < first, (first, last)


def test_the_smoke_test_runs_under_the_arm(capsys):
    """``smoke_train`` under ``completed_q_arm`` with the synthetic targets: every loss finite, and
    one step moves every parameter tensor the default smoke step moves - the main policy head and
    its submit readout among them, through the completed-Q loss alone (the policy's weight is 0);
    and the command line's ``--completed-q``."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2)
    report = smoke_train(cfg, steps=1, seed=0, loss_cfg=completed_q_arm(), completed_q=True)
    assert all(math.isfinite(v) for v in report.losses[0].values()) and report.losses[0]["completed_q"] > 0
    assert report.unchanged == smoke_train(cfg, steps=1, seed=0).unchanged
    assert not [name for name in report.unchanged if name.startswith(("policy.", "submit."))], report.unchanged
    assert main(["--steps", "1", "--fp32", "--completed-q"]) == 0  # the default network, one step
    out = capsys.readouterr().out
    assert out.startswith("step 0:") and " completed_q=" in out and " policy=" in out


# --- the trainer under the arm ------------------------------------------------------------------------


def test_synthetic_records_through_the_loop_under_the_arm():
    record, states = _record(4)
    model = build_model(NET, seed=4)
    trainer = Trainer(model, NET, _config(target=ALL, loss=completed_q_arm()))
    trainer.add_game(record, positions=states)
    for _ in range(MAX_STEPS):
        values = trainer.step()
        assert all(math.isfinite(v) for v in values.values())
        assert values["completed_q"] > 0 and "policy" in values
    assert optimiser_steps(model) == MAX_STEPS


def test_the_arms_loss_falls_on_a_fixed_synthetic_batch():
    """A fixed synthetic batch, three steps at a raised rate under the arm: every step's
    completed-Q loss below the last. Not training."""
    _, _, _, rows = _rows(5)
    model = build_model(NET, seed=5)
    trainer = Trainer(model, NET, _config(lr=1e-3, target=ALL, loss=completed_q_arm()))
    values = [trainer.step_on(rows) for _ in range(MAX_STEPS)]
    assert all(math.isfinite(v["total"]) for v in values)
    losses = [v["completed_q"] for v in values]
    assert losses[2] < losses[1] < losses[0], losses


def test_the_loss_arm_is_refused_without_its_target():
    """The trainer's guard: the arm's weight with the builder's switch off would train the main
    policy head on nothing."""
    model = build_model(NET, seed=0)
    with pytest.raises(ValueError, match="completed_q"):
        Trainer(model, NET, TrainerConfig(loss=completed_q_arm()))
    with pytest.raises(ValueError, match="completed_q"):
        Trainer(model, NET, TrainerConfig(loss=completed_q_arm(), target=BASE))
    Trainer(model, NET, TrainerConfig(loss=completed_q_arm(), target=replace(BASE, completed_q="full")))
    Trainer(model, NET, TrainerConfig(target=ALL))  # the target built and stored, nothing trained on it: weight 0


# --- section 15.2's I15 row, and record rows, under the arm ---------------------------------------------


@pytest.fixture(scope="module")
def evaluator():
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


@requires_engine
@pytest.mark.parametrize("game", ["flat", "ply", "tree"])
def test_the_arm_is_rederived_from_stored_records_as_written_at_generation(evaluator, game, tmp_path):
    """Section 15.2: "Targets re-derived from stored records = the targets written at generation" -
    with the completed-Q target on for every search: the header names the switch, the stored
    targets and the record re-derive each other bit for bit, a cheap search has a completed-Q target
    and no visit target, and a flat game has none."""
    with torch.no_grad():
        g = generate_game(evaluator, GAMES[game], net_cfg=NET, keep_positions=True, target_config=ALL)
    written = read_targets(write_targets(tmp_path / "targets.npz", g.targets))
    record = read_game(write_game(tmp_path / "record.npz", g.record))
    assert record.header["target_config"]["completed_q"] == "all"
    rederived = build_targets(record, ALL, positions=replay_positions(record))
    assert target_differences(written, replace(rederived, reply=[])) == []  # the file keeps no reply maps
    assert target_differences(g.targets, rederived) == []
    assert len(rederived.completed_q) == len(record.decisions) > 0
    if game == "flat":
        assert all(d.flat for d in record.decisions) and all(c is None for c in rederived.completed_q)
        return
    settings = ALL.completed_q_settings(submit_share_of(record))
    for d, c in zip(record.decisions, rederived.completed_q, strict=True):
        assert c is not None and np.array_equal(c, completed_q_policy(d, settings))
        assert math.isclose(float(c.sum()), 1.0, abs_tol=1e-12)
    cheap = [i for i, d in enumerate(record.decisions) if not d.full_search]
    assert cheap and all(rederived.policy[i] is None for i in cheap), "PCR (I5) mixes full and cheap searches"
    # the default build of the same record: everything but the completed-Q target is the same
    default = build_targets(record, BASE, positions=replay_positions(record))
    assert default.completed_q == [] and target_differences(default, replace(rederived, completed_q=[])) == []


@requires_engine
def test_record_rows_carry_the_arm_forward_only(evaluator, tmp_path):
    """A generated game through the trainer under the arm: every loss finite, forward only - more
    rows carry a completed-Q target than a visit target - and an optimiser step on them refused."""
    gen = GeneratorConfig(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=31, pcr=PCRConfig(full_fraction=0.5, cheap_playouts=2))
    with torch.no_grad():
        g = generate_game(evaluator, gen, net_cfg=NET)
    record = read_game(write_game(tmp_path / "game.npz", g.record))
    model = build_model(NET, seed=1)
    trainer = Trainer(model, NET, _config(target=ALL, loss=completed_q_arm()))
    assert trainer.add_game(record) > 0
    rows = list({r.index: r for r in trainer.window.rows()}.values())
    losses = trainer.losses_on(rows)
    assert all(math.isfinite(v) for v in losses.values()) and losses["completed_q"] > 0
    with_completed_q = sum(r.targets.completed_q[r.index] is not None for r in rows)
    with_visits = sum(r.targets.policy[r.index] is not None for r in rows)
    assert with_completed_q == len(rows) > with_visits > 0
    with pytest.raises(RuntimeError, match="synthetic rows only"):
        trainer.step()
    assert optimiser_steps(model) == 0 and trainer.steps == 0
