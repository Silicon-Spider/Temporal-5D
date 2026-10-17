"""The training smoke test: at most three optimiser steps, finite losses, parameters move."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from t5d.net import NetConfig, build_model, collate, synthetic_state, synthetic_states
from t5d.net.structure import TYPED_INDEX
from t5d.train.losses import LossConfig, compute_losses
from t5d.train.smoke import MAX_STEPS, smoke_train, synthetic_targets, with_actions

pytestmark = pytest.mark.grad


@pytest.mark.parametrize("trunk_dtype", ["bf16", "fp32"])
def test_smoke_three_steps(trunk_dtype):
    report = smoke_train(NetConfig(trunk_dtype=trunk_dtype), steps=MAX_STEPS, seed=0)
    assert report.steps == 3 and len(report.losses) == 3
    for step in report.losses:
        assert all(math.isfinite(v) for v in step.values())
    assert report.param_change > 0 and report.changed_fraction == 1.0


@pytest.mark.parametrize(
    "flags",
    [
        dict(rope_per_head=True, policy_promotion="offset", disp_register_pairs="shared"),
        dict(global_register_pairs="anchored", board_register_spatial="position_free", aux_global_pool="global_registers"),
        dict(norm_placement="post"),
        dict(ffn_activation="squared_relu"),
        dict(displacement_bias=False),
    ],
    ids=["b5_learned_offset_b15_shared", "anchored_regs_free", "post_norm", "squared_relu", "no_table"],
)
def test_smoke_with_the_options_on(flags):
    """Every option's parameters receive gradient and move: B15's table, a learned register
    coordinate or a promotion offset that the forward pass ignored would stay put. Ablation
    6y's arms - F5's post-norm, F6's squared ReLU (in the trunk only: the heads keep Mish) -
    train finite, every tensor moving."""
    report = smoke_train(NetConfig(trunk_dtype="fp32", **flags), steps=1, seed=0, board_counts=(1, 3, 6, 9))
    assert all(math.isfinite(v) for v in report.losses[0].values())
    assert report.changed_fraction == 1.0


def test_smoke_refuses_more_steps():
    with pytest.raises(ValueError):
        smoke_train(steps=4)


def test_every_loss_is_finite_and_reaches_its_head():
    cfg = NetConfig(trunk_dtype="fp32")
    model = build_model(cfg, seed=1)
    batch = collate(synthetic_states(1, [2, 5]))
    targets = synthetic_targets(batch, cfg, torch.Generator().manual_seed(1))
    for reduction in ("mean", "sum"):  # register D6's open knob
        losses = compute_losses(model(batch), targets, LossConfig(board_reduction=reduction))
        assert all(torch.isfinite(v) and v > 0 for v in losses.values()), losses
    heads = {
        "policy": model.policy,
        "value": model.value,
        "wdl": model.aux.wdl,
        "moves_left": model.aux.moves_left,
        "opp_reply": model.aux.opp_reply,
        "short_value": model.aux.short_value,
        "board": model.aux.board,
        "square": model.aux.square,
        "effort": model.aux.effort,
    }
    for name, head in heads.items():
        model.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), targets)
        losses[name].backward()
        grads = [p.grad for p in head.parameters()]
        assert all(g is not None and g.abs().sum() > 0 for g in grads), name
        # ... and through the trunk to the embeddings.
        assert model.embed.piece.weight.grad.abs().sum() > 0, name


def _bce_board(out, tgt):
    return F.binary_cross_entropy_with_logits(out.board_logits, tgt.board, reduction="none").sum(-1)


def _bce_square(out, tgt):
    return F.binary_cross_entropy_with_logits(out.square_logits, tgt.square, reduction="none").mean(-1)


def test_board_reduction_knob():
    """D6 ("sum-vs-mean loss reduction over boards"): a per-board loss and a per-square
    loss averaged over each board's squares are both reduced over a position's boards by
    the knob, and positions then weigh equally - recomputed board by board here."""
    cfg = NetConfig(trunk_dtype="fp32")
    model = build_model(cfg, seed=2).eval()
    states = [synthetic_state(4, 2), synthetic_state(5, 9)]
    batch = collate(states)
    with torch.no_grad():
        out = model(batch)
    tgt = synthetic_targets(batch, cfg, torch.Generator().manual_seed(2))
    bce_board, bce_square = _bce_board(out, tgt), _bce_square(out, tgt)
    for reduction in ("mean", "sum"):
        losses = compute_losses(out, tgt, LossConfig(board_reduction=reduction))
        expected_board, expected_square = [], []
        for i, state in enumerate(states):
            per_board = bce_board[i, : state.n_boards]
            per_board_squares = torch.stack([bce_square[i, b * 66 : b * 66 + 64].mean() for b in range(state.n_boards)])
            reduce = torch.mean if reduction == "mean" else torch.sum
            expected_board.append(reduce(per_board))
            expected_square.append(reduce(per_board_squares))
        assert torch.allclose(losses["board"], torch.stack(expected_board).mean(), rtol=1e-5), reduction
        assert torch.allclose(losses["square"], torch.stack(expected_square).mean(), rtol=1e-5), reduction
    # On a single N-board position, summing is N times averaging.
    single = collate([states[1]])
    with torch.no_grad():
        out1 = model(single)
    tgt1 = synthetic_targets(single, cfg, torch.Generator().manual_seed(3))
    mean = compute_losses(out1, tgt1, LossConfig(board_reduction="mean"))
    total = compute_losses(out1, tgt1, LossConfig(board_reduction="sum"))
    for name in ("board", "square"):
        assert torch.allclose(total[name], 9 * mean[name], rtol=1e-5), name


@pytest.mark.parametrize(("form", "n_new"), [("relative_cells", 4), ("board_bilinear", 7)])
def test_smoke_covers_stage_n2_parameters(form, n_new):
    """Every tensor stage N2 adds - B11's typed-edge scalars, B12's smolgen projections and
    its cell table (relative cells) or bilinear forms (board_bilinear) - receives gradient
    and moves in one optimiser step, with the routing table's masks (B10) and B15's
    displacement table in the same forward. On every head that
    can see them - the dense heads and the relations head, whose mask opens every typed
    relation - each typed scalar whose relation occurs in the batch moves, in both
    orientations. (On the other masked heads a type the mask never opens keeps its scalar:
    defence pairs, for one, are never move-graph pairs - see structure.py.)"""
    flags = dict(routing_table=True, typed_edge_bias=True, typed_edge_directed=True, board_pair_smolgen=True, smolgen_form=form)
    model = build_model(NetConfig(trunk_dtype="fp32", **flags), seed=0)
    table = model.typed_edges.table.detach().clone()
    report = smoke_train(model.cfg, steps=1, seed=0, board_counts=(1, 3, 6, 9), model=model)
    assert all(math.isfinite(v) for v in report.losses[0].values())
    assert report.unchanged == [] and report.changed_fraction == 1.0, report.unchanged
    new = [n for n, _ in model.named_parameters() if n.startswith(("typed_edges.", "smolgen."))]
    # typed table; smolgen context weight and bias, and the cell table - or board weight and bias, left, right
    assert len(new) == n_new, new
    batch = collate(synthetic_states(0, [1, 3, 6, 9]))
    present = torch.unique(batch.edges[..., 2][batch.edge_valid])
    types = sorted({TYPED_INDEX[int(r)] for r in present if TYPED_INDEX[int(r)] >= 0})
    assert types, "the smoke batch carries typed relations"
    moved = (model.typed_edges.table.detach() - table).abs() > 0
    seeing = [h for h, role in enumerate(model.cfg.head_roles) if role in ("dense", "relations")]
    assert moved[seeing][:, types].all(), "every present type's scalars on every head that sees them, both orientations"


@pytest.mark.parametrize("form", ["linear", "pointer"])
def test_the_policy_loss_covers_the_submit_action(form):
    """Register D8 (stage N5): the policy's target is one distribution over the moves and the
    submit action; its loss is the cross-entropy against the joint log-softmax - an illegal or
    absent submit carries neither probability nor target - and it reaches the submit readout
    and the acting-player embedding."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, submit_head=form)
    model = build_model(cfg, seed=4)
    batch = collate(with_actions(synthetic_states(4, [2, 4, 3, 5])))
    assert batch.submit_possible.tolist() == [False, True, False, True] and batch.submit_legal.tolist() == [False, True, False, False]
    targets = synthetic_targets(batch, cfg, torch.Generator().manual_seed(4))
    total = targets.policy.sum(dim=-1) + targets.submit
    assert torch.allclose(total, torch.ones(4), atol=1e-6)
    assert targets.submit[[0, 2, 3]].tolist() == [0.0, 0.0, 0.0] and float(targets.submit[1]) > 0
    out = model(batch)
    target = torch.cat([targets.policy, targets.submit[:, None]], dim=-1)
    logp = out.action_log_probs()
    per = [-(target[i] * logp[i]).masked_fill(~out.action_valid()[i], 0.0).sum() for i in range(4)]
    losses = compute_losses(out, targets)
    assert torch.allclose(losses["policy"], torch.stack(per).mean(), rtol=1e-5)
    losses["policy"].backward()
    if form == "linear":
        assert model.submit.out.weight.grad.abs().sum() > 0
    assert model.embed.acting.weight.grad.abs().sum() > 0
    assert model.policy.src.weight.grad.abs().sum() > 0
