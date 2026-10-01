"""Training smoke test: at most three optimiser steps on synthetic data.

Checks that every head's loss is finite, that gradients reach the parameters, and that
an optimiser step changes them. It is not training - the build brief allows a few
optimiser steps on synthetic data and no training runs (spec section 15).

Run: ``python -m t5d.train.smoke`` (from ``python/``).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch

from ..net.config import NetConfig
from ..net.heads import effort_bin, value_to_twohot
from ..net.model import T5DNet, build_model
from ..net.state import Batch, collate
from ..net.synthetic import synthetic_states
from .losses import LossConfig, Targets, compute_losses

MAX_STEPS = 3


@dataclass
class SmokeReport:
    steps: int
    losses: list[dict[str, float]]
    param_change: float  # L2 norm of the total parameter change
    changed_fraction: float  # fraction of parameter tensors that changed
    unchanged: list[str] = field(default_factory=list)  # names of the tensors that did not


def synthetic_targets(batch: Batch, cfg: NetConfig, generator: torch.Generator) -> Targets:
    """Random but well-formed targets for a batch (smoke testing only)."""
    bsz, length = batch.piece.shape
    g = generator

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    outcome = torch.randint(-1, 2, (bsz,), generator=g).float()
    visits = rand(bsz, batch.moves.shape[1]) * batch.move_valid
    policy = visits / visits.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    n_opp = batch.opp_move_valid.sum(dim=-1)
    opp = torch.where(n_opp > 0, (rand(bsz) * n_opp.clamp(min=1)).long(), torch.full((bsz,), -1))
    effort_visits = torch.randint(1, 4096, (bsz,), generator=g)
    converged = rand(bsz) < 0.8
    return Targets(
        value=value_to_twohot(outcome, cfg.value_bins),
        policy=policy,
        wdl=(1 - outcome).long(),  # +1 -> 0 win, 0 -> 1 draw, -1 -> 2 loss
        moves_left=torch.randint(0, 200, (bsz,), generator=g).float(),
        opp_reply=opp,
        short_value=value_to_twohot(rand(bsz) * 2 - 1, cfg.value_bins),
        board=(rand(bsz, batch.board_valid.shape[1], 3) < 0.3).float(),
        square=(rand(bsz, length, cfg.n_square_labels) < 0.2).float(),
        effort=effort_bin(effort_visits, converged, cfg.effort_bins),
    )


def smoke_train(
    cfg: NetConfig | None = None,
    steps: int = MAX_STEPS,
    seed: int = 0,
    board_counts: tuple[int, ...] = (1, 3, 6),
    lr: float = 1e-4,
    loss_cfg: LossConfig | None = None,
    model: T5DNet | None = None,
) -> SmokeReport:
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"the smoke test runs 1..{MAX_STEPS} optimiser steps, never more")
    cfg = NetConfig() if cfg is None else cfg
    model = build_model(cfg, seed=seed) if model is None else model
    model.train()
    batch = collate(synthetic_states(seed, list(board_counts)))
    gen = torch.Generator().manual_seed(seed)
    targets = synthetic_targets(batch, cfg, gen)
    names = [name for name, _ in model.named_parameters()]
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history: list[dict[str, float]] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = compute_losses(model(batch), targets, loss_cfg)
        values = {name: float(value.detach()) for name, value in losses.items()}
        for name, value in values.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"loss {name} is not finite: {value}")
        losses["total"].backward()
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"non-finite gradient in {name}")
        opt.step()
        history.append(values)
    deltas = [(p.detach() - b) for p, b in zip(model.parameters(), before, strict=True)]
    change = math.sqrt(sum(float(d.pow(2).sum()) for d in deltas))
    moved = [bool(d.abs().max() > 0) for d in deltas]
    changed = sum(moved) / max(1, len(deltas))
    unchanged = [name for name, m in zip(names, moved, strict=True) if not m]
    return SmokeReport(steps=steps, losses=history, param_change=change, changed_fraction=changed, unchanged=unchanged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32", action="store_true", help="FP32 trunk instead of the default BF16")
    args = parser.parse_args(argv)
    cfg = NetConfig(trunk_dtype="fp32" if args.fp32 else "bf16")
    report = smoke_train(cfg, steps=args.steps, seed=args.seed)
    for i, losses in enumerate(report.losses):
        parts = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"step {i}: {parts}")
    print(f"parameter change (L2): {report.param_change:.4e}; tensors changed: {report.changed_fraction:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
