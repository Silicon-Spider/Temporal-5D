"""Training smoke test: at most three optimiser steps on synthetic data.

Checks that every head's loss is finite, that gradients reach the parameters, and that
an optimiser step changes them. It is not training - the build brief allows a few
optimiser steps on synthetic data and no training runs (spec section 15).

Run: ``python -m t5d.train.smoke`` (from ``python/``).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace

import torch

from ..net.config import NetConfig
from ..net.heads import effort_bin
from ..net.model import T5DNet, build_model
from ..net.state import Batch, PositionState, collate
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
    """Random but well-formed targets for a batch (smoke testing only). Every third position is a
    cheap search (no policy or categorical target, section 12.5); one short-term horizon of the
    last position has no target (NaN)."""
    bsz, length = batch.piece.shape
    g = generator

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    outcome = torch.randint(-1, 2, (bsz,), generator=g).float()
    visits = rand(bsz, batch.moves.shape[1]) * batch.move_valid
    _, submit_legal = batch.submit_mask()
    submit_visits = rand(bsz) * submit_legal  # D8: the submit child's visits, where it is a legal candidate
    total = (visits.sum(dim=-1) + submit_visits).clamp(min=1e-9)
    policy = visits / total[:, None]
    n_opp = batch.opp_move_valid.sum(dim=-1)
    opp = torch.where(n_opp > 0, (rand(bsz) * n_opp.clamp(min=1)).long(), torch.full((bsz,), -1))
    effort_visits = torch.randint(1, 4096, (bsz,), generator=g)
    converged = rand(bsz) < 0.8
    full = torch.arange(bsz) % 3 != 2
    root_value = torch.where(full, rand(bsz) * 2 - 1, torch.full((bsz,), math.nan))  # D3: full searches only
    short_term = rand(bsz, len(cfg.short_term_horizons)) * 2 - 1
    short_term[-1, -1] = math.nan
    q, d = rand(bsz) * 2 - 1, rand(bsz) * 0.5
    d = torch.minimum(d, 1 - q.abs())
    return Targets(
        policy=policy,
        wdl=(1 - outcome).long(),  # +1 -> 0 win, 0 -> 1 draw, -1 -> 2 loss
        root_value=root_value,
        short_term=short_term,
        moves_left=torch.randint(0, 200, (bsz,), generator=g).float(),
        opp_reply=opp,
        board=(rand(bsz, batch.board_valid.shape[1], 3) < 0.3).float(),
        square=(rand(bsz, length, cfg.n_square_labels) < 0.2).float(),
        effort=effort_bin(effort_visits, converged, cfg.effort_bins),
        submit=submit_visits / total,
        full_search=full,
        root_wdl=torch.stack([(1 - d + q) / 2, d, (1 - d - q) / 2], dim=-1),  # the search's (q, d) as a WDL
    )


def with_actions(states: list[PositionState]) -> list[PositionState]:
    """The states with register D8's action varied (synthetic, for smoke testing): every other
    one is read as an action whose present has shifted - the acting player the present's
    opponent, submitting possible, and legal in every second such state - so that the
    acting-player input takes both values and the submit logit is offered, masked and live."""
    return [
        replace(s, acting_player=1 - s.player_to_move, submit_possible=True, submit_legal=(i // 2) % 2 == 0) if i % 2 else s
        for i, s in enumerate(states)
    ]


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
    batch = collate(with_actions(synthetic_states(seed, list(board_counts))))
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
