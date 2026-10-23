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

#: The optimiser steps a model has taken (an attribute of the model, so copies of it - the
#: SWA-published network among them - carry the count): the cap is per model, never per trainer
#: or per call, so a second trainer or another smoke run cannot add steps (BUILD_LOG issue 237(a)).
STEPS_ATTR = "t5d_optimiser_steps"


def optimiser_steps(model: torch.nn.Module) -> int:
    """The optimiser steps ``model`` has taken in this build (``STEPS_ATTR``)."""
    return int(getattr(model, STEPS_ATTR, 0))


def check_step(model: torch.nn.Module) -> None:
    """Refuse an optimiser step past ``MAX_STEPS`` on ``model`` - no training runs."""
    if optimiser_steps(model) >= MAX_STEPS:
        raise RuntimeError(f"the untrained build runs at most {MAX_STEPS} optimiser steps on a model (no training runs)")


def record_step(model: torch.nn.Module) -> None:
    """Count one optimiser step taken on ``model``."""
    setattr(model, STEPS_ATTR, optimiser_steps(model) + 1)


@dataclass
class SmokeReport:
    steps: int
    losses: list[dict[str, float]]
    param_change: float  # L2 norm of the total parameter change
    changed_fraction: float  # fraction of parameter tensors that changed
    unchanged: list[str] = field(default_factory=list)  # names of the tensors that did not


def synthetic_targets(batch: Batch, cfg: NetConfig, generator: torch.Generator, *, flat_row: int | None = None) -> Targets:
    """Random but well-formed targets for a batch (smoke testing only). Every third position is a
    cheap search (no policy or categorical target, section 12.5); one short-term horizon of the
    last position has no target (NaN). With ``flat_row``, that position is a decision of D2's flat
    form (issue 242): a random distribution over each of its head boards' options - the moves leaving
    it, leaving it - so that the flat policy loss reaches D2's leave head (``smoke_train`` adds one)."""
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
    # D4's reply prototype: a distribution over each position's ``opp_moves`` (synthetic, not mapped).
    opp = rand(*batch.opp_move_valid.shape) * batch.opp_move_valid
    opp = opp / opp.sum(dim=-1, keepdim=True).clamp(min=1e-9)
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
        board=(rand(bsz, batch.board_valid.shape[1], 3) < 0.3).float(),
        square=(rand(bsz, length, cfg.n_square_labels) < 0.2).float(),
        effort=effort_bin(effort_visits, converged, cfg.effort_bins),
        submit=submit_visits / total,
        full_search=full,
        root_wdl=torch.stack([(1 - d + q) / 2, d, (1 - d - q) / 2], dim=-1),  # the search's (q, d) as a WDL
        opp_reply=opp,
        **({} if flat_row is None else _synthetic_flat(batch, rand, flat_row)),
    )


def _synthetic_flat(batch: Batch, rand, row: int) -> dict:
    """``Targets``' flat fields with position ``row`` a flat decision (``synthetic_targets``)."""
    bsz, m_max = batch.moves.shape[0], batch.moves.shape[1]
    if row >= bsz:
        return {}
    src_board = batch.board[row].gather(0, batch.moves[row, :, 0].clamp(min=0))
    entries: list[tuple[int, int]] = []
    for b in torch.nonzero(batch.board_valid[row] & batch.board_head[row]).flatten().tolist():
        own = [i for i in range(m_max) if bool(batch.move_valid[row, i]) and int(src_board[i]) == b]
        entries += [(i, b) for i in own] + [(m_max + b, b)]
    k = len(entries)
    option = torch.full((bsz, k), -1, dtype=torch.long)
    group = torch.full((bsz, k), -1, dtype=torch.long)
    target = torch.zeros(bsz, k)
    if k:
        option[row] = torch.tensor([o for o, _ in entries])
        group[row] = torch.tensor([g for _, g in entries])
        w = rand(k) + 0.05
        for g in set(group[row].tolist()):
            on = group[row] == g
            target[row, on] = w[on] / w[on].sum()
    rows = torch.zeros(bsz, dtype=torch.bool)
    rows[row] = k > 0
    return dict(flat_rows=rows, flat_option=option, flat_group=group, flat_target=target)


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
    # One more position, a decision of D2's flat form: the leave head's loss (issue 242).
    batch = collate(with_actions(synthetic_states(seed, [*board_counts, 3])))
    gen = torch.Generator().manual_seed(seed)
    targets = synthetic_targets(batch, cfg, gen, flat_row=len(board_counts))
    names = [name for name, _ in model.named_parameters()]
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history: list[dict[str, float]] = []
    for _ in range(steps):
        check_step(model)
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
        record_step(model)
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
