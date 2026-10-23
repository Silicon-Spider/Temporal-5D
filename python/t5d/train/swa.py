"""Register I17: the network published to self-play is a stochastic weight average of recent
snapshots (spec section 12.8; Lc0's training configurations, Chessformer's Leela-CF with Nadam).

Written from the method's description (Izmailov et al. 2018; AGENTS.md: no Lc0 code). Every
``every_steps`` optimiser steps the trained parameters are folded into a running average capped at
``max_snapshots`` - ``avg <- avg + (theta - avg) / min(n, max_snapshots)`` - so the average weighs
the recent snapshots and forgets old ones geometrically past the cap. ``publish`` returns a copy of
the network with the averaged parameters (buffers as trained): the network self-play receives,
whose weight hash the I15 records name. The network has no batch statistics to recompute (its
norms are RMSNorm-style gains, F5) - checked. The cadence and the cap are PLACEHOLDERS (spec issue
236); the optimiser beside it is the trainer's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SWAConfig:
    enabled: bool = True  # I17 (6z: SWA on/off)
    every_steps: int = 100  # PLACEHOLDER
    max_snapshots: int = 10  # PLACEHOLDER

    def validate(self) -> None:
        if self.every_steps < 1 or self.max_snapshots < 1:
            raise ValueError("I17: every_steps and max_snapshots are positive")


class SWA:
    """The running average of a model's parameters (module docstring)."""

    def __init__(self, model: nn.Module, cfg: SWAConfig | None = None) -> None:
        self.cfg = SWAConfig() if cfg is None else cfg
        self.cfg.validate()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                raise ValueError("SWA here assumes no batch statistics: the network has a BatchNorm")
        self.average = {name: p.detach().clone().float() for name, p in model.named_parameters()}
        self.snapshots = 0

    def update(self, model: nn.Module, step: int) -> bool:
        """Fold the model's parameters in at every ``every_steps``-th step; whether it did."""
        if not self.cfg.enabled or step % self.cfg.every_steps != 0:
            return False
        self.snapshots += 1
        n = min(self.snapshots, self.cfg.max_snapshots)
        with torch.no_grad():
            for name, p in model.named_parameters():
                avg = self.average[name]
                if self.snapshots == 1:
                    avg.copy_(p.detach().float())
                else:
                    avg.add_((p.detach().float() - avg) / n)
        return True

    def publish(self, model: nn.Module) -> nn.Module:
        """A copy of ``model`` with the averaged parameters (the model itself when SWA is off or
        nothing was averaged yet - the raw weights are then what self-play receives)."""
        if not self.cfg.enabled or self.snapshots == 0:
            return model
        out = copy.deepcopy(model)
        with torch.no_grad():
            for name, p in out.named_parameters():
                p.copy_(self.average[name].to(p.dtype))
        return out

    def state_dict(self) -> dict:
        """A checkpoint keeps the average and its count beside the raw weights."""
        return {"average": {k: v.clone() for k, v in self.average.items()}, "snapshots": self.snapshots}


__all__ = ["SWA", "SWAConfig"]
