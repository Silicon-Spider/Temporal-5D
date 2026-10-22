"""The training loop of the untrained build (register M21): records -> sampler -> targets -> losses
-> optimiser steps -> SWA, smoke-tested only.

"The training loop is written and smoke-tested in the untrained build - targets with provenance
exclusions, losses, aux heads, PCR, the QAT hook - as the pilot run's prerequisite" (register M21).
**No training runs**: a ``Trainer`` refuses more than ``MAX_STEPS`` optimiser steps (the build
brief: "a few optimiser steps on synthetic data, nothing more"); the pilot run lifts the cap, not
this build.

* ``Trainer.add_game(record)`` - one game's I15 record: its targets built (``train.targets``, the
  reply prototype's from the inputs when the network has the head), its inputs rebuilt from the
  record (``selfplay.replay_positions``), its rows weighted by the sampler (I12, 6u's arms) and
  written into the replay window (I16) as frequency copies.
* ``Trainer.step()`` - a batch drawn uniformly from the window, collated (``targets.collate_rows``),
  forward, every head's loss (``losses.compute_losses``), backward, a finite check, one optimiser
  step, then SWA's update (I17). ``step_on(rows)`` does the same on given rows (a fixed batch).
* ``Trainer.publish()`` - the network self-play receives: SWA's average (I17).
* **G8's QAT hook is a seam, not built**: ``TrainerConfig.qat`` names it and ``apply_qat`` raises -
  the fake-quantisation hooks plug in there once the precision component's quantiser (G2) and the
  pinned format exist (register G8: "prepared, not adopted ... the simulated format is pinned with
  the inference path, K/V cache included").
* **C3's conditioning-offset training is parked** with the deferred cached configuration (C2;
  BUILD_LOG "Parked"): no hook here - the dense configuration trains no refresh offsets.
* The optimiser is AdamW (the smoke test's) or NAdam (Chessformer's, beside SWA in I17's
  provenance); its constants are PLACEHOLDERS (spec issue 237).

``synthetic_record(seed, states)`` builds a well-formed I15 record over synthetic inputs, so the
whole loop runs without the engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ..engine.capi import MOVE_DTYPE
from ..net.config import NetConfig
from ..net.model import T5DNet
from ..net.state import PositionState
from .losses import LossConfig, compute_losses
from .records import Children, DecisionRecord, GameRecord
from .sampler import BucketTracker, SamplerConfig, copies, row_weights
from .selfplay import replay_positions
from .smoke import MAX_STEPS
from .swa import SWA, SWAConfig
from .targets import Row, TargetConfig, build_targets, collate_rows
from .window import ReplayWindow, WindowConfig

OPTIMISERS: tuple[str, ...] = ("adamw", "nadam")


@dataclass(frozen=True)
class TrainerConfig:
    """The loop's settings - PLACEHOLDERS unless a document gives them (issue 237)."""

    batch_size: int = 64
    optimiser: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    max_steps: int = MAX_STEPS  # no training runs: at most MAX_STEPS
    qat: bool = False  # G8's seam (``apply_qat``): not built
    target: TargetConfig | None = None  # None: the network's horizons, the rest the defaults
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    swa: SWAConfig = field(default_factory=SWAConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self) -> None:
        if self.optimiser not in OPTIMISERS:
            raise ValueError(f"optimiser must be one of {OPTIMISERS}")
        if self.batch_size < 1 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("batch_size >= 1, lr > 0, weight_decay >= 0")
        if not 1 <= self.max_steps <= MAX_STEPS:
            raise ValueError(f"the untrained build runs 1..{MAX_STEPS} optimiser steps, never a training run")


def apply_qat(model: T5DNet, cfg: TrainerConfig) -> T5DNet:
    """G8's fake-quantisation hook - the seam where it plugs in (straight-through fake quantisation
    of the trunk's matrix multiplications, attention and the K/V cache, in the deployed format;
    heads and RoPE at FP32 - section 10.2). Not built: the precision component's quantiser and the
    pinned format come first (register G2, G8)."""
    raise NotImplementedError(
        "G8's QAT hook is not built: the fake-quantisation hooks plug in here once the precision component's quantiser and "
        "the pinned inference format exist (register G8, G2; section 10.2)"
    )


class Trainer:
    """The loop (module docstring)."""

    def __init__(
        self, model: T5DNet, net_cfg: NetConfig, cfg: TrainerConfig | None = None, *, tracker: BucketTracker | None = None
    ) -> None:
        self.cfg = TrainerConfig() if cfg is None else cfg
        self.cfg.validate()
        self.net_cfg = net_cfg
        self.target_cfg = self.cfg.target or TargetConfig(short_term_horizons=tuple(net_cfg.short_term_horizons))
        if tuple(self.target_cfg.short_term_horizons) != tuple(net_cfg.short_term_horizons):
            raise ValueError("the targets' short-term horizons are the network's (D3)")
        self.model = apply_qat(model, self.cfg) if self.cfg.qat else model
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimiser == "adamw":
            self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            self.opt = torch.optim.NAdam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.swa = SWA(self.model, self.cfg.swa)
        self.window = ReplayWindow(self.cfg.window)
        self.tracker = tracker
        self.rng = np.random.default_rng(self.cfg.seed)
        self.steps = 0
        self.history: list[dict[str, float]] = []

    # -- data --

    def add_game(self, record: GameRecord, positions: Sequence | None = None) -> int:
        """One game's rows into the window (module docstring); the rows written, copies counted."""
        positions = replay_positions(record) if positions is None else list(positions)
        targets = build_targets(record, self.target_cfg, positions=positions if self.net_cfg.opp_reply else None)
        weights = row_weights(record, targets, self.cfg.sampler, self.tracker)
        n = copies(weights, int(record.header.get("seed", 0) or 0))
        for i, p in enumerate(positions):
            self.window.add(Row(getattr(p, "state", p), targets, i), int(n[i]))
        return int(n.sum())

    # -- steps --

    def step(self) -> dict[str, float]:
        """One optimiser step on a batch drawn from the window."""
        return self.step_on(self.window.sample(self.cfg.batch_size, self.rng))

    def step_on(self, rows: Sequence[Row]) -> dict[str, float]:
        """One optimiser step on ``rows`` (refused past ``max_steps``: no training runs)."""
        if self.steps >= self.cfg.max_steps:
            raise RuntimeError(f"the untrained build runs at most {self.cfg.max_steps} optimiser steps (no training runs)")
        self.model.train()
        batch, tgt = collate_rows(rows, self.net_cfg)
        self.opt.zero_grad(set_to_none=True)
        losses = compute_losses(self.model(batch), tgt, self.cfg.loss)
        values = {k: float(v.detach()) for k, v in losses.items()}
        for k, v in values.items():
            if not math.isfinite(v):
                raise FloatingPointError(f"loss {k} is not finite: {v}")
        losses["total"].backward()
        for name, p in self.model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"non-finite gradient in {name}")
        self.opt.step()
        self.steps += 1
        self.swa.update(self.model, self.steps)
        self.history.append(values)
        return values

    def publish(self) -> T5DNet:
        """The network published to self-play (I17: the weight average)."""
        net = self.swa.publish(self.model)
        return net.eval()


# --- synthetic records ---------------------------------------------------------------------------------------


def synthetic_record(seed: int, states: Sequence[PositionState], *, full_every: int = 2) -> GameRecord:
    """A well-formed I15 record over synthetic inputs (smoke testing only): one decision per state,
    each its own turn, its candidates the state's moves (and submit where legal), random search
    statistics; every ``full_every``-th decision a cheap search; White wins."""
    rng = np.random.default_rng(seed)
    decisions = []
    for i, st in enumerate(states):
        m = int(st.moves.shape[0])
        offered = bool(st.submit_possible and st.submit_legal)
        cand = list(range(m)) + ([-1] if offered else [])
        n = len(cand)
        if n == 0:
            raise ValueError(f"state {i} has no action to record")
        logits = rng.normal(size=m).astype(np.float32)
        prior = np.exp(rng.normal(size=n))
        prior /= prior.sum()
        visits = rng.integers(1, 40, size=n)
        q = rng.uniform(-0.8, 0.8, size=n)
        dd = rng.uniform(0.0, 0.2, size=n)
        children = Children(
            candidate=cand,
            legality=[0] * n,
            visits=visits,
            prior=prior,
            prior_noise_free=prior,
            q=q,
            d=dd,
            value_variance=rng.uniform(0, 0.1, size=n),
            boost=[0.0] * n,
            exploration_scale=[1.0] * n,
            forced_visits=[0] * n,
            steered_visits=[0] * n,
            child_key=rng.integers(1, 1 << 62, size=n, dtype=np.uint64),
            evaluated=[True] * n,
            network_wdl=np.stack([(1 - dd + q) / 2, dd, (1 - dd - q) / 2], axis=1),
            network_moves_left=rng.uniform(0, 30, size=n),
            network_error=rng.uniform(0, 0.2, size=n),
            lo=[-1] * n,
            hi=[1] * n,
        )
        moves = np.zeros(m, dtype=MOVE_DTYPE)
        moves["from_square"] = np.arange(m) % 64
        acting = st.acting_player if st.acting_player is not None else st.player_to_move
        decisions.append(
            DecisionRecord(
                decision=i,
                action=i,
                move_in_action=0,
                acting_player=int(acting),
                moves_made=0,
                n_boards=int(st.n_boards),
                present=0,
                submit_possible=bool(st.submit_possible),
                submit_legal=bool(st.submit_legal),
                full_search=i % full_every != full_every - 1,
                playout_cap=64,
                playouts=int(visits.sum()),
                stop="playouts",
                move_temperature=0.9,
                root_policy_temperature=1.25,
                root_noise=True,
                search_seed=i,
                root_key=i + 1,
                epoch=0,
                raw_error=0.1,
                raw_moves_left=10.0,
                raw_submit_logit=float(rng.normal()) if offered else math.nan,
                root_evaluated=True,
                root_visits=int(visits.sum()) + 1,
                root_q=float((visits * q).sum() / visits.sum()),
                root_d=float((visits * dd).sum() / visits.sum()),
                root_cpuct=1.5,
                root_best_since=int(rng.integers(1, int(visits.sum()) + 1)),
                best_child=int(np.argmax(visits)),
                played_child=int(rng.integers(0, n)),
                choice="sampled",
                verified_illegal=0,
                raw_wdl_logits=rng.normal(size=3),
                raw_value_logits=rng.normal(size=51),
                root_wdl=np.full(3, 1 / 3),
                boards=np.stack([st.timeline, st.half_turn], axis=1),
                candidates=moves,
                raw_logits=logits,
                children=children,
            )
        )
    header = {
        "record_version": 1,
        "seed": seed,
        "search": {},
        "result": {"end": "checkmate", "winner": 0, "actions": len(states), "final_active_timelines": [0]},
        "resign": {"enabled": False, "events": []},
    }
    return GameRecord(header=header, decisions=decisions)


__all__ = ["OPTIMISERS", "Trainer", "TrainerConfig", "apply_qat", "synthetic_record"]
