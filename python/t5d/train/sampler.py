"""Which positions are trained on, and how often: register I12's surprise-weighted sampling - the
default - and ablation 6u's arms (spec sections 12.3, 12.8; register I12, I6).

"KataGo writes each full-search position into the training data with a frequency weight that is
half uniform and half proportional to the KL divergence from the network's prior to the search's
policy target, so positions where search corrected the network are seen more often, at full
weight rather than importance-scaled ... and adds a small, explicitly experimental weight for value
surprise ... Two rules here: the divergence is measured against the *pruned* target (section 12.2),
or verification steering would read as surprise; and it is additive" (section 12.8). Position
selection is safe under section 12.2 (it leaves targets untouched).

* ``row_weights(record, targets, cfg)`` - one frequency weight per decision, re-derivable from the
  record (I15). Arms (``SamplerConfig.arm``; 6u: "KataGo's surprise weighting (the default) vs
  uniform vs I6's epistemic triggers - persistent bucket, realised error, raw variance"):

  - ``"surprise"`` (I12, the default): over a game's full-search rows, ``w_i = u + p n s_i / sum s
    + v n r_i / sum r`` - ``s_i`` the KL from the raw prior to the pruned policy target
    (``targets.kl``, KataGo's direction ``KL(target || prior)``), ``r_i`` the value surprise
    ``KL(WDL target || the raw root WDL)`` (a PLACEHOLDER reading of the "small experimental"
    term), ``n`` the game's full-search rows, ``p`` the policy share (the register's half), ``v``
    the value share (a placeholder, small) and ``u = 1 - p - v``; a share whose sum is zero is
    spread uniformly. Cheap-search rows (I5) carry ``cheap_weight`` (a placeholder, 1).
  - ``"uniform"``: every row weight 1.
  - ``"raw_variance"`` (I6's arm, "expected to overfit aleatoric positions"): half uniform, half in
    proportion to the raw categorical's variance at the root.
  - ``"persistent_bucket"`` (I6's epistemic trigger, a prototype): rows whose (board-count range,
    phase) bucket's realised error - the squared gap between the raw root value and the game's
    result - stayed above ``bucket_threshold`` at each of the last ``persist_checkpoints``
    checkpoints (``BucketTracker``) are boosted by ``bucket_boost``; weights renormalised to a mean
    of 1 per game. I6's realised-error arm itself is superseded by I12 (the register) and is not an
    arm here.

* ``copies(weights, seed)`` - "at full weight rather than importance-scaled": a row is written
  ``floor(w)`` times and once more with probability ``frac(w)``, the draw seeded by the game's seed
  and the decision, so every copy trains at loss weight 1.

Every constant here not given by a document is a PLACEHOLDER (spec issue 236).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .records import GameRecord
from .selfplay import board_range, decision_random
from .targets import GameTargets

#: 6u's arms (module docstring).
ARMS: tuple[str, ...] = ("surprise", "uniform", "raw_variance", "persistent_bucket")
#: The draw stream of a row's fractional copy (``selfplay.decision_random``'s salt).
SALT_SAMPLE = 4


@dataclass(frozen=True)
class SamplerConfig:
    arm: str = "surprise"
    policy_share: float = 0.5  # I12: "half ... proportional to the KL divergence"
    value_share: float = 0.05  # I12: "a small, explicitly experimental weight for value surprise" - PLACEHOLDER
    cheap_weight: float = 1.0  # PLACEHOLDER: a cheap search's row (no policy target, I5)
    variance_share: float = 0.5  # the raw-variance arm's - PLACEHOLDER
    bucket_ranges: tuple[int, ...] = (50, 100, 200)  # board-count buckets (6b's ranges) - PLACEHOLDER
    phase_turns: tuple[int, ...] = (20, 60)  # phase buckets by turn - PLACEHOLDER
    bucket_threshold: float = 0.25  # mean squared realised error - PLACEHOLDER
    persist_checkpoints: int = 3  # PLACEHOLDER
    bucket_boost: float = 2.0  # PLACEHOLDER

    def validate(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS} (6u)")
        if self.policy_share < 0 or self.value_share < 0 or self.policy_share + self.value_share > 1:
            raise ValueError("I12: the policy and value shares are non-negative and at most 1 together")
        if self.cheap_weight < 0 or not 0 <= self.variance_share <= 1 or self.bucket_boost < 1 or self.persist_checkpoints < 1:
            raise ValueError("cheap_weight >= 0, variance_share in [0, 1], bucket_boost >= 1, persist_checkpoints >= 1")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def value_surprise(record: GameRecord, targets: GameTargets) -> np.ndarray:
    """``KL(WDL target || raw root WDL)`` per decision (0 where there is no WDL target)."""
    out = np.zeros(len(record.decisions), dtype=np.float64)
    for i, d in enumerate(record.decisions):
        t = targets.wdl[i]
        if not np.isfinite(t).all():
            continue
        p = _softmax(d.raw_wdl_logits)
        keep = t > 0
        out[i] = float((t[keep] * (np.log(t[keep]) - np.log(p[keep]))).sum())
    return out


def _bucket(record: GameRecord, i: int, cfg: SamplerConfig) -> tuple[str, int]:
    d = record.decisions[i]
    phase = sum(d.action >= t for t in cfg.phase_turns)
    return board_range(d.n_boards, cfg.bucket_ranges), phase


@dataclass
class BucketTracker:
    """I6's persistent-bucket trigger (a prototype): each (board-count range, phase) bucket's mean
    realised error per checkpoint - the squared gap between the raw root value (W - L) and the
    game's result, the acting player's view."""

    cfg: SamplerConfig = field(default_factory=SamplerConfig)
    history: dict[tuple[str, int], list[float]] = field(default_factory=dict)

    def add_checkpoint(self, games: Sequence[tuple[GameRecord, GameTargets]]) -> None:
        sums: dict[tuple[str, int], list[float]] = {}
        for record, targets in games:
            for i, d in enumerate(record.decisions):
                t = targets.wdl[i]
                if not np.isfinite(t).all():
                    continue
                p = _softmax(d.raw_wdl_logits)
                err = (float(p[0] - p[2]) - float(t[0] - t[2])) ** 2
                sums.setdefault(_bucket(record, i, self.cfg), []).append(err)
        for key in set(self.history) | set(sums):
            values = sums.get(key)
            self.history.setdefault(key, []).append(float(np.mean(values)) if values else math.nan)

    def triggered(self, key: tuple[str, int]) -> bool:
        last = self.history.get(key, [])[-self.cfg.persist_checkpoints :]
        return len(last) == self.cfg.persist_checkpoints and all(math.isfinite(x) and x > self.cfg.bucket_threshold for x in last)


def row_weights(
    record: GameRecord, targets: GameTargets, cfg: SamplerConfig | None = None, tracker: BucketTracker | None = None
) -> np.ndarray:
    """One frequency weight per decision of a game (module docstring)."""
    cfg = SamplerConfig() if cfg is None else cfg
    cfg.validate()
    n = len(record.decisions)
    if cfg.arm == "uniform" or n == 0:
        return np.ones(n, dtype=np.float64)
    full = targets.full_search.astype(bool)
    if cfg.arm == "raw_variance":
        var = np.array([_categorical_variance(d.raw_value_logits) for d in record.decisions])
        return _mix(np.ones(n, dtype=bool), var, cfg.variance_share)
    if cfg.arm == "persistent_bucket":
        if tracker is None:
            raise ValueError("the persistent-bucket arm reads a BucketTracker's checkpoints")
        w = np.array([cfg.bucket_boost if tracker.triggered(_bucket(record, i, cfg)) else 1.0 for i in range(n)])
        return w / w.mean()
    w = np.full(n, cfg.cheap_weight, dtype=np.float64)
    m = int(full.sum())
    if m == 0:
        return w
    s = np.where(full, targets.kl, 0.0)
    finite = np.isfinite(s)
    s = np.where(finite, s, np.max(s[finite], initial=0.0))  # a prior with no mass on a target entry: the game's largest
    r = np.where(full, value_surprise(record, targets), 0.0)
    uniform = 1.0 - cfg.policy_share - cfg.value_share
    part = np.full(m, uniform, dtype=np.float64)
    for share, x in ((cfg.policy_share, s[full]), (cfg.value_share, r[full])):
        total = float(x.sum())
        part += share * m * x / total if total > 0 else share
    w[full] = part
    return w


def _mix(rows: np.ndarray, x: np.ndarray, share: float) -> np.ndarray:
    total = float(x[rows].sum())
    w = np.ones(rows.shape[0], dtype=np.float64)
    if total > 0:
        w[rows] = (1.0 - share) + share * rows.sum() * x[rows] / total
    return w


def _categorical_variance(logits: np.ndarray) -> float:
    p = _softmax(logits)
    v = p.shape[0]
    c = (2.0 * np.arange(v) - (v - 1)) / (v - 1)
    m = float((p * c).sum())
    return float((p * (c - m) ** 2).sum())


def copies(weights: np.ndarray, game_seed: int) -> np.ndarray:
    """Each row's copy count: ``floor(w)``, plus one with probability ``frac(w)`` (seeded by the
    game and the decision) - every copy at full loss weight (I12: not importance-scaled)."""
    out = np.floor(weights).astype(np.int64)
    for i, w in enumerate(weights.tolist()):
        frac = w - math.floor(w)
        if frac > 0 and decision_random(game_seed, i, SALT_SAMPLE) < frac:
            out[i] += 1
    return out


__all__ = ["ARMS", "BucketTracker", "SamplerConfig", "copies", "row_weights", "value_surprise"]
