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

* **Register I22's board-count floors** (section 12, "Stratified sampling"; section 12.8: "Board-count
  stratification sits above it: the sampler draws by range, with a floor on each range's share of
  the compute, and applies the surprise weights inside each range"; provisional, OFF -
  ``SamplerConfig.range_floors``: the pilot run plays without them, section 15.7). "A floor in
  compute, not in samples: a 170-board sample costs ten times a 44-board one."

  - **Training compute** is section 9.1's analytic dense model at the trainer's own network and
    tokenisation: one row costs ``net.cost.dense_flops`` of its token count (``row_cost``; exact
    Python ints - ``board_cost`` gives 9.69x between 44 and 170 boards at the v1's size).
  - **The floor is a constraint** (``floored_shares``, water-filling): a range whose natural share
    of the window's compute is below its floor is raised to it, and the others keep their natural
    proportions among themselves - "if its games are already large, the floors bind nothing"
    (section 12). A range with no data has no floor and takes nothing; floors that sum past the
    whole are refused, never rescaled.
  - **The draw** (``range_draw``): a range with probability in proportion to its floored share
    over its mean row cost, then a row uniformly inside it (``ReplayWindow.sample_stratified``) -
    so compute is spent in the floored shares in expectation, and surprise weighting stays inside
    each range (a row's copies are its I12 weight). When no floor binds it is the uniform draw's
    distribution.
  - ``coverage_report`` reads the window per range - never pooled (register I22).

  The share is a PLACEHOLDER (section 16: a tenth per range with data) until the pilot run
  reports the engine's size distribution (M18). A floor on a range with few rows trains on those
  rows repeatedly: built as the spec words it ("each range that has data"), with no minimum row
  count.

Every constant here not given by a document is a PLACEHOLDER (spec issue 236).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..net.cost import dense_flops
from .recipe import range_index, row_tokens
from .records import GameRecord
from .selfplay import board_range, decision_random
from .targets import GameTargets

#: 6u's arms (module docstring).
ARMS: tuple[str, ...] = ("surprise", "uniform", "raw_variance", "persistent_bucket")
#: The draw stream of a row's fractional copy (``selfplay.decision_random``'s salt).
SALT_SAMPLE = 4
#: The slack of the floors' comparisons (shares are float64 fractions of 1).
_SHARE_TOLERANCE = 1e-12


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
    #: Register I22's board-count floors (module docstring; provisional): OFF - the pilot run plays
    #: without them (section 15.7; review decision 22). Off, the draw is the window's uniform one.
    range_floors: bool = False
    floor_share: float = 0.1  # PLACEHOLDER (section 16; review decision 23): a tenth of the training compute per range with data
    #: Section 16's board-count ranges - under 50, 50-99, 100-199, 200 and up (the lower bounds
    #: after the first): the ranges the floors hold a share for, the replay window counts its rows
    #: and their compute by, and register I23's token-budget batches are cut within.
    floor_ranges: tuple[int, ...] = (50, 100, 200)

    def validate(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS} (6u)")
        if self.policy_share < 0 or self.value_share < 0 or self.policy_share + self.value_share > 1:
            raise ValueError("I12: the policy and value shares are non-negative and at most 1 together")
        if self.cheap_weight < 0 or not 0 <= self.variance_share <= 1 or self.bucket_boost < 1 or self.persist_checkpoints < 1:
            raise ValueError("cheap_weight >= 0, variance_share in [0, 1], bucket_boost >= 1, persist_checkpoints >= 1")
        bounds = tuple(self.floor_ranges)
        if any(b < 1 for b in bounds) or any(b <= a for a, b in zip(bounds, bounds[1:], strict=False)):
            raise ValueError("I22: the board-count ranges' bounds are positive and strictly increasing")
        if self.floor_share < 0 or self.floor_share * (len(bounds) + 1) > 1 + _SHARE_TOLERANCE:
            raise ValueError("I22: a floor is a non-negative share, and the floors together are at most the whole")


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
    turn = record.start_turn + d.action  # a seeded game keeps its source's turn count (I22; 0 otherwise)
    phase = sum(turn >= t for t in cfg.phase_turns)
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


# --- register I22: board-count floors (module docstring) -------------------------------------------------


def range_names(ranges: Sequence[int]) -> list[str]:
    """The board-count ranges' labels in ``range_index``'s order (``selfplay.board_range``'s):
    with (50, 100, 200), "0-49", "50-99", "100-199", "200+"."""
    ranges = tuple(ranges)
    return [board_range(lo, ranges) for lo in (0, *ranges)]


def board_cost(n_boards: int, net_cfg, cells: int = 64) -> int:
    """The training compute of one position of ``n_boards`` boards under A1's tokens: one dense
    evaluation by section 9.1's model (``net.cost.dense_flops``) at ``n_boards x (cells +
    regs_per_board) + global_regs`` tokens - an exact Python int."""
    tokens = int(n_boards) * (int(cells) + net_cfg.regs_per_board) + net_cfg.global_regs
    return int(dense_flops(tokens, net_cfg)["total"])


def row_cost(row, net_cfg) -> int:
    """One row's training compute (module docstring): section 9.1's dense model at the row's own
    token count under ``net_cfg``'s tokenisation (``recipe.row_tokens``) - an exact Python int, so
    that a window's sums cannot overflow. Forward plus backward is a constant multiple, which
    leaves every share unchanged."""
    return int(dense_flops(row_tokens(row, net_cfg), net_cfg)["total"])


def floored_shares(compute: Sequence[float], floor: float | Sequence[float]) -> np.ndarray:
    """The share of the training compute each range is trained at (module docstring: the floor is a
    constraint). ``compute``: each range's total cost in the window (0: no data); ``floor``: one
    share for every range, or one per range. A range below its floor sits exactly at it; the others
    keep their natural proportions among themselves, scaled down - which can take the next range
    under its floor, so the pass repeats (at most once per range). Floors that bind nothing return
    the natural shares untouched; a range with no data has no floor and share 0. ``ValueError``
    when no range has data, or when the floors of the ranges with data sum past the whole."""
    c = np.asarray(compute, dtype=np.float64).reshape(-1)
    if c.size == 0 or not np.isfinite(c).all() or bool((c < 0).any()):
        raise ValueError("I22: each range's compute is a finite, non-negative figure")
    has = c > 0
    if not has.any():
        raise ValueError("I22: no board-count range has data")
    f = np.where(has, np.broadcast_to(np.asarray(floor, dtype=np.float64), c.shape), 0.0)
    if bool((f < 0).any()) or float(f.sum()) > 1.0 + _SHARE_TOLERANCE:
        raise ValueError("I22: a floor is a non-negative share, and the floors together are at most the whole")
    natural = c / c.sum()
    bound = has & (natural < f - _SHARE_TOLERANCE)
    if not bound.any():
        return natural
    while True:
        free = has & ~bound
        if not free.any():
            return f / f.sum()
        scale = (1.0 - float(f[bound].sum())) / float(natural[free].sum())
        shares = np.where(bound, f, np.where(free, natural * scale, 0.0))
        low = free & (shares < f - _SHARE_TOLERANCE)
        if not low.any():
            return shares
        bound |= low


def range_probabilities(shares: Sequence[float], unit_cost: Sequence[float]) -> np.ndarray:
    """The probability of drawing each range so that compute is spent in ``shares``: in proportion
    to ``shares / unit_cost`` (``unit_cost`` the mean cost of one draw from the range - a row's),
    normalised; 0 where either is 0."""
    s = np.asarray(shares, dtype=np.float64).reshape(-1)
    u = np.asarray(unit_cost, dtype=np.float64).reshape(-1)
    if s.shape != u.shape:
        raise ValueError("one share and one unit cost per range")
    on = (s > 0) & (u > 0)
    w = np.where(on, s / np.where(on, u, 1.0), 0.0)
    if float(w.sum()) <= 0:
        raise ValueError("I22: no board-count range has data")
    return w / w.sum()


def _strata_arrays(strata: Mapping[int, tuple[int, int]]) -> tuple[list[int], np.ndarray, np.ndarray]:
    keys = sorted(k for k, (rows, _) in strata.items() if rows > 0)
    if not keys:
        raise ValueError("I22: no board-count range has data")
    rows = np.array([float(strata[k][0]) for k in keys])
    cost = np.array([float(strata[k][1]) for k in keys])  # float() of a Python int of any size
    return keys, rows, cost


def range_draw(strata: Mapping[int, tuple[int, int]], cfg: SamplerConfig) -> dict[int, float]:
    """Each range's draw probability under the floors (module docstring), for
    ``ReplayWindow.sample_stratified``. ``strata``: ``ReplayWindow.strata()`` - per range index, its
    rows and their summed cost. Only ranges with rows are returned."""
    keys, rows, cost = _strata_arrays(strata)
    p = range_probabilities(floored_shares(cost, cfg.floor_share), cost / rows)
    return {k: float(x) for k, x in zip(keys, p, strict=True)}


def coverage_report(strata: Mapping[int, tuple[int, int]], cfg: SamplerConfig) -> dict[str, dict]:
    """The window read per board-count range - never pooled (register I22): by range label, its
    ``rows``, its share of the rows (``row_share`` - the uniform draw's probability), its natural
    share of the training compute (``compute_share``), the share the floors hold it at
    (``floored_share``) and the draw probability that delivers it (``draw``) - the last two as the
    floors give them, whether or not ``range_floors`` is on."""
    keys, rows, cost = _strata_arrays(strata)
    shares = floored_shares(cost, cfg.floor_share)
    draw = range_probabilities(shares, cost / rows)
    names = range_names(cfg.floor_ranges)
    return {
        names[k]: {
            "rows": int(strata[k][0]),
            "row_share": float(rows[i] / rows.sum()),
            "compute_share": float(cost[i] / cost.sum()),
            "floored_share": float(shares[i]),
            "draw": float(draw[i]),
        }
        for i, k in enumerate(keys)
    }


__all__ = [
    "ARMS",
    "BucketTracker",
    "SamplerConfig",
    "board_cost",
    "copies",
    "coverage_report",
    "floored_shares",
    "range_draw",
    "range_index",
    "range_names",
    "range_probabilities",
    "row_cost",
    "row_weights",
    "value_surprise",
]
