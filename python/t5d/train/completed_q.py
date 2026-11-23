"""Register I21: the completed-Q policy target, re-derived from I15's records (spec section 12.2).

Gumbel AlphaZero's policy target (Danihelka, Guez, Schrittwieser and Silver, "Policy improvement
by planning with Gumbel", ICLR 2022): ``pi' = softmax(logit + sigma(completed q))`` over the
root's actions - the network's raw logits plus each action's value in logit units. A visited
child's value is its backed-up one; an action the search never visited takes the root's mixed
value, the network's own value blended with the prior-weighted mean of the visited children's,
weighted by the visits::

    v_mix    = (v_net + sum N x (sum_visited P q / sum_visited P)) / (1 + sum N)
    sigma(q) = (c_visit + max N) x c_scale x q

**A prototype arm, not the builder's default.** ``targets.build_targets`` writes the pruned-visit
target (I2, I5) and, behind ``TargetConfig.completed_q`` (off by default), this one beside it -
never in its place. ``losses.completed_q_arm`` is the loss arm that trains the main policy head on
it (weight 0 by default), and ``target_arms`` holds the arms' configurations, the record-level
comparison of the two targets and the held-out yardstick. **The ranking itself is not built**:
per-arm trainings on a shared dataset (M24) wait for the pilot run's data. The arm can be ranked
from records as they are: this target reads nothing the records do not already hold - the raw
logits (``raw_logits``, ``raw_submit_logit``), the root's raw WDL (``raw_wdl_logits``) and each
root child's visits and backed-up value (``Children.visits``, ``Children.q``). That is the claim
section 12.8 makes of the record format, held here by tests.

**What it does not read is the point** (section 12.2). Not the search's prior, which carries the
root noise and the asset map's boosts; not the steered or forced counts. The visits enter in
three places, none of them a share of the target - which children have a value at all, sigma's
scale (the largest count), and the weight the visited children's mean takes in the mixed value -
so a steered visit at the root reveals a child's value and little else: neither exclusion nor
provenance (I2) has anything to act on. The values are the network's own, backed up, so the
source rule (D5) holds; steering deeper in the tree can still shape them, the residual 6e
measures for every search-value target.

Readings, each a named PLACEHOLDER of ``CompletedQConfig``:

* ``c_visit`` 50 and ``c_scale`` 1: the paper's values for Go and chess.
* ``value_range``: the paper's values lie in [0, 1]; the search's q is W - L in [-1, 1].
  ``"unit"`` (the default) maps q to (q + 1) / 2 before sigma; ``"signed"`` uses q as it is,
  which doubles the scale.
* The root's own value ``v_net`` is W - L of the network's raw WDL at the root
  (``root_network_value``); where the record holds none, the recorded root search value.
* Every value is the acting player's view (D8), as ``Children.q`` and ``root_q`` are.
* ``submit_share``: the search's placeholder prior for a legal submit the network gave no logit
  (``targets.raw_prior``); 0 by default, as there.
* Full and cheap searches alike. The target improves on the prior at any simulation count, which
  is why I5's cheap searches may emit it (6l); which decisions to use is the caller's choice - in
  the builder, ``TargetConfig.completed_q``: the full searches, the cheap ones, or all.
* A search that visited no child: the target is the raw prior (there is no value to complete
  with). The builder emits it, so that which decisions emit is a function of the search class
  alone; it can arise only at a cap of a playout or two.
* D2's flat form: a flat decision's root children are whole turns, and this target's reading over
  them is open (register part K) - ``completed_q_policy`` returns None there: the builder writes
  no target and the loss skips the row.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .records import DecisionRecord
from .targets import raw_prior

#: How the search's q (W - L, in [-1, 1]) enters sigma: mapped onto [0, 1], or as it is.
VALUE_RANGES: tuple[str, ...] = ("unit", "signed")


@dataclass(frozen=True)
class CompletedQConfig:
    """The completed-Q target's settings - each a PLACEHOLDER (register I21; module docstring)."""

    c_visit: float = 50.0
    c_scale: float = 1.0
    value_range: str = "unit"
    submit_share: float = 0.0

    def validate(self) -> None:
        if not (math.isfinite(self.c_visit) and self.c_visit >= 0.0):
            raise ValueError("c_visit is finite and >= 0")
        if not (math.isfinite(self.c_scale) and self.c_scale >= 0.0):
            raise ValueError("c_scale is finite and >= 0")
        if self.value_range not in VALUE_RANGES:
            raise ValueError(f"value_range must be one of {VALUE_RANGES}")
        if not 0.0 <= self.submit_share <= 1.0:
            raise ValueError("submit_share in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def root_network_value(d: DecisionRecord) -> float:
    """The network's own value of the root, W - L of its raw WDL there (the acting player's view);
    the recorded root search value where the record holds no raw WDL."""
    z = d.raw_wdl_logits.astype(np.float64)
    if not np.isfinite(z).all():
        return float(d.root_q)
    e = np.exp(z - z.max())
    p = e / e.sum()
    return float(p[0] - p[2])


def mixed_value(prior: np.ndarray, q: np.ndarray, visits: np.ndarray, v_net: float) -> float:
    """The paper's ``v_mix`` (module docstring): ``v_net`` blended with the prior-weighted mean of
    the visited actions' values, weighted by the visits; ``v_net`` itself where nothing was visited
    (or the visited actions carry no prior)."""
    visited = visits > 0
    total = float(visits.sum())
    mass = float(prior[visited].sum())
    if total <= 0.0 or mass <= 0.0:
        return float(v_net)
    mean = float((prior[visited] * q[visited]).sum()) / mass
    return (float(v_net) + total * mean) / (1.0 + total)


def sigma(q: np.ndarray, max_visits: float, cfg: CompletedQConfig) -> np.ndarray:
    """The paper's monotone transform of a value into logit units: ``(c_visit + max N) c_scale q``,
    q first mapped onto [0, 1] under ``value_range="unit"``."""
    v = (q + 1.0) / 2.0 if cfg.value_range == "unit" else q
    return (cfg.c_visit + float(max_visits)) * cfg.c_scale * v


def action_statistics(d: DecisionRecord) -> tuple[np.ndarray, np.ndarray]:
    """Each action's visits and backed-up value, over the candidate moves and submit (``M + 1``,
    float64; D8): the root children's, placed at their candidates. An action with no child, or
    with an unvisited one, has 0 visits and a NaN value."""
    m = d.move_count
    visits = np.zeros(m + 1, dtype=np.float64)
    q = np.full(m + 1, np.nan, dtype=np.float64)
    c = d.children
    for k, n, value in zip(c.candidate.tolist(), c.visits.tolist(), c.q.tolist(), strict=True):
        if k < -1:
            raise ValueError("a whole-turn child has no action of its own (D2's flat form)")
        i = k if k >= 0 else m
        visits[i] += float(n)
        if n > 0:
            q[i] = float(value)
    return visits, q


def completed_values(d: DecisionRecord, cfg: CompletedQConfig | None = None) -> tuple[np.ndarray, float] | None:
    """Each action's completed value (``M + 1``, float64) and the mixed value the unvisited ones
    take; None at a flat decision, or where the root offers no action."""
    cfg = cfg or CompletedQConfig()
    cfg.validate()
    if d.flat:
        return None
    prior = raw_prior(d, cfg.submit_share)
    if not (prior > 0).any():
        return None
    visits, q = action_statistics(d)
    v_mix = mixed_value(prior, q, visits, root_network_value(d))
    return np.where(visits > 0, q, v_mix), v_mix


def completed_q_policy(d: DecisionRecord, cfg: CompletedQConfig | None = None) -> np.ndarray | None:
    """Register I21's policy target at decision ``d``: ``softmax(logit + sigma(completed q))`` over
    the candidate moves and submit (``M + 1``, float64; D8), the logits the network's raw ones.
    An action the raw prior gives no mass - a submit that is not offered - keeps none. None at a
    flat decision, or where the root offers no action (module docstring)."""
    cfg = cfg or CompletedQConfig()
    done = completed_values(d, cfg)
    if done is None:
        return None
    completed, _ = done
    prior = raw_prior(d, cfg.submit_share)
    offered = prior > 0
    visits, _ = action_statistics(d)
    z = np.log(prior[offered]) + sigma(completed[offered], float(visits.max()), cfg)
    e = np.exp(z - z.max())
    out = np.zeros(d.move_count + 1, dtype=np.float64)
    out[offered] = e / e.sum()
    return out
