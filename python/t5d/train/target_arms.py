"""Register M24's target arms for I21 (spec sections 12.2, 12.5, 13.1): the completed-Q policy
target against the pruned-visit one - the arms' configurations, the record-level comparison of the
two targets, and the held-out yardstick an arm's network is read by.

Section 12.2 ranks the completed-Q target (Danihelka, Guez, Schrittwieser and Silver, "Policy
improvement by planning with Gumbel", ICLR 2022; ``train.completed_q``) "first as a target arm on
the shared dataset (section 13.1), against the pruned-visit target, with no new self-play", and
puts a prediction on record: level where both are built from full searches, clearly ahead where
both are built from cheap ones. Section 13.1's target arms "re-derive targets and weights from the
same records first". This module is the part of that which needs no dataset:

* ``ARMS`` - each arm as (``TargetConfig.completed_q``, whether the main policy head trains on the
  completed-Q target): ``visits_full`` (``BASELINE``: the as-built target and loss),
  ``completed_q_full`` (like for like with it), ``completed_q_cheap`` and ``completed_q_all`` (I5's
  arm of 6l: cheap searches emitting policy targets). ``arm_configs`` and ``arm_trainer_config``
  turn a name into the builder's and the loss's settings (``losses.completed_q_arm``). **An arm
  changes one thing, the main policy head's target**: the soft policy, D9's optimistic policy, the
  sampler's surprise weights and the recorded divergence read the pruned-visit target in every arm
  (PROVISIONAL: sections 12.2, 12.4 and 12.8 do not say what they read under the arm; this is the
  reading built, for the user to confirm before the ranking is run - with the soft policy at eight
  times the policy's weight, a second pass at soft weight 0 in both arms is the control).
* ``compare_records`` - what the two targets look like on the records themselves, per search class
  (full, cheap): no network, no engine, no training. It answers, before any training, the questions
  the paper's constants raise under PUCT's allocation (``c_visit`` 50 and ``c_scale`` 1 were set
  for sequential halving, where the largest count is modest; a PUCT root puts most of its visits
  on one child, so sigma's scale is large and the target can be close to one-hot): each target's
  entropy and divergence from the raw prior, how often their top moves agree, how far apart they
  are (total variation), how much of the completed-Q target sits on moves the search never visited,
  and how often its top move is one the search visited once. A cheap search's pruned-visit
  distribution is reported here as a statistic only - the builder writes no such target (I5).
  ``format_comparison`` prints it; ``python -m t5d.train.target_arms <record.npz ...>`` does both.
* ``evaluate_rows`` - **the held-out yardstick, the same for every arm**, forward only. An arm's
  own policy loss is measured against its own target, so losses do not compare across target arms.
  Every arm is therefore scored on the same held-out rows against the same references. Primary
  (``rank_arms``' default key): ``top1_visits``, the share of held-out full searches whose most
  visited move - the argmax of the pruned-visit target, the move the search plays at temperature 0,
  a reference that takes neither target's form - is the raw policy's top action. Beside it: the
  raw policy's cross-entropy against each reference (``ce_visits``; ``ce_completed_q_full`` and
  ``_cheap``), top-1 against the completed-Q targets, and the WDL loss (``wdl``: value-type targets
  are unchanged by the arm, so it should not move). Sums and counts are accumulated over the rows,
  so the figures do not depend on the batch size beyond the forward pass's own float32 rounding
  (measured: 9e-8 relative at most, FP32 trunk, on synthetic rows batched 1 to 6 at a time; the
  tests allow 1e-5). PROVISIONAL: section 13.1 says only
  "ranked by policy and value loss and accuracy, then searchless and search Elo for the finalists";
  this yardstick is the reading built, to be confirmed before the ranking is run.
* ``rank_arms`` - arm names ordered by one of the yardstick's figures.

**Not built: the ranking itself.** M24 and section 15.7's gate 3 define it as supervised trainings
per arm on the pilot run's shared dataset, then Elo for the finalists. There is no dataset before
the pilot run, the brief forbids training runs, and ``trainer.Trainer`` refuses optimiser steps on
record rows and more than ``smoke.MAX_STEPS`` per model. So there is no per-arm training driver
here, no split of a dataset into training and held-out games, and no Elo: this module holds no
optimiser step at all. Also waiting on that driver: a ``visits_cheap`` arm (the prediction's second
half compares the two targets built from cheap searches; the builder writes no pruned-visit target
for a cheap search, I5, and the policy loss is gated on ``full_search``), and a reading of the
completed-Q target for D2's flat form (``completed_q_policy`` returns None there, so every arm
leaves flat decisions on their per-board pruned-visit target; they are counted and skipped here).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import torch

from ..net.config import NetConfig
from .completed_q import action_statistics
from .losses import LossConfig, completed_q_arm, wdl_target
from .records import GameRecord, read_game
from .targets import Row, TargetConfig, build_targets, collate_rows, kl_divergence, policy_target, raw_prior, submit_share_of
from .trainer import TrainerConfig

#: The target arms: a name -> (``TargetConfig.completed_q``, whether the main policy head trains on
#: the completed-Q target instead of the pruned-visit one).
ARMS: dict[str, tuple[str, bool]] = {
    "visits_full": ("off", False),
    "completed_q_full": ("full", True),
    "completed_q_cheap": ("cheap", True),
    "completed_q_all": ("all", True),
}

#: The arm every other is ranked against: the as-built pruned-visit target of full searches.
BASELINE = "visits_full"

#: The search classes ``compare_records`` reports (section 12.5's playout-cap randomisation).
SEARCH_CLASSES: tuple[str, ...] = ("full", "cheap")

#: ``compare_records``' statistics, in the order ``format_comparison`` prints them: the counts ...
COMPARISON_COUNTS: tuple[str, ...] = ("decisions", "flat", "visit_targets", "completed_q_targets", "visit_kl_infinite")
#: ... and the means, each over the decisions that have the quantity (NaN where none does).
COMPARISON_MEANS: tuple[str, ...] = (
    "actions",
    "playouts",
    "visited",
    "visit_entropy",
    "completed_q_entropy",
    "visit_kl",
    "completed_q_kl",
    "top1_agree",
    "total_variation",
    "completed_q_unvisited_mass",
    "completed_q_top1_visits",
    "completed_q_top1_single_visit",
)

#: ``evaluate_rows``' references: each has a cross-entropy ``ce_<name>``, a top-1 agreement
#: ``top1_<name>`` and a row count ``n_<name>``.
YARDSTICK_REFERENCES: tuple[str, ...] = ("visits", "completed_q_full", "completed_q_cheap")


# --- the arms' configurations --------------------------------------------------------------------------


def arm_configs(arm: str, target: TargetConfig | None = None, loss: LossConfig | None = None) -> tuple[TargetConfig, LossConfig]:
    """The target builder's and the loss's settings under ``arm`` (``ARMS``): ``target`` with the
    arm's ``completed_q`` switch, and ``loss`` - as given under the baseline, with the policy's
    weight moved onto the completed-Q loss otherwise (``losses.completed_q_arm``). Nothing else
    differs between arms."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)} (register I21, M24)")
    searches, on_completed_q = ARMS[arm]
    target = replace(TargetConfig() if target is None else target, completed_q=searches)
    loss = LossConfig() if loss is None else loss
    return target, completed_q_arm(loss) if on_completed_q else loss


def arm_trainer_config(arm: str, net_cfg: NetConfig, base: TrainerConfig | None = None) -> TrainerConfig:
    """``base`` (None: the defaults) with ``arm``'s target and loss settings - the target's
    short-term horizons the network's where ``base`` names no target, as the trainer reads it."""
    base = TrainerConfig() if base is None else base
    target = base.target or TargetConfig(short_term_horizons=tuple(net_cfg.short_term_horizons))
    target, loss = arm_configs(arm, target, base.loss)
    return replace(base, target=target, loss=loss)


# --- the two targets on the records --------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    """A distribution's entropy in nats."""
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


class _Tally:
    """Counts, and sums with their own denominators (``compare_records``)."""

    def __init__(self) -> None:
        self.counts = dict.fromkeys(COMPARISON_COUNTS, 0)
        self.sums = dict.fromkeys(COMPARISON_MEANS, 0.0)
        self.n = dict.fromkeys(COMPARISON_MEANS, 0)

    def add(self, name: str, value: float) -> None:
        self.sums[name] += float(value)
        self.n[name] += 1

    def result(self) -> dict[str, float]:
        out = {name: float(count) for name, count in self.counts.items()}
        out.update({name: self.sums[name] / self.n[name] if self.n[name] else math.nan for name in COMPARISON_MEANS})
        return out


def compare_records(records: Iterable[GameRecord], cfg: TargetConfig | None = None) -> dict[str, dict[str, float]]:
    """The pruned-visit and the completed-Q targets side by side on ``records``, per search class
    (``SEARCH_CLASSES``) - from the records alone (module docstring). Each game's targets are built
    under ``cfg`` (None: the defaults) with the completed-Q target on for every search; a decision's
    visit distribution is ``policy_target`` of its pruned visits (for a cheap search a statistic
    only), its raw prior ``raw_prior`` with the record's submit share, its visits per action the
    recorded ones (``completed_q.action_statistics``: not pruned - what the completed-Q target reads).

    Counts: ``decisions`` (the ply split's and the tree form's, the ones compared), ``flat`` (D2's
    flat-form decisions: counted, skipped - no completed-Q target), ``visit_targets`` and
    ``completed_q_targets`` (decisions with each), ``visit_kl_infinite`` (a visited action the raw
    prior gives no mass). Means, each over the decisions that have the quantity (NaN: none):
    ``actions`` (those the raw prior offers), ``playouts``, ``visited`` (actions with a visit),
    ``visit_entropy`` and ``completed_q_entropy`` (nats), ``visit_kl`` and ``completed_q_kl``
    (KL(target || raw prior), the finite ones), ``top1_agree`` (over decisions with both targets:
    the completed-Q target's top action is a most visited one - ties in the visits count as
    agreement), ``total_variation`` (half the L1 distance between the two, over the same decisions),
    ``completed_q_unvisited_mass`` (the completed-Q target's mass on actions with no visit),
    ``completed_q_top1_visits`` (the recorded visits of its top action) and
    ``completed_q_top1_single_visit`` (the share of completed-Q targets whose top action was visited
    exactly once: one lucky backed-up value taking the mass)."""
    cfg = replace(TargetConfig() if cfg is None else cfg, completed_q="all")
    tallies = {name: _Tally() for name in SEARCH_CLASSES}
    for record in records:
        targets = build_targets(record, cfg)
        share = submit_share_of(record)
        for i, d in enumerate(record.decisions):
            tally = tallies["full" if d.full_search else "cheap"]
            if d.flat:
                tally.counts["flat"] += 1
                continue
            tally.counts["decisions"] += 1
            prior = raw_prior(d, share)
            visits, _ = action_statistics(d)
            tally.add("actions", int((prior > 0).sum()))
            tally.add("playouts", d.playouts)
            tally.add("visited", int((visits > 0).sum()))
            v = policy_target(d, targets.pruned[i])
            c = targets.completed_q[i]
            if v is not None:
                tally.counts["visit_targets"] += 1
                tally.add("visit_entropy", _entropy(v))
                kl = kl_divergence(v, prior)
                if math.isfinite(kl):
                    tally.add("visit_kl", kl)
                else:
                    tally.counts["visit_kl_infinite"] += 1
            if c is not None:
                top = int(np.argmax(c))
                tally.counts["completed_q_targets"] += 1
                tally.add("completed_q_entropy", _entropy(c))
                kl = kl_divergence(c, prior)
                if math.isfinite(kl):
                    tally.add("completed_q_kl", kl)
                tally.add("completed_q_unvisited_mass", float(c[visits <= 0].sum()))
                tally.add("completed_q_top1_visits", float(visits[top]))
                tally.add("completed_q_top1_single_visit", float(visits[top] == 1))
                if v is not None:
                    tally.add("top1_agree", float(v[top] >= v.max()))
                    tally.add("total_variation", 0.5 * float(np.abs(v - c).sum()))
    return {name: tally.result() for name, tally in tallies.items()}


def format_comparison(stats: Mapping[str, Mapping[str, float]]) -> str:
    """``compare_records``' result as a plain table: a row per statistic, a column per search class
    (counts as integers, means to four places, ``-`` where there is none)."""
    classes = list(stats)
    width = max(len(name) for name in (*COMPARISON_COUNTS, *COMPARISON_MEANS))
    lines = [" ".join(["statistic".ljust(width), *(c.rjust(12) for c in classes)])]
    for name in (*COMPARISON_COUNTS, *COMPARISON_MEANS):
        cells = []
        for c in classes:
            value = float(stats[c].get(name, math.nan))
            if math.isnan(value):
                cells.append("-".rjust(12))
            elif name in COMPARISON_COUNTS:
                cells.append(f"{int(value):d}".rjust(12))
            else:
                cells.append(f"{value:.4f}".rjust(12))
        lines.append(" ".join([name.ljust(width), *cells]))
    return "\n".join(lines)


# --- the held-out yardstick ----------------------------------------------------------------------------


def evaluate_rows(model: torch.nn.Module, rows: Sequence[Row], net_cfg: NetConfig, *, batch_size: int = 64) -> dict[str, float]:
    """The yardstick every target arm's network is read by (module docstring), on ``rows`` -
    held-out rows, their targets built with the completed-Q target on for every search
    (``TargetConfig(completed_q="all")``) - forward only: no gradient, no optimiser step, the
    network left in the mode it was given in. Whatever the arm trained on, the figures are of the
    **main policy head's** raw distribution over each decision's actions (the candidate moves and,
    where legal, submit):

    * ``ce_visits``, ``top1_visits``, ``n_visits`` - against the pruned-visit target, over the rows
      that carry one (a full search's, not a flat decision's): the cross-entropy (the policy loss's
      own form) and the share of rows whose top action is a most visited one (a tie in the target
      counts for either).
    * ``ce_completed_q_full``, ``top1_completed_q_full``, ``n_completed_q_full``, and the same three
      with ``_cheap`` - against the completed-Q target, over the rows that carry one, split by
      ``Targets.full_search`` (the row carries a full search's policy-type target; a full search that
      visited no child, which has none, counts with the cheap ones).
    * ``wdl``, ``n_wdl`` - the WDL head's cross-entropy against the game result (``wdl_target`` at
      ``q_ratio`` 0), over the rows of scored games.

    Each figure is its sum over the rows divided by its count; a figure with no row is NaN (rows
    built with the switch off have no completed-Q target)."""
    if batch_size < 1:
        raise ValueError("batch_size >= 1")
    sums = {f"{kind}_{name}": 0.0 for name in YARDSTICK_REFERENCES for kind in ("ce", "top1")} | {"wdl": 0.0}
    counts = dict.fromkeys((*YARDSTICK_REFERENCES, "wdl"), 0)

    def score(name: str, target: torch.Tensor, keep: torch.Tensor, logp: torch.Tensor, pred: torch.Tensor) -> None:
        per = -(target * logp).sum(dim=-1)
        hit = target.gather(1, pred[:, None])[:, 0] >= target.max(dim=-1).values
        sums[f"ce_{name}"] += float(per[keep].double().sum())
        sums[f"top1_{name}"] += float(hit[keep].sum())
        counts[name] += int(keep.sum())

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch, tgt = collate_rows(rows[start : start + batch_size], net_cfg)
                out = model(batch)
                valid = out.action_valid()
                device = valid.device
                logits = out.action_logits().float().masked_fill(~valid, float("-inf"))
                logp = torch.log_softmax(logits, dim=-1).masked_fill(~valid, 0.0)
                pred = logits.argmax(dim=-1)
                offered = valid.any(dim=-1)
                bsz = valid.shape[0]
                full = torch.ones(bsz, dtype=torch.bool) if tgt.full_search is None else tgt.full_search.bool()
                flat = torch.zeros(bsz, dtype=torch.bool) if tgt.flat_rows is None else tgt.flat_rows.bool()
                full, flat = full.to(device), flat.to(device)
                submit = tgt.policy.new_zeros(bsz) if tgt.submit is None else tgt.submit
                visit = torch.cat([tgt.policy, submit[:, None]], dim=-1).float().to(device)
                score("visits", visit, full & ~flat & (visit.sum(dim=-1) > 0) & offered, logp, pred)
                if tgt.completed_q is not None:
                    cq_submit = tgt.completed_q.new_zeros(bsz) if tgt.completed_q_submit is None else tgt.completed_q_submit
                    cq = torch.cat([tgt.completed_q, cq_submit[:, None]], dim=-1).float().to(device)
                    has = (cq.sum(dim=-1) > 0) & offered
                    score("completed_q_full", cq, has & full, logp, pred)
                    score("completed_q_cheap", cq, has & ~full, logp, pred)
                wt = wdl_target(tgt, 0.0).to(device)
                has_wdl = torch.isfinite(wt).all(dim=-1)  # a NaN row: no target (an unscored game)
                wdl_logp = torch.log_softmax(out.wdl_logits.float(), dim=-1)
                per = -(torch.where(has_wdl[:, None], wt, torch.zeros_like(wt)) * wdl_logp).sum(dim=-1)
                sums["wdl"] += float(per[has_wdl].double().sum())
                counts["wdl"] += int(has_wdl.sum())
    finally:
        model.train(was_training)

    result: dict[str, float] = {}
    for name in YARDSTICK_REFERENCES:
        n = counts[name]
        result[f"ce_{name}"] = sums[f"ce_{name}"] / n if n else math.nan
        result[f"top1_{name}"] = sums[f"top1_{name}"] / n if n else math.nan
        result[f"n_{name}"] = float(n)
    result["wdl"] = sums["wdl"] / counts["wdl"] if counts["wdl"] else math.nan
    result["n_wdl"] = float(counts["wdl"])
    return result


def rank_arms(results: Mapping[str, Mapping[str, float]], key: str = "top1_visits", *, higher_is_better: bool = True) -> list[str]:
    """The arms of ``results`` (a name -> ``evaluate_rows``' figures) ordered by ``key``, best first:
    the highest by default (an agreement), the lowest with ``higher_is_better=False`` (a loss); an
    arm whose figure is NaN or missing comes last, and equal figures are ordered by name. It orders
    what it is given: whether a difference is one - beyond the seed-to-seed spread - is the
    ranking's to establish."""

    def order(name: str) -> tuple[int, float, str]:
        value = float(results[name].get(key, math.nan))
        if math.isnan(value):
            return (1, 0.0, name)
        return (0, -value if higher_is_better else value, name)

    return sorted(results, key=order)


# --- the comparison from the command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The pruned-visit and completed-Q policy targets compared on game records (register I21).")
    parser.add_argument("records", nargs="+", help="game records (.npz, register I15's format)")
    parser.add_argument("--c-visit", type=float, default=TargetConfig().completed_q_c_visit, help="sigma's c_visit (a placeholder)")
    parser.add_argument("--c-scale", type=float, default=TargetConfig().completed_q_c_scale, help="sigma's c_scale (a placeholder)")
    parser.add_argument("--value-range", default=TargetConfig().completed_q_value_range, help="'unit' or 'signed' (a placeholder)")
    args = parser.parse_args(argv)
    cfg = TargetConfig(completed_q_c_visit=args.c_visit, completed_q_c_scale=args.c_scale, completed_q_value_range=args.value_range)
    stats = compare_records((read_game(path) for path in args.records), cfg)
    print(format_comparison(stats))
    return 0


__all__ = [
    "ARMS",
    "BASELINE",
    "COMPARISON_COUNTS",
    "COMPARISON_MEANS",
    "SEARCH_CLASSES",
    "YARDSTICK_REFERENCES",
    "arm_configs",
    "arm_trainer_config",
    "compare_records",
    "evaluate_rows",
    "format_comparison",
    "main",
    "rank_arms",
]


if __name__ == "__main__":
    raise SystemExit(main())
