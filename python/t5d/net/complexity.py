"""The refresh ladder's complexity trigger (register C24, prototype; spec section 7.9,
"Complexity as a trigger", CHECKPOINT 4).

Rung 1 of the refresh ladder (C3a, ``tree.py``) refreshes a node reached by a travel move
that created a timeline, once its visits reach V - the one *known* discontinuity. C24 is
a variant beside it: **refresh more in complex positions**. Behind its own flag,
``CacheConfig.complexity_refresh`` (off by default), and gated like rung 1 on 6c's
existence test (the ruling: "The complexity trigger (C24) goes behind a flag alongside the
refresh ladder's rung 1").

**The signals** (section 7.9's list, each read where the spec's rule allows - "the signal
must exist *before* the evaluation it decides: the position's structure, or the parent's
outputs"):

* ``boards`` - the node's board count (structure);
* ``active_timelines`` - the node's timelines that are active, one per head (structure);
* ``travel`` - travel moves on the path from the node's last refresh down to the node
  itself (the tree's structure; ``tree.travel_since_refresh``);
* ``effort`` - the effort head's predicted difficulty **of the parent's evaluation**
  (section 8.6's head, D4: categorical over log2-visit bins with a censored top bin): the
  bin index at quantile ``complexity_effort_quantile`` of the parent's distribution
  (``effort_difficulty``). ``None`` - the parent's outputs are absent - contributes
  nothing; the signal is never guessed and never taken from the node's own evaluation.

**The rule** (every number a named ``CacheConfig`` parameter with a placeholder default -
the spec gives none; spec issue, see ``CacheConfig``): a weighted sum of the signals,

    score = w_boards * boards + w_timelines * active_timelines + w_travel * travel + w_effort * effort

fires when ``score >= complexity_threshold`` at a non-root node not yet refreshed whose
visits have reached ``complexity_visit_threshold``, while the search's budget of
complexity refreshes (``complexity_refresh_budget``) lasts. Weights are non-negative, so
the score is monotone in every signal - more complexity never means fewer refreshes - and
a weight of zero switches its signal off. With the placeholder weights each signal alone
reaches the threshold at a stated value (128 boards, 8 active timelines, 2 travel moves,
effort bin 12 - a predicted 4,096 visits), and any mix of them fires on the way there.

**Add-only.** The effort signal is the network's own rating of the position - the
confidence family (sections 8.6, 10.3, 14.4) - and so add-only; the build makes the whole
trigger add-only, whatever its signals: it can only *add* refreshes to those the same
configuration makes with the flag off, never remove or delay one. ``tree.SearchTree.visit``
decides rung 1 first, by its own unchanged rule, which reads nothing C24 changes (the move
type, the node's visits, the node's own refreshed flag); C24 is consulted only where rung
1 did not fire, and the budget limits C24's refreshes alone. A node refreshed early by
C24 is a node rung 1 would refresh later: the refresh is a dense pass over the same
position, so its history K/V are the same values - only sooner.

**Training.** None new: conditioning-offset training (C3) already covers any refresh
frequency - a node refresh resets the offset to 0, within the trained range (section 7.9).

**Not built:** C25 (a complexity-adaptive number of dense top layers) and the scanner
NNUE - both held open at CHECKPOINT 4. C24 refreshes the frozen history with the same
whole-scope fork as rung 1 (``CachedEvaluator.refresh``); it changes when, never how.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .heads import categorical_probs, effort_quantile
from .state import PositionState


@dataclass(frozen=True)
class ComplexitySignals:
    """C24's inputs for one node, all of which exist before the node's evaluation."""

    boards: int  # the node's board count (structure)
    active_timelines: int  # the node's active timelines (structure)
    travel: int  # travel moves since the node's last refresh (the tree's structure)
    effort: int | None = None  # the parent's predicted difficulty (an effort bin), None if absent


def active_timelines(state: PositionState) -> int:
    """The position's active timelines: its heads whose timeline is active (each timeline
    has one head, and the active flag is a property of the timeline)."""
    return int(np.count_nonzero(state.is_head & state.active))


def effort_difficulty(effort_logits: torch.Tensor, quantile: float) -> int:
    """A predicted difficulty from an evaluation's effort head (section 8.6): the bin
    index - log2 of the predicted visits; the censored "did not converge" bin the largest -
    at ``quantile`` of the distribution. ``effort_logits``: one position's ``(bins,)``."""
    logits = effort_logits.detach().reshape(-1)
    return int(effort_quantile(categorical_probs(logits), quantile))


def structural_signals(state: PositionState, travel: int, effort: int | None = None) -> ComplexitySignals:
    """The signals of a node at ``state``: its board count and active timelines, the travel
    since its last refresh (the caller's: the path is the tree's, not the position's), and
    the parent's predicted difficulty if the caller has it."""
    return ComplexitySignals(int(state.n_boards), active_timelines(state), int(travel), None if effort is None else int(effort))


def complexity_score(signals: ComplexitySignals, config) -> float:
    """C24's score: the weighted sum of the signals (``config``: a ``CacheConfig``)."""
    score = (
        config.complexity_board_weight * signals.boards
        + config.complexity_timeline_weight * signals.active_timelines
        + config.complexity_travel_weight * signals.travel
    )
    if signals.effort is not None:
        score += config.complexity_effort_weight * signals.effort
    return float(score)


def complexity_fires(signals: ComplexitySignals, config) -> bool:
    """Whether the score reaches the threshold (the visit gate, the root and the budget
    are the tree's: ``tree.SearchTree.visit``)."""
    return complexity_score(signals, config) >= config.complexity_threshold


__all__ = [
    "ComplexitySignals",
    "active_timelines",
    "complexity_fires",
    "complexity_score",
    "effort_difficulty",
    "structural_signals",
]
