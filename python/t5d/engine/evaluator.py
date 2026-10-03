"""The evaluator side of the search's external-evaluator protocol (stage N4).

The search (engine stage E6a, PUCT over the engine's own tree) evaluates leaves outside
itself: it collects a batch of leaf positions, hands them to an evaluator, and takes back,
for each leaf, **policy logits over that leaf's candidate moves** (spec §8.1: the pointer
head scores generator-supplied moves only) and **a 51-bin value distribution** (§8.3, D3).
This module is that evaluator, written against an abstract interface:

* ``Evaluator`` - a callable from a list of positions (``convert.Position`` or
  ``PositionState``) to a list of ``Evaluation``, in the same order.
* ``NetworkEvaluator`` - the dense path: positions batched through ``collate`` and the
  model's forward, ``max_batch`` at a time; positions of different board counts and sizes
  share a batch (padding is masked).
* ``CachedNetworkEvaluator`` - the cached path (C2, ``net/cache.py``): one position at a time
  against the per-board K/V store, in the epoch of the current search (``start(root)``).
* ``SearchDriver`` - what the search side provides: ``collect`` a batch of leaves,
  ``deliver`` their evaluations, ``finished``. ``serve`` runs the loop and checks every
  evaluation against its leaf.
* ``CStepDriver`` - **the adapter point** for E6a's C step API, whose signatures are not
  settled: every method raises ``NotImplementedError`` and says what it must do.

**The contract an evaluation keeps** (``check_evaluation``): ``policy_logits`` has exactly
one finite entry per candidate move of the leaf, in the leaf's move order - for a position
converted by ``convert.to_position`` that is the engine's move order, so logit ``i`` scores
the move ``t5d_game_apply_expanded(i)`` would play; ``value_logits`` has the network's
``value_bins`` (51) finite entries, bins centred on ``-1 + 2k/50`` (``net/heads.py``), and
``value_probs`` is their softmax. **The value is from the perspective of the player to
move** (``h_present & 1``, the network's target convention, ``net/heads.py``) - the side
whose candidate moves the policy scores. Promotions to different pieces tie under
``policy_promotion="none"`` (literal §8.1; build-log spec issue 9(d)).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..net.cache import CachedEvaluator, Epoch
from ..net.model import NetOutput, T5DNet
from ..net.state import PositionState, collate, relayout
from .convert import Position

PositionLike = Position | PositionState


def state_of(position: PositionLike) -> PositionState:
    """The ``PositionState`` of a converted position (or the state itself)."""
    return position.state if isinstance(position, Position) else position


def move_count(position: PositionLike) -> int:
    return int(state_of(position).moves.shape[0])


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One leaf's evaluation (see the module docstring for the contract)."""

    policy_logits: np.ndarray  # (M,) float32: one per candidate move, in the leaf's move order
    value_logits: np.ndarray  # (V,) float32: V = value_bins, the player to move's perspective

    @property
    def value_probs(self) -> np.ndarray:
        """The 51-bin value distribution (softmax of the logits, float32)."""
        z = self.value_logits.astype(np.float64)
        e = np.exp(z - z.max())
        return (e / e.sum()).astype(np.float32)

    @property
    def value_mean(self) -> float:
        """The distribution's mean on [-1, 1]."""
        v = self.value_logits.shape[0]
        centres = (2.0 * np.arange(v) - (v - 1)) / (v - 1)
        return float((self.value_probs.astype(np.float64) * centres).sum())

    def policy_probs(self) -> np.ndarray:
        """Softmax over the candidate moves (empty when there are none)."""
        if self.policy_logits.size == 0:
            return self.policy_logits.astype(np.float32)
        z = self.policy_logits.astype(np.float64)
        e = np.exp(z - z.max())
        return (e / e.sum()).astype(np.float32)


@runtime_checkable
class Evaluator(Protocol):
    """Positions in, evaluations out, in the same order."""

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]: ...


def _split(out: NetOutput, states: Sequence[PositionState]) -> list[Evaluation]:
    """Per-position evaluations of a batch's output (row ``i`` is ``states[i]``)."""
    logits = out.policy_logits.detach().float().cpu()
    valid = out.move_valid.detach().cpu()
    values = out.value_logits.detach().float().cpu()
    results = []
    for i, state in enumerate(states):
        m = int(state.moves.shape[0])
        if int(valid[i].sum()) != m or (m and not bool(valid[i, :m].all())):
            raise RuntimeError(f"position {i}: the batch's valid moves are not its {m} candidate moves")
        results.append(Evaluation(policy_logits=logits[i, :m].numpy().copy(), value_logits=values[i].numpy().copy()))
    return results


class NetworkEvaluator:
    """The dense path: ``collate`` and one forward per ``max_batch`` positions. ``scope``
    overrides the network's attention scope (None: its own). The model is used as given
    (put it in ``eval()`` mode); positions converted with other register counts than the
    network's are re-laid out (``state.relayout``, exact)."""

    def __init__(self, model: T5DNet, *, max_batch: int = 32, device: torch.device | str | None = None, scope: str | None = None) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.model = model
        self.max_batch = int(max_batch)
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.scope = scope
        self.batches = 0
        self.positions = 0

    def _fit(self, state: PositionState) -> PositionState:
        cfg = self.model.cfg
        if (state.regs_per_board, state.global_regs) != (cfg.regs_per_board, cfg.global_regs):
            return relayout(state, cfg.regs_per_board, cfg.global_regs)
        return state

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        results: list[Evaluation] = []
        for start in range(0, len(positions), self.max_batch):
            states = [self._fit(state_of(p)) for p in positions[start : start + self.max_batch]]
            batch = collate(states).to(self.device)
            with torch.no_grad():
                out = self.model(batch, scope=self.scope)
            results.extend(_split(out, states))
            self.batches += 1
            self.positions += len(states)
        return results


class CachedNetworkEvaluator:
    """The cached path (C2; ``net/cache.py``): each position evaluated incrementally - its
    frozen boards' K/V from the store, computed once per digest - in the epoch of the
    current search. ``start(root)`` opens a search (``CachedEvaluator.root_epoch``: a root
    refresh under ``CacheConfig.root_refresh``, else the causal epoch)."""

    def __init__(self, cached: CachedEvaluator) -> None:
        self.cached = cached
        self.epoch: Epoch | None = None

    @property
    def model(self) -> T5DNet:
        return self.cached.model

    def _fit(self, state: PositionState) -> PositionState:
        cfg = self.model.cfg
        if (state.regs_per_board, state.global_regs) != (cfg.regs_per_board, cfg.global_regs):
            return relayout(state, cfg.regs_per_board, cfg.global_regs)
        return state

    def start(self, root: PositionLike) -> Epoch:
        with torch.no_grad():
            self.epoch = self.cached.root_epoch(self._fit(state_of(root)))
        return self.epoch

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        if self.epoch is None:
            raise RuntimeError("no search is open: call start(root) first")
        results = []
        with torch.no_grad():
            for p in positions:
                state = self._fit(state_of(p))
                out = self.cached.evaluate(state, self.epoch)
                results.extend(_split(out, [state]))
        return results


# --- the search side ------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf:
    """A leaf the search wants evaluated: the driver's handle for it and its position."""

    id: int
    position: PositionLike


@runtime_checkable
class SearchDriver(Protocol):
    """The search side of the protocol (one search; synchronous)."""

    def collect(self, max_leaves: int) -> Sequence[Leaf]:
        """Up to ``max_leaves`` leaves to evaluate; empty only when the search is finished."""
        ...

    def deliver(self, results: Sequence[tuple[int, Evaluation]]) -> None:
        """The evaluations of collected leaves, by leaf id."""
        ...

    def finished(self) -> bool: ...


def check_evaluation(leaf: Leaf, evaluation: Evaluation, value_bins: int | None = None) -> None:
    """Raise ``ValueError`` unless ``evaluation`` keeps the contract for ``leaf``."""
    m = move_count(leaf.position)
    pl, vl = evaluation.policy_logits, evaluation.value_logits
    if pl.shape != (m,):
        raise ValueError(f"leaf {leaf.id}: {pl.shape[0] if pl.ndim == 1 else pl.shape} policy logits for {m} candidate moves")
    if not np.all(np.isfinite(pl)):
        raise ValueError(f"leaf {leaf.id}: a policy logit is not finite")
    if vl.ndim != 1 or (value_bins is not None and vl.shape[0] != value_bins):
        raise ValueError(f"leaf {leaf.id}: value logits of shape {vl.shape}, expected ({value_bins},)")
    if not np.all(np.isfinite(vl)):
        raise ValueError(f"leaf {leaf.id}: a value logit is not finite")


@dataclass
class ServeStats:
    """What ``serve`` did: rounds (collect - evaluate - deliver), leaves, and wall time in each
    phase (seconds)."""

    rounds: int = 0
    leaves: int = 0
    largest_batch: int = 0
    collect_seconds: float = 0.0
    evaluate_seconds: float = 0.0
    deliver_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)


def serve(
    driver: SearchDriver, evaluator: Evaluator, *, batch_size: int = 32, value_bins: int | None = 51, max_rounds: int | None = None
) -> ServeStats:
    """Run one search's evaluation loop: collect up to ``batch_size`` leaves, evaluate them
    in one call, check each evaluation against its leaf (``check_evaluation``), deliver;
    until the driver is finished (or ``max_rounds``)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    stats = ServeStats()
    while not driver.finished():
        if max_rounds is not None and stats.rounds >= max_rounds:
            break
        t0 = time.perf_counter()
        leaves = list(driver.collect(batch_size))
        t1 = time.perf_counter()
        if not leaves:
            if driver.finished():
                break
            raise RuntimeError("the driver returned no leaves but is not finished (a synchronous search always has one)")
        if len(leaves) > batch_size:
            raise RuntimeError(f"the driver returned {len(leaves)} leaves for a batch of {batch_size}")
        evaluations = evaluator([leaf.position for leaf in leaves])
        t2 = time.perf_counter()
        if len(evaluations) != len(leaves):
            raise RuntimeError(f"the evaluator returned {len(evaluations)} evaluations for {len(leaves)} leaves")
        for leaf, ev in zip(leaves, evaluations, strict=True):
            check_evaluation(leaf, ev, value_bins)
        driver.deliver([(leaf.id, ev) for leaf, ev in zip(leaves, evaluations, strict=True)])
        t3 = time.perf_counter()
        stats.rounds += 1
        stats.leaves += len(leaves)
        stats.largest_batch = max(stats.largest_batch, len(leaves))
        stats.batch_sizes.append(len(leaves))
        stats.collect_seconds += t1 - t0
        stats.evaluate_seconds += t2 - t1
        stats.deliver_seconds += t3 - t2
    return stats


# =============================================================================================
# ADAPTER POINT: the search's C step API (engine stage E6a).
#
# E6a is adding to the engine's C API a step interface for an external evaluator: collect
# a batch of leaf positions; accept policy logits over each leaf's candidate moves and a
# 51-bin value distribution. Its signatures are not settled, so nothing below guesses them.
# When they land, implement CStepDriver against them (in this file, declaring the new
# functions in capi.py's _SIGNATURES):
#
#  * __init__: hold the search handle (and the network's register counts, for to_position).
#  * collect(n): ask the step API for up to n leaves. For each leaf obtain its expansion -
#    if the API hands out a t5d_expansion per leaf, copy it at once with
#    Expansion.from_c(...) before the next call (the pointer lifetime rule of t5d/capi.h),
#    then convert.to_position(...). The leaf's candidate moves MUST be the moves the API
#    will match the logits against, in the same order: to_position keeps the expansion's
#    order (logit i = expansion move i). If the API scores a different move list (legal
#    moves only, or whole actions under D2's other arm), map the logits here, explicitly.
#  * deliver(results): pass, per leaf id, policy_logits (float32, M) and the value as the API
#    wants it - Evaluation.value_probs (51 float32 probabilities summing to 1) or
#    value_logits. The value is the player to move's (module docstring).
#  * finished(): whether the search is done (budget spent, or terminal root).
#
# A state whose present has shifted inside an unsubmitted action cannot be converted
# (convert.PresentMismatch); the step API should hand out leaves at action boundaries, or
# mid-action before the shift (the ply split, D2). Under the default rules (Q4, ruled R8)
# search meets such states - the mover may go on moving until they submit - and register D8
# gives them to the network through a global acting-player input and a submit logit
# (the expansion's acting_player / submit_possible / submit_legal): stage N5.
# =============================================================================================


class CStepDriver:
    """**Adapter point** for E6a's C step API (see the comment above): not implemented until
    the API's signatures are settled."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "CStepDriver waits on the search's C step API (engine stage E6a): implement collect/deliver/finished against "
            "its signatures - see the ADAPTER POINT comment in t5d/engine/evaluator.py"
        )

    def collect(self, max_leaves: int) -> Sequence[Leaf]:  # pragma: no cover - adapter point
        raise NotImplementedError

    def deliver(self, results: Sequence[tuple[int, Evaluation]]) -> None:  # pragma: no cover - adapter point
        raise NotImplementedError

    def finished(self) -> bool:  # pragma: no cover - adapter point
        raise NotImplementedError


__all__ = [
    "CStepDriver",
    "CachedNetworkEvaluator",
    "Evaluation",
    "Evaluator",
    "Leaf",
    "NetworkEvaluator",
    "SearchDriver",
    "ServeStats",
    "check_evaluation",
    "serve",
    "state_of",
]
