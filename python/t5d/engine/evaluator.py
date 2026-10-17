"""The evaluator side of the search's external-evaluator protocol (stages N4, N5).

The search (engine stage E6a, PUCT over the engine's own tree) evaluates leaves outside
itself: it collects a batch of leaf positions, hands them to an evaluator, and takes back,
for each leaf, **policy logits over that leaf's candidate moves** (spec §8.1: the pointer
head scores generator-supplied moves only), **the submit action's logit** where the leaf
offers a legal submission (register D8, §8.2: read from a global register) and **a 51-bin
value distribution** (§8.3, D3). This module is that evaluator, written against an abstract
interface:

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
* ``CStepDriver`` - the ``SearchDriver`` over the engine's C step API (``t5d/capi_search.h``,
  ``search.py``; stage N5): gathers a batch of leaves, converts each leaf's expansion into
  the network's input, delivers move logits, submit logits and values; ``run_search`` runs
  one search with it.

**The contract an evaluation keeps** (``check_evaluation``): ``policy_logits`` has exactly
one finite entry per candidate move of the leaf, in the leaf's move order - for a position
converted by ``convert.to_position`` that is the engine's move order, so logit ``i`` scores
the move ``t5d_game_apply_expanded(i)`` would play (and, for a search leaf, the move
``leaf.expansion.moves[i]``); ``submit_logit`` is finite where the leaf offers a legal
submission and NaN otherwise - not offered, or masked as illegal (C4); ``value_logits`` has
the network's ``value_bins`` (51) finite entries, bins centred on ``-1 + 2k/50``
(``net/heads.py``), and ``value_probs`` is their softmax. **The value is from the
perspective of the acting player** (register D8: the player whose action is in progress
until it submits - the present's colour except past the shift; the network's target
convention, ``net/heads.py``, and the search's) - the side whose candidate moves the policy
scores. Promotions to different pieces tie under ``policy_promotion="none"`` (literal §8.1;
build-log spec issue 9(d)).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ..net.cache import CachedEvaluator, Epoch, state_digest
from ..net.model import NetOutput, T5DNet
from ..net.state import PositionState, collate, relayout
from .capi import T5D_VALUE_LOGITS, T5D_VALUE_PROBABILITIES
from .convert import Position, to_position
from .errors import EngineInternalError
from .search import LeafView, Search

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
    value_logits: np.ndarray  # (V,) float32: V = value_bins, the acting player's perspective
    #: D8: the submit action's logit, read from a global register - finite where the leaf
    #: offers a legal submit, NaN otherwise (not offered, or masked as illegal; NaN where it is
    #: legal stands for "none given": the search's placeholder share).
    submit_logit: float = math.nan

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

    def action_probs(self) -> np.ndarray:
        """``(M + 1,)``: the decision's distribution - the candidate moves, then submit (D8) -
        a softmax over the moves and, where its logit is finite, submit (0 otherwise)."""
        z = np.append(self.policy_logits.astype(np.float64), self.submit_logit)
        live = np.isfinite(z)
        out = np.zeros(z.shape[0], dtype=np.float64)
        if live.any():
            e = np.exp(z[live] - z[live].max())
            out[live] = e / e.sum()
        return out.astype(np.float32)

    def policy_probs(self) -> np.ndarray:
        """The candidate moves' probabilities (empty when there are none), normalised together
        with submit where its logit is finite - so they and ``submit_prob`` sum to one."""
        return self.action_probs()[:-1]

    @property
    def submit_prob(self) -> float:
        """Submit's probability in the decision's distribution (0 where it has no logit)."""
        return float(self.action_probs()[-1])


@runtime_checkable
class Evaluator(Protocol):
    """Positions in, evaluations out, in the same order."""

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]: ...


def _split(out: NetOutput, states: Sequence[PositionState]) -> list[Evaluation]:
    """Per-position evaluations of a batch's output (row ``i`` is ``states[i]``)."""
    logits = out.policy_logits.detach().float().cpu()
    valid = out.move_valid.detach().cpu()
    values = out.value_logits.detach().float().cpu()
    submit = out.submit_logits.detach().float().cpu()
    submit_valid = out.submit_valid.detach().cpu()
    results = []
    for i, state in enumerate(states):
        m = int(state.moves.shape[0])
        if int(valid[i].sum()) != m or (m and not bool(valid[i, :m].all())):
            raise RuntimeError(f"position {i}: the batch's valid moves are not its {m} candidate moves")
        s = float(submit[i])
        legal = bool(submit_valid[i])
        if legal and not math.isfinite(s):
            # NaN would read as "no logit given" (the search's placeholder share): refuse it as a
            # non-finite move logit is refused.
            raise ValueError(f"position {i}: the network's submit logit is {s} where submitting is legal")
        results.append(
            Evaluation(
                policy_logits=logits[i, :m].numpy().copy(),
                value_logits=values[i].numpy().copy(),
                submit_logit=s if legal else math.nan,  # masked (-inf) or absent: no logit
            )
        )
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
        """Open a search at ``root``: its epoch (the current one again when ``root`` is the
        current epoch's anchor - the same search continuing)."""
        state = self._fit(state_of(root))
        if self.epoch is not None and self.epoch.anchor == state_digest(state):
            return self.epoch
        with torch.no_grad():
            self.epoch = self.cached.root_epoch(state)
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
    """Raise ``ValueError`` unless ``evaluation`` keeps the contract for ``leaf``: the move
    logits as before; the submit logit (D8) never infinite, and NaN unless the leaf offers a
    legal submit (where NaN means "none given")."""
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
    s = float(evaluation.submit_logit)
    if math.isinf(s):
        raise ValueError(f"leaf {leaf.id}: the submit logit is infinite (masked or absent is NaN)")
    state = state_of(leaf.position)
    if not math.isnan(s) and not (state.submit_possible and state.submit_legal):
        raise ValueError(f"leaf {leaf.id}: a submit logit for a leaf that offers no legal submit")


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
    until the driver is finished (or ``max_rounds``). If evaluating or checking a batch fails,
    a driver with an ``abandon`` method is told before the exception leaves (the step API's
    virtual loss is released)."""
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
        try:
            if len(leaves) > batch_size:
                raise RuntimeError(f"the driver returned {len(leaves)} leaves for a batch of {batch_size}")
            evaluations = evaluator([leaf.position for leaf in leaves])
            if len(evaluations) != len(leaves):
                raise RuntimeError(f"the evaluator returned {len(evaluations)} evaluations for {len(leaves)} leaves")
            for leaf, ev in zip(leaves, evaluations, strict=True):
                check_evaluation(leaf, ev, value_bins)
        except BaseException:
            abandon = getattr(driver, "abandon", None)
            if abandon is not None:
                abandon()
            raise
        t2 = time.perf_counter()
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


# --- the C step API (engine stage E6a; stage N5) ------------------------------------------------


class CStepDriver:
    """The ``SearchDriver`` over the engine's C step API (``t5d/capi_search.h``; ``search.py``):
    one search, run for ``playouts`` simulations from now - the loop of the engine's own
    ``Search::run`` (``src/mcts.cpp``), with Python as the evaluator.

    ``collect(n)``: stop when the playouts are spent or the root needs no evaluation
    (terminal or solved); otherwise ``t5d_search_gather`` up to ``n`` leaves within the
    remaining playouts - a gather that completed visits without the evaluator (terminals,
    cache hits) and wants no evaluation is followed by another; one that made no progress
    ends the run (exhausted). Each leaf's expansion is copied out at once
    (``Expansion.from_c``) and converted (``convert.to_position``) with the network's register
    counts: its candidate moves are the leaf's, in the leaf's order, so logit ``i`` scores
    ``leaf.expansion.moves[i]``; its acting player and submit flags are the leaf's (D8 - under
    the whole-turn split no leaf offers submit: no end-of-turn token, spec issue 92).

    ``deliver(results)``: every leaf of the batch, by leaf id: the move logits concatenated
    in leaf order, one submit logit per leaf - the evaluation's where the leaf offers a legal
    submit, NaN elsewhere (not offered; or illegal, which the search prunes) - and the 51
    value logits (``T5D_VALUE_LOGITS``), from the leaf's acting player. A refused delivery
    raises the engine's error; the engine has abandoned the batch.

    The search's ``expand`` options must give every board's features (``"all"``): the
    network's input is the full history. Errors from the engine raise as exceptions
    (``errors.py``); ``abandon`` releases a waiting batch (``run_search`` and ``serve`` call it
    when evaluation fails). ``batch`` is the waiting batch's leaves (``LeafView``: key, epoch,
    D8's flags, the expansion), in the order the evaluator receives their positions.

    ``value_kind``: what an evaluation's ``value_logits`` hold - ``T5D_VALUE_LOGITS`` (the
    network's, the default) or ``T5D_VALUE_PROBABILITIES`` (an evaluator with exact
    probabilities - the engine's built-in evaluators rewritten in Python, for bit-identical
    comparisons with the engine's own runs)."""

    def __init__(
        self, search: Search, *, playouts: int, regs_per_board: int = 2, global_regs: int = 16, value_kind: int = T5D_VALUE_LOGITS
    ) -> None:
        if playouts < 0:
            raise ValueError("playouts must not be negative")
        if search.square_features != "all":
            raise ValueError(f"the network needs every board's features: the search expands with {search.square_features!r} features")
        if value_kind not in (T5D_VALUE_LOGITS, T5D_VALUE_PROBABILITIES):
            raise ValueError("value_kind must be T5D_VALUE_LOGITS or T5D_VALUE_PROBABILITIES")
        self.search = search
        self.playouts = int(playouts)
        self.regs_per_board = int(regs_per_board)
        self.global_regs = int(global_regs)
        self.value_kind = int(value_kind)
        self._p0 = search.info().playouts
        self._waiting: list[LeafView] | None = None
        self._done = False
        #: Why the run stopped: "playouts", "finished" (the root needs no evaluation) or
        #: "exhausted" (no simulation could be made); None while it runs.
        self.stop: str | None = None
        self.gathers = 0
        self.convert_seconds = 0.0

    def _finish(self, why: str) -> list[Leaf]:
        self._done, self.stop = True, why
        return []

    @property
    def batch(self) -> tuple[LeafView, ...]:
        """The waiting batch's leaves, in order (empty when none waits)."""
        return tuple(self._waiting or ())

    def collect(self, max_leaves: int) -> list[Leaf]:
        if self._waiting is not None:
            raise RuntimeError("a batch is waiting for its evaluations: deliver or abandon it first")
        while True:
            info = self.search.info()
            played = info.playouts - self._p0
            if played >= self.playouts:
                return self._finish("playouts")
            if info.root_finished:
                return self._finish("finished")
            n = self.search.gather(max_leaves, self.playouts - played)
            self.gathers += 1
            if n == 0:
                if self.search.info().playouts == info.playouts:
                    return self._finish("exhausted")
                continue
            t0 = time.perf_counter()
            try:
                views = [self.search.leaf_view(i) for i in range(n)]
                leaves = [Leaf(v.index, self._position(v)) for v in views]
            except BaseException:
                self.search.abandon()
                raise
            self.convert_seconds += time.perf_counter() - t0
            self._waiting = views
            return leaves

    def _position(self, v: LeafView) -> Position:
        x = v.expansion
        if v.acting_player != x.acting_player:
            raise EngineInternalError(f"leaf {v.index}: its acting player {v.acting_player} is not its expansion's {x.acting_player}")
        if (v.submit_possible, v.submit_legal) != (x.submit_possible, x.submit_legal):
            # The whole-turn split offers no submit action (the leaf says so; spec issue 92).
            x = replace(x, submit_possible=v.submit_possible, submit_legal=v.submit_legal)
        return to_position(x, regs_per_board=self.regs_per_board, global_regs=self.global_regs, scope=False)

    def deliver(self, results: Sequence[tuple[int, Evaluation]]) -> None:
        views = self._waiting
        if views is None:
            raise RuntimeError("no batch is waiting")
        by_id = dict(results)
        if sorted(by_id) != list(range(len(views))) or len(results) != len(views):
            self.abandon()
            raise RuntimeError(f"deliver: evaluations for leaves {sorted(by_id)}, the batch is leaves 0..{len(views) - 1}")
        evs = [by_id[i] for i in range(len(views))]
        logits = np.concatenate([np.asarray(ev.policy_logits, dtype=np.float32).reshape(-1) for ev in evs] + [np.zeros(0, np.float32)])
        values = np.stack([np.asarray(ev.value_logits, dtype=np.float32) for ev in evs])
        offered = [v.submit_possible and v.submit_legal for v in views]
        submit = np.array([ev.submit_logit if ok else math.nan for ev, ok in zip(evs, offered, strict=True)], dtype=np.float32)
        self._waiting = None  # delivered or refused, the engine no longer holds the batch
        self.search.deliver(logits, values, submit_logits=submit, value_kind=self.value_kind)

    def abandon(self) -> None:
        """Release a waiting batch (nothing backed up)."""
        if self._waiting is not None:
            self._waiting = None
            self.search.abandon()

    def finished(self) -> bool:
        return self._done


def engine_epoch(epoch: Epoch) -> int:
    """The engine's 64-bit epoch number for a cached-path epoch (its key, hashed): the search
    caches evaluations per (state, moves made in its action, epoch) (register A6), so
    evaluations made under different refresh conditioning are never served for one another;
    the shared causal epoch maps to one number."""
    return int.from_bytes(hashlib.blake2b(epoch.key.encode("utf-8"), digest_size=8).digest(), "little")


def start_epoch(search: Search, evaluator: Evaluator, *, regs_per_board: int = 2, global_regs: int = 16) -> int | None:
    """Open ``evaluator``'s epoch at ``search``'s root and set the search's epoch to it - for an
    evaluator whose evaluations depend on the root (``start(root)``: the cached path, C3's root
    refresh; ``CachedNetworkEvaluator``), so that a search sharing another's evaluation cache
    (C14) never reuses evaluations made under another refresh. The engine epoch set, or None
    for an evaluator without ``start`` (its evaluations are a function of the leaf's input alone:
    the epoch stays as it is). The root's position is the root state's own expansion with the
    search's expansion options."""
    start = getattr(evaluator, "start", None)
    if not callable(start):
        return None
    opts = search.params.expand
    relations = [rid for rid in range(32) if (int(opts.relations) >> rid) & 1]
    with search.root_game() as root:
        position = root.position(regs_per_board=regs_per_board, global_regs=global_regs, relations=relations, window_k=int(opts.window_k))
    number = engine_epoch(start(position))
    if search.info().epoch != number:
        search.set_epoch(number)
    return number


def run_search(
    search: Search,
    evaluator: Evaluator,
    *,
    playouts: int,
    batch_size: int = 8,
    regs_per_board: int = 2,
    global_regs: int = 16,
    value_kind: int = T5D_VALUE_LOGITS,
) -> tuple[ServeStats, CStepDriver]:
    """Run ``playouts`` simulations of ``search`` with ``evaluator`` through the step API
    (``CStepDriver`` and ``serve``, at most ``batch_size`` leaves per evaluation): the loop's
    statistics and the driver (its ``stop`` says why it ended). An evaluator whose evaluations
    depend on the search's root gets its epoch first (``start_epoch``)."""
    start_epoch(search, evaluator, regs_per_board=regs_per_board, global_regs=global_regs)
    driver = CStepDriver(search, playouts=playouts, regs_per_board=regs_per_board, global_regs=global_regs, value_kind=value_kind)
    stats = serve(driver, evaluator, batch_size=batch_size)
    return stats, driver


class UniformEvaluator:
    """Logits 0 for every move, no submit logit (the search's placeholder share), the value 0
    (every bin's mass on the centre bin) - the engine's built-in ``uniform`` evaluator, in
    Python: a search driven by it through the step API is the engine's own run, bit for bit."""

    def __init__(self, value_bins: int = 51) -> None:
        self.value_logits = np.full(value_bins, -1.0e4, dtype=np.float32)
        self.value_logits[value_bins // 2] = 0.0

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        return [Evaluation(np.zeros(move_count(p), dtype=np.float32), self.value_logits.copy()) for p in positions]


__all__ = [
    "CStepDriver",
    "CachedNetworkEvaluator",
    "Evaluation",
    "Evaluator",
    "Leaf",
    "NetworkEvaluator",
    "SearchDriver",
    "ServeStats",
    "UniformEvaluator",
    "check_evaluation",
    "engine_epoch",
    "run_search",
    "start_epoch",
    "serve",
    "state_of",
]
