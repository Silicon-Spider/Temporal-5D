"""The evaluator side of the search's external-evaluator protocol (stages N4, N5).

The search (engine stage E6a, PUCT over the engine's own tree) evaluates leaves outside
itself: it collects a batch of leaf positions, hands them to an evaluator, and takes back,
for each leaf, **policy logits over that leaf's candidate moves** (spec §8.1: the pointer
head scores generator-supplied moves only), **the submit action's logit** where the leaf
offers a legal submission (register D8, §8.2: read from the pooled global registers) and
**D3's value heads** (§8.3, corrected at CHECKPOINT 7): the WDL - the value search uses -
the 51-bin categorical over the root search value, the error head's output and moves-left
(D4). This module is that evaluator, written against an abstract interface:

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
  the network's input, delivers move logits, submit logits and the value heads; ``run_search``
  runs one search with it.

**The contract an evaluation keeps** (``check_evaluation``): ``policy_logits`` has exactly
one finite entry per candidate move of the leaf, in the leaf's move order - for a position
converted by ``convert.to_position`` that is the engine's move order, so logit ``i`` scores
the move ``t5d_game_apply_expanded(i)`` would play (and, for a search leaf, the move
``leaf.expansion.moves[i]``); ``submit_logit`` is finite where the leaf offers a legal
submission and NaN otherwise - not offered, or masked as illegal (C4); ``wdl_logits`` has 3
finite entries (win, draw, loss): **the value is W - L** of their softmax (``value_mean``),
backed up by the search with the draw probability beside it; ``value_logits`` - the
categorical, optional (None) - has the network's ``value_bins`` (51) finite entries, bins
centred on ``-1 + 2k/50`` (``net/heads.py``): its variance is the search's default
uncertainty signal (6m), its mean is not the value; ``error`` (the error head's output) and
``moves_left`` are >= 0, or NaN for none given. **Every value head is from the perspective of
the acting player** (register D8: the player whose action is in progress until it submits -
the present's colour except past the shift; the network's target convention,
``net/heads.py``, and the search's) - the side whose candidate moves the policy scores.
Promotions to different pieces get different logits: the policy adds a learned per-piece
offset (§8.1, D1; M20). The network reads the position from the acting player's
side (register A7: ranks mirrored and timelines negated when Black acts - ``net/state.py``);
that moves no token, so the logits come back in the leaf's move order for either colour, and
nothing here maps them.
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

#: The ``NetOutput`` field the error head's output is read from (D3; 6m's other uncertainty
#: signal) - the network's ``value_error`` (``net/heads.py``'s ``ValueHeads``, the network lane's
#: D3); an output without it gives NaN (none). Pinned by ``tests/engine/test_evaluator.py``: a
#: ``NetOutput`` field for the error head under any other name fails that test rather than going
#: unread. With ``uncertainty_source``
#: ``T5D_UNCERTAINTY_ERROR_HEAD`` and E2, E3 or E5's term on, the search refuses a delivery
#: without it (the engine's ``uncertainty_missing`` counts it where nothing reads it).
ERROR_HEAD_FIELD = "value_error"

#: Which policy head an evaluator delivers as the prior (register D9): the policy (``"normal"``), or
#: the optimistic policy (``"optimistic"``) - match play only, "kept out of self-play" (section
#: 8.1): ``netplay.play_game`` refuses an optimistic evaluator in self-play. The values are the
#: same heads' either way.
POLICY_SOURCES: tuple[str, ...] = ("normal", "optimistic")


def state_of(position: PositionLike) -> PositionState:
    """The ``PositionState`` of a converted position (or the state itself)."""
    return position.state if isinstance(position, Position) else position


def move_count(position: PositionLike) -> int:
    return int(state_of(position).moves.shape[0])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One leaf's evaluation (see the module docstring for the contract)."""

    policy_logits: np.ndarray  # (M,) float32: one per candidate move, in the leaf's move order
    wdl_logits: np.ndarray  # (3,) float32: win, draw, loss - D3's value, the acting player's perspective
    #: D3: the categorical over the root search value, (V,) float32 with V = value_bins, the acting
    #: player's perspective - or None (not given: the search's categorical signal is then 0).
    value_logits: np.ndarray | None = None
    #: D8: the submit action's logit, read from the pooled global registers - finite where the leaf
    #: offers a legal submit, NaN otherwise (not offered, or masked as illegal; NaN where it is
    #: legal stands for "none given": the search's placeholder share).
    submit_logit: float = math.nan
    #: D3: the error head's output (its predicted squared short-term error), >= 0; NaN: none.
    error: float = math.nan
    #: D4: moves-left, in the network's unit (turns, M20), >= 0; NaN: none.
    moves_left: float = math.nan
    #: D2's flat form: one "leave it" logit per board of the leaf's expansion (``board_count``,
    #: its order; NaN: none for that board - the search's ``leave_share``), read at the acting
    #: player's playable boards by the flat form only - or None: none given. The network gives
    #: its flat head's (``NetOutput.leave_logits``, every board; spec issue 241).
    board_logits: np.ndarray | None = None
    #: The prior the policy logits are (``POLICY_SOURCES``): the normal policy's, or D9's optimistic
    #: policy's - carried by the evaluation itself, so that a wrapper around an evaluator cannot hide
    #: it from the self-play guard (``serve``'s ``self_play``).
    policy: str = "normal"

    @property
    def wdl_probs(self) -> np.ndarray:
        """``(3,)``: win, draw, loss (softmax of the logits, float32)."""
        return _softmax(self.wdl_logits).astype(np.float32)

    @property
    def value_mean(self) -> float:
        """The value search uses (D3): W - L, on [-1, 1]."""
        p = _softmax(self.wdl_logits)
        return float(p[0] - p[2])

    @property
    def value_probs(self) -> np.ndarray:
        """The categorical (softmax of its logits, float32); ``ValueError`` if none was given."""
        if self.value_logits is None:
            raise ValueError("this evaluation has no categorical")
        return _softmax(self.value_logits).astype(np.float32)

    def _centres(self) -> np.ndarray:
        v = self.value_probs.shape[0]
        return (2.0 * np.arange(v) - (v - 1)) / (v - 1)

    @property
    def categorical_mean(self) -> float:
        """The categorical's mean on [-1, 1] - not the value (D3)."""
        return float((self.value_probs.astype(np.float64) * self._centres()).sum())

    @property
    def categorical_variance(self) -> float:
        """The categorical's variance - the search's default uncertainty signal (6m)."""
        p, c = self.value_probs.astype(np.float64), self._centres()
        m = float((p * c).sum())
        return float((p * (c - m) ** 2).sum())

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


def _split(out: NetOutput, states: Sequence[PositionState], policy: str = "normal") -> list[Evaluation]:
    """Per-position evaluations of a batch's output (row ``i`` is ``states[i]``): the policy
    (``policy``: the normal policy's logits and submit logit, or D9's optimistic policy's),
    the submit logit and D3's value heads - the WDL, the categorical, the error head's output
    (``ERROR_HEAD_FIELD``; NaN for an output without one) and moves-left in turns (``expm1`` of
    the head's ``log1p``, floored at 0) and D2's per-board "leave it" logits (``NetOutput.
    leave_logits``, one per board in the state's order - the expansion's; None for an output without
    them). The short-term values are training targets' heads the search does not read (section 8.3:
    the error head and D9 learn from them)."""
    if policy not in POLICY_SOURCES:
        raise ValueError(f"policy must be one of {POLICY_SOURCES}")
    moves_out, submit_out = out.policy_logits, out.submit_logits
    if policy == "optimistic":
        if out.optimistic_logits is None or out.optimistic_submit_logits is None:
            raise ValueError("an optimistic evaluator needs a network with D9's head (NetConfig.optimistic_policy)")
        moves_out, submit_out = out.optimistic_logits, out.optimistic_submit_logits
    logits = moves_out.detach().float().cpu()
    valid = out.move_valid.detach().cpu()
    values = out.value_logits.detach().float().cpu()
    wdl = out.wdl_logits.detach().float().cpu()
    error_out = getattr(out, ERROR_HEAD_FIELD, None)
    errors = None if error_out is None else error_out.detach().float().cpu().reshape(-1)
    moves_left = torch.expm1(out.moves_left_log.detach().double().cpu().clamp(min=0.0))
    submit = submit_out.detach().float().cpu()
    submit_valid = out.submit_valid.detach().cpu()
    leave_out = getattr(out, "leave_logits", None)
    leave = None if leave_out is None else leave_out.detach().float().cpu()
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
                wdl_logits=wdl[i].numpy().copy(),
                value_logits=values[i].numpy().copy(),
                submit_logit=s if legal else math.nan,  # masked (-inf) or absent: no logit
                error=math.nan if errors is None else float(errors[i]),
                moves_left=float(moves_left[i]),
                board_logits=None if leave is None else leave[i, : state.n_boards].numpy().copy(),
                policy=policy,
            )
        )
    return results


def _check_policy(model: T5DNet, policy: str) -> None:
    if policy not in POLICY_SOURCES:
        raise ValueError(f"policy must be one of {POLICY_SOURCES}")
    if policy == "optimistic" and getattr(model, "optimistic", None) is None:
        raise ValueError("an optimistic evaluator needs a network with D9's head (NetConfig.optimistic_policy)")


def policy_source(evaluator) -> str:
    """The prior an evaluator delivers (``POLICY_SOURCES``): its ``policy``, or ``"normal"`` for
    one that names none."""
    return getattr(evaluator, "policy", "normal")


class NetworkEvaluator:
    """The dense path: ``collate`` and one forward per ``max_batch`` positions. ``scope``
    overrides the network's attention scope (None: its own); the network refuses a
    block-causal override of a dense-scope network while its B16 is switched on
    (``T5DNet.make_context``: its frozen boards would lose their status with no B16 table to
    give it back). The model is used as given (put it in ``eval()`` mode); positions converted
    with other register counts than the network's are re-laid out (``state.relayout``, exact).
    ``policy``: the prior delivered (``POLICY_SOURCES``) - ``"optimistic"`` (D9) for match play
    only, on a network with the optimistic head."""

    def __init__(
        self,
        model: T5DNet,
        *,
        max_batch: int = 32,
        device: torch.device | str | None = None,
        scope: str | None = None,
        policy: str = "normal",
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        _check_policy(model, policy)
        self.policy = policy
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
            results.extend(_split(out, states, self.policy))
            self.batches += 1
            self.positions += len(states)
        return results


class CachedNetworkEvaluator:
    """The cached path (C2; ``net/cache.py``): each position evaluated incrementally - its
    frozen boards' K/V from the store, computed once per digest - in the epoch of the
    current search. ``start(root)`` opens a search (``CachedEvaluator.root_epoch``: a root
    refresh under ``CacheConfig.root_refresh``, else the causal epoch). ``policy``: as
    ``NetworkEvaluator``'s."""

    def __init__(self, cached: CachedEvaluator, *, policy: str = "normal") -> None:
        _check_policy(cached.model, policy)
        self.policy = policy
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
                results.extend(_split(out, [state], self.policy))
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
    logits as before; D3's heads - 3 finite WDL logits, the categorical None or ``value_bins``
    finite logits, the error and moves-left NaN or finite and >= 0; the submit logit (D8) never
    infinite, and NaN unless the leaf offers a legal submit (where NaN means "none given"); the
    prior one of ``POLICY_SOURCES``."""
    if evaluation.policy not in POLICY_SOURCES:
        raise ValueError(f"leaf {leaf.id}: the prior {evaluation.policy!r} is not one of {POLICY_SOURCES}")
    m = move_count(leaf.position)
    pl, wl, vl = evaluation.policy_logits, evaluation.wdl_logits, evaluation.value_logits
    if pl.shape != (m,):
        raise ValueError(f"leaf {leaf.id}: {pl.shape[0] if pl.ndim == 1 else pl.shape} policy logits for {m} candidate moves")
    if not np.all(np.isfinite(pl)):
        raise ValueError(f"leaf {leaf.id}: a policy logit is not finite")
    if np.shape(wl) != (3,):
        raise ValueError(f"leaf {leaf.id}: WDL logits of shape {np.shape(wl)}, expected (3,)")
    if not np.all(np.isfinite(wl)):
        raise ValueError(f"leaf {leaf.id}: a WDL logit is not finite")
    if vl is not None:
        if vl.ndim != 1 or (value_bins is not None and vl.shape[0] != value_bins):
            raise ValueError(f"leaf {leaf.id}: value logits of shape {vl.shape}, expected ({value_bins},)")
        if not np.all(np.isfinite(vl)):
            raise ValueError(f"leaf {leaf.id}: a value logit is not finite")
    for name in ("error", "moves_left"):
        x = float(getattr(evaluation, name))
        if not math.isnan(x) and not (math.isfinite(x) and x >= 0.0):
            raise ValueError(f"leaf {leaf.id}: the {name.replace('_', '-')} output is {x} (>= 0, or NaN for none)")
    s = float(evaluation.submit_logit)
    if math.isinf(s):
        raise ValueError(f"leaf {leaf.id}: the submit logit is infinite (masked or absent is NaN)")
    state = state_of(leaf.position)
    if not math.isnan(s) and not (state.submit_possible and state.submit_legal):
        raise ValueError(f"leaf {leaf.id}: a submit logit for a leaf that offers no legal submit")
    if evaluation.board_logits is not None:
        bl = np.asarray(evaluation.board_logits)
        if bl.shape != (state.n_boards,):
            raise ValueError(f"leaf {leaf.id}: board logits of shape {bl.shape}, expected ({state.n_boards},) - one per board")
        if np.isinf(bl).any():
            raise ValueError(f"leaf {leaf.id}: a board logit is infinite (NaN is none given)")


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
    driver: SearchDriver,
    evaluator: Evaluator,
    *,
    batch_size: int = 32,
    value_bins: int | None = 51,
    max_rounds: int | None = None,
    self_play: bool = False,
) -> ServeStats:
    """Run one search's evaluation loop: collect up to ``batch_size`` leaves, evaluate them
    in one call, check each evaluation against its leaf (``check_evaluation``), deliver;
    until the driver is finished (or ``max_rounds``). If evaluating or checking a batch fails,
    a driver with an ``abandon`` method is told before the exception leaves (the step API's
    virtual loss is released). ``self_play``: the search is self-play's, so an evaluation that
    carries D9's optimistic prior is refused before it is delivered (section 8.1: "kept out of
    self-play") - whatever wraps the evaluator, since each evaluation names its prior."""
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
                if self_play and ev.policy != "normal":
                    raise ValueError(f"leaf {leaf.id}: the {ev.policy} prior in self-play (register D9, section 8.1: match play only)")
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
    D2's flat form no leaf offers submit: the generator ends turns itself).

    ``deliver(results)``: every leaf of the batch, by leaf id: the move logits concatenated
    in leaf order, one submit logit per leaf - the evaluation's where the leaf offers a legal
    submit, NaN elsewhere (not offered; or illegal, which the search prunes) - and D3's heads,
    from the leaf's acting player: the 3 WDL logits, the 51 categorical logits (given for every
    leaf of the batch or for none), the error head's output and moves-left (NaN: none); and
    D2's flat form's per-board "leave it" logits where an evaluation gives them
    (``Evaluation.board_logits``: each leaf's ``expansion.board_count``, in its order; NaN for
    a leaf that gives none - the search's ``leave_share`` there). The network gives its flat
    head's logit for every board (``NetOutput.leave_logits``; spec issue 241), so a flat search
    it evaluates never falls back on ``leave_share``. Each leaf's move logits must number its
    candidate moves and its board logits its boards: a leaf whose do not is refused here, before
    the engine sees the batch (``ValueError``; the batch abandoned). A refused delivery raises
    the engine's error; the engine has abandoned the batch.

    The search's ``expand`` options must give every board's features (``"all"``): the
    network's input is the full history. Errors from the engine raise as exceptions
    (``errors.py``); ``abandon`` releases a waiting batch (``run_search`` and ``serve`` call it
    when evaluation fails). ``batch`` is the waiting batch's leaves (``LeafView``: key, epoch,
    D8's flags, the expansion), in the order the evaluator receives their positions.

    ``value_kind``: what an evaluation's ``wdl_logits`` and ``value_logits`` hold -
    ``T5D_VALUE_LOGITS`` (the network's, the default) or ``T5D_VALUE_PROBABILITIES`` (an
    evaluator with exact probabilities - the engine's built-in evaluators rewritten in Python,
    for bit-identical comparisons with the engine's own runs)."""

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
            # The flat form offers no submit action (the leaf says so: the generator ends turns).
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
        # Each leaf's own counts, checked leaf by leaf: concatenated, two leaves' errors could
        # cancel (one too many, the next one too few) and shift every later leaf's logits.
        for v, ev in zip(views, evs, strict=True):
            moves = np.asarray(ev.policy_logits).size
            if moves != len(v.expansion.moves):
                self.abandon()
                raise ValueError(f"deliver: leaf {v.index} has {moves} move logits for its {len(v.expansion.moves)} candidate moves")
            if ev.board_logits is not None and np.asarray(ev.board_logits).size != v.expansion.board_count:
                self.abandon()
                raise ValueError(
                    f"deliver: leaf {v.index} has {np.asarray(ev.board_logits).size} board logits for its {v.expansion.board_count} boards"
                )
        logits = np.concatenate([np.asarray(ev.policy_logits, dtype=np.float32).reshape(-1) for ev in evs] + [np.zeros(0, np.float32)])
        wdl = np.stack([np.asarray(ev.wdl_logits, dtype=np.float32) for ev in evs])
        given = [ev.value_logits is not None for ev in evs]
        if any(given) and not all(given):
            self.abandon()
            raise RuntimeError("deliver: a categorical for some leaves of the batch and not others (all or none)")
        values = np.stack([np.asarray(ev.value_logits, dtype=np.float32) for ev in evs]) if all(given) else None
        errors = np.array([ev.error for ev in evs], dtype=np.float32)
        moves_left = np.array([ev.moves_left for ev in evs], dtype=np.float32)
        boards = None
        if any(ev.board_logits is not None for ev in evs):
            # D2's flat form: each leaf's board logits, NaN for a leaf that gives none.
            boards = np.concatenate(
                [
                    np.full(v.expansion.board_count, np.nan, np.float32)
                    if ev.board_logits is None
                    else np.asarray(ev.board_logits, dtype=np.float32).reshape(-1)
                    for v, ev in zip(views, evs, strict=True)
                ]
            )
        offered = [v.submit_possible and v.submit_legal for v in views]
        submit = np.array([ev.submit_logit if ok else math.nan for ev, ok in zip(evs, offered, strict=True)], dtype=np.float32)
        self._waiting = None  # delivered or refused, the engine no longer holds the batch
        self.search.deliver(
            logits,
            wdl,
            values=values,
            errors=errors,
            moves_left=moves_left,
            submit_logits=submit,
            board_logits=boards,
            wdl_kind=self.value_kind,
            value_kind=self.value_kind,
        )

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
    self_play: bool = False,
) -> tuple[ServeStats, CStepDriver]:
    """Run ``playouts`` simulations of ``search`` with ``evaluator`` through the step API
    (``CStepDriver`` and ``serve``, at most ``batch_size`` leaves per evaluation): the loop's
    statistics and the driver (its ``stop`` says why it ended). An evaluator whose evaluations
    depend on the search's root gets its epoch first (``start_epoch``). ``self_play``: refuse
    D9's optimistic prior (``serve``) - a self-play search, including M21's generator, sets it."""
    if self_play and policy_source(evaluator) != "normal":
        raise ValueError("an optimistic evaluator searches at match play only, never in self-play (register D9, section 8.1)")
    start_epoch(search, evaluator, regs_per_board=regs_per_board, global_regs=global_regs)
    driver = CStepDriver(search, playouts=playouts, regs_per_board=regs_per_board, global_regs=global_regs, value_kind=value_kind)
    stats = serve(driver, evaluator, batch_size=batch_size, self_play=self_play)
    return stats, driver


class UniformEvaluator:
    """Logits 0 for every move, no submit logit (the search's placeholder share), WDL logits
    (0, 0, 0) - the value exactly 0, the draw probability 1/3 - and the categorical's mass on
    its centre bin, and every board's "leave it" logit 0 (D2's flat form reads them; the other
    splits read none) - the engine's built-in ``uniform`` evaluator, in Python: a search driven
    by it through the step API is the engine's own run, bit for bit."""

    def __init__(self, value_bins: int = 51) -> None:
        self.value_logits = np.full(value_bins, -1.0e4, dtype=np.float32)
        self.value_logits[value_bins // 2] = 0.0

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        return [
            Evaluation(
                np.zeros(move_count(p), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                self.value_logits.copy(),
                board_logits=np.zeros(state_of(p).n_boards, dtype=np.float32),
            )
            for p in positions
        ]


__all__ = [
    "CStepDriver",
    "CachedNetworkEvaluator",
    "ERROR_HEAD_FIELD",
    "Evaluation",
    "Evaluator",
    "Leaf",
    "NetworkEvaluator",
    "POLICY_SOURCES",
    "SearchDriver",
    "ServeStats",
    "UniformEvaluator",
    "check_evaluation",
    "engine_epoch",
    "policy_source",
    "run_search",
    "start_epoch",
    "serve",
    "state_of",
]
