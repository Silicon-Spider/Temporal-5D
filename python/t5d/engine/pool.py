"""Token-budget batches and the pool of concurrent games behind one evaluator (register G12).

Spec section 7.8 (2026-11-15 review): at the sizes real games have the engine is batch-bound - a
44-board position is an eighteenth of a 240-board one - so one game's search cannot fill a batch
without paying for it in virtual loss. "Self-play fills batches across games": many games run at
once, each contributes a leaf or two, and "each call [is] filled to a token budget, small positions
many to a batch and a 240-board position alone", from "positions of similar size". Section 15.7's
Gate 2 names it as the pilot run's configuration. This module is that pool, in four layers:

* ``plan_batch`` / ``plan_batches`` - the batch planner, a pure function of the waiting positions'
  sizes. The budget counts PADDED tokens, ``len(batch) x the longest member`` (section 7.8: "a
  padded batch pays for its longest member"); the oldest waiting position is always served (no
  starvation; a position larger than the budget goes alone), and the others join nearest in size
  first while the batch stays within the budget, the position cap and the padding ceiling. A
  position's size is ``token_count`` - the tokens of the input the search hands over.
* ``TokenBudgetEvaluator`` - one caller, no threads: its positions re-batched by the planner.
* ``SharedEvaluator`` and its ``PooledEvaluator`` clients - many callers: each client is an
  ``Evaluator`` whose call queues its positions with one service thread, which plans batches over
  everything waiting, calls the real evaluator once per batch and hands every caller its own
  evaluations, in its own order. Nothing in the engine changes: between ``gather`` and ``deliver``
  a search is passive (its waiting batch holds virtual loss), so any number of searches can wait on
  one forward.
* ``GamePool`` - ``games`` worker threads, each playing one game at a time through its own client:
  ``run(play, configs)`` calls ``play(evaluator, config)`` for every config and yields the results
  as they complete. ``t5d.train.selfplay_pool`` runs the training generator on it.

**The numbers are PLACEHOLDERS.** Section 7.8 gives the token budget neither a unit nor a size,
and its gain is "a section 15.3 figure per board-count range, deferred to GPU time": Gate 1 step 2's
batching curve sets ``TOKEN_BUDGET``, ``MAX_POSITIONS``, ``MAX_PADDING`` and ``GAMES`` per card.
No speed is measured here, and ``PoolStats`` is information, never a section 15.3 figure.

**What a pooled game's evaluations are.** The pool adds nothing to a game and takes nothing from
it: a position goes to the real evaluator unchanged and its evaluation comes back to the call that
asked. So with an evaluator that is an exact function of each position, or with
``BatchConfig.merge`` off (every request is then the very call the game alone makes), a game played
in the pool is the game played alone, bit for bit - its record included. With MERGED forwards of a
real network that holds only up to the network's batch-equivalence tolerance (spec section 15.2:
"a position evaluated in a padded batch = the same position evaluated alone", held at 1e-5 on an
FP32 trunk by ``tests/net/test_invariance.py`` and, at this module's level, by
``tests/engine/test_pool.py``): a forward's last bits move with the batch's size and padding, so a
pooled game's evaluations can differ from the alone game's in their last bits, a search can then
choose otherwise and the two games diverge - and two pooled runs need not agree with each other,
since which positions share a batch follows thread timing. Measured on three tiny games
(docs/MEASUREMENTS.md, 2026-11-22): no merged-forward record equalled the alone game's in eight
runs - the raw outputs apart by up to 4.5e-7 - though the moves played were the same.
``merge=False`` is the deterministic mode, and the baseline arm of the batching curve.

**Threads.** Game workers are threads and one more thread serves the evaluator. The engine's C
calls release the GIL (``capi.py`` loads the library with ``ctypes.CDLL``) and so do torch's
operations; a leaf's copy-out and conversion, ``collate`` and the split of a batch's output hold
it. A search handle - and the evaluation cache a game's searches share (C14) - lives on its game's
thread only: the pool never shares one between games. Grad mode is thread-local and on in a new
thread, so the service thread and every worker run under ``torch.no_grad()``; the service thread
also takes the torch thread count of the thread that made the pool where its own differs.
Per-leaf Python cost has no figure yet (section 15.3: input assembly, "no estimate"): should it cap
a compiled inference path, workers become processes and ``PooledEvaluator`` is the boundary.

**Back-pressure.** A client serves one search at a time, so each game has at most one request
outstanding and the waiting positions number at most the games times a game's own batch size; the
token budget bounds every forward; ``GamePool`` pulls a config only when a worker is free and its
results queue is bounded (``results_ahead``), so workers stall when the consumer lags.

**Shutdown and failure.** ``close()`` releases every waiting caller at once (``PoolClosed``), then
waits for the forward in flight - a call into the evaluator or the engine cannot be interrupted;
the threads are daemons. An exception from the real evaluator, or a wrong number of evaluations,
closes the pool and reaches every caller then waiting as ``EvaluatorFailed`` (its cause the
original), built in the caller's own thread; ``GamePool.run`` raises it in the consumer whatever
``on_error`` says. A game's own exception reaches the consumer as ``GameFailed`` (``"raise"``) or
in its ``PoolResult`` (``"collect"``).

**Not built.** G12's match-play sentence - "at match play the batch is capped as a share of the
simulations so far, with speculative prefetch for what the cap leaves empty" - changes how one
search gathers and is not here. The pool serves the dense path only: an evaluator whose evaluations
depend on a search's root (``start``: the cached path's epochs) is refused. No worker processes.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .evaluator import Evaluation, Evaluator, PositionLike, policy_source, state_of

#: PLACEHOLDER (register G12; Gate 1 step 2's batching curve sets it per card): the padded tokens
#: of one forward, ``len(batch) x the longest member``. 65,536 is 22 positions at 44 boards (2,920
#: tokens each) and 4 at 240 (15,856). Section 7.8's two examples cannot both hold under one linear
#: budget - "tens of positions in every batch" at 44 boards needs at least 58,400 (twenty), "a
#: 240-board position alone" less than 31,712. Ruled 2026-11-23 (BUILD_LOG spec issue 724; the
#: user left the choice to the session that built this): the inference budget and the training
#: budget are two settings, both in padded tokens. This one, inference's, takes the first example
#: - a forward keeps no activations for a backward pass, so four 240-board positions share one -
#: and the trainer's takes the second (``train.recipe.PILOT_TOKEN_BUDGET``, 16,000). Neither
#: figure is measured.
TOKEN_BUDGET = 65_536
#: PLACEHOLDER (G12): the most positions in one forward, whatever their size.
MAX_POSITIONS = 64
#: PLACEHOLDER (G12; section 7.8: "bucketing by token count ... keeps the padding small"): the
#: largest share of a batch's padded tokens that may be padding.
MAX_PADDING = 0.25
#: PLACEHOLDER (G12): the games a ``GamePool`` runs at once.
GAMES = 32
#: The board-count ranges ``PoolStats`` reports by (section 15.7's, and I18's): their lower bounds
#: after the first.
BOARD_RANGES: tuple[int, ...] = (50, 100, 200)

#: The service thread's name, and the prefix of the game workers' (``t5d-game-<k>``).
SERVICE_THREAD = "t5d-evaluator"
GAME_THREAD = "t5d-game"
ON_ERROR: tuple[str, ...] = ("raise", "collect")

_POLL = 0.1  # seconds: how often a blocked queue operation looks at the stop flag


class PoolClosed(RuntimeError):
    """The pool was closed: a waiting request was cancelled, or a call came after the close."""


class EvaluatorFailed(RuntimeError):
    """The real evaluator raised, or returned the wrong number of evaluations (``__cause__``)."""


class GameFailed(RuntimeError):
    """A game's ``play`` raised (``GamePool.run`` under ``on_error="raise"``; ``__cause__``)."""

    def __init__(self, index: int, config: Any, message: str) -> None:
        super().__init__(message)
        self.index = index
        self.config = config


@dataclass(frozen=True)
class BatchConfig:
    """How batches are formed (module docstring). Every number is a PLACEHOLDER (G12)."""

    #: The padded tokens of one forward: ``len(batch) x the longest member``.
    token_budget: int = TOKEN_BUDGET
    #: The most positions in one forward.
    max_positions: int = MAX_POSITIONS
    #: The largest share of a batch's padded tokens that may be padding, in [0, 1).
    max_padding: float = MAX_PADDING
    #: Merge the waiting positions into planned batches. False: every request is its own call of
    #: the real evaluator, whole and unchanged - the call the game alone makes (the deterministic
    #: mode; the batching curve's baseline arm).
    merge: bool = True
    #: Merging only: how long the service thread holds the oldest waiting request for more to
    #: arrive, unless every client is already waiting or the planned batch is full. 0: greedy - a
    #: batch is whatever is waiting.
    max_wait_ms: float = 0.0

    def validate(self) -> None:
        if self.token_budget < 1 or self.max_positions < 1:
            raise ValueError("token_budget and max_positions must be positive")
        if not 0.0 <= self.max_padding < 1.0:
            raise ValueError("max_padding is a share of a batch's padded tokens: in [0, 1)")
        if not (math.isfinite(self.max_wait_ms) and self.max_wait_ms >= 0.0):
            raise ValueError("max_wait_ms must be finite and not negative")

    def to_dict(self) -> dict:
        return asdict(self)


def token_count(position: PositionLike) -> int:
    """A position's size for the budget: the tokens of its input (``66 N + 16`` for ``N`` 8x8 boards
    with the default registers). Under register A5 the network converts to piece tokens at its own
    edge (``NetworkEvaluator``), so this over-counts there - ``size_of`` is the hook."""
    return int(state_of(position).n_tokens)


def board_count(position: PositionLike) -> int:
    return int(state_of(position).n_boards)


def plan_batch(sizes: Sequence[int], cfg: BatchConfig) -> list[int]:
    """The next batch from positions of ``sizes``, given in arrival order: their indices, ascending.
    Index 0 - the oldest - is always in it; the others are tried nearest in size to it first (ties:
    the older) and join while the batch has at most ``max_positions`` members, its padded tokens
    (members x the longest) stay within ``token_budget`` and its padding within ``max_padding`` of
    them. A position that does not fit is passed over, not waited for. Pure and deterministic."""
    if not sizes:
        return []
    first = int(sizes[0])
    chosen = [0]
    longest = total = first
    for i in sorted(range(1, len(sizes)), key=lambda k: (abs(int(sizes[k]) - first), k)):
        n = len(chosen) + 1
        if n > cfg.max_positions or n * longest > cfg.token_budget:
            break  # full: any further member pads to at least the longest so far
        size = int(sizes[i])
        padded = n * max(longest, size)
        if padded > cfg.token_budget or padded - (total + size) > cfg.max_padding * padded:
            continue
        chosen.append(i)
        longest = max(longest, size)
        total += size
    return sorted(chosen)


def plan_batches(sizes: Sequence[int], cfg: BatchConfig) -> list[list[int]]:
    """Every position of ``sizes`` in a batch: ``plan_batch`` over what is left, until nothing is -
    each batch as indices into ``sizes``, each index exactly once."""
    left = list(range(len(sizes)))
    out: list[list[int]] = []
    while left:
        chosen = plan_batch([sizes[i] for i in left], cfg)
        out.append([left[k] for k in chosen])
        taken = set(chosen)
        left = [i for k, i in enumerate(left) if k not in taken]
    return out


def _is_full(sizes: Sequence[int], cfg: BatchConfig) -> bool:
    """Whether a planned batch of ``sizes`` can take no further position of its longest's size."""
    return len(sizes) >= cfg.max_positions or (len(sizes) + 1) * max(sizes) > cfg.token_budget


def _range_name(n_boards: int, ranges: tuple[int, ...] = BOARD_RANGES) -> str:
    """The board-count range ``n_boards`` falls in, named as ``train.selfplay.board_range`` does."""
    lo = 0
    for hi in ranges:
        if n_boards < hi:
            return f"{lo}-{hi - 1}"
        lo = hi
    return f"{lo}+"


def _check_inner(inner: Evaluator, cfg: BatchConfig) -> None:
    """Refuse an evaluator the pool cannot serve: one whose evaluations depend on a search's root
    (``start``), and, when merging, one that would split a planned batch again by a count."""
    if callable(getattr(inner, "start", None)):
        raise ValueError(
            "an evaluator with start(root) evaluates in the epoch of one search (the cached path, C2/C3): "
            "it cannot be shared by concurrent searches - the pool serves the dense path only (register G12)"
        )
    max_batch = getattr(inner, "max_batch", None)
    if cfg.merge and max_batch is not None and int(max_batch) < cfg.max_positions:
        raise ValueError(
            f"the evaluator's max_batch ({int(max_batch)}) is below the pool's max_positions ({cfg.max_positions}): "
            "it would split a planned batch again by a count - build it with max_batch >= max_positions"
        )


def _sizes(positions: Sequence[PositionLike], size_of: Callable[[PositionLike], int]) -> list[int]:
    sizes = [int(size_of(p)) for p in positions]
    if any(s < 1 for s in sizes):
        raise ValueError("a position's size must be positive")
    return sizes


@dataclass
class PoolStats:
    """What the pool's evaluator did - information for the run's manifest, never a section 15.3
    figure. A "forward" is one call of the real evaluator: one forward of the network when merging
    (``_check_inner``: the evaluator may not split a planned batch); with merging off a request the
    evaluator splits by its own count is still one call here."""

    forwards: int = 0
    positions: int = 0
    requests: int = 0  # calls of the callers (one per round of a search)
    largest_batch: int = 0
    tokens: int = 0  # the positions' own tokens
    padded_tokens: int = 0  # each forward's members x its longest member
    cross_caller_forwards: int = 0  # forwards holding positions of two or more requests
    split_requests: int = 0  # requests whose positions went in two or more forwards
    evaluate_seconds: float = 0.0  # inside the real evaluator
    wait_seconds: float = 0.0  # the callers' time from request to evaluations, summed over requests
    batch_histogram: dict[int, int] = field(default_factory=dict)  # forwards by their number of positions
    #: By the board-count range (``BOARD_RANGES``) of each forward's largest member: forwards,
    #: positions, tokens, padded_tokens, seconds.
    by_range: dict[str, dict] = field(default_factory=dict)

    def record(self, sizes: Sequence[int], boards: Sequence[int], callers: int, seconds: float) -> None:
        """Count one forward of positions of ``sizes`` and ``boards`` from ``callers`` requests."""
        n, tokens, padded = len(sizes), sum(sizes), len(sizes) * max(sizes)
        self.forwards += 1
        self.positions += n
        self.largest_batch = max(self.largest_batch, n)
        self.tokens += tokens
        self.padded_tokens += padded
        self.cross_caller_forwards += int(callers >= 2)
        self.evaluate_seconds += seconds
        self.batch_histogram[n] = self.batch_histogram.get(n, 0) + 1
        row = self.by_range.setdefault(
            _range_name(max(boards)), {"forwards": 0, "positions": 0, "tokens": 0, "padded_tokens": 0, "seconds": 0.0}
        )
        row["forwards"] += 1
        row["positions"] += n
        row["tokens"] += tokens
        row["padded_tokens"] += padded
        row["seconds"] += seconds

    @property
    def mean_batch(self) -> float:
        """Positions per forward."""
        return self.positions / self.forwards if self.forwards else 0.0

    @property
    def padding_share(self) -> float:
        """The share of the padded tokens that was padding."""
        return 1.0 - self.tokens / self.padded_tokens if self.padded_tokens else 0.0

    def fill(self, token_budget: int) -> float:
        """The share of the budget the forwards used: padded tokens over forwards x the budget."""
        return self.padded_tokens / (self.forwards * token_budget) if self.forwards else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["batch_histogram"] = {str(k): v for k, v in sorted(self.batch_histogram.items())}
        d["by_range"] = {k: dict(v) for k, v in sorted(self.by_range.items(), key=lambda kv: int(kv[0].rstrip("+").split("-")[0]))}
        d["mean_batch"] = self.mean_batch
        d["padding_share"] = self.padding_share
        return d


class TokenBudgetEvaluator:
    """One caller's positions through the batch planner, no threads: ``inner`` is called once per
    planned batch (``plan_batches``) and the evaluations come back in the caller's order. With
    ``merge`` off it is ``inner(positions)``. To the tolerance of spec section 15.2's padded-batch
    row each evaluation is the position's alone; bit for bit only with ``merge`` off or an evaluator
    exact per position (module docstring)."""

    def __init__(self, inner: Evaluator, cfg: BatchConfig | None = None, *, size_of: Callable[[PositionLike], int] = token_count) -> None:
        self.cfg = BatchConfig() if cfg is None else cfg
        self.cfg.validate()
        _check_inner(inner, self.cfg)
        self.inner = inner
        self._size_of = size_of
        self.stats = PoolStats()

    @property
    def policy(self) -> str:
        """The prior delivered: the real evaluator's (``evaluator.policy_source``)."""
        return policy_source(self.inner)

    @property
    def model(self):
        """The real evaluator's network (None for one without)."""
        return getattr(self.inner, "model", None)

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        positions = list(positions)
        if not positions:
            return []
        sizes = _sizes(positions, self._size_of)
        boards = [board_count(p) for p in positions]
        self.stats.requests += 1
        groups = plan_batches(sizes, self.cfg) if self.cfg.merge else [list(range(len(positions)))]
        results: list = [None] * len(positions)
        for group in groups:
            t0 = time.perf_counter()
            out = list(self.inner([positions[i] for i in group]))
            seconds = time.perf_counter() - t0
            if len(out) != len(group):
                raise RuntimeError(f"the evaluator returned {len(out)} evaluations for {len(group)} positions")
            for i, evaluation in zip(group, out, strict=True):
                results[i] = evaluation
            self.stats.record([sizes[i] for i in group], [boards[i] for i in group], 1, seconds)
        self.stats.split_requests += int(len(groups) > 1)
        return results


@dataclass(eq=False)
class _Request:
    """One call of a client: its evaluations by slot, filled by the service thread."""

    results: list
    remaining: int
    started: float
    forwards: int = 0  # the forwards its positions have gone in so far
    event: threading.Event = field(default_factory=threading.Event)
    #: Set when the request ends without its evaluations: ("failed", the evaluator's exception) or
    #: ("closed", None). The exception itself is built in the caller's thread.
    failure: tuple[str, BaseException | None] | None = None


@dataclass(frozen=True, eq=False)
class _Item:
    """One waiting position: its request, its slot there, its size and its board count."""

    request: _Request
    slot: int
    position: PositionLike
    size: int
    boards: int


class SharedEvaluator:
    """One evaluator behind many callers (module docstring). ``client()`` gives a caller its
    ``PooledEvaluator``; the service thread (``SERVICE_THREAD``, started with the first client)
    takes batches from everything waiting - planned by ``plan_batch`` when ``cfg.merge``, else the
    oldest request whole - and calls ``inner`` once per batch, outside the lock.

    ``stats`` is written by the service thread: read it once the pool is closed. ``size_of``: a
    position's size for the budget (``token_count``)."""

    def __init__(self, inner: Evaluator, cfg: BatchConfig | None = None, *, size_of: Callable[[PositionLike], int] = token_count) -> None:
        self.cfg = BatchConfig() if cfg is None else cfg
        self.cfg.validate()
        _check_inner(inner, self.cfg)
        self.inner = inner
        self.stats = PoolStats()
        self._size_of = size_of
        self._cond = threading.Condition()
        self._pending: list[_Item] = []  # arrival order
        self._open: set[_Request] = set()  # requests without all their evaluations: waiting or in flight
        self._clients = 0
        self._closed = False
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._torch_threads = torch.get_num_threads()

    # -- the callers' side --

    def client(self) -> PooledEvaluator:
        """A new client: an ``Evaluator`` for one search at a time (close it when its game ends)."""
        with self._cond:
            if self._closed:
                raise PoolClosed(self._closed_message()) from self._failure
            self._clients += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._serve, name=SERVICE_THREAD, daemon=True)
                self._thread.start()
        return PooledEvaluator(self)

    def _release(self) -> None:
        with self._cond:
            self._clients -= 1
            self._cond.notify_all()  # the clients still registered may now all be waiting

    def _closed_message(self) -> str:
        return "the shared evaluator is closed" + ("" if self._failure is None else f": its evaluator failed ({self._failure!r})")

    def _submit(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        positions = list(positions)
        if not positions:
            return []
        sizes = _sizes(positions, self._size_of)  # outside the lock
        boards = [board_count(p) for p in positions]
        request = _Request(results=[None] * len(positions), remaining=len(positions), started=time.perf_counter())
        with self._cond:
            if self._closed:
                raise PoolClosed(self._closed_message()) from self._failure
            self._open.add(request)
            self._pending.extend(_Item(request, slot, p, sizes[slot], boards[slot]) for slot, p in enumerate(positions))
            self.stats.requests += 1
            self._cond.notify_all()
        request.event.wait()
        if request.failure is not None:  # the exception is this thread's own: built here, not shared
            kind, cause = request.failure
            if kind == "failed":
                raise EvaluatorFailed(f"the pool's evaluator failed: {cause!r}") from cause
            raise PoolClosed("the shared evaluator was closed while this request waited")
        return request.results

    # -- the service thread --

    def _serve(self) -> None:
        try:
            if torch.get_num_threads() != self._torch_threads:
                torch.set_num_threads(self._torch_threads)
            with torch.no_grad():  # thread-local, and on in a new thread
                while True:
                    with self._cond:
                        batch = self._take()
                    if batch is None:
                        return
                    t0 = time.perf_counter()
                    results = list(self.inner([item.position for item in batch]))
                    seconds = time.perf_counter() - t0
                    if len(results) != len(batch):
                        raise RuntimeError(f"the evaluator returned {len(results)} evaluations for {len(batch)} positions")
                    with self._cond:
                        self._complete(batch, results, seconds)
        except BaseException as exc:  # noqa: BLE001 - whatever ends this thread must reach the callers
            with self._cond:
                self._failure = exc
                self._closed = True
                self._cancel(("failed", exc))
                self._cond.notify_all()

    def _take(self) -> list[_Item] | None:
        """The next batch (the lock held; waits for one), or None once the pool is closed."""
        while True:
            if self._closed:
                return None
            if not self._pending:
                self._cond.wait()
                continue
            if not self.cfg.merge:
                head = self._pending[0].request
                batch = [item for item in self._pending if item.request is head]
                self._pending = [item for item in self._pending if item.request is not head]
                return batch
            chosen = plan_batch([item.size for item in self._pending], self.cfg)
            hold = self._hold(chosen)
            if hold > 0.0:
                self._cond.wait(hold)
                continue
            taken = set(chosen)
            batch = [self._pending[i] for i in chosen]
            self._pending = [item for i, item in enumerate(self._pending) if i not in taken]
            return batch

    def _hold(self, chosen: list[int]) -> float:
        """Seconds still to wait for more positions before the planned batch goes (0: now)."""
        if self.cfg.max_wait_ms <= 0.0 or len(self._open) >= self._clients:
            return 0.0  # greedy, or no client is left to ask
        if _is_full([self._pending[i].size for i in chosen], self.cfg):
            return 0.0
        return self._pending[0].request.started + self.cfg.max_wait_ms / 1000.0 - time.perf_counter()

    def _complete(self, batch: list[_Item], results: list[Evaluation], seconds: float) -> None:
        now = time.perf_counter()
        requests = {id(item.request): item.request for item in batch}
        for request in requests.values():
            request.forwards += 1
        for item, evaluation in zip(batch, results, strict=True):
            request = item.request
            if request.failure is not None:
                continue  # cancelled while its forward ran
            request.results[item.slot] = evaluation
            request.remaining -= 1
            if request.remaining == 0:
                self._open.discard(request)
                self.stats.wait_seconds += now - request.started
                self.stats.split_requests += int(request.forwards > 1)
                request.event.set()
        self.stats.record([item.size for item in batch], [item.boards for item in batch], len(requests), seconds)

    def _cancel(self, failure: tuple[str, BaseException | None]) -> None:
        """End every open request without its evaluations (the lock held)."""
        for request in self._open:
            request.failure = failure
            request.event.set()
        self._open.clear()
        self._pending.clear()

    # -- lifetime --

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failure(self) -> BaseException | None:
        """The exception that closed the pool (the real evaluator's), or None."""
        return self._failure

    @property
    def clients(self) -> int:
        """The clients registered (made and not closed)."""
        return self._clients

    @property
    def waiting(self) -> int:
        """The requests without all their evaluations: waiting for a forward, or in one."""
        return len(self._open)

    @property
    def running(self) -> bool:
        """Whether the service thread is alive (after ``close``: a forward is still in flight)."""
        return self._thread is not None and self._thread.is_alive()

    def close(self, timeout: float | None = None) -> None:
        """Close the pool: every waiting caller raises ``PoolClosed`` at once and later calls are
        refused; then wait (at most ``timeout`` seconds) for the service thread - a forward in
        flight is finished, its evaluations dropped. Idempotent."""
        with self._cond:
            if not self._closed:
                self._closed = True
                self._cancel(("closed", None))
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def __enter__(self) -> SharedEvaluator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PooledEvaluator:
    """One caller's ``Evaluator`` behind a ``SharedEvaluator``: a call queues its positions and
    blocks until the service thread has evaluated them all. ``policy`` and ``model`` are the real
    evaluator's, so D9's self-play guard (``evaluator.policy_source``) and a record's network
    condition (``train.selfplay.network_info``) see through it. Deliberately without ``start`` (it
    is no root-dependent evaluator: ``evaluator.start_epoch`` leaves the search's epoch alone) and
    without ``max_batch``. It serves one search at a time: a second call while one waits is
    refused."""

    def __init__(self, shared: SharedEvaluator) -> None:
        self._shared = shared
        self._busy = threading.Lock()
        self._state = threading.Lock()
        self._closed = False

    @property
    def policy(self) -> str:
        return policy_source(self._shared.inner)

    @property
    def model(self):
        return getattr(self._shared.inner, "model", None)

    def __call__(self, positions: Sequence[PositionLike]) -> list[Evaluation]:
        if not self._busy.acquire(blocking=False):
            raise RuntimeError(
                "a pooled evaluator serves one search at a time: it was called while another call waited (one client per game)"
            )
        try:
            if self._closed:
                raise RuntimeError("this pooled evaluator is closed")
            return self._shared._submit(positions)
        finally:
            self._busy.release()

    def close(self) -> None:
        """Deregister from the pool (idempotent): the service thread stops counting on this caller."""
        with self._state:
            if self._closed:
                return
            self._closed = True
        self._shared._release()

    def __enter__(self) -> PooledEvaluator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass(frozen=True)
class PoolResult:
    """One game of a ``GamePool`` run: its config's index in the run, the config, what ``play``
    returned - or the exception it raised (``on_error="collect"``; an index of -1: the configs'
    iterable raised) - and ``play``'s wall time."""

    index: int
    config: Any
    value: Any
    error: BaseException | None
    seconds: float


_DONE = object()  # a worker's last word on the results queue


class GamePool:
    """``games`` concurrent games behind one evaluator (module docstring). ``evaluator`` is the real
    one; each game gets its own ``PooledEvaluator``. ``on_error``: ``"raise"`` - the first game that
    fails ends the run with ``GameFailed`` - or ``"collect"`` - its ``PoolResult`` carries the
    exception and the other games go on. ``results_ahead``: how many finished games may wait for
    the consumer (``games`` when None) before the workers stall. A pool runs once; use it as a
    context manager, or ``close()`` it."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        games: int = GAMES,
        batch: BatchConfig | None = None,
        on_error: str = "raise",
        results_ahead: int | None = None,
        size_of: Callable[[PositionLike], int] = token_count,
    ) -> None:
        if games < 1:
            raise ValueError("games must be positive")
        if on_error not in ON_ERROR:
            raise ValueError(f"on_error must be one of {ON_ERROR}")
        if results_ahead is not None and results_ahead < 1:
            raise ValueError("results_ahead must be positive (None: the number of games)")
        self.games = int(games)
        self.on_error = on_error
        self.results_ahead = self.games if results_ahead is None else int(results_ahead)
        self.shared = SharedEvaluator(evaluator, batch, size_of=size_of)
        self._stop = threading.Event()
        self._results: queue.Queue = queue.Queue(maxsize=self.results_ahead)
        self._feed_lock = threading.Lock()
        self._exhausted = False
        self._workers: list[threading.Thread] = []
        self._ran = False

    @property
    def evaluator(self) -> Evaluator:
        """The real evaluator."""
        return self.shared.inner

    @property
    def batch(self) -> BatchConfig:
        return self.shared.cfg

    @property
    def stats(self) -> PoolStats:
        """The evaluator's statistics (read them once the run has ended)."""
        return self.shared.stats

    def run(self, play: Callable[[Evaluator, Any], Any], configs: Iterable[Any]) -> Iterator[PoolResult]:
        """Play every config: ``play(evaluator, config)`` on a worker thread, ``evaluator`` that
        game's client. The results come in COMPLETION order (each names its config's index);
        ``configs`` is read lazily, one config per free worker, so it may be endless - stop
        reading and close the pool. The threads start with the first result asked for. Raises
        ``GameFailed`` (``on_error="raise"``), and ``EvaluatorFailed`` in either mode once the
        results completed before the evaluator's failure have been yielded."""
        if self._ran:
            raise RuntimeError("a GamePool runs once: make another for another run")
        if self._stop.is_set() or self.shared.closed:
            raise PoolClosed("the pool is closed")
        self._ran = True
        return self._iterate(play, enumerate(configs))

    def _iterate(self, play: Callable[[Evaluator, Any], Any], feed: Iterator[tuple[int, Any]]) -> Iterator[PoolResult]:
        try:
            self._workers = [
                threading.Thread(target=self._work, args=(play, feed), name=f"{GAME_THREAD}-{k}", daemon=True) for k in range(self.games)
            ]
            for worker in self._workers:
                worker.start()
            live = len(self._workers)
            while live:
                try:
                    item = self._results.get(timeout=_POLL)
                except queue.Empty:
                    if self._stop.is_set():
                        raise PoolClosed("the pool was closed while its run was being read") from None
                    continue
                if item is _DONE:
                    live -= 1
                    continue
                if item.error is not None and self.on_error == "raise":
                    what = "the configs' iterable raised" if item.index < 0 else f"game {item.index} failed"
                    raise GameFailed(item.index, item.config, f"{what}: {item.error!r}") from item.error
                yield item
            failure = self.shared.failure
            if failure is not None:  # never a run that ends silently short
                raise EvaluatorFailed(f"the pool's evaluator failed: {failure!r}") from failure
        finally:
            self.close()

    def _next(self, feed: Iterator[tuple[int, Any]]) -> tuple[int, Any] | BaseException | None:
        """The next (index, config); None at the end; the exception the iterable raised."""
        with self._feed_lock:
            if self._exhausted or self._stop.is_set():
                return None
            try:
                return next(feed)
            except StopIteration:
                self._exhausted = True
                return None
            except BaseException as exc:  # noqa: BLE001 - reported to the consumer
                self._exhausted = True
                return exc

    def _work(self, play: Callable[[Evaluator, Any], Any], feed: Iterator[tuple[int, Any]]) -> None:
        try:
            with torch.no_grad():  # thread-local, and on in a new thread
                while not self._stop.is_set() and not self.shared.closed:
                    job = self._next(feed)
                    if job is None:
                        return
                    if isinstance(job, BaseException):
                        self._put(PoolResult(-1, None, None, job, 0.0))
                        return
                    index, config = job
                    t0 = time.perf_counter()
                    try:
                        with self.shared.client() as evaluator:
                            value = play(evaluator, config)
                    except (PoolClosed, EvaluatorFailed):
                        return  # cancelled by the shutdown or by the evaluator's failure: the run reports that, not this game
                    except BaseException as exc:  # noqa: BLE001 - the game's own failure, for the consumer
                        result = PoolResult(index, config, None, exc, time.perf_counter() - t0)
                    else:
                        result = PoolResult(index, config, value, None, time.perf_counter() - t0)
                    self._put(result)
                    if result.error is not None and self.on_error == "raise":
                        return
        finally:
            self._put(_DONE)

    def _put(self, item: object) -> bool:
        """Queue ``item`` for the consumer, stalling while the queue is full (back-pressure) and
        giving up once the pool is stopped."""
        while True:
            try:
                self._results.put(item, timeout=_POLL)
                return True
            except queue.Full:
                if self._stop.is_set():
                    return False

    @property
    def alive(self) -> int:
        """The pool's threads still alive (0 after a ``close()`` that waited for them all)."""
        return int(self.shared.running) + sum(w.is_alive() for w in self._workers)

    def close(self, timeout: float | None = None) -> None:
        """Stop the run: no further config is started, every game waiting on the evaluator is
        cancelled (its ``play`` sees ``PoolClosed``), and the threads are joined, each for at most
        ``timeout`` seconds - a game inside an engine call ends that call first. Idempotent."""
        self._stop.set()
        self.shared.close(timeout)
        me = threading.current_thread()
        for worker in self._workers:
            if worker is not me:
                worker.join(timeout)

    def __enter__(self) -> GamePool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = [
    "BOARD_RANGES",
    "GAMES",
    "GAME_THREAD",
    "MAX_PADDING",
    "MAX_POSITIONS",
    "ON_ERROR",
    "SERVICE_THREAD",
    "TOKEN_BUDGET",
    "BatchConfig",
    "EvaluatorFailed",
    "GameFailed",
    "GamePool",
    "PoolClosed",
    "PoolResult",
    "PoolStats",
    "PooledEvaluator",
    "SharedEvaluator",
    "TokenBudgetEvaluator",
    "board_count",
    "plan_batch",
    "plan_batches",
    "token_count",
]
