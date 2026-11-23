"""Test helpers for the pool of concurrent games (``t5d/engine/pool.py``; register G12), shared by
``tests/engine/test_pool.py`` and ``tests/train/test_selfplay_pool.py``.

* ``DigestEvaluator`` - an evaluator that is an EXACT function of each position: whatever batch a
  position arrives in, and on whatever thread, its evaluation is the same bits. Random at a
  realistic scale (spec section 15.2: never zeros), and different for any two positions - so an
  evaluation handed to the wrong caller, or in the wrong order, shows.
* ``RecordingEvaluator`` - keeps every call it passes on (the positions and the evaluations).
* ``GatedEvaluator`` - holds its first calls until the test opens the gate: interleavings by
  events, never by sleeping.
* ``FailingEvaluator`` - raises on one call, or returns one evaluation too few.
* ``same_evaluation`` (bitwise, NaN equal to NaN), ``pool_threads`` (the pool's threads alive),
  ``wait_for`` (a condition polled against a deadline - for a state another thread reaches).
"""

from __future__ import annotations

import hashlib
import math
import threading
import time

import numpy as np

from t5d.engine.evaluator import Evaluation, state_of
from t5d.engine.pool import GAME_THREAD, SERVICE_THREAD
from t5d.net.cache import state_digest

#: How long a helper waits for another thread before the test fails (seconds): generous, never a
#: delay a passing test pays.
PATIENCE = 60.0


class DigestEvaluator:
    """An exact function of each position (module docstring): a generator seeded by BLAKE2b of the
    position's digest (``net.cache.state_digest``: boards, flags, acting player, turn state), its
    candidate moves and its submit flags draws float32 policy logits, 3 WDL logits, 51 value logits,
    a submit logit (kept only where the position offers a legal submit), the error and moves-left
    (>= 0) and one board logit per board - an evaluation that keeps ``check_evaluation``'s contract."""

    def __init__(self, value_bins: int = 51) -> None:
        self.value_bins = int(value_bins)

    def evaluate(self, position) -> Evaluation:
        state = state_of(position)
        h = hashlib.blake2b(digest_size=8)
        h.update(state_digest(state).encode("ascii"))
        h.update(np.ascontiguousarray(state.moves).tobytes())
        h.update(bytes([int(state.submit_possible), int(state.submit_legal)]))
        rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
        policy = rng.standard_normal(int(state.moves.shape[0])).astype(np.float32)
        wdl = rng.standard_normal(3).astype(np.float32)
        value = rng.standard_normal(self.value_bins).astype(np.float32)
        submit = float(np.float32(rng.standard_normal()))
        error = float(np.float32(rng.uniform(0.0, 0.5)))
        moves_left = float(np.float32(rng.uniform(0.0, 40.0)))
        boards = rng.standard_normal(state.n_boards).astype(np.float32)
        offered = bool(state.submit_possible and state.submit_legal)
        return Evaluation(
            policy_logits=policy,
            wdl_logits=wdl,
            value_logits=value,
            submit_logit=submit if offered else math.nan,
            error=error,
            moves_left=moves_left,
            board_logits=boards,
        )

    def __call__(self, positions) -> list[Evaluation]:
        return [self.evaluate(p) for p in positions]


class RecordingEvaluator:
    """Passes every call on to ``inner`` and keeps it: ``calls`` is the list of (positions,
    evaluations), in call order. It shows the pool what ``inner`` shows (``model``, ``policy``,
    ``max_batch``, where ``inner`` has them)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[list, list]] = []
        for name in ("model", "policy", "max_batch"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def __call__(self, positions):
        positions = list(positions)
        out = list(self.inner(positions))
        self.calls.append((positions, out))
        return out


class GatedEvaluator:
    """``inner``, with its first ``gated`` calls held until ``gate`` is set. ``entered`` is set once
    the first call has begun (it is then inside the evaluator, holding its batch); a held call fails
    after ``PATIENCE`` seconds rather than hang the suite."""

    def __init__(self, inner, gated: int = 1) -> None:
        self.inner = inner
        self.gated = int(gated)
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.calls = 0

    def __call__(self, positions):
        self.calls += 1
        self.entered.set()
        if self.calls <= self.gated and not self.gate.wait(PATIENCE):
            raise TimeoutError("the gated evaluator was never released")
        return self.inner(positions)


class FailingEvaluator:
    """``inner``, except on call ``fail_on`` (1 is the first): it raises ``error``, or - with
    ``short`` - returns one evaluation too few."""

    def __init__(self, inner, fail_on: int, *, error: BaseException | None = None, short: bool = False) -> None:
        self.inner = inner
        self.fail_on = int(fail_on)
        self.error = ArithmeticError("the evaluator failed (a test's fault)") if error is None else error
        self.short = short
        self.calls = 0

    def __call__(self, positions):
        self.calls += 1
        if self.calls == self.fail_on:
            if self.short:
                return self.inner(positions)[:-1]
            raise self.error
        return self.inner(positions)


def _same_array(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _same_float(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def same_evaluation(a: Evaluation, b: Evaluation) -> bool:
    """Two evaluations equal bit for bit (dtypes and shapes included; NaN equal to NaN)."""
    return (
        _same_array(a.policy_logits, b.policy_logits)
        and _same_array(a.wdl_logits, b.wdl_logits)
        and _same_array(a.value_logits, b.value_logits)
        and _same_array(a.board_logits, b.board_logits)
        and _same_float(float(a.submit_logit), float(b.submit_logit))
        and _same_float(float(a.error), float(b.error))
        and _same_float(float(a.moves_left), float(b.moves_left))
        and a.policy == b.policy
    )


def pool_threads() -> list[str]:
    """The names of the pool's threads still alive (the service thread and the game workers)."""
    return sorted(t.name for t in threading.enumerate() if t.name == SERVICE_THREAD or t.name.startswith(GAME_THREAD + "-"))


def wait_for(condition, what: str = "the condition") -> None:
    """Poll ``condition()`` until it holds: a state another thread is on its way to. Fails after
    ``PATIENCE`` seconds; a passing test never waits longer than the other thread takes."""
    deadline = time.monotonic() + PATIENCE
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.002)
