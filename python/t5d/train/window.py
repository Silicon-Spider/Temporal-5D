"""Register I16's replay window: it grows sublinearly with the data, and I12's surprise weighting
applies within it (spec section 12.8: "KataGo sampled uniformly from a window that grew from
250,000 to about 22 million samples over its main run; here a sublinear growth rule").

The rule is KataGo's shape - ``window(N) = N`` up to ``c`` rows, then ``c (1 + b ((N / c)^a -
1) / a)`` - with KataGo's constants (c = 250,000, a = 0.75, b = 0.4) as PLACEHOLDERS until the
pilot run scales them to 5D (register part K; spec issue 236). Rows are counted as written, I12's
frequency copies included (a placeholder: KataGo counts its written rows). The window holds the
most recent rows; a batch is drawn uniformly from them.

Register I22 (section 12, "Stratified sampling"; provisional, off in the sampler): each row may be
written with its board-count range (``stratum``) and its training compute (``cost`` - an exact
Python int, ``sampler.row_cost``), which the window keeps per range under eviction (``strata``)
and can draw by (``sample_stratified``: a range by given probabilities, then a row uniformly inside
it - copies included, so surprise weighting stays inside each range). ``sample`` is unchanged.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    min_rows: int = 250_000  # c - KataGo's, PLACEHOLDER
    exponent: float = 0.75  # a - KataGo's, PLACEHOLDER
    expand: float = 0.4  # b - KataGo's, PLACEHOLDER

    def validate(self) -> None:
        if self.min_rows < 1 or not 0.0 < self.exponent < 1.0 or self.expand <= 0.0:
            raise ValueError("I16: min_rows >= 1, 0 < exponent < 1 (sublinear), expand > 0")


def window_rows(total: int, cfg: WindowConfig | None = None) -> int:
    """The window's size after ``total`` rows have been written (module docstring)."""
    cfg = WindowConfig() if cfg is None else cfg
    cfg.validate()
    c = cfg.min_rows
    if total <= c:
        return int(total)
    a, b = cfg.exponent, cfg.expand
    return int(min(total, c * (1.0 + b * ((total / c) ** a - 1.0) / a)))


class ReplayWindow:
    """The most recent ``window_rows(written)`` rows; ``add`` appends a row ``copies`` times -
    with its board-count range and training compute when given (register I22; module docstring)."""

    def __init__(self, cfg: WindowConfig | None = None) -> None:
        self.cfg = WindowConfig() if cfg is None else cfg
        self.cfg.validate()
        self.written = 0
        self._rows: deque = deque()
        # I22: each row's (stratum, cost), parallel to ``_rows``; each range's rows (the same
        # objects, in the window's order - the oldest row of the window is the oldest of its
        # range) and summed cost (Python ints: exact at any window size).
        self._meta: deque[tuple[int, int]] = deque()
        self._pools: dict[int, deque] = {}
        self._cost: dict[int, int] = {}

    def add(self, row, copies: int = 1, *, stratum: int = 0, cost: int = 1) -> None:
        stratum, cost = int(stratum), int(cost)
        if cost < 0:
            raise ValueError("I22: a row's training compute is non-negative")
        for _ in range(int(copies)):
            self._rows.append(row)
            self._meta.append((stratum, cost))
            self._pools.setdefault(stratum, deque()).append(row)
            self._cost[stratum] = self._cost.get(stratum, 0) + cost
            self.written += 1
        size = window_rows(self.written, self.cfg)
        while len(self._rows) > size:
            self._rows.popleft()
            old, old_cost = self._meta.popleft()
            self._pools[old].popleft()
            self._cost[old] -= old_cost
            if not self._pools[old]:
                del self._pools[old], self._cost[old]

    def __len__(self) -> int:
        return len(self._rows)

    def rows(self) -> list:
        return list(self._rows)

    def sample(self, n: int, rng: np.random.Generator) -> list:
        """``n`` rows drawn uniformly (with replacement) from the window."""
        if not self._rows:
            raise ValueError("the window is empty")
        idx = rng.integers(0, len(self._rows), size=int(n))
        return [self._rows[int(i)] for i in idx]

    def strata(self) -> dict[int, tuple[int, int]]:
        """Per board-count range with rows in the window: (its rows, their summed cost) - copies
        counted, rows written without a range in range 0 at cost 1 (register I22)."""
        return {k: (len(self._pools[k]), self._cost[k]) for k in sorted(self._pools)}

    def sample_stratified(self, n: int, rng: np.random.Generator, probabilities: Mapping[int, float]) -> list:
        """``n`` rows drawn with replacement, each by range first - ``probabilities`` by range
        index (``sampler.range_draw``; a range left out has probability 0) - then uniformly among
        that range's rows, copies included (register I22). ``ValueError`` for an empty window, a
        positive probability on a range with no rows, or probabilities that do not sum to 1."""
        if not self._rows:
            raise ValueError("the window is empty")
        if any(p < 0 for p in probabilities.values()) or any(p > 0 and k not in self._pools for k, p in probabilities.items()):
            raise ValueError("I22: a draw probability is non-negative, and positive only for a range with rows in the window")
        keys = sorted(self._pools)
        p = np.array([float(probabilities.get(k, 0.0)) for k in keys], dtype=np.float64)
        if abs(float(p.sum()) - 1.0) > 1e-9:
            raise ValueError("I22: the ranges' draw probabilities sum to 1")
        which = rng.choice(len(keys), size=int(n), p=p / p.sum())
        out = []
        for w in which.tolist():
            pool = self._pools[keys[w]]
            out.append(pool[int(rng.integers(0, len(pool)))])
        return out


__all__ = ["ReplayWindow", "WindowConfig", "window_rows"]
