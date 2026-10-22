"""Register I16's replay window: it grows sublinearly with the data, and I12's surprise weighting
applies within it (spec section 12.8: "KataGo sampled uniformly from a window that grew from
250,000 to about 22 million samples over its main run; here a sublinear growth rule").

The rule is KataGo's shape - ``window(N) = N`` up to ``c`` rows, then ``c (1 + b ((N / c)^a -
1) / a)`` - with KataGo's constants (c = 250,000, a = 0.75, b = 0.4) as PLACEHOLDERS until the
pilot run scales them to 5D (register part K; spec issue 236). Rows are counted as written, I12's
frequency copies included (a placeholder: KataGo counts its written rows). The window holds the
most recent rows; a batch is drawn uniformly from them.
"""

from __future__ import annotations

from collections import deque
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
    """The most recent ``window_rows(written)`` rows; ``add`` appends a row ``copies`` times."""

    def __init__(self, cfg: WindowConfig | None = None) -> None:
        self.cfg = WindowConfig() if cfg is None else cfg
        self.cfg.validate()
        self.written = 0
        self._rows: deque = deque()

    def add(self, row, copies: int = 1) -> None:
        for _ in range(int(copies)):
            self._rows.append(row)
            self.written += 1
        size = window_rows(self.written, self.cfg)
        while len(self._rows) > size:
            self._rows.popleft()

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


__all__ = ["ReplayWindow", "WindowConfig", "window_rows"]
