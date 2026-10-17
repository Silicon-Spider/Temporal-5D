"""Log-scale buckets of integer offsets, clamped (T5's relative buckets; register B12).

Board-pair smolgen's relative cells (``smolgen.py``) index its per-head table by these
buckets of the signed Delta-time and the absolute Delta-timeline between two boards. They
were B6's binned biases' buckets (stage N1); B6 is superseded by B15's displacement bias
(CHECKPOINT 6) and removed, and its buckets live on here for B12, whose register row
records it "as built over relative board-pair cells".

A bucket of an offset's size is exact below ``n_exact``, then log2-spaced up to
``max_distance``, then clamped: every larger offset shares the last bucket.
"""

from __future__ import annotations

import math

import torch


def log_bucket(distance: torch.Tensor, n_exact: int, max_distance: int) -> torch.Tensor:
    """Bucket of an integer offset's size ``|distance|``: exact below ``n_exact``, then
    log2-spaced up to ``max_distance``, clamped (every larger offset shares the last bucket)."""
    distance = distance.abs()
    n_log = int(math.floor(math.log2(max_distance / n_exact)))
    safe = distance.clamp(min=1).float()
    log_idx = n_exact + torch.floor(torch.log2(safe / n_exact)).long()
    log_idx = log_idx.clamp(max=n_exact + n_log)
    return torch.where(distance < n_exact, distance.long(), log_idx)


def n_log_buckets(n_exact: int, max_distance: int) -> int:
    """How many buckets ``log_bucket`` uses."""
    return n_exact + int(math.floor(math.log2(max_distance / n_exact))) + 1


def signed_log_bucket(offset: torch.Tensor, n_exact: int, max_distance: int) -> torch.Tensor:
    """Bucket of a signed offset: ``b`` for ``offset >= 0`` and ``n - 1 + b`` for ``offset < 0``
    (``b >= 1`` there), ``b`` the bucket of ``|offset|`` and ``n`` their count - so the two
    signs share bucket 0 and nothing else."""
    n = n_log_buckets(n_exact, max_distance)
    b = log_bucket(offset, n_exact, max_distance)
    return torch.where(offset >= 0, b, n - 1 + b)


def n_signed_log_buckets(n_exact: int, max_distance: int) -> int:
    """How many buckets ``signed_log_bucket`` uses: ``2n - 1``."""
    return 2 * n_log_buckets(n_exact, max_distance) - 1
