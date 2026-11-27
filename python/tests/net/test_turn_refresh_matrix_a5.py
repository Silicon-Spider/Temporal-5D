"""C27's full matrix under register A5, the slow rest of ``test_turn_refresh.py``'s T6 with a
piece-centric network (decision 7): seed 1 under the two-parent reading and the slow cases, on every
geometry and epoch (the per-commit case, seed 0 under the lineage reading, stays in
``test_turn_refresh.py``; T6 and its helpers are that module's, imported here unchanged).

A file of its own (2026-11-28) because CI's shards never split a file (see
``test_turn_refresh_matrix.py``).
"""

from __future__ import annotations

import pytest

from tests.net.test_cache import GEOMETRIES
from tests.net.test_turn_refresh import EPOCHS, GAPS, T6_CASES, T6_SLOW_CASES, _t6


@pytest.fixture(scope="module", autouse=True)
def _report_gaps():
    yield
    print("\nC27 on the fixture (A5, full matrix), largest gaps against the references:", {k: f"{v:.2e}" for k, v in GAPS.items()})


@pytest.mark.slow
@pytest.mark.parametrize("epoch", EPOCHS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_supersession_deferred_under_a5_full_matrix(geometry, epoch):
    """The rest of T6's matrix under A5: seed 1 under the two-parent reading, and the slow cases."""
    _t6(geometry, epoch, T6_CASES[1:] + T6_SLOW_CASES, piece_centric=True)
