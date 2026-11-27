"""C27's full matrix, the slow rest of ``test_turn_refresh.py``'s T6: section 15.2's row "Tree form
with the cached configuration on: supersession recompute at the next evaluated node = recompute at
supersession" on every geometry and epoch, seeds 0 and 1 under their other causal-past reading and
seed 2 under both (the per-commit cases stay in ``test_turn_refresh.py``; T6 and its helpers are
that module's, imported here unchanged).

A file of its own (2026-11-28) because CI's shards never split a file: T6's whole matrix in one
module took about 51 minutes on the development machine and passed a 90-minute CI job alone.
"""

from __future__ import annotations

import pytest

from tests.net.test_cache import GEOMETRIES
from tests.net.test_turn_refresh import EPOCHS, GAPS, T6_SLOW_CASES, _t6


@pytest.fixture(scope="module", autouse=True)
def _report_gaps():
    yield
    print("\nC27 on the fixture (full matrix), largest gaps against the references:", {k: f"{v:.2e}" for k, v in GAPS.items()})


@pytest.mark.slow
@pytest.mark.parametrize("epoch", EPOCHS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_supersession_deferred_to_the_next_evaluated_node_is_exact_full_matrix(geometry, epoch):
    """The rest of T6's matrix: seeds 0 and 1 under their other reading, seed 2 under both."""
    _t6(geometry, epoch, T6_SLOW_CASES)
