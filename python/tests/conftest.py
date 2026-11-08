"""Shared fixtures. Every random input is seeded; models are built with fixed seeds."""

from __future__ import annotations

import os

import pytest
import torch

from t5d.net import NetConfig, build_model


@pytest.fixture(scope="session")
def cfg_fp32() -> NetConfig:
    """The F1 network with an FP32 trunk: exact comparisons need FP32 arithmetic."""
    return NetConfig(trunk_dtype="fp32")


@pytest.fixture(scope="session")
def model_fp32(cfg_fp32: NetConfig):
    return build_model(cfg_fp32, seed=0).eval()


@pytest.fixture(scope="session")
def model_default():
    """The default configuration exactly as shipped (BF16 trunk, G1)."""
    return build_model(NetConfig(), seed=0).eval()


@pytest.fixture
def engine_required():
    """A test that needs the engine library (``tests.engine.helpers.requires_engine``): runs
    when it loads, fails when a library is found but refused (or none is built under
    ``T5D_REQUIRE_ENGINE=1``), skips only when none is built and none is required."""
    from tests.engine.helpers import engine_required as decide

    decide()


@pytest.fixture(autouse=True)
def _seeded_global_rng():
    """Each test sees torch's global RNG seeded to the same state, and leaves it as it
    found it: a draw a test forgot to seed cannot depend on which tests ran before it."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        yield


@pytest.fixture(autouse=True)
def _no_grad_by_default(request):
    """Tests run without autograd unless marked ``grad``."""
    if request.node.get_closest_marker("grad"):
        yield
    else:
        with torch.no_grad():
            yield


def pytest_configure(config):
    config.addinivalue_line("markers", "grad: the test needs autograd")
    config.addinivalue_line("markers", "slow: a long test (the full 16-layer network); deselect with -m 'not slow'")


def pytest_collection_modifyitems(config, items):
    """``T5D_TEST_SHARD=i/n`` keeps shard ``i`` of ``n`` of the collected test files (CI runs the
    suite as ``n`` parallel jobs, ``i`` from 0). The unit is the file, never the test: some
    modules' tests share module-level state in order (a case that checks what its siblings
    recorded), so a module is never split. Files go to shards greedily by test count, largest
    first (ties by path), each to the shard with the fewest tests so far - so the shards are
    disjoint, their union is the whole collection, and every job that collects the same files
    (CI's jobs check out one commit) computes the same split. Unset, every test runs."""
    spec = os.environ.get("T5D_TEST_SHARD", "").strip()
    if not spec:
        return
    index, _, count = spec.partition("/")
    i, n = int(index), int(count)
    if not 0 <= i < n:
        raise pytest.UsageError(f"T5D_TEST_SHARD={spec!r}: want i/n with 0 <= i < n")
    sizes: dict[str, int] = {}
    for item in items:
        module = item.nodeid.split("::", 1)[0]
        sizes[module] = sizes.get(module, 0) + 1
    loads = [0] * n
    shard_of: dict[str, int] = {}
    for module, size in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        target = min(range(n), key=lambda k: (loads[k], k))
        shard_of[module] = target
        loads[target] += size
    kept, dropped = [], []
    for item in items:
        (kept if shard_of[item.nodeid.split("::", 1)[0]] == i else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept
