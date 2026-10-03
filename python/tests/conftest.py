"""Shared fixtures. Every random input is seeded; models are built with fixed seeds."""

from __future__ import annotations

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
