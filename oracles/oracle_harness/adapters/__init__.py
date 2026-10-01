"""Sources of observations: the three oracles (five searches) and the engine."""

from __future__ import annotations

from ..lockfile import Lock
from ..util import HarnessError
from .base import RunOptions, Source
from .chessin5d import ChessIn5dSource
from .cwmtt import CwmttSource
from .engine import EngineSource
from .ftxi import FtxiSource

ORACLE_SOURCES = ("ftxi", "ftxi:naive", "cwmtt", "cwmtt:naive", "chessin5d")
# One source per implementation. How an implementation replays a record - which
# actions it accepts, how many it reads - does not depend on which of its
# searches then counts the actions, so that is compared once, through these.
PRIMARY_SOURCES = ("ftxi", "cwmtt", "chessin5d")

__all__ = ["ORACLE_SOURCES", "PRIMARY_SOURCES", "EngineSource", "RunOptions", "Source", "make_source", "oracle_of"]


def oracle_of(source: str) -> str:
    """The implementation a source belongs to: "ftxi:naive" is ftxi."""
    return source.split(":")[0]


def make_source(name: str, lock: Lock) -> Source:
    if name in ("ftxi", "ftxi:naive"):
        return FtxiSource(name, lock)
    if name in ("cwmtt", "cwmtt:naive"):
        return CwmttSource(name, lock)
    if name == "chessin5d":
        return ChessIn5dSource(lock)
    raise HarnessError(f"unknown source '{name}' (known: {', '.join(ORACLE_SOURCES)})")
