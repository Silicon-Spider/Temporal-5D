"""Sources of observations: the three rules oracles (five searches), the notation
repository's parser (a parse-only oracle), and the engine."""

from __future__ import annotations

from ..lockfile import Lock
from ..util import HarnessError
from .base import RunOptions, Source
from .chessin5d import ChessIn5dSource
from .cwmtt import CwmttSource
from .engine import EngineSource
from .ftxi import FtxiSource
from .notation import NotationSource

ORACLE_SOURCES = ("ftxi", "ftxi:naive", "cwmtt", "cwmtt:naive", "chessin5d", "notation")
# One source per implementation. How an implementation replays a record - which
# actions it accepts, how many it reads - does not depend on which of its
# searches then counts the actions, so that is compared once, through these.
PRIMARY_SOURCES = ("ftxi", "cwmtt", "chessin5d", "notation")
# Sources that read the notation and judge nothing (lock file role
# "parse-oracle"): they say what position they set up at each ply, the side they
# read as moving next, and how many actions they read. Accepting an action is
# not a verdict from them - they apply any move they can place, legal or not -
# so their acceptance never contradicts a refusal, and never stands behind a
# replay as a rules oracle's would. A move they cannot apply is a soft rejection:
# the move text was not read, which says nothing about the move.
PARSE_ONLY_SOURCES = ("notation",)

__all__ = ["ORACLE_SOURCES", "PARSE_ONLY_SOURCES", "PRIMARY_SOURCES", "EngineSource", "RunOptions", "Source",
           "make_source", "oracle_of"]


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
    if name == "notation":
        return NotationSource(lock)
    raise HarnessError(f"unknown source '{name}' (known: {', '.join(ORACLE_SOURCES)})")
