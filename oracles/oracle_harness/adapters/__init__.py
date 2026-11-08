"""Sources of observations: the rules oracles (their searches), the notation
repository's parser (a parse-only oracle), and the engine."""

from __future__ import annotations

from ..lockfile import Lock
from ..util import HarnessError
from .base import RunOptions, Source
from .chessin5d import ChessIn5dSource
from .cwmtt import CwmttSource
from .engine import EngineSource
from .ftxi import FtxiSource
from .kairos import KairosSource
from .notation import NotationSource

# The sources the gate runs, records (golden files) and compares: a tier may list
# only these, and a tier that lists them all over every record tier is a full one.
ORACLE_SOURCES = ("ftxi", "ftxi:naive", "cwmtt", "cwmtt:naive", "chessin5d", "notation")
# Sources that are wired - built, and asked by `observe` - but not yet enabled:
# no tier may list them, nothing compares them, and a recording of one - only by
# `golden --write --source <it>` - is checked apart from the enabled sources'
# (snapshot.recorded_pending). Kairos (decision H52) waits here: "kairos:parse", its
# parse mode (what it set up and how it replayed each record, no rules answer), then
# its two searches in full mode. CHECKPOINT 10 put both after the release tier's
# sign-off run, which leaves Kairos out (docs/BUILD_LOG.md, issues 530 and 574).
# Enabling a source is a reviewed change: move it into ORACLE_SOURCES, list it in
# gate.toml's nightly and release tiers, and record it - `golden --write --repeat 2
# --source <each>` on a quiet machine - with exactly one of its implementation's
# sources in PRIMARY_SOURCES at every stage, never two (each replay difference would
# be reported twice, and need two divergence entries). Kairos's: "kairos:parse" while
# parse mode is enabled alone - otherwise none of its refusals or lengths is compared.
# At full mode's enabling, one of two (the enabling chooses; issue 530): "kairos" in
# its place - and then a refusal of "kairos:parse"'s past where "kairos" was cut by
# its budget is compared through no source, and the ruleset may still gate on the
# others; or "kairos:parse" kept for good, which closes that gap but needs
# observations.judges_actions (an engine's hard refusal of an action it plays would
# go uncompared) and divergences.py's rule for replay entries (which names the
# implementation) changed first.
PENDING_SOURCES = ("kairos:parse", "kairos", "kairos:naive")
# One source per implementation (tests/test_repository.py holds every implementation
# a tier may run to exactly one). How an implementation replays a record - which
# actions it accepts and refuses, how many it reads - does not depend on which of its
# searches then counts the actions, so that is compared once, through these; the
# replay of a source not listed here is compared with nothing.
PRIMARY_SOURCES = ("ftxi", "cwmtt", "chessin5d", "notation")
# Sources that judge nothing. The parse-only oracles (lock file role "parse-oracle")
# read the notation: they say what position they set up at each ply, the side they
# read as moving next, and how many actions they read. Accepting an action is not a
# verdict from them - they apply any move they can place, legal or not - so their
# acceptance never contradicts a refusal, and never stands behind a replay as a rules
# oracle's would. A move they cannot apply is a soft rejection: the move text was not
# read, which says nothing about the move. An oracle asked in parse mode is treated
# the same way, by its source name, though it reads otherwise: "kairos:parse"
# resolves only moves Kairos generates and refuses an action that leaves a royal
# capturable (cannot_submit), but it answers no rules question, so its acceptance of
# an action is left uncompared by choice and it vouches for no ruleset (H47). Its
# refusals and where its reading ends are compared only through its implementation's
# primary source (above): not at all while it is pending.
PARSE_ONLY_SOURCES = ("notation", "kairos:parse")

__all__ = ["ORACLE_SOURCES", "PARSE_ONLY_SOURCES", "PENDING_SOURCES", "PRIMARY_SOURCES", "EngineSource", "RunOptions",
           "Source", "make_source", "oracle_of"]


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
    if name in ("kairos", "kairos:naive", "kairos:parse"):
        return KairosSource(name, lock)
    raise HarnessError(f"unknown source '{name}' (known: {', '.join(ORACLE_SOURCES + PENDING_SOURCES)})")
