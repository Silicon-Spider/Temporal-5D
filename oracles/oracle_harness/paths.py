"""Filesystem layout of the harness. Everything is derived from this file's location."""

from __future__ import annotations

import os
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = HARNESS_ROOT.parent

# The harness directory as the repository spells it ("oracles"). The boundary
# check is phrased in terms of this one name.
HARNESS_REL = HARNESS_ROOT.name

THIRD_PARTY = HARNESS_ROOT / "third_party"
THIRD_PARTY_REL = f"{HARNESS_REL}/third_party"
DRIVERS = HARNESS_ROOT / "drivers"

LOCK_FILE = HARNESS_ROOT / "oracles.lock.toml"
GATE_FILE = HARNESS_ROOT / "gate.toml"
CORPUS_DIR = HARNESS_ROOT / "corpus"
CORPUS_MANIFEST = CORPUS_DIR / "manifest.toml"
DIVERGENCES_FILE = HARNESS_ROOT / "known-divergences.toml"
GOLDEN_DIR = HARNESS_ROOT / "golden"
DISAGREEMENTS_FILE = HARNESS_ROOT / "DISAGREEMENTS.md"
PROVENANCE_FILE = REPO_ROOT / "PROVENANCE.md"


def cache_dir() -> Path:
    """Toolchains, oracle builds and run reports. Regenerable; never committed.

    ``ORACLE_HARNESS_CACHE`` relocates it, which matters on Windows when the
    repository path is long enough to push build trees past MAX_PATH.
    """
    override = os.environ.get("ORACLE_HARNESS_CACHE")
    return Path(override).resolve() if override else HARNESS_ROOT / ".cache"
