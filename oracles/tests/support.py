"""Test fixtures: throwaway git repositories with a fake oracle pinned in them."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import config, golden, paths
from oracle_harness.lockfile import Lock, Pin
from oracle_harness.util import sha256_bytes

# Tests must not depend on, or be changed by, the machine's git configuration.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Harness Test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Harness Test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ALLOW_PROTOCOL": "file",
}

FAKE_LICENCE = "Fake Oracle Licence\n\nDo what you like.\n"

# Eight lines of "logic", long enough to fingerprint.
FAKE_ORACLE_SOURCE = """\
// fake oracle rules
#include <vector>
int count_legal_actions(const Position &position, int cap) {
    int total = 0;
    for (const Timeline &timeline : position.timelines()) {
        for (const Move &move : timeline.pseudo_legal_moves()) {
            if (!leaves_royal_in_check(position, move)) {
                total += 1;
                if (total == cap) return total;
            }
        }
    }
    return total;
}
"""

# One long line of data: copied whole, it is a copy on its own.
FAKE_ORACLE_TABLE = "static const int CENTRALITY[64] = {" + ", ".join(
    str(10 + (min(file, 7 - file) + min(rank, 7 - rank)) * 7) for rank in range(8) for file in range(8)) + "};\n"

FAKE_RECORD = '[Board "Standard"]\r\n[Mode "5D"]\r\n\r\n1. e3 / e6\r\n'

# The repository's own boundary policy, without its waivers: the tests hold the
# check to the lists it will actually be run with.
BOUNDARY = dataclasses.replace(config.load().boundary, waivers=())


def golden_files_predate_this_schema(source: str) -> str | None:
    """Why a test of the repository's own golden files cannot run yet, if it cannot.

    Golden files written in an older schema are refused by the harness, and
    `verify` reports them stale until they are re-recorded (golden --write). Until
    then a test that reads them has nothing to read: it is skipped with this
    reason, while `verify` fails. Any other problem with the files is not excused.
    """
    path = golden.golden_path(source)
    try:
        schema = json.loads(path.read_text(encoding="utf-8")).get("schema")
    except (OSError, ValueError, AttributeError):
        return None
    if schema == golden.SCHEMA:
        return None
    return (f"{path.name} is in golden schema {schema}, not {golden.SCHEMA}: re-record the golden files "
            "(python oracles/harness.py golden --write)")


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "protocol.file.allow=always", "-c", "core.autocrlf=false", *args],
        capture_output=True, text=True, check=False, env={**os.environ, **GIT_ENV},
    )
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {done.stderr}")
    return done.stdout.strip()


def _force_remove(function, path, _error):
    os.chmod(path, stat.S_IWRITE)
    function(path)


class Sandbox:
    """A superproject laid out like this repository, with one fake oracle pinned."""

    def __init__(self, test: unittest.TestCase):
        self.root = Path(tempfile.mkdtemp(prefix="oracle-harness-"))
        test.addCleanup(shutil.rmtree, self.root, onexc=_force_remove)
        patcher = mock.patch.dict(os.environ, GIT_ENV)
        patcher.start()
        test.addCleanup(patcher.stop)

        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        git(self.upstream, "init", "-q", "-b", "main")
        self.write(self.upstream / "LICENSE", FAKE_LICENCE)
        self.write(self.upstream / "src" / "rules.cpp", FAKE_ORACLE_SOURCE)
        self.write(self.upstream / "src" / "tables.cpp", FAKE_ORACLE_TABLE)
        (self.upstream / "games").mkdir()
        (self.upstream / "games" / "sample.5dpgn").write_bytes(FAKE_RECORD.encode())
        git(self.upstream, "add", "-A")
        git(self.upstream, "commit", "-q", "-m", "fake oracle")
        self.commit = git(self.upstream, "rev-parse", "HEAD")

        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        self.pin_path = "oracles/third_party/fake"
        git(self.repo, "submodule", "add", "-q", "--name", "oracle-fake", self.upstream.as_posix(), self.pin_path)
        self.checkout = self.repo / self.pin_path

        patcher = mock.patch.object(paths, "REPO_ROOT", self.repo)
        patcher.start()
        test.addCleanup(patcher.stop)

    @staticmethod
    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def add(self, relative: str, text: str) -> None:
        """Create a file in the superproject and stage it."""
        self.write(self.repo / relative, text)
        git(self.repo, "add", "--", relative)

    def pin(self, **overrides) -> Pin:
        values = dict(
            id="fake", role="oracle", upstream="example/fake-oracle", url=self.upstream.as_posix(),
            submodule="oracle-fake", path=self.pin_path, commit=self.commit,
            tree=git(self.upstream, "rev-parse", "HEAD^{tree}"), ref="main", committed="2026-01-01",
            licence="Fake", licence_file="LICENSE", licence_sha256=sha256_bytes(FAKE_LICENCE.encode()),
            summary="a fake oracle", used_as="in tests", tier="commit", notes=(),
            dependency_names=("fake-oracle-pkg",),
        )
        values.update(overrides)
        return Pin(**values)

    def lock(self, **overrides) -> Lock:
        return Lock(pins={"fake": self.pin(**overrides)})
