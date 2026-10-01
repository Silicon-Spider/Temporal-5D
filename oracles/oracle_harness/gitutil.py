"""Thin wrappers over the git CLI.

Pinned content is always read from the object store at the locked commit, never
from a work tree: a work tree can be edited, and on Windows its line endings
depend on the local autocrlf setting.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .util import HarnessError

# Paths are printed as they are, not octal-escaped.
_PLAIN_PATHS = ["-c", "core.quotepath=false"]
# An export must be byte-exact whatever the machine's line-ending settings are.
# (Only `git archive` needs this: `cat-file` never converts. `git status` must
# NOT get it - it has to judge a work tree by the settings it was checked out
# under, or every file of a CRLF checkout looks modified.)
_NO_CONVERSION = ["-c", "core.autocrlf=false", "-c", "core.eol=lf"]


def git_bytes(repo: Path, args: Sequence[str], *, check: bool = True) -> bytes:
    command = ["git", "-C", os.fspath(repo), *_PLAIN_PATHS, *args]
    done = subprocess.run(command, capture_output=True, check=False)
    if check and done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise HarnessError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return done.stdout


def git(repo: Path, args: Sequence[str], *, check: bool = True) -> str:
    return git_bytes(repo, args, check=check).decode("utf-8", "replace").strip()


def git_ok(repo: Path, args: Sequence[str]) -> bool:
    command = ["git", "-C", os.fspath(repo), *_PLAIN_PATHS, *args]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


def is_checkout(path: Path) -> bool:
    """True when ``path`` is the root of its own git work tree (an initialised submodule)."""
    if not (path / ".git").exists():
        return False
    top = git(path, ["rev-parse", "--show-toplevel"], check=False)
    return bool(top) and Path(top).resolve() == path.resolve()


def head_commit(repo: Path) -> str:
    return git(repo, ["rev-parse", "HEAD"])


def has_commit(repo: Path, commit: str) -> bool:
    return git_ok(repo, ["cat-file", "-e", f"{commit}^{{commit}}"])


def tree_of(repo: Path, commit: str) -> str:
    return git(repo, ["rev-parse", f"{commit}^{{tree}}"])


def blob_id(repo: Path, commit: str, path: str) -> str | None:
    out = git(repo, ["rev-parse", "--verify", "--quiet", f"{commit}:{path}"], check=False)
    return out or None


def read_blob(repo: Path, blob: str) -> bytes:
    return git_bytes(repo, ["cat-file", "blob", blob])


def read_path(repo: Path, commit: str, path: str) -> bytes:
    return git_bytes(repo, ["cat-file", "blob", f"{commit}:{path}"])


def list_tree(repo: Path, commit: str) -> list[tuple[str, str, str]]:
    """(mode, object id, path) for every entry of the commit's tree, recursively."""
    raw = git_bytes(repo, ["ls-tree", "-r", "-z", commit])
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, _kind, oid = meta.decode().split()
        entries.append((mode, oid, path.decode("utf-8", "replace")))
    return entries


def worktree_changes(repo: Path) -> list[str]:
    """Anything that makes the work tree differ from HEAD, including untracked files."""
    out = git(repo, ["status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"])
    return [line for line in out.splitlines() if line.strip()]


def export_commit(repo: Path, commit: str, destination: Path) -> None:
    """Write the exact tree of ``commit`` into ``destination`` (which must be empty)."""
    import io
    import tarfile

    destination.mkdir(parents=True, exist_ok=True)
    archive = git_bytes(repo, [*_NO_CONVERSION, "archive", "--format=tar", commit])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")


def tracked_files(repo: Path, *, include_untracked: bool) -> list[tuple[str, str]]:
    """(mode, path) for the index, plus untracked-but-not-ignored files when asked.

    Untracked files get mode "untracked": a boundary violation should be reported
    before it is committed, not after.
    """
    raw = git_bytes(repo, ["ls-files", "-s", "-z"])
    entries: list[tuple[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        entries.append((meta.decode().split()[0], path.decode("utf-8", "replace")))
    if include_untracked:
        raw = git_bytes(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
        entries += [("untracked", item.decode("utf-8", "replace")) for item in raw.split(b"\0") if item]
    return entries


def index_gitlink(repo: Path, path: str) -> str | None:
    """The commit a submodule path is pinned to in the index, if it is a gitlink there."""
    out = git(repo, ["ls-files", "-s", "--", path])
    for line in out.splitlines():
        meta, _, entry_path = line.partition("\t")
        mode, oid, _stage = meta.split()
        if mode == "160000" and entry_path == path:
            return oid
    return None
