"""Recorded-answer integrity (decision H33).

Recorded answers are immutable, with committed content hashes verified before
every comparison; the comparison logic lives in harness code the engine adapter
cannot modify; harness changes require review. Here:

  * MANIFEST - the committed SHA-256 of every file in oracles/golden/ (the golden
    files, their cost files, the calibration reference), written by
    `golden --write`. golden.load() verifies a file against it before parsing a
    byte of it, and `verify` checks the whole directory: a file edited by hand,
    a file nobody recorded, a file that went missing. It lives beside the files
    it covers rather than in the lock file: the lock file holds the pins, a
    reviewed change with its own procedure, and every re-recording would
    otherwise touch it. The format is sha256sum's, so
    `cd oracles/golden && sha256sum -c MANIFEST` checks it without the harness
    (the harness folds CRLF to LF before hashing, so a Windows checkout with
    converted line endings hashes as committed).
    What MANIFEST catches is an edit that does not rewrite MANIFEST too: an
    accident, a partial edit, a bad merge, a file changed while a run lasts. A
    file edited together with its line of MANIFEST - one sha256sum away - passes
    it, and only review catches that. Two things make such an edit show in
    review: MANIFEST changes only with the files it covers, and what each tier
    requires is a table of DISAGREEMENTS.md, which `verify` re-renders from the
    golden and cost files and gate.toml (stability.required_table) - a cost file
    edited to shrink a required set changes a line of it;
  * a hash of every harness file, taken when a gate run starts and again when it
    ends - every file under oracles/ but its caches (.cache/ and the like, and
    __pycache__/, whose bytecode harness.py never runs) and the pinned oracles
    under third_party/: anything that changed, appeared or went while it ran -
    the engine adapter appending a known divergence to excuse itself, say, or
    planting a module - fails the run ("harness-modified"). What it does not
    cover is the oracle builds under .cache/build, which the gate executes; what
    they answer is still held to the golden files' required sets.

What remains is trust in review: CODEOWNERS puts oracles/ and .github/ (the
workflows that decide which tier runs, and CODEOWNERS itself) under its owner's
review, which binds once branch protection requires code-owner review.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import paths
from .util import Finding, HarnessError, sha256_text

MANIFEST = "MANIFEST"
_LINE = re.compile(r"^([0-9a-f]{64})  ([^/\\\s][^/\\]*)$")
# How many independent recordings of every source the golden files must hold, at
# the least (H32): one recording cannot show a value unstable. `golden --write`
# makes this many unless told otherwise, and `verify` refuses fewer.
MIN_RECORDINGS = 2
RERECORD = f"python {paths.HARNESS_REL}/harness.py golden --write --repeat {MIN_RECORDINGS}"


class IntegrityError(HarnessError):
    """A recorded-answer file that does not match its committed hash, or that has none."""


def manifest_path() -> Path:
    return paths.GOLDEN_DIR / MANIFEST


def recorded_files() -> list[Path]:
    """Every file in the golden directory that MANIFEST must cover: all of them but MANIFEST itself."""
    if not paths.GOLDEN_DIR.is_dir():
        return []
    return sorted(path for path in paths.GOLDEN_DIR.iterdir() if path.is_file() and path.name != MANIFEST)


def read_manifest() -> dict[str, str] | None:
    """File name -> its committed SHA-256; None when there is no MANIFEST."""
    path = manifest_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    entries: dict[str, str] = {}
    for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        if not line:
            continue
        match = _LINE.match(line)
        if match is None:
            raise IntegrityError(f"{MANIFEST} line {number} is not '<sha256>  <file name>': {line[:80]!r}")
        if match.group(2) in entries:
            raise IntegrityError(f"{MANIFEST} lists {match.group(2)} twice")
        entries[match.group(2)] = match.group(1)
    return entries


def render_manifest(hashes: dict[str, str]) -> str:
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))


def write_manifest(hashes: dict[str, str]) -> None:
    manifest_path().write_text(render_manifest(hashes), encoding="utf-8", newline="\n")


def verified_bytes(path: Path, manifest: dict[str, str] | None) -> bytes:
    """The bytes of a recorded-answer file, once they are shown to be the ones MANIFEST commits to."""
    data = path.read_bytes()
    if manifest is None:
        raise IntegrityError(f"{path.name}: there is no {MANIFEST}, so nothing shows these answers are the ones "
                             f"recorded - re-record them: {RERECORD}")
    expected = manifest.get(path.name)
    if expected is None:
        raise IntegrityError(f"{path.name} is not listed in {MANIFEST}: golden --write did not write it")
    if sha256_text(data) != expected:
        raise IntegrityError(f"{path.name} does not match its hash in {MANIFEST}: it was changed after golden --write "
                             "wrote it (recorded answers are never edited by hand)")
    return data


def verify_manifest() -> list[Finding]:
    """Does every file in the golden directory match MANIFEST, and MANIFEST list every file?"""
    try:
        manifest = read_manifest()
    except IntegrityError as error:
        return [Finding("golden-integrity", str(error))]
    files = {path.name: path for path in recorded_files()}
    if manifest is None:
        if not files:
            return []  # nothing recorded, nothing to vouch for: golden-missing says so
        return [Finding("golden-integrity",
                        f"oracles/golden has no {MANIFEST}: the recorded answers carry no committed hashes - "
                        f"re-record them: {RERECORD}")]
    findings = []
    for name in sorted(manifest.keys() - files.keys()):
        findings.append(Finding("golden-integrity", f"{MANIFEST} lists {name}, which is not there"))
    for name in sorted(files.keys() - manifest.keys()):
        findings.append(Finding("golden-integrity", f"{name} is in oracles/golden but not in {MANIFEST}: golden --write "
                                                    "did not write it"))
    for name in sorted(files.keys() & manifest.keys()):
        if sha256_text(files[name].read_bytes()) != manifest[name]:
            findings.append(Finding("golden-integrity", f"{name} does not match its hash in {MANIFEST}: changed after "
                                                        "golden --write wrote it"))
    return findings


# --- the harness itself, while a gate run lasts ------------------------------------


def _skipped(folder: Path) -> bool:
    """A directory whose files no gate run compares against or executes: a cache, or the pinned oracles."""
    return folder.name.startswith(".") or folder.name == "__pycache__" or folder.resolve() in (
        paths.THIRD_PARTY.resolve(), paths.cache_dir().resolve())


def _files_under(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for folder, children, files in os.walk(root):
        here = Path(folder)
        children[:] = sorted(child for child in children if not _skipped(here / child))
        found += [here / name for name in sorted(files)]
    return found


def harness_files() -> list[Path]:
    """Every harness file a gate run compares against or executes - and every other file beside them.

    Every file under oracles/ but its caches and the pinned oracles: the
    harness's code and tests, its entry point, the drivers, the recorded
    answers, the known divergences, the gate's policy, the corpus, the rendered
    disagreements and the pins - so that a module added while a run lasts fails
    it as surely as an edit. And those of them a test points elsewhere,
    wherever they are.
    """
    found = _files_under(paths.HARNESS_ROOT)
    for folder in (paths.DRIVERS, paths.GOLDEN_DIR, paths.CORPUS_DIR):
        found += _files_under(folder)
    found += [paths.DIVERGENCES_FILE, paths.GATE_FILE, paths.DISAGREEMENTS_FILE, paths.LOCK_FILE]
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(os.path.normcase(os.path.abspath(path)), path)
    return list(unique.values())


def _name(path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def hash_harness() -> dict[str, str]:
    """Each harness file (harness_files) -> the SHA-256 of its content, or "missing"."""
    hashes = {}
    for path in harness_files():
        try:
            hashes[_name(path)] = sha256_text(path.read_bytes())
        except FileNotFoundError:
            hashes[_name(path)] = "missing"
    return hashes


def changes(before: dict[str, str], after: dict[str, str]) -> list[Finding]:
    """Every harness file that changed, appeared or disappeared between two hash_harness() calls."""
    findings = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name, "missing"), after.get(name, "missing")
        if old == new:
            continue
        if old == "missing":
            what = "appeared"
        elif new == "missing":
            what = "disappeared"
        else:
            what = "changed"
        findings.append(Finding("harness-modified",
                                f"{name} {what} while the gate ran - what was compared is no longer what is on disk, "
                                "and nothing this run concluded stands"))
    return findings
