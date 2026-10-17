"""The rulebook pin (decision H38).

docs/rules/README.md is the project's rulebook: when the engine and an oracle
disagree it decides, and every known divergence that is a rules matter cites it.
It is "pinned, gate re-run on change": gate.toml [rulebook] holds the SHA-256 of
the text the gate's known divergences were adjudicated against, and the static
checks - `verify`, and so every gate tier - fail once the file no longer matches
it. Updating the pin (`rulebook --pin`) is a deliberate, reviewed act: read the
change, re-check every known divergence that cites the rulebook, pin it, re-run
the gate.

The pin lives in gate.toml, with the gate's policy (and under the same review,
CODEOWNERS), not in golden/MANIFEST: MANIFEST is rewritten by every
`golden --write`, so re-recording the oracles would re-pin the rulebook as a
side effect - and the recorded answers do not depend on the rulebook at all.
The text is hashed with CRLF folded to LF, so a Windows checkout hashes like one
on Linux.
"""

from __future__ import annotations

from pathlib import Path

from . import config, paths
from .util import Finding, HarnessError, sha256_text

PIN_COMMAND = f"python {paths.HARNESS_REL}/harness.py rulebook --pin"


def rulebook_path(spec: config.Rulebook) -> Path:
    return paths.REPO_ROOT / spec.path


def digest(spec: config.Rulebook) -> str | None:
    """The SHA-256 of the rulebook's text as it is on disk now; None when there is no such file."""
    try:
        return sha256_text(rulebook_path(spec).read_bytes())
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None


def verify(spec: config.Rulebook) -> list[Finding]:
    """Is the rulebook still the text gate.toml pins?"""
    current = digest(spec)
    if current is None:
        return [Finding("rulebook-missing", f"the rulebook, {spec.path} (gate.toml [rulebook] path), is not there: "
                                            "the known divergences cite a text nobody can read")]
    if current != spec.sha256:
        return [Finding("rulebook-changed", (
            f"the rulebook changed: re-run the gate and update the pin with {PIN_COMMAND} - {spec.path} is pinned "
            f"at {spec.sha256[:12]} in gate.toml [rulebook] and is now {current[:12]}; it is a reviewed change "
            "(H38): read the diff, re-check every known divergence that cites the rulebook against it, then pin it "
            "and re-run the gate"))]
    return []


def pin(gate_file: Path | None = None) -> tuple[str, str]:
    """Write the rulebook's current hash into gate.toml [rulebook] sha256; returns (the old pin, the new one).

    Only that one value of the file changes: its comments and every other
    setting are kept as they are.
    """
    gate_file = gate_file or paths.GATE_FILE
    spec = config.load(gate_file).rulebook
    current = digest(spec)
    if current is None:
        raise HarnessError(f"the rulebook, {spec.path}, is not there: there is nothing to pin")
    config.write_sha256(gate_file, "rulebook", current)
    if config.load(gate_file).rulebook.sha256 != current:  # what was written must read back as the pin
        raise HarnessError(f"{gate_file.name}: the rulebook pin did not read back as written")
    return spec.sha256, current
