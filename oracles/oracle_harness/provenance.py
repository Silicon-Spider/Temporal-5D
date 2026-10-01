"""PROVENANCE.md: what came from where, under which licence, in which role.

The prose of that file is written by hand. The inventory inside it is rendered
from the lock file and the corpus manifest, and `verify` fails when the two have
drifted apart - a provenance file that lags the pins is worse than none.
"""

from __future__ import annotations

from pathlib import Path

from . import paths
from .corpus import LOCAL, Record
from .lockfile import Lock
from .util import Finding, HarnessError

BEGIN = "<!-- BEGIN GENERATED INVENTORY - python oracles/harness.py provenance --write -->"
END = "<!-- END GENERATED INVENTORY -->"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_block(lock: Lock, records: list[Record]) -> str:
    lines = [
        BEGIN,
        "",
        "### Pinned repositories",
        "",
        "| Id | Upstream | Pinned commit | Licence | Role | CI tier |",
        "|---|---|---|---|---|---|",
    ]
    for pin in lock.pins.values():
        lines.append(
            f"| `{pin.id}` | [{_cell(pin.upstream)}]({pin.url}) | `{pin.commit}` ({_cell(pin.ref)}, {pin.committed}) "
            f"| {_cell(pin.licence)} | {pin.role} | {pin.tier or '-'} |"
        )
    for pin in lock.pins.values():
        lines += ["", f"#### `{pin.id}` - {pin.upstream}", "", f"- **What it is:** {pin.summary}",
                  f"- **How it is used:** {pin.used_as}", f"- **Where:** `{pin.path}` (submodule `{pin.submodule}`)",
                  f"- **Licence file:** `{pin.licence_file}`, SHA-256 `{pin.licence_sha256}`"]
        lines += [f"- {note}" for note in pin.notes]

    lines += ["", "### Toolchains", "", "| Tool | Required | Why |", "|---|---|---|"]
    for name, table in lock.toolchains.items():
        if table.get("version"):
            required = f"exactly {table['version']}"
            if table.get("dist"):
                required += "; checksum-pinned archive for " + ", ".join(sorted(table["dist"]))
        elif table.get("requires"):
            required = table["requires"]
        elif table.get("min_version"):
            required = f"{table['min_version']} or newer"
        else:
            required = "any"
        if table.get("bootstrap_pip"):
            required += f"; `--bootstrap` installs {table['bootstrap_pip']}"
        lines.append(f"| {name} | {_cell(required)} | {_cell(table.get('reason', ''))} |")

    lines += ["", "### Corpus records", "",
              f"{len(records)} records, listed one by one with blob ids and hashes in `oracles/corpus/manifest.toml`.",
              "", "| Source | Records | Licence they are used under |", "|---|---|---|"]
    sources: dict[str, list[Record]] = {}
    for record in records:
        sources.setdefault(record.source, []).append(record)
    for source, items in sources.items():
        where = "this repository" if source == LOCAL else f"`{source}` (read in place at the pinned commit, not copied)"
        licences = sorted({item.licence for item in items})
        lines.append(f"| {where} | {len(items)} | {_cell('; '.join(licences))} |")
    lines += ["", END]
    return "\n".join(lines)


def _split(text: str) -> tuple[str, str, str]:
    start, stop = text.find(BEGIN), text.find(END)
    if start < 0 or stop < 0 or stop < start:
        raise HarnessError(f"{paths.PROVENANCE_FILE.name}: missing the generated-inventory markers")
    return text[:start], text[start:stop + len(END)], text[stop + len(END):]


def check(lock: Lock, records: list[Record], path: Path | None = None) -> list[Finding]:
    path = path or paths.PROVENANCE_FILE
    if not path.exists():
        return [Finding("provenance-missing", f"{path.name} does not exist")]
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    try:
        _, current, _ = _split(text)
    except HarnessError as error:
        return [Finding("provenance-markers", str(error))]
    if current != render_block(lock, records):
        return [Finding(
            "provenance-stale",
            f"{path.name}: the inventory no longer matches the lock file and corpus manifest "
            f"(run: python {paths.HARNESS_REL}/harness.py provenance --write)")]
    return []


def write(lock: Lock, records: list[Record], path: Path | None = None) -> bool:
    """Refresh the inventory in place. Returns True when the file changed."""
    path = path or paths.PROVENANCE_FILE
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    before, _, after = _split(text)
    updated = before + render_block(lock, records) + after
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True
