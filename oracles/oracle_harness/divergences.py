"""known-divergences.toml: the places where an oracle is wrong and the engine is right.

When the engine and an oracle disagree, the rulebook decides. If the engine is
wrong it is fixed. If the oracle is wrong, the disagreement is recorded here with
the rule that settles it, so that it stays expected without hiding anything else:

  * an entry excuses exactly one observation - this oracle, at this pin, on this
    record, at this position, saying this, where the engine says that;
  * what the entry says the oracle says is checked against the golden files, so
    an entry cannot describe an answer the oracle does not give;
  * if either value changes, the entry no longer matches and the gate fails;
  * if the disagreement stops occurring, the entry is stale and the gate fails;
  * an entry with no citation is not an entry.

The citation is normally to the rulebook (`rulebook = "..."`). One kind of
disagreement is not about the rules at all: an oracle that stops at move text it
cannot read ("no move I can generate matches this") where the engine plays on.
That may be the oracle's notation, not its rules, and then the authority is the
notation specification: such an entry cites it with `notation = "..."` instead.
An entry carries exactly one of the two.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import observations, paths
from .adapters import ORACLE_SOURCES, PRIMARY_SOURCES, oracle_of
from .corpus import Record
from .lockfile import Lock
from .observations import SOFT_REJECTIONS, Disagreement, Observation
from .util import Finding, HarnessError, load_toml

SCHEMA = 1
KINDS = ("side", "in_check", "legal_actions", "terminal", "legality", "length")
# Kinds that describe how an implementation replays the record. They are
# compared once per implementation, through its primary source.
REPLAY_KINDS = ("legality", "length")
_ID = re.compile(r"^KD-\d{4}$")
_PLACEHOLDER = re.compile(r"^\W*(tbd|todo|unknown|n/?a|none|fixme|\?+)\W*$", re.I)
_MIN_CITATION = 20

# (source, record id) -> what the golden file records that source saying, if anything.
Recorded = Callable[[str, str], Observation | None]


@dataclass(frozen=True)
class Divergence:
    id: str
    oracle: str  # the oracle source that is wrong, e.g. "chessin5d" or "cwmtt:naive"
    pin: str
    record: str
    record_sha256: str
    kind: str
    index: int
    engine_value: str
    oracle_value: str
    upstream: str
    added: str
    rulebook: str = ""  # the rule that settles it, and where it is stated
    notation: str = ""  # or: the part of the notation specification the oracle does not read
    notes: str = ""

    def key(self) -> tuple:
        return (self.oracle, self.record, self.kind, self.index)

    def matches(self, disagreement: Disagreement) -> bool:
        """``disagreement`` must be engine-vs-oracle, with the engine as side ``a``."""
        return (
            (disagreement.b, disagreement.record, disagreement.kind, disagreement.index) == self.key()
            and disagreement.a_value == self.engine_value
            and disagreement.b_value == self.oracle_value
        )


def load(path: Path | None = None) -> list[Divergence]:
    path = path or paths.DIVERGENCES_FILE
    data = load_toml(path)
    if data.get("schema") != SCHEMA:
        raise HarnessError(f"{path.name}: expected schema = {SCHEMA}")
    entries = []
    for number, table in enumerate(data.get("divergence", []), 1):
        try:
            entries.append(Divergence(
                id=table["id"], oracle=table["oracle"], pin=table["pin"], record=table["record"],
                record_sha256=table["record_sha256"], kind=table["kind"], index=table["index"],
                engine_value=table["engine_value"], oracle_value=table["oracle_value"],
                upstream=table["upstream"], added=table["added"],
                rulebook=table.get("rulebook", ""), notation=table.get("notation", ""), notes=table.get("notes", ""),
            ))
        except KeyError as missing:
            raise HarnessError(f"{path.name}: divergence #{number} is missing {missing}") from None
    return entries


def _may_be_notation(entry: Divergence) -> bool:
    """Is this the one kind of disagreement the notation specification can settle?"""
    if entry.kind == "length":
        return True
    soft = tuple(f"rejected:{reason}" for reason in SOFT_REJECTIONS)
    return entry.kind == "legality" and entry.oracle_value in soft


def verify(entries: list[Divergence], lock: Lock, records: list[Record], recorded: Recorded | None = None) -> list[Finding]:
    """Is every entry well formed, bound to the current pins and corpus, and true of the golden files?

    ``recorded`` looks an oracle's answer up in the golden files; without it the
    last of those checks is skipped.
    """
    findings: list[Finding] = []
    by_id = {record.id: record for record in records}
    seen_ids: set[str] = set()
    seen_keys: dict[tuple, str] = {}
    for entry in entries:
        tag = f"known divergence {entry.id}"
        if not _ID.match(str(entry.id)):
            findings.append(Finding("divergence-id", f"{tag}: ids look like KD-0001"))
        if entry.id in seen_ids:
            findings.append(Finding("divergence-duplicate", f"{tag}: id used twice"))
        seen_ids.add(entry.id)
        if entry.key() in seen_keys:
            findings.append(Finding("divergence-duplicate", f"{tag}: same observation as {seen_keys[entry.key()]}"))
        seen_keys[entry.key()] = entry.id

        known_oracle = entry.oracle in ORACLE_SOURCES
        if not known_oracle:
            findings.append(Finding("divergence-oracle", f"{tag}: oracle must be one of {ORACLE_SOURCES}"))
        else:
            pin = lock.pin(oracle_of(entry.oracle))
            if entry.pin != pin.commit:
                findings.append(Finding(
                    "divergence-pin",
                    f"{tag}: verified at {str(entry.pin)[:12]}, but {pin.id} is now pinned at {pin.commit[:12]} - "
                    "re-verify it against the new pin"))
        record = by_id.get(entry.record)
        if record is None:
            findings.append(Finding("divergence-record", f"{tag}: record '{entry.record}' is not in the corpus"))
        elif record.sha256 != entry.record_sha256:
            findings.append(Finding("divergence-record", f"{tag}: record '{entry.record}' has changed since this was recorded"))
        known_kind = entry.kind in KINDS
        if not known_kind:
            findings.append(Finding("divergence-kind", f"{tag}: kind must be one of {KINDS}"))
        elif entry.kind in REPLAY_KINDS and known_oracle and entry.oracle not in PRIMARY_SOURCES:
            findings.append(Finding(
                "divergence-oracle",
                f"{tag}: a {entry.kind} entry names the implementation, not one of its searches - "
                f"use '{oracle_of(entry.oracle)}'"))
        valid_index = isinstance(entry.index, int) and not isinstance(entry.index, bool) and entry.index >= 0
        if not valid_index:
            findings.append(Finding("divergence-index", f"{tag}: index must be a non-negative integer"))
        valid_values = True
        for name in ("engine_value", "oracle_value"):
            if not isinstance(getattr(entry, name), str) or not getattr(entry, name):
                valid_values = False
                findings.append(Finding("divergence-value", f"{tag}: {name} must be the value as the gate prints it, as a string"))
        if entry.engine_value == entry.oracle_value:
            findings.append(Finding("divergence-value", f"{tag}: engine_value and oracle_value are the same - that is not a divergence"))

        rulebook, notation = str(entry.rulebook).strip(), str(entry.notation).strip()
        if rulebook and notation:
            findings.append(Finding(
                "divergence-citation", f"{tag}: cites both the rulebook and the notation specification - say which one settles it"))
        elif notation and known_kind and valid_values and not _may_be_notation(entry):
            findings.append(Finding(
                "divergence-citation",
                f"{tag}: only an oracle that could not match the move text (rejected:{'|'.join(SOFT_REJECTIONS)}), "
                "or read a different number of actions, can be settled by the notation specification - "
                "this one needs the rulebook"))
        else:
            citation = rulebook or notation
            if len(citation) < _MIN_CITATION or _PLACEHOLDER.match(citation):
                findings.append(Finding(
                    "divergence-citation",
                    f"{tag}: needs the citation that settles it - the rule (or, for unread notation, the part of the "
                    "notation specification), and where it is stated"))
        upstream = str(entry.upstream).strip()
        reported = upstream.startswith(("https://", "http://"))
        unreported = upstream.lower().startswith("unreported:") and len(upstream) > len("unreported:") + 5
        if not (reported or unreported):
            findings.append(Finding("divergence-upstream", f"{tag}: upstream must be the URL of the report, or 'unreported: <why>'"))
        try:
            date.fromisoformat(str(entry.added))
        except ValueError:
            findings.append(Finding("divergence-date", f"{tag}: added must be an ISO date (YYYY-MM-DD)"))

        if recorded is not None and known_oracle and record is not None and known_kind and valid_index and valid_values:
            golden = recorded(entry.oracle, entry.record)
            said = None if golden is None else observations.value_at(golden, entry.kind, entry.index)
            if said is None:
                findings.append(Finding(
                    "divergence-golden",
                    f"{tag}: the golden file records nothing from {entry.oracle} about {entry.kind} at {entry.index} "
                    f"of {entry.record} - an entry is about an answer the oracle gives"))
            elif said != entry.oracle_value:
                findings.append(Finding(
                    "divergence-golden",
                    f"{tag}: says {entry.oracle} answers {entry.oracle_value}; the golden file has it answering {said}"))
    return findings


def looked_at(entry: Divergence, engine: Observation, oracle: Observation) -> bool:
    """Did both reports actually cover the observation this entry is about?

    Only then can its absence mean the divergence is gone. A run that was cut
    short before reaching it has simply not looked.
    """
    if entry.kind in REPLAY_KINDS:
        return all(report.status in ("ok", "rejected") for report in (engine, oracle))
    for report in (engine, oracle):
        if entry.index >= len(report.positions) or getattr(report.positions[entry.index], entry.kind, None) is None:
            return False
    return True


def apply(
    entries: list[Divergence],
    disagreements: list[Disagreement],
    checked: set[str],
) -> tuple[list[Disagreement], list[Divergence]]:
    """Sort engine-vs-oracle disagreements into (unexplained, stale entries).

    ``checked`` holds the ids of the entries whose observation this run looked at
    (see looked_at); an entry is stale when it was looked at and did not occur.
    """
    unexplained = []
    used: set[str] = set()
    for disagreement in disagreements:
        match = next((entry for entry in entries if entry.matches(disagreement)), None)
        if match is None:
            unexplained.append(disagreement)
        else:
            used.add(match.id)
    stale = [entry for entry in entries if entry.id not in used and entry.id in checked]
    return unexplained, stale
