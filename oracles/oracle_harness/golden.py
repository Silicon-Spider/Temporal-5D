"""Golden observations: what each pinned oracle says about each pinned record.

Pinned oracles over a pinned corpus have pinned answers. Committing those answers
does three jobs:

  * CI can tell that an oracle built on another machine, compiler or OS still
    says what it said when it was reviewed;
  * a pin bump shows up as a reviewable diff of verdicts, not just of a hash;
  * the places where the oracles contradict each other are on record before the
    engine exists (DISAGREEMENTS.md is rendered from these files).

A golden value is a statement by that oracle, not a statement of truth.

The files hold verdicts only. An oracle's free-text messages stay out of them:
they quote the record they are about, and no record is redistributed from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import observations, paths
from .adapters import ORACLE_SOURCES, PRIMARY_SOURCES, RunOptions, Source, oracle_of
from .corpus import Record
from .observations import OBSERVABLES, Disagreement, Observation
from .runner import Results
from .util import Finding, HarnessError

SCHEMA = 1
_RECORDED = ("status", "reason", "actions_accepted", "last_action")


@dataclass
class Golden:
    source: str
    signature: dict
    sha256: dict[str, str]  # record id -> hash of the record these answers are about
    observations: dict[str, Observation]


def golden_path(source: str) -> Path:
    return paths.GOLDEN_DIR / f"{source.replace(':', '-')}.json"


def _cap(signature: dict) -> int | None:
    return signature.get("cap")


def dump(source: Source, options: RunOptions, records: list[Record], results: Results) -> str:
    """Deterministic text, one position per line, so a changed verdict is a one-line diff."""
    lines = ["{", f'  "schema": {SCHEMA},', f'  "source": {json.dumps(source.name)},',
             f'  "signature": {json.dumps(source.signature(options), sort_keys=True)},', '  "records": {']
    ordered = sorted(records, key=lambda record: record.id)
    for number, record in enumerate(ordered):
        data = observations.to_json(results[(source.name, record.id)])
        lines.append(f"    {json.dumps(record.id)}: {{")
        lines.append(f'      "sha256": {json.dumps(record.sha256)},')
        for key in _RECORDED:
            lines.append(f'      "{key}": {json.dumps(data[key])},')
        if data["positions"]:
            lines.append('      "positions": [')
            rows = [f"        {json.dumps(row)}" for row in data["positions"]]
            lines.append(",\n".join(rows))
            lines.append("      ]")
        else:
            lines.append('      "positions": []')
        lines.append("    }" + ("," if number + 1 < len(ordered) else ""))
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def load(source: str) -> Golden:
    path = golden_path(source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HarnessError(
            f"no golden observations for '{source}' ({path.name}); "
            f"run: python {paths.HARNESS_REL}/harness.py golden --write") from None
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path.name}: invalid JSON: {error}") from None
    if data.get("schema") != SCHEMA or data.get("source") != source:
        raise HarnessError(f"{path.name}: not a schema-{SCHEMA} golden file for '{source}'")
    signature = data.get("signature", {})
    golden = Golden(source=source, signature=signature, sha256={}, observations={})
    for record_id, entry in data.get("records", {}).items():
        try:
            golden.sha256[record_id] = entry.get("sha256", "")
            golden.observations[record_id] = observations.from_json(source, record_id, _cap(signature), entry)
        except (AttributeError, KeyError, IndexError, TypeError) as error:
            raise HarnessError(f"{path.name}: the entry for {record_id} is malformed ({error!r})") from None
    return golden


def lookup():
    """(source, record id) -> the golden observation, or None. Files are read on first use."""
    loaded: dict[str, Golden | None] = {}

    def recorded(source: str, record_id: str) -> Observation | None:
        if source not in loaded:
            try:
                loaded[source] = load(source)
            except HarnessError:
                loaded[source] = None  # reported by verify_static
        golden = loaded[source]
        return None if golden is None else golden.observations.get(record_id)

    return recorded


def verify_static(sources: list[Source], options: RunOptions, records: list[Record]) -> list[Finding]:
    """Do the committed golden files still describe the current pins, drivers and corpus?"""
    findings: list[Finding] = []
    expected = {record.id: record.sha256 for record in records}
    for source in sources:
        try:
            golden = load(source.name)
        except HarnessError as error:
            findings.append(Finding("golden-missing", str(error)))
            continue
        name = golden_path(source.name).name
        current = source.signature(options)
        if golden.signature != current:
            changed = sorted(key for key in golden.signature.keys() | current.keys() if golden.signature.get(key) != current.get(key))
            findings.append(Finding(
                "golden-stale",
                f"{name}: recorded under a different {', '.join(changed)} - "
                "regenerate with `golden --write` and review the diff"))
        for record_id in sorted(expected.keys() - golden.sha256.keys()):
            findings.append(Finding("golden-incomplete", f"{name}: no entry for corpus record {record_id}"))
        for record_id in sorted(golden.sha256.keys() - expected.keys()):
            findings.append(Finding("golden-orphan", f"{name}: entry for {record_id}, which is not in the corpus"))
        for record_id in sorted(expected.keys() & golden.sha256.keys()):
            if golden.sha256[record_id] != expected[record_id]:
                findings.append(Finding("golden-stale", f"{name}: {record_id} was recorded for different record content"))
            for problem in observations.validate(golden.observations[record_id]):
                findings.append(Finding("golden-invalid", f"{name}: {record_id}: {problem}"))
    return findings


def check_run(source: str, records: list[Record], results: Results, min_coverage: float) -> tuple[list[Finding], tuple[int, int]]:
    """Does a fresh run say what the golden file says?"""
    findings: list[Finding] = []
    golden = load(source)
    reproduced = total = 0
    for record in records:
        recorded = golden.observations.get(record.id)
        fresh = results.get((source, record.id))
        if recorded is None or fresh is None:
            continue
        recorded.source = f"{source}@golden"
        for disagreement in observations.compare(recorded, fresh):
            findings.append(Finding("golden-mismatch", disagreement.describe()))
        changed = observations.status_changed(recorded, fresh)
        if changed:
            findings.append(Finding("golden-status", f"{source} on {record.id}: {changed}"))
        # A checkmate or stalemate verdict is rare and it is what the gate is
        # for: one that the fresh run reached and could not repeat is a finding,
        # not a point of coverage lost.
        for position in recorded.positions:
            if position.terminal in ("checkmate", "stalemate") and position.index < len(fresh.positions) \
                    and fresh.positions[position.index].terminal is None:
                findings.append(Finding(
                    "golden-lost-verdict",
                    f"{source} on {record.id}: position {position.index} was recorded as {position.terminal}; "
                    "the fresh run reached it and gave no verdict"))
        got, want = observations.coverage(recorded, fresh)
        reproduced, total = reproduced + got, total + want
    if total and reproduced / total < min_coverage:
        findings.append(Finding(
            "golden-coverage",
            f"{source}: reproduced only {reproduced}/{total} recorded values "
            f"({reproduced / total:.1%} < {min_coverage:.0%}) - too many time-outs or failures to call this run a check"))
    return findings, (reproduced, total)


def _pairs() -> list[tuple[str, str]]:
    return [(first, second) for index, first in enumerate(ORACLE_SOURCES) for second in ORACLE_SOURCES[index + 1:]]


def disagreements(records: list[Record]) -> list[Disagreement]:
    """Every contradiction between two oracles' golden answers, in a stable order."""
    loaded = {source: load(source) for source in ORACLE_SOURCES}
    found: list[Disagreement] = []
    for record in sorted(records, key=lambda item: item.id):
        for first, second in _pairs():
            a = loaded[first].observations.get(record.id)
            b = loaded[second].observations.get(record.id)
            if a is not None and b is not None:
                # How a record is replayed is a property of the implementation,
                # so it is compared once per pair of implementations.
                replay = first in PRIMARY_SOURCES and second in PRIMARY_SOURCES
                found += observations.compare(a, b, replay=replay)
    return found


def _agreement(a: Observation, b: Observation, name: str) -> tuple[int, int, int, int]:
    """(both answered, equal, compatible but not equal, contradictory) for one observable."""
    both = equal = loose = contradictory = 0
    clashes = {item.index for item in observations.compare(a, b, replay=False) if item.kind == name}
    for pa, pb in zip(a.positions, b.positions, strict=False):
        va, vb = getattr(pa, name), getattr(pb, name)
        if va is None or vb is None:
            continue
        both += 1
        if pa.index in clashes:
            contradictory += 1
        elif name == "legal_actions" and a.cap is not None and b.cap is not None and va >= a.cap and vb >= b.cap:
            loose += 1  # both stopped counting at the cap: only "at least this many" was compared
        elif va == vb and observations.decisive(name, va):
            equal += 1
        else:
            loose += 1  # "no_action" beside a checkmate or stalemate, or a capped count beside a larger one
    return both, equal, loose, contradictory


def render_disagreements(records: list[Record]) -> str:
    loaded = {source: load(source) for source in ORACLE_SOURCES}
    found = disagreements(records)
    lines = [
        "# Oracle disagreements",
        "",
        "<!-- Generated by `python oracles/harness.py golden --write` from oracles/golden/*.json. Do not edit. -->",
        "",
        "Where the pinned oracles contradict each other on the pinned corpus. At each of",
        "these points at least one oracle is wrong, or reads the rules differently; **none of",
        "them has been adjudicated**. They are recorded so that they are known before the",
        "engine meets them: when the engine disagrees with an oracle here, the rulebook",
        "decides, and the oracle that turns out to be wrong gets an entry in",
        "`known-divergences.toml` with its citation.",
        "",
        "An oracle that cannot answer a question (or ran out of time) says nothing, and a",
        "silence never counts as a disagreement. \"Soft\" legality entries cannot tell a rules",
        "disagreement from notation the rejecting oracle does not read.",
        "",
        "## What each oracle could say",
        "",
        "| Source | ok | rejected | unsupported | error | timeout | positions | known values |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for source in ORACLE_SOURCES:
        items = [loaded[source].observations[record.id] for record in records if record.id in loaded[source].observations]
        counts = {status: sum(item.status == status for item in items) for status in observations.STATUSES}
        lines.append(
            f"| `{source}` | {counts['ok']} | {counts['rejected']} | {counts['unsupported']} | {counts['error']} "
            f"| {counts['timeout']} | {sum(len(item.positions) for item in items)} | {sum(item.known() for item in items)} |"
        )

    lines += [
        "",
        "## Which rulesets each oracle replays",
        "",
        "Records replayed to their end, or to an action the oracle refuses, out of the records",
        "the corpus holds for that ruleset; then the positions at which each observable has an",
        "answer from at least one oracle. \"(custom positions)\" gathers the records that set up",
        "their own position, with 5DFEN boards or a custom board: they exercise the rules, but",
        "they are evidence about no named variant.",
        "",
        "| Ruleset | Records | " + " | ".join(f"`{source}`" for source in ORACLE_SOURCES)
        + " | Positions | check | counts | mate |",
        "|---|---|" + "---|" * len(ORACLE_SOURCES) + "---|---|---|---|",
    ]
    rulesets: dict[str, list[Record]] = {}
    for record in records:
        rulesets.setdefault(record.ruleset, []).append(record)
    for ruleset, items in sorted(rulesets.items()):
        cells = []
        for source in ORACLE_SOURCES:
            reports = [loaded[source].observations.get(item.id) for item in items]
            cells.append(str(sum(report is not None and report.status in ("ok", "rejected") for report in reports)))
        positions = 0
        answered = dict.fromkeys(OBSERVABLES, 0)
        for item in items:
            reports = [report for source in ORACLE_SOURCES if (report := loaded[source].observations.get(item.id)) is not None]
            reached = max((len(report.positions) for report in reports), default=0)
            positions += reached
            for index in range(reached):
                for name in OBSERVABLES:
                    answered[name] += any(
                        index < len(report.positions) and observations.decisive(name, getattr(report.positions[index], name))
                        for report in reports)
        lines.append(f"| {ruleset} | {len(items)} | " + " | ".join(cells)
                     + f" | {positions} | " + " | ".join(str(answered[name]) for name in OBSERVABLES) + " |")

    lines += [
        "",
        "## How much two oracles were compared",
        "",
        "Positions at which both sources of a pair answered the same question. \"Loosely\" means",
        "the two answers are compatible but do not pin each other down: both counts stopped at",
        "the cap (only \"at least that many\" was compared), or one source says only that no",
        "action exists where the other says which kind of end it is. Pairs of searches inside",
        "one implementation are included: they are the check on the search, not on the rules.",
        "",
        "| Between | Observable | Both answered | Equal | Loosely | Contradictory |",
        "|---|---|---|---|---|---|",
    ]
    for first, second in _pairs():
        for name in OBSERVABLES:
            totals = [0, 0, 0, 0]
            for record in records:
                a = loaded[first].observations.get(record.id)
                b = loaded[second].observations.get(record.id)
                if a is not None and b is not None:
                    totals = [total + part for total, part in zip(totals, _agreement(a, b, name), strict=True)]
            if totals[0]:
                same = " (one implementation)" if oracle_of(first) == oracle_of(second) else ""
                lines.append(f"| `{first}` vs `{second}`{same} | {name} | " + " | ".join(str(total) for total in totals) + " |")

    lines += ["", "## Summary", ""]
    if not found:
        lines.append("No disagreements.")
    else:
        tally: dict[tuple[str, str, str], int] = {}
        for item in found:
            kind = item.kind + (" (soft)" if item.soft else "")
            tally[(item.a, item.b, kind)] = tally.get((item.a, item.b, kind), 0) + 1
        lines += ["| Between | Kind | Count |", "|---|---|---|"]
        for (a, b, kind), count in sorted(tally.items()):
            lines.append(f"| `{a}` vs `{b}` | {kind} | {count} |")
        lines += ["", "## Detail", ""]
        current = None
        for item in found:
            if item.record != current:
                if current is not None:
                    lines.append("")
                current = item.record
                lines.append(f"### `{item.record}`")
                lines.append("")
            where = f"action {item.index}" if item.kind in ("legality", "length") else f"position {item.index}"
            soft = " *(soft)*" if item.soft else ""
            lines.append(f"- {where}, {item.kind}{soft}: `{item.a}` = {item.a_value}; `{item.b}` = {item.b_value}")
    return "\n".join(lines) + "\n"
