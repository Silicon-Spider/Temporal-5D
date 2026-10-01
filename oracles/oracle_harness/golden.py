"""Golden observations: what each pinned oracle says about each pinned record.

Pinned oracles over a pinned corpus have pinned answers. Committing those answers
does three jobs:

  * CI can tell that an oracle built on another machine, compiler or OS still
    says what it said when it was reviewed;
  * a pin bump shows up as a reviewable diff of verdicts, not just of a hash;
  * the places where the oracles contradict each other are on record before the
    engine exists (DISAGREEMENTS.md is rendered from these files).

A golden value is a statement by that oracle, not a statement of truth.

The files hold verdicts, and hashes of what each oracle set up - never an
oracle's free-text messages, which quote the record they are about, and no
record is redistributed from here. A row is one position: the columns of
observations.ROW_FIELDS, the answers and then the fingerprint, placement hash,
and board and piece counts of the oracle's state event (schema 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import observations, paths
from .adapters import ORACLE_SOURCES, PRIMARY_SOURCES, RunOptions, Source, oracle_of
from .corpus import Record
from .observations import OBSERVABLES, PARSE_FIELD, PARSE_KINDS, Disagreement, Observation
from .runner import Results
from .util import Finding, HarnessError

SCHEMA = 2
_RECORDED = ("status", "reason", "actions_accepted", "last_action")


class StaleSchema(HarnessError):
    """A golden file written in a schema this harness no longer reads."""


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
    if not isinstance(data, dict) or data.get("source") != source:
        raise HarnessError(f"{path.name}: not a golden file for '{source}'")
    if data.get("schema") != SCHEMA:
        raise StaleSchema(
            f"{path.name}: written in golden schema {data.get('schema')}, and this harness reads schema {SCHEMA} "
            "(a position also records what the oracle set up there) - re-record it: "
            f"python {paths.HARNESS_REL}/harness.py golden --write")
    signature = data.get("signature", {})
    golden = Golden(source=source, signature=signature, sha256={}, observations={})
    for record_id, entry in data.get("records", {}).items():
        try:
            golden.sha256[record_id] = entry.get("sha256", "")
            golden.observations[record_id] = observations.from_json(source, record_id, _cap(signature), entry)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
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
        except StaleSchema as error:
            findings.append(Finding("golden-stale", str(error)))
            continue
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


def _shared(a: Observation, b: Observation) -> list[tuple[observations.Position, observations.Position]]:
    """The positions two reports both reached, up to where they parted (a placement divergence)."""
    return list(zip(a.positions, b.positions, strict=False))[:observations.first_placement_divergence(a, b)]


def _agreement(a: Observation, b: Observation, name: str) -> tuple[int, int, int, int, int, int, int]:
    """(both answered, equal, compatible but not equal, contradictory, same state, same boards, unverified).

    For one observable. Only positions before the two parted are counted: from a
    placement divergence on they answer about different positions, and are not
    compared. Each position counted is one of three: the same state - both said
    what they set up, unmoved markers included, and it is the same; the same
    boards - the same boards and pieces, but markers that differ or that one of
    the two does not give; or unverified - one said nothing of what it set up.
    """
    both = equal = loose = contradictory = exact = boards = unverified = 0
    clashes = {item.index for item in observations.compare(a, b, replay=False) if item.kind == name}
    for pa, pb in _shared(a, b):
        va, vb = getattr(pa, name), getattr(pb, name)
        if va is None or vb is None:
            continue
        both += 1
        if observations.same_state(pa, pb):
            exact += 1
        elif observations.aligned(pa, pb):
            boards += 1
        else:
            unverified += 1
        if pa.index in clashes:
            contradictory += 1
        elif name == "legal_actions" and a.cap is not None and b.cap is not None and va >= a.cap and vb >= b.cap:
            loose += 1  # both stopped counting at the cap: only "at least this many" was compared
        elif va == vb and observations.decisive(name, va):
            equal += 1
        else:
            loose += 1  # "no_action" beside a checkmate or stalemate, or a capped count beside a larger one
    return both, equal, loose, contradictory, exact, boards, unverified


def _parse_agreement(a: Observation, b: Observation) -> tuple[int, int, int, int]:
    """(both described, same boards and pieces, both marked, same markers) for one pair of reports.

    Counted up to and including the position where the two parted, if they
    did: after it they describe different games.
    """
    parted = observations.first_placement_divergence(a, b)
    stop = None if parted is None else parted + 1
    described = same = marked = same_marks = 0
    for pa, pb in list(zip(a.positions, b.positions, strict=False))[:stop]:
        match = observations.aligned(pa, pb)
        if match is None:
            continue
        described += 1
        same += match
        if match and None not in (pa.fingerprint, pb.fingerprint):
            marked += 1
            same_marks += pa.fingerprint == pb.fingerprint
    return described, same, marked, same_marks


def _state_cell(report: Observation, item: Disagreement) -> str:
    position = report.positions[item.index]
    counts = observations.counts_text(position)
    return f"`{getattr(position, PARSE_FIELD[item.kind])}`" + (f" ({counts})" if counts else "")


def _parse_section(records: list[Record], loaded: dict[str, Golden], found: list[Disagreement]) -> list[str]:
    lines = [
        "## Parse divergences",
        "",
        "Each source says what it set up at every position, and two sources' positions are",
        "compared there before anything else (decision H30). A **placement** divergence is the",
        "first position at which the two hold different boards - different pieces, or the same",
        "pieces on boards at other coordinates: from there on they answer about different",
        "positions, so nothing either says there or after it is compared, and no rules",
        "disagreement from there on is listed below. It is settled against the notation",
        "specification. A **marker-only** divergence is a run of positions at which they hold",
        "the same boards and pieces but mark different ones as unmoved, listed where it begins;",
        "it suppresses nothing, and it is settled against the notation specification's \"5DFEN",
        "Strictness\" or the rulebook (what having moved means). The hashes are the notation",
        "specification's full-state hash - without the unmoved markers for a placement",
        "divergence, with them for a marker-only one. Two searches of one implementation replay",
        "the record with the same code, so a parse divergence between them is a bug in the",
        "harness or its driver, not a reading.",
        "",
    ]
    parse = [item for item in found if item.kind in PARSE_KINDS]
    if parse:
        lines += [
            "| Record | Between | Position | Divergence | First source | Second source | Not compared |",
            "|---|---|---|---|---|---|---|",
        ]
        for item in parse:
            a = loaded[item.a].observations[item.record]
            b = loaded[item.b].observations[item.record]
            if item.kind == "placement":
                kind, skipped = "placement", str(min(len(a.positions), len(b.positions)) - item.index)
            else:
                kind, skipped = "marker-only", "-"
            same = " (one implementation)" if oracle_of(item.a) == oracle_of(item.b) else ""
            lines.append(f"| `{item.record}` | `{item.a}` vs `{item.b}`{same} | {item.index} | {kind} "
                         f"| {_state_cell(a, item)} | {_state_cell(b, item)} | {skipped} |")
        lines += ["", "\"Not compared\" counts the positions both sources reached from the divergence on.", ""]
    totals: dict[tuple[str, str], list[int]] = {}
    for first, second in _pairs():
        for record in records:
            a = loaded[first].observations.get(record.id)
            b = loaded[second].observations.get(record.id)
            if a is not None and b is not None:
                counts = _parse_agreement(a, b)
                totals[(first, second)] = [x + y for x, y in zip(totals.get((first, second), [0] * 4), counts, strict=True)]
    rows = [(pair, counts) for pair, counts in totals.items() if counts[0]]
    if not rows:
        lines += ["No two sources said what they set up at the same position: nothing was compared.", ""]
        return lines
    if not parse:
        lines += ["None.", ""]
    lines += [
        "Positions at which both sources of a pair said what they set up, and how many of",
        "those held the same boards and pieces; then the positions at which both also marked",
        "the unmoved pieces (a source that does not track them says only which boards it holds",
        "and where the pieces are), and how many of those were marked the same way.",
        "",
        "| Between | Both described | Same boards | Both marked | Same markers |",
        "|---|---|---|---|---|",
    ]
    for (first, second), counts in rows:
        same = " (one implementation)" if oracle_of(first) == oracle_of(second) else ""
        lines.append(f"| `{first}` vs `{second}`{same} | " + " | ".join(str(count) for count in counts) + " |")
    lines.append("")
    return lines


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
        "disagreement from notation the rejecting oracle does not read. Two oracles that read",
        "the record's moves as different positions are a parse divergence, listed first: what",
        "they say about the rules from there on is about two different positions, and is not",
        "compared.",
        "",
        "## What each oracle could say",
        "",
        "\"Known values\" counts answers; \"described\" counts the positions at which the oracle",
        "said what it had set up, and \"with markers\" those at which it also said which pieces",
        "had not moved.",
        "",
        "| Source | ok | rejected | unsupported | error | timeout | positions | known values | described | with markers |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for source in ORACLE_SOURCES:
        items = [loaded[source].observations[record.id] for record in records if record.id in loaded[source].observations]
        counts = {status: sum(item.status == status for item in items) for status in observations.STATUSES}
        described = sum(position.placement is not None for item in items for position in item.positions)
        marked = sum(position.fingerprint is not None for item in items for position in item.positions)
        lines.append(
            f"| `{source}` | {counts['ok']} | {counts['rejected']} | {counts['unsupported']} | {counts['error']} "
            f"| {counts['timeout']} | {sum(len(item.positions) for item in items)} | {sum(item.known() for item in items)} "
            f"| {described} | {marked} |"
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
        "\"Same state\" counts the positions known to be the same position: both sources said",
        "what they had set up there, unmoved markers included, and it was the same. \"Same",
        "boards\" counts those at which they held the same boards and pieces, but marked",
        "different pieces as unmoved or one of them does not give the markers; \"unverified\"",
        "those at which one of the two said nothing of what it set up, so that the two are",
        "aligned only by counting actions. Positions from a placement divergence on are not",
        "compared, and not counted.",
        "",
        "| Between | Observable | Both answered | Equal | Loosely | Contradictory | Same state | Same boards "
        "| Unverified |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for first, second in _pairs():
        for name in OBSERVABLES:
            totals = [0] * 7
            for record in records:
                a = loaded[first].observations.get(record.id)
                b = loaded[second].observations.get(record.id)
                if a is not None and b is not None:
                    totals = [total + part for total, part in zip(totals, _agreement(a, b, name), strict=True)]
            if totals[0]:
                same = " (one implementation)" if oracle_of(first) == oracle_of(second) else ""
                lines.append(f"| `{first}` vs `{second}`{same} | {name} | " + " | ".join(str(total) for total in totals) + " |")

    lines += ["", *_parse_section(records, loaded, found)]
    lines += ["## Summary", ""]
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
    rules = [item for item in found if item.kind not in PARSE_KINDS]
    if rules:
        lines += ["", "## Detail", "", "The rules disagreements; the parse divergences are listed above.", ""]
        current = None
        for item in rules:
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
