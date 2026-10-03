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
record is redistributed from here. Each record's entry names the text its
answers are about by its SHA-256: the stored record's, or - for a record read as
converted from the official client's export form (export_form.py) - the
converted text's, so that answers about one text are not taken for answers
about another. A row is one position: the columns of
observations.ROW_FIELDS, the answers and then the fingerprint, placement hash,
and board and piece counts of the oracle's state event.

Schema 3 (decision H32, stable answers only) adds what was not answered, and why:

  * the signature records the budget the answers were recorded under (the
    run's timeout_s; for Chess in 5D the mate search's budget and heap too), so
    answers recorded under another budget are stale;
  * "horizon": {"position": K, "budget": "timeout_s=300"} - the run was cut at
    its deadline while it was searching position K (whose state is recorded,
    its answers are not): from there on everything is unknown at that budget;
  * "unknown": {"<k>": {"<field>": "<limit>"}} - a null inside the answered
    positions that a limit caused (Chess in 5D's mate search: its time budget or
    its heap). A null with no limit stays a bare null: the source never answers
    that question;
  * "unstable": {"<k>": {"<field>": [v1, v2, ...]}} - a value that differed
    between independent recordings (golden --write --repeat N). The position's
    value is null, so nothing is compared with it.

Beside each golden file is its cost file (<source>.cost.json): for each record,
when each position's answers arrived, in seconds from the start of the run -
the evidence for which answers each tier can require (stability.py). Costs vary
from one recording to the next, so they are kept out of the golden files: a diff
of a golden file is a diff of verdicts. calibration.json holds the reference time
of the calibration probe on the recording machine. MANIFEST commits to the hash
of every one of these files (integrity.py, H33), and every load verifies it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from . import integrity, observations, paths
from .adapters import ORACLE_SOURCES, PRIMARY_SOURCES, RunOptions, Source, oracle_of
from .corpus import Record
from .integrity import MIN_RECORDINGS, RERECORD
from .observations import ANSWERS, OBSERVABLES, PARSE_FIELD, PARSE_KINDS, ROW_FIELDS, Disagreement, Observation
from .util import Finding, HarnessError, sha256_text

SCHEMA = 3
COST_SCHEMA = 1
CALIBRATION_SCHEMA = 1
CALIBRATION_FILE = "calibration.json"
_RECORDED = ("status", "reason", "actions_accepted", "last_action")
# What schema 3 added, for the message that asks for a re-recording.
_SCHEMA_NOTE = "budgets, horizons, unknown and unstable values, cost files, MANIFEST"


class StaleSchema(HarnessError):
    """A golden file written in a schema this harness no longer reads."""


@dataclass(frozen=True)
class Horizon:
    """Where a recording run was cut at its deadline."""

    position: int | None  # the position it was searching (its state is recorded, its answers are not); None: none reached
    budget: str  # the limit that cut it: "timeout_s=300"

    def describe(self) -> str:
        where = "before its first position" if self.position is None else f"at position {self.position}"
        return f"cut at {self.budget} {where}"


@dataclass
class Golden:
    source: str
    signature: dict
    sha256: dict[str, str]  # record id -> hash of the text these answers are about (Record.fed_sha256)
    observations: dict[str, Observation]
    recordings: int = 1
    horizon: dict[str, Horizon] = field(default_factory=dict)  # record id -> where its run was cut
    unknown: dict[str, dict[int, dict[str, str]]] = field(default_factory=dict)  # record -> position -> field -> limit
    unstable: dict[str, dict[int, dict[str, list]]] = field(default_factory=dict)  # record -> position -> field -> values
    digest: str = ""  # SHA-256 of the bytes this was read from


@dataclass(frozen=True)
class RecordCost:
    """When one source's answers about one record arrived: seconds from the start of its run, rounded up."""

    positions: tuple[float | None, ...]  # per position, when what is recorded about it had arrived
    end: float | None  # when the outcome had (None: the run was cut, and has none)
    mate: tuple[float | None, ...] = ()  # Chess in 5D: per position, how long its check test and mate search took

    @property
    def cost(self) -> float:
        """The cost of the slowest answer: the record's cost."""
        known = [value for value in (*self.positions, self.end) if value is not None]
        return max(known, default=0.0)


@dataclass
class Costs:
    source: str
    signature: dict
    recordings: int
    records: dict[str, RecordCost]
    digest: str = ""


@dataclass(frozen=True)
class Calibration:
    """The calibration probe as measured on the machine the golden files were recorded on."""

    source: str
    record: str
    positions: int
    signature: dict
    seconds: float
    platform: str
    cpu_count: int | None
    digest: str = ""


@dataclass
class Recorded:
    """One source's answers about one record, ready to write: merged over its recordings and classified (H32)."""

    observation: Observation
    cost: RecordCost
    horizon: Horizon | None = None
    unknown: dict[int, dict[str, str]] = field(default_factory=dict)
    unstable: dict[int, dict[str, list]] = field(default_factory=dict)


def golden_path(source: str) -> Path:
    return paths.GOLDEN_DIR / f"{source.replace(':', '-')}.json"


def cost_path(source: str) -> Path:
    return paths.GOLDEN_DIR / f"{source.replace(':', '-')}.cost.json"


def calibration_path() -> Path:
    return paths.GOLDEN_DIR / CALIBRATION_FILE


def signature(source: Source, options: RunOptions) -> dict:
    """What determines a source's answers: its own signature (pin, search, cap, code) and its budget (H32)."""
    return {**source.signature(options), "budget": source.budget(options)}


def round_up(seconds: float | None) -> float | None:
    """A cost in seconds, rounded up to two significant figures (0.01234 -> 0.013, 123.4 -> 130.0)."""
    if seconds is None:
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return 0.0
    value = Decimal(repr(float(seconds)))
    return float(value.quantize(Decimal(1).scaleb(value.adjusted() - 1), rounding=ROUND_CEILING))


def _cap(signature: dict) -> int | None:
    return signature.get("cap")


def row_value(name: str, value: object) -> object:
    """A value as a golden row holds it: the side as "w" or "b", everything else as it is."""
    return value[0] if name == "side" and isinstance(value, str) else value


def _from_row_value(name: str, value: object) -> object:
    return {"w": "white", "b": "black"}.get(value, value) if name == "side" else value


def _by_index(mapping: dict[int, dict[str, object]]) -> str:
    """{position: {field: x}} as one line of JSON, positions in order and fields in row order."""
    ordered = {str(index): {name: mapping[index][name] for name in ROW_FIELDS if name in mapping[index]}
               for index in sorted(mapping)}
    return json.dumps(ordered)


# --- writing --------------------------------------------------------------------


def dump(source: str, signature: dict, records: list[Record], recorded: dict[str, Recorded], *, recordings: int) -> str:
    """Deterministic text, one position per line, so a changed verdict is a one-line diff."""
    lines = ["{", f'  "schema": {SCHEMA},', f'  "source": {json.dumps(source)},',
             f'  "signature": {json.dumps(signature, sort_keys=True)},', f'  "recordings": {recordings},',
             '  "records": {']
    ordered = sorted(records, key=lambda record: record.id)
    for number, record in enumerate(ordered):
        item = recorded[record.id]
        data = observations.to_json(item.observation)
        lines.append(f"    {json.dumps(record.id)}: {{")
        lines.append(f'      "sha256": {json.dumps(record.fed_sha256)},')
        for key in _RECORDED:
            lines.append(f'      "{key}": {json.dumps(data[key])},')
        if item.horizon is not None:
            lines.append(f'      "horizon": {json.dumps({"position": item.horizon.position, "budget": item.horizon.budget})},')
        if item.unknown:
            lines.append(f'      "unknown": {_by_index(item.unknown)},')
        if item.unstable:
            lines.append(f'      "unstable": {_by_index(item.unstable)},')
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


def dump_costs(source: str, signature: dict, records: list[Record], recorded: dict[str, Recorded], *,
               recordings: int) -> str:
    """One record per line: its cost, when its outcome arrived, and when each position's answers did."""
    lines = ["{", f'  "schema": {COST_SCHEMA},', f'  "source": {json.dumps(source)},',
             f'  "signature": {json.dumps(signature, sort_keys=True)},', f'  "recordings": {recordings},',
             '  "unit": "seconds from the start of the run until it had arrived, rounded up to 2 significant figures; '
             'the most over the recordings",',
             '  "records": {']
    ordered = sorted(records, key=lambda record: record.id)
    for number, record in enumerate(ordered):
        cost = recorded[record.id].cost
        entry: dict[str, object] = {"cost": cost.cost, "end": cost.end, "positions": list(cost.positions)}
        if cost.mate:
            entry["mate"] = list(cost.mate)
        lines.append(f"    {json.dumps(record.id)}: {json.dumps(entry)}" + ("," if number + 1 < len(ordered) else ""))
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def dump_calibration(calibration: Calibration) -> str:
    body = {"schema": CALIBRATION_SCHEMA, "source": calibration.source, "record": calibration.record,
            "positions": calibration.positions, "signature": calibration.signature, "seconds": calibration.seconds,
            "platform": calibration.platform, "cpu_count": calibration.cpu_count}
    return json.dumps(body, indent=1, sort_keys=True) + "\n"


# --- reading --------------------------------------------------------------------


def peek_schema(path: Path) -> object:
    """The schema a file says it is in, without trusting anything else in it; None when it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("schema") if isinstance(data, dict) else None


def _read(path: Path, manifest: dict[str, str] | None, what: str) -> tuple[dict, str]:
    try:
        raw = integrity.verified_bytes(path, manifest)
    except FileNotFoundError:
        raise HarnessError(f"no {what} ({path.name}); record it: {RERECORD}") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"{path.name}: invalid JSON: {error}") from None
    if not isinstance(data, dict):
        raise HarnessError(f"{path.name}: not a JSON object")
    return data, sha256_text(raw)


def _int_keys(value: object, where: str) -> dict[int, dict]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} is not an object")
    found = {}
    for key, fields in value.items():
        if not (isinstance(key, str) and key.isdigit()) or not isinstance(fields, dict) or not fields:
            raise ValueError(f"{where}: {key!r} is not a position with fields")
        if any(name not in ROW_FIELDS for name in fields):
            raise ValueError(f"{where} at position {key}: a field that is not a column ({', '.join(fields)})")
        found[int(key)] = fields
    return found


def load(source: str, manifest: dict[str, str] | None = None) -> Golden:
    """A golden file, read only once its bytes are shown to be the ones MANIFEST commits to (H33)."""
    path = golden_path(source)
    manifest = integrity.read_manifest() if manifest is None else manifest
    schema = peek_schema(path)
    if schema is not None and schema != SCHEMA:
        raise StaleSchema(f"{path.name}: golden schema {schema}; this harness reads schema {SCHEMA} ({_SCHEMA_NOTE}) - "
                          f"re-record: {RERECORD}")
    data, digest = _read(path, manifest, f"golden observations for '{source}'")
    if data.get("source") != source:
        raise HarnessError(f"{path.name}: not a golden file for '{source}'")
    if data.get("schema") != SCHEMA:
        raise StaleSchema(f"{path.name}: written in golden schema {data.get('schema')}, not {SCHEMA}")
    signature = data.get("signature", {})
    recordings = data.get("recordings", 1)
    golden = Golden(source=source, signature=signature, sha256={}, observations={},
                    recordings=recordings if isinstance(recordings, int) and not isinstance(recordings, bool) else 0,
                    digest=digest)
    for record_id, entry in data.get("records", {}).items():
        try:
            golden.sha256[record_id] = entry.get("sha256", "")
            golden.observations[record_id] = observations.from_json(source, record_id, _cap(signature), entry)
            if "horizon" in entry:
                horizon = entry["horizon"]
                position, budget = horizon["position"], horizon["budget"]
                if not (position is None or (isinstance(position, int) and not isinstance(position, bool))) \
                        or not isinstance(budget, str):
                    raise ValueError("horizon needs a position (a whole number, or null) and a budget")
                golden.horizon[record_id] = Horizon(position, budget)
            if "unknown" in entry:
                golden.unknown[record_id] = _int_keys(entry["unknown"], "unknown")
            if "unstable" in entry:
                golden.unstable[record_id] = {
                    index: {name: [_from_row_value(name, value) for value in values] for name, values in fields.items()}
                    for index, fields in _int_keys(entry["unstable"], "unstable").items()}
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise HarnessError(f"{path.name}: the entry for {record_id} is malformed ({error!r})") from None
    return golden


def load_costs(source: str, manifest: dict[str, str] | None = None) -> Costs:
    path = cost_path(source)
    manifest = integrity.read_manifest() if manifest is None else manifest
    data, digest = _read(path, manifest, f"cost file for '{source}'")
    if data.get("source") != source:
        raise HarnessError(f"{path.name}: not a cost file for '{source}'")
    if data.get("schema") != COST_SCHEMA:
        raise StaleSchema(f"{path.name}: written in cost schema {data.get('schema')}, and this harness reads "
                          f"{COST_SCHEMA} - re-record it: {RERECORD}")

    def seconds(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{value!r} is not a number of seconds")
        return float(value)

    records: dict[str, RecordCost] = {}
    for record_id, entry in data.get("records", {}).items():
        try:
            records[record_id] = RecordCost(positions=tuple(seconds(value) for value in entry["positions"]),
                                            end=seconds(entry["end"]),
                                            mate=tuple(seconds(value) for value in entry.get("mate", ())))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise HarnessError(f"{path.name}: the entry for {record_id} is malformed ({error!r})") from None
    recordings = data.get("recordings", 1)
    return Costs(source=source, signature=data.get("signature", {}), records=records, digest=digest,
                 recordings=recordings if isinstance(recordings, int) and not isinstance(recordings, bool) else 0)


def load_calibration(manifest: dict[str, str] | None = None) -> Calibration:
    path = calibration_path()
    manifest = integrity.read_manifest() if manifest is None else manifest
    data, digest = _read(path, manifest, "calibration reference")
    if data.get("schema") != CALIBRATION_SCHEMA:
        raise StaleSchema(f"{path.name}: written in schema {data.get('schema')}, not {CALIBRATION_SCHEMA}")
    try:
        seconds = data["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError("seconds must be a positive number")
        return Calibration(source=str(data["source"]), record=str(data["record"]), positions=int(data["positions"]),
                           signature=dict(data["signature"]), seconds=float(seconds), platform=str(data["platform"]),
                           cpu_count=data.get("cpu_count"), digest=digest)
    except (KeyError, TypeError, ValueError) as error:
        raise HarnessError(f"{path.name}: malformed ({error!r})") from None


def read_previous(source: str) -> dict[str, Observation] | None:
    """What the golden file on disk records ``source`` saying, read leniently - or None where it cannot be read.

    Any schema whose rows this harness reads, and no MANIFEST check: this is
    only for `golden --write` to say what a new recording no longer holds that
    the file it replaces did (stability.losses), never for a comparison.
    """
    try:
        data = json.loads(golden_path(source).read_text(encoding="utf-8"))
        cap = _cap(data.get("signature", {}))
        return {record_id: observations.from_json(source, record_id, cap, entry)
                for record_id, entry in data["records"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


# --- static checks --------------------------------------------------------------


def _problems(golden: Golden, record_id: str) -> list[str]:
    """What is wrong with one record's classification: a horizon, unknown or unstable entry that does not fit its row."""
    found = []
    observation = golden.observations[record_id]
    horizon = golden.horizon.get(record_id)
    last = len(observation.positions) - 1 if observation.positions else None
    if (observation.status == "timeout") != (horizon is not None):
        found.append("a run cut at its deadline (status timeout) and a horizon go together")
    elif horizon is not None:
        if horizon.position != last:
            found.append(f"the horizon is at position {horizon.position}, but the run's last position is {last}")
        if not horizon.budget.strip():
            found.append("the horizon does not say which budget cut the run")
    for kind, table in (("unknown", golden.unknown.get(record_id, {})), ("unstable", golden.unstable.get(record_id, {}))):
        for index, fields in table.items():
            if index >= len(observation.positions):
                found.append(f"an {kind} entry for position {index}, which the run did not reach")
                continue
            for name, detail in fields.items():
                if getattr(observation.positions[index], name) is not None:
                    found.append(f"position {index}: {name} is listed as {kind} and has a value")
                if kind == "unknown" and (not isinstance(detail, str) or not detail.strip()):
                    found.append(f"position {index}: {name} is unknown without the limit that caused it")
                if kind == "unknown" and name not in ANSWERS:
                    found.append(f"position {index}: {name} is not an answer a limit can leave unknown")
                if kind == "unstable" and (not isinstance(detail, list) or len(detail) < 2):
                    found.append(f"position {index}: {name} is unstable without the values that differed")
    return found


def verify_static(sources: list[Source], options_for, records: list[Record], snapshot) -> list[Finding]:
    """Do the committed golden and cost files still describe the current pins, drivers, budgets and corpus?

    ``options_for(source)`` gives the options each source is recorded under;
    ``snapshot`` holds the files as they were read and verified (snapshot.py).
    """
    findings: list[Finding] = list(snapshot.manifest_problems)
    expected = {record.id: record.fed_sha256 for record in records}
    readings = {record.id: record.reading for record in records}
    for source in sources:
        problem = snapshot.problems.get(source.name)
        if problem:
            findings += problem
            continue
        golden, costs = snapshot.golden[source.name], snapshot.costs[source.name]
        name = golden_path(source.name).name
        current = signature(source, options_for(source))
        if golden.signature != current:
            changed = sorted(key for key in golden.signature.keys() | current.keys()
                             if golden.signature.get(key) != current.get(key))
            findings.append(Finding(
                "golden-stale",
                f"{name}: recorded under a different {', '.join(changed)} - re-record ({RERECORD}) and review the diff"))
        if golden.recordings < MIN_RECORDINGS:
            findings.append(Finding(
                "golden-stale",
                f"{name}: recorded {golden.recordings} time(s); the gate needs {MIN_RECORDINGS} independent recordings of "
                "every source, so that a value that differs between them is marked unstable instead of being required "
                f"(H32) - re-record: {RERECORD}"))
        if costs.signature != golden.signature or costs.recordings != golden.recordings:
            findings.append(Finding("golden-stale", f"{cost_path(source.name).name}: its signature or its number of "
                                                    "recordings is not its golden file's - the costs were measured for "
                                                    "other answers"))
        for record_id in sorted(expected.keys() - golden.sha256.keys()):
            findings.append(Finding("golden-incomplete", f"{name}: no entry for corpus record {record_id}"))
        for record_id in sorted(golden.sha256.keys() - expected.keys()):
            findings.append(Finding("golden-orphan", f"{name}: entry for {record_id}, which is not in the corpus"))
        for record_id in sorted(expected.keys() & golden.sha256.keys()):
            if golden.sha256[record_id] != expected[record_id]:
                now = f" (the record is now {readings[record_id]})" if readings[record_id] else ""
                findings.append(Finding("golden-stale", f"{name}: {record_id} was recorded for different record "
                                                        f"content{now} - re-record ({RERECORD}) and review the diff"))
            observation = golden.observations[record_id]
            for problem_text in [*observations.validate(observation), *_problems(golden, record_id)]:
                findings.append(Finding("golden-invalid", f"{name}: {record_id}: {problem_text}"))
            cost = costs.records.get(record_id)
            if cost is None:
                findings.append(Finding("golden-incomplete", f"{cost_path(source.name).name}: no entry for {record_id}"))
            elif len(cost.positions) != len(observation.positions) or None in cost.positions \
                    or (cost.end is None) != (observation.status == "timeout") \
                    or (cost.mate and len(cost.mate) != len(observation.positions)):
                findings.append(Finding("golden-invalid", f"{cost_path(source.name).name}: {record_id}: the costs do "
                                                          "not fit the recorded positions and outcome"))
        for record_id in sorted(costs.records.keys() - golden.sha256.keys()):
            findings.append(Finding("golden-orphan", f"{cost_path(source.name).name}: entry for {record_id}, which has "
                                                     "no golden entry"))
    return findings


# --- what a recording says, for DISAGREEMENTS.md ---------------------------------


def _pairs() -> list[tuple[str, str]]:
    return [(first, second) for index, first in enumerate(ORACLE_SOURCES) for second in ORACLE_SOURCES[index + 1:]]


def disagreements(records: list[Record], loaded: dict[str, Golden]) -> list[Disagreement]:
    """Every contradiction between two oracles' golden answers, in a stable order."""
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


def render_disagreements(records: list[Record], loaded: dict[str, Golden], required: list[str] | None = None) -> str:
    """DISAGREEMENTS.md, from the golden files of every oracle source (as read and verified).

    ``required`` is the section that says what each tier requires of each
    source (stability.required_table), rendered from the cost files and
    gate.toml as well.
    """
    found = disagreements(records, loaded)
    lines = [
        "# Oracle disagreements",
        "",
        "<!-- Generated by `python oracles/harness.py golden --write` (or `golden --render`) from oracles/golden/ "
        "and oracles/gate.toml. Do not edit. -->",
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
        "had not moved. \"Timeout\" counts the records whose recording was cut at its deadline",
        "(from the position it was searching on, everything is unknown at that budget);",
        "\"unknown at a limit\" the answers a limit left out inside the positions it answered",
        "(a mate search's time or memory), and \"unstable\" the values that differed between",
        "independent recordings - neither is compared with anything.",
        "",
        "| Source | ok | rejected | unsupported | error | timeout | positions | known values | unknown at a limit "
        "| unstable | described | with markers |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for source in ORACLE_SOURCES:
        golden = loaded[source]
        present = [record for record in records if record.id in golden.observations]
        items = [golden.observations[record.id] for record in present]
        counts = {status: sum(item.status == status for item in items) for status in observations.STATUSES}
        described = sum(position.placement is not None for item in items for position in item.positions)
        marked = sum(position.fingerprint is not None for item in items for position in item.positions)
        unknown = sum(len(fields) for record in present for fields in golden.unknown.get(record.id, {}).values())
        unstable = sum(len(fields) for record in present for fields in golden.unstable.get(record.id, {}).values())
        lines.append(
            f"| `{source}` | {counts['ok']} | {counts['rejected']} | {counts['unsupported']} | {counts['error']} "
            f"| {counts['timeout']} | {sum(len(item.positions) for item in items)} | {sum(item.known() for item in items)} "
            f"| {unknown} | {unstable} | {described} | {marked} |"
        )
    if required is not None:
        lines += ["", *required]
    converted = [record for record in sorted(records, key=lambda item: item.id) if record.reading]
    if converted:
        lines += [
            "",
            "## Records not read as stored",
            "",
            "Decision H28 has one exception, ruled at CHECKPOINT 3: a record of cwmtt's in the official",
            "client's export form, whose moves name their board twice (`(0T1)Pe2(0T1)e3`), is converted to",
            "the notation specification's 5dpgn (`oracles/oracle_harness/export_form.py`) before any source",
            "reads it, and everything here about it is about the converted text.",
            "",
            *[f"- `{record.id}`: {record.reading}" for record in converted],
        ]

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
