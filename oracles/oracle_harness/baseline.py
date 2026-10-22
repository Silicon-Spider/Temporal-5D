"""The engine's baseline: what it read on the last green run (decision H47, CHECKPOINT 8).

The commit tier is a smoke test (H47): it gates no ruleset, so it cannot fail an
engine for coming through none. It still fails an engine that declines a record
it read on the last green run, or that reads none. The last green run is on
record here: oracles/engine-baseline.json holds, for every record of the tier
gate.toml [baseline] names (the commit tier), how the engine's replay of it
ended and how many positions it reported, on the run the file was recorded
from - a complete run of that tier, the engine in full mode, that passed.

It is a recorded answer, and is kept like one (H33):

  * written only by an explicit command, `harness.py baseline --write`, which
    runs the tier in full and records what the engine read only if the run is
    green - never by a gate run, and never by the engine: the file lives in the
    harness, which every gate run hashes at its start and end, and which nothing
    outside it may name;
  * pinned: gate.toml [baseline] sha256 holds the SHA-256 of its text (CRLF
    folded to LF), written by the same command - beside the rulebook's pin, and
    for the same reason: golden/MANIFEST is rewritten by every `golden --write`,
    and re-recording the oracles must never re-pin the engine's baseline. Every
    read verifies it against the pin first: a gate run reads it once, before
    anything is compared (snapshot.take), `verify` checks it, and
    `baseline --write` reads the baseline it replaces only once it is shown to be
    the one pinned - otherwise it runs and writes nothing;
  * one record per line, so that a re-recording is reviewed as a diff - and
    `baseline --write` lists what the new file no longer holds: a record the
    engine now reads less of is a loss of checking, never churn.

What the pin catches is an edit that does not re-pin as well: an accident, a
partial edit, a bad merge, a file changed while a run lasts. A file edited
together with its pin - one SHA-256 away - passes every check; as for MANIFEST
(H33), only review catches that. What makes such an edit show in review: the
file and its pin change only together, one line per record, so a hand edit is a
diff of a record's line and of the pin with no `baseline --write` run behind it;
and `verify` fails a baseline that holds no line for a record of its tier, so a
line cannot simply be deleted. Recording a LOSS with `baseline --write` to make a
gate run pass is the same edit by other means, and is never done.

What a gate run compares - in every tier that runs the engine in full mode, a
restricted run included, over the records the baseline holds that the run ran:

  * the engine **declines a record it read on the last green run** when it reads
    less of it now: it reports fewer positions than the baseline, or as many but
    without the verdict the baseline ended with - an end other than `ok` where
    the baseline ended `ok`, or other than a refusal as illegal (`illegal_move`,
    `cannot_submit`) where the baseline ended in one. A record the baseline gives
    no position (a ruleset the engine declined, or a set-up it could not read)
    was not read, and there is nothing to decline; a record read further than the
    baseline is no decline (record a new baseline to hold the engine to it); and
    a run that ended in an error or a time-out fails the gate on its own
    (engine-failed), so it is not reported again as a decline;
  * the engine **reads none** when, in a complete run of a tier, it replays none
    of the tier's records: none ends `ok`, or in a refusal as illegal - a refusal
    as unreadable, a declined ruleset, an error and a time-out replay nothing.
    That judges a tier as a whole, so a restricted run does not apply it (gate.py).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import config, paths
from .corpus import Record, select
from .observations import HARD_REJECTIONS, UNREADABLE, Observation
from .util import Finding, HarnessError, platform_key, sha256_text

SCHEMA = 1
WRITE_COMMAND = f"python {paths.HARNESS_REL}/harness.py baseline --write"
RESTORE_COMMAND = f"git checkout -- {paths.HARNESS_REL}/{paths.ENGINE_BASELINE_FILE.name}"
# How a run that read the record ended with a verdict on its last action: every action replayed, or one refused as
# illegal. A refusal as unreadable is no verdict (the gate's own rule: oracles/README.md, "The engine contract").
_REFUSED = tuple(f"rejected:{reason}" for reason in HARD_REJECTIONS)
# The ends a green run's engine can have: an error or a time-out fails the gate (engine-failed).
_ENDS = ("ok", "unsupported", *_REFUSED, f"rejected:{UNREADABLE}")


class BaselineError(HarnessError):
    """The baseline cannot be compared with: it is missing, does not match its pin, or cannot be read."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Entry:
    """How the engine read one record on the last green run."""

    sha256: str  # the text it was given (Record.fed_sha256): what is recorded is about that text
    end: str  # "ok", "rejected:<reason>" or "unsupported"
    positions: int  # how many positions it reported


@dataclass
class Baseline:
    tier: str  # the tier whose green run this is
    recorded: str  # when: YYYY-MM-DD
    # The engine as that run ran it: its command as gate.toml configures it, its mode, the SHA-256 of the program
    # file, the platform. For the reader; nothing compares it.
    engine: dict
    records: dict[str, Entry] = field(default_factory=dict)
    digest: str = ""  # SHA-256 of the text this was read from

    def read(self) -> int:
        """How many records the engine replayed on that run (to their end, or to an action it held illegal)."""
        return sum(entry.end == "ok" or entry.end in _REFUSED for entry in self.records.values())


def end_of(observation: Observation) -> str:
    """How a run ended, as the baseline writes it: "ok", "rejected:<reason>", "unsupported", "error", "timeout"."""
    return f"rejected:{observation.reason}" if observation.status == "rejected" else observation.status


def replays(observation: Observation) -> bool:
    """Did the engine replay the record - read it to its end, or to an action it holds illegal?"""
    return observation.status == "ok" or (observation.status == "rejected" and observation.reason in HARD_REJECTIONS)


def describe_read(end: str, positions: int) -> str:
    """How far a run read a record, in words: "read it to its end (17 positions)"."""
    if end == "ok":
        return f"read it to its end ({positions} positions)"
    if end in _REFUSED:
        return f"refused action {positions} as {end.split(':', 1)[1]} ({positions} positions)"
    if end == f"rejected:{UNREADABLE}":
        if not positions:
            return "refused its set-up as unreadable (no position)"
        return f"could not read action {positions} ({positions} positions)"
    if end == "unsupported":
        return "declined its ruleset (no position)"
    return f"stopped with {end} after {positions} position(s)"


def reads_less(then_end: str, then_positions: int, now_end: str, now_positions: int) -> bool:
    """Does a run that ended ``now_end`` after ``now_positions`` read less of a record than the baseline's?"""
    if now_positions < then_positions:
        return True
    if then_end == "ok":
        return now_end != "ok"
    if then_end in _REFUSED:
        return now_positions == then_positions and now_end not in _REFUSED
    return False


# --- writing ------------------------------------------------------------------------


def from_run(tier: str, records: list[Record], reads: dict[str, Observation], engine_run: dict | None,
             recorded: str | None = None) -> Baseline:
    """The baseline of one green run: how the engine read each of ``records``, from its reports ``reads``."""
    run = engine_run or {}
    engine = {"command": list(run.get("command") or []), "mode": run.get("mode"), "platform": platform_key(),
              "sha256": run.get("sha256")}
    entries = {record.id: Entry(sha256=record.fed_sha256, end=end_of(reads[record.id]),
                                positions=len(reads[record.id].positions))
               for record in records}
    return Baseline(tier=tier, recorded=recorded or time.strftime("%Y-%m-%d"), engine=engine, records=entries)


def dump(baseline: Baseline) -> str:
    """Deterministic text, one record per line, so that a change in what the engine read is a one-line diff."""
    lines = ["{", f'  "schema": {SCHEMA},', f'  "tier": {json.dumps(baseline.tier)},',
             f'  "recorded": {json.dumps(baseline.recorded)},',
             f'  "engine": {json.dumps(baseline.engine, sort_keys=True)},',
             '  "unit": "end: how the engine\'s replay of the record ended; positions: how many it reported; sha256: '
             'the record text it was given",',
             '  "records": {']
    ordered = sorted(baseline.records.items())
    for number, (record_id, entry) in enumerate(ordered):
        item = {"end": entry.end, "positions": entry.positions, "sha256": entry.sha256}
        lines.append(f"    {json.dumps(record_id)}: {json.dumps(item)}" + ("," if number + 1 < len(ordered) else ""))
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def write(baseline: Baseline, gate_file: Path | None = None) -> tuple[str, str]:
    """Write the baseline file, then pin its hash in gate.toml [baseline] sha256; returns (the old pin, the new one).

    The file first: an interrupted write leaves a file its pin does not match, which `verify` reports.
    """
    text = dump(baseline)
    paths.ENGINE_BASELINE_FILE.write_text(text, encoding="utf-8", newline="\n")
    return pin(sha256_text(text.encode("utf-8")), gate_file)


def pin(digest: str, gate_file: Path | None = None) -> tuple[str, str]:
    """Write ``digest`` into gate.toml [baseline] sha256 - only that one value; returns (the old pin, the new one)."""
    gate_file = gate_file or paths.GATE_FILE
    spec = config.load(gate_file).baseline
    if spec is None:
        raise HarnessError(f"{gate_file.name}: no [baseline] table to write the pin into")
    config.write_sha256(gate_file, "baseline", digest)
    after = config.load(gate_file).baseline
    if after is None or after.sha256 != digest:  # what was written must read back as the pin
        raise HarnessError(f"{gate_file.name}: the baseline's pin did not read back as written")
    return spec.sha256, digest


# --- reading ------------------------------------------------------------------------


def _entry(record_id: str, item: object) -> Entry:
    if not isinstance(item, dict) or set(item) != {"end", "positions", "sha256"}:
        raise ValueError(f"{record_id}: an entry holds exactly end, positions and sha256")
    end, positions, digest = item["end"], item["positions"], item["sha256"]
    if end not in _ENDS:
        raise ValueError(f"{record_id}: end {end!r} is none a green run can have ({', '.join(_ENDS)})")
    if isinstance(positions, bool) or not isinstance(positions, int) or positions < 0:
        raise ValueError(f"{record_id}: positions must be a whole number")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{record_id}: sha256 must name the record text the engine was given")
    if (end == "ok" or end in _REFUSED) and positions < 1:
        raise ValueError(f"{record_id}: a run that ended {end} reported at least one position")
    if end == "unsupported" and positions:
        raise ValueError(f"{record_id}: a declined ruleset is declined before the first position")
    return Entry(sha256=digest, end=end, positions=positions)


def parse(raw: bytes, digest: str = "") -> Baseline:
    """A baseline from its text; HarnessError when it is not one."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"invalid JSON: {error}") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise HarnessError(f"not a baseline in schema {SCHEMA}")
    tier, recorded, engine, records = data.get("tier"), data.get("recorded"), data.get("engine"), data.get("records")
    if not isinstance(tier, str) or not tier or not isinstance(recorded, str) or not isinstance(engine, dict) \
            or not isinstance(records, dict):
        raise HarnessError("it needs a tier, the date it was recorded, the engine and the records")
    try:
        entries = {record_id: _entry(record_id, item) for record_id, item in records.items()}
    except ValueError as error:
        raise HarnessError(str(error)) from None
    return Baseline(tier=tier, recorded=recorded, engine=engine, records=entries, digest=digest)


def load(pin_digest: str, path: Path | None = None) -> Baseline:
    """The baseline, read only once its bytes are shown to be the ones gate.toml pins (H33).

    The only way the baseline is read - by a gate run, `verify`, and `baseline --write`, which lists what a new
    baseline no longer holds against this one: never against a file edited by hand.
    """
    path = path or paths.ENGINE_BASELINE_FILE
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        raise BaselineError("engine-baseline-missing",
                            f"{path.name}: not there, though gate.toml [baseline] pins it - restore it "
                            f"({RESTORE_COMMAND})") from None
    digest = sha256_text(raw)
    if digest != pin_digest:
        raise BaselineError("engine-baseline-integrity",
                            f"{path.name} does not match its pin in gate.toml [baseline] ({pin_digest[:12]}; it is "
                            f"now {digest[:12]}): it was changed after `baseline --write` wrote it - a recorded answer "
                            f"is never edited by hand; restore it ({RESTORE_COMMAND})")
    try:
        return parse(raw, digest)
    except HarnessError as error:
        raise BaselineError("engine-baseline-invalid", f"{path.name}: {error}") from None


# --- checking -----------------------------------------------------------------------


def static_findings(cfg: config.GateConfig, records: list[Record], loaded: Baseline | None,
                    problems: list[Finding]) -> list[Finding]:
    """Is the baseline there, the file its pin names, and about the corpus as it is - every record of its tier?

    Required wherever an engine is compared in full mode: without it the commit
    tier could not tell a record the engine declines from one it never read
    (H47). ``loaded`` and ``problems`` are what the snapshot read (snapshot.take).
    A record of the tier it holds no line for is one no tier holds the engine to:
    added to the corpus since the baseline was recorded, or its line deleted.
    """
    spec = cfg.baseline
    if spec is None or not spec.sha256:
        if not cfg.engine_command or cfg.engine_mode != "full":
            return []
        where = "gate.toml has no [baseline] table" if spec is None else "gate.toml [baseline] pins no baseline"
        return [Finding("engine-baseline-missing",
                        f"{where}: there is no record of what the engine read on the last green run, so the commit "
                        f"tier cannot tell a record it now declines from one it never read (H47) - after a green run "
                        f"of the tier, record one: {WRITE_COMMAND}")]
    if problems:
        return list(problems)
    if loaded is None:
        return [Finding("engine-baseline-missing", f"{paths.ENGINE_BASELINE_FILE.name} was not read")]
    findings = []
    if loaded.tier != spec.tier:
        findings.append(Finding("engine-baseline-stale",
                                f"{paths.ENGINE_BASELINE_FILE.name} records a green run of the {loaded.tier} tier, and "
                                f"gate.toml [baseline] names {spec.tier} - record it again: {WRITE_COMMAND}"))
    corpus = {record.id: record for record in records}
    for record_id, entry in sorted(loaded.records.items()):
        record = corpus.get(record_id)
        if record is None:
            findings.append(Finding("engine-baseline-stale",
                                    f"{paths.ENGINE_BASELINE_FILE.name} holds {record_id}, which is not in the corpus - "
                                    f"record it again: {WRITE_COMMAND}"))
        elif entry.sha256 != record.fed_sha256:
            findings.append(Finding("engine-baseline-stale",
                                    f"{paths.ENGINE_BASELINE_FILE.name}: {record_id} was recorded for other record "
                                    f"content - record it again ({WRITE_COMMAND}) and review the diff"))
    tier = cfg.tiers.get(spec.tier)
    for record in select(records, tier.record_tiers) if tier is not None else []:
        if record.id not in loaded.records:
            findings.append(Finding("engine-baseline-stale",
                                    f"{paths.ENGINE_BASELINE_FILE.name} holds no line for {record.id}, a record of the "
                                    f"{spec.tier} tier, so no tier holds the engine to what it reads of it - record it "
                                    f"again ({WRITE_COMMAND}) and review the diff"))
    return findings


def compare(loaded: Baseline, records: list[Record], reads: dict[str, Observation]) -> tuple[list[Finding], str]:
    """(an engine-declined finding for each record the engine reads less of than on the last green run, a summary).

    Over the records of ``records`` that the baseline holds - recorded for the
    same text - and that the engine was run on (``reads``).
    """
    findings: list[Finding] = []
    compared = further = 0
    for record in records:
        entry = loaded.records.get(record.id)
        observed = reads.get(record.id)
        if entry is None or observed is None or entry.sha256 != record.fed_sha256:
            continue
        if observed.status in ("error", "timeout"):
            continue  # a breach of the engine contract, or a time-out: engine-failed names the record already
        compared += 1
        end, positions = end_of(observed), len(observed.positions)
        if reads_less(entry.end, entry.positions, end, positions):
            findings.append(Finding(
                "engine-declined",
                f"{record.id}: the engine {describe_read(end, positions)}, where on the last green run "
                f"({loaded.tier} tier, recorded {loaded.recorded}) it {describe_read(entry.end, entry.positions)} - "
                "it declines a record it read then (H47)"))
        elif positions > entry.positions:
            further += 1
    summary = (f"{compared} record(s) compared with what the engine read on the last green run ({loaded.tier} tier, "
               f"recorded {loaded.recorded}): {len(findings)} declined")
    if further:
        summary += (f"; {further} read further than then - record a new baseline ({WRITE_COMMAND}) to hold the engine "
                    "to them")
    return findings, summary


def read_nothing(records: list[Record], reads: dict[str, Observation]) -> Finding | None:
    """An engine-read-nothing finding where the engine replayed none of ``records`` it was run on; otherwise None."""
    present = [reads[record.id] for record in records if record.id in reads]
    if not present or any(replays(observation) for observation in present):
        return None
    ends = ", ".join(f"{count} {end}" for end, count in Counter(end_of(item) for item in present).most_common())
    return Finding("engine-read-nothing",
                   f"the engine replayed none of the {len(present)} record(s) it was given - none to its end, nor to "
                   f"an action it holds illegal ({ends}): an engine that reads nothing is compared with nothing (H47)")


def changes(previous: Baseline | None, new: Baseline) -> list[str]:
    """What a new baseline holds differently from the one it replaces, for review - a loss first of all.

    ``previous``: the baseline gate.toml pinned, read verified (load); None where none was pinned.
    """
    if previous is None:
        return [f"no baseline was pinned before: all {len(new.records)} record(s) are new"]
    lost, gained, other = [], [], []
    for record_id in sorted(previous.records.keys() | new.records.keys()):
        old, fresh = previous.records.get(record_id), new.records.get(record_id)
        if fresh is None:
            lost.append(f"LOSS: {record_id} is no longer held (the engine {describe_read(old.end, old.positions)})")
        elif old is None:
            other.append(f"added: {record_id} - the engine {describe_read(fresh.end, fresh.positions)}")
        elif old.sha256 != fresh.sha256:
            other.append(f"{record_id}: recorded for new record content - the engine "
                         f"{describe_read(fresh.end, fresh.positions)}")
        elif reads_less(old.end, old.positions, fresh.end, fresh.positions):
            lost.append(f"LOSS: {record_id} - the engine {describe_read(fresh.end, fresh.positions)}, where it "
                        f"{describe_read(old.end, old.positions)}")
        elif (fresh.end, fresh.positions) != (old.end, old.positions):
            kind = "further" if fresh.positions > old.positions else "changed"
            gained.append(f"{kind}: {record_id} - the engine {describe_read(fresh.end, fresh.positions)}, where it "
                          f"{describe_read(old.end, old.positions)}")
    return lost + gained + other
