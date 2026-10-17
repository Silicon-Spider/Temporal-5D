"""The gate: static verification, tiered oracle runs, and the engine differential.

Tiers (design spec 11.0):

  commit   every push - the boundary, the pins, the corpus, and the fast oracle
  nightly  the two heavier oracles as well, on the whole corpus
  release  before any generator release or proof-channel enablement - everything
           in nightly, and it fails unless an engine was compared and came
           through at least one named variant

Only a tier that runs every oracle search over the whole corpus (nightly,
release) gates a ruleset, and there an engine with no finding against it that
came through no ruleset fails (engine-nothing-gated). Any other tier - commit - is
a smoke test (H47): an engine with no finding against it passes, and the summary
and report say that the tier gates nothing. In every tier the engine fails on an
unexplained divergence (one known-divergences.toml does not record), a telltale or
a breach of the engine contract. The rule is keyed to what a tier runs, never to
its name; a run restricted to some sources or records keeps its tier's rule, and
only a tier that runs every oracle search over the whole corpus may require an
engine (config.load refuses any other: it could gate nothing).

Each run reads what it compares against once, verified (snapshot.py, H33); then
requires 100% of each tier's required set of recorded answers (stability.py,
H32); fails on any engine answer that reproduces a recorded oracle error
(divergences.telltales, H34); and fails if any harness file changed while it ran.

No proof may enter the solved store until the generator has passed this gate
(decision H23). While there is no engine to compare, the oracle layer can be
green, but the differential gate has not run and the result says so.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import (
    boundary,
    config,
    corpus,
    divergences,
    fidelity,
    golden,
    integrity,
    lockfile,
    observations,
    paths,
    provenance,
    rulebook,
    runner,
    snapshot,
    stability,
)
from .adapters import (
    ORACLE_SOURCES,
    PARSE_ONLY_SOURCES,
    PRIMARY_SOURCES,
    EngineSource,
    RunOptions,
    Source,
    make_source,
    oracle_of,
)
from .adapters.base import number
from .adapters.ftxi import FtxiSource
from .corpus import CUSTOM_POSITIONS, Record
from .lockfile import Lock
from .observations import HARD_REJECTIONS, OBSERVABLES, REPLAY_KINDS, UNREADABLE, Observation
from .snapshot import Snapshot
from .util import Finding, HarnessError, is_windows, platform_key, sha256_file, sha256_text

Log = Callable[[str], None]
Section = tuple[str, list[Finding]]

# A report that followed the record to its end, or to the action it refuses.
_COMPLETE = ("ok", "rejected")
_LABELS = {"in_check": "check", "legal_actions": "count", "terminal": "mate"}
# How long the calibration probe may take, on any machine.
PROBE_TIMEOUT_S = 900.0
# What a tier that does not run every oracle search over the whole corpus says of an engine compared in full mode
# (H47), and why a ruleset that met the gating conditions with its oracles alone is still not gated.
SMOKE_TEST = ("this tier gates nothing - only a tier that runs every oracle search over the whole corpus gates a "
              "ruleset, or fails an engine that came through none (H47)")
SMOKE_REASON = "a smoke test: it met the gating conditions with this tier's oracles alone"


@dataclass(frozen=True)
class Gated:
    """What stands behind one ruleset the engine came through."""

    records: int
    positions: int  # positions the engine reported, over all the ruleset's records
    checked: dict[str, int]  # observable -> positions at which some oracle's answer settles it and agrees with the engine
    oracles: int  # the fewest independent implementations that followed any one record as far as the engine
    # Positions at which some oracle held the engine's boards and pieces (equal
    # placement hashes): the record's moves read alike, markers not compared.
    aligned: int = 0
    # Of those, positions at which some oracle held exactly the engine's state,
    # unmoved markers included (equal fingerprints).
    matched: int = 0

    def describe(self) -> str:
        checked = ", ".join(f"{_LABELS[name]} {self.checked[name]}" for name in OBSERVABLES)
        return (f"{self.records} record{'s' if self.records != 1 else ''}, {self.positions} positions; "
                f"an oracle held the engine's boards and pieces at {self.aligned} of them, and its exact state "
                f"(markers included) at {self.matched}; "
                f"independently answered: {checked}; each record followed as far as the engine went by at least "
                f"{self.oracles} oracle{'s' if self.oracles != 1 else ''}")


@dataclass(frozen=True)
class ParseAgreement:
    """How far an engine in parse mode was compared, and how far it agreed."""

    records: int = 0  # records the tier ran
    read: int = 0  # of them, the records the engine replayed to their end, or to an action it refused
    positions: int = 0  # positions the engine reported over those
    # Positions at which some oracle said what it set up, up to and including where it parted from the engine:
    # where it parted, it described a position other than the engine's.
    described: int = 0
    boards: int = 0  # positions at which some oracle held the engine's boards and pieces
    states: int = 0  # positions at which some oracle held its exact state, unmoved markers included
    sides: int = 0  # positions at which some oracle gave the side to move
    sides_agreed: int = 0  # of them, positions at which every oracle that gave one gave the engine's
    replays: int = 0  # implementation reports whose replay of a record was compared with the engine's
    replays_agreed: int = 0  # of them, those that read, accepted and refused the same actions
    # Of them, those that differed only where the engine played an action the oracle held illegal: in parse mode
    # that is no verdict (the engine has no check detection), so it is neither a disagreement nor an agreement.
    replays_unjudged: int = 0
    setups_refused: int = 0  # records whose set-up the engine refused as unreadable (action 0)

    def describe(self) -> str:
        refused = f", and refused the set-up of {self.setups_refused}" if self.setups_refused else ""
        unjudged = (f", and with {self.replays_unjudged} more but for actions it played that the oracle held illegal "
                    "(no verdict in parse mode)") if self.replays_unjudged else ""
        return (f"the engine read {self.read} of {self.records} records ({self.positions} positions){refused}; an "
                f"oracle said what it set up at {self.described} of those positions, held the engine's boards and "
                f"pieces at {self.boards} and its exact state (markers included) at {self.states}; the side to move "
                f"was compared at {self.sides} and agreed at {self.sides_agreed}; the engine's replay - the actions it "
                f"read, accepted and refused - was compared with {self.replays} implementation reports and agreed "
                f"with {self.replays_agreed}{unjudged}; check, counts and mate were not compared, and no ruleset is "
                "gated")


@dataclass(frozen=True)
class Probe:
    """The calibration probe: a fixed workload, timed here and on the machine the golden files were recorded on."""

    source: str
    record: str
    positions: int
    seconds: float | None = None  # on this machine
    reference_s: float | None = None  # on the recording machine
    note: str = ""

    @property
    def factor(self) -> float | None:
        """How many times slower this machine ran the probe than the recording machine did."""
        if self.seconds is None or not self.reference_s:
            return None
        return self.seconds / self.reference_s

    def describe(self) -> str:
        what = f"{self.source} over the first {self.positions} positions of {self.record}"
        if self.seconds is None:
            return f"{what}: not measured ({self.note})"
        here = f"{what}: {self.seconds:.2f} s on this machine"
        if self.factor is None:
            return f"{here}; no reference from the recording machine ({self.note or 'none recorded'})"
        return (f"{here}, {self.reference_s:.2f} s on the recording machine - this machine took "
                f"{self.factor:.2f}x the recording machine's time")


@dataclass
class GateResult:
    tier: str
    sections: list[Section] = field(default_factory=list)  # the oracle layer
    engine_sections: list[Section] = field(default_factory=list)  # the engine against it
    tallies: dict[str, stability.Tally] = field(default_factory=dict)  # source -> its runs against their required sets
    engine: str = "not configured"  # not configured | passed | failed
    engine_mode: str = "full"  # full | parse
    # The rulesets that met the gating conditions (engine_differential) with the oracles this run ran, and why the
    # others did not. A gating verdict only where the tier runs every oracle (H47): in a smoke test nothing is
    # gated, and what may be said of the rulesets is `verdicts`.
    gated: dict[str, Gated] = field(default_factory=dict)
    ungated: dict[str, str] = field(default_factory=dict)  # ruleset -> why it is not gated
    parse: ParseAgreement | None = None  # an engine compared in parse mode
    partial: bool = False
    require_engine: bool = False
    # The tier runs every oracle search over the whole corpus. A tier's name promises nothing. A run restricted to
    # some sources or records keeps its tier's value (and is marked partial).
    full_strength: bool = False
    margin: float = 1.0
    calibration: Probe | None = None
    # record id -> how every source read it, for the records not read as stored:
    # converted from the official client's export form, or given to no source (export_form.py).
    converted: dict[str, str] = field(default_factory=dict)
    # The engine as this run ran it - its command, the file that ran and that file's SHA-256, its time limit
    # (engine_program) - or None where no engine ran.
    engine_run: dict | None = None

    def findings(self) -> list[Finding]:
        return [finding for _, items in (*self.sections, *self.engine_sections) for finding in items]

    @property
    def oracle_layer_ok(self) -> bool:
        return not any(items for _, items in self.sections)

    @property
    def variants(self) -> dict[str, Gated]:
        """The gated rulesets that are named variants - what a proof could be about.

        Only what this run may state as gated (`verdicts`): in a smoke test, none.
        """
        gated, _ = self.verdicts
        return {ruleset: item for ruleset, item in gated.items() if ruleset != CUSTOM_POSITIONS}

    @property
    def smoke_test(self) -> bool:
        """Was an engine compared in full mode by a tier that does not run every oracle search over the whole corpus?

        Such a tier is a smoke test (H47): it gates nothing. An engine with no
        finding against it passes it, whether or not some ruleset met the gating
        conditions with the tier's oracles alone, and its summary and report say
        that the tier gates nothing. Keyed to what the tier runs (full_strength),
        never to its name.
        """
        return self.engine != "not configured" and self.engine_mode == "full" and not self.full_strength

    @property
    def verdicts(self) -> tuple[dict[str, Gated], dict[str, str]]:
        """(the gated rulesets, every other ruleset -> why it is not gated), as this run may state them.

        In a smoke test nothing is gated: a ruleset that met the gating
        conditions with the tier's oracles alone is not gated either, and says so.
        """
        if not self.smoke_test:
            return self.gated, self.ungated
        met = {ruleset: f"{SMOKE_REASON} ({item.describe()})" for ruleset, item in self.gated.items()}
        return {}, dict(sorted({**self.ungated, **met}.items()))

    @property
    def ok(self) -> bool:
        if not self.oracle_layer_ok or self.engine == "failed":
            return False
        if self.engine == "passed":
            if self.engine_mode == "parse":
                # An engine compared in parse mode gates nothing; a tier that
                # requires an engine requires one compared on the rules.
                return not self.require_engine
            if self.require_engine:
                # A tier that requires an engine requires it to come through a
                # variant a proof could be about - one this run may state as
                # gated, so never in a smoke test (config.load refuses a tier that
                # requires an engine and is one).
                return bool(self.variants)
            # In a tier that runs every oracle search over the whole corpus, an
            # engine with no finding against it that came through no ruleset has
            # not passed: agreeing with the oracles about nothing is not a pass.
            # Any other tier is a smoke test that gates nothing (H47): an engine
            # with no finding against it passes.
            return bool(self.gated) or self.smoke_test
        return not self.require_engine

    @property
    def proof_channel(self) -> str:
        if self.engine == "not configured":
            return "DISABLED - no engine has been compared against the oracles (H23)"
        if self.engine_mode == "parse":
            return ("DISABLED - parse mode: the engine was compared on what it set up and how it replayed each record, "
                    "never on check, counts or mate (H23)")
        if not self.ok:
            return "DISABLED - the gate failed"
        if self.partial or not (self.require_engine and self.full_strength):
            return ("DISABLED - only a complete run of a tier that requires an engine and runs every oracle "
                    "over the whole corpus can enable it")
        if not self.variants:
            return "DISABLED - the engine came through no named variant"
        return "may be enabled for: " + "; ".join(_describe_gated(self.variants))


def _describe_gated(gated: dict[str, Gated]) -> list[str]:
    return [f"{ruleset} ({item.describe()})" for ruleset, item in sorted(gated.items())]


def _options(cfg: config.GateConfig, timeout_s: float, mate_budget_ms: int,
             stop_after: tuple[int, str] | None = None) -> RunOptions:
    return RunOptions(cap=cfg.cap, timeout_s=timeout_s, mate_budget_ms=mate_budget_ms, stop_after=stop_after)


def recording_options(cfg: config.GateConfig, source: Source) -> RunOptions:
    """The options a source's golden file is recorded under: the recording tier's budgets."""
    return _options(cfg, cfg.recording.timeout_for(source.name), cfg.recording.mate_budget_ms)


def _progress(log: Log) -> Callable[[Observation], None]:
    def report(observation: Observation) -> None:
        status = "stopped" if observation.reason == "stopped" else observation.status
        log(f"  {observation.source:<12} {observation.record:<40} {status:<11} "
            f"{len(observation.positions):>4} positions  {observation.elapsed_s:6.1f}s")

    return report


def engine_argv(command: tuple[str, ...] | list[str]) -> list[str]:
    """The engine command as the gate runs it: a program named by a relative path is the repository root's.

    The command runs from the repository root (adapters/engine.py). Linux and macOS find a relative program from
    there; Windows finds it from the directory the harness itself was started in - another checkout's build, or
    none, if that is not the root. Resolved here, every platform runs the file engine_command_findings checks,
    wherever the harness was started. Only the program: its arguments are the program's to read, and it runs in
    the root. A bare name is looked up on PATH, as before.
    """
    argv = list(command)
    if argv and not os.path.isabs(argv[0]) and os.path.dirname(argv[0]):
        argv[0] = str(paths.REPO_ROOT / argv[0])
    return argv


def engine_program(argv: list[str]) -> dict:
    """Which file the engine command runs: so that a run says which build of the engine it compared.

    The program as run (engine_argv) and the file found for it - on Windows, a program named without an extension
    runs as its .exe; a bare name, the file on PATH - with its SHA-256 and when it was last written. No file found:
    the engine cannot be started, and every record fails with it.
    """
    if not argv:
        return {}
    program = argv[0]
    if os.path.dirname(program):
        candidates = [Path(program)]
        if is_windows() and not Path(program).suffix:
            candidates.append(Path(program + ".exe"))
        found = next((candidate for candidate in candidates if candidate.is_file()), None)
    else:
        on_path = shutil.which(program)
        found = Path(on_path) if on_path else None
    if found is None:
        return {"program": program, "file": None, "sha256": None, "modified": None}
    return {"program": program, "file": str(found), "sha256": sha256_file(found),
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(found.stat().st_mtime))}


def _describe_engine_run(run: dict) -> str:
    if not run.get("file"):
        return f"{run.get('program')}: no such file - has the engine been built?"
    return f"{run['file']} (SHA-256 {run['sha256'][:16]}, written {run['modified']})"


def engine_command_findings(command: tuple[str, ...] | list[str]) -> list[Finding]:
    """The engine's adapter is part of the engine: it may not be a program kept in the harness.

    The harness holds what every oracle says about every record. An adapter
    living beside those files could hand them back instead of asking the engine,
    and the boundary check - which watches what is outside the harness - would
    never look at it.
    """
    findings = []
    # The directory's name anywhere in an argument: a path into it, a module in
    # it (-m oracles.x), a shell line that changes into it (cd oracles && ...).
    names_harness = re.compile(rf"(?<!\w){re.escape(paths.HARNESS_REL)}(?!\w)", re.I)
    for part in command:
        try:
            inside = (paths.REPO_ROOT / part).resolve().is_relative_to(paths.HARNESS_ROOT.resolve())
        except (OSError, ValueError):
            inside = False
        if inside or names_harness.search(part):
            findings.append(Finding(
                "engine-in-harness",
                f"[engine] command names '{part}', which is inside {paths.HARNESS_REL}/ - "
                "the engine's adapter belongs with the engine, outside the harness"))
    return findings


def _calibration_findings(cfg: config.GateConfig, lock: Lock, records: list[Record], frozen: Snapshot) -> list[Finding]:
    """Is the calibration reference the one for the probe gate.toml names, measured with the current code?"""
    spec = cfg.calibration
    if spec.record not in {record.id for record in records}:
        return [Finding("calibration", f"gate.toml [calibration] record '{spec.record}' is not in the corpus")]
    if frozen.calibration is None:
        return [Finding("golden-missing", frozen.calibration_problem or f"no {golden.CALIBRATION_FILE}")]
    recorded = frozen.calibration
    current = make_source(spec.source, lock).signature(RunOptions(cap=cfg.cap))
    if (recorded.source, recorded.record, recorded.positions) != (spec.source, spec.record, spec.positions) \
            or recorded.signature != current:
        return [Finding("golden-stale", f"{golden.CALIBRATION_FILE}: measured with another probe, or with other code, "
                                        f"than gate.toml [calibration] names - re-record: {golden.RERECORD}")]
    return []


def _rendered(cfg: config.GateConfig, records: list[Record], frozen: Snapshot) -> str:
    """DISAGREEMENTS.md as the snapshot's golden and cost files and gate.toml make it."""
    return golden.render_disagreements(records, frozen.golden,
                                       stability.required_table(cfg, records, frozen.golden, frozen.costs))


def static_checks(lock: Lock, cfg: config.GateConfig, records: list[Record], frozen: Snapshot, *,
                  include_untracked: bool = True) -> list[Section]:
    """Everything that can be checked without building or running an oracle, against one verified snapshot."""
    sections: list[Section] = [
        ("pins", lockfile.verify(lock)),
        ("corpus", corpus.verify(records, lock) + corpus.verify_local(records)
         + corpus.verify_excluded(corpus.load_excluded(), lock)),
        ("known divergences", divergences.verify(frozen.divergences, lock, records, frozen.recorded)),
        # The rulebook the known divergences cite, still the text they were adjudicated against (H38).
        ("rulebook", rulebook.verify(cfg.rulebook)),
        ("boundary", boundary.check(lock, cfg.boundary, include_untracked=include_untracked)
         + engine_command_findings(cfg.engine_command)),
        ("provenance", provenance.check(lock, records)),
    ]
    sources = [make_source(name, lock) for name in ORACLE_SOURCES]
    golden_findings = golden.verify_static(sources, lambda source: recording_options(cfg, source), records, frozen)
    golden_findings += _calibration_findings(cfg, lock, records, frozen)
    if not golden_findings:
        # Rendered from the golden files, their costs and gate.toml's budgets: what a tier requires is a table in
        # it, so a required set cannot shrink without a reviewed line of it changing (H32, H33).
        rendered = _rendered(cfg, records, frozen)
        on_disk = ""
        if paths.DISAGREEMENTS_FILE.exists():
            on_disk = paths.DISAGREEMENTS_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
        if on_disk != rendered:
            golden_findings.append(Finding(
                "disagreements-stale",
                f"{paths.DISAGREEMENTS_FILE.name} does not match the golden files, their costs and gate.toml's budgets "
                f"(run: python {paths.HARNESS_REL}/harness.py golden --render, and review the diff - a changed line in "
                "'What each tier requires' is a change in what the gate checks)"))
    sections.append(("golden observations", golden_findings))
    return sections


PROOF_SECTION = "required sets of a tier that can enable the proof channel"


def proof_channel_readiness(cfg: config.GateConfig, records: list[Record], frozen: Snapshot) -> list[Finding] | None:
    """The static non-vacuity check of every tier that can enable the proof channel (H32), or None when not computable.

    It replaces a ratio of recorded answers fixed in the code (90%): such a tier
    must require every terminal verdict each source recorded and the reviewed
    share of its stable values ([stability] proof_share). Like that ratio, it
    is about enabling the proof channel: `verify` prints it, and a run of such a
    tier fails on it - a run of any other tier, which can enable nothing, does not.
    """
    if any(frozen.problems.get(name) for name in ORACLE_SOURCES) or frozen.manifest_problems:
        return None  # the golden files say why
    return stability.proof_channel_findings(cfg, records, frozen)


def _vouches(answer: Observation, observed: Observation) -> bool:
    """Did this oracle follow the record as far as the engine did, and to the same end?

    Where the engine refuses an action, only an oracle that refuses the same
    action as illegal stands behind it. One that merely stopped reading there
    does not, and neither does one that could not match the move's text: that
    may be notation it does not read, which confirms nothing about the move.
    And where the engine could not read the action itself, it gave no verdict
    for anything to stand behind.
    """
    if not observed.positions or _unread(observed):
        return False
    if observed.status == "rejected":
        return (answer.status == "rejected" and answer.reason in HARD_REJECTIONS
                and answer.actions_accepted == observed.actions_accepted)
    return len(answer.positions) >= len(observed.positions)


def _unread(observed: Observation) -> bool:
    """Did the engine stop because it could not read the record - an action's text, or its set-up?"""
    return observed.status == "rejected" and observed.reason == UNREADABLE


def _telltale_findings(entries: list[divergences.Divergence], engine: EngineSource, records: list[Record],
                       engine_results: runner.Results, recorded: divergences.Recorded, *, judges_actions: bool = True
                       ) -> tuple[list[Finding], set[str], set[str]]:
    """(findings, the ids of the entries reproduced, the records they are about) - H34."""
    reports = {record.id: engine_results[(engine.name, record.id)] for record in records
               if (engine.name, record.id) in engine_results}
    findings, ids, about = [], set(), set()
    for entry, value in divergences.telltales(entries, reports, recorded, judges_actions=judges_actions):
        where = f"action {entry.index}" if entry.kind in REPLAY_KINDS else f"position {entry.index}"
        said = f"{entry.kind} = {value}" if value == entry.oracle_value else \
            f"{entry.kind}: the engine says {value}, {entry.oracle} recorded {entry.oracle_value}"
        findings.append(Finding(
            "telltale",
            f"{entry.id}: the engine reproduces {entry.oracle}'s recorded error at {entry.record} {where} "
            f"({said}): its logic may have been copied from that oracle"))
        ids.add(entry.id)
        about.add(entry.record)
    return findings, ids, about


def without_unstable(results: runner.Results, frozen: Snapshot | None) -> runner.Results:
    """The fresh oracle reports with every value their recording marks unstable taken out (H32).

    A value that differed between independent recordings is excluded from every
    comparison: it can neither fail the engine nor stand behind it.
    """
    if frozen is None:
        return results
    masked: runner.Results = {}
    for (name, record_id), report in results.items():
        unstable = frozen.golden[name].unstable.get(record_id, {}) if name in frozen.golden else {}
        masked[(name, record_id)] = report if not unstable else dataclasses.replace(report, positions=[
            dataclasses.replace(position, **dict.fromkeys(unstable.get(position.index, {}), None))
            for position in report.positions])
    return masked


def _fresh(results: runner.Results) -> divergences.Recorded:
    """The fresh oracle reports, standing in for the recorded ones."""
    return lambda source, record_id: results.get((source, record_id))


def _stale_findings(stale: list[divergences.Divergence]) -> list[Finding]:
    return [Finding("divergence-stale",
                    f"{entry.id}: no longer occurs ({entry.oracle} on {entry.record}, {entry.kind} at {entry.index}) - "
                    "remove it, or find out why")
            for entry in stale]


def engine_differential(
    engine: EngineSource,
    oracle_sources: list[Source],
    records: list[Record],
    results: runner.Results,
    engine_results: runner.Results,
    entries: list[divergences.Divergence],
    min_checked: float = 1.0,
    recorded: divergences.Recorded | None = None,
) -> tuple[list[Finding], dict[str, Gated], dict[str, str]]:
    """Compare the engine with every oracle report; returns (findings, gated, ungated).

    What the engine set up is compared with what each oracle set up before
    anything else: where an oracle's placement parts from the engine's, that is
    a parse divergence - a finding unless a known divergence excuses it - and
    from that position on the oracle answers about another position, so it
    cannot stand behind the record and none of its answers there or after
    counts as checking the engine. Where only the unmoved markers differ, the
    oracle read the same moves to the same pieces and still stands behind the
    record (a difference that decides an action shows as a legality
    divergence), but at each position where its markers differ from the
    engine's - excused or not - its answers are about a position with other
    rights to castle or to step twice, and do not count as checking the engine.

    An engine answer equal to a known divergence's recorded oracle error, at
    that place, is a telltale (H34): a finding ahead of everything else, and
    never "stale". ``recorded`` gives each oracle's recorded answers (the
    snapshot's golden files) - where the oracle did not run, they are all
    there is; by default the fresh oracle reports stand in for them.

    A ruleset counts as gated only if, over all of its records:

      * the engine replayed each one - read it to its end, or to an action it
        holds illegal (a refusal as unreadable is no verdict) - and no
        unexplained divergence or telltale remains;
      * each one was followed as far as the engine went by at least one oracle;
      * for each observable - check, count, mate - some oracle gave an answer
        that settles it at no less than ``min_checked`` of the engine's positions.

    Without the last two an engine could "pass" a ruleset nothing had looked at:
    several variants are read by a single oracle, and one of those has no check
    verdict to give.
    """
    findings: list[Finding] = []
    found: list[observations.Disagreement] = []
    looked: set[str] = set()
    vouched: dict[str, set[str]] = {}
    reported: dict[str, int] = {}
    settled: dict[str, dict[str, int]] = {}
    unsettled: dict[str, dict[str, int]] = {}  # answered only by oracles whose markers differ from the engine's there
    aligned: dict[str, int] = {}
    matched: dict[str, int] = {}
    for record in records:
        observed = engine_results[(engine.name, record.id)]
        if observed.status in ("error", "timeout"):
            findings.append(Finding(
                "engine-failed", f"engine on {record.id}: {observed.status} ({observed.reason}) {observed.detail[:300]}"))
        answers = {source.name: results.get((source.name, record.id)) for source in oracle_sources}
        compared = {}
        parted: dict[str, int] = {}  # implementation -> the first position at which it set up something else
        for name, answer in answers.items():
            if answer is not None:
                # How a record is replayed belongs to the implementation, not to
                # each of its searches: one entry excuses one such disagreement.
                compared[name] = observations.compare(observed, answer, replay=name in PRIMARY_SOURCES)
                found += compared[name]
                index = observations.first_placement_divergence(observed, answer)
                if index is not None:
                    parted[oracle_of(name)] = min(parted.get(oracle_of(name), index), index)
        # An implementation that read the record differently from the engine -
        # an action accepted by one and refused by the other, a record that ends
        # earlier for one of them, a move read as another position - does not
        # stand behind the engine's replay of it, even where a known divergence
        # excuses the difference.
        misread = {oracle_of(name) for name, items in compared.items()
                   if any(item.kind in divergences.REPLAY_KINDS for item in items)} | set(parted)
        # A parse-only source plays any move it can place, legal or not: it says
        # what it read, and stands behind no replay as a rules oracle does.
        vouched[record.id] = {oracle_of(name) for name, answer in answers.items()
                              if answer is not None and name not in PARSE_ONLY_SOURCES
                              and oracle_of(name) not in misread and _vouches(answer, observed)}
        # A position's question counts as independently answered only where an
        # oracle's answer settles it and agrees with the engine's. An answer the
        # engine contradicts confirms nothing - also when the contradiction is on
        # record as the oracle's error - and an answer about another position
        # confirms nothing either: from where the oracle parted from the engine,
        # and wherever its unmoved markers differ from the engine's.
        answered = [dict.fromkeys(OBSERVABLES, False) for _ in observed.positions]
        withheld = [dict.fromkeys(OBSERVABLES, False) for _ in observed.positions]
        boards_held = [False for _ in observed.positions]  # some oracle held the engine's boards and pieces
        state_held = [False for _ in observed.positions]  # some oracle held exactly the engine's state
        for name, answer in answers.items():
            if answer is None:
                continue
            clashes = {(item.kind, item.index) for item in compared[name]}
            shared = list(zip(answered, withheld, observed.positions, answer.positions, strict=False))
            for slot, held, mine, theirs in shared[:parted.get(oracle_of(name))]:
                boards_held[mine.index] = boards_held[mine.index] or bool(observations.aligned(mine, theirs))
                exact = observations.same_state(mine, theirs)
                state_held[mine.index] = state_held[mine.index] or bool(exact)
                for observable in OBSERVABLES:
                    agrees = (observable, theirs.index) not in clashes
                    confirms = agrees and observations.decisive(observable, getattr(theirs, observable))
                    if exact is False:
                        held[observable] = held[observable] or confirms
                    else:
                        slot[observable] = slot[observable] or confirms
            looked.update(
                entry.id for entry in entries
                if (entry.oracle, entry.record) == (name, record.id)
                and divergences.looked_at(entry, observed, answer))
        reported[record.id] = len(observed.positions)
        settled[record.id] = {observable: sum(slot[observable] for slot in answered) for observable in OBSERVABLES}
        unsettled[record.id] = {observable: sum(held[observable] and not slot[observable]
                                                for slot, held in zip(answered, withheld, strict=True))
                                for observable in OBSERVABLES}
        aligned[record.id] = sum(boards_held)
        matched[record.id] = sum(state_held)

    told, told_ids, told_records = _telltale_findings(entries, engine, records, engine_results,
                                                      recorded or _fresh(results))
    findings = told + findings
    unexplained, stale = divergences.apply(entries, found, looked, told_ids)
    for item in unexplained:
        findings.append(Finding("divergence", item.describe()))
    findings += _stale_findings(stale)

    diverged = {item.record for item in unexplained}
    rulesets: dict[str, list[Record]] = {}
    for record in records:
        rulesets.setdefault(record.ruleset, []).append(record)
    gated: dict[str, Gated] = {}
    ungated: dict[str, str] = {}
    for ruleset, items in sorted(rulesets.items()):
        unplayed = [item.id for item in items if engine_results[(engine.name, item.id)].status not in _COMPLETE]
        unread = [item.id for item in items if _unread(engine_results[(engine.name, item.id)])]
        alone = [item.id for item in items if not vouched[item.id]]
        positions = sum(reported[item.id] for item in items)
        checked = {name: sum(settled[item.id][name] for item in items) for name in OBSERVABLES}
        thin = [name for name in OBSERVABLES if checked[name] < min_checked * positions]
        if any(item.id in told_records for item in items):
            ungated[ruleset] = "the engine reproduces a recorded oracle error (telltale)"
        elif any(item.id in diverged for item in items):
            ungated[ruleset] = "unexplained divergence"
        elif unplayed:
            ungated[ruleset] = f"the engine did not replay {_some(unplayed)}"
        elif unread:
            ungated[ruleset] = f"the engine could not read {_some(unread)} (a refusal as unreadable is no verdict)"
        elif alone:
            ungated[ruleset] = f"no oracle followed {_some(alone)} as far as the engine went"
        elif thin:
            name = thin[0]
            ungated[ruleset] = (f"the oracles give a {_LABELS[name]} verdict at only {checked[name]} of the engine's "
                                f"{positions} positions (the floor is {min_checked:.0%})")
            other = sum(unsettled[item.id][name] for item in items)
            if other:
                ungated[ruleset] += (f"; at {other} more, the only {_LABELS[name]} verdict that agrees comes from an "
                                     "oracle whose unmoved markers differ from the engine's there, which does not count")
        else:
            gated[ruleset] = Gated(records=len(items), positions=positions, checked=checked,
                                   oracles=min(len(vouched[item.id]) for item in items),
                                   aligned=sum(aligned[item.id] for item in items),
                                   matched=sum(matched[item.id] for item in items))
    return findings, gated, ungated


def parse_differential(
    engine: EngineSource,
    oracle_sources: list[Source],
    records: list[Record],
    results: runner.Results,
    engine_results: runner.Results,
    entries: list[divergences.Divergence],
    recorded: divergences.Recorded | None = None,
) -> tuple[list[Finding], ParseAgreement]:
    """Compare an engine in parse mode with every oracle report; returns (findings, agreement).

    What is compared: what the engine set up with what each oracle set up
    (placement and marker-only divergences, findings unless excused), the side to
    move, and how it replayed each record - how many actions it read, which it
    accepted and which it refused - with each implementation's replay. Never
    check, counts or mate: the engine answers none of them in parse mode. Like a
    parse-only source, an engine in parse mode plays any action it reads, so its
    playing one an oracle holds illegal is no verdict against the oracle - nor,
    for a known divergence, a look at it or a telltale; its refusal of one an
    oracle played is compared like any other, and so is its refusal of a
    record's set-up (action 0) that an oracle set up. Nothing is gated.
    """
    findings: list[Finding] = []
    found: list[observations.Disagreement] = []
    looked: set[str] = set()
    tally = dict.fromkeys(("read", "positions", "described", "boards", "states", "sides", "sides_agreed",
                           "replays", "replays_agreed", "replays_unjudged", "setups_refused"), 0)
    for record in records:
        observed = engine_results[(engine.name, record.id)]
        if observed.status in ("error", "timeout"):
            findings.append(Finding(
                "engine-failed", f"engine on {record.id}: {observed.status} ({observed.reason}) {observed.detail[:300]}"))
        refused_setup = observations.refused_setup(observed)
        tally["setups_refused"] += refused_setup
        complete = observed.status in _COMPLETE and not refused_setup
        if complete:
            tally["read"] += 1
            tally["positions"] += len(observed.positions)
        described = [False] * len(observed.positions)
        boards = [False] * len(observed.positions)
        states = [False] * len(observed.positions)
        sides = [False] * len(observed.positions)
        clashed = [False] * len(observed.positions)
        for source in oracle_sources:
            answer = results.get((source.name, record.id))
            if answer is None:
                continue
            compared = observations.compare(observed, answer, replay=source.name in PRIMARY_SOURCES)
            # The engine's playing an action this oracle holds illegal is no verdict in parse mode.
            unjudged = [item for item in compared
                        if item.kind == "legality" and item.a_value == "accepted" and not item.soft]
            items = [item for item in compared if item not in unjudged]
            found += items
            clashes = {(item.kind, item.index) for item in items}
            parted = observations.first_placement_divergence(observed, answer)
            shared = list(zip(observed.positions, answer.positions, strict=False))
            for mine, theirs in shared[:None if parted is None else parted + 1]:
                held = observations.aligned(mine, theirs)
                described[mine.index] = described[mine.index] or held is not None
                if mine.index == parted:
                    continue  # where it parted from the engine, it described another position
                boards[mine.index] = boards[mine.index] or bool(held)
                states[mine.index] = states[mine.index] or bool(observations.same_state(mine, theirs))
                if theirs.side is not None:
                    sides[mine.index] = True
                    clashed[mine.index] = clashed[mine.index] or ("side", mine.index) in clashes
            if source.name in PRIMARY_SOURCES and complete and answer.status in _COMPLETE:
                tally["replays"] += 1
                if not any(item.kind in REPLAY_KINDS for item in items):
                    tally["replays_unjudged" if unjudged else "replays_agreed"] += 1
            looked.update(entry.id for entry in entries
                          if (entry.oracle, entry.record) == (source.name, record.id)
                          and divergences.looked_at(entry, observed, answer, judges_actions=False))
        if complete:
            tally["described"] += sum(described)
            tally["boards"] += sum(boards)
            tally["states"] += sum(states)
            tally["sides"] += sum(sides)
            tally["sides_agreed"] += sum(given and not clash for given, clash in zip(sides, clashed, strict=True))

    told, told_ids, _ = _telltale_findings(entries, engine, records, engine_results, recorded or _fresh(results),
                                           judges_actions=False)
    findings = told + findings
    unexplained, stale = divergences.apply(entries, found, looked, told_ids)
    findings += [Finding("divergence", item.describe()) for item in unexplained]
    findings += _stale_findings(stale)
    agreement = ParseAgreement(records=len(records), **tally)
    # Agreeing with the oracles about nothing is not a pass.
    if not agreement.read:
        findings.append(Finding("engine-nothing-compared",
                                "in parse mode the engine replayed no record of this tier, so nothing was compared"))
    elif not agreement.described and not agreement.replays:
        findings.append(Finding("engine-nothing-compared",
                                f"in parse mode the engine replayed {agreement.read} record(s), but no oracle said what it "
                                "set up at any of their positions, and no implementation's replay of them was compared: "
                                "nothing was compared"))
    return findings, agreement


def _some(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def calibrate(lock: Lock, cfg: config.GateConfig, records: list[Record], reference: golden.Calibration | None) -> Probe:
    """Time the calibration probe ([calibration] in gate.toml) on this machine, alone.

    A fixed, deterministic workload - one search over the first N positions of
    one record - timed by when position N-1's answers arrived. Beside the time
    the recording machine took for it, it says how much slower this machine is,
    so that a required value a slow runner fails to reach can be told from a
    regression: "this runner is k x slower than the recording machine; margin 4".
    """
    spec = cfg.calibration
    # A reference measured for another probe says nothing about this one (verify reports it stale).
    same = reference is not None and (reference.source, reference.record, reference.positions) == (
        spec.source, spec.record, spec.positions)
    probe = Probe(source=spec.source, record=spec.record, positions=spec.positions,
                  reference_s=reference.seconds if same else None,
                  note='' if same or reference is None else 'the recorded reference is for another probe')
    record = next((item for item in records if item.id == spec.record), None)
    if record is None:
        return dataclasses.replace(probe, note=f"record {spec.record} is not in the corpus")
    source = make_source(spec.source, lock)
    reason = source.available()
    if reason:
        return dataclasses.replace(probe, note=reason)
    options = RunOptions(cap=cfg.cap, timeout_s=PROBE_TIMEOUT_S, mate_budget_ms=cfg.recording.mate_budget_ms,
                         stop_after=(spec.positions - 1, "position"))
    observation = runner.observe(source, record.id, corpus.feed(record, lock), options)
    index = spec.positions - 1
    arrived = observation.arrival_s[index] if index < len(observation.arrival_s) else None
    if arrived is None or index >= len(observation.positions) or not observation.positions[index].known():
        return dataclasses.replace(probe, note=(f"it did not reach position {index} ({observation.status}"
                                                f"{'/' + observation.reason if observation.reason else ''})"))
    return dataclasses.replace(probe, seconds=round(arrived, 3))


def run_tier(
    tier_name: str,
    *,
    only_sources: list[str] | None = None,
    only_records: list[str] | None = None,
    jobs: int | None = None,
    include_untracked: bool = True,
    log: Log = print,
) -> GateResult:
    lock = lockfile.load()
    cfg = config.load()
    tier = cfg.tier(tier_name)
    all_records = corpus.load()
    # Everything compared against is read once, here, and verified (H33).
    frozen = snapshot.take()
    # What the tier runs, not what this run ran: a run restricted with --source or --record keeps its tier's rule
    # (H47) and is marked partial below, which keeps the proof channel shut (spec issue 124 in the build log).
    result = GateResult(tier=tier_name, require_engine=tier.require_engine, full_strength=tier.full_strength,
                        engine_mode=cfg.engine_mode, margin=cfg.stability.margin)
    result.sections += static_checks(lock, cfg, all_records, frozen, include_untracked=include_untracked)
    if tier.can_enable_proof_channel:
        # H32: a tier that can enable the proof channel must not require next to nothing.
        readiness = proof_channel_readiness(cfg, all_records, frozen)
        if readiness is not None:  # otherwise the golden files cannot be read, and their section says why
            result.sections.append((PROOF_SECTION, readiness))

    names = list(tier.sources)
    if only_sources:
        unknown = [name for name in only_sources if name not in tier.sources]
        if unknown:
            raise HarnessError(f"tier '{tier_name}' does not run: {', '.join(unknown)}")
        names = [name for name in names if name in only_sources]
    records = corpus.select(all_records, tier.record_tiers, only_records)
    result.partial = len(names) < len(tier.sources) or bool(only_records)
    if not records:
        result.sections.append(("records", [Finding("no-records", f"tier '{tier_name}' selects no corpus records: nothing was checked")]))
    result.converted = {record.id: record.reading for record in records if record.reading}
    for record_id, reading in result.converted.items():
        log(f"{record_id}: {reading}")

    sources: list[Source] = []
    unavailable: list[Finding] = []
    for name in names:
        source = make_source(name, lock)
        reason = source.available()
        if reason:
            unavailable.append(Finding("source-unavailable", f"{name}: {reason}"))
        else:
            sources.append(source)
    result.sections.append(("oracle builds", unavailable))

    # The program the command names is the repository root's on every platform (engine_argv).
    engine = EngineSource(engine_argv(cfg.engine_command), cfg.engine_mode)
    engine_runs = engine.available() is None
    log("timing the calibration probe")
    result.calibration = calibrate(lock, cfg, all_records, frozen.calibration)
    log(f"  {result.calibration.describe()}")

    # What this tier requires of each source on each record (H32). A run that
    # needs nothing more may be stopped once its last required position is in -
    # but never where an engine is compared (every oracle answer may confirm or
    # contradict it) and never in a tier that can enable the proof channel, which
    # runs every search to the recording budget.
    early = not engine_runs and not tier.can_enable_proof_channel
    requirements: dict[tuple[str, str], stability.Requirement] = {}
    for source in sources:
        if not frozen.complete(source.name):
            continue
        for record in records:
            if record.id in frozen.golden[source.name].observations and record.id in frozen.costs[source.name].records:
                requirements[(source.name, record.id)] = stability.requirement(
                    frozen.golden[source.name], frozen.costs[source.name], record.id, tier.timeout_for(source.name),
                    tier.mate_budget_ms, cfg.stability.margin, source.searched)

    def options_for(source: Source, record: Record) -> RunOptions:
        need = requirements.get((source.name, record.id))
        stop = need.stop_after if early and need is not None else None
        return _options(cfg, tier.timeout_for(source.name), tier.mate_budget_ms, stop)

    log(f"running {len(sources)} source(s) over {len(records)} record(s)")
    results = runner.run_sources(sources, records, lock, options_for, jobs=jobs, progress=_progress(log))
    result.sections.append(("source self-consistency", runner.self_consistency(results)))

    golden_findings: list[Finding] = []
    for source in sources:
        if not frozen.complete(source.name):
            reasons = "; ".join(problem.message for problem in frozen.problems.get(source.name, []))
            golden_findings.append(Finding("golden-missing", f"{source.name}: not compared with its recorded answers - "
                                                             f"{reasons or 'they could not be read'}"))
            continue
        found, tally = stability.check_run(source, records, results, frozen.golden[source.name],
                                           frozen.costs[source.name], tier, cfg.stability.margin)
        golden_findings += found
        result.tallies[source.name] = tally
    result.sections.append(("fresh run vs required sets", golden_findings))

    drivers = [source for source in sources if isinstance(source, FtxiSource)]
    if drivers:
        log("checking the ftxi driver against the stock 5dtools CLI")
        fidelity_findings: list[Finding] = []
        for source in drivers:
            # Every position of the main search in a tier that runs everything;
            # otherwise a sample of each record.
            full = tier.full_strength and source.name in PRIMARY_SOURCES
            fidelity_findings += fidelity.check(source, records, results, lock, cfg.cap,
                                                timeout_s=tier.timeout_for(source.name), full=full, jobs=jobs)
        result.sections.append(("ftxi driver vs stock 5dtools", fidelity_findings))

    if engine_runs:
        # The engine's own time limit: a tier's timeout_s is a budget for the oracles - it decides what the tier
        # requires of them (H32) - and the gate compares the engine's answers, not its speed.
        engine_timeout = cfg.engine_timeout(tier)
        result.engine_run = {"command": list(cfg.engine_command), "mode": cfg.engine_mode,
                             "timeout_s": engine_timeout, **engine_program(engine.command)}
        log(f"running the engine adapter ({cfg.engine_mode} mode, {number(engine_timeout)} s a record): "
            f"{_describe_engine_run(result.engine_run)}")
        engine_results = runner.run_sources([engine], records, lock,
                                            lambda source, record: _options(cfg, engine_timeout, tier.mate_budget_ms),
                                            jobs=jobs, progress=_progress(log))
        engine_findings = runner.self_consistency(engine_results)
        # The divergences are the snapshot's, read before the engine ran: an
        # engine cannot excuse itself by writing to the file (H33). And what an
        # oracle's recordings disagreed about is compared with nothing (H32).
        answers = without_unstable(results, frozen)
        if cfg.engine_mode == "parse":
            differential, result.parse = parse_differential(engine, sources, records, answers, engine_results,
                                                            frozen.divergences, frozen.recorded)
            engine_findings += differential
        else:
            differential, result.gated, result.ungated = engine_differential(
                engine, sources, records, answers, engine_results, frozen.divergences, tier.min_checked,
                frozen.recorded)
            engine_findings += differential
            if not engine_findings and not result.gated and tier.full_strength:
                # Agreeing with the oracles about nothing is not a pass - in a tier
                # that runs every oracle search over the whole corpus. Any other
                # tier is a smoke test that gates nothing (H47), and says so. Keyed
                # to what the tier runs, never to its name.
                engine_findings.append(Finding(
                    "engine-nothing-gated",
                    "nothing was found against the engine (no unexplained divergence, telltale or breach of the "
                    "contract), but it came through no ruleset either: "
                    + "; ".join(f"{ruleset} - {reason}" for ruleset, reason in sorted(result.ungated.items()))))
        result.engine_sections.append(("engine vs oracles", engine_findings))
        result.engine = "failed" if engine_findings else "passed"
        results.update(engine_results)
    # Nothing the run compared against or executed may have changed while it ran (H33).
    result.sections.append(("harness unchanged while the gate ran", integrity.changes(frozen.harness, integrity.hash_harness())))
    _write_report(result, results, cfg, tier, sources, jobs, frozen)
    return result


def _write_report(result: GateResult, results: runner.Results, cfg: config.GateConfig | None = None,
                  tier: config.Tier | None = None, sources: list[Source] | None = None, jobs: int | None = None,
                  frozen: Snapshot | None = None) -> Path:
    reports = paths.cache_dir() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    budgets, signatures = {}, {}
    for source in sources or []:
        options = _options(cfg, tier.timeout_for(source.name), tier.mate_budget_ms)
        budgets[source.name] = source.budget(options)
        signatures[source.name] = golden.signature(source, options)
    gated, ungated = result.verdicts
    body = {
        "tier": result.tier,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ok": result.ok,
        "oracle_layer_ok": result.oracle_layer_ok,
        "partial": result.partial,
        "engine": result.engine,
        "engine_mode": result.engine_mode,
        # Which build of the engine this run compared, and under what time limit (null where no engine ran).
        "engine_run": result.engine_run,
        # In a smoke test (H47) nothing is gated, and this says why; otherwise null.
        "smoke_test": SMOKE_TEST if result.smoke_test else None,
        "gated": {ruleset: asdict(item) for ruleset, item in gated.items()},
        "ungated": ungated,
        "parse_agreement": asdict(result.parse) if result.parse is not None else None,
        "proof_channel": result.proof_channel,
        # The records every source read other than as stored (export_form.py).
        "converted": result.converted,
        # What a reader needs to tell a slow runner from a regression (H32).
        "budgets": budgets,
        "margin": result.margin,
        "jobs": jobs or runner.default_jobs(),
        "platform": {"key": platform_key(), "system": platform.platform(), "python": platform.python_version()},
        "cpu_count": os.cpu_count(),
        "signatures": signatures,
        "recorded_signatures": {name: item.signature for name, item in (frozen.golden.items() if frozen else ())},
        "calibration": {**asdict(result.calibration), "factor": result.calibration.factor} if result.calibration else None,
        "required": {name: {**asdict(tally), "unknown": dict(tally.unknown)} for name, tally in result.tallies.items()},
        "findings": [{"section": title, "code": item.code, "message": item.message}
                     for title, items in (*result.sections, *result.engine_sections) for item in items],
        "observations": {f"{source}|{record}": {**observations.to_json(item), "elapsed_s": round(item.elapsed_s, 3)}
                         for (source, record), item in sorted(results.items())},
    }
    # A partial run never stands in for a complete one, not even in the cache.
    path = reports / f"gate-{result.tier}{'-partial' if result.partial else ''}.json"
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    return path


def _summarise_sections(sections: list[Section]) -> list[str]:
    lines = []
    for title, findings in sections:
        lines.append(f"{'FAIL' if findings else ' ok '}  {title}" + (f" ({len(findings)})" if findings else ""))
        lines += [f"        {finding}" for finding in findings[:40]]
        if len(findings) > 40:
            lines.append(f"        ... and {len(findings) - 40} more (see the report in the harness cache)")
    return lines


def summarise(result: GateResult) -> str:
    lines = _summarise_sections(result.sections)
    for name, tally in result.tallies.items():
        lines.append(f"        required {name}: {tally.describe()}")
    if result.calibration is not None:
        lines.append(f"        calibration: {result.calibration.describe()}")
    unreproduced = any(finding.code == "golden-unreproduced" for finding in result.findings())
    if unreproduced and result.calibration is not None and result.calibration.factor is not None:
        factor = result.calibration.factor
        speed = f"{factor:.1f}x slower than" if factor >= 1 else f"faster than ({factor:.1f}x the time of)"
        lines.append(f"        this runner is {speed} the recording machine; margin {number(result.margin)}")
    lines += _summarise_sections(result.engine_sections)
    lines.append("")
    scope = " (PARTIAL RUN - restricted sources or records)" if result.partial else ""
    lines.append(f"oracle layer [{result.tier}]: {'ok' if result.oracle_layer_ok else 'FAILED'}{scope}")
    if result.engine == "not configured":
        lines.append("engine differential: NOT RUN - no engine adapter is configured in oracles/gate.toml")
    elif result.engine_mode == "parse":
        agreement = result.parse.describe() if result.parse is not None else "nothing was compared"
        lines.append(f"engine differential: PARSE MODE - {result.engine.upper()}{scope}: {agreement}")
    else:
        if result.smoke_test:
            # Passed or failed, it never reads as gating (H47).
            lines.append(f"engine differential: SMOKE TEST - {result.engine.upper()}{scope}: {SMOKE_TEST}")
        else:
            lines.append(f"engine differential: {result.engine.upper()}{scope}")
        gated, ungated = result.verdicts
        for described in _describe_gated(gated):
            lines.append(f"  gated:     {described}")
        for ruleset, reason in sorted(ungated.items()):
            lines.append(f"  NOT gated: {ruleset} - {reason}")
    lines.append(f"proof channel: {result.proof_channel}")
    if result.require_engine and result.engine == "not configured":
        lines.append("this tier requires an engine: FAILED")
    elif result.require_engine and result.engine_mode == "parse":
        lines.append("this tier requires an engine compared on the rules, not in parse mode: FAILED")
    elif result.require_engine and result.engine == "passed" and not result.variants:
        # No finding says why: the engine passed the differential, but this run states no named variant gated.
        lines.append("this tier requires an engine that came through a named variant: FAILED - it came through none "
                     "(custom positions are evidence about no variant)")
    return "\n".join(lines)


def _measure_probe(lock: Lock, cfg: config.GateConfig, records: list[Record], log: Log) -> golden.Calibration | None:
    probe = calibrate(lock, cfg, records, None)
    log(f"  calibration probe: {probe.describe()}")
    if probe.seconds is None:
        return None
    spec = cfg.calibration
    return golden.Calibration(source=spec.source, record=spec.record, positions=spec.positions,
                              signature=make_source(spec.source, lock).signature(RunOptions(cap=cfg.cap)),
                              seconds=probe.seconds, platform=f"{platform_key()} ({platform.platform()})",
                              cpu_count=os.cpu_count())


def write_golden(*, only_sources: list[str] | None = None, jobs: int | None = None,
                 repeat: int = golden.MIN_RECORDINGS, log: Log = print) -> list[Finding]:
    """Re-run the oracles over the whole corpus, ``repeat`` times each, and record what they say (H32, H33).

    Recorded under the recording tier's budgets ([stability] recording_tier).
    Every value is classified across the recordings (stability.merge): what
    differed is unstable, what a budget cut is unknown at that budget. Then the
    golden files, their cost files, the calibration reference and MANIFEST are
    written together, and DISAGREEMENTS.md is rendered from what was written.
    Last, it says what the new recording no longer holds that the files it
    replaced did (stability.losses): a verdict or an outcome gone, a run cut
    earlier - each a loss of checking for review, never routine churn.
    """
    if repeat < 1:
        raise HarnessError("--repeat must be at least 1")
    lock = lockfile.load()
    cfg = config.load()
    records = corpus.load()
    blocking = lockfile.verify(lock) + corpus.verify(records, lock)
    if blocking:
        return blocking
    names = only_sources or list(ORACLE_SOURCES)
    sources = [make_source(name, lock) for name in names]
    missing = [Finding("source-unavailable", f"{source.name}: {source.available()}") for source in sources if source.available()]
    if missing:
        return missing

    # A file this run does not rewrite keeps its hash in MANIFEST - once it is
    # shown to match it: re-hashing whatever is on disk would launder an edit.
    try:
        manifest = integrity.read_manifest()
    except integrity.IntegrityError as error:
        return [Finding("golden-integrity", str(error))]
    probe_source = make_source(cfg.calibration.source, lock)
    rewritten = {path.name for source in sources for path in (golden.golden_path(source.name), golden.cost_path(source.name))}
    if probe_source.available() is None:
        rewritten.add(golden.CALIBRATION_FILE)
    kept = [path for path in integrity.recorded_files() if path.name not in rewritten]
    if kept:
        if manifest is None:
            return [Finding("golden-integrity",
                            f"there is no {integrity.MANIFEST} to vouch for {', '.join(path.name for path in kept)}, which "
                            "this run would not rewrite: record every source (golden --write without --source)")]
        tampered = [path.name for path in kept if manifest.get(path.name) != sha256_text(path.read_bytes())]
        if tampered:
            return [Finding("golden-integrity",
                            f"{name} does not match {integrity.MANIFEST}, and this run would not rewrite it: restore it, "
                            "or record its source as well") for name in tampered]

    def options_for(source: Source, record: Record | None = None) -> RunOptions:
        return recording_options(cfg, source)

    previous = {source.name: golden.read_previous(source.name) for source in sources}
    before = {source.name: golden.signature(source, options_for(source)) for source in sources}
    calibration = None
    if golden.CALIBRATION_FILE in rewritten:
        log("timing the calibration probe on this machine")
        calibration = _measure_probe(lock, cfg, records, log)
        if calibration is None:
            return [Finding("calibration", "the calibration probe could not be timed on this machine; nothing was written")]
    log(f"recording {len(sources)} source(s) over {len(records)} record(s), {repeat} time(s) each")
    runs = runner.record_sources(sources, records, lock, options_for, repeat=repeat, jobs=jobs, progress=_progress(log))
    contradictions = [Finding("self-contradiction", f"{run.source} on {run.record}: {problem}")
                      for found in runs.values() for run in found for problem in observations.validate(run)]
    if contradictions:
        return contradictions
    merged: dict[tuple[str, str], golden.Recorded] = {}
    unstable: list[Finding] = []
    for source in sources:
        for record in records:
            outcome = stability.merge(source, options_for(source), runs[(source.name, record.id)])
            if isinstance(outcome, Finding):
                unstable.append(outcome)
            else:
                merged[(source.name, record.id)] = outcome
                for problem in observations.validate(outcome.observation):
                    unstable.append(Finding("golden-invalid", f"{source.name} on {record.id}, merged over its "
                                                              f"recordings: {problem}"))
    if unstable:
        return unstable
    # The signature names the code that produced the answers. Code edited while
    # the oracles ran would be named for answers it did not produce.
    moved = [source.name for source in sources if golden.signature(source, options_for(source)) != before[source.name]]
    if moved:
        return [Finding("golden-code-changed",
                        f"{name}: its adapter or driver changed while it ran - nothing was written; run it again: {golden.RERECORD}")
                for name in moved]
    paths.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {path.name: manifest[path.name] for path in kept} if kept else {}
    written: dict[Path, str] = {}
    for source in sources:
        recorded = {record.id: merged[(source.name, record.id)] for record in records}
        signed = before[source.name]
        written[golden.golden_path(source.name)] = golden.dump(source.name, signed, records, recorded, recordings=repeat)
        written[golden.cost_path(source.name)] = golden.dump_costs(source.name, signed, records, recorded,
                                                                    recordings=repeat)
    if calibration is not None:
        written[golden.calibration_path()] = golden.dump_calibration(calibration)
    # The answers first, then the hashes that vouch for them: an interrupted write
    # leaves files MANIFEST does not match, which verify reports.
    for path, text in written.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        hashes[path.name] = sha256_text(text.encode("utf-8"))
        log(f"wrote {_shown(path)}")
    integrity.write_manifest(hashes)
    log(f"wrote {_shown(integrity.manifest_path())}")
    render_disagreements(log=log)
    lost = [line for source in sources if previous[source.name] is not None
            for line in stability.losses(source.name, previous[source.name],
                                         {record.id: merged[(source.name, record.id)] for record in records})]
    if lost:
        log("")
        log(f"Compared with the golden files it replaced, this recording no longer holds {len(lost)} thing(s). Each "
            "is a loss of checking, not churn: re-record on a quiet machine, or raise the budget - never accept one to "
            "make a gate run pass.")
        for line in lost:
            log(f"  {line}")
    return []


def _shown(path: Path) -> str:
    """A path as the log shows it: from the repository root, where it is inside it."""
    try:
        return path.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def render_disagreements(*, log: Log = print) -> None:
    """Render DISAGREEMENTS.md from the golden and cost files, each read only once verified against MANIFEST,
    and gate.toml's budgets (what each tier requires)."""
    frozen = snapshot.take()
    problems = [problem.message for source in ORACLE_SOURCES for problem in frozen.problems.get(source, [])
                if source not in frozen.golden]
    if problems:
        raise HarnessError(f"{paths.DISAGREEMENTS_FILE.name} was not rendered: {problems[0]}")
    text = _rendered(config.load(), corpus.load(), frozen)
    paths.DISAGREEMENTS_FILE.write_text(text, encoding="utf-8", newline="\n")
    log(f"wrote {_shown(paths.DISAGREEMENTS_FILE)}")
