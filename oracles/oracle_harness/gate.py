"""The gate: static verification, tiered oracle runs, and the engine differential.

Tiers (design spec 11.0):

  commit   every push - the boundary, the pins, the corpus, and the fast oracle
  nightly  the two heavier oracles as well, on the whole corpus
  release  before any generator release or proof-channel enablement - everything
           in nightly, and it fails unless an engine was compared and at least
           one ruleset came through

No proof may enter the solved store until the generator has passed this gate
(decision H23). While there is no engine to compare, the oracle layer can be
green, but the differential gate has not run and the result says so.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import boundary, config, corpus, divergences, fidelity, golden, lockfile, observations, paths, provenance, runner
from .adapters import ORACLE_SOURCES, PRIMARY_SOURCES, EngineSource, RunOptions, Source, make_source, oracle_of
from .adapters.ftxi import FtxiSource
from .corpus import CUSTOM_POSITIONS, Record
from .lockfile import Lock
from .observations import HARD_REJECTIONS, OBSERVABLES, Observation
from .util import Finding, HarnessError

Log = Callable[[str], None]
Section = tuple[str, list[Finding]]

# A report that followed the record to its end, or to the action it refuses.
_COMPLETE = ("ok", "rejected")
_LABELS = {"in_check": "check", "legal_actions": "count", "terminal": "mate"}
# The share of its recorded answers every oracle search must reproduce in a run
# that can enable the proof channel. Deliberately not in gate.toml: a tier can
# list a source and still starve it of time, or set its floor to nothing, and its
# list of sources would say it ran.
PROOF_CHANNEL_COVERAGE = 0.9


@dataclass(frozen=True)
class Gated:
    """What stands behind one ruleset the engine came through."""

    records: int
    positions: int  # positions the engine reported, over all the ruleset's records
    checked: dict[str, int]  # observable -> positions at which some oracle's answer settles it and agrees with the engine
    oracles: int  # the fewest independent implementations that followed any one record as far as the engine

    def describe(self) -> str:
        checked = ", ".join(f"{_LABELS[name]} {self.checked[name]}" for name in OBSERVABLES)
        return (f"{self.records} record{'s' if self.records != 1 else ''}, {self.positions} positions; "
                f"independently answered: {checked}; each record followed as far as the engine went by at least "
                f"{self.oracles} oracle{'s' if self.oracles != 1 else ''}")


def _share(stats: tuple[int, int] | None) -> float:
    return stats[0] / stats[1] if stats and stats[1] else 0.0


@dataclass
class GateResult:
    tier: str
    sections: list[Section] = field(default_factory=list)  # the oracle layer
    engine_sections: list[Section] = field(default_factory=list)  # the engine against it
    coverage: dict[str, tuple[int, int]] = field(default_factory=dict)
    engine: str = "not configured"  # not configured | passed | failed
    gated: dict[str, Gated] = field(default_factory=dict)
    ungated: dict[str, str] = field(default_factory=dict)  # ruleset -> why it is not gated
    partial: bool = False
    require_engine: bool = False
    # The tier runs every oracle search over the whole corpus. A tier's name promises nothing.
    full_strength: bool = False

    def findings(self) -> list[Finding]:
        return [finding for _, items in (*self.sections, *self.engine_sections) for finding in items]

    @property
    def oracle_layer_ok(self) -> bool:
        return not any(items for _, items in self.sections)

    @property
    def variants(self) -> dict[str, Gated]:
        """The gated rulesets that are named variants - what a proof could be about."""
        return {ruleset: gated for ruleset, gated in self.gated.items() if ruleset != CUSTOM_POSITIONS}

    @property
    def ok(self) -> bool:
        if not self.oracle_layer_ok or self.engine == "failed":
            return False
        if self.engine == "passed":
            # An engine that contradicted nothing and came through no ruleset
            # was not compared with anything; a tier that requires an engine
            # requires it to come through a variant a proof could be about.
            return bool(self.variants) if self.require_engine else bool(self.gated)
        return not self.require_engine

    @property
    def proof_channel(self) -> str:
        if self.engine == "not configured":
            return "DISABLED - no engine has been compared against the oracles (H23)"
        if not self.ok:
            return "DISABLED - the gate failed"
        if self.partial or not (self.require_engine and self.full_strength):
            return ("DISABLED - only a complete run of a tier that requires an engine and runs every oracle "
                    "over the whole corpus can enable it")
        starved = [name for name in ORACLE_SOURCES if _share(self.coverage.get(name)) < PROOF_CHANNEL_COVERAGE]
        if starved:
            return (f"DISABLED - {', '.join(starved)} reproduced less than {PROOF_CHANNEL_COVERAGE:.0%} of its recorded "
                    "answers in this run, so it was listed but not heard")
        if not self.variants:
            return "DISABLED - the engine came through no named variant"
        return "may be enabled for: " + "; ".join(_describe_gated(self.variants))


def _describe_gated(gated: dict[str, Gated]) -> list[str]:
    return [f"{ruleset} ({item.describe()})" for ruleset, item in sorted(gated.items())]


def _options(cfg: config.GateConfig, timeout_s: float, mate_budget_ms: int) -> RunOptions:
    return RunOptions(cap=cfg.cap, timeout_s=timeout_s, mate_budget_ms=mate_budget_ms)


def _progress(log: Log) -> Callable[[Observation], None]:
    def report(observation: Observation) -> None:
        log(f"  {observation.source:<12} {observation.record:<40} {observation.status:<11} "
            f"{len(observation.positions):>4} positions  {observation.elapsed_s:6.1f}s")

    return report


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


def static_checks(lock: Lock, cfg: config.GateConfig, records: list[Record], *, include_untracked: bool = True) -> list[Section]:
    """Everything that can be checked without building or running an oracle."""
    sections: list[Section] = [
        ("pins", lockfile.verify(lock)),
        ("corpus", corpus.verify(records, lock) + corpus.verify_local(records)
         + corpus.verify_excluded(corpus.load_excluded(), lock)),
        ("known divergences", divergences.verify(divergences.load(), lock, records, golden.lookup())),
        ("boundary", boundary.check(lock, cfg.boundary, include_untracked=include_untracked)
         + engine_command_findings(cfg.engine_command)),
        ("provenance", provenance.check(lock, records)),
    ]
    sources = [make_source(name, lock) for name in ORACLE_SOURCES]
    golden_findings = golden.verify_static(sources, RunOptions(cap=cfg.cap), records)
    if not golden_findings:
        rendered = golden.render_disagreements(records)
        on_disk = ""
        if paths.DISAGREEMENTS_FILE.exists():
            on_disk = paths.DISAGREEMENTS_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
        if on_disk != rendered:
            golden_findings.append(Finding(
                "disagreements-stale",
                f"{paths.DISAGREEMENTS_FILE.name} does not match the golden files "
                f"(run: python {paths.HARNESS_REL}/harness.py golden --render)"))
    sections.append(("golden observations", golden_findings))
    return sections


def _vouches(answer: Observation, observed: Observation) -> bool:
    """Did this oracle follow the record as far as the engine did, and to the same end?

    Where the engine refuses an action, only an oracle that refuses the same
    action as illegal stands behind it. One that merely stopped reading there
    does not, and neither does one that could not match the move's text: that
    may be notation it does not read, which confirms nothing about the move.
    """
    if not observed.positions:
        return False
    if observed.status == "rejected":
        return (answer.status == "rejected" and answer.reason in HARD_REJECTIONS
                and answer.actions_accepted == observed.actions_accepted)
    return len(answer.positions) >= len(observed.positions)


def engine_differential(
    engine: EngineSource,
    oracle_sources: list[Source],
    records: list[Record],
    results: runner.Results,
    engine_results: runner.Results,
    entries: list[divergences.Divergence],
    min_checked: float = 1.0,
) -> tuple[list[Finding], dict[str, Gated], dict[str, str]]:
    """Compare the engine with every oracle report; returns (findings, gated, ungated).

    A ruleset counts as gated only if, over all of its records:

      * the engine replayed each one, and no unexplained divergence remains;
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
    for record in records:
        observed = engine_results[(engine.name, record.id)]
        if observed.status in ("error", "timeout"):
            findings.append(Finding(
                "engine-failed", f"engine on {record.id}: {observed.status} ({observed.reason}) {observed.detail[:300]}"))
        answers = {source.name: results.get((source.name, record.id)) for source in oracle_sources}
        compared = {}
        for name, answer in answers.items():
            if answer is not None:
                # How a record is replayed belongs to the implementation, not to
                # each of its searches: one entry excuses one such disagreement.
                compared[name] = observations.compare(observed, answer, replay=name in PRIMARY_SOURCES)
                found += compared[name]
        # An implementation that read the record differently from the engine -
        # an action accepted by one and refused by the other, or a record that
        # ends earlier for one of them - does not stand behind the engine's
        # replay of it, even where a known divergence excuses the difference.
        misread = {oracle_of(name) for name, items in compared.items()
                   if any(item.kind in divergences.REPLAY_KINDS for item in items)}
        vouched[record.id] = {oracle_of(name) for name, answer in answers.items()
                              if answer is not None and oracle_of(name) not in misread and _vouches(answer, observed)}
        # A position's question counts as independently answered only where an
        # oracle's answer settles it and agrees with the engine's. An answer the
        # engine contradicts confirms nothing - also when the contradiction is on
        # record as the oracle's error.
        answered = [dict.fromkeys(OBSERVABLES, False) for _ in observed.positions]
        for name, answer in answers.items():
            if answer is None:
                continue
            clashes = {(item.kind, item.index) for item in compared[name]}
            for slot, theirs in zip(answered, answer.positions, strict=False):
                for observable in OBSERVABLES:
                    agrees = (observable, theirs.index) not in clashes
                    slot[observable] = slot[observable] or (agrees and observations.decisive(observable, getattr(theirs, observable)))
            looked.update(
                entry.id for entry in entries
                if (entry.oracle, entry.record) == (name, record.id)
                and divergences.looked_at(entry, observed, answer))
        reported[record.id] = len(observed.positions)
        settled[record.id] = {observable: sum(slot[observable] for slot in answered) for observable in OBSERVABLES}

    unexplained, stale = divergences.apply(entries, found, looked)
    for item in unexplained:
        findings.append(Finding("divergence", item.describe()))
    for entry in stale:
        findings.append(Finding(
            "divergence-stale",
            f"{entry.id}: no longer occurs ({entry.oracle} on {entry.record}, {entry.kind} at {entry.index}) - "
            "remove it, or find out why"))

    diverged = {item.record for item in unexplained}
    rulesets: dict[str, list[Record]] = {}
    for record in records:
        rulesets.setdefault(record.ruleset, []).append(record)
    gated: dict[str, Gated] = {}
    ungated: dict[str, str] = {}
    for ruleset, items in sorted(rulesets.items()):
        unplayed = [item.id for item in items if engine_results[(engine.name, item.id)].status not in _COMPLETE]
        alone = [item.id for item in items if not vouched[item.id]]
        positions = sum(reported[item.id] for item in items)
        checked = {name: sum(settled[item.id][name] for item in items) for name in OBSERVABLES}
        thin = [name for name in OBSERVABLES if checked[name] < min_checked * positions]
        if any(item.id in diverged for item in items):
            ungated[ruleset] = "unexplained divergence"
        elif unplayed:
            ungated[ruleset] = f"the engine did not replay {_some(unplayed)}"
        elif alone:
            ungated[ruleset] = f"no oracle followed {_some(alone)} as far as the engine went"
        elif thin:
            name = thin[0]
            ungated[ruleset] = (f"the oracles give a {_LABELS[name]} verdict at only {checked[name]} of the engine's "
                                f"{positions} positions (the floor is {min_checked:.0%})")
        else:
            gated[ruleset] = Gated(records=len(items), positions=positions, checked=checked,
                                   oracles=min(len(vouched[item.id]) for item in items))
    return findings, gated, ungated


def _some(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


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
    result = GateResult(tier=tier_name, require_engine=tier.require_engine, full_strength=tier.full_strength)
    result.sections += static_checks(lock, cfg, all_records, include_untracked=include_untracked)

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

    def options_for(source: Source) -> RunOptions:
        return _options(cfg, tier.timeout_for(source.name), tier.mate_budget_ms)

    log(f"running {len(sources)} source(s) over {len(records)} record(s)")
    results = runner.run_sources(sources, records, lock, options_for, jobs=jobs, progress=_progress(log))
    result.sections.append(("source self-consistency", runner.self_consistency(results)))

    golden_findings: list[Finding] = []
    for source in sources:
        try:
            found, stats = golden.check_run(source.name, records, results, tier.min_coverage_for(source.name))
        except HarnessError as error:
            found, stats = [Finding("golden-missing", str(error))], (0, 0)
        golden_findings += found
        result.coverage[source.name] = stats
    result.sections.append(("fresh run vs golden observations", golden_findings))

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

    engine = EngineSource(list(cfg.engine_command))
    if engine.available() is None:
        log("running the engine adapter")
        engine_results = runner.run_sources([engine], records, lock, options_for, jobs=jobs, progress=_progress(log))
        engine_findings = runner.self_consistency(engine_results)
        differential, result.gated, result.ungated = engine_differential(
            engine, sources, records, results, engine_results, divergences.load(), tier.min_checked)
        engine_findings += differential
        if not engine_findings and not result.gated:
            # Agreeing with the oracles about nothing is not a pass.
            engine_findings.append(Finding(
                "engine-nothing-gated",
                "the engine contradicted no oracle, but it came through no ruleset either: "
                + "; ".join(f"{ruleset} - {reason}" for ruleset, reason in sorted(result.ungated.items()))))
        result.engine_sections.append(("engine vs oracles", engine_findings))
        result.engine = "failed" if engine_findings else "passed"
        results.update(engine_results)
    _write_report(result, results)
    return result


def _write_report(result: GateResult, results: runner.Results) -> Path:
    reports = paths.cache_dir() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    body = {
        "tier": result.tier,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ok": result.ok,
        "oracle_layer_ok": result.oracle_layer_ok,
        "partial": result.partial,
        "engine": result.engine,
        "gated": {ruleset: asdict(item) for ruleset, item in result.gated.items()},
        "ungated": result.ungated,
        "proof_channel": result.proof_channel,
        "coverage": {name: list(stats) for name, stats in result.coverage.items()},
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
    for name, (reproduced, total) in result.coverage.items():
        share = f"{reproduced / total:.1%}" if total else "n/a"
        lines.append(f"        coverage {name}: {reproduced}/{total} recorded values reproduced ({share})")
    lines += _summarise_sections(result.engine_sections)
    lines.append("")
    scope = " (PARTIAL RUN - restricted sources or records)" if result.partial else ""
    lines.append(f"oracle layer [{result.tier}]: {'ok' if result.oracle_layer_ok else 'FAILED'}{scope}")
    if result.engine == "not configured":
        lines.append("engine differential: NOT RUN - no engine adapter is configured in oracles/gate.toml")
    else:
        lines.append(f"engine differential: {result.engine.upper()}{scope}")
        for described in _describe_gated(result.gated):
            lines.append(f"  gated:     {described}")
        for ruleset, reason in sorted(result.ungated.items()):
            lines.append(f"  NOT gated: {ruleset} - {reason}")
    lines.append(f"proof channel: {result.proof_channel}")
    if result.require_engine and result.engine == "not configured":
        lines.append("this tier requires an engine: FAILED")
    return "\n".join(lines)


def write_golden(*, only_sources: list[str] | None = None, jobs: int | None = None, log: Log = print) -> list[Finding]:
    """Re-run the oracles over the whole corpus and record what they say."""
    lock = lockfile.load()
    cfg = config.load()
    records = corpus.load()
    blocking = lockfile.verify(lock) + corpus.verify(records, lock)
    if blocking:
        return blocking
    # Recorded under the release tier's budgets: the longest ones, so as few
    # answers as possible are lost to time-outs.
    budget = cfg.tier("release")
    names = only_sources or list(ORACLE_SOURCES)
    sources = [make_source(name, lock) for name in names]
    missing = [Finding("source-unavailable", f"{source.name}: {source.available()}") for source in sources if source.available()]
    if missing:
        return missing

    def options_for(source: Source) -> RunOptions:
        return _options(cfg, budget.timeout_for(source.name), budget.mate_budget_ms)

    before = {source.name: source.signature(options_for(source)) for source in sources}
    results = runner.run_sources(sources, records, lock, options_for, jobs=jobs, progress=_progress(log))
    contradictions = runner.self_consistency(results)
    if contradictions:
        return contradictions
    # The signature names the code that produced the answers. Code edited while
    # the oracles ran would be named for answers it did not produce.
    moved = [source.name for source in sources if source.signature(options_for(source)) != before[source.name]]
    if moved:
        return [Finding("golden-code-changed",
                        f"{name}: its adapter or driver changed while it ran - nothing was written; run golden --write again")
                for name in moved]
    paths.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        text = golden.dump(source, options_for(source), records, results)
        golden.golden_path(source.name).write_text(text, encoding="utf-8", newline="\n")
        log(f"wrote {golden.golden_path(source.name).relative_to(paths.REPO_ROOT).as_posix()}")
    render_disagreements(log=log)
    return []


def render_disagreements(*, log: Log = print) -> None:
    text = golden.render_disagreements(corpus.load())
    paths.DISAGREEMENTS_FILE.write_text(text, encoding="utf-8", newline="\n")
    log(f"wrote {paths.DISAGREEMENTS_FILE.relative_to(paths.REPO_ROOT).as_posix()}")
