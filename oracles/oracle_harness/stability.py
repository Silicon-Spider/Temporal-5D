"""Stable answers only (decision H32; design spec 11.0, "Running the gate").

An oracle that does not reproduce its own recorded answers is nondeterministic
or limited by its search budget. Budget-limited answers are recorded as "unknown
at budget X"; nondeterministic ones are marked unstable and excluded; and the
gate then requires 100% of the stable recorded answers each tier can reach - a
tolerance would hide regressions inside itself.

Recording (golden --write --repeat N) runs every source N times, in independent
processes, and merge() classifies every value:

  * the same in every recording: recorded;
  * absent from a recording because a limit stopped it there - the run's deadline
    while it searched that position, or a limit inside the run (Chess in 5D's mate
    search) - and the same wherever present: unknown at that budget, recorded as
    the run's horizon or as an "unknown" entry, and never compared;
  * different between recordings, or absent for no reason a budget gives:
    unstable, recorded with the values seen, and never compared.

The cost of each recorded position is when its answers arrived, from the start of
the run, the most over the recordings (golden.RecordCost). A recorded value is
REQUIRED in a tier when it is recorded (stable, not unknown) and its cost times
the margin ([stability] margin in gate.toml) fits the tier's budget for that
source - and, for a Chess in 5D mate verdict, when its mate search's cost times
the margin fits the tier's mate budget. The margin is at least 1, so no tier can
require an answer that arrived after its own deadline on the recording machine:
a required value the tier's budget cannot reach is impossible by construction.

A gate run must then reproduce every required value of its tier, identically;
any other value it produces must agree with the recording; and a recorded value
it does not produce outside the required set is excused only where a budget
explains it - the run never got there, was cut while it searched there, or ran
into a limit there - and counted, by the budget that put it outside the set,
never turned into a ratio. Any other value that goes missing is a finding: a
run that reached a position and answered it, and no longer gives an answer it
used to, has lost it to no budget.

The tier that can enable the proof channel is held to more (proof_channel_findings):
its required set must hold every terminal verdict, and the reviewed share of
what each source would have recorded but for its recording budget - a share
whose denominator a starved budget cannot shrink, because the positions behind
a horizon count in it.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import dataclass, field, replace

from . import corpus, observations
from .adapters import RunOptions, Source, oracle_of
from .adapters.base import number
from .config import GateConfig, Tier
from .corpus import Record
from .golden import Costs, Golden, Horizon, RecordCost, Recorded, round_up, row_value
from .observations import ANSWERS, ROW_FIELDS, Observation, Position
from .util import Finding

# A terminal verdict: the game is over, and the source says so. Rare, and what the
# gate is for: the tier that can enable the proof channel must require every one.
TERMINAL_VERDICTS = ("checkmate", "stalemate", "no_action")


# --- recording ---------------------------------------------------------------------


def _said(run: Observation) -> str:
    return f"{run.status}{'/' + run.reason if run.reason else ''} after {run.actions_accepted} action(s)"


def _explained(source: Source, run: Observation, index: int, name: str, said: object, limit: str | None) -> bool:
    """Is this recording's value here present - or its absence a budget's doing: a limit, or its deadline while it searched here?"""
    cut_here = observations.cut_short(run) and index == len(run.positions) - 1 and name in source.searched
    return said is not None or limit is not None or cut_here


def merge(source: Source, options: RunOptions, runs: list[Observation]) -> Recorded | Finding:
    """One source's independent recordings of one record, merged and classified - or why they cannot be.

    ``runs`` are the recordings, all made with ``options``. The outcome is the
    one thing that is never marked unstable: recordings that ended differently,
    not because one of them ran out of time, replay the record nondeterministically,
    and every value of the record is then in doubt - that is a finding for a
    person, and nothing is written.
    """
    record_id = runs[0].record
    deadline = f"timeout_s={number(options.timeout_s)}"
    cut = [run for run in runs if observations.cut_short(run)]
    finished = [run for run in runs if not observations.cut_short(run)]
    outcomes = {(run.status, run.reason, run.actions_accepted, run.last_action) for run in finished}
    if len(outcomes) > 1:
        return Finding("golden-unstable-outcome", (
            f"{source.name} on {record_id}: its recordings ended differently ("
            f"{'; '.join(sorted({_said(run) for run in finished}))}), and not because one ran out of time - it replays "
            "this record nondeterministically: investigate before recording it"))
    if finished and cut and max(len(run.positions) for run in cut) > len(finished[0].positions):
        return Finding("golden-unstable-outcome", (
            f"{source.name} on {record_id}: a recording cut at its deadline got further than one that ended "
            f"({_said(finished[0])}) - it replays this record nondeterministically: investigate before recording it"))

    length = min(len(run.positions) for run in runs)
    horizon = Horizon(length - 1 if length else None, deadline) if cut else None
    at_horizon = horizon.position if horizon else None
    rows: list[Position] = []
    unknown: dict[int, dict[str, str]] = {}
    unstable: dict[int, dict[str, list]] = {}
    for index in range(length):
        merged: dict[str, object] = {}
        for name in ROW_FIELDS:
            seen = [observations.value(run, index, name) for run in runs]
            # Only an answer can be unknown at a limit; what a source set up it says before any search.
            limits = [source.limits(run, index, options).get(name) if name in ANSWERS else None for run in runs]
            present = [item for item in seen if item is not None]
            merged[name] = None
            named = sorted({limit for limit in limits if limit})
            if not present:
                # Nobody said it: a limit's doing, or a question the source never answers (a bare null).
                if named and not (index == at_horizon and name in source.searched):
                    unknown.setdefault(index, {})[name] = ", ".join(named)
                continue
            distinct = {json.dumps(row_value(name, item)) for item in present}
            silent = [not _explained(source, run, index, name, said, limit)
                      for run, said, limit in zip(runs, seen, limits, strict=True)]
            if len(distinct) == 1 and not any(silent):
                if len(present) == len(runs):
                    merged[name] = present[0]
                elif named and not (index == at_horizon and name in source.searched):
                    unknown.setdefault(index, {})[name] = ", ".join(named)
                continue
            # The recordings differ here, or one is silent for no reason a budget gives.
            values = distinct | ({"null"} if any(silent) else set())
            unstable.setdefault(index, {})[name] = [json.loads(text) for text in sorted(values)]
        rows.append(Position(index, **merged))

    def arrived(run: Observation, index: int) -> float | None:
        return run.arrival_s[index] if index < len(run.arrival_s) else None

    fallback = max(run.elapsed_s for run in runs)
    # A position is reported only after every position before it - cwmtt's, for one,
    # exists only once its CLI has counted every earlier one, whenever its state
    # driver printed what was set up there - so its cost is at least theirs.
    costs = tuple(itertools.accumulate(
        (round_up(max((at for run in runs if (at := arrived(run, index)) is not None), default=fallback))
         for index in range(length)), max))
    end = None if cut else round_up(max(run.ended_s if run.ended_s is not None else run.elapsed_s for run in finished))
    mate: tuple[float | None, ...] = ()
    if "mate_budget_ms" in source.budget(options):
        spent = [[run.extras.get(index, {}).get("ms") for run in runs] for index in range(length)]
        mate = tuple(round_up(max(ms) / 1000) if (ms := [item for item in items if isinstance(item, (int, float))
                                                          and not isinstance(item, bool)]) else None
                     for items in spent)
    if cut:
        status, reason, accepted, last_action = "timeout", "timeout", max(length - 1, 0), None
    else:
        status, reason, accepted, last_action = next(iter(outcomes))
    observation = Observation(source=source.name, record=record_id, status=status, positions=rows,
                              actions_accepted=accepted, reason=reason, last_action=last_action, cap=runs[0].cap)
    return Recorded(observation=observation, cost=RecordCost(positions=costs, end=end, mate=mate), horizon=horizon,
                    unknown=unknown, unstable=unstable)


def losses(source: str, previous: dict[str, Observation], recorded: dict[str, Recorded]) -> list[str]:
    """What a new recording no longer holds that the golden file it replaces did, one line each.

    A terminal verdict gone - behind the new recording's horizon, unknown at a
    limit, unstable - an outcome gone because the run was cut, and a run cut at
    an earlier position than before. A lower budget, or a recording made on a
    loaded machine, shows here first; review it as a loss of checking.
    """
    found = []
    for record_id in sorted(previous.keys() & recorded.keys()):
        old, item = previous[record_id], recorded[record_id]
        new = item.observation
        for position in old.positions:
            if position.terminal not in TERMINAL_VERDICTS or observations.value(new, position.index, "terminal"):
                continue
            if "terminal" in item.unstable.get(position.index, {}):
                why = "unstable between the recordings"
            elif "terminal" in item.unknown.get(position.index, {}):
                why = f"unknown at {item.unknown[position.index]['terminal']}"
            elif item.horizon is not None and position.index >= (item.horizon.position or 0):
                why = f"the recording was {item.horizon.describe()}"
            else:
                why = "not given"
            found.append(f"{source} on {record_id}: its {position.terminal} verdict at position {position.index} is "
                         f"gone ({why})")
        if not observations.cut_short(old) and observations.cut_short(new):
            horizon = item.horizon.describe() if item.horizon is not None else "cut"
            found.append(f"{source} on {record_id}: its outcome ({_said(old)}) is gone - the recording was {horizon}")
        elif observations.cut_short(old) and observations.cut_short(new) and len(new.positions) < len(old.positions):
            found.append(f"{source} on {record_id}: cut at position {len(new.positions) - 1}, where the file it replaces "
                         f"was cut at position {len(old.positions) - 1}")
    return found


# --- what a tier requires ---------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """What one tier requires of one source's fresh run on one record."""

    values: frozenset[tuple[int, str]]  # (position, field): must be reproduced, identically
    outcome: bool  # the run's outcome (status, reason, actions accepted) must be reproduced
    cost: dict[int, float]  # each recorded position's cost: the evidence
    # Every recorded value outside the set -> why: "unknown at budget timeout_s=120", ...
    excluded: dict[tuple[int, str], str]
    outcome_excluded: str | None  # why the outcome is outside the set; None when it is required or there is none
    # Where a run that needs nothing else may be stopped: the event that completes
    # the last position holding a required value. None: it runs to its end.
    stop_after: tuple[int, str] | None
    horizon: Horizon | None

    @property
    def last(self) -> int | None:
        return max((index for index, _ in self.values), default=None)


def requirement(recorded: Golden, costs: Costs, record_id: str, timeout_s: float, mate_budget_ms: int | None,
                margin: float, searched: tuple[str, ...] = ANSWERS) -> Requirement:
    """The required set of one record (module docstring), and why everything else recorded is outside it.

    ``searched`` are the fields a run has only once it has searched a position
    (they arrive with its position event); the rest arrive before, with its state.
    """
    observation = recorded.observations[record_id]
    cost = costs.records[record_id]
    budget = f"unknown at budget timeout_s={number(timeout_s)}"
    mated = f"unknown at budget mate_budget_ms={number(mate_budget_ms)}" if mate_budget_ms is not None else budget
    values: set[tuple[int, str]] = set()
    excluded: dict[tuple[int, str], str] = {}
    spent: dict[int, float] = {}
    so_far = 0.0  # a position arrives after every position before it: its cost is at least theirs
    for position in observation.positions:
        index = position.index
        at = cost.positions[index] if index < len(cost.positions) else None
        if at is not None:
            so_far = max(so_far, at)
            at = spent[index] = so_far
        for name in ROW_FIELDS:
            if getattr(position, name) is None:
                continue
            if at is None or at * margin > timeout_s:
                excluded[(index, name)] = budget
                continue
            if name == "terminal" and cost.mate and mate_budget_ms is not None:
                mate = cost.mate[index] if index < len(cost.mate) else None
                if mate is None or mate * margin * 1000 > mate_budget_ms:
                    excluded[(index, name)] = mated
                    continue
            values.add((index, name))
    ended = observation.status != "timeout"
    outcome = ended and cost.end is not None and cost.end * margin <= timeout_s
    stop_after = None
    if values and not outcome:
        last = max(index for index, _ in values)
        stop_after = (last, "position" if any((last, name) in values for name in searched) else "state")
    return Requirement(values=frozenset(values), outcome=outcome, cost=spent, excluded=excluded,
                       outcome_excluded=budget if ended and not outcome else None, stop_after=stop_after,
                       horizon=recorded.horizon.get(record_id))


@dataclass
class Summary:
    """A required set over a tier's records, and why the rest of what is recorded - or would be - is outside it."""

    values: int = 0
    outcomes: int = 0
    stable: int = 0  # recorded values (stable and known): the required ones and the rest
    recorded_outcomes: int = 0  # records whose outcome is recorded: the run was not cut at the recording budget
    excluded: Counter = field(default_factory=Counter)  # recorded values outside the set, by budget
    outcomes_excluded: Counter = field(default_factory=Counter)
    horizons: Counter = field(default_factory=Counter)  # records cut at the recording budget, by budget
    # Values behind those horizons, by budget: what a recording the budget had not cut would hold and this one
    # does not - the positions a complete replay of the record reaches, at as many values as the source gives.
    behind: Counter = field(default_factory=Counter)
    limited: Counter = field(default_factory=Counter)  # values a limit left unknown in the recording, by limit
    unstable: int = 0
    terminals: int = 0  # terminal verdicts recorded
    terminals_required: int = 0
    left_out: list[tuple[str, int, str, float]] = field(default_factory=list)  # recorded terminal verdicts outside the set
    # Terminal verdicts another source recorded where this source's recording was cut: (record, position,
    # verdict, the source that recorded it, this source's horizon). Not recorded here, so not required either.
    cut_off: list[tuple[str, int, str, str, Horizon]] = field(default_factory=list)

    @property
    def answerable(self) -> int:
        """What the source would have recorded but for its recording budget, its limits and its instability.

        The denominator of the proof channel's share: lowering the recording
        budget moves values behind a horizon, not out of this count.
        """
        return self.stable + sum(self.behind.values()) + sum(self.limited.values()) + self.unstable

    def describe(self) -> str:
        parts = [f"{self.values} values and {self.outcomes} outcomes required (of {self.stable} recorded values and "
                 f"{self.recorded_outcomes} recorded outcomes)"]
        for label in sorted(set(self.excluded) | set(self.outcomes_excluded)):
            outcomes = self.outcomes_excluded.get(label, 0)
            parts.append(f"{label}: {self.excluded.get(label, 0)} values"
                         + (f", {outcomes} outcome{'s' if outcomes != 1 else ''}" if outcomes else "")
                         + " (recorded cost x margin above it)")
        for budget, count in sorted(self.horizons.items()):
            parts.append(f"cut at the recording budget {budget}: {count} record{'s' if count != 1 else ''}, about "
                         f"{self.behind.get(budget, 0)} values behind {'them' if count != 1 else 'it'} (everything from "
                         "the position it was searching on is unknown)")
        for limit, count in sorted(self.limited.items()):
            parts.append(f"unknown at the recording budget {limit}: {count} values")
        if self.unstable:
            parts.append(f"unstable (differed between recordings): {self.unstable} values")
        if self.terminals or self.cut_off:
            parts.append(f"{self.terminals_required} of {self.terminals} recorded terminal verdicts required"
                         + (f", {len(self.cut_off)} more cut off by the recording budget" if self.cut_off else ""))
        return "; ".join(parts)


def _width(recorded: Golden) -> int:
    """The most values the source recorded at any one position: what it gives at a position it answers in full."""
    return max((sum(getattr(position, name) is not None for name in ROW_FIELDS)
                for observation in recorded.observations.values() for position in observation.positions), default=0)


def _replay_length(source: str, record_id: str, loaded: dict[str, Golden]) -> int:
    """How many positions a complete replay of the record holds: another search of the source's implementation
    that was not cut, or failing that any other source that was not (the most of them - an estimate that errs high)."""
    same, other = [], []
    for name, golden in loaded.items():
        report = golden.observations.get(record_id) if name != source else None
        if report is None or observations.cut_short(report) or not report.positions:
            continue
        (same if oracle_of(name) == oracle_of(source) else other).append(len(report.positions))
    return max(same or other or [0])


def _behind(observation: Observation, width: int, replay_length: int) -> int:
    """About how many values a recording cut at its horizon does not hold that an uncut one would."""
    if not observation.positions:
        return width * replay_length
    there = sum(getattr(observation.positions[-1], name) is not None for name in ROW_FIELDS)
    return width * max(replay_length - len(observation.positions), 0) + max(width - there, 0)


def _cut_off(source: str, record_id: str, observation: Observation, horizon: Horizon,
             loaded: dict[str, Golden]) -> list[tuple[str, int, str, str, Horizon]]:
    """The terminal verdicts other sources recorded from where this source's recording was cut.

    Only from sources that set up what this one did up to that point: one that
    read the record otherwise answers about other positions. Each is named by
    the source that recorded it - another search of this source's own
    implementation first, where one did.
    """
    start = horizon.position or 0
    found: dict[int, tuple[str, int, str, str, Horizon]] = {}
    for name in sorted(loaded, key=lambda item: (oracle_of(item) != oracle_of(source), item)):
        report = loaded[name].observations.get(record_id) if name != source else None
        if report is None:
            continue
        parted = observations.first_placement_divergence(observation, report)
        if parted is not None and parted <= start:
            continue
        for position in report.positions[start:]:
            if position.terminal in TERMINAL_VERDICTS and position.index not in found:
                found[position.index] = (record_id, position.index, position.terminal, name, horizon)
    return [found[index] for index in sorted(found)]


def summarise(recorded: Golden, costs: Costs, records: list[Record], tier: Tier, source: str, margin: float,
              loaded: dict[str, Golden] | None = None) -> Summary:
    """One source's required set in one tier. ``loaded`` holds every source's golden file, for what a cut
    recording does not hold (the length of a complete replay, terminal verdicts others recorded behind its horizon)."""
    summary = Summary()
    loaded = loaded if loaded is not None else {source: recorded}
    width = _width(recorded)
    for record in records:
        if record.id not in recorded.observations or record.id not in costs.records:
            continue
        need = requirement(recorded, costs, record.id, tier.timeout_for(source), tier.mate_budget_ms, margin)
        observation = recorded.observations[record.id]
        summary.values += len(need.values)
        summary.outcomes += need.outcome
        summary.recorded_outcomes += observation.status != "timeout"
        summary.stable += len(observations.values(observation))
        summary.excluded.update(need.excluded.values())
        if need.outcome_excluded:
            summary.outcomes_excluded[need.outcome_excluded] += 1
        if need.horizon is not None:
            summary.horizons[need.horizon.budget] += 1
            summary.behind[need.horizon.budget] += _behind(observation, width,
                                                           _replay_length(source, record.id, loaded))
            summary.cut_off += _cut_off(source, record.id, observation, need.horizon, loaded)
        for fields in recorded.unknown.get(record.id, {}).values():
            summary.limited.update(fields.values())
        summary.unstable += sum(len(fields) for fields in recorded.unstable.get(record.id, {}).values())
        for position in observation.positions:
            if position.terminal in TERMINAL_VERDICTS:
                summary.terminals += 1
                if (position.index, "terminal") in need.values:
                    summary.terminals_required += 1
                else:
                    summary.left_out.append((record.id, position.index, position.terminal,
                                             need.cost.get(position.index, 0.0)))
    return summary


def _budget_text(tier: Tier, source: str, mate_budget: bool) -> str:
    text = f"timeout_s={number(tier.timeout_for(source))}"
    return f"{text}, mate_budget_ms={tier.mate_budget_ms}" if mate_budget else text


def describe(cfg: GateConfig, records: list[Record], snapshot) -> list[str]:
    """What `verify` prints: per tier and source, how many values are required, and why the rest are not."""
    lines = [f"required sets (H32): a recorded value is required in a tier when its recorded cost x "
             f"{number(cfg.stability.margin)} fits the tier's budget; recorded under the "
             f"'{cfg.stability.recording_tier}' tier's budgets"]
    for tier in cfg.tiers.values():
        chosen = corpus.select(records, tier.record_tiers)
        for source in tier.sources:
            if not snapshot.complete(source):
                lines.append(f"  {tier.name:<8} {source:<12} not computed: its golden or cost file is not usable "
                             "(see the findings above)")
                continue
            summary = summarise(snapshot.golden[source], snapshot.costs[source], chosen, tier, source,
                                cfg.stability.margin, snapshot.golden)
            lines.append(f"  {tier.name:<8} {source:<12} {summary.describe()}")
    return lines


def required_table(cfg: GateConfig, records: list[Record], loaded: dict[str, Golden],
                   costs: dict[str, Costs]) -> list[str]:
    """The section of DISAGREEMENTS.md that says what each tier requires of each source (H32).

    It is rendered from the golden files, the cost files and gate.toml, and
    `verify` re-renders it: what a tier requires cannot shrink - a budget
    lowered, a cost file edited, a recording made on a loaded machine - without
    a line of this table changing in review.
    """
    lines = [
        "## What each tier requires",
        "",
        "Decision H32. A recorded value is required in a tier when its recorded cost - when it arrived, from the",
        f"start of its run on the recording machine - times {number(cfg.stability.margin)} (`[stability] margin` in "
        "`gate.toml`) fits the",
        "tier's budget for that source (and a Chess in 5D mate verdict's mate search, its mate budget); an outcome",
        "likewise. A run of the tier fails unless it reproduces every required value, identically. A changed line",
        "here is a change in what the gate checks - review it like a changed verdict, and never lower a budget, or",
        "re-record on a loaded machine, to make a run pass.",
        "",
        "| Tier | Source | Budget | Required values | Recorded values | Required outcomes | Recorded outcomes "
        "| Required terminal verdicts | Recorded terminal verdicts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tier in cfg.tiers.values():
        chosen = corpus.select(records, tier.record_tiers)
        for source in tier.sources:
            if source not in loaded or source not in costs:
                continue
            summary = summarise(loaded[source], costs[source], chosen, tier, source, cfg.stability.margin, loaded)
            budget = _budget_text(tier, source, "mate_budget_ms" in loaded[source].signature.get("budget", {}))
            lines.append(f"| {tier.name} | `{source}` | {budget} | {summary.values} | {summary.stable} "
                         f"| {summary.outcomes} | {summary.recorded_outcomes} | {summary.terminals_required} "
                         f"| {summary.terminals} |")
    return lines


def proof_channel_findings(cfg: GateConfig, records: list[Record], snapshot) -> list[Finding]:
    """The static non-vacuity check of every tier that can enable the proof channel.

    Such a tier requires 100% of each source's required set, so the set must
    not be empty in all but name. It must hold every terminal verdict the
    source recorded - and every one another source recorded where this one's
    recording was cut at its budget - and at least the reviewed share ([stability]
    proof_share) of what the source would have recorded but for its recording
    budget: its stable values, the values behind its horizons (the positions a
    complete replay of the record reaches), the values a limit left unknown and
    the unstable ones. The recording budget is this tier's, so starving a
    source - a low budget, or a loaded machine - moves values behind horizons,
    and the share falls instead of rising. Otherwise a source could be listed in
    the tier and starved of time, and the tier would pass on the strength of
    answers nobody was asked for.
    """
    findings = []
    for tier in cfg.tiers.values():
        if not tier.can_enable_proof_channel:
            continue
        chosen = corpus.select(records, tier.record_tiers)
        for source in tier.sources:
            if not snapshot.complete(source):
                continue  # golden-missing, golden-stale or golden-integrity says why
            summary = summarise(snapshot.golden[source], snapshot.costs[source], chosen, tier, source,
                                cfg.stability.margin, snapshot.golden)
            for record_id, index, verdict, cost in summary.left_out:
                findings.append(Finding(
                    "proof-channel-vacuous",
                    f"tier '{tier.name}' does not require {source}'s {verdict} verdict at {record_id} position {index} "
                    f"(recorded cost {cost} s; it needs cost x {number(cfg.stability.margin)} within "
                    f"timeout_s={number(tier.timeout_for(source))}, and its mate search within "
                    f"mate_budget_ms={tier.mate_budget_ms} where it has one): a tier that can enable the proof channel "
                    "requires every terminal verdict - raise its budget"))
            for record_id, index, verdict, other, horizon in summary.cut_off:
                findings.append(Finding(
                    "proof-channel-vacuous",
                    f"tier '{tier.name}' does not require {source}'s verdict at {record_id} position {index}: its "
                    f"recording was {horizon.describe()}, and {other} recorded {verdict} there - a tier that can enable "
                    "the proof channel requires every terminal verdict: raise its budget (the recording budget is this "
                    "tier's, so that means re-recording)"))
            answerable = summary.answerable
            if not answerable:
                findings.append(Finding(
                    "proof-channel-vacuous",
                    f"tier '{tier.name}' lists {source}, which recorded nothing on the tier's records: a tier that can "
                    "enable the proof channel cannot require nothing of a source it runs"))
            elif summary.values < cfg.stability.proof_share * answerable:
                findings.append(Finding(
                    "proof-channel-vacuous",
                    f"tier '{tier.name}' requires {summary.values} of the {answerable} values {source} would record on "
                    f"its records but for its budgets ({summary.stable} recorded, about {sum(summary.behind.values())} "
                    f"behind horizons, {sum(summary.limited.values())} unknown at a limit, {summary.unstable} unstable): "
                    f"{summary.values / answerable:.1%}, below the reviewed share of "
                    f"{cfg.stability.proof_share:.0%} in gate.toml [stability] proof_share - raise its budget"))
    return findings


# --- a gate run against its required set -------------------------------------------


@dataclass
class Tally:
    """How one source's fresh runs in a gate tier compared with their required sets."""

    values: int = 0  # required values
    missing: int = 0  # required values the runs did not produce
    contradicted: int = 0  # required values the runs produced, with another value
    outcomes: int = 0
    outcomes_missing: int = 0  # required outcomes not reproduced: the run was cut short, or ended otherwise
    outside: int = 0  # recorded values outside the required sets that the runs produced too (each compared)
    # Recorded values outside the sets the runs did not produce where a budget explains it, by that budget.
    unknown: Counter = field(default_factory=Counter)
    lost: int = 0  # recorded values outside the sets the runs did not produce where no budget explains it
    stopped: int = 0  # runs stopped once everything their tier requires had arrived

    def describe(self) -> str:
        text = f"{self.values - self.missing - self.contradicted} of {self.values} required values reproduced"
        if self.missing or self.contradicted:
            text += f" ({self.missing} not produced, {self.contradicted} produced otherwise)"
        if self.outcomes:
            text += f", {self.outcomes - self.outcomes_missing} of {self.outcomes} required outcomes"
        text += f"; {self.outside} more recorded values produced outside the required set (each compared)"
        if self.lost:
            text += f"; {self.lost} recorded values outside it not produced where no budget explains it"
        for label, count in sorted(self.unknown.items()):
            text += f"; {label}: {count} recorded values not produced"
        if self.stopped:
            text += f"; {self.stopped} run{'s' if self.stopped != 1 else ''} stopped once every required value had arrived"
        return text


def _unreproduced(source: str, record_id: str, fresh: Observation, need: Requirement,
                  missing: list[tuple[int, str]], margin: float, timeout_s: float) -> list[Finding]:
    findings = []
    reached = len(fresh.positions)
    cut = observations.cut_short(fresh)
    lost = [item for item in missing if cut and item[0] >= reached - 1]
    if lost:
        index, name = lost[0]
        where = f"at position {reached - 1}" if reached else "before its first position"
        why = "stopped" if fresh.reason == "stopped" else f"cut at timeout_s={number(timeout_s)}"
        findings.append(Finding("golden-unreproduced", (
            f"{source} on {record_id}: the run was {why} after {fresh.elapsed_s:.1f} s, {where}, before its last required "
            f"position {need.last} - {len(lost)} required value(s) not produced, the first at position {index}: {name} "
            f"(recorded cost {need.cost.get(index, 0.0)} s; required because cost x {number(margin)} fits "
            f"{number(timeout_s)} s)")))
    rest: dict[int, list[str]] = {}
    for index, name in missing:
        if (index, name) not in lost:
            rest.setdefault(index, []).append(name)
    for index, names in sorted(rest.items()):
        limit = fresh.extras.get(index, {}).get("limit")
        findings.append(Finding("golden-unreproduced", (
            f"{source} on {record_id} position {index}: {', '.join(names)} recorded (cost {need.cost.get(index, 0.0)} s) "
            f"and not produced by this run{f' (it ran into {limit})' if limit else ''} - the run took "
            f"{fresh.elapsed_s:.1f} s")))
    return findings


def unreached(source: Source, fresh: Observation, index: int, name: str, options: RunOptions) -> bool:
    """Does a budget explain why a fresh run did not produce a recorded value?

    It does where the run never got to that position (it was cut, or stopped,
    before it - or it ended earlier, which the comparison of outcomes reports),
    where it was cut while it searched that very position and the value comes
    from that search, and where it names a limit it ran into there (Chess in 5D's
    mate search). Nowhere else: a run that answered the position, or set it up,
    and no longer gives a value it used to has lost it to no budget.
    """
    if index >= len(fresh.positions):
        return True
    if observations.cut_short(fresh) and index == len(fresh.positions) - 1 and name in source.searched:
        return True
    return name in source.limits(fresh, index, options)


def _lost(source: str, record_id: str, fresh: Observation, need: Requirement,
          lost: list[tuple[tuple[int, str], str]]) -> list[Finding]:
    """One finding per record for the recorded values outside the required set that went missing for no budget."""
    if not lost:
        return []
    (index, _), label = min(lost, key=lambda item: (item[0][0], ROW_FIELDS.index(item[0][1])))
    names = [name for (at, name), _ in sorted(lost, key=lambda item: ROW_FIELDS.index(item[0][1])) if at == index]
    ended = "" if observations.cut_short(fresh) else f" and ended {fresh.status}"
    return [Finding("golden-unreproduced", (
        f"{source} on {record_id}: {len(lost)} recorded value(s) outside this tier's required set not produced, though "
        f"the run reached their positions{ended} and ran into no limit there - no budget explains the loss; the first "
        f"at position {index}: {', '.join(names)} (recorded cost {need.cost.get(index, 0.0)} s; {label})"))]


def check_run(source: Source, records: list[Record], results, recorded: Golden, costs: Costs, tier: Tier,
              margin: float) -> tuple[list[Finding], Tally]:
    """Does a fresh run reproduce its tier's required set, and agree with every other recorded value it produced?

    And is every recorded value it did not produce outside the required set
    explained by a budget (unreached)? One that is not is a finding too: the
    tier does not require the value, but losing it is no budget's doing.
    """
    findings: list[Finding] = []
    tally = Tally()
    name = source.name
    timeout_s = tier.timeout_for(name)
    options = RunOptions(timeout_s=timeout_s, mate_budget_ms=tier.mate_budget_ms)
    for record in records:
        golden_report = recorded.observations.get(record.id)
        fresh = results.get((name, record.id))
        if golden_report is None or fresh is None or record.id not in costs.records:
            continue
        need = requirement(recorded, costs, record.id, timeout_s, tier.mate_budget_ms, margin, source.searched)
        # Everything both said must agree, required or not.
        shown = replace(golden_report, source=f"{name}@golden")
        for disagreement in observations.compare(shown, fresh):
            findings.append(Finding("golden-mismatch", disagreement.describe()))
        changed = observations.status_changed(golden_report, fresh)
        if changed:
            findings.append(Finding("golden-status", f"{name} on {record.id}: {changed}"))
        if need.outcome:
            tally.outcomes += 1
            if observations.cut_short(fresh):
                tally.outcomes_missing += 1
                findings.append(Finding("golden-unreproduced", (
                    f"{name} on {record.id}: the run was cut short ({fresh.reason}) after {fresh.elapsed_s:.1f} s, "
                    f"before its outcome ({_said(golden_report)}, recorded at {costs.records[record.id].end} s), "
                    f"which this tier requires")))
            elif changed:
                tally.outcomes_missing += 1  # golden-status says how
        missing: list[tuple[int, str]] = []
        for item in need.values:
            produced = observations.value(fresh, *item)
            if produced is None:
                missing.append(item)
            elif produced != observations.value(golden_report, *item):
                tally.contradicted += 1  # golden-mismatch (or the placement divergence before it) says how
        missing.sort(key=lambda item: (item[0], ROW_FIELDS.index(item[1])))
        tally.values += len(need.values)
        tally.missing += len(missing)
        lost: list[tuple[tuple[int, str], str]] = []
        for item, label in need.excluded.items():
            if observations.value(fresh, *item) is not None:
                tally.outside += 1
            elif unreached(source, fresh, *item, options):
                tally.unknown[label] += 1
            else:
                lost.append((item, label))
        tally.lost += len(lost)
        tally.stopped += fresh.reason == "stopped"
        findings += _unreproduced(name, record.id, fresh, need, missing, margin, timeout_s)
        findings += _lost(name, record.id, fresh, need, lost)
    return findings, tally
