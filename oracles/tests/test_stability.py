"""Stable answers only (decision H32): recordings merged and classified, the required sets
derived from them, and a gate run held to 100% of its required set."""

import dataclasses
import random
import unittest
from collections import Counter

from oracle_harness import config, golden, lockfile, observations, stability
from oracle_harness.adapters import RunOptions, Source
from oracle_harness.adapters.chessin5d import ChessIn5dSource
from oracle_harness.config import Stability, Tier
from oracle_harness.corpus import Record
from oracle_harness.golden import Costs, Golden, Horizon
from oracle_harness.observations import Observation, Position
from oracle_harness.snapshot import Snapshot

W = ("white", False, 20, "none")
B = ("black", False, 20, "none")
SEARCHING = (None, None, None, None)
OPTIONS = RunOptions(cap=1000, timeout_s=300, mate_budget_ms=60000)


class Search(Source):
    """A stand-in for one oracle search: answers every observable, says nothing it set up."""

    def __init__(self, name="ftxi:naive"):
        self.name = name


class Counter_(Search):
    """A cwmtt-like search: only the count and the verdict come from the search."""

    searched = ("legal_actions", "terminal")


def run(rows, status="ok", reason="", at=None, elapsed=None, extras=None, source="ftxi:naive", record="r/1"):
    """One recording: positions from rows, arriving a second apart unless ``at`` says when."""
    positions = [Position(index, *row) for index, row in enumerate(rows)]
    arrival = list(at) if at is not None else [float(index + 1) for index in range(len(positions))]
    observation = Observation(source=source, record=record, status=status, positions=positions,
                              actions_accepted=max(len(positions) - 1, 0), reason=reason, cap=1000,
                              arrival_s=arrival, extras=dict(extras or {}),
                              elapsed_s=elapsed if elapsed is not None else len(positions) + 0.5)
    if status != "timeout":
        observation.ended_s = observation.elapsed_s
    return observation


def cut(rows, **options):
    """A recording cut at its deadline while it searched its last position."""
    return run([*rows, SEARCHING], status="timeout", reason="timeout", **options)


class MergeTests(unittest.TestCase):
    """golden --write --repeat N: independent recordings, every value classified."""

    def merge(self, *runs, source=None, options=OPTIONS):
        merged = stability.merge(source or Search(), options, list(runs))
        self.assertIsInstance(merged, golden.Recorded, merged)
        self.assertEqual(observations.validate(merged.observation), [])
        return merged

    def test_recordings_that_agree_are_recorded_as_they_are(self):
        merged = self.merge(run([W, B], at=[1.0, 2.0]), run([W, B], at=[1.5, 1.8]))
        self.assertEqual([position.side for position in merged.observation.positions], ["white", "black"])
        self.assertEqual((merged.horizon, merged.unknown, merged.unstable), (None, {}, {}))
        # The cost of each position is the most any recording took, rounded up.
        self.assertEqual(merged.cost.positions, (1.5, 2.0))
        self.assertEqual(merged.observation.status, "ok")

    def test_a_value_that_differs_between_recordings_is_unstable_and_compared_with_nothing(self):
        merged = self.merge(run([W, B]), run([W, ("black", False, 21, "none")]))
        self.assertIsNone(merged.observation.positions[1].legal_actions)
        self.assertEqual(merged.unstable, {1: {"legal_actions": [20, 21]}})
        self.assertEqual(merged.observation.positions[1].side, "black")  # the rest of the position stands

    def test_a_silence_no_budget_explains_is_unstable_too(self):
        merged = self.merge(run([W, B]), run([W, ("black", None, 20, "none")]))
        self.assertEqual(merged.unstable, {1: {"in_check": [False, None]}})

    def test_a_run_cut_at_its_deadline_leaves_a_horizon_where_it_was_searching(self):
        merged = self.merge(cut([W, B, W]), cut([W, B, W, B, W]))
        observation = merged.observation
        # Everything from the shorter recording's search on is unknown at the budget.
        self.assertEqual(merged.horizon, Horizon(3, "timeout_s=300"))
        self.assertEqual((observation.status, len(observation.positions)), ("timeout", 4))
        self.assertEqual(observation.positions[3], Position(3))  # the other recording's answers there are not kept
        self.assertEqual((merged.unknown, merged.unstable), ({}, {}))
        self.assertIsNone(merged.cost.end)

    def test_an_end_only_one_recording_reached_is_unknown_at_the_budget(self):
        # cwmtt:naive's final verdict on cwmtt/ctp1 arrived at 257 s of 300 in one recording; under load the other was cut.
        merged = self.merge(run([W, B, W]), cut([W, B]))
        self.assertEqual(merged.horizon, Horizon(2, "timeout_s=300"))
        self.assertEqual(merged.observation.status, "timeout")
        self.assertEqual(merged.observation.positions[2], Position(2))

    def test_a_null_a_limit_caused_is_unknown_at_that_limit(self):
        source = ChessIn5dSource(lockfile.load())
        limited = {1: {"ms": 61000.0, "limit": "mate_budget_ms"}}
        answered = {1: {"ms": 900.0}}
        verdicts = [("white", False, None, "none"), ("black", False, None, "none")]
        unanswered = [verdicts[0], ("black", False, None, None)]
        # One recording ran out of mate budget where the other answered; then both did.
        merged = self.merge(run(verdicts, extras=answered, source="chessin5d"),
                            run(unanswered, extras=limited, source="chessin5d"), source=source)
        self.assertEqual(merged.unknown, {1: {"terminal": "mate_budget_ms=60000"}})
        self.assertIsNone(merged.observation.positions[1].terminal)
        merged = self.merge(run(unanswered, extras=limited, source="chessin5d"),
                            run(unanswered, extras={1: {"ms": 2.0, "limit": "mate_heap_mb"}}, source="chessin5d"),
                            source=source)
        self.assertEqual(merged.unknown, {1: {"terminal": "mate_budget_ms=60000, mate_heap_mb=1024"}})
        # Its mate search's cost is kept beside the position's: the mate budget bounds it (H32).
        self.assertEqual(merged.cost.mate, (None, 61.0))

    def test_a_null_no_limit_explains_stays_a_bare_null(self):
        # Chess in 5D never counts: that is no budget's doing, and nothing is listed.
        verdicts = [("white", False, None, "none")]
        merged = self.merge(run(verdicts), run(verdicts))
        self.assertEqual((merged.unknown, merged.unstable), ({}, {}))
        self.assertIsNone(merged.observation.positions[0].legal_actions)

    def test_recordings_that_end_differently_are_refused_not_merged(self):
        refused = stability.merge(Search(), OPTIONS, [run([W, B]), run([W], status="rejected", reason="illegal_move")])
        self.assertEqual(refused.code, "golden-unstable-outcome")
        self.assertIn("ok after 1 action(s)", refused.message)
        further = stability.merge(Search(), OPTIONS, [run([W]), cut([W, B])])
        self.assertEqual(further.code, "golden-unstable-outcome")

    def test_a_position_costs_at_least_what_every_earlier_one_did(self):
        # cwmtt's state driver printed what was set up at position 2 at once, but the position exists in its
        # report only once the CLI has counted positions 0 and 1 - the second of which took 50 s.
        sided = run([W, W, ("white", None, None, None)], status="timeout", reason="timeout", at=[1.0, 50.0, 0.04])
        merged = self.merge(sided, source=Counter_("cwmtt:naive"))
        self.assertEqual(merged.cost.positions, (1.0, 50.0, 50.0))

    def test_a_search_cut_where_another_answered_is_explained_only_by_its_own_search(self):
        # A cwmtt-like source: its side comes with what it set up, not with its search.
        merged = self.merge(cut([W], source="cwmtt:naive"),
                            run([W, (None, None, 20, "none")], source="cwmtt:naive"), source=Counter_("cwmtt:naive"))
        self.assertEqual(merged.horizon, Horizon(1, "timeout_s=300"))


def recorded_golden(name, reports: dict[str, golden.Recorded], recordings=2) -> tuple[Golden, Costs]:
    """In-memory golden and cost files, as golden.load and load_costs would read them."""
    signature = {"pin": "p", "budget": {"timeout_s": 300}}
    loaded = Golden(source=name, signature=signature, sha256={key: "s" for key in reports},
                    observations={key: item.observation for key, item in reports.items()}, recordings=recordings,
                    horizon={key: item.horizon for key, item in reports.items() if item.horizon},
                    unknown={key: item.unknown for key, item in reports.items() if item.unknown},
                    unstable={key: item.unstable for key, item in reports.items() if item.unstable})
    costs = Costs(source=name, signature=signature, recordings=recordings,
                  records={key: item.cost for key, item in reports.items()})
    return loaded, costs


def tier(name="nightly", timeout=120.0, mate=20000, sources=("ftxi:naive",), require_engine=False):
    return Tier(name=name, sources=sources, record_tiers=("commit",), timeout_s=timeout, source_timeout_s={},
                mate_budget_ms=mate, min_checked=0.75, require_engine=require_engine)


def record(name="r/1"):
    return Record(id=name, source="fake", path=name, blob="b", sha256="s", size=1, board="Standard", licence="x",
                  tier="commit")


class RequirementTests(unittest.TestCase):
    def need(self, merged, timeout=120.0, mate=20000, margin=4.0, searched=observations.ANSWERS):
        loaded, costs = recorded_golden("ftxi:naive", {"r/1": merged})
        return stability.requirement(loaded, costs, "r/1", timeout, mate, margin, searched)

    def test_a_value_is_required_where_its_cost_times_the_margin_fits_the_budget(self):
        merged = stability.merge(Search(), OPTIONS, [run([W, B, W], at=[10.0, 30.0, 31.0], elapsed=40.0)])
        need = self.need(merged)
        self.assertEqual({index for index, _ in need.values}, {0, 1})  # 30 x 4 = 120 fits; 31 x 4 does not
        self.assertEqual(set(need.excluded.values()), {"unknown at budget timeout_s=120"})
        self.assertEqual(len(need.excluded), 4)
        self.assertFalse(need.outcome)  # the end came at 40 s: 160 > 120
        self.assertEqual(need.outcome_excluded, "unknown at budget timeout_s=120")
        self.assertEqual(len(self.need(merged, margin=1.0).values), 12)
        self.assertTrue(self.need(merged, margin=1.0, timeout=40.0).outcome)

    def test_a_mate_verdict_also_needs_its_mate_search_to_fit_the_mate_budget(self):
        source = ChessIn5dSource(lockfile.load())
        rows = [("white", False, None, "none"), ("black", True, None, "checkmate")]
        merged = stability.merge(source, OPTIONS, [run(rows, source="chessin5d", extras={0: {"ms": 1.0}, 1: {"ms": 6000.0}})])
        need = self.need(merged, timeout=600.0, mate=20000)
        self.assertNotIn((1, "terminal"), need.values)  # 6 s x 4 = 24 s > 20 s
        self.assertEqual(need.excluded[(1, "terminal")], "unknown at budget mate_budget_ms=20000")
        self.assertIn((1, "in_check"), need.values)  # answered before the mate search, whatever it took
        self.assertIn((1, "terminal"), self.need(merged, timeout=600.0, mate=60000).values)

    def test_where_a_run_may_stop(self):
        # Nothing past position 1 is required: the run may stop once position 1's answers are in.
        merged = stability.merge(Search(), OPTIONS, [run([W, B, W], at=[10.0, 30.0, 31.0], elapsed=40.0)])
        self.assertEqual(self.need(merged).stop_after, (1, "position"))
        # Where its outcome is required it runs to its end.
        self.assertIsNone(self.need(merged, margin=1.0).stop_after)
        # Where only what it set up at the last required position is required, it may stop on the state.
        described = run([W, SEARCHING], status="timeout", reason="timeout", at=[1.0, 2.0])
        described.positions = [dataclasses.replace(position, placement=f"{index:032x}", boards=1, pieces=32)
                               for index, position in enumerate(described.positions)]
        merged = stability.merge(Search(), OPTIONS, [described])
        self.assertEqual(self.need(merged).stop_after, (1, "state"))
        # A source whose side to move comes with its state (cwmtt) needs only that, there.
        sided = run([W, ("white", None, None, None)], status="timeout", reason="timeout", at=[1.0, 2.0])
        merged = stability.merge(Counter_("cwmtt:naive"), OPTIONS, [sided])
        self.assertEqual(self.need(merged, searched=Counter_.searched).stop_after, (1, "state"))

    def test_a_recorded_cost_is_read_as_cumulative(self):
        # A cost file in which a later position costs less than an earlier one (the nightly run on 2026-10-01
        # caught cwmtt:naive's horizon states recorded that way) is read with each position costing at least
        # what every earlier one did: the state at position 2 is not required where position 1 is not.
        merged = stability.merge(Counter_("cwmtt:naive"), OPTIONS,
                                 [run([W, W, ("white", None, None, None)], status="timeout", reason="timeout",
                                      at=[1.0, 50.0, 0.04])])
        merged.cost = dataclasses.replace(merged.cost, positions=(1.0, 50.0, 0.04))
        need = self.need(merged, searched=Counter_.searched)
        self.assertEqual({index for index, _ in need.values}, {0})
        self.assertEqual(need.cost[2], 50.0)
        self.assertEqual(need.stop_after, (0, "position"))

    def test_a_required_value_a_tier_cannot_reach_is_impossible_by_construction(self):
        # Whatever the costs, margin and budget: a required value arrived within the tier's budget.
        generator = random.Random(32)
        for _ in range(200):
            count = generator.randint(1, 12)
            at = sorted(generator.uniform(0.0, 600.0) for _ in range(count))
            merged = stability.merge(Search(), OPTIONS, [run([W] * count, at=at, elapsed=at[-1] + 1.0)])
            margin = generator.choice([1.0, 1.5, 4.0, 10.0])
            timeout = generator.choice([60.0, 120.0, 300.0])
            need = self.need(merged, timeout=timeout, margin=margin)
            self.assertTrue(all(need.cost[index] <= timeout for index, _ in need.values))
        # And gate.toml cannot set a margin below 1 (test_config).


class CheckRunTests(unittest.TestCase):
    """A gate run against its required set: 100%, by name, and nothing else turned into a ratio."""

    def check(self, merged, fresh, timeout=120.0, margin=4.0, source=None):
        loaded, costs = recorded_golden("ftxi:naive", {"r/1": merged})
        results = {("ftxi:naive", "r/1"): fresh}
        return stability.check_run(source or Search(), [record()], results, loaded, costs, tier(timeout=timeout), margin)

    def desktop(self, count=25, slow_from=None):
        """The recording machine: positions a second apart, or from ``slow_from`` on, 200 s apart."""
        at = [float(index + 1) if slow_from is None or index < slow_from else 200.0 * (index - slow_from + 1)
              for index in range(count)]
        return run([W] * count, at=at, elapsed=at[-1] + 0.5)

    def test_a_run_that_reproduces_its_required_set_passes(self):
        merged = stability.merge(Search(), OPTIONS, [self.desktop(), self.desktop()])
        findings, tally = self.check(merged, self.desktop())
        self.assertEqual(findings, [])
        self.assertEqual((tally.values, tally.missing, tally.outcomes, tally.outcomes_missing), (100, 0, 1, 0))
        self.assertIn("100 of 100 required values reproduced, 1 of 1 required outcomes", tally.describe())

    def test_the_nightly_gate_fails_at_95_9_percent_until_the_missing_answers_are_classified(self):
        # On the CI runner the naive searches lost 4.1% of their recorded values to the 120 s budget,
        # and the gate passed on a coverage floor of 60%. Now: the recording says those values arrived
        # early on the recording machine - so they are required - and losing them fails, by name.
        merged = stability.merge(Search(), OPTIONS, [self.desktop(), self.desktop()])
        runner = cut([W] * 24, elapsed=120.1)  # cut while it searched position 24: 96 of 100 values
        findings, tally = self.check(merged, runner)
        self.assertEqual(tally.missing, 4)
        self.assertEqual([item.code for item in findings], ["golden-unreproduced", "golden-unreproduced"])
        outcome, values = findings
        self.assertIn("ftxi:naive on r/1: the run was cut short (timeout) after 120.1 s, before its outcome (ok after 24 "
                      "action(s), recorded at 26.0 s), which this tier requires", outcome.message)
        self.assertEqual(values.message, "ftxi:naive on r/1: the run was cut at timeout_s=120 after 120.1 s, at position 24, "
                                         "before its last required position 24 - 4 required value(s) not produced, the "
                                         "first at position 24: side (recorded cost 25.0 s; required because cost x 4 "
                                         "fits 120 s)")

    def test_once_classified_the_desktop_run_passes_at_100_percent_of_its_required_set(self):
        # Re-recorded twice: the last position took the recording machine 200 s - beyond 120 / 4 - so
        # a nightly run that cannot reach it is not failed for it; the value is counted as unknown at
        # the nightly budget, never as a ratio.
        first, second = self.desktop(slow_from=24), self.desktop(slow_from=24)
        merged = stability.merge(Search(), OPTIONS, [first, second])
        nightly = cut([W] * 24, elapsed=120.0)
        findings, tally = self.check(merged, nightly)
        self.assertEqual(findings, [])
        self.assertEqual((tally.values, tally.missing), (96, 0))
        self.assertEqual(tally.unknown, Counter({"unknown at budget timeout_s=120": 4}))
        self.assertIn("unknown at budget timeout_s=120: 4 recorded values not produced", tally.describe())
        self.assertNotIn("%", tally.describe())

    def test_a_value_produced_outside_the_required_set_must_still_agree(self):
        merged = stability.merge(Search(), OPTIONS, [self.desktop(slow_from=24)])
        wrong = run([W] * 24 + [("white", False, 21, "none")])
        findings, tally = self.check(merged, wrong)
        self.assertEqual([item.code for item in findings], ["golden-mismatch"])
        self.assertEqual(tally.outside, 4)

    def test_a_required_value_lost_inside_an_answered_position_is_named(self):
        merged = stability.merge(Search(), OPTIONS, [self.desktop()])
        fresh = self.desktop()
        fresh.positions[7] = dataclasses.replace(fresh.positions[7], terminal=None)
        fresh.extras = {7: {"limit": "mate_budget_ms"}}
        findings, _ = self.check(merged, fresh)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "golden-unreproduced")
        self.assertIn("r/1 position 7: terminal recorded (cost 8.0 s) and not produced by this run (it ran into "
                      "mate_budget_ms)", findings[0].message)

    def test_a_run_stopped_once_its_required_set_was_in_passes(self):
        merged = stability.merge(Search(), OPTIONS, [self.desktop(slow_from=24)])
        stopped = run([W] * 24, status="timeout", reason="stopped", elapsed=24.0)
        findings, tally = self.check(merged, stopped)
        self.assertEqual(findings, [])
        self.assertEqual(tally.stopped, 1)

    def recorded_late(self):
        """Positions 0 and 1 cost 0.1 and 0.2 s on record, 2 and 3 cost 40 and 50 s - outside a 120 s tier's set
        (x 4) - and position 3 is checkmate."""
        rows = [W, B, ("white", True, 30, "none"), ("black", True, 0, "checkmate")]
        return stability.merge(Search(), OPTIONS, [run(rows, at=[0.1, 0.2, 40.0, 50.0], elapsed=50.5)])

    def test_a_value_outside_the_required_set_that_goes_missing_for_no_budget_is_a_finding(self):
        # The run finished well inside its budget, reported every position - and gave no in_check at position 2
        # and no verdict at 3. The tier does not require them, but no budget explains losing them.
        merged = self.recorded_late()
        fresh = run([W, B, ("white", None, 30, "none"), ("black", True, 0, None)], at=[0.1, 0.2, 30.0, 40.0], elapsed=60.0)
        findings, tally = self.check(merged, fresh)
        self.assertEqual([item.code for item in findings], ["golden-unreproduced"])
        self.assertIn("ftxi:naive on r/1: 2 recorded value(s) outside this tier's required set not produced, though the run "
                      "reached their positions and ended ok and ran into no limit there - no budget explains the loss; the "
                      "first at position 2: in_check (recorded cost 40.0 s; unknown at budget timeout_s=120)",
                      findings[0].message)
        self.assertEqual((tally.lost, sum(tally.unknown.values())), (2, 0))
        self.assertIn("2 recorded values outside it not produced where no budget explains it", tally.describe())
        # So too in a run cut later, at a position it had answered.
        cut_later = run([W, B, ("white", None, 30, "none"), SEARCHING], status="timeout", reason="timeout",
                        at=[0.1, 0.2, 30.0, 119.0], elapsed=120.0)
        findings, _ = self.check(merged, cut_later)
        self.assertEqual([item.code for item in findings], ["golden-unreproduced"])
        self.assertIn("1 recorded value(s) outside this tier's required set not produced", findings[0].message)

    def test_a_value_outside_the_required_set_is_excused_only_where_a_budget_explains_it(self):
        merged = self.recorded_late()
        # Cut while it searched position 2: what that search gives there, and everything after it, is unknown at the
        # budget - not what it had set up there before it searched.
        searching = run([W, B, SEARCHING], status="timeout", reason="timeout", at=[0.1, 0.2, 0.3], elapsed=120.0)
        findings, tally = self.check(merged, searching)
        self.assertEqual(findings, [])
        self.assertEqual(tally.unknown, Counter({"unknown at budget timeout_s=120": 8}))
        stated = dataclasses.replace(merged.observation, positions=[
            dataclasses.replace(position, placement=f"{position.index:032x}", boards=1, pieces=32)
            for position in merged.observation.positions])
        merged = dataclasses.replace(merged, observation=stated)
        searching.positions = [dataclasses.replace(position, placement=f"{position.index:032x}", boards=1, pieces=32)
                               for position in searching.positions]
        self.assertEqual(self.check(merged, searching)[0], [])
        unstated = dataclasses.replace(searching, positions=[*searching.positions[:2], Position(2)])
        findings, _ = self.check(merged, unstated)
        self.assertEqual([item.code for item in findings], ["golden-unreproduced"])
        self.assertIn("the first at position 2: placement, boards, pieces", findings[0].message)
        # A limit the run names there: Chess in 5D's mate search ran out of its budget.
        source = ChessIn5dSource(lockfile.load())
        rows = [("white", False, None, "none"), ("black", True, None, "checkmate")]
        limited = stability.merge(source, OPTIONS, [run(rows, at=[0.1, 40.0], elapsed=41.0, source="chessin5d")])
        fresh = run([rows[0], ("black", True, None, None)], at=[0.1, 30.0], elapsed=31.0, source="chessin5d",
                    extras={1: {"limit": "mate_budget_ms"}})
        loaded, costs = recorded_golden("chessin5d", {"r/1": limited})
        results = {("chessin5d", "r/1"): fresh}
        findings, tally = stability.check_run(source, [record()], results, loaded, costs, tier(), 4.0)
        self.assertEqual(findings, [])
        self.assertEqual(sum(tally.unknown.values()), 1)
        fresh.extras = {}
        self.assertEqual([item.code for item in stability.check_run(source, [record()], results, loaded, costs, tier(),
                                                                       4.0)[0]], ["golden-unreproduced"])

    def test_a_required_value_contradicted_or_an_outcome_changed_is_not_counted_as_reproduced(self):
        merged = stability.merge(Search(), OPTIONS, [self.desktop(count=3)])
        contradicted = self.desktop(count=3)
        contradicted.positions[1] = dataclasses.replace(contradicted.positions[1], legal_actions=21)
        findings, tally = self.check(merged, contradicted)
        self.assertEqual([item.code for item in findings], ["golden-mismatch"])
        self.assertEqual((tally.values, tally.missing, tally.contradicted), (12, 0, 1))
        self.assertIn("11 of 12 required values reproduced (0 not produced, 1 produced otherwise), 1 of 1 required "
                      "outcomes", tally.describe())
        crashed = run([W, W], status="error", reason="crash", elapsed=2.0)
        findings, tally = self.check(merged, crashed)
        self.assertEqual(sorted({item.code for item in findings}), ["golden-status", "golden-unreproduced"])
        self.assertEqual((tally.outcomes, tally.outcomes_missing), (1, 1))
        self.assertIn("8 of 12 required values reproduced (4 not produced, 0 produced otherwise), 0 of 1 required "
                      "outcomes", tally.describe())


class LossTests(unittest.TestCase):
    """golden --write says what a new recording no longer holds that the files it replaces did."""

    def test_a_verdict_an_outcome_or_positions_lost_are_each_named(self):
        previous = {"r/1": run([W, B, ("white", True, 0, "checkmate")]),
                    "r/2": run([W, B, W, SEARCHING], status="timeout", reason="timeout"),
                    "r/3": run([W, B])}
        recorded = {"r/1": stability.merge(Search(), OPTIONS, [cut([W])]),
                    "r/2": stability.merge(Search(), OPTIONS, [cut([W])]),
                    "r/3": stability.merge(Search(), OPTIONS, [run([W, B])])}
        lines = stability.losses("ftxi:naive", previous, recorded)
        self.assertEqual(lines, [
            "ftxi:naive on r/1: its checkmate verdict at position 2 is gone (the recording was cut at timeout_s=300 at "
            "position 1)",
            "ftxi:naive on r/1: its outcome (ok after 2 action(s)) is gone - the recording was cut at timeout_s=300 at "
            "position 1",
            "ftxi:naive on r/2: cut at position 1, where the file it replaces was cut at position 3"])
        self.assertEqual(stability.losses("ftxi:naive", {"r/3": run([W, B])}, recorded), [])


class DriverDeadlineTests(unittest.TestCase):
    def test_a_driver_deadline_leaves_the_state_unknown_and_only_an_answer_is_recorded_as_unknown(self):
        # cwmtt's state driver ran out of time before position 1: its side and state are unknown there at the
        # deadline (CwmttSource.limits), and a recording never lists a state field as unknown.
        from oracle_harness.adapters.cwmtt import DRIVER_DEADLINE, CwmttSource

        source = CwmttSource("cwmtt:naive", lockfile.load())
        stated = run([W, (None, None, 20, "none")], source="cwmtt:naive", extras={1: {"limit": DRIVER_DEADLINE}})
        stated.positions[0] = dataclasses.replace(stated.positions[0], placement="0" * 32, boards=1, pieces=32)
        self.assertEqual(set(source.limits(stated, 1, OPTIONS)), {"side", "fingerprint", "placement", "boards", "pieces"})
        self.assertEqual(source.limits(stated, 0, OPTIONS), {})
        merged = stability.merge(source, OPTIONS, [stated, stated])
        self.assertEqual(merged.unknown, {1: {"side": "timeout_s=300"}})


def snapshot_of(**sources) -> Snapshot:
    frozen = Snapshot()
    for name, reports in sources.items():
        frozen.golden[name.replace("_", ":")], frozen.costs[name.replace("_", ":")] = recorded_golden(
            name.replace("_", ":"), reports)
    return frozen


class ProofChannelTests(unittest.TestCase):
    """The static non-vacuity check of the tier that can enable the proof channel (it replaces a 90% ratio)."""

    def cfg(self, release_timeout=300.0, share=0.9):
        loaded = config.load()
        release = tier("release", timeout=release_timeout, mate=60000, sources=("ftxi:naive",), require_engine=True)
        # A tier that requires an engine and runs every source over every record tier:
        release = dataclasses.replace(release, sources=loaded.tier("release").sources, record_tiers=("commit", "nightly"))
        return dataclasses.replace(loaded, tiers={"release": release},
                                   stability=Stability(margin=4.0, recording_tier="release", proof_share=share))

    def reports(self, mate_at):
        rows = [W, B, ("white", True, 0, "checkmate")]
        return {"r/1": stability.merge(Search(), OPTIONS, [run(rows, at=[1.0, 2.0, mate_at], elapsed=mate_at + 0.1)])}

    def test_every_terminal_verdict_must_be_required(self):
        frozen = snapshot_of(ftxi_naive=self.reports(mate_at=70.0))
        self.assertEqual(stability.proof_channel_findings(self.cfg(), [record()], frozen), [])
        late = snapshot_of(ftxi_naive=self.reports(mate_at=80.0))  # 80 x 4 > 300
        found = stability.proof_channel_findings(self.cfg(), [record()], late)
        self.assertEqual([item.code for item in found], ["proof-channel-vacuous", "proof-channel-vacuous"])
        self.assertIn("does not require ftxi:naive's checkmate verdict at r/1 position 2", found[0].message)
        self.assertIn("requires 8 of the 12 values ftxi:naive would record on its records but for its budgets (12 "
                      "recorded, about 0 behind horizons, 0 unknown at a limit, 0 unstable): 66.7%, below the reviewed "
                      "share of 90%", found[1].message)
        self.assertEqual(stability.proof_channel_findings(self.cfg(release_timeout=400.0), [record()], late), [])

    def test_the_summary_says_why_values_are_not_required(self):
        frozen = snapshot_of(ftxi_naive=self.reports(mate_at=80.0))
        lines = stability.describe(self.cfg(), [record()], frozen)
        self.assertIn("recorded under the 'release' tier's budgets", lines[0])
        self.assertIn("  release  ftxi:naive   8 values and 0 outcomes required (of 12 recorded values and 1 recorded "
                      "outcomes); unknown at budget timeout_s=300: 4 values, 1 outcome (recorded cost x margin above it); "
                      "0 of 1 recorded terminal verdicts required", lines)
        self.assertTrue(any("not computed" in line for line in lines))  # the sources with no files

    def test_starving_the_recording_budget_makes_the_check_harder_not_easier(self):
        # The release tier's budgets are the recording budgets. Fed 300 s, ftxi:naive's last two positions (100 s
        # each, the second a checkmate) are recorded but not required. Starved to 60 s, its recording is cut while it
        # searches position 10: they vanish behind a horizon - and must not vanish from the check with them.
        rows = [W, B] * 5 + [W, ("black", True, 0, "checkmate")]
        quick = [0.1 * (index + 1) for index in range(10)]
        ftxi = stability.merge(Search("ftxi"), OPTIONS, [run(rows, at=[*quick, 1.1, 1.2], elapsed=1.3, source="ftxi")])
        fed = stability.merge(Search(), OPTIONS, [run(rows, at=[*quick, 100.0, 100.0], elapsed=100.5)])
        starved_options = dataclasses.replace(OPTIONS, timeout_s=60)
        starved = stability.merge(Search(), starved_options,
                                  [run([*rows[:10], SEARCHING], status="timeout", reason="timeout", at=[*quick, 1.5],
                                       elapsed=60.0)])
        found = stability.proof_channel_findings(self.cfg(), [record()], snapshot_of(ftxi={"r/1": ftxi},
                                                                                   ftxi_naive={"r/1": fed}))
        self.assertEqual(len(found), 2)
        self.assertIn("does not require ftxi:naive's checkmate verdict at r/1 position 11", found[0].message)
        self.assertIn("requires 40 of the 48 values ftxi:naive would record", found[1].message)
        found = stability.proof_channel_findings(self.cfg(release_timeout=60.0), [record()],
                                                 snapshot_of(ftxi={"r/1": ftxi}, ftxi_naive={"r/1": starved}))
        self.assertEqual([item.code for item in found], ["proof-channel-vacuous", "proof-channel-vacuous"])
        self.assertIn("does not require ftxi:naive's verdict at r/1 position 11: its recording was cut at timeout_s=60 "
                      "at position 10, and ftxi recorded checkmate there", found[0].message)
        self.assertIn("requires 40 of the 48 values ftxi:naive would record on its records but for its budgets (40 "
                      "recorded, about 8 behind horizons", found[1].message)

    def test_a_source_that_recorded_nothing_is_not_required_to_have_nothing(self):
        nothing = {"r/1": stability.merge(Search(), OPTIONS, [run([], status="unsupported", reason="variant")])}
        found = stability.proof_channel_findings(self.cfg(), [record()], snapshot_of(ftxi_naive=nothing))
        self.assertEqual([item.code for item in found], ["proof-channel-vacuous"])
        self.assertIn("lists ftxi:naive, which recorded nothing on the tier's records", found[0].message)

    def test_what_each_tier_requires_is_a_table_for_review(self):
        frozen = snapshot_of(ftxi_naive=self.reports(mate_at=80.0))
        lines = stability.required_table(self.cfg(), [record()], frozen.golden, frozen.costs)
        self.assertEqual(lines[0], "## What each tier requires")
        self.assertTrue(any("times 4 (`[stability] margin` in `gate.toml`) fits the" in line for line in lines))
        self.assertEqual(lines[-1], "| release | `ftxi:naive` | timeout_s=300 | 8 | 12 | 0 | 1 | 0 | 1 |")
        # A lowered budget is a changed line.
        lowered = stability.required_table(self.cfg(release_timeout=120.0), [record()], frozen.golden, frozen.costs)
        self.assertEqual(lowered[-1], "| release | `ftxi:naive` | timeout_s=120 | 8 | 12 | 0 | 1 | 0 | 1 |")


if __name__ == "__main__":
    unittest.main()
