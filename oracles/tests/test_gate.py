import contextlib
import dataclasses
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import support
from oracle_harness import baseline, config, fingerprint, gate, golden, integrity, lockfile, paths, snapshot, stability, util
from oracle_harness.adapters import ORACLE_SOURCES, PARSE_ONLY_SOURCES, EngineSource, RunOptions, Source
from oracle_harness.corpus import CUSTOM_POSITIONS, Record
from oracle_harness.divergences import Divergence
from oracle_harness.gate import Gated, GateResult
from oracle_harness.observations import Observation, Position
from oracle_harness.util import Finding, HarnessError

W = ("white", False, 20, "none")
B = ("black", False, 20, "none")
B_CHECK = ("black", True, 20, "none")
# What cwmtt can say about a position: how many actions, and nothing else.
COUNT = (None, None, 20, "none")
# What Chess in 5D can say: everything but the count.
W_VERDICT = ("white", False, None, "none")
B_VERDICT = ("black", False, None, "none")
TZ = "Standard - Turn Zero"
FULL = {"in_check": 2, "legal_actions": 2, "terminal": 2}


class Named(Source):
    def __init__(self, name):
        self.name = name
        self.oracle = name.split(":")[0]


def record(name, board="Standard", fen=False):
    return Record(id=name, source="fake", path=name, blob="b", sha256="s", size=1, board=board, licence="x",
                  tier="commit", fen=fen)


def obs(source, record_id, rows, status="ok", reason=""):
    positions = [Position(index, *row) for index, row in enumerate(rows)]
    return Observation(source=source, record=record_id, status=status, positions=positions, reason=reason,
                       actions_accepted=max(len(positions) - 1, 0), cap=1000)


def hashed(number):
    return f"{number:032x}"


def described(observation, placements, fingerprints=None):
    """The same report, saying what it set up at each position (one board, 32 pieces)."""
    fingerprints = fingerprints or [None] * len(placements)
    observation.positions = [
        dataclasses.replace(position, placement=hashed(placement), fingerprint=None if marked is None else hashed(marked),
                            boards=1, pieces=32)
        for position, placement, marked in zip(observation.positions, placements, fingerprints, strict=True)]
    return observation


def state_text(number):
    """A hash as the gate prints it."""
    return f"{hashed(number)} (1 board, 32 pieces)"


class DifferentialCase(unittest.TestCase):
    """The engine differential's fixtures: three oracle searches over two records of two rulesets."""

    def setUp(self):
        self.engine = EngineSource(["engine"])
        self.oracles = [Named("ftxi"), Named("ftxi:naive"), Named("chessin5d")]
        self.records = [record("r/std"), record("r/tz", TZ)]

    def reports(self, engine_rows=None, oracle_reports=None):
        """(oracle results, engine results): every source says [W, B] about every record unless told otherwise."""
        engine_rows = engine_rows or {}
        oracle_reports = oracle_reports or {}
        results, engine_results = {}, {}
        for item in self.records:
            for oracle in self.oracles:
                value = oracle_reports.get((oracle.name, item.id), [W, B])
                results[(oracle.name, item.id)] = value if isinstance(value, Observation) else obs(oracle.name, item.id, value)
            value = engine_rows.get(item.id, [W, B])
            engine_results[("engine", item.id)] = value if isinstance(value, Observation) else obs("engine", item.id, value)
        return results, engine_results

    def run_differential(self, engine_rows=None, oracle_reports=None, entries=(), min_checked=1.0, recorded=None):
        results, engine_results = self.reports(engine_rows, oracle_reports)
        return gate.engine_differential(self.engine, self.oracles, self.records, results, engine_results,
                                        list(entries), min_checked, recorded)

    def silence(self, *names, record_id="r/tz", reason="variant"):
        """Oracle reports saying these sources do not read that record."""
        return {(name, record_id): obs(name, record_id, [], status="unsupported", reason=reason) for name in names}

    def divergence(self, **changes):
        values = dict(id="KD-0001", oracle="chessin5d", pin="p", record="r/std", record_sha256="s", kind="in_check",
                      index=1, engine_value="True", oracle_value="False", rulebook="x" * 30,
                      upstream="https://example.com/1", added="2026-10-01")
        values.update(changes)
        return Divergence(**values)


class EngineDifferentialTests(DifferentialCase):
    def test_an_engine_that_agrees_with_every_oracle_passes_every_ruleset(self):
        findings, gated, ungated = self.run_differential()
        self.assertEqual(findings, [])
        # Two searches of one oracle are one independent check, not two.
        expected = Gated(records=1, positions=2, checked=FULL, oracles=2)
        self.assertEqual(gated, {"Standard": expected, TZ: expected})
        self.assertEqual(ungated, {})

    def test_any_unexplained_divergence_fails_and_ungates_that_ruleset(self):
        findings, gated, ungated = self.run_differential({"r/tz": [W, B_CHECK]})
        self.assertEqual([item.code for item in findings], ["divergence"] * 3)  # one per oracle source
        self.assertIn("r/tz position 1: in_check: engine=True vs ftxi=False", findings[0].message)
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "unexplained divergence"})

    def test_a_known_divergence_is_excused_where_it_was_recorded(self):
        # The engine and ftxi say check; Chess in 5D says no check, and that is on record.
        reports = {("ftxi", "r/std"): [W, B_CHECK], ("ftxi:naive", "r/std"): [W, B_CHECK]}
        findings, gated, _ = self.run_differential({"r/std": [W, B_CHECK]}, reports, [self.divergence()])
        self.assertEqual(findings, [])
        self.assertIn("Standard", gated)

    def test_a_known_divergence_excuses_nothing_but_itself(self):
        # The same entry, and the engine also differs from Chess in 5D on the other record.
        reports = {("ftxi", "r/std"): [W, B_CHECK], ("ftxi:naive", "r/std"): [W, B_CHECK],
                   ("ftxi", "r/tz"): [W, B_CHECK], ("ftxi:naive", "r/tz"): [W, B_CHECK]}
        findings, gated, ungated = self.run_differential({"r/std": [W, B_CHECK], "r/tz": [W, B_CHECK]}, reports, [self.divergence()])
        self.assertEqual([item.code for item in findings], ["divergence"])
        self.assertIn("r/tz position 1: in_check: engine=True vs chessin5d=False", findings[0].message)
        self.assertEqual((list(gated), list(ungated)), (["Standard"], [TZ]))

    def test_a_known_divergence_that_stops_occurring_is_stale(self):
        # Chess in 5D now says check, as the engine does: the oracle's error is gone.
        fixed = {(name, "r/std"): [W, B_CHECK] for name in ("ftxi", "ftxi:naive", "chessin5d")}
        findings, _, _ = self.run_differential({"r/std": [W, B_CHECK]}, fixed, entries=[self.divergence()])
        self.assertEqual([item.code for item in findings], ["divergence-stale"])
        self.assertIn("remove it, or find out why", findings[0].message)

    def test_an_engine_that_reproduces_a_recorded_oracle_error_is_a_telltale(self):
        # H34. The entry says Chess in 5D is wrong to call position 1 "no check". An engine
        # that says "no check" there too has, probably, Chess in 5D's logic. The disagreement
        # no longer occurs - and the remedy is never "remove it".
        findings, gated, ungated = self.run_differential(entries=[self.divergence()])
        self.assertEqual([item.code for item in findings], ["telltale"])
        self.assertEqual(findings[0].message,
                         "KD-0001: the engine reproduces chessin5d's recorded error at r/std position 1 (in_check = "
                         "False): its logic may have been copied from that oracle")
        self.assertNotIn("Standard", gated)
        self.assertEqual(ungated["Standard"], "the engine reproduces a recorded oracle error (telltale)")

    def test_a_telltale_needs_no_run_of_the_oracle_it_points_at(self):
        # Chess in 5D does not run in this tier; what it recorded is in the golden file.
        self.oracles = [Named("ftxi"), Named("ftxi:naive")]
        recorded = obs("chessin5d", "r/std", [W_VERDICT, B_VERDICT])
        engine = {"r/std": [W, B]}
        findings, _, _ = self.run_differential(engine, entries=[self.divergence()],
                                               recorded=lambda source, record_id: recorded if source == "chessin5d" else None)
        self.assertEqual([item.code for item in findings], ["telltale"])
        # Not where the engine holds another position than the one the oracle answered about.
        placed = described(obs("chessin5d", "r/std", [W_VERDICT, B_VERDICT]), [1, 3])
        elsewhere = {"r/std": described(obs("engine", "r/std", [W, B]), [1, 2], [11, 12])}
        reports = {(name, "r/std"): described(obs(name, "r/std", [W, B]), [1, 2]) for name in ("ftxi", "ftxi:naive")}
        findings, _, _ = self.run_differential(elsewhere, reports, entries=[self.divergence()],
                                               recorded=lambda source, record_id: placed if source == "chessin5d" else None)
        self.assertEqual(findings, [])

    def test_a_known_divergence_the_oracle_never_reached_is_not_stale(self):
        # Chess in 5D ran out of time before position 1: nothing was looked at.
        short = obs("chessin5d", "r/std", [W], status="timeout", reason="timeout")
        findings, _, _ = self.run_differential(oracle_reports={("chessin5d", "r/std"): short}, entries=[self.divergence()])
        self.assertEqual(findings, [])

    def test_an_engine_that_breaks_on_a_record_fails(self):
        broken = obs("engine", "r/tz", [W], status="error", reason="crash")
        findings, gated, ungated = self.run_differential({"r/tz": broken})
        self.assertEqual([item.code for item in findings], ["engine-failed"])
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "the engine did not replay r/tz"})

    def test_an_unsupported_ruleset_is_not_a_failure_but_is_not_gated_either(self):
        unsupported = obs("engine", "r/tz", [], status="unsupported", reason="variant")
        findings, gated, ungated = self.run_differential({"r/tz": unsupported})
        self.assertEqual(findings, [])
        self.assertEqual(list(gated), ["Standard"])
        self.assertIn(TZ, ungated)

    def test_a_ruleset_no_oracle_replayed_is_never_gated(self):
        # The engine replays it happily - but nothing independent looked.
        reports = self.silence("ftxi", "ftxi:naive", "chessin5d")
        findings, gated, ungated = self.run_differential(oracle_reports=reports)
        self.assertEqual(findings, [])
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "no oracle followed r/tz as far as the engine went"})

    def test_an_oracle_that_stopped_short_does_not_stand_behind_the_record(self):
        reports = {(oracle.name, "r/tz"): obs(oracle.name, "r/tz", [W], status="timeout", reason="timeout")
                   for oracle in self.oracles}
        _, gated, ungated = self.run_differential(oracle_reports=reports)
        self.assertNotIn(TZ, gated)
        self.assertEqual(ungated, {TZ: "no oracle followed r/tz as far as the engine went"})

    def test_a_refusal_by_the_engine_needs_an_oracle_that_refuses_the_same_action(self):
        refused = obs("engine", "r/tz", [W], status="rejected", reason="illegal_move")
        # Every oracle merely stopped reading after the first position: nobody confirms the refusal.
        stopped = {(oracle.name, "r/tz"): obs(oracle.name, "r/tz", [W], status="unsupported", reason="notation")
                   for oracle in self.oracles}
        findings, gated, ungated = self.run_differential({"r/tz": refused}, stopped)
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "no oracle followed r/tz as far as the engine went"})
        # One that cannot match the move's text confirms nothing about the move either.
        unreadable = dict(stopped)
        unreadable[("ftxi", "r/tz")] = obs("ftxi", "r/tz", [W], status="rejected", reason="no_matching_move")
        findings, gated, ungated = self.run_differential({"r/tz": refused}, unreadable)
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "no oracle followed r/tz as far as the engine went"})
        # One oracle refuses the same action as illegal: it stands behind the refusal - alone, which gates nothing (H47).
        agreeing = dict(stopped)
        agreeing[("ftxi", "r/tz")] = obs("ftxi", "r/tz", [W], status="rejected", reason="cannot_submit")
        findings, gated, ungated = self.run_differential({"r/tz": refused}, agreeing)
        self.assertEqual(findings, [])
        self.assertTrue(ungated[TZ].startswith("only one oracle followed r/tz (ftxi) as far as the engine went"))
        # A second implementation refuses it too: the engine's refusal is confirmed by two, and the ruleset is gated.
        agreeing[("chessin5d", "r/tz")] = obs("chessin5d", "r/tz", [W_VERDICT], status="rejected", reason="illegal_move")
        findings, gated, _ = self.run_differential({"r/tz": refused}, agreeing, min_checked=0.5)
        self.assertEqual(findings, [])
        self.assertEqual(gated[TZ].oracles, 2)

    def test_text_the_engine_could_not_read_is_a_soft_divergence_only_the_notation_can_excuse(self):
        # Full mode: the engine's parser could not read r/tz's first action, which every oracle played.
        unread = obs("engine", "r/tz", [W], status="rejected", reason="unreadable")
        findings, gated, ungated = self.run_differential({"r/tz": unread})
        self.assertEqual([item.code for item in findings], ["divergence"] * 2)  # ftxi and Chess in 5D
        self.assertTrue(all("action 1: legality: engine=rejected:unreadable vs" in item.message and "(soft" in item.message
                            for item in findings))
        self.assertEqual(ungated[TZ], "unexplained divergence")
        # Excused by entries citing the notation specification, it is still no verdict: the ruleset stays ungated.
        entries = [self.divergence(id=f"KD-000{number}", oracle=name, record="r/tz", kind="legality", index=1,
                                   engine_value="rejected:unreadable", oracle_value="accepted", rulebook="",
                                   notation="x" * 30)
                   for number, name in enumerate(("ftxi", "chessin5d"), 1)]
        findings, gated, ungated = self.run_differential({"r/tz": unread}, entries=entries)
        self.assertEqual(findings, [])
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "the engine could not read r/tz (a refusal as unreadable is no verdict)"})
        # Not even an oracle that refuses the same action as illegal stands behind a refusal to read it.
        refusing = {(oracle.name, "r/tz"): obs(oracle.name, "r/tz", [W], status="rejected", reason="illegal_move")
                    for oracle in self.oracles}
        findings, gated, ungated = self.run_differential({"r/tz": unread}, refusing)
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "the engine could not read r/tz (a refusal as unreadable is no verdict)"})

    def test_a_set_up_the_engine_could_not_read_is_a_soft_divergence_at_action_0(self):
        refused = obs("engine", "r/tz", [], status="rejected", reason="unreadable")
        findings, gated, ungated = self.run_differential({"r/tz": refused})
        self.assertEqual([item.code for item in findings], ["divergence"] * 2)
        self.assertTrue(all("action 0 (the record's set-up): legality: engine=rejected:unreadable vs" in item.message
                            and "(soft" in item.message for item in findings))
        entries = [self.divergence(id=f"KD-000{number}", oracle=name, record="r/tz", kind="legality", index=0,
                                   engine_value="rejected:unreadable", oracle_value="accepted", rulebook="",
                                   notation="x" * 30)
                   for number, name in enumerate(("ftxi", "chessin5d"), 1)]
        findings, gated, ungated = self.run_differential({"r/tz": refused}, entries=entries)
        self.assertEqual(findings, [])
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "the engine could not read r/tz (a refusal as unreadable is no verdict)"})

    def test_a_ruleset_whose_only_oracle_has_no_check_verdict_is_not_gated(self):
        # cwmtt is the only oracle that knows several variants, and it says
        # nothing about check. Agreeing with it about the counts is not a pass -
        # and one oracle alone gates nothing in any case (H47).
        self.oracles = [Named("ftxi"), Named("cwmtt"), Named("chessin5d")]
        reports = self.silence("ftxi", "chessin5d")
        reports[("cwmtt", "r/tz")] = obs("cwmtt", "r/tz", [COUNT, COUNT])
        findings, gated, ungated = self.run_differential(oracle_reports=reports, min_checked=0.75)
        self.assertEqual(findings, [])
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "only one oracle followed r/tz (cwmtt) as far as the engine went - a ruleset is "
                                       "gated only when at least 2 answer for each of its records (H47)"})
        # Two oracles that follow it and give no check verdict between them leave check unanswered: not gated either.
        reports[("chessin5d", "r/tz")] = obs("chessin5d", "r/tz", [("white", None, None, "none"), ("black", None, None, "none")])
        findings, gated, ungated = self.run_differential(oracle_reports=reports, min_checked=0.75)
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "the oracles give a check verdict at only 0 of the engine's 2 positions (the floor is 75%)"})

    def test_a_ruleset_is_gated_only_when_two_oracles_answer_for_each_of_its_records(self):
        # H47 (CHECKPOINT 8): "a single oracle cannot tell its own bug from ours". Two searches of one implementation are
        # one oracle, and a parse-only source never counts.
        reader = PARSE_ONLY_SOURCES[0]
        self.records = [record("r/tz", TZ)]
        for oracles in ([Named("ftxi")], [Named("ftxi"), Named("ftxi:naive")], [Named("ftxi"), Named(reader)]):
            self.oracles = oracles
            reports = {(reader, "r/tz"): described(obs(reader, "r/tz", [("white", None, None, None), ("black", None, None, None)]),
                                                   [1, 2])}
            engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2])}
            findings, gated, ungated = self.run_differential(engine, reports)
            self.assertEqual((findings, gated), ([], {}), [oracle.name for oracle in oracles])
            self.assertEqual(ungated, {TZ: "only one oracle followed r/tz (ftxi) as far as the engine went - a ruleset "
                                           "is gated only when at least 2 answer for each of its records (H47)"})
        # Two implementations: gated, and the count says so.
        self.oracles = [Named("ftxi"), Named("chessin5d")]
        findings, gated, ungated = self.run_differential()
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated[TZ].oracles, gate.GATING_ORACLES)
        self.assertIn("each record followed as far as the engine went by at least 2 oracles", gated[TZ].describe())
        # Every record needs its two: one record that only one oracle follows ungates its ruleset, naming that oracle.
        self.records = [record("r/tz", TZ), record("r/tz2", TZ), record("r/tz3", TZ)]
        reports = self.silence("chessin5d", record_id="r/tz2", reason="notation")
        findings, gated, ungated = self.run_differential(oracle_reports=reports)
        self.assertEqual((findings, gated), ([], {}))
        self.assertEqual(ungated[TZ], "only one oracle followed r/tz2 (ftxi) as far as the engine went - a ruleset is "
                                      "gated only when at least 2 answer for each of its records (H47)")

    def test_a_no_action_verdict_does_not_settle_which_kind_of_end_it_is(self):
        self.oracles = [Named("ftxi"), Named("cwmtt"), Named("chessin5d")]
        mate = ("black", True, 0, "checkmate")
        reports = self.silence("ftxi")
        reports[("cwmtt", "r/tz")] = obs("cwmtt", "r/tz", [COUNT, (None, None, 0, "no_action")])
        reports[("chessin5d", "r/tz")] = obs("chessin5d", "r/tz", [W_VERDICT, ("black", True, None, None)])
        findings, gated, ungated = self.run_differential({"r/tz": [W, mate]}, reports, min_checked=1.0)
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "the oracles give a mate verdict at only 1 of the engine's 2 positions (the floor is 100%)"})

    def test_observables_may_be_covered_by_different_oracles(self):
        # One oracle counts, another gives the verdicts: between them every question has an answer.
        self.oracles = [Named("ftxi"), Named("cwmtt"), Named("chessin5d")]
        reports = self.silence("ftxi")
        reports[("cwmtt", "r/tz")] = obs("cwmtt", "r/tz", [COUNT, COUNT])
        reports[("chessin5d", "r/tz")] = obs("chessin5d", "r/tz", [W_VERDICT, B_VERDICT])
        findings, gated, ungated = self.run_differential(oracle_reports=reports)
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated[TZ], Gated(records=1, positions=2, checked={"in_check": 2, "legal_actions": 2, "terminal": 2}, oracles=2))

    def test_the_floor_is_a_share_of_the_engines_positions(self):
        self.records = [record("r/a", TZ), record("r/b", TZ)]
        # r/b is followed by two oracles, neither of which gives a check verdict there.
        self.oracles = [Named("ftxi"), Named("cwmtt"), Named("chessin5d")]
        silent = [("white", None, None, "none"), ("black", None, None, "none")]
        reports = {("ftxi", "r/b"): obs("ftxi", "r/b", [], status="unsupported", reason="notation"),
                   ("cwmtt", "r/b"): obs("cwmtt", "r/b", [COUNT, COUNT]),
                   ("chessin5d", "r/b"): obs("chessin5d", "r/b", silent),
                   ("ftxi", "r/a"): [W, B, W, B, W, B], ("cwmtt", "r/a"): [COUNT] * 6,
                   ("chessin5d", "r/a"): [W_VERDICT, B_VERDICT] * 3}
        engine = {"r/a": [W, B, W, B, W, B]}
        _, gated, ungated = self.run_differential(engine, reports, min_checked=0.75)
        self.assertEqual(gated[TZ], Gated(records=2, positions=8, checked={"in_check": 6, "legal_actions": 8, "terminal": 8}, oracles=2))
        _, gated, ungated = self.run_differential(engine, reports, min_checked=0.8)
        self.assertEqual(gated, {})
        self.assertIn("check verdict at only 6 of the engine's 8 positions", ungated[TZ])

    def test_the_count_is_the_weakest_record_of_the_ruleset(self):
        self.oracles.append(Named("cwmtt"))
        self.records.append(record("r/std2"))
        reports = self.silence("chessin5d", record_id="r/std2", reason="notation")
        _, gated, _ = self.run_differential(oracle_reports=reports)
        self.assertEqual(gated["Standard"].oracles, 2)  # r/std has three behind it (ftxi, cwmtt, Chess in 5D), r/std2 two
        self.assertEqual(gated["Standard"].records, 2)
        self.assertEqual(gated[TZ].oracles, 3)

    def test_rejecting_a_move_the_oracles_play_is_a_divergence(self):
        rejected = obs("engine", "r/std", [W], status="rejected", reason="illegal_move")
        findings, gated, _ = self.run_differential({"r/std": rejected})
        # How a record is replayed is compared once per implementation, not once per search.
        self.assertEqual([item.code for item in findings], ["divergence"] * 2)
        self.assertTrue(all("legality" in item.message for item in findings))
        self.assertEqual({item.message.split(" vs ")[1].split("=")[0] for item in findings}, {"ftxi", "chessin5d"})
        self.assertNotIn("Standard", gated)

    def test_one_entry_excuses_an_implementations_refusal(self):
        # ftxi cannot match the move text of action 2; the engine and Chess in 5D play it.
        stuck = {("ftxi", "r/std"): obs("ftxi", "r/std", [W, B], status="rejected", reason="no_matching_move"),
                 ("ftxi:naive", "r/std"): obs("ftxi:naive", "r/std", [W, B], status="rejected", reason="no_matching_move"),
                 ("chessin5d", "r/std"): [W, B, W]}
        findings, _, _ = self.run_differential({"r/std": [W, B, W]}, stuck)
        self.assertEqual([item.code for item in findings], ["divergence"])
        entry = self.divergence(oracle="ftxi", kind="legality", index=2, engine_value="accepted",
                                oracle_value="rejected:no_matching_move", rulebook="", notation="x" * 30)
        findings, gated, ungated = self.run_differential({"r/std": [W, B, W]}, stuck, [entry])
        self.assertEqual(findings, [])
        # Only Chess in 5D went as far as the engine: one oracle, which gates nothing (H47).
        self.assertNotIn("Standard", gated)
        self.assertTrue(ungated["Standard"].startswith("only one oracle followed r/std (chessin5d) as far as the engine"))

    def test_an_oracle_whose_reading_was_excused_does_not_stand_behind_the_record(self):
        # The engine stops after the first position and calls the record over;
        # entries excuse every oracle's disagreement about its length. Nothing
        # left followed the record as the engine read it.
        self.records = [record("r/std")]
        short = obs("engine", "r/std", [W])
        entries = [self.divergence(id=f"KD-000{number}", oracle=name, kind="length", index=1, engine_value="0 actions",
                                   oracle_value="1 actions", rulebook="", notation="x" * 30)
                   for number, name in enumerate(("ftxi", "chessin5d"), 1)]
        findings, gated, ungated = self.run_differential({"r/std": short}, entries=entries)
        self.assertEqual(findings, [])
        self.assertEqual(gated, {})
        self.assertEqual(ungated, {"Standard": "no oracle followed r/std as far as the engine went"})

    def test_an_answer_the_engine_contradicts_is_not_counted_as_independent(self):
        # Chess in 5D is the only source of check verdicts here. The engine
        # contradicts it at position 1 and an entry says Chess in 5D is wrong:
        # then nothing stands behind the engine's own check verdict there.
        self.oracles = [Named("cwmtt"), Named("chessin5d")]
        reports = {("cwmtt", "r/tz"): obs("cwmtt", "r/tz", [COUNT, COUNT]),
                   ("chessin5d", "r/tz"): obs("chessin5d", "r/tz", [W_VERDICT, B_VERDICT])}
        entry = self.divergence(record="r/tz", kind="in_check", index=1, engine_value="True", oracle_value="False")
        findings, _, ungated = self.run_differential({"r/tz": [W, B_CHECK]}, reports, [entry], min_checked=1.0)
        self.assertEqual(findings, [])
        self.assertEqual(ungated[TZ], "the oracles give a check verdict at only 1 of the engine's 2 positions (the floor is 100%)")

    def test_records_with_their_own_start_position_form_their_own_ruleset(self):
        self.records = [record("r/std"), record("r/fen", "Standard", fen=True), record("r/custom", "Custom", fen=True)]
        _, gated, _ = self.run_differential()
        self.assertEqual(sorted(gated), [CUSTOM_POSITIONS, "Standard"])
        self.assertEqual((gated["Standard"].records, gated[CUSTOM_POSITIONS].records), (1, 2))


class ParseDifferentialTests(DifferentialCase):
    """What the engine set up is compared with what each oracle set up, first (decision H30)."""

    def everyone(self, record_id, rows, placements, fingerprints=None, names=("ftxi", "ftxi:naive", "chessin5d")):
        return {(name, record_id): described(obs(name, record_id, rows), placements, fingerprints) for name in names}

    def placement_entries(self, engine_says, oracle_says, names=("ftxi", "ftxi:naive", "chessin5d")):
        return [self.divergence(id=f"KD-000{number}", oracle=name, record="r/tz", kind="placement", index=1,
                                engine_value=state_text(engine_says), oracle_value=state_text(oracle_says),
                                rulebook="", notation="x" * 30)
                for number, name in enumerate(names, 1)]

    def test_positions_set_up_alike_are_counted_as_aligned(self):
        engine = {item.id: described(obs("engine", item.id, [W, B]), [1, 2], [11, 12]) for item in self.records}
        reports = {**self.everyone("r/std", [W, B], [1, 2], [11, 12]), **self.everyone("r/tz", [W, B], [1, 2], [11, 12])}
        findings, gated, ungated = self.run_differential(engine, reports)
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated[TZ], Gated(records=1, positions=2, checked=FULL, oracles=2, aligned=2, matched=2))
        self.assertIn("2 positions; an oracle held the engine's boards and pieces at 2 of them, and its exact state "
                      "(markers included) at 2;", gated[TZ].describe())

    def test_an_oracle_that_does_not_track_markers_verifies_the_boards_only(self):
        engine = {item.id: described(obs("engine", item.id, [W, B]), [1, 2], [11, 12]) for item in self.records}
        reports = {**self.everyone("r/std", [W, B], [1, 2]), **self.everyone("r/tz", [W, B], [1, 2])}
        findings, gated, _ = self.run_differential(engine, reports)
        self.assertEqual(findings, [])
        self.assertEqual((gated[TZ].aligned, gated[TZ].matched), (2, 0))
        self.assertEqual(gated[TZ].checked, FULL)  # nothing known to differ: its answers count

    def test_an_oracle_that_says_nothing_of_its_state_leaves_the_alignment_unverified(self):
        engine = {item.id: described(obs("engine", item.id, [W, B]), [1, 2]) for item in self.records}
        _, gated, _ = self.run_differential(engine)
        self.assertEqual((gated[TZ].aligned, gated[TZ].matched), (0, 0))

    def test_a_placement_divergence_is_a_finding_and_nothing_after_it_is_compared(self):
        # The engine reads r/tz's first action as another position than every
        # oracle does - and then, there, disagrees about check. Only the parse
        # divergence is reported: the check verdicts are about different positions.
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B_CHECK]), [1, 2])}
        reports = self.everyone("r/tz", [W, B], [1, 3])
        findings, gated, ungated = self.run_differential(engine, reports)
        self.assertEqual([item.code for item in findings], ["divergence"] * 3)  # one per oracle source
        self.assertTrue(findings[0].message.startswith(
            f"r/tz position 1: placement: engine={state_text(2)} vs ftxi={state_text(3)} (parse divergence"))
        self.assertTrue(all("placement" in item.message for item in findings))
        self.assertEqual((list(gated), ungated), (["Standard"], {TZ: "unexplained divergence"}))

    def test_an_excused_placement_divergence_still_leaves_the_oracle_unable_to_stand_behind_the_record(self):
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2])}
        reports = self.everyone("r/tz", [W, B], [1, 3])
        findings, gated, ungated = self.run_differential(engine, reports, self.placement_entries(2, 3))
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "no oracle followed r/tz as far as the engine went"})

    def test_answers_from_where_an_oracle_parted_from_the_engine_check_nothing(self):
        # ftxi parts from the engine at position 1, and an entry excuses that. Its
        # count there is about another position, and neither Chess in 5D nor a
        # second oracle that follows the record gives a count: position 1's count
        # stands unchecked.
        self.oracles.append(Named("cwmtt"))
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2])}
        reports = {**self.everyone("r/tz", [W, B], [1, 3], names=("ftxi", "ftxi:naive")),
                   **self.everyone("r/tz", [W_VERDICT, B_VERDICT], [1, 2], names=("chessin5d", "cwmtt"))}
        entries = self.placement_entries(2, 3, names=("ftxi", "ftxi:naive"))
        findings, gated, ungated = self.run_differential(engine, reports, entries)
        self.assertEqual(findings, [])
        self.assertEqual(ungated, {TZ: "the oracles give a count verdict at only 1 of the engine's 2 positions "
                                       "(the floor is 100%)"})

    def unmoved_entries(self, index, engine_says, oracle_says, names=("ftxi", "ftxi:naive", "chessin5d"), first=1):
        return [self.divergence(id=f"KD-{number:04d}", oracle=name, record="r/tz", kind="unmoved", index=index,
                                engine_value=state_text(engine_says), oracle_value=state_text(oracle_says),
                                rulebook="", notation="x" * 30)
                for number, name in enumerate(names, first)]

    def test_a_marker_only_divergence_is_a_finding_that_suppresses_nothing(self):
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B_CHECK]), [1, 2], [11, 12])}
        reports = self.everyone("r/tz", [W, B], [1, 2], [11, 22])
        findings, gated, _ = self.run_differential(engine, reports)
        kinds = sorted(item.message.split(": ")[1] for item in findings)
        self.assertEqual(kinds, ["in_check"] * 3 + ["unmoved"] * 3)  # the check disagreement is still there

    def test_an_oracle_whose_markers_differ_follows_the_record_but_checks_nothing_there(self):
        # Excused, the oracles still stand behind the record - they read the same
        # moves to the same pieces - but at position 1 they hold other unmoved
        # markers than the engine: their answers there are about another position.
        entries = self.unmoved_entries(1, 12, 22)
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2], [11, 12])}
        reports = self.everyone("r/tz", [W, B], [1, 2], [11, 22])
        findings, gated, ungated = self.run_differential(engine, reports, entries)
        self.assertEqual(findings, [])
        self.assertEqual(ungated[TZ], "the oracles give a check verdict at only 1 of the engine's 2 positions (the "
                                      "floor is 100%); at 1 more, the only check verdict that agrees comes from an "
                                      "oracle whose unmoved markers differ from the engine's there, which does not count")
        # Where an oracle that holds the engine's exact state answers too, the position is checked.
        self.oracles.append(Named("cwmtt"))
        reports[("cwmtt", "r/tz")] = described(obs("cwmtt", "r/tz", [W, B]), [1, 2], [11, 12])
        findings, gated, ungated = self.run_differential(engine, reports, entries)
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated[TZ], Gated(records=1, positions=2, checked=FULL, oracles=3, aligned=2, matched=2))

    def test_each_run_of_differing_markers_is_its_own_divergence(self):
        # The markers differ at positions 1 and 2, agree at 3, and differ again at 4:
        # two runs, two divergences - one entry cannot excuse what comes after the markers met again.
        rows = [W, B, W, B, W]
        engine = {"r/tz": described(obs("engine", "r/tz", rows), [1, 2, 3, 4, 5], [11, 12, 13, 14, 15])}
        reports = self.everyone("r/tz", rows, [1, 2, 3, 4, 5], [11, 22, 23, 14, 25])
        findings, _, _ = self.run_differential(engine, reports)
        self.assertEqual(sorted(item.message.split(":")[0] for item in findings),
                         ["r/tz position 1"] * 3 + ["r/tz position 4"] * 3)
        findings, _, _ = self.run_differential(engine, reports, self.unmoved_entries(1, 12, 22))
        self.assertEqual([item.message.split(" vs ")[0] for item in findings],
                         [f"r/tz position 4: unmoved: engine={state_text(15)}"] * 3)
        both = self.unmoved_entries(1, 12, 22) + self.unmoved_entries(4, 15, 25, first=4)
        findings, _, _ = self.run_differential(engine, reports, both)
        self.assertEqual(findings, [])

    def test_a_parse_only_source_stands_behind_no_replay_and_checks_no_verdict(self):
        # The parser reads all of r/tz as the engine does, and nothing else reads it:
        # what was set up is verified, but no rules oracle followed the record.
        reader = PARSE_ONLY_SOURCES[0]
        self.oracles = [Named("ftxi"), Named(reader)]
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2])}
        reports = {**self.silence("ftxi"), (reader, "r/tz"): described(obs(reader, "r/tz", [("white", None, None, None),
                                                                                          ("black", None, None, None)]), [1, 2])}
        findings, gated, ungated = self.run_differential(engine, reports, min_checked=0.5)
        self.assertEqual(findings, [])
        self.assertEqual(ungated[TZ], "no oracle followed r/tz as far as the engine went")
        # Nor does it count towards the two oracles a ruleset needs (H47): ftxi alone follows r/std.
        self.assertTrue(ungated["Standard"].startswith("only one oracle followed r/std (ftxi) "), ungated["Standard"])
        # And its playing on is no verdict against the engine's refusal of an action.
        refused = described(obs("engine", "r/tz", [W], status="rejected", reason="illegal_move"), [1])
        findings, _, _ = self.run_differential({"r/tz": refused}, reports, min_checked=0.5)
        self.assertEqual(findings, [])

    def test_a_placement_entry_that_no_longer_occurs_is_stale(self):
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2])}
        reports = self.everyone("r/tz", [W, B], [1, 2])
        findings, _, _ = self.run_differential(engine, reports, self.placement_entries(2, 3))
        self.assertEqual([item.code for item in findings], ["divergence-stale"] * 3)

    def test_an_entry_about_a_position_after_the_parting_is_neither_used_nor_stale(self):
        # The engine and ftxi now part at position 0: nothing after it was compared,
        # so an entry about check at position 1 was not looked at.
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [5, 2])}
        reports = self.everyone("r/tz", [W, B], [1, 2])
        check = self.divergence(record="r/tz", oracle="chessin5d", index=1)
        findings, _, _ = self.run_differential(engine, reports, [check])
        self.assertEqual([item.code for item in findings], ["divergence"] * 3)
        self.assertTrue(all("position 0: placement" in item.message for item in findings))


class EngineParseModeTests(DifferentialCase):
    """An engine in parse mode: compared on what it set up, whose move it is and how it replayed each record."""

    def setUp(self):
        super().setUp()
        self.engine = EngineSource(["engine"], "parse")

    def parsed(self, record_id, placements, sides=("white", "black", "white"), status="ok", reason=""):
        rows = [(side, None, None, None) for side in sides[:len(placements)]]
        return described(obs("engine", record_id, rows, status=status, reason=reason), placements,
                         [placement + 10 for placement in placements])

    def stated(self, name, record_id, rows, placements, **options):
        return described(obs(name, record_id, rows, **options), placements, [placement + 10 for placement in placements])

    def run_parse(self, engine_rows=None, oracle_reports=None, entries=()):
        engine_rows = engine_rows or {item.id: self.parsed(item.id, [1, 2]) for item in self.records}
        oracle_reports = oracle_reports if oracle_reports is not None else {
            (oracle.name, item.id): self.stated(oracle.name, item.id, [W, B], [1, 2])
            for oracle in self.oracles for item in self.records}
        results, engine_results = self.reports(engine_rows, oracle_reports)
        return gate.parse_differential(self.engine, self.oracles, self.records, results, engine_results, list(entries))

    def test_an_engine_that_reads_every_record_as_the_oracles_do_agrees_and_gates_nothing(self):
        findings, agreement = self.run_parse()
        self.assertEqual(findings, [])
        # Two records, two positions each; ftxi and Chess in 5D each replayed both (ftxi:naive is ftxi's second search).
        self.assertEqual(agreement, gate.ParseAgreement(records=2, read=2, positions=4, described=4, boards=4, states=4,
                                                        sides=4, sides_agreed=4, replays=4, replays_agreed=4))
        self.assertIn("the engine read 2 of 2 records (4 positions)", agreement.describe())
        self.assertIn("check, counts and mate were not compared, and no ruleset is gated", agreement.describe())

    def test_check_counts_and_mate_are_never_compared(self):
        # Every oracle says check at position 1; the engine, in parse mode, says nothing about it.
        reports = {(oracle.name, item.id): self.stated(oracle.name, item.id, [W, B_CHECK], [1, 2])
                   for oracle in self.oracles for item in self.records}
        findings, _ = self.run_parse(oracle_reports=reports)
        self.assertEqual(findings, [])

    def test_what_it_set_up_and_whose_move_it_is_are_compared(self):
        engine = {"r/std": self.parsed("r/std", [1, 3]), "r/tz": self.parsed("r/tz", [1, 2], sides=("white", "white"))}
        findings, agreement = self.run_parse(engine)
        kinds = sorted({item.message.split(": ")[1] for item in findings})
        self.assertEqual(kinds, ["placement", "side"])
        self.assertTrue(all(item.code == "divergence" for item in findings))
        self.assertEqual((agreement.boards, agreement.sides_agreed), (3, 2))

    def test_how_it_replayed_each_record_is_compared_with_each_implementation(self):
        # It could not read the second action, which every oracle played: a soft legality divergence, from ftxi and Chess in 5D.
        unread = self.parsed("r/std", [1], status="rejected", reason="unreadable")
        findings, agreement = self.run_parse({"r/std": unread, "r/tz": self.parsed("r/tz", [1, 2])})
        self.assertEqual([item.code for item in findings], ["divergence"] * 2)
        self.assertTrue(all("legality: engine=rejected:unreadable vs" in item.message for item in findings))
        self.assertTrue(all("(soft" in item.message for item in findings))
        self.assertEqual((agreement.replays, agreement.replays_agreed), (4, 2))
        # It read fewer actions than the oracles found, and called the record over: a length divergence.
        short = self.parsed("r/std", [1])
        findings, _ = self.run_parse({"r/std": short, "r/tz": self.parsed("r/tz", [1, 2])})
        self.assertEqual(sorted({item.message.split(": ")[1] for item in findings}), ["length"])

    def test_playing_an_action_an_oracle_holds_illegal_is_no_verdict_in_parse_mode(self):
        # Without check detection the engine plays what it reads, as a parse-only source does.
        refusals = {(oracle.name, "r/std"): self.stated(oracle.name, "r/std", [W], [1], status="rejected",
                                                         reason="cannot_submit") for oracle in self.oracles}
        findings, agreement = self.run_parse(oracle_reports={**refusals, **{
            (oracle.name, "r/tz"): self.stated(oracle.name, "r/tz", [W, B], [1, 2]) for oracle in self.oracles}})
        self.assertEqual(findings, [])
        # Not counted as agreeing either: the two replays differ there, and nothing says who is right.
        self.assertEqual((agreement.replays, agreement.replays_agreed, agreement.replays_unjudged), (4, 2, 2))
        self.assertIn("and with 2 more but for actions it played that the oracle held illegal (no verdict in parse mode)",
                      agreement.describe())
        # Its refusing one an oracle played as illegal is still compared.
        illegal = self.parsed("r/std", [1], status="rejected", reason="illegal_move")
        findings, _ = self.run_parse({"r/std": illegal, "r/tz": self.parsed("r/tz", [1, 2])})
        self.assertTrue(findings)
        self.assertTrue(all("rejected:illegal_move" in item.message and "(soft" not in item.message for item in findings))

    def test_an_engine_that_read_nothing_has_not_passed(self):
        declined = {item.id: obs("engine", item.id, [], status="unsupported", reason="variant") for item in self.records}
        findings, agreement = self.run_parse(declined)
        self.assertEqual([item.code for item in findings], ["engine-nothing-compared"])
        self.assertEqual(agreement.read, 0)

    def test_a_telltale_is_a_telltale_in_parse_mode_too(self):
        # The entry says ftxi sets up position 1 wrongly; the engine sets up exactly what ftxi does.
        entry = self.divergence(oracle="ftxi", record="r/std", kind="placement", index=1,
                                engine_value=state_text(9), oracle_value=state_text(2), rulebook="", notation="x" * 30)
        findings, _ = self.run_parse(entries=[entry])
        self.assertEqual([item.code for item in findings], ["telltale"])

    def everyone_reads(self):
        return {(oracle.name, item.id): self.stated(oracle.name, item.id, [W, B], [1, 2])
                for oracle in self.oracles for item in self.records}

    def test_an_engine_that_crashes_or_runs_out_of_time_fails(self):
        for status, reason in (("error", "crash"), ("timeout", "timeout")):
            broken = self.parsed("r/tz", [1], status=status, reason=reason)
            findings, agreement = self.run_parse({"r/std": self.parsed("r/std", [1, 2]), "r/tz": broken})
            self.assertEqual([item.code for item in findings], ["engine-failed"], status)
            self.assertEqual(agreement.read, 1, status)  # the record it broke on was not read

    def test_an_action_an_oracle_could_not_read_and_the_engine_played_is_a_soft_divergence(self):
        reports = self.everyone_reads()
        for oracle in self.oracles:
            reports[(oracle.name, "r/std")] = self.stated(oracle.name, "r/std", [W], [1], status="rejected",
                                                          reason="no_matching_move")
        findings, agreement = self.run_parse(oracle_reports=reports)
        self.assertEqual([item.code for item in findings], ["divergence"] * 2)  # ftxi and Chess in 5D
        self.assertTrue(all("action 1: legality: engine=accepted vs" in item.message and "(soft" in item.message
                            for item in findings))
        self.assertEqual((agreement.replays, agreement.replays_agreed, agreement.replays_unjudged), (4, 2, 0))

    def test_an_entry_that_no_longer_occurs_is_stale_in_parse_mode_too(self):
        entry = self.divergence(oracle="ftxi", record="r/std", kind="placement", index=1,
                                engine_value=state_text(9), oracle_value=state_text(3), rulebook="", notation="x" * 30)
        findings, _ = self.run_parse(entries=[entry])
        self.assertEqual([item.code for item in findings], ["divergence-stale"])

    def test_a_rules_entry_about_an_action_the_engine_played_is_neither_stale_nor_a_telltale(self):
        # From full mode: ftxi wrongly refuses r/std action 1 as illegal. The engine in parse mode plays it - no
        # verdict - so the entry is not looked at, and not stale.
        reports = self.everyone_reads()
        for name in ("ftxi", "ftxi:naive"):
            reports[(name, "r/std")] = self.stated(name, "r/std", [W], [1], status="rejected", reason="illegal_move")
        refused = self.divergence(oracle="ftxi", kind="legality", index=1, engine_value="accepted",
                                  oracle_value="rejected:illegal_move")
        findings, agreement = self.run_parse(oracle_reports=reports, entries=[refused])
        self.assertEqual(findings, [])
        self.assertEqual((agreement.replays_agreed, agreement.replays_unjudged), (3, 1))
        # Chess in 5D wrongly plays it (the engine, in full mode, refuses it). Playing it in parse mode reproduces
        # nothing: no telltale, and not stale either.
        played = self.divergence(kind="legality", index=1, engine_value="rejected:illegal_move", oracle_value="accepted")
        findings, _ = self.run_parse(entries=[played])
        self.assertEqual(findings, [])

    def test_a_set_up_the_engine_could_not_read_is_a_soft_divergence_the_notation_can_settle(self):
        refused = obs("engine", "r/std", [], status="rejected", reason="unreadable")
        engine = {"r/std": refused, "r/tz": self.parsed("r/tz", [1, 2])}
        findings, agreement = self.run_parse(engine)
        self.assertEqual([item.code for item in findings], ["divergence"] * 2)  # ftxi and Chess in 5D set it up
        self.assertTrue(all("action 0 (the record's set-up): legality: engine=rejected:unreadable vs" in item.message
                            and "(soft" in item.message for item in findings))
        self.assertEqual((agreement.read, agreement.setups_refused, agreement.replays), (1, 1, 2))
        self.assertIn("refused the set-up of 1", agreement.describe())
        entries = [self.divergence(id=f"KD-000{number}", oracle=name, kind="legality", index=0,
                                   engine_value="rejected:unreadable", oracle_value="accepted", rulebook="",
                                   notation="x" * 30)
                   for number, name in enumerate(("ftxi", "chessin5d"), 1)]
        findings, _ = self.run_parse(engine, entries=entries)
        self.assertEqual(findings, [])
        # An engine that comes to set it up, as each oracle wrongly does, reproduces their errors.
        findings, _ = self.run_parse(entries=entries)
        self.assertEqual([item.code for item in findings], ["telltale", "telltale"])

    def test_the_agreement_counts_an_exact_state_only_where_the_markers_agree_too(self):
        reports = {(oracle.name, item.id): described(obs(oracle.name, item.id, [W, B]), [1, 2], [11, 99])
                   for oracle in self.oracles for item in self.records}
        findings, agreement = self.run_parse(oracle_reports=reports)
        self.assertEqual((agreement.described, agreement.boards, agreement.states), (4, 4, 2))
        self.assertEqual({item.message.split(": ")[1] for item in findings}, {"unmoved"})

    def test_where_an_oracle_parted_from_the_engine_it_described_another_position(self):
        engine = {"r/std": self.parsed("r/std", [1, 3]), "r/tz": self.parsed("r/tz", [1, 2])}
        _, agreement = self.run_parse(engine)
        self.assertEqual((agreement.described, agreement.boards, agreement.states), (4, 3, 3))

    def test_a_replay_an_oracle_did_not_finish_is_not_compared(self):
        reports = self.everyone_reads()
        reports[("ftxi", "r/std")] = self.stated("ftxi", "r/std", [W], [1], status="timeout", reason="timeout")
        findings, agreement = self.run_parse(oracle_reports=reports)
        self.assertEqual(findings, [])
        self.assertEqual((agreement.replays, agreement.replays_agreed), (3, 3))

    def test_an_engine_compared_with_nothing_has_not_passed(self):
        # It read both records - and no oracle read either: no state compared, no replay compared.
        reports = {**self.silence("ftxi", "ftxi:naive", "chessin5d", record_id="r/std"),
                   **self.silence("ftxi", "ftxi:naive", "chessin5d", record_id="r/tz")}
        findings, agreement = self.run_parse(oracle_reports=reports)
        self.assertEqual([item.code for item in findings], ["engine-nothing-compared"])
        self.assertEqual((agreement.read, agreement.described, agreement.replays), (2, 0, 0))


class WriteGoldenTests(unittest.TestCase):
    """golden --write [--repeat N]: every value classified across the recordings, and every file vouched for (H32, H33)."""

    class Signed(Named):
        def __init__(self, name, drifts=False):
            super().__init__(name)
            self.drifts, self.looks = drifts, 0

        def available(self):
            return None

        def signature(self, options):
            self.looks += 1
            return {"adapter_sha256": str(self.looks) if self.drifts else "same"}

    PROBE = golden.Calibration(source="ftxi:naive", record="r/1", positions=1, signature={"pin": "p"}, seconds=12.5,
                               platform="test", cpu_count=4)

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(paths, "GOLDEN_DIR", self.directory / "golden"))
        self.enterContext(mock.patch.object(paths, "DISAGREEMENTS_FILE", self.directory / "DISAGREEMENTS.md"))
        for target, name, value in ((gate.lockfile, "verify", []), (gate.corpus, "load", [record("r/1")]),
                                    (gate.corpus, "verify", []), (gate, "render_disagreements", None),
                                    (gate, "_measure_probe", self.PROBE)):
            self.enterContext(mock.patch.object(target, name, return_value=value))

    def write(self, source, *runs, repeat=None):
        """golden --write for one source whose recordings of r/1 are ``runs``."""
        self.recorder = self.enterContext(mock.patch.object(
            gate.runner, "record_sources", return_value={(source.name, "r/1"): list(runs)}))
        self.enterContext(mock.patch.object(gate, "make_source", return_value=source))
        found = gate.write_golden(only_sources=[source.name], repeat=repeat or len(runs), log=lambda _: None)
        return [finding.code for finding in found]

    def text(self, name):
        return (self.directory / "golden" / name).read_text(encoding="utf-8")

    def test_the_answers_are_signed_by_the_code_that_produced_them_and_the_budget_they_had(self):
        codes = self.write(self.Signed("ftxi"), obs("ftxi", "r/1", [W]))
        self.assertEqual(codes, [])
        self.assertIn('"adapter_sha256": "same"', self.text("ftxi.json"))
        self.assertIn('"budget": {"timeout_s": 600.0}', self.text("ftxi.json"))  # the release tier's, which records
        self.assertIn('"recordings": 1', self.text("ftxi.json"))

    def test_nothing_is_written_when_the_code_changed_while_the_oracles_ran(self):
        codes = self.write(self.Signed("ftxi", drifts=True), obs("ftxi", "r/1", [W]))
        self.assertEqual(codes, ["golden-code-changed"])
        self.assertFalse((self.directory / "golden").exists())

    def test_each_source_is_recorded_n_times_and_what_differed_is_unstable(self):
        first = obs("ftxi", "r/1", [W, B])
        second = obs("ftxi", "r/1", [W, ("black", False, 21, "none")])
        codes = self.write(self.Signed("ftxi"), first, second)
        self.assertEqual(codes, [])
        self.assertEqual(self.recorder.call_args.kwargs["repeat"], 2)
        self.assertIn('"unstable": {"1": {"legal_actions": [20, 21]}}', self.text("ftxi.json"))
        self.assertIn('        ["b", false, null, "none", null, null, null, null]\n', self.text("ftxi.json"))
        self.assertIn('"recordings": 2', self.text("ftxi.cost.json"))
        # MANIFEST vouches for every file written: the answers, their costs, the calibration reference.
        self.assertEqual(sorted(integrity.read_manifest()), ["calibration.json", "ftxi.cost.json", "ftxi.json"])
        self.assertEqual(integrity.verify_manifest(), [])
        self.assertEqual(json.loads(self.text("calibration.json"))["seconds"], 12.5)

    def test_recordings_that_ended_differently_write_nothing(self):
        codes = self.write(self.Signed("ftxi"), obs("ftxi", "r/1", [W, B]),
                           obs("ftxi", "r/1", [W], status="rejected", reason="illegal_move"))
        self.assertEqual(codes, ["golden-unstable-outcome"])
        self.assertFalse((self.directory / "golden").exists())

    def test_a_file_this_run_does_not_rewrite_keeps_its_hash_only_if_it_still_matches(self):
        self.assertEqual(self.write(self.Signed("ftxi"), obs("ftxi", "r/1", [W])), [])
        planted = self.directory / "golden" / "cwmtt.json"
        # Recording ftxi again: cwmtt.json is not rewritten, and MANIFEST does not vouch for it.
        planted.write_text("{}", encoding="utf-8")
        self.assertEqual(self.write(self.Signed("ftxi"), obs("ftxi", "r/1", [W])), ["golden-integrity"])
        planted.unlink()
        # A file it keeps that was edited since MANIFEST was written is not laundered into a new MANIFEST:
        # here the calibration reference, kept because the probe's source cannot run on this machine.
        calibration = self.directory / "golden" / "calibration.json"
        calibration.write_text(calibration.read_text(encoding="utf-8").replace("12.5", "1.0"), encoding="utf-8")
        unavailable = self.Signed("ftxi:naive")
        unavailable.available = lambda: "not built"

        def make(name, lock):
            return unavailable if name == "ftxi:naive" else self.Signed("ftxi")

        recordings = {("ftxi", "r/1"): [obs("ftxi", "r/1", [W])]}
        with mock.patch.object(gate, "make_source", side_effect=make), \
                mock.patch.object(gate.runner, "record_sources", return_value=recordings):
            found = gate.write_golden(only_sources=["ftxi"], log=lambda _: None)
        self.assertEqual([finding.code for finding in found], ["golden-integrity"])
        self.assertIn("calibration.json does not match MANIFEST", found[0].message)


class EngineCommandTests(unittest.TestCase):
    def codes(self, *command):
        return [finding.code for finding in gate.engine_command_findings(list(command))]

    def test_the_adapter_lives_with_the_engine(self):
        self.assertEqual(self.codes("cargo", "run", "--release", "-p", "engine-adapter", "--"), [])
        self.assertEqual(self.codes("python", "engine/adapter.py"), [])
        self.assertEqual(self.codes(), [])

    def test_an_adapter_kept_in_the_harness_is_refused(self):
        # It would sit beside every answer the oracles have given.
        self.assertEqual(self.codes("python", "oracles/engine_adapter.py"), ["engine-in-harness"])
        self.assertEqual(self.codes("python", "oracles\\tools\\adapter.py"), ["engine-in-harness"])
        self.assertEqual(self.codes("node", str(paths.HARNESS_ROOT / "adapter.js")), ["engine-in-harness"])
        self.assertEqual(self.codes("sh", "-c", "cat oracles/golden/ftxi.json"), ["engine-in-harness"])
        # However it gets there: a module, a shell line, an option.
        self.assertEqual(self.codes("python", "-m", "oracles.engine_shim"), ["engine-in-harness"])
        self.assertEqual(self.codes("sh", "-c", "cd oracles && python shim.py"), ["engine-in-harness"])
        self.assertEqual(self.codes("cmd", "/c", "cd Oracles & shim.bat"), ["engine-in-harness"])
        self.assertEqual(self.codes("engine", "--oracles-dir=oracles"), ["engine-in-harness"])

    def test_a_relative_program_is_the_repository_roots_wherever_the_harness_was_started(self):
        # Windows finds a relative program from the directory the harness was started in, not the one the command
        # runs in: resolved against the root, every platform runs the file engine_command_findings checks.
        built = str(paths.REPO_ROOT / "build" / "engine" / "t5d-gate")
        self.assertEqual(gate.engine_argv(["./build/engine/t5d-gate", "--x"]), [built, "--x"])
        self.assertEqual(gate.engine_argv(("build/engine/t5d-gate",)), [built])
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
            self.assertEqual(gate.engine_argv(["./build/engine/t5d-gate"]), [built])
        # Only the program: its arguments are its own to read, in the root it runs in.
        self.assertEqual(gate.engine_argv(["python", "engine/adapter.py"]), ["python", "engine/adapter.py"])
        # An absolute program, and a bare name (looked up on PATH), as they are.
        self.assertEqual(gate.engine_argv([sys.executable, "a.py"]), [sys.executable, "a.py"])
        self.assertEqual(gate.engine_argv(["t5d-gate"]), ["t5d-gate"])
        self.assertEqual(gate.engine_argv([]), [])

    def test_a_run_names_the_file_it_ran(self):
        # So a report says which build was compared - and a stale or missing one shows.
        with tempfile.TemporaryDirectory() as folder:
            program = Path(folder) / "adapter.bin"
            program.write_bytes(b"an engine")
            ran = gate.engine_program([str(program), "--x"])
            self.assertEqual((ran["program"], ran["file"], ran["sha256"]),
                             (str(program), str(program), util.sha256_bytes(b"an engine")))
            self.assertIn("SHA-256", gate._describe_engine_run(ran))
            # Windows runs a program named without an extension as its .exe.
            (Path(folder) / "t5d.exe").write_bytes(b"x")
            self.assertEqual(gate.engine_program([str(Path(folder) / "t5d")])["file"],
                             str(Path(folder) / "t5d.exe") if util.is_windows() else None)
            missing = gate.engine_program([str(Path(folder) / "build" / "t5d-gate")])
            self.assertEqual((missing["file"], missing["sha256"]), (None, None))
            self.assertIn("no such file - has the engine been built?", gate._describe_engine_run(missing))


class ResultTests(unittest.TestCase):
    ONE = Gated(records=1, positions=2, checked=FULL, oracles=2, aligned=2, matched=2)
    THREE = Gated(records=4, positions=90, checked={"in_check": 80, "legal_actions": 85, "terminal": 90}, oracles=3,
                  aligned=88, matched=70)
    ONLY_FTXI = {"sources": ["ftxi"], "records": None}

    def result(self, tier="commit", engine="not configured", findings=(), engine_findings=(), gated=None,
               restricted_to=None, require_engine=False, full_strength=False, mode="full", parse=None, calibration=None,
               tallies=None):
        result = GateResult(tier=tier, engine=engine, gated=dict(gated or {}), restricted_to=restricted_to,
                            require_engine=require_engine, full_strength=full_strength, engine_mode=mode, parse=parse,
                            margin=4.0, calibration=calibration, tallies=dict(tallies or {}))
        result.sections.append(("section", list(findings)))
        if engine != "not configured":
            result.engine_sections.append(("engine vs oracles", list(engine_findings)))
        return result

    def release(self, **options):
        return self.result(tier="release", require_engine=True, full_strength=True, **options)

    def nightly(self, **options):
        return self.result(tier="nightly", full_strength=True, **options)

    def test_without_an_engine_the_oracle_layer_can_pass_but_the_proof_channel_stays_off(self):
        result = self.result()
        self.assertTrue(result.ok)
        self.assertTrue(result.proof_channel.startswith("DISABLED"))
        self.assertIn("NOT RUN", gate.summarise(result))

    def test_a_tier_that_requires_an_engine_cannot_pass_without_one(self):
        result = self.release()
        self.assertTrue(result.oracle_layer_ok)
        self.assertFalse(result.ok)
        self.assertTrue(result.proof_channel.startswith("DISABLED"))
        self.assertIn("this tier requires an engine: FAILED", gate.summarise(result))

    def test_an_engine_that_came_through_nothing_has_not_passed_a_tier_that_runs_every_oracle(self):
        # "Passed" with no ruleset gated would be a pass on the strength of nothing.
        self.assertFalse(self.release(engine="passed").ok)
        self.assertFalse(self.nightly(engine="passed").ok)
        self.assertTrue(self.nightly(engine="passed", gated={"Standard": self.ONE}).ok)
        for result in (self.release(engine="passed"), self.nightly(engine="passed", gated={"Standard": self.ONE})):
            self.assertFalse(result.smoke_test)
            self.assertNotIn("SMOKE TEST", gate.summarise(result))
        self.assertEqual(self.nightly(engine="passed", gated={"Standard": self.ONE}).verdicts, ({"Standard": self.ONE}, {}))

    def test_any_other_tier_is_a_smoke_test_that_gates_nothing(self):
        # H47: a tier that does not run every oracle search over the whole corpus (the commit tier) does not check that
        # some ruleset is gated. An engine with no finding against it passes it, gated or not - and the tier says that it
        # gates nothing, so that a smoke pass never reads as gating.
        nothing = self.result(engine="passed")
        self.assertTrue(nothing.smoke_test)
        self.assertTrue(nothing.ok)
        self.assertTrue(nothing.proof_channel.startswith("DISABLED - only a complete run of a tier that requires an "
                                                         "engine and runs every oracle"), nothing.proof_channel)
        self.assertIn(f"engine differential: SMOKE TEST - PASSED: {gate.SMOKE_TEST}", gate.summarise(nothing))
        # A ruleset that met the gating conditions with the tier's oracles alone is not gated either, and says so.
        met = self.result(engine="passed", gated={"Standard": self.THREE})
        met.ungated = {"Misc - Small": "no oracle followed notation/game as far as the engine went"}
        self.assertTrue(met.ok)
        self.assertEqual(met.verdicts, ({}, {"Misc - Small": "no oracle followed notation/game as far as the engine went",
                                            "Standard": f"{gate.SMOKE_REASON} ({self.THREE.describe()})"}))
        text = gate.summarise(met)
        self.assertNotIn("gated:     ", text)
        self.assertIn(f"NOT gated: Standard - {gate.SMOKE_REASON} (4 records, 90 positions; ", text)
        self.assertIn("NOT gated: Misc - Small - no oracle followed notation/game as far as the engine went", text)
        self.assertTrue(met.proof_channel.startswith("DISABLED"))
        # A restricted run is no smoke test - no tier result at all (H47, CHECKPOINT 8). An unexplained divergence
        # fails the smoke test like any tier.
        restricted = self.result(engine="passed", restricted_to=self.ONLY_FTXI)
        self.assertFalse(restricted.smoke_test)
        self.assertNotIn("SMOKE TEST", gate.summarise(restricted))
        failed = self.result(engine="failed", engine_findings=[Finding("divergence", "r/1 position 3: ...")])
        self.assertFalse(failed.ok)
        self.assertIn(f"engine differential: SMOKE TEST - FAILED: {gate.SMOKE_TEST}", gate.summarise(failed))
        # Neither without an engine nor in parse mode: there is no full-mode differential to read as gating.
        self.assertFalse(self.result().smoke_test)
        self.assertFalse(self.result(engine="passed", mode="parse", parse=gate.ParseAgreement(records=1, read=1)).smoke_test)

    def test_a_tier_is_a_smoke_test_by_what_it_runs_never_by_its_name(self):
        # H47 is keyed to full_strength - every oracle search over the whole corpus - and a tier's name promises nothing.
        # A "nightly" that runs less (cut to fewer sources or records) is a smoke test...
        cut = self.result(tier="nightly", engine="passed")
        self.assertTrue(cut.smoke_test)
        self.assertTrue(cut.ok)
        self.assertIn(f"engine differential: SMOKE TEST - PASSED: {gate.SMOKE_TEST}", gate.summarise(cut))
        self.assertEqual(self.result(tier="nightly", engine="passed", gated={"Standard": self.ONE}).verdicts[0], {})
        # ...and a "commit" that runs everything is not: it fails an engine that came through no ruleset, and gates.
        full = self.result(tier="commit", full_strength=True, engine="passed")
        self.assertFalse(full.smoke_test)
        self.assertFalse(full.ok)
        self.assertNotIn("SMOKE TEST", gate.summarise(full))
        gated = self.result(tier="commit", full_strength=True, engine="passed", gated={"Standard": self.ONE})
        self.assertTrue(gated.ok)
        self.assertEqual(gated.verdicts, ({"Standard": self.ONE}, {}))
        self.assertIn("gated:     Standard (1 record, 2 positions; ", gate.summarise(gated))

    def test_a_tier_that_requires_an_engine_requires_a_named_variant_its_run_may_state_as_gated(self):
        # gate.toml refuses require_engine on a tier that does not run every oracle search over the whole corpus
        # (test_config). Built by hand, such a result fails: what met the gating conditions in a smoke test is not
        # gated, so the pass can never rest on rulesets the summary prints as NOT gated - and the summary says why.
        smoke = self.result(tier="release", require_engine=True, engine="passed", gated={"Standard": self.THREE})
        self.assertTrue(smoke.smoke_test)
        self.assertEqual(smoke.variants, {})
        self.assertFalse(smoke.ok)
        self.assertTrue(smoke.proof_channel.startswith("DISABLED"))
        text = gate.summarise(smoke)
        self.assertIn(f"NOT gated: Standard - {gate.SMOKE_REASON}", text)
        self.assertIn("this tier requires an engine that came through a named variant: FAILED - it came through none", text)
        # In a tier that runs everything, variants are the gated named variants, as before.
        self.assertEqual(self.release(engine="passed", gated={"Standard": self.THREE}).variants, {"Standard": self.THREE})
        self.assertNotIn("this tier requires", gate.summarise(self.release(engine="passed", gated={"Standard": self.THREE})))

    def test_the_report_of_a_smoke_test_holds_no_gated_ruleset_and_says_why(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(paths, "cache_dir", return_value=Path(folder)):
            smoke = json.loads(gate._write_report(self.result(engine="passed", gated={"Standard": self.THREE}), {})
                               .read_text(encoding="utf-8"))
            self.assertEqual((smoke["tier"], smoke["ok"], smoke["smoke_test"], smoke["gated"]),
                             ("commit", True, gate.SMOKE_TEST, {}))
            self.assertEqual(smoke["ungated"], {"Standard": f"{gate.SMOKE_REASON} ({self.THREE.describe()})"})
            full = json.loads(gate._write_report(self.nightly(engine="passed", gated={"Standard": self.THREE}), {})
                              .read_text(encoding="utf-8"))
            self.assertEqual((full["tier"], full["smoke_test"], full["gated"], full["ungated"]),
                             ("nightly", None, {"Standard": dataclasses.asdict(self.THREE)}, {}))

    def test_a_restricted_run_never_produces_a_tier_verdict(self):
        # H47 (CHECKPOINT 8): a run restricted with --source or --record "never produces a tier verdict; label its output
        # 'restricted - not a tier result'". It reports its findings, and its exit code reflects them only.
        for tier, options in (("release", dict(require_engine=True, full_strength=True)),
                              ("nightly", dict(full_strength=True)), ("commit", {})):
            for engine in ("passed", "not configured"):
                clean = self.result(tier=tier, engine=engine, gated={"Standard": self.THREE},
                                    restricted_to={"sources": None, "records": ["r/1", "r/2"]}, **options)
                clean.ungated = {"Misc - Small": "no oracle followed notation/game as far as the engine went"}
                # Whatever the tier would require of a complete run - an engine, a ruleset gated - it found nothing.
                self.assertTrue(clean.ok, (tier, engine))
                self.assertEqual((clean.verdicts, clean.variants, clean.smoke_test), (({}, {}), {}, False), (tier, engine))
                self.assertEqual(clean.proof_channel, f"DISABLED - {gate.RESTRICTED}")
                text = gate.summarise(clean)
                self.assertIn(f"{gate.RESTRICTED}: tier '{tier}' restricted to records r/1, r/2 - 0 findings", text)
                for verdict in ("oracle layer [", "gated:", "NOT gated:", "SMOKE TEST", "this tier requires", "PASSED",
                                "FAILED", "may be enabled"):
                    self.assertNotIn(verdict, text, (tier, engine))
                self.assertTrue(text.splitlines()[-1] == f"proof channel: DISABLED - {gate.RESTRICTED}", text)
        # Any finding is its exit code - in a section of the oracle layer, or against the engine.
        for found in (dict(findings=[Finding("golden-unreproduced", "ftxi on r/1: ...")]),
                      dict(engine="failed", engine_findings=[Finding("engine-declined", "r/1: ...")])):
            dirty = self.result(restricted_to=self.ONLY_FTXI, **found)
            self.assertFalse(dirty.ok)
            text = gate.summarise(dirty)
            self.assertIn(f"{gate.RESTRICTED}: tier 'commit' restricted to sources ftxi - 1 finding", text)
            self.assertNotIn("oracle layer [", text)
        self.assertIn(f"engine differential ({gate.RESTRICTED}): 1 finding against the engine; a restricted run gates no "
                      "ruleset", text)
        # Its report says so, holds no verdict of a tier, and never stands in for a complete run's in the cache.
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(paths, "cache_dir", return_value=Path(folder)):
            restricted = self.release(engine="passed", gated={"Standard": self.THREE}, restricted_to=self.ONLY_FTXI)
            path = gate._write_report(restricted, {})
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path.name, "gate-release-restricted.json")
            self.assertEqual((report["restricted"], report["restricted_to"], report["ok"], report["oracle_layer_ok"],
                              report["engine"], report["gated"], report["ungated"], report["smoke_test"]),
                             (gate.RESTRICTED, self.ONLY_FTXI, True, None, "compared", {}, {}, None))
            self.assertEqual(report["proof_channel"], f"DISABLED - {gate.RESTRICTED}")
            complete = json.loads(gate._write_report(self.release(engine="passed", gated={"Standard": self.THREE}), {})
                                  .read_text(encoding="utf-8"))
            self.assertEqual((complete["restricted"], complete["restricted_to"], complete["engine"]),
                             (None, None, "passed"))
            self.assertEqual(sorted(Path(folder, "reports").iterdir()),
                             [Path(folder, "reports", "gate-release-restricted.json"),
                              Path(folder, "reports", "gate-release.json")])

    def test_only_a_complete_run_of_a_full_strength_tier_that_requires_an_engine_enables_the_proof_channel(self):
        passed = dict(engine="passed", gated={"Standard": self.THREE, "Very Small - Open": self.ONE})
        channel = self.release(**passed).proof_channel
        self.assertTrue(channel.startswith(
            "may be enabled for: Standard (4 records, 90 positions; an oracle held the engine's boards and pieces "
            "at 88 of them, and its exact state (markers included) at 70; "), channel)
        self.assertIn("independently answered: check 80, count 85, mate 90", channel)
        self.assertIn("at least 3 oracles", channel)
        self.assertIn("Very Small - Open (1 record, 2 positions; an oracle held the engine's boards and pieces at 2 "
                      "of them, and its exact state (markers included) at 2; ", channel)
        # A tier is judged by what it runs, not by what it is called.
        self.assertTrue(self.result(tier="release", require_engine=True, **passed).proof_channel.startswith("DISABLED"))
        self.assertTrue(self.result(tier="nightly", full_strength=True, **passed).proof_channel.startswith("DISABLED"))
        self.assertEqual(self.release(restricted_to=self.ONLY_FTXI, **passed).proof_channel,
                         f"DISABLED - {gate.RESTRICTED}")
        self.assertTrue(self.result(tier="anything", require_engine=True, full_strength=True, **passed)
                        .proof_channel.startswith("may be enabled"))

    def test_custom_positions_never_enable_the_proof_channel(self):
        # A tier that requires an engine requires it to come through a variant a proof could be about.
        only_custom = self.release(engine="passed", gated={CUSTOM_POSITIONS: self.ONE})
        self.assertFalse(only_custom.ok)
        self.assertTrue(only_custom.proof_channel.startswith("DISABLED"))
        # No finding says why the gate failed, so the summary does.
        self.assertIn("this tier requires an engine that came through a named variant: FAILED - it came through none "
                      "(custom positions are evidence about no variant)", gate.summarise(only_custom))
        self.assertTrue(self.nightly(engine="passed", gated={CUSTOM_POSITIONS: self.ONE}).ok)
        both = self.release(engine="passed", gated={CUSTOM_POSITIONS: self.ONE, "Standard": self.ONE})
        self.assertTrue(both.ok)
        self.assertNotIn(CUSTOM_POSITIONS, both.proof_channel)
        self.assertIn(CUSTOM_POSITIONS, gate.summarise(both))  # still reported as gated

    def test_a_required_set_that_is_too_thin_fails_the_gate_not_a_ratio_at_run_time(self):
        # The 90% coverage ratio is gone: verify's static check of the tier's required sets (H32)
        # is a finding of the oracle layer, and any finding fails the gate and keeps the channel shut.
        thin = Finding("proof-channel-vacuous", "tier 'release' requires 10 of chessin5d's 2614 stable recorded values")
        result = self.release(engine="passed", gated={"Standard": self.THREE}, findings=[thin])
        self.assertFalse(result.ok)
        self.assertEqual(result.proof_channel, "DISABLED - the gate failed")
        self.assertFalse(hasattr(gate, "PROOF_CHANNEL_COVERAGE"))

    def test_an_engine_in_parse_mode_gates_nothing_and_never_opens_the_proof_channel(self):
        agreement = gate.ParseAgreement(records=41, read=30, positions=612, described=600, boards=600, states=590,
                                        sides=600, sides_agreed=600, replays=70, replays_agreed=70)
        passed = self.result(engine="passed", mode="parse", parse=agreement)
        self.assertTrue(passed.ok)
        self.assertEqual(passed.proof_channel, "DISABLED - parse mode: the engine was compared on what it set up and how "
                                               "it replayed each record, never on check, counts or mate (H23)")
        text = gate.summarise(passed)
        self.assertIn("engine differential: PARSE MODE - PASSED: the engine read 30 of 41 records (612 positions)", text)
        self.assertTrue(text.splitlines()[-2].startswith("engine differential: PARSE MODE"), text)
        # Not even in the tier that could enable it, and that tier fails without a full engine.
        release = self.release(engine="passed", mode="parse", parse=agreement)
        self.assertFalse(release.ok)
        self.assertTrue(release.proof_channel.startswith("DISABLED - parse mode"))
        self.assertIn("this tier requires an engine compared on the rules, not in parse mode: FAILED", gate.summarise(release))
        failed = self.result(engine="failed", mode="parse", parse=agreement,
                             engine_findings=[Finding("divergence", "r/1 position 3: placement: ...")])
        self.assertFalse(failed.ok)
        self.assertIn("engine differential: PARSE MODE - FAILED", gate.summarise(failed))

    def test_the_summary_counts_what_was_required_and_says_how_slow_the_runner_was(self):
        tally = stability.Tally(values=100, missing=4, outcomes=1, outcomes_missing=1, outside=7,
                                unknown=Counter({"unknown at budget timeout_s=120": 3}))
        probe = gate.Probe(source="ftxi:naive", record="cwmtt/manyChecks", positions=38, seconds=30.5, reference_s=12.2)
        lost = Finding("golden-unreproduced", "ftxi:naive on cwmtt/ctp1: the run was cut at timeout_s=120 ...")
        text = gate.summarise(self.result(findings=[lost], tallies={"ftxi:naive": tally}, calibration=probe))
        self.assertIn("required ftxi:naive: 96 of 100 required values reproduced (4 not produced, 0 produced otherwise), "
                      "0 of 1 required outcomes; 7 more recorded values produced outside the required set (each "
                      "compared); unknown at budget timeout_s=120: 3 recorded values not produced", text)
        self.assertIn("calibration: ftxi:naive over the first 38 positions of cwmtt/manyChecks: 30.50 s on this machine, "
                      "12.20 s on the recording machine - this machine took 2.50x the recording machine's time", text)
        self.assertIn("this runner is 2.5x slower than the recording machine; margin 4", text)
        # Said only where a required value went missing.
        self.assertNotIn("this runner is", gate.summarise(self.result(tallies={"ftxi:naive": tally}, calibration=probe)))

    def test_any_finding_fails_the_gate(self):
        result = self.release(engine="passed", gated={"Standard": self.THREE}, findings=[Finding("x", "y")])
        self.assertFalse(result.ok)
        self.assertFalse(result.oracle_layer_ok)
        self.assertTrue(result.proof_channel.startswith("DISABLED"))
        self.assertIn("FAIL", gate.summarise(result))

    def test_a_failed_engine_differential_fails_the_gate_but_not_the_oracle_layer(self):
        # In every tier: a smoke test (H47) included.
        for result in (self.result(engine="failed", engine_findings=[Finding("divergence", "r/1 position 3: ...")]),
                       self.nightly(engine="failed", engine_findings=[Finding("divergence", "r/1 position 3: ...")])):
            self.assertFalse(result.ok)
            self.assertTrue(result.oracle_layer_ok)
            text = gate.summarise(result)
            self.assertIn(f"oracle layer [{result.tier}]: ok", text)
            self.assertIn("engine differential: SMOKE TEST - FAILED" if result.smoke_test else
                          "engine differential: FAILED", text)
            self.assertEqual([item.code for item in result.findings()], ["divergence"])

    def test_the_summary_says_why_a_ruleset_is_not_gated(self):
        result = self.nightly(engine="passed", gated={"Standard": self.THREE})
        result.ungated = {"Misc - Small": "no oracle followed notation/game as far as the engine went"}
        text = gate.summarise(result)
        self.assertIn("gated:     Standard (4 records, 90 positions; an oracle held the engine's boards and pieces at "
                      "88 of them, and its exact state (markers included) at 70; "
                      "independently answered: check 80, count 85, mate 90; "
                      "each record followed as far as the engine went by at least 3 oracles)", text)
        self.assertIn("NOT gated: Misc - Small - no oracle followed notation/game as far as the engine went", text)



# --- the whole gate, in a world of its own ---------------------------------------------

START = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"
AFTER_E3 = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*"
STATES = [{"event": "state", "index": 0, "size": "8x8", "markers": True, "boards": [f"0:1:w:{START}"]},
          {"event": "state", "index": 1, "size": "8x8", "markers": True,
           "boards": [f"0:1:w:{START}", f"0:1:b:{AFTER_E3}"]}]

# The engine fixture: it prints the lines its plan holds for the record on stdin - and,
# before it does, does to the harness whatever the plan says. A stand-in for the engine
# adapter, in a temporary directory: what it plants is planted on purpose, to show
# that the gate catches it.
ENGINE = r"""
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
record = sys.stdin.buffer.read().decode()
for action in plan.get("tamper", []):
    with open(action["path"], "a", encoding="utf-8") as handle:
        handle.write(action["text"])
for event in plan["records"][record]:
    if event["event"] == "state" and plan.get("stateless"):
        continue
    if "--parse-only" in sys.argv and event["event"] == "position" and not plan.get("answers"):
        event = {**event, "in_check": None, "legal_actions": None, "terminal": None}
    print(json.dumps(event), flush=True)
"""


class Canned(Source):
    """An oracle search that answers from a table, and remembers what it was asked."""

    def __init__(self, name, table, asked):
        self.name, self.oracle = name, name.split(":")[0]
        self.table, self.asked = table, asked

    def available(self):
        return None

    def signature(self, options):
        return {"pin": "canned", "search": self.name, "cap": options.cap}

    def observe(self, record_id, data, options):
        self.asked.append((self.name, record_id, options))
        return dataclasses.replace(self.table[(self.name, record_id)])


class GateRunTests(unittest.TestCase):
    """run_tier end to end: canned oracles, golden files written by the harness's own writers, a real engine process."""

    RECORDS = [record("r/std"), record("r/tz", TZ)]

    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, target in (("GOLDEN_DIR", self.folder / "golden"), ("GATE_FILE", self.folder / "gate.toml"),
                             ("DIVERGENCES_FILE", self.folder / "known-divergences.toml"),
                             ("DISAGREEMENTS_FILE", self.folder / "DISAGREEMENTS.md"),
                             ("ENGINE_BASELINE_FILE", self.folder / "engine-baseline.json")):
            self.enterContext(mock.patch.object(paths, name, target))
        paths.DIVERGENCES_FILE.write_text("schema = 1\n", encoding="utf-8")
        self.gate_toml()
        self.summaries = [fingerprint.parse_state(event).summary() for event in STATES]
        self.asked = []
        self.table = {(name, item.id): self.answer(name, item.id) for name in ORACLE_SOURCES for item in self.RECORDS}
        for target, name, value in ((gate.corpus, "load", self.RECORDS), (gate.corpus, "verify", []),
                                    (gate.corpus, "verify_local", []), (gate.corpus, "verify_excluded", []),
                                    (gate.corpus, "load_excluded", []), (gate.lockfile, "verify", []),
                                    (gate.boundary, "check", []), (gate.provenance, "check", []),
                                    (gate.rulebook, "verify", [])):
            self.enterContext(mock.patch.object(target, name, return_value=value))
        # What every source is given for a record (corpus.feed; runner.observe hands it over).
        self.enterContext(mock.patch.object(gate.corpus, "feed",
                                            side_effect=lambda item, lock: gate.corpus.Feed(data=self.text(item.id).encode())))
        self.enterContext(mock.patch.object(gate, "make_source",
                                            side_effect=lambda name, lock: Canned(name, self.table, self.asked)))
        self.report = self.enterContext(mock.patch.object(gate, "_write_report"))
        self.cfg = config.load()
        self.record_golden()
        # What the honest engine of this world read on its last green run: every record to its end (H47).
        self.record_baseline()

    @staticmethod
    def text(record_id):
        return f'[Board "Standard"]\n{record_id}\n'

    def record_baseline(self, ends=None):
        """This world's engine baseline, written and pinned by the harness's own writer (as baseline --write does).

        ``ends``: record id -> (end, positions); every record not named was read to its end, two positions.
        """
        ends = ends or {}
        reads = {}
        for item in self.RECORDS:
            end, positions = ends.get(item.id, ("ok", 2))
            status, _, reason = end.partition(":")
            reads[item.id] = Observation(source="engine", record=item.id, status=status, reason=reason,
                                         positions=[Position(index) for index in range(positions)],
                                         actions_accepted=max(positions - 1, 0), cap=1000)
        recorded = baseline.from_run("commit", self.RECORDS, reads, {"command": ["engine"], "mode": "full",
                                                                      "sha256": "e" * 64}, recorded="2026-10-17")
        baseline.write(recorded)
        return recorded

    def gate_toml(self, *, command=(), mode="full", naive_timeout=None, edit=None):
        """The repository's gate.toml, with this engine command and mode; ``edit`` reshapes it further (text -> text)."""
        text = (paths.HARNESS_ROOT / "gate.toml").read_text(encoding="utf-8")
        # Whatever command the repository configures, this world runs this one (none by default): a world that kept
        # the real engine would run it on canned records. Every such line is counted, so a second one fails here.
        text, found = re.subn(r"(?m)^command = \[.*\]$", lambda _: f"command = {json.dumps(list(command))}", text)
        self.assertEqual(found, 1, "gate.toml's [engine] command line")
        text = text.replace('mode = "full"', f'mode = "{mode}"')
        text = text.replace('source = "ftxi:naive"\nrecord = "cwmtt/manyChecks"\npositions = 38',
                            'source = "ftxi"\nrecord = "r/std"\npositions = 1')
        if naive_timeout is not None:
            text = text.replace('[tier.commit]\nsources = ["ftxi", "notation"]',
                                '[tier.commit]\nsources = ["ftxi", "ftxi:naive", "notation"]')
            text = text.replace("min_checked = 0.75\n\n# The naive searches",
                                f'min_checked = 0.75\n\n[tier.commit.source_timeout_s]\n"ftxi:naive" = {naive_timeout}\n\n'
                                "# The naive searches", 1)
        if edit is not None:
            text = edit(text)
        paths.GATE_FILE.write_text(text, encoding="utf-8")
        # This world's engine baseline, not the repository's: pinned as it is (none, before setUp records one).
        if config.load(paths.GATE_FILE).baseline is not None:
            recorded = paths.ENGINE_BASELINE_FILE.read_bytes() if paths.ENGINE_BASELINE_FILE.exists() else None
            config.write_sha256(paths.GATE_FILE, "baseline", util.sha256_text(recorded) if recorded is not None else "")
        if (paths.GOLDEN_DIR / "MANIFEST").exists():
            # What each tier requires is a table of DISAGREEMENTS.md, rendered from gate.toml's budgets too. (A test
            # that has broken a golden file on purpose wants its run to find that, not this.)
            with contextlib.suppress(HarnessError):
                gate.render_disagreements(log=lambda _: None)

    def answer(self, name, record_id, *, cut_at=None):
        rows = [W, B] if name not in PARSE_ONLY_SOURCES else [("white", None, None, None), ("black", None, None, None)]
        positions = [Position(index, *row, **summary)
                     for index, (row, summary) in enumerate(zip(rows, self.summaries, strict=True))]
        status, reason = "ok", ""
        if cut_at is not None:
            positions = positions[:cut_at] + [Position(cut_at, **self.summaries[cut_at])]
            status, reason = "timeout", "timeout"
        return Observation(source=name, record=record_id, status=status, reason=reason, positions=positions,
                           actions_accepted=len(positions) - 1, cap=1000,
                           arrival_s=[0.01 * (index + 1) for index in range(len(positions))],
                           ended_s=None if cut_at is not None else 0.05, elapsed_s=0.05)

    def record_golden(self, costs=None, unstable=None):
        """Golden, cost and calibration files and MANIFEST, as golden --write would write them for the canned answers."""
        costs = costs or {}
        unstable = unstable or {}
        for name in ORACLE_SOURCES:
            source = Canned(name, self.table, [])
            answers = {item.id: support.recorded(self.table[(name, item.id)], costs.get((name, item.id)),
                                                 unstable=unstable.get((name, item.id)))
                       for item in self.RECORDS}
            support.write_recorded(name, golden.signature(source, gate.recording_options(self.cfg, source)), self.RECORDS,
                                   answers, manifest=False)
        probe = golden.Calibration(source="ftxi", record="r/std", positions=1, platform="test", cpu_count=1, seconds=0.5,
                                   signature=Canned("ftxi", self.table, []).signature(RunOptions(cap=self.cfg.cap)))
        golden.calibration_path().write_text(golden.dump_calibration(probe), encoding="utf-8", newline="\n")
        support.vouch()
        gate.render_disagreements(log=lambda _: None)

    def engine(self, flips=(), tamper=(), mode="full", naive_timeout=None, declines=False, unreadable=False, edit=None,
               **breach):
        """The engine fixture: it says what the oracles say, except check at each (record, position) in ``flips``.

        ``declines`` makes it decline records as a ruleset it does not implement (allowed by the contract): every
        record (True), or those named. ``unreadable`` makes it refuse every record's first action as text it cannot
        read. ``edit`` reshapes gate.toml (gate_toml). ``breach`` breaks the contract on purpose: answers=True answers
        the rules questions in parse mode too, stateless=True prints no state event.
        """
        plan = {"records": {}, "tamper": list(tamper), **breach}
        for item in self.RECORDS:
            events = []
            for index, state in enumerate(STATES):
                row = (W, B)[index]
                events.append(state)
                events.append({"event": "position", "index": index, "side": row[0],
                               "in_check": (item.id, index) in flips or row[1], "legal_actions": row[2],
                               "terminal": row[3]})
            events.append({"event": "end", "status": "ok", "actions_accepted": 1})
            if declines is True or (declines and item.id in declines):
                events = [{"event": "end", "status": "unsupported", "actions_accepted": 0, "reason": "variant"}]
            elif unreadable:
                events = [*events[:2], {"event": "end", "status": "rejected", "actions_accepted": 0, "reason": "unreadable"}]
            plan["records"][self.text(item.id)] = events
        (self.folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (self.folder / "engine.py").write_text(ENGINE, encoding="utf-8")
        self.gate_toml(command=[sys.executable, str(self.folder / "engine.py"), str(self.folder / "plan.json")], mode=mode,
                       naive_timeout=naive_timeout, edit=edit)

    def run_gate(self, tier="commit", **restricted):
        """``restricted``: only_sources and only_records, as `gate --source` and `--record` pass them."""
        return gate.run_tier(tier, include_untracked=False, log=lambda _: None, **restricted)

    def withhold_check_verdicts(self):
        """No oracle gives a check verdict at position 1 of either record, as recorded: no ruleset reaches the floor
        (75% of the engine's positions) in any tier, although an honest engine agrees with every answer there is."""
        for (name, _), answer in self.table.items():
            if name not in PARSE_ONLY_SOURCES:
                answer.positions[1] = dataclasses.replace(answer.positions[1], in_check=None)
        self.record_golden()

    # A tier by another name that runs every oracle search over the whole corpus, and nightly cut to the commit
    # records (gate_toml's edit): H47 goes by what a tier runs.
    EVERYTHING = (f"\n[tier.everything]\nsources = {json.dumps(list(ORACLE_SOURCES))}\n"
                  f"record_tiers = {json.dumps(list(config.RECORD_TIERS))}\ntimeout_s = 300\nmin_checked = 0.75\n")

    @staticmethod
    def reshaped(text):
        nightly = 'record_tiers = ["commit", "nightly"]\ntimeout_s = 300\n'
        assert text.count(nightly) == 1, "gate.toml's nightly tier is not where the test looks for it"
        return text.replace(nightly, 'record_tiers = ["commit"]\ntimeout_s = 300\n') + GateRunTests.EVERYTHING

    @staticmethod
    def codes(result):
        return sorted({finding.code for finding in result.findings()})

    @staticmethod
    def entry(oracle):
        return (f'\n[[divergence]]\nid = "KD-0001"\noracle = "{oracle}"\npin = "{lockfile.load().pin(oracle).commit}"\n'
                'record = "r/std"\nrecord_sha256 = "s"\nkind = "in_check"\nindex = 1\nengine_value = "True"\n'
                'oracle_value = "False"\nrulebook = "A made-up citation, long enough to pass, for a test of the mechanism."\n'
                'upstream = "unreported: planted by a test"\nadded = "2026-10-01"\n')

    def test_an_honest_engine_passes(self):
        self.engine()
        result = self.run_gate()
        self.assertEqual(result.findings(), [])
        self.assertEqual((result.engine, result.ok), ("passed", True))
        # ftxi, the tier's one rules oracle, is the only one behind each record: a ruleset needs two (H47), and the
        # commit tier is a smoke test that gates nothing anyway.
        self.assertEqual(result.gated, {})
        self.assertEqual(result.ungated, {
            ruleset: f"only one oracle followed {record_id} (ftxi) as far as the engine went - a ruleset is gated only "
                     "when at least 2 answer for each of its records (H47)"
            for ruleset, record_id in (("Standard", "r/std"), (TZ, "r/tz"))})
        self.assertTrue(result.smoke_test)
        self.assertEqual(result.verdicts[0], {})
        # It read both records as it did on the last green run.
        self.assertEqual(result.baseline_note, "2 record(s) compared with what the engine read on the last green run "
                                               "(commit tier, recorded 2026-10-17): 0 declined")
        self.assertIn(f"  baseline:  {result.baseline_note}", gate.summarise(result))
        self.assertEqual(sorted(result.engine_reads), ["r/std", "r/tz"])
        self.assertIn("harness unchanged while the gate ran", [title for title, _ in result.sections])
        self.assertIn("rulebook", [title for title, _ in result.sections])
        self.assertEqual(result.tallies["ftxi"].values, 32)  # two records, two positions, eight values each
        self.assertEqual(result.tallies["ftxi"].outcomes, 2)
        self.assertEqual(result.calibration.seconds, 0.01)  # position 0's answers arrived at 0.01 s
        self.assertEqual(result.calibration.factor, 0.02)
        self.report.assert_called_once()
        self.assertEqual(result.converted, {})
        # The run says which program it compared, as it ran it.
        self.assertEqual((result.engine_run["program"], result.engine_run["file"], result.engine_run["mode"]),
                         (sys.executable, str(Path(sys.executable)), "full"))
        self.assertEqual(result.engine_run["sha256"], util.sha256_file(Path(sys.executable)))

    def test_the_engine_runs_under_its_own_time_limit(self):
        # [engine] timeout_s, or the tier's own timeout_s where that is longer: never the oracles' budget alone, which
        # decides what a tier requires of them (H32) - the gate compares the engine's answers, not its speed.
        limits = []
        run_sources = gate.runner.run_sources

        def spy(sources, records, lock, options_for, **options):
            limits.extend(options_for(source, item).timeout_s for source in sources if source.name == "engine"
                          for item in records)
            return run_sources(sources, records, lock, options_for, **options)

        def engine_timeout(value):
            def edit(text):
                line = '\ntimeout_s = 300\n# "full"'  # [engine] timeout_s, as the repository sets it
                self.assertEqual(text.count(line), 1, "gate.toml's [engine] timeout_s line")
                return text.replace(line, f'\ntimeout_s = {value}\n# "full"')
            return edit

        for value, tier, expected in ((300, "commit", 300.0), (10, "commit", 60.0), (300, "release", 600.0)):
            self.engine(edit=engine_timeout(value))
            limits.clear()
            with mock.patch.object(gate.runner, "run_sources", side_effect=spy):
                result = self.run_gate(tier)
            self.assertEqual((set(limits), result.engine_run["timeout_s"]), ({expected}, expected), (tier, value))
        # The oracles keep their budgets.
        self.assertEqual({options.timeout_s for name, _, options in self.asked if name == "ftxi:naive"}, {300.0})

    def test_a_relative_engine_command_runs_the_repository_roots_program_wherever_the_harness_was_started(self):
        # Windows looks for a relative program from the directory the harness was started in: the gate resolves it
        # against the root first (engine_argv), so this run, started elsewhere, runs - here fails to find - the root's.
        self.gate_toml(command=["./no-such-directory/engine-stand-in"])
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
            result = self.run_gate()
        root = str(paths.REPO_ROOT / "no-such-directory" / "engine-stand-in")
        self.assertEqual((result.engine_run["program"], result.engine_run["file"]), (root, None))
        # Each record fails it (an error is not a decline as well: engine-failed names the record); and it read none.
        self.assertEqual(self.codes(result), ["engine-failed", "engine-read-nothing"])
        self.assertEqual(result.engine, "failed")

    def test_only_a_tier_that_runs_every_oracle_fails_an_engine_that_came_through_no_ruleset(self):
        # H47. No ruleset reaches the floor in any tier, although the engine agrees with every answer there is.
        self.withhold_check_verdicts()
        self.engine()
        # The commit tier is a smoke test: it passes the engine, and says that it gates nothing.
        smoke = self.run_gate("commit")
        self.assertEqual(smoke.findings(), [])
        self.assertEqual((smoke.engine, smoke.ok, smoke.smoke_test, smoke.gated), ("passed", True, True, {}))
        self.assertTrue(smoke.proof_channel.startswith("DISABLED"))
        text = gate.summarise(smoke)
        self.assertIn(f"engine differential: SMOKE TEST - PASSED: {gate.SMOKE_TEST}", text)
        self.assertIn("NOT gated: Standard - only one oracle followed r/std (ftxi) as far as the engine went", text)
        self.assertIs(self.report.call_args.args[0], smoke)
        # A tier that runs every oracle search over the whole corpus fails it, as before.
        for tier in ("nightly", "release"):
            result = self.run_gate(tier)
            self.assertEqual(self.codes(result), ["engine-nothing-gated"], tier)
            self.assertEqual((result.engine, result.ok, result.smoke_test), ("failed", False, False), tier)
            self.assertEqual(result.proof_channel, "DISABLED - the gate failed", tier)
            self.assertIn("engine differential: FAILED", gate.summarise(result), tier)

    def test_an_engine_that_declines_a_record_it_read_on_the_last_green_run_fails_every_tier(self):
        # H47 (CHECKPOINT 8): "the commit tier fails if the engine declines a record it read on the last green run". It
        # declines r/tz, which it read to its end then; it still reads r/std, so it reads something.
        self.engine(declines={"r/tz"})
        for tier in ("commit", "nightly", "release"):
            result = self.run_gate(tier)
            self.assertEqual(self.codes(result), ["engine-declined"], tier)
            self.assertEqual((result.engine, result.ok), ("failed", False), tier)
            [declined] = result.findings()
            self.assertEqual(declined.message,
                             "r/tz: the engine declined its ruleset (no position), where on the last green run (commit "
                             "tier, recorded 2026-10-17) it read it to its end (2 positions) - it declines a record it "
                             "read then (H47)")
            self.assertIn("2 record(s) compared with what the engine read on the last green run (commit tier, recorded "
                          "2026-10-17): 1 declined", gate.summarise(result))
        self.assertIn(f"engine differential: SMOKE TEST - FAILED: {gate.SMOKE_TEST}", gate.summarise(self.run_gate()))
        # A record it did not read on the last green run is not one it can decline: the smoke test passes it.
        self.record_baseline({"r/tz": ("unsupported", 0)})
        self.engine(declines={"r/tz"})
        result = self.run_gate("commit")
        self.assertEqual((result.findings(), result.ok), ([], True))
        self.assertIn("2 record(s) compared with what the engine read on the last green run (commit tier, recorded "
                      "2026-10-17): 0 declined", gate.summarise(result))
        # Nor is a record it now reads further than then: the summary asks for a new baseline, to hold it to that.
        self.record_baseline({"r/tz": ("rejected:unreadable", 1)})
        self.engine()
        result = self.run_gate("commit")
        self.assertEqual((result.findings(), result.ok), ([], True))
        self.assertIn("0 declined; 1 read further than then - record a new baseline (python oracles/harness.py baseline "
                      "--write) to hold the engine to them", result.baseline_note)

    def test_an_engine_that_reads_none_fails_every_tier(self):
        # H47 (CHECKPOINT 8): "... or reads none". Compared with nothing, it contradicts nothing - and fails, the smoke
        # test included, even where the last green run read nothing to decline either.
        self.record_baseline({"r/std": ("unsupported", 0), "r/tz": ("unsupported", 0)})
        self.engine(declines=True)
        for tier in ("commit", "nightly", "release"):
            result = self.run_gate(tier)
            self.assertEqual(self.codes(result), ["engine-read-nothing"], tier)
            self.assertEqual((result.engine, result.ok), ("failed", False), tier)
            self.assertEqual(result.findings()[0].message,
                             "the engine replayed none of the 2 record(s) it was given - none to its end, nor to an "
                             "action it holds illegal (2 unsupported): an engine that reads nothing is compared with "
                             "nothing (H47)")
        # Where the last green run read them, each record is a decline as well.
        self.record_baseline()
        self.engine(declines=True)
        result = self.run_gate("commit")
        self.assertEqual(sorted(finding.code for finding in result.findings()),
                         ["engine-declined", "engine-declined", "engine-read-nothing"])
        # Refusing every record as unreadable reads none either: a refusal as unreadable is no verdict.
        self.record_baseline({"r/std": ("rejected:unreadable", 1), "r/tz": ("rejected:unreadable", 1)})
        self.engine(unreadable=True)
        result = self.run_gate("commit")
        self.assertIn("engine-read-nothing", self.codes(result))
        self.assertIn("(2 rejected:unreadable)", next(finding.message for finding in result.findings()
                                                       if finding.code == "engine-read-nothing"))

    def test_a_tier_is_judged_by_what_it_runs_never_by_its_name(self):
        # H47 is keyed to Tier.full_strength. "everything" runs every oracle search over the whole corpus: it gates, and
        # fails an engine that came through no ruleset. "nightly", cut to the commit records, is a smoke test.
        self.engine(edit=self.reshaped)
        tiers = config.load().tiers
        self.assertEqual((tiers["everything"].full_strength, tiers["nightly"].full_strength), (True, False))
        everything = self.run_gate("everything")
        self.assertEqual((everything.findings(), everything.ok, everything.smoke_test), ([], True, False))
        self.assertEqual(sorted(everything.verdicts[0]), ["Standard", TZ])
        self.assertIn("gated:     Standard (", gate.summarise(everything))
        cut = self.run_gate("nightly")
        self.assertEqual((cut.findings(), cut.ok, cut.smoke_test), ([], True, True))
        self.assertEqual(cut.verdicts[0], {})  # what met the gating conditions there is not gated
        self.assertIn(f"engine differential: SMOKE TEST - PASSED: {gate.SMOKE_TEST}", gate.summarise(cut))
        # Where nothing can be gated: the one fails the engine, the other passes it.
        self.withhold_check_verdicts()
        self.engine(edit=self.reshaped)
        everything = self.run_gate("everything")
        self.assertEqual(self.codes(everything), ["engine-nothing-gated"])
        self.assertEqual((everything.engine, everything.ok), ("failed", False))
        cut = self.run_gate("nightly")
        self.assertEqual((cut.findings(), cut.engine, cut.ok), ([], "passed", True))

    def test_a_restricted_run_never_produces_a_tier_verdict(self):
        # H47 (CHECKPOINT 8): "A restricted run (--source/--record) never produces a tier verdict; label its output
        # 'restricted - not a tier result'". Its exit code reflects its findings only. (This replaces the reading of
        # spec issue 124, under which such a run kept its tier's rule.)
        self.engine()
        restricted = self.run_gate("release", only_sources=["ftxi", "notation"])
        self.assertEqual((restricted.restricted, restricted.restricted_to), (True, {"sources": ["ftxi", "notation"],
                                                                                    "records": None}))
        self.assertEqual((restricted.findings(), restricted.ok, restricted.smoke_test), ([], True, False))
        self.assertEqual((restricted.verdicts, restricted.variants), (({}, {}), {}))
        self.assertEqual(restricted.proof_channel, f"DISABLED - {gate.RESTRICTED}")
        text = gate.summarise(restricted)
        self.assertIn(f"{gate.RESTRICTED}: tier 'release' restricted to sources ftxi, notation - 0 findings", text)
        for verdict in ("oracle layer [", "gated:", "SMOKE TEST", "this tier requires", "PASSED", "FAILED"):
            self.assertNotIn(verdict, text)
        self.assertIs(self.report.call_args.args[0], restricted)  # its report: test_a_restricted_run_never_...(ResultTests)
        # Naming every source of the tier is a restriction all the same.
        self.assertTrue(self.run_gate("commit", only_sources=["ftxi", "notation"]).restricted)
        # A finding is a finding: an unexplained divergence, and a record the engine read on the last green run and
        # now declines - though not reading anything, a check of a tier as a whole, is not looked for.
        self.engine(flips={("r/std", 1)})
        flipped = self.run_gate("commit", only_records=["r/std"])
        self.assertEqual((self.codes(flipped), flipped.ok), (["divergence"], False))
        self.engine(declines=True)
        declined = self.run_gate("commit", only_records=["r/std"])
        self.assertEqual((self.codes(declined), declined.ok), (["engine-declined"], False))
        self.assertIn(f"engine differential ({gate.RESTRICTED}): 1 finding against the engine", gate.summarise(declined))
        # A tier that requires an engine, run restricted with none configured: no verdict, so nothing to fail.
        self.gate_toml()
        unconfigured = self.run_gate("release", only_records=["r/tz"])
        self.assertEqual((unconfigured.engine, unconfigured.findings(), unconfigured.ok), ("not configured", [], True))
        self.assertNotIn("this tier requires an engine", gate.summarise(unconfigured))
        # Where a complete run of the tier fails on a check of the tier as a whole - here nightly's, that some ruleset is
        # gated - a restricted run finds nothing, and exits 0.
        self.withhold_check_verdicts()
        self.engine()
        self.assertEqual(self.codes(self.run_gate("nightly")), ["engine-nothing-gated"])
        quiet = self.run_gate("nightly", only_sources=["ftxi", "cwmtt", "chessin5d"])
        self.assertEqual((quiet.findings(), quiet.ok), ([], True))

    def test_an_unexplained_divergence_fails_every_tier_the_smoke_test_included(self):
        # No known divergence records the flip. (A recorded one fails no tier - test_gate_live; a telltale fails the
        # commit tier too - test_an_engine_that_reproduces_a_recorded_oracle_error_is_caught_where_that_oracle_does_not_run.)
        self.engine(flips={("r/std", 1)})
        for tier in ("commit", "nightly", "release"):
            result = self.run_gate(tier)
            self.assertEqual(self.codes(result), ["divergence"], tier)
            self.assertEqual((result.engine, result.ok), ("failed", False), tier)
            self.assertEqual(result.proof_channel, "DISABLED - the gate failed", tier)
            self.assertIn("engine differential: SMOKE TEST - FAILED" if tier == "commit" else
                          "engine differential: FAILED", gate.summarise(result), tier)

    def test_a_record_read_converted_from_the_export_form_is_named_in_the_log_and_the_report(self):
        # export_form.py (CHECKPOINT 3): every source reads such a record converted, and a run says which were.
        converted = dataclasses.replace(self.RECORDS[0], converted="c" * 64)
        self.RECORDS = [converted, self.RECORDS[1]]
        self.enterContext(mock.patch.object(gate.corpus, "load", return_value=self.RECORDS))
        self.record_golden()  # recorded about the converted text, as golden --write records it
        self.record_baseline()  # and so is the engine's baseline (baseline --write)
        self.engine()
        logged = []
        result = gate.run_tier("commit", include_untracked=False, log=logged.append)
        self.assertEqual(result.findings(), [])
        self.assertEqual(result.converted, {"r/std": converted.reading})
        self.assertIn(f"r/std: {converted.reading}", logged)
        self.assertIs(self.report.call_args.args[0], result)  # the report's "converted" is result.converted

    def test_a_rulebook_that_changed_since_its_pin_fails_every_tier(self):
        # H38: the rulebook is pinned, and the gate is re-run on a change - after the change is reviewed and pinned.
        self.engine()
        changed = Finding("rulebook-changed", "the rulebook changed: re-run the gate and update the pin with ...")
        with mock.patch.object(gate.rulebook, "verify", return_value=[changed]) as verify:
            result = self.run_gate()
        verify.assert_called_once_with(self.cfg.rulebook)
        self.assertEqual(self.codes(result), ["rulebook-changed"])
        self.assertFalse(result.ok)

    def test_an_engine_that_writes_itself_a_known_divergence_while_it_runs_fails(self):
        # It contradicts every oracle about check at r/std position 1, and appends the entry that would excuse it.
        self.engine(flips={("r/std", 1)}, tamper=[{"path": str(paths.DIVERGENCES_FILE), "text": self.entry("ftxi")}])
        result = self.run_gate()
        self.assertFalse(result.ok)
        # The entry it wrote excuses nothing: the run compares against the divergences read when it began.
        self.assertEqual(self.codes(result), ["divergence", "harness-modified"])
        modified = [finding.message for finding in result.findings() if finding.code == "harness-modified"]
        self.assertTrue(any("known-divergences.toml changed while the gate ran" in message for message in modified))

    def test_an_engine_that_edits_a_golden_file_while_it_runs_fails(self):
        self.engine(tamper=[{"path": str(golden.golden_path("ftxi")), "text": " "}])
        result = self.run_gate()
        self.assertFalse(result.ok)
        self.assertEqual(self.codes(result), ["harness-modified"])
        self.assertIn("golden/ftxi.json changed while the gate ran", result.findings()[0].message)

    def test_an_engine_that_edits_its_baseline_while_it_runs_fails(self):
        # The baseline lives in the harness: every gate run hashes it at its start and its end (H33).
        self.engine(declines={"r/tz"}, tamper=[{"path": str(paths.ENGINE_BASELINE_FILE), "text": " "}])
        result = self.run_gate()
        self.assertFalse(result.ok)
        self.assertEqual(self.codes(result), ["engine-declined", "harness-modified"])
        self.assertTrue(any("engine-baseline.json changed while the gate ran" in finding.message
                            for finding in result.findings()))

    def test_a_baseline_that_is_not_the_one_recorded_is_never_compared(self):
        # H33: a recorded answer is verified against its pin before anything is compared with it. Edited by hand - here
        # to hold the engine to nothing on r/tz - it is refused, and the engine's declining r/tz is compared with nothing.
        self.engine(declines={"r/tz"})
        text = paths.ENGINE_BASELINE_FILE.read_text(encoding="utf-8")
        edited = text.replace('"r/tz": {"end": "ok", "positions": 2', '"r/tz": {"end": "unsupported", "positions": 0')
        self.assertNotEqual(edited, text)
        paths.ENGINE_BASELINE_FILE.write_text(edited, encoding="utf-8")
        result = self.run_gate()
        self.assertEqual(self.codes(result), ["engine-baseline-integrity"])
        self.assertIn(gate.BASELINE_SECTION, [title for title, items in result.sections if items])
        self.assertIsNone(result.baseline_note)
        self.assertFalse(result.ok)
        # Gone, with its pin still in gate.toml: missing.
        paths.ENGINE_BASELINE_FILE.unlink()
        self.assertEqual(self.codes(self.run_gate()), ["engine-baseline-missing"])
        # No pin at all, with an engine compared in full mode: missing - the commit tier could not tell a record the
        # engine declines from one it never read (H47). In parse mode, or with no engine, none is needed.
        self.engine()
        self.assertEqual(self.codes(self.run_gate()), ["engine-baseline-missing"])
        self.engine(mode="parse")
        self.assertEqual(self.codes(self.run_gate()), [])
        self.gate_toml()
        self.assertEqual(self.codes(self.run_gate()), [])
        # Recorded for other record content: stale, and the record is compared with nothing.
        self.record_baseline()
        self.RECORDS = [dataclasses.replace(self.RECORDS[0], sha256="t"), self.RECORDS[1]]
        self.enterContext(mock.patch.object(gate.corpus, "load", return_value=self.RECORDS))
        self.record_golden()
        self.engine(declines={"r/std"})
        result = self.run_gate()
        self.assertEqual(self.codes(result), ["engine-baseline-stale"])
        self.assertIn("r/std was recorded for other record content", result.findings()[0].message)
        self.assertEqual(result.baseline_note, "1 record(s) compared with what the engine read on the last green run "
                                               "(commit tier, recorded 2026-10-17): 0 declined")

    def test_baseline_write_records_what_the_engine_read_on_a_green_run_and_pins_it(self):
        # H47, H33: written only by an explicit command, from a green run of the [baseline] tier, then pinned.
        paths.ENGINE_BASELINE_FILE.unlink()
        self.engine()
        self.assertEqual(self.codes(self.run_gate()), ["engine-baseline-missing"])
        result, changed = gate.write_baseline(include_untracked=False, log=lambda _: None)
        self.assertEqual((result.ok, result.restricted, result.tier), (True, False, "commit"))
        self.assertEqual(changed, ["no earlier baseline could be read: all 2 record(s) are new"])
        written = paths.ENGINE_BASELINE_FILE.read_bytes()
        self.assertEqual(config.load().baseline.sha256, util.sha256_text(written))
        recorded = baseline.parse(written)
        self.assertEqual((recorded.tier, recorded.engine["mode"], recorded.records),
                         ("commit", "full", {item.id: baseline.Entry(sha256="s", end="ok", positions=2)
                                             for item in self.RECORDS}))
        # One record per line, so that a re-recording is reviewed as a diff.
        self.assertIn('    "r/std": {"end": "ok", "positions": 2, "sha256": "s"},\n', written.decode("utf-8"))
        passed = self.run_gate()
        self.assertEqual((passed.findings(), passed.ok), ([], True))
        # A record the engine now declines is recorded only from a run green on everything else - and listed as a loss.
        self.engine(declines={"r/tz"})
        self.assertEqual(self.codes(self.run_gate()), ["engine-declined"])
        _, changed = gate.write_baseline(include_untracked=False, log=lambda _: None)
        self.assertEqual(changed, ["LOSS: r/tz - the engine declined its ruleset (no position), where it read it to its "
                                   "end (2 positions)"])
        self.assertEqual(self.codes(self.run_gate()), [])
        # A run that is not green records nothing: a divergence, or an engine that reads none.
        before, pin = paths.ENGINE_BASELINE_FILE.read_bytes(), config.load().baseline.sha256
        for broken in (dict(flips={("r/std", 1)}), dict(declines=True)):
            self.engine(**broken)
            result, changed = gate.write_baseline(include_untracked=False, log=lambda _: None)
            self.assertIsNone(changed, broken)
            self.assertFalse(result.ok, broken)
            self.assertEqual((paths.ENGINE_BASELINE_FILE.read_bytes(), config.load().baseline.sha256), (before, pin))
        # It records the engine in full mode, from a tier gate.toml names: nothing else.
        self.engine(mode="parse")
        with self.assertRaisesRegex(HarnessError, "full mode"):
            gate.write_baseline(include_untracked=False, log=lambda _: None)
        self.gate_toml()
        with self.assertRaisesRegex(HarnessError, "no engine is configured"):
            gate.write_baseline(include_untracked=False, log=lambda _: None)
        self.engine(edit=lambda text: re.sub(r'(?ms)^\[baseline\]\ntier = "commit"\nsha256 = "[0-9a-f]*"\n', "", text))
        self.assertIsNone(config.load().baseline)
        with self.assertRaisesRegex(HarnessError, "no \\[baseline\\] table"):
            gate.write_baseline(include_untracked=False, log=lambda _: None)

    def test_a_golden_file_edited_before_the_run_is_never_compared(self):
        path = golden.golden_path("ftxi")
        path.write_text(path.read_text(encoding="utf-8").replace('"none"', '"stalemate"', 1), encoding="utf-8")
        self.engine()
        result = self.run_gate()
        self.assertFalse(result.ok)
        self.assertIn("golden-integrity", self.codes(result))
        self.assertNotIn("ftxi", result.tallies)  # nothing was compared with it

    def test_an_engine_that_reproduces_a_recorded_oracle_error_is_caught_where_that_oracle_does_not_run(self):
        # The entry says Chess in 5D - which the commit tier does not run - is wrong to say "no check"
        # at r/std position 1. The engine says "no check" there too: a telltale (H34).
        paths.DIVERGENCES_FILE.write_text("schema = 1\n" + self.entry("chessin5d"), encoding="utf-8")
        self.engine()
        result = self.run_gate()
        self.assertEqual(self.codes(result), ["telltale"])
        self.assertFalse(result.ok)
        self.assertIn("the engine reproduces chessin5d's recorded error at r/std position 1 (in_check = False)",
                      result.findings()[0].message)

    def test_the_gate_requires_every_required_value_and_none_that_its_budget_cannot_reach(self):
        # ftxi:naive runs in this commit tier with a 120 s budget, and is cut while it searches r/std position 1.
        self.table[("ftxi:naive", "r/std")] = self.answer("ftxi:naive", "r/std", cut_at=1)
        # Recorded at 2 ms, position 1 is required (0.002 x 4 << 120): losing it fails the gate, by name.
        self.engine(naive_timeout=120)
        result = self.run_gate()
        self.assertEqual(self.codes(result), ["golden-unreproduced"])
        self.assertTrue(all("ftxi:naive on r/std" in finding.message for finding in result.findings()))
        # Recorded at 50 s - beyond 120 / 4, within the release tier's 300 / 4 - it is not required here: it is
        # counted as unknown at the budget.
        self.table[("ftxi:naive", "r/std")] = self.answer("ftxi:naive", "r/std")
        self.record_golden(costs={("ftxi:naive", "r/std"): [0.001, 50.0]})
        self.table[("ftxi:naive", "r/std")] = self.answer("ftxi:naive", "r/std", cut_at=1)
        result = self.run_gate()
        self.assertEqual(result.findings(), [])
        tally = result.tallies["ftxi:naive"]
        self.assertEqual((tally.missing, tally.unknown), (0, Counter({"unknown at budget timeout_s=120": 4})))

    def test_a_run_stops_early_only_where_no_engine_needs_the_rest(self):
        # Position 1 of ftxi:naive cost 50 s on record: at 120 s nothing past position 0 is required.
        self.record_golden(costs={("ftxi:naive", item.id): [0.001, 50.0] for item in self.RECORDS})
        self.gate_toml(naive_timeout=120)
        self.run_gate()
        stops = {(name, record_id): options.stop_after for name, record_id, options in self.asked if name == "ftxi:naive"}
        self.assertEqual(stops, {("ftxi:naive", "r/std"): (0, "position"), ("ftxi:naive", "r/tz"): (0, "position")})
        # Where an engine is compared, every oracle answer may confirm or contradict it: no run is stopped.
        self.asked.clear()
        self.engine(naive_timeout=120)
        self.run_gate()
        self.assertTrue(all(options.stop_after is None for name, _, options in self.asked if name == "ftxi:naive"))

    def test_the_tier_that_can_enable_the_proof_channel_fails_on_a_vacuous_required_set(self):
        # ftxi:naive's r/std position 1 cost 200 s on record: beyond the release tier's 300 / 4. That leaves its
        # required set there 24 of 32 recorded values, below the reviewed share (90%).
        self.record_golden(costs={("ftxi:naive", "r/std"): [0.001, 200.0]})
        release = self.run_gate("release")
        self.assertEqual(self.codes(release), ["proof-channel-vacuous"])
        self.assertIn(gate.PROOF_SECTION, [title for title, items in release.sections if items])
        self.assertTrue(release.proof_channel.startswith("DISABLED"))
        # A tier that can enable nothing is not failed for it; verify only reports it.
        self.assertEqual(self.run_gate("commit").findings(), [])
        found = gate.proof_channel_readiness(self.cfg, self.RECORDS, snapshot.take())
        self.assertEqual([item.code for item in found], ["proof-channel-vacuous"])

    def test_no_oracle_run_is_stopped_where_an_engine_in_parse_mode_is_compared(self):
        self.record_golden(costs={("ftxi:naive", item.id): [0.001, 50.0] for item in self.RECORDS})
        self.engine(mode="parse", naive_timeout=120)
        self.run_gate()
        naive = [options for name, _, options in self.asked if name == "ftxi:naive"]
        self.assertEqual([options.stop_after for options in naive], [None, None])

    def test_an_engine_in_parse_mode_that_breaks_its_contract_fails(self):
        for breach in ({"answers": True}, {"stateless": True}):
            self.engine(mode="parse", **breach)
            result = self.run_gate()
            self.assertEqual((result.engine, result.ok), ("failed", False), breach)
            failed = [finding.message for finding in result.findings() if finding.code == "engine-failed"]
            self.assertTrue(failed and all("engine contract" in message for message in failed), breach)

    def test_a_value_the_recordings_disagreed_about_is_compared_with_nothing(self):
        # ftxi's two recordings of r/std differed about check at position 1: it is unstable, recorded as null. This
        # run says check there, where the engine says no check: that neither fails the engine nor stands behind it.
        recorded = self.answer("ftxi", "r/std")
        recorded.positions[1] = dataclasses.replace(recorded.positions[1], in_check=None)
        self.table[("ftxi", "r/std")] = recorded
        self.record_golden(unstable={("ftxi", "r/std"): {1: {"in_check": [False, True]}}})
        fresh = self.answer("ftxi", "r/std")
        fresh.positions[1] = dataclasses.replace(fresh.positions[1], in_check=True)
        self.table[("ftxi", "r/std")] = fresh
        self.engine()
        result = self.run_gate()
        self.assertEqual(result.findings(), [])
        self.assertEqual(result.engine, "passed")

    def test_a_lowered_budget_is_a_changed_line_of_what_each_tier_requires(self):
        # H32: what a tier requires is a table of DISAGREEMENTS.md, re-rendered by verify from the golden and cost
        # files and gate.toml - a budget lowered (here the commit tier's, 60 s to 30 s) cannot shrink it unseen.
        self.record_golden(costs={("ftxi", item.id): [0.001, 10.0] for item in self.RECORDS})
        self.assertIn("| commit | `ftxi` | timeout_s=60 | 32 | 32 | 2 | 2 | 0 | 0 |",
                      paths.DISAGREEMENTS_FILE.read_text(encoding="utf-8"))
        lowered = paths.GATE_FILE.read_text(encoding="utf-8").replace("timeout_s = 60", "timeout_s = 30", 1)
        paths.GATE_FILE.write_text(lowered, encoding="utf-8")
        sections = dict(gate.static_checks(lockfile.load(), config.load(), self.RECORDS, snapshot.take(),
                                           include_untracked=False))
        self.assertEqual([item.code for item in sections["golden observations"]], ["disagreements-stale"])
        gate.render_disagreements(log=lambda _: None)
        self.assertIn("| commit | `ftxi` | timeout_s=30 | 16 | 32 | 0 | 2 | 0 | 0 |",
                      paths.DISAGREEMENTS_FILE.read_text(encoding="utf-8"))

    def test_an_engine_in_parse_mode_is_compared_on_its_reading_only(self):
        # It would contradict every oracle about check at r/std position 1; in parse mode it answers no check at all.
        self.engine(flips={("r/std", 1)}, mode="parse")
        result = self.run_gate()
        self.assertEqual(result.findings(), [])
        self.assertEqual((result.engine_mode, result.engine, result.gated), ("parse", "passed", {}))
        self.assertEqual(result.parse.read, 2)
        text = gate.summarise(result)
        self.assertIn("engine differential: PARSE MODE - PASSED: the engine read 2 of 2 records (4 positions)", text)
        self.assertTrue(result.proof_channel.startswith("DISABLED - parse mode"))


if __name__ == "__main__":
    unittest.main()
