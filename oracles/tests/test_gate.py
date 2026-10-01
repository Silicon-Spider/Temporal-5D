import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import gate, paths
from oracle_harness.adapters import ORACLE_SOURCES, PARSE_ONLY_SOURCES, EngineSource, Source
from oracle_harness.corpus import CUSTOM_POSITIONS, Record
from oracle_harness.divergences import Divergence
from oracle_harness.gate import Gated, GateResult
from oracle_harness.observations import Observation, Position
from oracle_harness.util import Finding

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

    def run_differential(self, engine_rows=None, oracle_reports=None, entries=(), min_checked=1.0):
        """Every source says [W, B] about every record unless told otherwise."""
        engine_rows = engine_rows or {}
        oracle_reports = oracle_reports or {}
        results, engine_results = {}, {}
        for item in self.records:
            for oracle in self.oracles:
                value = oracle_reports.get((oracle.name, item.id), [W, B])
                results[(oracle.name, item.id)] = value if isinstance(value, Observation) else obs(oracle.name, item.id, value)
            value = engine_rows.get(item.id, [W, B])
            engine_results[("engine", item.id)] = value if isinstance(value, Observation) else obs("engine", item.id, value)
        return gate.engine_differential(self.engine, self.oracles, self.records, results, engine_results,
                                        list(entries), min_checked)

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
        findings, _, _ = self.run_differential(entries=[self.divergence()])
        self.assertEqual([item.code for item in findings], ["divergence-stale"])

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
        # One oracle refuses the same action as illegal: the engine's refusal is confirmed.
        agreeing = dict(stopped)
        agreeing[("ftxi", "r/tz")] = obs("ftxi", "r/tz", [W], status="rejected", reason="cannot_submit")
        findings, gated, _ = self.run_differential({"r/tz": refused}, agreeing)
        self.assertEqual(findings, [])
        self.assertEqual(gated[TZ].oracles, 1)

    def test_a_ruleset_whose_only_oracle_has_no_check_verdict_is_not_gated(self):
        # cwmtt is the only oracle that knows several variants, and it says
        # nothing about check. Agreeing with it about the counts is not a pass.
        self.oracles = [Named("ftxi"), Named("cwmtt"), Named("chessin5d")]
        reports = self.silence("ftxi", "chessin5d")
        reports[("cwmtt", "r/tz")] = obs("cwmtt", "r/tz", [COUNT, COUNT])
        findings, gated, ungated = self.run_differential(oracle_reports=reports, min_checked=0.75)
        self.assertEqual(findings, [])
        self.assertEqual(list(gated), ["Standard"])
        self.assertEqual(ungated, {TZ: "the oracles give a check verdict at only 0 of the engine's 2 positions (the floor is 75%)"})

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
        # r/b is read by nobody who gives a check verdict.
        self.oracles = [Named("ftxi"), Named("cwmtt")]
        reports = {("ftxi", "r/b"): obs("ftxi", "r/b", [], status="unsupported", reason="notation"),
                   ("cwmtt", "r/b"): obs("cwmtt", "r/b", [COUNT, COUNT])}
        engine = {"r/a": [W, B, W, B, W, B]}
        _, gated, ungated = self.run_differential(engine, {**reports, ("ftxi", "r/a"): [W, B, W, B, W, B],
                                                           ("cwmtt", "r/a"): [COUNT] * 6}, min_checked=0.75)
        self.assertEqual(gated[TZ], Gated(records=2, positions=8, checked={"in_check": 6, "legal_actions": 8, "terminal": 8}, oracles=1))
        _, gated, ungated = self.run_differential(engine, {**reports, ("ftxi", "r/a"): [W, B, W, B, W, B],
                                                           ("cwmtt", "r/a"): [COUNT] * 6}, min_checked=0.8)
        self.assertEqual(gated, {})
        self.assertIn("check verdict at only 6 of the engine's 8 positions", ungated[TZ])

    def test_the_count_is_the_weakest_record_of_the_ruleset(self):
        self.records.append(record("r/std2"))
        reports = self.silence("chessin5d", record_id="r/std2", reason="notation")
        _, gated, _ = self.run_differential(oracle_reports=reports)
        self.assertEqual(gated["Standard"].oracles, 1)
        self.assertEqual(gated["Standard"].records, 2)

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
        findings, gated, _ = self.run_differential({"r/std": [W, B, W]}, stuck, [entry])
        self.assertEqual(findings, [])
        self.assertEqual(gated["Standard"].oracles, 1)  # only Chess in 5D went as far as the engine

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
        # count there is about another position, and Chess in 5D gives no count:
        # position 1's count stands unchecked.
        engine = {"r/tz": described(obs("engine", "r/tz", [W, B]), [1, 2])}
        reports = {**self.everyone("r/tz", [W, B], [1, 3], names=("ftxi", "ftxi:naive")),
                   **self.everyone("r/tz", [W_VERDICT, B_VERDICT], [1, 2], names=("chessin5d",))}
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
        self.assertEqual(ungated, {TZ: "no oracle followed r/tz as far as the engine went"})
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


class WriteGoldenTests(unittest.TestCase):
    class Signed(Named):
        def __init__(self, name, drifts):
            super().__init__(name)
            self.drifts, self.looks = drifts, 0

        def available(self):
            return None

        def signature(self, options):
            self.looks += 1
            return {"adapter_sha256": str(self.looks) if self.drifts else "same"}

    def write(self, source):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for target, name, value in (
                (gate.lockfile, "verify", []), (gate.corpus, "load", [record("r/1")]), (gate.corpus, "verify", []),
                (gate, "make_source", source),
                (gate.runner, "run_sources", {(source.name, "r/1"): obs(source.name, "r/1", [W])}),
                (gate, "render_disagreements", None)):
            self.enterContext(mock.patch.object(target, name, return_value=value))
        self.enterContext(mock.patch.object(paths, "REPO_ROOT", directory))
        self.enterContext(mock.patch.object(paths, "GOLDEN_DIR", directory / "golden"))
        return [finding.code for finding in gate.write_golden(only_sources=[source.name], log=lambda _: None)], directory

    def test_the_answers_are_signed_by_the_code_that_produced_them(self):
        codes, directory = self.write(self.Signed("ftxi", drifts=False))
        self.assertEqual(codes, [])
        self.assertIn('"adapter_sha256": "same"', (directory / "golden" / "ftxi.json").read_text(encoding="utf-8"))

    def test_nothing_is_written_when_the_code_changed_while_the_oracles_ran(self):
        codes, directory = self.write(self.Signed("ftxi", drifts=True))
        self.assertEqual(codes, ["golden-code-changed"])
        self.assertFalse((directory / "golden").exists())


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


class ResultTests(unittest.TestCase):
    ONE = Gated(records=1, positions=2, checked=FULL, oracles=1, aligned=2, matched=2)
    THREE = Gated(records=4, positions=90, checked={"in_check": 80, "legal_actions": 85, "terminal": 90}, oracles=3,
                  aligned=88, matched=70)

    def result(self, tier="commit", engine="not configured", findings=(), engine_findings=(), gated=None, partial=False,
               require_engine=False, full_strength=False, coverage=None):
        result = GateResult(tier=tier, engine=engine, gated=dict(gated or {}), partial=partial,
                            require_engine=require_engine, full_strength=full_strength,
                            coverage=dict(coverage if coverage is not None else {name: (10, 10) for name in ORACLE_SOURCES}))
        result.sections.append(("section", list(findings)))
        if engine != "not configured":
            result.engine_sections.append(("engine vs oracles", list(engine_findings)))
        return result

    def release(self, **options):
        return self.result(tier="release", require_engine=True, full_strength=True, **options)

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

    def test_an_engine_that_came_through_nothing_has_not_passed(self):
        # "Passed" with no ruleset gated would be a pass on the strength of nothing.
        self.assertFalse(self.release(engine="passed").ok)
        self.assertFalse(self.result(engine="passed").ok)
        self.assertTrue(self.result(engine="passed", gated={"Standard": self.ONE}).ok)

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
        self.assertTrue(self.release(partial=True, **passed).proof_channel.startswith("DISABLED"))
        self.assertTrue(self.result(tier="anything", require_engine=True, full_strength=True, **passed)
                        .proof_channel.startswith("may be enabled"))

    def test_custom_positions_never_enable_the_proof_channel(self):
        # A tier that requires an engine requires it to come through a variant a proof could be about.
        only_custom = self.release(engine="passed", gated={CUSTOM_POSITIONS: self.ONE})
        self.assertFalse(only_custom.ok)
        self.assertTrue(only_custom.proof_channel.startswith("DISABLED"))
        self.assertTrue(self.result(engine="passed", gated={CUSTOM_POSITIONS: self.ONE}).ok)
        both = self.release(engine="passed", gated={CUSTOM_POSITIONS: self.ONE, "Standard": self.ONE})
        self.assertTrue(both.ok)
        self.assertNotIn(CUSTOM_POSITIONS, both.proof_channel)
        self.assertIn(CUSTOM_POSITIONS, gate.summarise(both))  # still reported as gated

    def test_a_source_that_was_listed_but_not_heard_keeps_the_proof_channel_shut(self):
        # A tier can list every source and still starve one of time; what counts is what it answered.
        passed = dict(engine="passed", gated={"Standard": self.THREE})
        heard = {name: (95, 100) for name in ORACLE_SOURCES}
        self.assertTrue(self.release(coverage=heard, **passed).proof_channel.startswith("may be enabled"))
        starved = {**heard, "chessin5d": (0, 2614)}
        channel = self.release(coverage=starved, **passed).proof_channel
        self.assertTrue(channel.startswith("DISABLED - chessin5d reproduced less than 90%"), channel)
        missing = {name: stats for name, stats in heard.items() if name != "cwmtt:naive"}
        self.assertIn("cwmtt:naive", self.release(coverage=missing, **passed).proof_channel)

    def test_any_finding_fails_the_gate(self):
        result = self.release(engine="passed", gated={"Standard": self.THREE}, findings=[Finding("x", "y")])
        self.assertFalse(result.ok)
        self.assertFalse(result.oracle_layer_ok)
        self.assertTrue(result.proof_channel.startswith("DISABLED"))
        self.assertIn("FAIL", gate.summarise(result))

    def test_a_failed_engine_differential_fails_the_gate_but_not_the_oracle_layer(self):
        result = self.result(engine="failed", engine_findings=[Finding("divergence", "r/1 position 3: ...")])
        self.assertFalse(result.ok)
        self.assertTrue(result.oracle_layer_ok)
        text = gate.summarise(result)
        self.assertIn("oracle layer [commit]: ok", text)
        self.assertIn("engine differential: FAILED", text)
        self.assertEqual([item.code for item in result.findings()], ["divergence"])

    def test_the_summary_says_why_a_ruleset_is_not_gated(self):
        result = self.result(engine="passed", gated={"Standard": self.THREE})
        result.ungated = {"Misc - Small": "no oracle followed notation/game as far as the engine went"}
        text = gate.summarise(result)
        self.assertIn("gated:     Standard (4 records, 90 positions; an oracle held the engine's boards and pieces at "
                      "88 of them, and its exact state (markers included) at 70; "
                      "independently answered: check 80, count 85, mate 90; "
                      "each record followed as far as the engine went by at least 3 oracles)", text)
        self.assertIn("NOT gated: Misc - Small - no oracle followed notation/game as far as the engine went", text)


if __name__ == "__main__":
    unittest.main()
