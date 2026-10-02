import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import golden, integrity, paths, snapshot
from oracle_harness.adapters import ORACLE_SOURCES, RunOptions, Source
from oracle_harness.corpus import CUSTOM_POSITIONS, Record
from oracle_harness.golden import Horizon
from oracle_harness.observations import Observation, Position
from support import recorded, vouch, write_recorded

OPTIONS = RunOptions(cap=1000, timeout_s=300)


def hashed(number: int) -> str:
    return f"{number:032x}"


def described(observation, placements, fingerprints=None, boards=1, pieces=32):
    """The same report, with what the source set up at each position."""
    fingerprints = fingerprints or [None] * len(placements)
    observation.positions = [
        dataclasses.replace(position, placement=hashed(placement), fingerprint=None if marked is None else hashed(marked),
                            boards=boards, pieces=pieces)
        for position, placement, marked in zip(observation.positions, placements, fingerprints, strict=True)]
    return observation


class FakeSource(Source):
    def __init__(self, name, pin="p1", adapter="a1"):
        self.name = name
        self.pin = pin
        self.adapter = adapter

    def signature(self, options):
        return {"pin": self.pin, "cap": options.cap, "adapter_sha256": self.adapter}


def record(name="r/1", sha="a" * 64, board="Standard", fen=False):
    return Record(id=name, source="fake", path=name, blob="b", sha256=sha, size=1, board=board,
                  licence="x", tier="commit", fen=fen)


def obs(source, record_id, rows, status="ok", reason="", accepted=None):
    positions = [Position(index, *row) for index, row in enumerate(rows)]
    return Observation(source=source, record=record_id, status=status, positions=positions, reason=reason,
                       actions_accepted=max(len(positions) - 1, 0) if accepted is None else accepted, cap=1000)


W = ("white", False, 20, "none")
B = ("black", False, 20, "none")
MATE = ("white", True, 0, "checkmate")


class GoldenCase(unittest.TestCase):
    """A golden directory of its own, and the repository's known divergences."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        patcher = mock.patch.object(paths, "GOLDEN_DIR", self.folder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.records = [record("r/1"), record("r/2", "b" * 64)]

    def write(self, source, results, records=None, manifest=True, **classified):
        """Golden and cost files for one source, from {(source, record id): Observation or golden.Recorded}."""
        records = records or self.records
        answers = {record_id: value for (_, record_id), value in results.items()}
        write_recorded(source.name, golden.signature(source, OPTIONS), records, answers, manifest=manifest)
        return golden.golden_path(source.name).read_text(encoding="utf-8")

    def results(self, name, second=(W, B)):
        return {(name, "r/1"): obs(name, "r/1", [W, B, MATE]), (name, "r/2"): obs(name, "r/2", list(second))}

    def write_all(self, reports=None):
        """Golden files for every source: [W, B, MATE] on r/1 and [W, B] on r/2 unless ``reports`` says otherwise."""
        reports = reports or {}
        for name in ORACLE_SOURCES:
            results = self.results(name)
            for (source, record_id), value in reports.items():
                if source == name:
                    kept = isinstance(value, (Observation, golden.Recorded))
                    results[(name, record_id)] = value if kept else obs(name, record_id, value)
            self.write(FakeSource(name), results, manifest=False)
        vouch()

    def loaded(self):
        return {name: golden.load(name) for name in ORACLE_SOURCES}

    def verify(self, *sources, options=OPTIONS, records=None):
        frozen = snapshot.take([source.name for source in sources])
        return golden.verify_static(list(sources), lambda source: options, records or self.records, frozen)


class GoldenFileTests(GoldenCase):
    def test_round_trip(self):
        source = FakeSource("ftxi")
        results = self.results("ftxi")
        described(results[("ftxi", "r/1")], [1, 2, 3], [11, 12, 13])
        text = self.write(source, results)
        loaded = golden.load("ftxi")
        self.assertEqual(loaded.signature, {"pin": "p1", "cap": 1000, "adapter_sha256": "a1", "budget": {"timeout_s": 300}})
        self.assertEqual(loaded.sha256, {"r/1": "a" * 64, "r/2": "b" * 64})
        self.assertEqual(loaded.observations["r/1"].positions, results[("ftxi", "r/1")].positions)
        self.assertEqual(loaded.observations["r/2"].positions, results[("ftxi", "r/2")].positions)
        signature = golden.signature(source, OPTIONS)
        items = {record_id: recorded(value) for (_, record_id), value in results.items()}
        self.assertEqual(golden.dump("ftxi", signature, self.records, items, recordings=2), text)  # deterministic
        self.assertIn('"schema": 3,', text)
        self.assertEqual((loaded.horizon, loaded.unknown, loaded.unstable), ({}, {}, {}))

    def test_the_budget_is_part_of_the_signature(self):
        # H32: answers recorded under one budget are not the answers of another.
        source = FakeSource("ftxi")
        self.write(source, self.results("ftxi"))
        self.assertEqual(self.verify(source), [])
        found = self.verify(source, options=RunOptions(cap=1000, timeout_s=120))
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("recorded under a different budget", found[0].message)

    def test_a_cut_run_a_limit_and_an_instability_are_recorded_as_such(self):
        source = FakeSource("chessin5d")
        cut = obs("chessin5d", "r/1", [W, (None, None, None, None)], status="timeout", reason="timeout")
        limited = obs("chessin5d", "r/2", [W, ("black", False, 20, None)])
        unsteady = dataclasses.replace(limited.positions[0], side=None)
        limited.positions[0] = unsteady
        answers = {("chessin5d", "r/1"): recorded(cut, [0.1, 0.2], horizon=Horizon(1, "timeout_s=300")),
                   ("chessin5d", "r/2"): recorded(limited, [0.1, 0.2],
                                                  unknown={1: {"terminal": "mate_budget_ms=60000"}},
                                                  unstable={0: {"side": ["b", "w"]}})}
        text = self.write(source, answers)
        self.assertIn('      "horizon": {"position": 1, "budget": "timeout_s=300"},\n', text)
        self.assertIn('      "unknown": {"1": {"terminal": "mate_budget_ms=60000"}},\n', text)
        self.assertIn('      "unstable": {"0": {"side": ["b", "w"]}},\n', text)
        loaded = golden.load("chessin5d")
        self.assertEqual(loaded.horizon, {"r/1": Horizon(1, "timeout_s=300")})
        self.assertEqual(loaded.unknown, {"r/2": {1: {"terminal": "mate_budget_ms=60000"}}})
        self.assertEqual(loaded.unstable, {"r/2": {0: {"side": ["black", "white"]}}})  # read back in full
        self.assertEqual(self.verify(source), [])
        costs = golden.load_costs("chessin5d")
        self.assertEqual((costs.records["r/1"].positions, costs.records["r/1"].end), ((0.1, 0.2), None))
        self.assertEqual(costs.records["r/1"].cost, 0.2)

    def test_a_classification_that_does_not_fit_its_row_is_invalid(self):
        source = FakeSource("ftxi")
        cut = obs("ftxi", "r/1", [W, (None, None, None, None)], status="timeout", reason="timeout")
        cases = {
            "no horizon on a cut run": {("ftxi", "r/1"): recorded(cut, [0.1, 0.2])},
            "a horizon somewhere else": {("ftxi", "r/1"): recorded(cut, [0.1, 0.2], horizon=Horizon(0, "timeout_s=300"))},
            "unknown, with a value": {("ftxi", "r/1"): recorded(obs("ftxi", "r/1", [W]), [0.1],
                                                                unknown={0: {"terminal": "mate_budget_ms=1"}})},
            "unstable, with a value": {("ftxi", "r/1"): recorded(obs("ftxi", "r/1", [W]), [0.1],
                                                                 unstable={0: {"side": ["w", "b"]}})},
        }
        for name, answers in cases.items():
            self.write(source, {**self.results("ftxi"), **answers})
            self.assertIn("golden-invalid", [item.code for item in self.verify(source)], name)

    def test_one_position_per_line(self):
        results = self.results("ftxi")
        described(results[("ftxi", "r/1")], [1, 2, 3], [11, 12, None], boards=2, pieces=63)
        text = self.write(FakeSource("ftxi"), results)
        # The answers, then the fingerprint, placement hash, and board and piece counts.
        self.assertIn(f'        ["w", true, 0, "checkmate", null, "{hashed(3)}", 2, 63]\n', text)
        self.assertIn(f'        ["b", false, 20, "none", "{hashed(12)}", "{hashed(2)}", 2, 63],\n', text)
        self.assertIn('        ["b", false, 20, "none", null, null, null, null]\n', text)  # r/2: nothing said

    def test_a_file_recorded_once_is_stale(self):
        # H32: one recording cannot show a value unstable, so none would be excluded - every value would be required.
        source = FakeSource("ftxi")
        answers = {record_id: value for (_, record_id), value in self.results("ftxi").items()}
        write_recorded("ftxi", golden.signature(source, OPTIONS), self.records, answers, recordings=1)
        found = self.verify(source)
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("recorded 1 time(s); the gate needs 2 independent recordings", found[0].message)
        self.assertIn("golden --write --repeat 2", found[0].message)
        # A cost file measured over other recordings than its golden file's does not fit it either.
        write_recorded("ftxi", golden.signature(source, OPTIONS), self.records, answers)
        items = {record_id: recorded(value) for record_id, value in answers.items()}
        golden.cost_path("ftxi").write_text(golden.dump_costs("ftxi", golden.signature(source, OPTIONS), self.records,
                                                              items, recordings=3), encoding="utf-8", newline="\n")
        vouch()
        found = self.verify(source)
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("ftxi.cost.json: its signature or its number of recordings", found[0].message)

    def test_the_file_a_recording_replaces_is_read_leniently_to_say_what_it_lost(self):
        # golden --write compares a new recording with the file on disk (stability.losses): any schema whose rows
        # this harness reads, and no MANIFEST - it is a report, never a comparison.
        self.write(FakeSource("ftxi"), self.results("ftxi"), manifest=False)
        path = golden.golden_path("ftxi")
        old = json.loads(path.read_text(encoding="utf-8"))
        old["schema"] = 2
        path.write_text(json.dumps(old), encoding="utf-8")
        previous = golden.read_previous("ftxi")
        self.assertEqual(previous["r/1"].positions[2].terminal, "checkmate")
        path.write_text("{", encoding="utf-8")
        self.assertIsNone(golden.read_previous("ftxi"))
        self.assertIsNone(golden.read_previous("cwmtt"))  # no file at all

    def test_an_older_schema_is_refused_and_the_message_says_to_re_record(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        path = golden.golden_path("ftxi")
        old = json.loads(path.read_text(encoding="utf-8"))
        old["schema"] = 2
        path.write_text(json.dumps(old), encoding="utf-8")
        vouch()
        with self.assertRaises(golden.StaleSchema) as caught:
            golden.load("ftxi")
        self.assertIn("golden --write --repeat 2", str(caught.exception))
        found = self.verify(FakeSource("ftxi"))
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("golden schema 2", found[0].message)
        self.assertIsNone(snapshot.take(["ftxi"]).recorded("ftxi", "r/1"))  # reported by verify, not here

    def test_an_oracles_own_words_are_not_recorded(self):
        # They quote the record they are about, and no record is redistributed from here.
        results = self.results("ftxi")
        results[("ftxi", "r/2")] = obs("ftxi", "r/2", [W], status="rejected", reason="no_matching_move")
        results[("ftxi", "r/2")].detail = "Invalid move: (0T3)Qd1>>(0T1)d3 of the record"
        text = self.write(FakeSource("ftxi"), results)
        self.assertNotIn("Qd1", text)
        self.assertNotIn("detail", text)
        loaded = golden.load("ftxi").observations["r/2"]
        self.assertEqual((loaded.status, loaded.reason, loaded.actions_accepted, loaded.detail), ("rejected", "no_matching_move", 0, ""))

    def test_colon_in_a_source_name_is_not_a_file_name(self):
        self.assertEqual(golden.golden_path("cwmtt:naive").name, "cwmtt-naive.json")
        self.assertEqual(golden.cost_path("cwmtt:naive").name, "cwmtt-naive.cost.json")

    def test_static_checks_notice_stale_and_incomplete_files(self):
        source = FakeSource("ftxi")
        self.write(source, self.results("ftxi"))

        def codes(*sources, **options):
            return sorted({item.code for item in self.verify(*sources, **options)})

        self.assertEqual(codes(source), [])
        self.assertEqual(codes(FakeSource("ftxi", pin="p2")), ["golden-stale"])
        self.assertEqual(codes(source, options=RunOptions(cap=50, timeout_s=300)), ["golden-stale"])
        extra = self.records + [record("r/3", "c" * 64)]
        self.assertEqual(codes(source, records=extra), ["golden-incomplete"])
        self.assertEqual(codes(source, records=self.records[:1]), ["golden-orphan"])
        changed = [self.records[0], record("r/2", "f" * 64)]
        self.assertEqual(codes(source, records=changed), ["golden-stale"])
        self.assertEqual(codes(FakeSource("cwmtt")), ["golden-missing"])

    def test_answers_recorded_by_other_adapter_code_are_stale_and_the_message_says_what_changed(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        found = self.verify(FakeSource("ftxi", adapter="a2"))
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("recorded under a different adapter_sha256 -", found[0].message)
        found = self.verify(FakeSource("ftxi", pin="p2", adapter="a2"), options=RunOptions(cap=50, timeout_s=300))
        self.assertIn("recorded under a different adapter_sha256, cap, pin -", found[0].message)

    def test_a_golden_file_that_contradicts_itself_is_invalid(self):
        results = self.results("ftxi")
        results[("ftxi", "r/2")] = obs("ftxi", "r/2", [W, B], accepted=7)
        self.write(FakeSource("ftxi"), results)
        self.assertEqual([item.code for item in self.verify(FakeSource("ftxi"))], ["golden-invalid"])

    def test_costs_that_do_not_fit_the_answers_are_invalid(self):
        results = self.results("ftxi")
        results[("ftxi", "r/2")] = recorded(obs("ftxi", "r/2", [W, B]), [0.1])  # one cost for two positions
        self.write(FakeSource("ftxi"), results)
        self.assertEqual([item.code for item in self.verify(FakeSource("ftxi"))], ["golden-invalid"])

    def test_a_malformed_file_is_an_error_not_a_crash(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        path = golden.golden_path("ftxi")
        path.write_text(path.read_text(encoding="utf-8").replace('"status": "ok",', '"status_": "ok",', 1), encoding="utf-8")
        vouch()
        with self.assertRaises(golden.HarnessError):
            golden.load("ftxi")
        self.assertEqual([item.code for item in self.verify(FakeSource("ftxi"))], ["golden-missing"])

    def test_costs_are_rounded_up_to_two_significant_figures(self):
        cases = {0.01234: 0.013, 123.4: 130.0, 0.5: 0.5, 2.0: 2.0, 2.01: 2.1, 99.1: 100.0, 0.0: 0.0, 1e-7: 1e-7}
        for seconds, expected in cases.items():
            self.assertEqual(golden.round_up(seconds), expected, seconds)
            self.assertGreaterEqual(golden.round_up(seconds), seconds)
        self.assertIsNone(golden.round_up(None))


class IntegrityOnLoadTests(GoldenCase):
    """H33: a recorded answer is read only once its bytes are shown to be the ones MANIFEST commits to."""

    def test_a_file_changed_after_it_was_written_is_refused(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        path = golden.golden_path("ftxi")
        path.write_text(path.read_text(encoding="utf-8").replace('"none"', '"stalemate"', 1), encoding="utf-8")
        with self.assertRaises(integrity.IntegrityError) as raised:
            golden.load("ftxi")
        self.assertIn("does not match its hash in MANIFEST", str(raised.exception))
        found = self.verify(FakeSource("ftxi"))
        self.assertEqual(sorted({item.code for item in found}), ["golden-integrity"])
        # A cost file is held to it the same way.
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        golden.cost_path("ftxi").write_text("{}", encoding="utf-8")
        self.assertIn("golden-integrity", [item.code for item in self.verify(FakeSource("ftxi"))])

    def test_line_endings_alone_do_not_count_as_a_change(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        path = golden.golden_path("ftxi")
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(len(golden.load("ftxi").observations), 2)

    def test_without_a_manifest_nothing_is_read(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"), manifest=False)
        with self.assertRaises(integrity.IntegrityError):
            golden.load("ftxi")
        self.assertIn("golden-integrity", [item.code for item in self.verify(FakeSource("ftxi"))])

    def test_a_file_nobody_recorded_is_found(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        (self.folder / "planted.json").write_text("{}", encoding="utf-8")
        found = self.verify(FakeSource("ftxi"))
        self.assertEqual([(item.code, "planted.json" in item.message) for item in found], [("golden-integrity", True)])


class DisagreementTests(GoldenCase):
    def test_disagreements_are_derived_from_the_golden_files(self):
        self.write_all({("chessin5d", "r/2"): [W, ("black", True, 20, "none")]})
        loaded = self.loaded()
        found = golden.disagreements(self.records, loaded)
        self.assertEqual({(item.a, item.b) for item in found},
                         {tuple(sorted((name, "chessin5d"), key=ORACLE_SOURCES.index)) for name in ORACLE_SOURCES
                          if name != "chessin5d"})
        self.assertTrue(all((item.record, item.kind, item.index) == ("r/2", "in_check", 1) for item in found))
        text = golden.render_disagreements(self.records, loaded)
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 1 |", text)
        self.assertIn("### `r/2`", text)
        self.assertNotIn("### `r/1`", text)
        self.assertEqual(text, golden.render_disagreements(self.records, loaded))

    def test_what_each_tier_requires_is_a_section_of_its_own(self):
        self.write_all()
        section = ["## What each tier requires", "", "| commit | `ftxi` | timeout_s=60 | 1 | 1 | 1 | 1 | 0 | 0 |"]
        text = golden.render_disagreements(self.records, self.loaded(), section)
        self.assertIn("\n## What each tier requires\n\n| commit | `ftxi` | timeout_s=60 | 1 | 1 | 1 | 1 | 0 | 0 |\n\n"
                      "## Which rulesets each oracle replays\n", text)
        self.assertNotIn("## What each tier requires", golden.render_disagreements(self.records, self.loaded()))

    def test_how_a_record_is_replayed_is_compared_once_per_pair_of_implementations(self):
        # ftxi cannot match the move text of action 1 of r/2; everyone else plays it.
        stuck = {(name, "r/2"): obs(name, "r/2", [W], status="rejected", reason="no_matching_move")
                 for name in ("ftxi", "ftxi:naive")}
        self.write_all(stuck)
        found = [item for item in golden.disagreements(self.records, self.loaded()) if item.kind == "legality"]
        self.assertEqual([(item.a, item.b, item.soft) for item in found],
                         [("ftxi", "cwmtt", True), ("ftxi", "chessin5d", True), ("ftxi", "notation", True)])

    def test_the_report_says_how_much_was_actually_compared(self):
        capped = ("white", False, 1000, "none")
        counts_only = [(None, None, 20, "none"), (None, None, 1000, "none"), (None, None, 0, "no_action")]
        self.write_all({
            ("ftxi", "r/1"): [W, capped, MATE],
            ("cwmtt", "r/1"): counts_only,
            ("ftxi:naive", "r/1"): recorded(obs("ftxi:naive", "r/1", [W], status="timeout", reason="timeout"), [0.1],
                                            horizon=Horizon(0, "timeout_s=300")),
        })
        text = golden.render_disagreements(self.records, self.loaded())
        # ftxi and cwmtt both counted three positions of r/1 and two of r/2: four counts
        # are equal and one pair is equal only in having stopped at the cap. Neither
        # said what it set up, so none of the five is known to be the same position.
        self.assertIn("| `ftxi` vs `cwmtt` | legal_actions | 5 | 4 | 1 | 0 | 0 | 0 | 5 |", text)
        # On the end of r/1 cwmtt says only that nothing can be played.
        self.assertIn("| `ftxi` vs `cwmtt` | terminal | 5 | 4 | 1 | 0 | 0 | 0 | 5 |", text)
        # Two searches of one implementation are marked as such; the timed-out one compared one position.
        self.assertIn("| `ftxi` vs `ftxi:naive` (one implementation) | in_check | 3 | 3 | 0 | 0 | 0 | 0 | 3 |", text)
        self.assertIn("No two sources said what they set up at the same position: nothing was compared.", text)

    def test_the_report_counts_what_was_cut_left_unknown_or_unstable(self):
        limited = obs("chessin5d", "r/2", [W, ("black", False, 20, None)])
        unsteady = obs("ftxi", "r/2", [W, ("black", None, 20, "none")])
        self.write_all({
            ("chessin5d", "r/2"): recorded(limited, unknown={1: {"terminal": "mate_heap_mb=1024"}}),
            ("ftxi", "r/2"): recorded(unsteady, unstable={1: {"in_check": [False, True]}}),
        })
        text = golden.render_disagreements(self.records, self.loaded())
        self.assertIn("| `chessin5d` | 2 | 0 | 0 | 0 | 0 | 5 | 19 | 1 | 0 | 0 | 0 |", text)
        self.assertIn("| `ftxi` | 2 | 0 | 0 | 0 | 0 | 5 | 19 | 0 | 1 | 0 | 0 |", text)

    def described_all(self, changes=None):
        """Golden files in which every source also says what it set up, the same everywhere.

        ``changes`` maps (source, record id) to the (rows, placements, fingerprints) of that report instead.
        """
        changes = changes or {}
        reports = {}
        for name in ORACLE_SOURCES:
            for record_id, usual_rows, usual_placements in (("r/1", [W, B, MATE], [1, 2, 3]), ("r/2", [W, B], [1, 4])):
                usual = (usual_rows, usual_placements, [placement + 10 for placement in usual_placements])
                rows, placements, fingerprints = changes.get((name, record_id), usual)
                reports[(name, record_id)] = described(obs(name, record_id, rows), placements, fingerprints)
        self.write_all(reports)

    def test_aligned_positions_are_counted_where_both_said_what_they_set_up(self):
        self.described_all()
        text = golden.render_disagreements(self.records, self.loaded())
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 5 | 5 | 0 | 0 | 5 | 0 | 0 |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | 5 | 5 | 5 | 5 |", text)  # both described, same boards, both marked, same markers
        self.assertIn("## Parse divergences", text)
        self.assertIn("None.", text)
        self.assertIn("| `ftxi` | 2 | 0 | 0 | 0 | 0 | 5 | 20 | 0 | 0 | 5 | 5 |", text)  # described, with markers

    def test_a_parse_divergence_is_listed_first_and_hides_what_it_makes_meaningless(self):
        # chessin5d reads r/2's first action as another position - and says, there,
        # that black is in check. That is not a check disagreement: it is about another position.
        self.described_all({("chessin5d", "r/2"): ([W, ("black", True, 20, "none")], [1, 9], [11, 19])})
        loaded = self.loaded()
        text = golden.render_disagreements(self.records, loaded)
        found = golden.disagreements(self.records, loaded)
        self.assertTrue(found)
        self.assertEqual({(item.kind, item.index, "chessin5d" in (item.a, item.b)) for item in found}, {("placement", 1, True)})
        parse = text.index("## Parse divergences")
        self.assertLess(parse, text.index("## Summary"))
        self.assertIn(f"| `r/2` | `ftxi` vs `chessin5d` | 1 | placement | `{hashed(4)}` (1 board, 32 pieces) "
                      f"| `{hashed(9)}` (1 board, 32 pieces) | 1 |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | placement | 1 |", text)  # in the summary
        self.assertNotIn("## Detail", text)  # no rules disagreement is left to detail
        # Position 1 of r/2 is not compared, so it is not counted as compared either.
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 4 | 4 | 0 | 0 | 4 | 0 | 0 |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | 5 | 4 | 4 | 4 |", text)

    def test_a_marker_only_divergence_is_listed_and_hides_nothing(self):
        # Same pieces at r/1 position 1, marked differently - and a check disagreement there, still compared.
        self.described_all({("chessin5d", "r/1"): ([W, ("black", True, 20, "none"), MATE], [1, 2, 3], [11, 99, 13])})
        loaded = self.loaded()
        text = golden.render_disagreements(self.records, loaded)
        self.assertIn(f"| `r/1` | `ftxi` vs `chessin5d` | 1 | marker-only | `{hashed(12)}` (1 board, 32 pieces) "
                      f"| `{hashed(99)}` (1 board, 32 pieces) | - |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 5 | 4 | 0 | 1 | 4 | 1 | 0 |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | 5 | 5 | 5 | 4 |", text)
        self.assertIn("- position 1, in_check: `ftxi` = False; `chessin5d` = True", text)  # in the rules detail
        self.assertEqual(text, golden.render_disagreements(self.records, loaded))  # deterministic

    def test_searches_of_one_implementation_must_set_up_the_same_positions(self):
        self.described_all({("ftxi:naive", "r/1"): ([W, B, MATE], [1, 2, 5], [11, 12, 15])})
        kinds = {(item.a, item.b, item.kind) for item in golden.disagreements(self.records, self.loaded())}
        self.assertIn(("ftxi", "ftxi:naive", "placement"), kinds)

    def test_the_report_files_records_by_ruleset(self):
        self.records = [record("r/1"), record("r/2", "b" * 64, board="Standard - Reversed Royalty", fen=True)]
        self.write_all({("chessin5d", "r/2"): obs("chessin5d", "r/2", [], status="unsupported", reason="fen")})
        text = golden.render_disagreements(self.records, self.loaded())
        header = next(line for line in text.splitlines() if line.startswith("| Ruleset |"))
        self.assertTrue(header.endswith("| Positions | check | counts | mate |"))
        # ruleset, records, records replayed by each source, positions, then positions with each verdict
        replayed = " | ".join("0" if name == "chessin5d" else "1" for name in ORACLE_SOURCES)
        self.assertIn(f"| {CUSTOM_POSITIONS} | 1 | {replayed} | 2 | 2 | 2 | 2 |", text)
        self.assertIn(f"| Standard | 1 | {' | '.join('1' for _ in ORACLE_SOURCES)} | 3 | 3 | 3 | 3 |", text)
        self.assertNotIn("| Standard - Reversed Royalty |", text)

    def test_a_snapshot_reads_the_files_once_and_survives_a_missing_one(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        frozen = snapshot.take(["ftxi", "cwmtt"])
        self.assertEqual(len(frozen.recorded("ftxi", "r/1").positions), 3)
        self.assertIsNone(frozen.recorded("ftxi", "r/9"))
        self.assertIsNone(frozen.recorded("cwmtt", "r/1"))  # no such file: reported by verify, not here
        self.assertEqual([item.code for item in frozen.problems["cwmtt"]], ["golden-missing"])
        # Changing the file afterwards changes nothing the snapshot holds.
        golden.golden_path("ftxi").write_text("{}", encoding="utf-8")
        self.assertEqual(len(frozen.recorded("ftxi", "r/1").positions), 3)


if __name__ == "__main__":
    unittest.main()
