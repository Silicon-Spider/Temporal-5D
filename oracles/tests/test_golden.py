import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import golden, paths
from oracle_harness.adapters import ORACLE_SOURCES, RunOptions, Source
from oracle_harness.corpus import CUSTOM_POSITIONS, Record
from oracle_harness.observations import Observation, Position

OPTIONS = RunOptions(cap=1000)


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


class GoldenFileTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = mock.patch.object(paths, "GOLDEN_DIR", Path(folder.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.records = [record("r/1"), record("r/2", "b" * 64)]

    def write(self, source, results, records=None):
        records = records or self.records
        text = golden.dump(source, OPTIONS, records, results)
        golden.golden_path(source.name).write_text(text, encoding="utf-8")
        return text

    def results(self, name, second=(W, B)):
        return {(name, "r/1"): obs(name, "r/1", [W, B, MATE]), (name, "r/2"): obs(name, "r/2", list(second))}

    def write_all(self, reports=None):
        """Golden files for every source: [W, B, MATE] on r/1 and [W, B] on r/2 unless ``reports`` says otherwise."""
        reports = reports or {}
        for name in ORACLE_SOURCES:
            results = self.results(name)
            for (source, record_id), value in reports.items():
                if source == name:
                    results[(name, record_id)] = value if isinstance(value, Observation) else obs(name, record_id, value)
            self.write(FakeSource(name), results)

    def test_round_trip(self):
        source = FakeSource("ftxi")
        results = self.results("ftxi")
        described(results[("ftxi", "r/1")], [1, 2, 3], [11, 12, 13])
        text = self.write(source, results)
        loaded = golden.load("ftxi")
        self.assertEqual(loaded.signature, {"pin": "p1", "cap": 1000, "adapter_sha256": "a1"})
        self.assertEqual(loaded.sha256, {"r/1": "a" * 64, "r/2": "b" * 64})
        self.assertEqual(loaded.observations["r/1"].positions, results[("ftxi", "r/1")].positions)
        self.assertEqual(loaded.observations["r/2"].positions, results[("ftxi", "r/2")].positions)
        self.assertEqual(golden.dump(source, OPTIONS, self.records, results), text)  # deterministic
        self.assertIn('"schema": 2,', text)

    def test_one_position_per_line(self):
        results = self.results("ftxi")
        described(results[("ftxi", "r/1")], [1, 2, 3], [11, 12, None], boards=2, pieces=63)
        text = self.write(FakeSource("ftxi"), results)
        # The answers, then the fingerprint, placement hash, and board and piece counts.
        self.assertIn(f'        ["w", true, 0, "checkmate", null, "{hashed(3)}", 2, 63]\n', text)
        self.assertIn(f'        ["b", false, 20, "none", "{hashed(12)}", "{hashed(2)}", 2, 63],\n', text)
        self.assertIn('        ["b", false, 20, "none", null, null, null, null]\n', text)  # r/2: nothing said

    def test_a_schema_1_file_is_refused_and_the_message_says_to_re_record(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        path = golden.golden_path("ftxi")
        old = json.loads(path.read_text(encoding="utf-8"))
        old["schema"] = 1
        for entry in old["records"].values():
            entry["positions"] = [row[:4] for row in entry["positions"]]
        path.write_text(json.dumps(old), encoding="utf-8")
        with self.assertRaises(golden.StaleSchema) as caught:
            golden.load("ftxi")
        self.assertIn("golden --write", str(caught.exception))
        found = golden.verify_static([FakeSource("ftxi")], OPTIONS, self.records)
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("schema 1", found[0].message)
        self.assertIsNone(golden.lookup()("ftxi", "r/1"))  # reported by verify, not here

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

    def test_static_checks_notice_stale_and_incomplete_files(self):
        source = FakeSource("ftxi")
        self.write(source, self.results("ftxi"))
        codes = lambda found: sorted({item.code for item in found})  # noqa: E731
        self.assertEqual(golden.verify_static([source], OPTIONS, self.records), [])
        self.assertEqual(codes(golden.verify_static([FakeSource("ftxi", pin="p2")], OPTIONS, self.records)), ["golden-stale"])
        self.assertEqual(codes(golden.verify_static([source], RunOptions(cap=50), self.records)), ["golden-stale"])
        extra = self.records + [record("r/3", "c" * 64)]
        self.assertEqual(codes(golden.verify_static([source], OPTIONS, extra)), ["golden-incomplete"])
        self.assertEqual(codes(golden.verify_static([source], OPTIONS, self.records[:1])), ["golden-orphan"])
        changed = [self.records[0], record("r/2", "f" * 64)]
        self.assertEqual(codes(golden.verify_static([source], OPTIONS, changed)), ["golden-stale"])
        self.assertEqual(codes(golden.verify_static([FakeSource("cwmtt")], OPTIONS, self.records)), ["golden-missing"])

    def test_answers_recorded_by_other_adapter_code_are_stale_and_the_message_says_what_changed(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        found = golden.verify_static([FakeSource("ftxi", adapter="a2")], OPTIONS, self.records)
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("recorded under a different adapter_sha256 -", found[0].message)
        found = golden.verify_static([FakeSource("ftxi", pin="p2", adapter="a2")], RunOptions(cap=50), self.records)
        self.assertIn("recorded under a different adapter_sha256, cap, pin -", found[0].message)

    def test_a_golden_file_that_contradicts_itself_is_invalid(self):
        results = self.results("ftxi")
        results[("ftxi", "r/2")] = obs("ftxi", "r/2", [W, B], accepted=7)
        self.write(FakeSource("ftxi"), results)
        found = golden.verify_static([FakeSource("ftxi")], OPTIONS, self.records)
        self.assertEqual([item.code for item in found], ["golden-invalid"])

    def test_a_fresh_run_that_matches_passes_with_full_coverage(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        findings, stats = golden.check_run("ftxi", self.records, self.results("ftxi"), 1.0)
        self.assertEqual((findings, stats), ([], (20, 20)))

    def test_a_changed_verdict_fails(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        fresh = self.results("ftxi", second=(W, ("black", True, 20, "none")))
        findings, _ = golden.check_run("ftxi", self.records, fresh, 1.0)
        self.assertEqual([item.code for item in findings], ["golden-mismatch"])
        self.assertIn("r/2 position 1: in_check", findings[0].message)

    def test_a_changed_outcome_fails(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        for status, reason in (("unsupported", "variant"), ("error", "crash")):
            fresh = self.results("ftxi")
            fresh[("ftxi", "r/2")] = obs("ftxi", "r/2", [W], status=status, reason=reason)
            findings, _ = golden.check_run("ftxi", self.records, fresh, 0.0)
            self.assertEqual([item.code for item in findings], ["golden-status"], status)

    def test_a_lost_mate_verdict_is_a_finding_not_a_point_of_coverage(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        fresh = self.results("ftxi")
        fresh[("ftxi", "r/1")] = obs("ftxi", "r/1", [W, B, ("white", True, 0, None)])
        findings, stats = golden.check_run("ftxi", self.records, fresh, 0.5)
        self.assertEqual([item.code for item in findings], ["golden-lost-verdict"])
        self.assertIn("position 2 was recorded as checkmate", findings[0].message)
        # One the fresh run did not reach costs coverage only.
        fresh[("ftxi", "r/1")] = obs("ftxi", "r/1", [W, B], status="timeout", reason="timeout")
        findings, _ = golden.check_run("ftxi", self.records, fresh, 0.5)
        self.assertEqual(findings, [])

    def test_a_malformed_file_is_an_error_not_a_crash(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        path = golden.golden_path("ftxi")
        path.write_text(path.read_text(encoding="utf-8").replace('"status": "ok",', '"status_": "ok",', 1), encoding="utf-8")
        with self.assertRaises(golden.HarnessError):
            golden.load("ftxi")
        self.assertEqual([item.code for item in golden.verify_static([FakeSource("ftxi")], OPTIONS, self.records)], ["golden-missing"])

    def test_timeouts_cost_coverage_not_correctness(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        fresh = self.results("ftxi")
        fresh[("ftxi", "r/1")] = obs("ftxi", "r/1", [W], status="timeout", reason="timeout")
        findings, stats = golden.check_run("ftxi", self.records, fresh, 0.5)
        self.assertEqual((findings, stats), ([], (12, 20)))
        findings, _ = golden.check_run("ftxi", self.records, fresh, 0.9)
        self.assertEqual([item.code for item in findings], ["golden-coverage"])

    def test_disagreements_are_derived_from_the_golden_files(self):
        self.write_all({("chessin5d", "r/2"): [W, ("black", True, 20, "none")]})
        found = golden.disagreements(self.records)
        self.assertEqual({(item.a, item.b) for item in found},
                         {tuple(sorted((name, "chessin5d"), key=ORACLE_SOURCES.index)) for name in ORACLE_SOURCES
                          if name != "chessin5d"})
        self.assertTrue(all((item.record, item.kind, item.index) == ("r/2", "in_check", 1) for item in found))
        text = golden.render_disagreements(self.records)
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 1 |", text)
        self.assertIn("### `r/2`", text)
        self.assertNotIn("### `r/1`", text)
        self.assertEqual(text, golden.render_disagreements(self.records))

    def test_how_a_record_is_replayed_is_compared_once_per_pair_of_implementations(self):
        # ftxi cannot match the move text of action 1 of r/2; everyone else plays it.
        stuck = {(name, "r/2"): obs(name, "r/2", [W], status="rejected", reason="no_matching_move")
                 for name in ("ftxi", "ftxi:naive")}
        self.write_all(stuck)
        found = [item for item in golden.disagreements(self.records) if item.kind == "legality"]
        self.assertEqual([(item.a, item.b, item.soft) for item in found],
                         [("ftxi", "cwmtt", True), ("ftxi", "chessin5d", True), ("ftxi", "notation", True)])

    def test_the_report_says_how_much_was_actually_compared(self):
        capped = ("white", False, 1000, "none")
        counts_only = [(None, None, 20, "none"), (None, None, 1000, "none"), (None, None, 0, "no_action")]
        self.write_all({
            ("ftxi", "r/1"): [W, capped, MATE],
            ("cwmtt", "r/1"): counts_only,
            ("ftxi:naive", "r/1"): obs("ftxi:naive", "r/1", [W], status="timeout", reason="timeout"),
        })
        text = golden.render_disagreements(self.records)
        # ftxi and cwmtt both counted three positions of r/1 and two of r/2: four counts
        # are equal and one pair is equal only in having stopped at the cap. Neither
        # said what it set up, so none of the five is known to be the same position.
        self.assertIn("| `ftxi` vs `cwmtt` | legal_actions | 5 | 4 | 1 | 0 | 0 | 0 | 5 |", text)
        # On the end of r/1 cwmtt says only that nothing can be played.
        self.assertIn("| `ftxi` vs `cwmtt` | terminal | 5 | 4 | 1 | 0 | 0 | 0 | 5 |", text)
        # Two searches of one implementation are marked as such; the timed-out one compared one position.
        self.assertIn("| `ftxi` vs `ftxi:naive` (one implementation) | in_check | 3 | 3 | 0 | 0 | 0 | 0 | 3 |", text)
        self.assertIn("No two sources said what they set up at the same position: nothing was compared.", text)

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
        text = golden.render_disagreements(self.records)
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 5 | 5 | 0 | 0 | 5 | 0 | 0 |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | 5 | 5 | 5 | 5 |", text)  # both described, same boards, both marked, same markers
        self.assertIn("## Parse divergences", text)
        self.assertIn("None.", text)
        self.assertIn("| `ftxi` | 2 | 0 | 0 | 0 | 0 | 5 | 20 | 5 | 5 |", text)  # described, with markers

    def test_a_parse_divergence_is_listed_first_and_hides_what_it_makes_meaningless(self):
        # chessin5d reads r/2's first action as another position - and says, there,
        # that black is in check. That is not a check disagreement: it is about another position.
        self.described_all({("chessin5d", "r/2"): ([W, ("black", True, 20, "none")], [1, 9], [11, 19])})
        text = golden.render_disagreements(self.records)
        found = golden.disagreements(self.records)
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
        text = golden.render_disagreements(self.records)
        self.assertIn(f"| `r/1` | `ftxi` vs `chessin5d` | 1 | marker-only | `{hashed(12)}` (1 board, 32 pieces) "
                      f"| `{hashed(99)}` (1 board, 32 pieces) | - |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | in_check | 5 | 4 | 0 | 1 | 4 | 1 | 0 |", text)
        self.assertIn("| `ftxi` vs `chessin5d` | 5 | 5 | 5 | 4 |", text)
        self.assertIn("- position 1, in_check: `ftxi` = False; `chessin5d` = True", text)  # in the rules detail
        self.assertEqual(text, golden.render_disagreements(self.records))  # deterministic

    def test_searches_of_one_implementation_must_set_up_the_same_positions(self):
        self.described_all({("ftxi:naive", "r/1"): ([W, B, MATE], [1, 2, 5], [11, 12, 15])})
        kinds = {(item.a, item.b, item.kind) for item in golden.disagreements(self.records)}
        self.assertIn(("ftxi", "ftxi:naive", "placement"), kinds)

    def test_the_report_files_records_by_ruleset(self):
        self.records = [record("r/1"), record("r/2", "b" * 64, board="Standard - Reversed Royalty", fen=True)]
        self.write_all({("chessin5d", "r/2"): obs("chessin5d", "r/2", [], status="unsupported", reason="fen")})
        text = golden.render_disagreements(self.records)
        header = next(line for line in text.splitlines() if line.startswith("| Ruleset |"))
        self.assertTrue(header.endswith("| Positions | check | counts | mate |"))
        # ruleset, records, records replayed by each source, positions, then positions with each verdict
        replayed = " | ".join("0" if name == "chessin5d" else "1" for name in ORACLE_SOURCES)
        self.assertIn(f"| {CUSTOM_POSITIONS} | 1 | {replayed} | 2 | 2 | 2 | 2 |", text)
        self.assertIn(f"| Standard | 1 | {' | '.join('1' for _ in ORACLE_SOURCES)} | 3 | 3 | 3 | 3 |", text)
        self.assertNotIn("| Standard - Reversed Royalty |", text)

    def test_lookup_reads_the_files_once_and_survives_a_missing_one(self):
        self.write(FakeSource("ftxi"), self.results("ftxi"))
        recorded = golden.lookup()
        self.assertEqual(len(recorded("ftxi", "r/1").positions), 3)
        self.assertIsNone(recorded("ftxi", "r/9"))
        self.assertIsNone(recorded("cwmtt", "r/1"))  # no such file: reported elsewhere, not here


if __name__ == "__main__":
    unittest.main()
