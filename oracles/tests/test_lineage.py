"""H47's two-oracle bar counted by lineage (CHECKPOINT 10): an oracle derived from another's code counts with it, and a
lineage vouches through any member that agrees and is not a recorded divergence, at most once."""

import tempfile
import unittest
from pathlib import Path

from oracle_harness import gate, lockfile, paths, provenance
from oracle_harness.adapters import EngineSource
from oracle_harness.corpus import load as load_corpus
from oracle_harness.divergences import Divergence
from oracle_harness.util import HarnessError
from test_gate import B_CHECK, COUNT, TZ, B, Named, W, obs, record

# The lineage table as the lock file declares it for the rules oracles (Kairos counts with ftxi).
LINEAGES = {"ftxi": "ftxi", "cwmtt": "cwmtt", "chessin5d": "chessin5d", "kairos": "ftxi", "notation": "notation"}


def differential(reports: dict[str, list], *, lineages=None, entries=(), min_checked=1.0, rulesets=("Standard", TZ)):
    """The engine differential over one record per ruleset; the engine says [W, B] everywhere, each source what
    ``reports`` gives (a list of rows, or None for a record it does not read). Returns (findings, gated, ungated)."""
    engine = EngineSource(["engine"])
    oracles = [Named(name) for name in reports]
    records = [record(f"r/{index}", board) for index, board in enumerate(rulesets)]
    results, engine_results = {}, {}
    for item in records:
        for name, rows in reports.items():
            results[(name, item.id)] = (obs(name, item.id, [], status="unsupported", reason="variant") if rows is None
                                        else obs(name, item.id, rows))
        engine_results[("engine", item.id)] = obs("engine", item.id, [W, B])
    return gate.engine_differential(engine, oracles, records, results, engine_results, list(entries), min_checked,
                                    None, lineages)


def ftxi_check_divergence(record_id: str) -> Divergence:
    """ftxi's 'check' at position 1 where the engine says no check, on record as ftxi's error - an R1 entry's shape."""
    return Divergence(id="KD-9001", oracle="ftxi", pin="p", record=record_id, record_sha256="s", kind="in_check", index=1,
                      engine_value="False", oracle_value="True", rulebook="docs/rules/README.md R1: " + "x" * 30,
                      upstream="https://example.com/1", added="2026-11-01")


class LineageTableTests(unittest.TestCase):
    """The lineage table: one declared place (the lock file), rendered into PROVENANCE.md, the gate's input."""

    def test_every_oracle_declares_its_lineage_and_kairos_counts_with_ftxi(self):
        lock = lockfile.load()
        self.assertEqual(lock.lineages(), LINEAGES)
        for pin in lock.oracles():
            # What was read, and found, is said beside the lineage: NOTICE, credits, licence, pinned code.
            self.assertIn("Read at the pin", pin.lineage_note, pin.id)
        self.assertIn("NOTICE", lock.pin("kairos").lineage_note)
        self.assertIn("cwmtt", lock.pin("ftxi").lineage_note)  # the method ftxi adapts, said there

    def test_provenance_renders_the_lineage_so_verify_holds_the_two_together(self):
        lock = lockfile.load()
        block = provenance.render_block(lock, load_corpus())
        self.assertIn("| Lineage |", block)
        self.assertIn("- **Lineage:** `ftxi` - " + lock.pin("kairos").lineage_note, block)
        self.assertEqual(provenance.check(lock, load_corpus()), [])

    def load(self, change):
        text = paths.LOCK_FILE.read_text(encoding="utf-8")
        old, new = change
        self.assertIn(old, text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oracles.lock.toml"
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return lockfile.load(path)

    def test_a_lineage_names_a_root_of_the_same_role(self):
        bad = [('lineage = "ftxi"\nlineage_note = "Read at the pin (2026-11-01): its NOTICE', 'lineage = "stockfish"\n'
                'lineage_note = "Read at the pin (2026-11-01): its NOTICE'),  # no such pin
               ('lineage = "cwmtt"', 'lineage = "kairos"'),  # a chain: kairos is itself in ftxi's lineage
               ('lineage = "cwmtt"', 'lineage = "notation"')]  # a parse-only oracle's lineage for a rules oracle
        for change in bad:
            with self.subTest(change=change[1][:40]), self.assertRaises(HarnessError):
                self.load(change)
        with self.assertRaises(HarnessError):  # an oracle that does not say its lineage
            self.load(('lineage = "chessin5d"\n', ""))


class TwoOracleBarTests(unittest.TestCase):
    def test_ftxi_and_kairos_agreeing_count_once(self):
        for reports in ({"ftxi": [W, B], "kairos": [W, B]},
                        {"ftxi": [W, B], "ftxi:naive": [W, B], "kairos": [W, B], "kairos:naive": [W, B]}):
            with self.subTest(sources=list(reports)):
                findings, gated, ungated = differential(reports, lineages=LINEAGES)
                self.assertEqual((findings, gated), ([], {}))
                for reason in ungated.values():
                    self.assertIn("only one oracle followed", reason)
                    self.assertIn("ftxi's lineage (ftxi, kairos)", reason)
                    self.assertIn("one lineage", reason)
                # Without the lineage table each implementation counts apart, and both rulesets would be gated.
                findings, gated, ungated = differential(reports)
                self.assertEqual((findings, set(gated), ungated), ([], {"Standard", TZ}, {}))

    def test_kairos_beside_an_independent_lineage_counts(self):
        for other in ("cwmtt", "chessin5d"):
            with self.subTest(other=other):
                findings, gated, ungated = differential({other: [W, B], "kairos": [W, B]}, lineages=LINEAGES)
                self.assertEqual((findings, set(gated), ungated), ([], {"Standard", TZ}, {}))
                self.assertEqual({item.oracles for item in gated.values()}, {2})

    def test_kairos_is_its_lineages_vouch_where_ftxi_reads_nothing(self):
        # The one place it adds a vouch (H52): a ruleset ftxi does not read, followed by cwmtt and Kairos.
        findings, gated, ungated = differential({"ftxi": None, "kairos": [W, B], "cwmtt": [W, B]}, lineages=LINEAGES)
        self.assertEqual((findings, set(gated), ungated), ([], {"Standard", TZ}, {}))
        _, gated, ungated = differential({"ftxi": None, "cwmtt": [W, B]}, lineages=LINEAGES)
        self.assertEqual(gated, {})  # cwmtt alone

    def test_at_an_r1_position_kairos_vouches_for_the_lineage_where_ftxis_answer_is_a_recorded_divergence(self):
        # ftxi says check at position 1, on record as its error (R1's shape); Kairos agrees with the engine; cwmtt gives
        # counts only. The check at position 1 is settled by Kairos alone - the lineage's vouch through its agreeing
        # member - and the record is still followed by two lineages, not three.
        entries = [ftxi_check_divergence("r/0")]
        reports = {"ftxi": [W, B_CHECK], "kairos": [W, B], "cwmtt": [COUNT, COUNT]}
        findings, gated, ungated = differential(reports, lineages=LINEAGES, entries=entries, rulesets=("Standard",))
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated["Standard"].checked["in_check"], 2)
        self.assertEqual(gated["Standard"].oracles, 2)  # cwmtt and ftxi's lineage, once
        # Without Kairos the excused answer confirms nothing: the check is settled at one position of two.
        _, gated, ungated = differential({"ftxi": [W, B_CHECK], "cwmtt": [COUNT, COUNT]}, lineages=LINEAGES,
                                         entries=entries, rulesets=("Standard",))
        self.assertEqual(gated, {})
        self.assertIn("check verdict at only 1 of the engine's 2 positions", ungated["Standard"])

    def test_a_member_that_misread_the_record_leaves_the_vouch_to_another(self):
        # ftxi stops at action 1 where the engine plays on (a replay divergence, excused or not): it vouches for nothing,
        # and Kairos, which followed, is the lineage's vouch beside Chess in 5D.
        stopped = obs("ftxi", "r/0", [W], status="rejected", reason="no_matching_move")
        reports = {"kairos": [W, B], "chessin5d": [W, B]}
        engine = EngineSource(["engine"])
        records = [record("r/0")]
        results = {("ftxi", "r/0"): stopped, **{(name, "r/0"): obs(name, "r/0", rows) for name, rows in reports.items()}}
        engine_results = {("engine", "r/0"): obs("engine", "r/0", [W, B])}
        entry = Divergence(id="KD-9002", oracle="ftxi", pin="p", record="r/0", record_sha256="s", kind="legality",
                           index=1, engine_value="accepted", oracle_value="rejected:no_matching_move",
                           notation="README of adri326/5dchess-notation: " + "x" * 30, upstream="unreported: test",
                           added="2026-11-01")
        findings, gated, ungated = gate.engine_differential(
            engine, [Named("ftxi"), Named("kairos"), Named("chessin5d")], records, results, engine_results, [entry],
            1.0, None, LINEAGES)
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated["Standard"].oracles, 2)

    def test_the_existing_oracles_gate_exactly_as_before(self):
        # Every one of the four is its own lineage: the table changes no verdict or message without Kairos.
        cases = [{"ftxi": [W, B]}, {"ftxi": [W, B], "ftxi:naive": [W, B]}, {"ftxi": [W, B], "chessin5d": [W, B]},
                 {"ftxi": [W, B], "ftxi:naive": [W, B], "cwmtt": [W, B], "cwmtt:naive": [W, B], "chessin5d": [W, B]},
                 {"cwmtt": [W, B], "chessin5d": [W, B]}, {"ftxi": None, "cwmtt": [W, B], "chessin5d": [W, B]},
                 {"ftxi": [W, B], "cwmtt": None}, {"ftxi": [W, B_CHECK], "cwmtt": [COUNT, COUNT]}]
        for reports in cases:
            with self.subTest(sources=reports):
                self.assertEqual(differential(reports, lineages=LINEAGES), differential(reports))


if __name__ == "__main__":
    unittest.main()
