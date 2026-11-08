"""H47's two-oracle bar counted by lineage (CHECKPOINT 10): an oracle derived from another's code counts with it, and a
lineage vouches through any member that agrees and is not a recorded divergence, at most once."""

import tempfile
import unittest
from pathlib import Path

import test_notation
from oracle_harness import gate, lockfile, paths, provenance
from oracle_harness.adapters import EngineSource
from oracle_harness.corpus import load as load_corpus
from oracle_harness.divergences import Divergence
from oracle_harness.gate import Gated
from oracle_harness.util import HarnessError
from test_gate import B_CHECK, COUNT, FULL, IDENTITY, TZ, B, Named, W, obs, record

# The lineage table as the lock file declares it for the rules oracles (Kairos counts with ftxi).
LINEAGES = {"ftxi": "ftxi", "cwmtt": "cwmtt", "chessin5d": "chessin5d", "kairos": "ftxi", "notation": "notation"}
NOTE = "an oracle derived from another's code counts with it as one lineage"


def differential(reports: dict[str, list], *, lineages, entries=(), min_checked=1.0, rulesets=("Standard", TZ)):
    """The engine differential over one record per ruleset; the engine says [W, B] everywhere, each source what
    ``reports`` gives (a list of rows, or None for a record it does not read). ``lineages`` is required, as it is of
    the gate (it fails closed): LINEAGES, or IDENTITY for every implementation its own. Returns (findings, gated,
    ungated)."""
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
                                    None, lineages=lineages)


def one_ruleset(results: dict, record_ids: list[str], sources: list[str], entries=()):
    """The engine differential over Standard records the engine reads as [W, B], with these oracle reports."""
    engine = EngineSource(["engine"])
    records = [record(record_id) for record_id in record_ids]
    engine_results = {("engine", record_id): obs("engine", record_id, [W, B]) for record_id in record_ids}
    return gate.engine_differential(engine, [Named(name) for name in sources], records, results, engine_results,
                                    list(entries), 1.0, None, lineages=LINEAGES)


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

    def load_text(self, text: str) -> lockfile.Lock:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oracles.lock.toml"
            path.write_text(text, encoding="utf-8")
            return lockfile.load(path)

    def load(self, change):
        """The repository's lock file with one change: ``old`` must occur in it exactly once."""
        text = paths.LOCK_FILE.read_text(encoding="utf-8")
        old, new = change
        self.assertEqual(text.count(old), 1, old)
        return self.load_text(text.replace(old, new, 1))

    def test_a_lineage_names_a_root_of_the_same_role(self):
        kairos = 'lineage_note = "Read at the pin (2026-11-01; corrected 2026-11-02): its NOTICE'
        bad = [(f'lineage = "ftxi"\n{kairos}', f'lineage = "stockfish"\n{kairos}'),  # no such pin
               ('lineage = "cwmtt"', 'lineage = "kairos"'),  # a chain: kairos is itself in ftxi's lineage
               ('lineage = "cwmtt"', 'lineage = "notation"')]  # a parse-only oracle's lineage for a rules oracle
        for change in bad:
            with self.subTest(change=change[1][:40]), self.assertRaises(HarnessError):
                self.load(change)
        with self.assertRaises(HarnessError):  # an oracle that does not say its lineage
            self.load(('lineage = "chessin5d"\n', ""))

    def test_a_parse_oracles_lineage_is_checked_too(self):
        # The notation parser's lineage must name a pinned parse-oracle: neither a stranger nor a rules oracle.
        for new in ('lineage = "stockfish"\n', 'lineage = "ftxi"\n'):
            with self.subTest(new=new), self.assertRaises(HarnessError):
                self.load(('lineage = "notation"\n', new))

    def test_an_oracle_must_say_how_its_lineage_was_read(self):
        # lineage_note is required beside lineage - not renamed, not empty, not blank.
        old = 'lineage = "chessin5d"\nlineage_note = "'
        for new in ('lineage = "chessin5d"\nlineage_remark = "', 'lineage = "chessin5d"\nlineage_note = "" #"',
                    'lineage = "chessin5d"\nlineage_note = "   " #"'):
            with self.subTest(new=new), self.assertRaises(HarnessError):
                self.load((old, new))

    def test_a_corpus_source_declares_no_lineage(self):
        # A pin the harness never runs has no lineage to count: either key alone, or both, is refused.
        lineage, note = test_notation._LINEAGE.split("\n")
        for keys in (test_notation._LINEAGE, lineage, note):
            text = test_notation._LOCK.format(role="corpus-source", tier="", sha1="0" * 40, sha256="0" * 64, lineage=keys)
            with self.subTest(keys=keys), self.assertRaises(HarnessError):
                self.load_text(text)
        # The same pin without them loads.
        self.assertEqual(self.load_text(test_notation._LOCK.format(role="corpus-source", tier="", sha1="0" * 40,
                                                                   sha256="0" * 64, lineage="")).oracles(), [])

    def test_the_table_fails_closed(self):
        # engine_differential takes the table as a required keyword - no default that would count every implementation
        # apart - and an implementation it compares that the table does not name is a HarnessError.
        engine = EngineSource(["engine"])
        records = [record("r/0")]
        results = {(name, "r/0"): obs(name, "r/0", [W, B]) for name in ("ftxi", "kairos")}
        engine_results = {("engine", "r/0"): obs("engine", "r/0", [W, B])}
        oracles = [Named("ftxi"), Named("kairos")]
        with self.assertRaises(TypeError):
            gate.engine_differential(engine, oracles, records, results, engine_results, [], 1.0, None)  # type: ignore[call-arg]
        for table in ({}, {"ftxi": "ftxi"}):
            with self.subTest(table=table), self.assertRaises(HarnessError):
                gate.engine_differential(engine, oracles, records, results, engine_results, [], 1.0, None, lineages=table)


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
                    self.assertIn(NOTE, reason)
                # With every implementation its own lineage they would count apart, and both rulesets would be gated.
                findings, gated, ungated = differential(reports, lineages=IDENTITY)
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
        results = {("ftxi", "r/0"): stopped, **{(name, "r/0"): obs(name, "r/0", [W, B]) for name in ("kairos", "chessin5d")}}
        entry = Divergence(id="KD-9002", oracle="ftxi", pin="p", record="r/0", record_sha256="s", kind="legality",
                           index=1, engine_value="accepted", oracle_value="rejected:no_matching_move",
                           notation="README of adri326/5dchess-notation: " + "x" * 30, upstream="unreported: test",
                           added="2026-11-01")
        findings, gated, ungated = one_ruleset(results, ["r/0"], ["ftxi", "kairos", "chessin5d"], [entry])
        self.assertEqual((findings, ungated), ([], {}))
        self.assertEqual(gated["Standard"].oracles, 2)

    def test_a_member_that_misread_is_left_out_of_its_lineages_label(self):
        # ftxi reads one action further than the engine - a length divergence, excused - so it followed as far as the
        # engine went, but read the record otherwise: it vouches for nothing, and is not named. Kairos alone is the
        # lineage's vouch, and the reason names it alone.
        entry = Divergence(id="KD-9003", oracle="ftxi", pin="p", record="r/0", record_sha256="s", kind="length", index=2,
                           engine_value="1 actions", oracle_value="2 actions",
                           notation="README of adri326/5dchess-notation: " + "x" * 30, upstream="unreported: test",
                           added="2026-11-02")
        results = {("ftxi", "r/0"): obs("ftxi", "r/0", [W, B, W]), ("kairos", "r/0"): obs("kairos", "r/0", [W, B])}
        findings, gated, ungated = one_ruleset(results, ["r/0"], ["ftxi", "kairos"], [entry])
        self.assertEqual((findings, gated), ([], {}))
        self.assertTrue(ungated["Standard"].startswith(
            "only one oracle followed r/0 (ftxi's lineage (kairos)) as far as the engine went"), ungated["Standard"])

    def test_a_lineage_followed_only_by_a_derived_member_is_named_with_that_member(self):
        # ftxi does not read the record; Kairos (ftxi's lineage) follows it alone. The reason must not say ftxi followed.
        findings, gated, ungated = differential({"ftxi": None, "kairos": [W, B]}, lineages=LINEAGES, rulesets=("Standard",))
        self.assertEqual((findings, gated), ([], {}))
        self.assertTrue(ungated["Standard"].startswith(
            "only one oracle followed r/0 (ftxi's lineage (kairos)) as far as the engine went"), ungated["Standard"])
        self.assertIn(NOTE, ungated["Standard"])

    def test_the_one_lineage_note_reads_every_record_the_reason_names(self):
        # Two records of one ruleset, both followed by one lineage: r/0 by ftxi alone, r/1 by ftxi and Kairos. The reason
        # names both, and the second's label is why the note is there.
        results = {("ftxi", "r/0"): obs("ftxi", "r/0", [W, B]), ("ftxi", "r/1"): obs("ftxi", "r/1", [W, B]),
                   ("kairos", "r/0"): obs("kairos", "r/0", [], status="unsupported", reason="variant"),
                   ("kairos", "r/1"): obs("kairos", "r/1", [W, B])}
        findings, gated, ungated = one_ruleset(results, ["r/0", "r/1"], ["ftxi", "kairos"])
        self.assertEqual((findings, gated), ([], {}))
        self.assertIn("r/0 (ftxi), r/1 (ftxi's lineage (ftxi, kairos))", ungated["Standard"])
        self.assertIn(NOTE, ungated["Standard"])

    def test_the_one_lineage_note_is_not_given_for_a_record_the_reason_does_not_name(self):
        # r/0 is followed by two lineages - ftxi's (ftxi and Kairos) and cwmtt - and is not named; r/1 by cwmtt alone.
        # The reason names r/1 (cwmtt): nothing in it counts once for being derived, so there is no note.
        results = {**{(name, "r/0"): obs(name, "r/0", [W, B]) for name in ("ftxi", "kairos", "cwmtt")},
                   ("ftxi", "r/1"): obs("ftxi", "r/1", [], status="unsupported", reason="variant"),
                   ("kairos", "r/1"): obs("kairos", "r/1", [], status="unsupported", reason="variant"),
                   ("cwmtt", "r/1"): obs("cwmtt", "r/1", [W, B])}
        findings, gated, ungated = one_ruleset(results, ["r/0", "r/1"], ["ftxi", "kairos", "cwmtt"])
        self.assertEqual((findings, gated), ([], {}))
        self.assertTrue(ungated["Standard"].startswith("only one oracle followed r/1 (cwmtt) as far as the engine went"),
                        ungated["Standard"])
        self.assertNotIn(NOTE, ungated["Standard"])

    def test_the_existing_oracles_gate_exactly_as_they_did_before_lineages_were_counted(self):
        # Every one of the four is its own lineage. The expected values are what 28a102b's gate - before this table -
        # gave for these eight combinations (run on that commit's harness, 2026-11-02): findings, every Gated field and
        # every ungated reason. Only Gated.describe's wording has changed since ("oracle lineages").
        one = ("only one oracle followed r/{} (ftxi) as far as the engine went - a ruleset is gated only when at least 2 "
               "answer for each of its records (H47)")
        alone = {"Standard": one.format(0), TZ: one.format(1)}

        def both(oracles):
            return {"Standard": Gated(records=1, positions=2, checked=FULL, oracles=oracles),
                    TZ: Gated(records=1, positions=2, checked=FULL, oracles=oracles)}

        divergence = [("divergence", f"r/{index} position 1: in_check: engine=False vs ftxi=True") for index in (0, 1)]
        cases = [({"ftxi": [W, B]}, [], {}, alone),
                 ({"ftxi": [W, B], "ftxi:naive": [W, B]}, [], {}, alone),
                 ({"ftxi": [W, B], "chessin5d": [W, B]}, [], both(2), {}),
                 ({"ftxi": [W, B], "ftxi:naive": [W, B], "cwmtt": [W, B], "cwmtt:naive": [W, B], "chessin5d": [W, B]},
                  [], both(3), {}),
                 ({"cwmtt": [W, B], "chessin5d": [W, B]}, [], both(2), {}),
                 ({"ftxi": None, "cwmtt": [W, B], "chessin5d": [W, B]}, [], both(2), {}),
                 ({"ftxi": [W, B], "cwmtt": None}, [], {}, alone),
                 ({"ftxi": [W, B_CHECK], "cwmtt": [COUNT, COUNT]}, divergence, {},
                  {"Standard": "unexplained divergence", TZ: "unexplained divergence"})]
        for reports, findings, gated, ungated in cases:
            with self.subTest(sources=list(reports)):
                found, gated_now, ungated_now = differential(reports, lineages=LINEAGES)
                self.assertEqual(([(item.code, item.message) for item in found], gated_now, ungated_now),
                                 (findings, gated, ungated))


if __name__ == "__main__":
    unittest.main()
