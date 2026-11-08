"""Kairos in parse mode (decision H52; CHECKPOINT 9's brief: "parse mode first, then full mode", reordered by
CHECKPOINT 10): source kairos:parse, pending until after the release tier's sign-off run, which leaves Kairos out."""

import unittest

from oracle_harness import build, config, corpus, gate, golden, lockfile, observations
from oracle_harness.adapters import (
    ORACLE_SOURCES,
    PARSE_ONLY_SOURCES,
    PENDING_SOURCES,
    PRIMARY_SOURCES,
    EngineSource,
    RunOptions,
    make_source,
)
from oracle_harness.adapters.kairos import KairosSource
from oracle_harness.divergences import Divergence
from test_gate import B_CHECK, COUNT, IDENTITY, B, Named, W, described, obs, record
from test_kairos import ENDS_IN_A_VARIATION
from test_lineage import LINEAGES, differential, ftxi_check_divergence

# What a parse-only source prints at two positions when its driver strays: a check answer at position 1.
STRAY = [("white", None, None, None), ("black", False, None, None)]

PARSE = "kairos:parse"


class ParseModeTests(unittest.TestCase):
    def setUp(self):
        self.lock = lockfile.load()

    def test_it_is_a_pending_parse_only_source_in_no_tier(self):
        self.assertIn(PARSE, PENDING_SOURCES)
        self.assertIn(PARSE, PARSE_ONLY_SOURCES)
        self.assertNotIn(PARSE, ORACLE_SOURCES)
        self.assertNotIn(PARSE, PRIMARY_SOURCES)
        for name, tier in config.load().tiers.items():
            self.assertNotIn(PARSE, tier.sources, name)

    def test_it_answers_no_rules_question(self):
        source = make_source(PARSE, self.lock)
        self.assertIsInstance(source, KairosSource)
        self.assertFalse({"in_check", "legal_actions", "terminal"} & set(source.answers))
        self.assertTrue({"state", "side", "legality"} <= set(source.answers))
        self.assertEqual(source.diagnostics, ())
        signature = source.signature(RunOptions(cap=1000))
        self.assertEqual(signature["search"], "parse")
        self.assertNotIn("cap", signature)

    def test_its_playing_an_action_is_no_verdict_and_kaiross_searches_still_judge(self):
        self.assertFalse(observations.judges_actions(PARSE))
        self.assertFalse(observations.judges_actions("notation"))
        for name in ("kairos", "kairos:naive", "ftxi", "cwmtt", "chessin5d"):
            self.assertTrue(observations.judges_actions(name), name)

    def test_it_vouches_for_no_ruleset(self):
        # Beside ftxi alone it adds no second lineage, under the lineage table or with every implementation its own.
        for lineages in (LINEAGES, IDENTITY):
            with self.subTest(lineages=lineages):
                findings, gated, ungated = differential({"ftxi": [("white", False, 20, "none"),
                                                                  ("black", False, 20, "none")],
                                                         PARSE: [("white", None, None, None),
                                                                 ("black", None, None, None)]}, lineages=lineages)
                self.assertEqual((findings, gated), ([], {}))
                for reason in ungated.values():
                    self.assertIn("only one oracle followed", reason)
                    self.assertIn("(ftxi)", reason)

    def test_an_answer_a_parse_only_source_prints_all_the_same_settles_nothing(self):
        # R1's shape: ftxi's check at position 1 is a recorded divergence and cwmtt gives counts only, so the check is
        # settled at one position of two. A stray check answer from a parse-only source - Kairos's parse mode, or the
        # notation parser - must not settle the other: it answers no rules question, whatever its driver prints.
        entries = [ftxi_check_divergence("r/0")]
        for source in PARSE_ONLY_SOURCES:
            with self.subTest(source=source):
                reports = {"ftxi": [W, B_CHECK], "cwmtt": [COUNT, COUNT], source: STRAY}
                findings, gated, ungated = differential(reports, lineages=LINEAGES, entries=entries,
                                                        rulesets=("Standard",))
                self.assertEqual((findings, gated), ([], {}))
                self.assertIn("check verdict at only 1 of the engine's 2 positions", ungated["Standard"])

    def test_what_a_parse_only_source_held_still_counts_and_its_recorded_divergences_are_still_looked_at(self):
        # Only its answers are set aside. The boards it held count towards those compared: here it alone gives them.
        engine = EngineSource(["engine"])
        records = [record("r/0")]
        engine_results = {("engine", "r/0"): described(obs("engine", "r/0", [W, B]), [1, 2])}
        results = {("ftxi", "r/0"): obs("ftxi", "r/0", [W, B]), ("chessin5d", "r/0"): obs("chessin5d", "r/0", [W, B]),
                   (PARSE, "r/0"): described(obs(PARSE, "r/0", STRAY), [1, 2])}
        sources = [Named("ftxi"), Named("chessin5d"), Named(PARSE)]
        findings, gated, _ = gate.engine_differential(engine, sources, records, results, engine_results, [], 1.0, None,
                                                      lineages=LINEAGES)
        self.assertEqual(findings, [])
        self.assertEqual((gated["Standard"].aligned, gated["Standard"].oracles), (2, 2))
        # And an entry of its own that no longer occurs - it now reads the action - is reported stale.
        entry = Divergence(id="KD-9004", oracle="notation", pin="p", record="r/0", record_sha256="s", kind="legality",
                           index=1, engine_value="accepted", oracle_value="rejected:no_matching_move",
                           notation="README of adri326/5dchess-notation: " + "x" * 30, upstream="unreported: test",
                           added="2026-11-02")
        results = {("ftxi", "r/0"): obs("ftxi", "r/0", [W, B]), ("notation", "r/0"): obs("notation", "r/0", STRAY)}
        findings, _, _ = gate.engine_differential(engine, [Named("ftxi"), Named("notation")], records, results,
                                                  {("engine", "r/0"): obs("engine", "r/0", [W, B])}, [entry], 1.0, None,
                                                  lineages=LINEAGES)
        self.assertEqual([item.code for item in findings], ["divergence-stale"])

    def test_a_record_that_ends_in_a_variation_is_refused_before_it_runs(self):
        observation = make_source(PARSE, self.lock).observe("r/variation", ENDS_IN_A_VARIATION, RunOptions(cap=1000))
        self.assertEqual((observation.status, observation.reason, observation.cap), ("unsupported", "notation", None))


def _kairos_built() -> bool:
    return build.load(lockfile.load().pin("kairos")) is not None


@unittest.skipUnless(_kairos_built(), "the kairos oracle is not built (python oracles/harness.py build --oracle kairos)")
class LiveParseModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = {record.id: record for record in corpus.load()}

    def observe(self, source: str, record_id: str, timeout_s: float = 120):
        record = self.records[record_id]
        return make_source(source, self.lock).observe(record_id, corpus.feed(record, self.lock).data,
                                                      RunOptions(cap=1000, timeout_s=timeout_s))

    def test_it_says_what_it_set_up_and_whose_move_it_is_and_nothing_more(self):
        recorded = golden.load("ftxi").observations["cwmtt/standard"]
        observation = self.observe(PARSE, "cwmtt/standard")
        self.assertEqual((observation.status, observation.actions_accepted), (recorded.status, recorded.actions_accepted))
        self.assertEqual([p.fingerprint for p in observation.positions], [p.fingerprint for p in recorded.positions])
        self.assertEqual([p.side for p in observation.positions], [p.side for p in recorded.positions])
        for position in observation.positions:
            self.assertEqual((position.in_check, position.legal_actions, position.terminal), (None, None, None))

    def test_it_replays_as_the_searches_do_refusals_included(self):
        for record_id in ("local/ignores-check", "notation/1.2", "cwmtt/futures", "cwmtt/JustKings"):
            parsed = self.observe(PARSE, record_id)
            searched = self.observe("kairos", record_id, timeout_s=300)
            self.assertNotEqual(searched.status, "timeout", record_id)
            self.assertEqual((parsed.status, parsed.reason, parsed.actions_accepted),
                             (searched.status, searched.reason, searched.actions_accepted), record_id)
            self.assertEqual([p.fingerprint for p in parsed.positions], [p.fingerprint for p in searched.positions])
        self.assertEqual(self.observe(PARSE, "local/ignores-check").reason, "cannot_submit")


if __name__ == "__main__":
    unittest.main()
