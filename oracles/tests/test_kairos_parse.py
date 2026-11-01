"""Kairos in parse mode (decision H52: "parse mode first, then full mode"): source kairos:parse, pending until after
the release tier's sign-off run, which leaves Kairos out (CHECKPOINT 10)."""

import unittest

from oracle_harness import build, config, corpus, golden, lockfile, observations
from oracle_harness.adapters import (
    ORACLE_SOURCES,
    PARSE_ONLY_SOURCES,
    PENDING_SOURCES,
    PRIMARY_SOURCES,
    RunOptions,
    make_source,
)
from oracle_harness.adapters.kairos import KairosSource
from test_kairos import ENDS_IN_A_VARIATION
from test_lineage import LINEAGES, differential

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
        # Beside ftxi alone it adds no second lineage, under the lineage table or without it.
        for lineages in (LINEAGES, None):
            with self.subTest(lineages=lineages):
                findings, gated, ungated = differential({"ftxi": [("white", False, 20, "none"),
                                                                  ("black", False, 20, "none")],
                                                         PARSE: [("white", None, None, None),
                                                                 ("black", None, None, None)]}, lineages=lineages)
                self.assertEqual((findings, gated), ([], {}))
                for reason in ungated.values():
                    self.assertIn("only one oracle followed", reason)
                    self.assertIn("(ftxi)", reason)

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
