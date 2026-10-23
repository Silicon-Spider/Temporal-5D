"""The Kairos sources (decision H52): wired but enabled in no tier, what each answers, and its driver."""

import json
import tempfile
import unittest
from pathlib import Path

from oracle_harness import build, config, corpus, golden, lockfile, paths
from oracle_harness.adapters import (
    ORACLE_SOURCES,
    PENDING_SOURCES,
    PRIMARY_SOURCES,
    RunOptions,
    make_source,
    oracle_of,
)
from oracle_harness.adapters.base import parse_jsonl
from oracle_harness.adapters.kairos import KairosSource
from oracle_harness.util import Completed, HarnessError

START = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"
# A record whose move text ends in a parenthesised variation: Kairos's reader, a port of
# ftxi's, would replay the variation as the main line.
ENDS_IN_A_VARIATION = b'[Board "Standard"]\n1. (0T1)e2e3 / (0T1)e7e6\n(2. (0T2)d2d3)\n'


def _lines(*events) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


class PendingTests(unittest.TestCase):
    """Wired - built, observed, tested - and enabled in no tier until its golden files are recorded."""

    def test_its_two_searches_are_pending_sources_of_one_implementation(self):
        self.assertEqual(PENDING_SOURCES, ("kairos", "kairos:naive"))
        self.assertFalse(set(PENDING_SOURCES) & set(ORACLE_SOURCES))
        self.assertFalse(set(PENDING_SOURCES) & set(PRIMARY_SOURCES))
        lock = lockfile.load()
        for name in PENDING_SOURCES:
            source = make_source(name, lock)
            self.assertIsInstance(source, KairosSource)
            self.assertEqual((source.name, source.oracle, oracle_of(name)), (name, "kairos", "kairos"))

    def test_no_tier_lists_a_pending_source(self):
        for name, tier in config.load().tiers.items():
            self.assertFalse(set(tier.sources) & set(PENDING_SOURCES), name)

    def test_a_tier_cannot_list_it_before_it_is_enabled(self):
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        old = ('sources = ["ftxi", "ftxi:naive", "cwmtt", "cwmtt:naive", "chessin5d", "notation"]\n'
               'record_tiers = ["commit", "nightly"]\ntimeout_s = 300\n')
        self.assertIn(old, text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gate.toml"
            path.write_text(text.replace(old, old.replace('"notation"]', '"notation", "kairos"]'), 1), encoding="utf-8")
            with self.assertRaises(HarnessError):
                config.load(path)

    def test_a_recorded_source_has_golden_files_and_a_pending_one_has_none(self):
        # Moving Kairos into ORACLE_SOURCES and recording it on a quiet machine go together (BUILD_LOG issue 530).
        for name in ORACLE_SOURCES:
            self.assertTrue(golden.golden_path(name).exists(), name)
        for name in PENDING_SOURCES:
            self.assertFalse(golden.golden_path(name).exists(), name)
            self.assertFalse(golden.cost_path(name).exists(), name)


class AnswersTests(unittest.TestCase):
    def setUp(self):
        self.lock = lockfile.load()

    def test_the_hypercuboid_search_gives_no_count_and_the_exhaustive_one_does(self):
        hypercuboid, naive = (make_source(name, self.lock) for name in PENDING_SOURCES)
        self.assertNotIn("legal_actions", hypercuboid.answers)
        self.assertIn("legal_actions", naive.answers)
        for source in (hypercuboid, naive):
            self.assertTrue({"state", "side", "in_check", "terminal", "legality"} <= set(source.answers))
            self.assertEqual(source.diagnostics, ("actions",))

    def test_the_signature_names_the_pin_the_search_and_the_code(self):
        options = RunOptions(cap=1000)
        hypercuboid, naive = (make_source(name, self.lock).signature(options) for name in PENDING_SOURCES)
        self.assertEqual((hypercuboid["search"], naive["search"]), ("hypercuboid", "naive"))
        # The cap bounds the count, which only the exhaustive search gives.
        self.assertNotIn("cap", hypercuboid)
        self.assertEqual(naive["cap"], 1000)
        for signature in (hypercuboid, naive):
            self.assertEqual(signature["pin"], self.lock.pin("kairos").commit)
            self.assertRegex(signature["driver_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(signature["adapter_sha256"], r"^[0-9a-f]{64}$")

    def test_a_record_that_ends_in_a_variation_is_refused_before_it_runs(self):
        for name, cap in (("kairos", None), ("kairos:naive", 1000)):
            observation = make_source(name, self.lock).observe("r/variation", ENDS_IN_A_VARIATION, RunOptions(cap=1000))
            self.assertEqual((observation.status, observation.reason, observation.actions_accepted, observation.cap),
                             ("unsupported", "notation", 0, cap))
            self.assertEqual(observation.positions, [])


class DriverOutputTests(unittest.TestCase):
    """What the driver prints, read as the harness reads every driver."""

    def parse(self, stdout: bytes, cap):
        return parse_jsonl("kairos", "r/1", Completed(0, stdout, b"", False), cap, 0.1)

    def test_the_hypercuboid_search_reports_its_verdicts_and_a_null_count(self):
        out = _lines({"event": "start", "driver": "kairos"},
                     {"event": "state", "index": 0, "size": "8x8", "markers": True, "boards": [f"0:1:w:{START}"]},
                     {"event": "position", "index": 0, "side": "white", "in_check": False, "legal_actions": None,
                      "terminal": "none", "ms": 0.1},
                     {"event": "end", "status": "ok", "actions_accepted": 0})
        observation = self.parse(out, None)
        self.assertEqual(observation.status, "ok")
        (position,) = observation.positions
        self.assertEqual((position.legal_actions, position.terminal, position.in_check), (None, "none", False))
        self.assertEqual(position.fingerprint, "d574889fd9da3f2bc65249ff27249b00")

    def test_a_record_its_reader_fails_is_unsupported_from_position_zero(self):
        out = _lines({"event": "end", "status": "unsupported", "actions_accepted": 0, "reason": "parse_failed",
                      "detail": "Invalid move (0T1)Pe2(0T1)e3 at T1 white: no legal move matches"})
        observation = self.parse(out, 1000)
        self.assertEqual((observation.status, observation.reason, observation.positions), ("unsupported", "parse_failed", []))


def _kairos_built() -> bool:
    pin = lockfile.load().pin("kairos")
    return build.load(pin) is not None


@unittest.skipUnless(_kairos_built(), "the kairos oracle is not built (python oracles/harness.py build --oracle kairos)")
class LiveDriverTests(unittest.TestCase):
    """The built driver against records whose answers are known (the other oracles' golden files)."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = {record.id: record for record in corpus.load()}

    def observe(self, source: str, record_id: str):
        record = self.records[record_id]
        observation = make_source(source, self.lock).observe(record_id, corpus.feed(record, self.lock).data,
                                                             RunOptions(cap=1000, timeout_s=120))
        self.assertNotIn(observation.status, ("error", "timeout"), observation.detail)
        return observation

    def test_it_sets_up_and_replays_what_ftxi_recorded_and_ends_in_checkmate(self):
        recorded = golden.load("ftxi").observations["cwmtt/standard"]
        for source in PENDING_SOURCES:
            observation = self.observe(source, "cwmtt/standard")
            self.assertEqual((observation.status, observation.actions_accepted), (recorded.status, recorded.actions_accepted))
            self.assertEqual([p.fingerprint for p in observation.positions], [p.fingerprint for p in recorded.positions])
            self.assertEqual([p.side for p in observation.positions], [p.side for p in recorded.positions])
            self.assertEqual(observation.positions[-1].terminal, "checkmate")
            self.assertTrue(observation.positions[-1].in_check)
        naive = self.observe("kairos:naive", "cwmtt/standard")
        self.assertEqual(naive.positions[0].legal_actions, 20)
        hypercuboid = self.observe("kairos", "cwmtt/standard")
        self.assertTrue(all(p.legal_actions is None for p in hypercuboid.positions))

    def test_an_action_that_ignores_a_check_cannot_be_submitted(self):
        recorded = golden.load("ftxi").observations["local/ignores-check"]
        self.assertEqual((recorded.status, recorded.reason), ("rejected", "cannot_submit"))
        for source in PENDING_SOURCES:
            observation = self.observe(source, "local/ignores-check")
            self.assertEqual((observation.status, observation.reason, observation.actions_accepted),
                             ("rejected", "cannot_submit", recorded.actions_accepted))
            self.assertEqual(len(observation.positions), recorded.actions_accepted + 1)

    def test_the_actions_diagnostic_lists_the_exhaustive_count(self):
        record = self.records["cwmtt/standard"]
        observation = make_source("kairos:naive", self.lock).observe(
            "cwmtt/standard", corpus.feed(record, self.lock).data, RunOptions(cap=1000, timeout_s=120, actions_at=0))
        (event,) = [event for event in observation.diagnostics if event.get("event") == "actions"]
        self.assertEqual((event["index"], event["count"], event["truncated"], len(event["actions"])), (0, 20, False, 20))


if __name__ == "__main__":
    unittest.main()
