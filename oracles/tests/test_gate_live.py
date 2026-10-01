"""The whole gate, end to end, with a stand-in engine.

There is no engine yet, so these tests plug in a program that answers every
record with what ftxi's golden file says - and then ones that get a verdict
wrong, leave one out, or decline every record - to show that the gate passes the
first, fails the others, and accepts a wrong verdict only once the difference is
on record with a citation.

They run the real ftxi oracle and are skipped where it has not been built.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import build, config, corpus, gate, gitutil, golden, lockfile, paths
from oracle_harness.util import sha256_bytes

# argv: answers.json, then what to do differently -
#   -                     nothing: echo the oracle
#   flip:<record>@<k>     the opposite check verdict at position k of that record
#   null:<record>@<k>     no check verdict at all there
#   unsupported           decline every record
ENGINE = r'''
import json, hashlib, sys
answers = json.load(open(sys.argv[1], encoding="utf-8"))
mode, _, where = sys.argv[2].partition(":")
entry = answers[hashlib.sha256(sys.stdin.buffer.read()).hexdigest()]
if mode == "unsupported" or entry["status"] == "unsupported":
    print(json.dumps({"event": "end", "status": "unsupported", "actions_accepted": 0, "reason": "variant"}), flush=True)
    sys.exit(0)
sides = {"w": "white", "b": "black", None: None}
for index, (side, in_check, legal, terminal) in enumerate(entry["positions"]):
    if where == f"{entry['id']}@{index}":
        in_check = (not in_check) if mode == "flip" else None
    print(json.dumps({"event": "position", "index": index, "side": sides[side], "in_check": in_check,
                      "legal_actions": legal, "terminal": terminal}), flush=True)
# The engine reads the notation with its own parser: where the oracle could not
# match a move, the engine's way of stopping at the same action is to call it illegal.
reason = "illegal_move" if entry["status"] == "rejected" else entry["reason"]
print(json.dumps({"event": "end", "status": entry["status"], "actions_accepted": entry["actions_accepted"],
                  "reason": reason}), flush=True)
'''

# A position where ftxi says "not in check", in a ruleset ftxi replays completely.
WRONG_AT = ("ftxi/verysmall", 1)
WRONG_BOARD = "Very Small - Open"
# A ruleset in which ftxi stops at a move it cannot match. The stand-in stops at
# the same action - but ftxi's "no matching move" confirms nothing about it.
UNCONFIRMED_BOARD = "Standard - Turn Zero"


def _ftxi_built() -> bool:
    pin = lockfile.load().pin("ftxi")
    return gitutil.is_checkout(pin.checkout) and build.load(pin) is not None


@unittest.skipUnless(_ftxi_built(), "the ftxi oracle is not built")
class GateWithAStandInEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = corpus.load()
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        root = Path(cls.folder.name)

        recorded = json.loads(golden.golden_path("ftxi").read_text(encoding="utf-8"))["records"]
        answers = {}
        for record in cls.records:
            fed = corpus.read(record, cls.lock)
            answers[sha256_bytes(fed)] = {"id": record.id, **recorded[record.id]}
        cls.answers = root / "answers.json"
        cls.answers.write_text(json.dumps(answers), encoding="utf-8")
        cls.engine = root / "engine.py"
        cls.engine.write_text(ENGINE, encoding="utf-8")
        cls.root = root

    def run_gate(self, mode: str = "-", divergences_text: str = "schema = 1\n"):
        command = json.dumps([sys.executable, str(self.engine), str(self.answers), mode])
        gate_file = self.root / "gate.toml"
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        self.assertIn("\ncommand = []\n", text)
        gate_file.write_text(text.replace("\ncommand = []\n", f"\ncommand = {command}\n"), encoding="utf-8")
        divergences_file = self.root / "known-divergences.toml"
        divergences_file.write_text(divergences_text, encoding="utf-8")
        # A test run must not overwrite the report of a real gate run.
        with mock.patch.object(paths, "GATE_FILE", gate_file), \
                mock.patch.object(paths, "DIVERGENCES_FILE", divergences_file), \
                mock.patch.object(gate, "_write_report") as report:
            self.assertEqual(len(config.load().engine_command), 4)
            result = gate.run_tier("commit", include_untracked=False, log=lambda message: None)
        report.assert_called_once()
        return result

    def test_an_engine_that_agrees_with_the_oracle_passes(self):
        result = self.run_gate()
        self.assertEqual(result.findings(), [])
        self.assertEqual(result.engine, "passed")
        self.assertTrue(result.ok)
        self.assertIn(WRONG_BOARD, result.gated)
        # ftxi gave every verdict at every position the stand-in reports.
        behind = result.gated[WRONG_BOARD]
        self.assertEqual(set(behind.checked.values()), {behind.positions})
        self.assertEqual(behind.oracles, 1)
        # A ruleset the stand-in reports as unsupported (it echoes ftxi, which does
        # not know this board) was never compared, so it is not gated.
        self.assertIn("Focused - Just Kings", result.ungated)
        # Nor is one where the only oracle stopped because it could not read a move.
        self.assertIn("no oracle followed", result.ungated[UNCONFIRMED_BOARD])
        # A commit-tier pass never enables the proof channel.
        self.assertTrue(result.proof_channel.startswith("DISABLED"))

    def test_one_wrong_verdict_fails_the_gate(self):
        record, index = WRONG_AT
        result = self.run_gate(f"flip:{record}@{index}")
        self.assertEqual(result.engine, "failed")
        self.assertFalse(result.ok)
        self.assertTrue(result.oracle_layer_ok)  # the oracles are fine; the engine is not
        messages = [finding.message for finding in result.findings()]
        self.assertEqual(messages, [f"{record} position {index}: in_check: engine=True vs ftxi=False"])
        self.assertNotIn(WRONG_BOARD, result.gated)
        self.assertIn(WRONG_BOARD, result.ungated)

    def test_one_missing_verdict_fails_the_gate(self):
        # Saying nothing is not a way past a comparison.
        record, index = WRONG_AT
        result = self.run_gate(f"null:{record}@{index}")
        self.assertEqual(result.engine, "failed")
        self.assertFalse(result.ok)
        findings = result.findings()
        self.assertEqual([finding.code for finding in findings], ["engine-failed"])
        self.assertIn("engine contract", findings[0].message)
        self.assertIn("in_check", findings[0].message)
        self.assertNotIn(WRONG_BOARD, result.gated)

    def test_an_engine_that_declines_everything_has_not_passed(self):
        result = self.run_gate("unsupported")
        self.assertEqual(result.gated, {})
        self.assertEqual(result.engine, "failed")
        self.assertFalse(result.ok)
        self.assertEqual([finding.code for finding in result.findings()], ["engine-nothing-gated"])

    def test_a_recorded_divergence_with_a_citation_is_accepted_and_nothing_else_is(self):
        record_id, index = WRONG_AT
        record = next(item for item in self.records if item.id == record_id)
        entry = f'''schema = 1
[[divergence]]
id = "KD-0001"
oracle = "ftxi"
pin = "{self.lock.pin("ftxi").commit}"
record = "{record_id}"
record_sha256 = "{record.sha256}"
kind = "in_check"
index = {index}
engine_value = "True"
oracle_value = "False"
rulebook = "A made-up citation, long enough to pass, for a test of the mechanism only."
upstream = "unreported: this divergence exists only inside a unit test"
added = "2026-10-01"
'''
        accepted = self.run_gate(f"flip:{record_id}@{index}", divergences_text=entry)
        self.assertEqual(accepted.findings(), [])
        self.assertEqual(accepted.engine, "passed")

        # The same entry, with the engine now agreeing with the oracle: stale.
        stale = self.run_gate(divergences_text=entry)
        self.assertEqual([finding.code for finding in stale.findings()], ["divergence-stale"])

        # The same entry without its citation is refused before anything runs.
        bare = entry.replace("A made-up citation, long enough to pass, for a test of the mechanism only.", "TBD")
        refused = self.run_gate(f"flip:{record_id}@{index}", divergences_text=bare)
        self.assertIn("divergence-citation", [finding.code for finding in refused.findings()])
        self.assertFalse(refused.ok)

        # An entry that misstates what the oracle says is refused too: the golden file knows.
        misquoted = entry.replace('oracle_value = "False"', 'oracle_value = "True"').replace(
            'engine_value = "True"', 'engine_value = "False"')
        refused = self.run_gate(divergences_text=misquoted)
        self.assertIn("divergence-golden", [finding.code for finding in refused.findings()])


if __name__ == "__main__":
    unittest.main()
