"""The whole gate, end to end, with a stand-in engine.

There is no engine yet, so these tests plug in a program that answers every
record with what the ftxi driver printed for it - and then ones that get a
verdict wrong, leave one out, decline every record, read a move as another
position, mark a piece's having moved differently, or say nothing of what they
set up - to show that the gate passes the first, fails the others, and accepts a
wrong verdict only once the difference is on record with a citation.

How the stand-in is fed. The engine contract asks for a state event before every
position: every board of the multiverse, as the engine set it up. No golden file
holds that - they hold its hashes - so the stand-in cannot be fed from them, and
nothing in the harness may make up boards. Instead the harness's own ftxi source
is run over the corpus once, and the lines its driver printed are recorded; the
stand-in replays them, cut down to exactly the fields the engine contract allows
(a driver's timings, and its start event, are not passed on). The recording is
made through the same adapter the gate uses, so a record that adapter declines
without running its driver is declined by the stand-in too.

This is a fixture of the gate's comparison machinery and nothing else: it shows
what the gate does with an engine's output, never what any engine would say.
The stand-in is not, and must never become, a way for an engine to answer: the
gate refuses an engine command inside the harness, and an engine that passed
an oracle's answers on would have to come by them some other way - which is
what the tests below plant on purpose, in a temporary directory, and nowhere else.

They run the real ftxi oracle, and are skipped where it has not been built or
where the golden files predate the harness's schema.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import build, config, corpus, gate, gitutil, lockfile, paths, util
from oracle_harness.adapters import RunOptions, make_source
from oracle_harness.adapters import base as adapter_base
from oracle_harness.util import sha256_bytes
from support import golden_files_predate_this_schema

# argv: lines.json, then any number of "<mode>:<record>@<k>" (or a bare mode) -
#   -                       nothing: replay the oracle's driver
#   flip:<record>@<k>       the opposite check verdict at position k of that record
#   null:<record>@<k>       no check verdict at all there
#   misread:<record>@<k>    from position k on, a piece of another colour on the first board:
#                           a move read as another position than the oracle's
#   unmark:<record>@<k>     at position k, one unmoved marker fewer: the same pieces, marked otherwise
#   stateless               no state events at all
#   unsupported             decline every record
#   tamper:<path>           append a line to that file before answering (a harness file - a test's copy of it)
# The harness appends --cap N, and --parse-only in parse mode: then the stand-in answers
# no rules question at all (in_check, legal_actions and terminal null).
ENGINE = r'''
import json, hashlib, sys
recorded = json.load(open(sys.argv[1], encoding="utf-8"))
parse_only = "--parse-only" in sys.argv
specs = [spec for spec in sys.argv[2:sys.argv.index("--cap")]]
modes = dict(spec.partition(":")[::2] for spec in specs)
if "tamper" in modes:
    with open(modes["tamper"], "a", encoding="utf-8") as handle:
        handle.write("# written by the engine while the gate ran\n")
entry = recorded.get(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())
if "unsupported" in modes or entry is None or entry["lines"] is None:
    print(json.dumps({"event": "end", "status": "unsupported", "actions_accepted": 0, "reason": "variant"}), flush=True)
    sys.exit(0)

def at(mode, index, onwards=False):
    record, _, k = modes.get(mode, "@").partition("@")
    return record == entry["id"] and (index >= int(k) if onwards else index == int(k))

def misread(board):
    label, turn, colour, placement = board.split(":", 3)
    first = next(i for i, char in enumerate(placement) if char.isalpha())
    return ":".join((label, turn, colour, placement[:first] + placement[first].swapcase() + placement[first + 1:]))

for line in entry["lines"].splitlines():
    event = json.loads(line)
    kind, index = event.get("event"), event.get("index")
    if kind == "state":
        if "stateless" in modes:
            continue
        boards = list(event["boards"])
        if at("misread", index, onwards=True):
            boards[0] = misread(boards[0])
        if at("unmark", index):
            marked = next(i for i, board in enumerate(boards) if "*" in board)
            boards[marked] = boards[marked].replace("*", "", 1)
        out = {"event": "state", "index": index, "size": event["size"], "markers": event["markers"], "boards": boards}
    elif kind == "position":
        out = {key: event[key] for key in ("event", "index", "side", "in_check", "legal_actions", "terminal")}
        if at("flip", index):
            out["in_check"] = not out["in_check"]
        if at("null", index):
            out["in_check"] = None
        if parse_only:
            out.update(in_check=None, legal_actions=None, terminal=None)
    elif kind == "end":
        # The engine reads the notation with its own parser: where the oracle could not
        # match a move, the engine's way of stopping at the same action is to call it illegal.
        out = {"event": "end", "status": event["status"], "actions_accepted": event["actions_accepted"]}
        if event["status"] == "rejected":
            out["reason"] = "illegal_move"
        elif event.get("reason"):
            out["reason"] = event["reason"]
    else:
        continue  # the driver's start event, and anything else of its own
    print(json.dumps(out), flush=True)
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
        stale = golden_files_predate_this_schema("ftxi")
        if stale:
            raise unittest.SkipTest(stale)
        cls.lock = lockfile.load()
        cls.records = corpus.load()
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        root = Path(cls.folder.name)

        # Record what the ftxi driver prints for each record, through the gate's
        # own ftxi source, under the budget the golden files are recorded with.
        source = make_source("ftxi", cls.lock)
        printed: dict[str, str] = {}

        def recording(argv, **options):
            done = util.run(argv, **options)
            printed[sha256_bytes(options["stdin"])] = done.out_text()
            return done

        budget = config.load().tier("release")
        recorded = {}
        with mock.patch.object(adapter_base, "run", recording):
            for record in cls.records:
                fed = corpus.read(record, cls.lock)
                observation = source.observe(record.id, fed, RunOptions(cap=config.load().cap,
                                                                         timeout_s=budget.timeout_for("ftxi")))
                if observation.status == "timeout":
                    raise AssertionError(f"ftxi did not finish {record.id} within its recording budget")
                recorded[sha256_bytes(fed)] = {"id": record.id, "lines": printed.get(sha256_bytes(fed))}
        if not any('"event":"state"' in (entry["lines"] or "").replace(" ", "") for entry in recorded.values()):
            raise AssertionError("the built ftxi driver prints no state events: rebuild it from a driver that does "
                                 "(python oracles/harness.py build --oracle ftxi)")
        cls.recorded = root / "lines.json"
        cls.recorded.write_text(json.dumps(recorded), encoding="utf-8")
        cls.engine = root / "engine.py"
        cls.engine.write_text(ENGINE, encoding="utf-8")
        cls.root = root

    def run_gate(self, *modes: str, divergences_text: str = "schema = 1\n", mode: str = "full"):
        command = json.dumps([sys.executable, str(self.engine), str(self.recorded), *(modes or ("-",))])
        gate_file = self.root / "gate.toml"
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        self.assertIn("\ncommand = []\n", text)
        text = text.replace("\ncommand = []\n", f"\ncommand = {command}\n").replace('mode = "full"', f'mode = "{mode}"')
        gate_file.write_text(text, encoding="utf-8")
        divergences_file = self.root / "known-divergences.toml"
        divergences_file.write_text(divergences_text, encoding="utf-8")
        # A test run must not overwrite the report of a real gate run. It runs the
        # commit tier's ftxi source only: the stand-in replays ftxi, and what any
        # other source of the tier says differently from ftxi (the notation
        # parser, which cannot read every record ftxi plays) is a disagreement
        # between oracles - DISAGREEMENTS.md's business, not this test's.
        with mock.patch.object(paths, "GATE_FILE", gate_file), \
                mock.patch.object(paths, "DIVERGENCES_FILE", divergences_file), \
                mock.patch.object(gate, "_write_report") as report:
            self.assertGreaterEqual(len(config.load().engine_command), 4)
            result = gate.run_tier("commit", only_sources=["ftxi"], include_untracked=False, log=lambda message: None)
        report.assert_called_once()
        return result

    def test_an_engine_that_agrees_with_the_oracle_passes(self):
        result = self.run_gate()
        self.assertEqual(result.findings(), [])
        self.assertEqual(result.engine, "passed")
        self.assertTrue(result.ok)
        self.assertIn(WRONG_BOARD, result.gated)
        # ftxi gave every verdict at every position the stand-in reports, and
        # said what it set up at each, unmoved markers included: every position
        # is known to be the same one.
        behind = result.gated[WRONG_BOARD]
        self.assertEqual(set(behind.checked.values()), {behind.positions})
        self.assertEqual((behind.aligned, behind.matched), (behind.positions, behind.positions))
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

    def test_an_engine_that_does_not_say_what_it_set_up_fails_the_gate(self):
        result = self.run_gate("stateless")
        self.assertEqual(result.engine, "failed")
        codes = {finding.code for finding in result.findings()}
        self.assertEqual(codes, {"engine-failed"})
        self.assertTrue(all("a state event with markers true" in finding.message for finding in result.findings()))
        self.assertEqual(result.gated, {})

    def test_a_move_read_as_another_position_is_a_parse_divergence_and_hides_the_rest(self):
        # The stand-in sets up another position from WRONG_AT on, and gives the
        # wrong check verdict there too. Only the parse divergence is reported:
        # its check verdict is about another position than ftxi's.
        record, index = WRONG_AT
        result = self.run_gate(f"misread:{record}@{index}", f"flip:{record}@{index}")
        self.assertEqual(result.engine, "failed")
        messages = [finding.message for finding in result.findings()]
        self.assertEqual(len(messages), 1, messages)
        self.assertTrue(messages[0].startswith(f"{record} position {index}: placement: engine="), messages[0])
        self.assertIn("parse divergence", messages[0])
        self.assertNotIn(WRONG_BOARD, result.gated)

    def test_a_marker_only_divergence_is_reported_beside_a_rules_disagreement(self):
        record, index = WRONG_AT
        result = self.run_gate(f"unmark:{record}@{index}", f"flip:{record}@{index}")
        messages = sorted(finding.message for finding in result.findings())
        self.assertEqual(len(messages), 2, messages)
        self.assertTrue(messages[0].startswith(f"{record} position {index}: in_check: engine=True vs ftxi=False"))
        self.assertTrue(messages[1].startswith(f"{record} position {index}: unmoved: engine="), messages[1])

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

        # The same entry, with the engine now saying what the oracle is on record as getting wrong: a
        # telltale (H34) - its logic may come from that oracle - and never "stale", whose remedy is to remove it.
        copied = self.run_gate(divergences_text=entry)
        self.assertEqual([finding.code for finding in copied.findings()], ["telltale"])
        self.assertIn(f"the engine reproduces ftxi's recorded error at {record_id} position {index} (in_check = False)",
                      copied.findings()[0].message)
        self.assertFalse(copied.ok)

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

    def test_an_engine_that_writes_to_the_harness_while_it_runs_fails(self):
        # The stand-in appends a line to the known-divergences file (the test's copy) while it runs.
        result = self.run_gate(f"tamper:{self.root / 'known-divergences.toml'}")
        self.assertFalse(result.ok)
        self.assertEqual({finding.code for finding in result.findings()}, {"harness-modified"})

    def test_an_engine_in_parse_mode_is_compared_on_its_reading_only(self):
        # A wrong check verdict is never seen in parse mode: it answers none.
        record, index = WRONG_AT
        result = self.run_gate(f"flip:{record}@{index}", mode="parse")
        self.assertEqual(result.findings(), [])
        self.assertEqual((result.engine_mode, result.engine, result.gated), ("parse", "passed", {}))
        self.assertGreater(result.parse.boards, 0)
        self.assertEqual(result.parse.boards, result.parse.described)
        self.assertIn("engine differential: PARSE MODE - PASSED", gate.summarise(result))
        self.assertTrue(result.proof_channel.startswith("DISABLED - parse mode"))
        # What it set up is still compared.
        misread = self.run_gate(f"misread:{record}@{index}", mode="parse")
        self.assertEqual([finding.code for finding in misread.findings()], ["divergence"])
        self.assertIn("placement", misread.findings()[0].message)


if __name__ == "__main__":
    unittest.main()
