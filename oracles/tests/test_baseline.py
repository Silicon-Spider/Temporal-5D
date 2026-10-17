"""The engine's baseline: what it read on the last green run (decision H47, CHECKPOINT 8; baseline.py)."""

import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import baseline, config, paths
from oracle_harness.corpus import Record
from oracle_harness.observations import Observation, Position
from oracle_harness.util import Finding, HarnessError, sha256_text


def record(name, sha256="s"):
    return Record(id=name, source="fake", path=name, blob="b", sha256=sha256, size=1, board="Standard", licence="x",
                  tier="commit")


def run(record_id, status="ok", positions=3, reason=""):
    """An engine report: ``positions`` positions, ending ``status`` (``reason``)."""
    return Observation(source="engine", record=record_id, status=status, reason=reason,
                       positions=[Position(index) for index in range(positions)],
                       actions_accepted=max(positions - 1, 0), cap=1000)


ENGINE = {"command": ["./build/engine/t5d-gate"], "mode": "full", "sha256": "e" * 64}
RECORDS = [record("r/ok"), record("r/refused"), record("r/unread"), record("r/setup"), record("r/declined")]
READS = {"r/ok": run("r/ok"), "r/refused": run("r/refused", "rejected", 4, "cannot_submit"),
         "r/unread": run("r/unread", "rejected", 10, "unreadable"), "r/setup": run("r/setup", "rejected", 0, "unreadable"),
         "r/declined": run("r/declined", "unsupported", 0, "variant")}


class ReadingTests(unittest.TestCase):
    """What "declines a record it read on the last green run" and "reads none" mean, exactly."""

    def test_a_record_is_declined_when_less_of_it_is_read_than_on_the_last_green_run(self):
        less = baseline.reads_less
        # Fewer positions than then, however it ends.
        self.assertTrue(less("ok", 17, "ok", 16))
        self.assertTrue(less("rejected:unreadable", 10, "rejected:unreadable", 9))
        self.assertTrue(less("rejected:unreadable", 10, "unsupported", 0))
        self.assertTrue(less("rejected:cannot_submit", 4, "rejected:illegal_move", 3))
        # As many, without the verdict it ended with then: other than ok where it ended ok ...
        self.assertTrue(less("ok", 17, "rejected:unreadable", 17))
        self.assertTrue(less("ok", 17, "error", 17))
        # ... or other than a refusal as illegal where it ended in one (a refusal as unreadable is no verdict).
        self.assertTrue(less("rejected:cannot_submit", 4, "rejected:unreadable", 4))
        self.assertTrue(less("rejected:illegal_move", 4, "timeout", 4))
        # Not a decline: the same reading, a refusal as illegal for another reason, or reading further.
        self.assertFalse(less("ok", 17, "ok", 17))
        self.assertFalse(less("rejected:cannot_submit", 4, "rejected:illegal_move", 4))
        self.assertFalse(less("rejected:unreadable", 10, "rejected:unreadable", 10))
        self.assertFalse(less("rejected:unreadable", 10, "ok", 17))
        self.assertFalse(less("rejected:illegal_move", 4, "ok", 17))
        # A record it read nothing of then - declined, or its set-up refused - cannot be declined.
        for end in ("unsupported", "rejected:unreadable"):
            for now in (("unsupported", 0), ("rejected:unreadable", 0), ("ok", 5)):
                self.assertFalse(less(end, 0, *now), (end, now))

    def test_reading_a_record_is_replaying_it(self):
        # To its end, or to an action held illegal: a refusal as unreadable, a declined ruleset, an error and a time-out
        # replay nothing.
        self.assertEqual({record_id: baseline.replays(item) for record_id, item in READS.items()},
                         {"r/ok": True, "r/refused": True, "r/unread": False, "r/setup": False, "r/declined": False})
        self.assertFalse(baseline.replays(run("r/x", "error", 3, "crash")))
        self.assertFalse(baseline.replays(run("r/x", "timeout", 3, "timeout")))

    def test_how_far_a_run_read_is_said_in_words(self):
        self.assertEqual([baseline.describe_read(baseline.end_of(item), len(item.positions)) for item in READS.values()],
                         ["read it to its end (3 positions)", "refused action 4 as cannot_submit (4 positions)",
                          "could not read action 10 (10 positions)", "refused its set-up as unreadable (no position)",
                          "declined its ruleset (no position)"])


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.recorded = baseline.from_run("commit", RECORDS, READS, ENGINE, recorded="2026-10-17")

    def compare(self, **changed):
        return baseline.compare(self.recorded, RECORDS, {**READS, **changed})

    def test_the_same_reading_is_no_finding(self):
        self.assertEqual(self.compare(), ([], "5 record(s) compared with what the engine read on the last green run "
                                             "(commit tier, recorded 2026-10-17): 0 declined"))

    def test_each_record_read_less_is_a_finding_that_says_how_far_it_was_read_then_and_now(self):
        found, summary = self.compare(**{"r/ok": run("r/ok", "unsupported", 0, "variant"),
                                         "r/unread": run("r/unread", "rejected", 2, "unreadable")})
        self.assertEqual(found, [
            Finding("engine-declined", "r/ok: the engine declined its ruleset (no position), where on the last green run "
                                       "(commit tier, recorded 2026-10-17) it read it to its end (3 positions) - it "
                                       "declines a record it read then (H47)"),
            Finding("engine-declined", "r/unread: the engine could not read action 2 (2 positions), where on the last "
                                       "green run (commit tier, recorded 2026-10-17) it could not read action 10 (10 "
                                       "positions) - it declines a record it read then (H47)")])
        self.assertTrue(summary.endswith(": 2 declined"))

    def test_reading_further_is_no_finding_but_is_said(self):
        found, summary = self.compare(**{"r/unread": run("r/unread", "ok", 12), "r/declined": run("r/declined", "ok", 5)})
        self.assertEqual(found, [])
        self.assertEqual(summary, "5 record(s) compared with what the engine read on the last green run (commit tier, "
                                  "recorded 2026-10-17): 0 declined; 2 read further than then - record a new baseline "
                                  f"({baseline.WRITE_COMMAND}) to hold the engine to them")

    def test_what_the_baseline_does_not_hold_or_was_not_run_is_not_compared(self):
        # A record not in the baseline, one recorded for other text, one the run did not run, and an error or a
        # time-out (engine-failed reports those).
        records = [*RECORDS, record("r/new")]
        records[0] = record("r/ok", sha256="t")
        reads = {**READS, "r/new": run("r/new", "unsupported", 0, "variant"),
                 "r/refused": run("r/refused", "error", 0, "crash")}
        del reads["r/unread"]
        found, summary = baseline.compare(self.recorded, records, reads)
        self.assertEqual(found, [])
        self.assertTrue(summary.startswith("2 record(s) compared"), summary)

    def test_reading_none_is_a_finding_of_its_own(self):
        self.assertIsNone(baseline.read_nothing(RECORDS, READS))
        none = {record_id: item for record_id, item in READS.items() if not baseline.replays(item)}
        self.assertIsNone(baseline.read_nothing(RECORDS, {**none, "r/refused": READS["r/refused"]}))
        finding = baseline.read_nothing(RECORDS, {**none, "r/ok": run("r/ok", "timeout", 2, "timeout"),
                                                  "r/refused": run("r/refused", "rejected", 1, "unreadable")})
        self.assertEqual(finding, Finding(
            "engine-read-nothing",
            "the engine replayed none of the 5 record(s) it was given - none to its end, nor to an action it holds "
            "illegal (3 rejected:unreadable, 1 timeout, 1 unsupported): an engine that reads nothing is compared with "
            "nothing (H47)"))
        self.assertIsNone(baseline.read_nothing([], {}))  # no record: the tier's no-records finding says so


class FileTests(unittest.TestCase):
    """The file: one line per record, pinned in gate.toml, verified before every comparison (H33)."""

    def setUp(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(paths, "ENGINE_BASELINE_FILE", folder / "engine-baseline.json"))
        self.gate_file = folder / "gate.toml"
        self.enterContext(mock.patch.object(paths, "GATE_FILE", self.gate_file))
        self.gate_file.write_bytes((paths.HARNESS_ROOT / "gate.toml").read_bytes())
        config.write_sha256(self.gate_file, "baseline", "")
        self.recorded = baseline.from_run("commit", RECORDS, READS, ENGINE, recorded="2026-10-17")

    def cfg(self, **engine):
        return dataclasses.replace(config.load(), **engine)

    def test_written_then_pinned_and_read_back_as_written(self):
        old, new = baseline.write(self.recorded)
        text = paths.ENGINE_BASELINE_FILE.read_text(encoding="utf-8")
        self.assertEqual((old, new, config.load().baseline.sha256), ("", sha256_text(text.encode()), new))
        self.assertIn('    "r/refused": {"end": "rejected:cannot_submit", "positions": 4, "sha256": "s"},\n', text)
        self.assertIn('"engine": {"command": ["./build/engine/t5d-gate"], "mode": "full", "platform": ', text)
        loaded = baseline.load(new)
        self.assertEqual((loaded.tier, loaded.recorded, loaded.records, loaded.digest),
                         ("commit", "2026-10-17", self.recorded.records, new))
        self.assertEqual(loaded.read(), 2)
        # A second recording of the same reading is the same text: nothing in it but what was read changes.
        self.assertEqual(baseline.dump(dataclasses.replace(loaded, digest="")), text)

    def test_a_file_that_is_not_the_one_pinned_is_never_read(self):
        _, pin = baseline.write(self.recorded)
        text = paths.ENGINE_BASELINE_FILE.read_text(encoding="utf-8")
        paths.ENGINE_BASELINE_FILE.write_text(text.replace('"positions": 10', '"positions": 1'), encoding="utf-8")
        with self.assertRaises(baseline.BaselineError) as raised:
            baseline.load(pin)
        self.assertEqual(raised.exception.code, "engine-baseline-integrity")
        self.assertIn("a recorded answer is never edited by hand", str(raised.exception))
        paths.ENGINE_BASELINE_FILE.unlink()
        with self.assertRaises(baseline.BaselineError) as raised:
            baseline.load(pin)
        self.assertEqual(raised.exception.code, "engine-baseline-missing")
        # One that matches its pin and is no baseline: invalid.
        for bad in ('{"schema": 2}', '{"schema": 1, "tier": "commit", "recorded": "x", "engine": {}, "records": '
                                    '{"r/1": {"end": "error", "positions": 3, "sha256": "s"}}}',
                    '{"schema": 1, "tier": "commit", "recorded": "x", "engine": {}, "records": '
                    '{"r/1": {"end": "ok", "positions": 0, "sha256": "s"}}}', "not json"):
            paths.ENGINE_BASELINE_FILE.write_text(bad, encoding="utf-8")
            with self.assertRaises(baseline.BaselineError, msg=bad) as raised:
                baseline.load(sha256_text(bad.encode()))
            self.assertEqual(raised.exception.code, "engine-baseline-invalid", bad)

    def test_the_static_check(self):
        cfg = self.cfg(engine_command=("./build/engine/t5d-gate",))
        # Not pinned, with an engine compared in full mode: missing - and not needed without one, or in parse mode.
        [missing] = baseline.static_findings(cfg, RECORDS, None, [])
        self.assertEqual(missing.code, "engine-baseline-missing")
        self.assertIn(baseline.WRITE_COMMAND, missing.message)
        self.assertEqual(baseline.static_findings(self.cfg(engine_command=()), RECORDS, None, []), [])
        self.assertEqual(baseline.static_findings(self.cfg(engine_command=("x",), engine_mode="parse"), RECORDS, None, []),
                         [])
        _, pin = baseline.write(self.recorded)
        cfg = self.cfg(engine_command=("./build/engine/t5d-gate",))
        loaded = baseline.load(pin)
        self.assertEqual(baseline.static_findings(cfg, RECORDS, loaded, []), [])
        # What the snapshot could not read is reported as it was found.
        problem = Finding("engine-baseline-integrity", "...")
        self.assertEqual(baseline.static_findings(cfg, RECORDS, None, [problem]), [problem])
        # Stale: another tier's run, a record the corpus no longer holds, a record whose text changed.
        stale = dataclasses.replace(loaded, tier="nightly")
        self.assertEqual([item.code for item in baseline.static_findings(cfg, RECORDS, stale, [])],
                         ["engine-baseline-stale"])
        corpus = [record("r/ok", sha256="t"), *RECORDS[1:4]]
        messages = [item.message for item in baseline.static_findings(cfg, corpus, loaded, [])]
        self.assertEqual(len(messages), 2)
        self.assertIn("holds r/declined, which is not in the corpus", messages[0])
        self.assertIn("r/ok was recorded for other record content", messages[1])

    def test_a_new_baseline_lists_what_it_holds_otherwise_losses_first(self):
        self.assertEqual(baseline.changes(None, self.recorded), ["no earlier baseline could be read: all 5 record(s) are new"])
        reads = {**READS, "r/ok": run("r/ok", "unsupported", 0, "variant"), "r/unread": run("r/unread", "ok", 12)}
        del reads["r/declined"]
        records = [item for item in RECORDS if item.id != "r/declined"] + [record("r/added")]
        reads["r/added"] = run("r/added", "ok", 2)
        new = baseline.from_run("commit", records, reads, ENGINE)
        self.assertEqual(baseline.changes(self.recorded, new), [
            "LOSS: r/declined is no longer held (the engine declined its ruleset (no position))",
            "LOSS: r/ok - the engine declined its ruleset (no position), where it read it to its end (3 positions)",
            "further: r/unread - the engine read it to its end (12 positions), where it could not read action 10 "
            "(10 positions)",
            "added: r/added - the engine read it to its end (2 positions)"])
        # The previous file is read leniently - unpinned - for this list only.
        self.assertIsNone(baseline.read_previous())
        baseline.write(self.recorded)
        self.assertEqual(baseline.read_previous().records, self.recorded.records)

    def test_the_pin_needs_its_table(self):
        self.gate_file.write_text(self.gate_file.read_text(encoding="utf-8").replace("[baseline]\n", "[baseline_]\n"),
                                  encoding="utf-8")
        with self.assertRaises(HarnessError):
            baseline.pin("f" * 64)


if __name__ == "__main__":
    unittest.main()
