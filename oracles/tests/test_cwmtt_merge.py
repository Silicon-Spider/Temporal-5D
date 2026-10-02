"""cwmtt's two programs, merged: the stock CLI's answers and the state driver's
positions (decision H30). A count must never be paired with the state of another
position, so every way the two can read a record differently has to end in an
error that says so."""

import json
import unittest

from oracle_harness import build, corpus, lockfile
from oracle_harness.adapters import RunOptions, make_source
from oracle_harness.adapters.cwmtt import DRIVER_DEADLINE, merge
from oracle_harness.observations import validate
from oracle_harness.util import Completed, run

STANDARD_START_HASH = "d574889fd9da3f2bc65249ff27249b00"  # the notation specification's own example

START = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"
AFTER_E3 = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*"
AFTER_E6 = "r*nbqk*bnr*/p*p*p*p*1p*p*p*/4p3/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*"
# The boards cwmtt holds after 0, 1 and 2 actions of "1. e3 / e6".
BOARDS = [
    [f"0:1:w:{START}"],
    [f"0:1:w:{START}", f"0:1:b:{AFTER_E3}"],
    [f"0:1:w:{START}", f"0:1:b:{AFTER_E3}", f"0:2:w:{AFTER_E6}"],
]
SIDES = ["white", "black", "white"]


def cli(stdout: str, stderr: str = "", *, returncode=0, timed_out=False) -> Completed:
    return Completed(None if timed_out else returncode, stdout.encode(), stderr.encode(), timed_out)


def driver(positions: int, *, status="ok", parsed=None, unparsed="", failure=None, layout="Standard",
           extra=(), returncode=0, timed_out=False, end=True) -> Completed:
    """A state driver run that reports positions 0..positions-1, then ends as told."""
    events = [{"event": "start", "driver": "cwmtt", "schema": 1, "algorithm": "fast", "layout": layout,
               "parsed_actions": positions - 1 if parsed is None else parsed, "unparsed": unparsed}]
    for index in range(positions):
        events.append({"event": "state", "index": index, "size": "8x8", "markers": True, "boards": BOARDS[index]})
        events.append({"event": "position", "index": index, "side": SIDES[index]})
    events += list(extra)
    if end:
        closing = {"event": "end", "status": status, "actions_accepted": max(positions - 1, 0),
                   "parsed_actions": positions - 1 if parsed is None else parsed, "unparsed": unparsed,
                   "failure": failure}
        if status == "rejected":
            closing |= {"reason": "unrecognised", "detail": failure}
        elif status == "error":
            closing |= {"reason": "crash", "detail": "Prelude.head: empty list"}
        events.append(closing)
    out = "".join(json.dumps(event) + "\n" for event in events)
    return Completed(None if timed_out else returncode, out.encode(), b"", timed_out)


def merged(cli_run: Completed, driver_run: Completed, declared: str = "Standard"):
    return merge("cwmtt", "r/1", cli_run, driver_run, declared, 1000, 0.1)


class MergeTests(unittest.TestCase):
    def assertStateless(self, observation):
        self.assertTrue(all(position.placement is None and position.side is None for position in observation.positions))

    def test_the_two_runs_are_merged_by_position(self):
        observation = merged(cli("20 20 20 \n"), driver(3))
        self.assertEqual((observation.status, observation.actions_accepted), ("ok", 2))
        self.assertEqual([p.legal_actions for p in observation.positions], [20, 20, 20])
        self.assertEqual([p.side for p in observation.positions], SIDES)
        self.assertEqual([p.boards for p in observation.positions], [1, 2, 3])
        self.assertEqual(observation.positions[0].fingerprint, STANDARD_START_HASH)
        self.assertTrue(all(p.in_check is None for p in observation.positions))  # nobody answered it
        self.assertEqual(validate(observation), [])

    def test_the_outcome_is_the_clis(self):
        observation = merged(cli("20 20 \n", "In check\n"), driver(2, status="rejected", parsed=2, failure="In check"))
        self.assertEqual((observation.status, observation.reason, observation.actions_accepted),
                         ("rejected", "cannot_submit", 1))
        self.assertTrue(all(p.placement is not None for p in observation.positions))

    def test_a_different_number_of_positions_is_an_error(self):
        observation = merged(cli("20 20 20 \n"), driver(2))
        self.assertEqual((observation.status, observation.reason), ("error", "driver_mismatch"))
        self.assertIn("3 position(s), the state driver 2", observation.detail)
        self.assertStateless(observation)
        self.assertEqual([p.legal_actions for p in observation.positions], [20, 20, 20])  # the CLI's, untouched

    def test_a_different_replay_failure_is_an_error(self):
        failed_only_in_the_cli = merged(cli("20 20 \n", "In check\n"), driver(2))
        failed_only_in_the_driver = merged(cli("20 20 \n"), driver(2, status="rejected", parsed=2, failure="In check"))
        failed_differently = merged(cli("20 20 \n", "In check\n"),
                                    driver(2, status="rejected", parsed=2, failure="No hop to timeline 1"))
        for observation in (failed_only_in_the_cli, failed_only_in_the_driver, failed_differently):
            self.assertEqual((observation.status, observation.reason), ("error", "driver_mismatch"))
            self.assertStateless(observation)

    def test_different_unread_text_is_an_error(self):
        for cli_run, driver_run in (
            (cli('Unparsed: "? / e6\\n"\n20 20 \n'), driver(2, unparsed="")),
            (cli('Unparsed: "? / e6\\n"\n20 20 \n'), driver(2, unparsed="? / e5\n")),
            (cli("20 20 \n"), driver(2, unparsed="? / e6\n")),
        ):
            observation = merged(cli_run, driver_run)
            self.assertEqual((observation.status, observation.reason), ("error", "driver_mismatch"), observation.detail)

    def test_the_same_unread_text_is_read_through_haskells_escapes(self):
        observation = merged(cli('Unparsed: "? {caf\\233} / e6\\n"\n20 20 \n'), driver(2, unparsed="? {café} / e6\n"))
        self.assertEqual((observation.status, observation.reason), ("unsupported", "notation"))
        self.assertTrue(all(p.placement is not None for p in observation.positions))

    def test_a_layout_other_than_the_declared_one_is_an_error(self):
        # The adapter predicts which layout cwmtt sets up (would_misread); the
        # driver reports the one it did. A wrong prediction is never merged.
        observation = merged(cli("20 20 \n"), driver(2), declared="Standard - Turn Zero")
        self.assertEqual((observation.status, observation.reason), ("error", "driver_mismatch"))
        self.assertIn("Standard - Turn Zero", observation.detail)

    def test_the_position_after_half_an_action_is_dropped_with_its_state(self):
        observation = merged(cli('Unparsed: "=Q / Kd7\\n"\n20 20 \n'), driver(2, unparsed="=Q / Kd7\n"))
        self.assertEqual((observation.status, observation.reason, len(observation.positions)), ("unsupported", "notation", 1))
        self.assertEqual(observation.positions[0].fingerprint, STANDARD_START_HASH)
        self.assertEqual(validate(observation), [])

    def test_a_cli_cut_short_keeps_what_was_set_up_where_it_stopped(self):
        observation = merged(cli("20 20 3", timed_out=True), driver(3))
        self.assertEqual((observation.status, len(observation.positions), observation.actions_accepted), ("timeout", 3, 2))
        last = observation.positions[-1]
        self.assertEqual((last.legal_actions, last.terminal, last.side, last.boards), (None, None, "white", 3))
        self.assertEqual(validate(observation), [])

    def test_a_cli_crash_stays_the_outcome(self):
        observation = merged(cli("20 ", "cwmtt: Irrefutable pattern failed\n", returncode=1), driver(3))
        self.assertEqual((observation.status, observation.reason), ("error", "crash"))
        self.assertEqual([p.legal_actions for p in observation.positions], [20, None])
        self.assertTrue(all(p.placement is not None for p in observation.positions))

    def test_a_cli_that_reports_more_than_the_driver_reached_is_an_error(self):
        observation = merged(cli("20 20 20 3", timed_out=True), driver(2))
        self.assertEqual((observation.status, observation.reason), ("error", "driver_mismatch"))

    def test_a_failed_driver_leaves_an_error_without_state(self):
        crashed = merged(cli("20 20 \n"), driver(2, end=False, returncode=1))
        self.assertEqual((crashed.status, crashed.reason), ("error", "driver_error"))
        self.assertStateless(crashed)
        threw = merged(cli("20 20 \n"), driver(2, status="error"))
        self.assertEqual((threw.status, threw.reason), ("error", "driver_error"))
        self.assertIn("empty list", threw.detail)
        unreadable = merged(cli("20 20 \n"), Completed(0, b'{"event":"state","index":0,"size":"8x8"}\n', b"", False))
        self.assertEqual((unreadable.status, unreadable.reason), ("error", "driver_error"))

    def test_a_driver_that_contradicts_itself_is_an_error(self):
        for driver_run in (driver(2, parsed=5),  # replayed everything it parsed, it says - but 1 of 5
                           driver(2, failure="In check"),  # a failure on a replay that did not fail
                           driver(2, status="rejected", parsed=1, failure="x"),  # refused an action it never parsed
                           driver(2, status="rejected", parsed=2, failure="")):  # refused one without saying why
            observation = merged(cli("20 20 \n"), driver_run)
            self.assertEqual((observation.status, observation.reason), ("error", "driver_error"), observation.detail)

    def test_a_driver_cut_short_keeps_the_clis_outcome_and_the_states_it_printed(self):
        # The driver searches nothing, so this is rare; the CLI's answers and
        # outcome stand, and what the driver did print is paired with them.
        observation = merged(cli("20 20 \n"), driver(1, end=False, timed_out=True))
        self.assertEqual((observation.status, observation.reason), ("ok", ""))
        self.assertIn("the state driver ran out of time after 1 position(s)", observation.detail)
        self.assertEqual([p.legal_actions for p in observation.positions], [20, 20])
        self.assertEqual([p.fingerprint for p in observation.positions], [STANDARD_START_HASH, None])
        self.assertEqual(validate(observation), [])
        # Where it did not get to, the side and the state are unknown at its deadline (H32): a run that loses them
        # there has lost them to a budget.
        self.assertEqual(observation.extras, {1: {"limit": DRIVER_DEADLINE}})
        source = make_source("cwmtt", lockfile.load())
        self.assertEqual(source.limits(observation, 1, RunOptions(timeout_s=120)),
                         dict.fromkeys(("side", "fingerprint", "placement", "boards", "pieces"), "timeout_s=120"))
        self.assertEqual(source.limits(observation, 0, RunOptions(timeout_s=120)), {})
        # A CLI that refused an action keeps its verdict.
        refused = merged(cli("20 20 \n", "In check\n"), driver(2, end=False, timed_out=True, parsed=2))
        self.assertEqual((refused.status, refused.reason), ("rejected", "cannot_submit"))
        self.assertTrue(all(p.placement is not None for p in refused.positions))

    def test_a_driver_cut_short_is_still_held_to_the_clis_reading(self):
        for cli_run, driver_run, declared in (
                (cli("20 20 \n"), driver(1, end=False, timed_out=True), "Standard - Turn Zero"),
                (cli('Unparsed: "? / e6\\n"\n20 20 \n'), driver(1, end=False, timed_out=True, unparsed="? / e5\n"),
                 "Standard"),
                (cli("20 20 \n"), driver(3, end=False, timed_out=True), "Standard")):  # more than the replay has
            observation = merged(cli_run, driver_run, declared)
            self.assertEqual((observation.status, observation.reason), ("error", "driver_mismatch"), observation.detail)
            self.assertStateless(observation)
        # One that ran out of time before saying anything leaves the CLI's report as it was.
        silent = merged(cli("20 20 \n"), Completed(None, b"", b"", True))
        self.assertEqual((silent.status, len(silent.positions)), ("ok", 2))
        self.assertIn("before its start event", silent.detail)
        self.assertStateless(silent)

    def test_the_position_after_half_an_action_is_never_reported_however_the_cli_ended(self):
        # The parser stopped inside action 1 and the replay played the fragment:
        # position 1 belongs to no game the record describes.
        record = 'Unparsed: "=Q / Kd7\\n"\n'
        for cli_run, status in ((cli(record + "20 ", timed_out=True), "timeout"),
                                (cli(record + "20 20 ", timed_out=True), "timeout"),
                                (cli(record + "20 ", "cwmtt: Prelude.head\n", returncode=1), "error")):
            observation = merged(cli_run, driver(2, unparsed="=Q / Kd7\n"))
            self.assertEqual((observation.status, len(observation.positions)), (status, 1), cli_run.out_text())
            self.assertEqual(observation.positions[0].fingerprint, STANDARD_START_HASH)
            self.assertEqual(observation.actions_accepted, 0)
            self.assertEqual(validate(observation), [])
        # Nor by a driver cut short.
        cut = merged(cli(record + "20 20 \n"), driver(2, unparsed="=Q / Kd7\n", end=False, timed_out=True))
        self.assertEqual((cut.status, len(cut.positions)), ("unsupported", 1))

    def test_a_merged_position_has_arrived_when_both_its_halves_have(self):
        # The CLI's counts at 1, 2 and 3 s; the driver's states a little after its own start. Each
        # program runs against its own deadline, so each half is timed from its own program's start.
        counts = b"20 20 20 \n"
        cli_run = Completed(0, counts, b"", False, arrivals=((3, 1.0), (6, 2.0), (len(counts), 3.0)), elapsed_s=3.2)
        plain = driver(3)
        offsets = [index for index, byte in enumerate(plain.stdout) if byte == ord("\n")]
        # The driver's lines a half second apart: its position events (the 3rd, 5th and 7th lines) at 1.5, 2.5, 3.5 s.
        arrivals = tuple((offset + 1, 0.5 * (number + 1)) for number, offset in enumerate(offsets))
        timed = Completed(0, plain.stdout, b"", False, arrivals=arrivals, elapsed_s=4.0)
        observation = merged(cli_run, timed)
        self.assertEqual(observation.status, "ok")
        self.assertEqual(observation.arrival_s, [1.5, 2.5, 3.5])  # each the later of its count and its state
        self.assertEqual(observation.ended_s, 4.0)  # the later of the CLI's end and the driver's end event
        # Where the driver's state is not paired (the two read the record differently), only the count is timed.
        mismatched = merged(cli_run, Completed(0, driver(2).stdout, b"", False, arrivals=arrivals[:5], elapsed_s=4.0))
        self.assertEqual((mismatched.reason, mismatched.arrival_s), ("driver_mismatch", [1.0, 2.0, 3.0]))

    def test_a_cli_stopped_once_it_had_every_count_needed_keeps_the_next_state(self):
        # A gate run that needs only positions 0 and 1 stops the CLI; the driver runs to its end.
        observation = merged(Completed(None, b"20 20 2", b"", False, stopped=True), driver(3))
        self.assertEqual((observation.status, observation.reason), ("timeout", "stopped"))
        self.assertEqual([p.legal_actions for p in observation.positions], [20, 20, None])
        self.assertEqual(observation.positions[2].boards, 3)  # what cwmtt set up where the CLI stopped
        self.assertEqual(validate(observation), [])

    def test_diagnostics_are_passed_on(self):
        explain = {"event": "explain", "index": 0, "playable": [{"L": "0", "T": 1, "c": "w"}], "present": 1}
        observation = merged(cli("20 \n"), driver(1, extra=[explain]))
        self.assertEqual(observation.diagnostics, [explain])

    def test_the_signature_covers_the_driver(self):
        signature = make_source("cwmtt", lockfile.load()).signature(RunOptions())
        self.assertIn("driver_sha256", signature)
        self.assertIn("adapter_sha256", signature)


class LiveDriverTests(unittest.TestCase):
    """Run only where cwmtt is built."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = {record.id: record for record in corpus.load()}
        cls.built = build.load(cls.lock.pin("cwmtt"))

    def events(self, record_id: str, *args: str) -> list[dict]:
        if self.built is None or "driver" not in self.built.artifacts:
            self.skipTest("cwmtt is not built")
        data = corpus.read(self.records[record_id], self.lock)
        done = run([self.built.artifact("driver"), *args], stdin=data, timeout=120)
        self.assertEqual(done.returncode, 0, done.err_text())
        return [json.loads(line) for line in done.out_text().splitlines()]

    def test_the_standard_start_is_the_notations_example(self):
        source = make_source("cwmtt", self.lock)
        if source.available():
            self.skipTest(source.available())
        data = corpus.read(self.records["cwmtt/standard"], self.lock)
        observation = source.observe("cwmtt/standard", data, RunOptions(cap=1000, timeout_s=120))
        self.assertEqual(observation.status, "ok")
        self.assertEqual(observation.positions[0].fingerprint, STANDARD_START_HASH)
        self.assertEqual(len(observation.positions), 17)
        self.assertTrue(all(p.placement is not None for p in observation.positions))

    def test_turn_zero_sets_up_two_boards(self):
        state = next(event for event in self.events("cwmtt/small") if event["event"] == "state")
        self.assertEqual(sorted(board[:6] for board in state["boards"]), ["0:0:b:", "0:1:w:"])

    def test_the_actions_listed_are_the_actions_counted(self):
        # notation/1.2, position 15: cwmtt counts 54 legal actions there (ftxi 52).
        events = self.events("notation/1.2", "--actions", "15")
        listed = next(event for event in events if event["event"] == "actions")
        self.assertEqual((listed["index"], listed["count"], listed["truncated"], len(listed["actions"])), (15, 54, False, 54))
        self.assertEqual(len({json.dumps(action, sort_keys=True) for action in listed["actions"]}), 54)


if __name__ == "__main__":
    unittest.main()
