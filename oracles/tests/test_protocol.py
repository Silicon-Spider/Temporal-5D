import json
import sys
import tempfile
import unittest
from pathlib import Path

from oracle_harness import lockfile, observations
from oracle_harness.adapters import EngineSource, RunOptions, Source
from oracle_harness.adapters import chessin5d as chessin5d_adapter
from oracle_harness.adapters.base import code_sha256, parse_jsonl, run_protocol, stop_condition, watch_events
from oracle_harness.adapters.cwmtt import (
    classify_warning,
    count_arrivals,
    cwmtt_tags,
    parse_count_output,
    remainder_kind,
    unparsed_is_only_trailer,
    watch_counts,
    would_misread,
)
from oracle_harness.adapters.engine import REJECTIONS, contract_violation
from oracle_harness.adapters.ftxi import ends_in_a_variation
from oracle_harness.observations import HARD_REJECTIONS, SOFT_REJECTIONS, Observation, Position
from oracle_harness.util import Completed


def lines(*events) -> bytes:
    return "".join(json.dumps(event) + "\n" for event in events).encode()


def position(index, side="white", in_check=False, legal=20, terminal="none"):
    return {"event": "position", "index": index, "side": side, "in_check": in_check,
            "legal_actions": legal, "terminal": terminal}


def end(status="ok", accepted=0, **extra):
    return {"event": "end", "status": status, "actions_accepted": accepted, **extra}


START = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"
AFTER_E3 = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*"
# Boards of the Standard start, and after 1. e3: the start board, and the board
# that move made, with black to move on it.
BOARDS = {0: [f"0:1:w:{START}"], 1: [f"0:1:w:{START}", f"0:1:b:{AFTER_E3}"]}
START_HASH = "d574889fd9da3f2bc65249ff27249b00"  # the notation specification's own example


def state(index, boards=None, markers=True, **extra):
    return {"event": "state", "index": index, "size": "8x8", "markers": markers,
            "boards": boards if boards is not None else BOARDS[min(index, 1)], **extra}


def turn(index, **answers):
    """What a source prints for one position: its state, then its answers."""
    return [state(index), position(index, **answers)]


class JsonLinesTests(unittest.TestCase):
    def parse(self, stdout: bytes, *, returncode=0, timed_out=False, stderr=b""):
        done = Completed(None if timed_out else returncode, stdout, stderr, timed_out)
        return parse_jsonl("src", "r/1", done, 1000, 0.5)

    def test_complete_run(self):
        out = lines({"event": "start"}, position(0), position(1, side="black"), end(accepted=1))
        observation = self.parse(out)
        self.assertEqual((observation.status, observation.actions_accepted, len(observation.positions)), ("ok", 1, 2))
        self.assertEqual(observation.positions[1].side, "black")
        self.assertEqual(observation.cap, 1000)

    def test_rejection_carries_its_reason(self):
        out = lines(position(0), end("rejected", reason="illegal_move", detail="Ke3", last_action=True))
        observation = self.parse(out)
        self.assertEqual((observation.status, observation.reason, observation.detail, observation.last_action),
                         ("rejected", "illegal_move", "Ke3", True))

    def test_every_known_rejection_reason_is_read(self):
        for reason in (*HARD_REJECTIONS, *SOFT_REJECTIONS):
            observation = self.parse(lines(position(0), end("rejected", reason=reason)))
            self.assertEqual((observation.status, observation.reason), ("rejected", reason))

    def test_killed_run_keeps_what_was_answered(self):
        out = lines(position(0), position(1)) + b'{"event":"position","index":2,"si'
        observation = self.parse(out, timed_out=True)
        self.assertEqual((observation.status, len(observation.positions), observation.actions_accepted), ("timeout", 2, 1))

    def test_crash_without_end_event_is_an_error(self):
        observation = self.parse(lines(position(0)), returncode=134, stderr=b"out of memory")
        self.assertEqual((observation.status, observation.reason), ("error", "crash"))
        self.assertIn("out of memory", observation.detail)
        self.assertEqual(len(observation.positions), 1)

    def test_malformed_events_are_not_trusted(self):
        bad_events = [
            position(3),  # wrong index
            {**position(0), "side": "purple"},
            {**position(0), "terminal": "winning"},
            {**position(0), "in_check": "yes"},
            {**position(0), "legal_actions": -1},
            {**position(0), "legal_actions": True},
        ]
        for event in bad_events:
            observation = self.parse(lines(event, end()))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), event)

    def test_a_source_cannot_claim_a_timeout_or_an_unknown_status(self):
        for status in ("timeout", "fine", None):
            out = lines(position(0), end(status))
            self.assertEqual(self.parse(out).status, "error", status)

    def test_the_tally_must_match_the_positions(self):
        # Two positions are the record after 0 and after 1 action. A run that
        # claims otherwise has lost count, and nothing it says is compared.
        for status in ("ok", "rejected", "unsupported"):
            for claimed in (0, 2, 5, -1, "1", None, True):
                out = lines(position(0), position(1, side="black"), end(status, claimed, reason="illegal_move"))
                observation = self.parse(out)
                self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), (status, claimed))
                self.assertEqual(observation.actions_accepted, 1)

    def test_a_reported_error_is_counted_by_what_was_observed(self):
        # A driver that failed while looking at position 2 had already accepted two actions.
        out = lines(position(0), position(1, side="black"), end("error", 2, reason="exception", detail="boom"))
        observation = self.parse(out)
        self.assertEqual((observation.status, observation.reason, observation.actions_accepted), ("error", "exception", 1))

    def test_a_replay_without_a_starting_position_is_not_a_replay(self):
        for status in ("ok", "rejected"):
            observation = self.parse(lines(end(status, reason="illegal_move")))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), status)
        self.assertEqual(self.parse(lines(end("unsupported", reason="variant"))).status, "unsupported")

    def test_an_invented_rejection_reason_is_not_a_rejection(self):
        observation = self.parse(lines(position(0), end("rejected", reason="did_not_like_it")))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))


def timed(events: list[dict], every: float = 1.0, *, timed_out=False, stopped=False, returncode=0) -> Completed:
    """A run that printed one event per line, the k-th line arriving at (k+1) x ``every`` seconds."""
    data, arrivals = b"", []
    for number, event in enumerate(events, 1):
        data += (json.dumps(event) + "\n").encode()
        arrivals.append((len(data), number * every))
    killed = timed_out or stopped
    return Completed(None if killed else returncode, data, b"", timed_out, arrivals=tuple(arrivals), stopped=stopped,
                     elapsed_s=len(events) * every + 0.5)


class ArrivalTimeTests(unittest.TestCase):
    """The cost of each position's answers: when its last line arrived (H32)."""

    def test_each_position_is_timed_by_the_line_that_completed_it(self):
        done = timed([*turn(0), *turn(1, side="black"), end(accepted=1)])
        observation = parse_jsonl("src", "r/1", done, 1000, 9.0)
        self.assertEqual(observation.arrival_s, [2.0, 4.0])  # the position events, not the state events
        self.assertEqual(observation.ended_s, 5.0)

    def test_a_search_cut_short_is_timed_by_its_state(self):
        done = timed([*turn(0), state(1)], timed_out=True)
        observation = parse_jsonl("src", "r/1", done, 1000, 9.0)
        self.assertEqual((observation.status, observation.arrival_s, observation.ended_s), ("timeout", [2.0, 3.0], None))

    def test_a_run_stopped_with_everything_it_needed_is_cut_short_not_failed(self):
        done = timed([*turn(0), state(1)], stopped=True)
        observation = parse_jsonl("src", "r/1", done, 1000, 9.0)
        self.assertEqual((observation.status, observation.reason), ("timeout", "stopped"))
        # A line cut off mid-write is what a stop leaves behind, as a time-out does.
        cut = Completed(None, lines(*turn(0)) + b'{"event":"state","ind', b"", False, stopped=True)
        self.assertEqual(parse_jsonl("src", "r/1", cut, 1000, 1.0).reason, "stopped")

    def test_a_stop_condition_watches_for_the_event_it_was_given(self):
        for kind, events, expected in (
                ("position", [state(0), position(0), state(1)], False),
                ("position", [state(0), position(0), state(1), position(1, side="black")], True),
                ("state", [state(0), position(0), state(1)], True),
                ("state", [state(0), position(0)], False)):
            watch = watch_events(1, kind)
            data = lines(*events)
            # Fed in awkward pieces: a line may arrive in several.
            fired = [watch(data[start:start + 7]) for start in range(0, len(data), 7)]
            self.assertEqual(any(fired), expected, (kind, events))
        self.assertIsNone(stop_condition(RunOptions()))
        with self.assertRaises(ValueError):
            stop_condition(RunOptions(stop_after=(1, "end")))

    def test_a_protocol_run_is_stopped_once_its_last_needed_event_is_in(self):
        script = ("import json, time\n"
                  "for k in range(100):\n"
                  "    print(json.dumps({'event': 'position', 'index': k, 'side': 'white', 'in_check': False,\n"
                  "                      'legal_actions': 1, 'terminal': 'none'}), flush=True)\n"
                  "    time.sleep(0.05)\n")
        observation = run_protocol("src", "r/1", [sys.executable, "-c", script], b"", RunOptions(timeout_s=60,
                                                                                                 stop_after=(3, "position")),
                                   1000)
        self.assertEqual((observation.status, observation.reason), ("timeout", "stopped"))
        self.assertGreaterEqual(len(observation.positions), 4)
        self.assertLess(len(observation.positions), 100)
        self.assertTrue(all(isinstance(at, float) for at in observation.arrival_s))


class StateEventTests(unittest.TestCase):
    """What a source says it set up (decision H30), read beside its answers."""

    def parse(self, stdout: bytes, *, returncode=0, timed_out=False, exact_fields=False):
        done = Completed(None if timed_out else returncode, stdout, b"", timed_out)
        return parse_jsonl("src", "r/1", done, 1000, 0.5, exact_fields=exact_fields)

    def test_a_state_event_is_read_into_its_position(self):
        observation = self.parse(lines(*turn(0), *turn(1, side="black"), end(accepted=1)))
        self.assertEqual(observation.status, "ok")
        first, second = observation.positions
        self.assertEqual((first.fingerprint, first.boards, first.pieces), (START_HASH, 1, 32))
        self.assertEqual((second.side, second.boards, second.pieces), ("black", 2, 64))
        self.assertNotEqual(second.fingerprint, first.fingerprint)
        self.assertEqual(len(second.placement), 32)

    def test_a_source_that_cannot_say_what_it_set_up_says_nothing(self):
        observation = self.parse(lines(position(0), position(1, side="black"), end(accepted=1)))
        self.assertEqual(observation.status, "ok")
        self.assertTrue(all(p.fingerprint is None and p.placement is None and p.boards is None for p in observation.positions))
        # Without markers there is no fingerprint, only the placement.
        unmarked = [f"0:1:w:{START.replace('*', '')}"]
        observation = self.parse(lines(state(0, unmarked, markers=False), position(0), end()))
        self.assertIsNone(observation.positions[0].fingerprint)
        self.assertIsNotNone(observation.positions[0].placement)

    def test_a_search_cut_short_keeps_what_was_set_up(self):
        # The state is printed before the search: a time-out in the search loses the answers, not the state.
        stdout = lines(*turn(0), state(1))
        for outcome in (dict(timed_out=True), dict(returncode=137)):
            observation = self.parse(stdout, **outcome)
            self.assertEqual(len(observation.positions), 2, outcome)
            cut = observation.positions[1]
            self.assertEqual((cut.side, cut.in_check, cut.legal_actions, cut.terminal), (None, None, None, None))
            self.assertEqual((cut.boards, cut.pieces), (2, 64))
            self.assertEqual(observation.actions_accepted, 1)
        self.assertEqual(self.parse(stdout, timed_out=True).status, "timeout")
        self.assertEqual(self.parse(stdout, returncode=137).reason, "crash")
        # A driver that failed during the search, and said so, is an error with the state kept.
        failed = self.parse(stdout + lines(end("error", 1, reason="exception", detail="out of memory")))
        self.assertEqual((failed.status, failed.reason, len(failed.positions)), ("error", "exception", 2))

    def test_a_run_that_says_it_finished_must_report_every_position_it_set_up(self):
        for status in ("ok", "rejected", "unsupported"):
            observation = self.parse(lines(*turn(0), state(1), end(status, 1, reason="illegal_move")))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), status)
            self.assertIn("has no position event", observation.detail)

    def test_state_events_come_one_per_position_and_first(self):
        cases = {
            "after its position": [position(0), state(0), end()],
            "for a later position": [state(1), position(0), end()],
            "twice": [state(0), state(0), position(0), end()],
            "skipping a position": [state(0), state(1), position(0), end()],
            "with no position first": [*turn(0), state(2), end(accepted=1)],
        }
        for name, events in cases.items():
            observation = self.parse(lines(*events))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), name)
            self.assertIn("invalid state event", observation.detail, name)

    def test_a_malformed_state_event_is_not_trusted(self):
        bad = state(0, ["0:1:w:8/8/8/8/8/8/8"])
        observation = self.parse(lines(bad, position(0), end()))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        self.assertIn("7 rank(s), but the board is 8 high", observation.detail)

    def test_a_drivers_own_fields_are_kept_but_never_compared(self):
        timed = {**position(0), "ms": 12.5, "limit": "mate search heap"}
        observation = self.parse(lines(state(0), timed, end()))
        self.assertEqual(observation.status, "ok")
        self.assertEqual(observation.extras, {0: {"ms": 12.5, "limit": "mate search heap"}})
        self.assertEqual(observation.positions[0].fingerprint, START_HASH)
        # A timing is a duration in milliseconds, or the driver is broken.
        for ms in (-1, "fast", True, None):
            observation = self.parse(lines({**position(0), "ms": ms}, end()))
            self.assertEqual(observation.reason, "malformed_output", ms)
        self.assertEqual(self.parse(b'{"event":"position","index":0,"side":null,"in_check":null,"legal_actions":null,'
                                    b'"terminal":null,"ms":NaN}\n' + lines(end())).reason, "malformed_output")

    def test_diagnostics_are_kept_apart_from_the_answers(self):
        explain = {"event": "explain", "index": 0, "phantom_timelines": ["0"]}
        actions = {"event": "actions", "index": 0, "count": 0, "truncated": False, "actions": []}
        observation = self.parse(lines(state(0), explain, actions, position(0), end()))
        self.assertEqual(observation.status, "ok")
        self.assertEqual(observation.diagnostics, [explain, actions])
        self.assertEqual(len(observation.positions), 1)
        self.assertEqual(self.parse(lines(state(0), position(0), end())).diagnostics, [])

    def test_exact_fields_cover_the_state_event(self):
        extra = state(0, note="hello")
        self.assertEqual(self.parse(lines(extra, position(0), end())).status, "ok")
        observation = self.parse(lines(extra, position(0), end()), exact_fields=True)
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))

    def test_a_broken_event_is_not_hidden_by_a_time_out(self):
        # A label leak, a "*" after a bishop, a wrong rank count: the run was
        # broken before it was killed, and says so - it is not an ordinary time-out.
        broken = lines(*turn(0), state(1, ["0:1:w:BAD"]), position(1))
        for outcome in (dict(timed_out=True), dict()):
            observation = self.parse(broken, **outcome)
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), outcome)
            self.assertIn("1 rank(s), but the board is 8 high", observation.detail)
            self.assertEqual(len(observation.positions), 1)
        self.assertIn("killed for time", self.parse(broken, timed_out=True).detail)
        bad_position = lines(*turn(0), state(1), {**position(1), "side": "purple"})
        self.assertEqual(self.parse(bad_position, timed_out=True).reason, "malformed_output")

    def test_only_the_last_line_of_a_killed_run_may_be_cut_off(self):
        cut = lines(*turn(0), state(1)) + b'{"event":"position","ind'
        self.assertEqual(self.parse(cut, timed_out=True).status, "timeout")
        # A line nobody can read, with more output after it, is not what a kill leaves behind.
        garbled = lines(*turn(0)) + b"Board destroyed: 8/8/8\n" + lines(state(1))
        observation = self.parse(garbled, timed_out=True)
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        self.assertIn("Board destroyed", observation.detail)

    def test_a_source_that_says_what_it_set_up_says_it_at_every_position(self):
        cases = {
            "a position without its state": ([*turn(0), position(1, side="black"), *turn(2), end(accepted=2)],
                                             "no state event for position 1"),
            "states only from position 1": ([position(0), *turn(1, side="black"), end(accepted=1)],
                                            "first state event is for position 1"),
            "markers that change": ([*turn(0), state(1, [f"0:1:w:{START.replace('*', '')}",
                                                         f"0:1:b:{AFTER_E3.replace('*', '')}"], markers=False),
                                     position(1, side="black"), end(accepted=1)],
                                    "markers is false here and was true before"),
        }
        for name, (events, words) in cases.items():
            observation = self.parse(lines(*events))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), name)
            self.assertIn(words, observation.detail, name)

    def test_an_oracles_own_hash_is_held_to_the_fingerprint(self):
        # The harness's canonical form against one it did not write.
        agreed = self.parse(lines(state(0, hash=START_HASH), position(0), end()))
        self.assertEqual((agreed.status, agreed.positions[0].fingerprint), ("ok", START_HASH))
        for name, event in {
            "a different hash": state(0, hash="0" * 32),
            "a hash with no markers": state(0, [f"0:1:w:{START.replace('*', '')}"], markers=False, hash=START_HASH),
            "not a string": state(0, hash=7),
        }.items():
            observation = self.parse(lines(event, position(0), end()))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), name)
        # The engine may add nothing to its state event, a hash included.
        refused = self.parse(lines(state(0, hash=START_HASH), position(0), end()), exact_fields=True)
        self.assertEqual(refused.reason, "malformed_output")

    def test_numbers_too_long_to_convert_are_malformed_output_not_a_crash(self):
        # Python refuses to convert a string of more than 4300 digits; nothing a
        # source prints may take the harness down with it.
        huge = "1" * 5000
        for event in (state(0, [f"0:1:w:{huge}/8/8/8/8/8/8/8"]), state(0, [f"0:{huge}:w:{START}"]),
                      state(0, [f"{huge}:1:w:{START}"]), state(0, size=f"{huge}x8")):
            observation = self.parse(lines(event, position(0), end()))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        too_many = b'{"event":"position","index":0,"side":"white","in_check":false,"legal_actions":' + \
            huge.encode() + b',"terminal":"none"}\n'
        observation = self.parse(too_many + lines(end()))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))


class DiagnosticArgumentTests(unittest.TestCase):
    """The hook through which `observe` asks a driver for its diagnostic events."""

    class Driven(Source):
        diagnostics = ("explain", "actions")

    def test_the_common_flags_for_what_a_source_can_print(self):
        source = self.Driven()
        self.assertEqual(source.diagnostic_args(RunOptions()), [])
        self.assertEqual(source.diagnostic_args(RunOptions(explain=True)), ["--explain"])
        self.assertEqual(source.diagnostic_args(RunOptions(explain=True, actions_at=0)), ["--explain", "--actions", "0"])

    def test_a_source_that_prints_none_is_asked_for_none(self):
        self.assertEqual(Source().diagnostic_args(RunOptions(explain=True, actions_at=4)), [])
        self.assertEqual(EngineSource(["engine"]).diagnostic_args(RunOptions(explain=True, actions_at=4)), [])


class SignatureTests(unittest.TestCase):
    def test_code_hash_follows_content_and_name_but_not_line_endings(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.py", Path(folder) / "b.py"
            first.write_bytes(b"x = 1\ny = 2\n")
            second.write_bytes(b"z = 3\n")
            original = code_sha256(first, second)
            first.write_bytes(b"x = 1\r\ny = 2\r\n")
            self.assertEqual(code_sha256(first, second), original)
            first.write_bytes(b"x = 1\ny = 3\n")
            self.assertNotEqual(code_sha256(first, second), original)
            self.assertNotEqual(code_sha256(second, first), code_sha256(first, second))


class EngineCommandTests(unittest.TestCase):
    """The engine seam, exercised end to end with a stand-in engine."""

    RECORD = b'[Board "Standard"]\n1. e3 / e6\n'

    def run_engine(self, body: str, cap: int = 7, mode: str = "full") -> Observation:
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "engine.py"
            script.write_text("import json, sys\n" + body, encoding="utf-8")
            engine = EngineSource([sys.executable, str(script)], mode)
            self.assertIsNone(engine.available())
            return engine.observe("r/1", self.RECORD, RunOptions(cap=cap, timeout_s=30))

    def emit(self, *events) -> str:
        return "".join(f"print(json.dumps({event!r}))\n" for event in events)

    def test_the_engine_receives_the_record_and_the_cap(self):
        observation = self.run_engine(
            "record = sys.stdin.read()\n"
            "cap = int(sys.argv[sys.argv.index('--cap') + 1])\n"
            f"print(json.dumps({state(0)!r}))\n"
            "print(json.dumps({'event': 'position', 'index': 0, 'side': 'white', 'in_check': False,\n"
            "                  'legal_actions': min(cap, len(record)), 'terminal': 'none'}))\n"
            "print(json.dumps({'event': 'end', 'status': 'ok', 'actions_accepted': 0}))\n"
        )
        self.assertEqual(observation.status, "ok")
        # The record is longer than the cap: the count the engine reports is the cap it was given.
        self.assertGreater(len(self.RECORD), 7)
        self.assertEqual(observation.positions[0].legal_actions, 7)
        self.assertEqual(observation.cap, 7)
        self.assertEqual(observation.positions[0].fingerprint, START_HASH)

    def test_an_engine_that_dies_is_an_error_not_a_verdict(self):
        observation = self.run_engine("sys.exit(3)\n")
        self.assertEqual((observation.status, observation.reason), ("error", "crash"))

    def test_no_command_means_no_engine(self):
        self.assertIn("no engine adapter", EngineSource([]).available())

    def test_the_engine_may_not_leave_a_question_unanswered(self):
        # An oracle may say nothing. The engine is what is being certified: a
        # null from it would be a verdict nobody compared.
        for field in ("side", "in_check", "legal_actions", "terminal"):
            observation = self.run_engine(self.emit(state(0), {**position(0), field: None}, end()))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), field)
            self.assertIn(field, observation.detail)

    def test_the_engine_says_what_it_set_up_before_every_position(self):
        # Without it nothing shows that the engine and an oracle are answering
        # about the same position (decision H30).
        fine = self.run_engine(self.emit(*turn(0), *turn(1, side="black"), end("ok", 1)))
        self.assertEqual(fine.status, "ok")
        self.assertEqual([p.boards for p in fine.positions], [1, 2])
        for name, (events, words) in {
            "no state at all": ([position(0), end()], "a state event with markers true"),
            # Any source that says what it set up says it at every position: the protocol itself refuses this.
            "no state for the second position": ([*turn(0), position(1, side="black"), end("ok", 1)],
                                                 "no state event for position 1"),
            "no unmoved markers": ([state(0, [f"0:1:w:{START.replace('*', '')}"], markers=False), position(0), end()],
                                   "a state event with markers true"),
        }.items():
            observation = self.run_engine(self.emit(*events))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), name)
            self.assertIn(words, observation.detail, name)

    def test_the_engines_state_event_has_exactly_the_protocols_fields(self):
        for event in (state(0, ms=0.1), {key: value for key, value in state(0).items() if key != "size"}):
            observation = self.run_engine(self.emit(event, position(0), end()))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), event)
            self.assertIn("invalid state event", observation.detail)

    def test_the_engine_must_say_which_kind_of_end_it_is(self):
        observation = self.run_engine(self.emit(state(0), position(0, legal=0, terminal="no_action"), end()))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))

    def test_the_engine_refuses_an_action_as_illegal_or_as_unreadable_and_nothing_else(self):
        # An oracle's reasons for not matching the move text are not the engine's to give.
        for reason in ("no_matching_move", "ambiguous_move", "notation_mismatch", "unrecognised"):
            observation = self.run_engine(self.emit(*turn(0), end("rejected", reason=reason)))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), reason)
            self.assertIn("refuses an action as one of", observation.detail)
        # Its own parser could not read the move text: "unreadable", in full mode as in parse mode.
        for reason in (*HARD_REJECTIONS, "unreadable"):
            observation = self.run_engine(self.emit(*turn(0), end("rejected", reason=reason)))
            self.assertEqual((observation.status, observation.reason), ("rejected", reason))
        self.assertEqual(set(SOFT_REJECTIONS) & set(REJECTIONS), {"unreadable"})

    def test_the_engine_says_exactly_what_the_protocol_asks(self):
        # A field of its own - a timing, say - is refused: an oracle driver's
        # output handed on as the engine's carries fields like that.
        observation = self.run_engine(self.emit(state(0), {**position(0), "ms": 0.1}, end()))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        missing = {key: value for key, value in position(0).items() if key != "terminal"}
        self.assertEqual(self.run_engine(self.emit(state(0), missing, end())).status, "error")

    def test_an_engine_that_loses_count_is_an_error(self):
        # Two positions, and a claim of five accepted actions.
        observation = self.run_engine(self.emit(*turn(0), *turn(1, side="black"), end("ok", 5)))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        self.assertIn("claims 5 accepted action(s)", observation.detail)

    def test_unsupported_is_said_before_the_first_position(self):
        self.assertEqual(self.run_engine(self.emit(end("unsupported", reason="variant"))).status, "unsupported")
        late = self.run_engine(self.emit(*turn(0), end("unsupported", reason="variant")))
        self.assertEqual((late.status, late.reason), ("error", "malformed_output"))

    def test_contract_violation_reads_a_report(self):
        known = dict(fingerprint="a" * 32, placement="b" * 32, boards=1, pieces=32)
        full = [Position(0, "white", False, 20, "none", **known), Position(1, "black", True, 0, "checkmate", **known)]
        fine = Observation(source="engine", record="r/1", status="ok", positions=full, actions_accepted=1, cap=7)
        self.assertIsNone(contract_violation(fine))
        unmarked = Observation(source="engine", record="r/1", status="ok", actions_accepted=0, cap=7,
                               positions=[Position(0, "white", False, 20, "none", placement="b" * 32, boards=1, pieces=32)])
        self.assertIn("markers true", contract_violation(unmarked))
        failed = Observation(source="engine", record="r/1", status="error", reason="crash", actions_accepted=0)
        self.assertIsNone(contract_violation(failed))  # already an error; nothing more to say

    # --- parse mode -------------------------------------------------------------------

    def parsed(self, index, side="white"):
        """What an engine in parse mode prints for one position: its state, and a position event with no answers."""
        return [state(index), {"event": "position", "index": index, "side": side, "in_check": None,
                               "legal_actions": None, "terminal": None}]

    def test_parse_mode_asks_for_it_after_the_cap(self):
        observation = self.run_engine(
            "assert sys.argv[1:] == ['--cap', '7', '--parse-only'], sys.argv\n"
            + self.emit(*self.parsed(0), *self.parsed(1, "black"), end("ok", 1)), mode="parse")
        self.assertEqual((observation.status, observation.actions_accepted), ("ok", 1))
        self.assertEqual([p.side for p in observation.positions], ["white", "black"])
        self.assertTrue(all(p.in_check is None and p.legal_actions is None and p.terminal is None
                            for p in observation.positions))
        self.assertEqual(observation.positions[0].fingerprint, START_HASH)
        # Full mode asks no such thing, and holds the engine to every answer.
        self.assertEqual(self.run_engine(self.emit(*self.parsed(0), end())).reason, "malformed_output")
        self.assertEqual(EngineSource(["e"], "parse").argv(RunOptions(cap=3)), ["e", "--cap", "3", "--parse-only"])
        with self.assertRaises(ValueError):
            EngineSource(["e"], "half")

    def test_in_parse_mode_the_engine_answers_no_rules_question(self):
        # A guess at check, a count or a verdict is refused: nothing would compare it.
        for name, value in (("in_check", False), ("legal_actions", 20), ("terminal", "none")):
            event = {**self.parsed(0)[1], name: value}
            observation = self.run_engine(self.emit(state(0), event, end()), mode="parse")
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), name)
            self.assertIn(name, observation.detail)
        # Its side to move and what it set up it must still give.
        silent = self.run_engine(self.emit(state(0), {**self.parsed(0)[1], "side": None}, end()), mode="parse")
        self.assertIn("`side` is null", silent.detail)
        stateless = self.run_engine(self.emit(self.parsed(0)[1], end()), mode="parse")
        self.assertIn("markers true", stateless.detail)

    def test_in_parse_mode_the_engine_refuses_an_action_as_unreadable_illegal_or_unsubmittable(self):
        for reason in REJECTIONS:
            observation = self.run_engine(self.emit(*self.parsed(0), end("rejected", reason=reason)), mode="parse")
            self.assertEqual((observation.status, observation.reason), ("rejected", reason))
        for reason in ("no_matching_move", "ambiguous_move"):
            observation = self.run_engine(self.emit(*self.parsed(0), end("rejected", reason=reason)), mode="parse")
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), reason)
        # And unsupported is still said before the first position.
        late = self.run_engine(self.emit(*self.parsed(0), end("unsupported", reason="variant")), mode="parse")
        self.assertEqual(late.reason, "malformed_output")
        self.assertEqual(self.run_engine(self.emit(end("unsupported", reason="variant")), mode="parse").status,
                         "unsupported")

    def test_the_engine_is_never_stopped_early(self):
        # A gate run may stop an oracle once its required positions are in; never the engine.
        observation = self.run_engine(self.emit(*turn(0), *turn(1, side="black"), end("ok", 1)))
        self.assertEqual(observation.status, "ok")
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "engine.py"
            script.write_text("import json\n" + self.emit(*turn(0), *turn(1, side="black"), end("ok", 1)), encoding="utf-8")
            engine = EngineSource([sys.executable, str(script)])
            stopped = engine.observe("r/1", self.RECORD, RunOptions(cap=7, timeout_s=30, stop_after=(0, "position")))
        self.assertEqual((stopped.status, len(stopped.positions)), ("ok", 2))

    def test_the_engine_may_refuse_a_records_set_up_as_unreadable_in_either_mode(self):
        # Its parser cannot read the record's headers or board strings: there is no position 0 to report.
        refusal = end("rejected", reason="unreadable")
        for mode in ("parse", "full"):
            observation = self.run_engine(self.emit(refusal), mode=mode)
            self.assertEqual((observation.status, observation.reason, observation.actions_accepted, observation.positions),
                             ("rejected", "unreadable", 0, []), mode)
            self.assertTrue(observations.refused_setup(observation), mode)
            self.assertEqual(observations.validate(observation), [], mode)
            # Only as unreadable: a set-up is no action that can be illegal.
            for reason in HARD_REJECTIONS:
                refused = self.run_engine(self.emit(end("rejected", reason=reason)), mode=mode)
                self.assertEqual((refused.status, refused.reason), ("error", "malformed_output"), (mode, reason))
        # Never from an oracle's driver.
        done = Completed(0, lines(refusal), b"", False)
        self.assertEqual(parse_jsonl("ftxi", "r/1", done, 1000, 0.1).reason, "malformed_output")

    def test_in_parse_mode_the_engine_prints_exactly_the_protocols_fields(self):
        # An oracle driver's own timing, passed off as the engine's, is refused - in parse mode as in full mode.
        timed = {**self.parsed(0)[1], "ms": 3}
        observation = self.run_engine(self.emit(state(0), timed, end()), mode="parse")
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        # And a field left out is not a null: the six fields, every one of them.
        short = {key: value for key, value in self.parsed(0)[1].items() if key != "terminal"}
        observation = self.run_engine(self.emit(state(0), short, end()), mode="parse")
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        extra_state = state(0, hash=START_HASH)
        observation = self.run_engine(self.emit(extra_state, self.parsed(0)[1], end()), mode="parse")
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))


class CwmttOutputTests(unittest.TestCase):
    def parse(self, stdout: str, stderr: str = "", *, returncode=0, timed_out=False):
        done = Completed(None if timed_out else returncode, stdout.encode(), stderr.encode(), timed_out)
        return parse_count_output("cwmtt", "r/1", done, 1000, 0.1)

    def summary(self, stdout: str, stderr: str = ""):
        observation = self.parse(stdout, stderr)
        return observation.status, observation.reason, len(observation.positions), observation.actions_accepted

    def test_counts_become_positions(self):
        observation = self.parse("20 20 32 1000 0 \n")
        self.assertEqual(observation.status, "ok")
        self.assertEqual([p.legal_actions for p in observation.positions], [20, 20, 32, 1000, 0])
        self.assertEqual([p.terminal for p in observation.positions], ["none"] * 4 + ["no_action"])
        self.assertTrue(all(p.in_check is None and p.side is None for p in observation.positions))
        self.assertEqual(observation.actions_accepted, 4)

    def test_a_replay_failure_is_a_rejection(self):
        failure = "s:(0,2,Nothing,Nothing) d:(Nothing,Nothing,Just 4,Just 2) No matching moves\n"
        self.assertEqual(self.summary("20 20 31 \n", failure), ("rejected", "no_matching_move", 3, 2))

    def test_unread_moves_mark_the_record_partly_read(self):
        # The parser stopped between two actions: what it read, it replayed faithfully.
        self.assertEqual(self.summary('Unparsed: "? / (0T3)Nc6!\\n4. (0T4)d4"\n20 20 24 \n'), ("unsupported", "notation", 3, 2))
        self.assertEqual(self.summary('Unparsed: "5. (0T5)O-O / (0T5)Be7\\n"\n20 20 24 \n'), ("unsupported", "notation", 3, 2))
        self.assertEqual(self.summary('Unparsed: "(3b. a6 {an aside})\\n3b. Nf6\\n"\n20 20 24 \n'),
                         ("unsupported", "notation", 3, 2))

    def test_a_failure_after_a_clean_stop_is_still_a_rejection(self):
        # Whole actions were read and one of them was refused: that is cwmtt's verdict on it.
        self.assertEqual(self.summary('Unparsed: "? / (0T3)Nc6!\\n"\n20 20 \n', "In check\n"), ("rejected", "cannot_submit", 2, 1))

    def test_a_failure_on_half_an_action_is_not_a_verdict(self):
        # The parser stopped at the "?" inside a two-move action, replayed the
        # first move alone, and complained that the second timeline had not moved.
        stdout = 'Unparsed: "? (1T4)a5?\\n5. (0T5)Nc3"\n20 20 24 30 \n'
        observation = self.parse(stdout, "Did not move in all active present timelines\n")
        self.assertEqual((observation.status, observation.reason), ("unsupported", "notation"))
        # Every position it reported precedes the fragment, so all of them stand.
        self.assertEqual((len(observation.positions), observation.actions_accepted), (4, 3))
        self.assertIn("Did not move", observation.detail)

    def test_a_position_reached_by_half_an_action_is_dropped(self):
        # No complaint this time: the fragment was playable on its own. The
        # position it led to belongs to no record.
        stdout = 'Unparsed: "? (1T4)a5?\\n5. (0T5)Nc3"\n20 20 24 30 \n'
        self.assertEqual(self.summary(stdout), ("unsupported", "notation", 3, 2))
        self.assertEqual(self.summary('Unparsed: "=Q / Kd7\\n"\n20 20 \n'), ("unsupported", "notation", 1, 0))
        # Nothing to drop when not a single action was read.
        self.assertEqual(self.summary('Unparsed: "[r*nbqk*bnr*/8:0:1:w]\\n1. e3"\n20 \n'), ("unsupported", "notation", 1, 0))

    def test_an_unread_result_token_is_not_unread_moves(self):
        self.assertEqual(self.parse('Unparsed: "1-0\\n"\n20 20 0 \n').status, "ok")
        self.assertEqual(self.parse('Unparsed: "1-0 {White wins; (0T6)Qh5 attacks}\\n"\n20 0 \n').status, "ok")
        self.assertEqual(self.parse('Unparsed: "?!\\n"\n20 20 \n').status, "ok")  # a glyph on the last move

    def test_a_killed_run_never_yields_half_a_number(self):
        observation = self.parse("20 20 3", timed_out=True)
        self.assertEqual(observation.status, "timeout")
        self.assertEqual([p.legal_actions for p in observation.positions], [20, 20])
        self.assertEqual(observation.actions_accepted, 1)

    def test_each_count_is_timed_as_it_arrives(self):
        # The CLI streams its counts on one line; the space after a count says it is whole.
        data = b'Unparsed: "1-0\\n"\n20 20 32 \n'
        start = data.index(b"20 20")
        arrivals = ((start + 3, 1.0), (start + 6, 2.5), (len(data), 4.0))  # "20 " / "20 " / "32 \n"
        done = Completed(0, data, b"", False, arrivals=arrivals, elapsed_s=4.2)
        self.assertEqual(count_arrivals(done), [1.0, 2.5, 4.0])
        observation = parse_count_output("cwmtt", "r/1", done, 1000, 4.2)
        self.assertEqual((observation.arrival_s, observation.ended_s), ([1.0, 2.5, 4.0], 4.2))
        # Digits inside the Unparsed: line are not counts.
        self.assertEqual(count_arrivals(Completed(0, b'Unparsed: "12 34 "\n7 \n', b"", False)), [None])

    def test_a_count_run_may_be_stopped_once_it_has_the_counts_needed(self):
        watch = watch_counts(3)
        self.assertFalse(watch(b'Unparsed: "1 2 3 4 5 '))  # not counts
        self.assertFalse(watch(b'"\n20 2'))
        self.assertFalse(watch(b"0 3"))  # "20 20 " and an unfinished "3"
        self.assertTrue(watch(b"2 1"))  # "20 32 ..." : three whole counts
        stopped = self.parse("20 20 32 ", timed_out=False)
        self.assertEqual(stopped.status, "ok")
        done = Completed(None, b"20 20 32 ", b"", False, stopped=True)
        observation = parse_count_output("cwmtt", "r/1", done, 1000, 1.0)
        self.assertEqual((observation.status, observation.reason, len(observation.positions)), ("timeout", "stopped", 3))

    def test_a_crash_is_an_error(self):
        self.assertEqual(self.parse("", "cwmtt: Irrefutable pattern failed", returncode=1).status, "error")
        self.assertEqual(self.parse("").status, "error")  # exit 0 but no counts at all

    def test_warning_classification(self):
        self.assertEqual(classify_warning("s:(0,2) d:(1,2) No matching moves"), "no_matching_move")
        self.assertEqual(classify_warning("Ambigous: Nbd2 or Nfd2..."), "ambiguous_move")
        self.assertEqual(classify_warning("Move number expected 3 found 4"), "notation_mismatch")
        self.assertEqual(classify_warning("Did not move in all active present timelines"), "cannot_submit")
        self.assertEqual(classify_warning("In check"), "cannot_submit")
        self.assertEqual(classify_warning("more than 1 move from timeline 0"), "illegal_move")
        self.assertEqual(classify_warning("more than 1 hop to timeline 2"), "illegal_move")

    def test_only_listed_messages_are_verdicts_on_the_rules(self):
        # A message this adapter has never seen is not presented as "cwmtt calls
        # the action illegal": it may be notation.
        for message in ("Not a playable timeline: 3", "No hop to timeline 1", "Unable to resolve branches: conflicts among x",
                        "something a later cwmtt says"):
            self.assertIn(classify_warning(message), SOFT_REJECTIONS, message)

    def test_what_the_parser_left_unread(self):
        cases = {
            '"1-0\\n"': "none",
            '"\\n{a comment}\\n"': "none",
            '"1/2-1/2"': "none",
            '"/ \\n"': "none",  # a record that stops after White's move
            '"??\\n"': "none",
            '"/ (0T5)Be7\\n"': "boundary",
            '"?? / (0T5)Be7\\n"': "boundary",
            '"5. (0T5)O-O / (0T5)Be7\\n"': "boundary",
            '"12w. (0T12)O-O"': "boundary",
            '"(3b. Nf6)\\n3b. d5"': "boundary",
            '"? (1T4)a5?\\n5. e4"': "mid_action",
            '"=Q / Kd7"': "mid_action",
            '"(0T4)O-O-O / Kd7"': "mid_action",
            '"1-0 \\8212 then more"': "mid_action",  # an escape it does not decode
            "garbage": "mid_action",
        }
        for text, expected in cases.items():
            self.assertEqual(remainder_kind("Unparsed: " + text), expected, text)

    def test_escapes_are_read_as_haskell_writes_them(self):
        # A long dash in a closing comment arrives as \8212; it is still only a comment.
        self.assertEqual(remainder_kind('Unparsed: "1-0 {White resigns \\8212 lost on time}\\n"'), "none")
        self.assertEqual(remainder_kind('Unparsed: "\\SOH\\DEL\\&\\x41 / e5"'), "mid_action")
        self.assertEqual(remainder_kind('Unparsed: "?\\t\\n4. e4"'), "boundary")
        self.assertEqual(remainder_kind('Unparsed: "\\z?"'), "mid_action")  # no such escape: cannot tell

    def test_trailer_recognition_is_conservative(self):
        self.assertTrue(unparsed_is_only_trailer('Unparsed: "1-0\\n"'))
        self.assertFalse(unparsed_is_only_trailer('Unparsed: "/ (0T5)Be7\\n"'))
        self.assertFalse(unparsed_is_only_trailer("Unparsed: garbage"))


class CwmttHeaderTests(unittest.TestCase):
    LAYOUTS = ["Standard", "Standard - Turn Zero"]

    def test_tags_are_read_the_way_cwmtt_reads_them(self):
        text = ('[Mode "5D"]\n; a comment\n[ Board   "Standard - Turn Zero" ]{another}\n'
                '[White "Mage O\\" Chicken"]\n\n1. e3 / e6\n')
        self.assertEqual(cwmtt_tags(text), {"Mode": "5D", "Board": "Standard - Turn Zero", "White": 'Mage O" Chicken'})
        self.assertEqual(cwmtt_tags('[Board "Standard"][Board "Standard - Turn Zero"]'), {"Board": "Standard - Turn Zero"})

    def test_reading_stops_at_the_first_thing_that_is_not_a_tag(self):
        # A 5DFEN board, a key cwmtt's grammar does not allow, a missing quote.
        self.assertEqual(cwmtt_tags('[Size "5x5"]\n[3k1/5/5/1R3/K4:0:1:w]\n[Board "Standard - Turn Zero"]'), {"Size": "5x5"})
        self.assertEqual(cwmtt_tags('[Time-Control "5+3"]\n[Board "Standard - Turn Zero"]'), {})
        self.assertEqual(cwmtt_tags('1. e3 / e6\n[Board "Standard - Turn Zero"]'), {})

    def refusal(self, record: str):
        refused = would_misread(record.encode(), self.LAYOUTS)
        return None if refused is None else refused[0]

    def test_a_record_cwmtt_reads_as_declared_is_asked(self):
        self.assertIsNone(self.refusal('[Board "Standard - Turn Zero"]\n1. e3 / e6\n'))
        self.assertIsNone(self.refusal('[Mode "5D"]\n[Board "Standard"]\n1. e3 / e6\n'))
        self.assertIsNone(self.refusal("1. e3 / e6\n"))  # no header: Standard, and cwmtt plays Standard

    def test_a_board_cwmtt_does_not_know_is_refused(self):
        self.assertEqual(self.refusal('[Board "Misc - Small"]\n1. e3 / e6\n'), "variant")

    def test_a_board_header_cwmtt_would_not_reach_is_refused(self):
        # cwmtt knows the variant, but each of these keeps its parser from the
        # tag - so it would replay Standard and answer about that.
        for record in (
            '[board "Standard - Turn Zero"]\n1. e3 / e6\n',  # its key is case-sensitive
            '[Time-Control "5+3"]\n[Board "Standard - Turn Zero"]\n1. e3 / e6\n',
            '1. e3 / e6\n[Board "Standard - Turn Zero"]\n',
        ):
            self.assertEqual(self.refusal(record), "variant", record)
        # The same slips under a Standard header change nothing: Standard is what it plays anyway.
        self.assertIsNone(self.refusal('[board "Standard"]\n1. e3 / e6\n'))

    def test_a_start_position_or_promotion_rule_cwmtt_ignores_is_refused(self):
        self.assertEqual(self.refusal('[Board "Standard"]\n[r*nbqk*bnr*/8/8/8/8/8/8/R*NBQK*BNR*:0:1:w]\n1. e3 / e6\n'), "fen")
        self.assertEqual(self.refusal('[Board "Standard"]\n[Promotions "Q,N"]\n1. e3 / e6\n'), "promotions")
        self.assertEqual(self.refusal('[Board "Standard"]\n[Timeline "even"]\n1. e3 / e6\n'), "timeline")

    def test_a_variant_named_only_by_a_variant_header_is_refused(self):
        # ftxi reads [Variant] when there is no [Board]; cwmtt reads only [Board] and would play Standard.
        self.assertEqual(self.refusal('[Variant "Standard - Turn Zero"]\n1. e3 / e6\n'), "variant")


class NotationGuardTests(unittest.TestCase):
    """Records an oracle's own reader would take for another game."""

    def test_chess_in_5d_is_not_asked_about_notation_it_misreads(self):
        refused = {
            "variation": b"1. e4 / e5\n(2. d4 / d5)\n2. Nf3 / Nc6\n",
            "coloured turn": b"1w. d4 / d5\n2. e4 / e5\n",
            "black turn": b"1. d4 / d5\n2. e4\n2b. e5\n",
            "timeline": b'[Board "Standard"]\n[Timeline "even"]\n1. e3 / e6\n',
        }
        for name, record in refused.items():
            self.assertIsNotNone(chessin5d_adapter.would_misread(record), name)
        # Superphysical coordinates and comments are not variations.
        fine = b"1. (0T1)e3 / (0T1)e6\n2. (0T2)Qh5 {see (2. e4)} / (0T2)Nc6\n"
        self.assertIsNone(chessin5d_adapter.would_misread(fine))

    def test_ftxi_is_not_asked_about_a_record_that_ends_in_a_variation(self):
        # Its replay takes the last continuation written, which there is the side line.
        self.assertTrue(ends_in_a_variation(b"1. e4 / e5\n2. Nf3 / Nc6\n(2. d4 / d5)\n"))
        self.assertTrue(ends_in_a_variation(b"1. e4 / e5\n(2. d4 / (0T2)d5 1-0 {end})\n1-0\n"))
        self.assertTrue(ends_in_a_variation(b"1. e4 / e5\n(2. d4 / d5\n"))  # never closed
        # A variation with the main line after it, as ftxi writes them, is read right.
        self.assertFalse(ends_in_a_variation(b"1. e4 / e5\n(2. d4 / d5)\n2. Nf3 / Nc6\n"))
        self.assertFalse(ends_in_a_variation(b"1. (0T1)e3 / (0T1)e6 (>L1)\n"))  # an annotation, not a variation
        self.assertFalse(ends_in_a_variation(b"1. e4 / e5 {(2. d4) was also possible}\n"))


class MateProofTests(unittest.TestCase):
    """A mate verdict is a proof - an exhaustive search that terminated; an exhausted budget is unknown (H37)."""

    @staticmethod
    def chessin5d(*events):
        return parse_jsonl("chessin5d", "r/1", Completed(0, lines(*events), b"", False), None, 0.1)

    def test_a_search_that_ran_out_of_its_budget_gives_no_verdict(self):
        limited = {**position(0, legal=None, terminal=None), "ms": 60000.5, "limit": "mate_budget_ms"}
        observation = self.chessin5d(state(0), limited, end())
        self.assertIsNone(chessin5d_adapter.verdict_beyond_its_limit(observation))
        self.assertIsNone(observation.positions[0].terminal)
        source = chessin5d_adapter.ChessIn5dSource(lockfile.load())
        self.assertEqual(source.limits(observation, 0, RunOptions(mate_budget_ms=60000)), {"terminal": "mate_budget_ms=60000"})

    def test_a_verdict_beside_the_limit_its_search_ran_into_is_no_proof(self):
        for terminal in ("checkmate", "stalemate", "none"):
            claimed = {**position(0, legal=None, terminal=terminal), "ms": 60000.5, "limit": "mate_budget_ms"}
            problem = chessin5d_adapter.verdict_beyond_its_limit(self.chessin5d(state(0), claimed, end()))
            self.assertIsNotNone(problem, terminal)
            self.assertIn("an exhausted budget is unknown, never a verdict (H37)", problem)
        # A verdict from a search that finished is one.
        finished = {**position(0, legal=None, terminal="checkmate"), "ms": 12.0}
        self.assertIsNone(chessin5d_adapter.verdict_beyond_its_limit(self.chessin5d(state(0), finished, end())))


if __name__ == "__main__":
    unittest.main()
