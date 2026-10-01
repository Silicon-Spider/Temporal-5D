import json
import sys
import tempfile
import unittest
from pathlib import Path

from oracle_harness.adapters import EngineSource, RunOptions
from oracle_harness.adapters import chessin5d as chessin5d_adapter
from oracle_harness.adapters.base import code_sha256, parse_jsonl
from oracle_harness.adapters.cwmtt import (
    classify_warning,
    cwmtt_tags,
    parse_count_output,
    remainder_kind,
    unparsed_is_only_trailer,
    would_misread,
)
from oracle_harness.adapters.engine import contract_violation
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

    def run_engine(self, body: str, cap: int = 7) -> Observation:
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "engine.py"
            script.write_text("import json, sys\n" + body, encoding="utf-8")
            engine = EngineSource([sys.executable, str(script)])
            self.assertIsNone(engine.available())
            return engine.observe("r/1", self.RECORD, RunOptions(cap=cap, timeout_s=30))

    def emit(self, *events) -> str:
        return "".join(f"print(json.dumps({event!r}))\n" for event in events)

    def test_the_engine_receives_the_record_and_the_cap(self):
        observation = self.run_engine(
            "record = sys.stdin.read()\n"
            "cap = int(sys.argv[sys.argv.index('--cap') + 1])\n"
            "print(json.dumps({'event': 'position', 'index': 0, 'side': 'white', 'in_check': False,\n"
            "                  'legal_actions': min(cap, len(record)), 'terminal': 'none'}))\n"
            "print(json.dumps({'event': 'end', 'status': 'ok', 'actions_accepted': 0}))\n"
        )
        self.assertEqual(observation.status, "ok")
        # The record is longer than the cap: the count the engine reports is the cap it was given.
        self.assertGreater(len(self.RECORD), 7)
        self.assertEqual(observation.positions[0].legal_actions, 7)
        self.assertEqual(observation.cap, 7)

    def test_an_engine_that_dies_is_an_error_not_a_verdict(self):
        observation = self.run_engine("sys.exit(3)\n")
        self.assertEqual((observation.status, observation.reason), ("error", "crash"))

    def test_no_command_means_no_engine(self):
        self.assertIn("no engine adapter", EngineSource([]).available())

    def test_the_engine_may_not_leave_a_question_unanswered(self):
        # An oracle may say nothing. The engine is what is being certified: a
        # null from it would be a verdict nobody compared.
        for field in ("side", "in_check", "legal_actions", "terminal"):
            observation = self.run_engine(self.emit({**position(0), field: None}, end()))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), field)
            self.assertIn(field, observation.detail)

    def test_the_engine_must_say_which_kind_of_end_it_is(self):
        observation = self.run_engine(self.emit(position(0, legal=0, terminal="no_action"), end()))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))

    def test_the_engine_refuses_an_action_only_as_illegal(self):
        for reason in SOFT_REJECTIONS:
            observation = self.run_engine(self.emit(position(0), end("rejected", reason=reason)))
            self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"), reason)
        for reason in HARD_REJECTIONS:
            observation = self.run_engine(self.emit(position(0), end("rejected", reason=reason)))
            self.assertEqual((observation.status, observation.reason), ("rejected", reason))

    def test_the_engine_says_exactly_what_the_protocol_asks(self):
        # A field of its own - a timing, say - is refused: an oracle driver's
        # output handed on as the engine's carries fields like that.
        observation = self.run_engine(self.emit({**position(0), "ms": 0.1}, end()))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        missing = {key: value for key, value in position(0).items() if key != "terminal"}
        self.assertEqual(self.run_engine(self.emit(missing, end())).status, "error")

    def test_an_engine_that_loses_count_is_an_error(self):
        # Two positions, and a claim of five accepted actions.
        observation = self.run_engine(self.emit(position(0), position(1, side="black"), end("ok", 5)))
        self.assertEqual((observation.status, observation.reason), ("error", "malformed_output"))
        self.assertIn("claims 5 accepted action(s)", observation.detail)

    def test_unsupported_is_said_before_the_first_position(self):
        self.assertEqual(self.run_engine(self.emit(end("unsupported", reason="variant"))).status, "unsupported")
        late = self.run_engine(self.emit(position(0), end("unsupported", reason="variant")))
        self.assertEqual((late.status, late.reason), ("error", "malformed_output"))

    def test_contract_violation_reads_a_report(self):
        full = [Position(0, "white", False, 20, "none"), Position(1, "black", True, 0, "checkmate")]
        fine = Observation(source="engine", record="r/1", status="ok", positions=full, actions_accepted=1, cap=7)
        self.assertIsNone(contract_violation(fine))
        failed = Observation(source="engine", record="r/1", status="error", reason="crash", actions_accepted=0)
        self.assertIsNone(contract_violation(failed))  # already an error; nothing more to say


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


if __name__ == "__main__":
    unittest.main()
