import unittest

from oracle_harness import observations
from oracle_harness.observations import Observation, Position


def obs(source, rows, status="ok", cap=1000, accepted=None, reason="", last_action=None):
    positions = [Position(index, *row) for index, row in enumerate(rows)]
    if accepted is None:
        accepted = max(len(positions) - 1, 0)
    return Observation(source=source, record="r/1", status=status, positions=positions,
                       actions_accepted=accepted, reason=reason, cap=cap, last_action=last_action)


START = ("white", False, 20, "none")
REPLY = ("black", False, 20, "none")
MATE = ("white", True, 0, "checkmate")


class CompareTests(unittest.TestCase):
    def kinds(self, a, b, **options):
        return [(item.kind, item.index, item.a_value, item.b_value, item.soft) for item in observations.compare(a, b, **options)]

    def test_identical_reports_agree(self):
        self.assertEqual(self.kinds(obs("a", [START, REPLY]), obs("b", [START, REPLY])), [])

    def test_unknown_never_disagrees(self):
        blank = (None, None, None, None)
        self.assertEqual(self.kinds(obs("a", [START, REPLY]), obs("b", [blank, blank])), [])

    def test_each_observable_is_compared(self):
        a = obs("a", [("white", True, 5, "none")])
        b = obs("b", [("black", False, 6, "none")])
        self.assertEqual(self.kinds(a, b), [
            ("side", 0, "white", "black", False),
            ("in_check", 0, "True", "False", False),
            ("legal_actions", 0, "5", "6", False),
        ])

    def test_a_capped_count_means_at_least(self):
        capped = obs("a", [("white", False, 1000, "none")])
        self.assertEqual(self.kinds(capped, obs("b", [("white", False, 1000, "none")])), [])
        # The other source counted with a higher cap and found more: compatible.
        self.assertEqual(self.kinds(capped, obs("b", [("white", False, 4321, "none")], cap=10000)), [])
        # It counted exactly, and found fewer than the cap: a contradiction.
        self.assertEqual(self.kinds(capped, obs("b", [("white", False, 999, "none")], cap=10000)),
                         [("legal_actions", 0, ">=1000", "999", False)])

    def test_terminal_kinds(self):
        def terminal(a, b):
            return self.kinds(obs("a", [(None, None, None, a)]), obs("b", [(None, None, None, b)]))

        self.assertEqual(terminal("checkmate", "checkmate"), [])
        self.assertEqual(terminal("checkmate", "no_action"), [])  # cwmtt cannot say why nothing is playable
        self.assertEqual(terminal("stalemate", "no_action"), [])
        self.assertEqual(terminal("checkmate", "stalemate"), [("terminal", 0, "checkmate", "stalemate", False)])
        self.assertEqual(terminal("none", "no_action"), [("terminal", 0, "none", "no_action", False)])

    def test_rejecting_an_action_the_other_played_is_a_legality_disagreement(self):
        full = obs("a", [START, REPLY, START])
        hard = obs("b", [START, REPLY], status="rejected", reason="illegal_move")
        self.assertEqual(self.kinds(full, hard), [("legality", 2, "accepted", "rejected:illegal_move", False)])
        self.assertEqual(self.kinds(hard, full), [("legality", 2, "rejected:illegal_move", "accepted", False)])

    def test_a_rejection_that_may_be_notation_is_soft(self):
        full = obs("a", [START, REPLY, START])
        for reason in observations.SOFT_REJECTIONS:
            soft = obs("b", [START, REPLY], status="rejected", reason=reason)
            self.assertEqual(self.kinds(full, soft), [("legality", 2, "accepted", f"rejected:{reason}", True)])

    def test_both_rejecting_the_same_action_agree(self):
        a = obs("a", [START, REPLY], status="rejected", reason="illegal_move")
        b = obs("b", [START, REPLY], status="rejected", reason="cannot_submit")
        self.assertEqual(self.kinds(a, b), [])

    def test_a_source_that_stopped_for_other_reasons_is_not_contradicted(self):
        full = obs("a", [START, REPLY, START])
        for status in ("unsupported", "timeout", "error"):
            self.assertEqual(self.kinds(full, obs("b", [START], status=status, accepted=0)), [], status)

    def test_two_complete_replays_of_different_length_disagree(self):
        self.assertEqual(self.kinds(obs("a", [START, REPLY, START]), obs("b", [START, REPLY])),
                         [("length", 2, "2 actions", "1 actions", False)])
        self.assertEqual(self.kinds(obs("b", [START, REPLY]), obs("a", [START, REPLY, START])),
                         [("length", 2, "1 actions", "2 actions", False)])

    def test_a_record_cannot_end_where_another_source_played_on(self):
        # "b" read one action and called the record finished; "a" went on to a
        # third position before it stopped for a reason of its own. A parser that
        # silently drops the rest of a record must not pass for one that agrees.
        ended = obs("b", [START, REPLY])
        for status, reason in (("timeout", "timeout"), ("unsupported", "notation")):
            went_on = obs("a", [START, REPLY, START], status=status, reason=reason)
            self.assertEqual(self.kinds(went_on, ended), [("length", 2, "at least 2 actions", "1 actions", False)], status)
        # Stopping at the same place without having seen anything more contradicts nothing.
        self.assertEqual(self.kinds(obs("a", [START, REPLY], status="timeout", reason="timeout"), ended), [])
        self.assertEqual(self.kinds(obs("a", [START], status="unsupported", reason="notation"), ended), [])

    def test_a_record_cannot_end_before_an_action_another_source_refused(self):
        # "a" found a second action in the record and refused it; "b" says the
        # record has one action. "b" never read the action it would have had to
        # judge - which is how a parser that drops an illegal move would look.
        ended = obs("b", [START, REPLY])
        refused_next = obs("a", [START, REPLY], status="rejected", reason="cannot_submit")
        self.assertEqual(self.kinds(refused_next, ended), [("length", 2, "at least 2 actions", "1 actions", False)])
        self.assertEqual(self.kinds(ended, refused_next), [("length", 2, "1 actions", "at least 2 actions", False)])
        # If what "a" could not take may have been notation, the entry is soft.
        unread = obs("a", [START, REPLY], status="rejected", reason="no_matching_move")
        self.assertEqual(self.kinds(unread, ended), [("length", 2, "at least 2 actions", "1 actions", True)])
        # A refusal further on is a third action seen.
        later = obs("a", [START, REPLY, START], status="rejected", reason="illegal_move")
        self.assertEqual(self.kinds(later, ended), [("length", 2, "at least 3 actions", "1 actions", False)])
        # Two sources that refuse the same action have read the same record.
        self.assertEqual(self.kinds(refused_next, obs("b", [START, REPLY], status="rejected", reason="illegal_move")), [])

    def test_how_a_record_was_replayed_can_be_left_out(self):
        # Two searches of one implementation replay a record the same way; that
        # is compared once, through the implementation's primary source.
        full = obs("a", [START, REPLY, START])
        refused = obs("b", [START, REPLY], status="rejected", reason="illegal_move")
        shorter = obs("b", [START, ("black", True, 20, "none")])
        self.assertEqual(self.kinds(full, refused, replay=False), [])
        self.assertEqual(self.kinds(full, shorter, replay=False), [("in_check", 1, "False", "True", False)])

    def test_records_must_match(self):
        other = obs("b", [START])
        other.record = "r/2"
        with self.assertRaises(ValueError):
            observations.compare(obs("a", [START]), other)


class ValueAtTests(unittest.TestCase):
    """value_at() is what a known-divergence entry is checked against: it must print what compare() prints."""

    def test_positions(self):
        report = obs("a", [START, ("black", True, 1000, "none"), MATE])
        self.assertEqual(observations.value_at(report, "in_check", 1), "True")
        self.assertEqual(observations.value_at(report, "legal_actions", 0), "20")
        self.assertEqual(observations.value_at(report, "legal_actions", 1), ">=1000")
        self.assertEqual(observations.value_at(report, "terminal", 2), "checkmate")
        self.assertEqual(observations.value_at(report, "side", 2), "white")
        self.assertIsNone(observations.value_at(report, "in_check", 3))
        self.assertIsNone(observations.value_at(obs("a", [(None, None, 20, None)]), "in_check", 0))

    def test_legality_and_length(self):
        played = obs("a", [START, REPLY, START])
        refused = obs("b", [START, REPLY], status="rejected", reason="no_matching_move")
        self.assertEqual(observations.value_at(played, "legality", 2), "accepted")
        self.assertEqual(observations.value_at(refused, "legality", 2), "rejected:no_matching_move")
        self.assertEqual(observations.value_at(refused, "legality", 1), "accepted")
        self.assertIsNone(observations.value_at(refused, "legality", 3))
        self.assertEqual(observations.value_at(played, "length", 3), "2 actions")
        self.assertEqual(observations.value_at(refused, "length", 2), "at least 2 actions")  # it read the one it refused
        self.assertIsNone(observations.value_at(obs("a", [], status="unsupported"), "length", 1))
        # Actions are counted from 1, and a length disagreement sits next to the actions read.
        self.assertIsNone(observations.value_at(played, "legality", 0))
        self.assertIsNone(observations.value_at(played, "length", 9999))
        self.assertIsNone(observations.value_at(played, "length", 0))
        self.assertIsNone(observations.value_at(refused, "length", 3))

    def test_it_agrees_with_compare(self):
        a = obs("a", [START, ("black", True, 1000, "none"), START])
        b = obs("b", [START, ("black", False, 7, "none")], status="rejected", reason="cannot_submit")
        for item in observations.compare(a, b):
            self.assertEqual(observations.value_at(a, item.kind, item.index), item.a_value, item)
            self.assertEqual(observations.value_at(b, item.kind, item.index), item.b_value, item)


class ValidateTests(unittest.TestCase):
    def test_a_consistent_report_is_valid(self):
        self.assertEqual(observations.validate(obs("a", [START, REPLY, MATE])), [])
        self.assertEqual(observations.validate(obs("a", [], status="unsupported")), [])
        self.assertEqual(observations.validate(obs("a", [START], status="timeout")), [])

    def test_contradictions_are_reported(self):
        cases = {
            "count says playable, terminal says not": [("white", True, 3, "checkmate")],
            "count says nothing playable, terminal says playable": [("white", False, 0, "none")],
            "checkmate without check": [("white", False, 0, "checkmate")],
            "stalemate in check": [("white", True, 0, "stalemate")],
            "count above the cap": [("white", False, 1001, "none")],
            "unknown side": [("purple", False, 1, "none")],
        }
        for name, rows in cases.items():
            self.assertTrue(observations.validate(obs("a", rows)), name)

    def test_every_status_must_account_for_its_positions(self):
        # Position k exists exactly when k actions were accepted, whatever ended the run.
        for status in observations.STATUSES:
            self.assertTrue(observations.validate(obs("a", [START, REPLY], status=status, accepted=5)), status)
            self.assertTrue(observations.validate(obs("a", [START, REPLY], status=status, accepted=0)), status)
        missing = obs("a", [START, REPLY])
        missing.actions_accepted = None
        self.assertTrue(observations.validate(missing))

    def test_a_replay_has_at_least_its_starting_position(self):
        for status in ("ok", "rejected"):
            self.assertTrue(observations.validate(obs("a", [], status=status)), status)

    def test_nothing_is_played_from_a_position_with_no_legal_action(self):
        # A source that calls a position mate and then accepts a move from it
        # has not checked that move.
        problems = observations.validate(obs("a", [START, MATE, START]))
        self.assertEqual(len(problems), 1)
        self.assertIn("position 1: no legal action exists, yet action 2 was accepted", problems[0])
        counted = [START, (None, None, 0, "no_action"), (None, None, 20, "none")]
        self.assertTrue(observations.validate(obs("a", counted)))

    def test_positions_must_be_numbered_in_order(self):
        broken = obs("a", [START, REPLY])
        broken.positions[1] = Position(7, *REPLY)
        self.assertTrue(observations.validate(broken))


class GoldenSupportTests(unittest.TestCase):
    def test_status_change_is_reported(self):
        before = obs("a", [START, REPLY])
        after = obs("a", [START], status="rejected", reason="illegal_move")
        self.assertIn("was ok", observations.status_changed(before, after))
        self.assertIsNone(observations.status_changed(before, obs("a", [START, REPLY])))

    def test_how_far_a_run_got_before_its_time_ran_out_is_never_pinned(self):
        done = obs("a", [START, REPLY])
        slow = obs("a", [START], status="timeout", reason="timeout")
        self.assertIsNone(observations.status_changed(done, slow))
        self.assertIsNone(observations.status_changed(slow, done))
        self.assertIsNone(observations.status_changed(slow, obs("a", [START, REPLY, START], status="timeout", reason="timeout")))

    def test_a_finished_run_cannot_have_stopped_before_the_point_a_timed_out_one_reached(self):
        slow = obs("a", [START, REPLY, START], status="timeout", reason="timeout")
        refused = obs("a", [START, REPLY], status="rejected", reason="illegal_move")
        self.assertIn("reached position 2 before timing out", observations.status_changed(slow, refused))
        self.assertIsNone(observations.status_changed(slow, obs("a", [START, REPLY, START, REPLY])))

    def test_a_crash_after_a_recorded_time_out_is_a_crash(self):
        # However far it got first.
        slow = obs("a", [START, REPLY], status="timeout", reason="timeout")
        died = obs("a", [START, REPLY, START, REPLY], status="error", reason="crash")
        self.assertIn("the recorded run timed out, now error/crash", observations.status_changed(slow, died))

    def test_a_crash_is_an_outcome_like_any_other(self):
        done = obs("a", [START, REPLY])
        for reason in ("crash", "exception", "malformed_output"):
            died = obs("a", [START], status="error", reason=reason)
            self.assertIsNotNone(observations.status_changed(done, died), reason)
            self.assertIsNotNone(observations.status_changed(died, done), reason)
        self.assertFalse(observations.cut_short(obs("a", [START], status="error", reason="crash")))

    def test_coverage_counts_recorded_values_that_were_reproduced(self):
        golden = obs("a", [START, REPLY])
        self.assertEqual(observations.coverage(golden, obs("a", [START, REPLY])), (8, 8))
        self.assertEqual(observations.coverage(golden, obs("a", [START], status="timeout")), (4, 8))
        self.assertEqual(observations.coverage(golden, obs("a", [(None, None, 20, None), REPLY])), (5, 8))

    def test_what_settles_a_question(self):
        self.assertTrue(observations.decisive("in_check", False))
        self.assertTrue(observations.decisive("legal_actions", 0))
        self.assertTrue(observations.decisive("terminal", "none"))
        self.assertTrue(observations.decisive("terminal", "stalemate"))
        self.assertFalse(observations.decisive("terminal", "no_action"))  # which kind of end is it?
        self.assertFalse(observations.decisive("in_check", None))

    def test_json_round_trip(self):
        original = obs("a", [START, ("black", True, 20, "none"), (None, None, None, None)],
                       status="rejected", reason="cannot_submit", accepted=2, last_action=True)
        original.detail = "why"
        restored = observations.from_json("a", "r/1", 1000, observations.to_json(original))
        self.assertEqual(restored.positions, original.positions)
        for name in ("status", "reason", "detail", "actions_accepted", "last_action", "cap"):
            self.assertEqual(getattr(restored, name), getattr(original, name), name)


if __name__ == "__main__":
    unittest.main()
