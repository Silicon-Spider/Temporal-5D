import dataclasses
import unittest

from oracle_harness import observations
from oracle_harness.adapters import PARSE_ONLY_SOURCES
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


def hashed(number: int) -> str:
    """A stand-in hash: 32 lower-case hex digits."""
    return f"{number:032x}"


def set_up(observation, placements, fingerprints=None, boards=1, pieces=32):
    """Give the positions of a report what the source set up: placement hashes, and fingerprints if marked."""
    fingerprints = fingerprints or [None] * len(placements)
    observation.positions = [
        dataclasses.replace(position, placement=None if placement is None else hashed(placement),
                            fingerprint=None if marked is None else hashed(marked),
                            boards=None if placement is None else boards, pieces=None if placement is None else pieces)
        for position, placement, marked in zip(observation.positions, placements, fingerprints, strict=True)]
    return observation


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

    def test_a_parse_only_source_playing_on_is_no_verdict_against_a_refusal(self):
        # It applies any move it can place, legal or not (adapters.PARSE_ONLY_SOURCES):
        # playing on says only that it read the move.
        reader = PARSE_ONLY_SOURCES[0]
        read_on = obs(reader, [START, REPLY, START])
        held_illegal = obs("b", [START, REPLY], status="rejected", reason="cannot_submit")
        self.assertEqual(self.kinds(read_on, held_illegal), [])
        self.assertEqual(self.kinds(held_illegal, read_on), [])
        # Against a refusal that was a failure to read the move text, it shows the text can be read.
        unread = obs("b", [START, REPLY], status="rejected", reason="no_matching_move")
        self.assertEqual(self.kinds(read_on, unread), [("legality", 2, "accepted", "rejected:no_matching_move", True)])
        # Its own failure to read a move is a soft refusal like any other.
        stuck = obs(reader, [START, REPLY], status="rejected", reason="no_matching_move")
        self.assertEqual(self.kinds(obs("b", [START, REPLY, START]), stuck),
                         [("legality", 2, "accepted", "rejected:no_matching_move", True)])
        self.assertFalse(observations.judges_actions(reader))
        self.assertTrue(observations.judges_actions("ftxi:naive"))


class ParseComparisonTests(unittest.TestCase):
    """What the two sources set up is compared first (decision H30)."""

    def kinds(self, a, b, **options):
        return [(item.kind, item.index, item.a_value, item.b_value) for item in observations.compare(a, b, **options)]

    def test_the_same_positions_agree(self):
        a = set_up(obs("a", [START, REPLY, START]), [1, 2, 3], [11, 12, 13])
        b = set_up(obs("b", [START, REPLY, START]), [1, 2, 3], [11, 12, 13])
        self.assertEqual(self.kinds(a, b), [])

    def test_a_placement_divergence_is_reported_once_at_the_first_position(self):
        a = set_up(obs("a", [START, REPLY, START, REPLY]), [1, 2, 3, 4])
        b = set_up(obs("b", [START, REPLY, START, REPLY]), [1, 2, 7, 8], boards=2, pieces=31)
        self.assertEqual(self.kinds(a, b), [("placement", 2, f"{hashed(3)} (1 board, 32 pieces)",
                                             f"{hashed(7)} (2 boards, 31 pieces)")])
        self.assertEqual(observations.first_placement_divergence(a, b), 2)

    def test_nothing_is_compared_from_a_placement_divergence_on(self):
        # Every answer from position 2 on differs - and none of it is listed:
        # it is about two different positions.
        a = set_up(obs("a", [START, REPLY, START, REPLY]), [1, 2, 3, 4])
        b = set_up(obs("b", [START, REPLY, ("black", True, 5, "none"), ("white", True, 0, "checkmate")]), [1, 2, 7, 8])
        self.assertEqual([kind for kind, *_ in self.kinds(a, b)], ["placement"])
        # Nor is an action played from a position the two no longer share: refused or read past the end.
        refused = set_up(obs("b", [START, REPLY, START], status="rejected", reason="illegal_move"), [1, 2, 7])
        self.assertEqual([kind for kind, *_ in self.kinds(a, refused)], ["placement"])
        ended = set_up(obs("b", [START, REPLY, START]), [1, 2, 7])
        self.assertEqual([kind for kind, *_ in self.kinds(a, ended)], ["placement"])
        # Before the divergence everything is compared as usual.
        early = set_up(obs("b", [("black", False, 20, "none"), REPLY, START, REPLY]), [1, 2, 7, 8])
        self.assertEqual([kind for kind, *_ in self.kinds(a, early)], ["placement", "side"])
        # And where the two never parted, a refusal is a legality disagreement as before.
        stopped = set_up(obs("b", [START, REPLY], status="rejected", reason="illegal_move"), [1, 2])
        self.assertEqual([kind for kind, *_ in self.kinds(a, stopped)], ["legality"])

    def test_a_marker_only_divergence_suppresses_nothing(self):
        # The same pieces, marked differently from position 1 on (a source that
        # keeps a rook's marker after its king has moved, say): reported once,
        # beside a rules disagreement that is still compared.
        a = set_up(obs("a", [START, REPLY, START]), [1, 2, 3], [11, 12, 13])
        b = set_up(obs("b", [START, ("black", True, 20, "none"), START]), [1, 2, 3], [11, 22, 23])
        self.assertEqual(self.kinds(a, b), [
            ("unmoved", 1, f"{hashed(12)} (1 board, 32 pieces)", f"{hashed(22)} (1 board, 32 pieces)"),
            ("in_check", 1, "False", "True"),
        ])

    def test_each_run_of_differing_markers_is_reported_where_it_begins(self):
        # Differ at 1 and 2, agree at 3, differ again at 4: two runs, two entries.
        a = set_up(obs("a", [START, REPLY, START, REPLY, START]), [1, 2, 3, 4, 5], [11, 12, 13, 14, 15])
        b = set_up(obs("b", [START, REPLY, START, REPLY, START]), [1, 2, 3, 4, 5], [11, 22, 23, 14, 25])
        self.assertEqual([(kind, index) for kind, index, *_ in self.kinds(a, b)], [("unmoved", 1), ("unmoved", 4)])
        # A position whose markers one side does not give neither ends a run nor starts one.
        gap = set_up(obs("b", [START, REPLY, START, REPLY, START]), [1, 2, 3, 4, 5], [11, 22, None, 24, 25])
        self.assertEqual([(kind, index) for kind, index, *_ in self.kinds(a, gap)], [("unmoved", 1)])

    def test_the_same_state_includes_the_markers(self):
        def at(placement, fingerprint=None):
            return Position(0, placement=hashed(placement), fingerprint=None if fingerprint is None else hashed(fingerprint))

        self.assertTrue(observations.same_state(at(1, 11), at(1, 11)))
        self.assertFalse(observations.same_state(at(1, 11), at(1, 21)))
        self.assertTrue(observations.aligned(at(1, 11), at(1, 21)))  # the same boards and pieces all the same
        self.assertIsNone(observations.same_state(at(1, 11), at(1)))  # one side does not track the markers
        self.assertIsNone(observations.same_state(at(1, 11), Position(0)))

    def test_a_marker_only_divergence_is_looked_for_only_before_the_two_part(self):
        a = set_up(obs("a", [START, REPLY, START]), [1, 2, 3], [11, 12, 13])
        b = set_up(obs("b", [START, REPLY, START]), [1, 9, 3], [11, 19, 23])
        self.assertEqual([(kind, index) for kind, index, *_ in self.kinds(a, b)], [("placement", 1)])
        c = set_up(obs("c", [START, REPLY, START]), [1, 2, 9], [21, 12, 19])
        self.assertEqual([(kind, index) for kind, index, *_ in self.kinds(a, c)], [("unmoved", 0), ("placement", 2)])

    def test_a_source_that_says_nothing_about_its_state_never_diverges(self):
        a = set_up(obs("a", [START, REPLY]), [1, 2], [11, 12])
        silent = obs("b", [START, REPLY])
        self.assertEqual(self.kinds(a, silent), [])
        unmarked = set_up(obs("b", [START, REPLY]), [1, 2])  # placements only: no fingerprint
        self.assertEqual(self.kinds(a, unmarked), [])
        partly = set_up(obs("b", [START, REPLY]), [None, 5])
        self.assertEqual([(kind, index) for kind, index, *_ in self.kinds(a, partly)], [("placement", 1)])

    def test_two_searches_of_one_implementation_must_parse_alike(self):
        # Legality and length are left out between searches; what was set up is not.
        a = set_up(obs("a", [START, REPLY]), [1, 2])
        b = set_up(obs("b", [START, REPLY]), [1, 3])
        self.assertEqual([kind for kind, *_ in self.kinds(a, b, replay=False)], ["placement"])

    def test_the_description_says_what_a_parse_divergence_means(self):
        a = set_up(obs("a", [START]), [1], [11])
        b = set_up(obs("b", [START]), [2], [12])
        described = observations.compare(a, b)[0].describe()
        self.assertTrue(described.startswith(f"r/1 position 0: placement: a={hashed(1)} (1 board, 32 pieces) vs b="))
        self.assertIn("parse divergence", described)


class ParseValueTests(unittest.TestCase):
    def test_value_at_prints_what_compare_prints(self):
        a = set_up(obs("a", [START, REPLY, START]), [1, 2, 3], [11, 12, 13])
        for b in (set_up(obs("b", [START, REPLY, START]), [1, 2, 9], [11, 12, 19]),
                  set_up(obs("b", [START, REPLY, START]), [1, 2, 3], [11, 22, 13])):
            found = observations.compare(a, b)
            self.assertTrue(found)
            for item in found:
                self.assertEqual(observations.value_at(a, item.kind, item.index), item.a_value, item)
                self.assertEqual(observations.value_at(b, item.kind, item.index), item.b_value, item)

    def test_value_at_is_silent_where_nothing_was_said(self):
        a = set_up(obs("a", [START, REPLY]), [1, None])
        self.assertIsNone(observations.value_at(a, "placement", 1))
        self.assertIsNone(observations.value_at(a, "unmoved", 0))  # no markers
        self.assertIsNone(observations.value_at(a, "placement", 5))
        self.assertEqual(observations.value_at(a, "placement", 0), f"{hashed(1)} (1 board, 32 pieces)")

    def test_aligned(self):
        self.assertTrue(observations.aligned(Position(0, placement=hashed(1)), Position(0, placement=hashed(1))))
        self.assertFalse(observations.aligned(Position(0, placement=hashed(1)), Position(0, placement=hashed(2))))
        self.assertIsNone(observations.aligned(Position(0, placement=hashed(1)), Position(0)))


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
        # Actions are counted from 1 - action 0 is the record's set-up, which a source that reported position 0
        # read - and a length disagreement sits next to the actions read.
        self.assertEqual(observations.value_at(played, "legality", 0), "accepted")
        self.assertIsNone(observations.value_at(obs("a", [], status="unsupported"), "legality", 0))
        self.assertIsNone(observations.value_at(played, "legality", -1))
        self.assertIsNone(observations.value_at(played, "length", 9999))
        self.assertIsNone(observations.value_at(played, "length", 0))
        self.assertIsNone(observations.value_at(refused, "length", 3))

    def test_it_agrees_with_compare(self):
        a = obs("a", [START, ("black", True, 1000, "none"), START])
        b = obs("b", [START, ("black", False, 7, "none")], status="rejected", reason="cannot_submit")
        for item in observations.compare(a, b):
            self.assertEqual(observations.value_at(a, item.kind, item.index), item.a_value, item)
            self.assertEqual(observations.value_at(b, item.kind, item.index), item.b_value, item)


class SetUpRefusalTests(unittest.TestCase):
    """An engine in parse mode that cannot read a record's headers or board strings refuses its set-up: action 0."""

    def refusal(self):
        return obs("engine", [], status="rejected", reason="unreadable")

    def test_it_is_a_refusal_of_action_0(self):
        refusal = self.refusal()
        self.assertTrue(observations.refused_setup(refusal))
        self.assertEqual(observations.refused_action(refusal), 0)
        self.assertEqual(observations.value_at(refusal, "legality", 0), "rejected:unreadable")
        self.assertIsNone(observations.value_at(refusal, "legality", 1))
        self.assertIsNone(observations.value_at(refusal, "length", 1))
        self.assertEqual(observations.validate(refusal), [])
        # A refusal of an action, or a run that reported nothing for any other reason, is not one.
        self.assertEqual(observations.refused_action(obs("a", [START], status="rejected", reason="unreadable")), 1)
        self.assertFalse(observations.refused_setup(obs("a", [], status="unsupported")))
        self.assertTrue(observations.validate(obs("a", [], status="rejected", reason="illegal_move")))

    def test_a_source_that_set_the_record_up_contradicts_it_softly_and_nothing_else_is_compared(self):
        for rows in ([START], [START, REPLY, MATE]):
            oracle = obs("ftxi", rows)
            found = observations.compare(self.refusal(), oracle)
            self.assertEqual([(item.kind, item.index, item.a_value, item.b_value, item.soft) for item in found],
                             [("legality", 0, "rejected:unreadable", "accepted", True)])
            self.assertIn("action 0 (the record's set-up): legality: engine=rejected:unreadable vs ftxi=accepted",
                          found[0].describe())
        # A parse-only source that set it up contradicts it too: the refusal is a failure to read.
        self.assertEqual(len(observations.compare(self.refusal(), obs(PARSE_ONLY_SOURCES[0], [START]))), 1)
        # A source that did not set it up either agrees.
        for other in (self.refusal(), obs("cwmtt", [], status="unsupported", reason="fen")):
            self.assertEqual(observations.compare(self.refusal(), other), [])


class AgreementTests(unittest.TestCase):
    """Two reports agree about an observation by compare()'s own rule - the telltales' measure (H34)."""

    def test_by_the_comparisons_rule_not_by_the_same_words(self):
        capped = obs("a", [START, ("black", False, 1000, "none")])
        larger = obs("b", [START, ("black", False, 1000, "none")], cap=5000)
        larger.positions[1] = dataclasses.replace(larger.positions[1], legal_actions=1200)
        self.assertTrue(observations.agrees(capped, larger, "legal_actions", 1))
        self.assertFalse(observations.agrees(capped, obs("c", [START, ("black", False, 999, "none")]), "legal_actions", 1))
        no_action = obs("cwmtt", [START, (None, None, 0, "no_action")])
        self.assertTrue(observations.agrees(obs("e", [START, MATE]), no_action, "terminal", 1))
        self.assertFalse(observations.agrees(obs("e", [START, REPLY]), no_action, "terminal", 1))
        # Two refusals of one action agree, whatever reason each gives; a refusal and a play do not.
        soft = obs("ftxi", [START], status="rejected", reason="no_matching_move")
        hard = obs("engine", [START], status="rejected", reason="illegal_move")
        self.assertTrue(observations.agrees(hard, soft, "legality", 1))
        self.assertFalse(observations.agrees(obs("engine", [START, REPLY]), soft, "legality", 1))
        self.assertIsNone(observations.agrees(obs("engine", [START, REPLY]), soft, "in_check", 1))  # one says nothing
        # Two readings of a record's length agree where neither ends before an action the other found.
        self.assertTrue(observations.agrees(obs("a", [START, REPLY]), obs("b", [START, REPLY]), "length", 2))
        self.assertFalse(observations.agrees(obs("a", [START, REPLY]), obs("b", [START, REPLY, START]), "length", 2))
        marked = set_up(obs("a", [START]), [1], [11])
        self.assertTrue(observations.agrees(marked, set_up(obs("b", [START]), [1], [11]), "unmoved", 0))
        self.assertFalse(observations.agrees(marked, set_up(obs("b", [START]), [1], [12]), "unmoved", 0))
        self.assertTrue(observations.agrees(marked, set_up(obs("b", [START]), [1], [12]), "placement", 0))

    def test_printed_values_agree_by_the_same_rule(self):
        self.assertTrue(observations.texts_agree("legal_actions", ">=1000", "1200"))
        self.assertFalse(observations.texts_agree("legal_actions", "998", "1000"))
        self.assertTrue(observations.texts_agree("terminal", "checkmate", "no_action"))
        self.assertTrue(observations.texts_agree("legality", "rejected:illegal_move", "rejected:no_matching_move"))
        self.assertFalse(observations.texts_agree("legality", "accepted", "rejected:cannot_submit"))
        self.assertTrue(observations.texts_agree("in_check", "False", "False"))


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

    def test_what_was_set_up_is_well_formed(self):
        self.assertEqual(observations.validate(set_up(obs("a", [START, REPLY]), [1, 2], [11, 12])), [])
        self.assertEqual(observations.validate(set_up(obs("a", [START, REPLY]), [1, None])), [])
        cases = {
            "upper-case hex": dict(placement="A" * 32, boards=1, pieces=32),
            "too short": dict(placement="a" * 31, boards=1, pieces=32),
            "not a string": dict(placement=7, boards=1, pieces=32),
            "a bad fingerprint": dict(fingerprint="x" * 32, placement="a" * 32, boards=1, pieces=32),
            "negative boards": dict(placement="a" * 32, boards=-1, pieces=32),
            "boolean pieces": dict(placement="a" * 32, boards=1, pieces=True),
            "a fingerprint without a placement": dict(fingerprint="a" * 32, boards=1, pieces=32),
            "a placement without counts": dict(placement="a" * 32),
        }
        for name, fields in cases.items():
            report = obs("a", [START])
            report.positions[0] = Position(0, *START, **fields)
            self.assertTrue(observations.validate(report), name)

    def test_what_was_set_up_is_said_from_the_start_and_marked_throughout(self):
        # Said at every position, or only up to where the source stopped saying it.
        self.assertEqual(observations.validate(set_up(obs("a", [START, REPLY, START]), [1, 2, None])), [])
        gap = observations.validate(set_up(obs("a", [START, REPLY, START]), [1, None, 3]))
        self.assertEqual(len(gap), 1)
        self.assertIn("position 2 says what was set up there, and position 1 does not", gap[0])
        self.assertTrue(observations.validate(set_up(obs("a", [START, REPLY]), [None, 2])))
        # Markers at one position and not at another.
        mixed = observations.validate(set_up(obs("a", [START, REPLY]), [1, 2], [11, None]))
        self.assertEqual(len(mixed), 1)
        self.assertIn("unmoved markers are given at some positions and not at others", mixed[0])


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

    def test_the_values_of_a_report_are_its_answers_and_what_it_set_up(self):
        # What a golden file records, and what required sets are made of (H32).
        self.assertEqual(observations.values(obs("a", [START, (None, None, 20, None)])),
                         [(0, "side"), (0, "in_check"), (0, "legal_actions"), (0, "terminal"), (1, "legal_actions")])
        described = set_up(obs("a", [START]), [1], [11])
        self.assertEqual(len(observations.values(described)), 8)
        # Without markers there is no fingerprint: one value fewer.
        self.assertEqual(len(observations.values(set_up(obs("a", [START]), [1]))), 7)
        cut = set_up(obs("a", [START, (None, None, None, None)], status="timeout"), [1, 2], [11, 12])
        self.assertEqual([name for index, name in observations.values(cut) if index == 1],
                         ["fingerprint", "placement", "boards", "pieces"])

    def test_a_value_is_read_by_position_and_field(self):
        report = obs("a", [START, REPLY])
        self.assertEqual(observations.value(report, 1, "side"), "black")
        self.assertIsNone(observations.value(report, 2, "side"))  # never reached
        self.assertIsNone(observations.value(report, -1, "side"))

    def test_unreadable_is_a_soft_rejection(self):
        # The engine's parser in parse mode could not read the move text: notation, maybe not rules.
        self.assertIn("unreadable", observations.SOFT_REJECTIONS)
        full = obs("a", [START, REPLY, START])
        unread = obs("engine", [START, REPLY], status="rejected", reason="unreadable")
        self.assertEqual(self.kinds(full, unread), [("legality", 2, "accepted", "rejected:unreadable", True)])

    def kinds(self, a, b, **options):
        return [(item.kind, item.index, item.a_value, item.b_value, item.soft) for item in observations.compare(a, b, **options)]

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
        set_up(original, [1, 2, 3], [11, None, 13])
        original.detail = "why"
        restored = observations.from_json("a", "r/1", 1000, observations.to_json(original))
        self.assertEqual(restored.positions, original.positions)
        for name in ("status", "reason", "detail", "actions_accepted", "last_action", "cap"):
            self.assertEqual(getattr(restored, name), getattr(original, name), name)

    def test_a_row_holds_the_answers_then_what_was_set_up(self):
        report = set_up(obs("a", [START]), [1], [11], boards=3, pieces=60)
        self.assertEqual(observations.to_json(report)["positions"],
                         [["w", False, 20, "none", hashed(11), hashed(1), 3, 60]])
        self.assertEqual(observations.ROW_FIELDS, ("side", "in_check", "legal_actions", "terminal",
                                                   "fingerprint", "placement", "boards", "pieces"))
        # A row of the old four columns is not read as one of the new.
        with self.assertRaises(ValueError):
            observations.from_json("a", "r/1", 1000, {"status": "ok", "positions": [["w", False, 20, "none"]]})


if __name__ == "__main__":
    unittest.main()
