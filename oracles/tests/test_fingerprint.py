"""The state event and the canonical fingerprint (decision H30)."""

import json
import unittest

from oracle_harness import build, fingerprint, lockfile
from oracle_harness.adapters import RunOptions, make_source
from oracle_harness.fingerprint import StateError, canonical_actions, md5_hex, parse_state, timeline_label
from oracle_harness.util import run

STANDARD = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*"
# The Standard board after 1. e3: the pawn has moved, so it has lost its marker.
AFTER_E3 = "r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/4P3/P*P*P*P*1P*P*P*/R*NBQK*BNR*"
# The notation specification's published full-state hashes (README "Hashing", "Examples").
STANDARD_HASH = "d574889fd9da3f2bc65249ff27249b00"
TWO_TIMELINE_HASH = "3672761404ffcd15ae644c75401812be"


def state(*boards, size="8x8", markers=True, index=0, **extra):
    return {"event": "state", "index": index, "size": size, "markers": markers, "boards": list(boards), **extra}


def rejected(test: unittest.TestCase, event: dict, words: str, **options) -> None:
    with test.assertRaises(StateError) as caught:
        parse_state(event, **options)
    test.assertIn(words, str(caught.exception))


class PublishedHashTests(unittest.TestCase):
    """The notation specification's two examples are this module's conformance tests."""

    def test_the_standard_start(self):
        parsed = parse_state(state(f"0:1:w:{STANDARD}"))
        self.assertEqual(parsed.canonical(), f"[{STANDARD}:0:1:w]")
        self.assertEqual(parsed.fingerprint, STANDARD_HASH)
        self.assertEqual(parsed.summary(), {"fingerprint": STANDARD_HASH, "placement": md5_hex(f"[{STANDARD}:0:1:w]".replace("*", "")),
                                            "boards": 1, "pieces": 32})

    def test_the_two_timeline_standard_start(self):
        # Listed in either order: the harness sorts, -0 before +0.
        for boards in ((f"-0:1:w:{STANDARD}", f"+0:1:w:{STANDARD}"), (f"+0:1:w:{STANDARD}", f"-0:1:w:{STANDARD}")):
            parsed = parse_state(state(*boards))
            self.assertEqual(parsed.fingerprint, TWO_TIMELINE_HASH, boards)
            self.assertEqual((len(parsed.boards), parsed.pieces), (2, 64))


class CanonicalOrderTests(unittest.TestCase):
    def labels(self, *boards):
        return [(board.timeline, board.turn, board.colour) for board in parse_state(state(*boards)).boards]

    def test_timelines_ascend_with_minus_zero_before_plus_zero(self):
        boards = [f"{label}:1:w:{STANDARD}" for label in ("+1", "+0", "-1", "-0", "2", "-2")]
        self.assertEqual([label for label, _, _ in self.labels(*boards)], ["-2", "-1", "-0", "+0", "+1", "+2"])
        odd = [f"{label}:1:w:{STANDARD}" for label in ("1", "0", "-1")]
        self.assertEqual([label for label, _, _ in self.labels(*odd)], ["-1", "0", "+1"])

    def test_then_turn_then_white_before_black(self):
        boards = [f"0:2:b:{STANDARD}", f"0:1:b:{AFTER_E3}", f"0:2:w:{STANDARD}", f"0:1:w:{STANDARD}", f"0:10:w:{STANDARD}"]
        self.assertEqual(self.labels(*boards), [("0", 1, "w"), ("0", 1, "b"), ("0", 2, "w"), ("0", 2, "b"), ("0", 10, "w")])
        # Turn zero sorts first (Standard - Turn Zero starts with a board at T0).
        self.assertEqual(self.labels(f"0:1:w:{STANDARD}", f"0:0:b:{STANDARD}"), [("0", 0, "b"), ("0", 1, "w")])

    def test_the_order_given_does_not_change_the_fingerprint(self):
        boards = [f"0:1:w:{STANDARD}", f"0:1:b:{AFTER_E3}", f"+1:1:b:{AFTER_E3}"]
        expected = parse_state(state(*boards)).fingerprint
        self.assertEqual(parse_state(state(*reversed(boards))).fingerprint, expected)
        self.assertEqual(parse_state(state(*boards[1:], boards[0])).fingerprint, expected)

    def test_unsigned_positive_labels_are_written_with_their_sign(self):
        self.assertEqual(timeline_label("1"), "+1")
        self.assertEqual(timeline_label("+12"), "+12")
        self.assertEqual(timeline_label("-3"), "-3")
        for label in ("0", "-0", "+0"):
            self.assertEqual(timeline_label(label), label)
        unsigned = parse_state(state(f"0:1:w:{STANDARD}", f"1:1:b:{AFTER_E3}"))
        signed = parse_state(state(f"0:1:w:{STANDARD}", f"+1:1:b:{AFTER_E3}"))
        self.assertEqual(unsigned.canonical(), signed.canonical())
        self.assertIn(":+1:1:b]", unsigned.canonical())


class PlacementTests(unittest.TestCase):
    def test_runs_of_empty_squares_are_rewritten_from_the_squares(self):
        # Ten files wide: "10" is one run of ten, not a one and a zero.
        parsed = parse_state(state("0:1:w:10/k9/10/9K", size="10x4", markers=False))
        self.assertEqual(parsed.boards[0].placement, "10/k9/10/9K")
        self.assertEqual(parsed.pieces, 2)
        self.assertEqual(parsed.boards[0].canonical(), "[10/k9/10/9K:0:1:w]")

    def test_boards_that_are_not_eight_by_eight(self):
        rook_tactics = parse_state(state("0:1:w:4k/5/5/5/K1R2", size="5x5", markers=False))
        self.assertEqual(rook_tactics.canonical(), "[4k/5/5/5/K1R2:0:1:w]")
        self.assertIsNone(rook_tactics.fingerprint)  # no markers, no fingerprint
        self.assertEqual(rook_tactics.placement, md5_hex("[4k/5/5/5/K1R2:0:1:w]"))
        wide = parse_state(state("0:1:w:k*3/4/4/3K*/4/4", size="4x6"))
        self.assertEqual((wide.width, wide.height, wide.pieces), (4, 6, 2))

    def test_the_placement_hash_is_the_fingerprint_without_markers(self):
        marked = parse_state(state(f"0:1:w:{STANDARD}"))
        unmarked = parse_state(state(f"0:1:w:{STANDARD.replace('*', '')}", markers=False))
        self.assertEqual(marked.placement, unmarked.placement)
        self.assertNotEqual(marked.fingerprint, marked.placement)
        self.assertEqual(marked.placement, md5_hex(f"[{STANDARD.replace('*', '')}:0:1:w]"))

    def test_every_piece_letter_in_both_colours(self):
        parsed = parse_state(state("0:1:w:PNBRQKUDSWCY/pnbrqkudswcy", size="12x2", markers=False))
        self.assertEqual(parsed.pieces, 24)

    def test_only_pawns_rooks_kings_and_brawns_carry_the_marker(self):
        parsed = parse_state(state("0:1:w:P*R*K*W*/p*r*k*w*", size="4x2"))
        self.assertEqual(parsed.boards[0].placement, "P*R*K*W*/p*r*k*w*")


class RejectionTests(unittest.TestCase):
    def test_a_bad_timeline_label(self):
        for label in ("01", "+-1", "", "a", "1.5", "--0", "+", " 1"):
            rejected(self, state(f"{label}:1:w:{STANDARD}"), "bad timeline label")

    def test_a_bad_turn_or_colour(self):
        rejected(self, state(f"0:-1:w:{STANDARD}"), "bad turn")
        rejected(self, state(f"0:01:w:{STANDARD}"), "bad turn")
        rejected(self, state(f"0:1:white:{STANDARD}"), "bad colour")
        rejected(self, state(f"0:1:{STANDARD}"), "expected \"<L>:<T>:<c>:<placement>\"")
        rejected(self, state(STANDARD), "expected \"<L>:<T>:<c>:<placement>\"")
        # The 5DFEN order ("placement:L:T:c") is not the state event's.
        rejected(self, state(f"{STANDARD}:0:1:w"), "bad timeline label")

    def test_the_wrong_number_of_ranks(self):
        rejected(self, state("0:1:w:8/8/8/8/8/8/8"), "7 rank(s), but the board is 8 high")
        rejected(self, state(f"0:1:w:{STANDARD}/8"), "9 rank(s), but the board is 8 high")

    def test_a_rank_of_the_wrong_width(self):
        rejected(self, state("0:1:w:9/8/8/8/8/8/8/8"), "rank 8 is 9 square(s) wide, but the board is 8 wide")
        rejected(self, state("0:1:w:8/8/8/8/8/8/8/7"), "rank 1 is 7 square(s) wide")
        rejected(self, state("0:1:w:8/8/8//8/8/8/8"), "rank 5 is 0 square(s) wide")
        # Two adjacent numbers are one number: "44" is forty-four squares.
        rejected(self, state("0:1:w:44/8/8/8/8/8/8/8"), "44 square(s) wide")

    def test_an_unknown_letter(self):
        for text in ("X7", "+P7", "4 4", "4.4", "Ö7"):
            rejected(self, state(f"0:1:w:{text}/8/8/8/8/8/8/8"), "unknown letter")

    def test_a_marker_after_a_piece_that_does_not_carry_one(self):
        for letter in "NBQUDSCYnbqudscy":
            rejected(self, state(f"0:1:w:{letter}*7/8/8/8/8/8/8/8"), f"'*' after '{letter}'")

    def test_a_marker_from_a_source_that_says_it_tracks_none(self):
        rejected(self, state(f"0:1:w:{STANDARD}", markers=False), "markers is false")

    def test_a_marker_that_follows_no_piece(self):
        for text in ("*8", "P**7", "4*4"):
            rejected(self, state(f"0:1:w:{text}/8/8/8/8/8/8/8"), "'*' that does not follow a piece letter")

    def test_empty_runs_and_leading_zeros(self):
        rejected(self, state("0:1:w:08/8/8/8/8/8/8/8"), "without leading zeros")
        rejected(self, state("0:1:w:P07/8/8/8/8/8/8/8"), "without leading zeros")

    def test_the_event_itself(self):
        good = state(f"0:1:w:{STANDARD}")
        for name, value, words in (
                ("index", -1, "bad index"), ("index", "0", "bad index"), ("index", True, "bad index"),
                ("size", "8", "bad size"), ("size", "0x8", "bad size"), ("size", 64, "bad size"),
                ("markers", "yes", "bad markers"), ("markers", None, "bad markers"),
                ("boards", [], "non-empty list"), ("boards", f"0:1:w:{STANDARD}", "non-empty list"),
                ("boards", [7], "a board is a string")):
            rejected(self, {**good, name: value}, words)
        rejected(self, {**good, "event": "position"}, "not a state event")

    def test_a_board_listed_twice(self):
        rejected(self, state(f"0:1:w:{STANDARD}", f"0:1:w:{STANDARD}"), "two boards at timeline 0, turn 1, w")
        rejected(self, state(f"1:1:w:{STANDARD}", f"+1:1:w:{STANDARD}"), "two boards at timeline +1")

    def test_a_central_timeline_spelled_two_ways(self):
        # An odd game has "0"; an even one has "-0" and "+0". Never both: that is a
        # source whose internal numbering leaked into its labels.
        rejected(self, state(f"0:1:w:{STANDARD}", f"+0:1:w:{STANDARD}"), "mix '0' with '-0'/'+0'")

    def test_exact_fields_are_held_to_on_request(self):
        extra = state(f"0:1:w:{STANDARD}", ms=1.0)
        self.assertEqual(parse_state(extra).fingerprint, STANDARD_HASH)  # an oracle may add fields
        rejected(self, extra, "a state event has exactly", exact_fields=True)
        self.assertEqual(parse_state(state(f"0:1:w:{STANDARD}"), exact_fields=True).fingerprint, STANDARD_HASH)


def move(source, target, promotion=None):
    def end(where):
        label, turn, colour, square = where.split(":")
        return {"L": label, "T": int(turn), "c": colour, "square": square}
    return {"from": end(source), "to": end(target), "promotion": promotion}


class ActionsEventTests(unittest.TestCase):
    def event(self, actions, count=None, truncated=False):
        return {"event": "actions", "index": 3, "count": len(actions) if count is None else count,
                "truncated": truncated, "actions": actions}

    def test_moves_and_actions_are_sorted_and_labels_spelled_one_way(self):
        second = [move("1:2:w:e2", "1:2:w:e4"), move("0:2:w:a7", "0:2:w:a8", "q")]
        first = [move("0:2:w:b1", "0:2:w:c3")]
        shown = canonical_actions(self.event([second, first]))
        self.assertEqual(shown["count"], 2)
        self.assertEqual([[m["from"]["square"] for m in action] for action in shown["actions"]], [["a7", "e2"], ["b1"]])
        self.assertEqual(shown["actions"][0][1]["from"]["L"], "+1")
        self.assertEqual(shown["actions"][0][0]["promotion"], "Q")
        # The same lists, written in another order, come out the same.
        self.assertEqual(canonical_actions(self.event([first, list(reversed(second))])), shown)

    def test_squares_sort_by_number_not_by_text(self):
        far = move("0:1:w:a10", "0:1:w:a9")
        near = move("0:1:w:a9", "0:1:w:a8")
        shown = canonical_actions(self.event([[far], [near]]))
        self.assertEqual([action[0]["from"]["square"] for action in shown["actions"]], ["a9", "a10"])

    def test_a_list_cut_at_its_limit_says_so(self):
        listed = [[move("0:1:w:e2", "0:1:w:e4")]]
        self.assertTrue(canonical_actions(self.event(listed, count=9000, truncated=True))["truncated"])
        for count, truncated in ((9000, False), (0, True)):
            with self.assertRaises(StateError):
                canonical_actions(self.event(listed, count=count, truncated=truncated))
        with self.assertRaises(StateError):
            canonical_actions(self.event([[move("0:1:w:e2", "0:1:w:e4")]] * (fingerprint.ACTIONS_LIMIT + 1)))

    def test_a_drivers_own_fields_are_kept_ahead_of_the_list(self):
        # They say how far the list is the source's whole answer: a budget that ran
        # out, an enumeration that failed part-way.
        listed = [[move("0:1:w:e2", "0:1:w:e4")]]
        shown = canonical_actions({**self.event(listed, count=7, truncated=True), "complete": False,
                                   "limit": "actions_budget_ms", "error": "boom"})
        self.assertEqual(list(shown), ["event", "index", "count", "truncated", "complete", "limit", "error", "actions"])
        self.assertEqual((shown["limit"], shown["error"], shown["complete"]), ("actions_budget_ms", "boom", False))

    def test_malformed_moves_are_refused(self):
        for bad in (move("01:1:w:e2", "0:1:w:e4"), move("0:1:x:e2", "0:1:w:e4"), move("0:1:w:e0", "0:1:w:e4"),
                    move("0:1:w:e2", "0:1:w:e4", "X"), {"from": {"L": "0"}, "to": {}}, "e2e4"):
            with self.assertRaises(StateError, msg=str(bad)):
                canonical_actions(self.event([[bad]]))
        with self.assertRaises(StateError):
            canonical_actions(self.event([[]]))


# The notation specification's two-timeline Standard start, as a custom record,
# and three turns: white branches into +1 at its third, black answers into -1.
TWO_TIMELINE_GAME = (
    '[Size "8x8"]\n[Board "custom"]\n'
    f"[{STANDARD}:-0:1:w]\n[{STANDARD}:+0:1:w]\n\n"
    "1. (-0T1)Nf3 (+0T1)Nf3 / (-0T1)Nf6 (+0T1)Nf6\n"
    "2. (-0T2)Nc3 (+0T2)Nc3 / (-0T2)Nc6 (+0T2)Nc6\n"
    "3. (-0T3)a3 (+0T3)Nc3>>(+0T2)c5 / (+1T2)Nf6>>(-0T1)f6\n"
).encode()


class EvenGameLabelTests(unittest.TestCase):
    """Each driver names the timelines of an even game itself - ftxi from its own
    line numbers, 5d-chess-js through the library's export - and nothing in the
    corpus has one they both read. Run where the oracles are built."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()

    def observe(self, name):
        source = make_source(name, self.lock)
        if source.available():
            self.skipTest(source.available())
        return source.observe("even", TWO_TIMELINE_GAME, RunOptions(cap=1000, timeout_s=120))

    def test_the_drivers_agree_from_the_specifications_example_on(self):
        reports = {name: self.observe(name) for name in ("ftxi", "chessin5d")}
        for name, report in reports.items():
            self.assertEqual((report.status, len(report.positions)), ("ok", 7), (name, report.detail))
            self.assertEqual(report.positions[0].fingerprint, TWO_TIMELINE_HASH, name)
        self.assertEqual([p.fingerprint for p in reports["ftxi"].positions],
                         [p.fingerprint for p in reports["chessin5d"].positions])

    def test_the_branches_are_named_plus_one_and_minus_one(self):
        built = build.load(self.lock.pin("ftxi"))
        if built is None:
            self.skipTest("ftxi is not built")
        done = run([built.artifact("driver")], stdin=TWO_TIMELINE_GAME, timeout=120)
        states = [event for event in map(json.loads, done.out_text().splitlines()) if event["event"] == "state"]
        labels = [sorted({board.split(":")[0] for board in state["boards"]}) for state in states]
        self.assertEqual(labels[0], ["+0", "-0"])
        self.assertEqual(labels[5], ["+0", "+1", "-0"])  # white's branch
        self.assertEqual(labels[6], ["+0", "+1", "-0", "-1"])  # black's


if __name__ == "__main__":
    unittest.main()
