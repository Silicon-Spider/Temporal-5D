"""The check that the harness's ftxi driver says what ftxi's own tools say,
exercised with a stand-in for those tools."""

import dataclasses
import json
import unittest
from unittest import mock

from oracle_harness import fidelity
from oracle_harness.corpus import Record
from oracle_harness.fingerprint import md5_hex
from oracle_harness.observations import Observation, Position
from oracle_harness.util import Completed

W = ("white", False, 20, "none")
B = ("black", False, 31, "none")
B_CHECK = ("black", True, 1, "none")
MATE = ("white", True, 0, "checkmate")
RECORD = Record(id="r/1", source="fake", path="p", blob="b", sha256="s", size=1, board="Standard", licence="x", tier="commit")


def said(text: str, returncode: int = 0) -> Completed:
    return Completed(returncode, text.encode(), b"", False)


class StockTool:
    """What the stock CLI would print about a game whose positions are ``rows``.

    Position k holds k + 1 boards of 8x8 unless ``boards`` and ``size`` say otherwise.
    """

    name = "ftxi"

    def __init__(self, rows, *, perftest=None, verdict="", boards=None, size=(8, 8), stats_fields=True):
        self.rows = rows
        digits = "".join("1" if row[2] else "0" for row in rows)
        self.perftest = (perftest if perftest is not None else digits) + verdict
        self.boards = boards if boards is not None else [index + 1 for index in range(len(rows))]
        self.size = size
        self.stats_fields = stats_fields
        self.timeouts: set[str] = set()
        self.calls: list[tuple] = []

    def _answer(self, tool: str, completed: Completed) -> Completed:
        return Completed(None, b"", b"", True) if tool in self.timeouts else completed

    def stock_perftest(self, data, timeout_s):
        self.calls.append(("perftest",))
        return self._answer("perftest", said(self.perftest + "\n"))

    def stock_count(self, data, actions, cap, timeout_s):
        self.calls.append(("count", actions))
        return self._answer("count", said(json.dumps({"legal_actions": self.rows[actions][2]}) + "\n"))

    def stock_stats(self, data, actions, timeout_s):
        self.calls.append(("stats", actions))
        stats = {"side_to_move": self.rows[actions][0], "actions_played": actions}
        if self.stats_fields:
            stats.update(board_count=self.boards[actions], board_width=self.size[0], board_height=self.size[1])
        return self._answer("stats", said(json.dumps(stats) + "\n"))

    drawings: dict[int, str] = {}

    def stock_print(self, data, actions, timeout_s):
        self.calls.append(("print", actions))
        return self._answer("print", said(self.drawings.get(actions, "")))


# `5dtools print --at`, as ftxi draws a 4x4 "Very Small - Open" board on its 8x8 grid.
DRAWN = """State(present=1, player=0):
Multiverse present: T1w
lines range:(0,0)\tactive range:(0,0)
L0T1w  aka.(0T1)
################
################
################
################
n b r k ########
. . . p ########
P . . . ########
K R B N ########
"""
DRAWN_PLACEMENT = md5_hex("[nbrk/3p/P3/KRBN:0:1:w]")


def report(rows, status="ok", reason="", last_action=None, *, boards=None, size="8x8") -> Observation:
    """The driver's report: position k holds k + 1 boards (or ``boards[k]``), of ``size`` (None: no size given)."""
    counts = boards if boards is not None else [index + 1 for index in range(len(rows))]
    positions = [Position(index, *row, boards=counts[index]) for index, row in enumerate(rows)]
    # The driver's position event carries its timing and the boards' size; the harness keeps both as extras.
    extras = {index: {"ms": 0.5, **({"size": size} if size is not None else {})} for index in range(len(rows))}
    return Observation(source="ftxi", record="r/1", status=status, positions=positions, reason=reason,
                       actions_accepted=len(positions) - 1, cap=1000, last_action=last_action, extras=extras)


class FidelityTests(unittest.TestCase):
    def check(self, stock: StockTool, observation: Observation, *, full=True):
        with mock.patch.object(fidelity.corpus, "read", return_value=b"1. e3 / e6\n"):
            found = fidelity.check(stock, [RECORD], {("ftxi", "r/1"): observation}, None, 1000, timeout_s=5, full=full, jobs=1)
        return [finding.message for finding in found]

    def test_a_driver_that_says_what_the_stock_tool_says_passes(self):
        rows = [W, B, W, B, MATE]
        stock = StockTool(rows, verdict="\nTurn 3w: Checkmate")
        self.assertEqual(self.check(stock, report(rows)), [])
        # Every position was asked about, for its count and for the side to move.
        self.assertEqual(sorted(call for call in stock.calls if call[0] == "count"), [("count", index) for index in range(5)])
        self.assertEqual(sorted(call for call in stock.calls if call[0] == "stats"), [("stats", index) for index in range(5)])

    def test_a_sample_is_three_positions(self):
        rows = [W, B] * 5
        stock = StockTool(rows)
        self.assertEqual(self.check(stock, report(rows), full=False), [])
        self.assertEqual(sorted(call[1] for call in stock.calls if call[0] == "count"), [0, 5, 9])

    def test_each_kind_of_difference_is_reported(self):
        rows = [W, B, MATE]
        wrong_count = StockTool([W, ("black", False, 30, "none"), MATE])
        self.assertEqual(self.check(wrong_count, report(rows)), ["ftxi on r/1: position 1: driver counts 31, stock count says 30"])
        wrong_side = StockTool([W, ("white", False, 31, "none"), MATE])
        self.assertEqual(self.check(wrong_side, report(rows)),
                         ["ftxi on r/1: position 1: driver says black to move, stock stats says white"])
        wrong_digits = StockTool(rows, perftest="111")
        self.assertEqual(self.check(wrong_digits, report(rows)), ["ftxi on r/1: driver says 110, stock perftest says 111"])
        wrong_verdict = StockTool(rows, verdict="\nTurn 2w: Stalemate")
        self.assertEqual(self.check(wrong_verdict, report(rows)), ["ftxi on r/1: driver says checkmate, stock perftest says stalemate"])

    def test_the_boards_are_held_to_the_stock_stats_in_number_and_size(self):
        rows = [W, B, MATE]
        self.assertEqual(self.check(StockTool(rows, boards=[1, 3, 3]), report(rows)),
                         ["ftxi on r/1: position 1: driver holds 2 board(s), stock stats counts 3"])
        small = StockTool(rows, size=(5, 5))
        self.assertEqual(self.check(small, report(rows, size="5x5")), [])
        self.assertEqual(self.check(small, report(rows)),
                         [f"ftxi on r/1: position {index}: driver gives boards of 8x8, stock stats says 5x5" for index in range(3)])
        # Width before height: a 4-wide, 6-high board is "4x6", never "6x4".
        self.assertEqual(self.check(StockTool(rows, size=(4, 6)), report(rows, size="6x4"))[0],
                         "ftxi on r/1: position 0: driver gives boards of 6x4, stock stats says 4x6")

    def test_a_driver_that_does_not_say_what_it_set_up_is_a_finding(self):
        rows = [W, B]
        no_state = report(rows, boards=[None, 2])
        self.assertEqual(self.check(StockTool(rows), no_state),
                         ["ftxi on r/1: position 0: the driver gave no state event, so no boards to count; stock stats counts 1"])
        self.assertEqual(self.check(StockTool(rows), report(rows, size=None)),
                         [f"ftxi on r/1: position {index}: driver gives boards of no size, stock stats says 8x8" for index in range(2)])

    def test_a_stock_stats_without_the_boards_is_a_finding(self):
        rows = [W, B]
        messages = self.check(StockTool(rows, stats_fields=False), report(rows))
        self.assertEqual(messages, [f"ftxi on r/1: position {index}: stock `stats --at` gave no board count and size"
                                    for index in range(2)])

    def test_the_stock_replay_goes_on_where_the_driver_refuses_an_unsafe_action(self):
        # The driver refuses action 3 (it leaves a royal capturable); the stock
        # replay does not test for that and prints two more digits. Up to the
        # refusal the two agree, and that is what is compared.
        rows = [W, B, B_CHECK]
        refused = report(rows, status="rejected", reason="cannot_submit", last_action=False)
        self.assertEqual(self.check(StockTool(rows, perftest="11111"), refused), [])
        self.assertEqual(self.check(StockTool(rows, perftest="1011"), refused), ["ftxi on r/1: driver says 111, stock perftest says 1011"])

    def test_any_other_refusal_must_stop_the_stock_tool_at_the_same_place(self):
        rows = [W, B, W]
        unmatched = report(rows, status="rejected", reason="no_matching_move", last_action=False)
        self.assertEqual(self.check(StockTool(rows), unmatched), [])
        self.assertEqual(self.check(StockTool(rows, perftest="1111"), unmatched),
                         ["ftxi on r/1: driver says 111, stock perftest says 1111"])

    def test_a_stock_tool_that_does_not_answer_is_a_finding_not_a_skipped_check(self):
        rows = [W, B]
        for tool, expected in (("perftest", "stock perftest did not finish"), ("count", "stock `count --at` did not finish"),
                               ("stats", "stock `stats --at` gave no side to move")):
            stock = StockTool(rows)
            stock.timeouts.add(tool)
            messages = self.check(stock, report(rows))
            self.assertTrue(messages and all(expected in message for message in messages), (tool, messages))

    def test_the_boards_are_held_to_the_stock_drawing_of_them(self):
        rows = [W]
        stock = StockTool(rows, boards=[1], size=(4, 4))
        stock.drawings = {0: DRAWN}
        driver = report(rows, boards=[1], size="4x4")
        driver.positions[0] = dataclasses.replace(driver.positions[0], placement=DRAWN_PLACEMENT, pieces=10)
        self.assertEqual(self.check(stock, driver), [])
        self.assertIn(("print", 0), stock.calls)
        # Another piece, or another label for the same board - a leaked index, say - is a finding.
        for drawing in (DRAWN.replace("K R B N", "K R B ."), DRAWN.replace("aka.(0T1)", "aka.(-1T1)")):
            stock.drawings = {0: drawing}
            messages = self.check(stock, driver)
            self.assertEqual(len(messages), 1, messages)
            self.assertIn("the driver's boards are not the ones stock `print --at` draws", messages[0])
        stock.drawings = {0: "State(present=1, player=0):\n"}
        self.assertIn("could not be read as boards: it drew no board", self.check(stock, driver)[0])

    def test_the_drawing_is_read_as_a_state(self):
        self.assertEqual(fidelity.printed_placement(DRAWN, 4, 4), DRAWN_PLACEMENT)
        # Every square outside the board is drawn "#", and none inside it.
        for width, height in ((3, 4), (4, 3), (5, 4)):
            with self.assertRaises(ValueError):
                fidelity.printed_placement(DRAWN, width, height)
        even = DRAWN.replace("L0T1w  aka.(0T1)", "L-1T1w  aka.(-0T1)") + DRAWN.split("\n", 3)[3].replace(
            "L0T1w  aka.(0T1)", "L0T1w  aka.(+0T1)")
        self.assertEqual(fidelity.printed_placement(even, 4, 4),
                         md5_hex("[nbrk/3p/P3/KRBN:-0:1:w][nbrk/3p/P3/KRBN:+0:1:w]"))

    def test_only_a_complete_driver_report_is_held_against_the_stock_tool(self):
        rows = [W, B]
        for status in ("timeout", "unsupported", "error"):
            stock = StockTool(rows)
            self.assertEqual(self.check(stock, report(rows, status=status)), [])
            self.assertEqual(stock.calls, [])


if __name__ == "__main__":
    unittest.main()
