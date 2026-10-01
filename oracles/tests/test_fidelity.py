"""The check that the harness's ftxi driver says what ftxi's own tools say,
exercised with a stand-in for those tools."""

import json
import unittest
from unittest import mock

from oracle_harness import fidelity
from oracle_harness.corpus import Record
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
    """What the stock CLI would print about a game whose positions are ``rows``."""

    name = "ftxi"

    def __init__(self, rows, *, perftest=None, verdict=""):
        self.rows = rows
        digits = "".join("1" if row[2] else "0" for row in rows)
        self.perftest = (perftest if perftest is not None else digits) + verdict
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
        return self._answer("stats", said(json.dumps({"side_to_move": self.rows[actions][0], "actions_played": actions}) + "\n"))


def report(rows, status="ok", reason="", last_action=None) -> Observation:
    positions = [Position(index, *row) for index, row in enumerate(rows)]
    return Observation(source="ftxi", record="r/1", status=status, positions=positions, reason=reason,
                       actions_accepted=len(positions) - 1, cap=1000, last_action=last_action)


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

    def test_only_a_complete_driver_report_is_held_against_the_stock_tool(self):
        rows = [W, B]
        for status in ("timeout", "unsupported", "error"):
            stock = StockTool(rows)
            self.assertEqual(self.check(stock, report(rows, status=status)), [])
            self.assertEqual(stock.calls, [])


if __name__ == "__main__":
    unittest.main()
