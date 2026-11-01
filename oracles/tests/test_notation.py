"""The notation repository's parser as a parse-only oracle (decision H26): the
adapter's refusals, the build's reading of the parser's table of boards, the
lock-file role and the registry, and - where it is built - what the driver
reports."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from oracle_harness import build, corpus, lockfile, paths
from oracle_harness.adapters import (
    ORACLE_SOURCES,
    PARSE_ONLY_SOURCES,
    PENDING_SOURCES,
    PRIMARY_SOURCES,
    RunOptions,
    make_source,
    oracle_of,
)
from oracle_harness.adapters.notation import NotationSource, would_misread
from oracle_harness.fingerprint import md5_hex
from oracle_harness.observations import SOFT_REJECTIONS
from oracle_harness.util import HarnessError, run

# Rows of the parser's table of named boards, as build.notation_boards() reads it.
BOARDS = {"STANDARD": "8x8", "STANDARD - TURN ZERO": "8x8", "VERY SMALL - OPEN": "4x4", "MISC - TIMELINE INVASION": "5x5"}
STANDARD_FEN = "[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"


def refusal(text: str) -> str | None:
    found = would_misread(text.encode(), BOARDS)
    return None if found is None else found[0]


class RefusalTests(unittest.TestCase):
    """What the adapter refuses before the parser runs."""

    def test_a_standard_game_is_read(self):
        self.assertIsNone(refusal('[Board "Standard"]\n[Mode "5D"]\n\n1. e4 / e5\n'))
        # No Board header: Standard, the notation's default, and the parser's.
        self.assertIsNone(refusal('[Mode "5D"]\n\n1. e4 / e5\n'))

    def test_a_board_the_parser_does_not_know_is_refused(self):
        # It would set up one empty board without a word.
        self.assertEqual(refusal('[Board "Misc - Small"]\n[Size "5x5"]\n\n1. c3\n'), "variant")
        self.assertEqual(refusal('[Board "Custom - Even"]\n[Size "5x5"]\n[k4/5/5/5/4K:-0:1:w]\n\n1. Ke2\n'), "variant")

    def test_a_name_is_matched_as_the_parser_matches_it(self):
        # It upper-cases the name, reads a header name in any case, and reads
        # Variant where there is no Board.
        self.assertIsNone(refusal('[board "standard - turn zero"]\n\n1. e3\n'))
        self.assertIsNone(refusal('[Variant "Standard - Turn Zero"]\n\n1. e3\n'))

    def test_a_named_layout_needs_the_size_it_is_drawn_for(self):
        # The parser sets every layout up on boards of the Size header's size, or 8x8.
        self.assertEqual(refusal('[Board "Very Small - Open"]\n\n1. Rb4\n'), "size")
        self.assertEqual(refusal('[Board "Very Small - Open"]\n[Size "5x5"]\n\n1. Rb4\n'), "size")
        self.assertIsNone(refusal('[Board "Very Small - Open"]\n[Size "4x4"]\n\n1. Rb4\n'))
        self.assertEqual(refusal('[Board "Standard"]\n[Size "5x5"]\n\n1. e4\n'), "size")
        # It splits the size on a lower-case "x" and nothing else.
        self.assertEqual(refusal('[Board "Standard"]\n[Size "8X8"]\n\n1. e4\n'), "size")
        self.assertIsNone(refusal('[Board "Standard"]\n[Size "8x8"]\n\n1. e4\n'))

    def test_5dfen_boards_are_read_only_under_a_custom_board(self):
        self.assertEqual(refusal(f'[Board "Standard"]\n{STANDARD_FEN}\n\n1. e4\n'), "fen")
        self.assertIsNone(refusal(f'[Board "custom"]\n{STANDARD_FEN}\n\n1. e4\n'))
        # notation/1.4, as the notation repository writes it.
        self.assertIsNone(refusal('[size "5x5"]\n[puzzle "mate-in-1"]\n[variant "custom"]\n[4k/5/5/5/K1R2:0:1:w]\n\n'
                                  '1. Kb2 / Ke4\n'))
        # A custom board and no board string: the parser would set up nothing.
        self.assertEqual(refusal('[Board "Custom"]\n[Size "5x5"]\n\n1. Kb2\n'), "fen")

    def test_headers_the_parser_does_not_read_are_refused(self):
        self.assertEqual(refusal('[Board "Standard"]\n[Timeline "even"]\n\n1. e4\n'), "timeline")
        self.assertEqual(refusal('[Board "Standard"]\n[Promotions "Q,R"]\n\n1. e4\n'), "promotions")

    def test_a_comment_never_closed_is_refused(self):
        # The parser's tokeniser loops forever on one.
        self.assertEqual(refusal('[Board "Standard"]\n\n1. e4 {unfinished / e5\n'), "notation")
        self.assertIsNone(refusal('[Board "Standard"]\n\n1. e4 {fine} / e5 {also { fine}\n'))


class BoardTableTests(unittest.TestCase):
    """build.notation_boards: the parser's table of named boards, read at build time."""

    def test_the_table_is_kept_by_upper_case_name(self):
        output = json.dumps({"VERY SMALL": "4x4", "STANDARD": "8x8"})
        self.assertEqual(build.notation_boards(output), {"STANDARD": "8x8", "VERY SMALL": "4x4"})

    def test_a_table_the_adapter_does_not_expect_stops_the_build(self):
        for output in ("not JSON", "[]", json.dumps({"VERY SMALL": "4x4"}), json.dumps({"STANDARD": "8x8", "Small": "5x5"}),
                       json.dumps({"STANDARD": "8 by 8"}), json.dumps({"STANDARD": ["8x8"]})):
            with self.assertRaises(HarnessError, msg=output):
                build.notation_boards(output)


_LOCK = """schema = 1
[repo.x]
role = "{role}"
{tier}
upstream = "a/b"
url = "https://example.invalid/a/b"
submodule = "x"
path = "oracles/third_party/x"
commit = "{sha1}"
tree = "{sha1}"
ref = "main"
committed = "2026-01-01"
licence = "MIT"
licence_file = "LICENSE"
licence_sha256 = "{sha256}"
summary = "s"
used_as = "u"
"""


class RoleTests(unittest.TestCase):
    """The lock file's roles: which pins the harness runs."""

    def load(self, role: str, tier: str) -> lockfile.Lock:
        text = _LOCK.format(role=role, tier=f'tier = "{tier}"' if tier else "", sha1="0" * 40, sha256="0" * 64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oracles.lock.toml"
            path.write_text(text, encoding="utf-8")
            return lockfile.load(path)

    def test_a_parse_oracle_is_run_and_needs_a_tier(self):
        lock = self.load("parse-oracle", "commit")
        self.assertEqual([pin.id for pin in lock.oracles()], ["x"])
        self.assertTrue(lock.pin("x").parse_only)
        with self.assertRaises(HarnessError):
            self.load("parse-oracle", "")

    def test_a_rules_oracle_is_not_parse_only(self):
        self.assertFalse(self.load("oracle", "nightly").pin("x").parse_only)

    def test_a_corpus_source_is_never_run_and_takes_no_tier(self):
        self.assertEqual(self.load("corpus-source", "").oracles(), [])
        with self.assertRaises(HarnessError):
            self.load("corpus-source", "nightly")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.lock = lockfile.load()

    def test_the_parser_is_a_parse_only_oracle_of_its_own(self):
        pin = self.lock.pin("notation")
        self.assertEqual((pin.role, pin.parse_only, pin.tier), ("parse-oracle", True, "commit"))
        # One source, so its replay - how many actions it read - is compared as an implementation's.
        self.assertIn("notation", ORACLE_SOURCES)
        self.assertIn("notation", PRIMARY_SOURCES)
        self.assertEqual(set(PARSE_ONLY_SOURCES), {pin.id for pin in self.lock.pins.values() if pin.parse_only})

    def test_every_pin_the_harness_runs_has_a_recipe_and_a_source(self):
        run_by_harness = {pin.id for pin in self.lock.oracles()}
        # A pin the harness runs has a source the gate compares - or one wired and not yet enabled (Kairos, H52).
        self.assertEqual(run_by_harness, {oracle_of(name) for name in ORACLE_SOURCES + PENDING_SOURCES})
        self.assertEqual(set(build.buildable(self.lock)), run_by_harness)
        self.assertEqual(build.recipe_tools("notation"), ("node",))

    def test_it_answers_no_rules_question(self):
        source = make_source("notation", self.lock)
        self.assertIsInstance(source, NotationSource)
        self.assertFalse({"in_check", "legal_actions", "terminal", "legality"} & set(source.answers))
        # It explains how it read each action; it lists no legal actions.
        self.assertEqual(source.diagnostic_args(RunOptions(explain=True, actions_at=3)), ["--explain"])


class LiveTests(unittest.TestCase):
    """Run only where the parser is exported (python oracles/harness.py build --oracle notation)."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.source = make_source("notation", cls.lock)
        cls.records = {record.id: record for record in corpus.load()}

    def setUp(self):
        reason = self.source.available()
        if reason is not None:
            self.skipTest(reason)

    def observe(self, record_id: str):
        return self.source.observe(record_id, corpus.read(self.records[record_id], self.lock), RunOptions(timeout_s=60))

    def ask(self, text: str, **options):
        """What the source says about a record that is not in the corpus."""
        return self.source.observe("test/handwritten", text.encode(), RunOptions(timeout_s=60, **options))

    def driver(self, text: str, *args: str) -> list[dict]:
        """The driver's own output, with no adapter in front of it."""
        root = build.load(self.lock.pin("notation")).artifact("root")
        done = run([shutil.which("node"), paths.DRIVERS / "notation" / "driver.js", root, *args],
                   stdin=text.encode(), timeout=60)
        return [json.loads(line) for line in done.out_text().splitlines()]

    def test_the_notation_records(self):
        outcomes = {}
        for record_id in self.records:
            if record_id.startswith("notation/"):
                observation = self.observe(record_id)
                outcomes[record_id] = (observation.status, observation.reason, observation.actions_accepted)
        self.assertEqual(outcomes, {
            "notation/1.1": ("ok", "", 15), "notation/1.2": ("ok", "", 22), "notation/1.4": ("ok", "", 4),
            "notation/Bg2": ("ok", "", 25), "notation/aivsai": ("ok", "", 54), "notation/e3": ("ok", "", 15),
            # Board "Misc - Small" is not in the parser's table.
            "notation/game": ("unsupported", "variant", 0),
            "notation/game-2": ("ok", "", 45), "notation/game-3": ("ok", "", 18),
            "notation/invasion": ("ok", "", 22), "notation/marauders": ("ok", "", 2),
        })

    def test_position_zero_of_a_standard_game_is_the_standard_board(self):
        observation = self.observe("notation/1.1")
        first = observation.positions[0]
        # The notation specification's own Standard example, its unmoved markers taken
        # out: the parser keeps no record of them, and says so (no fingerprint).
        self.assertEqual((first.fingerprint, first.placement, first.boards, first.pieces),
                         (None, md5_hex(STANDARD_FEN.replace("*", "")), 1, 32))
        self.assertEqual([position.side for position in observation.positions[:4]], ["white", "black", "white", "black"])
        # After the last action nothing in the record says who moves next.
        self.assertIsNone(observation.positions[-1].side)
        self.assertTrue(all(position.in_check is None and position.legal_actions is None and position.terminal is None
                            for position in observation.positions))

    def test_it_holds_no_action_illegal(self):
        # local/ignores-check: every rules oracle refuses action 4, which leaves
        # Black in check. The parser plays it as written: its acceptance is no verdict.
        observation = self.observe("local/ignores-check")
        self.assertEqual((observation.status, observation.actions_accepted), ("ok", 4))

    def test_a_move_it_cannot_apply_is_a_soft_rejection(self):
        cases = {
            "no_matching_move": '[Board "Standard"]\n\n1. e5\n',
            "ambiguous_move": '[Board "custom"]\n[Size "5x5"]\n[k4/5/5/5/N1N1K:0:1:w]\n\n1. Nb3\n',
            # A pawn reaching the last rank with no promotion written.
            "notation_mismatch": '[Board "custom"]\n[Size "5x5"]\n[k4/3P1/5/5/4K:0:1:w]\n\n1. d5\n',
        }
        for reason, text in cases.items():
            observation = self.ask(text)
            self.assertEqual((observation.status, observation.reason, observation.actions_accepted, len(observation.positions)),
                             ("rejected", reason, 0, 1), reason)
            self.assertIn(reason, SOFT_REJECTIONS)

    def test_a_record_it_cannot_tokenise_yields_no_position(self):
        # It tokenises the whole record before setting anything up.
        observation = self.ask('[Board "Standard"]\n\n1. e4 / e5\n2w. Nf3\n')
        self.assertEqual((observation.status, observation.reason, observation.positions), ("unsupported", "notation", []))

    def test_a_move_it_cannot_place_stops_the_reading_after_what_it_read(self):
        # No super-physical coordinates on a board of two timelines.
        observation = self.ask('[Board "Misc - Timeline Invasion"]\n[Size "5x5"]\n\n1. a2\n')
        self.assertEqual((observation.status, observation.reason, len(observation.positions)), ("unsupported", "notation", 1))

    def test_what_the_parser_prints_never_reaches_the_protocol(self):
        # It prints a board after a move followed by an "@" comment, and its errors before it exits.
        events = self.driver('[Board "Standard"]\n\n1. e4 {@} / e5\n2. Qxe8\n', "--board", "Standard")
        self.assertEqual([event["event"] for event in events],
                         ["start", "state", "position", "state", "position", "state", "position", "end"])
        self.assertEqual((events[-1]["status"], events[-1]["actions_accepted"]), ("rejected", 2))

    def test_the_driver_checks_the_set_up_itself(self):
        # What the adapter refuses before it runs, the driver refuses again, from what the parser set up.
        def reason(text: str, board: str) -> str:
            events = self.driver(text, "--board", board)
            self.assertEqual([event["event"] for event in events], ["end"])
            return events[0]["reason"]

        self.assertEqual(reason('[Board "Very Small - Open"]\n\n1. Rb4\n', "Very Small - Open"), "size")
        self.assertEqual(reason('[Board "Misc - Small"]\n[Size "5x5"]\n\n1. c3\n', "Misc - Small"), "variant")
        self.assertEqual(reason('[Board "Standard"]\n[k7/8/8/8/8/8/8/K7:0:1:w]\n\n1. Kb2\n', "Standard"), "fen")
        self.assertEqual(reason('[Board "custom"]\n[Size "5x5"]\n[k4/5/5/5/4K:0:1:w]\n{[k4/5/5/5/4K:1:1:w]}\n\n1. Ke2\n',
                                "custom"), "fen")
        # Two Board headers: the harness reads the first, the parser the last.
        self.assertEqual(reason('[Board "Standard"]\n[Board "Standard - Turn Zero"]\n\n1. e3\n', "Standard"), "variant")
        self.assertEqual(reason('[Board "Standard"]\n\n1. e4 {unfinished / e5\n', "Standard"), "notation")

    def test_the_signed_zero_defect_is_shown_not_fixed(self):
        # parse_5dfen stores a "-0" board string as a signed zero, which the parser writes "0".
        alone = self.driver('[Board "custom"]\n[Size "5x5"]\n[k4/5/5/5/4K:-0:1:w]\n\n1. (0T1)Ke1e2\n', "--board", "custom")
        self.assertEqual([event["boards"] for event in alone if event["event"] == "state"][0], ["0:1:w:k4/5/5/5/4K"])
        # Beside "+0" it would be an odd and an even game at once: refused, under a
        # reason of its own - golden files record the reason, not the detail, and
        # must not take the defect for a set-up the parser merely could not do.
        both = self.driver('[Board "custom"]\n[Size "5x5"]\n[k4/5/5/5/4K:-0:1:w]\n[k4/5/5/5/4K:+0:1:w]\n\n'
                           '1. (+0T1)Ke1e2\n', "--board", "custom")
        self.assertEqual([(event["event"], event.get("status"), event.get("reason")) for event in both],
                         [("end", "unsupported", "signed_zero")])
        unset = self.driver('[Board "custom"]\n[k4/5/5/5/4K:0:1:w]\n\n1. Ke2\n', "--board", "custom")  # 5x5 on 8x8
        self.assertEqual([(event["event"], event.get("status"), event.get("reason")) for event in unset],
                         [("end", "unsupported", "setup")])

    def test_a_turn_zero_and_an_even_layout_are_labelled_as_the_notation_writes_them(self):
        turn_zero = self.driver('[Board "Standard - Turn Zero"]\n\n1. e3\n', "--board", "Standard - Turn Zero")
        self.assertEqual([event["boards"] for event in turn_zero if event["event"] == "state"][0],
                         ["0:0:b:rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "0:1:w:rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"])
        invasion = self.driver('[Board "Misc - Timeline Invasion"]\n[Size "5x5"]\n\n1. (-0T1)a2\n',
                               "--board", "Misc - Timeline Invasion")
        self.assertEqual([board.split(":")[0] for board in [e for e in invasion if e["event"] == "state"][0]["boards"]],
                         ["-0", "+0"])

    def test_explain_shows_the_moves_as_the_parser_resolved_them(self):
        observation = self.ask('[Board "Standard"]\n\n1. e4 / d5\n2. exd5\n', explain=True)
        explained = {event["index"]: event for event in observation.diagnostics if event["event"] == "explain"}
        self.assertEqual(sorted(explained), [0, 1, 2, 3])
        self.assertEqual(explained[0]["layout"], "named")
        self.assertEqual(explained[3]["moves"], [{"from": {"L": "0", "T": 2, "c": "w", "square": "e4", "piece": "P"},
                                                  "to": {"L": "0", "T": 2, "c": "w", "square": "d5", "piece": "p"},
                                                  "promotion": None}])
        self.assertEqual(len(observation.positions), 4)


if __name__ == "__main__":
    unittest.main()
