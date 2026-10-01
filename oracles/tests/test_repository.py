"""Checks against this repository itself: its configuration, and (when the
oracles have been built) the oracles' answers to questions with known answers."""

import json
import unittest

from oracle_harness import build, config, corpus, gitutil, lockfile
from oracle_harness.adapters import ORACLE_SOURCES, PARSE_ONLY_SOURCES, RunOptions, make_source
from oracle_harness.util import run
from support import golden_files_predate_this_schema


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.cfg = config.load()

    def test_tiers_follow_the_design(self):
        # Design spec 11.0: ftxi on every commit; cwmtt and Chess in 5D nightly
        # and before any generator release or proof-channel enablement. The
        # notation repository's parser (parse-only, H26) is fast: every tier.
        self.assertEqual(self.cfg.tier("commit").sources, ("ftxi", "notation"))
        for name in ("nightly", "release"):
            oracles = {source.split(":")[0] for source in self.cfg.tier(name).sources}
            self.assertEqual(oracles, {"ftxi", "cwmtt", "chessin5d", "notation"}, name)
        self.assertTrue(self.cfg.tier("release").require_engine)
        self.assertFalse(self.cfg.tier("commit").require_engine)

    def test_every_source_is_run_by_some_tier(self):
        run = {source for tier in self.cfg.tiers.values() for source in tier.sources}
        self.assertEqual(run, set(ORACLE_SOURCES))

    def test_each_oracle_runs_no_earlier_than_its_pinned_tier(self):
        for source in self.cfg.tier("commit").sources:
            self.assertEqual(self.lock.pin(source.split(":")[0]).tier, "commit", source)

    def test_no_engine_is_configured_yet(self):
        # Flip this together with gate.toml when the rules engine gains its adapter.
        self.assertEqual(self.cfg.engine_command, ())

    def test_the_tier_that_can_enable_the_proof_channel_runs_everything(self):
        # Every search of every oracle, over every record, and it insists on an engine.
        release = self.cfg.tier("release")
        self.assertEqual(set(release.sources), set(ORACLE_SOURCES))
        self.assertEqual(set(release.record_tiers), set(corpus.TIERS))
        self.assertTrue(release.full_strength)
        self.assertTrue(release.require_engine)
        self.assertFalse(self.cfg.tier("commit").full_strength)

    def test_no_tier_gates_a_ruleset_on_an_observable_nobody_answered(self):
        for tier in self.cfg.tiers.values():
            self.assertGreaterEqual(tier.min_checked, 0.5, tier.name)

    def test_every_pinned_repository_names_what_the_engine_must_not_depend_on(self):
        for pin in self.lock.pins.values():
            self.assertTrue(pin.dependency_names, pin.id)


def _built(oracle: str) -> bool:
    lock = lockfile.load()
    pin = lock.pin(oracle)
    return gitutil.is_checkout(pin.checkout) and build.load(pin) is not None


class LiveOracleTests(unittest.TestCase):
    """Run only where the oracles are built; CI builds them before the tests."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.cfg = config.load()
        cls.records = {record.id: record for record in corpus.load()}

    def observe(self, source_name: str, record_id: str):
        source = make_source(source_name, self.lock)
        if source.available() is not None:
            self.skipTest(source.available())
        data = corpus.read(self.records[record_id], self.lock)
        return source.observe(record_id, data, RunOptions(cap=self.cfg.cap, timeout_s=120, mate_budget_ms=20000))

    # cwmtt/standard is a sixteen-action game ending in checkmate. cwmtt's own
    # repository records the expected counts (tests/results/Correctness.txt).
    STANDARD_COUNTS = [20, 20, 32, 23, 46, 33, 49, 44, 64, 65, 29, 1000, 44, 1000, 37, 1000, 0]

    def test_the_count_sources_reproduce_the_published_counts(self):
        for name in ("ftxi", "ftxi:naive", "cwmtt", "cwmtt:naive"):
            observation = self.observe(name, "cwmtt/standard")
            self.assertEqual(observation.status, "ok", name)
            self.assertEqual([position.legal_actions for position in observation.positions], self.STANDARD_COUNTS, name)

    def test_the_verdict_sources_call_the_final_position_checkmate(self):
        for name in ("ftxi", "chessin5d"):
            observation = self.observe(name, "cwmtt/standard")
            final = observation.positions[-1]
            self.assertEqual((final.side, final.in_check, final.terminal), ("white", True, "checkmate"), name)

    def test_the_first_position_is_whites_with_twenty_actions(self):
        first = self.observe("ftxi", "cwmtt/standard").positions[0]
        self.assertEqual((first.side, first.in_check, first.legal_actions, first.terminal), ("white", False, 20, "none"))

    def test_an_oracle_is_never_asked_about_a_game_it_would_misread(self):
        # "Misc - Small" is a board none of the three knows; two of them would
        # silently replay it on the Standard layout, and the notation's parser
        # on a single empty board.
        for name in ("cwmtt", "chessin5d", "notation"):
            observation = self.observe(name, "notation/game")
            self.assertEqual((observation.status, observation.reason, observation.positions), ("unsupported", "variant", []), name)
        # A record that sets up its own position with 5DFEN boards.
        for name in ("cwmtt", "chessin5d", "notation"):
            observation = self.observe(name, "ftxi/niceAndrey")
            self.assertEqual((observation.status, observation.reason, observation.positions), ("unsupported", "fen", []), name)

    def ask(self, source_name: str, record: str):
        """What a source says about a record that is not in the corpus."""
        source = make_source(source_name, self.lock)
        if source.available() is not None:
            self.skipTest(source.available())
        return source.observe("test/handwritten", record.encode(), RunOptions(cap=self.cfg.cap, timeout_s=120, mate_budget_ms=20000))

    def test_every_oracle_refuses_an_action_that_leaves_a_royal_capturable(self):
        # local/ignores-check: Black is in check after 2. Qh5 and plays a6 anyway.
        # An oracle that "accepts" that has not checked the action, and its
        # acceptance of any other action would mean nothing. (ftxi's own replay
        # tools do accept it; the harness driver asks ftxi's legality test
        # before submitting.) A parse-only source judges no action, and is not asked.
        for name in ORACLE_SOURCES:
            if name in PARSE_ONLY_SOURCES:
                continue
            observation = self.observe(name, "local/ignores-check")
            self.assertEqual((observation.status, observation.reason, observation.actions_accepted, observation.last_action),
                             ("rejected", "cannot_submit", 3, None if name.startswith("cwmtt") else True), name)
        for name in ("ftxi", "chessin5d"):
            self.assertEqual(self.observe(name, "local/ignores-check").positions[3].in_check, True, name)
        # The refusal is of that action, not of the game: the same moves with more after them stop at the same place.
        longer = '[Board "Standard"]\n[Mode "5D"]\n\n1. e4 / f5\n2. Qh5 / a6\n3. a3 / a5\n'
        for name in ("ftxi", "cwmtt", "chessin5d"):
            observation = self.ask(name, longer)
            self.assertEqual((observation.status, observation.reason, observation.actions_accepted), ("rejected", "cannot_submit", 3), name)

    def test_the_two_oracles_that_name_the_end_of_a_game_tell_stalemate_from_checkmate(self):
        # The corpus the pinned projects ship holds no stalemate at all.
        expected = {"local/stalemate": (False, "stalemate"), "local/stalemate-with-a-past": (False, "stalemate"),
                    "local/checkmate": (True, "checkmate")}
        for record_id, (in_check, terminal) in expected.items():
            for name in ("ftxi", "ftxi:naive", "chessin5d"):
                observation = self.observe(name, record_id)
                final = observation.positions[-1]
                self.assertEqual((observation.status, final.side, final.in_check, final.terminal),
                                 ("ok", "black", in_check, terminal), (record_id, name))
            counts = [position.legal_actions for position in self.observe("ftxi", record_id).positions]
            self.assertEqual(counts[-1], 0, record_id)
            self.assertTrue(all(counts[:-1]), record_id)
            # cwmtt reads no 5DFEN position, and is not asked.
            self.assertEqual(self.observe("cwmtt", record_id).reason, "variant", record_id)

    def test_a_legal_reply_to_the_same_check_is_accepted(self):
        answered = '[Board "Standard"]\n[Mode "5D"]\n\n1. e4 / f5\n2. Qh5 / g6\n'
        for name in ("ftxi", "cwmtt", "chessin5d"):
            observation = self.ask(name, answered)
            self.assertEqual((observation.status, observation.actions_accepted), ("ok", 4), name)

    def test_a_board_header_an_oracle_would_not_read_is_refused_not_misread(self):
        # The same Turn Zero game under a header each of the two lenient readers
        # would skip - and then replay on the Standard layout.
        lowercase = '[board "Standard - Turn Zero"]\n[Mode "5D"]\n\n1. e3 / e6\n'
        self.assertEqual((self.ask("cwmtt", lowercase).status, self.ask("cwmtt", lowercase).reason), ("unsupported", "variant"))
        spaced = '[ Board "Standard - Turn Zero" ]\n[Mode "5D"]\n\n1. e3 / e6\n'
        self.assertEqual((self.ask("chessin5d", spaced).status, self.ask("chessin5d", spaced).reason), ("unsupported", "variant"))
        # Written the usual way, both read it.
        usual = '[Board "Standard - Turn Zero"]\n[Mode "5D"]\n\n1. e3 / e6\n'
        for name in ("cwmtt", "chessin5d"):
            self.assertEqual(self.ask(name, usual).status, "ok", name)

    def test_a_game_named_by_a_variant_header_is_played_as_that_variant_or_not_at_all(self):
        # ftxi reads [Variant] in place of a missing [Board]; the other two do not
        # read it at all, and would play Standard.
        record = '[Variant "Standard - Turn Zero"]\n[Mode "5D"]\n\n1. e3 / e6\n'
        turn_zero = self.ask("ftxi", record)
        standard = self.ask("ftxi", '[Board "Standard"]\n[Mode "5D"]\n\n1. e3 / e6\n')
        self.assertEqual(turn_zero.status, "ok")
        self.assertNotEqual([p.legal_actions for p in turn_zero.positions], [p.legal_actions for p in standard.positions])
        for name in ("cwmtt", "chessin5d"):
            observation = self.ask(name, record)
            self.assertEqual((observation.status, observation.reason), ("unsupported", "variant"), name)

    def test_a_custom_position_the_library_sets_up_only_in_part_is_refused(self):
        # 5d-chess-js reads a board string only when it has a line to itself.
        same_line = '[Board "Custom"]\n[Size "5x5"] [3k1/5/5/1R3/K4:0:1:w]\n\n1. (0T1)Ka1a2 / (0T1)Kd5d4\n'
        observation = self.ask("chessin5d", same_line)
        self.assertEqual((observation.status, observation.reason), ("unsupported", "fen"))
        self.assertIn("set up 0 board(s)", observation.detail)

    def test_ftxi_is_not_asked_about_a_record_ending_in_a_variation(self):
        record = '[Board "Standard"]\n[Mode "5D"]\n\n1. e3 / e6\n2. d3 / d6\n(2. c3 / c6)\n'
        observation = self.ask("ftxi", record)
        self.assertEqual((observation.status, observation.reason, observation.positions), ("unsupported", "notation", []))

    def test_a_result_token_is_not_a_move(self):
        def ending(result: str) -> str:
            return f'[Board "Standard"]\n[Mode "5D"]\n\n1. e3 / e6\n2. d3 / d6 {result}\n'

        for result in ("1-0", "0-1", "1/2-1/2"):
            for name in ("ftxi", "cwmtt", "chessin5d"):
                observation = self.ask(name, ending(result))
                self.assertEqual((observation.status, observation.actions_accepted), ("ok", 4), (name, result))
        # PGN's "game not finished" marker. Two oracles read past it; ftxi's
        # parser does not take it, and refuses the whole record rather than guess.
        for name in ("cwmtt", "chessin5d"):
            observation = self.ask(name, ending("*"))
            self.assertEqual((observation.status, observation.actions_accepted), ("ok", 4), name)
        refused = self.ask("ftxi", ending("*"))
        self.assertEqual((refused.status, refused.reason, refused.positions), ("unsupported", "parse_error", []))

    def test_every_check_disagreement_is_at_an_optional_timeline(self):
        # oracles/README.md explains the check disagreements between ftxi and
        # Chess in 5D by timelines the side to move may play on but need not.
        # This keeps that explanation honest when a pin or the corpus moves.
        if not _built("ftxi"):
            self.skipTest("ftxi is not built")
        for source in ("ftxi", "chessin5d"):
            stale = golden_files_predate_this_schema(source)
            if stale:
                self.skipTest(stale)
        from oracle_harness import golden

        tools = build.load(self.lock.pin("ftxi")).artifact("tools")
        ftxi, chessin5d = golden.load("ftxi"), golden.load("chessin5d")
        disagreements = [
            (record_id, mine.index, mine.in_check, theirs.in_check)
            for record_id, report in ftxi.observations.items() if record_id in chessin5d.observations
            for mine, theirs in zip(report.positions, chessin5d.observations[record_id].positions, strict=False)
            if None not in (mine.in_check, theirs.in_check) and mine.in_check != theirs.in_check
        ]
        self.assertTrue(disagreements, "no check disagreement is left: update oracles/README.md")
        for record_id, index, mine, theirs in disagreements:
            self.assertEqual((mine, theirs), (True, False), (record_id, index))
            turn = "0" if index == 0 else f"{(index + 1) // 2}{'w' if index % 2 else 'b'}"
            data = corpus.read(self.records[record_id], self.lock)
            done = run([tools, "stats", "--at", turn, "--format", "json"], stdin=data, timeout=60)
            optional = json.loads(done.out_text())["timelines"]["optional"]
            self.assertGreater(optional, 0, (record_id, index))

    def test_cwmtt_layouts_at_the_pin(self):
        if not _built("cwmtt"):
            self.skipTest("cwmtt is not built")
        # A pin bump that changes this list changes which records cwmtt may be asked about.
        self.assertEqual(build.load(self.lock.pin("cwmtt")).info["layouts"], [
            "Standard", "Standard - Half Reflected", "Standard - Reversed Royalty", "Standard - Turn Zero",
            "Focused - Just Pawns", "Focused - Just Brawns", "Focused - Just Kings", "Very Small - Open",
            "Misc - Timeline Battleground",
        ])

    def test_chessin5d_dependencies_at_the_pin(self):
        if not _built("chessin5d"):
            self.skipTest("chessin5d is not built")
        dependencies = build.load(self.lock.pin("chessin5d")).info["dependencies"]
        self.assertEqual({name: item["version"] for name, item in dependencies.items()},
                         {"blueimp-md5": "2.18.0", "module-alias": "2.2.2", "present": "1.0.0"})


if __name__ == "__main__":
    unittest.main()
