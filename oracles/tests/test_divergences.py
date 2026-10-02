import dataclasses
import tempfile
import unittest
from pathlib import Path

from oracle_harness import corpus, divergences, lockfile, snapshot
from oracle_harness.divergences import Divergence
from oracle_harness.observations import Disagreement, Observation, Position
from oracle_harness.util import HarnessError
from support import golden_files_predate_this_schema

RULE = "Rulebook, section on check: a royal attacked on any board after a pass is in check."
SPEC = "5dpgn specification, section on jumps: the capture marker on a jump is optional."


def codes(findings):
    return sorted({finding.code for finding in findings})


class DivergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = corpus.load()
        cls.record = next(record for record in cls.records if record.id == "cwmtt/standard")
        cls.entry = Divergence(
            id="KD-0001", oracle="chessin5d", pin=cls.lock.pin("chessin5d").commit,
            record=cls.record.id, record_sha256=cls.record.sha256, kind="in_check", index=13,
            engine_value="True", oracle_value="False", rulebook=RULE,
            upstream="https://gitlab.com/5d-chess/5d-chess-js/-/issues/1", added="2026-10-01",
        )

    def verify(self, **changes):
        return divergences.verify([dataclasses.replace(self.entry, **changes)], self.lock, self.records)

    def test_a_complete_entry_passes(self):
        self.assertEqual(self.verify(), [])
        self.assertEqual(self.verify(oracle="cwmtt:naive", pin=self.lock.pin("cwmtt").commit, kind="legal_actions"), [])
        self.assertEqual(self.verify(upstream="unreported: upstream is archived and takes no issues"), [])

    def test_an_entry_without_a_citation_is_not_an_entry(self):
        for citation in ("", "   ", "TBD", "todo", "?", "n/a", "see rules"):
            self.assertEqual(codes(self.verify(rulebook=citation)), ["divergence-citation"], citation)

    def test_unread_move_text_is_settled_by_the_notation_specification(self):
        soft = dict(kind="legality", index=17, engine_value="accepted", oracle_value="rejected:no_matching_move",
                    oracle="ftxi", pin=self.lock.pin("ftxi").commit, rulebook="", notation=SPEC)
        self.assertEqual(self.verify(**soft), [])
        self.assertEqual(self.verify(**{**soft, "oracle_value": "rejected:unrecognised"}), [])
        self.assertEqual(self.verify(**{**soft, "kind": "length", "engine_value": "20 actions", "oracle_value": "16 actions"}), [])
        # It may equally be a rule the oracle gets wrong: then it cites the rulebook like any other.
        self.assertEqual(self.verify(**{**soft, "rulebook": RULE, "notation": ""}), [])
        # Not both, and not a placeholder.
        self.assertEqual(codes(self.verify(**{**soft, "rulebook": RULE})), ["divergence-citation"])
        self.assertEqual(codes(self.verify(**{**soft, "notation": "TBD"})), ["divergence-citation"])

    def test_move_text_the_engine_could_not_read_is_settled_by_the_notation_specification(self):
        # The engine, in either mode, refuses as "unreadable" text an oracle played - an action's, or at index 0 the
        # record's set-up: whether that text is valid is the specification's to say, and only it can excuse that.
        unread = dict(kind="legality", index=17, engine_value="rejected:unreadable", oracle_value="accepted",
                      oracle="ftxi", pin=self.lock.pin("ftxi").commit, rulebook="", notation=SPEC)
        self.assertEqual(self.verify(**unread), [])
        self.assertEqual(self.verify(**{**unread, "index": 0}), [])
        found = self.verify(**{**unread, "rulebook": RULE, "notation": ""})
        self.assertEqual(codes(found), ["divergence-citation"])
        self.assertIn("cite it with notation", found[0].message)

    def test_nothing_else_is_settled_by_the_notation_specification(self):
        # A verdict on a position, or an action the oracle holds illegal, is a matter of the rules.
        self.assertEqual(codes(self.verify(rulebook="", notation=SPEC)), ["divergence-citation"])
        hard = dict(kind="legality", index=17, engine_value="accepted", oracle_value="rejected:cannot_submit",
                    oracle="ftxi", pin=self.lock.pin("ftxi").commit, rulebook="", notation=SPEC)
        self.assertEqual(codes(self.verify(**hard)), ["divergence-citation"])
        engine_refuses = {**hard, "engine_value": "rejected:illegal_move", "oracle_value": "accepted"}
        self.assertEqual(codes(self.verify(**engine_refuses)), ["divergence-citation"])

    def test_a_parse_divergence_is_settled_by_the_notation_specification(self):
        placement = dict(kind="placement", index=4, engine_value="a" * 32 + " (1 board, 32 pieces)",
                         oracle_value="b" * 32 + " (1 board, 32 pieces)", rulebook="", notation=SPEC)
        self.assertEqual(self.verify(**placement), [])
        # Two readings of the move text: never the rulebook's to settle.
        self.assertEqual(codes(self.verify(**{**placement, "rulebook": RULE, "notation": ""})), ["divergence-citation"])
        self.assertEqual(codes(self.verify(**{**placement, "notation": ""})), ["divergence-citation"])
        self.assertEqual(codes(self.verify(**{**placement, "notation": "TBD"})), ["divergence-citation"])
        # Which pieces carry the unmoved marker is the notation's; what having moved means, the rulebook's.
        unmoved = {**placement, "kind": "unmoved"}
        self.assertEqual(self.verify(**unmoved), [])
        self.assertEqual(self.verify(**{**unmoved, "rulebook": RULE, "notation": ""}), [])
        self.assertEqual(codes(self.verify(**{**unmoved, "rulebook": RULE})), ["divergence-citation"])

    def test_a_parse_divergence_may_name_any_search(self):
        # Each search says what it set up, and each is compared: an entry is per search, like a verdict's.
        naive = dict(kind="placement", index=4, engine_value="a" * 32, oracle_value="b" * 32, rulebook="",
                     notation=SPEC, oracle="ftxi:naive", pin=self.lock.pin("ftxi").commit)
        self.assertEqual(self.verify(**naive), [])

    def test_how_a_record_is_replayed_belongs_to_the_implementation(self):
        # Legality is compared through the primary source only, so an entry for
        # a second search would excuse nothing and then be reported stale.
        naive = dict(kind="legality", index=17, engine_value="accepted", oracle_value="rejected:illegal_move",
                     oracle="ftxi:naive", pin=self.lock.pin("ftxi").commit)
        self.assertEqual(codes(self.verify(**naive)), ["divergence-oracle"])
        self.assertEqual(self.verify(**{**naive, "oracle": "ftxi"}), [])

    def test_an_entry_is_bound_to_the_oracle_pin_it_was_verified_at(self):
        self.assertEqual(codes(self.verify(pin="0" * 40)), ["divergence-pin"])

    def test_an_entry_is_bound_to_the_record_content(self):
        self.assertEqual(codes(self.verify(record_sha256="0" * 64)), ["divergence-record"])
        self.assertEqual(codes(self.verify(record="cwmtt/no-such-record")), ["divergence-record"])

    def test_field_formats(self):
        self.assertEqual(codes(self.verify(id="divergence one")), ["divergence-id"])
        self.assertEqual(codes(self.verify(oracle="official-client")), ["divergence-oracle"])
        self.assertEqual(codes(self.verify(kind="vibes")), ["divergence-kind"])
        self.assertEqual(codes(self.verify(index=-1)), ["divergence-index"])
        self.assertEqual(codes(self.verify(engine_value=True)), ["divergence-value"])
        self.assertEqual(codes(self.verify(oracle_value="True")), ["divergence-value"])
        self.assertEqual(codes(self.verify(upstream="will report later")), ["divergence-upstream"])
        self.assertEqual(codes(self.verify(upstream="unreported:")), ["divergence-upstream"])
        self.assertEqual(codes(self.verify(added="yesterday")), ["divergence-date"])

    def test_duplicates(self):
        twin = dataclasses.replace(self.entry, id="KD-0002")
        self.assertEqual(codes(divergences.verify([self.entry, twin], self.lock, self.records)), ["divergence-duplicate"])
        self.assertEqual(codes(divergences.verify([self.entry, self.entry], self.lock, self.records)), ["divergence-duplicate"])


class GoldenCrossCheckTests(unittest.TestCase):
    """An entry says what an oracle answers. The golden files say so too, and they are the record."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = corpus.load()
        cls.record = next(record for record in cls.records if record.id == "cwmtt/standard")

    def entry(self, **changes):
        values = dict(
            id="KD-0001", oracle="chessin5d", pin=self.lock.pin("chessin5d").commit, record=self.record.id,
            record_sha256=self.record.sha256, kind="in_check", index=1, engine_value="True", oracle_value="False",
            rulebook=RULE, upstream="https://example.com/1", added="2026-10-01")
        values.update(changes)
        return Divergence(**values)

    @staticmethod
    def recorded(rows, status="ok", reason=""):
        positions = [Position(index, *row) for index, row in enumerate(rows)]
        report = Observation(source="chessin5d", record="cwmtt/standard", status=status, positions=positions,
                             reason=reason, actions_accepted=max(len(positions) - 1, 0), cap=1000)
        return lambda source, record_id: report if (source, record_id) == ("chessin5d", "cwmtt/standard") else None

    def verify(self, entry, recorded):
        return divergences.verify([entry], self.lock, self.records, recorded)

    def test_an_entry_that_quotes_the_oracle_correctly_passes(self):
        quiet = ("white", False, None, "none")
        self.assertEqual(self.verify(self.entry(), self.recorded([quiet, quiet])), [])
        legality = self.entry(kind="legality", index=2, engine_value="accepted", oracle_value="rejected:cannot_submit")
        self.assertEqual(self.verify(legality, self.recorded([quiet, quiet], "rejected", "cannot_submit")), [])

    def test_an_entry_that_misquotes_the_oracle_is_refused(self):
        check = ("white", True, None, "none")
        found = self.verify(self.entry(), self.recorded([check, check]))
        self.assertEqual(codes(found), ["divergence-golden"])
        self.assertIn("answers False; the golden file has it answering True", found[0].message)

    def test_an_entry_about_something_the_oracle_never_said_is_refused(self):
        quiet = ("white", False, None, "none")
        silent = ("white", None, None, "none")
        for recorded in (self.recorded([quiet]), self.recorded([quiet, silent]), lambda source, record_id: None):
            self.assertEqual(codes(self.verify(self.entry(), recorded)), ["divergence-golden"])

    def test_an_entry_about_what_an_oracle_set_up_is_checked_too(self):
        positions = [Position(0, "white", False, None, "none", fingerprint="1" * 32, placement="2" * 32, boards=1, pieces=32)]
        report = Observation(source="chessin5d", record="cwmtt/standard", status="ok", positions=positions,
                             actions_accepted=0, cap=1000)

        def recorded(source, record_id):
            return report if (source, record_id) == ("chessin5d", "cwmtt/standard") else None

        said = "2" * 32 + " (1 board, 32 pieces)"
        entry = self.entry(kind="placement", index=0, engine_value="3" * 32 + " (1 board, 32 pieces)", oracle_value=said,
                           rulebook="", notation=SPEC)
        self.assertEqual(self.verify(entry, recorded), [])
        unmoved = dataclasses.replace(entry, kind="unmoved", oracle_value="1" * 32 + " (1 board, 32 pieces)")
        self.assertEqual(self.verify(unmoved, recorded), [])
        misquoted = dataclasses.replace(entry, oracle_value="2" * 32)  # without the counts the gate prints
        self.assertEqual(codes(self.verify(misquoted, recorded)), ["divergence-golden"])

    def test_the_repository_golden_files_are_what_an_entry_is_checked_against(self):
        stale = golden_files_predate_this_schema("chessin5d")
        if stale:
            self.skipTest(stale)
        recorded = snapshot.take(["chessin5d"]).recorded
        said = recorded("chessin5d", "cwmtt/standard")
        self.assertIsNotNone(said)
        final = len(said.positions) - 1
        true_entry = self.entry(kind="terminal", index=final, engine_value="stalemate", oracle_value="checkmate")
        self.assertEqual(self.verify(true_entry, recorded), [])
        false_entry = self.entry(kind="terminal", index=final, engine_value="checkmate", oracle_value="stalemate")
        self.assertEqual(codes(self.verify(false_entry, recorded)), ["divergence-golden"])
        self.assertIsNone(recorded("chessin5d", "no/such-record"))


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.entry = Divergence(
            id="KD-0001", oracle="chessin5d", pin="p", record="r/1", record_sha256="s", kind="in_check", index=13,
            engine_value="True", oracle_value="False", rulebook="x" * 30, upstream="https://example.com/1", added="2026-10-01",
        )
        self.occurrence = Disagreement("r/1", "engine", "chessin5d", "in_check", 13, "True", "False")

    def test_a_recorded_divergence_is_excused(self):
        unexplained, stale = divergences.apply([self.entry], [self.occurrence], {"KD-0001"})
        self.assertEqual((unexplained, stale), ([], []))

    def test_anything_else_is_new(self):
        elsewhere = dataclasses.replace(self.occurrence, index=14)
        other_oracle = dataclasses.replace(self.occurrence, b="ftxi")
        other_kind = dataclasses.replace(self.occurrence, kind="terminal")
        unexplained, _ = divergences.apply([self.entry], [self.occurrence, elsewhere, other_oracle, other_kind], {"KD-0001"})
        self.assertEqual(unexplained, [elsewhere, other_oracle, other_kind])

    def test_an_entry_does_not_hide_a_regression_at_the_same_place(self):
        # Same oracle, same position - but now it is the engine that changed its answer.
        regressed = dataclasses.replace(self.occurrence, a_value="False", b_value="True")
        oracle_moved = dataclasses.replace(self.occurrence, b_value="unknown")
        for changed in (regressed, oracle_moved):
            unexplained, stale = divergences.apply([self.entry], [changed], {"KD-0001"})
            self.assertEqual(unexplained, [changed])
            self.assertEqual(stale, [self.entry])

    def test_an_entry_that_no_longer_occurs_is_stale(self):
        _, stale = divergences.apply([self.entry], [], {"KD-0001"})
        self.assertEqual(stale, [self.entry])

    def test_an_entry_is_not_stale_when_its_observation_was_not_looked_at(self):
        _, stale = divergences.apply([self.entry], [], set())
        self.assertEqual(stale, [])


class TelltaleTests(unittest.TestCase):
    """H34: every recorded oracle error is a copy detector. An engine that reproduces one was probably copied."""

    def entry(self, **changes):
        values = dict(id="KD-0001", oracle="chessin5d", pin="p", record="r/1", record_sha256="s", kind="in_check",
                      index=1, engine_value="True", oracle_value="False", rulebook="x" * 30,
                      upstream="https://example.com/1", added="2026-10-01")
        values.update(changes)
        return Divergence(**values)

    @staticmethod
    def report(rows, status="ok", reason=""):
        positions = [Position(index, *row) for index, row in enumerate(rows)]
        return Observation(source="engine", record="r/1", status=status, positions=positions, reason=reason,
                           actions_accepted=max(len(positions) - 1, 0), cap=1000)

    def test_an_engine_that_gives_the_recorded_error_is_a_telltale(self):
        copied = self.report([("white", False, 20, "none"), ("black", False, 20, "none")])
        self.assertEqual(divergences.telltales([self.entry()], {"r/1": copied}), [(self.entry(), "False")])
        right = self.report([("white", False, 20, "none"), ("black", True, 20, "none")])
        self.assertEqual(divergences.telltales([self.entry()], {"r/1": right}), [])
        # An engine that did not reach the place, or did not run the record, says nothing either way.
        self.assertEqual(divergences.telltales([self.entry()], {"r/1": self.report([("white", False, 20, "none")])}), [])
        self.assertEqual(divergences.telltales([self.entry()], {}), [])

    def test_every_kind_is_a_telltale_parse_kinds_and_replays_included(self):
        placed = self.report([("white", False, 20, "none")])
        placed.positions = [dataclasses.replace(placed.positions[0], placement="b" * 32, fingerprint="c" * 32,
                                                boards=1, pieces=32)]
        state = "b" * 32 + " (1 board, 32 pieces)"
        placement = self.entry(kind="placement", index=0, engine_value="a" * 32 + " (1 board, 32 pieces)",
                               oracle_value=state)
        self.assertEqual(len(divergences.telltales([placement], {"r/1": placed})), 1)
        refused = self.report([("white", False, 20, "none")], status="rejected", reason="cannot_submit")
        legality = self.entry(kind="legality", index=1, engine_value="accepted", oracle_value="rejected:cannot_submit")
        self.assertEqual(divergences.telltales([legality], {"r/1": refused}), [(legality, "rejected:cannot_submit")])

    @staticmethod
    def oracle(source, rows, status="ok", reason=""):
        positions = [Position(index, *row) for index, row in enumerate(rows)]
        report = Observation(source=source, record="r/1", status=status, positions=positions, reason=reason,
                             actions_accepted=max(len(positions) - 1, 0), cap=1000)
        return lambda name, record_id: report if (name, record_id) == (source, "r/1") else None

    def test_agreeing_with_the_error_by_the_comparisons_rule_is_reproducing_it(self):
        # ftxi could not match action 2's text (a soft refusal); an engine whose parser fails on the same text -
        # refusing it as unreadable, or calling it illegal - reproduces that, whatever reason it gives.
        quiet = ("white", False, 20, "none")
        entry = self.entry(oracle="ftxi", kind="legality", index=2, engine_value="accepted",
                           oracle_value="rejected:no_matching_move")
        ftxi = self.oracle("ftxi", [quiet, quiet], status="rejected", reason="no_matching_move")
        for reason in ("illegal_move", "unreadable", "cannot_submit"):
            copied = self.report([quiet, quiet], status="rejected", reason=reason)
            self.assertEqual(divergences.telltales([entry], {"r/1": copied}, ftxi), [(entry, f"rejected:{reason}")])
            self.assertEqual(len(divergences.telltales([entry], {"r/1": copied})), 1)  # no recorded report at hand
        self.assertEqual(divergences.telltales([entry], {"r/1": self.report([quiet] * 3)}, ftxi), [])
        # A hard refusal reproduced for another reason is reproduced all the same.
        cannot = self.entry(oracle="cwmtt", kind="legality", index=2, engine_value="accepted",
                            oracle_value="rejected:cannot_submit")
        cwmtt = self.oracle("cwmtt", [(None, None, 20, "none")] * 2, status="rejected", reason="cannot_submit")
        illegal = self.report([quiet, quiet], status="rejected", reason="illegal_move")
        self.assertEqual(divergences.telltales([cannot], {"r/1": illegal}, cwmtt), [(cannot, "rejected:illegal_move")])

    def test_a_verdict_that_only_says_no_action_is_reproduced_by_either_end(self):
        # cwmtt wrongly finds no action at position 1; an engine that calls it checkmate or stalemate agrees.
        rows = [(None, None, 20, "none"), (None, None, 0, "no_action")]
        cwmtt = self.oracle("cwmtt", rows)
        count = self.entry(oracle="cwmtt", kind="legal_actions", engine_value="25", oracle_value="0")
        verdict = self.entry(id="KD-0002", oracle="cwmtt", kind="terminal", engine_value="none", oracle_value="no_action")
        copied = self.report([("white", False, 20, "none"), ("black", True, 0, "checkmate")])
        self.assertEqual(divergences.telltales([count, verdict], {"r/1": copied}, cwmtt),
                         [(count, "0"), (verdict, "checkmate")])
        right = self.report([("white", False, 20, "none"), ("black", False, 25, "none")])
        self.assertEqual(divergences.telltales([count, verdict], {"r/1": right}, cwmtt), [])
        # A count at the cap agrees with any as large.
        capped = self.entry(oracle="cwmtt", kind="legal_actions", engine_value="998", oracle_value=">=1000")
        wide = self.oracle("cwmtt", [(None, None, 20, "none"), (None, None, 1000, "none")])
        at_cap = self.report([("white", False, 20, "none"), ("black", False, 1000, "none")])
        self.assertEqual(divergences.telltales([capped], {"r/1": at_cap}, wide), [(capped, ">=1000")])

    def test_only_while_the_record_still_says_what_the_entry_says(self):
        copied = self.report([("white", False, 20, "none"), ("black", False, 20, "none")])
        moved = self.oracle("chessin5d", [("white", False, None, "none"), ("black", True, None, "none")])
        self.assertEqual(divergences.telltales([self.entry()], {"r/1": copied}, moved), [])

    def test_in_parse_mode_playing_an_action_reproduces_no_oracles_acceptance_of_it(self):
        # Chess in 5D wrongly plays action 2. An engine in parse mode plays what it reads: that is no verdict.
        quiet = ("white", None, None, None)
        entry = self.entry(kind="legality", index=2, engine_value="rejected:illegal_move", oracle_value="accepted")
        c5 = self.oracle("chessin5d", [("white", False, None, "none")] * 3)
        reader = self.report([quiet, quiet, quiet])
        self.assertEqual(divergences.telltales([entry], {"r/1": reader}, c5), [(entry, "accepted")])  # full mode
        self.assertEqual(divergences.telltales([entry], {"r/1": reader}, c5, judges_actions=False), [])
        # Its refusing an action is a statement, in parse mode as in full mode.
        cwmtt = self.entry(oracle="cwmtt", kind="legality", index=2, engine_value="accepted",
                           oracle_value="rejected:illegal_move")
        refuses = self.oracle("cwmtt", [(None, None, 20, "none")] * 2, status="rejected", reason="illegal_move")
        unread = self.report([quiet, quiet], status="rejected", reason="unreadable")
        self.assertEqual(divergences.telltales([cwmtt], {"r/1": unread}, refuses, judges_actions=False),
                         [(cwmtt, "rejected:unreadable")])

    def test_a_telltale_is_never_called_stale(self):
        # The disagreement no longer occurs because the engine now says what the oracle wrongly says:
        # "remove it" would be exactly the wrong advice.
        _, stale = divergences.apply([self.entry()], [], {"KD-0001"}, telltales={"KD-0001"})
        self.assertEqual(stale, [])
        _, stale = divergences.apply([self.entry()], [], {"KD-0001"})
        self.assertEqual([item.id for item in stale], ["KD-0001"])


class LookedAtTests(unittest.TestCase):
    def setUp(self):
        self.entry = Divergence(
            id="KD-0001", oracle="chessin5d", pin="p", record="r/1", record_sha256="s", kind="in_check", index=1,
            engine_value="True", oracle_value="False", rulebook="x" * 30, upstream="https://example.com/1", added="2026-10-01",
        )

    @staticmethod
    def report(rows, status="ok"):
        positions = [Position(index, *row) for index, row in enumerate(rows)]
        return Observation(source="s", record="r/1", status=status, positions=positions, actions_accepted=len(positions) - 1)

    def test_both_reports_must_answer_that_question_at_that_position(self):
        known = ("white", False, 20, "none")
        silent = ("white", None, 20, "none")
        full = self.report([known, known])
        self.assertTrue(divergences.looked_at(self.entry, full, full))
        self.assertFalse(divergences.looked_at(self.entry, full, self.report([known], status="timeout")))
        self.assertFalse(divergences.looked_at(self.entry, full, self.report([known, silent])))
        self.assertFalse(divergences.looked_at(self.entry, self.report([known], status="error"), full))

    def test_legality_needs_two_complete_replays(self):
        legality = dataclasses.replace(self.entry, kind="legality")
        known = ("white", False, 20, "none")
        done = self.report([known, known])
        refused = self.report([known], status="rejected")
        self.assertTrue(divergences.looked_at(legality, done, refused))
        self.assertFalse(divergences.looked_at(legality, done, self.report([known], status="timeout")))
        self.assertFalse(divergences.looked_at(legality, done, self.report([], status="unsupported")))

    def test_in_parse_mode_an_action_the_engine_played_is_looked_at_only_beside_a_failure_to_read_it(self):
        # The engine in parse mode plays what it reads: beside an oracle's hard refusal or its acceptance, its
        # playing an action is no verdict, and an entry about it is not looked at (so never stale).
        known = ("white", None, None, None)
        played = self.report([known, known])
        entry = dataclasses.replace(self.entry, kind="legality", index=1)
        illegal = self.report([known], status="rejected")
        illegal.reason = "illegal_move"
        unread = self.report([known], status="rejected")
        unread.reason = "no_matching_move"
        self.assertTrue(divergences.looked_at(entry, played, illegal))
        self.assertFalse(divergences.looked_at(entry, played, illegal, judges_actions=False))
        self.assertFalse(divergences.looked_at(entry, played, played, judges_actions=False))
        self.assertTrue(divergences.looked_at(entry, played, unread, judges_actions=False))
        # Its refusal is a statement: looked at.
        refused = self.report([known], status="rejected")
        refused.reason = "unreadable"
        self.assertTrue(divergences.looked_at(entry, refused, played, judges_actions=False))

    @staticmethod
    def placed(report, *placements, marked=False):
        report.positions = [dataclasses.replace(position, placement=f"{placement:032x}", boards=1, pieces=32,
                                                fingerprint=f"{placement + 100:032x}" if marked else None)
                            for position, placement in zip(report.positions, placements, strict=True)]
        return report

    def test_a_parse_entry_needs_both_reports_to_say_what_they_set_up(self):
        known = ("white", False, 20, "none")
        placement = dataclasses.replace(self.entry, kind="placement")
        both = self.placed(self.report([known, known]), 1, 2)
        self.assertTrue(divergences.looked_at(placement, both, self.placed(self.report([known, known]), 1, 3)))
        self.assertFalse(divergences.looked_at(placement, both, self.report([known, known])))
        unmoved = dataclasses.replace(self.entry, kind="unmoved")
        self.assertFalse(divergences.looked_at(unmoved, both, both))  # placements only: no markers to compare
        marked = self.placed(self.report([known, known]), 1, 2, marked=True)
        self.assertTrue(divergences.looked_at(unmoved, marked, marked))

    def test_nothing_after_two_reports_part_is_looked_at(self):
        # They part at position 1: an entry about check there, or about an action
        # played after it, was not looked at - only the parting itself was.
        known = ("white", False, 20, "none")
        engine = self.placed(self.report([known, known, known]), 1, 2, 3)
        oracle = self.placed(self.report([known, known, known]), 1, 9, 9)
        self.assertFalse(divergences.looked_at(self.entry, engine, oracle))
        self.assertTrue(divergences.looked_at(dataclasses.replace(self.entry, kind="placement"), engine, oracle))
        self.assertFalse(divergences.looked_at(dataclasses.replace(self.entry, kind="placement", index=2), engine, oracle))
        self.assertFalse(divergences.looked_at(dataclasses.replace(self.entry, kind="unmoved"), engine, oracle))
        self.assertFalse(divergences.looked_at(dataclasses.replace(self.entry, kind="length", index=2), engine, oracle))
        self.assertTrue(divergences.looked_at(dataclasses.replace(self.entry, index=0), engine, oracle))


class FileTests(unittest.TestCase):
    def test_the_repository_file_loads_and_is_valid(self):
        entries = divergences.load()
        self.assertEqual(divergences.verify(entries, lockfile.load(), corpus.load(), snapshot.take().recorded), [])

    def test_the_file_can_be_read_from_the_bytes_already_in_hand(self):
        # A gate run parses the very bytes it hashed (snapshot.py): no second read to race.
        body = "\n".join(('schema = 1', '[[divergence]]', 'id = "KD-0001"', 'oracle = "ftxi"', 'pin = "p"',
                          'record = "r/1"', 'record_sha256 = "s"', 'kind = "in_check"', 'index = 1',
                          'engine_value = "True"', 'oracle_value = "False"', 'rulebook = "x"',
                          'upstream = "unreported: test"', 'added = "2026-10-01"', '')).encode()
        self.assertEqual([entry.id for entry in divergences.load(data=body)], ["KD-0001"])
        with self.assertRaises(HarnessError):
            divergences.load(data=b"schema = [")

    def test_a_missing_field_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "known-divergences.toml"
            path.write_text('schema = 1\n[[divergence]]\nid = "KD-0001"\n', encoding="utf-8")
            with self.assertRaises(HarnessError):
                divergences.load(path)

    def test_either_citation_may_be_given_in_the_file(self):
        body = ('id = "KD-0001"\noracle = "ftxi"\npin = "p"\nrecord = "r/1"\nrecord_sha256 = "s"\nkind = "legality"\n'
                'index = 3\nengine_value = "accepted"\noracle_value = "rejected:no_matching_move"\n'
                'upstream = "unreported: test"\nadded = "2026-10-01"\n')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "known-divergences.toml"
            path.write_text(f'schema = 1\n[[divergence]]\n{body}notation = "{SPEC}"\n', encoding="utf-8")
            entry = divergences.load(path)[0]
            self.assertEqual((entry.rulebook, entry.notation), ("", SPEC))
            path.write_text(f'schema = 1\n[[divergence]]\n{body}rulebook = "{RULE}"\n', encoding="utf-8")
            entry = divergences.load(path)[0]
            self.assertEqual((entry.rulebook, entry.notation), (RULE, ""))


if __name__ == "__main__":
    unittest.main()
