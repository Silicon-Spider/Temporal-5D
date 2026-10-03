"""cwmtt's records in the official client's export form, converted to 5dpgn (export_form.py; CHECKPOINT 3).

The expectations are written by hand from the pinned notation specification
(oracles/third_party/notation/README.md): "Export or minimal notation" says what
an export move is, "Moves", "Castling", "Jumps", "Promotions" and "Omission" what
the same move is in 5dpgn.
"""

import dataclasses
import unittest
from unittest import mock

from oracle_harness import corpus, export_form, gitutil, golden, lockfile, runner
from oracle_harness.adapters import ORACLE_SOURCES, RunOptions, make_source
from oracle_harness.adapters.engine import EngineSource
from oracle_harness.corpus import Record
from oracle_harness.observations import Observation, validate
from oracle_harness.util import HarnessError, sha256_bytes
from support import Sandbox, git
from test_golden import FakeSource, GoldenCase

MANY2 = "cwmtt/many2"

# Tokens of cwmtt/many2 as the client wrote them, and as the specification writes
# the same move. A move on one board loses its second board ("Moves": physical
# moves, preceded by their board; "Omission": the piece's square in full); a jump
# is already in the specification's form ("Jumps") and is left as it is; a turn
# number glued to its first move stays glued ("Turns").
MANY2_TOKENS = [
    # moves on one board
    ("1.(0T1)Pe2(0T1)e3", "1.(0T1)Pe2e3"),  # white pawn, after the turn number
    ("(0T1)Pe7(0T1)e6", "(0T1)Pe7e6"),  # black pawn
    ("(1T1)Pe7(1T1)e6", "(1T1)Pe7e6"),  # on a positive timeline
    ("(3T1)Nb8(3T1)c6", "(3T1)Nb8c6"),  # knight
    ("(4T3)Qd8(4T3)f6", "(4T3)Qd8f6"),  # queen
    ("5.(0T4)Bf1(0T4)b5", "5.(0T4)Bf1b5"),  # bishop
    ("(-6T2)Qd1(-6T2)h5", "(-6T2)Qd1h5"),  # on a negative timeline
    ("7.(-10T2)Qf3(-10T2)a3", "7.(-10T2)Qf3a3"),  # a two-digit timeline
    ("(13T3)Qf6(13T3)f2", "(13T3)Qf6f2"),  # a capture: the export marks none, and none is added
    ("(12T4)Ke1(12T4)f2", "(12T4)Ke1f2"),  # white king
    ("(19T3)Ke8(19T3)f7", "(19T3)Ke8f7"),  # black king
    ("(-19T4)Ph2(-19T4)h4", "(-19T4)Ph2h4"),  # a pawn's double step
    ("20.(-16T3)Pg2(-16T3)f3", "20.(-16T3)Pg2f3"),  # a pawn's diagonal capture
    ("(22T3)Qd7(22T3)e6", "(22T3)Qd7e6"),  # the record's last move
    # jumps, already in the specification's form
    ("2.(0T2)Ng1>>(0T1)g3", "2.(0T2)Ng1>>(0T1)g3"),  # branching, after the turn number
    ("(1T2)Ke8>>(0T2)e7", "(1T2)Ke8>>(0T2)e7"),  # a king's branching jump
    ("(1T3)Pe2>>(0T3)e2", "(1T3)Pe2>>(0T3)e2"),  # a pawn's
    ("(-1T3)Ke1>>(-1T2)e2", "(-1T3)Ke1>>(-1T2)e2"),  # back in time on its own timeline
    ("(-5T2)Ng3>(-3T3)g3", "(-5T2)Ng3>(-3T3)g3"),  # non-branching
    ("9.(14T4)Bf1>(13T4)f2", "9.(14T4)Bf1>(13T4)f2"),  # non-branching, after the turn number
    ("(6T4)Qh4>(14T4)h4", "(6T4)Qh4>(14T4)h4"),  # non-branching, eight timelines away
    ("30.(17T5)Qa4>>(17T2)d7", "30.(17T5)Qa4>>(17T2)d7"),  # branching, three turns back
    # the separator of the two players' moves
    ("/", "/"),
]


def checked_out(lock, pin_id: str) -> bool:
    pin = lock.pin(pin_id)
    return gitutil.is_checkout(pin.checkout) and gitutil.has_commit(pin.checkout, pin.commit)


class ConversionTests(unittest.TestCase):
    """The conversion as text: no corpus, no oracle."""

    def convert(self, text: str) -> export_form.Conversion:
        return export_form.convert(text.encode())

    def assertConverts(self, before: str, after: str, moves: int = 1):
        conversion = self.convert(before)
        self.assertEqual((conversion.text.decode(), conversion.moves, conversion.refused), (after, moves, ""), before)

    def assertRefused(self, text: str, quoted: str, why: str):
        conversion = self.convert(text)
        self.assertTrue(conversion.export_form, text)
        self.assertIn(f"'{quoted}'", conversion.refused)
        self.assertIn(why, conversion.refused)
        # Never half converted: the record comes back as it was, and no move counts as rewritten.
        self.assertEqual((conversion.text, conversion.moves), (text.encode(), 0), text)

    def test_a_move_on_one_board_loses_its_second_board(self):
        self.assertConverts("1. (0T1)Pe2(0T1)e4 / (0T1)Pe7(0T1)e5\n", "1. (0T1)Pe2e4 / (0T1)Pe7e5\n", 2)
        self.assertConverts("1. (0T1)e2(0T1)e4", "1. (0T1)e2e4")  # "the piece's letter (optional)": none is added
        self.assertConverts("1. (0T1)RQd1(0T1)d2", "1. (0T1)RQd1d2")  # a two-letter piece ("Moves": RQ, PR)
        self.assertConverts("1. (L0 T1)Pe2(L0 T1)e4", "1. (L0 T1)Pe2e4")  # the long form of a board ("Coordinates")
        self.assertConverts("1. (+1T2)Ne1(1T2)f3", "1. (+1T2)Ne1f3")  # "+1" and "1" are one timeline
        self.assertConverts("1. (0T10)Nb1(0T10)c3", "1. (0T10)Nb1c3")
        self.assertConverts("1. (0T1)Ra10(0T1)a12", "1. (0T1)Ra10a12")  # a board taller than nine ranks

    def test_castling_is_the_kings_move_as_the_specification_writes_it(self):
        # "Export moves": castling "is encoded as the king moving two pieces: (0T6)Ke1(0T6)g1";
        # "Castling": K<file_from><rank_from><file_to><rank_to>, of which O-O is a shortcut.
        self.assertConverts("6. (0T6)Ke1(0T6)g1 / (0T6)Ke8(0T6)c8", "6. (0T6)Ke1g1 / (0T6)Ke8c8", 2)

    def test_what_follows_a_move_is_copied_as_it_stands(self):
        # "Promotions" (=Q before the check mark), the check marks, the assessments and
        # annotation tokens the specification writes after a move, and a comment.
        self.assertConverts("9. (0T9)Pe7(0T9)e8=Q+", "9. (0T9)Pe7e8=Q+")
        self.assertConverts("9. (0T9)Qd1(0T9)h5#!", "9. (0T9)Qd1h5#!")
        self.assertConverts("9. (0T9)Qd1(0T9)h5*", "9. (0T9)Qd1h5*")
        self.assertConverts("9. (0T9)Qd1(0T9)h5 (~T3)", "9. (0T9)Qd1h5 (~T3)")
        self.assertConverts("9. (0T9)Qd1(0T9)h5{a comment}", "9. (0T9)Qd1h5{a comment}")
        self.assertConverts("9. (0T9)Qd1(0T9)h5/ (0T9)Ke8(0T9)d8", "9. (0T9)Qd1h5/ (0T9)Ke8d8", 2)

    def test_a_record_without_the_export_form_comes_back_unchanged(self):
        for text in ("1. e3 / e6\n2. (0T2)Ng1>>(0T1)g3 / (1T1)e6\n",  # 5dpgn, jumps included
                     "1. (0T1)Ng1>x(0T1)g3",  # a capturing jump
                     "1. (0T1)a2a3 / (0T1)Pe7e6",  # the long form the converted text uses
                     "1. (-1T4)Nc3>>x(0T2)c3+~ (>L2) (~T1)",  # the specification's "fully-detailled" move
                     "1b. (0T10)Qg6+\n2w. (0T11)Kh1 / (0T11)Bb7+\n",  # turns written 1b. and 2w.
                     ""):
            conversion = self.convert(text)
            self.assertFalse(conversion.export_form, text)
            self.assertEqual((conversion.text, conversion.moves, conversion.refused), (text.encode(), 0, ""), text)

    def test_headers_board_strings_and_comments_are_never_looked_inside(self):
        text = ('[Event "(0T1)Pe2(0T1)e4"]\n[Board "custom"]\n[4k/5/5/5/K1R2:0:1:w]\n'
                "1. Kb2 {not (0T1)Pe2(0T1)e4, nor (0T1)Ng1(1T1)g3} / Ke4\n")
        conversion = self.convert(text)
        self.assertFalse(conversion.export_form)
        self.assertEqual(conversion.text, text.encode())
        # Beside a move that is converted, they are copied as they stand.
        mixed = text + "2. (0T2)Re1(0T2)e5 {(0T1)Pe2(0T1)e4}\n"
        self.assertConverts(mixed, text + "2. (0T2)Re1e5 {(0T1)Pe2(0T1)e4}\n")

    def test_a_move_between_two_boards_without_a_jump_marker_is_refused(self):
        # "Jumps": > or >> - which one is a question about the position.
        why = "moves from one board to another without '>' or '>>'"
        self.assertRefused("1. (0T1)Ng1(1T1)g3", "(0T1)Ng1(1T1)g3", why)
        self.assertRefused("1. (0T2)Ng1(0T1)g3", "(0T2)Ng1(0T1)g3", why)
        # -0, +0 and 0 are three timelines ("Notes": even-numbered starting boards).
        self.assertRefused("1. (-0T1)Pe7(+0T1)e6", "(-0T1)Pe7(+0T1)e6", why)
        self.assertRefused("1. (0T1)Pe7(+0T1)e6", "(0T1)Pe7(+0T1)e6", why)

    def test_one_inexpressible_move_refuses_the_whole_record(self):
        text = "1. (0T1)Pe2(0T1)e3 / (0T1)Pe7(0T1)e6\n2. (0T2)Ng1(1T1)g3 / (0T2)Pd7(0T2)d6\n"
        self.assertRefused(text, "(0T2)Ng1(1T1)g3", "without '>' or '>>'")

    def test_the_first_inexpressible_move_is_the_one_named(self):
        text = "1. (0T1)Pe2x(0T1)d3 / (0T1)Ng1(1T1)g3"
        self.assertRefused(text, "(0T1)Pe2x(0T1)d3", "is not an export move")

    def test_text_with_the_export_forms_hallmark_that_is_no_export_move_is_refused(self):
        why = "is not an export move"
        self.assertRefused("1. (0T1)Pe2x(0T1)d3", "(0T1)Pe2x(0T1)d3", why)  # a capture mark: not in "Export moves"
        self.assertRefused("1. (0T1)Pe2(0T1)e3)", "(0T1)Pe2(0T1)e3)", why)  # a variation's bracket glued on
        self.assertRefused("1. (0T1)Pe2(0T1)e3e4", "(0T1)Pe2(0T1)e3e4", why)
        self.assertRefused("1. (0T1)e4(0T1)Pe2(0T1)e3", "(0T1)e4(0T1)Pe2(0T1)e3", why)  # glued to another move
        self.assertRefused("1. (0T1)Pe2(0T1)e3(0T1)e4", "(0T1)Pe2(0T1)e3(0T1)e4", why)
        self.assertRefused("1. x(0T1)Pe2(0T1)e3", "x(0T1)Pe2(0T1)e3", why)

    def test_the_export_forms_turns_are_refused(self):
        # "Export turns": w. and b. with no number; the specification's turns are numbered ("Turns").
        self.assertRefused("w. (0T1)d2(0T1)d4\nb. (0T1)d7(0T1)d6\n", "w.", "is a turn of the export form")

    def test_the_conversion_is_byte_stable(self):
        text = "[Board \"Standard\"]\n1.(0T1)Pe2(0T1)e3 / (0T1)Pe7(0T1)e6 \n2.(0T2)Ng1>>(0T1)g3 / (1T1)Pe7(1T1)e6 \n"
        first, second = self.convert(text), self.convert(text)
        self.assertEqual(first, second)
        # Converting the converted text changes nothing: nothing in it is in the export form.
        again = export_form.convert(first.text)
        self.assertEqual((again.text, again.moves, again.export_form), (first.text, 0, False))
        # Every byte but the second boards is kept, in order - the trailing spaces included.
        self.assertEqual(first.text.decode(), text.replace("(0T1)e3", "e3").replace("(0T1)e6", "e6")
                         .replace("(1T1)e6", "e6"))


class CorpusTests(unittest.TestCase):
    """The repository's own corpus: what every source is fed."""

    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.records = {record.id: record for record in corpus.load()}

    def stored(self, record: Record) -> bytes:
        """The record's stored bytes with CRLF folded: what every source was fed before the conversion existed."""
        return corpus._stored_bytes(record, self.lock).replace(b"\r\n", b"\n")

    def readable(self) -> list[Record]:
        found = [record for record in self.records.values()
                 if record.source == corpus.LOCAL or checked_out(self.lock, record.source)]
        if len(found) != len(self.records):
            self.skipTest("not every pinned source is checked out (run: python oracles/harness.py sync)")
        return found

    def many2(self) -> tuple[Record, bytes, export_form.Conversion]:
        if not checked_out(self.lock, "cwmtt"):
            self.skipTest("cwmtt is not checked out (run: python oracles/harness.py sync)")
        record = self.records[MANY2]
        text = self.stored(record)
        return record, text, export_form.convert(text)

    def test_many2_token_by_token(self):
        _, text, conversion = self.many2()
        self.assertEqual(conversion.refused, "")
        before, after = text.decode().split(), conversion.text.decode().split()
        # Nothing is added or taken away but boards inside a move: the tokens line up one for one.
        self.assertEqual(len(before), len(after))
        pairs = list(zip(before, after, strict=True))
        for token, expected in MANY2_TOKENS:
            self.assertIn(token, before, f"{token} is not a token of {MANY2}")
            self.assertEqual({converted for original, converted in pairs if original == token}, {expected}, token)
        # Every token that changed lost exactly its second board, as the first board wrote it.
        changed = [(original, converted) for original, converted in pairs if original != converted]
        self.assertEqual(len(changed), conversion.moves)
        for original, converted in changed:
            board = original[original.index("("):original.index(")") + 1]
            self.assertEqual(original.count(board), 2, original)
            self.assertEqual(converted, original[:original.rindex(board)] + original[original.rindex(board) + len(board):])
        # 204 moves: 153 on one board, rewritten; 51 jumps, kept; and 31 turns, so 31 separators.
        jumps = [original for original, converted in pairs if original == converted and ">" in original]
        self.assertEqual((conversion.moves, len(jumps), before.count("/")), (153, 51, 31))
        # The headers, line breaks and trailing spaces are the record's own.
        self.assertEqual(conversion.text.splitlines()[:8], text.splitlines()[:8])
        self.assertEqual(conversion.text.count(b"\n"), text.count(b"\n"))
        self.assertTrue(conversion.text.splitlines()[8].endswith(b" "))

    def test_every_other_record_is_fed_exactly_as_before(self):
        # The identity, on every record of the corpus but one, whatever its source.
        for record in self.readable():
            stored = self.stored(record)
            given = corpus.feed(record, self.lock)
            if record.id == MANY2:
                self.assertNotEqual(given.data, stored)
                self.assertEqual((given.converted, given.unsupported), (153, ""))
                continue
            self.assertEqual(export_form.convert(stored), export_form.Conversion(text=stored), record.id)
            self.assertEqual(given, corpus.Feed(data=stored), record.id)
            self.assertEqual(corpus.read(record, self.lock), stored, record.id)
            self.assertEqual((record.converted, record.fed_sha256, record.reading), ("", record.sha256, ""), record.id)

    def test_many2_is_fed_as_its_manifest_entry_pins_it(self):
        record, _, conversion = self.many2()
        self.assertEqual(record.converted, sha256_bytes(conversion.text))
        self.assertEqual(record.fed_sha256, record.converted)
        self.assertEqual(corpus.read(record, self.lock), conversion.text)
        self.assertIn("read as converted from the official client's export form", record.reading)
        self.assertNotIn("record-converted", {item.code for item in corpus.verify([record], self.lock)})
        # The engine is given that text on standard input, like every other source.
        given = []

        class Listener(EngineSource):
            def observe(self, record_id, data, options):
                given.append(data)
                return Observation(source=self.name, record=record_id, status="ok", actions_accepted=0)

        runner.observe(Listener(["engine"], "full"), record.id, corpus.feed(record, self.lock), RunOptions())
        self.assertEqual(given, [conversion.text])

    def test_a_conversion_that_moved_stops_the_run_and_verify_says_what_to_pin(self):
        record, _, conversion = self.many2()
        moved = dataclasses.replace(record, converted="0" * 64)
        with self.assertRaisesRegex(HarnessError, "manifest says converted"):
            corpus.feed(moved, self.lock)
        found = corpus.verify([moved], self.lock)
        self.assertEqual([item.code for item in found], ["record-converted"])
        self.assertIn(sha256_bytes(conversion.text), found[0].message)
        with self.assertRaisesRegex(HarnessError, "manifest says converted"):
            corpus.feed(dataclasses.replace(record, converted=""), self.lock)


class GivenToNoSourceTests(unittest.TestCase):
    """A record with a move the specification cannot express: unsupported for every source, with the reason."""

    def test_every_source_reports_it_unsupported_and_none_is_run(self):
        lock = lockfile.load()
        why = "the record is in the official client's export form, and '(0T1)Ng1(1T1)g3' moves from one board"
        given = corpus.Feed(data=b"1. (0T1)Ng1(1T1)g3\n", unsupported=why)
        sources = [*(make_source(name, lock) for name in ORACLE_SOURCES), EngineSource(["./engine"], "full"),
                   EngineSource(["./engine"], "parse")]
        for source in sources:
            # The harness refused the record: no source is run on it.
            with mock.patch.object(type(source), "observe", side_effect=AssertionError("a source was run")):
                observation = runner.observe(source, "r/1", given, RunOptions(cap=1000))
            self.assertEqual((observation.source, observation.status, observation.reason, observation.detail,
                              observation.actions_accepted, observation.positions),
                             (source.name, "unsupported", corpus.EXPORT_FORM, why, 0, []), source.name)
            self.assertEqual(validate(observation), [], source.name)
            self.assertEqual(observation.cap, 1000 if "legal_actions" in source.answers else None, source.name)


class ManifestTests(unittest.TestCase):
    """The manifest pins what the sources read: `converted`, checked by verify and by every read."""

    EXPORTED = "[Board \"Standard\"]\r\n1.(0T1)Pe2(0T1)e3 / (0T1)Pe7(0T1)e6\r\n"
    REFUSED = "[Board \"Standard\"]\n1.(0T1)Pe2(0T1)e3 / (0T1)Ng8(1T1)f6\n"

    def setUp(self):
        self.box = Sandbox(self)
        for name, text in (("exported", self.EXPORTED), ("refused", self.REFUSED)):
            (self.box.upstream / "games" / f"{name}.5dpgn").write_bytes(text.encode())
        git(self.box.upstream, "add", "-A")
        git(self.box.upstream, "commit", "-q", "-m", "records in the export form")
        git(self.box.checkout, "pull", "-q", "origin", "main")
        self.lock = self.box.lock(commit=git(self.box.upstream, "rev-parse", "HEAD"))
        # The fake oracle stands in for cwmtt, the source whose records are converted.
        patcher = mock.patch.object(corpus, "CONVERTED_SOURCE", "fake")
        patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, name: str, **changes) -> Record:
        data = (self.box.upstream / "games" / f"{name}.5dpgn").read_bytes()
        record = Record(id=f"fake/{name}", source="fake", path=f"games/{name}.5dpgn",
                        blob=git(self.box.upstream, "rev-parse", f"HEAD:games/{name}.5dpgn"),
                        sha256=sha256_bytes(data), size=len(data), board="Standard", licence="Fake", tier="commit")
        return dataclasses.replace(record, **changes)

    def codes(self, record: Record) -> list[str]:
        return sorted({item.code for item in corpus.verify([record], self.lock)})

    def test_a_record_in_the_export_form_is_fed_converted_once_pinned(self):
        converted = b'[Board "Standard"]\n1.(0T1)Pe2e3 / (0T1)Pe7e6\n'
        self.assertEqual(self.codes(self.record("exported")), ["record-converted"])
        with self.assertRaises(HarnessError):
            corpus.feed(self.record("exported"), self.lock)
        pinned = self.record("exported", converted=sha256_bytes(converted))
        self.assertEqual(self.codes(pinned), [])
        self.assertEqual(corpus.feed(pinned, self.lock), corpus.Feed(data=converted, converted=2))
        self.assertEqual(pinned.fed_sha256, sha256_bytes(converted))
        # The stub `corpus --unlisted` prints for it says so.
        stub = corpus.describe_candidate("fake", pinned.path, pinned.blob, self.lock)
        self.assertEqual(stub["converted"], pinned.converted)

    def test_a_record_with_an_inexpressible_move_is_pinned_unsupported_and_given_to_no_source(self):
        self.assertEqual(self.codes(self.record("refused")), ["record-converted"])
        self.assertIn("(0T1)Ng8(1T1)f6", corpus.verify([self.record("refused")], self.lock)[0].message)
        pinned = self.record("refused", converted=corpus.UNSUPPORTED)
        self.assertEqual(self.codes(pinned), [])
        given = corpus.feed(pinned, self.lock)
        self.assertEqual(given.data, self.REFUSED.encode())
        self.assertIn("'(0T1)Ng8(1T1)f6' moves from one board to another", given.unsupported)
        with self.assertRaisesRegex(HarnessError, "is given to no source"):
            corpus.read(pinned, self.lock)
        self.assertEqual(pinned.fed_sha256, pinned.sha256)
        self.assertIn("given to no source", pinned.reading)
        self.assertEqual(corpus.describe_candidate("fake", pinned.path, pinned.blob, self.lock)["converted"],
                         corpus.UNSUPPORTED)

    def test_only_records_in_the_export_form_of_the_converted_source_say_converted(self):
        self.assertEqual(self.codes(self.record("sample", board="Standard")), [])
        self.assertEqual(self.codes(self.record("sample", converted="0" * 64)), ["record-converted"])
        with mock.patch.object(corpus, "CONVERTED_SOURCE", "cwmtt"):
            # Another source's record in the export form is fed as stored, and must not say converted.
            self.assertEqual(corpus.feed(self.record("exported"), self.lock).data, self.EXPORTED.replace("\r\n", "\n").encode())
            found = corpus.verify([self.record("exported", converted="0" * 64)], self.lock)
            self.assertEqual([item.code for item in found], ["record-converted"])
            self.assertIn("only cwmtt's records are converted", found[0].message)


class GoldenTests(GoldenCase):
    """Answers recorded about one text are not taken for answers about another."""

    def test_answers_recorded_for_the_stored_record_are_stale_once_it_is_read_converted(self):
        source = FakeSource("ftxi")
        self.write(source, self.results("ftxi"))
        self.assertEqual(self.verify(source), [])
        converted = [dataclasses.replace(self.records[0], converted="c" * 64), self.records[1]]
        found = self.verify(source, records=converted)
        self.assertEqual([item.code for item in found], ["golden-stale"])
        self.assertIn("r/1 was recorded for different record content (the record is now read as converted", found[0].message)
        # Recorded about the converted text, they are current - and the file names that text.
        text = self.write(source, self.results("ftxi"), records=converted)
        self.assertIn('"sha256": "' + "c" * 64 + '"', text)
        self.assertEqual(self.verify(source, records=converted), [])

    def test_disagreements_md_says_which_records_were_read_converted(self):
        self.write_all()
        plain = golden.render_disagreements(self.records, self.loaded())
        self.assertNotIn("## Records not read as stored", plain)
        converted = [dataclasses.replace(self.records[0], converted="c" * 64), self.records[1]]
        text = golden.render_disagreements(converted, self.loaded())
        self.assertIn("## Records not read as stored", text)
        self.assertIn("- `r/1`: read as converted from the official client's export form", text)
        self.assertNotIn("- `r/2`", text)


if __name__ == "__main__":
    unittest.main()
