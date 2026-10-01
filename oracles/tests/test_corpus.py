import dataclasses
import unittest
from unittest import mock

from oracle_harness import corpus, headers, lockfile, paths
from oracle_harness.corpus import Excluded, Record
from oracle_harness.util import HarnessError, sha256_bytes
from support import FAKE_RECORD, Sandbox, git


def codes(findings):
    return sorted({finding.code for finding in findings})


class HeaderTests(unittest.TestCase):
    def test_board_header(self):
        self.assertEqual(corpus.board_header(b'[Mode "5D"]\n[Board "Standard - Turn Zero"]\n1. e3'), "Standard - Turn Zero")
        self.assertEqual(corpus.board_header(b'[board   "custom" ]'), "custom")
        self.assertIsNone(corpus.board_header(b'[Size "5x5"]\n1. Ka2 / Kd4'))

    def test_fen_boards_are_recognised(self):
        standard = b"[r*nbqk*bnr*/p*p*p*p*p*p*p*p*/8/8/8/8/P*P*P*P*P*P*P*P*/R*NBQK*BNR*:0:1:w]"
        self.assertTrue(corpus.has_fen_boards(b'[Board "custom"]\n' + standard + b"\n1. e3"))
        self.assertTrue(corpus.has_fen_boards(b"[3k1/5/5/1R3/K4:-0:1:b]\n"))
        self.assertTrue(corpus.has_fen_boards(b"  [3rk/5/K2r1/5/5:+1:12:w]  \r\n".replace(b"\r\n", b"\n")))

    def test_headers_and_moves_are_not_fen_boards(self):
        for text in (b'[Board "Standard"]', b'[Date "2021.01.22"]', b"1. (0T1)Nf3 / (0T1)Nf6", b'[Result "1-0"]'):
            self.assertFalse(corpus.has_fen_boards(text), text)

    def test_any_header_can_be_asked_for(self):
        record = b'[Board "Standard"]\n[promotions "Q,N"]\n1. e3 / e6\n'
        self.assertEqual(headers.header(record, "Promotions"), "Q,N")
        self.assertIsNone(headers.header(record, "Size"))
        self.assertIsNone(headers.header(b'[Boardroom "x"]', "Board"))

    def test_a_board_string_is_found_wherever_a_reader_would_take_it(self):
        # On a line with a header, before a comment, with the colour in capitals,
        # and in a record stored with Windows line endings.
        for text in (b'[Size "5x5"] [3k1/5/5/1R3/K4:0:1:w]\n1. Ka2 / Kd4\n',
                     b"[3k1/5/5/1R3/K4:0:1:w] {the position}\n",
                     b"[3k1/5/5/1R3/K4:0:1:W]\n",
                     b'[Board "Custom"]\r\n[3k1/5/5/1R3/K4:0:1:w]\r\n\r\n1. Ka2 / Kd4\r\n'):
            self.assertTrue(corpus.has_fen_boards(text), text)
        self.assertEqual(headers.fen_board_count(b"[3k1/5/5/1R3/K4:0:1:w]\n[3k1/5/5/1R3/K4:1:1:w]\n"), 2)

    def test_the_declared_board_falls_back_to_a_variant_header(self):
        # ftxi reads [Variant] when there is no [Board]; so does the harness.
        self.assertEqual(headers.declared_board(b'[Variant "Standard - Turn Zero"]\n1. e3 / e6\n'), "Standard - Turn Zero")
        self.assertEqual(headers.declared_board(b'[Board "Standard"]\n[Variant "custom"]\n'), "Standard")
        self.assertIsNone(headers.declared_board(b"1. e3 / e6\n"))


class RulesetTests(unittest.TestCase):
    def ruleset(self, board, fen=False):
        return Record(id="r/1", source="s", path="p", blob="b", sha256="h", size=1, board=board, licence="x",
                      tier="commit", fen=fen).ruleset

    def test_a_record_counts_towards_the_variant_it_declares(self):
        self.assertEqual(self.ruleset("Standard - Turn Zero"), "Standard - Turn Zero")
        self.assertEqual(self.ruleset(""), "Standard")  # the notation's default

    def test_a_record_with_its_own_start_position_counts_towards_no_variant(self):
        # Whatever it calls itself: the harness cannot tell that the position is the variant's.
        for board in ("", "Custom", "Standard - Reversed Royalty"):
            self.assertEqual(self.ruleset(board, fen=True), corpus.CUSTOM_POSITIONS, board)
        # And a custom set-up is custom even where no board string was recognised.
        for board in ("Custom", "custom", "Custom - Even", "Even", "Odd"):
            self.assertEqual(self.ruleset(board), corpus.CUSTOM_POSITIONS, board)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox(self)
        self.lock = self.box.lock()
        data = FAKE_RECORD.encode()
        self.record = Record(
            id="fake/sample", source="fake", path="games/sample.5dpgn",
            blob=git(self.box.upstream, "rev-parse", "HEAD:games/sample.5dpgn"),
            sha256=sha256_bytes(data), size=len(data), board="Standard", licence="Fake", tier="commit",
        )

    def verify(self, **changes):
        return corpus.verify([dataclasses.replace(self.record, **changes)], self.lock)

    def test_a_faithful_entry_passes(self):
        self.assertEqual(self.verify(), [])

    def test_the_record_is_read_from_the_pinned_commit_with_line_endings_folded(self):
        self.assertIn("\r\n", FAKE_RECORD)
        self.assertEqual(corpus.read(self.record, self.lock), FAKE_RECORD.replace("\r\n", "\n").encode())

    def test_editing_the_work_tree_cannot_change_the_corpus(self):
        (self.box.checkout / "games" / "sample.5dpgn").write_text("1. d4 / d5\n")
        self.assertEqual(corpus.read(self.record, self.lock), FAKE_RECORD.replace("\r\n", "\n").encode())

    def test_every_pinned_field_is_checked(self):
        self.assertEqual(codes(self.verify(blob="0" * 40)), ["record-blob"])
        self.assertEqual(codes(self.verify(sha256="0" * 64)), ["record-sha256"])
        self.assertEqual(codes(self.verify(size=1)), ["record-size"])
        self.assertEqual(codes(self.verify(path="games/missing.5dpgn")), ["record-missing"])
        self.assertEqual(codes(self.verify(source="nowhere")), ["record-source"])
        self.assertEqual(codes(self.verify(board="Standard - Turn Zero")), ["record-board"])
        self.assertEqual(codes(self.verify(tier="weekly")), ["record-tier"])
        self.assertEqual(codes(self.verify(licence=" ")), ["record-licence"])
        self.assertEqual(codes(self.verify(id="No Slash")), ["record-id"])
        self.assertEqual(codes(self.verify(fen=True)), ["record-fen"])

    def test_a_record_stored_with_windows_line_endings_is_judged_as_it_is_fed(self):
        # The sources get CRLF folded to LF; a board string must be seen the same way.
        data = b'[Board "Custom"]\r\n[Size "5x5"]\r\n[3k1/5/5/1R3/K4:0:1:w]\r\n\r\n1. Ka2 / Kd4\r\n'
        self.box.write(self.box.upstream / "games" / "crlf.5dpgn", "")
        (self.box.upstream / "games" / "crlf.5dpgn").write_bytes(data)
        git(self.box.upstream, "add", "-A")
        git(self.box.upstream, "commit", "-q", "-m", "a record with CRLF")
        git(self.box.checkout, "pull", "-q", "origin", "main")
        lock = self.box.lock(commit=git(self.box.upstream, "rev-parse", "HEAD"))
        record = Record(id="fake/crlf", source="fake", path="games/crlf.5dpgn",
                        blob=git(self.box.upstream, "rev-parse", "HEAD:games/crlf.5dpgn"),
                        sha256=sha256_bytes(data), size=len(data), board="Custom", licence="Fake", tier="commit", fen=True)
        self.assertEqual(corpus.verify([record], lock), [])
        self.assertEqual(codes(corpus.verify([dataclasses.replace(record, fen=False)], lock)), ["record-fen"])

    def test_a_record_with_fen_boards_must_be_marked(self):
        fen_record = '[Board "Custom"]\n[3k1/5/5/1R3/K4:0:1:w]\n\n1. Ka2 / Kd4\n'
        self.box.write(self.box.upstream / "games" / "custom.5dpgn", fen_record)
        git(self.box.upstream, "add", "-A")
        git(self.box.upstream, "commit", "-q", "-m", "a custom position")
        git(self.box.checkout, "pull", "-q", "origin", "main")
        lock = self.box.lock(commit=git(self.box.upstream, "rev-parse", "HEAD"))
        stub = next(corpus.describe_candidate(pin, path, blob, lock)
                    for pin, path, blob in corpus.unlisted([self.record], lock))
        self.assertEqual((stub["id"], stub["board"], stub["fen"]), ("fake/custom", "Custom", True))
        record = Record(licence="Fake", tier="commit", **stub)
        self.assertEqual(corpus.verify([record], lock), [])
        self.assertEqual(record.ruleset, corpus.CUSTOM_POSITIONS)
        self.assertEqual(codes(corpus.verify([dataclasses.replace(record, fen=False)], lock)), ["record-fen"])

    def test_reading_a_record_whose_hash_is_wrong_fails(self):
        with self.assertRaises(HarnessError):
            corpus.read(dataclasses.replace(self.record, sha256="0" * 64), self.lock)

    def test_duplicates_are_reported(self):
        twin = dataclasses.replace(self.record, id="fake/twin")
        self.assertEqual(codes(corpus.verify([self.record, twin], self.lock)), ["record-duplicate"])
        self.assertEqual(codes(corpus.verify([self.record, self.record], self.lock)), ["record-duplicate", "record-duplicate-id"])

    def test_also_at_must_be_the_same_blob(self):
        self.assertEqual(self.verify(also_at=("fake:games/sample.5dpgn",)), [])
        self.assertEqual(codes(self.verify(also_at=("fake:LICENSE",))), ["record-also-at"])

    def test_unlisted_records_are_found_unless_excluded(self):
        self.assertEqual(corpus.unlisted([self.record], self.lock), [])
        found = corpus.unlisted([], self.lock)
        self.assertEqual([(pin, path) for pin, path, _ in found], [("fake", "games/sample.5dpgn")])
        excluded = [Excluded("fake", "games/sample.5dpgn", "a duplicate")]
        self.assertEqual(corpus.unlisted([], self.lock, excluded), [])
        self.assertEqual(corpus.verify_excluded(excluded, self.lock), [])
        self.assertEqual(codes(corpus.verify_excluded([Excluded("fake", "games/gone.5dpgn", "")], self.lock)),
                         ["excluded-missing", "excluded-reason"])

    def test_a_stub_describes_a_candidate(self):
        pin, path, blob = corpus.unlisted([], self.lock)[0]
        stub = corpus.describe_candidate(pin, path, blob, self.lock)
        self.assertEqual((stub["id"], stub["sha256"], stub["board"]), ("fake/sample", self.record.sha256, "Standard"))

    def test_records_of_our_own_are_listed_and_kept_in_one_place(self):
        home = self.box.repo / "oracles" / "corpus"
        data = b"1. e3 / e6\n"
        self.box.write(home / "records" / "ours.5dpgn", data.decode())
        local = Record(id="local/ours", source="local", path="oracles/corpus/records/ours.5dpgn", blob="",
                       sha256=sha256_bytes(data), size=len(data), board="", licence="ours", tier="commit")
        with mock.patch.object(paths, "CORPUS_DIR", home), mock.patch.object(paths, "CORPUS_MANIFEST", home / "manifest.toml"):
            self.assertEqual(corpus.verify([local], self.lock) + corpus.verify_local([local]), [])
            self.assertEqual(corpus.read(local, self.lock), data)
            self.assertEqual(local.ruleset, "Standard")
            # A record file nobody listed, and a listed one kept somewhere git may rewrite its line endings.
            self.assertEqual(codes(corpus.verify_local([])), ["record-unlisted"])
            elsewhere = dataclasses.replace(local, path="engine/ours.5dpgn")
            self.assertEqual(codes(corpus.verify_local([elsewhere])), ["record-placement", "record-unlisted"])
            self.assertEqual(codes(corpus.verify([dataclasses.replace(local, sha256="0" * 64)], self.lock)), ["record-sha256"])

    def test_selection_by_tier_and_by_id(self):
        nightly = dataclasses.replace(self.record, id="fake/slow", tier="nightly")
        both = [self.record, nightly]
        self.assertEqual(corpus.select(both, ("commit",)), [self.record])
        self.assertEqual(corpus.select(both, ("commit", "nightly")), both)
        self.assertEqual(corpus.select(both, ("commit",), ["fake/slow"]), [nightly])
        with self.assertRaises(HarnessError):
            corpus.select(both, ("commit",), ["fake/unknown"])


class RepositoryCorpusTests(unittest.TestCase):
    def test_the_manifest_loads_and_names_pinned_sources(self):
        lock = lockfile.load()
        records = corpus.load()
        self.assertGreater(len(records), 30)
        self.assertEqual(len({record.id for record in records}), len(records))
        self.assertTrue(all(record.source in lock.pins or record.source == corpus.LOCAL for record in records))
        self.assertTrue(all(entry.reason.strip() for entry in corpus.load_excluded()))

    def test_the_corpus_holds_what_the_pinned_sources_lack(self):
        # A stalemate, and an action every oracle has to refuse: written for this project.
        local = {record.id for record in corpus.load() if record.source == corpus.LOCAL}
        self.assertLessEqual({"local/stalemate", "local/checkmate", "local/ignores-check"}, local)
        self.assertEqual(corpus.verify_local(corpus.load()), [])


if __name__ == "__main__":
    unittest.main()
