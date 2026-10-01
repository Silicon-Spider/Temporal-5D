import tempfile
import unittest
from pathlib import Path

from oracle_harness import corpus, lockfile, provenance


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.lock = lockfile.load()
        self.records = corpus.load()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "PROVENANCE.md"

    def codes(self):
        return [finding.code for finding in provenance.check(self.lock, self.records, self.path)]

    def test_inventory_lists_every_pin_with_its_commit_and_licence(self):
        block = provenance.render_block(self.lock, self.records)
        for pin in self.lock.pins.values():
            self.assertIn(pin.commit, block)
            self.assertIn(pin.licence_sha256, block)
            self.assertIn(pin.url, block)
        self.assertIn(f"{len(self.records)} records", block)

    def test_write_then_check(self):
        self.path.write_text(f"# Provenance\n\nintro\n\n{provenance.BEGIN}\nold\n{provenance.END}\n\noutro\n", encoding="utf-8")
        self.assertEqual(self.codes(), ["provenance-stale"])
        self.assertTrue(provenance.write(self.lock, self.records, self.path))
        self.assertEqual(self.codes(), [])
        self.assertFalse(provenance.write(self.lock, self.records, self.path))
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Provenance\n\nintro\n\n"))
        self.assertTrue(text.endswith("\n\noutro\n"))

    def test_hand_edits_inside_the_inventory_are_noticed(self):
        self.path.write_text(f"{provenance.BEGIN}\n{provenance.END}\n", encoding="utf-8")
        provenance.write(self.lock, self.records, self.path)
        edited = self.path.read_text(encoding="utf-8").replace("BSD-2-Clause", "MIT")
        self.path.write_text(edited, encoding="utf-8")
        self.assertEqual(self.codes(), ["provenance-stale"])

    def test_missing_file_and_missing_markers(self):
        self.assertEqual(self.codes(), ["provenance-missing"])
        self.path.write_text("# Provenance\n", encoding="utf-8")
        self.assertEqual(self.codes(), ["provenance-markers"])


if __name__ == "__main__":
    unittest.main()
