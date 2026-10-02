"""The rulebook pin (decision H38): the rulebook the known divergences cite is pinned in gate.toml, and the static
checks - so every gate tier - fail once it changes, until the change is reviewed and the pin moved."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from oracle_harness import config, paths, rulebook
from oracle_harness.util import HarnessError, sha256_text

TEXT = "# Rules\n\n- **R1** A royal is in check if ...\n"


class RulebookPinTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "docs" / "rules").mkdir(parents=True)
        self.book = self.root / "docs" / "rules" / "README.md"
        self.book.write_bytes(TEXT.encode())
        # The repository's own gate.toml, pinned at this rulebook.
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        current = config.load().rulebook.sha256
        self.assertIn(f'sha256 = "{current}"', text)
        self.gate_file = self.root / "gate.toml"
        self.gate_file.write_bytes(text.replace(current, sha256_text(TEXT.encode())).encode())
        patcher = mock.patch.object(paths, "REPO_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def spec(self):
        return config.load(self.gate_file).rulebook

    def test_the_repository_pins_its_rulebook(self):
        self.assertEqual(config.load().rulebook.path, "docs/rules/README.md")

    def test_the_text_it_is_pinned_at_passes(self):
        self.assertEqual(rulebook.verify(self.spec()), [])
        # A checkout with CRLF line endings holds the same text.
        self.book.write_bytes(TEXT.replace("\n", "\r\n").encode())
        self.assertEqual(rulebook.verify(self.spec()), [])

    def test_a_changed_rulebook_fails_until_the_pin_is_moved(self):
        self.book.write_bytes(TEXT.replace("check", "check (amended)").encode())
        found = rulebook.verify(self.spec())
        self.assertEqual([item.code for item in found], ["rulebook-changed"])
        self.assertTrue(found[0].message.startswith(
            f"the rulebook changed: re-run the gate and update the pin with {rulebook.PIN_COMMAND}"))
        self.assertEqual(rulebook.PIN_COMMAND, f"python {paths.HARNESS_REL}/harness.py rulebook --pin")

    def test_a_missing_rulebook_fails(self):
        self.book.unlink()
        self.assertEqual([item.code for item in rulebook.verify(self.spec())], ["rulebook-missing"])
        with self.assertRaises(HarnessError):
            rulebook.pin(self.gate_file)

    def test_the_pin_command_moves_the_pin_and_nothing_else(self):
        before = self.gate_file.read_text(encoding="utf-8")
        old = self.spec().sha256
        self.book.write_bytes(TEXT.replace("check", "check (amended)").encode())
        self.assertEqual(rulebook.pin(self.gate_file), (old, sha256_text(self.book.read_bytes())))
        after = self.gate_file.read_text(encoding="utf-8")
        self.assertEqual(after, before.replace(old, sha256_text(self.book.read_bytes())))
        self.assertEqual(rulebook.verify(self.spec()), [])
        # Pinning the text already pinned changes nothing.
        self.assertEqual(rulebook.pin(self.gate_file), (self.spec().sha256, self.spec().sha256))
        self.assertEqual(self.gate_file.read_text(encoding="utf-8"), after)

    def test_a_malformed_pin_is_refused(self):
        text = self.gate_file.read_text(encoding="utf-8")
        spec = self.spec()
        for old, new in ((f'sha256 = "{spec.sha256}"', 'sha256 = "tbd"'),
                         (f'sha256 = "{spec.sha256}"', f'sha256 = "{spec.sha256.upper()}"'),
                         ('path = "docs/rules/README.md"', 'path = "../rules.md"'),
                         ('path = "docs/rules/README.md"', 'path = "/etc/rules.md"'),
                         ('path = "docs/rules/README.md"', 'path = "docs\\\\rules\\\\README.md"'),
                         ("[rulebook]\n", "[not_the_rulebook]\n")):
            self.assertIn(old, text)
            self.gate_file.write_text(text.replace(old, new), encoding="utf-8")
            with self.assertRaises(HarnessError, msg=new):
                config.load(self.gate_file)


if __name__ == "__main__":
    unittest.main()
