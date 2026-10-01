import tempfile
import unittest
from pathlib import Path

from oracle_harness import config, paths
from oracle_harness.util import HarnessError


class TierConfigTests(unittest.TestCase):
    """gate.toml may not quietly turn a listed source into a decoration."""

    def load(self, replacements: dict[str, str]):
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        for old, new in replacements.items():
            self.assertIn(old, text)
            text = text.replace(old, new)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gate.toml"
            path.write_text(text, encoding="utf-8")
            return config.load(path)

    def test_the_repository_file_loads(self):
        tiers = config.load().tiers
        self.assertEqual(set(tiers), {"commit", "nightly", "release"})

    def test_a_coverage_floor_near_nothing_is_refused(self):
        # A source starved of time reproduces nothing; with no floor it would pass.
        with self.assertRaises(HarnessError):
            self.load({"[tier.release.source_min_coverage]\n": "[tier.release.source_min_coverage]\nchessin5d = 0.0\n"})
        with self.assertRaises(HarnessError):
            self.load({"min_coverage = 0.95\n": "min_coverage = 0.2\n"})

    def test_budgets_for_a_source_the_tier_does_not_run_are_refused(self):
        # Probably a typo - and a budget that silently applies to nothing.
        with self.assertRaises(HarnessError):
            self.load({"[tier.nightly.source_timeout_s]\n": '[tier.nightly.source_timeout_s]\n"chessin5d:naive" = 10\n'})

    def test_a_timeout_must_be_positive(self):
        with self.assertRaises(HarnessError):
            self.load({"timeout_s = 600\n": "timeout_s = 0\n"})

    def test_min_checked_cannot_be_zero(self):
        with self.assertRaises(HarnessError):
            self.load({"min_checked = 0.75\nrequire_engine": "min_checked = 0\nrequire_engine"})

    def test_a_waiver_needs_the_text_it_excuses_and_a_reason(self):
        waiver = '\n[[boundary.waiver]]\npath = "tools/x.py"\nrule = "references-harness"\nreason = "because"\n'
        with self.assertRaises(HarnessError):
            self.load({"copy_long_line_chars = 160\n": "copy_long_line_chars = 160\n" + waiver})
        complete = waiver.replace('reason = "because"', 'text = "oracles/harness.py"\nreason = "because"')
        loaded = self.load({"copy_long_line_chars = 160\n": "copy_long_line_chars = 160\n" + complete})
        self.assertEqual([(item.path, item.text) for item in loaded.boundary.waivers], [("tools/x.py", "oracles/harness.py")])

    def test_the_copy_detector_cannot_be_switched_off_by_its_settings(self):
        # A window of one line would flag every shared brace; a window wider than
        # any file, or a long-line length below the line length, makes no sense.
        for old, new in (("copy_window_lines = 6\n", "copy_window_lines = 1\n"),
                         ("copy_min_line_chars = 12\n", "copy_min_line_chars = 0\n"),
                         ("copy_long_line_chars = 160\n", "copy_long_line_chars = 10\n")):
            with self.assertRaises(HarnessError, msg=new):
                self.load({old: new})
        self.assertEqual(config.load().boundary.copy_long_line_chars, 160)


if __name__ == "__main__":
    unittest.main()
