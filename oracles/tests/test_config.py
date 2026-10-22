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

    def test_a_coverage_floor_is_refused_not_ignored(self):
        # H32: no tolerance. A file that still sets one must not let anyone believe a floor is in force.
        for line in ("min_coverage = 0.95\n", "[tier.release.source_min_coverage]\nchessin5d = 0.9\n"):
            with self.assertRaises(HarnessError) as raised:
                self.load({"require_engine = true\n": f"require_engine = true\n{line}"})
            self.assertIn("100%", str(raised.exception))

    def test_the_stability_policy(self):
        stability = config.load().stability
        self.assertGreaterEqual(stability.margin, 1.0)
        self.assertEqual(config.load().recording.name, stability.recording_tier)
        # Below 1 a tier could require an answer that came after its own deadline on the recording machine.
        with self.assertRaises(HarnessError):
            self.load({"margin = 4\n": "margin = 0.5\n"})
        # The golden files are recorded under a tier that runs every source.
        with self.assertRaises(HarnessError):
            self.load({'recording_tier = "release"\n': 'recording_tier = "commit"\n'})
        with self.assertRaises(HarnessError):
            self.load({'recording_tier = "release"\n': 'recording_tier = "weekly"\n'})
        for share in ("0", "1.5"):
            with self.assertRaises(HarnessError):
                self.load({"proof_share = 0.9\n": f"proof_share = {share}\n"})
        self.assertEqual(self.load({"margin = 4\n": "margin = 1\n"}).stability.margin, 1.0)

    def test_the_calibration_probe_is_a_named_workload(self):
        probe = config.load().calibration
        self.assertGreaterEqual(probe.positions, 1)
        for old, new in (('source = "ftxi:naive"\n', 'source = "stockfish"\n'), ("positions = 38\n", "positions = 0\n"),
                         ('record = "cwmtt/manyChecks"\n', 'record = ""\n')):
            with self.assertRaises(HarnessError, msg=new):
                self.load({old: new})

    def test_the_engine_mode(self):
        self.assertEqual(config.load().engine_mode, "full")
        self.assertEqual(self.load({'mode = "full"\n': 'mode = "parse"\n'}).engine_mode, "parse")
        with self.assertRaises(HarnessError):
            self.load({'mode = "full"\n': 'mode = "rules-ish"\n'})

    def test_the_engines_time_limit_is_its_own_or_the_tiers_where_that_is_longer(self):
        # [engine] timeout_s: how slow the engine may be on a record, in every tier. Never less than the tier's own
        # timeout_s - no tier gives the engine less time than it did - and it changes no budget of an oracle's.
        line = '\ntimeout_s = 300\n# "full"'
        cfg = self.load({line: '\ntimeout_s = 200\n# "full"'})
        self.assertEqual(cfg.engine_timeout_s, 200.0)
        self.assertEqual({name: cfg.engine_timeout(tier) for name, tier in cfg.tiers.items()},
                         {"commit": 200.0, "nightly": 300.0, "release": 600.0})
        self.assertEqual({name: tier.timeout_s for name, tier in cfg.tiers.items()},
                         {name: tier.timeout_s for name, tier in config.load().tiers.items()})
        # Left out: the tier's own.
        cfg = self.load({line: '\n# "full"'})
        self.assertIsNone(cfg.engine_timeout_s)
        self.assertEqual(cfg.engine_timeout(cfg.tier("commit")), cfg.tier("commit").timeout_s)
        for value in ("0", "-5", '"5m"', "true"):
            with self.assertRaises(HarnessError, msg=value):
                self.load({line: f'\ntimeout_s = {value}\n# "full"'})

    def test_the_engines_baseline_names_a_tier_and_its_pin(self):
        # H47: what the engine read on the last green run of a tier, pinned like the rulebook (baseline.py).
        spec = config.load().baseline
        self.assertEqual(spec.tier, "commit")
        self.assertRegex(spec.sha256, r"^(?:[0-9a-f]{64})?$")
        table = f'[baseline]\ntier = "commit"\nsha256 = "{spec.sha256}"\n'
        for bad in ('[baseline]\ntier = "weekly"\nsha256 = ""\n', '[baseline]\ntier = "commit"\nsha256 = "ABC"\n',
                    '[baseline]\ntier = "commit"\nsha256 = ""\nmargin = 2\n', '[baseline]\nsha256 = ""\n'):
            with self.assertRaises(HarnessError, msg=bad):
                self.load({table: bad})
        # Before one is recorded, the pin is empty; without the table there is none to compare with.
        self.assertEqual(self.load({table: '[baseline]\ntier = "commit"\nsha256 = ""\n'}).baseline,
                         config.Baseline(tier="commit", sha256=""))
        self.assertIsNone(self.load({table: ""}).baseline)

    def test_a_pin_is_written_into_its_own_table_and_nowhere_else(self):
        # The baseline's pin and the rulebook's are both `sha256 = "..."`: writing one leaves the other, and every
        # comment and setting of the file, as it was - whatever its line endings.
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        digest = "f" * 64
        for newline in ("\n", "\r\n"):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "gate.toml"
                path.write_bytes(text.replace("\n", newline).encode("utf-8"))
                config.write_sha256(path, "baseline", digest)
                written = path.read_bytes().decode("utf-8")
                loaded = config.load(path)
                self.assertEqual((loaded.baseline.sha256, loaded.rulebook), (digest, config.load().rulebook))
                expected = text.replace(f'tier = "commit"\nsha256 = "{config.load().baseline.sha256}"',
                                        f'tier = "commit"\nsha256 = "{digest}"').replace("\n", newline)
                self.assertEqual(written, expected)
                with self.assertRaises(HarnessError):
                    config.write_sha256(path, "no-such-table", digest)

    def test_only_a_tier_that_requires_an_engine_and_runs_everything_can_enable_the_proof_channel(self):
        tiers = config.load().tiers
        self.assertEqual([name for name, tier in tiers.items() if tier.can_enable_proof_channel], ["release"])

    def test_only_a_tier_that_runs_every_oracle_search_over_the_whole_corpus_may_require_an_engine(self):
        # Such a tier fails unless an engine came through a named variant, and any other tier is a smoke test that
        # gates nothing (H47): no engine could ever pass it. Refused, not left to fail with nothing to say why.
        for old, new in (('record_tiers = ["commit"]\ntimeout_s = 60\n',
                          'record_tiers = ["commit"]\ntimeout_s = 60\nrequire_engine = true\n'),
                         ('record_tiers = ["commit", "nightly"]\ntimeout_s = 600\n',
                          'record_tiers = ["commit"]\ntimeout_s = 600\n')):
            with self.assertRaises(HarnessError, msg=new) as raised:
                self.load({old: new})
            self.assertIn("require_engine needs a tier that runs every oracle source over every record tier",
                          str(raised.exception))
        self.assertTrue(config.load().tier("release").require_engine)

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

    def test_a_mate_budget_is_a_positive_whole_number_of_milliseconds(self):
        for value in ("0", "2.5", '"60s"'):
            with self.assertRaises(HarnessError, msg=value):
                self.load({"mate_budget_ms = 60000\n": f"mate_budget_ms = {value}\n"})

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
