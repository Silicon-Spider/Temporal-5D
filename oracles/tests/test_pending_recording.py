"""Recording a pending source (CHECKPOINT 10: Kairos's parse mode before the release tier's sign-off run, which leaves
Kairos out): `golden --write --source <pending>` may record it, verify holds its files to the same static checks, and
nothing any tier, verify's checks of the enabled sources or DISAGREEMENTS.md reads changes."""

import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import support
from oracle_harness import build, cli, config, corpus, gate, golden, lockfile, paths, snapshot, stability
from oracle_harness.adapters import ORACLE_SOURCES, PENDING_SOURCES, RunOptions, make_source
from oracle_harness.observations import Observation
from oracle_harness.util import HarnessError

PARSE = "kairos:parse"


class PendingRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = lockfile.load()
        cls.cfg = config.load()
        cls.records = corpus.load()

    def setUp(self):
        # The repository's own golden files, copied: what verify and every tier read today.
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.golden_dir = Path(self.folder.name) / "golden"
        shutil.copytree(paths.GOLDEN_DIR, self.golden_dir)
        patcher = mock.patch.object(paths, "GOLDEN_DIR", self.golden_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def probe(self, seconds: float = 12.5) -> golden.Calibration:
        """The calibration probe gate.toml names, as timed with the current code on some machine."""
        spec = self.cfg.calibration
        signature = make_source(spec.source, self.lock).signature(RunOptions(cap=self.cfg.cap))
        return golden.Calibration(source=spec.source, record=spec.record, positions=spec.positions, signature=signature,
                                  seconds=seconds, platform="test-machine", cpu_count=4, measured="2026-11-02T10:00:00")

    def record_pending(self, *, signature_change: dict | None = None, leave_out: str | None = None,
                       probe: golden.Calibration | None | bool = True) -> None:
        """A recording of kairos:parse as golden --write would leave it - its probe in its cost file, MANIFEST rewritten
        over the directory. ``probe``: True for a current probe, None for none, or the probe to carry."""
        source = make_source(PARSE, self.lock)
        signature = {**golden.signature(source, gate.recording_options(self.cfg, source)), **(signature_change or {})}
        records = [record for record in self.records if record.id != leave_out]
        answers = {record.id: support.recorded(Observation(source=PARSE, record=record.id, status="unsupported",
                                                           reason="parse_failed", actions_accepted=0), end=0.02)
                   for record in records}
        carried = self.probe() if probe is True else probe
        golden.golden_path(PARSE).write_text(golden.dump(PARSE, signature, records, answers, recordings=2),
                                             encoding="utf-8", newline="\n")
        golden.cost_path(PARSE).write_text(golden.dump_costs(PARSE, signature, records, answers, recordings=2,
                                                             calibration=carried), encoding="utf-8", newline="\n")
        support.vouch()

    def read_back(self):
        """What verify and a tier read: the snapshot, the golden section, DISAGREEMENTS.md as rendered, readiness."""
        frozen = snapshot.take()
        return (frozen, gate.golden_section(self.lock, self.cfg, self.records, frozen),
                gate._rendered(self.cfg, self.records, frozen),
                stability.required_table(self.cfg, self.records, frozen.golden, frozen.costs),
                gate.proof_channel_readiness(self.cfg, self.records, frozen))

    def test_golden_write_may_name_a_pending_source(self):
        arguments = cli.parser().parse_args(["golden", "--write", "--source", PARSE])
        self.assertEqual(arguments.source, [PARSE])
        # Recorded under the recording tier's own budget: no tier lists it, so no budget of its own applies.
        options = gate.recording_options(self.cfg, make_source(PARSE, self.lock))
        self.assertEqual(options.timeout_s, self.cfg.recording.timeout_s)

    def test_recording_a_pending_source_changes_nothing_read_for_the_enabled_sources(self):
        before_frozen, before_section, before_text, before_required, before_ready = self.read_back()
        self.assertEqual(before_section, [])  # the repository's golden files pass verify today
        self.record_pending()
        frozen, section, text, required, ready = self.read_back()
        # The pending recording is on file, under MANIFEST ...
        self.assertTrue(golden.golden_path(PARSE).exists() and golden.cost_path(PARSE).exists())
        # ... and never in the snapshot every tier compares with and DISAGREEMENTS.md is rendered from.
        self.assertEqual(set(frozen.golden), set(before_frozen.golden))
        self.assertEqual(set(frozen.golden), set(ORACLE_SOURCES))
        self.assertNotIn(PARSE, frozen.golden)
        for name in ORACLE_SOURCES:
            self.assertEqual(frozen.golden[name].digest, before_frozen.golden[name].digest, name)
            self.assertEqual(frozen.costs[name].digest, before_frozen.costs[name].digest, name)
        self.assertEqual(frozen.problems, before_frozen.problems)
        self.assertEqual(frozen.manifest_problems, before_frozen.manifest_problems)
        # Verify's golden section - the enabled sources' checks, then the pending recording's - finds nothing.
        self.assertEqual(section, [])
        self.assertEqual(text, before_text)  # DISAGREEMENTS.md, byte for byte
        self.assertEqual(required, before_required)  # what each tier requires
        self.assertEqual(ready, before_ready)  # the proof-channel readiness of the release tier
        # No tier lists it, and the tiers that run everything still do.
        for name, tier in self.cfg.tiers.items():
            self.assertFalse(set(tier.sources) & set(PENDING_SOURCES), name)
        self.assertTrue(self.cfg.tier("nightly").full_strength and self.cfg.tier("release").full_strength)

    def test_a_stale_or_incomplete_pending_recording_fails_verify_on_its_own(self):
        for change, leave_out, code in (({"driver_sha256": "0" * 64}, None, "golden-stale"),
                                        (None, self.records[0].id, "golden-incomplete")):
            with self.subTest(code=code):
                self.record_pending(signature_change=change, leave_out=leave_out)
                frozen, section, text, _, _ = self.read_back()
                self.assertTrue(section, code)
                self.assertEqual({finding.code for finding in section}, {code})
                # Only the pending recording is named; the enabled sources' checks are as before.
                self.assertTrue(all("kairos-parse" in finding.message for finding in section), section)
                four = golden.verify_static([make_source(name, self.lock) for name in ORACLE_SOURCES],
                                            lambda source: gate.recording_options(self.cfg, source), self.records, frozen)
                self.assertEqual(four, [])

    def test_a_pending_recording_carries_its_own_probe_and_is_read_on_the_references_time(self):
        reference = snapshot.take().calibration
        self.record_pending(probe=self.probe(seconds=reference.seconds * 2))  # timed on a machine twice as slow
        frozen = snapshot.take()
        pending = snapshot.recorded_pending(frozen.manifest, frozen.calibration)
        costs = pending.costs[PARSE]
        self.assertEqual(costs.calibration.seconds, reference.seconds * 2)
        self.assertEqual(costs.calibration.measured, "2026-11-02T10:00:00")
        written = golden.load_costs(PARSE, frozen.manifest)  # as written: 0.02 s an outcome
        for record_id, cost in costs.records.items():
            self.assertAlmostEqual(cost.end, written.records[record_id].end / 2)
        # No probe of its own: a finding, and its costs are read as they are.
        self.record_pending(probe=None)
        frozen = snapshot.take()
        found = gate.pending_golden_findings(self.lock, self.cfg, self.records, frozen)
        self.assertEqual([finding.code for finding in found], ["golden-stale"])
        self.assertIn("carries none", found[0].message)
        # A probe timed with other code: a finding too.
        self.record_pending(probe=dataclasses.replace(self.probe(), signature={"pin": "other"}))
        found = gate.pending_golden_findings(self.lock, self.cfg, self.records, snapshot.take())
        self.assertEqual([finding.code for finding in found], ["golden-stale"])
        self.assertIn("other code", found[0].message)

    def test_an_enabled_sources_cost_file_has_no_probe_and_is_read_as_written(self):
        frozen = snapshot.take()
        for name in ORACLE_SOURCES:
            self.assertIsNone(frozen.costs[name].calibration, name)
            self.assertEqual(frozen.costs[name].records, golden.load_costs(name, frozen.manifest).records, name)

    @unittest.skipUnless(build.load(lockfile.load().pin("kairos")) is not None, "the kairos oracle is not built")
    def test_recording_only_a_pending_source_leaves_the_calibration_and_the_enabled_sources_untouched(self):
        # golden --write --source kairos:parse, end to end (the real driver; the probe's timing stood in for).
        manifest_before = (self.golden_dir / "MANIFEST").read_text(encoding="utf-8").splitlines()
        files = [golden.CALIBRATION_FILE] + [golden.golden_path(name).name for name in ORACLE_SOURCES] \
            + [golden.cost_path(name).name for name in ORACLE_SOURCES]
        before = {name: (self.golden_dir / name).read_bytes() for name in files}
        disagreements = Path(self.folder.name) / "DISAGREEMENTS.md"
        probe = self.probe(seconds=13.0)
        with mock.patch.object(paths, "DISAGREEMENTS_FILE", disagreements), \
                mock.patch.object(gate, "_measure_probe", return_value=probe):
            findings = gate.write_golden(only_sources=[PARSE], jobs=4, repeat=2, log=lambda message: None)
        self.assertEqual(findings, [])
        for name in files:
            self.assertEqual((self.golden_dir / name).read_bytes(), before[name], name)
        manifest_after = (self.golden_dir / "MANIFEST").read_text(encoding="utf-8").splitlines()
        self.assertEqual([line for line in manifest_after if "kairos-parse" not in line], manifest_before)
        self.assertEqual(sum("kairos-parse" in line for line in manifest_after), 2)
        self.assertFalse(disagreements.exists())  # DISAGREEMENTS.md not rendered: a pending recording does not touch it
        costs = golden.load_costs(PARSE)
        self.assertEqual((costs.calibration.seconds, costs.calibration.platform), (13.0, "test-machine"))
        self.assertTrue(costs.calibration.measured)  # stamped when written
        frozen = snapshot.take()
        self.assertEqual(gate.golden_section(self.lock, self.cfg, self.records, frozen), [])

    def test_a_pending_source_without_a_recording_asks_nothing(self):
        frozen = snapshot.take()
        self.assertEqual(gate.pending_golden_findings(self.lock, self.cfg, self.records, frozen), [])
        self.assertEqual(snapshot.recorded_pending(frozen.manifest).golden, {})

    def test_a_cost_file_without_its_golden_file_is_a_finding(self):
        self.record_pending()
        golden.golden_path(PARSE).unlink()
        support.vouch()
        frozen = snapshot.take()
        found = gate.pending_golden_findings(self.lock, self.cfg, self.records, frozen)
        self.assertEqual([finding.code for finding in found], ["golden-missing"])


class PendingSourcesStayOutOfTheTiers(unittest.TestCase):
    def test_a_tier_that_lists_a_pending_source_is_refused(self):
        text = paths.GATE_FILE.read_text(encoding="utf-8")
        old = 'sources = ["ftxi", "notation"]'
        self.assertIn(old, text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gate.toml"
            path.write_text(text.replace(old, f'sources = ["ftxi", "notation", "{PARSE}"]', 1), encoding="utf-8")
            with self.assertRaises(HarnessError):
                config.load(path)

    def test_full_strength_is_still_every_enabled_source(self):
        tier = config.load().tier("release")
        self.assertTrue(tier.full_strength)
        without = dataclasses.replace(tier, sources=tuple(name for name in tier.sources if name != "notation"))
        self.assertFalse(without.full_strength)


if __name__ == "__main__":
    unittest.main()
