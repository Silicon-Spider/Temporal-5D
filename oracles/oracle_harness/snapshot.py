"""One verified snapshot of everything a gate run compares against (decision H33).

A gate run reads the golden files, their cost files, the calibration reference,
the engine's baseline (H47) and known-divergences.toml once, when it starts -
each recorded-answer file only after its bytes are shown to match MANIFEST, and
the baseline only after they match its pin in gate.toml - and hands that
snapshot to every consumer: the static checks, the comparison of fresh oracle
runs with the recorded answers, and the engine differential - its comparison
with the engine's baseline included. Nothing is read from those files
again while the run lasts, so nothing the engine adapter does to them can change
what it is compared with: the divergences that excuse it are the ones on record
when the run began. And every harness file that is compared or executed is
hashed at the start and again at the end (integrity.py), so a change made while
the run lasted fails it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import baseline as engine_baseline
from . import config, divergences, golden, integrity, paths
from .adapters import ORACLE_SOURCES, PENDING_SOURCES
from .divergences import Divergence
from .observations import Observation
from .util import Finding, HarnessError, sha256_text


@dataclass
class Snapshot:
    # Every harness file compared or executed -> its hash, when the snapshot was taken.
    harness: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, str] | None = None
    golden: dict[str, golden.Golden] = field(default_factory=dict)
    costs: dict[str, golden.Costs] = field(default_factory=dict)
    calibration: golden.Calibration | None = None
    divergences: list[Divergence] = field(default_factory=list)
    # Why a source's golden or cost file is not in the snapshot (stale, missing, tampered with), by source.
    problems: dict[str, list[Finding]] = field(default_factory=dict)
    # What is wrong with MANIFEST and the golden directory as a whole.
    manifest_problems: list[Finding] = field(default_factory=list)
    calibration_problem: str = ""
    # The engine's baseline (H47), read when the snapshot was asked for it and gate.toml pins one; otherwise None, and
    # why it could not be read where it was asked for.
    baseline: engine_baseline.Baseline | None = None
    baseline_problems: list[Finding] = field(default_factory=list)

    def recorded(self, source: str, record_id: str) -> Observation | None:
        """What the golden file records ``source`` saying about a record, or None."""
        loaded = self.golden.get(source)
        return None if loaded is None else loaded.observations.get(record_id)

    def complete(self, source: str) -> bool:
        """Are this source's golden and cost files both in the snapshot?"""
        return source in self.golden and source in self.costs


def _relative(path) -> str:
    try:
        return path.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_sources(snapshot: Snapshot, sources, read: dict[str, str]) -> None:
    """Read each source's golden and cost files into the snapshot, each verified against its MANIFEST; why one could
    not be read goes into snapshot.problems. ``read`` gets each file read and the hash of its bytes."""
    for source in sources:
        problems: list[Finding] = []
        try:
            loaded = golden.load(source, snapshot.manifest)
            snapshot.golden[source] = loaded
            read[_relative(golden.golden_path(source))] = loaded.digest
        except golden.StaleSchema as error:
            problems.append(Finding("golden-stale", str(error)))
        except integrity.IntegrityError as error:
            problems.append(Finding("golden-integrity", str(error)))
        except HarnessError as error:
            problems.append(Finding("golden-missing", str(error)))
        if source in snapshot.golden:
            try:
                costs = golden.load_costs(source, snapshot.manifest)
                # On calibration.json's time base (a no-op for an enabled source's cost file, which has no probe).
                snapshot.costs[source] = costs.on_reference(snapshot.calibration)
                read[_relative(golden.cost_path(source))] = costs.digest
            except golden.StaleSchema as error:
                problems.append(Finding("golden-stale", str(error)))
            except integrity.IntegrityError as error:
                problems.append(Finding("golden-integrity", str(error)))
            except HarnessError as error:
                problems.append(Finding("golden-missing", str(error)))
        if problems:
            snapshot.problems[source] = problems


def recorded_pending(manifest: dict[str, str] | None, calibration: golden.Calibration | None = None) -> Snapshot:
    """The recordings of the pending sources that have one (adapters.PENDING_SOURCES: wired, in no tier), read on their
    own - never into the snapshot ``take`` gives every tier, verify's checks of the enabled sources and
    DISAGREEMENTS.md, which holds ORACLE_SOURCES only. A pending source's recording is checked (verify) and kept, and
    compared with nothing until the source is enabled. ``manifest`` is the snapshot's: one MANIFEST vouches for all;
    ``calibration`` the snapshot's reference, which a pending recording's costs are read against (Costs.on_reference)."""
    names = [name for name in PENDING_SOURCES if golden.golden_path(name).exists() or golden.cost_path(name).exists()]
    pending = Snapshot(manifest=manifest, calibration=calibration)
    _read_sources(pending, names, {})
    return pending


def take(sources: tuple[str, ...] | list[str] = ORACLE_SOURCES, *, baseline_pin: config.Baseline | None = None
         ) -> Snapshot:
    """Hash the harness, then read and verify every recorded-answer file and the known divergences, once.

    ``baseline_pin`` (gate.toml [baseline]): read the engine's baseline as well, verified against its pin - where it
    pins one.
    """
    snapshot = Snapshot(harness=integrity.hash_harness())
    snapshot.manifest_problems = integrity.verify_manifest()
    try:
        snapshot.manifest = integrity.read_manifest()
    except integrity.IntegrityError:
        snapshot.manifest = None  # verify_manifest has said why
    read: dict[str, str] = {}  # file -> hash of the bytes the snapshot was made from
    # The calibration reference first: a cost file that carries a probe of its own is read on its time base.
    try:
        snapshot.calibration = golden.load_calibration(snapshot.manifest)
        read[_relative(golden.calibration_path())] = snapshot.calibration.digest
    except HarnessError as error:
        snapshot.calibration_problem = str(error)
    _read_sources(snapshot, sources, read)
    try:
        data = paths.DIVERGENCES_FILE.read_bytes()
    except FileNotFoundError:
        raise HarnessError(f"missing file: {paths.DIVERGENCES_FILE}") from None
    snapshot.divergences = divergences.load(data=data)  # parsed from the very bytes that are hashed
    read[_relative(paths.DIVERGENCES_FILE)] = sha256_text(data)
    if baseline_pin is not None and baseline_pin.sha256:
        try:
            snapshot.baseline = engine_baseline.load(baseline_pin.sha256)
            read[_relative(paths.ENGINE_BASELINE_FILE)] = snapshot.baseline.digest
        except engine_baseline.BaselineError as error:
            snapshot.baseline_problems.append(Finding(error.code, str(error)))
    # What was read must be what was hashed a moment before: a file changed in
    # between would leave the run comparing against something it never hashed.
    for name, digest in read.items():
        if snapshot.harness.get(name, digest) != digest:
            snapshot.manifest_problems.append(Finding(
                "harness-modified", f"{name} changed while the gate was reading it - run the gate again"))
    return snapshot
