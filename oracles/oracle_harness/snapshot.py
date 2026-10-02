"""One verified snapshot of everything a gate run compares against (decision H33).

A gate run reads the golden files, their cost files, the calibration reference
and known-divergences.toml once, when it starts - each recorded-answer file only
after its bytes are shown to match MANIFEST - and hands that snapshot to every
consumer: the static checks, the comparison of fresh oracle runs with the
recorded answers, and the engine differential. Nothing is read from those files
again while the run lasts, so nothing the engine adapter does to them can change
what it is compared with: the divergences that excuse it are the ones on record
when the run began. And every harness file that is compared or executed is
hashed at the start and again at the end (integrity.py), so a change made while
the run lasted fails it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import divergences, golden, integrity, paths
from .adapters import ORACLE_SOURCES
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


def take(sources: tuple[str, ...] | list[str] = ORACLE_SOURCES) -> Snapshot:
    """Hash the harness, then read and verify every recorded-answer file and the known divergences, once."""
    snapshot = Snapshot(harness=integrity.hash_harness())
    snapshot.manifest_problems = integrity.verify_manifest()
    try:
        snapshot.manifest = integrity.read_manifest()
    except integrity.IntegrityError:
        snapshot.manifest = None  # verify_manifest has said why
    read: dict[str, str] = {}  # file -> hash of the bytes the snapshot was made from
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
                snapshot.costs[source] = costs
                read[_relative(golden.cost_path(source))] = costs.digest
            except golden.StaleSchema as error:
                problems.append(Finding("golden-stale", str(error)))
            except integrity.IntegrityError as error:
                problems.append(Finding("golden-integrity", str(error)))
            except HarnessError as error:
                problems.append(Finding("golden-missing", str(error)))
        if problems:
            snapshot.problems[source] = problems
    try:
        snapshot.calibration = golden.load_calibration(snapshot.manifest)
        read[_relative(golden.calibration_path())] = snapshot.calibration.digest
    except HarnessError as error:
        snapshot.calibration_problem = str(error)
    try:
        data = paths.DIVERGENCES_FILE.read_bytes()
    except FileNotFoundError:
        raise HarnessError(f"missing file: {paths.DIVERGENCES_FILE}") from None
    snapshot.divergences = divergences.load(data=data)  # parsed from the very bytes that are hashed
    read[_relative(paths.DIVERGENCES_FILE)] = sha256_text(data)
    # What was read must be what was hashed a moment before: a file changed in
    # between would leave the run comparing against something it never hashed.
    for name, digest in read.items():
        if snapshot.harness.get(name, digest) != digest:
            snapshot.manifest_problems.append(Finding(
                "harness-modified", f"{name} changed while the gate was reading it - run the gate again"))
    return snapshot
