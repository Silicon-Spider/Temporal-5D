"""gate.toml: how the gate runs - tiers, budgets, the engine command, boundary policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import paths
from .adapters import ORACLE_SOURCES
from .corpus import TIERS as RECORD_TIERS
from .util import HarnessError, load_toml

SCHEMA = 1
# The lowest share of its recorded answers a tier may let a source get away with.
MIN_COVERAGE_FLOOR = 0.5


@dataclass(frozen=True)
class Tier:
    name: str
    sources: tuple[str, ...]
    record_tiers: tuple[str, ...]
    timeout_s: float
    source_timeout_s: dict[str, float]
    mate_budget_ms: int
    min_coverage: float
    source_min_coverage: dict[str, float]
    min_checked: float
    require_engine: bool

    def timeout_for(self, source: str) -> float:
        return self.source_timeout_s.get(source, self.timeout_s)

    def min_coverage_for(self, source: str) -> float:
        return self.source_min_coverage.get(source, self.min_coverage)

    @property
    def full_strength(self) -> bool:
        """Does this tier run every oracle search over the whole corpus?

        Only such a tier can vouch for the generator; a tier's name promises nothing.
        """
        return set(self.sources) == set(ORACLE_SOURCES) and set(self.record_tiers) == set(RECORD_TIERS)


@dataclass(frozen=True)
class Waiver:
    path: str
    rule: str
    text: str  # the offending text this waiver is for; it excuses nothing else in the file
    reason: str


@dataclass(frozen=True)
class Boundary:
    documentation: tuple[str, ...]
    plumbing: tuple[str, ...]
    manifests: tuple[str, ...]
    copy_window_lines: int
    copy_min_line_chars: int
    copy_long_line_chars: int  # one line this long, identical to an oracle's, is a copy on its own
    waivers: tuple[Waiver, ...]


@dataclass(frozen=True)
class GateConfig:
    cap: int
    engine_command: tuple[str, ...]
    tiers: dict[str, Tier]
    boundary: Boundary

    def tier(self, name: str) -> Tier:
        try:
            return self.tiers[name]
        except KeyError:
            raise HarnessError(f"no tier '{name}' in gate.toml (have: {', '.join(self.tiers)})") from None


def load(path: Path | None = None) -> GateConfig:
    path = path or paths.GATE_FILE
    data = load_toml(path)
    if data.get("schema") != SCHEMA:
        raise HarnessError(f"{path.name}: expected schema = {SCHEMA}")

    cap = int(data.get("observe", {}).get("cap", 0))
    if cap <= 0:
        raise HarnessError(f"{path.name}: [observe] cap must be a positive integer")

    command = data.get("engine", {}).get("command", [])
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise HarnessError(f"{path.name}: [engine] command must be a list of strings (empty while there is no engine)")

    tiers: dict[str, Tier] = {}
    for name, table in (data.get("tier") or {}).items():
        where = f"{path.name} [tier.{name}]"
        sources = tuple(table.get("sources", ()))
        unknown = [source for source in sources if source not in ORACLE_SOURCES]
        if not sources or unknown:
            raise HarnessError(f"{where}: sources must be a non-empty subset of {ORACLE_SOURCES} (unknown: {unknown})")
        record_tiers = tuple(table.get("record_tiers", ()))
        if not record_tiers or any(tier not in RECORD_TIERS for tier in record_tiers):
            raise HarnessError(f"{where}: record_tiers must be a non-empty subset of {RECORD_TIERS}")
        coverage = float(table.get("min_coverage", 1.0))
        source_coverage = {key: float(value) for key, value in table.get("source_min_coverage", {}).items()}
        source_timeouts = {key: float(value) for key, value in table.get("source_timeout_s", {}).items()}
        strays = sorted((set(source_coverage) | set(source_timeouts)) - set(sources))
        if strays:
            raise HarnessError(f"{where}: budgets or floors for sources the tier does not run: {', '.join(strays)}")
        if not all(MIN_COVERAGE_FLOOR <= value <= 1.0 for value in (coverage, *source_coverage.values())):
            # A floor near zero lets a source be listed and never heard: starved
            # of time, it would reproduce nothing and pass.
            raise HarnessError(f"{where}: min_coverage values must lie in {MIN_COVERAGE_FLOOR}..1")
        if not all(value > 0 for value in (float(table["timeout_s"]), *source_timeouts.values())):
            raise HarnessError(f"{where}: timeouts must be positive")
        checked = float(table.get("min_checked", 1.0))
        if not 0.0 < checked <= 1.0:
            # Zero would let a ruleset be gated on an observable no oracle ever answered.
            raise HarnessError(f"{where}: min_checked must be above 0 and at most 1")
        tiers[name] = Tier(
            name=name,
            sources=sources,
            record_tiers=record_tiers,
            timeout_s=float(table["timeout_s"]),
            source_timeout_s=source_timeouts,
            mate_budget_ms=int(table.get("mate_budget_ms", 5000)),
            min_coverage=coverage,
            source_min_coverage=source_coverage,
            min_checked=checked,
            require_engine=bool(table.get("require_engine", False)),
        )
    if not tiers:
        raise HarnessError(f"{path.name}: no [tier.*] tables")

    raw = data.get("boundary", {})
    waivers = []
    for number, item in enumerate(raw.get("waiver", ()), 1):
        try:
            waiver = Waiver(str(item["path"]), str(item["rule"]), str(item["text"]), str(item["reason"]))
        except KeyError as missing:
            raise HarnessError(f"{path.name}: boundary waiver #{number} is missing {missing}") from None
        if not waiver.reason.strip() or not waiver.text.strip():
            raise HarnessError(
                f"{path.name}: the boundary waiver for {waiver.path} needs the offending `text` it excuses and a `reason`")
        waivers.append(waiver)
    boundary = Boundary(
        documentation=tuple(raw.get("documentation", ())),
        plumbing=tuple(raw.get("plumbing", ())),
        manifests=tuple(raw.get("manifests", ())),
        copy_window_lines=int(raw.get("copy_window_lines", 6)),
        copy_min_line_chars=int(raw.get("copy_min_line_chars", 12)),
        copy_long_line_chars=int(raw.get("copy_long_line_chars", 160)),
        waivers=tuple(waivers),
    )
    if not (boundary.copy_window_lines >= 2 and 0 < boundary.copy_min_line_chars < boundary.copy_long_line_chars):
        raise HarnessError(f"{path.name}: [boundary] needs copy_window_lines >= 2 and "
                           "0 < copy_min_line_chars < copy_long_line_chars")
    return GateConfig(cap=cap, engine_command=tuple(command), tiers=tiers, boundary=boundary)
