"""gate.toml: how the gate runs - tiers, budgets, stability policy, the rulebook pin, the engine command, boundary policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .adapters import ORACLE_SOURCES
from .adapters.engine import MODES as ENGINE_MODES
from .corpus import TIERS as RECORD_TIERS
from .util import HarnessError, load_toml

SCHEMA = 1
# Keys a tier no longer reads. Each was a tolerance - a share of the recorded
# answers a run could leave unreproduced - and the gate now requires every value
# of the tier's required set instead (H32). A file that still sets one is refused,
# so that nobody believes a floor is in force.
_REMOVED_TIER_KEYS = {
    "min_coverage": "the gate requires 100% of each tier's required set ([stability], H32)",
    "source_min_coverage": "the gate requires 100% of each tier's required set ([stability], H32)",
}


@dataclass(frozen=True)
class Tier:
    name: str
    sources: tuple[str, ...]
    record_tiers: tuple[str, ...]
    timeout_s: float
    source_timeout_s: dict[str, float]
    mate_budget_ms: int
    min_checked: float
    # Fails unless an engine came through a named variant. Only on a full-strength tier: any other gates nothing (H47).
    require_engine: bool

    def timeout_for(self, source: str) -> float:
        return self.source_timeout_s.get(source, self.timeout_s)

    @property
    def full_strength(self) -> bool:
        """Does this tier run every oracle search over the whole corpus?

        Only such a tier can vouch for the generator; a tier's name promises nothing.
        """
        return set(self.sources) == set(ORACLE_SOURCES) and set(self.record_tiers) == set(RECORD_TIERS)

    @property
    def can_enable_proof_channel(self) -> bool:
        """A tier that requires an engine and runs every oracle search over the whole corpus."""
        return self.require_engine and self.full_strength


@dataclass(frozen=True)
class Stability:
    """H32: which recorded answers each tier requires."""

    # A recorded value is required in a tier when its recorded cost times this
    # fits the tier's budget for its source. At least 1, so that a tier never
    # requires an answer that came later than its own deadline on the recording machine.
    margin: float
    # The tier whose budgets the golden files are recorded under.
    recording_tier: str
    # For the tier that can enable the proof channel: the least share of each
    # source's stable recorded values its required set may hold.
    proof_share: float


@dataclass(frozen=True)
class Calibration:
    """The fixed workload that says how fast a machine is, against the one the golden files were recorded on."""

    source: str
    record: str
    positions: int  # the first this many positions of the record's main line


@dataclass(frozen=True)
class Rulebook:
    """The rulebook and the hash of the text it is pinned at (decision H38; rulebook.py)."""

    path: str  # from the repository root, with forward slashes: "docs/rules/README.md"
    sha256: str  # of its text, CRLF folded to LF


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
    stability: Stability
    calibration: Calibration
    rulebook: Rulebook
    engine_mode: str = "full"

    def tier(self, name: str) -> Tier:
        try:
            return self.tiers[name]
        except KeyError:
            raise HarnessError(f"no tier '{name}' in gate.toml (have: {', '.join(self.tiers)})") from None

    @property
    def recording(self) -> Tier:
        """The tier whose budgets the golden files are recorded under."""
        return self.tier(self.stability.recording_tier)


def _number(table: dict, key: str, where: str, default: float | None = None) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(f"{where}: {key} must be a number")
    return float(value)


def _stability(data: dict, tiers: dict[str, Tier], name: str) -> Stability:
    table = data.get("stability")
    where = f"{name} [stability]"
    if not isinstance(table, dict):
        raise HarnessError(f"{where}: missing - it sets the margin and the recording tier of the required sets (H32)")
    margin = _number(table, "margin", where)
    if margin < 1.0:
        # Below 1 a tier could require an answer that arrived after its own
        # deadline on the recording machine: one no run of it can reproduce.
        raise HarnessError(f"{where}: margin must be at least 1")
    recording = str(table.get("recording_tier", ""))
    if recording not in tiers:
        raise HarnessError(f"{where}: recording_tier must name a tier (have: {', '.join(tiers)})")
    if not set(ORACLE_SOURCES) <= set(tiers[recording].sources):
        raise HarnessError(f"{where}: the recording tier '{recording}' must run every oracle source")
    share = _number(table, "proof_share", where)
    if not 0.0 < share <= 1.0:
        raise HarnessError(f"{where}: proof_share must be above 0 and at most 1")
    return Stability(margin=margin, recording_tier=recording, proof_share=share)


def _calibration(data: dict, name: str) -> Calibration:
    table = data.get("calibration")
    where = f"{name} [calibration]"
    if not isinstance(table, dict):
        raise HarnessError(f"{where}: missing - it names the workload that measures a machine's speed")
    source, record = str(table.get("source", "")), str(table.get("record", ""))
    positions = table.get("positions")
    if source not in ORACLE_SOURCES:
        raise HarnessError(f"{where}: source must be one of {ORACLE_SOURCES}")
    if not record:
        raise HarnessError(f"{where}: record must name a corpus record")
    if isinstance(positions, bool) or not isinstance(positions, int) or positions < 1:
        raise HarnessError(f"{where}: positions must be a whole number of at least 1")
    return Calibration(source=source, record=record, positions=positions)


def _rulebook(data: dict, name: str) -> Rulebook:
    table = data.get("rulebook")
    where = f"{name} [rulebook]"
    if not isinstance(table, dict):
        raise HarnessError(f"{where}: missing - it pins the rulebook the known divergences cite (H38)")
    path, digest = table.get("path"), table.get("sha256")
    parts = path.split("/") if isinstance(path, str) else []
    if not parts or "\\" in path or path.startswith("/") or ":" in parts[0] or any(part in ("", ".", "..") for part in parts):
        raise HarnessError(f"{where}: path must name a file in the repository, from its root, with forward slashes")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HarnessError(f"{where}: sha256 must be the 64 lower-case hex digits of the rulebook's SHA-256")
    return Rulebook(path=path, sha256=digest)


def load(path: Path | None = None) -> GateConfig:
    path = path or paths.GATE_FILE
    data = load_toml(path)
    if data.get("schema") != SCHEMA:
        raise HarnessError(f"{path.name}: expected schema = {SCHEMA}")

    cap = int(data.get("observe", {}).get("cap", 0))
    if cap <= 0:
        raise HarnessError(f"{path.name}: [observe] cap must be a positive integer")

    engine = data.get("engine", {})
    command = engine.get("command", [])
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise HarnessError(f"{path.name}: [engine] command must be a list of strings (empty while there is no engine)")
    mode = engine.get("mode", "full")
    if mode not in ENGINE_MODES:
        raise HarnessError(f"{path.name}: [engine] mode must be one of {ENGINE_MODES}")

    tiers: dict[str, Tier] = {}
    for name, table in (data.get("tier") or {}).items():
        where = f"{path.name} [tier.{name}]"
        removed = [key for key in _REMOVED_TIER_KEYS if key in table]
        if removed:
            raise HarnessError(f"{where}: {removed[0]} is no longer read - {_REMOVED_TIER_KEYS[removed[0]]}")
        sources = tuple(table.get("sources", ()))
        unknown = [source for source in sources if source not in ORACLE_SOURCES]
        if not sources or unknown:
            raise HarnessError(f"{where}: sources must be a non-empty subset of {ORACLE_SOURCES} (unknown: {unknown})")
        record_tiers = tuple(table.get("record_tiers", ()))
        if not record_tiers or any(tier not in RECORD_TIERS for tier in record_tiers):
            raise HarnessError(f"{where}: record_tiers must be a non-empty subset of {RECORD_TIERS}")
        source_timeouts = {key: float(value) for key, value in table.get("source_timeout_s", {}).items()}
        strays = sorted(set(source_timeouts) - set(sources))
        if strays:
            raise HarnessError(f"{where}: budgets for sources the tier does not run: {', '.join(strays)}")
        if not all(value > 0 for value in (_number(table, "timeout_s", where), *source_timeouts.values())):
            raise HarnessError(f"{where}: timeouts must be positive")
        mate_budget = table.get("mate_budget_ms", 5000)
        if isinstance(mate_budget, bool) or not isinstance(mate_budget, int) or mate_budget <= 0:
            raise HarnessError(f"{where}: mate_budget_ms must be a positive whole number")
        checked = float(table.get("min_checked", 1.0))
        if not 0.0 < checked <= 1.0:
            # Zero would let a ruleset be gated on an observable no oracle ever answered.
            raise HarnessError(f"{where}: min_checked must be above 0 and at most 1")
        tiers[name] = Tier(
            name=name,
            sources=sources,
            record_tiers=record_tiers,
            timeout_s=_number(table, "timeout_s", where),
            source_timeout_s=source_timeouts,
            mate_budget_ms=mate_budget,
            min_checked=checked,
            require_engine=bool(table.get("require_engine", False)),
        )
        if tiers[name].require_engine and not tiers[name].full_strength:
            # Such a tier requires an engine to come through a named variant, and a tier that does not run every
            # oracle search over the whole corpus is a smoke test that gates nothing (H47): no engine could pass it.
            raise HarnessError(f"{where}: require_engine needs a tier that runs every oracle source over every record "
                               "tier - any other tier is a smoke test that gates nothing (H47), so no engine could "
                               "come through a named variant in it")
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
    return GateConfig(cap=cap, engine_command=tuple(command), tiers=tiers, boundary=boundary,
                      stability=_stability(data, tiers, path.name), calibration=_calibration(data, path.name),
                      rulebook=_rulebook(data, path.name), engine_mode=mode)
