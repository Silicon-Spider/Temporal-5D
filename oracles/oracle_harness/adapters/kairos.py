"""Kairos (FirePlank/5d-chess-engine): the fourth rules oracle (decision H52).

Two of its action searches are exposed as separate sources, as ftxi's and cwmtt's
are: its hypercuboid search ("kairos") and its exhaustive enumerator
("kairos:naive"). What each answers was confirmed at the pin before the driver was
written (PROVENANCE.md, "Kairos, the fourth oracle"):

- both: the check (its own test, which passes the active boards in the present),
  what it set up at every position (decision H30) and how it replayed the record;
- "kairos": the mate verdict of its own mate test, an exhaustive hypercuboid
  search - and **no count**. That search leaves out the actions that waste a move
  (and yields one action per allowed promotion piece), so the number of actions it
  yields is not the gate's count, and the driver reports none;
- "kairos:naive": the count - its exhaustive enumerator lists every legal action,
  its branching moves in every order, one action per promoting move - and the
  verdict where it finds no action (checkmate in check, else stalemate: the rule
  its own mate test applies).

Kairos's 5dpgn reader resolves the whole record before it returns and fails all of
it on one action it cannot match or submit, variations included: such a record is
reported unsupported from position 0 (reason "parse_failed"), never as a refusal.
Its reader does not test royal safety; the driver replays the main line through
Kairos's own referee (the one its game loader uses), so an action that leaves a
royal capturable is refused there - it cannot be submitted (cannot_submit, as the
other oracles report it) - and the positions before it stand.
A line that ends in variations is replayed with its last variation as the main
line - its reader is a port of ftxi's - so ftxi's guard refuses such a record here
too, before it runs.

These sources are wired - built, observed, tested - but enabled in no tier
(adapters.PENDING_SOURCES) until their golden files are recorded on a quiet
machine and the user has ruled how far Kairos, derived in part from ftxi, counts
as a second oracle (docs/BUILD_LOG.md, issues 526 and 530).
"""

from __future__ import annotations

from pathlib import Path

from .. import build, fingerprint, paths
from ..lockfile import Lock
from ..observations import Observation
from . import base, ftxi
from .base import RunOptions, Source, code_sha256, run_protocol

_SEARCHES = {"kairos": "hypercuboid", "kairos:naive": "naive"}
_DRIVER = paths.DRIVERS / "kairos"


class KairosSource(Source):
    oracle = "kairos"
    # The driver speaks the common --actions K flag; it has no --explain.
    diagnostics = ("actions",)

    def __init__(self, name: str, lock: Lock):
        self.name = name
        self.search = _SEARCHES[name]
        self.pin = lock.pin("kairos")
        counts = self.search == "naive"
        self.answers = ("state", "side", "in_check", *(("legal_actions",) if counts else ()), "terminal", "legality")

    @property
    def counts(self) -> bool:
        return "legal_actions" in self.answers

    def _build(self) -> build.Build | None:
        return build.load(self.pin)

    def available(self) -> str | None:
        return None if self._build() else "kairos is not built (run: python oracles/harness.py build --oracle kairos)"

    def signature(self, options: RunOptions) -> dict:
        signature = {"pin": self.pin.commit, "search": self.search}
        if self.counts:
            signature["cap"] = options.cap
        return {**signature,
                "driver_sha256": code_sha256(*(_DRIVER / name for name in build.KAIROS_DRIVER_FILES)),
                "adapter_sha256": code_sha256(Path(__file__), Path(base.__file__), Path(fingerprint.__file__),
                                              Path(ftxi.__file__))}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        cap = options.cap if self.counts else None
        if ftxi.ends_in_a_variation(data):
            return Observation(source=self.name, record=record_id, status="unsupported", reason="notation",
                               detail="the record ends with a parenthesised variation, which Kairos's reader "
                                      "(a port of ftxi's) replays as the main line",
                               actions_accepted=0, cap=cap)
        driver = self._build().artifact("driver")
        argv = [driver, "--cap", str(options.cap), "--search", self.search, *self.diagnostic_args(options)]
        return run_protocol(self.name, record_id, argv, data, options, cap)
