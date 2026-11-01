"""Kairos (FirePlank/5d-chess-engine): the fourth rules oracle (decision H52).

It is enabled in two stages, as H52 orders: "parse mode first, then full mode".

- "kairos:parse", parse mode: what Kairos set up at every position (decision H30),
  the side to move and how it replayed the record - the same replay as the two
  searches below, its refusals included - and a null for every question. Like the
  notation parser it is a parse-only source (adapters.PARSE_ONLY_SOURCES): its
  playing an action is no verdict, and it stands behind no ruleset (H47). Pending,
  like the searches: recorded in the quiet-machine block, enabled after the release
  tier's sign-off run, which leaves Kairos out (CHECKPOINT 10; docs/BUILD_LOG.md,
  issue 574).

Full mode exposes two of its action searches as separate sources, as ftxi's and
cwmtt's are: its hypercuboid search ("kairos") and its exhaustive enumerator
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
reported unsupported from position 0 (reason "parse_failed"), never as a refusal. A
move it cannot play and an action that does not move the present fail there, so
Kairos never gives an illegal_move refusal or a present-not-moved one. Its reader
does not test royal safety; the driver replays the main line through Kairos's own
referee (the one its game loader uses), so an action that leaves a royal capturable
is refused there - it cannot be submitted (cannot_submit, as the other oracles
report it) - and the positions before it stand. That is its only refusal: its
legality answer covers royal safety alone, and never stands behind another
oracle's illegal_move or present-not-moved refusal.
A line that ends in variations is replayed with its last variation as the main
line - its reader is a port of ftxi's - so ftxi's guard refuses such a record here
too, before it runs.

All three are wired - built, observed, tested - and pending (adapters.PENDING_SOURCES).
Kairos is in ftxi's lineage (CHECKPOINT 10): the gate counts the two as one lineage
for H47's two-oracle bar (the lock file's `lineage`, PROVENANCE.md "Lineages").
"""

from __future__ import annotations

from pathlib import Path

from .. import build, fingerprint, paths
from ..lockfile import Lock
from ..observations import Observation
from . import base, ftxi
from .base import RunOptions, Source, code_sha256, run_protocol

_SEARCHES = {"kairos": "hypercuboid", "kairos:naive": "naive", "kairos:parse": "parse"}
_DRIVER = paths.DRIVERS / "kairos"


class KairosSource(Source):
    oracle = "kairos"

    def __init__(self, name: str, lock: Lock):
        self.name = name
        self.search = _SEARCHES[name]
        self.pin = lock.pin("kairos")
        if self.search == "parse":
            # Parse mode answers no question: what it set up, whose move, how it replayed the record. Its refusals
            # are statements (a royal left capturable); its playing an action is none (PARSE_ONLY_SOURCES).
            self.answers = ("state", "side", "legality", "length")
            self.diagnostics = ()
        else:
            counts = self.search == "naive"
            self.answers = ("state", "side", "in_check", *(("legal_actions",) if counts else ()), "terminal",
                            "legality")
            # The driver speaks the common --actions K flag; it has no --explain.
            self.diagnostics = ("actions",)

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
