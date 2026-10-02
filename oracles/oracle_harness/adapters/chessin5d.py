"""Chess in 5D (5d-chess-js): check verdicts, mate verdicts and move legality
from the implementation that shares no code or method with the other two.

Each record runs in its own Node process, and inside it the library's mate
search runs in a worker with its own heap limit: on wide positions that search
exhausts memory well inside its time budget. Exhausting either costs one unknown
verdict; positions answered before a time-out of the whole record are kept.

The driver is told which variant the record declares - its Board header, or its
Variant header when it has none - and refuses the record if the library would
set up any other game for it. A Timeline header, which the library does not read,
and two notations its move-text reader gets wrong without noticing are refused
here, before it runs.

At every position the driver also says what the library holds there (decision
H30): every board, as the library's own 5DFEN export writes it - the string its
full-state hash is taken of - with the library's own timeline and turn labels
and unmoved markers, and that hash itself (chess.hash), which the harness's
fingerprint must equal (base.py). A custom set-up whose boards the library names
at other coordinates than the record's is refused, like any other misreading.
When `observe` asks, it adds two diagnostics, never compared: what the library's
check test advanced and the captures it found (--explain), and the legal actions
at one position as the library's own enumerator lists them, kept where the
library's move() plays each move and its check and present tests pass the result
- the tests submit() applies to an action; not its mate search, which submit()
runs first (--actions K). A mate search that did not finish leaves its position
event a "limit" naming what stopped it, and no verdict: a mate verdict is a proof,
and an exhausted budget is unknown, never mate (H37) - a position that names a
limit and gives a verdict anyway makes the run malformed. The driver's header
says how each is obtained.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .. import build, fingerprint, headers, paths
from ..lockfile import Lock
from ..observations import Observation
from . import base
from .base import RunOptions, Source, code_sha256, run_protocol

_MATE_HEAP_MB = 1024
_COMMENT = re.compile(rb"\{[^{}]*\}")
# The library splits move text on turn numbers written "12." only. It reads a
# "12w." or "12b." marker as part of a move, drops or merges actions around it,
# and carries on; it replays a parenthesised variation as part of the main line.
_COLOURED_TURN = re.compile(rb"(?<![\w.])\d+\s*[wbWB]\s*\.")
_VARIATION = re.compile(rb"\(\s*\d+\s*[wbWB]?\s*\.")


def would_misread(data: bytes) -> tuple[str, str] | None:
    """(reason, detail) when the library would replay another game than the record's.

    The variant and any 5DFEN boards are checked by the driver, which can ask the
    library what it set up. What is refused here the library never reports.
    """
    if headers.header(data, "Timeline") is not None:
        return "timeline", "the record carries a Timeline header, which the library does not read"
    text = _COMMENT.sub(b" ", data)
    if _VARIATION.search(text):
        return "notation", "the record holds a parenthesised variation, which the library replays as part of the main line"
    if _COLOURED_TURN.search(text):
        return "notation", 'the record numbers turns as "12w." / "12b.", which the library does not split actions on'
    return None


def verdict_beyond_its_limit(observation: Observation) -> str | None:
    """What is wrong with a run that gives a mate verdict where its mate search ran into a limit, if it does.

    A mate verdict is a proof - an exhaustive search that terminated - and an
    exhausted budget is "unknown", never "mate" (decision H37). The library's own
    search answers checkmate when it runs out of time; the driver turns that into
    a null and names the limit. A position that names a limit and still gives a
    verdict (checkmate, stalemate, or "none") is a claim nothing proved: the run
    is malformed, and nothing in it is compared or recorded.
    """
    for position in observation.positions:
        limit = observation.extras.get(position.index, {}).get("limit")
        if limit is not None and position.terminal is not None:
            return (f"position {position.index}: a mate verdict ({position.terminal}) beside the limit its mate search "
                    f"ran into ({limit}) - an exhausted budget is unknown, never a verdict (H37)")
    return None


class ChessIn5dSource(Source):
    name = "chessin5d"
    oracle = "chessin5d"
    answers = ("state", "side", "in_check", "terminal", "legality")
    # The driver speaks the common flags (--explain, --actions K), so the base
    # class's diagnostic_args() asks for these.
    diagnostics = ("explain", "actions")
    # Its mate verdicts are bounded by the mate search's budget and heap as well as by the run's deadline.
    budget_fields = ("timeout_s", "mate_budget_ms")

    def __init__(self, lock: Lock):
        self.pin = lock.pin("chessin5d")
        self.driver = paths.DRIVERS / "chessin5d" / "driver.js"

    def _build(self) -> build.Build | None:
        return build.load(self.pin)

    def available(self) -> str | None:
        if not self._build():
            return "chessin5d is not built (run: python oracles/harness.py build --oracle chessin5d)"
        if not shutil.which("node"):
            return "node is not on PATH"
        return None

    def budget(self, options: RunOptions) -> dict:
        return {**super().budget(options), "mate_heap_mb": _MATE_HEAP_MB}

    def limits(self, observation: Observation, index: int, options: RunOptions) -> dict[str, str]:
        """The driver names what stopped a mate search that did not finish ("limit"): its terminal is unknown at that limit.

        The library's own time-out flag and the driver's deadline past it are
        both the mate search's time budget; "mate_heap_mb" is the worker's heap.
        """
        if index >= len(observation.positions) or observation.positions[index].terminal is not None:
            return {}
        limit = observation.extras.get(index, {}).get("limit")
        if limit in ("mate_budget_ms", "deadline"):
            return {"terminal": f"mate_budget_ms={base.number(options.mate_budget_ms)}"}
        if limit == "mate_heap_mb":
            return {"terminal": f"mate_heap_mb={_MATE_HEAP_MB}"}
        return {"terminal": str(limit)} if isinstance(limit, str) and limit else {}

    def signature(self, options: RunOptions) -> dict:
        return {"pin": self.pin.commit, "driver_sha256": code_sha256(self.driver),
                "adapter_sha256": code_sha256(Path(__file__), Path(base.__file__), Path(fingerprint.__file__),
                                              Path(headers.__file__))}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        refused = would_misread(data)
        if refused:
            return Observation(source=self.name, record=record_id, status="unsupported", reason=refused[0],
                               detail=refused[1], actions_accepted=0)
        declared = headers.declared_board(data) or "Standard"
        argv = [shutil.which("node"), self.driver, self._build().artifact("root"), "--board", declared,
                "--mate-budget-ms", str(options.mate_budget_ms), "--mate-heap-mb", str(_MATE_HEAP_MB),
                *self.diagnostic_args(options)]
        observation = run_protocol(self.name, record_id, argv, data, options, None)
        problem = verdict_beyond_its_limit(observation)
        if problem:
            # The positions are kept for whoever debugs the driver; the status makes sure nothing is concluded from them.
            observation.status, observation.reason, observation.detail = "error", "malformed_output", problem
        return observation
