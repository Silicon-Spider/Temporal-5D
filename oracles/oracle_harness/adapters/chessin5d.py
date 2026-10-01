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
event a "limit" naming what stopped it. The driver's header says how each is
obtained.
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


class ChessIn5dSource(Source):
    name = "chessin5d"
    oracle = "chessin5d"
    answers = ("state", "side", "in_check", "terminal", "legality")
    # The driver speaks the common flags (--explain, --actions K), so the base
    # class's diagnostic_args() asks for these.
    diagnostics = ("explain", "actions")

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

    def signature(self, options: RunOptions) -> dict:
        return {"pin": self.pin.commit, "mate_heap_mb": _MATE_HEAP_MB, "driver_sha256": code_sha256(self.driver),
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
        return run_protocol(self.name, record_id, argv, data, options, None)
