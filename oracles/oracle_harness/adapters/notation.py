"""The parser bundled with the notation specification (adri326/5dchess-notation):
a parse-only oracle (decision H26).

It reads a record and applies each move as written; it has no notion of check,
legality or the end of a game. So it contributes what a parser can: at every
position the boards it holds (the state event, H30), the side it reads as moving
next, and how many actions it read - never a check, count or mate answer, and
never a verdict on a move. A move it cannot apply is a failure to read the move
text (a soft rejection), and its acceptance of an action is not a legality
verdict: it would accept an action that leaves a king in check.

Each record runs in its own Node process, through oracles/drivers/notation,
which watches the parser's one entry point (pgn.parse) and checks, after the
parser's own set-up, that it set up the game the record declares. Refused here,
before it runs, is what the parser cannot be trusted with:

  * a comment that is never closed - its tokeniser loops forever on one;
  * a Timeline or Promotions header, which it does not read;
  * a variant it does not know (it would set up one empty board), and a named
    layout set up on boards of another size: it takes the size from the Size
    header, or 8x8, never from its own layout;
  * 5DFEN board strings under any Board but "custom" (it ignores them there),
    and Board "custom" with no board string.

The driver repeats the variant, size and 5DFEN checks against what the parser
actually set up, which this module can only predict from the parser's table of
named boards (read at build time: build.notation_boards).
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

# The size of a board as the parser reads a Size header: "<width>x<height>".
_SIZE = re.compile(r"\s*(\d+)\s*x\s*(\d+)\s*")


def _size(text: str) -> tuple[int, int] | None:
    match = _SIZE.fullmatch(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def would_misread(data: bytes, boards: dict[str, str]) -> tuple[str, str] | None:
    """(reason, detail) when the parser would not read this record as the game it declares.

    ``boards`` is the parser's table of named boards: the upper-case name it
    looks a Board header up under, and the size of that layout ("5x5").
    """
    if data.rfind(b"{") > data.rfind(b"}"):
        return "notation", "the record holds a comment that is never closed, on which the parser's tokeniser loops forever"
    for name in ("Timeline", "Promotions"):
        if headers.header(data, name) is not None:
            return name.lower(), f"the record carries a {name} header, which the parser does not read"
    declared = headers.declared_board(data) or "Standard"
    if declared.lower() == "custom":
        if not headers.has_fen_boards(data):
            return "fen", 'the record\'s Board is "custom" but it carries no 5DFEN board string: the parser would set up no board'
        return None
    layout = boards.get(declared.upper())
    if layout is None:
        return "variant", f'the record\'s variant, "{declared}", is not one the parser knows: it would set up a single empty board'
    if headers.has_fen_boards(data):
        return "fen", (f'the record carries 5DFEN boards, which the parser reads only under Board "custom": '
                       f'it would replay its own layout of "{declared}"')
    size = headers.header(data, "Size")
    wanted = _size(layout)
    if wanted is None or _size(size or "8x8") != wanted:
        written = f'its Size header says "{size}"' if size is not None else "it has no Size header (the parser then takes 8x8)"
        return "size", (f'the parser sets its "{declared}" layout ({layout}) up on boards of the size the Size header gives, '
                        f"and {written}")
    return None


class NotationSource(Source):
    name = "notation"
    oracle = "notation"
    # What it set up, the side it reads as moving next, and how many actions it
    # read. Not "legality": it accepts any action it can apply (PARSE_ONLY_SOURCES).
    answers = ("state", "side", "length")
    diagnostics = ("explain",)

    def __init__(self, lock: Lock):
        self.pin = lock.pin("notation")
        self.driver = paths.DRIVERS / "notation" / "driver.js"

    def _build(self) -> build.Build | None:
        return build.load(self.pin)

    def available(self) -> str | None:
        if not self._build():
            return "notation is not built (run: python oracles/harness.py build --oracle notation)"
        if not shutil.which("node"):
            return "node is not on PATH"
        return None

    def signature(self, options: RunOptions) -> dict:
        return {"pin": self.pin.commit, "driver_sha256": code_sha256(self.driver),
                "adapter_sha256": code_sha256(Path(__file__), Path(base.__file__), Path(fingerprint.__file__),
                                              Path(headers.__file__))}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        built = self._build()
        refused = would_misread(data, built.info["boards"])
        if refused:
            return Observation(source=self.name, record=record_id, status="unsupported", reason=refused[0],
                               detail=refused[1], actions_accepted=0)
        declared = headers.declared_board(data) or "Standard"
        argv = [shutil.which("node"), self.driver, built.artifact("root"), "--board", declared,
                *self.diagnostic_args(options)]
        return run_protocol(self.name, record_id, argv, data, options, None)
