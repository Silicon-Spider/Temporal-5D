"""What the harness reads out of a game record: which game it declares itself to be.

The harness treats a record as opaque text. Each oracle reads it with its own
parser, and so will the engine (decision H28). The helpers here exist for one
reason: an oracle handed a board variant, a start position or a promotion rule
it does not support replays some other game and answers confidently about that,
so the harness has to know what a record declares before deciding whom to ask.

They are not rules logic and not a 5dpgn parser. A header is looked for
leniently - any case, any spacing, anywhere - because the question is "what does
this record say it is", not "is this record well formed". Whether a particular
oracle would read the same header the same way is that oracle's adapter's
business.
"""

from __future__ import annotations

import re

# A 5DFEN board string: [<pieces>:<timeline>:<turn>:<w|b>]. Looked for anywhere
# in the record - on a line of its own or not, followed by a comment or not -
# because every reader that finds one sets up that position instead of the
# variant's. (A header, [Key "value"], cannot match: it holds a space and quotes.)
_FEN_BOARD = re.compile(rb"\[[A-Za-z0-9/*+]+:[+-]?\d+:\d+:[wbWB]\]")


def header(data: bytes, name: str) -> str | None:
    """The value of the record's first ``[Name "value"]`` header, or None."""
    pattern = re.compile(rb"\[\s*" + re.escape(name.encode()) + rb'\s+"([^"]*)"\s*\]', re.I)
    match = pattern.search(data)
    return match.group(1).decode("utf-8", "replace").strip() if match else None


def board_header(data: bytes) -> str | None:
    """The record's Board header, if it has one."""
    return header(data, "Board")


def declared_board(data: bytes) -> str | None:
    """The variant the record says it is played on.

    The Board header; failing that, a Variant header, which ftxi reads in its
    place. None when the record names no variant (the notation's default is
    Standard).
    """
    return board_header(data) or header(data, "Variant")


def fen_board_count(data: bytes) -> int:
    """How many 5DFEN board strings the record carries."""
    return len(_FEN_BOARD.findall(data))


def has_fen_boards(data: bytes) -> bool:
    """Whether the record sets up its own starting position with 5DFEN board strings."""
    return _FEN_BOARD.search(data) is not None
