"""cwmtt's records in the official client's export form, converted to 5dpgn.

One of cwmtt's records, `cwmtt/many2`, was written by the official game client:
every move names its board twice - `(0T1)Pe2(0T1)e3`, `(-6T2)Qd1(-6T2)h5` - beside
jumps written as the notation writes them, `(0T2)Ng1>>(0T1)g3`. A move that names
its board twice is the pinned notation specification's export form
(adri326/5dchess-notation, the `notation` pin, README.md, "Export or minimal
notation" > "Export moves": "the super-physical coordinates of the starting
board", "the piece's letter (optional)", "the physical coordinates of the moving
piece", "the super-physical coordinates of the target board", "the physical
coordinates of the target square"), which the specification itself calls "still
a proposal". Of the gate's sources only cwmtt read it; the engine's parser is
written from the notation (H28), and is not extended to read it.

The user's ruling at CHECKPOINT 3: "the cwmtt driver converts the export form to
the canonical 5dpgn form from the pinned notation spec; if it expresses something
the spec can't, mark the record unsupported. The engine's parser is not
extended." Converting only what cwmtt reads would change nothing (cwmtt reads
either form), so the side of the harness that supplies cwmtt's records converts
them (corpus.feed), and the converted text is what every source reads, the
engine included. A record with a move the specification cannot express is given
to no source: every source is told it is unsupported, with the reason
(runner.observe) - it is never partly converted.

This is notation, not play. Nothing here reads, guesses or checks a position -
which piece stands where, whose move it is, what a move captures - and nothing
here is a 5dpgn parser: it finds one textual pattern and rewrites it.

What is rewritten. An export move whose two boards are the same board is a
physical move, and loses its second board:

    (<L>T<T>)[piece]<from>(<L>T<T>)<to>   ->   (<L>T<T>)[piece]<from><to>

- "Moves" > "Physical moves": a move on one board is written as in standard
  algebraic notation, and "must be preceded by their board's coordinates if there
  is more than one timeline" - the first board stays, as written.
- "Notes" > "Omission": "As in SAN, parts of the source square's coordinates may
  be omitted", and in 5D "the source square's coordinates should still be
  specified": the piece's square written in full is the move with nothing
  omitted (the specification's own game writes `(3T11)Qg7g6*!`).
- "Castling": the export form writes castling as the king's two-square move,
  `(0T6)Ke1(0T6)g1` ("Export moves"); it becomes `(0T6)Ke1g1`, the form castling
  takes in the specification, `K<file_from><rank_from><file_to><rank_to>`, of
  which `O-O` is "a shortcut".
- The piece's letter is kept as written, `P` for a pawn included ("Moves" lists
  `P`; "Jumps" writes `(0T6)Pd5>(1T6)d5`), and stays absent where the export
  left it out.
- Nothing is added. The export form marks no capture, check or promotion, and
  deciding one would mean reading the position. A suffix written after the
  target square - a promotion `=Q` ("Promotions"), a check mark `+`, `*` or `#`,
  `!` or `?`, an annotation token `(~T2)` - is copied as it stands.

Two boards are the same board when they name the same timeline and turn
("Coordinates": `(L<a> T<b>)` or `(<a>T<b>)`; "Notes" > "Even-numbered starting
boards": the `+` sign "can be omitted for other integers" than `-0` and `+0`, so
`(+1T2)` and `(1T2)` are one board, while `-0`, `+0` and `0` are three labels).

What makes a record unsupported (Conversion.refused):

- an export move between two different boards. The specification writes a move
  to another board as a jump, with `>` or `>>` ("Jumps": "`>>` if the jump is
  branching, `>` if the jump isn't branching") - and which of the two is a
  question about the position: whether the target board is one the mover may
  play on;
- any other move text that names a board straight after a square, the export
  form's hallmark, without being an export move as "Export moves" defines it -
  a capture mark before the second board, a variation's bracket glued to the
  target square, an export move glued to the end of another move;
- the export form's own turn tokens, `w.` and `b.` ("Export turns"): only moves
  are converted, and the specification's turns are numbered ("Turns").

What is never touched: headers and 5DFEN board strings (`[...]`, "Headers",
"5DFEN and custom variants"), comments (`{...}`, "Comments"), turn numbers,
separators and white space, and every move the specification already writes so,
jumps included. A record without the hallmark comes back unchanged, byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# What the conversion never looks inside: a header or 5DFEN board string, and a
# comment ("Comments": "Nested comments are not supported"). One left open runs
# to the end of the record.
_OPAQUE = re.compile(rb"\[[^\]]*(?:\]|\Z)|\{[^}]*(?:\}|\Z)")


def _board(name: str = "") -> bytes:
    """A board's super-physical coordinates ("Coordinates"): (L<a> T<b>) or (<a>T<b>)."""
    timeline = rb"(?P<%s_timeline>" % name.encode() if name else b"(?:"
    turn = rb"(?P<%s_turn>" % name.encode() if name else b"(?:"
    return rb"\((?:L)?" + timeline + rb"[+-]?[0-9]+)\s*T" + turn + rb"[0-9]+)\)"


# A square ("Coordinates": a1 is the bottom-left corner) - a file letter and a rank.
_SQUARE = rb"[a-z][1-9][0-9]*"
# The pieces' letters ("Moves"), the two-letter ones first.
_PIECE = rb"RQ|PR|[PWKCQYSNRBUD]"

# An export move ("Export moves"). It starts a move - it is not glued to a letter,
# a digit or another board - and ends where the move does: at white space or the
# end of the record, or where a suffix the specification writes after a move
# begins (=, +, *, #, ~, !, ?, an annotation token or a comment) or the
# separator of the two players' moves.
_EXPORT_MOVE = re.compile(
    rb"(?<![A-Za-z0-9>)\]])"
    rb"(?P<origin>" + _board("origin") + rb")"
    rb"(?P<piece>" + _PIECE + rb")?"
    rb"(?P<source>" + _SQUARE + rb")"
    rb"(?P<target>" + _board("target") + rb")"
    rb"(?P<square>" + _SQUARE + rb")"
    rb"(?=[\s=+*#~!?({/]|\Z)"
)
# The export form's hallmark: a board named straight after a square (with at most
# a capture mark between). The specification never writes that: after a piece's
# square comes the target square, or `>` / `>>` and then a board ("Jumps").
_HALLMARK = re.compile(_SQUARE + rb"x?" + _board())
# The export form's turn tokens ("Export turns"): `w.` and `b.` without a number.
_EXPORT_TURN = re.compile(rb"(?<![^\s/])[wb]\.(?=\s|\Z)")


@dataclass(frozen=True)
class Conversion:
    """What converting one record gives."""

    # What every source reads: the converted record - or the record itself,
    # unchanged, where it is not in the export form or is refused.
    text: bytes
    moves: int = 0  # export moves rewritten
    # Why the record cannot be given to any source, quoting the move text that
    # has no form in the specification; empty when it can.
    refused: str = ""

    @property
    def export_form(self) -> bool:
        """Is the record in the export form - converted, or refused?"""
        return self.moves > 0 or bool(self.refused)


def _timeline(label: bytes) -> tuple[str, int]:
    """A timeline label as a value: `+1` and `1` are one timeline, `-0`, `+0` and `0` three."""
    value = int(label)
    if value == 0:
        return (label[:1].decode() if label[:1] in (b"+", b"-") else "", 0)
    return ("", value)


def _same_board(match: re.Match) -> bool:
    return (_timeline(match["origin_timeline"]) == _timeline(match["target_timeline"])
            and int(match["origin_turn"]) == int(match["target_turn"]))


def _token(text: bytes, start: int, end: int) -> str:
    """The white-space-delimited token around text[start:end], as the reason quotes it."""
    while start > 0 and not text[start - 1:start].isspace():
        start -= 1
    while end < len(text) and not text[end:end + 1].isspace():
        end += 1
    return text[start:end].decode("utf-8", "replace")


def _move_text(data: bytes) -> list[tuple[int, int]]:
    """The spans of the record outside its headers, board strings and comments."""
    spans, start = [], 0
    for match in _OPAQUE.finditer(data):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(data):
        spans.append((start, len(data)))
    return spans


def convert(data: bytes) -> Conversion:
    """``data`` (a record, line endings folded to LF) with its export moves in the specification's form.

    Returns the record unchanged where it has no export move. Where one of its
    moves has no form in the specification, the record is refused: ``text`` is
    the record unchanged, and ``refused`` says why - nothing is half converted.
    """
    spans = _move_text(data)
    if not any(_HALLMARK.search(data, start, end) for start, end in spans):
        return Conversion(text=data)
    problems: list[tuple[int, str]] = []  # (offset, why), the first in the record wins
    rewrites: list[tuple[int, int, bytes]] = []  # (start, end, replacement)
    for start, end in spans:
        covered: list[tuple[int, int]] = []
        for match in _EXPORT_MOVE.finditer(data, start, end):
            covered.append(match.span())
            if _same_board(match):
                origin, piece, source, square = match.group("origin", "piece", "source", "square")
                rewrites.append((*match.span(), origin + (piece or b"") + source + square))
            else:
                problems.append((match.start(), (
                    f"'{_token(data, *match.span())}' moves from one board to another without '>' or '>>': the "
                    "notation specification writes such a move as a jump, branching or not (\"Jumps\"), and which "
                    "it is depends on the position, which the conversion does not read")))
        for match in _HALLMARK.finditer(data, start, end):
            if not any(left <= match.start() and match.end() <= right for left, right in covered):
                problems.append((match.start(), (
                    f"'{_token(data, *match.span())}' names a board straight after a square, but is not an export "
                    "move (\"Export or minimal notation\", \"Export moves\"): the notation specification has no form "
                    "for it")))
        for match in _EXPORT_TURN.finditer(data, start, end):
            problems.append((match.start(), (
                f"'{_token(data, *match.span())}' is a turn of the export form (\"Export turns\"): only moves are "
                "converted, and the notation specification numbers its turns (\"Turns\")")))
    if problems:
        return Conversion(text=data, refused=min(problems)[1])
    pieces, at = [], 0
    for start, end, replacement in rewrites:
        pieces += [data[at:start], replacement]
        at = end
    pieces.append(data[at:])
    return Conversion(text=b"".join(pieces), moves=len(rewrites))
