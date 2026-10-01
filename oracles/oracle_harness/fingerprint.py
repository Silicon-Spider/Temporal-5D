"""What a source says it set up at each position, and the fingerprint the harness
computes from it (decision H30; design spec 11.0, "Running the gate").

Positions are aligned by counting submitted actions. On its own that cannot tell
a rules disagreement from two parsers that read one move as two different legal
actions: after such a move every position is a different position under the
same number. So every source also says, at every position and before it
searches anything, what it has on its boards:

    {"event":"state","index":K,"size":"WxH","markers":true,
     "boards":["<L>:<T>:<c>:<placement>", ...]}

  index    the position: K submitted actions accepted (0 is the set-up)
  size     the width and height of the boards the source set up
  markers  whether the source tracks which pieces have not moved; when false,
           no "*" may appear
  boards   every board of the multiverse - superseded and head boards, every
           timeline, both colours - in any order. L is the timeline as the
           notation writes it ("0" for the central timeline of a game with an
           odd number of starting timelines, "-0" and "+0" for the central pair
           of an even one, else a signed or unsigned integer), T the turn as
           displayed, c "w" or "b" for the side to move on that board, and the
           placement is the 5DFEN pieces field: ranks from the top down
           separated by "/", files from a, the letters P N B R Q K U D S W C Y
           (white upper case, black lower case), runs of empty squares as
           base-10 numbers, and - only when markers is true - a "*" directly
           after every pawn, rook, king or brawn that has not moved, and
           after no other piece.

The harness - never a source - checks every board against the size, spells
each timeline label one way ("+n" for a positive timeline), re-writes the runs
of empty squares, and computes:

  fingerprint  the notation specification's full-state hash (README "Hashing"
               and "5DFEN Strictness"), as read here (below): each board
               written "[placement:L:T:c]", sorted by timeline (-0 before +0),
               then by turn, then white's board before black's, concatenated
               with nothing between, and hashed with MD5 (lower-case hex).
               None when markers is false.
  placement    the same hash of the same string with every "*" removed: every
               board - its timeline, turn and colour as well as its pieces -
               without the unmoved markers, which every source can export
  boards       how many boards; pieces: how many pieces on them

The specification's two published examples - the Standard start and the
two-timeline Standard start - are conformance tests of this module. Three
readings it leaves open are fixed here:

  * the order of a turn's two boards: white's first, which is the order of time;
  * the "+" a positive timeline may omit: the canonical form always writes it;
  * which pieces are marked. Its 5DFEN section says a piece "sensitive to having
    been moved already or not" that "hasn't moved yet" must be followed by "*";
    "5DFEN Strictness" names those pieces - pawn, rook, king, brawn - and says
    they are marked "if needed". Read together: every pawn, rook, king and
    brawn that has not moved carries the marker, whether or not a castling or
    a double step is still open to it (a rook keeps its marker after its king
    has moved), and no other piece does. A source whose set-up holds a piece
    as moved before it has moved - a layout drawn without the markers -
    differs from one that follows this, and the fingerprints show it.

A source may add the hash its own implementation computes of the same state
("hash", on an oracle's state event only): the harness then checks its own
fingerprint against it (adapters/base.py), which holds this module to an
implementation that was not written from it.

The same module reads the timeline labels and squares of the diagnostic
"actions" event (`observe --actions K`), which lists a source's legal actions so
that two sources' lists can be laid side by side.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

PIECES = "PNBRQKUDSWCY"
# The pieces whose having moved changes what they may do, and so the only ones
# that carry the unmoved marker (README "5DFEN Strictness"): pawn, rook, king, brawn.
UNMOVED = "PRKW"
STATE_FIELDS = ("event", "index", "size", "markers", "boards")
# The most actions one "actions" event may list.
ACTIONS_LIMIT = 5000
# The fields of an "actions" event the harness reads; a driver may add its own.
_ACTIONS_FIELDS = ("event", "index", "count", "truncated", "actions")

_LETTERS = frozenset(PIECES + PIECES.lower())
_DIGITS = frozenset("0123456789")
# Every number in a state event is bounded before it is converted: Python refuses
# to convert a string of more than a few thousand digits (a ValueError, not a
# StateError), and a source's output is not trusted to keep within that.
_MAX_DIGITS = 9
_LABEL = re.compile(r"([+-]?)(0|[1-9][0-9]{0,8})")
_TURN = re.compile(r"0|[1-9][0-9]{0,8}")
_SIZE = re.compile(r"([1-9][0-9]{0,3})x([1-9][0-9]{0,3})")
_SQUARE = re.compile(r"([a-z])([1-9][0-9]{0,3})")
_HEX = re.compile(r"[0-9a-f]{32}")


class StateError(ValueError):
    """A state (or actions) event that does not say what the protocol asks for."""


def md5_hex(text: str) -> str:
    # Not a security use: the notation specification names MD5 for its hashes.
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def is_hash(value: object) -> bool:
    """Is this a hash as the harness writes one: 32 lower-case hex digits?"""
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def timeline_label(text: object) -> str:
    """The one spelling the harness uses for a timeline label.

    "0", "-0" and "+0" stay as they are (they are three different timelines of
    two different kinds of game); any other positive timeline gains its "+".
    """
    match = _LABEL.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise StateError(f"bad timeline label {text!r}: an integer written as the notation writes it "
                         "('0', '-0', '+0', '1' or '+1', '-1', ...)")
    sign, digits = match.groups()
    if digits == "0":
        return sign + "0"
    return (sign or "+") + digits


def timeline_rank(label: str) -> int:
    """Sort key for a canonical label: ... -1 < -0 < 0 < +0 < +1 ... (doubled, so the central pair fits)."""
    if label == "-0":
        return -1
    if label == "+0":
        return 1
    return 2 * int(label)


def parse_size(text: object) -> tuple[int, int]:
    match = _SIZE.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise StateError(f"bad size {text!r}: the board's width and height, as \"8x8\"")
    return int(match.group(1)), int(match.group(2))


def parse_placement(text: str, width: int, height: int, markers: bool) -> tuple[str, int]:
    """(the canonical placement field, the number of pieces) of one board.

    Runs of empty squares are re-written from the squares themselves, so the
    canonical text does not depend on how a source happened to write them.
    """
    ranks = text.split("/")
    if len(ranks) != height:
        raise StateError(f"{len(ranks)} rank(s), but the board is {height} high")
    canonical: list[str] = []
    pieces = 0
    for number, rank in enumerate(ranks):
        name = f"rank {height - number}"  # the first rank written is the top one
        written: list[str] = []
        squares = blank = 0
        position = 0
        while position < len(rank):
            char = rank[position]
            if char in _DIGITS:
                end = position
                while end < len(rank) and rank[end] in _DIGITS:
                    end += 1
                run = rank[position:end]
                if run[0] == "0":
                    raise StateError(f"{name}: '{run}' - a run of empty squares is a number from 1, "
                                     "written without leading zeros")
                if len(run) > _MAX_DIGITS:
                    raise StateError(f"{name}: a run of empty squares written with {len(run)} digits")
                blank += int(run)
                squares += int(run)
                position = end
            elif char in _LETTERS:
                if blank:
                    written.append(str(blank))
                    blank = 0
                marked = rank[position + 1:position + 2] == "*"
                if marked and not markers:
                    raise StateError(f"{name}: '*' after '{char}', but markers is false - a source that does not "
                                     "track unmoved pieces writes no '*'")
                if marked and char.upper() not in UNMOVED:
                    raise StateError(f"{name}: '*' after '{char}' - only a pawn, rook, king or brawn "
                                     f"({', '.join(UNMOVED)}) carries the unmoved marker")
                written.append(char + "*" if marked else char)
                pieces += 1
                squares += 1
                position += 2 if marked else 1
            elif char == "*":
                raise StateError(f"{name}: a '*' that does not follow a piece letter")
            else:
                raise StateError(f"{name}: unknown letter {char!r} - pieces are {' '.join(PIECES)} "
                                 "(white upper case, black lower case) and empty squares are numbers")
        if blank:
            written.append(str(blank))
        if squares != width:
            raise StateError(f"{name} is {squares} square(s) wide, but the board is {width} wide")
        canonical.append("".join(written))
    return "/".join(canonical), pieces


@dataclass(frozen=True)
class Board:
    timeline: str  # canonical label
    turn: int
    colour: str  # "w" | "b": the side to move on this board
    placement: str  # canonical placement field
    pieces: int

    def canonical(self) -> str:
        """The board as the notation specification hashes it."""
        return f"[{self.placement}:{self.timeline}:{self.turn}:{self.colour}]"

    def sort_key(self) -> tuple[int, int, int]:
        return timeline_rank(self.timeline), self.turn, 0 if self.colour == "w" else 1


def parse_board(text: object, width: int, height: int, markers: bool) -> Board:
    if not isinstance(text, str):
        raise StateError(f"a board is a string \"<L>:<T>:<c>:<placement>\", not {text!r}")
    try:
        parts = text.split(":", 3)
        if len(parts) != 4:
            raise StateError("expected \"<L>:<T>:<c>:<placement>\"")
        label, turn, colour, placement = parts
        timeline = timeline_label(label)
        if _TURN.fullmatch(turn) is None:
            raise StateError(f"bad turn {turn!r}: the turn as displayed, a whole number from 0")
        if colour not in ("w", "b"):
            raise StateError(f"bad colour {colour!r}: 'w' or 'b', the side to move on the board")
        canonical, pieces = parse_placement(placement, width, height, markers)
    except StateError as error:
        shown = text if len(text) <= 100 else text[:97] + "..."
        raise StateError(f"board {shown!r}: {error}") from None
    return Board(timeline, int(turn), colour, canonical, pieces)


@dataclass(frozen=True)
class State:
    """One state event, read and put in canonical order."""

    index: int
    width: int
    height: int
    markers: bool
    boards: tuple[Board, ...]  # sorted as the full-state hash sorts them

    def canonical(self) -> str:
        """The notation specification's full-state string."""
        return "".join(board.canonical() for board in self.boards)

    @property
    def fingerprint(self) -> str | None:
        return md5_hex(self.canonical()) if self.markers else None

    @property
    def placement(self) -> str:
        return md5_hex(self.canonical().replace("*", ""))

    @property
    def pieces(self) -> int:
        return sum(board.pieces for board in self.boards)

    def summary(self) -> dict:
        """The four values a Position carries: fingerprint, placement, boards, pieces."""
        return {"fingerprint": self.fingerprint, "placement": self.placement,
                "boards": len(self.boards), "pieces": self.pieces}


def _index(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StateError(f"bad index {value!r}: the position, a whole number from 0")
    return value


def parse_state(event: dict, *, exact_fields: bool = False) -> State:
    """Read a state event, refusing anything the protocol does not allow.

    ``exact_fields`` refuses an event with any field beyond the protocol's five
    (the engine is held to that; see adapters/engine.py).
    """
    if not isinstance(event, dict) or event.get("event") != "state":
        raise StateError("not a state event")
    if exact_fields and set(event) != set(STATE_FIELDS):
        raise StateError(f"fields {sorted(event)}; a state event has exactly {list(STATE_FIELDS)}")
    index = _index(event.get("index"))
    width, height = parse_size(event.get("size"))
    markers = event.get("markers")
    if not isinstance(markers, bool):
        raise StateError(f"bad markers {markers!r}: true or false")
    texts = event.get("boards")
    if not isinstance(texts, list) or not texts:
        raise StateError("boards must be a non-empty list: every board of the multiverse")
    boards = [parse_board(text, width, height, markers) for text in texts]
    seen: set[tuple[str, int, str]] = set()
    for board in boards:
        key = (board.timeline, board.turn, board.colour)
        if key in seen:
            raise StateError(f"two boards at timeline {board.timeline}, turn {board.turn}, {board.colour}")
        seen.add(key)
    labels = {board.timeline for board in boards}
    if "0" in labels and labels & {"-0", "+0"}:
        raise StateError("timeline labels mix '0' with '-0'/'+0': a game has one central timeline or a central "
                         "pair, never both - check how the source spells its timelines")
    return State(index=index, width=width, height=height, markers=markers,
                 boards=tuple(sorted(boards, key=Board.sort_key)))


# --- the diagnostic "actions" event (observe --actions K) --------------------

def _coordinate(value: object, where: str) -> dict:
    if not isinstance(value, dict) or not {"L", "T", "c", "square"} <= set(value):
        raise StateError(f"{where}: expected {{\"L\", \"T\", \"c\", \"square\"}}, got {value!r}")
    turn, colour, square = value["T"], value["c"], value["square"]
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
        raise StateError(f"{where}: bad turn {turn!r}")
    if colour not in ("w", "b"):
        raise StateError(f"{where}: bad colour {colour!r}")
    if not isinstance(square, str) or _SQUARE.fullmatch(square) is None:
        raise StateError(f"{where}: bad square {square!r}")
    return {**value, "L": timeline_label(value["L"])}


def _coordinate_key(coordinate: dict) -> tuple:
    file, rank = _SQUARE.fullmatch(coordinate["square"]).groups()
    return timeline_rank(coordinate["L"]), coordinate["T"], coordinate["c"] != "w", file, int(rank)


def _move(value: object, where: str) -> dict:
    if not isinstance(value, dict) or not {"from", "to"} <= set(value):
        raise StateError(f"{where}: a move is {{\"from\": ..., \"to\": ..., \"promotion\": ...}}, got {value!r}")
    promotion = value.get("promotion")
    if promotion is not None and not (isinstance(promotion, str) and promotion in _LETTERS):
        raise StateError(f"{where}: bad promotion {promotion!r}: null or a piece letter")
    return {**value, "from": _coordinate(value["from"], f"{where} from"), "to": _coordinate(value["to"], f"{where} to"),
            "promotion": promotion.upper() if promotion else None}


def _move_key(move: dict) -> tuple:
    return _coordinate_key(move["from"]), _coordinate_key(move["to"]), move["promotion"] or ""


def canonical_actions(event: dict) -> dict:
    """An "actions" event with its labels spelled one way and its lists sorted.

    Each action's moves are sorted, then the actions themselves, so two
    sources' lists of the same position can be compared line by line. A
    promotion letter is written upper case: the mover's colour says whose piece
    it is. Fields a driver adds of its own - what cut its list short, or what
    went wrong while it listed - are kept as printed, ahead of the list: they
    say how far the list can be read as the source's whole answer.
    """
    if not isinstance(event, dict) or event.get("event") != "actions":
        raise StateError("not an actions event")
    index = _index(event.get("index"))
    count, truncated, actions = event.get("count"), event.get("truncated"), event.get("actions")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise StateError(f"bad count {count!r}")
    if not isinstance(truncated, bool):
        raise StateError(f"bad truncated {truncated!r}: true or false")
    if not isinstance(actions, list) or len(actions) > ACTIONS_LIMIT:
        raise StateError(f"actions must be a list of at most {ACTIONS_LIMIT} actions")
    if count < len(actions) or (not truncated and count != len(actions)):
        raise StateError(f"count {count} does not fit {len(actions)} listed action(s) (truncated: {truncated})")
    listed = []
    for number, action in enumerate(actions):
        if not isinstance(action, list) or not action:
            raise StateError(f"action {number}: a non-empty list of moves")
        moves = [_move(move, f"action {number} move {position}") for position, move in enumerate(action)]
        listed.append(sorted(moves, key=_move_key))
    listed.sort(key=lambda moves: [_move_key(move) for move in moves])
    own = {key: value for key, value in event.items() if key not in _ACTIONS_FIELDS}
    return {"event": "actions", "index": index, "count": count, "truncated": truncated, **own, "actions": listed}
