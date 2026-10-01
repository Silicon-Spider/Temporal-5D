"""penteract/cwmtt: legal-action counts from the original hypercuboid implementation.

Driven through its stock CLI, unmodified: `cwmtt count <algorithm> <cap>` prints
one capped count per position of the record. Two sources are exposed - the
hypercuboid search ("cwmtt", its `fast` algorithm) and its exhaustive search
("cwmtt:naive").

The CLI says nothing about check and cannot tell checkmate from stalemate, so
those fields stay unknown and a position with no action is reported as
"no_action".

Its parser stops, without failing, at the first thing it does not read - an
annotation glyph, castling, a parenthesised variation - prints the rest as
`Unparsed:` and replays what it had. Most of this file is about telling what that
means for the positions it then reports.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .. import build, headers
from ..lockfile import Lock
from ..observations import Observation, Position
from ..util import Completed, run
from .base import RunOptions, Source, code_sha256

_ALGORITHMS = {"cwmtt": "fast", "cwmtt:naive": "naive"}
# A count is only trusted once the space that follows it has been printed: the
# CLI writes unbuffered, so a killed run can end in half a number.
_COUNT = re.compile(r"(\d+) ")
# The escapes Haskell's `show` writes into a string literal (Haskell 2010 report,
# section 2.6): a decimal, hex or octal code, a control character, an ASCII
# name, the empty escape \& and the single-letter ones, and a string gap.
_ASCII_NAMES = (
    "NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL", "BS", "HT", "LF", "VT", "FF", "CR", "SO", "SI",
    "DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB", "CAN", "EM", "SUB", "ESC", "FS", "GS", "RS", "US", "SP",
)
_NAMED = {name: chr(code) for code, name in enumerate(_ASCII_NAMES)} | {"SP": " ", "DEL": "\x7f"}
_LETTERS = {"a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\", '"': '"', "'": "'"}
_ESCAPE = re.compile(
    r"\\(?:(\d+)|x([0-9a-fA-F]+)|o([0-7]+)|\^([@A-Z\[\\\]^_])|("
    + "|".join(sorted(_NAMED, key=len, reverse=True)) + r""")|(&)|([abfnrtv\\"'])|(\s+\\))""")

# What may stand between two things the parser reads: white space, comments, and
# the annotation glyphs ("?", "!!") it knows nothing of.
_FILLER = re.compile(r"(?:\s+|\{[^{}]*\}|;[^\n]*|[?!]+)*")
# What may be left unread without a move having been missed: a dangling "/" with
# no reply after it, a PGN result token, and comments.
_NO_MOVES_LEFT = re.compile(r"\s*/?\s*(1-0|0-1|1/2-1/2|\*)?\s*(\{[^{}]*\}\s*)*")
# Where a new action starts: the separator between the two players' actions, a
# turn number, or a parenthesised variation (which opens with a turn number).
_ACTION_START = re.compile(r"/|\(?\s*\d+\s*[wb]?\s*\.")

# cwmtt's replay failures, by what they say about the record (BuildGame.hs).
# Only the messages listed as hard are verdicts on the rules. Everything else -
# including anything not listed at all - may be notation cwmtt reads differently,
# and is never presented as "cwmtt calls this action illegal".
_HARD_WARNINGS = {
    "Did not move in all active present timelines": "cannot_submit",
    "In check": "cannot_submit",
    "more than 1 move from timeline": "illegal_move",
    "more than 1 hop to timeline": "illegal_move",
}

# cwmtt's tag grammar (parseAnnotations in PGNParser.hs): a run of
# [Symbol "string"] pairs at the very start of the record, with white space and
# comments allowed between the tokens. It stops at the first thing that is not one.
_TAG_SPACE = r"(?:\s|;[^\n]*(?:\n|$)|\{[^}]*(?:\}|$))*"
_TAG = re.compile(
    _TAG_SPACE + r"\[" + _TAG_SPACE + r"([A-Za-z0-9_]+)" + _TAG_SPACE + r'"((?:\\.|[^"\\])*)"' + _TAG_SPACE + r"\]", re.S)


def classify_warning(text: str) -> str:
    """Map cwmtt's replay failure message to a rejection reason."""
    text = text.strip()
    for message, reason in _HARD_WARNINGS.items():
        if text.startswith(message):
            return reason
    if "No matching moves" in text:
        return "no_matching_move"
    if text.startswith("Ambigous"):
        return "ambiguous_move"
    if "expected" in text and "found" in text:
        return "notation_mismatch"
    return "unrecognised"


def _unescape(match: re.Match) -> str:
    decimal, hexadecimal, octal, control, name, empty, letter, gap = match.groups()
    if decimal or hexadecimal or octal:
        code = int(decimal) if decimal else int(hexadecimal, 16) if hexadecimal else int(octal, 8)
        return chr(code) if code <= 0x10FFFF else "�"
    if control:
        return chr(ord(control) - 64)
    if name:
        return _NAMED[name]
    if letter:
        return _LETTERS[letter]
    return ""  # \& and string gaps stand for nothing


def _unparsed_text(line: str) -> str | None:
    """The text of an `Unparsed: "..."` line, or None when it cannot be decoded.

    The line holds a Haskell string literal, as `show` writes one: anything
    outside printable ASCII - a long dash in a comment, say - arrives escaped.
    A backslash that starts no escape leaves the answer at "cannot tell".
    """
    body = line[len("Unparsed: "):].strip()
    if len(body) < 2 or not (body.startswith('"') and body.endswith('"')):
        return None
    text = body[1:-1]
    if "\\" in _ESCAPE.sub("", text):
        return None
    return _ESCAPE.sub(_unescape, text)


def remainder_kind(line: str) -> str:
    """What cwmtt's parser left unread, given its `Unparsed:` line.

    "none"        nothing that could hold a move: the record was read to its end
    "boundary"    the parser stopped between two actions; the actions it read are whole
    "mid_action"  it stopped inside an action - the last action it read is a
                  fragment, and the position after it is not a position of the record

    Anything that cannot be decoded is "mid_action": the reading that claims least.
    """
    text = _unparsed_text(line)
    if text is None:
        return "mid_action"
    left = text[_FILLER.match(text).end():]
    if _NO_MOVES_LEFT.fullmatch(left):
        return "none"
    return "boundary" if _ACTION_START.match(left) else "mid_action"


def unparsed_is_only_trailer(line: str) -> bool:
    """Does an `Unparsed: "..."` line hold nothing that could be a move?"""
    return remainder_kind(line) == "none"


def parse_count_output(source: str, record_id: str, done: Completed, cap: int, elapsed: float) -> Observation:
    lines = done.out_text().replace("\r\n", "\n").split("\n")
    unparsed = next((line for line in lines if line.startswith("Unparsed: ")), "")
    counts_line = next((line for line in reversed(lines) if line and not line.startswith("Unparsed: ")), "")
    counts = [int(text) for text in _COUNT.findall(counts_line)]
    positions = [
        Position(index, legal_actions=count, terminal="no_action" if count == 0 else "none")
        for index, count in enumerate(counts)
    ]
    observation = Observation(source=source, record=record_id, status="ok", positions=positions, cap=cap, elapsed_s=elapsed)
    warning = done.err_text().strip()
    unread = remainder_kind(unparsed) if unparsed else "none"
    if done.timed_out:
        observation.status, observation.reason = "timeout", "timeout"
    elif done.returncode != 0 or not positions:
        observation.status, observation.reason = "error", "crash"
        observation.detail = f"exit code {done.returncode}; {warning[-300:]}"
    elif unread == "mid_action":
        # The last action cwmtt read is a fragment of the record's action. If the
        # replay failed, it failed at or before that fragment and every position
        # reported is sound - but the failure says nothing about the record. If
        # it did not fail, the fragment was played, and the position after it
        # belongs to some other game.
        if not warning and len(positions) > 1:
            positions.pop()
        observation.status, observation.reason = "unsupported", "notation"
        observation.detail = (unparsed + (f" | {warning}" if warning else ""))[:300]
    elif warning:
        observation.status, observation.reason = "rejected", classify_warning(warning)
        observation.detail = warning[:300]
    elif unread == "boundary":
        # The parser stopped early. What it read was replayed faithfully; what
        # it did not read may hold more moves.
        observation.status, observation.reason = "unsupported", "notation"
        observation.detail = unparsed[:300]
    observation.actions_accepted = max(len(positions) - 1, 0)
    return observation


def cwmtt_tags(text: str) -> dict[str, str]:
    """The tags of a record exactly as cwmtt's parser reads them; a repeated tag's last value wins."""
    tags: dict[str, str] = {}
    position = 0
    while match := _TAG.match(text, position):
        tags[match.group(1)] = re.sub(r"\\(.)", r"\1", match.group(2), flags=re.S)
        position = match.end()
    return tags


def would_misread(data: bytes, layouts: list[str]) -> tuple[str, str] | None:
    """(reason, detail) when cwmtt would replay this record as a different game.

    cwmtt replays on the Standard layout, without a word, any record whose
    Board it does not know or whose Board tag its parser does not reach; it
    never reads a Variant header, skips 5DFEN boards and never reads a
    Promotions or Timeline header. An answer about some other game is worse
    than none, so it is never asked for.
    """
    declared = headers.declared_board(data) or "Standard"
    if declared not in layouts:
        return "variant", f'the record\'s variant, "{declared}", is not one of cwmtt\'s layouts'
    tag = cwmtt_tags(data.decode("utf-8", "replace")).get("Board", "")
    replayed = tag if tag in layouts else "Standard"
    if replayed != declared:
        unread = "Board header" if headers.board_header(data) is not None else "Variant header"
        return "variant", f'cwmtt does not read this record\'s {unread} ("{declared}") and would replay it on Standard'
    if headers.has_fen_boards(data):
        return "fen", "the record carries 5DFEN boards, which cwmtt does not read"
    for name in ("Promotions", "Timeline"):
        if headers.header(data, name) is not None:
            return name.lower(), f"the record carries a {name} header, which cwmtt does not read"
    return None


class CwmttSource(Source):
    oracle = "cwmtt"
    answers = ("legal_actions", "terminal", "legality")

    def __init__(self, name: str, lock: Lock):
        self.name = name
        self.algorithm = _ALGORITHMS[name]
        self.pin = lock.pin("cwmtt")

    def _build(self) -> build.Build | None:
        return build.load(self.pin)

    def available(self) -> str | None:
        return None if self._build() else "cwmtt is not built (run: python oracles/harness.py build --oracle cwmtt)"

    def signature(self, options: RunOptions) -> dict:
        return {"pin": self.pin.commit, "algorithm": self.algorithm, "cap": options.cap,
                "adapter_sha256": code_sha256(Path(__file__), Path(headers.__file__))}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        built = self._build()
        refused = would_misread(data, built.info["layouts"])
        if refused:
            return Observation(source=self.name, record=record_id, status="unsupported", reason=refused[0],
                               detail=refused[1], actions_accepted=0, cap=options.cap)
        started = time.monotonic()
        try:
            done = run([built.artifact("cli"), "count", self.algorithm, str(options.cap)],
                       stdin=data, timeout=options.timeout_s)
        except OSError as error:
            return Observation(source=self.name, record=record_id, status="error", reason="spawn",
                               detail=str(error), actions_accepted=0, cap=options.cap)
        return parse_count_output(self.name, record_id, done, options.cap, time.monotonic() - started)
