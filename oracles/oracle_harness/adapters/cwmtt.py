"""penteract/cwmtt: legal-action counts from the original hypercuboid implementation.

Two programs answer for it, and this adapter merges them by position:

- Its stock CLI, unmodified: `cwmtt count <algorithm> <cap>` prints one capped
  count per position of the record. It is the source of every count cwmtt gives
  here, and of how far the record is replayed. Two sources are exposed -
  the hypercuboid search ("cwmtt", its `fast` algorithm) and its exhaustive search
  ("cwmtt:naive").
- The harness's state driver (oracles/drivers/cwmtt/Driver.hs), which reads and
  replays the record through the same cwmtt code and prints, at each position, the
  side to move and every board cwmtt holds there (its state event, decision H30) -
  which the CLI has no command for - and nothing else. Asked by `observe`, it also
  prints the diagnostics: what cwmtt computes about each position, and its legal
  actions at one.

The two are merged only once they are shown to have read and replayed the record
alike: the layout the record declares, the same text left unread, the same replay
failure, the same positions. Where they did not, the report is an error
("driver_mismatch") and carries no state: a count must never be paired with the
state of another position.

The CLI says nothing about check and cannot tell checkmate from stalemate, so
those fields stay unknown and a position with no action is reported as
"no_action".

Its parser stops, without failing, at the first thing it does not read - an
annotation glyph, castling, a parenthesised variation - prints the rest as
`Unparsed:` and replays what it had. Most of this file is about telling what that
means for the positions it then reports.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path

from .. import build, fingerprint, headers, paths
from ..lockfile import Lock
from ..observations import FINGERPRINT_FIELDS, Observation, Position
from ..util import Completed, run
from . import base
from .base import RunOptions, Source, code_sha256

_DRIVER = paths.DRIVERS / "cwmtt" / "Driver.hs"

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
    return remainder_text_kind(_unparsed_text(line))


def remainder_text_kind(text: str | None) -> str:
    """remainder_kind() of the unread text itself, as the state driver prints it (None: it could not be read)."""
    if text is None:
        return "mid_action"
    left = text[_FILLER.match(text).end():]
    if _NO_MOVES_LEFT.fullmatch(left):
        return "none"
    return "boundary" if _ACTION_START.match(left) else "mid_action"


def unparsed_is_only_trailer(line: str) -> bool:
    """Does an `Unparsed: "..."` line hold nothing that could be a move?"""
    return remainder_kind(line) == "none"


def _cli_output(done: Completed) -> tuple[str, list[int], str]:
    """What a `cwmtt count` run printed: its `Unparsed:` line ("" if none), its counts, and its replay failure."""
    lines = done.out_text().replace("\r\n", "\n").split("\n")
    unparsed = next((line for line in lines if line.startswith("Unparsed: ")), "")
    counts_line = next((line for line in reversed(lines) if line and not line.startswith("Unparsed: ")), "")
    # A count too long to be one (or to convert) is kept as -1, which validate() reports.
    counts = [int(text) if len(text) <= 9 else -1 for text in _COUNT.findall(counts_line)]
    return unparsed, counts, done.err_text().strip()


def parse_count_output(source: str, record_id: str, done: Completed, cap: int, elapsed: float) -> Observation:
    unparsed, counts, warning = _cli_output(done)
    positions = [
        Position(index, legal_actions=count, terminal="no_action" if count == 0 else "none")
        for index, count in enumerate(counts)
    ]
    observation = Observation(source=source, record=record_id, status="ok", positions=positions, cap=cap, elapsed_s=elapsed)
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


def _event(done: Completed, kind: str) -> dict | None:
    """The first event of this kind a protocol run printed, if it printed one."""
    for raw in done.out_text().splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == kind:
            return event
    return None


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _driver_problem(driven: Observation, start: dict | None, end: dict | None) -> str:
    """Why the state driver's run cannot be used at all, or "" when it can."""
    if driven.status == "error":
        return f"the state driver failed ({driven.reason}): {driven.detail}"
    if start is None or end is None:
        return "the state driver printed no " + ("start" if start is None else "end") + " event"
    parsed, accepted = end.get("parsed_actions"), driven.actions_accepted
    if not _count(parsed) or not isinstance(end.get("unparsed"), str) or "failure" not in end:
        return f"the state driver's end event lacks parsed_actions, unparsed or failure: {json.dumps(end)[:200]}"
    failure = end["failure"]
    if driven.status not in ("ok", "rejected"):
        return f"the state driver ended '{driven.status}'"
    if driven.status == "ok" and failure is not None:
        return f"the state driver reports a replay failure, yet ended 'ok': {str(failure)[:100]!r}"
    if driven.status == "ok" and parsed != accepted:
        return f"the state driver replayed {accepted} of the {parsed} action(s) it parsed, yet reports no failure"
    if driven.status == "rejected" and (not isinstance(failure, str) or not failure.strip() or parsed <= accepted):
        return f"the state driver's replay stopped after {accepted} of {parsed} parsed action(s) without saying why"
    return ""


def _mismatch(counted: Observation, cli: Completed, driven: Observation, start: dict, end: dict,
              declared: str) -> str:
    """How the stock CLI and the state driver read or replayed the record differently, or "" if they did not.

    Each check is on something both computed with the same cwmtt code from the
    same bytes: a difference means the two runs are not of the same game, and
    no state of the driver's may be paired with a count of the CLI's.
    """
    if start.get("layout") != declared:
        return f"cwmtt set up the layout {start.get('layout')!r}, but the record declares {declared!r}"
    unparsed, counts, warning = _cli_output(cli)
    printed = len(counts)
    # A run the CLI did not finish says less: it may have stopped before its
    # replay failure, or (in principle) before its Unparsed: line.
    finished = counted.status not in ("timeout", "error")
    remainder = end["unparsed"]
    if unparsed:
        text = _unparsed_text(unparsed)
        if text is not None and text != remainder:
            return f"the two left different text unread: the stock CLI {text[:60]!r}, the state driver {remainder[:60]!r}"
    elif finished and remainder:
        return f"the state driver left {remainder[:60]!r} unread, where the stock CLI read the record to its end"
    failure = end["failure"] or ""
    if finished and failure.strip() != warning:
        def said(message: str) -> str:
            return f"failed: {message[:100]!r}" if message else "did not fail"

        return f"the stock CLI's replay {said(warning)}, the state driver's {said(failure.strip())}"
    reported = len(driven.positions)
    if finished and printed != reported:
        return f"the stock CLI reported {printed} position(s), the state driver {reported}"
    if not finished and printed > reported:
        return f"the stock CLI reported {printed} position(s) before it stopped, the state driver only {reported}"
    return ""


def _fragment(remainder: object, parsed: object) -> int | None:
    """The index of the position after half an action, if cwmtt's reading left one; else None.

    When the parser stops inside an action, the last action it read is a
    fragment, and a replay that plays it reaches a position of no game the
    record describes: the one after ``parsed`` actions. No position from there
    on is reported, by either program, however the CLI's run ended. (A replay
    that failed at or before the fragment never reached it.)
    """
    if _count(parsed) and parsed > 0 and isinstance(remainder, str) and remainder_text_kind(remainder) == "mid_action":
        return parsed
    return None


def _pair(counted: Observation, states: list[Position], limit: int | None) -> Observation:
    """The CLI's positions, below ``limit``, with what the state driver says cwmtt set up at each it reached.

    Where the CLI stopped while counting the next position (it ran out of time,
    or failed), and cwmtt's replay had reached it, what was set up there is
    known and the count is not: that position is kept, with its state only.
    """
    def with_state(answered: Position) -> Position:
        if answered.index >= len(states):
            return answered  # the driver did not get this far
        state = states[answered.index]
        return dataclasses.replace(answered, side=state.side,
                                   **{name: getattr(state, name) for name in FINGERPRINT_FIELDS})

    answered = counted.positions if limit is None else counted.positions[:limit]
    positions = [with_state(position) for position in answered]
    if (counted.status in ("timeout", "error") and len(states) > len(positions)
            and (limit is None or len(positions) < limit)):
        positions.append(with_state(Position(len(positions))))
    counted.positions = positions
    counted.actions_accepted = max(len(positions) - 1, 0)
    return counted


def _cut_short_driver(counted: Observation, cli: Completed, driven: Observation, start: dict | None,
                      declared: str) -> tuple[str, str]:
    """(why the states of a state driver that ran out of time cannot be paired, and whether that is a
    mismatch - "driver_mismatch" - or only too little to go on - "")."""
    if start is None:
        return "it ran out of time before its start event", ""
    if start.get("layout") != declared:
        return f"cwmtt set up the layout {start.get('layout')!r}, but the record declares {declared!r}", "driver_mismatch"
    unparsed, counts, _ = _cli_output(cli)
    remainder = start.get("unparsed")
    if not isinstance(remainder, str):
        return "its start event does not say what it left unread", ""
    text = _unparsed_text(unparsed) if unparsed else ("" if counts else None)  # None: the CLI printed nothing yet
    if text is not None and text != remainder:
        return (f"the two left different text unread: the stock CLI {text[:60]!r}, the state driver "
                f"{remainder[:60]!r}"), "driver_mismatch"
    if counted.status not in ("timeout", "error") and len(driven.positions) > len(counts):
        return (f"the stock CLI's replay has {len(counts)} position(s), but the state driver reported "
                f"{len(driven.positions)} before it ran out of time"), "driver_mismatch"
    return "", ""


def merge(source: str, record_id: str, cli: Completed, driver: Completed, declared: str, cap: int,
          elapsed: float) -> Observation:
    """One report from a stock CLI run and a state driver run on the same record.

    Every answer, and the outcome, are the CLI's (parse_count_output). The driver
    adds the side to move and what cwmtt set up at each of the CLI's positions -
    and at the next one, when the CLI stopped while counting it - but only once
    the two are shown to have read and replayed the record alike (_mismatch).
    Otherwise the report is an error and carries no state. ``declared`` is the
    variant the record declares, which cwmtt must have set up.

    The driver searches nothing, so it seldom runs out of time; when it does,
    the CLI's answers and outcome stand, and the states it printed are paired
    with them as far as it got, once its start event shows it set up the
    declared layout and left the CLI's text unread.
    """
    counted = parse_count_output(source, record_id, cli, cap, elapsed)
    driven = base.parse_jsonl(source, record_id, driver, None, elapsed)
    counted.diagnostics = driven.diagnostics
    start, end = _event(driver, "start"), _event(driver, "end")

    def failed(reason: str, problem: str) -> Observation:
        if counted.status == "error":
            # The CLI failed too; that is the outcome, and the driver's is noted.
            counted.detail = f"{counted.detail} | {problem}"[:300]
        else:
            counted.status, counted.reason, counted.detail = "error", reason, problem[:300]
        return counted

    def noted(note: str) -> None:
        counted.detail = f"{counted.detail} | {note}"[:300] if counted.detail else note[:300]

    if driven.status == "timeout":
        problem, kind = _cut_short_driver(counted, cli, driven, start, declared)
        if kind:
            return failed(kind, f"the state driver ran out of time, and {problem}")
        if problem:
            noted(f"the state driver ran out of time: {problem}; no state is paired")
            return counted
        noted(f"the state driver ran out of time after {len(driven.positions)} position(s)")
        return _pair(counted, driven.positions, _fragment(start["unparsed"], start.get("parsed_actions")))
    problem = _driver_problem(driven, start, end)
    if problem:
        return failed("driver_error", problem)
    problem = _mismatch(counted, cli, driven, start, end, declared)
    if problem:
        return failed("driver_mismatch", problem)
    return _pair(counted, driven.positions, _fragment(end["unparsed"], end["parsed_actions"]))


class CwmttSource(Source):
    oracle = "cwmtt"
    answers = ("state", "side", "legal_actions", "terminal", "legality")
    diagnostics = ("explain", "actions")

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
                "driver_sha256": code_sha256(_DRIVER),
                "adapter_sha256": code_sha256(Path(__file__), Path(base.__file__), Path(fingerprint.__file__),
                                              Path(headers.__file__))}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        built = self._build()
        refused = would_misread(data, built.info["layouts"])
        if refused:
            return Observation(source=self.name, record=record_id, status="unsupported", reason=refused[0],
                               detail=refused[1], actions_accepted=0, cap=options.cap)
        started = time.monotonic()
        try:
            cli = run([built.artifact("cli"), "count", self.algorithm, str(options.cap)],
                      stdin=data, timeout=options.timeout_s)
            driver = run([built.artifact("driver"), "--algorithm", self.algorithm, *self.diagnostic_args(options)],
                         stdin=data, timeout=options.timeout_s)
        except OSError as error:
            return Observation(source=self.name, record=record_id, status="error", reason="spawn",
                               detail=str(error), actions_accepted=0, cap=options.cap)
        declared = headers.declared_board(data) or "Standard"
        return merge(self.name, record_id, cli, driver, declared, options.cap, time.monotonic() - started)
