"""Does the harness's ftxi driver report what ftxi's own tools report?

The driver exists to ask what the stock CLI does not answer in one pass (the
check verdict at every position). For everything the stock CLI does answer, the
two must agree exactly - otherwise a bug in harness glue would be passed off as
the oracle's verdict. What can be cross-checked this way:

  * whether a legal action exists at each position, and the verdict where none
    does (`5dtools perftest`);
  * the number of legal actions (`5dtools count --at`);
  * the side to move, and how many boards the multiverse holds and their size
    (`5dtools stats --at`): the boards are the ones the driver's state event
    lists (decision H30), counted by the harness, and the size is the one the
    driver gives for them, which its position event repeats;
  * the boards themselves (`5dtools print --at`, which draws every board ftxi
    holds under ftxi's own name for it - "L0T1w  aka.(0T1)"): read back into a
    state, their placement hash must be the one the driver's state event gave.
    That holds the driver's timeline labels, its walk over the boards and the
    pieces it lists to ftxi's own output. The unmoved markers are not drawn, so
    they are not held to it.

The check verdict at a position that still has a legal action cannot be: no
stock command prints it. It comes from the same library call the stock perftest
makes at terminal positions (has_phantom_check), which is the most the harness
can say for it.

cwmtt's answers come from its stock CLI directly, and its state driver is held
to that CLI on every run, by its adapter (adapters/cwmtt.py, merge): the same
layout, the same text left unread, the same replay failure, the same positions.
The boards the driver lists are not held to cwmtt's own drawing of them (`cwmtt
print`, a picture of the last position only); they are compared with the other
oracles' at every position instead. 5d-chess-js's state event carries the
library's own full-state hash (chess.hash), which the harness's fingerprint
must equal at every position (adapters/base.py) - no separate check is needed.
The notation repository's parser has no command that prints its boards.
"""

from __future__ import annotations

import json
import re

from . import corpus, fingerprint
from .adapters.ftxi import FtxiSource
from .corpus import Record
from .lockfile import Lock
from .observations import Observation, Position
from .runner import Results, parallel_map
from .util import Finding


def _perftest_expectation(observation: Observation) -> tuple[str, str | None]:
    """The digit string and verdict `5dtools perftest` should print for this record."""
    digits = ""
    verdict = None
    for position in observation.positions:
        if position.legal_actions:
            digits += "1"
        else:
            digits += "0"
            verdict = position.terminal
            break
    return digits, verdict


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _boards_differences(position: Position, driver_size: object, stats: dict) -> list[str]:
    """Where the driver's boards at one position differ from `5dtools stats` there: in number, or in size."""
    count, width, height = stats.get("board_count"), stats.get("board_width"), stats.get("board_height")
    if not (_whole(count) and _whole(width) and _whole(height)):
        return ["stock `stats --at` gave no board count and size"]
    found = []
    if position.boards is None:
        found.append(f"the driver gave no state event, so no boards to count; stock stats counts {count}")
    elif position.boards != count:
        found.append(f"driver holds {position.boards} board(s), stock stats counts {count}")
    if driver_size != f"{width}x{height}":
        shown = driver_size if isinstance(driver_size, str) else "no size"
        found.append(f"driver gives boards of {shown}, stock stats says {width}x{height}")
    return found


# A board as `5dtools print` heads it: ftxi's internal line, turn and colour, then
# the board's name as the notation writes it ("aka.(-0T3)").
_PRINTED_BOARD = re.compile(r"L-?\d+T\d+([wb])\s+aka\.\(([+-]?\d+)T(\d+)\)\s*")


def printed_placement(text: str, width: int, height: int) -> str:
    """The placement hash of the boards `5dtools print` drew, read as a state event (markers false).

    Each board is a grid of two-character cells - a piece letter, "." for an
    empty square, "#" for a square outside the board - rank 8 of ftxi's largest
    board first; a WxH board is its bottom H rows and first W files, and every
    other cell must be "#". Raises ValueError (a StateError among them) when the
    drawing cannot be read so.
    """
    boards: list[str] = []
    header: re.Match | None = None
    rows: list[str] = []

    def close() -> None:
        if header is None:
            return
        colour, label, turn = header.groups()
        cells = [row[0::2] for row in rows]
        if len(cells) < height or any(len(row) < width for row in cells):
            raise ValueError(f"the board {label}T{turn}{colour} is drawn smaller than {width}x{height}")
        board, outside = cells[len(cells) - height:], cells[:len(cells) - height]
        if any(set(row) - {"#"} for row in outside) or any(set(row[width:]) - {"#"} for row in board):
            raise ValueError(f"the board {label}T{turn}{colour} is drawn larger than {width}x{height}")
        ranks = []
        for row in board:
            squares = row[:width]
            if "#" in squares:
                raise ValueError(f"the board {label}T{turn}{colour} has a square outside the board inside it")
            ranks.append(re.sub(r"\.+", lambda run: str(len(run.group(0))), squares))
        boards.append(f"{label}:{turn}:{colour}:{'/'.join(ranks)}")

    for line in text.replace("\r", "").split("\n"):
        match = _PRINTED_BOARD.fullmatch(line)
        if match:
            close()
            header, rows = match, []
        elif header is not None and line.strip():
            if line.startswith(("State(", "Multiverse", "lines range")):
                break
            rows.append(line)
    close()
    if not boards:
        raise ValueError("it drew no board")
    state = fingerprint.parse_state({"event": "state", "index": 0, "size": f"{width}x{height}", "markers": False,
                                     "boards": boards})
    return state.placement


def _sample(count: int, full: bool) -> list[int]:
    if full or count <= 3:
        return list(range(count))
    return sorted({0, count // 2, count - 1})


def _check_record(source: FtxiSource, record: Record, observation: Observation, data: bytes, cap: int,
                  timeout_s: float, full: bool) -> list[Finding]:
    findings: list[Finding] = []
    tag = f"{source.name} on {record.id}"

    def report(message: str) -> None:
        findings.append(Finding("driver-fidelity", f"{tag}: {message}"))

    done = source.stock_perftest(data, timeout_s)
    if done.timed_out:
        report(f"stock perftest did not finish in {timeout_s:.0f}s, though the driver replayed the record")
    else:
        text = done.out_text().replace("\r", "")
        digits = re.match(r"[01]*", text).group(0)
        expected, verdict = _perftest_expectation(observation)
        if observation.status == "rejected" and observation.reason == "cannot_submit":
            # The stock replay does not test an action for royal safety, and it
            # lets a final action that cannot be submitted pass with a warning:
            # either way it goes on where the driver stops. Up to there, the
            # two must say the same.
            same = digits.startswith(expected)
        else:
            same = digits == expected
        if not same:
            report(f"driver says {expected or '(nothing)'}, stock perftest says {digits or '(nothing)'}")
        stock_verdict = re.search(r": (Checkmate|Stalemate)", text)
        if digits == expected and verdict and stock_verdict and stock_verdict.group(1).lower() != verdict:
            report(f"driver says {verdict}, stock perftest says {stock_verdict.group(1).lower()}")

    for index in _sample(len(observation.positions), full):
        position = observation.positions[index]
        done = source.stock_count(data, index, cap, timeout_s)
        if done.timed_out:
            report(f"position {index}: stock `count --at` did not finish in {timeout_s:.0f}s, though the driver counted it")
        else:
            try:
                stock = json.loads(done.out_text())["legal_actions"]
            except (json.JSONDecodeError, KeyError, TypeError):
                said = (done.err_text() or done.out_text()).strip()[:160]
                report(f"stock `count --at` gave no count at position {index}: {said}")
            else:
                if stock != position.legal_actions:
                    report(f"position {index}: driver counts {position.legal_actions}, stock count says {stock}")

        done = source.stock_stats(data, index, timeout_s)
        stats, side = None, None
        try:
            stats = None if done.timed_out else json.loads(done.out_text())
            side = stats["side_to_move"] if isinstance(stats, dict) else None
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        if side is None:
            report(f"position {index}: stock `stats --at` gave no side to move")
            continue
        if side != position.side:
            report(f"position {index}: driver says {position.side} to move, stock stats says {side}")
        # The size travels in the position event (the driver repeats its state
        # event's), which the harness keeps beside the answers, never compared.
        driver_size = observation.extras.get(index, {}).get("size")
        differences = _boards_differences(position, driver_size, stats)
        for message in differences:
            report(f"position {index}: {message}")
        if differences or position.placement is None:
            continue  # the boards cannot be compared one by one where their number or size already differs
        done = source.stock_print(data, index, timeout_s)
        if done.timed_out:
            report(f"position {index}: stock `print --at` did not finish in {timeout_s:.0f}s")
            continue
        try:
            drawn = printed_placement(done.out_text(), stats["board_width"], stats["board_height"])
        except ValueError as error:
            report(f"position {index}: stock `print --at` could not be read as boards: {error}")
            continue
        if drawn != position.placement:
            report(f"position {index}: the driver's boards are not the ones stock `print --at` draws (placement "
                   f"{position.placement} from the driver, {drawn} from the drawing)")
    return findings


def check(source: FtxiSource, records: list[Record], results: Results, lock: Lock, cap: int, *,
          timeout_s: float, full: bool, jobs: int | None = None) -> list[Finding]:
    """Compare one ftxi source's reports with the stock CLI run under the same search policy.

    ``full`` checks every position; otherwise three per record. The stock tool
    repeats work the driver already did within ``timeout_s``, so it gets the same
    budget, and running out of it is a finding, not a skipped check.
    """
    tasks = []
    for record in records:
        observation = results.get((source.name, record.id))
        # Only a record the driver replayed to its end (or to a refusal) has a
        # complete report to hold the stock tool against.
        if observation is not None and observation.status in ("ok", "rejected") and observation.positions:
            tasks.append((record, observation))

    def one(task: tuple[Record, Observation]) -> list[Finding]:
        record, observation = task
        return _check_record(source, record, observation, corpus.read(record, lock), cap, timeout_s, full)

    return [finding for found in parallel_map(one, tasks, jobs) for finding in found]
