"""The interface every source of observations implements, and the JSON Lines
protocol the process-based ones (harness drivers and the engine) speak.

Protocol - the record arrives on stdin, one JSON object per line leaves on stdout:

    {"event":"start", ...}                                   optional, informational
    {"event":"state","index":k,"size":"WxH","markers":bool,
     "boards":["<L>:<T>:<c>:<placement>", ...]}             what the source set up (fingerprint.py)
                                                             [,"hash":"<md5>"]: an oracle's own (below)
    {"event":"position","index":k,"side":"white"|"black"|null,
     "in_check":bool|null,"legal_actions":int|null,
     "terminal":"none"|"checkmate"|"stalemate"|"no_action"|null}
    {"event":"end","status":"ok"|"rejected"|"unsupported"|"error",
     "actions_accepted":n,"reason":"...","detail":"...","last_action":bool}

Position k is the state after k submitted actions of the record's main line;
position 0 is the starting position. So a run that accepted n actions reports
exactly the positions 0..n, and says so in ``actions_accepted``. Lines are
flushed as they are produced: a run that is killed for time still yields the
positions it answered.

The state event of position k comes before its position event, and before any
search at k, so a search that runs out of time cannot take with it what the
source set up: a state event whose position event never came - the run was
killed, or failed during the search - still yields position k, with what was
set up and no answers. A run that says it ended any other way must have
reported the position of every state event it printed. A source that cannot
export its state prints none (the engine must; adapters/engine.py); one that
does prints exactly one before every position event, from position 0, with
the same "markers" throughout. An oracle's driver may add "hash" to a state
event: its implementation's own full-state hash of what it exported, which the
harness's fingerprint must equal (fingerprint.py).

A run killed for time may end in a line cut off mid-write; that last line is
passed over. Any other line that is not a JSON event, and any event that breaks
the protocol, makes the run an error ("malformed_output") however it ended.

Diagnostic events - "explain" and "actions", printed only when `observe` asks
for them (Source.diagnostic_args) - are kept on the Observation for `observe` to
show, and are never compared. Fields a driver adds to a position event (its
timing, "ms"; the limit an unanswered question ran into, "limit") are kept on
the Observation too, and never compared. Any other event is ignored.

When each line arrived is kept as well (Observation.arrival_s, ended_s): the
cost of an answer is the time from the start of the run until it arrived, which
is what a deadline cuts (H32). A gate run that needs nothing past a position may
stop the run once that position has arrived (RunOptions.stop_after); the run is
then cut short like one that ran out of time, with the reason "stopped".
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import fingerprint
from ..observations import ANSWERS, HARD_REJECTIONS, SIDES, SOFT_REJECTIONS, STATUSES, TERMINALS, Observation, Position
from ..util import Completed, Stop, run

DIAGNOSTIC_EVENTS = ("explain", "actions")
# What a run may be stopped after: the state event of a position (what was set
# up there), or its position event (its answers too).
STOP_EVENTS = ("state", "position")


@dataclass(frozen=True)
class RunOptions:
    cap: int = 1000
    timeout_s: float = 120.0
    mate_budget_ms: int = 5000
    # Diagnostics, which only `observe` asks for - never a gate run, never golden --write:
    explain: bool = False  # an "explain" event after each state event
    actions_at: int | None = None  # an "actions" event listing the legal actions at this position
    # (index, "state" | "position"): the run may be stopped once that event of that
    # position has arrived - a gate run that needs nothing after it (H32). None: run to the end or the deadline.
    stop_after: tuple[int, str] | None = None


def number(value: float) -> str:
    """A budget as a limit names it: 300.0 -> "300", 0.5 -> "0.5"."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


class Source:
    """One way of asking "what does this implementation say about this record?"."""

    name: str = ""
    oracle: str | None = None  # pin id whose build this source needs; None for the engine
    answers: tuple[str, ...] = ()  # observables this source can report
    diagnostics: tuple[str, ...] = ()  # the diagnostic events its driver can print: "explain", "actions"
    # The RunOptions fields that bound this source's answers. They are its budget,
    # which a golden file's signature records: answers recorded under one budget
    # are not the answers of another (H32).
    budget_fields: tuple[str, ...] = ("timeout_s",)
    # The fields of a position a run has only once it has searched that position:
    # they arrive with its position event, after what it set up (its state event).
    # A run cut while it searched a position has its state, not these (H32).
    searched: tuple[str, ...] = ANSWERS

    def budget(self, options: RunOptions) -> dict:
        """The limits this source runs under with these options, by name: {"timeout_s": 300, ...}."""
        return {name: getattr(options, name) for name in self.budget_fields}

    def limits(self, observation: Observation, index: int, options: RunOptions) -> dict[str, str]:
        """The answers at one position that a limit left unanswered, each with the limit ("mate_budget_ms=60000").

        A null that no limit explains stays a bare null: the source never answers
        that question there. The deadline of the whole run is not one of these:
        a run cut at its deadline is cut from the position it was searching on.
        """
        return {}

    def diagnostic_args(self, options: RunOptions) -> list[str]:
        """Driver arguments asking for the diagnostic events ``options`` wants and this source can print.

        The default speaks the drivers' common flags, ``--explain`` and
        ``--actions K``, for the events named in ``diagnostics``; a driver with
        flags of its own overrides this. A source that names none is asked for
        nothing, and `observe` says so.
        """
        args: list[str] = []
        if options.explain and "explain" in self.diagnostics:
            args.append("--explain")
        if options.actions_at is not None and "actions" in self.diagnostics:
            args += ["--actions", str(options.actions_at)]
        return args

    def available(self) -> str | None:
        """None when ready to run, otherwise the reason it is not."""
        raise NotImplementedError

    def signature(self, options: RunOptions) -> dict:
        """What determines this source's answers besides the pin: recorded in golden files."""
        raise NotImplementedError

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        raise NotImplementedError


def code_sha256(*files: Path) -> str:
    """One hash over harness code that stands between an oracle and its recorded answers.

    It goes into the source's signature, so answers recorded by one version of
    an adapter or driver are not taken for the answers of another. Line endings
    are folded: a checkout on Windows must hash like one on Linux.
    """
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def _tail(text: str, limit: int = 300) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


POSITION_FIELDS = frozenset({"event", "index", "side", "in_check", "legal_actions", "terminal"})


def _state_problem(event: dict, positions: list[Position], pending: fingerprint.State | None,
                   exact_fields: bool, markers: bool | None) -> tuple[fingerprint.State | None, str]:
    """(the state event read, or None with why it cannot be).

    ``markers`` is what the run's earlier state events said of the unmoved
    markers, or None when this is its first state event.
    """
    try:
        state = fingerprint.parse_state(event, exact_fields=exact_fields)
    except (fingerprint.StateError, ValueError) as error:
        return None, str(error)
    if pending is not None and state.index == pending.index:
        return None, f"a second state event for position {state.index}"
    if state.index < len(positions):
        return None, f"a state event for position {state.index} after its position event - the state comes first"
    if state.index != len(positions):
        expected = f"position {len(positions)}" if pending is None else f"the position event of {pending.index}"
        return None, f"a state event for position {state.index} where {expected} comes next"
    if markers is None and positions:
        return None, (f"the run's first state event is for position {state.index}: a source that says what it set "
                      "up says it at every position, from position 0")
    if markers is not None and state.markers != markers:
        return None, (f"markers is {json.dumps(state.markers)} here and was {json.dumps(markers)} before: a source "
                      "tracks which pieces have not moved throughout a run, or not at all")
    if "hash" in event:
        # An oracle's own hash of the state it exported: the harness's
        # canonical form of that state must hash to the same.
        if state.fingerprint is None:
            return None, "a hash of the source's own, but markers false: there is no fingerprint to hold it to"
        if event["hash"] != state.fingerprint:
            return None, (f"the source's own hash of this state is {str(event['hash'])[:40]!r}, the harness's "
                          f"fingerprint of it {state.fingerprint}: the two canonical forms differ")
    return state, ""


def _bad_timing(event: dict) -> bool:
    """A driver's own timing, if it gives one, is a duration in milliseconds."""
    if "ms" not in event:
        return False
    value = event["ms"]
    return not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0


def output_lines(done: Completed) -> list[tuple[str, float | None]]:
    """The non-empty lines a run printed, each with when it had arrived (None where that was not measured).

    A line has arrived when its last byte has - its newline, or for a last line
    cut off mid-write, the last byte printed.
    """
    data = done.stdout
    found = []
    start = 0
    while start < len(data):
        stop = data.find(b"\n", start)
        end = len(data) if stop < 0 else stop
        text = data[start:end].decode("utf-8", "replace").strip()
        if text:
            found.append((text, done.arrived(min(end, len(data) - 1))))
        start = end + 1
    return found


def watch_events(index: int, kind: str) -> Stop:
    """A stop condition for util.run: the run's ``kind`` event ("state" or "position") for ``index``, or a later one, has arrived."""
    pending = bytearray()
    needle = f'"{kind}"'.encode()

    def arrived(chunk: bytes) -> bool:
        pending.extend(chunk)
        while (cut := pending.find(b"\n")) >= 0:
            line = bytes(pending[:cut])
            del pending[:cut + 1]
            if needle not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == kind and isinstance(event.get("index"), int) \
                    and not isinstance(event.get("index"), bool) and event["index"] >= index:
                return True
        return False

    return arrived


def stop_condition(options: RunOptions) -> Stop | None:
    """The stop condition of a protocol run with these options, if it may be stopped early."""
    if options.stop_after is None:
        return None
    index, kind = options.stop_after
    if kind not in STOP_EVENTS:
        raise ValueError(f"stop_after names an event the protocol does not have: {kind!r}")
    return watch_events(index, kind)


def parse_jsonl(source: str, record_id: str, done: Completed, cap: int | None, elapsed: float,
                *, exact_fields: bool = False, setup_refusals: tuple[str, ...] = ()) -> Observation:
    """Turn a protocol run into an Observation, distrusting everything in it.

    ``exact_fields`` refuses state and position events that carry anything
    beyond the protocol's fields (the oracle drivers add timings; the engine
    may not). ``setup_refusals`` are the reasons with which this source may end
    "rejected" before its first position, refusing the record's set-up - only the
    engine may, as "unreadable" (observations.refused_setup).
    """
    positions: list[Position] = []
    arrival: list[float | None] = []  # when each position's last line arrived
    extras: dict[int, dict] = {}
    diagnostics: list[dict] = []
    pending: fingerprint.State | None = None  # the state event of the position still to come
    pending_at: float | None = None  # when it arrived
    markers: bool | None = None  # what the run's state events say of the unmoved markers, once it has printed one
    end: dict | None = None
    ended_at: float | None = None
    # Two kinds of bad output. An event that was read and breaks the protocol is
    # never excused. A line that cannot be read at all is excused only where a
    # killed process leaves one behind: as its last line, cut off mid-write.
    invalid = ""
    unreadable: list[int] = []  # the numbers of the lines that are not a JSON event
    lines = output_lines(done)
    for number, (line, at) in enumerate(lines):
        try:
            event = json.loads(line)
            kind = event["event"]
        except (ValueError, KeyError, TypeError):  # ValueError: not JSON, or an integer too long to convert
            unreadable.append(number)
            continue
        if kind == "state":
            state, problem = _state_problem(event, positions, pending, exact_fields, markers)
            if problem:
                invalid = f"invalid state event ({problem}): {line[:160]}"
                break
            pending, pending_at, markers = state, at, state.markers
        elif kind == "position":
            side, in_check = event.get("side"), event.get("in_check")
            count, terminal = event.get("legal_actions"), event.get("terminal")
            if (event.get("index") != len(positions) or side not in (*SIDES, None)
                    or terminal not in (*TERMINALS, None) or not isinstance(in_check, (bool, type(None)))
                    or not (count is None or (isinstance(count, int) and not isinstance(count, bool) and count >= 0))
                    or (exact_fields and set(event) != POSITION_FIELDS) or _bad_timing(event)):
                invalid = f"invalid position event: {line[:160]}"
                break
            if markers is not None and pending is None:
                invalid = (f"invalid position event (no state event for position {len(positions)}, though the run "
                           f"said what it set up at the positions before it): {line[:160]}")
                break
            index = len(positions)
            set_up = pending.summary() if pending is not None else {}
            positions.append(Position(index, side, in_check, count, terminal, **set_up))
            arrival.append(at)
            pending, pending_at = None, None
            extra = {key: value for key, value in event.items() if key not in POSITION_FIELDS}
            if extra:
                extras[index] = extra
        elif kind in DIAGNOSTIC_EVENTS:
            diagnostics.append(event)
        elif kind == "end":
            end, ended_at = event, at
            break

    # A state event whose position event never came: the search at that
    # position ran out of time or failed. What the source set up is known;
    # nothing it would have answered is.
    dangling = pending is not None
    if dangling:
        positions.append(Position(len(positions), **pending.summary()))
        arrival.append(pending_at)

    # Position k exists exactly when k actions were accepted: the positions are
    # what was observed, so they - not the run's own tally - say how far it got.
    reached = max(len(positions) - 1, 0)
    observation = Observation(source=source, record=record_id, status="error", positions=positions,
                              actions_accepted=reached, cap=cap, elapsed_s=elapsed, extras=extras,
                              diagnostics=diagnostics, arrival_s=arrival, ended_s=ended_at)
    if invalid:
        # Broken before it was killed, if it was: the time-out does not hide it.
        observation.reason = "malformed_output"
        killed = " - and the run was then killed for time" if done.timed_out else ""
        observation.detail = invalid + (" - and the run was then stopped" if done.stopped else killed)
        return observation
    cut_off = done.killed and unreadable == [len(lines) - 1]
    if unreadable and not cut_off:
        observation.reason = "malformed_output"
        observation.detail = f"unreadable output line: {lines[unreadable[0]][0][:120]}"
        return observation
    if done.killed:
        # Cut short: its time ran out, or the gate stopped it with everything it needed in hand.
        observation.status, observation.reason = "timeout", "stopped" if done.stopped else "timeout"
        observation.ended_s = None
        return observation
    if end is None:
        observation.reason = "crash"
        observation.detail = f"exit code {done.returncode}; {_tail(done.err_text())}"
        return observation

    status, accepted, reason = end.get("status"), end.get("actions_accepted"), str(end.get("reason", ""))
    problem = ""
    if status not in STATUSES or status == "timeout":
        problem = "unknown status"
    elif dangling and status != "error":
        problem = (f"the state event of position {reached} has no position event, yet the run says it ended "
                   f"'{status}' - only a run that failed during the search may leave one out")
    elif not isinstance(accepted, int) or isinstance(accepted, bool):
        problem = "actions_accepted is not a number"
    elif status != "error" and accepted != reached:
        problem = f"claims {accepted} accepted action(s) but reported {len(positions)} position(s)"
    elif status in ("ok", "rejected") and not positions and not (status == "rejected" and reason in setup_refusals):
        problem = "replayed the record without reporting even its starting position"
    elif status == "rejected" and reason not in (*HARD_REJECTIONS, *SOFT_REJECTIONS):
        problem = f"unknown rejection reason '{reason}'"
    if problem:
        observation.reason, observation.detail = "malformed_output", f"invalid end event ({problem}): {json.dumps(end)[:200]}"
        return observation
    observation.status = status
    observation.reason = reason
    observation.detail = str(end.get("detail", ""))
    observation.last_action = end.get("last_action") if isinstance(end.get("last_action"), bool) else None
    return observation


def run_protocol(source: str, record_id: str, argv: Sequence, data: bytes, options: RunOptions,
                 cap: int | None, cwd=None, env=None, *, exact_fields: bool = False,
                 setup_refusals: tuple[str, ...] = ()) -> Observation:
    started = time.monotonic()
    try:
        done = run(argv, stdin=data, timeout=options.timeout_s, cwd=cwd, env=env, stop=stop_condition(options))
    except OSError as error:
        return Observation(source=source, record=record_id, status="error", reason="spawn", detail=str(error),
                           actions_accepted=0, cap=cap)
    observation = parse_jsonl(source, record_id, done, cap, time.monotonic() - started, exact_fields=exact_fields,
                              setup_refusals=setup_refusals)
    if observation.ended_s is None and not done.killed:
        # It ended without an end event (it died): its outcome was known when the process ended.
        observation.ended_s = done.elapsed_s or None
    return observation
