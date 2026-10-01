"""The interface every source of observations implements, and the JSON Lines
protocol the process-based ones (harness drivers and the engine) speak.

Protocol - the record arrives on stdin, one JSON object per line leaves on stdout:

    {"event":"start", ...}                                   optional, informational
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
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..observations import HARD_REJECTIONS, SIDES, SOFT_REJECTIONS, STATUSES, TERMINALS, Observation, Position
from ..util import Completed, run


@dataclass(frozen=True)
class RunOptions:
    cap: int = 1000
    timeout_s: float = 120.0
    mate_budget_ms: int = 5000


class Source:
    """One way of asking "what does this implementation say about this record?"."""

    name: str = ""
    oracle: str | None = None  # pin id whose build this source needs; None for the engine
    answers: tuple[str, ...] = ()  # observables this source can report

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


def parse_jsonl(source: str, record_id: str, done: Completed, cap: int | None, elapsed: float,
                *, exact_fields: bool = False) -> Observation:
    """Turn a protocol run into an Observation, distrusting everything in it.

    ``exact_fields`` refuses position events that carry anything beyond the
    protocol's fields (the oracle drivers add timings; the engine may not).
    """
    positions: list[Position] = []
    end: dict | None = None
    malformed = ""
    for raw in done.out_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            kind = event["event"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # A truncated final line is what a killed process leaves behind.
            malformed = malformed or f"unreadable output line: {line[:120]}"
            continue
        if kind == "position":
            side, in_check = event.get("side"), event.get("in_check")
            count, terminal = event.get("legal_actions"), event.get("terminal")
            if (event.get("index") != len(positions) or side not in (*SIDES, None)
                    or terminal not in (*TERMINALS, None) or not isinstance(in_check, (bool, type(None)))
                    or not (count is None or (isinstance(count, int) and not isinstance(count, bool) and count >= 0))
                    or (exact_fields and set(event) != POSITION_FIELDS)):
                malformed = malformed or f"invalid position event: {line[:160]}"
                break
            positions.append(Position(len(positions), side, in_check, count, terminal))
        elif kind == "end":
            end = event
            break

    # Position k exists exactly when k actions were accepted: the positions are
    # what was observed, so they - not the run's own tally - say how far it got.
    reached = max(len(positions) - 1, 0)
    observation = Observation(source=source, record=record_id, status="error", positions=positions,
                              actions_accepted=reached, cap=cap, elapsed_s=elapsed)
    if done.timed_out:
        observation.status, observation.reason = "timeout", "timeout"
        return observation
    if end is None or malformed:
        observation.reason = "malformed_output" if malformed else "crash"
        observation.detail = malformed or f"exit code {done.returncode}; {_tail(done.err_text())}"
        return observation

    status, accepted, reason = end.get("status"), end.get("actions_accepted"), str(end.get("reason", ""))
    problem = ""
    if status not in STATUSES or status == "timeout":
        problem = "unknown status"
    elif not isinstance(accepted, int) or isinstance(accepted, bool):
        problem = "actions_accepted is not a number"
    elif status != "error" and accepted != reached:
        problem = f"claims {accepted} accepted action(s) but reported {len(positions)} position(s)"
    elif status in ("ok", "rejected") and not positions:
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
                 cap: int | None, cwd=None, env=None, *, exact_fields: bool = False) -> Observation:
    started = time.monotonic()
    try:
        done = run(argv, stdin=data, timeout=options.timeout_s, cwd=cwd, env=env)
    except OSError as error:
        return Observation(source=source, record=record_id, status="error", reason="spawn", detail=str(error),
                           actions_accepted=0, cap=cap)
    return parse_jsonl(source, record_id, done, cap, time.monotonic() - started, exact_fields=exact_fields)
