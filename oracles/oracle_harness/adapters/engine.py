"""The engine under test.

The gate reaches the engine only across a process boundary: a command, named in
gate.toml, that reads a record on stdin and speaks the protocol of base.py. The
harness never imports the engine and the engine never imports the harness, so
neither build can come to depend on the other.

While no command is configured there is no engine to test, the differential
gate has not run, and the proof channel stays disabled (decision H23).

The engine is held to a stricter contract than an oracle. An oracle may say
nothing about a question it cannot answer; the engine is the thing being
certified, so a silence from it would be a verdict that was never compared. The
same goes for what it set up: an oracle that cannot export its state prints no
state event, but the engine prints one before every position it reports, with
the unmoved markers (markers true) - otherwise nothing would show that it and an
oracle are answering about the same position (decision H30).

In either mode the engine refuses an action only as unreadable (its parser could
not read the move text), illegal or unsubmittable. It may also refuse the
record's set-up as unreadable - its headers or board strings - ending "rejected"
with actions_accepted 0 before its first position: refusing "action 0", which
is compared like any refusal (observations.refused_setup). A refusal as
unreadable is no rules verdict: it is a soft legality divergence from every
oracle that played the action (or set the record up), which only a known
divergence citing the notation specification can excuse.

An engine that can read records but not yet judge them runs in parse mode
(gate.toml [engine] mode = "parse"): the harness appends "--parse-only", and the
engine says at every position what it set up and whose move it is, and answers
none of check, counts and mate - every one of them null, never a guess. The gate
then compares what it set up and how it replayed each record, and gates nothing.
"""

from __future__ import annotations

import dataclasses
import shlex

from .. import paths
from ..observations import HARD_REJECTIONS, OBSERVABLES, SETUP_REFUSALS, UNREADABLE, Observation
from .base import RunOptions, Source, run_protocol

MODES = ("full", "parse")
_ENGINE_TERMINALS = ("none", "checkmate", "stalemate")
# Why an engine may refuse an action, in either mode: it could not read the move
# text (or, before its first position, the record's set-up), or what it read
# cannot be played or submitted.
REJECTIONS = (UNREADABLE, *HARD_REJECTIONS)
PARSE_FLAG = "--parse-only"


def contract_violation(observation: Observation, mode: str = "full") -> str | None:
    """What an engine report does that the engine contract (in that mode) forbids, if anything."""
    if observation.status in ("ok", "rejected"):
        for position in observation.positions:
            if position.fingerprint is None:
                return (f"position {position.index}: the engine must say what it set up there - a state event "
                        "with markers true before the position event")
            if position.side is None:
                return f"position {position.index}: the engine must answer every observable, and `side` is null"
            if mode == "parse":
                answered = [name for name in OBSERVABLES if getattr(position, name) is not None]
                if answered:
                    return (f"position {position.index}: in parse mode the engine answers no rules question, and "
                            f"`{answered[0]}` is not null")
                continue
            for name in OBSERVABLES:
                if getattr(position, name) is None:
                    return f"position {position.index}: the engine must answer every observable, and `{name}` is null"
            if position.terminal not in _ENGINE_TERMINALS:
                return (f"position {position.index}: terminal is '{position.terminal}' - "
                        f"the engine must say which of {_ENGINE_TERMINALS} it is")
    if observation.status == "rejected" and observation.reason not in REJECTIONS:
        return (f"rejection reason '{observation.reason}' - the engine refuses an action as one of {REJECTIONS}: "
                f"'{UNREADABLE}' when it cannot read the move text (or the set-up), otherwise because the action is "
                "not legal")
    if observation.status == "unsupported" and observation.positions:
        return ("`unsupported` after reporting positions - it is a statement about the ruleset, "
                "made before the first position")
    return None


class EngineSource(Source):
    name = "engine"
    oracle = None
    answers = ("state", "side", "in_check", "legal_actions", "terminal", "legality")

    def __init__(self, command: list[str], mode: str = "full"):
        if mode not in MODES:
            raise ValueError(f"engine mode must be one of {MODES}, not {mode!r}")
        self.command = list(command)
        self.mode = mode

    def available(self) -> str | None:
        if not self.command:
            return "no engine adapter is configured ([engine] command in oracles/gate.toml is empty)"
        return None

    def signature(self, options: RunOptions) -> dict:
        return {"command": shlex.join(self.command), "cap": options.cap, "mode": self.mode}

    def argv(self, options: RunOptions) -> list[str]:
        return [*self.command, "--cap", str(options.cap), *([PARSE_FLAG] if self.mode == "parse" else [])]

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        # Exactly the protocol's fields, in state and position events alike: an
        # oracle driver's output passed off as the engine's carries fields of its
        # own, and is refused here. The engine is never stopped early. In either
        # mode it may refuse a record's set-up, as unreadable.
        options = dataclasses.replace(options, stop_after=None)
        observation = run_protocol(self.name, record_id, self.argv(options), data, options, options.cap,
                                   cwd=paths.REPO_ROOT, exact_fields=True, setup_refusals=SETUP_REFUSALS)
        violation = contract_violation(observation, self.mode)
        if violation:
            # The positions are kept for whoever debugs the adapter; the status
            # makes sure nothing is concluded from them.
            return dataclasses.replace(observation, status="error", reason="malformed_output",
                                       detail=f"engine contract: {violation}")
        return observation
