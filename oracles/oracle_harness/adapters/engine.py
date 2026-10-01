"""The engine under test.

The gate reaches the engine only across a process boundary: a command, named in
gate.toml, that reads a record on stdin and speaks the protocol of base.py. The
harness never imports the engine and the engine never imports the harness, so
neither build can come to depend on the other.

While no command is configured there is no engine to test, the differential
gate has not run, and the proof channel stays disabled (decision H23).

The engine is held to a stricter contract than an oracle. An oracle may say
nothing about a question it cannot answer; the engine is the thing being
certified, so a silence from it would be a verdict that was never compared.
"""

from __future__ import annotations

import dataclasses
import shlex

from .. import paths
from ..observations import HARD_REJECTIONS, Observation
from .base import RunOptions, Source, run_protocol

_ENGINE_TERMINALS = ("none", "checkmate", "stalemate")


def contract_violation(observation: Observation) -> str | None:
    """What an engine report does that the engine contract forbids, if anything."""
    if observation.status in ("ok", "rejected"):
        for position in observation.positions:
            for name in ("side", "in_check", "legal_actions", "terminal"):
                if getattr(position, name) is None:
                    return f"position {position.index}: the engine must answer every observable, and `{name}` is null"
            if position.terminal not in _ENGINE_TERMINALS:
                return (f"position {position.index}: terminal is '{position.terminal}' - "
                        f"the engine must say which of {_ENGINE_TERMINALS} it is")
    if observation.status == "rejected" and observation.reason not in HARD_REJECTIONS:
        return (f"rejection reason '{observation.reason}' - when the engine refuses an action it is saying the "
                f"action is not legal, and the reason is one of {HARD_REJECTIONS}")
    if observation.status == "unsupported" and observation.positions:
        return ("`unsupported` after reporting positions - it is a statement about the ruleset, "
                "made before the first position")
    return None


class EngineSource(Source):
    name = "engine"
    oracle = None
    answers = ("side", "in_check", "legal_actions", "terminal", "legality")

    def __init__(self, command: list[str]):
        self.command = list(command)

    def available(self) -> str | None:
        if not self.command:
            return "no engine adapter is configured ([engine] command in oracles/gate.toml is empty)"
        return None

    def signature(self, options: RunOptions) -> dict:
        return {"command": shlex.join(self.command), "cap": options.cap}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        argv = [*self.command, "--cap", str(options.cap)]
        # Exactly the protocol's fields: an oracle driver's output passed off as
        # the engine's carries fields of its own, and is refused here.
        observation = run_protocol(self.name, record_id, argv, data, options, options.cap, cwd=paths.REPO_ROOT,
                                   exact_fields=True)
        violation = contract_violation(observation)
        if violation:
            # The positions are kept for whoever debugs the adapter; the status
            # makes sure nothing is concluded from them.
            return dataclasses.replace(observation, status="error", reason="malformed_output",
                                       detail=f"engine contract: {violation}")
        return observation
