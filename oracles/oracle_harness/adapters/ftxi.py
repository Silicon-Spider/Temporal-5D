"""ftxi/5dchess_engine: the per-commit oracle.

Answers every observable. Two action searches are exposed as separate sources -
the hypercuboid search ("ftxi") and the plain depth-first one ("ftxi:naive") -
because they are different algorithms over the same rules, and the second is the
check on the first that does not share the hypercuboid method with cwmtt.

ftxi needs no guard against misreading a record's variant: it refuses a Board it
does not know, reads a Variant header in its place (and so does the harness),
reads the Size, Timeline and Promotions headers, and reads 5DFEN boards whatever
the header says. It needs one against a record that ends in a parenthesised
variation: its replay follows the last continuation written at each point, which
there is the variation.

At every position the driver also says what ftxi holds there (decision H30): every
board, as ftxi's own 5DFEN writer prints its pieces, with ftxi's turn, colour and
unmoved markers and the timeline named from ftxi's own parity. Its position event
repeats the board size beside its timing, for the driver check (fidelity.py). When
`observe` asks, it adds two diagnostics, never compared: the timelines ftxi
advances to phantom boards for its check test and the royal captures it finds
there (--explain), and the legal actions at one position as the search counts
them (--actions K). The driver's header says how each is obtained.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import build, fingerprint, paths
from ..lockfile import Lock
from ..observations import Observation
from ..util import Completed, run
from . import base
from .base import RunOptions, Source, code_sha256, run_protocol

_POLICIES = {"ftxi": "balanced", "ftxi:naive": "naive"}
_COMMENT = re.compile(rb"\{[^{}]*\}")
_VARIATION = re.compile(rb"\(\s*\d+\s*[wbWB]?\s*\.")
_NOTHING_MORE = re.compile(rb"\s*(1-0|0-1|1/2-1/2|\*)?\s*")


def ends_in_a_variation(data: bytes) -> bool:
    """Does the move text end with a parenthesised variation, and no main line after it?

    ftxi's grammar writes a variation before the continuation it is an
    alternative to, and its replay takes the last continuation at every point.
    With no continuation after it, the last one is the variation.
    """
    text = _COMMENT.sub(b" ", data)
    openings = list(_VARIATION.finditer(text))
    if not openings:
        return False
    depth = 0
    for index in range(openings[-1].start(), len(text)):
        depth += {ord("("): 1, ord(")"): -1}.get(text[index], 0)
        if depth == 0:
            return _NOTHING_MORE.fullmatch(text, index + 1) is not None
    return True  # never closed: nothing after it can be a main line


def _turn(actions: int) -> str:
    """The stock CLI's name for the state after ``actions`` submitted actions."""
    return "0" if actions == 0 else f"{(actions + 1) // 2}{'w' if actions % 2 else 'b'}"


class FtxiSource(Source):
    oracle = "ftxi"
    answers = ("state", "side", "in_check", "legal_actions", "terminal", "legality")
    # The driver speaks the common flags (--explain, --actions K), so the base
    # class's diagnostic_args() asks for these.
    diagnostics = ("explain", "actions")

    def __init__(self, name: str, lock: Lock):
        self.name = name
        self.policy = _POLICIES[name]
        self.pin = lock.pin("ftxi")

    def _build(self) -> build.Build | None:
        return build.load(self.pin)

    def available(self) -> str | None:
        return None if self._build() else "ftxi is not built (run: python oracles/harness.py build --oracle ftxi)"

    def signature(self, options: RunOptions) -> dict:
        driver = paths.DRIVERS / "ftxi"
        return {"pin": self.pin.commit, "policy": self.policy, "cap": options.cap,
                "driver_sha256": code_sha256(driver / "ftxi_driver.cpp", driver / "CMakeLists.txt"),
                "adapter_sha256": code_sha256(Path(__file__), Path(base.__file__), Path(fingerprint.__file__))}

    def observe(self, record_id: str, data: bytes, options: RunOptions) -> Observation:
        if ends_in_a_variation(data):
            return Observation(source=self.name, record=record_id, status="unsupported", reason="notation",
                               detail="the record ends with a parenthesised variation, which ftxi replays as the main line",
                               actions_accepted=0, cap=options.cap)
        driver = self._build().artifact("driver")
        argv = [driver, "--cap", str(options.cap), "--policy", self.policy, *self.diagnostic_args(options)]
        return run_protocol(self.name, record_id, argv, data, options, options.cap)

    # The calls below go to the oracle's stock CLI. They exist so the harness
    # can prove its own driver reports what the unmodified tool reports.

    def stock_perftest(self, data: bytes, timeout_s: float) -> Completed:
        return run([self._build().artifact("tools"), "perftest", self.policy], stdin=data, timeout=timeout_s)

    def stock_count(self, data: bytes, actions: int, cap: int, timeout_s: float) -> Completed:
        argv = [self._build().artifact("tools"), "count", "--at", _turn(actions), "--format", "json", self.policy, str(cap)]
        return run(argv, stdin=data, timeout=timeout_s)

    def stock_stats(self, data: bytes, actions: int, timeout_s: float) -> Completed:
        argv = [self._build().artifact("tools"), "stats", "--at", _turn(actions), "--format", "json"]
        return run(argv, stdin=data, timeout=timeout_s)

    def stock_print(self, data: bytes, actions: int, timeout_s: float) -> Completed:
        """Every board ftxi holds after ``actions`` actions, drawn by its own `print` command."""
        return run([self._build().artifact("tools"), "print", "--at", _turn(actions)], stdin=data, timeout=timeout_s)
