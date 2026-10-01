"""Does the harness's ftxi driver report what ftxi's own tools report?

The driver exists to ask what the stock CLI does not answer in one pass (the
check verdict at every position). For everything the stock CLI does answer, the
two must agree exactly - otherwise a bug in harness glue would be passed off as
the oracle's verdict. What can be cross-checked this way:

  * whether a legal action exists at each position, and the verdict where none
    does (`5dtools perftest`);
  * the number of legal actions (`5dtools count --at`);
  * the side to move (`5dtools stats --at`).

The check verdict at a position that still has a legal action cannot be: no
stock command prints it. It comes from the same library call the stock perftest
makes at terminal positions (has_phantom_check), which is the most the harness
can say for it.

cwmtt needs no such check (its stock CLI is used directly). 5d-chess-js ships no
command that reports verdicts to compare against.
"""

from __future__ import annotations

import json
import re

from . import corpus
from .adapters.ftxi import FtxiSource
from .corpus import Record
from .lockfile import Lock
from .observations import Observation
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
        try:
            side = None if done.timed_out else json.loads(done.out_text())["side_to_move"]
        except (json.JSONDecodeError, KeyError, TypeError):
            side = None
        if side is None:
            report(f"position {index}: stock `stats --at` gave no side to move")
        elif side != position.side:
            report(f"position {index}: driver says {position.side} to move, stock stats says {side}")
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
