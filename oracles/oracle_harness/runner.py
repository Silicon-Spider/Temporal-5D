"""Running sources over corpus records."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

from . import corpus, observations
from .adapters import RunOptions, Source
from .corpus import Record
from .lockfile import Lock
from .observations import Observation
from .util import Finding, kill_running

# (source name, record id) -> what that source said about that record.
Results = dict[tuple[str, str], Observation]


def default_jobs() -> int:
    # Every task is one single-threaded subprocess.
    return max(1, min(8, os.cpu_count() or 2))


def parallel_map[Item, Outcome](function: Callable[[Item], Outcome], items: Iterable[Item], jobs: int | None = None) -> list[Outcome]:
    """``function`` over ``items`` on a pool of threads, in order.

    If anything goes wrong - an interrupt, an exception - nothing still queued is
    started, and every command still running is killed before it propagates.
    """
    pool = ThreadPoolExecutor(max_workers=jobs or default_jobs())
    try:
        outcomes = list(pool.map(function, items))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        kill_running()
        raise
    pool.shutdown()
    return outcomes


def observe(source: Source, record_id: str, given: corpus.Feed, options: RunOptions) -> Observation:
    """What ``source`` says about one record, given it as the harness feeds every source (corpus.feed).

    A record no source is given - one in the official client's export form with
    a move the notation specification cannot express (export_form.py) - is
    unsupported for every source alike, with the reason, and the source is not
    run: the harness, not the source, refused it.
    """
    if given.unsupported:
        return Observation(source=source.name, record=record_id, status="unsupported", reason=corpus.EXPORT_FORM,
                           detail=given.unsupported, actions_accepted=0,
                           cap=options.cap if "legal_actions" in source.answers else None)
    return source.observe(record_id, given.data, options)


def run_sources(
    sources: list[Source],
    records: list[Record],
    lock: Lock,
    options_for: Callable[[Source, Record], RunOptions],
    *,
    jobs: int | None = None,
    progress: Callable[[Observation], None] | None = None,
) -> Results:
    """Ask every source about every record. Each (source, record) is one subprocess."""
    recorded = record_sources(sources, records, lock, options_for, repeat=1, jobs=jobs, progress=progress)
    return {key: runs[0] for key, runs in recorded.items()}


def record_sources(
    sources: list[Source],
    records: list[Record],
    lock: Lock,
    options_for: Callable[[Source, Record], RunOptions],
    *,
    repeat: int = 1,
    jobs: int | None = None,
    progress: Callable[[Observation], None] | None = None,
) -> dict[tuple[str, str], list[Observation]]:
    """Ask every source about every record ``repeat`` times, each time in a process of its own.

    Independent recordings are what tell a stable answer from one that depends
    on the run (H32). Every run of every (source, record) goes into one pool.
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    given = {record.id: corpus.feed(record, lock) for record in records}
    tasks = [(source, record) for _ in range(repeat) for source in sources for record in records]

    def one(task: tuple[Source, Record]) -> Observation:
        source, record = task
        observation = observe(source, record.id, given[record.id], options_for(source, record))
        if progress:
            progress(observation)
        return observation

    found: dict[tuple[str, str], list[Observation]] = {}
    for observation in parallel_map(one, tasks, jobs):
        found.setdefault((observation.source, observation.record), []).append(observation)
    return found


def self_consistency(results: Results) -> list[Finding]:
    """A source whose own report contradicts itself cannot be compared with anything."""
    findings = []
    for (source, record), observation in sorted(results.items()):
        for problem in observations.validate(observation):
            findings.append(Finding("self-contradiction", f"{source} on {record}: {problem}"))
    return findings
