"""Command line of the oracle harness: `python oracles/harness.py <command>`."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import unittest

from . import boundary, build, config, corpus, gate, lockfile, observations, paths, provenance, toolchain
from .adapters import ORACLE_SOURCES, RunOptions, make_source
from .util import Finding, HarnessError, format_findings, kill_running


def _say(message: str) -> None:
    # Flushed so that progress shows up in CI logs while a long run is going.
    print(message, flush=True)


def _report(sections: list[tuple[str, list[Finding]]]) -> int:
    for title, findings in sections:
        print(format_findings(title, findings))
    return 1 if any(findings for _, findings in sections) else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    for name, state in toolchain.doctor(lock):
        print(f"{name:<6} {state}")
    for pin in lock.oracles():
        built = build.load(pin)
        print(f"{pin.id:<10} {'built (' + built.key + ')' if built else 'not built'}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    lockfile.sync(lock)
    return _report([("pins", lockfile.verify(lock))])


def cmd_verify(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    sections = gate.static_checks(lock, config.load(), corpus.load(), include_untracked=not args.tracked_only)
    return _report(sections)


def cmd_boundary(args: argparse.Namespace) -> int:
    findings = boundary.check(lockfile.load(), config.load().boundary, include_untracked=not args.tracked_only)
    return _report([("boundary", findings)])


def cmd_build(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    blocking = lockfile.verify(lock)
    if blocking:
        return _report([("pins", blocking)])
    for oracle in args.oracle or build.buildable(lock):
        built = build.build(lock.pin(oracle), lock, bootstrap=args.bootstrap, force=args.force, log=_say)
        for key, value in built.info.items():
            print(f"  {key}: {value}")
    return 0


def cmd_observe(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    cfg = config.load()
    record = corpus.select(corpus.load(), corpus.TIERS, [args.record])[0]
    source = make_source(args.source, lock)
    reason = source.available()
    if reason:
        raise HarnessError(reason)
    options = RunOptions(cap=cfg.cap, timeout_s=args.timeout, mate_budget_ms=args.mate_budget_ms)
    observation = source.observe(record.id, corpus.read(record, lock), options)
    print(json.dumps({"source": observation.source, "record": observation.record, "cap": observation.cap,
                      "elapsed_s": round(observation.elapsed_s, 3), **observations.to_json(observation)}, indent=1))
    problems = observations.validate(observation)
    for problem in problems:
        print(f"self-contradiction: {problem}", file=sys.stderr)
    return 1 if problems else 0


def cmd_corpus(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    records = corpus.load()
    excluded = corpus.load_excluded()
    if args.unlisted:
        for pin_id, path, blob in corpus.unlisted(records, lock, excluded):
            print(json.dumps(corpus.describe_candidate(pin_id, path, blob, lock)))
        return 0
    for record in records:
        print(f"{record.id:<42} {record.tier:<8} {record.board}")
    return _report([("corpus", corpus.verify(records, lock) + corpus.verify_local(records) + corpus.verify_excluded(excluded, lock))])


def cmd_golden(args: argparse.Namespace) -> int:
    if args.write:
        findings = gate.write_golden(only_sources=args.source, jobs=args.jobs, log=_say)
        if findings:
            return _report([("golden --write", findings)])
        print("Golden observations rewritten. Review the diff before committing: apart from the signature line, every\n"
              "changed line is an answer that changed - or, for a run that timed out, a change in how far it got.")
        return 0
    gate.render_disagreements(log=_say)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    result = gate.run_tier(args.tier, only_sources=args.source, only_records=args.record, jobs=args.jobs,
                           include_untracked=not args.tracked_only, log=_say)
    print()
    print(gate.summarise(result))
    return 0 if result.ok else 1


def cmd_provenance(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    records = corpus.load()
    if args.write:
        changed = provenance.write(lock, records)
        print("PROVENANCE.md inventory " + ("updated" if changed else "already up to date"))
        return 0
    return _report([("provenance", provenance.check(lock, records))])


def cmd_selftest(args: argparse.Namespace) -> int:
    """The harness's own unit tests (they need git; the live ones need built oracles)."""
    tests = paths.HARNESS_ROOT / "tests"
    sys.path.insert(0, str(tests))
    suite = unittest.defaultTestLoader.discover(str(tests), pattern=args.pattern, top_level_dir=str(tests))
    outcome = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if outcome.wasSuccessful() else 1


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="harness.py",
        description="Differential-testing harness over the pinned 5D chess rules oracles.",
    )
    commands = top.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    add("doctor", cmd_doctor, "Show which build tools and oracle builds are present.")
    add("sync", cmd_sync, "Check out every pinned submodule at its locked commit (never recursively).")

    # What CI checks: the files git tracks. Their contents are still read from the
    # work tree, which in a fresh checkout is what was committed.
    tracked_only = "boundary check: skip untracked files (tracked ones are read as they are in the work tree)"

    sub = add("verify", cmd_verify, "Static checks: pins, corpus, known divergences, boundary, provenance, golden files.")
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("boundary", cmd_boundary, "Check that nothing outside oracles/ depends on the harness or on an oracle.")
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("build", cmd_build, "Build oracles from their locked commits into the harness cache.")
    sub.add_argument("--oracle", action="append", help="oracle to build (default: all)")
    sub.add_argument("--bootstrap", action="store_true",
                     help="fetch missing pinned tools (CMake/Ninja wheels, GHC archive) into the harness cache")
    sub.add_argument("--force", action="store_true", help="rebuild even if an up-to-date build exists")

    sub = add("observe", cmd_observe, "Print what one source says about one corpus record.")
    sub.add_argument("--source", required=True, choices=ORACLE_SOURCES)
    sub.add_argument("--record", required=True)
    sub.add_argument("--timeout", type=float, default=600.0)
    sub.add_argument("--mate-budget-ms", type=int, default=20000)

    sub = add("corpus", cmd_corpus, "List and verify the pinned corpus.")
    sub.add_argument("--unlisted", action="store_true", help="print manifest stubs for records in pinned sources that are not listed")

    sub = add("golden", cmd_golden, "Record what the oracles say (--write), or re-render DISAGREEMENTS.md (--render).")
    mode = sub.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="re-run the oracles over the whole corpus and rewrite the golden files")
    mode.add_argument("--render", action="store_true", help="only re-render DISAGREEMENTS.md from the golden files")
    sub.add_argument("--source", action="append", choices=ORACLE_SOURCES, help="limit --write to these sources")
    sub.add_argument("--jobs", type=int)

    sub = add("gate", cmd_gate, "Run a gate tier: static checks, oracle runs against golden, engine differential.")
    sub.add_argument("--tier", required=True)
    sub.add_argument("--source", action="append", help="restrict to these sources (marks the run partial)")
    sub.add_argument("--record", action="append", help="restrict to these records (marks the run partial)")
    sub.add_argument("--jobs", type=int)
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("provenance", cmd_provenance, "Check (default) or rewrite (--write) the generated inventory in PROVENANCE.md.")
    sub.add_argument("--write", action="store_true")

    sub = add("selftest", cmd_selftest, "Run the harness's own unit tests.")
    sub.add_argument("--pattern", default="test_*.py")
    sub.add_argument("-v", "--verbose", action="store_true")
    return top


def _interrupted(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Being told to stop (a CI job cancelled, a terminal closed) is handled like
    # Ctrl-C: the oracles run in process groups of their own, and would otherwise
    # outlive the harness.
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _interrupted)
    try:
        return args.handler(args)
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        kill_running()
        print("interrupted", file=sys.stderr)
        return 130
