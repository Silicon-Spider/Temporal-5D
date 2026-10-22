"""Command line of the oracle harness: `python oracles/harness.py <command>`."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import unittest

from . import (
    baseline,
    boundary,
    build,
    config,
    corpus,
    fingerprint,
    gate,
    golden,
    lockfile,
    observations,
    paths,
    provenance,
    rulebook,
    runner,
    snapshot,
    stability,
    toolchain,
)
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
    cfg = config.load()
    records = corpus.load()
    frozen = snapshot.take(baseline_pin=cfg.baseline)
    sections = gate.static_checks(lock, cfg, records, frozen, include_untracked=not args.tracked_only)
    code = _report(sections)
    # What each tier requires of each source, and why the rest is not required (H32).
    print()
    for line in stability.describe(cfg, records, frozen):
        print(line)
    # Whether the tier that can enable the proof channel requires enough to enable it. A
    # complete run of that tier fails on these; verify, which every tier runs, reports them
    # (and a restricted run, which can enable nothing, does not make the check - H47).
    readiness = gate.proof_channel_readiness(cfg, records, frozen)
    if readiness is not None:
        print()
        print(format_findings(f"{gate.PROOF_SECTION} (a complete run of such a tier fails on these; verify does not)",
                              readiness))
    return code


def cmd_boundary(args: argparse.Namespace) -> int:
    findings = boundary.check(lockfile.load(), config.load().boundary, include_untracked=not args.tracked_only)
    return _report([("boundary", findings)])


def cmd_build(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    blocking = lockfile.verify(lock)
    if blocking:
        return _report([("pins", blocking)])
    if args.plan:
        # One `key=value` line, so CI can append it to its step outputs.
        print(f"tools={','.join(build.tools_needed(lock, args.oracle or build.buildable(lock)))}")
        return 0
    for oracle in args.oracle or build.buildable(lock):
        built = build.build(lock.pin(oracle), lock, bootstrap=args.bootstrap, force=args.force, log=_say)
        for key, value in built.info.items():
            print(f"  {key}: {value}")
    return 0


def _diagnostic_line(event: dict) -> str:
    """A diagnostic event as `observe` prints it: an actions event with its lists sorted, as the harness sorts them."""
    if event.get("event") == "actions":
        try:
            event = fingerprint.canonical_actions(event)
        except ValueError as error:  # a StateError, or anything else a driver's output trips
            print(f"malformed actions event, shown as printed: {error}", file=sys.stderr)
    return json.dumps(event)


def cmd_observe(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    cfg = config.load()
    record = corpus.select(corpus.load(), corpus.TIERS, [args.record])[0]
    source = make_source(args.source, lock)
    reason = source.available()
    if reason:
        raise HarnessError(reason)
    asked = [name for name, wanted in (("explain", args.explain), ("actions", args.actions is not None)) if wanted]
    for name in asked:
        if name not in source.diagnostics:
            print(f"note: {source.name} prints no {name} events", file=sys.stderr)
    options = RunOptions(cap=cfg.cap, timeout_s=args.timeout, mate_budget_ms=args.mate_budget_ms,
                         explain=args.explain, actions_at=args.actions)
    given = corpus.feed(record, lock)
    if given.converted:
        print(f"note: {record.id} is {record.reading} ({given.converted} moves rewritten); the source reads that",
              file=sys.stderr)
    observation = runner.observe(source, record.id, given, options)
    print(json.dumps({"source": observation.source, "record": observation.record, "cap": observation.cap,
                      "elapsed_s": round(observation.elapsed_s, 3), **observations.to_json(observation)}, indent=1))
    problems = observations.validate(observation)
    for problem in problems:
        print(f"self-contradiction: {problem}", file=sys.stderr)
    # The diagnostics come after the report, one JSON object per line. They are
    # never compared, and never recorded.
    for event in observation.diagnostics:
        print(_diagnostic_line(event))
    listed = any(event.get("event") == "actions" for event in observation.diagnostics)
    if args.actions is not None and "actions" in source.diagnostics and not listed:
        reached = f"positions 0 to {len(observation.positions) - 1}" if observation.positions else "no position"
        print(f"note: no actions event for position {args.actions}; the run reported {reached} "
              f"({observation.status}{': ' + observation.reason if observation.reason else ''})", file=sys.stderr)
    return 1 if problems else 0


def _position_index(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("a position is a whole number from 0")
    return value


def _at_least_one(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("a whole number from 1")
    return value


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
        findings = gate.write_golden(only_sources=args.source, jobs=args.jobs, repeat=args.repeat, log=_say)
        if findings:
            return _report([("golden --write", findings)])
        print("Golden observations rewritten. Review the diff before committing: apart from the signature line, every\n"
              "changed line of a golden file is an answer that changed - or, for a run cut at its budget, a change in\n"
              "how far it got (its horizon), a value a limit left unknown, or one that differed between recordings\n"
              "(unstable). A horizon that moved earlier, a verdict or an outcome that went (listed above, if any) and a\n"
              "changed line of 'What each tier requires' in DISAGREEMENTS.md are losses of checking, not churn: never\n"
              "commit one to make a gate run pass. The cost files and MANIFEST change with every recording.")
        return 0
    gate.render_disagreements(log=_say)
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    result = gate.run_tier(args.tier, only_sources=args.source, only_records=args.record, jobs=args.jobs,
                           include_untracked=not args.tracked_only, log=_say)
    print()
    print(gate.summarise(result))
    return 0 if result.ok else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    """The engine's baseline (H47): check it (default), or record it from a green run of its tier (--write)."""
    if args.write:
        result, changed = gate.write_baseline(jobs=args.jobs, include_untracked=not args.tracked_only, log=_say)
        print()
        print(gate.summarise(result))
        print()
        if changed is None:
            print("The run is not green: no baseline was recorded. The engine's baseline holds what it read on a green "
                  "run of its tier, and nothing else (H47).")
            return 1
        print("The engine's baseline is recorded and pinned. Review the diff before committing: every changed line of\n"
              f"{paths.HARNESS_REL}/{paths.ENGINE_BASELINE_FILE.name} is a record the engine now reads otherwise. A LOSS below is a\n"
              "record the commit tier no longer holds the engine to - a loss of checking, never churn: never commit\n"
              "one to make a gate run pass.")
        for line in changed:
            print(f"  {line}")
        return 0
    cfg = config.load()
    frozen = snapshot.take(baseline_pin=cfg.baseline)
    findings = baseline.static_findings(cfg, corpus.load(), frozen.baseline, frozen.baseline_problems)
    loaded = frozen.baseline
    if not findings and loaded is not None:
        print(f"{paths.ENGINE_BASELINE_FILE.name} is the baseline gate.toml pins ({loaded.digest[:12]}): the green run "
              f"of the {loaded.tier} tier recorded {loaded.recorded}, in which the engine replayed {loaded.read()} of "
              f"{len(loaded.records)} records")
    elif not findings:
        print("no baseline is pinned, and none is needed: no engine is compared in full mode")
    return _report([("engine baseline", findings)])


def cmd_provenance(args: argparse.Namespace) -> int:
    lock = lockfile.load()
    records = corpus.load()
    if args.write:
        changed = provenance.write(lock, records)
        print("PROVENANCE.md inventory " + ("updated" if changed else "already up to date"))
        return 0
    return _report([("provenance", provenance.check(lock, records))])


def cmd_rulebook(args: argparse.Namespace) -> int:
    """The rulebook pin (H38): check it (default), or move it to the rulebook's current text (--pin)."""
    spec = config.load().rulebook
    if args.pin:
        old, new = rulebook.pin()
        if old == new:
            print(f"{spec.path} is already pinned at {new}")
            return 0
        print(f"{spec.path}: pin moved from {old} to {new} in {paths.HARNESS_REL}/gate.toml [rulebook].\n"
              "The rulebook decides every disagreement between the engine and an oracle (H38): review its diff,\n"
              "re-check every known divergence that cites it, re-run the gate (gate --tier commit here; the nightly\n"
              "tier in CI), and record the change in CHANGELOG.md.")
        return 0
    findings = rulebook.verify(spec)
    if not findings:
        print(f"{spec.path} is the text gate.toml pins ({spec.sha256})")
    return _report([("rulebook", findings)])


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

    sub = add("verify", cmd_verify,
              "Static checks: pins, corpus, known divergences, the rulebook pin, boundary, provenance, golden files.")
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("boundary", cmd_boundary, "Check that nothing outside oracles/ depends on the harness or on an oracle.")
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("build", cmd_build, "Build oracles from their locked commits into the harness cache.")
    sub.add_argument("--oracle", action="append", help="oracle to build (default: all)")
    sub.add_argument("--bootstrap", action="store_true",
                     help="fetch missing pinned tools (CMake/Ninja wheels, GHC archive) into the harness cache")
    sub.add_argument("--force", action="store_true", help="rebuild even if an up-to-date build exists")
    sub.add_argument("--plan", action="store_true",
                     help="build nothing; print the build tools still needed, as one line `tools=<names>`")

    sub = add("observe", cmd_observe, "Print what one source says about one corpus record.")
    sub.add_argument("--source", required=True, help=f"the source to ask ({', '.join(ORACLE_SOURCES)}, ...)")
    sub.add_argument("--record", required=True)
    sub.add_argument("--timeout", type=float, default=600.0)
    sub.add_argument("--mate-budget-ms", type=int, default=20000)
    sub.add_argument("--explain", action="store_true",
                     help="also print what the source's check test looked at, after each position's state (diagnostic)")
    sub.add_argument("--actions", type=_position_index, metavar="K",
                     help="also print the source's legal actions at position K, sorted (diagnostic)")

    sub = add("corpus", cmd_corpus, "List and verify the pinned corpus.")
    sub.add_argument("--unlisted", action="store_true", help="print manifest stubs for records in pinned sources that are not listed")

    sub = add("golden", cmd_golden, "Record what the oracles say (--write), or re-render DISAGREEMENTS.md (--render).")
    mode = sub.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="re-run the oracles over the whole corpus and rewrite the golden files")
    mode.add_argument("--render", action="store_true", help="only re-render DISAGREEMENTS.md from the golden files")
    sub.add_argument("--source", action="append", choices=ORACLE_SOURCES, help="limit --write to these sources")
    sub.add_argument("--repeat", type=_at_least_one, default=golden.MIN_RECORDINGS, metavar="N",
                     help="record each source N times, in independent processes; a value that differs between them "
                          f"is recorded as unstable (default {golden.MIN_RECORDINGS}, the fewest verify accepts)")
    sub.add_argument("--jobs", type=int)

    sub = add("gate", cmd_gate, "Run a gate tier: static checks, oracle runs against golden, engine differential.")
    sub.add_argument("--tier", required=True)
    restricted = ("a restricted run never produces a tier verdict: it reports its findings, labelled "
                  f"'{gate.RESTRICTED}', and exits 1 only on a finding")
    sub.add_argument("--source", action="append", help=f"restrict to these sources ({restricted})")
    sub.add_argument("--record", action="append", help=f"restrict to these records ({restricted})")
    sub.add_argument("--jobs", type=int)
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("baseline", cmd_baseline,
              "Check the engine's baseline - what it read on the last green run (H47) - or record it (--write).")
    sub.add_argument("--write", action="store_true",
                     help="run the baseline's tier (gate.toml [baseline]) in full with the engine and, only if the run "
                          "is green, record what the engine read and pin it in gate.toml")
    sub.add_argument("--jobs", type=int)
    sub.add_argument("--tracked-only", action="store_true", help=tracked_only)

    sub = add("rulebook", cmd_rulebook, "Check the rulebook pin (H38), or move it to the rulebook's current text (--pin).")
    sub.add_argument("--pin", action="store_true",
                     help="write the rulebook's current SHA-256 into gate.toml [rulebook] - after reviewing the change")

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
