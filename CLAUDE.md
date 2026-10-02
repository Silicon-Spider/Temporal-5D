# Temporal 5D chess engine

A neural engine for *5D Chess With Multiverse Time Travel*: a transformer evaluator
inside an Lc0/AlphaZero-style search, with an exact rules generator and a tactical
verification layer beside it.

## State of the repository

**Specification, the oracle layer and gate, and the rules engine's first stages (`engine/`). The network (`python/`) is being built.**

| Path | What it is |
|---|---|
| `docs/design/5d-chess-transformer-design.md` | The design spec: the reasoning. §11.0 is the rules engine and its gate; §15 is the build plan. |
| `docs/design/5d-chess-decisions.md` | The decision register: the *state* of every decision (committed, provisional, refused, ...). |
| `docs/design/scalable-transformers-architecture.md` | The general pattern the design instantiates. |
| `oracles/` | The differential-testing harness over three pinned, independent rules implementations (decision H27). Its manual is `oracles/README.md`. |
| `PROVENANCE.md` | What came from where, under which licence, in which role. |
| `CONTRIBUTING.md`, `AGENTS.md` | The rules around the oracles, for people and for agents. |
| `CHANGELOG.md` | Every change, newest first, with what it changed in the spec. |
| `docs/BUILD_LOG.md` | **Start here when resuming the build.** What is done, parked and next, keyed by register ID; the standing rules of the build; proposed spec edits. |
| `docs/MEASUREMENTS.md` | Every measured figure with its conditions, and the §15.6 verdicts. |
| `docs/rules/README.md` | The cited rules reference the engine is written from (proposed as the rulebook); open questions Q1-Q28. |
| `docs/checkpoints/` | Checkpoint reports - CHECKPOINT 1, the check disagreements, awaits the user's ruling. |
| `engine/` | The rules engine, C++20 (H24): immutable boards, the multiverse, 5DFEN, the H30 fingerprint, 5D Zobrist (H18), the turn rules, pseudo-legal move generation, the 5dpgn parser (H28), check status and submission legality (H36), legal actions, the naive and hypercuboid mate detectors (H37), the expansion and its measurement tools, and the gate adapter (parse and full mode). Its manual is `engine/README.md`. |

The build order is spec §15.1; where the build stands, and what is parked and why,
is in `docs/BUILD_LOG.md`. Build and test the engine with `py engine/build.py
--test` (CMake and Ninja from the project's `.venv`, see `engine/README.md`); the
engine is written from `docs/rules/README.md` and the notation specification text,
never from an oracle.

## Rules you must follow

@AGENTS.md

## Every change: the changelog and the spec

Every change is recorded in `CHANGELOG.md`, in the same edit. The spec and the
register are maintained upstream and **never edited here**: an edit they need is
proposed in `docs/BUILD_LOG.md` and applied upstream. The rule in full is the first
section of `AGENTS.md`, imported above; it binds you like the rest of that file.

## Working with the spec

- The spec holds reasoning; the register holds status. **A decision changes in both
  in the same edit** (upstream) - a register that lags the spec is worse than none.
- Before refusing or re-proposing a technique, search spec §14 (rejected
  approaches) and register part J (reversals). The process note at the end of
  §14.4 records what went wrong when a refusal was made from memory instead.
- Nothing in the spec has been measured. A measured figure goes into
  `docs/MEASUREMENTS.md` with its conditions, and the edit that writes it back over
  its estimate (M4) is proposed in `docs/BUILD_LOG.md`.
- The most load-bearing open item is the 5D repetition rules (register part K):
  five consumers wait on it. Resolve it first when the generator is written.

## Working with the oracle harness

```
python oracles/harness.py sync                 # check out the pinned submodules (never use --recursive)
python oracles/harness.py doctor               # which build tools and oracle builds are present
python oracles/harness.py build --bootstrap    # build the oracles; fetch missing pinned tools into oracles/.cache/
python oracles/harness.py verify               # static checks: pins, corpus, boundary, provenance, golden files
python oracles/harness.py selftest             # the harness's own unit tests
python oracles/harness.py gate --tier commit   # per-commit gate; nightly and release run all three oracles
```

- Python 3.12+ and git are all the harness itself needs. Building the oracles needs
  a C++23 compiler (g++ 13+), CMake, Ninja on Windows, GHC 9.10.3, and Node with
  npm; `--bootstrap` fetches CMake, Ninja and (on Windows) GHC into `oracles/.cache/`
  and changes nothing else on the machine.
- On Windows the interpreter may be `py` rather than `python`. A MSYS2 `ucrt64`
  g++ is found automatically.
- `oracles/.cache/` holds toolchains, builds and reports. It is ignored by git and
  can be deleted; `build --bootstrap` recreates it (on a machine that relied on
  `--bootstrap`, that means fetching the tools again).
- The oracles disagree with each other in places (`oracles/DISAGREEMENTS.md`).
  That is data, not a harness bug. Do not "fix" it.
- Editing an adapter or driver (`oracles/oracle_harness/adapters/`,
  `oracles/oracle_harness/headers.py`, `oracles/oracle_harness/fingerprint.py`,
  `oracles/drivers/`) - even a comment - makes the golden files stale by design;
  editing the ftxi driver also means rebuilding ftxi (`build --oracle ftxi`), and
  editing cwmtt's state driver (`oracles/drivers/cwmtt/Driver.hs`) rebuilding
  cwmtt (`build --oracle cwmtt`). Re-record them with
  `python oracles/harness.py golden --write --jobs 8` (about half an hour; run it in
  the background) and read the diff. Do not edit those files while it runs.
- A green gate with no engine configured means the *oracle layer* is healthy. It
  does not mean the generator has passed the gate: read the last lines of the gate
  output, which say whether the engine was compared and whether the proof channel
  may be enabled.
- CI (`.github/workflows/`, ubuntu-24.04): the commit tier runs on every push and
  pull request, the nightly tier every night and on demand (`gh workflow run
  oracle-nightly.yml -f tier=nightly`). Both first went green on 2026-10-01, and
  the golden files recorded on Windows reproduce exactly on Linux. They are not yet
  required status checks on `main`.
