# Instructions for coding agents

These rules bind you exactly as they bind human contributors (`CONTRIBUTING.md`).
The first applies to every change. The rest come from design spec §11.0 and
decisions H23–H29; `python oracles/harness.py verify` checks most of them
mechanically and both CI workflows run it, so do not spend effort routing around
them. (The workflows run on Linux - the commit tier on every push, the nightly tier
every night - but they are not yet required checks on the branch, so the check is
also only as good as your running it. Run it.)

## Every change: the changelog and the spec

A change is finished only when, in the same edit as the change itself:

1. **`CHANGELOG.md` records it.** Add an entry at the top of `## Unreleased`: the
   date, what changed and why, and a `Spec:` line that names the sections of the
   design spec and the register the change bears on - or says that no part of the
   spec describes what changed. Code, tests, configuration and documents all
   count; related edits share one entry.
2. **The spec is brought in line - upstream.** The design spec
   (`docs/design/5d-chess-transformer-design.md`) and the decision register
   (`docs/design/5d-chess-decisions.md`) are maintained upstream and copied in;
   **never edit them here.** Wherever they describe what you changed, write the
   edit they need under "Proposed spec edits" in `docs/BUILD_LOG.md` - where, what
   it says now, what it should say, why - and name it in the `Spec:` line. The
   spec and the register still change together, upstream; a measured figure goes
   into `docs/MEASUREMENTS.md` with its conditions, and its write-back (M4) is
   proposed the same way.

Bringing the spec in line never replaces deciding. If a change would contradict a
committed decision, or you cannot tell what the spec should say, stop and ask.
Never report a change as done without its changelog entry. Keep
`docs/BUILD_LOG.md` current: it is how the next session resumes.

## The oracles are not yours to reuse

`oracles/third_party/` holds independent implementations of the 5D chess rules,
pinned as submodules. They exist to be **compared with** the engine. Each thing the
engine reuses from one of them makes that oracle worthless for what the thing
covers - and the moment you are most tempted to reuse one, stuck on a subtle rules
bug, is the moment the oracle is most needed.

When you work on anything outside `oracles/`:

- **Never** import, include, link, load or reference anything under `oracles/`.
  (Citing anything in a comment or a document is fine, and so is naming the
  harness's manual, `oracles/README.md`, in code. A recipe - Makefile, justfile,
  `package.json`, `tox.ini` - may run `oracles/harness.py`, but not its `observe`
  or `golden` commands, which print oracle answers.)
- **Never** copy code from an oracle into the engine, and **never port it** to
  another language. Porting is reuse: it carries the oracle's bugs across and
  leaves nothing to compare against. This applies to move generation, check
  detection, mate and stalemate detection (the hypercuboid search included), the
  turn state machine, and the 5dpgn/5dfen parser. The boundary check catches a
  paste; it cannot catch a port, or a paste with its names changed. You are the
  check on that.
- **Never** make the engine call the harness or an oracle at run time, by any
  route - a subprocess included. The engine's answers are its own. The same goes
  for the engine's gate adapter (`[engine] command`): it lives with the engine,
  never under `oracles/`, and it asks the engine, never a recorded answer.
- **Never** add an oracle as a dependency (its git URL, its package name in a
  build manifest or an import, or a submodule of it outside
  `oracles/third_party/`), and never commit an oracle's files into the tree: the
  oracles are pinned as submodules, and nothing else belongs under
  `oracles/third_party/`.
- **Never** copy code from Stockfish or Lc0. They are GPLv3; take ideas from papers
  and the Chess Programming Wiki and write the code yourself.
- Write rules logic from the rulebook and the design spec. You may read an oracle's
  documentation and source to *understand* an algorithm, and you may cite it. You
  may not reproduce it.
- Write the engine's notation parser from the notation specification, never from
  an oracle's parser.

If you are stuck, say so. Do not reach for the oracle's implementation.

## When the engine and an oracle disagree

- The rulebook decides. Not the oracle, not the engine, and never the official game
  client (it is believed to miscall some positions as checkmate).
- If the engine is wrong, fix the engine.
- If the oracle is wrong, add an entry to `oracles/known-divergences.toml` **with
  the rulebook citation that settles it**. If you cannot cite the rule, you have
  not shown the oracle is wrong: stop and report the disagreement instead. (An
  oracle that merely cannot read a move's notation is cited against the notation
  specification instead - see the header of that file.)
- Never make the gate pass by deleting a corpus record, moving a record to a later
  tier, lowering a coverage floor or `min_checked`, adding a boundary waiver,
  editing a golden file by hand, making the engine adapter report `unsupported` or
  leave a field out, or adding a divergence entry you cannot justify. If one of
  those looks necessary, stop and ask.

## Things you must not touch without being asked

- The contents of `oracles/third_party/` (read-only; `verify` fails if a submodule
  is modified).
- The pins in `oracles/oracles.lock.toml`. A pin bump is a reviewed change with its
  own procedure (`CONTRIBUTING.md`).
- `oracles/golden/*.json` and `oracles/DISAGREEMENTS.md` by hand. They are written
  by `python oracles/harness.py golden --write`. Apart from the signature line, a
  changed line there is an answer an oracle changed (or, for a run that timed out,
  how far it got), and it must be explained.
- The adapters and drivers (`oracles/oracle_harness/adapters/`,
  `oracles/oracle_harness/headers.py`, `oracles/oracle_harness/fingerprint.py`,
  `oracles/drivers/`) in passing. They stand between an oracle and its recorded
  answers: editing one - a comment included - makes the golden files of the
  oracles it serves stale, and re-recording them takes about half an hour.
- Never run `git submodule update --init --recursive`. Use
  `python oracles/harness.py sync`.

## The proof channel

No proof may enter the solved store until the generator has passed the gate
(H23). The gate's output states whether the proof channel may be enabled; with no
engine adapter configured in `oracles/gate.toml` the answer is no, even when CI is
green. Do not enable the proof channel, or write code that assumes it is enabled,
on the strength of a commit- or nightly-tier pass.

## Commands

```
python oracles/harness.py sync                 # check out the pinned submodules
python oracles/harness.py doctor               # which build tools and oracle builds are present
python oracles/harness.py build [--bootstrap]  # build the oracles into oracles/.cache/
python oracles/harness.py verify               # pins, corpus, divergences, boundary, provenance, golden files
python oracles/harness.py selftest             # the harness's own tests
python oracles/harness.py gate --tier commit   # the per-commit gate (nightly | release for the full one)
```

Run `verify` and `gate --tier commit` before you report a change to anything the
gate covers as done, and report what they printed.
