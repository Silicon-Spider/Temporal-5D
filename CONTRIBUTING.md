# Contributing

The first section below applies to every change. The rest covers the one part of
the project that exists so far: the **rules oracles** and the gate built on them
(design spec §11.0; decisions H23–H29 in
`docs/design/5d-chess-decisions.md`). `AGENTS.md` states the same rules
for coding agents. The harness's own manual is `oracles/README.md`.

## Every change: the changelog and the spec

A change is finished only when, in the same commit as the change itself:

1. **`CHANGELOG.md` records it.** Add an entry at the top of `## Unreleased`: the
   date, what changed and why, and a `Spec:` line that names the sections of the
   design spec and the register the change bears on - or says that no part of the
   spec describes what changed. Code, tests, configuration and documents all
   count; related edits share one entry.
2. **The spec is brought in line - upstream.** The design spec
   (`docs/design/5d-chess-transformer-design.md`) and the decision register
   (`docs/design/5d-chess-decisions.md`) are maintained upstream and copied in;
   they are not edited in this repository. Wherever they describe what you
   changed, write the edit they need under "Proposed spec edits" in
   `docs/BUILD_LOG.md` (where, what it says now, what it should say, why), and
   name it in the `Spec:` line. Upstream, the reasoning in the spec and the status
   in the register change together - a register that lags the spec is worse than
   none. Measured figures go into `docs/MEASUREMENTS.md` with their conditions.

Bringing the spec in line never replaces deciding. If a change would contradict a
committed decision, or you cannot tell what the spec should say, raise it first.

## Why the oracles exist

Every exact thing the engine will claim - a proof, a solved position, a terminal
value - inherits its exactness from the rules generator. A bug there does not
misplay; it manufactures false facts that persist. So no proof enters the solved
store until the generator has passed **differential testing against independent
implementations**, and those implementations are kept in this repository, pinned,
before the generator is written - because a gate left to discretion is skipped
exactly when it is needed.

An oracle is worth what its independence is worth. That is what the rules below
protect.

## The boundary

> Everything under `oracles/` may use the engine.
> Nothing outside `oracles/` may use anything under `oracles/`.

`python oracles/harness.py boundary` checks it, and both CI workflows run it.
(The workflows run on Linux, the commit tier on every push and the nightly tier
every night, but they are not yet required checks on the branch: until they are,
the check binds only those who run it - so run it.)
Outside `oracles/` you may not:

- import, include, link or load anything from `oracles/`, or name a path into it;
- add the harness directory to the engine's build (workspace member, subdirectory,
  path dependency);
- depend on an oracle by another route - its repository URL anywhere, its
  package in a build manifest or an import, or a submodule of it anywhere but
  `oracles/third_party/`;
- commit files under `oracles/third_party/`, where only the pinned submodules
  belong, or leave a git repository inside the tree that is not a submodule;
- **copy or port an oracle's code.** Pasting it is caught mechanically, when six
  consecutive lines of logic, or one long line such as a data table, survive.
  Translating it into another language is not caught, nor is a paste with its
  names changed, and both are the same offence: a mate detector ported from
  ftxi's cannot be tested against ftxi's, because both now share its bugs.

The engine's own dependencies may be submodules wherever the engine wants them;
the lock file pins only the oracles.

Two references into `oracles/` are allowed outside comments: the harness's manual
(`oracles/README.md`) from anywhere, and the gate's entry point
(`oracles/harness.py`) from a recipe file - a `Makefile`, `justfile`,
`package.json` or `tox.ini` - run with any command except `observe` and `golden`,
which print what the oracles answer. A recipe that runs the gate is not a
dependency on it. A program that runs the harness is different: that is how an
engine would get at an oracle's answers. CI and editor configuration may name the
harness - CI runs it - but not an oracle's repository or package.

What you may do with the oracles: run them, read them to understand an algorithm
(the spec names ftxi's repository as the reference for the hypercuboid method),
and cite them. Algorithms and ideas are free to reimplement; code carries its
licence and its bugs.

The same goes for **parsers**. The engine's 5dpgn/5dfen parser is written from the
notation specification (H28). A parser shared with an oracle corrupts both sides of
a comparison identically, and the gate never sees it.

And for **GPL engines**: Stockfish and Lc0 are sources of ideas, reimplemented from
papers and the Chess Programming Wiki. Their code does not enter the engine (H29).

If the boundary check stops you and you believe it is wrong, the way through is a
reviewed waiver in `oracles/gate.toml` - not a restructuring that hides the
dependency. A waiver names the file, the rule, a piece of the offending text (the
path, name or import the finding quotes) and the reason. It excuses a finding of
that rule in that file only where the offending text contains its text - nothing
else in the file - and it fails the check itself once it no longer excuses
anything.

## When the engine and an oracle disagree

1. **The rulebook decides** - not the oracle, not the engine, and not the official
   game client, which is believed to be wrong in places and is never an oracle.
2. If the engine is wrong, fix the engine.
3. If the oracle is wrong, add an entry to `oracles/known-divergences.toml` with the
   rulebook citation that settles it, and report it upstream. An entry without a
   citation is rejected. An entry excuses one observation exactly; it stops
   matching if either side's answer changes, if the oracle's pin moves, or if the
   disagreement stops occurring, and what it says the oracle answers is checked
   against the golden files.
4. One kind of disagreement may not be about the rules at all: an oracle that
   stops because it cannot match a move's text, where the engine plays on. If the
   cause is notation the oracle does not read, the entry cites the notation
   specification instead (`notation = "..."`). If the oracle really holds the move
   illegal, it is a rules matter and needs the rulebook.
5. Any disagreement that is not recorded fails the gate. There is no "expected
   failures" list and no way to skip a record.

The oracles already disagree with **each other** in places: see
`oracles/DISAGREEMENTS.md`. None of those has been adjudicated. They are where the
first known divergences will come from.

## Changing a pin

A pin is not updated routinely. When one has to move:

1. Edit the entry in `oracles/oracles.lock.toml`: `commit`, `ref`, `committed`.
2. `python oracles/harness.py sync`, which checks the submodule out at the new
   commit; then `git add oracles/third_party/<id>` so that the gitlink in the index
   follows. Until it does, `verify` reports `gitlink-mismatch` and `build` refuses.
3. `python oracles/harness.py verify`. It prints the new commit's full `tree` id
   (`tree-mismatch`) and, if the licence file changed, its new `licence_sha256`
   (`licence-changed`). Copy the tree id into the lock file; read the new licence
   before you copy its hash, and update `PROVENANCE.md` if the terms moved.
4. Update `oracles/corpus/manifest.toml` for every record of that source whose
   blob changed (`verify` names them), and look at `corpus --unlisted` for new ones.
5. `python oracles/harness.py build`.
6. Re-verify every entry in `known-divergences.toml` bound to the old pin.
7. `python oracles/harness.py golden --write` (it records every source twice), and
   **read the diff**. Apart from each file's signature line, a changed line in
   `oracles/golden/` is an answer that changed - or, for a run that ran out of time,
   a change in how far it got. Read too what it lists at the end - verdicts,
   outcomes or positions the new recording no longer holds - and any changed line
   of "What each tier requires" in `oracles/DISAGREEMENTS.md`: those are losses of
   checking, never routine churn.
8. `python oracles/harness.py provenance --write`, and re-read the hand-written
   notes about that oracle in the lock file and in `oracles/README.md`.
9. Record the bump in `CHANGELOG.md` - the answers that moved, and any line of
   `oracles/DISAGREEMENTS.md` that appeared or went - and bring the spec in line
   where it describes that oracle (§11.0, and decisions H23-H27).

Never update submodules recursively (`git submodule update --init --recursive`):
the oracles' own submodules are not needed, and one is declared with an SSH URL.
Use `python oracles/harness.py sync`.

Editing a driver (`oracles/drivers/`), an adapter, or the `base.py`,
`fingerprint.py` and `headers.py` the adapters share (`oracles/oracle_harness/adapters/`,
`oracles/oracle_harness/fingerprint.py`, `oracles/oracle_harness/headers.py`) makes
the golden files stale in the same way as a pin bump - even when the edit is to a
comment: each golden file's signature holds a hash of the code that produced it.
Re-record them and read the diff.

## Adding game records

The corpus is a manifest, not a folder of copies: each third-party record is named
by pinned source, path, blob id and hash, and read in place.

- A record that already sits in a pinned source: `python oracles/harness.py corpus
  --unlisted` prints its manifest entry; add the licence it is used under, a tier,
  and any notes, then record what the oracles say about it (`golden --write`).
- A record of our own: put it under `oracles/corpus/records/` and list it with
  `source = "local"`. That directory is exempt from line-ending conversion, because
  a record is pinned by the hash of its bytes.
- An entry's `board` is the variant the record declares: its `Board` header, or
  its `Variant` header when it has none (`verify` checks it). A record that sets
  up its own position - with 5DFEN boards (`fen = true`) or a custom board - counts
  towards no named variant.
- Check the licence and terms of any third-party source of records before adding
  it, and give it a row in `PROVENANCE.md`.

## Before you push

Check that the change has its `CHANGELOG.md` entry and that the spec says what is
now true (first section above), then run:

```
python oracles/harness.py verify      # pins, corpus, divergences, boundary, provenance, golden files
python oracles/harness.py selftest    # the harness's own tests
python oracles/harness.py gate --tier commit
```

`python oracles/harness.py doctor` shows which build tools are present;
`build --bootstrap` fetches the pinned ones into `oracles/.cache/` without touching
the rest of the machine.

## The proof channel

The gate has three tiers: `commit` (ftxi, on every push), `nightly` (all three
oracles), and `release` (all three, and it fails unless an engine was compared and
came through at least one named variant). Only a complete run of a tier that
requires an engine and runs every oracle over the whole corpus - with every oracle
search reproducing 100% of that tier's required set of recorded answers, a set
that must hold every terminal verdict recorded (or cut off by the search's budget)
and the reviewed share of what the search would record but for its budget
(`oracles/README.md`, "Stable answers") - allows the proof channel
to be enabled, and only for the named variants that pass, with the engine compared
in full mode. Until the engine has an
adapter configured in `oracles/gate.toml`, the proof channel is disabled, whatever
colour CI is.
