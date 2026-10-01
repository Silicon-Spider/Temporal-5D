# Provenance

What in this repository came from where, under which licence, and in which role.

The decision register requires this file (H27). Its inventory is rendered from
`oracles/oracles.lock.toml` and `oracles/corpus/manifest.toml`;
`python oracles/harness.py verify` fails when the two have drifted apart. The prose
around the inventory is written by hand: keep it true in the same edit that changes
what it describes.

## The short version

- The rules oracles and the notation specification are present only as **pinned
  git submodules** under `oracles/third_party/`. A commit here carries a pointer to
  their commits, not their files.
- **The engine uses none of it.** The submodules exist so that independent
  implementations of the rules can be run against the engine (design spec §11.0,
  decisions H23–H29). `python oracles/harness.py boundary` fails any file outside
  `oracles/` that depends on them.
- Everything else was written for this project, with the exceptions listed under
  "What here did not originate here" below: one part of one test-time driver, and
  the wiring files of a developer tool. Neither is part of the engine.
- No game record is copied here. Third-party records - the `.5dpgn` files the
  pinned projects ship - are read in place from the submodules; the golden files
  hold the oracles' verdicts about them (numbers and a few fixed words), not their
  text, and not the oracles' messages, which quote it.
- Build tools fetched by `build --bootstrap` are third-party too, and are never
  committed: see "Fetched at build time" below.

## Roles

| Role | Meaning |
|---|---|
| oracle | An independent implementation of the rules, run at test time and compared with the engine. Never linked into, imported by or copied into the engine. |
| corpus-source | A repository the game-record corpus reads records from, in place. Not an oracle. |

## Inventory

<!-- BEGIN GENERATED INVENTORY - python oracles/harness.py provenance --write -->

### Pinned repositories

| Id | Upstream | Pinned commit | Licence | Role | CI tier |
|---|---|---|---|---|---|
| `ftxi` | [ftxi/5dchess_engine](https://github.com/ftxi/5dchess_engine) | `2ea9f1b82880f08faa9400d1ac20b20102604187` (v0.4.10, 2026-09-15) | BSD-2-Clause | oracle | commit |
| `cwmtt` | [penteract/cwmtt](https://github.com/penteract/cwmtt) | `9632e9969dd455c6e3d6c27c57c1936c251d7e97` (master, 2026-05-18) | GPL-3.0 (LICENSE file); cwmtt.cabal declares BSD3 - treated as GPL-3.0 until upstream resolves the conflict | oracle | nightly |
| `chessin5d` | [5d-chess/5d-chess-js](https://gitlab.com/5d-chess/5d-chess-js.git) | `ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b` (master, 2022-06-15) | AGPL-3.0-or-later | oracle | nightly |
| `notation` | [adri326/5dchess-notation](https://github.com/adri326/5dchess-notation) | `99f6163a9c95a6a9740b20fb5f3c7f6503a166a0` (master (v1.4.3-13), 2021-07-25) | X11 (MIT-style, with a no-advertising clause) | corpus-source | - |

#### `ftxi` - ftxi/5dchess_engine

- **What it is:** C++23 bitboard rules engine: move generation, check detection, hypercuboid and naive action searches
- **How it is used:** Built from an export of the pinned commit. Queried through oracles/drivers/ftxi (links its core library; part of that driver is adapted from its tools/position_tools.cpp and carries its licence notice) and cross-checked against its stock 5dtools CLI.
- **Where:** `oracles/third_party/ftxi` (submodule `oracle-ftxi`)
- **Licence file:** `LICENSE`, SHA-256 `c36ba7d9587abbf62264a9e096f35b9c9c57805e6a4ae169cfe6e03de5b79e5f`
- Pinned at the latest release tag rather than main: the eight commits after v0.4.10 rework the hypercuboid hot path and have not been released.
- Its own submodules (extern/pybind11, extern/client) are not needed and are never initialised; extern/client is declared with an SSH URL, so never update submodules recursively.
- Its test/pgn/README.txt lists twelve of its sixteen test records as taken from cwmtt, under the GPL-3.0. A thirteenth (100timelines) is cwmtt's 100_timelines again under another name, and is left out of the corpus; the other three are its own. All are read in place through the corpus manifest, never copied.
- At the pin a coordinate packs into 32 bits: 12 for the timeline, 12 for the turn, 4 each for rank and file (src/misc/vec4.h), so timelines and turns each span -2048..2047. The register's 'its 8-bit coordinate packing' (H24) describes an earlier tree.

#### `cwmtt` - penteract/cwmtt

- **What it is:** Haskell rules implementation and the original hypercuboid mate search, with a naive search beside it
- **How it is used:** Built from an export of the pinned commit with the pinned GHC. Queried through its stock CLI (`cwmtt count`).
- **Where:** `oracles/third_party/cwmtt` (submodule `oracle-cwmtt`)
- **Licence file:** `LICENSE`, SHA-256 `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`
- No z3 in the program at this commit. Upstream removed the z3-based search in 65e1bf9 (2023-03-26) because it was slower than the plain hypercuboid search; the design spec's 'Haskell and z3' describes the older tree. Its README still mentions z3 and its stack.yaml still lists the z3 package, but cwmtt.cabal depends only on base and containers, so GHC alone builds it.
- Its CLI reports whether a legal action exists, not why none does: it cannot tell checkmate from stalemate, and it has no check verdict.
- A Board header it does not know - or that its parser does not reach - is silently replayed on the Standard layout, 5DFEN boards are skipped and a Promotions header is never read; the adapter refuses such records instead of trusting the answer.
- Its parser stops without failing at the first thing it does not read (an annotation glyph, castling, a variation) and replays what it had. When that leaves the last action half read, the adapter reports the record as unsupported from there instead of passing the fragment's fate off as a verdict.

#### `chessin5d` - 5d-chess/5d-chess-js

- **What it is:** JavaScript rules library behind Chess in 5D (chessin5d.net): the most independent of the three implementations
- **How it is used:** Exported at the pinned commit; its three runtime npm dependencies are installed from its own package-lock.json. Queried through oracles/drivers/chessin5d, in a separate Node process per record.
- **Where:** `oracles/third_party/chessin5d` (submodule `oracle-chessin5d`)
- **Licence file:** `LICENSE`, SHA-256 `6f551a3f45c2c2235ec9b9e2d93c5896babab36caed027781b3e90df90876097`
- Chess in 5D's client (gitlab.com/5d-chess/chess-in-5d, AGPL-3.0) takes its rules from this library (dependency 5d-chess-js ^1.2.1). Only the rules library is pinned; the client adds no rules.
- Runtime npm dependencies, all MIT: module-alias 2.2.2, blueimp-md5 2.18.0, present 1.0.0.
- Defects at the pin, and what the driver does about each without touching rules logic: inCheckmate and inStalemate answer true when their search runs out of time (the driver calls the search itself and reads its timed-out flag); that search can exhaust memory on wide positions (it runs in a worker with its own heap limit); a Board header the library does not know, or that its header reader does not match, is silently replayed on the Standard layout, and 5DFEN boards are ignored unless Board is custom (the driver asks the library which variant it set up and refuses the record when that is not the one declared); a PGN result token at the end of the move text is read as moves (it is dropped); actions('object') throws a ReferenceError (not used).
- Its action enumeration yields ordered move sequences under its own options, so it contributes check and mate verdicts and move legality, not action counts.

#### `notation` - adri326/5dchess-notation

- **What it is:** The 5dpgn / 5dfen notation specification, with sample game records
- **How it is used:** Source of corpus records (test/*.5dpgn) and the pinned text of the notation the corpus is written in. Not an oracle.
- **Where:** `oracles/third_party/notation` (submodule `corpus-notation`)
- **Licence file:** `LICENSE`, SHA-256 `8d4c0ecb12b11ec195afed72693869a24cea48e8b160d2cc4b8f03d08f42a0ae`
- Not in the design spec's oracle table: pinned so the corpus has real game records that do not come from an oracle's own test suite.
- Its bundled JavaScript parser is not an oracle and is as off limits to the engine as everything else under third_party: the engine's parser is written from the specification (decision H28).

### Toolchains

| Tool | Required | Why |
|---|---|---|
| ghc | exactly 9.10.3; checksum-pinned archive for windows-x86_64 | The compiler of Stackage lts-24.40, the resolver cwmtt's stack.yaml names at the pin. cwmtt needs only GHC's boot packages, so no package snapshot is fetched. |
| cmake | 3.20 or newer; `--bootstrap` installs cmake==4.4.3 | Configures ftxi's own CMake project and the harness driver built against it. |
| ninja | any; `--bootstrap` installs ninja==1.13.2 | CMake generator: required on Windows, used elsewhere when present. |
| cxx | g++ 13 or newer | ftxi/5dchess_engine is C++23. Its own CI also builds with Apple Clang and MSVC 14.40; the harness looks only for g++ (set CXX to point at one). |
| node | 18 or newer | Runs the Chess in 5D library; npm installs its three pinned runtime dependencies from its own lockfile. |

### Corpus records

41 records, listed one by one with blob ids and hashes in `oracles/corpus/manifest.toml`.

| Source | Records | Licence they are used under |
|---|---|---|
| `cwmtt` (read in place at the pinned commit, not copied) | 21 | GPL-3.0, as part of penteract/cwmtt |
| `ftxi` (read in place at the pinned commit, not copied) | 5 | BSD-2-Clause, as part of ftxi/5dchess_engine; GPL-3.0: adapted from cwmtt (ftxi test/pgn/README.txt) |
| `notation` (read in place at the pinned commit, not copied) | 11 | X11, as part of adri326/5dchess-notation |
| this repository | 4 | written for this project; no third-party content |

<!-- END GENERATED INVENTORY -->

## Programs of ours that run third-party code

Two small programs in the harness load an oracle's code into their own process,
because that oracle offers no other way to ask the question:

| Program | Loads | Why |
|---|---|---|
| `oracles/drivers/ftxi/` (C++) | ftxi's core library, compiled from the pinned export, plus its `tools/search_tools.cpp` | Its stock CLI has no per-position check verdict, and its replay tools do not test a recorded action for royal safety. Where the stock `5dtools` CLI does answer - whether an action exists, the verdict when none does, the action count, the side to move - every gate run holds the driver to it. The check verdict at a position that still has a legal action has no stock counterpart to be held to. |
| `oracles/drivers/chessin5d/driver.js` (Node) | the 5d-chess-js library, from the pinned export | The library is an API. It ships a development console (`src/cli.js`), but no command that reports verdicts. |

cwmtt is driven through its own stock command-line program, unmodified.

These are test-time programs. They are built into `oracles/.cache/` (never
committed) and they are never shipped with or linked into the engine. The oracle
code they load stays in the oracle's repository under the oracle's licence. If the
harness is ever distributed in a form that bundles them with an oracle's code, that
oracle's licence governs the bundle - read it before doing so.

## What here did not originate here

| Where | From | Licence | What |
|---|---|---|---|
| `oracles/drivers/ftxi/ftxi_driver.cpp`: `apply_action()`, `mainline_action_count()`, and in `main()` the set-up that parses the record and the main-line walk | ftxi/5dchess_engine, `tools/position_tools.cpp` (`apply_action()`, `mainline_action_count()`, `read_position()`), at the pinned commit | BSD-2-Clause; the copyright notice and licence text are kept at the top of the file | About sixty lines that read a record and replay it through the oracle's own API. Adapted: each failure carries a reason, and an action is tested with the oracle's own legality test before it is submitted. |
| `.claude/helpers/graft-hooks.cjs`, `.claude/helpers/graft-statusline.cjs`, `.claude/skills/graft/SKILL.md`, and the graft entries in `.claude/settings.json`, `.mcp.json`, `.ignore` and `.gitignore` | written by `graft init` of [@nanonets/graft](https://github.com/NanoNets/context-graph-engine) 0.18.0 | MIT | Wiring for a code-navigation tool used by coding agents. Developer tooling: no part of the engine or of the harness, and nothing depends on it. Changed here: in both helpers, the first place they look for graft (`BAKED`), which `graft init` writes as an absolute path into the installing user's profile, is found from node's own install directory instead. |

The harness also *describes* how the oracles read their input, in code of its own,
so that it can tell when an oracle would misread a record: `cwmtt_tags()` in
`oracles/oracle_harness/adapters/cwmtt.py` follows cwmtt's tag grammar;
`would_misread()` in `adapters/chessin5d.py` and the result-token handling in
`oracles/drivers/chessin5d/driver.js` follow how 5d-chess-js splits move text; and
`ends_in_a_variation()` in `adapters/ftxi.py` follows which continuation ftxi
replays. All were written from reading or probing the oracle, in another language
or for another purpose; none copies it. That is permitted inside `oracles/` and
nowhere else.

## Fetched at build time

`python oracles/harness.py build --bootstrap` fetches what a machine lacks, and
`build` installs one oracle's dependencies. All of it lands in `oracles/.cache/`,
which git ignores; none of it is committed, and none of it is used by the engine.

| What | From | How it is pinned |
|---|---|---|
| CMake and Ninja | PyPI wheels, into a private virtualenv (no pip cache) | exact versions, from `[toolchain]` in `oracles/oracles.lock.toml` |
| GHC 9.10.3 (Windows only) | the binary archive on downloads.haskell.org | URL and SHA-256 in the lock file; a mismatch is refused |
| 5d-chess-js's three runtime dependencies (MIT) | npm, from the library's own `package-lock.json`, with install scripts disabled and npm's cache inside `oracles/.cache/` | the library's lockfile, at the pinned commit |

## Where the design spec and the pinned reality differ

The spec was written before the oracles were pinned. What pinning found:

- **cwmtt no longer uses z3.** The spec calls it "Haskell and z3" and describes its
  checkmate tool as z3-based. Upstream removed the z3 search in March 2023 (it was
  slower than the plain hypercuboid search); at the pinned commit the program
  depends only on GHC's boot packages (its `stack.yaml` still lists the z3 package,
  which nothing uses). cwmtt is still the heavier oracle to build, because it needs
  a Haskell compiler.
- **ftxi no longer packs coordinates into 8 bits.** The register (H24) gives "its
  8-bit coordinate packing caps timelines and time" as a reason not to build on
  ftxi. At the pinned commit a coordinate is 32 bits: 12 for the timeline, 12 for
  the turn, 4 each for rank and file. The other reasons in H24 stand; this one
  needs re-reading when the generator is written.
- **cwmtt's licence is declared two ways.** Its `LICENSE` file is the GPL-3.0; its
  `cwmtt.cabal` says `BSD3`. It is treated as GPL-3.0 here.
- **"Chess in 5D" means its rules library.** The site's client takes its rules from
  the separate `5d-chess-js` library (AGPL-3.0-or-later); that library is what is
  pinned.
- **ftxi is driven through a C++ driver and its stock CLI, not its Python
  bindings.** The bindings expose no query for the check verdict the gate compares.
- **A fourth repository is pinned:** the notation specification, as a corpus
  source. It is not in the spec's table and it is not an oracle.
- **The oracles disagree with each other** in places. Those are listed, not yet
  adjudicated, in `oracles/DISAGREEMENTS.md`.
- **No oracle answers everything.** The spec asks for a cross-check of move
  counts, check verdicts and mate/stalemate verdicts. cwmtt's CLI gives counts
  only; 5d-chess-js gives verdicts and no counts; only ftxi gives all three, and
  ftxi knows four board variants. What that leaves checked, variant by variant, is
  tabulated in `oracles/DISAGREEMENTS.md`.

## Not here, on purpose

- **The official game client** is not an oracle and is not used: it is believed to
  call some positions checkmate that are not (H25).
- **Stockfish and Lc0** (both GPLv3) are sources of ideas only. No code from either
  is in this repository in any form (H29).
- **No oracle's parser** is used by the engine, and none may be: a parser shared by
  the engine and an oracle corrupts both sides of a comparison identically (H28).

## Keeping this file true

- A pin changes in `oracles/oracles.lock.toml` first (the whole procedure is in
  `CONTRIBUTING.md`); then `python oracles/harness.py provenance --write` refreshes
  the inventory.
- If a pin bump changes a licence file, `verify` fails until the recorded hash is
  updated - which is the prompt to read the new licence.
- Anything new from outside the project gets a row here before it is merged, with
  its licence and its role. Code that the gate checks does not come from outside at
  all.
