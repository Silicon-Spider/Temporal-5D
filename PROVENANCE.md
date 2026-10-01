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
- The project's own code is licensed under the **GPL-3.0** (`LICENSE`): see "The
  project's licence" below.

## The project's licence

The project is licensed under the **GNU General Public License, version 3** - the
`LICENSE` file, the GPL-3.0 text, from the repository's initial commit (decision
M11).

- Its obligations attach when source or binaries are **distributed**. Private use,
  and running the engine as a service, distribute nothing and are unaffected.
- Code from elsewhere keeps its own licence, listed below. The BSD-2-Clause code
  adapted into the ftxi driver (`oracles/drivers/ftxi/ftxi_driver.cpp`, under
  "What here did not originate here") is GPL-compatible; its copyright notice and
  licence text are kept at the top of that file, as its licence requires. The
  oracles are pinned submodules under their own licences, used only by the test
  harness, and never part of the engine.
- Stockfish and Lc0 are GPL-3.0 too, so copying their code would be
  licence-compatible - and is still refused (H29): it would end sole copyright,
  which is what keeps relicensing or dual-licensing possible. Their ideas enter
  reimplemented, from papers and the Chess Programming Wiki; their code does not.
- **Trained weights** are an open question (register part K, M11): weights are
  data, and whether the licence reaches them is to be decided before any weights
  are distributed. None exist yet.

## Roles

| Role | Meaning |
|---|---|
| oracle | An independent implementation of the rules, run at test time and compared with the engine. Never linked into, imported by or copied into the engine. |
| parse-oracle | An independent reader of the notation, run at test time and compared with the engine and the oracles for the position it reads at every ply and how many actions it reads - never for a rules verdict (decision H26). Never linked into, imported by or copied into the engine. |
| corpus-source | A repository the game-record corpus reads records from, in place, and nothing more: never built, never run. (A pin in either role above may be a source of records too.) |

## Inventory

<!-- BEGIN GENERATED INVENTORY - python oracles/harness.py provenance --write -->

### Pinned repositories

| Id | Upstream | Pinned commit | Licence | Role | CI tier |
|---|---|---|---|---|---|
| `ftxi` | [ftxi/5dchess_engine](https://github.com/ftxi/5dchess_engine) | `2ea9f1b82880f08faa9400d1ac20b20102604187` (v0.4.10, 2026-09-15) | BSD-2-Clause | oracle | commit |
| `cwmtt` | [penteract/cwmtt](https://github.com/penteract/cwmtt) | `9632e9969dd455c6e3d6c27c57c1936c251d7e97` (master, 2026-05-18) | GPL-3.0 (LICENSE file); cwmtt.cabal declares BSD3 - treated as GPL-3.0 until upstream resolves the conflict | oracle | nightly |
| `chessin5d` | [5d-chess/5d-chess-js](https://gitlab.com/5d-chess/5d-chess-js.git) | `ee403c80e8c132fc4d168f5b99ddd9b8fe5cc17b` (master, 2022-06-15) | AGPL-3.0-or-later | oracle | nightly |
| `notation` | [adri326/5dchess-notation](https://github.com/adri326/5dchess-notation) | `99f6163a9c95a6a9740b20fb5f3c7f6503a166a0` (master (v1.4.3-13), 2021-07-25) | X11 (MIT-style, with a no-advertising clause) | parse-oracle | commit |

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
- **How it is used:** Built from an export of the pinned commit with the pinned GHC. Its counts, and how far it replays a record, come from its stock CLI (`cwmtt count`), unmodified. Beside it runs oracles/drivers/cwmtt, a state driver of the harness's own compiled against the same modules with the same flags (it calls them and copies nothing), which replays the record and reports, at every position, the boards cwmtt holds and the side to move - the CLI has no command for either; the adapter pairs the two only where they read and replayed the record alike.
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
- Defects at the pin, and what the driver does about each without touching rules logic: inCheckmate and inStalemate answer true when their search runs out of time (the driver calls the search itself and reads its timed-out flag); that search can exhaust memory on wide positions (it runs in a worker with its own heap limit); a Board header the library does not know, or that its header reader does not match, is silently replayed on the Standard layout, and 5DFEN boards are ignored unless Board is custom (the driver asks the library which variant it set up and refuses the record when that is not the one declared); a PGN result token at the end of the move text is read as moves (it is dropped); chess.actions() throws in its 'object' and 'json' formats (a ReferenceError) and in '5dpgn' (a TypeError), and in every format passes its promotion pieces where boardFuncs.moves() takes its 'spatial moves only' switch, so it would list no move through time or across timelines (not used: the driver's observe --actions diagnostic calls actionFuncs.actions() itself, with that switch off).
- Its action enumeration yields ordered move sequences under its own options, so it contributes check and mate verdicts and move legality, not action counts.

#### `notation` - adri326/5dchess-notation

- **What it is:** The 5dpgn / 5dfen notation specification, with sample game records and a parser written beside it
- **How it is used:** The pinned text of the notation the corpus is written in, and a source of corpus records (test/*.5dpgn). Its bundled parser (parsers/pgn.js and parsers/game.js, which need no package) is exported at the pinned commit and run as a parse-only oracle through oracles/drivers/notation, in a separate Node process per record: at every position, the boards it holds and the side it reads as moving next, and how many actions it read - never a rules verdict.
- **Where:** `oracles/third_party/notation` (submodule `corpus-notation`)
- **Licence file:** `LICENSE`, SHA-256 `8d4c0ecb12b11ec195afed72693869a24cea48e8b160d2cc4b8f03d08f42a0ae`
- Not in the design spec's table of rules oracles: pinned so the corpus has real game records that do not come from an oracle's own test suite, and so that the engine's parser is written from the specification (decision H28). Its bundled parser is a parse-only oracle (H26), and as off limits to the engine as everything else under third_party.
- The parser applies each move as written. It has no check, legality or end-of-game logic, and finds the moving piece with a movement test of its own only where the move text leaves the origin square out. So its acceptance of an action is no verdict - it accepts an action that leaves a king in check - and a move it cannot apply is a soft rejection.
- Defects at the pin, and what the harness does about each without touching the parser: it reports an error by printing it to stdout and calling process.exit(1) (the driver captures what it prints and turns the exit into an exception); a Board name it does not know gives one silently empty board, and a named layout is set up on boards of the Size header's size, or 8x8, never its own (both refused, by the adapter and again by the driver); it reads 5DFEN board strings only under Board "custom" (refused under any other); its tokeniser loops forever on a comment that is never closed (refused before it runs); it stores a 5DFEN "-0" board as a signed zero, which it writes back as "0" (left alone: a set-up that mixes it with "+0" is refused under a reason of its own, signed_zero, so that the recorded outcome names the defect; any other shows in the fingerprints); it drops the unmoved marker "*" (its state events say markers false); it splits a long-algebraic pawn move such as e2e4 into two moves (it then finds no pawn for the first, a soft rejection).
- X11 licence: the export the driver loads is the whole pinned tree, LICENSE included, in the harness cache, never committed or distributed; the project never uses the author's name to promote itself.

### Toolchains

| Tool | Required | Why |
|---|---|---|
| ghc | exactly 9.10.3; checksum-pinned archive for windows-x86_64 | The compiler of Stackage lts-24.40, the resolver cwmtt's stack.yaml names at the pin. cwmtt needs only GHC's boot packages, so no package snapshot is fetched. |
| cmake | 3.20 or newer; `--bootstrap` installs cmake==4.4.3 | Configures ftxi's own CMake project and the harness driver built against it. |
| ninja | any; `--bootstrap` installs ninja==1.13.2 | CMake generator: required on Windows, used elsewhere when present. |
| cxx | g++ 13 or newer | ftxi/5dchess_engine is C++23. Its own CI also builds with Apple Clang and MSVC 14.40; the harness looks only for g++ (set CXX to point at one). |
| node | 18 or newer | Runs the Chess in 5D library, whose three pinned runtime dependencies npm installs from its own lockfile, and the notation repository's parser, which needs none. |

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

Small programs in the harness load an oracle's code into their own process,
because that oracle offers no other way to ask the question:

| Program | Loads | Why |
|---|---|---|
| `oracles/drivers/ftxi/` (C++) | ftxi's core library, compiled from the pinned export, plus its `tools/search_tools.cpp` | Its stock CLI has no per-position check verdict, and its replay tools do not test a recorded action for royal safety. Where the stock `5dtools` CLI does answer - whether an action exists, the verdict when none does, the action count, the side to move, the number and size of the boards - every gate run holds the driver to it. The check verdict at a position that still has a legal action has no stock counterpart to be held to. |
| `oracles/drivers/cwmtt/Driver.hs` (Haskell) | cwmtt's own modules - its parser, its set-up, its replay and its two searches - compiled from the pinned export with the pinned GHC and the CLI's flags | Its stock CLI prints counts, and draws the boards only as a picture of the last position (`print`): never the boards it holds at each position, nor the side to move. The driver reads and replays the record with the cwmtt code the CLI's `count` runs - its parser, its set-up, and the replay step both replays scan - and prints at every position the boards cwmtt holds and the side to move (and, asked by `observe`, its diagnostics); it answers nothing the CLI answers. The adapter holds the two to each other on every run - same layout, same text left unread, same replay failure, same positions - and pairs a count with a state only then. |
| `oracles/drivers/chessin5d/driver.js` (Node) | the 5d-chess-js library, from the pinned export | The library is an API. It ships a development console (`src/cli.js`), but no command that reports verdicts. |
| `oracles/drivers/notation/driver.js` (Node) | the parser bundled with the notation specification (`parsers/pgn.js` and `parsers/game.js`), from the pinned export | It is a parse-only oracle (H26). The parser's one entry point reads a whole record and ends the process on an error; its command-line program converts records and reports no position. The driver watches the parser through the module objects it reaches its own helpers through, and reports the boards it holds at every position; it changes nothing in it. |

cwmtt's counts, and how far it replays a record, still come from its own stock
command-line program, unmodified; its state driver adds what cwmtt set up and the
side to move, and nothing else.

These are test-time programs. They are built into `oracles/.cache/` (never
committed) and they are never shipped with or linked into the engine. The oracle
code they load stays in the oracle's repository under the oracle's licence. If the
harness is ever distributed in a form that bundles them with an oracle's code, that
oracle's licence governs the bundle - read it before doing so. The notation
repository's parser is under the X11 licence: its notice goes with any copy, which
the export does (it is the whole pinned tree, `LICENSE` included), and its
no-advertising clause means the project never uses the author's name to promote
itself or anything it makes.

## What here did not originate here

The engine (`engine/`) is written for this project from `docs/rules/README.md`,
the design spec and the notation specification's text. Two pieces follow published
algorithms rather than code: `engine/src/md5.cpp` implements MD5 from RFC 1321
(its algorithm and test vectors), and `engine/src/zobrist.cpp` uses published
numeric constants (SplitMix64, the MurmurHash3 finaliser, xxHash64's primes) to
generate its keys. No code was copied into either.

| Where | From | Licence | What |
|---|---|---|---|
| `oracles/drivers/ftxi/ftxi_driver.cpp`: `apply_action()`, `mainline_action_count()`, and in `main()` the set-up that parses the record and the main-line walk | ftxi/5dchess_engine, `tools/position_tools.cpp` (`apply_action()`, `mainline_action_count()`, `read_position()`), at the pinned commit | BSD-2-Clause; the copyright notice and licence text are kept at the top of the file | About sixty lines that read a record and replay it through the oracle's own API. Adapted: each failure carries a reason, and an action is tested with the oracle's own legality test before it is submitted. |
| `docs/rules/README.md`: the quotations in each rule and in its Sources list | public web pages (Wikipedia, the 5D chess fandom wiki and wiki.gg, Steam guides and discussions, TV Tropes, reviews and forums), the game's in-game Guide texts (read as text from the locally installed client, Steam build 7653463) and the 5dpgn specification | each source's own terms; quoted briefly, attributed, for citation | Short verbatim quotations that support each rule statement; the statements, ratings and open questions are written for this project. No rules engine, oracle output or client logic was used. |
| `.claude/helpers/graft-hooks.cjs`, `.claude/helpers/graft-statusline.cjs`, `.claude/skills/graft/SKILL.md`, and the graft entries in `.claude/settings.json`, `.mcp.json`, `.ignore` and `.gitignore` | written by `graft init` of [@nanonets/graft](https://github.com/NanoNets/context-graph-engine) 0.18.0 | MIT | Wiring for a code-navigation tool used by coding agents. Developer tooling: no part of the engine or of the harness, and nothing depends on it. Changed here: in both helpers, the first place they look for graft (`BAKED`), which `graft init` writes as an absolute path into the installing user's profile, is found from node's own install directory instead. |

The harness also *describes* how the oracles read their input, in code of its own,
so that it can tell when an oracle would misread a record: `cwmtt_tags()` in
`oracles/oracle_harness/adapters/cwmtt.py` follows cwmtt's tag grammar;
`would_misread()` in `adapters/chessin5d.py` and the result-token handling in
`oracles/drivers/chessin5d/driver.js` follow how 5d-chess-js splits move text;
`ends_in_a_variation()` in `adapters/ftxi.py` follows which continuation ftxi
replays; and `would_misread()` in `adapters/notation.py` and the set-up checks in
`oracles/drivers/notation/driver.js` follow how the notation repository's parser
reads headers, board names and board sizes. All were written from reading or
probing the oracle, in another language or for another purpose; none copies it.
That is permitted inside `oracles/` and nowhere else.

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

The spec was written before the oracles were pinned. What pinning found, and where
the spec stands on each now (this file is authoritative on oracle internals, §11.0):

- **cwmtt no longer uses z3.** Upstream removed the z3 search in March 2023 (it was
  slower than the plain hypercuboid search); at the pinned commit the program
  depends only on GHC's boot packages (its `stack.yaml` still lists the z3 package,
  which nothing uses). cwmtt is still the heavier oracle to build, because it needs
  a Haskell compiler. *Corrected in the spec: register J20 removes the claim.*
- **ftxi no longer packs coordinates into 8 bits.** At the pinned commit a
  coordinate is 32 bits: 12 for the timeline, 12 for the turn, 4 each for rank and
  file. *Corrected: J20 removes it from H24's reasons, and H24 stands on the rest.*
- **A fourth repository is pinned:** the notation specification, as a corpus
  source and the text the engine's parser is written from. It is not a rules
  oracle; its bundled parser code counts as a parse-only oracle, and the harness
  runs it as one (source `notation`, role `parse-oracle`, every tier). *Now in the
  spec: H26 and the §11.0 table.*
- **ftxi is driven through a C++ driver and its stock CLI, not its Python
  bindings.** The bindings expose no query for the check verdict the gate compares.
  *Still differs: §11.0 ("Tiered CI") says "Python bindings"; an edit is proposed
  in `docs/BUILD_LOG.md`.*
- **cwmtt's licence is declared two ways.** Its `LICENSE` file is the GPL-3.0; its
  `cwmtt.cabal` says `BSD3`. It is treated as GPL-3.0 here.
- **"Chess in 5D" means its rules library.** The site's client takes its rules from
  the separate `5d-chess-js` library (AGPL-3.0-or-later); that library is what is
  pinned.
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
