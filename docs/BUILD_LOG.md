# Build log - the untrained build (spec §15)

The resume document. A fresh session should be able to continue from this file
alone: what is done, what is parked and why, and what is next. Status is keyed by
register ID (`docs/design/5d-chess-decisions.md`). Every measured figure, with its
conditions, is in `docs/MEASUREMENTS.md`. Every change is in `CHANGELOG.md`.

Status words: **built** (code exists) · **tested** (its tests and gate checks
pass) · **measured** (figure in MEASUREMENTS.md) · **parked** (waits on a named
dependency; resume when it lands) · **blocked** (needs a decision from the
user) · **not started** · **not built (rule)** (refused, held open, or excluded by
M5 - building it would break a rule).

---

## Where things stand

*Updated 2026-10-01.*

- **Current step:** 1 - commit the pending work, push, get the tiered CI green on
  Linux (H35).
- **Next:** step 2 - finish the gate (H30 fingerprints, H32 stable answers, H33
  integrity, H34 telltales), then CHECKPOINT 1 (check-disagreement clusters).
- **Waiting on the user:** nothing yet.

## The brief - standing rules for every session

The run is one continuous build of the whole engine, untrained, per §15, until
§15.5's "done means" holds. Condensed from the user's brief (2026-10-01):

- **No training runs.** Training code may be written and smoke-tested with a few
  optimiser steps on synthetic data, nothing more.
- **Order:** 0 pre-flight → 1 commit, push, Linux CI (H35) → 2 finish the gate
  (§11.0 "Running the gate") → 3 cluster the check disagreements (CHECKPOINT 1),
  and alongside it the work that does not depend on the check ruling → 4 after the
  ruling: check detection, turn legality, naive mate, hypercuboid, each against
  the gate → 5 Phase 1 in M10's order: movegen throughput, the |E| curve, scanner
  horizon with stand-pat/zugzwang (6b), each verdict applied from §15.6 (M7-M9)
  and recorded in MEASUREMENTS.md → 6 Phases 2-3: network stack and search with
  random weights, the scanner ladder with Lazy SMP, the §15.2 equivalence suite,
  the §15.3 cost measurements, the §15.4 invariants.
- **CHECKPOINT 1 does not halt the run.** Report each cluster with a
  representative position, every oracle's verdict and the rule question; mark the
  clusters about which boards count as sources of attack (§11.0 item 1); trace the
  two checkmate disagreements and the 52-vs-54 count. Park what depends on the
  ruling (check detection, legality, the mate detectors, the scanner and 6b,
  search on real positions); keep building the network stack with random weights
  and its equivalence tests, the cached path, typed edges, the output heads,
  movegen throughput with pseudo-legal moves (provisional) and the |E| curve from
  pseudo-legal destinations (an upper bound).
- **Dependency gates:** window code only after the |E| verdict (M7) - a
  pseudo-legal upper bound that keeps the window decides early, one that kills it
  waits for the legal curve; the NNUE and the scanner's pruning only after 6b
  passes (M8); if movegen fails M9, fix movegen before building search on it.
- **STOP and wait only when:** 6b's root-scan test fails (M8); a §15.6 result lands
  within ±10% of its threshold; the pre-flight fails or a problem blocks all
  remaining work; or everything else is done (then report for sign-off, proof
  channel disabled).
- **Log and continue:** a spec that is wrong, ambiguous, contradictory or silent on
  a committed-class point → record section and evidence below, park the item,
  continue; never improvise a design decision. A measured cost more than 2× off its
  §15.3 estimate → record it with its conditions and continue.
- **Rules:** §14.4's refusals and M5's not-built list bind. Build committed items;
  provisional and prototype items behind flags; deferred items only where §15.2 or
  §15.3 needs them; held-open items not at all. One local commit per component,
  each with its tests and gate checks. GitHub MCP read-only. Never force-push or
  rewrite pushed history. No absolute user paths in tracked files. **Never edit the
  spec or the register** - propose edits below; they are applied upstream.
- Licence: GPL-3.0 (keep LICENSE).

## Environment

- Development machine: Windows 11, AMD Ryzen 9 9900X (12 cores / 24 threads,
  AVX-512 incl. BF16), 32 GB RAM, AMD Radeon RX 9070 XT. **No CUDA**, so the
  network runs on CPU PyTorch (2.13.0+cpu) here; every GPU figure in §15.3 is
  measured on CPU and labelled so. No Rust, no MSVC: MSYS2 ucrt64 g++ 16.2, Python
  3.14 (`py`), Node 24.
- CI: GitHub Actions, ubuntu-24.04 (`.github/workflows/`). The repository is
  private, so Actions minutes are billed: the nightly tier runs every night.
- Oracle toolchains live in `oracles/.cache/` (`build --bootstrap`); never on the
  system.

## Steps

### 0. Pre-flight - done 2026-10-01

All five checks passed: register row M11 and spec §15.6 present; local `main`
already contained `origin/main` (f62ceed, the remote's LICENSE + README commit),
so no rebase was needed; no tracked file held an absolute user path (the graft
helpers' baked path had already been made relative by an amend to a56beb4, never
pushed); `oracles/.cache` is ignored (`.gitignore:6`); `git ls-remote origin`
and `git push --dry-run origin HEAD` succeeded. The remote is private.

### 1. Commit, push, Linux CI (H35) - in progress

Commits so far: the design documents (`docs/design/`), the oracle harness (H27),
agent permissions, CI caching with `build --plan`, and this log with the licence
record (M11).

### 2. Finish the gate - not started

### 3. CHECKPOINT 1 and the ruling-independent engine work - not started

### 4. After the ruling - parked (needs the check ruling)

### 5. Phase 1 measurements - not started

### 6. Phases 2-3 - not started

## Register status

### H - rules engine, gate, verification layer

| ID | Status | Notes |
|---|---|---|
| H1-H5 | parked | The scanner needs check detection and legal evasions (proofs need all-evasions generation): waits on the check ruling |
| H6 | parked | Material eval v1 with the scanner; NNUE v2 waits on 6b (M8) |
| H7 | not built (rule) | NNUE training: no training runs; NNUE waits on 6b |
| H8, H9, H20, H21 | not built (rule) | Held open / refused |
| H10-H14 | parked | Async ladder rungs: with the scanner |
| H15 | parked | Synchronous debug mode: with the scanner and search |
| H16, H25, H29 | not built (rule) | Refused |
| H17 | parked | Lazy SMP: with the scanner |
| H18 | not started | 5D Zobrist - step 3 (ruling-independent) |
| H19 | parked | Proof export: with the scanner |
| H22 | not started | Rules engine: representation, parser, per-board movegen in step 3; check, legality, mate in step 4 |
| H23 | built | The gate (oracle side); engine side waits for the engine |
| H24, H26, H28 | not started | In-house generator and parser (step 3) |
| H27 | built, tested | The oracle harness, committed 2026-10-01 |
| H30 | not started | Step 2 |
| H31 | not started | Step 3 (CHECKPOINT 1) |
| H32-H34 | not started | Step 2 |
| H35 | in progress | Step 1 |

### A-G, I - representation, network, search, training

| ID | Status | Notes |
|---|---|---|
| A1-A4 | not started | Tokens, features, flags, immutability (step 3/6) |
| A5 | not started | Piece-centric tokenisation, behind a flag |
| B1, B2, B4, B8, B9, B14 | not started | Committed network items |
| B5, B6, B10, B11, B12 | not started | Provisional / prototype: behind flags |
| B3 | not built (rule) | Contingency only; no trigger |
| B7 | not built (rule) | Deferred; no §15.2/§15.3 need (d_head is a config value anyway) |
| B13 | not built (rule) | Refused |
| C1, C4, C17, C19, C20 | not started | Dense first; the cached-path rows are needed by §15.2 |
| C2, C10 | not started | Deferred, but §15.2/§15.3 need the cached path |
| C3, C3a (rung 1) | not started | Behind flags; §15.2/§15.3 need them |
| C3a (rungs 2-3), C18, C22 | not built (rule) | Held open (M5 for rungs 2-3) |
| C5, C7 | parked | Window code only after the \|E\| verdict (M7) |
| C11 | not started | MCGS behind a flag; search on real positions waits on legality |
| C12, C13 | not started | Persistence (search) |
| C15 | not started | FP8 KV storage, varlen |
| C16 | not built (rule) | Deferred; last, only if node-starved |
| C21 | not built (rule) | Refused |
| D1, D3, D4 | not started | Output heads (step 6) |
| D2 | not started | Undecided: both arms awaited by trained ablations 6g/6k |
| E1, E5, E7, E9 | not started | Search |
| E2, E3, E8 | not started | Behind flags |
| E6 | not built (rule) | Refused |
| F1, F2 | not started | Model config; F2 behind a flag |
| F3 | not built (rule) | Refused |
| F4 | parked | NNUE only after 6b passes (M8) |
| G1 | not started | Precision rules |
| G2 | not started | FP8 attention - no FP8 kernels on this machine |
| G3, G4, G5, G6 | not built (rule) | Deferred / held open / refused |
| I1-I10 | not started | Training code may be written and smoke-tested only; I7 and I8 not built (M5) |

### M - the plan itself

| ID | Status | Notes |
|---|---|---|
| M7 | not started | Window verdict (step 5) |
| M8 | parked | Scanner verdict (needs the scanner) |
| M9 | not started | Movegen verdict, pseudo-legal first (provisional) |
| M11 | built | Licence recorded in PROVENANCE.md |

## Parked, and why

Nothing parked yet beyond the rows above that wait on the check ruling or on 6b.

## Spec issues (wrong, ambiguous, contradictory, silent)

None logged yet.

## Proposed spec edits

Edits the spec or the register needs, for applying upstream. Each: where, what it
says now, what it should say, why.

1. **§11.0 "Tiered CI", first sub-bullet**: says "ftxi's engine (fast, Python
   bindings) on every commit". The harness drives ftxi through a C++ driver and
   its stock `5dtools` CLI, because the Python bindings expose no per-position
   check verdict (PROVENANCE.md). Proposed: "ftxi's engine (fast; a C++ driver
   over its core library, held to its stock CLI) on every commit".

## Session log

- **2026-10-01** - Pre-flight passed. Design documents moved into `docs/design/`
  (they had been kept untracked as `Specification documents/`). Step 1 started.
