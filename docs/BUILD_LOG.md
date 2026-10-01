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

*Updated 2026-10-01, evening.*

- **Done:** pre-flight; step 1 (pushed; the commit and nightly tiers green on
  Linux - H35 holds, so engine commits may land); step 2A (H30 state
  fingerprints in every oracle, the notation parser as a parse-only oracle,
  golden files schema 2); the cited rules reference `docs/rules/README.md`;
  engine stage E1 and network stage N1 (both built and reviewed, not yet
  committed - see below).
- **Running:** CHECKPOINT 1 analysis (check clusters and the mate/count traces);
  engine stage E2 (rules mechanics, pseudo-legal movegen, the 5dpgn parser, the
  parse-mode gate adapter); network stage N2 (move-graph mask, routing, typed
  edges, smolgen, a memory-efficient attention path).
- **Next:** commit E1+E2 and N1+N2 when their workflows finish (each stage is
  committed as soon as it completes from now on); step 2B (H32 stable answers, H33
  integrity, H34 telltales, the engine parse mode in the gate) with one golden
  re-record using the CI runner's speed for the H32 margin; report CHECKPOINT 1;
  then movegen throughput (M9) and the |E| upper bound (M7) on pseudo-legal moves.
- **Waiting on the user:** nothing blocking yet; CHECKPOINT 1's report will ask for
  the check ruling (and which text is the rulebook), and the repetition rules.

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

### 1. Commit, push, Linux CI (H35) - done 2026-10-01

- Commits: the design documents (`docs/design/`), the oracle harness (H27), agent
  permissions, CI caching with `build --plan`, this log with the licence record
  (M11), the rules reference, and step 2A (H30). Pushed 2026-10-01
  (`f62ceed..a04678c`) once the gh token had the `workflow` scope (the user
  signed in).
- Commit tier, run 36827435595: green on the first try - verify, the ftxi and
  notation builds, selftest (462 tests; 16 skipped: live tests of oracles this
  tier does not build), gate: ftxi 7760/7760 and notation 2765/2765 recorded
  values reproduced. The golden files recorded on Windows reproduce exactly on
  Linux.
- Nightly tier, run 36827459456 (started by hand): green - GHC 9.10.3 installed
  (cache miss), all four oracles built, selftest 462 tests with no skip, gate
  nightly ok: ftxi, cwmtt, chessin5d and notation 100%; **ftxi:naive 6728/7016
  (95.9%) and cwmtt:naive 4876/4957 (98.4%)** against 99.1% / 99.3% on the
  desktop - the runner is slower, more naive searches hit their 120 s budget, and
  the gate passes only by the coverage floors that H32 removes. The nightly gate
  step took about 30 minutes on the runner.
- Not yet: the workflows as required status checks on `main` (branch protection
  is a repository setting - for the user).

### 2. Finish the gate - in progress

Design (from a six-agent investigation of the harness and every oracle, 2026-10-01):

- **H30 fingerprints.** Every source prints, per position and before any search,
  `{"event":"state","index":K,"size":"WxH","markers":bool,"boards":["L:T:c:placement",...]}`
  listing every board. The harness alone canonicalises and hashes: `fingerprint` =
  the notation spec's full-state hash (MD5 of the sorted `[placement:L:T:c]`
  strings; -0 before +0, then turn, then w before b), `placement` = the same
  without `*`, plus board and piece counts. Golden rows store hashes and counts
  only. Parse divergences are their own kinds: `placement` (suppresses every
  rules comparison from that position on) and `unmoved` (marker-only; suppresses
  nothing; counted per run of positions; an oracle's answers at positions where its
  markers differ from the engine's do not count as confirmation). ftxi: driver walks
  its boards (`get_fen<true>`); chessin5d: the library's own `fen(null, true)`
  export; cwmtt: a new Haskell driver (beside the stock CLI, which keeps answering
  the counts; merged by index); the notation repository's bundled parser becomes
  the parse-only source `notation` (H26, as the brief asks). Drivers gain
  `--explain` (ftxi: phantom-advanced timelines and checking moves) and
  `--actions K` (legal actions listed) for CHECKPOINT 1, shown by `observe` only.
- **H32 stable answers.** The nightly run loses 42 recorded values, all naive
  searches cut by the 120 s wall clock; no value has ever been contradicted, none
  is nondeterministic. Design: schema-2 golden entries make each non-answer
  explicit (`horizon` = run cut at its recording budget; `unknown` = null inside
  answered positions with the limit that caused it; `unstable` = differed between
  two recordings, nulled and listed); the recording budget joins each signature;
  per-position costs go in separate hashed files; each tier's required set is
  derived statically (cost x margin <= tier budget; margin 4 to start, then set
  from the first CI run's calibration); the gate requires 100% of the required
  set and fails by name on any missing value - every coverage floor and the 90%
  proof-channel ratio go. `golden --write --repeat 2` detects instability.
- **H33 integrity.** A hash manifest of the golden and cost files written by
  `golden --write`, verified by `verify` and before every comparison; one verified
  snapshot per gate run; `known-divergences.toml` no longer re-read after the
  engine runs (today it is - gate.py:389); a hash snapshot of the harness's files
  before and after the engine runs; CODEOWNERS on `oracles/`.
- **H34 telltales.** In the engine differential: an engine answer equal to a
  known divergence's recorded oracle error, at that position, is a telltale
  finding (fails the gate), ahead of "divergence-stale".
- **2A result (2026-10-01):** every oracle now reports its boards at every ply;
  across the whole corpus no two sources ever hold different boards at a shared
  position (ftxi-cwmtt 759 positions, ftxi-chessin5d 795, ftxi-notation 586,
  cwmtt-chessin5d 763, cwmtt-notation 555, chessin5d-notation 657). So the check,
  count and mate disagreements in `oracles/DISAGREEMENTS.md` are rules
  disagreements, not parse divergences. The only parse-layer difference is
  markers-only on Very Small - Open (ftxi's built-in layout holds the kings and
  rooks as moved; cwmtt marks them unmoved) - which leaves that variant ungated at
  the current pins until it is settled against the notation spec.
- Order: 2A = H30 (done) → 2B = H32 + H33 + H34 → one `golden --write
  --repeat 2` → DISAGREEMENTS.md re-rendered → parse divergences settled against
  the notation spec.

### 3. CHECKPOINT 1 and the ruling-independent engine work - in progress

**CHECKPOINT 1 - reported 2026-10-01** (full report:
`docs/checkpoints/checkpoint-1-check-clusters.md`; evidence in
`oracles/.cache/reports/checkpoint1/`, regenerable). After H30, all 81 positions
involved hold the same state in every source. The 78 check disagreements (ftxi
"check", Chess in 5D "no check") come from one difference: before looking for a
royal capture, ftxi passes every board the side to move may play (rules reference
Q10, reading 5), Chess in 5D only the mandatory ones (reading 2). All 217 of
ftxi's captures start on an optional board of the side to move that Chess in 5D
does not pass. Seven clusters, four rule questions, **all about which boards
count as sources of attack (spec §11.0 item 1)**:

| Cluster | Attacker's board | Capture | Positions | Question | Representative |
|---|---|---|---|---|---|
| CHK-A1 | active timeline, ahead of the present | same board | 40 | RQ1 | `cwmtt/silly@28` |
| CHK-A2 | active, ahead | onto a royal on a history board; branch active | 5 | RQ1 | `notation/Bg2@19` |
| CHK-A3 | active, ahead | onto a royal on a history board; branch **inactive** | 17 | RQ2 | `cwmtt/niceAndrey@57` |
| CHK-A4 | active, ahead | non-branching jump onto a royal's board | 2 | RQ1 | `cwmtt/silly@16` |
| CHK-A5 | active, ahead | onto a royal on another optional board (two passes) | 5 | RQ1 | `cwmtt/niceAndrey@41` |
| CHK-I1 | inactive timeline, behind the present | same board | 3 | RQ3 | `cwmtt/manyChecks@20` |
| CHK-I2 | inactive, behind | onto a royal on a history board; branch inactive | 6 | RQ4 | `cwmtt/standard@13` |

- RQ1: do pieces on an active optional board ahead of the present attack (is that
  timeline "eligible" for the phantom advance)? RQ2: as RQ1, and does a capture
  that would branch into an inactive timeline count? RQ3: do pieces on an inactive
  timeline's board attack? RQ4: RQ3 plus the branch question. Q1 (activation when
  the creator is behind on count) moves 18 CHK-A positions under its
  "consecutive" reading. The rules reference today (C6 medium, C7 medium) gives
  Chess in 5D's answer at all 78; readings 4 ([41]) or 5 (wiki.gg only) give
  ftxi's. No terminal verdict in the corpus turns on these 78 (a legal action
  exists at each).
- **The two mate disagreements** (Chess in 5D "checkmate", ftxi counts actions)
  trace to a Chess in 5D defect, not a cluster: its exhaustive BFS hands
  `boardFuncs.moves` a tree node instead of a board, generates nothing and
  answers checkmate; its own enumerator lists the same 25 / 21 legal actions as
  ftxi (and cwmtt). Rule: C5 (a legal turn exists). Every other Chess in 5D
  checkmate in the corpus is corroborated by another source.
- **52 vs 54** (`notation/1.2@15`): an ftxi defect - the black branch of its
  timeline double step is guarded by the white piece types, so it never generates
  a black pawn's or brawn's two-timeline step (it does generate white's). The two
  extra actions are black pawns stepping two timelines onto a history board
  (branching). Whether that step exists at all is rules reference Q15 (4.6 vs
  [72]); ftxi is wrong for one colour either way.
- **Asked of the user:** the check ruling (RQ1-RQ4, with Q1); which text is the
  rulebook (proposed: `docs/rules/README.md`); Q15 (pawn timeline double step);
  the repetition rules (the game documents none - D2, Q23 - so it is a project
  decision, register part K); castling's attack test (spec §11.0 item 4 vs rules
  S2, spec issue 10); whether to report the two oracle defects upstream.

The engine core is C++20 (g++; no Rust or MSVC on the machine), a shared library
with a C API for Python (ctypes), built with CMake/Ninja from a project `.venv`;
the network is PyTorch on CPU under `python/`.

- **Engine E1 + E2 - committed 2026-10-01** (`engine/`, 259 tests, warning-free,
  also under an undefined-behaviour sanitizer; CI builds it with g++ 13 and 14 on
  Linux): board, multiverse (persistent, history shared in chunks of 16), 5DFEN,
  the H30 fingerprint (both published notation hashes reproduce), 5D Zobrist
  (H18), the C API; the present, activation and reactivation, the turn state
  machine and submission's present-shift test; pseudo-legal moves for every piece
  kind in 4D; move application (non-branching and branching jumps, timeline
  numbering, J6's order dependence); castling with the on-board attack test (S2),
  en passant within a board, promotion; the 5dpgn parser written from the
  notation spec's text (300,000 fuzzed records, no protocol violation); the
  variants that have an independent source (rules V3); `t5d-gate --parse-only`
  speaking the harness protocol. Every open question of the rules reference is a
  named `RuleOptions` / `PgnOptions` parameter with the documented default (table
  in `engine/README.md`); a non-default choice changes the ruleset ID and so the
  Zobrist key. Hooks for the check ruling: `CheckRuling::royals_safe` (C4),
  `in_check` (C6), `has_legal_action` (C5), `CastlingAttacks::AnyBoard`.
- **Network N1** (dense v1 with random weights; 152 tests) is built and reviewed;
  N2 (the attention structure) is running; `python/` is committed when N2 lands.

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
| H18 | built, tested | 5D Zobrist: coordinate-free content hash per board, multiply-rotate with (timeline, half-turn), addition across boards, ruleset ID, player to move, parity, creators, board size; incremental equals from-scratch; duplicated boards never cancel (engine E1, E2) |
| H19 | parked | Proof export: with the scanner |
| H22 | built in part | Items 2 (the present-shift half), 4 (castling with the on-board attack test, en passant, promotion) and 5 (the present, activation, submission) built and tested; item 1 (check), the royal-safety half of 2 (C4) and item 3 (mate) parked for the check ruling, behind hooks |
| H23 | built | The gate (oracle side); engine side waits for the engine |
| H24, H26, H28 | built, tested | The generator and the 5dpgn/5DFEN parser are in-house (`engine/`), written from `docs/rules/README.md` and the notation spec's text; 259 engine tests; boundary clean |
| H27 | built, tested | The oracle harness, committed 2026-10-01 |
| H30 | built, tested | Step 2A (a04678c): every oracle reports its boards per ply; no placement divergence across the corpus |
| H31 | not started | Step 3 (CHECKPOINT 1) |
| H32-H34 | not started | Step 2 |
| H35 | done | Commit and nightly tiers green on Linux, 2026-10-01 (runs 36827435595, 36827459456) |

### A-G, I - representation, network, search, training

| ID | Status | Notes |
|---|---|---|
| A1-A3 | built, tested | Network N1: tokens, features, per-board flags, Δt-to-present (committed with `python/`) |
| A4 | built, tested | Engine: every board immutable once published; persistent multiverse sharing history in chunks |
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

Waiting on the check ruling (CHECKPOINT 1, 2026-10-01), as the brief directs:
- the engine's check detection (H22 item 1) and the phantom-board definition;
- turn legality at submission - the royal-safety half (C4); the present-shift half
  is built (E2);
- naive mate detection and the hypercuboid detector (H22 item 3);
- castling's cross-board attack option (spec issue 10);
- the scanner (H1-H5, H10-H17, H19) and experiment 6b (M8), hence the NNUE (F4);
- search on real positions (E1-E9 on 5D states), and the typed edges' "check"
  relation (B11);
- the legal-move re-measure of movegen throughput (M9) and the |E| curve from
  legal moves (M7) - their pseudo-legal versions proceed now.
Waiting on the user (no ruling yet): repetition rules (register part K) - every
consumer of repetition (MCGS graph-history, solved-store keys, df-pn, activation
semantics, variant terminal values).

## Spec issues (wrong, ambiguous, contradictory, silent)

Each: where, the evidence, what was done (never an improvised committed-class
decision), and what is parked.

1. **No rulebook is pinned** (§11.0 "settle by rulebook"; CLAUDE.md; oracles/README
   "Not adjudicated"). The game's official site publishes no rules text; the
   community wiki refuses automated fetches. The engine's rules are therefore
   written from public descriptions, collected with citations in
   `docs/rules/README.md` (no oracle code or oracle documentation used), and
   cross-checked by the gate; where sources conflict the point is an open question
   for the user. *Raised with CHECKPOINT 1: which text is the rulebook.*
2. **Notation spec contradicts itself** (oracles/third_party/notation at 99f6163;
   README vs fen.ebnf). Readings adopted by the harness's canonical fingerprint and
   by the engine's parser (H28), each to confirm: (a) fen.ebnf swaps the colour case
   of `piece_white`/`piece_black` - the prose and both examples (white upper case)
   are followed; (b) fen.ebnf allows digits only in the first row - the prose (any
   row) is followed; (c) README lets the `+` sign be omitted on positive timelines,
   fen.ebnf's `board_layer` requires it - the canonical form writes `+1`, and the
   parser accepts both; (d) the unmoved-marker set: fen.ebnf lists pawn, king,
   rook; README "5DFEN Strictness" adds brawn - the README's four are followed; (e)
   header spacing (fen.ebnf allows none, README requires one) - the parser accepts
   both; (f) `Promotions` is space-separated in one README place and
   comma-separated in another - the parser accepts both; (g) whether `Variant`
   substitutes for `Board` (fen.ebnf names `variant`) - accepted as a synonym when
   `Board` is absent, as the harness already does. (h) "5DFEN Strictness" says pawns, rooks,
   kings and brawns are marked unmoved "if needed"; the 5DFEN section says a
   move-sensitive piece that "hasn't moved yet" must carry `*`. Read as "when
   unmoved" (the fingerprint marks every unmoved pawn, rook, king and brawn,
   whatever castling or double-step rights remain).
3. **Notation spec "Hashing" leaves the full-state order open within a turn**
   (README:486-492 sorts by timeline then turn; each turn has a white and a black
   board). The canonical fingerprint puts white's board before black's (time
   order). It also marks the hashing standard "still just a proposal"; it is used
   as the H30 fingerprint because it is the pinned specification's own canonical
   form, with both published examples as conformance tests.
4. **No layouts for named variants in the notation spec**: the only name-to-layout
   table in the pinned notation repository is code (`BOARDS` in its parser), which
   the engine may not use (H28). Variant layouts for the engine come from public
   game documentation (`docs/rules/README.md`); a variant without an independent
   source stays `unsupported` in the engine and its ruleset ungated.
5. **C17 / §7.3 do not define a board's "causal past" in a multiverse**
   (committed-class). "Superseded boards attend only to their own past" can mean
   (i) boards earlier on the time axis - which breaks freezing, because a later
   branch into the past adds boards at earlier times; (ii) the board's lineage (its
   timeline's history and its parent's up to the branch point); or (iii) creation
   order - not state-defined across transpositions (§8.5). Done: the cached path is
   built generic over a state-defined causal-past relation, and the §15.2
   equivalence tests run with each candidate. **Parked: the choice** (for the user,
   with the cached-path decision C2).
6. **§11.3 / I9 key exact stores by ruleset, but §12.6's small-board structure
   variants share the rules**: a 5x5 position and the same pieces on 8x8 collided
   in the first Zobrist key (found by review). Done: the board size is a key
   ingredient (H18 lists "content + (timeline, time) + ruleset ID"; the size is
   added beside the ruleset). Proposed spec edit 2 below.
7. **No rule says whose turn it is in a parsed starting position with no moves**
   (§11.0 item 5; the notation's 5DFEN carries per-board colours, not the player to
   move). Done: the engine takes it from the caller, or infers it only where the
   boards settle it whatever the activity rule says (the earliest initial heads
   share one colour), and refuses otherwise. To revisit with E2's present rule.
8. **The 5DFEN `+` letter prefix** "extends the number of pieces" but no `+`
   piece is defined. Done: `+X` reads as `X` (both canonical writers drop the
   `+`), documented as a reading pending a ruling; rejecting it instead is a
   one-function change.
9. **Network stage N1 - committed-class points the spec leaves open** (each is a
   config parameter in `python/t5d/net/config.py`; every default is a
   placeholder, not a decision; the choice is the user's or a trained ablation's):
   (a) `global_register_pairs` (B4, B8; §4.1, §5.2): a global register has no
   board, so its position under RoPE is undefined - `position_free` (default:
   pairs with a global register unrotated on both sides) or `anchored`
   (coordinate 0 after recentring; exact and cheaper); (b)
   `board_register_spatial` (§5.3 "must not receive a (file, rank)", B4, B8):
   under RoPE an unrotated register sits at a1 - `learned_angle` (default),
   `position_free`, or `zero_angle` (the a1 collision, kept as an arm); (c)
   **§9.2 contradicts §12.4 / B8 / D7**: §9.2's table pools global aux heads from
   the global registers, while B8 and §12.4 say aux heads pool from the trunk and
   registers stay free scratch - `aux_global_pool` = `squares` (default) or
   `global_registers`; the value head reads the global registers (§8.3); (d)
   `policy_promotion` (D1; §8.1 gives promotions to different pieces identical
   scores) - `none` (default, literal §8.1) or `offset`; (e) B5's unstated
   details: the attack/defend head's rotary allocation, how near "near-NoPE" is
   (π/8 over an axis's range), the same-square head's split (6 timeline + 10
   time pairs); (f) `binned_bias_rope` (§4.1 "RoPE for file, rank and near-time"
   vs register B6 "RoPE + binned clamped biases") - `keep` (default) or
   `near_time`; (g) D4's opponent-reply target: the move space is undefined
   (replies on boards our move creates have no token yet); (h) D4's per-square
   "deep attack facts beyond the §6 input bias": the label set is a placeholder
   until the §6 input relations are fixed; (i) D4's moves-left unit
   (board-moves or turns) depends on D2; (j) D3's pooling operator ("pooled from
   global registers"; mean is used).
10. **Castling's attacked-square test: spec §11.0 item 4 vs the rules reference.**
    The spec says every "is this square attacked?" query inside castling is a 5D
    query; the rules reference (S2, high, a developer ruling) says castling is
    barred only by attacks from an enemy piece on the castling board. Done: the
    engine implements the rules reference by default; the spec's cross-board
    reading is a named parameter, unimplemented because it needs the attack
    definition of the pending check ruling. For the user, beside the check ruling.
11. **Reversed Royalty's king: spec §11.0 item 1 vs the rules reference.** The spec
    says royals are "one king per timeline, plus the royal queen in Reversed
    Royalty"; the rules reference (V3) gives Reversed Royalty a royal queen and a
    common king, which is not royal (4.12). The engine follows the rules
    reference. Proposed spec edit 11.
12. **The fingerprint cannot see en-passant rights** (H30; 5DFEN has no en-passant
    field). Two states that differ only in an en-passant right share a
    fingerprint, so a parse divergence confined to one is invisible to the gate's
    state comparison (it still shows as a count or legality difference). The
    engine keeps the right in its Zobrist key only while it can be used. Noted;
    no change.
13. **The engine contract and set-ups that cannot be built.** A record whose
    set-up cannot be built (a custom position without board strings,
    contradictory headers) can only end `rejected` / `unreadable` with no
    position; the harness contract allows a run with no position only for
    `unsupported`. The contract is amended with the parse mode (step 2B).
14. **Notation readings found by engine stage E2b.** S6 is silent on the
    notation's "more than one king for either player" clause (the parser reads
    O-O board by board; the whole-start reading is a parameter); the board names
    `Even`, `Odd` and `Custom ...` come from the harness contract, not the
    notation spec; the notation spec's own "Actual game" record holds two moves
    that are impossible as written (15. (1T13)Ba4+ and 16. (0T14)Bg2).
15. **Transposition families are smaller than factorial** (an interaction, not a
    contradiction). §8.5 and §11.3 say a multi-board turn yields a factorial
    family of move orders; under the Guide's reading of Q4, optional moves come
    before the move that shifts the present, a rewinding branch comes last, and J6
    makes some orders reach different states. It bears on MCGS (C11) and the TT's
    expected hit rate.

## Proposed spec edits

Edits the spec or the register needs, for applying upstream. Each: where, what it
says now, what it should say, why.

1. **§11.0 "Tiered CI", first sub-bullet**: says "ftxi's engine (fast, Python
   bindings) on every commit". The harness drives ftxi through a C++ driver and
   its stock `5dtools` CLI, because the Python bindings expose no per-position
   check verdict (PROVENANCE.md). Proposed: "ftxi's engine (fast; a C++ driver
   over its core library, held to its stock CLI) on every commit".
2. **§11.3 "The trap: XOR-Zobrist cancels in 5D" / H18**: the key's ingredients
   are content (coordinate-free), (timeline, time) by multiply-rotate, and the
   ruleset ID. Proposed: name every ingredient - board contents per (timeline,
   time), the ruleset ID, the player to move, the even/odd start, each timeline's
   creator, and the board size (structure variants share the rules but not the
   positions) - and say the mixed board terms combine by 64-bit addition.
3. **§11.0 "Running the gate" / H30**: proposed to define the fingerprint as the
   notation specification's full-state MD5 ("Hashing"), with white's board before
   black's within a turn (the notation is silent), beside a placement-only hash
   and board and piece counts; and to say the harness, not each implementation,
   canonicalises and hashes (H33); that the unmoved marker means "has not moved"
   (spec issue 2h); that a marker-only difference is reported per run and stops
   confirmation, not comparison; and that a source exporting its own hash (as
   5d-chess-js does) is held to the harness's fingerprint.
4. **§12.6, the ±0 note**: proposed to record the mapping the engine uses: `-0`
   → -1, `+0` → 0, `-n` → -(n+1), `+n` → n, so adjacent timelines differ by one.
5. **§9.2 "Aux heads" row**: says global aux heads pool from the global
   registers; §12.4, B8 and D7 say aux heads pool from the trunk and registers
   are never assigned aux targets. Proposed: "Mean-pool board squares (per-board
   and global)", or a decision recorded in the register (spec issue 9c).
6. **§5.3 and §4.1/§5.2**: proposed to say what a register's position is under
   RoPE - per-board registers ("must not receive a (file, rank)": a learned
   off-board coordinate, or spatially position-free pairs) and global registers
   (no board: position-free pairs, or anchored at the recentred origin) - spec
   issues 9a, 9b.
7. **§8.1**: proposed to say how promotions are scored (the bilinear form gives
   every promotion piece the same score) - spec issue 9d.
8. **§12.4**: proposed to define the opponent-reply move space, the per-square
   label set, and the moves-left unit - spec issues 9g-9i.
9. **§4.1**: proposed to fix B5's per-head details and how B6's binned biases
   relate to RoPE on the time and timeline axes - spec issues 9e, 9f.
10. **§11.0 "Running the gate" (settle by rulebook) and §15.1 Phase 0**: name the
    rulebook. Proposed: `docs/rules/README.md` - the cited rules reference built
    from the in-game Guide and community sources, never from an oracle - with its
    open questions settled by the user's rulings and recorded there.
11. **§11.0 item 1**: "plus the royal queen in Reversed Royalty" - proposed: "in
    Reversed Royalty the royal is the queen; its king is a common king, not royal"
    (spec issue 11).

## Session log

- **2026-10-01** - Pre-flight passed. Design documents moved into `docs/design/`
  (they had been kept untracked as `Specification documents/`). Step 1 started.
