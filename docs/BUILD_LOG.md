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

*Updated 2026-10-02.* Work in the tree that is not yet committed is listed under
**In the tree, uncommitted** - a session that resumes starts there.

- **Done:** pre-flight; step 1 (CI green on Linux, H35); step 2A (H30
  fingerprints, the notation parse-only oracle); the rules reference; CHECKPOINT 1
  - reported, then **ruled by the user** (register H36-H40 and I11 in the updated
  design documents, committed 666c4a0; rulings R1-R7 recorded in
  `docs/rules/README.md`, now the rulebook, H38); engine E1+E2 (ae35da3, CI green
  on g++ 13 and 14); network N1+N2 (bafa2c4; 447 tests; CI job `network`); the
  upstream drafts of the two oracle defects (3277bb2); engine E3 - expansion and
  the Phase 1 measurement tools (committed 2026-10-02; 299 tests); network N3 - the
  block-causal scope, the cached path and the §15.2 network rows (committed
  2026-10-02; 558 tests); engine E4 - check status and submission legality
  (H36), legal actions as the harness counts them, the naive and hypercuboid
  detectors under H37, `t5d-gate`'s full mode (merged 2026-10-02; 355 tests; the
  full corpus in 8.3 s).
- **In the tree, uncommitted (2026-10-02):**
  - *Step 2B* (H32 stable answers - required sets at 100%, `golden --write
    --repeat 2`, per-position cost files, calibration probe; H33 - golden
    `MANIFEST`, one verified snapshot per run, the harness-modified check,
    CODEOWNERS; H34 telltales; the engine's parse mode; H36's `in_check`
    definition; H37's guard against budget-limited verdicts; H38's rulebook pin in
    `gate.toml` with `harness.py rulebook --pin`): built and fixed after review
    (17 findings), selftest 589 OK. `verify` fails only on the golden files, which
    need `py oracles/harness.py golden --write --repeat 2 --jobs 8` - **on a quiet
    machine** (the recording measures each answer's cost, and the costs decide the
    required sets). Commit 2B with the re-recorded golden files, never without
    them. Also to apply with it: AGENTS.md's list of forbidden ways to make the
    gate pass (the coverage floors are gone - name budgets, `[stability] margin`,
    `proof_share`, re-recording on a loaded machine, hand-editing cost files,
    `calibration.json` or `MANIFEST`, moving the rulebook pin unreviewed) and
    CLAUDE.md's commands (`--repeat 2`, `rulebook`).
  - *Network N3's CPU wall timings* (the code is committed): need a quiet machine.
    From `python/`: `py -m t5d.net.cache_bench --boards 240 --idle-processes
    t5d-gate.exe t5d-measure.exe t5d-bench.exe t5d-tests-long.exe --skip-existing
    --out <dir>`, then `--boards 600 --dtype fp32` and `--boards 600 --dtype bf16`
    (about 45, 70 and 45 minutes); the benchmark itself refuses a loaded machine.
- **Next:** merge E3 and E4; wire the rulings into the gate (known-divergence
  entries for ftxi's 78 check verdicts and for both oracle defects, as telltales;
  the rulebook pin, H38; the engine configured, parse mode then full mode); the
  golden re-record with `--repeat 2`; M9 then M7 with legal moves; the scanner
  and 6b.
- **Waiting on the user:**
  - **Rules Q4** (may a player keep moving after the present has shifted?). The
    engine follows the Guide's "no". Evidence gathered since: nine corpus records
    by three authors make such moves; all three oracles allow them; and E4's
    review found that every legal-action count where an oracle counts more than
    the engine is explained by Q4 - under the other reading the engine matches
    ftxi and cwmtt. So Q4 decides whether the gate can pass on most of the
    corpus. Under the other reading the engine is ready except for search time
    (E4 fix round).
  - **Q5's alternative under R1**: with `mandatory = AllActive` (every active
    timeline must be played, not only those in the present), R1 still passes only
    the active boards in the present, so a stuck mandatory board ahead of the
    present gives stalemate, not checkmate. Default rules are unaffected.
  - Real game records for the |E| curve (M7: the corpus's real games reach about
    240-399 boards in two games, and only under Q4's other reading).

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

### 3. CHECKPOINT 1 and the ruling-independent engine work - done (ruled 2026-10-01)

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
- **Ruled 2026-10-01** by the user (register H36-H40 and I11; rulebook entries
  R1-R7): check status passes all *active* boards *in the present*, so RQ1-RQ4 are
  all "no" - ftxi's "check" at the 78 positions is wrong, Chess in 5D's "no check"
  right - and a royal capture counts even if it would branch into an inactive
  timeline; submission legality is a separate test, on the actual turn: any royal
  capture from any of the opponent's playable boards, active or inactive, makes
  it illegal; moving into check mid-turn is allowed (H36). A mate verdict needs an
  exhaustive search that terminated; an exhausted budget is "unknown" (H37).
  `docs/rules/README.md` is the rulebook, pinned, sources ranked (H38).
  Activation is the count rule (Q1 closed). The 18 positions Q1 would have moved
  stay where the count rule filed them: in the checkpoint's own data (clusters.json,
  whose count-rule activity matched cwmtt's at all 611 positions it reports) every
  one of them is check only under Q10's readings 4 and 5 - an attacker on an active
  optional board ahead of the present - so none is check under R1 and none flips.
  The engine's check detection is to confirm it independently once it can read
  those records (cwmtt niceAndrey and wide need Q4's ruling and the lenient
  annotation reading). The pawn's double step exists along both axes for both colours (Q15
  in part; ftxi's missing Black step is the defect). No repetition rule, a project
  decision (H39); self-play's turn cap is a convention scored as a draw (I11).
  Castling and en passant attack tests are per-board (H40; the engine's reading
  already). The two oracle defects are drafted for upstream in
  `docs/checkpoints/upstream/` (the user files them) and go into the
  known-divergences file as telltales. The drafts are done: each has a hand-built
  minimal record reproduced on the pinned code, the root cause at the pinned commit,
  a fix checked in a scratch copy (5d-chess-js: pass the node's board, `mate.js`
  157 and 161; ftxi: the Black guard at `multiverse_base.cpp` 937), and the rule
  from public sources. Two further ftxi leads, not yet checked against the
  rulebook: a White brawn's non-capturing two-axis step listed three times, and a
  segmentation fault in `5dtools ... naive` on a 5x5 custom record.
- **The engine's parse mode on the corpus** (`t5d-gate --parse-only`, before the
  harness's parse mode exists): 647 positions; the early stops fall into three
  kinds. (a) Rules Q4, moves after the present has shifted: the engine follows
  the Guide's reading and refuses them; nine records make such moves (cwmtt NP,
  NP0, niceAndrey, 100_timelines, futures, another; ftxi futures; notation 1.2 and
  game-2) - played records from three authors, which bears on Q4, a question for
  the user. (b) Present and timeline annotations that do not match the state -
  `(~T2)` after a first branch in cwmtt silly, wide and
  chessin5d-illegal-move-example: the notation spec calls these tokens "purely for
  better human comprehension" that "should match"; the engine rejects a mismatch
  (`verify_annotations`), the oracles ignore them. (c) Notation outside the
  notation spec: a long form `(0T1)Pe2(0T1)e3` (cwmtt many2), a jump without its
  source square `(-1T3)K>>(0T2)c3` (cwmtt ctp1) or without any coordinate `N>>d3`
  (ftxi verysmall), parenthesised variations (ftxi softmate), 5DFEN boards under a
  named variant (ftxi niceAndrey), a repeated turn number (notation aivsai), and
  two variants with no independent source (Timeline Invasion, Marauders). Kind (c)
  is classified against the notation spec when the harness's parse mode lands;
  kind (b) needs a parser policy (lenient by default is proposed: the spec says
  the tokens are for humans); kind (a) waits on Q4.

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
- **Network N1 + N2 - committed 2026-10-01** (`python/`, PyTorch on CPU, random
  weights; 447 tests, ruff clean; 41 + 32 injected mutants killed). N1: tokens
  (66N + 16, board-major), piece and per-board features with Δt-to-present, 4D
  RoPE (file 3 / rank 3 / timeline 4 / time 6 pairs, FP32), 2 + 16 registers, the
  dense trunk (d 256, L 16, 8 heads, FFN 8x, BF16 autocast), the bilinear pointer
  policy, the 51-bin value head and the auxiliary heads (21,358,601 parameters,
  F1; the L24-ff12 preset is 44.5M, F2). N2: the move-graph mask on 2 heads (B9),
  the routing table (B10), typed edge biases (B11) and board-pair smolgen (B12)
  behind flags, and B14's block-sparse attention in a CPU gather form - 4-board
  query blocks over whole key boards, plans built once per forward, per-block
  recomputation under autograd - with chunked and dense paths as references; at
  240 boards one FP32 forward takes about 9 s and 0.86 GB on the dev CPU
  (indicative, not a §15.3 figure). Every committed-class ambiguity is a named
  parameter with a placeholder default (spec issues 9 and 16).

### 4. After the ruling - E4 built and merged (2026-10-02); the gate run next

Engine stage E4, built in its own worktree and merged into main on 2026-10-02:
- **Check status** exactly as R1/H36: pass every active board in the present, then
  any royal capture (kings and royal queens, never a common king) from any of the
  opponent's playable boards, onto any board, branching captures counting even
  into a would-be-inactive timeline. **Submission legality**: the same question on
  the actual position after the turn, no pass. One royal-capture scan with the
  move generator's semantics, held equal to it on random multiverses. The default
  check ruling of `GameState::submit` and of the replay; `CastlingAttacks::AnyBoard`
  removed (H40).
- **Legal actions**: `count_legal_actions` counts what the harness defines - a
  move set, its timeline-creating moves in creation order;
  `count_legal_successors` counts distinct successor states (for search).
- **Mate detection** (C5, C8, C9, C13; H37): a naive depth-first detector and a
  hypercuboid detector designed in-house from spec §11.0 item 3 (design note in
  `engine/README.md`; no oracle source read - the rules-fidelity reviewer compared
  and found nothing ported). Checkmate or stalemate only from an exhaustive search;
  a node, time or memory budget answers "unknown" - tested for every node budget
  below the exhaustive need. Every search has a memory ceiling (2 GiB estimate by
  default).
- **`t5d-gate` full mode**: in_check, legal_actions (capped), terminal - never
  unknown: a hypercuboid-then-naive portfolio on doubling budgets, the first to
  finish answering, a disagreement ending the run in `error`. The whole corpus
  runs in 8.3 s (cwmtt/manyChecks 4.1 s; no process above 50 MB).
- **Reviews**: three adversarial reviews (rules fidelity, detectors and count, gate
  contract and tests) found no soundness bug - an independent brute-force reference
  matched both detectors on about 6,660 positions under 15 rulesets - but found a
  search-time defect (manyChecks unfinished after 25 minutes), the count
  definition, and test gaps; the fix round closed them, and its own brute-force
  reference found and fixed a naive miscount (two move sets reaching one state
  were merged). A follow-up review of the reworked hypercuboid regions is running.
- **The expansion's legal-moves switch** (`ExpandOptions::legal_moves`, off; spec
  issue 28): keeps the moves after which the action can still be finished legally.
  Median 11-218 ms per expansion from 100 to 400 boards, with a tail of minutes -
  unfit to turn on as it stands (spec issue 46).

### 5. Phase 1 measurements - tools built, measurements pending a quiet machine

Engine E3 (committed 2026-10-02) built what M9 and M7 need: `expand()` (the
mover's moves, the move graph, W(k) and E(k), side-to-move typed relations, board
and square features), its C API, `t5d-measure` (per-position W/E at k = 0/4/8/16,
and `--summary`, a curve de-duplicated by position and by game) and `t5d-bench`
(search-like streams, interleaved rounds, thread CPU time, per-round thread
scaling, raw / caliper-matched / no-move flatness series - labels, never
verdicts). Moves are still pseudo-legal; `expansion_moves()` in `src/expand.cpp`
is the one place to switch to E4's legal actions after the merge.

Preliminary figures (2026-10-02, dev, g++ 16.2 -O2, SYNTHETIC random-play
positions, 16 games per class; **machine loaded 17-97%, so not measurements**):
scope+relations expansion 41.6k/s per thread at N240 T8 (|E(8)| 12.9, 810 moves)
and 24.3k/s at N600 T12 (|E(8)| 38.1, 1,601 moves) - both about 2x M9's per-GPU
targets on one thread; 12 threads 9-11x. The history term left after the O(N)
restructure is about 0.7 ns per board (2% of an expansion at N600). Matched
flatness, three loaded rounds: +13% [-6, +17] - inconclusive. The corpus |E|
curve (with `--moves-after-shift`; distinct positions) reaches only 240-399
boards in 2 games: it cannot decide M7 (spec issue 26).

**Reading of the brief's STOP band** (the brief, not the spec, says "a §15.6
result within ±10% of its threshold" stops the run; spec issue 27): ±10% relative
to the threshold's value - throughput 18,000-22,000/s at 240 boards and
9,000-11,000/s at 600; flatness growth of 9-11% from 100 to 600 boards; the 6b
percentages 45-55% of positions (and 2.7-3.3 / 4.5-5.5 forcing half-turns at the
median). A result inside a band stops for the user.

Next: the quiet-machine run - `build/engine/t5d-bench --samples 15 --emit
build/m9/pos --json build/m9/m9.json` (accept only if the 1-thread load median is
near the bench's own) - then M9's verdict in `docs/MEASUREMENTS.md`, re-measured on
legal moves after E4; M7 on the corpus as an upper bound and a request to the user
for game records.

### 6. Phases 2-3 - not started

## Register status

### H - rules engine, gate, verification layer

| ID | Status | Notes |
|---|---|---|
| H1-H5 | not started | The scanner: after check detection and legality (E4) and the movegen measurement (M10) |
| H6 | not started | Material eval v1 with the scanner; NNUE v2 waits on 6b (M8) |
| H7 | not built (rule) | NNUE training: no training runs; NNUE waits on 6b |
| H8, H9, H20, H21 | not built (rule) | Held open / refused |
| H10-H14 | parked | Async ladder rungs: with the scanner |
| H15 | parked | Synchronous debug mode: with the scanner and search |
| H16, H25, H29 | not built (rule) | Refused |
| H17 | parked | Lazy SMP: with the scanner |
| H18 | built, tested | 5D Zobrist: coordinate-free content hash per board, multiply-rotate with (timeline, half-turn), addition across boards, ruleset ID, player to move, parity, creators, board size; incremental equals from-scratch; duplicated boards never cancel (engine E1, E2) |
| H19 | parked | Proof export: with the scanner |
| H22 | built, tested | All five items of §11.0 (E1, E2, E4); the register row's wording predates H36/H40 (proposed spec edit 40) |
| H23 | built | The gate (oracle side); engine side waits for the engine |
| H24, H26, H28 | built, tested | The generator and the 5dpgn/5DFEN parser are in-house (`engine/`), written from `docs/rules/README.md` and the notation spec's text; 259 engine tests; boundary clean |
| H27 | built, tested | The oracle harness, committed 2026-10-01 |
| H30 | built, tested | Step 2A (a04678c): every oracle reports its boards per ply; no placement divergence across the corpus |
| H31 | done | CHECKPOINT 1: seven clusters triaged (903f4ae), settled by the user's rulings (H36-H40) |
| H32-H34 | not started | Step 2 |
| H35 | done | Commit and nightly tiers green on Linux, 2026-10-01 (runs 36827435595, 36827459456) |
| H36 | built, tested | Engine E4: check status and submission legality per R1; the gate's comparison with the oracles is next |
| H37 | built, tested | Engine E4: verdicts only from exhaustive searches; node, time and memory budgets answer "unknown" (tested for every budget below the need) |
| H38 | done (gate pin pending) | `docs/rules/README.md` is the rulebook: source ranking, rulings R1-R7; the gate's pin of it is step 2's follow-up |
| H39 | done | No repetition rule; the engine has none to build |
| H40 | built | Castling's attack test per-board (S2) and en passant per-board (S7); `CastlingAttacks::AnyBoard` removed (E4) |

### A-G, I - representation, network, search, training

| ID | Status | Notes |
|---|---|---|
| A1-A3 | built, tested | Network N1: tokens, features, per-board flags, Δt-to-present |
| A4 | built, tested | Engine: every board immutable once published; persistent multiverse sharing history in chunks |
| A5 | not started | Piece-centric tokenisation, behind a flag |
| B1, B2, B4, B8 | built, tested | Network N1: one weight set, unified attention, 4D RoPE, 2 + 16 registers |
| B9 | built, tested | Network N2: move-graph mask on 2 heads (direction and register queries are placeholders, spec issue 16) |
| B14 | built in part | Network N2: block-sparse attention in a CPU gather form; the FlexAttention / GPU kernel is not built (no CUDA here) |
| B5, B6 | built, tested | Network N1, behind flags |
| B10, B11, B12 | built, tested | Network N2, behind flags (B12 as relative board-pair cells, full rank) |
| B3 | not built (rule) | Contingency only; no trigger |
| B7 | not built (rule) | Deferred; no §15.2/§15.3 need (d_head is a config value anyway) |
| B13 | not built (rule) | Refused |
| C1 | built, tested | `attn_scope = "dense"` is the default |
| C4 | built, tested | Full MHA (N1) |
| C17, C19, C20 | built, tested | Network N3: the block-causal scope (`attn_scope = "block_causal"`), the live set fully bidirectional, supersession recomputed once under the state-defined mask; the causal past is a parameter (spec issues 5, 29) |
| C2 | built (for §15.2/§15.3) | Network N3's cached path; the decision itself stays deferred (6c, §7.7) |
| C10 | parked | Lazy K/V for boards entering E: with the window, after M7 |
| C3, C3a (rung 1) | built, tested | Root refresh and the travel-triggered, visit-gated rung-1 fork (copy-on-write epochs), behind flags |
| C3a (rungs 2-3), C18, C22 | not built (rule) | Held open (M5 for rungs 2-3) |
| C5, C7 | parked | Window code only after the \|E\| verdict (M7) |
| C11 | not started | MCGS behind a flag; search on real positions waits on legality |
| C12, C13 | not started | Persistence (search); frozen K/V already persist across roots under C20's first reading |
| C15 | built in part | FP8 e4m3 frozen K/V (`frozen_kv_format = "fp8"`, applied by the forward itself); varlen/ragged kernels not built (none on CPU); C15 vs G1 is spec issue 34 |
| C16 | not built (rule) | Deferred; last, only if node-starved |
| C21 | not built (rule) | Refused; a mutant that snapshots is caught by the tests |
| D1, D3, D4 | built, tested | Network N1: pointer policy, 51-bin value head, auxiliary heads (D4's reply space, label set and moves-left unit are placeholders, spec issue 9) |
| D2 | not started | Undecided: both arms awaited by trained ablations 6g/6k |
| E1, E5, E7, E9 | not started | Search |
| E2, E3, E8 | not started | Behind flags |
| E6 | not built (rule) | Refused |
| F1, F2 | built, tested | 21,358,601 parameters (F1); the L24-ff12 preset, 44.5M (F2) |
| F3 | not built (rule) | Refused |
| F4 | parked | NNUE only after 6b passes (M8) |
| G1 | built, tested | BF16 trunk under autocast; FP32 heads and RoPE; the cache stores K/V in the live format (§15.2 row (e)) |
| G2 | not started | FP8 attention - no FP8 kernels on this machine |
| G3, G4, G5, G6 | not built (rule) | Deferred / held open / refused |
| I1-I10 | not started | Training code may be written and smoke-tested only; I7 and I8 not built (M5) |
| I11 | not started | Self-play's turn cap from the state, scored as a draw, its frequency logged: with self-play |

### M - the plan itself

| ID | Status | Notes |
|---|---|---|
| M7 | tools built | `t5d-measure`; the corpus cannot decide it (spec issue 26) - game records requested |
| M8 | not started | Scanner verdict (6b), after M9 (M10's order) |
| M9 | tools built | `t5d-bench`; preliminary only (loaded machine); the quiet-machine run is next |
| M11 | built | Licence recorded in PROVENANCE.md |

## Parked, and why

Unparked by the CHECKPOINT 1 rulings (2026-10-01): check detection, submission
legality, the mate detectors, the scanner and 6b, search on real positions, the
typed edges' check and pin relations, and the legal-move measurements (M9, M7).
Closed by them: repetition (H39 - there is none) and castling's cross-board attack
option (H40 - the per-board test is the rule, so `CastlingAttacks::AnyBoard` goes).

Still parked:
- the window code (C5, C7): until the |E| verdict (M5, M7);
- the NNUE (F4): until 6b passes (M5, M8);
- the M7 verdict's phases above about 130 boards: the corpus's real games end
  there (spec §10.4.2 asks for real game records) - waiting on game records from
  the user.

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
   with the cached-path decision C2). Built in N3 (2026-10-02) as `causal_past` =
   `lineage` (arrival side, the placeholder default) | `two_parent` (also the
   board a travelling piece departed from) | `creation`; see issues 29-30 and 35.
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
    **Resolved 2026-10-01** upstream: H40 and reversal 21 make the per-board test
    the rule; spec §11.0 item 4 now says so.
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
16. **Network N2's committed-class placeholders** (each a parameter, default
    first; nothing in the spec or register decides them): (a) `move_graph_direction`
    = symmetric - B9's mask direction; (b) `masked_head_registers` = read - what a
    register query on a masked head reads when no relation covers it; (c)
    `same_board_registers` = own_board - whether B10's "same board only" covers
    per-board registers as queries; (d) `relations_head_types` = all six B11 types
    (the narrow reading is attack and defence); (e) `relations_head_direction` =
    follows (a); (f) `rope_roles_source` = table; (g) under the routing table,
    `move_graph_heads` is not used and the table has one move-graph head. Also: §6
    calls board-pair smolgen "full-rank" while §9.2's sizing implies a rank-32
    bilinear form; the built default is full rank over relative board-pair cells,
    the bilinear form an option; and "14.7 MMAC per head per layer" is the cost of
    a 256-wide output over every pair, not of the generator. Proposed spec edits
    12-18.
17. **E's "some live piece"** (§10.4.1) is read as the mover's pieces: E comes
    from the move generator at the node, and the opponent's destinations enter at
    the child (§10.4.3's "search re-partitions per node"). R1 now makes the
    both-colours reading buildable; whether to is a scope choice, not built.
18. **Source boards outside W** (src_extra / src_only): the head of an inactive
    timeline is playable but gets no band; the spec does not say whether scope
    must hold the source square of a move for the policy pointer. Live in the
    corpus: src_only(8) > 0 at 9/7/7/8/13 distinct positions by size bucket.
    Counted apart, never folded into |E|.
19. **Travel-threat** is named in §6 and B11 and used as a class of forcing moves
    in §11.1; no edge is defined. Not built.
20. **"Per GPU's CPU allocation"** (§15.6 M9) is undefined: how many cores is one
    GPU's share? The bench reports per thread and per machine.
21. **What M9's "expansion" includes**: moves only, + move graph, + W/E, + typed
    relations, + per-board features? The full per-board feature export is O(N) by
    construction (it writes every board), so "flat in N" can hold only for the
    windowed expansion.
22. **"At fixed |E|"** (M9 flatness): |E| cannot be held fixed along a game; the
    bench approximates it by caliper-matching moves, |E(8)| and the mover's pieces
    within 10%.
23. **Branch origins: mandatory core or filler?** §10.4.3 lists them after the
    heads' recent k; §15.6's W column ((k+2)T - 1) counts them. |W(8)| with and
    without origins on the corpus: 34.5/33, 68/62, 122/114, 185/174 by bucket.
    The engine reports both.
24. **W(k)'s unit and the present band**: k counts half-turns (one board each);
    the present band is read as the active heads.
25. **5DFEN records no branch origins** (or creation order), so a position read
    from 5DFEN has a smaller W and no lineage parent - matters for the network's
    lineage-based causal scope (N3) too.
26. **The corpus cannot decide M7.** Real games in the corpus: 5 / 3 / 2 / 0 games
    in the 100-159 / 160-239 / 240-399 / 400+ buckets, at 8 / 16 / 24 timelines
    against §15.6's assumed 7-12, and only under the Q4 reading the engine does
    not use by default. §10.4.2 asks for real game records; requested from the
    user.
27. **The STOP band is the brief's, not the spec's.** "A §15.6 result within
    ±10% of its threshold" stops the run; the band's reading (relative to the
    threshold) is recorded in step 5. Not a spec edit.
28. **What is a "legal move" for the expansion after E4?** R1 judges legality per
    action ("moving into check during a turn is allowed"): a single move is legal
    if it is part of some legal action. The expansion will offer the moves that
    begin a legal action (E4's enumeration), to be confirmed at the merge.
29. **C17's causal past** (issue 5) is built three ways. `lineage`: a board's
    timeline predecessor and, for a branch's first board, its origin - the arrival
    side only; `two_parent`: also the board a travelling piece departed from;
    `creation`: every board created earlier (path-dependent - two move orders
    reaching one multiverse evaluate differently - and it cascades, issue 35). The
    time-axis reading breaks freezing and is not built. Lineage is the placeholder.
30. **The MCGS node key** (§8.5, C11, H18). Under lineage the network tells apart
    two states that differ only in a branch origin (a test pins it), but the 5D
    Zobrist key omits origins and 5DFEN records none; under two-parent the key
    also needs each travel's departure board; under creation no node key works
    (creation order is not a property of the state).
31. **A3 against C2/C17**: a frozen board's Δt-to-present and active flag change
    after it freezes. `frozen_features = intrinsic` (placeholder: frozen boards
    carry neither; a refresh gives them the anchor's values) | `current` (every
    shift of the present recomputes every frozen board, a cost §7.1 does not price).
32. **C20's "causal past plus the conditioned history of its refresh epoch"**:
    `supersession_scope = causal_past` (placeholder) | `past_and_epoch_history`. The
    union has a cycle under the creation reading unless the epoch's history is a
    fixed block (history boards read only history), which is how it is built.
33. **The §6 structure on frozen rows**: under the cached configuration the move
    graph, the relations head, typed biases and smolgen open or bias live query
    rows only (forced: they are present information).
34. **C15 against G1.** §10.3 says FP8 K/V obey §10.2 (cache and live share
    precision); that holds only for an FP8 trunk (G2, not built) or a network
    trained with FP8 frozen K/V (`frozen_kv_format = fp8`, built). With a BF16
    trunk, FP8 frozen K/V are RMS 8.7e-3 from the BF16 network - 3.8x the
    BF16-vs-FP32 gap.
35. **§7.3's supersession bound** (+12-24% upper bound) assumes one board reading
    all of T. Under the creation reading the cascade breaks it (analytic +194% to
    +966%; CPU +3,273% at 600 boards, 23 timelines, 280 boards recomputed for one
    jump); under lineage the counted cost is 0.3-12%.
36. **§7.1's "~8 live head boards" against real games.** The corpus's real games
    have 24 timelines at 240-399 boards: the paper cached-vs-dense ratio is then
    9.9x at 240 boards (not 29x) and 24.8x at 600 (not 49x), and the live-set
    ratio 10.8x (not 3.7x).
37. **§10.2's per-node tree memory** assumes about 528 novel tokens stored per
    node; under C19 the heads are recomputed every node and only superseded boards
    are stored: 1.08 MB each in BF16, 1-2 per node.
38. **§15.3's "KV memory per state"** counts every token; the cache holds frozen
    boards only (plus their trunk outputs, about 16 MB at 240 boards).
39. **"One refresh ≈ 10 node-evaluations"** is the windowed figure; with full
    history the model gives 29x at 240 boards and 49x at 600. The row stays
    unmeasured until the window exists.
40. **§7.6 says 0.164 TF per node; §7.9's table uses 0.159 TF** (528 queries, no
    global registers), giving +3.8% at 800 sims where §15.3 says +3.6%.
41. **Rung 1's trigger**: a move that creates a timeline (a non-branching jump does
    not); a fork moves the node and its descendants still in the old epoch.
42. **C15's varlen/ragged kernels** do not exist on CPU; not built.
43. **The block-causal full pass** (training under the scope, and history builds)
    costs 2.3-2.8x a dense pass on this CPU: per-row scope masks defeat the fast
    path. A block-sparse scope selection would fix it; not built.
44. **H37's "exhausted budget"**: the engine also has a memory ceiling, treated
    the same way (unknown; in the gate, an `error` end naming it). Proposed spec
    edit 38.
45. **§11.0 item 3's two detectors in the gate**: the spec says "naive as the fast
    path, hypercuboid for near-mate positions"; the gate's adapter runs them as a
    portfolio on doubling node budgets (hypercuboid first), the other
    cross-checking. Proposed spec edit 39.
46. **The legal-moves filter's cost** (spec issue 28): finding the moves that
    begin a legal action costs one search per move - median 11-218 ms per
    expansion at 100-400 boards, worst 247-522 s at 400-600. Against M9's targets
    (20,000 expansions/s at 240 boards) the expansion cannot filter this way; what
    the expansion offers (pseudo-legal moves, legal first moves, or whole actions -
    D2's ply split) needs deciding before search uses it.
47. **Register H22's wording** still says "phantom boards for the mate-vs-stalemate
    verdict" and "per-board chess rules as 5D queries", which H36 and H40 replaced.
    Proposed spec edit 40.

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
    open questions settled by the user's rulings and recorded there. **Applied
    upstream 2026-10-01** (H38; spec §11.0 "Triage by cluster").
11. **§11.0 item 1**: "plus the royal queen in Reversed Royalty" - proposed: "in
    Reversed Royalty the royal is the queen; its king is a common king, not royal"
    (spec issue 11).
12. **§6 / B9**: state the mask's direction, that a token always reads itself,
    what register queries on masked heads read, and that padding is excluded
    (spec issue 16a, b).
13. **§6 table / B9-B10**: say whether B9's "at least 2 heads" counts the
    relations head; define "attack/defend relations" in B11's types and that
    head's direction; state that "same board only" covers per-board registers as
    queries (16c-e, g).
14. **§4.1 / B5**: what per-head specialisation means when the routing table is
    off (16f).
15. **§6, §9.2, §14.4 / B12**: keep "full-rank" and specify the generator - the
    global latent read through a per-head table over relative board-pair cells,
    bilinear in (latent, cell), with no absolute board index; record the
    board-descriptor bilinear form as the rank-32 alternative; correct "14.7 MMAC
    per head per layer" (that is 256 x 240^2, a 256-wide output over every pair);
    say that pairs with a global register get no smolgen bias.
16. **§6 / B11**: one learned scalar per head per relation type, shared across
    layers, every head carrying every type.
17. **§6.1, §10.1 row 4, §13.1 item 6 / B14**: block-sparse attention is built in
    a CPU gather form; the FlexAttention kernel is still to build; under autograd
    the memory bound needs per-block recomputation.
18. **§6 vs §11**: say what the generator exports as "the move graph" (the side to
    move's moves - engine stage E3's reading - or both colours).
19. **§10.4.1**: define E as the out-of-window travel destinations of the
    mover's moves, a branching move counting at the history board it targets
    (spec issue 17).
20. **§10.4.3**: define W(k) - unit, present band - and say whether branch
    origins are core or filler (issues 23, 24).
21. **§15.6 / M9**: define the CPU allocation per GPU, what "expansion" includes,
    and how flatness at fixed |E| is measured (issues 20-22).
22. **§6 / B11**: define the travel-threat relation, or drop it and cite §11.1's
    forcing-move class instead (issue 19).
23. **§2.2**: the has-moved flag exists only for pawns, rooks, kings and brawns
    (5DFEN's `*`); say so.
27. **§7.3 / C17**: define a board's causal past (the chosen reading, spec issue
    29), and say that a frozen board reads only itself and frozen boards of its
    past.
28. **§8.5, §11.3, C11, H18**: the node key includes each timeline's branch origin
    under the lineage reading (and each travel's departure board under two-parent);
    the creation reading has no node key (issue 30).
29. **§2.3 / A3 with §7.3**: what a frozen board's state-dependent features are
    under the cached configuration (issue 31).
30. **§7.3 / C20**: what "plus" means, and the epoch's history as a fixed block
    (issue 32).
31. **§6 / B9-B12**: under the cached configuration, edge-derived masks, biases and
    smolgen apply to live query rows only (issue 33).
32. **§10.3 / C15 with G1**: FP8 frozen K/V need an FP8 trunk or FP8-frozen-K/V
    training (issue 34).
33. **§7.3**: the supersession bound depends on the causal-past reading; record the
    creation cascade (issue 35).
34. **§7.1, §13, §15.3**: restate the paper ratios per timeline count - 8 against
    the 24 the corpus's real games have (issue 36).
35. **§10.2's GQA memory table and §10.3**: the per-node delta is superseded boards
    only; KV per state counts frozen boards (issues 37, 38).
36. **§7.6, §7.9, §15.3**: reconcile 0.164 TF / 0.159 TF and +3.6% / +3.8% (issue
    40).
37. **§15.3 rows**: write back the measured figures (M4) once the quiet-machine
    wall timings exist, CPU-labelled; `docs/MEASUREMENTS.md` holds the counts now.
38. **§11.0 item 3 and H37**: "An exhausted budget yields 'unknown'" -> "An
    exhausted budget - nodes, time or memory - yields 'unknown'" (spec issue 44).
39. **§11.0 item 3**: after "Run both: naive as the fast path, hypercuboid for
    near-mate positions", add "the gate's adapter runs them as a portfolio on
    doubling node budgets, the other cross-checking" (issue 45).
40. **Register H22**: "royal-capture check detection (phantom boards for the
    mate-vs-stalemate verdict) ... per-board chess rules as 5D queries" -> "check
    status per H36 (pass on the active boards in the present) ... per-board castling
    and en passant attack tests (H40)" (issue 47).

## Session log

- **2026-10-01** - Pre-flight passed. Design documents moved into `docs/design/`
  (they had been kept untracked as `Specification documents/`). Step 1 started.
