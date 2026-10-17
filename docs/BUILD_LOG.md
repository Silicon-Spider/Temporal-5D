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

### Recovery after the lost machine (2026-10-17) - read this first

The development machine was lost after 2026-10-04 and rebuilt (same hardware, a
fresh Windows profile; "Environment" below, machine row `dev-r` in
`docs/MEASUREMENTS.md`). The repository came back as a fresh clone of the remote,
whose only branch is `main` at 3b33942 - the last commit of 2026-10-04 - with all
four submodules at their pinned commits and unmodified.

**Survived** (all of it on `main`): everything "On main" in "Current state
(2026-10-04)" below - the rules engine, the search core, verification v3's
components, the network N1-N5, C24, the oracle harness and the golden files,
t0database, and this log.

**Lost - never pushed - and to be redone:**
- **The gate with the engine configured** (branch
  `worktree-agent-a1a2868a52f6c83b9`, 61e9aa8): 113 triaged known divergences
  (110 R1, 3 notation), no engine defect, nightly passed. Redone under H47 (step 1a
  of the CHECKPOINT 6-7 plan).
- **Engine E10** (it was running in a worktree): Hunt under 100 boards, the df-pn
  settling scans under 50 and the asset map wired into the search, Guard's
  re-search hook (off), the offline audit lane's tooling (H44). Redone in step 2.
- Regenerable caches only: `oracles/.cache/` (rebuilt: every oracle built,
  GHC 9.10.3 fetched by `build --bootstrap`) and its reports (the 2026-10-02
  triage and CHECKPOINT 1 evidence - regenerable by the commands that made them);
  `build/`. Anything else done after 3b33942 and not pushed cannot be recovered
  from the repository; the lost branches above are the ones known.

**Main verified on `dev-r` (2026-10-17, 3b33942):** the engine from clean with
`--werror` (MSYS2 g++ 16.2.0, the `.venv`'s CMake 4.4.3 and Ninja 1.13.2) - 526 of
526 tests; the network - `ruff check` and `ruff format --check` clean, pytest
**775 passed, 2 skipped** (the two by design: no `T5D_RECORDS`, and the anchored
global registers arm) in 8 min 26 s with `T5D_REQUIRE_ENGINE=1`; `verify` ok (its
17 `proof-channel-vacuous` notes are H48's, as before); `selftest` 623 tests, 1
skipped (symlinks cannot be created by this Windows profile; CI runs it); `gate
--tier commit` ok - ftxi 8256 of 8256 and notation 3012 of 3012 required values
reproduced; calibration 10.00 s against the recording machine's 8.47 s (1.18x).
**The comparison with the last green CI run waits on `gh auth login`** (the
repository is private; `gh` has no login on the rebuilt machine).

### Current state (2026-10-04)

**On main (CI green):** the rules engine (E1-E5c, E7: Q4 as ruled, D8's submit
action, H46), the search core (E6a, reviewed: PUCT/MCGS, both D2 arms, the solver
behind H41's `proof_channel` switch, off), verification v3's components (E8+E9,
reviewed: df-pn, Guard, Hunt, the asset map, certificates, Hunt's pinned ground
truth), the network (N1-N5: dense and cached paths, D8's acting-player input and
submit logit, the C step API from Python, end-to-end games with random weights),
the complexity trigger (C24, flagged off), the oracle harness with cwmtt/many2
read converted, and t0database (real games, measurement only).

**Measured verdicts:** M9 movegen passes on M13's 2 cores; 6b, 6b′ failed above
50 boards (verification v3 adopted, CHECKPOINT 4); 6b″: **Hunt adopted under 100
boards, Guard in no range** (M17); the window (M7) **killed, provisionally**
(M16), re-checked by the pilot self-play run (M18) after §15.5.

**Held on a branch:** the gate with the engine configured (branch
`worktree-agent-a1a2868a52f6c83b9`, 61e9aa8): 113 new known divergences (110 R1,
3 notation), no engine defect, nightly PASSED; the commit tier fails on
`engine-nothing-gated` and the release tier's oracle layer on 17
proof-channel-vacuous budgets - both put to the user (Q-A, Q-B below). Merging it
as it is would turn the commit workflow red.

**Running:** engine **E10** - Hunt (< 100 boards), the df-pn settling scans (< 50)
and the asset map wired into the search per the adoption table; Guard's re-search
hook (off: not adopted); the offline audit lane's tooling (H44).

**Remaining to §15.5, in order** (the audit of 2026-10-04):
1. E10's review and merge.
2. **E11 - time management** (E7, committed; with Guard's reserve) and **E8 behind
   a flag** (prototype).
3. Missing tests: the search solver's proofs against df-pn (and, through the gate
   or the audit lane, the oracles' mate verdicts - H41); TT churn for the prover
   (6f); cached = dense on more real variants.
4. Prototype flags not yet built: G2 (FP8 attention, emulated on CPU); A5
   (piece-centric tokenisation) unless ruled deferred with the window.
5. Quiet-machine measurements: N3's CPU wall timings (about 3 h; commands in
   step 2 below), §15.6's Phase-3 movegen re-check, Guard's and Hunt's nodes/s.
6. The gate merge, once Q-A/Q-B are answered.
7. Write-backs (M4) and register fixes proposed (edits 91-97 below); the
   committed-class placeholders consolidated for the user (spec issue 121).
8. Report for sign-off, the proof channel disabled.

**Waiting on the user** (asked 2026-10-04):
- **Q-A** - the commit tier gates no ruleset with ftxi alone, so
  `engine-nothing-gated` fails it: (a) apply that rule only in tiers that run every
  oracle (recommended); (b) Chess in 5D and cwmtt in the commit tier; (c) add
  records ftxi alone can gate.
- **Q-B** - raise the naive searches' release budgets and re-record (17
  proof-channel-vacuous findings; not blocking while H41's sign-off is pending).
- **GPU-only cost figures** (spec issue 120): accept counted FLOPs and
  CPU-labelled wall times as the untrained build's write-back, the GPU figures
  named as deferred - or provide GPU access.
- **The committed-class placeholders and B7** (spec issue 121): rule them, or name
  the trained ablation each awaits.
- **A5 (piece-centric tokenisation)**: build behind its flag, or defer it with the
  window (they are "tested jointly").
- **D8's whole-turn arm** has no end-of-turn token (issue 92): build one, or
  record the whole-turn arm as move-only until 6g/6k.
- **The training rows (I1-I10)**: out of scope for the untrained build (the
  brief's reading), or smoke-test code for some (e.g. I5, playout-cap
  randomisation)?
- Standing: branch protection with the workflows as required checks; filing the
  two oracle bug drafts (`docs/checkpoints/upstream/`); applying the proposed spec
  edits upstream.

### CHECKPOINT 5 - ruled 2026-10-03 (no stop): the window killed provisionally; Q2

The user's rulings (design documents updated: M16 corrected, M18, H46, reversal 24,
§13's size-premise entry):
- **The window is killed, provisionally (M16); the weighting is not changed.**
  Under 50 boards the per-phase rule decides, and a result inside the ±10% band
  resolves to off with no stop. The window code stays unbuilt; a **pilot
  self-play run** before serious self-play (M18) re-checks it - the first step
  after §15.5, not part of the untrained build.
- **The size premise was reasoning, not data** (reversal 24, §13): decisions
  weighted by where compute goes (M7, C2, piece-centric tokenisation) use measured
  game-length distributions - the human games now, the pilot run later.
- **Q2 (H46, provisional): initial timelines count as created by neither player** -
  always active, never counted. Built: a named variant's set-up reads every one of
  its timelines as initial (Timeline Battleground's -1 and +1 had been read as
  one creation each side - the same activity, since the count rule treats the
  sides alike, but not "counted toward neither"); recorded as rulebook R10, the
  rulebook re-pinned.
- Continue as planned: E9 and E6a reviews and merges, E10, then 6b″ on a quiet
  machine.

### CHECKPOINT 4 - ruled 2026-10-03: the verification layer is v3 (§11.4); the run resumed

The design review concluded (design documents committed e668555; register H1, H2,
H6, H10-H12, H18, M5, M14 changed; C24, C25, H41-H45, M16, M17 and reversal 23
added): exact rules always; the search's own solver for proofs, gated as part of
the proof channel (H41: game-end scoring stays on, solved-status propagation and
export off until the gate passes, the search's soundness fixes land with tests and
the proof channel is signed off); **Guard** (H42) and **Hunt** (H43) at every size;
full settling scans only under 50 boards; an offline audit lane (H44); the
adversarial league primary above 50 boards (H45). Retired: the per-leaf estimate
channel (rung 1), rungs 3 and 5; the NNUE held open (H6). The user's rulings:
1. E8: carry its df-pn and incremental attack maps into Hunt and Guard; drop the
   per-leaf estimate channel; review, then merge.
2. Guard (H42): before the chosen move is played, iterative deepening from depth 1
   under a fixed budget - can the opponent force mate against it? Depth 1 checks
   only that one legal escape exists per opponent check. A refutation rejects the
   move and triggers a re-search from §8.6's reserve.
3. Hunt (H43): df-pn in prove-only mode at the root - stops at budget and claims
   nothing; a found mate is a proof under H37 and H41's gating; its partial lines
   as soft priors behind a flag, with §12.2's exclusions.
4. The offline audit lane's tooling (H44): the decision sampler, the offline
   proof runner, the per-range report of how often the chosen move allows a
   shallow forced mate; curriculum hooks as interfaces only - no training.
5. Hunt's ground truth: one long offline search per 6b position, recorded once,
   pinned and hashed like the recorded answers.
6. 6b″ (M17): same positions, per range, 5 threads, 50 ms; adopt Guard and Hunt
   where each passes; record the verdicts. **No outcome stops the run.**
7. The complexity trigger (C24) behind a flag alongside the refresh ladder's rung
   1. Not built: C25, the NNUE (held open), the retired rungs (M5).
8. Continue in CHECKPOINT 3's order: E6a's fixes and merge (with the H18
   acting-player test), the many2 re-record, the gate with the engine configured,
   then N5. Step 6 of the brief's "scanner ladder" reads as the reduced ladder:
   Guard, Hunt and the asset map.
M16 also fixed M7's readings; applied, the window is killed under human-game
weighting, which contradicts M16's own premise - put to the user (MEASUREMENTS, M7).

**Done since (2026-10-03/04):** engine **E9 merged** (Guard, Hunt, the asset map,
Hunt's ground truth and 6b″'s tool; two adversarial reviews and their fixes - the
check filter's unsoundness among them); network **N5 merged** (D8 in the network, the
search's C step API from Python, end-to-end games with random weights; reviewed);
network C24 merged (65acf68); **the many2 conversion
merged** with the golden files re-recorded on a quiet machine (only many2's
answers changed); **engine E6a merged** -
the search core after its three adversarial reviews and their fixes, with H41's
switch and H18's acting-player test (474 tests). **Running:** engine **E9** on
E8's branch (Guard, Hunt, the asset map as components; the estimate channel
dropped; Hunt's ground truth; the 6b″ tool). **6b″ decided (2026-10-04, quiet machine, MEASUREMENTS):** Hunt adopted under 100
boards, Guard in no range; no stop (M17). **Then:** E10 - Guard, Hunt, the asset map and the under-50 settling
scans wired into the search, and the offline audit lane's tooling (H44); 6b″ on a
quiet machine; the many2 re-record; the gate with the engine configured; N5.

### The CHECKPOINT 4 stop (2026-10-03) - the record


The verification redesign (M14) was built as a prototype (engine E8, branch
`worktree-agent-a4f2f93a8a448467b`, bd08c48 - not merged: it waits on the design
review) and 6b′ run on a quiet machine with M15's tests unchanged
(MEASUREMENTS, 6b′): the df-pn proof channel decides 77.4% of roots under 50
boards, **40.7% at 50-99, 23.7% at 100-199 and 0.0% at 200+**, against a 50% bar;
the estimate channel fits the rung-1 leaf budget in no range. M15's pre-registered
outcome: **STOP - "exact verification covers only small positions, and the
design's robustness argument above them rests on the adversarial league alone -
a design review, not a tuning pass."** Rung 1 stays dropped; below 50 boards
df-pn is kept (408 roots decided against brute force's 400).

What the review has to work from (E8's report, MEASUREMENTS):
- df-pn beats brute force at 50-99 boards (40.7% against 30.7%) but not above;
  ten times the time (500 ms) reaches only 45% / 34% / 2% - the trees' size, not
  a constant factor, is the shortfall.
- What "decided" means turns on the attacker's class, which no document defines
  (spec issue 67): checking first moves with one canonical completion each
  decides 77/41/24/0%; the literal question (every first move, every completion)
  9/3/1/0%. Most "decided" roots under 50 boards are roots with no checking action
  at all (issue 68).
- df-pn's work is 82-86% attack-map queries; the estimate channel's p90 cost is
  2.3x / 6.3x / 15x / 51x its budget by range.

**Also waiting on the user:** M7's early phase (STOP band, three questions, below);
whether the search's in-tree solver is the "proof channel" H23 keeps disabled
until the gate passes (§11.1 item 2: "Inject solved bounds into the tree") - E6a's
fix round puts it behind a flag, off by default, until the user rules.

**Continuing meanwhile** (the user's CHECKPOINT 2 ruling: work that does not
depend on the verification layer continues; CHECKPOINT 3's unpause order): E6a's
fixes from its three reviews, then its merge; the many2 golden re-record and the
gate with the engine configured; then N5. E5c's two owed long tests passed on
main (ccdbdbd): the evasion probe 1,807 proofs (989 confirmed by the game tree),
Lazy SMP 2,006 proofs over 2-8 threads (1,072 confirmed), no false proof;
`long_scanner_proofs_under_every_alternative_ruleset` still running (over 2.5 h).

### CHECKPOINT 3 - stopped 2026-10-03 on 6b's re-run, ruled the same day; the run resumed

The run stopped under M12's pre-registered rule: after the bounded optimisation
pass (E5c), at the scanner's design budget (5 threads, Lazy SMP), the 50 ms root
scan resolved 5 forcing half-turns in 76.7% of positions under 50 boards but
31.4% at 50-99, 24.7% at 100-199 and 2.4% at 200+; rung 1 and quiet-leaf reach
failed everywhere (MEASUREMENTS, 6b). The user's rulings (design documents
committed 1e3a210: §11.1 "The redesign after 6b", §15.6's CHECKPOINT 3 block,
register M14, M15, H1, H6, H10, reversal 22):
- **The verification redesign (M14), a prototype until 6b′ decides it:** split
  the scanner by channel - exactness (turn completion, all evasions, royal-capture
  checks) only on lines that can end in a proof; the estimate channel a cheap,
  inexact local search without turn completion, its outputs soft priors. The proof
  channel: depth-limited df-pn over the semimove-decomposed turn tree with
  hypercuboid-pruned completion, answering proven / disproven / unknown (H37).
  Incremental royal-attack maps. Then **6b′ (M15)** - same positions, per range,
  5 threads, 50 ms - with its outcome rules applied: every range from 50 boards up
  passes -> adopt (below 50, keep whichever of brute force and df-pn decides more
  roots); 50-199 pass, 200+ fail -> phase-dependent (proofs only at the root and
  PV transitions above 200; the gap recorded in §13); **50-199 fail -> STOP** for
  a design review. Rung 1 stays dropped and the NNUE deferred until then.
- **Unpause, in this order:** merge E7 together with the Python binding's update;
  rebase E6a onto it (the expansion struct changed) and merge after review; run
  the gate with the engine configured; then N5.
- **Q2:** quote the question, the candidate readings and what each changes
  (below, "Waiting on the user").
- **cwmtt/many2:** "the cwmtt driver converts the export form to the canonical
  5dpgn form from the pinned notation spec; if it expresses something the spec
  can't, mark the record unsupported. The engine's parser is not extended."
  Carried out as: cwmtt's side of the harness converts that record's text, and
  every source - the engine included - reads the converted text (converting only
  what cwmtt reads would change nothing: cwmtt reads both forms). The goldens it
  changes are re-recorded on a quiet machine.
- **Game records: approved** - MageOfficial/t0database (MIT) downloaded, pinned
  and recorded (`data/t0database/`, 901e5be); the |E| curve re-run on it. It tops
  out around 186 boards: the window verdict above that stays provisional.
- **The Lazy SMP timing test** runs before any merge that touches the scanner, on
  a machine with 6+ hardware threads, each run logged ("The Lazy SMP timing test
  log", below).

**Done since the ruling (2026-10-03):** engine **E7 merged** with the Python
binding (Q4 as ruled is the default; D8's acting player and submit availability
in the expansion; `t5d_layout`, checked by the binding at load); the
**t0database analysis** (M7 measured on real games - under 50 boards in the STOP
band, waiting on the user; the 45 refused games classified, the 13 present-shift
ones unfinished final turns, no engine defect); the **many2 conversion** built and
reviewed on branch `worktree-agent-aa6c512c8bab6f3a8` (2a0b207), not merged - it
merges with the re-recorded golden files. **Running:** engine **E8**, the M14
prototype and the 6b′ tool. **Then:** rebase E6a onto main, review, merge; on a
quiet machine, the decisive 6b′ run (STOP if 50-199 fail) and the many2 merge with
`golden --write --repeat 2`; the gate with the engine configured; N5; N3's CPU
wall times. **Long tests owed on main:** E5c's
`long_scanner_evasion_probe_keeps_the_proofs` and
`long_scanner_lazy_smp_no_false_proofs` (not yet run on the merged code) and
`long_scanner_proofs_under_every_alternative_ruleset` (over 1.5 h under either
reading; never run to completion).

### CHECKPOINT 2 - stopped 2026-10-02 on 6b, ruled the same day; the run resumed

The run stopped on experiment 6b (the root scan failed above 50 boards; inside the
±10% band under one Q4 reading) and on M9 with legality (1.01x on one core). The
user's rulings (register C23, D8, M12, M13; rulebook R8, R9; committed b2bfab9,
39523c6):
- **6b (M12):** read per board-count range, never pooled - a real failure above 50
  boards. Before any redesign, one bounded optimisation pass on in-check
  enumeration and the horizon mate tests, then 6b re-run unchanged at the scanner's
  design thread budget (5 threads, Lazy SMP). Outcomes, fixed: every range from 50
  boards up passes -> as designed; 50-199 pass and 200+ fail -> report, then
  phase-dependent verification (df-pn the leading candidate above); 50-199 fail ->
  STOP. Rung 1 and quiet-leaf reach are re-judged on the same run; both stay parked
  until then.
- **CPU (M13):** 8 cores per GPU - 2 expansion, 1 search and batching, 5 scanner.
  M9 passes on the 2 expansion cores (1.87x at 240 boards with fast legality).
- **Q4 (R8, D8): yes** - a player may keep moving after the present has shifted;
  the turn ends at submission. Submitting is an action: under the ply split its
  own policy logit from a global register, legal only when the submission passes
  C4; plus a global acting-player input (issue 48).
- **Causal past (C23):** the board's ancestry - the `lineage` reading.
- **Q5 (R9):** settled from the rulebook's ranked sources - only the active boards
  in the present are mandatory. **Q2: its sources are silent** - quoted to the user.
- **Real game records:** sources to propose with provenance; the user approves any
  download; the window verdict (M7) stays provisional until then.
- **cwmtt/many2's export-form move:** "the engine does not read it: normalise it in
  the cwmtt driver" - asked back (see "Waiting on the user").
- **E6a:** its interrupted self-play edit was diffed against its last commit,
  reviewed and completed (1c6956d in its worktree).

**Running (2026-10-02):** engine E7 (Q4 as the default; D8's acting player and
submit availability in the expansion and the C API; the corpus re-timed under the
new default), engine E5c (the bounded optimisation pass and Lazy SMP at 5 threads;
`t5d-scan6b` per range with M12's outcome rules), and the game-record source
research. **Then:** the quiet-machine 6b re-run and M12's outcome; the gate with
the engine configured (the triage re-run under Q4 = yes: the 94 R1 entries and the
other drafts that Q4's other reading needed); network N5 (D8's submit logit and
acting-player input; C23 fixed); E6a resumed (submit as an action, the MCGS key
with branch origins, C23); N3's CPU wall times.

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
  full corpus in 8.3 s); step 2B - the gate's stable answers, integrity,
  telltales, parse mode and rulebook pin (committed 2026-10-02 with the golden
  files re-recorded twice, schema 3).
- **(2026-10-02's "uncommitted", "next" and "waiting" lists are superseded by
  "Current state" above.)** N3's CPU wall timings still wait for a quiet machine:
  from `python/`, `py -m t5d.net.cache_bench --boards 240 --idle-processes
  t5d-gate.exe t5d-measure.exe t5d-bench.exe t5d-tests-long.exe --skip-existing
  --out <dir>`, then `--boards 600 --dtype fp32` and `--boards 600 --dtype bf16`
  (about 45, 70 and 45 minutes); the benchmark refuses a loaded machine.

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
- **The Lazy SMP timing test** (`scanner_lazy_smp_reaches_the_single_threads_depth`,
  which skips under 6 hardware threads): run it before any merge that touches
  the scanner (`engine/src/scanner*`, `search*`, `include/t5d/scanner.hpp`,
  `search.hpp`, or their tests), on a machine with 6+ hardware threads, and log
  every run in "The Lazy SMP timing test log" (the user, CHECKPOINT 3).
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

## The Lazy SMP timing test log

Each run of `scanner_lazy_smp_reaches_the_single_threads_depth` before a merge
that touches the scanner: random Standard and Two Timelines positions (the count
is printed) at 40 ms each, resolved depth summed over them
at 1 and at 5 threads (a proof or an exhausted tree counts 17); it passes when 5
threads reach at least one thread's sum (one retry on a shared machine).

| Date | Merge | Commit tested | Machine | Positions | 1 thread | 5 threads | Retry | Result |
|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | E7 | 2e7ee18 (E7's branch, main merged in) | dev, 24 logical; E8 running; load 5-12% before, 12-26% during | 32 | 293 | 311 / 313 / 309 (three runs) | none | pass x3 |
| 2026-10-03 | E7 | main + E7, merged tree before the commit | dev, 24 logical; E8 running; load 23% | 32 | 293 | 311 / 311 / 313 (three runs) | none | pass x3 |
| 2026-10-03 | E6a | ef3830f (E6a's rebased branch, before review) | dev, 24 logical; shared | 32 | 293 | 311 / 313 / 313 | none | pass x3 |
| 2026-10-03 | E6a | 9c49a47 (E6a after its review fixes) | dev, 24 logical; load 79-88% | 32 | 292 | 304 / 307 / 308 | none | pass x3 |
| 2026-10-03 | E6a | main + E6a, merged tree before the commit | dev, 24 logical; E9 running; load 8% | 32 | 293 | 313 / 313 / 313 | none | pass x3 |
| 2026-10-03 | E9 | 4ab7f81 (E8+E9's branch, before review) | dev, 24 logical; shared | 32 | 293 | 313 / 314 / 311 | none | pass x3 |
| 2026-10-04 | E9 | 9328e31 (after its review fixes) | dev, 24 logical; shared | 32 | 293 | 311 / 312 / 313 | none | pass x3 |
| 2026-10-04 | E9 | main + E9, merged tree before the commit | dev, 24 logical; nothing else running | 32 | 293 | 311 / 311 / 313 | none | pass x3 |

## Environment

- Development machine: Windows 11, AMD Ryzen 9 9900X (12 cores / 24 threads,
  AVX-512 incl. BF16), 32 GB RAM, AMD Radeon RX 9070 XT. **No CUDA**, so the
  network runs on CPU PyTorch (2.13.0+cpu) here; every GPU figure in §15.3 is
  measured on CPU and labelled so.
- **Rebuilt 2026-10-17 after the lost machine** (same hardware; a fresh Windows
  profile; machine row `dev-r` in `docs/MEASUREMENTS.md` - figures before the loss
  are `dev`): Windows 11 Pro 10.0.26300; **MSYS2 ucrt64 g++ 16.2.0** (Rev3 - the
  MSYS2 tree survived the loss), put first on PATH for every build
  (`$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH`, or `CXX`): a WinLibs
  GCC 16.2.0 is also installed and comes first on the user's PATH; **Python
  3.14.7** (`py`; 3.12.10 is also installed); the project `.venv` on 3.14.7
  (`--system-site-packages`) with torch 2.13.0+cpu, numpy 2.5.2, pytest, ruff
  0.16.5, cmake 4.4.3 and ninja 1.13.2 (the versions CI and the harness pin);
  Node 24.20.0 / npm 11.19.0; git 2.55; gh 2.102. Rust is installed but unused; no
  MSVC. Commits are made as the repository's existing author, set repo-locally.
- **Rebuilding from nothing** (what was done on 2026-10-17, in order): install
  git, Python 3.14, Node 24 and MSYS2 with `pacman -S mingw-w64-ucrt-x86_64-gcc`;
  clone, then `py oracles/harness.py sync`; `py -3.14 -m venv
  --system-site-packages .venv`; `.venv\Scripts\python -m pip install
  --index-url https://download.pytorch.org/whl/cpu torch==2.13.0` and `pip install
  numpy==2.5.2 pytest ruff==0.16.5 cmake==4.4.3 ninja==1.13.2`; `py
  oracles/harness.py build --bootstrap` (fetches GHC 9.10.3 into the cache); then
  the checks: `py engine/build.py --clean --werror --test`, from `python/` `ruff
  check .`, `ruff format --check .` and `T5D_REQUIRE_ENGINE=1 python -m pytest -q`,
  `py oracles/harness.py verify`, `selftest` and `gate --tier commit`.
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

### 2. Finish the gate - 2A and 2B done; the engine configured and triaged (2026-10-17)

**The gate with the engine configured (2026-10-17,** branch `lane/gate`; the work
lost with the old machine, 61e9aa8, redone under H47). `oracles/gate.toml`'s
`[engine] command` is `["./build/engine/t5d-gate"]`, full mode: the adapter as
`engine/build.py` builds it, relative to the repository root, no diagnostic option;
build it before a gate run and after any engine change - the gate runs whatever
build is there (`oracles/README.md`'s and `CONTRIBUTING.md`'s command lists say so;
`AGENTS.md`'s and `CLAUDE.md`'s, the agents' instructions, are the user's to change,
and do not yet). The harness takes a relative
program from the repository root on every platform (`gate.engine_argv`, a review
fix): Windows had looked for it from the directory the harness was started in, so
a lane's harness started from the main checkout ran main's build, and the
engine-in-harness check looked at another file than the one that ran. A run's
report names the file it ran and its SHA-256 (`engine_run`). The engine has its own
time limit, `[engine] timeout_s = 300` per record (the tier's own `timeout_s` where
longer; a review fix): CI's first commit-tier run with the engine (below) took
44.4 s on cwmtt/many2 on a runner that ran the calibration probe at 0.75x, the
runners range from 0.59x to 1.09x (eleven commit-tier runs, 2026-10-17), and the
commit tier's 60 s, which the engine had shared with the oracles, would not cover
the slower ones. It changes no oracle budget or required set. The engine is
main's (unchanged since 3b33942). Compared with every recorded oracle answer by
the harness's own comparison (a scratch script, not kept: `observations.compare` of
the engine's report on each record with all six golden files - what a release run
would compare on the recording machine, see "Runs" below), it disagrees at 169 places: the 56 entries of 2026-10-02 all
match (KD-0040 to KD-0043 too, now that R8 reads notation/1.2 to its end), and 113
are new - the lost branch's count: 72 ftxi and 38 ftxi:naive check verdicts at 72
positions (`in_check: engine=False vs ftxi=True`), and cwmtt/futures action 17
against cwmtt, Chess in 5D and the notation parser. A nightly run before the new
entries found 106 of them (its 120 s ftxi:naive budget stops short of niceAndrey
90-97, which only the release tier's 300 s reaches).

- **The check findings, by cluster (H31), settled by R1.** At each of the 72
  positions the engine's own royal-capture scan (`for_each_royal_capture`,
  `t5d/check.hpp`, over the multiverse with a passed copy of every playable board of
  the side to move - a scratch program linked to the engine's library, not kept)
  lists exactly the captures ftxi reports (`observe --source ftxi --explain`), 435 in
  all, and at the 38 ftxi:naive positions exactly those ftxi:naive reports; the
  engine's check status (R1's pass) finds none. Every capture starts on a board of
  the side to move's that R1 does not pass: an active board ahead of the present
  (RQ1, and RQ2 where it would branch into an inactive timeline) for all but four -
  100_timelines@13's (inactive, behind the present: CHK-I1, RQ3) and three of
  100_timelines@20's six (inactive, ahead; the position is decided by its active
  captures, as CHECKPOINT 1 found). Chess in 5D answers False at all 72 (support,
  never a vote). The 58 positions of CHECKPOINT 1's table re-derive exactly: side to
  move, present, mandatory boards, cluster, deciding capture, number of captures.
  cwmtt/many2's 14 (12, 14, 19, 21, 25, 31, 33, 36, 37, 39, 47, 49, 56, 60; ftxi:naive
  at 12 and 14), read converted since CHECKPOINT 3 and so not in the table, file as
  CHK-A1 by the report's method: all 253 of their captures start on active boards
  ahead of the present. Entries by cluster: CHK-A1 64, CHK-A3 31 (new text: RQ2),
  CHK-A5 10 (new text: RQ1 on both boards), CHK-A2 2, CHK-I1 2, CHK-A4 1. Five
  positions rest on R2 (Q1) as well, and their six entries say so: wide@24, 28, 30,
  32 (CHECKPOINT 1's own finding: under Q1's "consecutive" reading the present would
  be T7w and the position check) and 100_timelines@13 (`t5d-gate --rule
  activation=consecutive`, a consecutive reading with no count limit, makes +3
  active and the position check). Under that diagnostic reading the engine also
  refuses cwmtt/wide from action 23, niceAndrey from 30, many2 from 20 and NP and
  NP0 from 62: these records are legal games only under R2's count rule. **No engine
  defect.**
- **The notation finding.** cwmtt/futures action 17, `(L1T5)Qf4>>x(L0T5)e4~`: at
  position 16 the engine, ftxi, cwmtt and Chess in 5D hold one state, and (0T5)e4 is
  empty - the `x` marks a capture the jump does not make. The engine refuses the move
  as unreadable (its reader: a written `x` must capture), and ftxi refuses it too (no
  matching move), so there is no divergence with ftxi; cwmtt, Chess in 5D and the
  notation parser play it. KD-0167 to KD-0169 cite the specification's "Jumps"
  syntax ("`x` if a piece is being taken"); what a reader does with an `x` that
  disagrees, the specification does not say (spec issue 125). Standard - Turn Zero
  stays ungated: the engine cannot read the record to its end. **The three entries
  are provisional** (a review finding): they follow the engine reader's strict
  reading, not a settled text, and stay only on the user's ruling on issue 125 -
  marked so in their notes and in the file's header, which says how they go if the
  ruling is the other way. Without them the commit tier fails on the notation
  parser's playing action 17 (checked on cwmtt/futures alone: `[divergence]
  cwmtt/futures action 17: legality: engine=rejected:unreadable vs notation=accepted
  (soft: may be notation, not rules)`), and nightly on cwmtt's, Chess in 5D's and
  the notation parser's.
- **Runs** (this machine shared with other lanes: the timings are not measurements).
  `py engine/build.py --werror` (MSYS2 g++ 16.2.0) clean. `verify` ok - its 17
  `proof-channel-vacuous` notes are H48's. `selftest`: 633 tests, OK, 1 skipped.
  `gate --tier commit`: oracle layer ok (ftxi 8256 of 8256 and notation 3012 of 3012
  required values reproduced); `engine differential: SMOKE TEST - PASSED`, every
  ruleset NOT gated and none meeting the gating conditions with ftxi alone;
  calibration 1.25x the recording machine. `gate --tier nightly`: oracle layer ok,
  every source's required set reproduced; `engine differential: PASSED`, no finding;
  **gated: Standard - Half Reflected** (21 positions, at least 2 oracles behind each
  record) **and Standard - Reversed Royalty** (122 positions, at least 3);
  calibration 1.19x. NOT gated, and why: Standard (the engine cannot read
  ftxi/softmate, KD-0047), Turn Zero (cwmtt/futures, KD-0167 to KD-0169), Very
  Small - Open (cwmtt/ctp1, ftxi/verysmall), custom positions (ftxi/niceAndrey's
  set-up), Misc - Small (no oracle follows notation/game), the Focused variants and
  Timeline Battleground (no oracle gives a check verdict there), Timeline Invasion
  and Marauders (the engine does not set them up). The release tier was not run: its
  oracle layer fails on H48's notes until the re-recording. The comparison with
  every recorded value (above) covers what a release run would compare of the engine
  on the recording machine or a slower one - not on a faster one, which a CI runner
  can be (0.59x): there ftxi:naive's 300 s may reach past its recorded horizons
  (100_timelines 19, exiledKings 18, many2 16, niceAndrey 101, wide 11), where ftxi
  says check at R1 positions the engine answers False (100_timelines 20, exiledKings
  18, many2 19-60, niceAndrey 101-118, wide 12-46), and no entry can record those
  until H48's re-recording puts them in the golden file (`divergence-golden`). The
  proof channel stays disabled. After the review fixes (the same machine, as loaded):
  `py engine/build.py --werror` clean; `verify` ok (H48's 17 notes); `selftest` 638
  tests OK, 1 skipped; `gate --tier commit` ok - oracle layer ok (ftxi 8256 of 8256,
  notation 3012 of 3012), `engine differential: SMOKE TEST - PASSED`, the NOT gated
  reasons as before, calibration 1.33x; its log names the build it ran (the
  worktree's `t5d-gate.exe`, SHA-256 71a9d6a9...), and a one-record run started in
  another directory ran the same build (before the fix, Windows would have found
  none there); `gate --tier nightly` ok - every source's required set reproduced,
  `engine differential: PASSED`, gated Standard - Half Reflected and Standard -
  Reversed Royalty (2 and 3 oracles), calibration 1.20x. The engine's slowest
  records under the new limit: many2 28.7 s (commit) and 25.5 s (nightly).
- **CI.** The commit workflow ran on the branch at 3a46da9 with the engine compared
  (run 39009070712, ubuntu-24.04, success): selftest 633 tests OK (16 skipped), the
  engine built, `engine differential: SMOKE TEST - PASSED`, calibration 0.75x;
  cwmtt/many2 took the engine 44.4 s and niceAndrey 31.6 s. The nightly workflow has
  not yet run with the engine: before the merge, dispatch it on the branch (`gh
  workflow run oracle-nightly.yml -f tier=nightly --ref lane/gate`) and record its
  result here.

**The engine's first gate runs (2026-10-02,** engine at 0a44cbf / 4ed2e92;
reports in `oracles/.cache/reports/triage-2026-10-02/`, regenerable). With
`[engine] command = ["./build/engine/t5d-gate"]` set locally, every tier ran with
every required oracle answer reproduced. The nightly differential's 170 findings,
triaged: 37 are ftxi's and ftxi:naive's check verdicts that ruling R1 rules wrong
(20 positions, all among CHECKPOINT 1's 78); 2 are Chess in 5D's false checkmates
(the engine lists the same 25 and 21 legal actions as ftxi); 10 are records the
engine refuses as unreadable (7 settled by the notation spec - `K>>` and `N>>d3`
jumps without a source square, a parenthesised variation, 5DFEN boards under a
named Board - and 3 borderline: notation/aivsai's repeated turn number, now read
with a warning (4ed2e92), and cwmtt/many2's export-form move); 115 are rules Q4 -
every one disappears under `--moves-after-shift`; 6 are oracle notation errors
(ftxi drops Very Small - Open's unmoved markers; ftxi cannot read `+1` timeline
labels; the notation repository's parser cannot read `(0T1)a2a3`). **No engine
defect was found.** 56 entries are committed (37 R1, 2 + 4 telltales of the two
oracle defects - the double-step ones inert until Q4's other reading reaches
notation/1.2 position 15 - 7 notation, 6 oracle notation errors). With them the
nightly gate matches all reachable entries, none stale, no telltale; what remains
is the 115 Q4 findings and many2. **Gated: Standard - Half Reflected** (21
positions, check, counts and mate, at least two oracles each). Never gateable as
the corpus stands: custom positions (ftxi/niceAndrey's set-up is unreadable),
Focused - Just Brawns (no oracle gives check there), Misc - Small (no oracle
follows notation/game), Timeline Invasion and Marauders (no independent set-up).


**Step 2B (committed 2026-10-02).** H32: every source recorded twice
(`golden --write --repeat 2`); a value that differs between recordings is
unstable and compared by nothing; one a limit stopped is "unknown at budget X";
per-position costs in `oracles/golden/<source>.cost.json`; a tier requires
exactly the stable values whose recorded cost x `[stability] margin` (4) fits
its budget, and requires 100% of them - the coverage floors are gone; a
calibration probe says how much slower a runner is than the recording machine.
H33: `oracles/golden/MANIFEST` (SHA-256 of every recorded file, written by
`golden --write`, verified before any comparison); one verified snapshot per run;
the harness's files re-hashed after the engine runs (a change fails the run);
the harness imports its modules from source only; CODEOWNERS over `/oracles/`
and `/.github/` (binding once branch protection requires code-owner review).
H34: an engine answer that agrees, by the comparison's own rule, with a known
divergence's recorded oracle error is a telltale and fails the gate. The
engine's parse mode (`[engine] mode = "parse"`): states, replay length and
acceptance compared, never check, counts or mate. `rejected`/`unreadable` from
the engine, in either mode, is a soft legality divergence excusable only by a
notation citation, and keeps the ruleset ungated. `legal_actions` is defined:
an action is its set of moves with its timeline-creating moves in creation
order (ftxi and cwmtt list exactly that at every position both read). H36's
`in_check`; H37's guard (a source that names a limit and still gives a verdict
is malformed); H38's rulebook pin (`[rulebook]` in `gate.toml`,
`harness.py rulebook [--pin]`). The proof channel stays shut: the naive
searches' required sets fall short of `proof_share` (verify's
`proof-channel-vacuous`) until their release budgets are raised.

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
- **The legal moves in tiers (engine E5a, merged 2026-10-02):** `t5d/legal.hpp`
  decides a move's legality (C4, R1) by sound tiers - a standing royal capture;
  a "stripped multiverse" that certifies quiet moves and branches in bulk; a
  passed board's standing capture refuting moves; per-move checks; and search
  last - in exact, budgeted (legal / illegal / unknown, never illegal without
  proof) or fast mode. `ExpandOptions::legality` sets the expansion's mode
  (unknown moves kept and marked). 0 disagreements with the brute-force
  reference and a per-move search in about 113,000 long-run classifications. An
  independent review (1.1 million exact classifications, 20 rulesets) found one
  unsound case, fixed the same day: an en passant that uncovers a capture of the
  mover's royal was treated as a quiet move. At 240 boards the median expansion is
  26 µs pseudo-legal, 102 µs fast, 2.5 ms budgeted, 7.3 ms exact (E4's filter:
  157 ms); exact keeps a tail of minutes at 400-600 boards.

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

**The scanner and 6b - plan** (after M9: M10's order; and after E4, done):
- *Scanner* (H1, H3, H4, H5, H6 v1): node-capped (not depth-capped) alpha-beta over
  forcing moves - captures including cross-board and time-travel captures, checks
  including across timelines, branching moves that create a timeline with an
  immediate threat ("immediate threat" is undefined: a parameter and a spec
  issue). Material-count eval (5D piece values are a project parameter table -
  the spec gives none). Net-free ordering: MVV-LVA, killers, history,
  continuation history. TT keyed by the 5D Zobrist key with the ruleset ID (and
  branch origins if the lineage reading is adopted, spec issue 30). In check,
  ALL legal actions are generated (E4's enumeration) - that is what makes a mate
  found a proof (H2); a proof also needs the subtree searched exhaustively within
  the horizon, and a node cap reached inside a proof line yields an estimate,
  never a proof (H37's spirit). No speculative pruning on proof lines;
  ProbCut/futility estimate-only; LMR banned on evasions; null-move not built
  (it waits on the zugzwang answer). Depth counted in forcing half-turns (one
  side's whole action of forcing moves), board-move depth reported beside it.
  Synchronous debug mode (H15). Lazy SMP (H17) waits for Phase 2.
- *6b tool*: over turn-start positions of real records (verdicts) and synthetic
  ones (labelled): the horizon a node budget buys (deepest d fully resolved);
  rung 1's leaf budget = threads x measured scanner nodes/s / evaluation rate (the
  measured rate if one exists, else §15.6's ~9,600 nodes/s at 240 boards and
  ~4,600 at 600 - stated); the 50 ms root scan; quiet-leaf reach; stand-pat
  validity under 5D move obligations, operationalised before measuring (proposed:
  at a quiet position stand-pat is invalid if every legal action of the side to
  move loses material to the opponent's best forcing reply within a short fixed
  check). M8's thresholds, with the brief's ±10% STOP band; a failed root scan
  STOPS the run.

### 6. Phases 2-3 - in progress (see "Current state")

## Register status

### H - rules engine, gate, verification layer

| ID | Status | Notes |
|---|---|---|
| H1 | built (v3), tested | Verification v3 (§11.4): exact rules (E1-E4); the search's solver behind H41's switch (E6a); Guard, Hunt and the asset map (E9, reviewed); df-pn settling under 50 boards (E8's prover, carried into E9); the audit lane's tools (certificates, the truth recorder; E10 wires them). E5b/E5c's brute-force scanner kept as 6b′'s comparison. 6b″ decides adoption per range |
| H2 | built, tested | Proofs only from exhaustive verdicts and exhaustive defender enumerations; the per-leaf estimate channel removed (E9); estimates survive only as Hunt's flagged priors and the asset map's boosts; the search consumes in-tree bounds only behind H41's switch |
| H3 | built, tested | Node cap with iterative deepening in forcing half-turns; horizon per budget measured (6b) |
| H4 | built | MVV-LVA, killers, history, continuation history |
| H5 | built | Nothing speculative; null move and LMR not built |
| H6 | built (v1 material); NNUE held open | Material count with placeholder 5D piece values (spec issue 56), used by the asset map and the material evaluator; the scanner NNUE is held open (CHECKPOINT 4) |
| H7 | not built (rule) | NNUE training: held open with the NNUE (H6); no training runs |
| H8, H9, H20, H21 | not built (rule) | Held open / refused |
| H10 | reduced, built | Rung 1 dropped (6b); rungs 3 and 5 retired, not built (M5); rung 2's promotion side is Guard, its demotion audit the audit lane; rung 4, the asset map, built (E9) - the reduced ladder: Guard, Hunt, the asset map |
| H11 | superseded | By Guard (promotion) and the audit lane (demotion audit) |
| H12 | retired | With the NNUE and the per-node scans |
| H13, H14 | building (E10), reduced | Ladder scheduling for the reduced ladder (Guard, Hunt, the asset map) against shared movegen: E10. H14's rung-1 wording is obsolete (proposed edit 95) |
| H41 | built, tested | `SearchParams::proof_channel` (C `proof_channel`, `t5d-play --proof-channel`): propagation of solved status (marking nodes solved, pruning siblings, secure-child selection) and export, off by rule; game ends are scored as results. Was: building | The search's solver as part of the proof channel: game-end scoring on; propagation and export off until the gate passes (E6a's fix round) |
| H42 | built, tested; not adopted | Guard (E9): depth 1 completes in 65.1 / 15.8 / 5.2 / 0.0% of roots against 95% (6b″) - adopted in no range; §13: the chosen move goes unguarded. Built and kept (the audit lane's shallow-mate report uses its question offline; the search's re-search hook stays unwired-by-adoption) |
| H43 | built, tested; adopted under 100 boards | Hunt (E9): 69.0% / 63.6% of the truth's mates under 50 / 50-99 boards (6b″) - adopted there; 35.0% at 100-199 and 0 of 1 at 200+ - not adopted; §13: proofs there come only from the search's solver. Export refused until H41/H23 |
| H44 | built in part | The audit lane's tools: the certificate checker (E4 alone) and `t5d-hunt-truth` (E9); the decision sampler, the offline proof runner and the per-range report are E10's |
| H45 | not built (rule) | The adversarial league (M5) |
| H47 | built, tested | The check that some ruleset is gated (`engine-nothing-gated`, and `GateResult.ok`'s test of it) runs only in a tier that runs every oracle search over the whole corpus (`Tier.full_strength`: nightly, release) - keyed to what a tier runs, never its name (the self-tests pin both directions). Any other tier - commit - is a smoke test: an engine with no finding against it passes, gated or not, and the summary (`engine differential: SMOKE TEST - ...`) and the report (`smoke_test`; nothing under `gated`) say that it gates nothing and why. In every tier an unexplained divergence, a telltale or a contract breach fails the engine. `gate.toml` is refused if a tier that is not full strength sets `require_engine` (it could never pass); a tier that requires an engine reads only what its run may state as gated, and the summary says when it came through no named variant. A run restricted with `--source`/`--record` keeps its tier's rule (issue 124). The release tier's named-variant rule, the proof-channel line and parse mode are unchanged. Run with the engine configured since 2026-10-17: the commit tier's differential a smoke test, PASSED, with no ruleset meeting the gating conditions with ftxi alone; nightly PASSED, gating two (step 2). Spec issues 122-124, proposed edit 98 |
| H48 | not started | Before the proof-channel sign-off: the naive searches' release budgets raised and the golden files re-recorded twice, on a quiet machine, so that no proof-channel check is vacuous (`verify`'s 17 `proof-channel-vacuous` notes) |
| H15 | built, tested | The scanner's deterministic single-thread mode; the search's synchronous mode and digest (E6a; the digest in the C API) |
| H16, H25, H29 | not built (rule) | Refused |
| H17 | built in part | Lazy SMP in the scanner (E5c): same-root threads, a shared TT and a concurrent solved store; in-game decay and rung helpers wait for the search |
| H18 | built, tested | 5D Zobrist: coordinate-free content hash per board, multiply-rotate with (timeline, half-turn), addition across boards, ruleset ID, player to move, parity, creators, board size; incremental equals from-scratch; duplicated boards never cancel (engine E1, E2) |
| H19 | built, tested | Proofs export to the solved store at once; the TT holds estimates only |
| H22 | built, tested | All five items of §11.0 (E1, E2, E4); the register row's wording predates H36/H40 (proposed spec edit 40) |
| H23 | built, tested | The gate, oracle and engine sides. The engine is configured in `gate.toml` (`./build/engine/t5d-gate`, full mode, 2026-10-17; the root's build on every platform, with a time limit of its own, `[engine] timeout_s`) and compared in every tier: commit, a smoke test (H47), PASSED, locally and in CI (run 39009070712); nightly PASSED locally, gating Standard - Half Reflected and Standard - Reversed Royalty - not yet run in CI with the engine (step 2). The release tier, the one that can enable the proof channel, is not yet run with it: its oracle layer fails on H48's 17 notes until the re-recording. The proof channel stays disabled (H23, H41) |
| H24, H26, H28 | built, tested | The generator and the 5dpgn/5DFEN parser are in-house (`engine/`), written from `docs/rules/README.md` and the notation spec's text; 259 engine tests; boundary clean |
| H27 | built, tested | The oracle harness, committed 2026-10-01 |
| H30 | built, tested | Step 2A (a04678c): every oracle reports its boards per ply; no placement divergence across the corpus |
| H31 | done | CHECKPOINT 1: seven clusters triaged (903f4ae), settled by the user's rulings (H36-H40). Re-run 2026-10-17 with the engine configured: the 110 new check findings (72 positions) filed by cluster and settled by R1 (and R2 at five positions); the 58 positions of CHECKPOINT 1's table re-derive exactly, cwmtt/many2's 14 file as CHK-A1 (step 2) |
| H32 | built, tested | Required sets at 100%, two recordings, cost files, calibration (2B) |
| H33 | built, tested | MANIFEST, verified snapshot, harness-modified check, CODEOWNERS (2B) |
| H34 | built, tested | Telltales by the comparison's rule (2B); 169 entries: 56 of 2026-10-02 (the two oracle defects among them, R7) and 113 of 2026-10-17 (110 R1, 3 notation - the 3 provisional, pending spec issue 125) - every disagreement of the engine's with a recorded oracle answer is one, and every entry matches one; no telltale |
| H35 | done | Commit and nightly tiers green on Linux, 2026-10-01 (runs 36827435595, 36827459456) |
| H36 | built, tested | Engine E4: check status and submission legality per R1; compared with the oracles by the gate since 2026-10-17: every check disagreement is at a position where the oracle's every capture starts on a board R1 does not pass (CHECKPOINT 1's RQ1-RQ4), each on record with R1's citation; no engine defect |
| H37 | built, tested | Engine E4: verdicts only from exhaustive searches; node, time and memory budgets answer "unknown" (tested for every budget below the need) |
| H38 | done | `docs/rules/README.md` is the rulebook (rulings R1-R7), pinned in `gate.toml`; `verify` fails when it changes until re-pinned |
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
| C10 | not built (rule) | Lazy K/V for boards entering E: with the window, killed provisionally (M16) |
| C3, C3a (rung 1) | built, tested | Root refresh and the travel-triggered, visit-gated rung-1 fork (copy-on-write epochs), behind flags |
| C3a (rungs 2-3), C18, C22 | not built (rule) | Held open (M5 for rungs 2-3) |
| C5, C7 | not built (rule) | The window: killed, provisionally (M16, CHECKPOINT 5); re-checked by the pilot self-play run (M18) |
| C6 | step 1 done | The |E| curve measured (M7); the head-to-head is Ablation 1, halted with the window until the pilot run (M18) |
| C8 | honoured | C24's add-only rule (and rung 1's) - refreshes are only ever added |
| C9 | to check | Reported built in `expand.cpp` by the 2026-10-04 audit; to verify against its row |
| C11 | built behind a flag, tested | MCGS (E6a): identity is the state key and whether an action is in progress, compared exactly (§8.5's state, issue 87); awaiting 6k |
| C12, C13 | built, tested | Re-rooting with garbage collection, the persistence hook and pondering (E6a) |
| C14 | built, tested | Fresh searches per decision in self-play; the evaluation cache shared across them, purged by board count (issue 89); the C API's `t5d_search_new_sharing`; from Python (N5) only between searches with the same network input, with the cached path's refresh epochs set on the search |
| C23 | built, tested | The node key's default carries branch origins (lineage, `NodeKey::StateAndOrigins`); the network's causal past is the ruled ancestry (`causal_past="lineage"`, tested against ancestry computed from the branching record, N5) |
| C15 | built in part | FP8 e4m3 frozen K/V (`frozen_kv_format = "fp8"`, applied by the forward itself); varlen/ragged kernels not built (none on CPU); C15 vs G1 is spec issue 34 |
| C16 | not built (rule) | Deferred; last, only if node-starved |
| C21 | not built (rule) | Refused; a mutant that snapshots is caught by the tests |
| D1, D3, D4 | built, tested | Network N1: pointer policy, 51-bin value head, auxiliary heads (D4's reply space, label set and moves-left unit are placeholders, spec issue 9) |
| D5 | not started (rule) | A training principle |
| D6 | built (flag) | `board_reduction`, awaiting 6j |
| D7 | not built | 6p's aggregate initialisation (training-side); D8's submit logit reads a global register, in tension with D7's "registers stay free scratch" (spec issue 109) |
| D2 | built, tested | Both arms (`SearchParams::split`: the board-move split, default, and whole turns), driven from Python through the C step API with the network (N5); awaiting trained ablations 6g/6k |
| D8 | built, tested | Engine and search (E7, E6a): submit an explicit edge, legal only when C4 holds, never automatic. Network (N5): a global acting-player input, the submit logit from global register 0 (placeholders, issues 108-109), one distribution over moves + submit, values from the acting player's perspective |
| E4 | principle | "Narrow hard, because the auditor is watching": the auditor above 50 boards is now Hunt (< 100) and the solver, Guard adopted nowhere (6b″) - the premise is weaker there |
| E1, E5, E9 | built, tested | PUCT with entropy-adaptive top-k widening, FPU, root noise in self-play (E1); LCB with a visit floor (E5); the solver behind H41's switch, the secure child, a visit cap on solved children (E9) - every constant a placeholder (issues 80-86) |
| E7 | not started | Time management (with E10's Guard re-search reserve) |
| E2, E3 | built behind flags, tested | Off; awaiting 6m and 6t |
| E8 | not started | Behind a flag |
| E6 | not built (rule) | Refused |
| F1, F2 | built, tested | 21,359,370 parameters (F1, with N5's D8 embedding and readout); the L24-ff12 preset, 44.5M (F2) |
| F3 | not built (rule) | Refused |
| G7 | not started | SPSA/SPRT tooling (training-time) |
| F4 | not built (rule) | The NNUE is held open (H6, CHECKPOINT 4) |
| G1 | built, tested | BF16 trunk under autocast; FP32 heads and RoPE; the cache stores K/V in the live format (§15.2 row (e)) |
| G2 | not started | FP8 attention - no FP8 kernels on this machine |
| G3, G4, G5, G6 | not built (rule) | Deferred / held open / refused |
| I1-I8, I10 | not started (rule) | Training: no training runs (the brief); I7 and I8 not built (M5); whether any is smoke-tested in the untrained build is put to the user |
| I9 | built, tested | The ruleset ID in every state and store key (H18): the Zobrist key, the solved and disproof stores, the evaluation cache |
| I11 | built, tested | Self-play's turn cap from the state (and t5d-play's 300-board cap), scored as a draw, its frequency logged (issue 88) |

### M - the plan itself

| ID | Status | Notes |
|---|---|---|
| M14 | superseded | By v3 (H1): its df-pn and incremental attack maps carried into Guard and Hunt (E9); the per-leaf estimate channel removed |
| M15 | measured - STOP | 6b′ on a quiet machine: (A) 77.4% / 40.7% / 23.7% / 0.0%; (B) fits in no range; (C) df-pn 408 against brute force 400 under 50 boards - STOP, a design review |
| M16 | applied, confirmed | M7's readings (corrected at CHECKPOINT 5): time-share multiplier, human-game weighting, the per-phase rule under 50 boards with the band resolving to off - the window is killed, provisionally |
| M18 | not started (rule) | A pilot self-play run before serious self-play re-checks the window and the other compute-weighted decisions - after §15.5, not in the untrained build |
| H46 | built, tested | Q2 (provisional): initial timelines count as created by neither player - a variant's set-up reads every timeline as initial (rulebook R10) |
| M1-M6, M10 | applied | The plan's process rows: the build follows them (this log) |
| M17 | measured | 6b″ on a quiet machine (d96d6d9): Guard 65.1 / 15.8 / 5.2 / 0.0% - adopted in no range; Hunt 69.0 / 63.6 / 35.0 / 0.0% (200+: one truth mate, too few to read) - adopted under 100 boards; every refutation and mate confirmed by E4 |
| C24 | built, tested | The refresh ladder's complexity trigger behind a flag (`CacheConfig.complexity_refresh`, off; prototype, gated like rung 1 on 6c's existence test); add-only over rung 1 for every signal; every number a placeholder (spec issues 75-79) |
| C25 | not built (rule) | Held open |
| M7 | measured; killed, provisionally (M16, CHECKPOINT 5) | On real games (t0database, 716 games, max 186 boards): on at k = 8 from 50 to 186 boards, off under 50, and the kill clause kills it under the human-game weighting (on-phases 18-38% of compute); confirmed by the user; above 186 provisional (MEASUREMENTS) |
| M8 | measured - STOP | 6b re-run (M12): root scan 76.7% / 31.4% / 24.7% / 2.4% by range (< 50 / 50-99 / 100-199 / 200+); rung 1 and quiet-leaf reach fail everywhere; stand-pat valid |
| M12 | applied - STOP | The bounded optimisation pass (E5c) and the re-run at 5 threads, Lazy SMP: 50-199 boards fail -> STOP, redesign the verification layer |
| M13 | applied | 8 cores per GPU: M9 judged on 2 expansion cores (passes, 1.87x), the scanner measured at 5 |
| M9 | measured | Pseudo-legal: PASS (2.2x / 2.5x the targets on one thread), history term 2%; with E5a's fast legality: 1.01x at 240 boards on one core (in the band), 1.87x on two; 1.21x at 600 |
| M11 | built | Licence recorded in PROVENANCE.md |

## Parked, and why

Unparked by the CHECKPOINT 1 rulings (2026-10-01): check detection, submission
legality, the mate detectors, the scanner and 6b, search on real positions, the
typed edges' check and pin relations, and the legal-move measurements (M9, M7).
Closed by them: repetition (H39 - there is none) and castling's cross-board attack
option (H40 - the per-board test is the rule, so `CastlingAttacks::AnyBoard` goes).

Still parked:
- the window code (C5, C7, C10): **killed, provisionally** (M16, CHECKPOINT 5) -
  not built; the pilot self-play run (M18) after §15.5 re-checks it;
- the M7 verdict's phases above 186 boards: no public game record reaches them;
  the pilot self-play run measures the engine's own distribution.
Not parked any more, but not built by rule: the NNUE (F4, H6, H7) is held open
(CHECKPOINT 4); rung 1, rung 3 and rung 5 are retired (M5).

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
    user. *Answered to 186 boards (2026-10-03):* t0database's 716 games that replay
    - 14 reach 100 boards, 2 reach 160, none passes 186; active timelines by bucket
    1 / 2 / 4 / 2 where §15.6's table assumes 6-12. M7 is decided to 186 boards
    except under 50 (the STOP band); provisional above.
27. **The STOP band is the brief's, not the spec's.** "A §15.6 result within
    ±10% of its threshold" stops the run; the band's reading (relative to the
    threshold) is recorded in step 5. Not a spec edit.
28. **What is a "legal move" for the expansion after E4?** R1 judges legality per
    action ("moving into check during a turn is allowed"): a single move is legal
    if it is part of some legal action. The expansion will offer the moves that
    begin a legal action (E4's enumeration), to be confirmed at the merge.
   E5a: `legal_moves` / `keep_legal_moves` are exact via `t5d/legal.hpp`; the
   expansion offers exact or budgeted classification (`ExpandOptions::legality`,
   still off by default).
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
46. **The legal-moves filter's cost and what search does with "unknown".** E5a's
    tiers make the filter cheap where it can be (fast: about 4x pseudo-legal and
    bounded; budgeted: bounded by its budget) and exact on demand (the mate
    search's tail). On check-heavy random play the fast mode proves few moves legal.
    The open question is search's treatment of an unknown move. Proposal (E5a; edit
    43): (a) search's expansion is budgeted - proved-illegal moves dropped, legal and
    unknown kept and flagged; (b) an unknown move gets its prior (dropping unknown
    moves could empty a node that has legal moves, which would then look like a
    mate); (c) the first time selection picks an unknown edge it is verified (exact
    classification, or `can_finish_legally` under a larger budget): illegal - remove
    the edge, renormalise its siblings' priors, select again, nothing backed up;
    legal - proceed; still unknown - proceed, flagged, never part of a proof; (d) a
    node is terminal only on an exact verdict (H37's detectors); (e) proofs take only
    exact verdicts; (f) training's policy target is visit counts, and the network's
    move mask is legal or unknown. The cost falls on visited moves only.
47. **Register H22's wording** still says "phantom boards for the mate-vs-stalemate
    verdict" and "per-board chess rules as 5D queries", which H36 and H40 replaced.
    Proposed spec edit 40.
48. **§2.3 / A3 has no player-to-move feature apart from the present.** A state
    whose present has shifted inside an unsubmitted action reads as the
    opponent's turn. Under the default rules such a state has no moves (Q4), so
    search submits first; under Q4's other reading it has moves and the network
    cannot represent it - the converter refuses it (`PresentMismatch`). Proposed
    spec edit 42. Bears on the Q4 ruling. *Ruled (D8, CHECKPOINT 2):* a global
    acting-player input and a submit action. The engine side is built (E7:
    `acting_player`, `submit_possible`, `submit_legal` in the expansion); the
    network side (the input, the submit logit, and dropping `PresentMismatch`) is
    N5. *Closed (N5, 2026-10-04): built.*
49. **§8.1's "generator-supplied legal moves"**: the C API offers pseudo-legal
    moves (spec issues 28, 46), so the policy scores pseudo-legal moves until the
    expansion's move set is decided. Proposed spec edit 41.
50. **Lineage beyond the parent on engine positions**: the C API has no board
    creation order and no travel source (issues 29, 30), so on engine positions
    `two_parent` reads as `lineage` and `creation` is unusable; 5DFEN positions
    lose origins too (issue 25). Timeline Battleground's outer initial timelines
    count as "origins unknown" (their parent of -1 is right).
51. **§11.1 defines no forcing set.** Engine E5b's, each class a parameter:
    captures (within a board, across timelines, through time, en passant); checking
    moves (a piece of the mover's on a board the move creates, that board advanced
    by a pass, can capture an enemy royal - an approximation of R1: it may count a
    board ahead of the present, C7, and misses a check given from a board the move
    did not create); branch threats; promotions (off).
52. **"Immediate threat"** (§11.1's branching moves "with an immediate threat") is
    undefined: placeholder `ArrivalCapture` - the arriving piece, on the passed new
    board, can capture an enemy piece worth at least 300.
53. **"Depth-limited" (§11.1) against "node cap" (H3, "defining quiet")**: built as
    a node cap with iterative deepening in forcing half-turns.
54. **Forcing moves and 5D obligations**: a forcing action is a legal action that
    begins with a forcing move; the rest of the turn is finished by a completion
    rule (a parameter: quiet moves by the least valuable piece by default).
55. **"All evasions" in 5D** are all legal actions of a turn - a product over
    boards. The scanner lists them up to a cap (512); a capped node gives an
    estimate and is not fully resolved. In-check listings take 52% of the scanner's
    nodes.
56. **5D piece values**: none in the spec. P N B R Q at the chess convention;
    unicorn 250, dragon 200, princess 700, brawn 125, common king 300 are
    placeholders; royals 0. The eval counts every timeline's latest board.
57. **"Node"** is undefined: the scanner counts a move applied, a royal-capture
    scan, or one node of an E4 search it calls.
58. **6b's terms are undefined**: "fully resolved", "scan leaf", "quiet", the
    evaluation rate when none is measured (no GPU here: §15.6's design rates,
    9,600/s at 240 boards and 4,600/s at 600, are used), threads (1, and physical
    cores - 1), pooled or per-size nodes/s; and stand-pat validity (operationalised:
    at a quiet position, every legal action - or a sample above 32 - answered by the
    opponent's 2-half-turn forcing search; invalid if every action loses its static
    eval).
59. **Draws as solved values**: H2 names wins and losses; the scanner exports
    stalemates as Draw(0) proofs.
60. **Proof distance** is an upper bound in half-turns.
61. **Is 6b's root test pooled over a corpus or per phase?** The pooled figure
    rests on the corpus's many small positions; above 50 boards - the design's
    regime - the root scan fails (preliminary). M7's window rule is per phase.
62. **§15.6's rung-1 "measured evaluation rate"** does not exist without a GPU.
63. **§15.6 does not write the window's multiplier formula.** "A 1.5× all-in
    multiplier, computed with 90% of compute at the cached per-node cost ... and
    10% at the dense cost": of three readings, only the log-weighted
    m = m_cached^0.9 · m_dense^0.1 reproduces §15.6's own ceilings and multipliers
    (MEASUREMENTS, M7). Only the early phase's verdict depends on the choice.
64. **"Phase" is undefined for M7.** The rule and the kill clause are "per phase";
    whether a phase is a board-count bucket (which?), and whether its median N or
    each position's own multiplier stands for it, decides the early game: the
    50-board bucket is on at k = 4 (1.60x), 10-board phases put under 20 boards
    off, the per-position median is 1.45x.
65. **The kill clause's weights are unspecified**: whose phase distribution (human
    games spend 88% of their moves under 50 boards; self-play cannot be measured
    yet) and which per-move cost (cached, dense or blended). The spec's "the late
    game dominates" is false for the only real games there are.
66. **(Tool, not spec)** `t5d-measure`'s game grouping joins distinct games that
    share an opening of 10-21 positions (674 groups for 692 distinct real games);
    harmless for the corpus, misleading for a game database. To fix in the tool.
67. **The proof channel's forcing class is undefined** (§11.1 "The redesign after
    6b", M14, M15): which first moves and which completions the attacker may use.
    6b′ turns on it - checks with a canonical completion decide 77/41/24/0% by
    range, the literal question 9/3/1/0%. E8: named parameters, placeholder
    Checks + Canonical.
68. **"Disproven" in M15 is class-relative** ("no mate within d by the class"), and
    256 of the 356 disproofs under 50 boards are roots with no checking action at
    all. Do those count toward the 50%?
69. **"Hypercuboid-pruned completion" is undefined.** E8 reads it as sound pruning
    (standing captures, dead passes) plus E4's detector at the defender's turn
    nodes and E4's listings where turns do not decompose; enumerating completions
    with the hypercuboid search itself was not built.
70. **The estimate channel's depth, move set, local model and margins** are
    unspecified (E8: 3 plies, threats at the first, created boards passed, local
    check, delta pruning).
71. **"Fits the rung-1 leaf budget"** needs operationalising (E8, fixed before
    measuring: p90 of node cost over 5 × measured nodes/s / the size-matched
    evaluation rate).
72. **M15's "keep whichever decides more"** has no tie rule, and brute force's and
    df-pn's "decided" are defined differently.
73. **The semimove decomposition is exact only under free-order rulesets** (R8,
    the count rule, Rewinds, no play activation, castling on optional boards);
    §8.2 and §11.0 item 3 assume order independence. E8 falls back to E4's
    listings elsewhere.
74. **Disproofs need a home**: H19 and §11.3 cover proofs only; E8 keeps
    class-relative disproofs in a separate store, never the solved store.
75. **C24's rule is unspecified** (§7.9, register C24): four candidate signals
    joined by "or", with no combination rule, threshold or weights. Built as a
    weighted sum against a threshold, every number a named placeholder (threshold
    1.0; 1/128 per board, 1/8 per active timeline, 1/2 per travel move, 1/12 per
    effort bin). Tuning belongs to 6c, if rung 1 shows a signal.
76. **C24's visit gate and per-search bound**: the spec gives none. §7.9's
    breakeven argument (one whole-scope refresh is about ten node evaluations)
    applies, so the build uses rung 1's V = 50; "epochs must be bounded per search"
    has no number - C24 is capped at 8 refreshes per search (rung 1's are neither
    counted nor blocked). C24 never refreshes the root (C3's job).
77. **"Travel since the last refresh" is undefined**: built as every travel move,
    branching or not, on the path from the node's epoch anchor (its nearest
    refreshed ancestor, else the search root) down to the node, its own move
    included.
78. **The effort head as a node-level signal**: §8.6's labels are root-search
    efforts; C24 reads the parent's evaluation off-label at internal nodes - the
    bin at quantile 0.5 (placeholder). An unevaluated parent contributes nothing.
79. **Does C24 sit on top of rung 1 or replace it?** The spec says "a ladder
    variant", the register "add-only if self-rated". Built as an OR with rung 1
    decided first and unchanged, add-only for every signal against the flag-off
    configuration; whether a structural signal may ever subtract from rung 1's
    trigger is not ruled.

80. **The submit prior** (D8, §8.2) until the network's submit logit exists: a
    placeholder `submit_share` 0.5 of a decision's prior. With logit 0 beside N
    flat move logits, untrained actions played nearly every optional board (one
    game: 101 moves in 14 actions, 54 timelines by action 16).
81. **§8.2's whole-turn action prior** from single-move logits: the sum is built,
    the mean a flag.
82. **E1's forms**: entropy-adaptive top-k widening, FPU (applied to unvisited
    *instantiated* children - §8.4 says uninstantiated), root Dirichlet α at 5D
    branching - every constant a placeholder.
83. **E2's and E3's formulas and constants** (behind flags).
84. **E9**: "a visit threshold for solved nodes" read as a per-child visit cap of
    8; the secure child as Q − A/√n with A = 1.
85. **E5's LCB**: z = 3, a 15% visit floor, categorical weight 0.
86. **MCGS residuals** (C11): how a shared node's value is backed up through
    several parents.
87. **§8.5's "values depend only on the state"**: the state must include whether
    an action is in progress - a turn start and a mid-action state can share
    every board and the acting player (two reviews found graph search merging
    them, pruning a legal mating submission). Built: MCGS identity is (key, turn
    start); the network is never told whether a leaf is a turn start.
88. **The turn cap's definition** (I11) and its record: "*" with a Termination
    header for capped records; t5d-play's default 300-board cap (a uniform
    Standard game reached 509 boards before turn 20).
89. **C14**: the evaluation cache shared across self-play's fresh searches,
    keyed by state, purged by board count at creation and re-root.
90. **The legality policy (a)-(f) as built** (issue 46, proposal 43): unknown
    moves kept but never feeding proofs, verified when first selected.
91. **The material evaluator's piece values** follow issue 56's placeholders.
92. **D8's whole-turn half is unbuilt**: the whole-turn arm has no end-of-turn
    token (the submit logit is ignored there; the C API reports submit_possible
    = 0).
93. **Memory**: about 80 bytes per eagerly stored edge; the whole-turn listing
    cap lowered from 4096 to 256 actions per node (1.5 MB per node at 103 boards
    before); actions are not stored lazily.
94. **The search's mate-check budget** against the near-mate tail.
95. **At large N, legality classification dominates self-play time** (41 of 43 s
    at 336 boards; 56 s of the board-capped Standard smoke games).
(The in-tree solver before the gate, an E6a draft issue, is settled by H41.)

96. **The corpus's export-form record** (§11.0 "the game corpus is in its format";
    H28): cwmtt/many2 is written in the official client's export form - every move
    names its board twice, the notation specification's "Export moves", which the
    specification itself calls "still a proposal". Only cwmtt read it. Ruled at
    CHECKPOINT 3 and carried out in the harness: cwmtt's side of the harness
    converts the text, and every source - the engine included - reads the
    converted text (converting for cwmtt alone would change nothing); a record the
    specification cannot express is given to no source. H28's "every source reads
    the record as stored" now has this one exception.

97. **Guard under the ply split is checked at submission** (a reading: the
    opponent moves only after it).
98. **Guard's depth**: d = the opponent's actions, 2d − 1 half-turns; depth 1 =
    their mate in one.
99. **"Existence, not the full listing" removes only our listing**: exactness
    still needs every opponent action, a product over their boards; depth 1 fails
    at 50 ms (measured). The cheap reading - one escape per checking move with the
    rest of their action held quiet - is inexact and not built.
100. **Lazy SMP (M13's 5 threads) gives Guard's exhaustive refutation little**:
    helpers re-search the same product.
101. **"Prove-only ... never spends time disproving"**: read as stop at budget,
    claim nothing, share no disproofs; df-pn's usual thresholds kept (measured
    better than proof-number thresholds alone).
102. **Hunt's and the truth's forcing class remain placeholders** (checks,
    canonical completion) - carries issue 67.
103. **Hunt's ground truth is the union** of the literal class's and the default
    class's proofs at 4M nodes each (2M tree cap), chosen from a 101-root curve
    that found no new mate past 500k nodes; "none" means no mate found, not no
    mate. Ruling 5 says "one long offline search".
104. **6b″'s chosen action for Guard is the record's next action** (no search yet).
105. **A Hunt range with no truth mate has no verdict**; the rule is silent.
106. **The asset map's estimate and boost** (4 plies, 128 nodes, flag at 300,
    boost 0.25 × min(1, value/900)) are placeholders.
107. **Hunt's non-lazy root expansion caps it at 100+ boards** (measured: the
    100-199 range's 20 truth mates are mates in one); a lazily expanded root is the
    remedy, not built - the measured Hunt is the pre-registered one.
108. **Where D8's acting-player input enters is unspecified** (§2.3/A3). Built as
    `acting_player_input`: a learned embedding added to the global registers
    (placeholder; frozen boards' K/V never see it) or to every token
    (`all_tokens`, frozen boards excluded under intrinsic features).
109. **Which register the submit logit is read from, and how** (§8.2, D8): built
    as global register 0 (`submit_register`) with a linear readout (`submit_head`;
    `pointer` the other arm). Tension with B8/D7 ("registers stay free scratch",
    "never assigned aux targets"): §8.2 assigns the logit to a register
    explicitly; it is a policy output, not an aux target.
110. **The value/WDL perspective**: §8.3, §12.4 and N1's heads say "the player to
    move"; under D8 the search's API defines it as the acting player's - built so.
111. **Policy targets cover the submit child's visits**: §12.2's exclusion and
    provenance must cover them too; the whole-turn arm still has no end-of-turn
    token (issue 92).
112. **An illegal submit**: masked by the network (no probability) and delivered to
    the search as NaN, which prunes the edge (the C API refuses infinities) - a
    convention to record.
113. **F1's parameter count** is now 21,359,370 (+769: the acting-player embedding
    and the submit readout).
114. **Match play between two evaluators**: §8.7's "match play carries everything"
    assumes one engine. Built as one search and cache per evaluator, re-rooted
    with `advance_action`, never shared; match games are deterministic given the
    networks.
115. **The engine's `t5d_search_new_sharing` does not check** that donor and new
    search have the same expand, split and legality parameters (Python refuses a
    mismatch; the engine may want the same check).
116. **Refresh epochs in a shared cache** need the search's epoch set per root
    (`engine_epoch`, the epoch key hashed to 64 bits).
117. **6b″ counts only proofs E4 confirmed**; a refuted proof withholds every
    verdict; no cross-check, no verdict - a reading: the pre-registration does not
    say.
118. **"Too few to read"**: a Hunt verdict resting on fewer than 10 truth mates is
    flagged (placeholder; the pre-registration names no minimum).
119. **Ruling 5's "pinned and hashed"** read as extending to the record list
    (content IDs in order), the rules by value and a code signature.

120. **No GPU on this machine** (an RX 9070 XT without CUDA): the §15.3 figures
    that need one - FA4-class evaluations/s, the leaf-batching gain, G2's FP8
    attention, B14's FlexAttention kernel, C15's varlen kernels, §15.6's Phase-3
    movegen re-check against a GPU evaluation rate - cannot be measured here.
    §15.3 asks only that each figure carry its conditions. Put to the user: accept
    counted FLOPs plus CPU-labelled wall times as the untrained build's write-back
    with the GPU figures named as deferred, or provide GPU access.
121. **Committed-class placeholders awaiting neither a ruling nor a named
    ablation** (§15.5's last clause): spec issues 9(a-j), 16(a-g), 31
    (`frozen_features`), 32 (`supersession_scope`), 108 (`acting_player_input`),
    109 (`submit_register` and its readout), and register B7 (deferred with no
    named ablation). Put to the user to rule, or to name the trained ablation each
    awaits.

122. **"Gating needs two oracles" is not the gating rule as built** (§11.0 "Tiered
    CI": "a ruleset is gated only when two or more oracles answer for it"; register
    H47: "since gating needs two oracles"). The engine differential
    (`gate.engine_differential`; `oracles/README.md`, "The engine contract") gates a
    ruleset when the engine replayed each of its records, at least one oracle
    followed each as far as the engine went, and the oracles' agreeing verdicts
    settle check, count and mate at no less than `min_checked` of the engine's
    positions; `Gated.oracles` counts the oracles behind the weakest record, and it
    may be 1 - the harness's own tests meet the conditions with ftxi alone, canned
    and live. Whatever kept the commit tier from gating with the engine configured
    (Q-A, 2026-10-04: the branch is lost, its run not re-checked here), it was not
    a two-oracle rule. Built: H47 keyed to `Tier.full_strength` (the tier runs
    every oracle search over the whole corpus), never to a count of oracles; in any
    other tier a ruleset that met the gating conditions with its oracles alone is
    reported as not gated, with that reason (`gate.SMOKE_REASON`). No two-oracle
    requirement was added: it would also change what nightly and release gate (the
    variants one oracle reads) - a decision for the user. Put to the user: should a
    ruleset be gated only when two oracles stand behind it, in every tier? Until it
    is ruled, proposed edit 98 gives both texts.
123. **H47 is silent on an engine compared with nothing in a smoke test.** As
    built, an engine with no finding against it passes the commit tier's
    differential even when it declined every record (nothing was compared, so
    nothing was found); nightly and release fail it (`engine-nothing-gated`),
    and parse mode keeps its own floor (`engine-nothing-compared`) in every tier.
    The self-tests pin it (`test_gate_live`, `GateRunTests`), so the consequence is
    visible. Put to the user: should the smoke test hold full mode to parse mode's
    floor - the engine replayed at least one record, and an oracle's answer was
    compared with it? Nothing is parked on it.
124. **H47 is silent on a run of a full-strength tier restricted to some sources
    or records** (`gate --tier nightly --source ftxi`; "tiers that run every
    oracle"). Built - the literal reading: the run keeps its tier's rule
    (`GateResult.full_strength` is the tier's; the run is marked partial). It names
    as gated what met the gating conditions with the sources and records it ran -
    one oracle can (issue 122) - and fails an engine with no finding against it
    that came through no ruleset, so a run of sources that cannot gate (cwmtt alone:
    no check verdict) always fails such an engine. The proof channel stays off for
    every partial run, and CI runs none, so no verdict anyone relies on depends on
    it. The other reading: a run that leaves out one of its tier's sources is a
    smoke test and gates nothing. Pinned (`GateRunTests.test_a_restricted_run_keeps_its_tiers_rule`)
    and documented (`oracles/README.md`, "Other commands"). Put to the user;
    nothing is parked on it.
125. **A written `x` on a move that takes nothing** (§11.0 "Running the gate":
    parse divergences "are settled against the pinned notation specification";
    register H28). The specification's "Jumps" syntax has "`x` if a piece is being
    taken", and its own example reads `>>x` as "jumps and takes"; it says nothing of
    a reader that meets an `x` on a move that takes no piece - unlike `~`, `(>Lx)`
    and `(~Tx)`, which it calls "purely for better human comprehension". The engine's
    reader refuses such a move as unreadable (a written `x` must capture) and reads a
    capture written without one (`require_capture_marks` false): a reader's policy
    in each direction, not a rule. The corpus has one case, cwmtt/futures action 17
    (`(L1T5)Qf4>>x(L0T5)e4~`, onto an empty square): entries KD-0167 to KD-0169 record
    cwmtt's, Chess in 5D's and the notation parser's playing it as a lenient reading,
    citing the syntax - ftxi refuses it as the engine does. Taken as an annotation
    (a warning, like `~`), the move would be read, those three entries would go,
    ftxi's refusal would become a soft divergence whose entry would have to cite the
    text as valid notation, which the syntax does not support, and Standard - Turn
    Zero, whose record the engine would then read to its end, might gate. Built: the
    strict reading. Read the same way in the other direction - `x` if and only if a
    piece is taken - it would make the engine refuse ten records it now reads, each
    at a capture written without `x` (`t5d-gate --parse-only --pgn
    require_capture_marks=true`, 2026-10-17): cwmtt/100_timelines at action 12,
    ctp2 2, exiledKings 8, many2 7, manyChecks 7, niceAndrey 23,
    tesseractMageOChicken 25, ftxi/verysmall 1 (refused at 3 already),
    notation/aivsai 13 and notation/game-2 34; 89 of the 110 R1 entries of
    2026-10-17, and 20 earlier ones, sit past those points. So the engine's reading
    is lenient one way and strict the other, and the three entries are marked
    provisional - in their notes and in the file's header, which says how they go if
    the ruling is the other way: withdrawn with the reader's change, on the ruling's
    authority, not as telltales (a review finding). **Put to the user, and wanted
    before lane/gate is merged**: the gate's green rests on these three entries.
    Without them the commit tier fails on action 17 against the notation parser,
    and nightly against cwmtt, Chess in 5D and the notation parser; if the question
    is still open at the merge, the review's recommendation is to leave them out
    (`AGENTS.md`: no entry that cannot be justified may make the gate pass), and the
    gate is then red on that action alone. Nothing else is parked on it.
126. **"(and reported upstream)"** (§11.0, "A known-divergences file": "if the
    oracle is wrong, the divergence is recorded with its rulebook citation (and
    reported upstream)"; register H31, H34). None of the 169 entries is reported
    yet, each saying why: 147 - every one of R1's - "ftxi follows a different reading
    of the check test; settled by ruling R1", as the 37 of 2026-10-02 did; the six of
    the two defects R7 drafted for upstream (`docs/checkpoints/upstream/`) wait on the
    user to file them; ten record lenient readers ("no defect to report"); six
    oracle notation errors are "not yet drafted". As built, `verify` accepts an
    entry that says why it is unreported. Put to the user: report ftxi's reading of
    the check test upstream - one report would answer all 147 entries - or read
    §11.0's clause as applying to defects (proposed edit 99)? Nothing is parked on
    it.
127. **Issue 122 and proposed edit 98 (b), measured** (2026-10-17, the engine
    configured; step 2). Both are as put to the user; this is the measurement edit
    98 (b) waits on. In the commit tier no ruleset meets the gating conditions with
    ftxi alone: the engine cannot read a record of Standard (ftxi/softmate), of Turn
    Zero (cwmtt/futures), of Very Small - Open (cwmtt/ctp1, ftxi/verysmall) or the
    custom positions' (ftxi/niceAndrey's set-up) - the known divergences cite the
    notation for each; ftxi follows no record of Half Reflected, the Focused
    variants, Misc - Small or Timeline Battleground; the engine does not set up
    Timeline Invasion or Marauders; and Reversed Royalty has a check verdict at 85
    of its 122 positions (ftxi's at the other 37 are on record as wrong). Nightly
    gates Standard - Half Reflected and Standard - Reversed Royalty, with at least 2
    and 3 oracles behind each record: at this corpus a two-oracle rule (edit 98 (b))
    would leave no ruleset ungated that nightly now gates. Nothing is parked on it.
128. **The engine's time limit in the gate** (§11.0 "Tiered CI" and "Stable answers
    only", register H32: the tiers' budgets are the oracles'; nothing says how long
    the engine may take on a record). It had the tier's `timeout_s` - in the commit
    tier 60 s - and CI's first commit run with the engine (run 39009070712) took
    44.4 s on cwmtt/many2 on a runner that ran the calibration probe at 0.75x, the
    runners ranging from 0.59x to 1.09x. Built (a review fix): `[engine] timeout_s`,
    300 s a record in every tier, or the tier's own `timeout_s` where that is longer
    (release, 600 s), so no tier gives the engine less time than before; a record
    not finished by then fails the engine, as before. It touches no oracle budget,
    required set or golden file. Also: §11.0 says "a calibration probe scales
    budgets to slower runners" - as built the probe scales nothing: it says how much
    slower than the recording machine a runner is, so that a value a slow runner
    fails to reach can be told from a regression (proposed edit 100). Put to the
    user: the value, and whether the engine's limit should be per tier. Nothing is
    parked on it.

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
24. **§11.0 "Stable answers only"**: say how the required set is drawn - the
    stable values whose recorded cost times a margin fits the tier's budget,
    recorded at least twice - and that the gate requires 100% of it (H32).
25. **§11.0 "Recorded-answer integrity"**: name the mechanisms - the manifest
    verified before every comparison, the re-hash of the harness after the engine
    runs, imports from source only - and say plainly that the manifest protects
    against accident, not against an edit of both a golden file and the manifest:
    that is review's job (H33).
26. **§11.0 "Running the gate"**: add the engine's parse mode as the stage before
    full mode, and define `legal_actions` (an action is its set of moves, with
    its timeline-creating moves in creation order) - a protocol definition, not a
    rule.
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
41. **§8.1**: say whether the pointer head scores pseudo-legal moves, legal first
    moves, or actions - with D2 and spec issues 28, 46 (issue 49).
42. *Withdrawn - superseded by D8 (CHECKPOINT 2), which the upstream documents
    now carry.* ~~§2.3 / A3: either add the player to move as a global input
    feature, or state that leaves are evaluated only at action boundaries or
    before the present shifts - needed under Q4's other reading (issue 48).~~
43. **§6, §15.6 ("expansion - legal moves") and §11.0 item 3**: "legal moves" is
    not defined beyond R1. Proposed: "the moves not proved illegal - each marked
    legal or unknown - by the rules engine's sound tiers under a budget (exact on
    demand); search verifies an unknown move when it first selects it, and scores a
    node terminal only on an exact verdict" (issue 46).
44. **§11.1 recommended design, item 1**: "Node-capped alpha-beta over forcing
    actions (iterative deepening in forcing half-turns)" (issue 53).
45. **§11.1 "defining quiet"**: state the forcing set and define "immediate threat"
    (issues 51, 52).
46. **§11.1 two channels**: a loss proof needs the defender's legal actions
    enumerated exhaustively; a capped node gives an estimate; say whether stalemate
    draws are solved values (issues 55, 59).
47. **§11.3**: the TT holds no proofs; the solved store is read first.
48. **§15.6 6b**: define "fully resolved", "scan leaf", the evaluation-rate fallback
    and stand-pat validity, and say whether the verdicts are per phase (issues 58,
    61, 62).
49. **§15.3 "Scanner nodes/s"**: write back (M4) after the quiet run.
50. **Register H6**: a 5D piece-value table, pending a decision (issue 56).
51. **§15.6 "The window"**: write the multiplier's formula -
    m = m_cached^0.9 · m_dense^0.1 with T = 66N + 16 tokens and the N3 cost model
    - once the user confirms it (issue 63).
52. **§15.6 "The window"**: define a phase for the rule and for the kill clause,
    per the user's answer (issue 64), and whose distribution weights the kill
    clause (issue 65).
53. **§15.6's window table and the register (M4 write-back)**: add the measured
    rows - N 24 / 60 / 112 / 170, active T 1 / 2 / 4 / 2, W(8) 9 / 19 / 36 / 19,
    median |E(8)| 0 / 0 / 2 / 2, k = 8 ceilings −1.1 / 12.4 / 31.0 / 87.6, k = 4
    ceilings 2.9 / 20.4 / 44.0 / 95.6 - and qualify "the late game dominates" and
    "at ~100 boards it cannot pay at k = 8" against them.
54. **§10.4.2 step 1**: the |E| phase curve is measured on real games to 186
    boards; above that, provisional.
55. **The register's open item "Real game records for the |E| curve"**: records
    now measured (t0database); the verdict is provisional only above 186 boards.
56. **§11.1 "The redesign after 6b", part 2**: state the proof channel's question
    exactly - mate within d by the attacker's forcing class against every legal
    defence - with the class's parameters, that disproofs are class-relative and
    go to a disproof store, never the solved store, and the decomposition's
    exactness condition with its E4 fallback (issues 67-69, 73, 74) - subject to
    the design review.
57. **§11.1 part 1**: the estimate channel's local model (issue 70).
58. **§11.3 and register H19**: the disproof store, keyed by position, class and
    depth (issue 74).
59. **§15.6's CHECKPOINT 3 / 6b′ block and the register (M4 write-back)**: the
    "fits" operationalisation, both definitions of "decided", the tie rule, and the
    per-range figures (MEASUREMENTS, 6b′) (issues 71, 72).
60. **The register, M14 and M15**: 6b′ measured - STOP; the design review's
    outcome to follow.
61. **§7.9 "Complexity as a trigger"**: state the rule as built (a weighted score
    against a threshold; rung 1's visit gate; a per-search budget; never the root;
    travel from the epoch anchor; effort from the parent's evaluation at a
    quantile; add-only for every signal, rung 1 decided first and unchanged), name
    the placeholders, and give their tuning to 6c (issues 75-79).
62. **Register C24**: "add-only if self-rated" -> "add-only (as built, for every
    signal; rung 1 decided first)", noting "built behind a flag, off" (issue 79).
63. **§8.6, the effort head**: a second consumer - C24 reads the parent's predicted
    difficulty at internal nodes (off-label, confidence-family, add-only) (issue
    78).
64. **§8.5**: "values depend only on the state" -> the state includes whether an
    action is in progress; a turn start and a mid-action state with equal
    boards are different nodes; the network input is unchanged (issue 87).
65. **§15.4's invariants**: add - a submit edge leads to a turn start, a move
    edge to a mid-action state; no game end during an action; no dead end with a
    legal completion.
66. **§8.7 / C14**: the shared, state-keyed evaluation cache and its board-count
    purge (issue 89).
67. **§8.2's whole-turn arm**: the listing cap's memory role, and the missing
    end-of-turn token (issues 92, 93).
68. **§8's solver text** points at H41's switch (`proof_channel`).
69. **§8.4**: name E1's, E2/E3's, E5's and E9's constants, the submit prior and
    the whole-turn prior as parameters with placeholder defaults (issues 80-85).
70. **I11 / §8.6**: the turn cap's definition and the capped record's
    Termination header (issue 88).
71. **Register H28 and §11.0 ("Third-party code", the notation specification's
    row)**: H28 now says "Own 5dpgn/5dfen parser, written from the pinned notation
    specification - a shared parser corrupts both sides of a comparison
    identically"; the row says "the game corpus is in its format". Append to H28:
    "The engine's parser reads the specification's 5dpgn and is not extended to
    other notations. Every source is given the same text of a record; the one
    exception (CHECKPOINT 3): a corpus record in the official client's export
    form is converted by the harness to the specification's 5dpgn, as notation
    only, never reading the position, and every source, the engine included,
    reads the converted text; a record with a move the specification cannot
    express is given to no source." The row: "the game corpus is in its format (a
    record in the official client's export form is converted to it before any
    source reads it - CHECKPOINT 3)". Optionally H27/H33: the pinned corpus pins
    the converted text too. Why: the CHECKPOINT 3 ruling (issue 96).

72. **§11.4's Guard row**: the depth semantics, the submission reading, that
    exactness enumerates every opponent action ("existence" means ours), at most
    50 depths (issues 97-100).
73. **§11.4's Hunt row and §11.1**: prove-only as defined; export refused until
    H41/H23 (issue 101).
74. **§15.6's 6b″ block**: the readings (the chosen action, the truth's class
    union, budget and curve, no-verdict ranges); after the quiet run, the verdicts
    and the M4 write-back to MEASUREMENTS (issues 102-105).
75. **§13**: per 6b″'s verdicts ("the chosen move goes unguarded" where Guard
    fails; "proofs only from the search's solver" where Hunt fails).
76. **§11.2 rung 4**: the asset map as built (issue 106).
77. **§11.3 / M13**: Lazy SMP is no help for exhaustive refutations (issue 100).
78. **Register H1, H10, H42, H43, M14, M17**: as built.
79. **§2.3 / A3**: add the global acting-player input (issue 108).
80. **§8.2 / D8**: name register 0 and the linear readout as placeholders; one
    distribution over moves + submit; an illegal submit gets no probability and
    no logit is sent (issues 109, 112).
81. **§8.3 / §12.4 / D3 / D4**: value and WDL "from the acting player's
    perspective (D8)" (issue 110).
82. **§12.2 / §12.4**: the policy target is the visits over moves + submit; the
    exclusion and provenance apply to the submit child too (issue 111).
83. **§9.2 / F1**: 21,359,370 parameters (issue 113).
84. **§8.7 / C14 / C12**: one search and cache per evaluator in match play; a cache
    is shared only between searches with one evaluator and one network input, and
    evaluations are keyed by refresh epoch (issues 114-116).
85. **The register's D8 row**: built (engine, search, network).
86. **The register's open items**: issue 48 closed by D8; issues 5 and 29 closed
    by C23.
87. **§15.6's 6b″ block**: only confirmed proofs count; a refuted proof withholds
    the verdict; the shipped configuration is the measured one; the "too few to
    read" flag (issues 117, 118).
88. **§15.6 / ruling 5**: the truth is pinned with its provenance, and the tools
    refuse other records or rules (issue 119).
89. **H41/H23 and H43**: Hunt's export switch is refused like the search's proof
    store.
90. **§11.1 / H19, wherever disproof classes are described**: a disproof's class is
    every option that shapes the attacker's class.
91. **§15.2's rows that cannot exist in the untrained build**: "windowed = full
    restricted" (C5) and "lazy K/V = eager" (C10) - the window is killed,
    provisionally (M16): mark them deferred to the pilot self-play run (M18);
    "NNUE accumulator = recompute" (F4) - the NNUE is held open (H6): mark it
    held open.
92. **§15.3's rows that cannot be measured as written**: windowed throughput, the
    windowed "one refresh ≈ 10 evaluations", windowed KV memory (the window is
    killed provisionally); the local/global interleave (C18 held open); NNUE
    evaluation (held open); "rung 1 hidden under batch latency" (rung 1 retired) -
    mark each deferred or void, with the reason.
93. **§15.3's movegen row and §11's projection (M4 write-back)**: pseudo-legal
    44,047 actions/s at 240 boards and 24,951 at 600 (one thread); with fast
    legality 20,249/s at 240 on one thread (1.01x the target) and 37,316/s on M13's
    two expansion cores (1.87x), 12,133/s at 600 (MEASUREMENTS, M9).
94. **§15.3's scanner row (M4 write-back), restated for v3**: the brute-force
    scanner 787k nodes/s per thread alone and 566k at 5 threads (E5c), df-pn 239k
    (6b′); the shipped consumers are Guard and Hunt - their figures are nodes per
    root at 50 ms (6b″: Guard about 18-24k, Hunt 0.7-61k by range); supersedes
    proposed edit 49.
95. **Register F4, H7, H14, C5, C7**: F4 and H7 "Committed" -> held open with the
    NNUE (H6, CHECKPOINT 4); H14 restated for the reduced ladder (Guard, Hunt, the
    asset map); C5 and C7 "Provisional" -> killed, provisionally (M16), re-checked
    by M18.
96. **Register C9, D6, I9 and the process rows**: record their status as built
    (I9: the ruleset ID in every key) or to be verified (C9).
97. **§15.6's Phase-3 re-check**: "movegen ≥ 2x the measured evaluation rate"
    needs a GPU evaluation rate (issue 120); until then the CPU figure (N5's 6.9 F1
    evaluations/s on CPU) is the only one, and the re-check is deferred.
98. **§11.0 "Tiered CI" (from "The commit tier is a smoke test") and register
    H47 - pending the user's ruling on issue 122** (and on issues 123-124, whose
    answers go here too). §11.0 says "The commit tier is a smoke test: a ruleset is
    gated only when two or more oracles answer for it, so the check that *some*
    ruleset is gated runs only in the tiers that run every oracle — nightly and
    release (CHECKPOINT 6)"; H47 ends "since gating needs two oracles". The gate as
    built has no two-oracle rule (issue 122), so the text depends on the ruling:
    (a) **No two-oracle rule** (as built). §11.0: "The commit tier is a smoke test:
    it does not run every oracle search over the whole corpus, and gates no
    ruleset; an engine with no finding against it passes it, and the gate's output
    says that the tier gates nothing. A ruleset is gated, and the check that *some*
    ruleset is gated runs, only in a tier that runs every oracle search over the
    whole corpus — nightly and release (CHECKPOINT 6). In every tier an unexplained
    divergence, a telltale or a breach of the engine contract fails the engine."
    H47: "... the check that some ruleset is gated runs only in tiers that run
    every oracle search over the whole corpus (nightly, release); the commit tier
    gates nothing, and says so."
    (b) **A two-oracle rule.** §11.0's sentence and H47's reason stay, and the
    gate gains the condition: a ruleset is gated only when two independent oracles
    stand behind each of its records (`Gated.oracles` ≥ 2), in every tier - so the
    commit tier, with one rules oracle, could gate nothing anyway. It changes what
    nightly and release gate: a ruleset with a record that only one oracle follows
    as far as the engine goes is gated in no tier, and the proof channel can never
    be enabled for it - which rulesets, measured once the engine is configured.
    Why: §11.0 and H47 give a two-oracle rule as the reason the commit tier gates
    nothing; the gate has none, and whether to add one is the user's decision.
99. **§11.0 "A known-divergences file" - pending the user's ruling on issue 126.**
    It says: "if the oracle is wrong, the divergence is recorded with its rulebook
    citation (and reported upstream)". (a) **If ftxi's reading of the check test is
    not reported**: "... recorded with its citation - the rulebook's, or the pinned
    notation specification's where it settles the matter - and reported upstream
    when it is a defect of the oracle's (R7); an oracle that follows a reading the
    rulebook rules against, or reads notation more leniently, is recorded as
    unreported, with the reason". (b) **If it is reported**: no edit; the entries'
    `upstream` fields name the report once it is filed. Why: 147 entries say
    "unreported", and the spec's clause reads as though every entry is reported.
100. **§11.0 "Stable answers only"** (issue 128). It says: "each tier requires the
    stable values, recorded at least twice, whose recorded cost times a margin fits
    its budget, and a calibration probe scales budgets to slower runners." It should
    say: "... fits its budget; a calibration probe says how much slower than the
    recording machine a runner is, so that a value a slow runner fails to reach can
    be told from a regression - it scales no budget. The engine has a time limit of
    its own per record, never shorter than the tier's: the tiers' budgets are the
    oracles', and the gate compares the engine's answers, not its speed." Register
    H32 does not name the probe: no edit. Why: as built no budget is scaled, and the
    engine's limit is not a tier budget (issue 128).

## Session log

- **2026-10-01** - Pre-flight passed. Design documents moved into `docs/design/`
  (they had been kept untracked as `Specification documents/`). Step 1 started.
- **2026-10-04 to 2026-10-17 - the machine was lost.** Work after 3b33942 that
  had not been pushed went with it: the gate-with-engine branch and E10 ("Recovery
  after the lost machine" above). The user rebuilt the machine's general tools and
  cloned the repository; the design documents of CHECKPOINTS 6 and 7 arrived from
  upstream, uncommitted.
- **2026-10-17 - recovery** (the user's brief, step 0). The register holds every
  row the brief names (A7, A8, B15, B17, D9, F5, F6, G9, I12-I20, M19-M24); the clone
  is of the right remote, the four submodules at their pins and clean, `git
  ls-remote origin` and `git push --dry-run origin HEAD` work. Installed, each
  asked first: Python 3.14.7 (winget, user scope), the project `.venv`; GHC 9.10.3
  into the harness cache by `build --bootstrap`; the repository's author identity
  set repo-locally. MSYS2's g++ 16.2.0 survived and is the compiler, as before.
  Main verified (figures above). Not yet: the comparison with the last green CI
  run (no `gh` login). Rule from now on (M23): every working branch is pushed
  after each commit, and takes main by merge, never rebase.
