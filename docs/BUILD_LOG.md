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

**Survived** (all of it on `main`): everything "On main" in "State on 2026-10-04
(before the loss)" below - the rules engine, the search core, verification v3's
components, the network N1-N5, C24, the oracle harness and the golden files,
t0database, and this log.

**Lost - never pushed - and to be redone:**
- **The gate with the engine configured** (branch
  `worktree-agent-a1a2868a52f6c83b9`, 61e9aa8): 113 triaged known divergences
  (110 R1, 3 notation), no engine defect, nightly passed. Redone by the gate lane
  after H47 (gate step 2, in step 1 of the plan in "Current state (2026-10-17)").
- **Engine E10** (it was running in a worktree): Hunt under 100 boards, the df-pn
  settling scans under 50 and the asset map wired into the search, Guard's
  re-search hook (off), the offline audit lane's tooling (H44). Redone in step 2 of
  the plan.
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
**Compared with CI (2026-10-17, after the user's `gh auth login`): the same.** The
last green runs on the same code - the engine job on f07568c (g++ 13 and 14, 526
tests), the network job on 9be6f04 (775 passed, 2 skipped; no code changed since),
the commit tier on f07568c (ftxi 8256 of 8256, notation 3012 of 3012; calibration
1.08x on the runner) and the nightly tier on 3b33942 every night from 2026-10-04 to
2026-10-16 (every source's required set reproduced; calibration 0.95x) - give
exactly the local results. One blemish: **the engine job failed once on 3b33942
under g++-13** (run 37146667584) in `guard_at_five_threads_depth_one_is_the_opponents_mate_in_one`
(`test_guard.cpp` 83 and 85: `line_complete` and the line's mate both false) and
passed on the same code in f07568c's run - an intermittent test: Guard's
`line_complete` may be false under Lazy SMP by its own contract ("it stops where the
deciding thread's tree holds no continuation"), yet the five-thread test asserts it.
Queued for the engine lane (make the line complete from the certificate, or test
the documented contract).

### CHECKPOINT 8 - ruled 2026-10-17 (design documents d5cc8d1)

**The instruction files: the one edit the user allowed in each.** The user's words:
"CLAUDE.md and AGENTS.md: you may make exactly one edit to each - add the engine build
step (py engine/build.py) before the gate commands in their command lists. Nothing
else in those files; commit it separately and quote the diff in the build log." The
diff, in full:

```diff
--- a/AGENTS.md
+++ b/AGENTS.md
@@ -129,2 +129,3 @@ python oracles/harness.py verify               # pins, corpus, divergences, boun
 python oracles/harness.py selftest             # the harness's own tests
+py engine/build.py                             # build the engine first: the gate runs build/engine/t5d-gate
 python oracles/harness.py gate --tier commit   # the per-commit gate (nightly | release for the full one)
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -64,2 +64,3 @@ python oracles/harness.py selftest             # the harness's own unit tests
 python oracles/harness.py rulebook             # is docs/rules/README.md the pinned rulebook? (--pin after review)
+py engine/build.py                             # build the engine first: the gate runs build/engine/t5d-gate
 python oracles/harness.py gate --tier commit   # per-commit gate; nightly and release run all three oracles
```

**The rulings** (register H27, H47, H51, F6, M22, D2 as changed at CHECKPOINT 8), and
where each is built:
- **125 - capture markers read leniently (H51, a project decision).** The notation
  spec says when to write `x`, not how to read a superfluous or missing one: a move is
  read as its squares define it, a mismatch logged as a warning, the marker written
  strictly. The reader lenient in both directions, a test that the writer marks
  exactly the captures, a rulebook entry labelled a project decision, KD-0167-0169
  removed and ftxi's refusal recorded citing H51 and that entry, nightly re-run
  (Standard - Turn Zero gated if two oracles answer for it). Gate lane, "gate step
  3a" (running). The reader is a truth source: the pinned measurements (6b, 6b′,
  6b″, Hunt's truth) are kept reproducible, never re-recorded in it.
- **122 - a ruleset is gated only when two or more oracles answer for it** (H47: "as
  built the gate required one, and is aligned to two"); "a single oracle cannot tell
  its own bug from ours". **123 - yes**: the commit tier fails if the engine declines a
  record it read on the last green run, or reads none. **124 - no**: a restricted run
  (`--source`/`--record`) never produces a tier verdict; its output is labelled
  "restricted - not a tier result". **126 - report it too**, as a question citing the
  source and a few representative positions, not as a bug (H27: defects as bug
  reports, documented-rule disagreements as questions; drafted by the agent, filed by
  the owner) - drafted in `docs/checkpoints/upstream/`. Gate lane, "gate step 3b",
  after 3a; the gate lane merges once both are reviewed.
- **142 - the D2 build order: the tree form first** (it modifies the existing tree and
  is anchored by its equivalence to the ply split), **then the flat form; both are
  built.** M22's figures count playable boards, which include optional ones; per range
  it also reports the share of turns with more than one and with three or more
  mandatory boards, the optional moves actually played, and the compute share of
  multi-board turns (moves x per-move cost). Engine lane: the figures after H51
  merges (the reader changes what t0database replays, if anything), then the tree
  form, then the flat form.
- **204 - yes: Mish in every head's hidden layers, the value head included; squared
  ReLU stays a trunk-only arm** (F6). Network lane, its next component.
- **CLAUDE.md and AGENTS.md** - the one edit allowed in each, above (11caec9).

### CHECKPOINTS 6 and 7 - ruled upstream (design documents f07568c)

The user's rulings at CHECKPOINTS 6 and 7, as the design documents maintained
upstream record them - committed unedited (f07568c): 30 register rows added and 62
changed (H41-H46 also moved from part M into part H), reversals 25-29, new open items
(part K) and predictions (part L), and the spec's new §9.3 and §12.8. Quoted below:
the register's own words - its status column wherever a status is given.

**What they decide, by register ID:**
- **The network's inputs.** A6 "Committed": global turn-state inputs on the global
  registers - the acting player and colour, mandatory boards still to move, optional
  boards still playable, moves already made this turn - exact from the rules engine
  (§15.2 adds a recount test). A7 "Committed": perspective canonicalisation - pieces
  as ours/theirs, ranks mirrored and timelines negated when Black acts, outputs
  mapped back (§15.2 adds a colour-swap mirror test). A8 "Prototype" (6p): the
  board-layout embedding. A3: on frozen boards under the cached configuration,
  status enters as B16's key biases. A5: "build behind its flag (CHECKPOINT 6) — the
  window's provisional kill does not decide it".
- **Positions and attention.** B15 "Provisional (v1 default)": a learned 4D
  displacement bias beside RoPE, about 1.6M parameters, with a no-table arm in
  ablation 2; B6 "Superseded by B15 (CHECKPOINT 6)" (reversal 25). B16 "Committed
  (with the cached configuration)": score-time status key biases on frozen boards
  (settles issue 31). B17 "Prototype" (ablation 2): a key-side Shaw term on top of
  B15, value-side vectors held open. B8: per-board registers position-free on file
  and rank, global registers on every pair. B5: its as-built details are "v1
  placeholders" ("Provisional"; ablation 2). B9: the built details stand as v1 values
  (§6 "As built"); B12 is recorded as built. B11: travel-threat folded into attack
  (reversal 26) - five types.
- **The trunk.** F5 "Committed (v1; placement provisional)": RMSNorm-style layers
  without biases, no Q/K/V biases, pre-norm (post-norm with DeepNet initialisation
  is 6y's arm). F6 "Committed (v1)": Mish in the FFN (squared ReLU is 6y's arm). F1
  "Committed (v1; FFN ratio in question)" and F2: ablation 5 gains an arm, FFN 2x
  against 8x.
- **Heads.** D3 "Committed": "Value heads (corrected at CHECKPOINT 7)" (reversal 28) -
  four of them, as §8.3 counts: WDL on the game result is the value search uses; a
  51-bin categorical over the root search value of full searches, HL-Gauss-smoothed,
  feeds variance and overlap; short-term values at two or three horizons; an error
  head from the start; all four mean-pooled from the global registers, FP32. D4
  "Committed set (reply: prototype); weights ablated": the soft policy (the target to
  the power 1/4, weight 8), moves-left in turns, the opponent reply reinstated as a
  prototype at weight 0.15 (reversals 26, 29). D9 "Prototype" (6t): the optimistic
  policy, match play only. D1: promotions get a learned per-piece offset. D8: the
  submit logit is read from the pooled global registers, so D7 holds.
- **Turns and search.** D2 "Undecided — measured jointly with widening and MCGS,
  after the turn-width measurement (M22)", gate 6x: the ply split (built, the
  default), the tree form and the flat form; "The enumerating whole-turn arm and its
  256 cap are superseded by the flat form." M22 "Committed (process)": the turn
  width measured first. E1: root Dirichlet noise with its total concentration split
  across the legal moves (shaping a prototype) and a root policy temperature (I19);
  the widening schedule may depend on the mandatory-board count. C11 and H18:
  identity adds branch origins and whether a turn is in progress. C14: self-play's
  fresh searches share the state-keyed evaluation cache; one search and cache per
  evaluator in match play. E4 "Committed (principle, weakened)". E7: the second
  volatility source is Hunt's and the settling scans' proofs.
- **Verification and the gate.** H42 "Not adopted in any range" - "kept as the audit
  lane's offline question"; H43 "Adopted under 100 boards"; M17 "Measured: Hunt
  adopted under 100 boards, Guard in no range"; H1 records it. H14 "Committed
  (restated, CHECKPOINT 6)": budgets for Hunt, the asset map and the settling scans
  under 50 boards, Guard offline. H49 "Held open": Hunt with a lazily expanded root.
  H50 "Provisional (placeholders)": the 5D piece values. H47 and H48 "Committed
  (process)": the gate's tier reading and the naive searches' budgets (Q-A and Q-B,
  below). M8 "Measured — STOP at CHECKPOINT 3 → v3" and M15 "Measured — STOP at
  CHECKPOINT 4 → v3"; M9 "Measured — passes on two cores; the Phase-3 re-check waits
  for a GPU rate (M19)". F4 and H7 "Held open with the NNUE"; C5 and C7 "Killed,
  provisionally"; C10 "Deferred with the window (M16)".
- **Precision and serving.** G2 (Provisional, now gated by 6w): 8-bit inference, FP8
  before INT8, on the trunk's matrix multiplications, attention included. G8
  "Prepared (behind a flag)": QAT hooks, smoke-tested. G9 "Committed": PyTorch
  (torch.compile + FlexAttention) hosts the network and drives the C++ search
  through the C API; ONNX/TensorRT export "refused for now"; one bounded attempt on
  the AMD GPU. M19 "Committed (process)": no GPU (below).
- **Training.** §12.8's mechanics: I12 "Committed (default sampler)" (surprise
  weighting), I13 "Committed" (deblundering), I14 "Committed (gated)" (proof
  rescoring, after the sign-off), I15 "Committed (process)" (rich records, fixed
  before the pilot run), I16 "Committed (placeholder rule)" (the replay window), I17
  "Committed" (SWA), I18 "Committed" (the false-resign check), I19 "Committed
  (placeholders)" (exploration), I20 "Prototype" (KL-adaptive full searches, 6l).
  I2: search-value targets come from the pruned visits too. I6's realised-error arm
  is superseded by I12. M21 "Committed (process)": the training loop written and
  smoke-tested (below). M24 "Committed (process)": after the pilot run, network-only
  ablations train supervised on one shared dataset.
- **Process.** M20 "Committed": the placeholder dispositions (below). M23 "Committed
  (process)": every working branch is pushed after each commit and takes main by
  merge, never rebase. §14.4 adds, held open: our own next move as a head (Lc0's
  `policy_next`), the optimistic policy in self-play, contempt, H49, value-side Shaw
  vectors, LMR (moot); refused: ONNX/TensorRT export (for now), KataGo's
  convolution-era methods, symmetry augmentation beyond colour.

**The questions of 2026-10-04, answered** (the old "Waiting on the user" list):
- **Q-A** -> **H47**, option (a): the commit tier is a smoke test; the check that
  some ruleset is gated runs only in the tiers that run every oracle (nightly,
  release). Built on `lane/gate` - 2167092, the review's fixes in 7530e36 - not yet
  merged ("Now", below).
- **Q-B** -> **H48**: the naive searches' release budgets are raised and re-recorded
  twice before the proof-channel sign-off - not a §15.5 item; in step 2's
  quiet-machine block.
- **GPU-only cost figures** (issue 120) -> **M19**: counted FLOPs and CPU-labelled
  wall times are the untrained build's write-back; the GPU figures are named as
  deferred and measured on rented NVIDIA time before the pilot run; one bounded AMD
  attempt may come first (G9).
- **The committed-class placeholders and B7** (issue 121) -> **M20**: every
  placeholder but D8's readout form (`submit_head`, linear or pointer - not on M20's
  list; issue 109) disposed of, and B7's gate now names ablation 2's head-allocation
  evidence (row M20 under "Register status" says which dispositions change code).
- **A5** -> build behind its flag; the joint test with the window waits for the
  pilot run (M18).
- **D8's whole-turn token** (issue 92) -> none needed: the flat form supersedes the
  enumerating whole-turn arm and its 256 cap, and ends turns itself (D2, D8, §8.2).
- **The training rows** -> **M21** and §12.8's **I12-I20**: the training loop is
  written and smoke-tested on synthetic data - targets with §12.2's exclusions,
  losses, aux heads, PCR, the QAT hook, §12.8's mechanics - with no training; §15.5
  does not require it, the pilot run does. Part of it exists: the losses of today's
  heads and a smoke test of at most three optimiser steps on synthetic targets (N1+N2,
  N5; row M21). I1 (training itself) stays out; I7 and I8 stay not built (M5).
- Standing, unchanged: branch protection, the two oracle bug drafts, applying the
  proposed spec edits - most of which the new documents now carry (marked under
  "Proposed spec edits").

**What changes for the build:**
- The network's v1 changes - F5, F6, B15 (B6 removed), B8, A6, A7, D1, D3, D4, D8
  and M20's other dispositions; B11's five types behind its flag; B16 with the
  cached configuration. F1's 21,359,370 parameters, every seed-pinned test and N3's
  and N5's cost figures change with them.
- The C ABI grows append-only, each round with its Python mirror in the same commit:
  A6's counts and B11's edges (round 1); the search's value outputs, exploration and
  per-root-child records (round 2).
- D2 gains two arms after M22, and the whole-turn arm retires.
- E10 is redone to the adopted ranges (Hunt under 100 boards, the settling scans
  under 50, Guard offline).
- The training loop (M21) joins the untrained build, outside §15.5's "done means".
- M23: every working branch is pushed after each commit and takes main by merge,
  never rebase. Under that rule the work runs in three lanes - the main loop's plan
  ("Current state (2026-10-17)").
- Nothing enables the proof channel: Hunt's export, I14 and solved bounds stay off
  until the H23/H41 sign-off.

### Current state (2026-10-17)

**On main** (2026-10-17, evening): everything "On main" under "State on 2026-10-04"
below - the rules engine (E1-E5c, E7, H46); the search core (E6a: PUCT/MCGS, the ply
split and the whole-turn arm, the solver behind H41's `proof_channel` switch, off);
verification v3's components (E8+E9), none wired into the search; the network N1-N5;
C24 (off); the oracle harness; t0database - and, merged from the lanes, each reviewed:
gate step 1, H47 (c55ab40); F5 and F6 (8f71012); M22 measured (3d2e73a); B15 with B14's
score function (6ae00cd); A6's engine side, with Guard's five-thread line (e8de15e);
gate steps 2 and 3a - the engine configured in `gate.toml`, 113 known divergences
triaged again, H51's lenient capture markers with rulebook R11 (39bdf52); B11, B16 and
B17 (the merge after 39bdf52). The gate compares the engine in every tier: the commit
tier is a smoke test; nightly gates Standard - Half Reflected, Standard - Reversed
Royalty and Standard - Turn Zero (under the one-oracle rule gate step 3b replaces).
F1 is 22,961,802 parameters. GitHub Actions starts no job (the account's billing):
every check runs locally before each merge, nightly included.

**Now** - three lanes, the main loop's plan under M23's rule: each a git worktree
under `.claude/worktrees/` on its own branch, pushed after every commit, taking main
by merge, never rebase (re-check with `git branch -a -vv` and `git log
main..origin/lane/<lane>`):
- `lane/gate` - the harness - step 1 done: gate step 3b merged (H47 as amended at
  CHECKPOINT 8 - two oracles to gate a ruleset, the engine baseline
  `oracles/engine-baseline.json` written only by `harness.py baseline --write`,
  restricted runs never a tier verdict; the ftxi check-reading question drafted in
  `docs/checkpoints/upstream/` for the user to file). Next, in step 2: H48's
  re-recording on a quiet machine, then the release tier.
- `lane/engine` - C++ and the Python ABI mirror - the shared proof stores' data race
  under winpthreads' `std::shared_mutex` (found in A6; "Parked, and why"); next, ABI
  round 2 (D3's value outputs in the search, 6m's switch, E1/I19's exploration,
  I15/I2's per-child fields, a `legal_moves` option for `t5d_game_expand`, the
  draw/moves-left backup or its substitute), M22's further figures (CHECKPOINT 8), the
  D2 tree form, then the flat form.
- `lane/net` - the network and the training loop in Python - F6 in the heads (the
  ruling on issue 204) and M20's dispositions; next, A7 with its mirror test, A6's
  network inputs, the heads after engine round 2 (D3, D4's soft policy and reply
  prototype, D9, D7), C15's STE fix with G2's quantiser and G8's hook, A8, the D2 flat
  head, the training loop (M21), A5.

The main loop reviews each component and merges it into main, running the Lazy SMP
timing test first for any merge that touches the search or the provers ("The Lazy
SMP timing test log", below).

**The lanes' worktrees, and the one machine.** To recreate a lane on a fresh clone:
`git fetch origin`, then `git worktree add .claude/worktrees/<lane> lane/<lane>` (git
creates the local branch from `origin/lane/<lane>`) - or, for a lane never pushed,
`git worktree add -b lane/<lane> .claude/worktrees/<lane> origin/main`, pushed with
`git push -u origin lane/<lane>`. Link the main checkout's `.venv` into the worktree
as a directory junction (PowerShell: `New-Item -ItemType Junction -Path
.claude\worktrees\<lane>\.venv -Target .venv`), and build the engine in the worktree
(`py engine/build.py --test`): `gate.toml`'s `./build/engine/t5d-gate` resolves per
worktree. The gate worktree has no `oracles/.cache/` of its own: run the harness
there with `ORACLE_HARNESS_CACHE` set to the main checkout's `oracles/.cache` (its
reports then land there too). The lanes share `dev-r`: while a gate run, a golden recording, the
Lazy SMP timing test or any timing runs, the other lanes pause their heavy jobs
(pytest, about 8.5 min; the engine suite), and nothing under `oracles/` is edited in
the worktree whose gate is running (the gate would fail as harness-modified). The
cost of sharing shows in the gate lane's commit-tier run of 2026-10-17: calibration
1.44x, against 1.18x on the quiet machine ("Recovery", above).

**Number blocks**, so lanes never collide: spec issues - gate 122-139, engine
140-199, net 200-299, main loop 300 and up; proposed spec edits - gate 98-109,
engine 110-149, net 150-199, main loop 200 and up. Gaps between blocks are expected.

**Step 1 - CHECKPOINTS 6-7, by lane, in order** (where each lane stands: "Now"):
- **Gate:** gate step 1, H47; gate step 2 - the engine configured in
  `oracles/gate.toml`, the 113 known divergences triaged again (the 3 notation entries
  only with a notation-spec citation, else reported) - done when, with the engine
  configured, the commit and nightly tiers are green on a quiet machine, then nightly
  in CI (watched with `gh`): under H47 the commit tier is a smoke test that gates
  nothing, and the triaged entries include nightly-only sources (ftxi:naive, cwmtt,
  Chess in 5D). Later, in step 2: H48's re-recording, then the release tier.
- **Engine:** M22, the turn-width measurement. ABI round 1: A6's turn-state counts in
  the expansion (kept out of the truth sources), moves in the action, the count in the
  evaluation identity (EvalRequest, the evaluation cache key, MCGS identity - issue
  302), B11's pin and check edges behind the relation mask, the Python mirror in the
  same commit. ABI round 2: D3's value outputs in the search (WDL as the search value,
  the categorical for variance, the error head; one new deliver call carrying WDL, the
  categorical, the error, optional per-leaf moves-left and per-board logits), 6m's
  uncertainty-source switch, E1/I19's exploration (a total-concentration Dirichlet
  field appended beside `dirichlet_alpha`, shaping behind a flag, a root policy
  temperature), I15/I2's per-root-child fields (noise-free prior, value variance,
  boosts, forced and steered visits) and summary fields, a `legal_moves` option for
  `t5d_game_expand`, the draw/moves-left backup or a recorded substitute. Then D2's
  tree form - under the tree search only: the engine refuses it with MCGS (`graph`)
  as a parameter combination, parked and logged (issue 301; no document refuses it) -
  and the flat form (on the in-house hypercuboid search, the DFS kept bit-identical,
  after the gate is configured), each with its §15.2 row and §15.4's turn-structure
  invariants, retiring the enumerating whole-turn arm and its 256 cap.
- **Net:** F5 and F6; B15 replacing B6, with its no-table flag, and B14's eager
  FlexAttention score_mod with §15.2's B15 row; B16 (one owner), B17 behind its flag
  with its §15.2 row, B11's type set without travel-threat; M20's dispositions with
  A7 and its colour-swap mirror test (read as issue 304 says; frozen K/V keyed by
  perspective in the cached path, issue 303); A6's network inputs, after engine round
  1 is in main; the heads (D3, D4's soft policy and the reply prototype with its
  mapping test, D9 behind its flag, D7), after engine round 2; C15's STE fix
  (`fake_quantise` and `_frozen_kv_rounding`, with a gradient test - row C15), then
  G2's quantiser and G8's QAT hook; A8 behind its flag; the D2 flat head; the
  training loop, M21 (I15's records first, I2's targets, the new heads' losses beside
  today's, PCR with I20, I12 with 6u's and I6's arms, I13, I14's hook disabled, I16,
  I17, I18, I19's schedule, the QAT hook, the round-trip test); A5 last, dense scope
  only.
- **Any time:** G9's bounded AMD attempt, in `build/` only.

**Readings fixed for the lanes** (the main loop's, so a fresh session keeps them):
- A6's moves-made count in the evaluation identity (issue 302); A7's frozen K/V keyed
  by perspective (issue 303); §15.2's A7 row read with the acting colour equalised
  (issue 304); one turn unit, owned by the main loop (issue 305); no root-scan reserve
  while Guard is offline (issue 306).
- B16 at refreshes: the refresh stays as built (M20: intrinsic features, a refresh
  giving the anchor's values); B16 biases frozen keys at nodes, history boards
  included, not inside the refresh pass; §4.1's wording is the net lane's to log.
- Forced playouts: pruning at target construction (§12.5) is required; search-side
  forcing is behind a flag, off, or not built.
- The tree form's fixed board order (§8.2) is a separate `board_order` parameter, free
  order by default, outside §15.2's tree-form row; ftxi's fine tree, its prior art,
  is an oracle's - to understand, never to port (AGENTS.md).
- 6j's categorical-target arm (the root search value against the short-term value)
  is a switch in M21's target builder.
- Mish in the heads is not decided by F6: the value head's MLP keeps GELU, a
  placeholder (`lane/net`'s issue 204).
- D2-flat's per-board mandatory flag goes into ABI round 1's expansion; the net lane's
  D2 flat head waits for the engine's flat form in main.

**Lane stop conditions** (stop and report - AGENTS.md and the brief):
- Gate: a notation known divergence that cannot cite the notation specification -
  report those entries, keep the rest; a capture in any of many2's 16 R1 positions
  that falls outside CHECKPOINT 1's RQ1-RQ4 (no CHECKPOINT 1 analysis covers them).
- H48: no practicable naive budget makes every proof-channel check non-vacuous within
  a CI job's 240 minutes (several positions alone take over 300 s) - the documents do
  not say what follows.
- G9: any driver, Windows-security or compiler change on the machine needs the user.

**Step 2** - each lane's items start when its step 1 is done; only those under "Then,
every lane" wait for all three:
- Engine: E10 redone (Hunt under 100 boards and the settling scans under 50, both on
  by default; the asset map; Guard's hook off; the audit lane's tooling); E11, time
  management (E7 with 6o's arms; no root-scan reserve while Guard is offline - issue
  306), and E8 behind its flag; D4's per-square label generator; the missing tests -
  the solver against df-pn and, through the gate (gate lane) or the audit lane, the
  oracles' mate verdicts (H41); TT churn.
- Net: G2's FP8 emulation; the missing test cached = dense on every real variant.
- Then, every lane: §15.2's suite re-run on the finished v1 network; the
  quiet-machine block, alone on the machine (N3's wall timings, FFN 2x and 4x, the
  displacement-bias, Shaw, layout-embedding and normalisation overheads, legality
  classification, Guard's and Hunt's nodes/s, H48's re-recording - before the
  sign-off, not a §15.5 item); the release tier with the engine configured, after
  H48 - the gate evidence for the sign-off (H23); the write-backs (M4); the
  committed-class placeholders consolidated for the user, with the main loop's issues
  300-306.

Then the report for sign-off, the proof channel disabled.

**Parked, with the reason** (not built in the untrained build): I10's calibration
monitoring - it measures nothing on random weights (the pilot run); C3's
conditioning-offset training hook - with the deferred cached configuration (C2);
6p's other arms - after the pilot run (M24); MLH steering (I3's match play) - it
steers by a trained moves-left head once the outcome is settled: it waits on D4's
moves-left head (net lane, after engine round 2) and round 2's per-leaf moves-left,
and "settled" has no rule (issue 300); resumed behind a flag, off, once those are in
main and issue 300 is ruled. Early adjudication (§12.3) has no specification (issue
300) and is not built.

**Remaining to §15.5** - the three lanes in parallel, each in its order:
1. Gate: H47 reviewed and merged (gate step 1); the engine configured, the known
   divergences triaged again, nightly green with the engine configured - quiet
   machine, then CI (gate step 2); in step 2, the solver's proofs against the
   oracles' mate verdicts.
2. Engine: M22 reviewed and merged; ABI round 1; ABI round 2; D2's tree and flat
   forms with their §15.2 rows and §15.4's invariants, the whole-turn arm retired;
   then step 2's E10, E11 and E8, D4's label generator, the solver against df-pn, TT
   churn.
3. Net: F5 and F6 reviewed and merged; then in the order above to the D2 flat head
   (A6's inputs after engine round 1, the heads after round 2); A5 last; then step 2's
   G2 emulation and cached = dense on every real variant. M21's training loop sits in
   this order but outside §15.5's "done means" - the pilot run needs it.
4. When every lane is done: §15.2 re-run on the v1 network; the quiet-machine block
   (H48's re-recording in it - before the sign-off, not a §15.5 item); the release
   tier with the engine configured; the write-backs (M4); the committed-class
   placeholders consolidated for the user, with issues 300-306.
5. The report for sign-off, the proof channel disabled (H23, H41).

**Waiting on the user:** nothing that blocks (the `gh` login is done and the CI
comparison recorded in "Recovery", above); the standing items - branch protection with
the workflows as required checks; filing the two oracle bug drafts in
`docs/checkpoints/upstream/` (and, after gate step 3b, the ftxi question of issue
126); applying the proposed spec edits upstream. **GitHub Actions has started no job
since about 09:56 UTC on 2026-10-17** - every run, on main and on every lane, fails at
once with "The job was not started because recent account payments have failed or
your spending limit needs to be increased" (the account's 'Billing & plans'): until
it is fixed nothing is confirmed on Linux, and the main loop runs every check
locally, the nightly tier included, before each merge. **CLAUDE.md's table still says
"the user's rulings R1-R10"** - the rulebook now has R11 (H51); the instruction files
are the user's to edit. Ruled at CHECKPOINT 8: spec issues 122-126, 142 and 204
("CHECKPOINT 8", above). Put to the user, none blocking: 129 (the measurements
pinned before H51 - 6b, 6b′, 6b″, Hunt's truth - keep the reading they were made
under, a written x must capture, so they reproduce; proposed edit 101), 130 (how a
divergence settled by a project decision is recorded upstream - KD-0170 is neither
a defect nor a disagreement with a documented rule; proposed edit 102), 217 (merged with B11-B17 - B17's "exact near classes": two readings, each 1,493 classes, the built one a placeholder; proposed edit 161), two lines the gate lane proposes for AGENTS.md (never edit `oracles/engine-baseline.json` or its `[baseline]` pin by hand; never record a LOSS with `baseline --write` to make a gate run pass - the H47 row), 205
(merged with B15 - under the far-bin placeholder B6 is not B15's special case, as
§4.1, register B6 and reversal 25 say it is: keep the sentence and give the far
classes B6's cells, about 5.2M parameters, or keep §4.1's ~1.6M and drop the
sentence, proposed edit 159). The main loop's issues 300-306 go to the user
with step 2's consolidated list; nothing in step 1 waits on them.

### State on 2026-10-04 (before the loss)

*"Current state (2026-10-17)" above supersedes its held, running, remaining and
waiting lists: the branch it holds (61e9aa8) and the E10 it has running were lost
with the machine ("Recovery", above) and are redone in the plan.*

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
  rewrite pushed history; every working branch is pushed after each commit and
  takes main by merge, never rebase (M23). No absolute user paths in tracked files.
  **Never edit the spec or the register** - propose edits below; they are applied
  upstream.
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
| 2026-10-17 | A6 (the prover reports a searching thread's line) | main a8a5920 + lane/engine 3f57416, merged tree before the commit | dev-r, 24 logical; the gate and network lanes running; load 9-11% before, 20-22% after | 32 | 293 | 311 / 311 / 311 | none | pass x3 |
| 2026-10-17 | the gate lane: the engine configured, H51 (the reader; the 6b tools read as measured) | main e8de15e + lane/gate f5b6330, merged tree before the commit | dev-r, 24 logical; the network and engine lanes running; load 8-10% before, 8-12% after | 32 | 293 | 313 / 312 / 313 | none | pass x3 |
| 2026-10-17 | the stores' lock (`ShardLock`; before review) | lane/engine e8de15e + the stores' lock, the tree before the commit, inside its full `--werror --test` run | dev-r, 24 logical; other lanes running; load 27% before the build | 32 | 292 | 304 | none | pass |
| 2026-10-17 | the stores' lock (`ShardLock`; before review) | the same tree, the test alone | dev-r, 24 logical; another lane's Python job on about 12 cores and this branch's long tests on 1; load 100% before, 99% after | 32 | 293 / 292 / 292 | 308 / 312 / 309 | none | pass x3 |
| 2026-10-22 | the stores' lock (`ShardLock`; completed) | lane/engine e8de15e + the stores' lock, the tree before the commit, inside its full `--werror --test` run | dev-r, 24 logical; other lanes running; load not recorded | 32 | 291 | 313 | none | pass |
| 2026-10-22 | the stores' lock (`ShardLock`; completed) | the same tree, the test alone | dev-r, 24 logical; another lane's Python job running; load 94% before, 96% after | 32 | 292 / 292 / 293 | 308 / 311 / 309 | none | pass x3 |
| 2026-10-22 | the stores' lock (`ShardLock`; reviewed, 9c3c8c8) | main f18761f + lane/engine 9c3c8c8, merged tree before the commit | dev-r, 24 logical; the network lane running; load 51-53% before and after | 32 | 293 | 308 / 309 / 309 | none | pass x3 |

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
- **The notation finding** (ruled at CHECKPOINT 8 - H51, R11: the reader is lenient
  now, KD-0167 to KD-0169 are withdrawn and ftxi's refusal is KD-0170; "H51", below).
  cwmtt/futures action 17, `(L1T5)Qf4>>x(L0T5)e4~`: at
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
- **H51 - capture markers read leniently (2026-10-17; the user's CHECKPOINT 8 ruling on
  spec issue 125; register H51, rulebook R11).** *The reader*
  (`PgnOptions::capture_marks`, `CaptureMarks::Lenient` by default): a move is read as
  its squares and boards define it, whether or not `x` is written - the mark selects no
  move - and a mismatch either way is a warning in the reader's channel
  (`PgnReplay::warnings()`), which `t5d-gate` prints on stderr beside the annotation and
  turn-number warnings: the engine contract (the harness's manual) has no warning field,
  and nothing in it changed. The reading before H51 (`MarkedMustCapture`) and the letter
  both ways (`Strict`) stay behind the option (`--pgn capture_marks=...`; the old
  `require_capture_marks=true|false` keeps its meaning). *The writers* - two: the
  engine's `synthetic::pgn_move` (t5d-play's records and the synthetic ones;
  `to_string(Move)` is the export form, which has no mark) and N5's, in Python,
  `netplay.pgn_move` (`python/t5d/engine/netplay.py`: the network's self-play and match
  records, the same long form - left out of this survey at first, a review finding;
  *After the review*, below). `data/t0database/to_5dpgn.py` writes no move of its own: it
  copies the human games' text, marks and all. Both write `x` exactly on captures:
  `synthetic_moves_mark_exactly_the_captures_h51` writes every
  move offered at every position of 53 random games (9 variants under the default rules;
  every alternative rule option in Two Timelines) and judges each from the boards - the
  opponent's pieces on its target board against those on the board it makes there:
  14,081 physical captures (24 en passant), 6,293 by non-branching jumps, 12,192 by
  branches, and 408,383 quiet moves of those kinds, each kind required to occur; the
  records replay with no warning by default and to their end under the strict reading,
  as t5d-play's records do (`check_record`). *The corpus*, parse mode, this build
  against the parent's: one record reads further - cwmtt/futures, to its end (19
  actions, from 16); its action 17 is the corpus's only `x` on a move that takes
  nothing; ten records write captures without `x` and are read as before, now with a
  warning for each (100_timelines 17 moves, ctp2 33, exiledKings 4, manyChecks 39,
  niceAndrey 91, tesseractMageOChicken 3, ftxi/verysmall 1, notation/aivsai 2,
  notation/game-2 1; many2 as the gate converts it, not counted). *The rulebook*: R11,
  labelled a project decision (H51), with a conventions line and source [141]; the diff
  reviewed - no other entry changed - and re-pinned (`08fddc5e...` to `39e9ad58...`).
  *The known divergences*: the commit tier, run with the reader changed and the entries
  not yet, failed as expected - KD-0167 to KD-0169 as telltales (the engine now plays
  action 17 as cwmtt, Chess in 5D and the notation parser do) and `[divergence]
  cwmtt/futures action 17: legality: engine=accepted vs ftxi=rejected:no_matching_move
  (soft: may be notation, not rules)`. The three were withdrawn on the ruling's
  authority, as their header said; KD-0170 records ftxi's refusal, citing R11 and H51.
  Triaged by the rulebook, never by vote: at position 16 every source holds one state;
  ftxi's own listing of its legal actions there (`observe --source ftxi --record
  cwmtt/futures --actions 16`: 128 actions) holds the queen's jump (+1T5)f4 -> (0T5)e4,
  and its refusal parses the move with the capture flag set (`Invalid move: ... cp:yes
  ...`): it refuses the mark, not the move. Upstream: unreported - neither a defect nor
  a disagreement with a documented rule (spec issue 130). The nightly tier, run after
  the entries changed, found nothing more: at cwmtt/futures' positions 17 to 19,
  compared with the engine for the first time, the engine agrees with cwmtt and Chess in
  5D wherever each answers - Chess in 5D gives the check verdicts and calls checkmate at
  19 (it gives no counts); cwmtt gives the counts (no check verdict) and at 19 reports no
  legal action, `no_action`, which does not say checkmate or stalemate. So the mate at
  19 is the engine's and Chess in 5D's call alone (corrected in review: this read "as
  both say").
  *Runs* (the machine shared with other lanes; timings are not measurements): `py
  engine/build.py --werror --test` 550 of 550; `rulebook` ok after the pin; `verify` ok
  (H48's 17 notes); `selftest` 638 tests OK, 1 skipped; `gate --tier commit`: oracle
  layer ok (ftxi 8256 of 8256, notation 3012 of 3012), `engine differential: SMOKE
  TEST - PASSED`, calibration 1.54x; Standard - Turn Zero "no oracle followed
  cwmtt/futures, cwmtt/tesseractMageOChicken as far as the engine went" (ftxi stops at
  futures' action 17; the notation parser stands behind no replay). `gate --tier
  nightly`: oracle layer ok - every source's required set reproduced (ftxi 8256 of 8256,
  ftxi:naive 6556 of 6556, cwmtt 6272 of 6272, cwmtt:naive 4755 of 4755, Chess in 5D
  6461 of 6461, notation 3012 of 3012; calibration 1.47x); `engine differential:
  PASSED`, no finding; **gated: Standard - Half Reflected** (1 record, 21 positions, at
  least 2 oracles), **Standard - Reversed Royalty** (1 record, 122 positions, at least 3)
  **and Standard - Turn Zero** - newly, now that the engine reads cwmtt/futures to its
  end (7 records, 144 positions; check answered at 138, count and mate at 143; at least
  1 oracle). Behind each Turn Zero record, as the differential counts them (a scratch
  script over the run's report, not kept): cwmtt/100_timelines 3 (ftxi, cwmtt, Chess in
  5D), exiledKings 3, futures 2 (cwmtt, Chess in 5D - ftxi refuses action 17, KD-0170),
  small 3, tesseractMageOChicken 1 (cwmtt - ftxi stops at action 28 on the `+1` label,
  KD-0054, and Chess in 5D cannot read action 25's notation), ftxi/futures 3,
  notation/1.2 2 (ftxi, Chess in 5D). So the gate as built gates Turn Zero with one
  oracle behind tesseractMageOChicken; under the two-oracle rule of spec issue 122 (the
  next component) it would not gate Turn Zero - tesseractMageOChicken alone keeps it
  out - and would still gate Half Reflected and Reversed Royalty. NOT gated, and why, as
  before: Standard (ftxi/softmate unreadable, KD-0047), Very Small - Open (cwmtt/ctp1,
  ftxi/verysmall), the custom positions (ftxi/niceAndrey's set-up), Misc - Small (no
  oracle follows notation/game), the Focused variants and Timeline Battleground (no
  check verdict), Timeline Invasion and Marauders (not set up by the engine). The proof
  channel stays disabled. *t0database* (M7, M22; `py data/t0database/to_5dpgn.py
  build/t0db`, 761 records): the parent commit's `t5d-measure` and this one's give
  byte-identical output for M22's `--turns --turn-summary` and M7's `--moves-after-shift
  --summary`, with and without `--legal` - no record and no action changes (716 replay
  to the end; the gate adapter's end events identical too). No t0database record writes
  `x` on a move that takes nothing; 661 write captures without `x` - 8,034 moves - read
  as before, now with a warning each. `docs/MEASUREMENTS.md` unchanged (the engine lane
  re-measures M22 after the merge). *6b's measurements*: kept reproducible by reading
  with the reading they were measured under (spec issue 129; engine README, "Hunt's
  ground truth", *After H51*): 6b″'s tool accepts the pinned truth (825 roots), 6b's and
  6b′'s tools list the same 825 positions as the parent's build; the code signature
  moved (parent sha256:02859783..., this build sha256:21175476...); the truth's
  provenance records none, so the 6b″ tool reports it "unrecorded" beside this build's,
  as before. Not re-recorded, not re-run.
  *After the review* (2026-10-17; the review's findings on a77f635, fixed; the same
  machine, as loaded). **N5's writer**: `netplay.pgn_move` writes `x` from the move's
  capture flag, as `synthetic::pgn_move` does, and nothing tested it against the boards;
  and H51 had taken away the only check its records had - before H51 the gate refused a
  record with an `x` on a move that takes nothing, since then it reads it, with a
  warning on stderr that `replay_with_gate` never read (`Game.from_pgn` reads with the
  defaults and has no warning channel). Now `Replay.warnings` keeps the gate's reader
  warnings and `check_replay` reports each - a record N5 writes must replay with none,
  as t5d-play's must (`check_record`) - and `test_the_writer_marks_exactly_the_captures`
  (`python/tests/engine/test_netplay.py`) judges `netplay.pgn_move` from the boards as
  the C++ test judges its writer: every move offered at every position of 10 random
  pseudo-legal games (5 variants, optional moves included) and at two en passant
  positions - 420 captures on their boards, 3 en passant, 262 by non-branching jumps,
  386 by branches, and 12,831 quiet moves of those kinds, each kind required;
  `test_a_capture_mark_that_disagrees_is_a_problem`: an `x` added to a quiet move, or
  dropped from a capture, replays `ok` with one warning, which `check_replay` reports.
  The reviewers' slip - the capture mask also matching double steps - now fails the
  replay of `test_a_game_replays_legal`'s game (`(0T3)Pf2xf4` reported), and the
  board-judged test fails under it, under a dropped en passant mark and under a dropped
  branch-capture mark. **6b's tools**: no test ran `t5d-scan6b` or `t5d-scan6bp`, which
  read their records in loops of their own; `sixb_tools_read_records_as_6b_read_them`
  runs both on a record with an `x` on a jump that takes nothing: each lists
  `sixb_roots`' three roots (index and fingerprint) and stops where it stops; with
  either loop reading with the default (a mutation, reverted) it fails. **R11**: its
  source sentence had the specification call every token a writer adds "purely for
  better human comprehension"; the specification says it of the present-movement and
  created-timeline tokens only (T15's scope), and R11 now quotes it so - the ruling
  unchanged, KD-0170's citation unaffected; reviewed and re-pinned (`39e9ad58...` to
  `ab01119e...`). **The reader's warnings**: `PgnReplay::warnings()` and the engine
  README said each names the action and the move; a turn number's gives only its line
  and column - both now say so. **CI**: no run has started on `lane/gate` from a5e61d9
  on - a5e61d9's own runs (39014207162, -168, -174), the nightly dispatch 39014227471,
  the merge's (39019219551, -579, -605) and a77f635's (39022990526, -527, -538) were
  each "not started because recent account payments have failed or your spending limit
  needs to be increased" (GitHub's annotation; main's runs likewise). CI has not
  confirmed H51 on Linux, and the nightly dispatch above is still owed once billing is
  restored. **Stale elsewhere, recorded**: spec issue 127's measurement (a two-oracle
  rule ungating nothing nightly gates) predates H51 - it now says so - and H23's row
  still names nightly's two gated rulesets (left to the issue-122 component, which
  changes the gating; H23 is not among this component's rows); `CLAUDE.md`'s table still
  says the rulebook holds "the user's rulings R1-R10" - R1-R11 now; the instruction
  files are the user's to change, so that edit is put to the user (H38's row). *Runs*
  (timings are not measurements): `py engine/build.py --werror --test` 551 of 551;
  `T5D_REQUIRE_ENGINE=1 pytest tests/engine` 166 passed, 1 skipped (`test_netplay.py` 17
  of them); ruff check and format clean; `rulebook` ok after the pin; `verify` ok (H48's
  17 notes); `selftest` 638 tests OK, 1 skipped; `gate --tier commit`: oracle layer ok
  (ftxi 8256 of 8256, notation 3012 of 3012), `engine differential: SMOKE TEST -
  PASSED`, calibration 1.23x; `gate --tier nightly`: oracle layer ok - every source's
  required set reproduced (ftxi 8256 of 8256, ftxi:naive 6556 of 6556, cwmtt 6272 of
  6272, cwmtt:naive 4755 of 4755, Chess in 5D 6461 of 6461, notation 3012 of 3012;
  calibration 1.22x); `engine differential: PASSED`, no finding; gated, as before:
  **Standard - Half Reflected** (1 record, 21 positions, at least 2 oracles),
  **Standard - Reversed Royalty** (1 record, 122 positions, at least 3) and **Standard -
  Turn Zero** (7 records, 144 positions; check answered at 138, count and mate at 143; at
  least 1 oracle). Behind each Turn Zero record, by the differential's own functions
  over the run's report (a scratch script, not kept): 100_timelines 3, exiledKings 3,
  futures 2 (cwmtt, Chess in 5D), small 3, tesseractMageOChicken 1 (cwmtt), ftxi/futures
  3, notation/1.2 2 - so under the two-oracle rule of issue 122 Turn Zero would not be
  gated, tesseractMageOChicken alone keeping it out, and Half Reflected (cwmtt/test1:
  cwmtt and Chess in 5D) and Reversed Royalty (3) would; the proof channel stays
  disabled; t0database: this build's `t5d-measure` and the parent commit's (cb7af77)
  byte-identical, standard output and error, for `--turns --turn-summary` and
  `--moves-after-shift --summary`, each with and without `--legal` (716 of 761 records
  to their end in each); 6b's and 6b′'s tools list the same 825 positions as the
  parent's build, and 6b″'s tool accepts the pinned truth. The code signature moved
  again with the reader's header comment: this build's sha256:3bc1d730...; the truth's
  still unrecorded, as before.

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

**M22 - the turn width, measured (2026-10-17).** The last Phase 1 count, recorded
before the D2 tree and flat forms are built (§15.1; register M22) - the ply split,
D2's default arm, is built already (the D2 row below); 2864be0's message, "before
any D2 turn-construction arm is built", should have named the two turn-start forms:
`t5d-measure --turns
--turn-summary` over t0database (716 records replay, 692 games), the default rules,
parse mode; every action replayed move by move and checked against the replay; the
summary recomputed independently. Figures, definitions and the part L check in
`docs/MEASUREMENTS.md` ("Turn width (M22)"). Reviewed the same day: an optional move
now counts a jump from a mandatory board onto an optional one (T3, J4 - the move is
made on both boards), the summary gives the turns with more than one playable board
(where the arms differ) and with an optional board, its ranges, classes and
statistics live in `tools/turn_width.hpp` and are tested at their edges, and the D2
reading counts width at the turn's start, not by the moves played. The tool's
per-action counter (`tools/turn_width.hpp`, `TurnCounter`) is outside Hunt's truth
sources and can serve 6x's reports "by mandatory-board count". Open: how the figures
order the D2 work (issue 142).

### 6. Phases 2-3 - in progress (see "Current state")

## Register status

### H - rules engine, gate, verification layer

| ID | Status | Notes |
|---|---|---|
| H1 | built (v3's components), tested; not wired | Verification v3 (§11.4): exact rules (E1-E4); the search's solver behind H41's switch (E6a); Guard, Hunt and the asset map as components (E9, reviewed); df-pn settling (E8's prover, carried into E9); the audit lane's certificate checker and truth recorder. E5b/E5c's brute-force scanner kept as 6b′'s comparison. 6b″ decided (M17, now in the register): Hunt adopted under 100 boards, Guard in no range. Nothing is wired into the search on main: E10 (Hunt under 100 boards, the settling scans under 50, the asset map, the audit lane's tooling) was lost with the machine and is redone in step 2 |
| H2 | built, tested | Proofs only from exhaustive verdicts and exhaustive defender enumerations; the per-leaf estimate channel removed (E9); estimates survive only as Hunt's flagged priors and the asset map's boosts; the search consumes in-tree bounds only behind H41's switch |
| H3 | built, tested | Node cap with iterative deepening in forcing half-turns; horizon per budget measured (6b) |
| H4 | built | MVV-LVA, killers, history, continuation history |
| H5 | built | Nothing speculative; null move and LMR not built |
| H6 | built (v1 material); NNUE held open | Material count with placeholder 5D piece values (spec issue 56; now register H50), used by the asset map and the material evaluator; the scanner NNUE is held open (CHECKPOINT 4) |
| H7 | not built (rule) | NNUE training: held open with the NNUE (H6); no training runs |
| H8, H9, H20, H21 | not built (rule) | Held open / refused |
| H10 | reduced, built | Rung 1 dropped (6b); rungs 3 and 5 retired, not built (M5); rung 2's promotion side is Guard, its demotion audit the audit lane; rung 4, the asset map, built (E9). The reduced ladder as H14 restates it: Hunt, the asset map and the settling scans under 50 boards - Guard offline (adopted nowhere, 6b″) |
| H11 | superseded | By Guard (promotion) and the audit lane (demotion audit) |
| H12 | retired | With the NNUE and the per-node scans |
| H13 | built (component), not wired | The asset map (E9) as the register now records it: the material estimate, 4 plies and 128 nodes per board, flagged at 300, boost 0.25 × min(1, value/900), every number a placeholder (issue 106). Wiring its boosts into the search's priors was E10's - lost, redone in step 2 (with the steered visits flagged for I2) |
| H14 | not started (lost with E10) | Restated at CHECKPOINT 6: budgets for the reduced ladder - Hunt, the asset map, the settling scans under 50 boards (Guard offline) - against shared movegen, on M13's five cores. E10's scheduler, redone in step 2 |
| H41 | built, tested | `SearchParams::proof_channel` (C `proof_channel`, `t5d-play --proof-channel`) - the switch the register now names: propagation of solved status (marking nodes solved, pruning siblings, secure-child selection) and export, off by rule; game ends are scored as results. Not yet: the cross-checks of the solver's proofs against df-pn and the oracles' mate verdicts (§15.4) - step 2's missing tests |
| H42 | built, tested; not adopted | Guard (E9): depth 1 completes in 65.1 / 15.8 / 5.2 / 0.0% of roots against 95% (6b″) - adopted in no range; §13: the chosen move goes unguarded. Built and kept (the audit lane's shallow-mate report uses its question offline; the search's re-search hook stays unwired-by-adoption). Its five-thread depth-1 test failed once in CI (run 37146667584): a thread that read the root's proof from a helper's export - its tree empty - was the one reported, so the refutation's line was empty. Fixed in the prover's report (2026-10-17, `lane/engine`): the line of a thread that searched, complete at depth 1; the test pins the race and checks the line's contract at depth 2. What Guard decides is unchanged. Beyond depth 1 the line keeps its contract (it can stop where a node was solved from the store): a certificate holds the attacker's proving actions only, so the line could take the attacker's step past such a node from one but not the defender's longest defence, which needs an enumeration of the defender's actions (the first record of this fix gave another reason - that 6b″ ran Guard without certificates; it ran with them, for its cross-check against E4) |
| H43 | built, tested; adopted under 100 boards; not wired | Hunt (E9): 69.0% / 63.6% of the truth's mates under 50 / 50-99 boards (6b″) - adopted there; 35.0% at 100-199 and 0 of 1 at 200+ - not adopted; §13: proofs there come only from the search's solver. Export refused until H41/H23. Not wired into the search on main: E10 was lost; redone in step 2 (Hunt at the root under 100 boards, the measured configuration) |
| H44 | built in part | The audit lane's tools: the certificate checker (E4 alone) and `t5d-hunt-truth` (E9); the decision sampler, the offline proof runner and the per-range report were E10's - lost, redone in step 2 (curriculum hooks as interfaces only) |
| H45 | not built (rule) | The adversarial league (M5) |
| H46 | built, tested | Q2 (provisional): initial timelines count as created by neither player - a variant's set-up reads every timeline as initial (rulebook R10) |
| H47 | built, tested | Tiered CI reading (Q-A at CHECKPOINT 6, amended at CHECKPOINT 8; spec issues 122-124 ruled, built 2026-10-17). **Two oracles to gate a ruleset**: every record of a ruleset followed, as far as the engine went, by at least two independent implementations (`gate.GATING_ORACLES`; two searches of one count once, the notation parser never), in every tier - so the commit tier, with one rules oracle, can gate nothing at all. **The check that some ruleset is gated** (`engine-nothing-gated`, and `GateResult.ok`'s test of it) runs only in a tier that runs every oracle search over the whole corpus (`Tier.full_strength`: nightly, release) - keyed to what a tier runs, never its name; any other tier - commit - is a smoke test, and its summary (`engine differential: SMOKE TEST - ...`) and report (`smoke_test`; nothing under `gated`) say that it gates nothing. **The commit tier's last-green-run rule**: the engine's baseline, `oracles/engine-baseline.json` - written by `harness.py baseline --write` from a green run of the commit tier, pinned in `gate.toml` `[baseline]`, checked by `verify` and before every comparison; recorded 2026-10-17 from the engine as built on `lane/gate`, 35 of the 41 commit records replayed - and every tier fails an engine that declines a record it read then (`engine-declined`) or reads none (`engine-read-nothing`). **Restricted runs** (`--source`/`--record`) never produce a tier verdict: labelled "restricted - not a tier result", findings only, the exit code from them, no check of a tier as a whole - undoing issue 124's first reading. In every tier an unexplained divergence, a telltale or a contract breach fails the engine; `gate.toml` is refused if a tier that is not full strength sets `require_engine`; the release tier's named-variant rule, the proof-channel line and parse mode are unchanged for a complete run - a restricted run's proof channel reads `DISABLED - restricted - not a tier result`, and it makes neither parse mode's `engine-nothing-compared` floor nor the release tier's proof-channel readiness check (`proof-channel-vacuous`; review fix 2026-10-22). The baseline (review fixes 2026-10-22): `baseline --write` reads the baseline it replaces verified against its pin and records nothing over one that is not the one pinned, so that a hand edit cannot empty its `LOSS` list; `verify` fails a baseline that holds no line for a record of its tier; like MANIFEST's, the pin catches an accident, not the file and its pin edited together - review's job (H33). `AGENTS.md` names neither the baseline nor its pin: its "Never make the gate pass by ..." list should add "editing `oracles/engine-baseline.json` or its `gate.toml` `[baseline]` pin by hand, or recording a `LOSS` with `baseline --write`", and "Things you must not touch" the file and its pin, written only by `baseline --write` - the instruction files are the user's to change, so the edit is put to the user (as H38's for `CLAUDE.md`). Run 2026-10-17 with the engine configured: commit SMOKE TEST - PASSED (baseline: 41 records compared, 0 declined); nightly PASSED, gating Standard - Half Reflected (2 oracles behind its record) and Standard - Reversed Royalty (3), and not Standard - Turn Zero (one oracle behind cwmtt/tesseractMageOChicken); a restricted run (`gate --tier nightly --source ftxi --record cwmtt/standard`) labelled "restricted - not a tier result", 0 findings. Spec issues 122-124, 131-133; proposed edits 103-104 |
| H48 | not started | Before the proof-channel sign-off: the naive searches' release budgets raised and the golden files re-recorded twice, on a quiet machine, so that no proof-channel check is vacuous (`verify`'s 17 `proof-channel-vacuous` notes) |
| H15 | built, tested | The scanner's deterministic single-thread mode; the search's synchronous mode and digest (E6a; the digest in the C API) |
| H16, H25, H29 | not built (rule) | Refused |
| H17 | built in part | Lazy SMP in the scanner (E5c): same-root threads, a shared TT and a concurrent solved store; in-game decay and rung helpers wait for the search; no help for exhaustive refutations (measured on Guard, now in the register). **The shared stores' data race on Windows - fixed 2026-10-22** (`lane/engine`; found at E-A6's review, where `scanner_lazy_smp_store_refuses_contradictions` segfaulted in 11 of 600 runs on the development machine - MSYS2 ucrt64 g++ 16.2, Release; gdb: `same_position` under `ConcurrentSolvedStore::put`). The cause, confirmed in winpthreads' source and by experiment: libstdc++'s `std::shared_mutex` is a `pthread_rwlock_t` with the static initialiser; winpthreads initialises it on its first call and returns `EINVAL` to the threads that lose that race (8 threads on 64 fresh locks: 149,878 and 156,542 of 1,024,000 `pthread_rwlock_wrlock` calls, in two runs; `pthread_mutex_lock` never failed); libstdc++ ignores the error - so a losing thread ran its critical section without the lock, then unlocked a lock it did not hold. Every store the scanner's and the prover's threads share is first taken by all of them at once (a scanner's own; Guard's and Hunt's, built per call; the disproof stores). Fixed by construction: both stores' shards are behind `ShardLock` (`t5d/scanner.hpp`), a `std::mutex` - libstdc++'s `lock()` throws on any error the platform reports, never returning without the lock. A warm-up (each shard's lock taken once at construction; 0 crashes in 600 runs at E-A6's review) was not chosen: it rests on winpthreads' internals, not on the lock's contract, and keeps the slow lock. The audit: no other `std::shared_mutex`, `shared_lock` or `shared_timed_mutex` in `engine/`; the TTs are lockless; MCGS and the evaluation cache have no lock and are single-threaded (H15) - `t5d_search_new_sharing`, `EvalCache` and the Python `Search` now say that searches sharing a cache are used one at a time. Tests (`test_shared_stores.cpp`): `shared_store_lock_excludes_at_first_concurrent_use` and `shared_stores_hold_at_first_concurrent_use` - built with the old lock (`e8de15e` with the new test file, in a scratch copy) each failed in every run - 120 of 120 on 2026-10-17, 50 of 50 on 2026-10-22 (the lock test with 136-1,003 overlaps a run over 2026-10-17 and 2026-10-22, the store test by crashing) - and with `ShardLock` neither failed in 500 (2026-10-17) or 300 (2026-10-22) runs; `scanner_lazy_smp_store_refuses_contradictions` failed with the old stores in 14 of 600 runs on 2026-10-17 (10 crashes, 3 stores holding a position twice, 1 lost distance) and 85 of 600 on 2026-10-22 (58 crashes), and with the new in 0 of 1,000 each day. The cost, indicative (a shared machine; not in MEASUREMENTS): winpthreads' reader-writer lock takes about 1.4 µs a call and a process-wide spinlock, so the mutex is far faster here - store operations about 25× as many at 1 thread and 450-1,000× as many at 5 and 12; at 5 threads Hunt 1.45-1.69×, Guard 2.71-2.90× and the scanner 1.22-1.34× the nodes per second (two days; at 1 thread 0.94-1.07×, noise, and the same nodes and proofs as the old build). `scanner.hpp`, `scanner.cpp`, `prover.hpp` and `prover.cpp` are truth sources: the code signature changes, no verdict logic (M17's row; issue 147). Linux (glibc) not yet run: the engine job (ubuntu-24.04, g++ 13/14) runs the tests on every push, but no GitHub Actions job has started on any branch since 2026-10-17 09:56 UTC, the nightly of 2026-10-22 included ("recent account payments have failed or your spending limit needs to be increased") - re-run the engine workflow once billing is fixed. Issue 146, edit 120; the old lock also leaked Windows handles (about 2.75 per lock used; review of 0e4b6c5) |
| H18 | built, tested | 5D Zobrist: coordinate-free content hash per board, multiply-rotate with (timeline, half-turn), addition across boards, ruleset ID, player to move, parity, creators, board size; incremental equals from-scratch; duplicated boards never cancel (engine E1, E2) |
| H19 | built, tested | Proofs export to the solved store at once; the TT holds estimates only; class-relative disproofs live in their own store, never the solved store (E8, E9 - now in the register) |
| H22 | built, tested | All five items of §11.0 (E1, E2, E4); the register's wording now matches what is built (proposed edit 40, applied upstream) |
| H23 | built, tested | The gate, oracle and engine sides. The engine is configured in `gate.toml` (`./build/engine/t5d-gate`, full mode, 2026-10-17; the root's build on every platform, with a time limit of its own, `[engine] timeout_s`) and compared in every tier: commit, a smoke test (H47), PASSED, locally and in CI (run 39009070712); nightly PASSED locally (2026-10-17, under the two-oracle rule), gating Standard - Half Reflected and Standard - Reversed Royalty; Standard - Turn Zero, gated since H51, has one oracle behind cwmtt/tesseractMageOChicken and is not - not yet run in CI with the engine (step 2). Since CHECKPOINT 8 (H47, built 2026-10-17) a ruleset is gated only when two oracles stand behind each of its records, and the commit tier holds the engine to its baseline (spec issues 131-133). The release tier, the one that can enable the proof channel, is not yet run with it: its oracle layer fails on H48's 17 notes until the re-recording. The proof channel stays disabled (H23, H41) |
| H24, H26, H28 | built, tested | The generator and the 5dpgn/5DFEN parser are in-house (`engine/`), written from `docs/rules/README.md` and the notation spec's text (551 engine tests on `lane/gate` after H51's review); boundary clean. H28's one exception - the export-form record converted by the harness (CHECKPOINT 3) - is now in the register. Where the specification says when to write a token and not how to read one that disagrees, the reader reads leniently: the capture mark since H51 (2026-10-17, rulebook R11; `PgnOptions::capture_marks`), as the annotations (T15) and turn numbers before it - the move read as its squares define it, the mismatch a warning (`PgnReplay::warnings()`) |
| H27 | built, tested | The oracle harness, committed 2026-10-01. Tiered CI as H47 reads it since CHECKPOINT 8 (built 2026-10-17): ftxi per commit as a smoke test that gates nothing - two oracles are needed to gate a ruleset - and fails an engine that declines a record it read on the last green run, or reads none; cwmtt and Chess in 5D nightly and before releases; a run restricted to chosen sources or records is never a tier result. Upstream reports (CHECKPOINT 8: defects as bug reports, disagreements with a documented rule as questions citing the source; drafted by the agent, filed by the owner), in `docs/checkpoints/upstream/`: the two defects of R7 as bug reports (2026-10-01), and ftxi's reading of the check test as a question (2026-10-17, spec issue 126: five representative positions, one per rule question RQ1-RQ4; the 147 R1 entries' `upstream` names it) - none filed yet |
| H30 | built, tested | Step 2A (a04678c): every oracle reports its boards per ply; no placement divergence across the corpus |
| H31 | done | CHECKPOINT 1: seven clusters triaged (903f4ae), settled by the user's rulings (H36-H40). Re-run 2026-10-17 with the engine configured: the 110 new check findings (72 positions) filed by cluster and settled by R1 (and R2 at five positions); the 58 positions of CHECKPOINT 1's table re-derive exactly, cwmtt/many2's 14 file as CHK-A1 (step 2). The notation finding (cwmtt/futures action 17, an `x` on a jump that takes nothing) settled by the user's ruling, never by vote (H51, R11): ftxi's refusal on record citing R11 (KD-0170), KD-0167 to KD-0169 withdrawn; the record's positions the engine now reads past it, 17 to 19, agree with cwmtt and Chess in 5D wherever each answers - the mate at 19 is Chess in 5D's call and the engine's (cwmtt reports no legal action there, `no_action`, which does not say checkmate or stalemate, and gives no check verdict; nightly, 2026-10-17: no finding) |
| H32 | built, tested | Required sets at 100%, two recordings, cost files, calibration (2B) |
| H33 | built, tested | MANIFEST, verified snapshot, harness-modified check, CODEOWNERS (2B) |
| H34 | built, tested | Telltales by the comparison's rule (2B); 167 entries: 56 of 2026-10-02 (the two oracle defects among them, R7), 110 of 2026-10-17 (R1) and KD-0170 (R11: ftxi's refusal of an `x` that takes nothing) - every disagreement of the engine's with a recorded oracle answer is one, and every entry matches one; no telltale. KD-0167 to KD-0169, the strict reading's three, turned telltale when the reader turned lenient (the commit tier said so, as their header had said it would) and were withdrawn on the ruling's authority (H51) |
| H35 | done | Commit and nightly tiers green on Linux, 2026-10-01 (runs 36827435595, 36827459456) |
| H36 | built, tested | Engine E4: check status and submission legality per R1; compared with the oracles by the gate since 2026-10-17: every check disagreement is at a position where the oracle's every capture starts on a board R1 does not pass (CHECKPOINT 1's RQ1-RQ4), each on record with R1's citation; no engine defect |
| H37 | built, tested | Engine E4: verdicts only from exhaustive searches; node, time and memory budgets answer "unknown" (tested for every budget below the need) |
| H38 | done | `docs/rules/README.md` is the rulebook (rulings R1-R11), pinned in `gate.toml`; `verify` fails when it changes until re-pinned. R11 added 2026-10-17 (CHECKPOINT 8; H51, labelled a project decision: capture marks read leniently, with a conventions line and source [141]); its diff reviewed, no other entry changed, re-pinned (`rulebook --pin`: 08fddc5e... to 39e9ad58...) and the gate re-run. In review R11's source sentence was narrowed to what the specification says - "purely for better human comprehension" of the present-movement and created-timeline tokens only, T15's scope, not of every token a writer adds - the ruling unchanged, KD-0170's citation unaffected; re-pinned (39e9ad58... to ab01119e...) and the gate re-run. `CLAUDE.md`'s table still says the rulebook holds "the user's rulings R1-R10" - R1-R11 now; the instruction files are the user's to change (the one edit the user allowed in each at CHECKPOINT 8 was the engine build step), so the edit is put to the user |
| H39 | done | No repetition rule; the engine has none to build |
| H40 | built | Castling's attack test per-board (S2) and en passant per-board (S7); `CastlingAttacks::AnyBoard` removed (E4) |
| H49 | not built (rule) | Held open: Hunt with a lazily expanded root, the likely remedy above 100 boards - adopted only through a fresh pre-registered test (§11.4, §14.4) |
| H50 | built (placeholders) | The asset map's material values (`PieceValues`): P N B R Q at 100 / 300 / 300 / 500 / 900, unicorn 250, dragon 200, princess 700, brawn 125, common king 300, royals 0 - the register's provisional values exactly (issue 56) |
| H51 | built, tested | Capture markers read leniently (CHECKPOINT 8; rulebook R11, a project decision). The reader: `PgnOptions::capture_marks`, `Lenient` by default - a move read as its squares and boards define it, the mark selecting no move, a mismatch either way (`x` on a move that takes nothing, a capture without `x`) a warning (`PgnReplay::warnings()`; `t5d-gate` prints it on stderr, outside the protocol - the engine contract unchanged); `MarkedMustCapture` (the reading before H51) and `Strict` (the letter both ways) only by option (`--pgn capture_marks=...`). The writers - the engine's `synthetic::pgn_move` (t5d-play's and the synthetic records) and N5's `netplay.pgn_move` (the network's self-play and match records) - write `x` exactly on captures: the first tested on every move offered at every position of 53 random games - 9 variants under the default rules, every alternative rule option in Two Timelines - judged from the boards (14,081 physical captures, 24 of them en passant, 6,293 by non-branching jumps, 12,192 by branches, and quiet moves of each kind), the records replaying with no warning and under the strict reading; the second the same way in Python (`test_the_writer_marks_exactly_the_captures`: 10 random games under 5 variants and two en passant positions), and N5's replay check (`check_replay`) reports any warning of the gate's reader (a review fix: H51 had taken away its only guard, the gate's refusal of a superfluous `x`). ftxi's refusal of cwmtt/futures action 17 is KD-0170 (citing R11 and H51); KD-0167 to KD-0169 withdrawn. Gate: commit tier SMOKE TEST - PASSED; nightly PASSED, no finding, gating **Standard - Turn Zero** for the first time (7 records, 144 positions) beside Half Reflected and Reversed Royalty - its weakest record, cwmtt/tesseractMageOChicken, has one oracle behind it (cwmtt), cwmtt/futures two (cwmtt, Chess in 5D), so under the two-oracle rule of issue 122 (the next component) Turn Zero would not be gated - H23's row (nightly gating two rulesets) and spec issue 127's measurement (a two-oracle rule ungating nothing nightly gates) predate this: issue 127 now says so, H23's row is left to the issue-122 component. 6b's tools keep the reading before H51 (`sixb_pgn_options`; spec issue 129), so 6b's, 6b′'s and 6b″'s roots and Hunt's truth rebuild exactly (the engine's tests run `t5d-scan6b` and `t5d-scan6bp` themselves to hold them to it); t0database: no record or action changes (M7's and M22's commands byte-identical). Not yet confirmed in CI: no run has started on `lane/gate` from a5e61d9 on (account billing; step 2). Spec issues 125 (settled), 129, 130; proposed edits 101, 102 |

### A-G, I - representation, network, search, training

| ID | Status | Notes |
|---|---|---|
| A1-A3 | built, tested | Network N1: tokens, features, per-board flags, Δt-to-present. A3 on frozen boards under the cached configuration: intrinsic features (N3's default, ruled by M20), their status as B16's score-time key biases (`lane/net`, N-B16); live boards keep A3's features as inputs. A3's side to move becomes us/them with A7 |
| A4 | built, tested | Engine: every board immutable once published; persistent multiverse sharing history in chunks |
| A5 | not started | Piece-centric tokenisation - ruled at CHECKPOINT 6: build behind its flag; the window's provisional kill does not decide it, and the joint test with the window waits for the pilot run (M18). Last in the net lane, dense scope only (`piece_centric` raises today) |
| A6 | engine side built, tested; network inputs not started | Global turn-state inputs on the global registers. The acting player is in (D8, N5); the acting colour comes with A7. Engine ABI round 1 (2026-10-17, `lane/engine`): the counts are in the expansion (`mandatory_remaining`, `optional_playable`, `moves_made`; the state's own mandatory count - A6's commit changed no truth source, while the component's first commit, H42's fix, changed the prover's report, so the truth's code signature differs from main's) and the C API (`t5d_expansion`'s three appended fields; `t5d_game_summary.moves_made`; the game handle counts the moves of its action - 0 at a set-up, +1 per move, 0 after a submission or a replayed action), with the Python mirror (`Expansion`, `GameInfo.moves_made`; `Search`'s `turn_start` from the handle). The count is in the evaluation identity (issue 302): `EvalRequest`, the evaluation cache's key, the batch's de-duplication and MCGS identity, where it subsumes the turn-start flag; `t5d_search_new` takes the handle's count and checks `turn_start` against it (tested two moves into an action, from C++ and Python, where a flag's 0 or 1 cannot pass); `advance_moves` and `advance_action` carry the tree at each count (tested, tree and graph). §15.2's A6 row: `test_turn_state.cpp` recounts the three from the rulebook at every state of random play under every ruleset (both Q4 readings), through the expansion and a `t5d_game`, and checks the spec's arithmetic move by move where no activity changes (issues 143-145). Next: the net lane's inputs - `PositionState`'s counts, the embedding, the converter's copy and cross-check, and `cache.state_digest` (an epoch's anchor), which takes the count when the network's input does |
| A7 | not started | Perspective canonicalisation - pieces as ours/theirs, ranks mirrored and timelines negated when Black acts, outputs mapped back: net lane, §15.2's colour-swap mirror test (read with the acting colour equalised, issue 304), and frozen K/V keyed by perspective in the cached path (issue 303). Today the input is absolute (colour, side to move, raw coordinates). *N-M20: M20's dispositions are built (`lane/net`); A7 is a component of its own - `cache.board_ident(..., acting=True)` keys a frozen board by the acting player under either frozen-feature reading, unused until A7 (issue 303).* |
| A8 | not started | The board-layout embedding - a prototype (6p): net lane, behind its flag, with §15.3's cost row |
| B1, B2, B4 | built, tested | Network N1: one weight set, unified attention, 4D RoPE (whose positional term factorises per axis - joint-axis geometry is B15's, register B4) |
| B8 | built, tested | Network N1's 2 + 16 registers, at the ruled positions (CHECKPOINT 6, M20; `lane/net` N-M20): per-board registers rotated on timeline and time only, global registers position-free on every pair - no switch. The arms M20 disposed of are gone: `board_register_spatial` (`learned_angle`, the old default, with its 2 x 2 learned coordinates, and `zero_angle`), `global_register_pairs` (`anchored`) and the attention context's `global_free`. Aux heads pool the squares; the value-type heads and the submit logit pool the global registers by mean (D3, D8; issue 223). Cost: the CPU kernel path widens Q/K/V to 44 for every row and recomputes the register rows (indicatively +20% / +55% per attention call at 240 boards, BF16 / FP32, against one plain call - `attention.py`; not a measurement); N3's counted dense FLOPs (MEASUREMENTS, 2026-10-02) were taken with the old `learned_angle` registers - to re-count, with the wall times, in step 2's quiet-machine block |
| B9 | built, tested | Network N2: move-graph mask on 2 heads; the built direction, register reads and head counting are now the v1 values (§6 "As built", M20; issue 16) |
| B14 | built in part | Network N2: block-sparse attention in a CPU gather form. B14's score function built (`lane/net`, with B15): `attn_impl="flex"` (`python/t5d/net/flex.py`) runs torch's `flex_attention` with every bias computed inside its score function - B15's entry by integer arithmetic from the two tokens' coordinates, B16's status entry from a per-token index (N-B16), B17's precomputed query-class products at the pair's class (N-B17), B11's typed relations, B12's board-pair table, a caller's bias - every mask in its mask function (padding, the block-causal scope, section 6's roles from per-token data and one edge bit per pair), and a lossless block mask built from the board-major order at board granularity. Eager on CPU (PyTorch 2.13): forward only, every score materialised - small positions, §15.2's B15 row (spec issue 211). Under v1's register rule the masked heads' block masks have no empty block (issue 209). Compiled and GPU kernels need GPU time (G9, M19) |
| B5 | built, tested | Network N1, behind a flag (off); its as-built details are "v1 placeholders" (Provisional, CHECKPOINT 6; M20), tested by ablation 2 - §4.1 does not state them (edit 9) |
| B6 | superseded; removed | By B15 (CHECKPOINT 6, reversal 25; M20). Removed from the code with B15 (`lane/net`): `binned_bias`, `binned_bias_rope` (issue 9f) and their fields - no arm survives, ablation 2's arm is B15 on or off. Its log buckets live on in `python/t5d/net/buckets.py` for B12's relative cells (`smolgen_time_*`, `smolgen_timeline_*`; smolgen's 152 cells and 47,136 parameters unchanged, issue 212). Under B15's far-bin placeholder B6 is not B15's special case, which §4.1, the register's B6 row and reversal 25 say it is: beyond the exact classes its per-axis buckets are not kept, only Δtime's sign and the clamp (issue 205, edit 159; put to the user) |
| B10, B12 | built, tested | Network N2, behind flags (B12 as relative board-pair cells, full rank - the register now records it as built; the cells are the log buckets B6 used, re-homed in `buckets.py` when B6 went - issue 212). The register says smolgen "keeps its own normalisation" (F5); the built smolgen has none (`lane/net`'s issue 202) |
| B11 | built behind a flag, tested | Network N2; five types since `lane/net`'s N-B11 - attack, defence, pin, x-ray, check: travel-threat folded into attack (CHECKPOINT 6, reversal 26) and dropped from the network's type set (`config.TYPED_RELATIONS`, `state.RELATION_TYPES`; the synthetic generator's cross-board attacks are its travel threats). One scalar per head per type, shared across layers: 8 x 5 = 40 (80 directed; 48 and 96 before), only with `typed_edge_bias` on - the v1's count is unchanged. Live query rows only under the cached configuration (built at N3). The relation ids are the C ABI's: id 6 stays reserved (append-only), never emitted, refused by `PositionState`'s validation. The engine emits attack, defence and x-ray; pin and check - in the type set as the register lists them - stay empty until engine ABI round 1 builds them behind the relation mask. The engine lane's documents still name travel-threat among the network's relation ids (the relation-id comments of `records.h` and `capi.py`, `engine/README.md`'s (e), `convert.py`'s docstring, and `game.py`'s `relation_bits` docstring - "names the network's contract has but the engine does not build (pin, check, travel_threat)"): theirs to bring in line with ABI round 1 |
| B3 | not built (rule) | Contingency only; no trigger |
| B7 | not built (rule) | Deferred; its gate now names ablation 2's head-allocation evidence (issue 121's B7 part closed, M20); no §15.2/§15.3 need (d_head is a config value anyway) |
| B13 | not built (rule) | Refused |
| B15 | built, tested | The 4D displacement bias, the v1 default (`lane/net`, `python/t5d/net/displacement.py`): one learned scalar per head per layer, a 15 x 15 slice for each of 54 board-distance classes (the 45 exact ones and 9 far bins), 368 line entries, 162 register entries and a global one - 12,681 per head per layer, 1,623,168 parameters (§4.1's ~1.6M; 7.07% of the network). Its unstated details are named placeholders: the far bins (`disp_far_edges`, issue 205), the line set (`disp_line_max`, issue 206), the register and global entries (`disp_register_pairs`, issue 207), the initialisation (`disp_init_std`, issue 208); the far bins' form and the line set's time unit and partition are fixed in code, and under the far form B6 is not B15's special case (issue 205, put to the user). Ablation 2's no-table arm: `displacement_bias=False` - the table draws from its own generator, so the arm is the network without it (bit for bit the pre-B15 v1). Per layer through `AttnContext.for_layer(h, layer)`, the cached path included (a frozen board's rows carry it at every layer; geometry, so every row). Tested: the classes against §4.1 written out pair by pair (grid and twelve constructions - every variant's board size, Global Warming's 1 x 1 included - three placeholder settings; each placeholder switch checked to regroup pairs); §15.2's B15 row - through B14's score function = materialised, bitwise at the bias, 1e-5 at the output - and the CPU paths' outputs and gradients; §15.2 row 1 with it on (every `test_cache` case) and with the layers' tables far apart. Boards wider than 8 are refused (none in any variant). Signs are raw colour signs until A7. §15.3's cost row ("negligible") is for the quiet-machine block: on CPU every head now carries a per-row bias |
| B16 | built, tested | `lane/net`, N-B16 (`python/t5d/net/status.py`): under the block-causal scope with intrinsic frozen features, a learned key bias per head by the key board's status - its Δt-to-present bin and its timeline's active flag - on live query rows (head boards' tokens, global registers) against frozen boards' tokens (squares and per-board registers), recomputed from the evaluated position's present every evaluation; never on a frozen row (C17), never under the dense scope (the v1, every refresh pass). Its unstated details are named placeholders: the bins (`status_dt_exact = 4`, `status_dt_max = 128`: 19 signed log2 bins, issue 213 - which also reads "timeline status" as the active flag), shared across layers (`status_bias_per_layer`, issue 214), N(0, 0.1) from its own generator (`status_bias_init_std`, issue 215): 304 parameters, only in a network of the cached configuration (`status_key_bias`, on; the v1's count unchanged). The refresh stays as built (M20, the main loop's reading): history boards keep the anchor's features in their K/V and are biased at every node below the anchor; the anchor's own evaluation is the refresh pass and adds none (§7.9; `make_context(status_bias=False)`; issue 216). In every path: the structure's rows (the CPU paths), B14's score function, the dense reference, the cached live pass. Tested: the rows against §4.1 written out from the positions (every bin, past and future, active and inactive; §15.2's constructions; four placeholder settings), bitwise; the score function's term bitwise and attention through it; the CPU paths' outputs and gradients (the table's included); every implementation one function; §15.2 row 1 (every `test_cache` case with intrinsic frozen features - the default - runs with it on; the `current` test arm has none) and a moving present and an activating branch - frozen entries all hits, only the superseded board computed, the bias moved and not the K/V; purity and refresh forks with it on; the at-anchor rule, which the cached path and its reference share, against an oracle of its own (N-B11-B16-B17 review fixes): with the table changed between two fresh searches a root- or node-refreshed anchor's evaluation stays bit for bit and every node below it moves, in both paths - also when only statuses that the history boards alone hold change; the table's size, generator, reach, switches; translation and board order. A block-causal forward of a dense-scope network - which has no table - is refused while B16 is switched on (`make_context`: its frozen boards would lose their status with nothing in its place) |
| B17 | built behind its flag, tested | `lane/net`, N-B17 (`python/t5d/net/shaw.py`): `shaw_key_term` (off; it needs B15 - "on top of B15"): each unrotated query dotted with a learned vector for its pair's class, scaled by 1/sqrt(d_head), added to the score beside B15's scalar; value-side vectors not built (held open). The classes are B15's entries restricted to the exact near classes, read as the file-rank offsets within ±`shaw_near` = 2 (Δfile and Δrank) in each of the 45 exact board-distance classes, plus B15's line classes: 1,493 - a placeholder: the nearest board-distance classes with their full slices give the same count and costs and cover other pairs (issue 217, put to the user); a vector per class per head per layer, 6,115,328 parameters - 26.6% of the v1 - or 764,416 shared by the heads (`shaw_shared_heads`; issue 218); the unrotated query (forced) and Shaw et al.'s scale (issue 219); N(0, 0.1) from its own generator (issue 220). Two computations of one function: the gather - precomputed query-class products P = q R^T, then each pair's entry by its class - in every CPU path (any block, key selection or row set), the cached path (a frozen board's rows from its own queries) and B14's score function (P read inside the attention, the class from B15's entry computed there); the direct per-pair form in the dense reference. Tested: the class set written out from B15's layout (three settings); §15.2's B17 row - gather and direct equal the per-pair computation written out (FP64 to 1e-12, FP32 to roundoff) over every key, blocks, key selections, row sets and head subsets, on §15.2's twelve constructions, three seeds, initial and unit scale, shared and per head; the CPU paths' outputs and gradients (the vectors' included) against the direct term; the score function; every implementation one function; cached = one full pass, in the causal epoch and through root and node refreshes - supersession inside the refreshed subtree, both C20 readings (N-B11-B16-B17 review fixes); translation and board order with it on (a rotated query shown absolute); the count, the generator (off is the B15 network bit for bit), every layer's reach. §15.3's row: the analytic ratio C/T is 0.511 at 44 boards and 0.094 at 240 (§15.3: ~0.5x, ~9%), and the implementation issues exactly those products (counted FLOPs: 2 x body x (C + 1) x d_head x H x L on the one-block path). CPU wall time is the quiet-machine block's (`bench.py`'s `b17-shaw`) |
| C1 | built, tested | `attn_scope = "dense"` is the default |
| C4 | built, tested | Full MHA (N1) |
| C17, C19, C20 | built, tested | Network N3: the block-causal scope (`attn_scope = "block_causal"`), the live set fully bidirectional, supersession recomputed once under the state-defined mask - both of C20's readings built (`supersession_scope`; 6c decides, now in the register); the causal past is the ruled ancestry (`causal_past = "lineage"`, C23) |
| C2 | built (for §15.2/§15.3) | Network N3's cached path; the decision itself stays deferred (6c, §7.7), now re-weighed against the pilot run's game sizes (M18). The path absorbs B15, B16, A6 and A7 as they land (A7: frozen K/V keyed by perspective, issue 303); B15 is in (`lane/net`: a frozen board's bias rows built per layer, cached = one full pass with it on), and B16 (`lane/net`: the live pass adds the status biases from the node's present; frozen rows and entries never carry them; none at a refreshed epoch's anchor, and below it on every frozen key, history boards included - both halves tested against an oracle of their own); B17 runs through it too (a frozen board's rows get the Shaw term from its own queries; tested through root and node refreshes) |
| C10 | not built (rule) | Lazy K/V for boards entering E: deferred with the window (M16) |
| C3, C3a (rung 1) | built, tested | Root refresh and the travel-triggered, visit-gated rung-1 fork (copy-on-write epochs), behind flags. C3's conditioning-offset training hook is parked with the deferred cached configuration ("Parked", below) |
| C3a (rungs 2-3), C18, C22 | not built (rule) | Held open (M5 for rungs 2-3) |
| C5, C7 | not built (rule) | The window: killed, provisionally (M16, CHECKPOINT 5); re-checked by the pilot self-play run (M18) |
| C6 | step 1 done | The \|E\| curve measured (M7); the head-to-head is Ablation 1, halted with the window until the pilot run (M18) |
| C8 | honoured | C24's add-only rule (and rung 1's) - refreshes are only ever added |
| C9 | to check | Reported built in `expand.cpp` by the 2026-10-04 audit; to verify against its row |
| C11 | built behind a flag, tested | MCGS (E6a): identity is the state key with branch origins and the moves made in the action in progress (A6; since engine ABI round 1 - issue 302, proposed edit 200, and 119 for §11.3's full key), which subsumes whether an action is in progress (§8.5 and the register say only that: issue 87), compared exactly; one action's move orders still merge (tested); awaiting 6k |
| C12, C13 | built, tested | Re-rooting with garbage collection, the persistence hook and pondering (E6a) |
| C14 | built, tested | Fresh searches per decision in self-play; the evaluation cache shared across them, purged by board count (issue 89); the C API's `t5d_search_new_sharing`; from Python (N5) only between searches with the same network input, with the cached path's refresh epochs set on the search - as §8.7 now has it (the register's C14 names the shared, board-count-purged cache and one cache per evaluator in match play). Since engine ABI round 1 the cache's key holds A6's moves made in the action beside the state and the epoch (issue 302, proposed edit 200; edit 119 for §15.2's purity row, §7.9 and §8.5). Since round 2a an entry holds every D3 head (`Evaluation` 232 -> 256 bytes: the WDL, the error and moves-left outputs, the categorical made optional), returned bit for bit on a hit (tested) |
| C23 | built, tested | The node key's default carries branch origins (lineage, `NodeKey::StateAndOrigins`); the network's causal past is the ruled ancestry (`causal_past="lineage"`, tested against ancestry computed from the branching record, N5) |
| C15 | built in part | FP8 e4m3 frozen K/V (`frozen_kv_format = "fp8"`, applied by the forward itself); varlen/ragged kernels not built (none on CPU; GPU time, M19). The register: FP8 K/V obey G1 only with an FP8 trunk or FP8-frozen-K/V training (issue 34). **Known defect, not yet fixed:** `fake_quantise(x, "fp8")` (`kvformat.py`), which both the forward's FP8 rounding (`model._frozen_kv_rounding`) and the cached path use, casts to FP8 inside autograd with no straight-through estimator, so its backward casts the gradient to FP8 unscaled. Checked 2026-10-17 on CPU (torch 2.13.0+cpu; one 8 × 8 block, seed 0): with \|x\| about 0.1, 63 of 64 gradients were zero; with \|x\| about 1, they were 0.90-1.03 where 1 is expected. FP8-frozen-K/V training (the remedy above) and G8's hook need the net lane's STE fix, with a gradient test, before G2 and G8 |
| C16 | not built (rule) | Deferred; last, only if node-starved |
| C21 | not built (rule) | Refused; a mutant that snapshots is caught by the tests |
| D1 | built, tested | Network N1's pointer policy over the moves the engine supplies - in search, legal or unknown, unknowns verified when first selected (E5a, E6a), as §8.1 now says; the game API's expansion stays pseudo-legal until ABI round 2's `legal_moves` option. The promotion offset (CHECKPOINT 6, M20) is the v1 (`lane/net` N-M20): a learned per-piece logit read from the destination token, always on in the policy - `policy_promotion` and its `none` arm removed (+3,341 parameters). The opponent-reply head has none, as built at N1: which other pointer heads carry it, and the offset's form, are issue 222 |
| D3 | engine side built, tested (engine ABI round 2a); network heads built in part (the pre-CHECKPOINT-7 head) | The search (round 2a): WDL is the search's value - W − L backed up, the draw probability D beside it on every node and edge (issue 148) - the categorical optional and read only as an uncertainty signal (6m's switch, issue 149), the error head's output and moves-left per leaf, all through one new deliver call (`t5d_search_deliver_evaluations` and its `t5d_evaluations` struct; the old `t5d_search_deliver`, which backed up the categorical's mean, retired); each root child's search (q, d) and network WDL, moves-left and error recorded, and the root's own; the built-in evaluators give a WDL. The Python evaluator delivers the network's WDL, categorical and moves-left, and its error output once the net lane names one `value_error` (NaN until then); `Evaluation.value_mean` is now W − L. The network: N1's 51-bin value head over the pooled global registers, smoke-trained on the outcome. Its MLP is Mish (F6, CHECKPOINT 8). Pooling as M20 rules it (`lane/net` N-M20): the 51-bin head, WDL and the short-term value all read the mean of the global registers - WDL and the short-term value moved off the squares with `aux_global_pool` (issue 223); a test holds them invariant to permuting the registers' final states. Corrected at CHECKPOINT 7 (reversal 28): the categorical over the root search value (HL-Gauss), short-term values, the error head - the net lane's heads |
| D4 | built in part | Network N1's aux heads: WDL, moves-left, short value, effort, per-board, per-square (the label set stands as v1, M20) and the opponent reply, always built at weight 1.0. M20 in the code (`lane/net` N-M20): moves-left counts turns - one player's action each (issue 305) - in the head, its helper and the losses' docs (the record-to-target builder is M21's, and so is whether the action in progress counts: issue 224); moves-left and effort read the squares, WDL and the short value the global registers (D3; issue 223). To change (CHECKPOINT 7): the soft policy added (weight 8), the reply a prototype behind its flag at 0.15 with §15.2's mapping test - the net lane's heads, after engine round 2 (the reply head is left as built until then); the per-square label generator is step 2's. The effort label's source is in the search (engine ABI round 2c: `root_best_since`, the root's visits when its most visited child last changed - issue 154) |
| D5 | not started (rule) | A training principle |
| D6 | built (flag) | `board_reduction`, awaiting 6j |
| D7 | not built | 6p's aggregate initialisation (training-side) - parked with 6p's other arms. The tension with D8 is settled: the submit logit is read from the pooled global registers, so D7 holds (D8, M20) - in the code since `lane/net`'s N-M20, with the value-type heads pooled the same way |
| D2 | built in part | The ply split (`SearchParams::split`, the default) and the whole-turn arm, driven from Python through the C step API (N5). CHECKPOINT 6: the whole-turn arm and its 256 cap are superseded by the flat form; the tree form and the flat form are not started - engine lane, after M22 and ABI round 2: the tree form under the tree search only (the engine refuses it with MCGS as a parameter combination - parked and logged, issue 301; no document refuses it), the flat form on the in-house hypercuboid search after the gate is configured, each with its §15.2 row (the tree form with every interaction gate firing = the ply split; the flat form's generator = an exhaustive listing of legal turns in summed-logit order) and §15.4's turn-structure invariants; then the whole-turn arm retired; the flat form's head in the net lane. Gate 6x (with 6g, 6k) |
| D8 | built, tested | Engine and search (E7, E6a): submit an explicit edge, legal only when C4 holds, never automatic. Network (N5): a global acting-player input on the global registers (the ruled place, M20 - the `all_tokens` arm and `acting_player_input` removed, `lane/net` N-M20; with no global registers, an ablation-3 arm, on the board tokens - the live boards' under the block-causal scope - a placeholder fixed in code, issue 225, tested to reach every output under every register arm), one distribution over moves + submit, values from the acting player's perspective; the submit logit read from the pooled global registers - their mean, as the value head reads them (M20, N-M20: `submit_register` and register 0 gone; tested invariant to permuting the registers, moved by any one). The readout's form (`submit_head`: linear, 257 parameters, or the policy's bilinear form) is still a placeholder (issue 109). Under the flat form the generator ends turns itself (issue 92) |
| D9 | not started | The optimistic policy - a prototype (6t), match play only; its use in self-play is held open (§14.4): the net lane's heads, behind its flag, after engine round 2 |
| E4 | principle (weakened) | "Narrow hard, because the auditor is watching" - the register now says weakened: it holds where the settling scans (< 50 boards) and Hunt (< 100) run, neither wired into the search on main (E10 lost; step 2); above that, narrowing is a bet on the audit lane and the league |
| E1 | built, tested (the mandatory-count widening option not built) | PUCT with entropy-adaptive top-k widening and FPU; the policy temperature at every node. CHECKPOINT 7 (with I19), engine ABI round 2b: root noise in self-play with its total concentration (`dirichlet_concentration`, placeholder 0.03 x 361, appended beside the per-child `dirichlet_alpha`, which stays as its arm at 0) split across the root's open edges, per decision (issue 150); KataGo-style shaping a prototype behind `dirichlet_shaping`, off (its form a reading, issue 151); the root policy temperature (I19; issue 152), before the noise. "The widening schedule may depend on the mandatory-board count" - an option for 6g, not built (A6's count is in the expansion). Every constant a placeholder, named in §8.4's "Constants as built" (issue 82; edit 123) |
| E5, E9 | built, tested | LCB with a visit floor (E5); the solver behind H41's switch, the secure child, a visit cap on solved children (E9) - constants named as placeholders in §8.4 (issues 84-85). E5's optional term (weight 0) reads the child's uncertainty signal - 6m's switch, since engine ABI round 2a (issue 149) |
| E7 | not started | Time management, E11 in step 2 (engine lane): the moves-left integral allocator - moves-left counted in turns, converted to board-moves (issue 305) - and the stop/extend table, with 6o's arms (flat-evals and flat-wall-clock allocators, the signals on/off, the spiral guard at high N); the second volatility source is Hunt's and the settling scans' proofs (the register). The register keeps a "root-scan reserve" (§8.6); its one named consumer, Guard's re-search, is offline (adopted nowhere), so E11 holds none - a reading (issue 306) |
| E2, E3 | built behind flags, tested | Off; awaiting 6m and 6t. They read a node's uncertainty signal: the categorical's variance by default, the error head's output under 6m's switch (`uncertainty_source`, engine ABI round 2a; issue 149); their constants placeholders twice over (issues 81, 149) |
| E8 | not started | Behind its flag (6v), with E11 in step 2 |
| E6 | not built (rule) | Refused |
| F1, F2 | built, tested | 22,965,139 parameters (F1 after F5, B15 and M20: 21,338,634 after F5 - 20,736 fewer than N5's 21,359,370, gain-only norms, no Q/K/V biases, spec issue 203 - plus B15's 1,623,168, spec issue 210, plus M20's 3,337: the policy's promotion offset, 3,341, less the learned register coordinates, 4); the L24-ff12 preset, 46,894,547 (F2: 44,459,795 and B15's 2,434,752 over 24 layers). Unchanged by F6's Mish in the heads, by B11's fold (its flag is off; with it on, 40 scalars where there were 48) and by B16 (its 304 are in a network of the cached configuration only: 22,965,443 with `attn_scope = "block_causal"`) and B17 (off; with it on, 6,115,328 more - 29,080,467, or 23,729,555 with the vectors shared by the heads; spec issue 218) |
| F5 | built, tested | Network: RMSNorm-style norms (a learned gain, no centring, no bias; FP32) everywhere in the trunk, no Q/K/V biases, the attention-output and FFN biases kept (spec issue 200), pre-norm. 6y's arm - post-norm with DeepNet's initialisation, `norm_placement="post"` - through the dense trunk and the cached path, its unstated details placeholders (issue 201); smolgen has no norm to keep (issue 202) |
| F6 | built, tested | Network: Mish in the trunk FFN (`ffn_activation="mish"`) and in every head's hidden layers - the value head's MLP, the only head with one (CHECKPOINT 8's ruling on issue 204, `lane/net` N-F6heads: no switch - `head_activation` and its GELU removed); 6y's arm, squared ReLU (`"squared_relu"`), in the trunk only, through the dense trunk and the cached path - tested: no head follows it. Outside F6's words, kept as built: the embedding's Δt MLP (GELU) and B12's smolgen (SiLU) - spec issue 221 |
| F3 | not built (rule) | Refused |
| G7 | not started | SPSA/SPRT tooling (training-time) |
| F4 | not built (rule) | The NNUE is held open (H6, CHECKPOINT 4) |
| G1 | built, tested | BF16 trunk under autocast; FP32 heads and RoPE; the cache stores K/V in the live format (§15.2 row (e)) |
| G2 | not started | 8-bit inference, FP8 before INT8 (gate 6w): the trunk's matrix multiplications, attention included; heads and RoPE FP32. G2's quantiser serves G8's hook in the net lane; its CPU emulation is step 2's; real kernels need GPU time (M19). `fp8_attention` raises today |
| G3, G4, G5, G6 | not built (rule) | Deferred / held open / refused |
| G8 | not started | QAT prepared, not adopted: fake-quantisation hooks behind a flag in the training code, smoke-tested - the net lane, with G2's quantiser; the simulated format is pinned with the inference path. Written from papers: Monroe's `quantize` option (Lc0's training code) is an idea only (AGENTS.md, H29) |
| G9 | built in part | PyTorch drives the C++ search through the C API (N5, eager, on CPU). FlexAttention runs eagerly through B14's score function (`attn_impl="flex"`, `lane/net`: forward only on CPU). Not built: torch.compile and compiled FlexAttention kernels (GPU time, M19); ONNX/TensorRT export - refused for now (M5). The bounded AMD attempt (AMD's PyTorch for Windows on the RX 9070 XT, figures AMD-labelled): any time, in `build/` only |
| I1 | not started (rule) | Self-play training: no training runs in the untrained build (the brief); the pilot run (M18) comes after §15.5 |
| I2 | engine side built, tested (round 2c); targets not started | Targets from the pruned visits - since CHECKPOINT 7 the search-value targets too: M21's targets (net lane) from I15's records. The engine side (engine ABI round 2c, issue 153): what target-side pruning needs, per root child (visits, priors with and without the noise, q, value variance, E3's factor, `boost`, `steered_visits`, `forced_visits`) and the root's effective cPUCT; search-side forced playouts not built (§12.5's pruning is at target construction); the steering counts 0 until E10 wires the asset map and Hunt (no self-play data with E10's boosts before the pruning exists) |
| I3 | built in part | Match play as I3 configures it: root noise off, the final rule (LCB), the tree carried (`t5d-play --match`, E6a; netplay's `mode="match"`, N5). The mature regime is training-time. In the untrained build: the curriculum hooks as interfaces only (H44, step 2) and I18's resignation rule (M21); MLH steering and early adjudication parked (issue 300; "Parked", below) |
| I4 | principle | Confidence-family mechanisms never scale down the confidence-independent layer - followed: C24 is add-only; E8 (not started) to follow |
| I5 | not started | PCR in the self-play drivers and the training loop, smoke-tested (M21, net lane) |
| I6 | not started | A prototype: its epistemic triggers as sampler arms beside I12 (6u), in M21 |
| I7, I8 | not built (rule) | M5 |
| I10 | parked (monitoring); 6m's switch built, tested | Calibration monitoring measures nothing on random weights - the pilot run. The error head it watches is D3's (net lane). 6m's switch (`SearchParams::uncertainty_source`, categorical by default) built in engine ABI round 2a (issue 149) |
| I9 | built, tested | The ruleset ID in every state and store key (H18): the Zobrist key, the solved and disproof stores, the evaluation cache |
| I11 | built, tested | Self-play's turn cap from the state (and t5d-play's 300-board cap), scored as a draw, its frequency logged (issue 88) |
| I12 | not started | Surprise-weighted sampling, the default sampler (6u): M21, with 6u's and I6's arms |
| I13 | engine side built, tested; deblundering not started | Deblundering (6z): M21. What it needs from the search is built (engine ABI round 2a, issue 148): the draw probability backed up beside the value, so the best child's search WDL is its (q, d); moves-left not backed up - each root child's network moves-left (and WDL, error) recorded instead, the target builder adding the offset |
| I14 | not started | Proof rescoring (6z): M21 builds the hook disabled - hard-off until the H23/H41 sign-off |
| I15 | engine side built, tested (ABI round 2); the records not started | Rich training records, fixed before the pilot run: first in M21 (net lane), with §15.2's round-trip row. The engine side built: the root's raw outputs (round 2a: its WDL, moves-left, error; the categorical and the raw logits by the memo, keyed by `root_key`), the root children's search (q, d) and network outputs (2a), the root side table and summary fields (2c: `child_key`, `prior_noise_free`, `value_variance`, `boost`, `exploration_scale`, `forced_visits`, `steered_visits`; `root_key`, `root_cpuct`, `root_best_since` - issues 153, 154) and `t5d_game_expand`'s `legal_moves` with its budget, which rebuilds a decision's input (issue 155); proofs by reference wait on the sign-off (H23/H41) |
| I16 | not started | The replay window growing sublinearly (6z): M21, a placeholder rule |
| I17 | not started | SWA (6z): M21, smoke-tested |
| I18 | not started | The false-resign check: M21 - resignation disabled in 10% of self-play games, false resignations measured per board-count range |
| I19 | engine side built, tested (round 2b); the schedules not started | Exploration placeholders. Built (engine ABI round 2b): the root noise scaled to the open-edge count (E1), the shaping a prototype behind its flag, the root policy temperature as a parameter (`root_policy_temperature`, 1 by default) - issues 150-152, edit 123. The training generator's (M21, Python): the move-temperature schedule (placeholders: Lc0's T60, 0.9 -> 0.45, cut at move 60; a "move" is issue 305's turn) and the root temperature's self-play value and decay (KataGo 1.25 -> 1.1), set per search; today netplay's and selfplay's step schedule (1.0 for 16 decisions, then 0) and a root temperature of 1 |
| I20 | not started | KL-adaptive full searches - a prototype (6l): M21, behind a flag, with PCR |

### M - the plan itself

| ID | Status | Notes |
|---|---|---|
| M14 | superseded | By v3 (H1): its df-pn and incremental attack maps carried into Guard and Hunt (E9); the per-leaf estimate channel removed |
| M15 | measured - STOP | 6b′ on a quiet machine: (A) 77.4% / 40.7% / 23.7% / 0.0%; (B) fits in no range; (C) df-pn 408 against brute force 400 under 50 boards - STOP, a design review |
| M16 | applied, confirmed | M7's readings (corrected at CHECKPOINT 5): time-share multiplier, human-game weighting, the per-phase rule under 50 boards with the band resolving to off - the window is killed, provisionally |
| M18 | not started (rule) | A pilot self-play run before serious self-play re-checks the window and the other compute-weighted decisions - after §15.5, not in the untrained build |
| M19 | applied | No GPU: counted FLOPs and CPU-labelled wall times (`dev-r`) are the untrained build's write-back; the GPU figures - FA4-class evaluations/s, the leaf-batching gain, FP8, the block-sparse and varlen kernels, the Phase-3 movegen re-check - named as deferred, measured on rented NVIDIA time before the pilot run; the bounded AMD attempt is G9's |
| M20 | built, tested; the reply prototype parked with the heads (after engine round 2) | The placeholder dispositions (issue 121), each the code's default and its disposed-of arms removed where no named ablation needs them (`lane/net`; A7 is a component of its own). N-M20: the promotion offset in the policy (D1; `policy_promotion` gone); position-free per-board registers and global registers on every pair (B8; `board_register_spatial`, `global_register_pairs` and the learned coordinates gone); aux pooling from squares - moves-left, effort - with D3's value-type heads (WDL and the short value too) pooled from the global registers by mean (`aux_global_pool` gone; issue 223); moves-left in turns (issue 224); the acting player on the global registers (`acting_player_input` and `all_tokens` gone; with no global registers on the board tokens, issue 225; `cache.board_ident`'s `acting` keying kept for A7, issue 303); the submit logit from the pooled registers (D8; `submit_register` gone). Earlier: B6 removed (B15); B16's status biases with intrinsic frozen features (N-B16 - its `frozen_features = "current"` test arm stays, as N-B16 left it: no named ablation needs it, and whether to remove it is the main loop's call - the N-F6heads-M20 review); both supersession readings (C20; docs relabelled). Relabelled, not changed: B5's details (ablation 2), 16a-g (B9's "v1 as built", ablation 6a), the per-square labels (6j) - now v1 values, not placeholders; B7 has no code. Not here: the reply head's reinstatement as a prototype (D4) - the heads' rebuild after engine round 2; until then it is always built, as at N1. Not on M20's list: D8's readout form (`submit_head`, issue 109) |
| M21 | built in part | The training loop, written and smoke-tested, no training (§15.5 does not require it; the pilot run does). Built (N1+N2; the submit target at N5): losses for today's heads - value, policy with submit, WDL, moves-left, opponent reply, short value, per-board, per-square, effort, each at weight 1.0 (`t5d.train.losses`) - and a smoke test of at most three optimiser steps on synthetic targets (`t5d.train.smoke`). Missing - the net lane, after engine round 2 and the heads: I15's records first, then I2's targets with §12.2's exclusions and provenance, the new heads' losses (D3, D4, D9), PCR (I5) with I20, I12 with 6u's and I6's arms, I13, I14's hook disabled, I16-I19 (§12.8's mechanics), the QAT hook (G8), §15.2's round-trip test |
| M23 | applied | Every working branch is pushed after each commit and takes main by merge, never rebase. The three lanes and their number blocks are the main loop's plan under this rule ("Current state (2026-10-17)") |
| M24 | not started (rule) | Ablations on a shared dataset - after the pilot run; what the untrained build owes it is I15's record completeness |
| M1-M6, M10 | applied | The plan's process rows: the build follows them (this log) |
| M17 | measured | 6b″ on a quiet machine (d96d6d9): Guard 65.1 / 15.8 / 5.2 / 0.0% - adopted in no range; Hunt 69.0 / 63.6 / 35.0 / 0.0% (200+: one truth mate, too few to read) - adopted under 100 boards; every refutation and mate confirmed by E4. Measured with the stores' old lock (H17's row): the verdicts' soundness stands - every refutation and mate counted was confirmed by E4 alone, and no verdict logic has changed since (the truth's sources have, three times: README, "The truth's sources since M17") - but its 5-thread rates may understate Guard and Hunt (indicatively 2.7-2.9× and 1.45-1.7× the nodes per second with the fixed lock). Whether 6b″ is re-run on a quiet machine is the main loop's call (issue 147) |
| C24 | built, tested | The refresh ladder's complexity trigger behind a flag (`CacheConfig.complexity_refresh`, off; prototype, gated like rung 1 on 6c's existence test); add-only over rung 1 for every signal; every number a placeholder (spec issues 75-79) |
| C25 | not built (rule) | Held open |
| M7 | measured; killed, provisionally (M16, CHECKPOINT 5) | On real games (t0database, 716 games, max 186 boards): on at k = 8 from 50 to 186 boards, off under 50, and the kill clause kills it under the human-game weighting (on-phases 18-38% of compute); confirmed by the user; above 186 provisional (MEASUREMENTS) |
| M8 | measured - STOP | 6b re-run (M12): root scan 76.7% / 31.4% / 24.7% / 2.4% by range (< 50 / 50-99 / 100-199 / 200+); rung 1 and quiet-leaf reach fail everywhere; stand-pat valid |
| M12 | applied - STOP | The bounded optimisation pass (E5c) and the re-run at 5 threads, Lazy SMP: 50-199 boards fail -> STOP, redesign the verification layer |
| M13 | applied | 8 cores per GPU: M9 judged on 2 expansion cores (passes, 1.87x), the scanner measured at 5; with the scanner retired, its five cores serve Hunt, the settling scans and the audit lane (E10's scheduler, H14) |
| M9 | measured | Pseudo-legal: PASS (2.2x / 2.5x the targets on one thread), history term 2%; with E5a's fast legality: 1.01x at 240 boards on one core (in the band), 1.87x on two; 1.21x at 600. The Phase-3 re-check waits for a GPU evaluation rate (M19) |
| M22 | measured | Turn width on t0database (692 games, the default rules, `t5d-measure --turns --turn-summary`; `tools/turn_width.hpp`): more than one playable board at the turn's start - where the D2 arms differ - in 15.1% / 65.2% / 99.4% of turns (< 50 / 50-99 / 100-199 boards; 13 games at 100-199, none at 200+); more than one mandatory board in 5.8 / 27.1 / 64.0%, three or more in 0.7 / 6.7 / 16.0%; an optional board at the start in 10.5 / 44.5 / 65.7%; an optional move played in 1.2 / 2.5 / 1.1%; single-move turns 95.9 / 80.3 / 49.1% (a count of what was played, not of width). Part L's prediction holds (94.2% one mandatory board under 50). No threshold, no verdict: how it orders the D2 work is put to the user (issue 142); definitions issues 140-141, edits 110-115 |
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
  the pilot self-play run measures the engine's own distribution;
- I10's calibration monitoring: it measures nothing on random weights - the pilot
  run (the plan of 2026-10-17);
- C3's conditioning-offset training hook: with the deferred cached configuration
  (C2);
- 6p's other arms (per-square attacked-by, defended-by, pinned and in-check counts
  in the embedding; registers initialised from aggregates, D7): after the pilot run,
  on the shared dataset (M24);
- MLH steering (I3's match play): it waits on D4's moves-left head (net lane, after
  engine round 2) and round 2's per-leaf moves-left, and the documents give no rule
  for when an outcome is "settled" (issue 300) - resumed behind a flag, off, once
  those are in main and issue 300 is ruled; early adjudication (§12.3) has no
  specification (issue 300) and is not built.

Unparked 2026-10-22 (`lane/engine`): the shared stores' locks on Windows. Each shard
of `ConcurrentSolvedStore` and `DisproofStore` is now behind a `std::mutex` (`ShardLock`)
in place of a `std::shared_mutex` that this toolchain did not make exclusive on its first
concurrent use - correct by construction on every platform, with tests that failed on
every run of the old lock (H17's row). Left to the main loop: the Lazy SMP timing test
before the merge, and whether M17 is re-run on a quiet machine with the fixed lock (issue
147; its verdicts' soundness stands - every refutation and mate was confirmed by E4 alone).
Not yet confirmed on Linux (glibc): no GitHub Actions job has run since 2026-10-17
(billing), so the engine workflow is re-run once that is fixed.

Not parked any more, but not built by rule: the NNUE (F4, H6, H7) is held open
(CHECKPOINT 4); rung 1, rung 3 and rung 5 are retired (M5). Added to §14.4 at
CHECKPOINTS 6-7, held open: our own next move as a head (Lc0's `policy_next`), the
optimistic policy in self-play (D9 is match play only), contempt, Hunt with a lazily
expanded root (H49), value-side Shaw vectors, LMR (moot); refused: ONNX/TensorRT
export (for now; G9, M5), KataGo's convolution-era methods, symmetry augmentation
beyond colour.

## Spec issues (wrong, ambiguous, contradictory, silent)

Each: where, the evidence, what was done (never an improvised committed-class
decision), and what is parked.

**Numbering from 2026-10-17** (the main loop's three lanes, under M23's rule): a new
issue takes its number from its lane's block, so lanes never collide - the gate lane
122-139, the engine lane 140-199, the network lane 200-299, the main loop 300 and up;
gaps between blocks are expected. An issue the CHECKPOINT 6-7 documents settle is
marked *Settled upstream at CHECKPOINT 6/7*, with the register row or section that
settles it (*in part* where some of it stands); one an earlier ruling settled,
superseded or made moot says so; nothing is deleted.

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
   *Settled upstream at CHECKPOINT 6/7 in part: (d) - §2.2 (the has-moved flag,
   5DFEN's marker, for pawns, rooks, kings and brawns) and (h) - §11.0 ("the unmoved
   marker means 'has not moved'"); (a)-(c) and (e)-(g) are readings of the notation
   specification and stand.*
3. **Notation spec "Hashing" leaves the full-state order open within a turn**
   (README:486-492 sorts by timeline then turn; each turn has a white and a black
   board). The canonical fingerprint puts white's board before black's (time
   order). It also marks the hashing standard "still just a proposal"; it is used
   as the H30 fingerprint because it is the pinned specification's own canonical
   form, with both published examples as conformance tests.
   *Settled upstream at CHECKPOINT 6/7: H30 and §11.0 "Running the gate" (white's
   board before black's within a turn).*
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
   *Settled upstream at CHECKPOINT 6/7: H18 and §11.3 "The full key, as built" (the
   board size in the key).*
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
   *Settled upstream at CHECKPOINT 6/7: M20, with B8 and §5.3 (a, b), §9.2 (c), D1
   and §8.1 (d), B5 standing as v1 (e), B15 making (f) moot, D4 and §12.4 (g-i: the
   reply's move space; the label set standing as v1; moves-left in turns), D3 and
   §8.3 (j: the mean).* *(f)'s switch went with B6 when B15 was built (`lane/net`,
   2026-10-17).* *Built (`lane/net`, N-M20, 2026-10-17): (a, b) the registers
   position-free as ruled, `global_register_pairs` and `board_register_spatial` removed;
   (c) the squares for moves-left and effort, the global registers for the value-type
   heads (issue 223), `aux_global_pool` removed; (d) the offset, `policy_promotion`
   removed (issue 222); (e, h) relabelled as v1 values; (g) the reply's mapping is the
   heads' (after engine round 2); (i) turns (issue 224); (j) the mean, tested.*
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
    *Settled upstream at CHECKPOINT 6/7: §11.0 item 1 (in Reversed Royalty the royal
    is the queen; the king is a common king, not royal).*
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
    *Settled upstream at CHECKPOINT 6/7: §6 "As built - v1 values", B9 and B12 (a-g,
    the smolgen form and its rank, the 14.7 MMAC figure); M20: 16a-g stand as v1.* *(d)
    is B11's five types since travel-threat's fold (`lane/net`, N-B11, 2026-10-17).*
    *Relabelled (`lane/net`, N-M20): `config.py` documents (a-g) as the v1's values, no
    longer placeholders; the switches stay for ablations 4 and 6a.*
17. **E's "some live piece"** (§10.4.1) is read as the mover's pieces: E comes
    from the move generator at the node, and the opponent's destinations enter at
    the child (§10.4.3's "search re-partitions per node"). R1 now makes the
    both-colours reading buildable; whether to is a scope choice, not built.
    *Settled upstream at CHECKPOINT 6/7: §10.4.1's E row (the mover's moves at the
    node; the opponent's destinations enter at the child).*
18. **Source boards outside W** (src_extra / src_only): the head of an inactive
    timeline is playable but gets no band; the spec does not say whether scope
    must hold the source square of a move for the policy pointer. Live in the
    corpus: src_only(8) > 0 at 9/7/7/8/13 distinct positions by size bucket.
    Counted apart, never folded into |E|.
19. **Travel-threat** is named in §6 and B11 and used as a class of forcing moves
    in §11.1; no edge is defined. Not built.
    *Settled upstream at CHECKPOINT 6/7: B11, §6 and reversal 26 (travel-threat
    folded into attack).* *Applied in the network (`lane/net`, N-B11, 2026-10-17): no
    network relation carries it; the C ABI's id 6 stays reserved.*
20. **"Per GPU's CPU allocation"** (§15.6 M9) is undefined: how many cores is one
    GPU's share? The bench reports per thread and per machine.
    *Settled upstream at CHECKPOINT 2: M13 and §15.6's "CPU allocation" (8 cores per
    GPU as the reference, 2 of them for expansion; M9 judged on those 2).*
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
    *Settled upstream at CHECKPOINT 6/7: §10.4.3 (branch origins belong to the
    mandatory core).*
24. **W(k)'s unit and the present band**: k counts half-turns (one board each);
    the present band is read as the active heads.
    *Settled upstream at CHECKPOINT 6/7: §10.4.3 (k counts half-turns along a
    timeline; the present band is the active heads).*
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
    *Settled upstream at CHECKPOINT 6/7: §10.4.2 step 1 and register part K
    (measured to 186 boards, provisional above; the pilot run supplies the engine's
    own, M18).*
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
    *Settled upstream at CHECKPOINT 6/7: D1 and §8.1 "Which moves are scored" (those
    not proved illegal, each legal or unknown).*
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
    *Settled upstream at CHECKPOINT 6/7: §8.5, C11 and H18 (identity adds each
    timeline's branch origin and whether a turn is in progress).*
31. **A3 against C2/C17**: a frozen board's Δt-to-present and active flag change
    after it freezes. `frozen_features = intrinsic` (placeholder: frozen boards
    carry neither; a refresh gives them the anchor's values) | `current` (every
    shift of the present recomputes every frozen board, a cost §7.1 does not price).
    *Settled upstream at CHECKPOINT 6/7: B16 ("settles BUILD_LOG issue 31"), M20,
    §2.3 and §4.1 (intrinsic features; status as score-time key biases).* *Built
    (`lane/net`, N-B16, 2026-10-17): B16 on intrinsic frozen features; the refresh as
    built; issues 213-216.*
32. **C20's "causal past plus the conditioned history of its refresh epoch"**:
    `supersession_scope = causal_past` (placeholder) | `past_and_epoch_history`. The
    union has a cycle under the creation reading unless the epoch's history is a
    fixed block (history boards read only history), which is how it is built.
    *Settled upstream at CHECKPOINT 6/7: C20, M20 and §7.3 (the epoch's history a
    fixed block; both readings built, 6c decides).* *Relabelled (`lane/net`, N-M20):
    `config.py` calls `causal_past` the default arm until 6c, not a placeholder; the
    code was already both readings, each through §15.2's C20 row.*
33. **The §6 structure on frozen rows**: under the cached configuration the move
    graph, the relations head, typed biases and smolgen open or bias live query
    rows only (forced: they are present information).
    *Settled upstream at CHECKPOINT 6/7 in part: B11 and §6 (masks and typed biases
    on live query rows only); §6's smolgen paragraph does not say it (proposed edit
    31).*
34. **C15 against G1.** §10.3 says FP8 K/V obey §10.2 (cache and live share
    precision); that holds only for an FP8 trunk (G2, not built) or a network
    trained with FP8 frozen K/V (`frozen_kv_format = fp8`, built). With a BF16
    trunk, FP8 frozen K/V are RMS 8.7e-3 from the BF16 network - 3.8x the
    BF16-vs-FP32 gap.
    *Settled upstream at CHECKPOINT 6/7: C15 and §10.3 (FP8 frozen K/V obey G1 only
    with an FP8 trunk or FP8-frozen-K/V training).*
35. **§7.3's supersession bound** (+12-24% upper bound) assumes one board reading
    all of T. Under the creation reading the cascade breaks it (analytic +194% to
    +966%; CPU +3,273% at 600 boards, 23 timelines, 280 boards recomputed for one
    jump); under lineage the counted cost is 0.3-12%.
    *Settled upstream at CHECKPOINT 6/7: §7.3 (0.3-12% counted under ancestry; the
    creation cascade recorded).*
36. **§7.1's "~8 live head boards" against real games.** The corpus's real games
    have 24 timelines at 240-399 boards: the paper cached-vs-dense ratio is then
    9.9x at 240 boards (not 29x) and 24.8x at 600 (not 49x), and the live-set
    ratio 10.8x (not 3.7x).
    *Settled upstream at CHECKPOINT 6/7 in part: §7.1 and §13 (the paper ratio per
    timeline count); the live-set ratio and §15.3's "29x on paper" row are not
    restated (proposed edit 34).*
37. **§10.2's per-node tree memory** assumes about 528 novel tokens stored per
    node; under C19 the heads are recomputed every node and only superseded boards
    are stored: 1.08 MB each in BF16, 1-2 per node.
    *Settled upstream at CHECKPOINT 6/7: §10.2 (only boards superseded inside the
    search are stored, one or two per node).*
38. **§15.3's "KV memory per state"** counts every token; the cache holds frozen
    boards only (plus their trunk outputs, about 16 MB at 240 boards).
    *Settled upstream at CHECKPOINT 6/7: §15.3's KV row and §10.3 (frozen boards
    only, plus their trunk outputs).*
39. **"One refresh ≈ 10 node-evaluations"** is the windowed figure; with full
    history the model gives 29x at 240 boards and 49x at 600. The row stays
    unmeasured until the window exists.
    *Settled upstream at CHECKPOINT 6/7: §15.3's "One refresh" row (the windowed
    figure void; 29x and 49x with full history, to measure).*
40. **§7.6 says 0.164 TF per node; §7.9's table uses 0.159 TF** (528 queries, no
    global registers), giving +3.8% at 800 sims where §15.3 says +3.6%.
41. **Rung 1's trigger**: a move that creates a timeline (a non-branching jump does
    not); a fork moves the node and its descendants still in the old epoch.
42. **C15's varlen/ragged kernels** do not exist on CPU; not built.
    *Settled upstream at CHECKPOINT 6/7: M19 and §15.3 "Without a GPU" (varlen
    kernels deferred to GPU time).*
43. **The block-causal full pass** (training under the scope, and history builds)
    costs 2.3-2.8x a dense pass on this CPU: per-row scope masks defeat the fast
    path. A block-sparse scope selection would fix it; not built.
44. **H37's "exhausted budget"**: the engine also has a memory ceiling, treated
    the same way (unknown; in the gate, an `error` end naming it). Proposed spec
    edit 38.
    *Settled upstream at CHECKPOINT 6/7: H37 and §11.0 item 3 (nodes, time or
    memory).*
45. **§11.0 item 3's two detectors in the gate**: the spec says "naive as the fast
    path, hypercuboid for near-mate positions"; the gate's adapter runs them as a
    portfolio on doubling node budgets (hypercuboid first), the other
    cross-checking. Proposed spec edit 39.
    *Settled upstream at CHECKPOINT 6/7: §11.0 item 3 and H22 (the portfolio in the
    gate's adapter).*
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
    *Settled upstream at CHECKPOINT 6/7: D1 and §8.1 (the legality policy as built;
    issues 46, 90).*
47. **Register H22's wording** still says "phantom boards for the mate-vs-stalemate
    verdict" and "per-board chess rules as 5D queries", which H36 and H40 replaced.
    Proposed spec edit 40.
    *Settled upstream at CHECKPOINT 6/7: H22's wording (proposed edit 40).*
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
    *Settled upstream at CHECKPOINT 6/7: D1 and §8.1 "Which moves are scored".*
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
    *Settled upstream at CHECKPOINT 6/7: §11.1 "As built and measured, for the
    record" (the forcing set).*
52. **"Immediate threat"** (§11.1's branching moves "with an immediate threat") is
    undefined: placeholder `ArrivalCapture` - the arriving piece, on the passed new
    board, can capture an enemy piece worth at least 300.
    *Settled upstream at CHECKPOINT 6/7: §11.1 "As built and measured" (the arriving
    piece capturing an enemy piece worth at least 300).*
53. **"Depth-limited" (§11.1) against "node cap" (H3, "defining quiet")**: built as
    a node cap with iterative deepening in forcing half-turns.
    *Settled upstream at CHECKPOINT 6/7: §11.1 "As built and measured" (node-capped
    alpha-beta with iterative deepening in forcing half-turns).*
54. **Forcing moves and 5D obligations**: a forcing action is a legal action that
    begins with a forcing move; the rest of the turn is finished by a completion
    rule (a parameter: quiet moves by the least valuable piece by default).
    *Settled upstream at CHECKPOINT 6/7 in part: H43 and §11.4 name the canonical
    completion a placeholder for Hunt; the retired scanner's completion rule is not
    recorded.*
55. **"All evasions" in 5D** are all legal actions of a turn - a product over
    boards. The scanner lists them up to a cap (512); a capped node gives an
    estimate and is not fully resolved. In-check listings take 52% of the scanner's
    nodes.
    *Settled upstream at CHECKPOINT 6/7: §11.1 "As built and measured" (a loss proof
    needs the defender's legal actions enumerated exhaustively; a capped node yields
    an estimate).*
56. **5D piece values**: none in the spec. P N B R Q at the chess convention;
    unicorn 250, dragon 200, princess 700, brawn 125, common king 300 are
    placeholders; royals 0. The eval counts every timeline's latest board.
    *Settled upstream at CHECKPOINT 6/7: H50 (the values as placeholders,
    provisional).*
57. **"Node"** is undefined: the scanner counts a move applied, a royal-capture
    scan, or one node of an E4 search it calls.
58. **6b's terms are undefined**: "fully resolved", "scan leaf", "quiet", the
    evaluation rate when none is measured (no GPU here: §15.6's design rates,
    9,600/s at 240 boards and 4,600/s at 600, are used), threads (1, and physical
    cores - 1), pooled or per-size nodes/s; and stand-pat validity (operationalised:
    at a quiet position, every legal action - or a sample above 32 - answered by the
    opponent's 2-half-turn forcing search; invalid if every action loses its static
    eval).
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b block, "As measured" (stand-pat
    and the evaluation-rate fallback; the other terms by reference to
    MEASUREMENTS.md).*
59. **Draws as solved values**: H2 names wins and losses; the scanner exports
    stalemates as Draw(0) proofs.
    *Settled upstream at CHECKPOINT 6/7: §11.1 "As built and measured" (stalemates
    export as draw proofs).*
60. **Proof distance** is an upper bound in half-turns.
61. **Is 6b's root test pooled over a corpus or per phase?** The pooled figure
    rests on the corpus's many small positions; above 50 boards - the design's
    regime - the root scan fails (preliminary). M7's window rule is per phase.
62. **§15.6's rung-1 "measured evaluation rate"** does not exist without a GPU.
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b block (the design evaluation
    rates, 9,600 and 4,600/s, without a GPU).*
63. **§15.6 does not write the window's multiplier formula.** "A 1.5× all-in
    multiplier, computed with 90% of compute at the cached per-node cost ... and
    10% at the dense cost": of three readings, only the log-weighted
    m = m_cached^0.9 · m_dense^0.1 reproduces §15.6's own ceilings and multipliers
    (MEASUREMENTS, M7). Only the early phase's verdict depends on the choice.
    *Superseded at CHECKPOINT 4: M16 and §15.6's "Three readings, fixed at CHECKPOINT
    4" give the time-share (harmonic) combination M = 1 / (0.9 / m_cached + 0.1 /
    m_dense) - "not a weighted geometric mean"; edit 51 withdrawn.*
64. **"Phase" is undefined for M7.** The rule and the kill clause are "per phase";
    whether a phase is a board-count bucket (which?), and whether its median N or
    each position's own multiplier stands for it, decides the early game: the
    50-board bucket is on at k = 4 (1.60x), 10-board phases put under 20 boards
    off, the per-position median is 1.45x.
    *Settled upstream at CHECKPOINT 6/7: §15.6 "The window" (a phase is a
    board-count bucket judged at its measured median scope).*
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
    *Settled upstream at CHECKPOINT 6/7: H43 and §11.4's Hunt row (checking first
    moves, each completed into one canonical turn - a placeholder).*
68. **"Disproven" in M15 is class-relative** ("no mate within d by the class"), and
    256 of the 356 disproofs under 50 boards are roots with no checking action at
    all. Do those count toward the 50%?
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b′ block, "Readings used"
    (class-relative disproofs counted, roots with no checking move included).*
69. **"Hypercuboid-pruned completion" is undefined.** E8 reads it as sound pruning
    (standing captures, dead passes) plus E4's detector at the defender's turn
    nodes and E4's listings where turns do not decompose; enumerating completions
    with the hypercuboid search itself was not built.
70. **The estimate channel's depth, move set, local model and margins** are
    unspecified (E8: 3 plies, threats at the first, created boards passed, local
    check, delta pruning).
    *Moot since CHECKPOINT 4: the per-leaf estimate channel is retired (H2; §11.4
    "Retired"); edit 57 withdrawn.*
71. **"Fits the rung-1 leaf budget"** needs operationalising (E8, fixed before
    measuring: p90 of node cost over 5 × measured nodes/s / the size-matched
    evaluation rate).
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b′ block, "Readings used" ("fits"
    as built).*
72. **M15's "keep whichever decides more"** has no tie rule, and brute force's and
    df-pn's "decided" are defined differently.
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b′ block (no tie rule, "which the
    result never needed").*
73. **The semimove decomposition is exact only under free-order rulesets** (R8,
    the count rule, Rewinds, no play activation, castling on optional boards);
    §8.2 and §11.0 item 3 assume order independence. E8 falls back to E4's
    listings elsewhere.
74. **Disproofs need a home**: H19 and §11.3 cover proofs only; E8 keeps
    class-relative disproofs in a separate store, never the solved store.
    *Settled upstream at CHECKPOINT 6/7: H19 and §11.1 "As built and measured" (a
    separate disproof store keyed by position, class and depth).*
75. **C24's rule is unspecified** (§7.9, register C24): four candidate signals
    joined by "or", with no combination rule, threshold or weights. Built as a
    weighted sum against a threshold, every number a named placeholder (threshold
    1.0; 1/128 per board, 1/8 per active timeline, 1/2 per travel move, 1/12 per
    effort bin). Tuning belongs to 6c, if rung 1 shows a signal.
    *Settled upstream at CHECKPOINT 6/7: C24 and §7.9's "As built" (a weighted score
    against a threshold; every number a placeholder, tuned in 6c).*
76. **C24's visit gate and per-search bound**: the spec gives none. §7.9's
    breakeven argument (one whole-scope refresh is about ten node evaluations)
    applies, so the build uses rung 1's V = 50; "epochs must be bounded per search"
    has no number - C24 is capped at 8 refreshes per search (rung 1's are neither
    counted nor blocked). C24 never refreshes the root (C3's job).
    *Settled upstream at CHECKPOINT 6/7: §7.9's "As built" (rung 1's visit gate; at
    most 8 refreshes per search; never the root).*
77. **"Travel since the last refresh" is undefined**: built as every travel move,
    branching or not, on the path from the node's epoch anchor (its nearest
    refreshed ancestor, else the search root) down to the node, its own move
    included.
    *Settled upstream at CHECKPOINT 6/7: §7.9's "As built" (every travel move from
    the node's epoch anchor down to the node).*
78. **The effort head as a node-level signal**: §8.6's labels are root-search
    efforts; C24 reads the parent's evaluation off-label at internal nodes - the
    bin at quantile 0.5 (placeholder). An unevaluated parent contributes nothing.
    *Settled upstream at CHECKPOINT 6/7: §7.9 and §8.6 (the parent's predicted
    difficulty at a quantile; the effort head's second consumer).*
79. **Does C24 sit on top of rung 1 or replace it?** The spec says "a ladder
    variant", the register "add-only if self-rated". Built as an OR with rung 1
    decided first and unchanged, add-only for every signal against the flag-off
    configuration; whether a structural signal may ever subtract from rung 1's
    trigger is not ruled.
    *Settled upstream at CHECKPOINT 6/7: C24 and §7.9 (rung 1 decided first and
    unchanged; every signal only adds).*

80. **The submit prior** (D8, §8.2) until the network's submit logit exists: a
    placeholder `submit_share` 0.5 of a decision's prior. With logit 0 beside N
    flat move logits, untrained actions played nearly every optional board (one
    game: 101 moves in 14 actions, 54 timelines by action 16).
81. **§8.2's whole-turn action prior** from single-move logits: the sum is built,
    the mean a flag.
    *Settled upstream at CHECKPOINT 6/7: §8.4 "Constants as built" (the sum; the
    mean behind a flag).*
82. **E1's forms**: entropy-adaptive top-k widening, FPU (applied to unvisited
    *instantiated* children - §8.4 says uninstantiated), root Dirichlet α at 5D
    branching - every constant a placeholder.
    *Settled upstream at CHECKPOINT 6/7: E1 and §8.4 "Constants as built" (FPU on
    unvisited instantiated children; the root noise now a total concentration split
    across the legal moves).*
83. **E2's and E3's formulas and constants** (behind flags).
    *Settled upstream at CHECKPOINT 6/7: §8.4 "Constants as built".*
84. **E9**: "a visit threshold for solved nodes" read as a per-child visit cap of
    8; the secure child as Q − A/√n with A = 1.
    *Settled upstream at CHECKPOINT 6/7: §8.4 "Constants as built" (a visit cap of
    8; Q − A/√n with A = 1).*
85. **E5's LCB**: z = 3, a 15% visit floor, categorical weight 0.
    *Settled upstream at CHECKPOINT 6/7: §8.4 "Constants as built" (z = 3, a 15%
    floor, categorical weight 0).*
86. **MCGS residuals** (C11): how a shared node's value is backed up through
    several parents.
87. **§8.5's "values depend only on the state"**: the state must include whether
    an action is in progress - a turn start and a mid-action state can share
    every board and the acting player (two reviews found graph search merging
    them, pruning a legal mating submission). Built: MCGS identity is (key, turn
    start); the network is never told whether a leaf is a turn start.
    *Settled upstream at CHECKPOINT 6/7: §8.5, C11 and H18 - and A6: the network now
    tells a turn start from a mid-turn state, through the turn-state inputs (§2.3).*
88. **The turn cap's definition** (I11) and its record: "*" with a Termination
    header for capped records; t5d-play's default 300-board cap (a uniform
    Standard game reached 509 boards before turn 20).
    *Settled upstream at CHECKPOINT 6/7: I11 and §12.3 (the 300-board cap; a capped
    record's result and Termination header).*
89. **C14**: the evaluation cache shared across self-play's fresh searches,
    keyed by state, purged by board count at creation and re-root.
    *Settled upstream at CHECKPOINT 6/7: C14 and §8.7 (the shared, state-keyed
    evaluation cache, purged by board count).*
90. **The legality policy (a)-(f) as built** (issue 46, proposal 43): unknown
    moves kept but never feeding proofs, verified when first selected.
    *Settled upstream at CHECKPOINT 6/7: §8.1 (issues 46, 90).*
91. **The material evaluator's piece values** follow issue 56's placeholders.
    *Settled upstream at CHECKPOINT 6/7: H50.*
92. **D8's whole-turn half is unbuilt**: the whole-turn arm has no end-of-turn
    token (the submit logit is ignored there; the C API reports submit_possible
    = 0).
    *Settled upstream at CHECKPOINT 6/7: D2, D8 and §8.2 - the flat form supersedes
    the whole-turn arm and ends turns itself, so no end-of-turn token.*
93. **Memory**: about 80 bytes per eagerly stored edge; the whole-turn listing
    cap lowered from 4096 to 256 actions per node (1.5 MB per node at 103 boards
    before); actions are not stored lazily.
    *Settled upstream at CHECKPOINT 6/7: D2 and §8.2 (the enumerating whole-turn arm
    and its 256 cap superseded by the flat form).*
94. **The search's mate-check budget** against the near-mate tail.
95. **At large N, legality classification dominates self-play time** (41 of 43 s
    at 336 boards; 56 s of the board-capped Standard smoke games).
    *Settled upstream at CHECKPOINT 6/7: recorded in §13 and as §15.3's "Legality
    classification per expansion" row (measured, CPU); the cost itself stands.*
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
    *Settled upstream at CHECKPOINT 6/7: H28 and §11.0 "Running the gate" (the
    export-form exception); §11.0's "Third-party code" row still says only "the game
    corpus is in its format" (proposed edit 71).*

97. **Guard under the ply split is checked at submission** (a reading: the
    opponent moves only after it).
    *Settled upstream at CHECKPOINT 6/7: §11.4's Guard row (under the ply split it
    is checked at submission).*
98. **Guard's depth**: d = the opponent's actions, 2d − 1 half-turns; depth 1 =
    their mate in one.
    *Settled upstream at CHECKPOINT 6/7: H42 and §11.4's Guard row (depth d = the
    opponent's d actions, 2d − 1 half-turns).*
99. **"Existence, not the full listing" removes only our listing**: exactness
    still needs every opponent action, a product over their boards; depth 1 fails
    at 50 ms (measured). The cheap reading - one escape per checking move with the
    rest of their action held quiet - is inexact and not built.
    *Settled upstream at CHECKPOINT 6/7: §11.4's 6b″ result ("removes only our
    listing").*
100. **Lazy SMP (M13's 5 threads) gives Guard's exhaustive refutation little**:
    helpers re-search the same product.
    *Settled upstream at CHECKPOINT 6/7: H17 and §11.4 (Lazy SMP's helpers only
    re-search the same product).*
101. **"Prove-only ... never spends time disproving"**: read as stop at budget,
    claim nothing, share no disproofs; df-pn's usual thresholds kept (measured
    better than proof-number thresholds alone).
    *Settled upstream at CHECKPOINT 6/7: H43 and §11.4's Hunt row (it stops, claims
    nothing and shares no disproofs).*
102. **Hunt's and the truth's forcing class remain placeholders** (checks,
    canonical completion) - carries issue 67.
    *Settled upstream at CHECKPOINT 6/7: H43 and §11.4's Hunt row (a placeholder);
    the truth's class union in §15.6's 6b″ result.*
103. **Hunt's ground truth is the union** of the literal class's and the default
    class's proofs at 4M nodes each (2M tree cap), chosen from a 101-root curve
    that found no new mate past 500k nodes; "none" means no mate found, not no
    mate. Ruling 5 says "one long offline search".
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b″ result (the union of two
    attacker classes' proofs at 4M nodes each).*
104. **6b″'s chosen action for Guard is the record's next action** (no search yet).
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b″ result (the chosen action is
    the record's next action).*
105. **A Hunt range with no truth mate has no verdict**; the rule is silent.
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b″ result (a range with no
    established mate has no verdict).*
106. **The asset map's estimate and boost** (4 plies, 128 nodes, flag at 300,
    boost 0.25 × min(1, value/900)) are placeholders.
    *Settled upstream at CHECKPOINT 6/7: H13 and §11.2 "As built (v3)" (every number
    a placeholder).*
107. **Hunt's non-lazy root expansion caps it at 100+ boards** (measured: the
    100-199 range's 20 truth mates are mates in one); a lazily expanded root is the
    remedy, not built - the measured Hunt is the pre-registered one.
    *Settled upstream at CHECKPOINT 6/7: H49, §11.4 and §14.4 (a lazily expanded
    root, held open behind a fresh pre-registered test).*
108. **Where D8's acting-player input enters is unspecified** (§2.3/A3). Built as
    `acting_player_input`: a learned embedding added to the global registers
    (placeholder; frozen boards' K/V never see it) or to every token
    (`all_tokens`, frozen boards excluded under intrinsic features).
    *Settled upstream at CHECKPOINT 6/7: A6 and M20 (the acting player on the global
    registers).* *Built (`lane/net`, N-M20): `acting_player_input` and `all_tokens`
    removed; the embedding is on the global registers alone.* *With no global registers
    (an ablation-3 arm) it goes on the board tokens instead - issue 225 (N-F6heads-M20
    review fixes).*
109. **Which register the submit logit is read from, and how** (§8.2, D8): built
    as global register 0 (`submit_register`) with a linear readout (`submit_head`;
    `pointer` the other arm). Tension with B8/D7 ("registers stay free scratch",
    "never assigned aux targets"): §8.2 assigns the logit to a register
    explicitly; it is a policy output, not an aux target.
    *Settled upstream at CHECKPOINT 6/7 in part: D8 and M20 (the pooled global
    registers, so D7 holds); the readout form (linear or pointer, `submit_head`) is
    not ruled.* *Built (`lane/net`, N-M20): the logit reads the global registers' mean
    (`submit_register` removed); `submit_head` stays a placeholder - linear (257
    parameters) by default, the policy's bilinear form the other arm - awaiting a
    ruling or a named ablation.*
110. **The value/WDL perspective**: §8.3, §12.4 and N1's heads say "the player to
    move"; under D8 the search's API defines it as the acting player's - built so.
    *Settled upstream at CHECKPOINT 6/7: D3, §8.3 and §12.4 (the acting player's
    perspective).*
111. **Policy targets cover the submit child's visits**: §12.2's exclusion and
    provenance must cover them too; the whole-turn arm still has no end-of-turn
    token (issue 92).
    *Settled upstream at CHECKPOINT 6/7: §12.2 and §12.4 (moves and submit;
    exclusion and provenance cover the submit child); the whole-turn half by the
    flat form (D2).*
112. **An illegal submit**: masked by the network (no probability) and delivered to
    the search as NaN, which prunes the edge (the C API refuses infinities) - a
    convention to record.
    *Settled upstream at CHECKPOINT 6/7 in part: D8 and §8.2 (an illegal submit gets
    no probability); the NaN delivery convention is not stated.*
113. **F1's parameter count** is now 21,359,370 (+769: the acting-player embedding
    and the submit readout).
    *Settled upstream at CHECKPOINT 6/7: F1 and §9.2 (21,359,370 as built at N5).*
114. **Match play between two evaluators**: §8.7's "match play carries everything"
    assumes one engine. Built as one search and cache per evaluator, re-rooted
    with `advance_action`, never shared; match games are deterministic given the
    networks.
    *Settled upstream at CHECKPOINT 6/7: C14 and §8.7 (one search and cache per
    evaluator in match play).*
115. **The engine's `t5d_search_new_sharing` does not check** that donor and new
    search have the same expand, split and legality parameters (Python refuses a
    mismatch; the engine may want the same check).
116. **Refresh epochs in a shared cache** need the search's epoch set per root
    (`engine_epoch`, the epoch key hashed to 64 bits).
    *Settled upstream at CHECKPOINT 6/7: §8.7 and its persistence table (an
    evaluation is a function of the state and refresh epoch; C14).*
117. **6b″ counts only proofs E4 confirmed**; a refuted proof withholds every
    verdict; no cross-check, no verdict - a reading: the pre-registration does not
    say.
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b″ result (only engine-confirmed
    proofs count; a refuted proof withholds every verdict).*
118. **"Too few to read"**: a Hunt verdict resting on fewer than 10 truth mates is
    flagged (placeholder; the pre-registration names no minimum).
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b″ result (fewer than ten
    established mates flagged as too few to read).*
119. **Ruling 5's "pinned and hashed"** read as extending to the record list
    (content IDs in order), the rules by value and a code signature.
    *Settled upstream at CHECKPOINT 6/7: §15.6's 6b″ result (pinned with its
    provenance; the tools refuse other records or rules).*

120. **No GPU on this machine** (an RX 9070 XT without CUDA): the §15.3 figures
    that need one - FA4-class evaluations/s, the leaf-batching gain, G2's FP8
    attention, B14's FlexAttention kernel, C15's varlen kernels, §15.6's Phase-3
    movegen re-check against a GPU evaluation rate - cannot be measured here.
    §15.3 asks only that each figure carry its conditions. Put to the user: accept
    counted FLOPs plus CPU-labelled wall times as the untrained build's write-back
    with the GPU figures named as deferred, or provide GPU access.
    *Settled upstream at CHECKPOINT 6/7: M19 and §15.3 "Without a GPU" (counted
    FLOPs and CPU-labelled wall times; GPU figures deferred to rented NVIDIA time;
    one bounded AMD attempt).*
121. **Committed-class placeholders awaiting neither a ruling nor a named
    ablation** (§15.5's last clause): spec issues 9(a-j), 16(a-g), 31
    (`frozen_features`), 32 (`supersession_scope`), 108 (`acting_player_input`),
    109 (`submit_register` and its readout), and register B7 (deferred with no
    named ablation). Put to the user to rule, or to name the trained ablation each
    awaits.
    *Settled upstream at CHECKPOINT 6/7 in part: M20 (every placeholder but one
    disposed of; B7's gate now names ablation 2's head-allocation evidence); D8's
    readout form (`submit_head`, issue 109) is not on M20's list.* *Built (`lane/net`,
    N-M20): every disposition is the code's default (register row M20); the readings
    it took are issues 222-224.*

<!-- A new list, so that the numbers from 300 render as written. -->

300. **Early adjudication and MLH steering's "settled" outcome have no specification**
    (§12.3; register I3, I18). §12.3 gives match play "moves-left steering active —
    prefer faster wins and slower losses once the outcome is settled (Lc0's MLH
    pattern)" (I3: "MLH steering"), and self-play "resign thresholds and early
    adjudication", checked for false resignations (I18). No document says when an
    outcome counts as settled, how MLH's preference enters selection, or what
    adjudicates a game early and how such a game is scored and recorded (resignation
    has AlphaGo Zero's protocol, I18). Done: neither is built in the untrained build
    ("Parked, and why"); match play's noise-off, final-rule configuration is (I3). For
    the user with step 2's consolidated list; nothing in §15.5 waits on it.
301. **The tree form under MCGS** (§8.2, §8.5, §14.4; register D2, C11, C21, part K).
    Under graph search a mid-turn node is created by the first path to reach it, so
    the priors it inherits (from that path's last evaluated node) and whether it was
    evaluated at all (an interaction gate fires on the incoming move) are path
    properties; identity is the state with branch origins and whether a turn is in
    progress (§8.5, C11, H18), and §14.4 and C21 refuse path-dependent evaluations.
    The documents measure the arms jointly with MCGS (D2, part K, 6x with 6k) but do
    not say how the tree form and MCGS combine. Done (the plan): the engine lane
    builds the tree form under the tree search and refuses `graph` with it as a
    parameter combination - parked, not a design refusal; §15.2's tree-form row fires
    every gate, so nothing is inherited there and §15.5 does not wait on it.
    Candidates for the user: re-evaluate and re-prior an inherited node when a gated
    path reaches it; key tree-form nodes by their inheritance source as well; inherit
    only from the action's turn start. For the user before 6x and 6k, with step 2's
    consolidated list.
302. **A6's moves-made count against the evaluation's identity** (§2.3, §8.5, §8.7,
    §15.2; register A6, C11, C14, C19-C21, H18). A6 puts "moves already made this
    turn" into the network's input; §8.5, C11 and H18 add to identity only branch
    origins and whether a turn is in progress, and the evaluation cache is keyed by
    the state and refresh epoch (C14, §8.7). The keyed state does not fix the count
    (the plan's review found identical boards with counts 2 and 1, reached across
    turns), so two such nodes would share a cached evaluation the network computes
    differently - against §15.2's "same state + epoch → same evaluation". (Today's
    `EvalRequest` withholds even the turn-start bit: "a path property the network
    must not see".) Reading (the plan; engine ABI round 1): the count joins
    `EvalRequest`, the evaluation cache key, the batch de-duplication, Python's state
    digest and MCGS identity, where it subsumes the turn-in-progress bit; every
    ordering of one action's moves from one turn start has the same count, so MCGS
    keeps its intra-turn merges. Proposed edit 200.
303. **A7 under the cached configuration: frozen boards' K/V depend on the acting
    colour** (§2.2, §7.1, §10.3, §15.2's first row; register A7, C2, C17). A7 encodes
    every input from the acting player's side - ranks mirrored and timelines negated
    when Black acts - so a frozen board's keys and values differ by perspective, and
    cached = dense needs the cache to hold the acting player's. §2.2's "identity is
    unaffected" holds for state keys, not for cached K/V; no document prices the
    difference. Reading (the plan; net lane, with A7): frozen K/V keyed by perspective
    in the cached path (`cache.board_ident`'s `acting` option already keys by the
    acting player, for `acting_player_input = "all_tokens"` with `frozen_features =
    "current"`), and a refresh's history per perspective -
    up to about twice the frozen compute and KV memory, a cost for the deferred cached
    configuration (C2) to weigh. The dense v1 is unaffected. For the user with C2's
    decision. Proposed edit 201. *N-M20 (`lane/net`): `acting_player_input` and its
    `all_tokens` arm are gone; `board_ident(..., acting=True)` keys by the acting player
    under either frozen-feature reading and is unused by the cached path until A7.*
304. **§15.2's A7 row against A7's acting-colour input** (§2.2, §15.2; register A7).
    The row: a position and its colour-swapped mirror "give identical canonical inputs
    and outputs". A7 keeps the acting colour as a global input ("for the few
    colour-asymmetric facts"), and the mirror swaps the acting player, so the two
    canonical inputs differ in exactly that input and the outputs need not agree.
    Reading (the plan; net lane, with A7): the test checks that the acting colour is
    the only difference between the canonical inputs, and compares the outputs, mapped
    back, with that input equalised. Proposed edit 202.
305. **"Turn" in two units, and E7's allocator counts board-moves** (§2.3, §8.2, §8.3,
    §8.6, §12.4, §12.8; register A6, D3, D4, D8, E7, I11, I13, I19). D4 counts
    moves-left "in turns" (M20), and D3's short-term horizons, I13's band and I19's
    schedule ("cut at move 60") need a unit too. D8, A6 and §8.2 use "turn" for one
    player's action (a player "may keep moving ... until submitting"; "moves already
    made this turn"; "a turn with k mandatory boards"); I11's cap (`TurnCap`) and the
    notation count full turns, both players'. §8.6's moves-left integral projects N by
    "every board-move adds a board" and counts board-moves remaining. Reading (the
    plan's review; one owner, the main loop): a turn is one player's action, from a
    turn start to its submission - the unit of D4's moves-left, D3's horizons, I13's
    band and I19's schedule; I11's cap keeps the notation's full turns, as built;
    E11's allocator converts turns to board-moves by a boards-per-turn parameter per
    board-count range, its placeholder from M22's measurement until the pilot run
    measures the engine's own (M18). Proposed edit 203.
306. **E7's root-scan reserve has no named consumer while Guard is offline** (§8.6
    "Guards", §11.1 "Root rule", §11.4; register E7, H42). §8.6 holds "a reserve held
    for one re-search if that scan refutes", and E7 still lists the "root-scan
    reserve" (edited at CHECKPOINT 6/7); §11.4 names Guard's refutation as what draws
    on it ("triggers a re-search (§8.6's reserve)"), and Guard is adopted in no range
    (H42). No document says whether the settling scans under 50 boards - df-pn at the
    root and on PV changes - refute a chosen move and draw on it. Reading (the plan):
    E11 holds no reserve while Guard is offline; if the settling scans are ruled to
    trigger §11.1's re-search, the reserve returns for them, under 50 boards.
    Proposed edit 204.

<!-- A new list, so that the numbers from 122 render as written. -->

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
    *Ruled at CHECKPOINT 8: a ruleset is gated only when two or more oracles answer for it (H47) - gate step 3b.*
    *Built 2026-10-17 (gate step 3b): in every tier, each record of a ruleset followed as far as the engine went by at least two independent implementations (`gate.GATING_ORACLES`; a parse-only source never counts); nightly now gates Standard - Half Reflected (2 oracles behind its record) and Standard - Reversed Royalty (3), and no longer Standard - Turn Zero (cwmtt/tesseractMageOChicken: cwmtt alone). The reading is spec issue 131; proposed edit 103.*
123. **H47 is silent on an engine compared with nothing in a smoke test.** As
    built, an engine with no finding against it passes the commit tier's
    differential even when it declined every record (nothing was compared, so
    nothing was found); nightly and release fail it (`engine-nothing-gated`),
    and parse mode keeps its own floor (`engine-nothing-compared`) in every tier.
    The self-tests pin it (`test_gate_live`, `GateRunTests`), so the consequence is
    visible. Put to the user: should the smoke test hold full mode to parse mode's
    floor - the engine replayed at least one record, and an oracle's answer was
    compared with it? Nothing is parked on it.
    *Ruled at CHECKPOINT 8: yes - the commit tier fails if the engine declines a record it read on the last green run, or reads none (H47) - gate step 3b.*
    *Built 2026-10-17 (gate step 3b): the last green run is the engine's baseline, `oracles/engine-baseline.json` - written by `harness.py baseline --write` from a green run of the commit tier, pinned in `gate.toml` `[baseline]`, recorded from the engine as built on `lane/gate` (35 of the 41 commit records replayed); every tier fails an engine that declines a record it read then (`engine-declined`) or reads none (`engine-read-nothing`). What is recorded, "declines", "reads none" and why every tier: spec issue 132; proposed edits 103-104.*
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
    *Ruled at CHECKPOINT 8: no - a restricted run never produces a tier verdict; its output says "restricted - not a tier result" (H47) - gate step 3b.*
    *Built 2026-10-17 (gate step 3b), undoing the reading above: a run restricted with `--source` or `--record` is labelled "restricted - not a tier result" in its summary and report, states no ruleset gated or not, makes none of the checks of a tier as a whole, keeps the proof channel DISABLED and exits on its findings only (spec issue 133; proposed edit 103).*
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
    *Settled upstream at CHECKPOINT 8: register H51 and §11.0's "Capture markers are
    read leniently" (a project decision) - a move is read as its squares define it,
    a mismatch logged as a warning, the marker written strictly; built 2026-10-17
    (H51's row; rulebook R11; KD-0167 to KD-0169 withdrawn, ftxi's refusal KD-0170).*
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
    *Ruled at CHECKPOINT 8: report it too, as a question citing the source and a few representative positions, not as a bug (H27) - drafted in gate step 3b; the user files it.*
    *Built 2026-10-17 (gate step 3b): `docs/checkpoints/upstream/ftxi-check-test-passed-boards.md` asks ftxi's author which boards the check test should pass - `state::phantom()` passes every timeline whose last board is the player's, R1 only the active boards in the present - citing the sources R1 and C7 rest on as the rulebook ranks them (the current Guide, silent on it; the older guide text and the developer's remark, "in the present"; Wikipedia's sentence, which R1 adopts, wiki.gg's Exiling page and TV Tropes, with C7's caveat; the forum statements C7 cites, on boards ahead of the present and on inactive timelines) and the evidence the other way, with five positions, one from each of CHECKPOINT 1's rule questions RQ1-RQ4, re-observed through the pinned library (the captures ftxi's royal-capture scanner lists on the phantom, `check_position::for_phantom(...).checks()`, the view behind `has_phantom_check()`). The 147 R1 entries' `upstream` says the question is drafted, to be filed by the project owner; no binding or citation changed. Proposed edit 99 (b) applies: no edit until it is filed.*
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
    *Superseded by H51 (2026-10-17; H51's row): nightly now also gates Standard - Turn
    Zero, and one of its records, cwmtt/tesseractMageOChicken, has one oracle behind it
    (cwmtt) - so the two-oracle rule would ungate Turn Zero, and still gate Half
    Reflected and Reversed Royalty.*
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
129. **H51 and the measurements pinned before it** (§11.0 "Capture markers are read
    leniently", register H51; §15.6's 6b, 6b′ and 6b″ and the user's ruling 5 on Hunt's
    ground truth - "recorded once, pinned and hashed" - registers M8, M12, M15, M17,
    H43). H51 makes the reader's default lenient. 6b, 6b′, 6b″ and the truth built
    their roots from the corpus records under the reading before it, the default of
    their day: a written `x` must capture. Read leniently, cwmtt/futures - the 8th
    record of the list - goes on past its action 17, and 6b's construction (every turn
    start, the first of equal positions kept) would take its positions 17 to 19 from
    it rather than from ftxi/futures, later in the list, which holds the same three:
    the roots from 169 on renumbered, and root 168's next action - that jump -
    readable, so 6b″ would count that root for Guard. The truth's readers refuse a
    truth whose roots are not the records' (record, index, fingerprint), and the
    pinned figures would no longer rebuild. The documents do not say which reading a
    measurement pinned before a reader change keeps. Built: the tools that rebuild
    6b's roots - `t5d-scan6b`, `t5d-scan6bp`, `tools/sixb_roots` (so `t5d-scan6bpp` and
    `t5d-hunt-truth`) - read with the reading they were measured under, an explicit
    option (`tools/tool_support.hpp`, `sixb_pgn_options`: the defaults but
    `CaptureMarks::MarkedMustCapture`); checked against the parent commit's build -
    6b″'s tool accepts the pinned truth (all 825 roots), and 6b's and 6b′'s tools list
    the same 825 positions. The truth was not re-recorded, 6b″ not re-run. The other
    reading - every pinned measurement read under H51 - means re-recording the truth
    and re-running 6b, 6b′ and 6b″ on a quiet machine, for one root's Guard
    eligibility and three roots' attribution. Proposed edit 101. Nothing is parked on
    it.
130. **Upstream reports for a divergence that a project decision settles** (§11.0 "A
    known-divergences file": "if the oracle is wrong, the divergence is recorded with
    its rulebook citation (and reported upstream — defects as bug reports,
    disagreements with a documented rule as questions citing the source ...)";
    register H27, H51). ftxi's refusal of cwmtt/futures action 17 is recorded (KD-0170)
    citing R11 - a project decision, where the notation specification is silent and no
    other source speaks. It is no defect of ftxi's (its strict reading of the mark is
    one a reader may take where the text is silent: its own action listing holds the
    move), and there is no documented rule to put a question against; §11.0 and H27
    name only those two kinds of report. Built: `upstream = "unreported: ..."` with
    that reason, which `verify` accepts. Proposed edit 102. Nothing is parked on it.
131. **"Two or more oracles answer for it", as built** (§11.0 "Tiered CI": "a ruleset
    is gated only when two or more oracles answer for it"; register H47: "as built
    the gate required one, and is aligned to two"; the ruling on issue 122). The
    documents do not say what answering for a ruleset is. Built - a reading: the
    gate's own "standing behind" a record (`gate.engine_differential`'s vouching,
    unchanged), counted per record: a ruleset is gated only if each of its records
    was followed, as far as the engine went, by at least two independent
    implementations (`gate.GATING_ORACLES` = 2, over the count `Gated.oracles`
    already reported). Two searches of one implementation count once (ftxi and
    ftxi:naive are one oracle); the notation parser, which judges nothing, never
    counts; an oracle that replayed the record otherwise or parted from the
    engine's position does not count, as before. A record that only one oracle
    follows ungates its ruleset with its own reason ("only one oracle followed
    <record> (<oracle>) as far as the engine went ..."), after "no oracle
    followed" and before the verdict floor. The floor on check, count and mate
    (`min_checked`) is unchanged: any oracle's agreeing verdict settles a position,
    so two oracles that answer different questions - cwmtt the counts, Chess in 5D
    check and mate - between them gate a ruleset. The other reading, two agreeing
    verdicts for each observable, would also ungate Standard - Half Reflected
    (cwmtt/test1: cwmtt alone counts, Chess in 5D alone gives check). Consequence:
    the commit tier, with one rules oracle, can meet the gating conditions for no
    ruleset (`gate.SMOKE_REASON` now shows only in a smoke test of a tier with two
    oracles that is not full strength). Nightly, run 2026-10-17: PASSED, gating
    Standard - Half Reflected (its record followed by 2 oracles) and Standard -
    Reversed Royalty (by 3); Standard - Turn Zero, which it gated since H51, is not -
    cwmtt/tesseractMageOChicken has one oracle behind it, cwmtt - and the Focused
    variants and Timeline Battleground, ungated before on the check floor, now give
    the one-oracle reason (cwmtt alone). Proposed edit 103. Nothing is parked on it.
132. **The commit tier's last-green-run rule, as built** (§11.0 "Tiered CI": "The
    commit tier still fails if the engine declines a record it read on the last
    green run, or reads none"; register H47; H33; the ruling on issue 123). The
    documents do not say where the last green run is on record, what declining or
    reading is, or which tiers compare. Built - four readings:
    (a) **The record.** `oracles/engine-baseline.json`: for each record of the tier
    `gate.toml` `[baseline]` names (`commit`), how the engine's replay ended (`ok`,
    `rejected:<reason>`, `unsupported`), how many positions it reported, and the
    SHA-256 of the record text it was given; with the engine's command and mode, the
    platform and the program's SHA-256, for the reader. Kept as H33 keeps recorded
    answers: written only by `harness.py baseline --write`, which runs the tier in
    full and writes only from a run green on everything else - never by a gate run,
    and never by the engine (the file is in the harness, hashed at the start and end
    of every run, and nothing outside the harness may name it); pinned by its SHA-256
    in `gate.toml` `[baseline]`, as the rulebook is (H38) - not in `golden/MANIFEST`,
    which every `golden --write` rewrites; verified against its pin before every
    read - every comparison (the snapshot), `verify` (`engine-baseline-missing`,
    `-integrity`, `-invalid`, `-stale`, the last also for a record of the tier it
    holds no line for), and `baseline --write`'s read of the baseline it replaces,
    which records nothing over a file that is not the one pinned (review fix
    2026-10-22); one record per line, `baseline --write` listing what the new file
    no longer holds (`LOSS`) first. Like MANIFEST's (H33), the pin catches an
    accident, not the file and its pin edited together: that is review's job. So "the last green run"
    is the last one recorded: a gate run never records itself (H33), and a later
    green run moves the baseline only when someone records it - which is also what
    keeps an engine that comes to read less from moving it unreviewed. Recorded
    2026-10-17 from a green commit run of the engine as built on `lane/gate`
    (t5d-gate, SHA-256 ec0b3944...): 35 of 41 records replayed - not cwmtt/ctp1
    (unreadable after 10 positions), ftxi/softmate and ftxi/verysmall (after 3),
    ftxi/niceAndrey (set-up refused), notation/invasion and notation/marauders
    (declined).
    (b) **Declines a record it read.** It reads less of the record than then: fewer
    positions; or as many without the verdict it ended with then - other than `ok`
    where the baseline ended `ok` (the definition the plan suggested), or other than
    a refusal as illegal (`illegal_move`, `cannot_submit`) where it ended in one
    (added: a refusal as illegal that becomes a refusal as unreadable at the same
    action reads less, and no differential finding shows it - both runs refuse that
    action). A record the baseline gives no position cannot be declined; an `error`
    or a time-out is `engine-failed` already and is not reported again; a record
    read further is no decline (the summary counts them, and a new baseline holds
    the engine to them).
    (c) **Reads none.** In a complete run of a tier the engine replays no record -
    none ends `ok` or in a refusal as illegal (the gate's own "replayed": a refusal
    as unreadable is no verdict). The second half of the parse-mode floor issue 123
    offered - that an oracle's answer was compared with the engine - is not built:
    the register's words name only reading.
    (d) **Every tier.** H47 names the commit tier, the one that gates nothing (in
    nightly and release an engine that reads none fails already, on
    `engine-nothing-gated`). But the baseline is a fact about the engine, not the
    tier: nightly and release run every record the commit tier runs, with the same
    engine and a time limit no shorter, and a tier that runs every oracle should
    check at least what the smoke test checks - otherwise nightly could pass an
    engine the commit tier fails on the same commit. A restricted run reports each
    decline (a finding about one record) but not reading none (a check of a tier as
    a whole; issue 133); parse mode compares nothing with the baseline, which holds
    full-mode reads. While the corpus has no nightly-only record, the commit-only
    reading would differ only in the findings nightly and release print.
    Proposed edits 103 and 104. Nothing is parked on it.
133. **Restricted runs, as built** (§11.0 "Tiered CI": "a run restricted to chosen
    sources or records is a developer convenience and never produces a tier
    verdict"; register H47; the ruling on issue 124, "label its output 'restricted -
    not a tier result'"). Built: any `--source` or `--record` restricts a run, even
    one that names every source of the tier (the ruling names the options). Its
    summary and report say "restricted - not a tier result"; it prints no `oracle
    layer [tier]` line, no ruleset as gated or `NOT gated`, no smoke-test or
    "this tier requires" line, and its proof channel is `DISABLED - restricted - not
    a tier result`. It reports every finding - of the oracle layer and against the
    engine, a decline from the baseline included - and exits 1 on any, 0 on none.
    Not made, as checks of a tier as a whole: `engine-nothing-gated`,
    `engine-read-nothing`, parse mode's `engine-nothing-compared`, a tier's
    `require_engine`, and the release tier's proof-channel readiness
    (`proof-channel-vacuous`, which `verify` still prints; review fix 2026-10-22).
    Its report is `gate-<tier>-restricted.json` (`restricted`, `restricted_to`; the
    tier verdicts `ok` and `oracle_layer_ok` null - `ok` too since the review fix,
    so that no script reads a restricted run as a tier pass - its `findings` listed,
    `engine` "compared", nothing under `gated`/`ungated`), so it never stands in for
    a complete run's. This undoes the
    reading of issue 124 (a restricted run kept its tier's rule). Proposed edit 103.
    Nothing is parked on it.

<!-- A new list, so that the numbers from 200 render as written. -->

200. **F5's biases and norm, as built** (§9.3, register F5). (a) Register F5 names
    two bias rules - the norms have none ("no mean-centring, no bias") and the Q, K
    and V projections have none - and is silent on the attention-output and FFN
    biases; §9.3 keeps them, but only in passing, in its Mish bullet (Chessformer's
    networks "also keep their FFN and attention-output biases, as here"), while a
    broad reading of "no bias ... everywhere except inside smolgen" would drop them.
    Built as §9.3 reads: the old single `linear_bias` switch is split into
    `qkv_bias=False` and `out_ffn_bias=True`; the heads and the embedding's Δt MLP
    keep their biases. (b) "RMSNorm-style" implies a learned gain, and eps is not
    stated: 1e-5, kept from N1, inside the root as in PyTorch's `rms_norm`.
    Proposed edit 150.
201. **Ablation 6y's post-norm arm is unpinned beyond "Lc0's post-norm with DeepNet
    initialisation"** (§9.3, §13.1 6y, register F5). Built (`norm_placement="post"`)
    as DeepNorm from the paper (Wang et al. 2022, "DeepNet: Scaling Transformers to
    1,000 Layers", Figure 2) - Lc0's code is not consulted (AGENTS.md):
    x ← N(αx + Attn(x)), x ← N(αx + FFN(x)), α = (2L)^(1/4); DeepNet's
    initialisation as the paper's pseudocode draws it - Xavier normal, std
    gain·√(2/(fan_in + fan_out)), at gain β = (8L)^(-1/4) on the value,
    attention-output and both FFN projections and gain 1 on Q and K - and pre-norm's
    GPT-2 1/√(2L) dropped. Placeholders: the paper's constants rather than Lc0's;
    Xavier normal as the initialisation's base, the pseudocode's example ("it can be
    replaced with other standard initialization") - the rest of the network is LeCun
    normal, and the arm was first built on that base (N-F5F6's review fixes changed
    it), which starts the FFN's 256→2048 and 2048→256 matrices at 2.12x and 0.75x the
    pseudocode's std (the FFN branch 1.59x; the square attention projections are the
    same on either); no final norm (the last FFN norm is the output's); no norm on the
    embedding before block 0; board-pair smolgen reads the attention's input, which
    is then the block's input. Tested through the dense trunk and the cached path
    (cached = full pass, transpositions, root refresh, the BF16 cache). Proposed edit
    151.
202. **Smolgen "keeps its own normalisation"** (register B12: "(F5)"; F5:
    "everywhere except inside smolgen"): both presuppose a norm inside smolgen, as
    Lc0's has; the built smolgen (B12, either form) has none, and no document says to
    add one. Left as built - there is nothing to keep - and recorded. Proposed edit
    152.
203. **F1's parameter count** is now 21,338,634: 20,736 fewer than issue 113's
    21,359,370 - the norms' biases (33 × 256) and the Q/K/V projections' biases (16 ×
    3 × 256). F2's L24-ff12 preset is 44,456,458 (§9.2: 44.0M); the post-norm arm has
    256 fewer (no final norm). §9.2 already marks 21,359,370 as before CHECKPOINT 6's
    normalisation; B15's table will add ~1.6M. Proposed edit 153.
204. **Whether F6's Mish reaches the output heads** (register F6, §9.3, D3). F6 binds
    the "FFN activation"; its parenthetical "(Lc0 from T79; Chessformer's FFN and
    output heads)" and §9.3's "Chessformer uses it in the feed-forward sublayer and
    the output heads" describe Chessformer, and neither says whether this network's
    heads follow. The value head's hidden layer is the only head MLP (the auxiliary
    heads are linear): built as `head_activation`, placeholder "gelu" (as built at
    N1), "mish" available; CHECKPOINT 7's D3 head rewrite is where the heads change.
    The input embedding's Δt MLP keeps GELU (neither the FFN nor a head: an
    implementation choice). Squared ReLU, 6y's arm, is built as `relu(x)^2` (So et
    al. 2021, "Primer") at Mish's initialisation - the arm's initialisation is not
    stated; its BF16 gap is about twice the v1's (it squares its input's rounding),
    and the precision tests' bounds for it are doubled. `head_activation` is a
    committed-class placeholder that awaits neither a ruling nor a named ablation
    (6y's arms are the FFN's, not the heads'; M20's dispositions predate it): put to
    the user to rule, or to name the ablation. Proposed edit 154.
    *Ruled at CHECKPOINT 8: Mish in every head's hidden layers, the value head included; squared ReLU a trunk-only arm (F6) - the network lane.*
    *Built (`lane/net`, N-F6heads, 2026-10-17): the value head's hidden layer - the only head
    MLP - is Mish with no switch (`head_activation` removed rather than kept as an unruled
    arm); squared ReLU stays `ffn_activation`'s, and a test holds every head's output under
    the arm to the v1's, bit for bit, over one trunk output. The Δt MLP is issue 221. Still
    open: the squared-ReLU arm's initialisation (drawn at Mish's; proposed edit 154).*
205. **B15's far bins** (§4.1, register B15: "binned and clamped beyond"). §4.1 gives the
    exact classes and says the classes beyond them are "binned and clamped, so unseen
    board counts extrapolate as the binned biases did", with no bins - not their edges,
    not their signs, not whether each axis is binned or one distance. ~1.6M parameters
    (about 12,500 entries per head per layer) leaves room for 9-10 far slices beyond the
    45 exact ones; a per-axis product - each axis exact inside the box, two signed bins
    beyond - would be 72, ~3.4M. Built (placeholder, `disp_far_edges = (3, 5, 9)`): the
    exact box's own distance D = max(|Δtimeline|, ⌈|Δtime|/2⌉) - in timelines and whole
    turns; the exact classes are D ≤ 2 - log2-binned [3, 4], [5, 8], [9, ∞), the last
    clamped, and crossed with the sign of Δtime (past, same time, future; B6 kept Δtime
    signed and Δtimeline unsigned): 9 far classes, 54 in all - with the line and register
    entries 12,681 per head per layer, 1,623,168 parameters. Δtimeline's sign is kept
    inside the exact box (exact) and dropped in the far bins. Only the edges are a
    setting: the form - one distance D, crossed with Δtime's sign, Δtimeline's sign
    dropped - is fixed in code (`board_class` and the CPU paths' tiles), so a per-axis or
    timeline-signed form is a code change (the geometry, both implementations, the tests'
    reference, the pinned counts), not a switch. **Under this form B6 is not a special
    case of B15**, which §4.1 ("of which it is the special case that ignores file and
    rank"), register B6 ("B6 is its special case") and reversal 25 ("B6 its special
    case") say it is. B6 was two per-axis tables added together - signed Δtime buckets
    exact below 4 half-turns, then 4-7, 8-15, ..., 64-127, 128+ (19), and |Δtimeline|
    exact below 4, then 4-7, 8-15, 16-31, 32+ (8) - so it is a special case only if each
    far class lies within one of its cells, and the joint distance mixes them: Δ(timeline,
    half-turns) (0, 20) and (0, 100) share class 53 (B6's time buckets 6 and 8), (10, 0)
    and (30, 0) class 50 (timeline buckets 5 and 6), (3, 20) and (9, 20) class 53
    (timeline buckets 3 and 5). Beyond the box the far classes hold a B6 bias f(Δtime) +
    g(Δtimeline) only where it is a constant plus a function of Δtime's sign (a linear
    solve over |Δtimeline| ≤ 300, |Δtime| ≤ 600 half-turns), and the board-axis line clamp
    merges B6's buckets too (k ≥ 8 shares one entry). Lost against B6: its per-axis form
    beyond the box, and its resolution there - Δtime to 128 half-turns and |Δtimeline| to
    32, where B15 has three joint bins and clamps from D = 9 (17 half-turns or 9
    timelines). Kept: Δtime's sign and the clamped extrapolation guard. Far classes that
    refine B6's cells - the 131 of its 152 that reach beyond the box, 176 classes in all,
    with the line clamp raised to B6's 64 turns - would be 40,945 entries per head per
    layer, 5,240,960 parameters, over three times §4.1's ~1.6M: with a slice per class,
    "special case" and "~1.6M" cannot both hold. (B6's two tables added beyond the box, 27
    scalars per head per layer - 3,456 parameters - would keep both, at the cost of "one
    learned scalar ... indexed by the pair's displacement".) B6 was removed with no arm
    (M20), so ablation 2 cannot test what B6 resolved that B15 does not. Put to the user:
    keep the sentence (the far classes change, or B6's tables are added - edit 155) or the
    budget (the sentence goes - edit 159). Proposed edits 155, 159.
206. **B15's line classes** (§4.1: "exact line classes for displacements beyond them:
    movement along one axis only, or equal steps along two, three or four axes"). Built:
    (a) in whole turns - the rulebook's displacement has dT in whole turns and a slider's
    path runs over boards of the mover's colour (J7), so an odd Δtime is never a line,
    though B15's index counts half-turns; (b) "exact" read as one entry per direction and
    step k: the 64 directions that also move on the board, k = 3..7 (a square line stops
    at 7), and the 8 along the board axes only, where k is unbounded - clamped at
    `disp_line_max = 8` (placeholder: every longer line shares k = 8's entry); 368 entries;
    (c) a partition: a line's entry replaces its far slice's entry for the pair (one scalar
    per pair, as "one learned scalar per head per layer" and B17's "the full table's
    ~12,000 classes" read) rather than adding to it. So three entries per head per layer
    are never indexed: the (0, 0) cell of the three same-time far slices, whose
    displacements (the same square, timelines apart) are all lines. Only (b)'s clamp is a
    setting (`disp_line_max`); (a) and (c) are fixed in code - counting half-turns would
    change the line set, and the additive reading would put two entries on a line pair, a
    second lookup on every path - so either is a code change (the geometry, both
    implementations, the tests' reference, the pinned counts), not a switch. Proposed
    edit 155.
207. **B15's register and global entries** (§4.1: "per-board registers by their
    board-distance class, global registers one entry per head (§5.3)"). Per-board
    registers: whether the pair's kind enters is not said. Built (placeholder,
    `disp_register_pairs = "by_kind"`): an entry per class for each of square reads
    register, register reads square and register reads register (162); `"shared"`, one
    per class (54), is the other reading. Register slots are not told apart (ablation 3's
    4-per-board arm keeps the layout), and a register's (file, rank) never enters (B8's
    ruling). Global registers: "one entry per head" read per layer (the table is) and for
    either side of a pair. As built it moves only board queries' attention to the global
    registers: every key of a global query's row gets the same entry, which the softmax
    ignores - so the value head, pooling the global registers, cannot see the last layer's
    table (tested). A per-direction global entry (global reads board, board reads global)
    would reach the global rows. Proposed edit 155.
208. **B15's initialisation** (§4.1, §15.2): not stated. Built (placeholder,
    `disp_init_std = 0.1`): N(0, 0.1), the scale of the other learned bias tables (B11),
    drawn from a generator of B15's own seeded from the model's (as D8's are), so the
    no-table arm keeps every other weight and §15.2's "never zeros" holds. A trained
    network's starting table (zeros, say, as relative-bias tables often start) is a
    training choice the documents leave open. Proposed edit 155.
209. **A FlexAttention block mask in board-major order has no empty block on the masked
    heads under v1's register rule** (§6.1, register B14; B9's v1 values, issue 16b). §6.1:
    "FlexAttention exploits block sparsity losslessly". Under `masked_head_registers =
    "read"` (v1, M20) a register query on a masked head reads every key, and every board's
    tokens end with its registers, so every board-aligned block of query rows holds rows
    that read everything: B14's block mask (`flex.board_block_mask`, lossless by
    construction) computes every block of the move-graph heads; under `"self"` it skips
    67% of them on a 14-board position with board-aligned blocks of 66 tokens, and 22% with
    FlexAttention's default 128-token blocks, which straddle boards (counted once; the
    tests check that blocks are skipped, and that under "read" none is). The CPU
    block-sparse path computes those rows apart (`free_rows`); a GPU kernel must too - a
    second call over the register rows, or the registers ordered after the squares among
    the queries - and its tile size meets the 66-token board. A cost, not correctness; the
    kernel is G9's and M19's. Proposed edit 157.
210. **F1's and F2's counts with B15** (§9.2, register F1, F2; §4.1): the v1 is 22,961,802
    parameters - issue 203's 21,338,634 and B15's 1,623,168 (7.07%: §4.1's "~7% of the
    network" holds); F2's L24-ff12 is 46,891,210 (44,456,458 and 2,434,752 over 24
    layers). §9.2's "~21.5M with all heads" and "44.0M", and register F1's 21.5M, are the
    network without the table: `test_params.py` pins the totals and holds the no-table
    counts to them. Proposed edit 156 (supersedes 153). *After M20's dispositions
    (`lane/net`, N-M20): 22,965,139 (21,341,971 without the table; B15's share 7.07%) and
    F2 46,894,547 (44,459,795).*
211. **What stands in for "the score function" in the untrained build** (§15.2's B15 row,
    §10.2, register B14, G9, M19). §10.2: every bias "is a FlexAttention score function
    computed inside the kernel"; §15.2: "Displacement bias through the score function =
    the same bias materialised, on small positions". Here (PyTorch 2.13 on CPU)
    FlexAttention runs only uncompiled - it materialises every score, evaluates the score
    and mask functions pair by pair under vmap, and has no backward. Built: B14's score
    function for that path (`attn_impl = "flex"`, forward only; it refuses under autograd
    on CPU), and the row tested as written on twelve constructions (every variant's board
    size among them, Global Warming's 1 × 1 included) - the score function's
    term at every pair is the hand-materialised bias bit for bit, and attention through it
    is attention with the bias materialised (FP32 roundoff); gradients through the CPU
    block paths, which build the same rows by lookup. The eager path ignores the block
    mask's sparsity, so its losslessness is tested apart. Compiled kernels: GPU time (M19).
    Proposed edit 158.
212. **B12's relative cells outlive B6** (register B6, B12; §4.1, §6). B12's cells are B6's
    log buckets (B12: "as built over relative board-pair cells"); with B6 removed
    (superseded, M20) nothing else defines them. Kept as built - smolgen's 152 cells and
    47,136 parameters, bit for bit - in `buckets.py`, the fields renamed `smolgen_time_*`
    and `smolgen_timeline_*`. Re-basing the cells on B15's 54 board-distance classes (54 x
    8 x 32 + 8,224 = 22,048 parameters) is the other reading; no document asks for it.
    Proposed edit 159.
213. **B16's bins, its status and its switch** (§4.1: "per head, by the key board's
    Δt-to-present bin and timeline status"; register B16, class C). (a) No bins are given.
    Built (placeholder, `status_dt_exact = 4`, `status_dt_max = 128`): signed log2 buckets of
    Δt-to-present in half-turns - exact below 4, one per doubling up to 128, clamped beyond,
    past and future apart: 19 bins, the time buckets B6 used and smolgen keeps
    (`buckets.signed_log_bucket`). (b) "Timeline status" is read as A3's active/inactive flag,
    the one status of a frozen board's timeline that changes after it freezes (head/superseded
    is constant there: every keyed board is superseded) - 19 x 2 = 38 entries per head. (c)
    Whether B16 has an off position: its gate, 6c, names the refresh's arms, none of B16's.
    Built as `status_key_bias`, on; off is the network without the table (B16 draws from a
    generator of its own, so every other weight is the same) - for comparisons, no named arm.
    Proposed edit 160.
214. **B16 per layer or shared by the layers** (§4.1, register B16: "per head"). Neither
    says; B11 is shared across layers, B15 per layer. Built (placeholder,
    `status_bias_per_layer = False`): one table for every layer, as B11's - 8 x 38 = 304
    parameters (4,864 with one table per layer, 16 x 304). Proposed edit 160.
215. **B16's initialisation** (§4.1, §15.2): not stated. Built (placeholder,
    `status_bias_init_std = 0.1`): N(0, 0.1), the scale of B11's and B15's tables, from a
    generator of B16's own seeded from the model's (`model.b16_seed`), so switching it changes
    no other weight and §15.2's "never zeros" holds. A trained network's starting table is a
    training choice the documents leave open. Proposed edit 160.
216. **§4.1's "frozen boards therefore carry only intrinsic features" against the refresh as
    built; B16's rows; the anchor's own evaluation** (§2.3, §4.1, §7.3, §7.9; register A3, B16,
    C3, M20). (a) §4.1 says it of the whole cached configuration, and §2.3 that "frozen tokens
    keep only intrinsic features" - read literally, a refresh's history boards too. But M20
    ruled issue 31's option, "intrinsic (placeholder: frozen boards carry neither; a refresh
    gives them the anchor's values)", and the refresh - a dense pass - stays as built (the main
    loop's reading, "Current state"): a refreshed epoch's history boards carry the anchor's A3
    features in their K/V. Built so; B16 biases them at every node below the anchor, with the
    node's own status, like every other frozen key (the main loop's reading). (b) Which rows:
    §4.1 says "key biases", §7.3 "current status reaches it at score time", and neither names the
    query rows. Built (forced): live rows only - the head boards' tokens and the global registers
    - against frozen boards' tokens, squares and per-board registers; on a frozen row the bias
    would make the frozen board's outputs and K/V depend on the present, which C17 allows only
    at refreshes (§7.3's "reaches it" read as reaching the live rows that read it). (c) The
    anchor's own evaluation in its refreshed epoch: §7.9 - "The refresh pass itself is the
    ordinary dense model: at the root the two-level mask reduces to full dense attention" - so it
    adds no B16 there (`T5DNet.make_context(status_bias=False)`, in the cached path and in its
    reference); every node below the anchor carries it. Conditioning-offset training's k = 0 is
    then "fully dense", as §7.9 says, and k > 0 carries B16. Below the anchor a history board's
    status enters twice: as the anchor's values in its K/V and as the node's in B16. Built
    otherwise (B16 at the anchor too), the root's evaluation would not be the dense pass, which
    `test_cache.py` pins (C3). Proposed edit 160.
217. **B17's compact class set** (§4.1, register B17: "a compact class set (~1,400: the exact
    near classes plus the line classes)"; class C). Neither the set nor "the exact near
    classes" is defined. B15's exact classes are 45 board-distance classes (Δtimeline within
    ±2, Δtime within ±4 half-turns), each a 15 x 15 file-rank slice (10,125 cells), so "near"
    must cut either the file-rank offsets or the board-distance classes - and two readings give
    "~1,400" exactly alike: 1,493 classes with B15's line classes (368 at its defaults; they
    follow `disp_line_max`), so the same parameter count and both of §4.1's cost figures (0.51 x
    QK^T at 44 boards, 9.4% at 240; the cost depends on the class count alone). They cover
    different pairs. (i) **Built** (placeholder, `shaw_near = 2`): in every one of the 45 exact
    classes the offsets with Δfile and Δrank within ±2 (25 of 225; 1,125 classes) - knight and
    king relations on and across boards and each line's first two steps; on one board a rook,
    bishop or queen relation more than two squares away gets no term (`test_shaw.py` pins it).
    With ±1 the set is 773: 0.26 and 4.9%. (ii) **Not built**: the nearest board-distance
    classes with their full slices - say the same board and the four one timeline or one turn
    away (5 x 225 = 1,125) - every relation on the board, as in the Lc0 experiment §4.1 cites
    for B17 (Shaw vectors for every square pair of its one board), and on the four nearest
    boards; none on the other 40 exact classes - two timelines or two turns away (the knight's
    two-board leaps among them) and, with these four, every odd half-turn displacement. Which
    boards, and mixes of (i) and (ii), are further choices. Under either, the far slices' cells
    and every pair with a per-board or global register (no (file, rank), B8) get no term -
    placeholders too. (ii) or a mix is another `shaw.compact_classes`, not a setting. Put to
    the user; the documents do not choose. (Until the N-B11-B16-B17 review fixes this issue said
    "near" must cut the file-rank offset, which does not follow.) Proposed edit 161.
218. **B17's parameter count** (§4.1, register B17, F1; §9.2): no document states it. A vector
    per class per head per layer - each head's query dotted with its own vectors, as B15's
    table is per head per layer - is 1,493 x 32 x 8 x 16 = 6,115,328 parameters: 26.6% of the
    v1's 22,961,802 (21.0% of the network with it; 22,965,139 after M20, the same shares),
    almost four times B15's ~1.6M, which §4.1
    calls ~7% of the network; F2's preset would carry 9,172,992. Shaw et al. share theirs across
    heads: 1,493 x 32 x 16 = 764,416 (3.3%). Built (placeholder, `shaw_shared_heads = False`):
    per head; shared is the option. §15.3's cost does not tell them apart (the products are per
    head either way). Proposed edit 161.
219. **B17's query and scale** (§4.1, §14.1): neither is stated. Built: the unrotated query -
    forced: the class is a function of the pair's displacement, so q . R[c] is relative only for
    a query that carries no position; a rotated query would make it depend on the query's
    absolute angle (an absolute positional encoding, §14.1) and on the frame a computation
    centres its coordinates on, so the cached path's per-board frames could not equal the full
    pass (shown in `test_shaw.py`). The scale: 1/sqrt(d_head), Shaw et al.'s and the scores' own
    (placeholder). Proposed edit 161.
220. **B17's initialisation** (§4.1, §15.2): not stated. Built (placeholder, `shaw_init_std =
    0.1`): N(0, 0.1) from a generator of B17's own (`model.b17_seed`) - a query's entries have
    unit spread at initialisation, so the term starts with the spread of B15's scalars; the
    flag's off position is the B15 network bit for bit. Proposed edit 161.
221. **The network's other activations are outside F6's words** (§9.3, register F6, §2.2,
    §2.3, B12). F6 now names two places - "in the trunk FFN and every head's hidden layers"
    (CHECKPOINT 8) - and the network has activations in neither: the input embedding's Δt MLP
    (A3: `Linear(3, 32)`, GELU, `Linear(32, d)` - §2.3 says only "embedded per board"), and
    B12's smolgen context layer (`z = silu(W_z c + b_z)`, SiLU as in Lc0's smolgen; a
    prototype behind its flag). Two more are to come with their components: A6's turn-state
    inputs, "embedded into the global registers" in no stated form (an MLP would need one),
    and A8's "FFN after the embedding" (§2.2) - an FFN, but not the trunk's. All are inputs
    or input-side generators, committed-class like the rest of the input encoding (a
    network trained with one cannot run with another). Built: kept as they are - GELU in the
    Δt MLP (stage N1's), SiLU in smolgen - documented as outside F6 (`embed.py`,
    `smolgen.py`). For the user: Mish there too, or leave them implementation choices; A6's
    and A8's components will need the answer. Proposed edit 162.
222. **D1's promotion offset: its form, and which pointer heads carry it** (§8.1, register
    D1, D4, D9, M20; §12.4). §8.1: "Promotions add a learned per-piece logit offset ...
    (Chessformer adds a promotion bias the same way)"; M20: "promotion scoring -> learned
    per-piece offset (D1)". Neither says (a) whether the offset is a constant per piece or
    read from the position, nor (b) whether it belongs to every pointer head or to §8.1's
    policy alone - the opponent-reply prototype (D4), the soft policy ("its own source and
    destination projections and its own submit logit", §12.4) and D9's optimistic policy
    score promotions with the same bilinear form, and tie them the same way. Built (N-M20):
    (a) as N1's arm built it - one logit per piece type read from the destination token's
    final state, `Linear(256, 13)` (3,341 parameters; its bias is the constant part), added
    to the move's bilinear score; (b) the policy only: the reply head stays as built at N1,
    without one - its rebuild as D4's prototype is the heads' component (after engine round
    2), which builds the soft policy and D9 too. For the user with the heads: whether those
    heads carry the offset. Proposed edit 163.
223. **Which heads "aux pooling -> squares" covers** (M20, register B8, D3, D4, D7; §8.3,
    §9.2, §12.4). M20 rules "aux pooling -> squares" and "value pooling -> mean (D3)"; D4
    lists WDL, the short-term value, the value error and the value distribution among the
    auxiliary *targets*; §8.3 and D3 pool all four value heads "by mean from the global
    registers", and §9.2 puts them in its "Value heads" row, apart from the "Aux heads" row
    ("Mean-pool board squares, per-board and global"). As built at N1, WDL and the short
    value were auxiliary heads on the placeholder `aux_global_pool` (the squares by
    default). Read (N-M20): D3 decides where the value-type heads read - WDL and the
    short-term value join the 51-bin head on the global registers' mean (the error head
    will, with the heads' rebuild) - and M20's aux rule covers the other global auxiliary
    heads, moves-left and search effort, which read every square's mean; per-board heads
    read their board's squares, per-square heads every token. D7 holds: pooling all
    sixteen registers assigns none a role (D8's argument). `aux_global_pool` removed.
    Proposed edit 164.
224. **Moves-left "in turns" at a decision inside a turn** (§12.4, register D4, M20; issue
    305). The unit is a turn - one player's action, issue 305's reading - and the head now
    predicts log1p of the turns left (N-M20; the helper and the losses' docs say so). Not
    stated: at a decision partway through an action, whether the action in progress counts
    (it has not been submitted), and so whether a turn start and a mid-turn state of the
    same action share the target. The record-to-target builder is the training loop's (M21,
    I15's records) and fixes it there; the network does not depend on it. Proposed edit 165.
225. **Where the global inputs go in a network with no global registers** (§2.3, §13.1
    ablation 3, register D8, A6, B8, D3, M20). D8 commits "a global acting-player input"
    and M20 puts it on the global registers; A6's turn-state counts are "embedded into the
    global registers" too, and D3's value heads and D8's submit logit pool them. Ablation 3
    varies "registers on/off, 2 vs 4 per board", and no document says whether "off"
    removes the global registers or only the per-board ones, nor - if it does - where those
    inputs enter and what those heads read. Dropping the acting player with the registers
    is no reading of D8: the network then cannot tell an action whose present has shifted
    from the opponent's turn start on the same boards (§2.3's own case), so its values,
    which are the acting player's (D8), could not differ in sign. Built (placeholder, fixed
    in code; N-F6heads-M20 review fixes): with `global_regs = 0` the acting player's
    embedding is added to every board token - squares and per-board registers - and under
    the block-causal scope to the live (head) boards' only, so no frozen board's input
    depends on it; the value-type heads and the submit logit pool the squares (as built
    since N1). Tested: a flipped acting player moves every output under every register arm
    the configuration accepts, on synthetic states and real engine positions, and the cached
    path still equals the full pass. A6's counts meet the same question when they are built.
    For the user with ablation 3: whether its "off" arm removes the global registers, and if
    so where their inputs go. Proposed edit 166.

<!-- A new list, so that the numbers from 140 render as written. -->

140. **M22's "optional moves played" is undefined** (§8.2 "Measure first", §15.1's
    Phase 1 row, register M22 and part K). Measured as the moves made on a board that
    was not mandatory at the action's start (T6, R9) - on either board a move is made
    on (T3: its source and, for a jump that does not branch, its destination, J4). So
    a jump from an optional board onto a mandatory one counts (it also satisfies the
    mandatory board, T6), and so does a jump from a mandatory board onto an optional
    one; both kinds are counted apart. The first version went by the source board
    alone, which counted the first kind and left out the second (35 turns of the
    27,097: 24 / 10 / 1 by range) - an asymmetry the review found; that reading is
    kept beside the other (`optional_by_source`). A board's status can change
    mid-action: a rewind (T8, T12) releases the mandatory boards left unplayed, and a
    reactivation (T11, T12) can make a board mandatory, with or without a rewind.
    Such boards keep their status at the start; the moves on them are counted apart,
    and the reading by the boards' status when the move was made is computed beside
    it. On t0database it differs on 4 of 27,097 actions. A fourth reading, the moves
    beyond the obligations' minimum, is not computed. The documents are silent on all
    of it. "Mandatory boards per turn" is read as the count at the action's start
    (`GameState::mandatory_count()`), not the mandatory boards played; the two differ
    only where a rewind releases some, which is counted (and the moves from mandatory
    boards are fewer still where one jump plays two of them, also counted). Proposed
    edit 111.
141. **M22 names no board-count ranges and no unit** (§8.2, §15.1, register M22).
    Built with 6b's and 6b″'s ranges (< 50, 50-99, 100-199, 200+) by the board count
    at the action's start (every board, history included - M7's N), with ten-board
    buckets under 100; M7 used 1-49 / 50-99 / 100-159 / 160-239 / 240+, and the
    per-action lines allow any. The unit is the action (turn) of a distinct game: a
    record that equals or is a prefix of another is counted once (692 games, M7's
    strict count); M7's grouping (674 games) is a sensitivity reading, and the refused
    records' actions are reported apart. A range resting on fewer than 10 games is
    flagged too few to read (issue 118's convention, applied to games). Proposed
    edit 111.
142. **Nothing maps M22's figures to an order of the D2 work** (register part K:
    "Priority of the D2 arms (M22)"; §15.1: "Cheap measurements before the code they
    gate"). §15.6 sets no threshold, and both turn-start forms are Phase-2
    deliverables whatever the figure (§15.1's Phase 2 row; §15.5 needs both D2 rows
    of §15.2), so the measurement can order the work but not cancel an arm. Recorded
    without a verdict (MEASUREMENTS, "Turn width (M22)"); the order is put to the
    user. The figures, read by the turn's start, which is where the arms differ - not
    by the moves the player made: a turn that starts with one playable board is one
    move and a submission under every arm, and every other start is searched
    differently (the ply split evaluates every node inside the turn and, after the
    shift, offers submit beside each optional move; the tree form inherits the
    turn-start priors; the flat form gives optional boards a "leave it" option).
    Turns that start with more than one playable board: 15.1% of the human games'
    turns under 50 boards, 65.2% at 50-99 and 99.4% at 100-199 (13 games); with more
    than one mandatory board - §8.2's "k evaluations along every path" - 5.8%, 27.1%
    and 64.0%; with three or more - part L's regime for the turn-start forms - 0.7%,
    6.7% and 16.0%; with an optional board 10.5%, 44.5% and 65.7%, an optional move
    played in 1.2%, 2.5% and 1.1%. The turns of more than one move (4.1%, 19.7%,
    50.9%) understate it: 414, 220 and 23 turns start with two or more mandatory
    boards and are played with one move (a jump between two of them, or a rewinding
    branch). And 6x is "reported by mandatory-board count" (§13.1): its one-board
    class would hold both the turns where every arm is identical and those with
    optional boards beside the one (9.3%, 38.1% and 35.4% of all turns), where the
    arms differ - a report by playable boards as well would separate them. No human
    turn starts at 200 boards or more, so the regime of §8.2's cost argument is
    unmeasured until the pilot self-play run (M18). (The first version of this issue
    called the multi-move turns "where the arms differ"; corrected at review.)
    *Ruled at CHECKPOINT 8: the tree form first, then the flat form, both built (D2, M22); M22 also reports per range the mandatory shares (more than one, three or more), the optional moves actually played and the compute share of multi-board turns - the engine lane.*
143. **A6's arithmetic for the mandatory count is not the rules' count** (§2.3; register
    A6: "mandatory boards still to move — exact from the rules engine: −1 per ordinary
    move, −2 for a move between two present boards, 0 after a rewinding branch"). Checked
    against a recount from the rulebook at every state of random play under every ruleset
    (`engine/tests/test_turn_state.cpp`): the three cases hold wherever no timeline starts
    or stops counting, but the arithmetic leaves out (a) T6's arrivals - a jump that does
    not branch plays the board it lands on, so one from an optional board onto a mandatory
    one is −1 (the optional count −1 too), and one from a mandatory board onto an optional
    one is −1, neither "a move between two present boards"; (b) T11/T12 - a creation that
    reactivates a timeline behind the present rewinds it (Rewinds, the default), and the
    count rises when that board is the mover's (10 times in the default test's 3,842
    states, 752 in the long test's 77,926); (c) Q5's AllActive reading, under which a
    rewinding branch zeroes nothing; (d) Q3's play activation and Q6's deferrals, which
    change what counts. §2.3's "exact counts, never predicted ones: the rules engine knows
    them" governs: the count built is the state machine's own
    (`GameState::mandatory_count`), never the arithmetic. Proposed edit 116.
144. **"Optional boards still playable" and "moves already made this turn" are not
    defined** (§2.3; register A6). Read, from the rulebook's terms: optional = the acting
    player's playable boards (L5: the heads with the mover to move - active or not, in the
    present or not, T6) that are not mandatory, and none once no move may be made (Q4's
    Guide-literal reading, after the shift); a playable board with no legal move counts
    (L5 asks for none; the expansion's per-board `playable` flag marks the same set, so
    mandatory + optional is the number of boards it marks), and so do inactive and
    deferred (Q6) heads. Moves made = the moves made since the action's turn start (a turn
    being one player's action, issue 305), one per move - a jump between two boards is one
    move, though A6's "−2" counts the two boards it plays; 0 at a turn start. The
    alternatives - only boards with a legal move, only active ones, boards played instead
    of moves - are not built; each would be a few lines in `expand.cpp`. Proposed edit 117.
145. **§15.4's turn-structure invariant does not name the count** (§15.4: "a submit edge
    leads to a turn start and a move edge to a mid-turn state"; register A6, C11). With
    A6's count in a node's identity (issue 302) the invariant is the count's: 0 below a
    submit edge or a whole-turn action, the parent's + 1 below a move - in a graph as in a
    tree, every parent of a node agreeing. Built into `Search::check_invariants` and
    checked from outside the search (`mcts_every_node_counts_the_moves_made_since_its_turn_start`).
    Proposed edit 118.
146. **§11.3 calls Lazy SMP's races benign and says nothing of the stores its threads share**
    (§11.3 "Proofs never live only in the TT", "Determinism debt, extended"; register H17,
    H19). The TT's races are benign because it holds estimates: a torn or lost entry costs
    an ordering hint or a re-search. The solved store and the disproof store hold facts and
    are shared by the same threads, and a race there is not benign: as found (H17's row),
    a racing solved store held one position twice and lost a shorter distance - and a
    position held twice can hold two kinds of proof, the contradiction the store exists to
    refuse. The spec states no concurrency contract for them. As built: every call on a
    shard is serialised by an exclusive lock (`ShardLock`, a `std::mutex`), and the stores
    are tested fresh, written by 8 threads at once from their first use. Proposed edit 120.
147. **6b″'s figures were measured with the stores' racing, slow lock** (§11.4's 6b″
    result, §15.6; register M17, H42, H43). 6b″ ran Guard and Hunt at 5 threads, 50 ms per
    root, on the development machine, whose toolchain's `std::shared_mutex` both raced
    (H17's row) and cost 1.4-1.8 µs a call behind a process-wide spinlock. With the fixed
    lock, indicatively (a shared machine, test positions), Guard runs 2.7-2.9× and Hunt
    1.45-1.7× as many nodes per second at 5 threads. The verdicts' soundness stands - every refutation
    and mate counted was confirmed against E4 alone, and no verdict logic has changed - but
    the rates (Guard 65.1 / 15.8 / 5.2 / 0.0%, Hunt 69.0 / 63.6 / 35.0 / 0.0%) may understate
    what 50 ms reaches. Guard's bar (95%) is far from every range's rate; Hunt's at 100-199
    boards (50%; 7 of 20 truth mates found, 10 needed) is nearer. Whether to re-run 6b″ on a
    quiet machine (about a minute) is the main loop's call; no edit is proposed until a
    re-run says something.
148. **The search's draw and moves-left: backed up, or recorded** (§8.3, §12.8; register
    D3, D4, I13, I15; the plan critic's missing item 3). D3 makes the WDL "the value
    search uses", D3's `q_ratio` blends the WDL target towards "search value", I13 has
    earlier positions "take the best move's value and moves-left instead of the result",
    and I15 records "root and best/played children's values" - in Lc0's sense (q, d, m).
    No document says whether the search backs up a draw probability or a moves-left
    count, or what stands in for them; before engine ABI round 2 it backed up a scalar
    only. Done (engine ABI round 2a, an engineering reading, no committed-class choice):
    the **draw probability D is backed up** beside the value on every node and edge, as
    Lc0 backs up D - a leaf's D its WDL's (normalised), a mate's 0, a stalemate's and the
    turn cap's 1 (I11 scores it a draw), a solved node's 1 for a proven draw and 0 for a
    proven result; an MCGS residual corrects D as it corrects Q. D does not flip across
    a change of acting player. A root child's search WDL is so `((1 - d + q) / 2, d,
    (1 - d - q) / 2)` (`RootChild::d`, `t5d_root_child.d`). **Moves-left is not backed
    up**: what one edge adds to a moves-left count needs the unit of a turn (issue 305's
    reading: one player's action) and whether the action in progress counts (issue 224,
    open), so the search records each node's network moves-left as given and exposes,
    per root child, the child's own evaluation - its network WDL, moves-left and error
    output, in the root player's view (`RootChild::evaluated`, `network_wdl`,
    `network_moves_left`, `network_error`), and the root's own (`t5d_search_summary`'s
    `root_evaluated`, `root_wdl`, `root_moves_left`, `root_error`). I13's best-move value
    is the best child's backed-up (q, d), its moves-left the child's network output plus
    the offset the target builder (M21) adds. If moves-left is ever to be backed up
    (Lc0's M, e.g. for MLH's steering, issue 300), it is a later change to the backup
    once issues 224 and 305 are settled. Proposed edit 121.
149. **6m's switch: what the uncertainty signal is, and who reads it** (§8.3, §8.4, §13.1
    6m; register D3, E2, E3, E5, I10). I10: "6m decides which uncertainty signal search
    consumes"; §13.1 6m: "the categorical variance vs the error head - both trained from
    the start (§8.3) - as the signal search consumes". Neither names the consumers, the
    default, or how the error head's output becomes a variance. Done (engine ABI round
    2a; placeholders): `SearchParams::uncertainty_source` (`t5d_search_params.
    uncertainty_source`) - `Categorical` (the categorical's variance; the default, by
    D3's own words "its variance and overlap feed search, commit and time") or
    `ErrorHead` (the error head's output taken as the variance: it predicts a squared
    gap, a variance by construction) - is each node's variance, and so what all three of
    its readers take: E2's cPUCT scale and backup weight, E3's bonus and E5's optional
    LCB term. A signal not given (no categorical delivered; the error NaN) is 0 - no
    uncertainty. E2's and E3's constants (`variance_reference` 0.4, the prior weight, the
    clamps; issue 81) were sized against a categorical over outcomes; the categorical is
    now over root search values (reversal 28) or replaced by the error head, so they are
    placeholders twice over - SPSA (G7) with 6m. Proposed edit 122.
150. **The noise's "legal moves", and "root" under the ply split** (§8.4: "its total
    concentration split across the legal moves"; §12.8; register E1, I19). Under budgeted
    legality the root's moves include unknown ones; under the whole-turn split the root's
    children are the listed actions (at most `max_actions`); a legal submit is an edge
    beside the moves (D8). And under the ply split every decision is a fresh search (C14),
    so a k-board action meets k roots. The documents say neither. Done (engine ABI round
    2b; placeholders): the total splits across the root's **open edges** - the moves the
    expansion kept (legal or unknown), a legal submit, or the listed actions; a move later
    verified illegal is pruned with its noise, its siblings renormalised as ever. The noise
    and the root temperature are drawn **per decision** (each decision's search is a
    fresh root), as the noise was before. Proposed edit 123.
151. **KataGo's shaping, in the spec's paraphrase** (§8.4: "half the concentration on moves
    the prior ranks above most others, aimed at blind spots - a prototype (KataGo has not
    validated it)"; register E1, I19). "Ranks above most others" is not pinned: no weight,
    no reference prior, no cap. Done (a prototype behind `dirichlet_shaping`, off; every
    constant a placeholder): the shaped share (`dirichlet_shaping_share`, 0.5 - the
    paraphrase's "half") is spread in proportion to `max(0, ln min(P_i, cap) - mean_j ln
    min(P_j, cap))` - how far an edge's log-prior lies above the open edges' mean - with
    `cap` (`dirichlet_shaping_cap`) 0.01, so that the moves the prior already favours
    strongly gain no more than those at the cap (the "blind spots" reading); the rest is
    uniform, and the whole uniform where no edge lies above the mean. P is the root's
    prior before the noise (after the temperatures). Written from the spec's words and
    KataGo's published description of the idea, no code (H29). For 6g.
152. **The root policy temperature** (§8.4: "KataGo uses 1.25 decaying to 1.1, Lc0's
    training runs 1.2-1.45 (§12.8)"; register I19: "root policy temperature (KataGo 1.25 ->
    1.1; Lc0 1.2-1.45)"; §12.3's match play: "temperature zero"). (a) Lc0's 1.2-1.45 is a
    policy softmax temperature at every node, not a root one; the engine keeps both: the
    every-node `policy_temperature` (1, the arm the Lc0 figures describe) and
    `root_policy_temperature`, multiplied at the root. (b) No 5D decay schedule is given
    (KataGo's decays with the game, scaled to board size; the unit of a turn is issue 305's
    reading): `root_policy_temperature` defaults to 1 (off), and its self-play value and
    decay belong to the training generator (M21), which sets it per search beside the
    move-temperature schedule; the engine exposes the parameter and nothing more. (c) k0's
    entropy reads the tempered prior (the prior the root's children are ranked by). (d) It
    applies at a root's own expansion only: a root carried from an earlier search (match
    play) keeps its prior; at match play it is 1 - read as self-play only, like I19's title
    (§12.3's "temperature zero" is the move temperature). Proposed edit 123.
153. **"Visits with I2's forced and steered flags", and forced playouts** (register I15,
    I2; §12.2, §12.5, §12.8; the plan critic's contradiction 4). §12.5 adopts "KataGo's
    forced-playout pruning with √(playouts) decay at target construction"; E1 lists no
    forced playouts, and no document asks the search to force any; v3 has no
    scanner-forced expansions (rung 3 retired). "Steered" is §12.2's CHECKPOINT 4 sources:
    the asset map's boosts and Hunt's suggestions, neither wired (E10, step 2). Done
    (engine ABI round 2c): **search-side forced playouts not built**; target-side pruning
    is the training loop's (M21), and the search records what it needs - per root child
    its visits, its prior with and without the root noise, q, the variance of its
    backed-up values, E3's factor on its exploration term, the asset-map `boost` and
    `steered_visits` (0 until E10 wires them; E10 decides what counts as a steered visit)
    and `forced_visits` (0: not built) - and per root the effective cPUCT (E2's root scale
    included), so that a replica of the root's PUCT can compute KataGo's n_forced(c) =
    sqrt(k P(c) ΣN) and the pruning bound. Which prior the pruning compares with (noised
    or noise-free) and the provenance arm are the training loop's (I2, 6e). Proposed edit
    124.
154. **The effort label: what "stabilises" means** (§8.6: "visits until the decision
    stabilises"; §12.4: "Search effort to convergence - distributional over log-visit bins,
    censored top bin ... full-search moves only"; register D4). Neither says which choice
    stabilises (the most visited child, or the final rule's) or when. Done (engine ABI round
    2c; a reading): `root_best_since` - the root's visits when its most visited child last
    changed, a child taking over only with strictly more visits than the incumbent (a
    pruned incumbent found again among the rest). The censored bin ("did not converge
    within budget") and the bins are the target builder's (M21), reading `root_best_since`
    against the search's playouts. For a root carried by `advance` it counts from the
    re-root (its earlier history is not kept); self-play's fresh searches (C14) are exact.
    Proposed edit 125.
155. **A record's network input must be rebuildable, and the C API could not rebuild it**
    (register I15: "so targets and weights can be re-derived and later architectures
    retrained on the archive"; §12.8: "Data generated without these fields cannot get them
    back"; the training brief's conflict 3). A search leaf's candidate moves - the
    network's input and the policy's support - are the legal and unknown moves under the
    search's legality budget; `t5d_game_expand` offered only the pseudo-legal ones. Done
    (engine ABI round 2c): `t5d_expand_options` appends `legal_moves` and its budget
    (`legality_checks`, `legality_nodes`), honoured by `t5d_game_expand` with the leaf's
    own deterministic classification - the same moves in the same order, tested byte for
    byte; a search's `expand` options refuse it (its own parameters decide its leaves).
    The record must therefore keep the search's legality budget among its conditions
    ("search configuration"). A root child verified illegal after the network scored it
    leaves the record with the network's raw prior over the leaf's moves (the memo) and
    the search's priors renormalised over the survivors. Proposed edit 126.

## Proposed spec edits

Edits the spec or the register needs, for applying upstream. Each: where, what it
says now, what it should say, why.

**Numbering from 2026-10-17** (the main loop's three lanes, under M23's rule): a new
edit takes its number from its lane's block - the gate lane 98-109, the engine lane
110-149, the network lane 150-199, the main loop 200 and up; gaps between blocks are
expected. An edit the CHECKPOINT 6-7 documents carry is marked *Applied upstream
(CHECKPOINT 6/7)* (*in part*, with what is still needed, where they carry some of
it); one an earlier ruling carried, superseded or made moot says so (*Withdrawn*,
struck through, where nothing of it is left to apply); the others stand as written.

1. **§11.0 "Tiered CI", first sub-bullet**: says "ftxi's engine (fast, Python
   bindings) on every commit". The harness drives ftxi through a C++ driver and
   its stock `5dtools` CLI, because the Python bindings expose no per-position
   check verdict (PROVENANCE.md). Proposed: "ftxi's engine (fast; a C++ driver
   over its core library, held to its stock CLI) on every commit".
   *Applied upstream (CHECKPOINT 6/7).*
2. **§11.3 "The trap: XOR-Zobrist cancels in 5D" / H18**: the key's ingredients
   are content (coordinate-free), (timeline, time) by multiply-rotate, and the
   ruleset ID. Proposed: name every ingredient - board contents per (timeline,
   time), the ruleset ID, the player to move, the even/odd start, each timeline's
   creator, and the board size (structure variants share the rules but not the
   positions) - and say the mixed board terms combine by 64-bit addition.
   *Applied upstream (CHECKPOINT 6/7).*
3. **§11.0 "Running the gate" / H30**: proposed to define the fingerprint as the
   notation specification's full-state MD5 ("Hashing"), with white's board before
   black's within a turn (the notation is silent), beside a placement-only hash
   and board and piece counts; and to say the harness, not each implementation,
   canonicalises and hashes (H33); that the unmoved marker means "has not moved"
   (spec issue 2h); that a marker-only difference is reported per run and stops
   confirmation, not comparison; and that a source exporting its own hash (as
   5d-chess-js does) is held to the harness's fingerprint.
   *Applied upstream (CHECKPOINT 6/7).*
4. **§12.6, the ±0 note**: proposed to record the mapping the engine uses: `-0`
   → -1, `+0` → 0, `-n` → -(n+1), `+n` → n, so adjacent timelines differ by one.
   *Applied upstream (CHECKPOINT 6/7).*
5. **§9.2 "Aux heads" row**: says global aux heads pool from the global
   registers; §12.4, B8 and D7 say aux heads pool from the trunk and registers
   are never assigned aux targets. Proposed: "Mean-pool board squares (per-board
   and global)", or a decision recorded in the register (spec issue 9c).
   *Applied upstream (CHECKPOINT 6/7).*
6. **§5.3 and §4.1/§5.2**: proposed to say what a register's position is under
   RoPE - per-board registers ("must not receive a (file, rank)": a learned
   off-board coordinate, or spatially position-free pairs) and global registers
   (no board: position-free pairs, or anchored at the recentred origin) - spec
   issues 9a, 9b.
   *Applied upstream (CHECKPOINT 6/7).*
7. **§8.1**: proposed to say how promotions are scored (the bilinear form gives
   every promotion piece the same score) - spec issue 9d.
   *Applied upstream (CHECKPOINT 6/7).*
8. **§12.4**: proposed to define the opponent-reply move space, the per-square
   label set, and the moves-left unit - spec issues 9g-9i.
   *Applied upstream (CHECKPOINT 6/7): the reply's move space and moves-left in
   turns in §12.4; the per-square label set stands as a v1 placeholder (M20, 6j).*
9. **§4.1**: proposed to fix B5's per-head details and how B6's binned biases
   relate to RoPE on the time and timeline axes - spec issues 9e, 9f.
   *Applied upstream (CHECKPOINT 6/7) in part: B6's half is moot (B15 supersedes it,
   M20); B5's details are ruled v1 values (B5, M20) but §4.1 does not state them
   (near-NoPE at π/8, the same-square 6 + 10 split, the relations head's
   allocation).*
10. **§11.0 "Running the gate" (settle by rulebook) and §15.1 Phase 0**: name the
    rulebook. Proposed: `docs/rules/README.md` - the cited rules reference built
    from the in-game Guide and community sources, never from an oracle - with its
    open questions settled by the user's rulings and recorded there. **Applied
    upstream 2026-10-01** (H38; spec §11.0 "Triage by cluster").
11. **§11.0 item 1**: "plus the royal queen in Reversed Royalty" - proposed: "in
    Reversed Royalty the royal is the queen; its king is a common king, not royal"
    (spec issue 11).
    *Applied upstream (CHECKPOINT 6/7).*
12. **§6 / B9**: state the mask's direction, that a token always reads itself,
    what register queries on masked heads read, and that padding is excluded
    (spec issue 16a, b).
    *Applied upstream (CHECKPOINT 6/7).*
13. **§6 table / B9-B10**: say whether B9's "at least 2 heads" counts the
    relations head; define "attack/defend relations" in B11's types and that
    head's direction; state that "same board only" covers per-board registers as
    queries (16c-e, g).
    *Applied upstream (CHECKPOINT 6/7).*
14. **§4.1 / B5**: what per-head specialisation means when the routing table is
    off (16f).
    *Applied upstream (CHECKPOINT 6/7).*
15. **§6, §9.2, §14.4 / B12**: keep "full-rank" and specify the generator - the
    global latent read through a per-head table over relative board-pair cells,
    bilinear in (latent, cell), with no absolute board index; record the
    board-descriptor bilinear form as the rank-32 alternative; correct "14.7 MMAC
    per head per layer" (that is 256 x 240^2, a 256-wide output over every pair);
    say that pairs with a global register get no smolgen bias.
    *Applied upstream (CHECKPOINT 6/7).*
16. **§6 / B11**: one learned scalar per head per relation type, shared across
    layers, every head carrying every type.
    *Applied upstream (CHECKPOINT 6/7).*
17. **§6.1, §10.1 row 4, §13.1 item 6 / B14**: block-sparse attention is built in
    a CPU gather form; the FlexAttention kernel is still to build; under autograd
    the memory bound needs per-block recomputation.
18. **§6 vs §11**: say what the generator exports as "the move graph" (the side to
    move's moves - engine stage E3's reading - or both colours).
19. **§10.4.1**: define E as the out-of-window travel destinations of the
    mover's moves, a branching move counting at the history board it targets
    (spec issue 17).
    *Applied upstream (CHECKPOINT 6/7).*
20. **§10.4.3**: define W(k) - unit, present band - and say whether branch
    origins are core or filler (issues 23, 24).
    *Applied upstream (CHECKPOINT 6/7).*
21. **§15.6 / M9**: define the CPU allocation per GPU, what "expansion" includes,
    and how flatness at fixed |E| is measured (issues 20-22).
    *Applied upstream in part, at CHECKPOINT 2: the CPU allocation per GPU (§15.6,
    M13; issue 20); still needed: whether "expansion" counts the per-board features
    (§15.6 names "legal moves, attention masks, typed edges and E", text that predates
    issue 21) and how flatness at fixed |E| is measured (issue 22).*
22. **§6 / B11**: define the travel-threat relation, or drop it and cite §11.1's
    forcing-move class instead (issue 19).
    *Applied upstream (CHECKPOINT 6/7).*
23. **§2.2**: the has-moved flag exists only for pawns, rooks, kings and brawns
    (5DFEN's `*`); say so.
    *Applied upstream (CHECKPOINT 6/7).*
24. **§11.0 "Stable answers only"**: say how the required set is drawn - the
    stable values whose recorded cost times a margin fits the tier's budget,
    recorded at least twice - and that the gate requires 100% of it (H32).
    *Applied upstream (CHECKPOINT 6/7).*
25. **§11.0 "Recorded-answer integrity"**: name the mechanisms - the manifest
    verified before every comparison, the re-hash of the harness after the engine
    runs, imports from source only - and say plainly that the manifest protects
    against accident, not against an edit of both a golden file and the manifest:
    that is review's job (H33).
    *Applied upstream (CHECKPOINT 6/7).*
26. **§11.0 "Running the gate"**: add the engine's parse mode as the stage before
    full mode, and define `legal_actions` (an action is its set of moves, with
    its timeline-creating moves in creation order) - a protocol definition, not a
    rule.
    *Applied upstream (CHECKPOINT 6/7).*
27. **§7.3 / C17**: define a board's causal past (the chosen reading, spec issue
    29), and say that a frozen board reads only itself and frozen boards of its
    past.
    *Applied upstream (CHECKPOINT 6/7).*
28. **§8.5, §11.3, C11, H18**: the node key includes each timeline's branch origin
    under the lineage reading (and each travel's departure board under two-parent);
    the creation reading has no node key (issue 30).
    *Applied upstream (CHECKPOINT 6/7).*
29. **§2.3 / A3 with §7.3**: what a frozen board's state-dependent features are
    under the cached configuration (issue 31).
    *Applied upstream (CHECKPOINT 6/7).*
30. **§7.3 / C20**: what "plus" means, and the epoch's history as a fixed block
    (issue 32).
    *Applied upstream (CHECKPOINT 6/7).*
31. **§6 / B9-B12**: under the cached configuration, edge-derived masks, biases and
    smolgen apply to live query rows only (issue 33).
    *Applied upstream (CHECKPOINT 6/7) in part: §6 and B11 (masks and typed biases
    on live query rows only); still needed: the same in §6's smolgen paragraph.*
32. **§10.3 / C15 with G1**: FP8 frozen K/V need an FP8 trunk or FP8-frozen-K/V
    training (issue 34).
    *Applied upstream (CHECKPOINT 6/7).*
33. **§7.3**: the supersession bound depends on the causal-past reading; record the
    creation cascade (issue 35).
    *Applied upstream (CHECKPOINT 6/7).*
34. **§7.1, §13, §15.3**: restate the paper ratios per timeline count - 8 against
    the 24 the corpus's real games have (issue 36).
    *Applied upstream (CHECKPOINT 6/7) in part: §7.1 and §13; still needed: §15.3's
    "Cached cost per node" row (29x on paper).*
35. **§10.2's GQA memory table and §10.3**: the per-node delta is superseded boards
    only; KV per state counts frozen boards (issues 37, 38).
    *Applied upstream (CHECKPOINT 6/7).*
36. **§7.6, §7.9, §15.3**: reconcile 0.164 TF / 0.159 TF and +3.6% / +3.8% (issue
    40).
37. **§15.3 rows**: write back the measured figures (M4) once the quiet-machine
    wall timings exist, CPU-labelled; `docs/MEASUREMENTS.md` holds the counts now.
38. **§11.0 item 3 and H37**: "An exhausted budget yields 'unknown'" -> "An
    exhausted budget - nodes, time or memory - yields 'unknown'" (spec issue 44).
    *Applied upstream (CHECKPOINT 6/7).*
39. **§11.0 item 3**: after "Run both: naive as the fast path, hypercuboid for
    near-mate positions", add "the gate's adapter runs them as a portfolio on
    doubling node budgets, the other cross-checking" (issue 45).
    *Applied upstream (CHECKPOINT 6/7).*
40. **Register H22**: "royal-capture check detection (phantom boards for the
    mate-vs-stalemate verdict) ... per-board chess rules as 5D queries" -> "check
    status per H36 (pass on the active boards in the present) ... per-board castling
    and en passant attack tests (H40)" (issue 47).
    *Applied upstream (CHECKPOINT 6/7).*
41. **§8.1**: say whether the pointer head scores pseudo-legal moves, legal first
    moves, or actions - with D2 and spec issues 28, 46 (issue 49).
    *Applied upstream (CHECKPOINT 6/7).*
42. *Withdrawn - superseded by D8 (CHECKPOINT 2), which the upstream documents
    now carry.* ~~§2.3 / A3: either add the player to move as a global input
    feature, or state that leaves are evaluated only at action boundaries or
    before the present shifts - needed under Q4's other reading (issue 48).~~
43. **§6, §15.6 ("expansion - legal moves") and §11.0 item 3**: "legal moves" is
    not defined beyond R1. Proposed: "the moves not proved illegal - each marked
    legal or unknown - by the rules engine's sound tiers under a budget (exact on
    demand); search verifies an unknown move when it first selects it, and scores a
    node terminal only on an exact verdict" (issue 46).
    *Applied upstream (CHECKPOINT 6/7) - in §8.1 ("Which moves are scored").*
44. **§11.1 recommended design, item 1**: "Node-capped alpha-beta over forcing
    actions (iterative deepening in forcing half-turns)" (issue 53).
    *Applied upstream (CHECKPOINT 6/7).*
45. **§11.1 "defining quiet"**: state the forcing set and define "immediate threat"
    (issues 51, 52).
    *Applied upstream (CHECKPOINT 6/7).*
46. **§11.1 two channels**: a loss proof needs the defender's legal actions
    enumerated exhaustively; a capped node gives an estimate; say whether stalemate
    draws are solved values (issues 55, 59).
    *Applied upstream (CHECKPOINT 6/7).*
47. **§11.3**: the TT holds no proofs; the solved store is read first.
    *Applied upstream (CHECKPOINT 6/7) - in §11.1 "As built and measured" and H19.*
48. **§15.6 6b**: define "fully resolved", "scan leaf", the evaluation-rate fallback
    and stand-pat validity, and say whether the verdicts are per phase (issues 58,
    61, 62).
    *Applied upstream (CHECKPOINT 6/7) in part: §15.6's 6b block defines stand-pat
    validity and the evaluation-rate fallback, reads the verdicts per board-count
    range, and refers "the other terms" to docs/MEASUREMENTS.md; still needed, if the
    spec is to carry them: "fully resolved" and "scan leaf" defined in §15.6.*
49. **§15.3 "Scanner nodes/s"**: write back (M4) after the quiet run.
    *Applied upstream (CHECKPOINT 6/7) - as edit 94's "Prover nodes/s" row.*
50. **Register H6**: a 5D piece-value table, pending a decision (issue 56).
    *Applied upstream (CHECKPOINT 6/7) - as register H50 (provisional
    placeholders).*
51. *Withdrawn - superseded at CHECKPOINT 4: M16 and §15.6's "Three readings" give
    the time-share combination M = 1 / (0.9 / m_cached + 0.1 / m_dense), not this
    geometric form (issue 63).* ~~§15.6 "The window": write the multiplier's formula
    (m = m_cached^0.9 · m_dense^0.1 with T = 66N + 16 tokens and the N3 cost model)
    once the user confirms it (issue 63).~~
52. **§15.6 "The window"**: define a phase for the rule and for the kill clause,
    per the user's answer (issue 64), and whose distribution weights the kill
    clause (issue 65).
    *Applied upstream (CHECKPOINT 6/7).*
53. **§15.6's window table and the register (M4 write-back)**: add the measured
    rows - N 24 / 60 / 112 / 170, active T 1 / 2 / 4 / 2, W(8) 9 / 19 / 36 / 19,
    median |E(8)| 0 / 0 / 2 / 2, k = 8 ceilings −1.1 / 12.4 / 31.0 / 87.6, k = 4
    ceilings 2.9 / 20.4 / 44.0 / 95.6 - and qualify "the late game dominates" and
    "at ~100 boards it cannot pay at k = 8" against them.
    *Applied upstream (CHECKPOINT 6/7) in part: §15.6 gives the measured median N
    and active timelines and qualifies both claims; still needed: the measured W(8),
    median |E(8)| and ceiling rows in its table.*
54. **§10.4.2 step 1**: the |E| phase curve is measured on real games to 186
    boards; above that, provisional.
    *Applied upstream (CHECKPOINT 6/7).*
55. **The register's open item "Real game records for the |E| curve"**: records
    now measured (t0database); the verdict is provisional only above 186 boards.
    *Applied upstream (CHECKPOINT 6/7).*
56. **§11.1 "The redesign after 6b", part 2**: state the proof channel's question
    exactly - mate within d by the attacker's forcing class against every legal
    defence - with the class's parameters, that disproofs are class-relative and
    go to a disproof store, never the solved store, and the decomposition's
    exactness condition with its E4 fallback (issues 67-69, 73, 74) - subject to
    the design review.
    *Applied upstream (CHECKPOINT 6/7) in part: class-relative disproofs in a
    separate store (§11.1, H19) and the class as a placeholder (H43); still needed:
    the decomposition's exactness condition and its E4 fallback.*
57. *Withdrawn - moot since CHECKPOINT 4: the per-leaf estimate channel is retired
    (H2; §11.4 "Retired"; issue 70).* ~~§11.1 part 1: the estimate channel's local
    model (issue 70).~~
58. **§11.3 and register H19**: the disproof store, keyed by position, class and
    depth (issue 74).
    *Applied upstream (CHECKPOINT 6/7).*
59. **§15.6's CHECKPOINT 3 / 6b′ block and the register (M4 write-back)**: the
    "fits" operationalisation, both definitions of "decided", the tie rule, and the
    per-range figures (MEASUREMENTS, 6b′) (issues 71, 72).
    *Applied upstream (CHECKPOINT 6/7).*
60. **The register, M14 and M15**: 6b′ measured - STOP; the design review's
    outcome to follow.
    *Applied upstream (CHECKPOINT 6/7).*
61. **§7.9 "Complexity as a trigger"**: state the rule as built (a weighted score
    against a threshold; rung 1's visit gate; a per-search budget; never the root;
    travel from the epoch anchor; effort from the parent's evaluation at a
    quantile; add-only for every signal, rung 1 decided first and unchanged), name
    the placeholders, and give their tuning to 6c (issues 75-79).
    *Applied upstream (CHECKPOINT 6/7).*
62. **Register C24**: "add-only if self-rated" -> "add-only (as built, for every
    signal; rung 1 decided first)", noting "built behind a flag, off" (issue 79).
    *Applied upstream (CHECKPOINT 6/7).*
63. **§8.6, the effort head**: a second consumer - C24 reads the parent's predicted
    difficulty at internal nodes (off-label, confidence-family, add-only) (issue
    78).
    *Applied upstream (CHECKPOINT 6/7).*
64. **§8.5**: "values depend only on the state" -> the state includes whether an
    action is in progress; a turn start and a mid-action state with equal
    boards are different nodes; the network input is unchanged (issue 87).
    *Applied upstream (CHECKPOINT 6/7), with one change: the network now tells the
    two apart, through A6's turn-state inputs (§8.5, §2.3).*
65. **§15.4's invariants**: add - a submit edge leads to a turn start, a move
    edge to a mid-action state; no game end during an action; no dead end with a
    legal completion.
    *Applied upstream (CHECKPOINT 6/7).*
66. **§8.7 / C14**: the shared, state-keyed evaluation cache and its board-count
    purge (issue 89).
    *Applied upstream (CHECKPOINT 6/7).*
67. **§8.2's whole-turn arm**: the listing cap's memory role, and the missing
    end-of-turn token (issues 92, 93).
    *Applied upstream (CHECKPOINT 6/7) in substance, by a different ruling: the flat
    form supersedes the whole-turn arm, its cap and the token (D2, §8.2).*
68. **§8's solver text** points at H41's switch (`proof_channel`).
    *Applied upstream (CHECKPOINT 6/7) - in §11.1 and H41.*
69. **§8.4**: name E1's, E2/E3's, E5's and E9's constants, the submit prior and
    the whole-turn prior as parameters with placeholder defaults (issues 80-85).
    *Applied upstream (CHECKPOINT 6/7) in part: §8.4 "Constants as built" names
    E1's, E2/E3's, E5's and E9's constants and the whole-turn prior; still needed:
    the submit prior (`submit_share`).*
70. **I11 / §8.6**: the turn cap's definition and the capped record's
    Termination header (issue 88).
    *Applied upstream (CHECKPOINT 6/7) - in I11 and §12.3.*
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
    *Applied upstream (CHECKPOINT 6/7) in part: H28 and §11.0 "Running the gate"
    carry the exception; still needed: §11.0's "Third-party code" row.*

72. **§11.4's Guard row**: the depth semantics, the submission reading, that
    exactness enumerates every opponent action ("existence" means ours), at most
    50 depths (issues 97-100).
    *Applied upstream (CHECKPOINT 6/7) in part: the depth semantics, the submission
    reading and exactness over every opponent action (§11.4, H42); still needed: "at
    most 50 depths".*
73. **§11.4's Hunt row and §11.1**: prove-only as defined; export refused until
    H41/H23 (issue 101).
    *Applied upstream (CHECKPOINT 6/7).*
74. **§15.6's 6b″ block**: the readings (the chosen action, the truth's class
    union, budget and curve, no-verdict ranges); after the quiet run, the verdicts
    and the M4 write-back to MEASUREMENTS (issues 102-105).
    *Applied upstream (CHECKPOINT 6/7).*
75. **§13**: per 6b″'s verdicts ("the chosen move goes unguarded" where Guard
    fails; "proofs only from the search's solver" where Hunt fails).
76. **§11.2 rung 4**: the asset map as built (issue 106).
    *Applied upstream (CHECKPOINT 6/7).*
77. **§11.3 / M13**: Lazy SMP is no help for exhaustive refutations (issue 100).
    *Applied upstream (CHECKPOINT 6/7) - in H17 and §11.4.*
78. **Register H1, H10, H42, H43, M14, M17**: as built.
    *Applied upstream (CHECKPOINT 6/7) in part: H1, H42, H43 and M17 carry 6b″, and
    M14 has said "superseded by v3" since CHECKPOINT 4; still needed: H10 as built
    after 6b″ - it still reads "rung 2's promotion side becomes Guard", with Guard
    adopted nowhere (H42) - and the same in §11.2's status note (edit 205).*
79. **§2.3 / A3**: add the global acting-player input (issue 108).
    *Applied upstream (CHECKPOINT 6/7).*
80. **§8.2 / D8**: name register 0 and the linear readout as placeholders; one
    distribution over moves + submit; an illegal submit gets no probability and
    no logit is sent (issues 109, 112).
    *Applied upstream (CHECKPOINT 6/7) in part, by a different ruling: the logit is
    read from the pooled global registers, not register 0 (D8, M20); one
    distribution, and no probability for an illegal submit, are in §8.2; still
    needed: the readout named as a placeholder, and "no logit is sent".*
81. **§8.3 / §12.4 / D3 / D4**: value and WDL "from the acting player's
    perspective (D8)" (issue 110).
    *Applied upstream (CHECKPOINT 6/7).*
82. **§12.2 / §12.4**: the policy target is the visits over moves + submit; the
    exclusion and provenance apply to the submit child too (issue 111).
    *Applied upstream (CHECKPOINT 6/7).*
83. **§9.2 / F1**: 21,359,370 parameters (issue 113).
    *Applied upstream (CHECKPOINT 6/7).*
84. **§8.7 / C14 / C12**: one search and cache per evaluator in match play; a cache
    is shared only between searches with one evaluator and one network input, and
    evaluations are keyed by refresh epoch (issues 114-116).
    *Applied upstream (CHECKPOINT 6/7).*
85. *Withdrawn - the register holds the state of each decision, not its build
    status (its header); D8's build status is under "Register status" here.* ~~The
    register's D8 row: built (engine, search, network).~~
86. *Withdrawn - moot: the register's open items (part K) list none of issues 48, 5
    and 29, before CHECKPOINTS 6-7 or after.* ~~The register's open items: issue 48
    closed by D8; issues 5 and 29 closed by C23.~~
87. **§15.6's 6b″ block**: only confirmed proofs count; a refuted proof withholds
    the verdict; the shipped configuration is the measured one; the "too few to
    read" flag (issues 117, 118).
    *Applied upstream (CHECKPOINT 6/7).*
88. **§15.6 / ruling 5**: the truth is pinned with its provenance, and the tools
    refuse other records or rules (issue 119).
    *Applied upstream (CHECKPOINT 6/7).*
89. **H41/H23 and H43**: Hunt's export switch is refused like the search's proof
    store.
    *Applied upstream (CHECKPOINT 6/7).*
90. **§11.1 / H19, wherever disproof classes are described**: a disproof's class is
    every option that shapes the attacker's class.
    *Applied upstream (CHECKPOINT 6/7).*
91. **§15.2's rows that cannot exist in the untrained build**: "windowed = full
    restricted" (C5) and "lazy K/V = eager" (C10) - the window is killed,
    provisionally (M16): mark them deferred to the pilot self-play run (M18);
    "NNUE accumulator = recompute" (F4) - the NNUE is held open (H6): mark it
    held open.
    *Applied upstream (CHECKPOINT 6/7).*
92. **§15.3's rows that cannot be measured as written**: windowed throughput, the
    windowed "one refresh ≈ 10 evaluations", windowed KV memory (the window is
    killed provisionally); the local/global interleave (C18 held open); NNUE
    evaluation (held open); "rung 1 hidden under batch latency" (rung 1 retired) -
    mark each deferred or void, with the reason.
    *Applied upstream (CHECKPOINT 6/7).*
93. **§15.3's movegen row and §11's projection (M4 write-back)**: pseudo-legal
    44,047 actions/s at 240 boards and 24,951 at 600 (one thread); with fast
    legality 20,249/s at 240 on one thread (1.01x the target) and 37,316/s on M13's
    two expansion cores (1.87x), 12,133/s at 600 (MEASUREMENTS, M9).
    *Applied upstream (CHECKPOINT 6/7) in part: §15.3's movegen row is written back;
    still needed: §11's projection ("Likely to be the actual bottleneck").*
94. **§15.3's scanner row (M4 write-back), restated for v3**: the brute-force
    scanner 787k nodes/s per thread alone and 566k at 5 threads (E5c), df-pn 239k
    (6b′); the shipped consumers are Guard and Hunt - their figures are nodes per
    root at 50 ms (6b″: Guard about 18-24k, Hunt 0.7-61k by range); supersedes
    proposed edit 49.
    *Applied upstream (CHECKPOINT 6/7).*
95. **Register F4, H7, H14, C5, C7**: F4 and H7 "Committed" -> held open with the
    NNUE (H6, CHECKPOINT 4); H14 restated for the reduced ladder (Guard, Hunt, the
    asset map); C5 and C7 "Provisional" -> killed, provisionally (M16), re-checked
    by M18.
    *Applied upstream (CHECKPOINT 6/7).*
96. **Register C9, D6, I9 and the process rows**: record their status as built
    (I9: the ruleset ID in every key) or to be verified (C9).
    *Applied upstream (CHECKPOINT 6/7) in part: I9 ("the ruleset ID in every state
    and store key, as built"); C9, D6 and the process rows unchanged.*
97. **§15.6's Phase-3 re-check**: "movegen ≥ 2x the measured evaluation rate"
    needs a GPU evaluation rate (issue 120); until then the CPU figure (N5's 6.9 F1
    evaluations/s on CPU) is the only one, and the re-check is deferred.
    *Applied upstream (CHECKPOINT 6/7).*
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
    *Settled upstream at CHECKPOINT 8: the user ruled for a two-oracle rule (issue 122); §11.0 and register H47 now say the gate is aligned to two oracles, with the commit tier's baseline rule and restricted runs (issues 123-124).*
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
    *Ruled at CHECKPOINT 8 (issue 126): report it, as a question (H27; §11.0 now says so) - so (b), no edit; the entries' `upstream` fields name the report once filed (gate step 3b).*
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
101. **Register H51 and §11.0 "Capture markers are read leniently" - pending the
    user's view on issue 129.** H51 says: "...; the engine writes the marker strictly;
    a source that refuses such a record gets a known-divergence entry citing this row
    (KD-0167–0169's case: cwmtt/futures, action 17)". It should say: "...; the engine
    writes the marker strictly; a source that refuses such a record gets a
    known-divergence entry citing this row (cwmtt/futures action 17: ftxi, KD-0170;
    KD-0167–0169, which recorded the strict reading, withdrawn); measurements pinned
    before it (6b, 6b′, 6b″, Hunt's ground truth) keep the reading they were made
    under, a written x must capture, so they reproduce". §11.0, after "and the engine
    writes the marker strictly.": "Measurements pinned before this decision keep the
    reading they were made under, so they reproduce." Why: as built (issue 129), and
    the entries the register names were withdrawn on the ruling.
102. **§11.0 "A known-divergences file" and register H27 - pending the user's view on
    issue 130.** §11.0 says: "(and reported upstream — defects as bug reports,
    disagreements with a documented rule as questions citing the source, which also
    tests our own reading; the owner files them)". It should add: "; a divergence that
    a project decision settles, where no source states the rule, is recorded
    unreported, with that reason". H27, after "drafted by the agent and filed by the
    owner": "; divergences settled by a project decision are recorded unreported".
    Why: KD-0170 (issue 130) - neither a defect nor a disagreement with a documented
    rule.
103. **§11.0 "Tiered CI" and register H47 - as built at CHECKPOINT 8** (issues
    131-133). §11.0 now ends: "... The commit tier still fails if the engine declines
    a record it read on the last green run, or reads none; a run restricted to chosen
    sources or records is a developer convenience and never produces a tier verdict
    (CHECKPOINT 8)." It should add: "As built: a ruleset is gated only when each of
    its records is followed, as far as the engine went, by at least two independent
    implementations (two searches of one count once; a parse-only source never
    counts). The last green run is the engine's baseline - for each record of the
    commit tier, how the engine's replay ended and how many positions it reported on
    a green run - written by an explicit command from a green run, never by the
    gate, and pinned like any recorded answer; every tier compares the engine with
    it, over the records both hold. A restricted run reports its findings, labelled
    'restricted - not a tier result', and its exit code reflects them only." H47
    now: "... since gating needs two oracles — as built the gate required one, and is
    aligned to two; the commit tier still fails ..." It should say: "... since gating
    needs two oracles (each record of a ruleset followed by two independent
    implementations; built 2026-10-17); the commit tier still fails if the engine
    declines a record it read on the last green run (a pinned baseline, compared in
    every tier), or reads none; ...". Why: the documents say what is ruled and not how
    it is read; issues 131-133 give the readings built.
104. **§11.0 "Recorded-answer integrity" and register H33** (issue 132). §11.0 now:
    "Recorded answers are immutable, with committed content hashes verified before
    every comparison; the comparison logic lives in harness code the engine adapter
    cannot modify; harness changes require review." It should add: "The engine's
    baseline (H47) is a recorded answer too: written only by an explicit command
    from a green run, pinned by its hash in the gate's policy beside the rulebook's
    pin - not in the golden manifest, so that re-recording the oracles never re-pins
    it - and verified before every read, the command's own read of the baseline it
    replaces included; like the manifest, its pin guards against accident, not a
    deliberate edit of the file and its pin together - that is review's job." H33,
    after "verified before every comparison": "(the oracles' golden files, and the
    engine's baseline of what it read on the last green run)", and its sentence on
    the manifest's limit extended to the baseline's pin. Why: H33 covers recorded
    answers, and the baseline is one (issue 132).

<!-- A new list, so that the numbers from 200 render as written. -->

200. **§8.5, §8.7 and register C11, C14, H18 (identity and the evaluation cache's
    key)**: now - MCGS identity "adds branch origins and whether a turn is in
    progress"; the cache is "state-keyed (with ruleset and refresh epoch)". Proposed -
    both add the number of moves already made in the action in progress (A6's input),
    which subsumes "whether a turn is in progress" (0 at a turn start). Why: the
    network sees the count (A6), so an evaluation is a function of the state, the
    epoch and the count, and §15.2's purity row needs the key to carry everything the
    network sees (issue 302).
201. **§7.1, §10.3 and §15.3's cached rows, with §2.2's identity sentence**: now - no
    word on perspective under the cache; §2.2: "identity is unaffected, since state
    keys hash the state and carry the acting player". Proposed - under A7 a frozen
    board's K/V are per acting colour: up to two entries per frozen board and a
    refresh's history per perspective, priced beside the paper ratio; §2.2: state keys
    are unaffected, while cached K/V are keyed by perspective too (issue 303).
202. **§15.2's A7 row**: now - "give identical canonical inputs and outputs". Proposed -
    "give canonical inputs identical except the acting-colour input, and identical
    outputs (mapped back) when that input is equalised" (issue 304).
203. **§12.4's moves-left row, §8.3, §8.6 and §12.8 (register D3, D4, E7, I13, I19)**:
    now - moves-left "counted in turns", short-term values "at two or three horizons",
    I13's band and I19's cutoff with no unit; §8.6 in board-moves. Proposed - "a turn
    is one player's action, from a turn start to its submission (D8)" wherever these
    count turns; I11's cap counts the notation's full turns; the allocator converts
    turns to board-moves by a boards-per-turn figure per board-count range (M22's,
    then the pilot run's) (issue 305).
204. **§8.6 "Guards" (the reserve) and register E7 ("root-scan reserve")**: now - a
    reserve for one re-search if the root-rule scan refutes, which §11.4 has Guard
    draw on. Proposed - say what draws on it under v3: Guard's refutation, offline
    while Guard is adopted nowhere (H42), and whether the settling scans under 50
    boards do; while neither does, no reserve is held (issue 306).
205. **§12.3 ("The scanner's relative value rises") and §11.2's status note**: §12.3
    now - "(Since CHECKPOINT 4 that means Guard, Hunt and the search's solver at every
    size, with full settling scans only under 50 boards — §11.4.)"; §11.2 - "rung 2's
    promotion side becomes Guard". Proposed - §12.3: "(Since 6b″: the search's solver
    at every size, Hunt under 100 boards and the settling scans under 50; Guard is
    adopted in no range — §11.4, H42, H43.)"; §11.2: "rung 2's promotion side was to
    become Guard, which 6b″ adopted in no range". Why: both predate 6b″ and contradict
    H42 and H43; edit 75 covers §13's matching sentence, edit 78 H10.
206. **The register's status vocabulary** (its header table): it defines Committed,
    Provisional, Prototype, Held open, Deferred, Refused and Reversed; the rows also
    use Superseded (B6, H11; M14's "superseded by v3"), Retired (H12), Killed,
    provisionally (C5, C7), Prepared (G8), Measured (M8, M9, M15, M17), Not adopted in
    any range and Adopted under 100 boards (H42, H43), Undecided (D2), Open knob (D6)
    and Contingency only (B3). Proposed: define each, or map it to one of the seven -
    e.g. Superseded: replaced by the named row, not built; Retired: dropped from the
    design, not built; Killed, provisionally: not built until the named re-check;
    Prepared: built behind a flag and smoke-tested, not adopted; Measured: the figure
    recorded, its verdict in the row; Adopted under / Not adopted in: shipped as
    measured, per range. Why: the vocabulary is how a reader tells how firm a row is,
    and these words are outside it.

<!-- A new list, so that the numbers from 150 render as written. -->

150. **§9.3's first bullet and register F5**: now - the norms "no mean-centring, no
    bias" and "no biases in the Q, K and V projections", the attention-output and FFN
    biases kept only in the Mish bullet's "as here". Proposed - the first bullet and
    the row say it: "the attention-output and FFN projections keep their biases", and
    name the gain and eps if they are to be pinned (as built: a learned gain, eps 1e-5
    inside the root). Why: a register row that leaves them out can be read either way
    (issue 200).
151. **§9.3's second bullet, §13.1 6y and register F5**: now - "Lc0's post-norm with
    DeepNet initialisation". Proposed - name the arm as built, as placeholders until
    Lc0's arrangement is given where it differs: DeepNorm with the paper's constants
    for an L-layer stack, α = (2L)^(1/4) on the residual and β = (8L)^(-1/4) as the
    gain of the Xavier normal the value, attention-output and FFN projections are
    drawn from (Q and K at gain 1), as the paper's pseudocode draws them; no final
    norm; no norm on the embedding; smolgen reading the block's input (issue 201).
152. **Register B12**: now - "keeps its own normalisation (F5)". Proposed - "has no
    normalisation inside (none is built; F5's exception is moot)", unless one is to be
    added (issue 202).
153. **§9.2's Params row and register F1 (M4 write-back)**: now - "21,359,370 as built
    at N5, before CHECKPOINT 6's normalisation and displacement-bias changes".
    Proposed - "21,338,634 as built after F5 (before B15's displacement table)", and
    F2's preset 44,456,458 against the ladder's 44.0M (issue 203). Supersedes edit 83.
154. **Register F6 and §9.3's third bullet**: now - "Mish for v1 (Lc0 from T79;
    Chessformer's FFN and output heads)". Proposed - say whether the output heads'
    hidden activation is Mish too (as built: the value head's MLP keeps GELU, a
    placeholder), so that D3's head rewrite does not settle it by accident; and, for
    6y's squared-ReLU arm, its initialisation if it is not Mish's (issue 204).
    *Settled upstream at CHECKPOINT 8 in part: register F6 and §9.3 (Mish in the trunk FFN and
    every head's hidden layers; squared ReLU the trunk-only arm) - built (N-F6heads). Still
    open: the squared-ReLU arm's initialisation (as built: Mish's).*
155. **§4.1's B15 bullets and register B15**: now - "binned and clamped" (no bins), "exact
    line classes ... equal steps along two, three or four axes", "per-board registers by
    their board-distance class, global registers one entry per head", "about 1.6M
    parameters", no initialisation. Proposed - state the as-built placeholders, or what
    replaces them: far classes by D = max(|Δtimeline|, ⌈|Δtime|/2⌉), log2 bins from 3
    ([3, 4], [5, 8], 9+, the last clamped) crossed with the sign of Δtime; line classes in
    whole turns (an odd Δtime is never a line), one entry per direction and step,
    replacing the far slice's, board-axis lines clamped at 8 steps; per-board register
    entries by class and pair kind (square reads register, register reads square,
    register reads register); the global entry per head per layer, either side; 12,681
    entries per head per layer, 1,623,168 parameters; initialisation N(0, 0.1) (issues
    205-208). If the user rules that B6 stays B15's special case (issue 205), the far
    classes are instead B6's (Δtime, Δtimeline) cells beyond the exact box, with the
    board-axis line clamp at 64 turns (about 5.2M parameters), or B6's two tables are
    added beyond the box (3,456 more) - and "about 1.6M parameters" changes with the
    first.
156. **§9.2's Params row, register F1 and F2 (M4 write-back)**: now - F1 "~21.5M with heads
    (21,359,370 as built at N5)", F2 "L24-ff12 (44M)". Proposed - "22,961,802 as built
    after F5 and B15 (21,338,634 without B15's 1,623,168-parameter table)" and F2
    "46,891,210 (44,456,458 without the table)" (issue 210). Supersedes edit 153.
    *Moved by M20's dispositions (`lane/net`, N-M20: the promotion offset +3,341, the learned
    register coordinates -4): F1 22,965,139 (21,341,971 without the table), F2 46,894,547
    (44,459,795) - the figures this edit should carry.*
157. **§6.1 and register B14**: now - "FlexAttention exploits block sparsity losslessly".
    Proposed - add: "register rows that read every key on a masked head (v1's rule) are
    computed apart - a second call, or the registers ordered after the squares - or the
    block mask keeps every block" (issue 209).
158. **§15.2's B15 row, §10.2 and register B14, G9**: now - §15.2: "Displacement bias
    through the score function = the same bias materialised, on small positions"; §10.2
    ("The inference path"): every attention bias "is a FlexAttention score function
    computed inside the kernel"; register B14: "FlexAttention score_mod"; G9: "PyTorch
    (torch.compile + FlexAttention)". Proposed - name the untrained build's form in each:
    "the score function run by uncompiled FlexAttention on CPU (forward only, every score
    materialised: small positions), its gradients checked through the CPU block paths,
    which build the same biases by lookup; the compiled kernel on GPU time (M19)". §15.3's
    "Without a GPU" already defers "the FlexAttention block-sparse kernel" to GPU time and
    needs no edit (issue 211).
159. **Register B6 and B12, §4.1's B6 paragraph, reversal 25**: now - §4.1: B6 is
    "Superseded by the 4D displacement bias below (CHECKPOINT 6), of which it is the
    special case that ignores file and rank"; register B6: "Superseded by B15 (CHECKPOINT
    6) — B6 is its special case that ignores file and rank"; reversal 25: "A learned 4D
    displacement bias beside RoPE (B15), B6 its special case"; B12 "as built over
    relative board-pair cells". Proposed - B6's status: "Superseded by B15 and removed (no
    arm survives)". The "special case" clause, in all three places, as the user rules on
    issue 205: if B15's far classes stay as built (the ~1.6M holds), §4.1 reads
    "Superseded by the 4D displacement bias below (CHECKPOINT 6), which keeps its clamped
    extrapolation and Δtime's sign but not its per-axis buckets beyond the exact classes",
    and register B6 and reversal 25 drop "its special case" the same way; if the clause is
    to hold, the far classes change instead (edit 155) and the clause stays. B12: "over
    relative board-pair cells - the signed-Δtime x Δtimeline log buckets B6 used (19 x
    8), kept for smolgen" (issue 212). Why: as built the clause is false, and it is what
    says that removing B6, with no arm left in ablation 2, loses nothing (issue 205).
160. **§4.1's status paragraph ("Relative to the present, at score time"), §2.3's Δt bullet,
    §7.3's last sentence on frozen boards, and register B16 and A3 (with M20)**: now - §4.1:
    "Under the cached configuration frozen boards therefore carry only intrinsic features, and
    current status enters as score-time key biases — per head, by the key board's
    Δt-to-present bin and timeline status — recomputed every evaluation for the cost of a
    lookup"; §7.3: "carries only intrinsic features; current status reaches it at score time";
    register B16: "Score-time status key biases on frozen boards — Δt-to-present bin and
    timeline status — so frozen tokens carry intrinsic features only". Proposed - say (1) where
    the biases apply: on live query rows (head boards' tokens and the global registers) against
    frozen boards' tokens, never on a frozen row (C17); (2) the refresh: "frozen boards carry
    only intrinsic features, except a refreshed epoch's history boards, whose K/V carry the
    anchor's (§7.9, M20's ruling of issue 31); B16 biases them at every node below the anchor,
    and adds nothing at the anchor, whose evaluation is the refresh pass"; (3) the as-built
    placeholders, or what replaces them: signed log2 Δt bins exact below 4 half-turns and
    clamped at 128 (19), "timeline status" the active/inactive flag, one table per head shared
    by the layers (304 parameters), N(0, 0.1), and whether an off arm exists (issues
    213-216). Why: as written, §4.1 and §7.3 can be read to put intrinsic features and B16 into
    the refresh too, which M20 ruled against, and they leave every committed-class detail open.
161. **§4.1's Shaw paragraph ("Not Lc0's BT5 encoding"), register B17 and §15.3's B17 row**:
    now - "confined to a compact set of ~1,400 (the exact near classes plus the line classes)";
    register B17: "each query dotted with learned vectors for a compact class set (~1,400: the
    exact near classes plus the line classes)"; no parameter count, query, scale or
    initialisation. Proposed - state the as-built placeholders, or what replaces them: the
    classes - one of issue 217's two readings, each 1,493 classes at the same cost, which cover
    different pairs: the file-rank offsets within ±2 inside every one of B15's 45 exact
    board-distance classes (as built), or the nearest board-distance classes with their full 15 x
    15 slices (which ones) - plus B15's line classes (§15.3's "with ~1,400 classes" -> "with
    1,493 classes as built"), none for pairs with registers; a vector per class per head per layer,
    6,115,328 parameters (26.6% of the v1), or shared by the heads, 764,416 - say which; the
    unrotated query and the scale 1/sqrt(d_head); N(0, 0.1) (issues 217-220). Why: B17 is class
    C, each is a committed-class detail of ablation 2's arm, and its size - a quarter of the
    network per head - is not one to leave unstated.
162. **Register F6 and §9.3's Mish bullet (with §2.2, §2.3 and register A6, A8, B12)**: now -
    "Mish for v1, in the trunk FFN and every head's hidden layers (CHECKPOINT 8)"; §9.3: "Mish
    applies in every head's hidden layers too". Proposed - say whether the input side's MLPs
    follow: the Δt-to-present embedding (§2.3; as built GELU), A6's turn-state embedding and
    A8's "FFN after the embedding" (§2.2) when they are built, and B12's smolgen context layer
    (as built SiLU, Lc0's) - Mish everywhere, or "the embedding's and smolgen's activations
    are implementation choices". Why: F6 names two places and is read as naming only those;
    the input encoding is committed-class too (issue 221).
163. **§8.1's promotion sentence and register D1 (with D4, D9 and §12.4's soft policy)**: now -
    "Promotions add a learned per-piece logit offset"; D1: "promotions get a learned
    per-piece offset". Proposed - "a learned per-piece logit offset read from the
    destination square's final state (one logit per piece type)" - or a constant per piece,
    if that is meant - and say which pointer heads carry it: the policy (as built), and
    whether the soft policy, the opponent-reply prototype and D9's optimistic policy do
    (issue 222).
164. **Register M20's "aux pooling -> squares" and D4's target list, with §9.2's "Aux heads"
    row**: now - M20: "aux pooling -> squares; ... value pooling -> mean (D3)"; D4 lists
    WDL, short-term value, value error and the value distribution among the auxiliary
    targets. Proposed - M20: "aux pooling -> squares (the auxiliary heads other than D3's
    value heads - moves-left, search effort, per-board, per-square)"; §9.2's "Aux heads"
    row: "moves-left and effort mean-pool every square, per-board heads their board's;
    D3's value heads - WDL among them - pool the global registers (§8.3)". Why: as written
    WDL and the short-term value fall under both rules (issue 223).
165. **§12.4's moves-left row and register D4 (with the main loop's edit 203)**: now -
    "Moves-left — counted in turns, so both D2 arms share it". Proposed - add whether the
    action in progress counts at a decision inside it (the turns left after it is
    submitted, or including it), so the training loop's target is defined (issue 224).
166. **§13.1's ablation 3 and register D8 (with A6, D3 and §2.3)**: now - ablation 3:
    "Registers on/off, 2 vs 4 per board"; D8: "plus a global acting-player input" (M20: "on
    the global registers"). Proposed - say whether the "off" arm removes the global
    registers; if it does, say where the global inputs (D8's acting player, A6's turn-state
    counts) enter - as built, every board token, the live boards' under the block-causal
    scope - and what D3's value heads and D8's submit logit pool (as built, the squares).
    Why: otherwise the arm measures "no registers" mixed with "no D8 input" (issue 225).

<!-- A new list, so that the numbers from 110 render as written. -->

110. **§8.2 "Measure first" (M4 write-back of M22)**: says "How many turns have more
    than one mandatory board, and how many optional moves are played, per
    board-count range, on the human games — rules engine only, hours. Under 50
    boards the human games' median position has one active timeline, so most turns
    there are single moves, where every arm is identical; the arms differ only where
    turns are wide." Proposed, appended: "Measured (M22; t0database's 692 games,
    MEASUREMENTS), by the turn's start: turns with more than one playable board -
    where the arms differ - 15.1% under 50 boards, 65.2% at 50-99, 99.4% at 100-199
    (13 games); with more than one mandatory board 5.8%, 27.1%, 64.0%; with three or
    more 0.7%, 6.7%, 16.0%; with an optional board 10.5%, 44.5%, 65.7%; turns with an
    optional move played 1.2%, 2.5%, 1.1%. No human turn starts at 200 boards or
    more." Why: M4 - the measured figures go beside the paragraph's expectation, which
    they confirm under 50 boards when it is read by the turn's start (84.9% of the
    turns there start with one playable board, where every arm is identical) - not by
    the share played with one move (95.9%), which counts as single moves the turns
    with several boards that a jump between two of them or a rewinding branch
    resolves in one move, and every arm searches those differently.
111. **Register M22**: says "Turn-width measurement before the D2 arms: mandatory
    boards per turn and optional moves played, per board-count range, on the human
    games — rules engine only | Committed (process)". Proposed: record it measured
    (2026-10-17, MEASUREMENTS "Turn width (M22)"), with its definitions: mandatory
    boards = the active boards in the present at the action's start (R9); an
    optional move = a move made on a board not mandatory at the action's start - its
    source, or the destination of a jump that does not branch (T3, J4), so a jump
    across the line counts either way (issue 140); boards whose status changes
    mid-action - released by a rewind (T8, T12) or made mandatory by a reactivation
    (T11, T12) - keep their status at the action's start; ranges < 50 / 50-99 /
    100-199 / 200+ boards at the action's start; the unit the action of a distinct
    game (issue 141). Why: the measurement is done; the row should say what its words
    were taken to mean.
112. **Register part K, "Turn width on human games: share of turns with more than
    one mandatory board; optional moves played | Priority of the D2 arms (M22)"**:
    strike the item through as the resolved ones are ("~~...~~ — measured (M22):
    more than one mandatory board in 5.8% / 27.1% / 64.0% of turns under 50 / at
    50-99 / at 100-199 boards, more than one playable board in 15.1% / 65.2% /
    99.4%; optional moves played in 1.2% / 2.5% / 1.1%"), and keep its open half -
    how the figures order the D2 work - as a new item, or under the §8.2
    turn-construction item, until the user rules (issue 142). Why: the measured half
    is answered; the ordering half has no rule.
113. **Register part L, "Most human-game turns under 50 boards have exactly one
    mandatory board | §8.2 (turn-width measurement)"**: append the outcome as part L
    does for the others - "— **held** (M22): 94.2% of the turns under 50 boards
    (22,605 of 24,002, in 692 games)". Why: part L keeps each prediction with its
    outcome.
114. **§15.1's Phase 1 row "Turn width: mandatory boards per turn and optional moves
    played, per board-count range, on the human games | How much the D2 arms matter
    (§8.2)"**: record the outcome in its "Settles" column (measured, M22,
    MEASUREMENTS), as the scanner's row records 6b's; and note that §13.1 6x's "after
    the turn-width measurement (§15.1)" is now met. Why: the build order should show
    which Phase 1 measurements are done.
115. **Register D2 and part K's "§8.2 turn construction — ply split vs tree form vs
    flat form | Couples MCGS; measured jointly; turn-width measurement first
    (M22)"**: D2's status says "**Undecided** — measured jointly with widening and
    MCGS, after the turn-width measurement (M22)". Proposed: note in both, in the
    words edit 114 uses for §13.1 6x, that the turn-width measurement is done (M22,
    2026-10-17, MEASUREMENTS "Turn width (M22)"); D2 stays Undecided, decided by 6x.
    Why: edit 114 alone would leave the register's D2 status and part K's gate still
    waiting on M22 while the spec says the precondition is met - and a decision changes
    in the spec and the register in the same edit (CLAUDE.md).
116. **Register A6 and §2.3's turn-state bullet (the mandatory count)**: now - "exact from
    the rules engine: −1 per ordinary move, −2 for a move between two present boards, 0
    after a rewinding branch" (A6); "one fewer per ordinary move, two fewer for a move
    between two present boards (it creates a board on both), zero once a branch rewinds the
    present" (§2.3). Proposed: "exact from the rules engine - the state machine's count (T5,
    T6; Q5, Q6): a move lowers it by the mandatory boards it plays (one; two for a jump
    between two of them - a jump that does not branch plays the board it lands on, T6, so
    one from an optional board onto a mandatory board lowers it by one); a branch that
    rewinds the present zeroes it (T8; not under Q5's AllActive reading); a reactivation
    behind the present can raise it (T11, T12)". Why: the arithmetic as written leaves out
    T6's arrivals, T12's reactivations and Q5 (issue 143), and "exact counts, never
    predicted ones" makes the rules engine's count the input - which is what is built.
117. **§2.3 and register A6 (the two undefined counts)**: now - "optional boards still
    playable" and "moves already made this turn", undefined. Proposed: "optional boards
    still playable: the acting player's playable boards (L5) that are not mandatory -
    active or not, in the present or not (T6) - and none once no move may be made (Q4)";
    "moves already made this turn: the moves made since the action's turn start, one per
    move (a jump between two boards included); 0 at a turn start - no state records it, so
    the search and the game carry it, and it is part of identity (edit 200)". Why: issue
    144; as built.
118. **§15.4's turn structure**: now - "a submit edge leads to a turn start and a move edge
    to a mid-turn state". Proposed, appended: "and the moves made in the action (A6) count
    the move edges since the last submission: 0 below a submit edge or a whole-turn
    action, the parent's + 1 below a move, every parent of a graph node agreeing". Why:
    issue 145; the count is part of a node's identity (edit 200), and this invariant is
    what holds it.
119. **§11.3 "The full key, as built", §15.2's purity row, §7.9 and §8.5's purity
    sentence (the count in identity, beside edit 200)**: now - §11.3: "MCGS node identity
    adds each timeline's branch origin and whether a turn is in progress (§8.5)"; §15.2:
    "Same state + epoch → same evaluation; MCGS transposition hits return identical
    evaluations | Evaluation purity | C19–C21"; §7.9: "Evaluations become a function of
    (state, refresh epoch)"; §8.5: "so values depend only on the state (and the refresh
    epoch, §7.9)". Proposed - §11.3: "MCGS node identity adds each timeline's branch
    origin and the moves made in the action in progress (A6), which subsumes whether a
    turn is in progress (§8.5)"; §15.2: "Same state, moves made in its action and epoch →
    same evaluation; MCGS transposition hits return identical evaluations"; §7.9: "a
    function of (state, moves made in its action, refresh epoch)"; §8.5: "so values
    depend only on the state and the moves made in its action (and the refresh epoch,
    §7.9)". Why: edit 200 rewords the identity and the cache's key in §8.5's first
    paragraph, §8.7 and register C11, C14 and H18, but these four sentences state the
    same identity and purity - §11.3's is H18's twin (a decision changes in both), and
    with the count an input (A6) one state at one epoch has two evaluations where its
    counts differ, which the row's literal "same state + epoch" forbids (the engine's
    EQ-06 test asserts exactly that since engine ABI round 1; issue 302).
120. **§11.3, "Proofs never live only in the TT" and "Determinism debt, extended"
    (register H17, H19)**: now - the TT "is a lossy cache: entries are evicted and
    overwritten by benign races", proofs export to the permanent solved store, and "Lazy
    SMP's character is benign nondeterminism"; nothing on the stores' own concurrency.
    Proposed - after "the TT never holds the sole copy.": "The stores are shared by the
    threads too, and they are not the TT: they hold facts, so they are exact under
    concurrency - every probe and every write whole, no entry lost, doubled or torn; their
    races are never benign. (As built: 64 shards by key, each behind an exclusive lock; a
    reader-writer lock that did not exclude on its first concurrent use under one
    toolchain's threads library doubled and lost entries, 2026-10-17.)" And in
    "Determinism debt, extended": "Lazy SMP's character is benign nondeterminism in the
    TT - acceptable in production, banned while debugging; the stores of facts are never
    racy." Why: issue 146 - the spec's only word on the threads' races is "benign", which
    holds for the TT alone.
121. **§8.3's WDL paragraph, §12.8's deblundering and register D3, I13 (the search's draw
    and moves-left)**: now - WDL is "the value search uses"; I13's earlier positions "take
    the best move's value and moves-left"; nothing says what the search backs up beside
    the value. Proposed - in §8.3 after "the value search uses": "The search backs up W − L
    as Q and the draw probability D beside it, per node and per edge (Lc0's way; a mate
    has D 0, a stalemate and the turn cap D 1), so each root child carries a search WDL.
    Moves-left is not backed up until its turn unit and the action in progress are
    settled; the search records each node's network moves-left, WDL and error output,
    and the records keep the root children's." In I13: "the best move's value (its
    backed-up Q and D) and moves-left (its network output, offset by the target
    builder)". Why: issue 148.
122. **§13.1 6m, §8.4's E2/E3 paragraph and the constants list, register I10 and E2 (the
    uncertainty switch)**: now - "6m decides which uncertainty signal search consumes";
    E2 and E3 are described as reading "the categorical variance". Proposed - "6m's
    switch (as built `uncertainty_source`, the categorical's variance by default, the
    error head's output its arm, an absent signal 0) gives each node one variance, which
    E2's scale and backup weight, E3's bonus and E5's optional LCB term all read"; and in
    the constants list: "E2's and E3's constants, sized before CHECKPOINT 7 against an
    outcome categorical, are placeholders for the search-value categorical and the error
    head alike". Why: issue 149.
123. **§8.4's noise and temperature sentences, its constants list, and register E1, I19**:
    now - "Root Dirichlet noise during self-play, its total concentration split across the
    legal moves ... A root policy softmax temperature slightly above 1 flattens the prior
    before the noise: KataGo uses 1.25 decaying to 1.1, Lc0's training runs 1.2-1.45".
    Proposed - "... split across the root's open edges - the moves the expansion keeps,
    legal or unknown, and a legal submit (the listed actions under whole turns) - per
    decision, each decision's search being a fresh root. ... KataGo's root temperature is
    1.25 decaying to 1.1; Lc0's training runs use a policy softmax temperature of 1.2-1.45
    at every node (as built, the every-node `policy_temperature`, 1, and the root's
    `root_policy_temperature`, set per search by the training generator, multiply)"; in
    the constants list: "the noise's total concentration (0.03 x 361), the shaping's share
    (one half) and prior cap (0.01), the root policy temperature". Why: issues 150-152.
124. **§12.8's record list, §12.2 and register I15, I2 (forced and steered visits)**: now -
    "visit counts with §12.2's forced and steered flags". Proposed - "visit counts with the
    steered counts (the asset map's boosts, Hunt's suggestions) - forced counts only if
    forced playouts are ever built: as built the search forces none, and KataGo's
    forced-playout pruning (§12.5) is computed at target construction from each root
    child's prior (with and without the noise), visits, value and exploration factor and
    the root's effective cPUCT, all in the record". Why: issue 153.
125. **§8.6's effort paragraph and §12.4's effort row (register D4)**: now - "visits until
    the decision stabilises". Proposed - "the root's visits when its most visited child
    last changed (a child takes over with strictly more visits); the target builder bins it
    by log-visits and censors it against the search's playouts". Why: issue 154.
126. **§12.8's conditions and register I15**: now - "conditions (ruleset, board count,
    network, search configuration, temperature, PCR flag)". Proposed - add "the legality
    classification's budget, with which the engine's expansion rebuilds each decision's
    candidate moves - the network's input - exactly". Why: issue 155.

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
- **2026-10-17 - CHECKPOINTS 6 and 7 in the build log; three lanes.** The design
  documents committed unedited (f07568c). The main loop's plan written into "Current
  state (2026-10-17)": three lanes in worktrees (under M23's rule) and how to recreate
  them, number blocks for spec issues and proposed edits, step 1 by lane with the
  readings and stop conditions the lanes work under, step 2 by lane, the parked
  items. The register status brought up to date; the spec issues the new documents
  settle and the proposed edits they carry marked, after a review against the
  documents; the main loop's issues 300-306 and edits 200-206 logged. The lanes at
  16:54: `lane/gate` 7530e36 (H47 with its review's fixes; the gate with the engine
  configured in progress), `lane/engine` 2864be0 (M22 measured), `lane/net` f830f31
  (F5 and F6 built) - none merged.
- **2026-10-17 - the CI comparison** (after the user's `gh auth login`): main's
  local results equal the last green CI runs on the same code ("Recovery", above);
  the engine job's one failure on 3b33942 (g++-13) is the intermittent five-thread
  Guard test, queued for the engine lane.
- **2026-10-22 - engine ABI round 2a** (`lane/engine`): D3's value outputs in the search -
  WDL the search's value with its draw backed up beside it, moves-left recorded (issue
  148), one deliver call for every head (`t5d_search_deliver_evaluations`; the old call
  retired), 6m's switch (issue 149). Register D3, C14, E2/E3, E5, I10, I13; edits 121-122.
- **2026-10-22 - engine ABI round 2b** (`lane/engine`): E1 and I19's exploration - the
  root noise's total concentration split across the root's open edges (issue 150),
  KataGo-style shaping behind a flag (issue 151), a root policy temperature (issue 152);
  I19's schedules left to the training generator. Register E1, I19; edit 123.
- **2026-10-22 - engine ABI round 2c** (`lane/engine`): I15's and I2's fields - the root
  side table (`child_key`, `prior_noise_free`, `value_variance`, `boost`,
  `exploration_scale`, `forced_visits`, `steered_visits`), the summary's `root_key`,
  `root_cpuct` and `root_best_since`, `t5d_game_expand`'s `legal_moves` with its budget;
  forced playouts not built (issues 153-155). Register I2, I15, D4; edits 124-126.
