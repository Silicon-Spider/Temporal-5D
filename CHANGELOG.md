# Changelog

Every change to this repository is recorded here, in the same edit that makes it,
and the spec is brought in line with it - upstream: the spec and the register are
not edited here, and the edits they need are proposed in `docs/BUILD_LOG.md` (the
rule is in `AGENTS.md`). Newest first. Each entry says what changed and why, and
ends with a `Spec:` line: which sections of the design spec and the decision
register the change bears on and what was proposed for them, or that no part of
the spec describes the change.

## Unreleased

### 2026-10-17 - Engine A6: the turn-state counts in the expansion and the C API; the in-action move count in the evaluation identity

- Register A6 (Committed, class C; §2.3), the engine side - engine ABI round 1's first
  part. `engine/include/t5d/expand.hpp`, `src/expand.cpp`: item (h), the turn state -
  `mandatory_remaining()` (the boards the acting player must still play: the state
  machine's own count, T5, T6; Q5, Q6), `optional_playable()` (its other playable boards,
  L5; none once no move may be made, Q4) and `moves_made()` (the moves made in the action
  in progress). No state records that count - one `GameState` can be a turn start, or one
  or two moves into an action, by different paths - so `expand` takes it from its caller
  (`expand(state, options, out, moves_made)`; the three-argument form is a turn start's).
  Eager, O(timelines), allocation-free. This commit changes none of the truth's sources
  (`T5D_TRUTH_SOURCES`: the state machine, movegen, check, legal ... untouched). Commit 1
  changed two of them, `prover.hpp` and `prover.cpp` (the line the prover reports, only:
  no verdict, distance or node count changes), so after this component the truth's code
  signature differs from main's - as main's already differs from M17's (d96d6d9; 9be6f04
  typed the prover's hash constants). The main loop runs the Lazy SMP timing test before
  merging.
- `engine/include/t5d/capi.h`, `src/capi.cpp`, `src/capi_internal.hpp`: `t5d_expansion`
  gains `mandatory_remaining`, `optional_playable`, `moves_made` after D8's three fields
  (appended: every older offset kept; `t5d_layout` reports them); a `t5d_game` counts the
  moves of its action - 0 at every set-up (a turn start, T5), +1 per `t5d_game_apply` or
  `_apply_expanded`, 0 after `t5d_game_submit` or `t5d_game_replay_next`, carried by
  `t5d_game_copy` - reported as `t5d_game_summary.moves_made` (appended after `rules`)
  and passed to every `t5d_game_expand`.
- The count in the evaluation identity (issue 302, the main loop's reading, and the plan
  critic's settled one): the network will see it, so equal states with different counts
  are different inputs. `engine/include/t5d/mcts.hpp`, `src/mcts.cpp`: a node carries the
  moves made in its action (0 below a submit edge or a whole action, the parent's + 1
  below a move - the turn-start flag is its being 0); MCGS identity is the state key with
  the count, which subsumes the turn-in-progress bit (every ordering of one action's moves
  from one turn start has the same count, so MCGS keeps its merges inside an action);
  `EvalRequest::moves_made` carries it with the leaf's expansion; the evaluation cache's
  key (`EvalCache::find`/`insert`) and the batch's de-duplication match it beside the
  state and the epoch, so §15.2's purity rows hold (the comment that called the turn
  start "a path property the network must not see" is gone); `check_invariants` holds
  the count's turn structure (§15.4); the digest hashes it. `Search(root, params,
  moves_made, cache)` and `advance_to(target, moves_made)` take the count - a `bool`
  there (the old turn-start flag) is a compile error, not a silent 0 or 1 - and refuse
  a count of 0 where the root's obligations are met (a turn start owes a board, T5);
  `root_moves_made()`. `src/capi_search.cpp`, `include/t5d/capi_search.h`:
  `t5d_search_new` and `_new_sharing` take the game handle's count and refuse a
  `turn_start` that contradicts it; `t5d_search_root_game` hands the root's count back (a
  game handle with it); a leaf's count is `t5d_leaf.expansion.moves_made`
  (`t5d_search_leaf_state` returns a `t5d_state`, which holds no count). `src/selfplay.cpp`
  carries the count.
- Python mirror (`python/t5d/engine`): `capi.py`'s `CExpansion` and `GameSummary` (the
  layout check passes); `Expansion.mandatory_remaining`, `optional_playable`,
  `moves_made`; `GameInfo.moves_made` and `turn_start`; `Search`'s `turn_start` defaults
  to the handle's (given, the engine checks it); `netplay.py` no longer tracks it.
  Not here (the net lane's A6 inputs): `PositionState`'s counts, their embedding, the
  converter's copy, and `net/cache.state_digest` (an epoch's anchor), which takes the
  count when the network's input carries it.
- Tests: `engine/tests/test_turn_state.cpp` (new; §15.2's A6 row) - the three counts
  against a recount written from the rulebook at every state of random play under every
  ruleset, both Q4 readings among them, through `expand` and through a `t5d_game` playing
  the same moves; A6's arithmetic checked move by move wherever no activity changes, the
  rest counted (T6's arrivals, T11/T12's reactivations raising the count); the spec's
  cases by construction (a jump between two present boards, an optional board onto a
  mandatory one, a rewinding branch releasing an unplayed board, moves after the shift
  under R8 and the Guide-literal reading). `test_expand_reference.cpp`: (h) against the
  reference. `test_mcts.cpp`: EQ-06's
  `mcts_the_evaluation_cache_keeps_a_turn_start_apart_from_the_same_state_during_an_action`,
  equal states with 2, 1 and 0 moves made kept apart by MCGS while one action's orders
  merge, a batch asking once per (state, count), every node's count from its path, the
  root's count refused where it cannot be - each failing on the matching mutant (the
  cache key, the de-duplication or MCGS identity without the count). `test_capi_game.cpp`,
  `test_capi_search.cpp`: the handle's count and the search's; the layouts. Python:
  `test_binding.py` (the layout, the counts through apply, copy, submit and replay),
  `test_search_api.py` (the root's count). Engine 554 of 554 tests (`--werror`) - after a
  first full run of the final tree that ended with exit 1 and no summary line, inside
  `scanner_lazy_smp_store_refuses_contradictions` (a test this component does not touch:
  a crash in the shared stores on Windows, reproduced and traced at review - below); the
  long tests it touches 7 of 7 (the recount over 77,926 states); network 817 passed, 2
  skipped (`T5D_REQUIRE_ENGINE=1`); ruff clean; `verify` ok.
- `engine/README.md`: expansion (h), the game handle's count, MCGS identity and the
  evaluation cache with the count, the step API's `turn_start`, the readings.
  `docs/BUILD_LOG.md`: register A6 (engine side built, tested), C11, C14; spec issues
  143-145 (A6's arithmetic is not the rules' count; the two undefined counts read; §15.4's
  turn structure and the count); proposed edits 116-118.
- Review fixes, the same day (after 7bccea6). The C API's root count was tested only at
  0 and 1, which the old flag gives too: a search reading the handle's count as the flag
  passed every engine and Python test. `capi_search_takes_the_moves_made_from_the_game_a6`
  and `test_the_root_takes_a_count_the_turn_start_flag_cannot_give_a6` now build a search,
  and one sharing its cache, two moves into a three-board action: the root counts 2 (its
  game handle and its own leaf), a move below it 3. `advance_moves`' exact count (the
  root's + 1) and the tree it carries were untested - with a count of 1 there, a
  pondered subtree was dropped and the intermediate root miscounted, with no test
  failing; `mcts_advancing_a_move_at_a_time_carries_the_tree_at_each_count_a6` advances
  move by move (counts 1 and 2, then the submission's 0) and replays a two-move reply
  along the pondered line by `advance_action`, under tree and graph search, each new root
  the node the tree held, with its visits. Each fails on its mutant (the count read as the
  flag: 5 failed checks in C++, and the Python twin; `advance_moves` giving 1: 12). The
  docs that still said a batch asks one evaluation per distinct state and evaluations are
  cached per (state, epoch) - `capi_search.h` (`t5d_search_gather`, `_set_epoch`,
  `t5d_leaf.key`, `_leaf_state`), `mcts.hpp` (`EvalRequest::key`, `EvalCache`, `gather`,
  `epoch`, `batch_duplicates`), `mcts.cpp`, the README, `evaluator.py` and `search.py`'s
  `LeafView.key` - now say (state, moves made in its action, epoch), and that neither the
  key nor a leaf's `t5d_state` identifies its input; the README's list of game functions
  is whole again, and its summary of the search's tests names A6's. This entry no longer
  says `_leaf_state` hands the count back, or that M17's configuration keeps commit 1's
  signature (above), and commit 1's entry gives the true reason its line stays
  incomplete beyond depth 1, and §11.3 for Lazy SMP. The abort
  above, reproduced: `scanner_lazy_smp_store_refuses_contradictions` segfaults in 11 of
  600 runs here, in `ConcurrentSolvedStore::put` (gdb) - with MSYS2 ucrt64 g++ 16.2 a
  `std::shared_mutex` that several threads take for the first time at once does not
  always exclude a writer (a standalone test; none once each lock has been taken once,
  and none with `std::mutex`), and taking each shard's lock once at construction (an
  experiment, reverted) gave 0 in 600. Not fixed here: the scanner's and the prover's
  stores are among the truth's sources, M17 was measured on this toolchain, and Guard and
  Hunt's stores share the hazard - logged for the main loop (BUILD_LOG: H17's row,
  "Parked, and why"). Proposed edit 119: §11.3's full key, §15.2's purity row, §7.9 and
  §8.5, which state the identity edit 200 leaves out. Checks after the fixes: engine 555
  of 555 (`--werror`); the five-thread Guard tests 25 runs in a row (5 through
  `build.py`, 20 direct under the Python suite's load); network 818 passed, 2 skipped
  (`T5D_REQUIRE_ENGINE=1`); ruff clean; `verify` ok.
- Spec: §2.3 and register A6 (edits 116, 117; issues 143, 144); §8.5, §8.7 and register
  C11, C14, H18 - the count in identity and the cache's key, the main loop's proposed
  edit 200 (issue 302); §11.3's full key, §15.2's purity row (C19-C21: as built, the same
  state, count and epoch give the same evaluation), §7.9 and §8.5's purity sentence (edit
  119); §15.2's A6 row (built); §15.4's turn structure (edit 118, issue 145); §11.3's Lazy
  SMP (H17: the shared stores' lock on Windows, logged, no edit).

### 2026-10-17 - Engine A6, first commit: Guard's five-thread test against its contract; the prover reports the line of a thread that searched

- The cause of the intermittent failure of `guard_at_five_threads_depth_one_is_the_opponents_mate_in_one`
  (CI run 37146667584, g++-13: `line_complete` and the line's mate both false). Under
  Lazy SMP the first thread to prove the root exports the proof at once (H19), and a
  thread that builds its root after that - thread 0 most often, which builds its root
  after starting the four helpers, so on a loaded two-vCPU runner - reads the proof from
  the store and decides the root with no tree beneath it. The prover reported the
  lowest-numbered thread at the least distance: that thread, its line empty and its
  first action empty too. Made certain in a new test (a solved store that holds thread
  0's first lookup until the root's proof is exported): the old prover failed it on
  every run.
- `engine/src/prover.cpp` (`Prover::Impl::report_proof`): among the threads that proved
  the root at the least distance, those that searched their root come first, the first
  complete line is reported, else the longest; the first action is the reporting
  thread's (a root read from the store: the line's, from the certificate, if any).
  Sound and cheap: every thread's tree holds only exact results, a line is a walk of at
  most a few hundred nodes per thread, and the verdict and the distance are chosen as
  before - what Guard and Hunt decide is unchanged, and one thread reports exactly what
  it did. Which of the two remedies, and why: the line is now complete wherever some
  thread's tree holds it - always at depth 1, whose stores start empty (the thread that
  proved the root first searched it, and its line ends at the mated defender). Beyond
  that the documented contract stays: a line can still stop where every tree reached a
  node another thread (or an earlier depth) solved. A certificate does not complete it:
  it holds the attacker's proving actions only, so the line takes the attacker's step
  past such a node from it (`proof_line` does, when one is recorded) and stops at the
  defender's turn - continuing needs the defender's longest defence, which only an
  enumeration of the defender's actions finds (not cheap). So the tests check the
  contract. (Corrected at review: this entry and e554da0's message first gave as the
  reason that Guard records no certificate, 6b″ having measured it without one - but
  6b″'s tool ran Guard with `record_certificate` on, for its cross-check against E4
  (M17's "every refutation confirmed by E4"); only Guard's default records none.)
- `engine/include/t5d/prover.hpp`, `guard.hpp`, `engine/README.md` ("The proof's line"):
  `ProveResult::line`, `first_action` and `GuardResult::line` say whose line is reported
  and when it is complete.
- `engine/tests/test_guard.cpp`: the race pinned
  (`guard_at_five_threads_a_thread_late_to_its_root_reports_the_line_of_one_that_searched`:
  Guard's literal class, the default class and prove-only on three mates in one, the
  twin's included); the depth-1 properties at one and five threads keep `line_complete`
  (guaranteed now) and check the line's length; the depth-2 properties check every
  refutation's line against its contract - a prefix that replays, complete at depth 1,
  a complete line the opponent's mate at the refutation's depth by E4 - beside the
  references' confirmation of every refutation.
- `docs/BUILD_LOG.md`: register H42's row. The prover is among the truth's sources
  (`T5D_TRUTH_SOURCES`), so the code signature changes; the committed truth's
  provenance records none ("unrecorded"), and no verdict, distance or node count
  changes. The main loop runs the Lazy SMP timing test before merging.
- Spec: §11.4 (Guard; register H42), §11.1's df-pn, and §11.3's Lazy SMP (H17) and
  immediate export (H19) - no part of the spec describes the proof's line or which
  thread reports it; no edit proposed.
### 2026-10-17 - Build log: the CHECKPOINT 8 rulings and where each is built

- `docs/BUILD_LOG.md`, "CHECKPOINT 8": the user's rulings on spec issues 122-126, 142
  and 204 and the lane that builds each - H51 (the reader lenient on capture markers,
  the writer strict, a rulebook entry, KD-0167-0169 retired and ftxi's refusal
  recorded) as the gate lane's step 3a; the two-oracle rule, the commit tier's
  baseline rule, restricted runs and the ftxi upstream question as its step 3b; M22's
  further figures and the D2 order (the tree form, then the flat form) on the engine
  lane; Mish in the heads on the network lane. Issues 122-124, 142 and 204 marked
  ruled; proposed edit 98 settled upstream; "Waiting on the user" now holds only issue
  205 and the standing items.
- Spec: register H27, H47, H51, F6, M22 and D2 as changed at CHECKPOINT 8; no spec
  text edited here.

### 2026-10-17 - CLAUDE.md and AGENTS.md: the engine build before the gate (the one edit the user allowed in each)

- `CLAUDE.md`, `AGENTS.md`: one line in each command list, before `gate --tier
  commit`: `py engine/build.py` - the gate now runs the engine's adapter,
  `build/engine/t5d-gate`, so it must be built first. Nothing else in either file
  changed, as the user's CHECKPOINT 8 permission says; the diff is quoted in
  `docs/BUILD_LOG.md` ("CHECKPOINT 8").
- Spec: none - the instruction files; §11.0's gate is what the line serves.

### 2026-10-17 - Design documents: the CHECKPOINT 8 rulings, from upstream

- `docs/design/5d-chess-decisions.md`, `docs/design/5d-chess-transformer-design.md`:
  the copies maintained upstream after CHECKPOINT 8, committed unedited - the user's
  rulings on spec issues 122-126, 142 and 204. Register: H27 (upstream reports -
  defects as bug reports, documented-rule disagreements as questions citing the
  source, drafted by the agent and filed by the owner), H47 (gating aligned to two
  oracles; the commit tier still fails if the engine declines a record it read on the
  last green run, or reads none; restricted runs never produce a tier verdict), H51
  added (capture markers read leniently - a project decision: a move is read as its
  squares define it, a mismatch logged as a warning, the marker written strictly; a
  source that refuses such a record gets a known-divergence entry citing H51), F6
  (Mish in the trunk FFN and every head's hidden layers; squared ReLU a trunk-only
  arm), M22 (measured in part; the mandatory, three-or-more, optional-move and
  compute-share figures to follow), D2 (build order: the tree form, then the flat
  form). Spec: §8.2 (the measured turn width and the build order), §9.3 (Mish in the
  heads), §11.0 (the commit tier's two failure conditions, restricted runs, upstream
  reports, lenient capture markers).
- Spec: CHECKPOINT 8 as written upstream - §8.2, §9.3, §11.0; register D2, F6, H27,
  H47, H51, M22. Nothing in them edited here.

### 2026-10-17 - N-B15 review fixes: B6 is not B15's special case as built, recorded and put to the user; the placeholder switches checked to regroup pairs; every variant's board size in §15.2's B15 row

- `docs/BUILD_LOG.md`, spec issue 205: under B15's far-bin placeholder B6 is not B15's
  special case, which §4.1 ("of which it is the special case that ignores file and rank"),
  register B6 and reversal 25 say it is - and nothing recorded it. B6 was two per-axis tables
  (signed Δtime log buckets to 128 half-turns, |Δtimeline| to 32); the built far classes - one
  distance D in three bins, crossed with Δtime's sign - mix its cells, so beyond the exact box
  they hold a B6 bias only where it is a constant plus a function of Δtime's sign (a linear
  solve over the built classes, in a scratch script), and the board-axis line clamp merges
  its buckets too.
  Lost: B6's per-axis form and its far resolution; kept: Δtime's sign and the clamp. Far
  classes that refine B6's cells would cost 5,240,960 parameters, over three times §4.1's
  ~1.6M, so "special case" and "~1.6M" cannot both hold with a slice per class; B6's two
  tables added beyond the box (3,456 parameters) would keep both, at the cost of the
  one-scalar-per-pair reading. B6 went with no arm (M20), so ablation 2 cannot test what it
  resolved. Put to the user. Proposed edit 159 now reaches §4.1's B6 sentence, register B6's
  status text and reversal 25; edit 155 gives the alternatives. The register rows B6 and B15,
  `config.py` (`disp_far_edges`) and `displacement.py` say it beside the placeholder.
- Issues 205 and 206: the far bins' form (one distance, Δtime's sign, Δtimeline's sign
  dropped) and the line set's time unit and partition are fixed in code, and say so - only
  `disp_far_edges` and `disp_line_max` are settings, so another form is a code change (the
  geometry, both implementations, the tests' reference, the pinned counts). They are not made
  switches: each alternative is a different class geometry, the additive line reading would
  need a second lookup per pair on every path, and the documents lean to the readings built
  (§4.1's "one learned scalar ... indexed by the pair's displacement", B17's "~12,000
  classes"). `config.py`'s and `displacement.py`'s docstrings say the same.
- `tests/net/test_structure.py`: the no-inert-switch test also checks that each of B15's
  placeholder switches regroups the batch's pairs (which pairs share a scalar). Two of them,
  `disp_line_max` and `disp_register_pairs`, change the table's size, which is drawn from one
  stream, so the outputs differed whatever the switch did to the indexing: on scratch copies,
  a line clamp hard-coded at 3 and the register kind dropped from the index each passed the
  old test and fail the new one. `test_same_grouping` pins the helper.
- `tests/net/test_displacement.py`: the constructions of §15.2's B15 row ("every variant")
  now hold every variant's board size (rulebook V3), where they held 8 x 8 and 5 x 5 only:
  7 x 7, 6 x 6, 4 x 4 and 3 x 3 boards (`variant_sizes`), and the far lattice at 7 x 7, 4 x 4,
  3 x 3 and Global Warming's empty 1 x 1 (`far_variant_sizes`; `far_state` builds the 1 x 1
  by hand, which `synthetic_state` cannot). The classes and the score function's term match
  the reference bit for bit and attention through it is within 1e-5; the CPU paths' outputs
  and gradients are checked on the far lattice too.
- `config.py`: `attn_impl`'s docstring names the sixth path, "flex" (forward only on CPU -
  it refuses under autograd - small inputs), and says "all six", not "all five"; the
  L24-ff12 preset's "~44M" is marked as §9.2's count without B15's table (46,891,210 with
  it, issue 210).
- `bench.py`, `attention.py`: the "every switch" figures were measured with B6's binned
  biases on (since removed) and no B15, so they are not today's "n2-all"'s either.
- `docs/BUILD_LOG.md`: proposed edit 158 names §10.2, which its second quote is from, and
  quotes each place's current text; §15.3's "Without a GPU" already defers the kernel and
  needs no edit. Issue 211 and the B15 row: twelve constructions. The B15 entry below says
  where these fixes changed it.
- Spec: §4.1 (the B6 paragraph's "special case"; B15's far bins and line classes), §10.2 (the
  score function), §15.2 (the B15 row's constructions, "every variant"), §9.2 (F2's count, as
  issue 210 has it); register B6, B14, B15, F2, G9, M20; reversal 25. Proposed edits 155 (the
  alternatives if B6 is to stay a special case), 158 (§10.2 named, the current texts quoted),
  159 (extended to §4.1's B6 sentence, register B6's status text and reversal 25). No
  parameter changes (the v1 stays 22,961,802).

### 2026-10-17 - Network B15: the 4D displacement bias replacing B6, with its no-table flag; the score-function path and its equivalence test

- `python/t5d/net/displacement.py` (new): register B15 (provisional, the v1 default; §4.1,
  CHECKPOINT 6), written from §4.1 and the rulebook's conventions. One learned scalar per
  head per layer, indexed by the pair's displacement (key minus query): a 15 x 15 file-rank
  slice for each board-distance class - the 45 exact ones (|Δtimeline| ≤ 2, |Δtime| ≤ 4
  half-turns) and 9 far bins - exact line classes beyond them (one entry per direction and
  step, 368), register entries by class and pair kind (162) and one global entry: 12,681
  per head per layer, 1,623,168 parameters (§4.1's ~1.6M, 7.07% of the network). The
  unstated details are named placeholders, each a spec issue: the far bins
  (`disp_far_edges`, the board distance D = max(|Δtimeline|, ⌈|Δtime|/2⌉) log2-binned from
  3 and crossed with the sign of Δtime), the line set (in whole turns, so an odd Δtime is
  never a line; a partition; board-axis lines clamped at `disp_line_max = 8`), the register
  and global entries (`disp_register_pairs`), the initialisation (`disp_init_std`). Two
  implementations: `pair_entry` computes a pair's entry by integer arithmetic from the
  tokens' kinds and coordinates (B14's score function uses it), and the CPU paths look it up
  in a static class map by board-pair tile and token code. Boards wider than 8 are refused.
- `config.py`: `displacement_bias = True`; `False` is ablation 2's no-table arm. B15 draws
  its table from a generator of its own (`model.b15_seed`, as D8's parameters do), so the
  arm is the network without the table: checked against the parent commit, the arm's
  outputs and parameters are bit for bit the pre-B15 v1's (B9's default, routing with typed
  edges and smolgen, no masks, the chunked path, post-norm, and a cached evaluation).
- Per layer (`structure.py`, `attention.py`, `trunk.py`, `cache.py`): `AttnContext.for_layer(h,
  layer)` (the layer index now required; each `trunk.Block` knows its own) gives the
  structure B15's table of that layer and, as before, B12's table from `h`. Masks need no
  layer (`AttnStructure.mask_rows`, `attention._block_mask`). The cached path builds a
  frozen board's bias per layer over its key set (it built one for every layer before), and
  the live pass makes each layer's context; B15 biases frozen rows too (geometry -
  `scope.py`). `model.py`: `self.displacement`, `parameter_breakdown`'s
  `attn.displacement`.
- `python/t5d/net/flex.py` (new): B14's score function (committed engineering; §6.1, §10.2),
  `attn_impl = "flex"`: torch's `flex_attention` with every bias computed inside its score
  function - B15's entry from the two tokens' coordinates, B11's typed relations, B12's
  board-pair table, a caller's bias - every mask in its mask function (padding, the
  block-causal scope, section 6's roles from per-token data and one edge bit per pair), the
  positional pair rule by feature widening (at most three calls), and a lossless block mask
  built from the board-major order at board granularity. Eager on CPU: forward only, every
  score materialised - small positions (spec issue 211). Under v1's register rule the masked
  heads' block masks have no empty block (spec issue 209).
- B6 removed (superseded by B15, register B6, M20): `rope.py` loses `BinnedBias`, the
  near-time arm and the buckets; `binned_bias`, `binned_bias_rope`, `binned_bias_near_time`
  go (issue 9f's switch). The log buckets move to `buckets.py` for B12's relative cells,
  whose fields are renamed `smolgen_time_*` / `smolgen_timeline_*`: smolgen unchanged, 152
  cells and 47,136 parameters, bit for bit (spec issue 212). `bench.py`: the v1 default now
  carries B15, `n2-no-table` is ablation 2's arm (B9's default before CHECKPOINT 6).
- Tests: `tests/net/test_displacement.py` (new) - the classes against §4.1 written out pair
  by pair (a grid reaching every far bin and clamp, every token kind, ten adversarial
  constructions - twelve, every variant's board size among them, since N-B15's review fixes
  (above) - three placeholder settings; integers compared exactly); §15.2's B15 row -
  the score function's term at every pair equals the hand-materialised bias bit for bit, and
  attention through it equals attention with that bias materialised (measured under 1e-6,
  asserted 1e-5), three seeds, the table at its initial scale and at unit scale; the CPU
  paths' outputs and gradients (Q, K, V, the table) equal it too; the whole network through
  the score function equals the dense reference; the block mask is lossless and sparse
  where no register row reads everything; the table's size, its layers, the cached path
  with the layers' tables far apart, the no-table arm, translation and board order.
  Existing suites carry B15 from now on (the v1 default) and keep the no-bias paths covered
  through the no-table arm: `test_chunked` (four no-table combinations, two pair modes, the
  caller's mask and bias), `test_cache` (every case; plus no-table and moved placeholders),
  `test_structure` (B15 in every bias that adds up; four new switches - B15's placeholders
  also checked to regroup pairs since the review fixes; `flex` among the one function),
  `test_attention` (B5 with B15's bias, hand-materialised), `test_scope` (frozen
  rows keep B15), `test_invariance`, `test_d8`, `test_n2_model`, `test_trunk`, `test_smoke`,
  `tests/engine/test_network.py`. `test_params`: the v1 is 22,961,802 (21,338,634 and the
  table), F2's preset 46,891,210 (spec issue 210).
- Not measured: §15.3's "displacement bias and normalisation changes" row (the machine is
  loaded; the quiet-machine block). On CPU every head now carries a per-row bias, so v1's
  dense heads run block by block with it, as "every switch" did (`attention.py`,
  `bench.py`).
- `docs/BUILD_LOG.md`: register rows B15 (built, tested), B6 (superseded; removed), B14 and
  G9 (the score function), B10/B12, C2 (the cached path with B15), F1/F2, M20 (B6's removal
  done); spec issue 9's note (its (f) switch gone); spec issues 205-212; proposed edits
  155-159.
- Spec: §4.1 (the 4D displacement bias; B6's paragraph), §5.3 (register pairs), §6 and §6.1
  (B14's score function and block mask), §9.2 (parameter counts), §10.2 (the score
  function), §15.1 Phase 2, §15.2 (the B15 row; row 1, cached = dense, with B15 on; the
  C19-C21 and G1 rows under it), §15.3 (the displacement-bias row, to measure); register
  B4, B6, B8, B12, B14, B15, C2, C17, F1, F2, G9, M19, M20. Proposed edits 155 (B15's
  placeholders), 156 (F1/F2 counts; supersedes 153), 157 (§6.1's block mask and the
  register rows), 158 (the score function in the untrained build), 159 (B6 removed; B12's
  cells - since the review fixes also B6's "special case" clause in §4.1, register B6 and
  reversal 25, which the far bins as built make false: issue 205).

### 2026-10-17 - Build log: each lane's numbers render as written; issue 142 waits on the user

- `docs/BUILD_LOG.md`: an HTML comment at column 0 before the first spec issue of
  each number block (300, 122, 200) and the first proposed edit of each (200, 150),
  as the engine lane did for 140 and 110 - a Markdown ordered list renumbers its items
  from its first, so on GitHub issue 300 read as 122 and every later number with it.
  Issue 142 (merged with M22) joins "Waiting on the user".
- Spec: none.

### 2026-10-17 - M22: the turn-width measurement on t0database - mandatory boards per turn and optional moves played, per board-count range

- `engine/tools/turn_width.hpp`, `turn_width.cpp` (new, in `t5d_tools`; not among
  Hunt's truth sources): the turn width of one action, counted by the rules engine
  alone (`TurnCounter`, `classify_action`). At the action's start: the mandatory
  boards (the active boards in the present, T6 and R9), the playable and the optional
  boards. Then its moves: the optional ones (made on a board not mandatory at the
  start - the source, or the destination of a jump that does not branch: T3, J4),
  with the reading by the source board alone and the reading by the status when made
  beside it; the boards played (a jump that does not branch plays two); jumps between
  two mandatory boards (A6's "two fewer"), from an optional board onto a mandatory one
  and from a mandatory board onto an optional one; moves after the shift (R8);
  rewinds (T8, T12); and the boards a rewind releases or a reactivation (T11, T12)
  makes mandatory mid-action. And the summary's board-count ranges, mandatory classes
  and statistics (`kTurnRanges`, `kTurnBuckets`, `mandatory_class`, `turn_stats`).
  Why: register M22 must be recorded before the D2 tree and flat forms are built
  (§15.1; the ply split is built), and nothing in the engine looked inside an action.
- `engine/tools/t5d_measure.cpp`: `--turns` (a line per action) and `--turn-summary`
  (per board-count range at the action's start - < 50, 50-99, 100-199, 200+, ten-board
  buckets under 100, by mandatory class - under a strict reading that leaves out a
  record repeating another, M7's grouping as sensitivity, and the refused records'
  actions apart; a record that ends in "error" is in no reading, counted as
  `errors`). Every action is played twice, by the replay and move by move, and the
  two states must be equal. Without the new options the output is unchanged
  (byte-identical on t0database, with and without `--summary --legal`).
- `engine/tests/test_turnwidth.cpp` (18 tests: constructed actions with known
  counts; the summary's ranges and classes at their edges and its statistics on
  constructed widths; `t5d-measure` run on `engine/testdata/turn-width/` - the counts,
  how records are classified, and actions starting at 9/10, 49/50, 99/100 and
  199/200 boards); `engine/CMakeLists.txt`. 544 tests pass.
- `docs/MEASUREMENTS.md`: "Turn width (M22)", a new section for Phase 1 counts with
  no threshold. On t0database's 692 games under the default rules: more than one
  playable board at the turn's start - where the D2 arms differ - in 15.1% of turns
  under 50 boards, 65.2% at 50-99 and 99.4% at 100-199 (13 games; none at 200+); more
  than one mandatory board in 5.8 / 27.1 / 64.0%; an optional board at the start in
  10.5 / 44.5 / 65.7%, an optional move played in 1.2 / 2.5 / 1.1%; single-move turns
  95.9 / 80.3 / 49.1%. Part L's prediction ("most human-game turns under 50 boards
  have exactly one mandatory board") holds: 94.2%. No verdict.
- `engine/README.md`: t5d-measure's new options and lines; the tables.
- `docs/BUILD_LOG.md`: register M22 measured; step 5's M22 note; spec issues 140-142
  (what "optional moves played" means; the ranges and the unit; nothing maps the
  figures to an order of the D2 work - put to the user); proposed spec edits 110-115.
- Review fixes, the same day (after 2864be0). The D2 reading counted "where the arms
  differ" by the moves played (4.1 / 19.7 / 50.9% of turns of more than one move);
  it is now counted by the turn's start, as §8.2's cost argument and the built ply
  split count it (15.1 / 65.2 / 99.4%), in MEASUREMENTS, issue 142 and edit 110. An
  optional move went by its source board alone, so a jump from a mandatory board onto
  an optional one was not counted (35 turns of the 692 games) while its mirror was;
  it now counts - the move is made on both boards (T3, J4), and the measurement's
  brief defined optional moves as moves on boards not mandatory at the start - and
  both kinds are counted apart and tested (issue 140, edit 111). The summary's
  ranges, classes and statistics moved into `tools/turn_width.hpp` and are tested at
  their edges; new records test a truncated record and the strict prefix rule;
  fifteen mutations of the counting and the summary, the review's among them, now
  each fail a test. Also: parse mode's effect stated (under C4, t0db-0115 is refused
  at its 9th action - rules question Q8 - and the strict reading would hold 691
  games); edit 111 names both ways a board's status changes; register D2 and part
  K's turn-construction gate get the edit 6x gets (edit 115); the issue and edit
  lists start new Markdown lists, so that GitHub shows 140 and 110 as written; the
  `turn_summary_group` line and the `"error"` path documented; and the conditions
  name the commit the figures were counted at (607e4b1, rebuilt and re-run there:
  byte for byte), where the first version said only "this change", and say what the
  fixes changed in the run's output against 2864be0's (one count, fields added,
  nothing else).
- Spec: §8.2 "Measure first" and §15.1's Phase 1 row (the M4 write-back proposed:
  edits 110 and 114); §13.1 6x ("after the turn-width measurement", now met; edit
  114); register M22, part K's turn-width item and part L's prediction (edits
  111-113); register D2 and part K's §8.2 turn-construction item (edit 115); spec
  issues 140-142. Rules cited: T3-T8, T11, T12, J3-J6, R8, R9; Q8.
### 2026-10-17 - N-F5F6 review fixes: the post-norm arm's DeepNet initialisation as published; RMSNorm's eps tested

- `python/t5d/net/model.py`, `config.py`: under ablation 6y's post-norm arm the trunk's
  attention and FFN projections are drawn Xavier normal, std gain·√(2/(fan_in + fan_out)) -
  DeepNet's initialisation as the paper's pseudocode draws it (Wang et al. 2022, Figure 2:
  `xavier_normal_(w, gain=β)` for "ffn", "v_proj", "out_proj", gain 1 for "q_proj",
  "k_proj") - instead of β times the trunk's LeCun normal. The two coincide on the square
  attention projections; on the FFN's 256→2048 and 2048→256 matrices the LeCun base had
  drawn 2.12x and 0.75x the paper's std (the FFN branch 1.59x), which no document said. The
  task asked for the initialisation "as published"; the base is the pseudocode's example
  ("it can be replaced with other standard initialization"), so it is recorded as a
  placeholder (spec issue 201). The v1, F2's preset and every flag configuration draw bit
  for bit as before (checked tensor by tensor); under the arm only the FFN weights change,
  each the same draw rescaled.
- Tests (`tests/net/test_trunk.py`): the arm's Xavier stds pinned; RMSNorm's eps checked
  where it shows - on a small-RMS input, with no relative slack, eps outside the root or no
  eps is 0.95 off - and the unit-scale formula test's asserts given `rtol=0`;
  `NetConfig.norm_eps` reaches every trunk norm under both placements, stored and used. On
  scratch copies, 8 of 8 mutations are caught: eps outside the root or absent, a hard-coded
  1e-5 in the norm's forward or in any of the three norms' construction, a LeCun base under
  the arm, a scaled Q/K.
- The arm's docstring figures re-measured with the new initialisation (numerics, not
  timings): `test_precision`'s post-norm BF16 gap 0.06% RMS, max 0.8%, value logits 6.3e-4,
  policy 2.8e-3, value mean 2.1e-5 (1.5-7x below the v1's); `test_cache`'s 6y test, cached
  against the BF16 full pass, RMS 5.3e-4 / max 6.0e-3, value 1.9e-4, policy 1.9e-3 (2.2-6.4x
  below the v1's). Every bound unchanged.
- `attention.py`: its indicative timings and peak memory, copied from `bench.py`'s table,
  are marked as measured on the network before F5 and F6 and not re-measured, as
  `bench.py`'s are.
- `CHANGELOG.md`: the F5 and F6 entries' Spec lines cited "§15.2 (rows 1-3 ...)", but
  §15.2's second and third rows are the windowed and lazy-K/V rows deferred with the
  window (M18); the 6y test covers the cached = dense (C2, C17), supersession recompute
  (C20), same state + epoch / transposition (C19-C21) and FP8/BF16 cache precision (G1)
  rows, which the lines now name. The F5 entry's initialisation bullet and the F6 entry's
  post-norm ratio point here.
- `docs/BUILD_LOG.md`: spec issue 201 and proposed edit 151 state the Xavier base, issue 201
  as a placeholder with what the LeCun base had drawn.
- Spec: §9.3 (the second bullet: "post-norm with DeepNet initialisation"), §13.1 6y, §15.2
  (the four rows above, under the arm); register F5 (the arm), F6 (the citation only), G1.
  Proposed edit 151 amended (the arm's initialisation: Xavier normal at DeepNet's gains). No
  parameter changes (F1 stays 21,338,634).

### 2026-10-17 - Network F6: Mish in the FFN; squared ReLU behind a flag

- `python/t5d/net/trunk.py`, `config.py`: register F6 (committed, §9.3). The FFN's
  activation is `NetConfig.ffn_activation`: `"mish"` for v1 - `x tanh(softplus(x))` (Misra
  2019; PyTorch's `mish`), replacing stage N1's GELU - or ablation 6y's arm
  `"squared_relu"`, `relu(x)^2` (So et al. 2021, "Primer"), at Mish's initialisation (the
  arm's is not stated). Validated; GELU is no longer an FFN option.
- `heads.py`, `config.py`: whether Mish also covers the output heads is not decided by F6
  ("FFN activation"; its "Chessformer's FFN and output heads" describes Chessformer), so the
  value head's hidden activation - the only head MLP - stays GELU as a named placeholder,
  `head_activation` ("gelu", or "mish"; spec issue 204). `embed.py`: the Δt MLP keeps GELU,
  documented as outside F6.
- The cached path needs nothing new: it calls the block's own FFN (F5's block steps).
  §15.2's rows under the arm: `test_cache`'s 6y test runs both arms (cached = full pass,
  root refresh, transpositions, a transposition hit, the BF16 cache), and the every-switch
  matrix runs both arms together with the N2 structure.
- Tests: `test_trunk.py` - the FFN equal to its formula in FP64 under each activation (and
  not GELU); the value head's activation by hand under both placeholder values; validation.
  `test_structure`: "every switch changes the network" gains `ffn_activation`; every
  `attn_impl` one function under squared ReLU. `test_precision`: both 6y arms under G1.
  `test_smoke`: squared ReLU (with the heads' Mish) trains finite with every tensor moving.
- Precision bounds matched to the arm's arithmetic: squared ReLU squares its input's BF16
  rounding (twice the relative error), and its measured BF16 gap is up to twice the v1's
  (precision: 0.39% RMS against 0.24%; policy logits 0.021 against 0.010), so its bounds in
  `test_precision` and `test_cache`'s 6y test are the v1's doubled (2.9-20x and 3.6-9x
  above its measurements); the post-norm arm keeps the v1's (its gaps are 1.5-7x smaller, as
  re-measured after N-F5F6's review fixes; 1.6-6.5x before them).
  `test_fp8_cache_equals_the_fp8_network_at_fp32_resolution`: the trunk-RMS bound raised
  from 2e-4 to 4e-4 - FP8 rounding flips are discrete and heavy-tailed (over 12 seeds of the
  F5 and F6 networks each, a seed's largest was 1e-6 to 3e-5 typically and 7.8e-5 at most,
  2.6x under the old bound); the new bound sits 5x above that and 19x below mutant M14.
- Docstring figures re-measured after F6 (numerics, not timings): `test_precision`,
  `test_cache` (its header's claim that the 16-layer gaps are no larger than the 2-layer
  ones, written with F5, was wrong - they are up to 1.8x larger; corrected),
  `tests/engine/test_network.py`. `bench.py`: a `6y-squared-relu` configuration; its table
  marked as measured before F5 and F6.
- `docs/BUILD_LOG.md`: register row F6 (new); spec issue 204 (put to the user: the heads'
  activation awaits neither a ruling nor a named ablation); proposed edit 154.
- Spec: §9.3 (the Mish bullet), §13.1 6y, §15.1 Phase 2 ("Mish" among CHECKPOINT 7's
  items), §15.2 (the cached = dense (C2, C17), supersession recompute (C20), same state +
  epoch / transposition (C19-C21) and FP8/BF16 cache precision (G1) rows, under the arm);
  register F6, D3 (the heads' activation). No parameter changes (F1 stays 21,338,634).
  Proposed edit 154 (whether the output heads' activation is Mish; the squared-ReLU arm's
  initialisation).

### 2026-10-17 - Network F5: RMSNorm-style normalisation, no Q/K/V biases, pre-norm; the post-norm + DeepNet arm behind a flag

- `python/t5d/net/trunk.py`: register F5 (committed, §9.3). `RMSNorm` - a learned gain over
  the root mean square, no mean-centring, no bias (Zhang & Sennrich 2019), computed in FP32
  with autocast off (G1) - replaces LayerNorm in both sublayer norms of every block and the
  final norm. A block is now four steps (`attn_input`, the attention, `after_attn`,
  `after_ffn`) and the trunk ends in `Trunk.output`; they carry the placement,
  `NetConfig.norm_placement`: `"pre"` (the v1, as before) or ablation 6y's arm `"post"` -
  DeepNorm as published (Wang et al. 2022, Figure 2): x ← N(αx + sublayer(x)), α = (2L)^(1/4),
  no final norm, no norm on the embedding, smolgen reading the block's input (placeholders,
  spec issue 201).
- `attention.py`, `trunk.py`, `config.py`: the single `linear_bias` switch is split as §9.3
  says - `qkv_bias=False` (F5: no Q/K/V projection biases) and `out_ffn_bias=True` (the
  attention-output and FFN biases kept: "as here"; spec issue 200). `NetConfig(linear_bias=...)`
  now raises `TypeError`. `norm_placement` and `norm_eps` validated; `deepnorm_alpha` and
  `deepnorm_beta` derived from L. The heads, the embedding's Δt MLP and smolgen keep their
  biases; smolgen has no norm of its own to keep (spec issue 202).
- `model.py`: RMSNorm gains drawn 1 + N(0, `init_norm_std`), no bias draws; under post-norm
  DeepNet's initialisation replaces the GPT-2 1/√(2L): gain β = (8L)^(-1/4) on the value,
  attention-output and both FFN projections, Q and K at gain 1 - over the LeCun base as
  committed here, drawn Xavier normal as the paper's pseudocode since N-F5F6's review
  fixes (above). Every seed's global RNG stream shifts (fewer draws); no test pins a
  golden output.
- `cache.py`: `compute_board` and `live_pass` run the trunk's block steps instead of a
  hand-written pre-norm block, so the cached path computes the trunk's function under
  either placement (§15.2's rows hold under the arm).
- Parameters: F1 21,338,634 (−20,736: the norms' biases 33 × 256, the Q/K/V biases 16 × 3 ×
  256); F2's L24-ff12 preset 44,456,458, now pinned exactly; the post-norm arm 256 fewer.
- Tests: `tests/net/test_trunk.py` (new) - RMSNorm by its formula in FP64 and against
  PyTorch's `rms_norm`, no centring, FP32 under autocast; norms and biases where §9.3 puts
  them; the bias switches; each placement equal to its formula block by block (smolgen on
  and off), α applied; DeepNet's initialisation std ratios; the arm's count; validation.
  `test_cache`: the 6y arm through §15.2's rows (cached = full pass on two geometries, root
  refresh, transpositions bitwise, a transposition hit, the BF16 cache as the live
  precision), and the every-switch matrix gains post-norm with the N2 structure;
  `test_structure`: every `attn_impl` one function under post-norm, and "every switch
  changes the network" gains `norm_placement`, `qkv_bias`, `out_ffn_bias`; `test_n2_model`:
  smolgen reads each layer's attention input under both placements; `test_precision`: the
  post-norm arm under G1's bounds; `test_model`: the RMSNorm gains non-identity, post-norm
  too; `test_smoke`: the arm trains finite with every tensor moving; `test_params`: the new
  counts (its stale smolgen figure corrected to 47,136).
- Docstring figures re-measured on the new network (numerics, not timings; the machine is
  loaded but these do not depend on it): `test_precision` (BF16 gap 0.23% RMS, bounds 3-11x
  above), `test_cache`'s BF16 and FP8 gaps and mutant M14, `tests/engine/test_network.py`'s
  real-position gaps - each of the same size as its pre-F5 figure (within a third), except
  FP8-at-FP32's rare rounding flips (trunk RMS 6.1e-5 at seed 0, the largest of 12 seeds;
  under its 2e-4 bound and 120x below mutant M14). `bench.py`: a `6y-post-norm`
  configuration; its timing table marked as measured before F5 (not re-measured).
- `docs/BUILD_LOG.md`: register rows F5 (new) and F1/F2 (the count); spec issues 200-203;
  proposed edits 150-153.
- Spec: §9.2 (Params), §9.3 (all three bullets' F5 parts), §13.1 6y, §14.4 (KataGo's
  fixup, replaced by §9.3's RMSNorm-style layers - nothing built), §15.2 (the cached = dense
  (C2, C17), supersession recompute (C20), same state + epoch / transposition (C19-C21) and
  FP8/BF16 cache precision (G1) rows, under the arm), §15.3's "displacement bias and
  normalisation changes" row (not measured: no quiet machine); register F1, F2, F5, B12,
  G1. Proposed edits 150 (F5's biases stated in §9.3's first bullet and the register), 151
  (the post-norm arm's details as built), 152 (B12's "keeps its own normalisation"), 153
  (§9.2 and F1's count, M4).
### 2026-10-17 - Gate H47: the some-ruleset-is-gated check only in tiers that run every oracle

- `oracles/oracle_harness/gate.py`: the check that some ruleset is gated -
  `engine-nothing-gated`, raised when nothing else was found against a full-mode
  engine but it came through no ruleset, and `GateResult.ok`'s test of it - runs
  only in a tier that runs every oracle search over the whole corpus
  (`Tier.full_strength`: nightly, release), keyed to what a tier runs, never its
  name. Any other tier - commit, ftxi and the notation parser - is a smoke test
  (H47, Q-A): an engine with no finding against it passes its differential, gated
  or not. A smoke pass never reads as gating: the summary says `engine
  differential: SMOKE TEST - PASSED` (or `FAILED`) and that the tier gates nothing,
  and why (`SMOKE_TEST`); every ruleset is listed as `NOT gated`, one that met the
  gating conditions with the tier's oracles alone saying so (`SMOKE_REASON`); the
  report has no ruleset under `gated` and the reason under `smoke_test`. In every
  tier an unexplained divergence (one `known-divergences.toml` does not record), a
  telltale or a breach of the engine contract fails the engine. New:
  `GateResult.smoke_test` and `GateResult.verdicts` (what a run may state);
  `GateResult.variants` reads only what the run may state as gated, so a tier that
  requires an engine can never pass on rulesets its own summary prints as `NOT
  gated`; and when such a tier fails because the engine came through no named
  variant (custom positions at most) - which no finding says - the summary says so.
  A run restricted with `--source` or `--record` keeps its tier's rule (spec issue
  124). Unchanged: the rule in nightly and release, the release tier's requirement
  of a named variant, the proof-channel line, parse mode and its floor, and the
  differential itself (`GateResult.gated` still holds what met the conditions, in
  every tier). Why: with ftxi alone the commit tier gated no ruleset once the
  engine was configured (2026-10-04, Q-A), so the rule would have turned the commit
  workflow red; the ruling (CHECKPOINT 6) took option (a).
- `oracles/oracle_harness/config.py`: `gate.toml` is refused if a tier that does
  not run every oracle source over every record tier sets `require_engine` - such
  a tier gates nothing, so no engine could ever pass it (it would fail with nothing
  to say why).
- `oracles/tests/test_gate.py`, `oracles/tests/test_gate_live.py`,
  `oracles/tests/test_config.py`: both sides pinned - in a commit-shaped tier an
  agreeing engine that gates nothing passes (ok, the proof channel disabled, the
  output saying the tier gates nothing); in nightly- and release-shaped tiers it
  fails on `engine-nothing-gated`; an unexplained divergence fails every tier; an
  engine that declines every record passes only the smoke test; the report's
  `smoke_test` and `gated`; a real-ftxi commit run never prints a ruleset as gated.
  And where a tier's name and shape disagree - a "nightly" that runs less, a
  "commit" that runs everything, a custom-named tier that runs every oracle search
  over the whole corpus, nightly cut to the commit records - it goes by the shape
  (a rule keyed to the name now fails these tests); a restricted run of nightly
  keeps its rule;
  a tier that requires an engine reads only what its run may state as gated, and
  says why it failed; `require_engine` on a tier that is not full strength is
  refused. Tests that read a commit-shaped result as gating now use a nightly
  shape.
- `oracles/README.md` ("Tiers", "The engine contract", "Other commands"),
  `oracles/gate.toml` (comments only), `CONTRIBUTING.md` ("The proof channel") and
  `.github/workflows/oracle-nightly.yml` (its header comment): the commit tier is a
  smoke test that gates nothing, because it does not run every oracle search over
  the whole corpus - H47's own reason quoted beside the gating conditions as built;
  what fails the engine in every tier (a divergence on record does not); a
  restricted run keeps its tier's rule; `require_engine` only on a full-strength
  tier; the release tier's rule named as a named variant.
- `docs/BUILD_LOG.md`: register rows H47 (built, tested) and H48 (not started);
  spec issues 122-124, each put to the user; proposed spec edit 98, pending the
  ruling on issue 122, with the text for each answer.
- Spec: §11.0 "Tiered CI" and register H47 - proposed edit 98, pending the user's
  ruling: the spec's reason ("a ruleset is gated only when two or more oracles
  answer for it") is not the gating rule as built (spec issue 122, put to the
  user); H47 is silent on an engine compared with nothing in a smoke test (spec
  issue 123, put to the user) and on a restricted run of a full-strength tier
  (spec issue 124, put to the user). H23 and the proof channel unchanged.
### 2026-10-17 - Build log: the CHECKPOINT 6-7 rulings, the plan by lane, the register brought up to date

- `docs/BUILD_LOG.md`, "Where things stand": a section "CHECKPOINTS 6 and 7 - ruled
  upstream (design documents f07568c)" - what the rulings decide, by register ID
  with the register's status words; how each question of 2026-10-04's "Waiting on
  the user" is answered (Q-A -> H47, Q-B -> H48, the GPU question -> M19, issue
  121 -> M20 except D8's readout form, A5 -> behind its flag, D8's whole-turn token ->
  the flat form (D2), the training rows -> M21 and I12-I20); what changes for the
  build. Then "Current state (2026-10-17)": what is on main; where each lane stands,
  by commit, as of 16:54 (gate: H47 with its review's fixes, 7530e36, issues 122-124
  put to the user, edit 98 pending the ruling on 122, the gate with the engine
  configured in progress; engine: M22 measured, 2864be0, issue 142 put to the user;
  net: F5 and F6 built, f830f31, issue 204 put to the user - none merged); how to
  recreate a lane's worktree, and how the lanes share the one machine; the plan -
  three lanes in worktrees (the main loop's plan under M23's rule), the number blocks
  (spec issues: gate 122-139, engine 140-199, net 200-299, main loop 300 and up;
  proposed edits: gate 98-109, engine 110-149, net 150-199, main loop 200 and up),
  step 1 by lane in order (the gate lane's two steps named "gate step 1/2", not G1/G2,
  which are register IDs; gate step 2 done only when nightly is green with the engine
  configured; C15's STE fix before G2's quantiser and G8's hook; M21 building on the
  losses and smoke test that exist), the readings and stop conditions the lanes work
  under, step 2 by lane (the release tier with the engine configured as the
  sign-off's gate evidence, H23; §15.2 re-run on the v1 network), what is parked and
  why; the work remaining to §15.5, the lanes in parallel; what waits on the user
  (nothing that blocks: the standing items, and the lanes' issues put to the user as
  each lane merges). "Recovery": the comparison with the last green CI runs, made after
  the user's `gh auth login` - the same results; the engine job's one failure on
  3b33942 (g++-13) is an intermittent five-thread Guard test, queued for the engine
  lane. The old state stays as history under "State on 2026-10-04 (before
  the loss)", its held, running, remaining and waiting lists superseded; the recovery
  notes point at the plan.
- "Register status": rows added for A6-A8, B15-B17, D9, G8, G9, H49, H50, I12-I20,
  M19-M21 (M21 built in part: the losses of today's heads and a three-step smoke test,
  N1+N2 and N5), M23 and M24 (F5, F6, H47, H48 and M22 arrive with their lanes'
  commits, as a note under the heading says); rows split or corrected where the
  status or what is built changed - H1, H6, H10, H13/H14, H17, H19, H22, H23,
  H24/H26/H28, H36, H41, H43, H44, A1-A3, A5, B1/B2/B4/B8, B5/B6 (B6 superseded by
  B15), B7, B9, B10/B11/B12, B14, C2, C3, C10, C11, C14, C15 (with a known defect,
  checked on CPU: the FP8 fake-quantiser's backward flushes or distorts gradients),
  C17/C19/C20, D1-D4, D7, D8, E1-E5, E7-E9, F1/F2, G2, I1-I8 (I3 built in part: match
  play's noise-off, final-rule configuration), I10, M9, M13 - each saying what is
  built today: nothing of E10 on main, the old network defaults still in the code,
  the ABI rounds still to come. H46's row moved into the H table, as the register
  moved it.
- "Spec issues": 79 marked settled upstream at CHECKPOINT 6/7 - 3, 6, 9, 11, 16, 17,
  19, 23, 24, 26, 28, 30-32, 34, 35, 37-39, 42, 44-47, 49, 51-53, 55, 56, 58, 59,
  62, 64, 67, 68, 71, 72, 74-79, 81-85, 87-93, 95-108, 110, 111, 113, 114, 116-120 -
  and 7 in part (2, 33, 36, 54, 109, 112, 121), each with the register row or section
  that settles it; 20 settled at CHECKPOINT 2 (M13), 63 superseded by M16's formula
  (CHECKPOINT 4), 70 moot since v3 retired the estimate channel. New, the main loop's:
  300 (early adjudication and MLH steering's "settled" outcome unspecified), 301 (the
  tree form under MCGS - the engine refuses the combination as a parameter check,
  parked; no document refuses it), 302 (A6's count in the evaluation identity), 303
  (A7's frozen K/V per perspective under the cached configuration), 304 (§15.2's A7
  row against the acting-colour input), 305 (one turn unit; E7's allocator counts
  board-moves), 306 (the root-scan reserve with Guard offline). "Proposed spec
  edits": 72 marked
  applied upstream - 1-8, 11-16, 19, 20, 22-30, 32, 33, 35, 38-41, 43-47, 49, 50, 52,
  54, 55, 58-68, 70, 73, 74, 76, 77, 79, 81-84, 87-92, 94, 95, 97 - and 14 in part,
  with what is still needed (9, 21, 31, 34, 48, 53, 56, 69, 71, 72, 78, 80, 93, 96);
  51, 57, 85 and 86 withdrawn, superseded or moot; 17, 18, 36, 37 and 75 stand as
  written. New, the main loop's: 200-204 (for issues 302-306), 205 (§12.3's
  parenthetical and §11.2's status note after 6b″), 206 (the register's status
  vocabulary). The number blocks are stated at the top of both lists; nothing is
  renumbered or deleted.
- "Parked, and why": the plan's parked items (I10's calibration monitoring, C3's
  conditioning-offset training hook, 6p's other arms, MLH steering - with what it
  waits on - and early adjudication, issue 300) and §14.4's new held-open and refused
  entries. The brief: M23's rule beside "never force-push". The session log.
- Spec: register parts A-M as changed at CHECKPOINTS 6-7; no spec text edited here.
  Proposed, the main loop's: edits 200-204 for issues 302-306 (§2.2, §7.1, §8.3,
  §8.5, §8.6, §8.7, §10.3, §12.4, §12.8, §15.2's A7 row, §15.3's cached rows;
  registers C11, C14, D3, D4, E7, H18, I13, I19), 205 (§12.3 and §11.2's status note,
  after 6b″) and 206 (the register's status vocabulary).

### 2026-10-17 - Design documents: the CHECKPOINT 6 and 7 rulings, from upstream

- `docs/design/5d-chess-decisions.md`, `docs/design/5d-chess-transformer-design.md`,
  `docs/design/scalable-transformers-architecture.md`: the copies maintained
  upstream after CHECKPOINTS 6 and 7, committed unedited. Register rows added: A6
  (turn-state inputs), A7 (perspective canonicalisation), A8 (board-layout
  embedding), B15 (the 4D displacement bias, superseding B6), B16 (score-time status
  key biases), B17 (the key-side Shaw term), D9 (the optimistic policy), F5
  (normalisation and biases), F6 (Mish), G8 (QAT prepared), G9 (the inference path;
  ONNX/TensorRT refused for now), H47 (the "some ruleset is gated" check only in
  tiers that run every oracle - Q-A), H48 (the naive searches' release budgets
  raised and re-recorded before the sign-off - Q-B), H49 (Hunt with a lazily
  expanded root - held open), H50 (the asset map's 5D piece values - provisional
  placeholders), I12-I20 (§12.8's
  training mechanics: surprise sampling, deblundering, proof rescoring, rich
  records, the replay window, SWA, the false-resign check, exploration
  placeholders, KL-adaptive searches), M19 (no GPU: counted FLOPs and CPU-labelled
  wall times, GPU figures deferred), M20 (the placeholder dispositions), M21 (the
  training loop smoke-tested), M22 (the turn-width measurement before the D2 arms),
  M23 (push every branch; merge, never rebase), M24 (ablations on a shared
  dataset). Changed: A3, A5 (build behind its flag), B4-B9, B11, B12, C2, C5, C7,
  C10 (the window killed provisionally), C11, C14, C15, C20, C24, D1-D4 (D3 corrected:
  WDL on the result, the categorical over the root search value; D4's reply head
  reinstated as a prototype), D8, E1, E4, E7, F1, F2, F4 and H7 (held open with the
  NNUE), G2, H1, H5, H13, H14, H17-H19, H22, H28, H30, H32, H33, H37, H41-H44, I2,
  I3, I5, I6, I9-I11, M5, M8, M9, M13, M15, M17; reversals 25-29; part K and part L
  entries added. Many of the build log's proposed spec edits are now carried
  upstream; which ones, and the plan the rulings set, are in the build log.
- Spec: the whole of CHECKPOINTS 6 and 7, as written upstream - 139 changed
  passages in §0, §2.2-§2.3, §4.1-§4.2, §5.3, §6, §7.1, §7.3, §7.9, §8.1-§8.7, §9.2,
  the new §9.3, §10.2-§10.4, §11.0-§11.4, §12.2-§12.8 (§12.8 new), §13, §13.1,
  §14.1, §14.4, §15.1-§15.6 and the summary; the register's parts A-M; the general
  pattern document. Nothing in them edited here.

### 2026-10-17 - Recovery after the lost machine: the toolchain rebuilt, main verified, the losses listed

- `docs/BUILD_LOG.md`: "Recovery after the lost machine" at the top - what survived
  (everything on `main` at 3b33942), what was lost because it was never pushed (the
  gate-with-engine branch with its 113 triaged known divergences; engine E10) and
  where each is redone; main's verification on the rebuilt machine (engine 526 of
  526, network 775 passed and 2 skipped, `verify`, `selftest`, the commit-tier
  gate); the comparison with the last green CI run still to do. "Environment"
  rewritten for the rebuilt machine, with the recipe that rebuilt it. The session
  log records the loss and the recovery.
- `docs/MEASUREMENTS.md`: machine row `dev-r` (the same hardware, reinstalled;
  Python 3.14.7, CMake 4.4.3, the same g++ 16.2.0 and PyTorch 2.13.0+cpu): figures
  from before and after the loss carry different conditions; the gate's calibration
  probe ran 1.18x the recording machine's time on the first run after the rebuild.
- Spec: none - the record of a recovery; §15.3's conditions rule (every figure with
  its machine) is what the new machine row serves. Register M23 (push every branch
  after each commit) is the rule followed from here.

### 2026-10-04 - Build log: the current state, the work remaining to §15.5, and the register brought up to date

- `docs/BUILD_LOG.md`: a "Current state" section at the top (what is on main, the
  measured verdicts, what is held and running, the work remaining to §15.5 in
  order, and what waits on the user - the gate's Q-A and Q-B, GPU-only cost
  figures, the committed-class placeholders, A5, D8's whole-turn token, the
  training rows); 2026-10-02's stale lists superseded; "Parked" brought up to date
  (the window killed provisionally; the NNUE held open, the retired rungs);
  register rows H6, H7, H13/H14, C5/C7, C10, M7, I1-I10 corrected and C6, C8, C9,
  D5-D7, E4, G7, I9, M1-M6/M10 added (from a read-only audit of the repository
  against §15.5); spec issues 120-121 and proposed spec edits 91-97 (§15.2 and
  §15.3 rows that cannot exist in the untrained build; the movegen and scanner
  write-backs; register fixes).
- Spec: §15.2, §15.3, §15.5, §15.6; register as listed; no spec text edited here.

### 2026-10-04 - Engine CI: the prover's hash constants typed for g++ 13/14

- `engine/src/prover.cpp`: the prover's 64-bit hash constants are written as
  `std::uint64_t{...}` instead of `ull` literals. On Linux `std::uint64_t` is
  `unsigned long`, and g++ 13 and 14 rejected one expression mixing the two under
  `-Werror=sign-conversion` (CI's engine job failed on the E9 merge, d96d6d9); g++
  16 here did not. All seven literals converted; 526 tests pass.
- Spec: none - a portability fix.

### 2026-10-04 - 6b″ on a quiet machine: Hunt adopted under 100 boards, Guard in no range (M17)

- `docs/MEASUREMENTS.md`: experiment 6b″ (M17) with engine E9 merged (d96d6d9),
  on a quiet machine, every test as pre-registered and read per board-count
  range - the same 805 roots as 6b, 5 threads, 50 ms, Guard and Hunt as they ship,
  only E4-confirmed proofs counted. Guard's depth 1 completes in 65.1% / 15.8% /
  5.2% / 0.0% of roots (pass: 95%) - adopted in no range. Hunt finds 69.0% / 63.6%
  / 35.0% / 0 of 1 of the ground truth's mates (pass: 50%) - adopted under 100
  boards. Every one of Guard's 49 refutations and Hunt's 77 mates confirmed by E4
  alone. No outcome stops the build (M17). Hunt's ground truth recorded there too.
- `docs/BUILD_LOG.md`: register H42, H43, M17; the §13 consequences (the chosen move
  goes unguarded in every range; from 100 boards up, proofs only from the search's
  solver) are proposed spec edit 75.
- Spec: §11.4, §13, §15.6 (6b″, M17); register H42, H43, M17.

### 2026-10-04 - Engine E9: verification v3 - Guard, Hunt, the asset map; Hunt's ground truth and 6b″'s tool (reviewed)

- `engine/` (E8's prover and E9, merged from their worktree branch after two
  adversarial reviews and a fix round; main merged into it first):
  - The per-leaf estimate channel removed (rung 1, retired at CHECKPOINT 4).
  - E8's df-pn over the semimove-decomposed turn tree and its incremental
    royal-attack maps carried into Guard and Hunt (ruling 1): prove-only; the
    literal class entered once; a depth override; proof lines and certificates
    covering every proof the prover exports, from any thread or search; an
    attacker's turn asks E4 first; disproof classes carry every option that
    shapes them; the prover's relief test counts any inactive timeline (H46).
    The last-turn check filter was found unsound by the review (a jump that turns
    branching once its target board is played - J6's twin - was never counted):
    fixed conservatively, off by default, it now cuts almost nothing.
  - `t5d/certificate.hpp` checks a proof against E4 alone and names any position
    it cannot certify.
  - Guard (H42, ruling 2): before the chosen action is submitted, iterative
    deepening from depth 1 - can the opponent force mate against it? - exact at
    every completed depth, at most 50 depths, never a depth it did not search; a
    refutation is a certified forcing line; the re-search hook for §8.6's reserve.
  - Hunt (H43, ruling 3): prove-only df-pn at the root, claiming nothing at its
    budget; a found mate is a proof, its export refused until H41/H23; partial
    lines as soft priors behind a flag, off.
  - The superseded-board asset map (§11.2 rung 4): once per superseded board, a
    material estimate and a small qsearch, cached by content; a standing prior
    boost on travel into flagged boards.
  - Hunt's ground truth (ruling 5): `t5d-hunt-truth`;
    `engine/testdata/hunt-truth/` - one long offline search per 6b root (the
    union of the literal and the default class's proofs, 4M nodes each), 125 mates
    on 805 measured roots, every one confirmed by E4 alone; pinned by a MANIFEST
    with its provenance (the records' content IDs in order, the rules by value, a
    code signature); re-recorded with the reviewed prover as a check, every
    verdict unchanged.
  - 6b″'s tool (M17): `t5d-scan6bpp` - the same 805 roots, per range, 5 threads,
    50 ms, Guard and Hunt as they ship, only E4-confirmed proofs counted, every
    verdict withheld if one is refuted. The quiet-machine run follows.
- Tests: 526 (`--werror`; E9's default tests 84 s, each heavy case with a `long_`
  twin); the reviews' surviving mutations and every review fix undone: 26 of 26
  caught. The Lazy SMP timing test three times before the merge (logged); verify,
  boundary and the commit-tier gate green.
- `docs/BUILD_LOG.md`: register H1, H2, H10, H42-H44, M14, M17; the timing log;
  spec issues 97-107 and 117-119, proposed spec edits 72-78 and 87-90.
- Spec: §8.6, §11.1, §11.2 rung 4, §11.3, §11.4, §12.2, §13, §15.6 (6b″);
  register H1, H2, H10, H18, H19, H23, H41-H44, H46, M5, M13, M14, M17.

### 2026-10-04 - Network N5: D8 in the network; the search's C step API from Python; end-to-end games with random weights

- `python/t5d/net/` (merged from its worktree branch after an adversarial review
  and its fixes): `PositionState` carries the acting player and the submit
  action's possible/legal flags (validated against T5); a learned acting-player
  embedding added to the global registers (`acting_player_input`, placeholder;
  `all_tokens` the other arm); the submit action's own logit read from global
  register 0 (`submit_register`, placeholder) by a linear readout (`submit_head`,
  placeholder; `pointer` the other arm), offered where possible and masked unless
  legal; one policy distribution over moves + submit; value and WDL from the
  acting player's perspective; the policy loss includes the submit target. D8's
  parameters are drawn from their own seed-derived generator, so every pre-N5
  tensor draws as before under any flags. F1 = 21,359,370 parameters (+769).
  Dense = cached (§15.2) with the new input.
- `python/t5d/engine/`: the converter accepts states past the shift and checks
  every move against the acting player; C23's ancestry is confirmed as its
  lineage. `capi.py` mirrors `capi_search.h`, checked against
  `t5d_search_layout` at load; `Search` wraps the step API (the proof channel
  refused, H41); `CStepDriver` drives the engine's search with the network,
  bit-identical to `t5d_search_run` under the engine's own evaluators; C14's cache
  sharing only between searches with the same network input, refresh epochs set on
  the search; `netplay.py` plays self-play and match games (both D2 arms), writes
  5dpgn records, checks each result against the final position and replays every
  record through t5d-gate.
- Review fixes: the value's perspective and the logit order through the step API
  tested; a refused engine library fails the engine tests, and
  `T5D_REQUIRE_ENGINE=1` (set in CI) makes a missing one fail; CI builds t5d-gate;
  re-root fallbacks counted; a non-finite legal submit logit refused. 19 of 19
  mutations killed.
- End to end with random weights (CPU, information): 20 games, every record
  replayed legal; small network 1.37 decisions/s and 35 evaluations/s, F1 (BF16)
  0.60 and 6.9.
- Tests: Python 775 passed, 2 skipped (`T5D_REQUIRE_ENGINE=1`); ruff clean.
  `engine/README.md`: the E7 note on the network brought up to date.
- Spec: §2.3/A3, §8.2/D8, §8.3/D3, §12.2/§12.4, §9.2/F1, §7.3/C17/C23,
  §8.7/C14/C12, §15.2, §15.5; register D8, D2, C14, C23, H41; spec issues 108-116,
  proposed spec edits 79-86 in `docs/BUILD_LOG.md`.

### 2026-10-03 - Oracle harness: cwmtt/many2 read converted to 5dpgn by every source (CHECKPOINT 3); golden files re-recorded

- `oracles/oracle_harness/export_form.py` (new; merged from its worktree branch)
  carries out the user's ruling: "the cwmtt driver converts the export form to the
  canonical 5dpgn form from the pinned notation spec; if it expresses something
  the spec can't, mark the record unsupported. The engine's parser is not
  extended." A move that names its board twice on one board (the specification's
  "Export moves", `(0T1)Pe2(0T1)e3`) becomes the specification's physical move
  (`(0T1)Pe2e3`; castling `(0T6)Ke1g1`). Nothing is added - no capture, check or
  promotion mark, which would mean reading the position - and every other byte is
  copied. A move to another board without `>`/`>>`, other text with a board
  straight after a square, or the export form's `w.`/`b.` turns make the whole
  record unsupported. Written from the specification's text, not from any
  oracle's parser.
- `corpus.feed` (cwmtt's side of the harness, which supplies its records) converts
  a cwmtt record in the export form, and every source reads the converted text -
  the engine included, still on standard input; converting only what cwmtt reads
  would change nothing. A record no source is given is reported unsupported
  (`export_form`) for every source. Every other record is fed byte for byte as
  before. The manifest pins what the sources read (`converted`), checked by
  `verify` and every read; golden files and known divergences name a converted
  record by that hash. The gate's log and report and DISAGREEMENTS.md ("Records not
  read as stored") name converted records. many2: 153 of 204 moves rewritten.
- `oracles/golden/*`, `oracles/DISAGREEMENTS.md`: re-recorded with `golden --write
  --repeat 2` on a quiet machine (2026-10-03, 30 min). Only cwmtt/many2's answers
  changed, in every file: ftxi, Chess in 5D and the notation parser now read the
  game they refused at its first move (ftxi and the notation parser all 63
  positions at the commit tier); cwmtt's two answers are unchanged but for the
  record's hash. What each tier requires: more values everywhere (ftxi 7,760 ->
  8,256; notation 2,765 -> 3,012); ftxi:naive's many2 outcome, "rejected after 0
  actions" when it could not read the record, is now a run cut at its 300 s budget
  at position 16, so its nightly and release tiers require one outcome fewer (33
  -> 32), as does Chess in 5D's nightly tier (41 -> 40) - more checking of many2,
  not less. `verify`'s informational proof-channel notes: 17 (16 before).
- Tests: `oracles/tests/test_export_form.py` (21) and a gate-run test; selftest 623
  OK; verify, the commit-tier gate green. `oracles/README.md` ("How a record
  reaches the sources"; the soft legality entries).
- Spec: §11.0 ("Third-party code"), register H28 (and H27/H33): the one exception
  to "every source reads the record as stored" - spec issue 96, proposed spec edit
  71 in `docs/BUILD_LOG.md`.

### 2026-10-03 - Engine E6a: the search core - PUCT/MCGS, both D2 arms, the solver behind H41, self-play, the C step API

- `engine/` (merged from its worktree branch after three adversarial reviews and a
  fix round):
  - `t5d/mcts.hpp`: PUCT with entropy-adaptive top-k widening, FPU and root
    Dirichlet noise (E1; noise on in self-play, off in match play); E2 and E3
    behind flags; both D2 arms (the board-move split, default, and whole turns);
    MCGS behind a flag (C11), its identity the state key and whether an action is
    in progress, compared exactly; the solver's bounds from E4's exhaustive
    verdicts only, their propagation and export behind `proof_channel`, off by rule
    (H41) - game ends are still scored as results; final selection by LCB with a
    visit floor (E5) and the secure child (E9); re-rooting with garbage
    collection, persistence and pondering (C12, C13); virtual loss; the
    synchronous mode and digest (H15); E5a's legality tiers under the (a)-(f)
    policy; node, memory and time budgets with a stop reason.
  - D8 and R8: submitting is an explicit edge, offered where the obligations are
    met, legal only when C4 holds, never automatic; the acting player's value
    perspective; game ends only at turn starts (C5, E4's exhaustive detector);
    dead ends during an action pruned, never scored; the node key with the acting
    player (H18) and branch origins (C23).
  - `t5d/selfplay.hpp`: fresh searches per decision with a shared, board-count
    purged evaluation cache (C14), match play, the turn cap (I11), safety stops.
  - `t5d/capi_search.h`: the step API for an external evaluator (logits, submit
    logits, values as probabilities or logits), cache sharing, run limits, the
    digest, `t5d_search_layout`; embedded provisional structs last; caller errors
    and internal errors told apart.
  - `t5d-play`: self-play and match play to 5dpgn records (rules through the
    shared option reader, a 300-board default cap, `--proof-channel`,
    `--no-noise`).
- The reviews' findings (16) all fixed or logged: MCGS merging a turn start with a
  mid-action state (two reviewers; a legal mating submission pruned), the solver
  on before the gate, an `advance_action` that changed the tree on failure, epoch
  changes under a waiting batch, a leaking submit logit, an evaluation cache that
  never evicted and undercounted its bytes, no memory ceiling, non-append-stable C
  structs, exception safety, validation gaps, a deadline checked only at turn
  starts; 16 surviving mutations now killed (53 of 53 that build).
- Tests: 474 (`--werror`; the search's share 84); the Lazy SMP timing test three
  times before the merge (logged); verify, boundary and the commit-tier gate green.
- `docs/BUILD_LOG.md`, `docs/MEASUREMENTS.md`: register D2, D8, E1-E3, E5, E7-E9,
  C11-C14, C23, H2, H15, H41, I11; spec issues 80-95 and proposed spec edits 64-70;
  the timing log; the search's memory accounting and smoke games.
- Spec: §8.1-8.7, §11.1 item 2, §15.4; register D2, D8, E1-E9, C11-C14, C23, H2,
  H15, H18, H19, H23, H41, I11.

### 2026-10-03 - Network: the refresh ladder's complexity trigger (C24), behind a flag

- `python/t5d/net/complexity.py` (new; merged from its worktree branch): C24 as
  ruled at CHECKPOINT 4 ("goes behind a flag alongside the refresh ladder's rung
  1"): the signals - the node's board count and active timelines (structure), its
  travel since its last refresh (the tree's path), the parent's predicted
  difficulty from the effort head (§8.6, the bin at a quantile) - and the rule, a
  weighted sum against a threshold.
- `python/t5d/net/cache.py`: `CacheConfig.complexity_refresh` (prototype, off by
  default) and its named parameters, every number a placeholder: threshold 1.0;
  weights 1/128 per board, 1/8 per active timeline, 1/2 per travel move, 1/12 per
  effort bin; effort quantile 0.5; visit gate 50 (rung 1's V); at most 8
  complexity refreshes per search.
- `python/t5d/net/tree.py`: `SearchTree.visit` decides rung 1 first, unchanged,
  then C24 where rung 1 did not fire - add-only for every signal;
  `travel_since_refresh`; `Node.effort_logits` (the parent's output, recorded
  when the flag is on) and `Node.refreshed_by`. The node's own outputs are never
  read. A complexity refresh is rung 1's copy-on-write fork, at another node.
- Tests: `python/tests/net/test_complexity.py` (18): off changes nothing (bitwise
  rung 1 alone); on is add-only and fires by its rule; the effort signal is the
  parent's; the decision evaluates nothing; determinism; the fork; 13 mutants of
  the trigger killed. Python 649 passed, 2 skipped; ruff clean.
- Not built: C25 and the scanner NNUE (held open). No training commitment.
- Spec: §7.9 "Complexity as a trigger", §8.6 (the effort head); register C24,
  C3a; spec issues 75-79 and proposed spec edits 61-63 in `docs/BUILD_LOG.md`.

### 2026-10-03 - CHECKPOINT 5: the window killed provisionally; Q2 ruled - initial timelines created by neither player (H46)

- `docs/design/`: the upstream revision recording CHECKPOINT 5 - M16 corrected (under
  50 boards the per-phase rule decides; a result in the ±10% band resolves to off;
  the window killed, provisionally), M18 (decisions weighted by where compute goes
  use measured game-length distributions; a pilot self-play run before serious
  self-play), H46 (Q2), reversal 24 and §13's size-premise entry. Copied in
  unchanged.
- `docs/rules/README.md`: ruling R10 (Q2; H46, a provisional project decision):
  initial timelines count as created by neither player - always active, never
  counted toward either player's created timelines; Q2 closed provisionally; T10
  and L8 point to it. Rulebook re-pinned in `oracles/gate.toml` after review (no
  known divergence cites the changed passages); the gate re-run.
- `engine/`: a named variant's set-up now reads every one of its timelines as
  initial (`multiverse_from_fen` / `game_from_fen`'s `all_initial`, used by
  `variant_start`): Timeline Battleground's -1 and +1 have no creator and do not
  count toward either player's created timelines (they had been read as one
  creation each side - the same activity, since the count rule treats the sides
  alike, but a different Zobrist key and creator feature). Plain 5DFEN keeps its
  convention (its numbering implies a creator). `initial_credit`'s default cites
  R10/H46; its alternative stays for diagnosis. Tests: a new H46 test over every
  multi-timeline set-up; the Battleground and round-trip tests follow H46; the
  expansion reference test now draws unknown origins from positions read back
  from 5DFEN. 390 tests; the gate's output on cwmtt/ctp2 (Battleground) is
  byte-identical.
- `docs/BUILD_LOG.md`, `docs/MEASUREMENTS.md`: CHECKPOINT 5's rulings; register
  M7, M16, M18, H46; nothing waits on the user. `CLAUDE.md`: rulings R1-R10.
- Spec: §11.0 item 2, register H46 (built as ruled); M16, M18, reversal 24, §13
  (recorded); rules T10, T13, L8, Q2, R10.

### 2026-10-03 - Build log: CHECKPOINT 4's ruling; M7 re-read under M16

- `docs/BUILD_LOG.md`: CHECKPOINT 4 concluded - verification v3 (§11.4) and the
  user's eight rulings (E8 into Guard and Hunt, Guard, Hunt, the audit lane's
  tooling, Hunt's ground truth, 6b″, the complexity trigger, and CHECKPOINT 3's
  order), what is running and what follows; register rows H1, H10-H14, H41-H45,
  F4, M16, M17, C24, C25 and M7.
- `docs/MEASUREMENTS.md`: M7 re-read under M16's readings: on at k = 8 from 50 to
  186 boards (1.83x, 2.10x, 3.81x by the time-share multiplier), off under 50
  boards - and then killed by the kill clause, the on-phases carrying 18-38% of the
  compute under the human-game weighting M16 prescribes. That contradicts M16's
  premise that the under-50 phase is a small share; put to the user. Not a stop.
- Spec: §11.4, §15.6 (M16, M17, the window's rule); register H1, H10-H14,
  H41-H45, M7, M16, M17, C24, C25.

### 2026-10-03 - Design documents: the CHECKPOINT 4 ruling (verification v3), from upstream

- `docs/design/` replaced with the upstream revision recording CHECKPOINT 4: §11.4
  "The verification layer, v3" (exact rules always; the search's solver for
  proofs; Guard and Hunt at every size; full settling scans only under 50 boards;
  the offline audit lane; the adversarial league primary above 50 boards); §15.6's
  CHECKPOINT 4 result and 6b″'s pre-registered tests; M16's three readings of the
  window's rule; §11.1's ruling that the search's solver is part of the proof
  channel; §11.3's acting player in every state key; the complexity trigger (C24)
  and the dense top layers (C25, held open); register rows H1, H2, H6, H10-H12,
  H18, M5, M14 updated and C24, C25, H41-H45, M16, M17 and reversal 23 added; the
  architecture document's guard and hunt. Copied in unchanged.
- Spec: the documents themselves.

### 2026-10-03 - 6b′ on a quiet machine: the verification redesign fails at 50-199 boards - M15's STOP

- `docs/MEASUREMENTS.md`: experiment 6b′ (M15) with engine E8's prototype of the
  redesign (M14; branch bd08c48, not merged - it waits on the design review), on
  a quiet machine, every test as pre-registered and read per board-count range.
  The df-pn proof channel decides 77.4% of roots under 50 boards (pass), 40.7% at
  50-99, 23.7% at 100-199 and 0.0% at 200+ (fail); the estimate channel fits the
  rung-1 leaf budget in no range; below 50 boards df-pn decides more than brute
  force (408 against 400). M15's outcome: STOP - a design review.
- `docs/BUILD_LOG.md`: CHECKPOINT 4 - the STOP and what the review has to work
  from; register M14, M15; spec issues 67-74 and proposed spec edits 56-60 (E8's);
  the owed long tests' results (E5c's two pass on main).
- Spec: §11.1 "The redesign after 6b", §11.2 (rung 1, H10), §11.3 (H19), §15.6's
  CHECKPOINT 3 block (6b′); register H1, H2, H10, M14, M15.

### 2026-10-03 - Engine E7: Q4 as ruled (R8) by default; the acting player and the submit action (D8), with the binding

- `engine/` (merged from its worktree branch, main merged into it first):
  `RuleOptions::moves_after_shift` defaults to true - a player may keep moving
  after the present has shifted, until submitting (rulebook R8, register D8). The
  Guide-literal reading is a named alternative, for diagnosis. The default
  encoding and ruleset ID stay 0 (no default Zobrist key changed); encoding bit 3
  now marks the Guide-literal reading - the opposite of its old meaning - as
  `T5D_RULES_NO_MOVES_AFTER_SHIFT`, and `T5D_RULES_MOVES_AFTER_SHIFT` is 0.
  t5d-gate, t5d-measure, t5d-bench, t5d-scan6b and the C API's ruleset 0 follow
  the default; `--no-moves-after-shift` (t5d-gate also `--rule
  moves_after_shift=false`) is the diagnostic, `--moves-after-shift` is accepted
  and ignored. The hypercuboid's default regime is FreeOrder. Tests that assumed
  the old default ask for it explicitly; E5c's scanner tests pair the
  Guide-literal reading with the default over the same positions.
- D8: `Expansion::acting_player()`, `submit_possible()` (the action's obligations
  met, T5) and `submit_legal()` (and C4, exact), appended to `t5d_expansion` as
  three `int32_t` after `src_extra` (size 272 -> 288; every earlier offset kept;
  no board or square feature changed). `t5d_layout` reports every public struct's
  size and field offsets.
- `python/t5d/engine/` (the CHECKPOINT 3 ruling: merged together): `capi.py`
  mirrors the new fields and constants and checks every mirrored struct against
  `t5d_layout` at load, refusing a library of another stage instead of writing
  past a buffer; `Expansion` copies the fields; the converter still refuses states
  past the shift until N5 (the acting-player input and the submit logit).
- The corpus through t5d-gate's full mode under R8: 41 records in 36-44 s on a
  shared machine, none over 20 s, peak 131 MiB; the ten records that stopped on Q4
  read on (cwmtt/futures stops at action 17 on notation, as known); the output is
  byte-identical to the triage's earlier runs under this reading, so its 115 Q4
  findings agree with the oracles.
- Tests: C++ 389 (`--werror`); Python 631 passed, 2 skipped; ruff clean; verify,
  boundary and the commit-tier gate green (the engine is not yet configured in
  the gate). The Lazy SMP timing test, run before the merge as the user requires:
  three runs on E7's branch and three on the merged tree, 5 threads 309-313
  against 1 thread 293, no retry (`docs/BUILD_LOG.md`, the timing log).
- `docs/BUILD_LOG.md`: E7 done; issue 48's engine side built; proposed spec edit
  42 withdrawn (superseded by D8); the long tests still owed.
- Spec: §8.2, §11.0 item 2 (D8, R8), §2.3 / A3 (issue 48); §15.6 (6b's and M9's
  default streams are now R8's). No spec text edit needed.

### 2026-10-03 - M7 on real games: the |E| curve from t0database, its verdict, the refused games

- `docs/MEASUREMENTS.md`: the |E| phase curve over the 716 t0database games that
  replay to the end (901e5be, Q4 as ruled, legal and pseudo-legal destinations),
  the games' board counts (max 186; 14 reach 100 boards), and M7's verdict from
  §15.6 as written: on at k = 8 from 50 to 186 boards; under 50 boards inside the
  ±10% band (1.60x at k = 4, 1.45x at k = 8, against 1.5x), and the kill clause
  turns on it - stopped for the user; no data above 186 (provisional). The
  multiplier formula §15.6 leaves unwritten is the one reading that reproduces its
  own table (m_cached^0.9 · m_dense^0.1). The corpus section is kept, superseded.
- The 45 refused games classified (no engine change): 13 unfinished final actions
  (12 completable - 11 with a clock at 0:00 or a timeout - and one played from a
  position already checkmate; the engine is right under T4, T5, T10 and T11 by
  every reading the rulebook lists); 15 Standard - Defended Pawn and 2 Two
  Timelines games under the source's Turn Zero label, and 2 unknown fairy set-ups;
  10 pawn captures with a doubled file (`ccxd5`), one record without sub-turn
  separators, one stray token and one dropped turn.
- `docs/BUILD_LOG.md`: register M7, the parked window, spec issue 26 answered to
  186 boards, new issues 63-66, proposed edits 51-55, the three questions for the
  user.
- Spec: §15.6 "The window (|E| phase curve)", §10.4.2 step 1, register M7 (M4
  write-back proposed); rules T4, T5, T10, T11.

### 2026-10-03 - Build log: CHECKPOINT 3's rulings, Q2 written out, the timing-test rule

- `docs/BUILD_LOG.md`: the CHECKPOINT 3 rulings (the M14 redesign as a prototype
  and 6b′ with M15's outcome rules; the unpause order E7 -> E6a -> the gate -> N5;
  cwmtt/many2's conversion and how it is carried out; t0database approved), what
  is running and what follows; Q2 quoted with its candidate readings and what each
  changes (only even-start variants; nothing in today's corpus or t0database
  distinguishes them); a standing rule and a log for the Lazy SMP timing test
  before any merge that touches the scanner.
- Spec: §11.1, §15.6 (M14, M15); rules T10, T13, Q2; no spec change.

### 2026-10-03 - Real game records: t0database, pinned, for the |E| curve

- `data/t0database/`: MageOfficial/t0database's `Data.csv` and `LICENSE` (MIT) at
  commit `d49da405...`, copied byte for byte (the user approved the download at
  CHECKPOINT 3): 761 real games, 2021-2025, up to about 186 boards. A README with
  the pin and hashes, and `to_5dpgn.py`, which writes each game as a 5dpgn record
  with the source's own header convention (`[Board "Standard - Turn Zero"]` for a
  game written without headers) and touches no move. `.gitattributes` keeps their
  bytes; `PROVENANCE.md` lists them (measurement only: not an oracle, not a rules
  source, not the gate's corpus).
- Spec: §15.6 "The window (|E| phase curve)" and register M7 (the data the rule
  needs; spec issue 26); the measurement follows in its own entry.

### 2026-10-03 - Design documents: the CHECKPOINT 3 ruling, from upstream

- `docs/design/` replaced with the upstream revision recording CHECKPOINT 3: §11.1
  "The redesign after 6b" (split the scanner by channel - exactness only where a
  proof can result; the proof channel a depth-limited df-pn over the
  semimove-decomposed turn tree with hypercuboid-pruned completion; incremental
  royal-attack maps), §15.6's CHECKPOINT 3 block (6b′'s tests and outcome rules),
  §13's note that exact verification may not reach depth at high board counts;
  register rows H1, H6, H10 updated, M14 (the redesign, a prototype until 6b′) and
  M15 (6b′, pre-registered), reversal 22; the architecture document's §13.2.7 "Pay
  exactness only where a proof can result". Copied in unchanged.
- Spec: the documents themselves (§11.1, §13, §15.6; H1, H6, H10, M14, M15).

### 2026-10-03 - Engine CI: the Lazy SMP timing test needs cores to test anything

- `engine/tests/test_scanner.cpp`: the test that 5 Lazy SMP threads resolve at
  least one thread's depth in the same 40 ms of wall clock failed on CI's runners,
  which have 2-4 cores: there the 5 threads oversubscribe the machine and the claim
  is not being tested. It now runs only where the machine has at least 6 hardware
  threads, and says so when it skips. On the development machine it runs and passes
  as before.
- Spec: §11.3 (H17), M13; no spec change.

### 2026-10-03 - 6b re-run after the optimisation pass: M12's outcome is STOP

- `docs/MEASUREMENTS.md`: experiment 6b re-run as M12 rules - after E5c's bounded
  pass, at the scanner's design budget (5 threads, Lazy SMP), every definition
  unchanged, read per board-count range, on a quiet machine. The 50 ms root scan
  resolves 5 forcing half-turns in 76.7% of positions under 50 boards (pass), 31.4%
  at 50-99, 24.7% at 100-199 and 2.4% at 200+ (fail); rung 1 and quiet-leaf reach
  fail in every range. M12's pre-registered outcome: "50-199 boards fail -> STOP:
  redesign the verification layer".
- `docs/BUILD_LOG.md`: the STOP, what it asks of the user, and what is paused
  (engine E7 and E6a on their branches); register rows M8, M12, M13, H17.
- Spec: §15.6 (6b, M8, M12, M13), §11.1 (the kill condition), §11.3 (H17).

### 2026-10-03 - Engine E5c: M12's optimisation pass, Lazy SMP, 6b per board-count range

- `engine/` (merged from its worktree branch; M12, M13, H17, H19): the in-check
  stage - M12's one bounded pass on the two paths where 6b's first run spent 85% of
  its nodes. In check, up to 64 seeds in evasion order (the TT move, a capture of
  the checker, a royal move, a block, other captures, the rest) are made into legal
  actions and searched as each is built: a cutoff resolves the node, otherwise E4's
  enumeration runs as before; at the horizon one legal action answers "no mate",
  E4's detector deciding only failing that. Proofs as before (H2, H37);
  `evasion_probe = 0` reproduces E5b node for node. The two paths fell from 64-80%
  of the nodes below 200 boards to 45-51%, and from 50% to 16% at 200+; the
  completion fallbacks now lead at 100+ boards. Lazy SMP (H17) behind
  `ScannerOptions::threads`: a lockless shared TT (estimates only), a sharded
  concurrent solved store (proofs exported at once, contradictions refused, H19),
  per-thread histories, staggered depths and perturbed move orders; one thread is
  the deterministic mode (H15).
- `t5d-scan6b`: the scanner at M13's 5 threads - the 50 ms root scan by the wall
  clock, rung 1's budget - every measure per board-count range with M12's outcome
  rules applied; pooled lines labelled "not a verdict (M12)".
- Tests: 383 (5 new by default; 2 new long tests - no false proof with 2-8 threads).
- Spec: §11.1, §11.3, §15.6; H2, H15, H17, H19, H37, M8, M12, M13. The re-run of
  6b on a quiet machine follows.

### 2026-10-02 - Build log: public sources of real game records, proposed

- `docs/BUILD_LOG.md`: the game-record sources found (CHECKPOINT 2, item 6), for the
  user's approval - MageOfficial/t0database's 761 real games (MIT; pinned commit
  and blob recorded; up to about 186 boards), chessin5d.net's offline server and
  the community Discord servers (both need their owners). No public source reaches
  200-600 boards; M7 stays provisional. Nothing downloaded into the project.
- Spec: §15.6 (M7); the register's open item on real game records.

### 2026-10-02 - Build log: CHECKPOINT 2 ruled, the run resumed

- `docs/BUILD_LOG.md`: the STOP section becomes CHECKPOINT 2's record - the rulings
  (M12's 6b re-run and its outcome rules, M13's CPU allocation, Q4 = yes with D8's
  submit action and acting-player input, C23's ancestry, Q5 settled from the
  sources), what is running (engine E7 and E5c, the game-record research), what
  follows, and the two questions put back to the user (Q2, whose sources are
  silent; what "normalise it in the cwmtt driver" means for cwmtt/many2).
- Spec: none (status only).

### 2026-10-02 - The rulebook records the CHECKPOINT 2 rulings on Q4 and Q5

- `docs/rules/README.md`: **R8** - a player may keep moving after the present has
  shifted; the turn ends only at submission (Q4, register D8), with its sources
  (Wikipedia's "The player may also optionally make moves on any playable board
  where it is their turn" and the "Submit Moves" button; wiki.gg's Action page as
  support; the corpus records as client evidence). **R9** - only the active boards
  in the present are mandatory (Q5), settled from the ranked sources as the user
  asked: the in-game Guide's "A player can ignore any timelines that are not needed
  to advance or rewind The Present" outranks Wikipedia's former paragraph. T5's open
  clause, Q4 and Q5 marked accordingly. Q2 stays open: its sources are silent.
- `oracles/gate.toml`: the rulebook re-pinned (H38) after review; no known
  divergence cites T5, T6, Q4 or Q5 (their "RQ4" is CHECKPOINT 1's rule question 4);
  the commit-tier gate re-run: the oracle layer ok.
- Spec: §8.2 and §11.0 item 2 (D8, H38); no spec edit (the documents already say it).

### 2026-10-02 - Design documents: the CHECKPOINT 2 rulings, from upstream

- `docs/design/` replaced with the upstream revision recording the user's rulings at
  CHECKPOINT 2: register rows C23 (the causal past is the board's ancestry), D8
  (turns have variable length - a player may keep moving after the present has
  shifted, until submitting - so submit is an action, with a global acting-player
  input), M12 (6b read per board-count range, never pooled; the scanner at its design
  thread budget; one bounded optimisation pass, then 6b re-run unchanged, with
  outcome rules), M13 (8 cores per GPU: 2 expansion, 1 search and batching, 5
  scanner); real game records listed as open for the window verdict; spec §7.3,
  §8.2, §11.0 item 2 and §15.6 accordingly; the architecture document's note on
  stating a threshold's population and resources. Copied in unchanged.
- Spec: the documents themselves (§7.3, §8.2, §11.0, §15.6; C23, D8, M12, M13).

### 2026-10-02 - Experiment 6b and M9 with legality measured: the run stops for the user

- `docs/MEASUREMENTS.md`: experiment 6b (M8) on a quiet machine with `t5d-scan6b`
  over the corpus. The 50 ms root scan resolves 5 forcing half-turns in 60.9% of
  roots under the engine's default Q4 reading and 49.2% under the other - a fail
  inside the ±10% band - and in the design's regime, above 50 boards, fails under
  both (2.4-34.4% by size). Rung 1 and quiet-leaf reach fail; stand-pat is valid.
  Scanner speed about 1.03 million nodes/s on one core (a §15.3 figure). M9 with
  E5a's fast legality: 20,249 expansions/s at 240 boards on one core, 1.01x the
  target (inside the band), 1.87x on two; 1.21x at 600 boards.
- `docs/BUILD_LOG.md`: the STOP, the decisions it needs, and what is paused (engine
  E6a's search core, in its worktree) - register rows M8, M9.
- Spec: §15.6 (M8, M9), §11.1 (the kill condition), §15.3 (scanner nodes/s,
  movegen throughput); spec issues 20, 61, 62.

### 2026-10-02 - Engine E5b: the tactical scanner and experiment 6b's tool

- `engine/` (merged from its worktree branch; H1-H6, H15, H19): `t5d/scanner.hpp`,
  the verification layer's brute-force auditor - node-capped alpha-beta with
  iterative deepening over forcing actions (a legal action beginning with a forcing
  move; the forcing set, the branch-threat test and the completion rule are
  parameters), a material-count eval (placeholder 5D piece values), net-free
  ordering, no speculative pruning. In check every legal action is searched (E4's
  enumeration, to a cap). Proofs come only from E4's exhaustive verdicts and
  exhaustively enumerated defences, and export at once to a solved store probed
  before the TT, which holds estimates only. A deterministic single-thread mode and
  a stand-pat validity check. `t5d/check.hpp` gains `royal_capture_from`.
- `t5d-scan6b`: experiment 6b's five measures - horizon per node budget, the rung-1
  leaf test, the 50 ms root scan, quiet-leaf reach, stand-pat validity - with every
  condition and breakdowns by board count.
- Tests: 18 by default and 4 long, against E4's detectors and a brute-force game
  tree; no false proof under random node budgets and every alternative ruleset.
- `docs/BUILD_LOG.md`: register rows H1-H6, H15, H19; spec issues 51-62; proposed
  spec edits 44-50. The 6b verdict is measured next, on a quiet machine (the
  preliminary figures, from a loaded one, put the root scan near its threshold).
- Spec: §11.1-§11.3, §15.4, §15.6 (6b, M8), §15.3 (scanner nodes/s); H1-H6, H15,
  H18, H19, M8, D2; spec issues 51-62, proposed edits 44-50.

### 2026-10-02 - Engine: en passant is not a quiet move for the legality tiers

- `engine/src/legal.cpp`: tier 1's argument - a quiet action only adds blockers,
  so its royal captures are among the stripped multiverse's - fails for en
  passant, which empties the victim's square, one the capturer does not land on,
  and can open a line onto the mover's own royal. The classifier then called such
  an en passant legal in every mode (and `legal_moves()` listed it), though it
  begins no legal action. En passant now goes to tier 2's per-move checks. Found
  by an adversarial review that otherwise matched the classifier with an
  independent brute-force reference on 1,135,952 exact classifications (and 3.4
  million budgeted and fast ones, never contradicting) under 20 rulesets and
  combinations; random play had never produced the geometry.
- `engine/tests/test_detectors.cpp`: the review's two positions as a regression
  test (it fails before the fix).
- Spec: §11.0 item 2 (C4, R1); no spec text changes.

### 2026-10-02 - Engine E5a: the legal moves in tiers - exact, budgeted, fast

- `engine/` (merged from its worktree branch): a move's legality (C4, R1) is decided
  by sound tiers (`t5d/legal.hpp`, `classify_moves`) instead of a search per move:
  a standing royal capture (no move legal); a stripped multiverse that certifies
  quiet moves and branches in bulk, a passed board's standing capture refuting the
  moves that leave it, a finish certifying the quiet moves on the boards it plays;
  per-move checks; and search last. Modes: exact (default), budgeted (legal /
  illegal / unknown - never illegal without proof, H37's spirit) and fast.
  `ExpandOptions::legality` sets the expansion's mode; unknown moves are kept and
  marked. The hypercuboid's node budget now counts each step of its abstract check,
  so a node budget bounds time.
- `t5d-bench --legal` (`--legal-budget`, `--legal-profile`) and `t5d-measure
  --legal`: |E| from legal destinations (M7's tighter bound).
- Tests: the classification against the brute-force reference and a search per move,
  0 disagreements in about 113,000 long-run classifications; the long run takes
  minutes instead of an hour.
- `docs/MEASUREMENTS.md`: the legality profile (shared machine; M9 with legality
  still needs its quiet-machine run) and M7 from legal destinations (the 160-239
  bucket's median |E(8)| falls from 15.5 to 4.5; still no verdict).
  `docs/BUILD_LOG.md`: step 4, spec issues 28 and 46 (what search does with an
  unknown move - a proposal), proposed spec edit 43.
- Spec: §11.0 items 2-3 (C4, R1, H37), §15.6 (M7, M9); spec issues 28, 46; proposed
  edit 43.

### 2026-10-02 - Network N4: the engine binding; real positions as the network's input

- `python/t5d/engine/` (new): a ctypes binding to the engine's C API - the library
  found by an explicit path, `T5D_ENGINE_LIB` or `build/engine/`; games from a
  5dpgn record, 5DFEN or a variant; replay, apply, submit; the engine's move list
  (pseudo-legal - the C API has no legal-moves switch yet); each expansion copied
  out of the engine's buffers at once; one exception per status with the engine's
  message. `convert.py` turns an expansion into a validated `PositionState`
  (features, tokens, move graph, the attack/defence/x-ray relations, candidate
  moves in the engine's order - logit i scores engine move i -, the lineage parent,
  the W/E scope as a token subset) and refuses a state whose present has shifted
  inside an unsubmitted action (spec issue 48). `evaluator.py`: dense and cached
  network evaluators and the serving loop of the search's protocol; the adapter for
  engine E6a's step API is marked and waits for it. `play.py`, `bench.py`.
- `python/tests/engine/` (new, 72 tests; they skip without the library): hand-
  computed 5DFEN positions; over 100 real positions from 10 variants, random play
  and three records, each board checked against an independent decoding of the
  engine's canonical 5DFEN; the network (F1 and a small one) on real positions,
  dense and cached (cached = one full block-causal pass). Opt-in, `T5D_RECORDS`:
  every position of named record files (the gate corpus: 1,010 positions, all
  converted and checked).
- `.github/workflows/network.yml`: builds the engine's shared library first, so the
  binding's tests run on Linux; the job also runs when `engine/` changes.
- `docs/BUILD_LOG.md`: spec issues 48-50, proposed spec edits 41-42.
- Spec: §2.1-§2.3 (A1-A3), §6 (B9, B11), §7.3 (C17's lineage), §8.1 (D1), §8.3 (D3),
  §10.4.1 (W, E); spec issues 48-50, proposed edits 41-42.

### 2026-10-02 - The engine's first gate runs triaged; the first known divergences

- `oracles/known-divergences.toml`: 56 entries from the triage of the engine's
  first gate runs (engine configured locally only): 37 for ftxi's and ftxi:naive's
  check verdicts that ruling R1 rules wrong; Chess in 5D's two false checkmates
  (2) and ftxi's missing Black two-timeline double step (4) - the two oracle
  defects, recorded as telltales as ruling R7 asks; 7 for record text the notation
  specification rules out (jumps without a source square, a parenthesised
  variation, 5DFEN boards under a named Board), which the engine refuses and an
  oracle played; 6 for oracles' own notation errors (ftxi's missing unmoved markers
  on Very Small - Open, ftxi's refusal of `+1` timeline labels, the notation
  parser's refusal of `(0T1)a2a3`). Each cites the rulebook or the notation
  specification. `verify` passes; with the engine configured the nightly gate
  matches every reachable entry, none stale, no telltale.
- No engine defect was found. Standard - Half Reflected is the first gated ruleset.
  115 findings that rules Q4 explains remain, so the engine stays unconfigured in
  the committed `gate.toml` until the project owner rules on Q4.
- `docs/BUILD_LOG.md`: the triage (step 2), the plan, register rows H23, H34.
- Spec: §11.0 "Running the gate" (H23, H31, H34, H36, H38; R1, R4, R7); no spec
  text changes.

### 2026-10-02 - Engine: a turn number that does not count the turns is a warning

- `engine/`: by default (`PgnOptions::turn_number_mismatch_is_error`, false) a turn
  number that does not count the turns is kept in `PgnRecord::warnings` (and
  `PgnReplay::warnings()`, printed by `t5d-gate`) and reading goes on; a turn
  really missing still shows, since the next action's moves are checked like any
  other. notation/aivsai (turn 25 written twice) now replays all 54 actions. The
  same policy as the annotations below: the gate compares more, never less.
- Spec: §11.0 "Running the gate" (H28: what the engine reads); the notation's
  "Turns"; no spec text changes.

### 2026-10-02 - Engine: a disagreeing annotation is a warning, not a refusal

- `engine/`: a `~`, `(>Lx)` or `(~Tx)` that disagrees with the state no longer
  makes the move unreadable by default (`PgnOptions::annotation_mismatch_is_error`,
  false): the move is played and the disagreement kept in `PgnReplay::warnings()`,
  which `t5d-gate` prints on stderr; `--strict-annotations` restores the refusal.
  The notation specification calls these tokens "purely for better human
  comprehension" that "should match" the state; a wrong one is the writer's slip,
  and refusing hid the rest of three corpus records from the gate: cwmtt/silly now
  replays all 38 actions (CHECKPOINT 1's CHK-A1 representative, position 28, is
  not check under R1 - 37 legal actions, as ftxi and cwmtt count) and
  chessin5d-illegal-move-example all 66 (position 66: in check, 25 legal actions,
  not the checkmate Chess in 5D's mate search reported); cwmtt/wide now stops at
  rules Q4 instead. Tests: the strict reading kept for the annotation tests, a new
  test of the default.
- Spec: §11.0 "Running the gate" (H28, H30: what the engine reads); rules T15, Q26;
  no spec text changes.

### 2026-10-02 - Phase 1 measurements: movegen throughput (M9) and the corpus |E| curve (M7)

- `docs/MEASUREMENTS.md`: M9 measured on a quiet machine with `t5d-bench`.
  Pseudo-legal expansion (moves, move graph, W/E scope, typed relations) runs at
  44,047/s at 240 boards and 24,951/s at 600 on one thread - 2.2x and 2.5x M9's
  targets, outside the ±10% STOP band; the frozen-history term is 0.48 ns per board
  (2% of an expansion at 600 boards); matched flatness of the spec's expansion is
  +7% median, inconclusive by rounds. With the legal-moves filter the expansion
  takes milliseconds to minutes: a fail to fix before search. M7: the corpus curve
  recorded (upper bounds); no verdict - real games reach 240-399 boards in two
  games and run 16-24 active timelines.
- `docs/BUILD_LOG.md`: register rows M7, M9; the plan.
- Spec: §15.6 (M7, M9), §15.3 (movegen throughput); spec issues 20, 21, 26, 46.

### 2026-10-02 - The gate, step 2B: stable answers, integrity, telltales, parse mode, the rulebook pin

- `oracles/` (H32): every source is recorded twice (`golden --write --repeat 2`);
  a value that differs between recordings is unstable and compared by nothing, one
  a limit stopped is "unknown at budget X". Per-position costs go in
  `golden/<source>.cost.json`. A tier requires exactly the stable values whose
  recorded cost x `[stability] margin` fits its budget - and 100% of them: the
  coverage floors and the proof channel's 90% ratio are gone, replaced by a
  static non-vacuity check. A calibration probe reports how much slower a runner
  is than the recording machine. The golden files are re-recorded (schema 3).
- H33: `golden/MANIFEST` holds every recorded file's SHA-256, verified before any
  comparison; each gate run loads one verified snapshot (closing the window in
  which an engine could append a known divergence excusing itself); the harness's
  files are re-hashed after the engine runs and a change fails the run;
  `harness.py` imports its own modules from source only, so a planted bytecode or
  extension module is never loaded. `.github/CODEOWNERS` owns `/oracles/` and
  `/.github/`.
- H34: an engine answer that agrees with a known divergence's recorded oracle
  error, by the comparison's own rule, is a telltale: it fails the gate and is
  never reported stale.
- The engine's parse mode (`[engine] mode = "parse"`) compares states, replay
  length and acceptance, never check, counts or mate, and never enables the proof
  channel. In either mode an engine's `rejected`/`unreadable` - after positions,
  or refusing a set-up - is a soft legality divergence that only a notation
  citation can excuse, and the ruleset stays ungated.
- The CHECKPOINT 1 rulings: `in_check` means H36's check status; a source that
  names a search limit and still gives a verdict is malformed (H37); the rulebook
  is pinned in `gate.toml` (`harness.py rulebook`, `rulebook --pin`), and `verify`
  fails when it changes until it is re-pinned (H38).
- `legal_actions` is defined: an action is its set of moves, with its
  timeline-creating moves in the order they create timelines - two orders of two
  branching moves are two actions. ftxi and cwmtt list exactly that at every
  position both read; a protocol definition, not a rules ruling.
- `.github/workflows/oracle-commit.yml`, `oracle-nightly.yml`: the engine's gate
  adapter is built before the gate, and the build is checked to have changed
  nothing under `oracles/` (the harness-modified check only sees changes made
  while the gate runs).
- Reviewed adversarially (17 findings, all applied). 601 harness tests. The golden
  files were re-recorded on a quiet machine (the commit tier's calibration probe
  then ran at 1.00x the recording machine's time); no recorded answer changed.
- `AGENTS.md` and `CONTRIBUTING.md`: the forbidden ways of making the gate pass now
  name budgets, the margin, `proof_share`, cost files, `calibration.json`,
  `MANIFEST`, re-recording on a loaded machine and moving the rulebook pin
  unreviewed; `CLAUDE.md`: the two-recording command and `rulebook`.
- Spec: §11.0 "Running the gate" (H32, H33, H34, H36, H37, H38); proposed spec
  edits 24-26 in `docs/BUILD_LOG.md`.

### 2026-10-02 - Engine: the hypercuboid lets a castling precede moves that rewind the present

- `engine/src/hypercuboid.cpp`: the general realisation (alternative rulesets only)
  made the moves that create no timeline in ascending order between creations,
  assuming they commute. Under Q3 (activation by play), Q4's other reading and
  Q21's alternative (castling on mandatory boards only) they do not: a move on an
  inactive opponent timeline behind the present rewinds the present off the
  castling board, so the castling must come first. The hypercuboid then
  under-counted (238 of 254 actions in the reviewer's example; 165 of 400 positions
  of a random family), which the gate's cross-check would have reported as a
  detector disagreement; no wrong verdict was found. A castling is now exempt from
  the order. Found by a follow-up adversarial review that otherwise matched both
  detectors with an independent brute-force reference on 21,717 positions and
  73,013 mid-action states under 22 ruleset configurations, H37 holding under
  17,860 budgeted searches and at the memory ceiling.
- `engine/tests/test_detectors.cpp`: the reviewer's position as a regression test
  (it failed before the fix). `engine/README.md`: the realisation's description.
- Spec: §11.0 item 3 (H37); no spec text changes.

### 2026-10-02 - Engine stage E4: check, legality, legal actions, mate detection, the gate's full mode

- `engine/` (merged from its worktree branch): check status and submission
  legality exactly as rulebook R1 and H36 state them - one royal-capture scan with
  the move generator's semantics, held equal to it on random multiverses; the
  default check ruling of `GameState::submit` and of the replay (an action that
  leaves a royal capturable is `cannot_submit`; `CheckRuling::none()` keeps
  pseudo-legal play). `CastlingAttacks::AnyBoard` removed (H40).
- Legal actions: `count_legal_actions` counts the harness's actions (a move set,
  its timeline-creating moves in creation order); `count_legal_successors` counts
  distinct successor states; `list_legal_actions` lists them.
- Mate detection (C5, C8, C9, C13; H37): a naive depth-first detector and a
  hypercuboid detector designed in-house from spec §11.0 item 3 (design note in
  `engine/README.md`; no oracle source read), realising points by regime and
  eliminating whole boxes by an abstract check. Checkmate and stalemate only from
  an exhaustive search; any node, time or memory budget answers "unknown", tested
  for every node budget below the exhaustive need. Every search has a memory
  ceiling (unbudgeted naive diagnostics had reached 48 GB).
- `t5d-gate` full mode: in_check, legal_actions, terminal - unbudgeted, via a
  hypercuboid-then-naive portfolio on doubling budgets with a cross-check (a
  disagreement ends the run in `error`); diagnostics for hand investigation. The
  whole corpus runs in 8.3 s.
- `ExpandOptions::legal_moves` (off by default): the expansion's moves restricted to
  those that begin a legal action; too slow to turn on (spec issue 46).
- 355 tests, plus long cross-checks (`t5d-tests --long`) against a brute-force
  reference at turn starts and mid-action - 4,000 positions and 17,207 mid-action
  states, no disagreement. Reviewed adversarially three times: no soundness bug
  (an independent brute-force reference matched both detectors on about 6,660
  positions under 15 rulesets); the search-time defect, the count definition and
  the test gaps they found are fixed, and the fix round's reference caught and
  fixed a naive miscount.
- `docs/BUILD_LOG.md`: step 4, register rows H22, H36, H37, H40, spec issues 44-47,
  proposed spec edits 38-40. `docs/MEASUREMENTS.md`: the full-mode corpus timings
  and the legal-moves filter's cost.
- Spec: §11.0 items 1-5 (H22, H36, H37, H40), §15.6 (M9: the filter), spec issues
  28 and 44-47, proposed edits 38-40.

### 2026-10-02 - Network CI: the FP32-resolution FP8 test absorbs one rounding flip

- `python/tests/net/test_cache.py`: the test that the FP8 cache equals the FP8
  network's forward at FP32 resolution bounded the policy's largest logit gap at
  5e-4, tighter than one FP8 rounding flip, which a platform's FP32 summation order
  can tip: on Linux one flip moved a logit by 1.3e-3 and the network job failed
  (557 of 558 passed). The bound is now 5e-3; the trunk-RMS and value bounds,
  which separate a real fault from a flip by about 400x (mutant M14 gives trunk
  RMS 7.4e-3 against 1.2-1.8e-5), are unchanged.
- Spec: §15.2 row (e) - tolerance matched to the arithmetic, as its protocol says.

### 2026-10-02 - The network stack, stage N3: the block-causal scope and the cached path (random weights)

- `python/t5d/net/`: the block-causal scope (C17, C19) as `NetConfig.attn_scope`
  (`dense`, the default per C1; `block_causal`), through every attention path with
  rows built per block (`scope.py`); lineage on the input contract
  (`PositionState.parent` - the arrival side - `creation`, and `source`, a
  travel's departure board; validated, remapped, carried on `Batch`, recorded by
  the synthetic generator and the tree fixture). Committed-class points are
  parameters with placeholder defaults: `causal_past` (lineage | two_parent |
  creation), `frozen_features` (intrinsic | current), `supersession_scope` (C20's
  two readings), `frozen_kv_format` (match | fp8, applied by the forward itself).
  Under the scope the §6 structure opens live rows only.
- The cached path (C2 as built for §15.2 and §15.3, `cache.py`): per-board K/V
  content-addressed by (epoch, dependency digest) - §7.2's radix structure; every
  frozen board computed one at a time in its own frame, so evaluations are bitwise
  path-independent; supersession recomputed once under the state-defined mask
  (C20), never snapshotted (C21); the live pass per node; root refresh (C3) and
  the rung-1 node refresh (C3a, V = 50) forking epochs copy-on-write, history
  membership by intrinsic identity; storage in the network's own frozen-K/V format
  (G1). A search-tree fixture (`tree.py`), references with the two-level mask, the
  spec's cost model with counted FLOPs (`cost.py`), and a benchmark that measures
  its own machine's load and refuses a busy one (`cache_bench.py`).
- 111 tests for §15.2's network rows (a)-(e) over adversarial geometries; 558
  pass. Reviewed adversarially twice (equivalence and fidelity; cost method): 12
  findings, all addressed. 27 injected mutants, every behaviour-changing one
  caught.
- `docs/MEASUREMENTS.md`: the counted §15.3 figures at the spec's timeline counts
  and at the corpus's 24 (CPU wall times wait for a quiet machine).
  `docs/BUILD_LOG.md`: register rows C1-C4, C12, C15, C17, C19-C21, G1; spec
  issues 29-43; proposed spec edits 27-37.
- Spec: §2.3, §6, §7.1-§7.3, §7.6, §7.9, §8.5, §8.7, §10.2, §10.3, §11.3, §15.2,
  §15.3; A3, A4, B9-B12, C1-C3, C3a, C11, C15, C17, C19-C21, G1, H18. Spec issue 5
  is built as parameters (the choice is the user's); spec issues 29-43; proposed
  edits 27-37.

### 2026-10-02 - Engine CI: t5d-bench builds on Linux

- `engine/tools/t5d_bench.cpp`: the synthetic seeds are computed with
  `std::uint64_t` literals. With `ull` literals, g++ 13 and 14 on Linux - where
  `uint64_t` is `unsigned long`, not `unsigned long long` as on MinGW - warn under
  `-Wsign-conversion`, and the engine job failed on E3's commit.
- Spec: no part of the spec describes the build.

### 2026-10-02 - Engine stage E3: expansion, its C API, the Phase 1 measurement tools

- `engine/`: `expand()` (`t5d/expand.hpp`, `src/expand.cpp`, `t5d/records.h`)
  produces, for the player to move, the pseudo-legal moves (with capture, travel
  and branching flags), the move graph (B9), W(k) - each active timeline's band of
  k half-turns plus branch origins, reported apart - and E(k), the out-of-window
  travel destinations (§10.4.1), with source boards outside W counted apart;
  side-to-move typed relations behind flags (attack, defence, x-ray - B11,
  provisional; pin and check are E4's; travel-threat is undefined); board and
  square features in `python/t5d/net/state.py`'s layout, plus the lineage parent.
  No per-frozen-board work - per-board arrays are computed on first use - and no
  allocation in steady state. Moves are chosen in one place, ready for E4's legal
  actions.
- C API games: from a record, 5DFEN or a variant; replay, apply and submit under a
  per-game check ruling; expand into flat arrays.
- Tools: `t5d-measure` (per-position W/E at k = 0/4/8/16, and `--summary`, a curve
  de-duplicated by position and game) and `t5d-bench` (search-like streams,
  interleaved rounds, thread CPU time, thread scaling, raw / matched / no-move
  flatness series, reported as ranges and labels, never verdicts); a seeded
  generator of SYNTHETIC positions for throughput only.
- 40 new tests (299), including a ruleset-aware from-scratch reference; reviewed
  adversarially (19 findings, all dealt with: 18 fixed, one a question about the
  brief's STOP band); 20 injected mutants killed.
- `docs/BUILD_LOG.md`: step 5's state, preliminary figures (loaded machine - not
  measurements), the reading of the brief's ±10% STOP band, spec issues 17-28,
  proposed spec edits 19-23.
- Spec: §6 (B9, B11), §10.4.1-§10.4.3 (C5, C7 - measurement only; no window
  code), §11.1, §15.6 (M7, M9); spec issues 17-28 and proposed edits 19-23 in the
  build log.

### 2026-10-01 - Upstream drafts of the two oracle defects; network CI fix

- `docs/checkpoints/upstream/`: draft bug reports for the user to file (rulebook
  R7) - 5d-chess-js's exhaustive mate search passes a search-tree node where a board
  is expected, generates no moves and reports checkmate where a legal turn exists;
  ftxi's move generator never gives a Black pawn or brawn the two-timeline double
  step. Each has a hand-built minimal record verified on the pinned code, the root
  cause at the pinned commit, a fix checked in a scratch copy, and the rule from
  public sources; the README lists what to check before filing.
- `python/t5d/net/heads.py`: the value head's bin centres are computed from
  integers. `torch.linspace` put the middle centre at 2.2e-8 on Linux, not 0, and
  the first Linux run of the network job failed on it (446 of 447 passed); a draw
  must sit exactly on a centre.
- `.gitignore`: agent worktrees (`.claude/worktrees/`).
- `docs/BUILD_LOG.md`: the 18 Q1 positions confirmed unmoved under the count rule
  from the checkpoint's data; the drafts' state and two further ftxi leads.
- Spec: §11.0 "Triage by cluster" and H31 (the defects found by triage);
  §8.3 / D3 (the value head's 51 bins - no spec change); nothing else.

### 2026-10-01 - The network stack, stages N1 and N2 (random weights)

- `python/` (package `t5d`, PyTorch on CPU; `cd python; py -m pytest`), built to
  spec §2-§9 with random weights - no training run:
  - N1, dense v1: the token layout (66N + 16, board-major) and its input contract
    (`t5d/net/state.py`, the boundary the rules engine will fill); piece and
    per-board features with Δt-to-present; 4D RoPE with asymmetric pairs in FP32;
    2 per-board and 16 global registers; the trunk (d 256, L 16, 8 heads, FFN 8x)
    under BF16 autocast; the bilinear pointer policy over candidate moves, the
    51-bin value head and the auxiliary heads; 21,358,601 parameters (F1) and the
    L24-ff12 preset (F2); synthetic positions and a training smoke test of a few
    optimiser steps on synthetic data (allowed by the brief - no training run).
  - N2, the attention structure: the move-graph mask on 2 heads (B9); the routing
    table (B10), typed edge biases (B11) and board-pair smolgen (B12, full rank
    over relative board-pair cells) behind flags; B14's block-sparse attention in
    a CPU gather form, the default, with per-block recomputation under autograd,
    and chunked and dense paths as references.
  - 447 tests: masks against brute-force key sets, chunked and block-sparse equal
    dense (outputs and gradients), invariances (board order, translation, batch
    composition), precision bounds, parameter counts, no inert switch; 73
    injected mutants across the two stages, all killed. Each stage was reviewed
    adversarially and every finding (15 for N1, 12 for N2) fixed.
- `.github/workflows/network.yml`: lint and test on ubuntu-24.04 when `python/`
  changes, with PyTorch 2.13.0 (CPU), NumPy 2.5.2 and ruff 0.16.5.
- Spec: §2.1-§2.3, §3.1-§3.2, §4.1, §5, §6, §6.1, §8.1, §8.3, §9.2, §10.1, §10.2,
  §12.4, §13.1 item 6, §14.4; A1-A3, B1-B2, B4-B6, B8-B12, B14, D1, D3, D4, F1,
  F2, G1. Committed-class ambiguities are parameters with placeholder defaults
  (spec issues 9 and 16); proposed spec edits 5-9 (N1) and 12-18 (N2) in
  `docs/BUILD_LOG.md`.

### 2026-10-01 - The rulebook records the CHECKPOINT 1 rulings

- `docs/rules/README.md` is now the project's rulebook (H38): a preamble ranks the
  sources (in-game Guide > cited developer rulings > established references >
  forums), labels silent cases as project decisions, never cites an oracle, and
  says the gate pins the file. A new **Rulings** section records R1-R7 with the
  sources the user cited: check status and submission legality (R1, H36);
  activation is the count rule (R2, closing Q1); the rulebook itself (R3, H38);
  the pawn's double step along both axes for both colours (R4, Q15 in part); no
  repetition rule, a project decision, and self-play's turn cap a convention (R5,
  H39, I11); castling and en passant attack tests per-board (R6, H40); the two
  oracle defects go upstream (R7). The entries they settle say so: T10, 4.6, C3,
  C4, C6, C7, D2, S2, S7, Q1, Q10, Q15, Q21, Q23. One addition is marked
  *derived*, not ruled: a pinned attacker counts (C3, from R1's "could capture").
- `docs/BUILD_LOG.md`: CHECKPOINT 1 marked ruled; step 4 unparked; the parked list
  rewritten; the engine's parse-mode reading of the corpus classified (Q4, the
  annotation policy, notation outside the spec); register status for H31 and
  H36-H40, I11; spec issue 10 and proposed edit 10 resolved upstream.
- Spec: §11.0 items 1-5 and "Triage by cluster" (H31, H36-H40), §12.3 (I11) - in
  the documents already; no proposed edit.

### 2026-10-01 - Design documents: the CHECKPOINT 1 rulings, from upstream

- `docs/design/` replaced with the upstream revision that records the user's
  rulings on CHECKPOINT 1: register rows H36 (check status - pass on all active
  boards in the present - versus submission legality - any royal capture from any
  of the opponent's playable boards after the actual turn), H37 (a mate verdict is
  a proof: an exhaustive search that terminated; an exhausted budget is
  "unknown"), H38 (`docs/rules/README.md` is the rulebook), H39 (no repetition
  rule, a project decision), H40 (castling and en passant attack tests are
  per-board), I11 (self-play's turn cap, a convention scored as a draw); spec
  §11.0 items 1, 2, 3, 4 and 5 rewritten accordingly, §8.5, §8.7, §10.4.3, §12.3,
  §12.6 and §14.4 follow from H39, reversal 21 (castling's attack test) recorded,
  and the architecture document gains the append-only-state and terminal-proof
  notes. Copied in unchanged; nothing here edits them.
- Spec: the documents themselves (§8.5, §8.7, §10.4.3, §11.0, §12.3, §12.6, §14.4;
  H21, H36-H40, I11, part J reversal 21, part K).

### 2026-10-01 - Engine CI: build on g++ 13 and 14

- The engine's first Linux run failed: g++ 13 and 14 at -O2 warn, wrongly, that
  copying a disengaged `std::optional<Colour>` may read its payload
  (`-Wmaybe-uninitialized`), and warnings are errors. `engine/tests/test_zobrist.cpp`
  now assigns the optional only when it holds a value.
- `engine/build.py --keep-going` builds every target it can, so one run reports
  every compile error; the engine workflow uses it.
- Spec: no part of the spec describes the build scripts.

### 2026-10-01 - The rules engine, first stages (E1, E2): representation, turn rules, moves, parser

- `engine/` (C++20, CMake; `py engine/build.py --test`; manual `engine/README.md`),
  written from `docs/rules/README.md`, the design spec and the notation
  specification's text - never from an oracle (H24, H26, H28):
  - E1: immutable boards (A4) in a persistent multiverse; 5DFEN read and write;
    the H30 fingerprint (the notation spec's full-state hash; both published
    hashes reproduce) with an MD5 written from RFC 1321; the 5D Zobrist key (H18:
    a per-board content hash mixed with (timeline, half-turn) by multiply-rotate,
    combined by addition so duplicated boards never cancel, with the ruleset ID,
    player to move, parity, creators and board size); a C API for Python.
  - E2: the present, timeline activation and reactivation, the turn state machine
    and submission's present-shift test (T1-T13); pseudo-legal moves for every
    piece in 4D (4.1-4.13, J1-J9); move application with branching and timeline
    numbering; castling with the on-board attack test (S2), en passant within a
    board, promotion; the 5dpgn/5DFEN parser and the variants that have an
    independent source; `t5d-gate --parse-only`, the engine's gate adapter.
  - Every open question of the rules reference is a named ruleset parameter with
    the documented default. Check detection, the royal-safety test at submission
    and mate detection are hooks only: they wait on the check ruling (CHECKPOINT 1).
  - 259 tests, warning-free with -Werror, also under an undefined-behaviour
    sanitizer; the parser survived 300,000 fuzzed records. Each stage was reviewed
    adversarially (the one major finding - every move copied its timelines' whole
    history - is fixed by sharing history in chunks).
- `.github/workflows/engine.yml`: builds and tests the engine on ubuntu-24.04 with
  g++ 13 and 14, warnings as errors, without checking out any submodule.
- `CLAUDE.md` describes the engine; `PROVENANCE.md` notes that its MD5 follows RFC
  1321 and its Zobrist key generator uses published constants, with no code
  copied; `.gitignore` ignores the build tree and the project `.venv`.
- Spec: §11.0 (the rules engine's five pieces: items 2, 4 and 5 built; 1 and 3
  parked), §11.3 (the Zobrist key - proposed spec edit 2), H18, H22, H24, H28,
  H30; spec issues 11-15 and proposed spec edit 11 in `docs/BUILD_LOG.md`.

### 2026-10-01 - CHECKPOINT 1: the check disagreements, clustered and traced

- `docs/checkpoints/checkpoint-1-check-clusters.md`: the 78 check disagreements
  between ftxi and Chess in 5D (all at positions where both hold the same state,
  H30) in seven clusters by the attacker's situation, four rule questions, every
  oracle's verdict at a representative position drawn board by board, and the
  traces of the two mate disagreements (a Chess in 5D mate-search defect) and of
  the 52-vs-54 count (an ftxi move-generator defect for black pawns' two-timeline
  steps, beside rules question Q15). Written by a collector, a clusterer, three
  skeptics (16 findings, all applied) and a finaliser, from fresh runs of the
  pinned oracles; nothing adjudicated, no oracle's side taken.
- `docs/BUILD_LOG.md`: the checkpoint summary, the questions put to the user, and
  everything parked until the check ruling.
- Spec: §11.0 "Triage by cluster; settle by rulebook" (H31) - the triage is done;
  the settling waits on the user's ruling and the rulebook question (proposed spec
  edit 10). No spec text changes.

### 2026-10-01 - The tiered CI is green on Linux (H35)

- Pushed to the remote for the first time. The commit tier (run 36827435595) and
  the nightly tier (run 36827459456, started by hand) both passed on ubuntu-24.04
  at their first run: the golden files recorded on Windows reproduce exactly on
  Linux (ftxi, cwmtt, Chess in 5D and the notation parser 100%). The naive
  searches reproduce less on the slower runner (ftxi:naive 95.9%, cwmtt:naive
  98.4%) and pass only by the coverage floors that H32 replaces next.
- `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md` and `oracles/README.md` no longer
  say the workflows have never run; they say the workflows run on Linux and are
  not yet required checks on the branch.
- Spec: §11.0 "Linux CI before engine code" and §15.1 (Phase 0 order) - H35 is
  met; register H35 is done. No spec text changes.

### 2026-10-01 - The gate matches states, not move counts (H30)

- **Every source now says what it set up, at every position.** Before the position
  event and any search, each driver prints a `state` event listing every board it
  holds (`L:T:c:placement`). The harness alone (`oracles/oracle_harness/fingerprint.py`)
  validates, canonicalises and hashes it: `fingerprint` is the notation
  specification's full-state hash (MD5 of the sorted `[placement:L:T:c]` strings,
  -0 before +0, then turn, then white before black - the notation is silent on the
  colour order), `placement` the same without unmoved markers, plus board and
  piece counts. Both published hashes of the notation spec are conformance tests.
- **Parse divergences are their own class** (`placement`, `unmoved`). A placement
  divergence ends every comparison of that pair from that position on, and a
  known-divergence entry for one must cite the notation specification; a
  marker-only run is reported per run and stops the confirmation, not the
  comparison. The engine contract now requires a `state` event at every position.
- **Drivers.** ftxi: the board walk of its own API, held to the stock `5dtools`
  (`stats` board count and size, and its `print` drawing read back into a
  placement hash). Chess in 5D: the library's own full-state export, and its own
  hash must equal the harness's at every position; custom set-ups it relabels are
  refused. cwmtt: a new Haskell state driver (`oracles/drivers/cwmtt/Driver.hs`)
  compiled against cwmtt's modules runs beside the stock CLI, which still answers
  the counts; the adapter pairs the two only where they read and replayed the
  record alike, and cwmtt now reports the side to move. New: the notation
  repository's bundled parser is a parse-only oracle, `notation` (decision H26, as
  the brief asked), run through `oracles/drivers/notation/driver.js` in every tier;
  the commit workflow now sets up Node for it.
- **Diagnostics** for the check-cluster triage, never compared or recorded:
  `harness.py observe --explain` (ftxi: the timelines it advances to phantom boards
  and the checking moves it finds) and `observe --actions K` (every source's legal
  actions at position K, canonicalised for diffing).
- **Golden files re-recorded (schema 2)**: four new columns per position, and a
  `notation.json`. No answer of ftxi, Chess in 5D or cwmtt changed. cwmtt's
  positions now carry the side to move (1,598 filled, no conflict). A run cut by
  its time budget now keeps the position it was searching, with its boards and no
  answers, so the timed-out naive runs show one more position each. One recorded
  answer was lost to time: cwmtt:naive's final verdict on `cwmtt/ctp1` (about
  257 s of a 300 s budget on this desktop) timed out under load - the
  budget-limited case H32 classifies next.
- **Result:** across the whole corpus, no two sources hold different boards at any
  shared position, so every check, count and mate disagreement in
  `oracles/DISAGREEMENTS.md` is a rules disagreement. The one parse-layer difference
  is markers-only on Very Small - Open (ftxi's built-in layout holds the kings and
  rooks as moved), which leaves that variant ungated until settled. `verify`
  clean; `gate --tier commit`: ftxi 7760/7760 and notation 2765/2765 recorded values
  reproduced; selftest 454 tests.
- Also: `CLAUDE.md`, `AGENTS.md` and `CONTRIBUTING.md` list `fingerprint.py` among
  the files whose edit stales the golden files, and the cwmtt state driver among
  those that need a rebuild.
- Spec: §11.0 "Running the gate" (H30) and its oracle table (the notation parser as
  a parse-only oracle in every tier, H26/H27) - proposed edit 3 in
  `docs/BUILD_LOG.md` (the fingerprint, its readings, marker-only runs, the oracle
  hash check); spec issue 2h (what "unmoved" means). Register H30 is built; its row
  is unchanged.

### 2026-10-01 - A cited rules reference, from public non-oracle sources

- `docs/rules/README.md`: the rules of *5D Chess With Multiverse Time Travel* as
  testable statements (lattice and coordinates, turns and the present, active
  timelines, super-physical moves and branching, every piece's movement, castling,
  en passant, promotion, check, checkmate, stalemate, softmate, draws and
  repetition, variant starting positions), each with a confidence and verbatim
  citations, and 28 open questions where sources are silent or disagree. No rules
  implementation or its documentation was used, and no oracle was run or its
  output read: the reference has to be able to judge the oracles. Its official
  sources are the game's in-game Guide texts, read as text from the client
  installed on the development machine (no program logic inspected; the client is
  never an oracle, H25), and developer posts; community sources are weighted by
  author, so that implementers whose independence is unknown never carry a rule
  alone. Written by five researchers, a synthesizer, three citation skeptics (68
  problems found and fixed) and a reviser.
- Why: nothing in the repository pinned a rulebook (spec issue 1 in
  `docs/BUILD_LOG.md`); the engine's rules are written from this reference, and
  its open questions - which boards a check test passes (Q10), activation (Q1-Q3),
  repetition (Q23) - go to the user as rulings.
- Spec: §11.0 says disagreements are settled "by rulebook" and names none; the
  reference is proposed as the project's rulebook in `docs/BUILD_LOG.md`
  (proposed spec edit 10), pending the user's choice.

### 2026-10-01 - Project licence recorded (M11); the build log; spec edits proposed, not made

- `PROVENANCE.md` records the project's licence: the GPL-3.0, the `LICENSE` file
  from the repository's initial commit (register M11). Its obligations attach on
  distribution of source or binaries, not on private use or on running the engine
  as a service; the BSD-2 code adapted into the ftxi driver is compatible, with its
  notice kept; Stockfish and Lc0 code stays out to keep sole copyright (H29);
  whether the licence reaches trained weights is open, to decide before any are
  distributed. Its section on where the spec and the pinned reality differ now
  says which differences the current spec has corrected (J20, H26) and which
  remain.
- `docs/BUILD_LOG.md`: the resume document for the untrained build (spec §15) -
  the standing rules of the build, what is done, parked and next keyed by register
  ID, the spec issues found, and the spec edits proposed. `docs/MEASUREMENTS.md`:
  every measured figure with its conditions, and the §15.6 verdicts.
- The spec and the register are maintained upstream: `AGENTS.md`,
  `CONTRIBUTING.md` and `CLAUDE.md` now say they are never edited here - an edit
  they need is proposed in the build log and applied upstream, and a measured
  figure goes into `docs/MEASUREMENTS.md` with its write-back (M4) proposed.
- Spec: M11 is recorded as the register states it; no spec text changes.
  Proposed: §11.0 "Tiered CI" should say ftxi runs through a C++ driver, not its
  Python bindings (`docs/BUILD_LOG.md`, proposed spec edit 1).

### 2026-10-01 - CI: oracle builds cached, toolchains installed only for stale builds

- `.github/workflows/oracle-nightly.yml`: the oracle builds
  (`oracles/.cache/build`) are restored and saved as separate steps - saved as
  soon as they are built, so they survive a run whose gate then fails (a single
  cache step saves only when the whole job succeeds). After a driver or recipe
  change the newest older cache is restored, and the harness rebuilds only the
  builds whose own keys moved. GHC 9.10.3 is installed only when a stale build
  needs it - only cwmtt's does - so a nightly run with a current cwmtt build
  skips the heaviest toolchain entirely.
- `oracles/.cache/build/ftxi` in `oracle-commit.yml` is restored and saved the
  same way. That workflow can now also be started by hand.
- New: `python oracles/harness.py build --plan` builds nothing and prints, as one
  line `tools=<names>`, the build tools the oracles without an up-to-date build
  still need (`build.tools_needed()`, tested in `oracles/tests/test_build.py`).
  The workflow asks the harness instead of naming an oracle: CI configuration may
  not name an oracle's package, and cwmtt's repository name is its Haskell
  package name, which the boundary check refuses there.
- Why: H35 asks for the tiered CI green on Linux, and the nightly tier has to stay
  cheap enough to run every night.
- Spec: §11.0 ("Tiered CI") describes the tiers, not how CI caches; not changed.

### 2026-10-01 - Agent permissions: push allowed, force-push and GitHub file writes denied

- `.claude/settings.json`: `git push` is allowed without a prompt, from both the
  Bash and the PowerShell tool. Force pushes are denied, and deny rules beat
  allow rules: `--force` and `--force-with-lease` anywhere in the command, `-f`
  as its own argument, and a `+refspec`. The GitHub MCP server's three
  file-writing tools (`create_or_update_file`, `push_files`, `delete_file`) are
  denied, so an agent cannot commit to the remote except through a local commit
  and a push.
- The rules match the command's text, so they are a guard, not a boundary. They
  do not catch `git -C <dir> push -f`, a combined short flag such as `-uf`, a git
  alias, or a push run from a script. Checked by running each force form
  against a remote that does not exist, with `--dry-run`: every one was denied
  in both tools, and a plain push was not.
- Spec: no section describes the developer tooling; not changed.

### 2026-10-01 - The design documents, under docs/design/

- Added the design spec (`docs/design/5d-chess-transformer-design.md`), the
  decision register (`docs/design/5d-chess-decisions.md`) and the general
  architecture the design instantiates
  (`docs/design/scalable-transformers-architecture.md`), as supplied: the current
  copies, carrying register row M11 and spec §15.6. They were kept untracked in a
  folder named `Specification documents/`; every reference to that path
  (`AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, the boundary's documentation list
  in `oracles/gate.toml`, and a harness test that classifies this repository's own
  files) now names `docs/design/`. Their text is unchanged: they are maintained
  upstream.
- These copies already carry the corrections the oracle work below found owing:
  register J20 removes the claims that cwmtt's checkmate tool uses z3 and that
  ftxi packs a coordinate into 8 bits, and H26 names the notation specification as
  the fourth submodule.
- Spec: these are the spec and the register; not otherwise changed.

### 2026-10-01 - GitHub remote; the graft shims carry no local path

- Added the GitHub repository as `origin`
  (https://github.com/Silicon-Spider/Temporal-5D). GitHub created it with a
  commit of its own - `LICENSE` (the GPL-3.0 text) and a two-line `README.md` -
  so the graft commit, never pushed, was rebased onto it: e1b6980 is now
  a56beb4. Nothing has been pushed yet. The project's licence is still being
  decided, and whether that `LICENSE` file stays is part of the decision. The
  oracle work stays uncommitted, as before. `AGENTS.md`, `CLAUDE.md`,
  `CONTRIBUTING.md` and `oracles/README.md` no longer say the repository has no
  remote: they say nothing has been pushed to it.
- `.claude/helpers/graft-hooks.cjs` and `.claude/helpers/graft-statusline.cjs`:
  the first place the shims look for graft (`BAKED`), which `graft init` writes
  as an absolute path into the installing user's profile, is now the same
  directory found from node's own install directory (npm's global root on
  Windows), so the commit carries no local path. Re-running `graft init` writes
  the absolute path again. `PROVENANCE.md` records the change.
- Spec: no section describes the developer tooling, the remote, or the
  project's own licence; not changed.

### 2026-10-01 - Every change goes in the changelog and the spec

- Added this file, and a rule - stated in full in `AGENTS.md` (for coding agents,
  and imported by `CLAUDE.md`) and `CONTRIBUTING.md` (for people), and pointed to
  from `CLAUDE.md` - that a change is finished only when it is recorded here and
  the design spec and the decision register say what is now true, in the same
  edit. The pin procedure in `CONTRIBUTING.md` gains the step that does this for
  a pin bump.
- Why: the register already requires a decision to change in the spec and the
  register together; this carries the same discipline to every change, and gives
  one place to see what changed.
- Spec: no section describes how changes are recorded; not changed. The entry
  below still owes its spec updates.

### 2026-09-30 to 2026-10-01 - The oracle side of the rules-engine gate (H27)

- Added `oracles/`, a separate, standard-library-only Python project: the
  differential-testing harness of spec §11.0 (decisions H23-H29). Commands:
  `sync`, `doctor`, `build`, `verify`, `selftest`, `gate`, `golden`, `observe`,
  `corpus`, `boundary`, `provenance`. Its manual is `oracles/README.md`.
- Pinned as git submodules under `oracles/third_party/`: ftxi/5dchess_engine
  (v0.4.10), penteract/cwmtt, 5d-chess-js (the rules library of Chess in 5D), and
  adri326/5dchess-notation as a source of game records. Commits, licences and
  toolchains are in `oracles/oracles.lock.toml`.
- Drivers of our own for ftxi (C++; part of it adapted from ftxi's tools, under
  its BSD-2-Clause notice) and 5d-chess-js (Node). cwmtt runs through its stock
  command-line program.
- A pinned corpus of 41 game records (37 read in place from the submodules, 4
  written here); each oracle's recorded answers (`oracles/golden/`); the generated
  `oracles/DISAGREEMENTS.md`; an empty `oracles/known-divergences.toml`.
- The boundary check that keeps the engine from using the harness or an oracle; a
  gate in three tiers (commit, nightly, release); two GitHub Actions workflows,
  which have never run (the repository has no remote).
- `PROVENANCE.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`, `.gitattributes`,
  and `.gitignore` entries for the harness cache and Python byte-code.
- Why: H27 - the oracles and the gate exist before the rules engine, so that the
  gate cannot be skipped or bent to fit it.
- Spec: not yet brought in line. Pinning found statements in §11.0 and the register
  that are no longer true, listed in `PROVENANCE.md` under "Where the design spec
  and the pinned reality differ": cwmtt no longer uses z3; ftxi packs a coordinate
  into 32 bits, not 8 (one of H24's stated reasons); ftxi runs through a C++ driver
  and its stock CLI, not its Python bindings; a fourth repository, the notation
  specification, is pinned; and no oracle answers every question the gate asks.

### 2026-09-30 - graft wiring (commit e1b6980, rebased as a56beb4)

- Added the wiring of graft, a code-navigation tool for coding agents:
  `.claude/helpers/`, `.claude/skills/graft/`, and entries in
  `.claude/settings.json`, `.mcp.json`, `.ignore` and `.gitignore`. Developer
  tooling: nothing in the engine or the harness depends on it.
- Spec: not affected.
