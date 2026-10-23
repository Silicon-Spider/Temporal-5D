# Changelog

Every change to this repository is recorded here, in the same edit that makes it,
and the spec is brought in line with it - upstream: the spec and the register are
not edited here, and the edits they need are proposed in `docs/BUILD_LOG.md` (the
rule is in `AGENTS.md`). Newest first. Each entry says what changed and why, and
ends with a `Spec:` line: which sections of the design spec and the decision
register the change bears on and what was proposed for them, or that no part of
the spec describes the change.

## Unreleased

### 2026-10-23 - MT-04: cached = dense on every named variant the engine sets up

`python/tests/engine/test_cached_variants.py` (new; tests only - no code changed): section
15.2's protocol names "every variant" among its adversarial constructions, and cached = dense
had run on real positions of only three games. Now every variant of the engine's table (all 16,
`engine/src/variants.cpp`) is played at random from its start - two seeds, travel favoured for
White so that timelines branch and some go inactive, a move after which the action could neither
go on nor be submitted passed over (Timeline Tactician's first action, T13) - and every state is
evaluated: turn starts, mid-action, shifted (some with optional moves made after the shift) and
submitted. Boards 3x3 to 8x8, odd and even starts, the T0 board, the royal queen and princess,
brawns and unicorns (no variant holds a dragon); lineage only (C23).

- `test_cached_equals_full_pass_on_every_variant` (16 variants x 2 configurations): the
  positions of a game evaluated incrementally in one epoch opened at a mid-game root, the store
  shared, against `reference_output` - under both of C20's readings (`causal_past` over the whole
  game; `past_and_epoch_history` with the root's frozen boards as the epoch's history). The
  configurations: the v1's (B15's table, B16's status key biases, A6's turn-state counts and A7's
  frame - all defaults; A7 cached per perspective, issue 303), and the step-1 one with A8's
  board-layout embedding, D4's reply head and D9's optimistic policy on (D3's heads and D4's soft
  policy are on in both). FP32, 2 layers, `test_cache.py`'s `_close` (2e-5 trunk, 1e-4 heads).
- `test_step_api_cached_equals_dense_on_every_variant` (16): the C step API's own leaves
  (`run_search`), tree and graph mode (C11), the search's evaluation cache off, through
  `CachedNetworkEvaluator` against `NetworkEvaluator(scope="block_causal")` - and with C3's root
  refresh against the two-level reference; every leaf met twice is bitwise identical.
  `test_two_orders_of_one_action_through_the_step_api` (2): at roots with several boards to move,
  the tree search reaches states by two orders of moves (each a leaf of its own) and the graph
  search meets them as transposition hits - bitwise identical, in both epochs (EQ-04's
  engine-side case).
- `test_every_n2_switch_with_relations_on_three_variants` (7 flag sets x 3 variants): B5,
  B9-B12, B15's no-table arm, B17 and A8 with the structure, each network's positions converted
  with `relations_for` - attack, defence, pin, x-ray and check edges all present (engine ABI
  round 3).
- `test_f1_bf16_cached_on_three_variants` (slow, 3): F1 block-causal in BF16 on Turn Zero,
  Reversed Royalty and Timeline Strategos, at `test_network.py`'s BF16 bounds.

All 60 green: no variant breaks cached = dense. Largest gaps (the module prints them under
`-s`; in `docs/MEASUREMENTS.md`): FP32 trunk 1.9e-6, heads 1.7e-6, step-API evaluations
1.4e-6; BF16 trunk RMS 2.0e-3 / max 1.2e-2, policy 7.2e-3, value 1.3e-3 - the gaps
`test_network.py` and `test_cache.py` record. Wall time of the file: 603 s serial on the shared
machine at 100% CPU (other lanes' engine tests); the same cases ran 3-6x faster earlier in the
session on a less loaded machine. Spec issues 520-539 and edits 360-379 (this session's
block): none used.

Spec: §15.2 rows 1, 4, 5 and 6 and its protocol ("every variant"); register C2, C17, C19, C20,
C23, C3, C11, B15, B16, A6, A7, A8 - tested as built, no decision changed; no part of the spec
describes the test file, and no edit is proposed.

### 2026-10-23 - Engine ABI round 3 review fixes: the exchange label's recaptures read where they are made; the Q6 fallback, mid-action placement, the pin's royal half, the C API's bits and labels and every label reading tested; the labels' cost measured

Two reviews of engine ABI round 3 (5d8b12b, 0986472): one of the rules (probes on hand-built
positions and random play), one of 42 mutants, 14 of which survived every test tier - one
finding HIGH, seven MEDIUM, five LOW. All HIGH and MEDIUM fixed; the LOWs fixed but for the
deprecated alias (below).

- **The exchange label read "attacked" off the passed boards** (finding A-H1;
  `engine/src/labels.cpp`, `include/t5d/labels.hpp`): its recapturers were the occupant's
  side's defences of the square, and a defence relation targets only a board of the defender's
  colour - so on history boards, boards ahead of the present, inactive timelines and boards
  already played (87% of the squares labelled 1 on random play) none could exist. Now, by the
  main loop's ruling, the recapturers are read **where the recapture is made**: on a passed
  board as before (the side's defences of the square); everywhere else on the board the capture
  creates - the enemy's least valuable capture played (`GameState::apply`; the opponent's from
  its view as a turn start of its own), the enemy passing on the boards it has still to play,
  the occupant's side's pieces that can then take the capturer recapturing; with several least
  valuable capturers the enemy picks the best (a first version broke the tie by board order,
  and A7's mirror test caught it). Where that cannot be read (the pass undefined, Q6; an
  opponent's view that is no turn start) the label is `T5D_LABEL_UNKNOWN` - NaN, no target -
  not a guess. `SquareLabelStats` counts `exchange_replays` and `exchange_unknown` (C++ only:
  the C struct is unchanged). The reviewer's probe: 3,613 squares labelled 1 over 278 turn
  starts (3,139, 87%, on history or unpassed boards) became 1,451 (977, 67%); of the 3,139
  squares replayed, 977 still lose the occupant; none Unknown.
- **The Q6 fallback** (finding A-M1; `engine/src/views.cpp`): found by its precondition (a
  deferred timeline of the mover's, active, in the present) before `passed_timelines` is
  called, no longer by catching `std::logic_error` (which would have swallowed any error of
  that family); tested on a Deferred reactivation that reaches it. It still drops the whole
  pass; passing the non-deferred timelines alone would need a variant of check status's pass
  (`check.cpp`, untouched) - logged in issue 172 as the alternative.
- **Tests** (`engine/tests/test_views.cpp`, `test_capi_game.cpp`;
  `python/tests/engine/test_binding.py`, `test_perspective_engine.py`,
  `tests/train/test_targets.py`): a board played mid-action reported where it stands - a check,
  a pin and an arrival check (B-M1); no pin in front of an enemy that is not royal (B-M2); the
  C API's pin and check bits each alone (B-M3); the exchange's order, a royal recapturer's value
  and no x-ray recapturer, the arrival check's moved-piece rule, the royal zone's whole-turn step
  (B-M4); the exchange's recaptures on created boards - ahead of the present, on a history board,
  by either side - and Unknown under Q6 (A-H1); A7's mirror test now requires every label
  decided on both sides and every label, the mate's included, set somewhere (B-M5);
  `t5d_game_square_labels` at the C level (NULL arguments, NULL options' budget, a one-node
  budget's Unknown, the counts field by field) and `targets()`' NaN exactly at the 255s (B-L7);
  and every label 0, 1 or Unknown on random play under all 27 rulesets, turn starts and
  mid-action (1,331 states, no exchange Unknown; none at a default turn start, asserted) - an
  opponent's capture its turn-start state refuses (activity read by the count rule under a
  history-dependent ruleset) is left Unknown rather than thrown.
  The 14 surviving mutants re-run on a scratch copy: all 14 killed.
- **Refusals** (B-L8; `python/t5d/engine/game.py`, `train/targets.py`):
  `Game.square_labels(mate_nodes=...)` refuses a budget below 1 (a negative one wrapped to
  `UINT64_MAX` through ctypes); `TargetConfig.square_mate_nodes` refuses 0 (every mate label
  NaN, silently).
- **Wording** (A-L1, A-L2, A-L3): the acting player's check edges are empty in every position
  reached by legal play (13,750 decision states of random play: none), not "almost always"; its
  pins are x-ray-to-royal edges, not binding pins; the mate search under `T5D_NO_LIMIT` /
  `SearchBudget::unlimited()` has no node limit, not "exact" (the memory limit stays) -
  `labels.hpp`, `capi.h`, `capi.py`, `game.py`, `expand.hpp`, `engine/README.md`.
- **Not done**: a deprecated `T5D_RELATION_TRAVEL_THREAT` alias (B-L8). Round 3's removal of
  that macro from `records.h` is a **source break** for a C caller that named it (its value, 6,
  is `T5D_RELATION_RESERVED`, still refused); no alias is kept, since the type is gone from the
  register (B11's five) and the engine lane's documents no longer name it.
- **Cost** (B-M6): measured with the fix in - `docs/MEASUREMENTS.md`, "The per-square labels
  and the pin and check relations - cost per state" (indicative; no threshold, no verdict). On 12 random-play positions of 10-212 boards
  (Standard, 4 sizes x 3 seeds, travel_prob 0.5; the minimum of three timings, a shared machine):
  the labels 0.1-23 ms per state after (0.1-26 before; round 3's figure, 0.3-38 ms, was one
  timing each), the exchange's replays adding up to about 3 ms at 100-200 boards; pin and check
  add 0.0-0.2 ms to an expansion of 0.04-0.2 ms.
- **Docs**: `engine/README.md` (the views, (e)'s pin and check, "Per-square labels"),
  `docs/BUILD_LOG.md` (the B11 and D4 rows, issues 172-176, edit 136, the session log); the two
  round-3 entries below corrected where they claimed what was untrue or untested.

Spec: §12.4's per-square row and register D4 - proposed edit 136 amended (a label the engine
cannot decide is no target); BUILD_LOG issues 172-176 (176(a) the exchange's reading, 172 the
Q6 fallback's alternative, 173-174 the acting player's pins and checks). §6 and register B11:
edit 135 unchanged. Register A7 and §15.2's A7 row unchanged (the mirror test is stricter).

### 2026-10-23 - Engine ABI round 3, part 2: D4's per-square label generator, wired into M21's targets

§12.4's per-square row - "Deep attack facts beyond the §6 input bias (label set a placeholder;
6j) | Per square | Generator - exact" - had no generator: M21's targets left every per-square
label NaN. The engine now generates exactly the set `heads.SQUARE_LABELS` names (v1, M20), and
the training targets re-derive it from each decision's replayed state.

- **The generator** (`engine/include/t5d/labels.hpp`, `src/labels.cpp`): for every square of
  every board, the eight labels in `SQUARE_LABELS`' order, relative to the acting player (A7:
  "ours" its), each side read from its own view (issue 172), every rule through the engine's own
  predicates. The readings, placeholders (issues 175, 176): the static exchange (the swap over
  the enemy's captures and the side's defences, the scanner's H6 values, 20,000 for a royal);
  *(review correction, 2026-10-23: off the passed boards the side's defences could not exist -
  a defence targets only a board of the defender's colour - so there the label meant
  "attacked"; the recaptures are now read on the board the capture creates, above)*
  reach in exactly two moves and arrival check (from the moved piece only) under the static
  reading of a piece's next move (`append_moves_from`, new in `movegen.hpp`); the royal zone
  (the royal's square and one king step, any board); mate after a move that ends a legal action
  (E4's hypercuboid detector, 200,000 nodes each; undecided -> `T5D_LABEL_UNKNOWN`).
- **The C API** (`capi.h`, `capi.cpp`): `t5d_game_square_labels`, `t5d_label_options`,
  `t5d_square_labels`, `T5D_LABEL_*` (layouts in `t5d_layout`). **Python**: `Game.square_labels`
  (`SquareLabels.targets()`: NaN undecided), `capi.SQUARE_LABEL_IDS`.
- **M21's wiring** (`python/t5d/train/selfplay.py`, `targets.py`, `trainer.py`, `losses.py`;
  `engine/convert.py`; `net/heads.py`'s note that the generator was not built):
  `replay_positions(square_labels=True)` labels each replayed state (`Position.square_labels`); `build_targets` carries them (`GameTargets.square`; `reply=` to
  skip the reply maps), `collate_rows` puts them on the square tokens; the trainer asks for them
  (`TargetConfig.square_labels`, `square_mate_nodes`). Never stored in a record or the targets
  file - I15's round trip is untouched.
- **Tests**: `test_views.cpp` - every label on hand-built positions (royal zones across
  timelines; two-move reach on a board and through time; arrival checks on a board, through
  time and from the opponent's view; the exchange won and lost by each side; the back-rank
  mate, its refutation, a mate left unknown by a spent budget; the ids against the C API's);
  `test_perspective_engine.py` - A7's mirror on real positions (the same labels at the mirrored
  squares, every label but the mate's seen); `test_targets.py` - re-derived on every replay, not
  stored, on the square tokens, every loss finite; `test_trainer.py` - record rows carry them.
  *(Review correction, 2026-10-23: "every label" overstated it - the exchange's order, royal
  value and x-ray rule, the arrival check's moved-piece rule, the royal zone's turn step, the C
  API's labels and the NaN mask were untested, and the mirror test compared the mate label only
  where both searches decided it, so a side whose searches all stopped passed; all tested since,
  above.)*
- **Cost**: 0.3-38 ms per state on random-play positions of 10-212 boards (a shared machine).
  *(Review correction, 2026-10-23: the figure was in no measurements file - now in
  `docs/MEASUREMENTS.md` with its conditions, re-measured with the exchange fix, above.)*
- **Docs**: `engine/README.md` ("Per-square labels", the C API, the readings),
  `docs/BUILD_LOG.md` (the D4 and M21 rows, issues 175-176, edit 136, the session log).

Spec: §12.4's per-square row and register D4 (with M20) - proposed edit 136 (the v1 set named,
generated by the rules engine, re-derived rather than recorded); BUILD_LOG issues 175-176.
Register A7 and §15.2's A7 row unchanged (the labels are mirrored as the row asks of outputs).

### 2026-10-23 - Engine ABI round 3, part 1: B11's pin and check relations, each side read from its own view; id 6 reserved, travel-threat gone

Register B11 lists five typed relations - attack, defence, pin, x-ray, check - and the engine
built three; pin and check waited on check detection (H36), which E4 built. They are built now,
behind the relation mask, written from the rulebook; check detection's predicates are only
called, never changed.

- **The two views** (`engine/include/t5d/views.hpp`, `src/views.cpp`; spec issue 172): at a
  decision state only the acting player's pieces can move, so each side is read from its own
  view - the acting player's pieces in the state, the opponent's on the multiverse **check
  status passes** (R1: `passed_multiverse`, the acting player's unplayed active boards in the
  present advanced unchanged), a passed copy's squares reported at the board it copies. At a
  turn start that is R1's check status exactly; mid-action the pass of the boards still to play.
  *(Review correction, 2026-10-23: a board already played mid-action - standing at present + 1
  too - was untested, and so was the Q6 fallback, found by catching `std::logic_error`; both
  tested since, the fallback by its precondition, above.)*
- **Check** (`expand.cpp`; issue 174): a piece -> an enemy royal it can capture, by check
  detection's own scan (`for_each_royal_capture`): the opponent's captures of the acting
  player's royals in its view, then the acting player's of the opponent's in the state. At a
  turn start an opponent check edge exists exactly when `in_check` holds (held on 321 random
  turn starts of three variants, 9 in check). *(Review correction, 2026-10-23: the acting
  player's edges are empty in every position reached by legal play - 13,750 decision states,
  none - and its pins are x-ray-to-royal edges, not binding pins: issues 173-174.)*
- **Pin** (`movegen.hpp` `GeometricRelation::Kind::Pin`, `RelationKinds::pin`; issue 173): a
  slider -> the piece it pins - the first piece along its ray an enemy that is not royal, the
  next an enemy royal the slider would capture were the first absent - the x-ray's own walk,
  so across timelines and through time; the opponent's pins on the passed multiverse
  (`append_moves_of` with relations, new). Moves are generated exactly as before. *(Review
  correction, 2026-10-23: the rule's royal half - no pin in front of an enemy that is not
  royal - was untested; tested since.)*
- **The C ABI** (`records.h`, `capi.h`, `capi.cpp`): `T5D_RELATIONS_PIN`, `T5D_RELATIONS_CHECK`;
  id 6 is `T5D_RELATION_RESERVED` - never emitted, refused ("reserved"; it was travel-threat's,
  folded into attack at CHECKPOINT 6); `ExpandOptions::travel_threat` removed; the search
  accepts pin and check for its leaves (`mcts.cpp`). Pin and check make the expansion allocate.
  *(Review correction, 2026-10-23: removing `T5D_RELATION_TRAVEL_THREAT` is a source break for
  a C caller that named it - no alias kept; and the two new bits were tested only together,
  each alone since.)*
- **Python** (`python/t5d/engine/capi.py`, `game.py`, `convert.py`, `errors.py`):
  `BUILT_RELATIONS` holds all five (so `relations_for` gives a network with B10 or B11 on all
  five), `RESERVED_RELATIONS` names none; the docstrings no longer name travel-threat - the
  engine lane's documents the B11 row listed - and `net/state.py`'s relations paragraph no
  longer says pin and check stay empty.
- **Tests**: `engine/tests/test_views.cpp` (pins on a board, across timelines, through time,
  from both views; checks on a board, across timelines, through time, mid-action, by the acting
  player; check edges against `in_check` on random play); the reserved id refused
  (`test_capi_game`, `test_capi_search`), a search's leaves carrying all five as the game's own
  expansion (`capi_search_leaves_carry_the_network_input`). `python/tests/engine`: every
  conversion, the B11/B10 network test and A7's mirror test run with the five, each type seen
  on real positions; issue 229's anchor test now allows pin and check edges from the
  opponent's pieces.
- **Docs**: `engine/README.md` ((e), the status table, the readings), `docs/BUILD_LOG.md`
  (the B11 row, issues 172-174, edit 135, the session log).

Spec: §6 "Typed edge biases" and register B11 - proposed edit 135 (whose pieces and which
multiverse pin and check read; what a pin is in 5D; check as check detection's own capture);
BUILD_LOG issues 172-174. §15.2's A7 row is unchanged (the mirror test now covers pin and check).

### 2026-10-23 - A8 review fixes: the cost flagged under the 2x rule, the cached node's share counted, the FFN's form pinned, non-square padding and FP32 values tested

A review of register A8 (`lane/net` a1ec09b: spec fidelity, and 24 mutants run at test time, five
of which survived). In the network lane - no training run.

- **The cost was presented as confirmed while more than 2x off section 15.3's estimate.** The row's
  "~0.1% of attention at 44 boards" is the projection's figure (0.099%), but the row is register
  A8's, and A8 has "an FFN after the embedding": as built (8x, a placeholder) A8 is 4.50% of
  attention at 44 boards - 45x the estimate - and any per-token FFN, even one 1·d wide, makes it at
  least 0.66%, 6.6x. `docs/MEASUREMENTS.md` now flags it under the 2x rule (a "Flags" line in the
  A8 section; the §15.3 table's ratio column gives the register's A8, 45x and at least 6.6x, beside
  the projection's 0.99), the build log logs it (issue 240(l), the A8 row) and edit 186 says so.
  The parameters (2,629,120 against "~1.6M", 1.64x) are inside the rule.
- **A8's cost in the cached configuration, counted.** `CachedEvaluator.prepare` embeds every token
  of a node, frozen boards' included, so A8's per-token FFN is paid over the whole position at
  every node, while the node's trunk work is its live set's. Counted (`FlopCounterMode`, FP32, the
  cached configuration, `synthetic_state(0, N, n_timelines=8)` and its physical child against the
  root's store; FLOP counts do not depend on load): A8 adds 6.43 GFLOP to the child node's 63.83
  at 44 boards (+10.1%) and 34.28 to 238.19 at 240 (+14.4%) - 13% and 21% of section 7.6's analytic
  cached node - where it adds 2.0% and 0.62% to the dense forward; the live pass and the
  supersession are unchanged by it. Stated in issue 240(g), `docs/MEASUREMENTS.md` (A8, "In the
  cached configuration") and edit 186. `cache_bench` counts a node's preparation (`prepare_flops`,
  the node row's `counted_prepare_tf`) beside its live pass, which was all it counted.
  Embedding only the live and the missing boards' tokens would remove most of it: not built (the
  node's embedding is made before its missing boards are known) - logged in 240(g).
- **The FFN's documented form pinned**: `test_the_ffn_is_one_pre_norm_residual_block_on_every_valid_token`
  asserts bit for bit that the embedding is `x0 + where(valid, W2 mish(W1 RMSNorm(x0)), 0)`, with
  `x0` the `layout_ffn_ratio = 0` network's embedding of the same seed (the same projection and
  merge), and that every square, per-board register and global register row changes while padding
  does not. Each surviving mutant of the FFN - no residual, squares only, GELU, the FFN before the
  merge - fails it, and so does no norm (caught only incidentally before).
- **Smaller fixes.** The canonical-order test takes non-square boards (6 x 4, 3 x 5), so centring
  with the width and height swapped (the fifth survivor) fails it. The FP32 test compares the
  layout channels and the embedding with an FP64 reference of the same module, so BF16 rounding of
  the merge fails it directly (`layouts()` now builds the one-hot in the projection's dtype).
  `test_both_colours_acting_are_seen` and `test_storage_order_would_break_it_for_black` exercised
  no implementation: folded into the canonical-order test, which now asserts both acting colours at
  every size. `cost.layout_flops` returns zeros with the flag off, and its docstring says that
  `dense_flops` and `pass_flops` (section 9.1's model) leave A8 out as they leave every embedding
  out.
- **Mutants re-run** (the reviewer's plugin, no file edited): the five survivors, BF16 rounding of
  the merge and the FFN without its norm - each fails its own test directly.
- **The A8 entry below, corrected**: its "Measured" bullet called the projection's 0.099% "the
  row's ~0.1%" without the flag A8's 4.5% needed, and gave no cost for the cached configuration;
  its "Tests" bullet did not pin the FFN's form, and its FP32 check was of the dtype alone.

Spec: §2.2 (the board-layout paragraph), §7.6 (the cached node), §15.3 (the A8 row - flagged, more
than 2x off; its M4 write-back rewritten); register A8, M4. Issue 240(b), (g), (k), (l); proposed
edits 185 and 186 rewritten.

### 2026-10-23 - A8's board-layout embedding behind its flag (off), with section 15.3's cost row counted

Register A8 (a prototype, gate 6p; spec section 2.2), in the network lane - no training run.

- **`python/t5d/net/layout.py`** (new): `BoardLayoutEmbedding`. Per board, the one-hot layout - 64
  cells x 24 owner-and-piece states (12 piece types x ours/theirs, register A7), flattened to 1,536 -
  with each square at its **canonical** cell (ranks mirrored when Black acts; the build log's note
  under A7), projected by `Linear(1536, 64 x 16)` (1,573,888 parameters - section 2.2's "~1.6M"); each
  square token takes its own cell's 16 channels, added to the token input through a learned
  `Linear(16, d)` (the concatenation built as a sum, 4,096); then the "FFN after the embedding" - one
  pre-norm residual block, RMSNorm and Mish, 8x (1,051,136), on every valid token. 2,629,120 in all.
- **`NetConfig`**: `board_layout_embedding` (off), `layout_channels` (16, the spec's),
  `layout_ffn_ratio` (8, a placeholder; 0 drops the FFN - a cost probe) and `layout_padding` ("a1",
  a placeholder; "centre" the other arm): boards smaller than 8 x 8 padded in the canonical frame,
  larger ones refused. `cost.layout_flops`; `parameter_breakdown`'s `embed.layout`; `model.a8_seed`.
- **Off, bit for bit**: A8 draws nothing while made and is drawn last, from its own generator - every
  other parameter as before, with any flags; with its module removed the outputs are the flag-off
  network's. F1 unchanged (22,965,426; 25,594,546 with A8 on).
- **Every path**: the embedding precedes the trunk, so the six attention implementations agree with
  it on; the cached path needs no change (the layout is per board and intrinsic, keyed by the
  perspective already - issue 303 - and the FFN per token): cached = one full pass with A8 on. FP32,
  outside the trunk's autocast; never quantised under G2 (issue 238(d)).
- **Tests**: `tests/net/test_layout.py` (new) - off/on parameters and outputs, the count, the layout
  against one written from the position's absolute facts (both acting colours, sizes 8 to 1, both
  paddings), layer 0 depends only on the token's own board, the 24 states, precision, every
  attention path, cached = full (five geometries, three seeds), the cost row counted exactly, and
  three smoke optimiser steps on synthetic rows. `test_perspective.py`: section 15.2's A7 row with
  A8 on - every construction, both scopes and paddings, and the cached path from both perspectives;
  `test_cache.py`: A8 under every N2 switch. Mutants (storage order for Black, owners dropped, every
  square reading board 0) are each caught. *(Corrected by the review fixes above: the FFN's form was
  not pinned - four mutants of it survived - centring was tested on square boards only, and the FP32
  test checked the dtype alone.)*
- **Measured** (`docs/MEASUREMENTS.md`, "A8 - the board-layout embedding"): the projection 0.138
  GFLOP at 44 boards, 0.099% of attention by section 15.3's formula - the row's "~0.1%" - and with the
  register's FFN 6.29 GFLOP, 4.5% (2.0% of the counted forward); CPU wall time indicative only (the
  machine at 85-100% load): the embedding +18 ms at 44 boards, +103 ms at 240 (BF16). *(Corrected
  by the review fixes above: the row is register A8's, and A8 with its FFN is 45x the row's ~0.1% -
  flagged under the 2x rule, not a confirmation; and in the cached configuration A8 adds 10.1% /
  14.4% to a node at 44 / 240 boards, which this entry did not state.)*

Spec: §2.2 (the board-layout paragraph), §13.1 6p, §15.1 Phase 2, §15.2 (the A7 row and C2's
cached = dense, with A8 on), §15.3 (the A8 row - measured; its M4 write-back proposed); register A8,
A7, F5, F6, G2, M4. Issue 240 (A8's placeholders: the FFN's width, form and tokens, the concatenation
as a sum, the padding, larger boards, the 24 states, precision, initialisation); proposed edits 185
(§2.2 and register A8, as built) and 186 (§15.3's row, M4).

### 2026-10-23 - Precision review fixes: QAT records name the format, the attention operands tested operand by operand, the cache checks the format at every evaluation, non-finite scales kept non-finite

Two reviews of the precision component (G2's FP8 emulation and C15's estimator; G8's QAT hook):
a numerics review with an independent reference, and a 43-mutant pass that left nine real
survivors.

- **QAT's records named the wrong network** (HIGH): `apply_qat` switches the network in place, but
  the trainer's `net_cfg` - what it hands to self-play - kept `trunk_quant = "none"`, the records'
  network condition took the caller's configuration, and `network_id` hashed the weights alone:
  a QAT network's records said "none" with the unquantised network's id, so issue 238(j)'s claim
  was false. Now `Trainer.net_cfg` follows the switch; `selfplay.network_info` takes the
  `trunk_quant*` fields from the evaluated network and refuses a caller's configuration whose
  format disagrees (`quant.trunk_format`); `records.network_id` folds an 8-bit format into the hash
  (a network in "none" hashes as before - existing ids unchanged). Tested through generate ->
  steps -> publish -> generate under QAT (`test_qat.py`).
- **The attention operands, operand by operand** (HIGH): nothing checked that attention's Q, K, V
  and rotated Q, K were each quantised, or that RoPE came before the rounding - mutants leaving V,
  the unrotated or the rotated operands unrounded, or rotating rounded ones, all survived.
  `test_the_attention_operands_are_rounded_after_rope` captures `quantised_operands` on all six
  paths: each operand is its raw operand through E4M3 per token and head, a fixed point of the
  format, the rotated ones rounded after FP32 RoPE (which differs from rotating rounded ones);
  with `trunk_quant_attention=False` they are the raw operands bit for bit.
- **The cache checks the format at every evaluation** (`cache.CachedEvaluator`): it refused FP8
  attention at construction only, so a network switched in place afterwards (as QAT switches the
  trainer's) was served - trunk RMS 1.4e-2 from the full pass against the 6e-3 bound - and a store
  could mix two formats' K/V. It now records the format (trunk format and frozen-K/V format) and
  refuses at every epoch and evaluation once it changed. The in-place hazard for a
  `NetworkEvaluator` (it evaluates in FP8 afterwards, silently) is documented (issue 239(b)).
- **Non-finite slices** (`quant.scale_of`): under "pow2" a slice holding inf or NaN got a finite,
  platform-dependent scale (`frexp` of inf/NaN is unspecified), so [1, inf, 2, 3] came back finite
  and the trainer's guards could not fire; the non-finite scale is now kept under both rules (the
  slice comes out NaN). Tested on inf and NaN rows, through `QuantLinear` too.
- **What "K/V cache included" covers** (corrected in BUILD_LOG 239(a), the G8 row, edit 182 and
  the G8 entry below): QAT's forward is the full pass's, which rounds the frozen K/V as the cache
  stores them; the cache serves the projections-only network with FP8 frozen K/V (now tested:
  trunk RMS 2.8e-3, MEASUREMENTS) and refuses FP8 attention - QAT's default - so a block-causal
  network trained with it is one the cache refuses (logged as a choice, 239(i)); the double
  rounding of frozen K/V under FP8 attention is logged (238(k)).
- **Tests sharpened**: `apply_qat` keeping the network's `trunk_quant_*` settings (a fourth
  configuration of the forward test); the scale granularity made visible under "pow2" (weight rows
  and tokens binades apart, and an "absmax" case: per channel and per token, not per tensor, per
  input channel or per batch) and the "pow2" batch-independence bound tightened from 1e-2 to 1e-5
  over model seeds 0-5, not 0-2 (unmutated at most 5.2e-7; the projections' per-batch mutant
  2.1e-4, at seed 4 only - MEASUREMENTS); a scale already a power of two kept, not
  doubled; the element format's random-value check parametrised over devices (a GPU when present,
  M19).
- **Small fixes**: float64 input refused (FP32 is the input precision - it was rounded twice);
  `apply_qat` validates its configuration, so its "another 8-bit format" refusal is tested with a
  validated one; the clamp before the cast documented as unobservable on this backend (torch 2.13's
  CPU cast saturates by itself); the gradient figure given as bound 0.5, measured at most 0.117
  (CHANGELOG's G8 entry and the f2007c7 message said "within 0.12"); the DeepSeek-V3 report's
  finding that Hopper FP8 GEMMs accumulate in about 14 bits - verified against the report - logged
  as a placeholder for 6w's GPU arm (238(f)).

Spec: §10.2, §12.8, §15.1, §15.2; register G2, G8, C15, C2, I15, M19. Proposed: edit 182 reworded
(QAT's forward is the served format's full pass; the cache serves the projections-only format),
edit 183 (the accumulator: a Hopper FP8 GEMM keeps about 14 bits). Spec issues 238(a), (f), (j),
(k) and 239(a)-(c), (g)-(i).

### 2026-10-23 - G8's QAT hook behind its flag (off), plugged into M21's trainer

The precision component, step 2 of 2 (`lane/net`). The user's brief: "QAT hooks G8 (FP8 E4M3, flag
off, never the heads or RoPE; the FFN ratio stays a config parameter)" - edit 182 ruled for the
hook.

- **The hook** (`train/trainer.py`): `TrainerConfig.qat` (off) and `qat_format = "fp8_e4m3"`
  (provisional, pinned with the inference path); `apply_qat`, the M21 seam that raised, now
  switches the trainer's network in place to G2's FP8 trunk (`T5DNet.set_trunk_quant`) - fake
  quantisation with straight-through gradients, the heads, RoPE, embeddings and norms untouched,
  the network's frozen-K/V format kept - and refuses a network already in another 8-bit format.
  In place so that the network's optimiser-step count stays with it (issue 237(a)).
- **Tests** (`tests/train/test_qat.py`): off by default; QAT's forward equal bit for bit to G2's
  inference forward with the same weights - dense, the block-causal scope with FP8 frozen K/V
  ("K/V cache included"), BF16 with the chunked path; a smoke test of three optimiser steps on
  synthetic rows only (BF16 and FP32): losses finite, a dense gradient on every quantised weight,
  every trunk parameter moved, the fourth step refused; each projection's gradient within relative
  distance 0.12 of the unquantised network's (the flushing cast gave zero); FFN ratios 2 and 12.
  `test_trainer.py`'s seam check is gone with the seam.
- *Corrected by the precision review fixes (above):* "K/V cache included" meant the full pass's
  rounding of the frozen K/V, not the cache - which refuses QAT's default (FP8 attention); "within
  relative distance 0.12" was the measurement (at most 0.117), the test's bound is 0.5; the
  settings kept and the refusal of another format were untested; and the records of a QAT network
  named it unquantised (issue 238(j) said they named it).

Spec: §10.2, §15.1, §12.8; register G8 (built behind its flag, smoke-tested), M21 (the QAT hook
built), G2, C15, I17. Proposed: edit 182, rewritten as ruled (§15.1's M21 sentence and G8: the
hook built behind its flag on the provisional E4M3 emulation). Spec issues 239, 237(c) (ruled).

### 2026-10-23 - G2's FP8 E4M3 emulation behind its flag (off), and C15's straight-through fix

The precision component, step 1 of 2 (`lane/net`).

- **One FP8 definition** (`python/t5d/net/quant.py`): E4M3 (OCP's OFP8; PyTorch's
  `float8_e4m3fn`) - round to nearest even, saturating at +-448 (infinities too, clamped before the
  cast), NaN kept; scales absmax / 448, rounded up to a power of two by default (`"pow2"`, exact in
  BF16) or not (`"absmax"`); fake quantisation with a straight-through gradient. Tested against a
  decode of all 256 codes written from the OCP definition, at every midpoint between neighbours and
  either side of it, beyond the range and on 20,000 random values.
- **G2's 8-bit trunk, behind its flag** (`NetConfig.trunk_quant = "fp8_e4m3"`, off; it replaces
  `fp8_attention`, which raised): the six projections of every block are `quant.QuantLinear` -
  `nn.Linear` bit for bit when off, the same `state_dict` - quantising weights per output channel
  and inputs per token, FP32 accumulation; with `trunk_quant_attention` (default) attention's Q, K
  (after FP32 RoPE) and V per token and head, once per layer at every attention path's entry
  (`attention.quantised_operands`; the cache's paths and FlexAttention's included). Never the
  heads, RoPE, embeddings or norms (tested bit for bit). `T5DNet.set_trunk_quant` /
  `with_trunk_quant` switch a network (or a copy - its FP8 twin) without touching its weights.
  The cache takes the projections-only network and refuses FP8 attention: post-RoPE rounding
  depends on the frame, and the cache computes a frozen board in its own (measured). *Corrected by
  the precision review fixes (above):* the refusal was checked at construction only, a non-finite
  slice got a finite scale under "pow2", and the attention operands were not tested one by one. With the flag
  off the network is unchanged: the outputs of ten configurations (F1 in BF16 and FP32, FFN ratios
  2 and 12, the block-causal scope with FP8 frozen K/V, four attention paths) and the gradients of
  two checked bit for bit against the tree before the change (B15's table gradient aside: it
  varies from run to run by itself, a multithreaded scatter), and a test pins it.
- **C15's estimator defect fixed** (`kvformat.fake_quantise`): the FP8 frozen-K/V rounding cast
  inside autograd, which rounded the gradient onto the unscaled E4M3 grid (63 of 64 gradients zero
  at |x| about 0.1); it now rounds through `quant.py` with a straight-through gradient, its forward
  unchanged bit for bit - so FP8-frozen-K/V training, C15's second route, works.
- `NetConfig.validate` refuses `ffn_ratio < 1` (F1's 8x is a parameter: ablation 5 tests 2x).
- Measured on random weights (MEASUREMENTS, "G2's FP8 emulation"): the FP8 trunk sits at trunk RMS
  2.7-3.2e-2 from the same weights unquantised, about 12x the BF16-vs-FP32 gap.

Spec: §10.2, §8.3, §10.3, §15.2, §15.3; register G2 (built behind its flag), C15 (the defect
fixed), G1, G8, C2, C17, C19-C21, M19. Proposed: edit 183 (§10.2 and G2: the emulated format's
placeholders) and edit 184 (§15.2's protocol and G1 row: FP8 tolerances are rounding flips; the
cache refuses FP8 attention while C2 is deferred). Spec issue 238.

### 2026-10-23 - D2 flat form review fixes: children complete only when every successor is there; a point's realisations walked once; the per-board scores, widening, fallback and cost tested

Two reviews of D2's flat form (42f3590, 3251857, 95d23cc): one of soundness (probes against
E4's listing over 1,171 positions under seven rulesets) and a pass of 31 mutants of the flat
code, 12 of which survived every test tier - three findings HIGH, four MEDIUM, eight LOW.
Merged first: main (the network heads, M21's training loop) into `lane/engine` (22a7337,
M23's merge); there the training generator (`python/t5d/train/selfplay.py`) now refuses the
flat form by name until its policy target - D2's per-board marginals, the net lane's flat
head - exists (it named the retired whole-turn arm), issue 234(b) and the D2 row say so, and
its legal-moves flag no longer reads split 1 as the whole-turn arm.

- **Children complete only when they reach every successor** (HIGH, A-H1;
  `engine/src/mcts.cpp`, `mcts.hpp`): under Q3's play activation two orders of one set of
  moves can reach two states under one action key, and the generator's Actions unit, like
  E4's listing, keeps one - yet the node was marked complete once the generator was
  exhausted, or from an exhaustive fallback listing, and with the proof channel on the solver
  could have proven it lost (H37). Now complete only as successors or where play does not
  activate; the fallback's listing never under Q3; `check_invariants` refuses the rest.
  Tested on reviewer A's position (seed 6202, attempt 1482: 7 successors, 6 reached) with the
  proof channel on, all three arms.
- **A point's realisations walked once** (MEDIUM, A-M1, the main loop's decision;
  `engine/src/hypercuboid.cpp`, `t5d/turngen.hpp`): a point the 32-per-step cap stopped was
  re-queued and judged again from its first realisation - about n^2 / 64 realisations for n
  (5,040 cost 401,936 tested and 1,495,064 nodes; under the default budget 1,824 of 5,040
  emitted). It is now kept and resumed where the cap stopped it - a cursor in the
  realisation's depth-first walk (each choice point's option on the way down; a resumed walk
  replays that path, passes over the visit it stopped at, and goes on), under all three
  realisation regimes, the general one's tables kept for the point: 5,040 tested, 19,912 nodes,
  all 5,040 within 200,000 (`SearchStats::resumed` counts the resumptions). The detectors'
  depth-first walks never resume and are unchanged (the pin holds). A test seam
  (`detail::set_realisations_per_step`) lowers the cap: at 1 and 2, on 400 tiny multiverses
  (244 resumed - 45 under Q3, 43 under Q4's Guide-literal reading), the turns, scores and
  states are the shipped cap's (as successors the action standing for a state may differ:
  which of a point's realisations reaching it is collected first, documented).
- **`bound()` after a budget stop** (LOW, A-L1): the box being judged is queued again, so the
  bound still holds every unemitted turn (checked on every budget cut of §15.2's row); a
  generator stopped before queuing any box bounds nothing (+infinity, not -infinity).
- **Root noise never a non-finite score** (LOW, A-L2; `FlatPolicy::add_noise`): an option
  whose noised probability underflows keeps its own score plus log(1 - epsilon), or -10,000
  at epsilon 1; it threw "a non-finite score" from the generator.
- **The evaluation cache keys on board logits** (LOW, A-L3; `EvalCache::find`/`insert`, a
  defaulted `board_logits` argument): an entry stored from a request that asked for none is
  not served to a flat search's request; tested with a cache shared by a ply and a flat search.
- **Tests** (HIGH B-H1, MEDIUM B-M3-B-M5, LOW A-L4, B-L6; `engine/tests/test_mcts.cpp`,
  `test_turngen.cpp`): each root and evaluated child's edge - its score and prior - against a
  reference written from issues 164-166 (`PerBoard`, `reference_score`), to the bit, with
  distinct leave logits board by board, NaN on one board and none at all, both temperatures
  != 1, `leave_share` 0.3, both `flat_jumps`, and the root's noise-free priors under noise;
  k0 against the entropy and its cap, and at least ceil(k0 N^alpha) turns generated after N
  visits while the generator is live; the fallback's scores (arrivals included) and its
  de-duplication as successors, on a position with jumps between two present boards; the
  generators' cost under ties - 16 turns within 600-2,000 nodes at 7-11 tying boards, a
  40-playout uniform search's generators within 3,000 nodes each; a many-realisation point
  tested once each, tested - emitted <= 31 throughout; castling and en passant through the
  generator under the default rules, Q21's alternative and Q3 (the double step played, 5DFEN
  carrying no en passant right). The first flat test's comment claimed the scores were
  checked; it now points at the new test.
- **Mutants** (reviewer B's tooling, a scratch copy of this tree, tier A - `turngen`,
  `mcts_flat` - alone): all 12 survivors (M02, M07, M09, M16, M17, M18, M24, M25, M26, M27,
  M29, M30) are now caught by an assertion in 13-20 s, none by a timeout, none adjusted; and
  six reverts of these fixes (X1-X6) are caught alike.
- **Python** (LOW A-L5, B-L8; `python/t5d/engine/evaluator.py`): `CStepDriver`'s docstring
  says board logits are passed when an evaluation gives them (the network gives none yet);
  `deliver` checks each leaf's move and board logit counts against that leaf's and refuses a
  miscount (`ValueError`, the batch abandoned) - the batch total alone let one leaf's excess
  cancel another's shortfall; tested (`test_search_api.py`).
- **engine/README.md** (HIGH, B-H2): the C API paragraph describes `board_logits` as built (a
  wrong count `T5D_ERR_ARGUMENT`), not reserved and unsupported.
- Docs: the three flat-form entries below corrected where they claimed what was untrue or
  untested; docs/BUILD_LOG.md (the D2 row; issues 163, 165, 167-171; edits 132, 134);
  docs/MEASUREMENTS.md (the many-realisation point's cost before and after; the memory per
  flat node - 136 KB, the estimate x1.20 of the heap, about 137 KB and x1.23 at 3251857 -
  with their conditions; the tree form's deepest turn start, 3-6 with §8.2's gates and 3-8
  across its gate readings, reconciled with the D2 row - B-L9).

Spec: §8.2 (the flat form - completeness under Q3; proposed edit 132 extended), §8.4 (E1's
noise under the flat form), §11.0 item 3 (H37: a node's children complete), §15.2 (the flat
generator's row - the cursor and castling/en passant tested), §15.3 (proposed edit 131 -
the cost figures), §15.4 (proposed edit 134 extended); register D2, E1, H37, I15 (the
training generator's refusal, issue 234(b)). Spec issues 163, 165, 167-171 and 234(b).

### 2026-10-23 - M21 review fixes: inputs rebuilt under the trainer's configuration, steps on synthetic rows only and capped per model, I19's cutoff as Lc0's, I13 whole in the tree form, the untested claims tested

Two reviews of M21 (spec fidelity; a 46-mutant pass that left 19 mutants alive).

- **The trainer's inputs** (`train/trainer.py`, `train/selfplay.py`): `Trainer.add_game` rebuilt a
  record's inputs under the record's register counts and relations, not the trainer's - a trainer
  of another configuration got inputs its network does not take (a 1/4- and a 3/8-register
  trainer, a typed-relations one with 20 edges for 47). Now `replay_positions(record, net_cfg,
  window_k=...)` under the trainer's, with `TrainerConfig.window_k` (None: the record's; it sets
  the positions' W/E scope, the cached configuration's - a row keeps the state alone). Replay
  also refuses a record whose ruleset or rules encoding is not the replayed game's (the engine
  version a logged reading, issue 234(f)).
- **No training runs, to the letter** (the main loop's reading of the user's rule, issue 237(a)):
  optimiser steps only on synthetic rows (`synthetic_record`'s header says `synthetic`; `step_on`
  refuses any other row); record rows go through the loop forward only (`Trainer.losses_on`);
  `MAX_STEPS` is enforced per model (`smoke.check_step`/`record_step`: a count the model and its
  copies carry), so a second `Trainer`, a smoke run or the published copy cannot add steps.
- **I19's cutoff as Lc0 defines it** (the main loop's ruling, issue 236(d)): `TemperatureSchedule`
  now holds 0.9 before the cutoff turn 60 and a constant endgame temperature 0.45 from it
  (`endgame`; `temp-endgame` "doesn't decay"), a decay before the cutoff optional (`decay_to`,
  off). Commit 3 had decayed to 0.45 and cut to temperature 0 at turn 60.
- **I13 whole in the tree form** (`train/targets.py`, issue 235(g)): a best move-child the search
  never evaluated (inherited priors) had no network moves-left, so the WDL was blended and
  moves-left was not. It now takes the decision's own raw moves-left (`best_child_moves_left`);
  with no estimate at all moves-left is masked from that decision back.
- **Smaller**: `Trainer.publish` gives the network its SWA snapshot count, which the records'
  network condition now carries (it said 0); the generator records the prior-to-target divergence
  under the builder's defaults when given no target configuration (it was NaN), and takes an
  `on_decision` hook (the live search, the record, the input) for the tests; `synthetic_record`'s
  float32 fields are float32-exact, so it round-trips through a file.
- **Tests** - each tied to a finding, each mutant it names now killed: the value targets (search
  value, categorical, root WDL, short-term) from the pruned visits by hand; the builder's KL as
  KL(target || raw prior) on an asymmetric case and with the header's submit share; the replica's
  inputs (E3's factor, the search's and the noise-free prior, the root's recorded visits);
  short-term targets of a resigned game by hand; the submit child's moves-left offset; the tree
  form's deblundering (constructed, and a generated tree-form game); I20 running to the cap on a
  root that keeps moving; I18's sign (a value inside the threshold never resigns) and its 10%
  (the default, and the disabled share over 4000 seeds); each record held to the live search at
  generation (the root's cPUCT with E2 off and on, the best child, the place in the action, the
  pruned and unpruned policy target and root value); forced and steered counts through a file;
  newer, older and missing versions refused; another ruleset refused; a rebuilt input compared
  field for field with its W/E scope, under a window other than 8; a resigned record replayed and added to a trainer; a trainer of three other
  configurations rebuilding the generator's inputs under its own; the step cap per model and the
  synthetic-rows rule; the miniature loop with its steps on synthetic rows and the snapshot count
  in the next record.
- Spec: §12.2, §12.5, §12.8 (I15's records, deblundering, the exploration bullet), §15.1's M21
  sentence; register I2, I13, I15, I17, I18, I19, I20, M21. Issues 234(f)-(g), 235(a) and (g),
  236(d), 237(a), (e), (f); proposed edits 175, 176, 178 and 181 extended (181 now quotes Lc0's
  options). BUILD_LOG's I13, I15, I19 and M21 rows updated (M21 kept qualified: the QAT hook a seam
  until G8, the user ruling on edit 182).

### 2026-10-23 - D2's flat form, part 3: the turn generator's successors under Q3's play activation (found by the long run of spec 15.2's row)

`long_turngen_equals_the_brute_force_reference_on_tiny_positions` (3,000 tiny multiverses,
`t5d-tests --long`) failed in 10 positions, every one under Q3's play activation (the general
realisation): as successors, the generator emitted fewer states than the brute-force reference
(4 of 5 ... 1,056 of 1,064). Under Q3 two orders of one set of moves can reach two states with
one action key; the generator's run counted actions, so `record()` kept the first order's
action and dropped the second state's (inserted, with no action kept). The default run's 300
positions had not met the case.

- **`engine/src/hypercuboid.cpp`** (the generator only; the detector's pin holds): under
  `TurnUnit::Successors` the generator's run now counts successors - each state kept with an
  action reaching it - and the turns are collected from that listing; a point the budget stopped
  part-way emits none of its states under Successors (their kept actions are not final), so a
  stopped generator's output stays a prefix of the unstopped one's.
- **`engine/tests/test_turngen.cpp`**: the best-action-per-state check reads the listing's
  actions where they reach the state (the listing keeps one order per action, and under Q3
  another order can reach another state).
- Re-run: the long tiny-multiverse row (3,000 positions, 721,906 turns) and the long variants
  row (652 positions) pass, the default rows and the flat search's tests too; the pin is
  unchanged. *(Review correction, 2026-10-23: the generator's fix left the search arm
  marking a Q3 node complete from the Actions unit or the fallback's listing, which keep one
  state per action key - fixed by the review fixes.)*
- Docs: docs/BUILD_LOG.md (issue 163's note; the D2 row).

Spec: §15.2 (the flat generator's row - a defect its long run found, fixed); §8.2; register D2.
Spec issue 163 (the unit, as built).

### 2026-10-23 - M21 (4/4): the training loop end to end, smoke-tested - records, sampler, window, targets, losses, optimiser steps, SWA

- **`python/t5d/train/trainer.py`** (new): `Trainer` - a game's I15 record in (its inputs rebuilt
  from the record, its targets built, its rows weighted by I12's sampler and written into I16's window
  as frequency copies), a batch drawn from the window, collated, every head's loss, one optimiser step,
  SWA's update (I17); `publish` gives the averaged network self-play receives. At most `MAX_STEPS` (3)
  optimiser steps - no training runs. AdamW or NAdam, placeholders (issue 237). **G8's QAT hook is a
  named seam, not built** (`TrainerConfig.qat`, `apply_qat` raises until the precision component's
  quantiser and the pinned format exist - for the user, edit 182). **C3's conditioning-offset training
  stays parked** with the cached configuration. `synthetic_record` builds a well-formed record over
  synthetic inputs.
- **Tests** (`python/tests/train/test_trainer.py`): two tiny self-play games with a random network ->
  records written and read back -> three optimiser steps (every loss finite, a fourth refused) -> the
  published network differs from the raw one and drives self-play, its hash in the record; the same
  loop on synthetic records (AdamW and NAdam); the total loss falling at each of three steps on a fixed
  synthetic batch; the step cap and the QAT seam refusing.
- *Corrected by the review fixes (2026-10-23, above)*: the inputs were rebuilt under the record's
  network configuration, not the trainer's; the optimiser steps on self-play records broke the
  user's rule (steps on synthetic data only), and the cap was per `Trainer`, not per model; the
  published network's records said `swa_snapshots` 0.
- Spec: §15.1's M21 sentence, §10.2, §12.8; register M21, G8, C3, I17. Issue 237; proposed edit 182.
  BUILD_LOG's M21 and G8 rows and the C3 park updated.

### 2026-10-23 - M21 (3/4): section 12.8's mechanics - I12's sampler with 6u's arms, I16's window, I17's SWA; PCR, I20, I18 and I19 in the generator

- **`train/selfplay.py`**: playout cap randomisation (I5, `PCRConfig`: each decision full with
  probability 0.25, re-derivable from the seed; cheap searches at a smaller cap without noise; the
  difficulty mixture held below 1 and not built above 0 - issue 236); KL-adaptive full searches (I20,
  `KLAdaptiveConfig`, a prototype, off: chunks until the root's visit distribution stops moving,
  bounded by the full cap, the stop recorded as `kl`); resignation with the false-resign check (I18,
  `ResignConfig`: off in 10% of games by the seed, would-resign events recorded, `false_resign_report`
  per board-count range; the threshold None until the value head is calibrated); I19's schedules
  (`TemperatureSchedule`: the move temperature 0.9 -> 0.45 over 60 turns, 0 from turn 60; the root
  policy temperature 1.25 -> 1.1). The full cap's default is now 800 (a placeholder). A resignation
  ends the game `resign`, the other side winning, with the record's last decision playing nothing.
- **`train/sampler.py`** (new): I12's surprise weighting - half uniform, half in proportion to KL from
  the raw prior to the pruned target, a small value-surprise share - per game, written as frequency
  copies at full weight; 6u's arms: uniform, I6's raw variance and persistent bucket (`BucketTracker`).
- **`train/window.py`** (new): I16's sublinear replay window, KataGo's shape and constants as
  placeholders. **`train/swa.py`** (new): I17's capped running weight average and the published network.
- **Tests** (`python/tests/train/test_mechanics.py`; `test_records.py` and `test_targets.py` updated for
  PCR): the surprise weights by hand (mean 1, uniform on equal surprise, in proportion to the KL, value
  surprise, cheap rows), steering kept out of the surprise by the pruned target, the copies seeded and
  unbiased, 6u's arms; the window's growth (identity below c, monotone, sublinear, KataGo's 22M) and
  recency; SWA's capped mean and publication; the temperature schedules; PCR's seeded fraction and a
  game mixing full and cheap searches; I20 stopping on a quiet root; a resignation (record, result,
  5dpgn replayed by the engine) and the check's events and per-range rates.
- *Corrected by the review fixes (2026-10-23, above)*: "0 from turn 60" contradicted the Lc0
  semantics it cited (the endgame temperature 0.45 holds from the cutoff, undecayed); the I20 test
  could not tell a quiet root from a moving one; I18's sign and its 10% default were untested.
- Spec: §12.3, §12.5, §12.8, §13.1 6l, 6u, 6z; register I3, I5, I6, I12, I16, I17, I18, I19, I20, M21.
  Issue 236; proposed edits 179-181. BUILD_LOG's rows for these and M21 updated.

### 2026-10-23 - D2's flat form, part 2: the search arm; the enumerating whole-turn arm and its 256 cap retired

The flat form's search arm, on part 1's turn generator (§8.2: "PUCT runs over the generated
turns, and progressive widening pulls in the next-best legal turn as visits grow. This
replaces the enumerating whole-turn arm and its cap, and ends turns itself"), and the
retirement of the arm it supersedes (register D2, CHECKPOINT 6).

- **The arm** (`engine/include/t5d/mcts.hpp`, `engine/src/mcts.cpp`): `TurnSplit::Flat`, in
  slot 1 of the retired `WholeTurn`. A node is a turn start; its children whole legal turns
  generated best first, instantiated by widening (ceil(k0) at expansion; `widen` pulls the
  next-best as k(N) grows; with widening off, all). One evaluation per turn start: the move
  logits (the leaf's candidates classified as the ply split's - D1) and the new per-board
  "leave it" logits (`Evaluation::board_logits`, one per expansion board) give each playable
  board a log-softmax distribution over its moves, the non-branching jumps arriving on it and
  leaving it, at the policy temperature (the root's too); a turn's score is its options'
  log-probabilities summed, its prior their softmax over the generated turns. Every unstated
  reading is a parameter with a placeholder default, logged: `flat_unit` (successors),
  `flat_prior` (Generated; Product), `flat_jumps` (both boards; source only), `leave_share`
  (0.5 where no leave logit is given - the network has none yet: a documented fallback, not a
  refusal), `flat_k0_max` (16, on k0 from the boards' summed entropy), `generator_nodes`
  (200,000 a node), `flat_fallback` (64: a node whose generator stopped before its first turn
  takes E4's listing's turns, scored and ordered alike). Root noise per board before
  generation; the side table's noise-free priors from the noise-free scores. The generator
  ends turns itself (D8): no submit logit read. Terminal verdicts stay `detect_mate`'s; a
  stopped generator's node is incomplete, never proven lost (H37). Memory: each node counts
  its generator (`memory_bytes`, within x1.23 of the heap). A flat invariant in
  `check_invariants` (edges its turns in non-increasing score, complete only when the
  generator is done). *(Review correction, 2026-10-23: under Q3's play activation a node was
  marked complete with a successor missing - as actions, or from the fallback's listing -
  so "never proven lost" did not hold there with the proof channel on; fixed.)* Stats: `generated_turns`, `stopped_generators`, `fallback_listings`,
  `seconds_generate`.
- **The generator** (`engine/src/hypercuboid.cpp`, new code only - the detector's pin holds):
  ties now newest-first (depth-first; first-queued-first split a tying 66-board position into
  2^18 equal boxes), a point's realisations at most 32 per judgement (a point creating k
  timelines has up to k! - 20,160 successors at one 90-board node), its memory estimate per
  successor from the boards its action made. *(Review correction, 2026-10-23: each judgement
  walked the point again from its first realisation - about n^2 / 64 realisations for a
  point of n, 401,936 for 5,040 - so the default budget emitted only 1,824 of them; the
  point is now resumed where the cap stopped it.)*
- **The C API and its mirror** (`capi_search.h`, `capi_search.cpp`; `python/t5d/engine/capi.py`,
  `search.py`, `evaluator.py`): `T5D_SPLIT_FLAT` (1); `t5d_evaluations.board_logits`, the
  channel reserved in ABI round 2, now read - each leaf's `board_count` floats, NaN for none,
  checked under every split, read under the flat form; params appended (`flat_unit`,
  `flat_jumps`, `leave_share`, `flat_k0_max`, `flat_fallback`), and the whole-turn arm's slots
  kept and renamed (`flat_prior` - 0 is the old sum's softmax over the listing -,
  `retired_max_actions` - 0 or `T5D_NO_LIMIT` only -, `generator_nodes`); the summary's
  `generated_turns`, `stopped_generators`, `fallback_listings`. Python: `Search.deliver(...,
  board_logits=)`, `Evaluation.board_logits` (the network gives none; `UniformEvaluator`
  zeros, as the engine's), `CStepDriver` sends them, NaN for a leaf that gives none.
  *(Review correction, 2026-10-23: `CStepDriver`'s docstring still said it passed no board
  logits, and it checked only the batch's totals, not each leaf's counts; both fixed.)*
- **The built-in evaluators** give leave logits where the flat form asks (`EvalRequest::
  board_logits`): uniform and material 0, random random.
- **Retired** (the whole-turn arm): `TurnSplit::WholeTurn`, `ActionPrior`,
  `SearchParams::max_actions` and `listing_nodes`, `list_actions`, `SearchStats::
  truncated_listings` and `seconds_listing` (now `seconds_generate`; `GameRecord::
  seconds_search_listing` now `seconds_search_generate`), `T5D_SPLIT_WHOLE_TURN`,
  `T5D_ACTION_PRIOR_*`, `t5d-play --split turn` (refused, naming `--split flat`) and
  `--listing-cap`, netplay's `"turn"` split and `max_listing` (`"flat"` instead; its CLI's
  default arms ply and flat). `list_legal_actions` and its cap stay (E4, the gate adapter,
  certificates, the prover, the scanner, and the flat form's fallback).
- **Tests**: every whole-turn test moved to the flat form - two flat configurations in
  `kConfigs` (the invariants, re-rooting, the solver's proofs, MCGS keys, values, D3), the
  root's children with widening off the exact legal turns (successors and actions, both
  priors), a stopped generator never proven lost, the fallback (and stuck without it),
  *(review correction, 2026-10-23: that test's comment said each child's score was checked
  as its moves' per-board log-probabilities, but only the order, the prior sums and
  Product's exp(score) were; the scores, widening's count, the fallback's scores and
  de-duplication and the generators' cost on ties were untested - six of reviewer B's 31
  mutants of the per-board policy survived; tested since)*
  parameter refusals (C++ and C), self-play and match play under the flat form, `t5d-play
  --split flat` records replayed through `t5d-gate`, the C step loop equal to the C++ search
  by digest with board logits delivered (NaN on a third of the boards), memory within
  0.75-1.33 of the heap; in Python the engine's random evaluator through the step API bit for
  bit (leave logits included), NaN leave logits the search with none, the ply split reading
  none, the network driving a flat search deterministically, netplay's flat games replayed.
  `engine/tests/test_mcts.cpp`'s re-rooting test now picks a sibling with as many boards as
  the new root (a flat turn that branches makes more).
- **`t5d-tree-count`** gains the flat form as a fifth arm; its counts on the seven
  t0database turn starts in docs/MEASUREMENTS.md (deepest evaluated turn start 10-12 turns
  against the ply split's 3-7; 0-8 of ~400 generators fell back).
- Docs: engine/README.md (the arm, the retirement, the C API, t5d-play); docs/BUILD_LOG.md
  (rows D2, D8, E1; spec issues 164-171; issues 81, 92, 93 and edit 69 closed or overtaken;
  proposed edits 132-134); docs/MEASUREMENTS.md. *(Review correction, 2026-10-23: the
  README's C API paragraph still called `board_logits` reserved and `T5D_ERR_UNSUPPORTED`;
  rewritten.)*

Spec: §8.2 (the flat form - built; proposed edit 132), §8.4 (constants list and coupling -
proposed edit 133), §11.0 item 3, §15.2 (the flat row's arm tested), §15.3 (proposed edit
131, part 1), §15.4 (turn edges - proposed edit 134); register D2 (flat form built, the
whole-turn arm retired), D1, D8 (the generator ends turns - built), E1 (the flat form's k0,
noise and temperature). Spec issues 164-171 (placeholders and the generator's worst case).

### 2026-10-23 - M21 (2/4): the targets from I15's records - I2's pruning, D3's and D4's targets, I13, I14's refusal; section 15.2's round trip

- **`python/t5d/train/targets.py`** (new): every head's targets from a record alone, in float64,
  deterministically. The policy target from full searches only (I5) with KataGo's forced-playout
  pruning applied at target construction against a replica of the root's PUCT (§12.5), steered
  visits excluded or, under the provenance arm, down-weighted (I2, §12.2; the counts are 0 until
  E10); the soft-policy target; the search-value targets - D3's categorical over the root search
  value of full searches with 6j's short-term arm as a switch, the short-term values at the heads'
  horizons, the q_ratio blend's root WDL - from the pruned visits (I2, CHECKPOINT 7); WDL on the
  result and moves-left in turns (issue 224's placeholder: the turn in progress counts), both
  deblundered (I13: Lc0's threshold 0.1, the band a placeholder); travelled-into and
  timeline-active-at-end per board, tactical volatility masked until proofs exist; the effort label
  from the change point (issue 154); the reply prototype through `reply_map` from the rebuilt
  inputs; D9's weights stay the loss's. I14's proof rescoring refused while
  `PROOF_CHANNEL_SIGNED_OFF` is False. `TARGET_SOURCES` names each target's source - none a learned
  verifier (D5). `collate_rows` turns record rows into the network's batch and the losses' `Targets`.
- **`train/selfplay.py`**: `generate_game(target_config=...)` builds the targets at generation and
  records each decision's prior-to-target divergence and the configuration.
- **`train/losses.py`**: a missing target is masked - a NaN WDL row (an unscored game), moves-left
  (a censored game), per-board or per-square label; unchanged where every target is present.
- **Tests** (`python/tests/train/test_targets.py`): pruning against KataGo's amount derived in
  closed form, the best child untouched, the submit child pruned like a move, a one-visit remainder
  dropped, exclusion and provenance, the boost taken out; the pruned root value, soft policy, raw
  prior and KL by hand; WDL, moves-left, short-term targets and deblundering (rewrite, band,
  perspective, moves-left offset) on constructed games; travelled-into by brute force; effort
  labels; the I14 refusal; the losses' masks. **Section 15.2's I15 row**: on generated games under
  three game configurations and three target configurations, the targets re-derived from the stored
  record (its inputs rebuilt by replay) equal the targets written at generation, bit for bit, the
  reply maps included; record-derived batches give every loss finite, with and without the reply
  prototype (forward only).
- *Corrected by the review fixes (2026-10-23, above)*: no test pinned the search-value targets to
  the pruned visits, the builder's KL direction, the PUCT replica's inputs (E3's factor, the prior,
  the root's visits) or the moves-left offset of a submit best child, though this entry claimed
  them; the I13 blend was half-applied in the tree form (WDL without moves-left); short-term
  targets of a resigned game were untested.
- Spec: §8.3, §8.6, §12.2, §12.4, §12.5, §12.8, §15.2's I15 row; register I2, I5, I13, I14, D3, D4,
  D5, D9, M21. Issue 235; proposed edits 176-178. BUILD_LOG's I2, I13, I14 and M21 rows updated.

### 2026-10-23 - M21 (1/4): I15's training records and the self-play generator that writes them

Register M21's first part, in the net lane: the record format before anything that reads it.

- **`python/t5d/train/records.py`** (new): register I15's record, format version 1 - one
  compressed `.npz` per game (`allow_pickle=False`), a JSON header with the conditions (engine
  version, ruleset, variant or 5DFEN start, the network's weight hash and configuration, every
  `t5d_search_params` field - the legality budget among them - the input's window and relations,
  the generator's configuration, the result, the 5dpgn, the resignation events, the proof
  references: none, the proof channel being off), per decision the raw root outputs (WDL and
  categorical logits, a raw logit per candidate move and for submit, error, moves-left), the root
  search value (q, d), visits, cPUCT, the effort change point, the best and played children, the
  move and root temperatures, the PCR flag, cap and playouts, the seed and the prior-to-target
  divergence's slot, the decision's boards and candidate moves, and every root child's statistics
  (visits with the forced and steered counts, both priors, (q, d), value variance, E3's factor,
  E10's boost, its key and own network evaluation, the solver bounds). Another version is refused
  on reading; `record_differences` compares two records bit for bit.
- **`python/t5d/train/selfplay.py`** (new): the generator - netplay's self-play loop (turn starts
  judged first, a fresh search per decision sharing the cache, noise on, a chosen child made known
  legal) writing a record per game; the board-move split and the tree form (the whole-turn arm
  writes no policy target and is refused); D9's optimistic evaluator refused (`self_play=True` on
  every search); `replay_positions` rebuilds each decision's network input from a record alone.
  The raw root outputs are the network's evaluation of the decision's own input (issue 234's
  reading, not a memo of the search's). `engine/netplay.py` unchanged.
- **Tests** (`python/tests/train/test_records.py`): records written and read back bit for bit
  under four configurations (the ply split, the tree form, Two Timelines, a tight legality budget
  with unknown moves) and a custom 5DFEN start with branched timelines; each decision's input
  rebuilt from the stored record, digest for digest; the raw WDL equal to the engine's root WDL
  and the search's noise-free prior the softmax of the raw logits; the conditions present; another
  version refused; the weight hash following the weights.
- Nothing trains: the generator plays tiny games with random weights in the tests.
- *Corrected by the review fixes (2026-10-23, above)*: the bit-for-bit round trip cannot see a
  value recorded wrongly at generation (the cPUCT, the best child, the place in the action), the
  forced and steered counts were 0 in every tested record, "another version refused" was tested
  one way only, and replay ignored the header's ruleset and rules and the reader's network
  configuration.
- Spec: §12.8 "Rich training records", §15.2's I15 row (the targets' half lands with the target
  builder); register I15, M21, D2, D8, D9, H23/H41. Issue 234; proposed edit 175. BUILD_LOG's I15
  and M21 rows updated.

### 2026-10-23 - D2's flat form, part 1: the turn generator - the hypercuboid search best first (spec 15.2's row)

The flat form's generator (§8.2: "A generator assembles complete legal turns best-first by
summed logits, through the rules engine ... The hypercuboid search already walks this
product space for mate detection; ordered by prior, it becomes the turn generator"), the
first half of D2's flat form; its search arm, which retires the enumerating whole-turn arm,
follows.

- **`t5d/turngen.hpp`, `src/hypercuboid.cpp`**: `TurnGenerator` - the legal turns of a turn
  start, one at a time, in non-increasing score. A `TurnScorer` scores each option of each
  playable board (leaving it - the detector's Pass, §8.2's "leave it"; a move as its source
  board's option, a non-branching jump also as its destination's - "a move between two present
  boards resolves both"; J6's twin as its jump); a turn's score is the sum of its options'
  over the boards in ascending timeline. The generator is the hypercuboid detector's search
  with its stack replaced by a priority queue of boxes keyed by their bound (the sum of their
  axes' best remaining scores), each box narrowed as the detector narrows it and its best point
  judged by the detector's own `judge` - so the points are judged, and the turns emitted, in
  non-increasing score, and the search is complete by the detector's argument. Units: actions
  or distinct successor states (`TurnUnit`); ties deterministic; one budget over the
  generator's life, a stopped generator never exhausted (H37). New methods only in the
  detector's class: its depth-first search (`detect_mate`, the counts, the listings, the
  legal-move tiers) is unchanged.
- **The depth-first search pinned** (`engine/tests/test_detectors.cpp`,
  `detectors_the_hypercuboid_search_is_pinned`): every answer and statistic of the
  hypercuboid detector's (verdicts, witnesses, counts, listings, legal moves, the ways to
  finish after each first move; nodes, points, regions, boxes removed, the most positions
  held) on 571 positions - 300 tiny multiverses under random rulesets, random games from the
  sparse starts under five rulesets and from every named variant - unlimited and under node
  budgets that stop it, hashed; the hash recorded at 2f07079, before the generator, and equal
  after it. `tiny_position` moved, unchanged, from test_detectors.cpp to
  `engine/tests/test_tiny.hpp` to be shared.
- **Spec 15.2's row** (`engine/tests/test_turngen.cpp`): on 300 tiny multiverses (265 with
  two playable boards or more; all three regimes), five constructed multi-board presents
  (jumps between present boards, a rewinding branch) and random games from every variant's
  start, under continuous scores and small integer ones that tie, the whole output is the
  legal turns exactly, each once (against the brute-force reference and the engine's
  exhaustive listing, as actions and as successors), in non-increasing score, each score
  recomputed from the turn's moves alone bit for bit, equal to the sorted scores of every
  legal action, each successor's its best action's, deterministic, and a budget-stopped run a
  stopped prefix of the full one; `long_` runs ten times as many. *(Review correction,
  2026-10-23: no position of the row was shown to hold a castling or an en passant turn -
  tested since the review fixes, under three rulings; and a stopped generator's `bound()`
  could understate the turns it had not emitted - fixed.)*
- Docs: engine/README.md (the generator's design beside the hypercuboid's); docs/BUILD_LOG.md
  (the D2 row; spec issue 163; proposed edit 131).

Spec: §8.2 ("Flat form"), §11.0 item 3, §15.2 (the flat generator's row - built and passing),
§15.3 (a new row proposed: proposed edit 131); register D2 (its row in the build log; the
register's own row unchanged until the flat arm is built). Spec issue 163 (the generator's
unit, ties and budget - placeholders).

### 2026-10-23 - Network heads review fixes: the error target and D9's z as the register and KataGo have them; the losses' masks, moves-left's unit and the old parameters pinned

Two reviewers of N-D3, N-D4 and N-D9 (spec fidelity; 46 mutants). One high finding, three medium,
nine low; all fixed.

- **HIGH - the error head's target and D9's z** (`train/losses.py`): both had compared the
  short-term target with the network's W - L and called that KataGo's; it is not. KataGo's
  short-term error loss compares the detached short-term value prediction with the short-term
  target (Huber delta 0.4), and its optimistic z_value is (short-term target - short-term
  prediction) / sqrt(predicted short-term error + 1e-4) - which is also register D9's literal
  "how far the mover beat the predicted short-term value". Rebuilt that way:
  `error_target(predicted, short_term, horizon)` = (short-term output[h], without gradient -
  target[h])^2, and `optimistic_weights(short_term, predicted, value_error)`'s z over the
  short-term output at the error head's horizon. The Huber delta stays a named placeholder (1.0;
  KataGo's 0.4 noted). Section 8.3's "the value and the short-term value" is ambiguous: the W - L
  reading is recorded as the alternative for the user (issues 231(c), 233(a) and edits 172, 174
  rewritten); every false "KataGo's" removed (`losses.py`, `heads.py`, the tests' docstrings).
- **MEDIUM**: the error loss pinned to the Huber loss against the built target over the finite
  rows, at two horizons; the short-term loss pinned to the mean over the finite entries only, a
  row with no target getting no gradient (`tests/train/test_value_heads.py`); moves-left's unit
  across the network -> evaluator -> search boundary shown with a head biased to log1p(37) - the
  evaluation and the search's record of its root read 37 turns (`tests/engine/test_evaluator.py`;
  the random head's outputs clamp to 0, so the old checks could not tell turns from log1p).
- **LOW**: the pre-D3 parameters pinned - a fingerprint of the 64 parameters a small network had
  at 47bc857 (seed 7), recorded there, under the default flags and with every prototype on
  (`tests/net/test_params.py`; three float64 sums, not a hash of the bytes, so as not to depend
  on float32 libm across platforms); D9's submit mask tested on an offered but illegal submit
  (`tests/net/test_model.py`); the reply map's `ambiguous="turn"` arm on a reply with one masked
  and one clean played move, and each mapped move's piece checked against the decision board's
  same cell (`tests/train/test_reply_map.py`); the cache's equivalence test compares the submit
  readout and the masked submit logits (`tests/net/test_cache.py`); each `Evaluation` names its
  prior (`policy`), `check_evaluation` refuses an unknown one, and `serve` / `run_search` take
  `self_play` and refuse an optimistic evaluation there, however the evaluator is wrapped -
  `play_game` sets it, M21's generator must (`engine/evaluator.py`, `engine/netplay.py`); the
  match record's Engine header names the optimistic prior; issue 233(e) says "no I15 training
  record"; the effort loss reads full searches only (§12.4's "full-search moves only"); stale
  `heads.ValueHead` references (`net/config.py`, `net/trunk.py`).
- **The mutants re-run** (the reviewer's import-hook set; those whose code the high fix rewrote
  retargeted at the new code): every one tied to a finding is caught (err_catmean, st_no_nanmask,
  ev_ml_log, no_retired_draws, d9_submit_possible, reply_turn_arm), as are d3_global_stream by the
  new pin and two new ones (the `serve` guard, the effort mask). `play_game`'s own optimistic
  check, disabled, now goes unnoticed: `run_search` refuses the same evaluator with the same
  message - a second layer, kept because it refuses before the game is opened.
- F1 unchanged (22,965,426): no parameter added or moved.
- The three heads' entries below corrected where they said what was not so.
- Spec: §8.1 (D9's z), §8.3 (the error head), §12.4 (the effort row), §15.2's D4 row; register
  D3, D4, D9. Issues 231(c), 233(a), 233(e) rewritten; proposed edits 172 and 174 rewritten.

### 2026-10-23 - Network D9: the optimistic policy head behind its flag

Register D9 (§8.1, CHECKPOINT 7) - a prototype, behind its flag, off; match play only.

- **The head** (`net/model.py` `T5DNet.optimistic`, a `heads.ActionHead`;
  `NetConfig.optimistic_policy`, off; `optimistic_policy_dim` 256): its own f and g, D1's
  promotion offset and its own submit logit, masked as the policy's (`NetOutput.optimistic_logits`,
  `optimistic_submit_logits`, `optimistic_action_logits()`); drawn from a generator of its own
  (`model.d9_seed`) - switching it on changes no other parameter (tested). 135,182 parameters when
  on (F1 23,100,608); the v1 is unchanged at 22,965,426.
- **The loss** (`train/losses.py`): its actions against the normal policy target over the full
  searches, each sample weighted by `optimistic_weights` - clamp(0, 1, sigmoid((z - 1.5) x 3)),
  z = (short-term target - (W - L)) / sqrt(error head + eps) at the error head's horizon, the
  network's outputs without gradient (*corrected by the review fixes above: the baseline is the
  network's short-term output, D9's and KataGo's; W - L is the alternative*); `LossConfig.optimistic_centre`, `_scale`, `_eps`; weight 1.0.
  Placeholders: spec issue 233.
- **Match play only** (`engine/evaluator.py`, `engine/netplay.py`): `NetworkEvaluator` and
  `CachedNetworkEvaluator` take `policy="normal" | "optimistic"` (`POLICY_SOURCES`,
  `policy_source`) - the optimistic one delivers the optimistic head's logits as the prior with the
  same value heads, and is refused on a network without the head. `play_game` refuses an
  optimistic evaluator in self-play; `PlayConfig.optimistic` is match-only and both sides must
  deliver it; the CLI's `--optimistic` needs `--mode match` alone.
- **Tests**: the weight (0.5 at z = 1.5, towards 1 and 0, no gradient), the loss only under the
  flag and reaching only its own head, three optimiser steps reducing it
  (`tests/train/test_policy_heads.py`); the head, its own draws and endpoints
  (`tests/net/test_model.py`); cached = dense and the A7 mirror with it on; the optimistic evaluator,
  dense and cached (`tests/engine/test_evaluator.py`); self-play refusing it and a match with it
  replaying legal through `t5d-gate` (`tests/engine/test_netplay.py`).
- Spec: §8.1, §12.4 (the optimistic row), §14.4 (held open in self-play), §13.1 6t; register D9,
  D1, C14. Spec issue 233; proposed edit 174.

### 2026-10-23 - Network D4: the soft-policy head and the opponent-reply prototype

Register D4 (§12.4, CHECKPOINT 7, reversal 29) in the network.

- **The soft policy** (`net/heads.py` `ActionHead`, `AuxHeads.soft_policy`; `NetConfig.soft_policy`,
  on - 6j's on/off arm): "a second pointer head - its own source and destination projections and
  its own submit logit": f and g at width 128 (`soft_policy_dim`, a placeholder), D1's promotion
  offset, and a submit readout of the pooled global registers in `submit_head`'s form, masked
  exactly as the policy's (`NetOutput.soft_policy_logits`, `soft_submit_logits`,
  `soft_action_logits()`). Loss (`train/losses.py`): its own actions against the policy target to
  the power 1/4, renormalised (`soft_policy_target`, `LossConfig.soft_policy_power`), full searches
  only, weight 8 (the policy's 1).
- **The opponent-reply prototype** (`NetConfig.opp_reply`, off): its pointer head scores
  `PositionState.opp_moves`; loss `distribution_ce` against `Targets.opp_reply`, now the turn's
  distribution `(B, M2)` (positions without one skipped), weight 0.15. `train/reply_map.py`
  (`map_reply`) builds the move space: the opponent's next turn mapped onto one of our decision
  points - each reply board to its nearest ancestor there through the engine's lineage, the same
  square; every move touching a board two distinct reply boards map onto masked; candidates the
  mapped moves of every reply decision point; the target uniform over the played moves.
  `opp_moves` docs updated (`net/state.py`, `engine/convert.py`); inference supplies none.
- **The other pointer heads' offset**: `NetConfig.pointer_promotion_offset` (on, a placeholder,
  issue 222).
- **Draws**: each head from a generator of its own (`model.d4_soft_seed`, `d4_reply_seed`); N1's
  always-built reply head's draws kept in the global stream - every existing parameter unchanged
  (checked against the previous commit under three configurations), and switching either flag
  changes no other parameter (tested).
- **Parameters**: F1 22,965,426 (+69,390 the soft policy, -65,792 the reply now off; 23,034,559
  with the reply on), F2's L24-ff12 46,894,834; §9.2's aux budget 76,586. `netplay.SMALL` gives the
  soft head width 32.
- **Tests**: `tests/train/test_reply_map.py` - §15.2's D4 row on turns played through the engine
  from handwritten 5dpgn (a branching jump with White and with Black acting; two branches into one
  past board in Two Timelines, ambiguous and masked at the turn's start, mapped at the decision
  point between the jumps), each mapped square checked against the test's own lineage walk (the same algorithm, so not
  independent: *the review fixes above add a content check*), and the
  mapped space fed to the reply head; `tests/train/test_policy_heads.py` (the quarter-power target,
  the soft loss at weight 8 reaching only its own head, full searches only, the reply loss only
  under its flag at 0.15, a few optimiser steps reducing each); `tests/net/test_model.py` (each
  pointer head's own endpoints and offset, the soft submit logit masked like the policy's in both
  readout forms, the flags changing no other parameter); the A7 mirror (a reply arm added), cached =
  dense (a reply and pointer-submit arm), invariance and smoke tests over the new outputs;
  `tests/net/test_params.py`.
- Spec: §8.1 (promotions), §9.2 (Params; Aux heads), §12.4 (soft policy, reply), §12.5, §15.2's D4
  row; register D1, D4, D7, F1. Spec issue 232 (new) and 222 (built note); proposed edit 173; edit
  156's running figure updated.

### 2026-10-23 - Network D3: the value heads as corrected at CHECKPOINT 7

Register D3 (§8.3, corrected at CHECKPOINT 7, reversal 28) in the network: the 51 bins had been
the only value and trained on the outcome; now four heads, as the engine's search (ABI round 2a)
already consumes them.

- **The heads** (`net/heads.py` `ValueHeads`, `net/model.py`): over the global registers' mean,
  FP32, the acting player's perspective - one shared hidden layer with Mish (N1's value MLP), then
  **WDL** (the value search uses: `NetOutput.search_value()` = W - L, `wdl_probs()`), **the
  categorical** (51 bins, over the root search value; `categorical_mean()` and
  `categorical_variance()` - the search's default uncertainty signal, 6m - replace
  `value_mean()`/`value_variance()`), **short-term values** at `NetConfig.short_term_horizons`
  (2, 5 and 16 turns, placeholders; `NetOutput.short_term`, tanh) and **the error head**
  (`NetOutput.value_error`, softplus - the name the evaluator reads, `ERROR_HEAD_FIELD`). N1's
  WDL head moved onto the shared layer; its 51-bin short value head is gone. `heads.hl_gauss`
  (HL-Gauss targets) and `heads.wdl_value` added.
- **Draws**: the categorical and WDL readouts are drawn where N1's value output and WDL head
  were; the short-term and error outputs from a generator of their own (`model.d3_seed`); the
  retired short value head's draws are kept in the global stream (`T5DNet._retired_draws`, and
  its construction-time draw), so every parameter that existed before is unchanged, bit for bit
  (checked against the previous commit's weights under three configurations).
- **Losses** (`train/losses.py`): `Targets` takes the game result's WDL (class index or
  distribution), `root_value` (the root search value of full searches, NaN where none),
  `short_term` (per horizon, NaN where none), `full_search` (section 12.5's mask - the policy and
  the categorical train on full searches only) and `root_wdl` (for `LossConfig.q_ratio`, 0). The
  categorical's loss smooths the root value by HL-Gauss (sigma 0.75 bin widths); the error head's
  target is `(short_term[h] - (W - L))^2` with the network's value taken without gradient
  (`error_target`, `error_horizon` 0), Huber (*corrected by the review fixes above: the
  short-term output against its target, register D9's and KataGo's form*). Loss names `categorical`, `short_term`,
  `value_error` (was `value`, `short_value`). Every unstated detail a placeholder (spec issue
  231). `train/smoke.py`'s synthetic targets follow (every third position a cheap search).
- **The evaluator** (`engine/evaluator.py`) now delivers the error head's output with the WDL,
  the categorical and moves-left; docs updated.
- **Parameters**: F1 22,961,828 (was 22,973,907: three short-term outputs +771 and the error head
  +257, the short value head -13,107), F2's L24-ff12 46,891,236.
- **Tests**: `tests/train/test_value_heads.py` (HL-Gauss, the WDL target and its blend, the
  categorical on full searches only, no gradient from the error target into the WDL or
  short-term heads, missing targets skipped (only shown finite: *pinned by the review fixes*),
  three optimiser steps reducing each loss alone);
  `tests/net/test_model.py` (shapes, `search_value`, the heads FP32 under a BF16 autocast against
  an FP64 recomputation, the horizons' validation; the register-pool and permutation tests cover
  the four); `tests/engine/test_evaluator.py` (the error head read by its pinned name, and the
  network's heads reaching the search through the C step API - the search's record of its root
  equals the network's evaluation, the 6m signal at every leaf under either source); the A7
  mirror, cached = dense, purity and turn-state tests extended to the new outputs.
- Spec: §8.3, §8.4 (6m's signal), §9.2 (Params), §12.4 (value rows), §12.5; register D3, F1, I10,
  D7 (pooling unchanged). Spec issue 231; proposed edit 172; edit 156's running figure updated.

### 2026-10-23 - D2 tree form review fixes: a carried inherited root is evaluated; the inherited value's refusal; the untested readings tested; the counts committed

Two reviews of the tree form (f2b4679) - one finding HIGH, six MEDIUM, nine LOW, and a
pass of 24 mutants of its code of which 13 survived the tree tests.

- **The root is always evaluated - now true** (`engine/src/mcts.cpp`). A re-root onto
  a child the tree form inherited (`advance`, `advance_to`, `advance_moves`; match play
  carries the tree, C12: self-play's netplay "match" mode, `selfplay.cpp`'s match play,
  the C API's and Python's advance) left the new root inherited: the network never
  evaluated it, its priors stayed the ones inherited from the old turn start, its value
  the old parent's (a stuck one backed up that value), the C API reported
  `root_evaluated` 0 and a NaN root WDL - against mcts.hpp, the README, issue 161(f),
  the register row and the changelog, which all said the root is always evaluated.
  Now `select_once` makes such a root a leaf to the first simulation after the re-root
  (the batch's other simulations wait for it, as for a fresh root), and the evaluation
  re-evaluates it (`reevaluate`): its edges keep their children, visits, values and
  flags (C12, "carry statistics"), each takes the evaluation's logit for its move
  (matched by coordinates), the submit edge the submit logit; its value, uncertainty,
  error and moves-left become the evaluation's; its priors are recomputed as a fresh
  root's (the root temperature, no noise, I15's side table refilled), the edges
  ordered by them, every visited one still instantiated, the effort label found again;
  it is no longer inherited. Its children keep the priors they inherited. The
  invariant no longer excuses a visit ending at a stuck inherited root (none can be).
  Why this rather than correcting the documents: an inherited carried root would run a
  whole decision's search on priors the network never gave for that state, with no
  value of its own; the change costs one evaluation per re-root onto an inherited node,
  touches no other arm (no other node is ever inherited) and leaves the solver, D3's
  values, A6's count and I15's side table sound. The ply split and the whole-turn arm
  are unchanged bit for bit: 60 searches of each (five variants' starts and random-play
  positions, six configurations - MCGS, the solver with exact legality, E2/E3/root
  noise, the pseudo-legal seam), H15's digest before and after a re-root, the
  evaluations and `memory_bytes()`, all equal before and after the fix; under the tree
  form the same 60 searches are unchanged before the re-root, and 40 differ after it -
  exactly the 40 whose chosen child was an inherited node.
- **`tree_inherit_value` off is refused where the uncertainty signal is read**
  (`validate`): with E2, E3 or E5's terms on (`uncertainty_read`), an inherited node's
  signal would be read as 0 - no variance - as 6m refuses for a missing evaluation
  signal.
- **Tested** (`engine/tests/test_mcts_tree.cpp`; `python/tests/engine/
  test_search_api.py`): a carried inherited root evaluated once, by the first
  simulation, its edges' statistics kept, its value, logits and priors a fresh root's,
  never asked about again (C++, gates none and natural, one playout and a batch); the
  same through the Python step API with the network (root WDL defined, children's
  visits kept); the inherited value on (a node's eval_* its parent's, below turn starts
  too) and off (0), the searches differing and keeping the invariants, and the refusal
  for each reading term; the gates under the Mandatory reading of the boards still to
  move, on positions with an optional board where the readings differ (the between gate
  - a jump onto an inactive timeline's board - and the check gate); an inherited node's
  legality classification against the expansion's (every move unknown with no budget,
  illegal moves kept under the pseudo-legal seam, dead ends found without the
  network); promotions matched by the inherited logits (a four-piece promotion set);
  a move the parent did not have taking its lowest logit - reachable after all: a move
  proved illegal as the parent's next one (no legal completion begins with it) is legal
  once another board is played, 21 such edges on the promotion position (issue 161(b)
  corrected); a root child's `evaluated` false for an inherited child; the inherited submit logit (Python: the tree form's
  case of `test_the_submit_logit_reaches_the_submit_child`). Reviewer B's mutants
  re-run: every one tied to a finding is caught, but for one that cannot be (below).
- **The counts committed** (`engine/tools/t5d_tree_count.cpp`, `t5d-tree-count`; not
  one of the truth's sources): the uncommitted scratch harness the tree form's figures
  came from, as a tool; the figures, which it reproduces value for value, with their
  conditions in `docs/MEASUREMENTS.md`.
- **t5d-play**: `--tree-gates` with `--split ply` or `turn` is refused (it was read by
  nothing); the Engine header names the gates (`between,check`, not `6`).
- **Comments and documents**: mcts.hpp - the carried root, `memory_bytes()` not compared
  by §15.2's row, `Limits::bytes`' overshoot under the tree form, everything
  `tree_inherit_value` feeds; the C API header; engine/README.md.
- **Not changed**: a passed board keeping its en-passant square cannot change a
  gate (the square belongs to a double step of the side now to move on the copy, and an
  en-passant capture needs the other side's pawn), so that mutant is equivalent and no
  test can catch it; the between gate under the Mandatory reading reads the
  destination only (issue 157 says so; for the user).
- `docs/BUILD_LOG.md`: register row D2; issues 157, 159, 160, 161(b), (e), (f);
  proposed edit 129.
- Spec: §8.2 "Evaluating at turn boundaries" (the tree form; the root always
  evaluated), §8.4/C12 (the carried tree), 6m/I10 (the refusal), §15.2's tree-form row,
  §15.3 (proposed edit 130's row: the counts recorded in docs/MEASUREMENTS.md, not
  written back); register D2, C12, I10 - spec issues 157, 159, 160, 161; proposed edit
  129 amended. No new spec issue.

### 2026-10-23 - Engine D2: the tree form - turn-start priors inherited inside the turn; re-evaluation at interactions and complete turns

D2's tree form (§8.2, "Evaluating at turn boundaries"), the first of the two arms
CHECKPOINT 8's build order names, built behind `SearchParams::split` beside the ply
split (still the default) and the whole-turn arm (left as it is until the flat form
replaces it).

- **The arm** (`engine/include/t5d/mcts.hpp`, `engine/src/mcts.cpp`; `TurnSplit::Tree`):
  the ply split's tree, nodes, edges, submit and turn structure. A node during an
  action whose incoming move fires no enabled interaction gate is expanded without the
  network - its children exactly a leaf's (the same legality classification and
  submit), their priors its parent's logits matched move by move (a J6 branch takes the
  non-branching jump's; an unmatched move the parent's lowest; submit's logit carried
  down) - and the simulation runs on through it. The network is called at the root, at
  turn starts and at gated mid-turn nodes (ordinary leaves). [Corrected by the review
  fixes above: a root carried onto an inherited node was not evaluated; it is now.] An inherited node backs up
  nothing of its own: the leaf's value passes through it (its visits its children's);
  its FPU base and uncertainty signal are its parent's evaluation's
  (`tree_inherit_value`) [untested until the review fixes; off with a term that reads the
  signal now refused]. Each edge keeps the logit its prior came from (`Edge::logit`;
  `EdgeView::logit`), each node its submit logit;
  `NodeView::inherited`; `SearchStats::inherited_nodes`, `gate_fired`,
  `seconds_gates`; the digest folds the inherited flag in (zero for every other arm, so
  no digest changes). The ply split's code paths are shared, not copied
  (`set_priors`, `ply()`), so its searches are bit for bit what they were [in their
  digests; `memory_bytes()` grew - a node carries the submit logit and the inherited
  flag under every split: +4,096 bytes on most 200-playout ply searches (up to +16,384),
  +1,024 on 60-playout whole-turn ones].
- **The interaction gates**, §8.2's four, each reading a parameter with a placeholder
  default (`tree_gates`, `tree_still_to_move`, `tree_capture`, `tree_check`,
  `tree_present`): a non-branching jump; a capture of a piece with a pseudo-legal move
  onto a passed board still to move (R1's passing); the acting player's "in check" with
  the boards still to move passed (R1's construction) differing across the move; any
  move of the present. `kGateAlways` is the test seam for §15.2's row.
- **Refused under MCGS** (`graph`): `std::invalid_argument` naming spec issue 301
  (`T5D_ERR_ARGUMENT`, `EngineArgumentError`) - under graph search an inherited node's
  priors and its being evaluated would be the path's.
- **C API and Python, one commit** (`capi_search.h`, `capi_search.cpp`, append-only):
  `T5D_SPLIT_TREE`, `T5D_TREE_GATE_*`, `T5D_STILL_TO_MOVE_*`, `T5D_CAPTURE_*`,
  `T5D_CHECK_*`, `T5D_PRESENT_*`; `t5d_search_params` gains `tree_gates`,
  `tree_still_to_move`, `tree_capture`, `tree_check`, `tree_inherit_value`,
  `tree_present` (before `expand`), `t5d_search_summary` `inherited_nodes` and
  `gate_between`/`_capture`/`_check`/`_present`/`_always`; a tree-form leaf reports its
  submit flags as the ply split's. `python/t5d/engine/capi.py` and `search.py` mirror
  them (checked against `t5d_search_layout`), `netplay.py` plays `split="tree"`;
  `t5d-play --split tree --tree-gates`.
- **Tests**: `engine/tests/test_mcts_tree.cpp` (new) - §15.2's row bit for bit
  (digest, statistics, root children, across a re-root) on every named variant,
  random-play roots with many timelines and deep travel, roots during an action and
  past the shift, two evaluators at realistic scale, every tree configuration, budgeted
  and exact legality and the pseudo-legal seam, batched and synchronous; no gate - the
  network sees only the root and turn starts; one gate - exactly the mid-turn nodes whose
  move fires it, by predicates recomputed from the rules engine; the capture reading on
  constructed positions; the inherited logits (a J6 twin) and their softmax [the
  inherited submit logit, the Mandatory reading and an inherited node's legality were
  untested - the review fixes test them]; §15.4's
  turn structure and dead ends; the refusal; more turn starts per evaluation. Two
  tree-form configurations join `test_mcts.cpp`'s invariant suite (one test's sibling
  choice generalised: an inherited sibling holds no evaluation to keep); the C step API
  against the C++ search by digest, the C API's refusals, self-play end to end and in
  match play, the allocation-failure test, the memory estimate (x1.04) and t5d-play's
  records through the gate adapter under the tree form. Python: the step loop is the
  engine's run under the tree form (uniform and random evaluators), §15.2's row through
  the step API with the network at random weights (two variants and a root past the
  shift, three legality settings, batched and synchronous), the network asked only at
  turn starts without gates, the refusal, and netplay games under the tree form replayed
  by the gate adapter.
- **Measured, indicative** (counts; seven t0database turn starts, 1-4 mandatory boards,
  30-60 boards; the built-in random evaluator, synchronous, no cache, a scratch harness
  [committed by the review fixes as `t5d-tree-count`, the figures in
  docs/MEASUREMENTS.md]):
  at 400 evaluations the turn starts evaluated - ply split 74-122, tree form 111-142, with
  the present gate read as rewinds only 173-290, with no gate 400. The literal present
  gate fires at every action's shift (issue 160).
- `docs/BUILD_LOG.md`: register rows D2 and C11; issue 301's Built note; spec issues
  157-162; proposed edits 129-130. `engine/README.md`: stage E6a's arms, the gates'
  table, t5d-play's options, the choices list. The Lazy SMP timing test runs before the
  merge (the main loop).
- Spec: §8.2 "Evaluating at turn boundaries" (tree form), §15.2's D2 tree-form row,
  §15.4 (turn structure), §8.5/C11 and §14.4/C21 (the refusal), §13.1 6x, §15.3 (a new
  row); register D2, C11, D8, A6 - spec issues 157-162 and 301, proposed edits 129-130.

### 2026-10-23 - N-A6 review fixes: the optional count in the digest pinned; the batch fallback documented

- One reviewer, no high or medium findings (20 of 21 mutants caught; the survivor - a
  digest without the optional count - is now caught): `tests/net/test_turn_state.py`'s
  equal-boards cases vary the optional count alone; `Batch.turn_counts`' all-zero fallback
  documented as invalid before the shift and never reached through `collate`; issue 302's
  Built note.
- Spec: §2.3, §15.2's A6 and purity rows; register A6.

### 2026-10-23 - Network A6: the turn-state inputs embedded into the global registers

Register A6's network side (the engine side, ABI round 1, merged at e8de15e): the three exact
turn-state counts - mandatory boards still to move, optional boards still playable, moves
already made in the action - reach the network.

- **The state** (`net/state.py`). `PositionState` gains `mandatory_remaining`,
  `optional_playable` and `moves_made`; `turn_state` gives the counts in effect. A count given
  as None is derived from the boards under the default rules (`derived_turn_state`:
  mandatory = the active heads in the present with the acting player to move, optional = its
  other heads, moves made 0 before the shift and 1 past it) and re-derived whenever the state
  is rebuilt, so it follows the acting player; a given count is kept - by `permute_boards`,
  `translate`, `relayout` and `mirror` (the counts are the acting player's, which A7's mirror
  keeps). Validated: non-negative integers; submitting possible only with nothing owed (T5);
  mandatory and optional within the playable heads (L5). `collate` puts them in
  `Batch.turn_state` (`(B, 3)`; `turn_counts()` is 0 for a batch made without them).
  `TurnState` is exported.
- **The converter** (`engine/convert.py`) copies the expansion's counts and, unless
  `strict=False`, refuses (`PresentMismatch`) counts that break what the expansion defines them
  by: negative; mandatory + optional not the boards it marks playable; more mandatory boards
  than playable boards of active timelines; nothing owed and submitting not possible, or the
  reverse. Which playable boards are owed is the ruleset's, which the expansion does not name,
  so the split is the engine's.
- **The embedding** (`net/embed.py`, `net/model.py`, `net/config.py`): each count as
  `(n / 8, log1p n, [n = 0])`, the nine features through Linear(9, 32), GELU, Linear(32, d),
  one vector added to every global register beside D8's acting colour; with no global
  registers, issue 225's placeholder (the board tokens, the live boards' alone under the
  block-causal scope). The form is a placeholder of a committed-class silence, fixed in code
  (spec issue 230; `NetConfig.turn_state_hidden = 32`, `turn_state_scale = 8.0`). Its 8,768
  parameters draw nothing while made and are drawn from a generator of their own
  (`model.a6_seed`): no other parameter changed. F1 22,973,907 (was 22,965,139), F2's
  L24-ff12 46,903,315 (`tests/net/test_params.py`; the totals quoted in `shaw.py`, `config.py`
  and `test_displacement.py` updated).
- **The caches** (`net/cache.py`). `state_digest` - an epoch's anchor, and the Python
  evaluator's epoch reuse - carries the three counts (a refresh's dense pass lets every board
  read the global registers); frozen K/V never depend on them. A refresh's other-colour view
  (`viewed_by`) takes the counts its boards give that colour (the anchor's are its acting
  player's; issue 230 with 229). `python/t5d/engine/evaluator.py` and `search.py` are
  unchanged: leaves convert through `to_position`, which now copies the counts.
- **The fixtures** (`net/synthetic.py`, `net/tree.py`) produce turn starts with the boards'
  own counts, as documented. The tree fixture does not count moves per path: its moves are no
  actions (a physical move is made by whoever is to move on that timeline), and a per-path
  count split its two move orders of one multiverse - which `test_cache.py`'s transposition
  tests read as one state - into two (tried and reverted). Mid-turn states are its states with
  the counts given.
- **Tests.** `tests/engine/test_turn_state_engine.py` (section 15.2's A6 row on the network
  side): every state of eight seeded random plays under both Q4 readings - 956 states: turn
  starts, mid-action, past the shift with and without moves; jumps between two present boards,
  rewinding branches, T12 rises - converts to the expansion's counts, moves made the handle's
  count, mandatory and optional a recount from the rulebook's definitions; a mid-turn state and
  its own 5DFEN (a turn start with equal boards and acting player) differ in moves made alone,
  and in digest and output, full and cached; inconsistent counts refused.
  `tests/net/test_turn_state.py`: derivation and validation; each count moves exactly the
  tokens the global inputs enter and nothing else; every output moves with each count under
  every register arm and both scopes; equal boards with different counts are never confused by
  the store, an epoch or a refreshed epoch's anchor; cached = full pass along the tree
  fixture's paths with each node also given mid-turn counts; the parameters' draws.
  The A7 mirror tests compare the counts (a mid-turn construction added; the engine's 5DFEN
  mirror's own counts); `test_convert.py`'s relabelled turn start relabels its counts too, so
  that its moves are what is refused; `test_embed.py`'s global registers are equal given the
  acting player and the counts.
- `docs/BUILD_LOG.md`: register rows A6 (now built, tested) and F1; spec issue 225's A6
  sentence answered and issue 230 new; proposed edit 156's figures moved, edit 171 new.
- Spec: §2.3 (turn-state inputs), §7.9 (the refresh's other-colour view), §8.5 and §15.2's A6
  and purity rows, §9.2's Params row; register A6, F1, F2, D8, C3. Proposed edits 171 (the
  embedding's form, the other colour's counts) and 156 (the parameter count, M4).

### 2026-10-23 - Build log: issue 156 (a mislabel in CHECKPOINT 8's turn-width figures) and issues 148-155 wait on the user

- `docs/BUILD_LOG.md`, "Waiting on the user": spec issue 156 - the turn-width shares
  the main loop reported as "more than one playable board" (M22's merge message) are
  the shares with more than one mandatory board, and CHECKPOINT 8's §8.2 paragraph and
  register M22 carry the mislabel; proposed edits 127-128 correct them. Also engine ABI
  round 2's readings, issues 148-155.
- Spec: §8.2, register M22 (edits 127-128).

### 2026-10-23 - M22: the mandatory-board shares, optional moves played and the compute share of multi-board turns (the CHECKPOINT 8 ruling)

The user's ruling on spec issue 142 asked M22 to report, per board-count range, the
share of turns with more than one and with three or more mandatory boards, the optional
moves actually played, and the compute share of multi-board turns (moves x per-move
cost).

- **The summary names the four** (`engine/tools/turn_width.hpp`, `turn_width.cpp`,
  `t5d_measure.cpp`). `--turn-summary`'s lines gain `moves_total`,
  `optional_moves_total` and `optional_moves_share_of_moves` (the optional moves'
  share of turns and mean were there), and the compute shares - of the turns with more
  than one mandatory board, three or more, and more than one playable board:
  `compute_share_*` and `compute_share_*_constant` - with `dense_tflops`; the head line
  states the `cost_model`. Every value the output gave before is unchanged.
- **The cost of a move**: one evaluation - the ply split evaluates every node inside a
  turn (§8.2) - of the dense v1 network (§7) at the turn's board count N, by §9.1-9.2's
  analytic model as N3's `python/t5d/net/cost.py` computes it: 4 T^2 d L + 40 T d^2 L,
  T = 66N + 16, d = 256, L = 16 (`tools::dense_eval_flops`; 4.78 TFLOPs at 240 boards,
  §9.2's figure). A constant cost per move is reported beside it as the sensitivity
  reading. Arithmetic only: the measurement stays rules-engine-only.
- **Tests** (`engine/tests/test_turnwidth.cpp`): the dense cost worked by hand in
  integers at 0, 1, 10, 100 and 240 boards; the compute shares and their sums on six
  constructed widths with known answers; the optional moves' share of the moves; and
  the new fields end to end on the hand-written records (583 engine tests pass).
- **Measured on t0database** (692 games; `docs/MEASUREMENTS.md`, "Turn width (M22)",
  "The CHECKPOINT 8 figures"; recomputed independently from the per-turn lines, no
  difference): < 50 / 50-99 / 100-199 boards - more than one mandatory board 5.8% /
  27.1% / 64.0% of turns, three or more 0.7% / 6.7% / 16.0%; an optional move in 1.2% /
  2.5% / 1.1% of turns (0.012 / 0.025 / 0.011 per turn; 1.1% / 2.0% / 0.6% of all
  moves); the compute share of the turns with more than one mandatory board 18.0% /
  43.4% / 81.9% (33.3% pooled; 9.9% / 41.7% / 79.7% and 14.7% at a constant cost per
  move); of those with three or more 3.3% / 14.9% / 33.4% (10.3% pooled). Ten-board
  buckets under 100 too.
- **Found**: the 5.8% / 27.1% / 64.0% that the ruling, §8.2's CHECKPOINT 8 paragraph
  and register M22 call playable shares - "these bound the mandatory share from above"
  - are the mandatory shares; the playable ones are 15.1% / 65.2% / 99.4%. Part L's
  "at most one playable board at 94.2%" is likewise one mandatory board (one playable
  board: 84.9%). Spec issue 156, for the user.
- `docs/BUILD_LOG.md`: step 5's M22 entry, register row M22, issue 142's Built note,
  issue 156, proposed edits 127 and 128. `engine/README.md`: the new fields.

Spec: §8.2 ("Measure first" and the "Measured (CHECKPOINT 8, human games)" paragraph),
§15.1's Phase 1 row, §7, §9.1-9.2 (the cost model, read, not changed); register M22,
part L. Proposed edit 127 (M4: the figures written back into §8.2's "Still to report"
sentence and register M22, with issue 156's labels corrected) and 128 (part L's outcome
note; issue 156).

### 2026-10-22 - ABI round 2 review fixes: an MCGS residual fires on a draw lag, a missing uncertainty signal refused where it is read, the side table and the noise's edge set tested

Two reviews of engine ABI round 2 (7c650d3, 7828d0d, 9d919c2) found four medium faults
and eight low ones; reviewer B's twelve mutants left ten alive. Every medium one is
fixed, and every low one but part of B-L7 (below).

- **An MCGS residual fires on a draw lag** (A-M1, B-M2; register D3, C11; issue 148).
  `engine/src/mcts.cpp`'s `needs_residual` compared only the edge's Q with its shared
  child's, so with Q in line a draw probability that lagged was never corrected - the
  2a entry's "an MCGS residual corrects D as it corrects Q" was untrue (reviewer A's
  probe: Q 0 at every leaf, D 0.1 or 0.9 by node key - edges descended with their D 0.40
  from the child's). Now a lag beyond `residual_epsilon` in Q **or** D fires the
  residual, which corrects both (one epsilon for the two: a reading, issue 148).
- **A missing uncertainty signal is refused where it is read** (A-M2; register I10, 6m;
  issue 149). A node's variance was 0 when its evaluation lacked the signal
  `uncertainty_source` selects - which E2 reads as its cPUCT scale's floor (exploration
  halved) and its backup's largest weight; so a 6m arm on the error head, with the Python
  evaluator's `error` NaN until the network lane names `NetOutput.value_error`, would have
  run silently as another search. Chosen: **refuse**, not "absence is no prior" (weight 0
  in the blend, a neutral backup weight) - that too runs an arm other than the one asked
  for, mixed node by node, and an ablation should fail loudly. While E2's scale or weight,
  E3, or E5's term under the LCB rule reads the signal, an evaluation without it is
  refused as a malformed one (`std::invalid_argument`, `T5D_ERR_ARGUMENT`; the batch
  released) - a shared cache's entry without it too. Read by nothing, it is 0 and
  counted: `SearchStats::uncertainty_missing`, appended to `t5d_search_summary` (Python's
  `CSearchSummary` and `SearchInfo` mirror it).
- **The side table through the C API, tested** (B-M1). Four wrong copies in
  `capi_search.cpp` (the noise-free prior as the prior, the value variance as d, E3's
  factor as 1, the root cPUCT as the parameter) passed every test: the C step loop now
  runs with root noise, a root temperature of 1.3, E2 and E3 on, and compares each child's
  `prior_noise_free`, `value_variance`, `exploration_scale`, `child_key` and the summary's
  `root_cpuct`, `root_best_since` with the C++ search field for field, under both splits,
  tree and graph (and checks they are not vacuous: priors noised, factors above 1,
  variances not the draw). Python's noise-free-prior test, which held with the two priors
  equal, now checks the tempered softmax of known logits.
- **The noise's edge set, tested** (B-M3; register E1). Counting a pruned edge in the
  split survived every test (no noise test had a pruned edge; the spread's 15% tolerance
  cannot tell n from n + 1). `RootChild::noise_alpha` (C++ only, not in the C side table:
  root-only, `Edge` does not grow) records the concentration each root child's variate was
  drawn with; a test at a mid-action root with an illegal submit (the pseudo-legal seam)
  checks every child's alpha against `dirichlet_alphas` over the open edges, unshaped and
  shaped, the open alphas summing to the total.
- Low findings. A-L1, B-L4: a proven win's root child (q, d) = (1, 0), a proven draw's
  (0, 1), a solved drawn node's visits backing up D 1, a stuck node's - an Unknown verdict,
  or every child a dead end under the seam - its own D (the three zeroing mutants that survived now caught).
  A-L2, B-L6: `t5d_search_deliver_evaluations` checks `wdl_kind` and `value_kind` whether
  or not a categorical is given. A-L3: `root_q` and `root_d` documented as the backed-up
  averages, a solved root's too (`root_lo`, `root_hi` carry the proof) - not changed to
  the exact values, which would change `root_q`'s meaning for its callers. B-L5: E3's
  factor recomputed per child from its own node and edge. B-L7: the legal-moves
  equivalence's condition stated (`capi.h`, the README, `Game.expand`'s docstring: a
  board-move search's leaves with `legal_moves_only`, at its budget; a whole-turn
  search's are pseudo-legal) and tested in Python at budget (0, 0) where moves stay
  unknown; `Game.expand`'s default budget stays exact - "the legal moves" for a caller
  that is not rebuilding a search's input, which passes the search's budget explicitly
  (the docstring says so). B-L8: the error head's field name is
  `evaluator.ERROR_HEAD_FIELD`, pinned by a test (no other `NetOutput` field may name an
  error head; the field read when present); `t5d_search_params.dirichlet_alpha` is
  commented as unused while `dirichlet_concentration` is above 0 (the default).
- Search results change only where a fix requires: an MCGS (graph) search under the
  board-move split whose edges' D lags a shared child's while Q does not now backs up the
  residual - of 48 configurations (Standard and Two Timelines; tree and graph; both
  splits; E2/E3 off and on; the uniform, random and material evaluators; root noise; 1500
  playouts) one digest changed, the graph board-move random-evaluator search on Two
  Timelines (30 residuals -> 32); every tree search, every whole-turn search and the
  constant-draw evaluators' graph searches are bit-identical. A search with a reader on
  and an evaluator lacking the chosen signal now stops with an error instead of running
  on variance 0.
- Tests. `test_mcts.cpp`: `mcts_d3_an_mcgs_residual_corrects_a_draw_lag` (stepped one
  simulation at a time after a 2500-playout warm-up: every descended edge within epsilon
  of its child's D, every residual exact or as near as the clamp allows - 221 residuals,
  all by D alone, 476 descents checked), `mcts_6m_a_signal_that_is_read_must_be_given`,
  `mcts_e1_the_noise_splits_its_concentration_over_the_open_edges_only`,
  `mcts_d3_proven_stuck_and_dead_end_visits_back_up_their_draw`; E3's recomputation in
  `mcts_i15_the_root_side_table`. `test_capi_search.cpp`: the step loop's side table; the
  kinds; the refusal and the count through the C API; the layout. Python: the refusal and
  the count, the tempered noise-free prior, the rebuilt input's condition, the error
  head's name. Reviewer B's mutants re-run on the fixed tree, with three more undoing
  these fixes: all fifteen caught (the review's scratch tree).
  Checks: engine 581 of 581 (`--werror`); Python, the whole suite, 1086 passed, 1 skipped
  (`T5D_REQUIRE_ENGINE=1`; 26 min on a shared machine); `tests/engine` 191 passed, 1 skipped; ruff check and format
  clean; `harness verify` ok; `gate --tier commit`: the oracle layer ok, the engine
  differential "SMOKE TEST - PASSED" (41 records compared with the last green run, 0
  declined), the proof channel DISABLED.
- Corrected in this round's own entries (below, in brackets): 2a's residual claim and its
  test's, its "a signal not given is 0"; 2b's open-edge split as tested; 2c's "the side
  table through the C API" and its Python noise-free-prior test.
- `engine/README.md`: the residual and the 6m paragraphs, the C API's summary, the
  noise's alphas, the legal-moves condition, the tests, the readings. `docs/BUILD_LOG.md`:
  issues 148 and 149 corrected, register D3 and I10, proposed edit 122 amended.
- Spec: §8.3 / register D3, C11 - the residual corrects D with Q, a lag in either firing
  it (issue 148; the spec's residual text, §8.5's "residual-correction backup", says
  nothing narrower, so no edit); §13.1 6m, §8.4 / register I10, E2 - a missing signal
  refused while read (issue 149; proposed edit 122 amended); §12.8 / register I15 - the
  rebuilt input's condition (issue 155; edit 126 unchanged). No truth source changed.

### 2026-10-22 - Engine ABI round 2c: I15's record fields - the root side table, the root's key, cPUCT and effort label, `t5d_game_expand`'s legal moves; forced playouts not built

- Register I15 (Committed, process; §12.8: "raw root outputs ..., root and best/played
  children's values, visits with I2's forced and steered flags, prior-to-target divergence,
  proofs by reference, and conditions") and I2 (Committed; §12.2, §12.5: search-value and
  policy targets from the pruned visits) - the engine side, the plan critic's missing
  item 12, in round 2's last commit. `engine/include/t5d/mcts.hpp`, `src/mcts.cpp`: each
  `RootChild` gains **the root side table** - `child_key` (the child node's key; 0 for no
  node), `prior_noise_free` (the prior before the root noise: after the temperatures, the
  unknown-move scale and the submit share; held in a root-only vector filled by the root's
  own expansion and sorted with its edges, so `Edge` does not grow; renormalised with
  `prior` when a sibling is pruned; a carried root's is its prior), `value_variance` (of
  its backed-up values), `exploration_scale` (E3's factor on its exploration term, 1 with
  E3 off), `boost` and `steered_visits` (E10's asset-map boosts and Hunt's suggestions: 0,
  none wired until step 2's E10), `forced_visits` (0). `Search::root_cpuct()` (cpuct times
  E2's scale at the root) and `Search::root_best_since()`, D4's effort label: the root's
  visits when its most visited child last changed (strictly more visits to take over; a
  pruned incumbent found again; from the re-root for a carried root) - "visits until the
  decision stabilises" (§8.6; issue 154).
- **Forced playouts: not built** (the critic's contradiction 4; issue 153). §12.5 adopts
  KataGo's forced-playout *pruning* at target construction - the training loop's (M21) - and
  no document asks the search to force playouts; the record carries what that pruning and
  a replica of the root's PUCT need (each child's visits, both priors, q, value variance,
  E3's factor; the root's effective cPUCT). `forced_visits` exists, appended once, and stays
  0 unless forcing is ever built.
- `t5d_game_expand`'s **legal-moves option** (issue 155; the training brief's conflict 3):
  `engine/include/t5d/capi.h`, `src/capi.cpp` - `t5d_expand_options` appends
  `legal_moves`, `legality_checks`, `legality_nodes`; with `legal_moves` 1 the game's
  expansion is a search leaf's under the same budget (the legal and unknown moves, in the
  same order: the classification is deterministic), so a record's network input can be
  rebuilt. `t5d_search_params.expand` grows with it (the embedded struct stays last); a
  search refuses `expand.legal_moves` (its `legal_moves_only` and `legality_*` decide its
  leaves).
- C API (`capi_search.h`, `capi_search.cpp`): `t5d_root_child` appends the side table's
  seven fields; `t5d_search_summary` appends `root_key` (the memo's key for the root's raw
  outputs, which C14's shared cache hides from Python on a hit), `root_cpuct` and
  `root_best_since`; `t5d_search_layout` and `t5d_layout` report them. Python mirror in
  the same commit: `capi.py` (`ExpandOptions` 12 -> 32 bytes, `CSearchSummary`,
  `CRootChild`), `search.py` (`SearchInfo`, `RootChild`), `game.py` (`Game.expand` and
  `Game.position` take `legal_moves`, `legality_checks`, `legality_nodes`); the docstrings
  of `game.py` and `convert.py` that said the filter was not in the C API corrected.
  `mcts.hpp`'s and the README's (f): the policy target is built from the visits by I2's
  pruning at target construction, not taken raw.
- Search results do not change: the side table, the summary fields and the effort label
  are read, never steer (the root's edges are sorted by a stable sort of their indices,
  which orders them exactly as the stable sort of the edges did).
- Tests. `test_mcts.cpp`: `mcts_i15_the_root_side_table` (with root noise and a root
  temperature of 1.25: the noise-free prior is the tempered softmax, `prior - (1 - eps)
  prior_noise_free` non-negative and summing to eps; the noise-free prior equal to the
  prior without noise and at a carried root; `child_key` the node's; `value_variance` the
  edge's; E3's factor within [1, cap] and above 1 somewhere; the steering and forcing
  counts 0; both priors summing to 1 after verification removes illegal children);
  `mcts_i15_root_cpuct_and_the_effort_label` (cpuct with E2 off, its scaled value with
  E2 on; the effort label replayed one synchronous playout at a time over 400 playouts,
  and from a re-root). `test_capi_search.cpp`:
  `capi_search_game_expand_gives_a_leafs_legal_moves_i15` (byte for byte at the Standard
  start and with a king in check, where the pseudo-legal list has one move more; a bad
  flag refused; `root_key` the root leaf's key; the side table through the C API - [corrected
  by the review fixes: only with no noise and E2/E3 off, where four wrong copies passed;
  the C step loop now compares it field for field with noise, a root temperature, E2 and
  E3 on] - a search refusing `expand.legal_moves`); the layouts. Python (`tests/engine`): the record's
  input rebuilt by `Game.expand(legal_moves=True)` (both positions), `root_key`,
  `root_cpuct`, the effort label, the side table, the refusal; the noise-free prior beside
  the noised one [corrected by the review fixes: that test held with `prior_noise_free`
  equal to `prior`; it now checks the tempered softmax of known logits]. Checks: engine 577 of 577 (`--werror`); Python, the whole suite, 1083 passed, 1 skipped
  (`T5D_REQUIRE_ENGINE=1`; 19 min on a shared machine);
  ruff check and format clean; `harness verify` ok; `gate --tier commit`: the oracle layer ok, the
  engine differential "SMOKE TEST - PASSED" (41 records compared with the last green run, 0
  declined), the proof channel DISABLED.
- `engine/README.md`: "The records' fields", (f), the game C API's `legal_moves`, the
  readings (issues 153-155), the tests. `docs/BUILD_LOG.md`: register I2, I15 (their
  engine sides built), D4 (the effort label's source); spec issues 153-155; proposed edits
  124-126.
- Spec: §12.8's record list and §12.2/§12.5 (register I15, I2) - forced playouts not built,
  the pruning target-side, what the record carries for it (edit 124, issue 153); §8.6's and
  §12.4's effort label (register D4; edit 125, issue 154); §12.8's conditions - the
  legality budget, with which the input is rebuilt (edit 126, issue 155). No truth source
  changed (`capi.cpp` and `capi.h` are not among them).

### 2026-10-22 - Engine ABI round 2b: E1 and I19's exploration - the root noise's total concentration split across the open edges, shaping behind a flag, a root policy temperature

- Register E1 (Committed, class R; §8.4): "root Dirichlet with its total concentration
  split across the legal moves (KataGo; shaping a prototype) and a root policy temperature
  (I19); constants are named placeholders"; I19 (Committed placeholders; §8.4, §12.8).
  `engine/include/t5d/mcts.hpp`, `src/mcts.cpp`: `SearchParams::dirichlet_concentration`,
  the root noise's **total** (placeholder `kDirichletConcentration`, KataGo's 0.03 x 361 =
  10.83), split across the root's open edges by `dirichlet_alphas` - the moves the
  expansion kept (legal or unknown), a legal submit, or under whole turns the listed
  actions (issue 150) - so the noise's overall strength holds as the move count changes.
  [Corrected by the review fixes: which edges the split counts was untested - every noise
  test's root had no pruned edge; now checked alpha by alpha at a root with an illegal
  submit.]
  `dirichlet_alpha` keeps its per-child meaning, as the arm at `dirichlet_concentration`
  0, its draws exactly those before this commit. KataGo-style **shaping**, a prototype
  behind `dirichlet_shaping` (off): `dirichlet_shaping_share` (0.5, the paraphrase's
  "half") spread by how far each edge's log-prior, capped at `dirichlet_shaping_cap`
  (0.01), lies above the mean, the rest uniform - a reading of §8.4's paraphrase, from the
  spec's words and KataGo's published description, no code (issue 151, H29). The **root
  policy temperature** `root_policy_temperature` (1, off, by default): the root's own
  expansion takes its softmax at `policy_temperature x root_policy_temperature`, before
  the noise; k0 reads the tempered prior; a carried root is never re-tempered (issue
  152). Validation: the concentration finite and >= 0, the share in [0, 1], the cap in
  (0, 1], the temperature finite and > 0.
- I19's schedules are the training generator's (the plan critic's contradiction 9): the
  move-temperature schedule (Lc0's T60, 0.9 -> 0.45, cut at move 60) and the root
  temperature's self-play value and decay (KataGo 1.25 -> 1.1) are set per search by M21's
  generator, in Python; the engine exposes `root_policy_temperature` beside the move
  temperature `choose_by_visits` already takes, and no more. `selfplay.cpp`'s and
  `netplay.py`'s step schedules and a root temperature of 1 stand until then.
- Search results change only where the new semantics require: a search with root noise
  on now splits the total (self-play's default; `t5d-play`'s, `selfplay.cpp`'s and
  `netplay.py`'s noisy searches draw other noise); a search without noise is unchanged,
  and so is one with `dirichlet_concentration` 0.
- C API (`capi_search.h`, `capi_search.cpp`): `t5d_search_params` gains
  `dirichlet_concentration`, `dirichlet_shaping`, `dirichlet_shaping_share`,
  `dirichlet_shaping_cap`, `root_policy_temperature` in front of the embedded `expand`
  (reported by `t5d_search_layout`; `dirichlet_shaping` a 0/1 flag). Python mirror
  (`capi.py`'s `CSearchParams`) in the same commit; the layout check passes.
- Tests. `test_mcts.cpp`: `mcts_e1_dirichlet_alphas_split_the_total_concentration` (the
  total, an even split, exactly `dirichlet_alpha` per edge in the old arm; shaping: a
  uniform prior uniform, ranked edges never fewer, never below the uniform half's share,
  capped priors alike); `mcts_e1_the_noise_keeps_its_strength_as_the_move_count_changes`
  (600 seeds each: the noise's relative spread `n^2 Var / (n - 1)` 0.0852 at 20 moves and
  0.0856 at 62, against `1/(C + 1)` = 0.0845); `mcts_i19_the_root_policy_temperature_
  flattens_the_root_only_before_the_noise` (the root's priors the softmax at 1.25, every
  other node's at 1; with the noise, `P' - (1 - eps) P_T` non-negative and summing to eps;
  temperature 1 the default's search bit for bit); the new parameters' validation.
  `test_capi_search.cpp`: the defaults through the C API, the fields' layout in front of
  `expand`, a bad flag and a zero temperature refused. Python: the step loop equal to the
  engine's own run by digest with shaped and tempered noise and with the per-child arm
  (the uniform and the random evaluator); the defaults and the refusal by name.
  Checks: engine 574 of 574 (`--werror`); Python `tests/engine` 185 passed, 1 skipped
  (`T5D_REQUIRE_ENGINE=1`); ruff check and format clean; `harness verify` ok; `gate
  --tier commit`: the oracle layer ok, the engine differential "SMOKE TEST - PASSED" (41
  records compared with the last green run, 0 declined), the proof channel DISABLED.
- `engine/README.md`: the root's exploration under "Selection", the C API's new fields,
  self-play's schedules (the generator's), the tests, the readings (issues 150-152).
  `docs/BUILD_LOG.md`: register E1 (built; the mandatory-count widening option, a "may",
  not built), I19 (the engine side built; the schedules the generator's); spec issues
  150-152; proposed edit 123.
- Spec: §8.4's noise and temperature text and its constants list, register E1, I19 -
  the open edges, per decision, the shaping's form, Lc0's every-node temperature against
  KataGo's root one (edit 123; issues 150-152); §12.8's exploration and §12.3's match play
  - the root temperature self-play only, its schedule the generator's (issue 152). No
  truth source changed.

### 2026-10-22 - Engine ABI round 2a: D3's value outputs in the search - the WDL backed up with its draw, one deliver call for every head, 6m's switch

- Register D3 (Committed, class C; §8.3, corrected at CHECKPOINT 7, reversal 28): "WDL on
  the game result - the value search uses". The engine side, in engine ABI round 2's first
  commit (one round, one owner: the engine lane; the Python mirror in the same commit).
  `engine/include/t5d/mcts.hpp`, `src/mcts.cpp`: an `Evaluation` carries D3's heads from
  the leaf's acting player - `wdl` (win, draw, loss; required; `kWdl`), `value` (the
  51-bin categorical over the root search value, now `std::optional`), `error` (the error
  head's output, >= 0; NaN: none) and `moves_left` (D4; NaN: none). A leaf's value is
  W − L of the normalised WDL (`wdl_value`); the categorical's mean is no longer backed up
  (it was the leaf's value before - the reading reversal 28 corrected). The search refuses
  a WDL or categorical that is no distribution and a negative or infinite error or
  moves-left, releasing the batch.
- The draw/moves-left decision (the plan critic's missing item 3; issue 148): the **draw
  probability D is backed up** beside the value on every node and edge, Lc0's way - a
  leaf's its WDL's, a mate's 0, a stalemate's and the turn cap's 1 (I11: a draw), a solved
  node's 1 for a proven draw; an MCGS residual corrects D as it corrects Q [untrue as first
  built - only a Q lag fired a residual; corrected by the review fixes]; D does not flip
  across a change of acting player. So a root child's search WDL is `((1 - d + q) / 2, d,
  (1 - d - q) / 2)` - what I13's best-move value and D3's `q_ratio` blend need. **Moves-left
  is recorded, not backed up**: what an edge adds to the count needs the unit of a turn
  (issue 305) and whether the action in progress counts (issue 224, open); each node keeps
  its network outputs, and `RootChild` exposes the child's own evaluation in the root
  player's view (`evaluated`, `network_wdl`, `network_moves_left`, `network_error`) beside
  the backed-up `d`, so I13's targets are buildable: the best child's (q, d) and its
  network moves-left plus the target builder's offset. `check_invariants` holds the draws
  consistent through backup and within [0, 1]; the digest hashes them. `Node` +40 bytes,
  `Edge` +8 (one double).
- 6m's switch (register I10, §13.1 6m; the critic's missing item 2; issue 149):
  `SearchParams::uncertainty_source` - `Categorical` (the categorical's variance; the
  default, by D3's "its variance and overlap feed search") or `ErrorHead` (the error head's
  output, a squared gap, taken as the variance) - is each node's variance, which all three
  readers take: E2's cPUCT scale and backup weight, E3's bonus, E5's optional LCB term. A
  signal not given is 0 [corrected by the review fixes: refused while one of them reads
  it, 0 and counted otherwise]. E2's and E3's constants stay placeholders (issues 81, 149).
- The built-in evaluators give a WDL: `uniform` `{1, 1, 1}` (value exactly 0, draw 1/3),
  `random` a hash-derived draw share in [0, 0.5] with W − L its bump's mean scaled,
  `material` `wdl_of(v)` (no draw) beside `two_hot(v)`. `Search::run` resets every head of
  its reused buffers. Search results change only as the new value semantics require: the
  uniform evaluator's searches are bit-identical in every statistic but the new draw sums
  (its W − L is exactly the categorical's old mean, 0); `random` and `material` now back up
  their WDL's W − L instead of the categorical's mean (`material` the same value up to
  float rounding); every search hashes its draws into the digest; root noise and
  everything else unchanged.
- The C API (`engine/include/t5d/capi_search.h`, `src/capi_search.cpp`): **one new deliver
  call**, `t5d_search_deliver_evaluations(search, const t5d_evaluations*, error)` - a struct
  of the batch's arrays: the logits, the submit logits (optional), `T5D_WDL` (3) WDL
  numbers per leaf (required) and the categorical (51 per leaf, optional), each as
  probabilities or logits (`wdl_kind`, `value_kind`), one error-head output and one
  moves-left per leaf (optional; NaN at a leaf: none), and `board_logits` reserved for D2's
  flat form (not built: anything but NULL is `T5D_ERR_UNSUPPORTED`). A struct is appended
  to like the others, so the flat form's per-board logits need no third entry point. **The
  old `t5d_search_deliver` is retired**, not shimmed: it took the categorical alone and
  backed up its mean; a shim would have had to invent a draw probability the documents do
  not give, and its one caller, the Python binding, moves in this commit.
  `t5d_search_params` gains `uncertainty_source` (`T5D_UNCERTAINTY_CATEGORICAL` by
  default, `_ERROR_HEAD`) in front of the embedded `expand`; `t5d_search_summary` appends
  the root's backed-up `root_d` and its own evaluation (`root_evaluated`, `root_wdl`,
  `root_moves_left`, `root_error` - the network's raw outputs at the root, I15);
  `t5d_root_child` appends `d`, `evaluated`, `network_wdl`, `network_moves_left`,
  `network_error`; `t5d_search_layout` reports every new field and `t5d_evaluations`.
- Python mirror (`python/t5d/engine`): `capi.py` (`CEvaluations`, the new fields, the
  constants `T5D_WDL` and `T5D_UNCERTAINTY_*`; the layout check passes); `search.py`
  (`Search.deliver(logits, wdl, *, values=None, errors=None, moves_left=None,
  submit_logits=None, wdl_kind, value_kind)`; `SearchInfo` and `RootChild` gain the new
  fields); `evaluator.py` - `Evaluation(policy_logits, wdl_logits, value_logits=None,
  submit_logit, error, moves_left)`: `value_mean` is now W − L, the categorical's moments
  are `categorical_mean` and `categorical_variance`; `_split` delivers the network's WDL
  logits, its categorical, moves-left in turns (`expm1` of the head's `log1p`, floored at
  0) and the error head's output where the network has one (`NetOutput.value_error`; NaN
  until the network lane's D3 heads add it); `check_evaluation` checks every head;
  `CStepDriver.deliver` sends them all (the categorical for every leaf of a batch or none);
  `UniformEvaluator` gives WDL logits (0, 0, 0), bit-exact with the engine's `uniform`.
- Tests. `engine/tests/test_mcts.cpp`: `mcts_d3_the_wdl_is_the_value_and_its_draw_is_
  backed_up` (W − L backed up whatever the categorical says, D beside it, every node and
  edge, both splits, tree and graph - residual corrections included [whose draw correction
  this could not test: every leaf's D was 0.2; tested since the review fixes] - and the root
  children's `d` and network WDL); `mcts_d3_the_categorical_changes_only_the_uncertainty`
  (with E2, E3 and E5's term off two categoricals grow the same tree value for value; with
  E2 on they steer; under the error-head signal the categorical is unread and the error
  steers); `mcts_d3_6m_chooses_the_nodes_uncertainty_signal` (each signal, each absent:
  0); `mcts_d3_game_ends_back_up_their_draw_probability` (mate 0, stalemate and cap 1);
  `mcts_d3_the_evaluation_cache_returns_every_head` (§15.2's purity row: every head bit for
  bit, a fresh search's root from a shared cache included); `mcts_d3_the_builtin_
  evaluators_give_a_wdl`; `mcts_run_hands_the_evaluator_reset_buffers`; the malformed-
  evaluation test head by head; the turn-cap test's D. `test_capi_search.cpp`: the step
  loop with WDLs, categoricals, errors and moves-left equal to the C++ search by digest
  under both signals with E2 on, the root children's `q`, `d` and network outputs equal;
  the WDL as probabilities and logits (win, draw, loss not mirrored; the categorical not
  the value); every refusal of the batch struct, each releasing the batch; the layouts;
  `uncertainty_source` validated. Python (`tests/engine`): the engine's `random` evaluator
  rewritten with its WDL still equal to the engine's run by digest (both arms, tree and
  graph, noise, synchronous, past the shift); `test_d3_the_wdl_is_the_search_value_and_
  its_draw_is_backed_up` through the step API (the root's and the children's network
  outputs included); `test_d3_6m_the_uncertainty_source_is_a_parameter`; the
  acting-player tests with the WDL (the categorical set the other way round); the
  evaluator's heads against the network's own outputs, the helpers, `check_evaluation`.
  Checks: engine 571 of 571 (`py engine/build.py --werror --test`); Python `tests/engine`
  177 passed, 1 skipped (`T5D_REQUIRE_ENGINE=1`; the skip is `T5D_RECORDS`'); ruff check and
  format clean; `harness verify` ok; `gate --tier commit`: the oracle layer ok, the engine differential "SMOKE TEST - PASSED" (41
  records compared with the last green run, 0 declined), the proof channel DISABLED.
- `engine/README.md`: "Nodes and values", E2's and E5's signal, "The evaluator", the C
  API (the new call, the retirement, the new fields), the tests, the readings (issues 148,
  149). `docs/BUILD_LOG.md`: register D3 (engine side built), C14 (the cache's entries:
  `Evaluation` 232 -> 256 bytes), E2/E3, E5, I10 (6m's switch built), I13 (the engine side
  built); spec issues 148-149; proposed edits 121-122.
- Spec: §8.3 and register D3, I13 - the search backs up W − L and D, records moves-left
  (edit 121, issue 148); §13.1 6m, §8.4's E2/E3 text and constants, register I10, E2 - the
  switch, its readers and default (edit 122, issue 149); §15.2's purity row (C19-C21) - an
  evaluation's every head, as built; C14's cache entry size (an engineering figure, no
  edit). No truth source changed (`T5D_TRUTH_SOURCES`).

### 2026-10-23 - Build log: issues 225, 226 and 229 wait on the user

- `docs/BUILD_LOG.md`, "Waiting on the user": spec issue 225 (F6/M20: the acting
  player with no global registers) and 226 and 229 (A7: the per-square labels as
  ours/theirs; the other colour's refresh view), none blocking.
- Spec: none.

### 2026-10-22 - A7 review fixes: the other colour's refresh view without the anchor's moves; the mirror's own timeline map

Two reviewers' findings on the A7 commit (00c4c41); no wrong result in the canonicalisation
itself.

- **A refresh's other-colour view** (`net/cache.py`, `viewed_by`; spec issue 229, new). It
  swapped the acting player and cleared the submit flags but kept the anchor's moves and
  edges, which are the anchor's acting player's (the engine builds the move graph and the
  typed relations for the player to move's pieces only): from the other colour's side every
  move edge started on "their" pieces, and under B9's symmetric move-graph mask frozen squares
  attended along them in the dense refresh pass - an input no conversion produces (strict
  conversion refuses it), encoded into the history K/V of every node the other colour acts at,
  and invisible to the equivalence tests because `reference_output` built the same anchor.
  The view now carries no moves, no reply moves and no edges. Chosen over giving it the other
  colour's own moves: no expansion supplies them (the anchor's boards with the other colour
  acting are no game state, and the network computes no rule), while a position with no move
  is one conversion does produce. A committed-class silence under the deferred cached
  configuration (C2, C3), so a placeholder fixed in code, with no second arm until something
  supplies the other colour's move graph. Under the block-causal scope the edges open and bias
  live rows only, so a root's other-side history built without a refresh is unchanged.
  Tests: `test_the_other_perspectives_anchor_has_no_moves_or_edges` (synthetic, six
  constructions, two seeds) and `test_the_other_perspectives_anchor_on_real_positions` (the
  engine's positions, both acting colours) capture every dense pass of a refresh and of
  `reference_output`: the own view keeps the anchor's moves and edges (on the engine's, every
  source the acting player's), the other's has no move, reply move or edge - none whose source
  is not that view's acting player's - and cached = reference from both sides. Both fail
  against the old `viewed_by`.
- **The mirror's timeline map** (`net/state.py`, `mirror`). It negated timelines through
  `canonical_timeline`, the map under test; that map is self-inverse, so any wrong
  self-inverse map cancelled in the synthetic mirror test (patched to the identity, every
  output test still passed). `mirror` now writes the negation itself (`-1 - p` in a +-0
  variant, else `-p`), as it already wrote the rank mirror; its docstring says the engine test
  pins the map against the notation. Mutants of `canonical_timeline`, patched in at test time
  (no file kept edited), against `tests/net/test_perspective.py -k mirror`: the identity - 90
  failed, 18 passed; `-p` for `-1 - p` - 2 failed (`test_mirror_has_identical_canonical_inputs`
  on `even_start` and `shifted_submit_absent`; the outputs cannot see a shift of one, the
  coordinates being recentred).
- **The engine test's variants** (`tests/engine/test_perspective_engine.py`): it iterated
  `helpers.VARIANTS`, ten of the engine's variants, while its docstring and the A7 entry said
  "every variant". It now iterates `helpers.ENGINE_VARIANTS` (new): the engine's whole table,
  16 named variants (`Misc - Small` is `Small`'s alias); about 20 s for the file.
- **Documents.** The memory cost read as conditional ("once both colours' nodes are
  reached"); as built, `root_epoch` and `_refresh` compute the other side eagerly, so every
  store holds two entries per root (or refresh anchor) frozen board from the moment its epoch
  is made, and a board superseded later gets its second entry only when the other colour reads
  it: reworded in `docs/MEASUREMENTS.md` (the KV-memory row and the A7 conditions note),
  issue 303's note and the A7 entry below. `tie()` in `tests/net/test_perspective.py` says
  that the untied carrier assertion is what guards anything routed through the acting-colour
  embedding. `docs/BUILD_LOG.md`: the A7 row (the variants; the review fixes; a note for A8:
  the board-layout embedding must flatten each board in canonical (x, y) order, not storage
  order, or A7 breaks for Black), issue 303's note, spec issue 229 and proposed edit 170
  (new).
- Checks: network 1197 passed, 1 skipped (`T5D_REQUIRE_ENGINE=1`; the skip is `T5D_RECORDS`'s);
  `ruff check` and `ruff format --check` clean; `harness.py verify` ok.
- Spec: §7.9 (the refresh pass; conditioning-offset training), §7.7, §2.2, §15.2's A7 row,
  §15.3's KV-memory row; register A7, C2, C3, B9 - spec issue 229 (new: the other colour's
  refresh view, a placeholder) and proposed edit 170 (new), with the main loop's 201; issue
  303's note reworded.

### 2026-10-22 - Network A7: perspective canonicalisation, the colour-swap mirror test

Register A7 (Committed, class C; spec §2.2): the network reads every input from the acting
player's side, as AlphaZero, Lc0 and KataGo do.

- **The view** (`net/state.py`). `PositionState` keeps the engine's absolute facts; its
  `token_arrays` and `collate` give the network the acting player's view: each square's owner
  ours/theirs (`OWNER_*`, `PositionState.owner`) instead of its colour, each board's side to move
  us/them (§2.3), and when Black acts every rank mirrored (`y -> H - 1 - y`, §2.2's
  `r -> size + 1 - r`) and every timeline negated (`p -> -p`; in a ±0 variant `p -> -1 - p`,
  §2.2's `i -> -1 - i` in §12.6's index mapping, which swaps -0 and +0). File, time and Δt are
  unchanged. `PositionState.even_start` (new, default False) carries the variant's parity: the
  converter fills it from the engine's `Expansion.even`, `synthetic_state` and the tree fixture
  from their own. `Batch.colour` is now `Batch.owner`; `Batch.stm` is relative; `Batch.coords`'
  rank and timeline and `coord_ref`'s timeline reference are canonical.
- **Outputs mapped back.** The view moves no token, so every token-indexed output - the policy's
  moves, the reply's, the per-board and per-square outputs - maps back by the identity (policy
  logit `i` still scores engine move `i`, for either colour); the value is already the acting
  player's (D8). The per-square labels' white/black facts are relabelled ours/theirs
  (`heads.SQUARE_LABELS`; spec issue 226), and `heads.square_labels_by_colour` maps the logits
  back to the colour names (`SQUARE_LABELS_BY_COLOUR`).
- **The embedding** (`net/embed.py`). A2's colour table is the owner table (`embed.owner`: none,
  ours, theirs - renamed in place, so no parameter and no draw changes; F1 and `V1_TOTAL` as
  they were); the side-to-move flag reads us/them. D8's acting-player embedding stays where M20
  put it and is now A7's acting colour, "for the few colour-asymmetric facts (White moves
  first)" - one input for A6's "acting player and acting colour" (spec issue 228). The
  docstrings of `embed.py`, `model.py`, `state.py`, `convert.py` and `evaluator.py` say so;
  `embed.py`'s and `cache.py`'s premise that no board token depends on the acting player is gone.
- **The cached path** (`net/cache.py`; spec issue 303, as the plan read it). A frozen board's
  input is now the acting player's view of it, so every frozen board's K/V are keyed by the
  acting player (`board_ident(..., acting=True)` in the dependency digest); history membership
  stays by the intrinsic identity. A search root's frozen boards are built from both colours'
  sides when its epoch is made (`root_epoch`), and a refresh runs its dense pass twice - the
  anchor as it is and the same boards with the other colour acting (`viewed_by`, new) - so a
  node of either acting player reads history in its own frame and no node recomputes a root
  board. `reference_output`'s anchor pass is the node's view. The cost - twice the history
  build and the refresh, and, the other side being computed eagerly, two entries per root
  frozen board in every store from the moment its epoch is made (review fix) - is
  C2's to weigh (issue 303's note); `cache_bench` reports `analytic_both_perspectives` and
  `cache_bytes_per_perspective` beside its one-perspective figures.
- **The mirror** (`state.mirror`, with `mirror_tokens`, `square_mirror`, `canonical_timeline`
  and `canonical_rank`; `mirror` exported from `t5d.net`): a position's colour-swapped mirror -
  colours swapped, ranks mirrored (edges and moves remapped), timelines negated by A7's map,
  the acting player swapped - with every board one half-turn later so that its side to move is
  the other colour (spec issue 227: invisible to the network, which reads time relative to the
  present; the mirror of the mirror is the position two half-turns later).
- **Tests.** `tests/net/test_perspective.py` (new) - §15.2's A7 row read as issue 304 says:
  every field of the canonical input equal bit for bit at the mirrored token but the acting
  colour, the embeddings differing exactly where it enters (the global registers; with none,
  issue 225's board tokens); every output equal mapped back with the colour embedding's rows
  tied (a test hook), FP32 within 1e-5, the shipped BF16 network no further from its mirror
  than from its own FP32 computation; untied, the outputs differ; on 18 constructions (many
  timelines, deep travel, duplicated boards, ±0, fairy pieces, past the shift with submit
  legal, masked and absent, 7x7 to Global Warming's 1x1, far apart, 4/0 and 0/16 registers),
  3 seeds, both scopes, B5/B10/B11/B12/B17 and B15 off; the board order; the index mapping;
  the label map-back; and the cached path under both supersession readings, with and without
  a root refresh, equal to the full pass for both acting colours and equal across the mirror.
  `tests/engine/test_perspective_engine.py` (new) - the engine's own 5DFEN mirror (rows
  reversed, case swapped, labels negated, one half-turn later) converts to `mirror` of the
  conversion, moves, move graph and relations included, for both acting colours, ten of the
  engine's variants (review fix: all 16), the records' positions and random play; the network agrees on the two (both
  scopes, B11 on, moves matched in the engine's own order); positions past the shift agree
  with their mirrors. `test_d8.py` rewritten for A7: a flipped acting player changes every
  board token's encoding (owners, side to move, mirrored coordinates), and its colour
  embedding alone changes the embeddings only where it enters; cached = full pass with two
  entries per frozen board and no node recomputing one; a refresh stores each perspective's
  own K/V. `test_cache.py`'s counts are per perspective (the root's history twice; a node
  whose move gives the other colour the action reads its parent's boards from that side, so the
  purity, prefix-sharing, `"current"`-features and B16 tests evaluate that view first or count
  each (board, perspective) entry once); `tests/engine/test_network.py`'s shifted-action test:
  every token's embedding moves, the colour's own embedding only where it enters;
  `test_invariance.py` runs its storage-order and translation checks for both acting players;
  `test_embed.py` reads `Batch.owner`. Mutants patched in at test time (no file
  edited): no rank mirror, no timeline negation, absolute owners, absolute side to move,
  frozen keys without the perspective - each caught; `-p` for `-1 - p` in a ±0 variant
  changes no output (the coordinates are recentred) and is caught by the canonical-input test.
- `docs/BUILD_LOG.md`: register rows A7 (built, tested), A1-A3, A6, B15, C2, D4, D8; notes on
  issues 303 (built; its cost by construction) and 304 (built); spec issues 226-228 and
  proposed edits 167-169 (new). `docs/MEASUREMENTS.md`: a conditions note - the cached rows
  are one perspective's; under A7 the history build and the refresh double (analytic root
  refresh at 800 simulations 7.28% / 2.48% / 12.26% / 6.20%) and every store holds twice the
  KV bytes for its root's frozen boards (review fix: computed eagerly); arithmetic from the construction, the counts and wall times the
  quiet-machine block's.
- Checks: network 1190 passed, 1 skipped (`T5D_REQUIRE_ENGINE=1`; the skip is `T5D_RECORDS`'s);
  `ruff check` and `ruff format --check` clean; `harness.py verify` ok (pins, corpus, known
  divergences, rulebook, boundary, provenance, golden observations). F1 and `V1_TOTAL` unchanged.
- Spec: §2.2 (perspective canonicalisation), §2.3 (side to move us/them; the acting colour),
  §12.4's per-square row, §15.2's A7 row, §7.1, §7.9, §10.3 and §15.3's cached rows; register
  A7, A2, A3, A6, D4, D8, C2, C3, C17, M20 - spec issues 226 (the per-square labels'
  relabelling), 227 (the mirror's half-turn shift; the identity map-back), 228 (one
  acting-colour input), notes on 303 and 304; proposed edits 167-169 (new), with the main
  loop's 201 and 202, which this change builds.

### 2026-10-22 - Engine: the shared proof stores' locks correct on every platform (a data race under winpthreads' shared_mutex)

- The defect (register H17; found at E-A6's review on 2026-10-17 and parked there).
  `ConcurrentSolvedStore` (`engine/src/scanner.cpp`) and `DisproofStore`
  (`engine/src/prover.cpp`) guarded their 64 shards with `std::shared_mutex`, which with
  this machine's toolchain (MSYS2 ucrt64 g++ 16.2, winpthreads) does not exclude on a lock's
  first concurrent use. The cause, confirmed by experiment: libstdc++ builds `shared_mutex`
  on a `pthread_rwlock_t` with the static initialiser; winpthreads initialises it on its
  first call and returns `EINVAL` to the threads that lose that race (8 threads on 64 fresh
  locks: 149,878 and 156,542 of 1,024,000 `pthread_rwlock_wrlock` calls, in two runs); and
  libstdc++'s `shared_mutex` ignores the error (only an assertion build would see it). So
  the losing thread ran its critical section without the lock, then unlocked a lock it did
  not hold. Every store the scanner's and the prover's threads share is first taken by all
  of them at once: a scanner's own, and Guard's and Hunt's, built per call. A race in a
  proof store is a soundness hazard, not only a crash: under
  `scanner_lazy_smp_store_refuses_contradictions` the old stores crashed, held one position
  twice and lost a shorter distance (14 of 600 runs on 2026-10-17, 85 of 600 on 2026-10-22),
  and a position held twice can hold two kinds of proof.
- The fix, correct by construction on every platform: `t5d::ShardLock`
  (`engine/include/t5d/scanner.hpp`), a `std::mutex`, guards every shard of both stores, and
  every call - probes included - takes it with `std::lock_guard`. libstdc++'s
  `std::mutex::lock()` throws `std::system_error` on any error the platform reports, so it
  never returns without the lock: the correctness rests on that contract, not on how a
  threads library initialises a lock. (In the same experiment `pthread_mutex_lock` never
  failed; glibc's mutex needs no first-use initialisation.) The review's warm-up (each
  shard's lock taken once at construction) was not chosen: it would rest on winpthreads'
  internals, not on the lock's contract, and it keeps the slow lock (below).
- The audit of `engine/`. No other `std::shared_mutex`, `shared_lock` or
  `shared_timed_mutex` is left. The scanner's and the prover's TTs are lockless (atomics).
  The remaining locks are `std::mutex`: the serialisation of a caller's store that is not
  safe for threads, the prover's certificate, and the tools' output (`t5d-bench`'s pool
  waits on a `std::barrier`). MCGS and the evaluation cache have no lock and are
  single-threaded (H15). `t5d_search_new_sharing` (`capi_search.h`), `EvalCache`
  (`mcts.hpp`) and the Python `Search` docstring (`python/t5d/engine/search.py`) now say
  that the searches sharing one cache are used one at a time.
- Tests (`engine/tests/test_shared_stores.cpp`, new; `engine/CMakeLists.txt`):
  - `shared_store_lock_excludes_at_first_concurrent_use` runs 400 rounds of 64 fresh
    `ShardLock`s, with 8 threads released at once, each taking every lock in the same order
    and holding it a few microseconds. No thread may find another inside, and no increment
    may be lost.
  - `shared_stores_hold_at_first_concurrent_use` runs 150 rounds of a fresh
    `ConcurrentSolvedStore` and `DisproofStore`. 8 threads released at once probe and write
    185 positions spread over 60 of the 64 shards. Every shortest proof, every deepest
    disproof and exactly one entry per position must be kept.
  - Built against the old code (a scratch copy of `e8de15e` with this file, `ShardLock`
    named `std::shared_mutex`), each failed in every run: 50 of 50 on 2026-10-22 (the lock
    test with 136-876 overlaps a run, and lost increments in 31 runs; the store test by
    crashing), as in 120 of 120 on 2026-10-17. With `ShardLock` both passed 300 of 300 runs
    (500 of 500 on 2026-10-17).
  - `scanner_lazy_smp_store_refuses_contradictions` is kept: with the old stores it failed
    in 85 of 600 runs (58 crashes; 27 runs with a position held twice or a shorter distance
    lost), and with the new it passed 1,000 of 1,000 with 0 crashes (as on 2026-10-17).
- No result changes. No verdict logic, proof, disproof or distance is decided differently.
  At 1 thread the old and new builds searched the same nodes and found the same proofs (184
  positions from random play, 20,000 nodes each: Hunt 1,252,177 nodes and 39 mates, Guard
  2,786,845 and 30 refutations, the scanner 1,510,209 and 49 proofs).
- The cost, indicative only: the shared development machine, so not a measurement and not in
  MEASUREMENTS. winpthreads' reader-writer lock costs 1.4-1.8 µs a call uncontended (the
  mutex 9 ns), and every call takes one process-wide spinlock. Over 509 positions, at
  find/put mixes of 99/1, 90/10 and 50/50, the stores ran these operations per second, every
  thread together (2026-10-22, load 56%; the old store a replica, each shard's lock warmed
  first, so this is the cost alone):

  | Threads | Old (`std::shared_mutex`, shared probes) | `ConcurrentSolvedStore` | `DisproofStore` |
  |---|---|---|---|
  | 1 | 0.50-0.54 M | 12.5-13.9 M | 13.0-13.5 M |
  | 5 | 0.07 M | 39-43 M | 22-31 M |
  | 12 | 0.05 M | 44-51 M | 31-46 M |

  In the searches, with 20,000-node budgets on 184 positions from random play (the old and
  new builds alternated, the median of three), at 5 threads Hunt ran 1.45× (2026-10-17) and
  1.69× (2026-10-22, load 91-96%), Guard 2.71× and 2.90×, and the scanner 1.22× and 1.34× as
  many nodes per second as before; at 1 thread 0.94-1.07×, noise. On Linux (glibc) the old
  lock was correct, and the change there gives up readers' sharing of a shard (one of 64)
  for a plain mutex - not measured here.
- The truth's sources. `scanner.hpp`, `scanner.cpp`, `prover.hpp` and `prover.cpp` are in
  `T5D_TRUTH_SOURCES`, so the truth's code signature changes again. It already differed from
  M17's: `9be6f04` typed the prover's hash constants, and `e554da0` changed the prover's
  reported line. No verdict logic changes. M17's verdicts stand on what they were read from:
  every 6b″ refutation and mate was confirmed by E4 alone. M17's 5-thread rates, though, were
  measured with the old lock's race and cost, so they may understate what 50 ms reaches.
  Whether to re-run 6b″ on a quiet machine is the main loop's call (issue 147). This touches
  the scanner and the provers, so the main loop runs the Lazy SMP timing test before
  merging.
- Docs:
  - `engine/README.md`: "The stores' lock" under Lazy SMP (why not a reader-writer lock, why
    a mutex is correct by construction, why not a warm-up, the audit, the cost, the tests),
    with the scaling figures marked as taken with the old lock; "The truth's sources since
    M17" under 6b″.
  - `docs/BUILD_LOG.md`: the H17 and M17 rows; "Parked, and why" (the stores' locks
    unparked); spec issues 146-147; proposed edit 120; the Lazy SMP timing test log (this
    branch's runs, before review).
- Checks (MSYS2 ucrt64 g++ 16.2, Release, a shared machine):
  - `py engine/build.py --werror --test`: "t5d-tests: 557 passed, 0 failed, 557 run".
  - The two new tests, 300 runs each: 300 passed, 0 failed, 0 crashed, each. Against the old
    code, 50 runs each: the lock test 50 failed, the store test 50 crashed.
  - `scanner_lazy_smp_store_refuses_contradictions`, 1,000 runs: 1,000 passed, 0 failed, 0
    crashed (the old code, 600 runs: 515 passed, 27 failed, 58 crashed).
  - The Lazy SMP timing test (32 positions): 1 thread 292 / 292 / 293, 5 threads 308 / 311 /
    309, pass x3 (load 94-96%); 291 / 313 inside the full run (the timing test log).
  - The threaded long tests over the shared stores (`--long long_guard long_hunt
    long_prover_threads long_scanner_lazy_smp`): "t5d-tests: 6 passed, 0 failed, 6 run"
    (24 min). Their counts are those of the runs before the change (2026-10-17) but for
    two that threads make nondeterministic (H15): the prover's threads gave 159 proven, 166
    disproven, 59 unknown (one unknown more decided; 309 confirmed, 16 beyond the
    references), and the scanner's Lazy SMP 1,994 proofs agreeing with one thread's
    (1,996).
  - Python, from `python/`: `ruff check .` "All checks passed!"; `ruff format --check .` "72
    files already formatted"; `T5D_REQUIRE_ENGINE=1 pytest -q tests/engine` "168 passed, 1
    skipped".
  - `verify`: ok (its 17 `proof-channel-vacuous` notes, H48). `gate --tier commit`: "oracle
    layer [commit]: ok"; "engine differential: NOT RUN - no engine adapter is configured in
    oracles/gate.toml"; "proof channel: DISABLED".
  - Linux (glibc): not run. The engine job runs these tests on ubuntu-24.04 with g++ 13 and
    14 on every push, but no GitHub Actions job has started on any branch since 2026-10-17
    09:56 UTC, the nightly of 2026-10-22 included (the account's billing), so it waits for
    a re-run.
- Review fixes (one reviewer, three low findings): the evaluation cache's no-two-threads
  rule worded as the caller's duty in `capi_search.h`, `mcts.hpp` and `search.py`; the old
  lock's handle leak recorded (README "The stores' lock", the H17 row); the H17 row's
  overlap range dated.
- Spec: §11.3 (Lazy SMP, the shared stores; register H17, H19) - issue 146, proposed edit 120
  (the stores exact under concurrency, their races never benign). Register H41: the stores
  hold what the proof channel would export; it stays off, and this removes a soundness
  hazard from under it - no edit. §11.4's 6b″ result and §15.6 (register M17, H42, H43) -
  issue 147, no edit until a re-run. No part of the spec describes the stores' locks.

### 2026-10-22 - Build log: a stale note removed from "Register status"

- `docs/BUILD_LOG.md`: the preface saying H47, H48 and M22 had no row yet is gone -
  their rows arrived with the gate and engine lanes' merges.
- Spec: none.

### 2026-10-17 - N-F6heads-M20 review fixes: the acting player without global registers; the documents brought in line

Three reviewers' findings on the two commits below (none refuted):

- **D8 with no global registers (ablation 3's arms).** M20 put the acting player on the global
  registers alone, so with `global_regs = 0` it vanished: every output was bit-identical under
  either acting player and the embedding got no gradient - the arm measured "no registers"
  mixed with "no D8 input". `embed.py` now adds the acting player's embedding to every board
  token when there are no global registers - squares and per-board registers - and under the
  block-causal scope to the live (head) boards' tokens only, so no frozen board's input
  depends on it and the cached path stays exact (`InputEmbedding.forward`'s `block_causal`,
  passed by `T5DNet.embed_tokens`). A placeholder fixed in code (spec issue 225): no document
  says whether ablation 3's "off" removes the global registers, or where their inputs go; the
  value-type heads and the submit logit pool the squares there, as built since N1. No
  parameter changes. Tests: `test_d8.py` - the encoding test under three register arms (the
  carriers by scope), the acting player reaching the policy, every value-type head, moves-left,
  effort and the submit readout under all six register arms the configuration accepts and both
  scopes, the cached = full-pass and refresh rows with no global registers; `test_embed.py` -
  the embedding by hand with no global registers, dense and block-causal;
  `tests/engine/test_network.py` - the shifted action against the opponent's turn on real
  positions under three register arms, and the acting player alone (same boards, edges and
  moves) moving every value-type output and the submit readout. Mutants patched in at test
  time (no file edited): no fallback fails 17 tests; the fallback on frozen boards too fails
  the cached = full-pass rows with no global registers.
- **Docstrings.** The pre-M20 parameter totals (`config.py`'s presets comment: F2 46,894,547;
  `shaw.py`, `test_displacement.py`: the v1's 22,965,139); `rope.py`'s `rope_roles_source`
  now stands as v1 (M20, issue 16f), not an open point; D8's draw-order claim in `model.py` and
  `heads.py` narrowed to what holds - D8's parameters change no other parameter's draw - and
  M20's promotion offset noted as having moved every global-stream draw; the Δt MLP's and
  smolgen's activations listed among `config.py`'s committed-class placeholders (issue 221),
  the "not committed-class" sentence narrowed to the MLP's width and features, and `embed.py`
  pointing at issue 221 instead of calling the activation an implementation choice;
  `config.py`'s `ffn_activation` cites proposed edit 154 for the squared-ReLU arm's
  initialisation; `attention.py`'s and `bench.py`'s cost caveats name M20's register
  positions (about +35% / +11% per attention call over the old default, FP32 / BF16).
- `docs/MEASUREMENTS.md`: N3's counts and the §15.3 rows built on them carry a conditions note
  - counted on the network before F5, B15, B16 and M20's register positions; no new figure.
  The M20 entry's reason for not re-counting is corrected (counts do not depend on load), and
  its cost figures say "against one plain call".
- `docs/BUILD_LOG.md`: spec issue 225 and proposed edit 166 (new); notes on issue 108, 204
  (the squared-ReLU arm's initialisation still open), 218 (the post-M20 count) and 303
  (`board_ident`'s `acting` keying independent of the frozen-feature reading, unused until
  A7); register rows A7 (M20 built; A7 its own component), D8, M20 (the `"current"` arm kept
  with no named ablation - the main loop's call); the Register status preface no longer lists
  F5 and F6 as missing.
- Not done: `frozen_features = "current"` stays (removing it is a B16 change the component was
  told not to redo; recorded for the main loop).
- Checks: network 1067 passed, 1 skipped (`T5D_REQUIRE_ENGINE=1`; the skip is `T5D_RECORDS`'s,
  by design) in 23 min 30 s; ruff clean; `verify` ok (its 17 `proof-channel-vacuous` notes are
  H48's, as before).
- Spec: §2.3, §13.1 ablation 3 and register D8, A6, D3, M20 - spec issue 225, proposed edit 166;
  §9.3 and register F6 (issue 221's activations, committed-class; issue 204, edit 154);
  §15.3's dense and cached cost rows (the N3 counts predate M20; re-count pending, no
  figure changed); §9.2 and F1 (no parameter change: 22,965,139).

### 2026-10-17 - Network M20: the placeholder dispositions as the register rules them

- Register M20 (Committed; BUILD_LOG issue 121, CHECKPOINT 6) in the network: each
  disposition is now the code's default, and an arm it disposes of is removed where no named
  ablation needs it (an unknown field now, not a switch at a ruled default). By disposition:
  - **Promotion scoring -> learned per-piece offset (D1).** `heads.PointerHead` takes
    `promotion_offset`; the policy always has it - one logit per piece type read from the
    destination token, as N1's `offset` arm built it - and `policy_promotion` (with its
    `none` default) is gone. The opponent-reply head stays without one, as built at N1
    (issue 222: D1 names §8.1's policy; which other pointer heads carry it is the heads'
    rebuild's).
  - **Opponent-reply head -> a prototype (D4).** Not here: its reinstatement (behind a flag,
    weight 0.15, the reply mapped onto current squares) is the heads' component, after
    engine ABI round 2; the head is left always built, as at N1, until then.
  - **Aux pooling -> squares; value pooling -> mean (D3).** `aux_global_pool` is gone.
    `T5DNet.heads`: the value-type heads - the 51-bin value, WDL and the short-term value -
    read the global registers' mean (D3, §8.3: "each pooled by mean from the global
    registers"; WDL and the short value moved off the squares, which N1's placeholder default
    read), the other global auxiliary heads - moves-left, search effort - every square's
    mean; per-board and per-square heads as before (issue 223).
  - **Moves-left -> turns.** The head predicts log1p of the turns left, a turn one player's
    action (issue 305's unit): `heads.py`'s docstring, `moves_left_from_log`,
    `NetOutput.moves_left_log`, `losses.Targets.moves_left` and the losses' docstring. The
    record-to-target builder is the training loop's (M21), and so is whether the action in
    progress counts (issue 224); the smoke test's random targets are unit-free.
  - **Registers' positions (B8).** Per-board registers rotated on timeline and time only,
    global registers position-free on every pair - no switch: `board_register_spatial`
    (`learned_angle`, the old default, with its 2 x 2 learned coordinates, and `zero_angle`)
    and `global_register_pairs` (`anchored`) are gone, with `AttnContext.global_free`, the
    anchored rows in `cache.py`'s live pass and `flex.py`, and `model.coordinates`'s learned
    branch. `rope_recenter` stays (FP32 angle hygiene). The CPU fast path now widens every
    row's Q/K/V (44 wide for the B4 allocation) and recomputes the register rows:
    indicatively +20% / +55% per attention call at 240 boards against one plain call - about
    +11% / +35% over the old default (BF16 / FP32; `attention.py`'s earlier timings, not
    section 15.3 measurements). N3's counted dense FLOPs in `docs/MEASUREMENTS.md` were
    taken with the old `learned_angle` registers; they are not re-counted here (the counts
    do not depend on load, but counting runs the passes themselves - 28 TF at 600 boards -
    so step 2's quiet-machine block re-counts them with the wall times; corrected in the
    review fixes, which also note it in `docs/MEASUREMENTS.md`).
  - **Frozen features -> intrinsic, with B16; binned-bias-with-RoPE -> moot (B15);
    supersession scope -> both built (C20).** Already in the code (N-B16, N-B15, N3); checked,
    not redone. `config.py` now calls `supersession_scope`'s `causal_past` the default arm
    until 6c, not a placeholder. `frozen_features = "current"` stays the test arm N-B16 kept.
  - **Acting player -> global registers (D8).** `acting_player_input` and its `all_tokens`
    arm are gone; `embed.py` adds the embedding to the global registers alone (bit for bit
    the old default). `cache.board_ident` keeps its `acting` keying, now independent of
    `frozen_features` and unused by the cached path: the plan's critic kept it for A7's
    per-perspective frozen K/V (issue 303).
  - **Submit logit -> pooled global registers (D8).** `heads.SubmitHead` reads
    `global_register_pool` - the registers' mean, as the value head does - by either readout;
    `submit_register` and `submit_register_state` are gone. The readout's form stays the
    placeholder `submit_head` (linear, or the policy's bilinear form): not on M20's list
    (issue 109).
  - **B5's details, the per-square label set, 16a-g and B7 stand as v1.** Relabelled, not
    changed: `config.py` (`rope_roles_source`, `rope_relations_head`,
    `rope_same_square_split`, `rope_near_nope_max_angle`, `move_graph_heads`,
    `move_graph_direction`, `masked_head_registers`, `same_board_registers`,
    `relations_head_types`, `relations_head_direction`) and `heads.SQUARE_LABELS` now call
    them the v1's values, tested by ablations 2, 6a and 6j; B7 has no code.
- Parameters: F1 22,965,139 (+3,337: the offset's 3,341, less the 4 learned register
  coordinates); F2's preset 46,894,547; with the cached configuration 22,965,443.
  `test_params.py`'s V1_TOTAL and docstring, BUILD_LOG's F1 row.
- Docstrings: `config.py`, `model.py`, `heads.py`, `embed.py`, `attention.py`, `rope.py`,
  `cache.py`, `structure.py`, `state.py`, `flex.py`, `losses.py`, `engine/evaluator.py` and
  `engine/convert.py` (the pooled submit, the promotion offset); `engine/README.md`,
  `engine/include/t5d/capi_search.h` and `mcts.hpp` - comments only (the submit logit read
  from the pooled global registers); no truth source. The BF16 / FP8 gaps recorded in
  `test_precision.py`, `test_cache.py` and `tests/engine/test_network.py` are now labelled as
  measured before B15's table, the heads' Mish and M20's defaults changed the network - not
  re-measured in this component; their bounds hold under the new defaults (the suite passes).
- Tests. New: `test_model.py` - each global head reads its pool (perturbing every square
  moves only moves-left and effort, one global register only the value-type heads and the
  submit logit, the per-board registers none), the register-pooled heads invariant to
  permuting the global registers' final states, promotions of one move to four pieces get
  four logits whose bilinear part ties, a move without a promotion keeps its bilinear score;
  `test_d8.py` - the submit logit is the readout of the registers' mean under both forms,
  unmoved by permuting them, moved by any one, deaf to the squares, and the squares' mean
  with no global registers; `board_ident`'s `acting` keying; `test_rope.py` - a register's
  attention row blind to every token's file and rank while a square's is not, and, through
  the whole network (RoPE and B15 together), layer 0's register rows unchanged when every
  square's (file, rank) is scrambled; `test_params.py` - the removed switches are unknown
  fields, the policy and reply heads' counts. Updated to the ruled defaults: the arms'
  parametrisations in `test_attention` (the pair rule now crossed with ablation 3's register
  arms - no per-board registers, no global registers - where the widening or the merge drops
  out), `test_chunked` (the same arms through `_case`), `test_invariance`, `test_n2_model`
  (its anchored skip gone), `test_cache`, `test_scope`, `test_shaw`, `test_status`,
  `test_displacement`, `test_structure`, `test_d8` and `train/test_smoke`. Mutants patched
  in at test time (no file edited), each failing the new tests: WDL and the short value back
  on the squares; moves-left and effort on the registers; the submit logit from register 0
  (either readout); no promotion offset (the parameter count too); per-board registers
  rotated on file and rank (both register tests, through the attention and the whole
  network); the acting player on every board token (the encoding test, and the cached =
  full-pass row: frozen K/V keyed without it go stale).
- `docs/BUILD_LOG.md`: register rows M20, B8, D1, D3, D4, D7, D8, F1; spec issues 9, 16,
  32, 108, 109, 121 and 210 and proposed edit 156 marked built, relabelled or moved; issues
  222-224; proposed edits 163-165.
- Checks: engine 555 of 555 (`--werror`, rebuilt after the header comments); network 1040
  passed, 1 skipped (`T5D_REQUIRE_ENGINE=1`; the skip is `T5D_RECORDS`'s, by design - the
  anchored arm's skip went with the arm) in 20 min 20 s beside another lane's long engine
  tests; ruff clean; `verify` ok (its 17 `proof-channel-vacuous` notes are H48's, as before).
- Spec: register M20 and the rows it disposes (D1, D3, D4, D7, D8, B8, B16, C20, B15/B6, B5,
  B7, B9/B10) - as built now; §5.3 (registers' positions), §8.1 (the promotion offset),
  §8.2 (the pooled submit logit), §8.3 and §9.2 (value heads on the global registers, aux
  heads on the squares), §12.4 (moves-left in turns), §2.3 (the acting player on the global
  registers); §9.2 and F1 (the parameter count - proposed edit 156's figures move by 3,337,
  noted there); §15.2's rows (cached = dense, translation and storage-order invariance) run
  under the new defaults; §15.3's dense cost to re-count. Proposed edits 163 (the offset's
  form and heads), 164 (which heads "aux pooling -> squares" covers), 165 (moves-left's
  action in progress).

### 2026-10-17 - Network F6: Mish in every head's hidden layers (the user's ruling on issue 204)

- `python/t5d/net/heads.py`, `config.py`: register F6 as changed at CHECKPOINT 8 - "Mish for
  v1, in the trunk FFN and every head's hidden layers ...; squared ReLU as the trunk-only
  arm"; the user on spec issue 204: "yes: Mish in every head's hidden layers, the value head
  included; squared ReLU stays a trunk-only arm". The value head's hidden layer is the only
  head MLP (the policy and reply pointers are bilinear; the submit readout and the WDL,
  moves-left, short-value, effort, per-board and per-square heads linear): `ValueHead` now
  computes `out(mish(hidden(x)))` with no switch. `head_activation` (its placeholder "gelu"
  and the "mish" option) and `HEAD_ACTIVATIONS` are removed rather than kept as an arm no
  ruling or ablation names; nothing else needs a switch - 6y's squared ReLU stays
  `ffn_activation`'s, which only the trunk's FFN reads.
- No parameter changes (an activation has none): F1 stays 22,961,802. The value head's
  outputs change (GELU to Mish); no test pins them.
- Docstrings: `heads.py`, `config.py` (F6's bullet, the placeholder index, `ffn_activation`,
  `value_hidden`), `trunk.py` (only the trunk's FFN follows the switch), `embed.py` and
  `smolgen.py` (the Δt MLP's GELU and smolgen's SiLU, outside F6's words - spec issue 221).
- Tests: `test_trunk.py` - `test_every_heads_hidden_layer_is_mish` (the value head by hand in
  FP64 under both FFN arms, and not GELU) and `test_squared_relu_reaches_no_head` (the arm's
  network and the v1's, built from one seed, have the same parameters, and over one trunk
  output every head's output is equal bit for bit; the trunks differ); `head_activation` is
  refused as an unknown field. `test_smoke.py`'s squared-ReLU arm runs without it. Mutants
  patched in at test time (no file edited): a value head that follows the FFN's switch fails
  both new tests under the arm; a GELU value head fails the by-hand test.
- `docs/BUILD_LOG.md`: register F6; spec issue 204 marked built under its ruling; issue 221
  (the Δt MLP and smolgen's SiLU are outside F6's words, as A6's and A8's embeddings will
  be); proposed edit 154 settled upstream in part (the squared-ReLU arm's initialisation is
  still unstated), edit 162.
- Spec: §9.3's Mish bullet and register F6 - as built now; register D3 (the value head's
  MLP). §2.2, §2.3 and register A6, A8, B12: the activations F6 does not name (issue 221,
  proposed edit 162). §9.2 and F1: no parameter change.

### 2026-10-22 - Build log: the gate lane's step 1 done; issue 217 and two AGENTS.md lines wait on the user

- `docs/BUILD_LOG.md`, "Current state": the gate lane's step 1 is merged; "Waiting on
  the user" gains spec issue 217 (B17's class set) and the two lines the gate lane
  proposes for AGENTS.md (the engine baseline is never edited by hand; a LOSS is never
  recorded to make a gate run pass).
- Spec: none.

### 2026-10-22 - G3b-H47-CP8 review fixes: the baseline read verified before it is replaced; restricted runs skip the proof-channel readiness check; the ftxi question cites C7's sources

Thirteen review findings on the two entries below (none refuted).

- **`baseline --write` replaces only the pinned baseline** (`oracles/oracle_harness/gate.py`
  `write_baseline`, `baseline.py`). It read the file it replaces leniently
  (`read_previous`, removed), so a hand edit of the file could empty the `LOSS` list it
  prints. It now reads it with `baseline.load` against the `[baseline]` pin and, where the
  file is not the one pinned (edited, or gone), runs and writes nothing; the integrity and
  missing findings now say to restore the file (`git checkout -- oracles/engine-baseline.json`),
  never to record over it. `baseline.static_findings` also fails a baseline that holds no
  line for a record of its tier (`engine-baseline-stale`), so `verify` - not only
  `selftest` - catches a record added to the commit tier, or a line deleted. The docs
  (`baseline.py`, `integrity.py`, `gate.toml`'s comment, the README's integrity bullet and
  "The engine's baseline", `CONTRIBUTING.md`) now say that every read is verified and state
  the pin's limit as MANIFEST's: a file edited together with its pin passes it - review's
  job (H33). `CONTRIBUTING.md`'s "Changing a pin" (a new step 8) and "Adding game records"
  say to record the baseline again after a change to the commit tier's records;
  `test_repository`'s assertion names the command.
- **Restricted runs** (`gate.py`): a restricted run of the release tier no longer makes the
  proof-channel readiness check (`proof-channel-vacuous`, a check of the tier as a whole
  that only a complete run can act on; `verify` still prints it) - it failed every
  restricted release run on H48's 17 notes. A restricted report's `ok` is now null, like
  `oracle_layer_ok`, so that no script reads it as a tier pass; its `findings` say what it
  found. Tests: parse mode's floor with `floor=False`, a whole restricted parse-mode run
  of an engine that declines everything (no finding; the complete run fails on
  `engine-nothing-compared`), a restricted release run with no readiness section, the
  report's `ok`, and `baseline --write` refusing an edited or missing baseline.
- **Documents**: `oracles/README.md`'s smoke-test paragraph no longer says a commit-tier
  ruleset can meet the gating conditions with that tier's oracles alone. `docs/BUILD_LOG.md`:
  the H47 row (what changed for a restricted run; the `AGENTS.md` lines put to the user -
  never edit the baseline or its pin by hand, never record a `LOSS` to make a gate run
  pass); issue 126's ruling line in main's exact wording (a8a5920) and its built note;
  issues 132 (a) and 133; proposed edit 104 carries H33's double-edit limit to the
  baseline's pin.
- **The ftxi question** (`docs/checkpoints/upstream/ftxi-check-test-passed-boards.md`): the
  sources C7 cites added in the rulebook's rank order (wiki.gg's Exiling page, TV Tropes
  with C7's caveat, and the forum statements on boards ahead of the present - Asp,
  Patashu, Dark MAvL - and on inactive timelines - Perseus, Rayalot72); "92 positions in 12
  game records (91 distinct in 11 games: cwmtt/NP and NP0 are one game)"; the copy the
  attacker moves from is the next half-turn's board, distances counted from it; the
  captures named as those `check_position::for_phantom(...).checks()` lists (what the
  driver observed), not `phantom().find_checks()`; R1 quoted in full, no paraphrase in
  quotation marks, "cwmtt has no check verdict", and the inactive-timeline proviso for
  positions 3 and 5. `docs/checkpoints/upstream/README.md` and the entry below corrected
  to match.
- Checks (2026-10-22, MSYS2 g++; timings on a shared machine are not measurements): `py
  engine/build.py --werror` - nothing to rebuild; `harness.py verify` - every section ok,
  `engine baseline` among them, H48's 17 `proof-channel-vacuous` notes as before;
  `harness.py baseline` - the pinned baseline (ad689cf3be80), unchanged; `selftest` - 663
  tests OK, 1 skipped; `gate --tier commit` - ok, `SMOKE TEST - PASSED`, the baseline 41
  compared and 0 declined; `gate --tier nightly` - ok, `PASSED`, every source's required
  set reproduced, gating Standard - Half Reflected (2 oracles) and Standard - Reversed
  Royalty (3), as before; `gate --tier release --source notation --record notation/1.1
  --tracked-only` - `restricted - not a tier result: ... - 0 findings`, exit 0 (it exited
  1 on the 17 readiness findings before); `ruff check oracles` - clean.
- Spec: §11.0 "Tiered CI", "Recorded-answer integrity" and "A known-divergences file";
  register H47, H33, H27, H36 (R1, C7). Spec issues 126, 132, 133 (built notes corrected);
  proposed edit 104 (extended: the pin's limit). No golden file, budget, `min_checked`,
  margin, `proof_share`, waiver, adapter or driver changed; the baseline and its pin are
  unchanged.

### 2026-10-17 - Gate H27 (CHECKPOINT 8): the ftxi check-reading question drafted for upstream

The user's ruling on spec issue 126: "report it too, as a question citing the source and a
few representative positions, not as a bug: if ftxi's author can show the game behaves
ftxi's way, our rulebook is wrong and we want to know. Draft it in
docs/checkpoints/upstream/; I'll file it."

- `docs/checkpoints/upstream/ftxi-check-test-passed-boards.md` (new): a question to
  ftxi's author, in the form of the two bug-report drafts beside it. To tell checkmate from
  stalemate, `state::phantom()` passes every timeline whose last board is the player's to
  play; ruling R1 passes only the active boards in the present. The draft quotes R1, the
  code (two lines, with a permalink at the pin), the sources R1 rests on as the rulebook
  ranks them - the current in-game Guide (silent on which boards are passed), the older
  guide text and the developer's remark ("in the present"), Wikipedia's sentence R1
  adopts - and the evidence the other way (a client observation reported in the
  developer-quoting document, wiki.gg), and says which positions each reading puts in
  check. Five representative positions from public records, one from each of CHECKPOINT
  1's rule questions RQ1-RQ4 (`docs/checkpoints/checkpoint-1-check-clusters.md`: silly@28,
  Bg2@19, niceAndrey@57, manyChecks@20, standard@13): the record at its pinned commit, the
  `--at` turn, the boards the player must and may play, the capture ftxi finds and why R1
  does not count it. Each capture re-observed 2026-10-17 through the pinned library (the
  harness's ftxi driver, `observe --explain`: `has_phantom_check()` and the captures
  ftxi's royal-capture scanner lists on the phantom, `check_position::for_phantom(...)
  .checks()` - not `phantom().find_checks()`, as first written here; corrected in the
  review fixes above) and each position's timeline counts with the stock
  `5dtools stats`. `docs/checkpoints/upstream/README.md` lists it, with what to check
  before filing.
- `oracles/known-divergences.toml`: the 147 entries R1 settles (ftxi's and ftxi:naive's
  check verdicts, 92 positions in 12 records) now say `unreported: ... drafted as a
  question citing the source in docs/checkpoints/upstream/ftxi-check-test-passed-boards.md,
  to be filed by the project owner (CHECKPOINT 8)`; the header says what follows from the
  answer. No binding (oracle, pin, record, kind, index, values) and no citation changed.
- `oracles/README.md`, "Known divergences": defects reported as bugs and disagreements with
  a documented rule as questions (H27), and what is drafted. `docs/BUILD_LOG.md`: issue
  126's ruling and built notes, register row H27.
- Checks (2026-10-17): `py engine/build.py --werror` - nothing to rebuild; `harness.py verify` - every
  section ok, H48's 17 `proof-channel-vacuous` notes as before; `selftest` - 660 tests OK, 1
  skipped; `gate --tier commit` - ok, `SMOKE TEST - PASSED`, the baseline 41 compared and 0
  declined; `gate --tier nightly` - ok, `PASSED`, gating Standard - Half Reflected and Standard
  - Reversed Royalty, as before; `gate --tier nightly --source ftxi --record cwmtt/silly` -
  `restricted - not a tier result: ... - 0 findings`, exit 0.
- Spec: §11.0 "A known-divergences file" ("reported upstream - defects as bug reports,
  disagreements with a documented rule as questions citing the source ...; the owner files
  them"); register H27, H36 (R1). Spec issue 126 (ruled at CHECKPOINT 8; built); proposed
  edit 99 (b) applies - no edit until the question is filed.

### 2026-10-17 - Gate H47 (CHECKPOINT 8): two oracles to gate a ruleset; the commit tier's last-green-run rule; restricted runs never a tier verdict

The user's CHECKPOINT 8 rulings on spec issues 122-124 (register H47 as amended).

- **Two oracles to gate a ruleset** (issue 122: "a ruleset is gated only when two or more
  oracles answer for it. A single oracle cannot tell its own bug from ours").
  `oracles/oracle_harness/gate.py`: `engine_differential` gates a ruleset only if each of
  its records was followed, as far as the engine went, by at least `GATING_ORACLES` (2)
  independent implementations - the count `Gated.oracles` already gave; two searches of
  one implementation count once, the notation parser never. A record one oracle follows
  ungates its ruleset with its own reason, naming the record and the oracle. In every
  tier; the commit tier, with one rules oracle, can no longer meet the conditions at all.
  Nightly, run with the engine: PASSED, gating Standard - Half Reflected (two oracles behind its
  record) and Standard - Reversed Royalty (three) - and no longer Standard - Turn Zero, whose
  cwmtt/tesseractMageOChicken only cwmtt follows as far as the engine goes.
- **The commit tier's last-green-run rule** (issue 123: "the commit tier fails if the
  engine declines a record it read on the last green run, or reads none").
  `oracles/oracle_harness/baseline.py` (new): the engine's baseline,
  `oracles/engine-baseline.json` (new) - for each commit-tier record, how the engine's
  replay ended and how many positions it reported on a green run, one record per line.
  Kept as H33 keeps recorded answers: written only by the new `harness.py baseline
  --write`, which runs the `[baseline]` tier in full and records only from a run green
  on everything else, then lists what the new file no longer holds (`LOSS`) first; never
  by a gate run or the engine; pinned by its SHA-256 in `gate.toml` `[baseline]` (new) -
  beside the rulebook's pin, not in `golden/MANIFEST`, which every `golden --write`
  rewrites - and read once per run, verified (`snapshot.take`), and checked by `verify`
  (`engine-baseline-missing`, `-integrity`, `-invalid`, `-stale`); `harness.py baseline`
  checks it alone. Every tier fails an engine that reads less of a record than then -
  fewer positions, or as many without the verdict it ended with (`engine-declined`) -
  or replays none of the tier's records (`engine-read-nothing`); the summary and report
  say what was compared. Recorded from a green commit run of the engine as built on this
  branch: 35 of the 41 records replayed. `paths.ENGINE_BASELINE_FILE`;
  `integrity.harness_files` hashes it wherever a test points it; `config.Baseline`, and
  `config.write_sha256`, now also the rulebook's pin writer (`rulebook.pin`, unchanged
  in behaviour).
- **Restricted runs never a tier verdict** (issue 124: "A restricted run (--source/--record)
  never produces a tier verdict; label its output 'restricted - not a tier result'").
  `gate.py`: `GateResult.restricted_to` replaces `partial`; any `--source` or `--record`
  restricts. Such a run prints and reports `restricted - not a tier result`, no `oracle
  layer [tier]` line, no ruleset gated or not, no smoke-test line, a proof channel
  `DISABLED - restricted - not a tier result`; it reports every finding and its exit code
  reflects them only - none of the checks of a tier as a whole (`engine-nothing-gated`,
  `engine-read-nothing`, `engine-nothing-compared`, `require_engine`). Its report is
  `gate-<tier>-restricted.json`. This undoes the reading built for issue 124 (a
  restricted run kept its tier's rule).
- Tests: `oracles/tests/test_baseline.py` (new: the readings of "declines" and "reads
  none", the comparison, the file, its pin and its static check), and in `test_gate.py`
  the two-oracle rule (one oracle, two searches of one, a parse-only source, a weakest
  record), the baseline in whole gate runs (a decline in every tier, reading none, an
  engine that edits its baseline while it runs, a baseline edited by hand, missing or
  stale, `baseline --write` from green and non-green runs), restricted runs (result,
  summary, report); `test_config.py` (the `[baseline]` table; a pin written into its own
  table, either line ending); `test_integrity.py`; `test_repository.py` (the repository's
  baseline pinned, current and complete); `test_gate_live.py` - its runs are restricted
  runs, holding the stand-in to a baseline of its own honest run. 660 self-tests, from
  638: `test_baseline.py`'s 13, 6 more in `test_gate.py` (99 to 105), 2 in
  `test_config.py`, 1 in `test_repository.py`.
- Documents: `oracles/README.md` (the tier table, the engine contract's conditions, the
  smoke test, a new "The engine's baseline", recorded-answer integrity, "Other
  commands"); `oracles/gate.toml`'s comments; `CONTRIBUTING.md`; `docs/BUILD_LOG.md`
  (register rows H23, H27, H47; issues 122-124's built notes; issues 131-133; edits
  103-104).
- Checks (2026-10-17, MSYS2 g++ 16.2; timings on a shared machine are not
  measurements): `py engine/build.py --werror` - nothing to rebuild (the engine is
  unchanged on this branch); `harness.py verify` - every section ok, `engine baseline`
  among them, with H48's 17 `proof-channel-vacuous` notes as before; `harness.py
  baseline` - the pinned baseline (ad689cf3be80); `selftest` - 660 tests OK, 1 skipped
  (symlinks cannot be created by this Windows profile); `gate --tier commit` - ok, `SMOKE
  TEST - PASSED`, ftxi 8256 of 8256 and notation 3012 of 3012 required values
  reproduced, the baseline 41 records compared and 0 declined, Standard - Reversed Royalty
  `NOT gated` with the one-oracle reason (ftxi alone); `gate --tier nightly` - ok,
  `PASSED`, every source's required set reproduced, gating the two rulesets above; `gate
  --tier nightly --source ftxi --record cwmtt/standard` - `restricted - not a tier result:
  ... - 0 findings`, the proof channel `DISABLED - restricted - not a tier result`, exit 0,
  its report `gate-nightly-restricted.json`; `ruff check oracles` - clean.
- Spec: §11.0 "Tiered CI" and "Recorded-answer integrity"; register H47 (built), H33,
  H27, H23. Spec issues 122-124 (ruled at CHECKPOINT 8; built), 131 (the two-oracle
  rule per record), 132 (the baseline: what it records, "declines", "reads none", every
  tier), 133 (restricted runs); proposed edits 103 (§11.0 "Tiered CI", H47 as built) and
  104 (§11.0 "Recorded-answer integrity", H33: the baseline as a recorded answer).

### 2026-10-17 - Build log: the current state after the evening's merges

- `docs/BUILD_LOG.md`, "Current state (2026-10-17)": what is on main now - H47, F5/F6,
  M22, B15 with B14's score function, A6 with Guard's line, the gate with the engine
  configured and H51, B11/B16/B17 - what nightly gates, F1's count, and that GitHub
  Actions starts no job (the account's billing), so every check runs locally; what
  each lane is doing and does next.
- Spec: none.

### 2026-10-17 - N-B11-B16-B17 review fixes: B16 below a refreshed anchor tested against an oracle of its own; B17 through refreshed epochs; a dense network's block-causal forward without B16 refused; issue 217's second reading

- `tests/net/test_cache.py`: `test_status_biases_below_a_refreshed_anchor` (a root and a node
  refresh). The cached path and `reference_output` share one at-anchor rule - the anchor's own
  evaluation is the refresh pass and adds no B16, every node below it carries B16 on every
  frozen key, history boards included (issue 216) - and every refreshed-epoch row compared the
  one with the other, so B16 dropped at every node of a refreshed epoch, in both, passed the
  whole suite. The new test's oracle does not share the rule: B16's table changed between two
  fresh searches leaves the anchor's evaluation bit for bit and moves every node below it
  (0.33-0.46 in the trunk), in both paths; changing only the statuses that the history boards
  alone hold still moves every node (0.15-0.33) - B16 reaches the history keys. `cache.py`'s
  docstring points to it. Mutants patched in at test time (no file edited), over `test_cache`,
  `test_d8`, `test_complexity`, `test_status` and `test_shaw`: B16 dropped at every node of a
  refreshed epoch, in both paths, fails these two cases and nothing else (it failed nothing
  before); B16 at the anchor fails them too - 16 tests with the cached path alone carrying it
  (10 before), 9 with both paths (5 before; the other two new ones are B17's root-refresh
  cases below, through their dense-model check).
- `tests/net/test_shaw.py`: `test_the_cached_path_through_a_refreshed_epoch` - §15.2 row 1 with
  B17 on through root and node refreshes, supersession inside the refreshed subtree (§15.2's
  adversarial construction), both C20 readings, the vectors and B16's table at unit scale; with a
  root refresh the root's evaluation against the dense model. No B17 case ran a refresh before:
  a refresh pass computed without B17's term passed the suite, and fails all four cases now.
- `python/t5d/net/model.py`: `make_context` refuses a block-causal forward of a network of the
  dense scope while its `status_key_bias` is on with intrinsic frozen features (unless
  `status_bias=False`). B16's table is built by the configured scope (`status_bias_built`), a
  frozen board's Δt-to-present and active flag are stripped by the forward's: so
  `model(batch, scope="block_causal")` or `NetworkEvaluator(v1, scope="block_causal")` on the
  v1 dropped the frozen boards' status with nothing in its place, silently - §4.1's "current
  status enters as score-time key biases" (A3: "instead (B16)") broken for that forward (the
  cached path never reached it: `CachedEvaluator` refuses a dense-scope network). The forms that say what they
  compute stay allowed: `status_bias=False`, `status_key_bias=False`,
  `frozen_features="current"`. Docstrings: the model's, `status.py`'s, `config.py`'s
  `status_key_bias`, `NetworkEvaluator`'s `scope`. `tests/net/test_status.py`: the refusal through the model, its
  attention probabilities and the dense path's evaluator, and the allowed forms;
  `test_where_there_is_none` asks the dense network's block-causal context with
  `status_bias=False`.
- Spec issue 217 rewritten, with proposed edit 161, the B17 row, `shaw.py`, `config.py`'s
  `shaw_near` and `test_shaw.py`'s docstring: the built reading of "the exact near classes" -
  the file-rank offsets within ±2 in every one of B15's 45 exact board-distance classes - is one
  of two that give 1,493 classes, so the same parameter count and both of §4.1's cost figures.
  The other, the nearest board-distance classes with their full 15 x 15 slices (say the same
  board and the four one timeline or one turn away), covers every relation on one board - as
  built a rook, bishop or queen relation beyond two squares on one board gets no term - and none
  two boards away. Both put to the user; the issue said "near" must cut the file-rank offset,
  which does not follow. The second reading is not built (it would be another
  `shaw.compact_classes`).
- Records corrected. The B17 row's unescaped pipes (`|Δfile|, |Δrank|`) made 7 cells of a
  3-column row, and GitHub's renderer dropped everything after "the exact near classes -":
  reworded without pipes; C6's `|E|` escaped in passing (its cell rendered as "The"); every
  table row of the log now has its header's cell count. "Every `test_cache` case runs with B16
  on" (the B16 row, the B16 entry below, `test_cache.py`'s and `test_status.py`'s docstrings):
  every case with intrinsic frozen features - the `frozen_features="current"` test arm has none.
  "4,864 per layer" (issue 214, `config.py`, `status.py`, `test_params.py`, `test_status.py`,
  the B16 entry): 4,864 with one table per layer, 16 x 304 - 304 per layer. `config.py`'s index
  of committed-class placeholders names B16's four and B17's three. The B11 row and entry add
  `game.py`'s `relation_bits` docstring to the engine lane's files that still name travel-threat
  among the network's relation ids. The B16 and B17 entries' mutation checks were temporary
  edits of the worktree's own files, restored afterwards, not "scratch copies"; and B16 at the
  anchor fails the refresh-fork rows only when the cached path alone carries it.
- `docs/BUILD_LOG.md`: register rows B11, B16, B17, C2 and C6; spec issues 214 and 217;
  proposed edit 161. F1's count, A1-A3 and M20 are unchanged.
- Spec: §4.1 (B16's "current status enters as score-time key biases"; B17's compact class
  set), §7.9 (the anchor's evaluation is the refresh pass), §15.2 (row 1 with B16 and B17
  in refreshed epochs; "supersession inside a refreshed subtree"); register B16, B17, B11, A3,
  C2, C3, M20. Proposed edit 161 rewritten with issue 217's two readings. No new issue or edit:
  the refusal guards a function no configuration defines and decides nothing the documents
  leave open.

### 2026-10-17 - Network B17: the key-side Shaw term behind its flag

- `python/t5d/net/shaw.py` (new): register B17 (prototype, class C, ablation 2; §4.1), behind
  `shaw_key_term` (off; it needs B15: "on top of B15"). Each query - the unrotated one - dotted
  with a learned vector for its pair's class in a compact set, scaled by 1/sqrt(d_head), added
  to the score beside B15's scalar: Shaw et al.'s key-side relative term. Value-side vectors are
  held open (§14.4) and not built.
- Its unstated details are named placeholders, each a spec issue: the compact set - "the exact
  near classes plus the line classes" - read as B15's exact board-distance classes' cells with
  |Δfile|, |Δrank| <= `shaw_near` = 2, plus B15's line classes: 1,493 classes, which gives both
  of §4.1's cost figures (issue 217 - one of two readings that do, as the review fixes above
  record it); a vector per class per head per layer, 6,115,328 parameters
  (26.6% of the v1; `shaw_shared_heads`: 764,416, as Shaw et al. share them) - no document states
  the count (issue 218); the unrotated query (forced: a rotated one would be an absolute
  position, §14.1) and the scale (issue 219); N(0, 0.1) from a generator of its own,
  `model.b17_seed` (issue 220). Pairs with a register get no term.
- Section 15.2's B17 row, both computations built: the gather - precomputed query-class
  products P = q R^T, then each pair's entry gathered by its class (`gather_term`) - in every
  CPU path, block by block, over any key selection or row set (`attention.py`: `AttnContext.shaw`,
  beside the structure, since it needs each block's queries), in the cached path for a frozen
  board's rows from its own queries (`cache.py`), and in B14's score function, which reads P
  inside the attention with the class computed there from B15's entry (`flex.py`); and the
  direct per-pair form (`direct_term`, head by head over row chunks), which the dense reference
  materialises (`AttnContext.densified`). Classes come from B15's per-batch index through a
  static map (`displacement.lookup`, now shared with B15's own rows).
- `cost.py`: `qk_flops` and `shaw_flops` - section 15.3's B17 row is C/T of Q K^T: 0.511 at 44
  boards and 0.094 at 240 with 1,493 classes (§15.3: "~0.5x ... ~9%"). `bench.py`: a `b17-shaw`
  configuration for the quiet-machine block's wall times (none taken: the machine is loaded).
  `config.py`, `model.py` (`attn.shaw` in `parameter_breakdown`): the flag, its placeholders and
  validation, its generator.
- Tests: `tests/net/test_shaw.py` (new) - the class set written out from B15's layout (three
  settings, displacements one by one); section 15.2's row - the gather and the direct form equal
  the per-pair computation written out (FP64 to 1e-12, FP32 to roundoff) over every key, blocks,
  key selections, row sets and head subsets, on all twelve of §15.2's constructions, three
  seeds, initial and unit scale, shared and per head; the CPU paths' outputs and gradients (the
  vectors' included) against attention with the direct term materialised; B14's score function
  and attention through it; the whole network one function across all six implementations;
  cached = one full pass (block-causal, B16 on; through refreshed epochs only since the review
  fixes above); translation, raw-coordinate translation and
  board order with it on, and a rotated query shown absolute; the count, the generator (off is
  the B15 network bit for bit, zero vectors give B15's outputs), every layer's reach; the cost
  row, analytic and counted (the counted FLOPs exceed B17-off's by exactly the products).
  Existing suites carry it: `test_cache` (two every-switch cases), `test_structure` (three
  switches), `test_invariance`, `test_n2_model`. Checked by temporary mutants of the worktree's
  own files, restored afterwards: leaving it out of the block-sparse selection or of a frozen
  board's rows fails.
- `docs/BUILD_LOG.md`: register rows B17 (built behind its flag, tested), B14, C2 and F1 (the
  v1 unchanged; 29,077,130 with B17 on); spec issues 217-220; proposed edit 161.
- Spec: §4.1 ("Not Lc0's BT5 encoding"), §14.1, §14.4 (value-side vectors held open), §15.2
  (the B17 row), §15.3 (the B17 row: the analytic ratio and the counted products; the CPU wall
  time not measured - the quiet-machine block), §13.1 ablation 2; register B17, B15, B14, F1.
  Proposed edit 161 (issues 217-220).

### 2026-10-17 - Network B16: score-time status key biases on frozen boards

- `python/t5d/net/status.py` (new): register B16 (committed with the cached configuration,
  class C, gate 6c; §4.1, §7.3), with A3 and M20 ("frozen features -> intrinsic, with
  score-time status biases (B16)"; spec issue 31). Under the block-causal scope with intrinsic
  frozen features, a learned bias per head by the key board's status - the bin of its
  Δt-to-present and its timeline's active flag - on the scores of live query rows (head boards'
  tokens and the global registers) against the tokens of frozen boards, squares and per-board
  registers alike, recomputed from the evaluated position's own present at every evaluation.
  Never on a frozen row: the frozen board's outputs and K/V would then depend on the present
  (C17) - a forced reading. Never under the dense scope (the v1; every refresh pass).
- Its unstated details are named placeholders, each a spec issue: the bins (`status_dt_exact
  = 4`, `status_dt_max = 128`: 19 signed log2 bins, the time buckets B6 used and smolgen keeps;
  "timeline status" read as the active flag - issue 213), one table shared by the layers
  (`status_bias_per_layer`, issue 214), N(0, 0.1) from a generator of its own (`status_bias_init_std`,
  `model.b16_seed`, issue 215). `status_key_bias` (on) switches it; off is the network without
  it, bit for bit in every other weight.
- The refresh stays as built (M20; the main loop's reading): a refreshed epoch's history boards
  keep the anchor's A3 features in their K/V, and B16 biases them at every node below the
  anchor. At the anchor itself the evaluation is the refresh pass (§7.9: "at the root the
  two-level mask reduces to full dense attention"), so B16 adds nothing there:
  `T5DNet.make_context`/`forward(status_bias=False)`, which `cache.py` passes for the anchor's
  own state - in the cached path and in `reference_output` (issue 216, with §4.1's categorical
  "frozen boards therefore carry only intrinsic features", logged as the briefs' critic asked).
- `config.py`: `frozen_features` documented as ruled (M20) - "current" kept only as a test arm,
  with no B16 under it; B16's fields, validation and `status_bias_built` (the table exists only
  in a network of the cached configuration: 304 parameters there, none in the v1, whose count
  stays 22,961,802). `structure.py`: B16's rows (`status_rows`) in every bias row the CPU paths
  build - any block, key selection or row set - and `for_frozen_rows` (no B12, no B16) for a
  frozen board's rows in `cache.py`. `flex.py`: B16 inside B14's score function, from a
  per-token index. `model.py`: the table, its generator, `parameter_breakdown`'s
  `attn.status`. Docstrings: `embed.py`, `scope.py`, `cache.py`, `__init__.py`.
- Tests: `tests/net/test_status.py` (new) - the bins written out; the rows against §4.1 from
  the positions (every bin, past and future, active and inactive timelines, §15.2's
  constructions, four placeholder settings, any block, key selection and row set), bit for
  bit, alone and added to B15's; none outside the cached configuration, under a dense or
  status-free forward; a frozen board's status change moves the live outputs and leaves every
  frozen output and K/V bit for bit; B14's score function bitwise and attention through it;
  the CPU paths' outputs and gradients, the table's included; every implementation one
  function; the table's size, generator, reach, placeholders (each regroups the statuses) and
  validation; translation and board order. `test_cache.py`: every case with intrinsic frozen
  features runs with B16 on (the default; the `frozen_features="current"` test arm has none),
  and `test_status_biases_keep_frozen_boards_intrinsic` - a move that shifts the
  present and a branch that activates a timeline: every frozen entry kept, only the superseded
  board computed, cached = the full pass, the statuses moved. `test_scope.py`: live rows carry
  B16 besides the structure. `test_structure.py`: three new switches. `test_params.py`: 304 in
  the cached configuration (4,864 with one table per layer, 16 x 304), none in the v1. Checked
  by temporary mutants of the worktree's own files, restored afterwards: B16 on frozen rows
  fails the definition and cached = dense; B16 at the anchor fails the root-refresh and
  refresh-fork rows - in the cached path alone. (The review fixes above re-ran the anchor
  mutants on the suite before their new tests - B17's commit, fe99a55 - patched in at test
  time, over `test_cache`, `test_d8` and `test_complexity`: in the cached path alone B16 at the anchor fails those rows and every
  other that compares a refreshed anchor's evaluation with the reference or the dense model -
  the 6y, BF16, D8 and C24 cases, 10 in all; in both paths it fails the root-refresh, BF16 and
  6y rows, 5, and not the refresh-fork rows, which compare the two paths; and B16 dropped at
  every node of a refreshed epoch, in both paths, failed nothing - 134 passed - the gap the
  review fixes' new test closes.)
- `docs/BUILD_LOG.md`: register rows B16 (built, tested), A1-A3, B14, C2, F1 and M20; spec
  issue 31 annotated; spec issues 213-216; proposed edit 160.
- Spec: §2.3, §4.1 ("Relative to the present, at score time"), §7.3, §7.9 (the anchor's
  evaluation), §15.2 (row 1 and the purity rows with B16 on); register A3, B16, C2, C3, C17, M20.
  Proposed edit 160 (issues 213-216). No §15.3 figure is measured (B16 is a lookup per
  live-row x frozen-key pair, inside "Cached cost per node").

### 2026-10-17 - Network B11: travel-threat folded into attack

- `python/t5d/net/config.py`, `state.py`: register B11 (provisional, class R, ablation 2) as
  CHECKPOINT 6 rules it - "attack, defence, pin, x-ray, check - travel-threat folded into
  attack" (reversal 26; §6: a travel threat is a cross-board attack, which the attack type
  already carries). `TYPED_RELATIONS` loses `travel_threat` (five types) and `RELATION_TYPES`
  with it (relation ids 0-5, the C ABI's). The ABI's id 6 (`T5D_RELATION_TRAVEL_THREAT`) stays
  reserved - the ABI is append-only, and the engine never emits it - and `PositionState`'s
  validation now refuses it. Pin and check stay in the type set, as the register lists them:
  the engine builds them in its ABI round 1, so no position from the engine carries them yet.
- `synthetic.py`: the generator no longer emits a travel-threat edge beside each cross-board
  attack - the attack edge is the travel threat. Every other edge and the candidate moves are
  unchanged (the dropped edges drew nothing from the random stream).
- B11's table: 8 heads x 5 types = 40 parameters (80 directed), where it had 48 (96), and only
  with `typed_edge_bias` on - off by default, so the v1's 22,961,802 is unchanged. With B11 and
  B12 both on, smolgen (drawn after B11 from the same stream) starts from other values; no
  output is pinned anywhere.
- Docstrings: `structure.py` (the five types; its note that the check relation waits on the
  check ruling - stale since CHECKPOINT 1's rulings - replaced by what the engine builds),
  `heads.py`.
- Tests: `test_structure.py` - the type set and its ids, id 6 refused, the table's shape; the
  per-type test moves pin and check (it moved check and travel-threat); `test_state.py` - id 6
  refused; `test_reach_graph.py` - cross-board attacks exist, and no relation is
  travel-threat's; `test_n2_model.py` - 40 and 80 scalars.
- Not changed: the engine and its Python ABI mirror, which are the engine lane's. The
  relation-id comments of `records.h` and `capi.py`, `engine/README.md`'s (e),
  `convert.py`'s docstring and `game.py`'s `relation_bits` docstring ("names the network's
  contract has but the engine does not build (pin, check, travel_threat)"; added to this list
  by the review fixes above) still name travel-threat among the network's relation ids; the
  B11 row records it for engine ABI round 1, which builds pin and check there.
- `docs/BUILD_LOG.md`: register rows B11 and F1 (unchanged count); spec issues 16 ((d): five
  types) and 19 (applied) annotated.
- Spec: §6 ("Typed edge biases"), register B11 and reversal 26 - the change applies them as
  written; no edit is proposed. No part of the spec states B11's parameter count.
### 2026-10-17 - H51: capture markers read leniently - the reader in both directions, the writer strict, rulebook R11; ftxi's refusal recorded, KD-0167-0169 retired

- `engine/include/t5d/pgn.hpp`, `engine/src/pgn.cpp`: the user's ruling on spec issue 125
  (CHECKPOINT 8; register H51, spec §11.0 "Capture markers are read leniently"). The reader
  refused a written `x` that takes nothing and read a capture written without one (the bool
  `PgnOptions::require_capture_marks`, false by default, made only that second reading
  strict). It now has `PgnOptions::capture_marks`, `CaptureMarks::Lenient` by default - and
  the bool is gone: a move is read as its
  squares and boards define it, whether or not `x` is written, and the mark selects no move
  (a move that fits two pseudo-legal moves stays ambiguous whatever its mark says); a mark
  that disagrees in either direction - `x` on a move that takes nothing, a capture written
  without `x` - goes into the reader's warning channel, `PgnReplay::warnings()`, naming the
  action, the move as written and where. `t5d-gate` prints it on stderr, as it prints the
  annotation and turn-number warnings: the engine contract has no warning field, and its
  required fields are unchanged. `CaptureMarks::MarkedMustCapture` keeps the reading before
  H51 and `Strict` the letter both ways (a diagnostic), by option only;
  `capture_marks_name`.
- `engine/src/gate.cpp`, `engine/include/t5d/gate.hpp`: `--pgn
  capture_marks=lenient|marked_must_capture|strict`; the option's spelling before H51,
  `require_capture_marks=true|false`, keeps its meaning (strict / marked_must_capture).
- The pinned measurements (`engine/tools/tool_support.*`, `sixb_roots.*`, `t5d_scan6b.cpp`,
  `t5d_scan6bp.cpp`): `sixb_pgn_options()`, the reading 6b, 6b′, 6b″ and Hunt's ground truth
  were measured under - the reader's defaults of their day, which differ from today's only
  in `MarkedMustCapture`. `t5d-scan6b`, `t5d-scan6bp` and `sixb_roots` (`t5d-scan6bpp`,
  `t5d-hunt-truth`) read with it: read leniently, cwmtt/futures goes on past its action 17
  and 6b's construction would take three roots from it rather than from ftxi/futures,
  renumbering the roots from 169 and making root 168's next action readable - the truth's
  readers would refuse it. Checked against the parent commit's build: 6b″'s tool accepts the
  pinned truth (825 roots by record, index and fingerprint), and 6b's and 6b′'s tools list
  the same 825 positions. The truth was not re-recorded and 6b″ not re-run. The reader is
  among `T5D_TRUTH_SOURCES`, so the code signature changed (the truth's provenance records
  none: "unrecorded"); `engine/README.md` says so ("Hunt's ground truth", *After H51*).
- Tests (551 engine tests, from 544): `test_pgn_replay.cpp` - the lenient reading both
  ways, with its warnings, on physical moves, en passant and both kinds of jump (the
  corpus case's shape, `>>x ... ~`); the mark selecting no move (an `x` that would have
  told two moves apart under `infer_omitted_boards`); the two other readings.
  `test_gate.cpp` - the adapter reads `1. Nxf3 / Nc6` to its end, its events those of
  `1. Nf3 / Nc6` in both modes and the warning on the diagnostic channel; the options.
  `test_synthetic.cpp` - **the engine's writer** (`synthetic::pgn_move`, which t5d-play's
  records and the synthetic records use) writes `x` exactly on captures: every move offered
  at every position of 53 random games - 9 variants under the default rules, every
  alternative rule option in Two Timelines - is written and judged from the boards, never
  from the move's flags (the opponent's pieces on the target board against those on the
  board the move makes there): 32,566 captures (14,081 physical, 24 of them en passant;
  6,293 by non-branching jumps; 12,192 by branches) and 408,383 quiet moves of the same
  kinds, each kind required; the records replay with no warning under the default reading and to their
  end under the strict one. `test_selfplay.cpp` - t5d-play's records likewise.
  `test_sixbpp.cpp` - 6b's roots read as 6b read them: by `sixb_roots`, and (a review fix:
  no test ran either tool, which read their records in loops of their own) by `t5d-scan6b`
  and `t5d-scan6bp` themselves, run on a record with an `x` on a jump that takes nothing -
  each lists `sixb_roots`' three roots, index and fingerprint, and stops where it stops; with
  either loop reading with the default, the test fails (checked by a mutation, reverted).
  `engine/CMakeLists.txt` passes the tools' paths (`T5D_SCAN6B_EXE`, `T5D_SCAN6BP_EXE`).
- **N5's writer** (a review fix; `python/t5d/engine/netplay.py`,
  `python/tests/engine/test_netplay.py`): the network's self-play and match records are
  5dpgn too, written by `netplay.pgn_move` - left out of the writer survey above (the build
  log had said "the engine writes 5dpgn nowhere else"). It writes `x` from the move's capture
  flag, and H51 had taken away the only check its records had: before H51 the gate refused a
  record with `x` on a move that takes nothing; since, it reads it, with a warning on stderr
  that `replay_with_gate` never read (`Game.from_pgn` reads with the defaults and has no
  warning channel). Now `Replay.warnings` keeps the gate's reader warnings and `check_replay`
  reports each - a record N5 writes must replay with none, as t5d-play's must.
  `test_the_writer_marks_exactly_the_captures` judges `netplay.pgn_move` from the boards, as
  the C++ test judges its writer: every move offered at every position of 10 random
  pseudo-legal games under 5 variants (optional moves included) and at two en passant
  positions - 420 captures on their boards, 3 en passant, 262 by non-branching jumps, 386 by
  branches and 12,831 quiet moves of those kinds, each kind required;
  `test_a_capture_mark_that_disagrees_is_a_problem` - an `x` added to a quiet move, or
  dropped from a capture, replays `ok` with one warning, which `check_replay` reports. The
  reviewers' slip (the capture mask also matching double steps) now fails the replay of
  `test_a_game_replays_legal`'s game (`(0T3)Pf2xf4` reported), and the board-judged test
  fails under it, a dropped en passant mark and a dropped branch-capture mark.
- `docs/rules/README.md`: R11, capture marks read leniently - labelled a project decision
  (H51, CHECKPOINT 8), in the file's style, with its source ranking (R3) and the notation
  specification cited for notation only; a conventions line; source [141]. Its diff
  reviewed (no other entry changed) and re-pinned (`rulebook --pin`, `oracles/gate.toml`:
  `08fddc5e...` to `39e9ad58...`). In review, R11's source sentence was narrowed to what the
  specification says: it called "purely for better human comprehension" the present-movement
  and created-timeline tokens (`~`, `(>Lx)`, `(~Tx)` - T15's scope), not every token a writer
  adds, as R11 had it; the ruling unchanged, KD-0170's citation unaffected; reviewed and
  re-pinned (`39e9ad58...` to `ab01119e...`).
- `oracles/known-divergences.toml`: KD-0167 to KD-0169 withdrawn, on the ruling's authority
  (with the reader lenient they were telltales - the engine plays action 17 as cwmtt,
  Chess in 5D and the notation parser do - as their header had said); **KD-0170**: ftxi's
  refusal of that action (`rejected:no_matching_move`), citing R11 and H51 - triaged by the
  rulebook, not by vote: ftxi's own legal-action listing at position 16 holds the queen's
  jump, and its refusal parses the move with the capture flag set (`cp:yes`), so it refuses
  the mark, not the move; unreported upstream (spec issue 130). The header's paragraphs
  updated (a soft refusal may cite the rulebook's project decision where the specification
  is silent). `oracles/README.md` likewise.
- Gate (this machine shared with other lanes; timings not measurements): `verify` ok (H48's
  17 notes); `selftest` 638 tests OK, 1 skipped; `gate --tier commit`: SMOKE TEST - PASSED
  (before the entries changed it failed, as expected: KD-0167 to KD-0169 as telltales,
  ftxi's refusal unexplained); `gate --tier nightly`: every source's required set
  reproduced, `engine differential: PASSED` with no finding, gating Standard - Half
  Reflected (at least 2 oracles), Standard - Reversed Royalty (at least 3) and - newly, the
  engine now reading cwmtt/futures to its end - **Standard - Turn Zero** (7 records, 144
  positions, at least 1 oracle: cwmtt/futures has 2 behind it, cwmtt and Chess in 5D, which
  agree with the engine wherever each answers at its new positions 17 to 19 - checkmate at
  19 is the engine's call and Chess in 5D's (cwmtt reports no legal action there,
  `no_action`, which does not say checkmate or stalemate); but cwmtt/tesseractMageOChicken
  has 1, cwmtt, so the two-oracle rule of spec issue 122 - the next component - would not
  gate Turn Zero). The proof channel stays disabled. The Python engine-binding tests
  (`T5D_REQUIRE_ENGINE=1 pytest tests/engine`): 164 passed, 1 skipped.
  After the review fixes (the same machine, as loaded): `py engine/build.py --werror
  --test` 551 of 551; `rulebook` ok after the re-pin; `verify` ok (H48's 17 notes);
  `selftest` 638 tests OK, 1 skipped; `gate --tier commit`: oracle layer ok, `engine
  differential: SMOKE TEST - PASSED`, calibration 1.23x; `gate --tier nightly`: every
  source's required set reproduced, `engine differential: PASSED` with no finding, gating
  Standard - Half Reflected (at least 2 oracles), Standard - Reversed Royalty (at least 3)
  and Standard - Turn Zero (at least 1 - the same oracles behind each record as before,
  so under the two-oracle rule Turn Zero alone would drop out), calibration 1.22x; the
  Python engine-binding tests 166 passed, 1 skipped; ruff clean.
- t0database (M7, M22): converted (`py data/t0database/to_5dpgn.py build/t0db`, 761
  records) and measured with the parent commit's `t5d-measure` and this one's: M22's
  `--turns --turn-summary` and M7's `--moves-after-shift --summary`, with and without
  `--legal`, are byte-identical - no record and no action changes (716 replay to the end, as
  before). No t0database record writes `x` on a move that takes nothing; 661 of them write
  captures without `x` (8,034 moves), which the reader now reads with a warning, as before
  but for the warning. `docs/MEASUREMENTS.md` unchanged. Re-run after the review fixes: this
  build's `t5d-measure` and the parent commit's (cb7af77) byte-identical again, standard
  output and error, for `--turns --turn-summary` and `--moves-after-shift --summary`, each
  with and without `--legal` (716 of 761 records to their end in each).
- `engine/README.md` (the reader's readings, the adapter's warnings, the corpus note, 6b's
  roots, *After H51*), `docs/BUILD_LOG.md` (step 2; register rows H51 - new - H28, H31, H34,
  H38; spec issue 125 settled; issues 129 and 130; proposed edits 101 and 102).
- Review fixes in the documents: the reader's warnings - `PgnReplay::warnings()` and
  `engine/README.md` said each names the action and the move; a turn number's gives only its
  line and column (the comment in `pgn.hpp`, a truth source, so the code signature moved
  again: sha256:3bc1d730..., the truth's still unrecorded). The build log: the mate at
  cwmtt/futures 19 is Chess in 5D's call and the engine's - cwmtt reports `no_action`, which
  does not say which (step 2 and H31 had "as both say"); H51's writers; spec issue 127's
  measurement superseded (a two-oracle rule would now ungate Turn Zero) and H23's row stale
  on what nightly gates (left to the issue-122 component, which changes the gating); CI
  started no run on `lane/gate` from a5e61d9 on - account billing - so H51 is unconfirmed on
  Linux and the nightly dispatch still owed; `CLAUDE.md`'s "rulings R1-R10" (R1-R11 now) put
  to the user, the instruction files being theirs to change (H38's row).
- Spec: §11.0 ("Running the gate": capture markers read leniently - built; "A
  known-divergences file"), register H51 (built), H28, H31, H34, H38, H27, and the pinned
  measurements' registers M8, M12, M15, M17, H43 (their roots keep the reading they were
  measured under - spec issue 129). Spec issue 125 settled upstream; new issues 129 and 130;
  proposed edits 101 (H51 and §11.0: the pinned measurements' reading, the KD numbers) and
  102 (§11.0 and H27: divergences settled by a project decision recorded unreported). The
  review fixes change no decision and need no new edit: N5's writer is held to H51's "the
  engine writes the marker strictly"; register H23's row and spec issue 127 (gate lane) are
  stale on what nightly gates, issue 127 now marked superseded.
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
### 2026-10-17 - Gate: the engine configured; the commit tier green with it; 113 known divergences triaged against the rulebook

- `oracles/gate.toml`: `[engine] command = ["./build/engine/t5d-gate"]`, full mode -
  the engine's gate adapter as `engine/build.py` builds it, relative to the
  repository root, with no diagnostic option - and `[engine] timeout_s = 300`, the
  engine's own time limit per record (the tier's `timeout_s` where that is longer).
  Why: H23 - the gate compares the engine. It redoes the work lost with the old
  machine (61e9aa8), now under H47, so the commit tier is a smoke test that passes.
  The time limit is a review fix: the engine had shared the tier's budget, 60 s in
  the commit tier, and CI's first commit run with it took 44.4 s on cwmtt/many2 on
  a runner that ran the calibration probe at 0.75x (runners range 0.59x-1.09x), so
  a slower runner would fail every push on a time-out with no rules question
  behind it. No oracle budget, required set or golden file changes.
- `oracles/oracle_harness/gate.py`, `config.py` (review fixes): a program the engine
  command names by a relative path is the repository root's on every platform
  (`gate.engine_argv`) - Windows had looked for it from the directory the harness
  was started in, so a lane's harness started from the main checkout ran main's
  build and the engine-in-harness check looked at another file than the one that
  ran; the engine runs under `GateConfig.engine_timeout(tier)`; each run's report
  says which file it ran, its SHA-256, when it was written and the time limit
  (`engine_run`), and the progress log names it (or says it is missing).
- `oracles/known-divergences.toml`: 113 entries, KD-0057 to KD-0169 - every new
  disagreement of the engine's with a recorded oracle answer, triaged by cluster and
  settled by the rulebook (H31). 110 are ftxi's (72) and ftxi:naive's (38) check
  verdicts at 72 positions, ruled wrong by R1: at each, the engine's own capture
  scan with every playable board of the side to move passed lists exactly the
  captures ftxi (and ftxi:naive) report through `observe --explain`, and every one
  starts on a board R1 does not pass - an optional board ahead of the present, or an
  inactive timeline (CHECKPOINT 1's RQ1-RQ4); CHECKPOINT 1's table re-derives
  exactly at its 58 positions, and cwmtt/many2's 14, new, file as CHK-A1. Six rest
  on R2 (Q1) as well, and cite it. 3 are cwmtt/futures action 17, a jump written as
  a capture (`>>x`) onto an empty square, which the engine refuses as unreadable
  and cwmtt, Chess in 5D and the notation parser play: cited to the notation
  specification's "Jumps" syntax ("`x` if a piece is being taken"). No engine
  defect. The header says where the entries came from, and KD-0040 to KD-0043 no
  longer say they are inert until Q4 is ruled (R8 ruled it; they match). A review
  finding: the 3 notation entries rest on the engine reader's strict reading of a
  written `x`, which the specification's text does not settle (spec issue 125) -
  they are now marked provisional in their notes, citations and the header, which
  says how they are withdrawn (with the reader's change, not as telltales) if the
  user rules the other way; the ruling is wanted before the merge.
- `oracles/tests/test_repository.py`: the engine adapter is configured - the
  relative path, the root's build, full mode, outside the harness, a time limit of
  its own. `oracles/tests/test_gate.py` (`GateRunTests.gate_toml`) and
  `oracles/tests/test_gate_live.py` (`run_gate`): each world replaces the `[engine]
  command` line by pattern, whatever the repository configures, and every such line
  is counted, so a second one fails (a review fix: the count was capped at one), so
  no test runs the real engine on canned records; the live tests' docstring no
  longer says there is no engine. New tests (review fixes): a relative program is
  the root's wherever the harness was started, and a run started elsewhere runs
  the root's (`engine_argv`, end to end); the report names the file that ran, its
  .exe on Windows, or none; the engine runs under its own limit, or the tier's where
  that is longer, and the oracles keep their budgets; `[engine] timeout_s` is read
  and refused unless positive. Reverting either fix fails a test.
- `oracles/README.md` (the engine configured; building the engine before a gate
  run, in "Getting started"; the engine contract - the root's program, the report's
  `engine_run`, the engine's time limit; the check-disagreement and engine
  paragraphs of "What the disagreements are"; why Very Small - Open is ungated; the
  upstream reports none of the entries has yet; "Known divergences", the three
  provisional entries among them; "Other commands"; "What this is not, yet"),
  `CONTRIBUTING.md` ("Before you push": build the engine before the gate) and
  `engine/README.md` (the gate runs the engine; the provisional entries; the corpus
  timings dated - stage E7's, before many2 was read converted): brought in line.
- `docs/BUILD_LOG.md`: step 2 (the 2026-10-17 triage and runs; the root's program
  and the engine's time limit; the provisional entries; what the scratch comparison
  stands in for, and on which runners; CI - the commit tier green on Linux with the
  engine, run 39009070712, and the nightly to be dispatched on the branch before the
  merge); register rows H23, H31, H34, H36 and H47; spec issues 125 (a written `x`
  on a move that takes nothing - now with the ten records the same reading would
  refuse the other way, and wanted before the merge) and 126 ("(and reported
  upstream)"), both put to the user; spec issue 127 (issue 122's and edit 98 (b)'s
  measurement, moved out of their text, which stands as put to the user) and 128
  (the engine's time limit; the calibration probe scales nothing); proposed spec
  edits 99 (pending issue 126) and 100 (§11.0 "Stable answers only").
- Checks (2026-10-17, this machine shared with other lanes; timings are not
  measurements): `py engine/build.py --werror` clean; `verify` ok (H48's 17
  `proof-channel-vacuous` notes, as before); `selftest` 638 tests OK, 1 skipped;
  `gate --tier commit` ok - oracle layer ok, `engine differential: SMOKE TEST -
  PASSED` (calibration 1.33x); `gate --tier nightly` ok - oracle layer ok, `engine
  differential: PASSED`, gated Standard - Half Reflected and Standard - Reversed
  Royalty (calibration 1.20x). In CI (ubuntu-24.04) the commit tier passed with the
  engine at 3a46da9 (run 39009070712, calibration 0.75x); the nightly is still to be
  run there. The engine compared with every recorded oracle answer (all six golden
  files): 169 disagreements, every one matched by an entry, every entry matched. The
  proof channel stays disabled.
- Spec: §11.0 ("Tiered CI", "A known-divergences file", "Triage by cluster; settle
  by rulebook, never by vote", "Telltales", "Parse mode before full mode", "Stable
  answers only"); register H23, H27, H28, H31, H32, H34, H36, H38, H47; rulings R1,
  R2, R7 and R8 cited. Spec issues 125, 126 and 128 put to the user; spec issue 127
  measures what issue 122 and proposed edit 98 (b) wait on; proposed edits 99,
  pending issue 126, and 100. No spec text edited.

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
