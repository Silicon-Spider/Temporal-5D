# 5D Chess Engine — Decision Register

Companion to `5d-chess-transformer-design.md` (the spec) and
`scalable-transformers-architecture.md` (the general pattern). The spec holds the
reasoning; this register holds the **state** of every decision, so that "what did we
decide, and how firmly" has one answer. Section pointers (§) refer to the spec.

**Status vocabulary**

| Status | Meaning |
|---|---|
| Committed | Settled; part of the v1 build |
| Provisional | Adopted for v1, pending a named measurement |
| Prototype | Build behind a flag; adopt only if the ablation says so |
| Held open | A candidate with a reason to exist, not scheduled |
| Deferred | Decided to decide later, with a stated condition |
| Refused | See §14 for the principle each violates |
| Reversed | A prior decision changed — history in Part J |

**Class**: **C** = committed-class (irreversible once serious self-play training starts —
must be decided before it); **R** = retrofittable; — = process or principle.

---

## A. Representation (§1–2)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| A1 | One token per square; 2 registers per board; 16 global registers; T = 66N + 16 | Committed (v1) | C | Ablation 1 (piece-centric arm) | 2.1 |
| A2 | Piece type as token embedding; coordinates carried by positional encoding, not features | Committed | C | — | 2.2 |
| A3 | Per-board flags: head/superseded, active/inactive, side to move; present upgraded to continuous **Δt-to-present** | Committed | R | — | 2.3 |
| A4 | Superseded boards are immutable — the load-bearing property behind frozen K/V, frozen NNUE accumulators and the asset map | Committed (fact) | — | — | 2.4 |
| A5 | **Piece-centric tokenisation** (tokens for pieces + generator-supplied destination squares; ~2.3–2.7× fewer tokens) | Prototype — *reversed from refused* | C | Ablation 1, jointly with the window | 14.4 |

## B. Architecture and attention structure (§3–6)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| B1 | One transformer, one weight set; connectivity is the mask, not the weights | Committed | C | — | 3.1 |
| B2 | Unified attention over all tokens as the architecture | Committed | C | — | 3.2 |
| B3 | Factorised (axial) attention | Contingency only — three named triggers | C | — | 3.3 |
| B4 | 4D relative RoPE with asymmetric per-axis pairs (file 3, rank 3, timeline 4, time 6) | Committed | C | Ablation 2 | 4.1 |
| B5 | Per-head positional specialisation matching the routing table | Provisional | C | Ablation 2 | 4.1 |
| B6 | RoPE + **binned clamped biases** for Δtime/Δtimeline (extrapolation guard) | Prototype | C | Ablation 2 | 4.1 |
| B7 | d_head = 64 (4 heads) as the *second* resort for thin positional signal, before d_model | Deferred | C | — | 4.1, 13 |
| B8 | Registers: 2 per board, 16 global; aux heads pool from the trunk — registers stay free scratch | Committed | C | Ablation 3 | 5, 12.4 |
| B9 | Move-graph attention mask from the generator on ≥2 heads | Committed | R | Ablation 4 | 6 |
| B10 | Fixed multi-head routing table (heads 1–4 routed, 5–8 dense) | Prototype (speculative) | C | Ablation 6a | 6 |
| B11 | **Typed edge biases** (attack, defence, pin, x-ray, check, travel-threat), cross-board included | Provisional | R | Ablation 2 | 6 |
| B12 | **Board-pair smolgen** — full-rank biases from global registers | Prototype | R | With 6a | 6 |
| B13 | Token-level and within-board smolgen | Refused (8× the attention it modulates; rank-r ≡ +r head dims) | — | — | 6, 14.4 |
| B14 | Block-sparse mask kernels via board-major ordering; FlexAttention score_mod | Committed (engineering) | R | Ablation 6 | 6.1 |

## C. Attention scope, caching and persistence (§7, §8.5, §8.7, §10.4)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| C1 | **Build dense bidirectional first** — the baseline that prices the causal cost | Committed | C | — | 7.4 |
| C2 | Expected final: causal-along-time **cached** attention (~15–20× realistic, including the live-set recompute; ~12–18× after the supersession recompute at its upper bound) | Deferred — decided after the window measurement and the measured Elo cost, before serious self-play | C | 6c, §7.7 | 7.1–7.7 |
| C3 | Root-conditioned dense refresh once per game move (two-level mask); **trained by conditioning-offset sampling** — offsets k from search's depth distribution, k = 0 included, ~+10% total | Provisional under the cached config | C | 6c | 7.9 |
| C3a | Within-search refresh ladder: rung 1 (travel-triggered, visit-gated, whole scope) as the existence test; rung 2 probe-measured drift; rung 3 attention-weighted partial refresh as the design target *only if rung 1 shows a signal*; node refreshes **fork** the cache copy-on-write (per subtree / per board) — never overwrite; only the root refresh replaces it | Prototype (rung 1); Held open (rungs 2–3) | R — *conditional on C3's offset training*: a node refresh resets the offset to 0, within the trained range; without it, false | 6c sub-arm | 7.9 |
| C4 | Full MHA; GQA rejected | Committed | R | — | 10.2 |
| C5 | Window = W ∪ E, exact online-softmax merge; E is the rule-based reachable set; no learned eviction | Provisional — *an uncommitted decision of the committed class* | C | Ablation 1 cascade | 10.4 |
| C6 | Cascade: phase-stratified \|E\| first → provisional v1 adoption → two-arm head-to-head **under the cached config** (or scope randomisation) before serious self-play | Committed (process) | — | Ablation 1 | 10.4.2 |
| C7 | W = event-anchored priority queue on the E floor: present band → heads' recent k → branch origins → asset-flagged → near-reachable E₂ → recency; filler budget B; **E is never capped** | Provisional | C | k ∈ {0,4,8,16} arm | 10.4.3 |
| C8 | **Add-only inversion**: heuristics may add scope, never subtract below E | Committed (principle) | — | — | 10.4.3 |
| C9 | Inactive timelines are E-material; strategic-deadness pruning refused | Committed | — | — | 10.4.3 |
| C10 | Lazy K/V for boards entering E under the cache | Committed (engineering) | R | — | 10.4.1 |
| C11 | **MCGS** main search: state-keyed via the 5D-Zobrist scheme, residual backup; near-mandatory under ply-split | Prototype | R (couples to D2) | 6k | 8.5 |
| C12 | Tree reuse at match play: carry statistics, invalidate KV deltas, recompute lazily; re-rooting = GC of path-dependent structure only | Committed | R | 6n | 8.7 |
| C13 | Ponder = search the post-move position, re-root on the reply; no single-reply guess | Committed | R | 6n | 8.7 |
| C14 | Self-play runs **fresh** searches per move (no cross-move carry) | Committed | — | — | 8.7, 12.2 |
| C15 | FP8 KV cache; varlen/ragged kernels | Committed (engineering) | R | — | 10.3 |
| C16 | Dense vs 14+2 hybrid — bidirectional layers **on top only** (placement forced: a bidirectional layer invalidates every cached layer above it) | Deferred — last, only if the dense engine is node-starved | C | Ablation 8 | 7.6 |
| C17 | Block-causal mask (causal between boards, bidirectional within); registers read freely, being read by frozen tokens = present-to-frozen flow, allowed only at refreshes | Committed | C | — | 7.3 |
| C18 | Local/global layer interleave — within-board-only layers, causal cross-board every third; 2.4× cheaper per node, K/V at global layers only | Held open (layer-level fallback to head routing) | C | 6a arm | 7.6 |
| C19 | Live set fully bidirectional: every head attends to every head and is recomputed each node — necessary (cross-timeline threats run both time directions); ~3.7–5.6× the changed-boards-only cost, already in C2's figure | Committed | C | — | 7.3 |
| C20 | Supersession inside the search: recompute the board once under the state-defined mask (causal past + the history of its refresh epoch — the root's unless a node refresh above it forked one), shared down the subtree; +8–24% per node | Committed | C | — | 7.3 |
| C21 | Snapshot-at-supersession and edge-gated head-skip shortcuts | Refused — path-dependent evaluations break MCGS transposition evals and the training mask | — | — | 7.3, 14.4 |
| C22 | Conditioning audit on value-surprise promotions — refresh a node whose Q exceeds the incumbent by n before it takes the PV; catches over-valuation from stale conditioning only | Held open (gated on 6c) | R | 6c | 7.9, 11.2 |

## D. Output heads and targets (§8.1–8.3, §12.4)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| D1 | Bilinear pointer policy head over generator-supplied legal moves | Committed | C | — | 8.1 |
| D2 | Multi-move turns: board-move-as-ply vs whole turn — under the ply split, mid-turn check is legal and terminal detection runs only at turn boundaries; ftxi's semimove fine tree is prior art | **Undecided** — measured jointly with widening and MCGS | C | 6g, 6k | 8.2 |
| D3 | 51-bin **categorical** value head, FP32; its variance feeds search, commit and time | Committed | C | — | 8.3 |
| D4 | Auxiliary targets: WDL, moves-left, opponent reply, short-horizon value, **per-board travelled-into & timeline-survival** (primary), tactical volatility (proofs), per-square attack facts, **search effort** (distributional, censored top bin) | Committed set; weights ablated | C | 6j | 12.4 |
| D5 | Source rule: own-search targets are the norm; **learned-verifier estimates never enter a loss** | Committed (principle) | — | — | 12.4 |
| D6 | Sum-vs-mean loss reduction over boards | Open knob | — | 6j | 12.4 |
| D7 | Registers may be *initialised* from aggregates (6p) but never assigned aux targets | Committed | — | 6p | 12.4 |

## E. Search and time management (§8.4–8.6, §11.1)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| E1 | PUCT with top-k instantiation, entropy-adaptive progressive widening, FPU, root Dirichlet | Committed | R | 6g | 8.4 |
| E2 | Variance-scaled cPUCT + uncertainty-weighted backups (categorical variance) | Provisional | R | 6m | 8.4 |
| E3 | Per-child variance **bonus** — multiplicative on U, capped; prior→posterior variance; the mirror of E5 | Prototype | R | 6t | 8.4 |
| E4 | Narrow hard, because the auditor is watching | Committed (principle) | — | — | 8.4 |
| E5 | Final selection: visits baseline; **LCB** with ~15% visit floor; optional categorical term; path visits under MCGS | Committed | R | 6t | 8.6 |
| E6 | Raw depth / node-count factor in final selection | Refused | — | — | 14.4 |
| E7 | Time management: VOI framing; **moves-left integral allocator** on predictive nps(N); stop/extend table (smart-pruning stop, KLD, solved root, overlap+churn, dual-source volatility with disagreement-extends-hardest, falling eval, instability); extensions in evals, tight wall-clock hard bound; root-scan reserve | Committed | R | 6o | 8.6 |
| E8 | Difficulty-weighted allocation from the effort head's upper quantile; censored mass = won't-resolve flag; three guards | Prototype | R | 6v | 8.6 |
| E9 | Secure-child selection and visit threshold for solved nodes | Committed | R | — | 11.1 |

## F. Parameters (§9, §11.1)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| F1 | d = 256, L = 16, FFN 8×, 8 heads × d_head 32, ~21.5M with heads | Committed (v1) | C | Ablation 5 | 9.2 |
| F2 | Capacity ladder: FFN → depth → d_head → d_model; larger arm **L24-ff12 (44M)**; prior leans larger | Provisional | C | Ablation 5 | 9.2 |
| F3 | d = 512 | Refused (quadratic tax) | — | — | 9.2 |
| F4 | NNUE v1 spec: shared 768→256 feature transformer, 256 int16 accumulators per board, ~24 globals, eval + destination heads, single perspective, ~0.22M | Committed (as the v2 scanner eval) | R | 6d | 11.1 |

## G. Precision and serving (§10)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| G1 | BF16 trunk; FP32 value/policy heads and RoPE; cache and live computation share precision | Committed | R | — | 10.2 |
| G2 | FP8 attention later | Provisional | R | — | 10.2 |
| G3 | Full serving stack (paged/radix/cascade) | Deferred to the cached config | — | — | 10.3 |
| G4 | Early exit / adaptive depth | Held open (confidence family) | R | — | 10.3, 14.4 |
| G5 | Within-system distillation as a capacity tool | Held open | R | Ablation 5 arm | 10.3 |
| G6 | MLA-style latent KV / MoE / speculative decoding | Refused / low priority / subsumed by virtual-loss batching | — | — | 10.3 |
| G7 | SPSA + SPRT tuning discipline | Committed (process) | — | — | 14.4 |

## H. Verification layer (§11)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| H1 | Scanner: brute-force node-limited alpha-beta over forcing moves, **permanently on**; differently-wrong principle | Committed | R | 6b | 11.1 |
| H2 | Two channels: proofs (all-evasions) → solved bounds; estimates → prior boosts only | Committed | — | — | 11.1 |
| H3 | Node cap, not depth cap — termination unconditional; horizon-per-budget measured | Committed | — | 6b | 11.1 |
| H4 | Scanner move ordering is net-free (MVV-LVA, killers, history, continuation) | Committed | — | — | 11.1 |
| H5 | Pruning discipline: no speculative pruning on proof lines; ProbCut/futility/null-move estimate-only; LMR banned on evasions; null-move waits on the zugzwang answer | Committed | — | — | 11.1 |
| H6 | Scanner eval: material count v1 → 5D-NNUE v2 | Committed sequence | R | 6d (after 6b) | 11.1 |
| H7 | NNUE trained on scanner-labelled, quiet-only data; never distilled either way | Committed | — | — | 11.1 |
| H8 | NNUE threat features: same-board adopt (v2/v3); cross-board refused; head-only overlay v3-speculative | Held open / Refused | R | 6q | 11.1 |
| H9 | Attention over head-board accumulators (v3 cross-board awareness) | Held open | R | — | 11.1, 14.4 |
| H10 | Async priority ladder rungs 1–5, budgeted against shared movegen | Committed | R | 6e | 11.2 |
| H11 | Rung 2 two-sided: demotion audit at 2–3×, NNUE-vs-value disagreement → curriculum | Committed | R | 6i | 11.2 |
| H12 | Rung 3 two-stage: NNUE screen of the discard pile → forcing scan | Committed (v2-gated) | R | 6h | 11.2 |
| H13 | Rung 4 asset map via the destination head, once per superseded board | Committed | R | — | 11.2 |
| H14 | Scan budgets table; confidence-escalated rung 1 (dual trigger), add-only above the uniform baseline | Committed | — | — | 11.2 |
| H15 | Synchronous debug mode (single thread, deterministic TT) | Committed | — | — | 11.2, 11.3 |
| H16 | Ensemble-of-engines voting | Refused | — | — | 11.2 |
| H17 | Lazy SMP over the rungs; shared TT; per-rung histories decayed in-game, cleared between games | Committed | R | 6f | 11.3 |
| H18 | 5D Zobrist: content hash mixed with (timeline, time) by multiply-rotate; **ruleset ID in every exact-store key** | Committed | — | — | 11.3 |
| H19 | Proofs export to the permanent solved store immediately; never TT-only | Committed | — | — | 11.3 |
| H20 | Within-scanner correction history | Held open (v2) | R | — | 14.4 |
| H21 | Dedicated df-pn long-mate thread | Held open (v2; needs a 5D dominance definition — the repetition gate is gone) | R | — | 14.4 |
| H22 | Rules engine: every rule in the generator — royal-capture check detection (phantom boards for the mate-vs-stalemate verdict), turn-level legality at submission, mate/stalemate via naive DFS + hypercuboid, per-board chess rules as 5D queries, exact bookkeeping | Committed | — | — | 11.0 |
| H23 | Differential-testing gate against ftxi/5dchess_engine, penteract/cwmtt and Chess in 5D — move counts, check and mate verdicts over 5dpgn corpora; proof channel disabled until it passes; re-run per generator change and per variant | Committed (process) | — | — | 11.0 |
| H24 | Build the generator in-house; ftxi's engine as reference oracle, not the core — the generator must emit outputs no existing engine produces, and reuse would forfeit ftxi as an oracle (H26); oracle internals as pinned are recorded in PROVENANCE.md | Committed | — | — | 11.0 |
| H25 | Official game client as a rules oracle | Refused — believed to miscall some positions as checkmate | — | — | 11.0, 14.4 |
| H26 | Third-party code: oracles, not core — every reused component forfeits one independent oracle for what it covers; hypercuboid reimplemented in-house (naive DFS as the in-house cross-check; ftxi and cwmtt external; Chess in 5D the most independent); **the oracles are the only third-party 5D code in the repository** (test harness only) — nothing from another 5D project is vendored into the engine for any purpose; general libraries are ordinary package dependencies; the fourth submodule (adri326/5dchess-notation) is the notation specification, and any parser code it ships counts as a parse-only oracle | Committed | — | — | 11.0 |
| H27 | Oracles pre-included from the first commit: pinned submodules in a separate test harness (the three oracles plus the notation specification), boundary enforced by build structure and CI; tiered CI (ftxi per commit, cwmtt and Chess in 5D nightly and before releases); pinned corpus; known-divergences file with rulebook citations; provenance file; contributor and agent instructions | Committed (process) | — | — | 11.0 |
| H28 | Own 5dpgn/5dfen parser, written from the pinned notation specification (adri326/5dchess-notation) — a shared parser corrupts both sides of a comparison identically | Committed | — | — | 11.0 |
| H29 | Copying GPL engine code (Stockfish, Lc0) into the engine | Refused — licence-compatible under the project's GPL-3.0 (M11), but it would end sole copyright, which keeps relicensing or dual-licensing possible; other projects' code enters as ideas only (H26), reimplemented from papers and the Chess Programming Wiki | — | — | 11.0 |
| H30 | Match states, not move counts: a canonical fingerprint of each parsed position at every ply; parse divergences are their own class, triaged before rules comparisons and settled against the pinned notation specification | Committed (process) | — | — | 11.0 |
| H31 | Triage disagreements by cluster and settle each from the rulebook, never by oracle vote; check first; the engine's check detection waits on it | Committed (process) | — | — | 11.0 |
| H32 | Stable answers only: budget-limited → "unknown at budget X" or budget raised; nondeterministic → unstable, excluded; 100% required on stable answers | Committed | — | — | 11.0 |
| H33 | Recorded-answer integrity: immutable, content-hashed, verified before every comparison; comparison logic outside the engine adapter's reach; harness changes reviewed | Committed | — | — | 11.0 |
| H34 | Telltales: every recorded oracle error doubles as a copy detector — an engine that reproduces one probably copied that oracle's logic | Committed | — | — | 11.0 |
| H35 | Tiered CI green on Linux before the first engine commit | Committed (process) | — | — | 11.0, 15.1 |
| H36 | Check status vs submission legality (CHECKPOINT 1): status = passing on all active boards in the present, the opponent could capture a royal from any of their playable boards — optional boards ahead of the present and inactive timelines are not passed; legality = after the actual turn, a royal capture from any of the opponent's playable boards, active or inactive, makes the submission illegal; a capture counts even if it would branch into an inactive timeline | Committed | — | — | 11.0 |
| H37 | Mate verdicts are proofs: an exhaustive search that terminated; an exhausted budget is "unknown", never "mate"; the two detectors are cross-checked in the gate | Committed | — | — | 11.0 |
| H38 | The rulebook is docs/rules/README.md: sources ranked (in-game Guide > cited developer rulings > established references > forums), each entry cited with confidence, silent cases labelled project decisions and mirrored here, never an oracle as a source; pinned, gate re-run on change | Committed (process) | — | — | 11.0 |
| H39 | No repetition rule (project decision): passing is impossible and every move creates a board, so the full state cannot recur; graph-history interaction does not arise | Committed | — | — | 11.0, 8.5 |
| H40 | Castling and en passant attack tests are per-board: castling blocked only by attacks from pieces on the castling board, en passant per-board only | Committed | — | — | 11.0 |

## I. Training (§12)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| I1 | AlphaZero-style self-play with a curriculum over N | Committed | C | — | 12 |
| I2 | Contamination boundary, three channels: visit distributions (exclusion or provenance, √-decay at target construction); position selection safe; loss targets never learned-verifier | Committed | — | — | 12.2 |
| I3 | Mature regime: scanner permanent; three curriculum generators (scanner-flagged, adversary-found, uncertainty-prioritised); declining catch-rate is biased; held-out adversarial eval; match config noise off / τ = 0 / MLH steering / resign thresholds | Committed | — | — | 12.3 |
| I4 | **Two-family principle**: confidence-driven mechanisms never scale down the confidence-independent layer; confidence flows out of the net, never back in | Committed (principle) | — | — | 12.3, 14.4 |
| I5 | Playout cap randomisation; policy targets from full searches only; difficulty-biasing only as a mixture | Committed | — | 6l | 12.5 |
| I6 | Uncertainty-prioritised sampling — epistemic trigger (persistent bucket), additive to I3's other generators | Prototype | — | 6u | 12.3 |
| I7 | Adversarial league: per-checkpoint exploiters vs frozen net; PFSP variance-shaped; hopeless tail = diagnosis queue; pruning; **held-out adversaries**; heterogeneous attackers; **size classes** (coverage + robustness curve, not cheapness — victim evals dominate game cost), compute-denominated attack cost, size-stratified held-out set, specificity scoring, victim-sims curriculum, victim modelling; cost-of-attack + Elo; monitor from the start, train from the trigger | Committed | — | 6s | 12.7 |
| I8 | Variants: standard net absorbs rule-identical variants (self-identification principle); Reversed Royalty specialist warm-started with a scratch control; Princess interim specialist | Committed | — | 6r | 12.6 |
| I9 | Exact stores are per ruleset | Committed | — | — | 11.3, 12.6 |
| I10 | Calibration monitoring of the categorical variance by N and phase; trained error head as fallback | Committed (monitoring) | — | 6m | 14.4 |
| I11 | Self-play turn cap defined from the state, scored as a draw — a convention for ending runaway games, not a game rule; frequency logged | Committed | — | — | 12.3 |

---

## J. Reversals and corrections log

The design's history of changed minds, kept because refusals carry the assumptions of the
moment they were made.

| When (order) | Was | Now | Why |
|---|---|---|---|
| 1 | Two weight sets (intra- / inter-board) | One weight set; connectivity is the mask | Weight sharing is what makes parameters N-independent (§3.1) |
| 2 | Window worth ~9× | ~7× total | 9× was attention-only; FFN and projections don't shrink (§13.0) |
| 3 | Window and cache are substitutes | They compose — cached alone 30×, cached + window 70× | Arithmetic; the "substitutes" phrasing survived in two places until audited (§7.5, §10.4.2) |
| 4 | 30 → 148 → 337 evals/sec at 240 boards | 337 (dense), ~2,370 windowed | Three arithmetic corrections; recorded in §13.0 |
| 5 | Axial attention as an option | Contingency with three triggers | Dominated once the window and cache compose (§3.3) |
| 6 | NNUE rejected as a verifier *model* | Adopted as the scanner's leaf *eval* | Verdicts attach to (component, role); separable eval fine when search supplies dynamics (§11.1) |
| 7 | Quiescence termination as an existential question | Node cap — termination unconditional; horizon-per-budget is the question | dlshogi's leaf-solver cap (§11.1) |
| 8 | d_head thinness fix = d_head 64 first | Asymmetric per-axis pairs + per-head specialisation first; d_head 64 second | The even split was never required (§4.1) |
| 9 | "Exact sources only" for aux targets | Own-search targets are the norm; only learned-verifier estimates barred | The literal rule prohibited AlphaZero's own policy target (§12.4) |
| 10 | Subtree value bias and correction history = harmful | Admissible within-system at inference; refused across systems and in losses | Training rule was misapplied to inference-time statistics (§14.4) |
| 11 | Square-level smolgen refused; then two-level held open | Board-pair only; within-board costs 8× its block attention | Rescale check, applied twice (§6) |
| 12 | NNUE threat features: "never" | Same-board adopted; cross-board refused | Refused against evidence already in the survey — process note 1 (§14.4) |
| 13 | §7.9 dilemma: drop tree reuse or accept staleness | Carry statistics, invalidate KV deltas | Two kinds of staleness conflated (§8.7) |
| 14 | Uncertainty penalty is selection-only | Penalty at commit, *bonus* in exploration — resolve, then decide | LCB alone starves noisy-promising moves (§8.4, §8.6) |
| 15 | Piece-centric tokenisation declined | Prototype-tier, committed-class | Priced at single-board scale; two objections dissolved by typed edges and varlen kernels — process note 2 (§14.4) |
| 16 | Early §10.4.2 claims ("cache never justifies itself", leftover 9×) | Removed | Stale claims survive where the doc is cited but not reread |
| 17 | §7.6 hybrid framed as a Qwen3-Next-style 3:1 interleave; placement "untested, instinct late" | Bidirectional layers forced to the top; mid-stack placement refused | Linear-attention hybrids are all-causal, so interleaving invalidates nothing there; here a bidirectional layer invalidates the cache above it (§7.6) |
| 18 | Refresh ladder "no training commitment"; refresh training unspecified | Conditioning-offset training specified; the no-commitment claim holds only given it | Evaluations below a refresh see history conditioned on an ancestor — a regime dense training never shows (§7.9) |
| 19 | Summary: the window "determines whether the caching architecture is worth asking"; "run it with FP8"; window measured first | Window measured early because it sets throughput on every path (composes with the cache); FP8 later; rules engine and gate come first | The summary was never reread after §7.5's composition correction — reversal 16's lesson, one section over |
| 20 | §11.0 oracle facts taken from web research: cwmtt's checkmate tool uses z3; ftxi packs coordinates at 8 bits per axis (one of H24's two reasons) | Removed — neither is true at the pinned commits; PROVENANCE.md is authoritative on oracle internals; H24 stands on its remaining reasons | READMEs and docs describe a moment; pinned code is the ground truth. Caught by the implementation phase — what §15 is for |
| 21 | §11.0 item 4: castling's attacked-square test is a 5D query | Per-board only, for both castling out of and through check; en passant per-board | The documented rule (Wikipedia's rules summary, a developer ruling) contradicted an unsourced claim in the spec; raised by the rules reference at CHECKPOINT 1 |

## K. Open items

| Item | Gates | Where |
|---|---|---|
| ~~5D repetition rules~~ — **resolved at CHECKPOINT 1**: no repetition rule; the full state cannot recur (H39). Its five consumers are closed (§8.5, §8.7, §10.4.3, §12.6, §14.4) | — | 11.0 |
| Scanner horizon per node budget | NNUE v2 (6d), the quiet screen (6h), quiet-only NNUE training | 11.1, 6b |
| Stand-pat validity / 5D zugzwang | Null-move import into the scanner | 11.1 |
| §8.2 ply-split vs whole turn | Couples MCGS; measured jointly | 8.2, 6g, 6k |
| Sum-vs-mean reductions (loss over boards; NNUE over heads) | — | 12.4, 11.1 |
| Filler budget B's numeric value | Sized against batch memory at implementation | 10.4.3 |
| ±0 timeline index convention | Turn Zero support | 12.6 |
| Licence of trained weights (GPL-3.0 covers the code; weights are data) | Distributing any weights | 11.0, M11 |
| Window on/off, piece-centric on/off | Committed-class — decided before serious self-play | 10.4, ablation 1 |

## L. Predictions on record

| Prediction | Where pinned |
|---|---|
| k = 0 (heads + E + flagged only) loses the equal-sims arm meaningfully; ties go to the cheapest config | Ablation 1 |
| Windowed wins equal-wall-clock under the cached config (~80%); unwindowed wins equal-sims only ~55/45 (attention-dilution contrarian case); falsifier = coordinate-gap learning failure | *Conversation only — not yet pinned* |
| Windowed shows the higher initial exploit rate on scope-targeted attacks; filler tuning narrows it; closure open | §12.7 |
| Early in training small exploiters find most holes; at maturity only large or from-scratch ones do | §12.7 |
| Oracle check disagreements concentrate in which boards count as sources of attack — **confirmed at CHECKPOINT 1**: all seven clusters did | §11.0 |
| 6i demotion-refutation rate drops when uncertainty weighting lands — overlapping defences, not audit failure | §8.4 |
| Raw-variance sampling overfits aleatoric positions (6u) | Ablation 6u |
| Difficulty-weighted allocation's gain concentrates in the late game | §8.6 |
| The window is off in the early game (~100 boards) and kept on its late-game value: ~1.6× at 160 boards up to ~4× at 600 at \|E\| ≈ 25 | §15.6 |

## M. Implementation plan — the untrained build (§15)

| # | Decision | Status | Class | Gate | § |
|---|---|---|---|---|---|
| M1 | Build everything, train nothing, first: the untrained build verifies correctness and cost; quality needs training (the ablations) | Committed (process) | — | — | 15 |
| M2 | Phase order: rules engine + gate → zero-training measurements (\|E\|, 6b, movegen) → network stack and search with random weights, committed-class alternatives behind flags → equivalence suite green and cost figures measured → training | Committed (process) | — | — | 15.1 |
| M3 | Equivalence suite gates training — random weights at realistic scale, several seeds, tolerance matched to the arithmetic, adversarial constructions | Committed (process) | — | — | 15.2 |
| M4 | Every cost figure measured and written back over its estimate, with conditions, in the same edit as the register | Committed (process) | — | — | 15.3 |
| M5 | Not built in the untrained phase: adversarial league, variant specialists, refresh-ladder rungs 2–3; the window only if the \|E\| curve survives; the NNUE only after 6b | Committed | — | — | 15.1 |
| M6 | Phase 0 internal order: commit and Linux CI → fingerprints → check-cluster triage → check detection, legality, naive mate, hypercuboid, each against the gate; representation, parser, per-board movegen, Zobrist and fingerprint export proceed in parallel with triage | Committed (process) | — | — | 15.1 |
| M7 | Window threshold: per phase, on at k = 8 where median \|E\| is under the k = 8 ceiling (~31 / 77 / 169 / 290 at 160 / 240 / 400 / 600 boards), on at k = 4 where only its ceiling is met, off otherwise (at ~100 boards it cannot pay at k = 8); killed outright if the phases where it is on carry under half the compute. Ceilings shown at assumed timeline counts; the test applies the bar to measured W + measured median \|E\|. Break-even: 1.5× all-in multiplier, derived from the pre-registered Elo priors | Committed (pre-registered, derived) | — | Ablation 1 | 15.6 |
| M8 | Scanner thresholds (6b): rung 1 kept if its derived leaf budget resolves ≥ 3 forcing half-turns in ≥ 50% of leaves; verification layer as designed if the 50 ms root scan resolves ≥ 5 in ≥ 50% of roots, else STOP for redesign; NNUE v2 proceeds if ≥ 50% of scan leaves reach a quiet position; stand-pat bands ≤ 5% / 5–20% / > 20% invalid | Committed (pre-registered, anchored) | — | 6b | 15.6 |
| M9 | Movegen thresholds: expansion ≥ 20,000/s per GPU at 240 boards and ≥ 10,000/s at 600; flat in N at fixed \|E\|; re-checked at Phase 3 as ≥ 2× the measured evaluation rate | Committed (pre-registered, derived) | — | Phase 1 | 15.6 |
| M10 | Phase 1 order: movegen throughput first — it sets 6b's node budgets | Committed (process) | — | — | 15.6 |
| M11 | Project licence: GPL-3.0 — LICENSE from the repository's initial commit, recorded in PROVENANCE.md; obligations attach on distribution of source or binaries, not private use or running as a service; BSD-2 code adapted into the ftxi driver is compatible with its notice retained; whether the licence reaches trained weights is open, to decide before any weights are distributed | Committed | — | — | 11.0 |

---

*Maintenance rule: a decision changes here in the same edit that changes it in the spec.
A register that lags the spec is worse than none — it becomes the stale claim.*
