# 5D Chess Transformer Engine — Design Document

A neural evaluation architecture for 5D Chess With Multiverse Time Travel, targeting an
Lc0/AlphaZero-style search engine.

---

## 0. Reading this document

Sections 1–8 are architecture. Sections 9–12 are engineering. Section 13 is the honest
list of things that could be wrong. Section 14 records rejected approaches so they don't
get relitigated. Section 15 is the implementation plan: build everything untrained first.

**The state of every decision — adopted, provisional, prototype, held, refused, reversed —
lives in the companion `5d-chess-decisions.md`**, indexed to these sections with each
decision's class (committed before training vs retrofittable) and gating experiment. This
document holds the reasoning; that one holds the status. A change here is a change there
in the same edit.

**The architecture is unified attention** — one weight set, all tokens in one sequence,
structure carried by 4D positional encoding (RoPE plus a learned 4D displacement bias,
§4.1) and a move-graph mask.

Factorised (axial) attention was previously presented as a co-equal Option B with **[A]** /
**[B]** markers throughout. That framing is withdrawn: axial is dominated by both the window
and the cache (§3.3), and the markers implied a live choice that is not live. It is retained
as a single documented contingency with a stated trigger condition.

**A separate and larger decision** cuts across both: whether attention is **dense
bidirectional** or **causal along the time axis so that superseded boards cache**.
Earlier revisions of this document recommended causal-and-cached. That recommendation is
reversed — see §7. Build dense. The cache is a deferred optimisation whose cost cannot be
measured without a dense baseline to measure it against — and the first real data, human
games ending at a median of 44 boards (§13), weakens its case further.

**Most figures here are arithmetic, not measurements** — FLOP counts, or numbers borrowed
from LLM-serving literature that may not transfer — with confidence marked where it
matters. The exceptions have been measured since the build began: movegen throughput,
the verification tests (6b, 6b′, 6b″), the window's |E| curve and the sizes of real human
games. Each is marked where it is used, with its conditions in the repository's
docs/MEASUREMENTS.md, CPU-labelled until GPU time exists (§15.3).

---

## 1. The problem

5D Chess With Multiverse Time Travel arranges 8×8 boards on a 2D lattice indexed by
**(timeline, time)**. Pieces move within a board by normal chess rules, backwards along
the time axis, sideways along the timeline axis, and along diagonals that combine any of
these. Moving into a past board spawns a new timeline branching from that point.

This breaks the standard board-game network architecture in three places:

1. **Input size is unbounded.** The number of boards, N, grows during play. A fixed
   input tensor cannot represent the state. Mid-game states of 100–600 boards are
   normal (5–15 timelines × 20–40 half-turns of history).
2. **Output size is unbounded.** A move is (source board, source square) →
   (destination board, destination square). AlphaZero's fixed 8×8×73 policy plane has no
   analogue.
3. **The geometry is four-dimensional.** Movement is translation-invariant along four
   axes, not two, and long-range interactions cross board boundaries.

Any viable architecture must have parameter count independent of N and must generalise
to values of N unseen in training.

---

## 2. State representation

### 2.1 Tokens

**One token per square**, across every board in the state. Plus registers (§5).

Token count: `N × (64 + R)` where R is registers per board. At N = 240 and R = 2, that
is ~15,800 tokens.

### 2.2 Per-token features

| Feature | Encoding | Notes |
|---|---|---|
| Piece type | Learned embedding | ~13 states incl. empty |
| Colour | Learned embedding | |
| Has-moved flag | Learned embedding | Pawns, rooks, kings and brawns only (5DFEN's `*` marker) — castling, the double step |
| File, rank | Positional (§4) | |
| Timeline, time | Positional (§4) | |

### 2.3 Per-board categorical flags

These are *not* coordinates and need their own embeddings, added to every token on the
board:

- **Head vs superseded** — is this the live board on its timeline?
- **Active vs inactive timeline** — inactive timelines don't require moves
- **Present** — is this board at the minimum time among active timelines? Upgraded to a
  continuous **Δt-to-present** feature (time offset from the present line, embedded per
  board): relative RoPE gives Δt between tokens, but a token cannot read its own distance
  to the present without attending to it, and how live a board is drives obligations and
  activation. Free; keeps the binary flag as its zero. Under the cached configuration this
  and the active/inactive flag would go stale on frozen boards, so there they enter as
  score-time key biases instead (§4.1), and frozen tokens keep only intrinsic features.
- **Side to move**

**Global turn-state inputs** — embedded into the global registers, because they belong to
no board (§5.2):

- **Acting player** (D8): who is moving now. Per-board side-to-move flags cannot say
  whether a player has already submitted, and after the present shifts a state can share
  every board with the opponent's turn start.
- **Mandatory boards still to move**, exact from the rules engine: one fewer per ordinary
  move, two fewer for a move between two present boards (it creates a board on both),
  zero once a branch rewinds the present.
- **Optional boards still playable**, and **moves already made this turn** — the second
  also tells a turn start from a mid-turn state with the same boards, which the network
  could not do before (CHECKPOINT 6).

Exact counts, never predicted ones: the rules engine knows them. Committed-class — they
are inputs.

### 2.4 The critical structural property

**Superseded boards are permanently immutable.** Travelling into a past board creates a
new timeline rather than editing the existing one, so once a board is superseded its
contents are frozen for the rest of the game.

This is what makes the caching option in §7 *possible*. It does not make it *advisable* —
the contents being frozen doesn't mean the useful representation of them is.

Note that "past / present / future" is the wrong decomposition — timelines advance at
different rates, so heads sit at varying time coordinates. The split that matters is
**head (live) vs superseded (frozen)**.

---

## 3. Architecture

### 3.1 The weights-vs-masks distinction

This is the single most misunderstood point in the design, so it's stated plainly:

> Attention weights do not encode *who attends to whom*. The mask does.
>
> Q/K/V projections are shared across all positions by construction. A token attends to
> another token if and only if the mask permits it. Confining attention to within a board
> is a masking decision, not a weights decision.

Three consequences:

1. Parameter count is automatically independent of N. Nothing needs to be "repeated per
   board" — that is what weight sharing in attention already means.
2. Separate intra-board and inter-board weight sets are a *capacity* choice, not
   a structural requirement (relevant only under the §3.3 contingency). Tying them and alternating the mask per layer works and
   halves parameters. Run it as a baseline before assuming you need two sets.
3. Both passes are **self-attention over different token subsets**, not cross-attention.
   Cross-attention means Q from one stream and K/V from another; that is not what is
   happening here. The distinction determines which kernel and cache path applies.

### 3.2 Unified attention — the architecture

Every square token carries its full 4D coordinate, so a single attention operation over
all `N × (64+R)` tokens learns "same board / different board" from the coordinates
themselves.

- Any square-pair interaction resolves in **one layer**, including long 4D diagonals.
- One weight set.
- Structure comes from positional encoding (§4) and the move-graph mask (§6).

### 3.3 Contingency: factorised (axial) attention

Alternating masked passes. The natural factorisation is **four axial passes** — file,
rank, timeline, time — not the two-way within-board/across-board split, because
movement is along four axes and the four-way version matches the geometry.

- Cost: `T × (64 + N)` instead of `T²`. At N=240 roughly 25× cheaper.
- Cost of that saving: no single layer can model an arbitrary square pair. Long 4D
  diagonals need multiple hops. Budget **24+ layers** rather than 16.

If using the two-way split rather than four axial passes, board-level registers (§5) carry
the inter-board pass rather than pooled board vectors. Mean-pooling a board into one vector
destroys square identity, which is exactly what cross-board moves depend on — a bishop
attacking along a 4D diagonal is a square-to-square relation.

#### Why this is a contingency and not an option

| | Dense 16L | Axial 24L | Axial gains |
|---|---|---|---|
| Full history (240 boards) | 337/s | 1,606/s | 4.8× |
| Windowed (80 boards) | 2,366/s | 4,825/s | **2.0×** |

**Dominated in both directions.** The window (§10.4) gives 7.0× with *no* expressiveness
cost — better than axial's 4.8×. The cache (§7.1) gives 30× — six times better.

**A cost omitted from those figures.** Axial alternates intra-board passes (which want
board-major ordering, §6) with board-axis passes (which want square-major), so it needs a
transpose or strided access between passes. Real axial implementations pay this. The 4.8×
and 2.0× above ignore it, so **axial is worse than shown** — the demotion holds a fortiori.

**And it self-defeats.** Axial removes the quadratic term, but after windowing, attention is
only 67% of cost, capping the achievable gain near 3×. Axial then needs 50% more depth,
inflating the FFN and projection terms it does not touch. Its own attention share falls to
0.3% — it optimises what is already negligible while paying for the privilege.

General form of that principle: **axial attention is worth its expressiveness cost only
while attention dominates the cost model, and scope reduction removes that condition.**

#### The trigger condition

Axial is the only option that reduces cost **without dropping tokens and without causal
masking**. It becomes relevant if *all three* hold:

1. The window measurement (§10.4) comes back badly — most history is genuinely reachable
2. The causal constraint (§7.3) proves expensive in Elo *even with §7.9's root-conditioned
   refresh*
3. 337 evals/sec is too slow

**Condition 1 is a live risk, not a remote one.** Queens and rooks travel arbitrarily far
along the time axis, so the set of legal travel destinations may cover nearly all history.
That is the specific reason this contingency is documented rather than deleted.

---

## 4. Positional encoding

Absolute indices fail: games run long, and timeline indices are assigned by branching
order and are semi-arbitrary. **Relative encoding on all four axes** is what buys
generalisation to unseen N.

### 4.1 Recommended: 4D RoPE

Split head dimension into four groups, rotate each by its own coordinate
(Δfile, Δrank, Δtimeline, Δtime).

Applied to Q and K *before* the attention kernel, so on its own it runs on stock
FlashAttention with zero kernel work. That once looked like the decisive advantage over a
learned bias. It no longer decides anything: the typed edges (§6) and the binned biases
below already need a score-modifying kernel (FlexAttention), and a learned bias rides on
the same score function at no new kernel cost.

**Sizing the split — asymmetric, and per-head.** The even four-way split was never
required; two refinements replace an earlier "raise d_head to 64" caveat:

- **Allocate rotary pairs by coordinate range**: file 3, rank 3, timeline 4, time 6 (of
  16 pairs at d_head = 32) — files span 0–7 while time spans 0–40+.
- **Specialise per head, matching §6's routing table**: the same-board head spends all 32
  dims on file/rank; the same-square-across-boards head all of them on timeline/time; the
  move-graph head runs near-NoPE (its structure comes from the mask); dense heads carry
  the asymmetric four-axis split.

Positional capacity lands where each head's mask says it is needed, and the d_head = 32
thinness concern largely evaporates. If signal still proves weak after both, d_head = 64
(4 heads) is the next resort — before d_model. Folded into ablation 2.

**Hybrid with binned relative biases — the answer to the extrapolation risk.** Learned
per-head bias tables indexed by *bucketed* relative offset (the T5/ALiBi family) have
one property RoPE lacks: **clamped extrapolation** — offsets beyond the last bin share
its bias, so unseen Δtime values degrade gracefully instead of rotating into untrained
territory. That is exactly the coordinate-gap failure mode registered against the window
(§10.4). So: **RoPE for file, rank and near-time** (fine geometry, multiplicative);
**binned learned biases for Δtime and Δtimeline** (coarse, additive, log-scale bins,
clamped). Cost is bins × heads parameters and a gather in FlexAttention's score_mod. An
ablation-2 arm by name already; now with a reason to expect it to matter.
**Superseded by the 4D displacement bias below** (CHECKPOINT 6), of which it is the special
case that ignores file and rank.

**What RoPE cannot express — and the displacement bias that can** (CHECKPOINT 6). RoPE's
positional term is a sum over axes, so for any query and key the positional part of a
head's attention factorises into one factor per axis. Chess geometry does not factorise:
no head can favour displacements (1,1) and (2,2) while disfavouring (1,2) and (2,1), so
diagonals and knight moves are inexpressible as position — and in 5D so is every diagonal,
triagonal and knight relation across boards. (A single-axis line, such as "same square,
another time", does factorise; what fails is any relation that couples axes.) The content
route is closed too: Lc0's tokens carry absolute square embeddings, from which content can
rebuild such patterns, but A2 keeps coordinates out of ours. The typed edges (§6) deliver
*current* attacks exactly, not latent geometry — where a piece could go next, a fork one
move away.

The evidence is direct. In Chessformer's ablations, a static 15 × 15 table of per-head
biases indexed by (Δrank, Δfile) beat absolute embeddings by 40 ± 18 Elo at equal compute
(searchless, ~3.7M parameters), and Lc0's earlier 64 × 64 learned table let its model play
"as if 50% larger" at negligible throughput cost. Lc0's own relative-position experiment,
BT5, used a richer and costlier encoding — see below.

So: a **4D displacement bias** — one learned scalar per head per layer, indexed by the
pair's displacement (Δfile, Δrank, Δtimeline, Δtime in half-turns). Its shape, pinned at
CHECKPOINT 6:

- **A full 15 × 15 file-rank slice for every board-distance class.** Classes are exact up
  to two timelines and two turns apart (|Δtimeline| ≤ 2, |Δtime| ≤ 4 half-turns), then
  binned and clamped, so unseen board counts extrapolate as the binned biases did; the
  same-board class is Chessformer's table. Each board distance needs its own slice because
  a spatial offset changes meaning across boards. A knight moves two steps along one axis
  and one along another, any two of the four: on the same board a two-file offset is a
  rook-like relation, one timeline over the same offset is a knight jump, and the
  within-board knight offset is no move at all there. A slice shared across boards would
  learn the wrong geometry. Knight and king relations fall inside the exact classes.
- **Plus exact line classes** for displacements beyond them: movement along one axis only,
  or equal steps along two, three or four axes — the rook, bishop, unicorn and dragon
  lines — which the clamped far bins would blur, and along which 5D pieces slide across
  many timelines.

It **replaces** the binned Δtime/Δtimeline biases and sits beside RoPE, which stays for
content-modulated relative attention. Pairs involving registers get their own entries:
per-board registers by their board-distance class, global registers one entry per head
(§5.3). It is computed at score time from coordinates, so nothing cached goes stale. About
1.6M parameters — ~7% of the network — and no compute beyond a lookup per pair.
Provisional, the v1 default; ablation 2 keeps a no-table arm.

**Not Lc0's BT5 encoding — the cheaper member of its family.** Lc0's relative-position
experiment (Monroe's `use_rpe_q/k/v` flags; the team's 2024 preprint, *Mastering Chess with
a Transformer Model*) used Shaw-style relative position *representations*: learned vectors
added to queries, keys and values for each square pair, so the positional effect depends
on content. In the preprint's ablation they beat a scalar relative bias — policy accuracy
59.3% against 58.2%, absolute embeddings 57.4% — and the preprint itself notes their cost
is negligible at 64 tokens but grows with the token count. That cost is the obstacle here.
The value-side vectors need per-pair aggregation FlexAttention cannot express, so a custom
kernel. The key-side term dots every query with every displacement class: against the
full table's ~12,000 classes that is ~4× the QK^T product at 44 boards, but confined to a
compact set of ~1,400 (the exact near classes plus the line classes) it is ~half of QK^T
at 44 boards and ~9% at 240, because it grows with T rather than T². And in Shaw et al.'s
own ablation the key-side term alone matched the full version. So the scalar table is v1;
a **key-side, compact-class Shaw term** on top of it is a prototype arm of ablation 2
(B17), built behind a flag and costed in the untrained build; value-side vectors are held
open. BT5 itself is reported on forums to have played weaker than BT4 — a comparison across
different networks and runs, so it weighs against neither ablation — and the published
Chessformer settled instead on a *dynamic* bias generated from the board (GAB, smolgen's
lineage), whose affordable form here is board-pair smolgen (B12).

**Relative to the present, at score time.** A3's Δt-to-present and the active/inactive
flag describe a board's *current* status, which changes after the board freezes (§7.3).
Under the cached configuration frozen boards therefore carry only intrinsic features, and
current status enters as score-time **key biases** — per head, by the key board's
Δt-to-present bin and timeline status — recomputed every evaluation for the cost of a
lookup. Live tokens keep A3's features as inputs.

### 4.2 Alternative: learned relative bias

T5-style bias over 4D offsets. More expressive, but an arbitrary learned bias tensor
doesn't fit the stock kernel. Requires **FlexAttention**, which compiles arbitrary
`score_mod` functions into Flash-style kernels (~95% of FA3 forward performance on newer
backends).

Adopted as a supplement rather than a replacement (CHECKPOINT 6): the 4D displacement bias
in §4.1 sits beside RoPE.

### 4.3 Worth testing: ALiBi-style per-axis decay

Four distance penalties, one per axis. Trains short and extrapolates long — directly
addresses the unseen-N problem — and FlashAttention supports ALiBi natively.

**Caveat:** monotonic decay is wrong for chess. A rook seven squares away matters as much
as one square away. Needs per-head learned slopes including near-zero ones. Experiment,
not default.

### 4.4 Note on the timeline axis

Timeline distance is a genuine metric — moving n timelines is like moving n squares —
so Δtimeline is a legitimate relative coordinate. But branching order is partly arbitrary,
making it a noisier axis than the other three.

---

## 5. Registers (thinking tokens)

Dedicated learned tokens appended to each board, plus a global set. Based on the
"ViTs need registers" finding: models repurpose low-information tokens as global
scratchpads, producing high-norm artifacts that degrade the hijacked representations.

### 5.1 Why this matters more here than in vision

A board with 12 pieces has 52 empty squares carrying almost no information. The model
*will* hijack some. But empty squares are not semantically inert in chess — an empty
square's value is *which pieces control it*, which is load-bearing for check detection,
blocked diagonals, and time-travel arrival squares. Registers protect representations you
actively use.

### 5.2 Sizing

| | Count | Reasoning |
|---|---|---|
| Per board | **2** | Multiplies by N, so quadratic cost. 8 registers on 64 squares is +12.5% tokens and +27% attention cost |
| Global | **16** | Doesn't scale with N — effectively free. Be generous |

Global registers hold state belonging to no single board: side to move, where the present
line sits, material across all timelines, timeline-war status.

### 5.3 Implementation notes

- Registers need the board's (timeline, time) coordinate so they know what they summarise,
  but **must not** receive a (file, rank). Under RoPE that means **position-free on the
  spatial pairs**: per-board registers are rotated on the timeline and time pairs only, and
  global registers, which belong to no board, are position-free on every pair (ruled at
  CHECKPOINT 6 — a learned "not a square" angle is a (file, rank) in disguise). Pairs
  involving registers get their own entries in the displacement bias (§4.1).
- **Exclude from the policy head.** Registers are not move endpoints.
- Under the §3.3 contingency, per-board registers double as the board's summary token for
  the inter-board pass.
- Honest expectation: 0.5–2% improvement, not a step change. Cheap enough to take anyway.
  First thing to ablate.

---

## 6. Move-graph attention bias

The move generator already computes exactly which squares reach which, across all boards,
at unbounded range. Feed that in as an attention bias or hard mask on a subset of heads.

This converts the exotic geometry from something the model must learn into something it is
handed. Cheap, and probably worth more than extra depth.

**Allocation:** 2 heads with hard move-graph masks, remainder dense. Pieces interact via
more than legality — pins, defended squares, pawn structure — so full within-board
attention stays on top.

#### Fixed multi-head routing *(speculative — see caveat)*

Nothing requires the masked heads to share one mask. All of the following are computed by
the move generator already, cost no parameters, and need no router:

| Head | Relation |
|---|---|
| 1 | Legal move graph |
| 2 | Same board only |
| 3 | Same (file, rank) across boards — the time/timeline axis |
| 4 | Attack/defend relations |
| 5–8 | Dense |

FlexAttention's `score_mod` takes a head index, so per-head masks are directly
implementable.

**As built — v1 values, alternatives are 6a arms** (CHECKPOINT 6): masks are symmetric; a
token always reads itself and padding is excluded; a register query on a masked head reads
the registers its row may see; "same board only" includes the board's per-board registers
as queries; the relations head carries every typed-edge type and follows the mask's
direction; under the routing table the move-graph mask occupies one head and the relations
head counts toward B9's two; per-head positional specialisation (§4.1) follows the routing
table's roles.

**Typed edge biases — the masks upgraded.** Head 1 and head 4 currently receive binary
masks: connected or not. Stockfish's NNUE threat inputs carry *types* (which piece
attacks which), and the transformer-native version is a small learned bias per head
indexed by relation type — **attack, defence, pin, x-ray, check** — so a head knows not
just *that* two squares are related but *how*. (A sixth type, travel-threat, was listed
without a definition; a travel threat is a cross-board attack, which the attack type
already carries, so it is folded in — CHECKPOINT 6.) The principle that sorts
where such information belongs: the NNUE needs relations as *inputs* because it has no
attention; a transformer puts relations in *attention* and aggregates in *embeddings*
(6p), and never as extra tokens (§14.4). Cost is one learned scalar per head per type,
shared across layers, every head carrying every type (as built); the edge list is the
generator's existing output; exact-source. Unlike the
NNUE case (§11.1) there is no separability constraint here, so cross-board 4D attacks are
included. Under the cached configuration these masks and biases apply to live query
rows only — they are present information — and frozen rows are computed without them
(BUILD_LOG issue 33). A check arrives at layer 0 as a typed edge rather than being inferred over several
layers. Tested inside ablation 2 alongside the routing table.

**Smolgen — in at board-pair level, refused at token level, and why the need is smaller
here than in Lc0.** Lc0's smolgen compresses all square representations into a global
vector, runs it through MLPs, and generates a full heads × 64 × 64 bias added to the
attention logits — position-dependent attention patterns from global context, worth
~"50% larger model at 10% throughput cost" for them. Its generator's output is T × T,
and that is the whole story at our scale: with T in the thousands, a ~256-dim generator
costs **8× the attention it modulates** — square tokens or piece-centric alike — because
the projection is larger than the matrix. Nor does going low-rank help: a rank-r
generated bias u_iᵀMv_j is *identical* to appending r context-modulated dimensions to Q
and K, at +r/d_head cost — rank 8 is +25% attention for a weak bias; rank 32 doubles it.
That is §4's d_head lever in a costume, not a new mechanism. Token-level: refused.

The same arithmetic kills the *within-board* level: generating full-rank per-board
biases from each board's register costs 8× that head's block attention, and made
low-rank it reduces to what the same-board head already gets by attending to its
board's register. Dropped. What survives is the **board-pair level**: full-rank biases
over board pairs generated from the global registers — negligible against the attention it
modulates, and *full-rank*, the freedom that makes smolgen smolgen. As built: the global
latent is read through a per-head table over *relative* board-pair cells, bilinear in
(latent, cell), with no absolute board index; pairs with a global register get no smolgen
bias; the board-descriptor bilinear form is the rank-32 alternative. (An earlier "14.7
MMAC per head per layer" was 256 × 240² — a 256-wide output over every pair — not the
generator's cost.) That is the version in this design: learned, context-dependent board relevance
on top of the exact-but-static typed edges, aimed at the frozen-board relevance gap
alongside §7.9.

Why Lc0 needs it more: Lc0's 64 tokens carry absolute square embeddings and no
generator-supplied relations, so smolgen is how the model learns *which squares matter to
which, given the position*. Here the typed edges inject exact relations, 4D RoPE and the
displacement bias give relative geometry (§4.1), and the routing masks fix structure — what remains for a learned
dynamic bias is context-dependent *relevance*, which is a board-level quantity anyway.
Ablation-gated alongside the routing table.

**Why this is interesting:** heads 2 and 3 are precisely the intra-board and board-axis
passes of the §3.3 axial contingency — so this **recovers axial's inductive bias without
axial's costs.** The axis decomposition is distributed across heads *within* a layer rather
than across layers, so there is no extra depth and no transpose between passes (companion §9.3).

**Caveat: this is a research bet, not a spec item.** Wrong priors waste head capacity, and
there is no precedent for it in board-game engines. Ranked in §13.1 below the window
measurement and the positional ablation, above most of the rest.

**Do not confuse this with learned block-sparse routing**, which is rejected in §14.2 for a
different reason — training a router to discover which blocks matter duplicates an oracle
the generator already computes exactly.

### 6.1 Kernel viability

**Token ordering is a prerequisite, not a detail.** Order tokens **board-major** — all 66
tokens of board 1, then board 2, and so on. Board-level blocks are then contiguous and
block-sparse kernels exploit the move mask losslessly. Square-major ordering (all a1 squares
across boards, then all a2) makes the *same logical mask* strided, and block-sparsity
degrades to gather-bound. Same mask, same model, very different kernel performance.

Given board-major ordering: earlier concern was that irregular sparsity goes gather-bound
and loses to dense. **That concern is overstated here**: the mask is naturally
block-structured at board granularity, since a 66-token board is roughly a block and
board-to-board reachability is a block-level relation. FlexAttention exploits block sparsity
losslessly with block-mask memory `O((n/block)²)` instead of `O(n²)`.

Benchmark block-sparse against dense rather than assuming either.

---

## 7. Attention scope — dense, or causal-and-cached

**Build dense.** This section describes the caching alternative, why it looked compelling,
and why it is deferred rather than adopted. Earlier revisions recommended the opposite.

### 7.1 The caching option

5D chess has a monotonically growing immutable history (§2.4), structurally identical to
the KV cache of an autoregressive LM. Per search node you would compute only the **live
head boards** — typically 2–15 boards, a few hundred tokens — attending into a cached K/V
for the entire frozen history.

Paper cost reduction: `T_total²` → `T_new × T_total`, about **30×**.
That assumes ~8 live head boards, and the ratio falls as timelines multiply: the cwmtt
corpus's real games carry 24 timelines at 240–399 boards, where the paper ratio is 9.9× at
240 boards (not 29×) and 24.8× at 600 (not 49×). The human games carry one to four active
timelines, where it is higher (BUILD_LOG issue 36).

**Realistic reduction: ~15–20×.** An earlier revision said ~10×, discounting for tree
divergence — but with delta storage, divergence costs memory, not compute: each node's
delta is computed once regardless of how the tree branches. The real discount comes from
skinny-GEMM efficiency at small query counts, ragged batching, and CPU-side tree overhead.
The figure already includes recomputing *every* live head board at every node (§7.3's
live-set rule) — not just the boards the move changed. It does not include §7.3's
supersession recompute, which at its upper bound (+12–24% per node) trims the range to
~12–18×.

### 7.2 Tree structure (applies only if caching)

MCTS explores a tree, and every node in a subtree shares history up to its branch point.
This is **RadixAttention** (SGLang) applied to a search tree instead of a conversation
tree: cached prefixes live in a radix tree, and a new node reuses the KV of its longest
cached prefix, computing only the novel suffix. Boards before the divergence point are
shared across all descendants.

The mapping is unusually direct. In LLM serving the radix tree is an *incidental* structure
discovered across independent requests that happen to share prompts. Here the tree
**is** the search tree — the sharing structure is known ahead of time rather than
discovered, so cache placement and eviction can be driven by MCTS visit counts instead of
LRU.

### 7.3 What it costs

To freeze superseded K/V you need **causal structure along the time axis**: superseded
boards attend only to their own past, never to live heads.

The mask is **block-causal**: causal *between* boards, bidirectional *within* a board. All
of a board's tokens are created together, so full within-board attention is compatible
with freezing in every configuration. Registers follow the same logic by direction —
**reading is free, being read is not**. Global registers are live (recomputed every node;
the value head pools from them), so they read the whole scope at every node for sixteen
queries' cost. Frozen tokens *reading* the registers is present-to-frozen information
flow, which is exactly what invalidates a frozen K/V: it happens once per game move at the
root refresh (§7.9) — the mechanism by which frozen boards learn current relevance — and
inside the search only through the refresh ladder. Per-board registers belong to their
board's block: frozen with it, and a cheap summary key for live tokens.

**The causal past is the board's ancestry** (ruled at CHECKPOINT 2): its own timeline's
earlier boards and, recursively, the timeline it branched from up to the branch point.
Two conditions decide it. The attention set must be *state-defined* (C20), and it must
be *closed under future moves* — no later move may add a board to it, or a frozen K/V
goes stale. Ancestry satisfies both. Time-coordinate order fails closure: a later travel
can branch a new timeline into the past and add boards earlier than an already-frozen
board. Creation order fails state-definedness: two move orders that reach the same
multiverse create boards in different orders. What ancestry cannot see — parallel
timelines — is what the root refresh supplies (§7.9). A frozen board reads itself and the
frozen boards of its causal past and carries only intrinsic features; current status
reaches it at score time (§4.1).

**Among live boards, attention is full.** The current node's head boards attend to each
other bidirectionally and are recomputed at every node. This is necessary, not optional:
cross-timeline threats run in both time directions — a head at a later time can give
check to an earlier head on another timeline — and §6's typed check edges between heads
only work if the mask gives them an attention path. Strict time-causality among heads
would mask out exactly the present-board interactions the game turns on. The price is
recomputing every head at every node: ~3.7× the per-node cost of recomputing only the
changed boards at 240 boards, ~5.6× at 600. §7.1's realistic figure already includes it.

**Boards superseded inside the search are recomputed, never snapshotted.** A board created
at one node and superseded deeper in the tree is neither root history nor live. While live
it attended to whatever else was live *on that path*; freezing that snapshot makes
evaluations path-dependent — two move orders reaching the same multiverse would carry
different frozen K/V. That breaks §8.5's zero-cost transposition evaluations, which
require an evaluation to be a function of the state and the root, not the path; and it
hands the net representations its state-defined training mask never produced. Rule: at
supersession, recompute the board once under the state-defined mask — its causal past
plus the conditioned history of its **refresh epoch** (the root's, unless a node refresh
above it forked a new one, §7.9) — and share it down the subtree via prefix sharing
(§7.2). Recomputing against the root epoch inside a refreshed subtree would silently undo
the refresh for exactly the boards that subtree creates.
Upper-bound cost: +12–24% per node at 240 boards, +8–16% at 600; counted under the
ancestry reading, 0.3–12% (the creation reading would cascade — 280 boards recomputed for
one jump at 600 boards — one more reason it was rejected; BUILD_LOG issue 35). The refresh
epoch's history enters as a fixed block, history boards reading only history, so the
union stays acyclic; whether a recompute reads the causal past alone or that plus the
epoch's history is built both ways and decided by 6c with the cache. Two cheaper shortcuts
are refused for the same reason (§14.4).

You lose the ability for a frozen board's representation to be shaped by what is currently
relevant about it — "this board matters because there's a queen aimed at it three
timelines over." The reasoning can still happen, but only *in the head-board tokens*, which
then carry the entire burden of cross-time reasoning.

**This cost is unquantified and is the largest open risk in the design.** Two partial
mitigations exist: §7.9's root-conditioned refresh, and §12.4's per-board auxiliary
targets, which train frozen representations to predict their own future relevance.

### 7.4 Why dense wins for the first build

1. **Dense is already playable.** ~337 evals/sec (§9.2), ~2,370 with the window (§7.5).
   At 800 sims/move — AlphaZero's
   headline configuration against Stockfish 8 — that is 2.4 seconds per move. A real
   engine at long time controls.
2. **It deletes the largest unknown.** §7.3's cost cannot be estimated from first
   principles. Dense removes the question instead of betting on the answer.
3. **It deletes most of §10.** No paged allocation, no radix tree, no cascade inference,
   no copy-on-write. For a small team that is likely months of the most bug-prone work in
   the project.
4. **Full MHA is correct anyway.** GQA saves only 1.2% of dense cost and shrinks the key
   space on the axis §13 flags as marginal (§10.2), so this is not a concession made to
   get dense — it is the right choice independently.
5. **You cannot evaluate the trade without it.** The causal constraint's Elo cost is only
   measurable against a dense baseline. Building cached first means never learning what it
   cost you.

### 7.5 The window relative to the cache

**Correction to an earlier revision of this document, which claimed these were substitutes.
They are not — they compose.** The window shrinks `T_total`; the cache shrinks `T_new`.
They act on different factors of `T_new × T_total` and multiply:

| Configuration | TFLOPs | evals/sec | vs dense |
|---|---|---|---|
| Dense, full history | 4.78 | ~337 | — |
| Dense + window (80 boards) | 0.68 | ~2,370 | 7.0× |
| Cached, full history | 0.16 | ~10,100 | 30× |
| Cached + window | 0.07 | ~23,700 | **70×** |

So the window does not make the cache redundant, and the cache remains available on top of
it later.

**The dense recommendation survives, but for a different reason than previously stated.**
Not "the window replaces the cache" — rather: **dense + window is already fast enough.**
At ~2,370 evals/sec, 800 sims/move is 0.34 seconds. At that point the cache's added
complexity (§10.3) and unmeasured Elo cost (§7.3) are not worth paying up front. They
become worth revisiting only if throughput turns out to bind, and §12.1 gives the strongest
case that it might.

Note also that the window's advantage is **7×, not the 9× stated earlier** — 9× is the
attention-only figure, but FFN and projections are linear in T and drop only 3×.

### 7.6 The hybrid, if you later need it

If node count proves to bind strength more than net quality does, the fallback is not full
causal masking but a hybrid:

**14 causal layers that cache + 2 dense bidirectional layers that do not.**

Attention: `14 × 0.0086 + 2 × 0.257 ≈ 0.63 TF`; with FFN and projections the total is
~0.74 TF. Either way **6.5× over dense** — the linear terms inflate both sides by ~16%, so
the ratio is robust to the earlier attention-only framing.

**Placement is forced, not untested.** An earlier revision framed this as an interleave in
the 3:1 pattern of linear-attention hybrids (Qwen3-Next, Kimi Linear) and called placement
an untested instinct. Both wrong. A bidirectional layer at depth k lets present
information into frozen tokens at layer k, so every layer above k changes per node and
must be recomputed: **the cache survives only below the first bidirectional layer.** The
0.74 TF figure assumes the two dense layers sit on top. In the linear-attention hybrids
every layer is causal, so interleaving invalidates nothing — the analogy does not carry.

| Configuration (240 boards, per node) | TF | vs dense | vs pure cached |
|---|---|---|---|
| Dense (v1) | 4.78 | 1.0× | 29× |
| Bidirectional every third layer (3, 6, 9, 12, 15) | 3.05 | 1.6× | 18.6× |
| 2 bidirectional at depths 8 and 16 | 1.86 | 2.6× | 11.4× |
| 5 bidirectional on top | 1.61 | 3.0× | 9.8× |
| **14 cached + 2 bidirectional on top** | **0.74** | **6.5×** | **4.5×** |
| Pure cached | 0.16 | 29× | 1.0× |

At equal count, interleaved bidirectional layers cost roughly twice what the same layers
cost on top. And the within-search benefit they would buy — drift since the root — is the
refresh ladder's job (§7.9), at ~5–12% per search rather than ~19× per node.

#### The interleave that is cache-compatible: local/global layers

A different interleave does survive caching, because it runs on a different axis:
**within-board-only layers, with a cross-board (causal) layer every third** — the
local/global pattern Gemma 2 and 3 use, sliding-window local layers with periodic global
ones, chosen there for KV-cache memory. No present-to-frozen flow is introduced: the
cross-board layers stay causal. Measured against the same cost model:

- **Compute:** ~0.067 TF per node against 0.164 for pure cached — 2.4× cheaper — and 2.4×
  on the dense v1 configuration (1.96 vs 4.78 TF).
- **Memory:** frozen boards need K/V only at the global layers — the 600-board endgame
  cache drops from 649 MB to ~203 MB.
- **Cost:** cross-board reasoning gets 5 of 16 layers of depth; multi-hop cross-time
  combinations need hops.

It is the *layer-level* factorisation of §6's *head-level* routing: routing keeps
cross-board reach at every depth but saves less compute and less KV memory. So it is the
natural fallback if routing fails ablation 6a, and worth its own arm if cache memory binds.
Committed-class — the attention pattern sets the training distribution.

**§7.9 likely supersedes this.** Root-conditioned refresh recovers most of the same
bidirectional benefit for ~4% overhead, where the hybrid pays 4.6× over pure cached
(0.74 vs 0.16 TF). Try §7.9 first; the hybrid earns its cost only if *leaf*-level
conditioning — not just root-level — proves to matter.

### 7.7 This is a training-time commitment

**You cannot train dense and mask causally at inference.** The distribution shift would
wreck it. That is the *zero-shot* switch, and it stays refused. The untested middle path is
a fine-tuning phase with conditioning-offset sampling (§7.9) — the familiar pattern of
adapting a pretrained model to a new attention regime, as context-length extension does —
which would soften the commitment, not remove it. Measure it before relying on it. The decision must be made before serious self-play compute is spent — the
strongest argument for running the window measurement (§10.4) early rather than deferring
it.

### 7.8 Roofline note

Dense, the query is the full sequence, so you are firmly compute-bound and ridge points are
irrelevant. They matter only in the cached regime, where attention arithmetic intensity ≈
query length drops to a few hundred: **H100 ≈ 295, B200 ≈ 563 FLOP/byte**, putting you
compute-bound on Hopper and on the knee on Blackwell.

Related: **batching does not rescue you the way it does LLM serving.** At 15.8k tokens a
single position already saturates the GPU, so leaf batching buys ~1.2–1.5×, not the
order-of-magnitude it gives in decode-bound workloads.

---

### 7.9 Root-conditioned cache refresh

A refinement that substantially reduces §7.3's cost. **Not currently in the build** — it is
meaningless while dense — but it changes the economics of migrating.

#### The mechanism

Once per actual game move, run **one dense bidirectional pass over the whole history**,
conditioned on the current root position. Cache the resulting K/V. Every node in that
search tree then attends into history representations that already know what is currently
relevant — "this past board matters because there's a queen aimed at it right now."

#### Why the root, and not per node

Cache sharing is the binding constraint. A refreshed representation is reusable only by
nodes sharing its conditioning. Refresh per leaf and every node needs its own copy — the
cache is gone and you have rebuilt dense at higher complexity.

Refresh at the root and the entire tree shares one refresh.

| | Cached/node | Root refresh | 200 sims | 800 sims | 3200 sims |
|---|---|---|---|---|---|
| Full history | 0.159 TF | 4.78 TF | +15.0% | **+3.8%** | +0.9% |
| Windowed | 0.068 TF | 0.68 TF | +5.0% | **+1.3%** | +0.3% |

What remains lost relative to full dense is conditioning on each *leaf's* frontier, which
differs from the root by a handful of plies. That is a far narrower loss than the full
causal constraint.

#### How it is trained — conditioning offsets

The refresh pass itself is the ordinary dense model: at the root the two-level mask
reduces to full dense attention, exactly what dense training teaches. But evaluations
*below* the root see a regime dense training never shows — history conditioned on an
ancestor k plies up, live boards conditioned on the node itself. It is neither dense nor
plain causal, and §7.7's rule applies to it: a net cannot be trained on one mask and
deployed on another.

**Conditioning-offset training** closes the gap. For each training position, sample an
offset k from the depth distribution the search actually produces, k = 0 (fully dense)
included; run the refresh pass on the position k plies earlier in the game record;
evaluate the training position with the cached mask against that refreshed history.
Targets are unchanged — the position's own visits and outcome. The game-record ancestor
stands in for the tree ancestor: both are "history conditioned on a position k plies
back", which is the whole of what the regime is. Cost: one extra dense pass per training
sample; training is ~10% of the budget, so at most ~+10% total. It is committed-class
with the cached decision itself (§7.7). Plain causal needs none of this — its history
attends only to its own past in training and in search alike — so ablation 6c's refresh
arm must be offset-trained, or it compares a train/deploy-consistent causal net against an
off-distribution refreshed one and is biased against the refresh.

#### Selective refresh at internal nodes — a ladder, not a verdict

The *uniform* version — refresh at internal nodes generally — has bad economics: a
windowed refresh costs ~0.68 TF against ~0.068 TF per cached node, so **one refresh ≈ ten
node-evaluations**, and breakeven needs a subtree that will consume well over ten visits
*and* conditioning that has actually drifted. Intermediate nodes differ from the root by a
few plies, so for physical moves the drift is small and the root refresh is nearly as good
at depth six as a depth-six refresh would be. That argument stands. Targeted versions
change the arithmetic, and one 5D fact changes the premise:

**Travel drift is discontinuous.** A physical move shifts which old boards matter slowly;
a travel move inside the search creates a new timeline whose head sits in an old era, and
that era's frozen boards become live context — boards whose representations were
conditioned on a root where the timeline did not exist. Physical-move drift is continuous;
travel drift is a step. The refresh worth having is aimed there.

| Rung | Trigger | Detects | Cost |
|---|---|---|---|
| 0 | Root only | Nothing within a search | — |
| 1 | Travel move + visit gate (V ≈ 50) | The one *known* discontinuity, predicted from the move type | ~10 node-evals, whole scope; ~5–12% per search at plausible thresholds |
| 2 | **Probe-measured drift** — refresh a sampled handful of frozen boards, measure representation delta, fire the full refresh if it clears a threshold | Any drift, whatever caused it | ~0.5 node-evals to decide; ~10 if it fires |
| 3 | **Attention-weighted partial refresh** — a priority queue of frozen boards by (attention mass from the node's live tokens) × (staleness tag: conditioned before a structural event touching that board's era), refreshed within a per-search budget | Drift on exactly the boards this node uses | Linear in boards refreshed |

Depth-since-last-refresh alone is dominated: depth does not imply consumption. The
principled rule is **staleness × consumption** — §8.6's value-of-information frame applied
to conditioning — and rung 3 is its per-board form, the same floor-and-budgeted-queue
shape as §10.4.3's scope rule. Rung 2's threshold tunes itself online: track the mean
|Δvalue| that refreshes actually produce, tighten when they keep paying, loosen when they
don't — a within-system inference statistic (§14.4).

**Fork, never overwrite — except at the root.** The root refresh replaces the
frozen-history cache outright: every node in the search descends from the root, so every
node should see its conditioning. A node refresh is **copy-on-write**: the refreshed
node's subtree gets new K/V — the whole scope at rung 1, only the refreshed boards at
rung 3, the rest shared with the parent epoch — and siblings keep the parent epoch.
Overwriting in place would be a bug: sibling subtrees would attend into history
conditioned on a position they do not descend from, so their evaluations would depend on
whether and when a refresh happened elsewhere in the tree — on search order and thread
timing. Nondeterministic and silent. Each node carries its epoch; the next root refresh
starts a fresh baseline, and carried nodes keep their statistics with their K/V
invalidated (§8.7).

**What the targeted rungs cost that root-only does not.** Each refresh forks the frozen
K/V for its subtree (per subtree at rung 1, per board at rung 3), so epochs must be
bounded per search and they die with §8.7's garbage collection. Memory per fork scales
with what it refreshes — ~113 MB windowed or ~260 MB full history for a whole-scope fork
at 240 boards, a few boards' worth at rung 3. Evaluations become a
function of (state, refresh epoch), which chips at §8.5's zero-cost transposition hits
across epochs — accept the mixing and let residual backup absorb it. Rung 3 adds per-board
epoch tags and attention sets that mix conditioning board by board: a finer version of a
mix the cached configuration already has (root-conditioned frozen boards beside
node-computed live ones), so not categorically new, but it widens §7.3's train/deploy gap
slightly and enlarges the laziness-interaction surface (companion §8.5 class). On the
training side, nothing new — *given* conditioning-offset training (above): the refresh
pass treats the node as root, and a node refresh only resets the offset to 0, so every
offset up to the trained maximum is already in distribution. Without offset training this
claim would be false.

**Sequencing — the rule that matters more than the rung.** Every rung delivers the same
benefit at decreasing cost, so **rung 1 is the existence test and rung 3 is the efficiency
version**: if travel-triggered whole-scope refresh shows no Elo in 6c, the flexible form
cannot either — there is no benefit to deliver more cheaply. Build rung 1 as the 6c
sub-arm; adopt rung 3 as the design target only if rung 1 shows a signal. Do not engineer
the efficient form of a benefit that has not been shown to exist.

**Refresh as a verifier — the conditioning audit on promotions.** A node whose Q jumps
past the incumbent by n is a *value* event, not a drift event: most such jumps come from
ordinary tactics on live boards, where conditioning has not moved, so a value trigger is
a poor drift detector. It is a good *verifier*: a surprisingly strong line may owe its
value to stale conditioning — frozen boards that do not yet reflect a threat the new
position created — and refreshing the node before it takes the PV checks exactly that.
It completes the PV-transition checks of §11.2 rung 2: the scanner audits tactics on the
promoted line, the demotion audit re-examines the abandoned one, the refresh audits
conditioning on the promoted one. Trigger on the node's Q, never a single leaf eval
(leaf evals are too noisy). Asymmetric by construction: it catches over-valuation from
staleness and never under-valuation, because a good line hidden by stale conditioning
produces no jump to fire on. Gated on 6c like every rung.

**Complexity as a trigger** (a ladder variant, CHECKPOINT 4): refresh more in complex
positions — board count, active timelines, travel since the last refresh, or the effort
head's predicted difficulty. The signal must exist *before* the evaluation it decides:
the position's structure, or the parent's outputs. A trigger driven by the network's own
uncertainty belongs to the confidence family and is add-only. It needs no new training
commitment — conditioning-offset training already covers any refresh frequency — and it
waits on rung 1's existence test like the rest of the ladder. What it targets is
visibility, not checking: more dense attention is the same network looking harder, so
its confidently wrong spots stay confidently wrong.

As built (behind a flag, off): a weighted score against a threshold, gated by rung 1's
visit gate and capped at 8 refreshes per search, never at the root; "travel since the last
refresh" counts every travel move from the node's epoch anchor down to the node; the
effort signal is the parent's predicted difficulty at a quantile; rung 1 is decided first
and unchanged, and every signal only adds. Every number is a placeholder, tuned in 6c if
rung 1 shows a signal (BUILD_LOG issues 75–79).

#### A cost not in the table: subtree reuse across moves

Lc0-style engines reuse the subtree under the played move as the next search's starting
tree. Per-root refresh conflicts with that: deltas computed during the previous search were
conditioned on the *previous* root's refresh, so reusing them under the new refresh mixes
conditionings. **Resolved in §8.7:** a node's search statistics (visits, Q, bounds,
priors) are mildly stale and self-correct under fresh backups — carry them; its KV deltas
are genuinely poisoned — invalidate them and recompute lazily on next visit. The 20–50%
node savings survives; the refresh's residual cost here is lazy recompute on the revisited
fraction, not a forced choice between reuse and correctness.

#### It is a training-time commitment

This is a two-level mask, not an inference trick: history tokens bidirectional with respect
to the root, causal with respect to leaf-specific moves. It must be trained that way (§7.7).

#### Effect on the dense-vs-cached decision

§7.3's unquantified cost is the main argument against caching. Root refresh shrinks it to
something much narrower for ~4%. Combined with §12.1's self-play compute argument, this
makes the cached configuration **more attractive than §7.4 prices it** — the recommendation
to build dense first stands on sequencing grounds (§13.1), not on cached being wrong.

---

## 8. Output heads

### 8.1 Policy — bilinear pointer head

The output space is variable, so fixed logit planes don't work.

```
score(move) = f(h_src)ᵀ g(h_dst)
```

Evaluated only over generator-supplied legal moves. Handles arbitrary N for free.
`f` and `g` are separate learned projections — source and destination roles are not
symmetric.

**Promotions** add a learned per-piece logit offset, because the bilinear form gives every
promotion piece the same score (Chessformer adds a promotion bias the same way;
CHECKPOINT 6). **Which moves are scored:** those the rules engine has not proved illegal,
each marked legal or unknown by its sound tiers under a budget. Search verifies an unknown
move the first time it selects it, never lets one feed a proof, and scores a node terminal
only on an exact verdict (BUILD_LOG issues 46, 90).

### 8.2 Multi-move turns

A turn may require moves on several boards. Two options:

- **Autoregressive within the turn** — condition each board-move on the previous ones.
  More accurate, slower.
- **Each board-move as its own ply** — simpler, integrates cleanly with MCTS, but loses
  intra-turn coordination.

Undecided — two further arms below (CHECKPOINT 6); ablation 6x decides. One new coupling from §8.5: the ply-split option manufactures
factorial intra-turn transpositions, so choosing it effectively mandates graph search —
benchmark the pair, not the ply question alone.

**Legality under the ply split** (§11.0): check midway through a turn is legal — only the
submitted turn must leave every royal safe — so a mid-turn node in check is *not*
terminal, and a partial turn with no legal completion is a dead end to prune, not a loss.
Terminal detection runs only at turn boundaries. **Prior art:** ftxi's 5dchess_engine
already implements the ply split — its semimove "fine tree" builds one timeline's share of
the turn at a time, using the hypercuboid algorithm to keep the search tree's branching
factor from exploding exponentially with the number of timelines. Evidence for 6g and 6k,
not a decision.

**Turns have variable length, so submitting is an action** (Q4, ruled at CHECKPOINT 2).
A player may keep moving on their remaining playable boards after the present has
shifted; the turn ends only when they submit. So once the present has shifted, every
decision point offers "submit" alongside further optional moves. Under the ply split it
is an explicit action with its own policy logit, read from the pooled global registers as
the value head is — no single register is assigned a role, so D7 holds — and legal only
when the submission passes the legality test (§11.0 item 2). Moves and submit share one
distribution, and an illegal submit gets no probability. Under whole-turn generation the
turn generator ends turns itself (below). The network also needs a global **acting-player**
input: per-board side-to-move flags cannot say whether the player has already submitted.
Committed-class — it changes the policy head and the inputs.

#### Evaluating at turn boundaries — the turn-construction arms (CHECKPOINT 6)

One evaluation already scores every legal move on every board: that is what the pointer
head does (§8.1). The cost of multi-board turns lies in what happens next. Under the ply
split, every node inside a turn gets its own network evaluation, so a turn with k
mandatory boards costs k evaluations along every path and the same budget sees k times
fewer turns ahead. The whole-turn arm as first built avoided that but enumerated turns and
capped the list at 256 per node (BUILD_LOG issue 93): at a multi-board present the cap, not
the priors, bounds which turns search can ever see.

The fix is to evaluate once at the turn start, assemble the combination from those priors,
and evaluate again only when the turn is complete. Two forms, both D2 arms:

- **Tree form.** Keep the submove tree and PUCT. Inner nodes inherit the turn-start priors
  for the boards still to move; a simulation runs through the turn to a complete one and
  evaluates that. The network re-evaluates mid-turn only where a move interacts with the
  boards still to move — a move between them, a capture of a piece that could reach them,
  a change in check, a shift of the present. Resolving boards in a fixed order (ftxi's fine
  tree) also removes the k! orderings of one turn.
- **Flat form.** A generator assembles complete legal turns best-first by summed logits,
  through the rules engine: a move between two present boards resolves both; a rewinding
  branch makes the rest optional; optional boards carry a "leave it" option with its own
  prior; illegal submissions are skipped as they appear. The hypercuboid search already
  walks this product space for mate detection (§11.0 item 3); ordered by prior, it becomes
  the turn generator. PUCT runs over the generated turns, and progressive widening pulls in
  the next-best legal turn as visits grow. This replaces the enumerating whole-turn arm and
  its cap, and ends turns itself, so no separate end-of-turn token is needed.

Prior art for both: naïve MCTS for combinatorial action spaces (Ontañón), which chooses
which combinations to try from per-component estimates, and Sampled MuZero, which plans
over a sampled, growing subset of a large action space.

What is lost: the network no longer conditions one board's move on another's inside a
turn, except at the tree form's interaction points; coordination comes from evaluating
complete turns. And the flat form's policy targets are per-board visit marginals, so its
policy normalises per board rather than across all moves — a head change, which is why D2
is committed-class and is decided before serious training.

**Wider search where fewer boards must move** is a widening-schedule parameter (6g) — the
exponent and k₀ may depend on the mandatory-board count — tuned rather than designed;
under the flat form, small turn sets are covered fully anyway.

**Measure first.** How many turns have more than one mandatory board, and how many optional
moves are played, per board-count range, on the human games — rules engine only, hours.
Under 50 boards the human games' median position has one active timeline, so most turns
there are single moves, where every arm is identical; the arms differ only where turns are
wide.

**Prediction on record:** both turn-start-only forms beat the evaluate-every-node ply split
at equal wall-clock once turns have three or more mandatory boards, and the tree form beats
the flat form where boards interact.

### 8.3 Value head

Pooled from global registers, and **categorical (distributional) rather than scalar** —
BT4-style bins over the outcome. Still exact-source (the target is the game outcome), and
the distribution's variance reads off directly — which is what §8.4's variance-scaled
cPUCT consumes, so no separate uncertainty head is needed. Keep at higher precision than
the trunk if quantising (§10)
— evaluation discrimination between +0.3 and +0.5 is exactly what quantisation error eats.
Value and WDL are from the acting player's perspective (D8); pooling over the global
registers is a mean.

### 8.4 Selection at high branching

The policy head (§8.1) exists to narrow search, and at 5D branching that narrowing is a
**viability precondition, not a tuning choice**. The arithmetic: 800 sims against ~1,000
legal moves is a mean of under one visit per child. Standard PUCT with every edge
instantiated spends the whole budget giving children one visit each — a one-ply bush,
single-sample value backup, search adding roughly nothing over the raw policy. Every
node-count argument in this document (§7.4, §12.1) presupposes that sims convert to depth;
this section is what makes that true.

**Mechanism: top-k instantiation with entropy-adaptive progressive widening.**

- Instantiate the top-k children by prior; widen as visits accumulate (`k(n) = k₀·n^α`,
  standard progressive widening).
- Set k₀ per node from the **policy entropy** — already computed with every softmax, so
  free. Sharp distribution → narrow hard; uncertain → auto-widen. One mechanism serves
  the flat early-training policy and the sharp late one with no schedule to tune, and it
  self-adjusts at match play when the engine leaves familiar territory.
- FPU reduction on uninstantiated moves, so the widening frontier is explored
  pessimistically rather than optimistically.
- Root Dirichlet noise during self-play, as standard.

**Complements, not substitutes: variance-scaled cPUCT and uncertainty-weighted backups**
(KataGo, ~+75 Elo combined in Go). These act on the exploration constant and backup
weighting; entropy-adaptive widening acts on how many children get instantiated —
orthogonal levers, run all three. The node-level variance they need reads directly off the
categorical value head (§8.3). One expectation to log against ablation 6i: uncertainty
weighting and the demotion audit both guard against a single noisy deep eval flipping the
search, so expect the demotion-refutation rate to drop when this lands — overlapping
defences, not audit failure.

**Per-child variance bonus — the exploration mirror of §8.6's LCB rule.** LCB commits
by Q minus a few standard errors, so a promising-but-noisy child loses the vote until its
bound tightens, and its bound tightens only with visits. Without preferential visits the
penalty is self-fulfilling: the noisy candidate never gets the evidence that would let it
win. So search carries a **bonus** on high-variance children — bandit precedent UCB-V /
UCB1-Tuned — and the two mechanisms are one design: **resolve, then decide.** Each is
unsafe alone: bonus without LCB commits to noise, LCB without bonus buries novelty.

- **Multiplicative on PUCT's U term, capped — never additive.** At branching ≈ 1,000 an
  additive variance term could resurrect zero-prior moves; the policy prior stays in
  charge of what gets looked at.
- **Prior-to-posterior variance.** The categorical head (§8.3) supplies a variance
  estimate from the child's first evaluation — the prior — which blends into the
  empirical variance of backed-up values as visits accumulate. Classical UCB-V has only
  the latter and needs many samples; this design has a variance estimate at visit one.
- **Coherent with uncertainty-weighted backups**, not contradictory: uncertain values
  count less toward Q *while* search is directed to confirm them — active learning, not
  a tug of war.
- **Uninstantiated moves have no variance.** For the widening frontier the only handle
  is the parent's volatility, which the node-level scaling above already uses.
- **Confidence-family**, so the two-family rules apply (§12.3): a confident-wrong child
  has *low* variance and gets no bonus — that is the verification layer's and the
  league's job (§11.4) — and the
  categorical variance it consumes is under calibration monitoring (§14.4).

Precedent is thinner than the rest of this section: bandit theory supports it, but
KataGo and Lc0 have not adopted per-child bonuses, plausibly because priors dominate at
high branching — the same reason the multiplicative guard exists. Prototype tier;
expectation modest alone and larger in combination with LCB, which is how 6t measures
it.

The §12 curriculum over N is automatically a curriculum over branching — small-N early
games have far fewer legal moves — so coverage is best exactly when the policy is worst.
Previously an accident of the design; now a stated property of it.

**Narrow hard, because the auditor is watching — where one is.** Aggressive narrowing was
argued safe because the scanner would search the generator's full move output regardless
of what the tree instantiated (§11.2 rung 3). Verification v3 (§11.4) keeps that only in
part: under 50 boards the settling scans still cover the root and PV changes, and Hunt
(adopted under 100 boards) catches our own forced mates; Guard was adopted in no range, so
nothing checks the chosen move against a forced refutation at match time. Above 100 boards
the premise is weak, and robustness there rests on the offline audit lane and the
adversarial league — narrowing is a bet that they catch what the policy misses.

**Coverage gradient for a discarded move.** PV lines get full MCTS depth. Under v3,
discards get exact coverage only where the settling scans and Hunt run — under 50 and
under 100 boards — because the per-node sweep of discards (rung 3) and the NNUE quiet
screen were retired with the scanner (§11.4). Everywhere else a discarded move is covered
statistically, by widening and root noise, and offline by the audit lane. Stated as the
honest limitation in §13.

**Constants as built — placeholders, tuned by SPSA (G7):** E1's top-k, widening exponent,
FPU (applied to unvisited instantiated children) and root Dirichlet α; E2's and E3's
formulas; E5's LCB at z = 3 with a 15% visit floor and categorical weight 0; E9's visit cap
of 8 on solved children and secure child as Q − A/√n with A = 1; the whole-turn prior as the
sum of single-move logits (the mean behind a flag). Each is a named parameter (BUILD_LOG
issues 80–85).

**Coupling:** the §8.2 decision (board-move-as-ply vs whole turns) changes per-decision
branching ~5×, so the selection knobs and that decision are measured together (ablations
6g and 6x), not settled in sequence.

### 8.5 Transposition handling in the main search — MCGS

Previously unspecified; specified now as a **prototype-tier commitment**, with one hard
coupling.

5D produces transpositions structurally: moves on different boards within a turn largely
commute, so different orderings reach identical multiverses. The field has converged the
same way — KataGo ships graph search by default, Lc0 has an opt-in DAG search, and
published MCGS results show ~+69 Elo in chess with larger gains in higher-branching
variants.

**The coupling: under §8.2's board-move-as-ply option, MCGS is near-mandatory.**
Ply-splitting manufactures factorial sibling transpositions by construction; a plain tree
spends visits re-searching permutations of the same turn. If §8.2 lands on ply-split, the
tree-vs-graph question is effectively answered.

**The synergy the published figures don't include:** evaluations here are functions of
*state*, and both the eval cache and (in the cached configuration) the KV cache are
state-keyed — so a transposition hit is a **zero-cost network evaluation**, not just
shared visit statistics. MCGS is worth more in this design than in the engines it comes
from.

Mechanics: key nodes with the §11.3 5D-Zobrist scheme — already specified; reuse it — and
use residual-correction backup (Czech et al.) to avoid value double-counting through
shared nodes. Node identity adds two things the position hash lacks: each timeline's branch origin —
under the ancestry reading of the causal past (C23), two states that differ only in a
branch origin evaluate differently — and whether a turn is in progress. A turn start and a
mid-turn state can share every board and the acting player; two reviews found graph search
merging them and pruning a legal mating submission. The network tells them apart through
the turn-state inputs (§2.3).

Graph-history interaction does not arise. 5D has no repetition rule, and the full state
cannot recur within a game — passing is impossible and every move creates a board
(§11.0 item 5) — so values depend only on the state (and the refresh epoch, §7.9). The
one addition, self-play's turn cap (§12.3), is defined from the state, so it adds no path
dependence. (An earlier revision held this open as the design's most load-bearing
unresolved item.) Ablation 6k, measured jointly with 6g.

### 8.6 Time management — when the move commits

The unifying frame: every stop/extend signal is a proxy for **value-of-information versus
opportunity cost** — stop when further search cannot change this decision or is worth
less than the moves it is taken from. Signals act on the soft bound; the hard bound is
inviolable.

**Allocation: the moves-left integral.** 5D has a property no other engine's game has —
**deterministic state growth** (every board-move adds a board), so N's future trajectory
is known given the moves-left head, and cost is a known function of N (§9). Allocate this
move's share as its cost over the summed cost of all remaining moves, not 1/moves-left:

| At N=100, ~100 board-moves remaining | Value |
|---|---|
| Per-eval cost, now → final move | 1.0 → 3.4× |
| Naive equal-time share for this move | 1.00% |
| Equal-sims (integral) share | **0.48%** |
| Flat wall-clock leaves the last 20 moves | ~3.1× under-simmed |

Flat allocation starves exactly the phase where the multiverse is most tangled — engines
that estimate nps empirically must assume it persists; this design knows it will not.
Caveat: moves-left is a global target that dilutes with N (§12.4's own granularity
point), the noisiest input in the allocator — give it a conservative floor.

**Difficulty-weighted allocation — the effort head (§12.4).** Every signal below is
*ex post*, observed during search, and the integral above assumes equal sims per move. A
**predicted search effort** — visits until the decision stabilises — lets the allocator
plan **sims proportional to predicted difficulty**: know before the search starts that
the upcoming move is hard, and fund it from the moves that will not be (Stockfish's
complexity-scaled time is the crude precedent). The head is **distributional, not a
point estimate** — categorical over log-visit bins with a censored top bin, "did not
converge within budget" — because convergence labels are stochastic and heavy-tailed,
and searches stop at the cap. That one head yields the mean, the spread, any quantile
and the non-convergence probability; separate confidence and bound outputs are
unnecessary, the same lesson as §8.3's categorical value head. The allocator plans to an
**upper quantile** (tunable, 6v) — funding the mean under-funds half the time — and
treats high censored-bin mass as a *won't-resolve* flag: plan for the hard bound and
lean on LCB and the root-rule scan rather than chase convergence. Labels come only from
full-search moves — PCR's cheap-cap searches never converge and yield none. Three
guards: the prediction sets the plan and the ex-post signals correct it (a wrong "easy"
call still extends on instability or overlap); no move is planned below a hard floor of
evaluations regardless of predicted ease; the wall-clock maximum and the root-rule scan
are untouched. Confidence-family: it redistributes the soft budget, never the
confidence-independent layer. Expect the gain to concentrate in the late game, where
extension hits its cap. A second consumer: the refresh ladder's complexity trigger
(C24, §7.9) reads the *parent's* predicted difficulty at internal nodes — off-label, since
the labels are root-search efforts; confidence-family and add-only.

**Stop and extend signals:**

| Signal | Direction | Mechanism |
|---|---|---|
| Visit gap between top two moves unbridgeable in remaining budget | **Stop** | Lc0 smart pruning — the decision is mathematically settled |
| Root visit distribution stops changing | Stop | KLDGain pattern |
| Root solved | **Play immediately** | A proven result outranks visits |
| Top-two **distributional overlap** (categorical head, §8.3) high *and* KLD still moving | Extend | Genuine contest. Overlap alone can mean equivalent lines (transpositions), so churn is required too |
| Volatility from **either source** — net variance / §12.4 volatility prediction, or a proof from Hunt or the settling scans that contradicts the net (under 100 and 50 boards; the per-leaf estimate channel was retired, §11.4) | Extend | Dual differently-wrong sources; **disagreement extends hardest** — net-calm-but-prover-alarmed is §12.3's confident-wrong signature at the worst possible moment |
| Falling eval trajectory | Extend (asymmetric) | A rising eval does not extend |
| Best-move instability | Extend | Stockfish pattern |

**Guards:**

- **Extensions are denominated in evaluations; the hard maximum is wall-clock and held
  tight at high N.** nps collapses in big positions, so a "think longer" spiral at 500
  boards eats the clock — the extension mechanism must never be able to flag itself.
- **The §11.1 root-rule final scan sits inside the hard bound**, not after it; its
  escalated budget is part of the move's allocation, with a reserve held for one
  re-search if that scan refutes.

**Which move, once we stop: LCB with a visit floor.** The final-selection rule was
previously unstated except for solved roots (secure child, §11.1). Stated now:

- **Baseline: select by visits, never by Q.** Visits are the evidence measure — in MCTS
  each simulation adds one node, so subtree size under a child *is* its visit count —
  and a late-arriving appealing option cannot win the vote without sustained allocation.
- **Rule: among children clearing a visit-fraction floor (~15% of the max-visit child),
  select by the lower confidence bound of Q** — Q minus a few standard errors of the
  backed-up values, the error shrinking with visits (KataGo's selection rule). A high-Q,
  thin-evidence child has a wide bound and loses to a slightly lower-Q incumbent with
  deep evidence.
- **The penalty is selection-only, by design.** PUCT during search keeps its
  exploration term — penalising uncertainty *during* search would suppress the
  exploration that discovers the new option in the first place. Its exploration mirror is
  §8.4's per-child variance *bonus*: resolve, then decide. Self-play sampling from the
  visit distribution at temperature is untouched; LCB applies at match play.
- **Optional second uncertainty term:** the categorical head's variance at the child's
  node (§8.3), so the bound reflects both how much the search disagreed with itself and
  how unsure the net was — dual-source, ablation 6t.
- **MCGS caveat (§8.5):** transposition sharing lets a child inherit subtree statistics
  from other paths; use visits counted along the path as the evidence measure, not the
  shared node's total.

Together with the instability extension above and the root-rule final scan, this closes
the "switched to something appealing, stopped, hadn't looked deep enough" failure from
three directions: the switch extends the search, the chosen line is scanned exactly
beyond tree depth, and thin evidence cannot win the vote. A raw *depth* factor is
deliberately not among them — see §14.4.

### 8.7 Persistence across moves — carry, re-root, invalidate

The design's immutability spine means most engine state already carries across moves by
construction; this section specifies the rest, and resolves an either/or §7.9 left open.

#### What carries by construction

| Component | Lifetime | Why |
|---|---|---|
| Solved store (proofs) | Whole game | Game-theoretic position facts — §11.3 exports them from the lossy TT for exactly this |
| Asset map | Whole game | Superseded boards are immutable (§11.2 rung 4) |
| Scanner TT | Across moves | State-keyed; standard practice |
| NNUE accumulators | Frozen (superseded) / incremental (heads) — held open with the NNUE | §11.1, §11.4 |
| KV cache, frozen boards | Across moves in the cached configuration | §2.4 |
| Evaluation cache | Across self-play's fresh searches and across match-play moves, purged by board count | State-keyed (with ruleset and refresh epoch) — §8.5, C14 |
| Scanner history tables (killers, history, continuation — and correction, when v2 lands) | Across moves within a game, decayed; cleared between games | Per-thread — §11.3 |

#### The main tree: reuse specified, and §7.9's dilemma dissolved

Subtree reuse — the played line's subtree becomes the next search's starting tree — was
previously mentioned only as a §7.9 *cost* and never actually specified. Specified now:
**the tree is reused at match play.**

The §7.9 conflict conflated two things a node holds:

- **Search statistics** — visits, Q, solved bounds, priors, expanded structure. Mildly
  stale under a new root refresh (they aggregate old-conditioning evals) and
  self-correcting: fresh backups wash them out. This is the staleness every reusing
  engine already tolerates.
- **KV deltas** — genuinely conditioned on the previous root's refresh. Poisoned, not
  mildly stale.

Rule: **carry the statistics, invalidate the KV deltas, recompute representations lazily
on next visit.**

#### Deletion is garbage collection, and less final than it looks

On re-rooting, delete only *path-dependent* structure: under §8.5's MCGS this is GC of
nodes unreachable from the new root. State-keyed facts — proofs, TT entries, cached
evals, accumulators — persist, so a line abandoned this move that transposes back next
move re-hits everything for free. The solved store and asset map are never deleted; they
are position facts, not line opinions.

Position-keyed proofs are sound: with no repetition rule (§11.0 item 5), no proof can
depend on how the position was reached.

#### Pondering is the same mechanism

After our move commits, keep searching the resulting position (opponent to move). No
single-move ponder guess is needed — PUCT spreads effort across replies by prior — and
§12.4's opponent-reply head sharpens exactly the prior that allocates that effort, so
ponder-hit rates improve as a side effect of a target already adopted. On the opponent's
reply: re-root into their move's subtree, GC the rest.

#### Self-play is different — do not carry by default

A reused tree carries visits accumulated under the previous search, including
verification-steered ones (the asset map's boosts, Hunt's flagged priors) — so §12.2's exclusion would have to track provenance *across
moves*, and carried visits break §12.5's playout-cap accounting. The AlphaZero-lineage
convention is fresh fixed-budget searches per move in self-play, and it is the default
here: **match play carries everything; self-play runs fresh** — or, if reuse is ever
wanted there, persists the forced-visit flags across the reuse boundary (the harder path,
ablation 6n).

Fresh searches still share the state-keyed **evaluation cache** — a cache is not a tree,
and an evaluation is a function of the state and refresh epoch — purged by board count as
the game grows. In match play between two evaluators each has its own search and cache,
never shared; a cache is shared only between searches with one evaluator and one network
input (BUILD_LOG issues 89, 114–116).

---

## 9. Parameters

### 9.1 The governing constraint

Attention cost is `4T²dL`. **T enters squared; d and L enter linearly.** Halving board
count is worth 4×; halving width or depth is worth 2× each.

Consequence, which inverts normal transformer scaling advice: **narrow and deep, with
capacity bought in the FFN.** FFN cost is linear in T (`32Td²L` at the 8× ratio in §9.2); attention is quadratic.
Widening the FFN ratio buys parameters almost free at these sequence lengths, while
raising d_model is taxed quadratically. A square holds ~13 states — it does not need 768
dimensions.

### 9.2 Starting spec

| | Value | Reasoning |
|---|---|---|
| d_model | 256 | Low per-token entropy; taxed quadratically |
| Layers | 16 (24+ under the §3.3 contingency) | Depth is reasoning hops; only linear cost |
| Heads | 8 (d_head 32) | 4 routed + 4 dense per §6 (routing ablation-gated); RoPE specialised per head (§4.1) |
| FFN ratio | 8× for v1 — **in question**: Lc0 found little gain from wider FFNs; ablation 5 tests 2× (below) | Linear in T, not quadratic — but 37% of compute at the typical 44-board game |
| Registers/board | 2 | Multiplies by N |
| Global registers | 16 | Free |
| KV heads | 8 — full MHA | GQA saves 1.2% of dense cost and is retrofittable. See §10.2 |
| Value head | 51-bin categorical, FP32 | §8.3 — its variance feeds §8.4 and §8.6 |
| Aux heads | Mean-pool board squares, per-board and global — registers stay free scratch (B8, D7) | §12.4; ~0.1M total |
| Smolgen latent | 32, bilinear per head | §14.4 board-level variant; <0.1M |
| Params | ~21.5M with all heads (21,359,370 as built at N5, before CHECKPOINT 6's normalisation and displacement-bias changes) | Deliberately below AlphaZero's ~46M |

At 240 boards: ~15,800 tokens, **~4.1 TFLOPs attention** per position, roughly **337
evals/sec on Blackwell**. FFN and projections add ~0.67 TFLOPs — **16% of attention**,
14% of the ~4.78 TFLOPs total.

Batching adds only ~1.2–1.5× (§7.8), not the order of magnitude it gives decode-bound
workloads.

**Corrections from earlier revisions:** these figures were previously stated as 3.1 TFLOPs,
500 evals/sec and 6% FFN overhead. All three were wrong. None of the conclusions change —
narrow-and-deep still wins and FFN parameters are still nearly free — but the margin is
thinner than claimed.

**The capacity ladder** (240 boards, FA4-class throughput):

| Variant | Params | evals/s | vs base |
|---|---|---|---|
| base 256×16, ff8 | 21.0M | 337 | 1.00× |
| ff12 | 29.4M | 319 | 0.95× |
| ff16 | 37.7M | 303 | 0.90× |
| L20 | 26.2M | 270 | 0.80× |
| L24 | 31.5M | 225 | 0.67× |
| d320 | 32.8M | 261 | 0.77× |
| **60M-class: L24 ff12** | **44.0M** | **213** | **0.63×** |
| d512 — rejected | 83.9M | 148 | 0.44× |

FFN first (nearly free), then depth, then d_head, then d_model last — §9.1's ordering
with numbers attached. The larger ablation arm is **L24 ff12**, not a wider model. And
the prior has moved: MCGS's zero-cost transposition evals, PCR's cheap generation and the
cached path's 15–20× all improve the *nodes* side of the nodes-vs-quality trade, shifting
the expected winner of ablation 5 toward the larger net. §13.1's baseline discipline
still builds 21M first.

**The FFN rung is in question.** In Lc0's experiments chess transformers gained very little
from widening the FFN from 1× to 4×, and neither mixture-of-experts nor GLU helped; their
strongest network, BT4, runs at 1.5×. "FFN first, nearly free" was priced at 240 boards,
where the FFN is 11% of the cost. At the human games' median of 44 boards it is 37%:

| FFN ratio | Cost vs 8× at 44 boards | Evals/s |
|---|---|---|
| 8× (v1) | 1.00 | 1.00× |
| 4× | 0.81 | 1.23× |
| 2× | 0.72 | 1.39× |
| 1× | 0.67 | 1.49× |

Ablation 5 gains an arm: 2× against 8× at matched compute, the saved compute spent on
simulations. If 2× holds its strength, the ladder's first rung becomes depth and F1's FFN
ratio changes before serious self-play — it is committed-class. Lc0 measured at d =
768–1024 with 64 tokens; this network is far narrower and reasons across thousands of
tokens and many boards, so the finding is evidence, not proof.

### 9.3 Normalisation and biases (CHECKPOINT 6)

The spec never named the trunk's normalisation. It does now, because the choice is
committed-class: set before training, fixed after.

- **RMSNorm-style layers — no mean-centring, no bias — everywhere except inside smolgen
  (B12), and no biases in the Q, K and V projections.** Lc0's team reports this makes
  training about 10% faster and inference about 5% faster with no quality loss (its BT5
  change). Chessformer itself still uses ordinary layer normalisation, so this rests on
  Lc0's report rather than a published ablation. The output heads keep their biases; they
  are small and FP32.
- **Pre-norm for v1.** Lc0 uses post-norm with DeepNet initialisation. Pre-norm is the more
  forgiving choice for a self-play pipeline trained from scratch, where a stalled early run
  costs more than the last fraction of quality; Lc0's arrangement is ablation 6y's arm.

---

## 10. Performance engineering

### 10.1 Order of operations, by leverage

**Build order for the dense engine:**

| Rank | Technique | Gain | Exact? |
|---|---|---|---|
| 1 | History window (§10.4) — measure first | ~7× total | Exact (via §10.4.1) |
| 2 | FlashAttention-4 | 2–3× | Exact |
| 3 | FP8 attention | ~2× | ~1% |
| 4 | Block-sparse move mask (FlexAttention, board-major order — §6) | Benchmark | Exact |

**Deferred — only if the dense engine proves node-starved:**

| | Technique | Gain | Exact? |
|---|---|---|---|
| 5 | Causal masking + KV cache + tree prefix sharing (§7) | ~15–20× realistic | Exact*, unknown Elo cost |
| 6 | Cascade inference (FlashInfer) | Enables 5 at speed | Exact |
| 7 | Paged KV allocation with COW | Enables 5–6 | Exact |
| 8 | GQA 8:1 (retrofittable — §10.2) | 8× smaller tree memory | ~0.5% quality; harmful at d_head=32 |

\* Exact *given* the causal constraint in §7.3 — which is itself an architectural change
with an unmeasured cost, so the "exact" label describes the kernel, not the model.

**The window and the cache compose (§7.5) — they are not substitutes.** Even so, if dense +
window proves fast enough in practice, items 5–8 may never be needed. Measure before
committing.

### 10.2 Notes on specific items

The first three apply **only in the deferred cached configuration**. They are recorded
here so the option stays buildable, not because they are on the critical path.

**Cascade inference** splits attention into a shared prefix computed once at high
efficiency plus per-request suffixes, merged with online softmax. Your MCTS batch is a set
of leaves sharing most of their history — this is that shape exactly. It is what makes the
cache fast rather than merely correct.

**PagedAttention** matters more here than in LLM serving. MCTS creates and destroys
thousands of nodes with variable-length caches that grow as timelines spawn; contiguous
allocation fragments badly. **Copy-on-write prefix sharing** is the specific mechanism —
children share pages with the parent until they diverge.

**GQA — rejected for dense, retrofittable for cached.**

GQA reduces the number of *KV* heads, not query heads. Every query head still computes its
own scores against the shared K, so **the QK^T and PV matmuls are identical under MHA and
GQA** — and those are 86% of dense cost. Only the K/V projection GEMMs shrink:

| | Total | evals/sec | Params |
|---|---|---|---|
| MHA (8 KV heads) | 4.784 TF | 337 | 21.0M |
| GQA 2:1 | 4.751 TF | 340 | 19.9M |
| GQA 8:1 | 4.726 TF | 341 | 19.1M |

**1.2% of compute.** Bandwidth does not rescue it: dense attention runs at ~T/2 ≈ 7,900
FLOP/byte against a 563 ridge point, so you are compute-bound by an order of magnitude.

It is also *actively harmful at this width*. GQA 8:1 at d_head=32 leaves the entire key
space at 32 dimensions — 8 per axis under 4D RoPE, carrying all positional discrimination
across 15.9k tokens, on the axis §13 already flags as marginal.

**In the cached configuration** its value is tree memory, and the numbers are smaller than
an earlier revision of this document claimed. That revision assumed each node stores an
independent full cache, ignoring radix delta storage — the entire point of prefix sharing.
Correctly, at ~528 novel tokens per node:

| | Per node | 800 sims | 10k sims |
|---|---|---|---|
| MHA | 8.65 MB | 6.9 GB | 86.5 GB |
| GQA 8:1 | 1.08 MB | 0.87 GB | 10.8 GB |

Real headroom at high sim counts; not binding at 800. As built, the delta is smaller
still: the live set is recomputed every node (C19), so only boards superseded inside the
search are stored — 1.08 MB each in BF16, one or two per node (BUILD_LOG issue 37).

**It is not a pre-training commitment.** MHA converts to GQA by mean-pooling KV heads plus
uptraining at ~5% of original pretraining compute. So unlike the causal mask (§7.7), this
decision does not have to be made before spending self-play compute — which removes any
reason to hedge toward it now.

**FlashAttention is mandatory for memory, not just speed.** At 15k tokens the materialised
score matrix is ~503 MB per head per layer in bf16. Eight heads and one layer's
intermediate exceeds a card. FA4 on Blackwell reaches ~1,613 TFLOPs/s at 71% utilisation,
with gains strongest above 4k tokens.

**8-bit inference — FP8 first, QAT prepared.** Quantise the trunk's matrix
multiplications — weights and activations, attention included — and keep the policy and
value heads and RoPE at FP32 (§8.3: telling +0.3 from +0.5 is exactly what quantisation
error eats). Use FP8 before INT8: on H100-class tensor cores they run at the same peak
rate, and FP8 holds accuracy far better when applied after training. The size premise
raised the stakes (§13): at the human games' median of ~44 boards, the matrix
multiplications outside attention are almost half the cost, not the ~14% they were at
240 boards.

The chess precedent is Lc0. Its team reported success with INT8 quantisation that could
speed its models up by about 50% without quality degradation, and its training code
gained a QAT option the same week (Monroe's `quantize` flag) — a reasonable reading is
that QAT is what made 8-bit work for a chess transformer. So QAT is **prepared, not
adopted**: the training code carries fake-quantisation hooks behind a flag, smoke-tested
in the untrained build, and the exact simulated format is pinned once the inference path
is chosen. Lc0's ONNX-to-TensorRT route does not carry over (below).

The decision is the two-arm test (ablation 6w): FP8 applied after training against BF16,
at equal wall-clock and at equal search budget, per board range. QAT is switched on — as
a fine-tuning stage at the end of a training phase, not from the start — only if the
equal-budget arm shows a real loss and the equal-wall-clock arm says the speed is worth
keeping. Whatever runs, the matching rule holds: QAT simulates exactly the deployed
format, K/V cache included. Weight-only schemes (INT4 and below) target memory-bound
decoding and buy little here; ternary and 1.58-bit stay unproven. NNUE's integer success
does not transfer — it is a tiny MLP over a sparse feature transform, not an attention
stack.

**The inference path: PyTorch, not an exported graph** (CHECKPOINT 6). Lc0 runs its network
through ONNX and TensorRT because its engine is a single C++ binary and its network a fixed
64-token graph. Neither holds here. The search is C++, but the network runs in PyTorch and
drives the search through the engine's C API (as built at N5), so nothing needs exporting.
And the graph is not fixed: token counts vary with N, and every attention bias — typed
edges, the displacement bias, the masks — is a FlexAttention score function computed inside
the kernel. ONNX has no equivalent: each bias would have to be written out pair by pair —
at the median 44-board position, 8 heads × 2,920² ≈ 68 million entries per position per
bias, gigabytes per batch — or TensorRT given a custom attention plugin, which means
writing the attention kernel ourselves. TensorRT also runs only on NVIDIA hardware. So
PyTorch with torch.compile and FlexAttention is the inference path, and TensorRT is
revisited at 6w on NVIDIA hardware only if FP8 there leaves a gap worth a plugin. The
development machine's AMD RX 9070 XT is listed in AMD's PyTorch-for-Windows release notes;
whether torch.compile — which FlexAttention needs — works there is unstated, so it gets
one bounded attempt, with any figures labelled AMD.

### 10.3 The serving stack, consolidated *(deferred configuration only)*

**Nothing in this subsection is needed for the dense build.** It is the reference for the
cached configuration in §7.1, retained so that path stays open.

The named techniques are scattered across this document by topic. Collected here, with the
caveat that **they are not six independent choices** — three of them are layers of one
stack and one is the library that implements two of the others.

| Technique | Layer | Role in this engine | Section |
|---|---|---|---|
| **PagedAttention** | KV memory allocation | Non-contiguous fixed-size blocks; kills fragmentation as MCTS nodes churn. COW lets children share parent pages until divergence | §10.2 |
| **RadixAttention** | KV reuse policy | Sits *on top of* paged blocks. Radix tree of cached prefixes = the search tree; evict by visit count, not LRU | §7.2 |
| **FlashInfer** | Kernel library | Provides the cascade-inference and paged-attention kernels the two rows above depend on | below |
| **GQA** | Architecture | 8× smaller tree memory (6.9 GB → 0.87 GB at 800 sims). Retrofittable by uptraining, so not needed up front | §10.2 |
| **FlexAttention** | Kernel compilation | Makes the block-sparse move mask and any learned 4D bias runnable at near-FA3 speed | §4.2, §6.1 |
| **ALiBi** | Positional encoding | Candidate for 4D relative position; extrapolates to unseen N and is kernel-native. Needs per-head slopes incl. near-zero | §4.3 |

**Dependency order.** ALiBi vs RoPE is a training-time commitment and must be decided
first. GQA is *not* — it retrofits by uptraining at ~5% of pretraining compute (§10.2), so
it can wait until the cached path is actually being built. FlexAttention only matters once you
have a mask worth compiling. Paged → radix → cascade is a single stack built bottom-up, and
none of the three help without the causal constraint in §7.3 that makes caching valid at
all.

**FlashInfer specifically.** Worth naming separately because it is the concrete
implementation path rather than a technique: it supplies paged KV-cache kernels, cascade
inference with online-softmax merging, and FA-family kernels behind one interface. Building
paged allocation and cascade merging by hand is the alternative, and there is no reason to.

**What is deliberately absent.** vLLM and SGLang themselves — the *engines* — are not on
this list. They are request-oriented servers built around autoregressive generation, HTTP
scheduling, and continuous batching, none of which apply. Take PagedAttention and
RadixAttention as *algorithms* and implement them inside the search loop; do not try to
serve MCTS leaves as if they were chat completions.

**The second tier, sorted by the same principles:**

| Method | Verdict | Why |
|---|---|---|
| KV-cache quantisation (FP8 K/V) | Adopt | Halves the ~650 MB endgame-state cache (§8.7 persistence, companion §9.4 batching headroom); obeys §10.2 — cache and live computation share precision, which holds only with an FP8 trunk or a network trained with FP8 frozen K/V: with a BF16 trunk, FP8 frozen K/V sit RMS 8.7e-3 from the BF16 network, 3.8× the BF16-vs-FP32 gap (BUILD_LOG issue 34). KV per state counts frozen boards only, plus their trunk outputs (~16 MB at 240 boards) |
| Ragged / varlen attention kernels | Adopt | Variable per-position scope (§10.4.3) makes padded batches waste work; varlen FlashAttention avoids it |
| Speculative decoding | Mostly already present | Its structure is virtual-loss batched descent. The residual analogue — speculative leaf prefetch to fill under-filled match-play batches — is quality-neutral (state-keyed cache) and useless in self-play, where cross-game batching fills batches. Low priority |
| Early exit / adaptive depth | Prototype tier — confidence family | Easy positions exiting at layer 8 save ~50% on those evals; "easy" is self-rated, so it inherits §12.3's limit and the add-only rule: the scanner baseline never scales down with it. Needs trained exit heads; calibration monitoring (§14.4) applies |
| Within-system distillation (big net → smaller net) | Legitimate capacity-ladder tool | Does not cross the verifier boundary — the refused distillation is *between* systems. Ablation 5 gains a distilled-small arm |
| Latent-KV compression (MLA-style) | Refuse for now | Compresses keys through a low-rank bottleneck — thins the key space on the axis §4.1 and §10.2 flag as marginal; the GQA reasoning applies unchanged. Revisit only if cache memory binds |
| Mixture of experts | Low priority | Capacity at constant per-token cost — but the ladder's FFN rung is nearly free already (ff12 costs 5%), so MoE's advantage is small and its routing overhead is not, at 21M params |

The pattern: compute-reducing tricks transfer or are neutral; capacity-reducing tricks
on the key axis hit the same wall as GQA; self-assessment tricks join the confidence
family with the add-only rule attached. No new principle was needed to sort them.

### 10.4 The window — the primary lever, and the first thing to measure

**What counts as a board token at all.** Full history at 240 boards versus a windowed
history — all timelines, but only the last ~10 time steps plus boards that are actual legal
travel destinations — might reach N=80. That is **7.0× on total cost** (~2,370 evals/sec
dense) — 9× on attention alone, but FFN and projections are linear in T and drop only 3×.
It costs nothing expressively: you drop boards that are genuinely unreachable, rather than
restricting how the reachable ones communicate.

Window by **game rules**, not learned attention scores (§14.2). Size the window empirically
from real games **before locking any network dimensions.**

**Classification: an uncommitted decision of the committed class.** The design has not
adopted the window — §10.4.2 gates it on measurement — but once made, the choice is
§7.7-class: scope sets the token-count distribution the net trains on, and a net trained
windowed (N_eff ≈ 80 + |E|) deployed unwindowed at 240 boards is exactly the
out-of-distribution problem the §12 curriculum exists to prevent. The exact merge softens
this — a windowed state is a legitimate smaller state under relative encoding — but does
not erase it: windowed midgames carry coordinate-gap structure (large Δtime to E-boards)
that genuine early-game small states lack. **The decision precedes serious self-play
compute.**

#### 10.4.1 Exact windowing via online-softmax merge

First, a distinction an earlier revision muddled. **The window prunes the network's input
only.** The game state and move generator always remain complete: a windowed board still
exists, moves into it are still generated, search can still enter it. What an out-of-window
board loses is *evaluation awareness* — the net cannot see a threat there, so search must
stumble onto it unguided rather than being steered. (If an implementation pruned the state
itself, moves would genuinely be lost — that variant is the §14.2 hazard, and it is simply
not the design.) Even the milder eval-blindness is avoidable entirely:

Online softmax merges attention over **disjoint key partitions exactly**: carry
`(output, running_max, running_sum)` per partition, rescale, combine. No approximation. It
is the same mechanism FlashAttention is built from, and what makes Flash-Decoding and
cascade inference work.

So partition the history:

| Partition | Contents | Treatment |
|---|---|---|
| **W** | Boards inside the time window | Contiguous, block-structured, dense |
| **E** | Out-of-window boards that are legal travel destinations for some live piece — as built, the mover's moves at the node, a branching move counting at the history board it targets; the opponent's destinations enter at the child | Sparse gather |

Compute attention over W and E separately, merge with online softmax. The result is
**exact relative to full attention over W ∪ E**, and E comes from the move generator, so
the partition is rule-based with no learned pruning anywhere.

**This removes windowing's only correctness hazard.** Nothing reachable is ever dropped.
One mechanical note for the cached configuration: a board outside scope at creation never
received K/V — when a piece later lines up on it and it enters E, its K/V is computed
**lazily**, one board, once (the §8.7 pattern).

#### 10.4.2 The measurement, restated

**|E| is the window experiment, and it has predictable structure worth exploiting.**
Leapers contribute a small bounded set. Sliders reach unboundedly far *only along clear
rays* — a long-stationary piece blocks its own pure-time ray with its past selves, and 4D
diagonals need a different square clear on every intermediate board. Expect |E| modest in
dense middlegames (obstruction everywhere) and larger in sparse endgames (open lines
through time). **Measure the phase curve, not a single number** — and because the
online-softmax merge is exact for *any* rule-based partition, the scope rule may widen by
phase and remain exact. The window is a family of exact scope rules, not one fixed size.

The decision cascade, cheapest first:

1. **Phase-stratified |E| on real game records** — pure move-generator analysis, hours,
   no training. |E| large everywhere (thresholds pre-registered in §15.6) → the window dies; the fallback chain is the cache
   (§7, which composes with nothing lost), then the §3.3 axial contingency. Manageable →
   proceed. Measured (CHECKPOINT 5) on 716 human games to 186 boards — provisional above
   that — and the window is killed, provisionally (§15.6).
2. **Adopt provisionally for the v1 build** on arithmetic — 337 vs ~2,370 evals/sec is
   development velocity, and coverage is exact by §10.4.1.
3. **The committing head-to-head, before serious self-play compute (§7.7-class), in two
   arms:** **equal wall-clock** — deployment-relevant; the windowed net gets ~7× the sims
   and should win unless eval-blindness beyond one travel ply costs more than 7× nodes
   buys — and **equal sims**, which isolates and prices that blindness (expect a small
   deficit). Cheap alternative to two training runs: **scope randomisation** — window
   on/off stochastically across self-play games, one net competent at both scopes, A/B
   free at inference. Speculative; costs some capacity.

This resolves the §13 risk flagged as "the window may not be 7×" by measurement rather
than estimate.

**Confidence: low until step 1 runs.** The 7× figure is a guess about what fraction of
history is genuinely reachable, with no measurement behind it — which is precisely why
the phase curve is the first experiment, not a later optimisation. (An earlier revision
framed this as deciding whether the cache is worth building; §7.5 corrected that — they
compose, and the window's job is throughput on both paths, not replacing either.)

#### 10.4.3 Sizing W — a priority queue on the floor

An earlier revision dismissed "how deep should the window be" as unprincipled. Half
right: §10.4.1's E killed the *correctness* stake — nothing reachable is lost at any W —
but that left W an unprincipled free parameter. It has a principled shape once its actual
job is named.

**What W is for.** Not threat coverage: a threat needing one setup move before travelling
is search's job — E re-partitions at every node, so the child after the setup move has
the target in *its* E — and target juiciness is already scored by the asset map. W
provides **evaluation context**: the recent texture each head descended from, the
**present band** where move obligations bind, and **branch origins** — the board each
timeline split from is the minimal history explaining that timeline's existence. So W is
event-anchored, not a flat depth: per timeline, back to the branch origin or d_max steps,
whichever comes first; the present band always.

**The add-only inversion.** §14.2 banned learned and heuristic pruning because dropping a
reachable board is a silent blunder. That hazard lives entirely in the *subtract*
direction — and E is a guaranteed floor, so anything added on top of it is extra context
that no heuristic can turn into a correctness failure:

> **Heuristics may add scope, never subtract it below E.**

What §14.2 banned stays banned; what it accidentally also banned — heuristic
*enrichment* — is released, because the exact merge changed what is at stake. W becomes a
**budgeted priority queue**: present band → heads' recent k → branch origins →
asset-map-flagged boards (juicy-but-not-yet-reachable targets pre-loaded into the net's
view) → near-reachability boards (one alignment move from entering E — a cheap E₂
approximation) → recency filler. As built: k counts half-turns along a timeline (one board
each); the present band is the active heads; branch origins belong to the mandatory core,
as §15.6's W column assumes.

**Budget mechanics — three layers, one cap, per node.** Scope decomposes as the
**uncapped exact floor** (E), the **mandatory core** (present band plus each head's
recent context — this scales with active timeline count by necessity, since heads and
obligations exist per timeline), and the **budgeted filler** (the remaining tiers,
filling a token budget B). B exists for three reasons: batch-shape predictability (companion §9.4 —
padded batches want bounded token counts), memory provisioning, and
training-distribution stability (a bounded W keeps N_eff in a band the net has trained
on). **E is never capped**: a max on what we choose to show, none on what the rules
demand we show. Total scope therefore has a designed typical size and a rules-determined
tail — provision memory for the tail, and a routinely fat tail is cascade step 1's kill
condition doing its job (§10.4.2). Selection runs per node as search descends, at ~zero
cost: E falls out of legality generation, the asset map is a standing annotation, the
core is tree bookkeeping — and under the cached configuration a sliding scope invalidates
nothing, because scope decides which boards *participate in attention*, not their cached
values.

**Cost is soft** (8 active timelines, +|E| ≈ 25):

| W depth k | Scope boards | evals/s |
|---|---|---|
| 0 — floor + flagged only | 40 | ~7,100 |
| 4 | 72 | ~2,820 |
| 6 | 88 | ~2,020 |
| 8 | 104 | ~1,520 |
| 12 | 136 | ~950 |
| 16 | 168 | ~650 |
| full history | 240 | ~250 |

Doubling k from 4 to 8 costs ~1.9× — a knob to measure, not agonise over; the equal-sims
arm at k ∈ {0, 4, 8, 16} prices the context directly (ablation 1) — k = 0 is the
minimal heads-plus-reachable-plus-flagged variant.

**"Live timelines" resolves into the existing machinery.** Rules-**active** timelines get
W bands — obligations bind, heads act. Rules-**inactive** timelines carry no obligations,
but their boards remain travel destinations: they are **E-material**, in scope exactly
when reachable, with no separate decision needed. (Pinned at CHECKPOINT 1: activation follows the
documented rule in docs/rules/README.md, and a board on an inactive timeline where the
side to move would have to play first does not originate check — §11.0 item 1.)

**Refused: pruning by strategic deadness.** A rules-active timeline that looks hopeless
**stays in scope** — its obligations still bind, its boards remain reachable, and "looks
hopeless" is a learned evaluation. Rules-live is a scope category; strategically dead is
a judgement, and judgements don't prune.

**The compounding objection, stated at full strength.** The strongest case against the
window: the net cannot value what it cannot see; search allocation follows the net's
priors; so search never explores toward out-of-scope resources, and the blindness is
never corrected — the adversarial-Go loop applied to scope. The mitigation stack,
honestly graded:

| Layer | What it breaks | Strength |
|---|---|---|
| E is exact at one ply | Any strategy executable *now* is fully visible — the loop exists only for multi-ply quiet setups | Hard guarantee |
| Search re-partitions per node | One visit into the setup move and the child's eval sees the target (it entered that node's E) — the blindness lives in the root *prior*, not the child *evaluation* | Hard, given that first visit |
| Root Dirichlet + entropy widening (§8.4) | Every root move gets nonzero exploration probability — supplies the first visit at depth 1 | Statistical |
| Asset map (§11.2) | The moment the travel is legal anywhere in the tree, the standing boost fires and the child's value flows back | Strong at depth 1–2 |
| Priority queue pre-loads asset-flagged and near-reachable boards | The juiciest targets are in the net's view *before* being reachable | Partial |
| §12.4 travelled-into training | Games where deep travel happened teach frozen boards to advertise themselves — the loop closes asymptotically through the loss | Slow but structural |

What survives all six layers: quiet setups of two or more plies toward unflagged targets
— §13's registered residual, its mechanism now spelled out. Two consequences: the
adversarial suite (6h) must include **window-targeted attacks**, constructed positions
whose winning line runs through out-of-scope resources, because the compounding
hypothesis is exactly what that suite exists to falsify. And one framing correction:
search relies on the net for *allocation only* — legality, check detection and terminal
results are generator-exact at every node regardless of scope. The failure mode is missed
opportunity, never misunderstanding.

**The league closes the loop (§12.7).** Adversary-found exploits are the tuning signal
for the filler tiers: every exploit reveals a board that should have been in scope, and
the add-only rule means a filler rule that would have caught it can be added without
correctness analysis. Scope rules evolve under attack; the floor never moves.

---

## 11. Move generation (CPU side)

Likely to be the actual bottleneck. Lc0 splits between GPU (net) and CPU (move gen, tree
ops). With 240 boards live, move generation is enormously more expensive than in chess.

**Bitboards map well to this geometry:**

- Within-board attacks: standard magic bitboards per board.
- Time and timeline movement is "same file+rank, different board" — an occupancy AND
  across a timeline. Cheap.
- Diagonals combining spatial and temporal axes are shifted-ANDs.

This is the same computation that builds the attention mask (§6), so it is computed once
and used twice.

### 11.0 The rules engine

**Every rule lives in the generator.** Legality, check and game end are decided exactly by
code, never by the network: the policy head scores only generator-supplied moves (§8.1),
and the provers' proofs, the permanent solved store, §6's typed edges, §10.4's E set, the
asset map and every terminal value inherit their exactness from here. That makes this the
most correctness-critical code in the system — a bug here does not misplay, it
manufactures false proofs that persist.

**The five pieces:**

1. **Check detection.** Check means the opponent has a move that captures a royal piece,
   implemented directly by generating every move by the attacking colour that captures
   an enemy royal. Those captures come from any board the attacker could move on and run
   through time and across timelines. Royals are plural — one king per timeline; in
   Reversed Royalty the royal is the queen, and the king is a common king, not royal
   (§12.6). **Check status** (ruled at CHECKPOINT 1,
   docs/rules/README.md): a royal is in check if, passing on all *active boards in the
   present*, the opponent could capture it from any of their playable boards. Optional
   boards ahead of the present and boards on inactive timelines are not passed — the
   opponent can move there only if the side to move plays there first, and the
   submission test (item 2) catches any royal capture that results. A royal capture
   counts even if it would branch into a timeline that would be inactive: activity
   governs the present, not whether a capture is legal. Status decides checkmate versus
   stalemate. (This replaces an earlier paraphrase — "advance every eligible timeline to
   its phantom board" — that matched ftxi's over-broad reading; all 78 oracle check
   disagreements came from exactly this definition.)
2. **Legality is per turn, not per move.** A turn — an *action* — is a set of moves across
   boards, submitted together. Being in check midway through a turn is legal; only the
   submitted turn must leave every royal safe. The obligation to move on the boards the
   present line requires is enforced at submission too, and it is a minimum: a player
   may keep moving on their remaining playable boards after the present has shifted, and
   the turn ends only at submission (Q4). **Submission legality is a
   separate test from check status:** it applies to the actual turn, not a passing
   hypothetical — after it, a royal capture from any of the opponent's playable boards,
   active or inactive, makes the submission illegal. Consequence for the ply split:
   §8.2.
3. **Checkmate and stalemate — the genuinely hard one.** "Does any legal turn exist?" is a
   search over combinations of per-timeline choices, combinatorial in the number of
   timelines. Two detectors: **naive depth-first search**, fastest when options are
   abundant; and the **hypercuboid algorithm**, which treats a turn as one *semimove* per
   timeline — one timeline's share of the turn, with "play nothing here" a valid
   semimove — and searches that product space, with better worst-case performance,
   especially near mate where legal turns are sparse. Run both: naive as the fast path,
   hypercuboid for near-mate positions — the positions proofs come from. In the gate's
   adapter they run as a portfolio on doubling node budgets, hypercuboid first, the other
   cross-checking. **A mate
   verdict is a proof, so it requires an exhaustive search that terminated.** An
   exhausted budget — nodes, time or memory — yields "unknown", never "mate" — the failure behind the official
   client's false mate claims and Chess in 5D's (CHECKPOINT 1). The two detectors are
   cross-checked against each other in the gate. The hypercuboid
   method originates in penteract's cwmtt and is adapted, with improvements, in ftxi's
   5dchess_engine.
4. **Ordinary chess rules** — castling, en passant, promotion — apply board by board, and
   so do their attack tests: castling is blocked only if the king or the square it
   passes through is attacked by a piece *on the castling board*, and en passant is
   per-board only. A king in check from another board may castle, but the turn still
   cannot be submitted until that check is resolved somewhere. (Corrected — an earlier
   revision called these 5D queries without a source; the documented rule is per-board.)
5. **Game-state bookkeeping** — the present line, active and inactive timelines, whose
   turn it is on which board, submission — is an exact state machine. Activation follows
   the documented rule (docs/rules/README.md). In multi-timeline variants, initial
   timelines count as created by neither player — always active, and not counted toward
   either player's created timelines (Q2: a provisional project decision read from the
   documented rule's wording, which counts *created* timelines). **There is no repetition rule** (project
   decision): passing is impossible and every move creates at least one board, so the
   full state can never recur within a game. Values therefore depend only on the state,
   and graph-history interaction does not arise (§8.5, §8.7).

#### The correctness gate — differential testing

No proof enters the solved store until the generator has passed **differential testing
against independent implementations**:

| Reference | Why it is useful |
|---|---|
| ftxi/5dchess_engine (github.com/ftxi/5dchess_engine) | BSD-2; C++, Python and WASM bindings; reads arbitrary variants from 5dfen; ships `diff` (compares two algorithms) and `perftest` tools; bitboard movegen, check detection, both mate detectors |
| penteract/cwmtt (github.com/penteract/cwmtt) | Independent checkmate detection; the source of the hypercuboid method |
| Chess in 5D (chessin5d.net) | A third independent open-source implementation of the rules |

Cross-check move counts (perft-style), check verdicts and mate/stalemate verdicts over
corpora of real games in 5dpgn notation, and investigate every disagreement before the
proof channel is enabled. The gate re-runs on every generator change and per variant
(§12.6); once running, the adversarial league (§12.7) is the ongoing fuzzer.

**The official game client is never an oracle** — it is believed to incorrectly claim
some positions are checkmate (penteract's documentation). Agreement with it proves
nothing.

**Build, don't adopt.** The generator must emit things no existing engine produces — §6's
masks and typed edges, §10.4's E, ruleset-keyed Zobrist hashes (§11.3) — so it is built
in-house, with ftxi's engine as the reference oracle rather than the core. Reuse would also
forfeit ftxi as an oracle (below). Each oracle's implementation details *as pinned* are
recorded in the repository's PROVENANCE.md, which is authoritative over this document on
them.

#### Third-party code — oracles, not core

**Every component reused from another implementation costs one independent oracle for
what it covers.** A mate detector ported from ftxi's code cannot be differential-tested
against ftxi's detector — both would share the same bugs. So, for anything the gate
checks:

| Project | Role | Why |
|---|---|---|
| ftxi/5dchess_engine | Pinned test-time oracle; reference for understanding the hypercuboid algorithm | BSD-2, so reuse is legally easy — but reuse forfeits it as an oracle |
| penteract/cwmtt | Test-time oracle (independent checkmate detection); the original description of the hypercuboid method | Check its licence before copying anything |
| Chess in 5D | Third oracle; possible source of game records | Check its licence and terms |
| adri326/5dchess-notation — 5dpgn ("Shad's notation"), the fourth submodule | The pinned notation specification: our own parser is written from its text, the game corpus is in its format, and it settles parse divergences (below) | A parser shared by the engine and an oracle corrupts both sides of a comparison identically — the gate never sees it. If the repository ships parser code, that code counts as a parse-only oracle under the same harness rules — never a basis for ours |
| Stockfish, Lc0 (both GPLv3) | Ideas only, reimplemented from papers and the Chess Programming Wiki | The project is GPL-3.0 too, so copying would be licence-compatible — but it would end sole copyright, the one thing that keeps relicensing or dual-licensing possible, and other projects' code enters as ideas only |

The hypercuboid algorithm is reimplemented in-house. Its in-house cross-check is the naive
depth-first detector this section already runs beside it — two different algorithms that
must agree on every mate verdict. ftxi's detector and cwmtt are the external checks; ftxi's
adapts cwmtt's method, so a flaw in the method itself would not surface as disagreement
between those two, which makes Chess in 5D the most independent of the three. **The oracles are the only third-party 5D code in the repository**, and they live only
in the test harness: nothing from another 5D project is vendored into the engine for any
purpose, gated or not — everything else from other projects enters as ideas,
reimplemented. General-purpose libraries (PyTorch, FlexAttention, PyO3, test
frameworks) are ordinary package dependencies, not vendored source, and their bugs cannot
make the oracles agree falsely. The fourth submodule is the notation specification, not a
rules oracle; any parser code it ships is treated as a parse-only oracle, which keeps that
sentence true. On licences: algorithms and
ideas are free to reimplement, code carries its licence, and GPL obligations attach on
distribution; read each licence properly before copying anything. **The project's own
code is GPL-3.0** (LICENSE, from the repository's initial commit; recorded in
PROVENANCE.md). Its obligations attach when source or binaries are distributed — private
use and running the engine as a service are unaffected. The BSD-2 code adapted into the
ftxi driver is GPL-compatible, with its notice retained. Keeping third-party GPL code out
preserves sole copyright, so relicensing or dual-licensing stays possible later.
Whether the licence reaches trained weights is a separate, open question — decide it
before any weights are distributed.

**Pre-include the oracles; do not leave them to discretion.** The gate is a commitment
(H23), not a check to run when it seems necessary — and the moments it is most needed, a
subtle rules bug, are exactly the ones in which nobody knows it is necessary. So, from the
first commit:

- **Pinned, in-repository, behind an enforced boundary.** The oracles live in the
  repository as pinned submodules (or a lockfile-pinned fetch) under a separate test
  harness that is never a dependency of the engine's own build; CI fails any core module
  that imports from it. Left to discretion, the path of least resistance when stuck is
  copying an oracle's code into the core — exactly the reuse that destroys it.
- **Tiered CI.** ftxi's engine (fast; a C++ driver over its core library, held to its stock
  CLI) on every commit; cwmtt (Haskell toolchain, heavier) and Chess in 5D nightly and
  before any generator release or proof-channel enablement. The game-record corpus is
  pinned too. The commit tier is a smoke test: a ruleset is gated only when two or more
  oracles answer for it, so the check that *some* ruleset is gated runs only in the tiers
  that run every oracle — nightly and release (CHECKPOINT 6).
- **A known-divergences file.** When the engine and an oracle disagree, the rulebook
  decides; if the oracle is wrong, the divergence is recorded with its rulebook citation
  (and reported upstream), so it stays expected without masking future regressions. Any
  new divergence fails the gate. Oracles can be wrong — the official client is.
- **Provenance and instructions.** A provenance file records what came from where, under
  which licence, in which role; contributor instructions — for human developers and coding
  agents alike — state these rules, because an agent works from what the repository tells
  it.

#### Running the gate — matching, triage, stability, integrity

- **Match states, not move counts.** Every implementation exports a canonical fingerprint
  of the position it parsed — a serialisation or hash — at every ply; where an oracle
  cannot export its state, board and piece counts per ply are the minimum comparison. A
  fingerprint mismatch is a **parse divergence**, a class of its own, triaged before any
  rules comparison at that ply: comparing the rules verdicts of two different positions
  is meaningless. Parse divergences are settled against the pinned notation
  specification, as rules divergences are against the rulebook. Matching by move count
  cannot tell a rules disagreement from two parsers reading the same move differently. As
  built: the fingerprint is the notation specification's full-state MD5 — white's board
  before black's within a turn, where the specification is silent — beside a
  placement-only hash and board and piece counts; the harness alone canonicalises and
  hashes, holding any source that exports its own hash to the harness's; the unmoved
  marker means "has not moved"; a marker-only difference is reported per run and stops
  confirmation, not comparison. One exception to "every source reads the record as stored"
  (CHECKPOINT 3): a corpus record in the official client's export form is converted by the
  harness to the specification's 5dpgn, as notation only, and every source — the engine
  included — reads the converted text; a record the specification cannot express goes to
  no source.
- **Triage by cluster; settle by rulebook, never by vote.** Disagreements that survive
  fingerprinting are grouped by the attacker's situation — whose turn it is on the
  attacking board, active or inactive timeline, head or earlier board, ordinary or
  cross-board move — and each cluster is settled once from the rulebook and recorded in
  the known-divergences file with its citation. Two oracles agreeing is not evidence on
  its own (ftxi adapts cwmtt's method). **Check disagreements first** — check decides
  checkmate versus stalemate — and the engine's check detection is not written until they
  are settled. The expectation — that they would concentrate in which boards count as
  sources of attack — was confirmed at CHECKPOINT 1: all seven clusters did. **The
  rulebook is docs/rules/README.md**: sources ranked (the in-game Guide, then cited
  developer rulings, then established references, then forums), every entry cited with
  a confidence level, silent cases labelled project decisions and mirrored in the
  register, never an oracle as a source; pinned, with the gate re-run when it changes.
- **Stable answers only.** An oracle that does not reproduce its own recorded answers is
  nondeterministic or limited by its search budget. Budget-limited answers are recorded
  as "unknown at budget X", or the budget is raised until they are stable;
  nondeterministic ones are marked unstable and excluded. The gate then requires 100% on
  stable recorded answers — a tolerance hides regressions inside itself. As built: each tier
  requires the stable values, recorded at least twice, whose recorded cost times a margin
  fits its budget, and a calibration probe scales budgets to slower runners. Before the
  proof channel is signed off, the naive searches' release budgets are raised and
  re-recorded twice, so the values that now leave the proof-channel check vacuous become
  required (Q-B, CHECKPOINT 6).
- **Recorded-answer integrity.** Recorded answers are immutable, with committed content
  hashes verified before every comparison; the comparison logic lives in harness code the
  engine adapter cannot modify; harness changes require review. What remains is trust in
  review. The mechanisms: a manifest of content hashes verified before every comparison,
  the harness re-hashed after the engine runs, imports from source only. The manifest
  protects against accident, not against an edit of both a golden file and the manifest —
  that is review's job.
- **Telltales — the known-divergences file as a copy detector.** Every recorded oracle
  error is a telltale: if the engine ever reproduces one, its logic probably came from
  that oracle. The boundary check is syntactic and cannot catch translated or renamed
  code; telltales catch it behaviourally, for the buggy paths — review covers the rest.
- **Parse mode before full mode.** The engine runs first in parse mode — states, replay
  length and acceptance compared, never verdicts — then in full mode. `legal_actions`
  counts an action as its set of moves, with its timeline-creating moves in creation
  order: a protocol definition, not a rule.
- **Linux CI before engine code.** The tiered CI must have run green on Linux, the
  deployment platform, before the first engine commit — the gate as it will actually be
  used, not as it ran on one developer machine.

---

### 11.1 Tactical verification layer

> **Status (CHECKPOINT 4 design review):** the brute-force and df-pn verifiers below failed
> their pre-registered tests from 50 boards up (§15.6). The verification layer in force is
> **§11.4**. This section and §11.2–§11.3 remain as the design history; their mechanisms
> apply only where §11.4 keeps them.

Lc0 deliberately omits quiescence search, on the theory that MCTS extends along forcing
lines by itself. **That theory is weakest exactly where 5D chess sits**: enormous branching
means shallow trees, and tactics arrive from any of N boards across two extra axes. A
verification layer is more valuable here than in chess, not less.

#### The governing principle

> A verifier catches only what it is **differently wrong** about — not what it is more
> cheaply wrong about.

This disqualifies the obvious design. A distilled small model inherits the big model's
biases *by construction* — distillation optimises for exactly that — so it will confidently
agree with the mistakes it was built to catch.

#### Verifier options, ranked by decorrelation

| Option | Decorrelation | Cost | Verdict |
|---|---|---|---|
| **Brute-force tactical search** | **Total** — no learned component | Lowest | **Build this** |
| Same net, different search paradigm | Search errors decorrelated, eval errors not | Very low | **Build this second** |
| NNUE adapted to 5D | High — different features, different training signal | Medium (training pipeline) | Rejected as verifier *model*; **adopted as scanner leaf eval, v2** — see below |
| Independently-trained small transformer | **Low** | Medium | Skip |
| Distilled small model | ~None | Medium | Never |

**Independently-trained is weaker than it looks.** A different seed does not decorrelate
much; error correlation comes from the *data* and the *architecture's inductive biases*, not
initialisation. A second model sharing the 4D RoPE, the move-graph bias, and the same
self-play data shares most blind spots. Real decorrelation requires varying the inductive
bias or the training distribution — not the seed.

**NNUE — the verdict attaches to the role, not the architecture.** Its features, its
incremental update, and its search-labelled training data are all genuinely orthogonal to a
self-play attention net. But NNUE's feature space is *fixed size* — §1's input failure mode
— and the variable-N version that survives, a **per-board accumulator summed across
boards**, is separable across boards and **cannot represent cross-board interaction**. As an
independent verifier *model* that is disqualifying: cross-board tactics are the thing the
verifier exists to catch. As the *scanner's leaf eval* it is not — see "Scanner eval"
below, where the same architecture is adopted.

**The option not usually considered:** run the *existing* net over a small alpha-beta search
on flagged positions. Eval errors stay correlated, but *search* errors decorrelate — and
search misallocation is the failure mode identified in §13 as the real risk. No new model to
train.

#### Two output channels: proofs and estimates

Scanner findings are not one kind of thing, and the plumbing differs:

| Finding | Status | Enters the tree as |
|---|---|---|
| Mate / forced win or loss inside the horizon | **Proof** — exact, given all-evasions generation in check *and* a generator that has passed §11.0's differential-testing gate | Solved bound (MCTS-Solver style) |
| Trade, material swing, hanging piece | **Estimate** — the opponent may decline via a quiet move (stand-pat) | Prior boost / value adjustment on the flagged line |

**Proofs propagate as bounds** (MCTS-Solver, Winands et al.): a proven-won child makes its
parent proven-won; all children proven-lost makes the parent proven-lost; solved subtrees
leave selection and **stop consuming net evaluations** — the net is never asked about
positions the generator has already decided. This is the one place the verification layer
*reduces* network load rather than adding to it. (With a transposition table, proven bounds
transfer across transpositions, with the usual DAG care.) Two selection details from the
MCTS-Solver literature and Lc0's certainty propagation: once children are solved, final
move selection uses the **secure-child** rule rather than max-visits, and solved-node
selection during search needs a **visit threshold** to avoid narrow-path bias.

**The search's own solver is part of the proof channel** (ruled at CHECKPOINT 4).
Scoring actual game-end positions as results is unavoidable and stays on. Propagating
solved status through the tree — marking nodes solved, pruning siblings, secure-child
selection — and exporting to the permanent store is the proof channel, gated like any
other: off until the gate passes, the search's soundness fixes have landed with tests,
and the proof channel is signed off. With exact depth verification failing above 50
boards (§15.6), the solver becomes the main source of proofs in large positions, so its
soundness bar rises: its proofs are cross-checked against df-pn offline where df-pn
decides, and against the oracles' mate verdicts where they exist. As built, the switch is the
search's `proof_channel` parameter, off by default.

**Estimates must never be promoted to proofs.** Stand-pat is what keeps quiescence sound
for non-forcing findings — but stand-pat assumes declining quietly costs nothing, which
zugzwang already violates in chess endgames. Whether 5D's move obligations (every active
timeline must advance) create broader zugzwang-like cases is an open check that belongs
beside the termination experiment below.

#### Recommended design

1. **Scanner.** Depth-limited alpha-beta over forcing moves only, **material-count eval
   for v1** (upgrade path below), running on the generator you already have (§11). Exact
   within its horizon — *provided in-check nodes generate all evasions*, which is what
   upgrades its mate findings from estimates to proofs. Move ordering stays **net-free**
   — MVV-LVA, killers, history and continuation history. Using the policy head for ordering would wire
   the net's biases into the layer that exists to be differently wrong.
2. **Proof channel.** Inject solved bounds into the tree as above.
3. **Estimate channel.** When the scanner flags a line as tactically live that MCTS did not
   explore, force-expand it regardless of policy prior. This addresses search misallocation
   directly, which neither windowing nor caching touches.
4. **Adjudication.** Spend net evaluations on the quiet positions the estimate channel
   promotes.

**Pruning discipline inside the scanner — the two-channel rule, recursively.** Imported
alpha-beta accelerations split by channel exactly as scanner outputs do. **Continuation
history** (move ordering) is safe everywhere. **ProbCut, futility and null-move** are
estimate-channel only. **No speculative pruning on any line that can emit a proof**: a
reduced evasion that misses the one defence produces a *false mate proof*, which becomes a
solved bound — the worst failure the system can manufacture. LMR barely applies (the
scanner searches forcing moves; its late-quiet domain is nearly empty) and is banned on
evasion nodes outright. Null-move additionally waits on the 5D-zugzwang answer (open
problem below).

#### Where the loop lives — and when to stop

An outer "scan → follow up → scan again" cycle would be a second search loop built on top
of the search loop. **MCTS is already the rinse-and-repeat mechanism.** Three loops exist,
at different levels, and only one repeats conditionally:

| Loop | Repeats? | Terminates because |
|---|---|---|
| Quiescence, inside the scanner | Until quiet | Depth cap + delta pruning (open problem below) |
| Per leaf: scan, then net eval (pipelined — §11.2) | **Once per new leaf** | A leaf is evaluated exactly once; "repetition" is just search expanding more leaves |
| Root verification, after the budget | **Only on refutation** | See below |

**Root rule: finish unless refuted.** After the search budget, scan the principal
variation. Clean → play the move. Refuted — the scanner finds a forced loss search never
saw → the move cannot be played, so re-search with the refutation injected as a solved
bound. That loop converges because the scanner is deterministic and exact within its
horizon: rescanning old positions yields nothing new, each round scans only newly added
nodes, and findings are monotone. Cap it at one or two rounds regardless, because clocks
exist. (§11.2 rung 2 streams this check during search, so end-of-budget refutation should
be rare.)

**Why the unconditional version fails here specifically.** "Repeat until no volatile boards
remain" does not terminate in 5D chess: with N boards, something is nearly always en prise
somewhere, so global quiescence is an unreachable fixed point — and chasing it starves the
search that actually decides the move. Volatility is scoped to the PV and near-PV, not the
whole multiverse. This is the same explosion flagged below, one level up.

#### The open problem: defining "quiet"

Much harder than in chess. The forcing set must include checks delivered across timelines,
time-travel captures, and branching moves that create a timeline with an immediate threat.
That set is large, and **quiescence may not terminate quickly** — the classic explosion,
worse here.

Standard mitigations (delta pruning, material thresholds) still apply, but the
termination question itself closes by construction: **cap the scan in nodes, not depth**
(dlshogi caps leaf solver calls at ~50 nodes). Termination becomes unconditional, and the
open question reframes from existential to empirical: **what horizon does a k-node budget
buy at 5D forcing-branching?** Honesty note: 50 nodes ≈ mate-in-3 at shogi's forcing
branching; 5D's is larger, so coverage-per-node is worse and the budget likely bigger.
**Treat the horizon-per-budget measurement — together with the stand-pat / 5D-zugzwang
check above — as one early cheap experiment.** If the horizon a sane budget buys is
uselessly shallow, the verification layer needs a different shape, and better to learn
that before building around it.

#### Scanner eval: material count now, 5D-NNUE later

The scanner's leaf eval is a separate decision from the verifier-model question above, and
it gets the opposite answer.

**Why the role change dissolves the objection.** As a verifier model, NNUE's separability
is disqualifying (table above). As the scanner's leaf eval, the division of labour differs:
alpha-beta enumerates the forcing moves — cross-board captures, checks, travel threats —
explicitly, and the eval is consulted only at *quiet* leaves to adjudicate the resulting
balance. Search handles the dynamics; eval handles the statics. The thing the per-board sum
is blind to is no longer its job.

**Why the fit is unusually good here.**

- **Immutability (§2.4) maps directly onto the accumulator.** A physical move supersedes
  one head board and creates one: `acc(new) = acc(old) − feat(from) + feat(to)`. A travel
  move additionally spawns one timeline head with one added piece. **O(1) accumulator work
  per move regardless of N**, and superseded boards' accumulators are frozen forever —
  computed once, shared across the whole search tree.
- **A running sum over head boards** plus a few global features (timeline count, present
  position, material totals) feeds a tiny head. Sub-microsecond per eval on CPU — order
  10³× more evals/sec than the net, entirely CPU-side, zero GPU contention. That asymmetry
  is what lets the scanner afford deep forcing lines. (Order-of-magnitude claims from
  standard NNUE practice, not measured here — §0's disclaimer applies.)
- **HalfKP-style king-relative features do not transfer** — kings exist in multiplicate
  across boards. Start with plain (piece, square) per board.

**The 5D-specific argument.** "NNUE vs HCE" normally means NNUE against decades of encoded
human positional theory. That theory does not exist for 5D chess — nobody has reliable
intuitions about timeline-count value or cross-time pawn structure to hand-craft — so the
realistic HCE ceiling is material count plus guesses. NNUE learns its terms from
search-labelled data, sidestepping the missing expertise entirely. The gap it closes is
plausibly *larger* here than the well-documented Stockfish gain.

**What it upgrades and what it does not.** The proofs channel is untouched — mate proofs
are game-rule terminal, no eval involved — so the correctness-critical core of this section
does not depend on this decision. The estimate channel is what improves: fewer false flags
(material swings that were actually compensated), fewer false passes, a cleaner
force-expansion signal.

**The constraint that keeps it legitimate.** Train on **scanner-labelled data, never on
transformer outputs**. Bootstrap: label positions with deeper material-rooted scanner
searches, train, re-label with the improved scanner, iterate. Label and train on *quiet*
positions only — Suisho's smart-fen-skipping principle: a static eval trained on in-check
or mid-capture positions learns noise; skip them at dataset construction (one more
consumer of experiment 6b). Different features, different
training signal, different search paradigm — the error profile stays orthogonal. Distilling
from the transformer, or training on its self-play values, quietly converts the verifier
back into an echo (see the governing principle above).

**Caveats.**

- Stand-pat now trusts a learned eval rather than an exact count — a wrong judgement can
  prune lines material count would not. Standard delta-pruning margins mitigate; the
  zugzwang question is unaffected either way.
- Superseded boards as *branching resources* — a past board with a winning position is a
  place you can travel to — are invisible to a head-boards-only sum. Closed by the §11.2
  asset map: a standing scanner annotation rather than an NNUE feature.
- It is a real training pipeline, but the bar is low: it must beat material count, not
  catch the transformer's blind spots.

**Parameters (v1 spec).**

| | Value | Reasoning |
|---|---|---|
| Feature transformer | 768 (piece × square) → 256, **one matrix shared across all boards** | Weight sharing again — §3.1's principle, NNUE edition |
| Accumulator | 256 × int16 per board; frozen (superseded) / incremental (heads) | §2.4 |
| Global features | ~24 scalars (timeline count, present position, material, N) | Appended at the head |
| Eval head | (256+24) → 32 → 32 → cp + volatility flag, int8 | ~10k MACs — sub-microsecond confirmed |
| Destination head | Single board's accumulator + globals → 32 → score | The asset map (§11.2 rung 4) gets its own readout |
| Perspective | Single, colour-flipped | Dual-perspective is a v2 ablation, not a default |
| Total | **~0.22M params, ~0.26 MB quantised** | Suisho-class (256×2-32-32 lineage), deliberately |

The two models differ by ~100× in parameters — the differently-wrong architecture made
literal: 21M of learned judgement audited by 0.22M of exact search with a cheap static
sense.

**How good an evaluator it is — honestly.** In chess, NNUE replaced hand-crafted eval
for on the order of +100 Elo at introduction and has driven much of the gain since — but
that is NNUE *inside a deep search*, evaluated tens of millions of times per second. Its
static eval alone is strong-club level at best: far above hand-crafted eval, far below a
deep net. Ours is more limited still, deliberately — separable, and trained only on
scanner-labelled data. On absolute quality it is a *worse* evaluator than it could be,
because the two things it must be, cheap and differently wrong, both cap quality; if it
were as good as the transformer it would have failed one of its two jobs. Nobody should
later expect the scanner's eval to be *good* rather than good-enough-at-quiet-leaves and
different. The 5D-specific unknown is its gain over the v1 material count (6d); the
prior is that the gap is larger than in chess, since 5D has no hand-crafted positional
theory for material-count to compete with.

**Threat features (v2/v3) — adoptable exactly as far as separability reaches.**
Stockfish's current NNUE generation feeds explicit "who attacks whom" piece-pair features
alongside piece-square inputs — a half-year effort that passed SPRT and let the
accumulator *shrink*, because threats carry evaluative load the piece-square features had
been paying for. The evidence transfers here in two unequal halves:

- **Same-board threat pairs: adopt (v2/v3).** A pair whose attacker and victim share a
  board is a function of that board's content alone — separability holds, the frozen
  library holds, updates are chess-bounded. The same-board restriction makes the feature
  set per board *literally identical* to chess's threat set, which is why the transfer is
  unusually clean: every threat in Stockfish's evidence is same-board by definition.
  Attached ablation (6q): whether the 256 accumulator shrinks in compensation, per
  Stockfish's L1 lesson.
- **Cross-board threat pairs: refused, for structural reasons.** A cross-board pair has
  no home — assigning it to either accumulator makes that accumulator depend on another
  board's content, destroying compute-once-freeze-forever. Deeper: cross-board
  interaction is the NNUE's *designed* blindness, the property that keeps it differently
  wrong from the transformer; feeding it in as features smuggles the transformer's
  territory into the auditor. (Update cost — rays scaling with game length rather than
  eight squares — is real but tertiary.) A frozen-content-plus-live-overlay decomposition
  could technically host head-only cross-board threats; recorded as v3-speculative, not a
  plan.

(An earlier blanket refusal of NNUE threat features contradicted §14.4's own record of
Stockfish's Full_Threats lineage — corrected; the process failure is registered there.)

**Sequencing.** Material count remains v1 — free, exact, unbiased. Adopt the NNUE only
after the termination and stand-pat experiments (above) pass: no point training an eval
for a scanner shape that might change.


#### The redesign after 6b (CHECKPOINT 3)

The brute-force forcing scanner failed its pre-registered root-scan test from 50 boards
up, even after an optimisation pass and at its design thread budget (§15.6): 31.4% at
50–99 boards, 24.7% at 100–199, 2.4% at 200+, against a 50% bar. The profile says why.
Above 50 boards it reached 5 forcing half-turns almost only where it proved a result
outright, and its time went to *exactness*: completing each forcing line into a legal
turn (38–64% of the work at 100+ boards), royal-capture scans (24%) and in-check action
listings (23%). The design imported a chess unit, the forcing move, into a game where a
move is only part of a turn — so every line paid turn-level exactness, whether it could
end in a proof or only an estimate. The redesign has three parts:

1. **Split the scanner by channel.** Exactness — turn completion, all evasions,
   royal-capture checks — is paid only where a proof can result. The estimate channel
   becomes a cheap, inexact local search: capture and threat sequences without turn
   completion. Its findings stay soft priors (H2), so its errors cost search efficiency,
   never correctness — and moves only reach the tree through the generator, which is exact.
2. **The proof channel becomes depth-limited df-pn** over the semimove-decomposed turn
   tree (§8.2's ply split; ftxi's fine tree is the prior art), with hypercuboid-pruned
   completion — so a turn is completed incrementally, not re-solved per line, and an AND
   node's evasions are generated per semimove rather than listed whole. Its answer to "mate
   within 5 half-turns?" is proven, disproven or unknown — never "mate" on an exhausted
   budget (H37). dlshogi's df-pn mate solver is the precedent; the repetition gate is gone
   (H39), and plain df-pn needs no dominance definition.
3. **Incremental royal-attack maps.** Recompute attacks only from changed boards and the
   rays crossing them; frozen boards' attack sets are computed once.

It is decided by 6b′ (§15.6, M15). Until then rung 1 stays dropped and the NNUE deferred,
under their own rules.

#### As built and measured, for the record

The brute-force scanner (engine stages E5b–E5c) was node-capped alpha-beta with iterative
deepening in forcing half-turns. Its forcing set: captures (within a board, across
timelines, through time, en passant), checking moves, and branching moves with an
immediate threat — read as the arriving piece capturing an enemy piece worth at least 300;
promotions off. Rules that outlive it and bind v3's provers too: a loss proof needs the
defender's legal actions enumerated exhaustively, and a capped node yields an estimate,
never a proof; stalemates export as draw proofs; the TT holds estimates only and the
solved store is read first; and disproofs are class-relative — keyed by position, depth and
every option that shapes the attacker's class — and live in a separate disproof store,
never the solved store (BUILD_LOG issues 51–60, 74, 90). Its measured figures are in §15.3.

### 11.2 Asynchronous scanning — the scheduler

> **Status (CHECKPOINT 4):** rung 1 is dropped (6b); rung 3's two-stage screen and rung 5's
> scouting are retired; rung 2's promotion side becomes Guard and its demotion audit moves
> to the offline audit lane (§11.4); rung 4, the asset map, stays. The scheduling
> principles — budget against shared movegen, determinism debt — still hold.

§11.1 defined what the scanner computes; this section defines when. The naive synchronous
version — scanner on the critical path, per leaf — leaves most of its value unclaimed. The
naive opposite — "run it constantly in parallel" — starves the search.

#### The budget correction

The appeal of constant parallel scanning is that the scanner is CPU-side and the net is
GPU-side, so they never contend. Half true: there is no GPU contention. But §11 projects
**move generation as the likely bottleneck**, and the search's own expansion and the
scanner's forcing-line enumeration run on the *same* movegen. Unbounded background scanning
starves expansion — the thing that decides the move. The scanner gets a thread budget and a
priority ladder, not free rein.

#### The priority ladder

| Rung | Task | Trigger | Why this priority |
|---|---|---|---|
| 1 | Per-leaf scan, **pipelined under GPU batch latency** | Leaf enters the eval queue | Near-free: runs while the leaf waits for batching; merges at backup via standard async-MCTS machinery (virtual loss; solved bounds are monotone, so late arrival is safe) |
| 2 | Two-sided PV-transition verification (see below) | PV changes | A refutation found mid-search is a cheap redirection; found at the end it forces a re-search — and the *demoted* side is where the value head's judgement concentrates |
| 3 | Refutation sweep (two-stage — see below) | Idle movegen capacity | Low-visit **and uninstantiated** children of high-value nodes — where a policy miss hides |
| 4 | Superseded-board asset map | Once per newly superseded board | See below |
| 5 | Frontier scouting | Idle, optional | Speculative; carries the §12.2 contamination risk |

**Rung 1 should be built regardless of the rest**: it makes the §11.1 scanner nearly
latency-free using machinery (virtual loss, late merge of monotone bounds) that async MCTS
needs anyway. *(Superseded: rung 1 failed 6b in every range and is dropped — §11.4.)*

#### PV transitions are audited on both sides (rung 2)

A PV change is one event with two sides, and the original spec audited only one.

**Promotion side** (as before): the newly chosen line gets a verification scan.

**Demotion side** (new): when a heavily-visited line is abandoned — it was the PV, or
carried visits above a threshold, and selection walked away — the demotion point gets a
scan at **2–3× the routine horizon**, plus an NNUE-vs-value-head disagreement check.

Why the one-sided version was backwards: demoting a heavy line is a high-leverage
value-head judgement — one deep evaluation flipped the search's direction. Under the old
spec the chosen line was end-audited and the never-instantiated moves were screened
(rung 3), while the rejected-after-consideration lines — where the net most actively
exercised judgement — got the least scrutiny. Audit value concentrates exactly there.

Mechanics: if the deep scan refutes the demotion — the feared tactic fails, the abandoned
line stands — the finding enters the **estimate channel** as a prior boost for
re-expansion, never a bound, unless it is an actual proof, which exports as usual.
Demotions of heavy lines are rare — a handful per search — so the escalated budget costs
almost nothing.

Free synergy: demotion points where NNUE and the value head **disagree** are precisely the
gap-revealing positions §12.3 wants — the audit doubles as curriculum mining. A third
check, gated on 6c: a promotion driven by a value surprise can get a conditioning audit —
a node refresh before it takes the PV (§7.9).

#### The refutation sweep, upgraded (rung 3)

Two stages — the second is what makes "the scanner checks the discards" true for quiet
moves as well as tactical ones:

1. **NNUE static screen of the whole discard pile.** Every uninstantiated or low-visit
   move at swept nodes gets make-move + accumulator update + static eval —
   sub-microsecond each, so ~1,000 discards cost ~1 ms of CPU. Any move whose static eval
   jumps against the node's value is promoted to stage 2.
2. **Forcing-line scan** on the promotions, plus all forcing discards regardless of
   screen, producing the usual proof and estimate channels.

Constraints: the screen is an *estimate* — it feeds prior boosts and force-expansion,
never bounds. It inherits NNUE's separable blindness, so cross-board quiet merit passes
through unscreened. And under the v1 material eval the screen is blind to quiet merit
entirely — material does not change on a quiet move — so this stage is a consumer of the
NNUE upgrade (ablation 6d) and another argument for it.

#### The superseded-board asset map (rung 4)

Pieces on superseded boards never move — those boards are travel *destinations*, not
sources. So this scan is not tactical: it is a one-off static assessment of each frozen
board's value as a **branch target** — "the opponent is catastrophically weak here;
travelling in wins."

Immutability (§2.4) makes it almost free: each superseded board is scanned **exactly once
per game, ever**. A few hundred boards × (sub-microsecond NNUE eval + a small qsearch) is
sub-second CPU amortised over the whole game. Output: a standing prior boost on travel
moves into flagged boards. This closes the branching-resources caveat in the NNUE
subsection above — computed by the scanner as an annotation rather than learned as an NNUE
input.

Out-of-window, not-in-E boards get nothing: by construction no legal move touches them, so
there is no finding to have. The adjacent real work is incremental maintenance of E as
search descends — movegen bookkeeping, not evaluation.

As built (v3): on the material estimate, not the NNUE — 4 plies and 128 nodes per board,
flagged at 300, boost 0.25 × min(1, value/900) — every number a placeholder (BUILD_LOG
issue 106).

#### Scan budgets

Event-driven scans were previously unbudgeted; stated now:

| Scan | Budget |
|---|---|
| Rung 1 per-leaf; rung 2 promotion; rung 3 stage 2 | Routine horizon (the §11.1 node cap) |
| Rung 2 demotion audit | 2–3× routine |
| Root-rule final PV scan (§11.1) | Escalated — tens of milliseconds is acceptable once per move, on one line |
| Rung 1, low-confidence or predicted-volatile leaf (categorical-head variance high, or §12.4 volatility prediction high — dual-source) | 2–3× routine — confidence-escalated |
| Asset map qsearch (rung 4) | Small, fixed |

The principle: budget scales with leverage per scan, and the highest-leverage scans — one
line, once, before an irrevocable commitment — are the rarest, so escalation is nearly
free.

**Confidence may add scan budget, never reduce the baseline.** A low-confidence leaf
(variance read off the categorical head, §8.3) triggers the escalated scan — exact
verification routed to where the learned model admits weakness, at CPU cost hidden under
batch latency. The uniform baseline is untouchable: §12.3's mature-regime errors are
*confident*-wrong, and a budget that followed net confidence downward would de-audit
precisely those. Self-rated confidence covers known-unknowns only — the same add-only
shape as §10.4.3's scope rule.

#### Pondering

Running the whole stack on the opponent's clock is standard and orthogonal; it applies
identically to every rung. Mechanics in §8.7: search the post-move position with no
single-reply guess — PUCT allocates across replies by prior, sharpened by §12.4's
opponent-reply head — then re-root on the actual reply.

#### Determinism debt

Async timing kills reproducibility: bound arrival now depends on thread scheduling, and
bound-merge races produce plausible-but-wrong outputs — the same silent-failure class as
cache/precision mismatch (§10.2, companion §8.5). Retain a **synchronous debug mode**
(rung 1 only, forced to complete before eval). An unreproducible engine cannot be debugged
by A/B.

#### Rejected: the ensemble reading

"Run both systems in parallel" as two complete engines voting — transformer engine plus
pure-NNUE engine — fails on §11.1's governing principle: the pure-NNUE engine is
structurally weak here (separable eval, depth collapse under 5D branching), and a weak
correlated voter adds noise, not safety. The asynchronous reading above is the version that
survives.

### 11.3 Scanner parallelisation — Lazy SMP over the ladder

§11.2 allocated the scanner a thread budget; this section spends it. The scheme is Lazy
SMP, and the design constraint that matters is a hashing trap specific to 5D.

#### Why Lazy SMP, and why it fits unusually well

Stockfish's lesson: for alpha-beta, **sharing results beats scheduling work**. Threads
search near-independently and coordinate only through a shared transposition table; this
embarrassingly simple scheme out-scaled careful work-splitting (YBWC) and is how serious
engines now parallelise. §11.2 is already built on that philosophy — the solved-bounds
store is coordination-by-shared-memo — and the ladder improves on the standard setup:
instead of helper threads redundantly searching the same root at staggered depths, **the
rungs are the naturally-diverse helpers**. PV verification, the refutation sweep and the
asset map all write into one shared TT, so a line refuted by any rung is instantly dead
for all of them.

#### Shared TT, thread-local histories

The engine convention — one shared transposition table, per-thread history tables — holds
here for a sharper reason than write contention. The rungs *are* the threads, and they
sample different distributions: PV verification walks principal lines, the refutation
sweep walks policy discards, the asset map walks frozen boards. Pooling their ordering
and correction statistics would average across distributions and homogenise exactly the
path diversity that makes Lazy SMP work. So: killers, history, continuation history —
and the v2 correction table when it lands — are **per-rung**, carried across moves within
a game with decay (§8.7), and cleared between games.

#### The TT is worth more here than in chess

Transpositions come from commuting move orders, and 5D manufactures them structurally: a
turn contains moves on several boards, moves on different timelines usually commute, so
every multi-board turn generates a factorial family of orderings reaching the same
multiverse. The scanner will hit the TT constantly. Budget its size accordingly.

#### The trap: XOR-Zobrist cancels in 5D

The standard multiverse hash — XOR of per-board Zobrist hashes — silently aliases whenever
two boards have identical contents, because a⊕a = 0. In chess that is a curiosity. In 5D
it is **guaranteed**: every travel move creates a near-copy of the destination board on a
new timeline, and early-game boards are identical across timelines by symmetry.

The fix that keeps O(1) incrementality: hash board *contents* coordinate-free
(incrementally updatable per move as usual), then mix each board's content-hash with its
(timeline, time) key through a non-cancelling operation — multiply-rotate, not XOR — and
combine the mixed values. Cheap; it must be designed in, not discovered months later as
evaluations that are mysteriously, irreproducibly wrong.

**One more key ingredient, from §12.6: a ruleset ID in every exact-store key.** The
solved store, scanner TT and asset maps hold game-theoretic *facts*, and a fact proved
under standard rules is simply false under Reversed Royalty — same position, different
checks, different evasions, different mates. Cross-variant contamination of the solved
store would be the worst version of §12.6's false-proof warning: permanent, silent, and
imported rather than computed. Fold the ruleset ID into the position hash as one more
multiply-rotate ingredient. **And the turn phase** (CHECKPOINT 4): under Q4, a state where
the present has shifted but the player has not submitted has the same boards as the
opponent's turn start — only the acting player differs. So the acting player is part of
every state key; without it, graph search merges the two states and can prune a legal
mating submission, which the search's review found. The scanner itself needs per-variant care beyond eval values
— Reversed Royalty changes check detection and evasion generation, the exact machinery
that makes proofs proofs — and NNUEs, at 0.22M, are simply per-ruleset.

**The full key, as built:** each board's coordinate-free content hash mixed with its
(timeline, time) by multiply-rotate, the mixed terms combined by 64-bit addition, plus the
ruleset ID, the acting player, the even/odd start, each timeline's creator and the board
size — structure variants share rules but not positions, and a 5 × 5 position collided
with the same pieces on 8 × 8 until the size was added. MCGS node identity adds each
timeline's branch origin and whether a turn is in progress (§8.5).

#### Proofs never live only in the TT

The TT is a lossy cache: entries are evicted and overwritten by benign races. Fine for
estimates and move ordering; fatal for proofs — a mate bound that exists only in the TT
can vanish. Rule: **proofs export to the permanent solved store (§11.1) the moment they
are found**; the TT never holds the sole copy. Same class as the cache-precision rule
(§10.2): cheap to state now, silent Elo loss if learned later.

#### Determinism debt, extended

TT write races add a second layer of nondeterminism on top of §11.2's async timing. The
synchronous debug mode extends accordingly: **single thread, deterministic TT** (fixed
replacement policy, no races). Lazy SMP's character is benign nondeterminism — acceptable
in production, banned while debugging.


### 11.4 The verification layer, v3 — after the design review (CHECKPOINT 4)

Brute-force and df-pn verification both failed their pre-registered tests from 50 boards
up, and ten times the budget barely moved them (§15.6): the trees of forcing lines are too
large, because in 5D every "move" is a turn spread across many boards. Exact *settling* —
proving a forced result or proving there is none — works only in small positions. The bet
on a permanently-on exact verifier therefore holds only there. Above 50 boards the engine
stands where Lc0 and KataGo do: a network plus search, made robust by training. The
layer in force:

| Part | What it does | Where |
|---|---|---|
| Exact rules | Generator, legality and terminal detection, behind the gate (§11.0) | Always |
| Search-native proofs | The search's own solver (H41): proves what the search reaches, so it scales with the search rather than with the width of the forcing tree | Every size, gated |
| **Guard** | Before the chosen move is played: can the opponent force mate against it? Iterative deepening from depth 1 — depth d is the opponent's d actions (2d − 1 half-turns), so depth 1 is their mate in one; under the ply split it is checked at submission. Exact at the depth that completes; a refutation rejects the move and triggers a re-search (§8.6's reserve) | **Adopted in no range** (6b″) — kept as the audit lane's offline question |
| **Hunt** | At the root: do *we* have a forced mate? df-pn in prove-only mode — when the budget runs out it stops, claims nothing and shares no disproofs. Its forcing class is checking first moves, each completed into one canonical turn (a placeholder). A mate it finds is a proof, gated like any other: export stays off until the proof channel is signed off (H41). Partial attacking lines may be suggested to the search as soft priors, behind a flag, with §12.2's exclusions | **Under 100 boards** (6b″) |
| Full settling scans | df-pn at the root and on PV changes | Under 50 boards, where it passed |
| Offline audit lane | Samples self-play decisions — PV changes, demotions, final moves — and runs exact proof search on them with large budgets, off the critical path. Contradictions with the network feed the curriculum through position selection (§12.2's safe channel) and the adversarial suite | Every size; weak at 200+ |
| Adversarial league | The main robustness mechanism above 50 boards, with attackers aimed at large positions and attack cost measured per board range (§12.7) | Primary above 50 boards |

**Depth adapts by itself.** Guard and Hunt run under a fixed time budget with iterative
deepening, so they go deeper in simple positions and shallower in complex ones. No
complexity model sets their depth — it would only predict what the search finds anyway.
What changed is the *question*: settling needs disproof, the expensive half, and Guard
and Hunt each need only the cheap half for their purpose.

**Retired:** the per-leaf estimate channel (rung 1 — it fits its budget nowhere), rung 3's
refutation sweep and rung 5's scouting (both relied on cheap per-node scans), and the
NNUE's place in the build plan — its main job was evaluating the scanner's leaves, so it
returns to held open (§14.4). The asset map stays: it runs once per superseded board and
costs little, on a material-based estimate.

**Knock-ons.** Time management loses the scanner as a second volatility source above 50
boards; the network's own variance stands alone there (§8.6). "The exact layer stays on
at match play" (§12.3) now means Hunt under 100 boards, the solver at every size and full
settling under 50 boards; Guard runs nowhere at match time. The window's blindness and the verification gap now sit in
the same place — large positions (§13). The window's mitigations never relied on the
scanner, so its case stands, but large-position robustness rests on training.

**Whether shallow exact checks earn their time is measured, not assumed.** Trained
networks rarely blunder shallow tactics — their confident mistakes tend to be deep — but
5D's tactics across time may stay error-prone for longer. The offline audit lane reports
how often the network's chosen move allows a shallow forced mate, per board range. Guard
and Hunt are decided by 6b″ (§15.6).

**6b″ result (quiet machine): Hunt adopted under 100 boards, Guard in no range.** Guard's
depth 1 completed in 65.1%, 15.8%, 5.2% and 0.0% of roots (under 50, 50–99, 100–199 and
200+ boards) against a 95% bar. "Existence, not the full listing" removes only *our*
listing: exactness still needs every opponent action, a product over their boards, and
Lazy SMP's helper threads only re-search the same product. So the chosen move goes
unguarded at match time everywhere; Guard survives as the audit lane's offline question.
Hunt found 69.0% and 63.6% of the established forced mates under 50 and at 50–99 boards —
adopted there — and 35.0% at 100–199, not adopted (200+ had a single established mate, too
few to read). Its root is expanded eagerly, which caps it above 100 boards; a lazily
expanded root is the likely remedy, held open behind a fresh pre-registered test (§14.4).
Above 100 boards, proofs come only from the search's solver.
---

## 12. Training

Less developed than the rest of this document; flagged as such.

- **Curriculum over N.** Weight sharing gives runnability at any N, not competence.
  Attention entropy grows with token count, so a model trained on 5-board games degrades
  badly at 40. Train across the range.
- **Consider entropy-invariant attention scaling** (log-length temperature) as a
  mitigation.
- Self-play, AlphaZero-style, with the policy/value targets from MCTS visit counts.
- The causal constraint (§7.3) must be present during training if used at all, not bolted
  on at inference (§7.7).

### 12.1 The strongest argument for caching, recorded here rather than in §7

**Self-play generation is typically 90%+ of total compute in an AlphaZero pipeline, and it
is all forward passes.** A 10× on eval throughput is therefore close to 10× on the entire
training budget — and training budget plausibly binds final strength harder than per-move
node count does.

So the real trade is not "playing strength now with dense vs cached." It is **10× the
self-play data versus an unknown Elo cost from causal masking.**

This does not change the recommendation — you still cannot price the causal cost without a
dense baseline, and §7.5 may make the question moot — but it is the reason to revisit the
decision once, deliberately, after the window is measured and before committing serious
self-play compute.

### 12.2 Scanner steering must not contaminate policy targets

*Since CHECKPOINT 4 the steering sources are the asset map's boosts and Hunt's flagged
priors (§11.4); the rule below is unchanged.*

AlphaZero-style policy targets are visit distributions. Rungs 2–5 of §11.2 deliberately
steer visits — prior boosts, forced expansions. Fed to policy training unfiltered, those
steered visits train the net toward the scanner's material-greedy bias, violating the
differently-wrong property (§11.1) through the training loop rather than at inference.

Value targets are safe — game outcomes do not inherit steering. Policy targets need one of:

- **Exclusion**: scanner-forced visits removed from the target distribution (cleanest;
  loses some signal), or
- **Provenance**: scanner-injected visits recorded and down-weighted.

Which one is an open question. What is not open: shipping rungs 2–5 into self-play without
one of them.

(§12.4 extends this boundary to a third channel. The full accounting: **visit
distributions** — exclusion or provenance; **position selection** — safe; **loss
targets** — never learned-verifier output; own-search targets are the norm (§12.4).)

One scope note: the boundary must survive tree reuse. A carried tree (§8.7) imports
visits — including scanner-forced ones — from the previous move's search, so exclusion
would need provenance that crosses moves. Self-play therefore runs fresh searches per move
by default (§8.7); match play, which emits no targets, carries freely.

The policy target is the visit distribution over moves *and* submit (D8); exclusion and
provenance cover the submit child like any other.

### 12.3 The mature regime

Near convergence the system inverts in ways worth designing for rather than discovering.

**The failure mode inverts.** Early, top-k misses are frequent but random — noise and
widening cover them statistically. Late, misses are rare but *systematic*: self-play
converges policy and value together, so they share blind spots, and search cannot correct
what its own priors steer it away from. This is the documented adversarial-Go result —
superhuman engines beaten by trivial cyclic strategies their policies assigned near-zero
prior. Entropy-adaptive widening (§8.4) covers the case where the policy knows it is
uncertain; the confident-wrong case it cannot. The same limit holds for every
confidence-driven mechanism — value-variance cPUCT, §8.6's extensions, §11.2's scan
escalation: self-rated confidence covers known-unknowns, which is why the
confidence-independent layer (the uniform scan baseline, the demotion audit, the
adversarial suite) never scales down with it.

**The scanner's relative value rises — do not strip it.** The tempting mature-engine move
is removing safety layers because the net is strong. Backwards: the scanner is the only
component whose coverage is distribution-free — rules, not learning — so it cannot be
converged-toward. As the policy's residual errors become exclusively confident-systematic,
the differently-wrong auditor covers precisely the class nothing else does. It stays on at
match play, permanently. (Since CHECKPOINT 4 that means Guard, Hunt and the search's solver
at every size, with full settling scans only under 50 boards — §11.4.)

**Late self-play goes stale; the verification layer is the curriculum.** A converged
policy generates games the net already masters — near-zero gradient signal. The
gap-revealing positions are the ones the verification layer flags: the offline audit
lane's contradictions with the network (§11.4), Hunt's proofs against the network's view,
and settling-scan disagreements under 50 boards. Seed self-play
from them; §12.7's adversarial league is the second generator of such positions, at
scale. A third, **uncertainty-prioritised sampling** (prioritised-experience-replay
lineage), weights the training distribution toward positions the model is unsure about
— under two conditions. **Epistemic, not aleatoric:** raw categorical variance conflates
knowledge gaps with genuinely volatile positions where 50/50 is the *correct*
assessment, and prioritising the latter pours compute into irreducible uncertainty
(PER's known failure mode). The trigger is the error head's *realised* error, or better,
uncertainty that *persists across checkpoints* for a position class — the N/phase bucket
drift signal of §14.4; bucket-level persistence is epistemic, per-position variance is
often just the game being sharp. **Additive, never replacing:** uncertainty-driven
sampling covers known-unknowns by construction, and the confident-wrong class has *low*
uncertainty and would be systematically under-sampled by it — the audit lane and adversary
generators exist precisely because they do not consult the model's confidence (ablation
6u). §12.2-safe: the boundary there is
about *targets*, and position selection leaves targets untouched — they still come from
search outcomes.

**The metric warning.** Training on prover-caught errors migrates the net's remaining
errors beyond the provers' horizon — deep and quiet. A declining catch-rate late in
training is a *biased* health metric, not evidence of correctness. The honest probe for
the confident-quiet residual is held-out adversarial evaluation, in the spirit of the
attacks that exposed the Go blind spots. (§12.7 makes this continuous: attack cost
against held-out adversaries.)

**Match-play configuration.** Dirichlet noise off, temperature zero, verification v3 on (§11.4), and
moves-left steering active — prefer faster wins and slower losses once the outcome is
settled (Lc0's MLH pattern). The
now-calibrated value head earns resign thresholds and early adjudication in self-play,
buying back a meaningful slice of generation compute. Self-play also carries a **turn
cap defined from the state**, scored as a draw — a convention for ending runaway games,
not a game rule (5D has no repetition rule to end them, §11.0 item 5); log how often it
binds. As built: t5d-play's default cap is 300 boards, and a capped record carries the
result "*" with a Termination header (BUILD_LOG issue 88).

### 12.4 Auxiliary training targets

Sample efficiency is the binding resource (§12.1: generation is ~90% of the pipeline), and
auxiliary targets are the best-documented lever on it — KataGo's ownership and score
targets were among the largest contributors to its efficiency gains over vanilla
AlphaZero, and Lc0 adopted WDL and moves-left heads for related reasons. The case here is
stronger than the precedent, and it is structural.

#### Supervision sparsity scales with N

T = N×66+16 is a function of the position, not a constant: ~82 tokens on the opening's
single board, ~15,856 at the 240-board reference state, ~40,000 in a 600-board endgame.
One outcome scalar and one visit distribution supervise a token count that varies ~500×
across a game. Against Go's fixed 361 points per outcome: this design is
*denser*-supervised than Go at the opening and roughly 100× sparser in the late
multiverse. (The policy target partially self-scales — more boards, more legal moves — but
§12.3 establishes its effective information collapses as the policy sharpens, so it does
not rescue the late-game ratio.)

#### Granularity is the ranking criterion

| Granularity | Signal per position | Per-token supervision as N grows |
|---|---|---|
| Global (WDL, moves-left, reply prediction) | Constant | Dilutes ∝ 1/N |
| Per-board | ∝ N | **Constant** |
| Per-square | ∝ T | **Constant** |

Global targets are cheap and proven, but they dilute exactly as the main value target
does. Per-board and per-square targets are the only class whose supervision keeps pace
with the token count — the only class that addresses the scaling structure rather than
adding a constant.

#### The menu

| Target | Granularity | Source |
|---|---|---|
| WDL value (replaces scalar) | Global | Game record |
| Moves-left — counted in turns, so both D2 arms share it | Global | Game record |
| ~~Opponent reply prediction~~ — dropped from v1 (CHECKPOINT 6): Lc0's own training config weights such "future heads" zero as unstable, and here the reply's move space is undefined | Per move | Game record |
| Short-horizon value (t+k bootstrap) | Global | Search record |
| **Travelled-into within k plies** | **Per board** | Game record |
| **Timeline still active at game end** | **Per board** | Game record |
| Tactical volatility (forced swing within horizon) | Per board | **Proofs** — the solver, Hunt, the settling scans, the audit lane — exact |
| Deep attack facts beyond the §6 input bias (label set a placeholder; 6j) | Per square | Generator — exact |
| Search effort to convergence — distributional over log-visit bins, censored top bin (§8.6) | Global | Search record — full-search moves only |

The bolded pair is the KataGo-ownership analogue and the highest-value entry: dense
supervision on exactly the cross-time reasoning that is hardest to learn from a terminal
outcome — **which boards come to matter**.

Value and WDL targets are from the acting player's perspective (D8), and the policy target
covers moves and submit alike (§12.2).

#### The source rule — §12.2's third channel

An earlier revision said "exact sources only: game record, generator, scanner proofs."
As written that prohibits AlphaZero itself — the policy target is a visit distribution,
search-derived by definition, and the menu's short-horizon row is search-sourced. The
actual rule:

> **Own-search-derived targets are the AlphaZero norm. The prohibition is specific: the
> learned verifier's estimates never enter any loss.**

The asset map draws the line sharply: "this board *was* travelled into later" is a
game-record fact and an excellent target; "NNUE rates this board as a good travel asset"
is the same concept from the wrong side of the line — training on it wires NNUE's
separable blindness into the transformer, destroying the differently-wrong property
through the loss function.

#### Two synergies

- **Against §7.3's cost.** Under the cached configuration, frozen boards cannot be shaped
  by present relevance. A travelled-into head trains frozen representations to *predict
  their own future attractiveness as destinations* from content alone — not the same as
  seeing the present, but pressure exactly where the causal constraint hurts, and it
  compounds with §7.9 rather than overlapping it.
- **Against §12.3's staleness.** Policy targets flatten as self-play converges; auxiliary
  losses stay informative on converged data. The dense targets matter most exactly when
  vanilla signal is weakest — which is also how the curriculum falls: early training
  (small N) free-rides on mild sparsity, late training leans on the auxiliaries.

#### Costs and knobs

- **Loss weighting is a genuine new hyperparameter surface** — badly weighted auxiliaries
  interfere with the heads that play. KataGo needed tuning here; so will this.
- **Reduction over boards is an N-structured knob.** Summing per-board losses grows the
  auxiliary gradient share with N — arguably right, that is where sparsity bites — but
  risks swamping policy and value in big positions. Averaging is stable across the
  curriculum but dilutes with everything else. Neither is obviously correct; it lives in
  ablation 6j explicitly rather than surfacing as unexplained late-training instability
  when the N-distribution shifts.
- **Per-board labels are policy-dependent.** "Travelled-into" reflects what *this* policy
  did, so the target is non-stationary as play improves. KataGo's ownership target shares
  the property and works; note it, don't fear it.
- **Attack-map targets partially overlap the §6 input bias** — predicting what the model
  is handed as input teaches nothing. The row is specified as facts *beyond* the input
  bias, and it is the first candidate to cut if interference shows.
- **Registers stay free.** Auxiliary heads pool from the trunk; hard-assigning registers
  to auxiliary quantities conscripts the scratch space §5 exists to provide.
- Head compute is negligible — linear heads over d=256.

### 12.5 Playout cap randomisation

KataGo's generation scheme, adopted: most self-play moves run at a small playout cap, a
minority at full search, and **only full-search moves emit policy targets** — value
targets need quantity (many games), policy targets need quality (real searches).

Two reasons it is worth more here than in Go:

- **§12.4's per-board targets are game-record targets** — travelled-into and
  timeline-survival labels scale with game count exactly as value targets do, so cheap
  games multiply precisely the target class ranked highest. PCR attacks the evals/sec
  bottleneck (§12.1) on the two supervision channels that matter most.
- **It compounds with §12.3's resign and adjudication thresholds** — cheap games that
  also end early.

Mechanics: the no-policy-target flag is §12.2's existing plumbing; adopt KataGo's
forced-playout pruning with √(playouts) decay at target construction — the enforcement
point §12.2 already mandates. One dependency: at cap ≈ 100 against branching ≈ 1,000,
search without §8.4's narrowing is pure policy sampling — PCR presumes the selection
machinery, so ablation 6l runs after 6g. One temptation to resist: biasing which moves
get full searches by predicted difficulty (§8.6's effort head). Full searches are the
only source of policy targets, so pure difficulty-biasing starves easy positions of clean
targets and play in simple positions can rot; if used at all, a mixture with a random
component.

### 12.6 Variant generality

5D chess ships with variants — alternate starting positions (Defended Pawn), altered
lattice structure (Turn Zero's pre-game timelines), and altered rules (Reversed Royalty's
royal queen; Princess, Brawn and Common King pieces). They sort into three tiers:

| Tier | Examples | Adaptation requires |
|---|---|---|
| Position variants | Defended Pawn, alternate setups | Nothing — different initial states under identical rules; no starting position was ever assumed |
| Structure variants | Turn Zero, multi-timeline starts, small boards (5×5–7×7) | Coordinate and size handling only |
| Rule variants | Reversed Royalty, new piece types | Generator work + piece vocabulary + retraining or conditioning |

**Five decisions carry the generality**, none of them made for this purpose:

- **Rules live in the generator** (§11): legality, check and termination were never in
  the net — a royal queen is a generator change plus an embedding row.
- **The pointer policy head** (§8.1): fixed AlphaZero-style move planes break on any
  board-size or move-set change; a head that scores generator-supplied moves over
  whatever tokens exist does not. A 5×5 board is 27 tokens per board, and nothing else
  changes.
- **Per-square tokens, one weight set** (§3): no lattice shape assumed.
- **Relative RoPE** (§4): no origin assumed — Turn Zero's pre-game boards are just more
  lattice. (One convention note: 5D's ±0 timeline pair needs an index mapping so the
  encoding does not see two distinct timelines at distance zero. As built: −0 → −1, +0 → 0,
  −n → −(n+1), +n → n, so adjacent timelines differ by one.)
- **Global registers** (§5): the natural home for variant-conditioning flags.

And the curriculum bends the right way: small-board variants are *cheaper* than the
standard game — nearly-free training diversity.

**Costs, in descending order:**

- **Exactness is inherited per variant.** The verification layer's guarantees come from
  the generator, so a subtly wrong Reversed Royalty generator does not just misplay — it
  manufactures **false proofs** that enter the solved store as permanent facts. Variant
  support means a generator correctness suite per variant; this is the one place where
  "adapt" costs engineering rather than configuration.
- **Weights do not transfer where values invert.** Reversed Royalty flips the protection
  economy — shield the queen, spend the king — close to an adversarial distribution
  shift for a standard-trained net.
- **Accumulating small items:** material values per new piece (H50); NNUE
  feature-table extension and retrain if the NNUE returns; per-variant repetition terminal values — none needed,
  since no variant adds a repetition rule and the full state cannot recur (§11.0
  item 5).

**One net or many.** KataGo trains a single net across board sizes and rule sets via
input conditioning, and gains from the shared game-sense. This design is better placed
still: the pointer head and generator-driven masks mean even variants with different
state and move spaces share one architecture, and conditioning flags plus piece
embeddings carry rule identity. Prior: the conditioned generalist wins everything except
peak standard-rules strength, where a specialist keeps a small edge — the generalist
tax, measured by ablation 6r and possibly worth paying for the training diversity
alone.

**The grouping, settled — one principle does the sorting: conditioning is only needed
where identical states carry different rules; where states self-identify, variant
training is just opening diversity.** Turn Zero, Defended Pawn and Half Reflected are
rule-identical and self-identifying — a Defended Pawn game announces itself in the piece
placement, Turn Zero in its visible pre-game boards — so they need no flags and no
separate model: they fold into the standard net's curriculum as free diversity (broader
opening distribution, less opening-theory overfit, and Turn Zero exercises the
multi-timeline machinery earlier in the curriculum).

**Lineage: the diversity-trained standard net is the baseline every specialist refines.**

| Model | Trained | Notes |
|---|---|---|
| Standard net | Standard + Turn Zero + Defended Pawn + Half Reflected | The baseline; no conditioning flags needed |
| Reversed Royalty specialist | Warm-started from the baseline | Value inversion justifies the specialist; movement geometry, travel mechanics and tactics still transfer. **Scratch arm as control** — inherited king-protective bias is the named risk (6r) |
| Princess | Warm-started specialist, interim | Self-identifying piece, normal royal structure — the test case for folding into the conditioned main net (6r); specialist is the default until 6r says otherwise |

Warm-starting from the diversity-trained baseline beats warm-starting from a
plain-standard net: broader opening coverage and more robust representations make a
better fine-tuning base.

### 12.7 The adversarial league

§12.3 committed to adversarial *evaluation*; this section upgrades it to adversarial
*training* with a population, on two precedents. AlphaStar's league: main agents, *main
exploiters* trained to beat the current main agent, *league exploiters* trained against
the population of past agents, and prioritised fictitious self-play (PFSP) — opponents
sampled by how badly you lose to them. And the adversarial-Go lineage the design already
cites, whose follow-up finding shapes what success can mean: iterated adversarial
training raised the cost of attack substantially and never closed it — fresh adversaries
kept finding new holes.

**Boundary compliance, first.** The adversary supplies *positions* — the games it forces
the main net into — which is §12.2's explicitly safe channel. It never supplies a loss
target; the net trains on its own search outcomes in those games, as always.

**Mechanics:**

| Element | Specification | Why |
|---|---|---|
| Exploiters | Spawned per checkpoint, trained against the *frozen* main net | An exploit needs a stationary target; per-step is too expensive and too mobile |
| Economics | Narrow, not strong — ~10–20% of generation compute | Finding one hole is cheap; playing well is not. The KataGo adversary lost to amateurs |
| Population play | PFSP: n cheap test games (PCR-style caps) against the league, then training weighted by loss rate — variance-shaped, below | No compute spent beating opponents already beaten; the test games *are* the forgetting check |
| League pruning | Retire adversaries beaten decisively across several checkpoints; keep a representative set | Test cost is linear in league size |
| **Held-out adversaries** | A fraction of exploiters never trained against | The train/test split over adversaries — beating every adversary you trained on measures memorised exploits, not robustness |
| Heterogeneous attackers | Warm-started copies with varied search settings, a full-scope (unwindowed) net, a prover-heavy configuration (Hunt and the settling scans at large budgets) | Differently-wrong applied to attackers: diverse exploiters find diverse holes |
| Size classes | Tiny to large, reusing the §9.2 ladder architectures; small full-scope and small from-scratch variants | Exploit-class coverage and the robustness curve — not cheapness (below) |

**The weighting is variance-shaped, not monotone in loss.** An opponent winning 95%+
gives almost no learning signal — search cannot find the improving move if the exploit
lies beyond its horizon, so every game is lost identically and gradients carry nothing.
The informative band is roughly 30–70% win rate, where outcomes vary and search can
discover the difference: weight heaviest around 50%, tapering at both ends (AlphaStar's
lesson). The hopeless tail is a **diagnosis queue, not a training queue** — games against
opponents the net cannot beat become material for the question *what is the exploit?*,
a scope hole (§10.4.3's tuning loop), a generator bug (§12.6), or a confident-wrong
evaluation (§12.3's curriculum seeds), and are routed accordingly.

**When the league starts: a measurable trigger, not a fixed time.** §12.3's inversion
answers "why not from the start." Early errors are frequent and random; plain self-play
exposes them densely and is the most efficient signal available. Exploiters trained
against an early checkpoint learn nothing transferable — everything is a hole, so
"exploit" degenerates to "play competently" — and the main net changes so fast that
exploits go stale before they are trained against. Adversarial training earns its
compute when errors turn sparse and systematic. Trigger on any of: plain self-play win
rate against recent checkpoints saturating (π ≈ search(π) approaching its fixed point);
root policy entropy dropping below a threshold; the audit lane's catch-rate flattening (biased as
a correctness metric, fine as a phase indicator). **Monitor from the start, train from
the trigger**: league test games run early — cheap, and they establish the attack-cost
baseline curve — while training against adversaries waits.

**The metric is cost of attack.** Not win rate against known adversaries — biased, the
§12.3 warning again — but *how much exploiter-training compute a fresh held-out adversary
needs to reach X% wins*, per size class (below). Compute, not steps: steps are not
comparable across attacker sizes. Rising attack cost is robustness; a falling defended win rate is
memorisation. Standard match Elo runs alongside: adversarial training can tax ordinary
strength, and the tradeoff is measured, not assumed (ablation 6s).

**Adversary size — coverage and a curve, not cheapness.** The obvious case for small
exploiters is that they are cheap, and it is mostly false: every game against the engine
pays for the engine's search, and those evaluations dominate the bill.

| Adversary | Params | Eval cost vs main | Game cost, own-net victim model | Game cost, victim-modelled |
|---|---|---|---|---|
| Tiny (d128 L8) | 2.6M | 0.22× | 0.61× | 0.80× |
| Small (d192 L12) | 8.8M | 0.52× | 0.76× | 0.88× |
| Same as main | 21M | 1.00× | 1.00× | 1.00× |
| Large (L24 ff12) | 44M | 1.66× | 1.33× | 1.17× |
| Tiny, full-scope | 2.6M | 1.05× | 1.02× | 1.01× |

(Main net windowed at k = 8; game cost relative to a same-size adversary.) An
eight-times-smaller exploiter makes games only ~1.25–1.6× cheaper. The real cost lever is
**victim search budget**: train against a reduced-sims victim with sims raised as a
curriculum, and always *test* at match-play configuration — exploits found against
shallow search may not survive full search.

What size buys instead:

- **Exploit-class coverage.** Small nets find cheap, simple holes (the Go cyclic attack
  was a simple pattern); large nets can represent multi-stage quiet plans — the class
  §13's residual lives in.
- **Affordable variants.** A small full-scope attacker costs about what the windowed
  engine does per eval: the dedicated scope-exploit hunter at no premium. Small
  from-scratch exploiters pretrain cheaply in small-versus-small self-play (neither side
  is the engine), and from-scratch priors escape inherited blindness — a warm-started
  exploiter explores by the victim's own priors, the §10.4.3 compounding loop applied to
  attackers.
- **Robustness as a curve** — attack cost against attacker capacity. Small attackers
  still winning late in training means cheap holes remain: the most alarming signal the
  league can produce. Only large ones winning means the holes are deep, which is expected
  — attack cost rises without closing.

**Bounds.** *Floor*: an exploiter must be strong enough to reach the positions where the
exploit lives; below that it loses every game and is pruned — wasted training. *Ceiling*:
at or above the main net's size, wins may be general strength rather than a specific
hole, so every adversary is scored for **specificity** — its win rate against the victim
minus its win rate against a reference pool of older checkpoints. The Go adversary lost
to human amateurs: high specificity, a genuine exploit. A large adversary that beats
everyone is not an exploit finding; it is evidence the main net is under-capacity, routed
to ablation 5.

**Mechanics.** Held-out adversaries are **stratified by size class**, so the curve is
measurable; pruning keeps representatives **within each class**; the exploiter budget is
allocated across classes by **exploit yield per unit compute** (a bandit), so escalation
to larger classes emerges as small ones stop finding holes. **Victim modelling** — the
exploiter's search expands victim nodes with the frozen main net, as the Go adversaries
were trained with gray-box access to their victim — buys sample efficiency at the
victim-modelled cost column. Prediction on record: early in training, small exploiters
find most holes; at maturity, only large or from-scratch ones do.

**Why the exact layer makes the league fit unusually well — in small positions.** Where
exact verification reaches depth (under 50 boards for settling, under 100 for Hunt), it
strips *forcing* exploits at match play — proofs catch them, the audit lane logs them — so
adversaries are pushed toward *quiet* exploits there; above that, forcing exploits stay
open and the league is the primary defence (§11.4). The quiet residual is long positional traps,
confident-wrong evaluations. That is precisely §13's residual class, and adversary-found
positions are precisely the gap-revealing seeds §12.3 says late self-play starves for.
The league is the missing generator for a curriculum already asked for. It doubles as a
merciless generator test suite — adversaries find rules bugs if any exist — closing a
loop with §12.6's per-variant correctness requirement.

**The window, answered honestly.** The window creates a structural exploit class — quiet
multi-ply setups toward boards neither reachable, near-reachable nor asset-flagged — and
adversaries will find it, because scope vulnerabilities cannot be trained away: the net
cannot learn to see boards it is not shown. Three consequences. This is 6h's
window-targeted suite *automated* — manufactured at scale instead of hand-constructed.
Every exploit reveals a board that should have been in scope, so adversary findings are
the tuning signal for W's filler tiers (§10.4.3), and the add-only rule makes that
evolution safe by construction — the league drives scope-rule refinement, not just net
training. And the unwindowed net is not immune — the Go attacks succeeded against *full*
observation, because the vulnerability is confident-wrong evaluation, not missing input —
so the fair comparison is exploit-discovery rate at matched adversary budgets, which the
league supplies continuously as cascade kill condition 2 (§10.4.2). Prediction on record:
windowed shows the higher initial exploit rate on scope-targeted attacks; filler tuning
narrows the gap; whether it closes is open — with the equal-wall-clock node advantage
possibly letting the windowed net search its way out of traps the unwindowed net walks
into.

---

## 13. Where this is most likely wrong

**21M parameters may be too few.** Lc0 and AlphaZero sit at 20–50M for a *64-token*
problem. This is far harder. The counter is the standard search-engine trade: a net 30%
stronger per eval and 3× slower loses, and that trade is more brutal here than in chess.
Train 21M and ~60M, match at equal wall-clock, believe the result.

**d=256 may be too narrow for 4D geometry.** RoPE across four axes shares d_head four ways;
8 dims per axis is thin. First fix is §4.1's asymmetric, per-head allocation; d_head=64
(4 heads) is the second resort, before d_model=512. This is also why GQA is
rejected (§10.2) — it would shrink the key space on precisely the axis that is already
marginal. Lc0's evidence weakens the worry: its chess transformers did best with small
head depths, improving down to a head depth of 8 with smolgen — though with learned
position biases rather than 4D RoPE, so it is suggestive, not settling. The displacement bias (§4.1)
now carries the joint-axis geometry RoPE cannot, which relieves the d_head budget further.

**16 layers assumes unified attention.** Under the §3.3 contingency each layer covers one
axis, so reach is much shorter — 24+, which is itself part of why axial gains so little.

**The window may not be 7×.** §10.4's figure is unmeasured, and it is load-bearing —
it determines both the dense build's viability and whether §7's cache is ever worth
building. If reachable history is large, dense at 240 boards is ~337 evals/sec and the
cached path becomes much more attractive.

**The cache's ~10× may itself be optimistic.** It assumes ~8 live head boards against a
substantially shared history. The timeline-count dependence is now measured on real games: on paper,
9.9× at 240 boards with 24 timelines (§7.1). Real MCTS diverges, so effective sharing falls with depth —
possibly sharply.

**Dense may leave too much throughput on the table.** §12.1 is the counter-argument to the
whole §7.4 recommendation: if self-play compute is 90% of the budget, a 10× there may
outweigh an unmeasured Elo cost. **§7.9 strengthens this further** — root-conditioned
refresh shrinks the causal cost for ~4%, so the cached configuration is more attractive than
§7.4 prices it. Revisit once, after the window is measured.

**Search misallocation may dominate evaluation error.** The engine's likely failure mode is
not misjudging positions it examines but never examining the right ones — enormous branching
plus shallow trees. If so, the verification layer matters more than most of the network
decisions in this document — and since CHECKPOINT 4 it is narrower than designed (§11.4).

**Strong moves outside top-k are now a statistically-covered failure class.** The per-node
sweep that caught forcing discards exactly (rung 3) and the NNUE quiet screen were retired
with the scanner (§11.4). Exact coverage of discards remains only where the settling scans
(under 50 boards) and Hunt (under 100) run; elsewhere a discarded move is covered by
entropy widening, root noise and the offline audit lane — and at maturity this is exactly
where confident-systematic errors live (§12.3). Probe with held-out adversarial
evaluation; internal metrics will not see it. The §12.7 adversarial league is its active
probe.

**Exact verification does not reach depth above 50 boards — confirmed.** 6b failed
(CHECKPOINT 3) and so did the df-pn redesign (6b′, CHECKPOINT 4), even at ten times the
budget. Exact settling covers only small positions; above them the layer is Guard, Hunt,
the search's solver and the offline audit lane (§11.4). The confident-quiet residual
widens where games are largest — exactly where the window's blindness also lives — and
the adversarial league is the main robustness mechanism there, not a supplement. How
much this matters depends on how often games get large — see the next entry.

**The design's size premise came from reasoning, not data.** Its reference positions —
240 boards, with 100–600 treated as a normal midgame — were never measured. The first real
data, 761 human games, ends at a median of 44 boards and spends 88% of its moves under 50
(CHECKPOINT 5). If the engine's own games look similar, most compute is in small
positions: the window and the cache matter less, and exact verification, which works under
50 boards, covers most of the game. Every decision weighted by where compute goes — M7,
the cached decision (C2), piece-centric tokenisation — uses measured game-length
distributions: the human games now, the pilot self-play run before serious self-play.
Human games are a proxy, not a forecast: stronger defence could make engine games longer,
and sharper attack could make them shorter. Large positions still matter for robustness,
because games can grow, but they are the stress case, not the typical one.

**Legality classification may dominate self-play in big games.** Measured on random-weight
self-play: 41 of 43 seconds at 336 boards went to classifying move legality (BUILD_LOG
issue 95). Search verifies unknown moves lazily (§8.1), but expansion still classifies
every move under a budget, and the flat turn generator (§8.2) adds a submission check per
generated turn. Cheap legality matters as much as the network's speed in large positions;
its throughput is a §15.3 figure.

**Timeline index is a noisy axis.** Branching order is partly arbitrary. Δtimeline may need
different treatment from the other three.

### 13.0 Estimate revision history

Throughput estimates in this document have moved substantially, and the movement was
error-correction rather than design improvement. Recorded so the current figure is read
with appropriate scepticism:

Held at a consistent 1613 TFLOPS so the figures isolate the actual errors. The originally
quoted ~10/sec also assumed ~500 TFLOPS achieved throughput, so part of that apparent
improvement was learning FA4 exists rather than fixing arithmetic.

| Revision | Estimate | What was wrong |
|---|---|---|
| Initial | ~30/sec | 38k tokens and d=512 — both wrong (quoted as ~10/sec on a 500 TFLOPS assumption) |
| Corrected token count | ~148/sec | 240 boards ≈ 15.9k tokens, not 38k |
| Corrected width | ~337/sec | d_model 512 → 256 |
| Current (dense) | **~337/sec** | — |
| With window, if 7× holds | ~2,370/sec | Unmeasured. Previously quoted as ~3,000 on a 9× attention-only figure |

None of these are measurements.

### 13.1 Ablation order

1. **The |E| phase curve (§10.4.2)** — before anything else, including network
   dimensions; pure movegen analysis, no training. Then the §10.4.2 cascade: the two-arm
   head-to-head (equal wall-clock and equal sims) or scope randomisation, gated before
   serious self-play compute; plus the equal-sims arm at W-depth k ∈ {0, 4, 8, 16}
   (§10.4.3). Prediction on record: k = 0 loses equal-sims meaningfully — temporal
   texture and the coordinate-gap distribution both degrade — and if it ties, take it:
   ties go to the cheapest configuration. Piece-centric tokenisation (§14.4, reversed)
   joins the cascade as a second committed-class arm, tested jointly with the window
   since the two compose — boards × tokens-per-board — and pay most together in endgames.
2. 4D RoPE vs learned bias vs ALiBi-style (§4.1), and on top of RoPE the 4D displacement
   bias vs none (CHECKPOINT 6 — it replaces the binned-bias hybrid arm), then a key-side,
   compact-class Shaw term on top of it (B17), including per-head
   specialisation vs uniform asymmetric vs even split (§4.1) — the N-generalisation test;
   if this fails nothing else matters
3. Registers on/off, 2 vs 4 per board
4. Move-graph heads: 0 / 2 / 4
5. 21M vs the 60M-class L24-ff12 arm (44M — §9.2 ladder) at equal wall-clock; the prior
   now leans larger (§9.2); distilled-small vs from-scratch-small as a third arm (§10.3);
   FFN 2× vs 8× at matched compute, the saved compute spent on simulations (§9.2)
6. Block-sparse vs dense mask (requires board-major ordering — §6)
6a. Fixed multi-head routing (§6) — speculative; may recover axial's inductive bias for free.
    Arm: the layer-level alternative — local/global interleave, within-board-only layers
    with a causal cross-board layer every third (§7.6); committed-class
6b. Scanner horizon per node budget (termination is unconditional under the node cap) and
    stand-pat validity under 5D move obligations (§11.1) — cheap, early, shapes the layer;
    pass/fail thresholds pre-registered in §15.6 — measured: failed above 50 boards
    (CHECKPOINT 3); superseded by 6b′ and 6b″ (§15.6)
6c. Root-conditioned refresh (§7.9) vs plain causal — only under the cached configuration;
    sub-arm: root-only vs root + travel-triggered visit-gated refresh (the existence test
    for within-search drift) — rung 3's attention-weighted partial refresh only if the
    sub-arm shows a signal. The refresh arms must be conditioning-offset-trained (§7.9);
    arm: offset training on vs off, which measures the train/deploy gap directly. Ladder
    variant: the conditioning audit on value-surprise promotions
6d. ~~Scanner eval: material count vs 5D-NNUE~~ — void while the NNUE is held open (§11.4)
6e. ~~Async ladder: rung-1-only vs full ladder~~ — void, rung 1 retired (6b). Live
    remainder: exclusion vs provenance for policy targets (§12.2)
6f. Prover SMP scaling: measured for the retired scanner, and of no help to exhaustive
    refutations (§11.4); verify the proof-export invariant under TT churn for Hunt and the
    settling scans (§11.3)
6g. Selection knobs (§8.4): entropy-adaptive k₀ and widening exponent vs fixed top-k; FPU
    reduction — measured jointly with the §8.2 ply-split decision (branching differs ~5×);
    ftxi's semimove fine tree is the reference implementation for the ply-split arm; the
    widening schedule may depend on the mandatory-board count (§8.2)
6h. ~~Rung-3 quiet screen~~ — void (retired). Live remainder: held-out adversarial suite for the
    confident-quiet residual (§12.3, §13) — generated by §12.7's held-out adversaries
    rather than hand-constructed, with scope-targeting exploiters supplying the
    window-targeted attacks (§10.4.3)
6i. Demotion audit, moved from rung 2 to the offline audit lane (§11.4): the rate of
    demotion-refutations found — a direct probe of value-head reliability on deep lines
6j. Auxiliary targets (§12.4): target-set ablation (global-only vs +per-board vs full);
    loss weightings; sum-vs-mean reduction over boards, checked across the N curriculum
6k. MCGS vs tree main search (§8.5) — measured jointly with 6g's ply-split decision;
    verify residual-backup stability and repetition handling at 240-board midgames
6l. Playout cap randomisation (§12.5): cap ratio and full-search fraction — after 6g
6m. Variance-scaled cPUCT + uncertainty-weighted backups (§8.4), fed by the categorical
    head (§8.3); log the expected 6i catch-rate drop; derived variance vs a trained
    error head, judged on reliability curves by N-bucket (§14.4)
6n. Tree persistence (§8.7): reuse on/off at match play; lazy-recompute fraction under the
    §7.9 refresh; ponder-hit rate
6o. Time management (§8.6): integral allocator vs flat-evals vs flat-wall-clock; the
    stop/extend signal set on/off; verify the extension-spiral guard at high N
6p. Transformer input features — layer-0 aggregates the net would otherwise compute by
    summing over attention: per-square attacked-by / defended-by / pinned / in-check
    counts in the token embedding; per-board register *initialisation* from board-level
    aggregates (check status, material under attack, king-zone attackers); global
    registers initialised with material and timeline counts. Low priority; redundancy
    risk with heads 1 and 4 (§6), which inject the same relations structurally — but at
    zero cost, worth the arm. Initialisation does not conscript the registers (§12.4):
    they may overwrite it
6q. ~~NNUE same-board threat features~~ — void while the NNUE is held open (§11.4)
6r. Variant models (§12.6): Princess in the conditioned main net vs warm-started
    specialist — the generalist tax, measured; Reversed Royalty warm-start vs
    from-scratch — the inherited-king-bias control
6s. Adversarial league (§12.7) vs plain self-play at matched compute: attack cost against
    held-out fresh adversaries AND standard match Elo — the robustness-strength tradeoff;
    windowed vs unwindowed exploit-discovery rate at matched adversary budgets; attack
    cost against attacker capacity per size class — the robustness curve — with every
    adversary scored for specificity
6t. Final selection (§8.6): max-visits vs LCB with visit floor vs LCB + categorical
    variance, at match play — each LCB arm with and without §8.4's per-child variance
    bonus, since resolve-then-decide is one design
6u. Uncertainty-prioritised sampling (§12.3) vs uniform at matched data: persistent-bucket
    trigger vs realised error vs raw variance — the last expected to overfit aleatoric
    positions
6v. Difficulty-weighted allocation (§8.6): effort-quantile-weighted integral vs equal-sims;
    quantile choice; judged on late-game move quality where the gain should concentrate.
    Difficulty-prioritised curriculum overlaps 6u with the same aleatoric caveat — test as
    a 6u arm, not separately
6w. 8-bit inference (§10.2): FP8 applied after training vs BF16, at equal wall-clock and
    at equal search budget, per board range; a QAT fine-tuning arm only if the
    equal-budget arm loses
6x. Turn construction (§8.2, D2): the evaluate-every-node ply split vs the tree form
    (turn-start priors; re-evaluation only at interactions and complete turns) vs the flat
    form (best-first legal turns by summed logits, progressive widening); measured jointly
    with 6g and 6k and reported by mandatory-board count — after the turn-width
    measurement (§15.1)
6y. Normalisation placement (§9.3): pre-norm vs Lc0's post-norm with DeepNet
    initialisation, both RMSNorm-style without biases
7. Tied vs separate weight sets — only under the §3.3 contingency
8. **Dense vs 14+2 hybrid (§7.6)** — last, and only if the dense engine is node-starved.
   Requires the dense baseline from steps 1–6 to price it against.

---

## 14. Rejected approaches

### 14.1 Architectural

**Fixed-size input tensors.** N is unbounded. Non-starter.

**Mean-pooled board vectors for the inter-board pass.** Squeezing 64 squares into one
vector destroys square identity, which is precisely what cross-board moves depend on. Use
registers instead.

**AlphaZero-style fixed policy planes.** Output space is variable. Use the bilinear pointer
head.

**Absolute positional encoding.** Goes out of distribution as games run long and timelines
proliferate.

**Alpha-beta as the main search (transformer as its eval).** The pairing law: alpha-beta is
the paradigm for *many cheap* evaluations — depth goes as b^d — while prior-guided MCTS is
the paradigm for *few expensive* ones, because the policy prunes branching before expansion
rather than after it. At ~2,370 evals/sec and per-ply branching of 200+, a second of
alpha-beta buys full-width depth ~2. Eval cost sets the hybrid's balance, and it is already
set: alpha-beta's place was the scanner (§11.1, §11.3), where its eval costs
sub-microseconds — and even there it failed to reach depth above 50 boards (§11.4).

**A separate small policy network to pre-filter moves.** The policy function already
exists — the §8.1 head shares the trunk and costs seven orders of magnitude less than it,
so there is nothing to save. A *distilled* filter would gate the big model's view through
a worse copy of itself, and pruning is strictly more dangerous than ordering — a badly
ordered move is slow, a pruned move is invisible. Narrowing is done by the model's own
head under §8.4, audited where the verification layer reaches (§11.4).

### 14.2 From the attention-optimisation literature

**KV eviction (H2O, SnapKV, PyramidKV) — actively dangerous.** In an LLM, dropping a
low-attention token costs some recall. Here, a "low attention" superseded board may be the
exact time-travel destination that decides the position. That is not a quality regression,
it is a blunder — and it surfaces as Elo loss after thousands of games, not as a perplexity
bump.

StreamingLLM's *structure* (sinks + window) is fine; the principled version is §10.4 —
window by game rules, which gives a hard guarantee instead of a heuristic. (The ban is
directional — §10.4.3: with E as an exact floor, heuristic *addition* of scope above it
is safe; subtraction below it remains refused, including pruning rules-active timelines
for looking strategically dead.)

**Trained sparse attention (NSA, MoBA, DeepSeek Sparse Attention).** These train a router to
learn which blocks matter. The move generator already provides that oracle exactly. Spending
capacity rediscovering legality is strictly worse than using what you have.

**Linear attention and SSMs (Mamba, RWKV, RetNet, Performer, Linformer).** Two independent
disqualifiers: they impose a sequential ordering that a 4D lattice does not have, and their
consistent failure mode is *recall* — which is the entire task. "Is there a rook on that
file, across four dimensions" is not something an approximate-recall operator should be
asked.

**Speculative decoding, multi-token prediction, chunked prefill, continuous batching.** All
assume autoregressive token generation. There is none here.

**Ring Attention / DeepSpeed-Ulysses.** Sequence parallelism for 100k+ context. Not needed
at 15k.

### 14.3 Held open

**Mixture-of-Depths.** Routes only a subset of tokens through each layer. Empty squares and
frozen boards are exactly the tokens that don't need 16 layers. No precedent in board-game
engines — a research bet, not a shipping decision, but the domain fit is unusually good.

**MLA instead of GQA.** Better compression (~93% vs MHA) but a genuine pre-training
commitment, unlike GQA. Triply conditional: only relevant if you adopt the cached
configuration (§7.1), *and* run sim counts high enough for tree memory to bind (§10.2's
10k-sim row), *and* find GQA insufficient. Do not pre-commit to it.

**The cached configuration itself (§7).** Not rejected — deferred, with a specific
condition for revisiting it: after the window is measured, and weighed against §12.1's
self-play compute argument.

### 14.4 From the 2026 engine survey and subsequent review

A systematic comparison against Lc0, Stockfish, KataGo, dlshogi and YaneuraOu/Suisho —
independently re-analysed — produced the adoptions recorded in their home sections
(node-capped scanning, secure-child selection, the categorical value head, variance-cPUCT,
MCGS, time management, PCR, continuation-history ordering, quiet-only NNUE training).
This registry holds what stays open or refused, and the boundary clarification the
analysis forced.

**The boundary, stated precisely.** The differently-wrong rule is a boundary *between
systems*, and the exact-sources rule governs *loss targets*. Consequences: within-system
self-correction at inference is legitimate (Stockfish corrects its own static eval from
its own search; KataGo biases its own values from its own subtrees); the violations are
cross-system coupling at inference — one evaluator deferring to the other — and
learned-verifier estimates entering any loss.

**Held open (prototype tier):**

- **Subtree value bias** — reclassified from an earlier harmful verdict that misapplied
  the training rule to an inference-time search statistic. Within-system, never trains
  anything, +30–60 Elo in KataGo. Caveat: KataGo disables it under external-prior biasing
  modes as interfering — check against our prior-boost channels before enabling alongside
  them. The 5D bucketing key: **N and phase** — a persistent correction in an N-bucket is
  not a positional blind spot but measurable train/deploy drift. The same statistic
  serves three roles under three rules, all admissible: inference-time correction (this
  entry), offline drift diagnostic (unrestricted), and curriculum seeding from
  high-residual buckets (position selection — §12.2's safe channel). The same buckets
  carry **calibration monitoring**: reliability curves of the categorical head's
  variance, by N and phase — every confidence consumer (§8.4, §8.6, §11.2's escalation)
  assumes the variance means what it says, and §12.3's failure mode migrates exactly
  where calibration degrades. Persistent miscalibration in a bucket is the same drift
  signal as a persistent correction. Fallback if derived variance calibrates poorly: a
  trained error head predicting |value − later-search value| (KataGo's solution;
  legitimate under §12.4's own-search rule) — the 6m ablation arm.
- **Within-scanner correction history** — Stockfish's own pattern (own-search residuals
  correcting own static eval) transplanted whole into the scanner; held open with the
  NNUE (§11.4). Within-system, legitimate.
- **Board-pair smolgen** (§6) — full-rank biases over 240² board pairs generated from the
  global registers, negligible cost; the only level at which Lc0's construction survives
  our scale. Within-board generation costs 8× its block attention and, made low-rank,
  reduces to the same-board head attending to its register — dropped. Targets the
  frozen-board relevance gap alongside §7.9.
- **Local/global layer interleave** (§7.6) — within-board-only layers with a causal
  cross-board layer every third; cache-compatible, ~2.4× cheaper per node, K/V stored only
  at global layers (649 → ~203 MB at 600 boards). Layer-level fallback to head-level
  routing; committed-class; a 6a arm.
- **Complexity-adaptive number of dense top layers** (§7.6) — extra bidirectional layers on
  top of the cached stack for complex positions. On top only (C16's placement rule); each
  costs ~1.8× a whole cached node evaluation at 240 boards; committed-class, since training
  would have to randomise the number of dense top layers; and it builds on the deferred
  hybrid. Held unless 6c shows fresh conditioning is worth a lot.
- **The scanner NNUE** (§11.1) — moved from the build plan to held open at CHECKPOINT 4. Its
  main job, evaluating the scanner's leaves, went with the leaf scans, and quiet-leaf reach
  failed every range. Revisit only if a leaf-level role returns.
- **The opponent-reply auxiliary head** (§12.4) — dropped from v1 at CHECKPOINT 6. Lc0's own
  training config weights such "future heads" zero as unstable, and here the reply's move
  space is undefined — replies on boards our move creates have no token yet. Revisit with a
  defined move space.
- **Hunt with a lazily expanded root** (§11.4) — the likely remedy for its failure above 100
  boards; adopted only through a fresh pre-registered test.
- **Value-side Shaw vectors** (§4.1) — per-pair aggregation of learned vectors into the
  attention output; FlexAttention cannot express it, so it needs a custom kernel. Revisit
  only if B17's key-side term shows a signal.
- **Late move reductions** — moot while no alpha-beta search exists. MCTS's priors and
  widening are its analogue, giving late, low-prior moves few or no visits; df-pn (Hunt,
  the settling scans) is best-first by proof numbers, with no depth to reduce, and its
  ordering stays net-free (H4). If an estimate search returns with the NNUE, LMR applies to
  estimate lines only — never to proof lines or evasions (H5).
- **Dedicated df-pn long-mate thread** (dlshogi's root solver) — v2. Its repetition gate
  is gone (§11.0 item 5); it still needs a 5D dominance definition for the df-pn+
  accelerations.
- **Same-board NNUE threat features** (§11.1) — held open with the NNUE (6q void meanwhile); the head-only cross-board overlay is v3-speculative at most.
- **Early exit / adaptive depth** (§10.3) — confidence-family inference saving; allowable
  because the verification layer never scales down with self-rated ease; needs exit heads
  and calibration monitoring.

**Refused, with the violated principle:**

- **Token-level smolgen** (square or piece tokens) — a ~256-dim generator's T × T output
  costs 8× the attention it modulates at any T in our range, and a rank-r generated bias
  is identical to appending r context-modulated Q/K dims at +r/d_head (§6): the d_head
  lever in a costume. Board-pair is the salvage.
- **A small transformer as the scanner's leaf eval** — the pairing law applied to the
  scanner's own search: alpha-beta needs millions of *sequential* CPU evals, which cannot
  batch, so eval cost must be NNUE-class. A toy transformer (d=32, two layers) is ~330×
  an NNUE eval per board, a merely small one (d=64, four layers) ~2,400×, and none can
  do the O(1) accumulator update. Worse on independence too: same inductive-bias family
  as the main net — similarly wrong, less capably. The legitimate route to cross-board
  awareness in the scanner eval is attention over head-board *accumulators* (v3, §11.1).
- **Cross-system inference coupling** (dual-net fallback: the value head deferring to the
  scanner NNUE on "easy" positions) — breaks differently-wrong at inference and creates
  the shared blind spot the architecture exists to avoid.
- **Learned-verifier estimates as loss targets** — §12.4's actual rule.
- **Distillation in either direction** — standing refusal, reconfirmed by dlshogi keeping
  its solver undistilled.
- **A proofs-fed correction table** — incoherent, not merely marginal: proofs are
  game-theoretic values, not centipawn residuals; there is nothing to correct toward.
- **External tablebases and stored opening books** — unenumerable under unbounded N; the
  permanent solved store is the correct analogue.
- **Cross-board NNUE threat features** — destroy accumulator separability and the frozen
  library, and smuggle cross-board interaction into the component whose blindness there
  is load-bearing (§11.1).
- **Lossy duplicate-abstraction as input** (merging "similar" boards or standoff piece
  pairs into pre-digested units) — breaks uniform tokenisation and the pointer policy
  head, is the bitter-lesson trap, and 5D punishes it specifically: travel-created
  near-duplicates differ by tempo and parity, which is what decides games. The exact
  version of the instinct is §8.5's MCGS, which merges genuinely identical states for a
  zero-cost eval.
- **Content-keyed projection cache** (sharing pre-RoPE K/V projections across
  identical-content boards) — killed by the companion's §9.3 discipline: projections are ~3% of dense
  cost, so the ceiling is ~1.5%, and cache memory was not binding either. Recorded so it
  does not get re-proposed looking clever.
- **Latent-KV compression (MLA-style)** — a low-rank key bottleneck; the GQA refusal's
  reasoning applies unchanged (§10.3). Revisit only if cache memory binds.
- **A raw depth or node-count factor in final selection** — node count under a child
  *is* its visit count (double-counted), and depth-as-quality favours narrow forced
  continuations over flexible moves with many good replies. The exact version of what
  depth reaches for is the §11.1 root-rule scan, which extends the chosen line along
  forcing moves beyond tree depth; the statistical version is §8.6's LCB rule.
- **Threat relations as separate tokens** — edges outnumber squares by hundreds to
  thousands per position; threat tokens explode T². Relations go in attention (§6's typed
  edges), aggregates in embeddings (6p); the per-board registers already exist for
  board-level summaries.
- **Removing duplicate threats** — under immutability and caching, identical threats on
  frozen boards cost nothing (features and K/V computed once, never again); attention
  dilution is scope's job (§10.4), not feature dedup's; and a mutual attack is two
  directed relations, not a duplicate.
- **Piece-centric tokenisation — REVERSED: now a prototype-tier candidate, committed
  class.** Previously declined on three grounds, each re-examined. (1) Destination
  tokens for the pointer head: the earlier "~1.6×" was computed on a single board — in
  the multiverse only *head* boards carry the full legal-destination set and E-boards
  only their landing squares, so the overhead is ~250 tokens across a 240-board state,
  and the real saving is **2.3× tokens midgame, 2.7× in sparse endgames** — ~4.4–6.5× on
  dense attention, ~2.7× on the cached per-node term, the same on KV memory. (2) "Empty
  squares carry geometry" — dissolved by §6's typed edges, which deliver clear-ray,
  x-ray and pin facts exactly; the net never needed to infer them from empty tokens.
  (3) Uniform board shapes for batching — dissolved by §10.3's varlen kernels. What
  survives: absence-concepts (an open file) must be learned from piece tokens; no
  engine-strength precedent for piece-only transformer tokens exists, so the
  geometry-via-edges argument is reasoned, not measured; head boards' destination tokens
  are transient per node. Composes with the window (boards) and the cache; tested jointly
  with the window in ablation 1's cascade, and decided before serious self-play.
- **The model's own certainty as an input to the model** — circular: the position is
  evaluated once, statelessly, so the only uncertainty available at input time is its
  own prior output or a parent's, and self-conditioning on confidence invites feedback
  collapse (a confident prior begets a confident posterior). The standing principle:
  confidence signals flow *out* of the net into the search, commit, time-management and
  verification layers (§8.4, §8.6, §11.4), never back into the net, and never scale those
  layers down (§12.3).
- **Bidirectional layers anywhere but the top of a cached stack** (§7.6) — a bidirectional
  layer at depth k invalidates every cached layer above it; interleaved every third layer
  the stack runs at 1.6× over dense instead of the cache's 29×. Present-to-frozen flow
  enters at the top (the hybrid) or amortised (the root refresh and the refresh ladder),
  never mid-stack.
- **Snapshotting a head's last live K/V at supersession** (§7.3) — the frozen board then
  carries whatever else was live on that path: evaluations become path-dependent, breaking
  §8.5's zero-cost transposition evaluations and feeding the net representations its
  state-defined training mask never produced. Recompute once under the state-defined mask.
- **Skipping the live recompute for heads with no generator edge to the changed board**
  (§7.3) — roughly 2× cheaper per node, but dense heads couple every head to every other,
  so a skipped head's representation depends on when it was last recomputed: path
  dependence by another route. Both shortcuts fail silently — transposition hits return
  subtly different evaluations for the same position.
- **The official game client as a rules oracle** (§11.0) — believed to incorrectly claim
  some positions are checkmate; agreement with it proves nothing. The oracles are
  independent open implementations, cross-checked against each other.
- **Exact settling scans above 50 boards** (brute force or df-pn) — failed 6b and 6b′ even
  at ten times the budget; the forcing trees are too large (§15.6). Guard and Hunt ask
  narrower, one-sided questions instead (§11.4).
- **ONNX/TensorRT export of the main network** (§10.2) — for now. TensorRT is NVIDIA-only;
  ONNX cannot express score-function attention biases except by materialising them or
  through a custom plugin; and nothing needs exporting while PyTorch drives the C++ search
  through the C API. Revisit at 6w only if FP8 leaves a gap worth a plugin.

**Registry note on process.** One refusal in this design's history was made against
evidence already in this section: NNUE threat features were refused outright while the
survey's own Stockfish row recorded the Full_Threats lineage. The rule was sound; the
verdict was wrong because it was applied from memory. Before refusing a technique, grep
this registry. A second failure of the same family: the piece-centric decline above was
computed at the wrong *scale* — a single-board ratio where the multiverse was the unit —
and two of its objections had been dissolved by later decisions nobody re-checked it
against. Refusals carry the assumptions of the moment they were made; when the design
moves, re-derive them at the scale they would actually run at.

---

## 15. Implementation plan — the untrained build

**Build everything; train nothing — first.** An untrained network costs exactly what a
trained one does, and random weights are the ideal fixture for correctness testing. So
the untrained build verifies two of the design's three pillars — **correctness** and
**cost** — before any training compute is spent. Only the third, **quality** (whether an
idea improves play), needs training: that is the ablation programme (§13.1).

### 15.1 Build order

| Phase | Build | Settles |
|---|---|---|
| 0 | Rules engine (§11.0): bitboard multiverse with immutable boards; physical and cross-board move generation; royal-capture check detection; turn legality at submission; naive mate detection; 5dpgn/5dfen parser | — |
| 0 | Differential-testing harness over pre-included, pinned oracles in a separate test harness (§11.0) — ftxi's engine on every commit, cwmtt and Chess in 5D nightly; pinned corpus; known-divergences file | H23, the proof-channel gate |
| 0 | Present-line rules, pinned against the rulebook while writing the generator; no repetition rule (H39) | Closed at CHECKPOINT 1 |
| 1 | \|E\| phase curve over game records | Window cascade step 1 (§10.4.2) |
| 1 | Scanner with the material-count eval: horizon per node budget, stand-pat/zugzwang | 6b — failed above 50 boards; verification v3 (§11.4) |
| 1 | Movegen throughput at 100–600 boards | §11's bottleneck projection |
| 1 | Turn width: mandatory boards per turn and optional moves played, per board-count range, on the human games | How much the D2 arms matter (§8.2) |
| 2 | Network stack and search with random weights: dense v1 with CHECKPOINT 6's network items (RMSNorm-style normalisation without biases, the 4D displacement bias, the turn-state inputs, score-time status biases for frozen boards; the key-side Shaw term behind a flag), then committed-class alternatives behind flags (cached, piece-centric; the window only if the pilot run revives it); PUCT, widening, MCGS behind a flag, the D2 turn-construction arms, persistence, time management, LCB; verification v3's components and the audit lane's tooling; the training loop, smoke-tested on synthetic data | — |
| 3 | Equivalence suite green (§15.2); every cost figure measured and written back (§15.3) | Training may begin |

Three rules govern the order:

- **Generator and gate first.** Everything sits on them; building the network stack on an
  ungated generator means debugging two layers at once.
- **Cheap measurements before the code they gate.** The \|E\| curve before the window
  code, 6b before the NNUE and the scanner's pruning — a fat \|E\| or a useless horizon
  makes that code waste.
- **Committed-class alternatives behind flags.** The ablations need both arms anyway, so
  the flag is the deliverable.
- **Push every working branch** (CHECKPOINT 6, after a lost machine): each commit goes to the
  remote, and a pushed branch takes main by merge, never rebase, so it never needs a
  force-push.

**Within Phase 0, the order is:** commit and push, CI green on Linux → fingerprint
matching → check-cluster triage against the rulebook → the engine's check detection,
turn legality and naive mate detection, then the hypercuboid detector, each against the
gate. While triage runs, the generator work that does not depend on the check ruling
proceeds in parallel: board representation and the immutable multiverse, the in-house
parser, per-board movegen, Zobrist hashing with its duplicated-board tests, and
fingerprint export. The rest of the engine follows the phases above rather than
arriving all at once — Phase 1's cheap measurements decide which Phase 2 code is worth
writing.

Not built in the untrained phase: the adversarial league (training infrastructure,
trigger-gated, §12.7), the variant specialists (§12.6), refresh-ladder rungs 2–3 (gated on
rung 1 showing a trained signal, §7.9), the retired scanner rungs and the NNUE, now held
open (§11.4), and the opponent-reply head (dropped, §12.4). No ONNX/TensorRT export (§10.2).
The window is killed, provisionally (§15.6); the pilot self-play run re-checks it.

**The training loop is written and smoke-tested** (CHECKPOINT 6) — targets with §12.2's
provenance exclusions, losses and auxiliary heads, playout-cap randomisation, the QAT
hook — with a few optimiser steps on synthetic data and no training. §15.5 does not
require it; the pilot self-play run does, and a plumbing bug found there costs GPU time.

### 15.2 The equivalence suite — correctness without training

Two computation paths that must agree, compared directly:

| Test | Proves | Register |
|---|---|---|
| Cached attention = dense attention under the same mask | The cache is exact, not approximately right | C2, C17 |
| Windowed (W ∪ E, online-softmax merge) = full attention restricted to W ∪ E — deferred with the window to the pilot run (M18) | §10.4.1's exact-merge claim | C5 |
| Lazy K/V recompute = eager recompute — deferred with the window | §10.4.1's lazy K/V for boards entering E | C10 |
| Supersession recompute = from-scratch under the state-defined mask | No path dependence | C20 |
| A refresh fork leaves sibling subtrees unchanged | Copy-on-write isolation | C3a |
| Same state + epoch → same evaluation; MCGS transposition hits return identical evaluations | Evaluation purity | C19–C21 |
| Incremental Zobrist = from-scratch, especially on duplicated boards | The XOR trap | H18 |
| NNUE incremental accumulator = full recompute (random NNUE weights) — held open with the NNUE | O(1) updates are exact | F4 |
| FP8/BF16 cache precision = live precision | §10.2's matching rule | G1 |
| Displacement bias through the score function = the same bias materialised, on small positions | The score-function implementation is exact | B15 |
| Key-side Shaw term via precomputed query–class products = direct per-pair computation, on small positions | The gather implementation is exact | B17 |
| Tree form with every interaction gate firing = the evaluate-every-node ply split | Inheritance is the tree form's only difference | D2 |
| Flat form's generator = an exhaustive listing of legal turns, in non-increasing summed-logit order, on small positions | Best-first generation is complete and ordered | D2 |
| Turn-state counts = a brute-force recount from the rules engine's obligations | The new inputs are exact | A6 |

Protocol: random weights at realistic initialisation scale — never zeros or near-zeros,
which make distinct paths agree trivially; several seeds; tolerance matched to the
arithmetic (bit-exact for integer paths — Zobrist, NNUE int16 — and numerical tolerance
for BF16/FP8); and adversarial constructions: duplicated boards, deep travel, many
timelines, supersession inside a refreshed subtree, every variant.

### 15.3 The cost figures — estimated → to measure

Every economic argument in this document rests on these, and today they are arithmetic:

| Figure | Current estimate | § | Status |
|---|---|---|---|
| Dense cost per eval at 240 boards | 4.78 TF → ~337 evals/s (FA4-class) | 9, 13.0 | estimated → to measure |
| Windowed throughput | ~2,370 evals/s (the ~7× estimate); ~1,520/s at W-depth k = 8 | 10.4, 10.4.3 | deferred with the window (M18) |
| Cached cost per node | 0.16 TF (29× on paper); 15–20× realistic, ~12–18× after the supersession recompute | 7.1, 7.3 | estimated → to measure |
| Live-set recompute | 3.7–5.6× the changed-boards-only cost | 7.3 | estimated → to measure |
| Supersession recompute | +8–24% per node | 7.3 | estimated → to measure |
| Root refresh overhead (800 sims) | +3.6% (240 boards) / +6.1% (600) full history; +1.3% windowed | 7.9 | estimated → to measure |
| One refresh | ≈ 10 node-evaluations under cached + window; without the window the model gives 29× at 240 boards and 49× at 600 | 7.9 | windowed figure void (window killed); full-history figure to measure |
| KV memory per state | Frozen boards only, plus their trunk outputs (~16 MB at 240 boards); ~260 MB (240 boards) / ~649 MB (600) full history; FP8 halves | 10.3 | estimated → to measure (windowed figure deferred) |
| Local/global interleave | 0.067 TF per node | 7.6 | void while held open (C18) |
| Leaf-batching gain | 1.2–1.5× | 7.8 | estimated → to measure |
| Legality classification per expansion | Median at 240 boards: 26 µs pseudo-legal, 102 µs fast, 2.5 ms budgeted, 7.3 ms exact; 41 of 43 s of random-weight self-play at 336 boards | 11.0, 13 | **measured (CPU)** |
| FFN ratio 2× and 4× | 1.39× and 1.23× evals/s at 44 boards | 9.2 | estimated → to measure |
| Displacement bias and normalisation changes | Negligible: a table lookup per pair; fewer operations | 4.1, 9.3 | estimated → to measure |
| Key-side Shaw term (B17) | ~0.5× the QK^T product at 44 boards, ~9% at 240, with ~1,400 classes | 4.1 | estimated → to measure |
| NNUE eval | ~10k MACs, sub-microsecond | 11.1 | void while held open |
| Movegen throughput | Pseudo-legal: 44,047 actions/s at 240 boards and 24,951 at 600 on one thread. With fast legality: 20,249/s at 240 boards on one thread (1.01× the target), 37,316/s on M13's two expansion cores (1.87×), 12,133/s at 600 | 11, 15.6 | **measured (CPU)** — passes M9 on two cores |
| Prover nodes/s | Brute-force scanner 787k nodes/s on one thread, 566k per thread at five; df-pn 239k; Guard ~18–24k and Hunt 0.7–61k nodes per root at 50 ms. Rung 1 retired, so its latency row is void | 11, 15.6 | **measured (CPU)** |

Rule: each measured figure replaces its estimate *in the section that uses it*, with its
conditions (hardware, precision, batch size, board count, window setting), in the same
edit as this table and the register — the maintenance rule extended to numbers.

**Without a GPU (CHECKPOINT 6).** The untrained build writes back counted FLOPs and
CPU-labelled wall times. The figures that need a GPU — FA4-class evaluations/s, the
leaf-batching gain, FP8, the FlexAttention block-sparse kernel, varlen kernels and the
Phase-3 movegen re-check against a GPU evaluation rate — are named as deferred and
measured on rented NVIDIA time before the pilot self-play run, which needs it anyway. One
bounded attempt on the development machine's AMD GPU may supply AMD-labelled figures
first.

### 15.4 Search and prover mechanics — invariants

Testable with a random or uniform policy: visit counts consistent through backup; virtual
loss always released; solved bounds propagating and never contradicting; LCB and secure
child choosing only among eligible children; re-rooting keeping only reachable nodes while
state-keyed facts survive; the synchronous mode (H15) bit-reproducible; the provers' proofs
(Hunt, the settling scans, the search's solver) on constructed mate positions
cross-checked against the gate's references, and the solver's against df-pn and the
oracles' mate verdicts (H41); the proof-export invariant holding under TT churn (6f). Turn
structure: a submit edge leads to a turn start and a move edge to a mid-turn state; no game
ends during a turn; no dead end where a legal completion exists. The provers run fully
untrained on the material estimate.

### 15.5 Done means

An engine that plays legal 5D chess end to end with random weights; every equivalence test
green; every cost figure measured and written back over its estimate — CPU-labelled
where it needs a GPU the build does not have, and named as deferred (§15.3); and each
committed-class row in the register either decided or explicitly awaiting a named trained
ablation. Training then starts on a codebase whose correctness and economics are already
verified — so when something goes wrong in training, it is the learning, not the plumbing.

### 15.6 Phase 1 decision thresholds — pre-registered

Fixed before the numbers exist, so STOP 2 is a lookup rather than a debate: thresholds
set after the results tend to move toward the results. Each is labelled by how it was
set — **derived** from the cost model and the design's own priors, or **anchored** on
precedent with a judgement call. **Measure movegen throughput first** — it sets the
scanner's node budgets for 6b.

#### The window (\|E\| phase curve) — derived

The window's value is throughput; its cost is the blindness cascade step 3 prices in Elo
(§10.4.2). Break-even from the design's own priors: ~28–53 Elo per doubling of nodes
(ablation 1's pre-registered +80–150 for 7×) against a blindness cost of 10–40 Elo
(midpoint 25) gives ~0.62 doublings — a **1.5× all-in multiplier**, computed with 90% of
compute at the cached per-node cost (self-play generation) and 10% at the dense cost
(training passes). That is the cached end-state and so the conservative bar: if the
cache is rejected, the dense-path multiplier applies and the bar is easier to clear.

| Boards (N) | Active timelines | W at k = 8 | Max \|E\| at k = 8 | Max \|E\| at k = 4 |
|---|---|---|---|---|
| 100 | 6 | 59 | none — W alone exceeds break-even | 23 |
| 160 | 7 | 69 | 31 | 59 |
| 240 | 8 | 79 | 77 | 109 |
| 400 | 10 | 99 | 169 | 209 |
| 600 | 12 | 119 | 290 | 338 |

The table assumes the timeline counts shown. The rule itself applies the 1.5× bar to the
*measured* scope — measured W plus measured median |E| — so the ceilings move with the real
timeline counts; the table is the expectation, not the test.

**Rule:** per phase, the window is on at k = 8 where the measured median |E| is under the
k = 8 ceiling, on at k = 4 where only the k = 4 ceiling is met, and off where neither is.
The scope rule stays exact and state-defined either way (§10.4.2), so a phase-dependent
window keeps training and deployment consistent. **The window is killed outright if the
phases where it is on carry under half the compute** — weight each phase by per-move cost
× moves spent in it — the late game was expected to dominate, and in the human games it
does not (CHECKPOINT 5). The 95th-percentile |E| sets memory
provisioning, not the decision. **Three readings, fixed at CHECKPOINT 4.** The all-in
multiplier is the time-share (harmonic) combination M = 1 / (0.9 / m_cached + 0.1 /
m_dense), with 0.9 and 0.1 the shares of compute before windowing — not a weighted
geometric mean, which overstates whenever the two components differ. The kill clause is
weighted by real human games (MageOfficial/t0database) until self-play exists — a
conservative proxy for the window, since those games top out around 186 boards and the
window pays most above that — and re-checked against early self-play before serious
self-play compute. Under 50 boards the per-phase rule decides as in any phase, with one
tie-break: a result inside the ±10% band resolves to *off*, the simpler configuration,
without a stop. (An earlier version of this reading claimed that phase carries a small
share of compute. In the human games it carries about 80% — CHECKPOINT 5.)

**Result (CHECKPOINT 5): the window is killed, provisionally.** With the time-share
formula it pays at k = 8 from 50 to 186 boards (1.83×, 2.10× and 3.81× across the
measured phases), but in the human games those phases carry only 18–38% of compute (20%
blended): the games spend 88% of their moves under 50 boards and end at a median of 44.
The kill clause is working as intended — the window pays where little compute goes. It
stays unbuilt. Before serious self-play, a **pilot self-play run** measures the engine's
own game-length distribution; if the phases where the window pays carry at least half of
that compute, the window returns to the cascade (§10.4.2). Changing the weighting now,
because it killed the window, would be moving the line toward the result. A phase is a board-count bucket judged at its
measured median scope: the human games' four populated buckets have median N of 24, 60, 112
and 170 boards, with 1, 2, 4 and 2 active timelines — far fewer than the table's assumed
6–12, which is why measured W is so much smaller than the table's. Expectation on record,
made at the table's assumed timeline counts: off in the early game (at ~100 boards it cannot
pay at k = 8 even with |E| = 0 — true only at six timelines; at the measured one to four it
pays from 50 boards), kept on its late-game value — at
|E| ≈ 25 the all-in multiplier is ~1.6× at 160 boards, ~2.1× at 240, ~3.0× at 400 and
~4.0× at 600.

#### The scanner (6b) — anchored

Depth is counted in forcing half-turns (one side's complete turn, forcing moves only),
with board-move depth reported alongside for §8.2. The rung-1 leaf budget is what the
scanner threads can search while one leaf waits for its batched evaluation — threads ×
measured nodes/s ÷ evaluation rate; the root final scan gets 50 ms.

| Test | Pass | On fail |
|---|---|---|
| Rung-1 leaf scan at its derived budget | ≥ 3 forcing half-turns fully resolved in ≥ 50% of leaves | Drop rung 1; the higher rungs keep their larger budgets |
| Root final scan at 50 ms | ≥ 5 forcing half-turns (mate-in-3 depth — dlshogi's leaf-solver level) in ≥ 50% of roots | **STOP**: the verification layer needs a different shape (§11.1's kill condition) |
| Quiet-leaf reach | ≥ 50% of scan leaves reach a quiet position within budget | NNUE v2 and quiet-only training deferred — too few quiet leaves to evaluate |
| Stand-pat validity under 5D move obligations | Invalid in ≤ 5% of quiet positions | 5–20%: stand-pat disabled where obligations bind; > 20%: an obligation-aware bound replaces stand-pat — a redesign item |

The depth lines are anchored on dlshogi's leaf solver and the chess analogue; the 50%
lines and the stand-pat bands are judgement, fixed now so they cannot drift. As measured:
stand-pat counted invalid at a quiet position when every legal action — or a sample of 32
when there were more — lost its static evaluation to the opponent's two-half-turn forcing
reply; without a GPU the rung-1 budget used the design evaluation rates (9,600 and 4,600
evaluations/s); the other terms are operationalised in docs/MEASUREMENTS.md.

**How these are read (ruled at CHECKPOINT 2, after the first failing result).** The
tests are read *per board-count range*, as the window rule is — never pooled, because a
pooled pass driven by small early-game positions would hide a failure where the engine
actually operates. That is the stricter reading, which is why it may be fixed after the
fact without moving the line toward the result. The scanner is measured with its design
thread budget (§15.6's CPU allocation, Lazy SMP — §11.3), not a single thread. The first
measurement failed above 50 boards with 85% of the scanner's work in exhaustive
enumeration (all legal actions when in check, mate tests at the horizon), so before any
redesign: **one bounded optimisation pass on those two paths, then re-run 6b unchanged.**
Outcome rules, fixed now:

- Every range from 50 boards up passes → the verification layer stands as designed.
- 50–199 boards pass, 200+ fail → the verification layer becomes phase-dependent, like the
  window: brute-force scanning where it reaches depth, and above that the proof channel's
  leading candidate is **df-pn** (§14.4) — selective proof search, whose repetition gate is
  gone (H39); plain df-pn needs no dominance definition.
- 50–199 boards still fail → STOP: redesign the verification layer.

Rung 1 and quiet-leaf reach are re-measured in the same run and their own rules applied
to the new figures (drop rung 1, defer the NNUE). Until then both stay parked.

**CHECKPOINT 3 — result and the next test (pre-registered).** The re-run failed every range
from 50 boards up (31.4% / 24.7% / 2.4%, against 76.7% under 50 boards); rung 1 and
quiet-leaf reach failed everywhere, so rung 1 is dropped and the NNUE deferred. The
redesign (§11.1) is decided by **6b′**, on the same positions, read per range, at 5 threads:

| Test | Pass |
|---|---|
| Proof channel (df-pn, 50 ms root) | ≥ 50% of roots *decided* — mate within 5 half-turns proven or disproven |
| Estimate channel (leaf) | Fits the rung-1 leaf budget at the 90th percentile — if so, rung 1 returns, estimate channel only |

Outcomes, fixed now:

- Every range from 50 boards up passes → adopt. Below 50 boards, keep whichever of
  brute force and df-pn decides more roots.
- 50–199 pass, 200+ fails → phase-dependent: above 200 boards, proofs only at the root
  final scan and PV transitions; the coverage gap is recorded in §13 and the adversarial
  league gains scale-targeted attackers.
- 50–199 fail → STOP: exact verification covers only small positions, and the design's
  robustness argument above them rests on the adversarial league alone — a design review,
  not a tuning pass.

**CHECKPOINT 4 — 6b′ failed from 50 boards up.** df-pn decided 77.4% under 50 boards but
40.7%, 23.7% and 0.0% at 50–99, 100–199 and 200+ (brute force: 75.9 / 30.7 / 24.7 /
2.4%); ten times the budget reached only 45%, 34% and 2% above 50. Even the passes are
thin: 256 of the 356 disproofs under 50 boards are positions with no checking move, and
the literal question — every first move with every completion — is decided in 9 / 3 / 1 /
0% of roots. The estimate channel fits the rung-1 budget nowhere (2.3× to 51× over), so
rung 1 stays dropped; df-pn is kept under 50 boards (408 roots decided against 400). Per
M15: STOP for a design review — concluded: the verification layer is v3 (§11.4). Readings
used: "fits" meant the 90th-percentile node cost against five threads' measured nodes per
second divided by the size-matched evaluation rate; "decided" counted class-relative
disproofs, including roots with no checking move; and M15's "keep whichever decides more"
had no tie rule, which the result never needed.

**6b″ — Guard and Hunt (pre-registered at CHECKPOINT 4).** Same positions, read per
board-count range, 5 threads, 50 ms; both thresholds are judgement, fixed now:

| Test | Pass |
|---|---|
| Guard | Depth 1 completes in ≥ 95% of roots in every range — it is insurance, so it must run almost always; depth reached is reported |
| Hunt | Finds ≥ 50% of the forced mates within 5 half-turns that a long offline search establishes, per range |

Hunt's ground truth comes from one long offline search per position, recorded once.
Each part is adopted independently, in the ranges where it passes. Where Guard fails, the
chosen move goes unguarded in that range; where Hunt fails, proofs there come only from
the search's solver — both recorded in §13. No outcome stops the build: the layer's
floor (exact rules, the solver, the audit lane, the league) stands either way.

**Result: Hunt adopted under 100 boards, Guard in no range** (figures in §11.4). Readings fixed
before the run: the chosen action is the record's next action (no search existed yet);
Hunt's ground truth is the union of two attacker classes' proofs at 4M nodes each, pinned
with its provenance, and the tools refuse other records or rules; only proofs the rules
engine confirmed count, and a refuted proof withholds every verdict; a range with no
established mate has no verdict, and one resting on fewer than ten is flagged as too few to
read; the shipped configuration is the measured one.

#### Movegen throughput — derived targets

Search expands one node per network evaluation, so expansion — legal moves, attention
masks, typed edges and E — must outrun evaluation with headroom. At the realistic cached
rate (half the paper figure, §7.1), the most demanding adoptable configuration evaluates
~9,600 nodes/s per GPU at 240 boards (cached + window) and ~4,600/s at 600. Targets at
2× headroom:

| Check | Target | On fail |
|---|---|---|
| Expansion throughput per GPU's CPU allocation | ≥ 20,000/s at 240 boards; ≥ 10,000/s at 600 | Search is CPU-bound: optimise movegen before Phase 2 |
| Scaling at fixed \|E\| | Flat in N — ≤ 10% growth from 100 to 600 boards | Frozen boards' move and attack sets are not being cached: fix before Phase 2 |
| Phase 3 re-check | ≥ 2× the measured evaluation rate of the adopted configuration | As above |

Whatever CPU remains after expansion is the scanner's budget — which is why this
measurement runs first.

**CPU allocation (pinned at CHECKPOINT 2):** 8 cores per GPU as the reference — replace it
with the self-play machine's real count once known — split 2 for expansion, 1 for search
and batching, and 5 for the scanner. M9 is judged on the 2 expansion cores (it passes:
1.87× at 240 boards). With the scanner retired, the five scanner cores serve Hunt,
the settling scans and the audit lane.

**The Phase-3 re-check needs a GPU evaluation rate** (§15.3). Until one exists it is
deferred; the only rate today is the CPU's, 6.9 evaluations/s for F1.

---

## Summary — the shortest version

**Build the rules engine and its gate first (§11.0); then measure what the game is actually
like.** Both came before network dimensions and kernel work, and the measurements reshaped
the design: real games are small (median 44 boards), the window is killed provisionally,
and exact verification reaches depth only in small positions (§13, §15.6).

Then: single weight set, **dense bidirectional attention** over all square tokens, 4D RoPE
plus a learned 4D displacement bias (§4.1), RMSNorm-style normalisation without biases
(§9.3), 2 registers per board plus 16 global, move-graph mask on 2 heads, full MHA,
bilinear pointer policy head, ~21M parameters at d=256 × 16 layers with 8× FFN (2× under
test, §9.2).

Run it in PyTorch with FlexAttention — BF16 trunk, FP32 heads; later, FP8 for the trunk's
matrix multiplications, with QAT prepared behind a flag, and no ONNX/TensorRT export
(§10.2). No
paged allocation, no radix tree, no cascade
inference — none of that stack is needed until you commit to caching, and you should not
commit to caching until you have a dense baseline to price it against.

Revisit §7 once, after the pilot self-play run measures the engine's own game sizes (M18).
If its games are as small as human ones, dense is the destination; cached with
root-conditioned refresh (§7.9) earns its place only if large positions carry the compute.
(This summary previously named the cache as the expected destination; that was priced at
the 240-board premise the data overturned.)

**Build all of it untrained first (§15).** Correctness by equivalence, cost by measurement —
only quality needs training.

**Verification is exact where it can be, and narrow where it can't (§11.4).** Exact rules
everywhere; exact settling under 50 boards; Hunt looks for our own forced mates under 100;
Guard was adopted nowhere; above that the search's solver proves what it reaches, and the
adversarial league and the offline audit lane carry robustness.

**Multi-board turns are evaluated at turn boundaries (§8.2).** One evaluation scores every
board's moves; the combination is assembled from those priors, in the tree or flat form;
and the network sees exact turn-state counts (§2.3).
