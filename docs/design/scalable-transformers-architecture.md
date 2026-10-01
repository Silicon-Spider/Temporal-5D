# Scalable Transformers for Variable-Cardinality Structured State

A general architecture pattern for problems where the input is an unbounded, growing set of
structured substructures arranged on a lattice or graph — and a fixed-size model must
handle any cardinality.

Companion to `5d-chess-transformer-design.md`, which is one worked instantiation.

---

## 0. Scope of this document

**What this covers.** Problems where the state is `N` repeated substructures, each of fixed
internal size, arranged in a structured relationship, with `N` unbounded and varying at
inference time. The architecture must have parameter count independent of `N` and must
generalise to values of `N` unseen in training.

**What it does not cover.** Sequence modelling (ordinary LLMs), fixed-topology inputs
(images at fixed resolution, standard board games), or problems where `N` varies over a
small known range and padding to a maximum is acceptable.

**Honesty about novelty.** Almost every component below is off-the-shelf — axial attention,
registers, pointer networks, relative position bias, graph attention bias, KV caching. The
contribution is not new mechanisms. It is a **selection discipline**: which established
techniques apply to this problem class, which are neutral, and which are actively harmful.
The third category is the useful part, because it is where importing LLM practice
unexamined does damage.

**Nothing here is measured.** All figures are cost-model arithmetic or numbers borrowed
from adjacent literature. Confidence is marked where it matters.

---

## 1. The problem class

### 1.1 Definition

A state `S` consists of:

- **Substructures** `B₁ … B_N`, each with fixed internal size `k` (squares on a board,
  atoms in a residue, agents in a squad, cells in a region).
- **A structural relationship** between substructures — usually positions on a
  `m`-dimensional lattice, sometimes a general graph.
- **Unbounded, varying `N`.** It grows during an episode, differs between episodes, and
  training-time `N` does not cover inference-time `N`.

Total token count `T = N × k` (plus registers, §6).

### 1.2 The three failure modes of naive architectures

| Failure | Symptom | Fix |
|---|---|---|
| **Input** | Fixed input tensor cannot represent the state | Weight sharing via attention (§2) |
| **Output** | Fixed logit vector cannot express the action space | Pointer heads (§7) |
| **Cost** | `O(T²)` with `T` in the tens of thousands | Scope reduction (§3), then topology (§5) |

All three must be solved. Solving only the first is the common failure — a model that
*runs* at any `N` but cannot act at any `N`, or cannot afford to.

### 1.3 Where this pattern applies

- **Game engines with growing state** — variable-board games, card games with unbounded
  permanents in play, RTS with variable unit counts.
- **Multi-agent systems** — variable agent counts where each agent has fixed-dimension
  observations and agents interact pairwise.
- **Scene and object reasoning** — variable object counts in 3D scenes, each object with
  fixed-size feature descriptors, arranged in space.
- **Simulation state** — particle or cell systems where `N` changes over the episode.
- **Structured document collections** — variable document counts, each internally chunked,
  where cross-document relations are positional or typed.

### 1.4 Where it does not apply, and the honest boundary

If `N` is bounded and small, pad to the maximum. It is simpler, faster, and the pattern
below buys nothing.

If substructures have *no* stable internal size — genuinely ragged — the weight-sharing
argument still holds but the cost model in §9 does not, and the block-structured
optimisations in §5.3 and §11 stop applying.

If the relationship between substructures is unstructured (an arbitrary set with no
positions, no graph, no types), you have a Deep Sets / Set Transformer problem, and most
of §4 is wasted effort. **The value here comes from the structure, not the variability.**

---

## 2. The foundational principle: weights are not connectivity

This is the most commonly misunderstood point in the pattern, and it is worth stating
before anything else.

> Attention weights do not encode *who attends to whom*. **The mask does.**
>
> Q/K/V projections are shared across all positions by construction. A token attends to
> another if and only if the mask permits it.

### 2.1 What follows from it

**Parameter count is automatically independent of `N`.** Nothing needs to be replicated
per substructure. That is what weight sharing in attention already means. A design that
speaks of "repeating the weights for each substructure" is describing something the
mechanism does for free.

**Confining attention within a substructure is a masking decision, not a weights
decision.** A model with a single weight set that alternates masks per layer —
within-substructure, then across — is a valid architecture. It is not less expressive by
virtue of having one weight set; it is less expressive only where the mask says so.

**Separate weight sets are a capacity choice.** Giving intra- and inter-substructure passes
their own projections usually helps a little, because the two relations have different
statistics. It is not what makes cross-substructure communication *possible*. Run tied
weights as a baseline before assuming otherwise — it halves parameters.

### 2.2 Terminology discipline

Passes over different token subsets are **self-attention with different masks**, not
cross-attention. Cross-attention means queries from one stream and keys/values from
another. The distinction is not pedantry: it determines which kernel you can use and how
caching applies.

---

## 3. Tokenisation and scope — the primary lever

### 3.1 The dominant decision is what becomes a token at all

Attention cost is `O(T²)`. Every other optimisation in this document is a constant factor
on that term. **Reducing `T` is the only lever that moves the exponent's base**, and it is
routinely worth more than the entire kernel-optimisation stack combined.

Concretely, from the worked example: pruning substructure count by 3× yields ~7× on total
cost — larger than FlashAttention, FP8, and sparsity combined.

### 3.2 Scope reduction beats scope optimisation

Before optimising attention over `T` tokens, ask what fraction of those tokens can affect
the output. In most structured-state problems the answer is "far fewer than all of them":

- Substructures that are causally unreachable
- Substructures outside an interaction radius
- Substructures frozen so long ago that their state is subsumed by later ones

### 3.3 Prune by domain rule, never by learned importance

This distinction is the difference between a sound optimisation and a silent correctness
bug. See §11.3 — it recurs.

**Rule-based pruning** ("drop substructures that are provably unreachable") gives a hard
guarantee. If the rule is a proven superset of reachability, dropping is lossless.

**Learned pruning** ("drop substructures with low attention scores") gives a heuristic.
When the domain has exact structure — legal moves, physical adjacency, causal dependency —
the heuristic will occasionally drop something that mattered, and you will not find out
directly. You will find it as unexplained performance loss much later.

**Verify the rule exhaustively against the domain's own generator.** A pruning rule that is
*almost* a superset of reachability is a correctness bug wearing an optimisation's clothes.

### 3.3.1 Exact scope reduction via online-softmax merge

The correctness hazard above is avoidable outright, using a property of online softmax that
is usually treated as a kernel implementation detail:

> Online softmax merges attention over **disjoint key partitions exactly.** Carry
> `(output, running_max, running_sum)` per partition, rescale, combine.

This is the mechanism FlashAttention is built from, and what makes Flash-Decoding, cascade
inference, and hybrid causal/dense layer stacks work. As an *architectural* primitive it
licenses a two-partition design:

| Partition | Contents | Treatment |
|---|---|---|
| **Near** | Substructures inside a cheap, contiguous, block-structured region | Dense |
| **Far** | Substructures outside it that the domain rule says are still relevant | Sparse gather |

Merge with online softmax. The result is **exact** relative to full attention over the
union — so scope reduction stops being a quality/correctness trade and becomes a pure layout
optimisation.

**It also reframes the measurement.** The question is not "how large should the near region
be" — that has no principled answer. It is **"how large is Far?"**:

- **Far is small** → scope reduction is cheap and exact. Take it.
- **Far is large** → no scope reduction helps, because you cannot drop what is relevant.
  Fall through to the immutability pattern (§8) or the axial contingency (§5.2).

That is a cheaper, better-posed experiment than tuning a cutoff.

**The floor inverts the heuristic ban.** Once Far provides an exact floor, the hazard
that banned learned scope selection (§11.3's eviction rule) is gone in one direction:
heuristics can no longer cause a correctness failure by *adding* scope above the floor —
only by subtracting below it. The rule is therefore directional: **heuristics may add,
never subtract.** The cheap contiguous region becomes a budgeted priority queue filled by
any signal at hand — recency, structural anchors, even learned relevance — while the
floor alone carries correctness.

**Scope blindness compounds through allocation — plan for it.** The model cannot value
what it cannot see; search or selection follows the model's priors; so the process never
explores toward out-of-scope resources, and the blindness is never corrected. Mitigations
layer: the exact floor (anything actionable now is visible), per-node re-partition (one
step toward a resource brings it into the child's scope), guaranteed minimum exploration,
standing annotations on immutable state, pre-loading flagged items into scope, and
training targets that teach out-of-scope state to advertise itself. What survives is
multi-step quiet setups toward unflagged resources — so adversarial evaluation must
target the scope boundary specifically, because internal metrics inherit the same
blindness.

**Cap the filler, never the floor.** Scope decomposes as the exact floor (uncapped —
rules-determined), a mandatory core that scales with the state's live structure, and
heuristic filler under a token budget. The budget exists for batch-shape predictability,
memory provisioning and training-distribution stability — but it binds only the chosen
part of scope: a max on what you choose to include, none on what correctness demands.
Provision for the floor's tail, and treat a routinely fat tail as the signal that scope
reduction is the wrong tool for the domain.

### 3.4 Measure scope before choosing dimensions

Scope determines `T`; `T` determines the cost budget; the cost budget determines width,
depth, and every serving decision downstream. Choosing `d_model` before measuring the
realistic distribution of `T` on real episodes is choosing in the wrong order.

**This is the first experiment, not a later optimisation.**

---

## 4. Positional encoding on an `m`-dimensional lattice

### 4.1 Absolute encoding fails

Substructure indices are usually assigned by creation order, which is semi-arbitrary, and
they exceed the training range as episodes run long. Absolute encoding goes out of
distribution in exactly the regime you built this architecture to handle.

### 4.2 Relative encoding is what buys `N`-generalisation

Weight sharing (§2) gives you *runnability* at unseen `N`. It does **not** give you
*competence*. Relative encoding does, because the interaction rules are translation-
invariant along each axis, and relative encoding is how that symmetry enters the model.

**If this is wrong, nothing else in the architecture matters.** It should be near the top
of the ablation order.

### 4.3 Three options

| Approach | Extrapolates | Kernel-friendly | Notes |
|---|---|---|---|
| **`m`-dimensional RoPE** | Moderately | **Yes** — applied to Q/K pre-kernel | Split head dim `m` ways, rotate each group by its own coordinate |
| **Learned relative bias** | Poorly beyond training range | Needs FlexAttention or similar | Most expressive |
| **ALiBi-style per-axis decay** | **Best** | Yes, often natively supported | Monotonic decay may be wrong for the domain |

**Default to `m`-dimensional RoPE** unless you have a reason not to. It is the only option
that is both relative and free at the kernel level.

### 4.4 The head-dimension trap

Splitting `d_head` across `m` axes means each axis gets `d_head / m` dimensions. At
`d_head = 32` and `m = 4`, that is 8 dimensions per axis carrying all positional
discrimination across tens of thousands of tokens.

**Consequences:**

- Compute and apply the rotation in FP32, then cast to BF16 (§10).
- The even per-axis split is not required: **allocate rotary pairs by coordinate range**
  (short axes get fewer dims, long axes more), and when heads carry structural masks
  (§5.3), **specialise per head** — a head masked to one axis spends its whole rotary
  budget there, and hard-masked heads whose structure comes from the mask can run
  near-NoPE. Most of the thinness concern dissolves before any width change.
- If positional signal is still weak after both, raise `d_head` before raising
  `d_model`.
- **Do not adopt KV-head reduction** (GQA/MQA) — it shrinks the key space on precisely this
  already-marginal axis. See §12.2.

### 4.5 Not all axes are equal

Some lattice axes are genuine metrics (spatial distance, time). Others are index-like and
semi-arbitrary (creation order, ID assignment). The latter are noisier and may warrant
different treatment — a coarser bucketing, a learned-embedding fallback, or exclusion from
the relative scheme.

Identify which of your `m` axes are which before treating them uniformly.

---

## 5. Attention topology

Three options, in increasing order of structural injection.

### 5.1 Unified attention

All `T` tokens in one sequence, structure carried entirely by positional encoding.

- **Expressiveness:** maximal. Any token pair resolves in one layer.
- **Cost:** `4T²dL`.
- **When:** `T` is affordable after scope reduction. This is the default.

### 5.2 Factorised (axial) attention

Alternating masked passes, one per lattice axis.

- **Cost:** `T × (k + N)` per token instead of `T²` — often 20-30× cheaper.
- **Expressiveness cost:** no single layer models an arbitrary token pair. Interactions
  along lattice diagonals need multiple hops.
- **Requires more depth.** Budget 1.5× the layers of the unified version, since each layer
  covers one axis rather than all of them.
- **When:** `T` remains unaffordable after scope reduction.

**Factorise along all `m` axes, not into a two-way intra/inter split.** The two-way split is
the intuitive decomposition and the wrong one — it does not match the geometry of a lattice
with `m > 2`.

#### The self-defeat property

Axial is worth its expressiveness cost **only while attention dominates the cost model** —
and scope reduction (§3) removes that condition. This makes it far less useful than it first
appears, and the trap is that both look like "make attention cheaper."

A second cost, usually omitted: axial alternates intra-substructure passes (which want
substructure-major ordering) with cross-substructure passes (which want interleaved), so it
requires a transpose or strided access between passes. Naive comparisons ignore this and
flatter axial.

The main mechanism: axial eliminates the quadratic term but leaves the linear terms
untouched, *and* needs ~1.5× the depth, which inflates exactly those untouched terms. So its achievable
gain is bounded by attention's share of cost, and that share is precisely what scope
reduction has already collapsed.

Worked, from the companion document: at full scope, attention is 86% of cost and axial gains
4.8×. After a 3× scope reduction, attention is 67% and axial gains **2.0×** — while its own
attention share falls to 0.3%. It is optimising what is already negligible.

**Evaluate axial against the post-scope-reduction cost mix, never the naive one.** Measured
before scope reduction, it looks like a major lever. Measured after, it usually is not.

#### The corner where it survives

Axial is the only technique here that reduces cost **without dropping tokens and without
causal masking**. If scope reduction fails (most state is genuinely relevant) *and* the
immutability pattern's causal cost (§8.2) proves too high, it is what remains.

Document it against that condition rather than presenting it as a co-equal design option.

### 5.3 Oracle-biased attention

If the domain provides an exact structural relation — legal moves, chemical bonds, causal
edges, physical adjacency — inject it as an attention bias or hard mask on a subset of
heads.

This converts structure from something the model must learn into something it is handed.
It is usually cheap, because you are already computing the relation for other reasons.

**Head allocation:** a minority of heads with the hard structural mask, the majority dense.
Entities interact through more than the explicit relation, and a fully-masked model loses
everything the relation does not capture.

**Fixed multi-head routing.** The masked heads need not share one mask. If the domain
supplies several exact relations — adjacency, typed edges, axis-alignment, causal dependency
— give each its own head. No router, no learned parameters, all masks free from the
generator you are already running.

This has a useful property: assigning heads to *lattice axes* recovers axial attention's
inductive bias (§5.2) **without axial's costs** — the decomposition happens across heads
within a layer rather than across layers, so there is no extra depth and no transpose.

Speculative. Wrong priors waste head capacity, and precedent is thin outside graph
transformers. Distinguish sharply from **learned** block-sparse routing, rejected in §5.4
for the opposite reason.

**Token ordering determines whether this works at all.** Order tokens
**substructure-major** — all `k` tokens of substructure 1, then substructure 2, and so on.
Substructure-level blocks are then contiguous, and block-sparse kernels exploit a
substructure-level relation losslessly. Interleaved ordering (all position-1 tokens across
substructures, then all position-2) makes the *same logical mask* strided, and block
sparsity degrades to gather-bound. Same mask, same model, very different performance.

This is not kernel trivia — it decides whether §5.3 is viable, and it constrains §5.2 (see
below).

**Kernel viability:** given substructure-major ordering, whether this is fast depends on
whether the mask is **block-structured**. If substructures map to blocks and the relation is substructure-level,
block-sparse kernels exploit it losslessly. If the mask is fine-grained and irregular,
sparsity goes gather-bound and loses to dense. Determine which case you are in before
committing.

**Where oracle information belongs — relations, aggregates, never tokens.** A model
without attention needs pairwise relations as input features (NNUE-style pair inputs); a
transformer has attention, so pairwise relations go in as *edge biases* — and the masks
above upgrade naturally to **typed** edges, a small learned bias per head per relation
type, so a head knows how two tokens relate, not merely that they do. Aggregates the
model would otherwise compute by summing over attention (counts, board-level status) go
into token or register *embeddings* at layer 0, cheaply and with redundancy risk worth an
ablation. What relations must never become is extra *tokens*: edges outnumber tokens by
orders of magnitude, and tokenising them explodes the quadratic term. The test for any
proposed pre-processing: does the model receive *more* information at the same token
count (welcome) or *less* information at a different one (refused)?

### 5.4 The corollary: skip trained sparse attention

Methods that *learn* which blocks to attend to (NSA, MoBA, and relatives) train a router to
approximate a relevance oracle. **If your domain provides that oracle exactly, spending
model capacity to rediscover it is strictly worse than using it.**

This is a general property of structured-state problems and a genuine departure from LLM
practice, where no such oracle exists.

---

## 6. Registers

Dedicated learned tokens that give the model scratch space it would otherwise steal.

### 6.1 Why they matter more here than in vision

The finding they come from is that models repurpose low-information tokens as global
scratchpads, producing high-norm artifacts in those tokens' representations.

In structured-state problems this is worse, because **low-information tokens are usually
not low-*value* tokens.** An empty cell, an inactive agent, an unoccupied site — these
carry little intrinsic information but their representation often encodes something the
task depends on (what controls this cell, what could occupy this site). Hijacking destroys
exactly what you need.

### 6.2 The pricing asymmetry

| Type | Cost scaling | Guidance |
|---|---|---|
| **Per-substructure** | `O(N)` tokens → `O(N²)` attention | **Be stingy.** 2-4 |
| **Global** | Constant | **Be generous.** 16+ |

Per-substructure registers multiply by `N`. Vision-transformer register counts do not
transfer, because there the register count is fixed while patches scale — here both scale.
A guidance of "4-8 registers" borrowed from that literature is roughly double what the
pricing supports.

### 6.3 Implementation

- Give registers their substructure's lattice coordinate so they know what they summarise,
  but **not** an intra-substructure position. Use a learned "not a real position" embedding,
  or they collide with the geometry.
- **Exclude from pointer heads** (§7) — registers are not valid outputs.
- In the factorised topology (§5.2), per-substructure registers carry the inter-substructure
  pass. **This is strictly better than mean-pooling**, which destroys the internal identity
  that cross-substructure relations depend on.
- Honest expectation: 0.5-2%, sometimes nothing. Cheap enough to take, first thing to
  ablate.

---

## 7. Variable output spaces

### 7.1 Pointer heads

When the action space scales with `N`, fixed logit vectors cannot express it. Score
candidate actions by bilinear combination of their endpoint representations:

```
score(a → b) = f(h_a)ᵀ g(h_b)
```

evaluated only over candidates supplied by the domain's generator. Handles arbitrary `N`
for free.

`f` and `g` must be **separate projections** — source and destination roles are not
symmetric, and tying them imposes a symmetry the domain usually does not have.

### 7.2 Compositional actions

If a single decision comprises several sub-actions across substructures:

- **Autoregressive within the decision** — condition each on the previous. More accurate,
  slower, and introduces an ordering the domain may not have.
- **Decompose into independent steps** — simpler, integrates with search, loses
  intra-decision coordination.

No general answer. Benchmark both.

### 7.3 Scalar heads have a precision floor

Any head producing a scalar that will be *compared* — a value estimate, a ranking score —
has a discrimination requirement independent of the trunk's. See §10.2.

### 7.4 Supervision density scales with the state

In fixed-size problems, outcome-level supervision is a constant ratio — so many labelled
points per example. In variable-`N` problems the ratio is a *function of the state*: one
outcome label supervises `N × k` tokens, so per-token supervision dilutes as episodes
grow, often by orders of magnitude within a single episode.

The lever is auxiliary targets whose granularity matches the token structure.
Per-substructure and per-token targets deliver signal proportional to `N` and `T`, so
supervision keeps pace with what it supervises. Global auxiliaries (outcome
decompositions, horizon estimates) are cheap and proven but dilute exactly as the main
target does — they add a constant, not a scaling fix.

Two rules travel with this:

- **The verifier's estimates never enter the loss.** Self- and search-derived targets are
  the norm — the imitation target itself is search output — and episode records,
  generators and checker *proofs* are all fair sources. The one prohibition is the
  learned verifier's *estimates*: training on them wires the verifier's blind spots into
  the main model, destroying the differently-wrong property (§13.2.1) through the loss
  function. State the rule this precisely — an earlier phrasing, "exact sources only,"
  technically prohibited the imitation target itself.
- **Match the reduction to the size distribution.** Summing per-substructure losses grows
  the auxiliary gradient share with `N`; averaging keeps it stable but diluting. An
  `N`-structured hyperparameter — decide it explicitly, or it surfaces as unexplained
  instability when the size distribution shifts.

### 7.5 The variant test

A useful probe of an architecture's generality: how much changes when the domain's rules
change? In this pattern the answer should be — the generator changes, the vocabulary
gains rows, and the network's shape does not. Fixed output planes fail the test (any
change to the action space breaks the head); pointer outputs over generator-supplied
candidates pass it. Structure-agnostic tokenisation and relative encodings extend the
same property to state-space variants — sizes, extra lattice regions. Conditioning flags
in global registers carry rule identity, enabling one net across variants — with two
standing caveats: exact-checker correctness is inherited *per variant* (a wrong variant
generator manufactures false facts, not just weak play), and weights do not transfer
across variants that invert the value structure. Two additions from practice:
conditioning is only needed where identical states carry different rules —
self-identifying variant states are just training diversity, and fold into the main
model's curriculum for free. And exact stores are per-ruleset: fold rule identity into
the state key (§8.6's scheme takes one more ingredient), because a fact proved under one
rule set is false under another even when the states look identical.

### 7.6 Distributional heads for decision-consuming quantities

When a predicted quantity feeds a decision that needs more than its mean — an allocator
wants a quantile, a commit rule wants a bound, a scheduler wants a "won't resolve"
probability — make the head **distributional** rather than adding separate confidence
and bound outputs: categorical over bins yields mean, spread, any quantile and tail mass
from one head, and its calibration is directly checkable against realised outcomes. Two
details recur: use log-scale bins for heavy-tailed quantities (effort, duration), and
give **censored** labels — observations truncated by a budget — a top bin of their own,
so the head learns "beyond budget" as a class instead of absorbing it into a corrupted
mean.

---

## 8. The immutability pattern

### 8.1 When it applies

If your state is **append-only** — substructures are added and never subsequently modified —
the frozen portion has exactly the structure of an autoregressive KV cache. You can compute
only the live frontier per step and attend into cached representations of the history.

Test for applicability: *does any operation modify an existing substructure in place?* If
mutation happens, the pattern does not apply. If mutation is expressed as *appending a new
substructure* — which is common in versioned, branching, or event-sourced state — it does.

### 8.2 The cost

Freezing representations requires **causal masking along the append axis**: frozen
substructures attend only to their own history, never to the live frontier.

You lose the ability for a frozen substructure's representation to be conditioned on what
currently makes it relevant. The information is still available — the *frontier* tokens
attend bidirectionally to everything — but the frozen representation itself is built
context-free.

### 8.3 The right framing for that cost

This is the **bi-encoder versus cross-encoder** distinction from retrieval:

- **Causal / cached** = encode each historical item independently, once (bi-encoder).
- **Dense / bidirectional** = encode it jointly with the current query (cross-encoder).

Cross-encoders consistently win on ranking quality. Bi-encoders remain good enough that
the entire retrieval industry runs on them. That is the right prior for the size of the
cost: **real, moderate, not catastrophic.**

It also points at the mitigation: a **hybrid** — mostly causal layers that cache, plus a
small number of dense bidirectional layers — is precisely a bi-encoder with a cross-encoder
re-ranking stage. Expect most of the speedup with a fraction of the quality cost.

**Placement is forced, not a guess.** A bidirectional layer lets present information into
the frozen representations at its depth, so every layer above it must be recomputed per
query: the cache survives only below the *first* bidirectional layer. The bidirectional
layers therefore go on top — interleaved, the same layers cost roughly twice as much. The
3:1 interleave of linear-attention hybrids does not transfer: there every layer is causal,
so nothing above an attention layer is invalidated. Global tokens follow the same rule by
direction: reading the frozen history is free; being read by it is present-to-frozen flow,
paid for only at refreshes.

The interleave that *is* cache-compatible runs on a different axis:
**within-substructure-only layers with a periodic cross-substructure causal layer** — the
local/global pattern — which cuts per-query attention and stores K/V only at the global
layers. It is the layer-level factorisation of per-head structural routing (§5.3): cheaper
and lighter on memory, at the cost of cross-substructure reach at every depth.

Two further rules keep the cache honest. **The live set attends fully to itself** — the
substructures being recomputed each query see one another bidirectionally, because
interactions among them can run against the causal order, and structural edges between
them need an attention path; the cost is recomputing every live substructure per query,
not just the changed ones, and it belongs in the cost model from the start. And **a
substructure frozen mid-search is recomputed under the state-defined mask, never
snapshotted**: its last live representation attended to whatever else was live on that
path, and freezing it makes evaluations path-dependent — which silently breaks every
state-keyed cache and hands the model representations its training mask never produced.
Shortcuts that skip recomputation for "unaffected" live substructures fail the same way
whenever any dense attention couples them.

### 8.3.1 Query-conditioned refresh — recovering most of the lost quality

The bi-encoder framing suggests its own mitigation, and it is unusually cheap.

If the frozen representations are consumed by many related queries — a search tree, a batch
of variants, repeated evaluation against a slowly-changing context — **re-encode the frozen
history once per query *group*, conditioned on that group's shared root.** Every member of
the group then attends into representations that already know what is currently relevant.

The binding constraint is cache sharing: a refreshed representation is reusable only by
consumers sharing its conditioning. Refresh per consumer and the cache is gone. **Refresh at
the shared root and the whole group benefits from one pass.**

Cost, from the companion document: a full dense refresh amortised over 800 tree nodes is
**+3.8%** per node; over 3,200, +0.9%. What remains lost is conditioning on each consumer's
own delta from the root — typically small.

In retrieval terms this sits between bi-encoder and cross-encoder: encode once per query
*group* rather than once globally or once per query. It substantially narrows §8.2's cost,
and it is a **training-time structure** (a two-level mask), not an inference trick.

Refreshing at intermediate points has much worse economics — breakeven is around 10-30
consumers per refresh, and intermediate nodes differ little from the root.

When consumers *persist across* refresh boundaries — a search tree reused across decision
points — split what each node holds into two kinds of staleness. **Derived statistics**
(visit counts, aggregated values, bounds) are mildly stale and self-correct as fresh
results wash them out: carry them. **Cached representations conditioned on the old
refresh** are poisoned, not mildly stale: invalidate them and recompute lazily on next
use. This dissolves the apparent choice between dropping reuse and accepting corruption —
the two kinds of staleness were never the same thing.

Refreshing *within* a decision (at internal nodes) is a ladder, not a yes/no. Uniform
internal refresh fails on economics — a refresh costs many node-evaluations, and most
internal nodes differ little from the root. But conditioning can drift *discontinuously*
when a step changes which immutable substructures are live (a branch into an old region),
and those steps are where a targeted refresh pays. The general rule is **staleness ×
consumption**: refresh where the conditioning has moved *and* the subtree beneath will
consume many evaluations. Three flexible forms, in order of cost: predict drift from the
step type; *measure* it by refreshing a sampled few substructures and checking how far
their representations moved, with the threshold tuned online against realised benefit;
or refresh individual substructures ranked by attention mass × staleness within a budget.
Sequencing rule: the crude trigger is the existence test and the flexible one the
efficiency version — never build the efficient form of a benefit not yet shown to
exist. And a refresh below the root **forks** the cache copy-on-write for its
subtree — it never overwrites the shared cache, or sibling subtrees attend into history
conditioned on a position they do not descend from, and evaluations come to depend on
search order. Anything recomputed inside the refreshed subtree must use the subtree's
epoch, not the root's, or the refresh is silently undone for everything the subtree
creates.

**Train the regime you deploy.** A refresh pass is the ordinary dense model, but
everything evaluated *below* a refresh sees history conditioned on an ancestor rather than
on itself — a regime training on fully refreshed passes never shows. Sample a conditioning
offset per training example from the deployment's depth distribution (zero included),
refresh on the ancestor that many steps back, and evaluate against that history; targets
are unchanged. Without it, every refreshed configuration is evaluated off-distribution —
and any comparison against a plain causal baseline, which needs no offsets, is biased
against the refresh.

### 8.4 Composition with scope reduction

**Scope reduction and caching are complements, not substitutes.** Scope shrinks `T_total`;
caching shrinks `T_new`. They act on different factors of `T_new × T_total` and multiply.

This is easy to get wrong. Reasoning about each in isolation makes them look like competing
answers to the same question, and they are not.

### 8.5 Silent-failure risk

Caching introduces a failure mode dense attention structurally cannot have: **a cache that
disagrees with the live computation** — stale entries, mask mismatch, precision mismatch
between stored and computed values.

The output is plausible but wrong. It does not throw. In a system where quality is measured
statistically over many episodes, this can persist indefinitely.

Dense evaluation is a pure function of the state. That property is worth more during
development than its cost suggests.

### 8.6 Identity and hashing for branching state

Branching, append-only state guarantees near-duplicate substructures — branching *is*
copying. That breaks the standard trick of hashing a set of substructures by XOR-combining
per-substructure hashes: XOR cancels on duplicates (a⊕a = 0), so states differing only in
*where* identical substructures sit alias silently. In this problem class that is not a
rare collision; it is structural.

The fix that keeps O(1) incrementality: hash substructure *contents* coordinate-free
(incrementally updatable as usual), then mix each content-hash with its lattice coordinate
through a non-cancelling operation — multiply-rotate, not XOR — before combining. Design it
in from the start; discovered later, it presents as results that are mysteriously,
irreproducibly wrong.

A related gift of append-only state: if every step adds to the state, the full state can
never recur within an episode, so repetition rules have nothing to act on and
graph-history interaction cannot arise — values depend on the state alone, and
transposition sharing is sound without path bookkeeping. Check the domain's rules for
any other path-dependent term (move counters, repetition clauses) before relying on it.

---

## 9. The cost model

### 9.1 The arithmetic

Per layer, with `T` tokens, model width `d`, FFN ratio `r`:

| Term | FLOPs | Scaling in `T` |
|---|---|---|
| Attention (QK^T + PV) | `4T²d` | **Quadratic** |
| QKV + output projections | `8Td²` | Linear |
| FFN | `4Tdrd` = `4rTd²` | Linear |

At large `T` the quadratic term dominates completely — typically 85%+ of total cost.

### 9.2 The scaling inversion

Because attention is quadratic in `T` and linear in `d`:

> At extreme sequence lengths, go **narrow and deep, and buy capacity in the FFN.**

This inverts standard transformer scaling advice, and it holds for a specific reason:
per-token information content in structured-state problems is *low*. A token representing a
cell with a dozen possible states does not need the width a natural-language token needs.
Meanwhile `d` is taxed quadratically through the attention term while FFN width is not
taxed at all.

Widening the FFN ratio from 4× to 8× is nearly free at these lengths. Raising `d_model` is
not.

### 9.3 The arithmetic trap

**The single most common error in this analysis is computing a ratio on the attention term
and quoting it as a throughput number.**

Because attention is 85%+ of cost, the approximation is *nearly* right, which is precisely
why it survives review. But any optimisation that reduces `T` improves the linear terms only
linearly while improving the quadratic term quadratically — so the total improvement is
strictly less than the attention-term improvement.

Worked: a 3× reduction in `T` gives 9× on attention but ~7× on total.

**Discipline:** compute total cost including FFN and projections every time. State which
denominator any percentage uses. If a figure moves between revisions, record whether it
moved because the design improved or because the arithmetic was wrong — those are very
different facts and they are easy to conflate in your own favour.

And rerun the check when porting any per-pair mechanism across scale: a technique that is
trivial at 64 tokens is a ~62,000× larger quadratic object at 16k tokens. Smallness in
the source domain proves nothing in the target. The corollary runs the other way too: a
technique *declined* at one scale must be re-derived at the scale it would actually run
at — a per-substructure overhead priced on a single substructure can be a rounding error
across the whole state, and refusals carry the assumptions of the moment they were made.
When later decisions dissolve an objection, the refusal does not update itself.

And when a dynamic pairwise mechanism is too large at token granularity, look for its
**factorised form** — within-substructure terms generated from per-substructure
registers plus a substructure-pair term from global registers — which is typically
orders of magnitude smaller than the attention matrix it modulates. And check whether a
generated low-rank bias is extra attention dimensions in disguise: a rank-`r` bias
`u_iᵀ M v_j` is identical to appending `r` context-modulated dimensions to Q and K, at
`+r/d_head` cost — if that is the only affordable form, it is the head-dimension lever,
not a new mechanism, and should be priced as such. The full-rank freedom that makes such
generators valuable survives only at the granularity where the pair count is small.

### 9.4 Batching does not behave like LLM serving

At large `T`, a single instance already saturates the accelerator. Batching buys perhaps
1.2-1.5×, not the order of magnitude it gives decode-bound sequence generation. Plan
throughput on single-instance numbers.

### 9.5 Budgeting over an episode

When state growth is deterministic — each step adds a known increment to `N` — every
future step's cost is known from the cost model, so per-step budgets should be allocated
against the **integral of the remaining cost curve**, not against current throughput.
Systems that estimate throughput empirically must assume it persists, and starve the late
episode by exactly the cost ratio; in the companion's worked example a flat wall-clock
budget leaves the final phase ~3× under-computed. The steps-remaining estimate is usually
the noisiest input — floor it conservatively. And denominate any adaptive extensions in
compute units, with the wall-clock ceiling held tight where per-step cost peaks — an
extension mechanism must never be able to trigger itself.

---

## 10. Precision

### 10.1 Default assignment

| Component | Precision | Reasoning |
|---|---|---|
| Attention matmuls | BF16, FP8 later | Softmax is bounded, tolerant |
| Softmax, accumulation | **FP32** | Always |
| FFN, projections | BF16 | Small share of cost; low payoff to push |
| Positional encoding | **FP32** | See §10.3 |
| Discrimination-limited heads | **FP32** | See §10.2 |
| Master weights, optimiser | FP32 | Standard mixed precision |

**BF16 over FP16.** FP16's 5-bit exponent overflows near 65,504, and attention logits at
large `T` run high pre-softmax. BF16 keeps FP32's exponent range and eliminates loss
scaling.

### 10.2 The discrimination floor

Any head producing a scalar that will be **compared or ranked** has a precision requirement
independent of the trunk.

FP8 E4M3 has 3 mantissa bits — roughly 2 decimal digits. If meaningful decisions turn on
differences finer than that, FP8 does not merely add noise, it **collapses the ordering**,
and it does so *systematically* — the same inputs get misranked every time.

This is a real divergence from LLM practice. In language modelling, quantisation error
redistributes probability across a large vocabulary and is absorbed. In a ranking or
valuation head, it changes the argmax.

**Keep such heads in FP32.** They are usually a trivial fraction of parameters.

### 10.3 Positional encoding precision

Rotation angle error compounds with coordinate magnitude, and lattice coordinates in these
problems often far exceed the ranges positional schemes were designed for. Combined with the
head-dimension split (§4.4), this makes positional encoding unusually precision-sensitive.

Compute in FP32, cast to BF16 for the kernel.

### 10.4 Measuring quantisation impact

**Measure in task metric, not reconstruction error.** MSE against the full-precision model
hides systematic ranking errors, which are exactly the failure mode that matters. If your
task has an end-to-end evaluation, use it.

If you cache (§8): **stored and computed precision must match.** A mismatch is the silent
failure of §8.5 in its purest form.

---

## 11. What transfers from LLM serving

The serving literature is organised around a **prefill (compute-bound) versus decode
(bandwidth-bound)** split. Structured-state workloads often sit in neither. Placing yourself
correctly on that map determines which techniques apply.

The useful diagnostic: **attention arithmetic intensity ≈ query length.** Compare against
your accelerator's ridge point. Dense full-state evaluation is deep in compute-bound
territory; cached frontier evaluation sits near the knee.

### 11.1 Transfers well

- **FlashAttention.** Mandatory at large `T` — not for speed but for **memory**. The
  materialised score matrix is `T²` per head per layer and will not fit. Its **online
  softmax** component is also an architectural primitive in its own right (§3.3.1), not
  merely a kernel detail.
- **Paged KV allocation with copy-on-write.** Especially if you explore a *tree* of states
  with variable-length caches — fragmentation is worse here than in LLM serving.
- **Radix / prefix caching.** If your states form a tree with shared history, the mapping is
  exact.
- **Cascade inference.** Shared-prefix-plus-suffix decomposition matches the "batch of
  related states" shape directly.

### 11.2 Conditional

- **FlexAttention** — needed if using learned attention biases; unnecessary with RoPE.
- **KV-head reduction (GQA/MQA)** — see §12.2. Usually not worth it, and retrofittable.
- **FP8** — see §10.
- **Mixture-of-Depths** — routing low-information tokens through fewer layers fits this
  problem class unusually well. Speculative; no established precedent here.

### 11.3 Actively dangerous

**Learned KV eviction** (attention-score-based history pruning). In language modelling,
dropping a low-attention token costs some recall and degrades gracefully. In a
structured-state problem with **exact recall requirements**, a dropped substructure may be
the one that determined the outcome. That is not a quality regression — it is a correctness
failure, surfacing as unexplained metric loss long after the fact.

The structural idea (attention sinks plus a window) is fine. Use §3.3's rule-based version.

**Linear attention and state-space models.** Two independent disqualifiers:

1. They impose a **sequential ordering** that lattice-structured state does not have.
2. Their consistent failure mode is **recall** — and structured-state evaluation is
   usually recall-dominated. "Is there an X at position Y" is not a question to route
   through an approximate-recall operator.

**Speculative decoding, multi-token prediction, continuous batching.** All assume
autoregressive token generation. If your model does single forward passes, they are
meaningless.

### 11.4 The second tier

Sorted by the principles above rather than case by case. **KV-cache quantisation** and
**varlen/ragged kernels** transfer — memory and padding are real costs in variable-`N`
batching. **Early exit / adaptive depth** is conditional: a self-assessment mechanism,
so it joins the confidence family (§13.2.4), allowable only while the exact checker's
baseline never scales down with the model's self-rated ease. **Within-system
distillation** — a large model into a smaller one of the *same* role — is a legitimate
capacity tool; the refused distillation is *between* differently-wrong systems, not
within one. **Latent-KV compression** (MLA-style) fails for the reason KV-head reduction
fails: a low-rank bottleneck on a key axis already marginal under high-dimensional
positional encoding. **Mixture of experts** offers capacity at constant per-token cost,
but here the FFN ratio is already the nearly-free capacity rung, so the gain is small and
routing overhead is not. And speculative decoding, meaningless as stated, has a
structural cousin — speculative prefetch of evaluations to fill under-filled batches —
that is quality-neutral under state-keyed caching and matters only where batches are
latency-bound.

The pattern: compute-reducing tricks transfer or are neutral; capacity-reducing tricks on
the key axis hit the same wall; self-assessment tricks join the confidence family. No new
principle is needed to sort the second tier.

---

## 12. Decision sequencing

### 12.1 Committed versus retrofittable

The most useful planning distinction, and one that is rarely made explicit.

| Decision | Type | Why |
|---|---|---|
| Causal masking (§8) | **Committed** | Train/inference mask mismatch causes distribution shift. Cannot be added after training |
| Positional scheme (§4) | **Committed** | Baked into learned representations |
| Tokenisation and scope (§3) | **Committed** | Determines what the model has ever seen |
| Width, depth | **Committed** | Retraining required |
| KV-head reduction (§12.2) | **Retrofittable** | Uptrains at ~5% of original compute |
| Kernel choice, precision | **Free** | Inference-time |
| Serving stack (§11.1) | **Free** | Inference-time |

**Committed decisions must be made before spending serious training compute. Retrofittable
ones must not be hedged toward.** Adding a retrofittable feature "just in case" pays its
cost immediately for a benefit you may never need.

One further distinction: **a decision can be uncommitted (not yet made) and still
committed-class (irreversible once training starts).** Classify by class, not by status —
an unadopted committed-class choice must have its deciding experiment scheduled *before*
training, while an unadopted retrofittable one can simply wait. Conflating "we haven't
decided" with "we don't need to decide yet" is how committed-class decisions get made by
default instead of by measurement.

### 12.2 KV-head reduction: usually the wrong call here

Worth a specific note, because it is near-universal in LLM practice and the reasoning does
not carry over.

**GQA reduces the number of KV heads, not query heads.** Every query head still computes its
own scores against the shared keys, so **the attention matmuls are unchanged** — and those
are 85%+ of cost. Only the K/V projection GEMMs shrink, typically ~1% of total.

Its real value is **cache memory**, which matters only if you cache (§8) *and* memory
actually binds — check, with delta storage assumed, before believing it does.

Against that: it shrinks the key space, which under `m`-dimensional positional encoding is
already stretched thin (§4.4).

**And it retrofits.** So the correct treatment is: do not adopt it now, revisit only if a
cached deployment demonstrates memory pressure.

### 12.3 General ordering

1. **Measure `|Far|`** — the relevant-but-outside-cheap-scope set (§3.3.1) — on real
   episodes. This sets realistic `T` and determines whether §8 is a live question.
2. **Validate relative positional generalisation** at unseen `N` (§4.2). If this fails,
   stop — nothing else matters.
3. **Choose topology** (§5) against the measured `T`.
4. **Build the dense, uncached baseline.** It is the only thing you can price other options
   against.
5. **Ablate** registers, oracle-bias head count, width/depth at matched wall-clock.
6. **Only then** consider caching (§8), and only if throughput demonstrably binds.

Step 4 is the one most often skipped, and skipping it means never learning what the
optimisations cost.


### 12.4 Build untrained first

An untrained model costs exactly what a trained one does, and random weights are the
ideal fixture for correctness testing — so the whole system can be built and verified on
two of its three pillars before any training compute is spent. **Correctness** by
equivalence: every pair of computation paths that must agree (cached vs uncached, windowed
vs full-within-scope, lazy vs eager, incremental vs from-scratch, reduced vs full
precision) compared directly, with random weights at realistic scale (zeros make distinct
paths agree trivially), several seeds, and tolerances matched to the arithmetic. **Cost**
by measurement: every throughput, memory and overhead estimate the design's economics rest
on, measured and written back over its estimate with its conditions. Only **quality** —
whether an idea improves the outcome — needs training, and it is best measured on a
codebase whose correctness and economics are already known. Order still matters: the
domain generator and its differential-testing gate first, cheap measurements before the
code they gate, and committed-class alternatives behind flags, since the ablations need
both arms anyway. And pre-register the decision thresholds for the cheap measurements before the
numbers exist — derived from the cost model and the design's own priors where possible,
labelled as judgement where not. Thresholds set after the results tend to move toward the
results; set beforehand, the decision is a lookup.
---

## 13. Methodology and recurring failure modes

### 13.1 Build the simple version first, expect to replace it

There is a real distinction between **the right first build** and **the right final
architecture**, and they are frequently different.

The first build's job is to produce the baseline that prices everything else. An
optimisation whose cost you cannot measure is an optimisation you cannot justify. Building
the sophisticated version first means never learning what it cost.

State this explicitly in design documents, because "build X first" and "X is correct" read
identically and mean very different things.

### 13.2 Distinguish error correction from design improvement

When a performance estimate improves across revisions, determine whether the design got
better or the arithmetic got fixed. These feel identical from the inside and are opposite
facts. Keep a revision table with causes attached.

### 13.2.1 A verifier must be differently wrong, not more cheaply wrong

A common architecture is a cheap model auditing an expensive one — checking cases the
expensive model pruned, or flagging disagreement. The design fails in a specific,
predictable way:

> A verifier catches only what it is **differently wrong** about.

**Distillation is the worst case**, because it explicitly optimises the small model to
agree with the large one. It will confidently endorse exactly the errors it was built to
catch.

**Independent training of the same architecture is weaker than it looks.** Error correlation
comes from the training data and the architecture's inductive biases, not from
initialisation. A second model sharing your positional scheme, your structural bias, and
your data shares most of your blind spots. Ensembles give modest gains for exactly this
reason.

Decorrelation ranked:

| Verifier | Decorrelation | Note |
|---|---|---|
| Exact algorithmic check (no learned component) | **Total** | Also usually cheapest — dominates on both axes |
| Same model, different search or inference procedure | Procedure errors decorrelate; model errors do not | Free — no second model |
| Different architecture *and* different training signal | High | Often a research project of its own |
| Same architecture, independent training | Low | |
| Distilled model | ~None | Never |

**The two categories do different jobs and are not interchangeable.** Exact checks catch
*procedural* failures — missed cases within a horizon, unexplored branches. Learned
verifiers catch *evaluative* error in cases both models examine. Decide which failure mode
dominates before choosing, and note that in systems with large branching and shallow search,
procedural failure usually dominates.

**Verdicts attach to (component, role) pairs, not components.** The same architecture can
be disqualified in one role and correct in another: an evaluator whose blind spot is fatal
as an independent verifier can be exactly right as the leaf eval inside an exact search,
because the search enumerates the dynamics and the eval adjudicates only the statics. When
rejecting a component, record *which role* the rejection binds to — a wholesale rejection
silently forecloses roles that were never analysed.

**The boundary is between systems, and the training rule is about losses.** Two precision
points that prevent over-application: (1) within-system self-correction at inference — a
component adjusting its own outputs from its own deeper computation — is legitimate and
common in mature systems; the violation is *cross-system* coupling, one evaluator
deferring to the other. (2) The prohibition on verifier outputs applies to *loss
targets*; inference-time search statistics that never train anything are governed by (1),
not by the training rule. Misapplying either rule rejects sound techniques.

**Two operational corollaries.** Checker features are adoptable exactly as far as its
factorisation reaches: a feature that crosses the factorisation boundary — a
cross-substructure relation fed to a separable evaluator — destroys the incremental
properties the checker's economy rests on, and smuggles the main model's representational
territory into the auditor. And check the record before refusing: the commonest way a
sound rule yields a wrong verdict is being applied from memory against evidence already
gathered — a survey row contradicting the refusal in the same document. Grep the registry
first.

### 13.2.2 Proofs, estimates, and when verification loops stop

An exact checker's findings come in two grades, and conflating them wastes both:

- **Proofs** — outcomes decided within the checker's horizon. Propagate as hard bounds
  that *remove* work: solved regions leave the search and stop consuming model
  evaluations. This is the one place verification reduces model load rather than adding
  to it.
- **Estimates** — findings the adversary or environment can decline. Use as allocation
  signals (priors, forced expansion), never as bounds. And check the domain's version of
  the stand-pat assumption — that declining costs nothing — before trusting even that;
  domains with zugzwang-like states break it.

**Termination is event-driven, never scheduled.** Do not build an outer
verify → refine → verify loop on top of a search process — the search *is* the refinement
loop. Re-verification triggers on **refutation** (the checker overturns the chosen answer),
and it converges because a deterministic checker rescanning unchanged state yields nothing:
only newly added state needs checking, findings are monotone, and one or two rounds should
be a hard cap.

**Do not chase global quiescence.** In large-`N` systems, "nothing volatile anywhere" is
usually an unreachable fixed point, and pursuing it starves the decision that actually
matters. Scope volatility checks to the decision-relevant region — the chosen answer and
its near-alternatives, not the whole state.

**Committing under uncertainty.** Commit by evidence quantity with a lower-confidence-
bound correction, never by point estimate — a high-estimate, thin-evidence candidate
should lose to a slightly lower one with deep evidence. Keep the uncertainty penalty out
of exploration: applied during search it suppresses exactly the discovery it is meant to
guard. And let the exact checker, not a depth statistic, answer "did we look far
enough" — depth-as-quality favours narrow continuations over flexible ones, and the
checker extends the chosen line exactly along the moves that matter. And mirror the
commit-time penalty with an exploration-time *bonus*: a penalty on thin evidence is
self-fulfilling unless search preferentially resolves the uncertain candidates — explore
to resolve, then decide on what resolved. Keep the bonus multiplicative on the prior so
it cannot resurrect candidates the prior has already ruled out.

### 13.2.3 Running the checker concurrently

Once a checker exists, scheduling it is its own design problem. Four rules recur:

- **Budget against shared substrate, not against the accelerator.** "The checker is
  CPU-side, the model is GPU-side, so concurrency is free" fails whenever both consume the
  same generator or simulator. Give the checker a budget and a priority order; unbounded
  background checking starves the process that makes the decision.
- **Pipeline under existing latency first.** If model evaluations queue for batching, the
  check on each item can run inside that queue latency — near-free, needing only machinery
  async search wants anyway (late merge of monotone findings). This rung usually captures
  most of the value and should be built regardless of the rest.
- **Stream verification; don't end-load it.** A refutation found mid-process is a cheap
  redirection; the same refutation after the budget forces a re-run. Same work, better
  placement. Trigger on decision-change events.
- **Audit immutable state exactly once.** Append-only substructures (§8) can carry
  standing annotations — computed at creation, valid forever, amortised to nothing. Often
  the cheapest cover for a learned component's structural blind spots (e.g. a separable
  eval's cross-substructure blindness).
- **Parallelise by sharing results, not scheduling work.** Independent workers writing to
  a shared memo table scale better than orchestrated work-splitting, and a ladder of
  diverse audit tasks is a natural set of helper threads — path diversity is the point,
  not a defect. One invariant: the memo table is lossy (eviction, benign write races), so
  it may hold estimates but never the sole copy of a proof — proofs export to loss-proof
  storage the moment they are found.
- **Audit reversals hardest.** A reversal — the system abandoning a heavily-considered
  candidate — is a high-leverage judgement by the learned component, and the default audit
  pattern misses it: the chosen answer gets end-verification, the never-considered
  candidates get screening, and the rejected-after-consideration ones get nothing. Give
  reversal points an escalated checking budget (they are rare, so escalation is cheap),
  and route checker-vs-model disagreement there into the late curriculum (§13.2.4).

One hazard: if the checker steers the search that generates the learned model's imitation
targets, the model trains toward the checker's bias — the differently-wrong property
(§13.2.1) violated through the training loop rather than at inference. Exclude or
down-weight checker-forced visits from distribution-based targets; outcome-based targets
are safe. And the exclusion must survive state reuse: if search state persists across
decision points, checker-forced influence persists with it — either track provenance
across the reuse boundary, or generate training data with fresh searches per decision
(the common convention). Deployment, which emits no targets, carries freely.

And retain a synchronous mode: concurrency is paid for in reproducibility, and an
unreproducible system cannot be debugged by comparison.

### 13.2.4 The checker at maturity

As the learned component converges, its errors change character — from frequent-random to
rare-systematic-confident, because jointly trained components (policy and value, generator
and evaluator) converge together and share blind spots. Consequences:

- **Exact coverage gains relative value over time.** It is distribution-free — rules, not
  learning — so it cannot be converged-toward. Stripping the checker from a mature system
  removes coverage of exactly the error class that remains.
- **Checker-flagged states are the natural late curriculum.** Converged self-generation
  produces data the model already masters; the flags mark the gaps. Seeding generation
  from them is safe where the imitation-target boundary (§13.2.3) holds — selecting
  *positions* is not steering *targets*.
- **A falling catch-rate is a biased metric.** Training on caught errors migrates the
  residual beyond the checker's horizon. The honest probe is held-out adversarial
  evaluation, not internal agreement.
- **Let the model's own uncertainty set the pruning.** The entropy of an
  already-computed distribution is free; prune candidate sets hard where it is low, widen
  where it is high. This handles known uncertainty — the confident-wrong case is what the
  checker and adversarial evaluation exist for. Uncertainty may also *add* checking
  budget above the uniform baseline — routing exact verification to admitted weakness —
  but never reduce the baseline, which exists for the errors the model does not know it
  is making. And monitor the calibration of whatever uncertainty is consumed, bucketed by
  state size: mechanisms that spend it inherit its miscalibration exactly where the
  distribution drifts.
- **Self-correction tables double as drift detectors.** An inference-time correction
  keyed by state size or phase, persistently nonzero in one bucket, is a measured
  train/deploy-distribution gap — route that bucket to the generation curriculum. The
  same statistic legitimately serves three roles: online correction, offline diagnostic,
  and generation seeding.
- **Uncertainty may prioritise training data, with two guards.** Weighting the training
  distribution toward what the model is unsure about (prioritised-replay lineage) is a
  legitimate third curriculum source — but prioritise *epistemic* uncertainty, not
  aleatoric: raw predictive variance conflates knowledge gaps with inherently volatile
  states, and uncertainty that persists across checkpoints for a state class is the
  honest trigger. And keep it additive: uncertainty-driven sampling under-samples the
  confident-wrong class by construction, which is what the confidence-independent
  generators are for. Confidence signals flow *out* of the model into search, checking
  and data selection — never back into the model as an input, which is circular and
  invites feedback collapse.

### 13.2.5 Adversarial populations

Once the exact layer strips the cheap exploits, the remaining errors are quiet and
confident — and the efficient way to find them is to pay attackers to look. A population
of narrow exploiters trained against frozen checkpoints, sampled by loss rate, is the
late-stage curriculum generator; it costs a fraction of main training because finding a
hole is cheap and playing well is not. Three rules from the precedents: **hold some
adversaries out** — beating every adversary you trained on measures memorisation; the
honest robustness metric is **cost of attack** (compute a fresh adversary needs to reach
a win threshold — compute, not steps, so it compares across attacker sizes), not defended
win rate; and expect attack cost to *rise without closing*
— iterated adversarial training has never produced full robustness, only expensive
attacks. Scope vulnerabilities are the exception to "train it away": they are fixed by
additive scope rules under the floor-and-filler discipline (§3.3.1), with the
adversaries' findings as the tuning signal.

**Vary attacker size for coverage, not cost.** When the victim is the expensive party,
its evaluations dominate every game, so a much smaller attacker makes games only modestly
cheaper; the real cost lever is the victim's search budget, raised as a curriculum, with
testing always at deployment configuration. What size buys is coverage — small attackers
find simple holes, large ones represent multi-stage plans — and affordable variants: a
small attacker can afford full observation or from-scratch priors that escape the
victim's inherited blindness. Robustness then becomes a curve, attack cost against
attacker capacity, with the held-out set stratified by size. Bound it at both ends: an
attacker too weak to reach the exploitable states finds nothing, and one at or above the
victim's capacity may win by general strength — score every attacker for **specificity**
(win rate against the victim minus win rate against a reference pool) before counting
its wins as exploits.

### 13.2.6 The checker is only as exact as its rules implementation

Every guarantee the exact layer provides — proofs, permanent solved facts, structural
edges, terminal values — is inherited from the domain generator, so the generator is the
most correctness-critical code in the system: a bug there does not degrade play, it
manufactures false facts that persist. Gate the proof channel on **differential testing
against independent implementations** — move counts, check verdicts, terminal verdicts
over real game corpora — with every disagreement investigated before proofs are trusted,
and re-run on every generator change and every rule variant. Choose oracles for
independence, not authority: an official implementation known to be wrong in places is
not an oracle, and agreement with it proves nothing. And where legality is defined over a
composite action (a turn of several moves) rather than a single move, intermediate states
may legitimately violate the final constraint — terminal detection belongs at action
boundaries only.

**Oracle independence is the gate's real strength.** Every component reused from a
reference implementation forfeits that reference as an oracle for what it covers —
including a shared parser, which corrupts both sides of a comparison identically.
Reimplement the checked components and use the references for understanding and as
oracles — and let the oracles be the *only* way other implementations enter the
repository, confined to the test harness; everything else from them enters as ideas. Pre-include the oracles in the repository, pinned, behind a boundary the build
enforces: a gate left to discretion is skipped precisely when it is needed, and the path
of least resistance under pressure is copying an oracle's code into the core. Oracles can
be wrong too — record known divergences with the evidence that settles them, and fail on
any new one. A terminal verdict that rests on search — "no legal action exists" — is a proof
only if the search was exhaustive and terminated; an exhausted budget is "unknown", never
the verdict, which is the failure behind most false terminal claims. Four operating rules make the comparison mean something: **compare states, not
indices** — each implementation fingerprints the state it parsed, and a mismatch is a
parse divergence, triaged before any verdict at that step and settled against the pinned
format specification, as verdicts are against the rule source; **settle disagreements from
the rule source, never by vote**, clustered so each question is decided once; **require
exact reproduction on stable answers**, recording budget-limited answers as unknown and
excluding nondeterministic ones rather than tolerating a percentage; and **keep recorded
answers immutable**, hash-verified, outside the reach of the code under test. A bonus: a
recorded oracle error doubles as a copy detector — an implementation that reproduces it
probably borrowed that oracle's logic.

### 13.3 Separate the mechanism from the failure

Recurring confusion in this problem class: *does an architectural restriction cause
correctness failures?*

Usually not, if a **domain generator** handles correctness. If legality, reachability, or
validity is computed exactly outside the network, the network cannot cause an invalid
output — it can only misallocate attention or effort. Those are different problems with
different fixes.

But the corollary bites: **anything that removes candidates from the generator's input *can*
cause correctness failures.** Scope reduction (§3) is in this category; caching (§8) is not.
The optimisation that looks free is the one to verify hardest.

### 13.4 Percentages need denominators

"13% overhead" is meaningless without stating whether the denominator is the attention term
or total cost. Both are defensible; conflating them is how §9.3's trap operates.

### 13.5 Borrowed numbers need provenance

Figures imported from adjacent literature — "quantisation costs ~1%", "N registers is
standard", "GQA costs ~0.5% quality" — were measured on different problems with different
error tolerances. Carry the provenance with the number, and treat it as a prior rather than
a result.

### 13.6 Pricing a speed/quality trade needs two arms

Any modification that changes both throughput and quality — scope reduction, caching,
quantisation, a smaller model — cannot be evaluated with one experiment. Run two:
**equal wall-clock**, which answers the deployment question (does the speedup buy more
than the quality costs, at the budget that actually binds), and **equal internal work**
(equal evaluations, equal steps), which isolates the quality cost itself. One arm alone
is systematically misleading: equal wall-clock hides the quality cost inside the speed
gain, equal work hides the speed gain entirely. The trade is only priced when you hold
each side fixed once.

---

## 14. Summary

**The pattern in one paragraph.** Tokenise substructures uniformly. Let attention's inherent
weight sharing handle variable cardinality — connectivity is the mask, not the weights. Use
relative positional encoding on every lattice axis, because that is what generalises to
unseen cardinality. Add a small number of per-substructure registers and a generous number
of global ones. Inject any exact structural relation the domain provides as an attention
bias rather than learning it. Use pointer heads for variable output spaces, with FP32 on
anything that gets ranked. Measure realistic token counts before choosing dimensions,
because scope reduction dominates every other optimisation. Build dense and uncached first,
so you have something to price the alternatives against.

**The three things most likely to go wrong.** Positional encoding that does not generalise
to unseen `N`, which invalidates the whole design. Importing an LLM technique whose
assumptions do not hold — particularly anything that discards state based on learned
importance. And quoting attention-term ratios as total throughput, which flatters every
optimisation by 20-30%.

**The three things most likely to be over-engineered.** Caching before establishing that
throughput binds. KV-head reduction, which is retrofittable and usually not worth it.
Separate weight sets for masked passes that would work tied.

**The one thing most often mistaken for a kernel detail.** Token ordering (§5.3). It decides
whether block-sparse structural masks are exploitable or gather-bound, and it silently taxes
axial attention. Decide it with the architecture, not during optimisation.
