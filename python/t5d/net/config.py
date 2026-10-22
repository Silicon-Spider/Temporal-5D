"""Network configuration.

Defaults are the committed v1 network (register F1, spec section 9.2): d = 256, L = 16,
8 heads x d_head 32 (full MHA, C4), FFN 8x, 2 registers per board and 16 global (A1,
B8), a 51-bin categorical value head (D3), 4D RoPE with pairs file 3 / rank 3 /
timeline 4 / time 6 (B4), BF16 trunk with FP32 RoPE and heads (G1).

Stage N2 (the attention structure, spec section 6) adds the hard move-graph mask on 2
heads (B9, committed: on by default) and B14's attention paths (committed engineering):
``"block_sparse"`` (the default ``attn_impl``), whose masked heads read only the key
boards their query block can reach (board-major order, section 6.1), and ``"chunked"``,
the dense-mask arm of ablation 6. B14's FlexAttention score function (``"flex"``,
``flex.py``) runs eagerly on CPU - forward only, every score materialised: it proves the
in-kernel formulation exact on small positions; compiled kernels and their speed need a
GPU (G9, M19).

The build rules (spec section 15.1) decide what else appears here:

* **B15, the 4D displacement bias** (provisional, the v1 default; CHECKPOINT 6, section
  4.1; ``displacement.py``) is on: ``displacement_bias``, whose off position is ablation 2's
  no-table arm. Its unstated details are placeholders (below). It supersedes B6's binned
  Delta-time/Delta-timeline biases, which are removed (register B6, M20).
* **Provisional and prototype items** are behind flags, all default off: per-head
  positional specialisation (B5, provisional), the fixed routing table (B10, prototype),
  typed edge biases (B11, provisional), board-pair smolgen (B12, prototype), the key-side
  Shaw term on top of B15 (B17, prototype; ``shaw_key_term``, ``shaw.py``), and the larger
  capacity arm L24-ff12 (F2, provisional) as the named preset ``"L24-ff12"``.
* **Items of later stages** have their flag here so the config names every switch, but
  enabling one raises ``NotImplementedError`` until its stage lands: piece-centric
  tokenisation (A5), the history window (C5/C7, provisional; its code waits on the |E|
  verdict, M7) and FP8 attention (G2, provisional; no FP8 kernels on CPU).
* **Stage N3** adds the attention scope (``attn_scope``): ``"dense"`` (C1, the default -
  build dense first) or ``"block_causal"`` (C17 and C19, committed rows of a deferred
  configuration, C2 - built because section 15.2's equivalence suite and section 15.3's
  cost figures need it): causal between boards, bidirectional within a board, the live
  set (the heads) bidirectional among itself and reading all history, global registers
  reading everything and read only by live tokens (``scope.py``). The cached evaluation
  path that the scope makes possible (C2, C3, C3a rung 1, C15, C20) is ``cache.py``; its
  switches - the root refresh, the rung-1 node refresh and its visit gate, C24's
  complexity trigger (prototype, ``complexity.py``), the KV storage format - are the
  evaluator's (``CacheConfig``), not the network's. Four points the
  spec left open on this committed-class path were built as parameters (``causal_past`` -
  since ruled by C23: ancestry, the ``"lineage"`` default -, ``frozen_features`` - since ruled
  by M20: intrinsic, with B16's status biases -, ``supersession_scope`` - both readings built,
  6c decides (C20, M20) - and the rule that the section 6 structure opens live rows only,
  documented in ``scope.py``). **B16** (committed with the cached configuration, section 4.1;
  ``status.py``) gives frozen boards' current status to the live queries that read them as
  score-time key biases (``status_key_bias``, on), its bins, layer sharing and initialisation
  placeholders.
* **Stage N5** adds register D8 (committed): the global acting-player input, on the global
  registers (M20: "acting player -> global registers"; with none, on the board tokens - a
  placeholder, spec issue 225), and the submit action's policy logit, read from the pooled
  global registers (M20: "submit logit -> pooled global registers") by a readout whose form
  is not ruled (``submit_head``, a placeholder).
* **M20's placeholder dispositions** (committed, CHECKPOINT 6; register M20) are the
  defaults, and the arms they dispose of - no named ablation needing them - are gone: the
  registers' positions (B8: per-board registers rotated on timeline and time only, global
  registers position-free on every pair - no switch), the promotion offset in the policy
  (D1 - no switch), the auxiliary heads' pool (the squares; D3's value heads pool the global
  registers, by mean),
  moves-left in turns (D4), the acting player on the global registers and the pooled submit
  logit (D8). B5's details, the per-square label set, stage N2's placeholders (16a-g) and B7
  "stand as v1, tested by ablations 2, 6a and 6j" - the switches below stay, now the v1's
  values.
* **D3's value heads** (committed, corrected at CHECKPOINT 7, section 8.3; ``heads.ValueHeads``):
  WDL - the value search uses - the 51-bin categorical over the root search value, short-term
  values at ``short_term_horizons`` (placeholders in turns) and the error head, all FP32 from the
  pooled global registers; the targets' unstated details (HL-Gauss' sigma, the error head's
  horizon and loss) are the training code's placeholders (``train.losses.LossConfig``; spec
  issue 231).
* **D4's soft policy** (committed, CHECKPOINT 7, section 12.4) is on (``soft_policy``, 6j's
  on/off arm); **the opponent-reply head** is a prototype behind its flag, off (``opp_reply``);
  their widths and the other pointer heads' promotion offset are placeholders (spec issues 222,
  232).
* **F5** (committed, CHECKPOINT 6, section 9.3): every trunk norm RMSNorm-style - no
  mean-centring, no bias (``trunk.RMSNorm``) - no biases in the Q, K and V projections
  (``qkv_bias``), the attention-output and FFN biases kept (``out_ffn_bias``), pre-norm;
  ablation 6y's arm, Lc0's post-norm with DeepNet initialisation, is
  ``norm_placement="post"`` (the placement is provisional), with the arm's unstated
  details as placeholders.
* **F6** (committed, CHECKPOINTS 7-8, section 9.3): Mish in the trunk FFN
  (``ffn_activation``) and in every head's hidden layers (CHECKPOINT 8: the value heads' shared layer,
  ``heads.ValueHeads`` - no switch); ablation 6y's arm, squared ReLU, is
  ``ffn_activation="squared_relu"``, in the trunk only.
* **Where the spec is silent, ambiguous or contradictory on a committed-class point**
  the choice is a parameter and the default is marked as a placeholder, not a decision -
  or, where no second arm exists to switch to, a placeholder fixed in code and marked so:
  where D8's acting player enters a network with no global registers (an ablation-3 arm: the
  board tokens, the live boards' under the block-causal scope) and what the value-type heads
  and the submit logit pool there (the squares; spec issue 225, ``global_regs``), and the
  activations outside F6's words - the Delta-t embedding's MLP (GELU, ``embed.py``) and B12's
  smolgen (SiLU, ``smolgen.py``), input-side and committed-class like the rest of the input
  encoding (spec issue 221); D8's submit readout form (``submit_head``; spec issue 109 - not on M20's list), B15's far
  bins, line clamp, register entries and initialisation (``disp_far_edges``,
  ``disp_line_max``, ``disp_register_pairs``, ``disp_init_std``), B16's bins, layer sharing
  and initialisation (``status_dt_exact``, ``status_dt_max``, ``status_bias_per_layer``,
  ``status_bias_init_std``; spec issues 213-215), and B17's class set, head sharing and
  initialisation (``shaw_near``, ``shaw_shared_heads``, ``shaw_init_std``; spec issues 217,
  218, 220).
* **Stand as v1 (M20)**, each a switch whose default is now the v1's value, tested by the
  ablation that names it: B5's details (ablation 2) - ``rope_relations_head``,
  ``rope_same_square_split``, ``rope_near_nope_max_angle`` and ``rope_roles_source`` (how B5's
  per-head allocation meets B9's masks when the routing table is off); and stage N2's
  (issue 16a-g; register B9's "v1 as built", ablation 6a for the routing table's):
  ``move_graph_direction`` (which way a directed move edge opens the mask),
  ``masked_head_registers`` (what registers do on masked heads), ``same_board_registers``
  (what a per-board register reads on the same-board head), ``relations_head_types`` (which
  relations "attack/defend relations" covers), ``relations_head_direction`` (which way a
  typed edge opens the relations head) and how the routing table meets B9's head count
  (``move_graph_heads``, ``rope_head_roles``). The per-square label set
  (``heads.SQUARE_LABELS``) stands as v1 too (ablation 6j).
* Refused and held-open items have no switch at all (B3 axial, B13 token/within-board
  smolgen, C18 local/global interleave, C21 snapshot shortcuts, F3 d = 512, G4-G6;
  threat relations never become tokens, section 14.4).

Implementation choices where the spec is silent and the point is not committed-class
(initialisation scale, RoPE frequencies, head internals, the Delta-t embedding's MLP width and
input features - its activation is spec issue 221's, above) are documented beside their
fields and in the modules that use them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Named capacity presets (section 9.2's ladder). "F1" is the committed v1 network;
#: "L24-ff12" is F2's larger arm (provisional; section 9.2's ~44M with heads is the count
#: without B15's table - 46,903,315 as built with it, spec issue 210).
PRESETS: dict[str, dict] = {
    "F1": {},
    "L24-ff12": {"n_layers": 24, "ffn_ratio": 12},
}

#: Per-head positional roles for B5, in head order, following section 6's routing table
#: (head 1 legal move graph, 2 same board, 3 same square across boards, 4 attack/defend
#: relations, 5-8 dense).
ROUTING_TABLE_ROLES: tuple[str, ...] = (
    "move_graph",
    "same_board",
    "same_square",
    "relations",
    "dense",
    "dense",
    "dense",
    "dense",
)

_ROLES = {"move_graph", "same_board", "same_square", "relations", "dense"}

#: The typed relations of B11 (section 6): every relation type but the move graph itself -
#: "attack, defence, pin, x-ray, check". Travel-threat, once a sixth, is folded into attack
#: (CHECKPOINT 6, register B11, reversal 26: a travel threat is a cross-board attack, which the
#: attack type already carries); its C id stays reserved in the append-only ABI and is never
#: emitted. The engine builds attack, defence and x-ray; pin and check are its later work
#: (engine ABI round 1) - listed here as the register lists them, their edges empty until the
#: engine supplies them.
TYPED_RELATIONS: tuple[str, ...] = ("attack", "defence", "pin", "xray", "check")

#: Attention implementations (``attention.py``): "block_sparse" (B14, the default: the
#: fused kernel over board-aligned query blocks, each masked head reading only the key
#: boards its block can reach), "chunked" (the same blocks against every key - the dense-
#: mask arm of ablation 6), "sdpa" (N1's fused path in one block - materialises
#: (B, H, L, L) rows once a mask or a bias is present), "explicit" (materialised scores,
#: chunked over queries), "dense" (the reference: every mask and bias materialised at
#: (B, H, L, L) first - small inputs) and "flex" (B14's score function, ``flex.py``: torch's
#: FlexAttention with every bias and mask computed inside its score and mask functions -
#: eager on CPU, so forward only and every score materialised: small inputs).
ATTN_IMPLS: tuple[str, ...] = ("block_sparse", "chunked", "sdpa", "explicit", "dense", "flex")

#: B12's generator forms (``smolgen.py``).
SMOLGEN_FORMS: tuple[str, ...] = ("relative_cells", "board_bilinear")

#: Attention scopes (stage N3, ``scope.py``): C1's dense bidirectional scope, or C17/C19's
#: block-causal one.
ATTN_SCOPES: tuple[str, ...] = ("dense", "block_causal")
#: Readings of a board's causal past (spec issue 5, ``scope.py``).
CAUSAL_PASTS: tuple[str, ...] = ("lineage", "two_parent", "creation")
#: How frozen boards' K/V exist under the block-causal scope (``kvformat.py``).
FROZEN_KV_FORMATS: tuple[str, ...] = ("match", "fp8")
#: What a frozen board's state-dependent per-board features are under the block-causal
#: scope (``scope.py``).
FROZEN_FEATURES: tuple[str, ...] = ("intrinsic", "current")
#: Readings of C20's "causal past plus the conditioned history of its refresh epoch"
#: (``scope.py``).
SUPERSESSION_SCOPES: tuple[str, ...] = ("causal_past", "past_and_epoch_history")
#: How D8's submit logit is read from the pooled global registers (stage N5, M20;
#: ``heads.SubmitHead``).
SUBMIT_HEADS: tuple[str, ...] = ("linear", "pointer")
#: Where each sublayer's norm sits (F5; ``trunk.py``): "pre" (the v1) or "post" (ablation
#: 6y's arm: Lc0's post-norm with DeepNet initialisation).
NORM_PLACEMENTS: tuple[str, ...] = ("pre", "post")
#: The trunk FFN's activation (F6; ``trunk.FeedForward``): "mish" (the v1) or "squared_relu"
#: (ablation 6y's trunk-only arm). The heads' hidden layers are Mish under either.
FFN_ACTIVATIONS: tuple[str, ...] = ("mish", "squared_relu")
#: B15's entries for pairs with a per-board register (``NetConfig.disp_register_pairs``).
DISP_REGISTER_PAIRS: tuple[str, ...] = ("by_kind", "shared")


@dataclass(frozen=True)
class NetConfig:
    # --- F1: the committed v1 dimensions (section 9.2) ---
    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 8
    d_head: int = 32
    ffn_ratio: int = 8
    # --- A1 / B8: registers (sections 5.2, 5.3) ---
    #: Their positions have no switch (B8, ruled at CHECKPOINT 6 - M20; section 5.3): a
    #: per-board register is rotated on its board's timeline and time pairs only - every pair
    #: with one uses unrotated file/rank dims on both sides - and a global register on none -
    #: every pair with one is unrotated on both sides (``attention.py``'s pair rule; "a learned
    #: 'not a square' angle is a (file, rank) in disguise"). Both kinds add the learned
    #: additive "not a square" embedding (content, not position; ``embed.py``). Ablation 3
    #: varies the counts (registers on/off, 2 vs 4 per board), not the positions. With no
    #: global registers D8's acting player goes on the board tokens and the value-type heads and
    #: the submit logit pool the squares - placeholders (spec issue 225: no document says
    #: whether "off" removes the global registers, or where their inputs then go).
    regs_per_board: int = 2
    global_regs: int = 16
    # --- D3: categorical value head (section 8.3) ---
    value_bins: int = 51

    # --- B4: 4D RoPE (section 4.1) ---
    #: Rotary pairs per axis: file, rank, timeline, time. Must sum to d_head / 2.
    rope_pairs: tuple[int, int, int, int] = (3, 3, 4, 6)
    #: Coordinate range each axis's frequency ladder is sized to (implementation choice):
    #: the lowest frequency of an axis is pi / range, so offsets up to the range never
    #: alias. Files and ranks span 8; timelines are sized for 64 apart; time (half-turns)
    #: for 256 apart, beyond which the time axis rotates into untrained territory - the
    #: extrapolation risk B15's clamped far bins guard (B6's binned biases did, before it).
    rope_ranges: tuple[float, float, float, float] = (8.0, 8.0, 64.0, 256.0)
    #: Highest frequency of every axis (radians per unit offset), as in standard RoPE.
    rope_max_freq: float = 1.0
    #: Subtract a per-position reference (p_ref, h_present) from the timeline and time
    #: coordinates before computing angles. A pair rotated on both sides sees only the
    #: difference of its coordinates, so this changes no such score; it keeps the FP32
    #: angles small (G1). Off only to test the relative property of the rotation itself.
    rope_recenter: bool = True

    # --- G1: precision (section 10.2) ---
    #: "bf16": the trunk runs under CPU/GPU autocast to BF16 (matmuls and attention in
    #: BF16; FP32 master weights, residual stream and norms); "fp32": everything in FP32.
    #: RoPE angles and rotation, and every output head, are FP32 in both.
    trunk_dtype: str = "bf16"
    #: Attention implementation (``ATTN_IMPLS``; B14). "block_sparse" and "chunked" run
    #: PyTorch's scaled_dot_product_attention kernel over query blocks of
    #: ``attn_block_boards`` whole boards, with each block's masks and biases built from
    #: the sparse edge lists and board-pair tables - no (L, L) tensor per head exists,
    #: and under autograd none is kept for the backward pass either (``attn_recompute``).
    #: "block_sparse" hands each masked head's kernel call only the key boards the block's
    #: rows can reach (section 6.1); "chunked" scores every key and masks (ablation 6's
    #: dense-mask arm). "sdpa" is N1's fused path (one block), "explicit" materialised
    #: scores chunked over queries, "dense" the reference that materialises every mask and
    #: bias first, and "flex" B14's score function (``flex.py``): torch's FlexAttention with
    #: every bias and mask computed inside its score and mask functions - eager on CPU, so
    #: forward only there (under autograd it raises ``NotImplementedError``: not a training
    #: path on CPU) and every score materialised (small inputs). All six compute the same
    #: function on every valid query (the position-free pairs merged exactly - see
    #: attention.py); the tests compare them.
    attn_impl: str = "block_sparse"
    #: Query rows per chunk in the explicit path.
    attn_chunk: int = 1024
    #: Whole boards per query block in the block-sparse and chunked paths (implementation
    #: choice: 4 boards = 264 rows keeps a block's FP32 bias under 140 MB at 240 boards).
    attn_block_boards: int = 4
    #: Under autograd, the block-sparse and chunked paths recompute each query block in
    #: the backward pass (activation checkpointing, ``torch.utils.checkpoint``) instead of
    #: keeping its mask, bias and probabilities until then: the memory kept between the
    #: passes is then O(L) per head, not O(L^2), for one extra forward of the blocks.
    #: Implementation choice (False: faster, quadratic memory under autograd).
    attn_recompute: bool = True
    #: Block-sparse path: a query block whose key selection is more than this fraction of
    #: the kernel keys is computed against every key, masked, as in the chunked path -
    #: gathering nearly every key costs more than the few it skips save, and its plan would
    #: hold nearly a whole (rows, L) mask. Each block's plan is built once per forward and
    #: reused by every layer, so plans hold at most this fraction of the (L, L) bool mask
    #: per masked role - about 58 MB for the move graph at 240 boards, whose blocks select
    #: 24% of the key boards on average (implementation choice; 1.0 gathers every block).
    attn_sparse_max_fraction: float = 0.5

    # --- F5: normalisation and biases (committed; section 9.3, CHECKPOINT 6; trunk.py) ---
    #: Where each sublayer's norm sits (``NORM_PLACEMENTS``). Register F5: "pre-norm for v1,
    #: Lc0's post-norm with DeepNet initialisation as the arm" (ablation 6y; the placement is
    #: provisional, the RMSNorm-style norms are committed):
    #:   "pre"  - the v1: ``x + Attn(N(x))`` then ``x + FFN(N(x))``, and a final norm before
    #:            the heads;
    #:   "post" - 6y's arm, DeepNorm (Wang et al. 2022, "DeepNet: Scaling Transformers to
    #:            1,000 Layers"): ``x <- N(alpha x + Attn(x))`` then ``x <- N(alpha x + FFN(x))``,
    #:            alpha = ``deepnorm_alpha``, with DeepNet's initialisation (``deepnorm_beta``;
    #:            ``model._init_weights``). PLACEHOLDERS for what neither section 9.3 nor the
    #:            register states (spec issue 201) - DeepNet's published constants for an
    #:            L-layer stack, not Lc0's (whose code may not be consulted); the base of its
    #:            initialisation Xavier normal, as its pseudocode draws it; no final norm (the
    #:            last block's FFN norm is the output's, as in a post-norm encoder); no norm on
    #:            the embedding before block 0; board-pair smolgen reads the attention's input,
    #:            which is then the block's input.
    norm_placement: str = "pre"
    #: Every trunk norm is RMSNorm-style (F5: "no mean-centring, no bias"; ``trunk.RMSNorm``):
    #: ``x / sqrt(mean(x^2) + eps) * g`` with a learned gain ``g``. The gain is implied by
    #: "RMSNorm-style" and eps is not stated (spec issue 200): eps 1e-5, kept from stage N1.
    norm_eps: float = 1e-5
    #: Biases in the Q, K and V projections. F5 (committed): "no biases in the Q, K and V
    #: projections" (section 9.3). True is no arm of any ablation - 6y's arms are both
    #: without them; it exists so that the switch F5 turned off stays named.
    qkv_bias: bool = False
    #: Biases in the attention output projection and both FFN matrices. Section 9.3 keeps
    #: them: Chessformer's networks "also keep their FFN and attention-output biases, as
    #: here". Register F5 does not say so, and its "no bias" belongs to the norms (spec issue
    #: 200): True is that reading of section 9.3, not a new decision.
    out_ffn_bias: bool = True

    # --- F6: the activation (committed; section 9.3, CHECKPOINTS 7-8; trunk.py, heads.py) ---
    #: The trunk FFN's, ``FFN_ACTIVATIONS``. Register F6: "Mish for v1, in the trunk FFN and every
    #: head's hidden layers (CHECKPOINT 8) ...; squared ReLU as the trunk-only arm" (ablation 6y):
    #: "mish" - ``x tanh(softplus(x))`` (Misra 2019, "Mish: A Self Regularized Non-Monotonic
    #: Activation Function"); "squared_relu" - ``relu(x)^2`` (So et al. 2021, "Primer"), drawn at
    #: the same initialisation (the arm's is not stated - spec issue 204, proposed edit 154). Only the trunk's FFN
    #: follows this switch: the heads' hidden layers are Mish under both (``heads.ValueHead``).
    ffn_activation: str = "mish"

    # --- embeddings (A2, A3, B8, section 5.3) ---
    #: Hidden width of the small MLP that embeds Delta-t-to-present (its GELU: a placeholder
    #: fixed in code, spec issue 221 - ``embed.py``).
    dt_hidden: int = 32
    #: Delta-t is fed as (dt / dt_scale, sign(dt) log1p|dt|, [dt == 0]).
    dt_scale: float = 16.0
    #: Register A6's global turn-state counts (section 2.3: mandatory boards still to move,
    #: optional boards still playable, moves already made this turn; ``embed.py``): each count
    #: ``n`` is fed as ``(n / turn_state_scale, log1p n, [n == 0])``, the nine features through
    #: a small MLP of this hidden width (GELU, as Delta-t's) to ``d_model``, added to every global
    #: register. The spec says only "embedded into the global registers": the features, the
    #: scale and the width are a PLACEHOLDER of a committed-class silence, fixed in code (spec
    #: issue 230).
    turn_state_hidden: int = 32
    turn_state_scale: float = 8.0

    # --- heads (sections 8.1, 8.3, 12.4) ---
    #: Width of the policy pointer projections f and g (D1). Promotions add a learned
    #: per-piece logit offset read from the destination token (D1, section 8.1, M20:
    #: "promotion scoring -> learned per-piece offset" - no switch; ``heads.PointerHead``).
    policy_dim: int = 256
    #: D4's soft policy (CHECKPOINT 7, section 12.4): "a second pointer head - its own source and
    #: destination projections and its own submit logit" (``heads.ActionHead``), on - its off arm is
    #: 6j's "soft policy on/off". Its projections' width: COMMITTED-CLASS SILENCE (D4; spec issue
    #: 232) - placeholder default, not a decision: 128, half the policy's (section 9.2's "~0.1M"
    #: auxiliary budget).
    soft_policy: bool = True
    soft_policy_dim: int = 128
    #: D4's opponent-reply head - a PROTOTYPE (reinstated at CHECKPOINT 7: "weight 0.15, the
    #: opponent's next turn mapped onto current squares, ambiguous mappings masked"), behind its
    #: flag, off: its move space is ``PositionState.opp_moves`` as ``train.reply_map`` builds it.
    opp_reply: bool = False
    #: Width of the opponent-reply pointer projections (aux, ~0.1M budget of section 9.2).
    opp_policy_dim: int = 128
    #: Whether the pointer heads other than the policy (the soft policy, the reply, D9's
    #: optimistic policy) carry D1's per-piece promotion offset. COMMITTED-CLASS SILENCE (D1 names
    #: the policy; spec issue 222) - placeholder default: yes, since the bilinear form ties their
    #: promotions as it ties the policy's.
    pointer_promotion_offset: bool = True
    #: Hidden width of D3's value heads' shared layer (its activation is F6's Mish -
    #: ``heads.ValueHeads``: WDL, the categorical, the short-term values and the error head).
    #: Where each head reads has no switch (M20; ``model.T5DNet.heads``): D3's value heads the
    #: mean of the global registers (section 8.3: "each pooled by mean from the global
    #: registers"); the global auxiliary heads - moves-left, search effort - the mean of every
    #: square token (M20: "aux pooling -> squares"; section 12.4, B8, D7: registers stay free
    #: scratch); per-board heads each board's squares, per-square heads every token.
    value_hidden: int = 256
    #: D3's short-term value heads (section 8.3: "short-term value at two or three horizons -
    #: exponentially averaged future search values ... the horizons are placeholders scaled to 5D
    #: game length until the pilot run measures it"): one output per horizon, its mean horizon in
    #: turns - one player's action each (issue 305's unit, as moves-left's). COMMITTED-CLASS
    #: SILENCE (D3; spec issue 231) - placeholder default, not a decision: three horizons,
    #: KataGo's 6, 16 and 50 turns in Go scaled by about a third (a 5D game's length is not
    #: measured: register K, M18). Each output is a scalar on [-1, 1] (tanh), trained by squared
    #: error (``train.losses``) - the form a placeholder fixed in code (issue 231).
    short_term_horizons: tuple[float, ...] = (2.0, 5.0, 16.0)
    #: Per-square deep-attack-fact labels (``heads.SQUARE_LABELS``: the set stands as v1 - M20,
    #: ablation 6j).
    n_square_labels: int = 8
    #: Search-effort bins (section 8.6): log2-visit bins plus a censored top bin.
    effort_bins: int = 16

    # --- D8: the acting player and the submit action (stage N5; sections 2.3, 8.2) ---
    #: Turns have variable length (Q4, rulebook R8): once the present has shifted inside an
    #: action the colour to move on the present is the opponent's while the action is still
    #: the mover's, so per-board side-to-move flags cannot say whose action it is. Register
    #: D8: "a global acting-player input" - on the global registers (M20: "acting player ->
    #: global registers"; section 2.3's global turn-state inputs): a learned embedding (white /
    #: black) added to every global register token (``embed.py``), no switch. Global registers
    #: are live, so the cached path stays exact with no change to a frozen board's K/V. With
    #: no global registers (``global_regs = 0``, an ablation-3 arm) it is added to every board
    #: token instead - under the block-causal scope the live boards' only, so a frozen board's
    #: K/V still never depend on it. COMMITTED-CLASS SILENCE (D8; section 2.3 puts it "into the
    #: global registers" and says nothing of a network without them; spec issue 225) - a
    #: placeholder, fixed in code: dropping the input with the registers is no reading of D8.
    #:
    #: Under the board-move split submitting is "an explicit action with its own policy logit,
    #: read from the pooled global registers as the value head is" (section 8.2; M20: "submit
    #: logit -> pooled global registers", so no register is assigned a role and D7 holds): the
    #: mean of the global registers' final states (with no global registers - an ablation-3
    #: arm - the mean of the square tokens, as the value head falls back: a placeholder, spec
    #: issue 225). How the logit is
    #: read from it is not ruled (M20 leaves it out; spec issue 109). COMMITTED-CLASS SILENCE
    #: (D8) - placeholder default, not a decision. ``submit_head``:
    #:   "linear"  - its own learned linear readout of the pooled state (``heads.SubmitHead``):
    #:               d + 1 parameters, on the move logits' scale at initialisation;
    #:   "pointer" - the policy's own bilinear form with the pooled state at both ends,
    #:               ``f(r)^T g(r) / sqrt(d_policy)``: no parameter of its own, one scoring
    #:               function for moves and submit.
    #: The logit is offered where submitting is possible (the obligations met, T5:
    #: ``PositionState.submit_possible``) and masked to -inf unless it is legal (C4:
    #: ``submit_legal``); the policy distribution is then over the moves and submit together.
    submit_head: str = "linear"

    # --- initialisation (section 15.2: realistic scale, never zeros) ---
    #: Linear weights: normal with std 1/sqrt(fan_in) (LeCun). Under pre-norm the attention
    #: output and FFN output projections are further scaled by 1/sqrt(2 * n_layers) (GPT-2
    #: style); under post-norm the trunk's attention and FFN projections are drawn by DeepNet's
    #: initialisation instead - Xavier normal, std gain * sqrt(2 / (fan_in + fan_out)), at gain
    #: ``deepnorm_beta`` on the value, attention-output and both FFN projections and gain 1 on
    #: the query and key projections (``model._init_weights``).
    init_embed_std: float = 0.5
    #: Linear biases.
    init_bias_std: float = 0.02
    #: RMSNorm gains are 1 + N(0, init_norm_std): not the identity, so that a path that
    #: dropped or misplaced a norm's gain could not agree with one that kept it.
    init_norm_std: float = 0.1
    #: B11's typed-edge scalars (B15's table has its own, ``disp_init_std``).
    init_table_std: float = 0.1

    # --- B5: per-head positional specialisation (provisional; default off) ---
    rope_per_head: bool = False
    #: The head-role table, in head order (section 6's routing table by default). One
    #: table serves two switches so that they cannot disagree: under B5 it sets each
    #: head's rotary allocation (``rope.head_allocation``), under B10 (``routing_table``)
    #: each head's attention mask (``structure.py``).
    rope_head_roles: tuple[str, ...] = ROUTING_TABLE_ROLES
    #: Where B5 takes each head's rotary role from. Section 4.1 specialises heads
    #: "matching section 6's routing table", and register B5 (provisional) presupposes B10
    #: (prototype); with the routing table off, the move-graph masks of B9 sit on heads
    #: 0..k-1 whatever the table says. B5's as-built details are "v1 placeholders" (register
    #: B5) that STAND AS V1 (M20; spec issue 16f), tested by ablation 2:
    #:   "table" - roles from ``rope_head_roles`` by head index, whatever the masks (N1's
    #:             behaviour, the v1's: with B9's default, head 1 is move-graph masked and
    #:             rotates on file and rank only);
    #:   "masks" - each head's role is the role of the mask it carries: B9's move-graph
    #:             heads near-NoPE ("its structure comes from the mask"), every unmasked
    #:             head the dense B4 split.
    #: Under the routing table the two coincide (the masks are the table).
    rope_roles_source: str = "table"
    #: Allocation of the attack/defend-relations head (routing table head 4), which
    #: section 4.1 does not mention: "dense" (the B4 split; the v1's - B5's details stand as
    #: v1, M20, ablation 2) or "near_nope" (like the move-graph head).
    rope_relations_head: str = "dense"
    #: Same-square-across-boards head: rotary pairs on (timeline, time). B4's 4:6 ratio over 16
    #: pairs - B5's detail, standing as v1 (M20, ablation 2).
    rope_same_square_split: tuple[int, int] = (6, 10)
    #: Near-NoPE heads rotate every pair at a frequency giving at most this angle over its
    #: axis's whole range - B5's detail, standing as v1 (M20, ablation 2).
    rope_near_nope_max_angle: float = math.pi / 8

    # --- B15: the 4D displacement bias (provisional, the v1 default; section 4.1; displacement.py) ---
    #: One learned scalar per head per layer, indexed by the pair's displacement (Delta-file,
    #: Delta-rank, Delta-timeline, Delta-time in half-turns): a 15 x 15 file-rank slice per
    #: board-distance class (exact up to two timelines and two turns apart, binned and clamped
    #: beyond), exact line classes beyond the exact ones, register and global entries; beside
    #: RoPE, which stays (CHECKPOINT 6; it supersedes B6, removed). ``False`` is ablation 2's
    #: no-table arm: no table and no bias - every other weight as in the v1 (the table draws
    #: from a generator of its own).
    displacement_bias: bool = True
    #: Lower edges of the far bins of the board distance ``D = max(|Delta-timeline|,
    #: ceil(|Delta-time| / 2))`` (in timelines and whole turns) beyond the exact classes
    #: (``D <= 2``), each crossed with the sign of Delta-time; the last bin is clamped (it
    #: holds every larger distance). Section 4.1 says "binned and clamped, so unseen board
    #: counts extrapolate as the binned biases did" and gives no bins.
    #: COMMITTED-CLASS SILENCE (B15 is class C; spec issue 205) - placeholder, not a decision:
    #: log2-spaced from 3 - bins [3, 4], [5, 8], [9, inf) - x 3 signs = 9 far classes, which
    #: with the 45 exact ones gives section 4.1's ~1.6M parameters. The first edge is 3. Only
    #: the edges are a setting: the form - one distance D, crossed with Delta-time's sign,
    #: Delta-timeline's sign dropped - is fixed in code (``displacement.board_class``), and a
    #: per-axis or timeline-signed form would be a code change. Under this form B6 is not
    #: B15's special case, as section 4.1 says it is: no far class keeps B6's separate
    #: Delta-time and Delta-timeline buckets (issue 205, put to the user).
    disp_far_edges: tuple[int, ...] = (3, 5, 9)
    #: A line along the board axes only (timelines and turns, the square fixed) beyond the
    #: exact classes has an entry per direction and step k = 3 .. ``disp_line_max``, the last
    #: shared by every longer line (lines that also move on the board stop at k = 7).
    #: COMMITTED-CLASS SILENCE (spec issue 206) - placeholder, not a decision: 8. So are the
    #: line set's other details, which are fixed in code, not settings: a line counts whole
    #: turns (an odd Delta-time is never one), and its entry replaces the far slice's for the
    #: pair (one scalar per pair; adding the two instead would be a second lookup per pair).
    disp_line_max: int = 8
    #: Section 4.1: "per-board registers by their board-distance class". Whether the pair's
    #: kind enters is not said. COMMITTED-CLASS SILENCE (spec issue 207) - placeholder default,
    #: not a decision:
    #:   "by_kind" - an entry per class for each of square reads register, register reads
    #:               square and register reads register (3 x 54 = 162 per head per layer);
    #:   "shared"  - one entry per class for every pair with a per-board register (54).
    #: Register slots are not told apart, and a register's (file, rank) never enters (B8:
    #: position-free on file and rank). Global registers: one entry per head per layer for
    #: every pair with one, as section 4.1 says.
    disp_register_pairs: str = "by_kind"
    #: The table's initialisation, ``N(0, disp_init_std)`` from B15's own generator (section
    #: 15.2: realistic scale, never zeros). COMMITTED-CLASS SILENCE (spec issue 208) -
    #: placeholder, not a decision: 0.1, the scale of the other learned bias tables (B11).
    disp_init_std: float = 0.1

    # --- B17: the key-side Shaw term (prototype; default off; section 4.1; shaw.py) ---
    #: Each query (unrotated) dotted with a learned vector for its pair's class in a compact set,
    #: scaled by 1/sqrt(d_head) and added to the score - Shaw et al.'s key-side relative
    #: representation, "on top of B15" (it needs ``displacement_bias``). Ablation 2's arm, behind
    #: its flag; value-side vectors are held open (section 14.4) and not built.
    shaw_key_term: bool = False
    #: "A compact class set (~1,400: the exact near classes plus the line classes)" - section 4.1
    #: defines neither. COMMITTED-CLASS SILENCE (B17 is class C; spec issue 217) - placeholder, not
    #: a decision: inside B15's 45 exact board-distance classes, the file-rank offsets with
    #: |Delta-file|, |Delta-rank| <= ``shaw_near`` (2: 1,125 classes), plus B15's line classes
    #: (368): 1,493, which gives both of section 4.1's cost figures. Pairs with a register get none.
    #: The nearest board-distance classes with their full slices would give the same count and
    #: costs and cover other pairs (issue 217, put to the user) - a code change
    #: (``shaw.compact_classes``), not this setting.
    shaw_near: int = 2
    #: A vector per class per head per layer (False), or one per class per layer shared by the
    #: heads (True, as Shaw et al. share them). COMMITTED-CLASS SILENCE (spec issue 218) -
    #: placeholder: per head - 6,115,328 parameters at the v1's dimensions (764,416 shared).
    shaw_shared_heads: bool = False
    #: The vectors' initialisation, ``N(0, shaw_init_std)`` from B17's own generator.
    #: COMMITTED-CLASS SILENCE (spec issue 220) - placeholder: 0.1 - a query's entries have unit
    #: spread at initialisation, so the term starts with the spread of B15's scalars.
    shaw_init_std: float = 0.1

    # --- B9: the move-graph mask (committed; section 6) ---
    #: Heads 0..k-1 carry the hard move-graph mask, the rest are dense ("2 heads with hard
    #: move-graph masks, remainder dense"; ablation 4 runs 0 / 2 / 4).
    #: Under ``routing_table`` the table assigns every head and this count is not used:
    #: section 6's table has one move-graph head (head 1) beside the attack/defend relations
    #: head (head 4), and the table is built as written (``rope_head_roles`` can give it two).
    #: STANDS AS V1 (M20, spec issue 16g; register B9: "v1 as built: ... the relations head
    #: counts toward the two"), tested by ablation 6a with the table. ``validate`` refuses any
    #: count but the default under the routing table, so that an ablation-4 arm cannot run
    #: with the table's single move-graph head without saying so.
    move_graph_heads: int = 2
    #: Which way a directed edge ``(src, dst, relation)`` opens a masked head - the move
    #: graph (B9), and the relations head (B10) unless ``relations_head_direction`` says
    #: otherwise. Section 6 says "which squares reach which" and gives no direction.
    #: STANDS AS V1 (M20, spec issue 16a; register B9: "v1 as built: symmetric"):
    #:   "forward"   - the query is the edge's source: a piece's square reads the squares
    #:                 it reaches;
    #:   "reverse"   - the query is the edge's destination: a square reads the squares
    #:                 whose pieces reach it ("which pieces control it", section 5.1);
    #:   "symmetric" - both (the symmetric closure; each other arm is a restriction of it) -
    #:                 the v1's.
    move_graph_direction: str = "symmetric"
    #: What a register query does on a masked head (any head whose role is not "dense")
    #: where the head's relation does not cover it. Registers are never edge endpoints
    #: (section 5.3) and have no (file, rank), so the move-graph, relations and
    #: same-square relations cover no register; global registers hold no board (section
    #: 5.2), so no relation covers them at all. (A per-board register is on its board: the
    #: same-board head's relation can cover it - ``same_board_registers``.) A query's own
    #: token is always a key, so no valid token's row is ever empty.
    #: STANDS AS V1 (M20, spec issue 16b; section 6 is silent):
    #:   "read" - such a register reads every valid key, as on a dense head (for global
    #:            registers section 7.3's "reading is free, being read is not") - the v1's;
    #:   "self" - it reads only itself (the literal hard mask for every token).
    #: Either way a square never reads a register on a masked head, except its own
    #: board's per-board registers on the same-board head (they are on its board).
    masked_head_registers: str = "read"
    #: What a per-board register reads on the same-board head (B10). Section 6's head 2 is
    #: "Same board only", and a per-board register is on its board - "appended to each
    #: board" (section 5), it "belong[s] to [its] board's block" (section 7.3) - so that
    #: relation covers it as a query. STANDS AS V1 (M20, spec issue 16c; section 6 does not
    #: mention registers), tested by ablation 6a with the table:
    #:   "own_board" - it reads its own board's tokens, squares and per-board registers,
    #:                 as the board's squares do (the literal "same board only") - the v1's;
    #:   "registers" - it follows ``masked_head_registers``, as on every other masked head
    #:                 (stage N2's first build).
    same_board_registers: str = "own_board"

    # --- B10: the fixed routing table (prototype; default off; section 6) ---
    #: Heads take the roles of ``rope_head_roles`` (section 6's table by default: head 1
    #: legal move graph, 2 same board only, 3 same (file, rank) across boards, 4
    #: attack/defend relations, 5-8 dense - 0-based in code).
    routing_table: bool = False
    #: The relation types whose edges open the relations head ("attack/defend relations").
    #: STANDS AS V1 (M20, spec issue 16d), tested by ablation 6a with the table: every typed
    #: relation of B11 (section 6: heads 1 and 4 "receive binary masks: connected or not",
    #: which B11's types upgrade) - five since travel-threat was folded into attack
    #: (CHECKPOINT 6); the narrow reading is ("attack", "defence").
    relations_head_types: tuple[str, ...] = TYPED_RELATIONS
    #: Which way a typed edge opens the relations head: "forward", "reverse" or
    #: "symmetric", as ``move_graph_direction`` defines them; None - the v1's - takes
    #: ``move_graph_direction``'s value. STANDS AS V1 (M20, spec issue 16e; section 6 gives no
    #: direction for "attack/defend relations" either), tested by ablation 6a with the table.
    relations_head_direction: str | None = None

    # --- B11: typed edge biases (provisional; default off; section 6) ---
    #: A learned scalar per head per relation type (``TYPED_RELATIONS``: five), shared by all
    #: layers (section 6: "one learned scalar per head per type, shared across layers"), added
    #: to the logits of the related token pairs, cross-board pairs included - 8 x 5 = 40
    #: parameters (80 directed). Under the block-causal scope they bias live query rows only.
    typed_edge_bias: bool = False
    #: False: one scalar per head per type, added to both orientations of each related
    #: pair (the literal "a learned scalar per head per relation type"); True: two - one
    #: where the query is the relation's source, one where it is its target (implementation
    #: choice either way: B11 is class R).
    typed_edge_directed: bool = False

    # --- B12: board-pair smolgen (prototype; default off; sections 6, 9.2) ---
    #: Context-dependent biases over board pairs generated from the global registers
    #: (``smolgen.py``): latent 32, bilinear per head (section 9.2).
    board_pair_smolgen: bool = False
    smolgen_latent: int = 32
    #: The generator (``SMOLGEN_FORMS``; implementation choice - B12 is class R):
    #:   "relative_cells" - per head, the latent read through a learned table over
    #:                      relative board-pair cells (log-scale Delta-time / Delta-timeline
    #:                      buckets, ``smolgen_time_*`` / ``smolgen_timeline_*``): generated
    #:                      from the global registers alone and full-rank over board pairs
    #:                      (section 6, register B12);
    #:   "board_bilinear" - per head, a bilinear form between board descriptors taken
    #:                      from each board's per-board registers, modulated by the latent:
    #:                      rank <= ``smolgen_latent`` - the low-rank form section 6
    #:                      equates with extra Q/K dims (stage N2's first build).
    smolgen_form: str = "relative_cells"
    #: The relative cells' buckets (``buckets.py``; they were B6's, and B12's register row
    #: records smolgen "as built over relative board-pair cells"): offsets below ``*_exact``
    #: get their own bucket; beyond, log2-spaced buckets up to ``*_max``, then clamped (the
    #: last bucket is shared by every larger offset). Delta-time is signed (past and future
    #: differ); Delta-timeline symmetric (branching order makes its sign semi-arbitrary,
    #: section 4.4). 19 x 8 = 152 cells.
    smolgen_time_exact: int = 4
    smolgen_time_max: int = 128
    smolgen_timeline_exact: int = 4
    smolgen_timeline_max: int = 32

    # --- stage N3: the attention scope (C1, C17, C19; scope.py) ---
    #: "dense" (C1: build dense first - every token reads every token, section 6's masks
    #: aside) or "block_causal" (C17, C19): causal between boards, bidirectional within a
    #: board; a superseded (frozen) board's tokens - its per-board registers included -
    #: read their own board and the frozen boards of its causal past, never a live head
    #: and never a global register ("being read by frozen tokens = present-to-frozen flow,
    #: allowed only at refreshes"); a head board's tokens (the live set) read every board
    #: and the global registers; the global registers read everything. The cached path
    #: (C2, ``cache.py``) evaluates exactly this function incrementally. Committed-class
    #: (section 7.7: a net trained dense cannot be masked causally at inference).
    attn_scope: str = "dense"
    #: Which boards are a board's causal past (spec issue 5: C17 / section 7.3's "their own
    #: past" is undefined in a multiverse). RULED: register C23 (committed) makes it the
    #: board's ancestry - "its own timeline's earlier boards and, recursively, its branch
    #: parent up to the branch point" - which is ``"lineage"``, the default; the other two
    #: readings stay as arms of the equivalence tests, not alternatives to the decision
    #: (stage N5 checks the engine converter's lineage against C23 on real games):
    #:   "lineage"    - arrival-side lineage: the boards it was made from by content - its
    #:                  timeline's earlier boards and, through its branch origin (the board a
    #:                  travelling piece arrived on), the origin's own lineage, recursively
    #:                  (``PositionState.parent``). A travel arrival's departure board is not
    #:                  in it. Fixed when the board is created and made of superseded boards
    #:                  only, so a frozen board's K/V never need recomputing; state-defined
    #:                  given the branch origins - which 5DFEN does not record and the 5D
    #:                  Zobrist key (H18) omits, so an MCGS node key would have to add them;
    #:   "two_parent" - the same with the departure board as a second parent: a board a
    #:                  travelling piece arrived on also descends from the board the piece
    #:                  left (``PositionState.source``) and that board's lineage. Also fixed
    #:                  at creation and made of superseded boards (the jump supersedes its
    #:                  departure board), and also state-defined given origins and sources;
    #:   "creation"   - every board created before it (``PositionState.creation``). Not a
    #:                  function of the multiverse: two move orders create its boards in
    #:                  different orders (section 8.5's transpositions then evaluate
    #:                  differently), so no MCGS node key can serve it; and an earlier board
    #:                  that is still a head is not readable until it is superseded, which
    #:                  then changes the past of every frozen board created after it - the
    #:                  cache recomputes them all (measured: a cascade of hundreds of boards).
    #: (The time axis is not built: a later branch into the past adds boards at earlier
    #: times, so no frozen K/V could stay frozen.)
    causal_past: str = "lineage"
    #: What a frozen board's state-dependent per-board features are under the block-causal
    #: scope: Delta-t-to-present (A3) changes whenever the present moves and the timeline's
    #: active flag when the timeline is activated, after the board is frozen. RULED: M20
    #: ("frozen features -> intrinsic, with score-time status biases (B16)"; spec issue 31),
    #: with register A3 and B16 and sections 2.3, 4.1 and 7.3 - "intrinsic", the default:
    #:   "intrinsic" - a frozen board's tokens get neither (only its contents, side to move
    #:                 and the superseded flag): knowing where the present is would be
    #:                 present-to-frozen flow, which C17 allows only at refreshes (the root
    #:                 refresh, a dense pass, gives every board its features as of the
    #:                 root - kept as built, M20). Its current status reaches the live
    #:                 queries that read it as B16's score-time key biases
    #:                 (``status_key_bias``);
    #:   "current"   - a test arm only, no reading of the documents any more: a frozen board
    #:                 carries the evaluated state's values (A3 literally), so every shift of
    #:                 the present changes every frozen board's input and the cache recomputes
    #:                 them all - the cost section 7.1 does not price. No B16 under it.
    #: Live boards always carry both features.
    frozen_features: str = "intrinsic"
    #: B16 (committed with the cached configuration; sections 4.1, 7.3; register B16, A3, M20;
    #: ``status.py``): under the block-causal scope with ``frozen_features="intrinsic"`` a frozen
    #: board's current status - "the key board's Delta-t-to-present bin and timeline status" -
    #: enters as a learned key bias per head, recomputed every evaluation, on the scores of
    #: live query rows (head boards' tokens, global registers) against the tokens of frozen
    #: boards; frozen rows never get it (their K/V would go stale - C17). Built only for a
    #: network of that configuration (``status_bias_built``): inert, with no parameter, under
    #: the dense scope - the v1 - and under ``"current"``. A forward that overrides a dense-scope
    #: network's scope to the block-causal one is refused while this is on with intrinsic frozen
    #: features (``T5DNet.make_context``): its frozen boards would lose their status with no
    #: table to give it back. On by default; ``False`` is the network without it (every other
    #: weight the same: its table draws from a generator of its own) - no named ablation has that
    #: arm (gate 6c's are the refresh's; spec issue 213).
    status_key_bias: bool = True
    #: B16's Delta-t-to-present bins (``buckets.signed_log_bucket`` of ``h - h_present`` in
    #: half-turns): exact below ``status_dt_exact``, log2-spaced up to ``status_dt_max``, then
    #: clamped, past and future apart. Section 4.1 says "Delta-t-to-present bin" and gives no
    #: bins. COMMITTED-CLASS SILENCE (B16 is class C; spec issue 213) - placeholder, not a
    #: decision: 4 and 128, the time buckets B6 used and smolgen keeps - 19 bins. The timeline
    #: status is read as A3's active/inactive flag (head/superseded is constant on frozen
    #: boards): 19 x 2 = 38 statuses per head.
    status_dt_exact: int = 4
    status_dt_max: int = 128
    #: One table per layer, or one shared by all layers. Section 4.1 says "per head" only.
    #: COMMITTED-CLASS SILENCE (spec issue 214) - placeholder, not a decision: shared, as B11's
    #: scalars are (8 x 38 = 304 parameters; 4,864 with one table per layer, 16 x 304).
    status_bias_per_layer: bool = False
    #: The table's initialisation, ``N(0, status_bias_init_std)`` from B16's own generator
    #: (section 15.2: realistic scale, never zeros). COMMITTED-CLASS SILENCE (spec issue 215) -
    #: placeholder, not a decision: 0.1, the scale of B11's and B15's tables.
    status_bias_init_std: float = 0.1
    #: C20 (committed): a board superseded inside the search is recomputed "under the
    #: state-defined mask - its causal past plus the conditioned history of its refresh
    #: epoch". BOTH READINGS BUILT (C20, M20: "supersession scope -> both built, 6c decides"):
    #: 6c decides with the cache; "causal_past" is the default arm until then:
    #:   "causal_past"            - it reads its causal past only, the past's history
    #:                              boards carrying their epoch's (refreshed) K/V - so
    #:                              with no refresh the mask is the state's block-causal
    #:                              mask, independent of the search root;
    #:   "past_and_epoch_history" - it reads its causal past and every board of its
    #:                              epoch's history (the boards frozen at the epoch's
    #:                              anchor - the root, or the node that forked it), which
    #:                              makes the mask depend on the root even without a
    #:                              refresh ("a function of the state and the root",
    #:                              section 7.3).
    supersession_scope: str = "causal_past"
    #: How a frozen board's keys and values exist under the block-causal scope (C15 against
    #: G1; ``kvformat.py``): "match" (default) - the trunk's own precision, so a cache that
    #: stores them changes nothing (G1); "fp8" - every read of a frozen board's K/V goes
    #: through FP8 e4m3 with per-board, per-layer, per-head scales, in the ordinary forward
    #: as in the cache: the function a network must be trained on to be served from an FP8
    #: cache. PROTOTYPE FLAG of a committed-class point (it changes the function trained):
    #: C15 adopts FP8 K/V as "obeying section 10.2", which holds for an FP8 trunk (G2, not
    #: built), not for a BF16 trunk read through an FP8 cache it was not trained on.
    frozen_kv_format: str = "match"

    # --- later stages: named here, built there (enabling raises NotImplementedError) ---
    piece_centric: bool = False  # A5 (prototype, committed-class): not built in N1-N3
    history_window: bool = False  # C5/C7 (provisional): parked until the |E| verdict (M7)
    fp8_attention: bool = False  # G2 (provisional): no FP8 kernels on CPU

    # -- construction --

    @classmethod
    def preset(cls, name: str, **overrides) -> NetConfig:
        """A named preset: "F1" (default) or "L24-ff12" (F2's larger arm)."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    # -- derived --

    @property
    def d_ffn(self) -> int:
        return self.d_model * self.ffn_ratio

    @property
    def n_pairs(self) -> int:
        return self.d_head // 2

    @property
    def deepnorm_alpha(self) -> float:
        """DeepNorm's residual up-scaling under ``norm_placement="post"``: ``(2L)^(1/4)`` for an
        L-layer stack (Wang et al. 2022, Figure 2's table: encoder-only ``(2N)^(1/4)``,
        decoder-only ``(2M)^(1/4)`` - the same for one stack). A constant, not learned."""
        return (2 * self.n_layers) ** 0.25

    @property
    def deepnorm_beta(self) -> float:
        """DeepNet's initialisation down-scaling under ``norm_placement="post"``: ``(8L)^(-1/4)``
        (Wang et al. 2022, Figure 2), the gain of the Xavier normal the value, attention-output
        and FFN projections are drawn from; the query and key projections' gain is 1."""
        return (8 * self.n_layers) ** -0.25

    @property
    def head_roles(self) -> tuple[str, ...]:
        """Each head's attention role: the mask it carries ("dense" for none).

        Routing table on (B10): ``rope_head_roles``. Off: B9's ``move_graph_heads``
        move-graph heads first, the rest dense.
        """
        if self.routing_table:
            return tuple(self.rope_head_roles)
        k = self.move_graph_heads
        return ("move_graph",) * k + ("dense",) * (self.n_heads - k)

    @property
    def status_bias_built(self) -> bool:
        """Whether the network has B16's status key biases (``status_key_bias``): only under the
        block-causal scope with intrinsic frozen features - the cached configuration B16 is
        committed with."""
        return self.status_key_bias and self.attn_scope == "block_causal" and self.frozen_features == "intrinsic"

    @property
    def relations_direction(self) -> str:
        """The direction the relations head's edges open it (``relations_head_direction``)."""
        return self.move_graph_direction if self.relations_head_direction is None else self.relations_head_direction

    @property
    def rope_roles(self) -> tuple[str, ...]:
        """Each head's rotary role under B5 (``rope_roles_source``); ("dense",) without B5."""
        if not self.rope_per_head:
            return ("dense",)
        if self.rope_roles_source == "masks":
            return self.head_roles
        return tuple(self.rope_head_roles)

    def validate(self) -> None:
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model (full MHA, C4)")
        if self.d_head % 2:
            raise ValueError("d_head must be even (RoPE rotates pairs)")
        if sum(self.rope_pairs) != self.n_pairs or len(self.rope_pairs) != 4:
            raise ValueError(f"rope_pairs must be 4 entries summing to d_head / 2 = {self.n_pairs}")
        if any(r <= 0 for r in self.rope_ranges) or len(self.rope_ranges) != 4:
            raise ValueError("rope_ranges must be 4 positive numbers")
        if self.trunk_dtype not in ("bf16", "fp32"):
            raise ValueError("trunk_dtype must be 'bf16' or 'fp32'")
        if not self.norm_eps > 0:
            raise ValueError("norm_eps must be positive")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}")
        if self.attn_chunk < 1 or self.attn_block_boards < 1:
            raise ValueError("attn_chunk and attn_block_boards must be positive")
        if self.dt_hidden < 1 or self.turn_state_hidden < 1 or not self.dt_scale > 0 or not self.turn_state_scale > 0:
            raise ValueError("dt_hidden, turn_state_hidden, dt_scale and turn_state_scale must be positive")
        if self.value_bins < 2 or self.effort_bins < 2:
            raise ValueError("categorical heads need at least two bins")
        horizons = tuple(self.short_term_horizons)
        increasing = all(a < b for a, b in zip(horizons[:-1], horizons[1:], strict=True))
        if not horizons or not increasing or any(not (math.isfinite(h) and h > 0) for h in horizons):
            raise ValueError("short_term_horizons must be one or more finite positive horizons (turns), increasing (D3)")
        if self.soft_policy_dim < 1 or self.opp_policy_dim < 1:
            raise ValueError("soft_policy_dim and opp_policy_dim must be positive")
        if self.submit_head not in SUBMIT_HEADS:
            raise ValueError(f"submit_head must be one of {SUBMIT_HEADS}")
        if self.regs_per_board < 0 or self.global_regs < 0:
            raise ValueError("register counts must be non-negative")
        if self.rope_per_head or self.routing_table:
            if len(self.rope_head_roles) != self.n_heads:
                raise ValueError("rope_head_roles needs one role per head")
            if any(r not in _ROLES for r in self.rope_head_roles):
                raise ValueError(f"unknown head role; known: {sorted(_ROLES)}")
        if self.rope_per_head:
            if self.rope_relations_head not in ("dense", "near_nope"):
                raise ValueError("rope_relations_head must be 'dense' or 'near_nope'")
            if sum(self.rope_same_square_split) != self.n_pairs:
                raise ValueError("rope_same_square_split must sum to d_head / 2")
        if self.rope_roles_source not in ("table", "masks"):
            raise ValueError("rope_roles_source must be 'table' or 'masks'")
        edges = tuple(self.disp_far_edges)
        if (
            not edges
            or edges[0] != 3
            or any(not isinstance(e, int) for e in edges)
            or any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=True))
        ):
            raise ValueError(
                "disp_far_edges must be strictly increasing integers starting at 3 (the first board distance beyond the exact "
                "classes, section 4.1)"
            )
        if self.disp_line_max < 3:
            raise ValueError("disp_line_max must be at least 3 (a line beyond the exact classes moves 3 steps or more)")
        if self.disp_register_pairs not in DISP_REGISTER_PAIRS:
            raise ValueError(f"disp_register_pairs must be one of {DISP_REGISTER_PAIRS}")
        if not (math.isfinite(self.disp_init_std) and self.disp_init_std >= 0):
            raise ValueError("disp_init_std must be finite and non-negative")
        if self.shaw_key_term and not self.displacement_bias:
            raise ValueError("shaw_key_term (B17) is on top of B15: it needs displacement_bias (its classes are B15's)")
        if not 0 <= self.shaw_near <= 7:
            raise ValueError("shaw_near must be in 0..7 (a file-rank offset within B15's 15 x 15 slice)")
        if not (math.isfinite(self.shaw_init_std) and self.shaw_init_std >= 0):
            raise ValueError("shaw_init_std must be finite and non-negative")
        if self.status_dt_exact < 1 or self.status_dt_max < self.status_dt_exact:
            raise ValueError("status_dt_exact must be positive and status_dt_max not below it (B16's Delta-t bins)")
        if not (math.isfinite(self.status_bias_init_std) and self.status_bias_init_std >= 0):
            raise ValueError("status_bias_init_std must be finite and non-negative")
        if self.smolgen_time_exact < 1 or self.smolgen_timeline_exact < 1:
            raise ValueError("smolgen_time_exact and smolgen_timeline_exact must be positive")
        if self.smolgen_time_max < self.smolgen_time_exact or self.smolgen_timeline_max < self.smolgen_timeline_exact:
            raise ValueError("smolgen's *_max buckets must not be below their *_exact")
        if not 0 <= self.move_graph_heads <= self.n_heads:
            raise ValueError(f"move_graph_heads must be in 0..{self.n_heads}")
        default_heads = NetConfig.__dataclass_fields__["move_graph_heads"].default
        if self.routing_table and self.move_graph_heads != default_heads:
            raise ValueError(
                "move_graph_heads is not used under routing_table: the table (rope_head_roles) assigns every head, and "
                "section 6's table has one move-graph head - set rope_head_roles to choose the routed heads"
            )
        if self.move_graph_direction not in ("forward", "reverse", "symmetric"):
            raise ValueError("move_graph_direction must be 'forward', 'reverse' or 'symmetric'")
        if self.relations_head_direction not in (None, "forward", "reverse", "symmetric"):
            raise ValueError("relations_head_direction must be None, 'forward', 'reverse' or 'symmetric'")
        if self.masked_head_registers not in ("read", "self"):
            raise ValueError("masked_head_registers must be 'read' or 'self'")
        if self.same_board_registers not in ("own_board", "registers"):
            raise ValueError("same_board_registers must be 'own_board' or 'registers'")
        if not self.relations_head_types or any(t not in TYPED_RELATIONS for t in self.relations_head_types):
            raise ValueError(f"relations_head_types must be a non-empty subset of {TYPED_RELATIONS}")
        if self.smolgen_latent < 1:
            raise ValueError("smolgen_latent must be positive")
        if self.smolgen_form not in SMOLGEN_FORMS:
            raise ValueError(f"smolgen_form must be one of {SMOLGEN_FORMS}")
        for name, options in (
            ("norm_placement", NORM_PLACEMENTS),
            ("ffn_activation", FFN_ACTIVATIONS),
            ("attn_scope", ATTN_SCOPES),
            ("causal_past", CAUSAL_PASTS),
            ("frozen_features", FROZEN_FEATURES),
            ("supersession_scope", SUPERSESSION_SCOPES),
            ("frozen_kv_format", FROZEN_KV_FORMATS),
        ):
            if getattr(self, name) not in options:
                raise ValueError(f"{name} must be one of {options}")
        for name, stage in (
            ("piece_centric", "a later stage (A5)"),
            ("history_window", "a later stage, after the |E| verdict (C5, C7, M7)"),
            ("fp8_attention", "a later stage (G2)"),
        ):
            if getattr(self, name):
                raise NotImplementedError(f"{name} is built in stage {stage}, not in N1-N3")


F1 = NetConfig()
F2_LARGER_ARM = NetConfig.preset("L24-ff12")
