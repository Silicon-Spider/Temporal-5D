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
  typed edge biases (B11, provisional), board-pair smolgen (B12, prototype), and the larger
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
  spec leaves open on this committed-class path are parameters with placeholder defaults
  (``causal_past`` - since ruled by C23: ancestry, the ``"lineage"`` default -,
  ``frozen_features``, ``supersession_scope``, and the rule that the
  section 6 structure opens live rows only, documented in ``scope.py``).
* **Stage N5** adds register D8 (committed): the global acting-player input and the submit
  action's policy logit read from a global register. Where the input enters
  (``acting_player_input``), which register the logit is read from (``submit_register``) and
  how (``submit_head``) are not specified - parameters with placeholder defaults.
* **F5** (committed, CHECKPOINT 6, section 9.3): every trunk norm RMSNorm-style - no
  mean-centring, no bias (``trunk.RMSNorm``) - no biases in the Q, K and V projections
  (``qkv_bias``), the attention-output and FFN biases kept (``out_ffn_bias``), pre-norm;
  ablation 6y's arm, Lc0's post-norm with DeepNet initialisation, is
  ``norm_placement="post"`` (the placement is provisional), with the arm's unstated
  details as placeholders.
* **F6** (committed, CHECKPOINT 7, section 9.3): Mish in the FFN (``ffn_activation``);
  ablation 6y's arm, squared ReLU, is ``ffn_activation="squared_relu"``. Whether Mish
  reaches the output heads is not decided: ``head_activation`` (below).
* **Where the spec is silent, ambiguous or contradictory on a committed-class point**
  the choice is a parameter and the default is marked as a placeholder, not a decision:
  ``board_register_spatial`` (section 5.3 under RoPE), ``global_register_pairs`` (what
  a global register's position is under a relative encoding), ``aux_global_pool``
  (section 9.2 against section 12.4, B8 and D7), ``policy_promotion`` (section 8.1 says
  nothing about promotions), ``head_activation`` (whether F6's Mish reaches the heads),
  ``rope_relations_head`` and the near-NoPE and split details
  of B5, B15's far bins, line clamp, register entries and initialisation
  (``disp_far_edges``, ``disp_line_max``, ``disp_register_pairs``, ``disp_init_std``), and from
  stage N2: ``move_graph_direction`` (which way a directed move edge opens the mask),
  ``masked_head_registers`` (what registers do on masked heads), ``same_board_registers``
  (what a per-board register reads on the same-board head), ``relations_head_types``
  (which relations "attack/defend relations" covers), ``relations_head_direction``
  (which way a typed edge opens the relations head), ``rope_roles_source`` (how B5's
  per-head allocation meets B9's masks when the routing table is off) and how the routing
  table meets B9's head count (``move_graph_heads``, ``rope_head_roles``).
* Refused and held-open items have no switch at all (B3 axial, B13 token/within-board
  smolgen, C18 local/global interleave, C21 snapshot shortcuts, F3 d = 512, G4-G6;
  threat relations never become tokens, section 14.4).

Implementation choices where the spec is silent and the point is not committed-class
(initialisation scale, RoPE frequencies, head internals, the Delta-t embedding's MLP) are
documented beside their fields and in the modules that use them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Named capacity presets (section 9.2's ladder). "F1" is the committed v1 network;
#: "L24-ff12" is F2's larger arm (provisional; section 9.2's ~44M with heads is the count
#: without B15's table - 46,891,210 as built with it, spec issue 210).
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
#: Where register D8's acting-player input enters (stage N5; ``embed.py``).
ACTING_PLAYER_INPUTS: tuple[str, ...] = ("global_registers", "all_tokens")
#: How D8's submit logit is read from its global register (stage N5; ``heads.SubmitHead``).
SUBMIT_HEADS: tuple[str, ...] = ("linear", "pointer")
#: Where each sublayer's norm sits (F5; ``trunk.py``): "pre" (the v1) or "post" (ablation
#: 6y's arm: Lc0's post-norm with DeepNet initialisation).
NORM_PLACEMENTS: tuple[str, ...] = ("pre", "post")
#: The FFN's activation (F6; ``trunk.FeedForward``): "mish" (the v1) or "squared_relu"
#: (ablation 6y's arm).
FFN_ACTIVATIONS: tuple[str, ...] = ("mish", "squared_relu")
#: The value head's hidden activation (``heads.ValueHead``; ``NetConfig.head_activation``).
HEAD_ACTIVATIONS: tuple[str, ...] = ("gelu", "mish")
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
    # --- A1 / B8: registers (section 5.2) ---
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
    #: angles small (G1), and it is what places the anchor of
    #: ``global_register_pairs="anchored"``. Off only to test the relative property of
    #: the rotation itself.
    rope_recenter: bool = True
    #: Where a per-board register sits on the spatial (file, rank) axes. Section 5.3:
    #: registers get their board's (timeline, time) but "must not receive a (file, rank).
    #: Use a learned 'not a square' embedding on the spatial axes or they collide with
    #: the movement geometry." Under RoPE that admits three readings.
    #: COMMITTED-CLASS AMBIGUITY (B4, B8) - placeholder default, not a decision:
    #:   "learned_angle" - a learned off-lattice (file, rank) coordinate per register slot
    #:                     (initialised at the board centre, where no square is): a learned
    #:                     spatial position that is not a square's;
    #:   "position_free" - no spatial coordinate at all: every pair involving a per-board
    #:                     register uses unrotated file/rank dims on both sides (the
    #:                     timeline/time dims still rotate). The literal "must not receive
    #:                     a (file, rank)"; it widens the fast kernel's query/key features;
    #:   "zero_angle"    - unrotated (angle 0) spatial pairs on the register's side only.
    #:                     Under a relative encoding that is exactly square a1's position,
    #:                     i.e. the collision section 5.3 warns of; kept for ablation.
    #: Every option also adds the learned additive "not a square" embedding.
    board_register_spatial: str = "learned_angle"
    #: What position a global register has. Global registers hold no board (section
    #: 5.2), and a fixed angle on one side of a relative encoding is itself a position.
    #: COMMITTED-CLASS AMBIGUITY (B4, B8) - placeholder default, not a decision:
    #:   "position_free" - every pair involving a global register is unrotated on both
    #:                     sides (exact NoPE for those pairs); costs an exact log-sum-exp
    #:                     merge in the fast attention path;
    #:   "anchored"      - global registers are unrotated tokens at recentred coordinate 0,
    #:                     i.e. at (a1, p_ref, h_present): a board token's score with them
    #:                     depends on its file/rank and its offset from the lowest timeline
    #:                     and the present line. Translation-invariant because the anchor
    #:                     moves with the position; needs ``rope_recenter`` (with raw
    #:                     coordinates the anchor would be the absolute origin - an absolute
    #:                     positional encoding, section 14.1). One stock kernel call.
    global_register_pairs: str = "position_free"

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

    # --- F6: the FFN's activation (committed; section 9.3, CHECKPOINT 7; trunk.py) ---
    #: ``FFN_ACTIVATIONS``. Register F6: "Mish for v1 ... squared ReLU as the arm" (ablation
    #: 6y): "mish" - ``x tanh(softplus(x))`` (Misra 2019, "Mish: A Self Regularized
    #: Non-Monotonic Activation Function"); "squared_relu" - ``relu(x)^2`` (So et al. 2021,
    #: "Primer"), drawn at the same initialisation (the arm's is not stated - spec issue 204).
    ffn_activation: str = "mish"

    # --- embeddings (A2, A3, B8, section 5.3) ---
    #: Hidden width of the small MLP that embeds Delta-t-to-present.
    dt_hidden: int = 32
    #: Delta-t is fed as (dt / dt_scale, sign(dt) log1p|dt|, [dt == 0]).
    dt_scale: float = 16.0

    # --- heads (sections 8.1, 8.3, 12.4) ---
    #: Width of the policy pointer projections f and g (D1).
    policy_dim: int = 256
    #: How a promotion's piece enters the policy score. Section 8.1's score
    #: f(h_src)^T g(h_dst) cannot tell promotions to different pieces apart.
    #: COMMITTED-CLASS AMBIGUITY - placeholder default, not a decision:
    #:   "none"   - the literal section 8.1 formula (promotion variants tie);
    #:   "offset" - plus a learned per-piece offset read from the destination token.
    policy_promotion: str = "none"
    #: Width of the opponent-reply pointer projections (aux, ~0.1M budget of section 9.2).
    opp_policy_dim: int = 128
    #: Hidden width of the value head's MLP.
    value_hidden: int = 256
    #: The value head's hidden activation - the one output-head MLP (``heads.ValueHead``; the
    #: other heads are linear). Register F6 binds the "FFN activation"; its "(Chessformer's
    #: FFN and output heads)" and section 9.3's "Chessformer uses it in the feed-forward
    #: sublayer and the output heads" describe Chessformer, not whether this network's heads
    #: follow. COMMITTED-CLASS SILENCE (F6, D3; spec issue 204) - placeholder default, not a
    #: decision:
    #:   "gelu" - as built at stage N1 (exact erf form);
    #:   "mish" - F6's activation, as in Chessformer's heads.
    head_activation: str = "gelu"
    #: What the global auxiliary heads (WDL, moves-left, short-horizon value, search
    #: effort) read. Section 9.2's table: "Mean-pool board squares (per-board) / global
    #: registers (global)". Section 12.4: "Registers stay free. Auxiliary heads pool from
    #: the trunk; hard-assigning registers to auxiliary quantities conscripts the scratch
    #: space"; register B8: "aux heads pool from the trunk - registers stay free scratch";
    #: D7: registers are "never assigned aux targets".
    #: COMMITTED-CLASS CONTRADICTION (B8, D4) - placeholder default, not a decision:
    #:   "squares"          - the mean of every square token of the position (registers
    #:                        stay free: section 12.4, B8, D7);
    #:   "global_registers" - the mean of the global registers, the value head's input
    #:                        (section 9.2's table).
    #: The value head itself pools the global registers either way (D3, sections 7.3, 8.3).
    aux_global_pool: str = "squares"
    #: Per-square deep-attack-fact labels (provisional, undecided set: heads.SQUARE_LABELS).
    n_square_labels: int = 8
    #: Search-effort bins (section 8.6): log2-visit bins plus a censored top bin.
    effort_bins: int = 16

    # --- D8: the acting player and the submit action (stage N5; sections 2.3, 8.2) ---
    #: Turns have variable length (Q4, rulebook R8): once the present has shifted inside an
    #: action the colour to move on the present is the opponent's while the action is still
    #: the mover's, so per-board side-to-move flags cannot say whose action it is. Register
    #: D8: "a global acting-player input". Where it enters is not specified.
    #: COMMITTED-CLASS SILENCE (D8, A3) - placeholder default, not a decision:
    #:   "global_registers" - a learned acting-player embedding (white / black) added to every
    #:                        global register token: a global input on the global tokens
    #:                        (section 5.2), read by every live token. Global registers are
    #:                        live, so the cached path stays exact with no change to a frozen
    #:                        board's K/V; under the block-causal scope frozen boards never
    #:                        see it (C17: present-to-frozen flow only at refreshes);
    #:   "all_tokens"       - the same embedding also on every token of every board, like a
    #:                        section 2.3 per-board flag. Under the block-causal scope with
    #:                        ``frozen_features="intrinsic"`` a frozen board does not get it
    #:                        (it changes after the board is frozen); under ``"current"`` it
    #:                        does, and a frozen board's cache identity carries it.
    acting_player_input: str = "global_registers"
    #: Register D8: under the board-move split submitting is "an explicit action with its own
    #: policy logit, read from a global register" (section 8.2). Which register, and how the
    #: logit is read, are not specified. COMMITTED-CLASS SILENCE (D8) - placeholder defaults,
    #: not decisions. ``submit_register``: the index of the global register the logit is read
    #: from (0; with no global registers - an ablation-3 arm - the mean of the square tokens,
    #: as the value head falls back). ``submit_head``:
    #:   "linear"  - its own learned linear readout of that register (``heads.SubmitHead``):
    #:               d + 1 parameters, on the move logits' scale at initialisation;
    #:   "pointer" - the policy's own bilinear form with the register at both ends,
    #:               ``f(h_r)^T g(h_r) / sqrt(d_policy)``: no parameter of its own, one
    #:               scoring function for moves and submit.
    #: The logit is offered where submitting is possible (the obligations met, T5:
    #: ``PositionState.submit_possible``) and masked to -inf unless it is legal (C4:
    #: ``submit_legal``); the policy distribution is then over the moves and submit together.
    submit_register: int = 0
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
    #: B11's typed-edge scalars and learned register coordinates' noise (B15's table has its
    #: own, ``disp_init_std``).
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
    #: 0..k-1 whatever the table says. COMMITTED-CLASS AMBIGUITY (B5 is class C) -
    #: placeholder default, not a decision:
    #:   "table" - roles from ``rope_head_roles`` by head index, whatever the masks (N1's
    #:             behaviour: with B9's default, head 1 is move-graph masked and rotates
    #:             on file and rank only);
    #:   "masks" - each head's role is the role of the mask it carries: B9's move-graph
    #:             heads near-NoPE ("its structure comes from the mask"), every unmasked
    #:             head the dense B4 split.
    #: Under the routing table the two coincide (the masks are the table).
    rope_roles_source: str = "table"
    #: Allocation of the attack/defend-relations head (routing table head 4), which
    #: section 4.1 does not mention. COMMITTED-CLASS AMBIGUITY (B5 is class C):
    #: "dense" (the B4 split; placeholder default) or "near_nope" (like the move-graph head).
    rope_relations_head: str = "dense"
    #: Same-square-across-boards head: rotary pairs on (timeline, time). Placeholder
    #: default keeps B4's 4:6 ratio over 16 pairs.
    rope_same_square_split: tuple[int, int] = (6, 10)
    #: Near-NoPE heads rotate every pair at a frequency giving at most this angle over its
    #: axis's whole range. Placeholder default.
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

    # --- B9: the move-graph mask (committed; section 6) ---
    #: Heads 0..k-1 carry the hard move-graph mask, the rest are dense ("2 heads with hard
    #: move-graph masks, remainder dense"; ablation 4 runs 0 / 2 / 4).
    #: Under ``routing_table`` the table assigns every head and this count is not used.
    #: COMMITTED-CLASS AMBIGUITY (B9 committed "on >=2 heads"; B10 class C) - placeholder,
    #: not a decision: section 6's table has one move-graph head (head 1) beside the
    #: attack/defend relations head (head 4), and whether B9's ">=2" counts the relations
    #: head or the table should carry two move-graph heads is open. The table is built as
    #: written (``rope_head_roles`` can give it two), and ``validate`` refuses any count
    #: but the default under the routing table, so that an ablation-4 arm cannot run with
    #: the table's single move-graph head without saying so.
    move_graph_heads: int = 2
    #: Which way a directed edge ``(src, dst, relation)`` opens a masked head - the move
    #: graph (B9), and the relations head (B10) unless ``relations_head_direction`` says
    #: otherwise. Section 6 says "which squares reach which" and gives no direction.
    #: COMMITTED-CLASS AMBIGUITY (B9 is class R in the register, but the direction fixes
    #: what every masked head is trained to read) - placeholder default, not a decision:
    #:   "forward"   - the query is the edge's source: a piece's square reads the squares
    #:                 it reaches;
    #:   "reverse"   - the query is the edge's destination: a square reads the squares
    #:                 whose pieces reach it ("which pieces control it", section 5.1);
    #:   "symmetric" - both (the symmetric closure; each other arm is a restriction of it).
    move_graph_direction: str = "symmetric"
    #: What a register query does on a masked head (any head whose role is not "dense")
    #: where the head's relation does not cover it. Registers are never edge endpoints
    #: (section 5.3) and have no (file, rank), so the move-graph, relations and
    #: same-square relations cover no register; global registers hold no board (section
    #: 5.2), so no relation covers them at all. (A per-board register is on its board: the
    #: same-board head's relation can cover it - ``same_board_registers``.) A query's own
    #: token is always a key, so no valid token's row is ever empty.
    #: COMMITTED-CLASS AMBIGUITY (section 6 is silent) - placeholder default, not a decision:
    #:   "read" - such a register reads every valid key, as on a dense head (for global
    #:            registers section 7.3's "reading is free, being read is not");
    #:   "self" - it reads only itself (the literal hard mask for every token).
    #: Either way a square never reads a register on a masked head, except its own
    #: board's per-board registers on the same-board head (they are on its board).
    masked_head_registers: str = "read"
    #: What a per-board register reads on the same-board head (B10). Section 6's head 2 is
    #: "Same board only", and a per-board register is on its board - "appended to each
    #: board" (section 5), it "belong[s] to [its] board's block" (section 7.3) - so that
    #: relation covers it as a query. COMMITTED-CLASS AMBIGUITY (B10 is class C; section 6
    #: does not mention registers) - placeholder default, not a decision:
    #:   "own_board" - it reads its own board's tokens, squares and per-board registers,
    #:                 as the board's squares do (the literal "same board only");
    #:   "registers" - it follows ``masked_head_registers``, as on every other masked head
    #:                 (stage N2's first build).
    same_board_registers: str = "own_board"

    # --- B10: the fixed routing table (prototype; default off; section 6) ---
    #: Heads take the roles of ``rope_head_roles`` (section 6's table by default: head 1
    #: legal move graph, 2 same board only, 3 same (file, rank) across boards, 4
    #: attack/defend relations, 5-8 dense - 0-based in code).
    routing_table: bool = False
    #: The relation types whose edges open the relations head ("attack/defend relations").
    #: COMMITTED-CLASS AMBIGUITY (B10 is class C) - placeholder default, not a decision:
    #: every typed relation of B11 (section 6: heads 1 and 4 "receive binary masks:
    #: connected or not", which B11's types upgrade) - five since travel-threat was folded
    #: into attack (CHECKPOINT 6); the narrow reading is ("attack", "defence").
    relations_head_types: tuple[str, ...] = TYPED_RELATIONS
    #: Which way a typed edge opens the relations head: "forward", "reverse" or
    #: "symmetric", as ``move_graph_direction`` defines them; None - the placeholder
    #: default - takes ``move_graph_direction``'s value. COMMITTED-CLASS AMBIGUITY (B10 is
    #: class C; section 6 gives no direction for "attack/defend relations" either).
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
    #: active flag when the timeline is activated, after the board is frozen. The spec
    #: (A3, A4, C2, C17) is silent. COMMITTED-CLASS SILENCE - placeholder default, not a
    #: decision:
    #:   "intrinsic" - a frozen board's tokens get neither (only its contents, side to move
    #:                 and the superseded flag): knowing where the present is would be
    #:                present-to-frozen flow, which C17 allows only at refreshes (the root
    #:                refresh, a dense pass, gives every board its features as of the
    #:                root). The only reading under which a frozen board's K/V stay valid
    #:                between refreshes;
    #:   "current"   - a frozen board carries the evaluated state's values (A3 literally):
    #:                 every shift of the present changes every frozen board's input, so
    #:                 the cache recomputes them all - the cost section 7.1 does not price.
    #: Live boards always carry both features.
    frozen_features: str = "intrinsic"
    #: C20 (committed): a board superseded inside the search is recomputed "under the
    #: state-defined mask - its causal past plus the conditioned history of its refresh
    #: epoch". COMMITTED-CLASS AMBIGUITY - placeholder default, not a decision:
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
        if self.board_register_spatial not in ("learned_angle", "position_free", "zero_angle"):
            raise ValueError("board_register_spatial must be 'learned_angle', 'position_free' or 'zero_angle'")
        if self.global_register_pairs not in ("position_free", "anchored"):
            raise ValueError("global_register_pairs must be 'position_free' or 'anchored'")
        if self.global_register_pairs == "anchored" and not self.rope_recenter:
            raise ValueError(
                "global_register_pairs='anchored' needs rope_recenter=True: without recentring the anchor is the "
                "absolute origin, an absolute positional encoding (section 14.1)"
            )
        if self.aux_global_pool not in ("squares", "global_registers"):
            raise ValueError("aux_global_pool must be 'squares' or 'global_registers'")
        if self.policy_promotion not in ("none", "offset"):
            raise ValueError("policy_promotion must be 'none' or 'offset'")
        if self.value_bins < 2 or self.effort_bins < 2:
            raise ValueError("categorical heads need at least two bins")
        if self.acting_player_input not in ACTING_PLAYER_INPUTS:
            raise ValueError(f"acting_player_input must be one of {ACTING_PLAYER_INPUTS}")
        if self.submit_head not in SUBMIT_HEADS:
            raise ValueError(f"submit_head must be one of {SUBMIT_HEADS}")
        if self.submit_register < 0 or (self.global_regs > 0 and self.submit_register >= self.global_regs):
            raise ValueError(f"submit_register must name a global register (0..{max(self.global_regs - 1, 0)})")
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
            ("head_activation", HEAD_ACTIVATIONS),
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
