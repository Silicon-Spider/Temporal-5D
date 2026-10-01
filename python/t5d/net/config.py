"""Network configuration.

Defaults are the committed v1 network (register F1, spec section 9.2): d = 256, L = 16,
8 heads x d_head 32 (full MHA, C4), FFN 8x, 2 registers per board and 16 global (A1,
B8), a 51-bin categorical value head (D3), 4D RoPE with pairs file 3 / rank 3 /
timeline 4 / time 6 (B4), BF16 trunk with FP32 RoPE and heads (G1).

Stage N2 (the attention structure, spec section 6) adds the hard move-graph mask on 2
heads (B9, committed: on by default) and B14's attention paths (committed engineering):
``"block_sparse"`` (the default ``attn_impl``), whose masked heads read only the key
boards their query block can reach (board-major order, section 6.1), and ``"chunked"``,
the dense-mask arm of ablation 6. B14's FlexAttention ``score_mod`` is not built: it
needs a compiler (torch.compile has none on the development machine) and a GPU to be
measured - see ``attention.py``.

The build rules (spec section 15.1) decide what else appears here:

* **Provisional and prototype items** are behind flags, all default off: per-head
  positional specialisation (B5, provisional), binned clamped Delta-time/Delta-timeline
  biases (B6, prototype), the fixed routing table (B10, prototype), typed edge biases
  (B11, provisional), board-pair smolgen (B12, prototype), and the larger capacity arm
  L24-ff12 (F2, provisional) as the named preset ``"L24-ff12"``.
* **Items of later stages** have their flag here so the config names every switch, but
  enabling one raises ``NotImplementedError`` until its stage lands: piece-centric
  tokenisation (A5), the causal-and-cached attention scope (C2, deferred but needed by
  section 15.2's equivalence suite - stage N3), the history window (C5/C7, provisional;
  its code waits on the |E| verdict, M7) and FP8 attention (G2, provisional; no FP8
  kernels on CPU).
* **Where the spec is silent, ambiguous or contradictory on a committed-class point**
  the choice is a parameter and the default is marked as a placeholder, not a decision:
  ``board_register_spatial`` (section 5.3 under RoPE), ``global_register_pairs`` (what
  a global register's position is under a relative encoding), ``aux_global_pool``
  (section 9.2 against section 12.4, B8 and D7), ``policy_promotion`` (section 8.1 says
  nothing about promotions), ``rope_relations_head`` and the near-NoPE and split details
  of B5, ``binned_bias_rope`` (what B6's "RoPE for near-time" leaves rotated), and from
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
(norm, activation, initialisation scale, RoPE frequencies, head internals) are
documented beside their fields and in the modules that use them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Named capacity presets (section 9.2's ladder). "F1" is the committed v1 network;
#: "L24-ff12" is F2's larger arm (provisional; ~44M with heads per section 9.2).
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

#: The typed relations of B11 (section 6): every relation type but the move graph itself.
TYPED_RELATIONS: tuple[str, ...] = ("attack", "defence", "pin", "xray", "check", "travel_threat")

#: Attention implementations (``attention.py``): "block_sparse" (B14, the default: the
#: fused kernel over board-aligned query blocks, each masked head reading only the key
#: boards its block can reach), "chunked" (the same blocks against every key - the dense-
#: mask arm of ablation 6), "sdpa" (N1's fused path in one block - materialises
#: (B, H, L, L) rows once a mask or a bias is present), "explicit" (materialised scores,
#: chunked over queries) and "dense" (the reference: every mask and bias materialised at
#: (B, H, L, L) first - small inputs).
ATTN_IMPLS: tuple[str, ...] = ("block_sparse", "chunked", "sdpa", "explicit", "dense")

#: B12's generator forms (``smolgen.py``).
SMOLGEN_FORMS: tuple[str, ...] = ("relative_cells", "board_bilinear")


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
    #: extrapolation risk B6 guards.
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
    #: bias first. All five compute the same function on every valid query (the
    #: position-free pairs merged exactly - see attention.py); the tests compare them.
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

    # --- trunk details (implementation choices; spec silent) ---
    #: Pre-norm residual blocks with LayerNorm; FFN activation GELU (exact erf form).
    norm_eps: float = 1e-5
    #: Linear layers carry biases.
    linear_bias: bool = True

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

    # --- initialisation (section 15.2: realistic scale, never zeros) ---
    #: Linear weights: normal with std 1/sqrt(fan_in) (LeCun); the attention output and
    #: FFN output projections are further scaled by 1/sqrt(2 * n_layers) (GPT-2 style).
    init_embed_std: float = 0.5
    #: Linear and LayerNorm biases.
    init_bias_std: float = 0.02
    #: LayerNorm gains are 1 + N(0, init_norm_std): not the identity, so that a path that
    #: dropped or misplaced a norm's affine could not agree with one that kept it.
    init_norm_std: float = 0.1
    #: Learned bias tables (B6, B11's typed-edge scalars) and learned register coordinates' noise.
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

    # --- B6: binned clamped Delta-time / Delta-timeline biases (prototype; default off) ---
    binned_bias: bool = False
    #: What RoPE keeps when the biases are on. Section 4.1 says "RoPE for file, rank and
    #: near-time; binned learned biases for Delta-time and Delta-timeline" while register
    #: B6 says "RoPE + binned clamped biases". COMMITTED-CLASS AMBIGUITY:
    #: "keep" - B4's rotation unchanged, biases added (placeholder default);
    #: "near_time" - timeline pairs and the time pairs coarser than ``binned_bias_near_time``
    #:               half-turns are left unrotated; the biases carry the coarse offsets.
    binned_bias_rope: str = "keep"
    binned_bias_near_time: float = 16.0
    #: Buckets: offsets below ``*_exact`` get their own bucket; beyond, log2-spaced
    #: buckets up to ``*_max``, then clamped (the last bucket is shared by every larger
    #: offset). Delta-time is signed (past and future differ); Delta-timeline symmetric
    #: (branching order makes its sign semi-arbitrary, section 4.4).
    bias_time_exact: int = 4
    bias_time_max: int = 128
    bias_timeline_exact: int = 4
    bias_timeline_max: int = 32

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
    #: all six typed relations of B11 (section 6: heads 1 and 4 "receive binary masks:
    #: connected or not", which B11's types upgrade); the narrow reading is
    #: ("attack", "defence").
    relations_head_types: tuple[str, ...] = TYPED_RELATIONS
    #: Which way a typed edge opens the relations head: "forward", "reverse" or
    #: "symmetric", as ``move_graph_direction`` defines them; None - the placeholder
    #: default - takes ``move_graph_direction``'s value. COMMITTED-CLASS AMBIGUITY (B10 is
    #: class C; section 6 gives no direction for "attack/defend relations" either).
    relations_head_direction: str | None = None

    # --- B11: typed edge biases (provisional; default off; section 6) ---
    #: A learned scalar per head per relation type (``TYPED_RELATIONS``), shared by all
    #: layers (section 6: "a handful of learned scalars per head per type"), added to the
    #: logits of the related token pairs, cross-board pairs included.
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
    #:                      relative board-pair cells (B6's Delta-time / Delta-timeline
    #:                      buckets): generated from the global registers alone and
    #:                      full-rank over board pairs (section 6, register B12);
    #:   "board_bilinear" - per head, a bilinear form between board descriptors taken
    #:                      from each board's per-board registers, modulated by the latent:
    #:                      rank <= ``smolgen_latent`` - the low-rank form section 6
    #:                      equates with extra Q/K dims (stage N2's first build).
    smolgen_form: str = "relative_cells"

    # --- later stages: named here, built there (enabling raises NotImplementedError) ---
    piece_centric: bool = False  # A5 (prototype, committed-class): not built in N1 or N2
    cached_attention: bool = False  # C2 (deferred), C17 block-causal mask: stage N3
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
        if self.binned_bias_rope not in ("keep", "near_time"):
            raise ValueError("binned_bias_rope must be 'keep' or 'near_time'")
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
        for name, stage in (
            ("piece_centric", "a later stage (A5)"),
            ("cached_attention", "N3 (C2, C17)"),
            ("history_window", "a later stage, after the |E| verdict (C5, C7, M7)"),
            ("fp8_attention", "a later stage (G2)"),
        ):
            if getattr(self, name):
                raise NotImplementedError(f"{name} is built in stage {stage}, not in N1 or N2")


F1 = NetConfig()
F2_LARGER_ARM = NetConfig.preset("L24-ff12")
