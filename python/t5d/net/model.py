"""The dense v1 network: embeddings, trunk, heads (stage N1 of the untrained build).

``build_model(config, seed)`` returns a ``T5DNet`` with random weights at realistic
initialisation scale (spec section 15.2: "never zeros or near-zeros"), deterministically
from the seed. ``T5DNet(batch)`` returns a ``NetOutput`` with every head.

Initialisation (implementation choice where the spec is silent):

* Linear weights ``N(0, 1/fan_in)`` (LeCun normal) - attention logits then have
  roughly unit spread at initialisation, so attention is far from uniform. Under
  pre-norm (F5's v1) the two residual-output projections of every block (attention out,
  FFN out) are further scaled by ``1/sqrt(2L)`` (GPT-2), keeping the residual stream
  bounded over depth. Under the post-norm arm (ablation 6y: "post-norm with DeepNet
  initialisation") the trunk's projections are drawn by DeepNet's initialisation instead,
  as the paper's pseudocode draws it (Wang et al. 2022, Figure 2, ``deepnorm_init``):
  every attention and FFN projection Xavier normal, ``std = gain sqrt(2 / (fan_in +
  fan_out))``, at gain ``beta = (8L)^(-1/4)`` on the value, attention-output and both FFN
  projections and gain 1 on the query and key projections. The figure's caption lets
  another standard initialisation replace Xavier; the arm keeps the pseudocode's (a
  placeholder, spec issue 201). For the square attention projections Xavier and LeCun
  coincide; for the FFN's ``d -> 8d`` and ``8d -> d`` matrices Xavier draws both at
  ``sqrt(2/9)/sqrt(d)`` - about 0.47x and 1.33x LeCun's - so the FFN branch starts about
  0.63x as large as it would on a LeCun base.
* Linear biases ``N(0, init_bias_std)`` (the Q, K and V projections have none - F5);
  embeddings and the "not a square" vector ``N(0, init_embed_std)``; RMSNorm gains
  ``1 + N(0, init_norm_std)`` - near the identity, but not it, so that no parameter is
  zero or constant (section 15.2: zeros "make distinct paths agree trivially").
* B11's typed-edge scalars ``N(0, init_table_std)``; learned register coordinates (option
  ``learned_angle``) at the board centre plus ``N(0, init_table_std)`` noise; B12's weights
  by ``BoardPairSmolgen.reset_parameters`` (biases of spread of order one at
  initialisation).
* The learned register coordinates and the structure's flag-gated modules draw last, in
  the order register coordinates, B11, B12, and none of them draws from the RNG while
  it is constructed: switching one on changes its own weights and those of the
  flag-gated modules drawn after it, never the trunk's or the heads' - so an on/off
  comparison of B12, say, differs in B12 alone.
* B15's displacement table (the v1 default) ``N(0, disp_init_std)`` (a placeholder, spec
  issue 208), drawn - like D8's parameters, below - from a generator of its own seeded from
  the model's seed (``b15_seed``), never from the global stream, after everything else: its
  no-table arm (ablation 2) has every other weight of the v1, bit for bit, and no flag
  changes the table.

Attention structure (stage N2, ``structure.py``): ``make_context`` builds the batch's
masks (B9's move graph on ``move_graph_heads`` heads, or B10's routing table) and biases
(B11, B12, B15) once per forward, as sparse lists, board-pair tables and B15's per-batch
index; every layer's attention builds dense rows from them one query block at a time
(B14), with its own layer's tables (``AttnContext.for_layer``).

Attention scope (stage N3, ``scope.py``): ``NetConfig.attn_scope`` - dense (C1) or
block-causal (C17, C19) - and any forward may override it (``scope=``): the cached path's
refreshes are dense passes of a block-causal network (C3, ``cache.py``). Under the
block-causal scope ``make_context`` adds the scope's rows, keeps the edge-derived
structure to live rows, and the embedding drops a frozen board's state-dependent flags
(``frozen_features``). ``history`` gives each position's epoch history (C20's second
reading, ``supersession_scope``) and ``kv_hook`` sees every layer's keys and values.

Heads (``T5DNet.heads``) read the final-normed trunk: the policy pointer at each move's
source and destination tokens (D1), the value head the mean of the global registers
(D3, sections 7.3 and 8.3), the global auxiliary heads the pool ``aux_global_pool``
selects (an open committed-class point), per-board heads each board's square mean and
per-square heads every token (``heads.py``).

Register D8 (stage N5): the acting player is an input (``embed.py``,
``NetConfig.acting_player_input``) and the submit action's logit is read from a global
register (``heads.SubmitHead``), offered where submitting is possible and masked unless it is
legal; ``NetOutput.action_log_probs`` normalises the moves and submit together. Their
parameters (the acting-player embedding, the submit readout) draw nothing while they are
made and are drawn from a generator of their own, seeded from the model's seed
(``d8_seed``), never from the global stream: for a seed, every parameter that existed before
N5 is drawn exactly as it was - with any flags on and under either of D8's arms - and no flag
changes D8's weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .attention import AttnContext
from .config import NetConfig
from .displacement import DisplacementBias
from .embed import InputEmbedding
from .heads import (
    AuxHeads,
    PointerHead,
    SubmitHead,
    ValueHead,
    board_mean_pool,
    categorical_mean,
    categorical_probs,
    categorical_variance,
    global_register_pool,
    square_pool,
)
from .kvformat import fake_quantise
from .rope import rope_angles, rope_cos_sin, rope_tables, spatial_dims
from .scope import BlockCausalScope
from .smolgen import BoardPairSmolgen
from .state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, Batch
from .structure import TypedEdgeBias, build_structure
from .trunk import RMSNorm, Trunk


@dataclass
class NetOutput:
    """Every head's output for a batch (all FP32)."""

    policy_logits: torch.Tensor  # (B, M), -inf at padding
    value_logits: torch.Tensor  # (B, value_bins)
    wdl_logits: torch.Tensor  # (B, 3)
    moves_left_log: torch.Tensor  # (B,)  predicted log1p(board-moves left)
    opp_reply_logits: torch.Tensor  # (B, M2), -inf at padding
    short_value_logits: torch.Tensor  # (B, value_bins)
    board_logits: torch.Tensor  # (B, Nmax, 3): heads.BOARD_HEADS
    square_logits: torch.Tensor  # (B, L, n_square_labels); meaningful where square_mask
    effort_logits: torch.Tensor  # (B, effort_bins)
    move_valid: torch.Tensor
    opp_move_valid: torch.Tensor
    board_valid: torch.Tensor
    square_mask: torch.Tensor
    token_board: torch.Tensor  # (B, L) long: board index of each token, -1 where none
    # D8 (stage N5): the submit action - its logit read from a global register, (B,) each.
    submit_logits: torch.Tensor  # -inf where submit is no legal candidate (impossible, or illegal: masked)
    submit_raw: torch.Tensor  # the register's readout before the mask (finite everywhere)
    submit_possible: torch.Tensor  # bool: submit is offered (T5)
    submit_valid: torch.Tensor  # bool: offered and legal (C4) - the entry carries probability
    trunk: torch.Tensor | None = None  # (B, L, d) final-normed trunk output, if requested

    def value_probs(self) -> torch.Tensor:
        return categorical_probs(self.value_logits)

    def value_mean(self) -> torch.Tensor:
        return categorical_mean(self.value_probs())

    def value_variance(self) -> torch.Tensor:
        return categorical_variance(self.value_probs())

    def action_logits(self) -> torch.Tensor:
        """``(B, M + 1)``: the move logits, then the submit logit (D8) - every action of a
        decision under the board-move split; -inf where an entry is no legal candidate."""
        return torch.cat([self.policy_logits, self.submit_logits[:, None]], dim=-1)

    def action_valid(self) -> torch.Tensor:
        """``(B, M + 1)`` bool: the candidate moves, then whether submit is a legal candidate."""
        return torch.cat([self.move_valid, self.submit_valid[:, None]], dim=-1)

    def action_log_probs(self) -> torch.Tensor:
        """``(B, M + 1)``: log-softmax over each position's candidate moves and, where it is
        offered and legal, submit (D8): one distribution over the decision's actions; -inf at
        padding and on a masked or absent submit. A position with no candidate at all has
        -inf everywhere."""
        valid = self.action_valid()
        logits = self.action_logits().masked_fill(~valid, float("-inf"))
        logp = torch.log_softmax(logits, dim=-1)
        return torch.where(valid, logp, torch.full_like(logp, float("-inf")))

    def policy_log_probs(self) -> torch.Tensor:
        """``(B, M)``: the moves' log-probabilities (-inf at padding), normalised together with
        the submit action where it is a legal candidate - so the moves' probabilities and
        ``submit_log_prob``'s sum to one; over the moves alone where submit is not offered."""
        return self.action_log_probs()[:, :-1]

    def submit_log_prob(self) -> torch.Tensor:
        """``(B,)``: submit's log-probability in the decision's distribution (-inf where it is
        not offered or not legal)."""
        return self.action_log_probs()[:, -1]


class T5DNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        cfg = NetConfig() if cfg is None else cfg
        cfg.validate()
        self.cfg = cfg
        self.embed = InputEmbedding(cfg)
        self.trunk = Trunk(cfg)
        axis, freq = rope_tables(cfg)
        self.register_buffer("rope_axis", axis, persistent=False)
        self.register_buffer("rope_freq", freq, persistent=False)
        self.register_buffer("rope_spatial", spatial_dims(axis), persistent=False)
        learned_xy = cfg.board_register_spatial == "learned_angle" and cfg.regs_per_board > 0
        self.register_xy = nn.Parameter(torch.empty(cfg.regs_per_board, 2)) if learned_xy else None
        self.typed_edges = TypedEdgeBias(cfg) if cfg.typed_edge_bias else None
        self.smolgen = BoardPairSmolgen(cfg) if cfg.board_pair_smolgen else None
        self.displacement = DisplacementBias(cfg) if cfg.displacement_bias else None
        self.policy = PointerHead(cfg.d_model, cfg.policy_dim, promotion=cfg.policy_promotion)
        self.value = ValueHead(cfg)
        self.aux = AuxHeads(cfg)
        self.submit = SubmitHead(cfg)
        self._init_weights()

    # -- initialisation --

    #: Register D8's parameters (stage N5), drawn after every other trunk and head parameter
    #: and before the flag-gated modules (``_init_weights``).
    _D8_MODULES = ("embed.acting", "submit")

    #: The trunk projections that DeepNet's initialisation draws under post-norm (Wang et al.
    #: 2022, Figure 2): Xavier normal at gain beta ("ffn", "v_proj", "out_proj") and at gain
    #: 1 ("q_proj", "k_proj"); and those the GPT-2 scaling shrinks under pre-norm (the
    #: residual outputs).
    _DEEPNET_SCALED = ("attn.v_proj", "attn.out_proj", "ffn.lin_in", "ffn.lin_out")
    _DEEPNET_UNSCALED = ("attn.q_proj", "attn.k_proj")
    _GPT2_SCALED = ("attn.out_proj", "ffn.lin_out")

    def _linear_std(self, name: str, module: nn.Linear) -> float:
        """The std of the normal a Linear's weight is drawn from (the module docstring): LeCun
        normal, the pre-norm residual outputs at GPT-2's ``1/sqrt(2L)`` of it, and under
        post-norm the trunk's projections at DeepNet's Xavier normal - ``xavier_normal_(w,
        gain)``'s std, one draw per weight from the same stream whichever applies."""
        cfg = self.cfg
        trunk = name.startswith("trunk.")
        if trunk and cfg.norm_placement == "post" and name.endswith(self._DEEPNET_SCALED + self._DEEPNET_UNSCALED):
            gain = cfg.deepnorm_beta if name.endswith(self._DEEPNET_SCALED) else 1.0
            return gain * math.sqrt(2.0 / (module.in_features + module.out_features))
        std = module.in_features**-0.5
        if trunk and cfg.norm_placement == "pre" and name.endswith(self._GPT2_SCALED):
            std *= 1.0 / math.sqrt(2 * cfg.n_layers)
        return std

    def _init_weights(self) -> None:
        cfg = self.cfg
        with torch.no_grad():
            for name, module in self.named_modules():
                if name.split(".")[0] == "smolgen":
                    continue  # drawn last (reset_parameters)
                if any(name == d8 or name.startswith(d8 + ".") for d8 in self._D8_MODULES):
                    continue  # drawn after the rest (below)
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, self._linear_std(name, module))
                    if module.bias is not None:
                        nn.init.normal_(module.bias, 0.0, cfg.init_bias_std)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, 0.0, cfg.init_embed_std)
                elif isinstance(module, RMSNorm):
                    nn.init.normal_(module.weight, 1.0, cfg.init_norm_std)
            nn.init.normal_(self.embed.not_square, 0.0, cfg.init_embed_std)
            if self.register_xy is not None:
                self.register_xy.normal_(0.0, cfg.init_table_std)
                self.register_xy.add_(3.5)  # board centre: off-lattice, at no square
            # D8 (stage N5): from a generator of its own, seeded from the global one's seed
            # (``build_model``'s), never from the global stream - and D8's modules draw nothing
            # while they are made - so that every parameter that existed before N5 is drawn as
            # it was, whatever the flags and D8's own arms, and no flag changes D8's weights.
            d8 = torch.Generator(device="cpu").manual_seed(d8_seed(torch.initial_seed()))
            acting = torch.empty(self.embed.acting.weight.shape).normal_(0.0, cfg.init_embed_std, generator=d8)
            self.embed.acting.weight.copy_(acting)
            if self.submit.out is not None:
                out = self.submit.out
                out.weight.copy_(torch.empty(out.weight.shape).normal_(0.0, out.in_features**-0.5, generator=d8))
                out.bias.copy_(torch.empty(out.bias.shape).normal_(0.0, cfg.init_bias_std, generator=d8))
            if self.typed_edges is not None:
                nn.init.normal_(self.typed_edges.table, 0.0, cfg.init_table_std)
            if self.smolgen is not None:
                self.smolgen.reset_parameters(cfg)
            # B15 (the v1 default): from a generator of its own, as D8's parameters are, so that
            # its no-table arm (ablation 2) leaves every other weight as it is.
            if self.displacement is not None:
                b15 = torch.Generator(device="cpu").manual_seed(b15_seed(torch.initial_seed()))
                table = self.displacement.table
                table.copy_(torch.empty(table.shape).normal_(0.0, cfg.disp_init_std, generator=b15))

    # -- positional context --

    def coordinates(self, batch: Batch) -> torch.Tensor:
        """Per-token RoPE coordinates ``(B, L, 4)`` (file, rank, timeline, time), FP32.

        Global registers sit at 0 on every axis after recentring: the anchor
        ``(a1, p_ref, h_present)`` of ``global_register_pairs="anchored"`` (unused under
        ``"position_free"``, where no pair with a global register is rotated). Per-board
        registers keep their board's (timeline, time); their (file, rank) is 0, or the
        learned coordinate under ``board_register_spatial="learned_angle"`` (unused under
        ``"position_free"``).
        """
        coords = batch.coords.float().clone()
        if self.cfg.rope_recenter:
            coords[..., 2] -= batch.coord_ref[:, None, 0]
            coords[..., 3] -= batch.coord_ref[:, None, 1]
        is_global = (batch.kind == TOKEN_GLOBAL_REGISTER)[..., None]
        coords = torch.where(is_global, torch.zeros_like(coords), coords)
        if self.register_xy is not None:
            breg = batch.kind == TOKEN_BOARD_REGISTER
            xy = self.register_xy[batch.reg_index.clamp(min=0, max=self.register_xy.shape[0] - 1)]
            coords[..., :2] = torch.where(breg[..., None], xy, coords[..., :2])
        return coords

    def scope_of(self, scope: str | None) -> str:
        """The attention scope of a forward: ``scope``, or the config's when None."""
        scope = self.cfg.attn_scope if scope is None else scope
        if scope not in ("dense", "block_causal"):
            raise ValueError(f"unknown attention scope {scope!r}")
        return scope

    def embed_tokens(self, batch: Batch, scope: str | None = None) -> torch.Tensor:
        """Token embeddings under a scope: under the block-causal scope with
        ``frozen_features="intrinsic"`` frozen boards get no state-dependent flags."""
        intrinsic = self.scope_of(scope) == "block_causal" and self.cfg.frozen_features == "intrinsic"
        return self.embed(batch, frozen_intrinsic=intrinsic)

    def make_context(
        self,
        batch: Batch,
        attn_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        *,
        scope: str | None = None,
        history: torch.Tensor | None = None,
    ) -> AttnContext:
        """The attention context of a batch: RoPE tables, padding, the structure of stage N2
        (``structure.py``: B9/B10 masks, B11/B12/B15 biases - built sparsely, rows on demand),
        under the block-causal scope its rows (stage N3, ``scope.py``; ``history``: each
        position's epoch history, ``(B, Nmax)`` bool), and optionally a caller's dense
        ``attn_mask`` / ``attn_bias`` on top."""
        cfg = self.cfg
        coords = self.coordinates(batch)
        cos, sin = rope_cos_sin(rope_angles(coords, self.rope_axis, self.rope_freq))
        spatial_free = None
        if cfg.board_register_spatial == "position_free":
            spatial_free = batch.valid & (batch.kind == TOKEN_BOARD_REGISTER)
        block_scope = None
        if self.scope_of(scope) == "block_causal":
            block_scope = BlockCausalScope.build(batch, cfg, history)
        live = None if block_scope is None else block_scope.live_tokens
        live_boards = None if block_scope is None else block_scope.live_boards
        structure = build_structure(
            batch, cfg, coords, self.typed_edges, self.displacement, self.smolgen, live=live, live_boards=live_boards
        )
        tokens_per_board = max(lay.tokens_per_board for lay in batch.layouts)
        return AttnContext(
            cos=cos,
            sin=sin,
            body_len=batch.body_len,
            key_valid=batch.valid,
            mask=attn_mask,
            bias=attn_bias,
            compute_dtype=torch.bfloat16 if cfg.trunk_dtype == "bf16" else torch.float32,
            impl=cfg.attn_impl,
            chunk=cfg.attn_chunk,
            global_free=cfg.global_register_pairs == "position_free",
            spatial_free=spatial_free,
            spatial_dims=self.rope_spatial,
            structure=structure,
            block_rows=cfg.attn_block_boards * max(1, tokens_per_board),
            recompute=cfg.attn_recompute,
            sparse_max_fraction=cfg.attn_sparse_max_fraction,
            scope=block_scope,
        )

    # -- forward --

    def _autocast(self, device: torch.device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.cfg.trunk_dtype == "bf16")

    def trunk_output(self, batch: Batch, attn_mask=None, attn_bias=None, *, scope=None, history=None, kv_hook=None) -> torch.Tensor:
        """Final-normed trunk output ``(B, L, d)`` in FP32 (``scope``, ``history``,
        ``kv_hook``: see the module docstring). Under the block-causal scope with
        ``frozen_kv_format="fp8"`` every frozen board's K/V go through FP8 before attention
        (after ``kv_hook``, which sees them as computed)."""
        x = self.embed_tokens(batch, scope)
        ctx = self.make_context(batch, attn_mask, attn_bias, scope=scope, history=history)
        if self.scope_of(scope) == "block_causal" and self.cfg.frozen_kv_format != "match":
            kv_hook = _then(kv_hook, _frozen_kv_rounding(batch, self.cfg.frozen_kv_format))
        with self._autocast(x.device):
            h = self.trunk(x, ctx, kv_hook)
        return h.float()

    def forward(
        self, batch: Batch, attn_mask=None, attn_bias=None, return_trunk: bool = False, *, scope=None, history=None, kv_hook=None
    ) -> NetOutput:
        h = self.trunk_output(batch, attn_mask, attn_bias, scope=scope, history=history, kv_hook=kv_hook)
        return self.heads(h, batch, return_trunk=return_trunk)

    def heads(self, h: torch.Tensor, batch: Batch, return_trunk: bool = False) -> NetOutput:
        """Every head over a final-normed trunk output ``h`` ``(B, L, d)``, in FP32 (G1)."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            sq = batch.square_mask
            registers = global_register_pool(h, batch)  # the value head's input (D3)
            aux_in = registers if self.cfg.aux_global_pool == "global_registers" else square_pool(h, batch)
            boards = board_mean_pool(h, batch)
            possible, legal = batch.submit_mask()
            submit_raw = self.submit(h, batch, self.policy)  # D8: from a global register
            return NetOutput(
                policy_logits=self.policy(h, batch.moves, batch.move_valid, sq),
                value_logits=self.value(registers),
                wdl_logits=self.aux.wdl(aux_in),
                moves_left_log=self.aux.moves_left(aux_in).squeeze(-1),
                opp_reply_logits=self.aux.opp_reply(h, batch.opp_moves, batch.opp_move_valid, sq),
                short_value_logits=self.aux.short_value(aux_in),
                board_logits=self.aux.board(boards),
                square_logits=self.aux.square(h),
                effort_logits=self.aux.effort(aux_in),
                move_valid=batch.move_valid,
                opp_move_valid=batch.opp_move_valid,
                board_valid=batch.board_valid,
                square_mask=sq,
                token_board=batch.board,
                submit_logits=submit_raw.masked_fill(~legal.to(submit_raw.device), float("-inf")),
                submit_raw=submit_raw,
                submit_possible=possible,
                submit_valid=legal,
                trunk=h if return_trunk else None,
            )

    @torch.no_grad()
    def attention_probabilities(self, batch: Batch, layer: int = 0, attn_mask=None, attn_bias=None, *, scope=None) -> torch.Tensor:
        """Layer ``layer``'s attention probabilities ``(B, H, L, L)`` (explicit path)."""
        x = self.embed_tokens(batch, scope)
        ctx = self.make_context(batch, attn_mask, attn_bias, scope=scope)
        with self._autocast(x.device):
            return self.trunk.attention_probabilities(x, ctx, layer).float()


def _then(first, second):
    """A K/V hook that applies ``first`` (if any), then ``second``."""

    def hook(layer, k, v):
        if first is not None:
            k, v = first(layer, k, v)
        return second(layer, k, v)

    return hook


def _frozen_kv_rounding(batch: Batch, kv_format: str):
    """A K/V hook that passes every frozen board's K and V through ``kv_format`` - one scale
    per position, board and head, as the cache stores them (``kvformat.py``)."""
    layouts = batch.layouts
    frozen = batch.board_valid & ~batch.board_head  # (B, Nmax)
    tpbs = {lay.tokens_per_board for lay in layouts}

    def rounded(t: torch.Tensor) -> torch.Tensor:
        if len(tpbs) == 1:  # one board size: every board at once
            tpb = next(iter(tpbs))
            bsz, n_heads, _, dh = t.shape
            n_max = frozen.shape[1]
            body = t[:, :, : n_max * tpb].reshape(bsz, n_heads, n_max, tpb, dh)
            keep = frozen[:, None, :, None, None].to(t.device)
            body = torch.where(keep, fake_quantise(body, kv_format), body).reshape(bsz, n_heads, n_max * tpb, dh)
            return torch.cat([body, t[:, :, n_max * tpb :]], dim=2)
        out = t.clone()  # positions of different board sizes: board by board
        for i, lay in enumerate(layouts):
            for b in torch.nonzero(frozen[i]).flatten().tolist():
                sl = slice(b * lay.tokens_per_board, (b + 1) * lay.tokens_per_board)
                out[i : i + 1, :, sl] = fake_quantise(t[i : i + 1, :, sl], kv_format)
        return out

    def hook(layer, k, v):
        return rounded(k), rounded(v)

    return hook


def d8_seed(seed: int) -> int:
    """The seed of register D8's own generator (``T5DNet._init_weights``), from the model's."""
    return (int(seed) * 0x9E3779B97F4A7C15 + 0xD8) & 0xFFFFFFFFFFFFFFFF


def b15_seed(seed: int) -> int:
    """The seed of B15's own generator (its displacement table, ``T5DNet._init_weights``),
    from the model's."""
    return (int(seed) * 0x9E3779B97F4A7C15 + 0xB15) & 0xFFFFFFFFFFFFFFFF


def build_model(cfg: NetConfig | None = None, seed: int = 0) -> T5DNet:
    """A randomly initialised network, deterministic in ``seed`` (global RNG untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return T5DNet(cfg)


def parameter_breakdown(model: T5DNet) -> dict[str, int]:
    """Parameter counts by component, plus ``"total"``."""

    def count(mod: nn.Module | None) -> int:
        return 0 if mod is None else sum(p.numel() for p in mod.parameters())

    blocks = model.trunk.blocks
    out = {
        "embeddings": count(model.embed),
        "trunk.attention": sum(count(b.attn) for b in blocks),
        "trunk.ffn": sum(count(b.ffn) for b in blocks),
        "trunk.norms": sum(count(b.norm_attn) + count(b.norm_ffn) for b in blocks) + count(model.trunk.final_norm),
        "rope.register_xy": 0 if model.register_xy is None else model.register_xy.numel(),
        "attn.typed_edges": count(model.typed_edges),
        "attn.smolgen": count(model.smolgen),
        "attn.displacement": count(model.displacement),
        "head.policy": count(model.policy),
        "head.value": count(model.value),
        "head.submit": count(model.submit),
    }
    for name, mod in model.aux.named_children():
        out[f"aux.{name}"] = count(mod)
    out["total"] = sum(p.numel() for p in model.parameters())
    assert out["total"] == sum(v for k, v in out.items() if k != "total")
    return out
