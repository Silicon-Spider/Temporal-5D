"""The dense v1 network: embeddings, trunk, heads (stage N1 of the untrained build).

``build_model(config, seed)`` returns a ``T5DNet`` with random weights at realistic
initialisation scale (spec section 15.2: "never zeros or near-zeros"), deterministically
from the seed. ``T5DNet(batch)`` returns a ``NetOutput`` with every head.

Initialisation (implementation choice; the spec is silent):

* Linear weights ``N(0, 1/fan_in)`` (LeCun normal) - attention logits then have
  roughly unit spread at initialisation, so attention is far from uniform; the two
  residual-output projections of every block (attention out, FFN out) are further
  scaled by ``1/sqrt(2L)`` (GPT-2), keeping the residual stream bounded over depth.
* Linear biases ``N(0, init_bias_std)``; embeddings and the "not a square" vector
  ``N(0, init_embed_std)``; LayerNorm gain ``1 + N(0, init_norm_std)`` and bias
  ``N(0, init_bias_std)`` - near the identity, but not it, so that no parameter is
  zero or constant (section 15.2: zeros "make distinct paths agree trivially").
* B6 bias tables and B11's typed-edge scalars ``N(0, init_table_std)``; learned register
  coordinates (option ``learned_angle``) at the board centre plus ``N(0, init_table_std)``
  noise; B12's weights by ``BoardPairSmolgen.reset_parameters`` (biases of spread of order
  one at initialisation).
* The learned register coordinates and the structure's flag-gated modules draw last, in
  the order register coordinates, B6, B11, B12, and none of them draws from the RNG while
  it is constructed: switching one on changes its own weights and those of the
  flag-gated modules drawn after it, never the trunk's or the heads' - so an on/off
  comparison of B12, say, differs in B12 alone.

Attention structure (stage N2, ``structure.py``): ``make_context`` builds the batch's
masks (B9's move graph on ``move_graph_heads`` heads, or B10's routing table) and biases
(B6, B11, B12) once per forward, as sparse lists and board-pair tables; every layer's
attention builds dense rows from them one query block at a time (B14).

Heads (``T5DNet.heads``) read the final-normed trunk: the policy pointer at each move's
source and destination tokens (D1), the value head the mean of the global registers
(D3, sections 7.3 and 8.3), the global auxiliary heads the pool ``aux_global_pool``
selects (an open committed-class point), per-board heads each board's square mean and
per-square heads every token (``heads.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .attention import AttnContext
from .config import NetConfig
from .embed import InputEmbedding
from .heads import (
    AuxHeads,
    PointerHead,
    ValueHead,
    board_mean_pool,
    categorical_mean,
    categorical_probs,
    categorical_variance,
    global_register_pool,
    square_pool,
)
from .rope import BinnedBias, rope_angles, rope_cos_sin, rope_tables, spatial_dims
from .smolgen import BoardPairSmolgen
from .state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, Batch
from .structure import TypedEdgeBias, build_structure
from .trunk import Trunk


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
    trunk: torch.Tensor | None = None  # (B, L, d) final-normed trunk output, if requested

    def value_probs(self) -> torch.Tensor:
        return categorical_probs(self.value_logits)

    def value_mean(self) -> torch.Tensor:
        return categorical_mean(self.value_probs())

    def value_variance(self) -> torch.Tensor:
        return categorical_variance(self.value_probs())

    def policy_log_probs(self) -> torch.Tensor:
        """Log-softmax over each position's supplied moves (-inf at padding)."""
        return torch.log_softmax(self.policy_logits, dim=-1).masked_fill(~self.move_valid, float("-inf"))


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
        self.binned_bias = BinnedBias(cfg) if cfg.binned_bias else None
        self.typed_edges = TypedEdgeBias(cfg) if cfg.typed_edge_bias else None
        self.smolgen = BoardPairSmolgen(cfg) if cfg.board_pair_smolgen else None
        self.policy = PointerHead(cfg.d_model, cfg.policy_dim, promotion=cfg.policy_promotion)
        self.value = ValueHead(cfg)
        self.aux = AuxHeads(cfg)
        self._init_weights()

    # -- initialisation --

    def _init_weights(self) -> None:
        cfg = self.cfg
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        with torch.no_grad():
            for name, module in self.named_modules():
                if name.split(".")[0] == "smolgen":
                    continue  # drawn last (reset_parameters)
                if isinstance(module, nn.Linear):
                    std = module.in_features**-0.5
                    if name.endswith("attn.out_proj") or name.endswith("ffn.lin_out"):
                        std *= residual_scale
                    nn.init.normal_(module.weight, 0.0, std)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, 0.0, cfg.init_bias_std)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, 0.0, cfg.init_embed_std)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.normal_(module.weight, 1.0, cfg.init_norm_std)
                    nn.init.normal_(module.bias, 0.0, cfg.init_bias_std)
            nn.init.normal_(self.embed.not_square, 0.0, cfg.init_embed_std)
            if self.register_xy is not None:
                self.register_xy.normal_(0.0, cfg.init_table_std)
                self.register_xy.add_(3.5)  # board centre: off-lattice, at no square
            if self.binned_bias is not None:
                nn.init.normal_(self.binned_bias.time_table, 0.0, cfg.init_table_std)
                nn.init.normal_(self.binned_bias.timeline_table, 0.0, cfg.init_table_std)
            if self.typed_edges is not None:
                nn.init.normal_(self.typed_edges.table, 0.0, cfg.init_table_std)
            if self.smolgen is not None:
                self.smolgen.reset_parameters(cfg)

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

    def make_context(self, batch: Batch, attn_mask: torch.Tensor | None = None, attn_bias: torch.Tensor | None = None) -> AttnContext:
        """The attention context of a batch: RoPE tables, padding, the structure of stage N2
        (``structure.py``: B9/B10 masks, B6/B11/B12 biases - built sparsely, rows on demand),
        and optionally a caller's dense ``attn_mask`` / ``attn_bias`` on top."""
        cfg = self.cfg
        coords = self.coordinates(batch)
        cos, sin = rope_cos_sin(rope_angles(coords, self.rope_axis, self.rope_freq))
        spatial_free = None
        if cfg.board_register_spatial == "position_free":
            spatial_free = batch.valid & (batch.kind == TOKEN_BOARD_REGISTER)
        structure = build_structure(batch, cfg, coords, self.typed_edges, self.binned_bias, self.smolgen)
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
        )

    # -- forward --

    def _autocast(self, device: torch.device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.cfg.trunk_dtype == "bf16")

    def trunk_output(self, batch: Batch, attn_mask=None, attn_bias=None) -> torch.Tensor:
        """Final-normed trunk output ``(B, L, d)`` in FP32."""
        x = self.embed(batch)
        ctx = self.make_context(batch, attn_mask, attn_bias)
        with self._autocast(x.device):
            h = self.trunk(x, ctx)
        return h.float()

    def forward(self, batch: Batch, attn_mask=None, attn_bias=None, return_trunk: bool = False) -> NetOutput:
        return self.heads(self.trunk_output(batch, attn_mask, attn_bias), batch, return_trunk=return_trunk)

    def heads(self, h: torch.Tensor, batch: Batch, return_trunk: bool = False) -> NetOutput:
        """Every head over a final-normed trunk output ``h`` ``(B, L, d)``, in FP32 (G1)."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            sq = batch.square_mask
            registers = global_register_pool(h, batch)  # the value head's input (D3)
            aux_in = registers if self.cfg.aux_global_pool == "global_registers" else square_pool(h, batch)
            boards = board_mean_pool(h, batch)
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
                trunk=h if return_trunk else None,
            )

    @torch.no_grad()
    def attention_probabilities(self, batch: Batch, layer: int = 0, attn_mask=None, attn_bias=None) -> torch.Tensor:
        """Layer ``layer``'s attention probabilities ``(B, H, L, L)`` (explicit path)."""
        x = self.embed(batch)
        ctx = self.make_context(batch, attn_mask, attn_bias)
        with self._autocast(x.device):
            return self.trunk.attention_probabilities(x, ctx, layer).float()


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
        "rope.binned_bias": count(model.binned_bias),
        "attn.typed_edges": count(model.typed_edges),
        "attn.smolgen": count(model.smolgen),
        "head.policy": count(model.policy),
        "head.value": count(model.value),
    }
    for name, mod in model.aux.named_children():
        out[f"aux.{name}"] = count(mod)
    out["total"] = sum(p.numel() for p in model.parameters())
    assert out["total"] == sum(v for k, v in out.items() if k != "total")
    return out
