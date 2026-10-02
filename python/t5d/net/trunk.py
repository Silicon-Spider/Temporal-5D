"""The transformer trunk (B1, B2, C1, F1; spec sections 3, 7.4, 9.2).

One weight set; L identical pre-norm residual blocks, each unified dense bidirectional
attention over every token followed by a two-matrix FFN of width ``ffn_ratio * d``
(section 9.1's ``32 T d^2 L`` FFN cost at 8x is the two-matrix form). A final norm
precedes the heads.

Implementation choices (the spec is silent; none is committed-class):

* pre-norm residual blocks, ``x + Attn(LN(x))`` then ``x + FFN(LN(x))``;
* LayerNorm (with affine parameters, eps 1e-5);
* GELU (exact erf form) inside the FFN;
* Linear layers with biases.

Precision (G1): under ``trunk_dtype="bf16"`` the trunk runs inside autocast, so its
matmuls and the attention kernel run in BF16 while the residual stream, the norms and
the master weights stay FP32; RoPE is FP32 regardless (``rope.py``).

Each block hands its attention the context of its own layer (``AttnContext.for_layer``):
board-pair smolgen (B12) generates that layer's biases from the block's normed input.

``kv_hook(layer, k, v) -> (k, v)`` (stage N3) sees each layer's keys and values before
attention: the cached path captures them (a refresh, the root history) and the reference
of the two-level mask injects a refreshed history through it (``cache.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from .attention import AttnContext, SelfAttention
from .config import NetConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.lin_in = nn.Linear(cfg.d_model, cfg.d_ffn, bias=cfg.linear_bias)
        self.lin_out = nn.Linear(cfg.d_ffn, cfg.d_model, bias=cfg.linear_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_out(torch.nn.functional.gelu(self.lin_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = SelfAttention(cfg)
        self.norm_ffn = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, ctx: AttnContext, kv_hook: Callable | None = None) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.attn(h, ctx.for_layer(h), kv_hook)
        return x + self.ffn(self.norm_ffn(x))


class Trunk(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self, x: torch.Tensor, ctx: AttnContext, kv_hook: Callable | None = None) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x, ctx, None if kv_hook is None else partial(kv_hook, i))
        return self.final_norm(x.float())

    def attention_probabilities(self, x: torch.Tensor, ctx: AttnContext, layer: int) -> torch.Tensor:
        """Run blocks ``[0, layer)`` and return block ``layer``'s attention probabilities."""
        for block in self.blocks[:layer]:
            x = block(x, ctx)
        block = self.blocks[layer]
        h = block.norm_attn(x)
        return block.attn.probabilities(h, ctx.for_layer(h))
