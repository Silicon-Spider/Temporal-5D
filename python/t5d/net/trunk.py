"""The transformer trunk (B1, B2, C1, F1, F5; spec sections 3, 7.4, 9.2, 9.3).

One weight set; L identical residual blocks, each unified dense bidirectional attention
over every token followed by a two-matrix FFN of width ``ffn_ratio * d`` (section 9.1's
``32 T d^2 L`` FFN cost at 8x is the two-matrix form).

**Normalisation and biases (F5, section 9.3).** Every norm is RMSNorm-style
(``RMSNorm``: a learned gain over the root mean square - no mean-centring, no bias); the
Q, K and V projections have no biases (``NetConfig.qkv_bias``); the attention output
projection and both FFN matrices keep theirs (``NetConfig.out_ffn_bias``: section 9.3's
Chessformer networks "also keep their FFN and attention-output biases, as here"). Where
the norms sit is ``NetConfig.norm_placement``:

* ``"pre"`` (the v1): ``x + Attn(N(x))`` then ``x + FFN(N(x))``, and a final norm before
  the heads;
* ``"post"`` (ablation 6y's arm, "Lc0's post-norm with DeepNet initialisation"): DeepNorm
  (Wang et al. 2022), ``x <- N(alpha x + Attn(x))`` then ``x <- N(alpha x + FFN(x))``
  with the constant ``alpha = (2L)^(1/4)`` and DeepNet's initialisation
  (``model._init_weights``). No final norm - the last block's FFN norm is the output's -
  and no norm on the embedding before block 0 (placeholders, ``NetConfig.norm_placement``).

Implementation choice (the spec is silent; not committed-class): GELU (exact erf form)
inside the FFN.

Precision (G1): under ``trunk_dtype="bf16"`` the trunk runs inside autocast, so its
matmuls and the attention kernel run in BF16 while the residual stream, the norms and
the master weights stay FP32 (``RMSNorm`` turns autocast off and computes in FP32); RoPE
is FP32 regardless (``rope.py``).

Each block hands its attention the context of its own layer (``AttnContext.for_layer``):
board-pair smolgen (B12) generates that layer's biases from the attention's input - the
block's normed input under pre-norm, the block's input under post-norm.

A block is four steps - ``Block.attn_input``, the attention, ``Block.after_attn``,
``Block.after_ffn`` - and the trunk ends in ``Trunk.output``. The cached path
(``cache.py``) runs the same steps around its own attention, so that it computes the
trunk's function under either placement.

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


class RMSNorm(nn.Module):
    """``x / sqrt(mean(x^2) + eps) * g`` over the last dimension: F5's RMSNorm-style layer -
    no mean-centring, no bias, a learned gain ``g`` (Zhang & Sennrich 2019, "Root Mean Square
    Layer Normalization"; eps inside the root, as in PyTorch's own ``rms_norm``).

    G1: computed in FP32 with autocast off whatever the input's dtype; the output is FP32."""

    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))  # the gain; drawn by ``model._init_weights``

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.lin_in = nn.Linear(cfg.d_model, cfg.d_ffn, bias=cfg.out_ffn_bias)
        self.lin_out = nn.Linear(cfg.d_ffn, cfg.d_model, bias=cfg.out_ffn_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin_out(torch.nn.functional.gelu(self.lin_in(x)))


class Block(nn.Module):
    """One residual block: the attention sublayer, then the FFN sublayer, each with its
    RMSNorm (``norm_attn``, ``norm_ffn``) placed by ``NetConfig.norm_placement``."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.post_norm = cfg.norm_placement == "post"
        #: DeepNorm's residual up-scaling (post-norm; unused under pre-norm).
        self.alpha = cfg.deepnorm_alpha if self.post_norm else 1.0
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SelfAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def attn_input(self, x: torch.Tensor) -> torch.Tensor:
        """What the attention (and board-pair smolgen) reads from the residual stream ``x``:
        ``norm_attn(x)`` under pre-norm, ``x`` itself under post-norm."""
        return x if self.post_norm else self.norm_attn(x)

    def after_attn(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """The residual stream after the attention sublayer, whose output is ``a``."""
        return self.norm_attn(self.alpha * x + a) if self.post_norm else x + a

    def after_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """The residual stream after the FFN sublayer."""
        if self.post_norm:
            return self.norm_ffn(self.alpha * x + self.ffn(x))
        return x + self.ffn(self.norm_ffn(x))

    def forward(self, x: torch.Tensor, ctx: AttnContext, kv_hook: Callable | None = None) -> torch.Tensor:
        h = self.attn_input(x)
        return self.after_ffn(self.after_attn(x, self.attn(h, ctx.for_layer(h), kv_hook)))


class Trunk(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        #: The norm before the heads - pre-norm only: under post-norm the last block's FFN
        #: norm is the output's (placeholder, ``NetConfig.norm_placement``).
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.norm_placement == "pre" else None

    def output(self, x: torch.Tensor) -> torch.Tensor:
        """The trunk's output (FP32) from the last block's residual stream ``x``."""
        return x.float() if self.final_norm is None else self.final_norm(x)

    def forward(self, x: torch.Tensor, ctx: AttnContext, kv_hook: Callable | None = None) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x, ctx, None if kv_hook is None else partial(kv_hook, i))
        return self.output(x)

    def attention_probabilities(self, x: torch.Tensor, ctx: AttnContext, layer: int) -> torch.Tensor:
        """Run blocks ``[0, layer)`` and return block ``layer``'s attention probabilities."""
        for block in self.blocks[:layer]:
            x = block(x, ctx)
        block = self.blocks[layer]
        h = block.attn_input(x)
        return block.attn.probabilities(h, ctx.for_layer(h))
