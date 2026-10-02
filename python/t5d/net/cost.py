"""Section 15.3's cost model (stage N3): analytic FLOPs and K/V bytes, and FLOPs counted
from the implementation.

**Analytic** - the spec's own model (sections 9.1, 9.2, 7.1, 7.3, 7.6), applied to the
token counts of the configuration measured: attention ``4 * T_q * T_k * d * L`` (Q.K^T and
P.V, two FLOPs per multiply-add) and the linear terms ``(8 + 4 r) * T_q * d^2 * L`` (the Q,
K, V and output projections, and the two FFN matrices at ratio ``r``; ``40 T d^2 L`` at
r = 8). At 240 boards (T = 15,856, d = 256, L = 16) the dense figures are section 9.2's
4.12 + 0.67 = 4.78 TFLOPs, and with 8 live boards plus 16 global registers (544 queries)
the cached node is 0.164 TFLOPs - section 7.6's "pure cached 0.16" (section 7.9's table
uses 0.159, hence its +3.8% where section 15.3 says +3.6%). Embeddings, norms, softmax and
the output heads are left out, as the spec leaves them out. These are not measurements.

**Counted** - ``count_flops`` runs a callable under ``torch.utils.flop_counter``'s
``FlopCounterMode``, which counts every matmul the implementation actually issues. PyTorch
2.13 has no formula for the CPU fused attention op (``_scaled_dot_product_flash_attention_
for_cpu``), so one is registered here: ``4 * B * H * L_q * L_k * d_head`` per call - the
kernel's work over every key it is handed, masked or not. The counted figure therefore
includes what the cost model leaves out on purpose - masked scores the chunked path
computes anyway, the global-register merge's extra key, padded value widths - and is the
implementation's FLOP count, beside the model's.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode, flop_registry, register_flop_formula, sdpa_flop_count

from .config import NetConfig

_CPU_SDPA = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu


def _register_cpu_sdpa() -> None:
    if _CPU_SDPA in flop_registry:
        return

    @register_flop_formula(_CPU_SDPA)
    def _cpu_sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
        return sdpa_flop_count(query_shape, key_shape, value_shape)


_register_cpu_sdpa()


def count_flops(fn: Callable[[], object]) -> tuple[object, int]:
    """``fn()`` and the FLOPs of every matmul it issued (CPU fused attention included)."""
    with FlopCounterMode(display=False) as counter:
        result = fn()
    return result, int(counter.get_total_flops())


def attention_flops(t_query: int, t_key: int, cfg: NetConfig) -> int:
    """``4 * T_q * T_k * d * L``: scores and the weighted sum, every layer (section 9.1)."""
    return 4 * t_query * t_key * cfg.d_model * cfg.n_layers


def linear_flops(t_query: int, cfg: NetConfig) -> int:
    """``(8 + 4 r) * T_q * d^2 * L``: the four projections and the FFN, every layer."""
    return (8 + 4 * cfg.ffn_ratio) * t_query * cfg.d_model**2 * cfg.n_layers


def pass_flops(t_query: int, t_key: int, cfg: NetConfig) -> dict[str, int]:
    """Analytic FLOPs of ``t_query`` queries against ``t_key`` keys through the whole trunk."""
    attn, lin = attention_flops(t_query, t_key, cfg), linear_flops(t_query, cfg)
    return {"attention": attn, "linear": lin, "total": attn + lin}


def dense_flops(n_tokens: int, cfg: NetConfig) -> dict[str, int]:
    """One dense evaluation: every token against every token (section 9.2: 4.78 TF at 240 boards)."""
    return pass_flops(n_tokens, n_tokens, cfg)


def kv_bytes(n_tokens: int, cfg: NetConfig, bytes_per_value: float) -> float:
    """K and V of ``n_tokens`` tokens at every layer (section 10.3: ~260 MB at 240 boards in BF16)."""
    return n_tokens * cfg.n_layers * 2 * cfg.d_model * bytes_per_value
