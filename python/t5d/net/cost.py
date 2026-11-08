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

**Register A5** (``token_counts``, ``piece_centric_costs``): the same model at piece-centric token
counts - the dense pass, the analytic cached node and the frozen K/V, A1 against A5.

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


def qk_flops(t_query: int, t_key: int, cfg: NetConfig) -> int:
    """The ``Q K^T`` product alone: ``2 * T_q * T_k * d_head * H * L`` (half of
    ``attention_flops``) - what section 15.3's B17 row compares the Shaw term with."""
    return 2 * t_query * t_key * cfg.d_head * cfg.n_heads * cfg.n_layers


def shaw_flops(t_query: int, cfg: NetConfig, n_classes: int | None = None) -> int:
    """B17's precomputed query-class products (``shaw.gather_term``): ``2 * T_q * C * d_head * H *
    L`` - every query against every one of the ``C`` classes (``shaw.n_shaw_classes(cfg)`` when
    None), on every head and layer; the gather that follows is no multiply. Against
    ``qk_flops`` it is ``C / T_k``: section 15.3's "~0.5x the QK^T product at 44 boards, ~9% at
    240, with ~1,400 classes" (0.51 and 0.094 with the placeholder's 1,493)."""
    if n_classes is None:
        from .shaw import n_shaw_classes

        n_classes = n_shaw_classes(cfg)
    return 2 * t_query * n_classes * cfg.d_head * cfg.n_heads * cfg.n_layers


def layout_flops(n_boards: int, n_tokens: int, cfg: NetConfig) -> dict[str, int]:
    """Register A8's board-layout embedding (``layout.py``), as the implementation issues it for one
    position of ``n_boards`` boards and ``n_tokens`` tokens: the projection ``2 x 1,536 x 64 C`` per
    board (the dense one-hot product - section 15.3's "~0.1% of attention at 44 boards" is this
    term), the merge ``2 C d`` per token (every token: registers' are masked after) and the FFN
    ``4 d^2 r`` per token (``layout_ffn_ratio`` r; 0 without it). Norms, the one-hot and the gather
    are no multiply-adds. Every term is zero when ``cfg``'s flag (``board_layout_embedding``) is
    off: the network then has no A8.

    The FFN is per token, so it is paid wherever a position's tokens are embedded: in the dense
    pass over every token, and in the cached configuration at every node - ``CachedEvaluator.
    prepare`` embeds all of the node's tokens, frozen boards' included - where it is a larger share
    of the node than of the dense pass (``docs/MEASUREMENTS.md``, "A8"). ``dense_flops`` and
    ``pass_flops`` are section 9.1's model and leave A8 out, as they leave out every embedding."""
    from .layout import LAYOUT_CELLS, LAYOUT_FEATURES

    c, d, r = cfg.layout_channels, cfg.d_model, cfg.layout_ffn_ratio
    out = {
        "projection": 2 * n_boards * LAYOUT_FEATURES * LAYOUT_CELLS * c,
        "merge": 2 * n_tokens * c * d,
        "ffn": 4 * n_tokens * d * (r * d),
    }
    if not cfg.board_layout_embedding:
        out = dict.fromkeys(out, 0)
    out["total"] = sum(out.values())
    return out


# --- register A5: piece-centric tokens (section 14.4; state.to_piece_centric) -----------------------------


def token_counts(state, regs_per_board: int | None = None, global_regs: int | None = None) -> dict[str, int]:
    """One position's tokens under A1 (one per square) and under register A5 (its pieces and the
    distinct empty cells its candidate moves land on - ``state.to_piece_centric``'s set), from a
    square-centric or a piece-centric ``PositionState``; the register counts its own unless given.
    Keys: ``boards``, ``heads`` (the live set's boards under the block-causal scope), ``pieces``,
    ``destinations`` (D: the empty destination cells - a capture lands on a piece's token),
    ``head_pieces``, ``head_destinations`` (D on head boards), ``a1``, ``a5`` (``T``), and the
    analytic cached configuration's sets (``cached_*``): the live queries - A1 the heads' tokens and
    the global registers, A5 the heads' pieces and registers, every destination token (transient,
    never cached - BUILD_LOG issue 252's reading (f)) and the global registers - and the frozen
    boards' cached tokens."""
    r = state.regs_per_board if regs_per_board is None else int(regs_per_board)
    g = state.global_regs if global_regs is None else int(global_regs)
    occupied = state.piece != 0
    if state.token_cells is not None:
        cells = state.token_cells
    else:
        cells = occupied.copy()
        if state.moves.size:
            board, cell = state.layout.split_square_tokens(state.moves[:, 1])
            cells[board, cell] = True
    dest = cells & ~occupied
    head = state.is_head
    n, wh = state.n_boards, int(state.width) * int(state.height)
    heads = int(head.sum())
    pieces, d = int(occupied.sum()), int(dest.sum())
    head_pieces, head_d = int(occupied[head].sum()), int(dest[head].sum())
    return {
        "boards": n,
        "heads": heads,
        "pieces": pieces,
        "destinations": d,
        "head_pieces": head_pieces,
        "head_destinations": head_d,
        "a1": n * (wh + r) + g,
        "a5": pieces + d + n * r + g,
        "cached_live_a1": heads * (wh + r) + g,
        "cached_live_a5": head_pieces + heads * r + d + g,
        "cached_frozen_a1": (n - heads) * (wh + r),
        "cached_frozen_a5": (pieces - head_pieces) + (n - heads) * r,
    }


def piece_centric_costs(counts: dict[str, int], cfg: NetConfig, bytes_per_value: float = 2.0) -> dict[str, float]:
    """Section 9.1's model at ``token_counts``' figures, A1 against A5: the dense evaluation
    (``dense_flops``), the analytic cached node (the live queries against every token -
    ``pass_flops``; A5's destination tokens live, BUILD_LOG issue 252) and the frozen boards' K/V
    bytes per state (``kv_bytes``), with each ratio A1 / A5. Analytic: A5 under the block-causal
    scope is refused as built (issue 252), so its cached figures are this model's only."""
    out: dict[str, float] = {}
    for arm in ("a1", "a5"):
        t = counts[arm]
        out[f"dense_{arm}"] = dense_flops(t, cfg)["total"]
        out[f"dense_attention_{arm}"] = dense_flops(t, cfg)["attention"]
        out[f"node_{arm}"] = pass_flops(counts[f"cached_live_{arm}"], t, cfg)["total"]
        out[f"kv_{arm}"] = kv_bytes(counts[f"cached_frozen_{arm}"], cfg, bytes_per_value)
    for what in ("dense", "dense_attention", "node", "kv"):
        out[f"{what}_ratio"] = out[f"{what}_a1"] / out[f"{what}_a5"] if out[f"{what}_a5"] else float("inf")
    out["token_ratio"] = counts["a1"] / counts["a5"]
    return out
