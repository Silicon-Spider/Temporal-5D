"""Multi-head self-attention over all tokens of a position (B1, B2, C1, C4, B14).

One weight set, unified dense bidirectional attention over every token (spec sections
3.1-3.2, 7.4), full MHA with 8 heads of 32 (C4, section 10.2). Connectivity is a mask,
never a weight (B1): callers may pass any boolean mask and any additive per-head bias,
and stage N2's structure (``structure.py``: the move-graph mask, the routing table, typed
edge biases, board-pair biases, and B15's displacement bias - per layer: each layer's
context comes from ``AttnContext.for_layer``) supplies its masks and biases one query
block at a time.

**Positions (section 4.1, B4).** Q and K are rotated in FP32 (G1) before the kernel. A
token with no coordinate on an axis cannot be rotated on it under a relative encoding (a
fixed angle on one side is itself a position, ``rope.py``), so every pair that involves
such a token uses that axis's *unrotated* Q/K dims on both sides. The pair rule, for
query ``i`` and key ``j``: an axis is rotated iff both tokens have it. Two token classes
lack axes (B8 and section 5.3, ruled at CHECKPOINT 6 - M20):

* global registers (the tokens ``[body_len, L)``) lack all four axes, so their pairs use
  unrotated Q and K;
* per-board registers lack file and rank, so their pairs use unrotated file/rank dims and
  rotated timeline/time dims (``spatial_free``).

**Masks and biases.** ``key_valid`` (padding) always; ``mask`` (bool, broadcastable to
``(B, H, L, L)``, True = may attend) and ``bias`` (float, same shape) optionally, from the
caller; ``structure`` (``AttnStructure``), which builds the rows of its masks and biases
for any query block ``[start, stop)`` from sparse edge lists and small board-pair tables;
and ``scope`` (stage N3, ``scope.BlockCausalScope``: the block-causal scope, C17/C19),
whose rows come from a board-slot table the same way (``_block_rows`` combines all four;
the scope enters wherever the caller's mask does). A query row with no allowed key
outputs zeros, and its gradient stays finite; neither the structure nor the scope ever
empties a row. B17's key-side Shaw term (``ctx.shaw``, behind its flag; ``shaw.py``) depends
on the queries, so it is not one of the structure's rows: every path adds it beside them, from
the rows' unrotated queries - gathered from the precomputed query-class products, block by
block (any key selection, any row set), or pair by pair in the dense reference (section 15.2's
B17 row compares the two).

Six implementations compute the same function on every valid query
(``NetConfig.attn_impl``; a padding query's output is discarded, and the block-sparse
path does not compute it the reference's way):

* ``"block_sparse"`` (B14, the default): ``torch.nn.functional.scaled_dot_product_attention``
  (the kernel) over query blocks of whole boards (board-major order, section 6.1), the
  heads grouped by mask role. A masked role's call gets only the keys of the boards that
  the block's queries can reach (``AttnStructure.key_selection``, lossless: whole boards
  are the blocks, each block's mask and bias are gathered on those keys), so a masked head
  costs what its reach costs instead of a dense head plus a mask - section 6.1's
  block-sparse kernel, done by gathering key boards (one board is 66 contiguous tokens)
  instead of skipping key tiles. A query that reads every key on a masked head (a
  register under ``masked_head_registers="read"``) is computed apart, against every key,
  so that it cannot select every board for its block; a padding query is computed
  against its block's selection (zeros where there is none) and discarded. Dense heads
  read every key: in one call over all rows when nothing carries a per-row mask or
  bias, else block by block.
* ``"chunked"`` (ablation 6's dense-mask arm): the same query blocks against every key,
  masked keys masked rather than skipped. When nothing carries a bias, heads that share a
  mask role share one kernel call per block with a single ``(B, 1, block, L)`` mask and
  the dense heads run in one call over all rows; otherwise every head runs in one call
  per block with ``(B, H, block, L)`` masks and biases.

  For both, a block's masks and biases exist for its rows only, so a forward pass holds
  ``O(H * block * L)`` attention memory at a time - no ``(L, L)`` tensor per head exists
  at any point. Under autograd that holds between the passes too only because each block
  is recomputed in the backward pass (``recompute``, ``NetConfig.attn_recompute``, on by
  default: ``torch.utils.checkpoint`` around every block that carries a per-row mask or
  bias); without it the kernel keeps each block's mask - and, once a learned bias makes
  the mask require grad, the math kernel each block's probabilities - for the backward
  pass: ``O(H L^2)`` per layer. The dense heads' one call keeps only ``O(L)`` per head.
  Inside a block:

  - *Position-free global keys* are folded in exactly. For each kernel query ``i`` the
    global block's scores ``s_ig`` (unrotated) are computed explicitly, and one extra
    key is appended whose score is ``t_i = logsumexp_g s_ig`` (carried by one extra
    query dimension) and whose value is a unit marker. The kernel's softmax over
    ``[kernel keys, extra key]`` then has exactly the full softmax's normaliser, the
    marker channel returns the global block's total weight ``w_i``, and the output is
    ``o_i = o_kernel_i + w_i * softmax(s_i,global) V_global`` - identical to attending to
    all keys at once. Global-register queries (16 rows) are computed explicitly.
  - *Keys without file/rank* are handled by widening the features: queries
    ``[q_rot | q_fr]``, keys ``[k_rot with its file/rank dims zeroed if the key lacks
    them | k_fr if the key lacks them, else 0]``, where ``_fr`` are the unrotated
    file/rank dims. A key with file/rank scores ``q_rot . k_rot``; one without scores
    ``q_rot,time . k_rot,time + q_fr . k_fr`` - the pair rule. Queries without file/rank
    (2 per board) are recomputed by a second kernel call over their rows only, with Q
    and K rotated on timeline/time alone.
  - Values are zero-padded to the query/key width: PyTorch's fast CPU kernels need
    equal widths (unequal widths fall back to a path about 3x slower).
  - The kernel's widths (register G11; section 10.2, "Kernel-friendly widths"): as built
    the widened features are 45 wide - 32 rotated dims, the 12 unrotated file/rank dims,
    the merge's one - and 33 on the rows without file and rank; a masked fused kernel may
    want a multiple of 8 (BUILD_LOG issue 521). ``AttnContext.pad_multiple``
    (``NetConfig.attn_pad_multiple``; 1, the default in this build: as built) pads keys
    and values with zero columns to that multiple where they are assembled
    (``_KernelKeys.build``), and each call's queries follow the keys' width
    (``_kernel``): 48 and 40 at 8. Exact - a zero feature adds zero to every score, and
    the padded value columns are cut from the output.

* ``"sdpa"``: the chunked computation in one block (N1's fast path). Without masks or
  biases it is one kernel call; with them it materialises ``(B, H, L, L)`` rows - small
  inputs only.

  Cost of one attention call over one plain kernel call (every pair rotated, as the arms
  M20 removed - anchored global registers with learned- or zero-angle per-board registers -
  computed it), FP32 / BF16, on the development CPU (single position, median of 5-9 calls,
  PyTorch 2.13; indicative timings, not section 15.3 measurements): the global merge alone
  +28% / +7% at 60 boards and +15% / +8% at 240; position-free per-board registers alone
  +36% / +14% at 60 and +37% / +15% at 240; both - the pair rule as ruled (B8), now the
  only one - +59% / +26% and +55% / +20%. The widened features (width 44 instead of 32 for
  the default allocation) are most of it.
* ``"explicit"``: materialised scores, chunked over queries (``chunk`` rows), each
  chunk's mask and bias rows built for it alone.
* ``"dense"``: every mask and bias materialised at ``(B, H, L, L)`` first
  (``AttnContext.densified``), then the explicit computation - the reference the tests
  compare the others with (B17's term computed there pair by pair, ``shaw.direct_term``).
* ``"flex"`` (B14's score function, ``flex.py``): torch's ``flex_attention`` with a score
  function that computes every bias inside the attention - B15's from the tokens'
  coordinates, B11's from the pair's typed relations, B12's from the layer's board-pair
  table - and a mask function for every mask, with a block mask in board-major order.
  Eager on CPU: PyTorch 2.13's FlexAttention has no CPU backward and, uncompiled,
  materialises every score - so this path proves the in-kernel formulation exact on small
  positions (section 15.2's B15 row), and the compiled GPU kernel and its speed wait for GPU
  time (G9, M19).

Indicative cost (``bench.py``, whose docstring has the table; development CPU, PyTorch
2.13, one forward of the 16-layer network, one position, FP32; not section 15.3
measurements): at 240 boards B9's default takes 9.0 s block-sparse and 13.8 s chunked
against 8.5 s with no mask, every switch of stage N2 29.5 s and 39.3 s; peak memory stays
under 1.1 GB on both paths, against 3.4 GB for B9's default through the unchunked fused
path. The block-sparse plans of a forward hold about 58 MB per sparse role at 240 boards
(the move graph's blocks select 24% of the key boards on average). The timings and peak
memory above were measured on the network before F5 and F6 (LayerNorm with biases, biased
Q/K/V projections, GELU in the FFN) and before B15, as ``bench.py``'s table was - "every
switch" with B6's binned biases on, since removed - and have not been re-measured since;
the plans' size does not depend on any of them. B15 puts a learned bias on every head, so
v1's dense heads now run block by block with a per-row bias, as "every switch" did - and,
under autograd, in the CPU math kernel. They are also of the network before M20 made the
per-board registers position-free on file and rank (B8; until then they sat at a learned
off-lattice angle and only the global registers were merged): every row's features are now
widened and the register rows recomputed - by the per-call timings above, at 240 boards about
+35% / +11% per attention call (FP32 / BF16) over that default, whose merge alone cost +15% /
+8% against one plain call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import NetConfig
from .quant import AttnQuant, QuantLinear
from .rope import apply_rope
from .structure import memoised

if TYPE_CHECKING:
    from .scope import BlockCausalScope
    from .shaw import ShawRows
    from .structure import AttnStructure

NEG_INF = float("-inf")


@dataclass
class AttnContext:
    """Everything attention needs besides the residual stream; shared by all layers
    (``for_layer`` adds what one layer's structure generates from its own input)."""

    cos: torch.Tensor  # (B, Hr, L, d_head) fp32
    sin: torch.Tensor  # (B, Hr, L, d_head) fp32
    body_len: int  # global registers occupy [body_len, L)
    key_valid: torch.Tensor  # (B, L) bool
    mask: torch.Tensor | None = None  # bool, broadcastable to (B, H, L, L)
    bias: torch.Tensor | None = None  # float, broadcastable to (B, H, L, L)
    compute_dtype: torch.dtype = torch.float32
    impl: str = "sdpa"
    chunk: int = 1024
    #: (B, L) bool: tokens that lack file and rank - the per-board registers (B8; None: no
    #: such token). The global registers, ``[body_len, L)``, lack every axis: each pair with
    #: one is unrotated on both sides.
    spatial_free: torch.Tensor | None = None
    #: (Hr, d_head) bool: the dims of each head's file/rank pairs; needed with spatial_free.
    spatial_dims: torch.Tensor | None = None
    #: Stage N2's masks and biases, built per query block (None: none).
    structure: AttnStructure | None = None
    #: Query rows per block of the block-sparse and chunked paths (whole boards; None: one block).
    block_rows: int | None = None
    #: Under autograd, recompute every block that carries a per-row mask or bias in the
    #: backward pass instead of keeping it (``NetConfig.attn_recompute``).
    recompute: bool = False
    #: Block-sparse path: a block whose key selection exceeds this fraction of the kernel
    #: keys is computed against every key (``NetConfig.attn_sparse_max_fraction``).
    sparse_max_fraction: float = 0.5
    #: The block-causal scope (stage N3; None: the dense scope): rows built per block, and
    #: ANDed in wherever the caller's mask is.
    scope: BlockCausalScope | None = None
    #: B17's key-side Shaw term (``shaw.py``; None: off) - a function of each block's own
    #: (unrotated) queries, added to the scores beside the structure's biases.
    shaw: ShawRows | None = None
    #: G2's attention operands (``quant.AttnQuant``; None: not quantised): Q, K, their rotations
    #: and V fake-quantised per token and head at the entry of every implementation
    #: (``quantised_operands``), so that each selects from the same quantised values.
    quant: AttnQuant | None = None
    #: The multiple the kernel's feature widths are zero-padded to (register G11, section 10.2;
    #: ``NetConfig.attn_pad_multiple``; 1: as built). A context made without it - the cached
    #: path's frozen-board passes (``cache.py``) - pads nothing.
    pad_multiple: int = 1
    #: What no layer changes, built on first use and read by every later layer (register G11,
    #: section 10.2, requirement 1: "Structure is computed once per forward, not once per layer";
    #: ``structure.memoised``): which feature dims are file/rank ones, whether any token lacks
    #: them, each query block's rows that do and their gather indices, each masked role's free
    #: rows, the global registers' mask rows - integers, booleans and Python values, never a
    #: tensor a gradient flows through. ``for_layer`` hands it to the layer's context (a layer
    #: changes the structure's tables and B17's vectors, nothing the memo is built from); any
    #: other copy (``dataclasses.replace``, ``densified``) starts an empty one.
    memo: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def layer_dependent(self) -> bool:
        """Whether a layer's context differs from another's (``for_layer``)."""
        return self.shaw is not None or (self.structure is not None and self.structure.layer_dependent)

    def for_layer(self, h: torch.Tensor | None, layer: int) -> AttnContext:
        """The context of layer ``layer``, whose attention input is ``h`` ``(B, L, d)``
        (``trunk.Block.attn_input``): the structure's layer-dependent biases - board-pair
        smolgen's table generated from ``h`` (B12; only B12 needs ``h``), the layer's
        displacement table (B15) and B16's when it is per layer - and B17's vectors of the layer.
        The per-forward memo goes with it."""
        changes: dict = {}
        if self.structure is not None and self.structure.layer_dependent:
            changes["structure"] = self.structure.for_layer(h, layer)
        if self.shaw is not None:
            changes["shaw"] = self.shaw.for_layer(layer)
        if not changes:
            return self
        out = replace(self, **changes)
        out.memo = self.memo
        return out

    def densified(self) -> AttnContext:
        """The same attention with every mask and bias - padding, the caller's, the
        structure's and the scope's - materialised at ``(B, H, L, L)``: the reference, for
        small inputs. B17's term needs the queries, so it is kept beside the materialised rows
        and computed pair by pair (``shaw.direct_term``) - section 15.2's direct form."""
        length = self.key_valid.shape[1]
        allowed, bias = _block_rows(self, 0, length, key=("rows", 0, length))
        shaw = None if self.shaw is None else replace(self.shaw, direct=True)
        return replace(self, mask=allowed, bias=bias, structure=None, scope=None, shaw=shaw)


def _as4d(t: torch.Tensor | None) -> torch.Tensor | None:
    """A mask or bias broadcastable to (B, H, L, L), with its 4 dims made explicit."""
    if t is None or t.dim() == 4:
        return t
    return t.reshape((1,) * (4 - t.dim()) + tuple(t.shape))


def _block_rows(ctx: AttnContext, start: int, stop: int, key=None, keep_mask: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Allowed keys (bool) and additive bias (float or None) of query rows ``[start,
    stop)``, each broadcastable to ``(B, H, stop - start, L)``: padding, the caller's
    dense mask and bias, the scope's rows and the structure's masks and biases built for
    these rows.

    ``key``: the caller's name for these rows where it asks for them at every layer - the
    structure then builds what no layer changes of their bias once (``AttnStructure.bias_rows``).
    ``keep_mask``: keep the mask rows themselves in the context's memo under ``key`` - for the
    global registers' rows only, ``O(G L)``: a query block's mask is ``O(block L)``, and kept for
    every block it would be the ``(L, L)`` mask this path exists not to hold (B14)."""
    if keep_mask and key is not None:
        allowed = memoised(ctx.memo, ("mask", key), lambda: _block_mask(ctx, start, stop))
    else:
        allowed = _block_mask(ctx, start, stop)
    bias = None if ctx.bias is None else _rows(_as4d(ctx.bias), start, stop)
    if ctx.structure is not None:
        s_bias = ctx.structure.bias_rows(start, stop, key=key)
        if s_bias is not None:
            bias = s_bias if bias is None else bias + s_bias
    return allowed, bias


def _block_mask(ctx: AttnContext, start: int, stop: int) -> torch.Tensor:
    """The allowed keys of ``_block_rows`` alone - padding, the caller's mask, the scope and
    the structure's masks - which no layer changes."""
    allowed = ctx.key_valid[:, None, None, :]
    if ctx.mask is not None:
        allowed = allowed & _rows(_as4d(ctx.mask), start, stop)
    if ctx.scope is not None:
        allowed = allowed & ctx.scope.rows(start, stop)
    if ctx.structure is not None:
        s_allowed = ctx.structure.mask_rows(start, stop)
        if s_allowed is not None:
            allowed = allowed & s_allowed
    return allowed


def _plus(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    """``a + b`` where either may be absent."""
    if a is None:
        return b
    return a if b is None else a + b


def _span(rows: tuple[int, int], bsz: int, device) -> torch.Tensor:
    """``(B, stop - start)``: the query rows ``[start, stop)`` of every position."""
    return torch.arange(rows[0], rows[1], device=device).expand(bsz, rows[1] - rows[0])


def _spatial_free(ctx: AttnContext) -> torch.Tensor | None:
    """The tokens that lack file and rank, or None when the batch has none (looked at once per
    forward: the answer is the batch's, not a layer's)."""
    sf = ctx.spatial_free
    if sf is None or not memoised(ctx.memo, "any_spatial_free", lambda: bool(sf.any())):
        return None
    if ctx.spatial_dims is None:
        raise ValueError("spatial_free needs spatial_dims")
    return sf


def _time_rotated_one(ctx: AttnContext, x: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
    """``x`` rotated on timeline/time only: its file/rank dims left unrotated."""
    return torch.where(_file_rank(ctx, xr.device), x.float(), xr)


def _time_rotated(ctx: AttnContext, q, k, qr, kr) -> tuple[torch.Tensor, torch.Tensor]:
    """Q and K rotated on timeline/time only: their file/rank dims left unrotated."""
    return _time_rotated_one(ctx, q, qr), _time_rotated_one(ctx, k, kr)


def _softmax_rows(scores: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim in FP32; rows with no finite score give zeros."""
    scores = scores.float()
    has_any = torch.isfinite(scores).any(dim=-1, keepdim=True)
    safe = torch.where(has_any, scores, torch.zeros_like(scores))
    return torch.where(has_any, torch.softmax(safe, dim=-1), torch.zeros_like(scores))


def _scores(q: torch.Tensor, k: torch.Tensor, scale: float, cd: torch.dtype) -> torch.Tensor:
    return torch.matmul(q.to(cd), k.to(cd).transpose(-1, -2)).float() * scale


def _rows(t: torch.Tensor | None, start: int, stop: int) -> torch.Tensor | None:
    """Query rows [start, stop) of a tensor broadcastable to (B, H, L, L)."""
    if t is None or t.shape[-2] == 1:
        return t
    return t[..., start:stop, :]


def _cols(t: torch.Tensor | None, cols: slice) -> torch.Tensor | None:
    """Key columns ``cols`` of a tensor broadcastable to (B, H, Lq, L)."""
    if t is None or t.shape[-1] == 1:
        return t
    return t[..., cols]


def _take_cols(t: torch.Tensor | None, idx: torch.Tensor) -> torch.Tensor | None:
    """Key columns ``idx`` ``(B, K)`` - each position its own - of a tensor broadcastable
    to ``(B, X, Lq, Lk)``."""
    if t is None or t.shape[-1] == 1:
        return t
    bsz, k = idx.shape
    t = t.expand(bsz, *t.shape[1:])
    return t.gather(3, idx[:, None, None, :].expand(bsz, t.shape[1], t.shape[2], k))


def _heads(t: torch.Tensor | None, heads: torch.Tensor | None) -> torch.Tensor | None:
    """Heads ``heads`` of a tensor broadcastable to (B, H, ...) (all heads when None)."""
    if t is None or heads is None or t.shape[1] == 1:
        return t
    return t[:, heads]


def _block(t: torch.Tensor | None, rows: slice, cols: slice) -> torch.Tensor | None:
    """A (rows, cols) block of a tensor broadcastable to (B, H, L, L)."""
    if t is None:
        return None
    t = t if t.shape[-2] == 1 else t[..., rows, :]
    return t if t.shape[-1] == 1 else t[..., cols]


def _rows_at(t: torch.Tensor | None, idx: torch.Tensor) -> torch.Tensor | None:
    """Rows ``idx`` ``(B, R)`` of a tensor broadcastable to ``(B, H, L, X)``."""
    if t is None or t.shape[-2] == 1:
        return t
    bsz, n = idx.shape
    t = t.expand(bsz, *t.shape[1:])
    return t.gather(2, idx[:, None, :, None].expand(bsz, t.shape[1], n, t.shape[-1]))


def _row_index(sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Each position's rows where ``sel`` ``(B, Lq)``, padded to the most any position
    has: ``(take, put, real)`` ``(B, n)``. A padding entry is computed at a real row
    (``take``) and written to a dummy row ``Lq`` that is dropped (``put``), so it neither
    overwrites a row nor feeds a gradient twice. None when no row is selected."""
    nq = sel.shape[1]
    counts = sel.sum(dim=1)
    n = int(counts.max()) if sel.numel() else 0
    if n == 0:
        return None
    order = torch.argsort((~sel).to(torch.uint8), dim=1, stable=True)[:, :n]
    real = torch.arange(n, device=sel.device)[None, :] < counts[:, None]
    take = torch.where(real, order, order[:, :1])
    put = torch.where(real, order, torch.full_like(order, nq))
    return take, put, real


def _scatter_rows(out: torch.Tensor, put: torch.Tensor, rows_out: torch.Tensor) -> torch.Tensor:
    """``out`` ``(B, H, Lq, dv)`` with the rows ``put`` ``(B, n)`` replaced by ``rows_out``."""
    bsz, n_heads, nq, dv = out.shape
    padded = torch.cat([out, out.new_zeros(bsz, n_heads, 1, dv)], dim=2)
    index = put[:, None, :, None].expand(bsz, n_heads, put.shape[1], dv)
    return padded.scatter(2, index, rows_out.to(padded.dtype))[:, :, :nq]


def _maybe_recompute(fn: Callable[[], torch.Tensor], recompute: bool) -> torch.Tensor:
    """``fn()``; under autograd with ``recompute``, recomputed in the backward pass instead
    of keeping what it saves (activation checkpointing - exact, one extra forward)."""
    if recompute and torch.is_grad_enabled():
        return checkpoint(fn, use_reentrant=False, preserve_rng_state=False)
    return fn()


# --- reference ------------------------------------------------------------------------------


def quantised_operands(q, k, v, ctx: AttnContext):
    """``(q, k, v, qr, kr)``: the unrotated Q and K, V, and Q and K rotated by FP32 RoPE (G1) -
    each fake-quantised per token and head under G2's attention format (``ctx.quant``), else as
    they are. Every implementation takes its operands from here, once per layer."""
    qr = apply_rope(q, ctx.cos, ctx.sin)
    kr = apply_rope(k, ctx.cos, ctx.sin)
    if ctx.quant is None:
        return q, k, v, qr, kr
    return ctx.quant.operands(q, k, v, qr, kr)


def attention_explicit(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext, return_probs: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference attention; ``q``, ``k`` unrotated ``(B, H, L, dh)``, ``v`` ``(B, H, L, dv)``.

    Scores are materialised one chunk of ``ctx.chunk`` query rows at a time, with that
    chunk's mask and bias rows (``_block_rows``) and B17's term from the chunk's unrotated
    queries (``ctx.shaw``: gathered, or per pair in the dense reference)."""
    length, dh = q.shape[-2:]
    lb, cd = ctx.body_len, ctx.compute_dtype
    scale = dh**-0.5
    q, k, v, qr, kr = quantised_operands(q, k, v, ctx)
    sf = _spatial_free(ctx)
    if sf is not None:
        qt, kt = _time_rotated(ctx, q, k, qr, kr)
    global_free = lb < length
    outs, probs_all = [], []
    step = max(1, ctx.chunk)
    for start in range(0, length, step):
        stop = min(length, start + step)
        allowed, bias = _block_rows(ctx, start, stop, key=("rows", start, stop))
        s = _scores(qr[:, :, start:stop], kr, scale, cd)
        if sf is not None:
            # A pair with a token that lacks file/rank: those dims unrotated on both sides.
            pair = sf[:, None, start:stop, None] | sf[:, None, None, :]
            s = torch.where(pair, _scores(qt[:, :, start:stop], kt, scale, cd), s)
        if global_free:
            # A pair with a global register: unrotated on both sides.
            s[..., lb:] = _scores(q[:, :, start:stop], k[:, :, lb:], scale, cd)
            g0 = max(start, lb)
            if g0 < stop:
                s[:, :, g0 - start :] = _scores(q[:, :, g0:stop], k, scale, cd)
        if bias is not None:
            s = s + bias.float()
        if ctx.shaw is not None:
            s = s + ctx.shaw.rows(q[:, :, start:stop], _span((start, stop), q.shape[0], q.device))
        s = s.masked_fill(~allowed, NEG_INF)
        p = _softmax_rows(s)
        outs.append(torch.matmul(p.to(cd), v.to(cd)))
        if return_probs:
            probs_all.append(p)
    out = torch.cat(outs, dim=2)
    if return_probs:
        return out, torch.cat(probs_all, dim=2)
    return out


def attention_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
    """The reference with every mask and bias materialised first (``densified``)."""
    return attention_explicit(q, k, v, ctx.densified())


# --- fast path --------------------------------------------------------------------------------


@dataclass
class _Merge:
    """Each kernel query's position-free global block, carried by the extra key."""

    lse: torch.Tensor  # (B, H, Lq): log-sum-exp of the allowed global scores (0 if none)
    has: torch.Tensor  # (B, H, Lq) bool: some global key is allowed
    u: torch.Tensor  # (B, H, Lq, dv): softmax over the global block, times its values

    def rows(self, idx: torch.Tensor) -> _Merge:
        bsz, n_heads = self.lse.shape[:2]
        i3 = idx[:, None, :].expand(bsz, n_heads, idx.shape[1])
        i4 = i3[..., None].expand(-1, -1, -1, self.u.shape[-1])
        return _Merge(self.lse.gather(2, i3), self.has.gather(2, i3), self.u.gather(2, i4))


def _global_merge(q_rows, k_g, v_g, allowed, bias, scale: float, cd: torch.dtype) -> _Merge:
    s = _scores(q_rows, k_g, scale, cd)  # unrotated on both sides: position-free pairs
    if bias is not None:
        s = s + bias.float()
    s = s.masked_fill(~allowed, NEG_INF)
    # Rows where every global key is masked are replaced by zeros before logsumexp, so
    # that neither the value nor the gradient sees an all -inf row (NaN backward).
    has = torch.isfinite(s).any(dim=-1)
    s_safe = torch.where(has[..., None], s, torch.zeros_like(s))
    lse = torch.logsumexp(s_safe, dim=-1)
    p = torch.where(has[..., None], torch.exp(s_safe - lse[..., None]), torch.zeros_like(s))
    u = torch.matmul(p.to(cd), v_g.to(cd))
    return _Merge(torch.where(has, lse, torch.zeros_like(lse)), has, u)


def _pad_width(t: torch.Tensor, width: int) -> torch.Tensor:
    return t if t.shape[-1] == width else F.pad(t, (0, width - t.shape[-1]))


def padded_width(width: int, multiple: int) -> int:
    """``width`` rounded up to a multiple of ``multiple`` - the width the kernel's features are
    zero-padded to (register G11, section 10.2: "Fused attention kernels want head widths in
    multiples of 8"; ``NetConfig.attn_pad_multiple``). ``multiple`` 1 leaves it."""
    return -(-width // multiple) * multiple


@dataclass
class _KernelKeys:
    """The key side of kernel calls, prepared once and shared by every query block: keys
    and values padded to one width and cast to the compute dtype, with the extra key that
    carries the position-free global block appended when ``merged``.

    This is the one place the fast paths' kernel widths are set (register G11): ``build`` pads
    keys and values with zero columns to their common width rounded up to ``multiple``
    (``padded_width``), and ``_kernel`` pads each call's queries to the keys' width. A zero
    column adds zero to every score, and the padded value columns are cut from the output."""

    k: torch.Tensor  # (B, H, Lk (+1), W)
    v: torch.Tensor  # (B, H, Lk (+1), W)
    lk: int  # real keys, the extra one excluded
    dv: int
    merged: bool

    @classmethod
    def build(cls, kk: torch.Tensor, vv: torch.Tensor, merged: bool, cd: torch.dtype, multiple: int = 1) -> _KernelKeys:
        bsz, n_heads, lk, _ = kk.shape
        dv = vv.shape[-1]
        vv = vv.float()
        if merged:
            kk = torch.cat([kk.float(), kk.new_zeros(bsz, n_heads, lk, 1, dtype=torch.float32)], dim=-1)
            extra_k = kk.new_zeros(bsz, n_heads, 1, kk.shape[-1])
            extra_k[..., -1] = 1.0
            kk = torch.cat([kk, extra_k], dim=2)
            vv = torch.cat([vv, vv.new_zeros(bsz, n_heads, lk, 1)], dim=-1)
            extra_v = vv.new_zeros(bsz, n_heads, 1, dv + 1)
            extra_v[..., -1] = 1.0
            vv = torch.cat([vv, extra_v], dim=2)
        width = padded_width(max(kk.shape[-1], vv.shape[-1]), multiple)
        return cls(_pad_width(kk, width).to(cd), _pad_width(vv, width).to(cd), lk, dv, merged)

    def heads(self, heads: torch.Tensor | None) -> _KernelKeys:
        return _KernelKeys(_heads(self.k, heads), _heads(self.v, heads), self.lk, self.dv, self.merged)

    def select(self, idx: torch.Tensor, extra: bool = True) -> _KernelKeys:
        """The keys ``idx`` ``(B, K)`` (each position its own), with the extra key kept last
        when ``extra`` (dropped otherwise: then the selection takes no global merge - its
        keys' extra feature column is zero, so a query padded with a zero scores the same)."""
        bsz, k = idx.shape
        n_heads, width = self.k.shape[1], self.k.shape[-1]
        if bsz == 1 and self.k.shape[0] == 1:
            kk = self.k[:, :, : self.lk].index_select(2, idx[0])
            vv = self.v[:, :, : self.lk].index_select(2, idx[0])
        else:
            index = idx[:, None, :, None].expand(bsz, n_heads, k, width)
            kk = self.k[:, :, : self.lk].expand(bsz, -1, -1, -1).gather(2, index)
            vv = self.v[:, :, : self.lk].expand(bsz, -1, -1, -1).gather(2, index)
        merged = self.merged and extra
        if merged:
            kk = torch.cat([kk, self.k[:, :, self.lk :].expand(bsz, -1, -1, -1)], dim=2)
            vv = torch.cat([vv, self.v[:, :, self.lk :].expand(bsz, -1, -1, -1)], dim=2)
        return _KernelKeys(kk, vv, k, self.dv, merged)


def _mask_constant(value: float, cd: torch.dtype, device, memo: dict | None) -> torch.Tensor:
    """``value`` (-inf: a masked key; 0: the allowed extra key) as a scalar tensor of the compute
    dtype - what ``_additive_mask`` writes. With ``memo`` (the context's) it is made once per
    forward, not once per kernel call: on a device each is a host-to-device copy."""

    def make() -> torch.Tensor:
        return torch.tensor(value, dtype=cd, device=device) if value else torch.zeros((), dtype=cd, device=device)

    return make() if memo is None else memoised(memo, ("mask_constant", value, cd, device), make)


def _additive_mask(
    bias: torch.Tensor, allowed: torch.Tensor, extra: torch.Tensor | None, shape, cd: torch.dtype, memo: dict | None = None
) -> torch.Tensor:
    """The kernel's float mask ``(B, H, Lq, Lk (+1))``: ``bias`` where ``allowed``, -inf
    elsewhere, with the extra key's column (0 where ``extra``, -inf elsewhere) appended
    when ``extra`` is given. Written in one pass into the final buffer when no gradient
    flows into the bias (inference); differentiably otherwise."""
    if torch.is_grad_enabled() and bias.requires_grad:
        body = bias.float().expand(shape).masked_fill(~allowed.expand(shape), NEG_INF)
        if extra is None:
            return body.to(cd)
        extra_bias = torch.zeros_like(extra, dtype=torch.float32).masked_fill(~extra, NEG_INF)
        return torch.cat([body, extra_bias[..., None]], dim=-1).to(cd)
    bsz, n_heads, lq, lk = shape
    out = torch.empty(bsz, n_heads, lq, lk + (extra is not None), dtype=cd, device=bias.device)
    neg_inf = _mask_constant(NEG_INF, cd, bias.device, memo)
    torch.where(allowed, bias.to(cd), neg_inf, out=out[..., :lk])
    if extra is not None:
        out[..., lk] = torch.where(extra, _mask_constant(0.0, cd, bias.device, memo), neg_inf)
    return out


def _kernel(
    qq, keys: _KernelKeys, allowed, bias, merge: _Merge | None, scale: float, cd: torch.dtype, memo: dict | None = None, name=None
) -> torch.Tensor:
    """The kernel over queries ``qq`` ``(B, H, Lq, D)`` and prepared keys (``Lk`` real
    keys); ``allowed``/``bias`` broadcast to ``(B, H, Lq, Lk)``. With ``merge``, plus the
    extra key that carries the global block. A row with no allowed key outputs zeros.

    ``memo``: the context's per-forward memo, for the mask's constants (``_mask_constant``) and -
    with ``name``, the caller's name for a call it makes with the same ``allowed`` at every layer
    (a mask is the position's, never a layer's) - for which rows have an allowed key at all:
    that reduction over the mask is then done once per forward, not once per layer."""
    if (merge is not None) != keys.merged:
        raise ValueError("the keys were prepared for a different global merge")
    bsz, n_heads, lq, _ = qq.shape
    lk, dv = keys.lk, keys.dv
    kept = memo is not None and name is not None
    rows_any = memoised(memo, ("rows_any", name), lambda: allowed.any(dim=-1)) if kept else allowed.any(dim=-1)

    def rows_none() -> torch.Tensor:
        return memoised(memo, ("rows_none", name), lambda: ~rows_any) if kept else ~rows_any

    def every_row() -> bool:
        return memoised(memo, ("rows_all", name), lambda: bool(rows_any.all())) if kept else bool(rows_any.all())

    empty = None
    if merge is None:
        if allowed.shape[-2] == 1 and every_row():
            fixed = allowed
        else:
            # A row with nothing allowed attends everywhere (finite values and gradients)
            # and is zeroed below.
            empty = rows_none()
            fixed = allowed | empty[..., None]
        attn_mask = fixed if bias is None else _additive_mask(bias, fixed, None, (bsz, n_heads, lq, lk), cd, memo)
    else:
        qq = torch.cat([qq.float(), (merge.lse / scale)[..., None]], dim=-1)
        # The extra key is allowed where some global key is, and for a row with no allowed
        # key at all, which then attends to it alone: u = 0 there, so it outputs zeros.
        extra = merge.has | rows_none()
        if bias is None and allowed.shape[-2] == 1 and bool(extra.all()):
            # Padding only: the mask stays a broadcast key mask.
            attn_mask = torch.cat([allowed, allowed.new_ones(*allowed.shape[:-1], 1)], dim=-1)
        else:
            if bias is None:
                attn_mask = torch.cat([allowed.expand(bsz, n_heads, lq, lk), extra[..., None]], dim=-1)
            else:
                attn_mask = _additive_mask(bias, allowed, extra, (bsz, n_heads, lq, lk), cd, memo)
    o = F.scaled_dot_product_attention(_pad_width(qq, keys.k.shape[-1]).to(cd), keys.k, keys.v, attn_mask=attn_mask, scale=scale)
    if merge is not None:
        return o[..., :dv] + o[..., dv : dv + 1] * merge.u
    o = o[..., :dv]
    return o if empty is None else torch.where(empty[..., None], torch.zeros_like(o), o)


def _file_rank(ctx: AttnContext, device) -> torch.Tensor:
    """Each head's file/rank dims, ``(1, Hr, 1, d_head)`` bool - the model's, the same at every
    layer and for every query block: laid out once per forward."""
    return memoised(ctx.memo, ("file_rank", device), lambda: ctx.spatial_dims.to(device)[None, :, None, :])


def _wide_selection(ctx: AttnContext, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(file_rank, sel)``: ``_file_rank``, and the dims some head rotates on file or rank - the
    widened features' extra columns (found once per forward)."""
    sel = memoised(ctx.memo, ("wide", device), lambda: torch.nonzero(ctx.spatial_dims.to(device).any(dim=0)).flatten())
    return _file_rank(ctx, device), sel


def _widened_q(ctx: AttnContext, q, qr) -> torch.Tensor:
    """Kernel features of queries that have file/rank (see the module docstring)."""
    file_rank, sel = _wide_selection(ctx, qr.device)
    return torch.cat([qr, torch.where(file_rank, q.float(), 0.0)[..., sel]], dim=-1)


def _widened_k(ctx: AttnContext, sf: torch.Tensor, k, kr) -> torch.Tensor:
    """Kernel features of keys, some of which lack file/rank (``sf`` ``(B, Lk)``)."""
    file_rank, sel = _wide_selection(ctx, kr.device)
    k_fr = torch.where(file_rank, k.float(), 0.0)[..., sel]
    lacks = sf[:, None, :, None]
    k_rot = torch.where(file_rank & lacks, 0.0, kr)
    return torch.cat([k_rot, torch.where(lacks, k_fr, 0.0)], dim=-1)


def _widened(ctx: AttnContext, sf: torch.Tensor, q, k, qr, kr) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel features of queries that have file/rank over keys that may lack them (see
    the module docstring); ``sf`` covers the keys."""
    return _widened_q(ctx, q, qr), _widened_k(ctx, sf, k, kr)


def _spatial_free_rows(
    sf_rows,
    out,
    qt,
    tkeys: _KernelKeys,
    allowed,
    bias,
    merge,
    scale: float,
    cd: torch.dtype,
    index=None,
    memo: dict | None = None,
    name=None,
) -> torch.Tensor:
    """``out`` ``(B, H, Lq, dv)`` with the rows of queries that lack file/rank (``sf_rows``
    ``(B, Lq)``) recomputed from ``qt`` and ``tkeys`` (Q and K rotated on timeline/time
    only). ``index``: ``_row_index(sf_rows)`` where the caller already has it (the rows are a
    query block's, the same at every layer); ``memo``, ``name``: ``_kernel``'s."""
    if index is None:
        index = _row_index(sf_rows)
    if index is None:
        return out
    take, put, _ = index
    o_rows = _kernel(
        _rows_at(qt, take),
        tkeys,
        _rows_at(allowed, take),
        _rows_at(bias, take),
        None if merge is None else merge.rows(take),
        scale,
        cd,
        memo,
        name,
    )
    return _scatter_rows(out, put, o_rows)


@dataclass
class _Keys:
    """Everything on the key side of one fast-path call, shared by every query block."""

    kernel: _KernelKeys  # kernel keys: rotated, or widened
    rotated_t: _KernelKeys | None  # rotated on timeline/time only (spatial-free queries)
    k_g: torch.Tensor | None  # (B, H, G, dh): position-free global keys, unrotated
    v_g: torch.Tensor | None

    def heads(self, heads: torch.Tensor | None) -> _Keys:
        t = None if self.rotated_t is None else self.rotated_t.heads(heads)
        return _Keys(self.kernel.heads(heads), t, _heads(self.k_g, heads), _heads(self.v_g, heads))

    def global_cast(self, cd: torch.dtype) -> _Keys:
        """These keys with the global keys and values already in the compute dtype - what every
        query block's merge casts them to (``_global_merge``), done once for the layer."""
        if self.k_g is None:
            return self
        return _Keys(self.kernel, self.rotated_t, self.k_g.to(cd), self.v_g.to(cd))

    def select(self, idx: torch.Tensor, merge: bool = True) -> _Keys:
        """The kernel keys ``idx`` ``(B, K)``, and the global keys of the merge unchanged -
        or, without ``merge``, dropped with the extra key that carries them."""
        merge = merge and self.k_g is not None
        t = None if self.rotated_t is None else self.rotated_t.select(idx, merge)
        return _Keys(self.kernel.select(idx, merge), t, self.k_g if merge else None, self.v_g if merge else None)


def _attend_rows(
    qq,
    q_raw,
    qt,
    keys: _Keys,
    a_k,
    b_k,
    a_g,
    b_g,
    sf_rows,
    scale: float,
    cd: torch.dtype,
    sf_index=None,
    memo: dict | None = None,
    name=None,
) -> torch.Tensor:
    """Query rows against the prepared keys: ``qq`` their kernel-query features, ``q_raw``
    their unrotated queries (for the global merge), ``qt`` their timeline/time-only
    rotation (None without spatial-free queries); ``a_k``/``b_k`` their allowed keys and
    bias over the kernel keys, ``a_g``/``b_g`` over the position-free global keys;
    ``sf_index``: the spatial-free rows' gather indices where already built; ``memo`` and
    ``name`` (the caller's name for these rows, keys and heads, the same at every layer):
    ``_kernel``'s - each of the two calls under its own name."""
    merge = None
    if keys.k_g is not None:
        merge = _global_merge(q_raw, keys.k_g, keys.v_g, a_g, b_g, scale, cd)
    out = _kernel(qq, keys.kernel, a_k, b_k, merge, scale, cd, memo, None if name is None else (name, "rows"))
    if sf_rows is not None:
        sf_name = None if name is None else (name, "spatial-free rows")
        out = _spatial_free_rows(sf_rows, out, qt, keys.rotated_t, a_k, b_k, merge, scale, cd, sf_index, memo, sf_name)
    return out


def _per_row(t: torch.Tensor | None) -> bool:
    """Whether a caller's mask or bias differs between query rows."""
    return t is not None and _as4d(t).shape[-2] > 1


class _FastPath:
    """One call of the fast path: the rotated queries, the prepared keys, and how a set of
    query rows - a block ``(start, stop)`` or each position's own rows ``(B, n)`` - on a
    set of heads is computed against every key or against a key selection."""

    def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext) -> None:
        length, dh = q.shape[-2:]
        q, k, v, self.qr, kr = quantised_operands(q, k, v, ctx)
        #: The operands as every block reads them (G2's quantised ones under ``ctx.quant``).
        self.ctx, self.q, self.k, self.v, self.length = ctx, q, k, v, length
        self.cd = ctx.compute_dtype
        self.scale = dh**-0.5
        self.sf = sf = _spatial_free(ctx)
        self.kl = kl = ctx.body_len  # the kernel keys; the position-free global registers are merged
        self.kern, self.rest = slice(0, kl), slice(kl, length)
        merged = kl < length
        k_feat = kr[:, :, self.kern] if sf is None else _widened_k(ctx, sf[:, self.kern], k[:, :, self.kern], kr[:, :, self.kern])
        rotated_t = None
        if sf is not None:
            k_t = _time_rotated_one(ctx, k[:, :, self.kern], kr[:, :, self.kern])
            rotated_t = _KernelKeys.build(k_t, v[:, :, self.kern], merged, self.cd, ctx.pad_multiple)
        self.keys = _Keys(
            _KernelKeys.build(k_feat, v[:, :, self.kern], merged, self.cd, ctx.pad_multiple),
            rotated_t,
            k[:, :, self.rest] if merged else None,
            v[:, :, self.rest] if merged else None,
        )
        #: At inference: the query side of every kernel row, made once for the layer
        #: (``layer_queries``) - on every head, and per head group.
        self._queries: tuple | None = None
        self._group_queries: dict = {}

    def layer_queries(self, heads: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(qq, q, qt)`` of every kernel row ``[0, kl)`` on ``heads``: the kernel-query features,
        the unrotated queries in the compute dtype (the global merge's operand) and the
        timeline/time-only rotation (None when no token lacks file and rank). Made once for the
        layer and sliced by every query block (register G11) - a selection of rows and heads of an
        elementwise result is that result on the selection, bit for bit. Inference only: under
        autograd each block makes its own, as built (``query_side``). ``heads`` are recognised
        by identity: the structure's own index tensors (``AttnStructure.head_groups``)."""
        if self._queries is None:
            q_all, qr_all = self.q[:, :, self.kern], self.qr[:, :, self.kern]
            qq = qr_all if self.sf is None else _widened_q(self.ctx, q_all, qr_all)
            qt = None if self.sf is None else _time_rotated_one(self.ctx, q_all, qr_all)
            self._queries = (qq, q_all.to(self.cd), qt)
        if heads is None:
            return self._queries
        entry = self._group_queries.get(id(heads))
        if entry is None or entry[0] is not heads:  # the entry holds ``heads``, so its id cannot be reused
            entry = self._group_queries[id(heads)] = (heads, tuple(_heads(t, heads) for t in self._queries))
        return entry[1]

    def spatial_free_rows(self, rows, key=None):
        """``(sf_rows, index)``: the flags ``(B, n)`` of the query rows ``rows`` that lack file
        and rank - None when none of them does - and their gather indices (``_row_index``). Both
        are the batch's, not a layer's: for a block ``(start, stop)``, and for rows the caller
        names (``key``: the same rows under the same name at every layer), they are found once per
        forward and kept in the context's memo (register G11)."""
        if self.sf is None:
            return None, None

        def find():
            sf_c = self.sf[:, rows[0] : rows[1]] if isinstance(rows, tuple) else self.sf.gather(1, rows)
            return (sf_c, _row_index(sf_c)) if bool(sf_c.any()) else (None, None)

        if isinstance(rows, tuple):
            return memoised(self.ctx.memo, ("sf_rows", rows), find)
        return find() if key is None else memoised(self.ctx.memo, ("sf_rows", key), find)

    def query_side(self, rows, heads: torch.Tensor | None, key=None):
        """Kernel features, unrotated queries, time-only rotation, spatial-free flags and their
        gather indices (``spatial_free_rows``) of the query rows ``rows`` on ``heads``. At
        inference they are rows of the layer's query side (``layer_queries``); under autograd they
        are made for these rows, as built, so that the queries' gradients are accumulated as they
        always were."""
        sf_rows, sf_index = self.spatial_free_rows(rows, key)
        if not torch.is_grad_enabled():
            pick = (lambda t: t[:, :, rows[0] : rows[1]]) if isinstance(rows, tuple) else (lambda t: _rows_at(t, rows))
            qq, q_c, qt = self.layer_queries(heads)
            return pick(qq), pick(q_c), None if sf_rows is None else pick(qt), sf_rows, sf_index
        if isinstance(rows, tuple):
            q_c, qr_c = self.q[:, :, rows[0] : rows[1]], self.qr[:, :, rows[0] : rows[1]]
        else:
            q_c, qr_c = _rows_at(self.q, rows), _rows_at(self.qr, rows)
        qq = qr_c if self.sf is None else _widened_q(self.ctx, q_c, qr_c)
        qt = None if sf_rows is None else _time_rotated_one(self.ctx, q_c, qr_c)
        return _heads(qq, heads), _heads(q_c, heads), _heads(qt, heads), sf_rows, sf_index

    def caller_rows(self, rows, heads: torch.Tensor | None):
        """Allowed keys (padding, the caller's mask and the scope) and the caller's bias of
        the rows."""
        ctx = self.ctx
        allowed = ctx.key_valid[:, None, None, :]
        pick = (lambda t: _rows(t, *rows)) if isinstance(rows, tuple) else (lambda t: _rows_at(t, rows))
        if ctx.mask is not None:
            allowed = allowed & _heads(pick(_as4d(ctx.mask)), heads)
        if ctx.scope is not None:
            allowed = allowed & (ctx.scope.rows(*rows) if isinstance(rows, tuple) else ctx.scope.rows_at(rows))
        bias = None if ctx.bias is None else _heads(pick(_as4d(ctx.bias)), heads)
        return allowed, bias

    def attend(self, rows, heads, keys: _Keys, a_k, b_k, a_g, b_g, key=None) -> torch.Tensor:
        """``key`` (hashable; None: nothing is kept): the caller's name for this call where it
        makes it at every layer with the same rows, heads, keys and allowed keys - then what
        only the position decides is found once per forward and kept in the context's memo: a
        row tensor's spatial-free rows (``spatial_free_rows``) and which rows have an allowed
        key at all (``_kernel``)."""
        qq, q_c, qt, sf_rows, sf_index = self.query_side(rows, heads, key)
        return _attend_rows(qq, q_c, qt, keys, a_k, b_k, a_g, b_g, sf_rows, self.scale, self.cd, sf_index, self.ctx.memo, key)

    def full(self, rows, heads, keys: _Keys, allowed, bias, key=None) -> torch.Tensor:
        """The rows against every key (``key``: as ``attend``'s)."""
        a_k, b_k, a_g, b_g = _cols(allowed, self.kern), _cols(bias, self.kern), _cols(allowed, self.rest), _cols(bias, self.rest)
        return self.attend(rows, heads, keys, a_k, b_k, a_g, b_g, key)

    def shaw_rows(self, rows, heads: torch.Tensor | None, cols: torch.Tensor | None = None) -> torch.Tensor | None:
        """B17's term ``(B, Hg, n, Lk)`` of the query rows ``rows`` - a block ``(start, stop)`` or
        each position's rows ``(B, n)`` - on ``heads`` over every key (``cols`` None) or the keys
        ``cols`` ``(B, K)``, from the rows' unrotated queries (None: B17 off)."""
        if self.ctx.shaw is None:
            return None
        if isinstance(rows, tuple):
            q_c, index = self.q[:, :, rows[0] : rows[1]], _span(rows, self.q.shape[0], self.q.device)
        else:
            q_c, index = _rows_at(self.q, rows), rows
        return self.ctx.shaw.rows(_heads(q_c, heads), index, cols, heads)

    # -- the block computations --

    def all_heads_block(self, start: int, stop: int) -> torch.Tensor:
        """Every head of the rows ``[start, stop)`` in one call (the chunked path with a
        bias or a caller's per-row mask, and the one-block paths)."""
        allowed, bias = _block_rows(self.ctx, start, stop, key=("rows", start, stop))
        bias = _plus(bias, self.shaw_rows((start, stop), None))
        return self.full((start, stop), None, self.keys, allowed, bias, key=("rows", start, stop))

    def group_block(
        self, role: str, heads: torch.Tensor, start: int, stop: int, sparse: bool, group_keys: _Keys | None = None
    ) -> torch.Tensor:
        """The heads of one role over the rows ``[start, stop)`` - ``heads`` are the role's
        (``AttnStructure.head_groups``); ``sparse``: a masked role reads only its block's key
        selection (``AttnStructure.block_plan``), unless the plan is dense. ``group_keys``:
        ``self.keys.heads(heads)`` where the caller made them once for every block of the layer
        (``attention_sdpa``, at inference)."""
        structure = self.ctx.structure
        if group_keys is None:
            group_keys = self.keys.heads(heads)
        plan = None
        if sparse and role != "dense":
            plan = structure.block_plan(role, start, stop, self.kl, self.ctx.sparse_max_fraction)
        allowed, bias = self.caller_rows((start, stop), heads)
        if plan is None or plan.index is None:
            if role != "dense":
                allowed = allowed & (structure.role_rows(role, start, stop) & self.ctx.key_valid[:, None, :])[:, None]
            name = ("group", role, start, stop)  # the role's heads on these rows over every key, at every layer
            s_bias = structure.bias_rows(start, stop, heads=heads, key=name)
            if s_bias is not None:
                bias = s_bias if bias is None else bias + s_bias
            bias = _plus(bias, self.shaw_rows((start, stop), heads))
            return self.full((start, stop), heads, group_keys, allowed, bias, key=name)
        idx = plan.index
        if idx.shape[1] == 0:
            # Only free and padding queries: computed apart, or discarded.
            return self.q.new_zeros(self.q.shape[0], int(heads.shape[0]), stop - start, group_keys.kernel.dv, dtype=self.cd)
        a_k = plan.allowed[:, None]
        if self.ctx.mask is not None or self.ctx.scope is not None:
            a_k = a_k & _take_cols(_cols(allowed, self.kern), idx)
        b_k = _take_cols(_cols(bias, self.kern), idx)
        plan_key = ("plan", role, start, stop, self.kl, self.ctx.sparse_max_fraction)  # the plan's own (``block_plan``)
        s_bias = structure.bias_rows(start, stop, heads=heads, cols=idx, cols_valid=plan.valid, key=plan_key)
        if s_bias is not None:
            b_k = s_bias if b_k is None else b_k + s_bias
        b_k = _plus(b_k, self.shaw_rows((start, stop), heads, cols=idx))
        # No selecting query of a sparse plan reads a position-free global key, so the
        # selection drops the merge's extra key (the free and padding queries' outputs
        # here are replaced or discarded).
        return self.attend((start, stop), heads, group_keys.select(idx, merge=False), a_k, b_k, None, None, key=plan_key)

    def free_rows(self, role: str, heads: torch.Tensor, out: torch.Tensor, step: int, group_keys: _Keys | None = None) -> torch.Tensor:
        """``out`` with the role's free queries (``AttnStructure.free_rows``: they read every
        valid key) recomputed against every key, ``step`` rows per call. Which rows they are is
        the batch's: their gather indices are found once per forward (the context's memo).
        ``group_keys``: as ``group_block``'s."""
        ctx = self.ctx

        def find():
            free = ctx.structure.free_rows(role)
            return None if free is None else _row_index(free[:, : self.kl] & ctx.key_valid[:, : self.kl])

        index = memoised(ctx.memo, ("free_index", role, self.kl), find)
        if index is None:
            return out
        take, put, real = index
        if group_keys is None:
            group_keys = self.keys.heads(heads)
        row_free = ctx.mask is None and ctx.bias is None and ctx.scope is None and not ctx.structure.has_bias and ctx.shaw is None
        size = take.shape[1] if row_free else max(1, step)
        outs = []
        for a in range(0, take.shape[1], size):
            rows, rows_real = take[:, a : a + size], real[:, a : a + size]
            name = ("free", role, self.kl, a, size)  # these rows, at every layer

            def run(rows=rows, rows_real=rows_real, name=name) -> torch.Tensor:
                allowed, bias = self.caller_rows(rows, heads)
                s_bias = self.ctx.structure.bias_rows_at(rows, rows_real, heads=heads, key=name)
                if s_bias is not None:
                    bias = s_bias if bias is None else bias + s_bias
                bias = _plus(bias, self.shaw_rows(rows, heads))
                return self.full(rows, heads, group_keys, allowed, bias, key=name)

            outs.append(run() if row_free else _maybe_recompute(run, self.ctx.recompute))
        return _scatter_rows(out, put, torch.cat(outs, dim=2))


def attention_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext, block_rows: int | None = None, sparse: bool = False
) -> torch.Tensor:
    """Fast path: the kernel over every token but position-free global registers, whose
    pairs are merged exactly (see the module docstring), over query blocks of
    ``block_rows`` rows (one block when None) with each block's masks and biases built
    for its rows only; ``sparse``: block-sparse key selection on the masked heads. Rows
    that need no per-row mask or bias - every row when nothing but padding is masked, the
    dense heads' rows when nothing carries a bias - run in one call over all queries: a
    broadcast key mask costs no per-row memory. Under autograd with ``ctx.recompute``,
    every block that carries a per-row mask or bias is recomputed in the backward pass."""
    fast = _FastPath(q, k, v, ctx)
    kl, length, cd, scale = fast.kl, fast.length, fast.cd, fast.scale
    structure = ctx.structure
    per_row = structure is not None or _per_row(ctx.mask) or _per_row(ctx.bias) or ctx.scope is not None or ctx.shaw is not None
    step = kl if block_rows is None or not per_row else max(1, block_rows)
    blocks = [(start, min(kl, start + step)) for start in range(0, kl, step)]
    recompute = ctx.recompute and per_row
    # "No bias" also means no per-row mask from the caller or the scope: the dense heads
    # then run in one call over all rows.
    no_bias = (
        ctx.mask is None and ctx.bias is None and ctx.scope is None and ctx.shaw is None and (structure is None or not structure.has_bias)
    )
    if structure is not None and (sparse or no_bias):
        # By role: heads that share a mask role share one kernel call per block and one
        # (B, 1, block, L) mask (or, sparse, one key selection); without any bias the
        # dense heads run once over all rows with the padding mask.
        parts, order = [], []
        for role, heads in structure.head_groups():
            if role == "dense" and no_bias:
                allowed = ctx.key_valid[:, None, None, :]
                group_keys = fast.keys.heads(heads)
                parts.append(fast.full((0, kl), heads, group_keys, allowed, None, key=("padding", role)))
            else:
                # At inference the group's keys are taken once for the layer, not once per query
                # block (register G11: the same heads of the same keys, block after block). Under
                # autograd each block takes its own, as built, so that the keys' gradients are
                # accumulated block by block as they always were.
                gk = None if torch.is_grad_enabled() else fast.keys.heads(heads).global_cast(cd)
                outs = [
                    _maybe_recompute(
                        lambda role=role, heads=heads, s=s, e=e, gk=gk: fast.group_block(role, heads, s, e, sparse, gk), recompute
                    )
                    for s, e in blocks
                ]
                out_g = torch.cat(outs, dim=2)
                if sparse and role != "dense":
                    out_g = fast.free_rows(role, heads, out_g, step, gk)
                parts.append(out_g)
            order.append(heads)
        out = torch.cat(parts, dim=1)[:, torch.argsort(torch.cat(order))]
    else:
        outs = [_maybe_recompute(lambda s=s, e=e: fast.all_heads_block(s, e), recompute) for s, e in blocks]
        out = torch.cat(outs, dim=2)
    if kl == length:
        return out
    # Position-free global queries: every key, unrotated (the fast path's operands). Their mask
    # rows - G rows, the same at every layer - are built once per forward (the context's memo).
    allowed_g, bias_g = _block_rows(ctx, kl, length, key=("rows", kl, length), keep_mask=True)
    s_g = _scores(fast.q[:, :, fast.rest], fast.k, scale, cd)
    if bias_g is not None:
        s_g = s_g + bias_g.float()
    s_g = s_g.masked_fill(~allowed_g, NEG_INF)
    o_glob = torch.matmul(_softmax_rows(s_g).to(cd), fast.v.to(cd))
    return torch.cat([out.to(o_glob.dtype), o_glob], dim=2)


def attention_chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
    """Ablation 6's dense-mask arm: ``attention_sdpa`` over query blocks of
    ``ctx.block_rows`` rows (whole boards), every key scored and masked."""
    return attention_sdpa(q, k, v, ctx, block_rows=ctx.block_rows)


def attention_block_sparse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
    """B14's block-sparse path: ``attention_sdpa`` over query blocks of ``ctx.block_rows``
    rows (whole boards), each masked head reading only the key boards its block reaches."""
    return attention_sdpa(q, k, v, ctx, block_rows=ctx.block_rows, sparse=True)


def attention_flex(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
    """B14's score function: torch's FlexAttention with every bias and mask computed inside
    its score and mask functions (``flex.py``) - eager on CPU: forward only, small inputs."""
    from .flex import attention_flex as flex

    return flex(q, k, v, ctx)


ATTENTION = {
    "block_sparse": attention_block_sparse,
    "chunked": attention_chunked,
    "sdpa": attention_sdpa,
    "explicit": attention_explicit,
    "dense": attention_dense,
    "flex": attention_flex,
}


class SelfAttention(nn.Module):
    """Q/K/V/output projections around the attention implementations (``ATTENTION``). F5
    (section 9.3): no biases in the Q, K and V projections; the output projection keeps its
    bias (``NetConfig.qkv_bias``, ``NetConfig.out_ffn_bias``). The four are trunk matrix
    multiplications, so they carry G2's format (``quant.QuantLinear``; off: ``nn.Linear``)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        d = cfg.d_model

        def linear(bias: bool) -> QuantLinear:
            return QuantLinear(d, d, bias=bias, fmt=cfg.trunk_quant, rule=cfg.trunk_quant_scale, weight_scale=cfg.trunk_quant_weight_scale)

        self.q_proj = linear(cfg.qkv_bias)
        self.k_proj = linear(cfg.qkv_bias)
        self.v_proj = linear(cfg.qkv_bias)
        self.out_proj = linear(cfg.out_ffn_bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.n_heads, self.d_head).transpose(1, 2)

    def qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._split(self.q_proj(x)), self._split(self.k_proj(x)), self._split(self.v_proj(x))

    def output(self, o: torch.Tensor) -> torch.Tensor:
        """The output projection of attention outputs ``(B, H, L, dv)``."""
        bsz, _, length, _ = o.shape
        return self.out_proj(o.transpose(1, 2).reshape(bsz, length, -1))

    def forward(self, x: torch.Tensor, ctx: AttnContext, kv_hook: Callable | None = None) -> torch.Tensor:
        """``kv_hook(k, v) -> (k, v)``: sees (and may replace) this layer's unrotated keys
        and values ``(B, H, L, d_head)`` before attention - stage N3 captures K/V with it
        and injects cached ones (``cache.py``)."""
        q, k, v = self.qkv(x)
        if kv_hook is not None:
            k, v = kv_hook(k, v)
        return self.output(ATTENTION[ctx.impl](q, k, v, ctx))

    def probabilities(self, x: torch.Tensor, ctx: AttnContext) -> torch.Tensor:
        """Attention probabilities ``(B, H, L, L)`` (explicit path; for inspection)."""
        q, k, v = self.qkv(x)
        _, probs = attention_explicit(q, k, v, ctx, return_probs=True)
        return probs
