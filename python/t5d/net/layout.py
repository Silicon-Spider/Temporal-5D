"""Register A8, the board-layout embedding - a prototype behind its flag, off (spec section 2.2;
gate 6p; ``NetConfig.board_layout_embedding``).

Section 2.2: "Lc0 found its first attention layer doing little and fixed it by giving every
square its whole board's piece layout at layer 0: the board's one-hot piece map, flattened and
projected per square to a few channels, concatenated to the token input, with an FFN after the
embedding ... Here it is per board: one projection from the board's flattened piece map (64
squares x 24 owner-and-piece states) to each of its squares. At 16 channels per square that is
about 1.6M parameters ... Boards smaller than 8 x 8 are padded. An arm of 6p, committed-class like
any input." Register A8: "each square gets its whole board's flattened piece layout at layer 0,
projected per square (16 channels, ~1.6M parameters), with an FFN after the embedding".

As built (``BoardLayoutEmbedding``), per board of every position:

1. **The layout** - a one-hot over ``LAYOUT_FRAME x LAYOUT_FRAME = 64`` cells x ``LAYOUT_STATES =
   24`` owner-and-piece states, flattened to 1,536: state ``12 * (owner - 1) + (piece - 1)`` - ours
   first, then theirs (register A7's owners), each in ``state.PIECE_LETTERS`` order; an empty cell
   is all zero. Has-moved is not among the 24 states (read literally; spec issue 240(e)). The cell
   is the square's **canonical** coordinate (A7, section 15.2's A7 row - the build log's note under
   A7): ``(y_c + oy) * 8 + (x + ox)``, with ``y_c`` the rank as the acting player sees it (mirrored
   when Black acts) and ``x`` the file, read from the batch's canonical coordinates - never the
   storage order, which would break A7 for Black.
2. **Other board sizes** (spec issue 240(d)). A ``W x H`` board smaller than 8 x 8 is padded into
   the frame - "boards smaller than 8 x 8 are padded" - its placement unstated: a PLACEHOLDER,
   ``NetConfig.layout_padding``: ``"a1"`` (default) puts the board's canonical a1 on the frame's
   a1 (``ox = oy = 0``), ``"centre"`` centres it (``ox = (8 - W) // 2``, ``oy = (8 - H) // 2``).
   Both are placed in the canonical frame, so a position and its colour-swapped mirror get the same
   layout. The padding cells are empty, indistinguishable from empty squares (the 24 states have no
   "off the board"); the projection learns one map shared by every size. A board larger than 8 x 8
   is refused (``ValueError``): the spec's frame is 8 x 8 and no variant the engine sets up has one
   (a placeholder fixed in code).
3. **The projection** ``proj``: ``Linear(1536, 64 * C)`` with a bias, ``C = layout_channels``
   (section 2.2's 16): 1,573,888 parameters at 16 - the "~1.6M" (the spec's figure counts the
   weight, 1,572,864). Each square token takes the ``C`` channels of its own cell; registers and
   padding take none.
4. **"Concatenated to the token input"** (spec issue 240(c)): the token input here is a sum of
   learned embeddings - a linear map of a concatenated one-hot - so concatenating the ``C`` channels
   and projecting the concatenation to ``d`` is the sum with a learned ``Linear(C, d)`` without a
   bias (``merge``; 4,096 parameters at 16): the same function family, built as the sum.
5. **"With an FFN after the embedding"** (spec issue 240(b)): its form is unstated. A PLACEHOLDER:
   one pre-norm residual block, ``x + W2 mish(W1 RMSNorm(x))`` - F5's RMSNorm-style gain (no bias,
   ``norm_eps``), F6's Mish (fixed: the trunk's ablation-6y switch, ``ffn_activation``, is
   "trunk-only" and does not reach it; spec issue 221's question - the input side's activations -
   covers it), width ``layout_ffn_ratio * d`` (8x, the trunk's: 1,051,136 parameters with F5's
   kept biases, ``out_ffn_bias``), on every valid token - squares, per-board and global registers
   (Lc0's tokens are all squares, so "after the embedding" is every token); padding stays zero.
   ``layout_ffn_ratio = 0`` drops it - not the register's A8, which has one: section 15.3's cost
   row's "without the FFN" probe.

Every step is per board and per token: a token's layout channels read its own board's contents
and its own canonical cell, nothing of other boards and nothing of the board's state-dependent
flags - so under the block-causal scope a frozen board's embedding stays intrinsic (with its
perspective, as every board input under A7 - ``cache.board_ident(..., acting=True)``, spec issue
303), and the cached path, which embeds the node's tokens through ``T5DNet.embed_tokens``, computes
it as the full pass does (C2, C17: cached = dense). **Its cost there**: ``CachedEvaluator.prepare``
embeds every token of a node, frozen boards' included, so the FFN - per token, most of A8's
FLOPs - is paid over the whole position at every node, while the node's trunk work is its live
set's: a larger share of a cached node than of the dense pass (``docs/MEASUREMENTS.md``, "A8";
spec issue 240(g)). A frozen board's embedding feeds only its K/V, which a node finds stored, so
embedding only the live boards' and the missing frozen boards' tokens would remove most of it -
not built (the node's embedding is made before its missing boards are known).

**Precision.** The embedding runs before the trunk's autocast: FP32 under either G1 trunk. G2's
8-bit trunk does not reach it - its layers are plain ``nn.Linear``, never ``quant.QuantLinear``
(spec issue 238(d): "A8 when built" among the unquantised; a placeholder).

**Initialisation** (``model._init_weights``): nothing is drawn while the module is made, and its
parameters are drawn last, from a generator of their own (``model.a8_seed``) - LeCun normal
weights, biases ``N(0, init_bias_std)``, the norm's gain ``1 + N(0, init_norm_std)`` - so switching
it on changes no other parameter, and with it off the network is the network without it, bit for
bit.

Written from the spec's description; Lc0's code was not consulted (AGENTS.md).
"""

from __future__ import annotations

import torch
from torch import nn

from .config import NetConfig
from .state import N_PIECE_TYPES, OWNER_OURS, OWNER_THEIRS, TOKEN_SQUARE, Batch
from .trunk import RMSNorm

#: The layout's frame: 8 x 8 cells (section 2.2: "64 squares"); smaller boards are padded into it.
LAYOUT_FRAME = 8
LAYOUT_CELLS = LAYOUT_FRAME * LAYOUT_FRAME
#: Owner-and-piece states per cell (section 2.2: "24 owner-and-piece states"): 12 piece types x ours,
#: theirs.
LAYOUT_PIECES = N_PIECE_TYPES - 1
LAYOUT_STATES = 2 * LAYOUT_PIECES
#: The flattened layout's width: 64 x 24 = 1,536.
LAYOUT_FEATURES = LAYOUT_CELLS * LAYOUT_STATES


def layout_parameters(cfg: NetConfig) -> dict[str, int]:
    """A8's parameter count by part at ``cfg`` (whether or not its flag is on): the projection
    (weight and bias), the merge into ``d``, and the FFN (the norm's gain and both matrices with
    F5's biases) - ``"total"`` their sum."""
    d, c, r = cfg.d_model, cfg.layout_channels, cfg.layout_ffn_ratio
    proj = LAYOUT_FEATURES * LAYOUT_CELLS * c + LAYOUT_CELLS * c
    merge = c * d
    bias = int(cfg.out_ffn_bias)
    ffn = 0 if r == 0 else d + 2 * d * (r * d) + bias * (r * d + d)
    return {"projection": proj, "merge": merge, "ffn": ffn, "total": proj + merge + ffn}


class BoardLayoutEmbedding(nn.Module):
    """Register A8 (the module docstring): ``forward(batch, x)`` returns the token input ``x``
    ``(B, L, d)`` with each square's layout channels added and the embedding's FFN applied."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, c = cfg.d_model, cfg.layout_channels
        # Made without drawing from the RNG: drawn from A8's own generator (``model._init_weights``).
        self.proj = nn.utils.skip_init(nn.Linear, LAYOUT_FEATURES, LAYOUT_CELLS * c)
        self.merge = nn.utils.skip_init(nn.Linear, c, d, bias=False)
        self.ffn_ratio = cfg.layout_ffn_ratio
        if self.ffn_ratio:
            self.norm = RMSNorm(d, cfg.norm_eps)
            self.ffn_in = nn.utils.skip_init(nn.Linear, d, self.ffn_ratio * d, bias=cfg.out_ffn_bias)
            self.ffn_out = nn.utils.skip_init(nn.Linear, self.ffn_ratio * d, d, bias=cfg.out_ffn_bias)
        else:
            self.norm = self.ffn_in = self.ffn_out = None

    def offsets(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """``(ox, oy)``, each ``(B,)`` long: where each position's boards sit in the 8 x 8 frame
        (``NetConfig.layout_padding``); a board larger than the frame is refused."""
        sizes = torch.tensor([(lay.width, lay.height) for lay in batch.layouts], dtype=torch.long, device=batch.piece.device)
        if bool((sizes > LAYOUT_FRAME).any()):
            raise ValueError(
                f"the board-layout embedding (A8) takes boards up to {LAYOUT_FRAME} x {LAYOUT_FRAME} (section 2.2's frame); "
                f"got {sorted({(lay.width, lay.height) for lay in batch.layouts})}"
            )
        if self.cfg.layout_padding == "centre":
            pad = (LAYOUT_FRAME - sizes) // 2
            return pad[:, 0], pad[:, 1]
        zero = torch.zeros(len(batch.layouts), dtype=torch.long, device=batch.piece.device)
        return zero, zero

    def cells(self, batch: Batch) -> torch.Tensor:
        """``(B, L)`` long: each square token's cell in its board's canonical 8 x 8 frame, ``-1`` for
        every other token."""
        ox, oy = self.offsets(batch)
        x = batch.coords[..., 0].long() + ox[:, None]
        y = batch.coords[..., 1].long() + oy[:, None]  # the canonical rank (A7)
        square = batch.valid & (batch.kind == TOKEN_SQUARE)
        return torch.where(square, y * LAYOUT_FRAME + x, torch.full_like(x, -1))

    def layouts(self, batch: Batch) -> torch.Tensor:
        """``(B, Nmax, 1536)``: every board's flattened one-hot layout (cell-major, 24 states per
        cell), zero for padding boards; in the projection's dtype (FP32 - or FP64 for a test's
        reference)."""
        bsz = batch.batch_size
        n_max = batch.board_valid.shape[1]
        cell = self.cells(batch)
        occupied = (cell >= 0) & (batch.piece > 0) & ((batch.owner == OWNER_OURS) | (batch.owner == OWNER_THEIRS))
        state = LAYOUT_PIECES * (batch.owner - OWNER_OURS) + (batch.piece - 1)
        pos, tok = torch.nonzero(occupied, as_tuple=True)
        out = torch.zeros(bsz, n_max, LAYOUT_FEATURES, dtype=self.proj.weight.dtype, device=batch.piece.device)
        out[pos, batch.board[pos, tok], cell[pos, tok] * LAYOUT_STATES + state[pos, tok]] = 1.0
        return out

    def channels(self, batch: Batch) -> torch.Tensor:
        """``(B, L, C)``: each square token's ``C`` layout channels - the projection of its own
        board's layout at its own cell - and zero for every other token."""
        c = self.cfg.layout_channels
        per_board = self.proj(self.layouts(batch))  # (B, Nmax, 64 * C)
        bsz, n_max = per_board.shape[:2]
        per_cell = per_board.reshape(bsz, n_max * LAYOUT_CELLS, c)
        cell = self.cells(batch)
        square = cell >= 0
        index = torch.where(square, batch.board.clamp(min=0) * LAYOUT_CELLS + cell, torch.zeros_like(cell))
        gathered = torch.gather(per_cell, 1, index[..., None].expand(-1, -1, c))
        return torch.where(square[..., None], gathered, torch.zeros((), dtype=gathered.dtype, device=gathered.device))

    def forward(self, batch: Batch, x: torch.Tensor) -> torch.Tensor:
        square = (batch.valid & (batch.kind == TOKEN_SQUARE))[..., None]
        zero = torch.zeros((), dtype=x.dtype, device=x.device)
        x = x + torch.where(square, self.merge(self.channels(batch)), zero)
        if self.ffn_in is None:
            return x
        h = self.ffn_out(torch.nn.functional.mish(self.ffn_in(self.norm(x))))
        return x + torch.where(batch.valid[..., None], h, zero)
