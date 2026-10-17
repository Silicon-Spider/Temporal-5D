"""Board-pair smolgen (B12, prototype, flag ``board_pair_smolgen``; spec sections 6, 9.2, 14.4).

Section 6 keeps Lc0's smolgen at one level only: "full-rank biases over 240^2 board pairs
generated from the global registers", aimed at context-dependent board relevance on top
of the exact-but-static typed edges. Section 9.2 sizes it: "Smolgen latent 32, bilinear
per head", under 0.1M parameters. Token-level and within-board smolgen are refused
(B13, section 14.4) and do not exist here.

**The latent** (per layer, from that layer's attention input ``h`` - the normed residual
stream under F5's pre-norm, the block's input under its post-norm arm,
``trunk.Block.attn_input``): the global context ``c`` is the mean of the global registers'
states and ``z = silu(W_z c + b_z)`` (32-dim; SiLU as in Lc0's smolgen) - the part
generated "from the global registers". There is no norm inside the generator (F5's
"everywhere except inside smolgen" finds none to keep - build log, spec issue 202).

**Two generators** turn ``z`` into per-head biases over board pairs (``smolgen_form``;
B12 is class R, so the form is an implementation choice and both are built):

* ``"relative_cells"`` (the default): ``S_h(b1, b2) = z . W_h[cell(b1, b2)] / sqrt(32)``.
  ``cell`` is the pair's relative cell - B6's signed Delta-time bucket and Delta-timeline
  bucket of the key board against the query board (19 x 8 = 152 cells at B6's default
  buckets) - and ``W_h`` a learned 32-vector per cell per head. This is Lc0's construction
  with its output grid moved from absolute pairs to relative cells: Lc0's 64 x 64 output
  layer works because squares have fixed indices, while boards have none (N varies, and
  absolute board indices are an absolute positional encoding, section 14.1) - their
  offsets are what is fixed. It is bilinear per head in (z, the pair's cell), generated
  from the global registers alone, and full-rank over board pairs: a bias that is a
  function of the pair's offset is not a low-rank form ``u_b1^T M v_b2`` (the tests
  measure rank N at N = 40 on every head). What it cannot do is single out one board by
  its content - the global registers choose which *offsets* matter, not which boards.
* ``"board_bilinear"`` (stage N2's first build): each board ``b`` enters through a
  descriptor ``u_b = W_u r_b + b_u`` (32-dim), ``r_b`` the mean of the board's per-board
  registers - its summary tokens (sections 3.3, 5) - and ``S_h(b1, b2) = (A_h u_b1)^T
  diag(z) (B_h u_b2) / sqrt(32)``, with ``A_h``, ``B_h`` learned 32 x 32 matrices:
  context-modulated, asymmetric, and reading each board's own content. But its bias
  matrix has rank at most 32 per head (full rank only up to 32 boards), and it reads the
  per-board registers as well as the global ones: it is the low-rank form section 6
  equates with appending 32 context-modulated dims to Q and K - "the d_head lever in a
  costume" - at board granularity.

The bias applies to every token pair by the boards the two tokens belong to - squares and
per-board registers alike. **Global registers are excluded**: they belong to no board, so
a pair with one gets no smolgen bias (as with B6's binned biases); padding neither.

Weights are shared by all layers (implementation choice, forced by the budget: per-layer
weights would be 16 x 47.1k = 0.75M for relative cells, 16 x 32.8k = 0.53M for the
bilinear form, against section 9.2's "<0.1M"); the bias still differs per layer because
each layer's input does. Parameters: ``W_z`` 256 x 32 + 32 = 8,224, plus ``W`` 8 x 152 x
32 = 38,912 (relative cells: 47,136 in all) or ``W_u`` 8,224 and ``A``, ``B`` 2 x (8 x 32
x 32) (bilinear: 32,832). Cost per layer at 240 boards: relative cells 8 x 152 x 32 MAC
and one gather over the 240^2 pairs; bilinear ~6 MMAC for the descriptors and their
per-head projections and 8 x 240^2 x 32 = 14.7 MMAC for the pair forms - against
section 6's ~8 GMAC of attention per head per layer, negligible either way. (Section 6's
"14.7 MMAC per head per layer" is 256 x 240^2: a 256-wide output layer over every pair.)

Ablation arms without registers (ablation 3) fall back to square means: with no
per-board registers a board's descriptor (bilinear form) reads the mean of its squares,
and with no global registers the context reads the mean of every square of the position.

Initialisation (``reset_parameters``, called last by the model so that switching smolgen
on changes no other weight): ``W_z``, ``W_u`` LeCun normal, their biases
``N(0, init_bias_std)``; ``A``, ``B`` ``N(0, 1/latent)``; the cell table ``N(0, 1)`` - with
the ``1/sqrt(latent)`` scale both forms start with biases of spread of order one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import skip_init

from .config import NetConfig
from .rope import n_time_buckets, n_timeline_buckets, time_bucket, timeline_bucket
from .state import TOKEN_BOARD_REGISTER, TOKEN_SQUARE, Batch


@dataclass
class SmolgenInputs:
    """Where a layer's smolgen reads from, fixed per batch."""

    global_slice: slice  # the global registers' rows
    board_tokens: torch.Tensor  # (B, Nb, R') long: the tokens whose mean describes each board
    board_weight: torch.Tensor  # (B, Nb, R') float: 1 / count for real tokens, 0 for padding
    square_weight: torch.Tensor  # (B, L) float: 1 / squares of the position (no-global-registers arm)
    cell: torch.Tensor | None = None  # (B, Nb, Nb) long: relative cell of (query board, key board)


def relative_cells(ph: torch.Tensor, cfg: NetConfig) -> torch.Tensor:
    """``(B, Nb, Nb)`` relative cell of each (query board, key board) pair from per-board
    ``(p, h)`` ``(B, Nb, 2)``: B6's signed Delta-time bucket (key minus query) times the
    Delta-timeline buckets, plus the Delta-timeline bucket."""
    p, h = ph[..., 0].round().long(), ph[..., 1].round().long()
    dh = h[:, None, :] - h[:, :, None]
    dp = p[:, None, :] - p[:, :, None]
    return time_bucket(dh, cfg) * n_timeline_buckets(cfg) + timeline_bucket(dp, cfg)


def smolgen_inputs(batch: Batch, cfg: NetConfig, ph: torch.Tensor) -> SmolgenInputs:
    """The per-board register tokens of every board (or, with no per-board registers, its
    squares), the global registers' rows, and the board pairs' relative cells (``ph``:
    per-board ``(p, h)`` ``(B, Nb, 2)``)."""
    bsz, length = batch.kind.shape
    n_max = batch.board_valid.shape[1]
    kind = TOKEN_BOARD_REGISTER if batch.regs_per_board > 0 else TOKEN_SQUARE
    member = batch.valid & (batch.kind == kind) & (batch.board >= 0)
    per_board = int(max(lay.squares_per_board for lay in batch.layouts)) if kind == TOKEN_SQUARE else batch.regs_per_board
    tokens = torch.zeros(bsz, n_max, per_board, dtype=torch.long, device=batch.kind.device)
    weight = torch.zeros(bsz, n_max, per_board, dtype=torch.float32, device=batch.kind.device)
    for b in range(bsz):
        idx = torch.nonzero(member[b]).flatten()
        boards = batch.board[b, idx]
        slot = batch.reg_index[b, idx] if kind == TOKEN_BOARD_REGISTER else _rank_within(boards)
        tokens[b, boards, slot] = idx
        weight[b, boards, slot] = 1.0
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
    sq = batch.square_mask.float()
    square_weight = sq / sq.sum(dim=1, keepdim=True).clamp(min=1.0)
    cell = relative_cells(ph, cfg) if cfg.smolgen_form == "relative_cells" else None
    return SmolgenInputs(slice(batch.body_len, batch.body_len + batch.global_regs), tokens, weight, square_weight, cell)


def _rank_within(groups: torch.Tensor) -> torch.Tensor:
    """Position of each element within its group, for a sorted (board-major) group vector."""
    first = torch.searchsorted(groups, groups, right=False)
    return torch.arange(groups.shape[0], device=groups.device) - first


class BoardPairSmolgen(nn.Module):
    """B12: per-head board-pair biases generated from the global registers (``smolgen_form``).

    Constructed without drawing from the global RNG (``skip_init``, ``torch.empty``);
    ``reset_parameters`` draws its weights."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        r = cfg.smolgen_latent
        self.latent = r
        self.form = cfg.smolgen_form
        self.has_globals = cfg.global_regs > 0
        self.context = skip_init(nn.Linear, cfg.d_model, r)  # W_z: global registers -> latent z
        if self.form == "relative_cells":
            self.n_cells = n_time_buckets(cfg) * n_timeline_buckets(cfg)
            self.cells = nn.Parameter(torch.empty(cfg.n_heads, self.n_cells, r))  # W_h[cell]
        else:
            self.board = skip_init(nn.Linear, cfg.d_model, r)  # W_u: a board's registers -> descriptor u_b
            self.left = nn.Parameter(torch.empty(cfg.n_heads, r, r))  # A_h
            self.right = nn.Parameter(torch.empty(cfg.n_heads, r, r))  # B_h

    @torch.no_grad()
    def reset_parameters(self, cfg: NetConfig) -> None:
        """Draw every weight from the global RNG (see the module docstring)."""
        linears = [self.context] if self.form == "relative_cells" else [self.context, self.board]
        for lin in linears:
            nn.init.normal_(lin.weight, 0.0, lin.in_features**-0.5)
            nn.init.normal_(lin.bias, 0.0, cfg.init_bias_std)
        if self.form == "relative_cells":
            nn.init.normal_(self.cells, 0.0, 1.0)
        else:
            nn.init.normal_(self.left, 0.0, self.latent**-0.5)
            nn.init.normal_(self.right, 0.0, self.latent**-0.5)

    def forward(self, h: torch.Tensor, inputs: SmolgenInputs) -> torch.Tensor:
        """Board-pair biases ``(B, H, Nb, Nb)`` (query board, key board) from a layer's
        attention input ``h`` ``(B, L, d)``."""
        bsz, _, d = h.shape
        z = F.silu(self.context(self._context_input(h, inputs)))  # (B, r)
        if self.form == "relative_cells":
            values = torch.einsum("br,hcr->bhc", z, self.cells.to(z.dtype)) / math.sqrt(self.latent)  # (B, H, cells)
            n_max = inputs.cell.shape[1]
            index = inputs.cell.reshape(bsz, 1, n_max * n_max).expand(bsz, values.shape[1], n_max * n_max)
            return values.gather(2, index).reshape(bsz, values.shape[1], n_max, n_max)
        n_max, per_board = inputs.board_tokens.shape[1:]
        members = h.gather(1, inputs.board_tokens.reshape(bsz, -1, 1).expand(bsz, n_max * per_board, d))
        members = members.reshape(bsz, n_max, per_board, d)
        summary = (members * inputs.board_weight[..., None].to(members.dtype)).sum(dim=2)
        u = self.board(summary)  # (B, Nb, r)
        a = torch.einsum("bnk,hjk->bhnj", u, self.left.to(u.dtype))
        b = torch.einsum("bnk,hjk->bhnj", u, self.right.to(u.dtype))
        return torch.einsum("bhnj,bj,bhmj->bhnm", a, z.to(a.dtype), b) / math.sqrt(self.latent)

    def _context_input(self, h: torch.Tensor, inputs: SmolgenInputs) -> torch.Tensor:
        """The mean of the global registers (of every square, in the no-register arm)."""
        if self.has_globals:
            return h[:, inputs.global_slice].mean(dim=1)
        return (h * inputs.square_weight[..., None].to(h.dtype)).sum(dim=1)
