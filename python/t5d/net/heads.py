"""Output heads (D1, D3, D4; spec sections 8.1, 8.3, 8.6, 12.4). All run in FP32 (G1).

**Policy - bilinear pointer head (D1, section 8.1).** ``score(move) = f(h_src)^T
g(h_dst) / sqrt(d_policy)`` with separate learned projections ``f`` and ``g``,
evaluated only over the supplied candidate moves; padding entries score ``-inf``.
Endpoints are square tokens only - registers are never move endpoints (section 5.3).
Promotions add a learned per-piece logit offset read from the destination token, "because
the bilinear form gives every promotion piece the same score" (section 8.1, D1; M20:
"promotion scoring -> learned per-piece offset"): ``PointerHead``'s ``promotion_offset``,
always on in the policy.

**Submit (D8, section 8.2; stage N5).** Under the board-move split submitting is an action
with "its own policy logit, read from the pooled global registers as the value head is"
(``SubmitHead``; M20: no single register is assigned a role, so D7 holds): offered where
submitting is possible (T5), masked to -inf unless it is legal (C4), and normalised together
with the move logits - one distribution over the moves and submit (``NetOutput``). The
readout's form (linear, or the policy's bilinear form) is not ruled: ``NetConfig.submit_head``.

**Value (D3, section 8.3).** 51-bin categorical over outcomes in [-1, 1], bin centres
``-1 + 2k/50``; pooled by mean from the global registers (D3: "mean-pooled over the global
registers"; M20: "value pooling -> mean"), through a hidden layer with Mish (register
F6 since CHECKPOINT 8: "Mish ... in the trunk FFN and every head's hidden layers"; ablation
6y's squared ReLU is "the trunk-only arm", so no head follows ``NetConfig.ffn_activation``).
The value head's is the only hidden layer among the heads: the others are linear or
bilinear. Helpers give the distribution, its
mean and its variance (what variance-scaled cPUCT and the time manager consume). The
value is from the perspective of the **acting player** (``PositionState.acting_player``,
register D8: the player whose action is in progress until it submits - the present's colour,
``h_present & 1``, except once the present has shifted inside the action) - a target
convention for training, stated here so that every consumer reads it the same way; it is the
search's (``t5d/capi_search.h``: values "from the perspective of the leaf's acting
player").

**Auxiliary heads (D4, section 12.4).** Linear heads ("head compute is negligible -
linear heads over d = 256"). Where each reads (M20; ``model.T5DNet.heads``): the value-type
heads - WDL and the short-term value, beside the 51-bin value head - the mean of the
global registers (``global_register_pool``; D3 and section 8.3: "each pooled by mean from
the global registers"); the other global heads - moves-left, search effort - the mean of
every square token (``square_pool``; M20: "aux pooling -> squares" - section 12.4:
"Auxiliary heads pool from the trunk; hard-assigning registers to auxiliary quantities
conscripts the scratch space", register B8 and D7); the per-board heads the mean of each
board's *square* tokens (section 9.2), so the per-board registers stay free; the
per-square heads every token. Pooling all sixteen global registers assigns none a role, so
D7 holds for the value-type heads too (D8's argument for the submit logit).

* WDL: 3 logits (win, draw, loss for the acting player).
* Moves-left: one scalar, the predicted ``log1p`` of the number of turns left in the game,
  a turn being one player's action, from a turn start to its submission (D4: "moves-left
  (in turns)"; M20: "moves-left -> turns"; the unit is spec issue 305's, so every D2 arm
  shares it). Which turns count - the one in progress among them or not - is the target
  builder's (the training loop, M21); section 8.6's allocator converts turns to the
  board-moves it projects N by (E11's boards-per-turn figure, issue 305). Log scale because
  the target spans 0 to hundreds; train with a Huber/L2 loss on ``log1p(target)``
  (implementation choice).
* Opponent reply: a second pointer head over the opponent's candidate moves, always built,
  with no promotion offset (as built at N1: D1 names the policy). D4 reinstated it as a
  prototype at CHECKPOINT 7 - behind a flag, weight 0.15, its target mapped onto current
  squares - which the heads' rebuild after engine ABI round 2 builds; this head is left as
  it was until then.
* Short-horizon value: categorical like the value head.
* Per board: travelled-into within k plies; timeline still active at game end;
  tactical volatility - one logit each (k is a target-construction parameter).
* Per square: a small multi-label vector of deep attack facts (``SQUARE_LABELS``; the set
  stands as v1 - M20 - tested by ablation 6j).
* Search effort (section 8.6): categorical over log2-visit bins with a censored top bin
  ("did not converge within budget").
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import NetConfig
from .state import NO_PROMOTION, Batch

#: Per-board auxiliary logits, in output order.
BOARD_HEADS: tuple[str, ...] = ("travelled_into", "timeline_active_at_end", "tactical_volatility")

#: Per-square deep attack facts - the set STANDS AS V1 (M20: "the per-square label set ...
#: stand[s] as v1", tested by ablation 6j; register D4: "label set a placeholder"). Section
#: 12.4 specifies only "deep attack facts beyond the section 6 input bias" (generator-exact),
#: and that "predicting what the model is handed as input teaches nothing". The section 6
#: input is the legal move graph (B9, committed) and typed attack / defence / pin / x-ray /
#: check edges (B11, provisional; travel-threat folded into attack at CHECKPOINT 6), so the
#: one-step facts those state - attacked-by, pinned, a legal move's destination - are
#: excluded. Each candidate below is a fact no input relation states: multi-step (an
#: exchange sequence, two-move reach, the position after a move) or a geometric
#: neighbourhood that is no relation type (the royal zone; for the side to move it partly
#: overlaps the king's legal moves). The labels' generator (exact, from the rules engine) is
#: not built (step 2, the engine lane). Facts are about the whole multiverse, cross-board
#: lines included. Register A7: the network reads every input from the acting player's side,
#: so the colour-specific labels are the acting player's ("ours") and the opponent's
#: ("theirs") - a relabelling of the set's white/black facts, not a new set (spec issue 226) -
#: and ``square_labels_by_colour`` maps them back to colours:
SQUARE_LABELS: tuple[str, ...] = (
    "exchange_loses_occupant",  # the static exchange on the square loses material for the occupant's side
    "reachable_in_two_ours",  # some piece of the acting player reaches the square in exactly two of its own moves
    "reachable_in_two_theirs",
    "arrival_gives_check_ours",  # some piece of the acting player moving onto the square gives check from it
    "arrival_gives_check_theirs",
    "royal_zone_ours",  # within one 4D king step of a royal piece of the acting player
    "royal_zone_theirs",
    "arrival_gives_mate",  # some legal move of the acting player onto the square mates
)

#: The labels by colour (white/black), in ``SQUARE_LABELS``' order with "ours" read as White.
SQUARE_LABELS_BY_COLOUR: tuple[str, ...] = tuple(name.replace("_ours", "_white").replace("_theirs", "_black") for name in SQUARE_LABELS)


def square_labels_by_colour(square_logits: torch.Tensor, acting_player: torch.Tensor) -> torch.Tensor:
    """Register A7's "outputs mapped back" for the per-square labels: ``square_logits``
    ``(B, L, len(SQUARE_LABELS))`` (ours/theirs) as ``SQUARE_LABELS_BY_COLOUR`` (white/black),
    given each position's acting player ``(B,)`` (0 white, 1 black) - the ours and theirs
    channels swapped where Black acts. The tokens need no mapping (the canonical view moves
    none - ``state.py``)."""
    swap = torch.arange(len(SQUARE_LABELS), device=square_logits.device)
    for i, name in enumerate(SQUARE_LABELS):
        if name.endswith("_ours"):
            j = SQUARE_LABELS.index(name[: -len("_ours")] + "_theirs")
            swap[i], swap[j] = j, i
    black = (acting_player.to(square_logits.device) == 1)[:, None, None]
    return torch.where(black, square_logits[..., swap], square_logits)


# --- categorical helpers ------------------------------------------------------------------


def value_bin_centres(n_bins: int = 51, device: torch.device | str | None = None) -> torch.Tensor:
    """Bin centres on [-1, 1]: ``(2k - (n-1)) / (n-1)``; 51 bins put -1, 0 and +1 on centres.

    Computed from integers rather than with ``torch.linspace``, whose rounding differs by
    platform (on Linux the middle centre came out 2.2e-8, not 0): a draw must sit exactly
    on a centre."""
    k = torch.arange(n_bins, dtype=torch.float64, device=device)
    return ((2.0 * k - (n_bins - 1)) / (n_bins - 1)).to(torch.float32)


def categorical_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def categorical_mean(probs: torch.Tensor, centres: torch.Tensor | None = None) -> torch.Tensor:
    centres = value_bin_centres(probs.shape[-1], probs.device) if centres is None else centres
    return (probs * centres).sum(dim=-1)


def categorical_variance(probs: torch.Tensor, centres: torch.Tensor | None = None) -> torch.Tensor:
    centres = value_bin_centres(probs.shape[-1], probs.device) if centres is None else centres
    mean = (probs * centres).sum(dim=-1, keepdim=True)
    return (probs * (centres - mean) ** 2).sum(dim=-1)


def value_to_twohot(value: torch.Tensor, n_bins: int = 51) -> torch.Tensor:
    """Target distribution for a scalar value in [-1, 1]: weight split between the two
    nearest bin centres so that the target's mean is the value (exact on a centre)."""
    x = (value.float().clamp(-1.0, 1.0) + 1.0) * 0.5 * (n_bins - 1)
    lower = x.floor().long().clamp(max=n_bins - 1)
    upper = (lower + 1).clamp(max=n_bins - 1)
    w_upper = x - lower.float()
    out = torch.zeros(*value.shape, n_bins, dtype=torch.float32, device=value.device)
    out.scatter_add_(-1, lower[..., None], (1.0 - w_upper)[..., None])
    out.scatter_add_(-1, upper[..., None], w_upper[..., None])
    return out


def effort_bin(visits: torch.Tensor, converged: torch.Tensor, n_bins: int = 16) -> torch.Tensor:
    """Effort target bin: ``floor(log2 visits)`` capped at ``n_bins - 2`` when the decision
    converged within the search; the censored top bin ``n_bins - 1`` when it did not."""
    k = torch.floor(torch.log2(visits.float().clamp(min=1.0))).long().clamp(max=n_bins - 2)
    return torch.where(converged.bool(), k, torch.full_like(k, n_bins - 1))


def effort_bin_lower_edges(n_bins: int = 16) -> list[float]:
    """Lower visit edge of each bin; the censored bin's is infinite."""
    return [float(2**k) for k in range(n_bins - 1)] + [math.inf]


def effort_quantile(probs: torch.Tensor, q: float) -> torch.Tensor:
    """Bin index at the ``q`` quantile (the censored bin counts as the largest)."""
    cdf = probs.float().cumsum(dim=-1)
    return (cdf < q).sum(dim=-1).clamp(max=probs.shape[-1] - 1)


def effort_censored_prob(probs: torch.Tensor) -> torch.Tensor:
    """Probability of "did not converge within budget" - the won't-resolve flag of 8.6."""
    return probs[..., -1]


def moves_left_from_log(x: torch.Tensor) -> torch.Tensor:
    """Turns left - each one player's action (D4, M20; spec issue 305) - from the moves-left
    head's ``log1p`` prediction; never negative."""
    return torch.expm1(x.float().clamp(min=0.0))


# --- pooling ---------------------------------------------------------------------------------


def square_pool(h: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean of every square token of each position, ``(B, d)`` (registers excluded) - what
    the global auxiliary heads read (moves-left, search effort; M20: "aux pooling ->
    squares")."""
    w = batch.square_mask.to(h.dtype)
    return (h * w[..., None]).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1.0)


def global_register_pool(h: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean of the global registers, ``(B, d)`` - what the value-type heads (the 51-bin
    value, WDL, the short-term value; D3, section 8.3, M20: "value pooling -> mean") and D8's
    submit logit (section 8.2, M20) read. With no global registers (an ablation-3 arm),
    ``square_pool`` instead - as built since N1, a placeholder (no document says what these
    heads read without them; spec issue 225)."""
    if batch.global_regs > 0:
        return h[:, batch.body_len : batch.body_len + batch.global_regs].mean(dim=1)
    return square_pool(h, batch)


def board_mean_pool(h: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean of each board's square tokens, ``(B, Nmax, d)`` (registers excluded)."""
    bsz, _, d = h.shape
    n_max = batch.board_valid.shape[1]
    sq = batch.square_mask
    index = torch.where(sq, batch.board, torch.full_like(batch.board, n_max))  # dump bucket
    sums = h.new_zeros(bsz, n_max + 1, d).scatter_add_(1, index[..., None].expand(-1, -1, d), h * sq[..., None])
    counts = h.new_zeros(bsz, n_max + 1).scatter_add_(1, index, sq.to(h.dtype))
    return sums[:, :n_max] / counts[:, :n_max, None].clamp(min=1.0)


# --- heads -------------------------------------------------------------------------------------


class PointerHead(nn.Module):
    """``score = f(h_src)^T g(h_dst) / sqrt(dim)`` over supplied moves (D1), plus, with
    ``promotion_offset``, a learned per-piece offset for a promotion: ``promo(h_dst)[piece]``,
    one logit per piece type read from the destination token (section 8.1: "Promotions add a
    learned per-piece logit offset"; Chessformer's promotion bias). The policy has it (D1,
    M20); the opponent-reply head is built without it, as at N1 (D4's prototype is rebuilt
    with the heads)."""

    def __init__(self, d_model: int, dim: int, promotion_offset: bool, n_piece_types: int = 13, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.src = nn.Linear(d_model, dim, bias=bias)  # f
        self.dst = nn.Linear(d_model, dim, bias=bias)  # g
        self.promo = nn.Linear(d_model, n_piece_types, bias=bias) if promotion_offset else None

    def forward(self, h: torch.Tensor, moves: torch.Tensor, valid: torch.Tensor, square_mask: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, M)``; ``moves`` ``(B, M, 3)`` hold (src token, dst token, promotion)."""
        bsz, n_moves, _ = moves.shape
        if n_moves == 0:
            return h.new_zeros(bsz, 0)
        d = h.shape[-1]
        src_tok = moves[..., 0].clamp(min=0)
        dst_tok = moves[..., 1].clamp(min=0)
        ok_src = square_mask.gather(1, src_tok)
        ok_dst = square_mask.gather(1, dst_tok)
        if bool((valid & ~(ok_src & ok_dst)).any()):
            raise ValueError("a move endpoint is not a square token (registers are never move endpoints)")
        h_src = h.gather(1, src_tok[..., None].expand(bsz, n_moves, d))
        h_dst = h.gather(1, dst_tok[..., None].expand(bsz, n_moves, d))
        score = (self.src(h_src) * self.dst(h_dst)).sum(dim=-1) / math.sqrt(self.dim)
        if self.promo is not None:
            offsets = self.promo(h_dst).gather(-1, moves[..., 2:3].clamp(min=0)).squeeze(-1)
            score = score + torch.where(moves[..., 2] != NO_PROMOTION, offsets, torch.zeros_like(offsets))
        return score.masked_fill(~valid, float("-inf"))


class SubmitHead(nn.Module):
    """Register D8 (section 8.2): the submit action's own policy logit, read from the pooled
    global registers as the value head is (``global_register_pool``: their mean - M20, so no
    register is assigned a role and D7 holds) - by a learned linear readout (``"linear"``) or by
    the policy's bilinear form with the pooled state at both ends (``"pointer"``;
    ``NetConfig.submit_head``, the readout's form a placeholder - spec issue 109). Returns the
    unmasked logit ``(B,)``; the model masks it."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.form = cfg.submit_head
        # Made without drawing from the RNG; drawn from D8's own generator
        # (``model._init_weights``), so that it changes no other parameter's draw.
        self.out = nn.utils.skip_init(nn.Linear, cfg.d_model, 1) if self.form == "linear" else None

    def forward(self, h: torch.Tensor, batch: Batch, policy: PointerHead) -> torch.Tensor:
        r = global_register_pool(h, batch)
        if self.out is not None:
            return self.out(r).squeeze(-1)
        return (policy.src(r) * policy.dst(r)).sum(dim=-1) / math.sqrt(policy.dim)


class ValueHead(nn.Module):
    """``out(mish(hidden(pooled)))``: F6's Mish in the head's hidden layer, whatever the trunk
    FFN's activation (6y's squared ReLU is a trunk-only arm - CHECKPOINT 8)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.hidden = nn.Linear(cfg.d_model, cfg.value_hidden)
        self.out = nn.Linear(cfg.value_hidden, cfg.value_bins)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.mish(self.hidden(pooled)))


class AuxHeads(nn.Module):
    """D4's heads (the module docstring): what each reads is the model's (``T5DNet.heads``)."""

    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.wdl = nn.Linear(d, 3)
        self.moves_left = nn.Linear(d, 1)
        self.short_value = nn.Linear(d, cfg.value_bins)
        self.effort = nn.Linear(d, cfg.effort_bins)
        self.board = nn.Linear(d, len(BOARD_HEADS))
        self.square = nn.Linear(d, cfg.n_square_labels)
        self.opp_reply = PointerHead(d, cfg.opp_policy_dim, promotion_offset=False)
