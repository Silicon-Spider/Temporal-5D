"""Losses for the network's heads (spec sections 8.1, 8.3, 12.4).

Each head's loss reads a target whose interface is fixed here (``Targets``); building the
targets from records - the pruned root value (section 12.2), the short-term averages, the
game result's WDL, deblundering (I13) - is the training loop's record-to-target builder (M21,
I15), not this module's.

* **D3's value heads** (section 8.3, as corrected at CHECKPOINT 7), every target from the
  acting player's perspective (D8):

  - WDL: cross-entropy against the game result (a class index, or a distribution - I13's
    deblundered targets are), blended towards the search's own WDL at the root by
    ``LossConfig.q_ratio`` (Lc0's ``q_ratio``; "a tuned knob, default zero", 6z).
  - The categorical: cross-entropy against ``heads.hl_gauss`` of the root search value
    (``Targets.root_value``) - of full searches only (``Targets.full_search``; section 12.5),
    the value "recomputed from the pruned visit counts" (section 12.2) by the builder. HL-Gauss'
    sigma is a placeholder (``hl_gauss_sigma``, in bin widths: Farebrother et al.'s 0.75; spec
    issue 231).
  - Short-term values: squared error against each horizon's exponentially averaged future
    search values (``Targets.short_term``, NaN where a horizon has none) - the form a
    placeholder (issue 231).
  - The error head: Huber loss against ``(short_term[:, error_horizon] - (W - L))^2``, the value
    the network's own WDL with no gradient (KataGo's uncertainty target: the squared gap between
    the value and the short-term value) - so no gradient flows from the error target into the
    WDL or the short-term heads. Which horizon, the stop-gradient and the loss are placeholders
    (issue 231).

* Policy and opponent reply: cross-entropy over each position's supplied candidate moves
  against a target distribution (visit counts for the policy; a one-hot reply for the
  opponent-reply head). Padding entries carry no mass. The policy's distribution is over the
  decision's actions - the candidate moves and, where it is offered and legal, the submit
  action (register D8, stage N5: ``NetOutput.action_logits``); its target is the root's visits
  to each move and to the submit child (``Targets.submit``) - of full searches only (section
  12.5: "only full-search moves emit policy targets").
* Effort: cross-entropy over its bins (censored bin included). Moves-left: Huber loss on
  ``log1p(turns left)`` - a turn one player's action (D4, M20; spec issue 305).
* Per-board and per-square heads: binary cross-entropy with logits, masked to real
  boards and squares; a per-board loss sums its three targets, a per-square loss
  averages its labels and then the board's squares, so both are one number per board.
  Their reduction over a position's boards is register D6's open knob ("sum-vs-mean",
  ablation 6j): ``board_reduction="mean"`` (default) averages; ``"sum"`` sums them, so
  the auxiliary gradient grows with N. Positions are then averaged with equal weight.

Every loss weight is a placeholder (6j) unless a document gives it.

The source rule (D5, section 12.4) is the caller's to keep: every target here must come
from the game record, the generator, scanner proofs or the engine's own search - never
from the learned verifier's estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..net.heads import hl_gauss
from ..net.model import NetOutput


@dataclass
class Targets:
    policy: torch.Tensor  # (B, M) distribution over supplied moves (with ``submit``: over the actions)
    #: D3's WDL target - the game result, the acting player's (D8): ``(B,)`` class index (0 win,
    #: 1 draw, 2 loss) or ``(B, 3)`` distribution (I13's deblundered targets are).
    wdl: torch.Tensor
    #: D3's categorical target: ``(B,)`` the root search value of a full search, recomputed from
    #: the pruned visits (section 12.2), on [-1, 1]; NaN where there is none (the loss smooths it,
    #: ``heads.hl_gauss``).
    root_value: torch.Tensor
    #: D3's short-term targets: ``(B, K)`` on [-1, 1], one per ``NetConfig.short_term_horizons``;
    #: NaN where a horizon has none.
    short_term: torch.Tensor
    moves_left: torch.Tensor  # (B,) turns left (one player's action each, D4), >= 0
    opp_reply: torch.Tensor  # (B,) index into opp_moves, -1 where there is no target
    board: torch.Tensor  # (B, Nmax, 3) in {0, 1}
    square: torch.Tensor  # (B, L, n_square_labels) in {0, 1}
    effort: torch.Tensor  # (B,) bin index, -1 where there is no target (PCR cheap searches)
    #: (B,) the policy target's mass on the submit action (D8; ``policy`` and ``submit``
    #: together are one distribution over the decision's actions); None: none anywhere.
    submit: torch.Tensor | None = None
    #: (B,) bool: the position's search was a full one (section 12.5, I5) - only those emit
    #: policy and categorical targets; None: every one was.
    full_search: torch.Tensor | None = None
    #: (B, 3): the search's WDL at the root - ``((1 - d + q) / 2, d, (1 - d - q) / 2)`` from the
    #: backed-up (q, d) (spec issue 148) - for ``LossConfig.q_ratio``'s blend; None: not given.
    root_wdl: torch.Tensor | None = None


@dataclass
class LossConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "policy": 1.0,
            "wdl": 1.0,
            "categorical": 1.0,
            "short_term": 1.0,
            "value_error": 1.0,
            "moves_left": 1.0,
            "opp_reply": 1.0,
            "board": 1.0,
            "square": 1.0,
            "effort": 1.0,
        }
    )
    board_reduction: str = "mean"  # D6: "mean" or "sum" over a position's boards
    huber_delta: float = 1.0  # moves-left's Huber loss
    #: D3: the WDL target's blend towards the search's own WDL at the root (Lc0's ``q_ratio``):
    #: "a tuned knob, default zero" (section 8.3, 6z) - 0 is exactly the game result.
    q_ratio: float = 0.0
    #: D3: HL-Gauss' sigma, in bin widths (Farebrother et al.: 0.75) - PLACEHOLDER (issue 231).
    hl_gauss_sigma: float = 0.75
    #: D3: the short-term horizon the error head's target reads (an index into
    #: ``NetConfig.short_term_horizons``; KataGo's uses its shortest) - PLACEHOLDER (issue 231).
    error_horizon: int = 0
    #: D3: the error head's Huber delta - PLACEHOLDER (issue 231).
    error_huber_delta: float = 1.0


def categorical_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def masked_ce(logits: torch.Tensor, valid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over variable-length candidate lists; positions without moves skip."""
    if logits.shape[-1] == 0:
        return logits.new_zeros(())
    logp = torch.log_softmax(logits.float(), dim=-1).masked_fill(~valid, 0.0)
    per = -(target * logp).sum(dim=-1)
    has = valid.any(dim=-1)
    return per[has].mean() if bool(has.any()) else logits.new_zeros(())


def index_ce(logits: torch.Tensor, valid: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 0:
        return logits.new_zeros(())
    keep = index >= 0
    if not bool(keep.any()):
        return logits.new_zeros(())
    logp = torch.log_softmax(logits.float(), dim=-1).masked_fill(~valid, 0.0)
    return -logp[keep].gather(-1, index[keep][:, None]).mean()


def _board_square_mean(per_token: torch.Tensor, token_board: torch.Tensor, square_mask: torch.Tensor, n_boards: int) -> torch.Tensor:
    """Mean of a per-token quantity over each board's square tokens, ``(B, n_boards)``."""
    index = torch.where(square_mask, token_board, torch.full_like(token_board, n_boards))  # dump bucket
    w = square_mask.to(per_token.dtype)
    values = torch.where(square_mask, per_token, torch.zeros_like(per_token))
    sums = per_token.new_zeros(per_token.shape[0], n_boards + 1).scatter_add_(1, index, values)
    counts = per_token.new_zeros(per_token.shape[0], n_boards + 1).scatter_add_(1, index, w)
    return sums[:, :n_boards] / counts[:, :n_boards].clamp(min=1.0)


def _reduce_boards(per_board: torch.Tensor, board_valid: torch.Tensor, reduction: str) -> torch.Tensor:
    """D6's knob over each position's real boards, then the mean over positions."""
    per_position = torch.where(board_valid, per_board, torch.zeros_like(per_board)).sum(dim=1)
    if reduction == "mean":
        per_position = per_position / board_valid.sum(dim=1).clamp(min=1)
    elif reduction != "sum":
        raise ValueError("board_reduction must be 'mean' or 'sum' (register D6)")
    return per_position.mean()


def wdl_target(tgt: Targets, q_ratio: float) -> torch.Tensor:
    """D3's WDL target ``(B, 3)``: the game result (a class index made one-hot, or a
    distribution), blended towards the search's WDL at the root by ``q_ratio`` (0: the result)."""
    result = tgt.wdl
    if result.dim() == 1:
        result = F.one_hot(result.long(), 3).float()
    result = result.float()
    if q_ratio == 0.0:
        return result
    if not 0.0 <= q_ratio <= 1.0:
        raise ValueError("q_ratio must be in [0, 1] (D3, 6z)")
    if tgt.root_wdl is None:
        raise ValueError("q_ratio > 0 blends towards the search's WDL: Targets.root_wdl is needed")
    return (1.0 - q_ratio) * result + q_ratio * tgt.root_wdl.float()


def error_target(short_term: torch.Tensor, search_value: torch.Tensor, horizon: int) -> torch.Tensor:
    """D3's error head target ``(B,)``: the squared gap between the short-term value target at
    ``horizon`` and the network's value (W - L, with no gradient); NaN where the horizon has none."""
    if not 0 <= horizon < short_term.shape[-1]:
        raise ValueError(f"error_horizon {horizon} is not one of the {short_term.shape[-1]} short-term horizons")
    return (short_term[:, horizon].float() - search_value.detach().float()) ** 2


def _masked_mean(per: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """The mean of ``per`` where ``keep``; 0 (still attached to the graph) where nothing is kept."""
    if bool(keep.any()):
        return per[keep].mean()
    return (per * 0.0).nan_to_num().sum()


def compute_losses(out: NetOutput, tgt: Targets, cfg: LossConfig | None = None) -> dict[str, torch.Tensor]:
    """Per-head losses and their weighted sum under ``"total"``."""
    cfg = LossConfig() if cfg is None else cfg
    losses: dict[str, torch.Tensor] = {}
    bsz = tgt.policy.shape[0]
    full = torch.ones(bsz, dtype=torch.bool) if tgt.full_search is None else tgt.full_search.bool()
    full = full.to(out.wdl_logits.device)
    submit = tgt.policy.new_zeros(bsz) if tgt.submit is None else tgt.submit.to(tgt.policy.dtype)
    action_target = torch.cat([tgt.policy, submit[:, None]], dim=-1)
    losses["policy"] = masked_ce(out.action_logits()[full], out.action_valid()[full], action_target[full])

    # D3 (section 8.3).
    losses["wdl"] = -(wdl_target(tgt, cfg.q_ratio) * torch.log_softmax(out.wdl_logits.float(), dim=-1)).sum(dim=-1).mean()
    n_bins = out.value_logits.shape[-1]
    root = tgt.root_value.float()
    has_root = full & torch.isfinite(root)
    sigma = cfg.hl_gauss_sigma * 2.0 / (n_bins - 1)
    target = hl_gauss(torch.where(has_root, root, torch.zeros_like(root)), n_bins, sigma)
    per = -(target * torch.log_softmax(out.value_logits.float(), dim=-1)).sum(dim=-1)
    losses["categorical"] = _masked_mean(per, has_root)
    st = tgt.short_term.float()
    has_st = torch.isfinite(st)
    losses["short_term"] = _masked_mean((out.short_term - torch.where(has_st, st, torch.zeros_like(st))) ** 2, has_st)
    err = error_target(st, out.search_value(), cfg.error_horizon)
    has_err = torch.isfinite(err)
    per = F.huber_loss(out.value_error, torch.where(has_err, err, torch.zeros_like(err)), delta=cfg.error_huber_delta, reduction="none")
    losses["value_error"] = _masked_mean(per, has_err)

    losses["moves_left"] = F.huber_loss(out.moves_left_log, torch.log1p(tgt.moves_left.float()), delta=cfg.huber_delta)
    losses["opp_reply"] = index_ce(out.opp_reply_logits, out.opp_move_valid, tgt.opp_reply)

    bce_b = F.binary_cross_entropy_with_logits(out.board_logits, tgt.board.float(), reduction="none").sum(-1)
    losses["board"] = _reduce_boards(bce_b, out.board_valid, cfg.board_reduction)
    bce_s = F.binary_cross_entropy_with_logits(out.square_logits, tgt.square.float(), reduction="none").mean(-1)
    per_board = _board_square_mean(bce_s, out.token_board, out.square_mask, out.board_valid.shape[1])
    losses["square"] = _reduce_boards(per_board, out.board_valid, cfg.board_reduction)

    keep = tgt.effort >= 0
    losses["effort"] = (
        F.cross_entropy(out.effort_logits[keep].float(), tgt.effort[keep]) if bool(keep.any()) else out.effort_logits.new_zeros(())
    )
    losses["total"] = sum(cfg.weights.get(name, 0.0) * value for name, value in losses.items())
    return losses
