"""Losses for the network's heads (spec sections 8.1, 8.3, 12.4).

* Value and short-horizon value: cross-entropy against a categorical target over the
  51 bins (a one-hot or two-hot distribution, ``heads.value_to_twohot``).
* Policy and opponent reply: cross-entropy over each position's supplied candidate
  moves against a target distribution (visit counts for the policy; a one-hot reply for
  the opponent-reply head). Padding entries carry no mass. The policy's distribution is
  over the decision's actions - the candidate moves and, where it is offered and legal,
  the submit action (register D8, stage N5: ``NetOutput.action_logits``); its target is
  the root's visits to each move and to the submit child (``Targets.submit``).
* WDL: 3-class cross-entropy. Effort: cross-entropy over its bins (censored bin
  included). Moves-left: Huber loss on ``log1p(turns left)`` - a turn one player's action
  (D4, M20; spec issue 305); building the target from a game record is the training loop's
  (M21).
* Per-board and per-square heads: binary cross-entropy with logits, masked to real
  boards and squares; a per-board loss sums its three targets, a per-square loss
  averages its labels and then the board's squares, so both are one number per board.
  Their reduction over a position's boards is register D6's open knob ("sum-vs-mean",
  ablation 6j): ``board_reduction="mean"`` (default) averages; ``"sum"`` sums them, so
  the auxiliary gradient grows with N. Positions are then averaged with equal weight.

The source rule (D5, section 12.4) is the caller's to keep: every target here must come
from the game record, the generator, scanner proofs or the engine's own search - never
from the learned verifier's estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..net.model import NetOutput


@dataclass
class Targets:
    value: torch.Tensor  # (B, value_bins) distribution
    policy: torch.Tensor  # (B, M) distribution over supplied moves
    wdl: torch.Tensor  # (B,) class index: 0 win, 1 draw, 2 loss (the acting player's, D8)
    moves_left: torch.Tensor  # (B,) turns left (one player's action each, D4), >= 0
    opp_reply: torch.Tensor  # (B,) index into opp_moves, -1 where there is no target
    short_value: torch.Tensor  # (B, value_bins) distribution
    board: torch.Tensor  # (B, Nmax, 3) in {0, 1}
    square: torch.Tensor  # (B, L, n_square_labels) in {0, 1}
    effort: torch.Tensor  # (B,) bin index, -1 where there is no target (PCR cheap searches)
    #: (B,) the policy target's mass on the submit action (D8; ``policy`` and ``submit``
    #: together are one distribution over the decision's actions); None: none anywhere.
    submit: torch.Tensor | None = None


@dataclass
class LossConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "value": 1.0,
            "policy": 1.0,
            "wdl": 1.0,
            "moves_left": 1.0,
            "opp_reply": 1.0,
            "short_value": 1.0,
            "board": 1.0,
            "square": 1.0,
            "effort": 1.0,
        }
    )
    board_reduction: str = "mean"  # D6: "mean" or "sum" over a position's boards
    huber_delta: float = 1.0


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


def compute_losses(out: NetOutput, tgt: Targets, cfg: LossConfig | None = None) -> dict[str, torch.Tensor]:
    """Per-head losses and their weighted sum under ``"total"``."""
    cfg = LossConfig() if cfg is None else cfg
    losses: dict[str, torch.Tensor] = {}
    losses["value"] = categorical_ce(out.value_logits, tgt.value)
    submit = tgt.policy.new_zeros(tgt.policy.shape[0]) if tgt.submit is None else tgt.submit.to(tgt.policy.dtype)
    losses["policy"] = masked_ce(out.action_logits(), out.action_valid(), torch.cat([tgt.policy, submit[:, None]], dim=-1))
    losses["wdl"] = F.cross_entropy(out.wdl_logits.float(), tgt.wdl)
    losses["moves_left"] = F.huber_loss(out.moves_left_log, torch.log1p(tgt.moves_left.float()), delta=cfg.huber_delta)
    losses["opp_reply"] = index_ce(out.opp_reply_logits, out.opp_move_valid, tgt.opp_reply)
    losses["short_value"] = categorical_ce(out.short_value_logits, tgt.short_value)

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
