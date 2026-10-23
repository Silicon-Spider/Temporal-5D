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
  - The error head: Huber loss against ``(predicted short-term value - short-term target)^2`` at
    ``error_horizon`` - the network's own short-term output taken with no gradient (``error_target``),
    so no gradient flows from the error target into the short-term or WDL heads. Section 8.3's
    "the squared gap between the value and the short-term value" reads two ways; this is the
    reading of register D9's "the predicted short-term value" and KataGo's form (its short-term
    error loss: the detached short-term prediction against the short-term target, Huber delta
    0.4). The other reading - the network's value W - L against the short-term target - is
    recorded for the user (spec issue 231). Which horizon and the Huber delta are placeholders.

* Policy: cross-entropy over each position's supplied candidate moves against a target
  distribution (the root's visits). Padding entries carry no mass. The policy's distribution is
  over the decision's actions - the candidate moves and, where it is offered and legal, the
  submit action (register D8, stage N5: ``NetOutput.action_logits``); its target is the root's
  visits to each move and to the submit child (``Targets.submit``) - of full searches only
  (section 12.5: "only full-search moves emit policy targets").
* **D2's flat form's policy** (section 8.2: its policy "normalises per board rather than across all
  moves"; spec issue 242): at a flat decision the target is each playable board's visit marginal over
  its options - its moves (the policy's logits), the non-branching jumps arriving on it, and leaving it
  (D2's leave head, ``NetOutput.leave_logits``) - ``Targets.flat_*``; the loss is each board's
  cross-entropy against the log-softmax over that board's options, summed over the boards (a
  placeholder, ``LossConfig.flat_board_reduction``, the mean the alternative), averaged over the
  positions, full searches only, at weight 1 (``weights["flat_policy"]``, the policy's - a
  placeholder). It fits each board's marginal - the factorised target, the product of the per-board
  distributions; it is not the cross-entropy of the engine's turn prior, which (under
  ``T5D_FLAT_PRIOR_GENERATED``, the default) normalises the boards' product over the generated turns
  only (the two differ by that normaliser's log, issue 242(e)). It is what trains the leave head; a
  batch with flat rows needs ``weights["flat_policy"]`` (refused without it, so a caller's weights
  cannot drop it silently). Under ``"sum"`` a flat row weighs about its number of playable boards
  times a ply-split row. A flat row carries no ply-split policy target, and the soft policy, D9's
  optimistic policy and the reply prototype skip it (issue 242).
* **D4's soft policy** (section 12.4, CHECKPOINT 7): its own head's actions
  (``NetOutput.soft_action_logits``) against the policy target "raised to the power 1/4 and
  renormalised" (``soft_policy_target``; the power ``LossConfig.soft_policy_power``), "at eight
  times the policy loss weight" (``weights["soft_policy"]`` 8, the policy's 1), full searches
  only - the order (the pruned target first, section 12.2, then the power) the builder's.
* **D4's opponent-reply prototype** (behind ``NetConfig.opp_reply``): cross-entropy against
  ``Targets.opp_reply``, the opponent's next turn as a distribution over its mapped moves
  (``train.reply_map``), weight 0.15 (KataGo's); a position with no target (an ambiguous or
  absent reply) is skipped.
* Effort: cross-entropy over its bins (censored bin included), full searches only (section 12.4:
  "full-search moves only"; a cheap search's row is skipped whatever its target). Moves-left:
  Huber loss on ``log1p(turns left)`` - a turn one player's action (D4, M20; spec issue 305).
* Per-board and per-square heads: binary cross-entropy with logits, masked to real
  boards and squares; a per-board loss sums its three targets, a per-square loss
  averages its labels and then the board's squares, so both are one number per board.
  Their reduction over a position's boards is register D6's open knob ("sum-vs-mean",
  ablation 6j): ``board_reduction="mean"`` (default) averages; ``"sum"`` sums them, so
  the auxiliary gradient grows with N. Positions are then averaged with equal weight.
* **Missing targets** (M21's record-derived targets, ``train.targets``): a NaN WDL row (an unscored
  game), a NaN moves-left (a censored game) and a NaN per-board or per-square label (tactical
  volatility until proofs exist; the per-square labels until their generator) are no target - each
  loss averages over the targets present.

* **D9's optimistic policy** (a prototype behind ``NetConfig.optimistic_policy``; section 8.1):
  its own actions against the normal policy target (full searches only), each sample weighted by
  ``optimistic_weights`` - ``clamp(0, 1, sigmoid((z - 1.5) x 3))`` with ``z = (short-term target -
  predicted short-term value) / sqrt(error + eps)``, the network's short-term output and error
  head taken without gradient - register D9's "how far the mover beat the predicted short-term
  value", in units of the predicted error, and KataGo's z_value (no score term in 5D) - the loss
  the weighted sum over the full-search positions divided by their number. The horizon is the
  error head's (``error_horizon``), so the error head predicts the numerator's square; that, eps
  (KataGo's 1e-4) and the normalisation are placeholders (spec issue 233).

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
    #: 1 draw, 2 loss) or ``(B, 3)`` distribution (I13's deblundered targets are) - a NaN row: no
    #: target (an unscored game; ``train.targets``).
    wdl: torch.Tensor
    #: D3's categorical target: ``(B,)`` the root search value of a full search, recomputed from
    #: the pruned visits (section 12.2), on [-1, 1]; NaN where there is none (the loss smooths it,
    #: ``heads.hl_gauss``).
    root_value: torch.Tensor
    #: D3's short-term targets: ``(B, K)`` on [-1, 1], one per ``NetConfig.short_term_horizons``;
    #: NaN where a horizon has none.
    short_term: torch.Tensor
    moves_left: torch.Tensor  # (B,) turns left (one player's action each, D4), >= 0; NaN: no target (a censored game)
    board: torch.Tensor  # (B, Nmax, 3) in {0, 1}; NaN: no label (tactical volatility until proofs exist)
    square: torch.Tensor  # (B, L, n_square_labels) in {0, 1}; NaN: no label (no generator yet)
    effort: torch.Tensor  # (B,) bin index, -1 where there is no target (skipped too where not ``full_search``)
    #: (B,) the policy target's mass on the submit action (D8; ``policy`` and ``submit``
    #: together are one distribution over the decision's actions); None: none anywhere.
    submit: torch.Tensor | None = None
    #: (B,) bool: the position's search was a full one (section 12.5, I5) - only those emit
    #: policy and categorical targets; None: every one was.
    full_search: torch.Tensor | None = None
    #: (B, 3): the search's WDL at the root - ``((1 - d + q) / 2, d, (1 - d - q) / 2)`` from the
    #: backed-up (q, d) (spec issue 148) - for ``LossConfig.q_ratio``'s blend; None: not given.
    root_wdl: torch.Tensor | None = None
    #: D4's reply prototype: ``(B, M2)``, the opponent's next turn as a distribution over the mapped
    #: moves (``PositionState.opp_moves``; ``train.reply_map``), all zero where there is no target;
    #: None: none anywhere.
    opp_reply: torch.Tensor | None = None
    #: D2's flat form (issue 242), None where no row is flat: ``(B,)`` bool - the row's decision was a
    #: flat one (its policy target per board; no ply-split policy target) - and its target's entries,
    #: ``(B, K)``: each entry's option (a candidate move ``i``, or ``M + b`` - ``M`` the batch's
    #: padded move count - for leaving board ``b``), its board (-1: padding) and its probability.
    flat_rows: torch.Tensor | None = None
    flat_option: torch.Tensor | None = None
    flat_group: torch.Tensor | None = None
    flat_target: torch.Tensor | None = None


@dataclass
class LossConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "policy": 1.0,
            "flat_policy": 1.0,  # D2's flat form's per-board policy (it trains the leave head): a placeholder
            "wdl": 1.0,
            "categorical": 1.0,
            "short_term": 1.0,
            "value_error": 1.0,
            "soft_policy": 8.0,  # section 12.4: "at eight times the policy loss weight"
            "opp_reply": 0.15,  # section 12.4: "weight 0.15, KataGo's"
            "optimistic": 1.0,  # D9 (a prototype): a placeholder
            "moves_left": 1.0,
            "board": 1.0,
            "square": 1.0,
            "effort": 1.0,
        }
    )
    board_reduction: str = "mean"  # D6: "mean" or "sum" over a position's boards
    #: D2's flat form: the per-board cross-entropies "sum"med (the factorised target's cross-entropy,
    #: not the engine's turn prior's) or averaged ("mean") over a position's boards - PLACEHOLDER
    #: (issue 242(e)).
    flat_board_reduction: str = "sum"
    huber_delta: float = 1.0  # moves-left's Huber loss
    #: D3: the WDL target's blend towards the search's own WDL at the root (Lc0's ``q_ratio``):
    #: "a tuned knob, default zero" (section 8.3, 6z) - 0 is exactly the game result.
    q_ratio: float = 0.0
    #: D3: HL-Gauss' sigma, in bin widths (Farebrother et al.: 0.75) - PLACEHOLDER (issue 231).
    hl_gauss_sigma: float = 0.75
    #: D3: the short-term horizon the error head's target reads (an index into
    #: ``NetConfig.short_term_horizons``; KataGo's uses its shortest) - PLACEHOLDER (issue 231).
    error_horizon: int = 0
    #: D3: the error head's Huber delta - PLACEHOLDER (issue 231; KataGo's is 0.4).
    error_huber_delta: float = 1.0
    #: D4: the soft policy's power - "the policy target raised to the power 1/4" (section 12.4).
    soft_policy_power: float = 0.25
    #: D9: each sample's weight ``clamp(0, 1, sigmoid((z - centre) x scale))`` (section 8.1: 1.5 and
    #: 3, KataGo's), and the eps under the z-score's square root (KataGo's 1e-4) - PLACEHOLDER
    #: (issue 233).
    optimistic_centre: float = 1.5
    optimistic_scale: float = 3.0
    optimistic_eps: float = 1e-4


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


def distribution_ce(logits: torch.Tensor, valid: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over variable-length candidate lists, averaged over the positions whose
    target has mass (a position with none - no target - is skipped)."""
    if logits.shape[-1] == 0:
        return logits.new_zeros(())
    logp = torch.log_softmax(logits.float().masked_fill(~valid, float("-inf")), dim=-1).masked_fill(~valid, 0.0)
    per = -(target.float() * logp).sum(dim=-1)
    keep = (target.float().sum(dim=-1) > 0) & valid.any(dim=-1)
    return _masked_mean(per, keep)


def flat_policy_ce(out: NetOutput, tgt: Targets, keep: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """D2's flat form's policy loss (module docstring): each board's cross-entropy over its options -
    the policy's move logits and the board's leave logit, gathered by ``tgt.flat_option`` and
    normalised per ``tgt.flat_group`` - against ``tgt.flat_target``, reduced over the boards
    (``reduction``), averaged over the rows ``keep``."""
    if reduction not in ("sum", "mean"):
        raise ValueError("flat_board_reduction must be 'sum' or 'mean'")
    if out.leave_logits is None:
        raise ValueError("a flat policy target needs the network's leave head (NetOutput.leave_logits)")
    logits = torch.cat([out.policy_logits.float(), out.leave_logits.float()], dim=-1)
    option, group, target = tgt.flat_option.to(logits.device), tgt.flat_group.to(logits.device), tgt.flat_target.to(logits.device)
    valid = group >= 0
    n_groups = out.leave_logits.shape[1] + 1  # a dump group for the padding
    z = torch.where(valid, logits.gather(1, option.clamp(min=0)), torch.zeros_like(target, dtype=logits.dtype))
    g = torch.where(valid, group, torch.full_like(group, n_groups - 1))
    top = z.new_zeros(z.shape[0], n_groups).scatter_reduce(1, g, z.detach(), "amax", include_self=False)
    shifted = z - top.gather(1, g)
    sums = z.new_zeros(z.shape[0], n_groups).scatter_add(1, g, torch.where(valid, shifted.exp(), torch.zeros_like(shifted)))
    logp = shifted - sums.clamp(min=1e-30).log().gather(1, g)
    per_entry = torch.where(valid, -target.float() * logp, torch.zeros_like(logp))
    per_row = per_entry.sum(dim=-1)
    if reduction == "mean":
        boards = z.new_zeros(z.shape[0], n_groups).scatter_add(1, g, valid.to(z.dtype))[:, :-1]
        per_row = per_row / (boards > 0).sum(dim=-1).clamp(min=1)
    keep = keep & valid.any(dim=-1)
    return _masked_mean(per_row, keep)


def soft_policy_target(target: torch.Tensor, power: float = 0.25) -> torch.Tensor:
    """D4's soft policy target: each row raised to ``power`` and renormalised (section 12.4:
    "the policy target raised to the power 1/4 and renormalised"); a row with no mass stays zero."""
    t = target.float().clamp(min=0.0) ** power
    total = t.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, t / total.clamp(min=1e-30), torch.zeros_like(t))


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


def error_target(predicted: torch.Tensor, short_term: torch.Tensor, horizon: int) -> torch.Tensor:
    """D3's error head target ``(B,)``: the squared gap between the network's short-term value
    output at ``horizon`` (``predicted``, ``(B, K)``, taken with no gradient) and the short-term
    value target there (``short_term``, ``(B, K)``); NaN where the horizon has no target.
    Register D9's "predicted short-term value" and KataGo's form (spec issue 231; the reading
    with the network's W - L in place of the short-term output is the alternative recorded)."""
    if not 0 <= horizon < short_term.shape[-1]:
        raise ValueError(f"error_horizon {horizon} is not one of the {short_term.shape[-1]} short-term horizons")
    return (predicted[:, horizon].detach().float() - short_term[:, horizon].float()) ** 2


def optimistic_weights(
    short_term: torch.Tensor,
    predicted: torch.Tensor,
    value_error: torch.Tensor,
    *,
    centre: float = 1.5,
    scale: float = 3.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """D9's per-sample weights ``(B,)``: ``clamp(0, 1, sigmoid((z - centre) x scale))``, ``z =
    (short_term - predicted) / sqrt(value_error + eps)`` - ``short_term`` the short-term value
    target, ``predicted`` the network's short-term output at the same horizon - how far the mover
    beat the predicted short-term value (register D9), in units of the error head's predicted
    error (section 8.1; KataGo's z_value, without the score term). The network's outputs are
    taken without gradient, so the weight trains nothing; NaN where the short-term target is
    missing."""
    z = (short_term.float() - predicted.detach().float()) / torch.sqrt(value_error.detach().float().clamp(min=0.0) + eps)
    return torch.sigmoid((z - centre) * scale).clamp(0.0, 1.0)


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
    flat = torch.zeros_like(full) if tgt.flat_rows is None else tgt.flat_rows.bool().to(full.device)
    ply = full & ~flat  # a flat row's policy target is per board (D2; issue 242)
    submit = tgt.policy.new_zeros(bsz) if tgt.submit is None else tgt.submit.to(tgt.policy.dtype)
    action_target = torch.cat([tgt.policy, submit[:, None]], dim=-1)
    losses["policy"] = masked_ce(out.action_logits()[ply], out.action_valid()[ply], action_target[ply])
    if tgt.flat_rows is not None and bool(flat.any()):
        losses["flat_policy"] = flat_policy_ce(out, tgt, full & flat, cfg.flat_board_reduction)

    # D3 (section 8.3).
    wt = wdl_target(tgt, cfg.q_ratio).to(out.wdl_logits.device)
    has_wdl = torch.isfinite(wt).all(dim=-1)  # a NaN row: no target (an unscored game)
    per = -(torch.where(has_wdl[:, None], wt, torch.zeros_like(wt)) * torch.log_softmax(out.wdl_logits.float(), dim=-1)).sum(dim=-1)
    losses["wdl"] = _masked_mean(per, has_wdl)
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
    err = error_target(out.short_term, st, cfg.error_horizon)
    has_err = torch.isfinite(err)
    per = F.huber_loss(out.value_error, torch.where(has_err, err, torch.zeros_like(err)), delta=cfg.error_huber_delta, reduction="none")
    losses["value_error"] = _masked_mean(per, has_err)

    # D4 (section 12.4): the soft policy (full searches only) and the reply prototype (behind its flag).
    if out.soft_policy_logits is not None:
        soft = soft_policy_target(action_target, cfg.soft_policy_power)
        losses["soft_policy"] = masked_ce(out.soft_action_logits()[ply], out.action_valid()[ply], soft[ply])
    if out.opp_reply_logits is not None and tgt.opp_reply is not None:
        losses["opp_reply"] = distribution_ce(out.opp_reply_logits, out.opp_move_valid, tgt.opp_reply)

    # D9 (section 8.1; a prototype behind its flag): the normal target, each sample weighted.
    if out.optimistic_logits is not None:
        h = cfg.error_horizon
        w = optimistic_weights(
            tgt.short_term[:, h].float(),
            out.short_term[:, h],
            out.value_error,
            centre=cfg.optimistic_centre,
            scale=cfg.optimistic_scale,
            eps=cfg.optimistic_eps,
        )
        logits = out.optimistic_action_logits().masked_fill(~out.action_valid(), float("-inf"))
        logp = torch.log_softmax(logits.float(), dim=-1).masked_fill(~out.action_valid(), 0.0)
        per = -(action_target * logp).sum(dim=-1)
        keep = ply & torch.isfinite(w) & out.action_valid().any(dim=-1)
        losses["optimistic"] = _masked_mean(torch.where(keep, w, torch.zeros_like(w)) * per, keep)

    ml = tgt.moves_left.float()
    has_ml = torch.isfinite(ml)  # NaN: no target (a censored game)
    log_ml = torch.log1p(torch.where(has_ml, ml, torch.zeros_like(ml)))
    per = F.huber_loss(out.moves_left_log, log_ml, delta=cfg.huber_delta, reduction="none")
    losses["moves_left"] = _masked_mean(per, has_ml)

    # Per-board and per-square labels: a NaN label is none (masked) - each board sums its labels
    # present, each square averages its labels present (a square with none is left out).
    bt = tgt.board.float()
    has_b = torch.isfinite(bt)
    bce_b = F.binary_cross_entropy_with_logits(out.board_logits, torch.where(has_b, bt, torch.zeros_like(bt)), reduction="none")
    bce_b = (bce_b * has_b).sum(-1)
    losses["board"] = _reduce_boards(bce_b, out.board_valid, cfg.board_reduction)
    sq = tgt.square.float()
    has_s = torch.isfinite(sq)
    bce_s = F.binary_cross_entropy_with_logits(out.square_logits, torch.where(has_s, sq, torch.zeros_like(sq)), reduction="none")
    bce_s = (bce_s * has_s).sum(-1) / has_s.sum(-1).clamp(min=1)
    per_board = _board_square_mean(bce_s, out.token_board, out.square_mask & has_s.any(-1), out.board_valid.shape[1])
    losses["square"] = _reduce_boards(per_board, out.board_valid, cfg.board_reduction)

    keep = (tgt.effort >= 0) & full.to(tgt.effort.device)  # section 12.4: "full-search moves only"
    losses["effort"] = (
        F.cross_entropy(out.effort_logits[keep].float(), tgt.effort[keep]) if bool(keep.any()) else out.effort_logits.new_zeros(())
    )
    if "flat_policy" in losses and "flat_policy" not in cfg.weights:
        raise ValueError("a batch with flat rows needs LossConfig.weights['flat_policy'] (D2's flat form; 0 to switch it off)")
    losses["total"] = sum(cfg.weights.get(name, 0.0) * value for name, value in losses.items())
    return losses
