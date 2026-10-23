"""Every head's training targets, re-derived from I15's records (register M21; spec sections 8.3,
12.2, 12.4, 12.5, 12.8).

The builder reads only a ``records.GameRecord`` - a live game's or one read back from disk - so
section 15.2's row "targets re-derived from stored records = the targets written at generation"
compares one function over two inputs. Every target is computed in float64 from the record's
stored values, deterministically, so the comparison is exact. Every unstated detail is a named
parameter of ``TargetConfig`` with a placeholder default (spec issue 235).

* **The policy target (I2, I5, section 12.5)**: the root's visits over the candidate moves and
  submit (D8), **full searches only**, with KataGo's forced-playout pruning applied here, at
  target construction - "adopt KataGo's forced-playout pruning with sqrt(playouts) decay at
  target construction, the enforcement point section 12.2 already mandates". The most visited
  child c* keeps its visits; every other child c loses up to ``n_forced(c) = sqrt(k P(c) sum N)``
  of them while its PUCT score, recomputed with the reduced count, stays below c*'s - a replica
  of the engine's root selection (``mcts.cpp`` ``select_child``: ``q + c P sqrt(N_parent) / (1 +
  n)``, ``c`` the root's effective cPUCT, E3's factor on the exploration term) - E2's root scale
  and E3's factor as recorded, their values at the search's end, which the selection did not
  necessarily see throughout: exact under the defaults, where both are off (issue 235(a)); a child
  left with one visit is pruned outright. **Steered visits** (E10's boosts and Hunt's suggestions, section
  12.2) are removed under ``policy_mode="exclusion"`` or kept at ``provenance_weight`` under
  ``"provenance"`` (which one is open - 6e; the counts are 0 until E10 wires them), and the
  PUCT replica compares priors without the recorded boost. Recorded forced visits (none: forced
  playouts are not built, issue 153) are pruned like any other.
* **The soft policy (D4)**: the policy target raised to 1/4 and renormalised (``losses``' helper
  computes it in the loss; the builder keeps it for the record comparison).
* **Search-value targets from the pruned visits (I2, CHECKPOINT 7)**: each root's value
  recomputed as the pruned-visit-weighted average of its children's (q, d). **D3's categorical**
  is over that root search value of full searches, HL-Gauss-smoothed in the loss; 6j's other arm
  (``categorical_source="short_term"``) trains it on a short-term value instead. The search's own
  WDL at the root (``root_wdl``, for the loss's ``q_ratio`` blend, default 0) is the pruned (q, d).
* **WDL on the game result (D3)**, the acting player's (D8): a checkmate's winner, a stalemate or
  I11's turn cap a draw, a resignation the other side's win; a game that ended unscored
  (``"unknown"``, the safety stop) has no WDL target. Then **deblundered (I13)**.
* **Short-term values (D3)** at ``NetConfig.short_term_horizons`` (in turns - one player's action,
  issue 305): KataGo's exponential average of the future search values, ``ST_h(t) = (1 - l)[v_t +
  sum_{a > a_t} l^(a - a_t) u_a] + l^(A - a_t) z``, ``l = 1 - 1/h``: ``v_t`` the decision's own
  pruned root value, ``u_a`` each later turn's first decision's, ``z`` the game's result, every
  term in the decision's acting player's view; NaN in an unscored game.
* **Moves-left (D4)**: turns left to the game's end from the decision's turn - the turn in
  progress counted (issue 224's placeholder, ``moves_left_counts_current``) - so a turn's
  decisions share it; censored games (``moves_left_censored``) have none. Deblundered (I13).
* **Deblundering (I13)**: walking back from the game's end, a decision whose played child is not
  the best child (by pruned visits - I2) and whose value ``q`` falls ``delta`` below the best's
  blends the outcome targets it and every earlier decision take towards the best child's WDL (its
  backed-up (q, d)) and moves-left (its network output plus the turn its move ends, BUILD_LOG
  issue 148), with weight ``clamp((delta - (threshold - band / 2)) / band, 0, 1)`` - Lc0's
  threshold 0.1, the band a placeholder. A move-child the search never evaluated (D2's tree form
  expands mid-turn children on inherited priors) takes the decision's own raw moves-left - the
  same action's state; with no estimate at all, moves-left is masked from that decision back
  rather than left unblended beside a blended WDL (issue 235(g)). Game-record facts (the
  per-board labels) stay as recorded.
* **Per-board targets (D4)**: travelled-into - a played move within ``travel_horizon`` turns from
  the decision (its own included) lands on the board by travel (its destination, for a branching
  move the history board it targets) - and timeline-active-at-end; tactical volatility needs proofs
  and is NaN (masked) until the proof channel is signed off.
* **Per-square labels (D4, M20, A7)**: ``heads.SQUARE_LABELS``, exact from the rules engine's
  generator (``Game.square_labels``; engine README, "Per-square labels") on each decision's
  replayed state - relative to the acting player, as the network reads them - carried by the
  positions (``selfplay.replay_positions(square_labels=True)``: ``Position.square_labels``),
  never stored in the record or the targets file: re-derived like the reply prototype's. A label
  a mate search's budget left undecided is NaN (no target); a decision whose position carries
  none - a synthetic record's, or targets built without positions - has every label NaN.
* **Effort (D4, section 8.6)**: full searches only - ``effort_bin(root_best_since, converged)``,
  converged when the most visited child led for at least ``effort_stable_fraction`` of the root's
  visits (issue 154), or when I20's KL rule stopped the search.
* **The reply prototype (D4)**: ``train.reply_map`` over the opponent's next turn, from the
  decisions' rebuilt inputs (``reply_targets``).
* **D9's weights** need the trained network's own short-term and error outputs and are computed in
  the loss (``losses.optimistic_weights``, as KataGo does at training time); the records supply its
  short-term target and the full-search mask.
* **Proof rescoring (I14)** is a hook, disabled: ``TargetConfig.proof_rescoring`` is refused while
  ``PROOF_CHANNEL_SIGNED_OFF`` is False - no proof may enter before the H23/H41 sign-off.
* **The source rule (D5)**: every target here is from the game record, the engine's own search or
  (later) proofs - ``TARGET_SOURCES``; nothing reads a learned verifier.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch

from ..engine.capi import T5D_MOVE_TRAVEL
from ..net.state import collate
from .losses import Targets
from .records import DecisionRecord, GameRecord
from .reply_map import AMBIGUOUS_MODES, ReplyMap, map_reply

#: Register H23/H41: no proof enters a store, or a target, before the gate passes and the proof
#: channel is signed off. I14's rescoring is refused while this is False.
PROOF_CHANNEL_SIGNED_OFF = False

#: Each target's source (section 12.4's source rule, register D5: never a learned verifier).
TARGET_SOURCES: dict[str, str] = {
    "policy": "search record (full searches; pruned visits)",
    "soft_policy": "search record (full searches; pruned visits)",
    "categorical": "search record (full searches; pruned visits)",
    "root_wdl": "search record (pruned visits)",
    "short_term": "search record and game record",
    "wdl": "game record (deblundered from the search record)",
    "moves_left": "game record (deblundered from the search record)",
    "board": "game record",
    "effort": "search record (full searches)",
    "opp_reply": "game record",
    "tactical_volatility": "proofs (gated: H23, H41)",
    "square": "game record (each decision's replayed state, labelled by the rules engine's exact generator)",
}

#: Game ends with a result (scored): the rest have no WDL or short-term target.
SCORED_ENDS: tuple[str, ...] = ("checkmate", "stalemate", "turn_cap", "resign")


@dataclass(frozen=True)
class TargetConfig:
    """The target builder's settings - each a PLACEHOLDER unless a document gives it (issue 235)."""

    # I2 (section 12.2, 12.5): exclusion or provenance (open - 6e); KataGo's forced-playout pruning.
    policy_mode: str = "exclusion"
    provenance_weight: float = 0.5
    forced_k: float = 2.0  # KataGo's k in n_forced = sqrt(k P sum N)
    prune_prior: str = "search"  # the prior the pruning's PUCT replica reads: "search" (noised, KataGo's) or "noise_free"
    prune_single_visit: bool = True  # KataGo: a child reduced to one visit is pruned outright
    # D3
    categorical_source: str = "root_value"  # 6j: "root_value" or "short_term"
    categorical_horizon: int = 0  # the short-term arm's horizon (an index into short_term_horizons)
    short_term_horizons: tuple[float, ...] = (2.0, 5.0, 16.0)  # NetConfig's, in turns
    # D4
    soft_policy_power: float = 0.25  # section 12.4
    moves_left_counts_current: bool = True  # issue 224: the turn in progress counts
    moves_left_censored: tuple[str, ...] = ("unknown", "action_limit", "turn_cap", "resign")
    travel_horizon: int = 2  # turns
    effort_stable_fraction: float = 0.5
    effort_bins: int = 16
    reply_ambiguous: str = "move"  # train.reply_map
    square_labels: bool = True  # D4's per-square labels re-derived at replay (the trainer's replay_positions)
    square_mate_nodes: int = 200_000  # each arrival_gives_mate search's node budget (the engine's placeholder)
    # I13 (section 12.8): Lc0's threshold; the band a placeholder; cheap searches' choices included.
    deblunder: bool = True
    deblunder_threshold: float = 0.1
    deblunder_band: float = 0.05
    deblunder_cheap: bool = True
    # I14: refused until the sign-off (PROOF_CHANNEL_SIGNED_OFF).
    proof_rescoring: bool = False

    def validate(self) -> None:
        if self.policy_mode not in ("exclusion", "provenance"):
            raise ValueError("policy_mode must be 'exclusion' or 'provenance' (I2, section 12.2)")
        if self.prune_prior not in ("search", "noise_free"):
            raise ValueError("prune_prior must be 'search' or 'noise_free'")
        if self.categorical_source not in ("root_value", "short_term"):
            raise ValueError("categorical_source must be 'root_value' or 'short_term' (6j)")
        if not 0 <= self.categorical_horizon < len(self.short_term_horizons):
            raise ValueError("categorical_horizon indexes short_term_horizons")
        if any(not (math.isfinite(h) and h >= 1.0) for h in self.short_term_horizons):
            raise ValueError("short-term horizons are at least one turn")
        if not 0.0 <= self.provenance_weight <= 1.0 or self.forced_k < 0 or self.soft_policy_power <= 0:
            raise ValueError("provenance_weight in [0, 1], forced_k >= 0, soft_policy_power > 0")
        if self.deblunder_band < 0 or self.travel_horizon < 1 or not 0.0 <= self.effort_stable_fraction <= 1.0:
            raise ValueError("deblunder_band >= 0, travel_horizon >= 1, effort_stable_fraction in [0, 1]")
        if self.square_mate_nodes < 1:
            raise ValueError("square_mate_nodes >= 1 (0 would leave every mate label undecided)")
        if self.reply_ambiguous not in AMBIGUOUS_MODES:
            raise ValueError(f"reply_ambiguous must be one of {AMBIGUOUS_MODES}")
        if self.proof_rescoring and not PROOF_CHANNEL_SIGNED_OFF:
            raise ValueError(
                "proof rescoring (I14) is gated on the proof-channel sign-off (H23, H41): no proof may enter a target before it"
            )

    def to_dict(self) -> dict:
        return asdict(self)


# --- I2: the pruned visits ---------------------------------------------------------------------------


def pruned_visits(d: DecisionRecord, cfg: TargetConfig) -> np.ndarray:
    """Each root child's visits for the targets (float64): steered visits excluded or weighted
    (I2), then KataGo's forced-playout pruning against the most visited child (module docstring)."""
    c = d.children
    n = c.visits.astype(np.float64)
    steered = np.minimum(c.steered_visits.astype(np.float64), n)
    n = n - (steered if cfg.policy_mode == "exclusion" else (1.0 - cfg.provenance_weight) * steered)
    if n.size == 0 or not (n > 0).any():
        return n
    best = int(np.argmax(n))
    prior = (c.prior if cfg.prune_prior == "search" else c.prior_noise_free).astype(np.float64) - c.boost.astype(np.float64)
    prior = np.maximum(prior, 0.0)
    sqrt_parent = math.sqrt(max(float(d.root_visits), 1.0))
    cpuct = float(d.root_cpuct)
    scale = c.exploration_scale.astype(np.float64)
    q = c.q.astype(np.float64)
    total = float(n.sum())

    def puct(i: int, visits: float) -> float:
        return float(q[i]) + cpuct * float(prior[i]) * sqrt_parent / (1.0 + visits) * float(scale[i])

    bar = puct(best, float(n[best]))
    out = n.copy()
    for i in range(n.size):
        if i == best or out[i] <= 0:
            continue
        allowed = math.floor(math.sqrt(cfg.forced_k * float(prior[i]) * total))
        v = out[i]
        floor_v = max(0.0, v - allowed)
        while v - 1.0 >= floor_v and puct(i, v - 1.0) < bar:
            v -= 1.0
        if cfg.prune_single_visit and v <= 1.0:
            v = 0.0
        out[i] = v
    return out


def policy_target(d: DecisionRecord, pruned: np.ndarray) -> np.ndarray | None:
    """The pruned visits as one distribution over the candidate moves and submit (``M + 1``,
    float64; D8); None where no visit survives."""
    total = float(pruned.sum())
    if total <= 0:
        return None
    out = np.zeros(d.move_count + 1, dtype=np.float64)
    for k, v in zip(d.children.candidate.tolist(), pruned.tolist(), strict=True):
        out[k if k >= 0 else d.move_count] += v / total
    return out


def soft_policy(target: np.ndarray, power: float) -> np.ndarray:
    """D4's soft policy: ``target ** power`` renormalised (zeros stay zero)."""
    t = np.where(target > 0, target, 0.0) ** power
    s = float(t.sum())
    return t / s if s > 0 else t


def pruned_value(d: DecisionRecord, pruned: np.ndarray) -> tuple[float, float]:
    """The root's (q, d) recomputed as the pruned-visit-weighted average of its children's (I2,
    CHECKPOINT 7), the acting player's view; the recorded root's where no visit survives."""
    total = float(pruned.sum())
    if total <= 0:
        return float(d.root_q), float(d.root_d)
    w = pruned / total
    return float((w * d.children.q).sum()), float((w * d.children.d).sum())


def wdl_of(q: float, dd: float) -> np.ndarray:
    """A (q, d) pair as a WDL ``((1 - d + q) / 2, d, (1 - d - q) / 2)`` (issue 148)."""
    return np.array([(1.0 - dd + q) / 2.0, dd, (1.0 - dd - q) / 2.0], dtype=np.float64)


def raw_prior(d: DecisionRecord, submit_share: float = 0.0) -> np.ndarray:
    """The network's raw prior over the candidates and submit (``M + 1``): the softmax of the
    raw logits at temperature 1, submit included where it is a legal candidate - with the search's
    placeholder share where the network gave no submit logit."""
    offered = bool(d.submit_possible and d.submit_legal)
    z = d.raw_logits.astype(np.float64)
    out = np.zeros(d.move_count + 1, dtype=np.float64)
    with_submit = offered and math.isfinite(float(d.raw_submit_logit))
    if with_submit:
        z = np.append(z, float(d.raw_submit_logit))
    if z.size:
        e = np.exp(z - z.max())
        p = e / e.sum()
        out[: p.size] = p
    if offered and not with_submit:
        share = submit_share if d.move_count else 1.0
        out[: d.move_count] *= 1.0 - share
        out[d.move_count] = share
    return out


def kl_divergence(target: np.ndarray, prior: np.ndarray) -> float:
    """``KL(target || prior)`` (I12's "KL(prior -> policy target)", read as KataGo's); inf where the
    prior gives no mass to a target entry."""
    t = target > 0
    if not t.any():
        return math.nan
    p = prior[t]
    if (p <= 0).any():
        return math.inf
    return float((target[t] * (np.log(target[t]) - np.log(p))).sum())


def effort_bin(visits: int, converged: bool, n_bins: int = 16) -> int:
    """``heads.effort_bin`` for one decision: ``floor(log2 visits)`` capped at ``n_bins - 2``
    when converged, the censored top bin ``n_bins - 1`` otherwise."""
    if not converged:
        return n_bins - 1
    return min(int(math.floor(math.log2(max(int(visits), 1)))), n_bins - 2)


# --- the targets of one game -----------------------------------------------------------------------------


@dataclass
class GameTargets:
    """One game's targets, decision by decision (float64; NaN or -1 where there is none)."""

    policy: list[np.ndarray | None]  # (M + 1,) per decision: moves then submit; None: no target (cheap, or no visit)
    soft_policy: list[np.ndarray | None]
    pruned: list[np.ndarray]  # each decision's pruned visits per root child
    full_search: np.ndarray  # (D,) bool
    search_value: np.ndarray  # (D,) the pruned root value of every search
    categorical: np.ndarray  # (D,) D3's categorical target value (full searches; NaN otherwise)
    root_wdl: np.ndarray  # (D, 3) the pruned (q, d) as a WDL
    wdl: np.ndarray  # (D, 3) the result, deblundered; NaN rows: none
    moves_left: np.ndarray  # (D,) turns left, deblundered; NaN: none
    short_term: np.ndarray  # (D, K)
    deblunder_weight: np.ndarray  # (D,) the blend each decision's own blunder applied (0: none)
    board: list[np.ndarray]  # (N, 3) per decision: travelled-into, timeline-active-at-end, tactical volatility (NaN)
    effort: np.ndarray  # (D,) int64; -1: none
    kl: np.ndarray  # (D,) KL(policy target || raw prior); NaN: no target
    reply: list[ReplyMap | None] = field(default_factory=list)  # empty: not computed
    #: (boards, cells, len(SQUARE_LABELS)) per decision, NaN undecided; empty: not computed (no
    #: position carried labels). Re-derived from the replayed states, never stored.
    square: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.full_search.shape[0])


def _abs_wdl(view: np.ndarray, acting: int) -> np.ndarray:
    """A WDL in the acting player's view as (white wins, draw, black wins) - and back (its own inverse)."""
    return view if acting == 0 else view[::-1].copy()


def build_targets(
    record: GameRecord,
    cfg: TargetConfig | None = None,
    *,
    positions: Sequence | None = None,
    proofs=None,
    reply: bool | None = None,
) -> GameTargets:
    """Every target of one game from its record (module docstring). ``positions``: each decision's
    network input (``selfplay.replay_positions``, or kept at generation) - given, the reply
    prototype's targets are computed too (unless ``reply`` is False), and the per-square labels the
    positions carry (``Position.square_labels``) are taken. ``proofs``: I14's source - refused (the
    proof channel is off)."""
    cfg = TargetConfig() if cfg is None else cfg
    cfg.validate()
    if proofs is not None:
        raise ValueError("no proof may enter a target before the proof-channel sign-off (I14; H23, H41)")
    ds = record.decisions
    n = len(ds)
    end, winner = record.end, record.winner
    scored = end in SCORED_ENDS
    k_h = len(cfg.short_term_horizons)
    search = record.header.get("search", {})
    share = float(search.get("full", {}).get("submit_share", 0.0)) if search else 0.0

    pruned = [pruned_visits(d, cfg) for d in ds]
    policy: list[np.ndarray | None] = []
    soft: list[np.ndarray | None] = []
    full = np.zeros(n, dtype=bool)
    value = np.zeros(n, dtype=np.float64)
    root_wdl = np.zeros((n, 3), dtype=np.float64)
    kl = np.full(n, math.nan, dtype=np.float64)
    effort = np.full(n, -1, dtype=np.int64)
    for i, d in enumerate(ds):
        q, dd = pruned_value(d, pruned[i])
        value[i] = q
        root_wdl[i] = wdl_of(q, dd)
        pt = policy_target(d, pruned[i]) if d.full_search else None
        policy.append(pt)
        soft.append(None if pt is None else soft_policy(pt, cfg.soft_policy_power))
        full[i] = pt is not None
        if pt is not None:
            kl[i] = kl_divergence(pt, raw_prior(d, share))
        if d.full_search:
            converged = d.stop == "kl" or (d.root_visits - d.root_best_since) >= cfg.effort_stable_fraction * d.root_visits
            effort[i] = effort_bin(d.root_best_since, converged, cfg.effort_bins)

    # The game's end, in turns: a resignation's turn is the turn it happened in.
    n_turns = record.n_actions + (1 if ds and ds[-1].played_child < 0 else 0)
    # The result, in absolute terms: (white wins, draw, black wins).
    if not scored:
        result = np.full(3, math.nan)
    elif end in ("stalemate", "turn_cap"):
        result = np.array([0.0, 1.0, 0.0])
    else:
        result = np.array([1.0, 0.0, 0.0]) if winner == 0 else np.array([0.0, 0.0, 1.0])
    z_abs = result[0] - result[2] if scored else math.nan

    # Short-term values (D3): each turn's first decision's value, then the exponential averages.
    first_of_turn: dict[int, int] = {}
    for i, d in enumerate(ds):
        first_of_turn.setdefault(d.action, i)
    short = np.full((n, k_h), math.nan, dtype=np.float64)
    if scored:
        for i, d in enumerate(ds):
            for j, h in enumerate(cfg.short_term_horizons):
                lam = 1.0 - 1.0 / h
                acc = value[i]
                for a in range(d.action + 1, n_turns):
                    if a not in first_of_turn:
                        continue
                    s = first_of_turn[a]
                    sign = 1.0 if ds[s].acting_player == d.acting_player else -1.0
                    acc += lam ** (a - d.action) * sign * value[s]
                z = z_abs if d.acting_player == 0 else -z_abs
                short[i, j] = (1.0 - lam) * acc + lam ** (n_turns - d.action) * z

    # Outcome targets, deblundered (I13), walking back from the end.
    censored = end in cfg.moves_left_censored or not scored
    lead = 0 if cfg.moves_left_counts_current else 1
    active = result.copy()
    active_end = math.nan if censored else float(n_turns)  # the turn the game ends at, in the walk's estimate
    wdl = np.full((n, 3), math.nan, dtype=np.float64)
    moves_left = np.full(n, math.nan, dtype=np.float64)
    blend = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        d = ds[i]
        w = _blunder_weight(d, pruned[i], cfg)
        if w > 0:
            b = int(np.argmax(pruned[i]))
            best_abs = _abs_wdl(wdl_of(float(d.children.q[b]), float(d.children.d[b])), d.acting_player)
            if np.isfinite(active).all():
                active = (1.0 - w) * active + w * best_abs
            elif w >= 1.0:  # an unscored game's outcome is the best child's where the blend is whole
                active = best_abs
            m_b = best_child_moves_left(d, b)
            if math.isfinite(m_b):
                offset = 1.0 if d.children.candidate[b] < 0 else 0.0  # a submit ends the turn in progress
                end_b = d.action + lead + m_b + offset
                if math.isfinite(active_end):
                    active_end = (1 - w) * active_end + w * end_b
                elif w >= 1.0:
                    active_end = end_b
            else:
                # No estimate of the best child's moves-left: the blend cannot be applied whole, so
                # moves-left is masked here and before (until a whole blend restores it) - never the
                # WDL blended while moves-left keeps the result's (issue 235(g)).
                active_end = math.nan
            blend[i] = w
        wdl[i] = _abs_wdl(active, d.acting_player)
        moves_left[i] = active_end - d.action - lead if math.isfinite(active_end) else math.nan

    # D3's categorical target (6j's switch), full searches only.
    if cfg.categorical_source == "root_value":
        categorical = np.where(full, value, math.nan)
    else:
        categorical = np.where(full, short[:, cfg.categorical_horizon], math.nan)

    # Per-board labels (D4): game-record facts.
    final_active = set(int(t) for t in record.header["result"].get("final_active_timelines", []))
    landings: list[tuple[int, int, int, int]] = []  # (decision, turn, to_timeline, to_half_turn) of every travel played
    for i, d in enumerate(ds):
        m = d.played_move()
        if m is not None and int(m["flags"]) & T5D_MOVE_TRAVEL:
            landings.append((i, d.action, int(m["to_timeline"]), int(m["to_half_turn"])))
    board = []
    for i, d in enumerate(ds):
        horizon = d.action + cfg.travel_horizon
        later = {(tl, ht) for s, a, tl, ht in landings if s >= i and a < horizon}
        lab = np.zeros((d.n_boards, 3), dtype=np.float64)
        for b, (tl, ht) in enumerate(d.boards.tolist()):
            lab[b, 0] = 1.0 if (tl, ht) in later else 0.0
            lab[b, 1] = 1.0 if tl in final_active else 0.0
        lab[:, 2] = math.nan  # tactical volatility: proofs (gated)
        board.append(lab)

    replies: list[ReplyMap | None] = []
    if positions is not None and reply is not False:
        replies = reply_targets(record, positions, cfg)
    square = square_targets(record, positions) if positions is not None else []
    return GameTargets(
        policy=policy,
        soft_policy=soft,
        pruned=pruned,
        full_search=full,
        search_value=value,
        categorical=categorical,
        root_wdl=root_wdl,
        wdl=wdl,
        moves_left=moves_left,
        short_term=short,
        deblunder_weight=blend,
        board=board,
        effort=effort,
        kl=kl,
        reply=replies,
        square=square,
    )


def best_child_moves_left(d: DecisionRecord, b: int) -> float:
    """I13's moves-left estimate for root child ``b`` (turns, the child's own count): its network
    output; for a move-child the search never evaluated - D2's tree form expands mid-turn children
    on inherited priors - the decision's own raw moves-left, the child being a state of the same
    action (issue 235(g)); NaN where there is none (an unevaluated submit child)."""
    m = float(d.children.network_moves_left[b])
    if not math.isfinite(m) and int(d.children.candidate[b]) >= 0:
        m = float(d.raw_moves_left)
    return m


def _blunder_weight(d: DecisionRecord, pruned: np.ndarray, cfg: TargetConfig) -> float:
    """I13's blend weight at one decision: 0 unless its played child is not the best (by pruned
    visits) and falls more than the threshold below it, linear across the band."""
    if not cfg.deblunder or d.played_child < 0 or (not d.full_search and not cfg.deblunder_cheap):
        return 0.0
    if pruned.size == 0 or not (pruned > 0).any():
        return 0.0
    b = int(np.argmax(pruned))
    p = d.played_child
    if p == b or d.children.visits[p] <= 0:
        return 0.0
    delta = float(d.children.q[b]) - float(d.children.q[p])
    if cfg.deblunder_band == 0.0:
        return 1.0 if delta > cfg.deblunder_threshold else 0.0
    return float(min(1.0, max(0.0, (delta - (cfg.deblunder_threshold - cfg.deblunder_band / 2.0)) / cfg.deblunder_band)))


def reply_targets(record: GameRecord, positions: Sequence, cfg: TargetConfig) -> list[ReplyMap | None]:
    """D4's reply prototype per decision: the opponent's next turn (its decisions and played moves)
    mapped onto the decision's input (``train.reply_map``); None where the game has no next turn."""
    ds = record.decisions
    if len(positions) != len(ds):
        raise ValueError(f"{len(positions)} positions for {len(ds)} decisions")
    by_turn: dict[int, list[int]] = {}
    for i, d in enumerate(ds):
        by_turn.setdefault(d.action, []).append(i)
    out: list[ReplyMap | None] = []
    for i, d in enumerate(ds):
        nxt = by_turn.get(d.action + 1)
        if not nxt:
            out.append(None)
            continue
        played = []
        for s in nxt:
            r = ds[s]
            k = -1 if r.played_child < 0 else int(r.children.candidate[r.played_child])
            played.append(k)
        state = _state(positions[i])
        out.append(map_reply(state, [_state(positions[s]) for s in nxt], played, ambiguous=cfg.reply_ambiguous))
    return out


def square_targets(record: GameRecord, positions: Sequence) -> list[np.ndarray]:
    """D4's per-square targets per decision, from the labels the positions carry
    (``Position.square_labels``); empty unless every position carries them."""
    if len(positions) != len(record.decisions):
        raise ValueError(f"{len(positions)} positions for {len(record.decisions)} decisions")
    labels = [getattr(p, "square_labels", None) for p in positions]
    if not labels or any(lab is None for lab in labels):
        return []
    out = []
    for i, (d, given) in enumerate(zip(record.decisions, labels, strict=True)):
        lab = np.asarray(given, dtype=np.float64)
        if lab.ndim != 3 or lab.shape[0] != d.n_boards:
            raise ValueError(f"decision {i}: per-square labels of shape {lab.shape} for {d.n_boards} boards")
        out.append(lab)
    return out


def _state(p):
    return getattr(p, "state", p)


# --- comparing and storing -----------------------------------------------------------------------------


def _flat(name: str, value) -> list[np.ndarray]:
    if isinstance(value, np.ndarray):
        return [value]
    if name == "reply":
        out = []
        for r in value:
            out += [np.zeros(0)] if r is None else [r.moves, r.target, np.asarray(r.ambiguous_boards), np.asarray([r.masked])]
        return out
    return [np.full(0, math.nan) if v is None else np.asarray(v) for v in value]


def target_differences(a: GameTargets, b: GameTargets) -> list[str]:
    """Where two games' targets differ, bit for bit (NaN equal to NaN; empty: the same)."""
    out = []
    for f in fields(GameTargets):
        xa, xb = getattr(a, f.name), getattr(b, f.name)
        if not isinstance(xa, np.ndarray) and len(xa) != len(xb):
            out.append(f"{f.name}: {len(xa)} entries against {len(xb)}")
            continue
        for i, (u, v) in enumerate(zip(_flat(f.name, xa), _flat(f.name, xb), strict=True)):
            if u.dtype != v.dtype or u.shape != v.shape or not np.array_equal(u, v, equal_nan=u.dtype.kind == "f"):
                out.append(f"{f.name}[{i}]")
    return out


def _ragged(rows: Sequence[np.ndarray | None], width: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = np.array([r is not None for r in rows], dtype=bool)
    parts = [np.zeros((0,) if width is None else (0, width)) if r is None else np.asarray(r, dtype=np.float64) for r in rows]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, p in enumerate(parts):
        offsets[i + 1] = offsets[i] + p.shape[0]
    data = np.concatenate(parts) if parts else np.zeros((0,) if width is None else (0, width))
    return data, offsets, present


def _unragged(data: np.ndarray, offsets: np.ndarray, present: np.ndarray) -> list[np.ndarray | None]:
    return [data[offsets[i] : offsets[i + 1]].copy() if present[i] else None for i in range(len(present))]


_COLUMNS = ("full_search", "search_value", "categorical", "root_wdl", "wdl", "moves_left", "short_term", "deblunder_weight", "effort", "kl")


def write_targets(path: str | Path, t: GameTargets) -> Path:
    """The targets written at generation (one ``.npz`` beside the record; the reply maps are not
    stored - they are rebuilt from the inputs)."""
    arrays: dict[str, np.ndarray] = {}
    for name in ("policy", "soft_policy", "pruned", "board"):
        width = 3 if name == "board" else None
        arrays[f"{name}_data"], arrays[f"{name}_offset"], arrays[f"{name}_present"] = _ragged(getattr(t, name), width)
    for name in _COLUMNS:
        arrays[name] = getattr(t, name)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    Path(path).write_bytes(buf.getvalue())
    return Path(path)


def read_targets(path: str | Path) -> GameTargets:
    with np.load(Path(path), allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    kw = {name: a[name] for name in _COLUMNS}
    for name in ("policy", "soft_policy", "pruned", "board"):
        kw[name] = _unragged(a[f"{name}_data"], a[f"{name}_offset"], a[f"{name}_present"])
    return GameTargets(**kw)


# --- a batch's targets ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One training row: a decision's network input and its game's targets."""

    state: object  # PositionState
    targets: GameTargets
    index: int  # the decision
    synthetic: bool = False  # from a synthetic record (``trainer.synthetic_record``): the only rows the untrained build steps on


def collate_rows(rows: Sequence[Row], net_cfg) -> tuple:
    """The network's batch and the losses' ``Targets`` for ``rows`` (``losses.Targets``; padding
    carries no target). With the reply prototype on (``NetConfig.opp_reply``) each row's input
    carries its mapped reply moves (``PositionState.opp_moves``) and its target; otherwise none."""
    if not rows:
        raise ValueError("no rows")
    states = []
    replies = []
    for r in rows:
        st = _state(r.state)
        rep = r.targets.reply[r.index] if (net_cfg.opp_reply and r.targets.reply) else None
        st = replace(st, opp_moves=rep.moves if rep is not None else np.zeros((0, 3), dtype=np.int64))
        states.append(st)
        replies.append(rep)
    batch = collate(states)
    bsz, m_max = batch.moves.shape[0], batch.moves.shape[1]
    k = len(rows[0].targets.short_term[0]) if len(rows[0].targets) else 0
    if tuple(net_cfg.short_term_horizons) and k != len(net_cfg.short_term_horizons):
        raise ValueError(f"the targets carry {k} short-term horizons, the network {len(net_cfg.short_term_horizons)}")
    n_max = batch.board_valid.shape[1]
    length = batch.piece.shape[1]
    policy = torch.zeros(bsz, m_max, dtype=torch.float32)
    submit = torch.zeros(bsz, dtype=torch.float32)
    board = torch.zeros(bsz, n_max, 3, dtype=torch.float32)
    opp = torch.zeros(bsz, batch.opp_move_valid.shape[1], dtype=torch.float32)
    for i, (r, st) in enumerate(zip(rows, states, strict=True)):
        t, j = r.targets, r.index
        pt = t.policy[j]
        m = int(st.moves.shape[0])
        if pt is not None:
            if pt.shape[0] != m + 1:
                raise ValueError(f"row {i}: a policy target over {pt.shape[0] - 1} moves for an input with {m}")
            policy[i, :m] = torch.from_numpy(pt[:m]).float()
            submit[i] = float(pt[m])
        lab = t.board[j]
        if lab.shape[0] != st.n_boards:
            raise ValueError(f"row {i}: board labels for {lab.shape[0]} boards, the input has {st.n_boards}")
        board[i, : lab.shape[0]] = torch.from_numpy(lab).float()
        if replies[i] is not None:
            opp[i, : replies[i].target.shape[0]] = torch.from_numpy(replies[i].target).float()

    # D4's per-square labels onto the square tokens (board-major: board b's cell c is token
    # b * (W*H + R) + c, the tokens' storage order - A7 moves none); NaN elsewhere and where a row
    # has none.
    square = torch.full((bsz, length, net_cfg.n_square_labels), math.nan)
    for i, (r, st) in enumerate(zip(rows, states, strict=True)):
        if not r.targets.square:
            continue
        lab = r.targets.square[r.index]
        cells = int(st.width) * int(st.height)
        if lab.shape != (st.n_boards, cells, net_cfg.n_square_labels):
            raise ValueError(f"row {i}: per-square labels of shape {lab.shape} for {st.n_boards} boards of {cells} cells")
        tpb = st.layout.tokens_per_board
        tokens = (np.arange(st.n_boards)[:, None] * tpb + np.arange(cells)[None, :]).reshape(-1)
        square[i, torch.from_numpy(tokens)] = torch.from_numpy(lab.reshape(-1, net_cfg.n_square_labels)).float()

    def col(name, dtype=torch.float32):
        return torch.from_numpy(np.stack([np.asarray(getattr(r.targets, name)[r.index]) for r in rows])).to(dtype)

    targets = Targets(
        policy=policy,
        wdl=col("wdl"),
        root_value=col("categorical"),
        short_term=col("short_term"),
        moves_left=col("moves_left"),
        board=board,
        square=square,
        effort=col("effort", torch.long),
        submit=submit,
        full_search=col("full_search", torch.bool),
        root_wdl=col("root_wdl"),
        opp_reply=opp if net_cfg.opp_reply else None,
    )
    return batch, targets


__all__ = [
    "PROOF_CHANNEL_SIGNED_OFF",
    "Row",
    "SCORED_ENDS",
    "TARGET_SOURCES",
    "GameTargets",
    "TargetConfig",
    "best_child_moves_left",
    "build_targets",
    "collate_rows",
    "effort_bin",
    "kl_divergence",
    "policy_target",
    "pruned_value",
    "pruned_visits",
    "raw_prior",
    "read_targets",
    "reply_targets",
    "soft_policy",
    "square_targets",
    "target_differences",
    "wdl_of",
    "write_targets",
]
