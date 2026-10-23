"""The target builder (register M21; spec sections 8.3, 12.2, 12.4, 12.5, 12.8): I2's pruned
visits and their exclusions, D3's and D4's targets, I13's deblundering, I14's refusal - on
constructed records with hand-computed answers - and section 15.2's I15 row on generated games:
"targets re-derived from stored records = the targets written at generation", under several
configurations. Nothing is trained."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine.capi import MOVE_DTYPE, T5D_MOVE_BRANCHING, T5D_MOVE_TRAVEL
from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model
from t5d.net.heads import effort_bin as torch_effort_bin
from t5d.train.losses import compute_losses
from t5d.train.records import Children, DecisionRecord, GameRecord, read_game, write_game
from t5d.train.selfplay import GeneratorConfig, PCRConfig, generate_game, replay_positions
from t5d.train.targets import (
    PROOF_CHANNEL_SIGNED_OFF,
    TARGET_SOURCES,
    Row,
    TargetConfig,
    best_child_moves_left,
    build_targets,
    collate_rows,
    effort_bin,
    kl_divergence,
    noise_keep_of,
    policy_target,
    pruned_value,
    pruned_visits,
    raw_prior,
    read_targets,
    replica_prior,
    soft_policy,
    target_differences,
    write_targets,
)
from tests.engine.helpers import requires_engine

# --- constructed records ------------------------------------------------------------------------------


def _children(
    visits, q, prior, *, candidate=None, d=None, steered=None, moves_left=None, boost=None, noise_free=None, scale=None, evaluated=None
):
    n = len(visits)
    return Children(
        candidate=list(range(n)) if candidate is None else candidate,
        legality=[0] * n,
        visits=visits,
        prior=prior,
        prior_noise_free=prior if noise_free is None else noise_free,
        q=q,
        d=[0.2] * n if d is None else d,
        value_variance=[0.0] * n,
        boost=[0.0] * n if boost is None else boost,
        exploration_scale=[1.0] * n if scale is None else scale,
        forced_visits=[0] * n,
        steered_visits=[0] * n if steered is None else steered,
        child_key=list(range(1, n + 1)),
        evaluated=[True] * n if evaluated is None else evaluated,
        network_wdl=np.tile([0.4, 0.2, 0.4], (n, 1)),
        network_moves_left=[10.0] * n if moves_left is None else moves_left,
        network_error=[math.nan] * n,
        lo=[-1] * n,
        hi=[1] * n,
    )


def _moves(n: int, travel: dict[int, tuple[int, int]] | None = None) -> np.ndarray:
    m = np.zeros(n, dtype=MOVE_DTYPE)
    m["from_square"] = np.arange(n) % 64
    m["to_square"] = (np.arange(n) + 8) % 64
    for i, (tl, ht) in (travel or {}).items():
        m[i]["flags"] = T5D_MOVE_TRAVEL | T5D_MOVE_BRANCHING
        m[i]["to_timeline"], m[i]["to_half_turn"] = tl, ht
    return m


def _decision(i, children, *, action=None, acting=None, played=0, full=True, n_moves=None, boards=((0, 2),), **kw):
    n_moves = len(children) if n_moves is None else n_moves
    action = i if action is None else action
    fields = dict(
        decision=i,
        action=action,
        move_in_action=0,
        acting_player=action % 2 if acting is None else acting,
        moves_made=0,
        n_boards=len(boards),
        present=2,
        submit_possible=False,
        submit_legal=False,
        full_search=full,
        playout_cap=100,
        playouts=int(children.visits.sum()),
        stop="playouts",
        move_temperature=1.0,
        root_policy_temperature=1.0,
        root_noise=True,
        search_seed=0,
        root_key=0,
        epoch=0,
        raw_error=math.nan,
        raw_moves_left=10.0,
        raw_submit_logit=math.nan,
        root_evaluated=True,
        root_visits=int(children.visits.sum()) + 1,
        root_q=0.0,
        root_d=0.0,
        root_cpuct=1.5,
        root_best_since=1,
        best_child=int(np.argmax(children.visits)),
        played_child=played,
        choice="sampled",
        verified_illegal=0,
        raw_wdl_logits=np.zeros(3),
        raw_value_logits=np.zeros(51),
        root_wdl=np.full(3, 1 / 3),
        boards=np.array(boards),
        candidates=_moves(n_moves, kw.pop("travel", None)),
        raw_logits=np.zeros(n_moves),
        children=children,
    )
    fields.update(kw)
    return DecisionRecord(**fields)


def _game(decisions, *, end="checkmate", winner=0, active=(0,)):
    actions = max(d.action for d in decisions) + 1 if decisions else 0
    header = {"result": {"end": end, "winner": winner, "actions": actions, "final_active_timelines": list(active)}, "search": {}}
    return GameRecord(header=header, decisions=decisions)


NO_DEBLUNDER = TargetConfig(deblunder=False)

# --- I2: pruning, exclusion and provenance -----------------------------------------------------------


def _kata_go(n, q, p, best, c, sqrt_parent, k, scale=None):
    """KataGo's pruned count for a non-best child, derived in closed form (not by the builder's
    loop): the least n' >= n - floor(sqrt(k P sum N)) with q + c P sqrt(N) s / (1 + n') below the
    best child's score (s E3's factor, 1 by default); one visit left is none."""
    s = [1.0] * len(n) if scale is None else scale
    bar = q[best] + c * p[best] * sqrt_parent / (1 + n[best]) * s[best]
    total = sum(n)
    out = []
    for i in range(len(n)):
        if i == best:
            out.append(float(n[i]))
            continue
        lowest = n[i] - math.floor(math.sqrt(k * p[i] * total))
        if bar > q[i]:
            x = c * p[i] * s[i] * sqrt_parent / (bar - q[i]) - 1.0
            least_below = math.floor(x) + 1  # the least integer n' with n' > x
        else:
            least_below = n[i]  # it never falls below the bar: nothing is pruned
        v = max(lowest, least_below, 0)
        v = min(v, n[i])
        out.append(0.0 if v <= 1 else float(v))
    return out


def test_pruning_removes_kataGos_amount_and_never_touches_the_best():
    n, q, p = [80, 15, 5, 2], [0.3, -0.2, -0.4, -0.3], [0.4, 0.3, 0.2, 0.1]
    d = _decision(0, _children(n, q, p))
    got = pruned_visits(d, TargetConfig())
    expected = _kata_go(n, q, p, 0, 1.5, math.sqrt(sum(n) + 1), 2.0)
    assert got.tolist() == expected
    assert got[0] == 80 and (got[1:] <= np.array(n[1:])).all() and got[1] < 15


def test_the_submit_child_is_pruned_like_a_move():
    n, q, p = [60, 20], [0.3, -0.1], [0.6, 0.4]
    as_move = pruned_visits(_decision(0, _children(n, q, p)), TargetConfig())
    sub = _decision(0, _children(n, q, p, candidate=[0, -1]), n_moves=1, submit_possible=True, submit_legal=True)
    assert pruned_visits(sub, TargetConfig()).tolist() == as_move.tolist()
    target = policy_target(sub, as_move)
    assert target.shape == (2,) and target[1] == as_move[1] / as_move.sum()


def test_a_one_visit_remainder_is_dropped_and_all_but_the_best_gives_a_one_hot():
    d = _decision(0, _children([50, 2, 1], [0.5, -0.5, -0.5], [0.9, 0.05, 0.05]))
    got = pruned_visits(d, TargetConfig())
    assert got.tolist() == [50.0, 0.0, 0.0]
    assert policy_target(d, got).tolist() == [1.0, 0.0, 0.0, 0.0]
    keep = pruned_visits(d, TargetConfig(prune_single_visit=False, forced_k=0.0))
    assert keep.tolist() == [50.0, 2.0, 1.0]  # k = 0: nothing forced, nothing pruned


def test_steered_visits_are_excluded_or_down_weighted():
    """I2: exclusion removes the steered visits; provenance keeps them at its weight (open - 6e)."""
    kw = dict(steered=[0, 10, 0], q=[0.0, 0.0, 0.0], prior=[0.34, 0.33, 0.33])
    d = _decision(0, _children([40, 30, 30], **kw))
    off = TargetConfig(forced_k=0.0)
    assert pruned_visits(d, off).tolist() == [40.0, 20.0, 30.0]
    assert pruned_visits(d, replace(off, policy_mode="provenance", provenance_weight=0.5)).tolist() == [40.0, 25.0, 30.0]
    assert pruned_visits(d, replace(off, policy_mode="provenance", provenance_weight=1.0)).tolist() == [40.0, 30.0, 30.0]


def test_the_boost_is_taken_out_of_the_replica():
    """A boosted prior's exploration is the steering's: the replica compares priors without it."""
    base = _children([60, 30], [0.2, 0.0], [0.5, 0.5])
    boosted = _children([60, 30], [0.2, 0.0], [0.5, 0.75], boost=[0.0, 0.25])
    cfg = TargetConfig()
    assert pruned_visits(_decision(0, boosted), cfg).tolist() == pruned_visits(_decision(0, base), cfg).tolist()


def test_the_replica_unsteers_the_noised_prior_by_the_noise_share():
    """The E10 review (A's L3): the search's prior after the root noise is (1 - eps) x the steered
    noise-free prior plus the noise's part, so the unsteered prior with the same noise is that less
    (1 - eps) x the boost - not the whole boost. The builder reads eps from the record."""
    eps = 0.25
    unsteered = np.array([0.4, 0.3, 0.2, 0.1])
    boost = np.array([-0.05, 0.15, -0.04, -0.06])  # signed: the unboosted drained
    nf = unsteered + boost
    mix = eps * np.array([0.1, 0.5, 0.3, 0.1])
    prior = (1 - eps) * nf + mix
    n, q = [60, 30, 20, 12], [0.30, 0.10, 0.05, 0.0]
    d = _decision(0, _children(n, q, prior.tolist(), noise_free=nf.tolist(), boost=boost.tolist()), root_visits=400)
    cfg = TargetConfig()
    np.testing.assert_allclose(replica_prior(d, cfg, noise_keep=1 - eps), (1 - eps) * unsteered + mix)
    np.testing.assert_allclose(replica_prior(d, replace(cfg, prune_prior="noise_free")), unsteered)
    no_noise = replace(d, root_noise=False)
    np.testing.assert_allclose(replica_prior(no_noise, cfg, noise_keep=1 - eps), prior - boost)  # no noise: the whole boost
    # The builder reads eps from the header (the full search's parameters) ...
    game = _game([d])
    game.header["search"] = {"full": {"dirichlet_epsilon": eps}}
    assert noise_keep_of(game.header, d) == 1 - eps
    assert noise_keep_of(game.header, no_noise) == 1.0
    built = build_targets(game, NO_DEBLUNDER)
    np.testing.assert_allclose(built.pruned[0], pruned_visits(d, NO_DEBLUNDER, noise_keep=1 - eps))
    # ... and refuses a noised, steered decision whose header names no eps.
    with pytest.raises(ValueError, match="dirichlet_epsilon"):
        noise_keep_of({"search": {}}, d)
    assert noise_keep_of({"search": {}}, replace(d, children=_children(n, q, prior.tolist()))) == 1.0


def test_the_replica_reads_its_recorded_inputs():
    """The PUCT replica's inputs, each pinned: E3's factor on the exploration term, the prior
    ``prune_prior`` names (the search's, after the noise, or the noise-free one) and the root's
    recorded visits - not the children's sum (they differ wherever children were removed or the
    root was visited before them) - each moving the answer, which is KataGo's closed form."""
    n, q = [60, 30, 20, 12], [0.30, 0.10, 0.05, 0.0]
    p_search, p_free, scale = [0.4, 0.3, 0.2, 0.1], [0.55, 0.15, 0.2, 0.1], [1.0, 0.5, 2.0, 1.5]
    d = _decision(0, _children(n, q, p_search, noise_free=p_free, scale=scale), root_visits=400)
    search = pruned_visits(d, TargetConfig(prune_prior="search")).tolist()
    free = pruned_visits(d, TargetConfig(prune_prior="noise_free")).tolist()
    assert search == _kata_go(n, q, p_search, 0, 1.5, math.sqrt(400), 2.0, scale) == [60.0, 22.0, 20.0, 9.0]
    assert free == _kata_go(n, q, p_free, 0, 1.5, math.sqrt(400), 2.0, scale) == [60.0, 24.0, 20.0, 8.0]
    unscaled = _decision(0, _children(n, q, p_search, noise_free=p_free), root_visits=400)
    assert pruned_visits(unscaled, TargetConfig()).tolist() == [60.0, 22.0, 14.0, 8.0]  # E3's factor read
    assert pruned_visits(replace(d, root_visits=sum(n)), TargetConfig()).tolist() == [60.0, 22.0, 18.0, 8.0]  # the root's visits read


def test_the_value_targets_read_the_pruned_visits():
    """I2 (CHECKPOINT 7): where pruning removes visits, the search value, D3's categorical, the
    q_ratio blend's root WDL and the short-term values are the pruned visits' average - by hand
    from KataGo's closed form - and not the raw visits'."""
    n, q, p, dd = [80, 15, 5, 2], [0.3, -0.2, -0.4, -0.3], [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]
    d = _decision(0, _children(n, q, p, d=dd))
    cfg = replace(NO_DEBLUNDER, short_term_horizons=(2.0,))
    t = build_targets(_game([d], end="checkmate", winner=0), cfg)
    kept = np.array(_kata_go(n, q, p, 0, 1.5, math.sqrt(sum(n) + 1), 2.0))
    assert kept.sum() < sum(n)  # pruning removed visits
    v, dv = float((kept * q).sum() / kept.sum()), float((kept * dd).sum() / kept.sum())
    raw = float(np.dot(n, q) / sum(n))
    assert abs(v - raw) > 0.01
    assert t.search_value[0] == pruned_value(d, pruned_visits(d, cfg))[0]
    assert t.search_value[0] == pytest.approx(v, abs=1e-12) and t.categorical[0] == pytest.approx(v, abs=1e-12)
    assert t.root_wdl[0] == pytest.approx([(1 - dv + v) / 2, dv, (1 - dv - v) / 2], abs=1e-12)
    assert t.short_term[0, 0] == pytest.approx(0.5 * v + 0.5 * 1.0, abs=1e-12)  # l = 1/2, one turn to White's win


def test_the_pruned_root_value_is_the_pruned_visits_average():
    c = _children([60, 30, 10], [0.5, 0.1, -0.4], [0.5, 0.3, 0.2], d=[0.1, 0.3, 0.5])
    d = _decision(0, c)
    pr = np.array([60.0, 20.0, 0.0])
    q, dd = pruned_value(d, pr)
    assert q == pytest.approx((60 * 0.5 + 20 * 0.1) / 80) and dd == pytest.approx((60 * 0.1 + 20 * 0.3) / 80)
    assert pruned_value(d, np.zeros(3)) == (d.root_q, d.root_d)


def test_soft_policy_kl_and_the_raw_prior():
    t = np.array([0.64, 0.36, 0.0])
    s = soft_policy(t, 0.25)
    assert s[2] == 0 and s.sum() == pytest.approx(1.0) and s[0] / s[1] == pytest.approx((0.64 / 0.36) ** 0.25)
    d = _decision(0, _children([1, 1], [0, 0], [0.5, 0.5]), raw_logits=np.log([0.5, 0.25, 0.25]), n_moves=3)
    p = raw_prior(d)
    assert np.allclose(p, [0.5, 0.25, 0.25, 0.0])
    assert kl_divergence(np.array([1.0, 0, 0, 0]), p) == pytest.approx(math.log(2))
    sub = replace(d, submit_possible=True, submit_legal=True, raw_submit_logit=math.log(0.5))
    assert np.allclose(raw_prior(sub), [1 / 3, 1 / 6, 1 / 6, 1 / 3])
    assert np.allclose(raw_prior(replace(sub, raw_submit_logit=math.nan), submit_share=0.1), [0.45, 0.225, 0.225, 0.1])


def _kl_by_hand(t, p):
    return sum(a * math.log(a / b) for a, b in zip(t, p, strict=True) if a > 0)


def test_the_builders_kl_is_the_target_against_the_raw_prior():
    """I12/I15's divergence as the builder records it: KL(policy target || raw prior), KataGo's
    direction - by hand on an asymmetric case where the reverse differs - and, where the network
    gave no submit logit, the raw prior carrying the search's submit share from the header."""
    cfg = TargetConfig(deblunder=False, forced_k=0.0)  # nothing pruned: the target is the visits
    d = _decision(0, _children([60, 30, 10], [0.0, 0.0, 0.0], [0.4, 0.3, 0.3]), raw_logits=np.log([0.2, 0.3, 0.5]))
    t = build_targets(_game([d]), cfg)
    target, prior = [0.6, 0.3, 0.1], [0.2, 0.3, 0.5]
    assert t.kl[0] == pytest.approx(_kl_by_hand(target, prior), rel=1e-6)  # the logits stored as float32
    assert abs(_kl_by_hand(target, prior) - _kl_by_hand(prior, target)) > 0.05
    assert t.kl[0] == kl_divergence(t.policy[0], raw_prior(d))
    sub = _decision(
        0,
        _children([50, 30, 20], [0.0, 0.0, 0.0], [0.4, 0.3, 0.3], candidate=[0, 1, -1]),
        n_moves=2,
        raw_logits=np.log([0.25, 0.75]),
        submit_possible=True,
        submit_legal=True,
    )
    g = _game([sub])
    g.header["search"] = {"full": {"submit_share": 0.1}}
    t = build_targets(g, cfg)
    assert t.kl[0] == pytest.approx(_kl_by_hand([0.5, 0.3, 0.2], [0.25 * 0.9, 0.75 * 0.9, 0.1]), rel=1e-6)


def test_effort_bins_match_the_heads_helper():
    for visits in (1, 2, 3, 7, 8, 1000, 1 << 20):
        for converged in (True, False):
            want = int(torch_effort_bin(torch.tensor([visits]), torch.tensor([converged]), 16)[0])
            assert effort_bin(visits, converged, 16) == want


# --- outcome targets: WDL, moves-left, short-term, deblundering ------------------------------------------


def _line(n_turns, *, values=None, played=None, q_played=None, **game):
    """A game of one decision per turn, alternating players; each root has two children - the best
    (index 0, 50 visits, q = values[t]) and another (index 1, 10 visits, q = q_played[t]) - the
    choice ``played[t]``."""
    values = [0.0] * n_turns if values is None else values
    played = [0] * n_turns if played is None else played
    q_played = [v - 0.05 for v in values] if q_played is None else q_played
    ds = []
    for t in range(n_turns):
        c = _children([50, 10], [values[t], q_played[t]], [0.5, 0.5], d=[0.2, 0.3], moves_left=[float(n_turns - t + 5), 3.0])
        ds.append(_decision(t, c, played=played[t]))
    return _game(ds, **game)


def test_wdl_and_moves_left_follow_the_result_from_each_acting_player():
    g = _line(4, end="checkmate", winner=1)
    t = build_targets(g, NO_DEBLUNDER)
    assert t.wdl.tolist() == [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]]  # White acts at 0 and 2
    assert t.moves_left.tolist() == [4, 3, 2, 1]  # the turn in progress counted (issue 224)
    t2 = build_targets(g, replace(NO_DEBLUNDER, moves_left_counts_current=False))
    assert t2.moves_left.tolist() == [3, 2, 1, 0]
    draw = build_targets(_line(2, end="stalemate", winner=None), NO_DEBLUNDER)
    assert draw.wdl.tolist() == [[0, 1, 0], [0, 1, 0]]


def test_unscored_and_censored_games():
    for end in ("unknown", "action_limit"):
        t = build_targets(_line(3, end=end, winner=None), NO_DEBLUNDER)
        assert np.isnan(t.wdl).all() and np.isnan(t.moves_left).all() and np.isnan(t.short_term).all()
    capped = build_targets(_line(3, end="turn_cap", winner=None), NO_DEBLUNDER)
    assert capped.wdl.tolist() == [[0, 1, 0]] * 3 and np.isnan(capped.moves_left).all()  # I11: a draw; moves-left censored


def test_short_term_targets_by_hand():
    values = [0.2, -0.1, 0.4]
    g = _line(3, values=values, end="checkmate", winner=0)
    t = build_targets(g, replace(NO_DEBLUNDER, short_term_horizons=(2.0, 5.0)))
    v = t.search_value  # each root's pruned value (tested above), the acting player's view
    for j, h in enumerate((2.0, 5.0)):
        lam = 1 - 1 / h
        # decision 0 (White): own value, then turns 1 (Black, flipped) and 2 (White); the result +1
        st0 = (1 - lam) * (v[0] - lam * v[1] + lam**2 * v[2]) + lam**3 * 1.0
        st1 = (1 - lam) * (v[1] - lam * v[2]) + lam**2 * -1.0  # Black's view
        st2 = (1 - lam) * v[2] + lam * 1.0
        assert t.short_term[:, j] == pytest.approx([st0, st1, st2])
    assert t.categorical.tolist() == v.tolist() and v.tolist() != values  # the pruned values, not the root's q
    arm = build_targets(g, replace(NO_DEBLUNDER, short_term_horizons=(2.0, 5.0), categorical_source="short_term", categorical_horizon=1))
    assert arm.categorical.tolist() == pytest.approx(t.short_term[:, 1].tolist())


def test_deblundering_rewrites_the_blunder_and_earlier_positions_only():
    # Turn 2's player (White) plays child 1 at q 0.3 below the best's: a blunder; the game is lost by White.
    values = [0.0, 0.0, 0.5, 0.0]
    q_played = [-0.05, -0.05, 0.2, -0.05]
    g = _line(4, values=values, played=[0, 0, 1, 0], q_played=q_played, end="checkmate", winner=1)
    t = build_targets(g, TargetConfig())
    best = np.array([(1 - 0.2 + 0.5) / 2, 0.2, (1 - 0.2 - 0.5) / 2])  # the best child's (q, d) in White's view
    assert t.deblunder_weight.tolist() == [0, 0, 1, 0]
    assert t.wdl[3].tolist() == [1, 0, 0]  # after the blunder: the result (Black to act, Black won)
    assert t.wdl[2] == pytest.approx(best) and t.wdl[0] == pytest.approx(best)
    assert t.wdl[1] == pytest.approx(best[::-1])  # Black's view
    # Moves-left from the best child: its network output (4 - 2 + 5 = 7 turns left, its move staying in turn 2), counted
    # back to the earlier turns; after the blunder the game's own count.
    assert t.moves_left.tolist() == pytest.approx([7 + 2, 7 + 1, 7, 1])
    assert t.board[0][:, 0].tolist() == build_targets(g, NO_DEBLUNDER).board[0][:, 0].tolist()  # game-record facts unchanged


def test_deblundering_interpolates_across_the_band():
    cfg = TargetConfig(deblunder_threshold=0.1, deblunder_band=0.1)
    for delta, w in ((0.04, 0.0), (0.05, 0.0), (0.075, 0.25), (0.1, 0.5), (0.15, 1.0), (0.3, 1.0)):
        g = _line(2, values=[0.5, 0.0], played=[1, 0], q_played=[0.5 - delta, -0.05], end="checkmate", winner=0)
        t = build_targets(g, cfg)
        assert t.deblunder_weight[0] == pytest.approx(w, abs=1e-12)
        best = np.array([(1 - 0.2 + 0.5) / 2, 0.2, (1 - 0.2 - 0.5) / 2])
        assert t.wdl[0] == pytest.approx((1 - w) * np.array([1.0, 0, 0]) + w * best)
    zero = TargetConfig(deblunder_band=0.0)
    g = _line(2, values=[0.5, 0.0], played=[1, 0], q_played=[0.35, -0.05])
    assert build_targets(g, zero).deblunder_weight[0] == 1.0


def test_short_term_targets_of_a_resigned_game_by_hand():
    """A resignation's turn is the game's last (the record's action count leaves it out): its
    short-term values by hand, the resigner's last decision within [-1, 1]."""
    values = [0.2, -0.1, 0.4]
    g = _line(3, values=values, played=[0, 0, -1], end="resign", winner=1)  # White resigns at turn 2
    g.header["result"]["actions"] = 2  # as the generator writes it: the resigned turn is not an action played
    t = build_targets(g, replace(NO_DEBLUNDER, short_term_horizons=(2.0, 5.0)))
    v = t.search_value
    for j, h in enumerate((2.0, 5.0)):
        lam = 1 - 1 / h
        st0 = (1 - lam) * (v[0] - lam * v[1] + lam**2 * v[2]) + lam**3 * -1.0  # White lost
        st1 = (1 - lam) * (v[1] - lam * v[2]) + lam**2 * 1.0  # Black won
        st2 = (1 - lam) * v[2] + lam * -1.0
        assert t.short_term[:, j] == pytest.approx([st0, st1, st2], abs=1e-12)
    assert (np.abs(t.short_term) <= 1.0).all()
    assert t.wdl[2].tolist() == [0.0, 0.0, 1.0] and np.isnan(t.moves_left).all()


def test_a_submit_best_child_ends_its_turn():
    """I13's moves-left from a best child that is the submit: its network output plus the turn its
    submission ends (issue 148's offset); a move-child's stays in the turn."""

    def game(candidate):
        c = _children([50, 10], [0.5, 0.2], [0.5, 0.5], candidate=candidate, moves_left=[4.0, 3.0])
        d0 = _decision(0, c, played=1, n_moves=1, submit_possible=True, submit_legal=True)
        d1 = _decision(1, _children([10, 5], [0.0, -0.05], [0.5, 0.5]))
        return build_targets(_game([d0, d1], end="checkmate", winner=1), TargetConfig())

    sub, move = game([-1, 0]), game([0, -1])
    assert sub.deblunder_weight[0] == move.deblunder_weight[0] == 1.0
    assert sub.moves_left[0] == 4.0 + 1.0 and move.moves_left[0] == 4.0


def test_an_unevaluated_best_move_child_blends_moves_left_with_the_wdl():
    """D2's tree form expands mid-turn children on inherited priors, so a best move-child can carry
    no network moves-left: I13 then reads the decision's own raw moves-left (the same action's
    state) and blends it with the WDL - never the WDL alone; an unevaluated submit child gives no
    estimate, and moves-left is masked from that decision back (issue 235(g))."""
    best = np.array([(1 - 0.2 + 0.5) / 2, 0.2, (1 - 0.2 - 0.5) / 2])

    def game(candidate):
        c = _children([50, 10], [0.5, 0.2], [0.5, 0.5], candidate=candidate, moves_left=[math.nan, 3.0], evaluated=[False, True])
        d0 = _decision(0, c, played=1, n_moves=1, submit_possible=True, submit_legal=True, raw_moves_left=6.0)
        d1 = _decision(1, _children([10, 5], [0.0, -0.05], [0.5, 0.5]))
        return build_targets(_game([d0, d1], end="checkmate", winner=1), TargetConfig())

    move = game([0, -1])
    assert move.deblunder_weight[0] == 1.0 and move.wdl[0] == pytest.approx(best)
    assert move.moves_left.tolist() == [6.0, 1.0]  # the root's own 6 turns from turn 0; after the blunder the game's count
    sub = game([-1, 0])
    assert sub.deblunder_weight[0] == 1.0 and sub.wdl[0] == pytest.approx(best)
    assert math.isnan(sub.moves_left[0]) and sub.moves_left[1] == 1.0


# --- per-board labels -------------------------------------------------------------------------------------


def test_travelled_into_and_timeline_survival_by_brute_force():
    boards = ((0, 2), (0, 3), (-1, 3), (1, 4))
    ds = []
    lands = {1: (-1, 3), 3: (1, 4)}  # decision -> where its played move lands by travel
    for t in range(5):
        travel = {0: lands[t]} if t in lands else None
        c = _children([10, 5], [0.0, -0.1], [0.5, 0.5])
        ds.append(_decision(t, c, boards=boards, travel=travel))
    g = _game(ds, end="checkmate", winner=0, active=(0, 1))
    cfg = replace(NO_DEBLUNDER, travel_horizon=2)
    t = build_targets(g, cfg)
    for i, d in enumerate(ds):
        for b, key in enumerate(boards):
            want = any(s >= i and s < d.action + 2 and lands.get(s) == key for s in range(len(ds)))
            assert t.board[i][b, 0] == float(want), (i, b)
            assert t.board[i][b, 1] == float(key[0] in (0, 1))
        assert np.isnan(t.board[i][:, 2]).all()  # tactical volatility: proofs, gated


def test_effort_from_the_change_point():
    c = _children([90, 10], [0.0, 0.0], [0.5, 0.5])
    early = _decision(0, c, root_visits=100, root_best_since=10)
    late = _decision(1, c, root_visits=100, root_best_since=80)
    kl = _decision(2, c, root_visits=100, root_best_since=80, stop="kl")
    cheap = _decision(3, c, full_search=False)
    t = build_targets(_game([early, late, kl, cheap]), NO_DEBLUNDER)
    assert t.effort.tolist() == [3, 15, 6, -1]  # log2(10) -> 3; not stable -> censored; I20's stop converged; cheap none
    assert t.full_search.tolist() == [True, True, True, False] and t.policy[3] is None and np.isnan(t.categorical[3])


def test_proof_rescoring_is_refused_until_the_sign_off():
    assert PROOF_CHANNEL_SIGNED_OFF is False
    with pytest.raises(ValueError, match="sign-off"):
        TargetConfig(proof_rescoring=True).validate()
    with pytest.raises(ValueError, match="sign-off"):
        build_targets(_line(2), proofs={})


def test_no_target_reads_a_learned_verifier():
    """D5 (section 12.4's source rule): every target's source is the game record, the engine's own
    search or proofs - never a learned verifier's estimate."""
    for source in TARGET_SOURCES.values():
        assert any(s in source for s in ("game record", "search record", "proofs")) and "verifier" not in source


def test_targets_round_trip_through_a_file(tmp_path):
    g = _line(5, values=[0.1, 0.2, -0.3, 0.0, 0.4], played=[0, 1, 0, 1, 0], end="checkmate", winner=0)
    t = build_targets(g)
    back = read_targets(write_targets(tmp_path / "t.npz", t))
    assert target_differences(t, back) == []
    t2 = build_targets(g, replace(TargetConfig(), deblunder_threshold=0.0))
    assert target_differences(t, t2) != []


# --- section 15.2's I15 row on generated games --------------------------------------------------------------

NET = NetConfig(**SMALL)

GAMES = {
    "ply": GeneratorConfig(playouts=8, batch_size=4, cap_turns=3, cap_boards=20, seed=11),
    "tree": GeneratorConfig(split="tree", playouts=8, batch_size=4, cap_turns=3, cap_boards=20, seed=12),
    "two_timelines": GeneratorConfig(variant="Standard - Two Timelines", playouts=6, batch_size=3, cap_turns=2, cap_boards=20, seed=13),
}
TARGET_CONFIGS = {
    "default": TargetConfig(short_term_horizons=NET.short_term_horizons),
    "provenance_short_term_arm": TargetConfig(
        short_term_horizons=NET.short_term_horizons, policy_mode="provenance", categorical_source="short_term", deblunder_band=0.2
    ),
    "no_prune_turns_after": TargetConfig(
        short_term_horizons=NET.short_term_horizons, forced_k=0.0, moves_left_counts_current=False, deblunder_threshold=0.0
    ),
}


@pytest.fixture(scope="module")
def evaluator():
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


@requires_engine
@pytest.mark.parametrize("game", sorted(GAMES))
@pytest.mark.parametrize("tc", sorted(TARGET_CONFIGS))
def test_targets_rederived_from_stored_records_equal_those_written_at_generation(evaluator, game, tc, tmp_path):
    """Section 15.2: "Targets re-derived from stored records = the targets written at generation |
    The records carry what later choices need | I15" - every target array equal bit for bit, the
    reply prototype's included (its inputs rebuilt from the record)."""
    cfg = TARGET_CONFIGS[tc]
    with torch.no_grad():
        g = generate_game(evaluator, GAMES[game], net_cfg=NET, keep_positions=True, target_config=cfg)
    written = read_targets(write_targets(tmp_path / "targets.npz", g.targets))
    record = read_game(write_game(tmp_path / "record.npz", g.record))
    assert record.header["target_config"]["policy_mode"] == cfg.policy_mode
    rederived = build_targets(record, cfg, positions=replay_positions(record))
    assert target_differences(written, replace(rederived, reply=[])) == []  # the file keeps no reply maps
    assert target_differences(g.targets, rederived) == []  # the reply maps too (not stored: rebuilt)
    assert [d.prior_to_target_kl for d in record.decisions] == pytest.approx(rederived.kl.tolist(), nan_ok=True)
    assert all(r is None or r.target.sum() in (0.0, pytest.approx(1.0)) for r in rederived.reply)


@requires_engine
def test_tree_form_deblundering_blends_moves_left_wherever_it_blends_the_wdl(evaluator):
    """I13 on D2's tree form, where a visited move-child is often never evaluated (inherited
    priors): every such child has a moves-left estimate (the decision's own), and with every
    non-best choice a whole blend, each blunder's two outcome targets are both its best child's -
    the WDL and the moves-left, never the WDL alone (issue 235(g); the review's probe found the
    WDL rewritten on 29 decisions and moves-left on 16)."""
    cfg = GeneratorConfig(split="tree", playouts=24, batch_size=4, cap_turns=4, cap_boards=30, seed=7, pcr=PCRConfig(enabled=False))
    with torch.no_grad():
        g = generate_game(evaluator, cfg, net_cfg=NET)
    ds = g.record.decisions
    inherited = [
        (d, k)
        for d in ds
        for k in range(len(d.children))
        if d.children.visits[k] > 0 and d.children.candidate[k] >= 0 and not math.isfinite(d.children.network_moves_left[k])
    ]
    assert inherited, "the tree form leaves visited move-children unevaluated"
    assert all(math.isfinite(best_child_moves_left(d, k)) for d, k in inherited)
    tc = TargetConfig(
        short_term_horizons=NET.short_term_horizons,
        deblunder_threshold=-1.0,
        deblunder_band=0.0,
        moves_left_censored=("unknown", "action_limit"),  # the capped game's moves-left kept, to compare
    )
    t = build_targets(g.record, tc)
    assert np.isfinite(t.moves_left).all()
    whole, from_inherited = 0, 0
    for i, d in enumerate(ds):
        if t.deblunder_weight[i] == 0:
            continue
        assert t.deblunder_weight[i] == 1.0  # band 0: every blend whole
        b = int(np.argmax(t.pruned[i]))
        c = d.children
        mine = np.array([(1 - c.d[b] + c.q[b]) / 2, c.d[b], (1 - c.d[b] - c.q[b]) / 2])  # the acting player's view
        assert t.wdl[i] == pytest.approx(mine, abs=1e-12)
        assert t.moves_left[i] == pytest.approx(best_child_moves_left(d, b) + (1.0 if c.candidate[b] < 0 else 0.0), abs=1e-9)
        whole += 1
        from_inherited += int(c.candidate[b] >= 0 and not math.isfinite(c.network_moves_left[b]))
    assert whole >= 2 and from_inherited >= 1, (whole, from_inherited)


@requires_engine
@pytest.mark.parametrize("flags", [dict(), dict(opp_reply=True)], ids=["default", "reply"])
def test_record_targets_feed_every_loss(evaluator, flags):
    """A batch of record-derived rows: every loss finite (forward only - no optimiser step here)."""
    net = replace(NET, **flags)
    with torch.no_grad():
        g = generate_game(evaluator, GAMES["ply"], net_cfg=NET, keep_positions=True, target_config=TARGET_CONFIGS["default"])
    rows = [Row(p.state, g.targets, i) for i, p in enumerate(g.positions)]
    batch, tgt = collate_rows(rows, net)
    model = build_model(net, seed=1).eval()
    with torch.no_grad():
        losses = compute_losses(model(batch), tgt)
    assert all(torch.isfinite(v) for v in losses.values()), losses
    assert ("opp_reply" in losses) == bool(flags)
    full = tgt.full_search
    assert 0 < int(full.sum()) < len(rows), "PCR (I5) mixes full and cheap searches"
    assert torch.allclose(tgt.policy.sum(-1) + tgt.submit, full.float())  # cheap searches: no policy target
    assert bool((tgt.effort[~full] == -1).all()) and bool(torch.isnan(tgt.root_value[~full]).all())


def test_missing_targets_are_masked_in_the_losses():
    """A NaN WDL row, moves-left or per-board label is no target: the loss is the mean over the
    targets present - equal to the loss of the batch without them."""
    from t5d.net import collate, synthetic_states
    from t5d.train.smoke import synthetic_targets

    cfg = NetConfig(trunk_dtype="fp32", n_layers=1)
    model = build_model(cfg, seed=3).eval()
    batch = collate(synthetic_states(3, [2, 3, 2]))
    tgt = synthetic_targets(batch, cfg, torch.Generator().manual_seed(3))
    wdl = torch.nn.functional.one_hot(tgt.wdl, 3).float()
    with torch.no_grad():
        out = model(batch)
    full = compute_losses(out, replace(tgt, wdl=wdl))
    masked_wdl = wdl.clone()
    masked_wdl[1] = math.nan
    ml = tgt.moves_left.clone()
    ml[2] = math.nan
    board = tgt.board.clone()
    board[0, :, 2] = math.nan
    masked = compute_losses(out, replace(tgt, wdl=masked_wdl, moves_left=ml, board=board))
    keep = torch.tensor([True, False, True])
    lw = -(wdl * torch.log_softmax(out.wdl_logits, -1)).sum(-1)
    assert torch.allclose(masked["wdl"], lw[keep].mean())
    lm = torch.nn.functional.huber_loss(out.moves_left_log, torch.log1p(tgt.moves_left), reduction="none")
    assert torch.allclose(masked["moves_left"], lm[:2].mean())
    assert torch.isfinite(masked["board"]) and not torch.allclose(masked["board"], full["board"])
