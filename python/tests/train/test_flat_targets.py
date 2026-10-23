"""D2's flat form's policy target (spec section 8.2: "the flat form's policy targets are per-board visit
marginals"; BUILD_LOG issue 242): the visits over whole turns, pruned as at the ply split (I2),
marginalised onto each playable board's options - its moves, the non-branching jumps arriving on it,
leaving it - by hand on constructed roots; the divergence (I12), the loss that trains the leave head,
the record format (I15, version 2) and smoke steps on synthetic flat rows."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from t5d.engine.capi import MOVE_DTYPE, T5D_FLAT_JUMPS_BOTH, T5D_FLAT_JUMPS_SOURCE, T5D_MOVE_BRANCHING, T5D_MOVE_TRAVEL
from t5d.net import NetConfig, build_model, collate, synthetic_states
from t5d.train.losses import LossConfig, Targets, compute_losses, flat_policy_ce
from t5d.train.records import CANDIDATE_TURN, GameRecord, read_game, record_differences, write_game
from t5d.train.targets import (
    LEAVE,
    Row,
    TargetConfig,
    build_targets,
    collate_rows,
    flat_boards,
    flat_policy_target,
    prune_counts,
    pruned_visits,
    read_targets,
    target_differences,
    write_targets,
)
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from tests.train.test_targets import _children, _decision

# --- a constructed root ---------------------------------------------------------------------------------
#
# Boards (the input's order): 0 = timeline 0's head (0, 4), 1 = timeline 1's head (1, 4), 2 = timeline
# 0's past board (0, 3); the acting player's playable boards are 0 and 1. Candidates:
#   c0, c1 - physical moves on timeline 0;  c2 - a physical move on timeline 1;
#   c3 - a non-branching jump from timeline 0's head onto timeline 1's (an option of board 0, and an
#        arrival on board 1 under flat_jumps BothBoards);
#   c4 - a branching jump from timeline 1's head into timeline 0's past (board 1's option only).


def _move(ft, fh, fs, tt, th, ts, flags=0):
    m = np.zeros(1, dtype=MOVE_DTYPE)[0]
    m["from_timeline"], m["from_half_turn"], m["from_square"] = ft, fh, fs
    m["to_timeline"], m["to_half_turn"], m["to_square"] = tt, th, ts
    m["flags"] = flags
    return m


C = [
    _move(0, 4, 1, 0, 4, 2),
    _move(0, 4, 3, 0, 4, 4),
    _move(1, 4, 5, 1, 4, 6),
    _move(0, 4, 7, 1, 4, 8, T5D_MOVE_TRAVEL),
    _move(1, 4, 9, 0, 3, 10, T5D_MOVE_TRAVEL | T5D_MOVE_BRANCHING),
]
J6 = _move(0, 4, 7, 1, 4, 8, T5D_MOVE_TRAVEL | T5D_MOVE_BRANCHING)  # c3 played after board 1 moved: it branches (J6)
STRAY = _move(0, 4, 11, 0, 4, 12)  # a move the candidates do not hold (issue 168)
#: The turns and their visits.
TURNS = [
    ([C[0], C[2]], 10),  # board 0: c0, board 1: c2
    ([C[1]], 6),  # board 0: c1, board 1: leave
    ([C[3]], 4),  # board 0: c3; board 1: the arrival c3 (BothBoards) - outside its distribution (SourceOnly)
    ([C[2]], 3),  # board 0: leave, board 1: c2
    ([C[2], J6], 2),  # board 0: c3 (by its coordinates), board 1: c2
    ([STRAY, C[2]], 1),  # board 0: unmatched - left out, board 1: c2
]
EXACT = TargetConfig(forced_k=0.0, prune_single_visit=False, deblunder=False)


def _flat_decision(*, visits=None, steered=None, leave=(0.5, -0.25, 0.0), logits=(0.3, -0.1, 0.8, 0.2, -0.6), full=True, **kw):
    v = [t[1] for t in TURNS] if visits is None else visits
    n = len(TURNS)
    ch = _children(v, [0.1 * i for i in range(n)], [1.0 / n] * n, candidate=[CANDIDATE_TURN] * n, steered=steered)
    ch.turn_length = np.array([len(t[0]) for t in TURNS], dtype="<i4")
    d = _decision(
        0,
        ch,
        boards=((0, 4), (1, 4), (0, 3)),
        n_moves=5,
        full=full,
        flat=True,
        playable=[True, True, False],
        raw_leave_logits=np.array(leave, dtype=np.float32),
        turn_moves=np.array([m for t in TURNS for m in t[0]], dtype=MOVE_DTYPE),
        **kw,
    )
    d.candidates = np.array(C, dtype=MOVE_DTYPE)
    d.raw_logits = np.array(logits, dtype=np.float32)
    return d


def _game(decisions, flat_jumps=None):
    search = {} if flat_jumps is None else {"full": {"flat_jumps": flat_jumps, "leave_share": 0.5}}
    header = {"result": {"end": "checkmate", "winner": 0, "actions": 1, "final_active_timelines": [0, 1]}, "search": search}
    return GameRecord(header=header, decisions=decisions)


def _as_dict(rows):
    return {(int(b), int(c)): float(p) for b, c, p in rows}


def test_the_boards_options():
    d = _flat_decision()
    both, source = flat_boards(d, True), flat_boards(d, False)
    assert [(fb.board, fb.timeline, fb.own, fb.arrivals) for fb in both] == [(0, 0, (0, 1, 3), ()), (1, 1, (2, 4), (3,))]
    assert [(fb.board, fb.own, fb.arrivals) for fb in source] == [(0, (0, 1, 3), ()), (1, (2, 4), ())]
    assert both[1].options == (2, 4, 3, LEAVE)


def test_the_marginals_by_hand():
    """Board 0: c0 10, c1 6, c3 4 + 2 (the J6 twin by its coordinates), leave 3; the stray move's 1
    left out - over 25. Board 1: c2 10 + 3 + 2 + 1, the arrival c3 4, c4 0, leave 6 - over 26."""
    t = build_targets(_game([_flat_decision()]), EXACT)
    assert t.policy[0] is None and t.soft_policy[0] is None and bool(t.full_search[0])
    got = _as_dict(t.flat[0])
    want = {(0, 0): 10 / 25, (0, 1): 6 / 25, (0, 3): 6 / 25, (0, LEAVE): 3 / 25}
    want |= {(1, 2): 16 / 26, (1, 4): 0.0, (1, 3): 4 / 26, (1, LEAVE): 6 / 26}
    assert got.keys() == want.keys() and all(got[k] == pytest.approx(v, abs=1e-15) for k, v in want.items())
    # every option of every board is listed, in the boards' and options' order
    assert [(int(b), int(c)) for b, c, _ in t.flat[0]] == [(0, 0), (0, 1), (0, 3), (0, -1), (1, 2), (1, 4), (1, 3), (1, -1)]


def test_the_marginals_under_source_only():
    """flat_jumps SourceOnly: the arrival is no option of board 1 - its 4 visits left out there."""
    t = build_targets(_game([_flat_decision()], T5D_FLAT_JUMPS_SOURCE), EXACT)
    got = _as_dict(t.flat[0])
    assert got[(0, 3)] == pytest.approx(6 / 25) and (1, 3) not in got
    assert got[(1, 2)] == pytest.approx(16 / 22) and got[(1, LEAVE)] == pytest.approx(6 / 22) and got[(1, 4)] == 0.0


def test_steered_visits_are_excluded_or_weighted_as_at_the_ply_split():
    """I2: 4 of the first turn's 10 visits steered - excluded (6 left), or kept at the provenance
    weight 0.5 (8 left)."""
    steered = [4, 0, 0, 0, 0, 0]
    ex = _as_dict(build_targets(_game([_flat_decision(steered=steered)]), EXACT).flat[0])
    assert ex[(0, 0)] == pytest.approx(6 / 21) and ex[(1, 2)] == pytest.approx(12 / 22)
    prov = _as_dict(build_targets(_game([_flat_decision(steered=steered)]), replace(EXACT, policy_mode="provenance")).flat[0])
    assert prov[(0, 0)] == pytest.approx(8 / 23) and prov[(1, LEAVE)] == pytest.approx(6 / 24)


def test_the_pruning_per_board_by_hand():
    """Issue 242(b): the turns' visits marginalised first, then each board's option counts pruned as
    the ply split prunes root children - an option's prior its turns' priors summed, its value their
    visit-weighted mean (all 0 here) - against the board's most visited option. Turn priors 0.5, then
    0.1 each; visits 40, 12, 4, 3, 2, 1; 63 root visits (sqrt 7.937), cPUCT 1.5.

    Board 0 (the stray turn's 1 visit and 0.1 prior left out): c0 40 (prior 0.5), c1 12 (0.1), c3
    4 + 2 (0.2), leave 3 (0.1); the bar is c0's 1.5 * 0.5 * 7.937 / 41 = 0.1452; c1 may lose
    floor(sqrt(2 * 0.1 * 61)) = 3 and does (1.5 * 0.1 * 7.937 / 10 = 0.119 < bar): 9; c3 and leaving
    keep theirs - 40, 9, 6, 3 over 58. Board 1: c2 40 + 3 + 2 + 1 = 46 (prior 0.8), leave 12 (0.1),
    the arrival c3 4 (0.1); the bar 1.5 * 0.8 * 7.937 / 47 = 0.2027; leaving loses 3: 46, 0, 4, 9 over
    59. Pruned per turn (the arm), turn 1 loses 3 and the one-visit turn goes: board 1 is 45, 0, 4, 9
    over 58. The root's search value reads the turns' pruned visits under either arm."""
    prior = [0.5] + [0.1] * 5
    d = _flat_decision(visits=[40, 12, 4, 3, 2, 1])
    d = replace(d, children=replace(d.children, q=np.zeros(6), prior=np.array(prior), prior_noise_free=np.array(prior)))
    cfg = TargetConfig(deblunder=False)
    board = build_targets(_game([d]), cfg)
    got = _as_dict(board.flat[0])
    want = {(0, 0): 40 / 58, (0, 1): 9 / 58, (0, 3): 6 / 58, (0, LEAVE): 3 / 58}
    want |= {(1, 2): 46 / 59, (1, 4): 0.0, (1, 3): 4 / 59, (1, LEAVE): 9 / 59}
    assert got.keys() == want.keys() and all(got[k] == pytest.approx(v, abs=1e-15) for k, v in want.items()), got
    pruned = pruned_visits(d, cfg)
    assert pruned.tolist() == [40, 9, 4, 3, 2, 0]
    turn = build_targets(_game([d]), replace(cfg, flat_pruning="turn"))
    assert np.array_equal(turn.flat[0], flat_policy_target(d, pruned))
    got = _as_dict(turn.flat[0])
    assert got[(1, 2)] == pytest.approx(45 / 58) and got[(1, LEAVE)] == pytest.approx(9 / 58) and got[(0, 1)] == pytest.approx(9 / 58)
    for t in (board, turn):
        assert np.array_equal(t.pruned[0], pruned)
        assert t.search_value[0] == pytest.approx(float((pruned * d.children.q).sum() / pruned.sum()))
    with pytest.raises(ValueError, match="flat_pruning"):
        replace(cfg, flat_pruning="joint").validate()


def test_an_option_is_read_as_the_turns_that_take_it():
    """The per-board pruning's replica (issue 242(b)): an option's prior is its turns' priors summed -
    unvisited turns' included - and its value and exploration scale its turns' visit-weighted means;
    checked against ``prune_counts`` (the ply split's pruning) on those hand-listed readings (E3's
    scale 1 under the defaults; a non-unit one here, read the same way). Turn 1 (c1 on board 0, leaving
    board 1) is worth 0.2: its options keep their 12 visits, where at value 0 they lose 3; an unvisited
    turn's prior moves the second case."""
    prior = np.array([0.5] + [0.1] * 5)
    q = np.array([0.0, 0.2, 0.05, 0.0, 0.1, 0.0])
    scale = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    d = _flat_decision(visits=[40, 12, 4, 3, 2, 0])
    d = replace(d, children=replace(d.children, q=q, prior=prior, prior_noise_free=prior, exploration_scale=scale))
    cfg = TargetConfig(deblunder=False)
    got = build_targets(_game([d]), cfg).flat[0]
    sqrt_parent = math.sqrt(float(d.root_visits))
    # board 0: c0, c1, c3 (turns 2 and 4), leave (turn 3); the unvisited turn 5 is the stray move's
    n0 = np.array([40.0, 12.0, 6.0, 3.0])
    p0 = np.array([0.5, 0.1, 0.2, 0.1])
    q0 = np.array([0.0, 0.2, (4 * 0.05 + 2 * 0.1) / 6, 0.0])
    s0 = np.array([1.0, 1.0, (4 * 2.0 + 2 * 1.0) / 6, 1.0])
    # board 1: c2 (turns 0, 3, 4 and the unvisited 5), c4, the arrival c3 (turn 2), leave (turn 1)
    n1 = np.array([45.0, 0.0, 4.0, 12.0])
    p1 = np.array([0.8, 0.0, 0.1, 0.1])
    q1 = np.array([(2 * 0.1) / 45, 0.0, 0.05, 0.2])
    s1 = np.array([1.0, 1.0, 2.0, 1.0])
    want = []
    for n, p, qq, s in ((n0, p0, q0, s0), (n1, p1, q1, s1)):
        kept = prune_counts(n, p, qq, s, sqrt_parent, 1.5, cfg)
        want.append(kept / kept.sum())
        assert kept[0] == n[0]
    assert kept.tolist() == [45.0, 0.0, 4.0, 12.0]
    assert np.allclose(got[:, 2], np.concatenate(want), rtol=0, atol=1e-15)
    assert prune_counts(n1, p1, np.zeros(4), s1, sqrt_parent, 1.5, cfg)[3] == 9.0  # at value 0 leaving loses 3
    assert prune_counts(n0, p0, np.zeros(4), s0, sqrt_parent, 1.5, cfg)[1] == 9.0  # and c1 too
    # An unvisited turn's prior counts: c3's turns are turn 2 (12 visits) and turn 4 (none) - prior 0.2,
    # under which c3 keeps its 12 (1.5 * 0.2 * 7.937 / 12 = 0.198 > the bar 0.145); at 0.1 it would lose 3.
    d = _flat_decision(visits=[40, 6, 12, 3, 0, 1])
    d = replace(d, children=replace(d.children, q=np.zeros(6), prior=prior, prior_noise_free=prior))
    got = _as_dict(build_targets(_game([d]), cfg).flat[0])
    assert [got[(0, k)] for k in (0, 1, 3, LEAVE)] == pytest.approx([40 / 61, 6 / 61, 12 / 61, 3 / 61], abs=1e-15)


def _spread_decision():
    """The review's probe (issue 242(b)): the best turn (10 visits) moves a0 on board 0 and leaves
    board 1; five one-visit turns each move another a_k on board 0 and B on board 1. Equal priors and
    values."""
    a = [_move(0, 4, k, 0, 4, 20 + k) for k in range(6)]
    b = _move(1, 4, 5, 1, 4, 6)
    turns = [([a[0]], 10)] + [([a[k], b], 1) for k in range(1, 6)]
    n = len(turns)
    ch = _children([v for _, v in turns], [0.0] * n, [1.0 / n] * n, candidate=[CANDIDATE_TURN] * n)
    ch.turn_length = np.array([len(t) for t, _ in turns], dtype="<i4")
    d = _decision(
        0,
        ch,
        boards=((0, 4), (1, 4)),
        n_moves=7,
        full=True,
        flat=True,
        playable=[True, True],
        raw_leave_logits=np.zeros(2, np.float32),
        turn_moves=np.array([m for t, _ in turns for m in t], dtype=MOVE_DTYPE),
    )
    d.candidates = np.array([*a, b], dtype=MOVE_DTYPE)
    d.raw_logits = np.zeros(7, np.float32)
    return d


def test_an_option_spread_over_one_visit_turns_keeps_its_marginal():
    """The review's case (issue 242(b)): board 1's B has 5 visits over five one-visit turns against
    leaving's 10 in one. Marginalised first (the default), B keeps 1/3 (its five visits survive the
    board's pruning); pruned per turn (the arm), every one-visit turn goes and B with them - leaving
    1, B 0. Board 0's a1..a5 are one visit each on the board too: pruned under both (a0 1)."""
    d = _spread_decision()
    cfg = TargetConfig(deblunder=False)
    board = _as_dict(build_targets(_game([d]), cfg).flat[0])
    assert board[(1, LEAVE)] == pytest.approx(2 / 3) and board[(1, 6)] == pytest.approx(1 / 3)
    assert board[(0, 0)] == 1.0
    turn = _as_dict(build_targets(_game([d]), replace(cfg, flat_pruning="turn")).flat[0])
    assert turn[(1, LEAVE)] == 1.0 and turn[(1, 6)] == 0.0 and turn[(0, 0)] == 1.0
    unpruned = _as_dict(build_targets(_game([d]), EXACT).flat[0])
    assert unpruned[(0, 0)] == pytest.approx(2 / 3) and unpruned[(1, 6)] == pytest.approx(1 / 3)


def test_cheap_and_unvisited_flat_decisions_have_no_policy_target():
    cheap = build_targets(_game([_flat_decision(full=False)]), EXACT)
    assert cheap.flat[0] is None and cheap.policy[0] is None and not cheap.full_search[0] and math.isnan(cheap.kl[0])
    none = build_targets(_game([_flat_decision(visits=[0] * 6)]), EXACT)
    assert none.flat[0] is None and not none.full_search[0]


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def _kl(t, p):
    t, p = np.asarray(t), np.asarray(p)
    k = t > 0
    return float((t[k] * np.log(t[k] / p[k])).sum())


def test_the_divergence_is_the_boards_kls_summed():
    """I12: KL(marginal || raw prior) per board, summed - the raw prior the softmax of the board's raw
    logits (moves, arrivals) and its raw leave logit; with no leave logit, leaving at the share."""
    logits = (0.3, -0.1, 0.8, 0.2, -0.6)
    d = _flat_decision(logits=logits)
    t = build_targets(_game([d]), EXACT)
    logits = np.array(logits, dtype=np.float32).astype(np.float64)  # as the record keeps them
    b0 = _softmax([logits[0], logits[1], logits[3], 0.5])
    b1 = _softmax([logits[2], logits[4], logits[3], -0.25])
    want = _kl([10 / 25, 6 / 25, 6 / 25, 3 / 25], b0) + _kl([16 / 26, 0, 4 / 26, 6 / 26], b1)
    assert t.kl[0] == pytest.approx(want, abs=1e-12)
    d = _flat_decision(logits=tuple(logits), leave=(math.nan, -0.25, 0.0))
    p0 = np.append(0.5 * _softmax([logits[0], logits[1], logits[3]]), 0.5)
    want = _kl([10 / 25, 6 / 25, 6 / 25, 3 / 25], p0) + _kl([16 / 26, 0, 4 / 26, 6 / 26], b1)
    assert build_targets(_game([d]), EXACT).kl[0] == pytest.approx(want, abs=1e-12)


def test_a_flat_record_and_its_targets_round_trip(tmp_path):
    """I15's record version 2 carries the flat fields bit for bit; the targets re-derived from the
    stored record are those built from the live one."""
    d = _flat_decision()
    rec = _game([d])
    rec.header["record_version"] = 2
    back = read_game(write_game(tmp_path / "f.npz", rec))
    assert record_differences(rec, back) == []
    e = back.decisions[0]
    assert e.flat and e.playable.tolist() == [True, True, False] and e.turn(4).tobytes() == np.array([C[2], J6]).tobytes()
    a, b = build_targets(rec, EXACT), build_targets(back, EXACT)
    assert target_differences(a, b) == []
    assert target_differences(a, read_targets(write_targets(tmp_path / "t.npz", a))) == []


def test_a_flat_decisions_children_are_all_turns():
    d = _flat_decision()
    with pytest.raises(ValueError, match="turn"):
        _decision(0, replace(d.children, candidate=np.array([CANDIDATE_TURN] * 5 + [0])), flat=True, n_moves=5)
    with pytest.raises(ValueError, match="turn"):
        _decision(0, d.children, flat=False, n_moves=5, turn_moves=d.turn_moves)


# --- the loss ---------------------------------------------------------------------------------------------------


def test_the_flat_loss_by_hand():
    """Each board's cross-entropy over its options' logits (moves from the policy, leaving from the
    leave head), summed over the boards (the placeholder) or averaged, averaged over the rows."""
    policy = torch.tensor([[0.3, -0.1, 0.8, 0.2, -0.6], [1.0, 0.0, -1.0, float("-inf"), float("-inf")]])
    leave = torch.tensor([[0.5, -0.25, 7.0], [0.1, 0.2, 0.3]])
    out = type("Out", (), {"policy_logits": policy, "leave_logits": leave})()
    t0 = torch.tensor([0.4, 0.24, 0.24, 0.12, 16 / 26, 0.0, 4 / 26, 6 / 26])
    tgt = Targets(
        policy=torch.zeros(2, 5),
        wdl=torch.zeros(2),
        root_value=torch.zeros(2),
        short_term=torch.zeros(2, 1),
        moves_left=torch.zeros(2),
        board=torch.zeros(2, 3, 3),
        square=torch.zeros(2, 1, 1),
        effort=torch.zeros(2, dtype=torch.long),
        flat_rows=torch.tensor([True, True]),
        flat_option=torch.tensor([[0, 1, 3, 5 + 0, 2, 4, 3, 5 + 1], [0, 1, 5 + 2, -1, -1, -1, -1, -1]]),
        flat_group=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [2, 2, 2, -1, -1, -1, -1, -1]]),
        flat_target=torch.stack([t0, torch.tensor([0.5, 0.25, 0.25, 0, 0, 0, 0, 0])]),
    )

    def ce(z, t):
        return -float((torch.tensor(t, dtype=torch.float64) * torch.log_softmax(torch.tensor(z, dtype=torch.float64), 0)).sum())

    b0 = ce([0.3, -0.1, 0.2, 0.5], [0.4, 0.24, 0.24, 0.12])
    b1 = ce([0.8, -0.6, 0.2, -0.25], [16 / 26, 0.0, 4 / 26, 6 / 26])
    r1 = ce([1.0, 0.0, 0.3], [0.5, 0.25, 0.25])
    keep = torch.tensor([True, True])
    assert float(flat_policy_ce(out, tgt, keep, "sum")) == pytest.approx(((b0 + b1) + r1) / 2, abs=1e-5)
    assert float(flat_policy_ce(out, tgt, keep, "mean")) == pytest.approx(((b0 + b1) / 2 + r1) / 2, abs=1e-5)
    assert float(flat_policy_ce(out, tgt, torch.tensor([False, True]), "sum")) == pytest.approx(r1, abs=1e-5)


def _flat_rows(seed=4, counts=(3, 5, 4, 6)):
    states = synthetic_states(seed, list(counts), n_timelines=3)
    rec = synthetic_record(seed, states, full_every=4, flat=True)
    targets = build_targets(rec, TargetConfig(short_term_horizons=NetConfig().short_term_horizons))
    return [Row(s, targets, i, synthetic=True) for i, s in enumerate(states)], rec, targets


def test_synthetic_flat_rows_feed_the_flat_loss_only():
    """Flat rows carry a per-board target and no ply-split one: the policy loss, the soft policy and
    D9's optimistic policy skip them; the flat loss reaches the leave head and the policy's
    projections, and no other policy head."""
    rows, rec, targets = _flat_rows()
    assert any(t is not None for t in targets.flat) and all(p is None for p in targets.policy)
    net = NetConfig(trunk_dtype="fp32", n_layers=1, optimistic_policy=True)
    model = build_model(net, seed=2)
    batch, tgt = collate_rows(rows, net)
    assert tgt.flat_rows is not None and bool(tgt.flat_rows.any())
    with torch.enable_grad():
        losses = compute_losses(model(batch), tgt)
        assert torch.isfinite(losses["flat_policy"]) and float(losses["flat_policy"].detach()) > 0
        assert (
            float(losses["policy"].detach()) == 0.0
            and float(losses["soft_policy"].detach()) == 0.0
            and float(losses["optimistic"].detach()) == 0.0
        )
        model.zero_grad(set_to_none=True)
        losses["flat_policy"].backward()
    assert model.leave.out.weight.grad.abs().sum() > 0 and model.policy.src.weight.grad.abs().sum() > 0
    assert model.aux.soft_policy.pointer.src.weight.grad is None or model.aux.soft_policy.pointer.src.weight.grad.abs().sum() == 0
    assert model.optimistic.pointer.src.weight.grad is None or model.optimistic.pointer.src.weight.grad.abs().sum() == 0
    assert all(torch.isfinite(v) for v in losses.values())
    assert LossConfig().weights["flat_policy"] == 1.0


def test_smoke_steps_on_synthetic_flat_rows():
    """No training runs: at most three optimiser steps, on synthetic flat rows only - every loss
    finite, the leave head moved by them."""
    rows, _, _ = _flat_rows(seed=6)
    net = NetConfig(trunk_dtype="fp32", n_layers=1)
    model = build_model(net, seed=3)
    before = model.leave.out.weight.detach().clone()
    trainer = Trainer(model, net, TrainerConfig(max_steps=3, batch_size=4))
    with torch.enable_grad():
        for _ in range(3):
            values = trainer.step_on(rows)
            assert all(math.isfinite(v) for v in values.values()) and values["flat_policy"] > 0
        with pytest.raises(RuntimeError):
            trainer.step_on(rows)
    assert not torch.equal(before, model.leave.out.weight.detach())
    with pytest.raises(RuntimeError, match="synthetic"):
        Trainer(build_model(net, seed=4), net).step_on([replace(r, synthetic=False) for r in rows])


def test_a_synthetic_flat_record_round_trips(tmp_path):
    _, rec, targets = _flat_rows(seed=8)
    back = read_game(write_game(tmp_path / "s.npz", rec))
    assert record_differences(rec, back) == []
    assert target_differences(targets, build_targets(back, TargetConfig(short_term_horizons=NetConfig().short_term_horizons))) == []
    assert collate(synthetic_states(8, [3])).moves.shape[0] == 1


# --- the review's cases (issue 242; the mutation run's survivors) ---------------------------------------------

H = NetConfig().short_term_horizons


def test_the_default_flat_reduction_is_the_sum():
    """``compute_losses``' flat loss under the default ``LossConfig`` is the per-board sum (issue
    242(e)'s placeholder), not the mean - on rows where the two differ."""
    rows, _, _ = _flat_rows()
    net = NetConfig(trunk_dtype="fp32", n_layers=1)
    model = build_model(net, seed=2).eval()
    batch, tgt = collate_rows(rows, net)
    with torch.no_grad():
        out = model(batch)
        got = compute_losses(out, tgt)["flat_policy"]
        keep = tgt.full_search.bool() & tgt.flat_rows
        want, other = flat_policy_ce(out, tgt, keep, "sum"), flat_policy_ce(out, tgt, keep, "mean")
    assert LossConfig().flat_board_reduction == "sum" and not torch.isclose(want, other)
    assert torch.isclose(got, want, atol=1e-6)


def test_the_mean_counts_the_last_board():
    """A row with options on board 0 and on the padded width's last board (Nmax - 1, board 1 having
    none): the mean divides by two boards - the padding's group is a dump group of its own, never the
    last board's."""
    out = SimpleNamespace(policy_logits=torch.tensor([[0.3, -0.1, 0.8]]), leave_logits=torch.tensor([[0.5, 9.0, -0.25]]))
    tgt = Targets(
        policy=torch.zeros(1, 3),
        wdl=torch.zeros(1),
        root_value=torch.zeros(1),
        short_term=torch.zeros(1, 1),
        moves_left=torch.zeros(1),
        board=torch.zeros(1, 3, 3),
        square=torch.zeros(1, 1, 1),
        effort=torch.zeros(1, dtype=torch.long),
        flat_rows=torch.tensor([True]),
        flat_option=torch.tensor([[0, 3 + 0, 1, 2, 3 + 2, -1]]),
        flat_group=torch.tensor([[0, 0, 2, 2, 2, -1]]),
        flat_target=torch.tensor([[0.6, 0.4, 0.2, 0.3, 0.5, 0.0]]),
    )

    def ce(z, t):
        return -float((torch.tensor(t, dtype=torch.float64) * torch.log_softmax(torch.tensor(z, dtype=torch.float64), 0)).sum())

    b0, b2 = ce([0.3, 0.5], [0.6, 0.4]), ce([-0.1, 0.8, -0.25], [0.2, 0.3, 0.5])
    assert float(flat_policy_ce(out, tgt, torch.tensor([True]), "mean")) == pytest.approx((b0 + b2) / 2, abs=1e-5)
    assert float(flat_policy_ce(out, tgt, torch.tensor([True]), "sum")) == pytest.approx(b0 + b2, abs=1e-5)


@pytest.mark.parametrize("name", ["policy", "soft_policy", "optimistic"])
def test_flat_rows_do_not_dilute_the_ply_losses(name):
    """In a batch mixing ply-split and flat rows, the policy, soft-policy and optimistic losses are
    those of the ply rows alone: a flat row neither adds to them nor dilutes them (issue 242(h))."""
    flat = _flat_rows(seed=4)[0]
    states = synthetic_states(9, [2, 4, 3, 5], n_timelines=3)
    ply = build_targets(synthetic_record(9, states, full_every=3), TargetConfig(short_term_horizons=H))
    ply_rows = [Row(st, ply, i, synthetic=True) for i, st in enumerate(states)]
    net = NetConfig(trunk_dtype="fp32", n_layers=1, optimistic_policy=True)
    model = build_model(net, seed=2).eval()
    with torch.no_grad():
        b, t = collate_rows(ply_rows, net)
        alone = compute_losses(model(b), t)[name]
        b, t = collate_rows(ply_rows + flat, net)
        assert t.flat_rows is not None and bool(t.flat_rows.any())
        mixed = compute_losses(model(b), t)[name]
    assert float(alone) > 0
    assert float(mixed) == pytest.approx(float(alone), rel=1e-4, abs=1e-5)


def test_a_batch_with_flat_rows_needs_the_flat_weight():
    """A caller's ``LossConfig.weights`` without ``"flat_policy"`` is refused on a batch with flat rows
    - never the flat loss silently dropped from the total - and accepted on a batch without them."""
    rows, _, _ = _flat_rows()
    net = NetConfig(trunk_dtype="fp32", n_layers=1)
    model = build_model(net, seed=2).eval()
    weights = {k: v for k, v in LossConfig().weights.items() if k != "flat_policy"}
    with torch.no_grad():
        b, t = collate_rows(rows, net)
        with pytest.raises(ValueError, match="flat_policy"):
            compute_losses(model(b), t, LossConfig(weights=weights))
        assert "flat_policy" in compute_losses(model(b), t, LossConfig(weights={**weights, "flat_policy": 0.0}))
        states = synthetic_states(9, [2, 4], n_timelines=3)
        ply = build_targets(synthetic_record(9, states, full_every=1), TargetConfig(short_term_horizons=H))
        b, t = collate_rows([Row(st, ply, i, synthetic=True) for i, st in enumerate(states)], net)
        assert "flat_policy" not in compute_losses(model(b), t, LossConfig(weights=weights))


def test_the_fallback_prior_reads_the_recorded_leave_share():
    """I12 with no leave logit at a board and a recorded ``leave_share`` of 0.3 (not the symmetric
    0.5): the raw prior gives leaving 0.3 and the moves the other 0.7."""
    logits = np.array((0.3, -0.1, 0.8, 0.2, -0.6), dtype=np.float32).astype(np.float64)
    d = _flat_decision(logits=tuple(logits), leave=(math.nan, -0.25, 0.0))
    rec = _game([d])
    rec.header["search"] = {"full": {"flat_jumps": T5D_FLAT_JUMPS_BOTH, "leave_share": 0.3}}
    p0 = np.append(0.7 * _softmax([logits[0], logits[1], logits[3]]), 0.3)
    b1 = _softmax([logits[2], logits[4], logits[3], -0.25])
    want = _kl([10 / 25, 6 / 25, 6 / 25, 3 / 25], p0) + _kl([16 / 26, 0, 4 / 26, 6 / 26], b1)
    assert build_targets(rec, EXACT).kl[0] == pytest.approx(want, abs=1e-12)


def test_collate_refuses_a_flat_entry_outside_the_input():
    rows, _, targets = _flat_rows()
    i = next(k for k, f in enumerate(targets.flat) if f is not None)
    net = NetConfig(trunk_dtype="fp32", n_layers=1)
    for column, value in ((0, 99), (0, -1), (1, 99), (1, -2)):  # a board or a candidate the input does not have
        bad = [None if f is None else f.copy() for f in targets.flat]
        bad[i][0, column] = value
        t2 = replace(targets, flat=bad)
        with pytest.raises(ValueError, match="flat target entry"):
            collate_rows([replace(r, targets=t2) for r in rows], net)


def test_the_reply_prototype_has_no_target_at_flat_decisions():
    states = synthetic_states(4, [3, 5, 4, 6], n_timelines=3)
    rec = synthetic_record(4, states, full_every=4, flat=True)
    t = build_targets(rec, TargetConfig(short_term_horizons=H), positions=states)
    assert len(t.reply) == len(states) and all(r is None for r in t.reply)


def test_a_travel_later_in_a_flat_turn_is_a_landing():
    """The played turn is [c2, the J6 branch]: its second move travels onto (1, 4), so board 1 is
    travelled into - every move of the turn counts, not its first (issue 242(g))."""
    d = _flat_decision(played=4)
    assert len(d.played_moves()) == 2 and d.played_moves().tobytes() == np.array([C[2], J6]).tobytes()
    t = build_targets(_game([d]), EXACT)
    assert t.board[0][1, 0] == 1.0 and t.board[0][0, 0] == 0.0


def test_a_flat_decision_with_no_turn_children_is_refused():
    with pytest.raises(ValueError, match="turn"):
        _decision(0, _children([3, 2], [0.0, 0.1], [0.5, 0.5]), flat=True, n_moves=2)


def test_turn_lengths_must_sum_to_the_turn_moves():
    d = _flat_decision()
    with pytest.raises(ValueError, match="turn"):
        replace(d, turn_moves=d.turn_moves[:-1])


def test_a_best_turn_childs_moves_left_counts_the_turn_it_ends():
    """I13 at a flat decision (issue 242(g)): the best child is a whole turn, which ends the turn in
    progress as a submit does - its moves-left (10) plus one; an unevaluated best turn child has no
    estimate (the decision's own raw moves-left is a move child's fallback only) - masked."""
    d = _flat_decision(played=1)
    d = replace(d, children=replace(d.children, q=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])))
    cfg = replace(EXACT, deblunder=True)
    t = build_targets(_game([d]), cfg)
    assert t.deblunder_weight[0] == 1.0 and t.moves_left[0] == 11.0
    nan = replace(d, children=replace(d.children, network_moves_left=np.array([math.nan] + [10.0] * 5)))
    assert math.isnan(build_targets(_game([nan]), cfg).moves_left[0])


def test_targets_written_before_the_flat_target_are_refused(tmp_path):
    t = build_targets(_game([_flat_decision()]), EXACT)
    path = write_targets(tmp_path / "t.npz", t)
    with np.load(path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files if not k.startswith("flat_")}
    np.savez_compressed(tmp_path / "old.npz", **old)
    with pytest.raises(ValueError, match="flat"):
        read_targets(tmp_path / "old.npz")


def test_record_rows_with_flat_targets_are_refused_by_step_on():
    """Issue 237(a): a flat target does not make a row steppable - only synthetic rows are."""
    rows, _, targets = _flat_rows(seed=6)
    flat_only = [replace(r, synthetic=False) for r in rows if targets.flat[r.index] is not None]
    assert flat_only
    net = NetConfig(trunk_dtype="fp32", n_layers=1)
    with pytest.raises(RuntimeError, match="synthetic"):
        Trainer(build_model(net, seed=4), net, TrainerConfig(max_steps=3, batch_size=4)).step_on(flat_only)


def test_synthetic_flat_records_carry_arrivals():
    """``synthetic_moves``: a jump landing on a playable head is non-branching - an arrival of that
    board under BothBoards; some synthetic flat decision has one, so the smoke rows exercise it."""
    found = False
    for seed in range(4):
        states = synthetic_states(seed, [3, 5, 4, 6], n_timelines=3)
        rec = synthetic_record(seed, states, full_every=4, flat=True)
        found |= any(fb.arrivals for d in rec.decisions for fb in flat_boards(d))
    assert found


def _mv(**kw):
    return SimpleNamespace(**{n: kw.get(n, 0) for n in MOVE_DTYPE.names})


def test_kl_stop_shares_keep_turns_with_a_common_first_move_apart():
    """I20's visit shares under the flat form key a child by its whole turn: two turns sharing their
    first move are two entries."""
    from t5d.train.selfplay import _visit_shares

    a, b, c = _mv(from_square=1), _mv(from_square=2), _mv(from_square=3)
    kids = [SimpleNamespace(submit=False, moves=[a, b], visits=3), SimpleNamespace(submit=False, moves=[a, c], visits=1)]
    shares = _visit_shares(SimpleNamespace(children=lambda: kids))
    assert sorted(shares.values()) == [0.25, 0.75]
