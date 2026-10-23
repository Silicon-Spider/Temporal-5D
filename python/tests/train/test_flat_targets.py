"""D2's flat form's policy target (spec section 8.2: "the flat form's policy targets are per-board visit
marginals"; BUILD_LOG issue 242): the visits over whole turns, pruned as at the ply split (I2),
marginalised onto each playable board's options - its moves, the non-branching jumps arriving on it,
leaving it - by hand on constructed roots; the divergence (I12), the loss that trains the leave head,
the record format (I15, version 2) and smoke steps on synthetic flat rows."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine.capi import MOVE_DTYPE, T5D_FLAT_JUMPS_SOURCE, T5D_MOVE_BRANCHING, T5D_MOVE_TRAVEL
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


def test_the_pruning_is_the_ply_splits():
    """KataGo's pruning reads the turn children as it reads move children: the target is the marginal
    of ``pruned_visits``, which here prunes (the one-visit turn goes; the others lose their forced
    playouts)."""
    d = _flat_decision(visits=[40, 6, 4, 3, 2, 1])
    cfg = TargetConfig()
    pruned = pruned_visits(d, cfg)
    assert pruned.sum() < 56 and pruned[0] == 40 and pruned[5] == 0
    t = build_targets(_game([d]), cfg)
    assert np.array_equal(t.flat[0], flat_policy_target(d, pruned))
    assert t.search_value[0] == pytest.approx(float((pruned * d.children.q).sum() / pruned.sum()))


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
