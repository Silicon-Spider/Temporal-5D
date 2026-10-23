"""D2's flat head (spec section 8.2: "per-board policy marginals - a head change"; BUILD_LOG issue
241): one "leave it" logit per board, read from the board's pooled state by a linear readout - what
it reads (``NetConfig.leave_pool``) a placeholder. Checked against a direct computation, under both
pools and with no per-board registers, through the cached path, and under section 15.2's A7 mirror
with the head's other pool (the default pool's mirror is in ``test_perspective.py``'s every test)."""

from __future__ import annotations

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, mirror, synthetic_state
from t5d.net.heads import board_mean_pool
from t5d.net.state import TOKEN_BOARD_REGISTER, mirror_tokens
from tests.net.test_perspective import _acting, _close, _outputs, _pair, assert_mirrored, tie


def _small(**kw) -> NetConfig:
    return NetConfig(trunk_dtype="fp32", n_layers=2, **kw)


@pytest.mark.parametrize("pool", ["squares", "registers"])
@pytest.mark.parametrize("regs", [(2, 16), (4, 0), (0, 16)], ids=["v1", "regs_4_0", "regs_0_16"])
def test_the_leave_logit_is_a_linear_readout_of_the_boards_pool(pool, regs):
    """Each board's leave logit is ``leave.out`` applied to the mean of its square tokens
    (``"squares"``) or of its per-board registers (``"registers"``; with none, its squares - the
    fallback of spec issue 225), computed here by hand from the trunk."""
    rpb, glob = regs
    cfg = _small(leave_pool=pool, regs_per_board=rpb, global_regs=glob)
    model = build_model(cfg, seed=3).eval()
    states = [synthetic_state(s, n, n_timelines=3, regs_per_board=rpb, global_regs=glob) for s, n in ((1, 5), (2, 9))]
    batch = collate(states)
    out = model(batch, return_trunk=True)
    h = out.trunk
    assert out.leave_logits.shape == batch.board_valid.shape
    w, b = model.leave.out.weight[0].double(), model.leave.out.bias[0].double()
    for i, st in enumerate(states):
        for board in range(st.n_boards):
            on = batch.board[i] == board
            kind = batch.square_mask[i] if pool == "squares" or rpb == 0 else (batch.kind[i] == TOKEN_BOARD_REGISTER) & batch.valid[i]
            pooled = h[i][on & kind].double().mean(dim=0)
            want = float(pooled @ w + b)
            assert out.leave_logits[i, board].item() == pytest.approx(want, abs=1e-5)
    if pool == "squares":
        assert torch.allclose(out.leave_logits, model.leave.out(board_mean_pool(h, batch)).squeeze(-1))


def test_the_leave_logit_reads_its_own_boards_pool_only():
    """Only a board's own pooled state reaches its leave logit through the readout: the readout's
    gradient for board b's logit is board b's pool."""
    model = build_model(_small(), seed=4).eval()
    batch = collate([synthetic_state(5, 6, n_timelines=3)])
    with torch.enable_grad():
        out = model(batch, return_trunk=True)
        pooled = board_mean_pool(out.trunk, batch)
        for b in range(6):
            model.zero_grad(set_to_none=True)
            out.leave_logits[0, b].backward(retain_graph=True)
            assert torch.allclose(model.leave.out.weight.grad[0], pooled[0, b], atol=1e-6)


def test_the_leave_logit_is_on_the_move_logits_scale_at_initialisation():
    """Section 15.2's realistic initialisation scale: the leave logits spread like the move logits
    they are softmaxed with (both of order one - neither swamps the other at a board)."""
    model = build_model(NetConfig(trunk_dtype="fp32"), seed=0).eval()
    batch = collate([synthetic_state(s, 12, n_timelines=4) for s in range(4)])
    out = model(batch)
    leave = out.leave_logits[batch.board_valid]
    moves = out.policy_logits[out.move_valid]
    assert 0.05 < float(leave.std()) < 5.0 and 0.05 < float(moves.std()) < 5.0
    assert 0.1 < float(leave.std() / moves.std()) < 10.0, (float(leave.std()), float(moves.std()))
    assert torch.isfinite(out.leave_logits).all()


@pytest.mark.parametrize("construction", ["standard", "many_timelines", "even_start", "registers_4_0", "registers_0_16"])
def test_the_mirror_with_the_register_pool(construction):
    """Section 15.2's A7 row with the head reading the per-board registers: a position and its
    colour-swapped mirror give the same leave logits board for board (and every other output)."""
    for seed in (0, 1):
        s, m = _pair(construction, seed)
        cfg = _small(leave_pool="registers", regs_per_board=s.regs_per_board, global_regs=s.global_regs)
        model = tie(build_model(cfg, seed=seed).eval())
        a, b = _outputs(model, s), _outputs(model, m)
        assert_mirrored(a, b, torch.from_numpy(mirror_tokens(s.layout)), _acting(s, m))
        assert torch.allclose(a.leave_logits, b.leave_logits, atol=1e-5)


@pytest.mark.parametrize("pool", ["squares", "registers"])
def test_the_cached_path_gives_the_dense_leave_logits(pool):
    """The cached path's heads are the network's own (``T5DNet.heads``), so a cached evaluation's
    leave logits are the dense pass's - checked on a block-causal network against its dense pass."""
    from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output

    cfg = _small(leave_pool=pool, attn_scope="block_causal")
    model = build_model(cfg, seed=2).eval()
    for state in (synthetic_state(11, 8, n_timelines=3), mirror(synthetic_state(12, 7, n_timelines=3))):
        cached = CachedEvaluator(model, CacheConfig(root_refresh=False))
        got = cached.evaluate(state, cached.root_epoch(state))
        want = reference_output(model, state, history_from=state)
        _close(got, want)
        assert torch.allclose(got.leave_logits[got.board_valid], want.leave_logits[want.board_valid], atol=1e-4)


def test_the_cached_evaluator_delivers_the_dense_leave_logits():
    """``CachedNetworkEvaluator`` hands the engine the leave logits as ``NetworkEvaluator`` does
    (``Evaluation.board_logits``, one per board, the input's order): the dense evaluator's, to the
    cached path's tolerance."""
    import numpy as np

    from t5d.engine.evaluator import CachedNetworkEvaluator, Leaf, NetworkEvaluator, check_evaluation
    from t5d.net.cache import CacheConfig, CachedEvaluator

    model = build_model(_small(attn_scope="block_causal"), seed=5).eval()
    ev = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=False)))
    for i, st in enumerate((synthetic_state(13, 6, n_timelines=3), mirror(synthetic_state(14, 5, n_timelines=2)))):
        ev.start(st)
        got = ev([st])[0]
        want = NetworkEvaluator(model)([st])[0]
        assert got.board_logits is not None and got.board_logits.shape == (st.n_boards,)
        assert np.isfinite(got.board_logits).all() and np.allclose(got.board_logits, want.board_logits, atol=1e-4)
        check_evaluation(Leaf(i, st), got, 51)


def test_the_evaluator_delivers_every_boards_leave_logit():
    """``NetworkEvaluator`` hands the engine each position's leave logits (``Evaluation.board_logits``,
    one per board in the input's order - the expansion's), so the flat form never falls back on
    ``leave_share`` with the network; ``check_evaluation`` refuses a miscount or an infinite logit."""
    from dataclasses import replace

    import numpy as np

    from t5d.engine.evaluator import Leaf, NetworkEvaluator, check_evaluation

    model = build_model(_small(), seed=6).eval()
    states = [synthetic_state(s, n, n_timelines=3) for s, n in ((3, 4), (4, 9), (5, 1))]
    evs = NetworkEvaluator(model, max_batch=2)(states)
    out = model(collate(states))
    for i, (st, ev) in enumerate(zip(states, evs, strict=True)):
        assert ev.board_logits.shape == (st.n_boards,) and np.isfinite(ev.board_logits).all()
        assert np.allclose(ev.board_logits, out.leave_logits[i, : st.n_boards].numpy(), atol=1e-6)
        check_evaluation(Leaf(i, st), ev, 51)
    with pytest.raises(ValueError, match="board logits"):
        check_evaluation(Leaf(0, states[0]), replace(evs[0], board_logits=evs[0].board_logits[:-1]), 51)
    bad = evs[1].board_logits.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        check_evaluation(Leaf(1, states[1]), replace(evs[1], board_logits=bad), 51)
    nan = evs[1].board_logits.copy()
    nan[0] = np.nan
    check_evaluation(Leaf(1, states[1]), replace(evs[1], board_logits=nan), 51)  # NaN: none for that board


def test_an_unknown_leave_pool_is_refused():
    """``NetConfig.leave_pool`` is one of ``LEAVE_POOLS`` (issue 241(b)): anything else is refused."""
    from t5d.net.config import LEAVE_POOLS

    for pool in LEAVE_POOLS:
        NetConfig(leave_pool=pool).validate()
    with pytest.raises(ValueError, match="leave_pool"):
        NetConfig(leave_pool="globals").validate()
