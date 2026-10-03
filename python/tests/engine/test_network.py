"""Stage N4: the network (random weights) on real engine positions, dense and cached.

The committed F1 network (BF16 trunk, 16 layers) and a small FP32 configuration run on
positions the engine produced: starts of several variants (board sizes 3x3 to 8x8, odd and
even), the records' positions and random play with branched and inactive timelines. The
policy logits cover exactly each position's candidate moves - the engine's move list - and
nothing else; positions batch together; the cached path equals one full block-causal pass
on real positions (``net/cache.py``'s reference), at the tolerances of ``tests/net/test_cache.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.engine import Game
from t5d.engine.play import random_position_game
from t5d.net import NetConfig, build_model, collate
from t5d.net.cache import CachedEvaluator, reference_output
from tests.engine.helpers import RECORDS, requires_engine

pytestmark = requires_engine


def _record_positions(key: str) -> list:
    with Game.from_pgn(RECORDS[key]) as g:
        out = [g.position()]
        out.extend(g.position() for _ in g.replay())
    return out


@pytest.fixture(scope="module")
def mixed_positions():
    """Real positions of several sizes and shapes."""
    out = []
    for name in ("Standard", "Standard - Two Timelines", "Very Small - Open", "Focused - Just Kings", "Misc - Timeline Battleground"):
        with Game.variant(name) as g:
            out.append(g.position())
    out.append(_record_positions("branching")[-1])
    out.append(_record_positions("cross_timeline")[-1])
    for variant, boards in (("Standard", 12), ("Very Small - Open", 20)):
        with random_position_game(variant, seed=3, boards=boards) as g:
            out.append(g.position())
    return out


def _covers_exactly_the_moves(out, positions):
    for i, pos in enumerate(positions):
        m = pos.move_count
        assert int(out.move_valid[i].sum()) == m and bool(out.move_valid[i, :m].all())
        logits = out.policy_logits[i]
        assert torch.isfinite(logits[:m]).all()
        assert torch.isinf(logits[m:]).all() and (logits[m:] < 0).all()
        assert torch.isfinite(out.value_logits[i]).all() and out.value_logits.shape[1] == 51


def test_f1_runs_on_real_positions(model_default, mixed_positions):
    """The default configuration exactly as shipped (F1, BF16 trunk, dense) on each position."""
    for pos in mixed_positions[:7]:
        out = model_default(collate([pos.state]))
        _covers_exactly_the_moves(out, [pos])
        probs = out.value_probs()[0]
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_a_small_network_batches_mixed_real_positions(mixed_positions):
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=5).eval()
    batch = collate([p.state for p in mixed_positions])
    out = model(batch)
    _covers_exactly_the_moves(out, mixed_positions)
    for i, pos in enumerate(mixed_positions):  # a position's evaluation does not depend on its batch
        alone = model(collate([pos.state]))
        m = pos.move_count
        assert torch.allclose(out.policy_logits[i, :m], alone.policy_logits[0, :m], atol=1e-4)
        assert torch.allclose(out.value_logits[i], alone.value_logits[0], atol=1e-4)


def test_every_switch_on_real_positions(mixed_positions):
    """B5, B6, B10-B12 on, with the typed relations the engine builds."""
    flags = dict(rope_per_head=True, binned_bias=True, routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, **flags), seed=2).eval()
    from t5d.engine import relations_for

    rel = relations_for(model.cfg)
    assert rel == ("attack", "defence", "xray")
    with Game.from_pgn(RECORDS["branching"]) as g:
        g.replay_all()
        pos = g.position(relations=rel)
    assert (pos.state.edges[:, 2] > 0).any()
    out = model(collate([pos.state]))
    _covers_exactly_the_moves(out, [pos])


def _close(a, b, atol_trunk, atol):
    assert torch.allclose(a.trunk, b.trunk, atol=atol_trunk), float((a.trunk - b.trunk).abs().max())
    assert torch.allclose(a.value_logits, b.value_logits, atol=atol)
    v = a.move_valid
    assert torch.allclose(a.policy_logits[v], b.policy_logits[v], atol=atol)


@pytest.mark.parametrize("seed", [0, 1])
def test_cached_equals_one_full_pass_on_a_real_game(seed):
    """Positions along a real game, evaluated incrementally (one causal epoch, the store shared
    across them) against one full block-causal pass each - FP32, 2 layers."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal"), seed=seed).eval()
    cached = CachedEvaluator(model)
    positions = _record_positions("branching")
    epoch = cached.root_epoch(positions[0].state)
    for pos in positions:
        out = cached.evaluate(pos.state, epoch)
        _close(out, reference_output(model, pos.state), atol_trunk=2e-5, atol=1e-4)
        _covers_exactly_the_moves(out, [pos])
    assert cached.store.hits > 0  # later positions reuse earlier positions' frozen boards


def test_cached_equals_one_full_pass_on_random_play():
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal"), seed=4).eval()
    cached = CachedEvaluator(model)
    with random_position_game("Standard - Two Timelines", seed=7, boards=24) as g:
        root = g.position()
        epoch = cached.root_epoch(root.state)
        for move in range(min(3, root.move_count)):
            child = g.copy()
            child.expand(square_features="none")
            child.apply(move)
            if not child.info().obligations_met:
                pos = child.position()  # mid-action, before the shift
                _close(cached.evaluate(pos.state, epoch), reference_output(model, pos.state), atol_trunk=2e-5, atol=1e-4)
            child.close()
        _close(cached.evaluate(root.state, epoch), reference_output(model, root.state), atol_trunk=2e-5, atol=1e-4)


def test_f1_cached_on_a_real_position():
    """F1 as shipped but block-causal (BF16 trunk): cached against the BF16 full pass, at the
    BF16 bounds of tests/net/test_cache.py - about 4x the gaps measured here on these positions
    (3 seeds): trunk RMS 2.0e-3 / max 1.1e-2, policy 6.6e-3, value 1.2e-3."""
    model = build_model(NetConfig(attn_scope="block_causal"), seed=0).eval()
    cached = CachedEvaluator(model)
    positions = _record_positions("cross_timeline")
    epoch = cached.root_epoch(positions[2].state)
    for pos in positions[2:]:
        out = cached.evaluate(pos.state, epoch)
        ref = reference_output(model, pos.state)
        err = (out.trunk - ref.trunk).float()
        assert float(err.pow(2).mean().sqrt()) < 8e-3 and float(err.abs().max()) < 5e-2
        v = out.move_valid
        assert float((out.policy_logits[v] - ref.policy_logits[v]).abs().max()) < 3e-2
        assert float((out.value_logits - ref.value_logits).abs().max()) < 5e-3
        _covers_exactly_the_moves(out, [pos])


def test_a_position_with_no_moves_evaluates():
    """After Standard's one present board is played no further move exists, and the present
    has shifted: the action's only candidate is the submit action (D8), probability one."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1), seed=0).eval()
    with Game.variant("Standard") as g:
        g.apply(g.moves()[0])
        pos = g.position()
    assert pos.move_count == 0 and pos.shifted and pos.submit_possible and pos.submit_legal
    out = model(collate([pos.state]))
    assert out.policy_logits.shape == (1, 0) and torch.isfinite(out.value_logits).all()
    np.testing.assert_allclose(out.value_probs().sum().item(), 1.0, atol=1e-5)
    assert out.submit_log_prob().item() == pytest.approx(0.0, abs=1e-6)


def _shift_pair(variant: str, seed: int):
    """Seeded random play from ``variant`` to the first state whose present has shifted inside
    the action, and the same game submitted: (the shifted state's position, the submitted one's)."""
    from t5d.engine.play import ROYAL_IDS

    rng = np.random.default_rng(seed)
    g = Game.variant(variant)
    g.set_legality(False)
    try:
        for _ in range(200):
            if g.info().obligations_met:
                before = g.position()
                g.submit()
                return before, g.position()
            x = g.expand(square_features="none")
            g.apply(int(rng.choice(np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS)))))
        raise AssertionError("no shift met")
    finally:
        g.close()


@pytest.mark.parametrize("variant", ["Standard", "Standard - Two Timelines"])
def test_the_acting_player_alone_tells_a_shifted_action_from_the_opponents_turn(variant):
    """Register D8 on real positions: an action whose present has shifted and the opponent's
    turn start after its submission hold the same boards; the network's input differs in the
    acting player alone - on the global registers under the default arm - and the network's
    outputs differ."""
    shifted_pos, submitted = _shift_pair(variant, seed=11)
    a, b = shifted_pos.state, submitted.state
    assert a.shifted and not b.shifted and a.acting_player != b.acting_player
    assert np.array_equal(a.piece, b.piece) and np.array_equal(a.half_turn, b.half_turn) and np.array_equal(a.is_head, b.is_head)
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=1).eval()
    ba, bb = collate([a]), collate([b])
    differs = (model.embed_tokens(ba) - model.embed_tokens(bb)).abs().amax(dim=-1)[0] > 0
    from t5d.net.state import TOKEN_GLOBAL_REGISTER

    assert torch.equal(differs, ba.kind[0] == TOKEN_GLOBAL_REGISTER)
    oa, ob = model(ba), model(bb)
    assert not torch.allclose(oa.value_logits, ob.value_logits)
    assert bool(oa.submit_possible[0]) and not bool(ob.submit_possible[0])
