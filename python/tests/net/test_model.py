"""The dense v1 network end to end: shapes, determinism, non-trivial random init, heads."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, relayout, synthetic_state
from t5d.net.heads import (
    BOARD_HEADS,
    SQUARE_LABELS,
    board_mean_pool,
    categorical_mean,
    categorical_variance,
    effort_bin,
    effort_bin_lower_edges,
    effort_censored_prob,
    effort_quantile,
    moves_left_from_log,
    value_bin_centres,
    value_to_twohot,
)
from t5d.net.state import NO_PROMOTION, TOKEN_BOARD_REGISTER, PositionState


@pytest.mark.parametrize("n", [1, 2, 17, 60])
def test_output_shapes_default_config(model_default, n):
    """Every head's shape and dtype, on the default (BF16-trunk) configuration."""
    state = synthetic_state(100 + n, n)
    batch = collate([state])
    out = model_default(batch, return_trunk=True)
    t = 66 * n + 16
    assert batch.n_tokens == t
    assert out.trunk.shape == (1, t, 256)
    assert out.policy_logits.shape == (1, len(state.moves))
    assert out.opp_reply_logits.shape == (1, len(state.opp_moves))
    assert out.value_logits.shape == (1, 51) and out.short_value_logits.shape == (1, 51)
    assert out.wdl_logits.shape == (1, 3) and out.moves_left_log.shape == (1,)
    assert out.effort_logits.shape == (1, 16)
    assert out.board_logits.shape == (1, n, len(BOARD_HEADS))
    assert out.square_logits.shape == (1, t, len(SQUARE_LABELS))
    for name in (
        "policy_logits",
        "value_logits",
        "wdl_logits",
        "moves_left_log",
        "opp_reply_logits",
        "short_value_logits",
        "board_logits",
        "square_logits",
        "effort_logits",
        "trunk",
    ):
        tensor = getattr(out, name)
        assert tensor.dtype == torch.float32, f"{name}: heads are FP32 (G1)"
        assert torch.isfinite(tensor).all(), name


def test_padded_batch_shapes(model_fp32):
    states = [synthetic_state(1, 2), synthetic_state(2, 17), synthetic_state(3, 1)]
    batch = collate(states)
    out = model_fp32(batch)
    assert out.board_logits.shape == (3, 17, 3)
    assert out.policy_logits.shape == (3, max(len(s.moves) for s in states))
    for i, s in enumerate(states):
        assert torch.isinf(out.policy_logits[i, len(s.moves) :]).all()
        assert torch.isfinite(out.policy_logits[i, : len(s.moves)]).all()


def test_determinism():
    cfg = NetConfig(trunk_dtype="fp32")
    batch = collate([synthetic_state(5, 9), synthetic_state(6, 3)])
    a = build_model(cfg, seed=42)
    b = build_model(cfg, seed=42)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb and torch.equal(pa, pb), na
    oa, ob = a(batch), b(batch)
    for name in ("value_logits", "policy_logits", "board_logits", "square_logits", "opp_reply_logits", "wdl_logits"):
        assert torch.equal(getattr(oa, name), getattr(ob, name)), name
    c = build_model(cfg, seed=43)
    assert not torch.equal(c(batch).value_logits, oa.value_logits)
    # The global RNG is untouched by building a model (probed inside a fork, so that
    # this test's own reseeding does not leak into later tests either).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        x = torch.rand(3)
        torch.manual_seed(0)
        build_model(cfg, seed=1)
        assert torch.equal(torch.rand(3), x)


@pytest.mark.parametrize("flags", [dict(), dict(rope_per_head=True, binned_bias=True, policy_promotion="offset")], ids=["v1", "flags"])
def test_no_parameter_is_zero_or_constant(flags):
    """Section 15.2: random weights at realistic scale, never zeros or near-zeros (which
    make distinct paths agree trivially) - LayerNorm affines included."""
    model = build_model(NetConfig(**flags), seed=0)
    for name, p in model.named_parameters():
        assert p.abs().max() > 1e-3, f"{name} is (near) zero"
        if p.numel() > 1:
            assert p.std() > 1e-3, f"{name} is constant"
    gains = torch.cat([m.weight for m in model.modules() if isinstance(m, torch.nn.LayerNorm)])
    assert 0.5 < gains.mean() < 1.5 and gains.std() > 0.05, "LayerNorm gains near 1, not the identity"


def test_random_init_is_nontrivial(model_fp32):
    batch = collate([synthetic_state(8, 6)])
    out = model_fp32(batch, return_trunk=True)
    assert out.value_logits.std() > 0.05
    assert out.policy_logits[out.move_valid].std() > 0.05, "moves score differently"
    assert out.board_logits[0].std(dim=0).min() > 1e-3, "boards differ"
    trunk = out.trunk[0][batch.valid[0]]
    assert torch.pdist(trunk[:200]).min() > 1e-3, "no two tokens collapse to the same vector"
    probs = model_fp32.attention_probabilities(batch, layer=0)[0]  # (H, L, L)
    n = batch.n_tokens
    assert torch.allclose(probs.sum(-1), torch.ones(8, n), atol=1e-5)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
    assert entropy.mean() < math.log(n) - 0.1, "attention is not uniform"
    ratio = probs.max(-1).values / probs.min(-1).values.clamp_min(1e-30)
    assert ratio.median() > 3.0


def test_value_distribution_and_helpers(model_fp32):
    out = model_fp32(collate([synthetic_state(9, 4), synthetic_state(10, 7)]))
    probs = out.value_probs()
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)
    centres = value_bin_centres(51)
    assert centres[0] == -1 and centres[25] == 0 and centres[50] == 1
    mean = (probs * centres).sum(-1)
    assert torch.allclose(out.value_mean(), mean)
    assert torch.allclose(out.value_variance(), (probs * centres**2).sum(-1) - mean**2, atol=1e-6)
    assert torch.all(out.value_variance() >= 0)
    # Two-hot targets have the value as their mean, and sum to one.
    v = torch.tensor([-1.0, -0.37, 0.0, 0.5, 0.999, 1.0])
    target = value_to_twohot(v, 51)
    assert torch.allclose(target.sum(-1), torch.ones(6))
    assert torch.allclose(categorical_mean(target), v, atol=1e-6)
    assert torch.all(categorical_variance(target) < (2 / 50) ** 2)


@pytest.mark.parametrize("pool", ["squares", "global_registers"])
def test_value_and_global_aux_heads_read_their_pools(pool):
    """D3: the value head reads the mean of the global registers (sections 7.3, 8.3). The
    global aux heads read what ``aux_global_pool`` selects: every square token's mean (the
    placeholder default - section 12.4, B8, D7) or the global registers (section 9.2).
    Per-board registers feed neither."""
    model = build_model(NetConfig(trunk_dtype="fp32", aux_global_pool=pool), seed=0).eval()
    batch = collate([synthetic_state(16, 4), synthetic_state(17, 2)])
    out = model(batch, return_trunk=True)
    h = out.trunk
    sq = batch.square_mask
    registers = h[:, batch.body_len : batch.body_len + batch.global_regs].mean(1)
    squares = (h * sq[..., None]).sum(1) / sq.sum(1, keepdim=True)
    aux_in = squares if pool == "squares" else registers
    assert torch.allclose(out.value_logits, model.value(registers), atol=1e-5)
    assert torch.allclose(out.wdl_logits, model.aux.wdl(aux_in), atol=1e-5)
    assert torch.allclose(out.moves_left_log, model.aux.moves_left(aux_in).squeeze(-1), atol=1e-5)
    assert torch.allclose(out.short_value_logits, model.aux.short_value(aux_in), atol=1e-5)
    assert torch.allclose(out.effort_logits, model.aux.effort(aux_in), atol=1e-5)

    gen = torch.Generator().manual_seed(18)
    global_aux = ("wdl_logits", "moves_left_log", "short_value_logits", "effort_logits")

    def perturbed(rows: torch.Tensor):
        h2 = h.clone()
        h2[rows] += torch.randn(h2[rows].shape, generator=gen)
        return model.heads(h2, batch)

    every_square = perturbed(sq)
    assert torch.equal(every_square.value_logits, out.value_logits), "the value head ignores the squares"
    for name in global_aux:
        moved = not torch.allclose(getattr(every_square, name), getattr(out, name), atol=1e-4)
        assert moved == (pool == "squares"), name
    one_register = torch.zeros_like(sq)
    one_register[:, batch.body_len + 5] = True
    reg = perturbed(one_register)
    assert not torch.allclose(reg.value_logits, out.value_logits, atol=1e-4), "the value head reads the global registers"
    for name in global_aux:
        moved = not torch.allclose(getattr(reg, name), getattr(out, name), atol=1e-4)
        assert moved == (pool == "global_registers"), f"{name}: registers stay free under 'squares'"
    board_regs = perturbed(batch.kind == TOKEN_BOARD_REGISTER)
    for name in ("value_logits", "board_logits", *global_aux):
        assert torch.equal(getattr(board_regs, name), getattr(out, name)), f"{name} reads no per-board register"


def _with_promotions(state: PositionState) -> PositionState:
    """The state with four promotion variants of its first move appended."""
    src, dst = state.moves[0, :2]
    extra = np.array([[src, dst, promo] for promo in (2, 3, 4, 5)])
    return PositionState(**{**_fields(state), "moves": np.concatenate([state.moves, extra])})


@pytest.mark.parametrize("promotion", ["none", "offset"])
def test_pointer_heads_score_each_moves_own_endpoints(promotion):
    """D1, section 8.1: score(move) = f(h_src)^T g(h_dst) / sqrt(dim), read at the move's
    own source and destination tokens - for the policy and the opponent-reply head."""
    model = build_model(NetConfig(trunk_dtype="fp32", policy_promotion=promotion), seed=0).eval()
    batch = collate([_with_promotions(synthetic_state(13, 5)), synthetic_state(14, 2)])
    out = model(batch, return_trunk=True)
    h = out.trunk
    heads = (
        (model.policy, batch.moves, batch.move_valid, out.policy_logits),
        (model.aux.opp_reply, batch.opp_moves, batch.opp_move_valid, out.opp_reply_logits),
    )
    for head, moves, valid, logits in heads:
        for b in range(batch.batch_size):
            src, dst, promo = moves[b][valid[b]].unbind(-1)
            expected = (head.src(h[b, src]) * head.dst(h[b, dst])).sum(-1) / math.sqrt(head.dim)
            if promotion == "offset":
                offset = head.promo(h[b, dst]).gather(-1, promo[:, None]).squeeze(-1)
                expected = expected + torch.where(promo != NO_PROMOTION, offset, torch.zeros_like(offset))
            assert torch.allclose(logits[b][valid[b]], expected, atol=1e-5), (head, b)


def test_pointer_score_roles_are_asymmetric(model_fp32):
    """Section 8.1: f and g are separate - a -> b and b -> a score differently, and so do
    two moves into the same square from different sources."""
    state = synthetic_state(15, 3)
    a, b, c = (int(t) for t in state.moves[:3, 0])
    d = int(state.moves[0, 1])
    moves = np.array([[a, d, 0], [d, a, 0], [b, d, 0], [c, d, 0], [a, b, 0]])
    s = PositionState(**{**_fields(state), "moves": moves})
    logits = model_fp32(collate([s])).policy_logits[0]
    # A head that ignored a role would tie these exactly; random scores never come this close.
    assert (logits[0] - logits[1]).abs() > 1e-5, "a -> d against d -> a"
    assert torch.pdist(logits[[0, 2, 3], None]).min() > 1e-5, "same destination, different sources"
    assert (logits[0] - logits[4]).abs() > 1e-5, "same source, different destinations"


def test_policy_scores_only_supplied_moves_in_any_order(model_fp32):
    state = synthetic_state(11, 8)
    base = model_fp32(collate([state])).policy_logits[0]
    order = np.random.default_rng(0).permutation(len(state.moves))
    shuffled = PositionState(**{**_fields(state), "moves": state.moves[order]})
    perm = model_fp32(collate([shuffled])).policy_logits[0]
    assert torch.allclose(perm, base[torch.from_numpy(order)], atol=1e-6), "scores permute with the list"
    subset = PositionState(**{**_fields(state), "moves": state.moves[:5]})
    sub = model_fp32(collate([subset])).policy_logits[0]
    assert sub.shape == (5,) and torch.allclose(sub, base[:5], atol=1e-6), "a move's score ignores the others"
    empty = PositionState(**{**_fields(state), "moves": np.zeros((0, 3), dtype=np.int64)})
    assert model_fp32(collate([empty])).policy_logits.shape == (1, 0)


def test_policy_log_probs(model_fp32):
    """Log-softmax over each position's supplied moves; padding gets -inf."""
    out = model_fp32(collate([synthetic_state(1, 2), synthetic_state(2, 5)]))
    logp = out.policy_log_probs()
    for b in range(2):
        valid = out.move_valid[b]
        assert torch.allclose(logp[b][valid].exp().sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(logp[b][valid], torch.log_softmax(out.policy_logits[b][valid], dim=-1), atol=1e-6)
        assert torch.all(logp[b][~valid] == float("-inf"))


def test_policy_head_rejects_register_endpoints(model_fp32):
    batch = collate([synthetic_state(12, 3)])
    batch.moves[0, 0, 1] = 64  # board 0's first register
    with pytest.raises(ValueError, match="register"):
        model_fp32(batch)


def test_promotion_handling_is_a_parameter():
    state = synthetic_state(13, 2)
    pawn_dst = state.moves[0, 1]
    moves = np.array([[state.moves[0, 0], pawn_dst, promo] for promo in (2, 3, 4, 5)])
    s = PositionState(**{**_fields(state), "moves": moves})
    batch = collate([s])
    tie = build_model(NetConfig(trunk_dtype="fp32", policy_promotion="none"), seed=0)(batch).policy_logits[0]
    assert torch.allclose(tie, tie[0].expand(4)), "the literal section 8.1 score ties promotion variants"
    split = build_model(NetConfig(trunk_dtype="fp32", policy_promotion="offset"), seed=0)(batch).policy_logits[0]
    assert split.unique().numel() == 4


def test_board_pool_excludes_registers():
    batch = collate([synthetic_state(14, 3)])
    h = torch.randn(1, batch.n_tokens, 4, generator=torch.Generator().manual_seed(14))
    pooled = board_mean_pool(h, batch)
    for b in range(3):
        squares = h[0, b * 66 : b * 66 + 64]
        assert torch.allclose(pooled[0, b], squares.mean(0), atol=1e-6)
    h2 = h.clone()
    h2[0, 64:66] += 100.0  # board 0's registers
    h2[0, batch.body_len :] += 100.0  # global registers
    assert torch.allclose(board_mean_pool(h2, batch), pooled)


def test_register_ablation_arms_run():
    """Ablation 3's arms: no registers, and four per board (section 13.1)."""
    state = synthetic_state(15, 4)
    for r, g in ((0, 0), (4, 16), (2, 0)):
        cfg = NetConfig(trunk_dtype="fp32", regs_per_board=r, global_regs=g)
        out = build_model(cfg, seed=0)(collate([relayout(state, r, g)]))
        assert torch.isfinite(out.value_logits).all() and out.board_logits.shape == (1, 4, 3)


def test_effort_helpers():
    visits = torch.tensor([1, 2, 3, 1000, 100_000, 50])
    converged = torch.tensor([True, True, True, True, True, False])
    assert effort_bin(visits, converged, 16).tolist() == [0, 1, 1, 9, 14, 15]
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.5, 0.5]])
    assert effort_quantile(probs, 0.5).tolist() == [2, 2]
    assert effort_quantile(probs, 0.9).tolist() == [3, 3]
    assert effort_censored_prob(probs).tolist() == pytest.approx([0.4, 0.5])
    # Each bin's lower visit edge lands in that bin; the censored bin has none.
    edges = effort_bin_lower_edges(16)
    assert len(edges) == 16 and edges[-1] == math.inf
    lower = torch.tensor(edges[:-1]).long()
    assert effort_bin(lower, torch.ones(15, dtype=torch.bool), 16).tolist() == list(range(15))
    assert effort_bin(lower[1:] - 1, torch.ones(14, dtype=torch.bool), 16).tolist() == list(range(14))


def test_moves_left_from_log_inverts_the_target_transform():
    board_moves = torch.tensor([0.0, 1.0, 7.0, 250.0])
    assert torch.allclose(moves_left_from_log(torch.log1p(board_moves)), board_moves, rtol=1e-5)
    assert moves_left_from_log(torch.tensor([-0.4])).item() == 0.0, "never negative"


def test_later_stage_flags_raise():
    # B9-B12 are built from stage N2 on (tests/net/test_structure.py) and the block-causal
    # scope from stage N3 (attn_scope, tests/net/test_scope.py); these are not.
    flags = (
        dict(piece_centric=True),
        dict(history_window=True),
        dict(fp8_attention=True),
    )
    for flag in flags:
        with pytest.raises(NotImplementedError):
            build_model(NetConfig(**flag))
    build_model(NetConfig(attn_scope="block_causal", n_layers=1))


def test_open_choices_reject_unknown_values():
    fields = ("board_register_spatial", "global_register_pairs", "aux_global_pool", "policy_promotion")
    fields += ("attn_scope", "causal_past", "frozen_features", "supersession_scope")
    for field in fields:
        with pytest.raises(ValueError, match=field):
            NetConfig(**{field: "nonsense"}).validate()


def _fields(state: PositionState) -> dict:
    names = (
        "width",
        "height",
        "timeline",
        "half_turn",
        "is_head",
        "active",
        "side_to_move",
        "piece",
        "colour",
        "moved",
        "edges",
        "moves",
        "opp_moves",
        "regs_per_board",
        "global_regs",
    )
    return {n: getattr(state, n) for n in names}
