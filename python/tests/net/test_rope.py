"""4D RoPE (B4), per-head specialisation (B5), register positions; smolgen's log buckets (re-homed from B6)."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state, translate
from t5d.net.attention import AttnContext, attention_explicit, attention_sdpa
from t5d.net.buckets import log_bucket, n_log_buckets, n_signed_log_buckets, signed_log_bucket
from t5d.net.rope import (
    AXIS_FILE,
    AXIS_RANK,
    AXIS_TIME,
    AXIS_TIMELINE,
    apply_rope,
    axis_frequencies,
    rope_angles,
    rope_cos_sin,
    rope_tables,
)
from t5d.net.state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER


def test_frequency_ladders():
    file = axis_frequencies(3, 8.0)
    assert file == pytest.approx([1.0, 0.6267, 0.3927], abs=1e-4)
    time = axis_frequencies(6, 256.0)
    assert time[0] == 1.0 and time[-1] == pytest.approx(math.pi / 256)
    assert all(a > b for a, b in itertools.pairwise(time))
    # Geometric (standard RoPE form base^(-k/n)): constant ratio between rungs.
    ratios = [b / a for a, b in itertools.pairwise(time)]
    assert max(ratios) - min(ratios) < 1e-9


def test_default_allocation_is_the_b4_split():
    axis, freq = rope_tables(NetConfig())
    assert axis.shape == (1, 16) and freq.dtype == torch.float32
    counts = [int((axis == a).sum()) for a in range(4)]
    assert counts == [3, 3, 4, 6]  # file, rank, timeline, time


def test_per_head_allocation_b5():
    cfg = NetConfig(rope_per_head=True)
    axis, freq = rope_tables(cfg)
    assert axis.shape == (8, 16)
    roles = cfg.rope_head_roles
    for h, role in enumerate(roles):
        a, f = axis[h], freq[h]
        if role == "same_board":
            assert set(a.tolist()) == {AXIS_FILE, AXIS_RANK} and int((a == AXIS_FILE).sum()) == 8
        elif role == "same_square":
            assert set(a.tolist()) == {AXIS_TIMELINE, AXIS_TIME}
            assert [int((a == AXIS_TIMELINE).sum()), int((a == AXIS_TIME).sum())] == list(cfg.rope_same_square_split)
        elif role == "move_graph":
            ranges = torch.tensor(cfg.rope_ranges)[a]
            assert torch.all(f * ranges <= cfg.rope_near_nope_max_angle + 1e-6), "near-NoPE: small angles over the range"
        else:  # dense, and relations under the placeholder default
            assert [int((a == x).sum()) for x in range(4)] == [3, 3, 4, 6]
    near = rope_tables(NetConfig(rope_per_head=True, rope_relations_head="near_nope"))
    rel = roles.index("relations")
    assert torch.equal(near[1][rel], near[1][roles.index("move_graph")])


def test_rotation_is_relative():
    """q_i . k_j after rotation depends on coordinates only through their difference."""
    gen = torch.Generator().manual_seed(0)
    axis, freq = rope_tables(NetConfig())
    q = torch.randn(1, 1, 50, 32, generator=gen)
    k = torch.randn(1, 1, 50, 32, generator=gen)
    coords = torch.stack(
        [
            torch.randint(0, 8, (50,), generator=gen).float(),
            torch.randint(0, 8, (50,), generator=gen).float(),
            torch.randint(-10, 10, (50,), generator=gen).float(),
            torch.randint(0, 80, (50,), generator=gen).float(),
        ],
        dim=-1,
    )[None]

    def scores(c):
        cos, sin = rope_cos_sin(rope_angles(c, axis, freq))
        return apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).transpose(-1, -2)

    shift = torch.tensor([0.0, 0.0, 13.0, 300.0])
    assert torch.allclose(scores(coords), scores(coords + shift), atol=1e-4)
    # And it does depend on the differences.
    other = coords.clone()
    other[0, 0, 3] += 1
    assert not torch.allclose(scores(coords), scores(other), atol=1e-3)


def _context(batch, model, coords):
    cfg = model.cfg
    cos, sin = rope_cos_sin(rope_angles(coords, model.rope_axis, model.rope_freq))
    spatial_free = batch.valid & (batch.kind == TOKEN_BOARD_REGISTER) if cfg.board_register_spatial == "position_free" else None
    return AttnContext(
        cos=cos,
        sin=sin,
        body_len=batch.body_len,
        key_valid=batch.valid,
        global_free=cfg.global_register_pairs == "position_free",
        spatial_free=spatial_free,
        spatial_dims=model.rope_spatial,
    )


def _qkv(batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(batch.batch_size, 8, batch.n_tokens, 32, generator=gen) for _ in range(3)]


@pytest.mark.parametrize("impl", ["sdpa", "explicit"])
def test_global_registers_are_position_free(impl):
    """Under the default ``global_register_pairs="position_free"`` pairs with a global
    register use unrotated Q and K on both sides: whatever angle a global register is
    given, and wherever the boards sit, its attention is unchanged."""
    model = build_model(NetConfig(trunk_dtype="fp32"), seed=0)
    batch = collate([synthetic_state(0, 6)])
    q, k, v = _qkv(batch, 1)
    gen = torch.Generator().manual_seed(2)
    attend = attention_sdpa if impl == "sdpa" else attention_explicit
    coords = model.coordinates(batch)
    base = attend(q, k, v, _context(batch, model, coords))
    scrambled = coords.clone()
    scrambled[0, batch.body_len :] = torch.randn(16, 4, generator=gen) * 50
    assert torch.allclose(base, attend(q, k, v, _context(batch, model, scrambled)), atol=1e-5)
    shifted = coords.clone()
    shifted[0, : batch.body_len, 2] += 9
    shifted[0, : batch.body_len, 3] += 44
    g = slice(batch.body_len, None)
    assert torch.allclose(base[:, :, g], attend(q, k, v, _context(batch, model, shifted))[:, :, g], atol=1e-4)


def test_anchored_global_registers_need_recentring():
    """A global register unrotated on its own side sits at coordinate 0 of the frame the
    board tokens are rotated in. With recentred coordinates that is the anchor (a1, p_ref,
    h_present), which moves with the position, so translating the position changes
    nothing. With raw coordinates it would be the absolute origin - an absolute positional
    encoding (section 14.1) - so the config refuses that combination."""
    with pytest.raises(ValueError, match="rope_recenter"):
        NetConfig(global_register_pairs="anchored", rope_recenter=False).validate()
    model = build_model(NetConfig(trunk_dtype="fp32", global_register_pairs="anchored"), seed=0).eval()
    state = synthetic_state(5, 8)
    moved = translate(state, dp=4, dh=12)
    a, b = model(collate([state])), model(collate([moved]))
    assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.wdl_logits, b.wdl_logits)
    # The refused combination: the same attention in the raw frame leaks the translation
    # into the global registers' rows.
    batch, batch_moved = collate([state]), collate([moved])
    q, k, v = _qkv(batch, 3)

    def raw_frame(bt):
        coords = bt.coords.float().clone()
        coords[bt.kind == TOKEN_GLOBAL_REGISTER] = 0.0  # the absolute origin
        return attention_explicit(q, k, v, _context(bt, model, coords))

    g = slice(batch.body_len, None)
    assert not torch.allclose(raw_frame(batch)[:, :, g], raw_frame(batch_moved)[:, :, g], atol=1e-3)


@pytest.mark.parametrize("impl", ["sdpa", "explicit"])
def test_anchored_global_registers_see_board_positions(impl):
    """Under ``"anchored"`` a global register's attention depends on where the boards sit
    relative to the anchor (unlike ``"position_free"``)."""
    model = build_model(NetConfig(trunk_dtype="fp32", global_register_pairs="anchored"), seed=0)
    batch = collate([synthetic_state(0, 6)])
    q, k, v = _qkv(batch, 1)
    attend = attention_sdpa if impl == "sdpa" else attention_explicit
    coords = model.coordinates(batch)
    assert torch.all(coords[batch.kind == TOKEN_GLOBAL_REGISTER] == 0), "the anchor: recentred coordinate 0"
    shifted = coords.clone()
    shifted[0, : batch.body_len, 3] += 7
    g = slice(batch.body_len, None)
    base = attend(q, k, v, _context(batch, model, coords))
    assert not torch.allclose(base[:, :, g], attend(q, k, v, _context(batch, model, shifted))[:, :, g], atol=1e-3)


def test_register_spatial_modes():
    batch = collate([synthetic_state(1, 3)])
    breg = batch.kind == TOKEN_BOARD_REGISTER
    # Placeholder default: a learned off-lattice coordinate per register slot.
    learned = build_model(NetConfig(trunk_dtype="fp32"), seed=0)
    assert learned.cfg.board_register_spatial == "learned_angle"
    c = learned.coordinates(batch)
    slots = batch.reg_index[breg]
    assert torch.allclose(c[breg][:, :2], learned.register_xy[slots])
    assert torch.all(c[breg][:, 2] == batch.coords[breg][:, 2] - batch.coord_ref[0, 0]), "the board's timeline"
    assert torch.all(c[breg][:, 3] == batch.coords[breg][:, 3] - batch.coord_ref[0, 1]), "the board's time"
    # Off-lattice at initialisation: never exactly on a square's coordinate.
    assert torch.all((learned.register_xy - learned.register_xy.round()).abs() > 0.05)
    # zero_angle: the register rotates exactly as square a1 of its board does - the
    # collision with the movement geometry section 5.3 warns of (kept as an ablation arm).
    zero = build_model(NetConfig(trunk_dtype="fp32", board_register_spatial="zero_angle"), seed=0)
    cos, sin = rope_cos_sin(rope_angles(zero.coordinates(batch), zero.rope_axis, zero.rope_freq))
    reg, a1 = batch.body_len - 66 + 64, batch.body_len - 66  # board 2's first register, board 2's a1
    assert torch.equal(cos[0, :, reg], cos[0, :, a1]) and torch.equal(sin[0, :, reg], sin[0, :, a1])


@pytest.mark.parametrize("impl", ["sdpa", "explicit"])
def test_position_free_registers_have_no_file_or_rank(impl):
    """``board_register_spatial="position_free"``: a register's whole attention row is
    blind to every token's file and rank (its pairs use unrotated file/rank dims), yet
    sees timeline and time. Under ``learned_angle`` the same row is not blind."""
    batch = collate([synthetic_state(2, 4)])
    q, k, v = _qkv(batch, 4)
    attend = attention_sdpa if impl == "sdpa" else attention_explicit
    gen = torch.Generator().manual_seed(5)
    regs = (batch.kind == TOKEN_BOARD_REGISTER)[0]
    for spatial, blind in (("position_free", True), ("learned_angle", False)):
        model = build_model(NetConfig(trunk_dtype="fp32", board_register_spatial=spatial), seed=0)
        coords = model.coordinates(batch)
        base = attend(q, k, v, _context(batch, model, coords))[:, :, regs]
        moved_xy = coords.clone()
        moved_xy[..., :2] = torch.randint(0, 8, moved_xy[..., :2].shape, generator=gen).float()
        out_xy = attend(q, k, v, _context(batch, model, moved_xy))[:, :, regs]
        assert torch.allclose(base, out_xy, atol=1e-5) == blind, spatial
        moved_t = coords.clone()
        moved_t[0, :66, 3] += 5  # board 0 moves in time: the registers see it
        assert not torch.allclose(base, attend(q, k, v, _context(batch, model, moved_t))[:, :, regs], atol=1e-3)


def test_log_buckets_clamp():
    d = torch.tensor([0, 1, 3, 4, 7, 8, 127, 128, 129, 10_000])
    assert log_bucket(d, 4, 128).tolist() == [0, 1, 3, 4, 4, 5, 8, 9, 9, 9]
    assert log_bucket(-d, 4, 128).tolist() == log_bucket(d, 4, 128).tolist()


def test_signed_buckets():
    """Smolgen's signed Delta-time buckets (B12's relative cells): the two signs share bucket 0
    and nothing else, the last bucket of each sign is clamped."""
    d = torch.tensor([0, 1, 3, 4, 7, 8, 127, 128, 129, 10_000])
    pos, neg = signed_log_bucket(d, 4, 128), signed_log_bucket(-d, 4, 128)
    n = n_log_buckets(4, 128)
    assert n == 10 and n_signed_log_buckets(4, 128) == 19
    assert pos.tolist() == log_bucket(d, 4, 128).tolist()
    assert neg[0] == 0 and torch.equal(neg[1:], pos[1:] + n - 1)
    assert int(neg.max()) == 18 and int(pos.max()) == 9


def test_b6_is_gone():
    """B6 is superseded by B15 (CHECKPOINT 6, register B6) and removed with its switches - no
    arm survives (ablation 2's arm is B15 on or off)."""
    for name in ("binned_bias", "binned_bias_rope", "binned_bias_near_time", "bias_time_exact"):
        with pytest.raises(TypeError):
            NetConfig(**{name: True})
