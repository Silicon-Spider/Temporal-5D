"""Relative encoding (section 4): no absolute positions anywhere in the network.

* Storing the boards in a different order (timelines reordered) changes nothing but the
  order of per-board and per-square outputs.
* Translating every timeline by +k and every time by +2j changes nothing at all - the
  4D RoPE relative property, registers included (per-board registers rotate with their
  board; global registers are position-free, or anchored at the recentred origin) - and
  B15's displacement bias (the v1 default), which sees only displacements, and B17's
  key-side Shaw term (behind its flag), which dots the unrotated query with its pair's class.
* A position's outputs do not depend on what else is in its batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, permute_boards, synthetic_state, translate

FLAG_SETS = {
    "v1": dict(),
    "b5_offset": dict(rope_per_head=True, policy_promotion="offset"),
    "no_table": dict(displacement_bias=False),
    "registers_position_free": dict(board_register_spatial="position_free", aux_global_pool="global_registers"),
    "anchored_zero_angle": dict(global_register_pairs="anchored", board_register_spatial="zero_angle"),
    "b17_shaw": dict(shaw_key_term=True, shaw_init_std=1.0),
}

ATOL = 1e-5  # FP32 trunk: reordering changes only floating-point summation order


def _assert_same(a, b, atol=ATOL, board_perm=None, token_perm=None):
    for name in ("value_logits", "wdl_logits", "moves_left_log", "short_value_logits", "effort_logits"):
        x, y = getattr(a, name), getattr(b, name)
        assert torch.allclose(x, y, atol=atol), (name, float((x - y).abs().max()))
    for name, valid in (("policy_logits", a.move_valid), ("opp_reply_logits", a.opp_move_valid)):
        x, y = getattr(a, name)[valid], getattr(b, name)[valid]
        assert torch.allclose(x, y, atol=atol), (name, float((x - y).abs().max()))
    boards_a = a.board_logits[0] if board_perm is None else a.board_logits[0, board_perm]
    assert torch.allclose(boards_a, b.board_logits[0], atol=atol)
    sq_a = a.square_logits[0] if token_perm is None else a.square_logits[0, token_perm]
    mask = b.square_mask[0]
    assert torch.allclose(sq_a[mask], b.square_logits[0][mask], atol=atol)


def _token_perm(order: np.ndarray, n_boards: int) -> torch.Tensor:
    """Old token index of each token of the permuted position (66 per board, 16 global)."""
    new = [np.arange(66) + 66 * int(old) for old in order]
    return torch.from_numpy(np.concatenate(new + [66 * n_boards + np.arange(16)]))


@pytest.mark.parametrize("flags", list(FLAG_SETS), ids=list(FLAG_SETS))
def test_timeline_storage_order_does_not_matter(flags):
    model = build_model(NetConfig(trunk_dtype="fp32", **FLAG_SETS[flags]), seed=1).eval()
    state = synthetic_state(21, 24, n_timelines=6)
    timelines = np.unique(state.timeline)
    shuffled = np.random.default_rng(3).permutation(timelines)
    # Reorder whole timelines (and the boards within each, backwards for good measure).
    order = np.concatenate([np.flatnonzero(state.timeline == p)[::-1] for p in shuffled])
    assert not np.array_equal(order, np.arange(state.n_boards))
    a = model(collate([state]))
    b = model(collate([permute_boards(state, order)]))
    _assert_same(a, b, board_perm=torch.from_numpy(order), token_perm=_token_perm(order, state.n_boards))


#: Anchored global registers need recentring (validate() refuses raw coordinates).
TRANSLATION_CASES = [
    pytest.param(flags, recenter, id=f"{'recentred' if recenter else 'raw-coords'}-{flags}")
    for flags in FLAG_SETS
    for recenter in (False, True)
    if recenter or FLAG_SETS[flags].get("global_register_pairs") != "anchored"
]


@pytest.mark.parametrize(("flags", "recenter"), TRANSLATION_CASES)
def test_translation_invariance(flags, recenter):
    cfg = NetConfig(trunk_dtype="fp32", rope_recenter=recenter, **FLAG_SETS[flags])
    model = build_model(cfg, seed=2).eval()
    state = synthetic_state(22, 20)
    a = model(collate([state]))
    for dp, dh in ((7, 30), (-11, 400)):
        b = model(collate([translate(state, dp=dp, dh=dh)]))
        if recenter:
            # Recentring makes the translated input identical, bit for bit.
            assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.board_logits, b.board_logits)
        else:
            # Raw coordinates: the rotation itself must cancel the shift. FP32 angles of up
            # to ~400 rad carry ~3e-5 rad of rounding, yet the measured output change is
            # ~1.5e-6 at +400 half-turns (2.2e-6 at +1000), so the same bound holds.
            _assert_same(a, b)


def test_translation_invariance_default_config(model_default):
    """The shipped configuration (BF16 trunk, recentred coordinates): exact."""
    state = synthetic_state(23, 12)
    a = model_default(collate([state]))
    b = model_default(collate([translate(state, dp=5, dh=16)]))
    assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.policy_logits, b.policy_logits)


def test_batch_composition_does_not_matter(model_fp32):
    """Padding is masked: a position's outputs are the same alone and in any batch."""
    small, big, mid = synthetic_state(24, 3), synthetic_state(25, 11), synthetic_state(26, 6)
    alone = model_fp32(collate([small]))
    together = model_fp32(collate([big, small, mid]))
    assert torch.allclose(alone.value_logits[0], together.value_logits[1], atol=ATOL)
    m = alone.move_valid[0].sum()
    assert torch.allclose(alone.policy_logits[0, :m], together.policy_logits[1, :m], atol=ATOL)
    assert torch.allclose(alone.board_logits[0], together.board_logits[1, :3], atol=ATOL)
    nb = 3 * 66
    assert torch.allclose(alone.square_logits[0, :nb], together.square_logits[1, :nb], atol=ATOL)
