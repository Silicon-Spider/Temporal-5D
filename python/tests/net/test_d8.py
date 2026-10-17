"""Stage N5: register D8 in the network - the acting-player input and the submit action.

Turns have variable length (rulebook R8): once the present has shifted inside an action, the
boards' present names the opponent while the action is still the mover's. D8 gives the
network a global acting-player input and the submit action its own policy logit, read from a
global register, offered where submitting is possible and masked unless it is legal; the
policy is one distribution over the moves and submit.

Checked here on synthetic states (no engine): where the input enters the encoding, for each
arm of ``NetConfig.acting_player_input`` and each scope; that it reaches the outputs; the
submit logit's three cases (absent, masked, live) and both readouts; the joint
normalisation; section 15.2's dense-vs-cached equivalence with the new input (C2, C17, C19,
C20) under both arms, both ``frozen_features`` readings and a root refresh; the parameters and
their draw order. Real engine positions before and after the shift: ``tests/engine``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, permute_boards, synthetic_state, translate
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output
from t5d.net.heads import submit_register_state
from t5d.net.state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_PAD, TOKEN_SQUARE


def shifted(state, legal: bool | None = True):
    """The same boards read as an action whose present has shifted: the acting player is the
    present's opponent; submitting possible (``legal`` None: not offered at all)."""
    return replace(
        state,
        acting_player=1 - state.player_to_move,
        submit_possible=legal is not None,
        submit_legal=bool(legal),
    )


def _small(seed: int = 0, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=2, **flags), seed=seed).eval()


# --- the contract ------------------------------------------------------------------------------


def test_the_acting_player_defaults_to_the_present():
    s = synthetic_state(3, 7)
    assert s.acting_player == s.player_to_move and not s.shifted
    assert (s.submit_possible, s.submit_legal) == (False, False)
    t = shifted(s)
    assert t.acting_player == 1 - s.player_to_move and t.shifted and t.submit_possible and t.submit_legal


def test_the_action_is_validated():
    s = synthetic_state(4, 5)
    with pytest.raises(ValueError, match="submit_legal needs submit_possible"):
        replace(s, acting_player=1 - s.player_to_move, submit_legal=True)
    with pytest.raises(ValueError, match="present shifted"):
        replace(s, submit_possible=True)  # T5: no submission before the present has shifted
    with pytest.raises(ValueError, match="acting_player must be"):
        replace(s, acting_player=2)
    replace(s, acting_player=1 - s.player_to_move)  # shifted with submit not offered (whole-turn): allowed


def test_board_transforms_keep_the_action():
    s = shifted(synthetic_state(5, 6), legal=False)
    for t in (permute_boards(s, list(reversed(range(s.n_boards)))), translate(s, dp=3, dh=4)):
        assert (t.acting_player, t.submit_possible, t.submit_legal) == (s.acting_player, True, False)


def test_a_batch_carries_the_action():
    a, b = synthetic_state(6, 3), shifted(synthetic_state(7, 5), legal=False)
    batch = collate([a, b])
    assert batch.acting_player.tolist() == [a.player_to_move, 1 - b.player_to_move]
    assert batch.submit_possible.tolist() == [False, True] and batch.submit_legal.tolist() == [False, False]
    bare = replace(batch, acting_player=None, submit_possible=None, submit_legal=None)
    assert bare.acting().tolist() == [a.player_to_move, b.player_to_move]  # the present's colour
    assert [m.tolist() for m in bare.submit_mask()] == [[False, False], [False, False]]


# --- where the input enters ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm,scope,features",
    [
        ("global_registers", "dense", "intrinsic"),
        ("global_registers", "block_causal", "intrinsic"),
        ("all_tokens", "dense", "intrinsic"),
        ("all_tokens", "block_causal", "intrinsic"),
        ("all_tokens", "block_causal", "current"),
    ],
)
def test_the_acting_player_changes_the_encoding_exactly_where_it_should(arm, scope, features):
    """Two readings of the same boards that differ only in the acting player: the token
    embeddings differ on the global registers (the default arm) - or on every token of every
    board too (``all_tokens``), a frozen board's excepted under the block-causal scope with
    ``frozen_features="intrinsic"`` - and nowhere else: padding never."""
    model = _small(1, acting_player_input=arm, frozen_features=features)
    a, b = synthetic_state(11, 9, n_timelines=3), synthetic_state(12, 4)
    before = collate([a, b])
    after = collate([shifted(a), b])
    ea, eb = model.embed_tokens(before, scope), model.embed_tokens(after, scope)
    differs = (ea - eb).abs().amax(dim=-1) > 0
    kind = before.kind
    expected = torch.zeros_like(differs)
    expected[0] = kind[0] == TOKEN_GLOBAL_REGISTER
    if arm == "all_tokens":
        on_board = (kind[0] == TOKEN_SQUARE) | (kind[0] == TOKEN_BOARD_REGISTER)
        if scope == "block_causal" and features == "intrinsic":
            on_board &= before.is_head[0] == 1
        expected[0] |= on_board
    assert torch.equal(differs, expected)
    assert not differs[0][kind[0] == TOKEN_PAD].any() and not differs[1].any()
    if arm == "global_registers":
        assert int(differs[0].sum()) == a.global_regs


def test_the_acting_player_reaches_the_outputs():
    model = _small(2)
    s = synthetic_state(13, 6)
    one, two = model(collate([s])), model(collate([shifted(s, legal=None)]))
    assert not torch.allclose(one.value_logits, two.value_logits)
    assert not torch.allclose(one.policy_logits, two.policy_logits)


# --- the submit action -------------------------------------------------------------------------


def test_the_submit_logit_is_absent_masked_or_live():
    """Not offered (no shift, or a shift with submit not offered): -inf, and the moves alone
    carry the distribution. Offered but illegal (C4): masked to -inf - the readout exists, the
    distribution is the moves'. Offered and legal: a finite logit, normalised with the moves."""
    model = _small(3)
    s = synthetic_state(14, 8)
    cases = {"start": s, "unoffered": shifted(s, legal=None), "illegal": shifted(s, legal=False), "legal": shifted(s, legal=True)}
    outs = {name: model(collate([state])) for name, state in cases.items()}
    for name, out in outs.items():
        assert torch.isfinite(out.submit_raw).all(), name
        possible, legal = cases[name].submit_possible, cases[name].submit_legal
        assert bool(out.submit_possible[0]) == possible and bool(out.submit_valid[0]) == legal
        moves = out.policy_log_probs()[0].exp()
        sub = out.submit_log_prob()[0].exp()
        assert torch.allclose(moves.sum() + sub, torch.tensor(1.0), atol=1e-5), name  # one distribution
        if legal:
            assert torch.isfinite(out.submit_logits[0]) and out.submit_logits[0] == out.submit_raw[0]
            assert 0 < float(sub) < 1
        else:
            assert out.submit_logits[0] == float("-inf") and float(sub) == 0.0
            assert torch.allclose(out.policy_log_probs()[0], torch.log_softmax(out.policy_logits[0], dim=-1), atol=1e-6)
    # The mask decides nothing else: the same input with submit legal or illegal has the same
    # readout and move logits.
    assert torch.equal(outs["illegal"].submit_raw, outs["legal"].submit_raw)
    assert torch.equal(outs["illegal"].policy_logits, outs["legal"].policy_logits)
    legal = outs["legal"]
    joint = torch.log_softmax(torch.cat([legal.policy_logits[0], legal.submit_logits]), dim=0)
    assert torch.allclose(legal.action_log_probs()[0], joint, atol=1e-6)


def test_a_batch_mixes_the_three_cases():
    model = _small(4)
    states = [synthetic_state(20, 3), shifted(synthetic_state(21, 6), legal=False), shifted(synthetic_state(22, 5))]
    out = model(collate(states))
    logp = out.action_log_probs()
    assert torch.allclose(logp.exp().sum(dim=-1), torch.ones(3), atol=1e-5)
    assert out.submit_valid.tolist() == [False, False, True]
    assert out.action_valid().shape == (3, out.policy_logits.shape[1] + 1)
    for i, s in enumerate(states):  # each position's evaluation does not depend on its batch
        alone = model(collate([s]))
        assert torch.allclose(alone.submit_raw, out.submit_raw[i : i + 1], atol=1e-5)


def test_submit_alone():
    """No candidate move and a legal submit: the decision is submit, probability one; neither:
    no candidate, -inf everywhere (never NaN)."""
    model = _small(5)
    s = synthetic_state(23, 4)
    empty = replace(s, moves=np.zeros((0, 3), dtype=np.int64))
    out = model(collate([shifted(empty), empty]))
    assert out.policy_logits.shape == (2, 0)
    assert out.submit_log_prob().tolist()[0] == pytest.approx(0.0, abs=1e-6)
    assert out.action_log_probs()[1].tolist() == [float("-inf")]


@pytest.mark.parametrize("form", ["linear", "pointer"])
@pytest.mark.parametrize("register", [0, 5])
def test_the_submit_logit_is_read_from_its_global_register(form, register):
    model = _small(6, submit_head=form, submit_register=register)
    s = shifted(synthetic_state(24, 5))
    batch = collate([s])
    out = model(batch, return_trunk=True)
    h_r = out.trunk[0, batch.body_len + register]
    assert torch.equal(submit_register_state(out.trunk, batch, register)[0], h_r)
    if form == "linear":
        expected = model.submit.out(h_r)[0]
        assert parameter_breakdown(model)["head.submit"] == model.cfg.d_model + 1
    else:
        expected = (model.policy.src(h_r) * model.policy.dst(h_r)).sum() / math.sqrt(model.policy.dim)
        assert parameter_breakdown(model)["head.submit"] == 0
    assert torch.allclose(out.submit_raw[0], expected, atol=1e-6)
    # Only that register's state matters: perturb the trunk elsewhere, the readout stays.
    h = out.trunk.clone()
    others = [batch.body_len + g for g in range(s.global_regs) if g != register]
    h[0, others] += 1.0
    assert torch.allclose(model.heads(h, batch).submit_raw, out.submit_raw, atol=1e-6)


def test_no_global_registers_reads_the_squares():
    model = _small(7, global_regs=0)
    s = shifted(synthetic_state(25, 4, global_regs=0))
    out = model(collate([s]))
    assert torch.isfinite(out.submit_logits).all()


def test_the_configuration_is_validated():
    with pytest.raises(ValueError, match="acting_player_input"):
        NetConfig(acting_player_input="boards").validate()
    with pytest.raises(ValueError, match="submit_head"):
        NetConfig(submit_head="mlp").validate()
    with pytest.raises(ValueError, match="submit_register"):
        NetConfig(submit_register=16).validate()


# --- section 15.2: dense and cached agree with the new input --------------------------------------


def _close(a, b, atol_trunk=2e-5, atol=1e-4):
    assert torch.allclose(a.trunk, b.trunk, atol=atol_trunk), float((a.trunk - b.trunk).abs().max())
    assert torch.allclose(a.value_logits, b.value_logits, atol=atol)
    v = a.move_valid
    assert torch.allclose(a.policy_logits[v], b.policy_logits[v], atol=atol)
    assert torch.allclose(a.submit_raw, b.submit_raw, atol=atol)
    assert torch.equal(torch.isinf(a.submit_logits), torch.isinf(b.submit_logits))
    assert torch.allclose(a.action_log_probs()[a.action_valid()], b.action_log_probs()[b.action_valid()], atol=atol)


@pytest.mark.parametrize(
    "arm,features",
    [("global_registers", "intrinsic"), ("global_registers", "current"), ("all_tokens", "intrinsic"), ("all_tokens", "current")],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_cached_equals_one_full_pass_with_the_action(arm, features, seed):
    """Section 15.2 row (a) with D8: the same boards under both acting players and all three
    submit cases, evaluated incrementally from one store, each equal to one full block-causal
    pass. Under the default arm (and under ``all_tokens`` with intrinsic frozen features) the
    acting player never reaches a frozen board, so the second reading computes no frozen
    board; under ``all_tokens`` with ``current`` features it reaches every board, and the
    frozen boards are computed again, under their own identities."""
    model = build_model(
        NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", acting_player_input=arm, frozen_features=features), seed=seed
    ).eval()
    root = synthetic_state(300 + seed, 16, n_timelines=4, duplicate_prob=0.3)
    cached = CachedEvaluator(model)
    epoch = cached.root_epoch(root)
    computed = dict(cached.store.computed)
    for variant in (root, shifted(root, legal=None), shifted(root, legal=False), shifted(root, legal=True)):
        _close(cached.evaluate(variant, epoch), reference_output(model, variant))
    recomputed = sum(cached.store.computed.values()) - sum(computed.values())
    frozen = int((~root.is_head).sum())
    if arm == "all_tokens" and features == "current":
        assert recomputed == frozen  # one new entry per frozen board: the other acting player's
    else:
        assert recomputed == 0  # the frozen boards' K/V do not depend on the acting player


@pytest.mark.parametrize("arm", ["global_registers", "all_tokens"])
def test_a_refreshed_epoch_with_the_action(arm):
    """Under a root refresh (C3) the anchor's dense pass conditions the history on the anchor's
    acting player; a node whose acting player differs is still one function of (state, epoch):
    the cached path equals the two-level reference."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", acting_player_input=arm), seed=3).eval()
    root = synthetic_state(310, 14, n_timelines=3)
    cached = CachedEvaluator(model, CacheConfig(root_refresh=True))
    epoch = cached.root_epoch(root)
    for variant in (root, shifted(root, legal=True)):
        _close(cached.evaluate(variant, epoch), reference_output(model, variant, anchor=root))


# --- the parameters ------------------------------------------------------------------------------


D8_PARAMETERS = ("embed.acting.weight", "submit.out.weight", "submit.out.bias")
FLAG_SETS = [
    {},
    {"typed_edge_bias": True, "displacement_bias": False},
    {"rope_per_head": True, "routing_table": True, "typed_edge_bias": True, "board_pair_smolgen": True},
    {"board_register_spatial": "position_free", "policy_promotion": "offset"},
]


@pytest.mark.parametrize("flags", FLAG_SETS, ids=["default", "b11-no-table", "b5-b10-b11-b12", "position-free"])
def test_d8_parameters_and_their_draws(flags):
    """The acting-player embedding (2 x d) and the linear submit readout (d + 1) are D8's only
    parameters. They draw nothing while they are made and are drawn from a generator of their
    own, seeded from the model's seed - never from the global stream - so, whatever the flags,
    D8's arms (the submit readout's form, where the acting player enters) leave every other
    parameter as it is (checked against the pre-N5 package by hand: every older tensor equal,
    for these flag sets and both arms), and no flag changes D8's weights."""
    base = build_model(NetConfig(n_layers=2, **flags), seed=0)
    counts = parameter_breakdown(base)
    assert base.embed.acting.weight.shape == (2, 256) and counts["head.submit"] == 257
    base_params = dict(base.named_parameters())
    for arm in ({"submit_head": "pointer"}, {"acting_player_input": "all_tokens"}):
        other = build_model(NetConfig(n_layers=2, **flags, **arm), seed=0)
        for name, p in other.named_parameters():
            if name not in D8_PARAMETERS:
                assert torch.equal(p, base_params[name]), (arm, name)
    plain = dict(build_model(NetConfig(n_layers=2), seed=0).named_parameters())
    for name in D8_PARAMETERS:
        assert torch.equal(base_params[name], plain[name]), name  # no flag changes D8's weights
    reseeded = dict(build_model(NetConfig(n_layers=2, **flags), seed=1).named_parameters())
    assert not torch.equal(reseeded["embed.acting.weight"], base_params["embed.acting.weight"])  # a function of the seed
