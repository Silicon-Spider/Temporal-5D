"""Stage N5: register D8 in the network - the acting-player input and the submit action.

Turns have variable length (rulebook R8): once the present has shifted inside an action, the
boards' present names the opponent while the action is still the mover's. D8 gives the
network a global acting-player input - on the global registers (M20: "acting player -> global
registers") - and the submit action its own policy logit, read from the pooled global
registers (M20: "submit logit -> pooled global registers"), offered where submitting is
possible and masked unless it is legal; the policy is one distribution over the moves and
submit.

Checked here on synthetic states (no engine): where the input enters the encoding, under each
scope - on the global registers, or with none (ablation 3's arms without them) on the board
tokens, the live boards' under the block-causal scope (a placeholder, spec issue 225); that it
reaches the outputs under every register arm; the submit logit's three cases (absent, masked,
live), its pooled source and both readouts; the joint normalisation; section 15.2's
dense-vs-cached equivalence with the new input (C2, C17, C19, C20) under both
``frozen_features`` readings and a root refresh, with and without global registers; the
parameters and their draw order. Real engine positions before and after the shift:
``tests/engine``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, permute_boards, synthetic_state, translate
from t5d.net.cache import CacheConfig, CachedEvaluator, board_ident, reference_output
from t5d.net.heads import global_register_pool, square_pool
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


#: Every register arm the configuration accepts (ablation 3: registers on/off, 2 vs 4 per
#: board): (per-board, global).
REGISTER_ARMS = [(2, 16), (4, 16), (0, 16), (2, 0), (4, 0), (0, 0)]
REGISTER_IDS = [f"registers-{r}-{g}" for r, g in REGISTER_ARMS]


@pytest.mark.parametrize("registers", [(2, 16), (2, 0), (0, 0)], ids=["registers-2-16", "registers-2-0", "registers-0-0"])
@pytest.mark.parametrize(
    "scope,features",
    [("dense", "intrinsic"), ("block_causal", "intrinsic"), ("block_causal", "current")],
)
def test_the_acting_player_changes_the_encoding_exactly_where_it_should(scope, features, registers):
    """Two readings of the same boards that differ only in the acting player: the token
    embeddings differ on the global registers (M20: "acting player -> global registers") and
    nowhere else - no board token, frozen or live, under either scope or frozen-feature
    reading; padding never. With no global registers (an ablation-3 arm) they differ on the
    board tokens instead - squares and per-board registers - every board's under the dense
    scope, the head boards' alone under the block-causal one, either frozen-feature reading (a
    placeholder, spec issue 225: no frozen board's input depends on the acting player)."""
    r, g = registers
    model = _small(1, frozen_features=features, regs_per_board=r, global_regs=g)
    a = synthetic_state(11, 9, n_timelines=3, regs_per_board=r, global_regs=g)
    b = synthetic_state(12, 4, regs_per_board=r, global_regs=g)
    before = collate([a, b])
    after = collate([shifted(a), b])
    ea, eb = model.embed_tokens(before, scope), model.embed_tokens(after, scope)
    differs = (ea - eb).abs().amax(dim=-1) > 0
    kind = before.kind
    expected = torch.zeros_like(differs)
    if g:
        expected[0] = kind[0] == TOKEN_GLOBAL_REGISTER
    else:
        on_board = (kind[0] == TOKEN_SQUARE) | (kind[0] == TOKEN_BOARD_REGISTER)
        expected[0] = on_board if scope == "dense" else on_board & (before.is_head[0] == 1)
        assert bool(expected[0].any()) and (scope == "dense" or not bool(expected[0][on_board].all()))
    assert torch.equal(differs, expected)
    assert not differs[0][kind[0] == TOKEN_PAD].any() and not differs[1].any()
    if g:
        assert int(differs[0].sum()) == a.global_regs


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
@pytest.mark.parametrize("registers", REGISTER_ARMS, ids=REGISTER_IDS)
def test_the_acting_player_reaches_the_outputs(registers, scope):
    """The same boards under the other acting player (section 2.3's case: after the present
    shifts a state can share every board with the opponent's turn start) give other outputs -
    under every register arm the configuration accepts, those without global registers among
    them (spec issue 225), under either scope: the policy, the value-type heads, the squares'
    heads and the submit readout all move."""
    r, g = registers
    model = _small(2, regs_per_board=r, global_regs=g, attn_scope=scope)
    s = synthetic_state(13, 6, n_timelines=2, regs_per_board=r, global_regs=g)
    one, two = model(collate([s])), model(collate([shifted(s, legal=None)]))
    v = one.move_valid
    for name in ("value_logits", "wdl_logits", "short_value_logits", "moves_left_log", "effort_logits", "submit_raw"):
        assert not torch.allclose(getattr(one, name), getattr(two, name)), name
    assert not torch.allclose(one.policy_logits[v], two.policy_logits[v])


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
def test_the_submit_logit_is_read_from_the_pooled_global_registers(form):
    """Section 8.2 and M20: the submit logit is read "from the pooled global registers as the
    value head is" - their mean - by either readout (``submit_head``; the form is not ruled):
    every global register's final state counts alike, so permuting them changes nothing and
    changing any one changes the logit (no register has a role: D7), while the squares and the
    per-board registers do not enter."""
    model = _small(6, submit_head=form)
    s = shifted(synthetic_state(24, 5))
    batch = collate([s])
    out = model(batch, return_trunk=True)
    h = out.trunk
    pooled = h[0, batch.body_len : batch.body_len + s.global_regs].mean(0)
    assert torch.allclose(global_register_pool(h, batch)[0], pooled, atol=1e-6)
    if form == "linear":
        expected = model.submit.out(pooled)[0]
        assert parameter_breakdown(model)["head.submit"] == model.cfg.d_model + 1
    else:
        expected = (model.policy.src(pooled) * model.policy.dst(pooled)).sum() / math.sqrt(model.policy.dim)
        assert parameter_breakdown(model)["head.submit"] == 0
    assert torch.allclose(out.submit_raw[0], expected, atol=1e-6)
    g = torch.arange(batch.body_len, batch.body_len + s.global_regs)
    shuffled = h.clone()
    shuffled[:, g] = h[:, g.flip(0)]
    assert torch.allclose(model.heads(shuffled, batch).submit_raw, out.submit_raw, atol=1e-6), "no register has a role"
    for register in (0, 5, 15):
        one = h.clone()
        one[0, batch.body_len + register] += 1.0
        assert not torch.allclose(model.heads(one, batch).submit_raw, out.submit_raw, atol=1e-4), register
    body = h.clone()
    body[0, : batch.body_len] += 1.0  # every square and per-board register
    assert torch.equal(model.heads(body, batch).submit_raw, out.submit_raw)


def test_no_global_registers_reads_the_squares():
    """With no global registers (an ablation-3 arm) the pool falls back to the squares' mean, as
    the value head's does."""
    model = _small(7, global_regs=0)
    s = shifted(synthetic_state(25, 4, global_regs=0))
    batch = collate([s])
    out = model(batch, return_trunk=True)
    assert torch.isfinite(out.submit_logits).all()
    assert torch.allclose(out.submit_raw, model.submit.out(square_pool(out.trunk, batch)).squeeze(-1), atol=1e-6)


def test_the_configuration_is_validated():
    with pytest.raises(ValueError, match="submit_head"):
        NetConfig(submit_head="mlp").validate()
    for removed in ("acting_player_input", "submit_register"):  # M20 ruled both places: no switch
        with pytest.raises(TypeError):
            NetConfig(**{removed: 0})


# --- section 15.2: dense and cached agree with the new input --------------------------------------


def _close(a, b, atol_trunk=2e-5, atol=1e-4):
    assert torch.allclose(a.trunk, b.trunk, atol=atol_trunk), float((a.trunk - b.trunk).abs().max())
    assert torch.allclose(a.value_logits, b.value_logits, atol=atol)
    v = a.move_valid
    assert torch.allclose(a.policy_logits[v], b.policy_logits[v], atol=atol)
    assert torch.allclose(a.submit_raw, b.submit_raw, atol=atol)
    assert torch.equal(torch.isinf(a.submit_logits), torch.isinf(b.submit_logits))
    assert torch.allclose(a.action_log_probs()[a.action_valid()], b.action_log_probs()[b.action_valid()], atol=atol)


@pytest.mark.parametrize("global_regs", [16, 0])
@pytest.mark.parametrize("features", ["intrinsic", "current"])
@pytest.mark.parametrize("seed", [0, 1])
def test_cached_equals_one_full_pass_with_the_action(features, seed, global_regs):
    """Section 15.2 row (a) with D8: the same boards under both acting players and all three
    submit cases, evaluated incrementally from one store, each equal to one full block-causal
    pass. The acting player is on the global registers (M20) - with none, on the live boards'
    tokens (spec issue 225) - which are live either way, so it never reaches a frozen board -
    under either frozen-feature reading - and the second reading computes no frozen board."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", frozen_features=features, global_regs=global_regs)
    model = build_model(cfg, seed=seed).eval()
    root = synthetic_state(300 + seed, 16, n_timelines=4, duplicate_prob=0.3, global_regs=global_regs)
    cached = CachedEvaluator(model)
    epoch = cached.root_epoch(root)
    computed = dict(cached.store.computed)
    for variant in (root, shifted(root, legal=None), shifted(root, legal=False), shifted(root, legal=True)):
        _close(cached.evaluate(variant, epoch), reference_output(model, variant))
    recomputed = sum(cached.store.computed.values()) - sum(computed.values())
    assert recomputed == 0  # the frozen boards' K/V do not depend on the acting player


def test_a_frozen_boards_identity_carries_the_acting_player_only_when_asked():
    """``cache.board_ident``'s ``acting``: no frozen board's input depends on the acting player
    since M20 put it on the global registers alone, so the cached path keys boards without it
    (above: nothing recomputed) - the mechanism stays for A7's per-perspective keys (spec issue
    303), and keys by the acting player under either frozen-feature reading."""
    s = synthetic_state(305, 8, n_timelines=3)
    t = shifted(s)
    for b in np.flatnonzero(~s.is_head):
        for features in ("intrinsic", "current"):
            assert board_ident(s, int(b), features) == board_ident(t, int(b), features)
            assert board_ident(s, int(b), features, acting=True) != board_ident(t, int(b), features, acting=True)
            assert board_ident(s, int(b), features, acting=True) != board_ident(s, int(b), features)


@pytest.mark.parametrize("global_regs", [16, 0])
def test_a_refreshed_epoch_with_the_action(global_regs):
    """Under a root refresh (C3) the anchor's dense pass conditions the history on the anchor's
    acting player - through the global registers, or with none through every board token (spec
    issue 225); a node whose acting player differs is still one function of (state, epoch):
    the cached path equals the two-level reference."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", global_regs=global_regs), seed=3).eval()
    root = synthetic_state(310, 14, n_timelines=3, global_regs=global_regs)
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
    {"regs_per_board": 4, "global_regs": 0},
]


@pytest.mark.parametrize("flags", FLAG_SETS, ids=["default", "b11-no-table", "b5-b10-b11-b12", "registers-4-0"])
def test_d8_parameters_and_their_draws(flags):
    """The acting-player embedding (2 x d) and the linear submit readout (d + 1) are D8's only
    parameters. They draw nothing while they are made and are drawn from a generator of their
    own, seeded from the model's seed - never from the global stream - so, whatever the flags,
    the submit readout's other form leaves every other parameter as it is (checked against the
    pre-N5 package by hand at N5: every older tensor equal, for that stage's flag sets and arms),
    and no flag changes D8's weights."""
    base = build_model(NetConfig(n_layers=2, **flags), seed=0)
    counts = parameter_breakdown(base)
    assert base.embed.acting.weight.shape == (2, 256) and counts["head.submit"] == 257
    base_params = dict(base.named_parameters())
    for arm in ({"submit_head": "pointer"},):
        other = build_model(NetConfig(n_layers=2, **flags, **arm), seed=0)
        for name, p in other.named_parameters():
            if name not in D8_PARAMETERS:
                assert torch.equal(p, base_params[name]), (arm, name)
    plain = dict(build_model(NetConfig(n_layers=2), seed=0).named_parameters())
    for name in D8_PARAMETERS:
        assert torch.equal(base_params[name], plain[name]), name  # no flag changes D8's weights
    reseeded = dict(build_model(NetConfig(n_layers=2, **flags), seed=1).named_parameters())
    assert not torch.equal(reseeded["embed.acting.weight"], base_params["embed.acting.weight"])  # a function of the seed
