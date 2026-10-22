"""Register A6 in the network: the global turn-state counts (section 2.3).

"Mandatory boards still to move ... Optional boards still playable, and moves already made this
turn - the second also tells a turn start from a mid-turn state with the same boards ... Exact
counts, never predicted ones: the rules engine knows them" - embedded into the global
registers. The counts come from the engine (``tests/engine/test_turn_state_engine.py``: section
15.2's A6 row on the network side); here, on synthetic states (no engine): what a state derives
when it is built without them, how it validates and carries them; where they enter the
encoding - the global registers, or with none (issue 225's placeholder) the board tokens, the
live boards' under the block-causal scope - and that nothing else moves; that every output moves
with each count under every register arm; that two states with equal boards and different
counts evaluate differently and no cache confuses them (the anchor's digest, the frozen store,
a refreshed epoch); cached = one full pass with the counts varied; and the MLP's parameters and
their draws (a placeholder, spec issue 230).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import (
    NetConfig,
    TurnState,
    build_model,
    collate,
    mirror,
    parameter_breakdown,
    permute_boards,
    relayout,
    synthetic_state,
    translate,
)
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, state_digest, viewed_by
from t5d.net.model import a6_seed
from t5d.net.state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_SQUARE, TURN_STATE_FIELDS
from t5d.net.tree import Multiverse
from tests.net.test_d8 import REGISTER_ARMS, REGISTER_IDS, _close, shifted

#: A6's parameters: Linear(9, 32) and Linear(32, 256), with biases.
A6_PARAMETERS = ("embed.turn_in.weight", "embed.turn_in.bias", "embed.turn_out.weight", "embed.turn_out.bias")
A6_COUNT = 9 * 32 + 32 + 32 * 256 + 256


def _small(seed: int = 0, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=2, **flags), seed=seed).eval()


def _recount(s) -> TurnState:
    """The default rules' counts written out here, board by board: a board is playable when it is
    a head with the acting player to move (L5); mandatory when it is also an active timeline's
    head in the present (T6, InPresent); optional otherwise (moves allowed after the shift, R8)."""
    mandatory = optional = 0
    for b in range(s.n_boards):
        if s.is_head[b] and int(s.side_to_move[b]) == int(s.acting_player):
            if s.active[b] and int(s.half_turn[b]) == s.h_present:
                mandatory += 1
            else:
                optional += 1
    return TurnState(mandatory, optional, int(s.shifted))


# --- the contract --------------------------------------------------------------------------------


def test_a_state_built_without_them_derives_them():
    """A synthetic state is a turn start: the active present heads mandatory (at least one), the
    acting player's other heads optional, no move made. Past the shift - the same boards with the
    acting player swapped - they re-derive: nothing owed, the other colour's heads optional, one
    move made (the fewest a shift needs)."""
    seen_optional = 0
    for seed in range(8):
        s = synthetic_state(40 + seed, 14, n_timelines=5)
        assert (s.mandatory_remaining, s.optional_playable, s.moves_made) == (None, None, None)
        assert s.turn_state == s.derived_turn_state == _recount(s)
        assert s.turn_state.mandatory_remaining >= 1 and s.turn_state.moves_made == 0
        seen_optional += s.turn_state.optional_playable > 0
        t = shifted(s)
        assert t.turn_state == _recount(t) and t.turn_state.mandatory_remaining == 0 and t.turn_state.moves_made == 1
    assert seen_optional  # some state has an optional board


def test_given_counts_are_kept_and_derived_ones_follow_the_state():
    s = synthetic_state(50, 12, n_timelines=4)
    given = replace(s, moves_made=3)
    assert given.turn_state == TurnState(s.turn_state.mandatory_remaining, s.turn_state.optional_playable, 3)
    past = replace(given, acting_player=1 - s.player_to_move, submit_possible=True, submit_legal=True)
    assert past.turn_state.moves_made == 3 and past.turn_state.mandatory_remaining == 0  # given kept, derived re-derived
    exact = replace(s, mandatory_remaining=1, optional_playable=0, moves_made=2)
    for t in (
        permute_boards(exact, list(reversed(range(s.n_boards)))),
        translate(exact, dp=-2, dh=4),
        relayout(exact, 4, 0),
        mirror(exact),
    ):
        assert t.turn_state == TurnState(1, 0, 2)
    for t in (s, shifted(s), replace(s, moves_made=5)):  # derived counts derive alike on the mirror
        assert mirror(t).turn_state == t.turn_state


def test_the_turn_state_is_validated():
    s = synthetic_state(51, 12, n_timelines=4)
    playable = s.playable_heads
    with pytest.raises(ValueError, match="moves_made must be non-negative"):
        replace(s, moves_made=-1)
    with pytest.raises(ValueError, match="integer count"):
        replace(s, mandatory_remaining=True)
    with pytest.raises(ValueError, match="integer count"):
        replace(s, optional_playable=1.5)
    with pytest.raises(ValueError, match="exceeds"):
        replace(s, mandatory_remaining=playable, optional_playable=1)
    with pytest.raises(ValueError, match="mandatory_remaining == 0"):
        replace(shifted(s), mandatory_remaining=1)  # T5: submitting is possible once nothing is owed
    replace(s, mandatory_remaining=playable, optional_playable=0, moves_made=7)  # within L5's playable heads


def test_a_batch_carries_the_counts():
    a, b = synthetic_state(52, 6), replace(shifted(synthetic_state(53, 9), legal=None), moves_made=4)
    batch = collate([a, b])
    assert batch.turn_state.tolist() == [list(a.turn_state), list(b.turn_state)]
    assert batch.turn_state[1, TURN_STATE_FIELDS.index("moves_made")] == 4
    bare = replace(batch, turn_state=None)
    assert bare.turn_counts().tolist() == [[0, 0, 0], [0, 0, 0]]


# --- where the counts enter -------------------------------------------------------------------------


def _carriers(batch, scope: str, g: int) -> torch.Tensor:
    """The tokens of position 0 the global inputs enter: the global registers; with none, the
    board tokens - the head boards' alone under the block-causal scope (spec issue 225)."""
    kind = batch.kind[0]
    if g:
        return kind == TOKEN_GLOBAL_REGISTER
    on_board = (kind == TOKEN_SQUARE) | (kind == TOKEN_BOARD_REGISTER)
    return on_board if scope == "dense" else on_board & (batch.is_head[0] == 1)


@pytest.mark.parametrize("registers", [(2, 16), (2, 0), (0, 0)], ids=["registers-2-16", "registers-2-0", "registers-0-0"])
@pytest.mark.parametrize("scope", ["dense", "block_causal"])
def test_each_count_changes_the_encoding_exactly_where_it_should(scope, registers):
    """Each count, changed alone (the boards, the acting player and the other counts as they
    are), changes the embedding of exactly the tokens the global inputs enter - every global
    register (the same vector on each), or with none the board tokens, the live boards' alone
    under the block-causal scope, so no frozen board's input depends on it - and nothing else, in
    either position of the batch."""
    r, g = registers
    model = _small(4, regs_per_board=r, global_regs=g)
    a = synthetic_state(60, 10, n_timelines=4, regs_per_board=r, global_regs=g)
    b = synthetic_state(61, 5, regs_per_board=r, global_regs=g)
    base = collate([a, b])
    e0 = model.embed_tokens(base, scope)
    expected = torch.zeros(e0.shape[:2], dtype=torch.bool)
    expected[0] = _carriers(base, scope, g)
    assert bool(expected[0].any()) and (g or scope == "dense" or not bool(expected[0][base.kind[0] != TOKEN_GLOBAL_REGISTER].all()))
    for column in range(len(TURN_STATE_FIELDS)):
        for delta in (1, 2, 7):
            counts = base.turn_state.clone()
            counts[0, column] += delta
            e1 = model.embed_tokens(replace(base, turn_state=counts), scope)
            differs = (e1 - e0).abs().amax(-1) > 0
            assert torch.equal(differs, expected), (TURN_STATE_FIELDS[column], delta)
            if g:
                moved = (e1 - e0)[0][expected[0]]
                assert torch.allclose(moved, moved[:1].expand_as(moved), atol=1e-6), "one vector on every global register"


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
@pytest.mark.parametrize("registers", REGISTER_ARMS, ids=REGISTER_IDS)
def test_each_count_reaches_the_outputs(registers, scope):
    """The same position with one count changed gives other outputs - the policy, the value-type
    heads, the squares' heads and the submit readout - under every register arm the
    configuration accepts (those without global registers among them, spec issue 225), under
    either scope."""
    r, g = registers
    model = _small(5, regs_per_board=r, global_regs=g, attn_scope=scope)
    s = next(t for t in (synthetic_state(62 + i, 8, n_timelines=3, regs_per_board=r, global_regs=g) for i in range(50)) if t.turn_state[1])
    base = model(collate([s]))
    m, o, k = s.turn_state
    assert m >= 1 and o >= 1
    for name, t in (
        ("mandatory", replace(s, mandatory_remaining=m - 1)),
        ("optional", replace(s, optional_playable=o - 1)),
        ("moves", replace(s, moves_made=k + 2)),
    ):
        out = model(collate([t]))
        for head in ("value_logits", "wdl_logits", "short_value_logits", "moves_left_log", "effort_logits", "submit_raw"):
            assert not torch.allclose(getattr(base, head), getattr(out, head)), (name, head)
        assert not torch.allclose(base.policy_logits[base.move_valid], out.policy_logits[out.move_valid]), name


def test_the_features_of_a_count():
    model = _small(6)
    counts = torch.tensor([[0, 3, 16]])
    f = model.embed.turn_features(counts).reshape(3, 3)
    scale = model.cfg.turn_state_scale
    expected = torch.tensor([[0.0, 0.0, 1.0], [3 / scale, np.log1p(3), 0.0], [16 / scale, np.log1p(16), 0.0]])
    assert torch.allclose(f, expected.float())


# --- equal boards, different counts ------------------------------------------------------------------


def _same_boards(seed: int, global_regs: int = 16):
    """A turn start and three states with every board and the acting player equal: mid-turn after
    one and after two moves (section 2.3's case: "tells a turn start from a mid-turn state with the
    same boards"), one with a board fewer owed, and one with an optional board fewer (which only
    another ruleset could give equal boards - it pins the optional count in the digest)."""
    s = synthetic_state(seed, 14, n_timelines=4, duplicate_prob=0.3, global_regs=global_regs)
    m = s.turn_state.mandatory_remaining
    others = [replace(s, moves_made=1), replace(s, moves_made=2)]
    if m > 1:
        others.append(replace(s, mandatory_remaining=m - 1, moves_made=1))
    o = s.turn_state.optional_playable
    if o >= 1:
        others.append(replace(s, optional_playable=o - 1))
    return s, others


@pytest.mark.parametrize("global_regs", [16, 0])
def test_equal_boards_with_different_counts_are_never_confused(global_regs):
    """Section 15.2's purity row (C19-C21) with A6: an evaluation is a function of the state with
    its counts. States that share every board and the acting player but not the counts have
    different digests (an epoch's anchor, ``state_digest``) and different outputs; evaluated
    incrementally from one store, in one epoch, each equals its own full pass, and none computes
    a frozen board again - the frozen K/V never depend on the counts."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", global_regs=global_regs)
    model = build_model(cfg, seed=7).eval()
    s, others = _same_boards(400, global_regs)
    states = [s, *others]
    assert len({state_digest(t) for t in states}) == len(states)
    cached = CachedEvaluator(model)
    epoch = cached.root_epoch(s)
    computed = dict(cached.store.computed)
    outs = []
    for t in states:
        got = cached.evaluate(t, epoch)
        _close(got, reference_output(model, t))
        outs.append(got)
    assert cached.store.computed == computed  # no frozen board recomputed for other counts
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert not torch.allclose(outs[i].value_logits, outs[j].value_logits), (i, j)
    again = cached.evaluate(s, epoch)  # the store holds no evaluation of another state's counts
    assert torch.equal(again.value_logits, outs[0].value_logits) and torch.equal(again.trunk, outs[0].trunk)


@pytest.mark.parametrize("global_regs", [16, 0])
def test_a_refreshed_epochs_anchor_is_the_state_with_its_counts(global_regs):
    """Under a root refresh (C3) the dense pass over the anchor lets every board read the global
    registers, so the history K/V depend on the anchor's counts and the anchor's digest carries
    them: a node with the anchor's boards and acting player but other counts is not the anchor -
    its evaluation is the two-level one (B16 applied), equal to the reference - and a root with
    other counts opens an epoch of its own, whose history differs."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", global_regs=global_regs)
    model = build_model(cfg, seed=8).eval()
    s, others = _same_boards(410, global_regs)
    cached = CachedEvaluator(model, CacheConfig(root_refresh=True))
    epoch = cached.root_epoch(s)
    assert epoch.anchor == state_digest(s)
    at_anchor = cached.evaluate(s, epoch)
    _close(at_anchor, reference_output(model, s, anchor=s))
    for t in others:
        got = cached.evaluate(t, epoch)
        _close(got, reference_output(model, t, anchor=s))
        assert not torch.allclose(got.value_logits, at_anchor.value_logits)
    t = others[0]
    other_epoch = cached.root_epoch(t)
    assert other_epoch.anchor == state_digest(t) != epoch.anchor and other_epoch.key != epoch.key
    frozen = [int(b) for b in np.flatnonzero(~s.is_head)]
    if global_regs:  # the history reads the counts through the global registers
        mine = cached.store[(epoch.key, cached._plan(s, epoch).digest[frozen[0]])]
        theirs = cached.store[(other_epoch.key, cached._plan(t, other_epoch).digest[frozen[0]])]
        assert not torch.equal(mine.kv, theirs.kv)


def test_the_other_colours_view_derives_its_counts():
    """A refresh's view from the colour not acting at the anchor (``viewed_by``; spec issue 229)
    cannot keep the anchor's counts - they are its acting player's - so it takes the boards' own
    for that colour (spec issue 230): from an anchor past the shift, the opponent's turn start on
    these boards; from one before it, nothing owed and a move made."""
    s = replace(synthetic_state(420, 12, n_timelines=4), moves_made=2)
    view = viewed_by(s, 1 - int(s.acting_player))
    assert view.turn_state == view.derived_turn_state and view.turn_state.mandatory_remaining == 0 and view.turn_state.moves_made == 1
    past = replace(shifted(s), moves_made=3, mandatory_remaining=0, optional_playable=0)
    start = viewed_by(past, 1 - int(past.acting_player))
    assert start.turn_state == start.derived_turn_state and start.turn_state.moves_made == 0
    assert start.turn_state.mandatory_remaining >= 1
    assert viewed_by(past, int(past.acting_player)) is past  # the anchor's own view keeps its counts


@pytest.mark.parametrize("supersession", ["causal_past", "past_and_epoch_history"])
@pytest.mark.parametrize("global_regs", [16, 0])
def test_cached_equals_one_full_pass_on_a_tree_with_counts(global_regs, supersession):
    """Section 15.2 row (a) along the search-tree fixture's paths with the counts varied: the
    fixture's positions are turn starts (its moves are no actions - ``tree.py`` - so its two move
    orders of one multiverse stay one state), and each node is also evaluated as a mid-turn state
    of its boards, one and three moves into the action, and with a board fewer owed - from one
    store, each equal to its own full pass, no frozen board computed twice."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=1, attn_scope="block_causal", global_regs=global_regs, supersession_scope=supersession)
    model = build_model(cfg, seed=9).eval()
    root = Multiverse.from_state(synthetic_state(430, 10, n_timelines=3, global_regs=global_regs), seed=1)
    cached = CachedEvaluator(model)
    epoch = cached.root_epoch(root.state())
    history_from = root.state() if supersession != "causal_past" else None
    frontier, checked = [root], 0
    for _ in range(3):
        nxt = []
        for u in frontier:
            for p in sorted(u.heads())[:2]:
                child = u.physical(p)
                s = child.state()
                m = s.turn_state.mandatory_remaining
                variants = [s, replace(s, moves_made=1), replace(s, moves_made=3)]
                if m > 1:
                    variants.append(replace(s, mandatory_remaining=m - 1, moves_made=1))
                for t in variants:
                    computed = sum(cached.store.computed.values())
                    _close(cached.evaluate(t, epoch), reference_output(model, t, history_from=history_from))
                    if t is not s:
                        assert sum(cached.store.computed.values()) == computed  # the counts reach no frozen entry
                    checked += 1
                nxt.append(child)
        frontier = nxt
    assert checked >= 36


# --- the parameters -------------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [{}, {"regs_per_board": 4, "global_regs": 0}, {"typed_edge_bias": True, "submit_head": "pointer"}])
def test_a6_parameters_and_their_draws(flags):
    """The turn-state MLP - Linear(9, 32) and Linear(32, d) with biases, 8,768 parameters at
    d = 256 (a placeholder, spec issue 230) - draws nothing while it is made and is drawn from a
    generator of its own (``a6_seed``): its width changes no other parameter, no flag changes its
    weights, and a different seed does."""
    base = build_model(NetConfig(n_layers=2, **flags), seed=0)
    params = dict(base.named_parameters())
    assert sum(params[n].numel() for n in A6_PARAMETERS) == A6_COUNT == 8_768
    if not flags:
        assert parameter_breakdown(base)["embeddings"] == 19_584 + 2 * 256 + A6_COUNT
    wide = dict(build_model(NetConfig(n_layers=2, turn_state_hidden=48, **flags), seed=0).named_parameters())
    for name, p in wide.items():
        if name not in A6_PARAMETERS:
            assert torch.equal(p, params[name]), name  # A6 draws nothing from the global stream
    plain = dict(build_model(NetConfig(n_layers=2), seed=0).named_parameters())
    for name in A6_PARAMETERS:
        assert torch.equal(params[name], plain[name]), name
    gen = torch.Generator(device="cpu").manual_seed(a6_seed(0))
    expected = torch.empty(base.embed.turn_in.weight.shape).normal_(0.0, 9**-0.5, generator=gen)
    assert torch.equal(base.embed.turn_in.weight.detach(), expected)
    reseeded = dict(build_model(NetConfig(n_layers=2, **flags), seed=1).named_parameters())
    assert not torch.equal(reseeded["embed.turn_out.weight"], params["embed.turn_out.weight"])


def test_the_configuration_is_validated():
    for bad in ({"turn_state_hidden": 0}, {"turn_state_scale": 0.0}):
        with pytest.raises(ValueError, match="turn_state"):
            NetConfig(**bad).validate()
