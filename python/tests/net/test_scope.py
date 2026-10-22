"""Stage N3: the block-causal scope (C17, C19, C20) and the lineage it rests on.

The scope is checked against brute-force constructions written from its definition
(``scope.py``): who reads whom per board and per token, under both readings of the
causal past (spec issue 5) and both readings of C20; then through the network - every
attention implementation computes the same function under it, frozen tokens never read a
live head or a global register (a direct causality test), the section 6 structure opens
live rows only - B16's status biases among them - and a frozen board's state-dependent flags
follow ``frozen_features``.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, PositionState, build_model, collate, permute_boards, synthetic_state
from t5d.net.attention import _block_rows
from t5d.net.scope import BlockCausalScope, board_scope, causal_past, creation_matrix, lineage_matrix
from t5d.net.state import TOKEN_GLOBAL_REGISTER, TOKEN_PAD

GEOMETRIES = {
    "standard": dict(),
    "even_start": dict(even_start=True),
    "small_board": dict(width=5, height=5),
    "many_timelines": dict(n_timelines=9),
    "deep_travel": dict(deep_travel_prob=1.0, n_timelines=6),
}


def _states():
    for seed, (name, kw) in itertools.product(range(3), GEOMETRIES.items()):
        yield name, synthetic_state(100 + seed, 18, **kw)


# --- lineage on the input contract ------------------------------------------------------------


def test_synthetic_lineage_is_consistent():
    """Every parent is the timeline's previous board or a superseded branch origin one
    half-turn earlier on another timeline; creation follows the parents."""
    for _, state in _states():
        known = 0
        for b in range(state.n_boards):
            p, h, a = state.timeline[b], state.half_turn[b], state.parent[b]
            first = h == state.half_turn[state.timeline == p].min()
            if not first:
                assert state.timeline[a] == p and state.half_turn[a] == h - 1
            elif a >= 0:
                known += 1
                assert state.timeline[a] != p and state.half_turn[a] == h - 1 and not state.is_head[a]
                assert state.creation[a] < state.creation[b]
        assert len(set(state.creation.tolist())) == state.n_boards
    assert known > 0  # some branch origins are recorded


def test_lineage_validation_rejects_inconsistent_parents():
    state = synthetic_state(3, 12, n_timelines=3)
    names = ("width", "height", "timeline", "half_turn", "is_head", "active", "side_to_move", "piece", "colour", "moved")
    fields = {f: getattr(state, f) for f in names}
    first = np.array([state.half_turn[b] == state.half_turn[state.timeline == state.timeline[b]].min() for b in range(state.n_boards)])
    later = int(np.flatnonzero(~first)[0])
    branch = int(np.flatnonzero(first & (state.parent >= 0))[0])
    head = int(np.flatnonzero(state.is_head & (state.timeline != state.timeline[branch]))[0])

    def with_parent(b, a):
        parent = state.parent.copy()
        parent[b] = a
        return parent

    with pytest.raises(ValueError, match="previous board"):
        PositionState(**fields, parent=with_parent(later, -1))
    with pytest.raises(ValueError, match="parent"):
        PositionState(**fields, parent=with_parent(branch, head))  # not one half-turn earlier, or a head
    with pytest.raises(ValueError, match="another board"):
        PositionState(**fields, parent=np.arange(state.n_boards))
    with pytest.raises(ValueError, match="creation"):
        PositionState(**fields, parent=state.parent, creation=np.zeros(state.n_boards, dtype=np.int64))
    with pytest.raises(ValueError, match="creation"):
        PositionState(**fields, parent=state.parent, creation=state.creation.max() - state.creation)
    # Without lineage the parent defaults to the timeline's previous board (origins unknown).
    plain = PositionState(**fields)
    assert np.array_equal(plain.parent[~first], state.parent[~first]) and np.all(plain.parent[first] == -1)
    assert plain.creation is None


def test_permuting_boards_remaps_the_lineage():
    state = synthetic_state(8, 20, n_timelines=4)
    order = np.random.default_rng(0).permutation(state.n_boards)
    moved = permute_boards(state, order)
    new_index = np.argsort(order)
    for i, old in enumerate(order):
        a = state.parent[old]
        assert moved.parent[i] == (-1 if a < 0 else new_index[a])
    assert np.array_equal(moved.creation, state.creation[order])


# --- the causal past and the board-level scope ---------------------------------------------------


def _ancestors(parent, i):
    out, a = set(), int(parent[i])
    while a >= 0:
        out.add(a)
        a = int(parent[a])
    return out


def test_lineage_and_creation_relations():
    for _, state in _states():
        anc = lineage_matrix(state.parent)
        for i in range(state.n_boards):
            assert set(np.flatnonzero(anc[i]).tolist()) == _ancestors(state.parent, i)
        cre = creation_matrix(state.creation)
        assert np.array_equal(cre, state.creation[None, :] < state.creation[:, None])
        assert not np.any(np.diag(anc)) and not np.any(np.diag(cre))
        # Every ancestor was created first: the lineage is a sub-relation of creation.
        assert not np.any(anc & ~cre)
    with pytest.raises(ValueError, match="creation order"):
        causal_past(state.parent, None, "creation")


@pytest.mark.parametrize("supersession", ["causal_past", "past_and_epoch_history"])
@pytest.mark.parametrize("reading", ["lineage", "creation"])
def test_board_scope_matches_its_definition(reading, supersession):
    rng = np.random.default_rng(5)
    for _, state in _states():
        past = causal_past(state.parent, state.creation, reading)
        frozen = ~state.is_head
        history = frozen & (rng.random(state.n_boards) < 0.6)
        for hist in (None, history):
            got = board_scope(state.parent, state.creation, state.is_head, reading, supersession, hist)
            fixed = supersession == "past_and_epoch_history" and hist is not None
            for i, j in itertools.product(range(state.n_boards), repeat=2):
                if not frozen[i]:
                    want = True  # the live set reads every board (C19)
                elif i == j:
                    want = True  # within a board: bidirectional
                elif fixed and hist[i]:
                    want = bool(past[i, j] and hist[j])  # the epoch's history: a fixed block
                elif fixed and hist[j]:
                    want = True  # C20's second reading: the epoch's whole history
                else:
                    want = bool(past[i, j] and frozen[j])  # its past, never a live head
                assert got[i, j] == want, (reading, supersession, i, j)
            # Either way the frozen boards' reads have no cycle: the cache can compute them in order.
            reads = got & ~np.eye(state.n_boards, dtype=bool) & frozen[:, None]
            closure = reads.copy()
            for _ in range(state.n_boards):
                closure |= (closure.astype(np.int64) @ reads.astype(np.int64)) > 0
            assert not np.any(np.diag(closure))


def test_lineage_keys_are_superseded_for_good():
    """Under the lineage reading a frozen board reads only boards that are superseded - so
    its key set can never change; under creation an earlier head is not read until it is
    superseded."""
    seen_head_in_creation_past = False
    for _, state in _states():
        anc = lineage_matrix(state.parent)
        assert not np.any(anc[:, state.is_head]), "an ancestor is never a head"
        cre = creation_matrix(state.creation)
        seen_head_in_creation_past |= bool(np.any(cre[~state.is_head][:, state.is_head]))
    assert seen_head_in_creation_past


def _travelled_state(seed: int):
    """A position with travel arrivals that record their departure boards (the fixture)."""
    from t5d.net.tree import Multiverse

    u = Multiverse.from_state(synthetic_state(500 + seed, 14, n_timelines=4), seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        heads = u.heads()
        p = sorted(heads)[int(rng.integers(len(heads)))]
        sh = heads[p][1]
        targets = sorted(k for k in u.boards if k[1] % 2 == sh % 2 and k[1] <= sh and k != heads[p])
        u = u.jump(p, targets[int(rng.integers(len(targets)))]) if targets else u.physical(p)
    return u.state()


def _ancestors_two(parent, source, i):
    out, todo = set(), [i]
    while todo:
        j = todo.pop()
        for a in (int(parent[j]), int(source[j])):
            if a >= 0 and a not in out:
                out.add(a)
                todo.append(a)
    return out


def test_two_parent_lineage():
    """The two-parent reading: the closure over each board's parent and its travel source; it
    contains the arrival-side lineage, and every board of it is superseded (the jump
    supersedes its departure board) and created first."""
    for seed in range(4):
        state = _travelled_state(seed)
        assert np.any(state.source >= 0)
        two = lineage_matrix(state.parent, state.source)
        one = lineage_matrix(state.parent)
        for i in range(state.n_boards):
            assert set(np.flatnonzero(two[i]).tolist()) == _ancestors_two(state.parent, state.source, i)
        assert not np.any(one & ~two) and np.any(two & ~one)
        assert not np.any(two[:, state.is_head])
        assert not np.any(two & ~creation_matrix(state.creation))
        assert np.array_equal(causal_past(state.parent, state.creation, "two_parent", state.source), two)


@pytest.mark.parametrize("supersession", ["causal_past", "past_and_epoch_history"])
def test_two_parent_board_scope(supersession):
    rng = np.random.default_rng(9)
    for seed in range(3):
        state = _travelled_state(seed)
        past = causal_past(state.parent, state.creation, "two_parent", state.source)
        frozen = ~state.is_head
        history = frozen & (rng.random(state.n_boards) < 0.6)
        for hist in (None, history):
            got = board_scope(state.parent, state.creation, state.is_head, "two_parent", supersession, hist, state.source)
            fixed = supersession == "past_and_epoch_history" and hist is not None
            for i, j in itertools.product(range(state.n_boards), repeat=2):
                if not frozen[i] or i == j:
                    want = True
                elif fixed and hist[i]:
                    want = bool(past[i, j] and hist[j])
                elif fixed and hist[j]:
                    want = True
                else:
                    want = bool(past[i, j] and frozen[j])
                assert got[i, j] == want


def test_travel_sources_are_validated_and_remapped():
    state = _travelled_state(0)
    names = ("width", "height", "timeline", "half_turn", "is_head", "active", "side_to_move", "piece", "colour", "moved")
    fields = {f: getattr(state, f) for f in names}
    b = int(np.flatnonzero(state.source >= 0)[0])
    head = int(np.flatnonzero(state.is_head)[0])
    bad = state.source.copy()
    bad[b] = head
    with pytest.raises(ValueError, match="source"):
        PositionState(**fields, parent=state.parent, creation=state.creation, source=bad)
    bad[b] = b
    with pytest.raises(ValueError, match="source"):
        PositionState(**fields, parent=state.parent, creation=state.creation, source=bad)
    order = np.random.default_rng(1).permutation(state.n_boards)
    moved = permute_boards(state, order)
    new_index = np.argsort(order)
    for i, old in enumerate(order):
        a = state.source[old]
        assert moved.source[i] == (-1 if a < 0 else new_index[a])
    assert np.all(synthetic_state(3, 20).source == -1), "synthetic travel records no departure board"


# --- token rows ---------------------------------------------------------------------------------


@pytest.mark.parametrize("reading", ["lineage", "creation"])
def test_token_scope_matches_brute_force(reading):
    cfg = NetConfig(attn_scope="block_causal", causal_past=reading)
    states = [synthetic_state(11, 9, n_timelines=3), synthetic_state(12, 4)]
    batch = collate(states)
    scope = BlockCausalScope.build(batch, cfg)
    rows = scope.dense()[:, 0].numpy()
    for b, state in enumerate(states):
        boards = board_scope(state.parent, state.creation, state.is_head, reading)
        kind, board = batch.kind[b].numpy(), batch.board[b].numpy()
        real = kind != TOKEN_PAD
        for i in range(batch.n_tokens):
            if kind[i] in (TOKEN_PAD, TOKEN_GLOBAL_REGISTER):
                want = real  # padding (discarded) and global registers read every valid key
            else:
                head = bool(state.is_head[board[i]])
                on_board = board >= 0
                want = real & np.where(kind == TOKEN_GLOBAL_REGISTER, head, on_board & boards[board[i], np.maximum(board, 0)])
            assert np.array_equal(rows[b, i], want), (b, i)
    live = scope.live_tokens
    assert torch.equal(live, batch.valid & ((batch.kind == TOKEN_GLOBAL_REGISTER) | (batch.is_head == 1)))
    # Rows of any block, or of any rows, are the dense matrix's.
    assert np.array_equal(scope.rows(70, 140)[:, 0].numpy(), rows[:, 70:140])
    idx = torch.tensor([[3, 300, 64], [0, 5, 7]])
    assert np.array_equal(scope.rows_at(idx)[:, 0].numpy(), np.stack([rows[0, idx[0]], rows[1, idx[1]]]))


# --- through the network ------------------------------------------------------------------------

N2_FLAGS = {
    "v1": dict(move_graph_heads=0),
    "b9": dict(),
    "all": dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, rope_per_head=True),
    "no_table": dict(displacement_bias=False),
    "regs_self_mg4": dict(masked_head_registers="self", move_graph_heads=4),
}


@pytest.mark.parametrize("reading", ["lineage", "creation"])
@pytest.mark.parametrize("flags", list(N2_FLAGS), ids=list(N2_FLAGS))
def test_every_attention_path_computes_the_same_scope(flags, reading):
    """dense (every mask materialised), explicit, sdpa, chunked and block-sparse agree under
    the block-causal scope, with padding in the batch. "block_sparse_all" gathers every
    block's key selection (``attn_sparse_max_fraction=1.0``): at these sizes the same-square
    head's blocks reach every board and would otherwise fall back to dense plans, so this is
    what exercises the scope on a sparse selection (a mutant that skipped it survived)."""
    batch = collate([synthetic_state(21, 13, n_timelines=4), synthetic_state(22, 5)])
    outs = {}
    variants = {impl: dict(attn_impl=impl) for impl in ("dense", "explicit", "sdpa", "chunked", "block_sparse")}
    variants["block_sparse_all"] = dict(attn_impl="block_sparse", attn_sparse_max_fraction=1.0)
    for name, impl in variants.items():
        cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", causal_past=reading, **impl, **N2_FLAGS[flags])
        outs[name] = build_model(cfg, seed=4).eval()(batch, return_trunk=True)
    ref = outs["dense"]
    valid = batch.valid[..., None]
    for impl, out in outs.items():
        assert torch.allclose(out.trunk * valid, ref.trunk * valid, atol=2e-5), (impl, float(((out.trunk - ref.trunk) * valid).abs().max()))
        assert torch.allclose(out.value_logits, ref.value_logits, atol=1e-4), impl


@pytest.mark.grad
def test_gradients_agree_under_the_scope():
    """Training under the scope (section 7.7: the mask is a training-time commitment): the
    fast paths' gradients equal the dense reference's."""
    batch = collate([synthetic_state(31, 10, n_timelines=3)])
    grads = {}
    for impl in ("dense", "block_sparse", "chunked"):
        model = build_model(
            NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", attn_impl=impl, typed_edge_bias=True), seed=2
        )
        out = model(batch)
        (out.value_logits.sum() + out.policy_logits[out.move_valid].sum()).backward()
        grads[impl] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    for impl in ("block_sparse", "chunked"):
        for name, g in grads["dense"].items():
            assert torch.allclose(grads[impl][name], g, atol=1e-4, rtol=1e-4), (impl, name)


def test_no_superseded_board_means_dense():
    """With every board a head, the block-causal scope is the dense scope: bitwise."""
    for state in (synthetic_state(0, 1), synthetic_state(1, 1, width=6, height=6), synthetic_state(2, 2, even_start=True)):
        assert state.is_head.all()
        batch = collate([state])
        a = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal"), seed=1).eval()(batch, return_trunk=True)
        b = build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=1).eval()(batch, return_trunk=True)
        assert torch.equal(a.trunk, b.trunk)


def _with_board(state: PositionState, board: int, rng) -> PositionState:
    """``state`` with board ``board``'s contents shuffled."""
    piece, colour, moved = state.piece.copy(), state.colour.copy(), state.moved.copy()
    perm = rng.permutation(piece.shape[1])
    piece[board], colour[board], moved[board] = piece[board, perm], colour[board, perm], moved[board, perm]
    return replace(state, piece=piece, colour=colour, moved=moved)


@pytest.mark.parametrize("reading", ["lineage", "creation"])
def test_frozen_tokens_read_only_their_past(reading):
    """Causality, directly. A frozen board's frozen past is closed (an ancestor's ancestor is
    an ancestor; earlier-than is transitive), so a frozen board's output depends on the
    boards of its scope row and on nothing else: changing a head changes no frozen token,
    changing a frozen board changes exactly the frozen boards that read it. Under the dense
    scope both reach everything. (The synthetic move graph would change with the shuffled
    board; it is dropped here and the structure is tested apart.)"""
    rng = np.random.default_rng(0)
    empty = np.zeros((0, 3), dtype=np.int64)
    base = replace(synthetic_state(41, 16, n_timelines=4), edges=empty, moves=empty, opp_moves=empty)
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", causal_past=reading), seed=3).eval()
    dense = build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=3).eval()
    tpb = base.layout.tokens_per_board
    scope = board_scope(base.parent, base.creation, base.is_head, reading)
    a = model(collate([base]), return_trunk=True).trunk[0]
    d0 = dense(collate([base]), return_trunk=True).trunk[0]
    frozen = [int(b) for b in np.flatnonzero(~base.is_head)]
    readers = 0
    for changed in (int(np.flatnonzero(base.is_head)[0]), *frozen[::3]):
        other = _with_board(base, changed, rng)
        b = model(collate([other]), return_trunk=True).trunk[0]
        d1 = dense(collate([other]), return_trunk=True).trunk[0]
        for board in frozen:
            if board == changed:
                continue
            sl = slice(board * tpb, (board + 1) * tpb)
            if scope[board, changed]:
                readers += 1
                assert not torch.allclose(a[sl], b[sl]), (board, changed)
            else:
                assert torch.equal(a[sl], b[sl]), (board, changed)
            assert not torch.allclose(d0[sl], d1[sl]), (board, changed)
    assert readers > 0


def test_structure_opens_live_rows_only():
    """Under the block-causal scope the move graph, the relations head, B11's typed biases and
    B12's smolgen leave frozen query rows alone (``scope.py``); live rows keep them all, plus
    B16's status biases on frozen keys (``status.py``), and every row keeps B15's displacement
    bias - geometry."""
    from tests.net.test_displacement import reference_bias
    from tests.net.test_status import status_reference

    flags = dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, trunk_dtype="fp32", n_layers=1)
    state = synthetic_state(51, 12, n_timelines=3)
    batch = collate([state])
    causal = build_model(NetConfig(attn_scope="block_causal", **flags), seed=5).eval()
    dense = build_model(NetConfig(**flags), seed=5).eval()
    h = torch.randn(1, batch.n_tokens, 256, generator=torch.Generator().manual_seed(0))
    ctx_c = causal.make_context(batch).for_layer(h, 0)
    ctx_d = dense.make_context(batch).for_layer(h, 0)
    rows_c, bias_c = ctx_c.structure.rows(0, batch.n_tokens)
    rows_d, bias_d = ctx_d.structure.rows(0, batch.n_tokens)
    live = ctx_c.scope.live_tokens[0]
    frozen_square = (~live) & (batch.kind[0] == 0)
    roles = causal.cfg.head_roles
    for head, role in enumerate(roles):
        if role in ("move_graph", "relations"):
            # A frozen square reads only itself on an edge-derived head.
            sub = rows_c[0, head][frozen_square]
            assert torch.equal(sub.sum(dim=-1), torch.ones(sub.shape[0], dtype=torch.long))
        else:
            assert torch.equal(rows_c[0, head], rows_d[0, head]), role
        assert torch.equal(rows_c[0, head][live], rows_d[0, head][live]), role
    # Frozen rows keep B15's geometric bias and nothing else; live rows keep everything, and
    # B16's status biases on the frozen boards' keys besides.
    geometric = reference_bias(causal, batch, 0)[0]
    status = status_reference(causal, [state], 0)[0]
    assert torch.allclose(bias_c[0][:, ~live], geometric[:, ~live])
    assert torch.allclose(bias_c[0][:, live], bias_d[0][:, live] + status[:, live])
    assert bool((status[:, live] != 0).any()) and not bool((status[:, ~live] != 0).any())
    assert not torch.allclose(bias_d[0][:, ~live], geometric[:, ~live])
    # The scope itself is ANDed into every implementation's rows.
    allowed, _ = _block_rows(ctx_c, 0, batch.n_tokens)
    assert not torch.any(allowed[0][:, frozen_square][..., live])


@pytest.mark.parametrize("features", ["intrinsic", "current"])
def test_frozen_board_features(features):
    """``frozen_features``: under "intrinsic" a frozen board's embedding ignores
    Delta-t-to-present and the active flag (they change after it is frozen); under
    "current" it carries them; head boards always do; the dense scope is unchanged."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, attn_scope="block_causal", frozen_features=features), seed=6).eval()
    batch = collate([synthetic_state(61, 10, n_timelines=3)])
    shifted = replace(batch, dt=batch.dt + 7.0, active=1 - batch.active)
    a, b = model.embed_tokens(batch), model.embed_tokens(shifted)
    frozen = (batch.is_head == 0) & (batch.board >= 0)
    head = (batch.is_head == 1) & (batch.board >= 0)
    assert not torch.allclose(a[head], b[head])
    if features == "intrinsic":
        assert torch.equal(a[frozen], b[frozen])
    else:
        assert not torch.allclose(a[frozen], b[frozen])
    assert torch.equal(model.embed_tokens(batch, "dense"), model.embed(batch))
