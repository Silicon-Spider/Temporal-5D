"""Register A5 under the block-causal scope and the cached path (register C17 as the 2026-11-15
review's decision 7 rules it, settling BUILD_LOG issue 252; spec section 7.3): "piece-centric
destination tokens are live tokens wherever they sit - read only by live tokens, never cached".

A destination token on a superseded board - a travel's landing square on a history board - exists
because of the node's candidate moves (``PositionState.transient``). It reads what any live token
reads, carries a live token's input features, takes no B16 status key bias, is read by no frozen
token and is never stored; a superseded board's entry is its own tokens' - its pieces and its
per-board registers.

* The transient tokens: which they are (empty-cell tokens of superseded boards, A5 only), their
  scope (frozen rows never read them; they read every key), their inputs (A3's live features under
  intrinsic frozen features) and no B16 key bias on them.
* **No frozen token's K/V depends on a destination token**: one multiverse read with two candidate
  move sets - two sets of transient tokens - gives every superseded board the same entry, bit for
  bit, and the same K/V in the block-causal full pass.
* Section 15.2's cached rows, run for A5 (``test_cache.py``'s protocol and constructions: random
  weights at initialisation scale, FP32 tolerances matched to the arithmetic, BF16 and FP8 bounds
  measured, adversarial geometries): (a) C2/C17 - cached = one full pass under the block-causal mask,
  on every geometry under both C20 readings and the creation reading, under section 6's switches,
  at full depth (slow); (b) C20 - supersession recompute is path-independent; (c) C3a - a refresh
  fork leaves sibling subtrees unchanged, and supersession inside a refreshed subtree follows its
  anchor; C3 - at the root the two-level mask reduces to the dense pass; G1 - the BF16 cache is the
  live precision, and the FP8 cache is the FP8 network's own function.
* Every attention path computes one function under A5's block-causal scope.

Every geometry here holds transient tokens at every node (the fixture's synthetic candidate moves
land on superseded boards; asserted where a test depends on it). FP32, 2 layers unless named.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output
from t5d.net.scope import BlockCausalScope
from t5d.net.state import to_piece_centric
from t5d.net.tree import Multiverse, SearchTree
from tests.net.test_cache import GEOMETRIES, R1, R2, _bf16_trees, _close, _gaps, _identical, _jump_targets, _random_moves, _root
from tests.net.test_cache import _transpositions as _orders


def _model(seed: int = 0, dtype: str = "fp32", layers: int = 2, **flags):
    cfg = NetConfig(trunk_dtype=dtype, n_layers=layers, attn_scope="block_causal", piece_centric=True, **flags)
    return build_model(cfg, seed=seed).eval()


def _a5(state):
    """A synthetic position's A5 view (synthetic edges to cells that are no token are dropped)."""
    return to_piece_centric(state, strict=False)


def _transient_root(seed: int = 0, n: int = 16, timelines: int = 5):
    """A synthetic position whose candidate moves land on superseded boards."""
    for s in range(seed, seed + 60):
        state = synthetic_state(s, n, n_timelines=timelines, deep_travel_prob=1.0)
        if _a5(state).transient.any():
            return state
    raise AssertionError("no synthetic position with a destination on a superseded board")


def _without_frozen_destinations(state):
    """The same multiverse with every candidate move that lands on an empty cell of a superseded
    board removed: its A5 view holds no transient token."""
    lay = state.layout
    board, cell = lay.split_square_tokens(state.moves[:, 1])
    keep = state.is_head[board] | (state.piece[board, cell] != 0)
    return replace(state, moves=state.moves[keep], edges=np.zeros((0, 3), dtype=np.int64))


# --- the transient tokens --------------------------------------------------------------------------------


def test_the_transient_tokens_are_the_destinations_on_superseded_boards():
    """``PositionState.transient`` and ``Batch.transient`` agree; a transient token is a square token
    of a superseded board whose cell is empty (an A5 destination); a destination on a head board, a
    piece and a register never are; under A1 there is none; ``board_tokens(frozen=True)`` is the
    board less its transient tokens."""
    s = _transient_root(1)
    p = _a5(s)
    tok = p.token_arrays()
    batch = collate([p])
    t = p.transient
    assert t.any() and torch.equal(batch.transient[0], torch.from_numpy(t))
    assert np.all(tok["kind"][t] == 0) and np.all(tok["piece"][t] == 0) and np.all(tok["is_head"][t] == 0)
    head_dest = (tok["kind"] == 0) & (tok["piece"] == 0) & (tok["is_head"] == 1)
    assert head_dest.any() and not np.any(t & head_dest), "a destination on a head board is a head board's token"
    for b in range(p.n_boards):
        all_b, own_b = p.board_tokens(b), p.board_tokens(b, frozen=True)
        assert set(all_b) - set(own_b) == set(np.flatnonzero(t)) & set(all_b)
    assert not s.transient.any() and not collate([s]).transient.any()


def test_the_scope_hides_transient_tokens_from_frozen_rows():
    """Section 7.3: a transient token "is read only by live tokens, never by a frozen one", its own
    board's included, and "reads what any live token reads" - every valid key. Head rows and the
    global registers read it."""
    model = _model(0)
    p = _a5(_transient_root(2))
    batch = collate([p])
    rows = BlockCausalScope.build(batch, model.cfg).dense()[0, 0]  # (L, L)
    t = batch.transient[0]
    frozen = batch.valid[0] & (batch.board[0] >= 0) & (batch.is_head[0] == 0) & ~t
    live = batch.valid[0] & ~frozen
    assert not rows[frozen][:, t].any(), "a frozen row reads a transient token"
    assert rows[t].all(), "a transient row reads every key"
    assert rows[live][:, t].all(), "every live row reads every transient token"
    scope = BlockCausalScope.build(batch, model.cfg)
    assert torch.equal(scope.live_tokens[0], live)


def test_transient_tokens_carry_live_inputs_and_no_status_bias():
    """Section 7.3: a destination token "carries a live token's input features, its board's current
    status among them ... section 4.1's status key biases are for frozen keys and do not apply to it".
    Under intrinsic frozen features its embedding is the dense scope's (Delta-t and the active flag
    kept); a frozen token's is not; and B16's key entry is none on it."""
    model = _model(1)
    p = _a5(_transient_root(3))
    batch = collate([p])
    t = batch.transient[0]
    frozen = batch.valid[0] & (batch.board[0] >= 0) & (batch.is_head[0] == 0) & ~t
    with torch.no_grad():
        bc, dense = model.embed_tokens(batch, "block_causal")[0], model.embed_tokens(batch, "dense")[0]
    assert torch.equal(bc[t], dense[t])
    moving = frozen & (batch.dt[0] != 0)
    assert moving.any() and not torch.allclose(bc[moving], dense[moving])
    from t5d.net.status import status_index

    idx = status_index(batch, model.status, BlockCausalScope.build(batch, model.cfg).live_tokens)
    assert torch.all(idx.key[0][t] == model.status.n_status) and torch.all(idx.key[0][frozen] < model.status.n_status)
    assert idx.live[0][t].all()


def test_b12_biases_a_transient_row_as_a_live_row():
    """B12's board-pair biases are a live row's (``scope.py``): under A5's block-causal scope a
    transient token's row carries its board pair's entry, as a head's row does, and the frozen rows
    of its own board carry none - B12's rows by token (``structure.py``)."""
    model = _model(7, board_pair_smolgen=True)
    p = _a5(_transient_root(7))
    batch = collate([p])
    ctx = model.make_context(batch)
    h = torch.randn(1, batch.n_tokens, model.cfg.d_model, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        st = ctx.structure.for_layer(h, 0)
        plain = replace(st, smolgen=None, smolgen_in=None, pair_table=None, smolgen_rows=None, smolgen_tokens=None)
        length = batch.n_tokens
        term = st.bias_rows(0, length) - plain.bias_rows(0, length)  # (1, H, L, L): B12's alone
    board = st.board_idx[0]
    want = st.pair_table[0][:, board][:, :, board]  # (H, L, L): every row's board-pair entry
    t = batch.transient[0]
    frozen = batch.valid[0] & (board < st.n_boards) & (batch.is_head[0] == 0) & ~t
    head = batch.valid[0] & (batch.is_head[0] == 1)
    assert t.any() and torch.allclose(term[0][:, t], want[:, t], atol=1e-6) and want[:, t].abs().max() > 0
    assert torch.allclose(term[0][:, head], want[:, head], atol=1e-6)
    assert torch.all(term[0][:, frozen] == 0)


# --- no frozen K/V depends on a destination token ------------------------------------------------------------


@pytest.mark.parametrize("supersession", [R2, R1])
def test_no_frozen_kv_depends_on_a_destination_token(supersession):
    """One multiverse with two candidate move sets - every move, and every move but those landing on
    superseded boards' empty cells - so two sets of transient tokens. Every superseded board's entry
    (K/V at every layer, output, dependencies) is bit for bit the same in both, in the causal epoch;
    in the block-causal full pass every frozen token's K and V at every layer agree to the summation
    order; the transient tokens are there in one and not the other."""
    model = _model(4, supersession_scope=supersession)
    s = _transient_root(4)
    p, q = _a5(s), _a5(_without_frozen_destinations(s))
    assert p.transient.any() and not q.transient.any()
    stores = []
    for view in (p, q):
        ev = CachedEvaluator(model)
        with torch.no_grad():
            ev.evaluate(view, ev.root_epoch(view))
        stores.append(ev.store.entries)
    assert stores[0].keys() == stores[1].keys() and stores[0]
    for k, e in stores[0].items():
        f = stores[1][k]
        assert torch.equal(e.kv, f.kv) and torch.equal(e.out, f.out) and e.deps == f.deps
    captured = []
    for view in (p, q):
        kv = []
        with torch.no_grad():
            model(collate([view]), scope="block_causal", kv_hook=lambda layer, k, v, kv=kv: kv.append((k, v)) or (k, v))
        captured.append(kv)
    for b in np.flatnonzero(~p.is_head):
        tp, tq = torch.from_numpy(p.board_tokens(int(b), frozen=True)), torch.from_numpy(q.board_tokens(int(b), frozen=True))
        for (kp, vp), (kq, vq) in zip(*captured, strict=True):
            assert torch.allclose(kp[0, :, tp], kq[0, :, tq], atol=2e-5) and torch.allclose(vp[0, :, tp], vq[0, :, tq], atol=2e-5)


# --- (a) cached = one full pass ---------------------------------------------------------------------------


def _check_tree(model, root, seed: int, history: bool = False, cfg: CacheConfig | None = None, n: int = 6, what=()):
    """Every node of a random path: cached against the block-causal full pass. Returns the transient
    tokens met."""
    tree = SearchTree(CachedEvaluator(model, cfg), root)
    node, seen = tree.root, 0
    history_from = tree.view(tree.root) if history else None
    for move in [None, *_random_moves(root, np.random.default_rng(seed), n)]:
        if move is not None:
            node = tree.child(node, move)
        view = tree.view(node)
        seen += int(view.transient.sum())
        _close(tree.evaluate(node), reference_output(model, view, history_from=history_from), what=(*what, move))
    return seen


@pytest.mark.parametrize("supersession", [R2, R1])
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a5_cached_equals_one_full_pass(geometry, supersession):
    """(a), C2/C17 under A5: along random search paths - physical moves, jumps that branch and that
    do not, boards superseded inside the search - every node's cached evaluation equals one full pass
    of the A5 network over its A5 position under the block-causal mask, its destination tokens on
    superseded boards live in both. Each geometry and C20 reading a seed of its own."""
    case = list(GEOMETRIES).index(geometry) * 2 + (supersession == R1)
    model = _model(200 + case, supersession_scope=supersession)
    with torch.no_grad():
        seen = _check_tree(model, _root(geometry, case), case, history=supersession == R1, what=(geometry, supersession))
    assert seen > 0, "the path met no transient token"


@pytest.mark.parametrize("geometry", ["standard", "deep_travel", "many_timelines"])
def test_a5_cached_equals_one_full_pass_under_the_creation_reading(geometry):
    """(a) under ``causal_past="creation"``: a frozen board's past changes when an earlier head is
    superseded, and the cache recomputes it; the transient tokens are live under it as under any."""
    model = _model(230, causal_past="creation")
    with torch.no_grad():
        assert _check_tree(model, _root(geometry, 30), 30, what=(geometry,)) > 0


@pytest.mark.parametrize(
    "flags",
    [
        dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, rope_per_head=True),
        dict(routing_table=True, masked_head_registers="self", same_board_registers="registers"),
        dict(frozen_features="current"),
        dict(shaw_key_term=True, shaw_init_std=1.0, routing_table=True, typed_edge_bias=True, board_pair_smolgen=True),
        dict(board_layout_embedding=True, routing_table=True, typed_edge_bias=True, board_pair_smolgen=True),
        dict(global_regs=0, regs_per_board=4),
    ],
    ids=["n2-all", "routing-regs-self", "current-features", "b17-n2-all", "a8-n2-all", "no-global-registers"],
)
def test_a5_cached_equals_one_full_pass_with_the_switches(flags):
    """(a) under A5 with section 6's structure - the move-graph and relations masks, B11's typed
    edges and B12's smolgen open and bias live rows, a transient token's among them (B12's rows by
    token, ``structure.py``) - and with B17, A8, the frozen features' test arm, and no global
    registers (the acting colour and the turn state then reach the live board tokens, a transient
    token's among them: issue 225's placeholder)."""
    model = _model(240, **flags)
    regs = (flags.get("regs_per_board", 2), flags.get("global_regs", 16))
    state = synthetic_state(1240, 14, n_timelines=4, regs_per_board=regs[0], global_regs=regs[1])
    with torch.no_grad():
        assert _check_tree(model, Multiverse.from_state(state, seed=3), 3, what=tuple(flags)) > 0


@pytest.mark.slow
def test_a5_cached_equals_one_full_pass_at_full_depth():
    """(a) with the F1 network's 16 layers (the v1 default, B15's far tables on) under A5."""
    model = _model(9, layers=16)
    root = Multiverse.from_state(synthetic_state(1009, 40, n_timelines=7), seed=9)
    with torch.no_grad():
        assert _check_tree(model, root, 9, n=4) > 0


@pytest.mark.parametrize("impl", ["dense", "explicit", "sdpa", "chunked", "flex"])
def test_every_attention_path_is_one_function_under_a5(impl):
    """A5's block-causal forward through every attention implementation equals the block-sparse
    default (FP32, the summation order apart), with the structure's masks and biases on - the scope's
    transient slot read by each path's own mask and, for flex, its block mask."""
    flags = dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)
    base, other = _model(250, **flags), _model(250, attn_impl=impl, **flags)
    states = [_a5(_transient_root(5, n=10, timelines=4)), _a5(_transient_root(9, n=8, timelines=3))]
    with torch.no_grad():
        for batch in [collate([s]) for s in states] + ([collate(states)] if impl != "flex" else []):
            a = base(batch, return_trunk=True).trunk
            b = other(batch, return_trunk=True).trunk
            assert torch.allclose(a[batch.valid], b[batch.valid], atol=2e-5), (impl, float((a - b)[batch.valid].abs().max()))


# --- (b) supersession recompute: no path dependence -----------------------------------------------------


@pytest.mark.parametrize("supersession", [R2, R1])
def test_a5_supersession_recompute_is_path_independent(supersession):
    """(b), C20 under A5: two move orders reaching one multiverse - each in its own store - give one
    A5 position (its candidate moves seeded by the content) and bit-for-bit one evaluation, equal to
    the full pass."""
    model = _model(260, supersession_scope=supersession)
    root = _root("standard", 0)
    with torch.no_grad():
        for first, second in _orders(root):
            ends = []
            for moves in (first, second):
                tree = SearchTree(CachedEvaluator(model), root)
                node = tree.path(moves)
                for n in [node.parent, node]:  # on the way, so the stores' histories differ
                    tree.evaluate(n)
                ends.append((tree, node, tree.evaluate(node)))
            (t1, n1, o1), (_, n2, o2) = ends
            assert n1.universe.digest() == n2.universe.digest()
            assert _identical(o1, o2), (first, second)
            history_from = t1.view(t1.root) if supersession == R1 else None
            _close(o1, reference_output(model, t1.view(n1), history_from=history_from))


# --- (c) refresh forks, the root refresh ---------------------------------------------------------------


def _forked(model, root_refresh: bool, seed: int = 6):
    root = _root("standard", seed)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=root_refresh, node_refresh=True, refresh_visit_threshold=3)), root)
    ps = sorted(root.heads())
    past = [k for k in _jump_targets(root, ps[0]) if not root.is_head(k)]
    a = tree.child(tree.root, ("jump", ps[0], past[0]))
    b = tree.child(tree.root, ("physical", ps[1]))
    for parent in (a, b):
        node = parent
        for move in _random_moves(parent.universe, np.random.default_rng(seed + len(tree.root.children)), 3, jump_prob=0.3):
            node = tree.child(node, move)
    return tree, a, b


@pytest.mark.parametrize("root_refresh", [False, True], ids=["causal", "root-refresh"])
def test_a5_refresh_fork_leaves_siblings_unchanged(root_refresh):
    """(c), C3a under A5: a node refresh forks a new epoch for its subtree, copy-on-write - the
    sibling subtree's evaluations bit for bit unchanged, the parent epoch's entries untouched, the
    forked subtree the two-level mask anchored at it (the refresh's dense pass reads the anchor's
    destination tokens; its entries hold the boards' own tokens)."""
    model = _model(270)
    with torch.no_grad():
        tree, a, b = _forked(model, root_refresh)
        store = tree.evaluator.store
        nodes = list(tree.nodes())
        before = {id(n): tree.evaluate(n) for n in nodes}
        old = tree.root.epoch
        snapshot = {k: (e.kv.clone(), e.out.clone()) for k, e in store.entries.items() if k[0] == old.key}
        new = tree.refresh(a)
        after = {id(n): tree.evaluate(n) for n in nodes}
    assert tree.view(a).transient.any()
    forked = {id(a)} | {id(d) for d in a.descendants()}
    for n in nodes:
        if id(n) in forked:
            _close(after[id(n)], reference_output(model, tree.view(n), anchor=tree.view(a)))
        else:
            assert _identical(before[id(n)], after[id(n)])
    assert set(store.epoch_keys(old.key)) == set(snapshot)
    for k, (kv, out) in snapshot.items():
        assert torch.equal(store[k].kv, kv) and torch.equal(store[k].out, out)
    assert all(k[0] in (old.key, new.key) for k in store.entries)


def test_a5_supersession_inside_a_refreshed_subtree():
    """Below a node refresh, boards superseded inside the search are recomputed against the
    refreshed epoch's history - the two-level mask anchored at the refreshed node - under A5 too."""
    model = _model(280)
    root = _root("deep_travel", 7)
    with torch.no_grad():
        tree = SearchTree(CachedEvaluator(model, CacheConfig(node_refresh=True)), root)
        start = tree.path(_random_moves(root, np.random.default_rng(7), 2))
        tree.refresh(start)
        node = start
        for move in _random_moves(start.universe, np.random.default_rng(57), 5, jump_prob=0.4):
            node = tree.child(node, move)
            _close(tree.evaluate(node), reference_output(model, tree.view(node), anchor=tree.view(start)), what=move)
    assert tree.evaluator.store.computed["supersession"] > 0


@pytest.mark.parametrize("supersession", [R2, R1])
def test_a5_root_refresh_reduces_to_dense_at_the_root(supersession):
    """C3 under A5: with a root refresh the root's cached evaluation - its transient tokens live
    queries against the refreshed history - is the dense pass over its A5 position; below it, the
    two-level reference anchored at the root."""
    model = _model(290, supersession_scope=supersession)
    root = _root("many_timelines", 10)
    with torch.no_grad():
        tree = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=True)), root)
        view = tree.view(tree.root)
        assert view.transient.any()
        _close(tree.evaluate(tree.root), model(collate([view]), scope="dense", return_trunk=True))
        node = tree.path(_random_moves(root, np.random.default_rng(10), 4))
        _close(tree.evaluate(node), reference_output(model, tree.view(node), anchor=view))


# --- (e) storage precision (G1, C15) -------------------------------------------------------------------------


def test_a5_bf16_cache_is_the_live_precision():
    """(e), G1 under A5: the BF16 cache's evaluation is as far from the FP32 full pass as the BF16
    full pass is (``test_cache.py``'s bounds, measured there for A1)."""
    for seed in (0, 1):
        m16, m32 = _model(seed, "bf16"), _model(seed, "fp32")
        root, moves = _bf16_trees(seed)
        with torch.no_grad():
            tree = SearchTree(CachedEvaluator(m16), root)
            node = tree.root
            for move in [None, *moves]:
                if move is not None:
                    node = tree.child(node, move)
                view = tree.view(node)
                cached, live, exact = tree.evaluate(node), reference_output(m16, view), reference_output(m32, view)
                g_cached, g_live = _gaps(cached, exact), _gaps(live, exact)
                assert g_cached["trunk_rms"] < 1.3 * g_live["trunk_rms"] + 3e-4, (g_cached, g_live)
                g = _gaps(cached, live)
                assert g["trunk_rms"] < 6e-3 and g["trunk_max"] < 4e-2 and g["value"] < 5e-3 and g["policy"] < 2.5e-2, g


def test_a5_fp8_cache_equals_the_fp8_network():
    """(e), C15 under A5, at FP32 resolution and in BF16: the FP8 cache serves the FP8 network's own
    function - its forward rounds every superseded board's own tokens' K/V through FP8, one scale per
    board and head over those tokens, and leaves the transient tokens live (``model.py``) - within
    ``test_cache.py``'s bounds; the stored entries are FP8."""
    for seed in (0, 1, 2):
        root, moves = _bf16_trees(seed)
        bounds = (("fp32", dict(trunk_rms=4e-4, value=5e-5, policy=5e-3)), ("bf16", dict(trunk_rms=8e-3, value=6e-3, policy=4e-2)))
        for dtype, bound in bounds:
            model = _model(seed, dtype, frozen_kv_format="fp8")
            with torch.no_grad():
                tree = SearchTree(CachedEvaluator(model), root)
                node = tree.root
                for move in [None, *moves]:
                    if move is not None:
                        node = tree.child(node, move)
                    g = _gaps(tree.evaluate(node), model(collate([tree.view(node)]), return_trunk=True))
                    assert all(g[k] < v for k, v in bound.items()), (seed, dtype, g)
            entry = next(iter(tree.evaluator.store.entries.values()))
            assert entry.kv.dtype == torch.float8_e4m3fn and entry.scale is not None


# --- the refusals lifted -----------------------------------------------------------------------------------------


def test_what_issue_252_refused_is_accepted():
    """Decision 7 lifts issue 252's refusals: the configuration, a block-causal forward of an A5
    network, the cached evaluator and the cache bench take A5 (D4's reply prototype is still refused
    with it - issue 251)."""
    NetConfig(piece_centric=True, attn_scope="block_causal").validate()
    with pytest.raises(NotImplementedError, match="issue 251"):
        NetConfig(piece_centric=True, opp_reply=True).validate()
    dense_net = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=True), seed=0).eval()
    p = _a5(_transient_root(6, n=8, timelines=3))
    with torch.no_grad():
        dense_net(collate([p]), scope="block_causal", status_bias=False)
    model = _model(0, layers=1)
    ev = CachedEvaluator(model)
    with torch.no_grad():
        ev.evaluate(p, ev.root_epoch(p))
    from t5d.net.cache_bench import _conditions, _setup

    bench_model, state = _setup(8, 3, "fp32", "block_causal", 0, piece_centric=True, n_layers=1)
    assert state.piece_centric
    cond = _conditions(bench_model, state)
    assert cond["tokens_per_board"] is None and cond["live_tokens"] > state.global_regs and cond["frozen_tokens"] > 0
