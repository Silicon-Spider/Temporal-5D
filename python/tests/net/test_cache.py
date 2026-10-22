"""Stage N3: section 15.2's equivalence suite, network rows (C2, C3, C3a, C15, C17, C19-C21, G1).

(a) cached attention = one full pass under the same block-causal mask; (b) a supersession
recompute = from scratch under the state-defined mask - no path dependence; (c) a refresh
fork leaves sibling subtrees unchanged (copy-on-write); (d) same state + epoch -> same
evaluation, MCGS transposition hits identical; (e) the BF16 / FP8 cache = live precision.

Protocol (section 15.2): random weights at realistic initialisation scale, several seeds,
tolerances matched to the arithmetic, adversarial constructions. FP32 paths agree to the
last bits that their different summation orders allow (measured: under 2.4e-6 on the
unit-scale trunk output after 16 layers, so 2e-5 is asserted), and paths that compute the
same thing in the same order are compared bitwise. The BF16 and FP8 tests run 2 layers,
and their bounds sit 2.7-8x above the largest gaps measured on these trees at 2 layers
(seeds 0-2); at 16 layers the gaps are up to 1.8x larger. The largest over both depths:
BF16 cached against the BF16 full pass, RMS 1.9e-3 / max 1.3e-2 on the trunk, 1.2e-3 on
value logits, 8.4e-3 on policy logits - the size of the BF16 / FP32 gap itself (2.4e-3
RMS); FP8 cached against the full pass with FP8 frozen K/V, RMS 4.3e-3 / max 3.3e-2,
1.7e-3, 1.6e-2; FP8 against BF16, RMS 8.8e-3 / max 4.8e-2, 1.0e-2, 2.8e-2. (Re-measured
after F5's RMSNorm and F6's Mish, 2026-10-17; the figures before them were within a third
of these. Not re-measured since B15's table, F6's Mish in the heads and M20's defaults - the
per-board registers' positions among them - changed the network; the bounds still hold.)
Adversarial constructions: duplicated boards, deep travel, many timelines,
non-standard board sizes, even starts, register-count arms, and supersession inside a
refreshed subtree.

B16's status key biases (committed with the cached configuration, ``status.py``) are on in every
network here with intrinsic frozen features - the default; the ``frozen_features="current"`` test
arm has none: every row (a)-(e) holds with them, the full pass's live rows carrying the same
biases on the same frozen keys. At a refreshed epoch's anchor neither path adds them (its
evaluation is the refresh pass, section 7.9) - a rule the cached path and ``reference_output``
share, so ``test_status_biases_below_a_refreshed_anchor`` checks it against an oracle of its own.
``test_status_biases_keep_frozen_boards_intrinsic`` checks what they are for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net.cache import BoardKV, CacheConfig, CachedEvaluator, Epoch, KVStore, reference_output, viewed_by
from t5d.net.tree import Multiverse, SearchTree
from tests.net.test_status import status_keys

#: Root positions: adversarial constructions and variant geometries (section 15.2).
GEOMETRIES = {
    "standard": dict(n=14, kw=dict(n_timelines=4)),
    "duplicated": dict(n=14, kw=dict(n_timelines=4, duplicate_prob=0.7)),
    "deep_travel": dict(n=16, kw=dict(n_timelines=6, deep_travel_prob=1.0)),
    "many_timelines": dict(n=18, kw=dict(n_timelines=10)),
    "small_board": dict(n=14, kw=dict(n_timelines=4, width=5, height=5)),
    "even_start": dict(n=14, kw=dict(n_timelines=5, even_start=True)),
    "registers_4_0": dict(n=12, kw=dict(n_timelines=3, regs_per_board=4, global_regs=0)),
}
R1, R2 = "past_and_epoch_history", "causal_past"


def _model(seed: int = 0, dtype: str = "fp32", layers: int = 2, **flags):
    return build_model(NetConfig(trunk_dtype=dtype, n_layers=layers, attn_scope="block_causal", **flags), seed=seed).eval()


def _root(geometry: str, seed: int) -> Multiverse:
    spec = GEOMETRIES[geometry]
    return Multiverse.from_state(synthetic_state(1000 + seed, spec["n"], **spec["kw"]), seed=seed)


def _jump_targets(u: Multiverse, p: int) -> list[tuple[int, int]]:
    sh = u.heads()[p][1]
    return sorted(k for k in u.boards if k[1] % 2 == sh % 2 and k[1] <= sh and k != u.heads()[p])


def _random_moves(u: Multiverse, rng: np.random.Generator, n: int, jump_prob: float = 0.35) -> list[tuple]:
    """A path of physical moves and jumps (branching or not), drawn at random."""
    moves = []
    for _ in range(n):
        ps = sorted(u.heads())
        p = ps[int(rng.integers(len(ps)))]
        targets = _jump_targets(u, p)
        move = ("jump", p, targets[int(rng.integers(len(targets)))]) if targets and rng.random() < jump_prob else ("physical", p)
        moves.append(move)
        u = u.apply(move)
    return moves


def _close(a, b, atol_trunk=2e-5, atol=1e-4, what=""):
    """Every output of two evaluations, FP32 tolerance."""
    assert torch.allclose(a.trunk, b.trunk, atol=atol_trunk), (what, float((a.trunk - b.trunk).abs().max()))
    for name in ("value_logits", "wdl_logits", "moves_left_log", "short_value_logits", "effort_logits", "board_logits"):
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=atol), (what, name)
    for name, valid in (("policy_logits", a.move_valid), ("opp_reply_logits", a.opp_move_valid)):
        assert torch.allclose(getattr(a, name)[valid], getattr(b, name)[valid], atol=atol), (what, name)
    sq = a.square_mask
    assert torch.allclose(a.square_logits[sq], b.square_logits[sq], atol=atol), what


def _identical(a, b) -> bool:
    names = ("trunk", "value_logits", "policy_logits", "opp_reply_logits", "board_logits", "square_logits", "wdl_logits", "effort_logits")
    return all(torch.equal(getattr(a, n), getattr(b, n)) for n in names)


def _gaps(a, b) -> dict[str, float]:
    return {
        "trunk_rms": float((a.trunk - b.trunk).pow(2).mean().sqrt()),
        "trunk_max": float((a.trunk - b.trunk).abs().max()),
        "value": float((a.value_logits - b.value_logits).abs().max()),
        "policy": float((a.policy_logits[a.move_valid] - b.policy_logits[b.move_valid]).abs().max()),
    }


# --- (a) cached = one full pass --------------------------------------------------------------------


@pytest.mark.parametrize("supersession", [R2, R1])
@pytest.mark.parametrize("reading", ["lineage", "creation"])
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_cached_equals_one_full_pass(geometry, reading, supersession):
    """(a): along random search paths - physical moves, jumps that branch and jumps that do
    not, boards superseded inside the search - every node's cached evaluation equals one
    full pass of the model over its position with the block-causal mask. Seeds differ by
    case (each geometry, reading and C20 reading a seed of its own), two for the defaults."""
    case = list(GEOMETRIES).index(geometry) * 4 + 2 * (reading == "creation") + (supersession == R1)
    seeds = (case, 100 + case) if (reading, supersession) == ("lineage", R2) else (case,)
    for seed in seeds:
        model = _model(seed, causal_past=reading, supersession_scope=supersession)
        root = _root(geometry, seed)
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.root
        history_from = root.state() if supersession == R1 else None
        for move in [None, *_random_moves(root, np.random.default_rng(seed), 6)]:
            if move is not None:
                node = tree.child(node, move)
            ref = reference_output(model, node.state, history_from=history_from)
            _close(tree.evaluate(node), ref, what=(geometry, reading, supersession, seed, move))


@pytest.mark.parametrize(
    "flags",
    [
        dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, rope_per_head=True),
        dict(routing_table=True, masked_head_registers="self", same_board_registers="registers"),
        dict(move_graph_heads=4, rope_recenter=False),
        dict(frozen_features="current"),
        dict(norm_placement="post", ffn_activation="squared_relu", routing_table=True, typed_edge_bias=True, board_pair_smolgen=True),
        dict(displacement_bias=False, routing_table=True, typed_edge_bias=True),
        dict(disp_register_pairs="shared", disp_far_edges=(3, 4, 6), disp_line_max=4, disp_init_std=1.0),
        dict(shaw_key_term=True, shaw_init_std=1.0, routing_table=True, typed_edge_bias=True, board_pair_smolgen=True),
        dict(shaw_key_term=True, shaw_shared_heads=True, status_bias_per_layer=True, status_bias_init_std=1.0),
    ],
    ids=[
        "n2-all",
        "routing-regs-self",
        "mg4-raw",
        "current-features",
        "6y-arms-n2",
        "no-table-n2",
        "b15-placeholders",
        "b17-n2-all",
        "b17-shared-b16-per-layer",
    ],
)
def test_cached_equals_one_full_pass_with_every_switch(flags):
    """(a) under the section 6 structure and its register rules, raw coordinates and the frozen
    features' test arm (the registers' positions, M20's - B8 - in every case), under
    ablation 6y's two arms together with the structure on (F5's post-norm - smolgen then reads
    the block's input - and F6's squared ReLU), on B15's no-table arm (ablation 2), and with
    B15's placeholders moved (its table at unit scale), and with B17's key-side Shaw term (a frozen
    board's rows get it from its own queries) and B16's placeholders moved. B15 - the v1 default -
    is on in every other test here: a frozen board's rows carry it, layer by layer
    (``compute_board``)."""
    model = _model(3, **flags)
    root = _root("standard", 3)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.root
    for move in [None, *_random_moves(root, np.random.default_rng(3), 6)]:
        if move is not None:
            node = tree.child(node, move)
        _close(tree.evaluate(node), reference_output(model, node.state), what=move)


#: Ablation 6y's trunk arms (section 13.1) - F5's post-norm with DeepNet initialisation, F6's
#: squared ReLU - each with the factor its arithmetic puts on the v1's BF16 bounds: squared
#: ReLU squares its input's BF16 rounding, and its BF16 gaps are about twice the v1's.
ARMS_6Y = {"post-norm": (dict(norm_placement="post"), 1.0), "squared-relu": (dict(ffn_activation="squared_relu"), 2.0)}


@pytest.mark.parametrize("arm", list(ARMS_6Y))
def test_the_6y_arms_through_the_cached_path(arm):
    """Section 15.2's network rows under ablation 6y's arms - the cached path runs the trunk's
    own block steps, so it computes the arm's function: (a) cached = one full pass along random
    paths (two geometries), and with a root refresh the dense pass at the root and the
    two-level reference below it; (b) two move orders of one multiverse, each in its own
    store, bitwise identical and equal to the full pass; (d) a transposition hit recomputes
    nothing and is identical; (e) the BF16 cache is the live precision: as close to the FP32
    full pass as the BF16 full pass is, and within test_bf16_cache_is_the_live_precision's
    bounds of it times the arm's factor. Largest gaps of the cached evaluation against the
    BF16 full pass on these trees (seeds 0-2, 2 and 16 layers): post-norm RMS 5.3e-4 / max
    6.0e-3, value 1.9e-4, policy 1.9e-3 - 2.2-6.4x below the v1's; squared ReLU RMS 3.7e-3 /
    max 2.2e-2, value 2.1e-3, policy 1.9e-2 - about twice the v1's (at the tests' 2 layers
    3.6-9x below its doubled bounds)."""
    flags, scale = ARMS_6Y[arm]
    for geometry, seed in (("standard", 21), ("deep_travel", 22)):
        model = _model(seed, **flags)
        root = _root(geometry, seed)
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.root
        for move in [None, *_random_moves(root, np.random.default_rng(seed), 6)]:
            if move is not None:
                node = tree.child(node, move)
            _close(tree.evaluate(node), reference_output(model, node.state), what=(arm, geometry, move))
    model = _model(23, **flags)
    root = _root("many_timelines", 23)
    refreshed = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=True)), root)
    _close(refreshed.evaluate(refreshed.root), model(collate([root.state()]), scope="dense", return_trunk=True))
    node = refreshed.path(_random_moves(root, np.random.default_rng(23), 4))
    _close(refreshed.evaluate(node), reference_output(model, node.state, anchor=root.state()))

    model = _model(24, **flags)
    root = _root("standard", 24)
    for first, second in _transpositions(root):
        ends = []
        for moves in (first, second):
            tree = SearchTree(CachedEvaluator(model), root)
            node = tree.path(moves)
            tree.evaluate(node.parent)  # on the way, so the two stores' histories differ
            ends.append((node, tree.evaluate(node)))
        (n1, o1), (n2, o2) = ends
        assert n1.universe.digest() == n2.universe.digest() and _identical(o1, o2), (arm, first, second)
        _close(o1, reference_output(model, n1.state))
    tree = SearchTree(CachedEvaluator(model), root)
    first, second = _transpositions(root)[1]
    a = tree.evaluate(tree.path(first))
    computed = dict(tree.evaluator.store.computed)
    for n in [tree.path(second[:i]) for i in range(1, len(second) + 1)]:
        out = tree.evaluate(n)
    assert tree.evaluator.store.computed == computed, "a transposition recomputes nothing"
    assert _identical(out, a)

    for seed in (0, 1, 2):
        m16, m32 = _model(seed, "bf16", **flags), _model(seed, "fp32", **flags)
        root, moves = _bf16_trees(seed)
        tree = SearchTree(CachedEvaluator(m16), root)
        node = tree.root
        for move in [None, *moves]:
            if move is not None:
                node = tree.child(node, move)
            cached, live, exact = tree.evaluate(node), reference_output(m16, node.state), reference_output(m32, node.state)
            g_cached, g_live = _gaps(cached, exact), _gaps(live, exact)
            assert g_cached["trunk_rms"] < 1.3 * g_live["trunk_rms"] + 3e-4, (arm, g_cached, g_live)
            g = _gaps(cached, live)
            bounds = {"trunk_rms": 6e-3, "trunk_max": 4e-2, "value": 5e-3, "policy": 2.5e-2}
            assert all(g[k] < scale * b for k, b in bounds.items()), (arm, g)


@pytest.mark.slow
def test_cached_equals_one_full_pass_at_full_depth():
    """(a) with the F1 network's 16 layers and its B9 default, at a larger position."""
    model = build_model(NetConfig(trunk_dtype="fp32", attn_scope="block_causal"), seed=9).eval()
    root = Multiverse.from_state(synthetic_state(1009, 40, n_timelines=7), seed=9)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.root
    for move in [None, *_random_moves(root, np.random.default_rng(9), 4)]:
        if move is not None:
            node = tree.child(node, move)
        _close(tree.evaluate(node), reference_output(model, node.state), what=move)


# --- (b) supersession recompute: no path dependence ---------------------------------------------


def _transpositions(root: Multiverse) -> list[tuple[list, list]]:
    """Pairs of move orders that make the same moves - so reach the same multiverse - with
    boards superseded inside the search along the way."""
    ps = sorted(root.heads())
    a, b, c = ps[0], ps[1], ps[2]
    jump = ("jump", a, _jump_targets(root, a)[0])
    return [
        ([("physical", a), ("physical", b)], [("physical", b), ("physical", a)]),
        ([("physical", a), ("physical", a), ("physical", b)], [("physical", b), ("physical", a), ("physical", a)]),
        (
            [("physical", a), ("physical", b), ("physical", a), ("physical", b)],
            [("physical", b), ("physical", a), ("physical", b), ("physical", a)],
        ),
        ([jump, ("physical", c), ("physical", a)], [("physical", c), jump, ("physical", a)]),
    ]


@pytest.mark.parametrize("supersession", [R2, R1])
@pytest.mark.parametrize("geometry", ["standard", "duplicated", "even_start"])
def test_supersession_recompute_is_path_independent(geometry, supersession):
    """(b): two move orders reaching one multiverse - each in its own store, so every board
    superseded on the way is recomputed on each path - give bitwise-identical evaluations
    (lineage reading), equal to the full pass from scratch."""
    for seed in (0, 1) if geometry == "standard" else (2,):
        model = _model(seed, supersession_scope=supersession)
        root = _root(geometry, seed)
        for first, second in _transpositions(root):
            ends = []
            for moves in (first, second):
                tree = SearchTree(CachedEvaluator(model), root)
                node = tree.path(moves)
                for n in [node.parent, node]:  # evaluate on the way, so the stores' histories differ
                    tree.evaluate(n)
                ends.append((node, tree.evaluate(node)))
            (n1, o1), (n2, o2) = ends
            assert n1.universe.digest() == n2.universe.digest()
            assert _identical(o1, o2), (geometry, first, second)
            _close(o1, reference_output(model, n1.state, history_from=root.state() if supersession == R1 else None))


def test_creation_order_reading_is_path_dependent():
    """Spec issue 5, reading (iii), pinned: under ``causal_past="creation"`` two move orders
    that reach one multiverse create its boards in different orders, so a board superseded
    inside the search reads a different past - the evaluations differ (each still equals its
    own full pass). Section 8.5's zero-cost transposition evaluations do not hold for it."""
    model = _model(0, causal_past="creation")
    root = _root("standard", 0)
    first, second = _transpositions(root)[2]
    outs = []
    for moves in (first, second):
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.path(moves)
        out = tree.evaluate(node)
        _close(out, reference_output(model, node.state))
        outs.append((node, out))
    (n1, o1), (n2, o2) = outs
    assert n1.universe.digest() == n2.universe.digest() and not np.array_equal(n1.state.creation, n2.state.creation)
    assert not torch.allclose(o1.trunk, o2.trunk, atol=1e-3)


# --- (d) purity ----------------------------------------------------------------------------------


@pytest.mark.parametrize("root_refresh", [False, True], ids=["causal", "root-refresh"])
def test_same_state_and_epoch_same_evaluation(root_refresh):
    """(d): a node re-evaluated is bitwise the same; so is the same node in a second store;
    and an MCGS transposition hit in one store computes nothing new and returns an identical
    evaluation. The other order's intermediate nodes are evaluated first: under register A7 a
    node whose acting player differs reads its frozen boards from its own side, so its novel
    boards are computed once more for that perspective (spec issue 303) - the transposition
    itself then computes nothing."""
    model = _model(4)
    root = _root("standard", 4)
    config = CacheConfig(root_refresh=root_refresh)
    tree = SearchTree(CachedEvaluator(model, config), root)
    first, second = _transpositions(root)[1]
    node = tree.path(first)
    a = tree.evaluate(node)
    assert _identical(a, tree.evaluate(node))
    other = SearchTree(CachedEvaluator(model, config), root)
    assert _identical(a, other.evaluate(other.path(first)))
    for n in [tree.path(second[:i]) for i in range(1, len(second))]:
        tree.evaluate(n)
    computed = dict(tree.evaluator.store.computed)
    hits = tree.evaluator.store.hits
    out = tree.evaluate(tree.path(second))
    assert tree.evaluator.store.computed == computed, "a transposition recomputes nothing"
    assert tree.evaluator.store.hits > hits
    assert _identical(out, a)


def test_different_epochs_evaluate_differently():
    """Evaluations are a function of (state, epoch): one position, causal and refreshed."""
    model = _model(5)
    root = _root("standard", 5)
    node_moves = _random_moves(root, np.random.default_rng(5), 3)
    outs = []
    for config in (CacheConfig(), CacheConfig(root_refresh=True)):
        tree = SearchTree(CachedEvaluator(model, config), root)
        outs.append(tree.evaluate(tree.path(node_moves)))
    assert not torch.allclose(outs[0].trunk, outs[1].trunk, atol=1e-3)


# --- (c) refresh forks ----------------------------------------------------------------------------


def _forked_tree(model, root_refresh: bool, seed: int = 6):
    """A root with two sibling subtrees: A, reached by a branching jump, and B, by a physical
    move; each with two levels of children that supersede boards."""
    root = _root("standard", seed)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=root_refresh, node_refresh=True, refresh_visit_threshold=3)), root)
    ps = sorted(root.heads())
    past = [k for k in _jump_targets(root, ps[0]) if not root.is_head(k)]
    a = tree.child(tree.root, ("jump", ps[0], past[0]))
    b = tree.child(tree.root, ("physical", ps[1]))
    for parent in (a, b):
        rng = np.random.default_rng(seed + len(tree.root.children))
        node = parent
        for move in _random_moves(parent.universe, rng, 3, jump_prob=0.3):
            node = tree.child(node, move)
        tree.child(parent, ("physical", sorted(parent.universe.heads())[-1]))
    return tree, a, b


@pytest.mark.parametrize("root_refresh", [False, True], ids=["causal", "root-refresh"])
def test_refresh_fork_leaves_siblings_unchanged(root_refresh):
    """(c): a node refresh forks a new epoch for its subtree, copy-on-write: the sibling
    subtree's evaluations are bitwise unchanged, the parent epoch's entries untouched (same
    keys, same tensors), and the forked subtree equals the two-level mask anchored at it."""
    model = _model(6)
    tree, a, b = _forked_tree(model, root_refresh)
    store = tree.evaluator.store
    nodes = list(tree.nodes())
    before = {id(n): tree.evaluate(n) for n in nodes}
    old = tree.root.epoch
    snapshot = {k: (e.kv.clone(), e.out.clone()) for k, e in store.entries.items() if k[0] == old.key}
    new = tree.refresh(a)
    assert a.epoch is new and new.parent is old and new.kind == "node"
    assert all(d.epoch is new for d in a.descendants())
    assert tree.root.epoch is old and b.epoch is old and all(d.epoch is old for d in b.descendants())
    after = {id(n): tree.evaluate(n) for n in nodes}
    forked = {id(a)} | {id(d) for d in a.descendants()}
    for n in nodes:
        if id(n) in forked:
            assert not torch.allclose(before[id(n)].trunk, after[id(n)].trunk, atol=1e-3)
            _close(after[id(n)], reference_output(model, n.state, anchor=a.state))
        else:
            assert _identical(before[id(n)], after[id(n)])
    assert set(store.epoch_keys(old.key)) == set(snapshot), "the parent epoch gained or lost no entry"
    for k, (kv, out) in snapshot.items():
        assert torch.equal(store[k].kv, kv) and torch.equal(store[k].out, out)
    assert all(k[0] in (old.key, new.key) for k in store.entries)


def test_supersession_inside_a_refreshed_subtree():
    """Section 15.2's adversarial construction: below a node refresh, boards superseded inside
    the search are recomputed against the refreshed epoch's history - the two-level mask
    anchored at the refreshed node - not against the root's (section 7.3: that "would
    silently undo the refresh for exactly the boards that subtree creates")."""
    for seed in (7, 8):
        model = _model(seed)
        root = _root("deep_travel", seed)
        tree = SearchTree(CachedEvaluator(model, CacheConfig(node_refresh=True)), root)
        start = tree.path(_random_moves(root, np.random.default_rng(seed), 2))
        tree.refresh(start)
        node = start
        for move in _random_moves(start.universe, np.random.default_rng(seed + 50), 5, jump_prob=0.4):
            node = tree.child(node, move)
            out = tree.evaluate(node)
            _close(out, reference_output(model, node.state, anchor=start.state), what=move)
        assert tree.evaluator.store.computed["supersession"] > 0


@pytest.mark.parametrize("supersession", [R2, R1])
def test_root_refresh_reduces_to_dense_at_the_root(supersession):
    """C3: "at the root the two-level mask reduces to full dense attention": with a root
    refresh, the root's cached evaluation is the dense model's - and below it, the two-level
    reference anchored at the root."""
    model = _model(10, supersession_scope=supersession)
    root = _root("many_timelines", 10)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=True)), root)
    dense = model(collate([root.state()]), scope="dense", return_trunk=True)
    _close(tree.evaluate(tree.root), dense)
    node = tree.path(_random_moves(root, np.random.default_rng(10), 4))
    _close(tree.evaluate(node), reference_output(model, node.state, anchor=root.state()))


def test_rung_1_is_travel_triggered_and_visit_gated():
    """C3a rung 1: only a node reached by a branching travel move refreshes, once, at V visits."""
    model = _model(11)
    tree, a, b = _forked_tree(model, root_refresh=False)
    assert a.branched and not b.branched
    v = tree.evaluator.config.refresh_visit_threshold
    for _ in range(v - 1):
        assert not tree.visit(a)
    assert tree.visit(a) and a.refreshed and tree.refreshes == 1
    for _ in range(2 * v):
        assert not tree.visit(a)
        assert not tree.visit(b)
    off = SearchTree(CachedEvaluator(model, CacheConfig(node_refresh=False, refresh_visit_threshold=1)), tree.root.universe)
    node = off.child(off.root, a.move)
    assert not off.visit(node) and off.refreshes == 0
    assert CacheConfig().refresh_visit_threshold == 50  # section 7.9's V


# --- (e) storage precision -----------------------------------------------------------------------


def _bf16_trees(seed: int):
    root = _root("standard", seed)
    moves = _random_moves(root, np.random.default_rng(seed), 5)
    return root, moves


def test_bf16_cache_is_the_live_precision():
    """(e), G1: the BF16 cache stores K/V exactly as the BF16 trunk computes them (no
    rounding of its own), so a cached BF16 evaluation is as accurate as a live one: its
    distance to the FP32 full pass is the BF16 full pass's own distance (measured: RMS
    2.2-2.4e-3 for both, at 2 and 16 layers). At a refreshed root it is the dense BF16 pass
    up to summation order (bitwise on some positions; RMS under 1e-3 on the others)."""
    for seed in (0, 1, 2):
        m16, m32 = _model(seed, "bf16"), _model(seed, "fp32")
        root, moves = _bf16_trees(seed)
        refreshed = SearchTree(CachedEvaluator(m16, CacheConfig(root_refresh=True)), root)
        dense = m16(collate([root.state()]), scope="dense", return_trunk=True)
        g = _gaps(refreshed.evaluate(refreshed.root), dense)
        assert g["trunk_rms"] < 3e-3 and g["trunk_max"] < 4e-2 and g["value"] < 5e-3 and g["policy"] < 2.5e-2, g
        tree = SearchTree(CachedEvaluator(m16), root)
        for entry in tree.evaluator.store.entries.values():
            assert entry.kv.dtype == torch.bfloat16 and entry.scale is None
        node = tree.root
        for move in [None, *moves]:
            if move is not None:
                node = tree.child(node, move)
            cached, live, exact = tree.evaluate(node), reference_output(m16, node.state), reference_output(m32, node.state)
            g_cached, g_live = _gaps(cached, exact), _gaps(live, exact)
            assert g_cached["trunk_rms"] < 1.3 * g_live["trunk_rms"] + 3e-4, (g_cached, g_live)
            # And within tests/net/test_precision.py's BF16 bounds.
            assert g_cached["trunk_rms"] < 1e-2 and g_cached["trunk_max"] < 6e-2 and g_cached["value"] < 2e-2 and g_cached["policy"] < 3e-2
            g = _gaps(cached, live)
            assert g["trunk_rms"] < 6e-3 and g["trunk_max"] < 4e-2 and g["value"] < 5e-3 and g["policy"] < 2.5e-2, g


def test_fp8_cache_equals_the_fp8_network():
    """(e), C15: with ``frozen_kv_format="fp8"`` the network's ordinary forward - the one it
    would be trained with - rounds every read of a frozen board's K/V through FP8, and the
    FP8 cache serves exactly that function: equal to the plain forward within BF16 noise
    (measured RMS 2.9e-3), at half the BF16 cache's bytes. G1's claim stays with "match"
    (test_bf16_cache_is_the_live_precision); this is the FP8 network's own precision."""
    for seed in (0, 1, 2):
        m8 = _model(seed, "bf16", frozen_kv_format="fp8")
        root, moves = _bf16_trees(seed)
        tree = SearchTree(CachedEvaluator(m8), root)
        node = tree.root
        for move in [None, *moves]:
            if move is not None:
                node = tree.child(node, move)
            g = _gaps(tree.evaluate(node), m8(collate([node.state]), return_trunk=True))
            assert g["trunk_rms"] < 8e-3 and g["trunk_max"] < 1e-1 and g["value"] < 6e-3 and g["policy"] < 4e-2, g
        e8 = next(iter(tree.evaluator.store.entries.values()))
        e16 = next(iter(SearchTree(CachedEvaluator(_model(seed, "bf16")), root).evaluator.store.entries.values()))
        assert e8.kv.dtype == torch.float8_e4m3fn and e8.scale is not None
        assert e16.kv.dtype == torch.bfloat16 and e8.kv_bytes < 0.6 * e16.kv_bytes


def test_fp8_cache_equals_the_fp8_network_at_fp32_resolution():
    """(e), C15, at FP32 resolution: under the FP32 trunk the FP8 cache equals the FP8
    network's ordinary forward up to the rare FP8 rounding flip an FP32 rounding difference
    tips over (measured, the largest over these trees: trunk RMS 7.8e-5, value 4.4e-6, policy
    1.0e-4) - FP8 is the format's rounding, applied identically on both paths, and nothing
    else.

    The flips are discrete and their count heavy-tailed: over 12 seeds each of the F5 and F6
    networks a seed's largest trunk RMS was typically 1e-6 to 3e-5 and at most 7.8e-5, so the
    trunk bound, 4e-4, sits 5x above the largest (raised from 2e-4 with F6, when the margin
    had fallen to 2.6x). The policy bound is a max over logits, so it must absorb one such
    flip: on Linux CI one flip moved a logit by 1.3e-3 (FP32 summation order differs by
    platform). The trunk's RMS is what discriminates - a frozen board reading its own K/V
    unquantised (mutant M14) gives trunk RMS 7.6e-3 (19x the bound), value 3.5e-3, policy
    2.4e-2."""
    for seed in (0, 1, 2):
        model = _model(seed, frozen_kv_format="fp8")
        root, moves = _bf16_trees(seed)
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.root
        for move in [None, *moves]:
            if move is not None:
                node = tree.child(node, move)
            g = _gaps(tree.evaluate(node), model(collate([node.state]), return_trunk=True))
            assert g["trunk_rms"] < 4e-4 and g["value"] < 5e-5 and g["policy"] < 5e-3, g


def test_the_fp8_network_is_a_bounded_approximation_of_match():
    """C15 against G1, measured rather than claimed: the same weights with FP8 frozen K/V
    compute a different function from the "match" network - about 3.8x the BF16-vs-FP32 gap
    (measured RMS 8.8e-3). Bounded, and not zero (FP8 is in use); serving a "match" network
    from an FP8 cache would be this approximation, which is why the cache never does."""
    for seed in (0, 1):
        m8, m16 = _model(seed, "bf16", frozen_kv_format="fp8"), _model(seed, "bf16")
        root, moves = _bf16_trees(seed)
        node = root
        largest = 0.0
        for move in [None, *moves]:
            if move is not None:
                node = node.apply(move)
            batch = collate([node.state()])
            g = _gaps(m8(batch, return_trunk=True), m16(batch, return_trunk=True))
            assert g["trunk_rms"] < 3e-2 and g["value"] < 4e-2 and g["policy"] < 1e-1, g
            largest = max(largest, g["trunk_rms"])
        assert largest > 1e-3, "the FP8 network did not round"


# --- the store: sharing, structure, accounting ---------------------------------------------------


def test_prefix_sharing_computes_each_board_once():
    """Section 7.2: siblings share every inherited entry; a board superseded at a node is
    computed once (per perspective - register A7: the root's history is built from both sides,
    spec issue 303) and shared by its whole subtree; nothing is computed twice."""
    model = _model(12)
    root = _root("standard", 12)
    tree = SearchTree(CachedEvaluator(model), root)
    store = tree.evaluator.store
    n_root = int((~root.state().is_head).sum())
    assert store.computed["history"] == 2 * n_root  # both perspectives (A7)
    ps = sorted(root.heads())
    child = tree.child(tree.root, ("physical", ps[0]))
    tree.evaluate(child)
    assert store.computed["supersession"] == 1  # the old head of timeline ps[0]
    nodes = [child]
    for p in ps[1:]:  # grandchildren: each supersedes one more board, reuses the rest
        nodes.append(tree.child(child, ("physical", p)))
        tree.evaluate(nodes[-1])
    # Each distinct (board, perspective) entry the nodes read is computed exactly once: a
    # grandchild whose move gives the other colour the action reads child's superseded board
    # from that side too (A7).
    keys = set()
    for n in nodes:
        plan = tree.evaluator._plan(n.state, n.epoch)
        keys |= {plan.key(n.epoch, int(b)) for b in plan.frozen}
    novel = len(keys - {k for k, e in store.entries.items() if e.source == "history"})
    assert len(ps) <= novel <= 2 * len(ps) - 1
    assert store.computed["supersession"] == novel
    sibling = tree.child(tree.root, ("physical", ps[1]))
    tree.evaluate(sibling)  # its supersession was computed already, in child's subtree
    assert store.computed["supersession"] == novel
    assert len(store.entries) == 2 * n_root + novel


def test_the_store_is_a_radix_structure():
    """Under the lineage reading each entry's dependencies are the entries of the frozen
    boards of its lineage: the entries of an epoch form the board-derivation tree."""
    model = _model(13)
    root = _root("deep_travel", 13)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.path(_random_moves(root, np.random.default_rng(13), 6))
    tree.evaluate(node)
    plan = tree.evaluator._plan(node.state, node.epoch)
    state = node.state
    for b in plan.frozen:
        entry = tree.evaluator.store[plan.key(node.epoch, int(b))]
        lineage = set()
        a = int(state.parent[b])
        while a >= 0:
            lineage.add(plan.key(node.epoch, a))
            a = int(state.parent[a])
        assert set(entry.deps) == lineage
        assert all(d in tree.evaluator.store for d in entry.deps)


@pytest.mark.parametrize("dtype,fmt,itemsize", [("fp32", "match", 4), ("bf16", "match", 2), ("bf16", "fp8", 1)])
def test_kv_memory_accounting(dtype, fmt, itemsize):
    """Section 15.3's KV memory: per frozen board, layers x 2 x heads x tokens x d_head
    values of the storage format (plus, for FP8, one FP32 scale per layer, K-or-V and head) - per
    perspective: under register A7 the root's history is built from both sides, so the store
    holds two entries per frozen board (spec issue 303)."""
    model = _model(14, dtype, layers=3, frozen_kv_format=fmt)
    state = synthetic_state(1014, 10, n_timelines=3)
    ev = CachedEvaluator(model)
    ev.evaluate(state, ev.root_epoch(state))
    cfg = model.cfg
    per_board = cfg.n_layers * 2 * cfg.n_heads * state.layout.tokens_per_board * cfg.d_head * itemsize
    scales = cfg.n_layers * 2 * cfg.n_heads * 4 if fmt == "fp8" else 0
    n_frozen = int((~state.is_head).sum())
    assert ev.store.kv_bytes() == 2 * n_frozen * (per_board + scales)
    assert ev.store.out_bytes() == 2 * n_frozen * state.layout.tokens_per_board * cfg.d_model * 4


def test_current_features_recompute_when_the_present_moves():
    """Under ``frozen_features="current"`` a move that shifts the present changes every frozen
    board's input, and the cache recomputes them all (still equal to the full pass); under
    "intrinsic" the same move recomputes only the board it supersedes. The move that shifts the
    present gives the other colour the action here, so the parent's boards are first evaluated
    from that side (register A7: each perspective its own entries, spec issue 303)."""
    root = _root("standard", 15)
    state = root.state()
    live = state.is_head & state.active
    at_present = sorted(int(p) for p in state.timeline[live & (state.half_turn == state.h_present)])
    moves = [("physical", p) for p in at_present]  # the last of them moves the present
    counts = {}
    for features in ("intrinsic", "current"):
        model = _model(15, frozen_features=features)
        tree = SearchTree(CachedEvaluator(model), root)
        before_node = tree.path(moves[:-1])
        tree.evaluate(before_node)
        node = tree.child(before_node, moves[-1])
        assert node.state.h_present > before_node.state.h_present
        tree.evaluator.evaluate(viewed_by(before_node.state, node.state.acting_player), before_node.epoch)
        before = tree.evaluator.store.computed["supersession"]
        _close(tree.evaluate(node), reference_output(model, node.state))
        counts[features] = tree.evaluator.store.computed["supersession"] - before
    assert counts["intrinsic"] == 1
    assert counts["current"] == int((~node.state.is_head).sum())


def _present_moves(root: Multiverse) -> list[tuple]:
    """Physical moves on every active head at the present: the last one moves the present."""
    state = root.state()
    live = state.is_head & state.active
    return [("physical", int(p)) for p in sorted(state.timeline[live & (state.half_turn == state.h_present)])]


def _status_of(model, state) -> dict[tuple[int, int], int]:
    """B16's status entry of each frozen board of ``state`` - from the structure its evaluation
    builds - by the board's (timeline, half-turn)."""
    key = model.make_context(collate([state]), scope="block_causal").structure.status_index.key[0]
    tpb = state.layout.tokens_per_board
    return {(int(state.timeline[b]), int(state.half_turn[b])): int(key[b * tpb]) for b in np.flatnonzero(~state.is_head)}


def _activating_jump(root: Multiverse):
    """A branching jump after which a timeline that was inactive - with a frozen board - is active
    (the count rule: the opponent's new timeline makes room for one more of the creator's), and the
    timelines it activates; None if the root has none."""
    state = root.state()
    asleep = {int(p) for p in state.timeline[~state.active & ~state.is_head]}
    for p in sorted(root.heads()):
        for dest in _jump_targets(root, p):
            if root.is_head(dest):
                continue
            after = root.jump(p, dest).state()
            woken = asleep & {int(q) for q in after.timeline[after.active]}
            if woken:
                return ("jump", p, dest), woken
    return None


def test_status_biases_keep_frozen_boards_intrinsic():
    """B16's point (section 4.1): a frozen board's status changes after it freezes - a move that
    shifts the present moves its Delta-t-to-present, a branch can activate its timeline - and
    under the cached configuration that recomputes nothing. Every frozen board the parent holds
    keeps its entry (the same key - its digest holds no status - and the same stored K/V), only
    the board the move supersedes is computed, and the cached evaluation equals the full
    block-causal pass, whose live rows carry the same status biases on the same frozen keys.
    The status of boards both nodes hold did move: the bias moved, not the K/V. (The parent's
    boards are first evaluated from the node's side: under register A7 each perspective has its
    own entries - spec issue 303 - and the move may give the other colour the action.)"""
    model = _model(15)
    assert model.status is not None, "B16 is on in the cached configuration by default"
    root = _root("standard", 15)
    present = _present_moves(root)
    cases = [("present", root, present[:-1], present[-1], None)]
    for seed in range(60):
        other = Multiverse.from_state(synthetic_state(1500 + seed, 14, n_timelines=5), seed=seed)
        found = _activating_jump(other)
        if found is not None:
            cases.append(("activation", other, [], *found))
            break
    assert len(cases) == 2, "the fixture holds a branch that activates a timeline"
    for name, start, prefix, move, woken in cases:
        tree = SearchTree(CachedEvaluator(model), start)
        parent = tree.path(prefix)
        tree.evaluate(parent)
        node = tree.child(parent, move)
        view = viewed_by(parent.state, node.state.acting_player)
        tree.evaluator.evaluate(view, parent.epoch)
        store = tree.evaluator.store
        before = store.computed["supersession"]
        _close(tree.evaluate(node), reference_output(model, node.state), what=name)
        assert store.computed["supersession"] - before == 1, name
        plans = [tree.evaluator._plan(view, parent.epoch), tree.evaluator._plan(node.state, node.epoch)]
        index = [
            {(int(n.state.timeline[b]), int(n.state.half_turn[b])): int(b) for b in p.frozen}
            for n, p in zip((parent, node), plans, strict=True)
        ]
        old, new = _status_of(model, parent.state), _status_of(model, node.state)
        moved = [k for k in index[0] if k in index[1] and old[k] != new[k]]
        assert moved, (name, "a frozen board's status moved")
        for k in index[0]:
            if k in index[1]:
                ka, kb = plans[0].key(parent.epoch, index[0][k]), plans[1].key(node.epoch, index[1][k])
                assert ka == kb and ka in store, (name, k, "the same entry")
        if woken is not None:
            assert node.state.active[[index[1][k] for k in moved if k[0] in woken]].all()
            assert any(old[k] % 2 == 0 and new[k] % 2 == 1 for k in moved if k[0] in woken), "the active flag of its status"


def _refreshed_search(model, refresh: str, seed: int):
    """A search with a refreshed epoch - a root refresh, or a node refresh two moves down - and
    four nodes below its anchor, along a random path of moves and jumps (each supersedes boards,
    so every node holds frozen boards besides the epoch's history)."""
    root = _root("many_timelines" if refresh == "root" else "deep_travel", seed)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=refresh == "root", node_refresh=refresh == "node")), root)
    anchor = tree.root
    if refresh == "node":
        anchor = tree.path(_random_moves(root, np.random.default_rng(seed), 2))
        tree.refresh(anchor)
    nodes, node = [], anchor
    for move in _random_moves(anchor.universe, np.random.default_rng(seed + 50), 4, jump_prob=0.4):
        node = tree.child(node, move)
        nodes.append(node)
    return tree, anchor, nodes


@pytest.mark.parametrize("refresh", ["root", "node"])
def test_status_biases_below_a_refreshed_anchor(refresh):
    """B16 in a refreshed epoch, against an oracle of its own. The cached path and
    ``reference_output`` share one rule - the anchor's own evaluation is the refresh pass and adds
    no B16, and every node below the anchor carries it on every frozen key, the history boards'
    included (issue 216) - so comparing the two cannot test the rule. Here B16's table changes
    instead, between two fresh searches: the anchor's evaluation stays bit for bit as it was, in
    both paths, and every node below the anchor moves, in both. Then only the entries of statuses
    that the history boards hold and no other frozen board of a node does change: the node still
    moves, in both - its live rows read the history boards' keys (whose K/V are the refresh's, with
    no status of the node in them) with the node's own status."""
    seed = 12 if refresh == "root" else 13
    model = _model(seed)
    table = model.status.table.detach().clone()
    noise = torch.randn(table.shape, generator=torch.Generator().manual_seed(seed))

    def evaluations(delta, which=None):
        """With the table moved by ``delta``, in a fresh search (no entry shared with another run):
        the cached evaluation and the reference of the anchor and of each node below it (``which``:
        indices into that list)."""
        model.status.table.copy_(table + delta)
        tree, anchor, nodes = _refreshed_search(model, refresh, seed)
        picked = [anchor, *nodes] if which is None else [[anchor, *nodes][i] for i in which]
        return [(tree.evaluate(n), reference_output(model, n.state, anchor=anchor.state)) for n in picked], anchor, nodes

    try:
        base, anchor, nodes = evaluations(torch.zeros_like(table))
        moved, _, _ = evaluations(noise)
        for path in (0, 1):  # the cached path, the reference
            assert _identical(base[0][path], moved[0][path]), (path, "the anchor: the refresh pass adds no B16")
            for i in range(1, len(base)):
                assert not torch.allclose(base[i][path].trunk, moved[i][path].trunk, atol=1e-4), (path, i, "below the anchor")
        history = {(int(anchor.state.timeline[b]), int(anchor.state.half_turn[b])) for b in np.flatnonzero(~anchor.state.is_head)}
        checked = 0
        for i, node in enumerate(nodes, start=1):
            state, tpb = node.state, node.state.layout.tokens_per_board
            key, _ = status_keys(model, [state])  # written out from the node's position
            held: dict[bool, set[int]] = {True: set(), False: set()}
            for b in np.flatnonzero(~state.is_head):
                held[(int(state.timeline[b]), int(state.half_turn[b])) in history].add(int(key[0, b * tpb]))
            only = sorted(held[True] - held[False])
            if not only:
                continue
            delta = torch.zeros_like(table)
            delta[..., only] = noise[..., only]
            (out,), _, _ = evaluations(delta, [i])
            for path in (0, 1):
                assert not torch.allclose(base[i][path].trunk, out[path].trunk, atol=1e-4), (path, i, "the history boards' keys")
            checked += 1
        assert checked, "a node whose history boards hold a status no other frozen board of it holds"
    finally:
        model.status.table.copy_(table)


@pytest.mark.parametrize("supersession", [R2, R1])
@pytest.mark.parametrize("refresh", ["root", "node"])
def test_current_features_keep_the_epoch_history(refresh, supersession):
    """Under ``frozen_features="current"`` a board's input changes with the present, its
    membership of an epoch's history does not: after a move that shifts the present, a
    refreshed epoch still recognises every history board (none is recomputed) and equals the
    two-level mask anchored at its refresh."""
    model = _model(16, frozen_features="current", supersession_scope=supersession)
    root = _root("standard", 15)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=refresh == "root", node_refresh=True)), root)
    anchor = tree.root
    if refresh == "node":
        anchor = tree.child(tree.root, ("physical", sorted(root.heads())[-1]))
        tree.refresh(anchor)
    node = tree.path(_present_moves(anchor.universe), start=anchor)
    assert node.state.h_present > anchor.state.h_present
    before = tree.evaluator.store.computed["supersession"]
    _close(tree.evaluate(node), reference_output(model, node.state, anchor=anchor.state))
    history = int((~anchor.state.is_head).sum())
    assert tree.evaluator.store.computed["supersession"] - before == int((~node.state.is_head).sum()) - history


def test_current_features_keep_the_causal_history():
    """The same under C20's second reading without a refresh: the root's frozen boards stay
    the epoch's history after the present moves (the cache equals the full pass with that
    history)."""
    model = _model(17, frozen_features="current", supersession_scope=R1)
    root = _root("standard", 15)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.path(_present_moves(root))
    _close(tree.evaluate(node), reference_output(model, node.state, history_from=root.state()))


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_a_store_rooted_elsewhere_gives_the_same_bits(dtype):
    """Every frozen board is computed one at a time in a frame centred on itself, so an entry
    is one function of its digest: the same (state, causal epoch) evaluated in a store that
    started at an earlier root and in one rooted at the state itself is bitwise identical -
    in BF16 too, where a joint and a per-board computation would differ by up to 1.2e-2."""
    model = _model(18, dtype)
    root = _root("deep_travel", 18)
    moves = _random_moves(root, np.random.default_rng(18), 4)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.path(moves)
    a = tree.evaluate(node)
    other = CachedEvaluator(model)
    b = other.evaluate(node.state, other.root_epoch(node.state))
    assert _identical(a, b)


def _alternative_origin(state):
    """A branched timeline's first board and another valid branch origin for it (a superseded
    board one half-turn earlier on a third timeline), or None."""
    for b in range(state.n_boards):
        o = int(state.parent[b])
        p = state.timeline[b]
        if o < 0 or state.timeline[o] == p:
            continue
        for o2 in range(state.n_boards):
            if (
                o2 != o
                and state.half_turn[o2] == state.half_turn[o]
                and state.timeline[o2] != p
                and not state.is_head[o2]
                and state.creation[o2] < state.creation[b]
            ):
                return b, o2
    return None


def test_two_states_differing_only_in_an_origin_get_their_own_entries():
    """Under the lineage reading a branch's first board reads its origin's lineage, so two
    states that differ only in a branch origin - same boards, same contents - evaluate
    differently (an MCGS node key without origins would merge them: build log), and one store
    holds both correctly: the entry key carries the board's past, not only the board."""
    for seed in range(40):
        state = synthetic_state(1100 + seed, 16, n_timelines=6)
        found = _alternative_origin(state)
        if found is not None:
            break
    b, o2 = found
    parent = state.parent.copy()
    parent[b] = o2
    other = dataclasses.replace(state, parent=parent)
    model = _model(19)
    ev = CachedEvaluator(model)
    epoch = ev.root_epoch(state)
    one, two = ev.evaluate(state, epoch), ev.evaluate(other, epoch)
    _close(one, reference_output(model, state))
    _close(two, reference_output(model, other))
    assert not torch.allclose(one.trunk, two.trunk, atol=1e-3)


def test_boards_differing_only_in_has_moved_markers_get_their_own_entries():
    """A rook that went a1-a2-a1 is not an unmoved rook: a frozen board that differs from
    another only in its has-moved markers has its own identity and entry (one store, both
    equal to their full passes)."""
    state = synthetic_state(1200, 14, n_timelines=4)
    frozen = np.flatnonzero(~state.is_head)
    b = int(frozen[len(frozen) // 2])
    sq = int(np.flatnonzero(state.piece[b])[0])
    moved = state.moved.copy()
    moved[b, sq] = not moved[b, sq]
    other = dataclasses.replace(state, moved=moved)
    model = _model(20)
    ev = CachedEvaluator(model)
    epoch = ev.root_epoch(state)
    one, two = ev.evaluate(state, epoch), ev.evaluate(other, epoch)
    _close(one, reference_output(model, state))
    _close(two, reference_output(model, other))
    tpb = state.layout.tokens_per_board
    assert not torch.allclose(one.trunk[0, b * tpb : (b + 1) * tpb], two.trunk[0, b * tpb : (b + 1) * tpb], atol=1e-4)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_cached_equals_one_full_pass_two_parent(geometry):
    """(a) under the two-parent reading (a travel arrival also descends from its departure
    board): the fixture's jumps record their sources."""
    case = list(GEOMETRIES).index(geometry)
    model = _model(300 + case, causal_past="two_parent")
    root = _root(geometry, 300 + case)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.root
    moves = _random_moves(root, np.random.default_rng(300 + case), 6, jump_prob=0.6)
    for move in [None, *moves]:
        if move is not None:
            node = tree.child(node, move)
        _close(tree.evaluate(node), reference_output(model, node.state), what=move)
    assert np.any(node.state.source >= 0), "the path made a jump"


def test_the_evaluator_refuses_what_it_cannot_cache():
    with pytest.raises(ValueError, match="block-causal"):
        CachedEvaluator(build_model(NetConfig(n_layers=1), seed=0))
    with pytest.raises(NotImplementedError, match="board_bilinear"):
        CachedEvaluator(_model(0, board_pair_smolgen=True, smolgen_form="board_bilinear"))
    with pytest.raises(ValueError, match="frozen_kv_format"):
        NetConfig(frozen_kv_format="int4").validate()
    store = KVStore()
    entry = BoardKV(torch.zeros(1), None, torch.zeros(1))
    store.put(("e", b"k"), entry)
    with pytest.raises(RuntimeError, match="never overwritten"):
        store.put(("e", b"k"), entry)
    model = _model(0)
    state = synthetic_state(1016, 8, n_timelines=3)
    refreshed = CachedEvaluator(model, CacheConfig(root_refresh=True))
    epoch = refreshed.root_epoch(state)
    stranger = CachedEvaluator(model, CacheConfig(root_refresh=True), store=KVStore())
    with pytest.raises(RuntimeError, match="history board"):
        stranger.evaluate(state, epoch)
    assert isinstance(epoch, Epoch) and epoch.lineage() == [epoch]
