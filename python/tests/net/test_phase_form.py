"""C26, the phase form (CHECKPOINT 9; spec section 7.10; register C26), without the engine: the
cached path behind ``CacheConfig.phase_form`` (off by default) - its one predicate
(``CacheConfig.mode``), the dense epoch a root under N* gets (``CachedEvaluator.dense_epoch``), and
the search-tree fixture (``tree.py``) under a dense root and a cached one. Section 15.2's row
through the engine's own search is ``tests/engine/test_phase_form.py``.

* The defaults: off, N* = 50 (a placeholder); validation refuses an N* that is no positive integer,
  a switch that is no bool, and the phase form without C3's root refresh, which section 7.10 runs at
  or above N*; the configuration is fixed at construction.
* The dense epoch checks the network's format, as every epoch does; ``release`` drops a root
  refresh the evaluator made and nothing else (BUILD_LOG issues 273, 274).
* The predicate at its boundary: dense exactly under N*; never with the flag off.
* ``root_epoch``: under N* the evaluator's one dense epoch - nothing computed, nothing stored; at or
  above N* the root refresh, bit for bit the cached path's without the phase form.
* A tree rooted under N*: every node - those its moves take to N* boards or more included, the
  trap a per-node switch falls into - is the network's dense pass over its own position, bit for
  bit, in the caller's grad mode; nothing refreshes, whatever the ladder's flags say; nothing
  reaches the store, and ``prepare`` and ``refresh`` refuse the dense epoch. The same tree rooted at
  N* is the cached path's bit for bit and the two-level reference's at the FP32 tolerance, and
  rung 1 fires in it as without the phase form.

FP32, 2 layers, random weights at initialisation scale (``test_cache.py``'s networks and
geometries); bitwise where two paths run the same code on the same input.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, state_digest
from t5d.net.model import NetOutput
from t5d.net.tree import SearchTree
from tests.net.test_cache import GEOMETRIES, R1, R2, _close, _identical, _jump_targets, _model, _random_moves, _root

#: Every refresh trigger on, at its most eager: under N* none may fire.
EAGER_LADDER = dict(
    node_refresh=True,
    refresh_visit_threshold=1,
    complexity_refresh=True,
    complexity_threshold=1e-6,
    complexity_visit_threshold=1,
    complexity_refresh_budget=100,
)


def _phase(n_star: int, **kw) -> CacheConfig:
    """The phase form at N* = ``n_star`` (C3's root refresh on: the phase form requires it)."""
    return CacheConfig(root_refresh=True, phase_form=True, phase_boards=n_star, **kw)


def _bitwise(a: NetOutput, b: NetOutput) -> bool:
    """Every field of two network outputs bit for bit (None where either is None)."""
    for f in dataclasses.fields(NetOutput):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if x is None or y is None:
            if (x is None) != (y is None):
                return False
            continue
        if not torch.equal(x, y):
            return False
    return True


def _empty(store) -> bool:
    return not store.entries and not any(store.computed.values()) and store.hits == 0


# --- the configuration and the predicate -----------------------------------------------------------


def test_off_by_default_and_validated():
    """Off, N* = 50; an N* that is no positive integer is refused (0, -1, a bool, a float), and so is
    the phase form without the root refresh."""
    cfg = CacheConfig()
    assert cfg.phase_form is False and cfg.phase_boards == 50
    model = _model(0)
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="phase_boards"):
            CachedEvaluator(model, CacheConfig(phase_boards=bad))
    with pytest.raises(ValueError, match="root refresh"):
        CachedEvaluator(model, CacheConfig(phase_form=True))
    CachedEvaluator(model, _phase(1))
    CachedEvaluator(model, CacheConfig(phase_boards=np.int64(7)))
    for flag in ("root_refresh", "node_refresh", "complexity_refresh", "phase_form"):
        for bad in ("no", 0, 1, None):
            with pytest.raises(ValueError, match=f"{flag} must be a bool"):
                CachedEvaluator(model, dataclasses.replace(_phase(1), **{flag: bad}))
    CachedEvaluator(model, CacheConfig(root_refresh=np.bool_(True), phase_form=np.bool_(True)))


def test_the_configuration_is_fixed_at_construction():
    """The evaluator's configuration is read-only: ``validate`` ran on it at construction, so one
    swapped in afterwards - the phase form without the root refresh, say - would never be checked."""
    ev = CachedEvaluator(_model(0), _phase(10))
    with pytest.raises(AttributeError):
        ev.config = CacheConfig(phase_form=True)
    assert ev.config == _phase(10)


@pytest.mark.parametrize("call", ["root_epoch", "evaluate"])
def test_the_dense_epoch_checks_the_networks_format(call):
    """``test_quant``'s switched-network refusal in the dense epoch: every epoch and evaluation checks
    the network's format, the dense ones included."""
    model = build_model(NetConfig(n_layers=1, trunk_dtype="fp32", attn_scope="block_causal"), seed=0).eval()
    ev = CachedEvaluator(model, _phase(1000))
    root = _root("standard", 0).state()
    epoch = ev.root_epoch(root)
    assert epoch is ev.dense_epoch
    with torch.no_grad():
        ev.evaluate(root, epoch)
    model.set_trunk_quant("fp8_e4m3")
    try:
        with pytest.raises(RuntimeError, match="format changed"), torch.no_grad():
            if call == "root_epoch":
                ev.root_epoch(root)
            else:
                ev.evaluate(root, epoch)
    finally:
        model.set_trunk_quant("none")


def test_the_one_predicate_at_its_boundary():
    """``mode(n)``: dense exactly when the flag is on and n < N* - at N* - 1, N* and N* + 1, and at the
    default's 49 and 50; cached at any n with the flag off."""
    for n_star in (1, 2, 17, 50, 51):
        cfg = _phase(n_star)
        assert [cfg.mode(n) for n in (n_star - 1, n_star, n_star + 1)] == ["dense", "cached", "cached"], n_star
    default = _phase(50)
    assert (default.mode(49), default.mode(50)) == ("dense", "cached")
    off = CacheConfig(root_refresh=True, phase_boards=50)
    assert {off.mode(n) for n in (0, 1, 49, 50, 10**6)} == {"cached"}
    assert CacheConfig().mode(1) == "cached"


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_root_epoch_by_the_roots_board_count(geometry):
    """Under N* (N* = the root's boards + 1): the evaluator's one dense epoch - no key but "dense", no
    anchor, history or parent - and the store untouched; at N* (the root's boards): the root
    refresh, entry for entry and bit for bit the root refresh of the cached path without the phase
    form."""
    model = _model(1)
    root = _root(geometry, 1).state()
    n = root.n_boards
    below = CachedEvaluator(model, _phase(n + 1))
    epoch = below.root_epoch(root)
    assert epoch is below.dense_epoch and below.root_epoch(root) is epoch
    assert (epoch.key, epoch.kind, epoch.mode, epoch.parent, epoch.anchor, epoch.history, epoch.refreshed) == (
        "dense",
        "dense",
        "dense",
        None,
        None,
        frozenset(),
        False,
    )
    assert _empty(below.store)
    at = CachedEvaluator(model, _phase(n))
    plain = CachedEvaluator(model, CacheConfig(root_refresh=True))
    e, p = at.root_epoch(root), plain.root_epoch(root)
    assert e.kind == "root" and e.mode == "cached" and e.refreshed and e.anchor == state_digest(root) and e.key == p.key
    assert at.store.computed["refresh"] > 0 and at.store.computed == plain.store.computed
    assert at.store.entries.keys() == plain.store.entries.keys()
    for k, entry in at.store.entries.items():
        assert torch.equal(entry.kv, plain.store[k].kv) and torch.equal(entry.out, plain.store[k].out)


# --- the search-tree fixture -----------------------------------------------------------------------


def _paths(root, seed: int) -> list[list[tuple]]:
    """Two random paths of six moves, jumps (branching or not) one in two."""
    rng = np.random.default_rng(seed)
    return [_random_moves(root, rng, 6, jump_prob=0.5) for _ in range(2)]


def _grow(tree: SearchTree, paths) -> list:
    nodes = [tree.root]
    for path in paths:
        node = tree.root
        for move in path:
            node = tree.child(node, move)
            nodes.append(node)
    return nodes


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_tree_rooted_under_n_star_is_the_dense_pass_at_every_node(geometry):
    """Section 15.2's C26 row on the fixture: rooted under N* (N* = the root's boards + 4), every node
    of the tree - those with N* boards or more included - is the network's dense pass over its own
    position, every output bit for bit, in the caller's grad mode; with every refresh trigger at its
    most eager, no visit refreshes; ``refresh`` and ``prepare`` refuse the dense epoch; the store
    stays empty. Two seeds per geometry."""
    for seed in (0, 1):
        model = _model(40 + seed)
        root = _root(geometry, seed)
        n_star = root.state().n_boards + 4
        tree = SearchTree(CachedEvaluator(model, _phase(n_star, **EAGER_LADDER)), root)
        dense = tree.evaluator.dense_epoch
        assert tree.root.epoch is dense
        nodes = _grow(tree, _paths(root, seed))
        assert max(n.state.n_boards for n in nodes) >= n_star, (geometry, seed, "a node at N* or above")
        for node in nodes:
            assert node.epoch is dense
            assert _bitwise(tree.evaluate(node), model(collate([node.state]), scope="dense", return_trunk=True)), (geometry, seed)
            for _ in range(3):
                assert not tree.visit(node)
        assert tree.refreshes == tree.complexity_refreshes == 0 and not any(n.refreshed for n in nodes)
        with torch.enable_grad():
            out = tree.evaluate(nodes[-1])
            want = model(collate([nodes[-1].state]), scope="dense", return_trunk=True)
        assert out.trunk.requires_grad and _bitwise(out, want), "the caller's grad mode"
        with pytest.raises(ValueError, match="no cache to refresh"):
            tree.refresh(nodes[-1])
        with pytest.raises(ValueError, match="dense"):
            tree.evaluator.prepare(nodes[-1].state, dense)
        assert all(n.epoch is dense for n in nodes) and _empty(tree.evaluator.store)


def _branching_first(root) -> tuple:
    for p in sorted(root.heads()):
        past = [k for k in _jump_targets(root, p) if not root.is_head(k)]
        if past:
            return ("jump", p, past[0])
    raise AssertionError("no branching jump")


@pytest.mark.parametrize("supersession", [R2, R1])
def test_a_tree_rooted_at_n_star_is_the_cached_path(supersession):
    """Rooted at N* (N* = the root's boards): the root refresh opens the search, and every node is
    the cached path's without the phase form, bit for bit, and the two-level reference anchored at
    the root at the FP32 tolerance; rung 1 refreshes the branched child at its V-th visit, as without
    the phase form, and its subtree is the two-level reference anchored at it."""
    for geometry, seed in (("standard", 3), ("deep_travel", 4)):
        model = _model(50 + seed, supersession_scope=supersession)
        root = _root(geometry, seed)
        ladder = dict(node_refresh=True, refresh_visit_threshold=3)
        trees = [
            SearchTree(CachedEvaluator(model, cfg), root)
            for cfg in (_phase(root.state().n_boards, **ladder), CacheConfig(root_refresh=True, **ladder))
        ]
        jump = _branching_first(root)
        paths = [[jump, *_random_moves(root.apply(jump), np.random.default_rng(seed), 4)], *_paths(root, seed)]
        grown = [_grow(t, paths) for t in trees]
        assert trees[0].root.epoch.kind == "root"
        for a, b in zip(*grown, strict=True):
            out = trees[0].evaluate(a)
            assert _identical(out, trees[1].evaluate(b)), (geometry, seed)
            _close(out, reference_output(model, a.state, anchor=root.state()), what=(geometry, seed))
        branched = [t.child(t.root, jump) for t in trees]
        for _ in range(2):
            assert [t.visit(n) for t, n in zip(trees, branched, strict=True)] == [False, False]
        assert [t.visit(n) for t, n in zip(trees, branched, strict=True)] == [True, True]
        below = [list(n.descendants())[-1] for n in branched]
        out = trees[0].evaluate(below[0])
        assert _identical(out, trees[1].evaluate(below[1]))
        _close(out, reference_output(model, below[0].state, anchor=branched[0].state))


def test_a5_under_the_phase_form():
    """A5 under C26 (BUILD_LOG issue 266; decision 7 settled issue 252): the phase form takes a
    piece-centric network - a root under N* gets the dense epoch, whose evaluation is the network's
    dense pass over the A5 position; at N* the root refresh, whose root evaluation is that same dense
    pass (its destination tokens on superseded boards live)."""
    from t5d.net.state import to_piece_centric

    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, piece_centric=True, attn_scope="block_causal"), seed=0).eval()
    root = to_piece_centric(_root("deep_travel", 0).state(), strict=False)
    assert root.transient.any()
    with torch.no_grad():
        dense = model(collate([root]), scope="dense", return_trunk=True)
    for n_star, kind in ((root.n_boards + 1, "dense"), (root.n_boards, "root")):
        ev = CachedEvaluator(model, _phase(n_star))
        with torch.no_grad():
            epoch = ev.root_epoch(root)
            out = ev.evaluate(root, epoch)
        assert epoch.kind == kind
        if kind == "dense":
            assert _bitwise(out, dense)
        else:
            _close(out, dense)


def test_release_drops_only_a_root_refresh_it_made():
    """``release`` (BUILD_LOG issue 273): a root refresh this evaluator made goes from the store - its
    history and the entries computed under it - and only once; the causal epoch, the dense epoch, a
    node refresh and another evaluator's root refresh in a shared store are kept (issue 274)."""
    model = _model(2)
    root = _root("standard", 2)
    child = root.apply(("physical", sorted(root.heads())[0]))
    ev = CachedEvaluator(model, CacheConfig(root_refresh=True, node_refresh=True))
    other = CachedEvaluator(model, CacheConfig(root_refresh=True), ev.store)
    mine, theirs = ev.root_epoch(root.state()), other.root_epoch(child.state())
    with torch.no_grad():
        ev.evaluate(child.state(), mine)
    node = ev.refresh(child.state(), mine)
    keys = lambda: {k[0] for k in ev.store.entries}  # noqa: E731
    assert keys() == {mine.key, theirs.key, node.key} and ev.store.computed["supersession"] > 0
    n = len(ev.store.epoch_keys(mine.key))
    assert ev.release(theirs) == 0 and ev.release(node) == 0 and ev.release(ev.dense_epoch) == 0
    assert ev.release(mine) == n > 0 and keys() == {theirs.key, node.key} and ev.release(mine) == 0
    causal = CachedEvaluator(model)
    epoch = causal.root_epoch(root.state())
    assert causal.release(epoch) == 0 and {k[0] for k in causal.store.entries} == {"causal"}
