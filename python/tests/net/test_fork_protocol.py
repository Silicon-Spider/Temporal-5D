"""Section 15.2's C3a row under the full protocol: "A refresh fork leaves sibling subtrees unchanged |
Copy-on-write isolation | C3a".

``test_cache.py::test_refresh_fork_leaves_siblings_unchanged`` holds the row on one root (the
"standard" geometry, seed 6, the default C20 reading). Section 15.2's protocol asks for several
seeds and the adversarial constructions; this module runs the same assertions over them - the
duplicated, deep-travel, many-timeline, small-board and even-start geometries of ``test_cache.py``,
three seeds each, both C20 readings, the causal and the root-refreshed parent epoch - with no
change to the code under test (the gate lane's done-means audit, 2026-10-24).

The assertions are the row's: a node refresh forks a new epoch for its subtree; the sibling
subtree's evaluations are bitwise unchanged, the parent epoch's store entries untouched (same keys,
same tensors), and every node of the forked subtree equals the two-level mask anchored at the
refreshed node (FP32, ``test_cache.py``'s tolerances).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output
from t5d.net.tree import SearchTree
from tests.net.test_cache import R1, R2, _close, _identical, _jump_targets, _model, _random_moves, _root

GEOMETRIES = ("duplicated", "deep_travel", "many_timelines", "small_board", "even_start")
SEEDS = (0, 1, 2)


def _forked_tree(model, geometry: str, root_refresh: bool, seed: int):
    """A root with two sibling subtrees - A, reached by a jump into a past board (it branches),
    and B, by a physical move on another timeline - each two levels deep with boards superseded
    inside it (``test_cache.py``'s construction, on any geometry)."""
    root = _root(geometry, seed)
    cfg = CacheConfig(root_refresh=root_refresh, node_refresh=True, refresh_visit_threshold=3)
    tree = SearchTree(CachedEvaluator(model, cfg), root)
    ps = sorted(root.heads())
    jump = next(((p, k) for p in ps for k in _jump_targets(root, p) if not root.is_head(k)), None)
    assert jump is not None, (geometry, seed, "no past board to jump into")
    p, past = jump
    a = tree.child(tree.root, ("jump", p, past))
    b = tree.child(tree.root, ("physical", next(q for q in ps if q != p)))
    for parent in (a, b):
        rng = np.random.default_rng(seed + len(tree.root.children))
        node = parent
        for move in _random_moves(parent.universe, rng, 3, jump_prob=0.3):
            node = tree.child(node, move)
        tree.child(parent, ("physical", sorted(parent.universe.heads())[-1]))
    return tree, a, b


@pytest.mark.parametrize("supersession", [R2, R1])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_refresh_fork_leaves_siblings_unchanged_under_the_protocol(geometry, supersession):
    """(c) of ``test_cache.py`` on every adversarial geometry, three seeds, both C20 readings; the
    parent epoch causal for even seeds and root-refreshed for odd ones."""
    for seed in SEEDS:
        root_refresh = bool(seed % 2)
        model = _model(20 + seed, supersession_scope=supersession)
        tree, a, b = _forked_tree(model, geometry, root_refresh, seed)
        store = tree.evaluator.store
        nodes = list(tree.nodes())
        before = {id(n): tree.evaluate(n) for n in nodes}
        old = tree.root.epoch
        snapshot = {k: (e.kv.clone(), e.out.clone()) for k, e in store.entries.items() if k[0] == old.key}
        new = tree.refresh(a)
        what = (geometry, supersession, seed, root_refresh)
        assert a.epoch is new and new.parent is old and new.kind == "node", what
        assert all(d.epoch is new for d in a.descendants()), what
        assert tree.root.epoch is old and b.epoch is old and all(d.epoch is old for d in b.descendants()), what
        after = {id(n): tree.evaluate(n) for n in nodes}
        forked = {id(a)} | {id(d) for d in a.descendants()}
        changed = 0
        for n in nodes:
            if id(n) in forked:
                changed += not torch.allclose(before[id(n)].trunk, after[id(n)].trunk, atol=1e-3)
                _close(after[id(n)], reference_output(model, n.state, anchor=a.state), what=what)
            else:
                assert _identical(before[id(n)], after[id(n)]), what
        assert changed == len(forked), (what, "every forked node's evaluation moved with its epoch")
        assert set(store.epoch_keys(old.key)) == set(snapshot), (what, "the parent epoch gained or lost no entry")
        for k, (kv, out) in snapshot.items():
            assert torch.equal(store[k].kv, kv) and torch.equal(store[k].out, out), what
        assert all(k[0] in (old.key, new.key) for k in store.entries), what
