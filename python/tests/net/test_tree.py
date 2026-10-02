"""Stage N3: the search-tree fixture (``tree.py``) - lineage-consistent positions, moves that
never change an old multiverse, move orders that transpose, and epochs carried by nodes."""

from __future__ import annotations

import numpy as np
import pytest

from t5d.net import NetConfig, build_model, synthetic_state
from t5d.net.cache import CacheConfig, CachedEvaluator
from t5d.net.tree import Multiverse, SearchTree


def _root(seed: int = 0, **kw) -> Multiverse:
    kw = {"n_timelines": 4, **kw}
    return Multiverse.from_state(synthetic_state(2000 + seed, 14, **kw), seed=seed)


def test_from_state_round_trip():
    for kw in (dict(), dict(even_start=True), dict(width=5, height=5), dict(regs_per_board=0, global_regs=4)):
        state = synthetic_state(7, 16, n_timelines=5, **kw)
        again = Multiverse.from_state(state).state()
        for name in ("timeline", "half_turn", "is_head", "active", "piece", "colour", "moved", "parent", "creation", "source", "edges"):
            assert np.array_equal(getattr(again, name), getattr(state, name)), (kw, name)
        assert (again.regs_per_board, again.global_regs) == (state.regs_per_board, state.global_regs)


def test_a_physical_move_adds_one_board_and_supersedes_the_head():
    root = _root()
    p = sorted(root.heads())[0]
    head = root.heads()[p]
    child = root.physical(p)
    new = (p, head[1] + 1)
    assert set(child.boards) == set(root.boards) | {new}
    assert child.heads()[p] == new and child.boards[new].parent == head
    assert child.boards[new].creation == root.next_creation and child.next_creation == root.next_creation + 1
    assert root.heads()[p] == head and new not in root.boards, "the old multiverse is unchanged"
    state = child.state()  # validates
    assert not state.is_head[list(sorted(child.boards)).index(head)]


def test_jumps_branch_from_past_boards_and_continue_heads():
    root = _root(1)
    heads = root.heads()
    p = sorted(heads)[-1]
    sh = heads[p][1]
    past = [k for k in sorted(root.boards) if not root.is_head(k) and k[1] % 2 == sh % 2 and k[1] <= sh]
    branch = root.jump(p, past[0])
    assert root.branches(("jump", p, past[0]))
    new_timelines = set(branch.creators) - set(root.creators)
    assert len(new_timelines) == 1
    q = new_timelines.pop()
    mover = 1 if sh % 2 == 0 else 2
    assert branch.creators[q] == mover and (q > 0) == (mover == 1)
    first = branch.boards[(q, past[0][1] + 1)]
    assert first.parent == past[0] and first.creation == root.next_creation + 1
    assert first.source == heads[p], "the arrival records the board its piece departed from"
    assert branch.boards[(p, sh + 1)].parent == heads[p]
    branch.state()  # validates: the origin is superseded, one half-turn earlier
    others = [k for k in heads.values() if k[0] != p and k[1] % 2 == sh % 2 and k[1] <= sh]
    if others:
        target = others[0]
        plain = root.jump(p, target)
        assert not root.branches(("jump", p, target))
        assert set(plain.creators) == set(root.creators)
        assert plain.heads()[target[0]] == (target[0], target[1] + 1)
        plain.state()
    with pytest.raises(ValueError):
        root.jump(p, heads[p])
    with pytest.raises(ValueError):
        root.jump(p, (p, sh + 5))


def test_move_orders_transpose_with_different_creation_orders():
    root = _root(2)
    a, b = sorted(root.heads())[:2]
    one = root.physical(a).physical(b).physical(a)
    two = root.physical(b).physical(a).physical(a)
    assert one.digest() == two.digest()
    s1, s2 = one.state(), two.state()
    for name in ("timeline", "half_turn", "is_head", "active", "piece", "parent", "edges", "moves", "opp_moves"):
        assert np.array_equal(getattr(s1, name), getattr(s2, name)), name
    assert not np.array_equal(s1.creation, s2.creation)
    assert one.physical(a).digest() != two.physical(b).digest()


def test_nodes_carry_epochs_and_forks_reach_only_the_old_epoch():
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, attn_scope="block_causal"), seed=0).eval()
    root = _root(3)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(node_refresh=True)), root)
    ps = sorted(root.heads())
    a = tree.child(tree.root, ("physical", ps[0]))
    a1 = tree.child(a, ("physical", ps[1]))
    a2 = tree.child(a1, ("physical", ps[2]))
    b = tree.child(tree.root, ("physical", ps[1]))
    assert tree.child(tree.root, ("physical", ps[0])) is a
    assert all(n.epoch is tree.root.epoch for n in tree.nodes())
    deep = tree.refresh(a1)
    assert a1.epoch is deep and a2.epoch is deep and a.epoch is tree.root.epoch
    upper = tree.refresh(a)
    assert a.epoch is upper and a1.epoch is deep and a2.epoch is deep, "a deeper fork keeps its own epoch"
    assert b.epoch is tree.root.epoch and upper.parent is tree.root.epoch and deep.parent is tree.root.epoch
    late = tree.child(a, ("physical", ps[3]))
    assert late.epoch is upper
    assert [n for n in tree.nodes()][0] is tree.root and len(list(tree.nodes())) == 6
