"""C24: the refresh ladder's complexity trigger (spec section 7.9, "Complexity as a
trigger"; register C24, prototype; ``complexity.py``, ``tree.SearchTree.visit``).

What is held: off (the default), nothing changes - decisions, epochs and every evaluation
are bitwise those of rung 1 alone, whatever C24's other parameters say; on, it is add-only
- every node rung 1 refreshes is refreshed, never later - and it refreshes the complex
nodes by its rule (the weighted score at the threshold, the visit gate, never the root,
once per node, within the budget); its signals are the node's structure, its path and its
parent's outputs, never the node's own evaluation; and it is deterministic. A complexity
refresh is rung 1's fork (copy-on-write), at another node.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, synthetic_state
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output
from t5d.net.complexity import (
    ComplexitySignals,
    active_timelines,
    complexity_fires,
    complexity_score,
    effort_difficulty,
    structural_signals,
)
from t5d.net.tree import Multiverse, SearchTree, travel_since_refresh

#: A rule that fires nowhere: every weight zero.
NEVER = dict(complexity_board_weight=0.0, complexity_timeline_weight=0.0, complexity_travel_weight=0.0, complexity_effort_weight=0.0)


def _model(seed: int = 0, layers: int = 1):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=layers, attn_scope="block_causal"), seed=seed).eval()


def _root(seed: int, n: int = 14, timelines: int = 4) -> Multiverse:
    return Multiverse.from_state(synthetic_state(3000 + seed, n, n_timelines=timelines), seed=seed)


def _tree(model, root: Multiverse, **cfg) -> SearchTree:
    return SearchTree(CachedEvaluator(model, CacheConfig(**cfg)), root)


def _jump_targets(u: Multiverse, p: int) -> list[tuple[int, int]]:
    sh = u.heads()[p][1]
    return sorted(k for k in u.boards if k[1] % 2 == sh % 2 and k[1] <= sh and k != u.heads()[p])


def _branching_jump(u: Multiverse) -> tuple:
    for p in sorted(u.heads()):
        past = [k for k in _jump_targets(u, p) if not u.is_head(k)]
        if past:
            return ("jump", p, past[0])
    raise AssertionError("no branching jump")


def _random_moves(u: Multiverse, rng: np.random.Generator, n: int, jump_prob: float = 0.4) -> list[tuple]:
    moves = []
    for _ in range(n):
        ps = sorted(u.heads())
        p = ps[int(rng.integers(len(ps)))]
        targets = _jump_targets(u, p)
        move = ("jump", p, targets[int(rng.integers(len(targets)))]) if targets and rng.random() < jump_prob else ("physical", p)
        moves.append(move)
        u = u.apply(move)
    return moves


def _paths(root: Multiverse, seed: int) -> list[list[tuple]]:
    """Search paths from the root: one through a branching jump (rung 1's node), one through
    a branching jump below a physical move (a deeper rung-1 node), one of physical moves
    only, and random ones of depth three."""
    ps = sorted(root.heads())
    jump = _branching_jump(root)
    rng = np.random.default_rng(seed)
    first = ("physical", ps[0])
    out = [
        [jump, *_random_moves(root.apply(jump), rng, 2)],
        [first, _branching_jump(root.apply(first))],
        [("physical", ps[1]), ("physical", ps[2])],
    ]
    out += [_random_moves(root, rng, 3) for _ in range(3)]
    return out


def _simulate(tree: SearchTree, paths, sims: int, seed: int, evaluate: bool = False) -> list[tuple]:
    """A search's visits: each simulation walks one path from the root, visiting every node
    on it (a node reached for the first time is evaluated first, if ``evaluate``). Returns
    the refreshes, in order: (simulation, path, trigger, visits)."""
    rng = np.random.default_rng(seed)
    events = []
    for i in range(sims):
        path = paths[int(rng.integers(len(paths)))]
        node = tree.root
        if evaluate and node.visits == 0:
            tree.evaluate(node)
        tree.visit(node)
        for depth, move in enumerate(path):
            node = tree.child(node, move)
            if evaluate and node.visits == 0:
                tree.evaluate(node)
            if tree.visit(node):
                events.append((i, tuple(path[: depth + 1]), node.refreshed_by, node.visits))
    return events


def _by_path(tree: SearchTree) -> dict[tuple, object]:
    out = {(): tree.root}

    def walk(node, prefix):
        for child in node.children.values():
            key = (*prefix, child.move)
            out[key] = child
            walk(child, key)

    walk(tree.root, ())
    return out


def _identical(a, b) -> bool:
    names = ("trunk", "value_logits", "policy_logits", "opp_reply_logits", "board_logits", "square_logits", "wdl_logits", "effort_logits")
    names += ("short_term", "value_error")
    return all(torch.equal(getattr(a, n), getattr(b, n)) for n in names)


def _peaked(k: int, bins: int = 16) -> torch.Tensor:
    """Effort logits with (almost) all mass on bin ``k``."""
    logits = torch.full((bins,), -30.0)
    logits[k] = 30.0
    return logits


# --- off: nothing changes ------------------------------------------------------------------------


def test_off_by_default_changes_nothing():
    """The flag is off by default, and off it gates everything: with every other C24
    parameter set to fire everywhere, the decisions, the epochs and every evaluation are
    bitwise those of rung 1 alone, and nothing is recorded on the nodes."""
    assert CacheConfig().complexity_refresh is False
    model = _model(1)
    root = _root(1)
    paths = _paths(root, 1)
    eager = dict(
        complexity_threshold=1e-9,
        complexity_board_weight=10.0,
        complexity_timeline_weight=10.0,
        complexity_travel_weight=10.0,
        complexity_effort_weight=10.0,
        complexity_visit_threshold=1,
        complexity_refresh_budget=1000,
    )
    rung1 = _tree(model, root, node_refresh=True, refresh_visit_threshold=3)
    off = _tree(model, root, node_refresh=True, refresh_visit_threshold=3, **eager)
    events = _simulate(rung1, paths, 40, 1)
    assert events and all(e[2] == "rung1" for e in events)
    assert _simulate(off, paths, 40, 1) == events
    assert off.refreshes == rung1.refreshes and off.complexity_refreshes == 0
    a, b = _by_path(rung1), _by_path(off)
    assert a.keys() == b.keys()
    for key in a:
        na, nb = a[key], b[key]
        assert (na.epoch.key, na.epoch.kind, na.refreshed, na.refreshed_by) == (nb.epoch.key, nb.epoch.kind, nb.refreshed, nb.refreshed_by)
        assert _identical(rung1.evaluate(na), off.evaluate(nb)), key
        assert nb.effort_logits is None
    assert off.evaluator.store.computed == rung1.evaluator.store.computed


# --- on: add-only ----------------------------------------------------------------------------------


def test_add_only_over_rung_1():
    """On, C24 adds refreshes and never removes or delays one: every node rung 1 refreshes
    is refreshed, at the same visit or earlier (a complex branched node may go first), and
    at least one more node is; with a rule that fires nowhere, or a budget of zero, the
    decisions are rung 1's exactly."""
    model = _model(2)
    root = _root(2)
    paths = _paths(root, 2)
    base = dict(node_refresh=True, refresh_visit_threshold=3)
    # Fires at nodes with at least three more boards than the root (a jump and a move):
    # the deeper rung-1 node among them, at two visits where rung 1 waits for three.
    rule = dict(NEVER, complexity_threshold=float(root.state().n_boards + 3), complexity_visit_threshold=2)
    rule["complexity_board_weight"] = 1.0
    rung1 = _simulate(_tree(model, root, **base), paths, 60, 2)
    on_tree = _tree(model, root, **base, complexity_refresh=True, **rule)
    on = _simulate(on_tree, paths, 60, 2)
    first = {e[1]: e for e in on}
    assert len(first) == len(on), "a node refreshes once"
    for sim, path, _trigger, visits in rung1:
        assert path in first, "a rung-1 refresh was removed"
        assert first[path][0] <= sim and first[path][3] <= visits, "a rung-1 refresh was delayed"
    extra = set(first) - {e[1] for e in rung1}
    assert extra and all(first[p][2] == "complexity" for p in extra)
    assert any(first[e[1]][2] == "complexity" for e in rung1), "a complex branched node went first"
    assert on_tree.complexity_refreshes == sum(e[2] == "complexity" for e in on)
    for quiet in (dict(**NEVER), dict(rule, complexity_refresh_budget=0)):
        assert _simulate(_tree(model, root, **base, complexity_refresh=True, **quiet), paths, 60, 2) == rung1


# --- on: its rule ----------------------------------------------------------------------------------


def test_signals_are_the_structure_and_the_path():
    """Board count and active timelines are the node's position's; travel is counted from
    the node's last refresh (its nearest refreshed ancestor, else the root) down to the node,
    branching or not."""
    model = _model(3)
    root = _root(3)
    tree = _tree(model, root)
    n1 = tree.child(tree.root, _branching_jump(root))
    n2 = tree.child(n1, ("physical", sorted(n1.universe.heads())[0]))
    p = next(q for q in sorted(n2.universe.heads()) if _jump_targets(n2.universe, q))
    n3 = tree.child(n2, ("jump", p, _jump_targets(n2.universe, p)[-1]))
    assert [travel_since_refresh(n) for n in (tree.root, n1, n2, n3)] == [0, 1, 1, 2]
    for node in (tree.root, n1, n2, n3):
        s = tree.complexity_signals(node)
        u = node.universe
        assert s.boards == len(u.boards) == node.state.n_boards
        assert s.active_timelines == sum(u._active(q) for q in u.heads()) == active_timelines(node.state)
        assert s.effort is None, "no parent evaluated yet"
    tree.refresh(n1)
    assert [travel_since_refresh(n) for n in (n1, n2, n3)] == [0, 0, 1]
    tree.refresh(n3)
    assert travel_since_refresh(n3) == 0


def test_fires_by_its_rule_at_its_visit_gate():
    """A non-root node refreshes by complexity at its visit gate when its score reaches the
    threshold - exactly at it, not below it - once; the root never does."""
    model = _model(4)
    root = _root(4)
    probe = _tree(model, root)
    child_move = ("physical", sorted(root.heads())[0])
    n = probe.child(probe.root, child_move).state.n_boards
    for threshold, fires in ((float(n), True), (n + 0.5, False)):
        rule = dict(NEVER, complexity_board_weight=1.0, complexity_threshold=threshold, complexity_visit_threshold=4)
        tree = _tree(model, root, complexity_refresh=True, **rule)
        child = tree.child(tree.root, child_move)
        for _ in range(3):
            assert not tree.visit(child)
        assert tree.visit(child) is fires
        assert child.refreshed is fires and child.refreshed_by == ("complexity" if fires else None)
        for _ in range(8):
            assert not tree.visit(child), "once per node"
        assert tree.refreshes == tree.complexity_refreshes == int(fires)
    tree = _tree(model, root, complexity_refresh=True, **dict(NEVER, complexity_board_weight=1.0, complexity_visit_threshold=1))
    for _ in range(5):
        assert not tree.visit(tree.root), "the root's refresh is C3's"
    for field, signal in (("complexity_timeline_weight", "active_timelines"), ("complexity_travel_weight", "travel")):
        tree = _tree(model, root, complexity_refresh=True, **dict(NEVER, complexity_visit_threshold=1, **{field: 1.0}))
        node = tree.child(tree.root, _branching_jump(root))
        value = getattr(tree.complexity_signals(node), signal)
        assert value >= 1
        assert tree.complexity_due(node) is False, "below its visit gate"
        node.visits = 1
        assert tree.complexity_due(node)
        tree = _tree(
            model,
            root,
            complexity_refresh=True,
            **dict(NEVER, complexity_visit_threshold=1, complexity_threshold=value + 1.0, **{field: 1.0}),
        )
        node = tree.child(tree.root, _branching_jump(root))
        node.visits = 1
        assert not tree.complexity_due(node)


def test_the_budget_limits_complexity_refreshes_only():
    """At most ``complexity_refresh_budget`` complexity refreshes per search; rung 1's are not
    counted against it and still happen once it is spent."""
    model = _model(5)
    root = _root(5)
    rule = dict(NEVER, complexity_board_weight=1.0, complexity_visit_threshold=1, complexity_refresh_budget=2)
    tree = _tree(model, root, node_refresh=True, refresh_visit_threshold=1, complexity_refresh=True, **rule)
    quiet = [tree.child(tree.root, ("physical", p)) for p in sorted(root.heads())]
    assert [tree.visit(n) for n in quiet] == [True, True] + [False] * (len(quiet) - 2)
    assert tree.complexity_refreshes == 2
    branched = tree.child(tree.root, _branching_jump(root))
    assert tree.visit(branched) and branched.refreshed_by == "rung1"
    assert tree.refreshes == 3 and tree.complexity_refreshes == 2


def test_the_effort_signal_is_the_parents():
    """The effort signal is the parent's predicted difficulty, never the node's own: with
    effort the only signal, a node fires when its parent predicts a hard search and not when
    only the node itself does - and the decision is made without evaluating the node."""
    model = _model(6)
    root = _root(6)
    rule = dict(NEVER, complexity_effort_weight=1.0 / 8, complexity_visit_threshold=2)
    ps = sorted(root.heads())
    for parent_bin, own_bin, fires in ((15, 0, True), (0, 15, False), (8, 0, True), (7, 15, False)):
        tree = _tree(model, root, complexity_refresh=True, **rule)
        child = tree.child(tree.root, ("physical", ps[0]))
        grandchild = tree.child(child, ("physical", ps[1]))
        tree.root.effort_logits, child.effort_logits = _peaked(15 - parent_bin), _peaked(parent_bin)
        grandchild.effort_logits = _peaked(own_bin)

        def refuse(*args, **kwargs):
            raise AssertionError("the trigger evaluated a node")

        tree.evaluator.evaluate = refuse
        tree.evaluator.live_pass = refuse
        assert tree.complexity_signals(grandchild).effort == parent_bin
        assert not tree.visit(grandchild)
        assert tree.visit(grandchild) is fires, (parent_bin, own_bin)


def test_the_trigger_never_reads_the_nodes_own_evaluation():
    """Evaluating a node records its effort for its children and changes nothing in its own
    signals or decision."""
    model = _model(7)
    root = _root(7)
    rule = dict(complexity_visit_threshold=3, complexity_threshold=1.0)
    tree = _tree(model, root, complexity_refresh=True, **rule)
    tree.evaluate(tree.root)
    node = tree.child(tree.root, ("physical", sorted(root.heads())[0]))
    before = tree.complexity_signals(node)
    assert before.effort == effort_difficulty(tree.root.effort_logits, 0.5)
    out = tree.evaluate(node)
    assert torch.equal(node.effort_logits, out.effort_logits[0])
    assert tree.complexity_signals(node) == before
    for value in (_peaked(0), _peaked(15), torch.full((16,), float("nan"))):
        node.effort_logits = value
        assert tree.complexity_signals(node) == before
        assert tree.complexity_due(node) is False  # under its visit gate either way


# --- the pure rule ---------------------------------------------------------------------------------


def test_score_and_placeholder_defaults():
    """The score is the weighted sum (an absent effort contributes nothing); under the
    placeholder defaults each signal alone reaches the threshold at its stated value."""
    cfg = CacheConfig(complexity_refresh=True)
    zero = ComplexitySignals(0, 0, 0, None)
    assert complexity_score(zero, cfg) == 0.0 and not complexity_fires(zero, cfg)
    for field, at in (("boards", 128), ("active_timelines", 8), ("travel", 2), ("effort", 12)):
        assert complexity_fires(dataclasses.replace(zero, **{field: at}), cfg), field
        assert not complexity_fires(dataclasses.replace(zero, **{field: at - 1}), cfg), field
    assert complexity_fires(ComplexitySignals(64, 4, 0, None), cfg), "a mix fires on the way"
    custom = dataclasses.replace(cfg, complexity_board_weight=0.5, complexity_timeline_weight=2.0, complexity_travel_weight=3.0)
    assert complexity_score(ComplexitySignals(4, 1, 2, 0), dataclasses.replace(custom, complexity_effort_weight=7.0)) == 2.0 + 2.0 + 6.0
    assert complexity_score(ComplexitySignals(4, 1, 2, None), custom) == 10.0
    assert [effort_difficulty(_peaked(k), 0.5) for k in (0, 5, 15)] == [0, 5, 15]
    spread = torch.zeros(16)
    assert effort_difficulty(spread, 0.25) <= effort_difficulty(spread, 0.5) <= effort_difficulty(spread, 0.9)
    state = synthetic_state(3100, 12, n_timelines=3)
    s = structural_signals(state, 2, 5)
    assert s == ComplexitySignals(state.n_boards, int((state.is_head & state.active).sum()), 2, 5)
    assert CacheConfig().complexity_visit_threshold == CacheConfig().refresh_visit_threshold == 50


@pytest.mark.parametrize(
    "bad",
    [
        dict(complexity_visit_threshold=0),
        dict(complexity_refresh_budget=-1),
        dict(complexity_board_weight=-0.1),
        dict(complexity_effort_weight=float("inf")),
        dict(complexity_threshold=0.0),
        dict(complexity_threshold=float("nan")),
        dict(complexity_effort_quantile=0.0),
        dict(complexity_effort_quantile=1.5),
    ],
)
def test_invalid_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        CachedEvaluator(_model(0), CacheConfig(complexity_refresh=True, **bad))


# --- determinism, and the fork ---------------------------------------------------------------------


def test_deterministic():
    """Two searches built alike - fresh models, evaluators and stores - make the same
    decisions at the same visits and evaluate every node bitwise alike, the effort signal
    included (each node evaluated on first reach, as a search expands it)."""
    rule = dict(complexity_threshold=1.0, complexity_visit_threshold=2, complexity_refresh_budget=4)
    runs = []
    for _ in range(2):
        model = _model(8)
        root = _root(8)
        tree = _tree(model, root, node_refresh=True, refresh_visit_threshold=3, complexity_refresh=True, **rule)
        events = _simulate(tree, _paths(root, 8), 40, 8, evaluate=True)
        runs.append((tree, events))
    (t1, e1), (t2, e2) = runs
    assert e1 == e2 and any(e[2] == "complexity" for e in e1)
    n1, n2 = _by_path(t1), _by_path(t2)
    assert n1.keys() == n2.keys()
    for key in n1:
        assert n1[key].epoch.key == n2[key].epoch.key
        assert _signals(t1, n1[key]) == _signals(t2, n2[key])
        assert _identical(t1.evaluate(n1[key]), t2.evaluate(n2[key])), key


def _signals(tree: SearchTree, node) -> ComplexitySignals | None:
    return None if node.parent is None else tree.complexity_signals(node)


def test_a_complexity_refresh_is_rung_1s_fork():
    """A complexity refresh forks the node's subtree copy-on-write, as rung 1's does: the
    subtree evaluates as the two-level mask anchored at the node, everything else bitwise as
    before."""
    model = _model(9, layers=2)
    root = _root(9)
    ps = sorted(root.heads())
    tree = _tree(model, root, complexity_refresh=True, **dict(NEVER, complexity_board_weight=1.0, complexity_visit_threshold=1))
    a = tree.child(tree.root, ("physical", ps[0]))
    node = a
    for move in _random_moves(a.universe, np.random.default_rng(9), 2):
        node = tree.child(node, move)
    tree.child(a, ("physical", ps[1]))
    b = tree.child(tree.root, ("physical", ps[1]))
    tree.child(b, ("physical", ps[2]))
    nodes = list(tree.nodes())
    before = {id(n): tree.evaluate(n) for n in nodes}
    old = tree.root.epoch
    assert tree.visit(a) and a.refreshed_by == "complexity"
    assert a.epoch is not old and a.epoch.parent is old and a.epoch.kind == "node"
    forked = {id(a)} | {id(d) for d in a.descendants()}
    for n in nodes:
        after = tree.evaluate(n)
        if id(n) in forked:
            ref = reference_output(model, n.state, anchor=a.state)
            assert torch.allclose(after.trunk, ref.trunk, atol=2e-5)
            assert torch.allclose(after.value_logits, ref.value_logits, atol=1e-4)
        else:
            assert n.epoch is old and _identical(before[id(n)], after)
