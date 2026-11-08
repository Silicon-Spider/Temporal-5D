"""C27, refresh and supersession inside a turn (CHECKPOINT 9; spec sections 7.3's end and 7.9's
"Inside a turn, refresh only where the network runs"; register C27), without the engine: section
15.2's row "Tree form with the cached configuration on: supersession recompute at the next evaluated
node = recompute at supersession | The deferral is exact - the mask is state-defined | C27".

Under D2's tree form (section 8.2) a simulation passes through inner submove nodes that make no
network call. The search-tree fixture (``tree.py``) marks them: ``SearchTree(evaluates=...)`` takes
the gate decision on each new node's incoming edge, and a node it declines is ``inner``. The cached
path does nothing at an inner node; a board our submoves supersede there is computed when the
network next runs - ``CachedEvaluator.evaluate`` computes every frozen board without an entry - once
per board (with its past and its perspective, register A7) and epoch, and shared from there down.

* T6, the row: along paths with boards superseded in inner segments - the present moving inside one
  (the acting colour changes, so the next evaluated node reads the board from the other side) and,
  where the position has one, a jump that wakes an inactive timeline - the deferred arm (``D``:
  nothing at an inner node) and the eager arm (``S``: at every inner node, every frozen board of its
  position without an entry computed then, from both colours' sides) give every evaluated node bit
  for bit the same evaluation; ``D``'s entries are ``S``'s, tensor for tensor; every evaluated node
  is the from-scratch reference; and the deferral is real: some entry an evaluated node reads was
  computed by ``S`` at an inner node above it and was not in ``D``'s store before that evaluation.
  Every geometry of ``test_cache.py``, both C20 readings, three epochs - causal, a root refresh, and
  a node refresh at the first evaluated node below the root (supersession inside a refreshed
  subtree) - and three markings of the inner nodes, rotated; per commit seed 0 under the lineage
  reading and seed 1 under the two-parent one, the rest of seeds 0-2 x both readings slow.
  Evaluations alone (no deferral asserted) under the creation reading and
  ``frozen_features="current"``, whose entries are not shared across the present's moves.
* T7: once per board and epoch - two evaluated siblings below an inner node share the entry the
  first computes; a sibling acting for the other colour computes its own, once; after a refresh
  above the inner node the entry is computed once more, under the new epoch, and shared again.
* The triggers (C3a rung 1, C24) only at evaluated nodes: an inner node never refreshes, rung 1
  reads a branching travel move on the segment from the nearest evaluated node (per path - issue
  268), C24's travel counts inner travel and its effort is the nearest evaluated ancestor's (issue
  269); an inner node is never evaluated or refreshed. With every node evaluated (``evaluates`` None,
  or always true) the fixture is what it was.

FP32, 2 layers, random weights at initialisation scale (``test_cache.py``'s networks and
tolerances); bitwise between the two arms, which run the same code on the same entries.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.net import synthetic_state
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, viewed_by
from t5d.net.complexity import complexity_fires, effort_difficulty
from t5d.net.tree import Multiverse, SearchTree, evaluated_ancestor, segment_branched, travel_since_refresh
from tests.net.test_cache import (
    GEOMETRIES,
    R1,
    R2,
    _activating_jump,
    _close,
    _identical,
    _jump_targets,
    _model,
    _present_moves,
    _random_moves,
    _root,
)

EPOCHS = ("causal", "root", "node")
MARKINGS = ("segments", "end", "jumps")
#: The largest gaps met against the references (printed at the module's end under ``-s``).
GAPS = {"fp32 trunk": 0.0, "fp32 heads": 0.0}


@pytest.fixture(scope="module", autouse=True)
def _report_gaps():
    yield
    print("\nC27 on the fixture, largest gaps against the references:", {k: f"{v:.2e}" for k, v in GAPS.items()})


def _gap(out, ref) -> None:
    GAPS["fp32 trunk"] = max(GAPS["fp32 trunk"], float((out.trunk - ref.trunk).abs().max()))
    v = out.move_valid
    heads = [(out.value_logits, ref.value_logits), (out.wdl_logits, ref.wdl_logits), (out.policy_logits[v], ref.policy_logits[v])]
    GAPS["fp32 heads"] = max(GAPS["fp32 heads"], *(float((a - b).abs().max()) if a.numel() else 0.0 for a, b in heads))


def _depth(node) -> int:
    d = 0
    while node.parent is not None:
        d, node = d + 1, node.parent
    return d


def _turn_path(root: Multiverse, rng: np.random.Generator) -> tuple[list[tuple], set[int]]:
    """A path from ``root``: one or two random moves, the present's moves (the last one moves the
    present), a jump that wakes an inactive timeline where the position then has one, then random
    moves to 9-12 in all, jumps one in two. With the depths (1-based) of the present's moves and the
    waking jump - the moves a marking keeps inside an inner segment."""
    moves: list[tuple] = []
    u = root
    keep: set[int] = set()

    def play(ms, mark: bool = False):
        nonlocal u
        for m in ms:
            moves.append(m)
            u = u.apply(m)
            if mark:
                keep.add(len(moves))

    play(_random_moves(u, rng, int(rng.integers(1, 3)), jump_prob=0.5))
    play(_present_moves(u), mark=True)
    found = _activating_jump(u)
    if found is not None:
        play([found[0]], mark=True)
    play(_random_moves(u, rng, max(2, int(rng.integers(9, 13)) - len(moves)), jump_prob=0.5))
    return moves, keep


def _marking(name: str, moves: list[tuple], keep: set[int], rng: np.random.Generator, early: bool = False) -> set[int]:
    """The depths (1-based) the network evaluates along a path of ``moves``; the path's end is always
    one (an inner node always has an evaluated node below it), the root (depth 0) always. ``segments``:
    seeded runs, inner ones of 1-4 nodes, the present's moves and the waking jump inner; ``end``: the
    path's end alone; ``jumps``: the jumps (and the end), physical moves inner. ``early``: depth 1
    evaluated too - the node epoch's anchor, so that inner segments lie inside its subtree."""
    last = len(moves)
    first = {1} if early else set()
    if name == "end":
        return first | {last}
    if name == "jumps":
        return first | {d for d in range(1, last + 1) if moves[d - 1][0] == "jump"} | {last}
    evaluated, d = set(), 1
    while d <= last:
        run = int(rng.integers(1, 5))  # an inner run
        d += run
        if d <= last:
            evaluated.add(d)
            d += 1 + int(rng.integers(0, 2))
    evaluated -= keep
    return first | evaluated | {last}


def _config(epoch: str) -> CacheConfig:
    return CacheConfig(root_refresh=epoch == "root", node_refresh=epoch == "node")


def _build(model, root, moves, evaluated: set[int], epoch: str):
    """A tree whose path is ``moves``, the network evaluating the depths ``evaluated`` (and the
    root); with the node epoch, the first evaluated node below the root refreshed before anything
    is evaluated - every node below it, inner ones included, in its epoch."""
    tree = SearchTree(CachedEvaluator(model, _config(epoch)), root, evaluates=lambda parent, move, u: _depth(parent) + 1 in evaluated)
    path = [tree.root]
    for m in moves:
        path.append(tree.child(path[-1], m))
    assert [not n.inner for n in path] == [d == 0 or d in evaluated for d in range(len(path))]
    anchor = None
    if epoch == "node":
        anchor = next(n for n in path[1:] if not n.inner)
        tree.refresh(anchor)
        assert all(n.epoch is anchor.epoch for n in path[path.index(anchor) :]), "the fork reached the inner nodes below it"
    return tree, path, anchor


def _eager(ev: CachedEvaluator, node) -> set:
    """Recompute at supersession: every frozen board of ``node``'s position without an entry, from both
    colours' sides, in the node's epoch. The keys computed."""
    before = set(ev.store.entries)
    state = node.state
    for view in (state, viewed_by(state, 1 - int(state.acting_player))):
        work = ev.prepare(view, node.epoch)
        for b in ev.missing(work):
            ev.compute_board(work, b)
    return set(ev.store.entries) - before


def _reads(ev: CachedEvaluator, node) -> set:
    plan = ev._plan(node.state, node.epoch)
    return {plan.key(node.epoch, int(b)) for b in plan.frozen}


def _reference(model, node, epoch: str, root, anchor, supersession: str):
    if epoch == "root":
        return reference_output(model, node.state, anchor=root.state())
    if epoch == "node" and node.epoch is anchor.epoch:
        return reference_output(model, node.state, anchor=anchor.state)
    return reference_output(model, node.state, history_from=root.state() if supersession == R1 else None)


def _deferral_case(model, root, moves, evaluated, epoch, supersession, what, deferral: bool = True) -> int:
    """T6 on one path: the deferred arm against the eager one. Returns how many evaluated nodes read
    an entry the eager arm computed at an inner node above them and the deferred arm had not stored
    before their evaluation."""
    deferred, d_path, d_anchor = _build(model, root, moves, evaluated, epoch)
    eager, s_path, _ = _build(model, root, moves, evaluated, epoch)
    d_ev, s_ev = deferred.evaluator, eager.evaluator
    at_inner: set = set()
    deferred_reads = 0
    for dn, sn in zip(d_path, s_path, strict=True):
        if sn.inner:
            at_inner |= _eager(s_ev, sn)
            continue
        stored = set(d_ev.store.entries)
        out = deferred.evaluate(dn)
        assert _identical(out, eager.evaluate(sn)), (what, _depth(dn))
        ref = _reference(model, dn, epoch, root, d_anchor, supersession)
        if out.trunk.dtype == torch.float32:
            _gap(out, ref)
        _close(out, ref, what=(what, _depth(dn)))
        deferred_reads += bool((_reads(d_ev, dn) & at_inner) - stored)
    assert d_ev.store.entries.keys() <= s_ev.store.entries.keys(), what
    for k, e in d_ev.store.entries.items():
        f = s_ev.store[k]
        assert torch.equal(e.kv, f.kv) and torch.equal(e.out, f.out) and e.deps == f.deps, what
        assert (e.scale is None and f.scale is None) or torch.equal(e.scale, f.scale), what
    supersessions = sum(1 for e in d_ev.store.entries.values() if e.source == "supersession")
    assert d_ev.store.computed["supersession"] == supersessions, (what, "each board and epoch computed once")
    if deferral:
        assert at_inner, (what, "the eager arm computed boards at inner nodes")
    return deferred_reads


# --- T6: the row ---------------------------------------------------------------------------------------


#: T6's cases, (seed, causal-past reading): per commit seed 0 under the lineage reading and seed 1
#: under the two-parent one; slow, the other reading of each and seed 2 under both - the full matrix.
T6_CASES = ((0, "lineage"), (1, "two_parent"))
T6_SLOW_CASES = ((0, "two_parent"), (1, "lineage"), (2, "lineage"), (2, "two_parent"))


def _t6(geometry: str, epoch: str, cases) -> None:
    for seed, reading in cases:
        for supersession in (R2, R1):
            case = (seed * 4 + 2 * (supersession == R1) + (reading == "two_parent")) % len(MARKINGS)
            marking = MARKINGS[(case + EPOCHS.index(epoch)) % len(MARKINGS)]
            model = _model(600 + seed, causal_past=reading, supersession_scope=supersession)
            root = _root(geometry, seed)
            rng = np.random.default_rng([seed, EPOCHS.index(epoch), reading == "two_parent"])
            moves, keep = _turn_path(root, rng)
            evaluated = _marking(marking, moves, keep, rng, early=epoch == "node")
            what = (geometry, epoch, seed, supersession, reading, marking)
            assert _deferral_case(model, root, moves, evaluated, epoch, supersession, what) > 0, (what, "the deferral is real")


@pytest.mark.parametrize("epoch", EPOCHS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_supersession_deferred_to_the_next_evaluated_node_is_exact(geometry, epoch):
    """Section 15.2's C27 row (see the module docstring): seed 0 under the lineage reading and seed 1
    under the two-parent one, both C20 readings, the marking rotated over the cases."""
    _t6(geometry, epoch, T6_CASES)


@pytest.mark.slow
@pytest.mark.parametrize("epoch", EPOCHS)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_supersession_deferred_to_the_next_evaluated_node_is_exact_full_matrix(geometry, epoch):
    """The rest of T6's matrix: seeds 0 and 1 under their other reading, seed 2 under both."""
    _t6(geometry, epoch, T6_SLOW_CASES)


@pytest.mark.parametrize("flags", [dict(causal_past="creation"), dict(frozen_features="current")], ids=["creation", "current-features"])
def test_the_deferral_under_readings_whose_entries_are_not_shared(flags):
    """The creation reading (a frozen board's past can hold boards live at the inner node) and the
    current frozen features (an entry holds the present it was computed under): the two arms still
    evaluate every evaluated node alike and as the reference does - an eager entry may simply go
    unread - on two geometries and every epoch."""
    for geometry in ("standard", "deep_travel"):
        for epoch in EPOCHS:
            model = _model(640, **flags)
            root = _root(geometry, 0)
            rng = np.random.default_rng([7, EPOCHS.index(epoch)])
            moves, keep = _turn_path(root, rng)
            evaluated = _marking("segments", moves, keep, rng, early=epoch == "node")
            _deferral_case(model, root, moves, evaluated, epoch, R2, (flags, geometry, epoch), deferral=False)


@pytest.mark.slow
def test_the_deferral_in_bf16_and_fp8():
    """The row in BF16, and with the FP8 frozen-K/V format (C15): the two arms bit for bit, on one
    geometry (the reference comparison is FP32's; here the arms are compared with each other)."""
    for flags in (dict(), dict(frozen_kv_format="fp8")):
        model = _model(650, "bf16", **flags)
        root = _root("deep_travel", 0)
        rng = np.random.default_rng(650)
        moves, keep = _turn_path(root, rng)
        evaluated = _marking("segments", moves, keep, rng)
        deferred, d_path, _ = _build(model, root, moves, evaluated, "root")
        eager, s_path, _ = _build(model, root, moves, evaluated, "root")
        for dn, sn in zip(d_path, s_path, strict=True):
            if sn.inner:
                _eager(eager.evaluator, sn)
            else:
                assert _identical(deferred.evaluate(dn), eager.evaluate(sn)), (flags, _depth(dn))


# --- T7: once per board and epoch -------------------------------------------------------------------


def _b_key(ev: CachedEvaluator, node, move) -> tuple:
    """The entry key, in ``node``'s epoch and from its acting player's side, of the board that was the
    head of ``move``'s timeline before ``move`` (the board an inner move superseded; no later move on
    that timeline)."""
    state = node.state
    p = int(move[1])
    boards = sorted((b for b in range(state.n_boards) if int(state.timeline[b]) == p), key=lambda b: int(state.half_turn[b]))
    return ev._plan(state, node.epoch).key(node.epoch, boards[-2])


def test_once_per_board_and_epoch():
    """Root -> A (evaluated) -> I (inner: its move supersedes a head, b) -> E1, E2 (evaluated siblings,
    the present unmoved): nothing is computed at I; E1 computes b's entry, and E2 hits it - computing
    only the board its own move froze. Below I, the present's remaining moves give the other colour
    the action: that node reads b from its side - its own entry - and its transposition (the same
    moves in the other order) computes nothing. After a node refresh at A, an evaluated ancestor above
    I, E1 computes b once more, under the new epoch's key, and E2 hits it again."""
    model = _model(660)
    root = Multiverse.from_state(synthetic_state(6603, 16, n_timelines=8), seed=0)
    present = _present_moves(root)
    assert len(present) >= 4, "four heads at the present"
    tree = SearchTree(CachedEvaluator(model, CacheConfig(node_refresh=True)), root, evaluates=lambda parent, move, u: _depth(parent) != 1)
    a = tree.child(tree.root, present[0])
    i = tree.child(a, present[1])
    e1, e2 = tree.child(i, present[2]), tree.child(i, present[3])
    assert not a.inner and i.inner and not e1.inner and not e2.inner
    acting = int(e1.state.acting_player)
    assert int(e2.state.acting_player) == acting == int(i.state.acting_player)
    ev, store = tree.evaluator, tree.evaluator.store
    key = lambda node: _b_key(ev, node, present[1])  # noqa: E731
    tree.evaluate(tree.root)
    tree.evaluate(a)
    assert key(e1) == key(e2) and key(e1) not in store, "nothing computed at the inner node"
    computed = store.computed["supersession"]
    tree.evaluate(e1)
    assert key(e1) in store and store.computed["supersession"] > computed
    entry, before, hits = store[key(e1)], store.computed["supersession"], store.hits
    tree.evaluate(e2)
    assert store.computed["supersession"] == before + 1 and store.hits > hits, "E2 computed only its own move's board"
    assert store[key(e2)] is entry, "E2 read E1's entry of b"
    rest = present[2:]
    other = tree.path(rest, start=i)
    twin = tree.path([rest[1], rest[0], *rest[2:]], start=i)
    assert other is not twin and other.universe.digest() == twin.universe.digest()
    assert int(other.state.acting_player) != acting and key(other) != key(e1) and key(other) not in store
    tree.evaluate(other)
    assert key(other) in store, "the other colour's entry of b"
    before, hits = dict(store.computed), store.hits
    tree.evaluate(twin)
    assert store.computed == before and store.hits > hits, "its transposition computes nothing"
    old = key(e1)
    tree.refresh(a)
    assert i.epoch is a.epoch and e1.epoch is a.epoch and e2.epoch is a.epoch and key(e1) != old and key(e1) not in store
    computed = store.computed["supersession"]
    tree.evaluate(e1)
    assert key(e1) in store and store.computed["supersession"] > computed
    entry, before, hits = store[key(e1)], store.computed["supersession"], store.hits
    tree.evaluate(e2)
    assert store.computed["supersession"] == before + 1 and store.hits > hits and store[key(e2)] is entry, "shared again"


# --- the triggers: only at evaluated nodes ---------------------------------------------------------


def _branching(u: Multiverse) -> tuple:
    for p in sorted(u.heads()):
        past = [k for k in _jump_targets(u, p) if not u.is_head(k)]
        if past:
            return ("jump", p, past[0])
    raise AssertionError("no branching jump")


def _inner_at(*depths: int):
    """An ``evaluates`` predicate: the nodes at ``depths`` are inner."""
    return lambda parent, move, u: _depth(parent) + 1 not in depths


def test_rung_1_fires_only_at_evaluated_nodes():
    """C3a rung 1 under C27 (V = 3): root -> I (inner, a branching jump) -> E: I never refreshes, E does
    at its third visit, I keeps the old epoch; an evaluated sibling of E below I refreshes at its own
    third visit too (rung 1 per path, issue 268); with every node evaluated, I refreshes instead. Root
    -> I (inner, a physical move) -> E: no refresh. Root -> E1 (evaluated, branching) -> I -> E2: the
    branch is E1's, not on E2's segment - E2 is not eligible."""
    model = _model(670)
    root = _root("standard", 7)
    cfg = CacheConfig(node_refresh=True, refresh_visit_threshold=3)
    jump = _branching(root)
    tree = SearchTree(CachedEvaluator(model, cfg), root, evaluates=_inner_at(1))
    i = tree.child(tree.root, jump)
    heads = sorted(i.universe.heads())
    e, sibling = tree.child(i, ("physical", heads[0])), tree.child(i, ("physical", heads[1]))
    assert i.inner and not e.inner and i.branched and not e.branched and segment_branched(e)
    for _ in range(5):
        assert not tree.visit(i)
    old = i.epoch
    assert [tree.visit(e) for _ in range(3)] == [False, False, True] and e.refreshed_by == "rung1"
    assert i.epoch is old and not i.refreshed and e.epoch is not old
    assert [tree.visit(sibling) for _ in range(3)] == [False, False, True], "per path (issue 268)"
    plain = SearchTree(CachedEvaluator(model, cfg), root)
    pi = plain.child(plain.root, jump)
    assert not pi.inner and [plain.visit(pi) for _ in range(3)] == [False, False, True]
    quiet = SearchTree(CachedEvaluator(model, cfg), root, evaluates=_inner_at(1))
    qi = quiet.child(quiet.root, ("physical", sorted(root.heads())[0]))
    qe = quiet.child(qi, ("physical", sorted(qi.universe.heads())[1]))
    assert not any(quiet.visit(n) for n in (qi, qe, qi, qe, qi, qe, qe)) and quiet.refreshes == 0
    deep = SearchTree(CachedEvaluator(model, cfg), root, evaluates=_inner_at(2))
    e1 = deep.child(deep.root, jump)
    i2 = deep.child(e1, ("physical", sorted(e1.universe.heads())[0]))
    e2 = deep.child(i2, ("physical", sorted(i2.universe.heads())[1]))
    assert e1.branched and not e1.inner and i2.inner and not segment_branched(e2)
    assert not any(deep.visit(e2) for _ in range(6)), "the branch is E1's, not E2's segment's"


def test_c24_counts_inner_travel_at_the_next_evaluated_node():
    """C24 under C27 (travel weight 1/2, the other weights 0, threshold 1, visit gate 2, budget 8):
    root -> I1 (a jump) -> I2 (a jump) -> E: E fires at its second visit - two travel moves since the
    last refresh, made at inner nodes - and the budget counts it; I1 and I2 never fire. The effort
    signal at E is the nearest evaluated ancestor's (issue 269): None until it is evaluated, then its
    predicted difficulty."""
    model = _model(671)
    root = _root("deep_travel", 8)
    cfg = CacheConfig(
        complexity_refresh=True,
        complexity_travel_weight=0.5,
        complexity_board_weight=0.0,
        complexity_timeline_weight=0.0,
        complexity_effort_weight=0.0,
        complexity_threshold=1.0,
        complexity_visit_threshold=2,
        complexity_refresh_budget=8,
    )
    tree = SearchTree(CachedEvaluator(model, cfg), root, evaluates=_inner_at(1, 2))
    i1 = tree.child(tree.root, ("jump", *_jump_any(root)))
    i2 = tree.child(i1, ("jump", *_jump_any(i1.universe)))
    e = tree.child(i2, ("physical", sorted(i2.universe.heads())[0]))
    assert i1.inner and i2.inner and not e.inner and travel_since_refresh(e) == 2
    for _ in range(4):
        assert not tree.visit(i1) and not tree.visit(i2)
    assert complexity_fires(tree.complexity_signals(i2), cfg) and not tree.complexity_due(i2), "no trigger at an inner node"
    assert [tree.visit(e), tree.visit(e)] == [False, True] and e.refreshed_by == "complexity"
    assert tree.complexity_refreshes == 1
    assert evaluated_ancestor(e) is tree.root and tree.complexity_signals(e).effort is None
    tree.evaluate(tree.root)
    assert tree.complexity_signals(e).effort == effort_difficulty(tree.root.effort_logits, cfg.complexity_effort_quantile)


def _jump_any(u: Multiverse) -> tuple:
    for p in sorted(u.heads()):
        targets = _jump_targets(u, p)
        if targets:
            return p, targets[0]
    raise AssertionError("no jump")


def test_an_inner_node_is_never_evaluated_or_refreshed():
    """The refusals: evaluating an inner node, refreshing an inner node - and a dense epoch refuses a
    refresh wherever (C26: below N* there is no cache)."""
    model = _model(672)
    root = _root("standard", 9)
    tree = SearchTree(CachedEvaluator(model, CacheConfig(node_refresh=True)), root, evaluates=_inner_at(1))
    i = tree.child(tree.root, ("physical", sorted(root.heads())[0]))
    with pytest.raises(ValueError, match="inner"):
        tree.evaluate(i)
    with pytest.raises(ValueError, match="inner"):
        tree.refresh(i)
    dense = SearchTree(CachedEvaluator(model, CacheConfig(root_refresh=True, phase_form=True, phase_boards=1000)), root)
    with pytest.raises(ValueError, match="no cache to refresh"):
        dense.refresh(dense.child(dense.root, ("physical", sorted(root.heads())[0])))


def test_with_every_node_evaluated_the_fixture_is_what_it_was():
    """``evaluates`` None (the default) or always true: no node is inner, ``segment_branched`` is the
    node's own flag and ``evaluated_ancestor`` its parent - and C24's simulation (``test_complexity``)
    makes the same refreshes at the same visits."""
    from tests.net.test_complexity import _paths, _simulate
    from tests.net.test_complexity import _root as complexity_root

    model = _model(673, layers=1)
    root = complexity_root(10)
    cfg = CacheConfig(
        node_refresh=True, refresh_visit_threshold=3, complexity_refresh=True, complexity_threshold=1.0, complexity_visit_threshold=2
    )
    runs = []
    for evaluates in (None, lambda parent, move, u: True):
        tree = SearchTree(CachedEvaluator(model, cfg), root, evaluates=evaluates)
        runs.append(_simulate(tree, _paths(root, 10), 40, 10, evaluate=True))
        for node in tree.nodes():
            assert not node.inner and segment_branched(node) == node.branched
            assert evaluated_ancestor(node) is node.parent
    assert runs[0] == runs[1] and runs[0]
