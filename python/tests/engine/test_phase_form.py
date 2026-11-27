"""Section 15.2's C26 row through the engine's search (the C step API): "Phase form: with the cached
configuration on, a search rooted under N* equals the dense search node for node; one rooted at or
above it evaluates every node cached | The mode is the root's, never a node's | C26".

The phase form (CHECKPOINT 9, spec section 7.10) is the cached path behind
``CacheConfig.phase_form`` (off): ``CachedNetworkEvaluator`` opens each search at its root
(``start_epoch``), and the root's board count picks the mode of the whole search - under N* the
evaluator's one dense epoch, whose evaluations are the dense path itself
(``NetworkEvaluator(scope="dense")`` with the evaluator's ``max_batch``); at or above N* the root
refresh (C3). "The dense search" is read as the same network's dense pass, batched alike (issue
262; proposed edit 380): so node for node means bit for bit.

* Rooted under N* (R6a): the search is the dense search node for node - H15's digest, its counts,
  every evaluation bit for bit and in order - under the ply split (tree and graph), the tree form
  (its natural gates, and none), the flat form, the evaluation cache on and off, batched and
  synchronous; nothing reaches the store; every leaf carries the dense epoch's engine number (never
  0, the number of an evaluator without ``start``). N* is the root's boards + 1 or + 3: every move's
  child has N* boards or more at + 1 - the trap of a per-node switch, whose mutant
  (``_PerNodeSwitch``: each leaf's mode by its own board count) builds another search on every arm.
* Rooted at or above N* (R6b): every node cached - the search is the cached path's with the root
  refresh, bit for bit; every leaf the two-level reference anchored at the root; the root leaf the
  dense pass (issue 259's anchor), within ``ROOT_DENSE_BOUND``; every other leaf more than 1e-3 from
  its own dense pass; the search not the dense one.
* The boundary (R6c), one evaluator across crossings (R6d), the evaluation cache shared across the
  crossing (R6e, C14), a carried search crossing it (R6f, C12 - under graph search a cached path
  transposes into a carried node evaluated dense, BUILD_LOG issue 264, pinned), whole games crossing
  it (R6g), F1 in BF16 (R6h, slow).
* Units: N*'s placeholder is ``token_counts.N_STAR``'s; the dense epoch's engine number, and every
  root refresh's its own; the dense dispatch is ``NetworkEvaluator``'s, bit for bit, with either
  prior and the leave logits; its ``max_batch``, by default the dense evaluator's; it checks the
  network's format; the start shortcut; the refusal of node refreshes, after construction too; the
  store keeps only the live search's epoch (issue 273); A5 still refused first.

FP32, 2 layers, random weights at initialisation scale, the v1 defaults (B15, B16, A6, A7); the
verification ladder off in every search (from Python it never changes the tree, H41, and it costs
threads). Tolerances: bitwise where both sides run the same code on the same batches; against a
reference ``test_cached_variants``'s ``_evals_close`` (1e-4); "the modes differ": above 1e-3.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from t5d.engine import Game
from t5d.engine.capi import T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_FLAT, T5D_SPLIT_TREE
from t5d.engine.evaluator import (
    CachedNetworkEvaluator,
    CStepDriver,
    NetworkEvaluator,
    _split,
    engine_epoch,
    policy_source,
    serve,
    start_epoch,
    state_of,
)
from t5d.net import NetConfig, build_model, synthetic_state
from t5d.net.cache import CacheConfig, CachedEvaluator, KVStore, reference_output, state_digest
from t5d.net.state import to_piece_centric
from tests.engine.helpers import ENGINE_VARIANTS, requires_engine
from tests.engine.test_cached_variants import ROOT_DENSE_BOUND, THREE_VARIANTS, _evals_close, _logit_gap, _search_root

R1, R2 = "past_and_epoch_history", "causal_past"
SEEDS = (0, 1)
PLAYOUTS = 24
BATCH = 8
#: The verification ladder off (E10): it never changes the tree from Python (H41) and costs threads.
QUIET = dict(hunt=0, settling=0, asset_map=0, verification_threads=1)
#: The search arms the row is held under (``search_params`` overrides).
ARMS = {
    "ply-tree": dict(split=T5D_SPLIT_BOARD_MOVE),
    "ply-graph": dict(split=T5D_SPLIT_BOARD_MOVE, graph=1),
    "tree-form": dict(split=T5D_SPLIT_TREE),
    "tree-form-no-gates": dict(split=T5D_SPLIT_TREE, tree_gates=0),
    "flat": dict(split=T5D_SPLIT_FLAT),
}
#: The search's counts two node-for-node searches share (not E10's ladder counters).
COUNTS = (
    "playouts",
    "evaluations",
    "cache_hits",
    "nodes",
    "nodes_created",
    "transposition_hits",
    "checkmates",
    "stalemates",
    "turn_caps",
    "inherited_nodes",
    "gate_between",
    "gate_capture",
    "gate_check",
    "gate_present",
    "gate_always",
    "generated_turns",
)
#: The largest gaps met (printed at the module's end under ``-s``).
GAPS = {"cached leaf vs reference": 0.0, "cached root leaf vs dense": 0.0, "cached leaf vs dense (smallest)": math.inf}


@pytest.fixture(scope="module", autouse=True)
def _report_gaps():
    yield
    print("\nphase form, gaps met:", {k: f"{v:.2e}" for k, v in GAPS.items()})


def _model(seed: int, dtype: str = "fp32", layers: int = 2, **flags):
    return build_model(NetConfig(trunk_dtype=dtype, n_layers=layers, attn_scope="block_causal", **flags), seed=seed).eval()


def _phase(model, n_star: int, max_batch: int = BATCH, **cfg) -> CachedNetworkEvaluator:
    """The phase form at N* = ``n_star`` (with C3's root refresh, which it requires)."""
    config = CacheConfig(root_refresh=True, phase_form=True, phase_boards=n_star, **cfg)
    return CachedNetworkEvaluator(CachedEvaluator(model, config), max_batch=max_batch)


def _cached(model, max_batch: int = BATCH) -> CachedNetworkEvaluator:
    """The cached path at every size, with the root refresh: the phase form's mode at or above N*."""
    return CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=True)), max_batch=max_batch)


def _dense(model, max_batch: int = BATCH) -> NetworkEvaluator:
    """The dense search's evaluator: the network's dense pass."""
    return NetworkEvaluator(model, scope="dense", max_batch=max_batch)


def _same(a, b) -> bool:
    """Two evaluations bit for bit (NaN equal to NaN), their prior included."""
    arrays = ("policy_logits", "wdl_logits", "value_logits")
    if not all(np.array_equal(getattr(a, n), getattr(b, n)) for n in arrays):
        return False
    if (a.board_logits is None) != (b.board_logits is None):
        return False
    if a.board_logits is not None and not np.array_equal(a.board_logits, b.board_logits, equal_nan=True):
        return False
    scalars = ("submit_logit", "error", "moves_left")
    return all(np.array_equal(getattr(a, n), getattr(b, n), equal_nan=True) for n in scalars) and a.policy == b.policy


def _empty(store) -> bool:
    return not store.entries and not any(store.computed.values()) and store.hits == 0


class _Recorder:
    """The search's evaluator: forwards ``start`` (only when the evaluator has one - an evaluator
    without it keeps the search's epoch, ``start_epoch``) and its prior, and logs each leaf in order:
    its view (node key, moves made in its action, the leaf's engine epoch, its board count), its
    position and its evaluation, and each batch."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.policy = policy_source(inner)
        self.driver: CStepDriver | None = None
        self.views: list[tuple[int, int, int, int]] = []
        self.positions: list = []
        self.evals: list = []
        self.batches: list[int] = []
        self.started: list = []
        if callable(getattr(inner, "start", None)):
            self.start = self._start

    def _start(self, root):
        self.started.append(state_of(root))
        return self.inner.start(root)

    def __call__(self, positions):
        evs = self.inner(positions)
        for v, p in zip(self.driver.batch, positions, strict=True):
            self.views.append((v.key, v.expansion.moves_made, v.epoch, state_of(p).n_boards))
        self.positions.extend(positions)
        self.evals.extend(evs)
        self.batches.append(len(positions))
        return evs


def _drive(search, inner, playouts: int = PLAYOUTS, batch: int = BATCH) -> _Recorder:
    """``run_search``'s loop with the leaves recorded: the evaluator's epoch first, then the step
    API's driver served ``batch`` leaves at a time."""
    rec = _Recorder(inner)
    start_epoch(search, rec)
    rec.driver = CStepDriver(search, playouts=playouts)
    serve(rec.driver, rec, batch_size=batch)
    return rec


def _search(root: Game, inner, playouts: int = PLAYOUTS, batch: int = BATCH, **flags):
    """A search from ``root`` with ``inner`` as its evaluator (``search_params(seed=7, ...)``, the
    ladder off): the recorder, the summary and H15's digest."""
    from t5d.engine.search import Search, search_params

    with Search(root, params=search_params(seed=7, batch_size=batch, **QUIET, **flags)) as s:
        rec = _drive(s, inner, playouts, batch)
        return rec, s.info(), s.digest()


class _PerNodeSwitch:
    """The mutant C26 refuses: each leaf's mode by its own board count - a leaf under N* through the
    dense pass, one at or above it through the cached path anchored at the search's root (section
    7.10: "a per-node switch ... whenever a line crossed N*")."""

    def __init__(self, model, n_star: int) -> None:
        self.n_star = n_star
        self.dense = _dense(model)
        self.cached = _cached(model)

    def start(self, root):
        return self.cached.start(root)

    def __call__(self, positions):
        out: list = [None] * len(positions)
        low = [i for i, p in enumerate(positions) if state_of(p).n_boards < self.n_star]
        high = [i for i in range(len(positions)) if i not in low]
        for part, ev in ((low, self.dense), (high, self.cached)):
            if part:
                for i, e in zip(part, ev([positions[i] for i in part]), strict=True):
                    out[i] = e
        return out


def _node_for_node(p, p_info, p_digest, d, d_info, d_digest, what) -> None:
    """Two searches node for node: H15's digest, the counts, the leaves (key and moves made) and
    every evaluation bit for bit, in order."""
    assert p_digest == d_digest, what
    for name in COUNTS:
        assert getattr(p_info, name) == getattr(d_info, name), (what, name)
    assert len(p.evals) == len(d.evals) == p_info.evaluations > 0, what
    assert [v[:2] for v in p.views] == [v[:2] for v in d.views], what
    assert all(_same(a, b) for a, b in zip(p.evals, d.evals, strict=True)), (what, "the evaluations")


def _check_dense(root: Game, model, n_star: int, flags: dict, what, mutant: bool = True, playouts: int = PLAYOUTS):
    """R6a at one root and one arm: the phase form's search (N* = ``n_star``, above the root's board
    count) against the dense search; then the per-node switch against it. The phase-form recorder."""
    ev = _phase(model, n_star)
    p, p_info, p_digest = _search(root, ev, playouts, **flags)
    d, d_info, d_digest = _search(root, _dense(model), playouts, **flags)
    _node_for_node(p, p_info, p_digest, d, d_info, d_digest, what)
    assert ev.positions_by_mode == {"dense": len(p.evals), "cached": 0}, what
    assert _empty(ev.cached.store), what
    assert ev.epoch is ev.cached.dense_epoch, what
    number = engine_epoch(ev.epoch)
    assert number != 0 and {v[2] for v in p.views} == {number} and p_info.epoch == number, what
    assert {v[2] for v in d.views} == {0} and d_info.epoch == 0, what
    if mutant:
        m, _, m_digest = _search(root, _PerNodeSwitch(model, n_star), playouts, **flags)
        assert max(v[3] for v in m.views) >= n_star, (what, "the mutant met leaves at N* or above")
        assert m_digest != d_digest, (what, "the per-node switch builds another search")
    return p


# --- R6a: rooted under N*, the dense search node for node -----------------------------------------


@requires_engine
@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("variant", THREE_VARIANTS)
def test_rooted_under_n_star_the_search_is_the_dense_search_node_for_node(variant, arm):
    """Section 15.2's C26 row, its first half: rooted under N*, the phase form's search is the dense
    search node for node, and the per-node switch is not (``_check_dense``). Two seeds: seed 0 at N*
    the root's boards + 1 (every child at or above N*: the trap; the mutant), the evaluation cache
    on; seed 1 at + 3, the cache off (a state met twice then reaches the evaluator twice); the
    synchronous mode on the first variant's ply split. Coverage: seed 0's search meets leaves at N*
    boards or more."""
    synchronous = int(variant == THREE_VARIANTS[0] and arm == "ply-tree")
    with _search_root(variant) as root:
        n = root.info().boards
        for seed, n_star, cache in ((0, n + 1, {}), (1, n + 3, {"eval_cache_entries": 0})):
            flags = dict(ARMS[arm], synchronous=synchronous) | cache
            p = _check_dense(root, _model(seed), n_star, flags, (variant, arm, seed, n_star), mutant=seed == 0)
            if seed == 0:
                assert max(v[3] for v in p.views) >= n_star, (variant, arm, seed, "a leaf at N* or above")


@requires_engine
@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_rooted_under_n_star_on_every_variant(variant):
    """R6a on every variant of the engine's table, seed 0, N* the root's boards + 1, under the ply
    split and the tree form: the dense search node for node (no mutant: the arms above have it)."""
    model = _model(0)
    with _search_root(variant) as root:
        n = root.info().boards
        for arm in ("ply-tree", "tree-form"):
            p = _check_dense(root, model, n + 1, ARMS[arm], (variant, arm), mutant=False, playouts=12)
            assert max(v[3] for v in p.views) >= n + 1, (variant, arm)


# --- R6b: rooted at or above N*, every node cached ------------------------------------------------


def _check_cached(root: Game, model, n_star: int, flags: dict, what, against_dense: bool = True, playouts: int = PLAYOUTS):
    """R6b at one root and one arm (N* = ``n_star``, at most the root's board count): the phase form's
    search against the cached path's with the root refresh; each leaf against the two-level
    reference anchored at the root, the root leaf against the dense pass, the others far from it."""
    ev = _phase(model, n_star)
    p, p_info, p_digest = _search(root, ev, playouts, **flags)
    c, c_info, c_digest = _search(root, _cached(model), playouts, **flags)
    _node_for_node(p, p_info, p_digest, c, c_info, c_digest, what)
    assert ev.positions_by_mode == {"dense": 0, "cached": len(p.evals)}, what
    anchor = p.started[-1]
    assert ev.epoch.kind == "root" and ev.epoch.anchor == state_digest(anchor), what
    assert ev.cached.store.computed["refresh"] > 0, what
    number = engine_epoch(ev.epoch)
    assert {v[2] for v in p.views} == {number} and number != engine_epoch(ev.cached.dense_epoch), what
    dense = _dense(model)
    assert len(p.evals) > 1, what
    for i, (pos, out) in enumerate(zip(p.positions, p.evals, strict=True)):
        state = state_of(pos)
        ref = _split(reference_output(model, state, anchor=anchor), [state])[0]
        GAPS["cached leaf vs reference"] = max(GAPS["cached leaf vs reference"], _evals_close(out, ref, (what, i)))
        gap = _logit_gap(out, dense([pos])[0])
        if i == 0:  # a fresh search's first leaf is its root: the refresh pass is the dense pass
            assert state_digest(state) == state_digest(anchor), what
            assert gap < ROOT_DENSE_BOUND, (what, "the root leaf is the dense pass", gap)
            GAPS["cached root leaf vs dense"] = max(GAPS["cached root leaf vs dense"], gap)
        else:  # every other leaf: cached, conditioned on the root, not its own dense pass
            assert gap > 1e-3, (what, i, "a cached leaf is not the dense pass", gap)
            GAPS["cached leaf vs dense (smallest)"] = min(GAPS["cached leaf vs dense (smallest)"], gap)
    if against_dense:
        _, _, d_digest = _search(root, _dense(model), playouts, **flags)
        assert p_digest != d_digest, (what, "the search is not the dense one")
    return p


@requires_engine
@pytest.mark.parametrize("arm", ["ply-tree", "tree-form", "flat"])
@pytest.mark.parametrize("variant", THREE_VARIANTS)
def test_rooted_at_or_above_n_star_every_node_is_cached(variant, arm):
    """Section 15.2's C26 row, its second half: rooted at or above N*, every node is evaluated cached
    (``_check_cached``). Seed 0 at N* = the root's boards (the boundary) under C20's ``causal_past``,
    seed 1 at the root's boards - 2 under ``past_and_epoch_history``."""
    with _search_root(variant) as root:
        n = root.info().boards
        for seed, n_star, supersession in ((0, n, R2), (1, n - 2, R1)):
            model = _model(seed, supersession_scope=supersession)
            _check_cached(root, model, n_star, ARMS[arm], (variant, arm, seed), against_dense=seed == 0, playouts=12)


# --- R6c: the boundary -------------------------------------------------------------------------------


@requires_engine
def test_the_boundary_is_the_roots_board_count():
    """N* at the root's boards runs cached, one more runs dense - nothing else differs - at a random
    play root, at a mid-action root past the shift, at Turn Zero's start (the T0 board counted: N*
    equal to every board, it included, is cached), and the pseudo-legal start of
    ``test_cached_variants``' PGN with boards alike but for their past, as a cached root."""
    from tests.engine.test_search_api import _shifted_game_with_moves

    model = _model(2)
    roots = []
    with _search_root("Standard - Two Timelines") as g:
        roots.append(("random play", g.copy()))
    roots.append(("past the shift", _shifted_game_with_moves(seed=3)))
    roots.append(("turn zero", Game.variant("Standard - Turn Zero")))
    pgn = '[Board "Standard"]\n[Mode "5D"]\n\n1. Nf3 / Nf6\n2. Ng1 / Ng8\n3. e4 / e5\n'
    g = Game.from_pgn(pgn)
    for _ in range(5):
        g.replay_next()
    roots.append(("alike boards, other pasts", g))
    try:
        for name, root in roots:
            n = root.info().boards
            if name == "turn zero":
                assert root.position().state.half_turn.min() < 2 and n == root.position().state.n_boards, "the T0 board is a board"
            at = _phase(model, n)
            above = _phase(model, n + 1)
            for ev in (at, above):
                ev.start(root.position())
            assert at.epoch.kind == "root" and above.epoch is above.cached.dense_epoch, name
            arm = ARMS["ply-tree"]
            _check_dense(root, model, n + 1, arm, (name, "n + 1"), mutant=False, playouts=8)
            _check_cached(root, model, n, arm, (name, "n"), against_dense=False, playouts=8)
    finally:
        for _, root in roots:
            root.close()


@requires_engine
def test_a_branching_jump_takes_a_child_two_boards_up():
    """A root where the search meets a branching jump's child - two boards more, the jump's own board
    and the new timeline's first: with N* at the root's boards + 2 that child is exactly at N* and a
    physical move's child under it, and the search is still the dense search node for node."""
    model = _model(3)
    with _search_root("Very Small - Open") as root:
        n = root.info().boards
        start = root.position().state
        timelines, acting = len(set(start.timeline.tolist())), int(start.acting_player)
        p = _check_dense(root, model, n + 2, ARMS["ply-tree"], ("jump", "n + 2"), mutant=False, playouts=64)
        children = {
            (v[3], len(set(state_of(pos).timeline.tolist())))
            for v, pos in zip(p.views, p.positions, strict=True)
            if v[1] == 1 and int(state_of(pos).acting_player) == acting  # the root's children (it is a turn start)
        }
        assert (n + 2, timelines + 1) in children, ("a branching jump's child, at N*", sorted(children))
        assert any(b == n + 1 for b, _ in children), ("a physical move's child, under N*", sorted(children))


# --- R6d: one evaluator across crossings -------------------------------------------------------------


@requires_engine
def test_one_evaluator_across_crossings():
    """One evaluator, N* = the first root's boards + 1, follows play: a search at that root (dense -
    the dense search node for node), a search at a later root with N* boards or more (cached - the
    cached path's node for node, its own root refresh), then a new game's small root: dense again,
    with the same dense epoch object and number."""
    from t5d.engine.play import random_action

    model = _model(4)
    with _search_root("Standard - Two Timelines") as root:
        n = root.info().boards
        ev = _phase(model, n + 1)
        p, info, digest = _search(root, ev)
        _, d_info, d_digest = _search(root, _dense(model))
        assert digest == d_digest and ev.epoch is ev.cached.dense_epoch
        dense_epoch, number = ev.epoch, engine_epoch(ev.epoch)
        later = root.copy()
        later.set_legality(False)
        rng = np.random.default_rng(4)
        while later.info().boards < n + 1 or not later.info().turn_start:
            assert random_action(later, rng) >= 0
        with later:
            q, q_info, q_digest = _search(later, ev)
            assert q.evals, "a live later root"
            _, _, c_digest = _search(later, _cached(model))
            assert q_digest == c_digest and ev.epoch.kind == "root" and q_info.epoch == engine_epoch(ev.epoch) != number
            assert ev.positions_by_mode == {"dense": len(p.evals), "cached": len(q.evals)}
    with Game.variant("Standard - Two Timelines") as fresh:
        r, r_info, r_digest = _search(fresh, ev)
        _, _, d_digest = _search(fresh, _dense(model))
        assert r_digest == d_digest and ev.epoch is dense_epoch and r_info.epoch == number
        assert {v[2] for v in r.views} == {number}


# --- R6e: the evaluation cache shared across the crossing (C14) --------------------------------------


@requires_engine
def test_a_shared_cache_across_the_crossing():
    """C14 with the phase form (``test_search_api``'s template, one leaf at a time): a dense search at
    a root R, N* = R's boards + 1, then one move - the new root has N* boards or more - and a fresh
    search sharing R's evaluation cache beside an unshared one, the same evaluator: the same search
    (digest), the same evaluations, the epoch changed - the cache held states the dense search
    evaluated and the evaluator was asked about them again (never a dense evaluation served to a
    cached search). Below N* throughout (N* = 1000): two dense searches sharing the cache both carry
    the dense number, and the second takes cache hits - fewer evaluations than alone, the same search."""
    from t5d.engine.search import Search, search_params

    model = _model(5)
    params = dict(seed=4, batch_size=1, **QUIET)
    for n_star_above, crossing in ((1, True), (1000, False)):
        with _search_root("Standard - Two Timelines") as g:
            n = g.info().boards
            ev = _phase(model, n + 1 if crossing else n_star_above)
            first = Search(g, params=search_params(**params))
            f = _drive(first, ev, playouts=24, batch=1)
            first_epoch = first.info().epoch
            g.apply(first.child(first.choose()[0]).moves[0])
            shared = Search(g, turn_start=False, params=search_params(**params), share=first)
            first.close()
            s = _drive(shared, ev, playouts=24, batch=1)
            alone = Search(g, turn_start=False, params=search_params(**params))
            a = _drive(alone, ev, playouts=24, batch=1)
            si, ai = shared.info(), alone.info()
            assert shared.digest() == alone.digest() and si.epoch == ai.epoch, crossing
            if crossing:
                assert first_epoch == engine_epoch(ev.cached.dense_epoch) != si.epoch == engine_epoch(ev.epoch)
                assert si.evaluations == ai.evaluations and si.cache_hits == ai.cache_hits
                again = {v[:2] for v in f.views} & {v[:2] for v in s.views}
                assert again, "a state the dense search evaluated reached the evaluator again"
                assert all(_same(x, y) for x, y in zip(s.evals, a.evals, strict=True))
            else:
                assert first_epoch == si.epoch == engine_epoch(ev.cached.dense_epoch) != 0
                assert si.cache_hits > ai.cache_hits and si.evaluations < ai.evaluations
            shared.close()
            alone.close()


# --- R6f: a carried search crossing N* (C12) ----------------------------------------------------------


#: R6f's searches: (variant, network seed, search seed, playouts before the crossing, after it). The
#: graph arm's are larger, so that a line after the crossing meets a carried node (``_into_dense``);
#: their order is the order they are tried in, until one meets one.
CARRIED = [("Standard - Two Timelines", 6, 3, 48, 24)]
CARRIED_GRAPH = [
    ("Misc - Timeline Battleground", 7, 4, 256, 128),
    ("Standard - Two Timelines", 7, 4, 256, 128),
    ("Very Small - Open", 7, 4, 256, 128),
]


def _boards(state) -> set:
    """A position's boards by coordinates and contents (a position's boards are its predecessor's and
    the ones its move made)."""
    p, h = state.timeline, state.half_turn
    return {(int(p[b]), int(h[b]), state.piece[b].tobytes(), state.colour[b].tobytes()) for b in range(state.n_boards)}


def _into_dense(s, ev, dense: _Recorder, cached: _Recorder) -> int:
    """Under graph search, after the crossing: a node the cached search made (evaluated in the root
    refresh's epoch, never before) one move from a state the dense search evaluated and the cached
    one never did; the search advanced onto that node, its child by that move is the carried node -
    the edge made after the crossing, the node before it: a cached path transposed into a node
    evaluated dense - and that node holds the dense evaluation (its WDL the dense pass's, not the
    cached path's), never evaluated again. How many such children the node has (0: no candidate)."""
    first = lambda views: {v[:2]: i for i, v in reversed(list(enumerate(views)))}  # noqa: E731
    d_at, c_at = first(dense.views), first(cached.views)
    children = {c.child_key: c.index for c in s.children() if c.child_key}
    dense_only = {k: _boards(state_of(dense.positions[i])) for k, i in d_at.items() if k not in c_at}
    pairs = []
    for made, i in c_at.items():
        if made in d_at or made[0] not in children:
            continue
        boards = _boards(state_of(cached.positions[i]))
        pairs += [(made, x) for x, xb in dense_only.items() if x[1] == made[1] + 1 and boards < xb and len(xb - boards) in (1, 2)]
    if not pairs:
        return 0
    made = pairs[0][0]
    want = {x[0]: x for m, x in pairs if m == made}
    s.advance(children[made[0]])
    linked = [c for c in s.children() if c.child_key in want]
    for c in linked:
        i = d_at[want[c.child_key]]
        state = state_of(dense.positions[i])
        with torch.no_grad():
            now = _split(ev.cached.evaluate(state, ev.epoch), [state])[0]
        wdl = np.asarray(c.network_wdl)
        softmax = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()  # noqa: E731
        assert c.evaluated and np.abs(wdl - softmax(np.asarray(dense.evals[i].wdl_logits, np.float64))).max() < 1e-5, "the dense evaluation"
        assert np.abs(wdl - softmax(np.asarray(now.wdl_logits, np.float64))).max() > 1e-4, "not the cached path's"
    return len(linked)


def _carried_crossing(arm: str, variant: str, model_seed: int, search_seed: int, before_playouts: int, after_playouts: int) -> int:
    """R6f at one root: the dense search, the advance across N*, the cached search after it - and
    under graph search ``_into_dense``'s count."""
    from t5d.engine.search import Search, search_params

    model = _model(model_seed)
    with _search_root(variant) as root:
        n = root.info().boards
        ev = _phase(model, n + 1)
        with Search(root, params=search_params(seed=search_seed, **QUIET, **ARMS[arm])) as s:
            dense = _drive(s, ev, playouts=before_playouts)
            dense_number = s.info().epoch
            assert dense_number == engine_epoch(ev.cached.dense_epoch)
            kids = [c for c in s.children() if not c.submit and c.visits > 1]
            if arm.startswith("tree-form"):
                kids = [c for c in kids if not c.evaluated]
                assert kids, (arm, "an inherited child with visits")
            pick = max(kids, key=lambda c: c.visits)
            s.advance(pick.index)
            before = s.info()
            assert before.root_visits >= pick.visits > 1
            dense_before = ev.positions_by_mode["dense"]
            rec = _drive(s, ev, playouts=after_playouts)
            after = s.info()
            anchor = rec.started[-1]
            assert anchor.n_boards >= n + 1 and ev.epoch.kind == "root" and after.epoch == engine_epoch(ev.epoch) != dense_number
            assert after.root_visits >= before.root_visits + 1 and ev.positions_by_mode["dense"] == dense_before
            assert rec.evals and {v[2] for v in rec.views} == {after.epoch}
            for i, (pos, out) in enumerate(zip(rec.positions, rec.evals, strict=True)):
                state = state_of(pos)
                ref = _split(reference_output(model, state, anchor=anchor), [state])[0]
                GAPS["cached leaf vs reference"] = max(GAPS["cached leaf vs reference"], _evals_close(out, ref, (arm, i)))
            if arm.startswith("tree-form"):
                assert rec.views[0][0] == before.root_key and after.root_evaluated and not before.root_evaluated
                assert [v[0] for v in rec.views].count(before.root_key) == 1
            return _into_dense(s, ev, dense, rec) if arm == "ply-graph" else 0


@requires_engine
@pytest.mark.parametrize("arm", ["ply-tree", "ply-graph", "tree-form", "tree-form-no-gates"])
def test_a_carried_search_crossing_n_star(arm):
    """Match play carries the search (C12). Rooted under N* (the root's boards + 1) the search is
    dense; advanced onto its most visited child - for the tree form onto a child it inherited (never
    evaluated) - the new root has N* boards or more, and the next run opens the root refresh: the
    search's epoch moves from the dense number to the new epoch's, the root's visits are kept, no
    position is evaluated dense any more, and every new leaf is the two-level reference anchored at
    the new root; under the tree form the inherited root is the first new leaf, evaluated once.

    Under graph search (``ply-graph``) the re-root keeps the carried nodes in the graph's table, keyed
    by state and moves made, not by epoch, so the behaviour BUILD_LOG issue 264 records is pinned: a
    node the cached search made after the crossing transposes into a carried node evaluated dense,
    which is not evaluated again - the cached search reads that state's dense evaluation
    (``_into_dense``). Section 7.10's "transpositions never mix a dense and a cached evaluation of
    one state", read strictly, forbids it: a spec conflict for the user. If the strict reading is
    ruled, the engine drops or re-evaluates the carried nodes' table entries at the crossing, and
    this arm's assertion turns round. Tried on ``CARRIED_GRAPH``'s searches in order until one meets
    such a node."""
    met = 0
    for spec in CARRIED_GRAPH if arm == "ply-graph" else CARRIED:
        met += _carried_crossing(arm, *spec)
        if met:
            break
    assert met or arm != "ply-graph", "no search met a cached path transposing into a node evaluated dense"


# --- R6g: games crossing N* ---------------------------------------------------------------------------


class _Watch:
    """A game's evaluator that logs each search's start (the root's boards, the mode, the engine
    epoch it sets) and each batch with the mode it was evaluated in."""

    def __init__(self, inner: CachedNetworkEvaluator) -> None:
        self.inner = inner
        self.policy = inner.policy
        self.starts: list[tuple[int, str, int, object]] = []
        self.batches: list[tuple[str, int, list]] = []

    def start(self, root):
        epoch = self.inner.start(root)
        self.starts.append((state_of(root).n_boards, epoch.mode, engine_epoch(epoch), state_of(root)))
        return epoch

    def __call__(self, positions):
        evs = self.inner(positions)
        self.batches.append((self.inner.epoch.mode, len(self.starts) - 1, list(zip(positions, evs, strict=True))))
        return evs


GAMES = [("Standard", "ply", "self"), ("Standard - Two Timelines", "tree", "match")]
SLOW_GAMES = [
    ("Standard", "flat", "self"),
    ("Standard - Two Timelines", "ply", "match"),
    ("Standard", "tree", "self"),
    ("Standard - Two Timelines", "flat", "match"),
]


def _play(variant: str, split: str, mode: str, sample: int, playouts: int = 8, cap_turns: int = 3):
    from t5d.engine.netplay import PlayConfig, check_replay, gate_executable, network_config, play_game, replay_with_gate

    cfg = network_config("small")
    model = build_model(dataclasses.replace(cfg, attn_scope="block_causal"), seed=11).eval()
    with Game.variant(variant) as g:
        start = g.info().boards
    n_star = start + 3
    watch = _Watch(_phase(model, n_star))
    play = PlayConfig(
        variant=variant,
        split=split,
        mode=mode,
        playouts=playouts,
        batch_size=4,
        cap_turns=cap_turns,
        cap_boards=0,
        seed=5,
        hunt=False,
        settling=False,
        asset_map=False,
        verification_threads=1,
    )
    with torch.no_grad():
        res = play_game(watch, play, net_cfg=model.cfg)
    modes = [m for _, m, _, _ in watch.starts]
    assert "dense" in modes and "cached" in modes, (variant, split, mode, "the game crossed N*", modes)
    first_cached = modes.index("cached")
    assert all(m == "cached" for m in modes[first_cached:]), "never back to dense"
    assert all((b >= n_star) == (m == "cached") for b, m, _, _ in watch.starts), "the mode is the root's"
    dense = _dense(model, max_batch=BATCH)
    checked = 0
    for mode_used, which, items in watch.batches:
        positions = [p for p, _ in items]
        if mode_used == "dense":
            assert all(_same(a, b) for (_, a), b in zip(items, dense(positions), strict=True)), "a dense batch is the dense path's"
        else:
            anchor = watch.starts[which][3]
            for pos, out in items[:sample]:
                state = state_of(pos)
                ref = _split(reference_output(model, state, anchor=anchor), [state])[0]
                GAPS["cached leaf vs reference"] = max(GAPS["cached leaf vs reference"], _evals_close(out, ref, (variant, split, mode)))
                checked += 1
    assert checked > 0
    if mode == "match":
        assert res.reroot_fallbacks == 0, "every carried search re-rooted, across the crossing too (C12)"
    replay = replay_with_gate(res.record, gate=gate_executable())
    assert check_replay(res, replay) == [], replay.lines[-3:]
    return res, watch


@requires_engine
@pytest.mark.parametrize("variant,split,mode", GAMES)
def test_a_game_crossing_n_star(variant, split, mode):
    """Whole games (``netplay.play_game``) under the phase form, N* = the start's boards + 3: every
    search's mode is its root's - dense until the first root with N* boards or more, cached from it
    on, never back; every dense batch is the dense path's bit for bit; cached leaves (the first of each
    batch) are the two-level reference of their own search's root; the record replays legal through
    the gate. Match play's carried searches cross with their statistics (C12). 8 playouts a decision,
    a cap of 3 turns."""
    _play(variant, split, mode, sample=1)


@requires_engine
@pytest.mark.slow
@pytest.mark.parametrize("variant,split,mode", SLOW_GAMES)
def test_a_game_crossing_n_star_every_leaf(variant, split, mode):
    """The other games - the flat form, and the other mode of each split - every cached leaf checked
    against its reference."""
    _play(variant, split, mode, sample=10**6)


# --- R6h: F1 in BF16 (slow) ---------------------------------------------------------------------------


@requires_engine
@pytest.mark.slow
@pytest.mark.parametrize("variant", THREE_VARIANTS)
def test_f1_bf16_phase_form(variant):
    """F1 as shipped but block-causal (BF16, 16 layers): rooted under N* the search is the BF16 dense
    search node for node; rooted at N* its leaves are within ``test_network.py``'s BF16 bounds of the
    BF16 two-level reference."""
    model = build_model(NetConfig(attn_scope="block_causal"), seed=0).eval()
    with _search_root(variant) as root:
        n = root.info().boards
        _check_dense(root, model, n + 1, ARMS["ply-tree"], (variant, "bf16"), mutant=False, playouts=8)
        ev = _phase(model, n)
        p, _, _ = _search(root, ev, playouts=4)
        anchor = p.started[-1]
        for pos, out in zip(p.positions, p.evals, strict=True):
            state = state_of(pos)
            ref = _split(reference_output(model, state, anchor=anchor), [state])[0]
            assert np.abs(out.policy_logits - ref.policy_logits).max(initial=0.0) < 3e-2
            assert np.abs(out.value_logits - ref.value_logits).max() < 5e-3


# --- units ----------------------------------------------------------------------------------------------


def test_n_star_is_token_counts_placeholder():
    """One placeholder: the cached path's N* default is the N* A5's costs were pooled at."""
    from t5d.engine.token_counts import N_STAR

    assert CacheConfig().phase_boards == N_STAR == 50


def test_the_dense_epochs_engine_number():
    """The dense epoch's engine number (``engine_epoch``): never 0 (the epoch of a search whose
    evaluator has no ``start``), not the causal epoch's, not a root refresh's, and the same for every
    evaluator - so searches below N* share dense evaluations through the engine's cache (C14)."""
    model = _model(7)
    root = synthetic_state(70, 6, n_timelines=3)
    a, b = _phase(model, 1000), _phase(model, 1000)
    numbers = {engine_epoch(ev.start(root)) for ev in (a, b)}
    assert len(numbers) == 1 and 0 not in numbers
    (number,) = numbers
    causal = CachedEvaluator(model).root_epoch(root)
    refreshed = _phase(model, 1).cached.root_epoch(root)
    assert number not in {engine_epoch(causal), engine_epoch(refreshed)}
    assert a.epoch is a.cached.dense_epoch and a.epoch is not b.epoch and a.epoch.key == b.epoch.key == "dense"


def test_every_root_refresh_has_its_own_engine_number():
    """``engine_epoch`` hashes the epoch's whole key: two root refreshes - of two roots, or of one root
    twice - carry different engine numbers, so a shared evaluation cache (C14) never serves one
    refresh's evaluations to another; the dense number is none of them."""
    model = _model(7)
    a, b = synthetic_state(70, 6, n_timelines=3), synthetic_state(71, 7, n_timelines=3)
    cached = _phase(model, 1).cached
    numbers = [engine_epoch(cached.root_epoch(s)) for s in (a, b, a)]
    assert len(set(numbers)) == 3 and engine_epoch(_phase(model, 1000).cached.dense_epoch) not in numbers


@pytest.mark.parametrize("policy", ["normal", "optimistic"])
def test_the_dense_dispatch_is_the_dense_evaluator(policy):
    """Under N* the evaluator's evaluations are ``NetworkEvaluator(scope="dense")``'s bit for bit on the
    same batch - either prior (D9's optimistic one too), the leave logits included - and ``max_batch``
    splits the batch as the dense evaluator does; the start shortcut leaves the dense epoch as it is."""
    from t5d.train.smoke import with_actions

    model = _model(8, optimistic_policy=True)
    states = with_actions([synthetic_state(80 + i, n, n_timelines=2) for i, n in enumerate((3, 5, 2, 4, 6))])
    for max_batch in (2, 8):
        ev = CachedNetworkEvaluator(
            CachedEvaluator(model, CacheConfig(root_refresh=True, phase_form=True, phase_boards=1000)), policy=policy, max_batch=max_batch
        )
        dense = NetworkEvaluator(model, scope="dense", policy=policy, max_batch=max_batch)
        first = ev.start(states[0])
        assert ev.start(states[1]) is first is ev.cached.dense_epoch
        got, want = ev(states), dense(states)
        assert all(_same(a, b) for a, b in zip(got, want, strict=True)) and {e.policy for e in got} == {policy}
        assert all(e.board_logits is not None and e.board_logits.shape == (s.n_boards,) for e, s in zip(got, states, strict=True))
        assert ev.dense.batches == dense.batches == math.ceil(len(states) / max_batch)
        assert ev.positions_by_mode == {"dense": len(states), "cached": 0} and _empty(ev.cached.store)


def test_the_default_max_batch_is_the_dense_evaluators():
    """Built with defaults, the phase form's dense mode batches as the default dense evaluator does -
    one forward for 9 positions - so a default-built search under N* is the default dense search."""
    from t5d.train.smoke import with_actions

    model = _model(8)
    states = with_actions([synthetic_state(90 + i, 3 + i % 3, n_timelines=2) for i in range(9)])
    ev = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=True, phase_form=True, phase_boards=1000)))
    dense = NetworkEvaluator(model, scope="dense")
    ev.start(states[0])
    got, want = ev(states), dense(states)
    assert ev.dense.max_batch == dense.max_batch and ev.dense.batches == dense.batches == 1
    assert all(_same(a, b) for a, b in zip(got, want, strict=True))


def test_the_engine_path_refuses_node_refreshes():
    """The engine's search opens one epoch per search and refreshes no node (BUILD_LOG issue 271), so
    its evaluator refuses a cached evaluator configured for node refreshes - C3a's rung 1 or C24's -
    rather than ignore them; the root refresh and the phase form are its own and accepted. The
    refusal holds after construction: the cached evaluator's configuration cannot be swapped, and a
    cached evaluator swapped in for another is refused at ``start``."""
    model = _model(9)
    for cfg in (CacheConfig(node_refresh=True), CacheConfig(complexity_refresh=True), CacheConfig(root_refresh=True, node_refresh=True)):
        with pytest.raises(NotImplementedError, match="issue 271"):
            CachedNetworkEvaluator(CachedEvaluator(model, cfg))
    for cfg in (CacheConfig(), CacheConfig(root_refresh=True), CacheConfig(root_refresh=True, phase_form=True)):
        CachedNetworkEvaluator(CachedEvaluator(model, cfg))
    ev = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=True)))
    with pytest.raises(AttributeError):
        ev.cached.config = CacheConfig(root_refresh=True, node_refresh=True)
    ev.cached = CachedEvaluator(model, CacheConfig(root_refresh=True, node_refresh=True))
    with pytest.raises(NotImplementedError, match="issue 271"):
        ev.start(synthetic_state(72, 5, n_timelines=2))
    assert ev.epoch is None


def test_the_dense_dispatch_checks_the_networks_format():
    """Every epoch and evaluation checks the network's format (G8's switch in place; spec issue 238),
    the dense dispatch too: a network switched to another trunk format after ``start`` is refused
    before the dense evaluator runs it."""
    from t5d.train.smoke import with_actions

    model = build_model(NetConfig(n_layers=1, trunk_dtype="fp32", attn_scope="block_causal"), seed=0).eval()
    states = with_actions([synthetic_state(60 + i, 3, n_timelines=2) for i in range(2)])
    ev = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=True, phase_form=True, phase_boards=1000)))
    ev.start(states[0])
    ev(states)
    model.set_trunk_quant("fp8_e4m3")
    try:
        with pytest.raises(RuntimeError, match="format changed"):
            ev(states)
        assert ev.dense.batches == 1, "refused before the dense evaluator ran"
    finally:
        model.set_trunk_quant("none")


@requires_engine
def test_a5_under_the_phase_form_through_the_step_api():
    """A5 under C26 through the engine's search (decision 7 settled issue 252; issue 266): rooted
    under N* the phase form's search with a piece-centric network is the dense search node for node
    (its dense mode converts at the network's edge, as ``NetworkEvaluator`` does); rooted at N* every
    leaf - converted to A5 strictly - is the two-level reference anchored at the root."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", piece_centric=True), seed=11).eval()
    with _search_root("Standard - Two Timelines") as root:
        n = root.info().boards
        _check_dense(root, model, n + 1, ARMS["ply-tree"], ("a5", "n + 1"), mutant=False, playouts=12)
        ev = _phase(model, n)
        p, _, _ = _search(root, ev, playouts=12)
        anchor = to_piece_centric(p.started[-1], strict=True)
        assert ev.epoch.kind == "root"
        for pos, out in zip(p.positions, p.evals, strict=True):
            state = to_piece_centric(state_of(pos), strict=True)
            ref = _split(reference_output(model, state, anchor=anchor), [state])[0]
            _evals_close(out, ref, "a5")


def test_the_store_keeps_only_the_live_search():
    """``start`` drops the root refresh its new search replaces (BUILD_LOG issue 273): one evaluator
    following play - cached, cached, dense, cached, a new game cached - holds after each start only the
    live epoch's entries (none under a dense search), and the shared causal epoch's are kept. Only
    epochs it made: in a store shared with another evaluator (issue 274), the other's root refresh
    stays, and so does an epoch the evaluator holds without having made it."""
    model = _model(10)
    roots = [synthetic_state(110 + i, n, n_timelines=2) for i, n in enumerate((8, 9, 4, 10, 8))]
    ev = _phase(model, 6)  # the 4-board root is dense
    for i, root in enumerate(roots):
        epoch = ev.start(root)
        ev([root, roots[(i + 1) % len(roots)]])
        keys = {k[0] for k in ev.cached.store.entries}
        assert keys == (set() if epoch.mode == "dense" else {epoch.key}), (i, keys)
        assert ev.cached.store.computed["supersession"] + ev.cached.store.computed["refresh"] > 0
    assert [ev.positions_by_mode[m] > 0 for m in ("dense", "cached")] == [True, True]
    shared = KVStore()
    a = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=True), shared))
    b = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=True), shared))
    ea, eb = a.start(roots[0]), b.start(roots[1])
    a.start(roots[2])
    keys = {k[0] for k in shared.entries}
    assert ea.key not in keys and eb.key in keys and a.epoch.key in keys
    b.epoch = a.epoch  # an epoch b did not make
    b.start(roots[3])
    assert a.epoch.key in {k[0] for k in shared.entries}, "another evaluator's epoch is kept"
    causal = CachedNetworkEvaluator(CachedEvaluator(model))
    first = causal.start(roots[0])
    causal(roots[:2])
    causal.start(roots[1])
    assert causal.epoch is first or causal.epoch.key == first.key == "causal"
    assert {k[0] for k in causal.cached.store.entries} == {"causal"}, "the shared causal epoch is kept"
