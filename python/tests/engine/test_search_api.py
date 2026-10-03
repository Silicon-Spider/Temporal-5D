"""Stage N5: the search's C step API from Python (``t5d/engine/search.py``) and the network as
its evaluator (``evaluator.CStepDriver``).

* The binding mirrors ``t5d/capi_search.h``'s structs exactly - checked at load against
  ``t5d_search_layout`` as ``t5d/capi.h``'s are against ``t5d_layout`` - and refuses a library
  laid out otherwise.
* ``CStepDriver`` is the engine's own run loop: driven by the engine's uniform evaluator
  rewritten in Python, a search is bit for bit the engine's ``t5d_search_run`` (H15's
  digest), across both arms of D2, tree and graph, root noise and the synchronous mode.
* Logit ``i`` scores ``leaf.expansion.moves[i]``; the submit logit reaches the submit child
  (D8); NaN gives the search's placeholder share.
* With the network (random weights) a search is deterministic: same seed, same digest.
* C14: a fresh search per decision shares the evaluation cache (``t5d_search_new_sharing``):
  the same tree as an unshared search, with fewer evaluations.
* Errors come back as exceptions, and a failed evaluation releases the batch.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from t5d.engine import Game, NetworkEvaluator, UniformEvaluator, run_search
from t5d.engine.capi import (
    MIRRORED_SEARCH_STRUCTS,
    T5D_EVALUATOR_UNIFORM,
    T5D_SPLIT_BOARD_MOVE,
    T5D_SPLIT_WHOLE_TURN,
    CLeaf,
    Library,
    layout_mismatches,
    load_library,
)
from t5d.engine.errors import EngineArgumentError, EngineNotFoundError
from t5d.engine.evaluator import CStepDriver, Evaluation, serve
from t5d.engine.play import ROYAL_IDS
from t5d.engine.search import Search, search_params
from t5d.net import NetConfig, build_model
from tests.engine.helpers import requires_engine

pytestmark = requires_engine


def _small_net(seed: int = 0):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=1, d_model=64, n_heads=2, policy_dim=64, value_hidden=64), seed=seed).eval()


def _shifted_game_with_moves(seed: int = 0) -> Game:
    """Seeded random play from Two Timelines to a state whose present has shifted and that still
    has a move (R8: moves after the shift): a mid-action state offering moves and submit."""
    rng = np.random.default_rng(seed)
    g = Game.variant("Standard - Two Timelines")
    g.set_legality(False)
    for _ in range(400):
        x = g.expand(square_features="none")
        if g.info().obligations_met:
            if x.move_count and x.submit_legal:
                g.set_legality(True)
                return g
            g.submit()
            continue
        g.apply(int(rng.choice(np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS)))))
    raise AssertionError("no move after a shift with a legal submit met")


# --- layouts --------------------------------------------------------------------------------------


def test_the_search_structs_mirror_the_library():
    lib = load_library()
    assert layout_mismatches(lib) == []
    for cname, struct in MIRRORED_SEARCH_STRUCTS.items():
        assert struct._fields_, cname
    assert CLeaf._fields_[-1][0] == "expansion" and MIRRORED_SEARCH_STRUCTS["t5d_search_params"]._fields_[-1][0] == "expand"


def test_a_library_laid_out_otherwise_is_refused(monkeypatch):
    """A binding whose t5d_leaf differs from the library's - here, before D8's three flags - names
    every difference and refuses to load the library."""
    lib = load_library()

    class OldLeaf(ctypes.Structure):
        _fields_ = [f for f in CLeaf._fields_ if f[0] not in ("acting_player", "submit_possible", "submit_legal")]

    monkeypatch.setitem(MIRRORED_SEARCH_STRUCTS, "t5d_leaf", OldLeaf)
    found = layout_mismatches(lib)
    assert any(m.startswith("t5d_leaf: size") for m in found) and any(m.startswith("t5d_leaf.legality: offset") for m in found)
    assert all(m.startswith("t5d_leaf") for m in found)
    with pytest.raises(EngineNotFoundError, match="t5d_leaf"):
        Library(lib.path)


# --- parameters -----------------------------------------------------------------------------------


def test_parameters_default_and_override():
    p = search_params()
    assert (p.split, p.proof_channel, p.batch_size, p.expand.square_features, p.expand.window_k) == (T5D_SPLIT_BOARD_MOVE, 0, 8, 0, 8)
    q = search_params(split=T5D_SPLIT_WHOLE_TURN, seed=9, expand=(4, "all", 0))
    assert (q.split, q.seed, q.expand.window_k) == (T5D_SPLIT_WHOLE_TURN, 9, 4)
    with pytest.raises(ValueError, match="no field"):
        search_params(playouts=3)
    with pytest.raises(ValueError, match="H41"):
        search_params(proof_channel=1)


# --- the driver is the engine's own loop ----------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [{}, {"graph": 1}, {"split": T5D_SPLIT_WHOLE_TURN}, {"root_noise": 1}, {"batch_size": 1, "synchronous": 1}],
    ids=["ply-tree", "ply-graph", "whole-turn", "noise", "synchronous"],
)
@pytest.mark.parametrize("variant", ["Standard", "Standard - Two Timelines"])
def test_the_step_loop_is_the_engines_run(variant, flags):
    """The engine's uniform evaluator, rewritten in Python and delivered through the step API by
    ``CStepDriver``, builds the engine's own search: H15's digest equal to ``t5d_search_run``'s."""
    with Game.variant(variant) as g:
        a = Search(g, params=search_params(seed=3, **flags))
        stats, driver = run_search(a, UniformEvaluator(), playouts=64, batch_size=int(a.params.batch_size))
        b = Search(g, params=search_params(seed=3, **flags))
        b.run(T5D_EVALUATOR_UNIFORM, 64)
    assert a.digest() == b.digest()
    assert a.info().playouts == b.info().playouts == 64 and driver.stop == "playouts"
    assert sum(stats.batch_sizes) == a.info().evaluations


# --- what the evaluations mean --------------------------------------------------------------------


class Ramp:
    """Logit i = i / 4 for move i of every leaf, no submit logit, a flat value."""

    def __call__(self, positions):
        return [Evaluation(np.arange(p.move_count, dtype=np.float32) / 4, np.zeros(51, np.float32)) for p in positions]


def test_logit_i_scores_move_i():
    """A leaf's logit i scores its expansion's move i: with logits rising with the index, the
    root children's priors are a softmax over their moves' indices in the root's move list."""
    with Game.variant("Standard") as g:
        texts = g.expand(square_features="none").move_texts()
        s = Search(g, params=search_params(seed=1, widening=0))
        run_search(s, Ramp(), playouts=1)
    children = s.children()  # widening off: every child is instantiated
    assert len(children) == len(texts) and all(len(c.moves) == 1 and not c.submit for c in children)
    index = {c.index: texts.index(c.moves[0].text()) for c in children}
    assert sorted(index.values()) == list(range(len(texts)))
    z = sum(math.exp(i / 4) for i in range(len(texts)))
    for c in children:
        assert c.prior == pytest.approx(math.exp(index[c.index] / 4) / z, rel=1e-5)


class Submitter:
    """Move logits 0; the submit logit given (NaN: none) where the leaf offers a legal submit."""

    def __init__(self, submit: float) -> None:
        self.submit = submit
        self.seen: list = []

    def __call__(self, positions):
        self.seen.extend(positions)
        out = []
        for p in positions:
            s = self.submit if (p.state.submit_possible and p.state.submit_legal) else math.nan
            out.append(Evaluation(np.zeros(p.move_count, np.float32), np.zeros(51, np.float32), submit_logit=s))
        return out


@pytest.mark.parametrize("submit", [6.0, -6.0, math.nan])
def test_the_submit_logit_reaches_the_submit_child(submit):
    """D8 under the board-move split: at a state past the shift with moves left, the root offers
    the moves and submit; the network's submit logit sets submit's prior beside the moves'; NaN
    (none given) leaves the search's placeholder share (0.5)."""
    g = _shifted_game_with_moves(seed=2)
    with g:
        s = Search(g, turn_start=False, params=search_params(seed=1, widening=0))
        ev = Submitter(submit)
        run_search(s, ev, playouts=1)
    root = ev.seen[0]
    assert root.state.shifted and root.state.submit_possible and root.state.submit_legal and root.move_count > 0
    children = s.children()
    sub = [c for c in children if c.submit]
    assert len(sub) == 1
    m = root.move_count
    expected = 0.5 if math.isnan(submit) else math.exp(submit) / (math.exp(submit) + m)
    assert sub[0].prior == pytest.approx(expected, rel=1e-5)


def test_whole_turn_leaves_offer_no_submit():
    """Under the whole-turn split every node is a turn start and the submit action is not a
    decision of its own (no end-of-turn token: spec issue 92)."""
    with Game.variant("Standard - Two Timelines") as g:
        s = Search(g, params=search_params(split=T5D_SPLIT_WHOLE_TURN, seed=2))
        ev = Submitter(5.0)
        run_search(s, ev, playouts=24)
    assert ev.seen and not any(p.state.submit_possible or p.state.shifted for p in ev.seen)


# --- the network as the evaluator ------------------------------------------------------------------


@pytest.mark.parametrize("split", [T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_WHOLE_TURN], ids=["ply", "whole-turn"])
def test_the_network_drives_a_search_deterministically(split):
    """Random weights, root noise on: the same seeds give the same search (H15's digest), and the
    evaluations reach the search - another network's seed gives another search."""
    model = _small_net(0)
    digests = []
    for net_seed in (0, 0, 1):
        net = model if net_seed == 0 else _small_net(net_seed)
        with Game.variant("Standard - Two Timelines") as g:
            s = Search(g, params=search_params(split=split, seed=5, root_noise=1))
            stats, driver = run_search(s, NetworkEvaluator(net, max_batch=8), playouts=48, batch_size=8)
        assert driver.stop == "playouts" and s.info().playouts == 48 and stats.leaves == s.info().evaluations
        digests.append(s.digest())
    assert digests[0] == digests[1] != digests[2]


def test_the_network_past_the_shift():
    """A mid-action root past the shift: the network's leaves carry the acting player (not the
    present's colour) and the submit action, and the search runs to its budget."""
    g = _shifted_game_with_moves(seed=4)
    with g:
        s = Search(g, turn_start=False, params=search_params(seed=2))
        seen = []

        class Watch(NetworkEvaluator):
            def __call__(self, positions):
                seen.extend(positions)
                return super().__call__(positions)

        _, driver = run_search(s, Watch(_small_net(2)), playouts=32)
    assert driver.stop == "playouts"
    root = seen[0].state
    assert root.shifted and root.acting_player != root.player_to_move and root.submit_possible
    assert any(c.submit for c in s.children())


def test_fresh_searches_share_the_evaluation_cache():
    """C14: a fresh search per decision sharing the previous search's evaluation cache builds
    the same tree as an unshared one (one leaf at a time, so a cache hit and an evaluation make
    the same visit) with fewer evaluations - the shared states come from the cache."""
    net = NetworkEvaluator(_small_net(3))
    with Game.variant("Standard - Two Timelines") as g:
        first = Search(g, params=search_params(seed=4, batch_size=1))
        run_search(first, net, playouts=40, batch_size=1)
        pick, _ = first.choose()
        child = first.child(pick)
        g.apply(child.moves[0])
        shared = Search(g, turn_start=False, params=search_params(seed=4, batch_size=1), share=first)
        assert shared.info().cache_entries > 0
        first.close()  # the cache lives while a search holds it
        run_search(shared, net, playouts=40, batch_size=1)
        alone = Search(g, turn_start=False, params=search_params(seed=4, batch_size=1))
        run_search(alone, net, playouts=40, batch_size=1)
    a, b = shared.info(), alone.info()
    assert shared.digest() == alone.digest()
    assert a.cache_hits > 0 and a.evaluations < b.evaluations


# --- errors ---------------------------------------------------------------------------------------


def test_errors_come_back_as_exceptions():
    with Game.variant("Standard") as g:
        s = Search(g)
        with pytest.raises(EngineArgumentError):
            s.deliver(np.zeros(3, np.float32), np.zeros(51, np.float32))  # no batch waiting
        n = s.gather(4)
        assert n == 1
        with pytest.raises(EngineArgumentError, match="batch is waiting"):
            s.gather(4)
        with pytest.raises(EngineArgumentError):
            s.set_epoch(1)
        with pytest.raises(EngineArgumentError, match="logits"):
            s.deliver(np.zeros(3, np.float32), np.zeros(51, np.float32))  # 20 candidate moves
        assert s.gather(4) == 1  # the refused batch was abandoned
        m = s.leaf_view(0).expansion.move_count
        with pytest.raises(EngineArgumentError, match="infinite submit"):
            s.deliver(np.zeros(m, np.float32), np.zeros(51, np.float32), submit_logits=[math.inf])
        s.close()
        with pytest.raises(EngineArgumentError, match="closed"):
            s.info()


def test_a_failed_evaluation_releases_the_batch():
    def broken(positions):
        raise RuntimeError("the evaluator failed")

    with Game.variant("Standard") as g:
        s = Search(g)
        driver = CStepDriver(s, playouts=8)
        with pytest.raises(RuntimeError, match="evaluator failed"):
            serve(driver, broken, batch_size=4)
        assert s.gather(4) == 1  # no batch was left waiting
        s.abandon()


def test_the_driver_needs_every_boards_features():
    with Game.variant("Standard") as g:
        s = Search(g, params=search_params(expand=(8, "scope", 0)))
        with pytest.raises(ValueError, match="every board's features"):
            CStepDriver(s, playouts=4)


def test_the_leaf_carries_d8():
    """The leaf's own D8 fields equal its expansion's under the board-move split."""
    g = _shifted_game_with_moves(seed=6)
    with g:
        s = Search(g, turn_start=False)
        assert s.gather(1) == 1
        v = s.leaf_view(0)
        x = v.expansion
        assert (v.acting_player, v.submit_possible, v.submit_legal) == (x.acting_player, x.submit_possible, x.submit_legal)
        assert v.submit_possible and v.acting_player != x.present & 1
        assert v.legality is None or v.legality.shape == (x.move_count,)
        s.abandon()
