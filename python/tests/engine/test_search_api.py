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
    T5D_EVALUATOR_RANDOM,
    T5D_EVALUATOR_UNIFORM,
    T5D_SPLIT_BOARD_MOVE,
    T5D_SPLIT_WHOLE_TURN,
    T5D_VALUE_PROBABILITIES,
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
    # E1, I19 (engine ABI round 2b): the placeholders, the library's own defaults.
    assert p.dirichlet_concentration == pytest.approx(0.03 * 361) and p.dirichlet_shaping == 0
    assert (p.dirichlet_shaping_share, p.dirichlet_shaping_cap, p.root_policy_temperature) == (0.5, 0.01, 1.0)
    with Game.variant("Standard") as g, pytest.raises(EngineArgumentError, match="root_policy_temperature"):
        Search(g, params=search_params(root_policy_temperature=0.0))


# --- the driver is the engine's own loop ----------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"graph": 1},
        {"split": T5D_SPLIT_WHOLE_TURN},
        {"root_noise": 1},
        {"root_noise": 1, "dirichlet_shaping": 1, "root_policy_temperature": 1.3},
        {"root_noise": 1, "dirichlet_concentration": 0.0},
        {"batch_size": 1, "synchronous": 1},
    ],
    ids=["ply-tree", "ply-graph", "whole-turn", "noise", "noise-shaped-tempered", "noise-per-child", "synchronous"],
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


_M64 = (1 << 64) - 1


def _mix(z: int) -> int:
    """The engine's 64-bit mixer (``src/mcts_evaluators.cpp``)."""
    z &= _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def _unit(x: int) -> float:
    return float(x >> 11) * 2.0**-53


class PyRandom:
    """The engine's built-in ``random`` evaluator (``src/mcts_evaluators.cpp``) rewritten in
    Python - a function of the seed, each leaf's key and its epoch (``CStepDriver.batch``) - with
    its WDL and categorical as exact probabilities (``T5D_VALUE_PROBABILITIES``)."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.driver: CStepDriver | None = None
        self.shifted = 0

    def __call__(self, positions):
        views = self.driver.batch
        assert len(views) == len(positions)
        out = []
        for v, p in zip(views, positions, strict=True):
            n = v.expansion.move_count
            assert p.move_count == n
            self.shifted += p.state.shifted
            base = _mix(self.seed ^ _mix(v.key ^ _mix(v.epoch + 0x51)))
            logits = np.zeros(n, np.float32)
            for m in range(n):
                r = _mix(base + 0x9E3779B97F4A7C15 * (m + 1))
                logits[m] = np.float32(2.0 * (_unit(r) + _unit(_mix(r)) + _unit(_mix(r + 1)) - 1.5))
            mean = 1.6 * _unit(_mix(base ^ 0xA5A5)) - 0.8
            sd = 0.05 + 0.35 * _unit(_mix(base ^ 0x5A5A))
            bump = [math.exp(-0.5 * (((2.0 * k - 50) / 50.0 - mean) / sd) ** 2) for k in range(51)]
            total = 0.0
            for b in bump:
                total += b
            draw = 0.5 * _unit(_mix(base ^ 0xD4D4))
            wdl = np.array([(1.0 - draw) * (1.0 + mean) * 0.5, draw, (1.0 - draw) * (1.0 - mean) * 0.5], np.float32)
            out.append(Evaluation(logits, wdl, np.array([np.float32(b / total) for b in bump], np.float32)))
        return out


def _random_digests(game: Game, turn_start: bool, flags: dict, evaluator_seed: int, playouts: int = 96) -> tuple[int, int, int]:
    """(the step loop's digest with ``PyRandom``, the engine's own run's digest, shifted leaves met)."""
    a = Search(game, turn_start=turn_start, params=search_params(**flags))
    ev = PyRandom(evaluator_seed)
    ev.driver = CStepDriver(a, playouts=playouts, value_kind=T5D_VALUE_PROBABILITIES)
    serve(ev.driver, ev, batch_size=int(a.params.batch_size))
    b = Search(game, turn_start=turn_start, params=search_params(**flags))
    b.run(T5D_EVALUATOR_RANDOM, playouts, seed=evaluator_seed)
    out = (a.digest(), b.digest(), ev.shifted)
    a.close()
    b.close()
    return out


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"graph": 1},
        {"split": T5D_SPLIT_WHOLE_TURN},
        {"root_noise": 1},
        {"root_noise": 1, "dirichlet_shaping": 1, "root_policy_temperature": 1.3},
        {"root_noise": 1, "dirichlet_concentration": 0.0},
        {"batch_size": 1, "synchronous": 1},
    ],
    ids=["ply-tree", "ply-graph", "whole-turn", "noise", "noise-shaped-tempered", "noise-per-child", "synchronous"],
)
@pytest.mark.parametrize("variant", ["Standard", "Standard - Two Timelines"])
def test_the_engines_random_evaluator_through_the_step_api(variant, flags):
    """The engine's ``random`` evaluator rewritten in Python and delivered by the unmodified
    ``CStepDriver.deliver`` - distinct logits per move and values that vary leaf by leaf - builds
    the engine's own search, bit for bit: any change to the order of the logits or of the value
    bins, at any leaf, changes the digest."""
    for seed in (1, 7):
        with Game.variant(variant) as g:
            mine, engine, shifted = _random_digests(g, True, dict(flags, seed=seed), seed * 31 + 5)
        assert mine == engine, seed
        if flags.get("split") != T5D_SPLIT_WHOLE_TURN:
            assert shifted > 0  # leaves past the shift were evaluated


@pytest.mark.parametrize("graph", [0, 1], ids=["tree", "graph"])
@pytest.mark.parametrize("game_seed", [2, 4, 6])
def test_the_random_evaluator_from_a_root_past_the_shift(game_seed, graph):
    g = _shifted_game_with_moves(seed=game_seed)
    with g:
        mine, engine, shifted = _random_digests(g, False, {"seed": 3, "graph": graph}, 99)
    assert mine == engine and shifted > 0


class WhiteWins:
    """Value +1 for White, expressed from each leaf's acting player - the network's convention
    and the search's: the WDL's win where White acts, its loss where Black acts (and the
    categorical, which the search does not back up, the other way round). ``perspective=
    "present"`` expresses it from the present's colour instead - the negative control, wrong
    exactly at leaves past the shift."""

    def __init__(self, perspective: str = "acting") -> None:
        self.perspective = perspective
        self.leaves = 0
        self.shifted = 0

    def __call__(self, positions):
        out = []
        for p in positions:
            s = p.state
            who = s.acting_player if self.perspective == "acting" else s.player_to_move
            self.leaves += 1
            self.shifted += s.shifted
            wdl = np.full(3, -1.0e4, np.float32)
            wdl[0 if who == 0 else 2] = 0.0
            v = np.full(51, -1.0e4, np.float32)
            v[0 if who == 0 else 50] = 0.0  # the opposite: D3's value is the WDL alone
            out.append(Evaluation(np.zeros(p.move_count, np.float32), wdl, v))
        return out


def _off_sign(search: Search, root_player: int) -> list[float]:
    """The visited root children's q (the root player's view) that are not White's win."""
    want = 1.0 if root_player == 0 else -1.0
    return [c.q for c in search.children() if c.visits > 0 and abs(c.q - want) > 1e-9]


@pytest.mark.parametrize("graph", [0, 1], ids=["tree", "graph"])
@pytest.mark.parametrize("split", [T5D_SPLIT_BOARD_MOVE, T5D_SPLIT_WHOLE_TURN], ids=["ply", "whole-turn"])
@pytest.mark.parametrize("variant", ["Standard", "Standard - Two Timelines"])
def test_the_value_is_the_acting_players(variant, split, graph):
    """A value that says White wins, given from each leaf's acting player, backs up through the
    engine as White's win: every visited root child's q is +1 for White at the root - leaves past
    the shift included (the board-move split meets them after the move that shifts the present)."""
    with Game.variant(variant) as g:
        s = Search(g, params=search_params(seed=1, split=split, graph=graph))
        ev = WhiteWins()
        run_search(s, ev, playouts=200)
    info = s.info()
    assert info.checkmates == info.stalemates == info.turn_caps == 0  # the premise: no game end in the tree
    assert sum(c.visits > 0 for c in s.children()) > 1 and _off_sign(s, 0) == []
    if split == T5D_SPLIT_BOARD_MOVE:
        assert ev.shifted > 0


@pytest.mark.parametrize("graph", [0, 1], ids=["tree", "graph"])
@pytest.mark.parametrize("game_seed", [2, 4, 6])
def test_the_value_is_the_acting_players_from_a_root_past_the_shift(game_seed, graph):
    g = _shifted_game_with_moves(seed=game_seed)
    with g:
        root_player = g.info().to_move
        s = Search(g, turn_start=False, params=search_params(seed=1, graph=graph))
        ev = WhiteWins()
        run_search(s, ev, playouts=200)
    info = s.info()
    assert info.checkmates == info.stalemates == 0
    assert ev.shifted > 0 and sum(c.visits > 0 for c in s.children()) >= 1
    assert _off_sign(s, root_player) == []


def test_the_present_colour_is_not_the_acting_player():
    """The negative control: from the present's colour the same values are wrong past the shift,
    and some root child's q comes out off-sign - the test above can fail."""
    g = _shifted_game_with_moves(seed=2)
    with g:
        root_player = g.info().to_move
        s = Search(g, turn_start=False, params=search_params(seed=1))
        ev = WhiteWins(perspective="present")
        run_search(s, ev, playouts=200)
    assert ev.shifted > 0 and _off_sign(s, root_player) != []


class Heads:
    """D3's heads at every leaf, from White's view: W - L = ``white`` with a draw probability
    ``draw`` (as probabilities); a categorical whose mean is not the value; the error head's
    output and moves-left."""

    def __init__(self, white: float, draw: float, error: float = 0.25, moves_left: float = 7.5) -> None:
        self.white, self.draw, self.error, self.moves_left = white, draw, error, moves_left

    def __call__(self, positions):
        out = []
        for p in positions:
            v = self.white if p.state.acting_player == 0 else -self.white
            wdl = np.array([(1 - self.draw + v) / 2, self.draw, (1 - self.draw - v) / 2], np.float32)
            cat = np.zeros(51, np.float32)
            cat[3] = 1.0  # -0.88 (probabilities): never the value
            out.append(Evaluation(np.zeros(p.move_count, np.float32), wdl, cat, error=self.error, moves_left=self.moves_left))
        return out


@pytest.mark.parametrize("graph", [0, 1], ids=["tree", "graph"])
def test_d3_the_wdl_is_the_search_value_and_its_draw_is_backed_up(graph):
    """D3 through the step API: the WDL's W - L is what the search backs up (the categorical
    says otherwise), its draw probability is backed up beside it, and every root child reports
    the search's (q, d) and the network's own WDL, error and moves-left, in the root player's view
    - as does the root itself (I13's and I15's memo)."""
    with Game.variant("Standard - Two Timelines") as g:
        s = Search(g, params=search_params(seed=1, graph=graph))
        driver = CStepDriver(s, playouts=200, value_kind=T5D_VALUE_PROBABILITIES)
        serve(driver, Heads(0.4, 0.2), batch_size=8)
    info = s.info()
    assert info.checkmates == info.stalemates == info.turn_caps == 0
    assert info.root_q == pytest.approx(0.4, abs=1e-5) and info.root_d == pytest.approx(0.2, abs=1e-5)
    assert info.root_evaluated and info.root_wdl == pytest.approx((0.6, 0.2, 0.2), abs=1e-6)
    assert (info.root_error, info.root_moves_left) == (0.25, 7.5)
    visited = [c for c in s.children() if c.visits > 0]
    assert len(visited) > 1
    for c in visited:
        assert c.q == pytest.approx(0.4, abs=1e-5) and c.d == pytest.approx(0.2, abs=1e-5)
        assert c.evaluated and c.network_wdl == pytest.approx((0.6, 0.2, 0.2), abs=1e-6)
        assert (c.network_error, c.network_moves_left) == (0.25, 7.5)
    s.close()


def test_d3_6m_the_uncertainty_source_is_a_parameter():
    """6m (register I10): the categorical's variance by default, the error head's output on
    request - a node's variance for E2: with E2 on the two signals grow different searches."""
    from t5d.engine.capi import T5D_UNCERTAINTY_CATEGORICAL, T5D_UNCERTAINTY_ERROR_HEAD

    assert search_params().uncertainty_source == T5D_UNCERTAINTY_CATEGORICAL
    digests = []
    for source in (T5D_UNCERTAINTY_CATEGORICAL, T5D_UNCERTAINTY_ERROR_HEAD):
        with Game.variant("Standard") as g:
            s = Search(g, params=search_params(seed=1, variance_scaled_cpuct=1, uncertainty_source=source))
            serve(CStepDriver(s, playouts=120, value_kind=T5D_VALUE_PROBABILITIES), Heads(0.1, 0.3, error=0.9), batch_size=8)
            digests.append(s.digest())
            s.close()
    assert digests[0] != digests[1]


# --- what the evaluations mean --------------------------------------------------------------------


class Ramp:
    """Logit i = i / 4 for move i of every leaf, no submit logit, a flat value."""

    def __call__(self, positions):
        return [Evaluation(np.arange(p.move_count, dtype=np.float32) / 4, np.zeros(3, np.float32)) for p in positions]


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
            out.append(Evaluation(np.zeros(p.move_count, np.float32), np.zeros(3, np.float32), np.zeros(51, np.float32), submit_logit=s))
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


@pytest.mark.parametrize(
    "change,named",
    [
        ({"split": T5D_SPLIT_WHOLE_TURN}, "split"),
        ({"expand": (4, "all", 0)}, "expand.window_k"),
        ({"expand": (8, "all", 1 << 1)}, "expand.relations"),
        ({"legal_moves_only": 0}, "legal_moves_only"),
        ({"legality_checks": 7}, "legality_checks"),
    ],
)
def test_a_cache_is_shared_only_between_searches_with_one_input(change, named):
    """The review: a donor whose leaves give the network another input - other expansion
    options, the other split, another legality classification of the candidate moves - would
    serve its evaluations for the wrong input. Refused, naming what differs; the evaluation-
    neutral parameters may differ."""
    with Game.variant("Standard") as g:
        donor = Search(g, params=search_params(seed=1))
        with pytest.raises(ValueError, match=named):
            Search(g, params=search_params(seed=1, **change), share=donor)
        Search(g, params=search_params(seed=2, cpuct=2.0, root_noise=1, batch_size=3), share=donor).close()
        donor.close()


@pytest.mark.parametrize("refresh", [False, True], ids=["causal", "root-refresh"])
def test_the_cached_path_keeps_its_epochs_apart_in_a_shared_cache(refresh):
    """C14 with the cached path (C2): each search opens the evaluator's epoch at its root and the
    engine caches evaluations per (state, moves made, epoch) - so with C3's root refresh a fresh search
    sharing the previous one's cache reuses none of its evaluations (each root conditions its
    own history), while in the shared causal epoch it reuses them."""
    from t5d.engine.evaluator import CachedNetworkEvaluator, engine_epoch
    from t5d.net.cache import CacheConfig, CachedEvaluator

    model = build_model(
        NetConfig(trunk_dtype="fp32", n_layers=1, d_model=64, n_heads=2, policy_dim=64, value_hidden=64, attn_scope="block_causal"), seed=5
    ).eval()
    ev = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=refresh)))
    params = dict(seed=4, batch_size=1)
    with Game.variant("Standard - Two Timelines") as g:
        first = Search(g, params=search_params(**params))
        run_search(first, ev, playouts=24, batch_size=1)
        first_epoch = first.info().epoch
        g.apply(first.child(first.choose()[0]).moves[0])
        shared = Search(g, turn_start=False, params=search_params(**params), share=first)
        first.close()
        run_search(shared, ev, playouts=24, batch_size=1)
        alone = Search(g, turn_start=False, params=search_params(**params))
        run_search(alone, ev, playouts=24, batch_size=1)
    a, b = shared.info(), alone.info()
    assert a.epoch == b.epoch == engine_epoch(ev.epoch) and first_epoch != 0
    assert shared.digest() == alone.digest()
    if refresh:
        assert a.epoch != first_epoch and a.evaluations == b.evaluations  # nothing reused across refreshes
    else:
        assert a.epoch == first_epoch and a.evaluations < b.evaluations  # the causal epoch is shared


# --- errors ---------------------------------------------------------------------------------------


def test_errors_come_back_as_exceptions():
    with Game.variant("Standard") as g:
        s = Search(g)
        with pytest.raises(EngineArgumentError):
            s.deliver(np.zeros(3, np.float32), np.zeros(3, np.float32))  # no batch waiting
        n = s.gather(4)
        assert n == 1
        with pytest.raises(EngineArgumentError, match="batch is waiting"):
            s.gather(4)
        with pytest.raises(EngineArgumentError):
            s.set_epoch(1)
        with pytest.raises(EngineArgumentError, match="logits"):
            s.deliver(np.zeros(3, np.float32), np.zeros(3, np.float32))  # 20 candidate moves
        assert s.gather(4) == 1  # the refused batch was abandoned
        m = s.leaf_view(0).expansion.move_count
        with pytest.raises(EngineArgumentError, match="infinite submit"):
            s.deliver(np.zeros(m, np.float32), np.zeros(3, np.float32), submit_logits=[math.inf])
        assert s.gather(4) == 1
        with pytest.raises(EngineArgumentError, match="WDL"):
            s.deliver(np.zeros(m, np.float32), np.zeros(2, np.float32))  # D3: 3 per leaf
        assert s.gather(4) == 1
        with pytest.raises(EngineArgumentError, match="error head"):
            s.deliver(np.zeros(m, np.float32), np.zeros(3, np.float32), errors=[-1.0])
        assert s.gather(4) == 1
        with pytest.raises(ValueError, match="wdl_kind"):
            s.deliver(np.zeros(m, np.float32), np.zeros(3, np.float32), wdl_kind=9)
        s.deliver(np.zeros(m, np.float32), np.zeros(3, np.float32))  # the optional heads absent
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


# --- A6: the root's moves made in its action ------------------------------------------------------


def test_the_root_takes_the_games_moves_made_a6():
    """Register A6: a search's root takes the game handle's count of moves made in its action -
    ``turn_start`` defaults to it, and a ``turn_start`` that contradicts it is refused - and every
    leaf's expansion carries its own count: 0 at a turn start (which always owes a board, T5),
    the root's + 1 per move below it."""
    with Game.variant("Standard - Two Timelines") as g:
        with pytest.raises(EngineArgumentError, match="moves in its action"):
            Search(g, turn_start=False)
        s = Search(g)
        assert s.info().root_turn_start and s.root_game().info().moves_made == 0
        s.close()
        g.apply(g.moves()[0])
        assert g.info().moves_made == 1
        with pytest.raises(EngineArgumentError, match="moves in its action"):
            Search(g, turn_start=True)
        s = Search(g, params=search_params(seed=3))
        assert not s.info().root_turn_start
        with s.root_game() as root:
            assert root.info().moves_made == 1
        s.run(T5D_EVALUATOR_RANDOM, 120, seed=3)
        n = s.gather(16)
        counts = []
        for i in range(n):
            x = s.leaf_view(i).expansion
            counts.append(x.moves_made)
            assert x.moves_made >= 0 and (x.moves_made > 0 or x.mandatory_remaining > 0)
            assert (x.mandatory_remaining == 0) == x.submit_possible
        s.abandon()
        assert n > 0 and max(counts) >= 2  # moves below the root's first
        s.close()


def test_the_root_takes_a_count_the_turn_start_flag_cannot_give_a6():
    """Two moves into an action: the root's count is the handle's 2, not the 0 or 1 a boolean
    reading of it would give (the count of 2 in the test above comes from moves below the root,
    not from the handle). Three White boards in the present, all mandatory (T5): two played, one
    owed - the root's own leaf counts 2, a move below it 3; a search sharing the first's cache
    takes the handle's count too."""
    boards = ["[3k/4/4/K3:-1:1:w]", "[3k/4/4/K3:0:1:w]", "[3k/4/4/K3:+1:1:w]"]
    with Game.from_fen(boards) as g:
        for timeline in (-1, 0):
            g.apply(next(m for m in g.moves() if m.from_timeline == m.to_timeline == timeline and m.from_half_turn == m.to_half_turn))
        assert g.info().moves_made == 2
        with pytest.raises(EngineArgumentError, match="moves in its action"):
            Search(g, turn_start=True)
        s = Search(g, params=search_params(seed=3))
        shared = Search(g, params=search_params(seed=3), share=s)
        for each in (s, shared):
            with each.root_game() as root:
                assert root.info().moves_made == 2
            assert each.gather(1) == 1
            x = each.leaf_view(0).expansion  # the root's own
            assert (x.moves_made, x.mandatory_remaining, bool(x.submit_possible)) == (2, 1, False)
            each.abandon()
        shared.close()
        s.run(T5D_EVALUATOR_RANDOM, 40, seed=3)
        s.advance(next(c.index for c in s.children() if not c.submit and c.visits > 0))
        with s.root_game() as root:
            assert root.info().moves_made == 3
        s.close()
