"""Section 15.2 rows 1, 4, 5 and 6 on every named variant the engine sets up (MT-04; register C2,
C17, C19, C20, C23): the cached path equals one full block-causal pass on real engine positions.

``tests/net/test_cache.py`` holds the rows on synthetic multiverses; ``test_network.py`` and
``test_evaluator.py`` hold them on a few real games. Section 15.2's protocol names "every variant"
among its adversarial constructions, so here every variant of the engine's table
(``helpers.ENGINE_VARIANTS``, ``engine/src/variants.cpp``) is played at random from its start - two
seeds, travel favoured for White, so that timelines branch and some go inactive - and every state
of the play is evaluated: each turn start, each state inside an action (mid-action, D8), each
state whose present has shifted (with the shift reached, the action sometimes goes on with optional
moves before it is submitted), and the submitted state after it. The variants cover boards of 3x3
to 8x8, odd and even (+-0) starts, the T0 board, the royal queen and the princess (Reversed
Royalty), brawns (Just Brawns) and unicorns (Timeline Strategos); no variant of the table holds a
dragon. Engine positions use the lineage reading only (C23; spec issue 50).

* ``test_cached_equals_full_pass_on_every_variant``: the positions of a game evaluated
  incrementally - one epoch, one store shared along the game - against ``reference_output``,
  under both of C20's readings, with the v1 configuration (B15's table, B16's status key biases,
  A6's turn-state counts and A7's acting-player frame - each the default, A7 cached per
  perspective, spec issue 303) and with the step-1 heads and A8's board-layout embedding on.
* ``test_step_api_cached_equals_dense_on_every_variant``: the C step API's leaves (the engine's
  search, tree and graph mode, its evaluation cache off so that a transposition reaches the
  evaluator again), through ``CachedNetworkEvaluator`` and ``NetworkEvaluator(scope=
  "block_causal")`` - or, with C3's root refresh, against the two-level reference - and every
  leaf met twice (two move orders of one action, or two searches) evaluated bit for bit alike.
* ``test_every_n2_switch_with_relations_on_three_variants``: section 6's switches, with the
  typed relations the engine builds (``relations_for``: attack, defence, pin, x-ray, check).
* ``test_f1_bf16_cached_on_three_variants`` (slow): F1 as shipped but block-causal, BF16.

Tolerances are ``test_cache.py``'s: FP32 2e-5 on the trunk and 1e-4 on every head, bitwise where
two paths compute the same thing in the same order; the BF16 bounds of ``test_network.py``.
Random play is for exercising shapes, never evidence about real games (``play.py``).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from t5d.engine import Game
from t5d.engine.capi import T5D_MOVE_TRAVEL
from t5d.engine.evaluator import CachedNetworkEvaluator, NetworkEvaluator, _split, run_search, state_of
from t5d.engine.play import ROYAL_IDS
from t5d.net import NetConfig, build_model
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, state_digest
from t5d.net.state import PIECE_IDS, RELATION_IDS
from tests.engine.helpers import ENGINE_VARIANTS, requires_engine
from tests.net.test_cache import _close

pytestmark = requires_engine

R1, R2 = "past_and_epoch_history", "causal_past"
SEEDS = (0, 1)
BOARDS = 16  # random play stops at the first turn start with this many boards
PLAYOUTS = 16  # per search of the step-API test

#: The configurations of the main test: the v1's (B15, B16, A6 and A7 are its defaults), and
#: the same with the step-1 items that are off by default and that the cache supports - A8's
#: board-layout embedding (a prototype; per board and intrinsic, ``test_cache.py``'s a8 case),
#: D4's reply head and D9's optimistic policy (D3's heads and D4's soft policy are on in both).
CONFIGS = {
    "v1": {},
    "step1": dict(board_layout_embedding=True, opp_reply=True, optimistic_policy=True),
}

#: Section 6's switches (stage N2), each set with the typed relations it reads.
N2_FLAG_SETS = {
    "n2-all": dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, rope_per_head=True),
    "typed-directed": dict(typed_edge_bias=True, typed_edge_directed=True, routing_table=True, relations_head_direction="forward"),
    "routing-regs-self": dict(routing_table=True, masked_head_registers="self", same_board_registers="registers"),
    "mg4-raw-reverse": dict(move_graph_heads=4, rope_recenter=False, move_graph_direction="reverse"),
    "no-table-n2": dict(displacement_bias=False, routing_table=True, typed_edge_bias=True),
    "b17-n2-all": dict(shaw_key_term=True, shaw_init_std=1.0, routing_table=True, typed_edge_bias=True, board_pair_smolgen=True),
    "a8-n2-all": dict(board_layout_embedding=True, layout_padding="centre", routing_table=True, typed_edge_bias=True, rope_per_head=True),
}
THREE_VARIANTS = ("Standard - Two Timelines", "Misc - Timeline Battleground", "Very Small - Open")

#: The largest gaps met, per tolerance class (printed at the module's end under ``-s``).
GAPS = {"fp32 trunk": 0.0, "fp32 heads": 0.0, "step-api fp32": 0.0}


@pytest.fixture(scope="module", autouse=True)
def _report_gaps():
    yield
    print("\nlargest gaps, cached against dense:", {k: f"{v:.2e}" for k, v in GAPS.items()})


def _model(seed: int, dtype: str = "fp32", layers: int = 2, **flags):
    return build_model(NetConfig(trunk_dtype=dtype, n_layers=layers, attn_scope="block_causal", **flags), seed=seed).eval()


# --- random play: every state of every action ---------------------------------------------------


def _playable(game: Game) -> tuple[np.ndarray, object]:
    """The moves random play may make (never a royal capture) and the expansion."""
    x = game.expand(square_features="none")
    return np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS)), x


def _walk(variant: str, seed: int, boards: int = BOARDS, max_states: int = 60):
    """Seeded pseudo-legal random play from ``variant``'s start, yielding the game at every state
    (the same handle - read it before the next one) until a turn start with ``boards`` boards.
    Travel is drawn with probability 0.8 for White and 0.1 for Black (when both kinds exist), so
    that White's timelines outnumber Black's and some go inactive. With the present shifted the
    action goes on with an optional move three times in ten. A move after which the action can
    neither go on nor be submitted (Timeline Tactician's first action must jump into the board
    without a White piece, T13) is passed over for another."""
    rng = np.random.default_rng(seed)
    game = Game.variant(variant)
    game.set_legality(False)
    try:
        for _ in range(max_states):
            info = game.info()
            yield game
            if info.boards >= boards and info.moves_made == 0:
                return
            if info.obligations_met and (not info.may_move or rng.random() >= 0.3):
                game.submit()
                continue
            order = np.zeros(0, dtype=np.int64)
            if info.may_move:
                ok, x = _playable(game)
                travel = (x.moves["flags"][ok] & T5D_MOVE_TRAVEL) != 0
                if travel.any() and (~travel).any():
                    first = ok[travel] if rng.random() < (0.8 if info.to_move == 0 else 0.1) else ok[~travel]
                    order = np.concatenate([rng.permutation(first), rng.permutation(np.setdiff1d(ok, first))])
                else:
                    order = rng.permutation(ok)
            nxt = None
            for move in order.tolist():
                child = game.copy()
                child.set_legality(False)
                child.expand(square_features="none")
                child.apply(int(move))
                ci = child.info()
                if ci.obligations_met or (ci.may_move and _playable(child)[0].size):
                    nxt = child
                    break
                child.close()
            if nxt is None:
                if not info.obligations_met:
                    return  # stuck: no move lets the action go on
                game.submit()
                continue
            game.close()
            game = nxt
    finally:
        game.close()


def _positions(variant: str, seed: int, relations=()) -> list:
    return [g.position(relations=relations) for g in _walk(variant, seed)]


def _coverage(positions) -> dict[str, bool]:
    states = [p.state for p in positions]
    start = len(set(states[0].timeline.tolist()))
    return {
        "branched": any(len(set(s.timeline.tolist())) > start for s in states),
        "inactive": any(bool((~s.active).any()) for s in states),
        "mid-action": any(not s.shifted and int(s.turn_state[2]) > 0 for s in states),
        "shifted": any(s.shifted for s in states),
        "black acting": any(int(s.acting_player) == 1 for s in states),
    }


def _check(out, ref, what) -> None:
    """``test_cache.py``'s FP32 comparison of every output, the gaps recorded."""
    GAPS["fp32 trunk"] = max(GAPS["fp32 trunk"], float((out.trunk - ref.trunk).abs().max()))
    v = out.move_valid
    heads = [(out.value_logits, ref.value_logits), (out.wdl_logits, ref.wdl_logits), (out.submit_raw, ref.submit_raw)]
    heads += [(out.policy_logits[v], ref.policy_logits[v]), (out.board_logits, ref.board_logits)]
    GAPS["fp32 heads"] = max(GAPS["fp32 heads"], *(float((a - b).abs().max()) if a.numel() else 0.0 for a, b in heads))
    _close(out, ref, what=what)


# --- (a) and (b): cached = one full pass along a real game, every variant -----------------------


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_cached_equals_full_pass_on_every_variant(variant, config):
    """Section 15.2 rows 1 and 4 (C2, C17, C20): the positions of random play from ``variant``'s
    start - every turn start, mid-action, shifted and submitted state - evaluated incrementally in
    one epoch opened at a mid-game root, the store shared along the game, each equal to one full
    block-causal pass. Under ``causal_past`` (R2) the shared causal epoch evaluates the whole game,
    the root's past included; under ``past_and_epoch_history`` (R1) the root's frozen boards are
    the epoch's history and every later state is evaluated against the reference with that
    history. Two seeds per variant; together they branch, leave a timeline inactive, and reach
    every kind of state."""
    seen = dict.fromkeys(("branched", "inactive", "mid-action", "shifted", "black acting"), False)
    for seed in SEEDS:
        positions = _positions(variant, seed)
        seen = {k: seen[k] or v for k, v in _coverage(positions).items()}
        r = len(positions) // 3
        for supersession in (R2, R1):
            model = _model(10 * seed + (supersession == R1), supersession_scope=supersession, **CONFIGS[config])
            cached = CachedEvaluator(model)
            root = positions[r].state
            assert (~root.is_head).any(), "the root has a history"
            epoch = cached.root_epoch(root)
            history_from = root if supersession == R1 else None
            for i, pos in enumerate(positions if supersession == R2 else positions[r:]):
                ref = reference_output(model, pos.state, history_from=history_from)
                _check(cached.evaluate(pos.state, epoch), ref, what=(variant, config, seed, supersession, i))
            assert cached.store.hits > 0, "later states reuse earlier states' frozen boards"
    assert all(seen.values()), (variant, seen)


# --- (d): the C step API, cached against dense on the search's own leaves -----------------------


def _leaf_key(state) -> bytes:
    """What a leaf's evaluation is a function of: the state (its boards, flags, acting player and
    turn state - ``state_digest``), its candidate moves, edges and submit flags."""
    h = hashlib.blake2b(state_digest(state).encode(), digest_size=16)
    for a in (state.moves, state.edges):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(bytes([bool(state.submit_possible), bool(state.submit_legal)]))
    return h.digest()


def _evals_close(a, b, what) -> float:
    gap = 0.0
    for name in ("policy_logits", "wdl_logits", "value_logits"):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        np.testing.assert_allclose(x, y, atol=1e-4, rtol=0, err_msg=f"{what} {name}")
        gap = max(gap, float(np.abs(x - y).max()) if x.size else 0.0)
    for name in ("submit_logit", "error", "moves_left"):
        x, y = float(getattr(a, name)), float(getattr(b, name))
        assert np.isnan(x) == np.isnan(y) and (np.isnan(x) or abs(x - y) <= 1e-4 + 1e-5 * abs(y)), (what, name, x, y)
    return gap


def _evals_identical(a, b) -> bool:
    arrays = all(np.array_equal(getattr(a, n), getattr(b, n)) for n in ("policy_logits", "wdl_logits", "value_logits"))
    return arrays and all(np.array_equal(getattr(a, n), getattr(b, n), equal_nan=True) for n in ("submit_logit", "error", "moves_left"))


class _CachedAgainstDense:
    """The search's evaluator: ``CachedNetworkEvaluator``'s evaluations, each checked on its leaf
    against the dense path's - ``NetworkEvaluator(scope="block_causal")``, or with a root refresh
    the two-level reference (``reference_output`` with the search's root as the anchor) - and
    against every earlier evaluation of the same leaf, bit for bit (``seen``: shared across
    searches)."""

    def __init__(self, model, refresh: bool, seen: dict) -> None:
        self.model = model
        self.refresh = refresh
        self.cached = CachedNetworkEvaluator(CachedEvaluator(model, CacheConfig(root_refresh=refresh)))
        self.dense = NetworkEvaluator(model, scope="block_causal")
        self.root = None
        self.seen = seen
        self.leaves = self.repeats = 0
        self.keys: list[bytes] = []

    def start(self, root):
        self.root = state_of(root)
        return self.cached.start(root)

    def __call__(self, positions):
        evs = self.cached(positions)
        if self.refresh:
            refs = [_split(reference_output(self.model, state_of(p), anchor=self.root), [state_of(p)])[0] for p in positions]
        else:
            refs = self.dense(positions)
        for p, ev, ref in zip(positions, evs, refs, strict=True):
            GAPS["step-api fp32"] = max(GAPS["step-api fp32"], _evals_close(ev, ref, (self.refresh, self.leaves)))
            key = _leaf_key(state_of(p))
            self.keys.append(key)
            first = self.seen.setdefault(key, ev)
            if first is not ev:
                assert _evals_identical(first, ev), ("a leaf met twice is evaluated twice alike", self.refresh, self.leaves)
                self.repeats += 1
            self.leaves += 1
        return evs


def _search_root(variant: str, boards: int = 8) -> Game:
    """A copy of random play's first turn start with ``boards`` boards or more whose search has
    something to evaluate (random play is pseudo-legal: a turn start may be terminal, or play
    stuck before it - then the next seed's play)."""
    from t5d.engine.search import Search

    for seed in range(5, 15):
        for g in _walk(variant, seed, boards=boards):
            if g.info().boards >= boards and g.info().moves_made == 0:
                with Search(g) as probe:
                    if not probe.info().root_finished:
                        return g.copy()
    raise AssertionError(f"random play from {variant!r} reached no live turn start with {boards} boards")


def _search(model, root: Game, refresh: bool, graph: int, seen: dict, playouts: int = PLAYOUTS):
    """One search from ``root`` through the C step API with the cached path checked leaf by leaf
    (``_CachedAgainstDense``); its evaluation cache off. The evaluator and the search's summary."""
    from t5d.engine.search import Search, search_params

    params = search_params(seed=1 + graph, graph=graph, eval_cache_entries=0, batch_size=4)
    with Search(root, params=params) as search:
        ev = _CachedAgainstDense(model, refresh, seen)
        stats, _ = run_search(search, ev, playouts=playouts, batch_size=4)
        info = search.info()
    assert stats.leaves == ev.leaves > 0
    return ev, info


@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_step_api_cached_equals_dense_on_every_variant(variant):
    """Section 15.2 rows 1, 5 and 6 through the engine's search (the C step API, ``run_search``):
    from a random-play root with branched timelines, the search's leaves - mid-action, shifted and
    turn-start states, in the leaf's own move order - evaluated by the cached path equal the dense
    block-causal path's (with C3's root refresh, the two-level reference), in tree and in graph
    mode (C11). The search's evaluation cache is off, so a state the search reaches twice - two
    orders of one action's moves on two boards (an intra-turn transposition: the tree search meets
    it as two leaves; the graph search as one node, its second path a transposition hit) - reaches
    the evaluator again, as does a leaf the second search shares with the first; each such pair is
    bitwise identical, though the cached path's stores were filled along different paths."""
    model = _model(ENGINE_VARIANTS.index(variant))
    with _search_root(variant) as root:
        for refresh in (False, True):
            seen: dict = {}
            evs = [_search(model, root, refresh, graph, seen)[0] for graph in (0, 1)]
            assert evs[1].repeats > 0, (variant, refresh)  # the graph search met the tree search's leaves


@pytest.mark.parametrize("variant", ["Standard - Two Timelines", "Misc - Timeline Battleground"])
def test_two_orders_of_one_action_through_the_step_api(variant):
    """Section 15.2 row 4 (C20, C21) engine-side: at a root with several boards to move, the
    tree search (its evaluation cache off) reaches one state by two orders of moves and sends
    both leaves to the cached path, whose store reached each along its own path - the
    two evaluations are bitwise identical (``_CachedAgainstDense``); the graph search (C11) meets
    the same state as a transposition hit, and every leaf it evaluates is bitwise the tree
    search's. Both readings of C3: the causal epoch and the root refresh."""
    model = _model(2)
    with _search_root(variant) as root:
        assert int(root.position().state.turn_state[0]) >= 2, "several boards to move"
        for refresh in (False, True):
            seen: dict = {}
            tree, _ = _search(model, root, refresh, 0, seen, playouts=64)
            graph, info = _search(model, root, refresh, 1, seen, playouts=64)
            assert tree.leaves > len(set(tree.keys)), (variant, refresh, "no state reached twice")
            assert info.transposition_hits > 0 and graph.repeats > 0, (variant, refresh)


# --- every N2 switch, with the relations the engine builds ---------------------------------------


@pytest.mark.parametrize("flags", list(N2_FLAG_SETS))
def test_every_n2_switch_with_relations_on_three_variants(flags):
    """Row 1 under section 6's switches (B5, B9-B12, B15's no-table arm, B17, A8 with the
    structure), each network's positions converted with the relations it reads
    (``relations_for``) - attack, defence, pin, x-ray and check edges (B11, engine ABI round 3)
    among them - on three variants: 8x8 with an even start, 5x5 with three timelines, 4x4."""
    from t5d.engine import relations_for

    model = _model(7, **N2_FLAG_SETS[flags])
    rel = relations_for(model.cfg)
    types: set[int] = set()
    for variant in THREE_VARIANTS:
        positions = _positions(variant, seed=2, relations=rel)
        cached = CachedEvaluator(model)
        epoch = cached.root_epoch(positions[len(positions) // 3].state)
        for i, pos in enumerate(positions):
            types |= set(pos.state.edges[:, 2].tolist())
            _check(cached.evaluate(pos.state, epoch), reference_output(model, pos.state), what=(flags, variant, i))
    want = {RELATION_IDS["move"]} | {RELATION_IDS[r] for r in rel}
    assert types == want, (flags, sorted(types), sorted(want))


# --- F1 in BF16 -------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["Standard - Turn Zero", "Standard - Reversed Royalty", "Misc - Timeline Strategos"])
def test_f1_bf16_cached_on_three_variants(variant):
    """F1 as shipped but block-causal (BF16 trunk, 16 layers): cached against the BF16 full pass
    on random play, at ``test_network.py``'s BF16 bounds (``test_f1_cached_on_a_real_position``) -
    the T0 board, the royal queen and princess, the unicorn."""
    model = build_model(NetConfig(attn_scope="block_causal"), seed=0).eval()
    positions = _positions(variant, seed=3)[::2]
    special = {"Standard - Turn Zero": "the T0 board", "Standard - Reversed Royalty": "Y", "Misc - Timeline Strategos": "U"}[variant]
    if special in PIECE_IDS:
        assert any((p.state.piece == PIECE_IDS[special]).any() for p in positions), special
    else:
        assert any((p.state.half_turn < 2).any() for p in positions), special  # turn 0: half-turns 0 and 1
    cached = CachedEvaluator(model)
    epoch = cached.root_epoch(positions[len(positions) // 3].state)
    for i, pos in enumerate(positions):
        out = cached.evaluate(pos.state, epoch)
        ref = reference_output(model, pos.state)
        err = (out.trunk - ref.trunk).float()
        v = out.move_valid
        gaps = {
            "bf16 trunk rms": float(err.pow(2).mean().sqrt()),
            "bf16 trunk max": float(err.abs().max()),
            "bf16 policy": float((out.policy_logits[v] - ref.policy_logits[v]).abs().max()) if bool(v.any()) else 0.0,
            "bf16 value": float((out.value_logits - ref.value_logits).abs().max()),
        }
        for k, g in gaps.items():
            GAPS[k] = max(GAPS.get(k, 0.0), g)
        bounds = {"bf16 trunk rms": 8e-3, "bf16 trunk max": 5e-2, "bf16 policy": 3e-2, "bf16 value": 5e-3}
        assert all(gaps[k] < b for k, b in bounds.items()), (variant, i, gaps)
        assert torch.isfinite(out.value_logits).all()
