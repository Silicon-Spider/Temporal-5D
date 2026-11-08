"""Section 15.2's C27 row on real engine positions and through the engine's tree form: "Tree form with
the cached configuration on: supersession recompute at the next evaluated node = recompute at
supersession | The deferral is exact - the mask is state-defined | C27".

``tests/net/test_turn_refresh.py`` holds the row on the search-tree fixture. Here:

* T8, real actions (``test_cached_variants._walk``: seeded random play from each variant's start,
  every state of every action): the states the network evaluates are the turn starts - with, by
  marking, the first state of each action whose obligations are met (the present gate read
  literally, issue 160) or a seeded half of the mid-action states - and the others are the tree
  form's inner states. The deferred arm computes nothing at an inner state; the eager arm computes
  there every frozen board without an entry, from both colours' sides. Every evaluated state is bit
  for bit alike in the two arms and the from-scratch reference; the deferred arm's entries are the
  eager arm's; and some evaluated state reads an entry the eager arm computed at an inner state
  before it, which the deferred arm had not stored - on an action of two moves or more. Every engine
  variant, both C20 readings, the causal epoch from the walk's start and a root refresh at an
  action's turn start.
* T9, the step API's tree form (``T5D_SPLIT_TREE``), its natural gates and none: the evaluator is
  asked only at the root, turn starts and gated states, and every leaf is the from-scratch reference
  - causal (the block-causal pass), with the root refresh (the two-level reference) and under C26's
  phase form at N* = the root's boards; with no gates, a turn start's evaluation computes boards the
  turn's inner moves superseded (the deferral at work); and against the same root searched with
  every gate firing (the ply split's leaves, ``T5D_TREE_GATE_ALWAYS``), its own evaluator and store,
  every leaf both searches meet is bit for bit alike - turn starts among them.
* T10, the flat form: every leaf a turn start offering no submit, each the reference; nothing
  inherited.

FP32, 2 layers, random weights at initialisation scale; ``test_cache.py``'s tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.engine.capi import T5D_SPLIT_FLAT, T5D_SPLIT_TREE, T5D_TREE_GATE_ALWAYS
from t5d.engine.evaluator import CachedNetworkEvaluator, NetworkEvaluator, _split, state_of
from t5d.net import NetConfig, build_model
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, viewed_by
from tests.engine.helpers import ENGINE_VARIANTS, requires_engine
from tests.engine.test_cached_variants import THREE_VARIANTS, _evals_close, _search_root, _walk
from tests.engine.test_phase_form import _same
from tests.net.test_cache import _close, _identical

pytestmark = requires_engine

R1, R2 = "past_and_epoch_history", "causal_past"
MARKINGS = ("turn-starts", "present-gate", "half")
QUIET = dict(hunt=0, settling=0, asset_map=0, verification_threads=1)
#: The largest gaps met against the references (printed at the module's end under ``-s``).
GAPS = {"real actions, fp32 trunk": 0.0, "real actions, fp32 heads": 0.0, "step api, fp32 logits": 0.0}


@pytest.fixture(scope="module", autouse=True)
def _report_gaps():
    yield
    print("\nC27 on engine positions, largest gaps against the references:", {k: f"{v:.2e}" for k, v in GAPS.items()})


def _model(seed: int, dtype: str = "fp32", **flags):
    return build_model(NetConfig(trunk_dtype=dtype, n_layers=2, attn_scope="block_causal", **flags), seed=seed).eval()


# --- T8: real actions --------------------------------------------------------------------------------


def _states(variant: str, seed: int, boards: int = 12) -> list:
    return [g.position().state for g in _walk(variant, seed, boards=boards)]


def _evaluated(states: list, marking: str, rng: np.random.Generator) -> list[bool]:
    """Which states the network evaluates: every turn start (no move made in its action), and by
    ``marking`` also the first state of each action whose obligations are met (``present-gate``) or a
    seeded half of the mid-action states (``half``)."""
    out = []
    met_in_action = False
    for s in states:
        made = int(s.turn_state[2])
        if made == 0:
            met_in_action = False
            out.append(True)
            continue
        if marking == "present-gate" and s.submit_possible and not met_in_action:
            met_in_action = True
            out.append(True)
        elif marking == "half":
            out.append(bool(rng.random() < 0.5))
        else:
            out.append(False)
    return out


def _eager_at(ev: CachedEvaluator, state, epoch) -> set:
    before = set(ev.store.entries)
    for view in (state, viewed_by(state, 1 - int(state.acting_player))):
        work = ev.prepare(view, epoch)
        for b in ev.missing(work):
            ev.compute_board(work, b)
    return set(ev.store.entries) - before


def _reads(ev: CachedEvaluator, state, epoch) -> set:
    plan = ev._plan(state, epoch)
    return {plan.key(epoch, int(b)) for b in plan.frozen}


def _t8_case(model, states, evaluated, root_index: int, refresh: bool, supersession: str, what) -> int:
    """The deferred and the eager arm along ``states[root_index:]``; returns how many evaluated states
    read an entry the eager arm computed at an inner state before them and the deferred arm had not."""
    config = CacheConfig(root_refresh=refresh)
    deferred, eager = CachedEvaluator(model, config), CachedEvaluator(model, config)
    root = states[root_index]
    d_epoch, s_epoch = deferred.root_epoch(root), eager.root_epoch(root)
    at_inner: set = set()
    reads = 0
    for i in range(root_index, len(states)):
        state = states[i]
        if not evaluated[i]:
            at_inner |= _eager_at(eager, state, s_epoch)
            continue
        stored = set(deferred.store.entries)
        out = deferred.evaluate(state, d_epoch)
        assert _identical(out, eager.evaluate(state, s_epoch)), (what, i)
        if refresh:
            ref = reference_output(model, state, anchor=root)
        else:
            ref = reference_output(model, state, history_from=root if supersession == R1 else None)
        if out.trunk.dtype == torch.float32:
            GAPS["real actions, fp32 trunk"] = max(GAPS["real actions, fp32 trunk"], float((out.trunk - ref.trunk).abs().max()))
            v = out.move_valid
            heads = [(out.value_logits, ref.value_logits), (out.wdl_logits, ref.wdl_logits), (out.policy_logits[v], ref.policy_logits[v])]
            GAPS["real actions, fp32 heads"] = max(
                GAPS["real actions, fp32 heads"], *(float((a - b).abs().max()) if a.numel() else 0.0 for a, b in heads)
            )
        _close(out, ref, what=(what, i))
        reads += bool((_reads(deferred, state, d_epoch) & at_inner) - stored)
    assert deferred.store.entries.keys() <= eager.store.entries.keys(), what
    for k, e in deferred.store.entries.items():
        f = eager.store[k]
        assert torch.equal(e.kv, f.kv) and torch.equal(e.out, f.out) and e.deps == f.deps, what
    return reads


def _long_action(states: list, root_index: int) -> bool:
    """Whether an action of two moves or more lies after ``root_index``."""
    return any(int(s.turn_state[2]) >= 2 for s in states[root_index:])


@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_the_deferral_on_real_actions(variant):
    """T8 on every engine variant, seed 0: the causal epoch from the walk's start under C20's
    ``causal_past``, and a root refresh at a turn start a third of the way in under
    ``past_and_epoch_history``; the marking rotates over the variants."""
    k = ENGINE_VARIANTS.index(variant)
    states = _states(variant, 0)
    starts = [i for i, s in enumerate(states) if int(s.turn_state[2]) == 0]
    third = next(i for i in starts if i >= len(states) // 3)
    for case, (supersession, refresh, root_index) in enumerate(((R2, False, 0), (R1, True, third))):
        model = _model(700 + k, supersession_scope=supersession)
        marking = MARKINGS[(k + case) % len(MARKINGS)]
        evaluated = _evaluated(states, marking, np.random.default_rng([k, case]))
        what = (variant, supersession, refresh, marking)
        reads = _t8_case(model, states, evaluated, root_index, refresh, supersession, what)
        if _long_action(states, root_index):
            assert reads > 0, (what, "the deferral is real")


@pytest.mark.slow
@pytest.mark.parametrize("variant", ENGINE_VARIANTS)
def test_the_deferral_on_real_actions_seed_1(variant):
    """T8 with seed 1's play and every marking."""
    k = ENGINE_VARIANTS.index(variant)
    states = _states(variant, 1)
    starts = [i for i, s in enumerate(states) if int(s.turn_state[2]) == 0]
    third = next(i for i in starts if i >= len(states) // 3)
    for marking in MARKINGS:
        for supersession, refresh, root_index in ((R1, False, 0), (R2, True, third)):
            model = _model(720 + k, supersession_scope=supersession)
            evaluated = _evaluated(states, marking, np.random.default_rng([k, 1]))
            _t8_case(model, states, evaluated, root_index, refresh, supersession, (variant, marking, supersession, refresh))


@pytest.mark.slow
@pytest.mark.parametrize("variant", THREE_VARIANTS)
def test_the_deferral_on_real_actions_in_bf16(variant):
    """T8's two arms bit for bit in BF16 (the reference comparison is FP32's)."""
    states = _states(variant, 0)
    evaluated = _evaluated(states, "turn-starts", np.random.default_rng(0))
    model = _model(740, "bf16")
    deferred, eager = CachedEvaluator(model), CachedEvaluator(model)
    d_epoch, s_epoch = deferred.root_epoch(states[0]), eager.root_epoch(states[0])
    for i, state in enumerate(states):
        if evaluated[i]:
            assert _identical(deferred.evaluate(state, d_epoch), eager.evaluate(state, s_epoch)), (variant, i)
        else:
            _eager_at(eager, state, s_epoch)


# --- T9: the step API's tree form ----------------------------------------------------------------------


class _Leaves:
    """The search's evaluator: ``inner`` one position at a time (the cached path's own way), each leaf
    checked against its reference (``reference(state, root)``: an ``Evaluation``) and recorded - its
    identity (node key, moves made in its action), its evaluation, the boards its evaluation computed
    as supersessions, and whether it offered submit."""

    def __init__(self, inner: CachedNetworkEvaluator, reference) -> None:
        self.inner = inner
        self.policy = inner.policy
        self.reference = reference
        self.driver = None
        self.leaves: list[tuple[tuple[int, int], object, int, bool]] = []
        self.root = None
        self.start = self._start

    def _start(self, root):
        self.root = state_of(root)
        return self.inner.start(root)

    def __call__(self, positions):
        out = []
        store = self.inner.cached.store
        for v, p in zip(self.driver.batch, positions, strict=True):
            before = store.computed["supersession"]
            (ev,) = self.inner([p])
            gap = _evals_close(ev, self.reference(state_of(p), self.root), (v.key, v.expansion.moves_made))
            GAPS["step api, fp32 logits"] = max(GAPS["step api, fp32 logits"], gap)
            self.leaves.append(((v.key, v.expansion.moves_made), ev, store.computed["supersession"] - before, v.submit_possible))
            out.append(ev)
        return out


def _reference_for(model, mode: str):
    dense = NetworkEvaluator(model, scope="block_causal")

    def ref(state, root):
        if mode == "causal":
            return dense([state])[0]
        return _split(reference_output(model, state, anchor=root), [state])[0]

    return ref


def _tree_search(root, model, mode: str, flags: dict, playouts: int):
    """A tree-form search from ``root`` through the C step API, the evaluation cache off, every leaf
    checked against its reference: the recorder and the summary."""
    from t5d.engine.evaluator import CStepDriver, serve, start_epoch
    from t5d.engine.search import Search, search_params

    n = root.info().boards
    if mode == "causal":
        config = CacheConfig()
    elif mode == "refresh":
        config = CacheConfig(root_refresh=True)
    else:  # C26's phase form, N* at the root's boards: the cached half
        config = CacheConfig(root_refresh=True, phase_form=True, phase_boards=n)
    rec = _Leaves(CachedNetworkEvaluator(CachedEvaluator(model, config)), _reference_for(model, mode))
    with Search(root, params=search_params(seed=5, eval_cache_entries=0, batch_size=4, **QUIET, **flags)) as s:
        start_epoch(s, rec)
        rec.driver = CStepDriver(s, playouts=playouts)
        serve(rec.driver, rec, batch_size=4)
        return rec, s.info()


@pytest.mark.parametrize("mode", ["causal", "refresh", "phase-form"])
@pytest.mark.parametrize("variant", ["Standard - Two Timelines", "Misc - Timeline Battleground"])
def test_the_tree_form_through_the_step_api(variant, mode):
    """T9 (see the module docstring): under the tree form, natural gates and none, every leaf the
    evaluator is asked about is the reference; with no gates a turn start's evaluation computes boards
    the turn's inner moves superseded; and against the ply split's leaves (every gate firing), the
    leaves both searches meet - turn starts among them - are bit for bit alike."""
    model = _model(750, supersession_scope=R2)
    with _search_root(variant) as root:
        always, _ = _tree_search(root, model, mode, dict(split=T5D_SPLIT_TREE, tree_gates=T5D_TREE_GATE_ALWAYS), playouts=160)
        seen = {key: ev for key, ev, _, _ in always.leaves}
        root_key = always.leaves[0][0]
        shared_starts = 0
        for gates in (0, None):
            flags = dict(split=T5D_SPLIT_TREE) | ({} if gates is None else {"tree_gates": gates})
            rec, info = _tree_search(root, model, mode, flags, playouts=40)
            assert info.inherited_nodes > 0 and rec.leaves, (variant, mode, gates)
            if gates == 0:
                assert all(key[1] == 0 for key, _, _, _ in rec.leaves), "no gates: turn starts only"
                assert any(computed > 0 for _, _, computed, _ in rec.leaves[1:]), (variant, mode, "a turn start computed superseded boards")
            for key, ev, _, _ in rec.leaves:
                if key in seen:
                    assert _same(ev, seen[key]), (variant, mode, gates, key)
                    shared_starts += key[1] == 0 and key != root_key
        assert shared_starts > 0, (variant, mode, "a non-root turn start met by both searches")


# --- T10: the flat form -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["causal", "refresh"])
def test_the_flat_form_evaluates_turn_starts(mode):
    """T10: under the flat form every leaf is a turn start (no move made in its action, no submit
    offered) and the reference; nothing is inherited."""
    model = _model(760, supersession_scope=R2)
    with _search_root("Standard - Two Timelines") as root:
        rec, info = _tree_search(root, model, mode, dict(split=T5D_SPLIT_FLAT), playouts=24)
    assert len(rec.leaves) > 1 and info.inherited_nodes == 0
    assert all(key[1] == 0 and not submit for key, _, _, submit in rec.leaves)
