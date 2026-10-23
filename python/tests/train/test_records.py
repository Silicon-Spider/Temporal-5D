"""Register I15's training records (spec section 12.8): written by the generator, read back bit
for bit, their network input rebuilt from the record alone, their raw root outputs the root's
evaluation the search used, their conditions complete. Tiny games with random weights - nothing
is trained."""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from t5d.engine.capi import MOVE_DTYPE, T5D_LEGALITY_LEGAL, T5D_NO_LIMIT, CSearchParams
from t5d.engine.evaluator import NetworkEvaluator
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL
from t5d.engine.play import random_position_game
from t5d.engine.search import search_params
from t5d.net import NetConfig, build_model
from t5d.net.cache import state_digest
from t5d.net.state import PositionState
from t5d.train.records import (
    DECISION_FIELDS,
    RECORD_VERSION,
    GameRecord,
    RecordVersionError,
    from_arrays,
    network_id,
    params_dict,
    params_from_dict,
    read_game,
    record_differences,
    to_arrays,
    write_game,
)
from t5d.train.selfplay import (
    GeneratorConfig,
    generate_game,
    is_full_search,
    move_temperature,
    replay_positions,
    root_temperature,
    turn_moves,
)
from t5d.train.targets import TargetConfig, build_targets, flat_boards, turn_option
from tests.engine.helpers import requires_engine
from tests.train.test_targets import _kata_go

NET = NetConfig(**SMALL)


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(0)
    return NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8)


def _travel_start(seed: int) -> tuple[str, ...]:
    """A start with timelines branched by travel: seeded random play (travel preferred)."""
    with random_position_game("Standard", seed, boards=10, travel_prob=1.0) as g:
        return tuple(fen for _, _, fen in g.snapshot().boards)


#: Section 15.2's configurations: D2's three generator arms (the flat form's from Two Timelines' two
#: playable boards), two variants, a custom start with
#: branched timelines, a tight legality budget (unknown moves among the candidates) with another
#: history window than the default's, the root temperature and noise off.
CONFIGS = {
    "ply": GeneratorConfig(playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=3),
    "tree": GeneratorConfig(split="tree", playouts=8, batch_size=4, cap_turns=2, cap_boards=20, seed=4),
    "two_timelines": GeneratorConfig(variant="Standard - Two Timelines", playouts=6, batch_size=3, cap_turns=2, cap_boards=20, seed=5),
    "flat": GeneratorConfig(
        variant="Standard - Two Timelines", split="flat", playouts=10, batch_size=4, cap_turns=3, cap_boards=24, seed=9
    ),
    "budget": GeneratorConfig(
        playouts=8,
        batch_size=4,
        cap_turns=3,
        cap_boards=20,
        seed=6,
        window_k=2,
        root_noise=False,
        search_overrides=(("legality_checks", 0), ("legality_nodes", 0)),
    ),
}


def same_input(a: PositionState, b: PositionState) -> bool:
    """Two network inputs equal in every field (arrays element for element)."""
    for f in fields(PositionState):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                return False
        elif x != y:
            return False
    return True


def _generate(evaluator, cfg: GeneratorConfig, keep_positions: bool = True):
    with torch.no_grad():
        return generate_game(evaluator, cfg, net_cfg=NET, keep_positions=keep_positions)


@requires_engine
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_a_record_reads_back_bit_for_bit(evaluator, name, tmp_path):
    g = _generate(evaluator, CONFIGS[name])
    rec = g.record
    assert rec.decisions, "the game made decisions"
    path = write_game(tmp_path / f"{name}.npz", rec)
    back = read_game(path)
    assert record_differences(rec, back) == []
    assert back.header["record_version"] == RECORD_VERSION
    # The arrays' layout: every decision field present, offsets consistent.
    arrays = to_arrays(rec)
    for name_ in DECISION_FIELDS:
        assert arrays[f"d_{name_}"].shape == (len(rec.decisions),)
    assert arrays["c_offset"][-1] == sum(d.move_count for d in rec.decisions)
    assert arrays["e_offset"][-1] == sum(len(d.children) for d in rec.decisions)


@requires_engine
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_record_rebuilds_its_network_input(evaluator, name, tmp_path):
    """I15: "later architectures retrained on the archive" - each decision's input rebuilt from the
    stored record (the start, the played children, the recorded expand options and legality
    budget) has the record's candidate moves and the generation-time input - digest for digest,
    field for field, and with the recorded window's W/E scope (the "budget" game's window is 2,
    not the default 8)."""
    g = _generate(evaluator, CONFIGS[name])
    back = read_game(write_game(tmp_path / "g.npz", g.record))
    rebuilt = replay_positions(back)
    assert len(rebuilt) == len(g.positions) == len(back.decisions)
    for live, again, d in zip(g.positions, rebuilt, back.decisions, strict=True):
        assert state_digest(live.state) == state_digest(again.state)
        assert same_input(live.state, again.state)
        assert live.scope.k == again.scope.k == back.header["expand"]["window_k"]
        for name in ("window", "e_boards", "src_extra", "window_depth"):
            assert np.array_equal(getattr(live.scope, name), getattr(again.scope, name)), name
        assert again.moves.tobytes() == d.candidates.tobytes()
        keys = np.stack([again.state.timeline, again.state.half_turn], axis=1)
        assert np.array_equal(keys, d.boards)


@requires_engine
def test_a_custom_start_with_travel(evaluator, tmp_path):
    """A custom 5DFEN start with branched timelines (seeded random play preferring travel) plays,
    records and rebuilds like a variant's."""
    for seed in (1, 2, 3):
        try:
            fens = _travel_start(seed)
        except ValueError:
            continue
        cfg = GeneratorConfig(start_fen=fens, playouts=6, batch_size=3, cap_turns=0, cap_boards=16, max_actions=3, seed=seed)
        g = _generate(evaluator, cfg)
        back = read_game(write_game(tmp_path / f"c{seed}.npz", g.record))
        assert record_differences(g.record, back) == []
        assert back.header["variant"] == "custom" and back.header["start_fen"] == list(fens)
        if back.decisions:
            assert len(replay_positions(back)) == len(back.decisions)
        return
    pytest.skip("no seeded travel start reached 10 boards")


@requires_engine
def test_the_raw_root_outputs_are_the_searchs_root_evaluation(evaluator):
    """The raw outputs are the network's evaluation of the decision's input: its WDL is the
    engine's normalised root WDL (the evaluation the search used, often a cache hit), and its
    logits give the search's noise-free prior - a softmax at the root's temperature over the
    candidates and a legal submit (``set_priors``) - wherever every child is known legal."""
    g = _generate(evaluator, CONFIGS["ply"])
    compared = 0
    for d in g.record.decisions:
        assert d.root_evaluated
        p = np.exp(d.raw_wdl_logits.astype(np.float64) - d.raw_wdl_logits.max())
        assert np.allclose(p / p.sum(), d.root_wdl, atol=1e-5)
        c = d.children
        legal = bool((c.legality == T5D_LEGALITY_LEGAL).all())
        offered = d.submit_possible and d.submit_legal
        if not legal or len(c) != d.move_count + int(offered) or (offered and math.isnan(d.raw_submit_logit)):
            continue
        z = np.append(d.raw_logits.astype(np.float64), d.raw_submit_logit if offered else -np.inf) / d.root_policy_temperature
        prior = np.exp(z - z.max())
        prior /= prior.sum()
        expected = np.where(c.candidate >= 0, prior[np.maximum(c.candidate, 0)], prior[-1])
        assert np.allclose(c.prior_noise_free, expected, atol=1e-5)
        compared += 1
    assert compared > 0


@requires_engine
def test_the_records_conditions(evaluator, tmp_path):
    """The conditions I15 names: ruleset, board count, network, search configuration (every
    t5d_search_params field, the legality budget among them), temperature, the PCR flag; no
    proof (the channel is off - H23, H41); the visits' forced and steered counts (0 today)."""
    cfg = CONFIGS["budget"]
    g = _generate(evaluator, cfg, keep_positions=False)
    h = read_game(write_game(tmp_path / "g.npz", g.record)).header
    for key in ("ruleset", "rules", "engine_version", "network", "search", "expand", "generator_config", "pgn", "result", "value_source"):
        assert key in h, key
    assert h["network"]["id"] == network_id(evaluator.model) and h["network"]["config"]["d_model"] == NET.d_model
    full = params_from_dict(h["search"]["full"])
    assert (full.legality_checks, full.legality_nodes, full.root_noise) == (0, 0, 0)
    assert h["proof_refs"] == [] and all(d.proof_ref == -1 for d in g.record.decisions)
    assert h["value_source"] == "wdl"
    for d in g.record.decisions:
        assert d.n_boards == d.boards.shape[0] and d.full_search == is_full_search(cfg, d.decision)
        assert d.playout_cap == (cfg.playouts if d.full_search else cfg.pcr.cheap_cap(cfg.playouts))
        assert (d.children.forced_visits == 0).all() and (d.children.steered_visits == 0).all()
        assert d.move_temperature == move_temperature(cfg, d.action) and d.root_policy_temperature == root_temperature(cfg, d.action)
    # The tight budget leaves moves unknown at some root.
    assert any((d.children.legality != T5D_LEGALITY_LEGAL).any() for d in g.record.decisions)


def _live_kata_go(live, sqrt_parent, cpuct, k=2.0):
    """KataGo's pruned visits from the live search's own children (the closed form of
    ``tests.train.test_targets``), the most visited child kept."""
    n = [c.visits for c in live]
    best = int(np.argmax(n))
    return _kata_go(n, [c.q for c in live], [c.prior for c in live], best, cpuct, sqrt_parent, k, [c.exploration_scale for c in live])


@requires_engine
@pytest.mark.parametrize("e2", [False, True], ids=["e2_off", "e2_on"])
def test_the_record_holds_the_live_search(evaluator, e2):
    """At generation, each decision's record against the live search it was written from: the
    root's cPUCT the search's (cpuct x E2's root scale - 1 with E2 off), the best child the most
    visited, the decision's place in its action the handle's moves made; and the record-derived
    policy target and root value - pruned and unpruned - equal to ones computed from the live
    children and summary, the candidate indices matched move by move."""
    over = (("variance_scaled_cpuct", 1),) if e2 else ()
    cfg = GeneratorConfig(playouts=16, batch_size=4, cap_turns=3, cap_boards=20, seed=8, search_overrides=over)
    plain = TargetConfig(forced_k=0.0, prune_single_visit=False, deblunder=False, short_term_horizons=NET.short_term_horizons)
    pruned_cfg = replace(plain, forced_k=2.0, prune_single_visit=True)
    seen = {"scaled": 0, "mid_action": 0, "not_best": 0, "pruned": 0}

    def check(search, d, position):
        info, live, params = search.info(), search.children(), search.params
        assert d.root_cpuct == info.root_cpuct
        if e2:
            assert params.cpuct_scale_min <= d.root_cpuct / params.cpuct <= params.cpuct_scale_max
            seen["scaled"] += d.root_cpuct != params.cpuct
        else:
            assert d.root_cpuct == params.cpuct  # E2 off: the root scale is 1
        visits = [c.visits for c in live]
        assert visits[d.best_child] == max(visits) and d.best_child == search.choose_by_visits(0.0, 0.5)
        assert d.move_in_action == d.moves_made == int(position.state.moves_made)
        seen["mid_action"] += d.move_in_action > 0
        seen["not_best"] += d.played_child >= 0 and d.played_child != d.best_child
        assert d.root_visits == info.root_visits and d.root_q == info.root_q
        # The targets from the record against the live children (candidates matched by their fields).
        moves = [tuple(int(x) for x in row.tolist()) for row in position.moves]
        slot = [d.move_count if c.submit else moves.index(tuple(int(getattr(c.moves[0], f)) for f in MOVE_DTYPE.names)) for c in live]
        rec = GameRecord(header={"result": {"end": "unknown", "winner": None, "actions": 1}, "search": {}}, decisions=[d])
        live_pruned = _live_kata_go(live, math.sqrt(info.root_visits), info.root_cpuct)
        for tc, kept in ((plain, [float(v) for v in visits]), (pruned_cfg, live_pruned)):
            t = build_targets(rec, tc)
            total = sum(kept)
            seen["pruned"] += tc is pruned_cfg and total < sum(visits)
            if not d.full_search or total <= 0:
                continue
            want = np.zeros(d.move_count + 1)
            for s, v in zip(slot, kept, strict=True):
                want[s] += v / total
            assert np.allclose(t.policy[0], want, rtol=0, atol=1e-12)
            q = sum(v * c.q for v, c in zip(kept, live, strict=True)) / total
            assert t.search_value[0] == pytest.approx(q, abs=1e-12)

    with torch.no_grad():
        generate_game(evaluator, cfg, net_cfg=NET, on_decision=check)
    assert seen["mid_action"] and seen["not_best"] and seen["pruned"], seen
    assert seen["scaled"] or not e2


def test_forced_and_steered_counts_round_trip(tmp_path):
    """I2's flags are 0 in every generated game until forced playouts and E10's steering exist,
    so a synthetic record carries distinct non-zero counts through the file."""
    from t5d.net import synthetic_states
    from t5d.train.trainer import synthetic_record

    rec = synthetic_record(9, synthetic_states(9, [2, 3, 2]))
    for i, d in enumerate(rec.decisions):
        n = len(d.children)
        d.children.forced_visits = np.arange(1, n + 1, dtype="<i8") * (i + 1)
        d.children.steered_visits = np.arange(n, 0, -1, dtype="<i8") * 10 + i
    back = read_game(write_game(tmp_path / "s.npz", rec))
    assert record_differences(rec, back) == []
    for d, e in zip(rec.decisions, back.decisions, strict=True):
        assert e.children.forced_visits.tolist() == d.children.forced_visits.tolist() != e.children.steered_visits.tolist()
        assert e.children.steered_visits.tolist() == d.children.steered_visits.tolist()


@requires_engine
def test_replay_refuses_another_ruleset(evaluator):
    g = _generate(evaluator, CONFIGS["ply"], keep_positions=False)
    assert len(replay_positions(g.record)) == len(g.record.decisions)
    for key in ("ruleset", "rules"):
        bad = GameRecord(header={**g.record.header, key: int(g.record.header[key]) + 1}, decisions=g.record.decisions)
        with pytest.raises(ValueError, match=key):
            replay_positions(bad)


def test_search_params_round_trip():
    p = search_params(legality_checks=7, legality_nodes=T5D_NO_LIMIT, cpuct=1.75, expand=(6, "all", 3))
    d = params_dict(p)
    assert json.loads(json.dumps(d)) == d
    q = params_from_dict(d)
    assert bytes(q) == bytes(p) and isinstance(q, CSearchParams)


@pytest.mark.parametrize("version", [RECORD_VERSION + 1, RECORD_VERSION - 1, None], ids=["newer", "older", "none"])
def test_another_version_is_refused(tmp_path, version):
    """Another version is refused both ways - a newer one, an older one, and a header naming none."""
    header = {"result": {"end": "turn_cap", "winner": None, "actions": 0}}
    if version is not None:
        header["record_version"] = version
    arrays = to_arrays(GameRecord(header=header))
    with pytest.raises(RecordVersionError):
        from_arrays(arrays)
    np.savez(tmp_path / "x.npz", **arrays)
    with pytest.raises(RecordVersionError):
        read_game(tmp_path / "x.npz")
    ok = to_arrays(GameRecord(header={**header, "record_version": RECORD_VERSION}))
    assert from_arrays(ok).header["record_version"] == RECORD_VERSION


def test_the_network_id_follows_the_weights():
    a, b = build_model(NET, seed=0), build_model(NET, seed=0)
    assert network_id(a) == network_id(b)
    with torch.no_grad():
        next(b.parameters()).add_(1e-3)
    assert network_id(a) != network_id(b)


@requires_engine
def test_the_generator_refuses_the_retired_arm_and_the_optimistic_policy():
    """The retired whole-turn arm's name is refused; the flat form is not (its policy target exists
    since issue 242 - issue 234(b)'s refusal lifted)."""
    GeneratorConfig(split="flat").validate()
    with pytest.raises(ValueError, match="retired"):
        GeneratorConfig(split="turn").validate()
    cfg = NetConfig(**{**SMALL, "optimistic_policy": True})
    ev = NetworkEvaluator(build_model(cfg, seed=0).eval(), policy="optimistic")
    with pytest.raises(ValueError, match="D9"):
        generate_game(ev, CONFIGS["ply"], net_cfg=cfg)
    with Game.variant("Standard") as g:
        assert g.info().boards == 1


@requires_engine
def test_a_flat_self_play_game_reads_the_networks_leave_logits():
    """D2's flat form in self-play with the network (issues 241, 242): every evaluation the searches
    were delivered carries a finite leave logit for every board - no board falls back on
    ``leave_share`` - and at each decision, against the live search, the recorded raw leave logits
    are the network's at the root, the playable boards the expansion's, every child a whole turn
    whose moves the record keeps, its noise-free prior the softmax of the turns' scores from the
    recorded raw logits (the engine read the leave logits), and the record-derived flat target the
    marginal of the live children's visits; the game then rebuilds from its record and its rows feed
    the flat loss (forward only)."""
    from tests.engine.test_flat_head import Recorder, _scores, _softmax

    ev = Recorder(NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8))
    cfg = GeneratorConfig(variant="Standard - Two Timelines", split="flat", playouts=16, batch_size=4, cap_turns=4, cap_boards=30, seed=21)
    exact = TargetConfig(forced_k=0.0, prune_single_visit=False, deblunder=False, short_term_horizons=NET.short_term_horizons)
    seen = {"decisions": 0, "multi_board": 0, "target_multi": 0}

    def check(search, d, position):
        live = search.children()
        assert d.flat and len(d.children) == len(live) and bool(d.children.turn.all())
        for k, c in enumerate(live):
            assert d.turn(k).tobytes() == turn_moves([c]).tobytes()
        raw = ev.inner([position])[0]
        assert np.array_equal(d.raw_leave_logits, raw.board_logits) and np.isfinite(d.raw_leave_logits).all()
        assert np.array_equal(d.playable, playable_boards_of(position))
        t = d.root_policy_temperature * float(search.params.policy_temperature)
        want = _softmax(_scores(raw, position, d.playable, live, t, arrivals=True, leave_share=None))
        assert np.allclose(d.children.prior_noise_free, want, rtol=0, atol=2e-6)
        seen["decisions"] += 1
        seen["multi_board"] += int(d.playable.sum() >= 2)
        rec = GameRecord(header={"result": {"end": "unknown", "winner": None, "actions": 1}, "search": {}}, decisions=[d])
        tg = build_targets(rec, exact)
        if d.full_search and sum(c.visits for c in live) > 0:
            got = {(int(b), int(k)): p for b, k, p in tg.flat[0]}
            total = sum(c.visits for c in live)
            seen["target_multi"] += int(d.playable.sum() >= 2)
            for fb in flat_boards(d):
                for opt in fb.options:
                    want_p = sum(c.visits for c in live if turn_option(turn_moves([c]), fb, d.candidates) == opt) / total
                    assert got[(fb.board, opt)] == pytest.approx(want_p, abs=1e-12)

    with torch.no_grad():
        g = generate_game(ev, cfg, net_cfg=NET, keep_positions=True, on_decision=check)
    assert seen["decisions"] == len(g.record.decisions) >= 2 and seen["target_multi"] >= 1, seen
    assert ev.seen and all(e.board_logits is not None and np.isfinite(e.board_logits).all() for _, e in ev.seen)
    assert all(e.board_logits.shape == (p.state.n_boards,) for p, e in ev.seen)
    rebuilt = replay_positions(g.record)
    assert [state_digest(p.state) for p in rebuilt] == [state_digest(p.state) for p in g.positions]
    from t5d.train.losses import compute_losses
    from t5d.train.targets import Row, collate_rows

    rows = [Row(p.state, g.targets, i) for i, p in enumerate(g.positions)]
    batch, tgt = collate_rows(rows, NET)
    with torch.no_grad():
        losses = compute_losses(build_model(NET, seed=1).eval()(batch), tgt)
    assert all(torch.isfinite(v) for v in losses.values()) and "flat_policy" in losses


def playable_boards_of(position) -> np.ndarray:
    """The acting player's playable heads from the input itself (L5: the heads with the acting player
    to move) - the expansion's flag at a turn start."""
    s = position.state
    return s.is_head & (s.side_to_move == s.acting_player)


class _ArrivalBoost:
    """The network, with every non-branching jump's move logit raised by ``boost``: turns that take an
    arrival (the jump's option on the board it lands on, under ``flat_jumps`` BothBoards) lead the
    turn generator, so the search visits them."""

    def __init__(self, inner, boost: float = 6.0):
        self.inner, self.boost = inner, boost

    def __call__(self, positions):
        from dataclasses import replace as dc_replace

        from t5d.train.targets import is_arrival

        out = []
        for p, e in zip(positions, self.inner(positions), strict=True):
            up = np.array([self.boost if is_arrival(m) else 0.0 for m in p.moves], dtype=np.float32)
            out.append(dc_replace(e, policy_logits=(e.policy_logits + up).astype(np.float32)))
        return out


@requires_engine
def test_a_visited_arrival_is_marginalised_as_the_live_search_took_it():
    """Issue 242(a): a non-branching jump landing on a playable board is an option of that board too
    (BothBoards). With arrivals favoured, flat searches from Two Timelines' start (whose knights jump
    between the two present boards without branching) visit turns that take one; at each decision the
    record-derived target is the live children's visits marginalised - the arrival's included - under
    the unpruned configuration, and the default's per-board pruning keeps the visited arrival."""
    ev = _ArrivalBoost(NetworkEvaluator(build_model(NET, seed=0).eval(), max_batch=8))
    cfg = GeneratorConfig(
        variant="Standard - Two Timelines", split="flat", playouts=24, batch_size=4, cap_turns=2, cap_boards=30, seed=23, root_noise=False
    )
    exact = TargetConfig(forced_k=0.0, prune_single_visit=False, deblunder=False, short_term_horizons=NET.short_term_horizons)
    seen = {"arrival_visits": 0, "kept": 0}

    def check(search, d, position):
        if not d.full_search:
            return
        live = search.children()
        total = sum(c.visits for c in live)
        rec = GameRecord(header={"result": {"end": "unknown", "winner": None, "actions": 1}, "search": {}}, decisions=[d])
        got = {(int(b), int(k)): p for b, k, p in build_targets(rec, exact).flat[0]}
        dflt = {(int(b), int(k)): p for b, k, p in build_targets(rec, replace(exact, forced_k=2.0, prune_single_visit=True)).flat[0]}
        for fb in flat_boards(d):
            for opt in fb.options:
                takes = [c for c in live if turn_option(turn_moves([c]), fb, d.candidates) == opt]
                assert got[(fb.board, opt)] == pytest.approx(sum(c.visits for c in takes) / total, abs=1e-12)
                if opt in fb.arrivals and sum(c.visits for c in takes) > 0:
                    seen["arrival_visits"] += sum(c.visits for c in takes)
                    seen["kept"] += int(dflt[(fb.board, opt)] > 0)

    with torch.no_grad():
        generate_game(ev, cfg, net_cfg=NET, keep_positions=False, on_decision=check)
    assert seen["arrival_visits"] > 0 and seen["kept"] > 0, seen
