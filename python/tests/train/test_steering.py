"""Stage E10's steering, end to end (register I2, I15; spec 12.2): a self-play game whose roots
the asset map steers records non-zero steered visits per root child, and the policy targets
exclude them (or down-weight them, the provenance arm) - with the replica's prior the unsteered
one. Also the binding's view of the verification ladder: on by default in 6b''s ranges, its
proofs withheld while the proof channel is off (H41) - which Python cannot turn on.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from t5d.engine.capi import T5D_EVALUATOR_UNIFORM, load_library
from t5d.engine.evaluator import UniformEvaluator
from t5d.engine.game import Game
from t5d.engine.netplay import SMALL
from t5d.engine.search import Search, search_params
from t5d.net import NetConfig
from t5d.train.records import read_game, write_game
from t5d.train.selfplay import GeneratorConfig, PCRConfig, generate_game
from t5d.train.targets import TargetConfig, policy_target, pruned_visits
from tests.engine.helpers import requires_engine

NET = NetConfig(**SMALL)

#: White's superseded board (turn 1) holds White's queen against a bare king: the asset map flags
#: it, and the White king's travel move onto it is boosted (engine/tests/test_assets.cpp).
ASSET_START = ("[k4/5/5/5/Q3K:0:1:w]", "[k4/5/5/5/4K:0:1:b]", "[k4/5/5/5/3K1:0:2:w]")

#: White mates in one (engine/tests/test_v3_support.hpp).
MATE_IN_ONE = ("[k4/4Q/1K3/5/5:0:1:w]",)

#: Bare kings: no mate for either side (the settling scan disproves it in its class).
BARE_KINGS = ("[k4/5/5/5/4K:0:1:w]",)


@requires_engine
def test_the_ladder_is_on_by_default_and_its_proofs_withheld():
    lib = load_library()
    p = search_params(lib)
    assert (p.hunt, p.hunt_max_boards, p.settling, p.settling_max_boards, p.asset_map) == (1, 100, 1, 50, 1)
    assert (p.hunt_ms, p.settling_ms, p.verification_threads, p.hunt_suggest, p.proof_channel) == (50.0, 50.0, 5, 0, 0)
    with pytest.raises(ValueError, match="H41"):
        search_params(lib, proof_channel=1)
    assert (p.boost_cap, p.hunt_with_settling) == (0.5, 0)
    game = Game.from_fen(list(MATE_IN_ONE))
    try:
        # Under 50 boards the root's settling scan asks our mate in Hunt's place (hunt_with_settling
        # 0); settling_max_scans 0 leaves the root's question alone (the E10 review's C01: proven and
        # disproven reach their own fields).
        with Search(game, params=search_params(lib, synchronous=1, asset_map=0, settling_max_scans=0)) as s:
            s.run(T5D_EVALUATOR_UNIFORM, 100)
            info = s.info()
            assert (info.hunt_runs, info.settling_runs) == (0, 1)
            assert (info.settling_proven, info.settling_disproven) == (1, 0)  # the scan proves the mate ...
            assert info.proofs_withheld == 1 and info.proofs_injected == 0  # ... which does not enter the tree
            assert not info.root_solved and (info.root_lo, info.root_hi) == (-1, 1)
            assert all((c.lo, c.hi) == (-1, 1) for c in s.children())
        with Search(game, params=search_params(lib, synchronous=1, asset_map=0, hunt_with_settling=1)) as s:
            s.run(T5D_EVALUATOR_UNIFORM, 100)
            info = s.info()
            assert info.hunt_runs == 1 and info.hunt_mates == 1  # Hunt proves the mate ...
            assert info.settling_proven == 1  # ... and so does the root's settling scan
            assert info.proofs_withheld == 2 and info.proofs_injected == 0  # ... and neither enters the tree
            assert not info.root_solved
        with Search(game, params=search_params(lib, synchronous=1, asset_map=0, hunt=0, settling=0)) as s:
            s.run(T5D_EVALUATOR_UNIFORM, 100)
            info = s.info()
            assert (info.hunt_runs, info.settling_runs, info.proofs_withheld) == (0, 0, 0)
    finally:
        game.close()
    bare = Game.from_fen(list(BARE_KINGS))
    try:
        params = search_params(lib, synchronous=1, asset_map=0, settling_max_scans=0, settling_depth=3, hunt_depth=3)
        with Search(bare, params=params) as s:
            s.run(T5D_EVALUATOR_UNIFORM, 8)
            info = s.info()
            assert (info.settling_runs, info.settling_proven, info.settling_disproven) == (1, 0, 1)
    finally:
        bare.close()


@requires_engine
def test_a_steered_self_play_game_records_steered_visits_and_the_targets_exclude_them(tmp_path):
    cfg = GeneratorConfig(
        start_fen=ASSET_START,
        playouts=96,
        batch_size=4,
        cap_turns=3,
        cap_boards=12,
        max_actions=2,
        seed=5,
        window_k=4,
        pcr=PCRConfig(enabled=False),  # every search full: every decision a policy target
        asset_map=True,  # self-play's default is off since the E10 review (issue 179): on, to test the plumbing
        search_overrides=(("synchronous", 1), ("hunt", 0), ("settling", 0)),
    )
    live: list[tuple[list, object]] = []

    def on_decision(search, record, position):
        live.append((search.children(), record))

    g = generate_game(UniformEvaluator(), cfg, net_cfg=NET, on_decision=on_decision)
    ds = g.record.decisions
    assert ds
    steered_total = 0
    for children, d in live:
        c = d.children
        # The record carries the live search's steering, child for child.
        assert c.steered_visits.tolist() == [k.steered_visits for k in children]
        assert np.allclose(c.boost, [k.boost for k in children])
        assert np.allclose(c.prior_noise_free - c.boost, [k.prior_unsteered for k in children])
        assert (c.steered_visits <= c.visits).all()
        steered_total += int(c.steered_visits.sum())
    assert steered_total > 0, "the asset map steered no root visit"
    first = ds[0]
    assert (first.children.boost > 0).any()  # the king's travel move onto the flagged board
    exclusion = TargetConfig(policy_mode="exclusion", forced_k=0.0, prune_single_visit=False)
    full = replace(exclusion, policy_mode="provenance", provenance_weight=1.0)
    for d in ds:
        steered = d.children.steered_visits.astype(np.float64)
        visits = d.children.visits.astype(np.float64)
        out = pruned_visits(d, exclusion)
        # Excluded: no target counts a steered visit.
        assert (out <= visits - steered + 1e-9).all()
        np.testing.assert_allclose(out, visits - steered)
        np.testing.assert_allclose(pruned_visits(d, full), visits)
        if steered.sum() > 0 and (visits - steered).sum() > 0:
            assert not np.allclose(policy_target(d, out), policy_target(d, visits))
    # The records written and read back carry the same steering.
    path = tmp_path / "steered.t5drec"
    write_game(path, g.record)
    back = read_game(path)
    for a, b in zip(ds, back.decisions, strict=True):
        assert a.children.steered_visits.tolist() == b.children.steered_visits.tolist()
        assert np.array_equal(a.children.boost, b.children.boost)


@requires_engine
def test_self_play_runs_without_the_asset_map_by_default():
    """The E10 review (issues 179, 189): I2's steered-visit definition is unvalidated, so the
    generator's default turns the asset map off - on a position the map would steer, the records
    carry no boost and no steered visit; the engine's own default keeps the map on (match play)."""
    assert GeneratorConfig().asset_map is False
    assert search_params(load_library()).asset_map == 1
    cfg = GeneratorConfig(
        start_fen=ASSET_START,
        playouts=48,
        batch_size=4,
        cap_turns=3,
        cap_boards=12,
        max_actions=2,
        seed=5,
        window_k=4,
        pcr=PCRConfig(enabled=False),
        search_overrides=(("synchronous", 1), ("hunt", 0), ("settling", 0)),
    )
    g = generate_game(UniformEvaluator(), cfg, net_cfg=NET)
    assert g.record.decisions
    assert g.record.header["search"]["full"]["asset_map"] == 0
    for d in g.record.decisions:
        assert (d.children.steered_visits == 0).all()
        assert (d.children.boost == 0.0).all()
