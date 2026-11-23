"""6e's steering switches from Python (H13, H43): the summary's counter for Hunt's lines, netplay's
per-side settings, the switch off being the game without the feature, and the switch match's
pairing and tally (``t5d.engine.switch_match``). The engine-level tests are
``engine/tests/test_steering_switches.cpp``."""

from __future__ import annotations

import json

import pytest

from t5d.engine import Game, UniformEvaluator
from t5d.engine.capi import T5D_EVALUATOR_UNIFORM, layout_mismatches, load_library
from t5d.engine.netplay import RESERVED_OVERRIDES, PlayConfig, _decision_params, check_replay, play_game, replay_with_gate, steering_header
from t5d.engine.search import Search, search_params
from t5d.engine.switch_match import SWITCHES, main, play_switch_match
from t5d.net.config import NetConfig
from tests.engine.helpers import requires_engine
from tests.engine.test_netplay import GATE, gate_required, requires_gate  # noqa: F401 - gate_required: the fixture requires_gate names

pytestmark = requires_engine

#: White's superseded board holds White's queen against a bare king: the asset map flags it
#: (python/tests/train/test_steering.py's start, engine/tests/test_assets.cpp).
ASSET_START = ("[k4/5/5/5/Q3K:0:1:w]", "[k4/5/5/5/4K:0:1:b]", "[k4/5/5/5/3K1:0:2:w]")

#: Hunt shallow and cheap on the asset start, so that it stops at its budget with open root
#: children and suggests (engine/tests/test_steering_switches.cpp's hunt_lines).
HUNT_LINES = (("synchronous", 1), ("hunt_depth", 3), ("sync_job_nodes", 3000))


def _match(**kw) -> PlayConfig:
    base = dict(start_fen=ASSET_START, mode="match", playouts=16, batch_size=4, cap_turns=3, cap_boards=30, seed=3)
    base.update(kw)
    return PlayConfig(**base)


def test_the_binding_reports_hunts_suggestion_edges():
    assert layout_mismatches(load_library()) == []
    quiet = dict(synchronous=1, asset_map=0, settling=0, hunt_depth=3, sync_job_nodes=3000)
    with Game.from_fen(list(ASSET_START)) as g:
        with Search(g, params=search_params(hunt_suggest=1, **quiet)) as on:
            on.run(T5D_EVALUATOR_UNIFORM, 200)
            info = on.info()
            assert info.suggestion_edges > 0
            assert any(c.boost > 0 for c in on.children())
        with Search(g, params=search_params(hunt_suggest=0, **quiet)) as off, Search(g, params=search_params(hunt=0, **quiet)) as none:
            off.run(T5D_EVALUATOR_UNIFORM, 200)
            none.run(T5D_EVALUATOR_UNIFORM, 200)
            assert off.info().suggestion_edges == 0
            assert all(c.boost == 0 for c in off.children())
            assert off.digest() == none.digest()


def test_the_play_config_switches_default_to_the_built_search():
    cfg = PlayConfig()
    assert cfg.asset_map is True and cfg.hunt_suggest is False
    assert cfg.search_overrides == () and cfg.side_overrides == ((), ())
    assert steering_header(cfg) is None
    with Game.variant("Standard") as g:
        relations = 0
        now = _decision_params(g, cfg, relations, seed=5, noise=False)
        before = search_params(  # the call as it stood before the switches were plumbed
            g.lib,
            split=0,
            seed=5,
            root_noise=0,
            batch_size=cfg.batch_size,
            turn_cap_turns=cfg.cap_turns,
            turn_cap_boards=cfg.cap_boards,
            expand=(8, "all", relations),
            hunt=1,
            settling=1,
            asset_map=1,
            hunt_ms=cfg.hunt_ms,
            settling_ms=cfg.settling_ms,
            verification_threads=cfg.verification_threads,
        )
        assert bytes(now) == bytes(before)
        assert now.hunt_suggest == 0 and now.asset_map == 1
        # Match play: each side's own overrides, after the common ones.
        sided = _match(search_overrides=(("hunt_depth", 3),), side_overrides=((("asset_map", 0),), (("hunt_suggest", 1),)))
        white = _decision_params(g, sided, relations, seed=5, noise=False, side=0)
        black = _decision_params(g, sided, relations, seed=5, noise=False, side=1)
        assert (white.asset_map, white.hunt_suggest, white.hunt_depth) == (0, 0, 3)
        assert (black.asset_map, black.hunt_suggest, black.hunt_depth) == (1, 1, 3)


def test_overrides_are_refused_where_they_do_not_belong():
    with pytest.raises(ValueError, match="match play only"):
        PlayConfig(mode="self", side_overrides=((("asset_map", 0),), ())).validate()
    for name in ("split", "seed", "proof_channel"):
        assert name in RESERVED_OVERRIDES
        with pytest.raises(ValueError, match="RESERVED_OVERRIDES"):
            PlayConfig(mode="match", search_overrides=((name, 1),)).validate()
        with pytest.raises(ValueError, match="RESERVED_OVERRIDES"):
            PlayConfig(mode="match", side_overrides=(((name, 1),), ())).validate()
    with pytest.raises(ValueError, match="a pair"):
        PlayConfig(mode="match", side_overrides=((),)).validate()
    with pytest.raises(ValueError, match="pairs"):
        PlayConfig(mode="match", search_overrides=(("hunt_depth",),)).validate()
    with pytest.raises(ValueError, match="no field"):
        play_game(UniformEvaluator(), _match(search_overrides=(("no_such_field", 1),)), net_cfg=NetConfig())


def test_each_side_searches_with_its_own_setting():
    common = dict(hunt=False, settling=False, search_overrides=(("synchronous", 1),))
    white_on = play_game(
        UniformEvaluator(), _match(side_overrides=((("asset_map", 1),), (("asset_map", 0),)), **common), net_cfg=NetConfig()
    )
    assert white_on.asset_boosted_edges[0] > 0 and white_on.asset_boosted_edges[1] == 0
    assert "[Steering " in white_on.record and "white: asset_map=1" in white_on.record and "black: asset_map=0" in white_on.record
    black_on = play_game(
        UniformEvaluator(), _match(side_overrides=((("asset_map", 0),), (("asset_map", 1),)), **common), net_cfg=NetConfig()
    )
    assert black_on.asset_boosted_edges[0] == 0


def test_a_switch_off_on_both_sides_is_the_game_without_the_feature():
    common = dict(hunt=False, settling=False, search_overrides=(("synchronous", 1),), decision_log=True)
    a = play_game(UniformEvaluator(), _match(asset_map=False, **common), net_cfg=NetConfig())
    b = play_game(
        UniformEvaluator(), _match(asset_map=True, side_overrides=((("asset_map", 0),), (("asset_map", 0),)), **common), net_cfg=NetConfig()
    )
    c = play_game(UniformEvaluator(), _match(asset_map=False, **{**common, "hunt": True, "settling": True}), net_cfg=NetConfig())
    for other in (b, c):
        assert other.actions == a.actions
        assert other.final_fingerprint == a.final_fingerprint
        assert (other.decisions, other.playouts) == (a.decisions, a.playouts)
        for x, y in zip(other.decision_log, a.decision_log, strict=True):
            assert [(ch["visits"], ch["q"]) for ch in x["children"]] == [(ch["visits"], ch["q"]) for ch in y["children"]]
        assert other.asset_boosted_edges == [0, 0]
    assert a.asset_boosted_edges == [0, 0]
    assert c.proofs_injected == 0


def test_hunts_lines_on_one_side_only():
    cfg = _match(
        asset_map=False, settling=False, search_overrides=HUNT_LINES, side_overrides=((("hunt_suggest", 1),), (("hunt_suggest", 0),))
    )
    res = play_game(UniformEvaluator(), cfg, net_cfg=NetConfig())
    assert res.suggestion_edges[0] > 0 and res.suggestion_edges[1] == 0
    assert res.proofs_injected == 0
    assert res.turn_start_decisions[0] >= 1
    assert res.carried_turn_starts[0] <= res.turn_start_decisions[0]


@requires_gate
def test_the_switch_match_pairs_colours_and_tallies():
    base = _match(hunt=False, settling=False, search_overrides=(("synchronous", 1),))
    m = play_switch_match(UniformEvaluator(), base, "asset_map", starts=[ASSET_START], pairs=1, net_cfg=NetConfig())
    assert len(m.games) == 2 and m.on_side == [0, 1]
    for res, on in zip(m.games, m.on_side, strict=True):
        sides = res.config.side_overrides
        assert sides[on] == (("asset_map", 1),) and sides[1 - on] == (("asset_map", 0),)
        assert "[Steering " in res.record
        assert check_replay(res, replay_with_gate(res.record, gate=GATE)) == []
    assert m.wins + m.draws + m.capped + m.losses + m.unscored == 2
    assert m.edges_off == 0 and m.edges_on > 0
    assert set(m.summary()) >= {"wins", "draws", "capped", "losses", "unscored", "score", "edges_on", "edges_off"}
    with pytest.raises(ValueError, match="switch"):
        play_switch_match(UniformEvaluator(), base, "guard", starts=[ASSET_START])
    with pytest.raises(ValueError, match="side overrides"):
        play_switch_match(UniformEvaluator(), _match(side_overrides=((("asset_map", 1),), ())), "asset_map")
    with pytest.raises(ValueError, match="search_overrides"):
        play_switch_match(UniformEvaluator(), _match(search_overrides=(("asset_map", 0),)), "asset_map")
    assert set(SWITCHES) == {"asset_map", "hunt_suggest"}


def test_the_switch_match_command_line(capsys):
    argv = ["--switch", "asset_map", "--variant", "Very Small - Open", "--pairs", "1", "--playouts", "4", "--cap-turns", "2"]
    assert main([*argv, "--no-hunt", "--no-settling", "--no-replay", "--threads", "1"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    games, last = lines[:-1], lines[-1]
    assert len(games) == 2 and [g["on_side"] for g in games] == [0, 1]
    assert last["totals"]["games"] == 2
    assert "random" in last["conditions"]["weights"]
