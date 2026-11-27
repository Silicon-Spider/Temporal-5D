"""F1's CPU evaluation rate (``t5d.engine.evalrate``): the provers off unless asked, the totals as
rates of the sums, the network's calls filed by board-count range, and one short game of the F1
network end to end - its configuration printed in full before the game, its counts consistent. A
smoke run, never a timing: the machine running the tests is shared, and a test is not a
measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from t5d.engine import evalrate
from t5d.engine.evalrate import GameFigures, RangedEvaluator, game_seeds, play_config, range_of, totals
from t5d.engine.netplay import GameResult, PlayConfig, network_config
from t5d.net import build_model
from t5d.net.config import NetConfig
from t5d.net.synthetic import synthetic_state
from tests.engine.helpers import requires_engine


def test_the_provers_are_off_unless_asked():
    off = play_config("Standard", "ply", 1)
    assert (off.mode, off.hunt, off.settling, off.asset_map) == ("self", False, False, False)
    on = play_config("Standard", "ply", 1, provers=True)
    assert (on.hunt, on.settling, on.asset_map) == (True, True, True)
    assert replace(on, hunt=False, settling=False, asset_map=False) == off
    defaults = PlayConfig()  # with --provers, their own defaults
    assert (on.hunt_ms, on.settling_ms, on.verification_threads) == (defaults.hunt_ms, defaults.settling_ms, defaults.verification_threads)
    off.validate()
    on.validate()


def test_the_seeds_follow_netplay():
    assert game_seeds(1, 3) == [1, 102, 203]
    assert game_seeds(7, 1) == [7]


def test_the_totals_are_rates_of_the_sums():
    def game(decisions, evaluations, seconds, network, forwards, positions, end="turn_cap"):
        result = GameResult(config=PlayConfig(), record="", actions=[], end=end, winner=None)
        result.decisions, result.evaluations, result.seconds, result.network_seconds = decisions, evaluations, seconds, network
        return GameFigures(result, forwards, positions)

    t = totals([game(10, 100, 10.0, 8.0, 20, 100), game(2, 6, 0.5, 0.4, 3, 6, end="checkmate")])
    assert t["games"] == 2 and t["ends"] == {"turn_cap": 1, "checkmate": 1}
    assert (t["decisions"], t["evaluations"], t["forwards"], t["positions"]) == (12, 106, 23, 106)
    assert t["seconds"] == 10.5 and t["network_seconds"] == 8.4
    assert t["decisions_per_s"] == round(12 / 10.5, 3)
    assert t["evaluations_per_s"] == round(106 / 10.5, 2)  # 10.1/s - not the games' mean, 11.0
    assert t["forwards_per_s"] == round(23 / 10.5, 2)
    assert t["positions_per_forward"] == round(106 / 23, 2)
    assert t["network_positions_per_s"] == round(106 / 8.4, 2)
    assert t["network_share"] == round(8.4 / 10.5, 3)
    empty = totals([])
    assert empty["games"] == 0 and empty["evaluations_per_s"] == 0.0 and empty["positions_per_forward"] == 0.0


def test_the_ranges_are_6b_s():
    names = [range_of(n) for n in (1, 49, 50, 99, 100, 199, 200, 600)]
    assert names == ["< 50", "< 50", "50-99", "50-99", "100-199", "100-199", "200+", "200+"]


def test_each_call_is_filed_by_its_largest_position():
    evaluator = RangedEvaluator(build_model(network_config("small"), seed=1).eval(), max_batch=2)
    evaluator([synthetic_state(0, 3)])
    evaluator([synthetic_state(1, 4), synthetic_state(2, 60)])  # one forward, filed at 60 boards
    evaluator([synthetic_state(3, 5), synthetic_state(4, 6), synthetic_state(5, 7)])  # two forwards
    ranges = evaluator.by_range()
    assert set(ranges) == {"< 50", "50-99"}
    assert (ranges["< 50"]["forwards"], ranges["< 50"]["positions"]) == (3, 4)
    assert (ranges["50-99"]["forwards"], ranges["50-99"]["positions"]) == (1, 2)
    assert (evaluator.batches, evaluator.positions) == (4, 6)  # the evaluator's own counts, as before
    for r in ranges.values():
        assert r["seconds"] > 0 and r["positions_per_forward"] == round(r["positions"] / r["forwards"], 2)
        # The rate is of the unrounded seconds; the row's seconds are rounded to the millisecond.
        low, high = r["positions"] / (r["seconds"] + 5e-4), r["positions"] / max(r["seconds"] - 5e-4, 1e-9)
        assert low - 0.01 <= r["positions_per_s"] <= high + 0.01


@requires_engine
def test_one_short_f1_game_end_to_end(capsys):
    argv = ["--games", "1", "--variant", "Standard", "--cap-turns", "1", "--playouts", "4", "--batch", "4", "--threads", "2"]
    assert evalrate.main(argv) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 3
    conditions, game, total = lines[0]["conditions"], lines[1], lines[2]["totals"]

    # The configuration, in full, before the first game: the v1 network as shipped, every field.
    assert conditions["net"] == "F1"
    assert conditions["net_config"] == json.loads(json.dumps(asdict(NetConfig())))
    assert conditions["net_config"]["trunk_dtype"] == "bf16" and conditions["parameters"] > 20_000_000
    assert conditions["provers"].startswith("off")
    play = conditions["play"]
    assert (play["hunt"], play["settling"], play["asset_map"], play["mode"]) == (False, False, False, "self")
    assert (play["playouts"], play["batch_size"], play["cap_turns"], play["cap_boards"]) == (4, 4, 1, 80)
    assert conditions["variants"] == ["Standard"] and conditions["splits"] == ["ply"] and conditions["seeds"] == [1]
    assert conditions["games_per_variant_split"] == 1 and conditions["torch_threads"] == 2
    for key in ("started", "torch", "numpy", "python", "platform", "cpu", "logical_processors", "commit"):
        assert conditions[key], key
    assert "/" not in conditions["engine_library"] and "\\" not in conditions["engine_library"]  # a name, not a path

    # The game and the totals: every leaf the search counted reached the network, in forwards of at
    # most --batch positions.
    assert game["mode"] == "self" and game["seed"] == 1 and game["decisions"] > 0 and game["evaluations"] > 0
    assert game["positions"] == game["evaluations"]
    assert game["forwards"] >= 1 and game["positions"] <= 4 * game["forwards"]
    assert game["positions_per_forward"] == round(game["positions"] / game["forwards"], 2)
    assert (total["games"], total["decisions"], total["evaluations"]) == (1, game["decisions"], game["evaluations"])
    assert (total["forwards"], total["positions"]) == (game["forwards"], game["positions"])
    assert total["evaluations_per_s"] > 0 and total["decisions_per_s"] > 0 and total["network_positions_per_s"] > 0
    assert 0 < total["network_share"] <= 1
    # A one-turn Standard game stays under 50 boards: every call in that range.
    assert set(total["by_range"]) == {"< 50"}
    assert (total["by_range"]["< 50"]["forwards"], total["by_range"]["< 50"]["positions"]) == (game["forwards"], game["positions"])


@requires_engine
def test_the_provers_flag_reaches_the_games(capsys):
    argv = ["--net", "small", "--games", "1", "--variant", "Standard", "--cap-turns", "1", "--playouts", "2", "--provers"]
    assert evalrate.main(argv) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    conditions = lines[0]["conditions"]
    assert conditions["provers"].startswith("on")
    assert (conditions["play"]["hunt"], conditions["play"]["settling"], conditions["play"]["asset_map"]) == (True, True, True)
    assert lines[-1]["totals"]["games"] == 1
