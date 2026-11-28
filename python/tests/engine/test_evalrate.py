"""F1's CPU evaluation rate (``t5d.engine.evalrate``): the provers off unless asked, the totals as
rates of the sums, the network's calls filed by board-count range, where an evaluation's time goes
(section 15.3's input-assembly row: the conversion and the collation apart from the forward - in
each game's line, in the totals and by range; and ``t5d.engine.bench --assembly``, the same
without a forward), and one short game of the F1 network end to end - its configuration printed
in full before the game, its counts consistent. A smoke run and the arithmetic, never a timing:
the machine running the tests is shared, and a test is not a measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from t5d.engine import bench, evalrate
from t5d.engine.evalrate import (
    ASSEMBLY,
    PARTS,
    GameFigures,
    RangedEvaluator,
    assembly_breakdown,
    game_seeds,
    play_config,
    range_of,
    totals,
)
from t5d.engine.evaluator import Leaf, UniformEvaluator, serve
from t5d.engine.game import Game
from t5d.engine.netplay import GameResult, PlayConfig, network_config, play_game
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


def test_the_breakdown_is_assembly_against_the_forward_from_the_sums():
    assert PARTS == ("convert", "fit", "collate", "transfer", "forward", "split") and PARTS[:3] == ASSEMBLY
    seconds = {"convert": 0.2, "fit": 0.05, "collate": 0.25, "transfer": 0.0, "forward": 4.0, "split": 0.5}
    b = assembly_breakdown(seconds, 10)
    assert (b["convert_seconds"], b["fit_seconds"], b["collate_seconds"]) == (0.2, 0.05, 0.25)
    assert (b["transfer_seconds"], b["forward_seconds"], b["split_seconds"]) == (0.0, 4.0, 0.5)
    assert b["assembly_seconds"] == 0.5  # the conversion, the fit and the collation
    assert b["assembly_ms_per_position"] == 50.0
    assert b["assembly_share"] == 0.1  # 0.5 of the six parts' 5.0 s
    assert b["forward_ms_per_position"] == 400.0 and b["forward_share"] == 0.8
    empty = assembly_breakdown(dict.fromkeys(PARTS, 0.0), 0)
    assert empty["assembly_ms_per_position"] == 0.0 and empty["assembly_share"] == 0.0 and empty["forward_share"] == 0.0

    def game(parts, positions):
        return GameFigures(GameResult(config=PlayConfig(), record="", actions=[], end="turn_cap", winner=None), 1, positions, parts)

    # A game's line and the totals carry it: rates of the summed seconds, never a mean of the games'.
    one = game({"convert": 0.1, "fit": 0.0, "collate": 0.1, "transfer": 0.0, "forward": 0.8, "split": 0.0}, 4)
    two = game({"convert": 0.3, "fit": 0.0, "collate": 0.5, "transfer": 0.0, "forward": 8.0, "split": 0.2}, 16)
    assert one.row()["assembly_share"] == 0.2 and one.row()["assembly_ms_per_position"] == 50.0
    t = totals([one, two])
    assert t["assembly_seconds"] == 1.0 and t["forward_seconds"] == 8.8
    assert t["assembly_ms_per_position"] == 50.0  # 1.0 s over 20 positions
    assert t["assembly_share"] == 0.1  # 1.0 of 10.0 s - not the games' mean share, 0.144
    plain = GameFigures(GameResult(config=PlayConfig(), record="", actions=[], end="turn_cap", winner=None), 1, 4)
    assert plain.row()["assembly_seconds"] == 0.0  # an evaluator that keeps no parts: zeros, not an error


def test_the_evaluator_times_its_parts_and_files_the_conversion_by_range():
    evaluator = RangedEvaluator(build_model(network_config("small"), seed=1).eval(), max_batch=4)
    assert evaluator.parts() == dict.fromkeys(PARTS, 0.0)
    small, large = [synthetic_state(0, 3)], [synthetic_state(1, 4), synthetic_state(2, 60)]
    evaluator.note_conversion(small, 0.25)  # what the serve loop reports before each call
    evaluator(small)
    evaluator.note_conversion(large, 0.5)  # filed, as the call will be, at 60 boards
    evaluator(large)
    parts = evaluator.parts()
    assert parts["convert"] == 0.75 and evaluator.convert_seconds == 0.75
    assert parts["collate"] > 0 and parts["forward"] > 0 and parts["split"] > 0
    assert parts["fit"] >= 0 and parts["transfer"] >= 0
    ranges = evaluator.by_range()
    assert set(ranges) == {"< 50", "50-99"}
    assert (ranges["< 50"]["convert_seconds"], ranges["50-99"]["convert_seconds"]) == (0.25, 0.5)
    own = 0.0
    for name, r in ranges.items():
        assert set(assembly_breakdown(dict.fromkeys(PARTS, 0.0), 0)) <= set(r)
        # The evaluator's own five parts lie inside its call's time; the conversion is before it.
        inside = r["fit_seconds"] + r["collate_seconds"] + r["transfer_seconds"] + r["forward_seconds"] + r["split_seconds"]
        assert 0 < inside <= r["seconds"] + 5 * 5e-5 + 5e-4, name
        own += inside
        assert r["assembly_seconds"] == pytest.approx(r["convert_seconds"] + r["fit_seconds"] + r["collate_seconds"], abs=2e-4)
        assert 0 < r["assembly_share"] < 1 and 0 < r["forward_share"] < 1
        assert r["assembly_share"] + r["forward_share"] <= 1 + 1e-3
        assert r["assembly_ms_per_position"] == pytest.approx(1e3 * r["assembly_seconds"] / r["positions"], rel=1e-2, abs=1e-3)
    # The ranges' parts are the evaluator's, split by where each call was filed.
    assert own == pytest.approx(sum(parts[p] for p in PARTS[1:]), abs=1e-3)


def test_serve_reports_each_batchs_conversion():
    class Driver:
        """Two batches of leaves; ``convert_seconds`` grows by 1/8 s at each collect, as the step
        driver's does by what the conversion took."""

        def __init__(self, batches, counts=True):
            self.batches = list(batches)
            self.delivered = 0
            if counts:
                self.convert_seconds = 0.0

        def collect(self, max_leaves):
            if not self.batches:
                return []
            if hasattr(self, "convert_seconds"):
                self.convert_seconds += 0.125
            return [Leaf(i, state) for i, state in enumerate(self.batches.pop(0))]

        def deliver(self, results):
            self.delivered += len(results)

        def finished(self):
            return not self.batches

    class Noting(UniformEvaluator):
        def __init__(self):
            super().__init__()
            self.notes = []
            self.calls = []

        def note_conversion(self, positions, seconds):
            self.notes.append((len(positions), seconds, len(self.calls)))

        def __call__(self, positions):
            self.calls.append(len(positions))
            return super().__call__(positions)

    batches = [[synthetic_state(0, 2), synthetic_state(1, 3)], [synthetic_state(2, 4)]]
    driver, evaluator = Driver(batches), Noting()
    stats = serve(driver, evaluator, batch_size=2)
    assert (stats.rounds, stats.leaves, driver.delivered) == (2, 3, 3)
    assert stats.convert_seconds == 0.25  # each batch's share, summed
    # Told before the call it belongs to: the batch's positions and its conversion's seconds.
    assert evaluator.notes == [(2, 0.125, 0), (1, 0.125, 1)] and evaluator.calls == [2, 1]
    # A driver that keeps no count: nothing to add, and the evaluator is told 0.
    quiet, evaluator = Driver(batches, counts=False), Noting()
    assert serve(quiet, evaluator, batch_size=2).convert_seconds == 0.0
    assert evaluator.notes == [(2, 0.0, 0), (1, 0.0, 1)]
    # An evaluator without the method is served as before.
    assert serve(Driver(batches), UniformEvaluator(), batch_size=2).convert_seconds == 0.25


def test_the_assembly_rows_are_by_range_and_of_the_samples_sums():
    sample = bench.AssemblySample
    samples = [
        sample(boards=10, expand=5e-3, convert=1e-3, fit=0.0, collate={1: 2e-3, 4: 1e-3}),
        sample(boards=44, expand=7e-3, convert=3e-3, fit=1e-3, collate={1: 4e-3, 4: 2e-3}),
        sample(boards=60, expand=9e-3, convert=5e-3, fit=0.0, collate={1: 6e-3, 4: 3e-3}),
    ]
    assert samples[1].assembly(1) == pytest.approx(8e-3) and samples[1].assembly(4) == pytest.approx(6e-3)
    rows = bench.assembly_rows(samples, [1, 4])
    assert [r["range"] for r in rows] == ["< 50", "50-99"]  # a range with no sample gives no row
    low, mid = rows
    assert (low["samples"], low["boards"], mid["samples"], mid["boards"]) == (2, "10/44", 1, "60")
    assert low["expand ms"] == pytest.approx(6.0) and low["to_position ms"] == pytest.approx(2.0)
    assert low["fit ms"] == pytest.approx(0.5)
    assert low["collate ms/pos (B=1)"] == pytest.approx(3.0) and low["collate ms/pos (B=4)"] == pytest.approx(1.5)
    # The assembly is to_position + fit + collate, sample by sample: 3 and 8 ms at B=1.
    assert low["assembly ms/pos (B=1)"] == pytest.approx(5.5)
    assert low["assembly p95 (B=1)"] == pytest.approx(3.0 + 0.95 * 5.0)
    assert low["assembly ms/pos (B=4)"] == pytest.approx(4.0)  # 2 and 6 ms
    assert mid["assembly ms/pos (B=1)"] == pytest.approx(11.0) and mid["assembly p95 (B=4)"] == pytest.approx(8.0)
    assert bench.assembly_rows([], [1]) == []


@requires_engine
def test_the_assembly_bench_runs_without_a_model(capsys, monkeypatch):
    def no_model(*args, **kwargs):
        raise AssertionError("--assembly builds no model")

    monkeypatch.setattr(bench, "build_model", no_model)
    argv = ["--assembly", "--assembly-boards", "2", "5", "--seeds", "1", "--repeat", "1", "--batch", "1", "2"]
    assert bench.main(argv) == 0
    out = capsys.readouterr().out
    assert "no model built, no forward" in out and "input assembly per position, by board-count range" in out
    lines = out.splitlines()
    header = next(i for i, line in enumerate(lines) if line.lstrip().startswith("range"))
    assert "assembly ms/pos (B=2)" in lines[header] and "to_position ms" in lines[header]
    rows = [line for line in lines[header + 1 :] if line.strip()]
    assert len(rows) == 1 and rows[0].lstrip().startswith("< 50")  # 2 and 5 boards: one range


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

    # Where the time went (section 15.3's input-assembly row): the step driver's conversion and the
    # evaluator's collation apart from the forward, in the game's line, the totals and the range -
    # one game in one range, so the three agree.
    for row in (game, total, total["by_range"]["< 50"]):
        assert row["convert_seconds"] > 0 and row["collate_seconds"] > 0 and row["forward_seconds"] > 0
        assert row["transfer_seconds"] >= 0 and row["fit_seconds"] >= 0
        assert row["assembly_seconds"] == pytest.approx(row["convert_seconds"] + row["fit_seconds"] + row["collate_seconds"], abs=2e-4)
        assert 0 < row["assembly_share"] < row["forward_share"] < 1  # F1 on the CPU: the forward dominates
        assert row["assembly_ms_per_position"] > 0 and row["forward_ms_per_position"] > row["assembly_ms_per_position"]
    for key in ("convert_seconds", "collate_seconds", "forward_seconds", "assembly_seconds", "assembly_share"):
        assert game[key] == total[key] == total["by_range"]["< 50"][key], key
    # The conversion is the driver's, inside the game's collect phase; the evaluator's parts are
    # inside its network seconds.
    assert game["convert_seconds"] <= game["collect_seconds"] + 1e-3
    inside = game["fit_seconds"] + game["collate_seconds"] + game["transfer_seconds"] + game["forward_seconds"] + game["split_seconds"]
    assert inside <= game["network_seconds"] + 1e-3


@requires_engine
def test_the_callers_cut_ends_a_game_after_the_decision_in_flight():
    """``netplay.play_game``'s ``should_stop`` (``evalrate --max-seconds``): asked after each
    decision; the game ends there unscored, an action under way not recorded, its record whole."""
    net_cfg = network_config("small")
    evaluator = RangedEvaluator(build_model(net_cfg, seed=1).eval(), max_batch=2)
    for stop_at in (1, 2, 3):
        calls = []

        def should_stop(calls=calls, stop_at=stop_at):
            calls.append(len(calls))
            return len(calls) >= stop_at

        cfg = play_config("Standard", "ply", 1, playouts=2, batch_size=2, cap_turns=6)
        res = play_game(evaluator, cfg, net_cfg=net_cfg, should_stop=should_stop)
        assert (res.end, res.result, res.winner, res.unsettled) == ("cut", "*", None, "")
        assert len(calls) == stop_at  # asked once after each decision or action, and not again once it said stop
        assert 1 <= res.decisions <= stop_at and len(res.actions) < res.decisions
        if stop_at == 1:
            assert (res.decisions, res.actions) == (1, [])  # cut mid-action: the move made is not recorded
        # The record is a whole game so far: the engine's reader replays it to the game's last turn start.
        assert "cut (the caller stopped the game after a decision" in res.record
        with Game.from_pgn(res.record) as g:
            g.replay_all()
            assert len(g.snapshot().boards) == res.final_boards == len(res.actions) + 1
    # Without it nothing changes: the same game runs to its cap.
    whole = play_game(evaluator, play_config("Standard", "ply", 1, playouts=2, batch_size=2, cap_turns=1), net_cfg=net_cfg)
    assert whole.end == "turn_cap"


@requires_engine
def test_a_run_past_its_budget_reports_the_cut_game_and_starts_no_other(capsys):
    """``--max-seconds``: two games asked and a budget that is spent before the first decision
    returns - the first game is cut after that decision and reported, the second never starts, and
    the totals and the ranges are the cut game's."""
    base = ["--net", "small", "--variant", "Standard", "--playouts", "2"]
    assert evalrate.main([*base, "--games", "2", "--cap-turns", "6", "--max-seconds", "0.000001"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 3  # the conditions, the one game that started, the totals
    conditions, game, total = lines[0]["conditions"], lines[1], lines[2]["totals"]
    assert conditions["max_seconds"] == 1e-6 and conditions["games_planned"] == 2
    assert (game["end"], game["result"], game["decisions"], game["actions"]) == ("cut", "*", 1, 0)
    assert game["evaluations"] > 0 and game["positions"] == game["evaluations"] and game["forwards"] >= 1
    assert total["ends"] == {"cut": 1}
    assert (total["games"], total["games_planned"], total["games_cut"], total["games_not_played"]) == (1, 2, 1, 1)
    # A cut game's figures are valid: counts and sums over what it played, rates of those sums.
    assert (total["decisions"], total["evaluations"], total["positions"]) == (1, game["evaluations"], game["positions"])
    assert total["evaluations_per_s"] > 0 and total["decisions_per_s"] > 0 and total["assembly_seconds"] > 0
    assert total["by_range"]["< 50"]["positions"] == game["positions"]
    # A budget the run stays inside changes nothing, and is recorded.
    assert evalrate.main([*base, "--games", "1", "--cap-turns", "1", "--max-seconds", "86400"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["conditions"]["max_seconds"] == 86400.0 and lines[1]["end"] != "cut"
    assert (lines[2]["totals"]["games_cut"], lines[2]["totals"]["games_not_played"]) == (0, 0)
    # No budget: recorded as none.
    assert evalrate.main([*base, "--games", "1", "--cap-turns", "1"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["conditions"]["max_seconds"] is None


@requires_engine
def test_the_provers_flag_reaches_the_games(capsys):
    argv = ["--net", "small", "--games", "1", "--variant", "Standard", "--cap-turns", "1", "--playouts", "2", "--provers"]
    assert evalrate.main(argv) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    conditions = lines[0]["conditions"]
    assert conditions["provers"].startswith("on")
    assert (conditions["play"]["hunt"], conditions["play"]["settling"], conditions["play"]["asset_map"]) == (True, True, True)
    assert lines[-1]["totals"]["games"] == 1
