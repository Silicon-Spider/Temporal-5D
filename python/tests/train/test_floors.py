"""Register I22's board-count floors in the sampler (spec section 12, "Stratified sampling";
section 12.8; review decisions 22 and 23; ``train.sampler``, ``train.window``): the floor
arithmetic by hand-computed cases - a constraint, water-filled; a range with no data has no floor;
floors past the whole refused - training compute as section 9.1's dense model, the window's
per-range pools under eviction, and the trainer's draw: off (the default) the as-built uniform
draw object for object, on the floored shares of the compute, and both composing with register
I23's token-budget batches. Synthetic rows only; one optimiser step."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model, synthetic_state, synthetic_states
from t5d.net.cost import dense_flops, token_counts
from t5d.train.recipe import RecipeConfig, range_index, row_boards
from t5d.train.sampler import (
    SamplerConfig,
    _bucket,
    board_cost,
    coverage_report,
    floored_shares,
    range_draw,
    range_names,
    range_probabilities,
    row_cost,
)
from t5d.train.selfplay import board_range
from t5d.train.smoke import with_actions
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import ReplayWindow, WindowConfig
from tests.train.test_targets import _line

NET = NetConfig(**SMALL)
RANGES = (50, 100, 200)
SMALL_WINDOW = WindowConfig(min_rows=1000)

# --- ranges and compute ------------------------------------------------------------------------------------------


def test_range_index_matches_board_range():
    """Section 16's four ranges - under 50, 50-99, 100-199, 200 and up - indexed as they are named."""
    assert range_names(RANGES) == ["0-49", "50-99", "100-199", "200+"]
    for n in range(260):
        assert range_names(RANGES)[range_index(n, RANGES)] == board_range(n, RANGES)
    assert [range_index(n, RANGES) for n in (0, 49, 50, 99, 100, 199, 200, 10_000)] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_training_compute_is_the_dense_model():
    """Training compute is section 9.1's analytic dense cost at the trainer's network: section
    9.2's 0.262 TF at 44 boards and 4.78 TF at 240 for the v1, and the spec's "a 170-board sample
    costs ten times a 44-board one" (9.69x); a row's cost is that model at its own token count -
    register A5's count for a piece-centric network - as an exact Python int."""
    v1 = NetConfig()
    assert board_cost(44, v1) == 262_170_214_400 and board_cost(240, v1) == 4_784_195_108_864
    assert board_cost(170, v1) / board_cost(44, v1) == pytest.approx(9.687, rel=1e-3)
    state = synthetic_state(1, 5)
    assert row_cost(state, v1) == dense_flops(5 * 66 + 16, v1)["total"] == board_cost(5, v1)
    assert type(row_cost(state, v1)) is int and type(board_cost(5, v1)) is int
    a5 = replace(v1, piece_centric=True)
    assert row_cost(state, a5) == dense_flops(token_counts(state)["a5"], a5)["total"] < row_cost(state, v1)
    assert 22_000_000 * board_cost(240, v1) > 2**63  # why the sums are Python ints: a full window of large rows passes int64


# --- the floor arithmetic ------------------------------------------------------------------------------------------


def test_floors_that_bind_nothing_change_nothing():
    """Section 12: "if its games are already large, the floors bind nothing" - the natural shares
    come back untouched - the floor is a constraint, not a reservation."""
    assert floored_shares([4, 3, 2, 1], 0.1).tolist() == [0.4, 0.3, 0.2, 0.1]
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.uniform(0.1, 10.0, size=4)
        assert floored_shares(c, 0.0).tolist() == (c / c.sum()).tolist()


def test_a_bound_range_is_raised_and_the_rest_keep_their_proportions():
    t = floored_shares([80, 15, 4, 1], 0.1)
    assert t.tolist() == pytest.approx([0.8 * 0.8 / 0.95, 0.15 * 0.8 / 0.95, 0.1, 0.1], abs=1e-12)
    assert t.sum() == pytest.approx(1.0, abs=1e-12) and t[0] / t[1] == pytest.approx(80 / 15)


def test_raising_one_range_can_bind_the_next():
    """The 11% range falls to 9.2% once the two small ranges are raised, and is bound in the second pass."""
    assert floored_shares([85, 11, 3, 1], 0.1).tolist() == pytest.approx([0.7, 0.1, 0.1, 0.1], abs=1e-12)


def test_a_range_with_no_data_has_no_floor():
    """Section 12: "a tenth of the training compute held for each range that has data" - the human games'
    measured dense split (docs/MEASUREMENTS.md: 54.4 / 36.0 / 9.6 / nothing at 200 and up) holds a
    tenth for the 100-199 range and nothing for the empty one."""
    t = floored_shares([54.4, 36.0, 9.6, 0.0], 0.1)
    assert t.tolist() == pytest.approx([0.9 * 54.4 / 90.4, 0.9 * 36.0 / 90.4, 0.1, 0.0], abs=1e-12)
    assert floored_shares([0, 0, 5, 0], 0.1).tolist() == [0.0, 0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="I22"):
        floored_shares([0, 0, 0, 0], 0.1)


def test_floors_past_the_whole_are_refused():
    """Refused, never rescaled - rescaling would silently change what "a tenth" means. Floors that
    sum to exactly the whole are allowed and give each range its floor."""
    with pytest.raises(ValueError, match="at most the whole"):
        floored_shares([1, 1, 1, 1], 0.3)
    assert floored_shares([70, 20, 9, 1], 0.25).tolist() == pytest.approx([0.25] * 4, abs=1e-12)
    assert floored_shares([97, 1, 1, 1], [0.0, 0.2, 0.3, 0.4]).tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-12)
    assert floored_shares([1, 1, 0, 0], 0.4).tolist() == [0.5, 0.5, 0.0, 0.0]  # 0.4 x 2 ranges with data: within the whole
    for bad in (SamplerConfig(floor_share=0.3), SamplerConfig(floor_share=-0.1), SamplerConfig(floor_ranges=(100, 50))):
        with pytest.raises(ValueError, match="I22"):
            bad.validate()
        with pytest.raises(ValueError, match="I22"):
            TrainerConfig(sampler=bad).validate()
    with pytest.raises(ValueError, match="I22"):
        SamplerConfig(floor_ranges=(0, 50)).validate()
    SamplerConfig(floor_share=0.25).validate()
    SamplerConfig(floor_share=0.5, floor_ranges=(50,)).validate()
    with pytest.raises(ValueError, match="I22"):
        floored_shares([1, -1], 0.1)


def test_draw_probabilities_deliver_the_shares_in_compute():
    """A floor in compute, not in samples: a window of 9,900 / 80 / 15 / 5 rows at 24 / 70 / 150 /
    220 boards has three ranges under a tenth of the compute; the draw raises each to it - compute
    is spent 0.7 / 0.1 / 0.1 / 0.1 in expectation - and with no floor it is the uniform draw."""
    v1 = NetConfig()
    rows, boards = (9900, 80, 15, 5), (24, 70, 150, 220)
    strata = {k: (r, r * board_cost(b, v1)) for k, (r, b) in enumerate(zip(rows, boards, strict=True))}
    total = sum(c for _, c in strata.values())
    assert [c / total for _, c in strata.values()] == pytest.approx([0.9195, 0.0372, 0.0259, 0.0173], abs=1e-4)
    p = range_draw(strata, SamplerConfig())
    assert list(p) == [0, 1, 2, 3] and sum(p.values()) == pytest.approx(1.0)
    assert list(p.values()) == pytest.approx([0.96152, 0.02741, 0.00739, 0.00368], abs=1e-5)
    spent = np.array([p[k] * board_cost(b, v1) for k, b in enumerate(boards)])
    assert (spent / spent.sum()).tolist() == pytest.approx([0.7, 0.1, 0.1, 0.1], abs=1e-12)
    uniform = range_draw(strata, SamplerConfig(floor_share=0.0))
    assert list(uniform.values()) == pytest.approx([r / 10_000 for r in rows], abs=1e-15)
    report = coverage_report(strata, SamplerConfig())
    assert list(report) == ["0-49", "50-99", "100-199", "200+"]
    assert report["200+"]["rows"] == 5 and report["200+"]["row_share"] == pytest.approx(0.0005)
    assert report["200+"]["compute_share"] == pytest.approx(0.0173, abs=1e-4) and report["200+"]["floored_share"] == pytest.approx(0.1)
    assert report["200+"]["draw"] == p[3] and report["200+"]["draw"] / report["200+"]["row_share"] == pytest.approx(7.36, abs=0.01)
    partial = {0: (10, 1000), 2: (1, 50)}  # ranges without rows are left out, and keyed as the window keys them
    assert list(range_draw(partial, SamplerConfig())) == [0, 2] and list(coverage_report(partial, SamplerConfig())) == ["0-49", "100-199"]
    assert range_probabilities([0.5, 0.5, 0.0], [1.0, 4.0, 0.0]).tolist() == pytest.approx([0.8, 0.2, 0.0])
    with pytest.raises(ValueError, match="I22"):
        range_draw({}, SamplerConfig())


# --- the window ------------------------------------------------------------------------------------------------------


def test_the_window_keeps_each_range_under_eviction():
    """The window's per-range rows and exact compute follow its eviction; a row written without a
    range lands in range 0 at cost 1; and the uniform draw is untouched - the as-built draw, row
    for row."""
    w = ReplayWindow(WindowConfig(min_rows=10, exponent=0.5, expand=0.1))
    written = []
    for i in range(100):
        stratum, cost, copies = i % 3, 10**13 + i, 2 if i % 10 == 0 else 1
        w.add(i, copies, stratum=stratum, cost=cost)
        written += [(i, stratum, cost)] * copies
        kept = written[-len(w) :]
        strata = w.strata()
        assert sum(n for n, _ in strata.values()) == len(w) == len(kept)
        for k in range(3):
            mine = [c for _, s, c in kept if s == k]
            assert strata.get(k) == ((len(mine), sum(mine)) if mine else None)
    assert w.rows() == [i for i, _, _ in written[-len(w) :]] and len(w) < 20 < w.written
    plain = ReplayWindow(WindowConfig(min_rows=10, exponent=0.5, expand=0.1))
    for i in range(100):
        plain.add(i, copies=2 if i % 10 == 0 else 1)
    assert plain.strata() == {0: (len(plain), len(plain))} and plain.rows() == w.rows()
    rows = w.rows()
    drawn = w.sample(50, np.random.default_rng(0))
    assert drawn == [rows[i] for i in np.random.default_rng(0).integers(0, len(rows), size=50)]
    gone = ReplayWindow(WindowConfig(min_rows=2, exponent=0.5, expand=0.1))
    gone.add("a", stratum=3, cost=7)
    for _ in range(8):
        gone.add("b", stratum=1, cost=2)
    assert list(gone.strata()) == [1]  # an emptied range is absent
    with pytest.raises(ValueError, match="I22"):
        gone.add("c", cost=-1)


def test_the_window_draws_by_range_and_keeps_surprise_inside_it():
    """A range by the given probabilities, then a row uniformly among that range's written rows -
    a row written three times (its I12 weight) drawn three times as often."""
    w = ReplayWindow(SMALL_WINDOW)
    w.add("a1", 1, stratum=0, cost=5)
    w.add("a3", 3, stratum=0, cost=5)
    w.add("b", 2, stratum=1, cost=9)
    w.add("c", 1, stratum=2, cost=20)
    assert w.strata() == {0: (4, 20), 1: (2, 18), 2: (1, 20)}
    drawn = w.sample_stratified(20_000, np.random.default_rng(3), {0: 0.5, 2: 0.5})
    count = Counter(drawn)
    assert set(count) == {"a1", "a3", "c"}
    assert count["c"] / 20_000 == pytest.approx(0.5, abs=0.02)
    assert count["a3"] / count["a1"] == pytest.approx(3.0, abs=0.25)
    assert drawn == w.sample_stratified(20_000, np.random.default_rng(3), {0: 0.5, 2: 0.5})
    assert w.sample_stratified(5, np.random.default_rng(0), {1: 1.0}) == ["b"] * 5
    for bad in ({0: 0.5, 3: 0.5}, {0: 0.5, 1: 0.4}, {0: 1.5, 1: -0.5}):
        with pytest.raises(ValueError, match="I22"):
            w.sample_stratified(4, np.random.default_rng(0), bad)
    with pytest.raises(ValueError, match="empty"):
        ReplayWindow(SMALL_WINDOW).sample_stratified(1, np.random.default_rng(0), {0: 1.0})


# --- the trainer's draw ------------------------------------------------------------------------------------------


def _trainer(counts, seed, **cfg):
    states = with_actions(synthetic_states(seed, list(counts)))
    record = synthetic_record(seed, states)
    config = TrainerConfig(**{**dict(batch_size=8, window=SMALL_WINDOW), **cfg})
    trainer = Trainer(build_model(replace(NET, trunk_dtype="fp32"), seed=seed), NET, config)
    trainer.add_game(record, positions=states)
    return trainer, states


def test_floors_off_is_the_window_s_uniform_draw():
    """Off - the default: the pilot run plays without floors (section 15.7) - the trainer's draw is
    the as-built ``window.sample`` from its own generator, object for object; the window still
    reads rows and compute per range."""
    off = SamplerConfig()
    assert (off.range_floors, off.floor_share, off.floor_ranges) == (False, 0.1, (50, 100, 200))
    trainer, states = _trainer((1, 3, 2, 4, 2, 5), 5)
    rows = trainer.window.rows()
    as_built = [rows[i] for i in np.random.default_rng(trainer.cfg.seed).integers(0, len(rows), size=16)]
    drawn = trainer.draw_rows(16)
    assert len(drawn) == 16 and all(a is b for a, b in zip(drawn, as_built, strict=True))
    strata = trainer.window.strata()
    assert list(strata) == [0] and strata[0] == (len(rows), sum(row_cost(r, NET) for r in rows))
    assert {row_cost(s, NET) for s in states} == {dense_flops(66 * n + 16, NET)["total"] for n in (1, 2, 3, 4, 5)}


@pytest.mark.grad
def test_floors_on_in_the_trainer():
    """On, over four ranges of which three hold little of the compute: each range's share of the
    drawn rows' compute is at least its floor and is the floored share the report states; inside a
    range the draw follows the rows' copies; and a step on the draw runs (synthetic rows, one
    step)."""
    floors = SamplerConfig(range_floors=True, floor_share=0.2, floor_ranges=(2, 3, 4))
    names = range_names(floors.floor_ranges)
    trainer, _ = _trainer((1, 1, 1, 2, 2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6), 8, sampler=floors)
    strata = trainer.window.strata()
    assert list(strata) == [0, 1, 2, 3]
    report = coverage_report(strata, floors)
    natural = [report[name]["compute_share"] for name in names]
    assert sum(x < 0.2 for x in natural) == 3 and natural[3] > 0.6  # three ranges under the floor, the large one dominant
    assert [report[name]["floored_share"] for name in names] == pytest.approx([0.2, 0.2, 0.2, 0.4])
    drawn = trainer.draw_rows(6000)
    spent, count = np.zeros(4), np.zeros(4)
    for r in drawn:
        k = range_index(row_boards(r), floors.floor_ranges)
        spent[k] += row_cost(r, NET)
        count[k] += 1
    for k, name in enumerate(names):
        assert spent[k] / spent.sum() == pytest.approx(report[name]["floored_share"], abs=0.02), name
        assert spent[k] / spent.sum() >= 0.2 - 0.02, name
        assert count[k] / 6000 == pytest.approx(report[name]["draw"], abs=0.02), name
    assert all(report[name]["draw"] > report[name]["row_share"] for name in names[:3])  # the starved ranges are drawn more often
    written = Counter(id(r) for r in trainer.window.rows() if row_boards(r) == 6)
    seen = Counter(id(r) for r in drawn if row_boards(r) == 6)
    assert len(set(written.values())) > 1  # I12's copies differ inside the range
    for key, n in written.items():  # surprise weighting inside the range: a row is drawn as often as it was written
        assert seen[key] / sum(seen.values()) == pytest.approx(n / sum(written.values()), abs=0.03)
    assert all(math.isfinite(v) for v in trainer.step().values()) and trainer.steps == 1


@pytest.mark.grad
def test_floors_and_the_token_budget_compose():
    """One row-drawing path for both rulings: with the floors and register I23's token budget on,
    a pool is drawn by range under the floors and then cut into batches within those ranges - the
    rows served are exactly the floored draw's, each once."""
    floors = SamplerConfig(range_floors=True, floor_share=0.2, floor_ranges=(2, 3, 4))
    recipe = RecipeConfig(token_budget=900, pool_rows=40)
    counts = (1, 1, 1, 2, 2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6)
    trainer, _ = _trainer(counts, 9, sampler=floors, recipe=recipe)
    twin, _ = _trainer(counts, 9, sampler=floors, recipe=recipe)
    pool = twin.window.sample_stratified(40, np.random.default_rng(twin.cfg.seed), range_draw(twin.window.strata(), floors))
    served = [trainer.next_batch()]
    while trainer._pending:
        served.append(trainer.next_batch())
    assert sorted((r.index, row_boards(r)) for b in served for r in b) == sorted((r.index, row_boards(r)) for r in pool)
    assert all(len({range_index(row_boards(r), floors.floor_ranges) for r in b}) == 1 for b in served)
    assert len({range_index(row_boards(r), floors.floor_ranges) for r in pool}) == 4
    assert all(math.isfinite(v) for v in trainer.step().values())
    assert trainer.step_log[0]["padded_tokens"] <= 900 or trainer.step_log[0]["rows"] == 1


# --- I6's prototype bucket ---------------------------------------------------------------------------------------


def test_the_bucket_phase_reads_a_seeded_game_s_turn():
    """A seeded game keeps its source's turn count (section 12): I6's prototype phase bucket reads
    the source's turn plus the game's own action - and a game from the start is unchanged."""
    cfg = SamplerConfig(phase_turns=(20, 60))
    record = _line(2, end="checkmate", winner=0)
    assert record.start_turn == 0 and record.seeded_start is None
    assert _bucket(record, 0, cfg) == ("0-49", 0) == _bucket(record, 1, cfg)
    record.header["seeded_start"] = {"turn": 30}
    assert record.start_turn == 30 and _bucket(record, 0, cfg) == ("0-49", 1)
    record.header["seeded_start"] = {"turn": 59}
    assert _bucket(record, 0, cfg)[1] == 1 and _bucket(record, 1, cfg)[1] == 2
