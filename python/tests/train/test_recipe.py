"""Register I23's recipe for the pilot run (spec sections 12.8 and 7.8; ``train.recipe``): batches
assembled to a token budget within a board-count range, a linear warm-up into piecewise-constant
steps - both off by default, and the default proven the smoke test's recipe: the same rows drawn,
object for object, the same constant rate, the same first-step losses. At most three optimiser
steps per model, on synthetic rows (BUILD_LOG issue 237(a)); steps past the third are covered by
the pure schedule function alone.

Trained weights are compared by norms, never element for element: two identical backward passes
differ in the last bits of one gradient tensor as built, so weights after a step are not
bit-reproducible (forward losses are)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine.netplay import SMALL
from t5d.net import NetConfig, build_model, cost, synthetic_states
from t5d.net.state import to_piece_centric
from t5d.train.losses import compute_losses
from t5d.train.recipe import (
    PILOT_POOL_ROWS,
    PILOT_RECIPE,
    PILOT_TOKEN_BUDGET,
    PILOT_WARMUP_STEPS,
    RecipeConfig,
    assemble_batches,
    lr_factor,
    range_index,
    row_boards,
    row_tokens,
)
from t5d.train.sampler import SamplerConfig
from t5d.train.selfplay import board_range
from t5d.train.smoke import MAX_STEPS, optimiser_steps, with_actions
from t5d.train.swa import SWAConfig
from t5d.train.targets import Row, build_targets, collate_rows
from t5d.train.trainer import Trainer, TrainerConfig, synthetic_record
from t5d.train.window import WindowConfig

pytestmark = pytest.mark.grad

NET = NetConfig(**SMALL)
RANGES = (50, 100, 200)
SMALL_WINDOW = WindowConfig(min_rows=1000)


def _config(**kw) -> TrainerConfig:
    return TrainerConfig(**{**dict(batch_size=8, swa=SWAConfig(every_steps=1, max_snapshots=10), window=SMALL_WINDOW), **kw})


def _synthetic(seed: int, counts=(1, 3, 2, 4, 2, 5)):
    states = with_actions(synthetic_states(seed, list(counts)))
    record = synthetic_record(seed, states)
    targets = build_targets(record)
    return record, states, [Row(s, targets, i, synthetic=True) for i, s in enumerate(states)]


def _model(seed: int, net: NetConfig = NET):
    return build_model(replace(net, trunk_dtype="fp32"), seed=seed)


def _change(model, before) -> float:
    """The L2 norm of the model's total parameter change since ``before``."""
    return math.sqrt(sum(float((p.detach() - b).pow(2).sum()) for p, b in zip(model.parameters(), before, strict=True)))


# --- the default is the smoke test's recipe ---------------------------------------------------------------


def test_the_default_recipe_is_the_smoke_tests():
    """Off equals what was there before: the default recipe has no budget, no warm-up and no drops;
    its factor is exactly 1 at every step; a step's rows are the window's uniform draw of
    ``batch_size`` from the trainer's own generator - the as-built draw, object for object; the
    first step's losses are those of that batch on a twin network; and the optimiser's rate stays
    the constant ``lr`` exactly."""
    assert TrainerConfig().recipe == RecipeConfig()
    off = RecipeConfig()
    assert (off.token_budget, off.warmup_steps, off.lr_boundaries, off.lr_factors) == (None, 0, (), ())
    assert all(lr_factor(s, off) == 1.0 for s in (0, 1, 2, 10**6))
    record, states, _ = _synthetic(11)
    cfg = _config()
    trainer = Trainer(_model(11), NET, cfg)
    trainer.add_game(record, positions=states)
    as_built = trainer.window.sample(cfg.batch_size, np.random.default_rng(cfg.seed))  # what Trainer.step drew before I23
    drawn = trainer.next_batch()
    assert len(drawn) == 8 and all(a is b for a, b in zip(drawn, as_built, strict=True))

    stepping = Trainer(_model(11), NET, cfg)
    stepping.add_game(record, positions=states)
    twin = _model(11).train()
    batch, tgt = collate_rows(as_built, NET)
    want = {k: float(v.detach()) for k, v in compute_losses(twin(batch), tgt, cfg.loss).items()}
    assert stepping.step() == want  # the same rows, the same forward: bit for bit
    for _ in range(MAX_STEPS - 1):
        stepping.step()
    assert [g["lr"] for g in stepping.opt.param_groups] == [cfg.lr]
    assert [(e["step"], e["lr"], e["rows"]) for e in stepping.step_log] == [(i, cfg.lr, 8) for i in range(MAX_STEPS)]
    assert len(stepping.history) == MAX_STEPS and stepping.swa.snapshots == MAX_STEPS


# --- the schedule ---------------------------------------------------------------------------------------------


def test_the_schedule_is_a_linear_warm_up_then_steps():
    """Section 12.8's "a warm-up and a schedule" as built: ``min(1, (s + 1) / W)`` times the factor
    of the last boundary passed - never zero, the factors not cumulative, the two multiplying."""
    cfg = RecipeConfig(warmup_steps=4, lr_boundaries=(10, 20), lr_factors=(0.3, 0.1))
    cfg.validate()
    assert [lr_factor(s, cfg) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert [lr_factor(s, cfg) for s in (4, 9)] == [1.0, 1.0]
    assert [lr_factor(s, cfg) for s in (10, 19)] == [0.3, 0.3]
    assert [lr_factor(s, cfg) for s in (20, 10**6)] == [0.1, 0.1]
    inside = RecipeConfig(warmup_steps=4, lr_boundaries=(2,), lr_factors=(0.5,))
    assert lr_factor(2, inside) == 0.375  # 3/4 of the warm-up times the drop
    assert all(lr_factor(s, RecipeConfig(warmup_steps=1000)) > 0 for s in range(5))
    with pytest.raises(ValueError):
        lr_factor(-1, cfg)


def test_recipe_validation():
    for bad in (
        dict(token_budget=0),
        dict(pool_rows=0),
        dict(warmup_steps=-1),
        dict(lr_boundaries=(5, 5), lr_factors=(0.5, 0.25)),
        dict(lr_boundaries=(0,), lr_factors=(0.5,)),
        dict(lr_boundaries=(5,), lr_factors=()),
        dict(lr_boundaries=(5,), lr_factors=(0.0,)),
        dict(lr_boundaries=(5,), lr_factors=(1.5,)),
    ):
        with pytest.raises(ValueError, match="I23"):
            RecipeConfig(**bad).validate()
    with pytest.raises(ValueError, match="I23"):
        TrainerConfig(recipe=RecipeConfig(token_budget=0)).validate()
    with pytest.raises(ValueError, match="I23"):
        Trainer(build_model(NET, seed=0), NET, _config(recipe=RecipeConfig(warmup_steps=-1)))
    with pytest.raises(ValueError, match="strictly increasing"):
        assemble_batches([82, 82], [1, 1], 1000, (100, 50))
    with pytest.raises(ValueError):
        assemble_batches([82, 82], [1], 1000, RANGES)
    RecipeConfig(token_budget=1, lr_boundaries=(1, 2), lr_factors=(1.0, 0.5)).validate()


# --- the batches ----------------------------------------------------------------------------------------------


def test_batches_fill_the_token_budget_within_a_range():
    """Hand-computed (A1 tokens, 66 n + 16): five small rows fill a batch (a sixth, of 3,250 tokens,
    would make 6 x 3,250); the two 50-board rows go together and the 99-board row alone; a batch
    never crosses a range; the 240-board row is alone, and the 243-board row - longer than the
    budget - is still a batch, of one."""
    boards = [1, 1, 1, 2, 3, 49, 50, 50, 99, 100, 240, 243]
    tokens = [66 * n + 16 for n in boards]
    assert tokens == [82, 82, 82, 148, 214, 3250, 3316, 3316, 6550, 6616, 15856, 16054]
    assert assemble_batches(tokens, boards, 16_000, RANGES) == [[0, 1, 2, 3, 4], [5], [6, 7], [8], [9], [10], [11]]
    order = [7, 11, 0, 5, 3, 9, 1, 10, 4, 8, 2, 6]  # the same rows, shuffled: the same batches of the original rows
    shuffled = assemble_batches([tokens[i] for i in order], [boards[i] for i in order], 16_000, RANGES)
    assert {frozenset(order[i] for i in b) for b in shuffled} == {
        frozenset(b) for b in [[0, 1, 2, 3, 4], [5], [6, 7], [8], [9], [10], [11]]
    }
    assert assemble_batches([], [], 16_000, RANGES) == []
    for n in range(260):  # the cut's ranges are the labels' (selfplay.board_range)
        lo = (0, *RANGES)[range_index(n, RANGES)]
        assert board_range(n, RANGES) == board_range(lo, RANGES)


def test_every_row_once_one_range_and_never_over_budget():
    """The cut's guarantees on a human-like size distribution (most rows small, a long tail): every
    row exactly once, one range per batch, no batch of two or more over the budget, batches maximal
    - and the bound the budget buys: by section 9.1's model a batch within it costs no more than
    one position of ``token_budget`` tokens."""
    rng = np.random.default_rng(0)
    boards = np.minimum(1 + rng.geometric(1 / 24, size=500), 300).tolist()
    tokens = [66 * n + 16 for n in boards]
    budget = 16_000
    batches = assemble_batches(tokens, boards, budget, RANGES)
    assert sorted(i for b in batches for i in b) == list(range(500))
    full = NetConfig()
    cap = cost.dense_flops(budget, full)["total"]
    for b in batches:
        assert len({range_index(boards[i], RANGES) for i in b}) == 1
        longest = max(tokens[i] for i in b)
        if len(b) > 1:
            assert len(b) * longest <= budget
            assert len(b) * cost.dense_flops(longest, full)["total"] <= cap
    for a, b in zip(batches, batches[1:], strict=False):
        if range_index(boards[a[0]], RANGES) == range_index(boards[b[0]], RANGES):
            assert (len(a) + 1) * tokens[b[0]] > budget  # maximal: the next batch's first row would not have fitted
    assert max(len(b) for b in batches) > 20 and any(len(b) == 1 for b in batches)  # many small rows together, a large one alone


def test_the_pilot_recipe():
    """The pilot's preset: its PLACEHOLDER constants pinned, and section 7.8's own example under it
    - "small positions many to a batch and a 240-board position alone"."""
    PILOT_RECIPE.validate()
    assert (PILOT_TOKEN_BUDGET, PILOT_POOL_ROWS, PILOT_WARMUP_STEPS) == (16_000, 1_024, 1_000)
    assert RecipeConfig(token_budget=16_000, pool_rows=1_024, warmup_steps=1_000) == PILOT_RECIPE
    assert PILOT_RECIPE.lr_boundaries == () and PILOT_RECIPE.lr_factors == ()  # no drops until the run's budget is fixed
    assert assemble_batches([15_856] * 2, [240] * 2, PILOT_TOKEN_BUDGET, RANGES) == [[0], [1]]
    assert [len(b) for b in assemble_batches([2_920] * 12, [44] * 12, PILOT_TOKEN_BUDGET, RANGES)] == [5, 5, 2]
    assert lr_factor(0, PILOT_RECIPE) == 1 / 1000 and lr_factor(999, PILOT_RECIPE) == 1.0 == lr_factor(10**6, PILOT_RECIPE)
    assert TrainerConfig(recipe=PILOT_RECIPE).lr == TrainerConfig().lr == 1e-4  # the base rate is the as-built placeholder


def test_token_budget_batches_through_the_trainer():
    """The arm in the loop, three optimiser steps on synthetic rows. A pool is the window's own
    draw of ``pool_rows`` - the same rows the as-built draw gives - and each is served exactly
    once; every batch lies in one board-count range and collates to rows x its longest member, at
    most the budget unless it is one row; the step log reads what was trained."""
    counts = (1, 1, 2, 2, 3, 3, 5, 6, 8, 9)
    ranges = (3, 6)
    record, states, _ = _synthetic(21, counts)
    cfg = _config(recipe=RecipeConfig(token_budget=1200, pool_rows=24), sampler=SamplerConfig(floor_ranges=ranges))
    assert sorted({row_tokens(s, NET) for s in states}) == sorted({66 * n + 16 for n in counts})  # 82 .. 610 tokens

    drain = Trainer(_model(21), NET, cfg)
    drain.add_game(record, positions=states)
    pool = drain.window.sample(24, np.random.default_rng(cfg.seed))  # the as-built draw of 24 rows
    served = [drain.next_batch()]
    while drain._pending:
        served.append(drain.next_batch())
    assert sorted(id(r) for b in served for r in b) == sorted(id(r) for r in pool)  # every drawn row trained once
    assert len({range_index(row_boards(r), ranges) for r in pool}) == 3 and len(served) > 3

    model = _model(21)
    trainer = Trainer(model, NET, cfg)
    trainer.add_game(record, positions=states)
    for step in range(MAX_STEPS):
        rows = trainer.next_batch()
        assert len({range_index(row_boards(r), ranges) for r in rows}) == 1
        longest = max(row_tokens(r, NET) for r in rows)
        padded = collate_rows(rows, NET)[0].piece.numel()
        assert padded == len(rows) * longest and (padded <= 1200 or len(rows) == 1)
        assert all(math.isfinite(v) for v in trainer.step_on(rows).values())
        entry = trainer.step_log[-1]
        assert (entry["step"], entry["rows"], entry["padded_tokens"]) == (step, len(rows), padded)
        assert entry["tokens"] == sum(row_tokens(r, NET) for r in rows) and entry["boards"] == max(row_boards(r) for r in rows)
        assert entry["lr"] == cfg.lr
    assert optimiser_steps(model) == MAX_STEPS
    with pytest.raises(RuntimeError, match="no training runs"):
        trainer.step()


def test_step_draws_its_batch_from_the_pool():
    """``Trainer.step()`` under a token budget trains ``next_batch()``'s rows: its log matches the
    batch a twin trainer (same seed, same window) is served."""
    record, states, _ = _synthetic(22, (1, 1, 2, 3, 5, 8))
    cfg = _config(recipe=RecipeConfig(token_budget=700, pool_rows=12), sampler=SamplerConfig(floor_ranges=(3, 6)))
    a, b = Trainer(_model(22), NET, cfg), Trainer(_model(22), NET, cfg)
    for t in (a, b):
        t.add_game(record, positions=states)
    want = b.next_batch()
    a.step()
    assert a.step_log[0]["rows"] == len(want) and a.step_log[0]["tokens"] == sum(row_tokens(r, NET) for r in want)
    assert len(a._pending) == len(b._pending)


# --- the rate in the loop ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("optimiser", ["adamw", "nadam"])
def test_the_first_step_scales_with_the_warm_up(optimiser):
    """One step each from the same weights on the same rows: under a four-step warm-up the first
    step runs at a quarter of the rate, and the weights move a quarter as far (AdamW's and NAdam's
    first update is linear in the rate). Norms, with a tolerance: trained weights are not
    bit-reproducible as built (module docstring)."""
    _, _, rows = _synthetic(31)
    moved = {}
    for name, recipe in (("constant", RecipeConfig()), ("warm", RecipeConfig(warmup_steps=4))):
        model = _model(31)
        before = [p.detach().clone() for p in model.parameters()]
        trainer = Trainer(model, NET, _config(optimiser=optimiser, recipe=recipe))
        trainer.step_on(rows)
        moved[name] = _change(model, before)
        assert trainer.step_log[0]["lr"] == trainer.cfg.lr * (0.25 if name == "warm" else 1.0)
    assert moved["constant"] > 0
    assert moved["warm"] / moved["constant"] == pytest.approx(0.25, abs=1e-3)


def test_the_rate_follows_the_schedule_over_three_steps():
    """A two-step warm-up and a drop at step 2, in the loop: the rates 0.5, 1 and 0.1 of ``lr``, the
    optimiser left at the last; SWA's cadence, the losses' finiteness and the synthetic-rows rule
    untouched by the recipe."""
    _, _, rows = _synthetic(32)
    cfg = _config(lr=1e-3, recipe=RecipeConfig(warmup_steps=2, lr_boundaries=(2,), lr_factors=(0.1,)))
    model = _model(32)
    trainer = Trainer(model, NET, cfg)
    with pytest.raises(RuntimeError, match="synthetic rows only"):
        trainer.step_on([replace(r, synthetic=False) for r in rows])
    assert trainer.step_log == [] and optimiser_steps(model) == 0
    for _ in range(MAX_STEPS):
        assert all(math.isfinite(v) for v in trainer.step_on(rows).values())
    assert [e["lr"] for e in trainer.step_log] == pytest.approx([5e-4, 1e-3, 1e-4], rel=1e-12)
    assert trainer.opt.param_groups[0]["lr"] == trainer.step_log[-1]["lr"]
    assert trainer.swa.snapshots == MAX_STEPS


# --- a piece-centric network ---------------------------------------------------------------------------------


def test_a_piece_centric_network_budgets_its_own_tokens():
    """Register A5: the budget counts the tokens the collated batch holds - a piece-centric
    network's own, fewer than A1's - so a batch cut at them collates to rows x its longest member.
    Forward only."""
    net_a5 = NetConfig(**SMALL, piece_centric=True)
    record, states, rows = _synthetic(41, (1, 2, 2, 3, 4, 6))
    for r in rows:
        own = row_tokens(r, net_a5)
        assert own == to_piece_centric(r.state, strict=False).n_tokens < row_tokens(r, NET) == r.state.n_tokens
    tokens = [row_tokens(r, net_a5) for r in rows]
    budget = 2 * max(tokens)
    batches = assemble_batches(tokens, [row_boards(r) for r in rows], budget, RANGES)
    assert any(len(b) > 1 for b in batches)
    trainer = Trainer(build_model(net_a5, seed=41), net_a5, _config())
    for b in batches:
        chosen = [rows[i] for i in b]
        padded = collate_rows(chosen, net_a5)[0].piece.numel()
        assert padded == len(b) * max(tokens[i] for i in b) and (padded <= budget or len(b) == 1)
        assert all(math.isfinite(v) for v in trainer.losses_on(chosen).values())
    assert trainer.steps == 0 and torch.is_grad_enabled()
