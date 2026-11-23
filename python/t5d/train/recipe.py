"""Register I23's optimisation recipe for the pilot run: batches assembled to a token budget within
a board-count range, a learning-rate warm-up and a schedule (spec sections 12.8 and 7.8; register
I23, G12; the 2026-11-15 review). OFF by default: ``RecipeConfig()`` is the smoke test's recipe -
batches of a fixed count at a constant rate - and the pilot's is the named preset ``PILOT_RECIPE``,
every constant of it a PLACEHOLDER.

"The pilot run needs two things the smoke test did not. Batches are assembled to a *token budget*
within a board-count range, because a fixed count of positions varies a hundredfold in cost and
memory (section 7.8). And the learning rate has a warm-up and a schedule: a pre-norm transformer
trained from random weights on a self-play distribution that is itself moving wants both.
Retrofittable, and tuned on the pilot run" (section 12.8). "So the batch is sized by *work*, not by
a count - each call filled to a token budget, small positions many to a batch and a 240-board
position alone ... A batch holds positions of similar size. A padded batch pays for its longest
member; bucketing by token count ... keeps the padding small" (section 7.8).

* **The budget counts padded tokens** - a batch's rows times its longest member's tokens
  (``row_tokens``: the tokens the collated batch holds for a row under the trainer's network,
  registers included; register A5's count for a piece-centric network) - the unit the documents
  name each time they name one (sections 12.8 and 7.8, register G12 and I23). Attention is
  quadratic, so equal tokens are not equal work; what the budget does give is a bound: a batch
  within it costs, by section 9.1's model, no more than one position of ``token_budget`` tokens.
* **A pool, then a cut** (``assemble_batches``). The trainer draws ``pool_rows`` rows from the
  replay window exactly as it draws a fixed batch (``Trainer.draw_rows``: uniformly, or by register
  I22's floors when they are on), and the pool is cut into batches: sorted by board-count range and
  token count, filled until the next row would pass the budget, never across a range. So every row
  drawn is trained once: row frequencies are the window's own (I12's surprise copies, I16's window,
  I22's floors where on), and batching changes how rows are grouped into steps, never which rows
  are trained. Sorting the pool is how "positions of similar size" (G12) is met; a row longer than
  the budget is a batch of one - never dropped, never truncated ("a 240-board position alone").
* **The schedule** (``lr_factor``): a linear warm-up into piecewise-constant steps,
  ``lr x min(1, (s + 1) / warmup_steps) x`` the factor of the last boundary passed. Step drops need
  no horizon, and the pilot's length is not fixed.

**Consequences of the ruling, as built** (the pilot's to tune): one batch is one optimiser step and
every loss is a mean over the batch's positions, so a step holds anywhere from one row to a couple
of hundred and a row's weight in its step varies with its size; nothing is accumulated or rescaled.
Each batch has its own (rows, length) shape.

**Not built, because no document names it**: gradient clipping, a weight-decay change (it stays
``TrainerConfig.weight_decay``, 0), gradient accumulation, a per-row rescale of the loss, cosine or
inverse-square-root shapes, a budget in the cost model's units, any change to SWA (I17).

Written from the methods' descriptions - batching sentences of similar length to a token count
(Vaswani et al. 2017, "Attention Is All You Need", section 5.1), a linear warm-up, step drops
(Silver et al. 2018, "A general reinforcement learning algorithm that masters chess, shogi, and Go
through self-play") - with no code from Lc0, KataGo or any engine or library (AGENTS.md, H29).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

from ..net.cost import token_counts

#: The pilot's recipe - PLACEHOLDERS, none measured and none given by a document (register I23:
#: "Committed (placeholders)", tuned on the pilot run).
#: The budget: the smallest round figure under which section 7.8's own example holds - a 240-board
#: position (240 x 66 + 16 = 15,856 tokens) goes alone, five 44-board positions (2,920 each) go
#: together. The figure that fits is set by the rented card's memory and Gate 1's batching curve.
#: It is the training budget, a setting of its own beside inference's (``engine.pool.TOKEN_BUDGET``,
#: 65,536): a training step keeps every layer's activations for the backward pass, a forward does
#: not (ruled 2026-11-23, BUILD_LOG spec issue 724).
PILOT_TOKEN_BUDGET = 16_000
#: Rows drawn from the window per pool - PLACEHOLDER (no documentary basis).
PILOT_POOL_ROWS = 1_024
#: The linear warm-up's length in optimiser steps - PLACEHOLDER (no documentary basis).
PILOT_WARMUP_STEPS = 1_000


@dataclass(frozen=True)
class RecipeConfig:
    """Register I23's recipe (module docstring). The default is the smoke test's: no budget, no
    warm-up, no drops."""

    #: None: batches of ``TrainerConfig.batch_size``, as built. Else the most padded tokens - rows
    #: times the longest row's tokens - a batch of two or more rows may hold (``batch_size`` is
    #: then not read).
    token_budget: int | None = None
    #: Rows drawn from the window per pool when ``token_budget`` is set - PLACEHOLDER.
    pool_rows: int = PILOT_POOL_ROWS
    #: The linear warm-up's length in optimiser steps (0: none, as built).
    warmup_steps: int = 0
    #: The 0-based step indices from which ``lr_factors[i]`` applies (strictly increasing, >= 1).
    lr_boundaries: tuple[int, ...] = ()
    #: Multiples of ``TrainerConfig.lr`` in (0, 1], one per boundary - not cumulative.
    lr_factors: tuple[float, ...] = ()

    def validate(self) -> None:
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("I23: token_budget >= 1 (or None: batches of a fixed count)")
        if self.pool_rows < 1 or self.warmup_steps < 0:
            raise ValueError("I23: pool_rows >= 1, warmup_steps >= 0")
        bounds, factors = tuple(self.lr_boundaries), tuple(self.lr_factors)
        if len(bounds) != len(factors):
            raise ValueError("I23: one learning-rate factor per boundary")
        if any(b < 1 for b in bounds) or any(b <= a for a, b in zip(bounds, bounds[1:], strict=False)):
            raise ValueError("I23: the schedule's boundaries are step indices >= 1, strictly increasing")
        if any(not 0.0 < f <= 1.0 for f in factors):
            raise ValueError("I23: each learning-rate factor is in (0, 1]")


#: The pilot run's recipe, a named preset (``TrainerConfig(recipe=PILOT_RECIPE)``): every value a
#: PLACEHOLDER. The base rate stays ``TrainerConfig.lr`` (1e-4, spec issue 237(b)); no drops until
#: the run's step budget is fixed (the mechanism is built: ``lr_boundaries``, ``lr_factors``).
PILOT_RECIPE = RecipeConfig(token_budget=PILOT_TOKEN_BUDGET, pool_rows=PILOT_POOL_ROWS, warmup_steps=PILOT_WARMUP_STEPS)


def lr_factor(step: int, cfg: RecipeConfig) -> float:
    """The multiple of ``TrainerConfig.lr`` that optimiser step ``step`` runs at (0-based: the index
    of the step about to be taken): the warm-up's ``min(1, (step + 1) / warmup_steps)`` - never 0,
    and step ``warmup_steps - 1`` is the first at the full rate - times the factor of the last
    boundary passed (a boundary inside the warm-up multiplies). Exactly 1.0 at every step for
    ``RecipeConfig()``."""
    if step < 0:
        raise ValueError("the step index is >= 0")
    warm = min(1.0, (step + 1) / cfg.warmup_steps) if cfg.warmup_steps else 1.0
    passed = sum(step >= b for b in cfg.lr_boundaries)
    return warm * (float(cfg.lr_factors[passed - 1]) if passed else 1.0)


def range_index(n_boards: int, ranges: Sequence[int]) -> int:
    """The index of the board-count range ``n_boards`` falls in (``ranges`` the lower bounds after
    the first, as ``selfplay.board_range``'s): with (50, 100, 200), 49 -> 0, 50 -> 1, 200 -> 3."""
    return bisect.bisect_right(tuple(ranges), n_boards)


def _state(row):
    """A ``targets.Row``'s, a ``convert.Position``'s or a ``PositionState``'s own state."""
    for _ in range(2):
        row = getattr(row, "state", row)
    return row


def row_tokens(row, net_cfg) -> int:
    """The tokens the collated batch holds for ``row`` (a ``targets.Row``, a position or a state)
    under ``net_cfg``'s tokenisation, registers included: the state's own count, or register A5's
    for a piece-centric network, whose inputs are converted at the network's edge
    (``net.cost.token_counts`` - what ``state.to_piece_centric`` keeps). The one count the token
    budget (I23) and the floors' training compute (I22, ``sampler.row_cost``) both read."""
    state = _state(row)
    return int(token_counts(state)["a5"] if getattr(net_cfg, "piece_centric", False) else state.n_tokens)


def row_boards(row) -> int:
    """``row``'s board count."""
    return int(_state(row).n_boards)


def assemble_batches(tokens: Sequence[int], boards: Sequence[int], token_budget: int, ranges: Sequence[int]) -> list[list[int]]:
    """Cut rows into batches under ``token_budget`` padded tokens, each within one board-count range
    (module docstring): ``tokens[i]`` and ``boards[i]`` are row ``i``'s; the result is a list of
    batches of row indices. Pure and deterministic. Rows are walked in order of (range, tokens,
    index); a batch closes when the next row is of another range or would take its padded size -
    rows times the longest member, which ascending order makes the newest - past the budget.

    Guarantees: every index exactly once; one range per batch; ``len(batch) x max(tokens)`` at
    most the budget for every batch of two or more; a row longer than the budget alone; batches
    maximal (the next batch's first row would not have fitted). Returned in sorted order - the
    caller shuffles."""
    ranges = tuple(ranges)
    if any(b <= a for a, b in zip(ranges, ranges[1:], strict=False)):
        raise ValueError("the board-count ranges' bounds are strictly increasing")
    if token_budget < 1 or len(tokens) != len(boards):
        raise ValueError("token_budget >= 1, and one token count and one board count per row")
    index = [range_index(int(b), ranges) for b in boards]
    order = sorted(range(len(tokens)), key=lambda i: (index[i], int(tokens[i]), i))
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (index[i] != index[cur[0]] or (len(cur) + 1) * int(tokens[i]) > token_budget):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


__all__ = [
    "PILOT_POOL_ROWS",
    "PILOT_RECIPE",
    "PILOT_TOKEN_BUDGET",
    "PILOT_WARMUP_STEPS",
    "RecipeConfig",
    "assemble_batches",
    "lr_factor",
    "range_index",
    "row_boards",
    "row_tokens",
]
