"""Seeded random play through the binding: real engine positions of any size, for tests and
the benchmark.

Like the engine's own synthetic positions (engine README, "Synthetic positions"), each move
is drawn among the engine's pseudo-legal moves - never one that takes a royal piece (4.13)
- until the present has shifted (T5); then the action is submitted, pseudo-legally (the
game is set to ``set_legality(False)``: random play leaves royals capturable). Every rule
is the engine's; this module only chooses. ``travel_prob`` steers growth: with that
probability the move is drawn among the travel moves (when there are both kinds), else
among the rest; None draws uniformly among all.

**Random play is for exercising shapes and costs, never evidence about real games** (the
engine README says the same of its synthetic positions).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..net.state import PIECE_IDS
from .capi import T5D_MOVE_TRAVEL
from .game import Game

ROYAL_IDS = (PIECE_IDS["K"], PIECE_IDS["Y"])


def random_action(game: Game, rng: np.random.Generator, *, travel_prob: float | None = None) -> int:
    """Play one random action and submit it; the number of moves played, or -1 when stuck."""
    played = 0
    while True:
        info = game.info()
        if info.obligations_met:
            game.submit()
            return played
        if not info.may_move:
            return -1
        x = game.expand(square_features="none")
        ok = ~np.isin(x.moves["captured"], ROYAL_IDS)
        if not ok.any():
            return -1
        pool = np.flatnonzero(ok)
        if travel_prob is not None:
            travel = (x.moves["flags"][pool] & T5D_MOVE_TRAVEL) != 0
            if travel.any() and (~travel).any():
                pool = pool[travel] if rng.random() < travel_prob else pool[~travel]
        game.apply(int(rng.choice(pool)))
        played += 1


def random_game(
    variant: str = "Standard",
    seed: int = 0,
    *,
    target_boards: int | None = None,
    max_actions: int = 1000,
    travel_prob: float | None = None,
) -> Iterator[Game]:
    """Play from a variant's start, yielding the game after each action (the same handle,
    advanced in place - copy it to keep a position) until ``target_boards`` boards are
    reached, ``max_actions`` actions are played, or the game is stuck."""
    rng = np.random.default_rng(seed)
    game = Game.variant(variant)
    game.set_legality(False)
    for _ in range(max_actions):
        if random_action(game, rng, travel_prob=travel_prob) < 0:
            return
        yield game
        if target_boards is not None and game.info().boards >= target_boards:
            return


def random_position_game(
    variant: str = "Standard", seed: int = 0, *, boards: int, travel_prob: float | None = None, max_actions: int = 2000
) -> Game:
    """A copy of the first position of seeded random play with at least ``boards`` boards
    (``ValueError`` if the game ends first)."""
    last = None
    for game in random_game(variant, seed, target_boards=boards, max_actions=max_actions, travel_prob=travel_prob):
        last = game
    if last is None or last.info().boards < boards:
        raise ValueError(f"random play from {variant!r} (seed {seed}) stopped before {boards} boards")
    return last.copy()


__all__ = ["ROYAL_IDS", "random_action", "random_game", "random_position_game"]
