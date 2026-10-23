"""Register A6 on real engine positions: section 15.2's row on the network side.

    "Turn-state counts = a brute-force recount from the rules engine's obligations | The new
    inputs are exact | A6"

The engine's own side of the row is ``engine/tests/test_turn_state.cpp`` (a recount from the
rulebook at every state of random play under every ruleset). Here the network's input: at every
state of seeded random play - turn starts, mid-action states before the shift, states past it
with moves still allowed (R8) or none (Q4's Guide-literal reading), jumps between two present
boards, rewinding branches, T12's reactivations that raise the count (spec issue 143) - the
converted ``PositionState``'s counts are the expansion's, the moves made are the handle's count
of the moves played since the last submission, and the mandatory and optional counts equal a
recount written here from the rulebook's definitions on the converted boards. A turn start and
a mid-turn state with every board and the acting player equal - set up from the mid-turn
state's own 5DFEN, which the engine sets up as a turn start - differ in the moves made alone,
and the network tells them apart. Synthetic states, the embedding and the caches:
``tests/net/test_turn_state.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from t5d.engine import Game, to_position
from t5d.engine.capi import T5D_MOVE_BRANCHING, T5D_MOVE_TRAVEL, T5D_RULES_NO_MOVES_AFTER_SHIFT
from t5d.engine.play import ROYAL_IDS
from t5d.net import NetConfig, TurnState, build_model, collate
from t5d.net.cache import CachedEvaluator, reference_output, state_digest
from tests.engine.helpers import requires_engine

pytestmark = requires_engine

#: (variant, seed, travel probability) - travel-heavy play reaches branches, jumps and rewinds.
PLAYS = (
    ("Standard", 0, 0.5),
    ("Standard - Two Timelines", 1, 0.6),
    ("Standard - Two Timelines", 2, 0.9),
    ("Very Small - Open", 3, 0.7),
    ("Misc - Timeline Battleground", 4, 0.5),
    ("Small", 5, 0.8),
    ("Standard", 10, 0.7),
    ("Very Small - Open", 17, 0.7),
)


def recount(s, present: int, may_move: bool) -> TurnState:
    """The counts from the rulebook's definitions, board by board (default rules, InPresent): a
    board is playable when it is a head with the acting player to move (L5); mandatory when it is
    also an active timeline's head in the present (T6); optional when playable and not mandatory,
    while a move may be made (Q4). The moves made are not the boards' (the caller's count)."""
    mandatory = optional = 0
    for b in range(s.n_boards):
        if not (s.is_head[b] and int(s.side_to_move[b]) == int(s.acting_player)):
            continue
        if s.active[b] and int(s.half_turn[b]) == present:
            mandatory += 1
        elif may_move:
            optional += 1
    return TurnState(mandatory, optional, -1)


def _play(variant: str, seed: int, travel: float, rules: int, tally: dict, states: int = 200, max_boards: int = 80) -> None:
    """Seeded random play, every state converted and checked (the module docstring)."""
    rng = np.random.default_rng(seed)
    with Game.variant(variant, rules=rules) as g:
        g.set_legality(False)
        made = 0
        for _ in range(states):
            info = g.info()
            x = g.expand()
            s = to_position(x).state
            counts = s.turn_state
            assert counts == (x.mandatory_remaining, x.optional_playable, x.moves_made), (variant, made)
            assert counts.moves_made == made == info.moves_made, (variant, made)
            again = recount(s, x.present, x.may_move)
            assert counts[:2] == again[:2], (variant, made, counts, again)
            assert (counts.mandatory_remaining == 0) == s.submit_possible == s.shifted
            if rules == 0:
                assert counts[:2] == s.derived_turn_state[:2]  # the state's own derivation is the default rules'
            tally["turn starts"] += made == 0
            tally["mid-action"] += made > 0 and not s.shifted
            tally["past the shift, moves allowed"] += s.shifted and x.may_move
            tally["past the shift, no move"] += s.shifted and not x.may_move
            tally["optional boards"] += counts.optional_playable > 0
            if info.boards >= max_boards:
                return
            ok = np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS)) if x.move_count else np.zeros(0, dtype=np.int64)
            if s.shifted and (not ok.size or rng.random() < 0.6):
                g.submit()
                made = 0
                continue
            if not ok.size:
                return
            travel_moves = ok[(x.moves["flags"][ok] & T5D_MOVE_TRAVEL) != 0]
            pool = travel_moves if travel_moves.size and rng.random() < travel else ok
            i = int(rng.choice(pool))
            move = x.moves[i]
            before = counts.mandatory_remaining
            g.apply(i)
            made += 1
            after = g.expand(square_features="none").mandatory_remaining
            if before - after == 2:
                tally["a jump between two present boards (-2)"] += 1
            if move["flags"] & T5D_MOVE_BRANCHING and after == 0 and before >= 2:
                tally["a rewinding branch releasing a board (-> 0)"] += 1
            if after > before:
                tally["a rise (T12)"] += 1


def test_the_converted_counts_are_the_engines_and_a_recount():
    tally = dict.fromkeys(
        (
            "turn starts",
            "mid-action",
            "past the shift, moves allowed",
            "past the shift, no move",
            "optional boards",
            "a jump between two present boards (-2)",
            "a rewinding branch releasing a board (-> 0)",
            "a rise (T12)",
        ),
        0,
    )
    for rules in (0, T5D_RULES_NO_MOVES_AFTER_SHIFT):
        for variant, seed, travel in PLAYS:
            _play(variant, seed, travel, rules, tally)
    for name in (
        "turn starts",
        "mid-action",
        "past the shift, moves allowed",
        "past the shift, no move",
        "optional boards",
        "a jump between two present boards (-2)",
        "a rewinding branch releasing a board (-> 0)",
        "a rise (T12)",
    ):
        assert tally[name] > 0, (name, tally)


def _mid_turn_and_its_turn_start():
    """Two Timelines after White's (-0T1)e4: mid-action, +0 still owed - and the same boards set
    up from their own 5DFEN, a turn start (every set-up is one, T5)."""
    with Game.variant("Standard - Two Timelines") as g:
        g.apply(next(m for m in g.moves() if m.text(even=True) == "(-0T1)Pe2(-0T1)e4"))
        mid = g.position()
        strings = [fen for _, _, fen in g.snapshot().boards]
    with Game.from_fen(strings, parity="even") as g:
        start = g.position()
    return mid.state, start.state


def test_a_turn_start_and_a_mid_turn_state_with_equal_boards():
    """Section 2.3: moves made "also tells a turn start from a mid-turn state with the same
    boards, which the network could not do before" (CHECKPOINT 6). The two converted states
    share every board, the acting player, the moves and both other counts; they differ in the
    moves made alone, and so do their digests and every output - the full pass and the cached
    path alike, in one store."""
    mid, start = _mid_turn_and_its_turn_start()
    assert mid.turn_state == TurnState(1, 0, 1) and start.turn_state == TurnState(1, 0, 0)
    for name in ("timeline", "half_turn", "is_head", "active", "piece", "colour", "moved", "moves", "edges"):
        assert np.array_equal(getattr(mid, name), getattr(start, name)), name
    assert (mid.acting_player, mid.submit_possible) == (start.acting_player, start.submit_possible) == (0, False)
    assert state_digest(mid) != state_digest(start)
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=0).eval()
    a, b = model(collate([mid])), model(collate([start]))
    assert not torch.allclose(a.value_logits, b.value_logits)
    assert not torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid])
    block = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal"), seed=0).eval()
    cached = CachedEvaluator(block)
    epoch = cached.root_epoch(start)
    for s in (mid, start):
        got = cached.evaluate(s, epoch)
        ref = reference_output(block, s)
        assert torch.allclose(got.trunk, ref.trunk, atol=2e-5) and torch.allclose(got.value_logits, ref.value_logits, atol=1e-4)
    assert not torch.allclose(cached.evaluate(mid, epoch).value_logits, cached.evaluate(start, epoch).value_logits)


def test_an_inconsistent_turn_state_is_refused():
    """What the converter checks of the counts (``PresentMismatch`` unless ``strict=False``):
    the expansion's own definitions - non-negative, the mandatory and optional boards exactly
    the boards it marks playable, the mandatory ones on active timelines, nothing owed exactly
    when submitting is possible. Which playable boards are owed is the ruleset's, which the
    expansion does not name: a different split of the same boards converts (the engine's count
    is the truth, section 2.3; the recount above is the test of it)."""
    from dataclasses import replace

    import pytest

    from t5d.engine import PresentMismatch

    with Game.variant("Standard - Two Timelines") as g:
        start = g.expand()
    assert (start.mandatory_remaining, start.optional_playable, start.moves_made) == (2, 0, 0)
    for bad in (
        replace(start, moves_made=-1),
        replace(start, optional_playable=1),
        replace(start, mandatory_remaining=0, optional_playable=2),
        replace(start, active=np.array([True, False])),  # +0 inactive: it owes nothing
    ):
        with pytest.raises(PresentMismatch):
            to_position(bad)
    loose = to_position(replace(start, mandatory_remaining=0, optional_playable=2), strict=False)
    assert not loose.present_matches and loose.state.turn_state == TurnState(0, 2, 0)
    split = to_position(replace(start, mandatory_remaining=1, optional_playable=1))
    assert split.present_matches and split.state.turn_state == TurnState(1, 1, 0)
