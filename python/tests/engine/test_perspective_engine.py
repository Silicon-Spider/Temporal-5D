"""Register A7 on real engine positions: the engine's own colour-swapped mirror converts to the
network's ``state.mirror`` of the position, and the network reads both alike.

Section 15.2's A7 row ("A position and its colour-swapped mirror ... give identical canonical
inputs and outputs") is run on synthetic states in ``tests/net/test_perspective.py``. Here the
mirror is the engine's: a position's 5DFEN board strings mirrored by the notation alone - each
board's rows in reverse order with every piece letter's case swapped, its timeline label
negated (``+n <-> -n``, ``-0 <-> +0``; ``0`` stays) and its turn one half-turn later (White's
turn T becomes Black's turn T, Black's becomes White's T+1) - set up by the engine
(``Game.from_fen``), expanded and converted. The original is set up from its own strings too,
so both carry the same lineage (5DFEN records no branch origins). The engine computes the
mirror's moves, move graph and typed relations itself; they must be the mirror of the
original's - the rules are colour-symmetric but for who moves first, which the mirror keeps as
the acting colour - and the network's outputs on the two, with the acting-colour input
equalised, must agree mapped back. Positions: every variant's start and random play (branched
and inactive timelines, both acting colours), the records' positions, and positions past the
shift inside an unsubmitted action (mirrored by ``state.mirror`` alone: 5DFEN sets up turn
starts only).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.engine import Game, to_position_state
from t5d.engine.play import random_game
from t5d.net import NetConfig, build_model, collate, mirror, permute_boards
from t5d.net.state import OWNER_OURS, mirror_tokens
from tests.engine.helpers import RECORDS, VARIANTS, requires_engine
from tests.net.test_perspective import VALUE_HEADS, assert_mirrored, tie

pytestmark = requires_engine

RELATIONS = ("attack", "defence", "xray")


def mirror_board_string(text: str) -> str:
    """A 5DFEN board string's colour-swapped mirror (the module docstring)."""
    rows, timeline, turn, colour = text[1:-1].split(":")
    rows = "/".join(row.swapcase() for row in reversed(rows.split("/")))
    if timeline[:1] in "+-":
        timeline = {"+": "-", "-": "+"}[timeline[0]] + timeline[1:]
    h = 2 * int(turn) + (colour == "b") + 1
    return f"[{rows}:{timeline}:{h // 2}:{'wb'[h & 1]}]"


def _games() -> list[tuple[str, Game]]:
    out = []
    for name in VARIANTS:
        out.append((f"{name} start", Game.variant(name)))
        for i, game in enumerate(random_game(name, 3, target_boards=40, max_actions=24)):
            if i % 3 == 1:
                out.append((f"{name} action {i + 1}", game.copy()))
    for key, text in RECORDS.items():
        game = Game.from_pgn(text)
        for n in game.replay():
            out.append((f"{key} position {n}", game.copy()))
    return out


@pytest.fixture(scope="module")
def pairs():
    """``(label, original, the engine's mirror, order)``: ``order[i]`` is the engine's index of
    ``state.mirror``'s board ``i`` (the engine stores boards by (timeline, half-turn), and
    negated timelines reverse that order)."""
    out = []
    for label, game in _games():
        with game:
            snap = game.snapshot()
            parity = "even" if snap.even else "odd"
            strings = [fen for _, _, fen in snap.boards]
        with Game.from_fen(strings, parity=parity) as g:
            s = to_position_state(g.expand(relations=RELATIONS))
        with Game.from_fen([mirror_board_string(t) for t in strings], parity=parity) as g:
            e = to_position_state(g.expand(relations=RELATIONS))
        key = {(int(p), int(h)): i for i, (p, h) in enumerate(zip(e.timeline, e.half_turn, strict=True))}
        m = mirror(s)
        order = [key[(int(p), int(h))] for p, h in zip(m.timeline, m.half_turn, strict=True)]
        out.append((label, s, e, order))
    return out


def _rows(arr: np.ndarray) -> list[tuple[int, ...]]:
    return sorted(map(tuple, arr.tolist()))


def test_the_engines_mirror_is_the_mirror(pairs):
    """Board by board, square by square, move by move: the engine's conversion of the mirrored
    5DFEN equals ``state.mirror`` of the original's conversion - for both acting colours, odd and
    even variants, every board size."""
    assert {int(s.acting_player) for _, s, _, _ in pairs} == {0, 1}
    assert any(s.even_start for _, s, _, _ in pairs) and max(s.n_boards for _, s, _, _ in pairs) >= 30
    for label, s, engine, order in pairs:
        m, e = mirror(s), permute_boards(engine, order)
        for name in ("timeline", "half_turn", "is_head", "active", "side_to_move", "piece", "colour", "moved", "parent", "source"):
            assert np.array_equal(getattr(m, name), getattr(e, name)), (label, name)
        for name in ("acting_player", "submit_possible", "submit_legal", "even_start", "width", "height"):
            assert getattr(m, name) == getattr(e, name), (label, name)
        for name in ("moves", "edges"):
            assert _rows(getattr(m, name)) == _rows(getattr(e, name)), (label, name)  # the engine's order is its own
        assert e.moves.shape[0] == s.moves.shape[0] and e.moves.shape[0] > 0, label


def test_the_canonical_view_of_a_real_position(pairs):
    """When Black acts, Black's pieces are ours, its back rank is the canonical first, and the
    timelines are negated (``-0 <-> +0`` in an even variant)."""
    seen = 0
    for label, s, _, _ in pairs:
        if s.acting_player != 1:
            continue
        seen += 1
        tok = s.token_arrays()
        wh, tpb = s.width * s.height, s.width * s.height + s.regs_per_board
        board, square = np.divmod(np.arange(s.n_boards * tpb), tpb)
        sq = square < wh
        colour = s.colour[board[sq], square[sq]]
        assert np.array_equal(tok["owner"][: s.n_boards * tpb][sq] == OWNER_OURS, colour == 2), label
        assert np.array_equal(tok["y"][: s.n_boards * tpb][sq], s.height - 1 - square[sq] // s.width), label
        p = s.timeline[board]
        assert np.array_equal(tok["p"][: s.n_boards * tpb], -1 - p if s.even_start else -p), label
    assert seen >= 10


def _engine_row(row: tuple[int, int, int], new_board: np.ndarray, tpb: int) -> tuple[int, int, int]:
    """A move row of ``state.mirror``'s layout in the engine's board order."""
    src, dst, promo = row
    return int(new_board[src // tpb] * tpb + src % tpb), int(new_board[dst // tpb] * tpb + dst % tpb), promo


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
def test_real_positions_and_their_mirrors_give_the_same_outputs(pairs, scope):
    """The network (random weights, the acting-colour rows tied) on the original and on the
    engine's mirror: every global head equal; each move's logit equal to its mirrored move's
    (found in the engine's own move list); the per-board logits in the engine's board order;
    the per-square logits at the mirrored squares - B11's typed biases on, so the engine's
    relations are read."""
    model = tie(build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope=scope, typed_edge_bias=True), seed=11).eval())
    for label, s, e, order in pairs[::2]:
        with torch.no_grad():
            a, b = model(collate([s])), model(collate([e]))
        for name in VALUE_HEADS:
            assert torch.allclose(getattr(a, name), getattr(b, name), atol=1e-5), (label, name)
        m = mirror(s)
        tpb = s.width * s.height + s.regs_per_board
        where = {row: i for i, row in enumerate(map(tuple, e.moves.tolist()))}
        new_board = np.asarray(order)  # the mirror's board -> the engine's

        index = [where[_engine_row(row, new_board, tpb)] for row in map(tuple, m.moves.tolist())]
        assert torch.allclose(a.policy_logits[0], b.policy_logits[0, index], atol=1e-5), label
        assert torch.allclose(a.board_logits[0], b.board_logits[0, new_board], atol=1e-5), label
        tokens = torch.from_numpy(mirror_tokens(s.layout))
        engine_tokens = torch.from_numpy(np.concatenate([np.arange(tpb) + tpb * int(new_board[b]) for b in range(s.n_boards)]))
        mask = a.square_mask[0]
        sq_b = b.square_logits[0][engine_tokens[tokens[: s.n_boards * tpb]]]
        assert torch.allclose(a.square_logits[0][: s.n_boards * tpb][mask[: s.n_boards * tpb]], sq_b[mask[: s.n_boards * tpb]], atol=1e-5)


def test_past_the_shift_on_real_positions():
    """Inside an unsubmitted action whose present has shifted (the acting player is the
    present's opponent, submit offered): the position and its mirror agree, mapped back, under
    both scopes - the acting player's frame, not the present's."""
    count = 0
    for name in ("Standard", "Standard - Two Timelines", "Small", "Misc - Timeline Battleground"):
        for i, game in enumerate(random_game(name, 5, target_boards=30, max_actions=12)):
            if i % 3:
                continue
            with game.copy() as g:
                for move in g.moves():
                    trial = g.copy()
                    trial.apply(move)
                    s = to_position_state(trial.expand(relations=RELATIONS))
                    trial.close()
                    if s.shifted:
                        break
                else:
                    continue
            count += 1
            perm = torch.from_numpy(mirror_tokens(s.layout))
            for scope in ("dense", "block_causal"):
                model = tie(build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope=scope), seed=count).eval())
                with torch.no_grad():
                    a, b = model(collate([s])), model(collate([mirror(s)]))
                assert_mirrored(a, b, perm, (int(s.acting_player), 1 - int(s.acting_player)))
            assert s.submit_possible
    assert count >= 6
