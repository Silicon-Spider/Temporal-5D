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
equalised, must agree mapped back. Positions: the start of every named variant the engine sets
up (its whole table, ``helpers.ENGINE_VARIANTS``) and random play in each (branched and inactive
timelines, both acting colours), the records' positions, and positions past the
shift inside an unsubmitted action (mirrored by ``state.mirror`` alone: 5DFEN sets up turn
starts only).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from t5d.engine import Game, to_position_state
from t5d.engine.capi import T5D_LABEL_UNKNOWN
from t5d.engine.play import random_game
from t5d.net import NetConfig, build_model, collate, mirror, permute_boards
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, viewed_by
from t5d.net.heads import SQUARE_LABELS
from t5d.net.state import OWNER_OURS, RELATION_IDS, mirror_tokens
from tests.engine.helpers import ENGINE_VARIANTS, RECORDS, requires_engine
from tests.net.test_perspective import (
    VALUE_HEADS,
    _close,
    assert_mirrored,
    assert_other_perspective_anchor,
    dense_passes,
    foreign_sources,
    tie,
)

pytestmark = requires_engine

RELATIONS = ("attack", "defence", "pin", "xray", "check")


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
    for name in ENGINE_VARIANTS:
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
def fens():
    """``(label, parity, 5DFEN board strings)`` of every position (``_games``)."""
    out = []
    for label, game in _games():
        with game:
            snap = game.snapshot()
            out.append((label, "even" if snap.even else "odd", [fen for _, _, fen in snap.boards]))
    return out


def _mirror_order(s, e) -> list[int]:
    """The engine's index of ``state.mirror(s)``'s board ``i`` in ``e``, the engine's mirror."""
    key = {(int(p), int(h)): i for i, (p, h) in enumerate(zip(e.timeline, e.half_turn, strict=True))}
    m = mirror(s)
    return [key[(int(p), int(h))] for p, h in zip(m.timeline, m.half_turn, strict=True)]


@pytest.fixture(scope="module")
def pairs(fens):
    """``(label, original, the engine's mirror, order)``: ``order[i]`` is the engine's index of
    ``state.mirror``'s board ``i`` (the engine stores boards by (timeline, half-turn), and
    negated timelines reverse that order)."""
    out = []
    for label, parity, strings in fens:
        with Game.from_fen(strings, parity=parity) as g:
            s = to_position_state(g.expand(relations=RELATIONS))
        with Game.from_fen([mirror_board_string(t) for t in strings], parity=parity) as g:
            e = to_position_state(g.expand(relations=RELATIONS))
        out.append((label, s, e, _mirror_order(s, e)))
    return out


def _rows(arr: np.ndarray) -> list[tuple[int, ...]]:
    return sorted(map(tuple, arr.tolist()))


def test_the_engines_mirror_is_the_mirror(pairs):
    """Board by board, square by square, move by move: the engine's conversion of the mirrored
    5DFEN equals ``state.mirror`` of the original's conversion - for both acting colours, odd and
    even variants, every board size - and the engine's turn-state counts (register A6: the acting
    player's, which the mirror keeps) are the original's."""
    assert {int(s.acting_player) for _, s, _, _ in pairs} == {0, 1}
    assert any(s.even_start for _, s, _, _ in pairs) and max(s.n_boards for _, s, _, _ in pairs) >= 30
    for label, s, engine, order in pairs:
        m, e = mirror(s), permute_boards(engine, order)
        for name in ("timeline", "half_turn", "is_head", "active", "side_to_move", "piece", "colour", "moved", "parent", "source"):
            assert np.array_equal(getattr(m, name), getattr(e, name)), (label, name)
        for name in ("acting_player", "submit_possible", "submit_legal", "even_start", "width", "height", "turn_state"):
            assert getattr(m, name) == getattr(e, name), (label, name)
        for name in ("moves", "edges"):
            assert _rows(getattr(m, name)) == _rows(getattr(e, name)), (label, name)  # the engine's order is its own
        assert e.moves.shape[0] == s.moves.shape[0] and e.moves.shape[0] > 0, label
    # B11's five types are all exercised (pin and check since engine ABI round 3), the mirror's included.
    for rel in RELATIONS:
        assert any((s.edges[:, 2] == RELATION_IDS[rel]).any() for _, s, _, _ in pairs), rel


def test_the_per_square_labels_of_a_real_position_and_its_mirror(fens):
    """Register A7 for D4's per-square labels: the engine's labels of a position and of its
    colour-swapped mirror (each relative to its own acting player - "ours") are the same at the
    mirrored square - board for board (``state.mirror``'s order), rank for rank (``y -> H-1-y``).
    Every label is decided in these positions - the mate searches within the default budget, every
    exchange's recaptures readable at a turn start - on both sides, and every label is exact."""
    seen = np.zeros(len(SQUARE_LABELS), dtype=np.int64)
    for label, parity, strings in fens:
        with Game.from_fen(strings, parity=parity) as g:
            s = to_position_state(g.expand(square_features="all"))
            a = g.square_labels()
        with Game.from_fen([mirror_board_string(t) for t in strings], parity=parity) as g:
            e = to_position_state(g.expand(square_features="all"))
            b = g.square_labels()
        assert a.acting_player == s.acting_player and b.acting_player == 1 - a.acting_player, label
        w, h = s.width, s.height
        cell = np.arange(w * h)
        flip = (h - 1 - cell // w) * w + cell % w
        mapped = b.labels[_mirror_order(s, e)][:, flip]
        assert (a.labels != T5D_LABEL_UNKNOWN).all() and (mapped != T5D_LABEL_UNKNOWN).all(), label
        assert a.mate_unknown == 0 and b.mate_unknown == 0, label
        assert np.array_equal(a.labels, mapped), label
        seen += (a.labels == 1).reshape(-1, len(SQUARE_LABELS)).sum(axis=0)
    # Every label is set somewhere in these positions, the mate's included.
    assert (seen > 0).all(), dict(zip(SQUARE_LABELS, seen.tolist(), strict=True))


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


def test_the_other_perspectives_anchor_on_real_positions(pairs):
    """Spec issue 229 on the engine's own positions, both acting colours: a refresh's dense pass
    as the anchor is reads the engine's moves, move graph and typed relations - the moves, the
    move graph and attack, defence and x-ray starting on the acting player's pieces, and only pin
    and check also on the opponent's (each side read from its own view since engine ABI round 3,
    build-log spec issue 172); the pass from the other colour's side reads none of them; cached =
    reference from both sides."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, attn_scope="block_causal", typed_edge_bias=True), seed=7).eval()
    seen = set()
    for label, s, _, _ in pairs[1::4]:
        if s.is_head.all():
            continue
        seen.add(int(s.acting_player))
        cached = CachedEvaluator(model, CacheConfig(root_refresh=True))
        passes, handle = dense_passes(model)
        try:
            epoch = cached.root_epoch(s)
        finally:
            handle.remove()
        assert_other_perspective_anchor(passes, s)
        own = passes[0]
        n_move_edges = int((own.edges[0][own.edge_valid[0]][:, 2] == RELATION_IDS["move"]).sum())
        assert foreign_sources(own)[:2] == (0, 0) and n_move_edges > 0 and bool(own.move_valid.any()), label
        edges = own.edges[0][own.edge_valid[0]]
        theirs = edges[own.owner[0][edges[:, 0]] != OWNER_OURS]
        assert set(theirs[:, 2].tolist()) <= {RELATION_IDS["pin"], RELATION_IDS["check"]}, label
        for node in (s, viewed_by(s, 1 - int(s.acting_player))):
            _close(cached.evaluate(node, epoch), reference_output(model, node, anchor=s))
    assert seen == {0, 1}
