"""Register A5 (piece-centric tokens, a prototype behind ``NetConfig.piece_centric``) on real engine
positions: the conversion at the network's edge, the evaluator, games end to end and purity.

* The conversion (``state.to_piece_centric``, strict): engine data drops nothing - its edges are its
  moves' own pairs and its relations end on pieces - and the token set is the occupied cells and the
  empty destinations, recounted independently from the engine's own piece count
  (``Game.snapshot().pieces``) and its move records' flags: a capture lands on its piece (en passant
  excepted). Promotions share their destination's token; en passant, castling and branching travel
  are covered; the window's scope over an A5 layout is refused (BUILD_LOG issue 253).
* The evaluator (``NetworkEvaluator._fit``) converts: one finite logit per engine move in the engine's
  order (``check_evaluation``), and the logits equal the masked A1 reference of section 15.2's
  proposed A5 row (``tests/net/test_piece_centric.py``) on engine positions.
* Purity (section 15.2's C19-C21 row): two move orders that reach one state give identical A5 inputs
  and evaluations.
* Section 15.5's "plays legal 5D chess end to end with random weights" with an A5 network: short
  self-play and match games under the ply split, the flat form and the tree form, every record
  replayed by the engine's reader and its gate adapter (self-play shares the evaluation cache across
  fresh searches, C14).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.engine import EngineScope, Game, NetworkEvaluator
from t5d.engine.capi import T5D_MOVE_BRANCHING, T5D_MOVE_CAPTURE, T5D_MOVE_CASTLE, T5D_MOVE_EN_PASSANT, T5D_MOVE_PROMOTION
from t5d.engine.evaluator import Leaf, check_evaluation
from t5d.engine.netplay import PlayConfig, check_replay, network_config, play_game, replay_with_gate
from t5d.engine.play import random_position_game
from t5d.net import NetConfig, build_model, collate
from t5d.net.state import to_piece_centric
from tests.engine.helpers import RECORDS, VARIANTS, requires_engine
from tests.engine.test_netplay import GATE, gate_required, requires_gate  # noqa: F401 - the gate fixture
from tests.net.test_piece_centric import assert_conversion, assert_equivalent

pytestmark = requires_engine

ALL_RELATIONS = ("attack", "defence", "pin", "xray", "check")

#: Positions with the special moves on offer: en passant (White's exd6), castling (White's O-O,
#: both sides' bishops and knights out), a promotion (a 4x4 board, Q and R offered).
SPECIAL = {
    "en_passant": '[Mode "5D"]\n[Board "Standard"]\n\n1. e4 / a6\n2. e5 / d5\n',
    "castling": '[Mode "5D"]\n[Board "Standard"]\n\n1. e4 / e5\n2. Nf3 / Nc6\n3. Bc4 / Bc5\n',
    "promotion": '[Board "custom"]\n[Size "4x4"]\n[Promotions "Q,R"]\n[4/P3/4/K2k:0:1:w]\n',
}


def _final(text: str, **kw):
    with Game.from_pgn(text) as g:
        g.replay_all()
        return g.position(**kw), g.snapshot()


def _positions(relations=ALL_RELATIONS) -> list:
    """Real positions: variant starts, the records' and the special positions, random play with
    branched timelines - every typed relation built."""
    out = []
    for name in VARIANTS:
        with Game.variant(name) as g:
            out.append((g.position(relations=relations), g.snapshot()))
    for text in list(RECORDS.values()) + list(SPECIAL.values()):
        out.append(_final(text, relations=relations))
    for variant, boards, seed in (("Standard", 12, 3), ("Standard - Two Timelines", 16, 5), ("Very Small - Open", 20, 3)):
        with random_position_game(variant, seed=seed, boards=boards, travel_prob=0.3) as g:
            out.append((g.position(relations=relations), g.snapshot()))
    return out


@pytest.fixture(scope="module")
def positions():
    return _positions()


def _independent_count(pos, snap, regs: int = 2, glob: int = 16) -> int:
    """``T_A5`` from the engine's own piece count and its move records alone: a move lands on an
    empty cell unless it captures there (an en passant capture lands on an empty cell)."""
    m = pos.moves
    onto_piece = ((m["flags"] & T5D_MOVE_CAPTURE) != 0) & ((m["flags"] & T5D_MOVE_EN_PASSANT) == 0)
    empty = {(int(b), int(c)) for b, c, p in zip(m["to_board"], m["to_cell"], onto_piece, strict=True) if not p}
    return snap.pieces + len(empty) + pos.state.n_boards * regs + glob


def test_engine_data_drops_nothing_and_the_count_is_the_engines(positions):
    """Strict conversion of every real position (every relation built) drops no edge; the token set
    is the occupied cells and the empty destinations; the count equals the independent recount."""
    for pos, snap in positions:
        s = pos.state
        p = to_piece_centric(s)  # strict: raises on any dropped edge
        assert p.edges.shape == s.edges.shape and p.moves.shape == s.moves.shape
        assert_conversion(s, p)
        assert p.n_tokens == _independent_count(pos, snap)


def test_the_special_moves_tokens():
    """En passant lands on an empty token (the pawn taken keeps its own); castling's king lands on
    an empty token; two promotions share their destination's token; a capture points at the
    captured piece's token."""
    flags = {}
    for name, text in SPECIAL.items():
        pos, _ = _final(text)
        p = to_piece_centric(pos.state)
        tok = p.token_arrays()
        for row, m in enumerate(pos.moves):
            f = int(m["flags"])
            dst = int(p.moves[row, 1])
            if f & T5D_MOVE_EN_PASSANT:
                flags["en_passant"] = True
                assert tok["piece"][dst] == 0
            elif f & T5D_MOVE_CAPTURE:
                flags["capture"] = True
                assert tok["piece"][dst] == int(m["captured"])
            if f & T5D_MOVE_CASTLE:
                flags["castle"] = True
                assert tok["piece"][dst] == 0
        if name == "promotion":
            promos = p.moves[(pos.moves["flags"] & T5D_MOVE_PROMOTION) != 0]
            assert promos.shape[0] == 2 and len(set(promos[:, 1].tolist())) == 1
            flags["promotion"] = True
    assert flags == {"en_passant": True, "capture": True, "castle": True, "promotion": True}


def test_branching_travel_lands_on_a_history_board():
    """A branching jump's destination token sits on the history board it targets, with that board's
    flags (superseded)."""
    with random_position_game("Standard", seed=7, boards=16, travel_prob=0.5) as g:
        pos = g.position()
    rows = np.flatnonzero((pos.moves["flags"] & T5D_MOVE_BRANCHING) != 0)
    assert rows.size, "no branching jump on offer"
    p = to_piece_centric(pos.state)
    tok = p.token_arrays()
    for row in rows.tolist():
        dst = int(p.moves[row, 1])
        b = int(pos.moves["to_board"][row])
        assert tok["board"][dst] == b and tok["is_head"][dst] == 0 and not pos.state.is_head[b]


def test_the_window_scope_refuses_an_a5_layout():
    with Game.variant("Standard") as g:
        pos = g.position()
    assert isinstance(pos.scope, EngineScope)
    pos.scope.tokens(pos.state.layout)  # A1: fine
    with pytest.raises(NotImplementedError, match="issue 253"):
        pos.scope.tokens(to_piece_centric(pos.state).layout)


def test_the_evaluator_converts_and_keeps_the_contract(positions):
    """``NetworkEvaluator`` with an A5 network: every evaluation keeps the C step API's contract -
    one finite logit per engine move, in its order - batched and alone alike."""
    cfg = replace(network_config("small"), piece_centric=True)
    ev = NetworkEvaluator(build_model(cfg, seed=1).eval(), max_batch=4)
    states = [pos for pos, _ in positions]
    with torch.no_grad():
        batched = ev(states)
        for i, (pos, e) in enumerate(zip(states, batched, strict=True)):
            check_evaluation(Leaf(i, pos), e, cfg.value_bins)
            alone = ev([pos])[0]
            assert np.allclose(alone.policy_logits, e.policy_logits, atol=1e-4)


def test_the_logits_equal_the_masked_reference_on_engine_positions(positions):
    """Section 15.2's proposed A5 row on real positions (typed relations on: B11 reads them)."""
    flags = dict(trunk_dtype="fp32", n_layers=2, typed_edge_bias=True)
    m5 = build_model(NetConfig(piece_centric=True, **flags), seed=2).eval()
    m1 = build_model(NetConfig(**flags), seed=2).eval()
    states1 = [pos.state for pos, _ in positions]
    with torch.no_grad():
        for chunk in (states1[:6], states1[6:12], states1[12:]):
            assert_equivalent(m5, m1, chunk, [to_piece_centric(s) for s in chunk])


#: One action's two moves on Two Timelines' two boards, in either order: one multiverse (a move
#: order across actions changes the history boards, so a transposition lives inside an action).
TRANSPOSED = (
    '[Board "Standard - Two Timelines"]\n[Mode "5D"]\n\n1. (-0T1)e4 (+0T1)d4 / (-0T1)e5 (+0T1)Nf6\n',
    '[Board "Standard - Two Timelines"]\n[Mode "5D"]\n\n1. (+0T1)d4 (-0T1)e4 / (+0T1)Nf6 (-0T1)e5\n',
)


def test_two_move_orders_give_one_input_and_one_evaluation():
    """Purity (C19-C21): one state reached by two move orders - its A5 input arrays and its
    evaluation are identical, bit for bit."""
    finals = [_final(text) for text in TRANSPOSED]
    assert finals[0][1].fingerprint == finals[1][1].fingerprint
    cfg = replace(network_config("small"), piece_centric=True)
    ev = NetworkEvaluator(build_model(cfg, seed=3).eval())
    batches = [collate([to_piece_centric(pos.state)]) for pos, _ in finals]
    for name, value in batches[0].__dict__.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, batches[1].__dict__[name]), name
    with torch.no_grad():
        a, b = (ev([pos])[0] for pos, _ in finals)
    assert np.array_equal(a.policy_logits, b.policy_logits) and np.array_equal(a.wdl_logits, b.wdl_logits)
    assert np.array_equal(a.board_logits, b.board_logits)


@requires_gate
@pytest.mark.parametrize(
    "variant,split,mode",
    [("Standard", "ply", "self"), ("Standard - Two Timelines", "flat", "match"), ("Standard - Two Timelines", "tree", "self")],
)
def test_an_a5_network_plays_legal_games(variant, split, mode):
    """Section 15.5 with A5's tokens: every record legal, replayed by the gate adapter."""
    cfg = replace(network_config("small"), piece_centric=True)
    evaluators = [NetworkEvaluator(build_model(cfg, seed=s).eval(), max_batch=8) for s in (1, 2)]
    play = PlayConfig(variant=variant, split=split, mode=mode, playouts=8, batch_size=4, cap_turns=3, cap_boards=30, seed=3)
    with torch.no_grad():
        res = play_game(evaluators if mode == "match" else evaluators[0], play, net_cfg=cfg)
    assert res.actions and all(res.actions) and res.evaluations > 0
    replay = replay_with_gate(res.record, gate=GATE)
    assert check_replay(res, replay) == [], replay.lines[-3:]


def test_the_engines_cached_evaluator_refuses_a5():
    """The cached path is refused for A5 (BUILD_LOG issue 252) - the engine's wrapper too."""
    from types import SimpleNamespace

    from t5d.engine.evaluator import CachedNetworkEvaluator

    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=True), seed=0)
    with pytest.raises(NotImplementedError, match="issue 252"):
        CachedNetworkEvaluator(SimpleNamespace(model=model))
