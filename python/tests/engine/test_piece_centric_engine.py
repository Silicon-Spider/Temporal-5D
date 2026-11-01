"""Register A5 (piece-centric tokens, a prototype behind ``NetConfig.piece_centric``) on real engine
positions: the conversion at the network's edge, the evaluator, games end to end, purity, and the
token-count tool (``t5d.engine.token_counts``).

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
    branched timelines - every typed relation built; the last two random positions also with a
    search leaf's candidate moves (the legal ones, exact - the destinations then the legal moves')."""
    out = []
    for name in VARIANTS:
        with Game.variant(name) as g:
            out.append((g.position(relations=relations), g.snapshot()))
    for text in list(RECORDS.values()) + list(SPECIAL.values()):
        out.append(_final(text, relations=relations))
    for variant, boards, seed in (("Standard", 12, 3), ("Standard - Two Timelines", 16, 5), ("Very Small - Open", 20, 3)):
        with random_position_game(variant, seed=seed, boards=boards, travel_prob=0.3) as g:
            out.append((g.position(relations=relations), g.snapshot()))
            if variant != "Standard":
                out.append((g.position(relations=relations, legal_moves=True), g.snapshot()))
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


# --- the token-count tool (A5.5) -----------------------------------------------------------------------------------


def test_the_cost_models_count_is_the_engines(positions):
    from t5d.net.cost import token_counts

    for pos, snap in positions:
        assert token_counts(pos.state)["a5"] == _independent_count(pos, snap) == to_piece_centric(pos.state).n_tokens


def test_the_token_tool_counts_a_record():
    """``token_counts.record_positions`` on a record: one entry per action start, each the network's
    own construction and the independent recount, the legal moves a subset of the pseudo-legal."""
    found, fate = record_positions_from_text(RECORDS["branching"])
    assert fate == "ok" and len(found) == 7, "the start and one per action (three turns of two)"
    for entry, (pos, snap) in found:
        assert entry["pseudo"]["a5"] == _independent_count(pos, snap)
        assert entry["legal"]["a5"] <= entry["pseudo"]["a5"] and entry["legal"]["moves"] <= entry["pseudo"]["moves"]
        assert entry["pseudo"]["a1"] == 66 * pos.state.n_boards + 16


def record_positions_from_text(text: str):
    """``position_counts`` at every action start of a record held as text, beside each position."""
    from t5d.engine.token_counts import position_counts

    out = []
    with Game.from_pgn(text, legal=False) as g:
        out.append((position_counts(g), (g.position(), g.snapshot())))
        for _ in g.replay():
            out.append((position_counts(g), (g.position(), g.snapshot())))
    return out, "ok"


def test_the_token_tool_reads_files(tmp_path):
    from t5d.engine.token_counts import record_positions

    path = tmp_path / "game.5dpgn"
    path.write_text(RECORDS["cross_timeline"], encoding="utf-8")
    found, fate = record_positions(path)
    with Game.from_pgn(RECORDS["cross_timeline"], legal=False) as g:
        g.replay_all()
        boards = g.info().boards
    assert fate == "ok" and len(found) == 5 and found[-1]["boards"] == boards
    bad = tmp_path / "bad.5dpgn"
    bad.write_text('[Mode "5D"]\n[Board "Standard"]\n\n1. e5 / e4\n', encoding="utf-8")
    assert record_positions(bad)[1].startswith("refused")


# --- the review's gaps (A5 review fixes, 2026-10-24) --------------------------------------------------------------


def _strict_ok(seed: int, boards: int):
    """A synthetic square-centric state with no edges and no opponent moves (synthetic move edges
    reach empty cells; engine data's never do), so that a strict conversion takes it."""
    from t5d.net import synthetic_state

    none = np.zeros((0, 3), dtype=np.int64)
    return replace(synthetic_state(seed, boards), edges=none, opp_moves=none)


def test_the_evaluator_converts_strictly():
    """``NetworkEvaluator._fit`` converts strictly: an edge onto a cell that is no token is refused,
    never silently dropped."""
    from t5d.net import synthetic_state

    ev = NetworkEvaluator(build_model(NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=True), seed=0).eval())
    s = synthetic_state(11, 4)
    assert to_piece_centric(s, strict=False, return_dropped=True).dropped_edges > 0
    with pytest.raises(ValueError, match="strict"):
        ev([s])


def test_the_evaluator_converts_after_a_relayout():
    """A position of other register counts than an A5 network's is relaid out and then converted: the
    evaluation equals the network's on ``to_piece_centric(relayout(s, 4, 8))``."""
    from t5d.net.state import relayout

    cfg = NetConfig(trunk_dtype="fp32", n_layers=1, piece_centric=True, regs_per_board=4, global_regs=8)
    model = build_model(cfg, seed=0).eval()
    s = _strict_ok(12, 5)
    assert (s.regs_per_board, s.global_regs) == (2, 16)
    with torch.no_grad():
        e = NetworkEvaluator(model)([s])[0]
        ref = model(collate([to_piece_centric(relayout(s, 4, 8))]))
    assert e.policy_logits.shape == (s.moves.shape[0],) and np.isfinite(e.policy_logits).all()
    assert np.allclose(e.policy_logits, ref.policy_logits[0, : s.moves.shape[0]].numpy(), atol=1e-6)


def test_netplay_takes_piece_centric(monkeypatch):
    """``netplay --piece-centric`` builds piece-centric networks (and without the flag, not)."""
    from t5d.engine import netplay

    built = []

    class Stop(Exception):
        pass

    def fake_build(cfg, seed=0):
        built.append(cfg)
        raise Stop

    monkeypatch.setattr(netplay, "build_model", fake_build)
    threads = torch.get_num_threads()
    for flag, want in ((["--piece-centric"], True), ([], False)):
        with pytest.raises(Stop):
            netplay.main(["--net", "small", "--threads", str(threads), "--no-replay", *flag])
        assert built[-1].piece_centric is want


def test_the_bench_counts_a5s_tokens():
    """The engine bench under ``--piece-centric``: the tokens column is A5's and the conversion's own
    time is reported; under A1 the tokens are ``66 N + 16`` and no conversion time."""
    from t5d.engine.bench import forward

    with Game.variant("Standard") as g:
        games = {1: [g]}
        for pc in (True, False):
            cfg = replace(network_config("small"), piece_centric=pc, n_layers=1)
            (row,) = forward(games, [2], 1, cfg)
            state = g.position(regs_per_board=cfg.regs_per_board, global_regs=cfg.global_regs).state
            want = to_piece_centric(state).n_tokens if pc else state.n_tokens
            assert row["tokens"] == 2 * want and ("A5 convert ms" in row) is pc
            assert row["tokens"] == (2 * (32 + 16 + 2 + 16) if pc else 2 * (66 + 16))


def _counts(seed: int, boards: int) -> dict:
    """A ``position_counts``-shaped entry of a synthetic position (both move sets alike)."""
    from t5d.net import synthetic_state
    from t5d.net.cost import token_counts

    c = token_counts(synthetic_state(seed, boards, n_timelines=3))
    return {"key": f"{seed}-{boards}", "boards": boards, "legal": c, "pseudo": c}


def test_the_summary_pools_by_decisions_and_takes_the_phase_form():
    """``token_counts.summarise`` on hand-built positions (10 and 20 boards, weights 1 and 3; 60
    boards, weight 2): the median per-position ratio A1 / A5, every pooled figure weighted by the
    decisions, M16's blend (the pre-CHECKPOINT-9 reading) and C26's phase form - dense below N*,
    the cached node at or above it, each half apart."""
    from t5d.engine.token_counts import CACHED_SHARE, summarise
    from t5d.net.cost import piece_centric_costs

    cfg = NetConfig()
    entries = [_counts(1, 10), _counts(2, 20), _counts(3, 60)]
    weights = {entries[0]["key"]: 1, entries[1]["key"]: 3, entries[2]["key"]: 2}
    out = summarise({e["key"]: e for e in entries}, weights, n_stars=(50, 100))
    low = out["buckets"]["legal"]["1-49"]
    ratios = [e["legal"]["a1"] / e["legal"]["a5"] for e in entries[:2]]
    assert low["token_ratio_median"] == pytest.approx(sum(ratios) / 2) and low["token_ratio_median"] > 1
    assert low["positions"] == 2 and low["action_starts"] == 4
    k = [piece_centric_costs(e["legal"], cfg) for e in entries]
    w = [1, 3, 2]

    def pooled(what: str, arm: str, which=(0, 1, 2)) -> float:
        return sum(w[i] * k[i][f"{what}_{arm}"] for i in which)

    p = out["pooled"]["legal"]
    assert p["action_starts"] == 6
    assert p["dense_ratio"] == pytest.approx(pooled("dense", "a1") / pooled("dense", "a5"))
    assert p["node_ratio"] == pytest.approx(pooled("node", "a1") / pooled("node", "a5"))
    blend = {a: CACHED_SHARE * pooled("node", a) + (1 - CACHED_SHARE) * pooled("dense", a) for a in ("a1", "a5")}
    assert p["blend_ratio"] == pytest.approx(blend["a1"] / blend["a5"])
    at50, at100 = p["phase"]["50"], p["phase"]["100"]
    phase = {a: pooled("dense", a, (0, 1)) + pooled("node", a, (2,)) for a in ("a1", "a5")}
    assert at50["ratio"] == pytest.approx(phase["a1"] / phase["a5"])
    assert at50["dense_below_ratio"] == pytest.approx(pooled("dense", "a1", (0, 1)) / pooled("dense", "a5", (0, 1)))
    assert at50["cached_above_ratio"] == pytest.approx(k[2]["node_a1"] / k[2]["node_a5"])
    assert (at50["action_starts_below"], at50["action_starts_above"]) == (4, 2)
    assert at50["share_above_a1"] == pytest.approx(pooled("node", "a1", (2,)) / phase["a1"])
    assert at100["ratio"] == pytest.approx(p["dense_ratio"]) and np.isnan(at100["cached_above_ratio"])


def test_the_tool_weighs_repeats_and_leaves_a_refused_record_out(tmp_path):
    """``token_counts.measure``: a position reached in two records counts twice in the pooled
    figures (the decisions spent there) and once among the distinct positions; a record that does
    not replay to the end counts none of its positions; the JSON lines carry each position's
    decisions, which add up to the pooled count."""
    import json

    from t5d.engine.token_counts import measure, record_positions

    good = tmp_path / "a.5dpgn"
    good.write_text(RECORDS["branching"], encoding="utf-8")
    other = tmp_path / "b.5dpgn"
    other.write_text('[Board "Standard"]\n[Mode "5D"]\n\n1. Nf3 / Nf6\n2. Nc3 / Nc6\n', encoding="utf-8")
    bad = tmp_path / "c.5dpgn"
    bad.write_text('[Mode "5D"]\n[Board "Standard"]\n\n1. e4 / e5\n2. e5 / d6\n', encoding="utf-8")
    found = {p.name: record_positions(p) for p in (good, other, bad)}
    assert found["c.5dpgn"][1].startswith("refused") and len(found["c.5dpgn"][0]) >= 1
    starts = len(found["a.5dpgn"][0]) + len(found["b.5dpgn"][0])
    distinct = len({p["key"] for name in ("a.5dpgn", "b.5dpgn") for p in found[name][0]})
    assert distinct < starts, "the two records share their first positions"
    lines = tmp_path / "positions.jsonl"
    out = measure([good, other, bad], jsonl=lines)
    assert out["records"] == {"ok": 2, "refused": 1} and out["distinct_positions"] == distinct
    for move_set in ("legal", "pseudo"):
        assert out["pooled"][move_set]["action_starts"] == starts
    rows = [json.loads(line) for line in lines.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == distinct and sum(r["action_starts"] for r in rows) == starts
    assert max(r["action_starts"] for r in rows) == 2


def test_the_counted_forward_extrapolates_exactly():
    """``token_counts.flops_table``: the 16-layer count extrapolated from 1 and 2 layers equals the
    full-depth count (``check``), A1 and A5 - the claim the MEASUREMENTS figures rest on."""
    from t5d.engine.token_counts import flops_table
    from t5d.net import synthetic_state

    (first, second) = flops_table({"two boards": synthetic_state(0, 2)}, check=True)
    for arm in ("a1", "a5"):
        assert first[f"counted_full_{arm}"] == first[f"counted_{arm}"] > first[f"per_layer_{arm}"] > 0
    assert first["path"] == "block_sparse" and second["path"] == "chunked" and "counted_full_a1" not in second


def test_the_bench_takes_piece_centric(monkeypatch, capsys):
    """``bench --piece-centric`` builds the forward's network piece-centric and says so in its
    conditions (and without the flag, not)."""
    from t5d.engine import bench

    seen = []
    monkeypatch.setattr(bench, "_positions", lambda sizes, seeds, travel: {n: [] for n in sizes})
    monkeypatch.setattr(bench, "conversion", lambda games, repeat: [])
    monkeypatch.setattr(bench, "forward", lambda games, batches, repeat, cfg: seen.append(cfg) or [])
    monkeypatch.setattr(bench, "_table", lambda rows: "")
    grad = torch.is_grad_enabled()
    try:
        for flag, want in ((["--piece-centric"], True), ([], False)):
            assert bench.main(["--layers", "1", *flag]) == 0
            assert seen[-1].piece_centric is want
            assert ("tokens piece-centric (A5)" in capsys.readouterr().out) is want
    finally:
        torch.set_grad_enabled(grad)  # the bench switches autograd off for the process
