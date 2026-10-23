"""Register A8, the board-layout embedding (spec section 2.2; a prototype behind its flag, gate 6p;
``t5d/net/layout.py``).

Off - the default - the network is the network without it, bit for bit: no parameter, every other
parameter drawn as before, the same outputs. On: each square token's layer-0 input carries the
projection of its own board's one-hot layout, read in register A7's canonical (x, y) order, at its
own cell; boards smaller than 8 x 8 are padded (``layout_padding``, a placeholder); the
embedding's FFN follows; nothing of it is quantised (G2); every attention path and the cached path
compute the same function with it; and section 15.3's row - "~0.1% of attention at 44 boards;
~1.6M parameters at 16 channels" - is counted. The A7 mirror row with A8 on is in
``test_perspective.py``; the smoke optimiser steps are below.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, mirror, parameter_breakdown, synthetic_state
from t5d.net.cache import CachedEvaluator, reference_output
from t5d.net.config import ATTN_IMPLS
from t5d.net.cost import attention_flops, count_flops, layout_flops
from t5d.net.layout import LAYOUT_CELLS, LAYOUT_FEATURES, LAYOUT_STATES, BoardLayoutEmbedding, layout_parameters
from t5d.net.quant import QuantLinear
from t5d.net.state import OWNER_OURS, TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_SQUARE
from t5d.net.tree import SearchTree
from tests.net.test_cache import _close, _random_moves, _root
from tests.net.test_cache import _model as _cached_model
from tests.net.test_displacement import far_state

A8 = dict(board_layout_embedding=True)


def _model(seed: int = 0, layers: int = 2, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=layers, **flags), seed=seed).eval()


def _outputs_equal(a, b) -> None:
    for name in ("trunk", "value_logits", "wdl_logits", "short_term", "value_error", "moves_left_log", "board_logits", "square_logits"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert torch.equal(a.policy_logits, b.policy_logits) and torch.equal(a.submit_raw, b.submit_raw)


# --- off: the network without it, bit for bit --------------------------------------------------------


def test_off_by_default():
    cfg = NetConfig()
    assert cfg.board_layout_embedding is False
    assert (cfg.layout_channels, cfg.layout_ffn_ratio, cfg.layout_padding) == (16, 8, "a1"), "section 2.2's 16; the placeholders"
    model = build_model(cfg)
    assert model.embed.layout is None
    assert not any(name.startswith("embed.layout") for name, _ in model.named_parameters())
    assert parameter_breakdown(model)["embed.layout"] == 0


FLAG_SETS = {
    "v1": {},
    "flagged": dict(typed_edge_bias=True, board_pair_smolgen=True, shaw_key_term=True, opp_reply=True, optimistic_policy=True),
    "cached": dict(attn_scope="block_causal"),
    "no_ffn_centre": dict(layout_ffn_ratio=0, layout_padding="centre", layout_channels=4),
}


@pytest.mark.parametrize("flags", list(FLAG_SETS))
def test_switching_it_on_changes_no_other_parameter_or_output(flags):
    """On, every parameter of the network without it is drawn as before, bit for bit (A8 draws
    last, from a generator of its own), whatever the other flags; and with its module removed the
    network computes exactly the network built with the flag off."""
    for seed in (0, 7):
        off = _model(seed, **FLAG_SETS[flags])
        on = _model(seed, **FLAG_SETS[flags], board_layout_embedding=True)
        p_off, p_on = off.state_dict(), on.state_dict()
        assert set(p_on) - set(p_off) == {k for k in p_on if k.startswith("embed.layout.")} != set()
        assert all(torch.equal(p_off[k], p_on[k]) for k in p_off)
        batch = collate([synthetic_state(seed + 3, 9, n_timelines=3)])
        with_it = on(batch, return_trunk=True)
        on.embed.layout = None
        _outputs_equal(off(batch, return_trunk=True), on(batch, return_trunk=True))
        assert not torch.allclose(with_it.value_logits, off(batch).value_logits, atol=1e-4), "on, it is live"


def test_its_draws_are_its_own():
    """A8's parameters are a function of the model's seed alone: the other flags do not move them."""
    a = _model(5, **A8).embed.layout.state_dict()
    b = _model(5, **A8, typed_edge_bias=True, board_pair_smolgen=True, attn_scope="block_causal", opp_reply=True).embed.layout.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    c = _model(6, **A8).embed.layout.state_dict()
    assert not torch.equal(a["proj.weight"], c["proj.weight"])


# --- the parameter count (section 15.3's row: "~1.6M parameters at 16 channels") --------------------


def test_the_parameter_count():
    """At the v1's dimensions: the projection 1,536 -> 64 x 16 with its bias, 1,573,888 - section
    2.2's "about 1.6M" (the weight alone 1,572,864); the merge 16 -> 256, 4,096; the FFN at the
    placeholder's 8x, 1,051,136 (gain 256, 256 -> 2,048 -> 256 with F5's biases): 2,629,120, on
    the v1's 22,965,426 - 25,594,546. Without the FFN 1,577,984."""
    v1 = parameter_breakdown(build_model(NetConfig()))
    counts = parameter_breakdown(build_model(NetConfig(**A8)))
    assert counts["embed.layout"] == 1_573_888 + 4_096 + 1_051_136 == 2_629_120
    assert layout_parameters(NetConfig(**A8)) == {"projection": 1_573_888, "merge": 4_096, "ffn": 1_051_136, "total": 2_629_120}
    assert abs(LAYOUT_FEATURES * LAYOUT_CELLS * 16 - 1.6e6) / 1.6e6 < 0.02, "section 2.2: about 1.6M at 16 channels"
    assert counts["total"] == v1["total"] + 2_629_120 == 25_594_546
    assert {k: v for k, v in counts.items() if k not in ("embed.layout", "total")} == {
        k: v for k, v in v1.items() if k not in ("embed.layout", "total")
    }
    for kw in (
        dict(layout_ffn_ratio=0),
        dict(layout_channels=8, layout_ffn_ratio=2),
        dict(out_ffn_bias=False),
        dict(d_model=64, n_heads=2),
    ):
        cfg = NetConfig(**A8, **kw)
        assert parameter_breakdown(build_model(cfg))["embed.layout"] == layout_parameters(cfg)["total"], kw
    assert layout_parameters(NetConfig(**A8, layout_ffn_ratio=0))["total"] == 1_577_984


def test_the_configuration_is_validated():
    for bad in (dict(layout_channels=0), dict(layout_ffn_ratio=-1), dict(layout_padding="h8")):
        with pytest.raises(ValueError):
            NetConfig(**A8, **bad).validate()


# --- the layout: canonical order, per board, padded ---------------------------------------------------


def _expected_layouts(state, padding: str = "a1") -> np.ndarray:
    """``(N, 1536)``: the one-hot layouts written from the position's absolute facts - the
    acting player's pieces ours, ranks mirrored when Black acts (A7) - independently of the
    batch's canonical arrays."""
    w, h = int(state.width), int(state.height)
    ox, oy = ((8 - w) // 2, (8 - h) // 2) if padding == "centre" else (0, 0)
    out = np.zeros((state.n_boards, LAYOUT_FEATURES), dtype=np.float32)
    ours = 1 + int(state.acting_player)
    for b in range(state.n_boards):
        for s in range(w * h):
            piece = int(state.piece[b, s])
            if piece == 0:
                continue
            x, y = s % w, s // w
            y_c = y if int(state.acting_player) == 0 else h - 1 - y
            owner = 0 if int(state.colour[b, s]) == ours else 1
            out[b, ((y_c + oy) * 8 + (x + ox)) * LAYOUT_STATES + 12 * owner + piece - 1] = 1.0
    return out


@pytest.mark.parametrize("padding", ["a1", "centre"])
@pytest.mark.parametrize("size", [(8, 8), (7, 7), (6, 6), (6, 4), (5, 5), (4, 4), (3, 5), (3, 3), (1, 1)], ids=str)
def test_the_layout_is_read_in_canonical_order(size, padding):
    """Each board's one-hot layout is its pieces at their canonical cells (A7: ours and theirs,
    ranks mirrored when Black acts), with White and with Black acting - each position and its
    mirror, so both acting colours are seen at every size (reading the ranks in storage order
    puts Black's pieces at other cells, A7's warning, and fails here) - and smaller boards
    padded into the frame as ``layout_padding`` places them. The non-square sizes (6 x 4, 3 x 5)
    tell the file's offset from the rank's under ``"centre"``."""
    width, height = size
    layer = BoardLayoutEmbedding(NetConfig(**A8, layout_padding=padding))
    for seed in (0, 1):
        s = (
            far_state(40 + seed, size=1)
            if size == (1, 1)
            else synthetic_state(40 + seed, 6, n_timelines=3, width=width, height=height, fairy_prob=0.5)
        )
        assert (int(s.width), int(s.height)) == size
        acting = set()
        for state in (s, mirror(s)):
            acting.add(int(state.acting_player))
            got = layer.layouts(collate([state]))[0].numpy()
            assert np.array_equal(got, _expected_layouts(state, padding)), (size, int(state.acting_player))
            per_cell = got.reshape(state.n_boards, 8, 8, LAYOUT_STATES)
            ox, oy = ((8 - width) // 2, (8 - height) // 2) if padding == "centre" else (0, 0)
            inside = np.zeros((8, 8), dtype=bool)
            inside[oy : oy + height, ox : ox + width] = True
            assert not per_cell[:, ~inside].any(), "the padding cells are empty"
            assert per_cell.sum() == (state.piece > 0).sum()
        assert acting == {0, 1}


def test_boards_larger_than_the_frame_are_refused():
    layer = BoardLayoutEmbedding(NetConfig(**A8))
    with pytest.raises(ValueError, match="up to 8 x 8"):
        layer.layouts(collate([synthetic_state(1, 3, width=9, height=8)]))


def test_mixed_board_sizes_in_one_batch():
    """Positions of different sizes in one padded batch embed as each does alone."""
    model = _model(2, **A8, layout_padding="centre")
    states = [synthetic_state(50, 5, n_timelines=2, width=5, height=5), synthetic_state(51, 7), synthetic_state(52, 3, width=6, height=6)]
    batch = collate(states)
    together = model.embed_tokens(batch)
    for i, state in enumerate(states):
        alone = model.embed_tokens(collate([state]))[0]
        n_body = int(batch.n_body[i])
        assert torch.allclose(together[i, :n_body], alone[:n_body], atol=1e-6), i
        assert torch.allclose(together[i, batch.body_len :], alone[n_body:], atol=1e-6), i
        assert torch.equal(together[i, n_body : batch.body_len], torch.zeros_like(together[i, n_body : batch.body_len])), "padding"


def test_layer0_depends_only_on_the_tokens_own_board():
    """Perturbing a piece on one board changes the layout channels of that board's squares - every
    square of it - and of no other token; registers and padding carry none."""
    model = _model(3, **A8)
    layer = model.embed.layout
    state = synthetic_state(60, 8, n_timelines=3)
    batch = collate([state])
    base = layer.channels(batch)[0]
    square = batch.kind[0] == TOKEN_SQUARE
    assert torch.equal(base[~square], torch.zeros_like(base[~square]))
    for board in (0, 5):
        tokens = torch.nonzero((batch.board[0] == board) & square & (batch.piece[0] > 0)).flatten()
        t = int(tokens[0])
        piece = batch.piece.clone()
        piece[0, t] = 1 + (int(piece[0, t]) % 12)  # another piece type, same owner
        moved = layer.channels(replace(batch, piece=piece))[0]
        changed = (moved - base).abs().amax(-1) > 0
        on_board = (batch.board[0] == board) & square
        assert torch.equal(changed, on_board), board
        owner = batch.owner.clone()
        owner[0, t] = 3 - int(owner[0, t])  # ours <-> theirs
        assert torch.equal((layer.channels(replace(batch, owner=owner))[0] - base).abs().amax(-1) > 0, on_board)
    whole = model.embed_tokens(batch)[0]
    piece = batch.piece.clone()
    t = int(torch.nonzero((batch.board[0] == 2) & square & (piece[0] > 0)).flatten()[0])
    piece[0, t] = 1 + (int(piece[0, t]) % 12)
    differs = (model.embed_tokens(replace(batch, piece=piece))[0] - whole).abs().amax(-1) > 0
    assert torch.equal(differs, (batch.board[0] == 2) & square), "the FFN is per token: nothing else moves"


def test_the_owner_states():
    """The 24 states: ours first, then theirs, each in piece order; has-moved is not one of them."""
    layer = BoardLayoutEmbedding(NetConfig(**A8))
    state = synthetic_state(61, 3, n_timelines=1)
    batch = collate([state])
    lay = layer.layouts(batch)[0].reshape(-1, LAYOUT_CELLS, LAYOUT_STATES)
    t = int(torch.nonzero((batch.kind[0] == TOKEN_SQUARE) & (batch.piece[0] > 0)).flatten()[0])
    board, x, y = int(batch.board[0, t]), int(batch.coords[0, t, 0]), int(batch.coords[0, t, 1])
    state_index = 12 * (int(batch.owner[0, t]) - OWNER_OURS) + int(batch.piece[0, t]) - 1
    assert lay[board, y * 8 + x, state_index] == 1 and lay[board, y * 8 + x].sum() == 1
    moved = batch.moved.clone()
    moved[0, t] = 1 - int(moved[0, t])
    assert torch.equal(layer.layouts(replace(batch, moved=moved)), layer.layouts(batch))


# --- precision: FP32, never quantised (G1, G2) ---------------------------------------------------------


def test_it_is_fp32_and_never_quantised():
    """Under the BF16 trunk the embedding is FP32 (it runs before the trunk's autocast) - in
    dtype and in value: its layout channels and the embedding agree with an FP64 reference (the
    same module in double) to FP32's rounding, far inside a BF16 step - and under G2's 8-bit
    trunk none of its layers is a quantised projection and the embedding is unchanged, bit for
    bit (issue 238(d))."""
    batch = collate([synthetic_state(70, 6)])
    bf16 = build_model(NetConfig(n_layers=2, **A8), seed=1).eval()
    assert bf16.embed_tokens(batch).dtype == torch.float32
    layer = bf16.embed.layout
    ref = copy.deepcopy(layer).double()
    with torch.no_grad():
        got_c, ref_c = layer.channels(batch), ref.channels(batch)
        assert got_c.dtype == torch.float32 and ref_c.dtype == torch.float64
        assert (got_c.double() - ref_c).abs().max() <= 1e-6 * ref_c.abs().max(), "the channels: FP32 against FP64"
        bf16.embed.layout = None
        before = bf16.embed_tokens(batch)  # the token input A8 adds to
        bf16.embed.layout = layer
        got, want = bf16.embed_tokens(batch), ref(batch, before.double())
        assert (got.double() - want).abs().max() <= 1e-5 * want.abs().max(), "the embedding: FP32 against FP64"
    fp8 = bf16.with_trunk_quant("fp8_e4m3")
    assert not any(isinstance(m, QuantLinear) for m in fp8.embed.layout.modules())
    assert sum(isinstance(m, QuantLinear) for m in fp8.modules()) == 6 * 2
    assert torch.equal(fp8.embed_tokens(batch), bf16.embed_tokens(batch))
    assert not torch.equal(fp8(batch).value_logits, bf16(batch).value_logits), "the trunk is quantised"


# --- the FFN after the embedding: its form (spec issue 240(b), a placeholder) ----------------------------


def test_the_ffn_is_one_pre_norm_residual_block_on_every_valid_token():
    """Issue 240(b)'s documented form, bit for bit: with ``x0`` the embedding with the layout
    channels added and no FFN (the ``layout_ffn_ratio=0`` network of the same seed - A8's
    projection and merge are drawn first from its own generator, so they are the same
    parameters), the embedding is ``x0 + where(valid, W2 mish(W1 RMSNorm(x0)), 0)`` - after the
    layout channels, residual, pre-norm, Mish, on every valid token: the per-board and global
    registers change as the squares do, padding stays zero."""
    with_ffn = _model(9, **A8)
    without = _model(9, **A8, layout_ffn_ratio=0)
    layer = with_ffn.embed.layout
    for name in ("proj.weight", "proj.bias", "merge.weight"):
        assert torch.equal(layer.state_dict()[name], without.embed.layout.state_dict()[name]), name
    states = [synthetic_state(90, 6, n_timelines=3), synthetic_state(91, 4, width=6, height=6)]
    batch = collate(states)
    with torch.no_grad():
        x0 = without.embed_tokens(batch)
        h = layer.ffn_out(torch.nn.functional.mish(layer.ffn_in(layer.norm(x0))))
        want = x0 + torch.where(batch.valid[..., None], h, torch.zeros((), dtype=h.dtype))
        got = with_ffn.embed_tokens(batch)
    assert torch.equal(got, want)
    moved = (got - x0).abs().amax(-1) > 0
    for kind in (TOKEN_SQUARE, TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER):
        rows = batch.valid & (batch.kind == kind)
        assert rows.any() and bool(moved[rows].all()), f"the FFN reaches every valid token of kind {kind}"
    assert not bool(moved[~batch.valid].any()) and torch.equal(got[~batch.valid], x0[~batch.valid]), "padding untouched"


# --- every attention path, and the cached path ---------------------------------------------------------


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
def test_every_attention_path_computes_the_same_function(scope):
    """With A8 on, the six attention implementations agree (the embedding is before them all)."""
    state = synthetic_state(80, 7, n_timelines=3, width=6, height=6)
    batch = collate([state])
    ref = None
    for impl in ("dense", *[i for i in ATTN_IMPLS if i != "dense"]):
        model = _model(4, **A8, attn_impl=impl, attn_scope=scope)
        out = model(batch, return_trunk=True)
        if ref is None:
            ref = out
            continue
        assert torch.allclose(out.trunk, ref.trunk, atol=2e-5), (impl, float((out.trunk - ref.trunk).abs().max()))
        assert torch.allclose(out.value_logits, ref.value_logits, atol=1e-4), impl
        assert torch.allclose(out.policy_logits[out.move_valid], ref.policy_logits[ref.move_valid], atol=1e-4), impl


@pytest.mark.parametrize("geometry", ["standard", "deep_travel", "small_board", "even_start", "registers_4_0"])
def test_cached_equals_one_full_pass(geometry):
    """Section 15.2's first row (C2, C17) with A8 on: along random search paths every node's cached
    evaluation equals one full block-causal pass - the layout is per board and intrinsic (with its
    perspective), the FFN per token - several seeds, both paddings."""
    for seed, padding in ((0, "a1"), (1, "centre"), (2, "a1")):
        model = _cached_model(seed, board_layout_embedding=True, layout_padding=padding)
        root = _root(geometry, seed)
        tree = SearchTree(CachedEvaluator(model), root)
        node = tree.root
        for move in [None, *_random_moves(root, np.random.default_rng(seed), 5)]:
            if move is not None:
                node = tree.child(node, move)
            _close(tree.evaluate(node), reference_output(model, node.state), what=(geometry, seed, move))


# --- section 15.3's cost row -----------------------------------------------------------------------------


def test_the_cost_row():
    """Section 15.3: "~0.1% of attention at 44 boards". The projection is ``2 x 1,536 x 1,024``
    FLOPs per board: 0.138 GFLOP at 44 boards against attention's ``4 T^2 d L`` = 139.7 GFLOP (T =
    2,920) - 0.099%. The register's FFN, which the row leaves out, is ``4 T d^2 r`` = 6.12 GFLOP -
    4.4% of attention. And the implementation issues exactly ``layout_flops``: the counted FLOPs
    with A8 exceed those without by it, nothing else."""
    cfg = NetConfig(**A8)
    t = 44 * 66 + 16
    flops = layout_flops(44, t, cfg)
    assert layout_flops(44, t, NetConfig()) == dict.fromkeys(flops, 0), "off: no A8, no FLOPs"
    assert flops["projection"] == 2 * 44 * 1_536 * 1_024
    attention = attention_flops(t, t, cfg)
    assert abs(flops["projection"] / attention - 0.001) < 0.0001, "section 15.3: ~0.1% of attention at 44 boards"
    assert flops["ffn"] == 4 * t * 256 * 2048 and abs(flops["ffn"] / attention - 0.0438) < 0.0005
    for state, kw in ((synthetic_state(13, 5), {}), (synthetic_state(14, 4, width=5, height=5), dict(layout_ffn_ratio=0))):
        batch = collate([state])
        counted = {}
        for on in (True, False):
            model = _model(13, attn_impl="sdpa", board_layout_embedding=on, **kw)
            _, counted[on] = count_flops(lambda model=model, batch=batch: model(batch))
        assert counted[True] - counted[False] == layout_flops(state.n_boards, batch.n_tokens, NetConfig(**A8, **kw))["total"]


# --- smoke optimiser steps on synthetic rows (the user's rule: no training runs) --------------------------


@pytest.mark.grad
def test_smoke_steps_with_it_on():
    """Three optimiser steps on synthetic rows with A8 on: every loss finite, a gradient on every A8
    parameter, and every one of them moved."""
    from t5d.engine.netplay import SMALL
    from t5d.train.trainer import Trainer
    from tests.train.test_trainer import _config, _synthetic_rows

    _, _, rows = _synthetic_rows(0)
    net = NetConfig(**SMALL, **A8)
    model = build_model(net, seed=1)
    before = {n: p.detach().clone() for n, p in model.embed.layout.named_parameters()}
    trainer = Trainer(model, net, _config())
    for _ in range(3):
        losses = trainer.step_on(rows)
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
        for name, p in model.embed.layout.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    for name, p in model.embed.layout.named_parameters():
        assert not torch.equal(p.detach(), before[name]), name
