"""Input embeddings (A2, A3, B8, section 5.3): what reaches which token."""

from __future__ import annotations

import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net.state import TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER, TOKEN_SQUARE


def _embed():
    model = build_model(NetConfig(trunk_dtype="fp32"), seed=0)
    batch = collate([synthetic_state(31, 4)])
    return model.embed, batch


def _changed_tokens(embed, batch, mutate) -> torch.Tensor:
    base = embed(batch)
    for name, value in mutate.items():
        setattr(batch, name, value)
    return (embed(batch) - base).abs().amax(-1)[0] > 0


def test_board_features_reach_every_token_of_their_board_and_no_other():
    embed, batch = _embed()
    board1 = (batch.board == 1)[0]
    kinds = batch.kind[0]
    assert (kinds[board1] == TOKEN_BOARD_REGISTER).sum() == 2
    for name in ("is_head", "active", "stm"):
        flipped = getattr(batch, name).clone()
        flipped[0, board1] = 1 - flipped[0, board1]
        changed = _changed_tokens(*_embed(), {name: flipped})
        assert torch.equal(changed, board1), name
    dt = batch.dt.clone()
    dt[0, board1] += 2.0
    assert torch.equal(_changed_tokens(*_embed(), {"dt": dt}), board1)


def test_delta_t_embedding_distinguishes_the_present():
    embed, _ = _embed()
    feats = embed.dt_features(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert feats[:, 2].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0], "the binary present flag is Delta-t's zero"
    out = embed.dt_out(torch.nn.functional.gelu(embed.dt_in(feats)))
    assert torch.cdist(out, out).fill_diagonal_(1.0).min() > 1e-3


def test_square_features_only_reach_squares():
    embed, batch = _embed()
    piece = batch.piece.clone()
    sq = (batch.kind == TOKEN_SQUARE)[0]
    target = int(torch.nonzero(sq)[5])
    piece[0, target] = (piece[0, target] % 12) + 1
    owner = batch.owner.clone()
    owner[0, target] = 1 + (owner[0, target] % 2)
    changed = _changed_tokens(embed, batch, {"piece": piece, "owner": owner})
    assert changed.nonzero().flatten().tolist() == [target]


def test_registers_get_type_and_not_a_square_embeddings():
    embed, batch = _embed()
    x = embed(batch)[0]
    kinds = batch.kind[0]
    glob = x[kinds == TOKEN_GLOBAL_REGISTER]
    assert glob.shape[0] == 16 and torch.pdist(glob).min() > 1e-3, "each global register is distinct"
    # Global registers hold no board: identical whatever the boards carry - given the acting
    # player (register D8, stage N5: its embedding is added to every global register - M20) and
    # the turn-state counts (register A6, likewise).
    other = collate([synthetic_state(77, 9)])
    acting = int(batch.acting_player[0])
    if int(other.acting_player[0]) != acting:
        other.acting_player = torch.tensor([acting])
    other.turn_state = batch.turn_state.clone()
    assert torch.allclose(embed(other)[0][other.kind[0] == TOKEN_GLOBAL_REGISTER], glob)
    flipped = collate([synthetic_state(31, 4)])
    flipped.acting_player = 1 - flipped.acting_player
    shift = embed.acting.weight[1 - acting] - embed.acting.weight[acting]
    assert torch.allclose(embed(flipped)[0][kinds == TOKEN_GLOBAL_REGISTER], glob + shift, atol=1e-6)
    assert torch.equal(embed(flipped)[0][kinds != TOKEN_GLOBAL_REGISTER], x[kinds != TOKEN_GLOBAL_REGISTER])
    # Per-board register slots differ from each other and from squares.
    breg = x[kinds == TOKEN_BOARD_REGISTER]
    assert (breg[0] - breg[1]).abs().max() > 1e-3
    expected = embed.board_register.weight[0] + embed.not_square
    board0 = embed.head_flag(batch.is_head[0, 64]) + embed.active_flag(batch.active[0, 64]) + embed.stm_flag(batch.stm[0, 64])
    dt0 = embed.dt_out(torch.nn.functional.gelu(embed.dt_in(embed.dt_features(batch.dt[0, 64]))))
    assert torch.allclose(x[64], expected + board0 + dt0, atol=1e-6)


def test_with_no_global_registers_the_acting_player_is_on_the_board_tokens():
    """Register D8 with no global registers (an ablation-3 arm; a placeholder, spec issue 225):
    the acting player's embedding on every square and per-board register, by hand - and under
    the block-causal scope on the head boards' alone, so no frozen board's input depends on it."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, global_regs=0), seed=0)
    embed = model.embed
    batch = collate([synthetic_state(31, 9, n_timelines=3, global_regs=0)])
    flipped = collate([synthetic_state(31, 9, n_timelines=3, global_regs=0)])
    flipped.acting_player = 1 - flipped.acting_player
    acting = int(batch.acting_player[0])
    shift = embed.acting.weight[1 - acting] - embed.acting.weight[acting]
    kinds = batch.kind[0]
    on_board = (kinds == TOKEN_SQUARE) | (kinds == TOKEN_BOARD_REGISTER)
    head = on_board & (batch.is_head[0] == 1)
    assert bool(on_board.all()) and bool(head.any()) and not bool(head.all())  # no global register, frozen boards too
    for block_causal, carriers in ((False, on_board), (True, head)):
        before = embed(batch, block_causal=block_causal)[0]
        after = embed(flipped, block_causal=block_causal)[0]
        assert torch.allclose(after[carriers], before[carriers] + shift, atol=1e-6), block_causal
        assert torch.equal(after[~carriers], before[~carriers]), block_causal
    # The model passes the scope: every board token under the dense one, the heads' under the other.
    assert torch.equal(model.embed_tokens(batch, "dense"), embed(batch))
    assert torch.equal(model.embed_tokens(batch, "block_causal"), embed(batch, frozen_intrinsic=True, block_causal=True))
