"""Stage N2's attention structure (spec section 6): the masks of B9 and B10 against a
brute-force reference, no empty rows, how B5 meets the masks, typed edge biases (B11) on
exactly the related pairs, board-pair smolgen (B12) depending on the global registers
and on board pairs only, and no inert switch."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, relayout, synthetic_state
from t5d.net.config import ROUTING_TABLE_ROLES, SMOLGEN_FORMS, TYPED_RELATIONS
from t5d.net.rope import head_allocation, rope_tables
from t5d.net.state import (
    RELATION_IDS,
    RELATION_TYPES,
    TOKEN_BOARD_REGISTER,
    TOKEN_GLOBAL_REGISTER,
    TOKEN_PAD,
    TOKEN_SQUARE,
    PositionState,
)
from t5d.net.structure import TYPED_INDEX
from tests.net.test_attention import _b6_lookup, _has_axes, _reference_probs

# --- fixtures --------------------------------------------------------------------------------


def _edge_case_state(seed: int, n_boards: int) -> PositionState:
    """A synthetic state whose edge list adds the awkward cases: an exact duplicate, a
    self-loop, a mutual pair of the same type, and a cross-board edge of every relation."""
    state = synthetic_state(seed, n_boards)
    lay = state.layout
    rng = np.random.default_rng(seed)
    occupied = np.argwhere(state.piece != 0)
    a, b = (int(i) for i in rng.choice(occupied.shape[0], size=2, replace=False))
    tok = [int(occupied[i, 0]) * lay.tokens_per_board + int(occupied[i, 1]) for i in (a, b)]
    extra = [state.edges[0], state.edges[0], [tok[0], tok[0], RELATION_IDS["attack"]]]
    extra += [[tok[0], tok[1], RELATION_IDS["attack"]], [tok[1], tok[0], RELATION_IDS["attack"]]]
    far = lay.square_token(n_boards - 1, 3, 4)
    extra += [[tok[0], far, rel] for rel in range(len(RELATION_TYPES))]
    return replace(state, edges=np.concatenate([state.edges, np.array(extra, dtype=np.int64)]))


def _batch(seed: int = 0, sizes=(3, 1)):
    return collate([_edge_case_state(seed + i, n) for i, n in enumerate(sizes)])


def _model(**flags):
    return build_model(NetConfig(trunk_dtype="fp32", **flags), seed=0).eval()


# --- the brute-force key sets ----------------------------------------------------------------


def reference_allowed(batch, cfg: NetConfig) -> np.ndarray:
    """The documented key sets (``structure.py``), token by token: ``(B, H, L, L)`` bool,
    padding keys excluded."""
    kind, board, valid = batch.kind.numpy(), batch.board.numpy(), batch.valid.numpy()
    xy = batch.coords[..., :2].numpy().round().astype(np.int64)
    bsz, length = kind.shape
    out = np.zeros((bsz, cfg.n_heads, length, length), dtype=bool)
    for b in range(bsz):
        move, rel = defaultdict(set), defaultdict(set)  # query -> keys
        for (s, d, r), ok in zip(batch.edges[b].tolist(), batch.edge_valid[b].tolist(), strict=True):
            name = RELATION_TYPES[r] if ok else None
            if name == "move":
                target, direction = move, cfg.move_graph_direction
            elif name in cfg.relations_head_types:
                target = rel
                direction = cfg.move_graph_direction if cfg.relations_head_direction is None else cfg.relations_head_direction
            else:
                continue
            if direction in ("forward", "symmetric"):
                target[s].add(d)
            if direction in ("reverse", "symmetric"):
                target[d].add(s)
        on_board = (kind[b] == TOKEN_SQUARE) | (kind[b] == TOKEN_BOARD_REGISTER)
        for h, role in enumerate(cfg.head_roles):
            own_board = role == "same_board" and cfg.same_board_registers == "own_board"
            for i in range(length):
                if role == "dense" or kind[b, i] == TOKEN_PAD:
                    keys = np.flatnonzero(valid[b])
                elif kind[b, i] == TOKEN_BOARD_REGISTER and own_board:
                    keys = [i, *np.flatnonzero(on_board & (board[b] == board[b, i])).tolist()]
                elif kind[b, i] in (TOKEN_BOARD_REGISTER, TOKEN_GLOBAL_REGISTER):
                    keys = np.flatnonzero(valid[b]) if cfg.masked_head_registers == "read" else [i]
                elif role in ("move_graph", "relations"):
                    keys = [i, *(move if role == "move_graph" else rel)[i]]
                elif role == "same_board":
                    keys = [i, *np.flatnonzero(on_board & (board[b] == board[b, i])).tolist()]
                elif role == "same_square":
                    same = (kind[b] == TOKEN_SQUARE) & (xy[b, :, 0] == xy[b, i, 0]) & (xy[b, :, 1] == xy[b, i, 1])
                    keys = [i, *np.flatnonzero(same).tolist()]
                else:
                    raise AssertionError(role)
                out[b, h, i, keys] = True
        out[b] &= valid[b][None, None, :]
    return out


#: A routing table whose roles are not contiguous in head order: a role's heads are
#: gathered from across the table, so the by-role paths must put them back in place.
SCRAMBLED_ROLES = ("dense", "same_square", "move_graph", "dense", "relations", "same_board", "dense", "move_graph")

MASK_CONFIGS = {
    "b9_default": dict(),
    "b9_forward_self": dict(move_graph_direction="forward", masked_head_registers="self"),
    "b9_reverse_4heads": dict(move_graph_direction="reverse", move_graph_heads=4),
    "b10_routing": dict(routing_table=True),
    "b10_routing_narrow_self": dict(routing_table=True, relations_head_types=("attack", "defence"), masked_head_registers="self"),
    "b10_routing_forward": dict(routing_table=True, move_graph_direction="forward"),
    "b10_routing_registers": dict(routing_table=True, same_board_registers="registers"),
    "b10_relations_reverse": dict(routing_table=True, relations_head_direction="reverse"),
    "b10_scrambled_table": dict(routing_table=True, rope_head_roles=SCRAMBLED_ROLES),
}


@pytest.mark.parametrize("name", list(MASK_CONFIGS))
def test_masks_are_exactly_the_documented_key_sets(name):
    """B9 / B10: the structure's mask, row by row, against the brute-force key sets - and
    layer 0's attention puts weight on exactly those keys."""
    model = _model(**MASK_CONFIGS[name])
    batch = _batch(seed=1)
    ref = torch.from_numpy(reference_allowed(batch, model.cfg))
    ctx = model.make_context(batch)
    mask = ctx.structure.dense_mask() & batch.valid[:, None, None, :]
    assert torch.equal(mask, ref)
    probs = model.attention_probabilities(batch, layer=0)
    assert torch.equal(probs > 0, ref), "attention reaches exactly the allowed keys"


def test_head_roles():
    assert NetConfig().head_roles == ("move_graph", "move_graph") + ("dense",) * 6, "B9: 2 heads, heads 0 and 1"
    assert NetConfig(move_graph_heads=0).head_roles == ("dense",) * 8
    assert NetConfig(move_graph_heads=4).head_roles[:5] == ("move_graph",) * 4 + ("dense",)
    assert NetConfig(routing_table=True).head_roles == ROUTING_TABLE_ROLES, "B10: section 6's table"
    assert NetConfig(routing_table=True, rope_head_roles=SCRAMBLED_ROLES).head_roles == SCRAMBLED_ROLES
    with pytest.raises(ValueError, match="move_graph_heads"):
        NetConfig(move_graph_heads=9).validate()
    for heads in (0, 4):
        with pytest.raises(ValueError, match="move_graph_heads is not used under routing_table"):
            NetConfig(routing_table=True, move_graph_heads=heads).validate()
    fields = ("move_graph_direction", "masked_head_registers", "rope_roles_source", "same_board_registers")
    for field in (*fields, "relations_head_direction", "smolgen_form"):
        with pytest.raises(ValueError, match=field):
            NetConfig(**{field: "nonsense"}).validate()
    with pytest.raises(ValueError, match="relations_head_types"):
        NetConfig(relations_head_types=("move",)).validate()
    assert NetConfig().relations_direction == "symmetric"
    assert NetConfig(move_graph_direction="forward").relations_direction == "forward", "follows the move graph by default"
    assert NetConfig(move_graph_direction="forward", relations_head_direction="reverse").relations_direction == "reverse"


@pytest.mark.parametrize("registers", ["read", "self"])
def test_same_board_head_keeps_a_per_board_register_on_its_board(registers):
    """Section 6's head 2 is "same board only", and a per-board register is on its board:
    under ``same_board_registers="own_board"`` it reads exactly its board's tokens, as the
    board's squares do, whatever ``masked_head_registers`` says; under ``"registers"`` it
    follows ``masked_head_registers`` (every key, or itself). Global registers, which hold
    no board, follow ``masked_head_registers`` either way."""
    batch = collate([synthetic_state(3, 4)])
    head = ROUTING_TABLE_ROLES.index("same_board")
    board_tokens = torch.arange(66)  # board 0: squares 0..63, registers 64, 65
    rows = {}
    for mode in ("own_board", "registers"):
        model = _model(routing_table=True, masked_head_registers=registers, same_board_registers=mode)
        mask = model.make_context(batch).structure.dense_mask()[0, head] & batch.valid[0]
        rows[mode] = mask
        for square in (0, 37):
            assert torch.equal(torch.nonzero(mask[square]).flatten(), board_tokens), "a square reads its board"
    for register in (64, 65):
        assert torch.equal(torch.nonzero(rows["own_board"][register]).flatten(), board_tokens), register
        expected = batch.valid[0] if registers == "read" else torch.arange(batch.n_tokens) == register
        assert torch.equal(rows["registers"][register], expected), register
    glob = batch.body_len
    for mode in ("own_board", "registers"):
        expected = batch.valid[0] if registers == "read" else torch.arange(batch.n_tokens) == glob
        assert torch.equal(rows[mode][glob], expected), "a global register has no board"


@pytest.mark.parametrize("name", list(MASK_CONFIGS))
@pytest.mark.parametrize("n_boards", [1, 5, 17])
def test_no_row_is_ever_empty(name, n_boards):
    """Every row - valid tokens, registers, padding - keeps at least one allowed key, on
    every head and in any query block."""
    model = _model(**MASK_CONFIGS[name])
    batch = collate([synthetic_state(n_boards, n_boards), synthetic_state(n_boards + 1, max(1, n_boards // 3))])
    structure = model.make_context(batch).structure
    mask = structure.dense_mask() & batch.valid[:, None, None, :]
    assert mask.any(dim=-1).all()
    # Own token always allowed for valid tokens.
    diag = mask.diagonal(dim1=-2, dim2=-1)
    assert torch.equal(diag & batch.valid[:, None, :], batch.valid[:, None, :].expand_as(diag))
    # Rows built per block equal the dense rows (any block boundaries).
    length = batch.n_tokens
    for start, stop in ((0, 7), (7, 70), (70, length - 5), (length - 5, length)):
        rows, _ = structure.rows(start, stop)
        assert torch.equal(rows & batch.valid[:, None, None, :], mask[:, :, start:stop])


def test_masks_follow_the_edge_lists_only():
    """Changing a position's move edges changes the masked heads' rows and nothing else;
    the dense heads are untouched by any edge."""
    model = _model()
    state = synthetic_state(7, 4)
    batch = collate([state])
    move = state.edges[:, 2] == RELATION_IDS["move"]
    fewer = collate([replace(state, edges=state.edges[~move | (np.arange(len(move)) % 2 == 0)])])
    a = model.make_context(batch).structure.dense_mask()
    b = model.make_context(fewer).structure.dense_mask()
    assert not torch.equal(a[:, :2], b[:, :2])
    assert torch.equal(a[:, 2:], b[:, 2:]) and bool(a[:, 2:].all())


# --- B5 and the masks ---------------------------------------------------------------------------


def test_rope_roles_follow_the_table_or_the_masks():
    """One role table for B5 and B10. With the routing table off, ``rope_roles_source``
    decides: "table" keeps N1's roles by head index, "masks" gives each head the role of
    the mask it carries; under the routing table both are the table."""
    assert NetConfig(rope_per_head=True).rope_roles == ROUTING_TABLE_ROLES
    masks = NetConfig(rope_per_head=True, rope_roles_source="masks")
    assert masks.rope_roles == ("move_graph", "move_graph") + ("dense",) * 6
    for source in ("table", "masks"):
        routed = NetConfig(rope_per_head=True, routing_table=True, rope_roles_source=source)
        assert routed.rope_roles == routed.head_roles == ROUTING_TABLE_ROLES
    assert NetConfig(rope_roles_source="masks").rope_roles == ("dense",), "B5 off: one shared allocation"
    axis, freq = rope_tables(masks)
    near = head_allocation("move_graph", masks)
    assert freq[1].tolist() == pytest.approx([f for _, f in near]), "the second move-graph head runs near-NoPE"
    assert not torch.equal(rope_tables(NetConfig(rope_per_head=True))[1][1], freq[1])


@pytest.mark.parametrize(
    "flags",
    [
        dict(rope_per_head=True, routing_table=True),
        dict(rope_per_head=True, rope_roles_source="masks"),
        dict(rope_per_head=True, routing_table=True, global_register_pairs="anchored", board_register_spatial="position_free"),
    ],
    ids=["routed", "masks_source", "routed_anchored_regs_free"],
)
def test_each_head_rotates_by_its_role_under_its_mask(flags):
    """B5 with B9/B10: layer 0's probabilities, head by head, against a reference built
    from the head's own rotary allocation and its brute-force key set."""
    model = _model(**flags)
    cfg = model.cfg
    batch = _batch(seed=2, sizes=(2,))
    probs = model.attention_probabilities(batch, layer=0)
    block = model.trunk.blocks[0]
    q, k, _ = block.attn.qkv(block.norm_attn(model.embed(batch)))
    coords = model.coordinates(batch)
    allowed = torch.from_numpy(reference_allowed(batch, cfg))
    has = _has_axes(cfg, batch)
    for head, role in enumerate(cfg.rope_roles):
        alloc = head_allocation(role, cfg)
        axis = torch.tensor([[a for a, _ in alloc]])
        freq = torch.tensor([[f for _, f in alloc]], dtype=torch.float32)
        ref = _reference_probs(q[:, head : head + 1], k[:, head : head + 1], coords, has, axis, freq, allowed[:, head : head + 1])
        assert torch.allclose(probs[:, head : head + 1].double(), ref, atol=1e-5), (head, role)


# --- B11: typed edge biases -------------------------------------------------------------------


def reference_typed_bias(batch, table: torch.Tensor, directed: bool) -> torch.Tensor:
    """Edge by edge: each distinct typed edge (s, d, type) adds the type's "query is the
    source" scalar at (s, d) and its "query is the target" scalar at (d, s)."""
    bsz, length = batch.kind.shape
    out = torch.zeros(bsz, table.shape[0], length, length, dtype=torch.float64)
    for b in range(bsz):
        seen = set()
        for (s, d, r), ok in zip(batch.edges[b].tolist(), batch.edge_valid[b].tolist(), strict=True):
            if not ok or RELATION_TYPES[r] == "move" or (s, d, r) in seen:
                continue
            seen.add((s, d, r))
            t = TYPED_INDEX[r]
            out[b, :, s, d] += table[:, t, 0].double()
            out[b, :, d, s] += table[:, t, 1 if directed else 0].double()
    return out


@pytest.mark.parametrize("directed", [False, True])
def test_typed_biases_land_on_exactly_the_related_pairs(directed):
    model = _model(typed_edge_bias=True, typed_edge_directed=directed, move_graph_heads=0)
    table = model.typed_edges.table
    assert table.shape == (8, len(TYPED_RELATIONS), 2 if directed else 1)
    batch = _batch(seed=3)
    bias = model.make_context(batch).structure.dense_bias()
    ref = reference_typed_bias(batch, table.detach(), directed)
    assert torch.allclose(bias.double(), ref, atol=1e-6)
    related = ref.abs().sum(dim=1) > 0
    assert torch.equal(bias.abs().sum(dim=1) > 0, related), "non-zero exactly on related pairs"
    # One type's scalar moves exactly the pairs of that type, on that head.
    saved = table.detach().clone()
    for name in ("check", "travel_threat"):
        t = TYPED_RELATIONS.index(name)
        with torch.no_grad():
            table[3, t] += 1.0
        moved = model.make_context(batch).structure.dense_bias() - bias
        with torch.no_grad():
            table.copy_(saved)
        rows = reference_typed_bias(batch, torch.eye(len(TYPED_RELATIONS))[t][None, :, None].expand(1, -1, 2), directed=True)
        assert torch.equal(moved[:, 3] != 0, rows[:, 0] != 0), name
        assert torch.all(moved[:, [h for h in range(8) if h != 3]] == 0)


def test_typed_bias_counts_a_repeated_edge_once_and_a_mutual_relation_twice():
    model = _model(typed_edge_bias=True, move_graph_heads=0)
    state = synthetic_state(4, 2)
    s, d = (int(t) for t in state.edges[0, :2])
    base = replace(state, edges=np.array([[s, d, RELATION_IDS["pin"]]]))
    twice = replace(state, edges=np.array([[s, d, RELATION_IDS["pin"]]] * 3))
    mutual = replace(state, edges=np.array([[s, d, RELATION_IDS["pin"]], [d, s, RELATION_IDS["pin"]]]))
    w = model.typed_edges.table[:, TYPED_RELATIONS.index("pin"), 0]
    b1, b3, bm = (model.make_context(collate([x])).structure.dense_bias()[0] for x in (base, twice, mutual))
    assert torch.equal(b1, b3), "a relation holds once"
    assert torch.allclose(b1[:, s, d], w) and torch.allclose(b1[:, d, s], w)
    assert torch.allclose(bm[:, s, d], 2 * w), "a mutual relation is two directed relations"


def test_typed_biases_reach_masked_heads_only_on_allowed_pairs():
    """On a move-graph head a typed bias outside the move graph is masked away; on a dense
    head it moves the attention."""
    model = _model(typed_edge_bias=True)
    batch = _batch(seed=5, sizes=(2,))
    base = model.attention_probabilities(batch, layer=0)
    with torch.no_grad():
        model.typed_edges.table.add_(3.0)
    moved = model.attention_probabilities(batch, layer=0)
    allowed = torch.from_numpy(reference_allowed(batch, model.cfg))
    assert torch.equal(moved > 0, allowed) and torch.equal(base > 0, allowed)
    assert not torch.allclose(base[:, 2:], moved[:, 2:], atol=1e-4)


# --- B12: board-pair smolgen ------------------------------------------------------------------


def _smolgen_bias(model, batch, h: torch.Tensor) -> torch.Tensor:
    ctx = model.make_context(batch).for_layer(h)
    return ctx.structure.dense_bias()


@pytest.mark.parametrize("form", SMOLGEN_FORMS)
def test_smolgen_bias_is_board_pair_constant_and_skips_global_registers(form):
    model = _model(board_pair_smolgen=True, smolgen_form=form, move_graph_heads=0)
    batch = _batch(seed=6)
    h = torch.randn(batch.batch_size, batch.n_tokens, 256, generator=torch.Generator().manual_seed(6))
    bias = _smolgen_bias(model, batch, h)
    table = model.smolgen(h, model.make_context(batch).structure.smolgen_in)  # (B, H, Nb, Nb)
    for b in range(batch.batch_size):
        board = batch.board[b]
        has = batch.valid[b] & (board >= 0)
        assert torch.all(bias[b][:, ~has] == 0) and torch.all(bias[b][:, :, ~has] == 0), "no board, no smolgen bias"
        idx = board[has]
        expected = table[b][:, idx][:, :, idx]  # (H, n, n): by the two tokens' boards only
        assert torch.allclose(bias[b][:, has][:, :, has], expected, atol=1e-6)
        assert int(idx.max()) < int(batch.n_boards[b])


def _bucket(d: int, exact: int, maximum: int) -> int:
    """B6's bucket of an offset's size, written out: exact below ``exact``, then one bucket
    per doubling up to ``maximum``, the last shared by every larger offset."""
    d = abs(d)
    if d < exact:
        return d
    return exact + min(int(math.floor(math.log2(d / exact))), int(math.floor(math.log2(maximum / exact))))


def _cell(cfg: NetConfig, dp: int, dh: int) -> int:
    """The relative cell of a board pair (key minus query): signed Delta-time bucket times
    the Delta-timeline buckets, plus the Delta-timeline bucket."""
    n_time = _bucket(10**9, cfg.bias_time_exact, cfg.bias_time_max) + 1
    n_timeline = _bucket(10**9, cfg.bias_timeline_exact, cfg.bias_timeline_max) + 1
    t = _bucket(dh, cfg.bias_time_exact, cfg.bias_time_max)
    signed = t if dh >= 0 else n_time - 1 + t
    return signed * n_timeline + _bucket(dp, cfg.bias_timeline_exact, cfg.bias_timeline_max)


@pytest.mark.parametrize(("regs", "globals_"), [(2, 16), (0, 16), (2, 0)], ids=["registers", "no-board-registers", "no-globals"])
@pytest.mark.parametrize("form", SMOLGEN_FORMS)
def test_smolgen_follows_its_formula(form, regs, globals_):
    """``z = silu(W_z c + b_z)``, ``c`` the mean of the global registers (of every square of
    the position in the arm without them); relative cells ``S_h(b1, b2) = z . W_h[cell] /
    sqrt(32)``; bilinear ``S_h(b1, b2) = (A_h u_b1)^T diag(z) (B_h u_b2) / sqrt(32)``,
    ``u_b`` from the mean of board b's per-board registers (of its squares in the arm
    without them) - everything by hand, in FP64, on every board pair."""
    cfg = NetConfig(trunk_dtype="fp32", board_pair_smolgen=True, smolgen_form=form, regs_per_board=regs, global_regs=globals_)
    model = build_model(cfg, seed=0)
    state = relayout(synthetic_state(7, 9, n_timelines=3), regs, globals_)
    batch = collate([state])
    n, tpb = state.n_boards, 64 + regs
    h = torch.randn(1, batch.n_tokens, 256, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    sm = model.smolgen.double()
    table = sm(h, model.make_context(batch).structure.smolgen_in)[0]
    squares = torch.cat([h[0, b * tpb : b * tpb + 64] for b in range(n)])
    c = h[0, batch.body_len :].mean(0) if globals_ else squares.mean(0)
    z = torch.nn.functional.silu(sm.context(c))
    p, t = state.timeline.tolist(), state.half_turn.tolist()
    for b1, b2 in itertools.product(range(n), repeat=2):
        if form == "relative_cells":
            cell = _cell(cfg, p[b2] - p[b1], t[b2] - t[b1])
            expected = sm.cells[:, cell] @ z / math.sqrt(32)
        else:
            r1, r2 = ((h[0, b * tpb + 64 : b * tpb + tpb] if regs else h[0, b * tpb : b * tpb + 64]).mean(0) for b in (b1, b2))
            u1, u2 = sm.board(r1), sm.board(r2)
            expected = torch.stack([(sm.left[k] @ u1 * z * (sm.right[k] @ u2)).sum() for k in range(8)]) / math.sqrt(32)
        assert torch.allclose(table[:, b1, b2], expected, atol=1e-9), (b1, b2)


def test_relative_cells_are_full_rank_and_the_bilinear_form_is_not():
    """Section 6: board-pair smolgen is "full-rank ... the freedom that makes smolgen
    smolgen", where a rank-r generated bias is "the d_head lever in a costume". At 40
    boards the relative-cell table has rank 40 on every head; the bilinear form's rank is
    bounded by the latent, 32."""
    batch = collate([synthetic_state(21, 40)])
    h = torch.randn(1, batch.n_tokens, 256, generator=torch.Generator().manual_seed(21), dtype=torch.float64)
    ranks = {}
    for form in SMOLGEN_FORMS:
        model = _model(board_pair_smolgen=True, smolgen_form=form)
        table = model.smolgen.double()(h, model.make_context(batch).structure.smolgen_in)[0]  # FP64: an exact rank
        ranks[form] = [int(torch.linalg.matrix_rank(table[k])) for k in range(8)]
    assert ranks["relative_cells"] == [40] * 8, ranks
    assert max(ranks["board_bilinear"]) <= 32, ranks


@pytest.mark.parametrize("form", SMOLGEN_FORMS)
def test_smolgen_reads_the_global_registers(form):
    """Both forms are generated from the global registers; the bilinear form also reads
    each board's per-board registers, the relative-cell form nothing else; squares enter
    neither."""
    model = _model(board_pair_smolgen=True, smolgen_form=form, move_graph_heads=0)
    batch = _batch(seed=8, sizes=(4,))
    gen = torch.Generator().manual_seed(8)
    h = torch.randn(1, batch.n_tokens, 256, generator=gen)
    base = _smolgen_bias(model, batch, h)

    def moved(rows: torch.Tensor) -> bool:
        h2 = h.clone()
        h2[0, rows] += torch.randn(int(rows.sum()), 256, generator=gen)
        return not torch.allclose(_smolgen_bias(model, batch, h2), base, atol=1e-6)

    kind = batch.kind[0]
    assert moved(kind == TOKEN_GLOBAL_REGISTER), "generated from the global registers"
    assert moved(kind == TOKEN_BOARD_REGISTER) == (form == "board_bilinear"), "boards enter through their registers (bilinear)"
    assert not moved(kind == TOKEN_SQUARE), "squares do not enter"


def test_smolgen_needs_the_layer_input():
    model = _model(board_pair_smolgen=True)
    ctx = model.make_context(_batch(seed=9, sizes=(2,)))
    with pytest.raises(RuntimeError, match="for_layer"):
        ctx.structure.dense_bias()


def test_every_bias_adds_up():
    """B6, B11 and B12 together: a layer's bias is the B6 table lookup, plus the typed
    edges', plus that layer's smolgen table at the two tokens' boards - each built
    independently here."""
    model = _model(binned_bias=True, typed_edge_bias=True, board_pair_smolgen=True, move_graph_heads=0)
    batch = _batch(seed=10, sizes=(3, 2))
    h = torch.randn(batch.batch_size, batch.n_tokens, 256, generator=torch.Generator().manual_seed(10))
    ctx = model.make_context(batch)
    bias = ctx.for_layer(h).structure.dense_bias()
    b6 = _b6_lookup(model, batch, model.coordinates(batch))
    typed = reference_typed_bias(batch, model.typed_edges.table.detach(), directed=False)
    table = model.smolgen(h, ctx.structure.smolgen_in)
    smolgen = torch.zeros_like(bias)
    for b in range(batch.batch_size):
        has = batch.valid[b] & (batch.board[b] >= 0)
        idx = batch.board[b][has]
        block = torch.zeros(8, int(has.sum()), batch.n_tokens)
        block[:, :, has] = table[b][:, idx][:, :, idx]
        smolgen[b][:, has] = block
    assert torch.allclose(bias.double(), (b6 + smolgen).double() + typed, atol=1e-5)


# --- no inert switch ----------------------------------------------------------------------------

#: Each pair of configurations differs in one switch of stage N2 or of the trunk (F5, F6);
#: the network must differ. (``head_activation`` changes the value head alone - test_trunk.py.)
SWITCHES = {
    "norm_placement pre -> post": (dict(), dict(norm_placement="post")),
    "qkv_bias": (dict(), dict(qkv_bias=True)),
    "out_ffn_bias": (dict(), dict(out_ffn_bias=False)),
    "ffn_activation mish -> squared_relu": (dict(), dict(ffn_activation="squared_relu")),
    "move_graph_heads 0 -> 2": (dict(move_graph_heads=0), dict()),
    "move_graph_heads 2 -> 4": (dict(), dict(move_graph_heads=4)),
    "routing_table": (dict(), dict(routing_table=True)),
    "direction symmetric -> forward": (dict(), dict(move_graph_direction="forward")),
    "direction forward -> reverse": (dict(move_graph_direction="forward"), dict(move_graph_direction="reverse")),
    "registers read -> self": (dict(), dict(masked_head_registers="self")),
    "relations_head_types": (dict(routing_table=True), dict(routing_table=True, relations_head_types=("attack", "defence"))),
    "rope_roles_source": (dict(rope_per_head=True), dict(rope_per_head=True, rope_roles_source="masks")),
    "typed_edge_bias": (dict(), dict(typed_edge_bias=True)),
    "typed_edge_directed": (dict(typed_edge_bias=True), dict(typed_edge_bias=True, typed_edge_directed=True)),
    "board_pair_smolgen": (dict(), dict(board_pair_smolgen=True)),
    "smolgen_form": (dict(board_pair_smolgen=True), dict(board_pair_smolgen=True, smolgen_form="board_bilinear")),
    "same_board_registers": (dict(routing_table=True), dict(routing_table=True, same_board_registers="registers")),
    "relations_head_direction": (dict(routing_table=True), dict(routing_table=True, relations_head_direction="forward")),
}


@pytest.mark.parametrize("name", list(SWITCHES))
def test_every_switch_changes_the_network(name):
    a_flags, b_flags = SWITCHES[name]
    batch = collate([synthetic_state(11, 5)])
    a = _model(**a_flags)(batch)
    b = _model(**b_flags)(batch)
    assert not torch.allclose(a.value_logits, b.value_logits, atol=1e-4), name
    assert not torch.allclose(a.policy_logits, b.policy_logits, atol=1e-4), name


@pytest.mark.parametrize(("module", "form"), [("typed_edges", "relative_cells"), *(("smolgen", f) for f in SMOLGEN_FORMS)])
def test_every_new_parameter_tensor_reaches_the_outputs(module, form):
    model = _model(typed_edge_bias=True, board_pair_smolgen=True, typed_edge_directed=True, smolgen_form=form)
    batch = collate([synthetic_state(12, 6)])
    base = model(batch)
    gen = torch.Generator().manual_seed(12)
    for name, p in getattr(model, module).named_parameters():
        saved = p.detach().clone()
        with torch.no_grad():
            p.add_(torch.randn(p.shape, generator=gen) * 0.5)
        out = model(batch)
        with torch.no_grad():
            p.copy_(saved)
        assert not torch.allclose(out.value_logits, base.value_logits, atol=1e-4), f"{module}.{name}"
        assert not torch.allclose(out.board_logits, base.board_logits, atol=1e-4), f"{module}.{name}"


@pytest.mark.parametrize(
    "flags",
    [dict(), dict(routing_table=True), dict(norm_placement="post"), dict(ffn_activation="squared_relu")],
    ids=["b9", "routing", "post-norm", "squared-relu"],
)
def test_attention_implementations_are_one_function(flags):
    """``attn_impl`` changes how, never what: the one switch that must be inert - under
    ablation 6y's arms (F5's post-norm, F6's squared ReLU) too."""
    batch = collate([synthetic_state(13, 4), synthetic_state(14, 2)])
    outs = [
        build_model(NetConfig(trunk_dtype="fp32", attn_impl=impl, typed_edge_bias=True, board_pair_smolgen=True, **flags), seed=1)(batch)
        for impl in ("block_sparse", "chunked", "sdpa", "explicit", "dense")
    ]
    for out in outs[1:]:
        assert torch.allclose(out.value_logits, outs[0].value_logits, atol=1e-4)
