"""Register A7, perspective canonicalisation (spec section 2.2), and section 15.2's row for it:

    "A position and its colour-swapped mirror (ranks mirrored, timelines negated, owners and
    acting player swapped) give identical canonical inputs and outputs | Perspective
    canonicalisation is exact | A7"

Read as spec issue 304 says (proposed edit 202): A7 keeps "the acting colour ... a global
input" (section 2.2), and the mirror swaps it, so the two canonical inputs are identical *but
for that input*, and the outputs, mapped back, are identical when it is equalised - here by
tying the acting-colour embedding's two rows (``tie``), a test hook that changes nothing else.
Untied, the input is live and the outputs differ.

The mirror (``state.mirror``) swaps the colours, mirrors every board's ranks, negates the
timelines by A7's own map (``-p``, or ``-1 - p`` in a +-0 variant) and swaps the acting player;
every board moves one half-turn later so that its side to move is the other colour, a shift the
network cannot see (time is read relative to the present - section 4; spec issue 227). Its
squares sit at mirrored storage indices, so token-indexed comparisons go through
``mirror_tokens``; the move lists keep their order, so policy logit ``i`` of the mirror scores
the mirror of move ``i`` - A7's "outputs mapped back".

Section 15.2's protocol: random weights at realistic initialisation scale, several seeds, the
adversarial constructions - many timelines, deep travel, duplicated boards, a +-0 variant,
every variant's board size (7 x 7 to Global Warming's 1 x 1), fairy pieces, positions past the
shift with submit legal, masked and absent, a mid-turn state's turn-state counts (A6: the acting
player's, so the mirror keeps them), no global registers, far-apart boards - and the
flagged attention structure (B5, B10, B11, B12, B17, B15 off), in the dense scope and the
block-causal one, and through the cached path (section 15.2's first row must stay green: frozen
K/V are keyed by perspective, spec issue 303) - and with register A8's board-layout embedding on
(its layout read in canonical (x, y) order: ``test_the_mirror_with_the_board_layout_embedding``
and the cached path's). Tolerance: FP32 1e-5 (the mirror reorders each
board's tokens, so floating-point sums run in another order); the BF16 default no further from
its mirror than from its own FP32 computation. Real engine positions, the engine's own 5DFEN
mirror against ``mirror`` of the conversion: ``tests/engine/test_perspective_engine.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, mirror, permute_boards, synthetic_state, translate
from t5d.net.cache import CacheConfig, CachedEvaluator, reference_output, viewed_by
from t5d.net.heads import SQUARE_LABELS, SQUARE_LABELS_BY_COLOUR, square_labels_by_colour
from t5d.net.state import (
    OWNER_NONE,
    OWNER_OURS,
    OWNER_THEIRS,
    RELATION_IDS,
    TOKEN_BOARD_REGISTER,
    TOKEN_GLOBAL_REGISTER,
    TOKEN_SQUARE,
    canonical_rank,
    canonical_timeline,
    mirror_tokens,
)
from tests.net.test_d8 import shifted
from tests.net.test_displacement import far_state

SEEDS = (0, 1, 2)
ATOL = 1e-5  # FP32: the mirror changes only the order of floating-point sums

#: Section 15.2's adversarial constructions, each a position from a seed.
CONSTRUCTIONS = {
    "standard": lambda s: synthetic_state(s, 12, n_timelines=3),
    "many_timelines": lambda s: synthetic_state(s, 22, n_timelines=11),
    "deep_travel": lambda s: synthetic_state(s, 16, n_timelines=6, deep_travel_prob=1.0),
    "duplicated": lambda s: synthetic_state(s, 14, n_timelines=4, duplicate_prob=0.8),
    "even_start": lambda s: synthetic_state(s, 14, n_timelines=5, even_start=True),
    "fairy": lambda s: synthetic_state(s, 12, n_timelines=3, fairy_prob=0.7),
    "shifted_submit_legal": lambda s: shifted(synthetic_state(s, 12, n_timelines=3), legal=True),
    "shifted_submit_masked": lambda s: shifted(synthetic_state(s, 12, n_timelines=4), legal=False),
    "shifted_submit_absent": lambda s: shifted(synthetic_state(s, 10, n_timelines=3, even_start=True), legal=None),
    "mid_turn_counts": lambda s: replace(synthetic_state(s, 12, n_timelines=4), moves_made=3, optional_playable=0),
    "size_7": lambda s: synthetic_state(s, 8, n_timelines=3, width=7, height=7),
    "size_6": lambda s: synthetic_state(s, 8, n_timelines=3, width=6, height=6),
    "size_5": lambda s: synthetic_state(s, 10, n_timelines=4, width=5, height=5, deep_travel_prob=1.0),
    "size_4": lambda s: synthetic_state(s, 10, n_timelines=4, width=4, height=4, duplicate_prob=0.6),
    "size_3": lambda s: synthetic_state(s, 8, n_timelines=3, width=3, height=3),
    "far_size_1": lambda s: far_state(s, size=1),
    "far": far_state,
    "registers_4_0": lambda s: synthetic_state(s, 10, n_timelines=3, regs_per_board=4, global_regs=0),
    "registers_0_16": lambda s: synthetic_state(s, 10, n_timelines=3, regs_per_board=0),
}

#: The attention structure's flags (the v1 and the arms the network can be built with).
FLAG_SETS = {
    "v1": dict(),
    "b5_b10_b11": dict(rope_per_head=True, routing_table=True, typed_edge_bias=True),
    "b12": dict(board_pair_smolgen=True),
    "b17": dict(shaw_key_term=True, shaw_init_std=1.0),
    "no_table_pointer_submit": dict(displacement_bias=False, submit_head="pointer"),
    "d4_reply_d9": dict(opp_reply=True, optimistic_policy=True),
    "a8_centre": dict(board_layout_embedding=True, layout_padding="centre"),
    "d2_leave_registers": dict(leave_pool="registers"),  # D2's flat head's other pool (issue 241)
}

VALUE_HEADS = ("value_logits", "wdl_logits", "short_term", "value_error", "moves_left_log", "effort_logits", "submit_raw")


def tie(model):
    """The test hook that equalises A7's acting-colour input: the embedding's two rows made one
    (the white row), so the colour reaches nothing. Nothing else changes."""
    # Tied, anything routed through the acting-colour embedding - not only the colour itself -
    # is equalised too and could no longer differ between a position and its mirror. What
    # guards that route is the untied carrier assertion in
    # test_mirror_has_identical_canonical_inputs: untied, the embeddings differ exactly at the
    # tokens the acting colour enters and nowhere else.
    with torch.no_grad():
        model.embed.acting.weight[1] = model.embed.acting.weight[0]
    return model


def _model(seed: int, scope: str = "dense", tied: bool = True, **flags):
    cfg = NetConfig(trunk_dtype=flags.pop("trunk_dtype", "fp32"), n_layers=flags.pop("n_layers", 2), attn_scope=scope, **flags)
    model = build_model(cfg, seed=seed).eval()
    return tie(model) if tied else model


def _pair(construction: str, seed: int):
    s = CONSTRUCTIONS[construction](1000 * (seed + 1) + 17)
    return s, mirror(s)


def _config_for(state) -> dict:
    return dict(regs_per_board=state.regs_per_board, global_regs=state.global_regs)


# --- the transform ---------------------------------------------------------------------------------


def test_the_index_mapping():
    """Section 2.2: "+n <-> -n, and in +-0 variants +0 <-> -0 - i -> -1 - i in section 12.6's
    index mapping" (-0 is -1, +0 is 0); ranks r -> size + 1 - r. White's view is the identity."""
    p = np.array([-3, -2, -1, 0, 1, 2])
    assert canonical_timeline(p, 0).tolist() == p.tolist()
    assert canonical_timeline(p, 1).tolist() == [3, 2, 1, 0, -1, -2]  # odd: +n <-> -n, 0 stays
    assert canonical_timeline(p, 1, even_start=True).tolist() == [2, 1, 0, -1, -2, -3]  # -2 <-> +1 ... -0 <-> +0
    for size in (1, 3, 4, 5, 6, 7, 8):
        y = np.arange(size)
        assert canonical_rank(y, size, 0).tolist() == y.tolist()
        assert canonical_rank(y, size, 1).tolist() == (size - 1 - y).tolist()  # rank r -> size + 1 - r


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_the_mirror_is_a_valid_position_and_twice_a_translation(construction):
    """The mirror validates as a position (lineage one half-turn back, heads, the action), keeps
    the shift and the submit flags, and the mirror of the mirror is the position two half-turns
    later - to which the network is exactly invariant (``test_invariance.py``)."""
    for seed in SEEDS:
        s, m = _pair(construction, seed)
        assert m.acting_player == 1 - s.acting_player and m.shifted == s.shifted
        assert (m.submit_possible, m.submit_legal, m.even_start) == (s.submit_possible, s.submit_legal, s.even_start)
        assert np.array_equal(m.side_to_move, 1 - s.side_to_move) and m.h_present == s.h_present + 1
        twice, later = mirror(m), translate(s, dh=2)
        for f in dataclasses.fields(s):
            a, b = getattr(twice, f.name), getattr(later, f.name)
            assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b, f.name


# --- (a) identical canonical inputs, but for the acting colour ---------------------------------------


#: Per-token fields of a ``Batch``, compared through the token map; per-board and per-position ones
#: directly. ``coords`` and ``acting_player`` are checked by hand.
TOKEN_FIELDS = ("piece", "owner", "moved", "kind", "reg_index", "board", "is_head", "active", "stm", "dt", "valid")
DIRECT_FIELDS = (
    "n_body",
    "n_boards",
    "board_valid",
    "move_valid",
    "opp_move_valid",
    "edge_valid",
    "board_parent",
    "board_creation",
    "board_source",
    "board_head",
    "has_creation",
    "submit_possible",
    "submit_legal",
    "turn_state",
)


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_mirror_has_identical_canonical_inputs(construction):
    """Every field of the canonical input of S equals the mirror's, bit for bit, at the mirrored
    token - owners, pieces, has-moved, side to move, the board flags, Delta-t, the recentred
    coordinates, the moves and edges, the lineage, the submit flags - but the acting colour,
    which the mirror swaps. The raw time coordinate is one half-turn later on every board (the
    mirror's shift), which the recentred coordinates do not see; and the token embeddings, under
    either scope, are equal bit for bit once the acting colour is equalised, and differ exactly
    where it enters - the global registers, or with none the board tokens (spec issue 225)."""
    for seed in SEEDS:
        s, m = _pair(construction, seed)
        bs, bm = collate([s]), collate([m])
        perm = torch.from_numpy(mirror_tokens(s.layout))  # token t of S is token perm[t] of the mirror
        names = {f.name for f in dataclasses.fields(bs)}
        checked = set(TOKEN_FIELDS) | set(DIRECT_FIELDS) | {"coords", "coord_ref", "acting_player", "moves", "opp_moves", "edges"}
        checked |= {"body_len", "regs_per_board", "global_regs", "layouts", "piece_centric"}
        assert names == checked, names ^ checked  # every field of the input is compared below
        for name in TOKEN_FIELDS:
            assert torch.equal(getattr(bm, name)[0][perm], getattr(bs, name)[0]), name
        for name in DIRECT_FIELDS:
            assert torch.equal(getattr(bm, name), getattr(bs, name)), name
        for name in ("body_len", "regs_per_board", "global_regs", "layouts", "piece_centric"):
            assert getattr(bm, name) == getattr(bs, name), name
        for name in ("moves", "opp_moves", "edges"):
            a, b = getattr(bs, name), getattr(bm, name)
            assert torch.equal(b[..., :2], torch.where(a[..., :2] >= 0, perm[a[..., :2]], a[..., :2])), name
            assert torch.equal(b[..., 2], a[..., 2]), name
        on_board = (bs.kind[0] == TOKEN_SQUARE) | (bs.kind[0] == TOKEN_BOARD_REGISTER)
        cs, cm = bs.coords[0], bm.coords[0][perm]
        assert torch.equal(cm[:, :3], cs[:, :3]), "file, canonical rank, canonical timeline"
        assert torch.equal(cm[on_board, 3], cs[on_board, 3] + 1) and torch.equal(cm[~on_board, 3], cs[~on_board, 3])
        assert torch.equal(bm.coord_ref, bs.coord_ref + torch.tensor([[0.0, 1.0]])), "the canonical p_ref; the present one later"
        assert bm.acting_player.tolist() == [1 - int(bs.acting_player[0])]
        # The owners are ours and theirs: the acting player's pieces are ours in both.
        ours = s.colour == 1 + s.acting_player
        assert np.array_equal(s.owner == OWNER_OURS, ours) and np.array_equal(s.owner == OWNER_THEIRS, (s.colour > 0) & ~ours)
        assert np.array_equal(s.owner == OWNER_NONE, s.colour == 0)
        for scope in ("dense", "block_causal"):
            model = _model(seed, scope, tied=False, n_layers=1, **_config_for(s))
            assert torch.equal(model.coordinates(bm)[0][perm], model.coordinates(bs)[0]), scope
            es, em = model.embed_tokens(bs, scope)[0], model.embed_tokens(bm, scope)[0][perm]
            differs = (es - em).abs().amax(-1) > 0
            if s.global_regs:
                carriers = bs.kind[0] == TOKEN_GLOBAL_REGISTER
            else:
                carriers = on_board if scope == "dense" else on_board & (bs.is_head[0] == 1)
            assert torch.equal(differs, carriers), scope  # the acting colour is the only difference
            tie(model)
            assert torch.equal(model.embed_tokens(bm, scope)[0][perm], model.embed_tokens(bs, scope)[0]), scope


# --- (b) identical outputs, mapped back, with the acting colour equalised -----------------------------


def assert_mirrored(a, b, perm: torch.Tensor, acting: tuple[int, int], atol: float = ATOL) -> float:
    """The outputs of a position (``a``) and of its mirror (``b``) agree, mapped back: the
    global heads as they are (the value is the acting player's - D8), policy and reply logit
    ``i`` (move ``i`` of the mirror is the mirror of move ``i``), the per-board logits (board
    order kept) and the per-square logits at the mirrored token, ours and theirs - and mapped
    back to colours (``acting``: the two positions' acting players), white at a square of S is
    black at its mirror. Returns the largest difference."""
    worst = 0.0

    def close(x, y, what):
        nonlocal worst
        d = float((x - y).abs().max()) if x.numel() else 0.0
        worst = max(worst, d)
        assert d <= atol, (what, d)

    for name in VALUE_HEADS:
        close(getattr(a, name), getattr(b, name), name)
    assert torch.equal(torch.isinf(a.submit_logits), torch.isinf(b.submit_logits))
    assert torch.equal(a.move_valid, b.move_valid) and torch.equal(a.opp_move_valid, b.opp_move_valid)
    close(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], "policy")
    close(a.soft_policy_logits[a.move_valid], b.soft_policy_logits[b.move_valid], "soft policy")  # D4
    live = torch.isfinite(a.soft_submit_logits)
    assert torch.equal(live, torch.isfinite(b.soft_submit_logits))
    close(a.soft_submit_logits[live], b.soft_submit_logits[live], "soft submit")
    if a.optimistic_logits is not None:  # D9
        close(a.optimistic_logits[a.move_valid], b.optimistic_logits[b.move_valid], "optimistic policy")
    if a.opp_reply_logits is not None:
        close(a.opp_reply_logits[a.opp_move_valid], b.opp_reply_logits[b.opp_move_valid], "reply")
    close(a.action_log_probs()[a.action_valid()], b.action_log_probs()[b.action_valid()], "actions")
    close(a.board_logits, b.board_logits, "board")
    assert torch.equal(a.board_valid, b.board_valid)
    close(a.leave_logits[a.board_valid], b.leave_logits[b.board_valid], "leave")  # D2's flat head, board order kept
    mask = a.square_mask[0]
    close(a.square_logits[0][mask], b.square_logits[0][perm][mask], "square")
    colour_a = square_labels_by_colour(a.square_logits, torch.tensor([acting[0]]))[0]
    colour_b = square_labels_by_colour(b.square_logits, torch.tensor([acting[1]]))[0][perm]
    for i, name in enumerate(SQUARE_LABELS_BY_COLOUR):
        j = SQUARE_LABELS_BY_COLOUR.index(name.replace("_white", "_tmp").replace("_black", "_white").replace("_tmp", "_black"))
        close(colour_a[mask][:, i], colour_b[mask][:, j], name)
    return worst


def _outputs(model, state):
    with torch.no_grad():
        return model(collate([state]))


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_mirror_has_identical_outputs(construction, scope):
    """Section 15.2's A7 row on the network's outputs, every head, the v1, three seeds."""
    for seed in SEEDS:
        s, m = _pair(construction, seed)
        model = _model(seed, scope, **_config_for(s))
        assert_mirrored(_outputs(model, s), _outputs(model, m), torch.from_numpy(mirror_tokens(s.layout)), _acting(s, m))


def _acting(s, m) -> tuple[int, int]:
    return int(s.acting_player), int(m.acting_player)


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
@pytest.mark.parametrize("flags", list(FLAG_SETS))
def test_mirror_has_identical_outputs_under_every_attention_flag(flags, scope):
    """The same under the attention structure's flags (B5, B10, B11, B12, B17; B15 off with the
    pointer submit readout), on the constructions that stress them."""
    for i, construction in enumerate(("many_timelines", "even_start", "shifted_submit_legal", "size_5")):
        s, m = _pair(construction, i)
        model = _model(i, scope, **FLAG_SETS[flags], **_config_for(s))
        assert_mirrored(_outputs(model, s), _outputs(model, m), torch.from_numpy(mirror_tokens(s.layout)), _acting(s, m))


def test_mirror_at_realistic_scale_bf16():
    """The shipped configuration - BF16 trunk, every layer - on larger positions: the mirror is
    no further from the position than the position's BF16 computation is from its FP32 one
    (the tolerance matched to the arithmetic: the mirror reorders each board's sums); and the
    FP32 network at the same depth within 1e-5."""
    for seed, construction in enumerate(("many_timelines", "even_start", "shifted_submit_masked")):
        s, m = _pair(construction, seed)
        perm = torch.from_numpy(mirror_tokens(s.layout))
        bf16 = tie(build_model(NetConfig(), seed=seed).eval())
        fp32 = tie(build_model(NetConfig(trunk_dtype="fp32"), seed=seed).eval())
        assert_mirrored(_outputs(fp32, s), _outputs(fp32, m), perm, _acting(s, m))
        same = torch.arange(s.n_tokens)
        precision = max(assert_mirrored(_outputs(bf16, x), _outputs(fp32, x), same, _acting(x, x), atol=float("inf")) for x in (s, m))
        assert_mirrored(_outputs(bf16, s), _outputs(bf16, m), perm, _acting(s, m), atol=2 * precision)


def test_board_order_does_not_matter_to_the_mirror():
    """Negating the timelines reverses the engine's ascending board order: the mirror stored in
    the engine's order gives the same outputs, the per-board ones permuted."""
    s, m = _pair("many_timelines", 0)
    order = np.lexsort((m.half_turn, m.timeline))
    assert not np.array_equal(order, np.arange(m.n_boards))
    model = _model(0)
    a, b = _outputs(model, s), _outputs(model, permute_boards(m, order))
    for name in VALUE_HEADS:
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=ATOL), name
    assert torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=ATOL)
    assert torch.allclose(a.board_logits[0, torch.from_numpy(order)], b.board_logits[0], atol=ATOL)
    assert torch.allclose(a.leave_logits[0, torch.from_numpy(order)], b.leave_logits[0], atol=ATOL)  # D2's flat head


# --- (c) the acting colour is live ---------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["dense", "block_causal"])
@pytest.mark.parametrize("construction", ["standard", "even_start", "shifted_submit_legal", "registers_4_0"])
def test_acting_colour_is_the_only_difference(construction, scope):
    """With the acting-colour embedding's rows untied (the network as built), the position and
    its mirror differ - the input is live, so the outputs' equality above is the hook's doing -
    on every value-type head and the policy; tied, they agree (above)."""
    s, m = _pair(construction, 0)
    model = _model(5, scope, tied=False, **_config_for(s))
    a, b = _outputs(model, s), _outputs(model, m)
    for name in ("value_logits", "wdl_logits", "short_term", "value_error", "moves_left_log", "effort_logits", "submit_raw"):
        assert not torch.allclose(getattr(a, name), getattr(b, name), atol=1e-4), name
    assert not torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=1e-4)
    tie(model)
    assert_mirrored(_outputs(model, s), _outputs(model, m), torch.from_numpy(mirror_tokens(s.layout)), _acting(s, m))


def test_square_labels_map_back_to_colours():
    """The per-square labels are ours/theirs (a relabelling of the set's white/black facts,
    spec issue 226): mapped back, White acting reads them as they are, Black acting swaps every
    ours/theirs pair; the colour-free labels stay."""
    assert len(SQUARE_LABELS) == len(SQUARE_LABELS_BY_COLOUR) == 8
    assert [n for n in SQUARE_LABELS if "white" in n or "black" in n] == []
    logits = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    out = square_labels_by_colour(logits, torch.tensor([0, 1]))
    assert torch.equal(out[0], logits[0])
    for i, name in enumerate(SQUARE_LABELS):
        if name.endswith("_ours"):
            j = SQUARE_LABELS.index(name.replace("_ours", "_theirs"))
            assert torch.equal(out[1][:, i], logits[1][:, j]) and torch.equal(out[1][:, j], logits[1][:, i])
            assert SQUARE_LABELS_BY_COLOUR[i] == name.replace("_ours", "_white")
        elif not name.endswith("_theirs"):
            assert torch.equal(out[1][:, i], logits[1][:, i]) and SQUARE_LABELS_BY_COLOUR[i] == name


# --- the cached path: section 15.2's first row with the frame flipping -----------------------------------


R1, R2 = "past_and_epoch_history", "causal_past"
CACHED_CONSTRUCTIONS = ("standard", "duplicated", "deep_travel", "even_start", "shifted_submit_legal", "size_5")


def _close(a, b, atol_trunk=2e-5, atol=1e-4):
    assert torch.allclose(a.trunk, b.trunk, atol=atol_trunk), float((a.trunk - b.trunk).abs().max())
    for name in VALUE_HEADS:
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=atol), name
    assert torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=atol)
    assert torch.allclose(a.board_logits, b.board_logits, atol=atol)
    assert torch.allclose(a.leave_logits[a.board_valid], b.leave_logits[b.board_valid], atol=atol)  # D2's flat head


@pytest.mark.parametrize("supersession", [R1, R2])
@pytest.mark.parametrize("refresh", [False, True], ids=["causal", "root-refresh"])
@pytest.mark.parametrize("construction", CACHED_CONSTRUCTIONS)
def test_the_cached_path_and_the_mirror(construction, refresh, supersession):
    """Under the block-causal scope, through the cached evaluator (frozen K/V keyed by the
    acting player; a refresh's history per perspective - spec issue 303): a position and the
    same boards with the other colour acting (``viewed_by``), each in the epoch rooted at the
    position, equal one full pass (section 15.2's first row - with a refresh, the two-level
    reference whose anchor pass is the node's view); the mirror's, in the epoch rooted at the
    mirror, equal them mapped back (tied colour rows); and the store holds one entry per frozen
    board and perspective."""
    for seed in (0, 1):
        s, m = _pair(construction, seed)
        model = _model(seed, "block_causal", supersession_scope=supersession, **_config_for(s))
        cached = CachedEvaluator(model, CacheConfig(root_refresh=refresh))
        outs = {}
        for name, root in (("s", s), ("m", m)):
            epoch = cached.root_epoch(root)
            for view in (int(root.acting_player), 1 - int(root.acting_player)):
                node = viewed_by(root, view)
                got = cached.evaluate(node, epoch)
                want = reference_output(model, node, anchor=root) if refresh else reference_output(model, node, history_from=root)
                _close(got, want)
                outs[name, view == int(root.acting_player)] = (got, int(node.acting_player))
        perm = torch.from_numpy(mirror_tokens(s.layout))
        for own in (True, False):
            (a, acting_a), (b, acting_b) = outs["s", own], outs["m", own]
            assert_mirrored(a, b, perm, (acting_a, acting_b), atol=1e-4)
        n_frozen = int((~s.is_head).sum())
        assert len(cached.store.entries) == 2 * 2 * n_frozen  # two positions, two perspectives each


# --- the other colour's anchor: no move of the anchor's acting player (spec issue 229) ------------------


def dense_passes(model) -> tuple[list, object]:
    """``(passes, handle)``: every dense-scope forward of ``model`` from now on appends its batch
    to ``passes`` (a refresh's anchor passes, and ``reference_output``'s); ``handle.remove()``
    stops it."""
    passes: list = []

    def hook(module, args, kwargs):
        if kwargs.get("scope") == "dense":
            passes.append(args[0])

    return passes, model.register_forward_pre_hook(hook, with_kwargs=True)


def foreign_sources(batch) -> tuple[int, int, int]:
    """In a one-position batch: the move-graph edges, the moves and the edges of any relation
    whose source square does not hold one of the batch's acting player's pieces ("ours")."""
    owner = batch.owner[0]
    edges = batch.edges[0][batch.edge_valid[0]]
    moves = batch.moves[0][batch.move_valid[0]]
    move_edges = edges[edges[:, 2] == RELATION_IDS["move"]]
    return (
        int((owner[move_edges[:, 0]] != OWNER_OURS).sum()),
        int((owner[moves[:, 0]] != OWNER_OURS).sum()),
        int((owner[edges[:, 0]] != OWNER_OURS).sum()),
    )


def assert_other_perspective_anchor(passes, anchor) -> None:
    """The dense passes of one refresh over ``anchor``: its own view first, as it is (moves and
    edges kept), then the other colour's, with no move, reply move or edge at all - in
    particular none whose source is not that view's acting player's."""
    assert len(passes) == 2
    own, other = passes
    assert int(own.acting_player[0]) == int(anchor.acting_player) and int(other.acting_player[0]) == 1 - int(anchor.acting_player)
    whole = collate([anchor])
    for name in ("edges", "edge_valid", "moves", "move_valid"):
        assert torch.equal(getattr(own, name), getattr(whole, name)), name
    assert foreign_sources(other) == (0, 0, 0)
    assert not bool(other.edge_valid.any()) and not bool(other.move_valid.any()) and not bool(other.opp_move_valid.any())


@pytest.mark.parametrize("construction", CACHED_CONSTRUCTIONS)
def test_the_other_perspectives_anchor_has_no_moves_or_edges(construction):
    """A refresh's dense pass from the other colour's side (``viewed_by``) reads the anchor's
    boards with no moves and no edges: the anchor's are its acting player's, and kept they
    would start on "their" pieces, an input no conversion produces (spec issue 229). The same
    view in ``reference_output``'s anchor pass; and cached = reference for both perspectives."""
    for seed in (0, 1):
        s = CONSTRUCTIONS[construction](1000 * (seed + 1) + 17)
        assert s.edges.size and s.moves.size
        model = _model(seed, "block_causal", **_config_for(s))
        cached = CachedEvaluator(model, CacheConfig(root_refresh=True))
        passes, handle = dense_passes(model)
        try:
            epoch = cached.root_epoch(s)
            assert_other_perspective_anchor(passes, s)
            node = viewed_by(s, 1 - int(s.acting_player))
            passes.clear()
            want = reference_output(model, node, anchor=s)
            assert len(passes) == 1 and torch.equal(passes[0].edge_valid, collate([node]).edge_valid)
            assert foreign_sources(passes[0]) == (0, 0, 0) and not bool(passes[0].edge_valid.any())
        finally:
            handle.remove()
        _close(cached.evaluate(node, epoch), want)
        _close(cached.evaluate(s, epoch), reference_output(model, s, anchor=s))


# --- register A8: the board-layout embedding reads the canonical frame ---------------------------------


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_the_mirror_with_the_board_layout_embedding(construction):
    """Section 15.2's A7 row with A8 on (its flag; both paddings): every board's one-hot layout -
    read in canonical (x, y) order - is the mirror's, bit for bit; so are each square's layout
    channels at the mirrored token and, the acting colour equalised, the token embedding before
    the embedding's FFN (a network without it: the same draws) under either scope - with the FFN
    within 1e-6, the GEMM's rounding of rows in another order; and the outputs, mapped back, agree
    (FP32 1e-5), in the dense scope and the block-causal one."""
    for seed in (0, 1):
        s, m = _pair(construction, seed)
        bs, bm = collate([s]), collate([m])
        perm = torch.from_numpy(mirror_tokens(s.layout))
        padding = ("a1", "centre")[seed]
        for scope in ("dense", "block_causal"):
            model = _model(seed, scope, board_layout_embedding=True, layout_padding=padding, **_config_for(s))
            layer = model.embed.layout
            assert torch.equal(layer.layouts(bm), layer.layouts(bs)), scope
            assert torch.equal(layer.channels(bm)[0][perm], layer.channels(bs)[0]), scope
            bare = _model(seed, scope, board_layout_embedding=True, layout_padding=padding, layout_ffn_ratio=0, **_config_for(s))
            assert torch.equal(bare.embed.layout.proj.weight, layer.proj.weight)
            assert torch.equal(bare.embed_tokens(bm, scope)[0][perm], bare.embed_tokens(bs, scope)[0]), scope
            assert torch.allclose(model.embed_tokens(bm, scope)[0][perm], model.embed_tokens(bs, scope)[0], rtol=0, atol=1e-6), scope
            assert_mirrored(_outputs(model, s), _outputs(model, m), perm, _acting(s, m))


@pytest.mark.parametrize("refresh", [False, True], ids=["causal", "root-refresh"])
@pytest.mark.parametrize("construction", ("standard", "deep_travel", "even_start", "size_5"))
def test_the_cached_path_and_the_mirror_with_the_board_layout_embedding(construction, refresh):
    """The cached path with A8 on: each perspective's evaluation equals one full pass, and the
    mirror's equals it mapped back (a frozen board's layout is intrinsic, keyed by perspective)."""
    s, m = _pair(construction, 0)
    model = _model(0, "block_causal", board_layout_embedding=True, **_config_for(s))
    cached = CachedEvaluator(model, CacheConfig(root_refresh=refresh))
    outs = {}
    for name, root in (("s", s), ("m", m)):
        epoch = cached.root_epoch(root)
        for view in (int(root.acting_player), 1 - int(root.acting_player)):
            node = viewed_by(root, view)
            got = cached.evaluate(node, epoch)
            want = reference_output(model, node, anchor=root) if refresh else reference_output(model, node)
            _close(got, want)
            outs[name, view == int(root.acting_player)] = (got, int(node.acting_player))
    perm = torch.from_numpy(mirror_tokens(s.layout))
    for own in (True, False):
        (a, acting_a), (b, acting_b) = outs["s", own], outs["m", own]
        assert_mirrored(a, b, perm, (acting_a, acting_b), atol=1e-4)
