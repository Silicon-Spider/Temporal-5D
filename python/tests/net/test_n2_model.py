"""Stage N2 in the whole network: N1's invariances still hold with every new switch on,
parameter counts stay at section 9.2's figures, BF16 stays close to FP32, and board-pair
smolgen (B12) reads each layer's own attention input, passes gradient back into the trunk,
and switching it on changes no other weight."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, permute_boards, synthetic_state, translate
from t5d.net import attention as attention_module
from t5d.net.attention import AttnContext
from tests.net.test_invariance import ATOL, _assert_same, _token_perm
from tests.net.test_params import V1_TOTAL

N2_FLAG_SETS = {
    "b9_default": dict(),
    "routing_typed_smolgen": dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True),
    "mg4_all_b5": dict(
        move_graph_heads=4,
        typed_edge_bias=True,
        typed_edge_directed=True,
        board_pair_smolgen=True,
        rope_per_head=True,
        rope_roles_source="masks",
    ),
    "routing_regs_free_anchored": dict(
        routing_table=True,
        board_register_spatial="position_free",
        global_register_pairs="anchored",
        masked_head_registers="self",
        move_graph_direction="forward",
        board_pair_smolgen=True,
    ),
    "bilinear_smolgen_chunked": dict(board_pair_smolgen=True, smolgen_form="board_bilinear", typed_edge_bias=True, attn_impl="chunked"),
    "no_table_routing_typed": dict(displacement_bias=False, routing_table=True, typed_edge_bias=True),
    "b17_shaw_routing_typed": dict(shaw_key_term=True, routing_table=True, typed_edge_bias=True, attn_impl="chunked"),
}


@pytest.mark.parametrize("flags", list(N2_FLAG_SETS))
def test_timeline_storage_order_does_not_matter(flags):
    """Edges travel with their boards (``permute_boards``); masks, typed biases and
    smolgen's board pairs follow."""
    model = build_model(NetConfig(trunk_dtype="fp32", **N2_FLAG_SETS[flags]), seed=1).eval()
    state = synthetic_state(31, 24, n_timelines=6)
    shuffled = np.random.default_rng(3).permutation(np.unique(state.timeline))
    order = np.concatenate([np.flatnonzero(state.timeline == p)[::-1] for p in shuffled])
    a = model(collate([state]))
    b = model(collate([permute_boards(state, order)]))
    _assert_same(a, b, board_perm=torch.from_numpy(order), token_perm=_token_perm(order, state.n_boards))


@pytest.mark.parametrize("recenter", [True, False], ids=["recentred", "raw-coords"])
@pytest.mark.parametrize("flags", list(N2_FLAG_SETS))
def test_translation_invariance(flags, recenter):
    if not recenter and N2_FLAG_SETS[flags].get("global_register_pairs") == "anchored":
        pytest.skip("anchored global registers need recentring (validate() refuses raw coordinates)")
    model = build_model(NetConfig(trunk_dtype="fp32", rope_recenter=recenter, **N2_FLAG_SETS[flags]), seed=2).eval()
    state = synthetic_state(32, 20)
    a = model(collate([state]))
    for dp, dh in ((7, 30), (-11, 400)):
        b = model(collate([translate(state, dp=dp, dh=dh)]))
        if recenter:
            assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.board_logits, b.board_logits)
        else:
            _assert_same(a, b)


@pytest.mark.parametrize("flags", list(N2_FLAG_SETS))
def test_batch_composition_does_not_matter(flags):
    """A position's masks and biases come from its own lists: alone or batched, the same."""
    model = build_model(NetConfig(trunk_dtype="fp32", **N2_FLAG_SETS[flags]), seed=3).eval()
    small, big, mid = synthetic_state(34, 3), synthetic_state(35, 11), synthetic_state(36, 6)
    alone = model(collate([small]))
    together = model(collate([big, small, mid]))
    assert torch.allclose(alone.value_logits[0], together.value_logits[1], atol=ATOL)
    m = alone.move_valid[0].sum()
    assert torch.allclose(alone.policy_logits[0, :m], together.policy_logits[1, :m], atol=ATOL)
    assert torch.allclose(alone.board_logits[0], together.board_logits[1, :3], atol=ATOL)


def test_parameter_counts_stay_at_section_9_2():
    """B9 and B10 are masks (no parameters); B11 adds 40 scalars (80 directed: 8 heads x its
    five types - travel-threat folded into attack at CHECKPOINT 6); B12 adds
    47,136 with relative cells (8,224 for W_z, 8 x 152 x 32 for the cell table) and 32,832
    in its bilinear form (section 9.2: "<0.1M"); with everything on the total stays at section
    9.2's ~21.5M plus B15's ~1.6M displacement table (section 4.1; spec issue 210)."""
    assert parameter_breakdown(build_model(NetConfig()))["total"] == V1_TOTAL, "the B9 default adds no parameter"
    routed = parameter_breakdown(build_model(NetConfig(routing_table=True)))
    assert routed["total"] == V1_TOTAL
    full = parameter_breakdown(build_model(NetConfig(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)))
    assert full["attn.typed_edges"] == 8 * 5
    assert full["attn.smolgen"] == (256 * 32 + 32) + 8 * (19 * 8) * 32 == 47_136
    assert full["attn.smolgen"] < 0.1e6, "section 9.2: smolgen <0.1M"
    assert full["total"] == V1_TOTAL + 40 + 47_136
    assert abs(full["total"] - 23.1e6) / 23.1e6 < 0.02, "section 9.2: ~21.5M with all heads, and section 4.1's ~1.6M"
    bilinear = parameter_breakdown(build_model(NetConfig(board_pair_smolgen=True, smolgen_form="board_bilinear")))
    assert bilinear["attn.smolgen"] == 2 * (256 * 32 + 32) + 2 * (8 * 32 * 32) == 32_832
    directed = parameter_breakdown(build_model(NetConfig(typed_edge_bias=True, typed_edge_directed=True)))
    assert directed["attn.typed_edges"] == 8 * 5 * 2


@pytest.mark.parametrize("n", [2, 17])
def test_bf16_close_to_fp32_with_every_switch(n):
    """G1 with stage N2's switches on: the chunked BF16 trunk stays within N1's bounds
    (test_precision.py) of the FP32 one."""
    flags = N2_FLAG_SETS["mg4_all_b5"]
    m32 = build_model(NetConfig(trunk_dtype="fp32", **flags), seed=5).eval()
    m16 = build_model(NetConfig(trunk_dtype="bf16", **flags), seed=5).eval()
    batch = collate([synthetic_state(40 + n, n)])
    a, b = m32(batch, return_trunk=True), m16(batch, return_trunk=True)
    diff = (a.trunk - b.trunk)[batch.valid]
    assert diff.pow(2).mean().sqrt() < 0.01 and diff.abs().max() < 0.06
    assert (a.value_logits - b.value_logits).abs().max() < 0.02
    assert (a.policy_logits - b.policy_logits)[batch.move_valid].abs().max() < 0.03
    assert not torch.equal(a.trunk, b.trunk)


# --- B12 in the network: where it reads, where its gradient goes, what it changes ---------------


def _layer_contexts(monkeypatch) -> list[AttnContext]:
    """Record the context each layer's attention is called with."""
    seen: list[AttnContext] = []
    for name, attend in list(attention_module.ATTENTION.items()):

        def spy(q, k, v, ctx, _attend=attend):
            seen.append(ctx)
            return _attend(q, k, v, ctx)

        monkeypatch.setitem(attention_module.ATTENTION, name, spy)
    return seen


@pytest.mark.parametrize("placement", ["pre", "post"])
@pytest.mark.parametrize("form", ["relative_cells", "board_bilinear"])
def test_smolgen_reads_each_layers_attention_input(monkeypatch, form, placement):
    """Each layer's board-pair table is generated from that layer's attention input. Under
    pre-norm (F5's v1) that is the normed residual stream - ``blocks[i].norm_attn(x)`` - and not
    the residual stream itself; under the post-norm arm (6y) it is the block's input ``x``
    (the previous block's FFN norm normed it; block 0's is the embedding), not ``norm_attn(x)``:
    layers 0 and 1 rebuilt by hand."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, board_pair_smolgen=True, smolgen_form=form, norm_placement=placement)
    model = build_model(cfg, seed=8).eval()
    batch = collate([synthetic_state(80, 5)])
    seen = _layer_contexts(monkeypatch)
    with torch.no_grad():
        model(batch)
        smolgen_in = model.make_context(batch).structure.smolgen_in
        x = model.embed(batch)
        for layer, block in enumerate(model.trunk.blocks):
            h, other = (block.norm_attn(x), x) if placement == "pre" else (x, block.norm_attn(x))
            expected = F.pad(model.smolgen(h, smolgen_in).float(), (0, 1, 0, 1))
            assert torch.allclose(seen[layer].structure.pair_table, expected, atol=1e-6), layer
            assert not torch.allclose(F.pad(model.smolgen(other, smolgen_in).float(), (0, 1, 0, 1)), expected, atol=1e-3)
            x = block(x, model.make_context(batch))


@pytest.mark.grad
def test_smolgen_passes_gradient_back_into_the_trunk(monkeypatch):
    """The trunk is trained through smolgen too: detaching smolgen's input inside
    ``for_layer`` leaves the forward unchanged but changes the trunk's gradients."""
    batch = collate([synthetic_state(81, 4)])

    def run(detach: bool):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, board_pair_smolgen=True), seed=9)
        if detach:
            original = AttnContext.for_layer
            monkeypatch.setattr(AttnContext, "for_layer", lambda self, h, layer: original(self, h.detach(), layer))
        out = model(batch)
        out.value_logits.sum().backward()
        monkeypatch.undo()
        return out.value_logits.detach(), model.trunk.blocks[0].norm_attn.weight.grad.clone()

    (out_a, grad_a), (out_b, grad_b) = run(False), run(True)
    assert torch.equal(out_a, out_b), "the forward is the same"
    assert not torch.allclose(grad_a, grad_b, rtol=1e-4, atol=1e-7), "smolgen's path carries gradient into the trunk"


@pytest.mark.parametrize("flags", [dict(board_pair_smolgen=True), dict(board_pair_smolgen=True, smolgen_form="board_bilinear")])
def test_switching_smolgen_on_changes_no_other_weight(flags):
    """B12 is constructed without drawing from the RNG and drawn last: every weight but
    smolgen's is the same with it on as with it off, so an on/off comparison measures
    smolgen alone."""
    off = dict(build_model(NetConfig(), seed=3).named_parameters())
    on = dict(build_model(NetConfig(**flags), seed=3).named_parameters())
    assert set(on) - set(off) == {n for n in on if n.startswith("smolgen.")}
    for name, p in off.items():
        assert torch.equal(p, on[name]), name
