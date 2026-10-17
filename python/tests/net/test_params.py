"""Parameter counts against spec section 9.2 (F1 ~21.5M with all heads; F2's L24-ff12 44.0M) and
section 4.1 (B15's displacement table, "about 1.6M parameters - ~7% of the network").

Exact v1 count (default config): 22,961,802 =
    embeddings 20,096 + trunk attention 4,198,400 + trunk FFN 16,814,080
    + trunk norms 8,448 + learned register coordinates 4 + policy 131,584
    + value 78,899 + submit 257 + aux 86,866
    + B15's displacement table 1,623,168 (16 layers x 8 heads x 12,681 entries: 54
      board-distance classes x 225, 368 line entries, 162 register entries, 1 global - the
      unstated details at their placeholders, spec issues 205-208).
B15 (CHECKPOINT 6, the v1 default) added the table to 21,338,634; section 9.2's "~21.5M"
predates it (spec issue 210): the network without the table - ablation 2's no-table arm,
``displacement_bias=False`` - is 21,338,634, and that is what section 9.2's figure is held
to below.
F5 (CHECKPOINT 6, section 9.3) took 20,736 from stage N5's 21,359,370: the norms are
RMSNorm-style gains without biases (33 x 256 = 8,448 fewer) and the Q, K and V projections
have no biases (16 x 3 x 256 = 12,288 fewer); the attention-output and FFN biases stay.
Stage N5 (register D8) had added 769 to stage N1's 21,358,601: the acting-player embedding
(2 x 256, in the embeddings) and the submit action's linear readout of a global register
(256 + 1; ``submit_head="pointer"`` has none).
Section 9.2's "~21.5M with all heads" also counts board-pair smolgen (<0.1M, B12,
prototype): stage N2 builds it behind a flag, 47,136 parameters as relative cells
(test_n2_model.py); its "base 256x16, ff8: 21.0M" is the trunk. B9's default masks add no
parameter. F2's larger arm, L24-ff12, is 46,891,210 as built: 44,456,458 (section 9.2: 44.0M)
and B15's 24 x 8 x 12,681 = 2,434,752.
B16's status table (committed with the cached configuration) adds 304 to a block-causal
network and nothing to the v1, whose scope is dense.
"""

from __future__ import annotations

from t5d.net import NetConfig, build_model, parameter_breakdown

V1_TOTAL = 22_961_802
F2_TOTAL = 46_891_210
#: B15's table in the v1 (16 layers) and in F2's arm (24 layers).
V1_DISPLACEMENT = 16 * 8 * 12_681
F2_DISPLACEMENT = 24 * 8 * 12_681


def test_v1_parameter_count():
    counts = parameter_breakdown(build_model(NetConfig()))
    assert counts["total"] == V1_TOTAL, counts
    assert counts["attn.displacement"] == V1_DISPLACEMENT == 1_623_168, "B15: section 4.1's ~1.6M"
    assert abs(counts["attn.displacement"] / counts["total"] - 0.07) < 0.005, "section 4.1: ~7% of the network"
    assert abs(counts["total"] - V1_DISPLACEMENT - 21.5e6) / 21.5e6 < 0.02, "section 9.2: ~21.5M with all heads (before B15)"
    trunk = counts["trunk.attention"] + counts["trunk.ffn"] + counts["trunk.norms"]
    assert abs(trunk - 21.0e6) / 21.0e6 < 0.01, "section 9.2 ladder: base 256x16 ff8 = 21.0M"
    aux = sum(v for k, v in counts.items() if k.startswith("aux."))
    assert 0.05e6 < aux < 0.15e6, "section 9.2: aux heads ~0.1M total"
    assert counts["trunk.attention"] == 16 * (4 * 256 * 256 + 256), "full MHA, 8 KV heads (C4); F5: only the output projection has a bias"
    assert counts["trunk.ffn"] == 16 * (2 * 256 * 2048 + 2048 + 256), "FFN 8x, its biases kept (F5)"
    assert counts["trunk.norms"] == (2 * 16 + 1) * 256, "F5: RMSNorm gains, no biases"
    assert counts["rope.register_xy"] == 2 * 2, "learned (file, rank) per register slot (placeholder default)"
    assert counts["head.submit"] == 256 + 1, "D8: the submit logit's linear readout of a global register (placeholder)"
    assert counts["embeddings"] == 19_584 + 2 * 256, "D8: the acting-player embedding (stage N5)"


def test_f2_larger_arm_preset():
    cfg = NetConfig.preset("L24-ff12")
    assert (cfg.n_layers, cfg.ffn_ratio, cfg.d_model) == (24, 12, 256)
    counts = parameter_breakdown(build_model(cfg))
    assert counts["total"] == F2_TOTAL, counts["total"]
    assert counts["attn.displacement"] == F2_DISPLACEMENT, "B15's table: one per layer"
    assert abs(counts["total"] - F2_DISPLACEMENT - 44.0e6) / 44.0e6 < 0.02, "section 9.2: 44.0M (before B15)"
    assert NetConfig() == NetConfig.preset("F1"), "F1 stays the default"


def test_the_cached_configuration_adds_b16s_table():
    """B16 (committed with the cached configuration): the block-causal network carries its status
    table - 8 heads x 19 Delta-t bins x 2 timeline statuses = 304, shared by the layers (4,864 per
    layer; placeholders, spec issues 213-214) - and the v1, dense, none."""
    v1 = parameter_breakdown(build_model(NetConfig()))
    assert v1["attn.status"] == 0 and v1["total"] == V1_TOTAL
    cached = parameter_breakdown(build_model(NetConfig(attn_scope="block_causal")))
    assert cached["attn.status"] == 304 and cached["total"] == V1_TOTAL + 304
    per_layer = parameter_breakdown(build_model(NetConfig(attn_scope="block_causal", status_bias_per_layer=True)))
    assert per_layer["attn.status"] == 16 * 304 == 4_864
    current = parameter_breakdown(build_model(NetConfig(attn_scope="block_causal", frozen_features="current")))
    assert current["attn.status"] == 0 and current["total"] == V1_TOTAL


def test_flag_parameters_are_small():
    base = parameter_breakdown(build_model(NetConfig()))["total"]
    flagged = parameter_breakdown(build_model(NetConfig(policy_promotion="offset")))
    assert 0 < flagged["total"] - base < 10_000
    no_table = parameter_breakdown(build_model(NetConfig(displacement_bias=False)))
    assert no_table["attn.displacement"] == 0 and no_table["total"] == V1_TOTAL - V1_DISPLACEMENT, "ablation 2's no-table arm"
    assert "rope.binned_bias" not in no_table, "B6 is removed (superseded by B15)"
    for spatial in ("zero_angle", "position_free"):
        counts = parameter_breakdown(build_model(NetConfig(board_register_spatial=spatial)))
        assert counts["rope.register_xy"] == 0 and counts["total"] == V1_TOTAL - 4, spatial
