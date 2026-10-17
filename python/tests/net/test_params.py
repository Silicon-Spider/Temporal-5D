"""Parameter counts against spec section 9.2 (F1 ~21.5M with all heads; F2's L24-ff12 44.0M).

Exact v1 count (default config): 21,338,634 =
    embeddings 20,096 + trunk attention 4,198,400 + trunk FFN 16,814,080
    + trunk norms 8,448 + learned register coordinates 4 + policy 131,584
    + value 78,899 + submit 257 + aux 86,866.
F5 (CHECKPOINT 6, section 9.3) took 20,736 from stage N5's 21,359,370: the norms are
RMSNorm-style gains without biases (33 x 256 = 8,448 fewer) and the Q, K and V projections
have no biases (16 x 3 x 256 = 12,288 fewer); the attention-output and FFN biases stay.
Stage N5 (register D8) had added 769 to stage N1's 21,358,601: the acting-player embedding
(2 x 256, in the embeddings) and the submit action's linear readout of a global register
(256 + 1; ``submit_head="pointer"`` has none).
Section 9.2's "~21.5M with all heads" also counts board-pair smolgen (<0.1M, B12,
prototype): stage N2 builds it behind a flag, 47,136 parameters as relative cells
(test_n2_model.py); its "base 256x16, ff8: 21.0M" is the trunk. B9's default masks add no
parameter. F2's larger arm, L24-ff12, is 44,456,458 as built (section 9.2: 44.0M).
"""

from __future__ import annotations

from t5d.net import NetConfig, build_model, parameter_breakdown

V1_TOTAL = 21_338_634
F2_TOTAL = 44_456_458


def test_v1_parameter_count():
    counts = parameter_breakdown(build_model(NetConfig()))
    assert counts["total"] == V1_TOTAL, counts
    assert abs(counts["total"] - 21.5e6) / 21.5e6 < 0.02, "section 9.2: ~21.5M with all heads"
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
    assert abs(counts["total"] - 44.0e6) / 44.0e6 < 0.02, "section 9.2: 44.0M"
    assert NetConfig() == NetConfig.preset("F1"), "F1 stays the default"


def test_flag_parameters_are_small():
    base = parameter_breakdown(build_model(NetConfig()))["total"]
    flagged = parameter_breakdown(build_model(NetConfig(binned_bias=True, policy_promotion="offset")))
    assert flagged["rope.binned_bias"] == 8 * 19 + 8 * 8, "bins x heads (section 4.1)"
    assert flagged["total"] - base < 10_000
    for spatial in ("zero_angle", "position_free"):
        counts = parameter_breakdown(build_model(NetConfig(board_register_spatial=spatial)))
        assert counts["rope.register_xy"] == 0 and counts["total"] == V1_TOTAL - 4, spatial
