"""Parameter counts against spec section 9.2 (F1 ~21.5M with all heads; F2's L24-ff12 44.0M).

Exact v1 count (default config): 21,358,601 =
    embeddings 19,584 + trunk attention 4,210,688 + trunk FFN 16,814,080
    + trunk norms 16,896 + learned register coordinates 4 + policy 131,584
    + value 78,899 + aux 86,866.
Section 9.2's "~21.5M with all heads" also counts board-pair smolgen (<0.1M, B12,
prototype): stage N2 builds it behind a flag, 32,832 parameters (test_n2_model.py); its
"base 256x16, ff8: 21.0M" is the trunk. B9's default masks add no parameter.
"""

from __future__ import annotations

from t5d.net import NetConfig, build_model, parameter_breakdown

V1_TOTAL = 21_358_601


def test_v1_parameter_count():
    counts = parameter_breakdown(build_model(NetConfig()))
    assert counts["total"] == V1_TOTAL, counts
    assert abs(counts["total"] - 21.5e6) / 21.5e6 < 0.02, "section 9.2: ~21.5M with all heads"
    trunk = counts["trunk.attention"] + counts["trunk.ffn"] + counts["trunk.norms"]
    assert abs(trunk - 21.0e6) / 21.0e6 < 0.01, "section 9.2 ladder: base 256x16 ff8 = 21.0M"
    aux = sum(v for k, v in counts.items() if k.startswith("aux."))
    assert 0.05e6 < aux < 0.15e6, "section 9.2: aux heads ~0.1M total"
    assert counts["trunk.attention"] == 16 * 4 * (256 * 256 + 256), "full MHA, 8 KV heads (C4)"
    assert counts["trunk.ffn"] == 16 * (2 * 256 * 2048 + 2048 + 256), "FFN 8x"
    assert counts["rope.register_xy"] == 2 * 2, "learned (file, rank) per register slot (placeholder default)"


def test_f2_larger_arm_preset():
    cfg = NetConfig.preset("L24-ff12")
    assert (cfg.n_layers, cfg.ffn_ratio, cfg.d_model) == (24, 12, 256)
    counts = parameter_breakdown(build_model(cfg))
    assert abs(counts["total"] - 44.0e6) / 44.0e6 < 0.02, counts["total"]
    assert NetConfig() == NetConfig.preset("F1"), "F1 stays the default"


def test_flag_parameters_are_small():
    base = parameter_breakdown(build_model(NetConfig()))["total"]
    flagged = parameter_breakdown(build_model(NetConfig(binned_bias=True, policy_promotion="offset")))
    assert flagged["rope.binned_bias"] == 8 * 19 + 8 * 8, "bins x heads (section 4.1)"
    assert flagged["total"] - base < 10_000
    for spatial in ("zero_angle", "position_free"):
        counts = parameter_breakdown(build_model(NetConfig(board_register_spatial=spatial)))
        assert counts["rope.register_xy"] == 0 and counts["total"] == V1_TOTAL - 4, spatial
