"""Parameter counts against spec section 9.2 (F1 ~21.5M with all heads; F2's L24-ff12 44.0M) and
section 4.1 (B15's displacement table, "about 1.6M parameters - ~7% of the network").

Exact v1 count (default config): 22,969,139 =
    embeddings 28,864 + trunk attention 4,198,400 + trunk FFN 16,814,080
    + trunk norms 8,448 + policy 134,925 (its promotion offset 3,341 among them)
    + value 80,698 + submit 257 + D2's leave head 257 + aux 76,586 (the soft policy's 69,390 among them)
    + B15's displacement table 1,623,168 (16 layers x 8 heads x 12,681 entries: 54
      board-distance classes x 225, 368 line entries, 162 register entries, 1 global - the
      unstated details at their placeholders, spec issues 205-208)
    + B15's two per-axis far tables 3,456 (16 layers x 8 heads x (19 Delta-time + 8 Delta-timeline
      buckets): B6's tables, kept beyond the exact classes).
B15's per-axis far tables (register B6 and B15, section 4.1; the 2026-11-15 review, decision 1: "a
far pair ... adds B6's two clamped log-bucket scalars ... for 3,456 parameters") added 3,456 to
22,965,683 - a ruling that changed the default network. They are drawn from a generator of their
own, so every other parameter is as before, bit for bit: the fingerprints below leave the two new
tensors out and are unedited, and ``disp_far_axis=False`` is the 22,965,683 network. Ablation 2's
layer-shared arm (``disp_layer_shared``, off) is 21,444,179: B15 at 8 x (12,681 + 27) = 101,664, a
sixteenth of 1,626,624. (``test_displacement_far.py`` holds both arms' functions.)
Register D2's flat head (section 8.2: "per-board policy marginals - a head change") added 257 to
22,965,426: one "leave it" logit per board, a linear readout (256 + 1) of the board's pooled state
(placeholders, spec issue 241), always built and drawn from a generator of its own - every other
parameter as before, bit for bit (``test_the_parameters_before_d2s_flat_head_are_drawn_as_before``).
Register D4's heads (CHECKPOINT 7) added 3,598 to 22,961,828: the soft policy, on - its own f and
g at width 128 (65,792), D1's promotion offset (3,341) and its own submit readout (257), 69,390 -
and the opponent-reply head (65,792) now a prototype behind its flag, off (on: 69,133 with the
offset - 23,034,559).
Register D3's value heads (CHECKPOINT 7) took 12,079 from 22,973,907: the four heads share the
value MLP's hidden layer (256 -> 256, 65,792) - the categorical (13,107) and WDL (771, N1's WDL
head moved onto the shared layer) as before, three short-term outputs (771) and the error head
(257) new - and N1's 51-bin short value head (13,107) went.
Register A6 (the turn-state counts on the global registers) added 8,768 to 22,965,139, in the
embeddings: the counts' MLP, Linear(9, 32) and Linear(32, 256) with biases (its form a
placeholder, spec issue 230).
M20's dispositions (CHECKPOINT 6) added 3,337 to 22,961,802: the policy's promotion offset (D1:
a 256 -> 13 readout of the destination token, 3,341) and no learned register coordinates (B8:
per-board registers position-free on file and rank - the learned_angle arm's 2 x 2 gone, -4);
the pooled submit logit, the heads' pools, moves-left in turns and the acting player on the
global registers add nothing.
B15 (CHECKPOINT 6, the v1 default) added the table to 21,338,634 (21,341,971 with M20's
dispositions); section 9.2's "~21.5M" predates it (spec issue 210): the network without the
table - ablation 2's no-table arm, ``displacement_bias=False`` - was 21,341,971 then (21,342,515
today: neither the table nor the per-axis far tables), and that is what section 9.2's figure is
held to below.
F5 (CHECKPOINT 6, section 9.3) took 20,736 from stage N5's 21,359,370: the norms are
RMSNorm-style gains without biases (33 x 256 = 8,448 fewer) and the Q, K and V projections
have no biases (16 x 3 x 256 = 12,288 fewer); the attention-output and FFN biases stay.
Stage N5 (register D8) had added 769 to stage N1's 21,358,601: the acting-player embedding
(2 x 256, in the embeddings) and the submit action's linear readout - now of the pooled global
registers (M20) - (256 + 1; ``submit_head="pointer"`` has none).
Section 9.2's "~21.5M with all heads" also counts board-pair smolgen (<0.1M, B12,
prototype): stage N2 builds it behind a flag, 47,136 parameters as relative cells
(test_n2_model.py); its "base 256x16, ff8: 21.0M" is the trunk. B9's default masks add no
parameter. F2's larger arm, L24-ff12, is 46,900,275 as built: 44,460,339 (section 9.2: 44.0M; A6's
8,768 and D2's leave head's 257 among them, D3's rebuild -12,079, D4's +3,598), B15's table 24 x 8 x
12,681 = 2,434,752 and its per-axis far tables 24 x 8 x 27 = 5,184 (46,895,091 before them).
B16's status table (committed with the cached configuration) adds 304 to a block-causal
network and nothing to the v1, whose scope is dense.
Register A8's board-layout embedding (a prototype behind its flag, off) adds nothing to the v1;
on, 2,629,120 - 25,598,259 (``test_layout.py``; its FFN's width a placeholder, spec issue 240) -
and, at ablation 6p's second arm (section 2.2, the 2026-11-15 review: its FFN at 1x), 1,709,824 -
24,678,963; the two arms are the presets ``"6p-layout-ff8"`` and ``"6p-layout-ff1"``. (Before
B15's per-axis far tables: 25,594,803 and 24,675,507.)
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from t5d.net import NetConfig, build_model, parameter_breakdown
from t5d.net.bench import CONFIGS as BENCH_CONFIGS
from t5d.net.layout import layout_parameters

V1_TOTAL = 22_969_139
F2_TOTAL = 46_900_275
#: B15's table in the v1 (16 layers) and in F2's arm (24 layers).
V1_DISPLACEMENT = 16 * 8 * 12_681
F2_DISPLACEMENT = 24 * 8 * 12_681
#: B15's two per-axis far tables (B6's 19 Delta-time and 8 Delta-timeline buckets, per head per
#: layer; the 2026-11-15 review) in the v1 and in F2's arm - the default network's totals above
#: moved by them, from 22,965,683 and 46,895,091.
V1_FAR = 16 * 8 * (19 + 8)
F2_FAR = 24 * 8 * (19 + 8)
#: A8's board-layout embedding with its FFN at the placeholder's 8x and at ablation 6p's 1x arm.
A8_FF8 = 2_629_120
A8_FF1 = 1_709_824


def test_v1_parameter_count():
    counts = parameter_breakdown(build_model(NetConfig()))
    assert counts["total"] == V1_TOTAL, counts
    assert counts["attn.displacement"] == V1_DISPLACEMENT == 1_623_168, "B15: section 4.1's ~1.6M"
    assert counts["attn.displacement_far"] == V1_FAR == 3_456, "section 4.1: B6's per-axis terms, 3,456 parameters"
    assert abs(counts["attn.displacement"] / counts["total"] - 0.07) < 0.005, "section 4.1: ~7% of the network"
    assert abs(counts["total"] - V1_DISPLACEMENT - V1_FAR - 21.5e6) / 21.5e6 < 0.02, "section 9.2: ~21.5M with all heads (before B15)"
    trunk = counts["trunk.attention"] + counts["trunk.ffn"] + counts["trunk.norms"]
    assert abs(trunk - 21.0e6) / 21.0e6 < 0.01, "section 9.2 ladder: base 256x16 ff8 = 21.0M"
    aux = sum(v for k, v in counts.items() if k.startswith("aux."))
    assert 0.05e6 < aux < 0.15e6, "section 9.2: aux heads ~0.1M total (D4's soft policy among them)"
    assert counts["trunk.attention"] == 16 * (4 * 256 * 256 + 256), "full MHA, 8 KV heads (C4); F5: only the output projection has a bias"
    assert counts["trunk.ffn"] == 16 * (2 * 256 * 2048 + 2048 + 256), "FFN 8x, its biases kept (F5)"
    assert counts["trunk.norms"] == (2 * 16 + 1) * 256, "F5: RMSNorm gains, no biases"
    assert "rope.register_xy" not in counts, "B8 (M20): no learned register coordinate - position-free on file and rank"
    assert counts["head.policy"] == 2 * (256 * 256 + 256) + (256 * 13 + 13), "D1 (M20): f, g and the per-piece promotion offset"
    assert counts["head.value"] == (256 * 256 + 256) + (256 * 51 + 51) + (256 * 3 + 3) + (256 * 3 + 3) + (256 + 1), (
        "D3: the shared hidden layer, the categorical, WDL, three short-term outputs, the error head"
    )
    assert counts["aux.soft_policy"] == 2 * (256 * 128 + 128) + (256 * 13 + 13) + (256 + 1), "D4: f, g, the offset, its own submit"
    assert "aux.opp_reply" not in counts, "D4's reply is a prototype, off"
    reply = parameter_breakdown(build_model(NetConfig(opp_reply=True)))
    assert reply["aux.opp_reply"] == 2 * (256 * 128 + 128) + (256 * 13 + 13) and reply["total"] == V1_TOTAL + 69_133
    assert counts["head.submit"] == 256 + 1, "D8: the submit logit's linear readout of the pooled global registers (placeholder form)"
    assert counts["head.leave"] == 256 + 1, "D2's flat head: one leave logit per board, a linear readout of its pool (issue 241)"
    registers = parameter_breakdown(build_model(NetConfig(leave_pool="registers")))
    assert registers["head.leave"] == 257 and registers["total"] == V1_TOTAL, "the leave head's other pool adds nothing"
    assert counts["embeddings"] == 19_584 + 2 * 256 + 8_768, "D8: the acting-player embedding (stage N5); A6: the counts' MLP"


def test_f2_larger_arm_preset():
    cfg = NetConfig.preset("L24-ff12")
    assert (cfg.n_layers, cfg.ffn_ratio, cfg.d_model) == (24, 12, 256)
    counts = parameter_breakdown(build_model(cfg))
    assert counts["total"] == F2_TOTAL, counts["total"]
    assert counts["attn.displacement"] == F2_DISPLACEMENT, "B15's table: one per layer"
    assert counts["attn.displacement_far"] == F2_FAR == 5_184, "B15's per-axis far tables: one pair per layer"
    assert abs(counts["total"] - F2_DISPLACEMENT - F2_FAR - 44.0e6) / 44.0e6 < 0.02, "section 9.2: 44.0M (before B15)"
    assert NetConfig() == NetConfig.preset("F1"), "F1 stays the default"


def test_6p_board_layout_arm_presets():
    """Ablation 6p's two board-layout arms (register A8, section 2.2; the 2026-11-15 review: "the
    FFN's width is part of the arm: 6p tests it at 1x ... beside the 8x placeholder") are named - in
    ``PRESETS`` and, by the same names, in ``bench.CONFIGS`` - and their totals pinned. An arm differs
    from the v1 in A8 alone, and no default moved. (The built networks' breakdowns are
    ``test_layout.py``'s: these counts are ``layout_parameters``', which it holds the build to.)"""
    ff8, ff1 = NetConfig.preset("6p-layout-ff8"), NetConfig.preset("6p-layout-ff1")
    assert ff8 == NetConfig(board_layout_embedding=True) == NetConfig(board_layout_embedding=True, layout_ffn_ratio=8)
    assert ff1 == NetConfig(board_layout_embedding=True, layout_ffn_ratio=1)
    assert replace(ff1, board_layout_embedding=False, layout_ffn_ratio=8) == NetConfig(), "the arm differs from the v1 in A8 alone"
    assert NetConfig(**BENCH_CONFIGS["6p-layout-ff8"]) == ff8 and NetConfig(**BENCH_CONFIGS["6p-layout-ff1"]) == ff1
    assert layout_parameters(ff8)["total"] == A8_FF8 and layout_parameters(ff1)["total"] == A8_FF1
    assert A8_FF8 - A8_FF1 == 7 * (2 * 256 * 256 + 256) == 919_296, "seven fewer widths of the two matrices and the hidden bias"
    assert V1_TOTAL + A8_FF8 == 25_598_259 and V1_TOTAL + A8_FF1 == 24_678_963
    default = NetConfig()
    assert default == NetConfig.preset("F1"), "F1 stays the default"
    assert (default.board_layout_embedding, default.layout_ffn_ratio) == (False, 8), "the flag stays off, the placeholder at 8x"
    for unknown in ("6p-layout-ff2", "A8-ff1"):
        with pytest.raises(ValueError, match="unknown preset"):
            NetConfig.preset(unknown)


def test_the_cached_configuration_adds_b16s_table():
    """B16 (committed with the cached configuration): the block-causal network carries its status
    table - 8 heads x 19 Delta-t bins x 2 timeline statuses = 304, shared by the layers (4,864 with
    one table per layer, 16 x 304; placeholders, spec issues 213-214) - and the v1, dense, none."""
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
    flagged = parameter_breakdown(build_model(NetConfig(typed_edge_bias=True)))
    assert flagged["attn.typed_edges"] == 8 * 5 and flagged["total"] - base == 40, "B11: a scalar per head per type"
    pointer = parameter_breakdown(build_model(NetConfig(submit_head="pointer")))
    assert pointer["head.submit"] == 0 and pointer["total"] == V1_TOTAL - 2 * 257, (
        "D8's other readout form: each head's own f and g (the soft policy's too)"
    )
    no_table = parameter_breakdown(build_model(NetConfig(displacement_bias=False)))
    assert no_table["attn.displacement"] == no_table["attn.displacement_far"] == 0, "ablation 2's no-table arm: no B15 at all"
    assert no_table["total"] == V1_TOTAL - V1_DISPLACEMENT - V1_FAR == 21_342_515, "ablation 2's no-table arm"
    assert "rope.binned_bias" not in no_table, "B6 is removed (superseded by B15, which keeps its two tables beyond the exact classes)"
    pre_review = parameter_breakdown(build_model(NetConfig(disp_far_axis=False)))
    assert pre_review["attn.displacement_far"] == 0 and pre_review["attn.displacement"] == V1_DISPLACEMENT
    assert pre_review["total"] == V1_TOTAL - V1_FAR == 22_965_683, "disp_far_axis off: B15 as built before the 2026-11-15 review"


def test_m20s_dispositions_have_no_switch():
    """M20 disposed of these placeholders and no named ablation needs the arms: each switch is
    gone (an unknown field), not kept at a ruled default."""
    removed = ("board_register_spatial", "global_register_pairs", "policy_promotion")
    removed += ("aux_global_pool", "acting_player_input", "submit_register")
    for field in removed:
        with pytest.raises(TypeError):
            NetConfig(**{field: None})


#: The parameters the network had before register D3's value heads (``47bc857``, the merge before
#: N-D3), drawn at seed 7 in a small configuration, under their present names (N1's value output
#: is the categorical, its WDL head moved onto the value MLP) - pinned as ``_fingerprint`` of them,
#: recorded on that commit's build. D3, D4 and D9 add heads (``_NEW``) and retire two (N1's short
#: value head, its always-built reply head) whose draws stay in the global stream.
_SMALL = dict(trunk_dtype="fp32", n_layers=2, d_model=64, n_heads=2, ffn_ratio=4, policy_dim=64, value_hidden=64, opp_policy_dim=32)
_NEW = ("value.short_term.", "value.error.", "aux.soft_policy.", "aux.opp_reply.", "optimistic.", "leave.")
_PRE_D3 = (64, (-28.9795080817193, -896.406780811631, 15039.884536167086))
#: B15's two per-axis far tables (the 2026-11-15 review) are later than every fingerprint in this
#: file: each fingerprinted set leaves them out (their names would otherwise shift the ranks the
#: sums are weighted by), and the pinned counts and sums stand unedited - which is the evidence that
#: adding them moved no earlier parameter's draw. Their own draw is pinned in
#: ``test_displacement_far.py``.
_LATER = ("displacement.far_",)


def _fingerprint(params: dict) -> tuple[float, float, float]:
    """Three float64 sums over the tensors in name order - each tensor's dot product with a fixed
    exact pattern, summed plain and weighted by the tensor's rank, and the absolute sum: any draw
    moved or two tensors swapped changes them, and they need no bit-exact float32 libm across
    platforms (a hash of the bytes would)."""
    f = [0.0, 0.0, 0.0]
    for i, name in enumerate(sorted(params)):
        x = params[name].detach().double().flatten()
        w = torch.remainder(torch.arange(x.numel(), dtype=torch.float64) * 0.6180339887498949, 1.0) - 0.5
        d = float((w * x).sum())
        f[0] += d
        f[1] += (i + 1) * d
        f[2] += float(x.abs().sum())
    return f[0], f[1], f[2]


def _same(got, expected, n: int, values: int | None = None):
    """``got`` is the fingerprint ``expected`` up to the last-bit differences a platform's float32
    libm leaves in the draws themselves (a GitHub runner's differ from the development machine's at
    about 1e-8 relative): 1e-5 on the plain sum, ``n`` times that on the sum weighted by rank - a
    tensor's difference enters it up to ``n`` times over, ``n`` the number of tensors - and 1e-9
    relative on the absolute sum. A moved draw or two tensors swapped moves the sums far beyond.

    The 1e-9 is a large set's bound: over many values the per-value differences average out (the
    whole v1's sum differed by 1.8e-11 on a runner). A set of few values has little to average -
    B15's two far tables, 108 values here, differed by 2.0e-9 on a runner (the network job on
    main 032db11, 2026-11-23). ``values``, the number of values fingerprinted, fits the bound to
    what the sum holds: one float32 ulp a value (6e-8, taken as 1e-7) averaged over their square
    root, and never under 1e-9 - 9.6e-9 for those 108, where another draw moves the sum by several
    per cent."""
    assert got[0] == pytest.approx(expected[0], abs=1e-5) and got[1] == pytest.approx(expected[1], abs=1e-5 * n), got
    rel = 1e-9 if values is None else max(1e-9, 1e-7 / math.sqrt(values))
    assert got[2] == pytest.approx(expected[2], rel=rel), got


@pytest.mark.parametrize("flags", [{}, dict(opp_reply=True, optimistic_policy=True, soft_policy=False)], ids=["default", "prototypes"])
def test_the_parameters_before_d3_are_drawn_as_before(flags):
    """Every parameter that existed before D3 is drawn as before: D3's rebuild, D4's heads and
    D9's prototype - whatever their flags - leave every earlier parameter as it was drawn."""
    model = build_model(NetConfig(**_SMALL, **flags), seed=7)
    kept = {n: p for n, p in model.named_parameters() if not n.startswith(_NEW + _LATER)}
    count, expected = _PRE_D3
    assert len(kept) == count
    _same(_fingerprint(kept), expected, count)


#: Every parameter the network had before D2's flat head (``9c59cb0``, the merge before it), drawn
#: at seed 7 in the small configuration (default flags, and the prototypes' flags) and the v1 at
#: seed 0 - ``_fingerprint`` of them, recorded on that commit's build; and the leave head's own
#: draw at seed 7 (it draws from its own generator, so no flag and no pool changes it).
_PRE_D2 = {
    "default": (76, (-23.86661994843246, -1108.338941134584, 16781.204980568982)),
    "prototypes": (82, (-30.417464497016, -1169.5172809133992, 18881.094969031343)),
}
_PRE_D2_V1 = (230, (-94.90331727564617, -5889.620489603242, 770983.941444666))
_LEAVE_SEED_7 = (-0.16617493222981167, -0.3285210357709124, 5.933879833610263)


@pytest.mark.parametrize("pool", ["squares", "registers"])
@pytest.mark.parametrize("flags", ["default", "prototypes"])
def test_the_parameters_before_d2s_flat_head_are_drawn_as_before(flags, pool):
    """D2's flat head (the per-board leave logit, issue 241) changed no parameter that existed before
    it - whatever the flags and whichever pool it reads - and its own weights are drawn from its own
    generator, pinned."""
    kw = {} if flags == "default" else dict(opp_reply=True, optimistic_policy=True, soft_policy=False)
    model = build_model(NetConfig(**_SMALL, **kw, leave_pool=pool), seed=7)
    kept = {n: p for n, p in model.named_parameters() if not n.startswith(("leave.", *_LATER))}
    count, expected = _PRE_D2[flags]
    assert len(kept) == count
    _same(_fingerprint(kept), expected, count)
    leave = {n: p for n, p in model.named_parameters() if n.startswith("leave.")}
    _same(_fingerprint(leave), _LEAVE_SEED_7, len(leave))


@pytest.mark.slow
def test_the_v1s_parameters_before_d2s_flat_head_are_drawn_as_before():
    model = build_model(NetConfig(), seed=0)
    kept = {n: p for n, p in model.named_parameters() if not n.startswith(("leave.", *_LATER))}
    count, expected = _PRE_D2_V1
    assert len(kept) == count
    _same(_fingerprint(kept), expected, count)
