"""The trunk's normalisation, biases and activation (F5, F6; spec section 9.3) and ablation 6y's arms.

F5: every norm RMSNorm-style - no mean-centring, no bias - no Q/K/V biases, the
attention-output and FFN biases kept, the output heads' biases kept; pre-norm for v1,
"Lc0's post-norm with DeepNet initialisation" as 6y's arm (DeepNorm and DeepNet's
initialisation, Wang et al. 2022). F6: Mish in the FFN for v1, squared ReLU as 6y's arm; Mish
in every head's hidden layers too, under either arm - squared ReLU is a trunk-only arm
(CHECKPOINT 8). Each block is computed by hand here, from the formulas.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, synthetic_state
from t5d.net.smolgen import BoardPairSmolgen
from t5d.net.trunk import RMSNorm
from tests.net.test_params import V1_TOTAL

# --- the norm ------------------------------------------------------------------------------------


def _norm(d: int, seed: int) -> RMSNorm:
    norm = RMSNorm(d, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(d, generator=torch.Generator().manual_seed(seed)))
    return norm


def test_rmsnorm_is_the_root_mean_square_formula():
    """``x / sqrt(mean(x^2) + eps) * g``: against the formula in FP64 and against PyTorch's own
    ``rms_norm`` (an implementation independent of this one); a gain and nothing else. (At
    this input's scale where eps sits moves the output by under 1e-5: the next test checks it
    where it matters.)"""
    norm = _norm(16, 0)
    x = 3.0 * torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1))
    out = norm(x)
    x64 = x.double()
    expected = x64 / torch.sqrt(x64.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * norm.weight.double()
    assert out.dtype == torch.float32
    assert torch.allclose(out.double(), expected, rtol=0.0, atol=1e-6)
    assert torch.allclose(out, F.rms_norm(x, (16,), norm.weight, 1e-5), rtol=0.0, atol=1e-6)
    assert [name for name, _ in norm.named_parameters()] == ["weight"], "a gain only: no bias"


def test_rmsnorm_eps_sits_inside_the_root():
    """eps inside the root, as in PyTorch's ``rms_norm`` (spec issue 200(b)), where it shows: on
    a small-RMS input (x ~ 3e-3, so mean(x^2) is about eps) the norm is the FP64 formula and
    ``rms_norm`` with no relative slack (measured: to 1.9e-7, and bit for bit). eps outside the
    root, ``x / (sqrt(mean(x^2)) + eps)``, or no eps at all, is 0.95 away there."""
    norm = _norm(16, 6)
    x = 3e-3 * torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(7))
    out = norm(x)
    x64, g = x.double(), norm.weight.double()
    ms = x64.pow(2).mean(dim=-1, keepdim=True)
    assert torch.allclose(out.double(), x64 / torch.sqrt(ms + 1e-5) * g, rtol=0.0, atol=1e-6)
    assert torch.allclose(out, F.rms_norm(x, (16,), norm.weight, 1e-5), rtol=0.0, atol=1e-6)
    for wrong in (x64 / (torch.sqrt(ms) + 1e-5) * g, x64 / torch.sqrt(ms) * g):  # eps outside the root; none
        assert float((out.double() - wrong).abs().max()) > 0.1


@pytest.mark.parametrize("placement", ["pre", "post"])
def test_norm_eps_reaches_every_trunk_norm(placement):
    """``NetConfig.norm_eps`` is every trunk norm's eps, stored and used: at 1e-2 each norm of the
    model computes ``rms_norm`` with eps 1e-2 on a small-RMS input (x ~ 1e-2), where the default
    1e-5 gives outputs 9.5x larger."""
    model = build_model(NetConfig(n_layers=2, norm_placement=placement, norm_eps=1e-2), seed=0)
    norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 2 * 2 + (placement == "pre")
    x = 1e-2 * torch.randn(4, 256, generator=torch.Generator().manual_seed(8))
    for norm in norms:
        assert norm.eps == 1e-2
        out = norm(x)
        assert torch.allclose(out, F.rms_norm(x, (256,), norm.weight, 1e-2), rtol=0.0, atol=1e-6)
        assert not torch.allclose(out, F.rms_norm(x, (256,), norm.weight, 1e-5), rtol=0.0, atol=0.1)


def test_rmsnorm_does_not_centre():
    """No mean-centring: adding the same constant to every component moves the output - a
    LayerNorm would not see it - while a positive rescale does not, up to eps (relative
    ~eps / (2 mean x^2), about 5e-6 here)."""
    norm = _norm(16, 2)
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(3))
    out = norm(x)
    assert not torch.allclose(norm(x + 0.7), out, atol=1e-2), "a shift moves the output: no centring"
    assert torch.allclose(F.layer_norm(x + 0.7, (16,)), F.layer_norm(x, (16,)), atol=1e-5), "the contrast: a LayerNorm ignores it"
    assert torch.allclose(norm(4.0 * x), out, rtol=2e-5, atol=1e-6)


def test_rmsnorm_runs_in_fp32_under_autocast():
    """G1: the norms stay FP32 inside the BF16 trunk's autocast - a BF16 input is normalised in
    FP32, and the output is the FP32 computation of the same values, bit for bit."""
    norm = _norm(32, 4)
    x = torch.randn(6, 32, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = norm(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, norm(x.float()))


# --- where the norms and biases are --------------------------------------------------------------


@pytest.mark.parametrize("placement", ["pre", "post"])
def test_norms_and_biases_where_section_9_3_puts_them(placement):
    """No LayerNorm anywhere; every trunk norm an ``RMSNorm`` (2 per block, plus the final
    norm under pre-norm); no Q/K/V biases; the attention-output and FFN biases kept; every
    Linear outside the trunk - the output heads ("small and FP32"), the embedding's Delta-t
    MLP, smolgen - keeps its bias, and smolgen has no norm of its own."""
    model = build_model(NetConfig(n_layers=3, norm_placement=placement, board_pair_smolgen=True), seed=0)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    norms = [m for m in model.trunk.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 2 * 3 + (placement == "pre")
    assert all(isinstance(m, RMSNorm) for m in model.modules() if "norm" in type(m).__name__.lower())
    for block in model.trunk.blocks:
        attn, ffn = block.attn, block.ffn
        assert attn.q_proj.bias is None and attn.k_proj.bias is None and attn.v_proj.bias is None, "no Q/K/V biases"
        assert attn.out_proj.bias is not None and ffn.lin_in.bias is not None and ffn.lin_out.bias is not None
    outside = {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not name.startswith("trunk.")}
    assert {"value.hidden", "value.out", "policy.src", "aux.wdl", "embed.dt_in", "smolgen.context"} <= set(outside)
    assert all(m.bias is not None for m in outside.values()), "the heads, the embedding and smolgen keep their biases"
    assert isinstance(model.smolgen, BoardPairSmolgen)
    assert not any(isinstance(m, RMSNorm) for m in model.smolgen.modules())


def test_the_bias_switches():
    """``qkv_bias`` and ``out_ffn_bias`` - F5's split of the old single switch - each add exactly
    their biases: 3 x 256 per layer on Q, K and V; 256 + 2048 + 256 per layer on the output
    projection and the FFN."""
    cfg = dict(n_layers=2)
    base = parameter_breakdown(build_model(NetConfig(**cfg), seed=0))
    with_qkv = parameter_breakdown(build_model(NetConfig(qkv_bias=True, **cfg), seed=0))
    without = parameter_breakdown(build_model(NetConfig(out_ffn_bias=False, **cfg), seed=0))
    assert with_qkv["trunk.attention"] - base["trunk.attention"] == 2 * 3 * 256
    assert base["trunk.attention"] - without["trunk.attention"] == 2 * 256
    assert base["trunk.ffn"] - without["trunk.ffn"] == 2 * (2048 + 256)
    model = build_model(NetConfig(qkv_bias=True, out_ffn_bias=False, **cfg), seed=0)
    attn = model.trunk.blocks[0].attn
    assert attn.q_proj.bias is not None and attn.out_proj.bias is None and model.trunk.blocks[0].ffn.lin_in.bias is None


# --- the two placements, block by block ----------------------------------------------------------


def _by_hand(model, batch, alpha: float | None = None):
    """The trunk computed from the formulas, block by block: pre-norm ``x + Attn(N(x))``, ``x +
    FFN(N(x))``, final norm; post-norm ``N(alpha x + Attn(x))``, ``N(alpha x + FFN(x))``."""
    ctx = model.make_context(batch)
    x = model.embed(batch)
    post = model.cfg.norm_placement == "post"
    alpha = model.cfg.deepnorm_alpha if alpha is None else alpha
    for layer, block in enumerate(model.trunk.blocks):
        if post:
            x = block.norm_attn(alpha * x + block.attn(x, ctx.for_layer(x, layer)))
            x = block.norm_ffn(alpha * x + block.ffn(x))
        else:
            h = block.norm_attn(x)
            x = x + block.attn(h, ctx.for_layer(h, layer))
            x = x + block.ffn(block.norm_ffn(x))
    return x if post else model.trunk.final_norm(x)


@pytest.mark.parametrize("smolgen", [False, True], ids=["plain", "smolgen"])
@pytest.mark.parametrize("placement", ["pre", "post"])
def test_each_placement_is_its_formula(placement, smolgen):
    """The trunk equals its formula under each placement; under post-norm alpha = (2L)^(1/4)
    scales the residual (alpha = 1 computes something else) and there is no final norm."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=3, norm_placement=placement, board_pair_smolgen=smolgen)
    model = build_model(cfg, seed=3).eval()
    batch = collate([synthetic_state(50, 4), synthetic_state(51, 2)])
    trunk = model(batch, return_trunk=True).trunk
    assert torch.allclose(trunk, _by_hand(model, batch), atol=1e-6)
    if placement == "post":
        assert cfg.deepnorm_alpha == pytest.approx(6**0.25) and model.trunk.final_norm is None
        assert not torch.allclose(trunk, _by_hand(model, batch, alpha=1.0), atol=1e-3), "the residual is scaled"
    else:
        assert model.trunk.final_norm is not None


def test_the_post_norm_output_is_normed():
    """Under post-norm the last block's FFN norm is the trunk's output norm: each token's output
    is that norm's gain times a unit-RMS vector."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, norm_placement="post"), seed=4).eval()
    batch = collate([synthetic_state(52, 3)])
    out = model(batch, return_trunk=True).trunk[0][batch.valid[0]]
    gain = model.trunk.blocks[-1].norm_ffn.weight
    unit = out / gain
    assert torch.allclose(unit.pow(2).mean(dim=-1), torch.ones(unit.shape[0]), atol=1e-4)


# --- initialisation -------------------------------------------------------------------------------


def test_deepnet_initialisation_under_post_norm():
    """6y's arm: DeepNet's initialisation as the paper's pseudocode draws it (Wang et al. 2022,
    Figure 2: ``xavier_normal_(w, gain=beta)`` for "ffn", "v_proj", "out_proj" and
    ``gain=1`` for "q_proj", "k_proj") - Xavier normal, std ``gain sqrt(2 / (fan_in +
    fan_out))``, beta = (8L)^(-1/4). On the square attention projections that is LeCun's std
    times the gain; the FFN's two matrices are both drawn at ``beta sqrt(2 / 2304)``, not at
    beta times LeCun's ``1/sqrt(256)`` and ``1/sqrt(2048)`` (2.1x and 0.75x it). Under
    pre-norm LeCun, the GPT-2 ``1/sqrt(2L)`` on the two residual outputs, and nothing else
    scaled."""
    n_layers = 16
    beta = (8 * n_layers) ** -0.25
    assert NetConfig().deepnorm_beta == pytest.approx(beta)
    gpt2 = 1.0 / math.sqrt(2 * n_layers)
    pre = build_model(NetConfig(), seed=0)
    post = build_model(NetConfig(norm_placement="post"), seed=0)
    d, ffn = 256**-0.5, 2048**-0.5  # LeCun: 1/sqrt(fan_in)
    xavier_attn, xavier_ffn = math.sqrt(2 / (256 + 256)), math.sqrt(2 / (256 + 2048))
    expected = {  # name: (pre-norm std, post-norm std)
        "attn.q_proj": (d, xavier_attn),
        "attn.k_proj": (d, xavier_attn),
        "attn.v_proj": (d, beta * xavier_attn),
        "attn.out_proj": (gpt2 * d, beta * xavier_attn),
        "ffn.lin_in": (d, beta * xavier_ffn),
        "ffn.lin_out": (gpt2 * ffn, beta * xavier_ffn),
    }
    for layer in (0, 15):
        for name, (s_pre, s_post) in expected.items():
            full = f"trunk.blocks.{layer}.{name}.weight"
            assert float(pre.get_parameter(full).std()) == pytest.approx(s_pre, rel=0.02), (full, "pre")
            assert float(post.get_parameter(full).std()) == pytest.approx(s_post, rel=0.02), (full, "post")


def test_post_norm_parameter_count():
    """The post-norm arm has no final norm: 256 parameters fewer than the v1."""
    assert parameter_breakdown(build_model(NetConfig(norm_placement="post"), seed=0))["total"] == V1_TOTAL - 256


def test_trunk_options_are_validated():
    with pytest.raises(ValueError, match="norm_placement"):
        NetConfig(norm_placement="sandwich").validate()
    with pytest.raises(ValueError, match="ffn_activation"):
        NetConfig(ffn_activation="gelu").validate()  # N1's; F6 replaced it
    with pytest.raises(TypeError):
        NetConfig(head_activation="gelu")  # F6 rules the heads' (Mish, CHECKPOINT 8): no switch
    with pytest.raises(ValueError, match="norm_eps"):
        NetConfig(norm_eps=0.0).validate()
    with pytest.raises(TypeError):
        NetConfig(linear_bias=True)  # split into qkv_bias and out_ffn_bias (F5)


# --- the FFN's activation (F6) --------------------------------------------------------------------


def _mish(x: torch.Tensor) -> torch.Tensor:
    """Mish (Misra 2019): ``x tanh(softplus(x))``."""
    return x * torch.tanh(torch.log1p(torch.exp(x)))


@pytest.mark.parametrize("activation", ["mish", "squared_relu"])
def test_the_ffn_is_its_formula(activation):
    """F6: ``FFN(x) = W_out act(W_in x + b_in) + b_out`` - Mish for v1, squared ReLU ``relu(x)^2``
    as 6y's arm - by hand in FP64; and it is not stage N1's GELU."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, ffn_activation=activation), seed=6)
    ffn = model.trunk.blocks[0].ffn
    x = torch.randn(3, 7, 256, generator=torch.Generator().manual_seed(6))
    pre = x.double() @ ffn.lin_in.weight.double().T + ffn.lin_in.bias.double()
    act = _mish(pre) if activation == "mish" else pre.clamp(min=0.0) ** 2
    expected = act @ ffn.lin_out.weight.double().T + ffn.lin_out.bias.double()
    out = ffn(x)
    assert torch.allclose(out.double(), expected, atol=1e-4)
    assert not torch.allclose(out, ffn.lin_out(F.gelu(ffn.lin_in(x))), atol=1e-3), "not N1's GELU"


@pytest.mark.parametrize("ffn_activation", ["mish", "squared_relu"])
def test_every_heads_hidden_layer_is_mish(ffn_activation):
    """F6 (CHECKPOINT 8): Mish "in every head's hidden layers, the value head included" - the
    value head's MLP is the only head with one (the others are linear or bilinear) - by hand in
    FP64, under the v1's FFN and under 6y's squared ReLU alike; and it is not stage N1's GELU."""
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, ffn_activation=ffn_activation), seed=7).eval()
    pooled = torch.randn(4, 256, generator=torch.Generator().manual_seed(7))
    hidden = model.value.hidden(pooled).double()
    expected = _mish(hidden) @ model.value.out.weight.double().T + model.value.out.bias.double()
    out = model.value(pooled)
    assert torch.allclose(out.double(), expected, atol=1e-5)
    assert not torch.allclose(out, model.value.out(F.gelu(model.value.hidden(pooled))), atol=1e-3), "not N1's GELU"


def test_squared_relu_reaches_no_head():
    """6y's squared ReLU is "the trunk-only arm" (F6, CHECKPOINT 8). The arm's network and the
    v1's, built from one seed, have the same parameters (an activation draws nothing); over one
    trunk output their heads give the same outputs, bit for bit - a head that followed the FFN's
    switch would not - while the trunks differ."""
    batch = collate([synthetic_state(8, 4), synthetic_state(9, 2)])
    v1 = build_model(NetConfig(trunk_dtype="fp32", n_layers=1), seed=8).eval()
    arm = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, ffn_activation="squared_relu"), seed=8).eval()
    arm_params = dict(arm.named_parameters())
    assert all(torch.equal(p, arm_params[name]) for name, p in v1.named_parameters())
    h = torch.randn(batch.batch_size, batch.n_tokens, 256, generator=torch.Generator().manual_seed(8))
    a, b = v1.heads(h, batch), arm.heads(h, batch)
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        assert (x is None and y is None) or torch.equal(x, y), field.name
    assert not torch.allclose(v1.trunk_output(batch), arm.trunk_output(batch), atol=1e-4), "the arm is the trunk's"
