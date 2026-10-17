"""B17, the key-side Shaw term (spec section 4.1; register B17; ablation 2), and section 15.2's
B17 row: "Key-side Shaw term via precomputed query-class products = direct per-pair computation,
on small positions | The gather implementation is exact | B17".

* **The class set against its definition.** B17's class of a pair is B15's entry restricted to
  "the exact near classes plus the line classes" (the placeholder: the exact board-distance
  classes' cells with |Delta-file|, |Delta-rank| <= 2, and B15's line entries) - written out here
  from ``test_displacement.Reference``, B15's own written-out layout: 1,493 classes.
* **Section 15.2's row.** The term computed pair by pair here - each query dotted with its pair's
  class vector, from the hand-written classes - equals the module's gather (precomputed products,
  then a gather by class) and its direct form, over every key, query blocks, key selections, row
  sets and head subsets; FP64 to 1e-12, FP32 to FP32 roundoff; on section 15.2's constructions
  (every variant's board size, the far lattice with its lines, register arms), several seeds, the
  vectors at their initial scale and at unit scale. Through attention: the CPU paths (the gather)
  equal attention with the direct term materialised - outputs and gradients, the vectors'
  included; B14's score function reads the products inside the attention and equals it too; the
  whole network agrees across every implementation, and the cached path equals the full pass.
* **The query is unrotated** - so the term is relative: translation and board order leave the
  network as it was with B17 on, which a rotated query would not.
* **The vectors**: 6,115,328 per head per layer (764,416 shared by the heads), from a generator of
  their own; off is the B15 network, bit for bit; every layer's vectors reach the outputs.
* **Cost** (section 15.3): the products are ``C / T`` of ``Q K^T`` - 0.51 at 44 boards and 0.094 at
  240 - and the implementation issues exactly them (counted FLOPs).
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch

from t5d.net import NetConfig, build_model, collate, parameter_breakdown, permute_boards, synthetic_state, translate
from t5d.net.attention import attention_block_sparse, attention_chunked, attention_explicit, attention_sdpa
from t5d.net.cache import CachedEvaluator, reference_output
from t5d.net.cost import count_flops, qk_flops, shaw_flops
from t5d.net.displacement import DisplacementGeometry
from t5d.net.flex import ScoreInputs, attention_flex, make_score_mod, materialise_score
from t5d.net.rope import apply_rope
from t5d.net.shaw import compact_classes, direct_term, gather_term, n_shaw_classes
from t5d.net.tree import Multiverse, SearchTree
from tests.net.test_displacement import CONSTRUCTIONS, Reference, reference_bias, reference_entries

# --- the class set, written out ---------------------------------------------------------------------


def reference_classes(cfg: NetConfig) -> tuple[dict[int, int], int]:
    """B17's class of each of B15's entries, from ``Reference``'s layout (``test_displacement.py``):
    the exact classes' cells with |dx|, |dy| <= ``shaw_near``, numbered class by class, then the
    line entries in order; ``(classes, C)`` - an entry missing from ``classes`` has none."""
    ref = Reference(cfg.disp_far_edges, cfg.disp_line_max, cfg.disp_register_pairs)
    near = cfg.shaw_near
    classes: dict[int, int] = {}
    for bc in range(45):  # "the exact near classes"
        for dx, dy in itertools.product(range(-near, near + 1), repeat=2):
            classes[bc * 225 + (dx + 7) * 15 + (dy + 7)] = len(classes)
    for entry in range(ref.lines, ref.registers):  # "plus the line classes"
        classes[entry] = len(classes)
    return classes, len(classes)


def reference_class_of_pairs(batch, cfg: NetConfig) -> torch.Tensor:
    """Every pair's class ``(B, L, L)`` from ``reference_entries`` (B15's, by hand) - ``C`` where
    the pair has none."""
    classes, n = reference_classes(cfg)
    entries = reference_entries(batch, cfg)
    lut = torch.full((int(entries.max()) + 1,), n, dtype=torch.long)
    for entry, c in classes.items():
        if entry < lut.numel():
            lut[entry] = c
    return lut[entries]


def direct_reference(q: torch.Tensor, vectors: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """Section 15.2's "direct per-pair computation", in FP64: for every pair, the query dotted with
    its class's vector over sqrt(d_head) - zero for a pair with no class. ``q`` ``(B, H, L, d)``;
    ``vectors`` ``(H or 1, C, d)``; ``classes`` ``(B, L, L)``."""
    bsz, n_heads, length, d = q.shape
    out = torch.zeros(bsz, n_heads, length, length, dtype=torch.float64)
    v = vectors.double().expand(n_heads, -1, -1)
    for b, h in itertools.product(range(bsz), range(n_heads)):
        has = classes[b] < v.shape[1]
        i, j = torch.nonzero(has, as_tuple=True)
        out[b, h, i, j] = (q[b, h, i].double() * v[h, classes[b, i, j]]).sum(-1) / d**0.5
    return out


def _model(seed=0, layers=2, **flags):
    return build_model(NetConfig(trunk_dtype="fp32", n_layers=layers, shaw_key_term=True, **flags), seed=seed)


def test_the_class_set():
    """1,493 classes at the defaults (45 x 25 near cells + 368 lines) - section 4.1's ~1,400; the
    module's map from B15's entries is the written-out one under three settings; the classes a
    few displacements fall in, one by one."""
    for flags in (dict(), dict(shaw_near=1), dict(disp_line_max=4, shaw_near=3)):
        cfg = NetConfig(shaw_key_term=True, **flags)
        geom = DisplacementGeometry.from_config(cfg)
        compact, n = compact_classes(geom, cfg.shaw_near)
        want, n_ref = reference_classes(cfg)
        assert n == n_ref == n_shaw_classes(cfg)
        assert compact.tolist() == [want.get(e, n) for e in range(geom.n_entries + 1)], flags
    assert n_shaw_classes(NetConfig()) == 45 * 25 + 64 * 5 + 8 * 6 == 1_493
    geom, (compact, n) = DisplacementGeometry(), compact_classes(DisplacementGeometry(), 2)
    ref = Reference()

    def c(dx, dy, dp, dh, kq=0, kk=0):
        return int(compact[ref(kq, kk, dx, dy, dp, dh)])

    assert c(2, 1, 0, 0) < n and c(2, 0, 1, 0) < n and c(1, 2, 0, 4) < n, "knight and near relations, on and across boards"
    assert c(2, 0, 1, 0) != c(2, 0, 0, 0) != c(2, 0, -1, 0), "a class per board-distance class"
    assert c(3, 0, 0, 0) == n and c(7, 7, 2, 4) == n, "beyond the near window: no class"
    assert c(0, 0, 3, 0) < n and c(3, 3, 3, 6) < n and c(0, 0, 0, 2 * 40) < n, "the rook, dragon and clamped board lines"
    assert c(1, 2, 5, 1) == n, "a far cell: no class"
    assert c(0, 0, 0, 0, 1, 0) == n and c(0, 0, 0, 0, 2, 0) == n and c(0, 0, 0, 0, 3, 0) == n, "registers, globals, padding"
    assert c(0, 0, 0, 0) < n and c(0, 0, 0, 0) != c(1, 0, 0, 0), "the same square on the same board has its own class"


# --- section 15.2's row: the gather = the direct per-pair computation ----------------------------


@pytest.mark.parametrize("construction", list(CONSTRUCTIONS))
def test_gather_equals_direct(construction):
    """Section 15.2's B17 row, on the term itself: the module's gather (``gather_term``: P = q R^T,
    then gathered by class) and its direct form (``direct_term``) equal the per-pair computation
    written out here - FP64 to 1e-12, FP32 to FP32 roundoff - over every key, query blocks, key
    selections, row sets and head subsets; three seeds, the vectors at unit and initial scale,
    shared and per head."""
    for seed, (std, shared) in enumerate([(1.0, False), (0.1, False), (1.0, True)]):
        batch = collate(CONSTRUCTIONS[construction](40 + seed))
        length = batch.n_tokens
        model = _model(
            seed, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, shaw_init_std=std, shaw_shared_heads=shared
        )
        shaw = model.make_context(batch).for_layer(None, 1).shaw
        classes = reference_class_of_pairs(batch, model.cfg)
        rows = torch.arange(length).expand(batch.batch_size, length)
        assert torch.equal(shaw.classes(rows), classes), "the CPU paths' classes are the written-out ones"
        q = torch.randn(batch.batch_size, 8, length, 32, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        vectors = model.shaw.vectors[1].detach()
        ref = direct_reference(q, vectors, classes)
        padded = shaw.vectors  # (heads, C + 1, d): this layer's, the zero vector last
        assert torch.equal(padded[:, :-1], vectors) and not padded[:, -1].any()
        for compute in (gather_term, direct_term):
            got64 = compute(q, padded.double(), classes, 32**-0.5)
            assert got64.dtype == torch.float64 and torch.allclose(got64, ref, rtol=0, atol=1e-12), compute.__name__
            got32 = compute(q.float(), padded, classes, 32**-0.5)
            assert torch.allclose(got32.double(), ref, rtol=1e-5, atol=1e-6), (compute.__name__, float((got32.double() - ref).abs().max()))
        # Through ``ShawRows`` as the paths call it: blocks, key selections, row sets, head subsets.
        q32 = q.float()
        gen = torch.Generator().manual_seed(seed)
        for start, stop in ((0, 7), (30, 130), (length - 20, length)):
            span = torch.arange(start, stop).expand(batch.batch_size, -1)
            got = shaw.rows(q32[:, :, start:stop], span)
            assert torch.allclose(got.double(), ref[:, :, start:stop], rtol=1e-5, atol=1e-6)
        cols = torch.stack([torch.randperm(length, generator=gen)[:40] for _ in range(batch.batch_size)])
        heads = torch.tensor([1, 4, 6])
        got = shaw.rows(q32[:, heads, 10:30], torch.arange(10, 30).expand(batch.batch_size, -1), cols, heads)
        want = ref[:, heads, 10:30].gather(3, cols[:, None, None, :].expand(-1, 3, 20, -1))
        assert torch.allclose(got.double(), want, rtol=1e-5, atol=1e-6)
        picked = torch.stack([torch.randperm(length, generator=gen)[:25] for _ in range(batch.batch_size)])
        got = shaw.rows(q32.gather(2, picked[:, None, :, None].expand(-1, 8, -1, 32)), picked)
        assert torch.allclose(got.double(), ref.gather(2, picked[:, None, :, None].expand(-1, 8, -1, length)), rtol=1e-5, atol=1e-6)
        assert bool((ref != 0).any()), "the construction holds pairs with a class"


def _no_shaw(ctx, **changes):
    return replace(ctx, shaw=None, **changes)


def _qkv(batch, seed):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch.batch_size, 8, batch.n_tokens, 32)
    return tuple(torch.randn(*shape, generator=gen) for _ in range(3))


#: The CPU paths, as test_chunked.py runs them.
CPU_PATHS = {
    "block_sparse": lambda q, k, v, ctx: attention_block_sparse(q, k, v, replace(ctx, block_rows=66, sparse_max_fraction=1.0)),
    "block_sparse_mixed": lambda q, k, v, ctx: attention_block_sparse(
        q, k, v, replace(ctx, block_rows=2 * 66 + 5, sparse_max_fraction=0.3)
    ),
    "chunked": lambda q, k, v, ctx: attention_chunked(q, k, v, replace(ctx, block_rows=66)),
    "sdpa": attention_sdpa,
    "explicit": lambda q, k, v, ctx: attention_explicit(q, k, v, replace(ctx, chunk=37)),
}


@pytest.mark.grad
@pytest.mark.parametrize(
    "flags",
    [dict(), dict(routing_table=True), dict(board_register_spatial="position_free"), dict(shaw_shared_heads=True)],
    ids=["b9", "routing", "regs-free", "shared-heads"],
)
@pytest.mark.parametrize("construction", ["far", "small_boards_mixed", "registers_4_0"])
def test_cpu_paths_equal_the_direct_term(construction, flags):
    """The CPU paths gather the term block by block (key selections, free rows): their outputs and
    gradients - Q, K, V and the vectors - equal attention with the term computed pair by pair
    (``direct_term``, differentiable in Q and the vectors) and passed as a dense bias; two seeds,
    the vectors at unit scale."""
    for seed in (0, 1):
        batch = collate(CONSTRUCTIONS[construction](60 + seed))
        model = _model(seed, layers=1, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, shaw_init_std=1.0, **flags)
        qkv = _qkv(batch, seed)
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(batch.batch_size, 8, batch.n_tokens, 32, generator=gen) * batch.valid[:, None, :, None]
        valid = batch.valid[:, None, :, None]
        rows = torch.arange(batch.n_tokens).expand(batch.batch_size, -1)

        def run(attend, materialised: bool, model=model, batch=batch, qkv=qkv, w=w, rows=rows):
            model.zero_grad(set_to_none=True)
            ctx = model.make_context(batch).for_layer(None, 0)
            leaves = [t.clone().requires_grad_(True) for t in qkv]
            if materialised:
                term = direct_term(leaves[0], ctx.shaw.vectors, ctx.shaw.classes(rows), 32**-0.5)
                ctx = _no_shaw(ctx, bias=term)
            out = attend(*leaves, ctx)
            (out * w).sum().backward()
            return out.detach(), [t.grad for t in leaves] + [model.shaw.vectors.grad.clone()]

        out_r, grads_r = run(attention_explicit, materialised=True)
        assert grads_r[-1].abs().sum() > 0
        for name, attend in CPU_PATHS.items():
            out, grads = run(attend, materialised=False)
            assert torch.allclose(out * valid, out_r * valid, atol=1e-5), (name, float(((out - out_r) * valid).abs().max()))
            for i, (ga, gb) in enumerate(zip(grads_r, grads, strict=True)):
                tol = 1e-4 + 1e-5 * float(ga.abs().max())
                assert torch.allclose(ga, gb, atol=tol, rtol=0.0), (name, i, float((ga - gb).abs().max()))


@pytest.mark.parametrize("construction", ["far", "variant_sizes", "far_variant_sizes", "registers_0_16", "padded_1_2_5"])
def test_the_score_function_reads_the_products(construction):
    """B14's score function (``flex.py``) with B17: its term at every pair - B15's entry and the
    query's precomputed product with its pair's class, the class from B15's entry computed inside -
    is B15's bias materialised by hand plus the direct per-pair term (FP32 roundoff), and attention
    through it equals attention with both materialised; two seeds, at unit and initial scale."""
    for seed, std in enumerate((1.0, 0.1)):
        batch = collate(CONSTRUCTIONS[construction](70 + seed))
        model = _model(seed, regs_per_board=batch.regs_per_board, global_regs=batch.global_regs, shaw_init_std=std)
        ctx = model.make_context(batch).for_layer(None, 1)
        q, k, v = _qkv(batch, seed)
        inp = ScoreInputs.from_context(ctx, 8, q)
        term = materialise_score(make_score_mod(inp), batch.batch_size, 8, batch.n_tokens, batch.n_tokens)
        shaw = direct_reference(q, model.shaw.vectors[1].detach(), reference_class_of_pairs(batch, model.cfg))
        ref = reference_bias(model, batch, 1).double() + shaw
        assert torch.allclose(term.double(), ref, rtol=1e-5, atol=2e-6), float((term.double() - ref).abs().max())
        with pytest.raises(ValueError, match="needs the queries"):
            ScoreInputs.from_context(ctx, 8)
        valid = batch.valid[:, None, :, None]
        through = attention_flex(q, k, v, ctx)
        bare = replace(ctx.structure, disp=None, disp_index=None, disp_table=None)
        materialised = attention_explicit(q, k, v, replace(ctx, structure=bare, shaw=None, bias=ref.float()))
        assert torch.allclose(through * valid, materialised * valid, atol=1e-5), float(((through - materialised) * valid).abs().max())


@pytest.mark.parametrize(
    "flags",
    [
        dict(),
        dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, shaw_shared_heads=True),
        dict(attn_scope="block_causal", typed_edge_bias=True),
        dict(board_register_spatial="position_free", global_register_pairs="anchored", rope_per_head=True, shaw_init_std=1.0),
    ],
    ids=["b17", "n2-all-shared", "block-causal", "regs-free-anchored-b5"],
)
def test_every_implementation_is_one_function(flags):
    """The whole network with B17 through every attention implementation - the dense reference
    computing the term pair by pair, the others gathering it, the score function reading the
    products - one function."""
    batch = collate([synthetic_state(80, 7, n_timelines=3), synthetic_state(81, 2), synthetic_state(82, 4, width=5, height=5)])
    outs = {
        impl: _model(seed=8, attn_impl=impl, **flags).eval()(batch, return_trunk=True)
        for impl in ("dense", "explicit", "sdpa", "chunked", "block_sparse", "flex")
    }
    ref = outs["dense"]
    valid = batch.valid[..., None]
    for impl, out in outs.items():
        assert torch.allclose(out.trunk * valid, ref.trunk * valid, atol=2e-5), (impl, float(((out.trunk - ref.trunk) * valid).abs().max()))
        assert torch.allclose(out.value_logits, ref.value_logits, atol=1e-4), impl


@pytest.mark.parametrize("init_std", [0.1, 1.0])
def test_the_cached_path_equals_the_full_pass(init_std):
    """Section 15.2's row 1 with B17 on: a frozen board's rows get the term from its own queries,
    layer by layer (``compute_board``), the live rows in the live pass - cached = one full pass
    along a path that supersedes boards (B16 on too: the cached configuration's default)."""
    model = build_model(
        NetConfig(trunk_dtype="fp32", n_layers=2, attn_scope="block_causal", shaw_key_term=True, shaw_init_std=init_std), seed=9
    ).eval()
    root = Multiverse.from_state(synthetic_state(1009, 12, n_timelines=4), seed=9)
    tree = SearchTree(CachedEvaluator(model), root)
    node = tree.root
    ps = sorted(root.heads())
    for move in [None, ("physical", ps[0]), ("physical", ps[1]), ("physical", ps[0])]:
        if move is not None:
            node = tree.child(node, move)
        cached, ref = tree.evaluate(node), reference_output(model, node.state)
        assert torch.allclose(cached.trunk, ref.trunk, atol=2e-5), float((cached.trunk - ref.trunk).abs().max())
    assert tree.evaluator.store.computed["supersession"] >= 2


# --- the unrotated query ---------------------------------------------------------------------------


def test_the_query_is_unrotated():
    """The term is relative: with B17 on, translating every timeline and every board by +2j
    half-turns leaves the network as it was (bitwise, with RoPE's recentring; to roundoff on raw
    coordinates), and storing the boards in another order permutes it. A rotated query would not
    be: on raw coordinates the term from rotated queries moves under the same translation."""
    state = synthetic_state(90, 9, n_timelines=4)
    for recenter in (True, False):
        model = _model(seed=9, rope_recenter=recenter).eval()
        a = model(collate([state]), return_trunk=True)
        for dp, dh in ((3, 8), (-7, 100)):
            b = model(collate([translate(state, dp=dp, dh=dh)]), return_trunk=True)
            if recenter:
                assert torch.equal(a.value_logits, b.value_logits) and torch.equal(a.board_logits, b.board_logits)
            else:
                assert torch.allclose(a.trunk, b.trunk, atol=2e-5) and torch.allclose(a.value_logits, b.value_logits, atol=1e-4)
    order = np.random.default_rng(9).permutation(state.n_boards)
    perm = torch.from_numpy(np.concatenate([np.arange(66) + 66 * int(o) for o in order] + [66 * state.n_boards + np.arange(16)]))
    model = _model(seed=9).eval()
    a, c = model(collate([state]), return_trunk=True), model(collate([permute_boards(state, order)]), return_trunk=True)
    assert torch.allclose(c.trunk[0], a.trunk[0][perm], atol=2e-5)
    # The rotated query, on raw coordinates: the same pair's term moves with the translation.
    raw = _model(seed=9, rope_recenter=False)
    q = torch.randn(1, 8, collate([state]).n_tokens, 32, generator=torch.Generator().manual_seed(9))
    rows = torch.arange(q.shape[2])[None]
    terms = {}
    for name, s in (("base", state), ("moved", translate(state, dp=5, dh=12))):
        ctx = raw.make_context(collate([s])).for_layer(None, 0)
        terms[name] = (ctx.shaw.rows(q, rows), ctx.shaw.rows(apply_rope(q, ctx.cos, ctx.sin), rows))
    assert torch.equal(terms["base"][0], terms["moved"][0]), "unrotated: relative"
    assert not torch.allclose(terms["base"][1], terms["moved"][1], atol=1e-3), "rotated: absolute"


# --- the vectors -----------------------------------------------------------------------------------


def test_the_vectors_count_and_their_generator():
    """6,115,328 parameters per head per layer at the v1's dimensions (1,493 x 32 x 8 x 16; 764,416
    shared by the heads) - no document states it (spec issue 218); drawn from B17's own generator,
    so the flag's off position is the B15 network bit for bit, and no other flag changes them."""
    on = parameter_breakdown(build_model(NetConfig(shaw_key_term=True)))
    off = parameter_breakdown(build_model(NetConfig()))
    assert on["attn.shaw"] == 1_493 * 32 * 8 * 16 == 6_115_328 and on["total"] - off["total"] == 6_115_328 and off["attn.shaw"] == 0
    shared = parameter_breakdown(build_model(NetConfig(shaw_key_term=True, shaw_shared_heads=True)))
    assert shared["attn.shaw"] == 1_493 * 32 * 16 == 764_416
    assert abs(on["attn.shaw"] / off["total"] - 0.266) < 0.001 and abs(on["attn.shaw"] / on["total"] - 0.210) < 0.001
    a, b = _model(seed=11, layers=3), build_model(NetConfig(trunk_dtype="fp32", n_layers=3), seed=11)
    p_a, p_b = dict(a.named_parameters()), dict(b.named_parameters())
    assert set(p_a) - set(p_b) == {"shaw.vectors"}
    assert all(torch.equal(p, p_a[name]) for name, p in p_b.items())
    batch = collate([synthetic_state(11, 4)])
    with torch.no_grad():
        a.shaw.vectors.zero_()
    assert torch.allclose(a.eval()(batch, return_trunk=True).trunk, b.eval()(batch, return_trunk=True).trunk, atol=1e-6), (
        "zero vectors: B15"
    )
    base = _model(seed=11, layers=3).shaw.vectors
    for flags in (dict(typed_edge_bias=True, board_pair_smolgen=True), dict(attn_scope="block_causal"), dict(norm_placement="post")):
        assert torch.equal(_model(seed=11, layers=3, **flags).shaw.vectors, base), flags
    assert abs(float(base.std()) - 0.1) < 0.002, "N(0, 0.1)"
    with pytest.raises(ValueError, match="on top of B15"):
        NetConfig(shaw_key_term=True, displacement_bias=False).validate()
    for bad in (dict(shaw_near=-1), dict(shaw_near=8), dict(shaw_init_std=float("inf"))):
        with pytest.raises(ValueError, match="shaw_"):
            NetConfig(shaw_key_term=True, **bad).validate()


def test_every_layers_vectors_reach_the_outputs():
    """Perturbing one layer's vectors - the first, the last - moves the outputs; the others'
    layers read their own."""
    model = _model(seed=12, layers=3).eval()
    batch = collate([synthetic_state(12, 6)])
    base = model(batch, return_trunk=True)
    saved = model.shaw.vectors.detach().clone()
    for layer in (0, 2):
        with torch.no_grad():
            model.shaw.vectors[layer] += torch.randn(saved.shape[1:], generator=torch.Generator().manual_seed(layer))
        out = model(batch, return_trunk=True)
        with torch.no_grad():
            model.shaw.vectors.copy_(saved)
        assert not torch.allclose(out.trunk, base.trunk, atol=1e-4), layer
        assert not torch.allclose(out.policy_logits[out.move_valid], base.policy_logits[base.move_valid], atol=1e-4), layer
    ctx = model.make_context(batch)
    assert not torch.equal(ctx.for_layer(None, 0).shaw.vectors, ctx.for_layer(None, 1).shaw.vectors)


# --- cost (section 15.3) -----------------------------------------------------------------------------


def test_the_cost_row():
    """Section 15.3: "~0.5x the QK^T product at 44 boards, ~9% at 240, with ~1,400 classes". The
    products cost C / T of Q K^T: with 1,493 classes 0.511 at 44 boards (T = 2,920) and 0.094 at 240
    (T = 15,856). And the implementation issues exactly them: on the one-block path the counted
    FLOPs with B17 exceed those without by ``2 x body x (C + 1) x d_head x H x L`` - one product per
    body query and class (and the "no class" column), nothing else."""
    cfg = NetConfig(shaw_key_term=True)
    for boards, ratio in ((44, 0.511), (240, 0.0942)):
        t = boards * 66 + 16
        assert abs(shaw_flops(t, cfg) / qk_flops(t, t, cfg) - ratio) < 0.001, boards
    assert abs(shaw_flops(2_920, cfg) / qk_flops(2_920, 2_920, cfg) - 0.5) < 0.05, "section 15.3's ~0.5x"
    assert abs(shaw_flops(15_856, cfg) / qk_flops(15_856, 15_856, cfg) - 0.09) < 0.01, "section 15.3's ~9%"
    batch = collate([synthetic_state(13, 5)])
    counted = {}
    for on in (True, False):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl="sdpa", shaw_key_term=on), seed=13).eval()
        _, counted[on] = count_flops(lambda model=model: model(batch))
    body = int(batch.n_body[0])
    assert counted[True] - counted[False] == 2 * body * (1_493 + 1) * 32 * 8 * 2
