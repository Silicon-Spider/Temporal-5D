"""Register G11: structure built once per forward, and B15's layer-shared bias assembled once.

Spec section 10.2 (2026-11-15 review), requirement 1: "Structure is computed once per forward,
not once per layer ... It is built once, in compact form, and every layer reads it. Only the
learned tables differ by layer"; requirement 2: a materialised bias is "affordable once per
forward for a layer-shared table (section 4.1's arm), not sixteen times over, once per layer".
What is built here keeps what no layer changes in two per-forward memos
(``AttnContext.memo``, ``AttnStructure.memo``) and, at inference, takes each layer's keys and
queries once for the layer instead of once per query block. Held:

* **the hoisted inference path is the as-built path, bit for bit.** The as-built float
  operations still run under autograd (so that every gradient is accumulated as it always was),
  and the memos can be switched off by replacing ``memoised``; a forward with autograd on, no
  parameter requiring a gradient and nothing memoised is therefore the network as it was before
  this work, operation for operation, and the ordinary inference forward equals it with
  ``torch.equal`` - every output, every fast path and the explicit one, both precisions, the
  arms that change the structure's biases. Bitwise is the right bar: the operations kept are
  integer and boolean ones and selections of rows and heads from elementwise results, which
  commute with the selection exactly (no sum is taken in another order);
* the memos follow a layer's copy and no other copy;
* a named selection of bias rows is the unnamed one, bit for bit, and B15's lookups by prebuilt
  sides and by one index are the lookups by gathers;
* under ablation 2's layer-shared arm B15's rows are assembled once per forward at inference -
  not under autograd, never beyond the byte bound, and never written into;
* after the first layer no layer runs the functions that build the position's structure
  (the operator count's call paths say which functions ran).

The cross-build comparison - this tree against the tree before the work, forward, gradient and
cached paths, the 16-layer network at 24, 44 and 60 boards included - is recorded in
``docs/MEASUREMENTS.md`` ("G11 - the kernel's widths, where a layer's operators come from").

Protocol (section 15.2): random weights at ``build_model``'s scale, model seeds 0-2, one position
and padded batches of 4 to 12 boards, 2 and 3 layers, FP32 and BF16.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net import attention as attention_module
from t5d.net import displacement as displacement_module
from t5d.net import structure as structure_module
from t5d.net.bench import CONFIGS
from t5d.net.displacement import N_CODES, lookup, lookup_sides, pair_sides
from t5d.net.opcount import NO_FUNCTION, call_path, count_forward, count_ops, main, measure

#: The paths that read the structure's rows block by block or in one block.
FAST = ("block_sparse", "chunked", "sdpa")
ALL_BIASES = dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)
#: The arms that change what the structure adds, each beside the v1 default.
ARMS = {
    "v1": dict(),
    "layer-shared": dict(disp_layer_shared=True),
    "no-far-tables": dict(disp_far_axis=False),
    "no-table": dict(displacement_bias=False),
    "every-bias": ALL_BIASES,
    "every-bias-layer-shared": ALL_BIASES | dict(disp_layer_shared=True),
    "shaw": dict(shaw_key_term=True),
    "block-causal": dict(attn_scope="block_causal"),
    "no-board-registers": dict(regs_per_board=0),
    "no-global-registers": dict(global_regs=0),
    "registers-read-themselves": dict(masked_head_registers="self"),
    "one-board-blocks": dict(attn_block_boards=1),
}


def _states(seed: int, sizes: tuple[int, ...], **flags):
    registers = {k: v for k, v in flags.items() if k in ("regs_per_board", "global_regs")}
    return collate([synthetic_state(100 * seed + n, n, **registers) for n in sizes])


def _outputs(out) -> dict[str, torch.Tensor]:
    return {name: value for name, value in vars(out).items() if isinstance(value, torch.Tensor)}


def _as_built(model, batch):
    """One forward of ``model`` as the network ran before this work: autograd on, so that every
    float operation is the as-built one (per query block), with no parameter requiring a
    gradient, so that the masks take the inference path; and nothing memoised."""
    saved = (structure_module.memoised, attention_module.memoised)
    requires = [p.requires_grad for p in model.parameters()]

    def not_kept(memo, key, make):
        return make()

    try:
        structure_module.memoised = attention_module.memoised = not_kept
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.enable_grad():
            return model(batch, return_trunk=True)
    finally:
        structure_module.memoised, attention_module.memoised = saved
        for p, flag in zip(model.parameters(), requires, strict=True):
            p.requires_grad_(flag)


# --- the hoisted path is the as-built path ---------------------------------------------------------


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("impl", [*FAST, "explicit"])
def test_the_hoisted_forward_equals_the_as_built_one_bit_for_bit(impl, arm):
    flags = ARMS[arm]
    for seed, sizes, dtype in ((0, (9,), "fp32"), (1, (7, 3, 12), "bf16"), (2, (5, 8), "fp32")):
        if seed and (impl == "explicit" or (seed == 2 and arm not in ("v1", "layer-shared"))):
            continue  # one position on the explicit path; the third case for the v1 and the arm this work changes most
        cfg = NetConfig(trunk_dtype=dtype, n_layers=3, attn_impl=impl, attn_recompute=False, **flags)
        model = build_model(cfg, seed=seed).eval()
        batch = _states(seed, sizes, **flags)
        with torch.no_grad():
            hoisted = _outputs(model(batch, return_trunk=True))
        as_built = _outputs(_as_built(model, batch))
        assert hoisted.keys() == as_built.keys()
        for name, value in hoisted.items():
            assert torch.equal(value, as_built[name]), f"{impl}, {arm}, seed {seed}, {dtype}: {name} differs from the as-built forward"


def test_the_as_built_reference_runs_what_it_says(monkeypatch):
    """The reference of the test above is not the hoisted path by another name: under it nothing
    is kept in either memo and every query block takes its own keys; the inference forward keeps
    the memos and takes each head group's keys once a layer."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_recompute=False)
    model, batch = build_model(cfg, seed=0).eval(), _states(0, (9,))
    memos: list[dict] = []
    make_context = model.make_context

    def spy_context(*args, **kwargs):
        ctx = make_context(*args, **kwargs)
        memos.append(ctx.memo)
        memos.append(ctx.structure.memo)
        return ctx

    heads_calls: list[int] = []
    heads = attention_module._Keys.heads

    def spy_heads(self, which):
        heads_calls.append(1)
        return heads(self, which)

    monkeypatch.setattr(model, "make_context", spy_context)
    monkeypatch.setattr(attention_module._Keys, "heads", spy_heads)
    with torch.no_grad():
        model(batch)
    kept, per_layer = [len(m) for m in memos], len(heads_calls) // cfg.n_layers
    assert all(n > 0 for n in kept), "the inference forward fills both memos"
    assert per_layer == 2, "one set of keys per head group per layer (the move-graph heads, the dense heads)"
    memos.clear()
    heads_calls.clear()
    _as_built(model, batch)
    assert [len(m) for m in memos] == [0, 0], "the as-built reference keeps nothing"
    blocks = -(-9 // cfg.attn_block_boards)
    assert len(heads_calls) // cfg.n_layers == 2 * blocks + 1, "as built: per query block and group, and once for the free rows"


@pytest.mark.grad
@pytest.mark.parametrize("impl", FAST)
def test_under_autograd_only_what_carries_no_gradient_is_kept(impl):
    """A memo holds integers, booleans and Python values - never a tensor a gradient flows
    through - also under ablation 2's layer-shared arm, whose assembled rows are kept at
    inference only: a kept tensor would be one graph node for every layer, and a block
    recomputed in the backward pass must run what its forward ran."""
    for flags in (dict(), dict(disp_layer_shared=True), ALL_BIASES | dict(disp_layer_shared=True)):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, **flags), seed=1).train()
        batch = _states(1, (6, 3))
        ctx = model.make_context(batch)
        x = model.embed_tokens(batch)
        out = model.trunk(x, ctx)
        out.square().mean().backward()
        assert model.displacement.table.grad is not None and bool(model.displacement.table.grad.abs().sum() > 0)

        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (tuple, list)):
                for item in value:
                    yield from tensors(item)
            elif hasattr(value, "__dict__"):
                for item in vars(value).values():
                    yield from tensors(item)

        for memo in (ctx.memo, ctx.structure.memo):
            assert memo, "the memos are used under autograd too: integers are safe to keep"
            for key, value in memo.items():
                for t in tensors(value):
                    assert not t.requires_grad and (not t.is_floating_point() or t.dim() == 0), (impl, key, t.dtype, t.shape)
            assert not any(isinstance(key, tuple) and key[0] == "disp" for key in memo), "no assembled rows under autograd"


# --- the memos --------------------------------------------------------------------------------------


def test_a_memo_follows_a_layer_s_copy_and_no_other():
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, board_pair_smolgen=True, attn_scope="block_causal"), seed=0).eval()
    batch = _states(0, (5,))
    ctx = model.make_context(batch)
    h = torch.randn(1, batch.n_tokens, 256, generator=torch.Generator().manual_seed(0))
    layer = ctx.for_layer(h, 1)
    assert layer is not ctx and layer.structure is not ctx.structure
    assert layer.memo is ctx.memo and layer.structure.memo is ctx.structure.memo, "a layer changes nothing the memos hold"
    assert ctx.structure.for_frozen_rows().memo is ctx.structure.memo
    ctx.memo["probe"] = ctx.structure.memo["probe"] = 1
    for other in (replace(ctx, block_rows=66), replace(ctx, mask=None), layer.densified()):
        assert other.memo == {}, "any other copy may change what a memo was built from: it starts an empty one"
    assert replace(ctx.structure, registers="self").memo == {}
    plain = build_model(NetConfig(trunk_dtype="fp32", n_layers=1, displacement_bias=False, move_graph_heads=0), seed=0)
    bare = plain.make_context(batch)
    assert bare.structure is None and bare.for_layer(h, 0) is bare, "nothing layer-dependent: the context itself"


@pytest.mark.parametrize("arm", ["v1", "layer-shared", "no-far-tables", "every-bias", "every-bias-layer-shared", "block-causal"])
def test_a_named_selection_is_the_unnamed_one(arm):
    """``bias_rows(key=...)`` and ``bias_rows_at(key=...)``: the same tensor bit for bit as without
    the name, on first use and from the memo - over every key, over a key selection, on a head
    group, for a row span and for a row tensor."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, **ARMS[arm])
    model, batch = build_model(cfg, seed=2).eval(), _states(2, (7, 4))
    h = torch.randn(batch.batch_size, batch.n_tokens, cfg.d_model, generator=torch.Generator().manual_seed(2))
    gen = torch.Generator().manual_seed(3)
    cols = torch.stack([torch.randperm(batch.n_tokens, generator=gen)[:40].sort().values for _ in range(batch.batch_size)])
    valid = torch.ones_like(cols, dtype=torch.bool)
    rows = torch.stack([torch.randperm(batch.n_tokens, generator=gen)[:9] for _ in range(batch.batch_size)])
    heads = torch.tensor([1, 4, 6])
    for layer in (0, 1):
        plain = model.make_context(batch).for_layer(h, layer).structure
        named = model.make_context(batch).for_layer(h, layer).structure
        cases = {
            "span": lambda st, key: st.bias_rows(66, 200, key=key),
            "span-heads": lambda st, key: st.bias_rows(0, 66, heads=heads, key=key),
            "span-cols": lambda st, key: st.bias_rows(130, 264, heads=heads, cols=cols, cols_valid=valid, key=key),
            "rows": lambda st, key: st.bias_rows_at(rows, None, heads=heads, key=key),
        }
        for what, ask in cases.items():
            want = ask(plain, None)
            first, again = ask(named, ("test", what)), ask(named, ("test", what))
            assert torch.equal(first, want) and torch.equal(again, want), (arm, layer, what)
        assert plain.memo.keys() <= {"heads4", ("heads4", id(heads))}, "an unnamed selection keeps no rows and no sides"
        kept = [key for key in named.memo if isinstance(key, tuple) and key[0] in ("sides", "disp")]
        assert len([key for key in kept if key[0] == "sides"]) == len(cases)
        assert bool([key for key in kept if key[0] == "disp"]) == cfg.disp_layer_shared, (
            "assembled rows are kept under the layer-shared arm alone"
        )


@pytest.mark.grad
def test_the_lookups_by_sides_and_by_one_index_are_the_lookups_by_gathers():
    """B15's two lookups, restated: the entries through prebuilt sides (a row span by slicing, a
    row tensor by gathering) equal the entries written out from the index here; and the per-axis
    term by one index (inference) equals the two gathers (kept under autograd) bit for bit."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=1)
    model, batch = build_model(cfg, seed=0).eval(), _states(0, (20, 3))
    st = model.make_context(batch).for_layer(None, 0).structure  # under autograd: its far table carries a gradient
    index, class_map = st.disp_index, st.disp.class_map
    bsz, length = batch.batch_size, batch.n_tokens
    gen = torch.Generator().manual_seed(1)
    cols = torch.stack([torch.randperm(length, generator=gen)[:300] for _ in range(bsz)])
    span = (70, 701)
    rows = torch.arange(*span).expand(bsz, -1)
    b = torch.arange(bsz)[:, None, None]
    for keys in (None, cols):
        slot_k = index.slot if keys is None else index.slot.gather(1, keys)
        code_k = index.code if keys is None else index.code.gather(1, keys)
        tile = index.tiles[b, index.slot[:, span[0] : span[1], None], slot_k[:, None, :]]
        want = class_map.view(-1)[tile * N_CODES * N_CODES + index.code[:, span[0] : span[1], None] * N_CODES + code_k[:, None, :]]
        by_span = lookup_sides(index, class_map, pair_sides(index, None, keys, span))
        by_rows = lookup_sides(index, class_map, pair_sides(index, rows, keys))
        assert torch.equal(by_span, want) and torch.equal(by_rows, want) and torch.equal(lookup(index, class_map, rows, keys), want)
        assert torch.equal(st.disp_entries(rows, keys), want)
        for heads in (None, torch.tensor([0, 5])):
            with torch.no_grad():
                one_index = st.far_rows(rows, keys, heads)
            two_gathers = st.far_rows(rows, keys, heads)
            assert two_gathers.grad_fn is not None, "under autograd: the as-built gathers, into the far tables"
            assert torch.equal(one_index, two_gathers.detach())
            assert bool((one_index != 0).any()), "the batch holds far pairs: the comparison is of something"


# --- the layer-shared arm: assembled once per forward ----------------------------------------------


class _Lookups:
    """Counts B15's entry lookups (``displacement.lookup_sides``: one per assembly of bias rows)."""

    def __init__(self, monkeypatch) -> None:
        self.calls = 0
        real = displacement_module.lookup_sides

        def spy(index, table_map, sides):
            self.calls += 1
            return real(index, table_map, sides)

        monkeypatch.setattr(structure_module, "lookup_sides", spy)


@pytest.mark.parametrize("impl", [*FAST, "explicit", "dense"])
def test_the_layer_shared_arm_assembles_its_bias_once_per_forward(monkeypatch, impl):
    """At inference a forward of the layer-shared arm looks B15's entries up as often as a
    one-layer forward does - once per selection of rows, keys and heads - where the per-layer
    arm does it at every layer; and its outputs are those of the same network with nothing kept
    (``DISP_ONCE_BYTES`` 0: every layer assembles its own rows, as built)."""
    small = impl in ("explicit", "dense")
    batch = _states(4, (4, 2) if small else (9, 4))
    spy = _Lookups(monkeypatch)
    counts = {}
    for shared in (True, False):
        for layers in (1, 3):
            model = build_model(NetConfig(trunk_dtype="bf16", n_layers=layers, attn_impl=impl, disp_layer_shared=shared), seed=4).eval()
            spy.calls = 0
            out = model(batch, return_trunk=True)
            counts[(shared, layers)] = spy.calls
            if shared and layers == 3:
                kept = _outputs(out)
                with monkeypatch.context() as patch:
                    patch.setattr(structure_module, "DISP_ONCE_BYTES", 0)
                    per_layer = _outputs(model(batch, return_trunk=True))
                assert spy.calls == 4 * counts[(shared, layers)] > 0, "with nothing kept, every layer assembles its own"
                assert all(torch.equal(kept[name], per_layer[name]) for name in kept), "the kept rows change no output"
    assert counts[(True, 1)] > 0
    assert counts[(True, 3)] == counts[(True, 1)], f"{impl}: assembled once per forward ({counts})"
    assert counts[(False, 3)] == 3 * counts[(False, 1)], f"{impl}: the per-layer arm assembles at every layer ({counts})"


def test_the_kept_rows_are_bounded_and_never_written(monkeypatch):
    """What is kept stays within ``DISP_ONCE_BYTES`` - a selection that would pass it is
    assembled at every layer instead, with the same outputs - and B11's typed scalars, which are
    accumulated into the bias rows in place, go into a copy: after a forward every kept tensor
    still equals B15's rows assembled afresh."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=3, disp_layer_shared=True, typed_edge_bias=True)
    model, batch = build_model(cfg, seed=5).eval(), _states(5, (9, 4))
    memos: list[dict] = []
    make_context = model.make_context

    def spy_context(*args, **kwargs):
        ctx = make_context(*args, **kwargs)
        memos.append(ctx.structure.memo)
        return ctx

    monkeypatch.setattr(model, "make_context", spy_context)
    full = _outputs(model(batch, return_trunk=True))
    memo = memos[-1]
    everything = memo["disp_bytes"]
    kept = {key: value for key, value in memo.items() if isinstance(key, tuple) and key[0] == "disp"}
    assert (
        kept
        and everything == sum(t.numel() * t.element_size() for t in kept.values())
        and all(t.dtype == torch.float32 for t in kept.values())
    )
    # B15's rows alone, assembled afresh (the typed scalars are B11's, not B15's), for the
    # selections the block-sparse path names (``attention._FastPath``).
    untyped = replace(make_context(batch).structure, typed=None, typed_table=None)
    groups = dict(untyped.head_groups())
    compared = 0
    for (_, key), rows in kept.items():
        if key[0] == "group":
            want = untyped.bias_rows(key[2], key[3], heads=groups[key[1]])
        elif key[0] == "rows":
            want = untyped.bias_rows(key[1], key[2])
        else:
            continue
        assert torch.equal(rows, want), f"{key}: a kept tensor was written into"
        compared += 1
    assert compared > 0
    for budget in (everything // 3, 0):
        monkeypatch.setattr(structure_module, "DISP_ONCE_BYTES", budget)
        bounded = _outputs(model(batch, return_trunk=True))
        assert memos[-1].get("disp_bytes", 0) <= budget
        assert (budget == 0) == (not any(isinstance(key, tuple) and key[0] == "disp" for key in memos[-1]))
        assert all(torch.equal(full[name], bounded[name]) for name in full), f"budget {budget}: the bound changes no output"


# --- what the operator count shows ----------------------------------------------------------------

#: The functions that build what only the position decides: after the first layer none runs.
ONCE = (
    "attention._row_index",
    "attention._wide_selection.<lambda>",
    "attention._file_rank.<lambda>",
    "displacement.pair_sides",
    "displacement.PairSides.__post_init__",
    "structure.Pairs.span",
    "structure.AttnStructure.head_groups.build",
    "structure.AttnStructure.uncovered_registers.build",
)


@pytest.mark.parametrize("config", ["n2-default", "n2-default-sdpa", "n2-default-chunked", "b15-layer-shared"])
def test_no_layer_after_the_first_builds_the_position_s_structure(config):
    """Requirement 1, as far as the count can show it: the functions that build what no layer
    changes dispatch operators in the first layer and in none after it; every later layer
    dispatches fewer operators than the first, and all of them the same number. (The masks and
    B15's entries of a query block are rebuilt at every layer still: they are of size block x L,
    and kept for every block they would be the (L, L) tensors B14 exists not to hold.)"""
    layers = 3
    model = build_model(NetConfig(trunk_dtype="bf16", **(CONFIGS[config] | {"n_layers": layers})), seed=0).eval()
    count = count_forward(model, _states(0, (9,)), paths=True)

    def ran(layer: int) -> dict[str, int]:
        return {function: n for function, n, _ in count.functions(lambda scope: scope[:2] == ("layer", layer))}

    first, later = ran(0), [ran(i) for i in range(1, layers)]
    assert sum(first.get(function, 0) for function in ONCE) > 0, "the first layer builds it"
    for functions in later:
        assert not [function for function in ONCE if function in functions], config
    ops = count.layers()
    assert ops[0] > ops[1] == ops[2], (config, ops)
    if CONFIGS[config].get("disp_layer_shared"):
        assert "displacement.lookup_sides" in first and all("displacement.lookup_sides" not in f for f in later), "assembled once"
    else:
        assert all("displacement.lookup_sides" in f for f in later), "per-layer tables: each layer gathers its own"


def test_the_layer_shared_arm_dispatches_fewer_operators_a_layer():
    """No count is pinned; the order is: on either path a layer of the layer-shared arm dispatches
    fewer operators than a layer of the per-layer arm, at inference."""
    for shared, per_layer in (("b15-layer-shared", "n2-default"), ("b15-layer-shared-sdpa", "n2-default-sdpa")):
        rows = [measure(config, 9, dtype="bf16", layers=3) for config in (shared, per_layer)]
        assert rows[0]["layer_ops"][-1] < rows[1]["layer_ops"][-1], (shared, rows[0]["layer_ops"], rows[1]["layer_ops"])
        assert rows[0]["kernel_calls"] == rows[1]["kernel_calls"], "the kernel calls are the path's, not the arm's"


# --- the operator count's call paths ----------------------------------------------------------------


def test_call_paths_attribute_every_operator_and_change_no_count():
    model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2), seed=0).eval()
    batch = _states(0, (5,))
    plain, with_paths = count_forward(model, batch), count_forward(model, batch, paths=True)
    assert plain.signature() == with_paths.signature(), "asking for paths changes no count"
    assert not plain.paths and plain.functions() == []
    assert sum(with_paths.paths.values()) == with_paths.total()
    for scope in {scope for scope, _ in with_paths.ops}:
        functions = with_paths.functions(lambda s, scope=scope: s == scope)
        assert sum(n for _, n, _ in functions) == sum(n for (s, _), n in with_paths.ops.items() if s == scope)
        assert sum(c for _, _, c in functions) == sum(
            n for (s, name), n in with_paths.ops.items() if s == scope and with_paths.kinds[name] == "compute"
        )
    last = with_paths.functions(lambda scope: scope[:2] == ("layer", 1))
    assert [n for _, n, _ in last] == sorted((n for _, n, _ in last), reverse=True)
    names = {function for function, _, _ in last}
    assert "attention._kernel" in names and NO_FUNCTION not in names, "every operator of a layer is some network function's"
    fused = [path for (scope, path, name), _ in with_paths.paths.items() if name.startswith("aten::_scaled_dot_product_")]
    assert fused and all(path[-1] == "attention._kernel" for path in fused), "the kernel is called from one place"
    assert all(path[0] == "model.T5DNet.forward" for path in fused), "a path starts at the outermost network function"
    deep = {function for function, _, _ in with_paths.functions(depth=2)}
    assert "attention._attend_rows > attention._kernel" in deep and "attention._spatial_free_rows > attention._kernel" in deep
    with pytest.raises(ValueError, match="depth"):
        with_paths.functions(depth=0)

    a, b = torch.randn(3), torch.randn(3)
    _, outside = count_ops(lambda: a + b, paths=True)
    assert outside.functions() == [(NO_FUNCTION, 1, 1)], "an operator no network function issued"
    assert call_path(None) == ()


def test_measure_and_the_command_line_list_the_functions(capsys):
    row = measure("n2-default-sdpa", 3, dtype="fp32", layers=2, sites=5, depth=2)
    plain = measure("n2-default-sdpa", 3, dtype="fp32", layers=2)
    assert set(row) - set(plain) == {"sites"} and {k: v for k, v in row.items() if k != "sites"} == plain
    assert set(row["sites"]) == {"first_layer", "last_layer", "context"}
    for listed in row["sites"].values():
        assert len(listed) == 5 and all(isinstance(f, str) and n >= c >= 0 for f, n, c in listed)
        assert [n for _, n, _ in listed] == sorted((n for _, n, _ in listed), reverse=True)
    capsys.readouterr()
    assert main(["--config", "b15-layer-shared-sdpa", "--boards", "3", "--layers", "2", "--dtype", "fp32", "--sites", "4"]) == 0
    text = capsys.readouterr().out
    assert "operators by the network function that issued them, the first layer" in text and "the last layer" in text and "context" in text
    assert "attention." in text


def test_bench_names_reach_b15_s_arms_and_the_padded_widths():
    assert NetConfig(**CONFIGS["b15-layer-shared"]) == NetConfig(disp_layer_shared=True)
    assert NetConfig(**CONFIGS["b15-layer-shared-sdpa"]) == NetConfig(disp_layer_shared=True, attn_impl="sdpa")
    assert NetConfig(**CONFIGS["b15-no-far"]) == NetConfig(disp_far_axis=False)
    assert NetConfig(**CONFIGS["g11-pad8"]) == NetConfig(attn_pad_multiple=8)
    assert NetConfig(**CONFIGS["g11-pad8-sdpa"]) == NetConfig(attn_pad_multiple=8, attn_impl="sdpa")
    for name in ("b15-layer-shared", "b15-layer-shared-sdpa", "b15-no-far", "g11-pad8", "g11-pad8-sdpa"):
        NetConfig(**CONFIGS[name]).validate()
