"""B14: the block-sparse and chunked attention paths equal the dense reference.

The reference (``attention_dense``) materialises every mask and bias at (B, H, L, L) and
computes explicit scores; the chunked path runs the fused kernel over query blocks of
whole boards, building each block's masks and biases from the sparse structure, and the
block-sparse path also hands each masked head only the key boards its block can reach.
They must agree - outputs and gradients, to the parameters of B11, B12 and B15 included -
for several seeds, N in {1, 5, 17, 40} and every combination of stage N2's switches, with
B15's displacement bias (the v1 default) on and, for the paths a structure without a bias
takes, off (ablation 2's no-table arm); they
must never build a row set larger than one block, in the attention call or in the whole
network; the block-sparse selection must be lossless and actually skip boards; and under
autograd neither may keep an O(L^2) tensor for the backward pass.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest
import torch

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net import attention as attention_module
from t5d.net.attention import attention_block_sparse, attention_chunked, attention_dense, attention_explicit, attention_sdpa
from t5d.net.state import TOKEN_PAD
from t5d.net.structure import AttnStructure
from tests.net.test_structure import SCRAMBLED_ROLES

pytestmark = pytest.mark.grad

#: Every combination of stage N2's switches: B9's head count (0 / 2 / 4) or B10's routing
#: table, crossed with B11's typed edge biases and B12's board-pair smolgen.
N2_COMBOS = [
    dict(move_graph_heads=heads, routing_table=routing, typed_edge_bias=typed, board_pair_smolgen=smolgen)
    for (heads, routing), typed, smolgen in itertools.product(((0, False), (2, False), (4, False), (2, True)), (False, True), (False, True))
]
MG2, ROUTING, ROUTING_ALL = N2_COMBOS[4], N2_COMBOS[12], N2_COMBOS[-1]
#: The combinations without a learned bias under B15's no-table arm (ablation 2): a structure
#: with masks and no bias - or none at all - takes the by-role path whose dense heads run in
#: one call over every row.
NO_TABLE_COMBOS = [c | dict(displacement_bias=False) for c in N2_COMBOS if not (c["typed_edge_bias"] or c["board_pair_smolgen"])]
#: A routing table whose roles are not contiguous in head order, with and without biases.
SCRAMBLED = dict(routing_table=True, rope_head_roles=SCRAMBLED_ROLES)
SCRAMBLED_BIASED = SCRAMBLED | dict(typed_edge_bias=True, board_pair_smolgen=True, smolgen_form="board_bilinear")


def _with(attend, **overrides):
    """``attend`` with some of the context's fields overridden."""

    def run(q, k, v, ctx):
        return attend(q, k, v, replace(ctx, **overrides))

    run.__name__ = f"{attend.__name__}{overrides}"
    return run


#: The B14 paths: block-sparse gathering every block's selection, block-sparse with the
#: dense fallback for wide blocks in play, and the dense-mask arm (chunked).
FAST = {
    "block_sparse": _with(attention_block_sparse, sparse_max_fraction=1.0),
    "block_sparse_mixed": _with(attention_block_sparse, sparse_max_fraction=0.3),
    "chunked": attention_chunked,
}


def _combo_id(flags: dict) -> str:
    head = "routing" if flags["routing_table"] else f"mg{flags['move_graph_heads']}"
    extras = f"{'-typed' if flags['typed_edge_bias'] else ''}{'-smolgen' if flags['board_pair_smolgen'] else ''}"
    return f"{head}{extras}{'-no-table' if flags.get('displacement_bias') is False else ''}"


def _batch(n: int, seed: int):
    """A position of n boards beside a smaller one, so that padding is exercised."""
    return collate([synthetic_state(1000 * seed + n, n), synthetic_state(1000 * seed + n + 1, max(1, n // 3))])


def _learned(model) -> list[tuple[str, torch.Tensor]]:
    mods = {"typed": model.typed_edges, "smolgen": model.smolgen, "b15": model.displacement}
    return [(f"{k}.{n}", p) for k, m in mods.items() if m is not None for n, p in m.named_parameters()]


def _run(attend, model, batch, qkv, h, w, block_rows=None, recompute=None):
    """Output and gradients (q, k, v, every structure parameter) of one implementation,
    with a fresh context so that no graph is shared between implementations."""
    model.zero_grad(set_to_none=True)
    ctx = model.make_context(batch).for_layer(h, 0)
    if block_rows is not None:
        ctx = replace(ctx, block_rows=block_rows)
    if recompute is not None:
        ctx = replace(ctx, recompute=recompute)
    leaves = [t.clone().requires_grad_(True) for t in qkv]
    out = attend(*leaves, ctx)
    (out * w).sum().backward()
    grads = [t.grad for t in leaves] + [p.grad for _, p in _learned(model)]
    return out.detach(), grads


def _compare(batch, ref, other, atol_out=1e-5, atol_grad=1e-4, grad_rel=1e-5, what=""):
    """Outputs on valid rows within ``atol_out``; each gradient within ``atol_grad`` plus
    ``grad_rel`` times its largest entry. FP32 roundoff grows with the gradient's size: a
    typed-edge scalar's gradient sums thousands of pairs (|g| up to ~75 at N = 40 under
    masks), measured 0.6-7e-6 relative between the paths; with logits of order 100 the
    kernel's own arithmetic reaches 8e-5 relative."""
    valid = batch.valid[:, None, :, None]
    out_r, grads_r = ref
    out_o, grads_o = other
    assert torch.allclose(out_r * valid, out_o * valid, atol=atol_out), (what, float(((out_r - out_o) * valid).abs().max()))
    for i, (ga, gb) in enumerate(zip(grads_r, grads_o, strict=True)):
        assert (ga is None) == (gb is None), (what, i)
        if ga is not None:
            assert torch.isfinite(gb).all(), (what, i)
            tol = atol_grad + grad_rel * float(ga.abs().max())
            assert torch.allclose(ga, gb, atol=tol, rtol=0.0), (what, i, float((ga - gb).abs().max()))


def _case(flags: dict, n: int, seed: int, extra=None):
    cfg = NetConfig(trunk_dtype="fp32", n_layers=1, **flags, **(extra or {}))
    model = build_model(cfg, seed=seed)
    batch = _batch(n, seed)
    gen = torch.Generator().manual_seed(seed)
    shape = (batch.batch_size, cfg.n_heads, batch.n_tokens, cfg.d_head)
    qkv = [torch.randn(*shape, generator=gen) for _ in range(3)]
    h = torch.randn(batch.batch_size, batch.n_tokens, cfg.d_model, generator=gen)
    w = torch.randn(*shape, generator=gen) * batch.valid[:, None, :, None]
    return model, batch, qkv, h, w


@pytest.mark.parametrize("n", [1, 5, 17, 40])
@pytest.mark.parametrize("flags", N2_COMBOS + NO_TABLE_COMBOS, ids=[_combo_id(f) for f in N2_COMBOS + NO_TABLE_COMBOS])
def test_chunked_equals_dense(flags, n):
    """Outputs and gradients, two seeds, both paths, blocks of one and of three boards (the
    mixed sparse/dense variant on 3-board blocks from 17 boards, where both kinds occur)."""
    for seed in (0, 1):
        model, batch, qkv, h, w = _case(flags, n, seed)
        ref = _run(attention_dense, model, batch, qkv, h, w)
        for (name, attend), boards in itertools.product(FAST.items(), (1, 3)):
            if name == "block_sparse_mixed" and (n < 17 or boards == 1):
                continue
            other = _run(attend, model, batch, qkv, h, w, block_rows=66 * boards)
            _compare(batch, ref, other, what=(seed, name, boards))


@pytest.mark.parametrize("n", [5, 17])
@pytest.mark.parametrize("flags", [SCRAMBLED, SCRAMBLED_BIASED], ids=["scrambled", "scrambled-biased"])
def test_a_scrambled_routing_table_equals_dense(flags, n):
    """Roles that are not contiguous in head order: the by-role paths gather a role's
    heads from across the table and must put every head's output back in its place."""
    model, batch, qkv, h, w = _case(flags, n, seed=8, extra=dict(masked_head_registers="self"))
    ref = _run(attention_dense, model, batch, qkv, h, w)
    for name, attend in FAST.items():
        _compare(batch, ref, _run(attend, model, batch, qkv, h, w, block_rows=66), what=name)


#: The pair-rule modes (global registers position-free or anchored, per-board registers
#: position-free) and B5, which the fast paths also handle block by block - with B15's
#: displacement bias (the v1 default) and, for two of them, without it (no structure at all
#: under the dense flags).
PAIR_MODES = {
    "anchored": dict(global_register_pairs="anchored"),
    "regs_free": dict(board_register_spatial="position_free"),
    "regs_free_anchored_b5": dict(global_register_pairs="anchored", board_register_spatial="position_free", rope_per_head=True),
    "b5_regs_self": dict(rope_per_head=True, masked_head_registers="self", move_graph_direction="reverse"),
    "anchored_regs_self": dict(global_register_pairs="anchored", masked_head_registers="self", same_board_registers="registers"),
    "no_table_regs_free": dict(board_register_spatial="position_free", displacement_bias=False),
    "no_table_anchored": dict(global_register_pairs="anchored", displacement_bias=False),
}


@pytest.mark.parametrize("n", [5, 17])
@pytest.mark.parametrize("mode", list(PAIR_MODES))
@pytest.mark.parametrize("flags", [N2_COMBOS[0], N2_COMBOS[3], ROUTING_ALL], ids=["dense", "mg2-typed-smolgen", "routing-all"])
def test_chunked_equals_dense_in_every_pair_mode(flags, mode, n):
    model, batch, qkv, h, w = _case(flags, n, seed=2, extra=PAIR_MODES[mode])
    ref = _run(attention_dense, model, batch, qkv, h, w)
    runs = [(attend, rows) for attend in FAST.values() for rows in (66, 2 * 66 + 5)]
    for attend, rows in [*runs, (attention_sdpa, None), (attention_explicit, None)]:
        _compare(batch, ref, _run(attend, model, batch, qkv, h, w, block_rows=rows), what=(attend.__name__, rows))


@pytest.mark.parametrize("flags", [MG2, ROUTING_ALL], ids=["mg2", "routing-all"])
def test_chunked_is_stable_at_large_logits(flags):
    """Logits of order +-100: finite, and within FP32 roundoff of the reference - measured
    3e-5 relative on outputs and up to 8e-5 relative on gradients, the same for the
    unchunked fused path (the kernel's arithmetic, not the blocks)."""
    model, batch, qkv, h, w = _case(flags, 5, seed=3)
    qkv = [qkv[0] * 12.0, qkv[1] * 12.0, qkv[2]]
    ref = _run(attention_dense, model, batch, qkv, h, w)
    for name, attend in FAST.items():
        other = _run(attend, model, batch, qkv, h, w, block_rows=66)
        _compare(batch, ref, other, atol_out=5e-4, atol_grad=0.0, grad_rel=4e-4, what=name)


@pytest.mark.parametrize("flags", [MG2, ROUTING], ids=["mg2", "routing"])
def test_recompute_changes_nothing(flags):
    """Recomputing each block in the backward pass (``attn_recompute``) is exact: the same
    outputs and gradients as keeping every block, on both paths, with every bias on."""
    model, batch, qkv, h, w = _case(flags | dict(typed_edge_bias=True, board_pair_smolgen=True), 17, seed=9)
    for name, attend in FAST.items():
        kept = _run(attend, model, batch, qkv, h, w, block_rows=66, recompute=False)
        again = _run(attend, model, batch, qkv, h, w, block_rows=66, recompute=True)
        _compare(batch, kept, again, atol_out=1e-6, atol_grad=1e-6, what=name)


@pytest.mark.parametrize(
    "flags",
    [MG2, N2_COMBOS[7], ROUTING_ALL, dict(rope_per_head=True, typed_edge_bias=True), MG2 | dict(displacement_bias=False)],
    ids=["mg2", "mg2-typed-smolgen", "routing-all", "b5-typed", "mg2-no-table"],
)
def test_model_chunked_equals_dense(flags):
    """The whole network (two layers): every head's output and every parameter's gradient."""
    batch = _batch(5, seed=4)
    results = []
    for impl in ("dense", "chunked", "block_sparse"):
        model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, attn_block_boards=1, **flags), seed=4)
        out = model(batch)
        gen = torch.Generator().manual_seed(4)
        loss = (out.value_logits * torch.randn(out.value_logits.shape, generator=gen)).sum()
        loss = loss + (out.policy_logits[out.move_valid] * torch.randn(int(out.move_valid.sum()), generator=gen)).sum()
        loss = loss + (out.board_logits * torch.randn(out.board_logits.shape, generator=gen)).sum()
        loss.backward()
        results.append((out, {n: p.grad for n, p in model.named_parameters()}))
    (a, ga), *others = results
    for b, gb in others:
        assert torch.allclose(a.value_logits, b.value_logits, atol=1e-4)
        assert torch.allclose(a.policy_logits[a.move_valid], b.policy_logits[b.move_valid], atol=1e-4)
        assert torch.allclose(a.board_logits, b.board_logits, atol=1e-4)
        for name in ga:
            assert (ga[name] is None) == (gb[name] is None), name
            if name.startswith(("trunk.", "embed.", "typed_edges.", "smolgen.", "displacement.")):
                assert ga[name] is not None, f"{name}: the loss reaches every trunk and structure parameter"
            if ga[name] is not None:
                assert torch.allclose(ga[name], gb[name], atol=2e-4, rtol=1e-3), (name, float((ga[name] - gb[name]).abs().max()))


# --- the memory claims, checked structurally -------------------------------------------------


class _Spy:
    """Records every row span the structure is asked for and every kernel call's
    (query rows, keys) and mask shape."""

    def __init__(self, monkeypatch) -> None:
        self.spans: list[int] = []
        self.kernel: list[tuple[int, int, tuple | None]] = []
        for name in ("rows", "role_rows", "bias_rows", "key_selection"):
            original = getattr(AttnStructure, name)

            def spy(self_, *args, _original=original, _name=name, **kwargs):
                first = 1 if _name in ("role_rows", "key_selection") else 0  # after the role
                self.spans.append(args[first + 1] - args[first])
                return _original(self_, *args, **kwargs)

            monkeypatch.setattr(AttnStructure, name, spy)
        original_at = AttnStructure.bias_rows_at

        def spy_at(self_, rows, *args, **kwargs):
            self.spans.append(rows.shape[1])
            return original_at(self_, rows, *args, **kwargs)

        monkeypatch.setattr(AttnStructure, "bias_rows_at", spy_at)
        kernel = attention_module.F.scaled_dot_product_attention

        def spy_kernel(q, k, v, attn_mask=None, scale=None):
            self.kernel.append((q.shape[-2], k.shape[-2], None if attn_mask is None else tuple(attn_mask.shape)))
            return kernel(q, k, v, attn_mask=attn_mask, scale=scale)

        monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", spy_kernel)

    def clear(self) -> None:
        self.spans.clear()
        self.kernel.clear()

    def per_row_masks(self) -> list[tuple]:
        """The masks that differ between query rows (a broadcast key mask has one row)."""
        return [m for _, _, m in self.kernel if m is not None and m[-2] > 1]


@pytest.mark.parametrize("name", list(FAST))
def test_fast_paths_never_build_more_than_one_block(monkeypatch, name):
    """The memory claim, checked structurally: every row set either fast path asks the
    structure for, and every per-row mask it hands the kernel, spans at most one block of
    query rows - never (L, L). The dense reference, by contrast, asks for all L rows."""
    flags = dict(typed_edge_bias=True, board_pair_smolgen=True, board_register_spatial="position_free")
    model, batch, qkv, h, _ = _case(N2_COMBOS[0] | flags, 17, seed=5)
    block = 2 * 66
    spy = _Spy(monkeypatch)
    with torch.no_grad():
        ctx = replace(model.make_context(batch).for_layer(h, 0), block_rows=block)
        FAST[name](*qkv, ctx)
        assert spy.spans and max(spy.spans) <= block
        assert spy.per_row_masks() and all(m[-2] <= block for m in spy.per_row_masks())
        spy.clear()
        attention_dense(*qkv, ctx)
        assert max(spy.spans) == batch.n_tokens


@pytest.mark.parametrize("impl", ["block_sparse", "chunked"])
@pytest.mark.parametrize(
    "flags",
    [dict(), dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)],
    ids=["default", "routing-all-biases"],
)
def test_the_network_attends_one_block_at_a_time(monkeypatch, flags, impl):
    """The same claim on the network's own path, as ``make_context`` sets it up: every row
    span and every per-row kernel mask of a whole forward pass stays within one block of
    ``attn_block_boards`` boards - the default configuration included."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, **flags)
    model = build_model(cfg, seed=6).eval()
    batch = collate([synthetic_state(60, 12)])
    block = cfg.attn_block_boards * 66
    assert model.make_context(batch).block_rows == block
    spy = _Spy(monkeypatch)
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out.value_logits).all()
    assert spy.spans and max(spy.spans) <= block, max(spy.spans)
    masks = spy.per_row_masks()
    assert masks and max(m[-2] for m in masks) <= block


def test_block_sparse_reads_only_the_boards_its_block_reaches(monkeypatch):
    """The block-sparse selection is lossless (every key a query may read is selected),
    made of whole boards, and sparse: at 40 boards a 4-board block of the move-graph heads
    reads well under all of the key boards, so the kernel scores fewer keys than the
    chunked path does."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=1)
    model = build_model(cfg, seed=7)
    batch = collate([synthetic_state(70, 40), synthetic_state(71, 9)])
    structure = model.make_context(batch).structure
    length, kl, block = batch.n_tokens, batch.body_len, 4 * 66
    board = structure.board_idx
    fractions = []
    for start in range(0, kl, block):
        stop = min(kl, start + block)
        rel = structure.role_rows("move_graph", start, stop) & batch.valid[:, None, :]
        idx, valid = structure.key_selection("move_graph", start, stop, rel, kl)
        for b in range(batch.batch_size):
            chosen = set(idx[b][valid[b]].tolist())
            assert idx[b][valid[b]].tolist() == sorted(chosen), "each key once, in token order"
            free = structure.free_rows("move_graph")[b, start:stop]
            selecting = (batch.kind[b, start:stop] != TOKEN_PAD) & ~free
            needed = set(torch.nonzero(rel[b][selecting].any(dim=0)[:kl]).flatten().tolist())
            assert needed <= chosen, "lossless"
            boards = {int(board[b, t]) for t in chosen}
            whole = {t for t in range(kl) if int(board[b, t]) in boards and bool(batch.valid[b, t])}
            assert chosen == whole, "whole boards"
            if start >= int(batch.n_body[b]):
                assert not chosen, "a block of padding selects nothing"
            if b == 0:
                fractions.append(len(boards) / 40)
    assert max(fractions) <= 1.0 and sum(fractions) / len(fractions) < 0.8, fractions
    # The kernel sees the selection: fewer keys than every key on the move-graph heads.
    qkv = [torch.randn(batch.batch_size, 8, length, 32, generator=torch.Generator().manual_seed(s)) for s in range(3)]
    spy = _Spy(monkeypatch)
    with torch.no_grad():
        attention_block_sparse(*qkv, replace(model.make_context(batch).for_layer(None, 0), block_rows=block, sparse_max_fraction=1.0))
    # The move-graph heads' calls (B9's two heads; the six dense heads carry B15's per-row
    # bias and read every key).
    key_counts = [k for _, k, m in spy.kernel if m is not None and m[-2] > 1 and m[1] == 2]
    assert key_counts and min(key_counts) < kl and sum(key_counts) / len(key_counts) < 0.8 * kl


def _kept(attend, flags: dict, n: int, recompute: bool) -> tuple[int, list[tuple[int, ...]]]:
    """What autograd keeps for the backward pass of one attention call over one position of
    ``n`` boards - the tensors its forward saves outside any recomputed block: their bytes
    (each storage once) and the shapes of those that hold many query rows over every key."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=1, **flags)
    model = build_model(cfg, seed=10)
    batch = collate([synthetic_state(90 + n, n)])
    gen = torch.Generator().manual_seed(10)
    shape = (1, cfg.n_heads, batch.n_tokens, cfg.d_head)
    leaves = [torch.randn(*shape, generator=gen).requires_grad_(True) for _ in range(3)]
    h = torch.randn(1, batch.n_tokens, cfg.d_model, generator=gen)
    ctx = replace(model.make_context(batch).for_layer(h, 0), block_rows=4 * 66, recompute=recompute)
    saved: list[torch.Tensor] = []

    def pack(t: torch.Tensor):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = attend(*leaves, ctx)
    out.sum().backward()
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in saved}
    return sum(storages.values()), _attention_shaped(saved)


def _attention_shaped(saved: list[torch.Tensor]) -> list[tuple[int, ...]]:
    """Shapes of the saved (B, H, rows, keys) tensors that hold a block's mask, bias or
    probabilities: a board's rows (64) or more over 64 keys or more. (Q, K and V have
    d_head columns, the global registers' rows 16 rows, the FFN's tensors 3 dims.)"""
    return [tuple(t.shape) for t in saved if t.dim() == 4 and t.shape[-2] >= 64 and t.shape[-1] >= 64]


@pytest.mark.parametrize("name", ["block_sparse", "chunked"])
@pytest.mark.parametrize("flags", [MG2, ROUTING_ALL], ids=["mg2", "routing-all-biases"])
def test_autograd_keeps_no_quadratic_tensor(flags, name):
    """Under autograd each block is recomputed in the backward pass, so the forward keeps
    no attention-shaped tensor and what it keeps grows linearly in L: from 10 to 40 boards
    (L x 3.9) by less than 4.5x (measured 3.95-4.07). Without recomputation the kernel
    keeps every block's mask (and, once a learned bias needs gradient, its probabilities):
    the same measurement then finds them, and keeps 1.8-18x more at 40 boards (growing
    5.5-12x from 10) - the test sees the difference."""
    attend = FAST[name]
    small, small_rows = _kept(attend, flags, 10, recompute=True)
    big, big_rows = _kept(attend, flags, 40, recompute=True)
    assert not small_rows and not big_rows, big_rows
    assert big / small < 4.5, (small, big)
    kept_big, kept_rows = _kept(attend, flags, 40, recompute=False)
    assert kept_rows and kept_big > 1.5 * big, (big, kept_big)


@pytest.mark.parametrize("flags", [dict(), dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)], ids=["default", "all"])
def test_the_network_keeps_no_quadratic_tensor_for_backward(flags):
    """The autograd claim on the network's own path, as ``make_context`` sets it up
    (``attn_recompute`` on by default): a training forward of the whole network keeps no
    attention-shaped tensor for the backward pass."""
    cfg = NetConfig(trunk_dtype="fp32", n_layers=2, **flags)
    model = build_model(cfg, seed=12)
    batch = collate([synthetic_state(95, 12)])
    assert model.make_context(batch).recompute
    saved: list[torch.Tensor] = []

    def pack(t: torch.Tensor):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(batch)
    out.value_logits.sum().backward()
    assert saved and not _attention_shaped(saved), _attention_shaped(saved)


@pytest.mark.parametrize(
    ("flags", "extra"),
    [
        (ROUTING_ALL, {}),
        (MG2, dict(board_register_spatial="position_free")),
        (ROUTING_ALL, dict(board_register_spatial="position_free", global_register_pairs="anchored")),
        (SCRAMBLED, dict(global_register_pairs="anchored")),
        (MG2, dict(displacement_bias=False)),
    ],
    ids=["all", "mg2-regs-free", "routing-all-regs-free-anchored", "scrambled-anchored", "mg2-no-table"],
)
def test_block_size_does_not_change_the_function(flags, extra):
    """Board-aligned or not, one row or the whole body: the same outputs - blocks of 65
    rows split a board's two registers, so a block holds a single spatial-free query."""
    model, batch, qkv, h, _ = _case(flags, 5, seed=6, extra=extra)
    with torch.no_grad():
        ctx = model.make_context(batch).for_layer(h, 0)
        ref = attention_dense(*qkv, ctx)
        for (name, attend), rows in itertools.product(FAST.items(), (1, 17, 65, 66, 200, 10_000)):
            out = attend(*qkv, replace(ctx, block_rows=rows))
            valid = batch.valid[:, None, :, None]
            assert torch.allclose(out * valid, ref * valid, atol=1e-5), (name, rows)


#: Structures with and without biases: a caller's mask and bias must combine with each -
#: those without a bias (B15's no-table arm) take the by-role path, where the chunked arm's
#: dense heads run in one call over every row.
CALLER_CASES = {
    "routing-all": (ROUTING_ALL, {}),
    "mg2": (MG2, {}),
    "routing": (ROUTING, {}),
    "scrambled": (SCRAMBLED, {}),
    "mg2-no-table": (MG2, dict(displacement_bias=False)),
    "scrambled-no-table": (SCRAMBLED, dict(displacement_bias=False)),
}


@pytest.mark.parametrize("parts", ["mask", "bias", "both"])
@pytest.mark.parametrize("case", list(CALLER_CASES))
@pytest.mark.parametrize("impl", [attention_block_sparse, attention_chunked, attention_sdpa, attention_explicit])
def test_a_callers_mask_and_bias_combine_with_the_structure(impl, case, parts):
    """N1's API stays: a caller's dense mask and bias apply on top of the structure's -
    checked against the combination done by hand (the structure's dense mask and bias,
    the caller's, and padding, folded into one structure-free context)."""
    flags, extra = CALLER_CASES[case]
    model, batch, qkv, h, _ = _case(flags, 5, seed=7, extra=extra)
    gen = torch.Generator().manual_seed(7)
    shape = (batch.batch_size, 8, batch.n_tokens, batch.n_tokens)
    mask = (torch.rand(*shape, generator=gen) < 0.8) | torch.eye(batch.n_tokens, dtype=torch.bool)
    bias = torch.randn(*shape, generator=gen)
    mask = mask if parts in ("mask", "both") else None
    bias = bias if parts in ("bias", "both") else None
    with torch.no_grad():
        ctx = replace(model.make_context(batch).for_layer(h, 0), mask=mask, bias=bias, block_rows=66)
        structure = ctx.structure
        s_mask, s_bias = structure.dense_mask(), structure.dense_bias()
        full_mask = s_mask if mask is None else mask & s_mask
        full_bias = bias if s_bias is None else (s_bias if bias is None else bias + s_bias)
        ref = attention_explicit(*qkv, replace(ctx, mask=full_mask, bias=full_bias, structure=None))
        out = impl(*qkv, ctx)
    valid = batch.valid[:, None, :, None]
    assert torch.allclose(out * valid, ref * valid, atol=1e-5)


@pytest.mark.parametrize("flags", [dict(), dict(routing_table=True), dict(displacement_bias=False)], ids=["default", "routing", "no-table"])
def test_the_networks_attn_mask_and_attn_bias_reach_every_path(flags):
    """``model(batch, attn_mask=..., attn_bias=...)`` on the default network (B15's bias in the
    structure), under the routing table, and on B15's no-table arm (no bias in the structure:
    the by-role path): the same outputs on the block-sparse and chunked paths as on the dense
    reference."""
    batch = _batch(4, seed=11)
    gen = torch.Generator().manual_seed(11)
    shape = (batch.batch_size, 8, batch.n_tokens, batch.n_tokens)
    mask = (torch.rand(*shape, generator=gen) < 0.7) | torch.eye(batch.n_tokens, dtype=torch.bool)
    bias = torch.randn(*shape, generator=gen)
    outs = {}
    with torch.no_grad():
        for impl in ("dense", "chunked", "block_sparse"):
            model = build_model(NetConfig(trunk_dtype="fp32", n_layers=2, attn_impl=impl, attn_block_boards=1, **flags), seed=11)
            outs[impl] = model(batch, attn_mask=mask, attn_bias=bias)
        plain = model(batch)
    for impl in ("chunked", "block_sparse"):
        assert torch.allclose(outs[impl].value_logits, outs["dense"].value_logits, atol=1e-4), impl
        assert torch.allclose(outs[impl].board_logits, outs["dense"].board_logits, atol=1e-4), impl
    assert not torch.allclose(plain.value_logits, outs["block_sparse"].value_logits, atol=1e-3), "the mask and bias are not inert"
