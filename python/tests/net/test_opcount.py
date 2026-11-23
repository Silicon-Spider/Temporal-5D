"""Register G11's operator count (``t5d.net.opcount``): what it counts, that counting changes
nothing, and the constancy section 10.2's requirement 3 asks for - pinned where it holds.

Spec section 10.2, requirement 3 (2026-11-15 review): the number of launches in a forward is
"independent of N and of the batch ... It is counted, not assumed". The requirement binds the
compiled path; the eager paths are counted against it. Of the network as built:

* the one-block paths ("n1-dense" and "n2-default-sdpa" of ``bench.CONFIGS``:
  ``attn_impl="sdpa"``) meet the constancy - the same operators in the same places at every
  board count, for every position of one size and for every batch of 2 or more. Pinned here on
  the whole signature.
* "n2-default-chunked" counts the same for two positions of one size and across batch sizes,
  and grows with the board count (its kernel calls are per query block).
* **The v1 default, "n2-default" (``attn_impl="block_sparse"``), does not meet it**: its count
  grows with the board count, and two positions of one size count differently
  (``AttnStructure.block_plan`` branches on the position). No test here claims otherwise.
  ``test_the_block_paths_grow_with_the_board_count_and_the_tool_says_so`` pins that the tool
  reports it, and is to be replaced when a path that meets the requirement becomes the default.

What holds on every configuration, the default included: the operators outside the layers do
not depend on the board count or on the position (a Python loop over boards in
``T5DNet.make_context`` would break it); every layer makes the same number of kernel calls; the
layers after the first dispatch the same number of operators.

No count of the network is pinned: a count belongs to the PyTorch build and to the network as it
stands, and sibling items of the 2026-11-15 review move it. Networks: 3 layers, FP32 (BF16 once,
for the precision as shipped), random weights at ``build_model(seed=0)``'s scale, built once per
configuration; positions ``synthetic_state(seed, N)`` with N in {2, 5, 12} and seeds 0-2.
"""

from __future__ import annotations

import functools
import json
import math

import pytest
import torch
import torch.nn.functional as F

from t5d.net import NetConfig, build_model, collate, synthetic_state
from t5d.net.bench import CONFIGS
from t5d.net.opcount import OUTSIDE, PARTS, SYNC_OPS, VERDICTS, count_forward, count_ops, kernel_kind, main, measure, requirement

#: One query block whatever the board count (``attn_impl="sdpa"``).
ONE_BLOCK = ("n1-dense", "n2-default-sdpa")
#: Query blocks of ``attn_block_boards`` boards: the dense-mask arm and the v1 default.
BLOCK = ("n2-default-chunked", "n2-default")
EVERY = ONE_BLOCK + BLOCK
BOARDS = (2, 5, 12)
SEEDS = (0, 1, 2)
LAYERS = 3
#: The keys ``measure`` documents.
ROW_KEYS = {
    *("config", "dtype", "layers", "boards", "batch", "seed", "tokens", "torch"),
    *("ops", "compute", "free", "sync", "layer_ops", "layer_compute", "layer_parts"),
    *("kernel_calls", "kernel_fused", "kernel_math", "kernel_ops", "outside"),
}


@functools.cache
def _model(config: str, dtype: str = "fp32"):
    return build_model(NetConfig(trunk_dtype=dtype, **(CONFIGS[config] | {"n_layers": LAYERS})), seed=0).eval()


def _batch(*positions: tuple[int, int]):
    """A batch of ``(seed, boards)`` positions."""
    return collate([synthetic_state(seed, boards) for seed, boards in positions])


@functools.cache
def _count(config: str, *positions: tuple[int, int]):
    """The count of one forward of ``config``'s network over ``positions`` (kept: a count is
    deterministic, which ``test_counting_leaves_the_forward_and_the_model_as_they_were`` holds)."""
    return count_forward(_model(config), _batch(*positions))


def _hooks(model) -> int:
    """Forward hooks of every kind left on any module of ``model``."""
    kinds = ("_forward_hooks", "_forward_pre_hooks", "_forward_hooks_always_called", "_forward_hooks_with_kwargs")
    return sum(len(getattr(module, kind)) for module in model.modules() for kind in kinds)


def _rows(config: str, boards=BOARDS, batches=(1,)) -> list[dict]:
    return [measure(config, n, dtype="fp32", batch=b, layers=LAYERS, model=_model(config)) for n in boards for b in batches]


def test_an_operator_is_counted_once_and_classed_by_its_schema():
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    out, count = count_ops(lambda: a + b)
    assert torch.equal(out, a + b)
    assert dict(count.ops) == {(("all",), "aten::add.Tensor"): 1}, "one addition is one operator, in the scope of a count without a model"
    assert count.kinds == {"aten::add.Tensor": "compute"}
    assert (count.total(), count.total("compute"), count.total("free")) == (1, 1, 0)

    _, count = count_ops(lambda: (a + b).view(-1))
    assert (count.total("compute"), count.total("free")) == (1, 1), "a view aliases its input (its schema says so): free"

    w, bias = torch.randn(5, 4), torch.randn(5)
    out, count = count_ops(lambda: F.linear(a, w, bias))
    assert torch.equal(out, F.linear(a, w, bias))
    names = {name for _, name in count.ops}
    assert "aten::linear" not in names, "the count is taken after composite operators decompose"
    assert (count.total("compute"), count.total("free")) == (1, 1), f"a linear layer is one matrix product and one transpose view: {names}"


def test_what_is_never_dispatched_is_not_counted():
    a = torch.randn(3, 4)
    for what, fn in (
        ("an FP32 tensor's .float()", a.float),
        ("shape and rank reads", lambda: a.shape[0] + a.dim()),
        ("Python arithmetic", lambda: sum(i * i for i in range(10))),
    ):
        _, count = count_ops(fn)
        assert count.total() == 0, f"{what} reaches no backend operator"
        assert count.kernel_total() == 0 and count.sync() == 0


def test_a_host_synchronisation_is_counted():
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    _, count = count_ops(lambda: bool(a.sum() > 0))
    assert count.sync() == 1, "bool(tensor) reads one scalar back (aten::_local_scalar_dense)"
    _, count = count_ops(lambda: torch.nonzero(a > 0))
    assert count.sync() >= 1, "nonzero's result has a data-dependent size"
    assert all(count.kinds[name] == "compute" for _, name in count.ops if name in SYNC_OPS), "a synchronisation is no free operator"
    assert count.sync() <= count.total("compute")
    _, count = count_ops(lambda: a + b)
    assert count.sync() == 0
    _, count = count_ops(lambda: a[a > 0])
    assert count.sync() == 0 and count.total("compute") == 2, (
        "the figure is a lower bound: the nonzero inside boolean-mask indexing is issued by the index kernel's own C++ and never "
        "reaches the dispatcher (the comparison and the index are the two operators seen)"
    )


def test_a_kernel_call_is_counted_and_classed():
    q = torch.randn(1, 2, 5, 8)
    out, count = count_ops(lambda: F.scaled_dot_product_attention(q, q, q))
    assert torch.equal(out, F.scaled_dot_product_attention(q, q, q))
    assert count.kernel_total() == 1
    assert count.kernel_total("fused") + count.kernel_total("math") == 1
    inside = sum(count.kernel_ops.values())
    assert 1 <= inside <= count.total(), "the operators a kernel call dispatches are counted among the operators"

    _, count = count_ops(lambda: [F.scaled_dot_product_attention(q, q, q) for _ in range(3)])
    assert count.kernel_total() == 3
    _, count = count_ops(lambda: q @ q.transpose(-1, -2))
    assert count.kernel_total() == 0 and not count.kernel_ops, "a matmul is an operator, not an attention-kernel call"

    fused = "aten::_scaled_dot_product_flash_attention_for_cpu"
    assert kernel_kind([fused]) == "fused"
    assert kernel_kind(["aten::scalar_tensor", "aten::where.self", fused]) == "fused", "a mask conversion beside the fused operator"
    assert kernel_kind(["aten::bmm", "aten::_safe_softmax", "aten::bmm"]) == "math"
    assert kernel_kind([]) == "math"


def test_the_counts_partition_the_forward():
    """Every operator of a forward is in exactly one scope and of exactly one kind."""
    scopes = {(where,) for where in OUTSIDE} | {("layer", i, part) for i in range(LAYERS) for part in PARTS}
    for config in EVERY:
        count = _count(config, (0, 5))
        assert {scope for scope, _ in count.ops} <= scopes, config
        assert count.n_layers == LAYERS == len(count.layers()) == len(count.kernel_calls())
        assert count.total() == sum(count.layers()) + sum(count.outside().values()), config
        assert count.total() == count.total("compute") + count.total("free"), config
        assert count.layers() == [c + f for c, f in zip(count.layers("compute"), count.layers("free"), strict=True)]
        parts = count.layer_parts()
        assert set(parts) == set(PARTS)
        assert [sum(parts[part][i] for part in PARTS) for i in range(LAYERS)] == count.layers(), config
        assert all(parts["attention"][i] > 0 and parts["ffn"][i] > 0 and parts["norm"][i] > 0 for i in range(LAYERS)), config
        assert all(n > 0 for n in count.outside().values()), (config, count.outside())
        assert 0 < count.sync() <= count.total("compute")
        assert count.kernel_total() == count.kernel_total("fused") + count.kernel_total("math") == sum(count.kernel_calls()) > 0
        assert {scope[2] for scope, _ in count.kernel} == {"attention"}, "the kernel is called from the attention sublayer only"
        assert sum(count.kernel_ops.values()) >= count.kernel_total()
        top = count.top(3)
        assert len(top) == 3 and top[0][2] >= top[1][2] >= top[2][2] and all(count.kinds[name] == kind for name, kind, _ in top)


def test_counting_leaves_the_forward_and_the_model_as_they_were():
    """Bit for bit the uncounted forward's outputs (FP32, and BF16 as shipped), the same count
    twice, and no hook left on the model - also when the forward raises."""
    for dtype in ("fp32", "bf16"):
        model, batch = _model("n2-default", dtype), _batch((0, 5))
        plain = model(batch)
        counted, count = count_ops(lambda model=model, batch=batch: model(batch), model)
        for name, value in vars(plain).items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, getattr(counted, name)), f"{dtype}: {name} differs under counting"
        assert _hooks(model) == 0, "every hook is removed"
        assert count_forward(model, batch).signature() == count.signature(), "a count is deterministic"
        assert count.total() > 0 and count.n_layers == LAYERS

    model, batch = _model("n2-default"), _batch((0, 5))
    before, plain = _count("n2-default", (0, 5)).signature(), model(batch)

    def raises():
        model(batch)
        raise RuntimeError("after the forward")

    def kv_hook(layer, k, v):
        raise RuntimeError("inside a block's attention")

    with pytest.raises(RuntimeError, match="after the forward"):
        count_ops(raises, model)
    assert _hooks(model) == 0
    with pytest.raises(RuntimeError, match="inside a block's attention"):
        count_forward(model, batch, kv_hook=kv_hook)
    assert _hooks(model) == 0
    with pytest.raises(ValueError, match="unknown attention scope"):
        count_forward(model, batch, scope="no such scope")
    assert _hooks(model) == 0
    assert count_forward(model, batch).signature() == before, "a failed count leaves nothing behind for the next"
    assert torch.equal(model(batch).policy_logits, plain.policy_logits), "nor for the next plain forward"


def test_kernel_calls_equal_an_independent_recount(monkeypatch):
    """The network's kernel calls, recounted by wrapping the function it calls."""
    real = F.scaled_dot_product_attention
    calls: list[int] = []

    def wrapper(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    for config in ("n2-default", "n1-dense"):
        model, batch = _model(config), _batch((0, 5))
        counted = count_forward(model, batch).kernel_total()
        with monkeypatch.context() as patch:
            patch.setattr(torch.nn.functional, "scaled_dot_product_attention", wrapper)
            calls.clear()
            model(batch)
            recount = len(calls)
            behind_the_wrapper = count_forward(model, batch).kernel_total()
        assert counted == recount == behind_the_wrapper > 0, (config, counted, recount, behind_the_wrapper)


def test_outside_the_layers_the_count_does_not_grow_with_the_board_count():
    """Embedding, context, final norm and heads dispatch the same operators at every board count
    and for every position - on every configuration, the v1 default included."""
    for config in EVERY:
        seen = {
            (tuple(count.outside().items()), tuple(count.outside("compute").items()))
            for count in (_count(config, (seed, boards)) for boards in BOARDS for seed in SEEDS)
        }
        assert len(seen) == 1, f"{config}: operators outside the layers vary with the board count or the position: {sorted(seen)}"


def test_one_block_paths_do_not_grow_with_the_board_count():
    """Requirement 3's independence of N, where it holds as built: every operator of every scope,
    and the kernel calls, are the same at 2, 5 and 12 boards. No number is pinned."""
    for config in ONE_BLOCK:
        signatures = {_count(config, (0, boards)).signature() for boards in BOARDS}
        assert len(signatures) == 1, f"{config}: the count depends on the board count"


def test_kernel_calls_per_layer_are_constant_across_layers():
    """Every layer makes the same number of kernel calls and, after the first, dispatches the same
    number of operators; layer 0 may carry what is built on first use (the block-sparse plans)."""
    for config in EVERY:
        for boards in BOARDS:
            for seed in SEEDS:
                count = _count(config, (seed, boards))
                calls, layers = count.kernel_calls(), count.layers()
                assert len(set(calls)) == 1 and calls[0] > 0, (config, boards, seed, calls)
                assert len(set(layers[1:])) == 1, (config, boards, seed, layers)
                assert layers[0] >= layers[1], (config, boards, seed, layers)


def test_two_positions_of_one_size_count_the_same():
    """The one-block paths and the chunked arm. The block-sparse default is left out because it
    does not hold there: its key selection depends on the position (the module docstring)."""
    for config in (*ONE_BLOCK, "n2-default-chunked"):
        for boards in (5, 12):
            signatures = {_count(config, (seed, boards)).signature() for seed in SEEDS}
            assert len(signatures) == 1, f"{config}: two positions of {boards} boards count differently"


def test_the_count_does_not_grow_with_the_batch():
    """Requirement 3's independence of the batch, from two positions up (a single position takes
    fast paths of its own and is not compared): same-size batches of 2 and 4, and padded batches
    of 2 and 4. A padded batch is compared with a padded one - padding rows take a branch of
    their own, a constant offset - and, for the chunked arm, with one whose padding falls in the
    same query blocks (here both blocks of a 5-board batch hold the 2-board position's padding)."""
    for config in (*ONE_BLOCK, "n2-default-chunked"):
        two, four = _count(config, (0, 5), (1, 5)), _count(config, (0, 5), (1, 5), (2, 5), (3, 5))
        assert two.signature() == four.signature(), f"{config}: 2 and 4 positions of one size count differently"
        two, four = _count(config, (0, 2), (1, 5)), _count(config, (0, 2), (1, 5), (2, 3), (3, 5))
        assert two.signature() == four.signature(), f"{config}: padded batches of 2 and 4 positions count differently"


def test_the_block_paths_grow_with_the_board_count_and_the_tool_says_so():
    """The as-built eager default is NOT claimed to meet requirement 3: block-sparse and chunked
    make kernel calls per query block of ``attn_block_boards`` boards, so their count grows with
    the board count - and ``requirement`` reports it, while it finds the one-block paths constant.
    Replace this test when a path that meets the requirement becomes the default."""
    for config in BLOCK:
        small, large = _count(config, (0, 2)), _count(config, (0, 12))
        blocks = math.ceil(12 / _model(config).cfg.attn_block_boards)
        assert large.kernel_calls()[0] > small.kernel_calls()[0], config
        assert large.kernel_calls()[0] >= blocks, (config, large.kernel_calls(), blocks)
    for config in EVERY:
        verdicts = requirement(_rows(config))
        assert verdicts["per_layer_independent_of_boards"] is (config in ONE_BLOCK), (config, verdicts)
        for key in ("kernel_calls_equal_across_layers", "layers_equal_after_first", "outside_independent_of_boards"):
            assert verdicts[key] is True, (config, key, verdicts)


def _row(boards: int, batch: int, layer_ops, kernel_calls, outside: int = 7, seed: int = 0) -> dict:
    """A row with just the keys ``requirement`` reads."""
    counts = dict(layer_ops=list(layer_ops), kernel_calls=list(kernel_calls), outside=dict.fromkeys(OUTSIDE, outside))
    return dict(boards=boards, batch=batch, seed=seed, **counts)


def test_requirement_judges_only_what_it_was_given():
    """Constancy across layers, board counts and batch sizes of 2 and more - never a magnitude,
    and no verdict where the rows cannot give one."""
    assert tuple(requirement([_row(2, 1, [5, 4, 4], [2, 2, 2])])) == tuple(VERDICTS), "five verdicts, none about a magnitude"

    one_size = requirement(_rows("n1-dense", boards=(5,)))
    assert one_size["per_layer_independent_of_boards"] is None and one_size["outside_independent_of_boards"] is None
    assert one_size["per_layer_independent_of_batch"] is None, "one batch size"
    assert one_size["kernel_calls_equal_across_layers"] is True and one_size["layers_equal_after_first"] is True
    assert requirement(_rows("n1-dense", boards=(5,), batches=(1, 2)))["per_layer_independent_of_batch"] is None, "batch 1 is not compared"
    assert requirement(_rows("n1-dense", boards=(5,), batches=(2, 4)))["per_layer_independent_of_batch"] is True

    constant = [_row(n, b, [5, 4, 4], [2, 2, 2]) for n in (2, 12) for b in (1, 2, 4)]
    assert all(verdict is True for verdict in requirement(constant).values())
    huge = [_row(n, b, [5_000_000, 4_000_000, 4_000_000], [200_000] * 3, outside=10**9) for n in (2, 12) for b in (2, 4)]
    assert all(verdict is True for verdict in requirement(huge).values()), "a magnitude is reported, never judged"

    grows = [_row(2, 1, [5, 4, 4], [2, 2, 2]), _row(12, 1, [9, 8, 8], [6, 6, 6])]
    assert requirement(grows)["per_layer_independent_of_boards"] is False
    assert requirement(grows)["outside_independent_of_boards"] is True
    outside_grows = [_row(2, 1, [5, 4, 4], [2, 2, 2]), _row(12, 1, [5, 4, 4], [2, 2, 2], outside=8)]
    assert requirement(outside_grows)["outside_independent_of_boards"] is False
    assert requirement(outside_grows)["per_layer_independent_of_boards"] is True
    offset = [_row(5, 1, [5, 4, 4], [2, 2, 2]), _row(5, 2, [6, 5, 5], [2, 2, 2]), _row(5, 4, [6, 5, 5], [2, 2, 2])]
    assert requirement(offset)["per_layer_independent_of_batch"] is True, "batch 1's own fast paths are not growth with the batch"
    assert requirement([*offset, _row(5, 8, [7, 6, 6], [2, 2, 2])])["per_layer_independent_of_batch"] is False
    assert requirement([_row(5, 1, [5, 4, 4], [2, 3, 2])])["kernel_calls_equal_across_layers"] is False
    assert requirement([_row(5, 1, [5, 4, 3], [2, 2, 2])])["layers_equal_after_first"] is False
    assert requirement([_row(5, 1, [9, 4, 4], [2, 2, 2])])["layers_equal_after_first"] is True, "layer 0 may differ"
    shallow = requirement([_row(5, 1, [5], [2])])
    assert shallow["kernel_calls_equal_across_layers"] is None and shallow["layers_equal_after_first"] is None
    other_seed = [_row(2, 1, [5, 4, 4], [2, 2, 2], seed=0), _row(12, 1, [9, 8, 8], [6, 6, 6], seed=1)]
    assert requirement(other_seed)["per_layer_independent_of_boards"] is None, "rows of different positions' seeds are not compared"


def test_flex_is_refused():
    """``attn_impl="flex"`` cannot run under a dispatch mode (PyTorch 2.13): refused up front, with
    the reason, instead of failing inside the forward."""
    model = build_model(NetConfig(trunk_dtype="fp32", attn_impl="flex", n_layers=1), seed=0).eval()
    batch = _batch((0, 2))
    for run in (lambda: count_forward(model, batch), lambda: count_ops(lambda: model(batch), model)):
        with pytest.raises(ValueError, match="FlexAttention") as refused:
            run()
        assert "dispatch mode" in str(refused.value) and "attn_impl='flex'" in str(refused.value)
        assert _hooks(model) == 0


def test_measure_and_the_command_line(capsys):
    row = measure("n1-dense", 2, dtype="fp32", layers=2)
    assert set(row) == ROW_KEYS
    assert (row["config"], row["dtype"], row["layers"], row["boards"], row["batch"], row["seed"]) == ("n1-dense", "fp32", 2, 2, 1, 0)
    assert row["tokens"] == _batch((0, 2)).n_tokens and row["torch"] == torch.__version__
    assert len(row["layer_ops"]) == len(row["layer_compute"]) == len(row["kernel_calls"]) == 2
    assert set(row["layer_parts"]) == set(PARTS) and set(row["outside"]) == set(OUTSIDE)
    assert row["ops"] == row["compute"] + row["free"] == sum(row["layer_ops"]) + sum(row["outside"].values())
    assert row["kernel_fused"] + row["kernel_math"] == sum(row["kernel_calls"])
    assert json.loads(json.dumps(row)) == row, "a row is plain JSON"
    top = measure("n1-dense", 2, dtype="fp32", layers=2, ops=3)["top"]
    assert set(top) == {"last_layer", "context"} and all(len(items) == 3 for items in top.values())
    assert all(kind in ("free", "compute") and n > 0 for items in top.values() for _, kind, n in items)
    with pytest.raises(ValueError, match="batch"):
        measure("n1-dense", 2, dtype="fp32", layers=2, batch=0)
    capsys.readouterr()

    constant = ["--config", "n1-dense", "--boards", "2", "5", "--layers", "2", "--dtype", "fp32"]
    assert main([*constant, "--json"]) == 0
    objects = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(objects) == 3 and [set(o) for o in objects[:2]] == [ROW_KEYS, ROW_KEYS]
    assert [o["boards"] for o in objects[:2]] == [2, 5]
    assert set(objects[2]) == {"config", "dtype", "requirement"} and set(objects[2]["requirement"]) == set(VERDICTS)
    assert main([*constant, "--json", "--check"]) == 0, "nothing varies on the one-block path"
    capsys.readouterr()

    growing = ["--config", "n2-default", "--boards", "2", "12", "--layers", "2", "--dtype", "fp32"]
    assert main([*growing, "--check"]) == 1, "--check fails on a count that grows with the board count"
    text = capsys.readouterr().out
    assert all(label in text for label in VERDICTS.values())
    assert "NO" in text and "not judged" in text
    assert "counts, not timings" in text and "neither is a count of device launches" in text
    assert main(growing) == 0, "without --check the verdicts are reported, not enforced"
    capsys.readouterr()
    assert main([*constant, "--ops", "2"]) == 0
    assert "most frequent operators, the last layer" in capsys.readouterr().out
