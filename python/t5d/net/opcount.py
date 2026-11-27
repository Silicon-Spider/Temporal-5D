"""Operators and attention-kernel calls of one forward, counted on the CPU graph (register G11).

Spec section 10.2, requirement 3 (2026-11-15 review): "Launches per layer are a constant. The
number of device kernel launches in a forward is independent of N and of the batch, and small -
a few tens per layer - so that batching amortises them. It is counted, not assumed: a section
15.3 figure. The eager forward's operations can be counted on the CPU today; the compiled path's
launches are read by profiler on GPU time." This module is the first half of that last sentence:
it counts what one eager forward of ``T5DNet`` dispatches. It is a reporting tool - no code path
of the network changes, and there is no flag.

**Two figures, named for what they are. Neither is a count of device kernel launches.**

* *Operators dispatched*: every ATen operator the forward hands to the backend, counted after
  PyTorch's composite operators have decomposed - ``F.linear`` on a matrix is ``aten::t`` and
  ``aten::addmm``, never ``aten::linear``; a fused attention call is one
  ``aten::_scaled_dot_product_flash_attention_for_cpu``. Each operator has a kind
  (``op_kind``): "free" when its schema says the result aliases its input (a view: ``view``,
  ``slice``, ``expand``, ``transpose``, ...) or its name is in ``FREE_OPS`` (an allocation with
  no fill, ``_unsafe_view``); "compute" otherwise. The compute operators are the *proxy* for
  what the eager path launches on a device: each is at least one kernel or copy there, and one
  operator may launch several. The names in ``SYNC_OPS`` are also counted as host
  synchronisations (``.item()``, ``bool(tensor)``, ``nonzero``: the host waits for the device).
* *Attention-kernel calls*: calls of ``torch.nn.functional.scaled_dot_product_attention``
  (``attention._kernel`` is the network's one call site), each classed "fused" - PyTorch ran one
  fused attention operator - or "math" - it took its matmul-and-softmax path (``kernel_kind``).
  The operators a call dispatches are counted among the operators as well.

**Mechanism.** A ``TorchDispatchMode`` sees every operator at the bottom of the dispatcher (the
seam ``cost.count_flops`` already uses through ``FlopCounterMode``); a ``TorchFunctionMode``
sees the call of ``scaled_dot_product_attention`` itself - a function mode alone would count
Python API calls (property reads among them), not what the backend runs, and ``torch.profiler``
records nested events and times them. Forward hooks say where the forward is (``_attach``); every
hook is removed afterwards, whatever the forward did. The forward runs without autograd, and
its outputs are bit for bit those of a forward that is not counted
(``tests/net/test_opcount.py``).

**Call paths** (``count_ops(..., paths=True)``; ``--sites``). On request each operator is also
attributed to the network functions that were running when it was dispatched: the chain of this
package's frames above the operator, outermost first, each as ``module.qualname``
(``attention._FastPath.group_block``) - read off the interpreter's stack, so it costs time and
is off by default. ``OpCount.functions`` sums the operators by the innermost ``depth`` functions
of that chain: the map of where a layer's operators come from (``docs/MEASUREMENTS.md``,
"G11 - the kernel's widths, where a layer's operators come from"). It attributes dispatches, not time.

**Scopes.** Every operator is counted under one scope: ``("embed",)`` (the input embedding),
``("context",)`` (``T5DNet.make_context``: RoPE tables and the structure of stage N2),
``("final",)`` (the trunk outside its blocks: the final norm), ``("heads",)`` (everything after
the trunk), or ``("layer", i, part)`` with ``part`` one of ``PARTS``: "attention"
(``SelfAttention``: the projections, the attention implementation and the kernel calls), "ffn",
"norm" (the block's two norms) and "other" (``AttnContext.for_layer`` - the layer's own biases -
and the residual additions). A count taken without a model has the one scope ``("all",)``.

**What is not counted.**

* Python work, and anything that never reaches the dispatcher (``.shape``, ``.float()`` of an
  FP32 tensor, a no-op ``.to``).
* Operators a kernel issues inside its own C++ - the ``nonzero`` inside boolean-mask indexing
  is one - so the host-synchronisation figure is a lower bound.
* Device kernel launches as such: the count is of operators, on the CPU graph. Nor host-to-device
  copies: a tensor made from a Python number (``torch.tensor(x)``) reaches the dispatcher as the
  no-op ``aten::lift_fresh``, a view by its schema, and so is a free operator here.
* Time. A count does not depend on the machine, its load or its threads; it depends on the
  PyTorch build (how composites decompose), the configuration, the precision (autocast's casts
  are operators), whether the batch is one position or several, and - on the block paths - the
  position (below).
* The compiled path: ``torch.compile`` is not run here, and its launches are read by profiler
  on GPU time (section 10.2).
* ``attn_impl="flex"``: PyTorch 2.13's eager ``flex_attention`` is a higher-order operator with
  no rule for a dispatch mode, so the forward fails under one. ``count_forward`` refuses it up
  front (``_refuse_flex``); the eager count is not the figure of interest for that path anyway.
* Training steps, and the cached path: ``CachedEvaluator`` (``cache.py``) calls the blocks'
  steps directly, not through ``Block.__call__``, so the layer hooks never fire for it.
  ``count_forward`` covers ``T5DNet.forward`` only.

**What the count says of the network as built** (figures: ``docs/MEASUREMENTS.md``, "G11"; none
is pinned by a test). The requirement binds the compiled path; the eager paths are counted
against it, not claimed to meet it. The one-block paths - ``attn_impl="sdpa"`` (the
"n1-dense" and "n2-default-sdpa" configurations of ``bench.CONFIGS``) - dispatch the same
operators whatever the board count, the position and (from two positions up) the number of
positions in the batch; a batch whose positions differ in size, and so carries padding, takes a
branch of its own - a constant offset, as a single position does - so padded batches are
compared with padded ones. The block paths do not: "chunked" makes its kernel calls once per
query block of ``NetConfig.attn_block_boards`` boards, so its count grows with the board count;
and the v1 default, "block_sparse", makes them per query block and head group, so it grows the
same way, and besides differs between two positions of one size (``AttnStructure.block_plan``
takes a data-dependent branch, and a block's key selection may be empty) and builds each
block's plan in layer 0 (so layer 0 dispatches more than the layers after it).

``requirement`` judges that constancy - across layers, board counts and batch sizes of 2 and
more - and nothing else: "a few tens per layer" has no number in the spec, so the magnitude is
reported and never judged. A single position takes fast paths of its own, a constant offset, so
batch 1 is reported beside the larger batches and not compared with them.

Run from ``python/``: ``python -m t5d.net.opcount --config n2-default --boards 24 44``
(``--batch``, ``--dtype``, ``--layers``, ``--seed``; ``--ops K`` lists the most frequent
operators, ``--sites K`` the K network functions that issued the most (``--depth D``: by the
innermost D functions of the call path), ``--json`` prints one object per measurement,
``--check`` exits 1 when a constancy verdict is "NO"). Inputs are ``synthetic_state(seed + i,
boards)``, weights ``build_model(seed=0)``: random weights, which the counts do not depend on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from .bench import CONFIGS
from .config import NetConfig
from .model import build_model
from .state import collate
from .synthetic import synthetic_state

#: The scopes outside the trunk's blocks, in the order a forward passes through them.
OUTSIDE: tuple[str, ...] = ("embed", "context", "final", "heads")
#: The parts of one block (the third element of a layer's scope).
PARTS: tuple[str, ...] = ("attention", "ffn", "norm", "other")
#: Operators that are no schema view and still launch nothing: ``_unsafe_view`` (a view by
#: another name) and the allocations that write no element. A declared list - the one judgement
#: in ``op_kind``; an operator missing from it counts as "compute".
FREE_OPS: frozenset[str] = frozenset(
    {
        "aten::_unsafe_view",
        "aten::empty.memory_format",
        "aten::empty_like",
        "aten::new_empty",
        "aten::empty_strided",
        "aten::new_empty_strided",
    }
)
#: Operators whose result the host waits for: a scalar read (``.item()``, ``bool(tensor)``,
#: ``int(tensor)``), a result whose size depends on the data, a comparison returned as a Python
#: bool. Counted by name, so a synchronisation inside another operator's kernel is not seen.
SYNC_OPS: frozenset[str] = frozenset(
    {
        "aten::_local_scalar_dense",
        "aten::nonzero",
        "aten::masked_select",
        "aten::unique_dim",
        "aten::_unique2",
        "aten::unique_consecutive",
        "aten::equal",
    }
)
#: The fused attention operators' common prefix (on CPU:
#: ``aten::_scaled_dot_product_flash_attention_for_cpu``).
FUSED_PREFIX = "aten::_scaled_dot_product_"
#: ``requirement``'s verdicts and the words the command line prints for them.
VERDICTS: dict[str, str] = {
    "kernel_calls_equal_across_layers": "kernel calls equal across layers",
    "layers_equal_after_first": "layers equal after the first",
    "per_layer_independent_of_boards": "per layer independent of the board count",
    "per_layer_independent_of_batch": "per layer independent of the batch",
    "outside_independent_of_boards": "outside the layers independent of the board count",
}

#: The attention kernel, captured at import: a caller that wraps or replaces
#: ``torch.nn.functional.scaled_dot_product_attention`` afterwards still ends in this function,
#: and it is this function's calls that are counted.
_SDPA = torch.nn.functional.scaled_dot_product_attention

#: This package's directory and this file: a call path holds the frames of the first, never of
#: the second.
_PACKAGE = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
_THIS = os.path.normcase(os.path.abspath(__file__))
#: ``OpCount.functions``' name for an operator no function of this package was running for.
NO_FUNCTION = "(outside t5d.net)"
#: Code file -> its module's name in a call path (None: not this package's); filled on first sight.
_MODULE_OF: dict[str, str | None] = {}


def _module_of(filename: str) -> str | None:
    if filename not in _MODULE_OF:
        path = os.path.normcase(os.path.abspath(filename))
        inside = os.path.dirname(path) == _PACKAGE and path != _THIS
        _MODULE_OF[filename] = os.path.splitext(os.path.basename(filename))[0] if inside else None
    return _MODULE_OF[filename]


def call_path(frame) -> tuple[str, ...]:
    """The network functions running at ``frame``: every frame of this package's modules (this
    one excluded) from the outermost down to ``frame``, each ``module.qualname`` - a method
    ``attention._FastPath.attend``, a function ``displacement.lookup``, a closure by its
    enclosing function's name. Empty when none is running."""
    out = []
    while frame is not None:
        module = _module_of(frame.f_code.co_filename)
        if module is not None:
            out.append(f"{module}.{frame.f_code.co_qualname.replace('.<locals>', '')}")
        frame = frame.f_back
    return tuple(reversed(out))


def op_kind(func) -> str:
    """The kind of an operator overload (what ``__torch_dispatch__`` is handed), "free" or
    "compute": free when its schema returns an alias of an input (a view) or its name is in
    ``FREE_OPS``."""
    return "free" if func.is_view or func.name() in FREE_OPS else "compute"


def kernel_kind(names: Iterable[str]) -> str:
    """The kind of one attention-kernel call, "fused" or "math", from the operators it dispatched:
    fused if and only if exactly one of them is a fused attention operator (``FUSED_PREFIX``; a
    mask conversion beside it changes nothing)."""
    return "fused" if sum(name.startswith(FUSED_PREFIX) for name in names) == 1 else "math"


@dataclass
class OpCount:
    """What one counted run dispatched (the module docstring defines every term)."""

    #: ``(scope, operator name) -> count``; names as ``aten::add.Tensor``.
    ops: Counter = field(default_factory=Counter)
    #: ``operator name -> "free" | "compute"`` for every name in ``ops``.
    kinds: dict[str, str] = field(default_factory=dict)
    #: ``(scope, "fused" | "math") -> attention-kernel calls``.
    kernel: Counter = field(default_factory=Counter)
    #: ``scope -> operators dispatched inside attention-kernel calls`` (they are in ``ops`` too).
    kernel_ops: Counter = field(default_factory=Counter)
    #: The trunk's depth (0 for a count taken without a model).
    n_layers: int = 0
    #: ``(scope, call path, operator name) -> count`` (``call_path``): the same operators as
    #: ``ops``, by the network functions that issued them. Empty unless the count asked for paths.
    paths: Counter = field(default_factory=Counter)

    def _sum(self, keep: Callable[[tuple, str], bool]) -> int:
        return sum(n for (scope, name), n in self.ops.items() if keep(scope, name))

    def _is(self, name: str, kind: str | None) -> bool:
        return kind is None or self.kinds[name] == kind

    def total(self, kind: str | None = None) -> int:
        """Operators dispatched (``kind``: only the "compute" or only the "free" ones)."""
        return self._sum(lambda scope, name: self._is(name, kind))

    def sync(self) -> int:
        """Host synchronisations: the operators named in ``SYNC_OPS`` (a lower bound)."""
        return self._sum(lambda scope, name: name in SYNC_OPS)

    def layers(self, kind: str | None = None) -> list[int]:
        """Operators of each layer, layer 0 first."""
        out = [0] * self.n_layers
        for (scope, name), n in self.ops.items():
            if scope[0] == "layer" and self._is(name, kind):
                out[scope[1]] += n
        return out

    def layer_parts(self) -> dict[str, list[int]]:
        """Each layer's operators by part (``PARTS``): ``part -> [layer 0, layer 1, ...]``."""
        out = {part: [0] * self.n_layers for part in PARTS}
        for (scope, _), n in self.ops.items():
            if scope[0] == "layer":
                out[scope[2]][scope[1]] += n
        return out

    def outside(self, kind: str | None = None) -> dict[str, int]:
        """Operators outside the layers, by scope (``OUTSIDE``)."""
        return {where: self._sum(lambda scope, name, where=where: scope == (where,) and self._is(name, kind)) for where in OUTSIDE}

    def kernel_calls(self) -> list[int]:
        """Attention-kernel calls of each layer, layer 0 first."""
        out = [0] * self.n_layers
        for (scope, _), n in self.kernel.items():
            if scope[0] == "layer":
                out[scope[1]] += n
        return out

    def kernel_total(self, kind: str | None = None) -> int:
        """Attention-kernel calls of the whole run (``kind``: only "fused" or only "math")."""
        return sum(n for (_, k), n in self.kernel.items() if kind is None or k == kind)

    def top(self, k: int, scope_filter: Callable[[tuple], bool] | None = None) -> list[tuple[str, str, int]]:
        """The ``k`` most frequent operators as ``(name, kind, count)``, most frequent first (ties
        by name), over the scopes ``scope_filter`` accepts (None: every scope)."""
        by_name: Counter = Counter()
        for (scope, name), n in self.ops.items():
            if scope_filter is None or scope_filter(scope):
                by_name[name] += n
        ranked = sorted(by_name.items(), key=lambda item: (-item[1], item[0]))
        return [(name, self.kinds[name], n) for name, n in ranked[:k]]

    def functions(self, scope_filter: Callable[[tuple], bool] | None = None, depth: int = 1) -> list[tuple[str, int, int]]:
        """Operators by the network function that issued them, as ``(function, operators, compute
        operators)``, most operators first (ties by name), over the scopes ``scope_filter`` accepts
        (None: every scope). ``function`` is the innermost ``depth`` functions of the call path,
        joined by " > " (``NO_FUNCTION`` for an operator with none). Needs a count taken with
        paths (``count_ops(..., paths=True)``): empty otherwise."""
        if depth < 1:
            raise ValueError("depth must be at least 1")
        total: Counter = Counter()
        compute: Counter = Counter()
        for (scope, path, name), n in self.paths.items():
            if scope_filter is None or scope_filter(scope):
                function = " > ".join(path[-depth:]) if path else NO_FUNCTION
                total[function] += n
                if self.kinds[name] == "compute":
                    compute[function] += n
        return [(function, n, compute[function]) for function, n in sorted(total.items(), key=lambda item: (-item[1], item[0]))]

    def signature(self) -> tuple:
        """Everything counted, hashable: every operator of every scope, and the kernel calls. Two
        runs with one signature dispatched the same operators in the same places."""
        return (tuple(sorted(self.ops.items())), tuple(sorted(self.kernel.items())))


class _State:
    """Where the forward is (set by ``_attach``'s hooks) and what has been counted so far."""

    def __init__(self) -> None:
        self.count = OpCount()
        self.scoped = False  # a model's hooks are attached: scopes are the forward's
        self.phase = "context"  # the scope outside the blocks when no hooked module is running
        self.layer: int | None = None  # the running block's index
        self.parts: list[str] = []  # the hooked modules running, innermost last
        self.in_kernel = False
        self.kernel_names: list[str] = []  # the operators of the running kernel call
        self.paths = False  # also attribute every operator to its call path (``call_path``)

    def scope(self) -> tuple:
        if not self.scoped:
            return ("all",)
        if self.layer is not None:
            return ("layer", self.layer, self.parts[-1] if self.parts else "other")
        return (self.parts[-1],) if self.parts else (self.phase,)


class _Dispatch(TorchDispatchMode):
    """Counts every operator that reaches the bottom of the dispatcher, then runs it unchanged."""

    def __init__(self, state: _State) -> None:
        super().__init__()
        self.state = state

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        state = self.state
        name = func.name()
        scope = state.scope()
        state.count.ops[(scope, name)] += 1
        if name not in state.count.kinds:
            state.count.kinds[name] = op_kind(func)
        if state.in_kernel:
            state.kernel_names.append(name)
        if state.paths:
            state.count.paths[(scope, call_path(sys._getframe(1)), name)] += 1
        return func(*args, **(kwargs or {}))


class _KernelCalls(TorchFunctionMode):
    """Counts the calls of ``scaled_dot_product_attention`` (``_SDPA``) and classes each by the
    operators it dispatched; every other function passes through untouched."""

    def __init__(self, state: _State) -> None:
        super().__init__()
        self.state = state

    def __torch_function__(self, func, types, args=(), kwargs=None):
        state = self.state
        if func is not _SDPA or state.in_kernel:
            return func(*args, **(kwargs or {}))
        state.in_kernel, state.kernel_names = True, []
        try:
            return func(*args, **(kwargs or {}))
        finally:
            state.in_kernel = False
            scope = state.scope()
            state.count.kernel[(scope, kernel_kind(state.kernel_names))] += 1
            state.count.kernel_ops[scope] += len(state.kernel_names)


def _attach(model: nn.Module, state: _State, handles: list) -> None:
    """Forward hooks on ``model`` (a ``T5DNet``) that keep ``state``'s scope: the model itself
    (each forward starts in "context"), its embedding, its trunk ("final" inside it and outside
    the blocks, "heads" after it) and every block with its attention, FFN and two norms. Each
    handle goes into ``handles`` as it is made, so the caller removes them even if this raises;
    every hook that closes a scope is registered with ``always_call`` and so runs when the
    forward raises."""

    def phase(name: str):
        def hook(*_):
            state.phase = name

        return hook

    def layer(index: int | None):
        def hook(*_):
            state.layer = index

        return hook

    def push(name: str):
        def hook(*_):
            state.parts.append(name)

        return hook

    def pop(*_):
        state.parts.pop()

    def part(module: nn.Module, name: str) -> None:
        handles.append(module.register_forward_pre_hook(push(name)))
        handles.append(module.register_forward_hook(pop, always_call=True))

    handles.append(model.register_forward_pre_hook(phase("context")))
    part(model.embed, "embed")
    handles.append(model.trunk.register_forward_pre_hook(phase("final")))
    handles.append(model.trunk.register_forward_hook(phase("heads"), always_call=True))
    for index, block in enumerate(model.trunk.blocks):
        handles.append(block.register_forward_pre_hook(layer(index)))
        handles.append(block.register_forward_hook(layer(None), always_call=True))
        part(block.attn, "attention")
        part(block.ffn, "ffn")
        part(block.norm_attn, "norm")
        part(block.norm_ffn, "norm")
    state.scoped = True
    state.count.n_layers = len(model.trunk.blocks)


def _refuse_flex(model: nn.Module) -> None:
    """``attn_impl="flex"`` cannot be counted (the module docstring): say so before anything runs."""
    if getattr(getattr(model, "cfg", None), "attn_impl", None) == "flex":
        raise ValueError(
            "attn_impl='flex' cannot be counted here: PyTorch's eager FlexAttention (flex_attention, a higher-order operator) "
            f"has no rule for a dispatch mode in torch {torch.__version__}, so its forward fails under the counter. Count "
            "another attention path; the compiled FlexAttention path's launches are read by profiler on GPU time (spec "
            "section 10.2)"
        )


def count_ops(fn: Callable[[], object], model: nn.Module | None = None, paths: bool = False) -> tuple[object, OpCount]:
    """``fn()`` and the count of what it dispatched, without autograd. With ``model`` (a
    ``T5DNet`` that ``fn`` runs forward), the operators are scoped by where that forward is
    (the module docstring's scopes; whatever ``fn`` dispatches around the forward falls in
    "context" before it and "heads" after it); without, every operator is in the scope
    ``("all",)``. ``paths``: also attribute every operator to its call path (``OpCount.paths``;
    slower, and the counts are the same). Every hook is removed before this returns or raises."""
    state = _State()
    state.paths = paths
    handles: list = []
    try:
        if model is not None:
            _refuse_flex(model)
            _attach(model, state, handles)
        with torch.no_grad(), _KernelCalls(state), _Dispatch(state):
            result = fn()
    finally:
        for handle in handles:
            handle.remove()
    return result, state.count


def count_forward(model: nn.Module, batch, paths: bool = False, **forward_kwargs) -> OpCount:
    """The count of one ``model(batch, **forward_kwargs)`` (``T5DNet.forward``), without
    autograd (``paths``: with every operator's call path). Raises ``ValueError`` for a network
    of ``attn_impl="flex"``."""
    _, count = count_ops(lambda: model(batch, **forward_kwargs), model, paths)
    return count


def measure(
    config: str,
    boards: int,
    dtype: str = "bf16",
    batch: int = 1,
    seed: int = 0,
    layers: int | None = None,
    ops: int = 0,
    model: nn.Module | None = None,
    sites: int = 0,
    depth: int = 1,
) -> dict:
    """One counted forward as a row of plain numbers (JSON types only).

    The network is ``bench.CONFIGS[config]`` at precision ``dtype`` with ``layers`` layers (None:
    the configuration's), random weights (``build_model(seed=0)`` - or ``model``, a network the
    caller already built from exactly that configuration, so that several rows share one); the
    input is ``batch`` positions of ``boards`` boards, ``synthetic_state(seed + i, boards)`` -
    register A5's tokenisation of it for a piece-centric network (``bench.CONFIGS``'
    "a5-piece-centric").

    Keys: ``config``, ``dtype``, ``layers``, ``boards``, ``batch``, ``seed``, ``tokens`` (the
    batch's padded length), ``torch`` (the PyTorch version: the counts are that build's);
    ``ops``, ``compute``, ``free``, ``sync`` (the forward's totals); ``layer_ops`` and
    ``layer_compute`` (per layer), ``layer_parts`` (``part -> per layer``); ``kernel_calls``
    (per layer), ``kernel_fused``, ``kernel_math``, ``kernel_ops`` (totals); ``outside``
    (``OUTSIDE``'s scopes). With ``ops`` > 0 also ``top``: the ``ops`` most frequent operators
    ``[name, kind, count]`` of the last layer (``last_layer``) and of ``context``. With ``sites``
    > 0 also ``sites``: the ``sites`` network functions that issued the most operators,
    ``[function, operators, compute operators]`` (``OpCount.functions`` at ``depth``), of the
    first layer (``first_layer``: what is built on first use shows there), of the last layer
    and of ``context`` - the counts themselves are the same with or without it."""
    if batch < 1:
        raise ValueError("batch must be at least 1")
    if model is None:
        overrides = CONFIGS[config] | ({} if layers is None else {"n_layers": layers})
        model = build_model(NetConfig(trunk_dtype=dtype, **overrides), seed=0).eval()
    piece_centric = bool(getattr(getattr(model, "cfg", None), "piece_centric", False))  # A5's arm: its own tokens
    inputs = collate([synthetic_state(seed + i, boards, piece_centric=piece_centric) for i in range(batch)])
    count = count_forward(model, inputs, paths=sites > 0)
    last = count.n_layers - 1
    row = {
        "config": config,
        "dtype": dtype,
        "layers": count.n_layers,
        "boards": boards,
        "batch": batch,
        "seed": seed,
        "tokens": inputs.n_tokens,
        "torch": torch.__version__,
        "ops": count.total(),
        "compute": count.total("compute"),
        "free": count.total("free"),
        "sync": count.sync(),
        "layer_ops": count.layers(),
        "layer_compute": count.layers("compute"),
        "layer_parts": count.layer_parts(),
        "kernel_calls": count.kernel_calls(),
        "kernel_fused": count.kernel_total("fused"),
        "kernel_math": count.kernel_total("math"),
        "kernel_ops": sum(count.kernel_ops.values()),
        "outside": count.outside(),
    }
    if ops > 0:
        row["top"] = {
            "last_layer": [list(item) for item in count.top(ops, lambda scope: scope[:2] == ("layer", last))],
            "context": [list(item) for item in count.top(ops, lambda scope: scope == ("context",))],
        }
    if sites > 0:
        row["sites"] = {
            "first_layer": [list(item) for item in count.functions(lambda scope: scope[:2] == ("layer", 0), depth)[:sites]],
            "last_layer": [list(item) for item in count.functions(lambda scope: scope[:2] == ("layer", last), depth)[:sites]],
            "context": [list(item) for item in count.functions(lambda scope: scope == ("context",), depth)[:sites]],
        }
    return row


def _fmt(n: int) -> str:
    return f"{n:,}"


def _span(values: Iterable[int]) -> str:
    """One value when they are all equal, ``low-high`` when they are not, ``-`` for none."""
    values = list(values)
    if not values:
        return "-"
    low, high = min(values), max(values)
    return _fmt(low) if low == high else f"{_fmt(low)}-{_fmt(high)}"


def _groups(rows: Iterable[dict], key: Callable[[dict], tuple], vary: str) -> list[list[dict]]:
    """``rows`` grouped by ``key``, keeping the groups in which ``vary`` takes two values or more."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return [sorted(group, key=lambda r: r[vary]) for _, group in sorted(groups.items()) if len({r[vary] for r in group}) > 1]


def _per_layer(row: dict) -> tuple:
    return (tuple(row["layer_ops"]), tuple(row["kernel_calls"]))


def _across(group: list[dict], vary: str) -> str:
    """How the last layer's operators and kernel calls run across a group that should agree."""
    last_ops = " / ".join(_fmt(r["layer_ops"][-1]) for r in group)
    calls = " / ".join(_fmt(r["kernel_calls"][-1]) for r in group)
    values = " / ".join(str(r[vary]) for r in group)
    what = "boards" if vary == "boards" else "positions a batch"
    return f"the last layer dispatches {last_ops} operators and makes {calls} kernel calls at {values} {what}"


def requirement_report(rows: Sequence[dict]) -> dict[str, tuple[bool | None, str]]:
    """``requirement``'s verdicts, each with a note: for a False verdict the first rows that
    differ, for None why nothing was judged, for True an empty string."""
    rows = list(rows)
    report: dict[str, tuple[bool | None, str]] = {}

    def where(row: dict) -> str:
        return f"{row['boards']} boards, batch {row['batch']}"

    judged = [r for r in rows if len(r["kernel_calls"]) > 1]
    bad = [r for r in judged if len(set(r["kernel_calls"])) > 1]
    if not judged:
        report["kernel_calls_equal_across_layers"] = (None, "no row has two layers")
    else:
        note = f"{where(bad[0])}: {_span(bad[0]['kernel_calls'])} kernel calls a layer" if bad else ""
        report["kernel_calls_equal_across_layers"] = (not bad, note)

    judged = [r for r in rows if len(r["layer_ops"]) > 2]
    bad = [r for r in judged if len(set(r["layer_ops"][1:])) > 1]
    if not judged:
        report["layers_equal_after_first"] = (None, "no row has three layers")
    else:
        note = f"{where(bad[0])}: {_span(bad[0]['layer_ops'][1:])} operators a layer after the first" if bad else ""
        report["layers_equal_after_first"] = (not bad, note)

    by_boards = _groups(rows, lambda r: (r["batch"], r["seed"]), "boards")
    if not by_boards:
        report["per_layer_independent_of_boards"] = (None, "needs two board counts at one batch size")
        report["outside_independent_of_boards"] = (None, "needs two board counts at one batch size")
    else:
        bad_groups = [g for g in by_boards if len({_per_layer(r) for r in g}) > 1]
        note = f"batch {bad_groups[0][0]['batch']}: {_across(bad_groups[0], 'boards')}" if bad_groups else ""
        report["per_layer_independent_of_boards"] = (not bad_groups, note)
        bad_groups = [g for g in by_boards if len({tuple(sorted(r["outside"].items())) for r in g}) > 1]
        note = ""
        if bad_groups:
            group = bad_groups[0]
            sums = " / ".join(_fmt(sum(r["outside"].values())) for r in group)
            note = f"batch {group[0]['batch']}: {sums} operators outside the layers at {' / '.join(str(r['boards']) for r in group)} boards"
        report["outside_independent_of_boards"] = (not bad_groups, note)

    by_batch = _groups([r for r in rows if r["batch"] >= 2], lambda r: (r["boards"], r["seed"]), "batch")
    if not by_batch:
        report["per_layer_independent_of_batch"] = (None, "needs two batch sizes of 2 or more at one board count (batch 1 is not compared)")
    else:
        bad_groups = [g for g in by_batch if len({_per_layer(r) for r in g}) > 1]
        note = f"{bad_groups[0][0]['boards']} boards: {_across(bad_groups[0], 'batch')}" if bad_groups else ""
        report["per_layer_independent_of_batch"] = (not bad_groups, note)

    return {key: report[key] for key in VERDICTS}


def requirement(rows: Sequence[dict]) -> dict[str, bool | None]:
    """Section 10.2's requirement 3 as far as counting can judge it, over ``measure``'s rows of one
    configuration and precision. Constancy only - no verdict is about a magnitude ("a few tens
    per layer" has no number in the spec; the counts are reported). True, False, or None where the
    rows cannot say:

    * ``kernel_calls_equal_across_layers``: every row makes the same number of attention-kernel
      calls in each of its layers (None: no row has two layers).
    * ``layers_equal_after_first``: every row dispatches the same number of operators in each
      layer after layer 0, which may carry what is built on first use (None: no row has three).
    * ``per_layer_independent_of_boards``: at one batch size (and seed), every layer's operators
      and kernel calls are the same at every board count (None: no two board counts).
    * ``per_layer_independent_of_batch``: at one board count (and seed), the same across batch
      sizes of 2 and more (None: no two such batch sizes; batch 1 is never compared - a single
      position takes fast paths of its own).
    * ``outside_independent_of_boards``: at one batch size (and seed), the operators outside the
      layers (``OUTSIDE``) are the same at every board count (None: no two board counts)."""
    return {key: verdict for key, (verdict, _) in requirement_report(rows).items()}


#: The text table: column groups of ``(heading, row -> cell)``.
_TABLE: tuple[tuple[tuple[str, Callable[[dict], str]], ...], ...] = (
    (("boards", lambda r: _fmt(r["boards"])), ("batch", lambda r: _fmt(r["batch"])), ("tokens", lambda r: _fmt(r["tokens"]))),
    (
        ("ops", lambda r: _fmt(r["ops"])),
        ("compute", lambda r: _fmt(r["compute"])),
        ("free", lambda r: _fmt(r["free"])),
        ("sync", lambda r: _fmt(r["sync"])),
    ),
    (
        ("layer 0", lambda r: _fmt(r["layer_ops"][0])),
        ("layers 1+", lambda r: _span(r["layer_ops"][1:])),
        ("(compute)", lambda r: f"({_span(r['layer_compute'][1:])})"),
    ),
    tuple(("attn" if part == "attention" else part, lambda r, part=part: _fmt(r["layer_parts"][part][-1])) for part in PARTS),
    (
        ("kernel/layer", lambda r: _span(r["kernel_calls"])),
        ("fused", lambda r: _fmt(r["kernel_fused"])),
        ("math", lambda r: _fmt(r["kernel_math"])),
    ),
    tuple((where, lambda r, where=where: _fmt(r["outside"][where])) for where in OUTSIDE),
)
_LEGEND = (
    "  ops: operators dispatched; compute: those that are no view and no bare allocation (the eager proxy for device launches,\n"
    "  not a launch count); free: the rest; sync: host synchronisations by name (a lower bound). layer 0 / layers 1+: operators\n"
    "  of a layer (compute in brackets). attn, ffn, norm, other: the last layer's operators by part. kernel/layer: attention-\n"
    "  kernel calls a layer; fused, math: the forward's kernel calls by kind. embed, context, final, heads: outside the layers."
)


def _table(rows: Sequence[dict]) -> str:
    lines = [[[heading for heading, _ in group] for group in _TABLE]]
    lines += [[[cell(row) for _, cell in group] for group in _TABLE] for row in rows]
    widths = [[max(len(line[g][c]) for line in lines) for c in range(len(group))] for g, group in enumerate(_TABLE)]

    def text(line: list[list[str]]) -> str:
        return " | ".join(" ".join(cell.rjust(widths[g][c]) for c, cell in enumerate(group)) for g, group in enumerate(line))

    return "\n".join("  " + text(line) for line in lines)


def _top_lines(row: dict) -> list[str]:
    out = []
    for where, title in (("last_layer", "the last layer"), ("context", "context")):
        out.append(f"  most frequent operators, {title} ({row['boards']} boards, batch {row['batch']}):")
        out += [f"    {n:>7,}  {kind:<7}  {name}" for name, kind, n in row["top"][where]]
    return out


def _site_lines(row: dict) -> list[str]:
    out = []
    for where, title in (("first_layer", "the first layer"), ("last_layer", "the last layer"), ("context", "context")):
        out.append(f"  operators by the network function that issued them, {title} ({row['boards']} boards, batch {row['batch']}):")
        out.append("        ops  compute  function")
        out += [f"    {n:>7,}  {compute:>7,}  {function}" for function, n, compute in row["sites"][where]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    names = sorted(CONFIGS)
    parser.add_argument(
        "--config", nargs="+", default=["n2-default"], choices=names, metavar="NAME", help=f"of bench.CONFIGS: {', '.join(names)}"
    )
    parser.add_argument("--boards", type=int, nargs="+", default=[24, 44], help="board counts (synthetic_state(seed + i, N))")
    parser.add_argument("--batch", type=int, nargs="+", default=[1], help="positions a batch")
    parser.add_argument("--dtype", nargs="+", default=["bf16"], choices=["bf16", "fp32"], help="trunk precision")
    parser.add_argument("--layers", type=int, default=None, help="trunk depth (default: the configuration's, 16)")
    parser.add_argument("--seed", type=int, default=0, help="the first position's seed")
    parser.add_argument("--ops", type=int, default=0, metavar="K", help="also list the K most frequent operators")
    parser.add_argument("--sites", type=int, default=0, metavar="K", help="also list the K network functions that issued the most")
    parser.add_argument("--depth", type=int, default=1, metavar="D", help="with --sites: name a function by the innermost D of its path")
    parser.add_argument("--json", action="store_true", help="one JSON object per measurement and per verdict set")
    parser.add_argument("--check", action="store_true", help="exit 1 when a constancy verdict is NO (no magnitude is judged)")
    args = parser.parse_args(argv)
    failed = False
    if not args.json:
        print(f"t5d.net.opcount: torch {torch.__version__}; eager, no_grad; counts, not timings")
        print("  (operators dispatched and attention-kernel calls on the CPU graph - neither is a count of device launches)")
        print(_LEGEND)
    for config in args.config:
        for dtype in args.dtype:
            overrides = CONFIGS[config] | ({} if args.layers is None else {"n_layers": args.layers})
            model = build_model(NetConfig(trunk_dtype=dtype, **overrides), seed=0).eval()
            rows = [
                measure(config, boards, dtype, batch, args.seed, args.layers, ops=args.ops, model=model, sites=args.sites, depth=args.depth)
                for boards in args.boards
                for batch in args.batch
            ]
            report = requirement_report(rows)
            failed = failed or any(verdict is False for verdict, _ in report.values())
            if args.json:
                for row in rows:
                    print(json.dumps(row), flush=True)
                print(json.dumps({"config": config, "dtype": dtype, "requirement": {k: v for k, (v, _) in report.items()}}), flush=True)
                continue
            print(f"\n{config}, {dtype}, {rows[0]['layers']} layers, seed {args.seed}")
            print(_table(rows))
            print("  constancy (section 10.2, requirement 3; no magnitude is judged):")
            width = max(len(label) for label in VERDICTS.values())
            for key, label in VERDICTS.items():
                verdict, note = report[key]
                word = "not judged" if verdict is None else "yes" if verdict else "NO"
                print(f"    {label.ljust(width)}  {word}{': ' + note if note else ''}")
            if args.ops > 0:
                for row in rows:
                    print("\n".join(_top_lines(row)), flush=True)
            if args.sites > 0:
                for row in rows:
                    print("\n".join(_site_lines(row)), flush=True)
    return 1 if args.check and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
