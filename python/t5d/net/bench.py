"""Indicative cost of one forward pass (or one training step): wall time and peak memory (B14).

Each measurement runs in a fresh subprocess, so that the process's peak working set
belongs to that pass alone: the subprocess builds the model and the input, runs a small
warm-up pass, records its peak, then times one pass at the requested size and records the
peak again. A forward runs without autograd; a training step (``--train``) is a forward
with autograd and the backward of a loss on the value and policy logits. Reported: wall
time of the timed pass, the process's peak working set, and the peak's increase over the
pre-pass peak - the pass's own working set above weights and input.

These are development-machine figures for the attention structure of stage N2 - CPU,
PyTorch's stock kernels - not the section 15.3 cost measurements (those are GPU-class
figures with their conditions recorded in docs/MEASUREMENTS.md).

Run: ``python -m t5d.net.bench --boards 60 240 --dtype fp32 bf16`` (from ``python/``);
``--train --layers 2`` for a training step of a 2-layer trunk.

Measured 2026-10-01 (AMD Ryzen 9 9900X, 12 PyTorch threads, Windows 11, PyTorch
2.13.0+cpu; one position, ``synthetic_state(0, N)``: 3,976 tokens at 60 boards, 15,856 at
240; each figure one run on an otherwise idle machine, except * - one other single-threaded
process ran alongside). Seconds / peak GB / the pass's own GB:

====================== ========================= ========================= =========================
forward, 16 layers     60 boards FP32            240 boards FP32           240 boards BF16
====================== ========================= ========================= =========================
N1, no mask            0.86 / 0.42 / 0.12        8.5 / 0.69 / 0.37         19.5 / 0.66 / 0.29
B9 default, sparse     1.26 / 0.49 / 0.18        9.0 / 0.86 / 0.54         18.6 / 0.76 / 0.39
B9 default, chunked    1.42 / 0.48 / 0.18        13.8* / 0.80 / 0.49       26.3 / 0.70 / 0.33
B9 default, sdpa       1.35 / 0.56 / 0.26        14.4 / 3.38 / 3.06
routing, sparse        2.14 / 0.45 / 0.14        14.9* / 0.90 / 0.58
routing, chunked       2.52 / 0.47 / 0.17        28.5* / 0.81 / 0.50
every switch, sparse   2.90 / 0.48 / 0.14        29.5 / 0.93 / 0.58        45.3 / 0.94 / 0.53
every switch, chunked  2.96 / 0.52 / 0.18        39.3 / 1.06 / 0.71        61.5 / 1.04 / 0.64
every switch, sdpa     3.22 / 1.60 / 1.26
====================== ========================= ========================= =========================

A training step (FP32): at 40 boards, 16 layers, N1 1.86 s / 1.35 GB of its own; B9's
default 3.40 s / 1.47 GB with blocks recomputed, 3.03 s / 2.33 GB keeping them; every switch
7.57 s / 1.36 GB recomputed, 6.74 s / 6.15 GB keeping them. At 240 boards, 2 layers: N1
3.85 s / 1.29 GB, B9's default 5.60 s / 1.45 GB (5.31 s / 2.07 GB keeping its blocks), every
switch 25.1 s / 1.49 GB. Learned biases make the kernel's mask require gradient, which on
CPU forces PyTorch's math kernel inside every block that carries them (the time of "every
switch").
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

#: Named configurations: N1's dense baseline (no masks, the one-call fused path), stage
#: N2's default (B9 on 2 heads, block-sparse), the routing table alone, and every switch
#: of N2 on; "*-chunked" runs the same structure through ablation 6's dense-mask arm,
#: "*-sdpa" through the unchunked fused path, which materialises (B, H, L, L) masks, and
#: "*-keep" keeps every block for the backward pass instead of recomputing it
#: (``attn_recompute=False``; training steps only).
_ALL = dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True, binned_bias=True)
CONFIGS: dict[str, dict] = {
    "n1-dense": dict(move_graph_heads=0, attn_impl="sdpa"),
    "n2-default": dict(),
    "n2-default-chunked": dict(attn_impl="chunked"),
    "n2-default-sdpa": dict(attn_impl="sdpa"),
    "n2-routing": dict(routing_table=True),
    "n2-routing-chunked": dict(routing_table=True, attn_impl="chunked"),
    "n2-all": _ALL,
    "n2-all-chunked": _ALL | dict(attn_impl="chunked"),
    "n2-all-sdpa": _ALL | dict(attn_impl="sdpa"),
    "n2-default-keep": dict(attn_recompute=False),
    "n2-default-chunked-keep": dict(attn_impl="chunked", attn_recompute=False),
    "n2-all-keep": _ALL | dict(attn_recompute=False),
}


def _peak_bytes() -> int:
    """The process's peak working set (Windows) or maximum resident set size (POSIX)."""
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = ctypes.c_void_p  # a 64-bit pseudo-handle, not a C int
        info = ctypes.windll.psapi.GetProcessMemoryInfo
        info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not info(current(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def measure(config: str, boards: int, dtype: str, seed: int = 0, train: bool = False, layers: int | None = None) -> dict:
    """One measurement, in this process (call through ``run`` for a clean peak)."""
    import torch

    from .config import NetConfig
    from .model import build_model
    from .state import collate
    from .synthetic import synthetic_state

    overrides = CONFIGS[config] | ({} if layers is None else {"n_layers": layers})
    cfg = NetConfig(trunk_dtype=dtype, **overrides)
    model = build_model(cfg, seed=seed).eval()
    batch = collate([synthetic_state(seed, boards)])
    small = collate([synthetic_state(seed + 1, 2)])

    def step(b):
        if not train:
            with torch.no_grad():
                return model(b)
        out = model(b)
        (out.value_logits.sum() + out.policy_logits[out.move_valid].sum()).backward()
        model.zero_grad(set_to_none=True)
        return out

    step(small)  # warm-up: kernels, thread pool
    before = _peak_bytes()
    start = time.perf_counter()
    out = step(batch)
    seconds = time.perf_counter() - start
    after = _peak_bytes()
    assert torch.isfinite(out.value_logits).all()
    return {
        "config": config,
        "boards": boards,
        "tokens": batch.n_tokens,
        "edges": int(batch.edge_valid.sum()),
        "dtype": dtype,
        "layers": cfg.n_layers,
        "pass": "train" if train else "forward",
        "threads": torch.get_num_threads(),
        "seconds": round(seconds, 3),
        "peak_gb": round(after / 2**30, 3),
        "forward_gb": round((after - before) / 2**30, 3),
    }


def run(config: str, boards: int, dtype: str, timeout: float = 3600.0, train: bool = False, layers: int | None = None) -> dict:
    """``measure`` in a fresh subprocess; a failure (out of memory, timeout) is reported."""
    args = f"{config!r}, {boards}, {dtype!r}, train={train}, layers={layers}"
    code = f"import json; from t5d.net.bench import measure; print(json.dumps(measure({args})))"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"config": config, "boards": boards, "dtype": dtype, "error": f"timeout after {timeout:.0f} s"}
    if proc.returncode != 0:
        return {"config": config, "boards": boards, "dtype": dtype, "error": (proc.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boards", type=int, nargs="+", default=[60, 240])
    parser.add_argument("--dtype", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    parser.add_argument("--config", nargs="+", default=["n1-dense", "n2-default", "n2-all"], choices=sorted(CONFIGS))
    parser.add_argument("--train", action="store_true", help="time a training step (forward and backward) instead")
    parser.add_argument("--layers", type=int, default=None, help="trunk depth (default: the configuration's, 16)")
    args = parser.parse_args(argv)
    for boards in args.boards:
        for dtype in args.dtype:
            for config in args.config:
                print(json.dumps(run(config, boards, dtype, train=args.train, layers=args.layers)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
