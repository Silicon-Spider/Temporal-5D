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

The quiet-machine block's arms (2026-11-28; ``CONFIGS``, below; commands in docs/MEASUREMENTS.md
beside the rows they fill): section 15.3's FFN row - ``--config n2-default ffn-4x ffn-2x --boards
44 240`` (the trunk's FFN at 8x, 4x, 2x) - and A5's dense wall time beside A1's - ``--config
n2-default a5-piece-centric --boards 44``. A5's arm reads ``synthetic_state(0, N,
piece_centric=True)``: at 44 boards 1,378 tokens against A1's 2,920 (2.12x fewer), where the human
position of the A5 section's indicative figure has 1,491 (1.96x).

Options for the block: ``--embed`` also times ``T5DNet.embed_tokens`` alone on the same position,
in the same process after the forward (``embed_ms``, beside the forward's ``seconds``: register
A8's embedding-time column); ``--repeat R`` runs each (boards, dtype) block R times, the
configurations' order reversed every other repetition, each pass in its own process, then prints
each configuration's medians and minima (``"summary": "median"`` lines) - the A8 section's protocol
with R = 5; ``--threads T`` sets PyTorch's threads in every pass (default: PyTorch's own - 12 on
the development machine; the A5 section's indicative figure used 4). The A8 table's arms -
"n2-default" (off), "6p-layout-ff8" (A8), "a8-no-ffn" (A8 without FFN) - with "6p-layout-ff1":
``--config n2-default 6p-layout-ff8 a8-no-ffn 6p-layout-ff1 --boards 44 240 --dtype bf16 fp32
--embed --repeat 5``.

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

These figures are of the network before F5 and F6 (CHECKPOINTS 6-7: LayerNorm with
biases, biased Q/K/V projections, GELU in the FFN), which F5 replaced with gain-only RMSNorm
and no Q/K/V biases - less per-token work, no new matmul - and F6 with Mish; they were not
re-measured (no quiet machine since). They are also of the network before B15: the rows
"B9 default" are today's "n2-no-table", and the v1 default ("n2-default") carries B15's
learned bias on every head - per-row biases on the dense heads too, as "every switch" had
(section 15.3's "Displacement bias and normalisation changes" row is to measure). The
"every switch" figures (the table's rows and the training steps) had B6's binned biases on,
since removed, and no B15; today's "n2-all" carries B15 in their place - a per-layer lookup
on every head where B6 added one board-pair table per batch - so they are not "n2-all"'s
either. And they are of the network before M20's register positions (B8, CHECKPOINT 6): its
per-board registers sat at a learned off-lattice angle, so only the global registers' pairs
were merged; position-free per-board registers widen every row's features and recompute the
register rows - about +35% / +11% per attention call at 240 boards, FP32 / BF16, over that
default by ``attention.py``'s per-call timings (indicative, not measurements).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

#: Named configurations: N1's dense baseline (no masks, no displacement bias, the one-call
#: fused path), the v1 default (B9 on 2 heads and B15's displacement bias, block-sparse) and
#: its no-table arm ("n2-no-table": ablation 2, B15 off - the stage N2 default before
#: CHECKPOINT 6), the routing table alone, and every switch of N2 on; "*-chunked" runs the
#: same structure through ablation 6's dense-mask arm, "*-sdpa" through the unchunked fused
#: path, which materialises (B, H, L, L) masks, and "*-keep" keeps every block for the
#: backward pass instead of recomputing it (``attn_recompute=False``; training steps only);
#: "6y-*" are ablation 6y's trunk arms on the v1 default, "b17-shaw" is ablation 2's
#: key-side Shaw arm on it (section 15.3's B17 row: its wall time beside "n2-default"'s), and
#: "6p-layout-*" are ablation 6p's two board-layout arms on it (register A8, section 2.2: A8 on
#: with its own FFN at the placeholder's 8x and at 1x - ``config.PRESETS``' entries of the same
#: names; section 15.3's A8 row: their wall time beside "n2-default"'s, not yet measured).
#: "b15-*" are B15's arms on the v1 default (register B15, section 4.1; the 2026-11-15 review):
#: "b15-layer-shared" is ablation 2's layer-shared arm (``disp_layer_shared``: one table per head
#: for every layer, its bias assembled once per forward at inference - register G11), "-sdpa" the
#: same through the one-block path, and "b15-no-far" B15 without B6's two per-axis far tables
#: (``disp_far_axis=False``: the network as built before the review). "g11-pad8*" are the v1
#: default with the kernel's widths padded to a multiple of 8 (register G11, section 10.2;
#: ``attn_pad_multiple``). "ffn-2x" and "ffn-4x" are the v1 default with the trunk's FFN at 2x and
#: 4x of d (``ffn_ratio``; F1's is 8x - "n2-default"), all else equal: section 15.3's row "FFN ratio
#: 2x and 4x" (section 9.2's estimate, 1.39x and 1.23x evaluations/s at 44 boards - their wall
#: time beside "n2-default"'s). "a5-piece-centric" is the v1 default reading register A5's
#: piece-centric tokens (``piece_centric``, the dense scope; ``synthetic_state(...,
#: piece_centric=True)``, the same draws as A1's position): A5's dense wall time beside A1's
#: ("n2-default"). "a8-no-ffn" is A8 without its FFN (``layout_ffn_ratio=0``: the projection and
#: the merge alone - the A8 section's third column, "A8 without FFN", not register A8 itself),
#: for that table's embedding and forward times beside "n2-default" and the two 6p arms. The
#: operator counter's command line reaches all of them
#: (``python -m t5d.net.opcount --config b15-layer-shared``).
_ALL = dict(routing_table=True, typed_edge_bias=True, board_pair_smolgen=True)
CONFIGS: dict[str, dict] = {
    "n1-dense": dict(move_graph_heads=0, attn_impl="sdpa", displacement_bias=False),
    "n2-default": dict(),
    "n2-no-table": dict(displacement_bias=False),
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
    "6y-post-norm": dict(norm_placement="post"),
    "6y-squared-relu": dict(ffn_activation="squared_relu"),
    "b17-shaw": dict(shaw_key_term=True),
    "6p-layout-ff8": dict(board_layout_embedding=True, layout_ffn_ratio=8),
    "6p-layout-ff1": dict(board_layout_embedding=True, layout_ffn_ratio=1),
    "b15-layer-shared": dict(disp_layer_shared=True),
    "b15-layer-shared-sdpa": dict(disp_layer_shared=True, attn_impl="sdpa"),
    "b15-no-far": dict(disp_far_axis=False),
    "g11-pad8": dict(attn_pad_multiple=8),
    "g11-pad8-sdpa": dict(attn_pad_multiple=8, attn_impl="sdpa"),
    "ffn-2x": dict(ffn_ratio=2),
    "ffn-4x": dict(ffn_ratio=4),
    "a5-piece-centric": dict(piece_centric=True),
    "a8-no-ffn": dict(board_layout_embedding=True, layout_ffn_ratio=0),
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


def measure(
    config: str,
    boards: int,
    dtype: str,
    seed: int = 0,
    train: bool = False,
    layers: int | None = None,
    embed: bool = False,
    threads: int | None = None,
) -> dict:
    """One measurement, in this process (call through ``run`` for a clean peak). ``embed``: then
    also time ``T5DNet.embed_tokens`` alone on the same position (``embed_ms``; a forward only) -
    after the forward's timing and its peak are taken, so neither changes, and warm: the forward
    has just embedded that position. ``threads``: PyTorch's CPU threads (None: its default)."""
    import torch

    from .config import NetConfig
    from .model import build_model
    from .state import collate
    from .synthetic import synthetic_state

    if embed and train:
        raise ValueError("embed times the forward's embedding: not with a training step")
    if threads is not None:
        torch.set_num_threads(threads)
    overrides = CONFIGS[config] | ({} if layers is None else {"n_layers": layers})
    cfg = NetConfig(trunk_dtype=dtype, **overrides)
    model = build_model(cfg, seed=seed).eval()
    # Register A5's arm reads its own tokenisation of the same position (the network refuses A1's).
    batch = collate([synthetic_state(seed, boards, piece_centric=cfg.piece_centric)])
    small = collate([synthetic_state(seed + 1, 2, piece_centric=cfg.piece_centric)])

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
    row = {
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
    if embed:
        # The embedding alone, as the forward runs it: outside the trunk's autocast (FP32 under
        # either trunk), the network's own scope.
        with torch.no_grad():
            start = time.perf_counter()
            x = model.embed_tokens(batch)
            row["embed_ms"] = round((time.perf_counter() - start) * 1e3, 3)
        assert torch.isfinite(x).all()
    return row


def run(
    config: str,
    boards: int,
    dtype: str,
    timeout: float = 3600.0,
    train: bool = False,
    layers: int | None = None,
    embed: bool = False,
    threads: int | None = None,
) -> dict:
    """``measure`` in a fresh subprocess; a failure (out of memory, timeout) is reported."""
    args = f"{config!r}, {boards}, {dtype!r}, train={train}, layers={layers}, embed={embed}, threads={threads}"
    code = f"import json; from t5d.net.bench import measure; print(json.dumps(measure({args})))"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"config": config, "boards": boards, "dtype": dtype, "error": f"timeout after {timeout:.0f} s"}
    if proc.returncode != 0:
        return {"config": config, "boards": boards, "dtype": dtype, "error": (proc.stderr.strip().splitlines() or ["?"])[-1]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def summarise(rows: list[dict], configs: list[str]) -> list[dict]:
    """One (boards, dtype) block's repetitions, per configuration in ``configs``' order: the median
    and the minimum of its passes' times - the forward's (or training step's) ``seconds`` and, where
    timed, the embedding's ``embed_ms`` (A8's protocol: medians, minima beside them). A failed pass
    is counted (``failed``), never timed."""
    out = []
    for config in configs:
        mine = [r for r in rows if r["config"] == config]
        if not mine:
            continue
        ok = [r for r in mine if "error" not in r]
        line: dict = {"summary": "median", "config": config, "boards": mine[0]["boards"], "dtype": mine[0]["dtype"]}
        line |= {"runs": len(ok), "failed": len(mine) - len(ok)}
        if ok:
            line |= {k: ok[0][k] for k in ("pass", "layers", "threads", "tokens")}
            seconds = [r["seconds"] for r in ok]
            line |= {"seconds_median": round(statistics.median(seconds), 3), "seconds_min": min(seconds)}
            embed = [r["embed_ms"] for r in ok if "embed_ms" in r]
            if embed:
                line |= {"embed_ms_median": round(statistics.median(embed), 3), "embed_ms_min": min(embed)}
        out.append(line)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boards", type=int, nargs="+", default=[60, 240])
    parser.add_argument("--dtype", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    parser.add_argument("--config", nargs="+", default=["n1-dense", "n2-no-table", "n2-default", "n2-all"], choices=sorted(CONFIGS))
    parser.add_argument("--train", action="store_true", help="time a training step (forward and backward) instead")
    parser.add_argument("--layers", type=int, default=None, help="trunk depth (default: the configuration's, 16)")
    parser.add_argument(
        "--embed", action="store_true", help="also time T5DNet.embed_tokens alone, after each forward, in its process (embed_ms)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="passes per configuration in each (boards, dtype) block, the order reversed every other repetition; "
        "more than 1 adds each configuration's medians and minima (A8's protocol: 5)",
    )
    parser.add_argument("--threads", type=int, default=None, help="PyTorch's CPU threads in each pass (default: its own)")
    args = parser.parse_args(argv)
    if args.embed and args.train:
        parser.error("--embed times the forward's embedding: not with --train")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    for boards in args.boards:
        for dtype in args.dtype:
            rows = []
            for r in range(args.repeat):
                order = args.config if r % 2 == 0 else args.config[::-1]
                for config in order:
                    row = run(config, boards, dtype, train=args.train, layers=args.layers, embed=args.embed, threads=args.threads)
                    if args.repeat > 1:
                        row["repetition"] = r + 1
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            if args.repeat > 1:
                for line in summarise(rows, args.config):
                    print(json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
