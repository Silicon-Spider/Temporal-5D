"""Section 15.3's cost figures for the cached path (stage N3): measured on this CPU, beside the
spec's analytic model (``cost.py``).

The rows this stage can measure (``REPORT``): the dense cost per evaluation; the cached cost
per node; the live-set recompute ratio (every head against the boards a move changed - the
refused shortcut of section 14.4, timed as a cost probe only); the supersession recompute
overhead per node; the root-refresh overhead at 200 / 800 / 3200 simulations; one refresh
in node-evaluations (full history - the spec's row is windowed, and the window is not
built); and the KV memory per state. Each is given four ways:

* **analytic** - the spec's formula at the measured token counts (``cost.py``): not a
  measurement;
* **counted** - the FLOPs the implementation issues (``cost.count_flops``). The dense pass
  runs B14's block-sparse key selection on its two move-graph heads and the live pass does
  not, so the dense count is given both ways: block-sparse, and through the chunked path
  that scores every key (``dense_flops`` with ``attn_impl="chunked"``);
* **wall** - CPU wall time of this prototype. It includes what a fused GPU implementation
  would not do per node: gathering every cached board's K/V into one tensor, rotating every
  cached key (RoPE) and building the masks - about 58% of a live pass at 240 boards. Wall
  ratios are the prototype's, not the design's;
* **kernel** - the time spent inside the FLOP-carrying operations (the fused attention
  kernel and the matrix products), from PyTorch's profiler: the closest this CPU comes to
  what the spec's ratios describe.

**Conditions.** One position (batch 1) from ``synthetic_state``; the F1 network (d 256, L
16, 8 heads of 32, FFN 8x) with B9's two move-graph heads (the committed default), the
dense path through B14's block-sparse attention; the cached path under the block-causal
scope with the lineage causal past, ``frozen_features="intrinsic"``, C20's first reading
(the placeholder defaults), the creation reading too for the supersession figure, and an
FP8-frozen-K/V network for the FP8 storage figures; register A7's per-perspective frozen
K/V (spec issue 303): the search root's history and a refresh are built from both colours'
sides, so the store holds two entries per frozen board (``kv_memory``'s ``cache_bytes``; one
state's, ``cache_bytes_per_perspective``, is half) and a refresh is two dense passes (the
refresh rows; ``analytic_both_perspectives`` beside the spec's one-pass formula); no window (C5 is not built); random
weights; PyTorch's default intra-op threads (recorded). Timelines (= live head boards):
the spec's assumption - 8 at 240 boards, 12 at 600 (section 7.1's "~8 live head boards",
section 15.6's table) - and the engine corpus's real games, 24 timelines at 240-399 boards
(the corpus reaches no further, so 24 stands for 600 too).

**A quiet machine.** Every job runs in a fresh subprocess (its peak working set its own).
Before a job the benchmark waits (at most ``--max-wait`` s) until the CPU used by other
processes is under ``--max-other-cores`` and at least ``--min-free-gb`` of memory is
available, then samples both every two seconds while the job runs (``LoadMeter``: system
CPU time less the job's and its own). A job is ``contended`` when other processes used more
than that on average, or more than twice it in a tenth of the samples, or memory fell under
4 GB; ``unstable`` when its repeated timings spread more than ``SPREAD_LIMIT``. Such a job
runs again (``--retries``); one still flagged keeps its byte counts in the summary, its
times are left out.

Run (from ``python/``): ``py -m t5d.net.cache_bench --out DIR`` (``--quick`` for a smoke run
at 20 boards). Writes one JSON file per job and ``summary.json`` into ``DIR`` - keep it out
of the repository (``docs/MEASUREMENTS.md`` takes the figures, with their conditions);
``--skip-existing`` reuses clean job files, ``--summarise`` only rebuilds the summary.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time

from .bench import _peak_bytes

#: The spec's estimates these figures replace (section 15.3; sections 7.1, 7.3, 7.6, 7.9, 10.3).
REPORT = {
    "dense_eval": "Dense cost per eval at 240 boards: 4.78 TF -> ~337 evals/s (FA4-class)",
    "cached_node": "Cached cost per node: 0.16 TF (29x on paper); 15-20x realistic, ~12-18x after the supersession recompute",
    "live_set": "Live-set recompute: 3.7-5.6x the changed-boards-only cost",
    "supersession": "Supersession recompute: +8-24% per node",
    "root_refresh": "Root refresh overhead (800 sims): +3.6% (240 boards) / +6.1% (600) full history; +1.3% windowed",
    "one_refresh": "One refresh: ~10 node-evaluations under cached + window (windowed: not measured here)",
    "kv_memory": "KV memory per state: ~260 MB (240 boards) / ~649 MB (600) full history; FP8 halves",
}
SIMS = (200, 800, 3200)
SPREAD_LIMIT = 0.15
#: The operations that carry the FLOPs (``kernel_seconds``).
KERNEL_OPS = frozenset({"aten::_scaled_dot_product_flash_attention_for_cpu", "aten::addmm", "aten::mm", "aten::bmm", "aten::baddbmm"})
#: Timelines per board count: the spec's assumption, then the engine corpus's real games.
TIMELINES = {240: (8, 24), 600: (12, 24), 20: (3, 6)}


# --- timing --------------------------------------------------------------------------------------


def _median_time(fn, repeats: int) -> tuple[float, list[float]]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), [round(t, 4) for t in times]


def spread(times: list[float]) -> float:
    """(max - min) / median of repeated timings (0 for one)."""
    return 0.0 if len(times) < 2 else (max(times) - min(times)) / statistics.median(times)


def kernel_seconds(fn) -> float:
    """Seconds spent inside the FLOP-carrying operations (``KERNEL_OPS``) during ``fn()``."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    return sum(e.self_cpu_time_total for e in prof.key_averages() if e.key in KERNEL_OPS) / 1e6


# --- machine load ---------------------------------------------------------------------------------


class LoadMeter:
    """System CPU time, a process's CPU time and the available memory, sampled so that the CPU
    other processes use - ``other_cores`` between two samples - can be told apart from a
    job's own (Windows: ``GetSystemTimes``, ``GetProcessTimes``, ``GlobalMemoryStatusEx``;
    Linux: ``/proc``)."""

    def __init__(self) -> None:
        self.windows = sys.platform == "win32"
        self.ncpu = os.cpu_count() or 1

    def system_busy(self) -> float:
        if self.windows:
            idle, kernel, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong())
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            return (kernel.value + user.value - idle.value) / 1e7  # kernel time includes idle
        with open("/proc/stat", encoding="ascii") as f:
            fields = [int(x) for x in f.readline().split()[1:]]
        return (sum(fields) - fields[3] - fields[4]) / os.sysconf("SC_CLK_TCK")

    def process_cpu(self, proc: subprocess.Popen | None) -> float:
        """CPU seconds of ``proc`` (None: this process)."""
        if self.windows:
            current = ctypes.windll.kernel32.GetCurrentProcess
            current.restype = ctypes.c_void_p  # a pseudo-handle, not a C int
            handle = current() if proc is None else int(proc._handle)
            times = [ctypes.c_ulonglong() for _ in range(4)]
            func = ctypes.windll.kernel32.GetProcessTimes
            func.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4
            func(ctypes.c_void_p(handle), *[ctypes.byref(t) for t in times])
            return (times[2].value + times[3].value) / 1e7
        pid = "self" if proc is None else str(proc.pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
        except OSError:
            return 0.0

    def available_gb(self) -> float:
        if self.windows:

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong)
                    for n in ("total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_ext")
                ]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.avail / 2**30
        with open("/proc/meminfo", encoding="ascii") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return info.get("MemAvailable", 0) / 2**20

    def sample(self, proc: subprocess.Popen | None = None) -> tuple[float, float, float]:
        """(time, CPU seconds of every process but ``proc`` and this one, available GB)."""
        own = self.process_cpu(None) + (0.0 if proc is None else self.process_cpu(proc))
        return time.perf_counter(), self.system_busy() - own, self.available_gb()

    def quiet(self, seconds: float = 6.0) -> tuple[float, float]:
        """(cores other processes used over ``seconds``, available GB)."""
        t0, b0, _ = self.sample()
        time.sleep(seconds)
        t1, b1, gb = self.sample()
        return (b1 - b0) / (t1 - t0), gb


def running(names: tuple[str, ...]) -> list[str]:
    """Which of the named executables are running (``tasklist`` on Windows, ``ps`` elsewhere)."""
    if not names:
        return []
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=False).stdout
        images = {line.split(",")[0].strip('"').lower() for line in out.splitlines() if line}
    else:
        out = subprocess.run(["ps", "-A", "-o", "comm="], capture_output=True, text=True, check=False).stdout
        images = {line.strip().lower() for line in out.splitlines()}
    return sorted(n for n in names if n.lower() in images)


# --- jobs (each runs in its own subprocess) ---------------------------------------------------------------


def _setup(boards: int, timelines: int | None, dtype: str, scope: str, seed: int, **flags):
    import torch

    from .config import NetConfig
    from .model import build_model
    from .synthetic import synthetic_state

    torch.set_grad_enabled(False)
    model = build_model(NetConfig(trunk_dtype=dtype, attn_scope=scope, **flags), seed=seed).eval()
    return model, synthetic_state(seed, boards, n_timelines=timelines)


def _conditions(model, state) -> dict:
    import torch

    return {
        "boards": state.n_boards,
        "tokens": state.n_tokens,
        "tokens_per_board": state.layout.tokens_per_board,
        "global_regs": state.global_regs,
        "timelines": len(set(state.timeline.tolist())),
        "heads": int(state.is_head.sum()),
        "frozen_boards": int((~state.is_head).sum()),
        "dtype": model.cfg.trunk_dtype,
        "attn_impl": model.cfg.attn_impl,
        "move_graph_heads": model.cfg.move_graph_heads,
        "layers": model.cfg.n_layers,
        "frozen_kv_format": model.cfg.frozen_kv_format,
        "threads": torch.get_num_threads(),
        "batch": 1,
        "window": "none",
    }


def job_dense(boards: int, timelines: int | None, dtype: str, repeats: int, seed: int = 0) -> dict:
    """The dense network (C1): one forward over the whole position."""
    from .state import collate
    from .synthetic import synthetic_state

    model, state = _setup(boards, timelines, dtype, "dense", seed)
    batch = collate([state])
    model(collate([synthetic_state(seed + 1, 2)]))  # warm-up: kernels, thread pool
    before = _peak_bytes()
    seconds, times = _median_time(lambda: model(batch), repeats)
    peak = _peak_bytes()
    return {
        "job": "dense",
        **_conditions(model, state),
        "seconds": seconds,
        "times": times,
        "kernel_s": kernel_seconds(lambda: model(batch)),
        "spreads": {"dense": spread(times)},
        "peak_gb": peak / 2**30,
        "pass_gb": (peak - before) / 2**30,
    }


def job_dense_flops(boards: int, timelines: int | None, dtype: str, seed: int = 0, attn_impl: str = "block_sparse") -> dict:
    """FLOPs of one dense forward: block-sparse (B14, the default) or chunked (every key scored)."""
    from .cost import count_flops
    from .state import collate

    model, state = _setup(boards, timelines, dtype, "dense", seed, attn_impl=attn_impl)
    _, flops = count_flops(lambda: model(collate([state])))
    return {"job": "dense_flops", **_conditions(model, state), "flops": flops}


def _children(state, seed: int):
    """A physical child (one board superseded, one created) and, where the root allows them, a
    non-branching jump child (two superseded, two created) and a branching one (one
    superseded, two created)."""
    from .tree import Multiverse

    root = Multiverse.from_state(state, seed=seed)
    heads = root.heads()
    ps = sorted(heads, key=lambda p: heads[p][1])
    out = {"physical": (root, ("physical", ps[0]))}
    for p in sorted(heads, key=lambda p: -heads[p][1]):
        sh = heads[p][1]
        others = [k for k in heads.values() if k[0] != p and k[1] % 2 == sh % 2 and k[1] <= sh]
        past = [k for k in sorted(root.boards) if not root.is_head(k) and k[1] % 2 == sh % 2 and k[1] <= sh]
        if others and "jump" not in out:
            out["jump"] = (root, ("jump", p, others[0]))
        if past and "branch" not in out:
            out["branch"] = (root, ("jump", p, past[len(past) // 2]))
    return out


def job_cached(
    boards: int,
    timelines: int | None,
    dtype: str,
    repeats: int,
    seed: int = 0,
    causal_past: str = "lineage",
    frozen_kv_format: str = "match",
) -> dict:
    """The cached path: the history build, a node (the live pass), the changed-only probe,
    the supersession recompute of a physical move and of two jumps, a root refresh, and the
    store's bytes - with repeats and kernel times."""
    from .cache import CacheConfig, CachedEvaluator, KVStore
    from .state import collate
    from .synthetic import synthetic_state

    model, state = _setup(boards, timelines, dtype, "block_causal", seed, causal_past=causal_past, frozen_kv_format=frozen_kv_format)
    warm = CachedEvaluator(model)
    small = synthetic_state(seed + 1, 3)
    warm.evaluate(small, warm.root_epoch(small))
    out = {"job": "cached", **_conditions(model, state), "causal_past": causal_past}
    ev = CachedEvaluator(model)
    start = time.perf_counter()
    epoch = ev.root_epoch(state)
    out["history_build_s"] = time.perf_counter() - start
    out["kv_bytes"] = ev.store.kv_bytes()
    out["out_bytes"] = ev.store.out_bytes()
    out["entries"] = len(ev.store.entries)
    work = ev.prepare(state, epoch)
    assert not ev.missing(work)
    out["node_s"], out["node_times"] = _median_time(lambda: ev.evaluate(state, epoch), repeats)
    out["live_pass_s"], out["live_pass_times"] = _median_time(lambda: ev.live_pass(work), repeats)
    out["live_pass_kernel_s"] = kernel_seconds(lambda: ev.live_pass(work))
    out["prepare_s"], _ = _median_time(lambda: ev.prepare(state, epoch), repeats)
    spreads = {"node": spread(out["node_times"]), "live_pass": spread(out["live_pass_times"])}
    root_entries = dict(ev.store.entries)
    for name, (universe, move) in _children(state, seed).items():
        # Each child against the root's store alone (a sibling's supersession would be shared).
        cev = CachedEvaluator(model, store=KVStore(dict(root_entries)))
        child = universe.apply(move).state()
        cwork = cev.prepare(child, epoch)
        todo = cev.missing(cwork)
        per_board = []
        rounds = repeats if len(todo) <= 4 else 1  # a creation cascade is timed once
        for r in range(rounds):
            if r:
                for b in todo:
                    del cev.store.entries[cwork.plan.key(epoch, b)]
            times = []
            for b in todo:
                t0 = time.perf_counter()
                cev.compute_board(cwork, b)
                times.append(time.perf_counter() - t0)
            per_board.append(times)
        sup_kernel = 0.0
        if todo and len(todo) <= 4:
            for b in todo:
                del cev.store.entries[cwork.plan.key(epoch, b)]
            sup_kernel = kernel_seconds(lambda cev=cev, cwork=cwork, todo=todo: [cev.compute_board(cwork, b) for b in todo])
        totals = [sum(t) for t in per_board]
        changed = [i for i in range(child.n_boards) if child.creation[i] >= universe.next_creation]
        probe = lambda cev=cev, cwork=cwork, changed=changed: cev.live_pass(cwork, query_boards=changed)  # noqa: E731
        live = lambda cev=cev, cwork=cwork: cev.live_pass(cwork)  # noqa: E731
        probe_s, probe_times = _median_time(probe, repeats)
        live_s, live_times = _median_time(live, repeats)
        tpb = child.layout.tokens_per_board
        out[f"child_{name}"] = {
            "move": [move[0], *[list(x) if isinstance(x, tuple) else x for x in move[1:]]],
            "superseded": len(todo),
            "supersession_s": statistics.median(totals) if totals else 0.0,
            "supersession_times": [round(t, 4) for t in totals],
            "supersession_kernel_s": sup_kernel,
            "supersession_key_tokens": [(len(cwork.plan.deps[b]) + 1) * tpb for b in todo],
            "changed_boards": len(changed),
            "changed_only_live_s": probe_s,
            "changed_only_kernel_s": kernel_seconds(probe),
            "live_pass_s": live_s,
            "live_pass_kernel_s": kernel_seconds(live),
            "live_tokens": int(child.is_head.sum()) * tpb + child.global_regs,
            "changed_tokens": len(changed) * tpb + child.global_regs,
            "tokens": child.n_tokens,
        }
        spreads[f"{name}_probe"] = spread(probe_times)
        spreads[f"{name}_live"] = spread(live_times)
        if len(totals) > 1:
            spreads[f"{name}_supersession"] = spread(totals)
    out["peak_gb_cached"] = _peak_bytes() / 2**30
    # A root refresh (C3): the dense pass with K/V capture - one per perspective (A7) - and the store's entries, each in a
    # fresh evaluator so that no repeat holds another's epoch.
    refresh_times = []
    for _ in range(repeats):
        refresher = CachedEvaluator(model, CacheConfig(root_refresh=True))
        t0 = time.perf_counter()
        refresher.root_epoch(state)
        refresh_times.append(time.perf_counter() - t0)
        out["refresh_kv_bytes"] = refresher.store.kv_bytes()
        del refresher
    out["refresh_s"] = statistics.median(refresh_times)
    out["refresh_times"] = [round(t, 3) for t in refresh_times]
    out["refresh_kernel_s"] = kernel_seconds(lambda: CachedEvaluator(model, CacheConfig(root_refresh=True)).root_epoch(state))
    spreads["refresh"] = spread(refresh_times)
    plain = collate([state])
    out["dense_same_process_s"], _ = _median_time(lambda: model(plain, scope="dense"), 1)
    out["peak_gb"] = _peak_bytes() / 2**30
    out["spreads"] = spreads
    return out


def job_cached_flops(boards: int, timelines: int | None, dtype: str, seed: int = 0, causal_past: str = "lineage") -> dict:
    """The FLOPs the cached path issues: the live pass, the changed-only probe, each
    supersession, the refresh."""
    from .cache import CacheConfig, CachedEvaluator, KVStore
    from .cost import count_flops

    model, state = _setup(boards, timelines, dtype, "block_causal", seed, causal_past=causal_past)
    ev = CachedEvaluator(model)
    epoch = ev.root_epoch(state)
    work = ev.prepare(state, epoch)
    out = {"job": "cached_flops", **_conditions(model, state), "causal_past": causal_past}
    _, out["live_pass_flops"] = count_flops(lambda: ev.live_pass(work))
    root_entries = dict(ev.store.entries)
    for name, (universe, move) in _children(state, seed).items():
        cev = CachedEvaluator(model, store=KVStore(dict(root_entries)))
        child = universe.apply(move).state()
        cwork = cev.prepare(child, epoch)
        flops = []
        for b in cev.missing(cwork):
            _, f = count_flops(lambda b=b, cev=cev, cwork=cwork: cev.compute_board(cwork, b))
            flops.append(f)
        changed = [i for i in range(child.n_boards) if child.creation[i] >= universe.next_creation]
        _, probe = count_flops(lambda cev=cev, cwork=cwork, changed=changed: cev.live_pass(cwork, query_boards=changed))
        _, live = count_flops(lambda cev=cev, cwork=cwork: cev.live_pass(cwork))
        out[f"child_{name}"] = {"supersession_flops": flops, "changed_only_flops": probe, "live_pass_flops": live}
    refresher = CachedEvaluator(model, CacheConfig(root_refresh=True))
    _, out["refresh_flops"] = count_flops(lambda: refresher.root_epoch(state))
    return out


# --- running jobs -------------------------------------------------------------------------------------


def run(
    job: str,
    timeout: float = 7200.0,
    idle: tuple[str, ...] = (),
    max_wait: float = 3600.0,
    max_other_cores: float = 1.0,
    min_free_gb: float = 16.0,
    **kwargs,
) -> dict:
    """A job in a fresh subprocess, on a quiet machine (see the module docstring): it waits at
    most ``max_wait`` s for one, samples the load while the job runs, and returns the job's
    result with ``load`` and ``contended`` (``unstable`` comes from the job's own spreads)."""
    code = f"import json; from t5d.net.cache_bench import job_{job}; print(json.dumps(job_{job}(**{kwargs!r})))"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meter = LoadMeter()
    waited = time.perf_counter()
    while time.perf_counter() - waited < max_wait:
        other, gb = meter.quiet()
        if other < max_other_cores and gb >= min_free_gb and not running(idle):
            break
    waited = time.perf_counter() - waited
    start = time.perf_counter()
    seen: set[str] = set(running(idle))
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = [meter.sample(proc)]
    other_cores: list[float] = []
    min_gb = samples[0][2]
    try:
        while True:  # communicate() drains both pipes while it waits: a large result cannot block the job
            try:
                stdout, stderr = proc.communicate(timeout=2)
                break
            except subprocess.TimeoutExpired:
                samples.append(meter.sample(proc))
                (t0, b0, _), (t1, b1, gb) = samples[-2], samples[-1]
                other_cores.append(max(0.0, (b1 - b0) / (t1 - t0)))
                min_gb = min(min_gb, gb)
                seen.update(running(idle))
                if time.perf_counter() - start > timeout:
                    proc.kill()
                    proc.communicate()
                    return {"job": job, **kwargs, "error": f"timeout after {timeout:.0f} s"}
    finally:
        if proc.poll() is None:
            proc.kill()
    mean_other = statistics.fmean(other_cores) if other_cores else 0.0
    busy_share = sum(o > 2 * max_other_cores for o in other_cores) / max(1, len(other_cores))
    contended = bool(seen) or mean_other > max_other_cores or busy_share > 0.1 or min_gb < 4.0
    load = {
        "mean_other_cores": round(mean_other, 2),
        "max_other_cores": round(max(other_cores, default=0.0), 2),
        "busy_share": round(busy_share, 3),
        "min_available_gb": round(min_gb, 1),
        "processes_seen": sorted(seen),
        "waited_s": round(waited, 1),
    }
    if proc.returncode != 0:
        return {"job": job, **kwargs, "load": load, "contended": contended, "error": (stderr.strip().splitlines() or ["?"])[-1]}
    result = json.loads(stdout.strip().splitlines()[-1])
    result["wall_s"] = time.perf_counter() - start
    result["load"] = load
    result["contended"] = contended
    result["unstable"] = max(result.get("spreads", {}).values(), default=0.0) > SPREAD_LIMIT
    return result


def _usable(r: dict | None) -> bool:
    """A result whose times may be reported (bytes are always usable)."""
    return r is not None and "error" not in r and not r.get("contended") and not r.get("unstable")


def summarise(results: list[dict]) -> dict:
    """Section 15.3's rows from the jobs: analytic, counted, wall and kernel figures, ratios."""
    from .config import NetConfig
    from .cost import dense_flops, kv_bytes, pass_flops

    cfg = NetConfig()
    by = {}
    for r in results:
        if "error" in r:
            continue
        key = (
            r["job"],
            r["boards"],
            r["timelines"],
            r["dtype"],
            r.get("frozen_kv_format", "match"),
            r.get("causal_past", ""),
            r.get("attn_impl", ""),
        )
        by[key] = r
    rows = []
    for (job, boards, timelines, dtype, fmt, reading, _), r in sorted(by.items(), key=lambda kv: [str(x) for x in kv[0]]):
        if job != "cached" or fmt != "match" or reading != "lineage":
            continue

        def find(name, impl="block_sparse", fmt="match", reading="", dt=dtype, timelines=timelines, boards=boards):
            for key, value in by.items():
                if key[:6] == (name, boards, timelines, dt, fmt, reading) and (name != "dense_flops" or key[6] == impl):
                    return value
            return None

        dense, flops_sparse = find("dense"), find("dense_flops", dt="fp32")
        flops_chunked = find("dense_flops", impl="chunked", dt="fp32")
        cflops = find("cached_flops", reading="lineage", dt="fp32")
        fp8 = find("cached", fmt="fp8", reading="lineage")
        creation = find("cached", reading="creation")
        t, tpb, g = r["tokens"], r["tokens_per_board"], r["global_regs"]
        live_t = r["heads"] * tpb + g
        ok, dense_ok = _usable(r), _usable(dense)
        a_dense, a_node = dense_flops(t, cfg)["total"], pass_flops(live_t, t, cfg)["total"]
        row = {
            "boards": boards,
            "timelines": timelines,
            "timeline_source": "spec" if TIMELINES.get(boards, (None,))[0] == timelines else "corpus",
            "heads": r["heads"],
            "dtype": dtype,
            "threads": r["threads"],
            "tokens": t,
            "live_tokens": live_t,
            "timings_usable": ok and dense_ok,
            "load": {"cached": r.get("load"), "dense": dense and dense.get("load")},
        }
        row["dense"] = {
            "analytic_tf": a_dense / 1e12,
            "counted_tf_block_sparse": flops_sparse and flops_sparse["flops"] / 1e12,
            "counted_tf_every_key": flops_chunked and flops_chunked["flops"] / 1e12,
            "wall_s": dense["seconds"] if dense_ok else None,
            "kernel_s": dense["kernel_s"] if dense_ok else None,
        }
        row["node"] = {
            "analytic_tf": a_node / 1e12,
            "counted_tf": cflops and cflops["live_pass_flops"] / 1e12,
            "wall_s": r["live_pass_s"] if ok else None,
            "evaluate_wall_s": r["node_s"] if ok else None,
            "kernel_s": r["live_pass_kernel_s"] if ok else None,
        }
        both = ok and dense_ok
        row["cached_vs_dense"] = {
            "analytic": a_dense / a_node,
            "counted_vs_block_sparse": (flops_sparse and cflops) and flops_sparse["flops"] / cflops["live_pass_flops"],
            "counted_vs_every_key": (flops_chunked and cflops) and flops_chunked["flops"] / cflops["live_pass_flops"],
            "wall_cpu_prototype": dense["seconds"] / r["live_pass_s"] if both else None,
            "kernel": dense["kernel_s"] / r["live_pass_kernel_s"] if both else None,
        }
        children = {}
        for name in ("physical", "jump", "branch"):
            c = r.get(f"child_{name}")
            if not c:
                continue
            cf = cflops and cflops.get(f"child_{name}")
            a_live = pass_flops(c["live_tokens"], c["tokens"], cfg)["total"]
            a_changed = pass_flops(c["changed_tokens"], c["tokens"], cfg)["total"]
            a_sup = sum(pass_flops(tpb, k, cfg)["total"] for k in c["supersession_key_tokens"])
            entry = {
                "superseded": c["superseded"],
                "changed_boards": c["changed_boards"],
                "live_set_ratio": {
                    "analytic": a_live / a_changed,
                    "counted": cf and cf["live_pass_flops"] / cf["changed_only_flops"],
                    "wall_cpu_prototype": c["live_pass_s"] / c["changed_only_live_s"] if ok else None,
                    "kernel": c["live_pass_kernel_s"] / c["changed_only_kernel_s"] if ok else None,
                },
                "supersession_overhead": {
                    "analytic": a_sup / a_live,
                    "counted": cf and sum(cf["supersession_flops"]) / cf["live_pass_flops"],
                    "wall_cpu_prototype": c["supersession_s"] / c["live_pass_s"] if ok else None,
                    "kernel": c["supersession_kernel_s"] / c["live_pass_kernel_s"] if ok and c["supersession_kernel_s"] else None,
                },
            }
            if creation and creation.get(f"child_{name}"):
                cc = creation[f"child_{name}"]
                a_sup_c = sum(pass_flops(tpb, k, cfg)["total"] for k in cc["supersession_key_tokens"])
                entry["supersession_creation"] = {
                    "superseded": cc["superseded"],
                    "analytic": a_sup_c / a_live,
                    "wall_cpu_prototype": cc["supersession_s"] / cc["live_pass_s"] if _usable(creation) else None,
                }
            children[name] = entry
        row["children"] = children
        refresh_ok = ok
        row["refresh"] = {
            "wall_s": r["refresh_s"] if refresh_ok else None,
            "kernel_s": r["refresh_kernel_s"] if refresh_ok else None,
            "in_node_evals_full_history": {
                "analytic": a_dense / a_node,
                "analytic_both_perspectives": 2 * a_dense / a_node,
                "wall_cpu_prototype": r["refresh_s"] / r["live_pass_s"] if refresh_ok else None,
                "kernel": r["refresh_kernel_s"] / r["live_pass_kernel_s"] if refresh_ok else None,
            },
            "overhead": {
                str(n): {
                    "analytic": a_dense / (n * a_node),
                    "analytic_both_perspectives": 2 * a_dense / (n * a_node),
                    "wall_cpu_prototype": r["refresh_s"] / (n * r["live_pass_s"]) if refresh_ok else None,
                    "kernel": r["refresh_kernel_s"] / (n * r["live_pass_kernel_s"]) if refresh_ok else None,
                }
                for n in SIMS
            },
        }
        row["kv_memory"] = {
            "cache_bytes": r["kv_bytes"],
            "cache_bytes_per_perspective": r["kv_bytes"] / 2,
            "cache_bytes_fp8_network": fp8 and fp8["kv_bytes"],
            "outputs_bytes": r["out_bytes"],
            "analytic_all_tokens_bf16": kv_bytes(t, cfg, 2),
            "analytic_frozen_tokens": kv_bytes(r["frozen_boards"] * tpb, cfg, 2 if dtype == "bf16" else 4),
            "per_superseded_board": kv_bytes(tpb, cfg, 2 if dtype == "bf16" else 4),
            "peak_gb_cached": r["peak_gb_cached"],
            "peak_gb_dense": dense and dense["peak_gb"],
        }
        rows.append(row)
    return {"report": REPORT, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory for the JSON results (outside the repository)")
    parser.add_argument("--boards", type=int, nargs="+", default=[240, 600])
    parser.add_argument("--dtype", nargs="+", default=["fp32", "bf16"], choices=["fp32", "bf16"])
    parser.add_argument("--quick", action="store_true", help="a smoke run at 20 boards, one repeat")
    parser.add_argument("--idle-processes", nargs="*", default=[], help="executables that must not run alongside a job")
    parser.add_argument("--max-other-cores", type=float, default=1.0, help="CPU other processes may use during a job")
    parser.add_argument("--min-free-gb", type=float, default=16.0, help="memory available before a job starts")
    parser.add_argument("--max-wait", type=float, default=3600.0, help="seconds to wait for a quiet machine per job")
    parser.add_argument("--retries", type=int, default=3, help="runs allowed for a contended or unstable job")
    parser.add_argument("--summarise", action="store_true", help="only rebuild summary.json from the job files in --out")
    parser.add_argument("--skip-existing", action="store_true", help="reuse a job's file in --out when it holds a usable result")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    def load_all() -> list[dict]:
        loaded = []
        for name in sorted(os.listdir(args.out)):
            if name.endswith(".json") and name != "summary.json":
                with open(os.path.join(args.out, name), encoding="utf-8") as f:
                    loaded.append(json.load(f))
        return loaded

    def write_summary(results: list[dict]) -> None:
        summary = summarise(results)
        summary["machine"] = {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0]}
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary, indent=1))

    if args.summarise:
        write_summary(load_all())
        return 0
    guard = dict(
        idle=tuple(args.idle_processes), max_other_cores=args.max_other_cores, min_free_gb=args.min_free_gb, max_wait=args.max_wait
    )

    def name_of(job: str, kwargs: dict) -> str:
        parts = [job, kwargs["boards"], kwargs["timelines"], kwargs["dtype"]]
        parts += [kwargs.get("frozen_kv_format", ""), kwargs.get("causal_past", ""), kwargs.get("attn_impl", "")]
        return "-".join(str(x) for x in parts if x != "")

    def clean(job: str, **kwargs) -> None:
        path = os.path.join(args.out, name_of(job, kwargs) + ".json")
        if args.skip_existing and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            if job.endswith("flops") and "error" not in old or _usable(old):
                return
        result: dict = {}
        for _ in range(max(1, args.retries)):
            result = run(job, **guard, **kwargs)
            if "error" not in result and not result.get("contended") and not result.get("unstable"):
                break
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (dict, list))}), flush=True)

    for boards in [20] if args.quick else args.boards:
        repeats = 1 if args.quick else (3 if boards <= 240 else 2)
        for timelines in TIMELINES.get(boards, (None,)):
            spec_count = timelines == TIMELINES.get(boards, (None,))[0]
            for dtype in args.dtype:
                if boards > 240 and dtype == "bf16" and not spec_count:
                    continue  # BF16 at 600 boards: the spec's timeline count only (run time)
                common = dict(boards=boards, timelines=timelines, dtype=dtype)
                clean("dense", repeats=2 if boards > 240 else repeats, **common)
                clean("cached", repeats=repeats, **common)
                if dtype == "bf16" and spec_count:
                    clean("cached", repeats=repeats, frozen_kv_format="fp8", causal_past="lineage", **common)
                if dtype == "fp32":
                    clean("dense_flops", **common)
                    clean("dense_flops", attn_impl="chunked", **common)
                    clean("cached_flops", **common)
                    if boards <= 240 or spec_count:
                        clean("cached", repeats=repeats, causal_past="creation", **common)
    write_summary(load_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
