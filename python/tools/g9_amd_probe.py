"""G9's bounded AMD attempt (register G9, M19; spec section 10.2): the project's network on a
ROCm GPU through AMD's PyTorch for Windows, every figure AMD-labelled and indicative.

Run from ``python/`` with the interpreter of a separate ROCm environment (never the project's
CPU one), one stage at a time, each printing one JSON object::

    <rocm-python> tools/g9_amd_probe.py env
    <rocm-python> tools/g9_amd_probe.py device
    <rocm-python> tools/g9_amd_probe.py numerics --impl block_sparse sdpa flex
    <rocm-python> tools/g9_amd_probe.py compile
    <rocm-python> tools/g9_amd_probe.py throughput --boards 44 112 240 --batch 1 4 8 16
    <rocm-python> tools/g9_amd_probe.py search --playouts 400
    <rocm-python> tools/g9_amd_probe.py --pad8 throughput ...   # the padded-width arm

Stages: ``env`` (versions, the device and its architecture); ``device`` (a matmul against the
CPU, BF16, the FP8 E4M3 dtype and ``torch._scaled_mm``); ``numerics`` (the F1 v1 network's
forward on the GPU against the same weights and inputs on the CPU - synthetic states and a
real engine position - in FP32 and BF16, per attention implementation); ``compile``
(``torch.compile`` of the network, of ``attn_impl="flex"``, and of a toy ``flex_attention``);
``throughput`` (the dense evaluator's evaluations per second, synchronised medians); ``search``
(one search through the C step API, ``run_search`` with ``NetworkEvaluator`` on the GPU).
A stage that fails records the exception's type and text instead of a figure: failure is a
valid outcome of a bounded attempt. Nothing here changes the network's code or picks a
device for it beyond ``.to(device)``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t5d.net import NetConfig, build_model  # noqa: E402
from t5d.net.state import collate  # noqa: E402
from t5d.net.synthetic import synthetic_state  # noqa: E402

DEV = "cuda"  # HIP presents as CUDA in PyTorch's ROCm builds


def install_pad8() -> None:
    """The ``--pad8`` arm, in this process only (the network's code is unchanged): every
    ``scaled_dot_product_attention`` call of ``attention.py`` gets its head width zero-padded
    to a multiple of 8 and its output cut back. Exact in arithmetic (a zero feature adds
    nothing to a score; the scale is passed explicitly). Why: the kernel's widened features
    (``_KernelKeys``: 45 and 33 wide in the v1 network) with a mask find no fused kernel in
    AMD's ROCm build, which then runs the math backend (measured, MEASUREMENTS' G9 section)."""
    import torch.nn.functional as F

    from t5d.net import attention

    sdpa = F.scaled_dot_product_attention

    def padded(q, k, v, attn_mask=None, scale=None, **kw):
        d, dv = q.shape[-1], v.shape[-1]
        w = -(-max(d, dv) // 8) * 8
        if w == d == dv:
            return sdpa(q, k, v, attn_mask=attn_mask, scale=scale, **kw)
        scale = d**-0.5 if scale is None else scale
        pad = torch.nn.functional.pad
        o = sdpa(pad(q, (0, w - d)), pad(k, (0, w - k.shape[-1])), pad(v, (0, w - dv)), attn_mask=attn_mask, scale=scale, **kw)
        return o[..., :dv]

    class _F:
        def __getattr__(self, name):
            return padded if name == "scaled_dot_product_attention" else getattr(F, name)

    attention.F = _F()


def _err(e: BaseException) -> dict:
    lines = traceback.format_exception_only(type(e), e)
    return {"error": type(e).__name__, "text": "".join(lines).strip()[-2000:]}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stage_env(_args) -> dict:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
    }
    try:
        import triton  # noqa: F401

        out["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        out["triton"] = _err(e)
    if torch.cuda.is_available():
        out["device_count"] = torch.cuda.device_count()
        p = torch.cuda.get_device_properties(0)
        out["device"] = {
            "name": torch.cuda.get_device_name(0),
            "gcnArchName": getattr(p, "gcnArchName", None),
            "total_memory_gb": round(p.total_memory / 2**30, 2),
            "multi_processor_count": p.multi_processor_count,
        }
        out["bf16_supported"] = torch.cuda.is_bf16_supported()
    return out


def stage_device(_args) -> dict:
    out: dict = {}
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2048, 2048, generator=g)
    b = torch.randn(2048, 2048, generator=g)
    ref = a.double() @ b.double()
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        try:
            c = (a.to(DEV, dt) @ b.to(DEV, dt)).double().cpu()
            ccpu = (a.to(dt) @ b.to(dt)).double()
            scale = ref.abs().max().item()
            out[f"matmul_{name}"] = {
                "gpu_vs_fp64_rel": (c - ref).abs().max().item() / scale,
                "cpu_vs_fp64_rel": (ccpu - ref).abs().max().item() / scale,
            }
        except Exception as e:  # noqa: BLE001
            out[f"matmul_{name}"] = _err(e)
    # FP8 E4M3: the dtype, a cast round trip, and the scaled GEMM (G2's emulation needs only the dtype).
    try:
        x = torch.randn(64, 64, device=DEV)
        q = x.clamp(-448, 448).to(torch.float8_e4m3fn)
        back = q.float().cpu()
        cpu_back = x.cpu().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        out["fp8_e4m3_cast"] = {"ok": True, "gpu_equals_cpu_cast": bool(torch.equal(back, cpu_back))}
    except Exception as e:  # noqa: BLE001
        out["fp8_e4m3_cast"] = _err(e)
    try:
        m, k, n = 256, 256, 256
        xa = torch.randn(m, k, device=DEV).to(torch.float8_e4m3fn)
        xb = torch.randn(n, k, device=DEV).to(torch.float8_e4m3fn).t()
        one = torch.tensor(1.0, device=DEV)
        r = torch._scaled_mm(xa, xb, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        exact = (xa.float() @ xb.float()).to(torch.bfloat16)
        out["fp8_scaled_mm"] = {"ok": True, "max_abs_vs_dequantised": (r.float() - exact.float()).abs().max().item()}
    except Exception as e:  # noqa: BLE001
        out["fp8_scaled_mm"] = _err(e)
    return out


# ---- the network ---------------------------------------------------------------------------


def _engine_states(cfg: NetConfig, n_states: int = 3, seed: int = 7, max_actions: int = 40) -> list:
    """Real engine positions: seeded pseudo-legal random play from the standard start (never a
    royal capture), the position at a few turn starts. Needs the engine library."""
    from t5d.engine import Game, relations_for
    from t5d.engine.evaluator import state_of
    from t5d.engine.play import ROYAL_IDS

    rng = np.random.default_rng(seed)
    game = Game.variant("Standard")
    game.set_legality(False)
    states = []
    try:
        for _ in range(max_actions * 4):
            info = game.info()
            if info.moves_made == 0 and info.boards >= 2 and rng.random() < 0.35:
                states.append(state_of(game.position(relations=relations_for(cfg))))
                if len(states) >= n_states:
                    break
            if info.obligations_met and (not info.may_move or rng.random() < 0.5):
                game.submit()
                continue
            if not info.may_move:
                break
            x = game.expand(square_features="none")
            ok = np.flatnonzero(~np.isin(x.moves["captured"], ROYAL_IDS))
            if ok.size == 0:
                if not info.obligations_met:
                    break
                game.submit()
                continue
            game.apply(int(rng.choice(ok)))
    finally:
        game.close()
    return states


def _compare(cpu_out, gpu_out) -> dict:
    res = {}
    mv = cpu_out.move_valid.cpu()
    sv = cpu_out.submit_valid.cpu()
    pairs = {
        "value_logits": (cpu_out.value_logits, gpu_out.value_logits),
        "wdl_logits": (cpu_out.wdl_logits, gpu_out.wdl_logits),
        "policy_logits": (cpu_out.policy_logits.cpu()[mv], gpu_out.policy_logits.cpu()[mv]),
        "submit_logits": (cpu_out.submit_logits.cpu()[sv], gpu_out.submit_logits.cpu()[sv]),
    }
    for k, (ref, got) in pairs.items():
        a = ref.detach().double().cpu()
        b = got.detach().double().cpu()
        if a.numel() == 0:
            continue
        d = (a - b).abs().max().item()
        res[k] = {"max_abs": d, "max_rel": d / max(a.abs().max().item(), 1e-6)}
    return res


def _net(dtype: str, impl: str, layers: int | None = None, seed: int = 0, **flags):
    kw = {"trunk_dtype": dtype, "attn_impl": impl, **flags}
    if layers is not None:
        kw["n_layers"] = layers
    return build_model(NetConfig(**kw), seed=seed).eval()


def stage_numerics(args) -> dict:
    out: dict = {}
    cfg = NetConfig()
    inputs = {f"synthetic_{n}": [synthetic_state(n, n)] for n in args.boards}
    try:
        eng = _engine_states(cfg)
        inputs["engine"] = eng
        out["engine_boards"] = [int(s.n_boards) if hasattr(s, "n_boards") else None for s in eng]
    except Exception as e:  # noqa: BLE001
        out["engine_states"] = _err(e)
    for impl in args.impl:
        for dtype in ("fp32", "bf16"):
            key = f"{impl}/{dtype}"
            try:
                cpu = _net(dtype, impl)
                gpu = copy.deepcopy(cpu).to(DEV)
                ref32 = _net("fp32", impl) if dtype == "bf16" else None  # same seed: same weights
                row = {}
                for name, states in inputs.items():
                    batch = collate(states)
                    with torch.no_grad():
                        c = cpu(batch)
                        g = gpu(batch.to(DEV))
                        _sync()
                        row[name] = {"gpu_vs_cpu_same_dtype": _compare(c, g)}
                        if ref32 is not None:
                            row[name]["gpu_bf16_vs_cpu_fp32"] = _compare(ref32(batch), g)
                out[key] = row
            except Exception as e:  # noqa: BLE001
                out[key] = _err(e)
    return out


def stage_compile(args) -> dict:
    out: dict = {}
    batch = collate([synthetic_state(44, args.boards)])
    # 1. torch.compile (inductor) of the v1 network's forward.
    for impl in args.impl:
        key = f"compile/{impl}/bf16"
        try:
            net = _net("bf16", impl).to(DEV)
            with torch.no_grad():
                eager = net(batch.to(DEV))
                fn = torch.compile(net, backend=args.backend)
                t0 = time.perf_counter()
                comp = fn(batch.to(DEV))
                _sync()
                out[key] = {"ok": True, "first_call_s": round(time.perf_counter() - t0, 1), "vs_eager": _compare(eager, comp)}
        except Exception as e:  # noqa: BLE001
            out[key] = _err(e)
        torch._dynamo.reset()
    # 2. A toy compiled flex_attention with a bias score_mod and a block mask (section 6.1's kernel).
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        B, H, L, D = 1, 8, 2920, 32
        q, k, v = (torch.randn(B, H, L, D, device=DEV, dtype=torch.bfloat16) for _ in range(3))
        bias = torch.randn(H, device=DEV)
        board = torch.arange(L, device=DEV) // 73

        def score_mod(score, b, h, qi, ki):
            return score + bias[h] * (board[qi] == board[ki])

        def mask_mod(b, h, qi, ki):
            return (board[qi] - board[ki]).abs() <= 3

        bm = create_block_mask(mask_mod, B, None, L, L, device=DEV)
        fa = torch.compile(flex_attention, backend=args.backend)
        r = fa(q, k, v, score_mod=score_mod, block_mask=bm)
        _sync()
        out["toy_flex_compiled"] = {"ok": True, "shape": list(r.shape)}
    except Exception as e:  # noqa: BLE001
        out["toy_flex_compiled"] = _err(e)
    return out


def _median_time(fn, reps: int) -> float:
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def stage_throughput(args) -> dict:
    out: dict = {"rows": []}
    for impl in args.impl:
        for dtype in args.dtype:
            net = _net(dtype, impl).to(DEV)
            for boards in args.boards:
                for bs in args.batch:
                    row = {"impl": impl, "dtype": dtype, "boards": boards, "batch": bs}
                    try:
                        states = [synthetic_state(1000 * boards + i, boards) for i in range(bs)]
                        batch = collate(states).to(DEV)
                        torch.cuda.reset_peak_memory_stats()

                        def fwd(net=net, batch=batch):
                            with torch.no_grad():
                                return net(batch)

                        s = _median_time(fwd, args.reps)
                        row.update(
                            tokens=int(batch.n_tokens),
                            median_s=round(s, 4),
                            evals_per_s=round(bs / s, 2),
                            peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        )
                    except Exception as e:  # noqa: BLE001
                        row.update(_err(e))
                        torch.cuda.empty_cache()
                    out["rows"].append(row)
                    print(json.dumps(row), file=sys.stderr, flush=True)
            del net
            torch.cuda.empty_cache()
    return out


def stage_search(args) -> dict:
    from t5d.engine import Game, NetworkEvaluator, Search, run_search, search_params

    out: dict = {}
    for device in args.devices:
        key = f"search/{device}/{args.dtype}"
        try:
            net = _net(args.dtype, args.impl).to(device)
            ev = NetworkEvaluator(net, max_batch=args.batch, device=device)
            game = Game.variant("Standard")
            params = search_params(seed=1, batch_size=args.batch)
            with Search(game, params=params) as search:
                t0 = time.perf_counter()
                stats, driver = run_search(search, ev, playouts=args.playouts, batch_size=args.batch)
                _sync()
                wall = time.perf_counter() - t0
                info = search.info()
            game.close()
            out[key] = {
                "playouts": args.playouts,
                "wall_s": round(wall, 2),
                "playouts_per_s": round(args.playouts / wall, 1),
                "leaves": stats.leaves,
                "leaves_per_s": round(stats.leaves / wall, 1),
                "rounds": stats.rounds,
                "largest_batch": stats.largest_batch,
                "evaluate_s": round(stats.evaluate_seconds, 2),
                "collect_s": round(stats.collect_seconds, 2),
                "deliver_s": round(stats.deliver_seconds, 2),
                "stop": getattr(driver, "stop", None),
                "root_visits": getattr(info, "visits", None),
            }
        except Exception as e:  # noqa: BLE001
            out[key] = _err(e)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pad8", action="store_true", help="pad the attention kernel's head width to a multiple of 8 (this process only)")
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("env")
    sub.add_parser("device")
    n = sub.add_parser("numerics")
    n.add_argument("--impl", nargs="+", default=["block_sparse", "sdpa", "flex"])
    n.add_argument("--boards", type=int, nargs="+", default=[2, 17, 44])
    c = sub.add_parser("compile")
    c.add_argument("--impl", nargs="+", default=["block_sparse", "sdpa", "flex"])
    c.add_argument("--boards", type=int, default=44)
    c.add_argument("--backend", default="inductor")
    t = sub.add_parser("throughput")
    t.add_argument("--impl", nargs="+", default=["block_sparse"])
    t.add_argument("--dtype", nargs="+", default=["bf16"])
    t.add_argument("--boards", type=int, nargs="+", default=[44, 112, 240])
    t.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16])
    t.add_argument("--reps", type=int, default=5)
    s = sub.add_parser("search")
    s.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    s.add_argument("--dtype", default="bf16")
    s.add_argument("--impl", default="block_sparse")
    s.add_argument("--playouts", type=int, default=400)
    s.add_argument("--batch", type=int, default=8)
    args = p.parse_args(argv)
    fn = {
        "env": stage_env,
        "device": stage_device,
        "numerics": stage_numerics,
        "compile": stage_compile,
        "throughput": stage_throughput,
        "search": stage_search,
    }[args.stage]
    if args.pad8:
        install_pad8()
    t0 = time.perf_counter()
    try:
        res = fn(args)
    except Exception as e:  # noqa: BLE001
        res = _err(e)
    res = {"stage": args.stage, "pad8": args.pad8, "seconds": round(time.perf_counter() - t0, 1), **res}
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
