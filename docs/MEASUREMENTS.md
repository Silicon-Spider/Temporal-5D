# Measurements

Every figure measured during the untrained build, with the conditions it was
measured under, beside the estimate it replaces (spec §15.3) and the verdict of
any pre-registered threshold it decides (spec §15.6, register M7-M9).

The spec and the register are not edited in this repository: a measured figure is
recorded here, and the edit that writes it back over its estimate (M4) is proposed
in `docs/BUILD_LOG.md`.

**Conditions recorded with every figure:** hardware, precision, batch size, board
count, window setting - and for movegen and |E|, whether moves are legal or
pseudo-legal (pseudo-legal figures are provisional: an |E| from pseudo-legal
destinations is an upper bound, and movegen throughput is re-measured once
legality exists).

A figure more than 2× off its §15.3 estimate is flagged here and logged in the
build log; the run continues. A §15.6 verdict within ±10% of its threshold stops
the run for the user.

## Machines

| Name | Hardware | Software |
|---|---|---|
| dev | AMD Ryzen 9 9900X (12 cores / 24 threads, AVX-512 with BF16), 32 GB RAM; AMD Radeon RX 9070 XT, unused (no CUDA) | Windows 11; MSYS2 ucrt64 g++ 16.2; Python 3.14; PyTorch 2.13.0+cpu |

## §15.3 cost figures

| Figure | Estimate (§15.3) | Measured | Conditions | Ratio | Date, commit |
|---|---|---|---|---|---|
| Dense cost per eval at 240 boards | 4.78 TF → ~337 evals/s (FA4-class) | counted 4.11 TF (B14 block-sparse), 4.91 TF (every key); CPU wall pending | N3 table below | counted / estimate 0.86-1.03 | 2026-10-02, N3 |
| Windowed throughput | ~2,370 evals/s; ~1,520/s at k = 8 | - | | | |
| Cached cost per node | 0.16 TF; 15-20× realistic | counted 0.168 TF; 24.4× / 29.2× counted (vs block-sparse / every-key dense) at 8 timelines; **8.4× / 9.9× at 24 timelines**; CPU wall pending | N3 table below | 1.05 (FLOPs); ratio at 24 timelines below the 15-20× - spec issue 36 | 2026-10-02, N3 |
| Live-set recompute | 3.7-5.6× changed-boards-only | counted 3.68× (240, 8 TL), 5.47× (600, 12), **10.84× at 24 timelines** | N3 table below | 1.0 at the spec's timelines; ~2-3× above it at 24 | 2026-10-02, N3 |
| Supersession recompute | +8-24% per node | counted, lineage: +5.9% / +12.0% (240, 8 TL; 1 / 2 boards), +0.3-2.8% elsewhere; creation reading cascades (spec issue 35) | N3 table below | within the bound under lineage | 2026-10-02, N3 |
| Root refresh overhead (800 sims) | +3.6% / +6.1% full; +1.3% windowed | analytic only so far (3.64% / 6.13% at the spec's timelines; 1.24% / 3.10% at 24); CPU wall pending | N3 table below | - | 2026-10-02, N3 |
| One refresh | ≈ 10 node-evaluations | not measured: the estimate is windowed and the window is not built (spec issue 39) | | | |
| KV memory per state | ~260 MB (240) / ~649 MB (600); ~113 MB windowed | BF16 250.9 MB (240, 8 TL), 635.8 MB (600, 12 TL) - frozen boards only, exact; FP8 50.1% | N3 table below | 0.97 / 0.98 | 2026-10-02, N3 |
| Local/global interleave | 0.067 TF per node | - | | | |
| Leaf-batching gain | 1.2-1.5× | - | | | |
| NNUE eval | ~10k MACs, sub-µs | - | | | |
| Movegen throughput | ≥ 20,000/s at 240; ≥ 10,000/s at 600 (§15.6) | - | | | |
| Scanner nodes/s | no figure | - | | | |

### Network N3 - the cached path, counted (2026-10-02)

Machine dev (counts do not depend on it), batch 1, `synthetic_state` seed 0, F1
network (d 256, L 16, 8 heads, FFN 8x) with B9's two move-graph heads, random
weights; cached path: block-causal scope, lineage, intrinsic frozen features, C20's
first reading; **no window**. *Analytic* = the spec's formula (4·Tq·Tk·d·L +
40·Tq·d²·L) at the measured token counts - a check of the arithmetic, not a
measurement. *Counted* = the FLOPs the implementation issued (PyTorch's
FlopCounterMode, with a registered formula for the CPU fused-attention op).
Timelines: the spec's assumption (8 at 240 boards, 12 at 600) and the corpus's
real games (24 at 240-399 boards; 24 stands in at 600, which the corpus does not
reach).

| Setting | Dense: analytic / counted block-sparse / counted every-key | Node: analytic / counted | Dense ÷ node: analytic / counted vs block-sparse / vs every-key | Live set (2 changed): analytic / counted | Supersession, counted, 1 / 2 boards | KV per state BF16 / FP8 / FP32 |
|---|---|---|---|---|---|---|
| 240 boards, 8 TL (spec) | 4.784 / 4.114 / 4.913 TF | 0.1641 / 0.1684 TF | 29.1 / 24.4 / 29.2x | 3.68 / 3.68x | 5.9% / 12.0% | 250.9 / 125.7 / 501.7 MB |
| 240 boards, 24 TL (corpus) | 4.784 / 4.159 / 4.913 | 0.4828 / 0.4956 | 9.9 / 8.4 / 9.9x | 10.81 / 10.84x | 0.9% / 1.9% | 233.6 / 117.0 / 467.1 |
| 600 boards, 12 TL (spec) | 27.375 / 22.297 / 28.180 | 0.5583 / 0.5744 | 49.0 / 38.8 / 49.1x | 5.46 / 5.47x | 0.6% / 2.8% | 635.8 / 318.5 / 1271.7 |
| 600 boards, 24 TL (corpus) | 27.375 / 22.490 / 28.180 | 1.1056 / 1.1378 | 24.8 / 19.8 / 24.8x | 10.81 / 10.84x | 0.3% / 0.9% | 622.9 / 312.0 / 1245.7 |

Root refresh at 800 sims, analytic: 3.64% / 1.24% / 6.13% / 3.10% (rows in
order). KV bytes are the cache's exact tensor bytes, frozen boards only; the trunk
outputs the cache also holds add about 16 MB at 240 boards and 40 MB at 600. The
spec's all-token figures (259.8 / 649.1 MB) are reproduced by the analytic model.

**Flags (more than 2x off, logged in the build log):** at the corpus's 24
timelines the counted cached-vs-dense ratio is 8.4-9.9x at 240 boards against the
spec's 15-20x realistic (29x paper), because the spec assumes ~8 live heads (spec
issue 36); the live-set ratio is 10.8x against 3.7-5.6x; under the creation
reading of the causal past, supersession cascades far past its +24% bound (issue
35).

**CPU wall times: pending.** The first round's wall times were taken without load
verification and are not recorded here (one 600-board BF16 job was visibly
disturbed). About 58% of a cached node's CPU time is the prototype's per-node
gather and rotation of every cached key; kernel-only ratios measured by a reviewer
were 3.64x (live set) and 14x (cached vs dense) at 240 boards, 8 timelines. The
quiet-machine re-run (commands in the build log) records wall and kernel time
with load samples per job.

## §15.6 verdicts

### Movegen throughput (M9) - not yet measured

### The window: |E| phase curve (M7) - not yet measured

### The scanner: 6b (M8) - not yet measured
