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
| Dense cost per eval at 240 boards | 4.78 TF → ~337 evals/s (FA4-class) | - | | | |
| Windowed throughput | ~2,370 evals/s; ~1,520/s at k = 8 | - | | | |
| Cached cost per node | 0.16 TF; 15-20× realistic | - | | | |
| Live-set recompute | 3.7-5.6× changed-boards-only | - | | | |
| Supersession recompute | +8-24% per node | - | | | |
| Root refresh overhead (800 sims) | +3.6% / +6.1% full; +1.3% windowed | - | | | |
| One refresh | ≈ 10 node-evaluations | - | | | |
| KV memory per state | ~260 MB (240) / ~649 MB (600); ~113 MB windowed | - | | | |
| Local/global interleave | 0.067 TF per node | - | | | |
| Leaf-batching gain | 1.2-1.5× | - | | | |
| NNUE eval | ~10k MACs, sub-µs | - | | | |
| Movegen throughput | ≥ 20,000/s at 240; ≥ 10,000/s at 600 (§15.6) | - | | | |
| Scanner nodes/s | no figure | - | | | |

## §15.6 verdicts

### Movegen throughput (M9) - not yet measured

### The window: |E| phase curve (M7) - not yet measured

### The scanner: 6b (M8) - not yet measured
