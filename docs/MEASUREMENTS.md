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
| Movegen throughput | ≥ 20,000/s at 240; ≥ 10,000/s at 600 (§15.6) | pseudo-legal expansion, 1 thread: 44,047/s at 240 boards, 24,951/s at 600 (scope + relations); with the legal-moves filter: milliseconds to minutes per expansion | M9 section below | 2.2x / 2.5x the target (pseudo-legal); far below it with the filter | 2026-10-02, 2740e66 |
| Scanner nodes/s | no figure | 1,036,000/s (Q4 per the Guide) and 1,033,000/s (Q4's other reading), one thread, pooled over the corpus; lower on large positions (85% of the nodes are E4 searches) | 6b section below | - | 2026-10-02, bde62c5 |

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

### Engine E4 - the gate's full mode on the corpus (2026-10-02)

dev, g++ 16.2 -O2, `t5d-gate --cap 1000` (default: hypercuboid-then-naive
portfolio, naive cross-check), one process at a time, 120 s limit per record,
default rules (Q4 per the Guide). All 41 corpus records: **8.3 s in total**, no
time-out, no error. Slowest: cwmtt/manyChecks 4.11 s for 53 positions (slowest
position 0.73 s; it took over 25 minutes before the fix round), notation/aivsai
1.24 s, exiledKings 0.45 s, ctp2 0.43 s; every other record under 0.3 s. Peak
working set at most 50 MB. With `--moves-after-shift` (Q4's other reading):
cwmtt/100_timelines 5.7 s (132 MB), ftxi/futures 0.14 s, cwmtt/niceAndrey 19.9 s
for 122 positions (48 MB).

The expansion's legal-moves filter (`ExpandOptions::legal_moves`; synthetic
random-play positions, 4 games x 5 positions per class, machine loaded):

| Class (N, T) | expand-scope median | + legal, median | + legal, mean | + legal, worst | moves kept |
|---|---|---|---|---|---|
| 100, 6 | 0.012 ms | 11 ms | 19 ms | 94 ms | 56% |
| 160, 7 | 0.019 ms | 50 ms | 53 ms | 149 ms | 67% |
| 240, 8 | 0.028 ms | 157 ms | 175 ms | 0.8 s | 29% |
| 400, 10 | 0.036 ms | 218 ms | 15.7 s | 247 s | 27% |
| 600, 12 | 0.056 ms | 0.8 ms | 36.3 s | 522 s | 7% |

Not an M9 figure: M9 measures the expansion without this filter (spec issue 46).

## §15.6 verdicts

### Movegen throughput (M9) - pseudo-legal: PASS; with fast legality: inside the STOP band at 240 boards on one core

**Conditions.** dev, quiet machine (the bench measured the machine busy 4-5%
during 1-thread samples - itself), Windows 11, g++ 16.2 `-O2 -DNDEBUG`, static
`t5d-bench --samples 15` at 2740e66 (15 interleaved rounds of 200 ms after a
warm-up; worker threads pinned, one per physical core first; thread CPU time by
QueryThreadCycleTime). SYNTHETIC positions from seeded random play, 16 games per
class, at the spec's assumed timeline counts (N100 T6 ... N600 T12); timed streams
are the positions' one-move children (6 per position per worker, disjoint across
workers). Integer work: no precision or batch. Window setting: W(8) and E(8)
computed (no network). Moves PSEUDO-LEGAL except where marked. Raw output and
JSON: `build/m9/` (git-ignored; reproducible from the command).

**Throughput, one thread** (wall /s, median over rounds [min, max]):

| Class | movegen | expand (every board's features) | expand + relations | expand-scope | **scope + relations** | M9 target |
|---|---|---|---|---|---|---|
| N100 T6 | 171,804 | 103,077 | 72,762 | 104,089 | 72,681 | - |
| N160 T7 | 139,053 | 79,681 | 57,311 | 82,731 | 59,008 | - |
| N240 T8 | 102,263 | 58,810 | 42,631 | 62,292 | **44,047 [32,544, 44,733]** | 20,000 |
| N400 T10 | 80,213 | 42,967 | 31,969 | 47,268 | 34,100 | - |
| N600 T12 | 58,002 | 30,374 | 23,183 | 34,156 | **24,951 [22,636, 25,129]** | 10,000 |

"scope + relations" is the configuration closest to the spec's "expansion - legal
moves, attention masks, typed edges and E" (spec issue 21): the moves, the move
graph, W/E scope with its features, the typed relations. It clears M9's targets
on ONE thread by 2.2x at 240 boards (1.63x in the slowest round) and 2.5x at 600:
outside the ±10% STOP band. With t threads it scales to 10.5x at 12 (one per
physical core) and 13.2x at 24; "per GPU's CPU allocation" is undefined (spec
issue 20), and one core already suffices.

**Flatness in N at fixed |E|** (M9: ≤ 10% growth from 100 to 600 boards).
SYNTHETIC flatness series at T = 8, 24 games, one thread, interleaved; growth =
N600 against N100 per round, median [min, max] over 15 rounds. *Matched*: 52
positions per level paired on moves, |E(8)| and the mover's pieces (within 10% /
max(1, 20%) / 10%; achieved mismatch ≤ 3.7% / 0.35 / 4.5%):

| Configuration | µs per expansion N100 → N160 → N240 → N400 → N600 | growth | reading |
|---|---|---|---|
| movegen | 3.11 → 4.13 → 4.74 → 4.39 → 2.96 | -5% [-28%, +5%] | below +10% in every round |
| expand-scope | 6.34 → 8.61 → 9.33 → 9.32 → 7.18 | +13% [+12%, +35%] | above +10% in every round, but not monotonic in N |
| **scope + relations** | 9.32 → 12.78 → 13.37 → 12.81 → 10.03 | **+7% [-17%, +13%]** | rounds straddle +10%: inconclusive |
| expand (every board) | 6.19 → 9.06 → 11.13 → 13.52 → 13.65 | +120% | O(N) by construction (spec issue 21) |

*No moves* (the history term alone - what M9's flatness test is after, "frozen
boards' move and attack sets are not being cached"): the windowed expansion's
per-board cost is **0.48 ns per board [0.41, 0.68] = 2% of a matched expansion at
N600**; frozen boards are not being re-processed. The matched series' level-to-
level variation (N160-N400 above both ends) is larger than its N100-N600 growth:
position sampling, not an N trend, dominates it.

**Verdict (provisional, pseudo-legal):** throughput PASS at both sizes, outside
the STOP band. Flatness: the history term is flat (2%); the spec's expansion
reads +7% median, inconclusive by rounds, none of the medians inside the 9-11%
STOP band; expand-scope's +13% is flagged (non-monotonic). Re-measure with legal
moves once the filter below is fast.

**With legality** - engine E5a's classifier in tiers (`t5d/legal.hpp`: exact;
budgeted, which never calls a move illegal without proof and leaves the rest
"unknown"; fast, budget 0). Profile per expansion of scope + relations with the
classification inside, one thread, `t5d-bench --legal-profile` at E5a's 8d08b09,
**machine shared with other agents' builds and tests** - not a verdict.
Synthetic M9 classes, 20 expansions each, median / p95 / worst (share of moves
left unknown):

| Class | pseudo-legal | fast | budgeted (64 checks, 2,000 nodes) | exact |
|---|---|---|---|---|
| N100 T6 | 21 / 43 / 43 µs | 86 / 114 / 114 µs (46%) | 2.0 / 2.9 / 2.9 ms (14%) | 1.5 / 17 / 17 ms |
| N160 T7 | 23 / 38 / 38 µs | 95 / 119 / 119 µs (64%) | 2.1 / 3.0 / 3.0 ms (44%) | 5.4 / 59 / 59 ms |
| N240 T8 | 26 / 46 / 46 µs | 102 / 162 / 162 µs (33%) | 2.5 / 4.1 / 4.1 ms (20%) | 7.3 / 148 / 148 ms |
| N400 T10 | 40 / 55 / 55 µs | 142 / 232 / 232 µs (47%) | 2.4 / 10 / 10 ms (25%) | 17 ms / 145 s / 145 s |
| N600 T12 | 54 / 84 / 84 µs | 162 / 273 / 273 µs (27%) | 0.75 / 11 / 11 ms (7%) | 0.72 ms / 200 s / 200 s |

Corpus records (every action start and its children, `--moves-after-shift`):

| Boards | expansions | pseudo-legal | fast | budgeted | exact |
|---|---|---|---|---|---|
| 1-49 | 858 | 2 / 8 / 25 µs | 6 / 27 / 96 µs (39%) | 22 µs / 454 µs / 2.1 ms (2%) | 25 µs / 422 µs / 22 ms |
| 50-99 | 319 | 4 / 21 / 60 µs | 16 / 106 / 218 µs (45%) | 0.41 / 2.0 / 3.4 ms (11%) | 0.26 / 4.0 / 33 ms |
| 100-159 | 261 | 11 / 42 / 109 µs | 55 / 186 / 418 µs (43%) | 1.1 / 3.2 / 8.1 ms (16%) | 0.82 ms / 14 ms / 3.5 s |
| 160-239 | 180 | 12 / 70 / 281 µs | 131 / 262 / 700 µs (47%) | 1.1 / 3.5 / 4.9 ms (24%) | 0.85 ms / 14 ms / 2.6 s |
| 240-399 | 138 | 67 / 327 / 359 µs | 288 µs / 1.6 ms / 2.1 ms (55%) | 3.9 / 11 / 18 ms (33%) | 7.1 / 144 / 250 ms |

Reading: the fast mode costs 3-5x the pseudo-legal expansion and is bounded
(worst ≤ 2.1 ms); the budgeted mode is bounded by its budget; exact keeps the
mate-search tail (minutes at 400-600 boards, against E4's per-move filter: 157 ms
median at 240 boards). On check-heavy random play the fast mode proves few moves
legal (it settles 36-73% of moves, as illegal); how search treats an unknown move
is spec issue 46. **M9 with legality, quiet machine** (`t5d-bench --samples 15 --legal
--legal-budget 0,0 --no-flatness` at bde62c5; 1-thread samples at 5% load, the
bench itself; the FAST mode - budget 0: unknown moves kept, 27-47% of moves unknown
by class). Scope + relations + legality, wall /s median [min, max] over 15 rounds:

| Class | 1 thread | 2 threads | 12 threads | M9 target | ratio, 1 thread |
|---|---|---|---|---|---|
| N100 T6 | 25,712 [20,199, 26,288] | 48,380 | 249,135 | - | - |
| N160 T7 | 18,555 [13,836, 19,421] | 37,062 | 205,314 | - | - |
| N240 T8 | **20,249 [14,391, 20,818]** | 37,316 | 190,897 | 20,000 | **1.01x** (0.72-1.04 over rounds) - **inside the ±10% band** |
| N400 T10 | 16,721 [13,385, 16,928] | 32,367 | 159,107 | - | - |
| N600 T12 | 12,133 [10,497, 12,288] | 23,674 | 128,736 | 10,000 | 1.21x - PASS |

Verdict: on ONE core per GPU the 240-board figure sits inside the STOP band; on two
cores it is 1.87x the target. Which applies turns on "per GPU's CPU allocation",
which the spec leaves undefined (spec issue 20) - a question for the user, with 6b
below. Exact legality would not meet the target at any core count (its tail is
minutes).

### The window: |E| phase curve (M7) - NO VERDICT: the corpus cannot decide it

`t5d-measure --summary` over the 41 corpus records (2026-10-02, 2740e66),
PSEUDO-LEGAL destinations (every |E| an upper bound), distinct positions, records
grouped into games (NP, NP0 and many2 one game; standard and softmate one; the two
futures one). Under Q4's other reading (`--moves-after-shift`, which the nine
records that play on after the shift need):

| N bucket | positions | games | active T (median) | \|W(8)\| (bands only) | \|E(8)\| median / p95 / max | \|E(4)\| median / p95 / max | games' medians of \|E(8)\| |
|---|---|---|---|---|---|---|---|
| 1-49 | 508 | 34 | 1 | 9 (9) | 0 / 3 / 17 | 1 / 5 / 19 | 0 [0, 2] |
| 50-99 | 90 | 7 | 4 | 34.5 (33) | 2 / 11 / 20 | 6.5 / 19 / 21 | 3 [0, 5] |
| 100-159 | 57 | 5 | 8 | 68 (62) | 3 / 20.2 / 24 | 9 / 27 / 34 | 4 [2, 19] |
| 160-239 | 42 | 3 | 16 | 122 (114) | 15.5 / 27 / 37 | 28 / 39 / 45 | 19 [10.5, 22] |
| 240-399 | 23 | 2 | 24 | 185 (174) | 10 / 23 / 23 | 27 / 46 / 50 | 11.5 [5, 18] |
| 400+ | 0 | 0 | | | | | |

From LEGAL destinations (engine E5a's exact classifier, `t5d-measure --summary
--moves-after-shift --legal`, same records), |E(8)| median / p95 / max and |E(4)|:
1-49: 0 / 3 / 15 and 0 / 5 / 16; 50-99: 2 / 11 / 18 and 5 / 18 / 19; 100-159: 3 /
16.4 / 22 and 8 / 27 / 32; 160-239: **4.5** / 18.95 / 24 (pseudo-legal 15.5) and 13
/ 33.95 / 39; 240-399: 8 / 21.9 / 22 and 26 / 41 / 47. Games' medians of |E(8)|:
0 [0, 2], 2 [0, 5], 4 [1, 6], 9.5 [0, 22], 10 [4, 16].

Under the engine's default Q4 reading nothing reaches 160 boards. No verdict: the
phases above 160 boards rest on 2-3 games, the 400-600 phases on none, and the
real games run 16-24 active timelines where §15.6's table assumes 7-12 - so W
itself, not |E|, dominates the scope (at 160-239 boards W(8) = 122 against the
table's 69). The rule needs real game records (spec issue 26; requested from the
user); the window code stays parked (M5, M7).

### The scanner: 6b (M8) - ROOT SCAN FAILS ABOVE 50 BOARDS; IN THE STOP BAND UNDER Q4'S OTHER READING - STOP

**Conditions.** dev, quiet machine (busy 5.4% over the run, all processes - the
tool itself), g++ 16.2 `-O2 -DNDEBUG`, engine at bde62c5 (E5b's scanner merged with
E5a's legality tiers). `t5d-scan6b --max-nodes 100000 --jobs 1 --root-wall
--per-position` over the 41 corpus records (every turn start; 1 leaf per root at 2
half-turns of random legal play; duplicate roots dropped; TT and histories cleared
per position; deterministic by node budget, plus one 50 ms wall-clock scan per
root). Scanner: forcing = captures, checks (the pass test), branch threats (the
arriving piece can capture at least 300), promotions not forcing; quiet completion;
stand-pat on; evasion cap 512; completion fallback 2,000 nodes; material eval with
placeholder 5D values (spec issue 56). Rung 1's evaluation rate is §15.6's design
rate (no GPU here; spec issue 62), "size-matched" = 9,600/s below 420 boards and
4,600/s above; threads 1 and 11 (physical cores - 1; the per-GPU allocation is
undefined, spec issue 20). Raw output: `build/6b/` (git-ignored; reproducible from
the command). Each run takes 1.5-2.5 minutes.

| Test (§15.6) | Pass | Q4 per the Guide (601 roots, 595 leaves; median 19 boards, max 140) | Q4's other reading (805 roots, 780 leaves; median 27, max 378) |
|---|---|---|---|
| (iii) root scan at 50 ms (51,800 nodes) | ≥ 5 forcing half-turns in ≥ 50% of roots | **60.9%** (wall clock 61.6%) - PASS pooled | **49.2%** (wall clock 49.4%), median horizon 4 - **FAIL, inside the ±10% band** |
| (iii) by board count | | < 50: 68.4% (484); 50-99: **34.4%** (90); 100-199: **14.8%** (27) | < 50: 64.7% (527); 50-99: **25.7%** (140); 100-199: **18.6%** (97); 200+: **2.4%** (41) |
| (ii) rung-1 leaf scan, 11 threads, size-matched rate (1,187 nodes) | ≥ 3 in ≥ 50% of leaves | 42.2% - FAIL (drop rung 1) | 33.3% - FAIL |
| (ii) 11 threads at 4,600/s (2,477 nodes) | | 50.4%, median 3 - inside the band | 41.8% - FAIL |
| (ii) 1 thread | | 12.1% (9,600/s), 21.8% (4,600/s) - FAIL | 9.6%, 17.4% - FAIL |
| (iv) quiet-leaf reach at the rung-1 budget | ≥ 50% | 4.9% (11 threads, size-matched) - FAIL (NNUE v2 deferred) | 4.6% - FAIL |
| (v) stand-pat invalid | ≤ 5% fine | 2.0% of 149 decided quiet positions - fine | 3.0% of 164 - fine (2-3 mandatory boards: 11.1% of 9) |

**Horizon per node budget** (roots resolving at least 5 forcing half-turns):
Guide reading 19.6% at 100 nodes, 41.6% at 1k, 53.4% at 10k, 62.6% at 100k; other
reading 15.0%, 32.4%, 43.2%, 50.8%. The passes rest largely on exhausted trees and
proofs (237 and 84 of the Guide reading's 366 passing roots) - small positions.

**Where the nodes go:** 85-86% inside E4's exhaustive searches - listing every legal
action in check (50-57%: in 5D "all evasions" is every legal action of a turn,
a product over boards, spec issue 55), mate tests at the horizon (23%) and
completion fallbacks (6-14%); about 80 nodes per tree position.

**Verdict.** The pooled root scan passes under the engine's default Q4 reading and
fails inside the ±10% band under the other; in the design's regime - 100 boards and
up - it fails under both (15-19% at 100-199 boards, 2.4% at 200+). Under the
brief's rules this STOPS the run for the user: "6b root-scan fails" and "a §15.6
result within ±10% of its threshold" (§11.1's kill condition: "the verification
layer needs a different shape"). Whether the test is pooled over a corpus or per
phase is undefined (spec issue 61). Rung 1 fails (drop rung 1, its own rule); the
quiet-leaf reach fails (NNUE v2 and quiet-only training deferred, M8); stand-pat is
valid.
