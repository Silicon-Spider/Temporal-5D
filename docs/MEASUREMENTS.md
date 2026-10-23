# Measurements

Every figure measured during the untrained build, with the conditions it was
measured under, beside the estimate it replaces (spec §15.3) and the verdict of
any pre-registered threshold it decides (spec §15.6, register M7-M9). Phase 1 counts
that decide no threshold (the turn width, M22; D2's tree form's evaluation counts) are
at the end.

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
| dev | AMD Ryzen 9 9900X (12 cores / 24 threads, AVX-512 with BF16), 32 GB RAM; AMD Radeon RX 9070 XT, unused (no CUDA) | Windows 11; MSYS2 ucrt64 g++ 16.2; Python 3.14; PyTorch 2.13.0+cpu - every figure up to 2026-10-04 |
| dev-r | The same machine, rebuilt after its loss (2026-10-17); GPU driver 32.0.31041.1004, the GPU still unused by the CPU figures | Windows 11 Pro 10.0.26300 (a fresh install); MSYS2 ucrt64 g++ 16.2.0 (Rev3); CMake 4.4.3 and Ninja 1.13.2 from the project `.venv`; Python 3.14.7; PyTorch 2.13.0+cpu; NumPy 2.5.2; Node 24.20.0; GHC 9.10.3 (harness cache) - every figure from 2026-10-17 |

`dev` and `dev-r` are different conditions: the operating system was reinstalled and
the Python patch level and CMake changed, so a figure from one is never compared with
one from the other as if they were the same measurement - a figure that matters is
re-measured on `dev-r`. The first sign is the gate's calibration probe (a fixed
ftxi:naive workload, `[calibration]` in `oracles/gate.toml`): on `dev-r`'s first
commit-tier run, right after the rebuild with no other job of the build running,
it took **10.00 s against 8.47 s on the machine that recorded the golden files
(1.18x)**. The golden files were recorded on `dev`; their costs decide which values
each tier requires, so a re-recording (H48) is done on a quiet `dev-r` (the
quiet-machine block).

## §15.3 cost figures

| Figure | Estimate (§15.3) | Measured | Conditions | Ratio | Date, commit |
|---|---|---|---|---|---|
| Dense cost per eval at 240 boards | 4.78 TF → ~337 evals/s (FA4-class) | counted 4.11 TF (B14 block-sparse), 4.91 TF (every key); CPU wall pending | N3 table below; the network before M20's register positions - re-count pending | counted / estimate 0.86-1.03 | 2026-10-02, N3 |
| Windowed throughput | ~2,370 evals/s; ~1,520/s at k = 8 | - | | | |
| Cached cost per node | 0.16 TF; 15-20× realistic | counted 0.168 TF; 24.4× / 29.2× counted (vs block-sparse / every-key dense) at 8 timelines; **8.4× / 9.9× at 24 timelines**; CPU wall pending | N3 table below; the network before M20's register positions - re-count pending | 1.05 (FLOPs); ratio at 24 timelines below the 15-20× - spec issue 36 | 2026-10-02, N3 |
| Live-set recompute | 3.7-5.6× changed-boards-only | counted 3.68× (240, 8 TL), 5.47× (600, 12), **10.84× at 24 timelines** | N3 table below; the network before M20's register positions - re-count pending | 1.0 at the spec's timelines; ~2-3× above it at 24 | 2026-10-02, N3 |
| Supersession recompute | +8-24% per node | counted, lineage: +5.9% / +12.0% (240, 8 TL; 1 / 2 boards), +0.3-2.8% elsewhere; creation reading cascades (spec issue 35) | N3 table below | within the bound under lineage | 2026-10-02, N3 |
| Root refresh overhead (800 sims) | +3.6% / +6.1% full; +1.3% windowed | analytic only so far (3.64% / 6.13% at the spec's timelines; 1.24% / 3.10% at 24); CPU wall pending | N3 table below; one dense pass - two under A7 (note below) | - | 2026-10-02, N3 |
| One refresh | ≈ 10 node-evaluations | not measured: the estimate is windowed and the window is not built (spec issue 39) | | | |
| KV memory per state | ~260 MB (240) / ~649 MB (600); ~113 MB windowed | BF16 250.9 MB (240, 8 TL), 635.8 MB (600, 12 TL) - frozen boards only, exact; FP8 50.1% | N3 table below; one perspective - under A7 every store holds two entries per root frozen board from the moment its epoch is made (note below) | 0.97 / 0.98 | 2026-10-02, N3 |
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

*Conditions note (2026-10-17, `lane/net`):* these counts are of the network as built
on 2026-10-02 - before F5, B15 and B16, and before M20's register positions (B8): its
per-board registers then sat at a learned off-lattice angle and only the global
registers' pairs were merged, where the v1 now widens every row's query and key
features (44 dims instead of 32 at the B4 allocation) and recomputes the register rows
on the CPU kernel path, dense and cached alike. They are no longer the v1's counted
cost; no figure here replaces them. The counts do not depend on load, but counting runs
the computations themselves (FlopCounterMode executes them - 28 TF for the 600-board
dense pass), so the re-count is step 2's quiet-machine block's, with the wall times.

| Setting | Dense: analytic / counted block-sparse / counted every-key | Node: analytic / counted | Dense ÷ node: analytic / counted vs block-sparse / vs every-key | Live set (2 changed): analytic / counted | Supersession, counted, 1 / 2 boards | KV per state BF16 / FP8 / FP32 |
|---|---|---|---|---|---|---|
| 240 boards, 8 TL (spec) | 4.784 / 4.114 / 4.913 TF | 0.1641 / 0.1684 TF | 29.1 / 24.4 / 29.2x | 3.68 / 3.68x | 5.9% / 12.0% | 250.9 / 125.7 / 501.7 MB |
| 240 boards, 24 TL (corpus) | 4.784 / 4.159 / 4.913 | 0.4828 / 0.4956 | 9.9 / 8.4 / 9.9x | 10.81 / 10.84x | 0.9% / 1.9% | 233.6 / 117.0 / 467.1 |
| 600 boards, 12 TL (spec) | 27.375 / 22.297 / 28.180 | 0.5583 / 0.5744 | 49.0 / 38.8 / 49.1x | 5.46 / 5.47x | 0.6% / 2.8% | 635.8 / 318.5 / 1271.7 |
| 600 boards, 24 TL (corpus) | 27.375 / 22.490 / 28.180 | 1.1056 / 1.1378 | 24.8 / 19.8 / 24.8x | 10.81 / 10.84x | 0.3% / 0.9% | 622.9 / 312.0 / 1245.7 |

Root refresh at 800 sims, analytic: 3.64% / 1.24% / 6.13% / 3.10% (rows in
order).

*Conditions note (2026-10-22, `lane/net` N-A7):* register A7 reads every board from the
acting player's side, so the cached path keys frozen K/V by the acting player (spec issue
303). Everything in this table is one perspective's. As built, a search root's history is
computed from both colours' sides and a refresh runs its dense pass twice, so by
construction the history build and the root-refresh overheads double (analytic at 800
simulations: 7.28% / 2.48% / 12.26% / 6.20%). The other side is computed eagerly
(`root_epoch`, `_refresh`), so every store holds two entries per root (or refresh anchor)
frozen board from the moment its epoch is made, whichever colours' nodes are then reached -
twice the KV bytes above for those boards, while one state's figure is unchanged; a board
frozen later (a superseded one) gets its second entry only when the other colour's nodes
read it. Arithmetic from the construction, not a measurement: the counts and
wall times are step 2's quiet-machine block's (`cache_bench`'s `analytic_both_perspectives`
and `cache_bytes_per_perspective`). KV bytes are the cache's exact tensor bytes, frozen boards only; the trunk
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

### The window: |E| phase curve (M7) on real games - M16 APPLIED: ON AT k = 8 FROM 50 TO 186 BOARDS, OFF UNDER 50 - AND THEN KILLED BY THE KILL CLAUSE UNDER HUMAN-GAME WEIGHTING

**M16 applied (CHECKPOINT 4, 2026-10-03).** The user fixed three readings: the
all-in multiplier is the time-share (harmonic) M = 1 / (0.9 / m_cached + 0.1 /
m_dense); the kill clause is weighted by real human games (t0database) until
self-play exists, and re-checked against early self-play before serious self-play
compute; under 50 boards the window is off ("that phase carries a small share of
compute and dense attention there is cheap, so the choice is immaterial"). On the
same measurements as below (legal |E|, each phase at its median N, W and |E|):

| Phase | M at k = 8 (time-share) | the geometric reading, for comparison | Verdict under M16 |
|---|---|---|---|
| 1-49 | - | 1.445 | **off** (M16) |
| 50-99 | **1.831** | 1.913 | on at k = 8 |
| 100-159 | **2.099** | 2.184 | on at k = 8 |
| 160-186 (2 games) | **3.807** | 4.276 | on at k = 8 |
| 187+ | - | - | no data: provisional |

**The kill clause** (each phase weighted by its unwindowed per-node cost × the
moves human games spend in it): the phases where the window is on (50 boards and
up) carry **18.4% of the cached compute, 38.0% of the dense, 20.4% blended 90/10**
(per bucket median; 19.6% / 41.4% / 21.8% taking each position's own N). That is
under half, so **the window is killed** under M16 as written.

**This contradicts M16's premise.** The ruling turns the window off under 50 boards
because "that phase carries a small share of compute"; under the weighting M16
itself prescribes, that phase carries about 80% of it (human games spend 87.8% of
their moves under 50 boards and end at a median of 44 boards). Not a STOP (M16: a
result in the ±10% band under 50 boards needs no stop; the kill clause's 20% is far
from its 50% line), so recorded, and the question put to the user. The window code
stays unbuilt either way (M5: the window only if the |E| curve survives); M16's
re-check against early self-play remains.

**Confirmed at CHECKPOINT 5:** the window is killed, provisionally; the weighting
is not changed (M16 corrected: under 50 boards the per-phase rule decides, a result
in the ±10% band resolving to off). A pilot self-play run before serious self-play
re-checks it (M18) - after §15.5, outside the untrained build.


2026-10-03, engine at 901e5be (Release, g++ 16.2, 383 tests passing). The data:
t0database (`data/t0database/`, pinned, MIT): `py data/t0database/to_5dpgn.py
build/t0db` writes its 761 games with the source's own `[Board "Standard - Turn
Zero"]`; `t5d-measure --moves-after-shift --summary [--legal]` (Q4 as ruled, R8)
over the **716 that replay to the end**. Positions are action starts, distinct by
fingerprint and side to move; board counts include the T0 board. Games: the tool's
grouping gives 674 (it joins identical records, prefixes, and 18 distinct games
that share a 10-21-position opening); strictly 692 distinct games. Cross-checked
by an independent re-computation of the summary from the per-position lines.

From LEGAL destinations (the verdict uses these):

| N bucket | positions | games | median N | active T (median) | \|W(8)\| (bands only) | \|W(4)\| | \|E(8)\| median / p95 / max | \|E(4)\| median / p95 / max | games' medians of \|E(8)\| |
|---|---|---|---|---|---|---|---|---|---|
| 1-49 | 19,410 | 674 | 24 | 1 | 9 (9) | 5 | 0 / 4 / 20 | 1 / 6 / 22 | 0 [0, 3] |
| 50-99 | 3,162 | 256 | 60 | 2 | 19 (18) | 11 | 0 / 11 / 30 | 2 / 14 / 32 | 0 [0, 9] |
| 100-159 | 171 | 14 | 112 | 4 | 36 (32) | 23 | 2 / 21 / 56 | 3 / 24.5 / 57 | 1.75 [0, 12] |
| 160-239 | 18 | 2 | 169.5 | 2 | 19 (18) | 11 | 2 / 17.15 / 18 | 2 / 24.15 / 25 | 7.5 [1, 14] |
| 240+ | 0 | 0 | | | | | | | |

PSEUDO-LEGAL (upper bounds), |E(8)| and |E(4)| median / p95 / max: 1-49 0 / 5 / 20
and 2 / 7 / 22; 50-99 1 / 13 / 33 and 4 / 16 / 35; 100-159 6 / 22.5 / 58 and 9 /
26.5 / 62; 160-239 8.5 / 17.15 / 18 and 11.5 / 25.3 / 27. Positions whose move
source lies outside both W(8) and E(8) (legal): 14 / 25 / 26 / 5.

The games: final board count median 44, p95 85, maximum 186 (t0db-0487, 10
timelines; then 178, 146, 142, 135, 116); 256 reach 50 boards, 14 reach 100, 2
reach 160, none passes 186. Positions per game median 38, max 100. Moves by bucket:
87.8% under 50 boards, 11.5% at 50-99, 0.6% at 100-159, 0.1% at 160-186. The 45
refused games, measured up to their refusal, would add 644 / 230 / 2 / 0 positions
and change no verdict.

**The verdict, §15.6's rule as written** (no bar, ceiling or threshold moved). §15.6
does not write the multiplier's formula; the one that reproduces its own table and
its "~1.6 / 2.1 / 3.0 / 4.0" at |E| ≈ 25 is the log-weighted mix
m = m_cached^0.9 · m_dense^0.1, from the N3 cost model (`python/t5d/net/cost.py`:
T = 66N + 16 tokens, attention 4·Tq·Tk·d·L, linear 40·Tq·d²·L, d = 256, L = 16),
so m_cached = (T_N + 2560) / (T_S + 2560) and m_dense = T_N(T_N + 2560) /
(T_S(T_S + 2560)) at scope S = W + median |E| boards. It reproduces the k = 8
ceilings as none / 31.0 / 76.3 / 166.9 / 285.2 (spec: none / 31 / 77 / 169 / 290)
and the k = 4 ceilings 23.7 / 59.0 / 108.3 / 206.9 / 333.2 (23 / 59 / 109 / 209 /
338); the harmonic reading falls 1-8 low, the arithmetic one gives 100 boards a
ceiling the spec says it has not. Confirmation asked of the user.

| Phase | N | k = 8: W + \|E\| | m (cached, dense) | ceiling \|E(8)\| | k = 4: W + \|E\| | m | ceiling \|E(4)\| | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1-49 | 24 | 9 + 0 | **1.445** (1.312, 3.442) | −1.1 | 5 + 1 | **1.603** | 2.9 | on at k = 4 - **inside the ±10% band: STOP** |
| 50-99 | 60 | 19 + 0 | 1.913 (1.707, 5.343) | 12.4 | 11 + 2 | 2.215 | 20.4 | **on at k = 8** |
| 100-159 | 112 | 36 + 2 | 2.184 (1.961, 5.755) | 31.0 | 23 + 3 | 2.686 | 44.0 | **on at k = 8** |
| 160-239 (data to 186) | 169.5 | 19 + 2 | 4.276 (3.474, 27.76) | 87.6 | 11 + 2 | 5.172 | 95.6 | on at k = 8 - **2 games** (t0db-0723 4.4x at 2 timelines; t0db-0487 1.73x at 9-10) |
| 240+ | - | - | - | - | - | - | - | **no data: provisional** |

Pseudo-legal |E| gives the same verdicts (1-49 at k = 4: 1.545). 50-99 and 100-159
are robust: per position at k = 8 the median multiplier is 1.86 and 2.10, and 82%
and 100% of positions clear the bar. 1-49 is not: per position its median is 1.45
at k = 4 (46% clear the bar); under the harmonic formula it would be off; in
10-board phases it splits - off under 20 boards, k = 4 at 20-29 (1.60x), k = 8
from 30 (1.73-1.97x).

**The kill clause** (weight each phase by its unwindowed per-node FLOPs × the
moves spent in it; a 90/10 cached/dense blend, cached-only and dense-only shown):
with 1-49 on (k = 4) the on-phases carry 100% - not killed; with 1-49 off they
carry 21.8% blended (19.6% cached, 41.4% dense; 12.2% of moves) - **killed**; with
10-board phases 69.8% blended (50.7% if 20-29 were also off). It rests on the human
games' phase distribution - 88% of moves under 50 boards, the opposite of the
spec's "the late game dominates" - with nodes per move assumed equal across phases.

**Verdict: on at k = 8 from 50 to 186 boards; under 50 boards inside the ±10%
band, and the kill clause turns on that phase - STOP for the user** (the formula,
the phase definition and the kill clause's distribution: questions in
`docs/BUILD_LOG.md`). Above 186 boards: no public data - provisional (the user's
ruling). The spec's "at ~100 boards it cannot pay at k = 8" does not hold in real
games, which run 4 active timelines there (the table assumes 6).

### The window on the corpus (M7) - superseded by the real games above


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

### Engine E6a - the search's memory accounting and smoke games (2026-10-03)

Dev machine, g++ 16.2 Release, shared. Accounting against a counting `operator new`
(Two Timelines starts after 2-12 actions, random evaluator, 300-2,000 playouts):
the evaluation cache's byte count 1.01-1.17x the heap it holds (the first estimate
was 1.4-2.2x low); `Search::memory_bytes()` 0.99-1.05x. Before the fixes, a
whole-turn node at 103 boards with `max_actions` 4096 held about 1.5 MB (now 256).
Smoke games (`t5d-play`, R8, the uniform evaluator unless named; every record
replays `ok` through t5d-gate):

| Game | Actions | End | Time |
|---|---|---|---|
| Standard, board-move split, tree | 32 | board cap (332 boards) | 56 s |
| Standard, board-move split, graph | 32 | board cap | 56 s |
| Standard, whole turn, tree | 28 | turn cap | 3.8 s |
| Standard, whole turn, graph | 28 | turn cap | 3.5 s |
| Two Timelines, board-move split, graph, match play | 18 | turn cap | 0.25 s |
| Two Timelines, whole turn, tree | 16 | turn cap | 0.9 s |
| Very Small - Open, material evaluator | 7 | checkmate 1-0 | 0.03 s |
| Standard, `--no-moves-after-shift` | 34 | board cap | 54 s |

At large N the time is legality classification (spec issue 95): in a tree-mode
Standard game to 336 boards, 41 of 43 s were leaf expansions with the legality
tiers, the search itself about 2 s.

### Engine E7 - the gate's full mode on the corpus under R8 (2026-10-02/03)

Machine: dev, g++ 16.2 -O2. `t5d-gate --cap 1000` with the default detector
portfolio, one process at a time, 600 s limit per record, the default rules (now
R8). Shared machine: other agents' t5d-scan6b runs during the first run, a long
test beside the second. 41 records: 36.1 s in all (first run), 43.7 s (second).
Slowest records: cwmtt/niceAndrey 14.3 s (19.9 s) for 122 positions, slowest
position 0.38 s; cwmtt/100_timelines 4.9 s (5.5 s), slowest position 1.9 s, 128
MiB working set; cwmtt/wide 4.6 s; cwmtt/manyChecks 4.4 s (slowest position 1.2 s,
101 MiB). Every other record under 2 s; peak private memory 131 MiB. Under
`--no-moves-after-shift` (the Guide-literal reading): 10.4 s, output identical to
E4's default.

### Verification v3: 6b″ (M17) - ADOPT HUNT UNDER 100 BOARDS; GUARD IN NO RANGE

2026-10-04. Engine at d96d6d9 (E9 merged: Guard, Hunt, the asset map, Hunt's
ground truth, `t5d-scan6bpp`), Release, g++ 16.2. Quiet machine: under 1% busy
before the run (all processes, 10 s), during it 15.9% (Guard) and 19.2% (Hunt) -
the tool's own threads. `t5d-scan6bpp --truth engine/testdata/hunt-truth
--record-list build/corpus.txt --per-position` over the 41 corpus records in the
manifest's order (the tool refuses any other list), rules R8/R10 (the default):
the same 805 roots as 6b, read per board-count range, 5 threads, 50 ms of wall
clock per root, Guard and Hunt as they ship (Guard's check filter off, the
attacker probe on). Wall 58.4 s. Raw output `build/6b/6bpp-quiet.*`
(git-ignored; reproducible). M17's tests, unchanged:

- **Guard** - depth 1 completes (safe to depth 1, or refuted at depth 1 with E4's
  confirmation) in ≥ 95% of a range's roots; the chosen action is the record's
  next action (788 roots: 13 records end there, 3 go on with an action the engine
  cannot read or play, 1 next action breaks C4).
- **Hunt** - finds ≥ 50% of the forced mates within 5 half-turns that the long
  offline search established (the pinned ground truth: 125 mates, every one
  confirmed by E4; sha256 `ce62e5f8...`), counted only when E4 confirms Hunt's.

| Boards | Guard: depth 1 completes | completed depth (roots) | refuted | Hunt: truth mates found | Verdict |
|---|---|---|---|---|---|
| < 50 | 65.1% of 513 - FAIL | 0: 218, 1: 266, 2: 16, 3: 2, 4: 1, 8: 10 | 42 (39 at depth 1) | **69.0% (49 of 71) - PASS** | **adopt Hunt** |
| 50-99 | 15.8% of 139 - FAIL | 0: 121, 1: 15, 8: 3 | 4 | **63.6% (21 of 33) - PASS** | **adopt Hunt** |
| 100-199 | 5.2% of 96 - FAIL | 0: 94, 1: 1, 8: 1 | 3 | 35.0% (7 of 20) - FAIL | neither |
| 200+ | 0.0% of 40 - FAIL | 0: 40 | 0 | 0 of 1 - FAIL, too few to read | neither |
| pooled (not a verdict) | 45.8% of 788 | | 49 | 61.6% (77 of 125) | |

Cross-checks against E4 alone (certificates): Guard's 49 refutations and Hunt's
77 mates all confirmed; none refuted, none incomplete. Median nodes per root:
Guard 23,918 / 17,805 / 22,513 / 20,796; Hunt 676 / 24,522 / 24,144 / 60,924.

**The verdicts (M17, applied as pre-registered; no outcome stops the build):**
**Hunt is adopted under 100 boards; Guard is adopted in no range.** §13 records:
the chosen move goes unguarded in every range; from 100 boards up, proofs come only
from the search's solver. Why Guard fails: to say "safe" at depth 1 it must answer
every opponent action, a product over their boards (spec issue 99). Why Hunt fails
at 100-199: all 20 truth mates there are mates in one that Hunt's own class finds
in the long search; its root expansion does not finish in 50 ms (issue 107; a
lazily expanded root is the remedy, not built).

**Hunt's ground truth** (ruling 5; `engine/testdata/hunt-truth/`, recorded at
1d2d0d7, 5,097 s on 6 workers): one long offline search per root - the union of
the literal class's (every first move, every completion) and the default class's
proofs, 4M nodes each, a 2M-node tree cap, one thread and fresh stores per root,
deterministic. The budget from a 101-root curve (mates found against nodes: 5, 6,
8, 8, 9, 11, 11, 11, 12, 12, 12, 12 at 2k ... 4M) that found no new mate past
500k. 825 roots, 805 measured, 125 mates (71 / 33 / 20 / 1 by range; 99 mates in
one, 16 in three, 10 in five; both classes 88, the default alone 32, the literal
alone 5), every one confirmed by E4. Re-recorded with the reviewed prover (4,962
s): every verdict equal.

### The verification redesign: 6b′ (M14, M15) - 50-199 BOARDS FAIL - M15 OUTCOME: STOP

2026-10-03. Engine E8 (the M14 prototype) at bd08c48 on its branch
`worktree-agent-a4f2f93a8a448467b` (main ccdbdbd merged in; not yet merged into
main), Release, g++ 16.2. Quiet machine: under 1% busy before the run (all
processes, measured over 10 s); during it 19.7% (A), 18.5% (C), 6.5% (B) - the
tool's own threads. `t5d-scan6bp --per-position --record-list build/corpus.txt`
over the 41 corpus records, rules R8 (the default): 805 roots and 780 leaves, the
same positions as 6b. Wall 43.9 s. Raw output `build/6b/6bp-quiet.*` (git-ignored;
reproducible). M15's tests, unchanged, read per board-count range:

- **(A) the proof channel** - depth-limited df-pn over the semimove-decomposed turn
  tree, 50 ms of wall clock at 5 threads per root; decided = mate within 5
  half-turns proven, or disproven within the attacker's class (placeholder class:
  checking first moves, canonical completion - spec issue 67); pass ≥ 50%.
- **(B) the estimate channel** - fits the rung-1 leaf budget at p90 (budget = 5 ×
  239,283 nodes/s / §15.6's size-matched evaluation rate = 125 nodes).
- **(C) brute force (the E5c scanner) against df-pn**, 50 ms at 5 threads.

| Boards | Roots | (A) df-pn decided | proven / disproven (no class action at the root) / unknown | (C) brute force decided | Leaves | (B) p90 cost / budget |
|---|---|---|---|---|---|---|
| < 50 | 527 | **77.4% PASS** | 52 / 356 (256) / 119 | 75.9% (400) | 495 | 2.34 - does not fit |
| 50-99 | 140 | **40.7% FAIL** | 23 / 34 (22) / 83 | 30.7% | 140 | 6.33 - does not fit |
| 100-199 | 97 | **23.7% FAIL** | 7 / 16 (12) / 74 | 24.7% | 100 | 15.15 - does not fit |
| 200+ | 41 | **0.0% FAIL** | 0 / 0 / 41 | 2.4% | 45 | 50.58 - does not fit |
| pooled (not a verdict) | 805 | 60.6% | 82 / 406 (290) / 317 | 58.1% | 780 | 6.70 |

Where df-pn's work went: attack-map queries 82-86% of node units in every range,
E4's searches 0.0-0.3%; standing-capture tests cut 1.2-1.5M partial actions per
range below 200 boards; median nodes per root 651 / 31,958 / 30,125 / 85,921. Below
50 boards, 256 of the 356 disproofs are roots with no action of the class at all
(spec issue 68). The estimate channel is 61-87% "tactically live" by range.

Information (E8's runs on a shared machine, not verdicts): 500 ms at 5 threads
lifts (A) to 45.0% / 34.0% / 2.4% above 50 boards - the shortfall is the trees'
size, not a constant factor; the literal question (every first move, every
completion) decides 9.3% / 2.9% / 1.0% / 0.0%; a one-ply estimate channel fits
below 200 boards (p90 0.20 / 0.53 / 0.99) but is no exchange search.

**M15's outcome, applied as pre-registered: 50-199 boards fail -> STOP: "exact
verification covers only small positions, and the design's robustness argument
above them rests on the adversarial league alone - a design review, not a tuning
pass."** Rung 1 stays dropped (the estimate channel fits in no range). Below 50
boards df-pn decides more roots than brute force (408 against 400): df-pn is kept
there.

### The scanner: 6b (M8, M12) - RE-RUN AFTER THE OPTIMISATION PASS: 50-199 BOARDS FAIL - M12 OUTCOME: STOP

**The re-run (M12), 2026-10-03.** Engine at 8841f4b (E5c merged: the bounded
optimisation pass on in-check enumeration and the horizon mate tests, and Lazy
SMP). Quiet machine (busy 5.8% over the run; 17.0% during the 5-thread wall-clock
phase - the tool's own 5 of 24 logical processors). `t5d-scan6b --max-nodes 100000
--jobs 1 --per-position --moves-after-shift` (Q4 ruled "yes", R8) over the 41
corpus records: 805 roots, 780 leaves, median 27 boards, max 378. The scanner at
M13's design budget: 5 threads, Lazy SMP, 50 ms of wall clock per root; rung 1's
budget = 5 threads x 787,018 nodes/s per thread / the evaluation rate (§15.6's
design rates, size-matched). Every test definition, threshold, budget and position
set unchanged from the first run. Read per board-count range (M12); pooled figures
are information. Raw output: `build/6b/e5c-quiet.*` (git-ignored; reproducible).

| Test | < 50 boards | 50-99 | 100-199 | 200+ | pooled (not a verdict) |
|---|---|---|---|---|---|
| (iii) root scan, 50 ms at 5 threads, ≥ 5 forcing half-turns in ≥ 50% | **76.7% PASS** (527) | **31.4% FAIL** (140) | **24.7% FAIL** (97) | **2.4% FAIL** (41) | 58.8% |
| (iii) the first run, same reading (1 thread, before the pass) | 64.7% | 25.7% | 18.6% | 2.4% | 49.2% |
| (iii) 1 thread now, 39,350 nodes (information) | 74.8% | 28.6% | 23.7% | 2.4% | 56.9% |
| (ii) rung 1, 5 threads, size-matched, ≥ 3 in ≥ 50% of leaves | 37.0% FAIL | 6.4% FAIL | 3.0% FAIL | 0.0% FAIL | 25.0% |
| (iv) quiet-leaf reach at that budget, ≥ 50% | 7.0% FAIL | 2.9% FAIL | 2.2% FAIL | 0.0% FAIL | 6.4% |
| (v) stand-pat invalid | 3.6% of 166 - fine | 1 of 1 decided (too few) | - | - | 4.2% |

What the 50-199 passes rest on: at 50-99 boards 44 roots (8 iterations to depth,
10 exhausted trees, 26 proofs); at 100-199, 24 (0, 5, 19). Above 50 boards the
scanner reaches 5 forcing half-turns essentially only where it proves a result.
Where the nodes go now: E4 searches 53.7% (listing in check 23.2%, completion
fallbacks 23.9%, horizon mate tests 6.4%), moves applied 22.2%, royal-capture scans
24.1%; 36.5 nodes per tree position (84 before the pass). By board count the
completion fallbacks lead at 100-199 (38.1% of nodes) and 200+ (63.5%). Speed:
787,018 nodes/s per thread alone, 566,458 per thread at 5 threads (72%).
Stand-pat at 2-3 mandatory boards: 20.0% of 10 decided - on a band edge, from too
few positions to read.

**M12's outcome, applied as pre-registered: 50-199 boards fail -> STOP: redesign
the verification layer.** Rung 1: dropped in every range (its own rule). Quiet-leaf
reach: NNUE v2 and quiet-only training deferred (M8).

**The first run (2026-10-02, before the pass)** - superseded by the re-run above;
kept for the record.

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

## Phase 1 counts with no threshold

### Turn width (M22) on the human games - one mandatory board in 94% of turns under 50 boards, 73% at 50-99, 36% at 100-199; the turns with more than one carry 18%, 43% and 82% of the dense compute (no threshold, no verdict)

**What it is.** Register M22 (Committed, process; spec §15.1's Phase 1 row and §8.2
"Measure first"): mandatory boards per turn and optional moves played, per
board-count range, on the human games, by the rules engine alone - before the D2
tree and flat forms are built (the ply split, D2's default arm, is built). §15.6
sets no threshold for it, so there is no verdict and no STOP band: it informs the
order of the D2 work (register part K) and checks part L's prediction.

**Conditions.** 2026-10-17, machine `dev-r` - counts, not timings, so the shared
machine's load does not matter. The rules engine is `main`'s at f07568c, unchanged;
the counting is `t5d-measure --turns --turn-summary` (`engine/tools/turn_width.hpp`)
at 607e4b1 (lane/engine: M22, 2864be0, and its review fixes; rebuilt at that commit,
the run reproduces the raw output byte for byte), Release, MSYS2 g++ 16.2.0 (Rev3),
`--werror`, 544 of 544 engine tests passing. Data: t0database (`data/t0database/`,
commit `d49da405...`, `Data.csv` sha256 `37fb7c1c...a4e`), converted by `py
data/t0database/to_5dpgn.py build/t0db`
(761 records, 237 empty rows skipped; the source's own `[Board "Standard - Turn
Zero"]`; converted twice, identical file for file), then `build/engine/t5d-measure
--turns --turn-summary build/t0db/*.5dpgn`. Rules: the default ruleset (encoding
`0x0`) - moves after the shift until submission (R8), only the active boards in the
present mandatory (R9), the count rule with reactivation and rewinds (R2, T11, T12),
initial timelines created by neither player (R10) - in parse mode, as M7: C4's
royal-safety test is not made at submission. The per-action counts do not depend on
it (they count the moves the records made), but which records replay to the end
does: replayed with C4, one record changes - t0db-0115, one of the 692 games below,
is refused at its 9th action (a black bishop on (0T5)e6 could capture White's king
on Turn Zero's T0 board by a branching jump; how the T0 board is played is rules
question Q8) - so under C4 the strict reading would hold 691 games, its 18 actions
out (the 8 before its refusal in the refused reading instead); every other record
reads the same way both times. Records: 761 read, **716 replay to the end**, 45
refused as at M7 (13 unfinished final actions, 17 other variants filed under the Turn
Zero label, 2 unknown fairy set-ups, 10 doubled-file pawn captures, 3 notation
faults). Of the 716, 24 equal or are a prefix of another record, by their sequence of
positions (a game recorded twice, or cut short), and are left out: **692 games**,
M7's strict count. Every one of the 29,292 actions read - the refused records' 1,165
included - was played twice, by the replay and move by move from its start, and
reached the same state. The summary was recomputed independently from the
per-action lines (a separate script working from the definitions): 1,648 figures, no
difference. Raw output `build/m22/turns.jsonl` (git-ignored; reproducible). This
section was first written from the tool at 2864be0. Against that run, the review
fixes changed one count - an optional move, at the action's start and when made,
now includes a jump from a mandatory board onto an optional one, which the first
version left out (36 actions in all, 35 of them in the 692 games) - and added
fields: the source-board reading, those jumps' own count, and the summary's shares of
turns with more than one playable board and with an optional board, among others.
Every other value of the output is unchanged, but for the summary's figures of that
count.

**Definitions** (`engine/tools/turn_width.hpp`). A *turn* is one player's action,
from its start to its submission (T1). At the action's start: the board count N
(every board, history boards and Turn Zero's T0 board included - M7's N); the
**mandatory boards**, the active boards in the present on the mover's colour (T6,
R9); the **optional boards**, the mover's other playable boards - ahead of the
present, or on an inactive timeline; the **playable boards**, both together. A move
is made on every board it starts or ends on (T3): its source board and, when it is a
jump that does not branch, its destination too (J4; a branching jump's destination
is a history board, J5). An **optional move** is a move made on a board that was not
mandatory at the action's start - so a jump from an optional board onto a mandatory
one counts, and so does a jump from a mandatory board onto an optional one (register
M22 does not define "optional moves played": build-log spec issue 140). Two other
readings are computed beside it: by the source board alone (it leaves out the jumps
from a mandatory board onto an optional one; it differs on 35 turns of 27,097) and
by the boards' status when each move was made (it differs on 4). A board's status
can change mid-action - a rewind (T8, T12) releases the mandatory boards left
unplayed, and a reactivation (T11, T12) can make a board mandatory, with or without
a rewind; such boards keep their status at the start here, and the moves on them are
counted apart. Ranges by N at the action's start: 6b's (the register names none:
spec issue 141), with ten-board buckets under 100. A range resting on fewer than 10
games is flagged too few to read (issue 118's convention, applied to games).

**Mandatory boards, playable boards and moves per turn** (the strict reading: every
action of the 692 games):

| N at the turn's start | turns | games | mandatory boards: 1 / 2 / 3 / 4+ | more than one | three or more | max | more than one playable board | an optional board at the start (mean per turn) | moves: 1 / 2 / 3 / 4+ | moves: mean / p90 / max |
|---|---|---|---|---|---|---|---|---|---|---|
| < 50 | 24,002 | 692 | **94.2%** / 5.1% / 0.6% / 0.2% | **5.8%** | 0.7% | 7 | **15.1%** | 10.5% (0.12) | 95.9% / 3.8% / 0.3% / 0.1% | 1.05 / 1 / 4 |
| 50-99 | 2,920 | 237 | **72.9%** / 20.4% / 3.8% / 2.9% | **27.1%** | 6.7% | 6 | **65.2%** | 44.5% (0.70) | 80.3% / 15.7% / 2.5% / 1.4% | 1.25 / 2 / 5 |
| 100-199 | 175 | 13 | **36.0%** / 48.0% / 2.3% / 13.7% | **64.0%** | 16.0% | 8 | **99.4%** | 65.7% (1.65) | 49.1% / 39.4% / 2.9% / 8.6% | 1.77 / 3 / 7 |
| 200+ | 0 | 0 | no data | | | | | | | |
| pooled (context) | 27,097 | 692 | 91.5% / 7.0% / 0.9% / 0.5% | 8.5% | 1.5% | 8 | 21.0% | 14.5% (0.20) | 93.9% / 5.3% / 0.6% / 0.3% | 1.07 / 1 / 7 |

The human games' turns lie 88.6% under 50 boards, 10.8% at 50-99 and 0.6% at
100-199; no turn starts above 178 boards (the last positions reach 186). A game
counts in a range when one of its turns starts there - so 237 and 13 games against
M7's 256 and 14, which counted the last position too. A turn with one playable board
is one mandatory board and nothing else: one move, then the submission (84.9% of the
turns under 50 boards, 34.8% at 50-99, 0.6% at 100-199).

**Optional moves and the rest of the turn** (strict):

| N | turns with an optional move | optional moves per turn (mean) | the same by the source board alone | jumps from a mandatory board onto an optional one | jumps from an optional board onto a mandatory one | jumps between two mandatory boards | with a jump | with a branch | rewound (T8, T12) | a mandatory board released | turns with a move after the shift (R8) | boards made mandatory mid-turn (T11, T12) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 50 | 1.2% (280 with one, 4 with two) | 0.012 | 1.1% | 24 | 29 | 229 | 5.3% | 4.1% | 3.8% | 1.8% (511 boards) | 11 | 2 |
| 50-99 | 2.5% (72, one each) | 0.025 | 2.1% | 10 | 12 | 149 | 15.1% | 9.4% | 8.6% | 5.8% (259) | 0 | 0 |
| 100-199 | 1.1% (2, one each) | 0.011 | 0.6% | 1 | 0 | 25 | 21.7% | 7.4% | 5.7% | 5.7% (25) | 0 | 0 |

No turn makes more than two optional moves. By the boards' status when each move was
made the shares are 1.2% / 2.5% / 1.1% too. The source-board reading is the one this
section first reported (1.1% / 2.1% / 0.6%).

**Ten-board buckets under 100** (strict):

| N | turns | games | one mandatory board | more than one | three or more | more than one playable board | one move | moves (mean) | an optional move |
|---|---|---|---|---|---|---|---|---|---|
| 1-9 | 5,524 | 692 | 99.98% | 0.02% | 0 | 0.07% | 99.98% | 1.000 | 0 |
| 10-19 | 6,551 | 688 | 99.0% | 1.0% | 0.03% | 5.7% | 99.4% | 1.006 | 0.6% |
| 20-29 | 5,504 | 648 | 93.8% | 6.2% | 0.2% | 16.7% | 95.8% | 1.043 | 1.7% |
| 30-39 | 3,901 | 535 | 87.3% | 12.7% | 1.9% | 30.5% | 90.5% | 1.107 | 2.2% |
| 40-49 | 2,522 | 395 | 80.3% | 19.7% | 3.3% | 45.0% | 86.0% | 1.163 | 2.4% |
| 50-59 | 1,418 | 237 | 76.3% | 23.7% | 4.4% | 53.9% | 83.1% | 1.200 | 2.4% |
| 60-69 | 738 | 136 | 73.2% | 26.8% | 7.9% | 69.9% | 80.6% | 1.267 | 2.8% |
| 70-79 | 387 | 73 | 65.4% | 34.6% | 9.3% | 78.6% | 74.9% | 1.310 | 2.3% |
| 80-89 | 241 | 42 | 67.6% | 32.4% | 10.8% | 83.8% | 74.7% | 1.365 | 2.5% |
| 90-99 | 136 | 28 | 66.2% | 33.8% | 8.8% | 86.8% | 75.7% | 1.331 | 1.5% |

**By mandatory boards at the start** (strict; moves per turn):

| N | mandatory | turns (games) | moves: 1 / 2 / 3 / 4+ | moves: mean / median / max | with a branch | rewound |
|---|---|---|---|---|---|---|
| < 50 | 1 | 22,605 (692) | 22,592 / 11 / 2 / 0 | 1.00 / 1 / 3 | 3.2% | 3.0% |
| < 50 | 2 | 1,226 (285) | 361 / 864 / 0 / 1 | 1.71 / 2 / 4 | 15.7% | 13.9% |
| < 50 | 3 | 133 (57) | 43 / 27 / 63 / 0 | 2.15 / 2 / 3 | 33.1% | 33.1% |
| < 50 | 4+ | 38 (14) | 10 / 2 / 7 / 19 | 2.92 / 3.5 / 4 | 28.9% | 28.9% |
| 50-99 | 1 | 2,128 (201) | 2,126 / 2 / 0 / 0 | 1.00 / 1 / 2 | 6.6% | 5.9% |
| 50-99 | 2 | 597 (130) | 168 / 429 / 0 / 0 | 1.72 / 2 / 2 | 13.4% | 12.1% |
| 50-99 | 3 | 110 (38) | 29 / 24 / 56 / 1 | 2.26 / 3 / 4 | 26.4% | 26.4% |
| 50-99 | 4+ | 85 (28) | 23 / 4 / 18 / 40 | 2.92 / 3 / 5 | 29.4% | 29.4% |
| 100-199 | 1 / 2 / 3 / 4+ | 63 (9) / 84 (8) / 4 (2) / 24 (4) | every class under 10 games - **too few to read** | 1.00 / 1.80 / 1.75 / 3.71 (means) | | |

A turn with k mandatory boards makes fewer than k moves when a jump between two of
them plays both (A6's "two fewer") or a branch rewinds the present and releases the
rest (T8): under 50 boards 450 turns do, 228 by such a jump and 222 by a rewind; at
50-99, 269 turns, 145 and 125 (one both). Played with a single move from two or more
mandatory boards: 414 turns under 50 boards (194 by a jump between two of them, 220
by a rewind), 220 at 50-99 (97, 123) and 23 at 100-199 (14, 9).

**Sensitivity.** Grouping the records into games as M7's tool does (674 games: two
endings of one game joined, each distinct action counted once): under 50 boards
23,773 turns, 94.1% with one mandatory board, 5.9% with more, 15.2% with more than one
playable board, 95.8% one move; 50-99 and 100-199 unchanged. The 45 refused records'
1,165 actions before their refusal (not de-duplicated; 943 / 221 / 1 by range) would
move none of the shares above by more than 0.9 points (50-99: more than one playable
board 65.2% -> 66.1%, more than one mandatory board 27.1% -> 27.7%, one move 80.3% ->
79.7%).

**Part L's prediction** - "Most human-game turns under 50 boards have exactly one
mandatory board" (§8.2): **holds**. 94.2% of the turns under 50 boards (22,605 of
24,002, in 692 games), 94.1% under the grouped reading, 94.2% with the refused records'
actions added. §8.2's own claim beside it - "most turns there are single moves, where
every arm is identical" - holds too, read by the turn's start as §8.2 derives it (from
"one active timeline") and as its cost argument counts ("a turn with k mandatory
boards costs k evaluations"): 84.9% of the turns under 50 boards start with one
playable board, one move and a submission under every arm. 95.9% of them are played
with one move, but that counts what the players chose, not what the search faces
(below).

**What it says about the D2 arms** (a reading, not a verdict: nothing maps the
figures to an order of work - spec issue 142). Where the arms differ is set at the
turn's start, not by the moves the player went on to make. A turn that starts with one
playable board is one move and a submission whatever the turn construction. Every
other start is searched differently by the arms: the ply split evaluates every node
inside the turn - "a turn with k mandatory boards costs k evaluations along every
path" (§8.2) - and, once the present has shifted, offers submit beside each further
optional move; the tree form inherits the turn-start priors for the boards still to
move; the flat form gives each optional board a "leave it" option. Turns that start
with more than one playable board: **15.1%** of the human games' turns under 50
boards, **65.2%** at 50-99, **99.4%** at 100-199 (13 games). Of all turns, those with
more than one mandatory board - §8.2's cost case - are 5.8%, 27.1% and 64.0%; those
with one mandatory board and optional boards beside it 9.3%, 38.1% and 35.4%. A turn
played with one move is not a narrow turn: 414 turns under 50 boards, 220 at 50-99 and
23 at 100-199 start with two or more mandatory boards and are played with a single
move - a jump between two of them (J4) or a rewinding branch (T8) - so the turns of
more than one move (4.1%, 19.7%, 50.9%) understate where the arms differ. Part L's
other prediction puts the turn-start-only forms ahead once turns have three or more
mandatory boards: 0.7%, 6.7% and 16.0% of turns. Optional moves are rarely played -
1.2%, 2.5% and 1.1% of turns, never more than two in a turn, moves after the shift
(R8) in 11 turns of 27,097 - but optional boards are there, to be played or left, at
10.5%, 44.5% and 65.7% of turn starts, and every arm must decide them whatever the
player chose. Nothing is measured at 200 boards and up, where §8.2's cost argument
mostly applies: the pilot self-play run (M18) is the first source of the engine's own
distribution there.

**The CHECKPOINT 8 figures (2026-10-23): mandatory boards, optional moves played and
the compute share of multi-board turns.** The user's ruling on spec issue 142: "Your
figures count playable boards, which include optional ones. Per range, also report:
the share of turns with more than one and with three or more mandatory boards, optional
moves actually played, and the compute share of multi-board turns (moves x per-move
cost)." The figures of 2026-10-17 that the ruling quotes - 5.8%, 27.1% and 64.0% -
were already the shares with more than one **mandatory** board (the tables above); the
playable shares are 15.1%, 65.2% and 99.4%. Spec §8.2's CHECKPOINT 8 paragraph and
register M22 and part L carry the two swapped (build-log spec issue 156). Below, each of
the four figures the ruling names, per range.

*Conditions.* 2026-10-23, machine `dev-r` - counts and arithmetic, not timings, so the
shared machine's load does not matter. The engine is `main`'s at 7f5f853 (after H51,
across which M22's command was byte-identical); the counting is `t5d-measure --turns
--turn-summary` with this entry's additions to `tools/turn_width.hpp` (the commit that
adds this paragraph, on 7f5f853), Release, MSYS2 g++ 16.2.0 (Rev3), `--werror`, 583 of
583 engine tests passing. Data and command as above: t0database at `d49da405...`
(`Data.csv` sha256 `37fb7c1c...`), converted afresh by `py data/t0database/to_5dpgn.py
build/t0db` (761 records, 237 empty rows skipped; identical, file for file, to a second
conversion), then `build/engine/t5d-measure --turns --turn-summary build/t0db/*.5dpgn` -
run twice, the same bytes. Rules: the default ruleset (`0x0`), parse mode, as above.
Records and games as above: 716 replay, 692 games, 27,097 turns, 29,073 moves. Every
line of the output that existed before - each turn's counts and every earlier summary
field - is unchanged; the new fields are `moves_total`, `optional_moves_total`,
`optional_moves_share_of_moves`, `compute_share_{mandatory_over_1, mandatory_over_2,
playable_over_1}` and their `_constant` readings, `dense_tflops`, and the head line's
`cost_model`. The figures were recomputed independently from the per-turn lines (a
separate script working from the definitions and the cost formula): 15 ranges x 16
figures, no difference.

*The cost of a move* (`tools/turn_width.hpp`, "THE COST OF A MOVE"). A turn's compute
is its moves x the cost of one move. Under the ply split - D2's built arm and default -
"every node inside a turn gets its own network evaluation, so a turn with k mandatory
boards costs k evaluations along every path" (§8.2), so a move costs one evaluation.
The documents leave its size open; two readings:

- **dense (the figure)**: one evaluation of the dense network - v1 is dense (§7: "Build
  dense") - at the turn's board count N, by §9.1-9.2's analytic model as stage N3's cost
  model computes it (`python/t5d/net/cost.py`, `dense_flops`; the window verdict above
  uses the same model): 4 T^2 d L + 40 T d^2 L FLOPs, T = 66N + 16 tokens, d = 256,
  L = 16 - 0.036 TFLOPs at 10 boards, 0.99 at 100, 4.78 at 240 (§9.2's figure). It
  weighs a move at 100 boards 28 times one at 10.
- **constant (the sensitivity reading)**: every move costs the same, so the compute
  share is the share of the moves.

N is the count at the turn's start for each of its moves (each move adds a board or
two, so a long turn's later moves are slightly understated - the multi-board shares, if
anything, too). The submission costs nothing. These are the shares of the **played
line's** compute - the moves the players made, each costed as the ply split costs a
node - not of a search: they stand for the ply split's search in proportion only if its
paths through a turn are as long as the turn played. What the turn-start forms save is
6x's to measure.

| N at the turn's start | turns | moves | more than one mandatory board | three or more | (more than one playable board) | turns with an optional move | optional moves per turn (mean) | optional moves, share of all moves | compute share of turns with more than one mandatory board: dense / constant | three or more: dense / constant | (more than one playable board: dense / constant) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| < 50 | 24,002 | 25,110 | **5.8%** | **0.7%** | 15.1% | **1.2%** | **0.012** | **1.1%** (288) | **18.0%** / 9.9% | 3.3% / 1.6% | 32.4% / 18.8% |
| 50-99 | 2,920 | 3,653 | **27.1%** | **6.7%** | 65.2% | **2.5%** | **0.025** | **2.0%** (72) | **43.4%** / 41.7% | 14.9% / 13.6% | 75.5% / 72.2% |
| 100-199 (13 games) | 175 | 310 | **64.0%** | **16.0%** | 99.4% | **1.1%** | **0.011** | **0.6%** (2) | **81.9%** / 79.7% | 33.4% / 31.0% | 99.8% / 99.7% |
| 200+ | 0 | 0 | no data | | | | | | | | |
| pooled (context) | 27,097 | 29,073 | 8.5% | 1.5% | 21.0% | 1.3% | 0.013 | 1.2% (362) | **33.3%** / 14.7% | 10.3% / 3.4% | 54.4% / 26.4% |

Ten-board buckets under 100 (strict; within a bucket N barely varies, so the dense and
constant shares nearly agree):

| N | turns | moves | more than one mandatory | three or more | turns with an optional move | optional moves per turn | optional moves / all moves | compute share, more than one mandatory: dense / constant | three or more: dense / constant |
|---|---|---|---|---|---|---|---|---|---|
| 1-9 | 5,524 | 5,525 | 0.02% | 0 | 0 | 0 | 0 | 0.06% / 0.04% | 0 / 0 |
| 10-19 | 6,551 | 6,593 | 1.0% | 0.03% | 0.6% | 0.006 | 0.6% (41) | 1.9% / 1.6% | 0.10% / 0.08% |
| 20-29 | 5,504 | 5,740 | 6.2% | 0.2% | 1.7% | 0.017 | 1.7% (96) | 10.5% / 10.0% | 0.5% / 0.4% |
| 30-39 | 3,901 | 4,318 | 12.7% | 1.9% | 2.2% | 0.023 | 2.0% (88) | 21.4% / 20.9% | 4.2% / 4.0% |
| 40-49 | 2,522 | 2,934 | 19.7% | 3.3% | 2.4% | 0.025 | 2.1% (63) | 31.0% / 30.9% | 6.6% / 6.6% |
| 50-59 | 1,418 | 1,701 | 23.7% | 4.4% | 2.4% | 0.024 | 2.0% (34) | 36.4% / 36.3% | 8.7% / 8.6% |
| 60-69 | 738 | 935 | 26.8% | 7.9% | 2.8% | 0.028 | 2.2% (21) | 42.2% / 42.1% | 16.8% / 16.8% |
| 70-79 | 387 | 507 | 34.6% | 9.3% | 2.3% | 0.023 | 1.8% (9) | 50.1% / 50.1% | 16.6% / 16.8% |
| 80-89 | 241 | 329 | 32.4% | 10.8% | 2.5% | 0.025 | 1.8% (6) | 50.5% / 50.5% | 22.7% / 22.8% |
| 90-99 | 136 | 181 | 33.8% | 8.8% | 1.5% | 0.015 | 1.1% (2) | 50.2% / 50.3% | 17.9% / 18.2% |

Where the compute lies, by range (strict, dense): under 50 boards 54.4% of it (86.4% of
the moves, 88.6% of the turns), at 50-99 36.0% (12.6%, 10.8%), at 100-199 9.6% (1.1%,
0.6%); 4,911 TFLOPs along the 692 games' played lines in all. The optional move is the
measured one of issue 140 (a move made on a board not mandatory at the turn's start,
either of its boards); its other readings are in the table above.

*Sensitivity.* Grouped into games as M7 does (674 games): under 50 boards more than one
mandatory board 5.9%, three or more 0.7%, an optional move in 1.2% of turns and 1.2% of
moves, compute share 18.1% / 10.0% (three or more 3.3% / 1.6%); 50-99 and 100-199
unchanged; pooled 33.4% / 14.8%. The 45 refused records' 1,165 actions added: no figure
of the table moves by more than 0.7 points (50-99: compute share 43.4% -> 44.1%, more
than one mandatory board 27.1% -> 27.7%).

*Read against part L and §8.2.*

- Part L, "Most human-game turns under 50 boards have exactly one mandatory board":
  **holds**, 94.2% (unchanged). The register's outcome note says "at most one playable
  board at 94.2%" - that figure is one mandatory board; one playable board is 84.9%
  (issue 156).
- Part L, the turn-start-only forms ahead "once turns have three or more mandatory
  boards" (6x decides it; nothing here tests it): on the human games that regime is
  0.7%, 6.7% and 16.0% of turns, 1.6%, 13.6% and 31.0% of moves, and **3.3%, 14.9% and
  33.4% of the dense compute** - pooled, 1.5% of turns, 3.4% of moves and **10.3% of
  the compute**. Where the prediction applies, it bears on a tenth of the human games'
  compute; above 200 boards, unmeasured, it may bear on more.
- §8.2, "under 50 boards, where the human games spend 88% of their moves": 86.4% of the
  moves (88.6% of the turns; M7 counted 87.8% of positions) - but 54.4% of the dense
  compute.
- §8.2, "at least 94% of turns are single-board and the arms nearly coincide": 94.2%
  have one mandatory board; 84.9% one playable board, where the arms are identical.
  The turns with more than one mandatory board there are 9.9% of the moves and 18.0% of
  the dense compute; those with more than one playable board 18.8% and 32.4%.
- §8.2, "the difference concentrates where the compute is": **holds under the dense
  model**. Pooled, turns with more than one mandatory board are 8.5% of turns, 14.7% of
  moves and 33.3% of the dense compute; within each range the dense share exceeds the
  turn share (18.0% against 5.8%, 43.4% against 27.1%, 81.9% against 64.0%). Under a
  constant cost per move the concentration is the moves' alone (14.7% pooled).
- §8.2, "Playable boards include optional ones, so these bound the mandatory share from
  above": the figures it quotes are the mandatory shares themselves; the playable ones
  are 15.1%, 65.2% and 99.4% (issue 156).
- Optional moves actually played: 1.2%, 2.5% and 1.1% of turns, 1.1%, 2.0% and 0.6% of
  moves (362 of 29,073 in all), never more than two in a turn - against optional boards
  at 10.5%, 44.5% and 65.7% of turn starts.

### D2's tree form against the ply split - evaluation counts on seven human-game turn starts (indicative; no threshold, no verdict)

**What it is.** What the tree form (register D2; spec §8.2, "Evaluating at turn
boundaries") buys at equal evaluations: the ply split evaluates the states between a
turn's moves, the tree form only where an interaction gate fires (build-log spec
issues 157-161), so more of a fixed evaluation budget reaches complete turns. Counts
from a handful of positions with the built-in random evaluator - **indicative**: they
show how much the gates' readings decide (issue 160), not what a trained network's
search would do. No §15.6 threshold; 6x (with 6g, 6k) compares the arms at equal
wall-clock, which these counts do not measure (proposed edit 130).

**Conditions.** 2026-10-23, machine `dev-r` - counts, not timings: deterministic, so
the shared machine's load does not matter. The search is lane/engine's at the D2
tree form's review fixes (the tree form, f2b4679, and the fixes; a fresh root's
search is unchanged by them - the counts below are, value for value, those the tree
form's commit message quoted from an uncommitted scratch harness). The counting is
`build/engine/t5d-tree-count build/t0db/*.5dpgn` (`engine/tools/t5d_tree_count.cpp`;
its header defines every count), Release, MSYS2 g++ 16.2.0 (Rev3), `--werror`. Data:
t0database (`data/t0database/`), converted by `py data/t0database/to_5dpgn.py
build/t0db` (761 records), read in the glob's order. **Positions**: the first turn
start at action 6 or later, not a game end, of each record that fills a class still
wanting one - three with two mandatory boards (the k-th with at least 20k boards),
three with three or more, one with one board and 30 boards or more; one per record;
seven found in the first 32 records, none refused. **Arms**: the ply split; the tree
form with §8.2's four gates at their placeholder readings; the tree form with no
gate; the tree form with the present gate read as rewinds only (`tree_present`
`Rewind`). Each a fresh search: the random evaluator (seed 1), synchronous (one leaf
per evaluation), no evaluation cache, square features off, every other parameter the
search's default (budgeted legality, widening on). Two runs per arm: to 400
evaluations (playouts unbounded), and to 400 playouts.

**At 400 evaluations - the turn starts the network evaluated** (the rest are mid-turn
leaves; the root counts as a turn start), and the deepest evaluated turn start (turns
below the root):

| Record, action | boards | mandatory | ply split | tree form, §8.2's gates | tree form, present = rewind | tree form, no gate |
|---|---|---|---|---|---|---|
| t0db-0001, 28 | 30 | 1 | 122 (7 deep) | 142 (6) | 290 (8) | 400 (8) |
| t0db-0003, 57 | 60 | 2 | 93 (3) | 123 (4) | 249 (5) | 400 (6) |
| t0db-0005, 26 | 34 | 4 | 74 (4) | 111 (4) | 218 (4) | 400 (7) |
| t0db-0007, 32 | 40 | 2 | 89 (4) | 128 (5) | 203 (6) | 400 (7) |
| t0db-0013, 50 | 60 | 2 | 91 (3) | 115 (3) | 276 (5) | 400 (5) |
| t0db-0017, 26 | 33 | 3 | 87 (4) | 121 (4) | 173 (5) | 400 (6) |
| t0db-0032, 29 | 37 | 4 | 81 (4) | 114 (4) | 227 (6) | 400 (6) |
| **range** | | | **74-122 (18-31%)** | **111-142 (28-36%)** | **173-290 (43-73%)** | **400 (100%)** |

**At 400 playouts - the mid-turn leaves by the gate that fired first** (between /
capture / check / present) under §8.2's gates: 3/2/2/251, 14/6/5/252, 4/2/2/264,
37/1/0/247, 24/13/13/229, 36/0/9/244, 28/1/1/251 (the rows' order above) - **the
literal present gate is 229-264 of each 400**, the other three together 7-50. Read as
rewinds only, the present gate fires 58-129 times and the check gate up to 92. The
nodes inherited (expanded without the network) at 400 playouts: 87-204 under §8.2's
gates, 269-501 with the present gate as rewinds, 571-840 with no gate; at 400
evaluations the arms made 400-425 playouts (a simulation that ends at a game end
needs no evaluation).

*Read against §8.2 and issue 160.* With the literal present gate the tree form
spends 28-36% of its evaluations on turn starts against the ply split's 18-31% - at
one mandatory board its shift is the turn's last move, so the two nearly coincide -
because the present's advance at every action's last mandatory move (T5) is itself a
gate: a turn costs at least two evaluations. Read as rewinds only, 43-73%; with no
gate every evaluation is a turn start's. The reading of "a shift of the present"
decides most of what the tree form buys (issue 160, for the user before 6x).

### D2's flat form beside them - the same seven turn starts (indicative; no threshold, no verdict)

**What it is.** The same counts for D2's flat form (whole legal turns generated best
first, build-log spec issues 163-171): every node is a turn start, so every evaluation
is one; what it shows is how deep a fixed evaluation budget reaches, and what the
generator does - turns generated, generators stopped by their budget, fallbacks to E4's
listing. Indicative, as above: the built-in random evaluator (whose leave logits are
random too), not a trained network.

**Conditions.** As above (2026-10-23, `dev-r`, counts, deterministic): lane/engine with
the flat form's search arm (the commit that retires the whole-turn arm), the same
`t5d-tree-count build/t0db/*.5dpgn`, its fifth arm `flat form` (`TurnSplit::Flat`, every
flat parameter at its placeholder: successors, the generated-turn softmax, jumps on both
boards, leave share 0.5, k0 cap 16, 200,000 generator nodes, fallback 64); the four
earlier arms' counts are unchanged, value for value.

| Record, action | boards | mandatory | turn starts evaluated (400 evals) | deepest evaluated turn start | turns generated | generators stopped (all fell back) |
|---|---|---|---|---|---|---|
| t0db-0001, 28 | 30 | 1 | 400 | 11 | 10,699 | 2 |
| t0db-0003, 57 | 60 | 2 | 400 | 10 | 10,852 | 0 |
| t0db-0005, 26 | 34 | 4 | 400 | 12 | 11,812 | 8 |
| t0db-0007, 32 | 40 | 2 | 400 | 11 | 10,819 | 0 |
| t0db-0013, 50 | 60 | 2 | 400 | 10 | 11,329 | 0 |
| t0db-0017, 26 | 33 | 3 | 400 | 10 | 11,508 | 5 |
| t0db-0032, 29 | 37 | 4 | 400 | 12 | 11,357 | 2 |

400-407 playouts per 400 evaluations; about 27 turns generated per node (k0's cap 16,
then widening). *Read against the table above:* the flat form's deepest evaluated turn
start lies 10-12 turns down where the ply split's lies 3-7 and the tree form's 3-6 with
§8.2's gates (3-8 across its three gate readings; the D2 row in docs/BUILD_LOG.md quotes
the first) - at
equal evaluations it searches far deeper in turns, and correspondingly narrower in each
(widening instantiates ceil(16 sqrt(N + 1)) turns of hundreds); 0-8 of the ~400
generators were stopped by their budget before their first turn and took E4's listing
(issue 171). No timings here: the generator's cost per node is not measured on a quiet
machine (proposed edit 131).

### D2's flat form: a point's realisations, and the memory a flat node holds (counts and heap; no threshold, no verdict)

**What it is.** (a) The turn generator's cost on one point of many realisations: k kings,
each able to step back one turn on its own timeline (a branch), the move scored 0 and every
other option -10, so the turn making all k branches is one point whose k! creation orders
are k! legal turns of the top score (J9). The generator records at most 32 realisations per
step; this counts what generating all k! costs. (b) The memory a flat-form node holds - its
generator's queued boxes, successor states and tables - against the ply split's: the
search's estimate (`memory_bytes`) and the heap it holds.

**Conditions.** Counts and heap bytes - deterministic, so the shared machine's load does not
matter. 2026-10-23, machine `dev-r`, Release, MSYS2 g++ 16.2.0 (Rev3), `--werror`.
(a) The position of `turngen_a_point_of_many_realisations_walks_each_once`
(`engine/tests/test_turngen.cpp`; 4x4 boards, timelines -3..+3, the default rules - the
FreeOrder realisation - and, at k = 5, Q3's play activation - the general realisation), the
generator unbudgeted, both units; *before*: reviewer A's probe of the review (a scratch
`rewalk3.cpp` against lane/engine 95d23cc, the same position and scorer); *after*: the test
at the review fixes. (b) `mcts_the_memory_estimate_follows_the_heap`
(`engine/tests/test_mcts_memory.cpp`): Standard - Two Timelines after six actions (each the
hypercuboid listing's first), the random evaluator (seed 7), batch 8, square features off,
no evaluation cache, every other parameter the default; 400 playouts under the flat form,
2,000 under the ply split; the heap counted by the tests' allocation hook. *Before*: as the
flat form's search arm (3251857) recorded it in build-log issue 171; *after*: the test's
printout at the review fixes.

| (a) k = 7, default rules | realisations tested | nodes | of 5,040 emitted under 200,000 nodes |
|---|---|---|---|
| before (each step judged the point again from its first realisation) | 401,936 | 1,495,064 | 1,824 |
| after (the point resumed where the cap stopped it; 157 resumptions) | 5,040 | 19,912 | 5,040 |

At k = 5 under Q3 (general realisation), after: 120 realisations, 120 tested, 510 nodes, 3
resumptions - as actions and as successors alike. Before, the cost grew about as n^2 / 64
for n realisations (each of the n / 32 steps walking up to n); after, linearly (each
realisation walked once, a resumption replaying one path of k moves).

| (b) Two Timelines + 6, tree (not graph) | nodes | estimate | heap | per node (heap) | estimate / heap |
|---|---|---|---|---|---|
| flat form, before (3251857, issue 171) | 400 | - | - | about 137 KB | x1.23 |
| flat form, after | 400 | 65,164,477 | 54,512,986 | 136 KB | x1.20 |
| ply split, after | 2,000 | 7,916,856 | 7,603,136 | 3.8 KB | x1.04 |

*Read against issue 171.* The quadratic cost of a many-realisation point was the review's
finding A-M1: under the search arm's default budget most of such a point's turns were never
emitted; resumed, all are, at about four nodes a realisation. A flat node holds about 36
times a ply node's memory (its generator lives as long as the node can widen); the estimate
stays within the test's 0.75-1.33 of the heap. For 6x: the memory per node by board-count
range (proposed edit 131).

### G2's FP8 emulation on random weights - numerics only (no threshold, no verdict; not 6w)

**What it is.** How far G2's FP8 E4M3 trunk (`NetConfig.trunk_quant = "fp8_e4m3"`, emulated -
`python/t5d/net/quant.py`, spec issue 238) moves the network's outputs from the same weights
unquantised: a plumbing figure for the emulation, not ablation 6w's (trained networks, a GPU,
match play) and not a speed (none is possible on this CPU, M19). Machine `dev-r`; PyTorch 2.13.0+cpu;
2026-10-23; one position per batch; random weights at initialisation scale; window none.

| Network | Precision | Positions | Trunk RMS (max) | Value logits (max) | Policy logits (max) | WDL's W - L (max) |
|---|---|---|---|---|---|---|
| F1 (16 layers), seeds 0, 5, 11 | BF16 trunk, FP8 attention included | `synthetic_state(30 + N, N)`, N = 2, 17 | 2.8-3.2e-2 (0.145) | 4.9e-2 | 0.120 | 6.4e-3 |
| the same | BF16, projections only | the same | 2.8-3.1e-2 (0.157) | 5.0e-2 | 0.129 | 4.7e-3 |
| the same | FP32 trunk, FP8 attention included | the same | 2.8-3.1e-2 (0.162) | 4.3e-2 | 0.143 | 6.0e-3 |
| the same | FP32, projections only | the same | 2.7-3.1e-2 (0.152) | 4.8e-2 | 0.125 | 6.3e-3 |

About 12x the BF16-vs-FP32 gap of the same network (2.4e-3, `tests/net/test_precision.py`); the
attention operands add little to the projections' share. The test bounds
(`tests/net/test_quant.py`) sit 3-4x above. The paths' agreement under FP8, the cache's and the
batch independence are in issue 238(b), (h), (i).

**The cache with FP8 frozen K/V** (the precision review fixes, 2026-10-23; same machine and
build): the cached path against the full pass, the projections alone in FP8 (`trunk_quant_attention
= False`), FP32 trunk, 2 layers, block-causal scope, model seeds 0-2, `test_cache.py`'s standard
trees, seeds 0-3, five moves each; worst over the 12 paths:

| Frozen K/V | Trunk RMS (max) | Value logits (max) | Policy logits (max) |
|---|---|---|---|
| FP8 (C15) | 2.81e-3 (5.52e-2) | 1.42e-3 | 1.50e-2 |
| exact ("match") | 2.85e-3 (5.33e-2) | 2.21e-3 | 2.51e-2 |

Rounding flips of a different FP32 summation order, as in issue 238(i); the test bounds (6e-3,
8e-3, 6e-2) hold for both. (The "match" row is above issue 238(h)'s 1.9e-3 / 1.8e-2 under what
were described as the same conditions - the worst case over a sample of flips; the bounds stand.)

**QAT's gradients against the unquantised network's** (G8, `tests/train/test_qat.py`; 2026-10-23,
same machine and build): SMALL network (`netplay.SMALL`), FP32 trunk, model seeds 0-2, one batch of
`_synthetic_rows(0)`, the total loss's gradient on each trunk projection weight with QAT on
against off, the same weights. Relative L2 distance at most 0.1162 (dense scope) and 0.1171
(block-causal scope, FP8 frozen K/V); norm ratios 0.968-1.029. The test's bounds: distance < 0.5,
ratio 0.5-2. With the estimator the STE replaced (autograd through the cast) every one of those
gradients is zero.

**Batch independence under FP8, per seed** (`test_an_evaluation_does_not_depend_on_its_batch`;
2026-10-23, same machine and build): FP32 trunk, 2 layers, model seeds 0-5, a position alone
against the same position second in a padded batch of two; trunk RMS over its tokens, per seed.
"pow2": 1.8e-8, 0, 0, 0, 0, 5.2e-7. "absmax": 5.2e-4, 2.0e-3, 1.9e-3, 3.1e-3, 4.8e-4, 2.6e-4.
Mutants (the review's plugin): one input scale per batch in the projections - "pow2" 1.2e-6, 0, 0,
5.6e-8, 2.1e-4, 5.2e-7, "absmax" 2.2-2.6e-2; one attention-operand scale per head over the batch -
"pow2" as unmutated, "absmax" 1.9-2.2e-2. So under "pow2" the granularity shows at one seed in
six; the test's bounds are 1e-5 ("pow2", over all six seeds) and 1e-2 ("absmax"), and the
granularity is checked directly elsewhere (issue 238(b)).
