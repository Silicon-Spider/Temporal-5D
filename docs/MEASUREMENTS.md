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
| dev-r-amd | `dev-r`'s AMD Radeon RX 9070 XT (gfx1201, 16 GB, 32 CUs), AMD Software: Adrenalin Edition 26.10.1, Windows driver 32.0.32015.7029 (2026-10-18) | `dev-r`'s Windows; AMD's PyTorch for Windows from ROCm 10.1.0: torch 2.14.0+rocm10.1.0 (HIP 7.16.26385, MIOpen 3.6.1), no Triton; Python 3.14.7 in a separate venv under `build/`; NumPy 2.5.3 - G9's bounded attempt only (2026-10-23), every figure AMD-labelled |

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
| Board-layout embedding (A8) | ~0.1% of attention at 44 boards; ~1.6M parameters at 16 channels | projection 1,573,888 parameters (1,572,864 without its bias), 0.138 GFLOP at 44 boards - **0.099%** of attention by §15.3's formula (0.072% of the counted kernel attention); with the register's FFN (8x, a placeholder) 2,629,120 parameters and 6.29 GFLOP - **4.5%** of attention, 2.0% of the counted forward (0.83% / 0.62% at 240 boards); in the cached configuration A8 adds **10.1%** / **14.4%** to a node (44 / 240 boards: every token embedded at every node); CPU wall indicative only: the embedding +18 ms at 44 boards, +103 ms at 240 (BF16), the forward's change inside the noise | A8 section below; FLOPs counted (load-independent); wall time on a loaded machine (85-100% CPU) | **45x - flagged** (A8 as the register defines it, its FFN at the placeholder's 8x; at least 6.6x with any per-token FFN); the projection alone 0.99 (FLOPs); parameters 1.64x (the projection's 0.98) | 2026-10-23, N-A8 (review fixes the same day) |
| Piece-centric tokens (A5) - no §15.3 row (proposed edit 197); §14.4's figures | 2.3x fewer tokens midgame, 2.7x in sparse endgames; ~250 destination tokens at 240 boards; ~4.4-6.5x on dense attention; ~2.7x on the cached per-node term, the same on KV | human games, legal moves, by bucket 1-49 / 50-99 / 100-159 / 160-186 boards: tokens 1.94 / 2.19 / 2.43 / 2.35x fewer; D (empty destination tokens) median 35 / 50 / 100 / 140; dense evaluation 2.49 / 3.34 / 4.35 / 4.37x cheaper; cached node (analytic, A5 refused there) 1.42 / 2.05 / 2.63 / 2.55x; **C26's phase form pooled by decisions (M18), N\* = 50: 2.45x - dense below N\* 2.46x, the cached node at or above it 2.11x** (N\* = 100: 2.73x; 2.73x / 2.62x); the pre-CHECKPOINT-9 reading: dense at every size 2.79x, cached 1.56x, M16's blend 2.19x; frozen K/V 2.05-2.51x; counted forward at 44 boards 2.66x (block-sparse) / 2.79x (every key); synthetic 240 / 600 boards 4.8-9.9x; CPU wall at 44 boards 3.0x (indicative, loaded) | A5 section below; 716 replayed games, 22,761 distinct positions, action starts only; F1, FP32 counts | under C26, legal destinations: tokens 0.84-1.06 (of 2.3); the cached node at or above N\* 1.28x below ~2.7x (N\* = 100: 1.03x), inside; dense: the range's conditions (240-600 boards) not reached by the games - synthetic 4.8-9.9x; KV inside. **No flag stands** (the 2026-10-23 flags compared across conditions - withdrawn, see the section) | 2026-10-23, N-A5; re-weighed 2026-10-24 |
| Leaf-batching gain | 1.2-1.5× | - | | | |
| NNUE eval | ~10k MACs, sub-µs | - | | | |
| Movegen throughput | ≥ 20,000/s at 240; ≥ 10,000/s at 600 (§15.6) | pseudo-legal expansion, 1 thread: 44,047/s at 240 boards, 24,951/s at 600 (scope + relations); with the legal-moves filter: milliseconds to minutes per expansion | M9 section below | 2.2x / 2.5x the target (pseudo-legal); far below it with the filter | 2026-10-02, b5c1176 |
| Scanner nodes/s | no figure | 1,036,000/s (Q4 per the Guide) and 1,033,000/s (Q4's other reading), one thread, pooled over the corpus; lower on large positions (85% of the nodes are E4 searches) | 6b section below | - | 2026-10-02, ad2769d |

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

### B15's two per-axis far tables and its layer-shared arm - parameters counted (2026-11-22; no threshold, no verdict)

**What it is.** Decision 1 of the 2026-11-15 review (register B6 and B15; §4.1): beyond its exact
classes B15 "adds B6's two clamped log-bucket scalars, one for Δtime and one for Δtimeline ... for
3,456 parameters", and ablation 2 "gains a layer-shared arm: one table per head for all sixteen
layers, a sixteenth of the parameters". Built 2026-11-22 (`python/t5d/net/displacement.py`): the
far tables are `NetConfig.disp_far_axis`, **on - the ruling changes the default network, so every
parameter total of the v1 and of F2's arm moves by them**; `disp_far_axis=False` is the network as
built before the review, bit for bit; the layer-shared arm is `disp_layer_shared`, off. As built
the two tables are per head per layer, over B6's buckets (19 signed Δtime buckets, exact below 4
half-turns and clamped at 128; 8 |Δtimeline| buckets, exact below 4 and clamped at 32), and are
added to every pair of board tokens whose boards lie beyond the exact classes. Machine `dev-r`;
PyTorch 2.13.0+cpu; parameters by `parameter_breakdown` of the built networks (exact; no
precision, batch, board count or window enters). FLOPs by `FlopCounterMode` over one forward:
FP32 trunk, the F1 network, random weights (model seed 0), one position `synthetic_state(0, 44)`
(T = 2,920), window none - counts do not depend on load (the machine was loaded).

**Parameters (exact).**

| Network | B15's table | Per-axis far tables | B15 | Total |
|---|---|---|---|---|
| v1 (F1), the default | 1,623,168 (16 x 8 x 12,681) | **3,456** (16 x 8 x (19 + 8): Δtime 2,432, Δtimeline 1,024) | 1,626,624 | **22,969,139** |
| v1, `disp_far_axis=False` - as built before the review | 1,623,168 | 0 | 1,623,168 | 22,965,683 |
| F2's larger arm (L24-ff12), the default | 2,434,752 (24 x 8 x 12,681) | 5,184 (24 x 8 x 27) | 2,439,936 | **46,900,275** (46,895,091 before) |
| v1, ablation 2's layer-shared arm (`disp_layer_shared`) | 101,448 (8 x 12,681) | 216 (8 x 27) | 101,664 | 21,444,179 |
| v1, ablation 2's no-table arm (`displacement_bias=False`) | 0 | 0 | 0 | 21,342,515 (unchanged) |

§4.1's "3,456 parameters": 3,456 - ratio 1.00. B15 with them is 7.08% of the v1 (§4.1's "~7% of the
network" is the table's 7.07%). §4.1's "a sixteenth of the parameters": 101,664 x 16 = 1,626,624,
exact - which holds only because the arm shares the far tables with the main table. Figures that
follow the v1's total: with A8 on 25,598,259 at the placeholder's 8x FFN and 24,678,963 at 1x; the
cached configuration (B16's 304) 22,969,443. Of the 152 (Δtime bucket, Δtimeline bucket) cells 131
are reached by some board pair beyond the exact classes and 21 never are (both buckets lie inside
them: |Δtime| below 4 with |Δtimeline| below 3); each of the 19 + 8 scalars per head per layer is
reached.

**FLOPs: unchanged.** One 44-board forward counts 316,148,587,072 FLOPs with the far tables (the
default), 316,148,587,072 without them and 316,148,587,072 under the layer-shared arm: the far
terms are gathers and additions, which the counter leaves out as it leaves out B15's own lookup.
Every counted figure in this file therefore stands; the parameter totals quoted by earlier dated
sections are the v1 of their dates.

**The off switch, checked across versions.** The network of the commit before this change
(`faaa64e`'s `python/t5d`, exported to a scratch directory) and this build with
`disp_far_axis=False`, at one seed on one batch (`far_state(7)`, `synthetic_state(8, 6,
n_timelines=3)`, `synthetic_state(9, 1)`): the SHA-256 of the whole `state_dict` and the trunk,
value, policy, per-board, per-square and leave outputs are bitwise identical in nine
configurations - the v1 in BF16 and in FP32, the cached configuration's network, the stage-N2
flags with B17, the chunked, dense and score-function paths, A8 at 1x, and F2's arm.

**Wall time and memory: not measured** - the machine was loaded. By the shapes (arithmetic, not a
measurement): the far terms add one (batch, heads, boards + 2, boards + 2) FP32 tensor per layer -
1.9 MB at 240 boards - and, per query block, one more (batch, heads, rows, keys) FP32 gather beside
B15's own - 134 MB transient at 240 boards with 4-board blocks. For the quiet-machine block:
`python -m t5d.net.bench --config n2-no-table n2-default --boards 44 240` (from `python/`), whose
"n2-default" now carries them. The layer-shared arm here shares the function and the count; the
bias "assembled once per forward" (§10.2) is G11's and is not built, so the arm's wall time today
is the default's.

### A8 - the board-layout embedding, counted and timed (2026-10-23)

**What it is.** Section 15.3's row "Board-layout embedding (A8) | ~0.1% of attention at 44
boards; ~1.6M parameters at 16 channels" for register A8 as built behind its flag
(`python/t5d/net/layout.py`, spec issue 240): the per-board projection of the one-hot layout
(1,536 -> 64 x 16, with a bias), the 16 -> 256 merge into the token input, and the register's
"FFN after the embedding" - one pre-norm residual block at 8x, a placeholder. Machine `dev-r`;
PyTorch 2.13.0+cpu, 12 threads; the F1 network (d 256, L 16, B9's two move-graph heads, B15, the
block-sparse path, dense scope); one position, `synthetic_state(0, N)` - T = 2,920 tokens at 44
boards, 15,856 at 240; random weights (model seed 0); window none. Arms: "off" (the v1), "A8" (the
flag on, its placeholders), "A8 without FFN" (`layout_ffn_ratio = 0`: the row's projection alone -
not the register's A8).

**Parameters (exact).** Projection 1,573,888 (the weight 1,572,864 - the spec's "~1.6M" - and a
bias of 1,024); merge 4,096; FFN 1,051,136 (gain 256; 256 -> 2,048 -> 256 with F5's biases): A8
2,629,120 - the v1's 22,965,426 becomes 25,594,546 (+11.4%); 1,577,984 without the FFN.

**FLOPs, counted** (PyTorch's `FlopCounterMode` over one forward, FP32 trunk - counts do not
depend on the precision or on load). The counted difference between the arms is exactly
`cost.layout_flops` at both sizes (also pinned by `test_layout.py`). *Attention, analytic* =
section 9.1's `4 T^2 d L`; *attention, counted* = the fused CPU kernel's and the global merge's
products (every key a kernel is handed - B15's per-row biases and the position-free register
rows widen it, as N3's table notes).

| Boards | A8 projection | merge | FFN | A8 total | Attention: analytic / counted | Forward counted (off) | Projection / attention: analytic / counted | A8 / attention: analytic / counted | A8 / forward |
|---|---|---|---|---|---|---|---|---|---|
| 44 | 0.138 GF | 0.024 | 6.124 | 6.286 | 139.7 / 193.6 GF | 316.1 GF | **0.099%** / 0.072% | **4.50%** / 3.25% | 1.99% |
| 240 | 0.755 | 0.130 | 33.25 | 34.14 | 4,119 / 4,872 | 5,538 | 0.018% / 0.016% | 0.83% / 0.70% | 0.62% |

Without the FFN A8 is 0.162 GFLOP at 44 boards (0.116% of analytic attention) and 0.885 at 240
(0.021%). Section 15.3's "~0.1%" is the projection's figure (ratio 0.99), but the row is register
A8's, and A8 has an FFN after the embedding: as built (8x, a placeholder) A8 is 4.50% of attention
at 44 boards, 45 times the row, and any per-token FFN - even one 1·d wide (0.765 GFLOP at 44
boards) - makes it at least 0.66%, 6.6 times (issue 240(b)). *(Corrected 2026-10-23, the review
fixes: this paragraph first called the projection's figure "confirmed" and left A8 unflagged.)*

**Flags (more than 2x off, logged in the build log):** §15.3's A8 row - "~0.1% of attention at 44
boards" - is 45x off for A8 as the register defines it (4.50% with its FFN at 8x), and at least
6.6x off with any per-token FFN; the projection alone (0.099%) is within it. The parameters
(2,629,120 against "~1.6M", 1.64x) are inside the 2x rule. Logged as spec issue 240(l); edit 186.

**In the cached configuration (counted, 2026-10-23, the review fixes).** `CachedEvaluator.prepare`
embeds every token of a node, frozen boards' included, so A8's per-token FFN is paid over the whole
position at every node while the node's trunk work is its live set's. Machine `dev-r`; PyTorch
2.13.0+cpu, 12 threads; FP32 trunk; the F1 network in the cached configuration (`attn_scope =
"block_causal"`, the placeholder defaults), random weights (model seed 0), A8 off and on;
`synthetic_state(0, N, n_timelines=8)` - 8 live heads, section 7.1's assumption - its store built
(`root_epoch`), then its physical child (`cache_bench._children`: one board superseded, one
created) evaluated against the root's store; window none; FLOPs by `FlopCounterMode` (counts do not
depend on load). The node = its preparation (the embedding) + the supersession of its missing
boards + the live pass + the heads; *analytic node* = section 7.6's `pass_flops` of the 544 live
tokens against all.

| Boards (child) | Child tokens | Preparation off / A8 | Supersession | Live pass | Heads | Node off / A8 | A8 / node: counted / analytic node | Dense forward off / A8 |
|---|---|---|---|---|---|---|---|---|
| 44 (45) | 2,986 | 0.050 / 6.478 GF | 3.069 | 60.68 | 0.032 | 63.83 / 70.26 GF | **+10.1%** / 13.0% (of 49.4 GF) | 312.7 / 319.0 GF (+2.0%) |
| 240 (241) | 15,922 | 0.264 / 34.54 | 12.90 | 224.94 | 0.085 | 238.19 / 272.47 | **+14.4%** / 20.8% (of 164.7 GF) | 5,501 / 5,535 (+0.62%) |

What A8 adds to the node is exactly `cost.layout_flops` at the child (6,428,114,944 and
34,279,407,616 FLOPs); the supersession, live pass and heads are the same FLOPs with it on or off.
(The dense forward here is the cached configuration's network run dense, 312.7 GF at 44 boards
where the dense-scope network above counts 316.1.) A frozen board's embedding feeds only its K/V,
which the node finds stored: embedding only the live and the missing boards' tokens would remove
most of A8's share - not built (issue 240(g)). `cache_bench` now counts a node's preparation beside
its live pass (`prepare_flops`).

**CPU wall time - indicative only.** Same build and positions; each arm built once, warmed on a
2-board position, then five repetitions interleaved (the arms' order reversed every other
repetition); medians, minima in brackets. **The machine was loaded**: other lanes' jobs held the
CPU at 85-100% (Windows' processor load, sampled before and after each block: 97/85, 94/100,
100/100), so forward times vary by more than A8's share and say nothing about it; the
embedding's own time (`T5DNet.embed_tokens`, milliseconds) is the usable figure. The 240-board
FP32 block was stopped unfinished (the machine at 100%). For the quiet-machine block.

| Boards, trunk | Embedding off / A8 / A8 without FFN (ms) | Forward off / A8 / A8 without FFN (s) | A8's embedding time / forward (off) |
|---|---|---|---|
| 44, BF16 | 1.69 (1.46) / 19.76 (14.65) / 3.55 (2.73) | 4.28 (3.50) / 4.48 (4.10) / 3.89 (3.69) | +18.1 ms, 0.42% |
| 44, FP32 | 2.51 (1.90) / 16.00 (13.73) / 4.23 (2.72) | 3.24 (2.92) / 3.27 (2.76) / 3.33 (2.81) | +13.5 ms, 0.42% |
| 240, BF16 | 15.6 (9.5) / 118.3 (102.1) / 22.9 (16.2) | 75.4 (71.3) / 73.9 (69.8) / 74.2 (68.5) | +102.8 ms, 0.14% |

The embedding runs in FP32 under either trunk (it precedes the trunk's autocast), so its time does
not follow the trunk's precision. Lc0's "5% latency" is a GPU figure for a different network and is
not compared.

**The table, quiet - command; not yet measured on a quiet machine.** The table above was timed by
a script the repository does not hold. Since 2026-11-28 (`lane/search`) the bench does it:
`--embed` times `T5DNet.embed_tokens` alone on the pass's position (`embed_ms`, beside the
forward's `seconds`), and `--repeat 5` runs the block five times, the arms' order reversed every
other repetition, then prints each arm's medians and minima (`"summary": "median"` lines). The
table's arms are "n2-default" (off), "6p-layout-ff8" (A8) and "a8-no-ffn" (A8 without FFN,
`layout_ffn_ratio = 0`); "6p-layout-ff1" (the next section) goes beside them. From `python/`:

```
python -m t5d.net.bench --config n2-default 6p-layout-ff8 a8-no-ffn 6p-layout-ff1 --boards 44 240 --dtype bf16 fp32 --embed --repeat 5
```

One difference from the table's protocol: each pass runs in a fresh process, which builds its
arm, warms it on a 2-board position, times one forward, then times the embedding of the same
position (warm: the forward has just embedded it). The table built each arm once and timed it
five times in one process. PyTorch's default threads (12), as the table used.

### A8's FFN at 1x - ablation 6p's second arm, counted (2026-11-22; no threshold, no verdict)

**What it is.** §2.2 since the 2026-11-15 review: "the FFN's width is part of the arm: 6p tests it
at 1× (about 0.3% of the forward) beside the 8× placeholder". The arm is the A8 section's network
with the embedding's FFN at width d instead of 8d - `NetConfig(board_layout_embedding=True,
layout_ffn_ratio=1)`, named `6p-layout-ff1` in `config.PRESETS` and `bench.CONFIGS` beside
`6p-layout-ff8`. Nothing else differs: at one seed the two arms share every parameter but the
FFN's two matrices and their biases. No default changed (the flag is off, the placeholder 8x).
Machine `dev-r`; PyTorch 2.13.0+cpu; FP32 trunk; the F1 network of this date (B9's two move-graph
heads, B15 with its per-axis far tables, the block-sparse path, dense scope); random weights
(model seed 0); one position, `synthetic_state(0, N)`; window none. FLOPs by `FlopCounterMode`
over one forward - counts do not depend on load (the machine was loaded: other jobs of the build).

**Parameters (exact).** The projection (1,573,888) and the merge (4,096) as at 8x; the FFN 131,840
(gain 256; 256 -> 256 -> 256 with F5's biases) against 1,051,136: A8 is **1,709,824** against
2,629,120 - 919,296 fewer. The v1's 22,969,139 becomes 24,678,963 (+7.4%), against 25,598,259 at
8x (+11.4%). (The A8 section above quotes the v1 of its date, 22,965,426 and 25,594,546. With
B15's per-axis far tables off - the network before the 2026-11-15 review - the three figures are
22,965,683, 24,675,507 and 25,594,803.)

**FLOPs.** The 44-board row is counted: one forward without A8 316,148,587,072 FLOPs (T = 2,920),
with the arm at 1x 317,076,380,224 and at 8x 322,434,603,584 - differences 927,793,152 and
6,286,016,512, exactly `cost.layout_flops` (pinned by `test_layout.py`). The 240-board row is
**derived**: `cost.layout_flops` (exact) against the 2026-10-23 table's denominators above, not
re-counted.

| Boards | A8 projection | merge | FFN at 1x | A8 at 1x | A8 at 8x | Attention: analytic / counted | Forward counted (off) | A8 at 1x / attention: analytic / counted | A8 / forward: 1x / 8x |
|---|---|---|---|---|---|---|---|---|---|
| 44 (counted) | 0.138 GF | 0.024 | 0.765 | **0.928** | 6.286 | 139.7 / 193.6 GF | 316.1 GF | 0.66% / 0.48% | **0.29%** / 1.99% |
| 240 (derived) | 0.755 | 0.130 | 4.157 | 5.041 | 34.14 | 4,119 / 4,872 | 5,538 | 0.12% / 0.10% | 0.091% / 0.62% |

**In the cached configuration - derived, not re-counted.** The 2026-10-23 count found that what A8
adds to a node is exactly `cost.layout_flops` at the child (every token embedded at every node).
At 1x that is 948,781,056 FLOPs on the 63.83 GF node at 44 boards - **+1.5%** (1.9% of the 49.4 GF
analytic node) - and 5,062,410,240 on 238.19 GF at 240 boards - **+2.1%** (3.1% of 164.7 GF) -
against +10.1% and +14.4% at 8x.

**Against the estimates.** §2.2's "about 0.3% of the forward" at 1x: 0.293% - ratio 0.98, inside
the 2x rule, no flag. Its "2.0% of the forward" at 8x is re-counted on today's network: 1.99%.
§15.3's own row for A8 ("~0.1% of attention at 44 boards") was flagged on 2026-10-23 for any
per-token FFN, a 1·d one included (0.66% of attention, 6.6x; issue 240(l)): that flag stands and
this arm is the figure it named.

**Wall time: not measured** - the machine was loaded. For the quiet-machine block: `python -m
t5d.net.bench --config n2-default 6p-layout-ff8 6p-layout-ff1 --boards 44 240` (from `python/`).
The A8 section's quiet command (above, 2026-11-28) runs this arm too, with the embedding's time
and five repetitions.

### FFN 2x and 4x - wall time against the v1's 8x (command; not yet measured on a quiet machine)

**What it will measure.** Section 15.3's row "FFN ratio 2× and 4× | 1.39× and 1.23× evals/s at 44
boards | 9.2 | estimated → to measure". Section 9.2's table prices the trunk's FFN ratio against the
v1's 8x at 44 boards - cost 0.81 (4x) and 0.72 (2x), evaluations/s 1.23x and 1.39x - from the
analytic FLOP model; ablation 5's arm (2x against 8x at matched compute) is a trained comparison and
not this row. The arms (2026-11-28, `lane/search`): `bench.CONFIGS`' "ffn-4x" and "ffn-2x", the v1
default with `ffn_ratio` 4 and 2 (`d_ffn` 1,024 and 512 against 2,048), every other setting the
v1's - "n2-default" is the baseline. The figure is the forward's wall-time ratio, n2-default's over
the arm's, per board count and precision: CPU-labelled (the row's evaluations/s are a GPU
figure's).

**Command** (from `python/`; one pass per subprocess, the A8 section's protocol - five
repetitions, the order reversed every other one, medians and minima printed per arm):

```
python -m t5d.net.bench --config n2-default ffn-4x ffn-2x --boards 44 240 --dtype fp32 bf16 --repeat 5
```

### A5 - piece-centric tokens: token counts on the human games, counted FLOPs (2026-10-23)

**What it is.** Section 14.4's figures for register A5 (piece-centric tokenisation, a prototype
behind `NetConfig.piece_centric`; BUILD_LOG issues 244-257), which no §15.3 row carries (proposed
edit 197): "~250 tokens across a 240-board state ... 2.3x tokens midgame, 2.7x in sparse endgames -
~4.4-6.5x on dense attention, ~2.7x on the cached per-node term, the same on KV memory". A5's tokens
as built: the pieces, the candidate moves' empty destination cells (D) and the registers, `T_A5 =
pieces + D + 2N + 16` against A1's `66N + 16`. M18 weighs this decision by "where compute goes - the
human games now"; process note 2 asks for the figures at the scale they would run at.

**Token counts on the human games** - `py -m t5d.engine.token_counts <t0db>` (`python/t5d/engine/
token_counts.py`, the engine binding only): t0database (`data/t0database/`, `Data.csv` sha256
`37fb7c1c...`) converted by `py data/t0database/to_5dpgn.py`, 761 records, replayed as the gate's
parse mode replays (`Game.from_pgn(legal=False)`); **716 replay to the end** and count (45 refused, as
M7's); at every action start the counts for the **legal** moves (`legal_moves=True`, exact: no
budget) and the pseudo-legal ones, through the network's own construction (`net.cost.token_counts`,
recounted independently from the engine's piece count and move flags in the tests); positions
distinct by fingerprint and side to move - **22,761**, the same per-bucket counts as M7's legal
table (19,410 / 3,162 / 171 / 18); the decisions behind them (28,843 action starts, repeats
included) weight the pooled figures. **Action starts only**: the ply split's mid-action leaves and a
search's deeper nodes are not measured. Engine at main `40a1f1c`, the tool as first committed
(`lane/net` `6670d9e` - `token_counts.py` first exists there, not in `bf49a6d` as first written
here); machine `dev-r`; counts do not depend on load. Re-run 2026-10-24 after main `b73d073` was
merged into `lane/net` (`1bd78d8`), with the tool of the A5 review fixes (the commit after it: C26's phase form added, the JSON
lines carrying each position's decisions): the same 22,761 positions and counts, line for line, and
the same per-bucket and pooled figures as before (52 s). The analytic costs are section 9.1's model (`cost.piece_centric_costs`,
d 256, L 16): the dense evaluation (`4 T^2 d L + 40 T d^2 L`); the cached node - the live queries
against every token, A1's the head boards' tokens and the 16 global registers, A5's the heads'
pieces and registers, **every destination token** (transient, never cached - BUILD_LOG issue 252's
reading (f)) and the 16 - and the frozen boards' K/V. A5's cached figures are this model's only: the
cached path refuses A5 as built (issue 252). *(2026-11-27: decision 7 built - the cached path now runs
A5 on exactly these sets, "The cached path under A5" below; its cost is not yet benched, so these
figures stay the analytic model's.)*

Legal moves - means over the distinct positions of each bucket (ratios A1 / A5 of the means):

| N bucket | positions | median N | pieces per board | D median / p95 / max | D on head boards | T_A1 / T_A5 | Tokens fewer (median per position) | Dense eval | Dense attention alone | Cached node | Frozen K/V |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-49 | 19,410 | 24 | 30.6 | 35 / 82 / 247 | 57% | 1,676 / 862 | **1.94x** (1.89) | **2.49x** | 3.90x | **1.42x** | 2.05x |
| 50-99 | 3,162 | 60 | 27.2 | 50 / 151 / 401 | 55% | 4,212 / 1,922 | 2.19x (2.18) | 3.34x | 4.81x | 2.05x | 2.26x |
| 100-159 | 171 | 112 | 24.3 | 100 / 203 / 306 | 60% | 7,860 / 3,232 | 2.43x (2.41) | 4.35x | 5.82x | 2.63x | 2.51x |
| 160-186 | 18 | 169.5 | 25.2 | 140 / 306 / 549 | 61% | 11,229 / 4,780 | 2.35x (2.55) | 4.37x | 5.43x | 2.55x | 2.43x |
| pooled by decisions (M18), every size - the pre-CHECKPOINT-9 reading | 28,843 decisions | | | | | | | 2.79x | | 1.56x | |

**Pooled under CHECKPOINT 9's phase form** (C26, spec section 7.10: a search whose root has fewer
than N\* boards runs dense, one at or above N\* cached; N\* a placeholder, 50). Each position is a
decision's root, so it is costed dense below N\* and as a cached node at or above it; section 7.10
and M18 weigh the cache by the compute above N\*, so that half is given apart. Weighted by
decisions (`token_counts --n-star`, `pooled.phase`):

| Move set | N\* | decisions below / at or above | Phase form pooled | Dense below N\* | Cached node at or above N\* | Share of A1's / A5's phase-form compute at or above N\* |
|---|---|---|---|---|---|---|
| legal | 50 (placeholder) | 25,336 / 3,507 | **2.45x** | **2.46x** | **2.11x** | 3.3% / 3.8% |
| legal | 100 (sensitivity) | 28,654 / 189 | 2.73x | 2.73x | 2.62x | 0.3% / 0.3% |
| pseudo-legal (sensitivity) | 50 | 25,336 / 3,507 | 2.40x | 2.43x | 1.79x | 3.3% / 4.4% |
| pseudo-legal (sensitivity) | 100 | 28,654 / 189 | 2.69x | 2.69x | 2.22x | 0.3% / 0.4% |

(Unweighted - one per distinct position - the review's re-pooling gave 2.48x / 2.12x, 2.76x / 2.62x,
2.43x / 1.80x and 2.72x / 2.22x.) A cached node costs a small fraction of a dense evaluation, so
on the human games the phase form's compute is almost all dense below N\*: what A5 buys there is
the dense figure. The pre-CHECKPOINT-9 reading, kept for comparison: M16's blend (0.9 cached, 0.1
dense, at every size) pooled **2.19x**. Under A1 the dense compute falls 58.6% / 35.2% / 5.2% /
1.0% in the four buckets. Pseudo-legal moves (upper bounds on D): D median / p95 / max 38 / 104 /
285, 73 / 191 / 403, 127 / 266 / 505, 212 / 502 / 626; tokens 1.93 / 2.16 / 2.40 / 2.32x fewer;
dense 2.46 / 3.27 / 4.27 / 4.27x; cached node 1.30 / 1.74 / 2.24 / 2.15x; pooled at every size:
dense 2.75x, node 1.42x, blend 2.07x. About 40% of the destination tokens land on history boards (travel
moves' landing squares), which bears on issue 252: §14.4 called only the head boards' destination
tokens transient. *(2026-11-28: the review of 2026-11-15 changed §14.4 - "destination tokens are
transient per node, which makes them live tokens under the cache (§7.3)" - and §7.3 puts them on
history boards too; decision 7, built 2026-11-27.)*

**FLOPs, counted** - `py -m t5d.engine.token_counts --flops --flops-record <t0db>/t0db-0002.5dpgn
--flops-check` and `--flops`: one dense forward of the F1 network (FP32 trunk - counts do not depend
on the precision or on load; B9's two move-graph heads, B15; random weights, model seed 0),
`FlopCounterMode` with the CPU fused-attention formula (`net/cost.py`), counted at 1 and 2 layers and
extrapolated to 16 - the layers are identical, so the difference is one layer's; checked at 44 boards,
where the full 16-layer counts are the extrapolation's exactly (280,815,012,416 and 105,427,533,568).
Positions: the first action start of t0db-0002 with 44 boards (the human median game length; legal
moves; T 2,920 / 1,491) and `synthetic_state(0, N, n_timelines=k)` at the spec's 8 and 12 timelines
and the corpus's 24 - synthetic boards hold 17-25 pieces and the generator's capped move sample gives
D 78-152, so their token ratios (2.4-3.4x) are the generator's, not a game's. Window none.

| Position | T_A1 / T_A5 | Analytic A1 / A5 (TF) | Counted block-sparse A1 / A5 | Counted every key A1 / A5 | Ratio: analytic / block-sparse / every key |
|---|---|---|---|---|---|
| human, 44 boards | 2,920 / 1,491 | 0.262 / 0.099 | 0.281 / 0.105 | 0.323 / 0.116 | **2.65 / 2.66 / 2.79x** |
| synthetic 240, 8 TL | 15,856 / 4,760 | 4.784 / 0.571 | 5.501 / 0.666 | 6.583 / 0.759 | 8.38 / 8.26 / 8.67x |
| synthetic 240, 24 TL | 15,856 / 6,606 | 4.784 / 0.992 | 5.563 / 1.162 | 6.583 / 1.335 | 4.82 / 4.79 / 4.93x |
| synthetic 600, 12 TL | 39,616 / 11,767 | 27.375 / 2.762 | 30.613 / 3.257 | 38.617 / 3.920 | 9.91 / 9.40 / 9.85x |
| synthetic 600, 24 TL | 39,616 / 15,924 | 27.375 / 4.822 | 30.882 / 5.627 | 38.618 / 6.830 | 5.68 / 5.49 / 5.65x |

The counted saving follows the analytic one to within 5% on both paths: A5 is A1's computation on a
smaller token set (no parameter added - issue 244(b)). The brief's reconstruction of §14.4's figures
(2.3x fewer tokens at 240 boards, 15,856 -> 6,894: 4.5x; 2.7x at 600: 6.6x - pinned by
`test_the_analytic_figures_section_14_4_rests_on`) holds for the dense evaluation's total, which is
what "~4.4-6.5x on dense attention" must mean (attention alone would be 5.3-7.3x); at the human median
of 44 boards the linear terms are about half the cost, so a 1.96x token cut buys 2.65x.

**CPU wall time - indicative only.** The F1 network as shipped (BF16 trunk, block-sparse), the
44-board human position, 4 torch threads, one warm-up then five forwards each: A1 5.65 s, A5 1.90 s
(medians; **3.0x**). The machine was loaded (other lanes' jobs; Windows' processor load 96%): for
the quiet-machine block. GPU figures, varlen kernels among them, are deferred (M19).

**A5's dense wall at 44 boards - command; not yet measured on a quiet machine.** The figure above
had no recorded command. The bench's arm "a5-piece-centric" (2026-11-28, `lane/search`): the v1
default with `piece_centric=True`, dense scope, beside "n2-default" (A1). From `python/`: `python -m
t5d.net.bench --config n2-default a5-piece-centric --boards 44 --dtype bf16 --threads 4 --repeat 5`
(the F1 network as shipped, BF16; the figure above's 4 threads and five forwards, as medians). Its
position is the bench's synthetic one, not the human position above: `synthetic_state(0, 44)` has
2,920 A1 tokens and 1,378 A5 tokens (2.12x fewer, against the human position's 1.96x), so its ratio
is not the figure above re-measured.

**Against §14.4's estimates (the 2x rule) - re-stated under CHECKPOINT 9's C26, on the
legal-destination set §14.4 speaks of** ("only head boards carry the full legal-destination set";
the pseudo-legal figures are a sensitivity, not the flagged ones). Tokens: 1.94-2.43x fewer by
bucket against "2.3x midgame, 2.7x sparse endgames" - 0.84-1.06 of 2.3 (the human games stop at 186
boards; their late buckets hold 189 positions): inside. Destination tokens: median 140 at 160-186
boards (0.87 per board, about 210 at 240) against "~250 at 240 boards" - inside. Cached per-node
term: under C26 the cache runs only at or above N\*, where the node is 2.11x cheaper (N\* = 50) or
2.62x (N\* = 100) against "~2.7x" - 1.28x and 1.03x below, **inside** (pseudo-legal 1.79x, 1.5x
below, inside). Dense evaluation: "~4.4-6.5x" is the dense evaluation's total at 240-600 boards (the
brief's reconstruction, above); the human games never reach those sizes, so the estimate's
conditions were not measured on games - synthetic positions there count 4.8-9.9x (the generator's
D), at or above the range; where C26 runs dense, below N\*, A5 buys 2.46x, a size §14.4 gives no
estimate for. KV: 2.05-2.51x against "the same" (2.7x) - inside. **No figure is flagged.**

Withdrawn (the 2026-10-23 reading): "cached node 2.1x below with pseudo-legal moves (1.30x) - flagged"
compared a cached node at every size, under 50 boards included, where C26 never caches, on the
pseudo-legal set §14.4 does not speak of; "dense 2.6x below its top" compared the 600-board 6.5x with
the under-50-board figure - estimate and measurement under different conditions, which the 2x rule
does not judge. What stays true at every size: A5's live set carries every destination token, about
35 under 50 boards beside a head board's 66 tokens under A1, so at one or two live heads a cached node
under 50 boards barely shrinks (1.42x) - moot under C26, which caches nothing there. Logged as BUILD_LOG
issue 256; the write-back is proposed edit 198.

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

### Kairos (H52) against the recorded answers - a smoke run, not a recording (2026-10-23; no threshold, no verdict)

`dev-r`, loaded by the other lanes; Kairos at `8bfea4a6e8d1`, its driver built with Rust
1.99.0 (x86_64-pc-windows-gnu, self-contained linker; fat LTO, 26 s). Both sources through
the harness's `runner.observe` over all 41 corpus records, 3 at a time - `kairos` 120 s a
record, `kairos:naive` 60 s - compared position by position with the committed golden files
of ftxi, cwmtt and Chess in 5D, up to the first placement divergence (there was none).
Nothing was written under `oracles/`. Detail and reading: `docs/BUILD_LOG.md`, issue 533.

| Against | Same boards and pieces | Same unmoved markers | Check agrees | Terminal agrees | Count agrees (`kairos:naive`) |
|---|---|---|---|---|---|
| ftxi | 987 / 987 | 987 / 987 | 855 / 986 | 986 / 986 | 841 / 843 |
| cwmtt | 803 / 803 | 780 / 803 (Very Small - Open) | - (no check verdict) | 802 / 802 ("no action") | 680 / 680 |
| Chess in 5D | 815 / 815 | 815 / 815 | 814 / 814 | 791 / 793 | - (no count) |

Records read: 32 to their end or refusal (31 to the end, local/ignores-check to its refusal);
8 `parse_failed` (Kairos's reader fails the whole record; first written as "31 ... 9", a
miscount corrected 2026-11-02 from the parse-mode dry run); the hypercuboid search cut at 120 s on 1 record (cwmtt/100_timelines, position 21
of 22), the exhaustive one at 60 s on 10.

### G11 - operators and attention-kernel calls of one forward, counted on the CPU graph (2026-11-22; counts; no threshold, no verdict)

**What it is.** §10.2's third requirement (2026-11-15 review; register G11): "Launches per layer
are a constant ... It is counted, not assumed: a §15.3 figure. The eager forward's operations can
be counted on the CPU today; the compiled path's launches are read by profiler on GPU time." This
is the first half of that sentence - what one eager forward dispatches - counted by
`python/t5d/net/opcount.py` (`python -m t5d.net.opcount`, from `python/`; its tests are
`python/tests/net/test_opcount.py`). **It reports two figures, and neither is a count of device
kernel launches:**

- **Operators dispatched**: every ATen operator the forward hands to the backend, after PyTorch's
  composite operators have decomposed (a `TorchDispatchMode`: a linear layer on a matrix is one
  transpose view and one `addmm`; a fused attention call is one operator). An operator is **free**
  when its schema says its result aliases its input (a view) or it is a bare allocation
  (`opcount.FREE_OPS`), and **compute** otherwise. The compute operators are the proxy for what the
  eager path launches on a device - each is at least one kernel or copy there, and one may launch
  several. **Sync** counts the operators the host waits on, by name (`_local_scalar_dense` -
  `.item()`, `bool(tensor)` - and `nonzero`): a lower bound, because the `nonzero` inside
  boolean-mask indexing is issued inside a kernel and never reaches the dispatcher.
- **Attention-kernel calls**: calls of `torch.nn.functional.scaled_dot_product_attention` (a
  `TorchFunctionMode`; `attention._kernel` is the network's one call site), each fused (one fused
  attention operator) or math (PyTorch's matmul-and-softmax path).

Not counted: Python work; device launches as such; time; the compiled path (`torch.compile` is not
run - its launches are a profiler figure on GPU time, §10.2); `attn_impl="flex"` (PyTorch 2.13's
eager `flex_attention` fails under a dispatch mode, so the tool refuses it); the cached path and
training steps.

**Conditions.** `dev-r`; PyTorch 2.13.0+cpu, Python 3.14.7; eager, `torch.no_grad()`. These are
counts, not timings: they do not depend on the machine, its threads or its load (other jobs were
running), and they do depend on the PyTorch build (how composites decompose), the configuration,
the precision (autocast's casts are operators), whether the batch is one position or several and,
on the block paths, the position. The F1 network (d 256, 16 layers, 8 heads, FFN 8x), random
weights (`build_model(seed=0)`; the counts do not depend on them), BF16 trunk unless stated; one
position unless stated, `synthetic_state(seed, N)` with seed 0 unless stated (a batch of b is
seeds 0 to b - 1, each of N boards); window none. The configurations are `net/bench.py`'s
`CONFIGS`: "n2-default" is the network as shipped (`attn_impl="block_sparse"`, query blocks of 4
boards). The tree is faaa64e with the uncommitted work of the 2026-11-15 review's implementers as
it stood when the counts were taken: **`NetConfig.disp_far_axis` existed and was True** (B15's
per-axis far tables - the section "B15's two per-axis far tables" above - on by default) and
`disp_layer_shared` existed and was False; the count with `disp_far_axis=False`, the network as
built before the review, is given below ("B15's arms"). The counted forward's outputs equal the
uncounted forward's bit for bit (checked on this network at 44 boards, seeds 0 and 44, and by the
tests), and a count repeats exactly: every figure here was taken at least twice, in separate
processes.

**The network as shipped, by board count** (`--config n2-default --boards 10 24 44 112 240`):

| Boards | Tokens | Operators | Compute | Free | Sync | Layer 0 | Layers 1-15, each (compute) | A layer's attention / FFN / norm / other | Kernel calls a layer | a forward |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 676 | 27,706 | 16,153 | 11,553 | 546 | 1,932 | 1,686 (977) | 1,645 / 14 / 12 / 15 | 14 | 224 |
| 24 | 1,600 | 46,123 | 26,729 | 19,394 | 914 | 3,324 | 2,821 (1,627) | 2,780 / 14 / 12 / 15 | 26 | 416 |
| 44 | 2,920 | 76,443 | 44,128 | 32,315 | 1,517 | 5,624 | 4,689 (2,696) | 4,648 / 14 / 12 / 15 | 46 | 736 |
| 112 | 7,408 | 163,556 | 93,586 | 69,970 | 3,127 | 12,592 | 10,032 (5,720) | 9,991 / 14 / 12 / 15 | 114 | 1,824 |
| 240 | 15,856 | 334,464 | 190,726 | 143,738 | 6,315 | 26,105 | 20,525 (11,666) | 20,484 / 14 / 12 / 15 | 244 | 3,904 |

Every kernel call is fused - on the CPU; which kernel a device takes is the device's (the AMD
run's calls fell back to the math backend, below). Outside the layers the count is the same at
every size, 484 operators: the embedding 66, the context (`make_context`) 245, the final norm 6,
the heads 167. A layer is its attention sublayer - 4,648 of a layer's 4,689 operators at 44
boards - and its kernel calls are 4 a query block of 4 boards (two head groups, each with a second
call for the rows that lack file and rank) plus 2 for the masked heads' free rows (4 at 240
boards, where those rows take two passes). Layer 0 dispatches more than the others: the
block-sparse plans are built on first use and reused.

**At 44 boards (2,920 tokens): precision, batch, position** (`--dtype fp32`; `--batch 1 2 8`;
`--seed 44`):

| Condition | Operators | Compute | Sync | Layer 0 | Layers 1-15, each (compute) | Kernel calls a layer | a forward |
|---|---|---|---|---|---|---|---|
| BF16, one position, seed 0 (the row above) | 76,443 | 44,128 | 1,517 | 5,624 | 4,689 (2,696) | 46 | 736 |
| FP32, one position, seed 0 | 73,195 | 40,880 | 1,517 | 5,421 | 4,486 (2,493) | 46 | 736 |
| BF16, a batch of 2 (seeds 0-1) | 79,866 | 46,225 | 1,610 | 5,807 | 4,905 (2,828) | 46 | 736 |
| BF16, a batch of 8 (seeds 0-7) | 79,866 | 46,225 | 1,610 | 5,807 | 4,905 (2,828) | 46 | 736 |
| BF16, one position, seed 44 | 77,568 | 44,811 | 1,548 | 5,684 | 4,760 (2,739) | 46 | 736 |

FP32 dispatches 203 fewer operators a layer (BF16's casts are operators). A batch of 2 and a batch
of 8 count the same here, 216 operators a layer above the single position - most of it the second
position's own, since on this path a block's plan depends on every position of the batch: seed 1
alone counts 4,831 a layer, and seed 0 twice in one batch 4,710, 21 above seed 0 once (those two
on a 3-layer network of the same configuration, whose layers count as these do). The kernel calls
do not move. Seed 44, `synthetic_state(44, 44)`, is the position of the AMD run ("G9's bounded AMD
attempt", below).

**Other configurations** (`--config n1-dense n2-no-table n2-default-chunked --boards 24 44 240`;
`--config n2-default-sdpa --boards 10 24 44`):

| Configuration | Boards | Operators | Compute | Sync | Layer 0 | Layers 1-15, each (compute) | Kernel calls a layer | a forward | Outside: embedding / context / final norm / heads |
|---|---|---|---|---|---|---|---|---|---|
| n1-dense | 24 | 5,183 | 3,229 | 114 | 307 | 307 (191) | 2 | 32 | 66 / 32 / 6 / 167 |
| n1-dense | 44 | 5,183 | 3,229 | 114 | 307 | 307 (191) | 2 | 32 | 66 / 32 / 6 / 167 |
| n1-dense | 240 | 5,183 | 3,229 | 114 | 307 | 307 (191) | 2 | 32 | 66 / 32 / 6 / 167 |
| n2-no-table | 24 | 25,048 | 15,393 | 738 | 2,018 | 1,515 (928) | 16 | 256 | 66 / 66 / 6 / 167 |
| n2-no-table | 44 | 37,272 | 22,776 | 1,101 | 3,187 | 2,252 (1,371) | 26 | 416 | 66 / 66 / 6 / 167 |
| n2-no-table | 240 | 116,189 | 70,782 | 3,499 | 12,474 | 6,894 (4,179) | 124 | 1,984 | 66 / 66 / 6 / 167 |
| n2-default-chunked | 24 | 24,372 | 13,790 | 547 | 1,493 | 1,493 (840) | 12 | 192 | 66 / 245 / 6 / 167 |
| n2-default-chunked | 44 | 40,932 | 22,990 | 947 | 2,528 | 2,528 (1,415) | 22 | 352 | 66 / 245 / 6 / 167 |
| n2-default-chunked | 240 | 203,220 | 113,150 | 4,867 | 12,671 | 12,671 (7,050) | 120 | 1,920 | 66 / 245 / 6 / 167 |
| n2-default-sdpa | 10 | 7,812 | 4,590 | 147 | 458 | 458 (265) | 2 | 32 | 66 / 245 / 6 / 167 |
| n2-default-sdpa | 24 | 7,812 | 4,590 | 147 | 458 | 458 (265) | 2 | 32 | 66 / 245 / 6 / 167 |
| n2-default-sdpa | 44 | 7,812 | 4,590 | 147 | 458 | 458 (265) | 2 | 32 | 66 / 245 / 6 / 167 |

**B15's arms** ("n2-default" at 44 boards, counted with `opcount.measure` on a network built with
the field changed - `CONFIGS` has no entry for them): with `disp_far_axis=False` 71,977 operators
(42,064 compute; layer 0 5,348, layers 1-15 4,413 each; context 195; 46 kernel calls a layer) - so
the far tables add 276 operators a layer and 50 to the context, +6.2% a forward at 44 boards
(+6.1% at 10 boards, 26,104 against 27,706; +7.3% at 240, 311,662 against 334,464), and no kernel
call. 71,977 is also what a prototype of this counter counted on the tree of 2026-11-15, before the
far tables existed (the review's plan for this item). With `disp_layer_shared=True` 76,248
operators (44,038 compute; layers 1-15 4,676 each; context 258): the arm shares the tables, not
yet the work - the bias "assembled once per forward" (§10.2) is not built.

**Where a layer's operators go** ("n2-default", 44 boards, a layer after the first: 4,689
operators; `--ops 12`): `unsqueeze` 791, `slice` 416, `expand` 346 and `view` 145 (free);
`index.Tensor` 351, `gather` 252, `_to_copy` 227, `where.self` 206, `add.Tensor` 140,
`mul.Tensor` 138, `cat` 106, `any.dim` 91 (compute) - the masks, the role rows and the bias
gathers built per query block, per head group and per layer. The context's 245 (additions,
selects, comparisons, casts) do not grow with the board count.

**Constancy - what requirement 3 asks, as far as counting can judge it** (the tool's verdicts;
`--check` exits 1 on a "NO"; no verdict is about a magnitude):

| Configuration | Kernel calls equal across layers | Layers equal after the first | Per layer independent of the board count | Per layer independent of the batch (2 and more) | Outside the layers independent of the board count |
|---|---|---|---|---|---|
| n1-dense | yes | yes | yes | yes | yes |
| n2-default-sdpa | yes | yes | yes | yes | yes |
| n2-default-chunked | yes | yes | **NO** | yes | yes |
| n2-no-table | yes | yes | **NO** | not judged | yes |
| n2-default | yes | yes | **NO** | yes | yes |

Rows judged. Board counts: 10 to 240 for "n2-default", 10 / 24 / 44 for "n2-default-sdpa", 24 / 44 /
240 for the others. Batches: 2 and 4 positions at 10 and 24 boards (`--boards 10 24 --batch 1 2 4`:
every configuration but "n2-no-table"), and 2 and 8 at 44 boards ("n2-default"). `--check` exits 1
on `--config n2-default --boards 24 44` and 0 on `--config n1-dense --boards 24 44 --batch 2 4`.
The default's "yes" on the batch is a fact about these positions, not a property of the path: on
block-sparse two positions of one size count differently (seeds 0, 1 and 44 above;
`AttnStructure.block_plan` branches on the position). The one-block path's constancy is structural
and is pinned by the tests ("n1-dense", "n2-default-sdpa": the same operators in the same places at
every board count, for every position, and for every batch of 2 or more). It says nothing of the
magnitude: that path dispatches 458 operators a layer (265 compute).

**Against §15.3's "Kernel launches per forward ... eager as built, about 10,000 a forward" -
FLAGGED on the default path (the 2x rule).** At 44 boards, BF16, one position:

| Path (44 boards, BF16, one position) | Operators dispatched | Compute | Sync | Kernel calls | Compute / 10,000 | Dispatched / 10,000 |
|---|---|---|---|---|---|---|
| n2-default (block-sparse, as shipped), seed 0 | 76,443 | 44,128 | 1,517 | 736 | 4.41 | 7.64 |
| the same, seed 44 (the AMD run's position) | 77,568 | 44,811 | 1,548 | 736 | 4.48 | 7.76 |
| the same with `disp_far_axis=False`, seed 0 | 71,977 | 42,064 | 1,517 | 736 | 4.21 | 7.20 |
| n2-default-chunked, seed 0 | 40,932 | 22,990 | 947 | 352 | 2.30 | 4.09 |
| n2-default-sdpa (one block), seed 0 | 7,812 | 4,590 | 147 | 32 | 0.46 | 0.78 |
| n1-dense (no mask, no table), seed 0 | 5,183 | 3,229 | 114 | 32 | 0.32 | 0.52 |

A count and that estimate are not the same quantity, and the row names neither the quantity nor
the attention path. "About 10,000" (§10.2's "some 10,000 small operations a forward"; "G9's
bounded AMD attempt" below) is an order of magnitude read from timings with launches made
blocking, on torch 2.14.0+rocm10.1.0, at lane/gate 529098d (before the far tables), on
`synthetic_state(44, 44)`: no operator was counted there (`python/tools/g9_amd_probe.py` counts
none). Read as the default path - what "as built" says - it is **4.4x below the compute operators
and 7.6x below the operators dispatched: flagged**. Read as the one-block path - whose timings
(240-580 ms) the sentence follows in §10.2 - it is within 2x of the operators dispatched (0.78)
and 2.2x above the compute operators. Per layer, against the requirement's "a few tens": the
default dispatches 4,689 operators (2,696 compute) in 46 kernel calls at 44 boards and 20,525
(11,666) in 244 at 240 boards; the one-block path 458 (265) in 2 at every size. The requirement
binds the compiled path, whose launches are a GPU-time figure: nothing here judges it. The spec is
not edited in this repository: the write-back of the counted figures over the row (M4), and the
row's missing quantity, path and board count as a spec issue, are for `docs/BUILD_LOG.md`.

### G11 - the kernel's widths, where a layer's operators come from, what is built once per forward (2026-11-23; counts and numerics; no threshold, no verdict)

**What it is.** Three pieces of §10.2's kernel-bound requirement (register G11, 2026-11-15
review), built in the network after the count above: (a) "Kernel-friendly widths" - decision 15,
settling BUILD_LOG issue 521 - as `NetConfig.attn_pad_multiple`, the multiple the attention
kernel's query and key widths are zero-padded to (**default 1, the widths as built**: a placeholder
of this build, not the ruling's 8 - which of the two is the v1's until GPU time is for the spec's
owner, through `docs/BUILD_LOG.md`); (b) where one layer's operators
come from, attributed by the counter's new call paths (`python -m t5d.net.opcount --sites K
--depth D`); (c) requirement 1, "structure is computed once per forward, not once per layer": what
no layer changes is now built in the first layer and read by the rest (two per-forward memos,
`AttnContext.memo` and `AttnStructure.memo`), each layer's keys and queries are taken once a layer
instead of once per query block, and under B15's layer-shared arm (`disp_layer_shared`) the
displacement bias is assembled once per forward - **all of it bit for bit the network as it was**.
Tests: `python/tests/net/test_kernel_pad.py`, `test_once_per_forward.py`.

**Conditions.** `dev-r`; PyTorch 2.13.0+cpu, Python 3.14.7; eager, `torch.no_grad()`. The F1
network (16 layers), `build_model(seed=0)`, BF16 unless stated, one position
`synthetic_state(0, N)` unless stated; the tree is faaa64e with the 2026-11-15 review's
uncommitted work (B15's far tables on) - "before" is that tree as it stood when this work began,
exported to a scratch copy, "after" the same tree with this work. Counts are load-independent (the
machine was loaded: other build jobs). Numerical comparisons are taken at one thread count on both
sides (3 threads; gradients 1): the CPU kernels' last bits depend on it - the untouched
FlexAttention path differs between 3 and 4 threads by 1.2e-7 - so a comparison across thread counts
proves nothing either way.

**(a) The kernel's widths.** Queries, keys and values are one width on the fast paths
(`_KernelKeys` pads the values to the widened keys' width for PyTorch's fast CPU kernels; at 8 the
padding is applied there and each call's queries follow); B14's score function pads queries and
keys only:

| Kernel call | As built | At 8 |
|---|---|---|
| Fast paths ("block_sparse", "chunked", "sdpa"): rows with every axis - 32 rotated dims + 12 unrotated file/rank dims (B8) + the global merge's extra key | **45** | 48 |
| Fast paths: rows without file and rank (per-board registers) - 32 dims rotated on timeline and time + the merge's one | **33** | 40 |
| "flex": all-axes rows / rows without file and rank / global registers' rows (queries and keys; values 32) | 76 / 64 / 32 | 80 / 64 / 32 |
| Fast paths without per-board registers (ablation 3) / without global registers / under B5 with the routing table | 33 / 44 and 32 / 65 and 33 | 40 / 48 and 32 / 72 and 40 |

BUILD_LOG issue 521's "45 and 33" are verified. The counted forward (`cost.count_flops`, whose
fused-attention formula takes each call's shapes) moves by exactly what the widths predict - on the
one-block path `4 H (T - G + 1) (3 (T - G) + 7 x 2N)` a layer, pinned by `test_kernel_pad.py` - and
the analytic model (§9.1, at `d_head`) does not move:

| Boards (tokens) | Path | Counted, as built | Counted, at 8 | Difference |
|---|---|---|---|---|
| 44 (2,920) | block-sparse (BF16; FP32 the same) | 316,148,587,072 | 329,711,987,264 | +13,563,400,192 (+4.29%) |
| 44 | chunked, and one-block | 322,777,811,520 | 336,651,905,600 | +13,874,094,080 (+4.30%) |
| 240 (15,856) | block-sparse | 5,537,741,455,424 | 5,881,890,414,656 | +344,148,959,232 (+6.21%) |
| 240 | chunked | 6,583,441,629,248 | 6,996,108,550,208 | +412,666,920,960 (+6.27%) |

(Analytic, unchanged: 262,170,214,400 at 44 boards, 4,784,195,108,864 at 240.)

**Numerically the forward is bit for bit; gradients are not.** Padded against unpadded, every
output equal (`torch.equal`): the 16-layer network at 44 boards (block-sparse BF16 and FP32,
one-block and chunked BF16) and at 240 boards (block-sparse and chunked, BF16); 105 small
configurations (2-3 layers; every attention implementation including "flex", both precisions, B15's
and the structure's arms, the scope, the register counts, B5, post-norm, G2's FP8 trunk) against
the tree before this work; and every case of `test_kernel_pad.py`. Under autograd the padded
network's gradients differ in the last bits - at most 7.2e-7 on any parameter (FP32, 2 layers, the
three fast paths): a learned bias sends the CPU kernel to its math path, whose backward products
run over the padded width. The test module holds the forward to the bounds the suite gives paths
of one function in another summation order (FP32 2e-5 trunk / 1e-4 heads; BF16 0.06 / 0.03) and
the gradients to `test_chunked.py`'s, since another platform's kernel need not reproduce the
bitwise forward (BUILD_LOG issue 602). A numerical comparison cannot see what the padding is - a
constant pad adds a constant to every score of a row, and softmax ignores it (padding with ones
moved the FP32 trunk by at most 1.4e-6) - so the zero columns, their place and the merge's column are
pinned structurally.

**(b) Where a layer's operators come from** (44 boards, a layer after the first, BF16, one
position; operators, compute in brackets; `--sites 300 --depth 2`, grouped by call path). The 24-board
rows group the same way (default path 2,821 -> 1,844). Layer 0 carries besides the block-sparse
plans (about 770 operators: `key_selection`, `role_rows` and `selecting_rows` under `block_plan`)
and, after this work, the first use of everything kept.

| Where | Class | Default (block-sparse) before -> after | One-block (sdpa) before -> after |
|---|---|---|---|
| Projections, norms, FFN, residuals | inherent | 61 (35) -> 61 (35) | 61 (35) -> 61 (35) |
| RoPE of Q and K; the key features (widened, time-only, padded) | inherent | 76 (51) -> 74 (51) | 76 (51) -> 74 (51) |
| The layer's B15 tables laid out (main, per-axis) | inherent: the layer's own tables | 13 (6) -> 13 (6) | 13 (6) -> 13 (6) |
| Kernel calls: the global merge, the call, the second call for rows without file and rank, its row gathers and scatter | inherent, per call | 1,895 (1,049) -> 1,803 (957) | 89 (49) -> 85 (45) |
| The global registers' rows (explicit scores) | inherent | 19 (13) -> 19 (13) | 19 (13) -> 19 (13) |
| Block loop and bookkeeping | inherent | 59 (30) -> 56 (29) | 13 (7) -> 13 (7) |
| Query features of the kernel rows | inherent; was repeated per query block | 399 (279) -> 96 (16) | 14 (9) -> 12 (7) |
| Key and value heads per head group | was repeated per query block | 138 (138) -> 12 (12) | 0 -> 0 |
| Found anew each layer: spatial-free rows and their gather indices, free rows, file/rank dims, uncovered registers, edge spans, head groups, row ranges, empty-row flags, mask constants, lookup sides | layer-independent, linear: **hoisted** | 583 (381) -> 0 | 41 (27) -> 0 |
| The kernel's float mask (the bias where allowed, -inf elsewhere) | quadratic outside the kernel; the layer's bias | 390 (212) -> 304 (172) | 18 (10) -> 14 (8) |
| B15's bias through the layer's table (main gather, per-axis term) | quadratic outside the kernel; the layer's table | 337 (169) -> 119 (95) | 24 (10) -> 8 (6) |
| B15's entries: the class map at (tile, codes) | quadratic outside the kernel; layer-independent, not kept | 390 (222) -> 144 (120) | 32 (18) -> 12 (10) |
| Masks: the role's rows over every key | quadratic outside the kernel; layer-independent, not kept | 203 (111) -> 176 (96) | 54 (30) -> 27 (15) |
| Row and column selections | free (views) | 126 (0) -> 126 (0) | 4 (0) -> 4 (0) |
| **A layer** | | **4,689 (2,696) -> 3,003 (1,602)** | **458 (265) -> 342 (203)** |

Under the layer-shared arm the two B15 rows go to 0 after the first layer (default 4,676 -> 2,727,
one-block 445 -> 309): the bias is assembled once. What remains per layer is, on the default path,
the per-call work of 46 kernel calls (60% of a layer) and the per-pair tensors outside the kernel
(25%) - masks, entries and biases built per query block over every key - and on the one-block path
mostly the layer's own work. **The per-pair rows are the in-kernel score function's work** (§10.2,
requirement 2): kept from layer to layer for every block they would be the (head, T, T) tensors
B14 exists not to hold, so they are not hoisted; the layer-independent ones among them (masks,
B15's entries) are the compact tensors a compiled kernel would read instead.

**(c) The counted forward before and after** (16 layers, BF16; operators dispatched, compute in
brackets; kernel calls and the operators outside the layers unchanged - context 245, 258 under the
layer-shared arm):

| Configuration | Boards, batch | Forward before -> after | Layer 0 | Layers 1-15, each | Sync | Kernel calls a layer |
|---|---|---|---|---|---|---|
| n2-default (block-sparse, as shipped) | 24, 1 | 46,123 (26,729) -> **30,876 (16,818)** | 3,324 -> 2,732 | 2,821 (1,627) -> 1,844 (999) | 914 -> 202 | 26 |
| n2-default | 44, 1 | 76,443 (44,128) -> **50,071 (26,825)** | 5,624 -> 4,542 | 4,689 (2,696) -> 3,003 (1,602) | 1,517 -> 325 | 46 |
| n2-default | 44, 2 | 79,866 (46,225) -> 53,344 (28,820) | 5,807 -> 4,710 | 4,905 (2,828) -> 3,210 (1,728) | 1,610 -> 418 | 46 |
| n2-default-sdpa (one block) | 24 and 44, 1 | 7,812 (4,590) -> **6,055 (3,645)** | 458 -> 441 | 458 (265) -> 342 (203) | 147 -> 41 | 2 |
| n2-default-sdpa | 44, 2 | 7,908 (4,686) -> 6,151 (3,741) | 464 -> 447 | 464 (271) -> 348 (209) | 147 -> 41 | 2 |
| n2-default-chunked | 24, 1 | 24,372 (13,790) -> 17,690 (9,825) | 1,493 -> 1,351 | 1,493 (840) -> 1,057 (583) | 547 -> 211 | 12 |
| n2-default-chunked | 44, 1 | 40,932 (22,990) -> 29,325 (16,005) | 2,528 -> 2,261 | 2,528 (1,415) -> 1,772 (963) | 947 -> 381 | 22 |
| b15-layer-shared (block-sparse) | 24, 1 | 45,928 (26,639) -> 28,386 (14,853) | 3,311 -> 2,719 | 2,808 (1,621) -> 1,678 (868) | 914 -> 202 | 26 |
| b15-layer-shared | 44, 1 | 76,248 (44,038) -> 45,931 (23,510) | 5,611 -> 4,529 | 4,676 (2,690) -> 2,727 (1,381) | 1,517 -> 325 | 46 |
| b15-layer-shared-sdpa | 24 and 44, 1 | 7,617 (4,500) -> 5,560 (3,315) | 445 -> 428 | 445 (259) -> 309 (181) | 147 -> 41 | 2 |

(`--config n2-default-sdpa n2-default b15-layer-shared b15-layer-shared-sdpa n2-default-chunked`;
the B15 arms now have `bench.CONFIGS` names.) At 44 boards the as-shipped forward dispatches 35%
fewer operators, 39% fewer compute operators and 79% fewer named host synchronisations; the
one-block path 22% / 21% / 72%. The verdicts of the count above do not change: the one-block path
is constant in N, the block paths are not (their kernel calls are per query block), and per layer
the default is still 3,003 operators in 46 kernel calls at 44 boards against "a few tens".

**What the hoist keeps, and the proof that it changes nothing.** The memos hold integers, booleans
and Python values - never a tensor a gradient flows through - and under autograd every float
operation runs as built (per query block), so gradients accumulate as before. The exception is the
layer-shared arm at inference: its B15 rows, FP32 as built, are kept up to
`structure.DISP_ONCE_BYTES` (2 GiB, an implementation choice), beyond which a selection is assembled per layer
as before - measured, one position keeps 76.8 / 250.2 / 1,445.5 MiB at 24 / 44 / 112 boards on the
block-sparse path and 78.1 / 260.2 / 1,674.8 MiB on the chunked path (the full (8, T, T) FP32);
at 240 boards (7.5 GiB a position) about the first quarter is kept. Bit for bit against the tree
before this work (`torch.equal` on every output; the state_dict's SHA-256 too):

| Set | Cases | Result |
|---|---|---|
| Forwards, 2-3 layers: 6 implementations x {default FP32 and BF16, one position and a padded batch of three, two model seeds, the layer-shared arm, no far tables, no table, every N2 bias, every bias with the layer-shared arm, B17, the block-causal scope, no per-board / no global registers, B5 with the routing table, post-norm, G2's FP8 trunk}, and B9's head counts, `masked_head_registers="self"`, the dense fallback for every block, one-board blocks, A8, A5 (block-sparse and chunked), F2's 24 layers | 105 | all bitwise |
| Forwards, 16 layers: default at 44 boards (BF16, FP32), at 24 + 17 boards (a batch), one-block and chunked at 24, the layer-shared arm at 24 + 10 boards (block-sparse) and 24 (one-block), default at 60 boards (2 layers) | 8 | all bitwise |
| Gradients of every parameter, 2 layers: the three fast paths x {default, layer-shared, every bias, blocks kept}, BF16, explicit (1 thread) | 14 | all bitwise |
| The cached path (`CachedEvaluator`, block-causal): default, layer-shared, typed + smolgen, routing table x FP32 / BF16 | 8 | all bitwise |
| The same 105 forwards at `attn_pad_multiple=8` against the tree before (unpadded) | 105 | all bitwise |

(Gradients compared with several threads differ between two runs of the *same* tree: B15's table
gradient, accumulated by `index_put_` across threads, by up to 2.4e-7 - so they are compared on one
thread.) `test_once_per_forward.py` holds the same within this tree: the inference forward against
an as-built reference (autograd on, nothing memoised) on 4 paths x 12 arms.

**Not hoisted, and why.** (1) The masks' rows and B15's entries of a query block (block x T
integers and booleans, layer-independent): kept for every block they are (T, T) per role and
per head group - requirement 2 forbids exactly that; the compiled score function computes them
from the compact tensors. (2) The kernel's float mask and B15's bias rows under per-layer tables:
the layer's own values. (3) The global merge computed for all rows at once and sliced: a matrix
product over another number of rows may round differently - not provably bit for bit. (4) The
layers' B15 tables laid out for all layers in the context (13 operators a layer): inherent (each
layer's own parameters), and under autograd the gather's gradient accumulation order would change.
(5) The cached path (`cache.py`, not this work's file): its frozen-board passes build a context of
their own without the field, so they are not padded (their context's memo does carry across a
board's layers); its live pass makes a fresh context per layer (`dataclasses.replace(ctx,
scope=None)`), so only the structure's memo carries across its layers.

**For the GPU session.** Measure the compiled path's launches and rate at 44 and 240 boards first
(§10.2, §15.7), with and without `attn_pad_multiple=8` on the fused-kernel paths: on the AMD card
the padding recovered 2.0-2.5x on the one-block path (the AMD attempt below, through a probe's
wrapper); on CUDA the memory-efficient kernel's own alignment rules decide. The counted cost of
padding is +4.3% of the forward at 44 boards and +6.2% at 240.

### G9's bounded AMD attempt - AMD-labelled, eager, indicative (2026-10-23; no threshold, no verdict)

**What it is.** Register G9's "one bounded attempt on the AMD RX 9070 XT via AMD's PyTorch
for Windows, figures AMD-labelled" (§10.2, M19), run in one session of about an hour and a
half. Every figure here is **AMD-labelled** (machine `dev-r-amd`), of the **eager, uncompiled**
network, on a machine **loaded at 97-100% CPU** throughout by other lanes' jobs (pytest and
`t5d-tests` processes; read with `Win32_Processor.LoadPercentage` before and after each run):
indicative only. None is an FA4-class figure, none is written over a §15.3 estimate, and none
decides a verdict (issue 522). The tool: `python/tools/g9_amd_probe.py`.

**Conditions.** `dev-r-amd` (Machines above): the RX 9070 XT, `gcnArchName` gfx1201, 15.92
GiB visible, 32 compute units (the Ryzen's integrated GPU, gfx1036, enumerates as device 1 and
was not used); AMD Software: Adrenalin Edition 26.10.1 (internal 26.20.15.07-261015a; Windows
driver 32.0.32015.7029 of 2026-10-18) - the mainstream package, newer than the "Adrenalin
Edition for ROCm 10.1" (26.10.41.05, driver 32.0.31041.5005) that AMD's ROCm 10.1.0
compatibility matrix lists; it worked, so nothing was changed. AMD's PyTorch for Windows
from AMD's ROCm 10.1.0 release (2026-10-05; PyTorch 2.14.0 on Windows for Python 3.11-3.14):
`torch[device-gfx1201]==2.14.0+rocm10.1.0` from `https://stable.repo.amd.com/rocm/whl-next/`
(HIP 7.16.26385, MIOpen 3.6.1, rocm-sdk 10.1.0, NumPy 2.5.3), in a separate venv on Python
3.14.7 under the git-ignored `build/` (the project `.venv`, torch 2.13.0+cpu, untouched); no
Triton (below). Smart App Control read as off; nothing on the machine changed. The network:
lane/gate at 529098d (F1, the v1 defaults, random weights, `build_model(seed=0)`); the engine
library `build/engine/t5d.dll` built from the same engine sources. Inputs `synthetic_state(N,
N)` (one position per batch element, distinct seeds); window none; medians of 3-5 synchronised
runs after a warm-up. "pad8" marks the padded-width arm (below): a probe-local wrapper, not the
network's code.

**1. Visible and correct - works.** `torch.cuda.is_available()` True; BF16 supported. A 2048^3
matmul against FP64: relative error 2.1e-6 FP32 (the CPU 2.8e-7), 3.5e-3 BF16 (the CPU's
equal). FP8 E4M3: the dtype exists, the GPU's cast equals the CPU's bit for bit, and
`torch._scaled_mm` (tensorwise scales) runs, exact against the dequantised matmul - G2's
emulation needs only the dtype. **The F1 forward on the GPU against the same weights and inputs
on the CPU** (synthetic 2, 17 and 44 boards; three engine positions of 3, 6 and 12 boards from
seeded random play from Standard; worst over the inputs; max absolute difference):

| Attention path | Precision | Value logits | WDL logits | Policy logits |
|---|---|---|---|---|
| block_sparse (the default), sdpa, flex (eager) | FP32 GPU vs FP32 CPU | 1.1e-6 | 1.1e-6 | 3.3e-6 |
| the same | BF16 GPU vs BF16 CPU | 1.6e-3 | 7.9e-4 | 7.5e-3 |
| the same | BF16 GPU vs FP32 CPU | 3.2e-3 | 2.1e-3 | 1.2e-2 |

The last row sits inside `tests/net/test_precision.py`'s BF16-vs-FP32 bounds (0.02 value, 0.03
policy logits) - the GPU's BF16 is G1's BF16. Submit logits were not compared (no submit is
legal at those positions). `attn_impl="flex"` runs eagerly on the GPU (every score
materialised, as on CPU).

**2. torch.compile and compiled FlexAttention - do not run.** `torch.compile(model)` (Inductor)
of the v1 network fails on every attention path with `torch._inductor.exc.TritonMissing: Cannot
find a working triton installation`; a toy `torch.compile(flex_attention)` (B=1, H=8, L=2,920,
D=32, a bias `score_mod`, a board-block `BlockMask` - `create_block_mask` itself runs, 75.6%
sparse) fails with `InductorError: LoweringException: ModuleNotFoundError: No module named
'triton'`. AMD's wheel index carries Triton for Linux only (every `triton-3.8.0+...rocm10.x`
wheel is `linux_x86_64`), and AMD's 7.2.1 Windows directory carries none. `backend="aot_eager"`
runs (it generates no kernel). Getting Triton on Windows would mean a community build or a
source build, which the attempt excludes. So §10.2's inference path - torch.compile plus
FlexAttention - cannot run on this machine through AMD's stack (issue 520).

**3. Kernel rates (AMD-labelled, the GPU alone).** BF16 GEMM 4096^3: 118-142 TFLOPS; FP32 15.
The trunk's shapes, BF16 / FP8 E4M3 (`_scaled_mm`, tensorwise) at T tokens: T = 2,920 - 256x2048
0.185 / 0.114 ms, 2048x256 0.079 / 0.070 ms, 256x256 0.040 / 0.041 ms; T = 15,856 - 0.228 /
0.169, 0.178 / 0.131, 0.082 / 0.070 ms; so FP8 is 1.0-1.6x BF16 at these shapes and 1.7x at
4096^3 (GEMM level only; noisy on the loaded machine). SDPA, BF16, 8 heads of 32: L = 2,920
0.20 ms unmasked, 0.32 ms with a full float mask; L = 15,856 3.3 / 3.5 ms. **AMD's fused
attention kernels (AOTriton, precompiled) take a mask only at a head width that is a multiple
of 8**: at 32/64/96/128 with a mask 0.25-0.49 ms; at 65/97/129 with a mask "no fused" - the math
backend, 6.5-7.9 ms (L = 2,920). Flash refuses any mask; the memory-efficient kernel takes one.
The network's fused kernel calls (`attention.py`'s `_kernel`, B14/M20's widened features) are
45 and 33 wide with a boolean key mask, so every one of them falls back to the math backend.
`torch.profiler` records no device kernels on this stack, so the kernel names were read from
these probes, not from a trace.

**4. The dense evaluator, evaluations/s** (F1, BF16 unless stated; eager):

| Path | 44 boards (2,920 tokens) | 112 boards (7,408) | 240 boards (15,856) |
|---|---|---|---|
| block_sparse (default), batch 1 / 8 | 0.47-0.58 / 1.7-1.9 | 0.16-0.20 / 0.34-0.48 | 0.07 / 0.16 |
| block_sparse, pad8, batch 1 / 8 | 0.68 / 2.6 | 0.35 / 0.70 | 0.12 / 0.23 |
| sdpa, batch 1 / 8 | 1.8 / 2.8 | - | - |
| sdpa, pad8, batch 1 / 4 / 8 / 16 | 4.2 / 6.6 / 7.1 / 1.6 | 1.2 / 0.25 (batch 4) | 0.06 (batch 1) |
| sdpa, pad8, FP32, batch 1 / 8 | 2.5 / 4.8 | - | 0.04 (batch 1) |
| chunked, explicit, dense; batch 1 / 8 | 0.34 / 2.1; 1.9 / 2.9; 2.4 / 2.9 | - | - |
| flex (eager), batch 1 / 8 | 0.19 / 0.20 | - | - |

Peak allocated memory: block_sparse 0.26 GB (44, batch 1) to 5.4 GB (240, batch 8); sdpa's
materialised (B, H, L, L) masks 0.8 GB (44, batch 1), 10.0 GB (44, batch 16), 15.3 GB (112,
batch 4), 17.3 GB / 24.8 GB (240, batch 1, BF16 / FP32) - past the card's 15.9 GiB, so those
rows spill into shared system memory, which is why they collapse. **Read.** The GPU works and
the kernels are fast (above), but the eager network is not kernel-bound: at 44 boards the
forward needs about 0.26 TF (0.12 TF of projections and FFN, 0.14 TF of every-key attention,
counted as N3 counts), 3-5 ms of GEMM and attention at these rates, against 240-580 ms
measured. With launches made blocking, the time spreads over
about 10,000 small operations a forward - the per-layer displacement bias, masks and role rows
built at (B, H, L, L) or by gathers (`displacement.lookup`, `structure.role_rows`/`disp_rows`),
host synchronisations (`nonzero`, `.item()`) and the math-backend attention; under N1's dense
baseline (no masks, no table) each block still took 16-27 ms, 13-16 ms of it attention. That is
the work torch.compile and FlexAttention's in-kernel score function are meant to fuse, and they
are what does not run here. The CPU on the same network (`net/bench.py`, `dev`, before F5/F6,
B15 and M20) took 8.5-19.5 s at 240 boards; the GPU's 8.6-13.7 s is no gain worth recording as
one.

**5. One search through the C API with the GPU evaluator - works.** `run_search` with
`NetworkEvaluator(model, max_batch=8, device="cuda")` from Standard's start
(`search_params(seed=1, batch_size=8)`, so positions of one or two boards), 200 playouts, leaves
in batches of up to 8 (26 rounds), stopped on playouts:

| Evaluator | Playouts/s | Wall (s) | Of which evaluating (s) |
|---|---|---|---|
| GPU, BF16, block_sparse | 11.1 | 18.0 | 17.9 |
| GPU, BF16, sdpa, pad8 | 16.9 | 11.9 | 11.8 |
| GPU, FP32, block_sparse | 12.6 | 15.9 | - |
| CPU (the same venv's torch 2.14, 12 threads), BF16 | 6.4 | 31.4 | 31.3 |
| CPU, FP32 | 2.1 | 95.5 | - |

The CPU rows ran on a CPU that other jobs held at 100%; they show the load, not the CPU's rate.
Collecting and delivering leaves took 0.1 s of each run: the search waits on the network.

**The attempt's answer: partly.** AMD's PyTorch for Windows runs the network on the RX 9070
XT, correct against the CPU in FP32 and BF16, and drives the C++ search through the C API; FP8
GEMMs run. torch.compile and compiled FlexAttention - the committed inference path (§10.2) -
do not, for want of Triton on Windows; the eager network is overhead- and memory-bound, so its
rates say nothing about the §15.3 rows (FA4-class evals/s, leaf batching, the block-sparse and
varlen kernels), which stay deferred to NVIDIA time (M19). The head-width finding (pad8: 2.0-2.5x
on the sdpa path at 44 boards, 1.2-2.2x on block_sparse) is device-neutral in form and is the
network lane's to take or leave (issue 521).

### F1's CPU evaluation rate - self-play through the engine (command; not yet measured on a quiet machine)

**What it will measure.** §15.6 cites "the only rate today is the CPU's, 6.9 evaluations/s for
F1". The figure comes from stage N5's end-to-end run (CHANGELOG, 2026-10-04, machine `dev`: 20
games, F1 at 21,359,370 parameters in BF16, 0.60 decisions/s and 6.9 evaluations/s). Its
command, games, variants and threads were not recorded, and it predates E10's provers and the
flat form. Since 2026-11-28 (`lane/search`) `python -m t5d.engine.evalrate` gives the figure a
command. It plays a fixed number of self-play games with the v1 network at random weights
through the engine (`netplay.play_game`: the engine's search through the C step API). Seeds,
variants, splits and threads are fixed.

The provers are off by default: no Hunt, no settling scans, no asset map. The figure is the
network's own rate within the search. `--provers` turns all three on. The game loop's exact
questions (H22's turn-start verdict and move verification) stay on; they are the game's.

Its first output line is the full configuration: the network's every field and parameter count,
the game settings, the seeds, PyTorch's threads, the machine, the engine library's file name and
the commit. Then one line per game, then the totals:
- `decisions_per_s` and `evaluations_per_s` over the games' wall time, as N5 reported them;
- the network's forward passes (`forwards`) and the positions they evaluated, with
  `positions_per_forward` and `forwards_per_s`;
- `network_positions_per_s`, positions over the network's own seconds, with `network_share`;
- `by_range`: the network's calls by board-count range (6b″'s ranges), each call filed under its
  largest position, with forwards, positions, seconds and positions/s. §15.6 reads a rate per
  board-count range; the pooled figure is the games' mix of sizes.
- since 2026-11-29 (`lane/engine`), where an evaluation's time goes - in each game's line, in the
  totals and in each range of `by_range`: the seconds of the conversion, the fit, the collation,
  the transfer, the forward and the split, with `assembly_ms_per_position` and `assembly_share`
  beside `forward_ms_per_position` and `forward_share` ("Input assembly per evaluation", below).

**Command** (from `python/`; the defaults written out):

```
python -m t5d.engine.evalrate --net F1 --games 1 --variant Standard --variant "Standard - Two Timelines" --split ply --threads 4 --seed 1
```

That is 2 games: Standard and Standard - Two Timelines, the ply split, 16 playouts and batches of
8 per decision, I11's caps at 6 turns or 80 boards (netplay's settings, as at N5), seeds 1 + 101
g, and 4 PyTorch threads (netplay's default and the runbook's candidate).

**Run time: not known.** On the shared machine (2026-11-28, load about 50% beside it), a 2-turn
Standard game (`--cap-turns 2`) took 70 s: 16 decisions, 178 evaluations at 2.5/s, 99.9% of the
time in the network, 15 boards at its end. A default 6-turn Standard game ran past 50 minutes
and was stopped. Today's v1 forward costs far more on the CPU than N5's network did: B15's bias
and M20's register rows came after N5. Budget an hour or more for the two default games, or
bound the run with `--cap-turns`. The figure is CPU-labelled and informational. §15.6's Phase-3
re-check stays deferred until a GPU rate exists.

### Input assembly per evaluation, by board-count range (commands; not yet measured on a quiet machine)

**What it will measure.** The CPU half of §15.3's row "Input assembly and host-to-device
transfer per evaluation" (no estimate; "to measure, by board-count range"; build-log issue
420, proposed edit 435). Making the network's input from a search leaf is three steps on the
CPU, whatever the device: the step driver's **conversion** (`CStepDriver`: the leaf's
expansion copied out of the engine, then `to_position`), the evaluator's **fit**
(`NetworkEvaluator._fit`: a re-layout, or register A5's conversion at the network's edge;
nothing for positions made with the network's own layout) and the **collation**
(`collate`: the padded batch's tensors). The **transfer** - the batch moved to the device -
is the row's other half: on the CPU there is nothing to move, and the figure waits for a GPU
(§15.7's Gate 1; M19).

**Two commands** (both from `python/`):

1. *Assembly alone, by range - no model, no forward, minutes.* Positions by seeded random
   pseudo-legal play from Standard through the binding (for cost only), at 10, 30, 44, 60,
   80, 120, 160 and 240 boards, three seeds each, filed in 6b″'s ranges; per position the
   engine's `expand` (the expansion and its copy out - given beside the assembly, not in it:
   the search pays the expansion in its gather), `to_position` as the step driver calls it,
   the fit, and `collate` for batches of 1 and 8 of one board count; **assembly =
   `to_position` + fit + `collate`**, per position: the median and 95th percentile over a
   range's (position, repeat) samples.

   ```
   python -m t5d.engine.bench --assembly --assembly-boards 10 30 44 60 80 120 160 240 --batch 1 8 --seeds 3 --repeat 5
   python -m t5d.engine.bench --assembly --assembly-boards 10 30 44 60 80 120 160 240 --batch 1 8 --seeds 3 --repeat 5 --piece-centric
   ```

   The second line is register A5's tokens (the fit is then `to_piece_centric`). One torch
   thread is enough: nothing here is a tensor kernel but `collate`'s copies.

2. *Inside real games, beside the forward.* `python -m t5d.engine.evalrate` (the F1 command
   above) now prints, in each game's line, in the totals and in each range of `by_range`:
   `convert_seconds`, `fit_seconds`, `collate_seconds`, `transfer_seconds`, `forward_seconds`,
   `split_seconds`; `assembly_seconds` (conversion + fit + collation),
   `assembly_ms_per_position` and `assembly_share` of the six parts' sum; and
   `forward_ms_per_position` and `forward_share`. A batch's conversion is filed in the range
   of its largest position, as its call is. This is the mixed-size batch a search makes (its
   padding is the largest position's); the first command's batches are of one size.

**Conditions to record with a figure.** The machine and its load (the first command does not
control it and says so), Python, PyTorch and NumPy, the engine library, the register counts
and the tokenisation - all printed. CPU-labelled; nothing here needs a GPU, and nothing here
is the transfer.

## §15.6 verdicts

### Movegen throughput (M9) - pseudo-legal: PASS; with fast legality: inside the STOP band at 240 boards on one core

**Conditions.** dev, quiet machine (the bench measured the machine busy 4-5%
during 1-thread samples - itself), Windows 11, g++ 16.2 `-O2 -DNDEBUG`, static
`t5d-bench --samples 15` at b5c1176 (15 interleaved rounds of 200 ms after a
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
classification inside, one thread, `t5d-bench --legal-profile` at E5a's ed70ca9,
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
--legal-budget 0,0 --no-flatness` at ad2769d; 1-thread samples at 5% load, the
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


2026-10-03, engine at c7840d8 (Release, g++ 16.2, 383 tests passing). The data:
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


`t5d-measure --summary` over the 41 corpus records (2026-10-02, b5c1176),
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

2026-10-04. Engine at f3d36c9 (E9 merged: Guard, Hunt, the asset map, Hunt's
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

**Guard's and Hunt's nodes/s - command; not yet measured on a quiet machine.** The
run above gave nodes per root, not nodes per second (§15.3's "Prover nodes/s" row
quotes nodes per root at 50 ms). Hunt often stops early - it proves a mate or
exhausts its class (median 676 nodes under 50 boards) - so nodes over the 50 ms
budget would understate it. Since 2026-11-28 (`lane/search`) the tool times Hunt
at each root as Guard is timed - from the call to its return - and writes it
(`"seconds"` in each position's `"hunt"` object); its summary prints, per range,
Guard's and Hunt's sum(nodes) / sum(seconds) ("NODES PER SECOND"), and the JSON's
per-range `guard` and `hunt` entries carry `timed_roots`, `nodes`, `seconds` and
`nodes_per_s`. The re-run is the command above (from the repository root, after
`build/corpus.txt` is rebuilt as `engine/README.md` says):
`build/engine/t5d-scan6bpp --truth engine/testdata/hunt-truth --record-list
build/corpus.txt --per-position --json build/6b/6bpp-quiet.json --summary
build/6b/6bpp-quiet.txt` - about a minute (58.4 s on 2026-10-04). It re-reads
M17's verdicts too (the lock changed since: build-log issue 147).

**Hunt's ground truth** (ruling 5; `engine/testdata/hunt-truth/`, recorded at
b1d0ca9, 5,097 s on 6 workers): one long offline search per root - the union of
the literal class's (every first move, every completion) and the default class's
proofs, 4M nodes each, a 2M-node tree cap, one thread and fresh stores per root,
deterministic. The budget from a 101-root curve (mates found against nodes: 5, 6,
8, 8, 9, 11, 11, 11, 12, 12, 12, 12 at 2k ... 4M) that found no new mate past
500k. 825 roots, 805 measured, 125 mates (71 / 33 / 20 / 1 by range; 99 mates in
one, 16 in three, 10 in five; both classes 88, the default alone 32, the literal
alone 5), every one confirmed by E4. Re-recorded with the reviewed prover (4,962
s): every verdict equal.

### The verification redesign: 6b′ (M14, M15) - 50-199 BOARDS FAIL - M15 OUTCOME: STOP

2026-10-03. Engine E8 (the M14 prototype) at 9c8916d on its branch
`worktree-agent-a4f2f93a8a448467b` (main 61c6480 merged in; not yet merged into
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

**The re-run (M12), 2026-10-03.** Engine at e88cf23 (E5c merged: the bounded
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
tool itself), g++ 16.2 `-O2 -DNDEBUG`, engine at ad2769d (E5b's scanner merged with
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
machine's load does not matter. The rules engine is `main`'s at 65b8cce, unchanged;
the counting is `t5d-measure --turns --turn-summary` (`engine/tools/turn_width.hpp`)
at c4b45eb (lane/engine: M22, f5948ab, and its review fixes; rebuilt at that commit,
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
section was first written from the tool at f5948ab. Against that run, the review
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
shared machine's load does not matter. The engine is `main`'s at da96457 (after H51,
across which M22's command was byte-identical); the counting is `t5d-measure --turns
--turn-summary` with this entry's additions to `tools/turn_width.hpp` (the commit that
adds this paragraph, on da96457), Release, MSYS2 g++ 16.2.0 (Rev3), `--werror`, 583 of
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
tree form's review fixes (the tree form, af7aef0, and the fixes; a fresh root's
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

### D2's tree form: evaluations per complete turn and inherited expansions per evaluation, by board-count range and mandatory count (command; not yet run)

**What it will count.** §15.3's row "Tree form: evaluations per complete turn, and inherited
expansions per evaluation, by board-count range and mandatory count" (no estimate; "to
measure (CPU)"). The seven turn starts above are indicative and of one range; this is the
row's own count. `t5d-tree-count --by-range` (`engine/tools/t5d_tree_count.cpp`; the cells,
the turn starts taken and the ratios in `engine/tools/turn_cells.hpp`; build-log issue 418
for the readings - the build's, for the user to confirm; the tool prints them with its
figures):

- **evaluations per complete turn** = (evaluated turn starts below the root + evaluated
  mid-turn nodes) / evaluated turn starts below the root. A complete turn is a submission
  reached and the position after it evaluated; the root's own evaluation is left out on both
  sides. Exactly 1 where no mid-turn node is evaluated (no gate); the ply split's is its
  moves per turn plus one, over the paths searched.
- **inherited expansions per evaluation** = mid-turn nodes expanded without the network
  (`SearchStats::inherited_nodes`) / evaluations: what the tree form still pays in legality
  classifications for each evaluation it makes.
- **cells**: 6b″'s ranges (< 50, 50-99, 100-199, 200+ boards) by 1, 2, 3 and 4 or more
  mandatory boards at the turn's start; per arm and cell the turn starts counted and each
  ratio's median and 95th percentile over them, and each range over every mandatory count.

**Conditions (as the command gives them).** Counts, not timings: deterministic, the same on
any machine and under any load - no quiet machine is needed. Data: t0database, converted by
`py data/t0database/to_5dpgn.py build/t0db` (761 records), listed in the glob's order.
Turn starts: those a record plays a further action from (so none is a game end), at action
indices that are multiples of 4, at most 25 in a cell and one of a record in a cell - a
function of the records' order and those numbers. The human games have no turn start at 200
boards or more, so that range will be empty. Arms: the ply split; the tree form with §8.2's
four gates at their placeholder readings; with the present gate read as rewinds only; with
no gate. Each a fresh search to 400 evaluations (playouts unbounded): the random evaluator
(seed 1), synchronous, no evaluation cache, square features off, every other parameter the
search's default - the seven-position table's search.

**Command** (from the repository's root; the defaults written out):

```
py data/t0database/to_5dpgn.py build/t0db
ls build/t0db/*.5dpgn > build/t0db.txt
build/engine/t5d-tree-count --by-range --record-list build/t0db.txt --every 4 --max-per-cell 25 --per-record 1 --evaluations 400 --arms ply,tree,tree-rewind,tree-none --threads 8 --json build/tree-count-by-range.json > build/tree-count-by-range.txt
```

`--threads` shares the turn starts' searches out; each search is its own, so the counts do
not depend on it (the tool's standard output is the same bytes on one thread and on three:
`tree_count_exe_by_range_counts_what_it_lists`).

**Expected run time: about 12 minutes on one thread, 2 to 3 minutes with `--threads 8`.**
From a timed sample (2026-11-29, `dev-r`, a shared machine - the network lane's tests beside
it; a time, not a figure of the row): the same command with `--max-per-cell 2 --threads 1`
counted 23 turn starts by 4 arms in 55.9 s - 2.4 s a turn start, 0.6 s a search - after
reading the 761 records in a third of a second (35 of them refused or not read to their
end; 29,282 turn starts played on from, 7,617 eligible at every fourth action). The full
command counts at most 300 turn starts (12 cells the human games can fill, 25 each; the
sample found one cell, 100-199 boards with three mandatory boards, short of two at one a
record). The wall time is printed on standard error. The sample's own counts are not
recorded here: two turn starts a cell are not the row's figure.

### D2's flat form: the turn generator's time to its first k turns and per widening step, by board-count range and mandatory count (command; not yet measured on a quiet machine)

**What it will measure.** §15.3's row "Turn generation (flat form): time to the first k
legal turns, and per widening step, by board-count range and mandatory count" (no estimate;
"to measure (CPU)"). `t5d-turngen-time` (`engine/tools/t5d_turngen_time.cpp`; what is timed
and how it is summed up in `engine/tools/turngen_time.hpp`; build-log issue 419 for the
readings - the build's, for the user to confirm; the tool prints them with its figures)
times one generator a position, made and asked as the search's flat form makes one at a
node:

- **the time to the first k turns**, k = 1, 8, 32: from just before the generator's
  construction (the state's copy, its axes, the scores) to the return of the `next()` that
  gives turn k;
- **a widening step**: one `next()` after the first 16 turns (the search's cap on what an
  expansion generates), 16 of them.

A position with fewer than k legal turns, or whose generator its node budget stopped first,
is counted and not timed at k.

**Conditions (as the command gives them; the tool prints them all).** One thread, pinned.
t5d-bench's method: every position timed once a round, 7 rounds interleaved over all of them
after a warm-up round; a position's figure the median over its rounds; a cell's the median
and 95th percentile over its positions (the times to k) or over their every widening step;
the machine's busy fraction read around every round and printed, with the compiler, the
flags and the CPU. The generator as the search runs it: successors, a node budget of 200,000
(placeholders there). **The scores are not a network's**: a seeded function of each option's
coordinates, uniform in [-3, 3] - what best-first generation costs depends on the scores,
and a trained policy's are the pilot run's to time (`--flat-scores` gives the all-ties
case). Positions: the human games' turn starts chosen as above (every 4th action a record
plays on from, at most 20 in a cell, one of a record in a cell), and - reported apart, never
in one figure - t5d-bench's synthetic classes (100, 160, 240, 400 and 600 boards; random
play, for cost only), two games a class.

**Command** (from the repository's root, on a quiet machine; `build/t0db.txt` as above):

```
build/engine/t5d-turngen-time --record-list build/t0db.txt --every 4 --max-per-cell 20 --per-record 1 --synthetic --games 2 --seed 1 --rounds 7 --k0 16 --widen 16 --generator-nodes 200000 --score-scale 3 --score-seed 1 --pin 2 --per-position --json build/turngen-time.json > build/turngen-time.txt
```

**Expected run time: seconds.** A smoke run on the shared machine (2026-11-29; two turn
starts a cell and one synthetic game a class, 2 rounds - 28 positions by 3 rounds) took 0.4 s
in all, the timed rounds 0.1 s of it; the full command is about ten times the positions and
three times the rounds. It is bounded by the node budget: a generator that spends all 200,000
nodes takes under a second. That run is not a measurement (the machine was 60-70% busy) and
its times are not recorded. What it showed about the figures' shape, which the tool's output
now carries: a generator records up to 32 realisations of one point in a step, so many a
`next()` returns a turn already found - the **median** widening step can be near zero while
the cost sits in the tail - and the tool prints the steps' **mean** beside their median and
95th percentile; and two of the three synthetic positions at 200 boards or more had no legal
turn at all (random pseudo-legal play), which the "exhausted" column counts. The arithmetic
is tested (`engine/tests/test_turn_cells.cpp`: the medians, percentiles and means, a turn no
position reached, and the tool run on records written there); no figure is recorded yet.

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
`rewalk3.cpp` against lane/engine 943e0d7, the same position and scorer); *after*: the test
at the review fixes. (b) `mcts_the_memory_estimate_follows_the_heap`
(`engine/tests/test_mcts_memory.cpp`): Standard - Two Timelines after six actions (each the
hypercuboid listing's first), the random evaluator (seed 7), batch 8, square features off,
no evaluation cache, every other parameter the default; 400 playouts under the flat form,
2,000 under the ply split; the heap counted by the tests' allocation hook. *Before*: as the
flat form's search arm (c39e253) recorded it in build-log issue 171; *after*: the test's
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
| flat form, before (c39e253, issue 171) | 400 | - | - | about 137 KB | x1.23 |
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

### G12's merged forwards against a position alone - numerics (2026-11-22; no threshold, no verdict)

**What it is.** How far an evaluation moves when its position shares a forward with other
positions, as the self-play pool's token-budget batches make it (`python/t5d/engine/pool.py`,
register G12) - the figure that conditions what "the same record" means for a pooled game. Not a
speed and not the batching gain (a GPU figure, Gate 1 step 2). Machine `dev-r`; PyTorch 2.13.0+cpu,
12 torch threads; `netplay.SMALL` (2 layers, d = 64, FP32 trunk), random weights at initialisation
scale; window none. Numerics do not depend on the machine's load (other jobs were running).

**One request through the planner** (`TokenBudgetEvaluator`, the placeholders as shipped: 65,536
padded tokens, 64 positions, 25% padding): `tests/engine/test_pool.py`'s six positions -
`synthetic_state(40 + i, N)` for N = 12, 2, 3, 7, 3, the third offering a submit, the second twice
- planned as two forwards (808 + 478 tokens; 148 + 214 + 214 + 148). The largest difference of any
head (policy, WDL, value and board logits, submit logit, error, moves-left) from the same position
evaluated alone, model seeds 0 to 3: **4.2e-7, 3.9e-7, 1.0e-6, 7.2e-7**. Never zero; the test's
bound is the network's FP32 batch-equivalence tolerance, 1e-5 (spec section 15.2's padded-batch
row; `tests/net/test_invariance.py`). With merging off the evaluations are the inner evaluator's
bit for bit.

**Whole games in the pool** (`train.selfplay_pool.generate_games`; the tiny games of
`tests/train/test_selfplay_pool.py`: its eight `CONFIGS` with the exact evaluator; with the network,
model seed 0, its three `NETWORK_CONFIGS` - the ply split, the tree form, a flat game on Two
Timelines, 8 playouts, turn cap 2). The ladder ran as built in every game:

| Evaluator, batching | Runs | Pooled record = the game alone |
|---|---|---|
| exact per position (`DigestEvaluator`), merged across games (budget 512, held), 8 games | 65 | every game, every run, byte for byte |
| the same, the shipped placeholders (greedy) on 3 workers; and merging off | 65 each | every game, every run |
| the network, merging off, 3 games | 6 | every game, every run |
| the network, merged (budget 4,096, 16 positions, held up to 2 s) | 8 (two probes of 4) | **no game in any run** |

In the last row the records differed in the recorded raw outputs and in the search statistics
computed from them: 54 to 92 fields a game in the first probe; raw logits by at most 4.5e-7 in the
second, whose twelve pooled games nevertheless played the same moves as the games alone (the same
5dpgn) and whose runs agreed with one another in one game of nine comparisons - which positions
share a forward follows thread timing. That the moves stayed the same is an observation at this
scale (random weights, 8 playouts), not a property: a last-bit difference can change a choice.

### The per-square labels and the pin and check relations - cost per state (indicative; no threshold, no verdict)

**What it is.** The wall time of D4's per-square label generator (`t5d_game_square_labels`
through `Game.square_labels`, engine ABI round 3) on one state, and what B11's pin and check
relations add to an expansion - before and after the round's review fixes, which read the
exchange label's recaptures on the board a capture creates off the passed boards (build-log
spec issue 176(a)). No threshold applies and none is proposed: §12.4 asks for the labels,
not for a speed; the figures say what a training replay pays per decision.

**Conditions.** 2026-10-23, machine `dev-r`, Release, MSYS2 g++ 16.2.0 (Rev3), one thread,
through the Python binding (ctypes; Python 3.14.7). The shared machine was not quiet: other
sessions' work ran beside it (not controlled); the review's mutation runs had ended. Positions:
`random_position_game("Standard", seed, boards=n, travel_prob=0.5)` for n = 10, 40, 100, 200
and seeds 0-2 (the first position of seeded random play with at least n boards; random play
is the engine's own, identical in both builds). Each call timed three times, the minimum
reported (the review's first figure, "0.3-38 ms per state", was one timing each). Labels:
the default options (200,000 mate nodes per search). Relations: `expand(square_features=
"none")` with attack, defence and x-ray, then with pin and check added. *Before*: the
review's copy of lane/engine 4a80ad5; *after*: the review fixes (this commit). Probe:
`probe_cost2.py` (scratch; the review's `probe_cost.py` with repeats and the exchange
counts).

| boards | first moves (ours, theirs) | mate searches | labels before (ms) | labels after (ms) | exchange = 1 before / after | expand, 3 relations (ms) | pin + check added (ms) |
|---|---|---|---|---|---|---|---|
| 11 | 75, 66 | 0 | 0.1 | 0.1 | 1 / 1 | 0.04 | 0.01 |
| 10 | 95, 26 | 0 | 0.3 | 0.2 | 1 / 0 | 0.04 | 0.01 |
| 10 | 118, 123 | 0 | 0.2 | 0.2 | 0 / 0 | 0.04 | 0.02 |
| 41 | 300, 189 | 0 | 0.5 | 0.5 | 6 / 5 | 0.06 | 0.02 |
| 48 | 460, 301 | 0 | 0.9 | 1.2 | 13 / 3 | 0.12 | 0.00 |
| 40 | 215, 164 | 0 | 0.7 | 0.8 | 10 / 2 | 0.05 | 0.02 |
| 109 | 1,124, 111 | 2 | 3.8 | 3.3 | 47 / 22 | 0.10 | 0.05 |
| 121 | 1,369, 723 | 113 | 25.7 | 22.7 | 35 / 15 | 0.14 | 0.12 |
| 103 | 1,050, 634 | 2 | 4.1 | 6.2 | 37 / 16 | 0.14 | 0.13 |
| 212 | 2,118, 536 | 60 | 17.0 | 18.7 | 110 / 49 | 0.20 | 0.21 |
| 208 | 1,852, 840 | 0 | 4.2 | 7.1 | 69 / 24 | 0.18 | 0.18 |
| 211 | 2,247, 79 | 2 | 6.9 | 7.4 | 65 / 37 | 0.19 | 0.13 |

(The expansion columns are the *after* run's; the *before* run's agree within the noise, its
pin and check adding 0.00-0.19 ms. No exchange label was Unknown in either run, and no mate
search stopped on its budget.)

*Read.* The labels cost 0.1-23 ms per state after the fixes (0.1-26 before): most of it the
second moves (tens of thousands at 200 boards) and, where many moves end a legal action in
check (113 mate searches at 121 boards), the mate searches. Replaying each capture off the
passed boards adds up to about 3 ms at 100-200 boards (a played capture, a pass and one
generation of the occupant's side's moves per replayed square) and halves the squares labelled
1 (the label no longer means "attacked" there). Pin and check add 0.0-0.2 ms to an expansion
of 0.04-0.2 ms - up to about double, small against the labels. A training replay labelling
every decision pays the labels' cost once per decision.

### Cached = dense on every engine variant (MT-04) - the gaps met (numerics; no threshold, no verdict)

**What it is.** The largest differences between the cached path (`python/t5d/net/cache.py`) and
the dense block-causal pass it must equal (`reference_output`; through the C step API,
`NetworkEvaluator(scope="block_causal")`, or with a root refresh the two-level reference) on real
engine positions of every named variant - what `python/tests/engine/test_cached_variants.py`
asserts against, recorded so that the next re-measurement of §15.2's tolerances (EQ-01) has the
real-variant figures beside the synthetic ones. No threshold applies: the bounds are
`tests/net/test_cache.py`'s and `tests/engine/test_network.py`'s.

**Conditions.** 2026-10-23, machine `dev-r`, PyTorch 2.13.0+cpu, Python 3.14.7, the engine library
built Release in `lane/gate` (main 03f4253); one position per evaluation; random weights at
initialisation scale; window none; lineage reading (C23). Positions: seeded pseudo-legal random
play from each of the engine's 16 variants (`_walk`: seeds 0 and 1, to the first turn start with
16 boards, every state of every action), the store shared along each game in one epoch opened a
third of the way in; the step API's leaves from a random-play root with 8 or more boards, 16
playouts per search (64 in the transposition test), tree and graph mode, evaluation cache off.
FP32: 2 layers; the v1 configuration and the step-1 one (A8, D4's reply head, D9 on), both C20
readings; seven N2 flag sets with `relations_for`'s relations on three variants. BF16: F1 (16
layers, block-causal), model seed 0, every second state of seed 3's play on Turn Zero, Reversed
Royalty and Timeline Strategos.

| Class | Cases | Largest gap | Bound |
|---|---|---|---|
| FP32 trunk, cached vs one full pass | 32 variant-configurations x 2 seeds x 2 readings; 7 N2 sets x 3 variants | 1.91e-6 (max abs) | 2e-5 |
| FP32 heads (value, WDL, submit readout, policy, per-board; D4's reply from the review fixes on) | the same | 1.67e-6 (the N2 sets alone 1.37e-6); 1.91e-6 with the reply head and the review fixes' store tests | 1e-4 |
| FP32 evaluations through the step API (policy, WDL, categorical logits) | 16 variants + 2, x causal / root refresh x tree / graph | 1.43e-6 | 1e-4 |
| Leaves met twice (transpositions; the second search's shared leaves) | the same | 0 (bitwise) | bitwise |
| BF16 trunk, cached vs the BF16 full pass | 3 variants | RMS 1.97e-3, max 1.15e-2 | RMS 8e-3, max 5e-2 |
| BF16 policy logits / value logits | 3 variants | 7.18e-3 / 1.32e-3 | 3e-2 / 5e-3 |

*Read.* The real variants' FP32 gaps are the summation-order flips `test_cache.py` records on
synthetic trees (under 2.4e-6 after 16 layers); the BF16 gaps match `test_network.py`'s on the
'cross_timeline' record (RMS 2.0e-3 / max 1.4e-2, policy 8.0e-3, value 1.1e-3) - the value's
1.32e-3 a little above that figure, 3.8x under its bound.

### Exact move legality's cost - issue 406's positions (indicative; no threshold, no verdict)

Build-log issues 406-408. `t5d-legality-cost` (engine 0.1.0, `lane/engine`, the tree of issue
406's commit), single-threaded, on `dev-r` with other lanes' jobs running - load 99-100%
throughout (shared machine: wall-clock seconds are indicative). Fixtures in
`engine/testdata/legality-cost/`; each record's `LegalityCost` header says where it comes from.
"Budget" is search nodes for the classification's tier 3 (checks unlimited unless stated) and
for E4's detector (hypercuboid). 2026-11-01.

| Fixture (rules) | Position | Classification | E4's detector |
|---|---|---|---|
| `q3-root` (Q3, play activation) | 5 boards, 3 playable timelines, 35 moves | settled at 2,000 nodes (240 spent, 0.001 s): 17 legal, 18 illegal | ongoing, 39 nodes |
| `q3-flat-node` (Q3) | 18 boards, 7 playable timelines, 141 moves | tiers 0-2 settle 75 (illegal); unknown 66 / 57 / 50 at 2,000 / 200,000 / 2 million nodes (0.004 / 0.43 / 3.4 s); **exact: 11,910,238 nodes, 19.0 s** - tier 3 proves 18 legal and 43 illegal (one exhaustive search each, 0.2-0.53 million nodes), 5 learned. The search's default budget (64 checks, 2,000 nodes): 78 unknown in 0.004 s | unknown at 2,000 nodes; ongoing at 43,424 nodes (0.09 s) |
| `vso-seed4-verify` (standard) | 192 boards, 48 playable timelines, 940 moves; one named (the move a lazy verification asked about) | unknown at 2,000 / 200,000 / 2 million nodes (0.06 / 0.75 / 6.6 s, checks 4,096); **exact: unknown when cut at 600 s (290 million nodes)**; a scratch probe on the same code, uncut, was stopped at 30 minutes | - |
| `vso-seed4-verdict` (standard) | 191 boards, 49 playable timelines, 982 moves (the turn start after 26 actions) | tiers 0-2 settle 617 (illegal); 365 unknown at 2 / 20 / 200 million nodes (4.0 / 38 / 352 s) | **unknown at 2 / 20 / 200 million nodes (3.8 / 37 / 346 s)**: 68 / 532 / 3,939 points judged, 63 / 518 / 3,857 of them unrealised, 64 / 528 / 3,935 abstract checks given up, 272 / 2,222 / 16,602 unrealisable boxes; the game's own verdict (no node limit) ran past 20 minutes |

Searches from `q3-root`, the flat form, 800 playouts, the uniform evaluator, E10's ladder off:
the search's default legality (64 checks, 2,000 nodes) 55.2 s (terminal checks 22.9 s, turn
generation 30.8 s, expansions 1.5 s); legality within 2 million nodes (MT-01's long twin) 131 s
(expansions 78 s); exact legality with a 60 s deadline - stopped at 60.1 s after 329 playouts
(expansions 41 s). Before the deadline reached the classification, the exact search ran past 17
minutes (issue 406). From `vso-seed4-verify`'s node, 4 playouts of the ply split at the search's
defaults: 2.4 s, 2.2 s of it in the lazy verifications (200,000 nodes each).

The game: `t5d-play --variant "Very Small - Open" --evaluator material --playouts 32
--cap-turns 14 --seed 4` (E10's ladder and the asset map on). Before the fix, with
`--max-seconds 900`: killed by an outer timeout at 40 minutes, inside the turn-start verdict
after 26 actions. After it, with `--max-seconds 120`: ended at 120.0 s, `time_limit`, 26 actions
(150 decisions, 4,800 playouts); search's terminal checks 46.2 s, leaf expansions 10.8 s,
verifications 62.8 s.

*Read.* Every budget is honoured - a node costs 1.5-3.5 µs at every position measured, the
time limit is read every 64 nodes - so the minutes come from the work the exact questions
require: illegality proofs one exhaustive search per move (Q3's position), and at ~190 boards
points that no creation order realises, judged one by one because the abstract check gives up
at its step limit (4,096) on nearly every box. What would change them is algorithmic (issue
407); whether the play loop's exact steps should carry a node budget is a decision (issue 408).

### An unscored game at 88 boards under the game loop's default budget (indicative; one game; no threshold, no verdict)

Build-log issues 413 and 414 (register H22, review decision 13). The engine's own test
`selfplay_games_end_by_checkmate_stalemate_or_the_cap` (`engine/tests/test_selfplay.cpp`), its
flat-form game from Very Small - Open: the material evaluator, 12 playouts per decision, seed 2, the
turn cap 12 turns or 300 boards, self-play's root noise, the asset map off, the ladder's defaults,
square features none; the game loop's node budget the default 2,000,000 (`kTerminalNodes`, a
placeholder). `lane/engine` at 193b517 and again with the review's fixes (73e6430), `dev-r`, other
lanes' jobs running - load 97-100%: wall-clock seconds are indicative. 2026-11-23/24.

| Game | Actions | Final position | End | Its turn-start verdicts | The game |
|---|---|---|---|---|---|
| Very Small - Open, flat form, seed 2 | 18 | 88 boards | unknown (the verdict: unsettled within 2,000,000 nodes) | 2.94 s / 3.33 s | 54.6 s / 56.3 s |

*Read.* One game of the test's ten, the only one that ended unscored; before the ruling its exact
verdict ran until it settled (the suite's slowest test). It shows the placeholder can end a game
well below the turn cap's 300 boards and near the generator's 80; how often, by board count, is the
pilot run's to measure (issue 413, for the user).

### The cached path at engine roots - C3's anchor (issue 259) (numerics; no threshold, no verdict)

**What it is.** How often a search root's pseudo-legal and legal move lists differ - where C3's root
refresh, anchored on `Game.position`'s default (pseudo-legal) expansion before issue 259, read
another move graph than the root leaf - and what it cost the root leaf's evaluation, which section
7.9 makes the dense pass. Recorded so that issue 259's fix has its evidence beside it.

**Conditions.** 2026-11-08, machine `dev-r`, PyTorch 2.13.0+cpu, Python 3.14.7, NumPy 2.5.2, 2 torch
threads, the engine library built Release in `lane/net` (main d6cbdb7); FP32, 2 layers, the v1
configuration under the block-causal scope, random weights at initialisation scale. Roots: the
first live turn start with 8 boards or more of seeded pseudo-legal random play
(`test_cached_variants._walk`, seeds 5-24), the search's default legality budget (64 checks, 2,000
nodes - here the same lists as exact). Gaps: the largest absolute difference over the root leaf's
move, WDL and categorical logits between the cached path with C3's root refresh (one playout
through the C step API, ply / tree / flat alike) and `NetworkEvaluator(scope="dense")` on the same
leaf, at the first root of `_differing_root` (Two Timelines seed 23, 8 boards; Timeline Battleground
seed 8, 11 boards), model seed 5 (C20's `causal_past`) and 6 (`past_and_epoch_history`).

| Variant | Roots whose two lists differ | Moves, pseudo-legal / legal, where they differ |
|---|---|---|
| Standard - Two Timelines | 1 of 20 | 52 / 50 |
| Misc - Timeline Battleground | 9 of 20 | 114/112, 89/78, 89/54, 86/59, 87/79, 72/71, 94/91, 152/150, 87/86 |

| Root leaf against the dense pass | Two Timelines (R2 / R1) | Timeline Battleground (R2 / R1) |
|---|---|---|
| Anchored on the pseudo-legal expansion (main before issue 259) | 1.09e-4 / 2.37e-5 | 3.57e-4 / 4.47e-5 |
| Anchored at the root leaf's input (the fix) | 4.77e-7 / 4.47e-7 | 8.34e-7 / 7.75e-7 |

*Read.* With the anchor right, the root leaf is the dense pass to its summation order (under 1e-6);
anchored wrong it misses by 24-460x that, sometimes under the 1e-4 FP32 tolerance the step-API tests
used - which is why MT-04 did not see it, and why the new test bounds the root leaf at 1e-5. The
two lists differ where a pseudo-legal move would leave a royal capturable, far more often on
Timeline Battleground's crowded 5x5 boards (three timelines from the start) than on Two Timelines.

### The phase form (C26) through the C step API - the gaps met, and the dense pass's last bits (numerics; no threshold, no verdict)

**What it is.** The figures behind section 15.2's C26 row as built (`python/tests/engine/test_phase_form.py`,
`python/tests/net/test_phase_form.py`; BUILD_LOG issues 260-266): how far the phase form's cached
half is from its references, how far the two modes are apart, and why the dense half must be the
dense evaluator itself (issue 262).

**Conditions.** 2026-11-08/09, machine `dev-r` (loaded: another lane's test suite ran beside these),
PyTorch 2.13.0+cpu, Python 3.14.7, NumPy 2.5.2, 2-3 torch threads, the engine library built Release in
`lane/net` (main d6cbdb7 + issue 259's fix); FP32, 2 layers, the v1 configuration under the
block-causal scope, random weights at initialisation scale, model seeds 0-6; the verification ladder
off in every search. Roots: `test_cached_variants._search_root` (random play's first live turn start
with 8 boards or more) on Two Timelines, Timeline Battleground and Very Small - Open, and every engine
variant for the dense half; N\* the root's boards + 1 or + 3 (dense) and the root's boards or - 2
(cached); 12-24 playouts a search, batches of 8 leaves. Games: `netplay`'s small network made
block-causal, N\* = the start's boards + 3.

| Quantity | Largest (or smallest, where marked) | Bound |
|---|---|---|
| Rooted under N\*: the phase form's search against the dense search (digest, counts, every evaluation) | bitwise equal, every arm | bitwise |
| Rooted at or above N\*: a cached leaf against the two-level reference anchored at the root (logits) | 1.31e-6 | 1e-4 |
| ... the root leaf against the dense pass (issue 259's anchor) | 9.54e-7 | 1e-4; 1e-5 (`ROOT_DENSE_BOUND`) from 2026-11-23 |
| ... a search's non-root leaves against the dense pass (the smallest search maximum) | 1.33e-1 (smallest) | above 1e-3 |
| ... every non-root leaf against the dense pass (the smallest over every leaf; 2026-11-23) | 5.35e-2 (smallest) | above 1e-3, per leaf |
| A carried graph search across N\*: a carried node reached after the crossing by a cached path - its WDL against the dense pass's, and against the cached path's evaluation of the same state (2026-11-23; BUILD_LOG issue 264) | 1e-6 (largest); 1.3e-3 to 5.0e-3 | - |
| The dense pass: a position alone against in a batch of four (logits; 3 seeds x 4 synthetic positions of 9-17 boards) | 1.13e-6 | - |
| The dense pass: grad against no-grad, alone | 1.37e-6 | - |
| The dense pass: the same batch twice | 0 (bitwise) | - |

*Added 2026-11-23 (C26/C27's review round; the same machine, loaded - other lanes' tests running -
PyTorch 2.13.0+cpu, Python 3.14.7, NumPy 2.5.2, 3 torch threads, the engine built in `lane/net` on
main 4e9267e).* The per-leaf row is the smallest gap met by R6b's 18 searches and R6c's, now asserted
per leaf (the review's own probe of the 18: 5.36e-2, on Very Small - Open, seed 1). The graph row:
the ply split with `graph=1`, N\* the root's boards + 1, 256 playouts before the crossing and 128 after
it, model and search seeds (6, 3) and (7, 4) on Timeline Battleground, Very Small - Open and Two
Timelines; 3 of the 6 carried searches met such a node (nine links in all), the WDLs compared as
probabilities (the engine's `network_wdl` and the softmax of each evaluation's logits).

*Read.* Under N\* the phase form is the dense search itself, so its figure is equality, not a gap.
At or above N\* the cached leaves sit on their references at the summation-order level, as MT-04's
do, while they differ from the dense pass by a tenth or more - the modes are far apart, so a search
that mixed them per node would be a different search (the per-node-switch mutant always is). The
dense pass is not a function of the position alone in its last bits (about 1e-6 here, with the batch
or the grad mode), which is why the phase form's dense mode is the dense evaluator itself, batch for
batch, and why §15.2's row is proposed reworded (edit 380).

### The cached path under A5 (decision 7) - the gaps met (numerics; no threshold, no verdict)

**What it is.** Section 15.2's cached rows run for register A5 once the 2026-11-15 review's decision
7 was built (BUILD_LOG issues 252 and 276; `python/tests/net/test_cache_a5.py`): how far the cached
path's evaluations are from their references with A5's destination tokens on superseded boards live.

**Conditions.** 2026-11-27, machine `dev-r` (loaded: the engine lane's and the main loop's jobs ran
beside it), PyTorch 2.13.0+cpu, Python 3.14.7, NumPy 2.5.2, 3 torch threads, `lane/net` on main
5e77028 with decision 7's change; 2 layers, the v1 configuration under the block-causal scope with
`piece_centric=True`, random weights at initialisation scale. Measured by a probe beside the tests,
not by the tests' own asserts:
- FP32: `test_cache.py`'s seven geometries under both C20 readings - 14 roots, model seeds 200-213,
  a random path of 6 moves each (98 nodes, 2,011 transient tokens met);
- the C3 row: the same 14 roots with the root refresh, each root's cached evaluation against the
  dense pass;
- BF16 and FP8: model seeds 0-2, `test_cache.py`'s BF16 trees (6 nodes each). The BF16 test asserted
  seeds 0-1 until 2026-11-28, and seeds 0-2 since.

*(2026-11-28, decision 7's review, SN-6: the FP8/BF16 trunk max and value were measured and added; the
margins below replace "hold with their margins"; the figures are the same probe's, re-run.)*

| Quantity | A5, largest | Bound asserted (`test_cache.py`'s) | A5's margin | A1's gap and margin (`test_cache.py`'s docstring) |
|---|---|---|---|---|
| FP32 cached against the block-causal full pass: trunk | 1.67e-6 | 2e-5 | 12x | under 2.4e-6 |
| ... value and policy logits | 1.43e-6 | 1e-4 | 70x | - |
| FP32 root refresh at the root against the dense pass: trunk | 9.54e-7 | 2e-5 | 21x | - |
| BF16 cached against the BF16 full pass: trunk RMS | 1.59e-3 | 6e-3 | 3.8x | 1.9e-3, 3.2x |
| ... trunk max | 1.20e-2 | 4e-2 | 3.3x | 1.3e-2, 3.1x |
| ... value / policy logits | 7.57e-4 / 6.46e-3 | 5e-3 / 2.5e-2 | 6.6x / 3.9x | 1.2e-3 / 8.4e-3, 4.2x / 3.0x |
| FP8 frozen K/V, FP32 trunk, cached against the FP8 network's forward: trunk RMS | 2.83e-5 | 4e-4 | 14x | 7.8e-5, 5.1x |
| ... value | 1.37e-5 | 5e-5 | 3.6x | 4.4e-6, 11x |
| ... policy | 6.20e-5 | 5e-3 | 81x | 1.0e-4, 50x |
| FP8 frozen K/V, BF16 trunk: trunk RMS / max | 2.48e-3 / 3.72e-2 | 8e-3 / 1e-1 | 3.2x / 2.7x | 4.3e-3 / 3.3e-2, 1.9x / 3.0x |
| ... value / policy | 1.20e-3 / 8.37e-3 | 6e-3 / 4e-2 | 5.0x / 4.8x | 1.7e-3 / 1.6e-2, 3.5x / 2.5x |

*Read.* A5's gaps sit where A1's do, and every bound measured for A1 holds for A5 by 2.7x or more.
One margin is narrower than A1's: the FP8 network's value logit under the FP32 trunk, whose largest
A5 gap (1.37e-5) is 3.1x A1's (4.4e-6), a margin of 3.6x against A1's 11x. That is the size of one FP8
rounding flip that an FP32 summation-order difference tips over (`test_cache.py`'s
`test_fp8_cache_equals_the_fp8_network_at_fp32_resolution`); A5's live pass sums over other key sets
than A1's, so its flips fall elsewhere. The bound is not loosened. The default path is unchanged bit for bit
(BUILD_LOG session log, 2026-11-27; `python -m t5d.net.default_path`).

### Refresh and supersession inside a turn (C27) - the deferral's gaps (numerics; no threshold, no verdict)

**What it is.** The figures behind section 15.2's C27 row as built (`python/tests/net/test_turn_refresh.py`,
`python/tests/engine/test_cached_tree_form.py`; BUILD_LOG issues 267-271): how far the deferred
recompute's evaluations are from their from-scratch references. Between the two arms - recompute
at the next evaluated node and recompute at supersession - nothing is measured: they are bit for
bit equal in every case (the entries are content-addressed and computed by one function).

**Conditions.** 2026-11-09, machine `dev-r` (loaded: another lane's test suite ran beside these),
PyTorch 2.13.0+cpu, Python 3.14.7, NumPy 2.5.2, 4 torch threads, the engine library built Release in
`lane/net`; FP32, 2 layers, the v1 configuration under the block-causal scope, random weights at
initialisation scale. Fixture: `test_cache.py`'s seven geometries, seeds 0-2, both C20 readings, the
lineage and two-parent readings (and, for evaluations alone, the creation reading and current frozen
features), causal, root-refresh and node-refresh epochs, paths of 9-15 moves with inner segments of
1-4 nodes. Engine: every variant's random play to 12 boards (seeds 0 and 1), turn starts evaluated
and, by marking, the present gate's state or half the mid-action states; the step API's tree form at
two random-play roots (natural gates and none; 40 playouts; against the every-gate search's 160),
causal, root refresh and C26's cached half; the flat form, 24 playouts.

| Quantity | Largest | Bound |
|---|---|---|
| Fixture: an evaluated node against its reference, trunk (max abs) | 1.91e-6 | 2e-5 |
| ... heads (value, WDL, policy) | 1.55e-6 | 1e-4 |
| Real actions: an evaluated state against its reference, trunk | 1.79e-6 | 2e-5 |
| ... heads | 1.43e-6 | 1e-4 |
| The step API's tree and flat forms: a leaf against its reference (move, WDL, categorical logits) | 1.43e-6 | 1e-4 |
| Deferred against eager, every evaluation and every shared entry | 0 (bitwise) | bitwise |

*Read.* The deferred recompute costs nothing in accuracy: every gap is the cached path's own
summation-order level (MT-04's figures), and the two arms never differ in a bit. What the deferral
changes is when a board is computed - at the next evaluated node, once per board, perspective and
epoch - and how many: a board superseded inside a turn and never read by an evaluated node (an
eager entry the turn's next evaluated nodes do not read, as after the present moves under the
current frozen features) is never computed.

### Row 11's FP8 cache at FP32 resolution at the v1's 16 layers - rounding flips compounding with depth, attributed (numerics; no threshold, no verdict)

**What it is.** `python/tests/net/test_cache.py`'s `test_fp8_cache_equals_the_fp8_network_at_fp32_resolution`
bounds the FP8 cache against the FP8 network's own forward under the FP32 trunk - section 15.2's "under
FP8, paths that differ only in summation order agree up to rare rounding flips ... so their tolerance
is measured on the flips" - at trunk RMS 4e-4, value 5e-5, policy 5e-3, measured at 2 layers. The
re-run of rows 1 and 11 (the entry below) found it over those bounds at the v1's 16 layers. This entry
is the attribution the main loop asked for (BUILD_LOG issue 278): whether the gap is flips alone or a
defect in where the cached path quantises, the flip gap measured at 16 layers, and mutant M14 measured
beside it.

**Conditions.** 2026-11-28, machine `dev-r` (this run's probes, two at a time), PyTorch 2.13.0+cpu,
Python 3.14.7, NumPy 2.5.2, `lane/net` at main 453ba11 (the network code of d470681); FP32 trunk, FP8
frozen K/V, `NetConfig()`'s v1 under the block-causal scope, 16 layers (and 2 for the comparison);
random weights at initialisation scale. Cases: `test_cache.py`'s seven geometries, model seed = tree
seed, a random 5-move path from each root (`_trees`), every node's cached evaluation against the
forward, under `torch.no_grad()`. 12 torch threads, except the 2-layer sweeps (8, run beside the 16-layer ones): the flips
depend on the summation order and so on the thread count - the same 2-layer case (small_board, seed 0)
gave a policy gap of at most 1.9e-4 at 12 threads and 2.5e-3 at 8.

**1. The attribution** (16 layers, small_board, seed 0 - the worst case of the re-run - its 6 nodes).
At every quantisation point of the cached path - a frozen board's stored K/V, which the board itself
reads in `compute_board`, later boards read, and the live set reads; nothing else is quantised - the
two paths' FP8 codes, per layer, per frozen board, K and V, with their FP32 inputs (the forward's taken
by a `kv_hook` before its rounding, the cache's kept beside each entry):
- Both paths round exactly their own inputs: the forward's rounded values equal its codes times its
  scale at every element (0 mismatches of 17.5 million), it rounds no live token (0), and the cache's
  self-read equals its stored codes, which equal the FP8 rounding of its own inputs under the scale of
  its own inputs (0).
- The scales come from the same slice on both paths - a board's own tokens, one layer, K or V, one
  head: at layer 0 they differ by at most 1.7e-7 (relative; FP32 noise). Where a different element
  attains the absmax (6 slices at layer 0), the absmax itself agrees to that 1.7e-7.
- 64,502 codes differ over the 16 layers. Every one is explained by its inputs: its two FP32 inputs lie
  on either side of the midpoint between the two codes (0 exceptions).
- The flips begin from FP32 noise and compound:

| Layer | Codes that differ (of 1,092,096) | ... from inputs within 8 FP32 ulps of the board's absmax | Largest input difference (relative to the board's absmax) | Control: the same, FP8 rounding the identity |
|---|---|---|---|---|
| 0 | 0 | - | 4.0e-7 | 4.0e-7 |
| 1 | 4 (largest input difference behind them 2.7e-7, 0.02 of an FP8 step) | 4 | 4.9e-7 | 5.0e-7 |
| 2 | 26 | 20 | 1.1e-5 | 5.4e-7 |
| 5 | 272 (1 by two steps) | 85 | 2.6e-4 | 5.5e-7 |
| 8 | 2,447 (230 by two or more) | 30 | 1.2e-3 | 6.6e-7 |
| 12 | 8,852 (978) | 27 | 2.4e-3 | 6.9e-7 |
| 15 | 13,038 = 1.2% (1,435) | 20 | 3.5e-3 | 7.5e-7 |

  Every layer from 1 on has 4-85 flips tipped by FP32 noise; the rest are tipped by inputs that earlier
  flips moved, and those inputs move further layer by layer, until some lie more than an FP8 step apart
  (codes two or three steps apart, inputs up to 2.4 steps apart).
- **The control** - FP8 rounding made the identity on both paths, everything else equal: the inputs
  stay within 4.0-7.8e-7 of each other at every layer, and the outputs at trunk RMS 2.6e-7 / max 2.0e-6,
  value 6.0e-7, policy 1.7e-6 - row 1's FP32 level (3.3e-6 at 16 layers). The 16-layer gap is FP8's.

So the gap is the flips alone; no defect in where the cached path quantises was found.

**2. The flip gap and M14 at 2 and 16 layers** (7 geometries x model seeds 0-11, 6 nodes each - 504
nodes per depth). M14 is the mutant of the test's docstring: a frozen board reads its own K/V
unquantised (`compute_board`'s self-read), its stored entry still FP8.

| Quantity, the largest over the 504 nodes | 2 layers | 16 layers | The 2-layer test's bound |
|---|---|---|---|
| Flips: trunk RMS | 1.43e-4 (registers_4_0, seed 7) | 9.67e-4 (small_board, seed 6) | 4e-4 |
| Flips: value | 3.33e-5 (small_board, seed 1) | 3.26e-4 (small_board, seed 6) | 5e-5 |
| Flips: policy | 2.47e-3 (small_board, seed 0) | 8.43e-3 (registers_4_0, seed 11) | 5e-3 |
| Flips: trunk max | 1.28e-2 | 1.91e-2 | - |
| Flips: each tree's largest trunk RMS - median / 90th percentile | 8.3e-6 / 5.8e-5 | 5.3e-4 / 7.7e-4 | - |
| M14: trunk RMS, smallest node / largest | 5.11e-3 / 8.01e-3 | 5.14e-3 / 7.56e-3 | - |
| M14's smallest over the flips' largest (trunk RMS) | 36x | 5.3x | - |

*Read.* At 2 layers the test's bounds hold on every geometry over 12 seeds - trunk RMS 2.8x, value
1.5x, policy 2.0x - and M14 at its smallest is 13x the trunk bound. At 16 layers the flips' gap is 7x
larger (trunk RMS) while M14's is the same as at 2 layers, so the two are only 5.3x apart: a trunk bound
5x above the flips' largest, as the test's docstring sets its own (about 5e-3), would sit at M14's
smallest node (5.1e-3, 1.0x) and under its largest by 1.5x - it would not catch M14. No 16-layer bound is
set on the flips (the main loop's instruction; part 3 asserts the rounding exactly instead); the 2-layer
test is unchanged. The flips are not rare at 16
layers (1.2% of the last layer's codes), which section 15.2's protocol assumes; BUILD_LOG issue 278
carries the question.

**3. The exact test** (2026-11-28, the main loop's decision on issue 278: (a) the 2-layer bound kept,
and (c) the rounding asserted exactly at depth). `test_fp8_cache_rounds_exactly_as_the_fp8_network_at_full_depth`
(slow; 16 layers, FP32 trunk, FP8 frozen K/V; small_board and deep travel, seed 0, 6 nodes each)
asserts the attribution's properties at every layer, frozen board, K and V and node - each path's
scale its own slice's, each path's rounding exact, the cache's self-read its stored codes times its
scale, every differing code straddled - and bounds by FP32 noise (4e-6, relative) what is not exact:
the two paths' scales and inputs up to the first layer with a flip, the inputs behind each first flip,
and, with FP8 rounding the identity, the inputs at every layer. Measured on three geometries (the
test's two and many timelines, 12 threads), the bounded figures and their margins:

| Figure (relative) | small_board | deep travel | many timelines | Margin under 4e-6 |
|---|---|---|---|---|
| The paths' scales, before the first flip | 5.1e-7 | 3.9e-7 | 4.3e-7 | 7.9x |
| The paths' inputs, before the first flip | 4.8e-7 | 4.7e-7 | 4.0e-7 | 8.3x |
| The inputs behind a first flip | 6.8e-8 | 1.5e-7 | 9.1e-8 | 27x |
| The first layer with a flip (every node) | 1 | 1 | 1 | - |
| Codes that differ, the tree's 6 nodes | 43,693 | 134,156 | 92,846 | - |

The control's inputs measured in part 1: at most 7.8e-7 at any layer (5.1x). The flip count follows
the FP32 summation order, which PyTorch's grad mode changes: the same case gives 64,502 differing codes
under `torch.no_grad()` (part 1's probes, layer by layer the same) and 43,693 in grad mode (the test),
each reproducible; the test asserts no count, only that some code differs. Run time at 12 threads: 7.4 s
(small_board) and 17.3 s (deep travel), the control included. Two mutants re-made and reverted:
M14 (`compute_board`'s self-read unquantised) fails assertion 3 at node 0, layer 0, K, board 1 ("the
cache's self-read is not its stored codes"); the cache's stored scale taken over every head
(`kvformat.quantise_fp8`) fails assertion 1 at the same place ("the cache's scale is not its own
slice's").

### Section 15.2's network rows on the finished v1 - rows 1 and 11 re-measured (numerics; no threshold, no verdict)

**What it is.** Step 2's "section 15.2 re-run on the finished v1 network" for rows 1 (cached = dense,
C2, C17) and 11 (cache precision = live precision, G1, C15): the largest gaps between the cached path
(`python/t5d/net/cache.py`) and its references, at the tests' 2 layers and at the v1's 16, on every
geometry of `python/tests/net/test_cache.py`, against the bounds the tests assert. The bounds were
last measured on 2026-10-17, before B15's table, F6's Mish in the heads and M20's defaults changed the
network. No threshold applies; the stop rule of the main loop's brief (a result within 10% of its
bound, or over it) applied to every row, and one row met it - the FP8 cache at FP32 resolution at 16
layers, the entry above. The tests that assert these bounds were extended with this run: BF16 and FP8
on every geometry, the every-switch sets on three seeds (BUILD_LOG, the audit's rows 1 and 11).

**Conditions.** 2026-11-28, machine `dev-r` with no other job of the build running (this run's own
probes ran two at a time), PyTorch 2.13.0+cpu, Python 3.14.7, NumPy 2.5.2, 12 torch threads, `lane/net`
at main d470681; random weights at initialisation scale. The network: `NetConfig()`'s v1 with
`attn_scope="block_causal"` - 16 layers, or 2 for the tests' depth - in FP32 or BF16 as each row says,
FP8 frozen K/V (`frozen_kv_format="fp8"`) where it says so. Measured by a probe beside the tests, on
the tests' own constructions:
- FP32 (row 1): 7 geometries x the lineage and creation readings x both C20 readings, model seeds
  1000 x seed + case and roots of their own; seeds 0-2 at 2 layers (588 nodes), seed 0 at 16 layers
  (196 nodes); random 6-move paths; cached against `reference_output` (from the root's history under
  the past-and-epoch-history reading); at each root, the root refresh against the dense pass.
- The every-switch sets (row 1): `test_cached_equals_one_full_pass_with_every_switch`'s 11 flag sets,
  standard geometry, model seeds 0-4, 2 layers (385 nodes).
- BF16 (rows 1 and 11): 7 geometries, model seeds 0-2 at 2 layers and 0-1 at 16, a random 5-move path
  from each root (`_trees`); cached against the BF16 full pass ("live") and against the FP32 one; the
  root refresh against the dense BF16 pass.
- FP8 (row 11): the same trees, seeds 0-2 at 2 layers and 0 at 16; cached against the FP8 network's
  own forward, under the BF16 trunk (the v1's configuration) and the FP32 trunk; the FP8 network
  against the "match" one (BF16).

| Quantity | 2 layers | 16 layers | Bound asserted | Margin (2 / 16) |
|---|---|---|---|---|
| FP32 cached against the full pass: trunk (max abs) | 2.38e-6 | 3.34e-6 | 2e-5 | 8.4x / 6.0x |
| ... heads (value, WDL, policy, per-board, submit) | 1.67e-6 | 2.03e-6 | 1e-4 | 60x / 49x |
| FP32 root refresh against the dense pass: trunk / heads | 1.43e-6 / 1.19e-6 | 2.15e-6 / 1.31e-6 | 2e-5 / 1e-4 | 14x / 9.3x (trunk) |
| The every-switch sets, seeds 0-4: trunk / heads | 1.91e-6 / 1.79e-6 | - | 2e-5 / 1e-4 | 10x / 56x |
| BF16 cached against live: trunk RMS / max | 1.64e-3 / 1.42e-2 | 1.96e-3 / 1.20e-2 | 6e-3 / 4e-2 | 3.7x, 2.8x / 3.1x, 3.3x |
| ... value / policy | 6.95e-4 / 7.85e-3 | 1.44e-3 / 9.73e-3 | 5e-3 / 2.5e-2 | 7.2x, 3.2x / 3.5x, 2.6x |
| BF16 cached against FP32: trunk RMS / max | 2.33e-3 / 1.44e-2 | 2.33e-3 / 1.37e-2 | 1e-2 / 6e-2 | 4.3x, 4.2x / 4.3x, 4.4x |
| ... value / policy | 3.95e-3 / 1.24e-2 | 3.81e-3 / 1.12e-2 | 2e-2 / 3e-2 | 5.1x, 2.4x / 5.3x, 2.7x |
| ... its RMS over `1.3 * live RMS + 3e-4` (live against FP32: 2.34e-3 at both depths) | 0.702 | 0.701 | 1 | 1.4x / 1.4x |
| BF16 refreshed root against the dense pass: trunk RMS / max | 6.70e-4 / 9.71e-3 | 1.29e-3 / 1.08e-2 | 3e-3 / 4e-2 | 4.5x, 4.1x / 2.3x, 3.7x |
| ... value / policy | 4.06e-4 / 4.99e-3 | 1.11e-3 / 7.07e-3 | 5e-3 / 2.5e-2 | 12x, 5.0x / 4.5x, 3.5x |
| FP8 frozen K/V, BF16 trunk, cached against the FP8 forward: trunk RMS / max | 3.00e-3 / 3.81e-2 | 3.94e-3 / 2.64e-2 | 8e-3 / 1e-1 | 2.7x, 2.6x / 2.0x, 3.8x |
| ... value / policy | 1.01e-3 / 1.88e-2 | 2.00e-3 / 1.90e-2 | 6e-3 / 4e-2 | 5.9x, 2.1x / 3.0x, 2.1x |
| FP8 frozen K/V, FP32 trunk, cached against the FP8 forward: trunk RMS | 8.01e-5 | 6.12e-4 | 4e-4 (2 layers) | 5.0x / **over** |
| ... value / policy | 3.58e-5 / 1.89e-4 | 1.97e-4 / 3.53e-3 | 5e-5 / 5e-3 (2 layers) | 1.4x, 26x / **over**, 1.4x |
| FP8 network against "match" (BF16): trunk RMS / max | 9.06e-3 / 5.59e-2 | 7.79e-3 / 5.40e-2 | 3e-2 / - | 3.3x / 3.9x |
| ... value / policy | 1.28e-2 / 3.53e-2 | 9.67e-3 / 3.18e-2 | 4e-2 / 1e-1 | 3.1x, 2.8x / 4.1x, 3.1x |
| ... each tree's largest RMS, the smallest of them (FP8 is in use) | 6.77e-3 | 6.97e-3 | above 1e-3 | 6.8x / 7.0x |

*Read.* Every bound but one holds on the finished v1, on every geometry and at both depths, by 1.4x
or more. The BF16 and FP8 figures are at most 28% above the 2026-10-17 ones, which were taken on the
standard geometry alone (the largest rise: the FP8 network's value logit against "match", 1.28e-2
against 1.0e-2); the FP32 trunk's at 16 layers (3.3e-6) is a little above the 2.4e-6 the tests'
docstring recorded. The BF16 cache is as accurate as the live BF16 pass at both depths (the cached RMS against
FP32 equals the live one's, 2.33e-3 against 2.34e-3). FP8 under the BF16 trunk - the v1's own
configuration - holds its bounds at 16 layers by 2.0x. The exception is the FP8 cache at FP32
resolution, whose bounds were measured at 2 layers: at 16 layers it is over them, and the entry above
attributes it. At 2 layers its value logit's margin, 1.4x, is the narrowest of the table (one FP8
rounding flip, as in the A5 entry above).

### F1 on engine positions under the cached configuration - re-measured on the finished v1 (numerics; no threshold, no verdict)

**What it is.** Rows 1 and 11 on real engine positions at F1's full size: F1 as shipped (`NetConfig()`,
BF16 trunk, 16 layers) under the block-causal scope, its cached evaluations against the BF16 full pass
(`reference_output`), on the positions `python/tests/engine/test_network.py`'s
`test_f1_cached_on_a_real_position` and `test_cached_variants.py`'s `test_f1_bf16_cached_on_three_variants`
assert on; and the same network with FP8 frozen K/V against the FP8 network's own forward, which no
test asserts on engine positions (compared here with `test_cache.py`'s FP8 bounds). Part of step 2's
re-run (the entry above); the figures before it were taken before B15's table, F6's Mish in the heads
and M20's defaults changed the network.

**Conditions.** 2026-11-28, machine `dev-r` (another probe of this run beside it), PyTorch 2.13.0+cpu,
Python 3.14.7, NumPy 2.5.2, 12 torch threads, the engine library built Release in `lane/net` at main
453ba11 (the network code of d470681); random weights at initialisation scale, model seeds 0-2.
Positions: the "cross_timeline" record's `positions[2:]` (3 states, its frozen boards including a
duplicated pair; the epoch opened at the first), and seed 3's random play on Standard - Turn Zero,
Standard - Reversed Royalty and Misc - Timeline Strategos, every second state (the epoch opened a third
of the way in) - 198 evaluations over both formats and the three seeds.

| Quantity | Largest | Where | Bound | Margin |
|---|---|---|---|---|
| F1 ("match"), cached against the BF16 full pass: trunk RMS / max | 1.99e-3 / 1.24e-2 | Timeline Strategos, seed 1 | 8e-3 / 5e-2 | 4.0x / 4.0x |
| ... policy / value logits | 9.53e-3 / 1.41e-3 | Reversed Royalty / Turn Zero, seed 1 | 3e-2 / 5e-3 | 3.1x / 3.6x |
| F1 with FP8 frozen K/V, cached against the FP8 network's forward: trunk RMS / max | 4.47e-3 / 3.22e-2 | Timeline Strategos / Reversed Royalty, seed 2 | 8e-3 / 1e-1 (`test_cache.py`'s) | 1.8x / 3.1x |
| ... policy / value logits | 1.96e-2 / 2.82e-3 | Turn Zero, seed 1 / Reversed Royalty, seed 0 | 4e-2 / 6e-3 | 2.0x / 2.1x |

*Read.* F1's own gaps on engine positions are the synthetic trees' at 16 layers (BF16 cached against
live, RMS 2.0e-3): the F1 tests' bounds hold by 3.1x or more. Against MT-04's figures of 2026-10-23
(one seed: RMS 1.97e-3, max 1.15e-2, policy 7.18e-3, value 1.32e-3) the policy logit is a third larger
over three seeds. With FP8 frozen K/V the engine positions' trunk RMS (4.47e-3) is above the synthetic
trees' at 16 layers (3.94e-3) and 1.8x inside `test_cache.py`'s FP8 bound; no test asserts FP8 on
engine positions.
