# Results

Every number here comes from a generated file under `results/`, and
`scripts/check_results.sh` verifies each one (rule 1). Each results file
begins with its own provenance block: commit, clean tree, machine and tool
versions. This is a model of no real chip. The numbers describe this model
with these modelling parameters.

Phases 1 and 2 have no hypotheses beyond "the hand derivations hold". Their
exit criteria are in [results/phase1](results/phase1) and
[results/phase2](results/phase2) (all hand-derived values matched; 200 of 200
random GEMMs bit-exact).

## Phase 3: accelerator grid, stall stack, roofline

Measured at commit 0d8ce61 by `scripts/phase3_results.sh`. Sources:
[exit_criterion.txt](results/phase3/exit_criterion.txt),
[sweep.csv](results/phase3/sweep.csv),
[functional_accel.csv](results/phase3/functional_accel.csv),
[mutation_check.txt](results/phase3/mutation_check.txt).

### Directed tests (hand-derived in the design note before running)

| Test | Hand-derived | Measured |
|---|---|---|
| P3-T1, one PE at the SRAM node: T_end | 192 | 192 |
| P3-T1 stall stack (compute / dram / network / dma / idle) | 86 / 52 / 7 / 27 / 20 | 86 / 52 / 7 / 27 / 20 |
| P3-T2, PE one hop away, second PE idle: T_end | 272 | 272 |
| P3-T2 stall stack of PE 0 | 86 / 52 / 65 / 49 / 20 | 86 / 52 / 65 / 49 / 20 |
| P3-T2 PE 1 idle | 272 | 272 |
| P3-T3, 2×2 grid: DRAM bytes (staging formula) | 24576 | 24576 |

Both build presets (Release and ASan/UBSan) pass all 79 tests with no
mismatches. All 100 random grid GEMMs agree bit-exactly between the model,
NumPy and the C++ reference, including 11 all −128 cases.

### Exit criterion: stall-stack assertions over the sweep

All 115 configurations ran. In every run, the model's own assertions held:
- per-PE stacks summed to T_end and the chip stack to P·T_end;
- compute cycles equalled the tile-latency formula;
- DRAM bytes equalled the staging formula;
- the classifier never met an unexplained stall.

The sweep added three more checks, which also held in every run:
- DRAM bytes = the Python formula count;
- achieved ≤ attainable;
- the simulator's busiest link was busy at least `B_max / L` cycles.

In total, 2778141 PE-cycles were classified, with zero failures.

![roofline](results/phase3/roofline.png)
![stall stacks](results/phase3/stall_stacks.png)

### Hypotheses (written in the design note §8 before measuring)

**H1: every run satisfies the assertions, and achieved ≤ attainable.**
**Confirmed.** All 115 runs, zero failures (above).

**H2: the dominant stall matches the predicted roofline region.**
**Partly refuted.**

| Predicted region | Configs | Expected stall wins | Others |
|---|---|---|---|
| compute | 37 | compute largest in 34 | idle 2, dram 1 |
| DRAM | 52 | dram the largest non-compute stall in 28 | idle 14, dma 10 |
| NoC | 26 | network the largest non-compute stall in 12 | idle 8, dram 3, dma 3 |

The formula region is right about which ceiling is lowest. It is often
wrong about what the PEs actually wait on.
- **idle:** synchronization on the staging ring, partial waves, and the
  write-back epilogue. These matter most in short-K configurations.
- **dma:** the on-chip fetch latency (SRAM access, mesh responses arriving
  at the DMA, LMEM writes). It is not a bandwidth ceiling, so a roofline
  cannot predict it.

Example, config 4 (16×16 arrays, 2×2 grid, W = 64, L = 64):
- predicted DRAM-bound, at 0.5290 of its DRAM ceiling;
- its largest non-compute stall is `dma`, not `dram`.

**H3: compute-bound §4.1 configs reach ≥ 75 % of peak at K = 1024.**
**Refuted.** Both compute-bound §4.1 configurations fall short:

| Config | Achieved / peak | Largest non-compute stall |
|---|---|---|
| 0 (4×4 arrays, 2×2 grid, W = 64, L = 64) | 0.7320 | dram |
| 2 (8×8 arrays, same system) | 0.7019 | dma |

I underestimated the staging prologue and the per-chunk fetch latency. The
fill/drain term alone (F ≤ 46 cycles against 1024 feed cycles per tile) was
not the main loss.

**H4: DRAM-bound configs reach ≥ 50 % of their DRAM ceiling.**
**Confirmed** for both §4.1 DRAM-bound configurations:
- config 1 (8×8, W = 16): 0.8251;
- config 4 (16×16, W = 64): 0.5290, only just over the line.

**H5: moving the SRAM from the corner to the centre of a 3×3 grid raises
throughput in NoC-bound configs.**
**Confirmed** for all three array sizes (achieved ops/cycle):

| Array | Corner | Centre |
|---|---|---|
| 4×4 | 93.6377 | 138.3265 |
| 8×8 | 188.6230 | 369.0725 |
| 16×16 | 377.5704 | 745.4626 |

The centre has four egress links instead of two, so the formula NoC ceiling
also rises: 384.0000 → 768.0000 ops/cycle for 16×16. With the 4×4 array, the centre placement
lifts the configuration far enough that the NoC is no longer its largest
stall. Compute is (config 6).

### What looked too good, and why it holds

NoC-bound runs reach up to 0.9833 of their NoC ceiling (config 9). Under
the project rules that is a suspect until checked. The sweep's link
cross-check explains it:
- the simulator's busiest link was busy 49536 cycles out of T_end = 49989;
- that is the formula's data bytes plus the one-cycle write acknowledgements
  on the same link.

So the link really is saturated. The mutation check also shows the stall
checks can fail: all 8 mutants were caught, including a classifier that
charges the fill/drain tail as idle (caught only by the compute
cross-check) and a scheduler that stages each block twice (caught by the
DRAM-bytes assertion).

Across all runs, efficiency against the attainable ceiling ranges from
0.2246 to 0.9847 (median 0.6165).

## Phase 4: precision modes, vector unit, attention

Measured at commit 84a082a by `scripts/phase4_results.sh`. Sources:
[exit_criterion.txt](results/phase4/exit_criterion.txt),
[study.csv](results/phase4/study.csv),
[functional_gemm.csv](results/phase4/functional_gemm.csv),
[functional_attn.csv](results/phase4/functional_attn.csv),
[mutation_check.txt](results/phase4/mutation_check.txt).
The reference configuration (A4) is a 2×2 grid of 16×16 arrays with
W = 64 B/cycle DRAM. All workloads use head dim 64 and one KV head
(design note §8.4).

### Directed tests (hand-derived in the design note before running)

| Test | Hand-derived | Measured |
|---|---|---|
| P4-T1, FP16 GEMM: T_end | 316 | 316 |
| P4-T1 stall stack (compute / dram / network / dma / idle) | 150 / 92 / 7 / 47 / 20 | 150 / 92 / 7 / 47 / 20 |
| P4-T2, FP32 GEMM: T_end | 564 | 564 |
| P4-T2 stall stack | 278 / 172 / 7 / 87 / 20 | 278 / 172 / 7 / 87 / 20 |
| P4-T3, BF16: same as FP16 | 316, 2816 B | 316, 2816 B |
| P4-T4, masked write-back: T_end, DRAM bytes | 186, 1344 | 186, 1344 |
| P4-V1, VU 2 rows: T_end FP16 / INT8 / FP32 | 84 / 82 / 88 | 84 / 82 / 88 |
| P4-V2, VU two P segments: T_end | 84 | 84 |

Both presets pass all 113 tests. All of the following are bit-exact three
ways (model, NumPy replica, standalone C++ reference):
- 200 random grid GEMMs, 50 per precision, including 25 large-magnitude
  cases;
- 80 random attention programs, 8 per mode × precision combination;
- all 54 study programs.

All 11 mutants were caught. For example, a fused multiply-add in the FP32
accumulator is caught only by the FP32 functional cases and a hand case;
the other precisions are immune because their products are exact.

### Hypotheses (written in the design note §10 before measuring)

**H1: hand derivations hold, everything is bit-exact, BF16 timing equals
FP16.** **Confirmed** (above).

**H2: lower precision speeds up prefill sub-linearly. Prediction at
L = 1024: FP16/INT8 in [1.3, 1.8], FP32/FP16 in [1.4, 1.95].**
**Refuted.** It is far more sub-linear than predicted, and for a reason the
prediction missed.

| Prefill L = 1024 | Measured | Bound-based prediction |
|---|---|---|
| FP16 / INT8 time | 1.0965 | 1.5214 |
| FP32 / FP16 time | 1.2550 | 1.6857 |

Precision barely matters because the Q·Kᵀ op dominates and is limited by
DRAM row misses, not by compute or bandwidth.
- **INT8:** Q·Kᵀ takes 1402154 of 1700845 cycles. 68928 of its 131072
  DRAM accesses are row misses, and the DRAM bus is busy only 0.0935 of
  the time.
- **Cause:** the S write-back is row-major with a row stride of
  4·L = 4096 B, exactly banks × row size. Every row of an output tile
  therefore lands in the same DRAM bank, in a different row, and the
  in-order FCFS controller (Phase 1, no reordering, no write buffer)
  serializes the misses.
- **Contrast:** P·V writes only a small O and stays near its bounds. At
  L = 256 the stride is a single DRAM row and the effect is much weaker.
- **What it shows:** the model's simplest DRAM component decides the
  headline number. The first Phase 5 exploration should be the
  controller (FR-FCFS, write buffering, bank hashing) or a tile-major S
  layout.

**H3: unfused softmax is ≥ 15 % of prefill at L = 1024 in every precision,
largest at INT8.** **Refuted.** The shares are 0.0997 (INT8), 0.1406
(FP16/BF16) and 0.1967 (FP32), and the share is largest at FP32. The
inflated Q·Kᵀ op shrinks every other share. FP32 is above the line
because its S read and P write are the largest.

**H4: decode Q·Kᵀ at INT8 is array-bound, not memory-bound.**
**Confirmed.** At T = 4096 the op's largest stall category is `compute`:
28160 PE-cycles, against 737 `dram` and 4311 `dma`. The formula roofline,
which counts only useful (unpadded) ops, predicts this op to be DRAM-bound.
The array instead spends its time on tiles that use 1 of 16 rows and pay
the fill/drain on every 64-step tile.
- **Beyond the hypothesis:** FP16 and FP32 also have `compute` as the
  largest category (44544 and 77312 PE-cycles). I had expected FP32 to be
  DRAM-bound.

**H5: grouped-query attention is nearly free on this array. T(g)/T(1) at
T = 4096, FP16: ≤ 1.25 for g ≤ 8, ≤ 1.6 for g = 16.** **Refuted.**

| g | T(g)/T(1) FP16 | per query head | Q·Kᵀ row misses | Q·Kᵀ bus busy |
|---|---|---|---|---|
| 1 | 1 | 1 | 759 | 0.7463 |
| 4 | 1.7074 | 0.4269 | 1664 | 0.3697 |
| 8 | 2.6095 | 0.3262 | 2704 | 0.2412 |
| 16 | 4.4110 | 0.2757 | 4736 | 0.1562 |

The extra query heads are indeed free in the array (padding absorbs them),
but every extra S row is written with a stride of 4·T bytes (a multiple of
banks × row size) into the same DRAM bank. This is the H2 mechanism again. Grouping still cuts the
cost per query head, to 0.3262 of multi-head at g = 8 (INT8: 0.4052), but
by much less than the ≥ 6× I predicted.

**H6: the logical 1×4 mapping makes decode Q·Kᵀ ≥ 1.6× faster than the
physical 2×2 mapping (INT8, T = 4096).** **Confirmed:** 14251 → 8516
cycles, 1.6734×. FP16 gains 1.6022× and FP32 1.5701×. The whole decode
program gains 1.4003× (INT8).

**H7: numerics, relative error of O vs FP64.** **Confirmed in every part.**

| Claim | Threshold | Measured |
|---|---|---|
| FP32 error | ≤ 1e-5 | at most 1.8681e-06 (decode T = 8192) |
| FP16 error | ≤ 2e-3 | at most 5.3203e-04 (decode T = 8192) |
| BF16 / FP16 error ratio | in [4, 16] | 7.4951 to 8.7821 |
| INT8 per-row P scale | ≤ 0.1 | at most 3.0267e-02 |
| INT8 fixed P scale at T = 4096 | ≥ 0.5 | 1.0000e+00 |
| fixed / per-row at T ≥ 1024 | > 5× | ≥ 26.6677 |

With the fixed scale 1/127 at T ≥ 4096, every probability rounds to zero,
so O is identically zero (relative error exactly 1). Per-row max scaling
costs nothing in this design, because the unnormalized exponentials are
quantized and 1/sum moves into the dequantization. It keeps the error at or
below 3.0267e-02 at every T.

**H8: VU busy fraction at prefill L = 1024 (FP16): ≥ 0.8 for V ≤ 32,
≤ 0.7 for V ≥ 64.** **Partly refuted.**

| V | softmax cycles | busy fraction |
|---|---|---|
| 16 | 262347 | 0.7963 |
| 32 | 262221 | 0.4218 |
| 64 | 262174 | 0.2343 |
| 128 | 262160 | 0.1406 |

The V ≥ 64 half holds. The V ≤ 32 half fails: the softmax takes 262160 to
262347 cycles at every V. It is not bound by the VU or by bus
bandwidth (bus busy 0.3750). It is bound by DRAM latency and row misses
(20480 misses per op) behind the in-order DMA: the P stores are scattered
Kt·e-byte segments with an R·Kt·e stride, interleaved with the S row loads.

![where attention time goes](results/phase4/attention_breakdown.png)
![decode numerics](results/phase4/numerics.png)
![attention roofline](results/phase4/attention_roofline.png)

### What looked too good or too bad, and why it holds

- **Prefill sits up to 6× above its lower bound** (INT8 L = 1024: 1700845
  cycles against a bound of 282688.0). This looked like a simulator bug
  until the per-op DRAM counters were checked. They show the row-miss
  serialization described under H2.
  - These counters were added to the study as instrumentation before the
    official run. The model itself was not changed.
  - Every op of every program also passes the T ≥ lower-bound assertion
    and the DRAM-bytes-equals-formula assertion.
- **INT8-fixed error of exactly 1.0000e+00** is not a failed run: the model,
  the NumPy replica and the C++ reference all produce an all-zero O, bit
  for bit.
