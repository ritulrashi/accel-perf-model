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
