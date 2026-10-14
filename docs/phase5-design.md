# Phase 5 Design Note — RTL Correlation of the Compute Unit, Microarchitecture Sweeps

Status: **approved 2026-10-14 (user: "go"); implementing**

## 0. What Phase 5 adds

Two independent parts.

**A. RTL correlation.** A SystemVerilog output-stationary systolic array
(INT8 → INT32) implements the **Phase 2 §2.2 cycle contract**. A Verilator
testbench drives it with the exact feed schedules the model's PE controllers
produced, and compares three things:
- when each tile drains;
- the values drained;
- **which cell performs which MAC in every cycle**.

**The contract is the referee, not the model.** When the RTL and the model
disagree, the side that departs from the contract text gets fixed and the
fix is logged (§5).

**B. Microarchitecture sweeps.** Phase 4's headline result was a DRAM
pathology: S rows at a stride of banks × row size all land in one bank, and
the in-order controller serializes their row misses. Part B adds one
controller-side remedy, an **XOR bank-address map**, with hand-derived
tests. It then runs two sweeps:
- **S1:** the row map vs the XOR map on the Phase 4 workloads. Does the
  pathology go away, and does the Phase 4 precision prediction then hold?
- **S2:** three **equal-MAC arrangements** (1×1 of 32×32, 2×2 of 16×16,
  4×4 of 8×8) on prefill and decode.

Not in Phase 5: an FR-FCFS controller, write buffering, a tile-major S
layout, FP modes in RTL, and synthesis. They go in LIMITATIONS.
Parallel sweep execution and CI are Phase 6.

## 1. The contract the RTL implements (Phase 2 §2.2, restated for ports)

Cycle t is the interval between clock edges t and t+1. Inputs for cycle t
are applied before edge t+1. A cell's MAC "happens in cycle t" when its
accumulator register takes the new value at edge t+1.

- **Feed:** during cycle `f + k` the edge sees one A column vector
  `a_col[0..R)` and one B row vector `b_row[0..C)`, with `in_valid = 1`.
- **Skew:** row i's A is delayed by i registers, and column j's B by j.
  A moves right one cell per cycle, and B moves down. So
  **`MAC(i, j, k)` happens in cycle `f + k + i + j`**.
- **Tags travel with A:** `first` (k = 0 of the tile's first chunk),
  `last` (the last k of its last chunk), and `k` (the global reduction
  index within the tile, used for tracing only).
- **First MAC of a tile:** a cell whose MAC carries `first` computes
  `acc = a·b`; every other MAC computes `acc = acc + a·b`. That lets a new
  tile start feeding in the very cycle the previous tile is done, with no
  separate clear cycle.
  - **Model mapping:** the model clears its accumulators at feed start. The
    two are equivalent because the contract forbids overlapping tiles
    (Phase 2 §2.3 rule 1).
- **Drain:** when the MAC carrying `last` reaches cell (R−1, C−1), in cycle
  `f_L + Kt + R + C − 3`, the next R cycles drain one row each, **row R−1
  first**. Each drain cycle shows `drain_valid`, `drain_row` and
  `drain_data[0..C)` (INT32) combinationally from the accumulators.
  - The drain covers `[f_L + Kt + R + C − 2, f_L + Kt + 2R + C − 2)`.
  - **Tile done** = the cycle after the last drain row,
    `f_L + Kt + 2R + C − 2`.
- **Bubbles:** cycles with `in_valid = 0` between chunks shift everything
  after them. The accumulators keep their sums.
- **Protocol check (sticky `error` output):** a `first` feed while a tile
  is still draining, or before its drain has started, sets the error. The
  testbench fails the case on any error.

**INT8 only (q = 1).** FP modes (q > 1) are a timing generalization of the
same contract (Phase 4 §1.2), and they are not built in RTL.

## 2. RTL structure (`rtl/`)

| File | Contents |
|---|---|
| `os_array.sv` | top: edge skew registers, R×C cell array, drain counter and row mux, protocol check, trace ports. Parameters `R`, `C`, `KW` (k-tag width) |
| `os_cell.sv` | one cell: A/B/tag pipeline registers, INT32 accumulator, clear-on-first MAC |

- **Style:** synchronous active-high reset, `always_ff` for state,
  `always_comb` for next state, no latches, no initial blocks.
- **Accumulation:** `$signed` 8×8 products summed into 32 bits. Overflow is
  impossible in the correlated cases: K ≤ 131071, as in the model.
- **Trace port (read-only, feeds nothing back):** `mac_valid[R][C]` and
  `mac_k[R][C]`. It shows whether each cell performs a MAC this cycle, and
  with which k.
- **Lint:** `verilator --lint-only -Wall` must be clean with no waivers.

## 3. Schedule export and the testbench (`tb/`, C++ with Verilator)

### 3.1 What the model exports

`apm_gemm` and `apm_accel` gain `schedule=FILE`. For every PE it writes the
schedule its controller actually ran:

```
pe <id> R <R> C <C> Kt <Kt>
step <s> tile <t> chunk <c> feed <f> a <hex R*Kt bytes> b <hex Kt*C bytes>
tile <t> done <cycle> c <hex 4*R*C bytes>        # the model's accumulator values at drain
```

The operand bytes are the ones the model's functional path read from LMEM at
feed start, and the C values are what it wrote at tile done. Nothing is
recomputed for the export.

### 3.2 Two expected traces, one RTL trace

- **Model side:** a pure C++ function, `contract_trace(schedule)` in
  `src/compute/contract_trace.{h,cpp}`, expands the model's feed cycles with
  the model's own contract functions (`mac_cycle`, `drain_start`,
  `tile_done` in `systolic.h`) into a per-cycle trace.
- **RTL side:** the testbench (`tb/os_array_tb.cpp`) drives the RTL from
  the same schedule:
  - in cycle `f + k` it presents column k of the step's A block and row k
    of its B block, with the tags;
  - every other cycle is a bubble.

  It records the same per-cycle trace from the trace ports.
- **Trace format** (one line per cycle with any MAC or drain; the cycle
  number is printed, so idle gaps are visible):

  ```
  <cycle> mac <k_00> <k_01> ... <k_(R-1)(C-1)> drain <row|->      ('.' = no MAC)
  ```

### 3.3 Three levels, each strictly stronger

| Level | RTL | Model | Must match |
|---|---|---|---|
| 1. Tile timing | first drain cycle and done cycle of every tile | `drain_start` / the PE's recorded tile-done cycle | exactly |
| 2. Values | the R drained rows, assembled into a tile | the exported C tile (the model's accumulators) | bit-exact |
| 3. **Every cycle** | the trace above | `contract_trace` of the exported schedule | byte-identical |

Level 3 is the point. Matching totals can hide compensating errors, and a
per-cell, per-cycle k cannot.

**`os_array_tb SCHEDULE [--trace-out FILE]`:**
- builds one Verilated model per (R, C) variant (§4) and dispatches on the
  schedule's header;
- exit status 0 only if all three levels match for every PE in the file;
- on the first mismatch, prints the cycle, the cell or tile, and both values.

### 3.4 Hand-derived traces (the guard against one author misreading the contract twice)

Both are written from the contract text, independently of either
implementation. The RTL **and** `contract_trace` are each compared against
them.

**HT1: R = 2, C = 3, Kt = 2, one chunk at f = 0** (the Phase 2 §5 unit
table). MACs are at cycles 0–4 and the drain covers [5, 7) (row 1, then
row 0). Done 7.

**HT2: R = 2, C = 2, Kt = 2, two chunks at f = 0 and f = 3** (a bubble in
cycle 2). Cells in the order 00 01 10 11:

```
0 mac 0 . . . drain -
1 mac 1 0 0 . drain -
2 mac . 1 1 0 drain -
3 mac 2 . . 1 drain -
4 mac 3 2 2 . drain -
5 mac . 3 3 2 drain -
6 mac . . . 3 drain -
7 mac . . . . drain 1
8 mac . . . . drain 0
```

Last MAC `f_L + Kt + R + C − 3 = 3 + 2 + 2 + 2 − 3 = 6`; drain [7, 9);
done 9.

## 4. Correlation program set

| Source | Schedules (one per PE) | (R, C) variants |
|---|---|---|
| HT1, HT2 | 2 (hand-written schedule files) | (2, 3), (2, 2) |
| Phase 2 directed P2-T1–T5b (`apm_gemm`), with level 1 also checked against the Phase 2 hand values (tile done 138; 330; 486; 418; 138/224; 244) | 6 | (4, 16) |
| The 200 Phase 2 random PE cases (`gemm_check` PE stream) | 200 | R ∈ {1, 2, 4, 8} × C ∈ {16, 32} |
| The 100 Phase 3 random grid cases (`gemm_check` accel stream): every PE with work. Their schedules have irregular bubbles from mesh and staging contention. | one per PE with tiles | R ∈ {1, 2, 4} × C ∈ {4, 16} |
| A4 INT8 attention GEMM ops: prefill L = 256 (Q·Kᵀ, P·V) and decode T = 4096, g = 8 (Q·Kᵀ, P·V), every PE | 16 | (16, 16) |

In total 14 RTL variants, each Verilated once.

## 5. Process when they disagree

1. Take the first mismatching cycle and cell (level 3), or tile (levels
   1–2).
2. Decide from the contract text (§1, Phase 2 §2.2–2.3) which side is
   wrong. Check the hand traces first. If the text is ambiguous, that is a
   contract bug: clarify it, and ask you if the clarification changes any
   published number.
3. Fix that side only. Never adjust the other side to agree.
4. Log the case in `docs/CORRELATION.md`: program, cycle, cell, root cause,
   side at fault, fix, commit. Cross-link it in `docs/BUGS.md`.
5. Rerun the whole set. If the model changed, rerun its Phase 1–4 suites as
   well. Any change to a published Phase 2–4 number would be reported to you
   before anything else.

## 6. DRAM bank-address map option (part B)

**`DramParams::bank_map`**:
- **`row`** (default; Phases 1–4 unchanged): `bank = row_index mod B`.
- **`xor`** (B a power of two):
  `bank = (row_index XOR (row_index / B)) mod B`.

In both maps, the row within the bank stays `row_index / B`, so the map is a
bijection. Its inverse is `row_index mod B = bank XOR ((row_index / B) mod B)`.
A unit test checks the bijection on 4096 row indices. Timing formulas are
unchanged (Phase 1 §3.4); only `map()` differs.

**Hand-derived tests** use the Phase 1 configuration: B = 4, rows 1024 B,
W = 16 (64 B → D = 4), `t_hit = 10`, `t_miss = 30`, and one TrafficGen on
the DRAM.

| Test | Requests | Map | Derivation | Expected responses |
|---|---|---|---|---|
| T5a | read @0 at 0, read @4096 at 1 (T4c's addresses) | xor | row index 4 → bank (4 ⊕ 1) mod 4 = 1, closed: start 1, col 21, data max(31, 34) = 34–38 | **34, 38** (T4c gives 34, 58) |
| T5b | read @0 at 0, read @5120 at 1 | xor | row index 5 → bank (5 ⊕ 1) mod 4 = 0, row 1: same-bank conflict as T4c, start 24, col 44, data 54–58 | **34, 58** (row map: bank 1, 34, 38). Hashing permutes conflicts; it does not remove them. |
| T5c | writes of 64 B @0, 4096, 8192, 12288 at cycles 0–3 (the S-row pattern) | row | all bank 0, rows 0–3: data ends 34; 58 (start 24); 82 (start 48, col 68); 106 (start 72, col 92) | **last response 106** |
| T5d | same as T5c | xor | banks 0, 1, 2, 3 (4m ⊕ m mod 4 = m): data 30–34, then the bus-limited 34–38, 38–42, 42–46 | **last response 46** |

## 7. Sweeps (part B)

All runs use `python/attn.py` with its bit-exact three-way check and FP64
error. They start from the A4 configuration of Phase 4 (§8.4) and change
only what each sweep names.

**S1: bank map** (30 programs). For each map ∈ {row, xor}:
- prefill L ∈ {256, 1024} × {INT8, FP16, BF16, FP32};
- decode T = 4096, g = 1 × the four precisions;
- GQA, T = 4096, g ∈ {4, 8, 16}, FP16.

**Cross-check:** every row-map program must reproduce its Phase 4
`study.csv` cycles exactly. This confirms that Phase 5 changed no Phase 4
timing. The script asserts it.

**S2: equal-MAC arrangements** (12 programs, xor map). There are 1024 MACs
in each arrangement:

| Arrangement | Grid | Array | Mesh | F = 2R + C − 2 |
|---|---|---|---|---|
| big | 1×1 | 32×32 | none (H = 0) | 94 |
| A4 | 2×2 | 16×16 | 2×2 | 46 |
| many | 4×4 | 8×8 | 4×4 | 22 |

Workloads:
- prefill L = 1024, FP16 and INT8;
- decode T = 4096, FP16 with g = 1, and INT8 with g = 8.

The SRAM stays at the corner (0, 0) in every grid, for comparability. Phase 3
H5 showed that a centre placement helps NoC-bound grids; that is noted, not
swept. Decode uses the logical 1×P mapping, as in Phase 4.

**Why equal-MAC.** With R·C·P fixed, `C·py_map = √(MACs)` is the same for
all three. So the DRAM operand traffic per GEMM is identical
(`|A|·tiles_n/py + |B|·tiles_m/px`), and the arrangements differ only in:
- fill/drain per tile (F);
- row utilization in decode (g/R);
- mesh load (B_max).

## 8. Hand-derived tests, summary

- **HT1, HT2:** the RTL and `contract_trace` both reproduce the traces in
  §3.4.
- **P2-T1–T5b schedules:** the RTL's tile-done cycles equal the Phase 2
  hand values.
- **T5a–T5d:** DRAM responses as in §6.
- **Unit tests:**
  - the XOR map's bijection, plus hand rows (`4m → bank m mod 4`);
  - `contract_trace` on HT1/HT2;
  - the schedule-file round trip (write, then parse, gives identical data).

## 9. Exit criterion

1. **Lint:** `verilator --lint-only -Wall` is clean for `rtl/`.
2. **Correlation:** every schedule of §4 matches at all three levels, or
   each deviation is root-caused and fixed per §5 and logged in
   `docs/CORRELATION.md`. The run reports schedules, tiles, cycles and MACs
   compared, with provenance.
3. **Tests:** all Phase 1–4 tests unchanged, plus HT1/HT2, T5a–T5d and the
   unit cases, in both presets.
4. **Sweeps S1 and S2:**
   - every op passes its in-model assertions, and every op time is at or
     above its formula lower bounds;
   - every program is bit-exact three ways;
   - the row-map rows of S1 equal Phase 4 exactly;
   - `results/phase5/sweep.csv`, a map-comparison plot and an arrangement
     plot.
5. **RESULTS.md** gets a Phase 5 section with hypothesis verdicts, checked
   by `scripts/check_results.sh`.

## 10. Mutation checks

| Mutant | Change | Must be caught by |
|---|---|---|
| R1 | RTL: row skew missing (row i's A not delayed) | HT1 and HT2 RTL traces, correlation (level 3) |
| R2 | RTL: drain row 0 first | HT1 trace, correlation (levels 2–3) |
| R3 | RTL: no clear on `first` (a tile adds onto the previous one) | correlation level 2 (any multi-tile case) |
| R4 | RTL: drain starts one cycle late | HT1 and HT2 traces, P2 directed level 1 |
| R5 | `contract_trace` uses `f + k + i` (drops j) | HT1/HT2 contract unit tests, correlation level 3 |
| R6 | Testbench presents B row k one cycle late | correlation levels 2–3 (the testbench is part of the comparison, so it must be able to fail) |
| X1 | XOR map ignored (row map used) | T5a, T5d |
| X2 | XOR map uses `row_index / (2B)` | unit bijection hand rows, T5d |

## 11. Hypotheses (before measuring)

**H1 (correlation):** every schedule matches at all three levels on the
first run. I expect any mismatch to be a testbench or export bug rather than
a contract disagreement, because Phases 2–4 already assert the contract's
consequences (tile latency, compute cross-check) in every run.

**H2 (XOR map fixes the prefill pathology).** At prefill L = 1024, INT8, the
Q·Kᵀ op (Phase 4: 1402154 cycles, bus busy 0.0935) gets **≥ 3× faster**
with the XOR map, and the whole program **≥ 2× faster**. The S rows
(stride = 4 rows) spread over all four banks (T5d), and their activations
overlap.

**H3 (Phase 4's precision prediction, retested).** With the XOR map, prefill
L = 1024 FP16/INT8 lands in **[1.3, 1.8]** and FP32/FP16 in **[1.4, 1.95]**:
the Phase 4 H2 ranges. This tests whether that hypothesis failed only
because of the controller.

**H4 (decode barely changes, GQA recovers).**
- Decode T = 4096, g = 1 changes by **at most ±10 %** with the XOR map,
  because it was not miss-bound (Phase 4 bus busy 0.63–0.83).
- GQA FP16 `T(16)/T(1)` drops from 4.4110 to **≤ 2.0**.

**H5 (equal-MAC prefill).** At L = 1024, FP16, XOR map, the 4×4-of-8×8
grid is the **slowest, by ≥ 1.5×** against the best. Its corner-SRAM mesh
link must carry the loads of 12 PEs: about 393k cycles per GEMM at
L = 64 B/cycle, against about 196k (DRAM) for 2×2. The 1×1-of-32×32 and
2×2-of-16×16 arrangements are **within 15 % of each other**: by bounds
about 462k vs 425k cycles. 1×1 pays F = 94 per tile; 2×2 pays mesh latency.

**H6 (equal-MAC decode).** At T = 4096, FP16, g = 1, the 1×1-of-32×32
array is the **slowest, by ≥ 1.6×** against the best. It uses 1 row of 32,
pays F = 94 on every 64-step tile, and has a single PE (by bounds about
46k cycles against about 23–27k for the others).

Each will be reported as confirmed, refuted or partly refuted, with the
numbers.

## 12. Decisions to confirm

1. **RTL scope:** the compute array only (INT8, q = 1, output-stationary,
   one accumulator set), not the PE controller, DMA or memories. Its
   inputs come from the model's recorded schedules.
2. **Clear-on-first tag** instead of the model's clear at feed start.
   They are equivalent under the no-overlap rule, and the protocol check
   enforces that rule (§1).
3. **Three-level comparison,** with the model side of level 3 computed by
   `contract_trace` from the model's own contract functions, plus the two
   hand-written traces as a referee independent of both (§3).
4. **14 compile-time (R, C) variants** rather than one runtime-configurable
   array (§4). This keeps the RTL like real fixed hardware.
5. **XOR bank map as the one controller remedy studied.** FR-FCFS, write
   buffering and a tile-major S layout are listed, not built (§6).
6. **The two sweeps and their configurations,** SRAM at the corner
   throughout (§7).

## 13. Code layout

```
rtl/os_array.sv, rtl/os_cell.sv           SystemVerilog array (§2)
tb/os_array_tb.cpp                         Verilator testbench: schedule in, three-level compare (§3)
tb/variants.cmake                          one verilate() per (R, C) variant
src/compute/contract_trace.{h,cpp}         contract expansion of a schedule (pure C++)
src/compute/schedule_io.{h,cpp}            schedule file write/parse (pure C++)
src/pe/pe.cpp, tools/apm_gemm.cpp, tools/apm_accel.cpp    schedule=FILE export
src/mem/dram_timing.{h,cpp}                bank_map = row | xor
tests/rtl/ht1.sched, ht1.trace, ht2.sched, ht2.trace      hand-written schedules and traces
tests/unit/contract_trace_test.cpp, dram_map_test (in dram_timing_test.cpp)
tests/sim/scenarios: t5a..t5d             DRAM map directed scenarios
python/correlate.py                        builds the §4 program set, runs model + os_array_tb, summarizes
python/sweep_p5.py                         S1, S2, plots, hypothesis numbers
scripts/phase5_exit.sh, phase5_mutation_check.sh, phase5_results.sh
docs/CORRELATION.md                        deviation log (§5)
```

## 14. Risks I'm flagging now

- **Verilator 5.020 (Ubuntu 24.04).** Its CMake `verilate()` with
  per-variant `-G` parameters and prefixes is the plan. The fallback is
  `verilator --cc --build` from custom commands. I'll record the version in
  provenance.
- **Build time:** 14 variants, the largest 16×16. If the full set makes
  ctest too slow, ctest builds only HT1, HT2 and (4, 16), and the exit
  script builds the rest.
- **The RTL is simulated, not synthesised.** It gives no timing closure,
  area or frequency. "Correlation" means cycle-behaviour equivalence with
  the contract and the model, and that is all I will claim.
- **Same author for model, contract and RTL.** The hand traces (§3.4) and
  the Phase 2 hand values (§4) come from the contract text, not from either
  implementation. They are the guard.
