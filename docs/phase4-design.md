# Phase 4 Design Note — Attention Workloads, Vector Unit, Precision Modes

Status: **approved 2026-10-14 (user: "go"); implementing**

## 0. What Phase 4 adds

```
 Python front end (python/attn.py)
   attention spec (prefill | decode, L or T, d, g, precision)
     │  lower
     ▼
   program = [ GEMM  S = Q·Kᵀ ] → [ VU softmax  P = softmax(S) ] → [ GEMM  O = P·V ]
     │  run each op on the model (one simulation per op, DRAM image handed over)
     ▼
   cycles per op, stall stacks, DRAM bytes, output O
     + bit-exact references (NumPy replica, standalone C++) + FP64 error

 Hardware (new block in bold):
   DRAM ─ global DMA ─ shared SRAM ─ mesh ─ PE grid     (Phase 3, now with precision modes)
   DRAM ─ **VU DMA ─ VMEM ─ vector unit (V lanes)**      (softmax)
```

Four things:
1. **Precision modes** INT8, FP16, BF16, FP32 for the matrix unit (§1). Each
   one changes operand bytes and array throughput.
2. **A bit-exact numerics spec** for every floating-point operation the model
   performs (§2). It lets the model match two independent references exactly,
   as INT8 does today.
3. **A vector unit (VU)** that runs a row-wise masked softmax (§3).
4. **Attention lowering** in a Python front end: prefill (causal, L queries)
   and decode (g query heads against a KV cache of length T), unfused,
   op by op (§4). Two GEMM-side additions make decode mappable: padded rows
   with a masked write-back, and a logical 1×P mapping (§5).

Not in Phase 4: fused (flash) attention, overlap between the VU and the
array, causal tile skipping, Q/K/V/O projections and MLPs, KV-cache append,
FP8. They go in LIMITATIONS at the end of the phase. Fusion and overlap are
candidate Phase 5 explorations.

## 1. Precision modes

### 1.1 What changes per mode

| Mode | Operand bytes `e` | Cycles per reduction step `q` | Accumulator (4 B) | Product |
|---|---|---|---|---|
| INT8 | 1 | 1 | INT32 | exact |
| FP16 | 2 | 2 | FP32 | exact in FP32 (11+11 ≤ 24 significand bits) |
| BF16 | 2 | 2 | FP32 | exact in FP32 (8+8 bits) |
| FP32 | 4 | 4 | FP32 | rounded to FP32 |

- **`q = e` (a fixed number of multiplier bits per cell).** A cell that does
  one INT8 MAC per cycle does one FP16 MAC every 2 cycles and one FP32 MAC
  every 4. So peak = `2·P·R·C / q` ops/cycle, and operand bytes per cycle at
  peak are the same in every mode. This is a modelling knob, not a datasheet
  ratio. `q` is configurable per mode, and the defaults are `q = e`.
- **Consequence I want to test, not assume:** with `q = e`, precision scales
  compute and operand traffic identically. Only the precision-independent
  terms keep lower precision from being proportionally faster: the 4-byte
  accumulator outputs (S, O), the FP32 softmax input, and the fill/drain
  per tile. See H2 in §10.
- **Accumulator outputs stay 4 bytes** (INT32 or FP32) in every mode, as in
  Phases 2–3. The GEMM output is therefore always 4 B per element.

### 1.2 Timing contract, generalized (Phase 2 §2.2 with `q`)

A chunk feed starting at f: `MAC(i, j, k)` occupies cycles
`[f + q·k + i + j, f + q·k + i + j + q)`. The edge holds each operand vector
for q cycles, and the skew registers still delay row i by i cycles and
column j by j cycles. Therefore:
- feed occupies `[f, f + q·Kt)`; the operand buffer is free at `f + q·Kt`;
- last MAC ends at `f_L + q·Kt + R + C − 3`;
- drain `[f_L + q·Kt + R + C − 2, f_L + q·Kt + 2R + C − 2)`;
- **tile done** = `f_L + q·Kt + F`, with `F = 2R + C − 2` **unchanged**;
- **`T_tile(R, C, K, q) = q·K + F`**.

For q = 1 this is exactly the Phase 2 contract, so the Phase 5 RTL (INT8)
spec does not change. The stall classifier's compute rule becomes
`[f, f + q·Kt)` plus the tail, and the §3.3-3 cross-check becomes
`compute_p = Σ_tiles (nc·q·Kt + F)`.

### 1.3 Bytes and capacity

- A block `R·Kt·e`, B block `Kt·C·e`, C tile `4·R·C` (unchanged).
- LMEM: `n_buf·(R·Kt + Kt·C)·e + n_out·4·R·C ≤ lmem_bytes`.
- SRAM ring slot: one chunk-set with e-byte blocks.
- DRAM-bytes formula (Phase 3 §2.2): `|A|`, `|B|` scale by e; `|C|` uses
  valid rows (§5.1).
- Shape rules unchanged: `Kt % 16 = 0` keeps every block a multiple of
  16 B for every e.
- The INT32 range check `K ≤ 131071` applies to INT8 only.

### 1.4 Regression

INT8 with `q = 1`, `e = 1` is the existing code path. Every Phase 1–3 test
must pass unchanged with identical expected values.

## 2. Numerics spec (what "bit-exact" means for floating point)

The model, a NumPy replica (`python/numerics.py`) and a standalone C++
reference (`tools/attn_ref.cpp`, no model headers) each implement this
section independently.

### 2.1 Formats and conversions

- FP16 and BF16 values are stored as their IEEE/bfloat16 bit patterns,
  little-endian, e = 2.
- **Conversion to FP32 is exact**: FP16 by decoding, including subnormals;
  BF16 by `bits << 16`.
- **FP32 → FP16 and FP32 → BF16 round to nearest, ties to even (RNE)**,
  including FP16 subnormals. No input can overflow (§2.4), so infinities and
  NaNs never occur. NaN handling is out of scope; NaN inputs are rejected.
- Inputs are generated as FP32 (`standard_normal(dtype=float32)`) and
  converted once on the host. This avoids double rounding from FP64, and the
  FP64 reference uses exactly those FP32 values.

### 2.2 GEMM accumulation

- Per output element, in **ascending global k** (the output-stationary
  order; chunking does not change it):
  - FP16/BF16: `acc = fl32(acc + a·b)`; the product is exact.
  - FP32: `p = fl32(a·b)`; `acc = fl32(acc + p)`. This is **not fused**
    (no FMA).
- **Build rule:** the whole project compiles with `-ffp-contract=off`.
  - GCC defaults to `-ffp-contract=fast` in GNU mode. It emits no FMA on
    baseline x86-64 today, but any `-march` flag would make the FP32
    results depend silently on the build.
  - Mutant M3 (§9) shows that the FP32 functional check catches an FMA.
- INT8: INT32, exact (unchanged).

### 2.3 The VU's exponential (`vu_exp`, FP32, x ≤ 0)

A hardware-style approximation, fully specified:
1. `t = fl32(x · LOG2E)`. Here `LOG2E` = the FP32 nearest to log2(e).
2. If `t < −125`, return 0. This flush keeps every result normal.
3. `n = nearbyint(t)` (ties to even), `f = t − n`. The subtraction is exact
   (Sterbenz), and `|f| ≤ 0.5`.
4. Evaluate `p = c6`, then `p = fl32(fl32(p·f) + c_k)` for k = 5…0. Each
   `c_k` is the FP32 nearest to `(ln 2)^k / k!`, so `c0 = 1`.
5. Return `ldexp(p, n)`. This is exact, because n ≥ −125 and p ≥ 2^−0.5.

**Accuracy:** the truncation term is `(0.5·ln2)^7/7! · 2^0.5 ≈ 1.7e-7`, about
2 ulp. **`vu_exp(0) = 1` exactly.** The coefficients are hard-coded as FP32
bit patterns in both implementations. A unit test recomputes them from the
formula.

### 2.4 Softmax of one row (the VU contract)

Input: a row s[0..n) (FP32, or INT32 in INT8 mode), a mask (§3.4), a scale
α (FP32; §4.4). Over the unmasked j:
1. `m = max_j s_j`. This is exact. In INT8 mode it is compared as INT32 and
   then converted, exactly, since |s| < 2^24.
2. `x_j = fl32(fl32(s_j − m) · α)`; `e_j = vu_exp(x_j)`. Masked j: `e_j = 0`.
   So `e_j ≤ 1`, and `e_j = 1` at the max.
3. **Sum in lane order.**
   - Lane `l < V` accumulates `e_l, e_{l+V}, e_{l+2V}, …` sequentially.
   - Then a pairwise tree: for w = V/2, V/4, …, 1, lane l += lane l + w.
   - `sum` = lane 0, so `sum ≥ 1`.
4. `r = fl32(1 / sum)` (correctly rounded). `p_j = fl32(e_j · r)`, so
   `p_j ≤ 1`.
5. Output encoding:
   - **FP16/BF16:** RNE of `p_j`.
   - **FP32:** `p_j`.
   - **INT8 `p_scale=fixed`:** `RNE(fl32(p_j · 127))`, a value in [0, 127];
     row scale `1/127`.
   - **INT8 `p_scale=row`:** `RNE(fl32(e_j · 127))`; row scale
     `fl32(r / 127)`. Because `p_j / max_j p_j = e_j`, this is per-row
     max-scaling at no extra cost. The normalization moves into the
     dequantization.

V must be a power of two.

## 3. Vector unit

### 3.1 Structure

`VecUnit` is a Phase 2 PE with a different compute engine:
- its own DmaEngine, on a DRAM port;
- VMEM, an `Sram` instance with Phase 1 timing on the DMA side;
- an **ideal** VU↔VMEM port, as for the array, with the same justification;
- a controller.

It is not on the mesh and does not use the shared SRAM. A VU op runs with
the PE grid idle (§4.3).

### 3.2 Timing of one row

Each row runs three passes (max; exp and sum; normalize and convert). A pass
streams `⌈n/V⌉` vectors one per cycle, and its result is available `t_vu`
cycles after the last vector enters (pipeline plus reduction tree). Each
pass depends on the previous one. Rows do not overlap.

```
T_row(n) = 3 · (⌈n / V⌉ + t_vu)
```

Masked elements cost the same as unmasked ones: the mask is an index
compare, so causal rows are not shortened.

### 3.3 Controller (Phase 2 §3.2 with rows as steps)

- **Load row r** (1-D descriptor, `4n` bytes from row-major S) into input
  buffer `r mod n_vbuf`. That buffer must be free: the compute of row
  `r − n_vbuf` has ended, because pass 2 overwrites the input with
  `e_j` and pass 3 reads it.
- **Compute row r** when it is loaded, the VU is free, and output buffer
  `r mod n_vout` is free (the store of row `r − n_vout` is done). Busy for
  `[c, c + T_row)`.
- **Store row r** at compute end. The P row goes straight into the **packed
  A layout of the next GEMM** (Phase 2 §3.1), as one 2-D descriptor:
  - row r is in block `mi = r / R`, local row `i = r mod R`;
  - segment c (`c = 0..n/Kt − 1`) goes to
    `p_base + ((mi·nc + c)·R + i)·Kt·e`;
  - in descriptor terms: `nc` rows of `Kt·e` bytes, destination stride
    `R·Kt·e`.

  So the host never repacks between ops.
- **INT8 `p_scale=row` only:** after the last row, one descriptor writes
  every row scale (`4·rows` bytes, rounded up to 16).
- `T_end` = the last store done.
- **Capacity check** (pure `plan_vu()`):
  `n_vbuf·4n + n_vout·n·e (+ scale buffer) ≤ vmem_bytes`. Requires
  `n % Kt = 0`.

### 3.4 Masks

- Row r is the query at position `r + q_off`. Column j is valid iff
  `j ≤ r + q_off` (causal), or always (no mask).
- Prefill: causal, `q_off = 0`. Decode: no mask, since the query attends to
  all T cached tokens, its own included.

### 3.5 VU stall stack (every VU-cycle gets one cause, asserted)

For each cycle in `[0, T_end)`:
- `busy`: inside a row compute;
- `load`: the next row is not loaded;
- `store`: the next row is loaded but its output buffer waits for a store,
  or every row is computed and stores are in flight;
- otherwise the classifier throws (unexplained).

**Asserted in every run:**
- Σ = T_end;
- `busy = Σ_rows T_row(n)` (the independent formula);
- DRAM bytes = `rows·n·(4 + e)` (+ scales).

## 4. Attention lowering (Python front end)

### 4.1 Workloads

One KV head with g query heads (grouped-query attention; g = 1 is
multi-head). Head dim d. A full model is H_kv repetitions of this. I do not
run the repetitions: under the barrier model they are identical up to DRAM
addresses, and that is stated wherever results are scaled.

| | Q | K, V | S, P | Mask |
|---|---|---|---|---|
| **prefill** | L × d (M = L) | L × d | L × L | causal |
| **decode** | g × d, padded to R rows | T × d (the KV cache) | g × T | none |

### 4.2 Program and layouts

```
op 1  GEMM   S[M×N] = Q[M×d] · Kᵀ[d×N]        A = Q (packed A), B = Kᵀ (packed B)
op 2  VU     P = softmax(S, mask, α)           reads S row-major, writes P in packed A layout (K-dim = N)
op 3  GEMM   O[M×d] = P[M×N] · V[N×d]          A = P, B = V (packed B)
```

- **KV-cache layout = the array's packed B layouts.**
  - K is stored as packed `Kᵀ`: block `(nj, c)` holds
    `K[nj·C .. +C][c·Kt .. +Kt]` transposed, Kt×C row-major.
  - V is stored as packed V: block `(nj, c)` = `V[c·Kt .. +Kt][nj·C .. +C]`.
  - This is a design choice: the cache is written once by the (unmodelled)
    projections in the format the matrix unit reads. Q is host-packed like
    any Phase 2 operand.
- **Only S → P crosses a layout boundary**, and the VU's store descriptor
  does it in hardware (§3.3). Host packing between ops is never used.
- **Shape rules** (no edge tiles, as before):
  - `d % C = 0`, `d % Kt = 0`;
  - prefill `L % lcm(R, C, Kt) = 0`; decode `T % lcm(C, Kt) = 0` and
    `g ≤ R`.
  - The front end rejects anything else with a message.

### 4.3 Execution model: barrier between ops, one simulation per op

- Op k+1 starts when op k's last write is done. Each op runs as its own
  simulation (`apm_op`), starting from an idle machine with all DRAM banks
  closed. The DRAM contents pass between ops as an image file.
- So **`T_program = Σ_ops T_op`** by definition. Ops never overlap, which is
  realistic for an unfused baseline.
- **Approximation:** open-row state does not carry across op boundaries. It
  costs at most `t_miss − t_hit` per bank per op. This goes in LIMITATIONS.
- **Alternative rejected:** one simulation with a sequencer re-arming the
  Phase 3 scheduler and PEs per op. It is more faithful by a few cycles, but
  it means invasive changes to Phase 3 code whose results are already
  published.

### 4.4 Scales

- **FP modes:** `α = fl32(1/√d)`. This is exact for d = 64.
- **INT8:**
  - Q, K and V each get a per-tensor symmetric scale
    `s_x = fl32(max|x| / 127)`, with `x_q = clip(RNE(x / s_x), −127, 127)`.
  - `α = fl32(fl32(s_q·s_k) · fl32(1/√d))`.
  - Quantization is host-side input preparation. The C++ reference receives
    the quantized tensors and scales as files.
- **Dequantization of O** (the INT32 PV output):
  `O = fl32(float(O_q) · fl32(s_row · s_v))`.
  - It is applied identically by the front end to the model output and by
    both references.
  - It is host post-processing at no modelled cost. LIMITATIONS: one multiply
    per output element, g·d elements per decode step.

### 4.5 Front-end interface

```
python3 python/attn.py --mode prefill --L 256 --d 64 --prec fp16 [--hw hw.json] [--tools build/release]
python3 python/attn.py --mode decode  --T 4096 --d 64 --g 8 --prec int8 --p-scale row
```

- **Output:** JSON with per-op cycles, stall stacks (GEMM: Phase 3
  categories per PE; VU: §3.5), DRAM bytes, and the program totals. With
  `--check`, also the bit-exact comparison and the FP64 errors.
- **Program-level PE stack:** each GEMM op contributes its six categories,
  and the VU op contributes `P·T_vu` PE-cycles to a seventh bucket,
  **`vu_wait`**. Asserted: Σ = P·T_program.

## 5. GEMM-side changes for decode

### 5.1 Padded rows with a masked write-back

- `m_valid ≤ M` is added to the layout. Rows `m_valid..M−1` of A are zero
  (host-zeroed; P's padding rows are never written by the VU, and the DRAM
  starts zeroed). They are computed, because the array cannot skip them,
  and **charged as compute**.
- **Only the write-back** (SRAM → DRAM) is limited to valid rows. The PE
  store into the SRAM C slot is unchanged.
- The DRAM-bytes formula uses `|C| = 4·m_valid·N`.
- **Useful ops** for the roofline: `2·m_valid·N·K`. Peak still counts every
  row, so padding shows up as lost efficiency, not as hidden work.

### 5.2 Logical mapping shape

- **Problem:** with the Phase 3 mapping, a GEMM with `tiles_m = 1` (every
  decode GEMM) gives tiles only to PEs with x = 0. On a 2×2 grid that leaves
  half the grid idle.
- **Fix:** the mapping uses a **logical** shape `px_map × py_map` (product
  P). Logical PE `(x', y')` = id `x'·py_map + y'`, placed at the physical
  node of that id, so the mesh is unchanged.
  - Wave tile `(bm·px_map + x', bn·py_map + y')`.
  - DRAM formula `|A|·tiles_n/py_map + |B|·tiles_m/px_map + |C|`.
- The default is the physical shape (Phase 3 unchanged). The front end uses
  `1 × P` for decode. Studied in H6.

## 6. Roofline and accounting

- **GEMM ops:** the Phase 3 roofline with `peak = 2·P·R·C / q`, e-scaled
  bytes, useful ops (§5.1) and the logical mapping in `B_max`.
  `python/roofline.py` gains `e`, `q`, `m_valid`, `px_map`, `py_map`.
- **VU op:** two ceilings in elements/cycle:
  - VU: `n / T_row(n)`;
  - DRAM: `W / (4 + e)`.
- **Program:**
  - achieved = useful ops / T_program;
  - plotted against program arithmetic intensity (useful ops / DRAM bytes)
    for prefill and decode at every precision.

## 7. Hand-derived tests (before running)

### Configuration P3 (unchanged from the Phase 3 note §5) for GEMM tests

DRAM W = 32 (D = 2), `t_hit = t_miss = 10`, Q = 16, global DMA M = 8;
SRAM 8 × 16 B, `t_sram = 2`, 64 KiB, ring D = 2; PE 4×16, Kt = 64,
`n_buf = n_out = 2`, LMEM 16 KiB, PE DMA M = 8; mesh h = 2, L = 16;
grid 1×1. GEMM 4×16×64 (one tile, one chunk).

The staging lemma is Phase 2 §3.3: `done_i = 14 + 2i`. PE loads at H = 0
complete burst i at `s + 6 + i` (P3-T1).

### P4-T1: P3-T1 in FP16 (e = 2, q = 2)
- **Staging:** A = 512 B (8 bursts), B = 2048 B (32 bursts). A staged
  `done_7` = 28; **B staged `done_39` = 92**.
- **Loads:** submitted 92, 40 bursts, **loaded 92 + 6 + 39 = 137**.
- **Feed:** `[137, 265)` (q·Kt = 128). Tile done 265 + 22 = **287**.
- **Store:** the C tile is still 256 B, so as in P3-T1 it takes 9 cycles:
  **296**.
- **Write-back:** 20 cycles: **T_end = 316**.
- **Stall stack:**
  - dram [0, 92) = 92;
  - loads [92, 137): burst 0 window gives network 2 and dma 4; bursts 1–39
    give dma 39;
  - compute 128 + 22 = 150;
  - store: dma 4, network 5;
  - idle 20.

  **Expected:** compute 150, dram 92, network 7, dma 47, sram_conflict 0,
  idle 20. **Σ = 316.** DRAM bytes = 512 + 2048 + 256 = **2816**.

### P4-T2: P3-T1 in FP32 (e = 4, q = 4)
- **Staging:** A = 16 bursts (staged 44); B = 64 bursts, **staged
  `done_79` = 172**.
- **Loads:** loaded 172 + 6 + 79 = **257**.
- **Compute:** feed `[257, 513)`, done **535**.
- **Store and write-back:** store **544**, **T_end = 564**.
- **Expected stack:** compute 278, dram 172, network 7, dma 87, idle 20.
  **Σ = 564.** DRAM bytes = **5376**.

### P4-T3: P4-T1 in BF16
Timing depends only on (e, q). **Expected:** cycles, stall stack and DRAM
bytes identical to P4-T1. The output differs; the functional check
covers it.

### P4-T4: P3-T1 (INT8) with `m_valid = 1`
Everything up to the store is as P3-T1 (store done 172). The write-back is
one 64 B row: SRAM read 172 → 174, DRAM column 174, data ends 186.
**T_end = 186.**

**Expected stack:** compute 86, dram 52, network 7, dma 27, idle 14.
**Σ = 186.** DRAM bytes = 256 + 1024 + 64 = **1344**.

### Configuration PV for VU tests

DRAM as P2 (W = 32, D = 2, `t_hit = t_miss = 10`, Q = 16, burst 64 B);
VMEM 8 × 16 B, `t_sram = 2`; VU DMA M = 8; V = 16, `t_vu = 4`;
`n_vbuf = n_vout = 2`. Lemma: load burst i done `14 + 2i`. A store of b
bursts from VMEM takes `t_sram + h + b·D = 12 + 2b` (Phase 2 store
derivation).

### P4-V1: 2 rows, n = 64, FP32 S → FP16 P, R = 4, Kt = 64 (nc = 1), no mask
- **Loads** (both submitted at 0; 256 B = 4 bursts each): row 0 done
  `done_3` = 20, row 1 done `done_7` = 28.
- **Compute:** `T_row = 3·(64/16 + 4) = 24`. Row 0 `[20, 44)`, row 1
  `[44, 68)`.
- **Stores** (128 B = 2 bursts, 16 cycles): row 0 `44 → 60`, row 1
  `68 → 84`.

**Expected T_end = 84.** VU stack: load 20, busy 48, store 16, **Σ = 84**.
P rows at `p_base + 0` and `p_base + 128`.

Variants:
- **INT8 fixed** (64 B, 1 burst, store 14): T_end **82**;
- **FP32 out** (256 B, store 20): T_end **88**.

### P4-V2: 1 row, n = 128, Kt = 64 (nc = 2), R = 4, FP16 out
- **Load:** 512 B, 8 bursts, done `done_7` = 28.
- **Compute:** `T_row = 3·(8 + 4) = 36`, so `[28, 64)`.
- **Store:** two 128 B segments at `p_base` and `p_base + R·Kt·e = 512`
  (same DRAM row), 4 bursts, **done 64 + 20 = 84**.

**Expected T_end = 84**, stack load 28, busy 36, store 20. The bytes at both
segment addresses are checked, and nothing else is written.

### Unit-level hand cases
- **FP16 RNE:**
  - 1.0 → `0x3C00`; 65504 → `0x7BFF`;
  - 2^−24 → `0x0001`; 2^−25 → `0x0000` (tie to even);
  - 1.5·2^−25 → `0x0001`;
  - 1 + 2^−11 → `0x3C00` (tie to even); 1 + 3·2^−11 → `0x3C02`.
  - Exhaustive: every FP16 pattern (except NaN) round-trips FP16 → FP32 →
    FP16 unchanged.
- **BF16 RNE:** 1 + 2^−8 → `0x3F80`; 1 + 3·2^−8 → `0x3F82`.
- **`vu_exp`:**
  - `vu_exp(0) = 1` exactly;
  - `t < −125` → 0;
  - relative error ≤ 4e-7 against `std::exp` (in double) on a grid of
    10⁵ points in [−87, 0].
- **Lane-order sum:** V = 2, e = [1, 2^−24, 2^−24, 2^−24].
  - Lanes: `1 + 2^−24 = 1` (tie to even) and `2^−24 + 2^−24 = 2^−23`.
  - Tree: **`1 + 2^−23`**.
  - A sequential sum gives **1**. This case exists so that mutant M5 is
    visible.
- **Softmax:**
  - an unmasked row of eight zeros, V = 4 → every `p = 0.125` exactly;
  - causal prefill row 0 → `p_0 = 1`, rest 0.
- **Packed-P store descriptor:** hand-listed addresses for R = 4, Kt = 16,
  n = 48, rows 0, 3, 4.
- **Logical mapping:** 1×4 on a 2×2 grid with a 1×8 tile grid.
  - PE ids 0–3 get tiles (0, 0..3) in wave 0 and (0, 4..7) in wave 1;
    physical nodes (0,0), (0,1), (1,0), (1,1).
  - DRAM-bytes formula vs hand count.
- **Capacity:** `plan_vu` accepts exactly the required VMEM and rejects one
  byte less. LMEM for FP32 P4-T2 accepts exactly `2·(1024 + 4096) + 512` and
  rejects one byte less.

## 8. Exit-criterion runs

1. **Directed:** P4-T1–T4, P4-V1–V2, the unit cases, and **all Phase 1–3
   tests unchanged**, in both presets.
2. **GEMM functional:** 200 random grid GEMMs, 50 per precision, bit-exact
   three ways:
   - the model;
   - NumPy (`python/numerics.py`, the §2.2 loop in FP32);
   - the C++ reference (`gemm_ref --prec`).

   Inputs: FP32 `standard_normal`, plus 10 % large-magnitude cases (×1000).
   INT8 as in Phase 3.
3. **Attention functional:** 80 random small programs, bit-exact three ways:
   - the model;
   - NumPy replica;
   - `attn_ref`.

   That is 8 seeds × {prefill, decode} × {FP32, FP16, BF16, INT8-row,
   INT8-fixed}, with L ∈ {32, 64, 96}, T ∈ {64, 128, 256},
   d ∈ {16, 32, 64}, g ∈ {1, 2, 4}, grids 1×1 / 1×2 / 2×2, R ∈ {4, 8},
   Kt ∈ {16, 32}. Only legal combinations (§4.2) are drawn, from a fixed
   seed list.
4. **Studies** (`python/study_p4.py`, sequential): every op must pass its
   in-model assertions, and every program must pass Σ = P·T_program. They
   run on the **A4 reference configuration**:
   - grid 2×2, R = C = 16, Kt = 64, `n_buf = n_out = 2`, ring D = 2;
   - W = 64, burst 64 B, `t_hit = 10`, `t_miss = 30`, Q = 32, global DMA
     M = 32, SRAM 16 banks;
   - mesh h = 2, L = 64;
   - VU V = 64, `t_vu = 4`, VU DMA M = 32, `n_vbuf = n_vout = 2`;
   - d = 64. Peak (INT8) = 2048 ops/cycle; F = 46.

   The study list:
   - prefill L ∈ {128, 256, 512, 1024} × 4 precisions (16);
   - decode, g = 1, logical 1×4, T ∈ {512, 1024, 2048, 4096, 8192} × 4
     precisions (20);
   - GQA, T = 4096, g ∈ {4, 8, 16} × {FP16, INT8} (6);
   - physical mapping, decode, T = 4096, g = 1 × 4 precisions (4);
   - VU lanes, prefill L = 1024, FP16, V ∈ {16, 32, 128} (3; V = 64 is
     already in the prefill list);
   - INT8-fixed for decode T ∈ {512 … 8192} (5).

   **54 programs.** INT8 programs use `p_scale=row` unless stated.
5. **Numerics:** relative Frobenius error
   `‖O − O₆₄‖_F / ‖O₆₄‖_F` and max abs error of the model's O vs an FP64
   reference on the unquantized FP32 inputs, for every study program.
   Inputs are N(0, 1).
6. **Outputs:**
   - `results/phase4/` `exit_criterion.txt`, `functional_gemm.csv`,
     `functional_attn.csv`, `study.csv`, `mutation_check.txt`;
   - plots: per-op time breakdown (prefill and decode × precision),
     program roofline, error vs T per precision;
   - RESULTS.md gets a Phase 4 section with verdicts.

## 9. Mutation checks

| Mutant | Change | Must be caught by |
|---|---|---|
| M1 | Feed lasts Kt, not q·Kt | P4-T1, compute cross-check |
| M2 | Block bytes ignore e (staging moves `Kt·C` bytes) | P4-T1, DRAM-bytes assertion, FP16 GEMM functional |
| M3 | FP32 accumulate uses `std::fma` | FP32 GEMM functional |
| M4 | FP32 → FP16 truncates instead of RNE | unit FP16 cases, FP16 attention functional |
| M5 | VU sums sequentially (not lane/tree order) | unit lane-order case, FP attention functional |
| M6 | VU ignores the causal mask | prefill attention functional |
| M7 | `T_row` uses 2 passes instead of 3 | P4-V1, VU busy cross-check |
| M8 | VU stores P row-major instead of packed | P4-V2 addresses, attention functional |
| M9 | Write-back ignores `m_valid` | P4-T4, decode DRAM-bytes assertion |
| M10 | Logical mapping ignored (physical used) | unit mapping case |
| M11 | `attn_ref` causal mask off by one (`j < i`) | three-way comparison flags the reference |

## 10. Hypotheses (before measuring)

The predictions use lower bounds per op, `max(compute, DRAM bytes / W)`,
summed over ops. The A4 configuration gives W = 64, P = 4, R = C = 16 and
F = 46. Phase 3 showed real runs sit well above such bounds (median 0.62 of
attainable), so the hypotheses are about **ratios and which side of a
bound** a run lands on, not absolute cycles.

**Prefill per op** (units L²/1024 cycles; e = q = 1, 2, 4):

| Op | Compute | DRAM | Bound INT8 / FP16 / FP32 |
|---|---|---|---|
| QKᵀ | `64q + 46` | `16·(4e + 4)` | 128 / 192 / 320 (DRAM) |
| softmax | VU `3·L/V → 48` | `16·(4 + e)` | 80 / 96 / 128 (DRAM) |
| PV | `64q` | `64e` | 64 / 128 / 256 (balanced) |
| **total** | | | **272 / 416 / 704** |

**Decode per op**, g = 1, logical 1×4 (units T/64 cycles):

| Op | Compute | DRAM | Bound INT8 / FP16 / FP32 |
|---|---|---|---|
| QKᵀ | `64q + 46` | `80e + 4` | 110 / 174 / 324 |
| softmax (1 row, serial) | | | ≈ 8 / 9 / 11 |
| PV | `64q` | `80e` | 80 / 160 / 320 |
| **total** | | | **≈ 198 / 343 / 655** |

In decode QKᵀ, the A (query) block is re-staged once per wave (`16eT`
bytes). That is the Phase 3 scheduler's lack of cross-wave reuse, and it is
counted.

- **H1 (correctness):** every directed value holds; all 200 GEMM and 80
  attention cases are bit-exact three ways; every op and program passes its
  sum assertions; BF16 timing equals FP16 timing exactly.
- **H2 (precision speedup is sub-linear):** prefill at L = 1024:
  - predicted FP16/INT8 time ratio **1.53**, FP32/FP16 **1.69**, not the
    2.0 that `q` alone suggests;
  - hypothesis: FP16/INT8 ∈ [1.3, 1.8] and FP32/FP16 ∈ [1.4, 1.95].
  - The reason: 4-byte S traffic and the FP32 softmax input do not shrink
    with precision.
- **H3 (unfused softmax is a large share of prefill):** the softmax op is
  ≥ 15 % of T_prefill at L = 1024 in every precision. Its share is largest
  at INT8 (bounds: 29 % / 23 % / 18 %).
- **H4 (decode QKᵀ is array-bound, not memory-bound, at INT8):**
  - Prediction: compute 110 vs DRAM 84 per T/64. The output-stationary
    array processes only C = 16 keys per tile, uses 1/16 of its rows (g = 1),
    and pays F = 46 on every 64-step tile.
  - Hypothesis: at T = 4096, INT8, `compute` is the largest category of the
    QKᵀ op's stall stack.
  - This contradicts the usual "decode is memory-bound". I expect it to hold
    only for this array organization and these knobs. FP16 is too close to
    call (174 vs 164), and FP32 should be DRAM-bound.
- **H5 (GQA is nearly free on this array):** at T = 4096, FP16:
  - predicted `T(g)/T(1)` = 1.06 (g = 4), 1.17 (g = 8), 1.41 (g = 16).
    Padded rows absorb the extra query heads; only the S, P and softmax
    traffic grows.
  - Hypothesis: ratio ≤ 1.25 for g ≤ 8 and ≤ 1.6 for g = 16. Per query
    head, that is a ≥ 6× cheaper decode at g = 8.
- **H6 (logical mapping):** decode QKᵀ, T = 4096, INT8. The physical 2×2
  mapping uses 2 of 4 PEs (predicted compute 220 vs 110), so the 1×4
  mapping is **≥ 1.6× faster** for that op.
- **H7 (numerics, N(0,1) inputs):** relative error of O vs FP64:
  - FP32 ≤ 1e-5;
  - FP16 ≤ 2e-3;
  - BF16 / FP16 error ratio ∈ [4, 16] (unit roundoff ratio 8);
  - INT8-row ≤ 0.1 at every T;
  - **INT8-fixed ≥ 0.5 at T = 4096**, and above 5× INT8-row at every
    T ≥ 1024.

  The reason for the last one: with `p ≈ 1/(1.65·T)`, `p·127 < 0.5` for
  almost every entry, so P rounds to zero. I expect this to be the clearest
  negative result of the phase.
- **H8 (VU lanes):** prefill L = 1024, FP16. Per row, the VU needs
  `3·(1024/V + 4)` cycles and the DMA 96. Prediction: VU-bound at
  V ≤ 32 (204, 108), DRAM-bound at V ≥ 64 (60, 36). Hypothesis: the VU
  busy fraction is ≥ 0.8 at V ∈ {16, 32} and ≤ 0.7 at V ∈ {64, 128}.

Each will be reported as confirmed, refuted or partly refuted, with the
numbers.

## 11. Decisions to confirm

1. **Precision model:** `e` bytes per operand, `q = e` cycles per reduction
   step, 4-byte accumulators and outputs in every mode,
   `T_tile = q·K + F` (§1).
2. **Bit-exact floating point:**
   - ascending-k FP32 accumulation without FMA, with `-ffp-contract=off`
     project-wide;
   - RNE conversions;
   - a specified polynomial `vu_exp` with a flush below `2^−125`;
   - lane/tree softmax summation (§2).
3. **Vector unit:** a separate block with its own DMA and VMEM on a DRAM
   port, V lanes, three dependent passes per row, rows not overlapped, P
   written directly in the next GEMM's packed layout (§3).
4. **Unfused program with a barrier between ops,** one simulation per op,
   DRAM image handed over, banks closed at each op start,
   `T_program = Σ T_op` (§4.3).
5. **Layouts and scope:**
   - the KV cache is stored in the array's packed B layouts;
   - projections and KV append are not modelled;
   - one KV head with g query heads;
   - INT8 per-tensor Q/K/V scales, P scaled fixed or per row, O
     dequantized on the host (§4).
6. **Decode support:** zero-padded rows charged as compute, a write-back
   masked to valid rows, and a logical `1×P` mapping for skinny GEMMs (§5).
7. **The A4 reference configuration and the 54-program study list** (§8).

## 12. Code layout

```
src/compute/numerics.{h,cpp}   Precision, e/q, FP16/BF16 RNE conversions, vu_exp, lane/tree sum (pure)
src/compute/systolic.*         q-aware feed length; INT32 or FP32 accumulators
src/pe/gemm_layout.*, pe_config.*, pe.cpp   e-byte blocks, m_valid, precision decode
src/soc/soc_plan.*, stall.*    e-scaled staging, logical mapping, masked write-back, q·Kt compute rule
src/vec/softmax.{h,cpp}        §2.4 row softmax, functional (pure)
src/vec/vu_plan.{h,cpp}        VMEM capacity, packed-P store descriptors (pure)
src/vec/vec_unit.{h,cpp}       VecUnit SystemC block (DMA + VMEM + controller)
src/vec/vu_stall.{h,cpp}       §3.5 classifier (pure)
tools/apm_op.cpp               one op (gemm | softmax) on a DRAM image; JSON out
tools/attn_ref.cpp             standalone C++ attention reference (no model headers)
tools/gemm_ref.cpp             + --prec
python/numerics.py             NumPy replica of §2
python/attn.py                 front end: spec → program, packing, run, references, FP64 error
python/attn_check.py           80-case three-way attention check
python/gemm_check.py           + --prec
python/roofline.py             + e, q, m_valid, logical mapping, VU ceilings
python/study_p4.py             §8.4 studies, CSV, plots, hypothesis numbers
tests/unit/numerics_test.cpp, softmax_test.cpp, vu_plan_test.cpp (+ soc_plan_test, systolic_test additions)
tests/sim/phase4_test.cpp      P4-T1..T4, P4-V1..V2
scripts/phase4_exit.sh, phase4_mutation_check.sh, phase4_results.sh
```
