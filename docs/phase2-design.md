# Phase 2 Design Note — Compute Unit, Local Memory, Double Buffering

Status: **approved 2026-10-14 (user: "go ahead with whatever is right"); implemented**

## 0. What Phase 2 adds

One processing element (PE): a systolic matmul unit, a per-PE local memory
(LMEM), and a controller that runs a GEMM `C = A·B` out of DRAM with double
buffering. For small cases the model moves and computes real data. Its result
must match NumPy and a standalone C++ checker exactly (integer mode: INT8
inputs, INT32 accumulation).

```
 DRAM ──(DMA, Phase 1)──► LMEM: [A buf 0][B buf 0][A buf 1][B buf 1][C out 0][C out 1]
                            │ ideal operand port (§2.4)          ▲ drain
                            ▼                                     │
                     Systolic array R×C (output-stationary) ──────┘
 DRAM ◄──(DMA)── LMEM C out buffers
```

Terminology: **PE** = one grid element (Phase 3 instantiates many). **Cell** =
one MAC position inside the R×C array.

## 1. One change to Phase 1: the DMA streams across descriptors

**Problem.** The Phase 1 DMA finishes one descriptor before starting the
next, so every A, B and C transfer pays the whole DRAM latency
(`t_hit + t_sram` or more) again. With separate A and B descriptors per
chunk, that exposed latency alone would make double buffering pointless at
any realistic setting. That would be a property of the DMA, not of the
compute/memory balance we want to study.

**Change.** Bursts of consecutive descriptors flow through the same burst
pipeline in order. The next descriptor's first read is issued as soon as a
buffer is free, possibly while the previous descriptor's writes are still in
flight. A descriptor completes at the write response of its own last burst.
Descriptors may differ in direction, so request-socket occupancy is tracked
**per socket** (the DRAM socket carries load reads and store writes). When a
read and a write want the same socket in the same cycle, **the write goes
first**: it frees a buffer and cannot deadlock.

**Regression.** Every Phase 1 scenario uses one descriptor per engine, so all
Phase 1 expected values are unchanged. The Phase 1 tests stay in the suite
unmodified. The DMA also gets a completion callback (`on_done(desc_id,
cycle)`), invoked at the cycle of the last write response.

## 2. Compute unit

### 2.1 Dataflow choice: output-stationary (OS)

| | Output-stationary (chosen) | Weight-stationary |
|---|---|---|
| Stays in cells | C partial sums (INT32) | B (weights) |
| Moves | A from the left, B from the top | A in, partial sums out |
| K longer than the array | Free: keep accumulating in place | Needs partial-sum spill and re-add every R of K |
| Output drain | Explicit drain phase (R cycles) | Outputs flow out continuously |
| RTL (Phase 5) | Simplest to write and check per cycle | Needs weight preload plus partial-sum handling |

OS fits Phase 2's "split K into chunks and keep accumulating" schedule,
gives a closed-form tile latency, and is the easiest to correlate per cycle in
Phase 5. Cost: the drain is exposed (§2.3) because there is one accumulator
set. Double-buffered accumulators would hide it, and are listed in
LIMITATIONS as an exploration option.

### 2.2 Cycle-level contract (the spec the Phase 5 RTL must implement)

Feed of one chunk starts at cycle f with Kt reduction steps. The edge reads
**one A column vector `A[0..R-1][k]` and one B row vector `B[k][0..C-1]` from
LMEM per cycle** at cycle `f + k`. Skew registers at the array edges delay
row i by i cycles and column j by j cycles. Therefore:

```
MAC(i, j, k) happens in cycle   f + k + i + j          (i < R, j < C, k < Kt)
```

Several chunks of one output tile are fed back to back. Chunk n starts at
`f_n ≥ f_{n-1} + Kt`, and a gap (operands not ready) is a bubble that shifts
everything after it. The accumulators keep summing across chunks.

After the **last** chunk of a tile (start `f_L`, length Kt):
- last MAC is in cycle `f_L + Kt + R + C − 3` (cell R−1, C−1);
- drain runs over cycles `[f_L + Kt + R + C − 2, f_L + Kt + 2R + C − 2)`.
  It writes one row of C INT32 results per cycle to the LMEM output buffer,
  row R−1 first, then row R−2, and so on;
- **tile done** = `f_L + Kt + 2R + C − 2`.

**Tile latency formula** (one chunk, from feed start to done):

```
T_tile(R, C, K) = K + 2R + C − 2  =  K (stream) + (R + C − 2) (fill skew) + R (drain)
```

Sanity case R=C=K=1: the MAC is in cycle f, the drain in cycle f+1, and the
tile is done at f+2 = f + 1 + 2 + 1 − 2. ✓

### 2.3 Accumulator and buffer rules

1. One accumulator set. Feed of tile t's first chunk may start only once
   tile t−1 is **done** (drained). So fill and drain are exposed once per
   output tile and amortized over K.
2. An operand buffer is free from the end of its chunk's feed (`f + Kt`).
   The edge has read every vector by then; the skew registers hold the rest.
3. Output buffer `t mod n_out` must be free (store of tile `t − n_out` done)
   before tile t's first chunk starts feeding. This is a conservative rule:
   it is checked at feed start, not drain start. It keeps the controller
   simple.

### 2.4 Local memory (LMEM)

- An instance of the Phase 1 `Sram` (banked, line-aligned). The DMA port is
  a TLM target with Phase 1 timing.
- **The operand/result port is ideal:** R + C bytes in and C×4 bytes out per
  cycle, never conflicting with the DMA. Justification: ping/pong buffers in
  real designs sit in separate banks, so the array and the DMA touch
  different banks. Cost: LMEM port bandwidth is not explored. This goes in
  LIMITATIONS.
- **Capacity check (two layers):**
  (a) At configuration, the controller requires
  `n_buf·(R·Kt + Kt·C) + n_out·4·R·C ≤ lmem_bytes`, otherwise it fails with
  a message naming every buffer.
  (b) At run time, every LMEM buffer comes from a bump allocator that throws
  on overflow or misalignment, and each DMA descriptor and compute access is
  bounds-checked against its buffer.
- **Accumulation-range check:** INT8×INT8 ≤ 2^14 in magnitude, so the INT32
  sum cannot overflow if `K·16384 ≤ 2^31 − 1`, i.e. **K ≤ 131071**. Larger K
  is rejected.

## 3. GEMM mapping and controller

### 3.1 Layout

The host packs operands in DRAM in tile order before the run (like weight
pre-packing). Every operand load is then one contiguous 1-D DMA:
- `A_pk` (base `a_base`): block (mi, c) = `A[mi·R .. +R][c·Kt .. +Kt]`,
  R×Kt row-major, at offset `(mi·nc + c)·R·Kt`.
- `B_pk` (base `b_base`): block (nj, c) = `B[c·Kt .. +Kt][nj·C .. +C]`,
  Kt×C row-major, at offset `(nj·nc + c)·Kt·C`.
- `C`: plain row-major M×N INT32 at `c_base`. The store is a 2-D descriptor:
  R rows of 4C bytes, stride 4N.

Constraints: M % R = 0, N % C = 0, K % Kt = 0, **Kt % 16 = 0** (so every A
and B block is a multiple of 16 bytes), and **C % 4 = 0** (so each 4C-byte C
row is 16-byte aligned). Violations are configuration errors. Blocks may
straddle DRAM rows. The DMA splits at burst-aligned DRAM addresses, so no
burst ever crosses a row (Phase 1 §3.6). The §5 test configuration happens
to keep every block inside one row, which keeps the hand derivations simple.
(The draft also required that for every configuration. That is not needed
for correctness, so it was dropped when implementing.)

**Loop order:** output tiles row-major over (mi, nj), and for each tile its
chunks c = 0..nc−1. Step `s = tile·nc + c`. There is **no operand reuse
across tiles**: B blocks are reloaded for every mi. That is deliberately
naive. Phase 3's scheduler can improve it, and the difference will be
measurable.

### 3.2 Controller (SC_METHOD state machine, acts at cycle boundaries)

Each wake-up runs, in order:
1. **Issue loads:** while step `ℓ` (next to load) has a free operand buffer
   (`ℓ < n_buf`, or the feed of step `ℓ − n_buf` has ended), submit A then B
   for step ℓ to the DMA.
2. **Start a feed:** if the array is not feeding, step `φ` (next to feed) is
   fully loaded, and (if c = 0) the previous tile is done and its output
   buffer is free, start feed `[now, now + Kt)`. The functional data path
   runs here (§4).
3. **Finish tiles:** at tile done, write the R×C INT32 result into the output
   buffer and submit its store.

Wake-ups: a DMA completion (same cycle, next delta), feed end, tile done, and
time 0. Reaction latency is 0 cycles. A load completing at cycle X lets a
feed start at X.

### 3.3 DMA burst-stream lemma (used by every derivation below)

Setting: `t_miss = t_hit = h`, D = burst/W cycles per burst, LMEM writes never
conflict, and M (DMA buffers) ≥ `(h + D + t_sram)/D + 1`. If the DMA is idle
and bursts are available from cycle s on without interruption, burst i
(i = 0, 1, …, counted across descriptors) completes its write at

```
done_i = s + h + D·(i+1) + t_sram
```

So n back-to-back bursts take `h + n·D + t_sram` (the Phase 1 H2 form).

*Proof sketch:* burst i is issued at `s+i` (i < M) or when buffer i−M frees
(`done_{i−M}`). Data goes on the bus at `max(issue + h, bank chain, bus_free)`.
The condition on M keeps `issue + h` ≤ the bus slot `s + h + D·i`. The
same-bank chain `col_{i−1} + D` never exceeds the bus slot either. So the bus
is the only limit, and burst i occupies `[s+h+D·i, s+h+D·(i+1))`.

The same argument shows that a stream submitted while an earlier one is still
draining continues seamlessly, provided its first burst is issued ≥ h cycles
before the bus would otherwise go idle. Each case below checks this.

## 4. Functional path and references (rule 4)

- **Model data path:** at feed start, the compute unit reads the chunk's
  A/B blocks from the LMEM backing store and adds `Σ_k A[i][k]·B[k][j]`
  (INT32) into its accumulators. At tile done it writes them to the output
  buffer. The DMA moves real bytes (Phase 1).
- **Reference 1, NumPy:** `python/gemm_check.py` generates random INT8 A and B
  (seeded `numpy.random.default_rng`), runs the model (`apm_gemm`, row-major
  binary in, C out), and compares with
  `A.astype(np.int32) @ B.astype(np.int32)`, bit-exact.
- **Reference 2, standalone C++:** `tools/gemm_ref.cpp`, a triple loop on
  the unpacked row-major inputs. It includes no model header and is compared
  on the same files.
- The packing in §3.1 is harness code. The references never see packed data,
  so a packing bug also shows up as a mismatch.
- **Cases:** N = 200 random cases. Each draws R ∈ {1,2,4,8}, C ∈ {16,32},
  Kt ∈ {16,32,64}, and M/R ∈ 1..4, N/C ∈ 1..3, nc ∈ 1..4. Values are
  uniform over [−128, 127], and 10 % of cases are all −128 (the largest
  accumulation magnitude). The seed list is fixed and recorded.

## 5. Hand-derived timing tests (before running)

Test configuration P2 (all parameters are modelling knobs):

| | |
|---|---|
| DRAM | B=4, rows 1024 B, **W=32 B/cycle (D=2)**, **t_hit = t_miss = 10** (row effects already covered in Phase 1; this isolates overlap), Q=16, burst 64 B |
| LMEM | 8 banks × 16 B, t_sram=2, 16 KiB |
| DMA | burst 64 B, **M=8** (lemma needs M ≥ (10+2+2)/2+1 = 8) |
| Array | **R=4, C=16**, Kt=64 → fill+drain `F = 2R + C − 2 = 22` |
| Blocks | A block 4×64 = 256 B (4 bursts), B block 64×16 = 1024 B (16 bursts): **20 bursts per step** |
| Bases | a_base 0, b_base 64 KiB, c_base 128 KiB |

Store of one tile (4 rows × 64 B, N = C so contiguous). LMEM reads at x..x+3
respond at x+2..x+5. DRAM writes are accepted at x+2..x+5 into a bank chain
(col x+2, x+4, x+6, x+8), data ends x+14, +16, +18, +20.
**Store = 20 cycles** (`t_sram + h + 4·D`).

Per step the bus needs 40 cycles and the feed 64. **Compute-bound** at W=32.

### P2-T1: one tile, one chunk (M=4, N=16, K=64)
Loads (20 bursts from 0): A done at `done_3 = 20`, B at `done_19 = 10+40+2 =
52`. Feed `[52,116)`. Tile done `52+64+22 = 138`. Store `[138 → 158]`.
**Expected: done 158** (= 52 + T_tile(4,16,64)=86 + 20).

### P2-T2: one tile, 4 chunks (K=256), double-buffered (n_buf=2)
Steps 0 and 1 load at 0 (40 bursts): done 52 and 92. Feeds:
- f0 = 52 → buffer 0 free 116, load step 2 at 116 (DMA idle since 92):
  done 116+52 = 168.
- f1 = max(116, 92) = 116 → buffer 1 free 180, load step 3 at 180: done 232.
- f2 = max(180, 168) = 180. f3 = max(244, 232) = 244.

Tile done 244+64+22 = 330. Store → 350.
**Expected: 350** = `L0 + nc·Kt + F + S` = 52 + 256 + 22 + 20.

### P2-T3: same as T2, single buffer (n_buf=1)
Load at 0 → 52, feed [52,116). Load at 116 → 168, feed [168,232). Load at
232 → 284, feed [284,348). Load at 348 → 400, feed [400,464). Done 486,
store → 506.
**Expected: 506** = `nc·(L + Kt) + F + S` = 4·116 + 42. Double buffering
saves 156 cycles here.

### P2-T4: T2 with W=16 B/cycle (D=4): memory-bound, double-buffered
Lemma with D=4 (M=8 ≥ (10+4+2)/4+1 = 5): burst i done at `16 + 4i`. Step c
is done at burst 20c+19: **92, 172**, and, if the stream stays continuous,
252, 332.
- f0 = 92, buffer 0 free 156 → step 2 submitted at 156. Bursts 36–39 are
  still in flight and bursts 40–43 are issued at 156–159. Their bus slots
  start at 170 ≥ 156+10, so the stream is continuous → step 2 done at 252.
- f1 = max(156, 172) = 172, free 236 → step 3 submitted at 236. Burst 60 is
  issued at 236 and its slot starts at 250 ≥ 246 → continuous → done 332.
- f2 = max(236, 252) = 252. f3 = max(316, 332) = 332. Tile done 332+64+22 =
  418.

Store with D=4: LMEM responses 420–423. DRAM bank chain col 420, 424, 428,
432, data ends 434, 438, 442, 446 → 446.
**Expected: 446** = `(h + 80·D + t_sram) + Kt + F + S` = 332 + 64 + 22 + 28.
The compute unit waits for loads (feed gaps: 172−156, 252−236, 332−316 =
16 cycles each).

### P2-T5: two output tiles (M=8, N=16, K=64), n_buf=2, n_out=2
Steps 0 and 1 (A0+B0, A1+B0) load at 0 → 52, 92.
- Tile 0: feed [52,116), done 138. Store 0 at 138 → 158.
- Tile 1: feed at max(138 (tile 0 done), 92) = 138, [138,202), done 224.
  Store 1 → 244.

**Expected: tile 0 done 138, store 0 done 158, tile 1 done 224, total 244.**
The store of tile 0 overlaps tile 1's feed, and F is exposed once per tile.

### P2-T5b: T5 with n_out = 1
Tile 1 can feed only once store 0 is done: f = max(138, 92, 158) = 158 →
done 158+86 = 244, store → 264. **Expected: 264.**

### P2-T6: DMA streams across descriptors (Phase 1 change, direct test)
One DMA, two 256 B DRAM→LMEM descriptors (different DRAM rows) submitted at
0, under P2. Lemma: 8 bursts back to back. **Expected: descriptor 0 done 20,
descriptor 1 done 28.** The old one-at-a-time DMA would give 20 and
20 + 20 = 40.

### P2-T7: capacity
P2 needs `2·(256+1024) + 2·256 = 3072 B`. LMEM 3072 → accepted. 3071 →
configuration error. K = 131072 (a multiple of Kt = 16, so only the range
check can reject it) → rejected. K = 131056 (the largest multiple of 16 within
the bound 131071) → accepted (range check only, not run). The checks run in
pure C++ (`plan_pe`) before any SystemC module is built. A full T1 run with
LMEM exactly 3072 B must still give 158.

### Unit-level formula tests
`T_tile` for (R,C,K) = (1,1,1) → 2, (4,16,64) → 86, (16,16,256) → 302. The
per-cycle MAC schedule `f + k + i + j` is checked on R=2, C=3, K=2 against
a hand-written table (first MAC 0, last MAC 4, drain [5,7), done 7).

## 6. Code layout

```
src/compute/systolic.{h,cpp}     tile_latency(), MAC schedule, functional accumulate (pure C++)
src/pe/lmem_alloc.{h,cpp}        bump allocator with capacity/alignment checks (pure C++)
src/pe/gemm_layout.{h,cpp}       packing/addresses for §3.1 (pure C++)
src/pe/pe_controller.{h,cpp}     SC_METHOD controller (§3.2)
src/pe/pe.{h,cpp}                PE = LMEM (Sram) + compute + controller + DMA, bound to a DRAM
src/dma/dma_engine.*             streaming across descriptors + on_done callback (§1)
tools/apm_gemm.cpp               run one GEMM from binary files; prints cycles JSON
tools/gemm_ref.cpp               standalone C++ reference
python/gemm_check.py             random cases, NumPy reference, 3-way comparison
tests/unit/systolic_test.cpp, lmem_alloc_test.cpp, gemm_layout_test.cpp
tests/sim/phase2_test.cpp        P2-T1..T7 (one simulation per process)
```

## 7. Mutation checks (the comparison must be able to fail)

| Mutant | Change | Must be caught by |
|---|---|---|
| N1 | `T_tile` drops the drain (K + R + C − 2) | P2-T1, unit formula |
| N2 | Accumulator overwrites instead of adds | functional (any nc>1 case) |
| N3 | DMA back to one descriptor at a time | P2-T6, P2-T2 |
| N4 | Feed does not wait for its loads | P2-T4, functional |
| N5 | Operand buffer freed at feed start (not end) | P2-T3 |
| N6 | Next tile does not wait for the previous drain | P2-T5 |
| N7 | Output buffer rule ignored | P2-T5b |
| N8 | Packing swaps two B blocks | functional only (timing unchanged) |
| N9 | C++ reference loop off by one in k | 3-way comparison flags the reference |

## 8. Hypotheses (written before measuring)

- H1 (compute-bound, double buffer): `T = L0 + nc·Kt + F + S`. P2-T2 = 350.
- H2 (single buffer): `T = nc·(L + Kt) + F + S`. P2-T3 = 506.
- H3 (memory-bound, double buffer): `T = (h + n·D + t_sram) + Kt + F + S`.
  P2-T4 = 446.
- H4: Fill+drain cost `F = 2R + C − 2` is paid once per output tile, so
  efficiency `nc·Kt / (nc·Kt + F)` rises with K. Checked in Phase 3/5 sweeps,
  not asserted here.
- H5: all 200 random cases match NumPy and the C++ reference bit-exactly.
  Mutants N2/N8/N9 show the comparison can fail.

## 9. Decisions to confirm

1. DMA streams across descriptors, write-first per socket (§1). Phase 1
   expected values are unchanged.
2. Output-stationary dataflow, one accumulator set, so fill and drain are
   exposed per tile (§2.1, §2.3).
3. Cycle contract `MAC(i,j,k) at f+k+i+j`, `T_tile = K + 2R + C − 2`, drain
   R cycles bottom row first (§2.2). This is the Phase 5 RTL spec.
4. Ideal LMEM compute port; the DMA side uses Phase 1 SRAM timing (§2.4).
5. Host-packed operands, naive loop order with no reuse (§3.1).
6. Test config with `t_miss = t_hit` to isolate overlap from row effects (§5).
7. N = 200 random functional cases, NumPy plus a standalone C++ reference.
