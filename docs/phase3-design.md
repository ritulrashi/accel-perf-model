# Phase 3 Design Note — Accelerator Top, Stall Stack, Roofline

Status: **approved 2026-10-14 (user: "go for it"); implemented**

## 0. What Phase 3 adds

```
            ┌────────── mesh (Px × Py routers, XY routing) ──────────┐
 DRAM ◄─► global DMA ◄─► shared SRAM ═ attached at router `sram_node`  │
 (Ph.1)    (Ph.1/2)      (banked, Ph.1)                                │
                              ▲  per-PE ports                          │
                              │                                        │
            PE(x,y) at router (x,y): PE DMA ◄─► LMEM ◄─► systolic array│
            (Phase 2 PE, operands now from the shared SRAM over the mesh)
            └───────────────────────────────────────────────────────────┘
 Scheduler: maps output tiles to PEs, stages operand blocks DRAM→SRAM
            (with reuse across PEs), writes C tiles back SRAM→DRAM.
```

Data path:
- **operands:** DRAM →(global DMA)→ shared SRAM →(mesh)→ PE LMEM.
- **results:** LMEM →(mesh)→ SRAM →(global DMA)→ DRAM.

PEs never touch DRAM directly. Everything they read crosses the mesh and
the SRAM, so every stall category is reachable.

## 1. Mesh network

### 1.1 Model (pure C++ `MeshTiming`, unit-tested)

- Px × Py routers. PE(x, y) sits at router (x, y). The shared SRAM attaches
  to router `sram_node` (configurable; default (0, 0), a corner).
- Directed links between neighbouring routers. **XY routing:** first along x,
  then along y. Hop count H = |dx| + |dy|.
- Parameters: hop latency `h` (cycles per link), link bandwidth `L`
  (bytes/cycle).
- Packet sizes:
  - read request and write acknowledgement: a control packet,
    `ser = 1` cycle;
  - read response and write request: carry the burst data,
    `ser = ⌈bytes / L⌉`.
- **Link reservation (virtual cut-through, unbounded router buffers).** A
  packet injected at cycle c goes through its links in path order:
  ```
  t_0 = c
  for each link l on the path:
      start_l      = max(t_l, link_free[l])
      link_free[l] = start_l + ser          # the link carries this packet for ser cycles
      t_{l+1}      = start_l + h            # the head reaches the next router
  delivered = t_H + ser   (H ≥ 1)           # the tail reaches the destination
  delivered = c + 1       (H = 0)           # local: through the router only, no link
  ```
  Unloaded latency is `H·h + ser`. Contention appears as later `start_l`.
  Backpressure is not modelled: buffers are unbounded, so an upstream link
  is never held by a blocked downstream one. This goes in LIMITATIONS.

### 1.2 TLM integration (`Mesh` module, AT like Phase 1)

- **PE side:** a multi-passthrough target socket, one port per PE (PE id =
  `x·Py + y`). Requests are sampled at mid-cycle like any Phase 1 target.
  Every request is accepted at its sample point (`END_REQ`; unbounded
  buffers), and its links are reserved there in round-robin port order.
- **SRAM side:** one initiator socket per PE, bound to SRAM ports 1..P (port
  0 is the global DMA). At the boundary of the delivery cycle, the mesh sends
  `BEGIN_REQ` on that PE's socket. Requests from one PE are always delivered
  in strictly increasing cycles (proof in the code comment), so the
  one-outstanding-request rule holds.
- **Responses:** the SRAM's `BEGIN_RESP` (at cycle r) is completed
  immediately. At the sample point of r the mesh reserves the reverse path
  (round-robin by port) and delivers `BEGIN_RESP` to the PE at the boundary
  of the delivery cycle.
- **The global DMA does not use the mesh** (a dedicated DRAM↔SRAM path).

## 2. Scheduler: mapping GEMM onto the grid

### 2.1 Tile → PE mapping and waves

Output tiles (R×C, as in Phase 2) form a `tiles_m × tiles_n` grid. They are
grouped into **waves** of Px×Py tiles. Wave (bm, bn) gives tile
`(bm·Px + x, bn·Py + y)` to PE(x, y). Waves are ordered row-major over
(bm, bn). A PE whose tile falls outside the matrix has nothing in that wave
(a partial wave). Each PE runs its tiles in wave order with the Phase 2
controller. PEs are **not** in lockstep.

### 2.2 Staging with reuse (the reason for the shared SRAM)

**Chunk-set** `(w, c)` = the distinct operand blocks that wave w needs for
chunk c:
- one A block per distinct tile row (up to Px);
- one B block per distinct tile column (up to Py).

A block is staged once per chunk-set and read by every PE that needs it.
That is the reuse.
- **DRAM traffic (full waves):**
  `|A|·tiles_n/Py + |B|·tiles_m/Px + |C|`. This formula is asserted against
  the DRAM byte counters in tests.
- **SRAM ring:** D slots (configurable), each sized for one full chunk-set,
  plus two C slots per PE (`4·R·C` bytes each). Capacity is checked like
  LMEM (Phase 2 §2.4, in pure `plan_soc()`).
- **Staging order:** chunk-sets in sequence q = 0, 1, … (waves in order, then
  chunks). Chunk-set q goes to slot q mod D. It is submitted to the global
  DMA (A blocks first, then B blocks) when its slot is free, meaning every
  consumer of chunk-set q−D has finished loading it into LMEM.
- **PE load gating:** a PE may submit its loads for step s only once **both**
  blocks of that step are staged (both block descriptors complete).
- **Results:** a PE's k-th tile is stored to its C slot k mod 2. That store
  may start only when the write-back of its tile k−2 is complete. When the
  store completes, the scheduler submits the write-back
  (SRAM → DRAM, row-major C).
- **Same-cycle order in the scheduler:** write-backs are submitted before
  staging (they free C slots).

### 2.3 PE changes (Phase 2 controller generalized)

The PE gets a `PeJob` interface in place of the fixed DRAM layout:
- operand source addresses per step;
- an `operands_ready(step)` gate;
- a store destination and `store_allowed(tile)` gate;
- completion callbacks: loads-done(step), store-done(tile).

The Phase 2 single-PE mode implements this interface with "always ready,
DRAM addresses", so **every Phase 2 test runs unchanged**.

## 3. Stall stack: every PE-cycle gets exactly one cause

### 3.1 Categories (by where the critical data is in that cycle)

| Category | A PE-cycle is charged here when… |
|---|---|
| `compute` | the array is feeding (`[f, f+Kt)`) or in a tile's fill/drain tail (`[last feed end, tile done)`, F cycles) |
| `dram` | the PE waits for an operand block that is **not on chip yet** (its staging was submitted but is not complete), or for a C slot whose write-back is in flight |
| `sram_conflict` | the critical burst is inside a banked memory (shared SRAM or LMEM) **waiting for a bank held by another request** |
| `network` | the critical burst (or its response or ack) is inside the mesh: link waits, hop latency and serialization |
| `dma` | the critical burst is anywhere else: queued in the PE's DMA, in a banked memory's base access latency, or in the LMEM write |
| `idle` | the PE has no work (no tile left, or none in this wave), its work is finished, or it waits for staging that cannot start because the ring slot is held by slower PEs (synchronization) |

"Critical burst" = the **oldest unfinished** burst of the descriptor(s) the
PE is waiting on (in-order completion semantics).

### 3.2 How it is computed (pure `classify_pe()`, post-hoc)

The model records timestamps; recording never changes timing:
- **per step:** feed start, loads complete;
- **per tile:** tile done, store submitted, store done;
- **per block:** staging submitted, staged;
- **per burst:** a trace of segments `[from, to) → category`, appended by
  each component that holds the burst's payload, via a TLM extension:
  - the mesh appends `network` for every packet, from injection to delivery;
  - a banked memory appends `sram_conflict` for `[a, a + conflict_delay)`
    and `dma` for the rest of `[a, response)`, where `conflict_delay = (max_i
    c_i) − (a + max_b own_b − 1)`. This is the extra delay caused by other
    requests, placed at the start of the residency by convention;
  - gaps between segments count as `dma`.

For each PE and each cycle c in `[0, T_end)`, the classifier applies these
rules in order:
1. Inside a feed or fill/drain tail → `compute`.
2. Else, if a step s* has not started feeding:
   - **loads complete:** the feed must be blocked by the output buffer.
     Classify the store of tile `t(s*) − n_out`, using rule 4.
   - **not loaded, some block unstaged:** submitted → `dram`; not submitted
     → `idle`.
   - **not loaded, both blocks staged:** classify the oldest unfinished
     burst of step s*'s loads by its trace at c.
3. Else, if a store is unfinished → rule 4 on the oldest one. Otherwise →
   `idle`.
4. **A store:** not submitted (C slot busy) → `dram`. Submitted → the
   oldest unfinished burst's trace at c.

**Every branch is asserted.** "Loads complete but no output-buffer reason",
or "staged but loads not submitted", throws `unexplained stall at PE p cycle
c`. The classifier therefore also cross-checks the controller.

### 3.3 Sum assertions (in every run, model aborts on violation)

1. Per PE: Σ categories = T_end, where `T_end` = the cycle the last
   write-back completes.
2. Whole chip: Σ over PEs = P·T_end.
3. **Independent cross-check:** `compute_p = Σ_{tiles of p} (nc·Kt + F)`
   (from the Phase 2 formula, not from the classifier).
4. DRAM bytes = the §2.2 formula (full-wave runs).

(3) and (4) are what make (1) non-trivial. A classifier that charged a
drain cycle to `idle` would still sum to T_end, but (3) would fail.

## 4. Roofline (hierarchical: compute, DRAM, NoC)

`ops = 2·M·N·K` (a MAC is 2 ops). `T_end` is measured.
- **Achieved** = `ops / T_end` (ops per cycle).
- **Peak** = `2·P·R·C` (every PE's array busy every cycle).
- **DRAM ceiling** = `AI_dram · W`, where `AI_dram = ops / DRAM bytes`
  (the §2.2 formula).
- **NoC ceiling** = `ops · L / B_max`, where `B_max` = load bytes on the
  busiest link out of `sram_node` (XY routing). PEs at `sram_node` use no
  link.
- **Attainable** = `min(peak, DRAM ceiling, NoC ceiling)`. **Predicted
  region** = whichever term is smallest.

`python/roofline.py` computes these from the formulas only (never from
simulator output), so they are the hand calculation. Assertion:
`achieved ≤ attainable` for every run. Exceeding a ceiling is impossible,
so it would be a bug.

### 4.1 Worked examples (square: R=C=a, grid g×g, Kt=64, M=N=2ag, K=1024, SRAM at corner (0,0))

DRAM bytes = `8agK + 16a²g²`, so **`AI_dram = agK / (K + 2ag)`**. Peak =
`2a²g²`. For g=2, the +x link out of (0,0) carries PEs (1,0) and (1,1),
8 tiles × 2aK load bytes, so **NoC ceiling = 2aL**.

| a | g | W | L | AI_dram | Peak | DRAM ceil | NoC ceil | Predicted |
|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 64 | 64 | 8192/1040 = 7.877 | 128 | 504.1 | 512 | **compute** |
| 8 | 2 | 16 | 64 | 16384/1056 = 15.52 | 512 | 248.2 | 1024 | **DRAM** |
| 8 | 2 | 64 | 64 | 15.52 | 512 | 993.0 | 1024 | **compute** |
| 16 | 2 | 64 | 16 | 32768/1088 = 30.12 | 2048 | 1927.5 | 512 | **NoC** |
| 16 | 2 | 64 | 64 | 30.12 | 2048 | 1927.5 | 2048 | **DRAM** |

**Closed form for when a square config is compute-bound** (DRAM vs peak,
large K): `agK/(K+2ag) · W ≥ 2a²g²` ⟺ approximately `W ≥ 2ag`. Bigger
arrays and grids need proportionally more DRAM bandwidth, because wave
reuse grows only linearly in a·g while peak grows quadratically.

## 5. Hand-derived system tests (before running)

Configuration P3:
- DRAM and global DMA as P2: W=32, `t_hit = t_miss = 10`, Q=16, M=8.
- SRAM: 8 banks × 16 B, `t_sram = 2`, 64 KiB, D = 2.
- PE: array 4×16, Kt=64, n_buf = n_out = 2, LMEM as P2, PE DMA M=8.
- Mesh: h = 2, L = 16 (data ser = 4, control ser = 1).

GEMM 4×16×64: one tile, one chunk.

### P3-T1: grid 1×1 (PE at `sram_node`, H = 0)

- **Staging** (burst lemma, Phase 2 §3.3): A staged 20, B staged 52.
- **Loads:** submitted at 52. Request delivered at issue+1, SRAM response
  +2, response delivered +1, LMEM write +2, so burst i is done at
  `58 + i` (one issue per cycle). A done 61, **loaded 77**.
- **Compute:** feed [77,141), tile done 77 + 86 = **163**.
- **Store:** LMEM reads 163..166 → responses 165..168. Writes delivered
  166..169, SRAM responses 168..171, acks delivered 169..172. **Store done
  172.**
- **Write-back:** submitted 172. SRAM reads → 174..177. DRAM bank chain:
  columns 174, 176, 178, 180, data 184–192. **T_end = 192.**

**Stall stack, PE0:**

| Cycles | Category | Count |
|---|---|---|
| [0,52) | dram (staging) | 52 |
| [52,77), burst 0 window [52,58) | network 1+1, dma 2+2 | |
| [52,77), bursts 1–19 | 1 LMEM cycle each, dma | 19 |
| [77,163) | compute | 86 |
| [163,172), burst 0 [163,169) | dma 4, network 2 | |
| [163,172), bursts 1–3 | ack cycle each, network | 3 |
| [172,192) | idle | 20 |

**Expected:** compute 86, dram 52, network 7, dma 27, sram_conflict 0,
idle 20. **Σ = 192.**

### P3-T2: grid 1×2, `sram_node` = (0,1), so PE(0,0) has H = 1 and PE(0,1) has no tile

- **Staging:** as T1, staged 52.
- **Loads, burst i:**
  - the request is delivered `issue + 3` (h + ser = 2 + 1), and the SRAM
    responds 2 later;
  - the 64 B response holds the SRAM→PE link for 4 cycles, so the link
    limits it: link start `57 + 4i`, delivered `63 + 4i`, LMEM done
    `65 + 4i`;
  - buffers never limit it (issue `≤ done_{i−8}`, checked for i ≥ 8).
  
  **Loaded 141** (burst 19).
- **Compute:** feed [141,205), tile done **227**.
- **Store:** 64 B write packets on the PE→SRAM link, starts 229, 233, 237,
  241, delivered +6. SRAM responses 237, 241, 245, 249. Acks +3 → **252.**
- **Write-back:** 252 → **T_end = 272**.

**Stall stack, PE(0,0):**
- dram [0,52) = 52.
- Loads [52,141):
  - burst 0 window [52,65): network 3+6, dma 2+2;
  - bursts 1–19 windows [61+4i, 65+4i): network 2, dma 2 each.
  - Total: network 47, dma 42.
- compute 86.
- Store [227,252):
  - burst 0 window [227,240): dma 4, network 9;
  - bursts 1–3: dma 1, network 3 each.
  - Total: network 18, dma 7.
- idle [252,272) = 20.

**Expected PE(0,0):** compute 86, dram 52, network 65, dma 49, idle 20,
sram_conflict 0. **Σ = 272.**
**Expected PE(0,1):** idle 272. **Chip Σ = 544.**

### Unit-level hand cases

- **`MeshTiming`** (h = 2, L = 16, 64 B packets, ser 4):
  - (0,0)→(2,1) at cycle 0: path +x, +x, +y, heads at 2, 4, 6, **delivered
    10**;
  - two packets (0,0)→(1,0) both at cycle 0: **6 and 10**;
  - P_a (1,0)→(2,0) is reserved before P_b (0,0)→(2,0), both at 0. P_b's
    head reaches (1,0) at 2, but the link is busy until 4: **P_b delivered
    4 + 2 + 4 = 10**;
  - H = 0: **c + 1**.
- **SRAM per-request conflict delay:** T3a (Phase 1) second request: delay
  1; T3c (intra-request): delay 0.
- **Classifier:** a synthetic trace for each rule in §3.2, each with
  hand-written expected counts. A synthetic "loaded but not fed and no
  reason" must throw.
- **Scheduler mapping:**
  - waves and chunk-sets for a 2×2 grid on 16×64 tiles, giving the expected
    staged block list (with dedup);
  - DRAM bytes formula vs hand count;
  - partial wave (1 tile, 2 PEs).

## 6. Exit-criterion runs

1. **Directed:** P3-T1 and P3-T2 (cycles and stall stacks), the unit
   cases, and all Phase 1/2 tests.
2. **Functional:** grid GEMMs bit-exact against NumPy and the C++
   reference. 100 random grid cases: g ∈ {1×1, 1×2, 2×2, 2×3}, a and Kt as
   in Phase 2, partial waves included.
3. **Sweep of ≥ 100 configurations** (`python/sweep_p3.py`, sequential).
   Grid, array, Kt, n_buf, D, W, L, h, SRAM banks, `sram_node` (corner or
   centre) and GEMM shapes are drawn from a fixed list. Every run must pass
   all §3.3 assertions. A CSV records each run's stall stack, `T_end`,
   achieved, attainable and predicted region.
4. **Roofline:** `results/phase3/roofline.png` shows the three ceilings and
   every sweep point. A table lists, for the §4.1 configs and the sweep,
   predicted region (formula) vs dominant stall category (measured).
5. **Plots:** stall stacks (stacked bars) for representative configs.

## 7. Mutation checks

| Mutant | Change | Must be caught by |
|---|---|---|
| S1 | Mesh ignores `link_free` (no link contention) | P3-T2, unit mesh contention |
| S2 | Mesh drops hop latency (`t_{l+1} = start_l`) | P3-T2, unit mesh |
| S3 | Classifier charges mesh segments as `dma` | P3-T2 stall stack |
| S4 | Classifier never charges `idle` (it falls through to `dma`) | P3-T1 and P3-T2 stall stacks |
| S5 | Classifier charges the fill/drain tail as `idle` | the compute cross-check (§3.3-3) in every run |
| S6 | No dedup: each PE's blocks staged separately | DRAM-bytes assertion (§3.3-4) |
| S7 | Ring slot freed when staging completes, not when consumers finish | grid functional check |
| S8 | Scheduler maps the PE's x to tile column (transposed mapping) | grid functional check |

## 8. Hypotheses (before measuring)

- **H1:** every run satisfies the §3.3 assertions, and achieved ≤
  attainable.
- **H2:** the dominant non-compute stall matches the predicted region:
  DRAM-bound → `dram`, NoC-bound → `network`. Compute-bound runs have
  `compute` as their largest category.
- **H3:** compute-bound §4.1 configs reach ≥ 75 % of peak at K = 1024 (fill
  and drain per tile is ≤ 46 cycles against 1024 feed cycles; the prologue
  and epilogue are short).
- **H4:** DRAM-bound configs reach ≥ 50 % of their DRAM ceiling. I am less
  sure here: DMA latency and row misses (`t_miss = 30` in the sweep) may
  keep the bus from saturating.
- **H5:** moving `sram_node` from the corner to the centre (more egress
  links) raises achieved throughput in NoC-bound configs.

Each will be reported as confirmed, refuted or partly refuted, with the
numbers.

## 9. Decisions to confirm

1. Data path DRAM → SRAM → mesh → LMEM, with the global DMA on a dedicated
   DRAM↔SRAM path (§0).
2. Mesh: XY routing, virtual cut-through with unbounded buffers, local
   delivery takes 1 cycle (§1).
3. Wave mapping (Px×Py tile blocks), chunk-set staging with dedup, an SRAM
   ring of D slots, per-PE C slots, per-step gating (§2).
4. Stall categories by data location, with the oldest unfinished burst as
   the critical burst. Synchronization waits count as `idle` (§3.1).
5. Post-hoc classifier on recorded timestamps, every branch asserted, plus
   the compute and DRAM-bytes cross-checks (§3.2–3.3).
6. Hierarchical roofline (compute, DRAM, NoC) from formulas, with the
   `achieved ≤ attainable` assertion (§4).

## 10. Implementation notes (added while building; nothing above changed)

- **Sweep memory system for the §4.1 and H5 configurations:**
  `sram_banks = 16`, global DMA `M = 32`, DRAM `Q = 32`, `t_miss = 30`.
  With `t_miss = 30` and one-beat bursts, saturating W = 64 needs about 33
  bursts in flight (Phase 2 burst lemma). With the default M = 8, the DMA
  rather than W would limit DRAM, and the DRAM ceiling would never be
  reachable. The random part of the sweep varies M ∈ {8, 32} and
  Q ∈ {8, 32} on purpose.
- **Extra cross-check in the sweep:** the simulator's busiest mesh link must
  have been busy for at least `B_max / L` cycles (the formula's load bytes
  on that link). This ties the NoC formula to the simulator in every run.
- **Two model bugs found while building:**
  - BUGS #7: PEs were woken only when a whole chunk-set was staged. The
    classifier's consistency assertion caught it.
  - BUGS #8: the DRAM rejected partial-beat requests on a wide bus. The
    first sweep run caught it.

