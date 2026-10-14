# Phase 1 Design Note — Memory System (DRAM channel, SRAM, DMA)

Status: **approved 2026-10-14 (user: "continue"); implemented**

## 0. Ground rules this note commits to

- **Time unit = 1 model cycle.** Every latency below is an integer number of
  cycles. In SystemC, 1 cycle is mapped to `sc_time(1, SC_NS)`. That mapping
  is only bookkeeping. It is not a clock frequency, and no result is ever
  reported in ns.
- **Every parameter is a modelling knob, not a datasheet value.** The values
  in §6 were chosen so the hand derivations stay small, not to resemble any
  real part.
- **Determinism.** There is no randomness and no wall-clock dependence in the
  timing path. Same-cycle ties are broken by an explicit arbitration rule
  (§3.3), never by SystemC process scheduling order.
- **Timing never decides correctness.** Phase 1 moves real bytes. Tests check
  the destination bytes against an expectation computed by a plain C++ loop
  over the descriptor, which shares no code with the model.

## 1. Scope

In Phase 1:
- `DramChannel`: one channel with B banks, per-bank open-row state, a shared
  data bus of W bytes/cycle, and row-hit/row-miss latencies.
- `Sram`: on-chip SRAM with Bs banks, each Wb bytes wide. Same-bank accesses
  serialize.
- `DmaEngine`: moves 2-D tiles DRAM→SRAM and SRAM→DRAM in bursts, using a
  bounded number of burst buffers.
- `TrafficGen`: a test-only initiator that issues a scripted list of requests.
  It lets the bank-conflict and row-hit/miss tests exercise one target
  without DMA timing mixed in.

Not in Phase 1: PEs, local memory, network, compute, stall attribution. The
counters added here (§5) are what Phase 3's stall stack will build on.

## 2. TLM coding style — decision: approximately-timed (AT) at contended targets, with analytic timing inside each target

### 2.1 Options considered

| | Loosely-timed (LT): `b_transport` + delay | Approximately-timed (AT): 4-phase `nb_transport` |
|---|---|---|
| Speed | Fastest, especially with temporal decoupling | Slower: more calls and events |
| Contention | **Wrong under temporal decoupling.** Initiators run ahead of `sc_time_stamp()`, so a target sees requests in *call order*, not *time order*, and a resource timeline gives the bank or bus to whoever called first. Without decoupling the order is right, but there is still no backpressure. | Target sees requests at their real time. `END_REQ` lets it refuse or delay acceptance (backpressure). |
| Same-cycle ties | Decided by SystemC's process order, which the LRM leaves unspecified | Target can collect all of a cycle's requests and arbitrate by an explicit policy |
| Multiple outstanding / out-of-order responses | Needs one thread per outstanding request | Native: `BEGIN_REQ`/`END_REQ` and `BEGIN_RESP`/`END_RESP` are separate events |

### 2.2 Where contention needs the AT phases

1. **DRAM channel request queue (`END_REQ`).** The channel queue has finite
   depth Q. When it is full, `END_REQ` is withheld, and under the base
   protocol the initiator cannot issue its next `BEGIN_REQ` on that socket.
   That is exactly backpressure. LT has no way to say "not yet".
2. **DRAM data bus and banks (`BEGIN_RESP` timing).** Several outstanding
   requests from several initiators share one bus. The response time of each
   depends on everything accepted before it, so responses are decoupled from
   requests and may complete in any order across initiators.
3. **SRAM banks (arbitration point).** Two initiators hitting the same bank
   in the same cycle must serialize by a defined rule. The AT target collects
   that cycle's `BEGIN_REQ`s and arbitrates explicitly (§3.3).
4. **DMA burst pipelining.** The DMA keeps up to M bursts in flight. AT
   expresses this directly. LT would need M threads.

### 2.3 Where AT is *not* used

- Descriptor submission to the DMA is a plain C++ call made at a given
  cycle (`submit(desc)`). It is configuration, not contended traffic.
- Test setup and checking use `transport_dbg` (backdoor, zero time) to fill
  and read memories. It never affects timing.

### 2.4 Analytic timing inside targets (and its cost)

When a target accepts a request, it computes the request's whole schedule
with integer formulas over its resource state: bank ready time, open row,
bus-free time (§3). The AT phases then deliver the response at the computed
cycle. **Why:** every cycle count can be derived by hand from §3, and the
pure timing classes (`DramTiming`, `SramTiming`) are unit-testable without
SystemC. **Cost:** decisions are final at acceptance, so the model is
**in-order FCFS per target**. It cannot reorder for row hits (FR-FCFS) or
cancel a scheduled access. This goes in LIMITATIONS.md. Reordering would
require deferring the decision until the bus is free, which this
structure allows later without changing the socket interface.

### 2.5 SC_METHOD-only processes

All model processes are `SC_METHOD`s (state machines driven by PEQ events),
with no `SC_THREAD`s. Reasons: (a) no coroutine stack switching, so ASan and
UBSan runs are reliable (ASan has known false positives on SystemC's
QuickThreads); (b) faster; (c) no hidden blocking. The cost is that the DMA
reads as an explicit state machine, not straight-line code.

## 3. Components, owned state, timing rules

### 3.1 Cycle structure and sample point

Each cycle c spans `[c, c+1)` ns. Initiators only **send** `BEGIN_REQ` at a
cycle boundary c. Each target **samples** its pending requests once per
cycle, at `c + 0.5` ns, and arbitrates them there. A request sampled at
`c + 0.5` has **accept cycle a = c**. Every request sent at cycle c is
visible at the sample point, whatever delta cycle it arrived in, so the
outcome is independent of process order. This works like a clocked arbiter
sampling its request lines. Responses (`BEGIN_RESP`) are sent at cycle
boundaries. The sample point is internal: no reported number is ever a
half cycle.

Base-protocol consequences we rely on:
- `END_REQ` is sent at the sample point, so an initiator can send its next
  request on that socket at **c + 1 at the earliest**: at most one request
  per socket per cycle.
- Initiators return `TLM_COMPLETED` to `BEGIN_RESP` (immediate `END_RESP`),
  so responses are never blocked.

### 3.2 Address maps and legality

> **Changed in Phase 3** (docs/BUGS.md #8): a DRAM request need only be
> 16-byte aligned and a multiple of 16 bytes; it occupies every W-byte bus
> beat it touches (D = beats touched). For W = 16, the value used by every
> Phase 1/2 test, this is identical to the rule below.

- **DRAM:** `col = addr % R`, `bank = (addr / R) % B`, `row = addr / (R·B)`.
  Consecutive R-byte rows rotate across banks. A request must be a multiple
  of W bytes, at most `burst_bytes`, and must not cross a row.
- **SRAM:** `line = addr / Wb`, `bank = line % Bs`. A request must be
  Wb-aligned and a multiple of Wb. It may span many lines, including several
  lines in the same bank.
- An illegal request is a modelling bug and terminates with an error, never
  a silent fix-up.

### 3.3 Arbitration (same cycle, same target)

Each target keeps a round-robin pointer `p` (initially 0) over its
initiator ports. At a sample point, the pending requests are granted in port
order starting at `p` and wrapping. Then `p = (first granted port) + 1 mod N`.
"Granted" means the request is accepted and its schedule computed. That
order is the order in which banks and the bus are reserved.

### 3.4 `DramChannel`

| Owns | Meaning |
|---|---|
| `open_row[b]` | Row open in bank b, or *none* (initially none) |
| `bank_ready[b]` | First cycle bank b can start a new access |
| `bus_free` | First cycle the data bus is free |
| queue occupancy | Requests accepted but not yet responded (≤ Q) |
| backing store | Bytes |

Parameters: `B, R, W, t_hit, t_miss (≥ t_hit), Q, burst_bytes`.

For a request of S bytes to bank b, row r, accepted at cycle a, with
D = S / W:

```
start      = max(a, bank_ready[b])
hit        = (open_row[b] == r)
col_issue  = start + (hit ? 0 : t_miss - t_hit)   # activation, if needed
data_start = max(col_issue + t_hit, bus_free)
data_end   = data_start + D                        # response sent at data_end
bus_free   = data_end
bank_ready[b] = col_issue + D                      # next column command to this bank
open_row[b]   = r
```

So a row hit costs `t_hit + D` unloaded and a row miss `t_miss + D`. A
stream of hits to an open row, with enough requests outstanding, puts data
on the bus back to back (peak W bytes/cycle). An empty bank counts as a miss
(simplification: no separate "closed" latency). Reads and writes have the
same timing (no read/write turnaround).

Queue: a request holds a slot from acceptance to its response cycle. When
all Q slots are held at a sample point, requests stay pending, and no
`END_REQ` is sent until a later sample point. A slot released by a response
at cycle c is available at the sample point of cycle c.

Data: the backing store is read or written when the request is accepted,
in grant order. FCFS makes that equivalent to doing it at the response.

### 3.5 `Sram`

| Owns | Meaning |
|---|---|
| `bank_free[k]` | First cycle bank k can serve a line |
| backing store | Bytes |

Parameters: `Bs, Wb, t_sram`. There is no request queue limit: the SRAM
accepts every request at its sample point (`END_REQ` then).

For a request covering lines `l_0 .. l_{n-1}`, accepted at cycle a, in
order:

```
c_i = max(a, bank_free[bank(l_i)]);  bank_free[bank(l_i)] = c_i + 1
response cycle = max_i(c_i) + t_sram
```

Each bank serves one line per cycle. Lines of the same request, or of
different requests, that land in one bank serialize. That is the bank
conflict.

### 3.6 `DmaEngine`

> **Changed in Phase 2** (docs/phase2-design.md §1): bursts now stream across
> descriptors instead of one descriptor at a time, and request-socket
> occupancy is tracked per socket (write first). Every Phase 1 scenario uses
> one descriptor per engine, so the expected values below are unchanged.

| Owns | Meaning |
|---|---|
| descriptor queue | Submitted tile descriptors (FIFO, processed one at a time) |
| burst list | Bursts of the current descriptor, in address order |
| M burst buffers | A buffer is held from **read issue** until **write response** |
| per-socket request state | Waiting for `END_REQ` or not |

Descriptor: `{dir (DRAM→SRAM | SRAM→DRAM), src, dst, rows, row_bytes,
src_stride, dst_stride}`. Each row is split into bursts of up to
`burst_bytes` that never cross a DRAM row (DRAM addresses are
`burst_bytes`-aligned in all Phase 1 tests).

Payloads are owned by the DMA (one read and one write payload per burst); no
memory manager is needed because no target keeps a payload after the
response.

Behaviour, per cycle boundary:
1. If a buffer is free, the read socket is idle, and bursts remain: send the
   next read `BEGIN_REQ`.
2. When a read response arrives at cycle r, send that burst's write at
   cycle r, or as soon as the write socket is idle, in read-response order.
3. When a write response arrives, free the buffer.
4. The descriptor **completes at the cycle of its last write response**. That
   cycle is the reported DMA latency (submitted at cycle 0).

Parameters: `burst_bytes, M`.

### 3.7 `TrafficGen` (test-only)

Issues a scripted list `(cycle, read|write, addr, size)`. Each request is
sent at `max(scripted cycle, first cycle the socket is idle)`. It records the
response cycle of each request.

## 4. Code layout

```
src/common/cycle.h            cycle <-> sc_time helpers, sample-point helper
src/common/rr_arbiter.h       round-robin grant order (pure C++)
src/mem/sampled_target.{h,cpp} common AT target: per-port pending request, mid-cycle sample, response PEQ, backing store
src/mem/dram_timing.{h,cpp}   §3.4 formulas, no SystemC
src/mem/sram_timing.{h,cpp}   §3.5 formulas, no SystemC
src/mem/dram_channel.{h,cpp}  TLM target wrapping DramTiming + queue + store
src/mem/sram.{h,cpp}          TLM target wrapping SramTiming + store
src/dma/dma_engine.{h,cpp}    initiator state machine
src/test/traffic_gen.{h,cpp}  scripted initiator
src/test/scenarios.{h,cpp}    builds each directed scenario; functional check loop (independent of the DMA)
tools/apm_scenario.cpp        runs a named directed scenario, prints JSON (cycles + counters)
tests/unit/                   GoogleTest on the pure timing classes (in-process)
tests/sim/                    GoogleTest on the SystemC scenarios
```

**One SystemC simulation per process.** SystemC cannot be re-elaborated
inside one process. The SystemC tests are registered with
`gtest_discover_tests`, which runs each test in its own process, and the
test `main` refuses to run more than one test per process. The **determinism
test** runs `apm_scenario two_dma` twice in separate processes and requires
byte-identical JSON.

The same hand-derived numbers are asserted at two levels. The unit level
checks the formulas. The SystemC level checks that the protocol plumbing
(sample points, `END_REQ`, PEQs, the DMA state machine) adds or loses no
cycles.

## 5. Counters (Phase 1)

- DRAM: `row_hits`, `row_misses`, `bus_busy_cycles`, `bytes_read`,
  `bytes_written`, `backpressure_cycles` (sum over sample points of the
  number of requests left pending, i.e. request-cycles of refused
  acceptance).
- SRAM: `lines_accessed`, `conflict_line_cycles` (Σ over lines of
  `c_i − a − (position of that line within its bank in an otherwise idle
  SRAM)`, which is 0 when no other request is in the way).
- DMA: per descriptor `submit_cycle`, `done_cycle`, `bursts`, `bytes`.

## 6. Directed tests, with expected cycles derived by hand **before running**

Default test configuration (every test uses it unless stated otherwise):

| | |
|---|---|
| DRAM | B=4 banks, R=1024 B rows, W=16 B/cycle, t_hit=10, t_miss=30, Q=8, burst_bytes=64 → D=4 for a 64 B burst |
| SRAM | Bs=8 banks, Wb=16 B, t_sram=2 |
| DMA | burst_bytes=64, M=4 |

The arbitration pointer starts at 0. Initiator ports are numbered in the
order given in each test.

### T1 — Single transfer: DMA, 256 B, DRAM[0] → SRAM[0], submitted at cycle 0

Bursts at 0, 64, 128, 192: all in bank 0, row 0 (closed → first is a miss).
Read requests are issued at cycles 0, 1, 2, 3 (M=4 does not bind).

| Burst | a | start | col_issue | data_start | data_end (read resp) | SRAM lines → banks | SRAM resp |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 20 (miss) | max(30, 0)=30 | 34 | 0–3 → 0–3 @34 | 36 |
| 1 | 1 | max(1,24)=24 | 24 | max(34,34)=34 | 38 | 4–7 → 4–7 @38 | 40 |
| 2 | 2 | 28 | 28 | max(38,38)=38 | 42 | 8–11 → 0–3 @42 | 44 |
| 3 | 3 | 32 | 32 | max(42,42)=42 | 46 | 12–15 → 4–7 @46 | 48 |

**Expected: done at cycle 48.** Closed form: `t_miss + n·D + t_sram =
30 + 16 + 2`. Counters: DRAM `row_misses=1`, `row_hits=3`,
`bus_busy_cycles=16`; SRAM `conflict_line_cycles=0`.

### T1b — Same transfer with M=1 (latency-bound)

A buffer is held until the write response, so each read waits for the
previous burst's write.
Burst 0: read 0→34, write →36. Burst 1: read at 36, hit: data 46–50, write
→52. Burst 2: read 52 → 66, write →68. Burst 3: read 68 → 82, write →84.
**Expected: done at cycle 84** = `(t_miss − t_hit) + 4·(t_hit + D + t_sram)
= 20 + 64`.

### T1c — Reverse direction: DMA, 256 B, SRAM[0] → DRAM[0], M=4

SRAM reads at 0, 1, 2, 3, on banks 0–3, 4–7, 0–3, 4–7 (the banks were
freed at cycles 1 and 2) → responses 2, 3, 4, 5. DRAM writes are accepted
at 2, 3, 4, 5:
w0: start 2, col 22 (miss), data 32–36. w1: start max(3, 26)=26, data
max(36,36)=36–40. w2: start 30, data 40–44. w3: start 34, data 44–48.
**Expected: done at cycle 48** = `t_sram + t_miss + 4·D`.

### T2 — Two overlapping DMAs on one channel

DMA0 (port 0): 256 B DRAM[0] → SRAM[0] (bank 0, row 0).
DMA1 (port 1): 256 B DRAM[1024] → SRAM[4096] (bank 1, row 0; SRAM line
256 → bank 0).
Both are submitted at cycle 0. Both issue reads at 0, 1, 2, 3. RR order per
cycle: c0 [0,1], c1 [1,0], c2 [0,1], c3 [1,0].

| Order | Req | start | col_issue | data_start | data_end |
|---|---|---|---|---|---|
| c0 #1 | D0 b0 | 0 | 20 | max(30,0)=30 | 34 |
| c0 #2 | D1 b0 | 0 | 20 | max(30,34)=34 | 38 |
| c1 #1 | D1 b1 | 24 | 24 | max(34,38)=38 | 42 |
| c1 #2 | D0 b1 | 24 | 24 | max(34,42)=42 | 46 |
| c2 #1 | D0 b2 | 28 | 28 | max(38,46)=46 | 50 |
| c2 #2 | D1 b2 | 28 | 28 | max(38,50)=50 | 54 |
| c3 #1 | D1 b3 | 32 | 32 | max(42,54)=54 | 58 |
| c3 #2 | D0 b3 | 32 | 32 | max(42,58)=58 | 62 |

SRAM writes fall at 34, 38, 42, 46, 50, 54, 58, 62: never two in the same
cycle, so there are no SRAM conflicts.
**Expected: DMA1 done at 60, DMA0 done at 64.** The bus is busy without a
gap from 30 to 62 (8 bursts × 4 = 32 cycles): the channel is saturated. The
last DMA finishes at `t_miss + 8·D + t_sram = 64`, versus 48 for one DMA
alone. Which DMA finishes first is set by the RR policy, and the test
asserts both values.

### T2b — Backpressure: T1 with DRAM queue Q=2

b0 and b1 are accepted at 0 and 1 (data 30–34, 34–38). b2 is sent at 2 but
stays pending until b0's slot frees at 34: a=34, start max(34, 28)=34, hit,
data max(44, 38)=44–48. b3 is sent at 35 (after b2's `END_REQ` at 34.5) and
accepted when b1 frees at 38: data max(48, 48)=48–52. Writes respond at
36, 40, 50, 54.
**Expected: done at cycle 54** (vs 48 with Q=8).
DRAM `backpressure_cycles = 32 + 3 = 35` (b2 pending in cycles 2–33, b3 in
cycles 35–37).

### T3 — SRAM bank conflict (two TrafficGens on the SRAM, ports 0 and 1)

- **T3a, conflict:** TG0 writes 16 B @0 (bank 0) at cycle 0. TG1 writes 16 B
  @128 (line 8 → bank 0) at cycle 0. TG0's line is served at 0, so bank 0 is
  free at 1. TG1's line is served at 1.
  **Expected: TG0 response 2, TG1 response 3.** `conflict_line_cycles=1`.
- **T3b, no conflict:** same, but TG1 writes @16 (bank 1).
  **Expected: both respond at 2.** `conflict_line_cycles=0`.
- **T3c, intra-request:** TG0 alone writes 256 B @0: lines 0–15 → each
  bank twice. Lines 0–7 are served at 0, lines 8–15 at 1.
  **Expected: response 3.** `conflict_line_cycles=0`, because serialization
  within one request is not counted as a conflict (§5).

### T4 — Row hit vs row miss (one TrafficGen on DRAM, 64 B reads)

- **T4a hit:** read @0 at 0 (miss) → response 34. Read @64 at 100 (bank 0,
  row 0 open) → col 100, data 110–114. **Expected: 34 and 114** (latencies
  34 = t_miss+D and 14 = t_hit+D).
- **T4b miss:** read @0 at 0 → 34. Read @4096 at 100 (bank
  (4096/1024)%4 = 0, row 1) → col 120, data 130–134.
  **Expected: 34 and 134** (latency 34).
- **T4c same-bank conflict, back to back:** read @0 at 0, read @4096 issued
  at 1. Start max(1, 24)=24, col 44, data max(54, 34)=54–58.
  **Expected: 34 and 58.**
- **T4d different bank, back to back:** read @0 at 0, read @1024 (bank 1,
  closed) at 1. Start 1, col 21, data max(31, 34)=34–38.
  **Expected: 34 and 38.** Comparing with T4c shows bank-level parallelism
  hiding the activation.

### Counters asserted alongside (read directly off the tables above)

T1: 4 bursts. T1c: `dram.bytes_written = 256`. T2: `row_misses = 2` (one
first burst per bank), `row_hits = 6`. T4a: `row_hits = 1`. T4b:
`row_misses = 2`. 2-D tests: 8 bursts (4 rows × 128 B / 64 B).

### Functional tests (data only, no timing asserted)

- T1, T1c and T2 also check the destination bytes against the reference
  loop.
- A 2-D strided tile (rows=4, row_bytes=128, src_stride=1024,
  dst_stride=256), in both directions: bytes checked. There is no timing
  assertion, because it adds no new timing rule.

### Determinism

`apm_scenario` runs T2 twice in separate processes, and the JSON (all
cycles and counters) must be byte-identical.

## 7. Mutation checks (can the tests fail?)

`scripts/phase1_mutation_check.sh` copies the tracked tree to a temp
directory, injects one mutant at a time, rebuilds, and records which tests
fail:

| Mutant | Change | Must be caught by |
|---|---|---|
| M1 | Row hit uses `t_miss` (no row-buffer benefit) | T1, T4a |
| M2 | SRAM ignores `bank_free` (no bank conflicts) | T3a |
| M3 | DRAM ignores `bus_free` (bus not shared) | T2, T4d (not T1: there the bus never binds, see T1 table) |
| M4 | DRAM queue never refuses (`Q` ignored) | T2b |
| M5 | Arbiter always grants port 0 first (no RR) | T2 |
| M6 | DMA frees a buffer at read response instead of write response | T1b |
| M7 | DRAM ignores `bank_ready` (no per-bank occupancy) | T4c |
| M8 | Targets drop write data (timing unchanged) | T1, T1c, both 2-D tests (data check) |

A mutant that no test catches is a gap in the tests, and it gets a new test
before Phase 1 is called done.

## 8. Build, environment, provenance

- `.devcontainer/` on Ubuntu 24.04 with one `setup.sh`: apt (build-essential,
  cmake, ninja, python3 + numpy + matplotlib, verilator), and **SystemC
  3.0.2 built from source at commit `70b0fc8e4a74acc677b0fc73cea08f940c2115d5`**,
  installed to `/opt/systemc-3.0.2` (C++17). GoogleTest is fetched by CMake
  `FetchContent`, pinned to a commit hash.
- `CMakePresets.json`: `release` (`-O3 -DNDEBUG`, used for results) and
  `asan` (ASan + UBSan, `-O1 -g`, used for correctness runs).
- `scripts/env_report.sh`: date, commit, clean/dirty tree, host/CPU/memory,
  and the versions of the compiler, cmake, ninja, SystemC (from `sc_version()`
  and the pinned commit stamp written by `setup.sh`), GoogleTest commit,
  python and numpy, and verilator.
- `scripts/phase1_exit.sh`: captures provenance first, builds both presets
  from scratch, runs ctest, and writes everything, including a table of
  *expected vs measured* cycles for every T-test, into a `mktemp -d`
  directory **outside the repo**. Only at the end does it move the results
  into `results/phase1/`, so provenance cannot read `tree: DIRTY` (rv32im
  bugs #2 and #7).
- Every SystemC test prints `cycles: <test> expected=<n> measured=<n>`, and
  the exit script collects those lines.

## 9. Hypotheses written before measuring

- H1: All the expected values in §6 hold exactly under both presets. Any
  deviation is either a model bug or a derivation error. Each one is
  root-caused and recorded in BUGS.md, and the hand derivation is never
  edited to match the simulator without a written reason.
- H2: For a long open-row stream with enough buffers, DMA latency is
  `t_miss + n·D + t_sram`. The bus is the only limit (T1).
- H3: With M=1, latency is `(t_miss−t_hit) + n·(t_hit + D + t_sram)`
  (T1b). Single outstanding burst = latency-bound.

## 10. Decisions to confirm

1. AT at contended targets, with analytic (formula) timing inside targets,
   which means in-order FCFS (§2.4).
2. Fixed sample point at mid-cycle, plus round-robin arbitration (§3.1, §3.3).
3. Empty bank = row miss (no separate "closed" latency). Same timing for
   reads and writes.
4. DMA buffers held from read issue to write response (bounded buffering).
5. SC_METHOD-only processes; one simulation per test process.
6. SystemC 3.0.2 (commit `70b0fc8`) from source.
