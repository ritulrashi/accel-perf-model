# Limitations

What this project does **not** model or verify. Updated every phase.

## Scope of the whole project
- **This is not a model of any real chip.** The architecture is loosely
  inspired by publicly described inference accelerators (a PE grid with
  per-PE local memory, a shared on-chip SRAM, off-chip DRAM, an on-chip
  network). Every latency, bandwidth and size is a configurable modelling
  parameter. None is a datasheet value, and no result says anything about
  the performance of a real product.
- **Time is in model cycles only.** The SystemC time base (1 cycle = 1 ns
  of simulated time) is bookkeeping. It is not a clock frequency.

## Memory system (Phase 1)
- **DRAM is one channel with in-order FCFS scheduling.** Each request's
  schedule is fixed when it is accepted. There is no FR-FCFS (row-hit-first)
  reordering, no request cancellation and no write buffering. A real
  controller would reorder to raise the row-hit rate (design note §2.4).
- **DRAM timing is two numbers: row hit and row miss.** An empty bank
  counts as a miss. There is no precharge/activate split (tRP, tRCD, tRAS,
  tRC), no tFAW/tRRD activation limits, no refresh, no read/write bus
  turnaround, and no power-down. Reads and writes have identical timing.
- **Fixed address map** (row-interleaved across banks: `bank = (addr / row) %
  banks`). Other maps (XOR bank hashing, channel interleaving) are not
  modelled.
- **Bank occupancy is simplified:** after a column command, a bank accepts
  its next command D cycles later (D = burst length on the bus). A miss
  that follows to the same bank can therefore start activating a new row
  while the previous burst's data is still on the bus. This is optimistic
  compared with real precharge rules.
- **SRAM has no port or crossbar limit.** Each bank serves one line per
  cycle, and any number of initiators can be served in the same cycle on
  different banks. Read and write latency are the same. There is no queue
  limit, so the SRAM never applies backpressure.
- **Requests must be aligned:** DRAM requests are bus-aligned, at most one
  burst, and never cross a row. SRAM requests are line-aligned. Illegal
  requests stop the simulation with an error. The DMA only generates legal
  ones.
- **DMA:** store-and-forward per burst, in-order reads and in-order writes,
  no address translation, no scatter/gather beyond 2-D strides. (Phase 1
  processed one descriptor at a time. Since Phase 2, bursts stream across
  descriptors; see the Phase 2 note §1.)
- **Same-cycle arbitration is round-robin by port number**, and port number
  is binding order. Other policies (age-based, QoS) are not modelled.
- **Directed tests only.** Phase 1 checks hand-derived cycle counts on a
  dozen small scenarios plus data correctness. It does not show that the
  timing is realistic, only that the model does what the design note says.

## Compute unit, local memory, double buffering (Phase 2)
- **Integer mode only:** INT8 inputs, INT32 accumulation. Floating-point
  precisions arrive in Phase 4.
- **One accumulator set (output-stationary).** Fill (R + C − 2) and drain (R)
  are exposed once per output tile. Designs with double-buffered
  accumulators overlap a tile's drain with the next tile's fill. Not
  modelled.
- **The array's operand/result port to LMEM is ideal:** R + C bytes in and
  4C bytes out per cycle, never conflicting with DMA traffic. Only the DMA
  side of LMEM has bank timing.
- **The functional data path applies a whole chunk at feed start** and
  writes the tile at drain end. Values are therefore exact, but there is no
  per-cycle data model of the array. The per-cycle contract (§2.2) is
  checked against RTL in Phase 5.
- **Host pre-packing is assumed and costs nothing.** Operands sit in DRAM in
  tile order. Packing time and the extra DRAM copy are not modelled.
- **Naive loop order:** B blocks are reloaded for every output-tile row, with
  no operand reuse across tiles. This is deliberate as a baseline. Better
  schedules come in Phase 3.
- **Controller decisions take zero cycles:** a feed can start in the cycle
  its last load completes. Real control logic would add a few cycles.
- **Output-buffer reuse is checked at feed start** (conservative): tile t
  waits for the store of tile t − n_out before its first chunk feeds, even
  though the buffer is only written at drain.
- **Shapes:** M % R = 0, N % C = 0, K % Kt = 0, Kt % 16 = 0, C % 4 = 0, and
  K ≤ 131071 (INT32 range). There is no edge-tile (padding) support.
- **Functional coverage:** 200 random cases over arrays R ∈ {1,2,4,8} ×
  C ∈ {16,32}, Kt ∈ {16,32,64}, small M/N/K, and both buffering modes. This
  is good evidence for these shapes, not a proof for all shapes.

## Accelerator grid, stall stack, roofline (Phase 3)
- **Mesh:** XY routing only, with virtual cut-through and unbounded router
  buffers. A blocked packet never holds an upstream link, so there is no
  wormhole blocking, no credit backpressure and no deadlock modelling.
  - **Simplified latency:** fixed per-hop latency, no router pipeline
    stages, and local delivery (H = 0) costs one cycle with no bandwidth
    limit.
  - **Traffic:** control packets (read requests, write acks) occupy a link
    for one cycle; nothing else (headers, routing info) is modelled.
- **One SRAM channel per PE.** Each PE has its own port into the shared
  SRAM, so the SRAM never serializes requests by port, only by bank. A
  real design would funnel them through fewer network-interface ports.
- **The global DMA has a dedicated DRAM↔SRAM path** outside the mesh, and
  the shared SRAM sits at a single mesh node.
- **The scheduler is static and simple:** waves of Px×Py output tiles,
  chunk-sets staged in order, one ring of D slots, per-step gating on the
  step's own two blocks.
  - **Not modelled:** reordering across waves, prefetch heuristics, B-block
    reuse across waves, and multicast of a shared block to several PEs.
    Each PE fetches its own copy over the mesh, which is why the NoC
    ceiling binds so often.
- **The stall stack is post-hoc and location-based.**
  - **Location:** a waiting cycle is charged to where the critical data is
    (the oldest unfinished burst), not to the "root cause" of the wait. For
    example, a burst queued behind another PE's traffic in the mesh counts
    as `network` even if the SRAM was the reason the earlier packet was
    late.
  - **Synchronization:** waits for the ring to free count as `idle`.
  - **Conflicts:** `sram_conflict` covers only the extra delay caused by
    other requests, placed at the start of the residency by convention.
- **Roofline ceilings are formulas.**
  - **NoC ceiling:** counts operand-load bytes on the busiest link. Control
    packets and the write traffic on reverse links are left out, so it is
    an upper bound, not a prediction.
  - **No on-chip SRAM ceiling:** the shared SRAM's bank bandwidth is not
    one of the ceilings. The sweep's §4.1 configurations use 16 banks so
    that it does not bind.
- **Sweep coverage:** 115 configurations, with grids up to 3×3 and arrays
  up to 16×16. The results describe this model at these sizes, not real
  accelerators.


## Precision modes, vector unit, attention (Phase 4)
- **Precision timing is one knob:** an e-byte mode takes `q = e` cycles per
  reduction step (a fixed number of multiplier bits per cell). Real arrays
  have mode-specific datapaths. FP8, TF32 and mixed-precision operand pairs
  are not modelled.
- **Accumulators and GEMM outputs are always 4 bytes** (INT32 or FP32). S is
  written to DRAM in FP32/INT32 even when P is FP16 or INT8.
- **Floating point is bit-exact to a spec, not to any hardware.**
  - FP32 accumulation is sequential in k, with no FMA and no wider
    accumulator.
  - `vu_exp` is one specified degree-6 polynomial that flushes results below
    2^−125 to zero.
  - NaN and infinity inputs are rejected or never generated, and overflow is
    impossible by construction.
- **Vector unit:**
  - one unit on its own DRAM port, not on the mesh and not using the shared
    SRAM;
  - rows are processed one at a time in three dependent passes, with no
    overlap between rows;
  - masked elements cost the same as unmasked ones;
  - the VU↔VMEM port is ideal.
- **Programs are unfused and serialized.**
  - Ops never overlap: the array idles during softmax, and the VU idles
    during GEMMs.
  - Each op starts as a separate simulation with all DRAM banks closed, so
    open rows do not carry across op boundaries.
  - There is no FlashAttention-style fusion. S and P make a full round trip
    through DRAM.
- **No causal tile skipping:** prefill computes all of S and all of P·V,
  including the masked upper triangle (about half the work).
- **Layouts are assumed, not produced.**
  - The KV cache is stored in the array's packed B layouts, and Q is
    host-packed, both at no cost. The Q/K/V/O projections and the KV-cache
    append are not modelled.
  - INT8 Q/K/V scales are computed from the data itself (ideal calibration).
  - INT8 O is dequantized on the host at no cost.
- **Shapes and padding:**
  - only M is padded (to R; the padding is computed and charged as compute);
  - N and K must divide C and Kt;
  - g ≤ R.
- **Scope of the workloads:** one KV head with g query heads, head dim 64,
  sequence lengths up to 1024 (prefill) and 8192 (decode). Multi-head is
  repetition and is not run.
- **Numerics use N(0,1) random inputs.** Real activations have outliers and
  structure (attention sinks, peaked rows). INT8 accuracy on real models can
  be far worse or better than measured here.

## RTL correlation, bank map, sweeps (Phase 5)
- **The RTL covers only the compute array:** INT8, q = 1, output-stationary,
  one accumulator set. The PE controller, DMA, LMEM, mesh and DRAM have no
  RTL. The array is driven by the schedules the model's controllers
  produced, so correlation shows that the array honours the contract for
  those schedules. It does not show that the controller's timing is right
  (that rests on the Phase 2–4 hand derivations).
- **Simulated, not synthesized:** there is no timing closure, area, power or
  frequency. "Correlation" means cycle-behaviour equivalence (tile timing,
  values, per-cell MAC per cycle) and nothing more.
- **Coverage:** 14 array shapes up to 16×16 (8×32), with schedules from the
  Phase 2/3 random GEMMs and four A4 attention GEMM ops. Bubble patterns are
  only those the model's controllers generate. Only two schedules (HT1, HT2)
  are hand-written.
- **Same author for contract, model and RTL.** The hand traces and the
  Phase 2 hand-derived done cycles are the only checks independent of both
  implementations.
- **DRAM:** the XOR bank map is the only controller-side remedy modelled,
  and it needs a power-of-two bank count. The controller is still in-order
  FCFS, with no reordering (FR-FCFS), no write buffering and no bank-aware
  scheduling.
- **Sweeps:** each point is a single configuration (A4 timing parameters,
  W = 64, mesh L = 64, SRAM at the corner, 1024 MACs in S2). The results rank
  arrangements of this model under these parameters, not real designs.
