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
- **DMA:** one descriptor at a time per engine (no overlap between
  descriptors), store-and-forward per burst, in-order writes, no address
  translation, no scatter/gather beyond 2-D strides.
- **Same-cycle arbitration is round-robin by port number**, and port number
  is binding order. Other policies (age-based, QoS) are not modelled.
- **Directed tests only.** Phase 1 checks hand-derived cycle counts on a
  dozen small scenarios plus data correctness. It does not show that the
  timing is realistic, only that the model does what the design note says.
