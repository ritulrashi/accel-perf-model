// PE configuration and LMEM buffer planning (Phase 2 design note §2.4). Pure
// C++ so the capacity checks are unit-testable without building a simulation.
#pragma once
#include <cstdint>
#include <vector>

#include "compute/systolic.h"
#include "dma/dma_types.h"
#include "mem/sram_timing.h"
#include "pe/gemm_layout.h"

namespace apm {

struct PeConfig {
  ArrayShape array;
  std::uint64_t kt = 64;
  unsigned n_buf = 2;  // operand (A+B) buffer sets
  unsigned n_out = 2;  // output (C tile) buffers
  SramParams lmem;
  DmaParams dma;
  // Phase 4 (design note §1): operand precision and cycles per reduction
  // step (0 = the mode's default, q = e).
  Precision prec = Precision::Int8;
  unsigned q = 0;

  unsigned elem() const { return elem_bytes(prec); }
  unsigned cycles_per_step() const { return q ? q : default_q(prec); }
  // Feed length of one chunk: q*Kt (Kt for INT8).
  Cycle feed_cycles() const { return cycles_per_step() * kt; }
  std::uint64_t a_block_bytes() const { return std::uint64_t{array.rows} * kt * elem(); }
  std::uint64_t b_block_bytes() const { return kt * array.cols * elem(); }

  // Bytes of LMEM the buffers need (§2.4a; Phase 4 §1.3 with e-byte operands).
  std::uint64_t lmem_required() const {
    return n_buf * (a_block_bytes() + b_block_bytes()) + n_out * 4ull * array.rows * array.cols;
  }
};

// LMEM addresses of every buffer.
struct PeBuffers {
  std::vector<std::uint64_t> a, b, out;
};

// Places the PE's LMEM buffers. Throws std::invalid_argument with a message
// naming the problem:
//   layer (a) total LMEM requirement vs capacity,
//   layer (b) each buffer through the line-aligned bump allocator.
PeBuffers plan_pe_buffers(const PeConfig& cfg);

// Phase 2 single-PE GEMM: validates the job against the configuration, then
// plan_pe_buffers().
PeBuffers plan_pe(const PeConfig& cfg, const GemmLayout& job);

}  // namespace apm
