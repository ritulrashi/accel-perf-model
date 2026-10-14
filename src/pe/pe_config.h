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

  // Bytes of LMEM the buffers need (§2.4a).
  std::uint64_t lmem_required() const {
    return n_buf * (std::uint64_t{array.rows} * kt + kt * array.cols) +
           n_out * 4ull * array.rows * array.cols;
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
