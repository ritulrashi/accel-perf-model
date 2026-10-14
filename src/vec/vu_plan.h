// Vector-unit configuration, job description and VMEM plan (Phase 4 design
// note §3). Pure C++.
//
// The VU processes the rows of a row-major S (rows x n, 4-byte words at
// s_base). Row r is loaded into input buffer r mod n_vbuf, computed in
// T_row = 3 * (ceil(n / V) + t_vu) cycles, and its P row is stored directly
// in the packed A layout of the next GEMM (Phase 2 §3.1, R rows x Kt per
// block): segment c of row r goes to p_base + ((mi*nc + c)*R + i)*Kt*e with
// mi = r / R, i = r mod R, nc = n / Kt. INT8 with per-row P scales also
// writes the rows' FP32 scales (one descriptor after the last row).
#pragma once
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"
#include "dma/dma_types.h"
#include "mem/sram_timing.h"
#include "vec/softmax.h"

namespace apm {

struct VuConfig {
  unsigned lanes = 16;  // V
  Cycle t_vu = 4;       // per-pass pipeline + reduction latency
  unsigned n_vbuf = 2, n_vout = 2;
  SramParams vmem;
  DmaParams dma;
};

struct VuJob {
  std::uint64_t rows = 1, n = 64;
  SoftmaxSpec spec;  // spec.lanes is taken from VuConfig::lanes
  std::uint64_t s_base = 0;
  std::uint64_t p_base = 0;  // packed A layout of the next GEMM
  unsigned r_pack = 4;       // R of the next GEMM
  std::uint64_t kt = 64;     // Kt of the next GEMM
  std::uint64_t scale_base = 0;  // INT8 with PScale::Row only

  unsigned elem() const { return elem_bytes(spec.prec); }
  std::uint64_t chunks() const { return n / kt; }
  bool writes_scales() const { return spec.prec == Precision::Int8 && spec.p_scale == PScale::Row; }
  std::uint64_t scale_bytes() const { return writes_scales() ? (4 * rows + 15) / 16 * 16 : 0; }
  // Bytes the packed P region occupies (rows padded to a multiple of R).
  std::uint64_t p_region_bytes() const { return (rows + r_pack - 1) / r_pack * r_pack * n * elem(); }
};

// T_row(n) = 3 * (ceil(n / V) + t_vu)  (§3.2).
inline Cycle row_cycles(const VuConfig& c, std::uint64_t n) {
  return 3 * ((n + c.lanes - 1) / c.lanes + c.t_vu);
}

struct VuBuffers {
  std::vector<std::uint64_t> in, out;
  std::uint64_t scales = 0;
};

// VMEM requirement: n_vbuf * 4n + n_vout * n * e (+ scale buffer).
std::uint64_t vmem_required(const VuConfig& c, const VuJob& j);
// Validates the job and places the VMEM buffers. Throws std::invalid_argument.
VuBuffers plan_vu(const VuConfig& c, const VuJob& j);

TileDesc vu_load(const VuJob& j, std::uint64_t vmem_dst, std::uint64_t row);
TileDesc vu_store(const VuJob& j, std::uint64_t vmem_src, std::uint64_t row);
TileDesc vu_scale_store(const VuJob& j, std::uint64_t vmem_src);
// Bytes moved to/from DRAM: rows * n * (4 + e) + scale bytes.
std::uint64_t vu_dram_bytes(const VuJob& j);

}  // namespace apm
