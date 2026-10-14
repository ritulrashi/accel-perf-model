// Runs one GEMM on one PE attached to one DRAM channel (Phase 2). Used by the
// Phase 2 tests and by tools/apm_gemm. One call per process.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mem/dram_timing.h"
#include "pe/gemm_layout.h"
#include "pe/pe.h"

namespace apm {

struct GemmRun {
  DramParams dram;
  PeConfig pe;
  GemmShape shape;
  // DRAM placement. With auto_layout, A at 0 and B and C each start on the
  // next DRAM row boundary, and DRAM capacity grows to fit.
  bool auto_layout = true;
  std::uint64_t a_base = 0, b_base = 0, c_base = 0;
  std::vector<std::int8_t> a, b;  // row-major M x K and K x N
};

struct GemmRunResult {
  bool complete = false;
  std::map<std::string, std::uint64_t> values;
  std::vector<std::int32_t> c;  // row-major M x N, read back from DRAM

  std::string json() const;  // deterministic: keys sorted
};

GemmLayout layout_for(const GemmRun& run);
GemmRunResult run_gemm(GemmRun run);

}  // namespace apm
