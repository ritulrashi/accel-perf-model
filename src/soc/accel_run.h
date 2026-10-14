// Runs one GEMM on the accelerator grid (Phase 3) and computes the stall
// stack. Every run enforces the design note §3.3 assertions and throws on a
// violation:
//   1. per PE, the categories sum to T_end;
//   2. over the chip, they sum to P * T_end;
//   3. each PE's compute cycles equal sum over its tiles of (nc*Kt + F);
//   4. DRAM bytes moved equal the plan's exact count (and, for full waves,
//      the closed-form formula);
// and the classifier itself throws on any unexplained stall.
// One call per process.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/trace.h"
#include "soc/accel.h"

namespace apm {

struct AccelRun {
  AccelConfig cfg;
  GemmShape shape;
  std::vector<std::int8_t> a, b;  // row-major M x K and K x N
};

struct AccelRunResult {
  bool complete = false;
  Cycle t_end = 0;
  std::vector<StallCounts> stalls;  // per PE
  StallCounts total{};
  std::map<std::string, std::uint64_t> values;
  std::vector<std::int32_t> c;  // row-major M x N, from DRAM

  std::string json() const;
};

GemmLayout accel_layout(const AccelRun& run);
AccelRunResult run_accel(AccelRun run);

}  // namespace apm
