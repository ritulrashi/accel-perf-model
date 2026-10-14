// Runs one GEMM on the accelerator grid (Phase 3) and computes the stall
// stack. Every run enforces the design note §3.3 assertions and throws on a
// violation:
//   1. per PE, the categories sum to T_end;
//   2. over the chip, they sum to P * T_end;
//   3. each PE's compute cycles equal sum over its tiles of (nc*q*Kt + F);
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
  AccelConfig cfg;  // cfg.grid.pe.prec selects the precision (Phase 4)
  GemmShape shape;
  std::vector<std::int8_t> a, b;  // INT8, row-major M x K and K x N
  // Phase 4: operands as raw e-byte elements (row-major, little-endian);
  // used instead of a/b when non-empty.
  std::vector<std::uint8_t> a_bytes, b_bytes;
  std::uint64_t m_valid = 0;  // 0 = all M rows valid
};

struct AccelRunResult {
  bool complete = false;
  Cycle t_end = 0;
  std::vector<StallCounts> stalls;  // per PE
  StallCounts total{};
  std::map<std::string, std::uint64_t> values;
  // Row-major valid-rows x N, from DRAM: INT32 values, or FP32 bit patterns
  // in the floating-point modes.
  std::vector<std::int32_t> c;
  std::vector<std::uint8_t> image;  // final DRAM contents (run_accel_image only)

  std::string json() const;
};

GemmLayout accel_layout(const AccelRun& run);
AccelRunResult run_accel(AccelRun run);
// Phase 4 (design note §4.3): one GEMM op of a program. A and B are already
// packed in `image` at the layout's bases; C is written row-major at c_base.
// DRAM capacity is the image size; the final image is returned.
AccelRunResult run_accel_image(AccelConfig cfg, const GemmLayout& l, std::vector<std::uint8_t> image);

}  // namespace apm
