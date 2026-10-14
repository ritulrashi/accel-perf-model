// Runs one softmax op of a program (Phase 4 design note §4.3) on a DRAM image:
// a DRAM channel and the vector unit, nothing else (the PE grid is idle during
// a VU op). Every run enforces (and throws on violation):
//   1. the VU stall stack sums to T_end;
//   2. busy cycles = rows * T_row(n)  (the independent formula, §3.2);
//   3. DRAM bytes = rows * n * (4 + e) + scale bytes;
// and the classifier throws on any unexplained cycle. One call per process.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mem/dram_timing.h"
#include "vec/vu_plan.h"
#include "vec/vu_stall.h"

namespace apm {

struct VuRunResult {
  bool complete = false;
  Cycle t_end = 0;
  VuCounts stalls{};
  std::map<std::string, std::uint64_t> values;
  std::vector<std::uint8_t> image;  // final DRAM contents

  std::string json() const;
};

VuRunResult run_vu_image(DramParams dram, const VuConfig& cfg, const VuJob& job, std::vector<std::uint8_t> image);

}  // namespace apm
