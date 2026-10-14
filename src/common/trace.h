// Stall categories and per-burst traces (Phase 3 design note §3). Pure C++.
//
// Every component that holds a traced burst's payload appends a segment
// [from, to) -> category describing what the burst was doing there. Cycles
// not covered by any segment (DMA queues, between legs) count as Dma.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"

namespace apm {

enum class StallCat : std::uint8_t { Compute, Dram, SramConflict, Network, Dma, Idle };
inline constexpr std::size_t kStallCats = 6;
using StallCounts = std::array<std::uint64_t, kStallCats>;

inline const char* stall_name(StallCat c) {
  static const char* const names[] = {"compute", "dram", "sram_conflict", "network", "dma", "idle"};
  return names[static_cast<std::size_t>(c)];
}

struct Segment {
  Cycle from, to;  // [from, to)
  StallCat cat;
};

struct BurstRecord {
  Cycle enq = 0;   // descriptor submit cycle
  Cycle done = 0;  // write response cycle
  std::vector<Segment> segs;
};

// Where a burst is at cycle c: the segment containing c, else Dma.
inline StallCat locate(const BurstRecord& b, Cycle c) {
  for (const Segment& s : b.segs)
    if (s.from <= c && c < s.to) return s.cat;
  return StallCat::Dma;
}

}  // namespace apm
