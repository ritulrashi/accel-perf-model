// Banked SRAM timing formulas (design note §3.5). Each bank serves one Wb-byte
// line per cycle; lines that land in the same bank serialize.
#pragma once
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"

namespace apm {

struct SramParams {
  unsigned banks = 8;        // Bs
  unsigned bank_bytes = 16;  // Wb: line width of one bank
  Cycle t_sram = 2;          // from the last line's bank cycle to response
  std::uint64_t capacity_bytes = 1 << 20;

  void validate() const;  // throws std::invalid_argument
};

struct SramSchedule {
  Cycle response;
  std::uint64_t lines;
  // Sum over lines of the cycles that line waited for another request's use
  // of its bank. Serialization within the request itself is not counted.
  std::uint64_t conflict_line_cycles;
  // Extra cycles this request's response is late because of other
  // requests: (last line cycle) - (accept + most lines in any one bank - 1).
  // Used by the Phase 3 stall stack (sram_conflict).
  Cycle conflict_delay;
};

class SramTiming {
 public:
  explicit SramTiming(const SramParams& p);

  unsigned bank_of(std::uint64_t addr) const;
  void check_legal(std::uint64_t addr, std::uint64_t size) const;
  SramSchedule schedule(Cycle accept, std::uint64_t addr, std::uint64_t size);

  std::uint64_t lines_accessed() const { return lines_; }
  std::uint64_t conflict_line_cycles() const { return conflict_; }
  const SramParams& params() const { return p_; }

 private:
  SramParams p_;
  std::vector<Cycle> bank_free_;
  std::uint64_t lines_ = 0, conflict_ = 0;
};

}  // namespace apm
