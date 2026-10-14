// DRAM channel timing formulas (design note §3.4). No SystemC here: given the
// cycle a request was accepted, compute its full schedule and update bank,
// row-buffer and bus state. Requests must be presented in grant order; the
// result is in-order FCFS by construction.
#pragma once
#include <cstdint>
#include <optional>
#include <vector>

#include "common/cycle_types.h"

namespace apm {

// Bank-address map (Phase 5 design note §6). Row: bank = row_index mod B
// (Phases 1-4). Xor: bank = (row_index XOR row_index / B) mod B, B a power of
// two. The row within the bank is row_index / B in both, so both are bijections.
enum class BankMap : std::uint8_t { Row, Xor };

struct DramParams {
  // Smallest legal request and its alignment (bytes). Requests need not be
  // bus-aligned; they occupy every W-byte beat they touch.
  static constexpr unsigned kGranule = 16;
  unsigned banks = 4;
  std::uint64_t row_bytes = 1024;
  unsigned bus_bytes_per_cycle = 16;  // W
  Cycle t_hit = 10;                   // column access, row already open
  Cycle t_miss = 30;                  // activate (+precharge) + column access
  unsigned queue_depth = 8;           // Q: accepted-but-not-responded requests
  std::uint64_t burst_bytes = 64;     // largest legal request
  std::uint64_t capacity_bytes = 1 << 20;
  BankMap bank_map = BankMap::Row;

  void validate() const;  // throws std::invalid_argument
};

struct DramAddr {
  unsigned bank;
  std::uint64_t row;
};

struct DramSchedule {
  unsigned bank;
  std::uint64_t row;
  bool hit;
  Cycle start;       // bank available
  Cycle col_issue;   // row open, column command issued
  Cycle data_start;  // first data beat on the bus
  Cycle data_end;    // bus free again; the response is sent at this cycle
};

class DramTiming {
 public:
  explicit DramTiming(const DramParams& p);

  DramAddr map(std::uint64_t addr) const;
  // Throws std::invalid_argument on an illegal request (§3.2).
  void check_legal(std::uint64_t addr, std::uint64_t size) const;
  // Schedules one request accepted at cycle `accept`, mutating state.
  DramSchedule schedule(Cycle accept, std::uint64_t addr, std::uint64_t size);

  std::uint64_t row_hits() const { return row_hits_; }
  std::uint64_t row_misses() const { return row_misses_; }
  std::uint64_t bus_busy_cycles() const { return bus_busy_; }
  const DramParams& params() const { return p_; }

 private:
  DramParams p_;
  std::vector<std::optional<std::uint64_t>> open_row_;
  std::vector<Cycle> bank_ready_;
  Cycle bus_free_ = 0;
  std::uint64_t row_hits_ = 0, row_misses_ = 0, bus_busy_ = 0;
};

}  // namespace apm
