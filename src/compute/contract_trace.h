// The model's view of a PE schedule, cycle by cycle (Phase 5 design note
// §3.2): every feed of the schedule expanded with the model's own contract
// functions (systolic.h: mac_cycle, drain_start, tile_done; INT8, q = 1).
// Pure C++.
//
// One line per cycle that has a MAC or a drain row, in the format the RTL
// testbench also produces:
//   <cycle> mac <k_00> <k_01> ... <k_(R-1)(C-1)> drain <row|->
// k is the global reduction index within the tile ('.' = no MAC). The RTL
// side is compared to these lines byte for byte.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "common/cycle_types.h"
#include "compute/schedule_io.h"

namespace apm {

// Formats one trace line. ks holds R*C entries in row-major cell order,
// -1 = no MAC; drain_row -1 = no drain.
std::string trace_line(Cycle c, const std::vector<std::int64_t>& ks, int drain_row);

struct ContractTiming {
  std::vector<Cycle> drain_start;  // per tile, in tile order
  std::vector<Cycle> done;         // per tile: tile_done() of its last feed
};

// Throws std::invalid_argument if two MACs land on one cell in one cycle, or a
// drain overlaps another (the schedule would violate the contract).
std::vector<std::string> contract_trace(const PeSchedule& s, ContractTiming* timing = nullptr);

}  // namespace apm
