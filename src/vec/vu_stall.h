// VU stall stack (Phase 4 design note §3.5). Pure C++. Every cycle in
// [0, t_end) gets exactly one cause:
//   busy  - inside a row compute [start, start + T_row)
//   load  - the next row to compute is not loaded yet
//   store - the next row is loaded but its output buffer waits for a store
//           (row r - n_vout), or every row is computed and stores are in flight
// Anything else throws std::logic_error: the classifier doubles as a check on
// the controller.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"

namespace apm {

enum class VuCat : std::uint8_t { Busy, Load, Store };
inline constexpr std::size_t kVuCats = 3;
using VuCounts = std::array<std::uint64_t, kVuCats>;
inline const char* vu_cat_name(VuCat c) {
  static const char* const names[] = {"busy", "load", "store"};
  return names[static_cast<std::size_t>(c)];
}

struct VuRowTrace {
  std::int64_t loaded = -1, start = -1, store_done = -1;
};

VuCounts classify_vu(const std::vector<VuRowTrace>& rows, Cycle t_row, unsigned n_vout,
                     std::int64_t scale_done, Cycle t_end);

}  // namespace apm
