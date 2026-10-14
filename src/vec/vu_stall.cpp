#include "vec/vu_stall.h"

#include <stdexcept>
#include <string>

namespace apm {

namespace {

bool by(std::int64_t e, Cycle c) { return e >= 0 && static_cast<Cycle>(e) <= c; }

[[noreturn]] void unexplained(Cycle c, const std::string& why) {
  throw std::logic_error("unexplained VU cycle " + std::to_string(c) + ": " + why);
}

}  // namespace

VuCounts classify_vu(const std::vector<VuRowTrace>& rows, Cycle t_row, unsigned n_vout,
                     std::int64_t scale_done, Cycle t_end) {
  VuCounts n{};
  std::size_t next = 0;  // first row not started by cycle c
  for (Cycle c = 0; c < t_end; ++c) {
    while (next < rows.size() && by(rows[next].start, c)) ++next;
    VuCat cat;
    if (next > 0 && c < static_cast<Cycle>(rows[next - 1].start) + t_row) {
      cat = VuCat::Busy;  // rows never overlap, so only the latest started row can be running
    } else if (next < rows.size()) {
      const VuRowTrace& r = rows[next];
      if (!by(r.loaded, c)) {
        cat = VuCat::Load;
      } else if (next >= n_vout && !by(rows[next - n_vout].store_done, c)) {
        cat = VuCat::Store;
      } else {
        unexplained(c, "row " + std::to_string(next) + " loaded, output buffer free, VU idle, not started");
      }
    } else {
      bool pending = scale_done >= 0 && !by(scale_done, c);
      for (const VuRowTrace& r : rows) pending = pending || !by(r.store_done, c);
      if (!pending) unexplained(c, "every row stored but the run has not ended");
      cat = VuCat::Store;
    }
    ++n[static_cast<std::size_t>(cat)];
  }
  return n;
}

}  // namespace apm
