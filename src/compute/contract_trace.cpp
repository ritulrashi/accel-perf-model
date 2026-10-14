#include "compute/contract_trace.h"

#include <map>
#include <stdexcept>

#include "compute/systolic.h"

namespace apm {

std::string trace_line(Cycle c, const std::vector<std::int64_t>& ks, int drain_row) {
  std::string s = std::to_string(c) + " mac";
  for (std::int64_t k : ks) {
    s += ' ';
    s += k < 0 ? std::string(".") : std::to_string(k);
  }
  s += " drain ";
  s += drain_row < 0 ? std::string("-") : std::to_string(drain_row);
  return s;
}

std::vector<std::string> contract_trace(const PeSchedule& s, ContractTiming* timing) {
  const ArrayShape a{s.rows, s.cols};
  const std::size_t cells = std::size_t{s.rows} * s.cols;
  struct Ev {
    std::vector<std::int64_t> ks;
    int drain = -1;
  };
  std::map<Cycle, Ev> ev;
  auto at = [&](Cycle c) -> Ev& {
    Ev& e = ev[c];
    if (e.ks.empty()) e.ks.assign(cells, -1);
    return e;
  };
  ContractTiming t;
  for (const SchedStep& st : s.steps) {
    for (std::uint64_t k = 0; k < s.kt; ++k)
      for (unsigned i = 0; i < s.rows; ++i)
        for (unsigned j = 0; j < s.cols; ++j) {
          std::int64_t& slot = at(mac_cycle(st.feed, i, j, k)).ks[std::size_t{i} * s.cols + j];
          if (slot >= 0) throw std::invalid_argument("contract_trace: two MACs on one cell in one cycle");
          slot = static_cast<std::int64_t>(st.chunk * s.kt + k);
        }
    if (st.chunk + 1 == s.chunks) {  // last chunk: the drain follows
      const Cycle d0 = drain_start(a, st.feed, s.kt);
      for (unsigned n = 0; n < s.rows; ++n) {
        Ev& e = at(d0 + n);
        if (e.drain >= 0) throw std::invalid_argument("contract_trace: overlapping drains");
        e.drain = static_cast<int>(s.rows - 1 - n);  // row R-1 first
      }
      t.drain_start.push_back(d0);
      t.done.push_back(tile_done(a, st.feed, s.kt));
    }
  }
  std::vector<std::string> lines;
  lines.reserve(ev.size());
  for (const auto& [c, e] : ev) lines.push_back(trace_line(c, e.ks, e.drain));
  if (timing) *timing = std::move(t);
  return lines;
}

}  // namespace apm
