#include "soc/stall.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace apm {

namespace {

// Category for cycles with no work and for synchronization waits.
constexpr StallCat kIdle = StallCat::Idle;
// Category of a tile's fill/drain tail (part of the tile latency formula).
constexpr StallCat kTail = StallCat::Compute;

struct Busy {
  Cycle from, to;
  StallCat cat;
  bool operator<(const Busy& o) const { return from < o.from; }
};

// "Event e had happened by cycle c."
bool by(std::int64_t e, Cycle c) { return e >= 0 && static_cast<Cycle>(e) <= c; }

[[noreturn]] void unexplained(unsigned pe, Cycle c, const std::string& why) {
  throw std::logic_error("unexplained stall at PE " + std::to_string(pe) + " cycle " +
                         std::to_string(c) + ": " + why);
}

StallCat oldest_unfinished(const std::vector<BurstRecord>& bursts, Cycle c, unsigned pe,
                           const char* what) {
  for (const BurstRecord& b : bursts)
    if (b.done > c) return locate(b, c);
  unexplained(pe, c, std::string("waiting on ") + what + " whose bursts are all done");
}

StallCat classify_store(const PeTrace& t, std::uint64_t tile, Cycle c, unsigned pe) {
  const TileTrace& tt = t.tiles.at(tile);
  if (!by(tt.done, c)) unexplained(pe, c, "store of a tile that is not done");
  if (!by(tt.store_submit, c)) return StallCat::Dram;  // C slot: write-back in flight
  return oldest_unfinished(tt.store_bursts, c, pe, "a store");
}

}  // namespace

StallCounts classify_pe(const PeTrace& t, Cycle t_end, unsigned pe) {
  // Compute intervals: every feed, and every tile's fill/drain tail. They do
  // not overlap (one feed at a time; a tile's first feed waits for the
  // previous tile's drain), so a sorted sweep suffices.
  std::vector<Busy> busy;
  for (std::size_t s = 0; s < t.steps.size(); ++s) {
    const StepTrace& st = t.steps[s];
    if (st.feed < 0) continue;
    const Cycle f = static_cast<Cycle>(st.feed);
    busy.push_back({f, f + t.kt, StallCat::Compute});
    if ((s + 1) % t.chunks == 0) {  // last chunk of its tile: the tail follows
      const TileTrace& tt = t.tiles.at(st.tile);
      if (tt.done < 0) unexplained(pe, f, "last chunk fed but tile never done");
      busy.push_back({f + t.kt, static_cast<Cycle>(tt.done), kTail});
    }
  }
  std::sort(busy.begin(), busy.end());

  StallCounts n{};
  std::size_t bi = 0, sp = 0;
  for (Cycle c = 0; c < t_end; ++c) {
    while (bi < busy.size() && busy[bi].to <= c) ++bi;
    while (sp < t.steps.size() && by(t.steps[sp].feed, c)) ++sp;

    StallCat cat;
    if (bi < busy.size() && busy[bi].from <= c) {
      cat = busy[bi].cat;
    } else if (sp < t.steps.size()) {
      const StepTrace& s = t.steps[sp];
      if (by(s.loaded, c)) {
        const bool first_chunk = sp % t.chunks == 0;
        if (!first_chunk || s.tile < t.n_out || by(t.tiles.at(s.tile - t.n_out).store_done, c))
          unexplained(pe, c, "step " + std::to_string(sp) + " loaded but not fed");
        cat = classify_store(t, s.tile - t.n_out, c, pe);  // waiting for the output buffer
      } else if (!by(s.a.staged, c)) {
        cat = by(s.a.submit, c) ? StallCat::Dram : kIdle;
      } else if (!by(s.b.staged, c)) {
        cat = by(s.b.submit, c) ? StallCat::Dram : kIdle;
      } else {
        if (!by(s.loads_submit, c))
          unexplained(pe, c, "step " + std::to_string(sp) + " staged but loads not submitted");
        cat = oldest_unfinished(s.load_bursts, c, pe, "loads");
      }
    } else {
      cat = kIdle;
      for (std::uint64_t k = 0; k < t.tiles.size(); ++k)
        if (!by(t.tiles[k].store_done, c)) {
          cat = classify_store(t, k, c, pe);
          break;
        }
    }
    ++n[static_cast<std::size_t>(cat)];
  }
  return n;
}

}  // namespace apm
