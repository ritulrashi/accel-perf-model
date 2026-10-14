// Post-hoc stall classifier (Phase 3 design note §3.2). Pure C++.
//
// Input: the timestamps a PE run recorded (feeds, loads, tiles, stores),
// the staging times of each step's operand blocks, and the trace of every
// load/store burst. Output: for each cycle in [0, t_end) exactly one category,
// counted per category. Rules, in order, for cycle c:
//   1. inside a feed [f, f+Kt) or a fill/drain tail [last feed end, tile done) -> compute
//   2. else, s* = first step not yet fed at c:
//      - its loads are complete: the feed can only be waiting for the output
//        buffer (tile t(s*) - n_out not yet stored) -> classify that store (rule 4)
//      - an operand block is not staged: staging submitted -> dram, else -> idle
//      - both staged: the oldest unfinished burst of the step's loads, by its trace
//   3. else an unfinished store -> rule 4 on the oldest; otherwise -> idle
//   4. a store not yet submitted (shared-SRAM C slot busy) -> dram; else the
//      oldest unfinished burst of the store, by its trace
// Any situation the rules cannot explain throws std::logic_error naming the
// PE and cycle: the classifier doubles as a consistency check on the model.
#pragma once
#include <cstdint>
#include <vector>

#include "common/trace.h"

namespace apm {

struct StageRec {
  std::int64_t submit = -1, staged = -1;
};

struct StepTrace {
  std::int64_t loads_submit = -1, loaded = -1, feed = -1;
  std::uint64_t tile = 0;  // PE-local tile index
  StageRec a, b;
  std::vector<BurstRecord> load_bursts;  // A bursts then B bursts (DMA stream order)
};

struct TileTrace {
  std::int64_t done = -1, store_submit = -1, store_done = -1;
  std::vector<BurstRecord> store_bursts;
};

struct PeTrace {
  Cycle kt = 0, fill_drain = 0;
  std::uint64_t chunks = 1;
  unsigned n_out = 2;
  std::vector<StepTrace> steps;
  std::vector<TileTrace> tiles;
};

StallCounts classify_pe(const PeTrace& t, Cycle t_end, unsigned pe_id);

}  // namespace apm
