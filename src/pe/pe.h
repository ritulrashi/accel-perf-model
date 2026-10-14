// One processing element (Phase 2 design note §2-§3, generalized in Phase 3
// §2.3): local memory (an Sram instance), a DMA engine, an output-stationary
// systolic unit, and the controller that runs a PeJob with double buffering.
//
// The controller is an SC_METHOD that acts at cycle boundaries. Each wake-up:
//   1. finishes the tile in the array if its drain ends now (result -> output
//      buffer, store queued);
//   2. submits queued stores the job allows (Phase 3: the shared-SRAM C slot
//      must be free);
//   3. submits loads (A then B) for every next step whose operand buffer is
//      free and whose operands the job says are ready;
//   4. starts the next feed if the array is free, the step's operands are
//      loaded, and (first chunk of a tile) the previous tile is done and the
//      tile's output buffer has been stored.
// It wakes on DMA completions (same cycle), poke() (same cycle), feed end,
// tile done, and at time 0. Reaction latency is zero.
#pragma once
#include <systemc>

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "common/cycle.h"
#include "compute/schedule_io.h"
#include "compute/systolic.h"
#include "dma/dma_engine.h"
#include "mem/sram.h"
#include "pe/pe_config.h"
#include "pe/pe_job.h"

namespace apm {

class Pe : public sc_core::sc_module {
 public:
  Sram lmem;
  DmaEngine dma;  // dma.dram_socket ("far" side) must be bound by the parent

  // `job` must outlive the PE.
  Pe(sc_core::sc_module_name name, const PeConfig& cfg, PeJob& job);

  // Wake the controller in the current cycle (the job's gates may have opened).
  void poke() { wake_now_.notify(sc_core::SC_ZERO_TIME); }

  bool finished() const { return stores_done_ == job_.tiles(); }
  Cycle done_cycle() const { return done_cycle_; }
  void report(std::map<std::string, std::uint64_t>& out) const;

  // Recorded timestamps for the Phase 3 stall stack (-1 = never happened).
  struct StepRec {
    std::int64_t loads_submit = -1, loaded = -1, feed = -1;
    std::size_t a_desc = 0, b_desc = 0;
  };
  struct TileRec {
    std::int64_t done = -1, store_submit = -1, store_done = -1;
    std::size_t store_desc = 0;
  };
  const std::vector<StepRec>& step_records() const { return steps_; }
  const std::vector<TileRec>& tile_records() const { return tiles_; }

  // Phase 5 (design note §3.1): keep the operand blocks each feed read and the
  // values each drain wrote, so the schedule can be replayed into the RTL.
  // INT8 only (the RTL scope). Call before sc_start.
  void capture_schedule(bool on);
  PeSchedule schedule(unsigned pe_id) const;

 private:
  enum class Kind { LoadA, LoadB, Store };
  struct DescInfo {
    Kind kind;
    std::uint64_t index;  // step for loads, tile for stores
  };

  void act();
  bool operand_buffer_free(std::uint64_t step, Cycle now) const;
  void issue_loads(std::uint64_t step, Cycle now);
  void try_feed(Cycle now);
  void finish_tile(Cycle now);
  void try_stores(Cycle now);
  void record(std::size_t desc_id, DescInfo info);
  void on_dma_done(std::size_t id, Cycle at);
  void arm_wake(Cycle now);

  PeConfig cfg_;
  PeJob& job_;
  Accumulators acc_;
  std::vector<std::uint64_t> a_buf_, b_buf_, out_buf_;  // LMEM addresses

  std::vector<DescInfo> desc_info_;  // by DMA descriptor id
  std::uint64_t next_load_ = 0, next_feed_ = 0;
  std::vector<int> loads_done_;            // per step: 0..2 (A, B)
  std::vector<StepRec> steps_;
  std::vector<TileRec> tiles_;
  Cycle feeding_until_ = 0;                // array busy feeding until this cycle
  bool tile_in_array_ = false;             // accumulators hold an unfinished tile
  std::uint64_t cur_tile_ = 0;
  Cycle tile_done_at_ = 0;
  bool drain_pending_ = false;             // last chunk fed; tile_done_at_ is valid
  std::deque<std::uint64_t> store_q_;      // finished tiles whose store is not yet submitted
  std::uint64_t stores_done_ = 0;
  Cycle done_cycle_ = 0;
  bool capture_ = false;
  std::vector<std::vector<std::uint8_t>> cap_a_, cap_b_, cap_c_;
  sc_core::sc_event wake_now_, wake_timed_;
};

}  // namespace apm
