// One processing element (Phase 2 design note §2-§3): local memory (an Sram
// instance), a DMA engine, an output-stationary systolic unit, and the
// controller that runs a GEMM out of DRAM with double buffering.
//
// The controller is an SC_METHOD that acts at cycle boundaries. Each wake-up:
//   1. finishes the tile in the array if its drain ends now (result -> output
//      buffer, store submitted);
//   2. submits loads (A then B) for every next step whose operand buffer is free;
//   3. starts the next feed if the array is free, the step's operands are
//      loaded, and (first chunk of a tile) the previous tile is done and the
//      tile's output buffer has been stored.
// It wakes on DMA completions (same cycle), at feed end, at tile done, and at
// time 0. Reaction latency is zero: a load completing at X can feed at X.
#pragma once
#include <systemc>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/cycle.h"
#include "compute/systolic.h"
#include "dma/dma_engine.h"
#include "mem/sram.h"
#include "pe/gemm_layout.h"
#include "pe/pe_config.h"

namespace apm {

class Pe : public sc_core::sc_module {
 public:
  Sram lmem;
  DmaEngine dma;  // dma.dram_socket must be bound to a DRAM by the parent

  Pe(sc_core::sc_module_name name, const PeConfig& cfg, const GemmLayout& job);

  bool finished() const { return stores_done_ == layout_.tiles(); }
  Cycle done_cycle() const { return done_cycle_; }
  void report(std::map<std::string, std::uint64_t>& out) const;

 private:
  enum class Kind { LoadA, LoadB, Store };
  struct DescInfo {
    Kind kind;
    std::uint64_t index;  // step for loads, tile for stores
  };

  void act();
  bool operand_buffer_free(std::uint64_t step, Cycle now) const;
  void issue_loads(std::uint64_t step);
  void try_feed(Cycle now);
  void finish_tile(Cycle now);
  void record(std::size_t desc_id, DescInfo info);
  void on_dma_done(std::size_t id, Cycle at);
  void arm_wake(Cycle now);

  PeConfig cfg_;
  GemmLayout layout_;
  Accumulators acc_;
  std::vector<std::uint64_t> a_buf_, b_buf_, out_buf_;  // LMEM addresses

  std::vector<DescInfo> desc_info_;  // by DMA descriptor id
  std::uint64_t next_load_ = 0, next_feed_ = 0;
  std::vector<int> loads_done_;            // per step: 0..2 (A, B)
  std::vector<std::int64_t> loaded_at_;    // per step, -1 until both loads done
  std::vector<std::int64_t> feed_start_;   // per step, -1 until fed
  Cycle feeding_until_ = 0;                // array busy feeding until this cycle
  bool tile_in_array_ = false;             // accumulators hold an unfinished tile
  std::uint64_t cur_tile_ = 0;
  Cycle tile_done_at_ = 0;
  bool drain_pending_ = false;             // last chunk fed; tile_done_at_ is valid
  std::vector<std::int64_t> tile_done_;    // per tile
  std::vector<std::int64_t> store_done_;   // per tile
  std::uint64_t stores_done_ = 0;
  Cycle done_cycle_ = 0;
  sc_core::sc_event wake_now_, wake_timed_;
};

}  // namespace apm
