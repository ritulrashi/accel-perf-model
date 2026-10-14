// Vector unit (Phase 4 design note §3): a DMA engine on a DRAM port, a VMEM
// (an Sram instance; Phase 1 timing on the DMA side, ideal VU port), and a
// controller that runs the row softmax of a VuJob with double buffering.
//
// The controller is an SC_METHOD that acts at cycle boundaries. Each wake-up:
//   1. finishes the row in the VU if its compute ends now (store submitted;
//      after the last row, the INT8 per-row scales too);
//   2. submits the load of every next row whose input buffer is free (the
//      compute of row r - n_vbuf has ended: pass 2 overwrites the input);
//   3. starts the next row if the VU is free, the row is loaded, and its
//      output buffer is free (the store of row r - n_vout is done).
// The functional softmax runs at compute start (inputs cannot change before
// compute end). It wakes on DMA completions (same cycle) and compute end.
#pragma once
#include <systemc>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/cycle.h"
#include "dma/dma_engine.h"
#include "mem/sram.h"
#include "vec/vu_plan.h"

namespace apm {

class VecUnit : public sc_core::sc_module {
 public:
  Sram vmem;
  DmaEngine dma;  // dma.dram_socket must be bound by the parent

  VecUnit(sc_core::sc_module_name name, const VuConfig& cfg, const VuJob& job);

  struct RowRec {
    std::int64_t load_submit = -1, loaded = -1, start = -1, store_submit = -1, store_done = -1;
  };
  const std::vector<RowRec>& rows() const { return rows_; }
  std::int64_t scale_store_done() const { return scale_done_; }
  bool finished() const;
  Cycle done_cycle() const { return done_cycle_; }
  Cycle row_cycles() const { return t_row_; }

 private:
  enum class Kind { Load, Store, Scales };
  struct DescInfo {
    Kind kind;
    std::uint64_t row;
  };

  void act();
  void record(std::size_t id, DescInfo d);
  void on_dma_done(std::size_t id, Cycle at);

  VuConfig cfg_;
  VuJob job_;
  VuBuffers bufs_;
  Cycle t_row_;
  std::vector<RowRec> rows_;
  std::vector<DescInfo> desc_info_;
  std::vector<float> scales_;
  std::uint64_t next_load_ = 0, next_compute_ = 0, stores_done_ = 0;
  bool computing_ = false;
  std::uint64_t cur_row_ = 0;
  Cycle busy_until_ = 0;
  std::int64_t scale_done_ = -1;
  Cycle done_cycle_ = 0;
  sc_core::sc_event wake_now_, wake_timed_;
};

}  // namespace apm
