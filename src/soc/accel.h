// Accelerator top (Phase 3 design note §0-§2): DRAM channel, shared SRAM,
// scheduler with the global DMA, mesh, and a Px x Py grid of PEs.
//
// Scheduler (SC_METHOD at cycle boundaries; woken by its own events):
//   1. submits queued write-backs (SRAM C slot -> DRAM) first: they free C slots;
//   2. stages chunk-sets in order while their ring slot is free (every
//      consumer of chunk-set q - D has finished loading it), A blocks then B.
// Global-DMA completions (inside TLM paths) only record, then poke the PEs
// or wake the scheduler.
#pragma once
#include <systemc>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/cycle.h"
#include "dma/dma_engine.h"
#include "mem/dram_channel.h"
#include "mem/sram.h"
#include "noc/mesh.h"
#include "pe/pe.h"
#include "pe/pe_job.h"
#include "soc/soc_plan.h"
#include "soc/stall.h"

namespace apm {

class Scheduler : public sc_core::sc_module {
 public:
  DmaEngine gdma;

  Scheduler(sc_core::sc_module_name name, const SocPlan& plan);
  void attach_pes(std::vector<Pe*> pes) { pes_ = std::move(pes); }

  // For GridJob (called from inside TLM paths: record + notify only).
  bool block_staged(std::uint64_t q, std::size_t blk) const { return staged_.at(q).at(blk).staged >= 0; }
  bool writeback_done(unsigned pe, std::uint64_t k) const { return wb_done_.at(pe).at(k) >= 0; }
  void consumer_done(std::uint64_t q);
  void tile_stored(unsigned pe, std::uint64_t k);

  const StageRec& stage_rec(std::uint64_t q, std::size_t blk) const { return staged_.at(q).at(blk); }
  bool finished() const { return wb_count_ == wb_total_; }
  Cycle t_end() const { return t_end_; }

 private:
  struct DescKind {
    bool stage;
    std::uint64_t a, b;  // stage: (q, block); write-back: (pe, k)
  };
  void act();
  void on_gdma_done(std::size_t id, Cycle at);
  bool slot_free(std::uint64_t q) const;

  const SocPlan& plan_;
  std::vector<Pe*> pes_;
  std::vector<std::vector<StageRec>> staged_;        // [q][block]
  std::vector<std::uint64_t> blocks_left_;           // [q] blocks not yet staged
  std::vector<unsigned> consumers_left_;             // [q]
  std::vector<std::vector<std::int64_t>> wb_done_;   // [pe][k]
  std::deque<std::pair<unsigned, std::uint64_t>> wb_q_;
  std::vector<DescKind> desc_;
  std::uint64_t next_q_ = 0, wb_count_ = 0, wb_total_ = 0;
  Cycle t_end_ = 0;
  sc_core::sc_event wake_;
};

// One PE's view of the grid schedule.
class GridJob : public PeJob {
 public:
  GridJob(const SocPlan& plan, unsigned pe, Scheduler& sched);

  std::uint64_t tiles() const override { return plan_.tiles_of(pe_).size(); }
  std::uint64_t chunks() const override { return plan_.chunks(); }
  std::uint64_t a_src(std::uint64_t s) const override { return plan_.a_sram(pe_, s / chunks(), s % chunks()); }
  std::uint64_t b_src(std::uint64_t s) const override { return plan_.b_sram(pe_, s / chunks(), s % chunks()); }
  TileDesc store(std::uint64_t t, std::uint64_t lmem_src) const override { return plan_.pe_store(pe_, t, lmem_src); }
  bool operands_ready(std::uint64_t s) const override;
  bool store_allowed(std::uint64_t t) const override { return t < 2 || sched_.writeback_done(pe_, t - 2); }
  void on_loads_done(std::uint64_t s, Cycle) override { sched_.consumer_done(set_of(s)); }
  void on_store_done(std::uint64_t t, Cycle) override { sched_.tile_stored(pe_, t); }

  std::uint64_t set_of(std::uint64_t s) const;
  std::size_t a_block(std::uint64_t s) const;  // position of the step's A block in its chunk-set
  std::size_t b_block(std::uint64_t s) const;

 private:
  const SocPlan& plan_;
  unsigned pe_;
  Scheduler& sched_;
};

struct AccelConfig {
  DramParams dram;
  GridConfig grid;
  // Phase 5 (design note §3.1): non-empty = write every PE's feed schedule here.
  std::string schedule_path;
};

class Accel : public sc_core::sc_module {
 public:
  DramChannel dram;
  Sram sram;
  Scheduler sched;
  Mesh mesh;
  std::vector<std::unique_ptr<GridJob>> jobs;
  std::vector<std::unique_ptr<Pe>> pes;

  Accel(sc_core::sc_module_name name, const AccelConfig& cfg, const SocPlan& plan);
};

}  // namespace apm
