#include "soc/accel.h"

#include <algorithm>
#include <stdexcept>

namespace apm {

// ---------------------------------------------------------------- Scheduler

Scheduler::Scheduler(sc_core::sc_module_name name, const SocPlan& plan)
    : sc_module(name), gdma("gdma", plan.config().gdma), plan_(plan) {
  for (std::uint64_t q = 0; q < plan.chunk_sets(); ++q) {
    staged_.emplace_back(plan.chunk_set(q).blocks.size());
    blocks_left_.push_back(plan.chunk_set(q).blocks.size());
    consumers_left_.push_back(plan.chunk_set(q).consumers);
  }
  for (unsigned p = 0; p < plan.pes(); ++p) {
    wb_done_.emplace_back(plan.tiles_of(p).size(), -1);
    wb_total_ += plan.tiles_of(p).size();
  }
  gdma.set_done_callback([this](std::size_t id, Cycle at) { on_gdma_done(id, at); });
  SC_METHOD(act);  // also runs at time 0: the first chunk-sets are staged then
  sensitive << wake_;
}

bool Scheduler::slot_free(std::uint64_t q) const {
  const unsigned d = plan_.config().ring_slots;
  return q < d || consumers_left_[q - d] == 0;
}

void Scheduler::consumer_done(std::uint64_t q) {
  if (--consumers_left_.at(q) == 0) wake_.notify(sc_core::SC_ZERO_TIME);
}

void Scheduler::tile_stored(unsigned pe, std::uint64_t k) {
  wb_q_.emplace_back(pe, k);
  wake_.notify(sc_core::SC_ZERO_TIME);
}

void Scheduler::act() {
  require(at_boundary(), "Scheduler: acting off a cycle boundary");
  const Cycle now = now_cycle();
  // 1. Write-backs first: they free C slots that stores are waiting for.
  while (!wb_q_.empty()) {
    const auto [pe, k] = wb_q_.front();
    wb_q_.pop_front();
    const std::size_t id = gdma.submit(plan_.writeback(pe, k));
    if (id >= desc_.size()) desc_.resize(id + 1);
    desc_[id] = {false, pe, k};
  }
  // 2. Staging, in chunk-set order, while the ring slot is free.
  while (next_q_ < plan_.chunk_sets() && slot_free(next_q_)) {
    const ChunkSet& set = plan_.chunk_set(next_q_);
    for (std::size_t b = 0; b < set.blocks.size(); ++b) {
      const StagedBlock& blk = set.blocks[b];
      const std::size_t id = gdma.submit(
          {DmaDir::DramToSram, blk.dram_addr, blk.sram_addr, 1, blk.bytes, blk.bytes, blk.bytes});
      if (id >= desc_.size()) desc_.resize(id + 1);
      desc_[id] = {true, next_q_, b};
      staged_[next_q_][b].submit = static_cast<std::int64_t>(now);
    }
    ++next_q_;
  }
}

void Scheduler::on_gdma_done(std::size_t id, Cycle at) {
  // Inside a TLM backward path: record, then poke/wake only.
  const DescKind& d = desc_.at(id);
  if (d.stage) {
    staged_[d.a][d.b].staged = static_cast<std::int64_t>(at);
    --blocks_left_[d.a];
    // PEs are gated per block (their own A and B), not per chunk-set, so every
    // staged block may unblock some PE (docs/BUGS.md #7).
    for (Pe* p : pes_) p->poke();
  } else {
    wb_done_[d.a][d.b] = static_cast<std::int64_t>(at);
    ++wb_count_;
    t_end_ = std::max(t_end_, at);
    pes_.at(d.a)->poke();
  }
}

// ---------------------------------------------------------------- GridJob

GridJob::GridJob(const SocPlan& plan, unsigned pe, Scheduler& sched) : plan_(plan), pe_(pe), sched_(sched) {}

std::uint64_t GridJob::set_of(std::uint64_t s) const {
  return plan_.set_index(plan_.tiles_of(pe_).at(s / chunks()).wave, s % chunks());
}

std::size_t GridJob::a_block(std::uint64_t s) const {
  // A blocks of a chunk-set are the present tile rows x = 0, 1, ... in order.
  return plan_.tiles_of(pe_).at(s / chunks()).x;
}

std::size_t GridJob::b_block(std::uint64_t s) const {
  const PeTile& t = plan_.tiles_of(pe_).at(s / chunks());
  std::size_t rows = 0;
  for (const StagedBlock& b : plan_.chunk_set(set_of(s)).blocks) rows += b.is_a ? 1 : 0;
  return rows + t.y;
}

bool GridJob::operands_ready(std::uint64_t s) const {
  const std::uint64_t q = set_of(s);
  return sched_.block_staged(q, a_block(s)) && sched_.block_staged(q, b_block(s));
}

// ---------------------------------------------------------------- Accel

Accel::Accel(sc_core::sc_module_name name, const AccelConfig& cfg, const SocPlan& plan)
    : sc_module(name),
      dram("dram", cfg.dram),
      sram("sram", cfg.grid.sram),
      sched("sched", plan),
      mesh("mesh", cfg.grid.mesh(), cfg.grid.sram_node, [&] {
        std::vector<Node> n;
        for (unsigned p = 0; p < plan.pes(); ++p) n.push_back(plan.node(p));
        return n;
      }()) {
  // SRAM port 0: global DMA. Ports 1..P: one mesh channel per PE.
  sched.gdma.dram_socket.bind(dram.socket);
  sched.gdma.sram_socket.bind(sram.socket);
  std::vector<Pe*> raw;
  for (unsigned p = 0; p < plan.pes(); ++p) {
    mesh.mem_side.bind(sram.socket);
    jobs.push_back(std::make_unique<GridJob>(plan, p, sched));
    pes.push_back(std::make_unique<Pe>(("pe" + std::to_string(p)).c_str(), cfg.grid.pe, *jobs.back()));
    pes.back()->dma.set_trace(true);
    pes.back()->dma.dram_socket.bind(mesh.pe_side);
    raw.push_back(pes.back().get());
  }
  sched.attach_pes(raw);
}

}  // namespace apm
