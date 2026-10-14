#include "soc/soc_plan.h"

#include <stdexcept>
#include <string>

namespace apm {

SocPlan::SocPlan(const GridConfig& cfg, const GemmLayout& dram_layout) : cfg_(cfg), l_(dram_layout) {
  l_.validate();
  if (l_.array.rows != cfg.pe.array.rows || l_.array.cols != cfg.pe.array.cols || l_.kt != cfg.pe.kt)
    throw std::invalid_argument("SocPlan: GEMM layout does not match the PE array/Kt");
  if (cfg.px == 0 || cfg.py == 0) throw std::invalid_argument("SocPlan: empty grid");
  if (cfg.ring_slots == 0) throw std::invalid_argument("SocPlan: ring needs at least one slot");
  if (cfg.sram_node.x >= cfg.px || cfg.sram_node.y >= cfg.py)
    throw std::invalid_argument("SocPlan: sram_node outside the grid");
  cfg.mesh().validate();
  cfg.sram.validate();
  plan_pe_buffers(cfg.pe);  // LMEM capacity of every PE

  waves_m_ = (l_.tiles_m() + cfg.px - 1) / cfg.px;
  waves_n_ = (l_.tiles_n() + cfg.py - 1) / cfg.py;

  // Shared-SRAM layout: D ring slots, then two C slots per PE.
  slot_bytes_ = cfg.px * l_.a_block_bytes() + cfg.py * l_.b_block_bytes();
  c_base_ = cfg.ring_slots * slot_bytes_;
  sram_required_ = c_base_ + 2ull * cfg.pes() * l_.c_tile_bytes();
  if (sram_required_ > cfg.sram.capacity_bytes)
    throw std::invalid_argument("SocPlan: shared SRAM too small: " + std::to_string(cfg.ring_slots) +
                                " ring slots x " + std::to_string(slot_bytes_) + " B + " +
                                std::to_string(2 * cfg.pes()) + " C slots x " +
                                std::to_string(l_.c_tile_bytes()) + " B = " +
                                std::to_string(sram_required_) + " B > capacity " +
                                std::to_string(cfg.sram.capacity_bytes) + " B");

  // Tiles per PE, in wave order.
  pe_tiles_.assign(cfg.pes(), {});
  for (std::uint64_t bm = 0; bm < waves_m_; ++bm)
    for (std::uint64_t bn = 0; bn < waves_n_; ++bn) {
      const std::uint64_t w = bm * waves_n_ + bn;
      for (unsigned x = 0; x < cfg.px; ++x)
        for (unsigned y = 0; y < cfg.py; ++y) {
          const std::uint64_t mi = bm * cfg.px + x, nj = bn * cfg.py + y;
          if (mi < l_.tiles_m() && nj < l_.tiles_n()) pe_tiles_[x * cfg.py + y].push_back({mi, nj, w, x, y});
        }
    }

  // Chunk-sets in staging order: waves in order, then chunks.
  for (std::uint64_t w = 0; w < waves(); ++w) {
    const std::uint64_t bm = w / waves_n_, bn = w % waves_n_;
    for (std::uint64_t c = 0; c < chunks(); ++c) {
      ChunkSet s{w, c, {}, 0};
      const std::uint64_t slot = set_index(w, c) % cfg.ring_slots;
      const std::uint64_t base = slot * slot_bytes_;
      unsigned rows = 0, cols = 0;
      for (unsigned x = 0; x < cfg.px; ++x) {
        const std::uint64_t mi = bm * cfg.px + x;
        if (mi >= l_.tiles_m()) continue;
        ++rows;
        s.blocks.push_back({true, mi, c, l_.a_block_addr(mi, c), base + x * l_.a_block_bytes(),
                            l_.a_block_bytes()});
      }
      for (unsigned y = 0; y < cfg.py; ++y) {
        const std::uint64_t nj = bn * cfg.py + y;
        if (nj >= l_.tiles_n()) continue;
        ++cols;
        s.blocks.push_back({false, nj, c, l_.b_block_addr(nj, c),
                            base + cfg.px * l_.a_block_bytes() + y * l_.b_block_bytes(),
                            l_.b_block_bytes()});
      }
      s.consumers = rows * cols;
      sets_.push_back(std::move(s));
    }
  }
}

std::uint64_t SocPlan::a_sram(unsigned pe, std::uint64_t k, std::uint64_t c) const {
  const PeTile& t = pe_tiles_.at(pe).at(k);
  const std::uint64_t slot = set_index(t.wave, c) % cfg_.ring_slots;
  return slot * slot_bytes_ + t.x * l_.a_block_bytes();
}

std::uint64_t SocPlan::b_sram(unsigned pe, std::uint64_t k, std::uint64_t c) const {
  const PeTile& t = pe_tiles_.at(pe).at(k);
  const std::uint64_t slot = set_index(t.wave, c) % cfg_.ring_slots;
  return slot * slot_bytes_ + cfg_.px * l_.a_block_bytes() + t.y * l_.b_block_bytes();
}

std::uint64_t SocPlan::c_slot(unsigned pe, std::uint64_t k) const {
  return c_base_ + (2ull * pe + k % 2) * l_.c_tile_bytes();
}

TileDesc SocPlan::writeback(unsigned pe, std::uint64_t k) const {
  const PeTile& t = pe_tiles_.at(pe).at(k);
  const std::uint64_t row = 4ull * l_.array.cols;
  return {DmaDir::SramToDram, c_slot(pe, k), l_.c_tile_addr(t.mi, t.nj), l_.array.rows, row, row,
          4 * l_.shape.n};
}

TileDesc SocPlan::pe_store(unsigned pe, std::uint64_t k, std::uint64_t lmem_src) const {
  // In the PE DMA's terms the "sram" side is its LMEM and the "dram" side is
  // the far side (the shared SRAM through the mesh). The C slot is contiguous.
  const std::uint64_t row = 4ull * l_.array.cols;
  return {DmaDir::SramToDram, lmem_src, c_slot(pe, k), l_.array.rows, row, row, row};
}

bool SocPlan::full_waves() const {
  return l_.tiles_m() % cfg_.px == 0 && l_.tiles_n() % cfg_.py == 0;
}

std::uint64_t SocPlan::dram_bytes_exact() const {
  std::uint64_t b = l_.c_bytes();
  for (const ChunkSet& s : sets_)
    for (const StagedBlock& blk : s.blocks) b += blk.bytes;
  return b;
}

std::uint64_t SocPlan::dram_bytes_formula() const {
  return l_.a_packed_bytes() * l_.tiles_n() / cfg_.py + l_.b_packed_bytes() * l_.tiles_m() / cfg_.px +
         l_.c_bytes();
}

}  // namespace apm
