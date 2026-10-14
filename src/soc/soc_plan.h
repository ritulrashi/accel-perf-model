// Static plan for running one GEMM on the PE grid (Phase 3 design note §2).
// Pure C++: tile -> PE mapping in waves, chunk-sets with deduplicated operand
// blocks, the shared-SRAM layout (staging ring + per-PE C slots) with its
// capacity check, and the DRAM byte counts used by the stall-stack and
// roofline cross-checks.
#pragma once
#include <cstdint>
#include <vector>

#include "dma/dma_types.h"
#include "mem/sram_timing.h"
#include "noc/mesh_timing.h"
#include "pe/gemm_layout.h"
#include "pe/pe_config.h"

namespace apm {

struct GridConfig {
  unsigned px = 1, py = 1;     // PE(x, y) at mesh router (x, y); PE id = x * py + y
  Node sram_node{0, 0};
  unsigned ring_slots = 2;     // D
  PeConfig pe;
  SramParams sram;             // shared SRAM
  DmaParams gdma;              // global DMA (DRAM <-> SRAM)
  Cycle hop = 2;               // mesh hop latency
  unsigned link_bytes = 16;    // mesh link bandwidth

  unsigned pes() const { return px * py; }
  MeshParams mesh() const { return MeshParams{px, py, hop, link_bytes}; }
};

struct StagedBlock {
  bool is_a;
  std::uint64_t index;  // mi for A, nj for B
  std::uint64_t chunk;
  std::uint64_t dram_addr, sram_addr, bytes;
};

struct ChunkSet {
  std::uint64_t wave, chunk;
  std::vector<StagedBlock> blocks;  // A blocks (by x), then B blocks (by y)
  unsigned consumers;               // tiles of the wave present in the matrix
};

struct PeTile {
  std::uint64_t mi, nj, wave;
  unsigned x, y;  // the PE
};

class SocPlan {
 public:
  // Throws std::invalid_argument on an invalid shape, grid or capacity.
  SocPlan(const GridConfig& cfg, const GemmLayout& dram_layout);

  unsigned pes() const { return cfg_.pes(); }
  Node node(unsigned pe) const { return Node{pe / cfg_.py, pe % cfg_.py}; }
  std::uint64_t waves() const { return waves_m_ * waves_n_; }
  std::uint64_t chunks() const { return l_.chunks(); }

  const std::vector<PeTile>& tiles_of(unsigned pe) const { return pe_tiles_.at(pe); }
  std::uint64_t chunk_sets() const { return sets_.size(); }
  const ChunkSet& chunk_set(std::uint64_t q) const { return sets_.at(q); }
  std::uint64_t set_index(std::uint64_t wave, std::uint64_t chunk) const { return wave * chunks() + chunk; }

  // Shared-SRAM addresses for PE `pe`'s k-th tile, chunk c.
  std::uint64_t a_sram(unsigned pe, std::uint64_t k, std::uint64_t c) const;
  std::uint64_t b_sram(unsigned pe, std::uint64_t k, std::uint64_t c) const;
  std::uint64_t c_slot(unsigned pe, std::uint64_t k) const;  // C slot k mod 2
  // Write-back descriptor for PE `pe`'s k-th tile (SRAM C slot -> row-major DRAM).
  TileDesc writeback(unsigned pe, std::uint64_t k) const;
  // PE store descriptor (LMEM -> shared-SRAM C slot), PE-DMA view.
  TileDesc pe_store(unsigned pe, std::uint64_t k, std::uint64_t lmem_src) const;

  std::uint64_t slot_bytes() const { return slot_bytes_; }
  std::uint64_t sram_required() const { return sram_required_; }
  bool full_waves() const;
  // Bytes the run must move to/from DRAM: every staged block plus C.
  std::uint64_t dram_bytes_exact() const;
  // §2.2 closed form, valid for full waves: |A|*tiles_n/Py + |B|*tiles_m/Px + |C|.
  std::uint64_t dram_bytes_formula() const;

  const GemmLayout& layout() const { return l_; }
  const GridConfig& config() const { return cfg_; }

 private:
  GridConfig cfg_;
  GemmLayout l_;
  std::uint64_t waves_m_ = 0, waves_n_ = 0;
  std::uint64_t slot_bytes_ = 0, c_base_ = 0, sram_required_ = 0;
  std::vector<std::vector<PeTile>> pe_tiles_;
  std::vector<ChunkSet> sets_;
};

}  // namespace apm
