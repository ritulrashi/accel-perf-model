// What a PE works on (Phase 3 design note §2.3). The PE controller asks the
// job where each step's operands come from, whether they may be loaded yet,
// and where each finished tile goes. Phase 2's single-PE GEMM straight out
// of DRAM is one implementation (DramGemmJob); the Phase 3 grid scheduler
// provides another. Pure C++.
#pragma once
#include <cstdint>

#include "common/cycle_types.h"
#include "dma/dma_types.h"
#include "pe/gemm_layout.h"

namespace apm {

class PeJob {
 public:
  virtual ~PeJob() = default;

  virtual std::uint64_t tiles() const = 0;
  virtual std::uint64_t chunks() const = 0;  // nc: every tile has the same number of chunks
  std::uint64_t steps() const { return tiles() * chunks(); }

  // Far-side (DRAM or shared-SRAM) address of step s's A / B block.
  virtual std::uint64_t a_src(std::uint64_t step) const = 0;
  virtual std::uint64_t b_src(std::uint64_t step) const = 0;
  // DMA descriptor that stores tile t from LMEM address `lmem_src`.
  virtual TileDesc store(std::uint64_t tile, std::uint64_t lmem_src) const = 0;

  virtual bool operands_ready(std::uint64_t /*step*/) const { return true; }
  virtual bool store_allowed(std::uint64_t /*tile*/) const { return true; }

  // Called from inside TLM backward paths: implementations must only record
  // and notify, never call back into a DMA.
  virtual void on_loads_done(std::uint64_t /*step*/, Cycle /*at*/) {}
  virtual void on_store_done(std::uint64_t /*tile*/, Cycle /*at*/) {}
};

// Phase 2: one PE, operands packed in DRAM, C written row-major to DRAM.
class DramGemmJob : public PeJob {
 public:
  explicit DramGemmJob(const GemmLayout& l) : l_(l) {}

  std::uint64_t tiles() const override { return l_.tiles(); }
  std::uint64_t chunks() const override { return l_.chunks(); }
  std::uint64_t a_src(std::uint64_t s) const override {
    const std::uint64_t t = s / chunks();
    return l_.a_block_addr(t / l_.tiles_n(), s % chunks());
  }
  std::uint64_t b_src(std::uint64_t s) const override {
    const std::uint64_t t = s / chunks();
    return l_.b_block_addr(t % l_.tiles_n(), s % chunks());
  }
  TileDesc store(std::uint64_t t, std::uint64_t lmem_src) const override {
    const std::uint64_t row = 4ull * l_.array.cols;
    return {DmaDir::SramToDram, lmem_src, l_.c_tile_addr(t / l_.tiles_n(), t % l_.tiles_n()),
            l_.array.rows, row, row, 4 * l_.shape.n};
  }

 private:
  GemmLayout l_;
};

}  // namespace apm
