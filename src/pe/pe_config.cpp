#include "pe/pe_config.h"

#include <stdexcept>
#include <string>

#include "pe/lmem_alloc.h"

namespace apm {

PeBuffers plan_pe(const PeConfig& cfg, const GemmLayout& job) {
  job.validate();
  if (job.array.rows != cfg.array.rows || job.array.cols != cfg.array.cols || job.kt != cfg.kt)
    throw std::invalid_argument("PE: GEMM layout does not match the PE's array/Kt");
  if (cfg.n_buf == 0 || cfg.n_out == 0) throw std::invalid_argument("PE: n_buf and n_out must be >= 1");
  cfg.lmem.validate();

  // Layer (a): the configuration as a whole.
  if (cfg.lmem_required() > cfg.lmem.capacity_bytes)
    throw std::invalid_argument(
        "PE: LMEM too small: " + std::to_string(cfg.n_buf) + " x (A " +
        std::to_string(job.a_block_bytes()) + " B + B " + std::to_string(job.b_block_bytes()) +
        " B) + " + std::to_string(cfg.n_out) + " x C " + std::to_string(job.c_tile_bytes()) +
        " B = " + std::to_string(cfg.lmem_required()) + " B > capacity " +
        std::to_string(cfg.lmem.capacity_bytes) + " B");

  // Layer (b): every buffer individually, line-aligned.
  LmemAllocator alloc(cfg.lmem.capacity_bytes, cfg.lmem.bank_bytes);
  PeBuffers b;
  for (unsigned i = 0; i < cfg.n_buf; ++i) {
    b.a.push_back(alloc.alloc("A" + std::to_string(i), job.a_block_bytes()));
    b.b.push_back(alloc.alloc("B" + std::to_string(i), job.b_block_bytes()));
  }
  for (unsigned o = 0; o < cfg.n_out; ++o)
    b.out.push_back(alloc.alloc("C" + std::to_string(o), job.c_tile_bytes()));
  return b;
}

}  // namespace apm
