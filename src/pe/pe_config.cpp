#include "pe/pe_config.h"

#include <stdexcept>
#include <string>

#include "pe/lmem_alloc.h"

namespace apm {

PeBuffers plan_pe(const PeConfig& cfg, const GemmLayout& job) {
  job.validate();
  if (job.array.rows != cfg.array.rows || job.array.cols != cfg.array.cols || job.kt != cfg.kt)
    throw std::invalid_argument("PE: GEMM layout does not match the PE's array/Kt");
  return plan_pe_buffers(cfg);
}

PeBuffers plan_pe_buffers(const PeConfig& cfg) {
  if (cfg.array.rows == 0 || cfg.array.cols == 0 || cfg.kt == 0)
    throw std::invalid_argument("PE: empty array or chunk");
  const std::uint64_t a_sz = std::uint64_t{cfg.array.rows} * cfg.kt;
  const std::uint64_t b_sz = cfg.kt * cfg.array.cols;
  const std::uint64_t c_sz = 4ull * cfg.array.rows * cfg.array.cols;
  if (cfg.n_buf == 0 || cfg.n_out == 0) throw std::invalid_argument("PE: n_buf and n_out must be >= 1");
  cfg.lmem.validate();

  // Layer (a): the configuration as a whole.
  if (cfg.lmem_required() > cfg.lmem.capacity_bytes)
    throw std::invalid_argument(
        "PE: LMEM too small: " + std::to_string(cfg.n_buf) + " x (A " +
        std::to_string(a_sz) + " B + B " + std::to_string(b_sz) +
        " B) + " + std::to_string(cfg.n_out) + " x C " + std::to_string(c_sz) +
        " B = " + std::to_string(cfg.lmem_required()) + " B > capacity " +
        std::to_string(cfg.lmem.capacity_bytes) + " B");

  // Layer (b): every buffer individually, line-aligned.
  LmemAllocator alloc(cfg.lmem.capacity_bytes, cfg.lmem.bank_bytes);
  PeBuffers b;
  for (unsigned i = 0; i < cfg.n_buf; ++i) {
    b.a.push_back(alloc.alloc("A" + std::to_string(i), a_sz));
    b.b.push_back(alloc.alloc("B" + std::to_string(i), b_sz));
  }
  for (unsigned o = 0; o < cfg.n_out; ++o)
    b.out.push_back(alloc.alloc("C" + std::to_string(o), c_sz));
  return b;
}

}  // namespace apm
