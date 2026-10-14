#include "vec/vu_plan.h"

#include <stdexcept>
#include <string>

#include "pe/lmem_alloc.h"

namespace apm {

std::uint64_t vmem_required(const VuConfig& c, const VuJob& j) {
  return c.n_vbuf * 4 * j.n + c.n_vout * j.n * j.elem() + j.scale_bytes();
}

VuBuffers plan_vu(const VuConfig& c, const VuJob& j) {
  auto fail = [&](const std::string& why) {
    throw std::invalid_argument("VU job " + std::to_string(j.rows) + " rows x " + std::to_string(j.n) + ": " + why);
  };
  if (c.lanes == 0 || (c.lanes & (c.lanes - 1)) != 0) fail("V must be a power of two");
  if (c.n_vbuf == 0 || c.n_vout == 0) fail("n_vbuf and n_vout must be >= 1");
  if (j.rows == 0 || j.n == 0) fail("empty job");
  if (j.kt == 0 || j.kt % 16 != 0) fail("Kt must be a multiple of 16");
  if (j.n % j.kt != 0) fail("n must be a multiple of Kt (packed P layout)");
  if (j.r_pack == 0) fail("R must be >= 1");
  if (j.s_base % 16 != 0 || j.p_base % 16 != 0 || j.scale_base % 16 != 0) fail("bases must be 16-byte aligned");
  c.vmem.validate();
  if (vmem_required(c, j) > c.vmem.capacity_bytes)
    fail("VMEM too small: " + std::to_string(c.n_vbuf) + " x in " + std::to_string(4 * j.n) + " B + " +
         std::to_string(c.n_vout) + " x out " + std::to_string(j.n * j.elem()) + " B + scales " +
         std::to_string(j.scale_bytes()) + " B = " + std::to_string(vmem_required(c, j)) + " B > capacity " +
         std::to_string(c.vmem.capacity_bytes) + " B");
  LmemAllocator alloc(c.vmem.capacity_bytes, c.vmem.bank_bytes);
  VuBuffers b;
  for (unsigned i = 0; i < c.n_vbuf; ++i) b.in.push_back(alloc.alloc("S" + std::to_string(i), 4 * j.n));
  for (unsigned i = 0; i < c.n_vout; ++i) b.out.push_back(alloc.alloc("P" + std::to_string(i), j.n * j.elem()));
  if (j.writes_scales()) b.scales = alloc.alloc("scales", j.scale_bytes());
  return b;
}

TileDesc vu_load(const VuJob& j, std::uint64_t vmem_dst, std::uint64_t row) {
  const std::uint64_t bytes = 4 * j.n;
  return {DmaDir::DramToSram, j.s_base + row * bytes, vmem_dst, 1, bytes, bytes, bytes};
}

TileDesc vu_store(const VuJob& j, std::uint64_t vmem_src, std::uint64_t row) {
  const std::uint64_t seg = j.kt * j.elem();
  const std::uint64_t mi = row / j.r_pack, i = row % j.r_pack;
  return {DmaDir::SramToDram, vmem_src, j.p_base + (mi * j.chunks() * j.r_pack + i) * seg,
          static_cast<std::uint32_t>(j.chunks()), seg, seg, j.r_pack * seg};
}

TileDesc vu_scale_store(const VuJob& j, std::uint64_t vmem_src) {
  const std::uint64_t b = j.scale_bytes();
  return {DmaDir::SramToDram, vmem_src, j.scale_base, 1, b, b, b};
}

std::uint64_t vu_dram_bytes(const VuJob& j) { return j.rows * j.n * (4 + j.elem()) + j.scale_bytes(); }

}  // namespace apm
