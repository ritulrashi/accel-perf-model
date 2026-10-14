// GEMM layout/packing, LMEM allocator, and PE capacity checks (Phase 2 note
// §2.4, §3.1, §5 P2-T7). Pure C++, no simulation.
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pe/gemm_layout.h"
#include "pe/lmem_alloc.h"
#include "pe/pe_config.h"

using apm::GemmLayout;
using apm::PeConfig;

namespace {

// The P2 test configuration: 4x16 array, Kt=64, two operand and two output buffers.
PeConfig p2_config(std::uint64_t lmem_bytes) {
  PeConfig c;
  c.array = {4, 16};
  c.kt = 64;
  c.n_buf = 2;
  c.n_out = 2;
  c.lmem.capacity_bytes = lmem_bytes;
  return c;
}

GemmLayout layout(std::uint64_t m, std::uint64_t n, std::uint64_t k, unsigned r, unsigned c,
                  std::uint64_t kt) {
  GemmLayout l;
  l.shape = {m, n, k};
  l.array = {r, c};
  l.kt = kt;
  l.b_base = 1 << 16;
  l.c_base = 1 << 17;
  return l;
}

}  // namespace

TEST(GemmLayout, Counts) {
  const auto l = layout(8, 32, 128, 2, 16, 32);
  EXPECT_EQ(l.chunks(), 4u);
  EXPECT_EQ(l.tiles_m(), 4u);
  EXPECT_EQ(l.tiles_n(), 2u);
  EXPECT_EQ(l.steps(), 32u);
  EXPECT_EQ(l.a_block_bytes(), 64u);
  EXPECT_EQ(l.b_block_bytes(), 512u);
  EXPECT_EQ(l.c_tile_bytes(), 128u);
}

TEST(GemmLayout, Addresses) {
  const auto l = layout(8, 32, 128, 2, 16, 32);
  EXPECT_EQ(l.a_block_addr(0, 0), 0u);
  EXPECT_EQ(l.a_block_addr(1, 2), (1u * 4 + 2) * 64);
  EXPECT_EQ(l.b_block_addr(1, 3), (1u << 16) + (1u * 4 + 3) * 512);
  // C tile (mi=3, nj=1): row 3*2 = 6, column 16, row-major INT32 with N = 32.
  EXPECT_EQ(l.c_tile_addr(3, 1), (1u << 17) + 4u * (6 * 32 + 16));
}

TEST(GemmLayout, PackingPlacesEveryElementOnce) {
  const auto l = layout(4, 32, 32, 2, 16, 16);
  std::vector<std::int8_t> a(4 * 32), b(32 * 32);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::int8_t>(i * 7 + 1);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::int8_t>(i * 13 + 5);
  const auto pa = l.pack_a(a), pb = l.pack_b(b);
  // A[i][k] lives in block (i / R, k / Kt) at (i % R) * Kt + k % Kt.
  for (std::uint64_t i = 0; i < 4; ++i)
    for (std::uint64_t k = 0; k < 32; ++k)
      ASSERT_EQ(static_cast<std::int8_t>(pa[l.a_block_addr(i / 2, k / 16) + (i % 2) * 16 + k % 16]),
                a[i * 32 + k]);
  // B[k][j] lives in block (j / C, k / Kt) at (k % Kt) * C + j % C.
  for (std::uint64_t k = 0; k < 32; ++k)
    for (std::uint64_t j = 0; j < 32; ++j)
      ASSERT_EQ(static_cast<std::int8_t>(
                    pb[l.b_block_addr(j / 16, k / 16) - l.b_base + (k % 16) * 16 + j % 16]),
                b[k * 32 + j]);
}

TEST(GemmLayout, RejectsUnsupportedShapes) {
  EXPECT_THROW(layout(6, 16, 64, 4, 16, 64).validate(), std::invalid_argument);   // M % R
  EXPECT_THROW(layout(4, 24, 64, 4, 16, 64).validate(), std::invalid_argument);   // N % C
  EXPECT_THROW(layout(4, 16, 96, 4, 16, 64).validate(), std::invalid_argument);   // K % Kt
  EXPECT_THROW(layout(4, 16, 24, 4, 16, 8).validate(), std::invalid_argument);    // Kt % 16
  EXPECT_THROW(layout(4, 6, 64, 4, 6, 64).validate(), std::invalid_argument);     // C % 4
  EXPECT_NO_THROW(layout(4, 16, 64, 4, 16, 64).validate());
}

// P2-T7: INT32 accumulation bound.
TEST(GemmLayout, P2T7_AccumulationRange) {
  EXPECT_THROW(layout(1, 4, 131072, 1, 4, 16).validate(), std::invalid_argument);
  EXPECT_NO_THROW(layout(1, 4, 131056, 1, 4, 16).validate());
}

TEST(LmemAllocator, AlignsAndRefusesOverflow) {
  apm::LmemAllocator a(100, 16);
  EXPECT_EQ(a.alloc("x", 20), 0u);
  EXPECT_EQ(a.alloc("y", 16), 32u);  // aligned up from 20
  try {
    a.alloc("z", 64);  // would need [48, 112)
    FAIL() << "expected overflow";
  } catch (const std::invalid_argument& e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("'z'"), std::string::npos);
    EXPECT_NE(m.find("x[0+20]"), std::string::npos);
    EXPECT_NE(m.find("y[32+16]"), std::string::npos);
  }
}

// P2-T7: 2 x (256 + 1024) + 2 x 256 = 3072 B.
TEST(PeConfig, P2T7_CapacityBoundary) {
  const auto job = layout(4, 16, 64, 4, 16, 64);
  EXPECT_EQ(p2_config(3072).lmem_required(), 3072u);
  const auto bufs = apm::plan_pe(p2_config(3072), job);
  EXPECT_EQ(bufs.a, (std::vector<std::uint64_t>{0, 1280}));
  EXPECT_EQ(bufs.b, (std::vector<std::uint64_t>{256, 1536}));
  EXPECT_EQ(bufs.out, (std::vector<std::uint64_t>{2560, 2816}));
  EXPECT_THROW(apm::plan_pe(p2_config(3071), job), std::invalid_argument);
}

TEST(PeConfig, RejectsMismatchedJob) {
  EXPECT_THROW(apm::plan_pe(p2_config(1 << 14), layout(4, 16, 64, 4, 16, 32)), std::invalid_argument);
}

// Phase 4 §7: LMEM for P4-T2 (FP32, 4x16 array, Kt = 64) is exactly 2 x (1024 + 4096) + 2 x 256.
TEST(PeConfig, P4Fp32CapacityBoundary) {
  PeConfig c = p2_config(10752);
  c.prec = apm::Precision::Fp32;
  EXPECT_EQ(c.lmem_required(), 10752u);
  EXPECT_EQ(c.feed_cycles(), 256u);
  EXPECT_NO_THROW(apm::plan_pe_buffers(c));
  c.lmem.capacity_bytes = 10751;
  EXPECT_THROW(apm::plan_pe_buffers(c), std::invalid_argument);
}
