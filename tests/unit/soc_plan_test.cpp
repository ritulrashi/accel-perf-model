// Scheduler plan: mapping, chunk-sets with dedup, SRAM layout, DRAM bytes
// (Phase 3 design note §2, §5 "Scheduler mapping").
#include <gtest/gtest.h>

#include <stdexcept>

#include "soc/soc_plan.h"

using apm::GemmLayout;
using apm::GridConfig;
using apm::SocPlan;

namespace {

GridConfig grid(unsigned px, unsigned py, std::uint64_t sram_bytes) {
  GridConfig g;
  g.px = px;
  g.py = py;
  g.ring_slots = 2;
  g.pe.array = {4, 16};
  g.pe.kt = 64;
  g.pe.lmem.capacity_bytes = 1 << 14;
  g.sram.capacity_bytes = sram_bytes;
  return g;
}

GemmLayout gemm(std::uint64_t m, std::uint64_t n, std::uint64_t k) {
  GemmLayout l;
  l.shape = {m, n, k};
  l.array = {4, 16};
  l.kt = 64;
  l.b_base = 1 << 16;
  l.c_base = 1 << 17;
  return l;
}

}  // namespace

// 2x2 grid, 16x64x128: tiles 4x4 -> 4 waves, nc = 2 -> 8 chunk-sets of
// 2 A (256 B) + 2 B (1024 B) blocks, 4 consumers each.
TEST(SocPlan, FullWaves2x2) {
  const SocPlan p(grid(2, 2, 1 << 16), gemm(16, 64, 128));
  EXPECT_EQ(p.waves(), 4u);
  EXPECT_EQ(p.chunk_sets(), 8u);
  for (std::uint64_t q = 0; q < 8; ++q) {
    EXPECT_EQ(p.chunk_set(q).blocks.size(), 4u);
    EXPECT_EQ(p.chunk_set(q).consumers, 4u);
  }
  // 8 x (2*256 + 2*1024) + C 4*16*64 = 20480 + 4096.
  EXPECT_EQ(p.dram_bytes_exact(), 24576u);
  // |A| 2048 * tiles_n/Py 2 + |B| 8192 * tiles_m/Px 2 + |C| 4096.
  EXPECT_EQ(p.dram_bytes_formula(), 24576u);
  EXPECT_TRUE(p.full_waves());
}

TEST(SocPlan, TilesAndAddresses) {
  const SocPlan p(grid(2, 2, 1 << 16), gemm(16, 64, 128));
  // PE 2 = (x=1, y=0): waves (0,0),(0,1),(1,0),(1,1) give (mi,nj) = (1,0),(1,2),(3,0),(3,2).
  const auto& t = p.tiles_of(2);
  ASSERT_EQ(t.size(), 4u);
  EXPECT_EQ(t[1].mi, 1u);
  EXPECT_EQ(t[1].nj, 2u);
  EXPECT_EQ(t[2].mi, 3u);
  EXPECT_EQ(t[2].nj, 0u);
  // slot = 2*256 + 2*1024 = 2560; ring of 2 -> C slots from 5120.
  EXPECT_EQ(p.slot_bytes(), 2560u);
  EXPECT_EQ(p.sram_required(), 5120u + 8u * 256u);
  EXPECT_EQ(p.a_sram(2, 1, 1), 2560u + 256u);       // wave 1, chunk 1 -> q 3 -> slot 1; x = 1
  EXPECT_EQ(p.b_sram(3, 0, 0), 2u * 256u + 1024u);  // q 0, slot 0; y = 1
  EXPECT_EQ(p.c_slot(2, 1), 5120u + 5u * 256u);
}

TEST(SocPlan, PartialWave) {
  const SocPlan p(grid(1, 2, 1 << 16), gemm(4, 16, 64));
  EXPECT_EQ(p.tiles_of(0).size(), 1u);
  EXPECT_EQ(p.tiles_of(1).size(), 0u);
  ASSERT_EQ(p.chunk_sets(), 1u);
  EXPECT_EQ(p.chunk_set(0).blocks.size(), 2u);
  EXPECT_EQ(p.chunk_set(0).consumers, 1u);
  EXPECT_FALSE(p.full_waves());
  EXPECT_EQ(p.dram_bytes_exact(), 256u + 1024u + 256u);
}

TEST(SocPlan, SramCapacity) {
  EXPECT_NO_THROW(SocPlan(grid(2, 2, 7168), gemm(16, 64, 128)));
  EXPECT_THROW(SocPlan(grid(2, 2, 7167), gemm(16, 64, 128)), std::invalid_argument);
}

TEST(SocPlan, SramNodeMustBeInTheGrid) {
  GridConfig g = grid(2, 2, 1 << 16);
  g.sram_node = {2, 0};
  EXPECT_THROW(SocPlan(g, gemm(16, 64, 128)), std::invalid_argument);
}
