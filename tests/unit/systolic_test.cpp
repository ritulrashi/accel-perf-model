// Systolic timing contract and functional accumulate (Phase 2 note §2.2, §5).
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "compute/systolic.h"

using apm::ArrayShape;

TEST(Systolic, TileLatencyFormula) {
  EXPECT_EQ(apm::tile_latency(ArrayShape{1, 1}, 1), 2u);
  EXPECT_EQ(apm::tile_latency(ArrayShape{4, 16}, 64), 86u);
  EXPECT_EQ(apm::tile_latency(ArrayShape{16, 16}, 256), 302u);
  EXPECT_EQ(apm::fill_drain(ArrayShape{4, 16}), 22u);
}

// R=2, C=3, Kt=2, feed at f=0. Hand-written table of MAC cycles f+k+i+j.
TEST(Systolic, MacScheduleMatchesHandTable) {
  const ArrayShape a{2, 3};
  const unsigned expected[2][2][3] = {
      {{0, 1, 2}, {1, 2, 3}},  // k = 0: row 0, row 1
      {{1, 2, 3}, {2, 3, 4}},  // k = 1
  };
  for (unsigned k = 0; k < 2; ++k)
    for (unsigned i = 0; i < 2; ++i)
      for (unsigned j = 0; j < 3; ++j) EXPECT_EQ(apm::mac_cycle(0, i, j, k), expected[k][i][j]);
  EXPECT_EQ(apm::drain_start(a, 0, 2), 5u);  // one cycle after the last MAC (4)
  EXPECT_EQ(apm::tile_done(a, 0, 2), 7u);    // drain [5,7) = R cycles
  EXPECT_EQ(apm::tile_done(a, 10, 2), 17u);
}

TEST(Systolic, AccumulateAddsAcrossChunks) {
  apm::Accumulators acc(ArrayShape{2, 2});
  // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A*B = [[19,22],[43,50]]
  const std::int8_t a[] = {1, 2, 3, 4};
  const std::int8_t b[] = {5, 6, 7, 8};
  acc.accumulate(a, b, 2);
  EXPECT_EQ(acc.values(), (std::vector<std::int32_t>{19, 22, 43, 50}));
  acc.accumulate(a, b, 2);  // a second chunk adds
  EXPECT_EQ(acc.values(), (std::vector<std::int32_t>{38, 44, 86, 100}));
  acc.clear();
  EXPECT_EQ(acc.values(), (std::vector<std::int32_t>{0, 0, 0, 0}));
}

TEST(Systolic, NegativeExtremes) {
  apm::Accumulators acc(ArrayShape{1, 1});
  const std::int8_t a[] = {-128, -128, 127};
  const std::int8_t b[] = {-128, 127, -128};
  acc.accumulate(a, b, 3);  // 16384 - 16256 - 16256
  EXPECT_EQ(acc.values()[0], -16128);
}

TEST(Systolic, IntAccumulationBound) {
  const std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  EXPECT_LE(std::int64_t{apm::kMaxIntK} * 16384, max32);
  EXPECT_GT((std::int64_t{apm::kMaxIntK} + 1) * 16384, max32);
}
