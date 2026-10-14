#include <gtest/gtest.h>

#include <vector>

#include "common/rr_arbiter.h"

using apm::RrArbiter;

TEST(RrArbiter, StartsAtPortZero) {
  RrArbiter a(3);
  EXPECT_EQ(a.order(), (std::vector<unsigned>{0, 1, 2}));
}

// The T2 sequence: c0 [0,1], c1 [1,0], c2 [0,1], c3 [1,0].
TEST(RrArbiter, T2_OrderAlternatesWhenBothRequestEveryCycle) {
  RrArbiter a(2);
  std::vector<std::vector<unsigned>> seen;
  for (int c = 0; c < 4; ++c) {
    auto o = a.order();
    seen.push_back(o);
    a.granted_first(o.front());
  }
  EXPECT_EQ(seen, (std::vector<std::vector<unsigned>>{{0, 1}, {1, 0}, {0, 1}, {1, 0}}));
}

TEST(RrArbiter, PointerSkipsPastTheFirstGrantedPort) {
  RrArbiter a(4);
  a.granted_first(2);
  EXPECT_EQ(a.order(), (std::vector<unsigned>{3, 0, 1, 2}));
  a.granted_first(3);
  EXPECT_EQ(a.pointer(), 0u);
}
