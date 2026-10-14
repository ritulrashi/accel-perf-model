// Mesh link timing against the hand cases in docs/phase3-design.md §5
// (h = 2, L = 16, 64 B packets -> ser 4).
#include <gtest/gtest.h>

#include "noc/mesh_timing.h"

using apm::MeshParams;
using apm::MeshTiming;
using apm::Node;

namespace {
MeshTiming mesh3x2() { return MeshTiming(MeshParams{3, 2, 2, 16}); }
}  // namespace

TEST(MeshTiming, XYRouteAndHops) {
  auto m = mesh3x2();
  EXPECT_EQ(m.hops(Node{0, 0}, Node{2, 1}), 3u);
  EXPECT_EQ(m.route(Node{0, 0}, Node{2, 1}).size(), 3u);
  EXPECT_EQ(m.ser(64), 4u);
  EXPECT_EQ(m.ser(1), 1u);
}

TEST(MeshTiming, SinglePacketThreeHops) {
  auto m = mesh3x2();
  EXPECT_EQ(m.send(0, Node{0, 0}, Node{2, 1}, 4), 10u);  // heads at 2, 4, 6; tail +4
}

TEST(MeshTiming, TwoPacketsShareALink) {
  auto m = mesh3x2();
  EXPECT_EQ(m.send(0, Node{0, 0}, Node{1, 0}, 4), 6u);
  EXPECT_EQ(m.send(0, Node{0, 0}, Node{1, 0}, 4), 10u);  // link busy [0,4)
}

TEST(MeshTiming, DownstreamLinkBusy) {
  auto m = mesh3x2();
  EXPECT_EQ(m.send(0, Node{1, 0}, Node{2, 0}, 4), 6u);   // reserves (1,0)->(2,0) [0,4)
  EXPECT_EQ(m.send(0, Node{0, 0}, Node{2, 0}, 4), 10u);  // head at (1,0) at 2, waits to 4
}

TEST(MeshTiming, LocalDeliveryTakesOneCycle) {
  auto m = mesh3x2();
  EXPECT_EQ(m.send(7, Node{1, 1}, Node{1, 1}, 4), 8u);
}
