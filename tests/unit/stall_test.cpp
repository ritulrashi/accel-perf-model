// Stall classifier on synthetic traces whose per-category counts were worked
// out by hand (Phase 3 design note §3.2, §5 "Classifier").
#include <gtest/gtest.h>

#include <stdexcept>

#include "soc/stall.h"

using apm::BurstRecord;
using apm::PeTrace;
using apm::StallCat;
using apm::StallCounts;

namespace {

std::uint64_t get(const StallCounts& n, StallCat c) { return n[static_cast<std::size_t>(c)]; }

// One tile, one chunk: Kt=4, F=2.
//  [0,3)  waiting for staging (submitted at 0)          dram 3
//  [3,8)  load burst: network [3,5), conflict [5,6), gap -> dma [6,8)
//  [8,14) feed [8,12) + tail [12,14)                     compute 6
//  [14,17) store burst: gap 14, network 15, gap 16        dma 2, network 1
//  [17,20) nothing left                                  idle 3
PeTrace one_tile() {
  PeTrace t;
  t.kt = 4;
  t.fill_drain = 2;
  t.chunks = 1;
  t.n_out = 2;
  apm::StepTrace s;
  s.a = {0, 3};
  s.b = {0, 3};
  s.loads_submit = 3;
  s.loaded = 8;
  s.feed = 8;
  s.tile = 0;
  s.load_bursts = {BurstRecord{3, 8, {{3, 5, StallCat::Network}, {5, 6, StallCat::SramConflict}}}};
  t.steps.push_back(s);
  apm::TileTrace tt;
  tt.done = 14;
  tt.store_submit = 14;
  tt.store_done = 17;
  tt.store_bursts = {BurstRecord{14, 17, {{15, 16, StallCat::Network}}}};
  t.tiles.push_back(tt);
  return t;
}

}  // namespace

TEST(Stall, EveryCategoryOnHandTrace) {
  const StallCounts n = apm::classify_pe(one_tile(), 20, 0);
  EXPECT_EQ(get(n, StallCat::Compute), 6u);
  EXPECT_EQ(get(n, StallCat::Dram), 3u);
  EXPECT_EQ(get(n, StallCat::SramConflict), 1u);
  EXPECT_EQ(get(n, StallCat::Network), 3u);
  EXPECT_EQ(get(n, StallCat::Dma), 4u);
  EXPECT_EQ(get(n, StallCat::Idle), 3u);
}

TEST(Stall, UnsubmittedStagingIsIdle) {
  PeTrace t = one_tile();
  t.steps[0].a = {2, 3};  // ring slot busy until 2: cycles 0,1 are synchronization
  const StallCounts n = apm::classify_pe(t, 20, 0);
  EXPECT_EQ(get(n, StallCat::Idle), 3u + 2u);
  EXPECT_EQ(get(n, StallCat::Dram), 1u);
}

// n_out = 1, two one-chunk tiles; tile 1 is loaded at 2 but must wait for
// the store of tile 0 (done 9) before it may feed.
//  0 dma (load burst), [1,4) compute, [4,9) store of tile 0: gap 4, network
//  [5,8), gap 8; [9,12) compute; [12,14) store of tile 1 (gaps) dma; 14 idle.
TEST(Stall, OutputBufferWaitFollowsTheBlockingStore) {
  PeTrace t;
  t.kt = 2;
  t.fill_drain = 1;
  t.chunks = 1;
  t.n_out = 1;
  apm::StepTrace s0, s1;
  s0.a = s0.b = s1.a = s1.b = {0, 0};
  s0.loads_submit = 0;
  s0.loaded = 1;
  s0.feed = 1;
  s0.tile = 0;
  s0.load_bursts = {BurstRecord{0, 1, {}}};
  s1.loads_submit = 1;
  s1.loaded = 2;
  s1.feed = 9;
  s1.tile = 1;
  s1.load_bursts = {BurstRecord{1, 2, {}}};
  t.steps = {s0, s1};
  apm::TileTrace t0, t1;
  t0.done = 4;
  t0.store_submit = 4;
  t0.store_done = 9;
  t0.store_bursts = {BurstRecord{4, 9, {{5, 8, StallCat::Network}}}};
  t1.done = 12;
  t1.store_submit = 12;
  t1.store_done = 14;
  t1.store_bursts = {BurstRecord{12, 14, {}}};
  t.tiles = {t0, t1};
  const StallCounts n = apm::classify_pe(t, 15, 0);
  EXPECT_EQ(get(n, StallCat::Compute), 6u);
  EXPECT_EQ(get(n, StallCat::Network), 3u);
  EXPECT_EQ(get(n, StallCat::Dma), 5u);
  EXPECT_EQ(get(n, StallCat::Idle), 1u);
}

TEST(Stall, LoadedButNotFedWithoutReasonThrows) {
  PeTrace t = one_tile();
  t.steps[0].feed = 10;  // loaded at 8, nothing blocks it, yet not fed until 10
  t.tiles[0].done = 16;
  t.tiles[0].store_submit = 16;
  t.tiles[0].store_done = 19;
  t.tiles[0].store_bursts = {BurstRecord{16, 19, {}}};
  EXPECT_THROW(apm::classify_pe(t, 20, 0), std::logic_error);
}

TEST(Stall, StagedButNotSubmittedThrows) {
  PeTrace t = one_tile();
  t.steps[0].loads_submit = 5;  // staged at 3 but the PE submitted its loads only at 5
  EXPECT_THROW(apm::classify_pe(t, 20, 0), std::logic_error);
}
