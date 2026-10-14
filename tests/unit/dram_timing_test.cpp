// DRAM formulas against the hand derivations in docs/phase1-design.md §6.
// The accept cycles fed in here are the ones derived in the note; the SystemC
// tests (tests/sim) check that the TLM plumbing produces those accept cycles.
#include <gtest/gtest.h>

#include <set>
#include <stdexcept>
#include <utility>

#include "mem/dram_timing.h"

using apm::DramParams;
using apm::DramTiming;

namespace {

void expect_sched(const apm::DramSchedule& s, bool hit, apm::Cycle start, apm::Cycle col,
                  apm::Cycle ds, apm::Cycle de) {
  EXPECT_EQ(s.hit, hit);
  EXPECT_EQ(s.start, start);
  EXPECT_EQ(s.col_issue, col);
  EXPECT_EQ(s.data_start, ds);
  EXPECT_EQ(s.data_end, de);
}

}  // namespace

TEST(DramTiming, AddressMapRotatesRowsAcrossBanks) {
  DramTiming t{DramParams{}};
  EXPECT_EQ(t.map(0).bank, 0u);
  EXPECT_EQ(t.map(0).row, 0u);
  EXPECT_EQ(t.map(1023).bank, 0u);
  EXPECT_EQ(t.map(1024).bank, 1u);
  EXPECT_EQ(t.map(1024).row, 0u);
  EXPECT_EQ(t.map(4096).bank, 0u);  // T4b: bank (4096/1024)%4 = 0
  EXPECT_EQ(t.map(4096).row, 1u);
}

TEST(DramTiming, RejectsIllegalRequests) {
  DramTiming t{DramParams{}};
  EXPECT_THROW(t.check_legal(0, 0), std::invalid_argument);
  EXPECT_THROW(t.check_legal(0, 8), std::invalid_argument);     // below the 16 B granule
  EXPECT_THROW(t.check_legal(8, 16), std::invalid_argument);    // not 16-byte aligned
  EXPECT_THROW(t.check_legal(0, 128), std::invalid_argument);   // > burst
  EXPECT_THROW(t.check_legal(1008, 32), std::invalid_argument); // crosses a row
  EXPECT_NO_THROW(t.check_legal(960, 64));
}

TEST(DramTiming, RejectsInconsistentParams) {
  DramParams p;
  p.t_miss = 5;  // < t_hit
  EXPECT_THROW(DramTiming{p}, std::invalid_argument);
}

// T1: four 64 B bursts to bank 0 row 0, accepted at 0..3.
TEST(DramTiming, T1_OpenRowStreamIsBusLimited) {
  DramTiming t{DramParams{}};
  expect_sched(t.schedule(0, 0, 64), false, 0, 20, 30, 34);
  expect_sched(t.schedule(1, 64, 64), true, 24, 24, 34, 38);
  expect_sched(t.schedule(2, 128, 64), true, 28, 28, 38, 42);
  expect_sched(t.schedule(3, 192, 64), true, 32, 32, 42, 46);
  EXPECT_EQ(t.row_misses(), 1u);
  EXPECT_EQ(t.row_hits(), 3u);
  EXPECT_EQ(t.bus_busy_cycles(), 16u);
}

// T2: two DMAs, banks 0 and 1, in round-robin grant order.
TEST(DramTiming, T2_TwoStreamsShareTheBus) {
  DramTiming t{DramParams{}};
  expect_sched(t.schedule(0, 0, 64), false, 0, 20, 30, 34);       // D0 b0
  expect_sched(t.schedule(0, 1024, 64), false, 0, 20, 34, 38);    // D1 b0
  expect_sched(t.schedule(1, 1088, 64), true, 24, 24, 38, 42);    // D1 b1
  expect_sched(t.schedule(1, 64, 64), true, 24, 24, 42, 46);      // D0 b1
  expect_sched(t.schedule(2, 128, 64), true, 28, 28, 46, 50);     // D0 b2
  expect_sched(t.schedule(2, 1152, 64), true, 28, 28, 50, 54);    // D1 b2
  expect_sched(t.schedule(3, 1216, 64), true, 32, 32, 54, 58);    // D1 b3
  expect_sched(t.schedule(3, 192, 64), true, 32, 32, 58, 62);     // D0 b3
  EXPECT_EQ(t.bus_busy_cycles(), 32u);
}

// T2b: with Q=2, b2 and b3 are accepted at 34 and 38.
TEST(DramTiming, T2b_LateAcceptance) {
  DramTiming t{DramParams{}};
  t.schedule(0, 0, 64);
  t.schedule(1, 64, 64);
  expect_sched(t.schedule(34, 128, 64), true, 34, 34, 44, 48);
  expect_sched(t.schedule(38, 192, 64), true, 38, 38, 48, 52);
}

// T1c: writes accepted at 2..5 (after the SRAM reads).
TEST(DramTiming, T1c_Writes) {
  DramTiming t{DramParams{}};
  expect_sched(t.schedule(2, 0, 64), false, 2, 22, 32, 36);
  expect_sched(t.schedule(3, 64, 64), true, 26, 26, 36, 40);
  expect_sched(t.schedule(4, 128, 64), true, 30, 30, 40, 44);
  expect_sched(t.schedule(5, 192, 64), true, 34, 34, 44, 48);
}

TEST(DramTiming, T4a_RowHitAfterIdle) {
  DramTiming t{DramParams{}};
  expect_sched(t.schedule(0, 0, 64), false, 0, 20, 30, 34);
  expect_sched(t.schedule(100, 64, 64), true, 100, 100, 110, 114);
}

TEST(DramTiming, T4b_RowConflictAfterIdle) {
  DramTiming t{DramParams{}};
  t.schedule(0, 0, 64);
  expect_sched(t.schedule(100, 4096, 64), false, 100, 120, 130, 134);
}

TEST(DramTiming, T4c_SameBankBackToBack) {
  DramTiming t{DramParams{}};
  t.schedule(0, 0, 64);
  expect_sched(t.schedule(1, 4096, 64), false, 24, 44, 54, 58);
}

TEST(DramTiming, T4d_OtherBankBackToBack) {
  DramTiming t{DramParams{}};
  t.schedule(0, 0, 64);
  expect_sched(t.schedule(1, 1024, 64), false, 1, 21, 34, 38);
}

// Phase 3 (BUGS #8): requests narrower than or misaligned to a wide bus
// occupy every beat they touch. W = 32: 16 B -> 1 beat; 32 B at +16 -> 2 beats.
TEST(DramTiming, PartialBeatsOnAWideBus) {
  DramParams p;
  p.bus_bytes_per_cycle = 32;
  p.t_miss = p.t_hit;  // isolate the beat count
  DramTiming t{p};
  expect_sched(t.schedule(0, 0, 16), false, 0, 0, 10, 11);      // 1 beat
  expect_sched(t.schedule(100, 16, 32), true, 100, 100, 110, 112);  // beats 0 and 1
  EXPECT_EQ(t.bus_busy_cycles(), 3u);
}


// Phase 5 §6: the xor map. Hand rows: row index 4m -> bank m mod 4; 5 -> bank 0.
TEST(DramTiming, XorMapHandRowsAndBijection) {
  DramParams p;
  p.bank_map = apm::BankMap::Xor;
  p.capacity_bytes = 1ull << 32;
  DramTiming t{p};
  for (unsigned m = 0; m < 8; ++m) EXPECT_EQ(t.map(4096ull * m).bank, m % 4) << m;
  EXPECT_EQ(t.map(5120).bank, 0u);
  EXPECT_EQ(t.map(5120).row, 1u);
  // Bijection on 4096 row indices: (bank, row) pairs are all distinct and invert.
  std::set<std::pair<unsigned, std::uint64_t>> seen;
  for (std::uint64_t ri = 0; ri < 4096; ++ri) {
    const auto a = t.map(ri * p.row_bytes);
    EXPECT_TRUE(seen.insert({a.bank, a.row}).second) << ri;
    EXPECT_EQ(a.row * p.banks + (a.bank ^ (a.row % p.banks)), ri);
  }
  p.banks = 3;
  EXPECT_THROW(p.validate(), std::invalid_argument);
}
