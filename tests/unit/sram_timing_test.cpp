// SRAM formulas against docs/phase1-design.md §6 (T3, and the SRAM side of T1/T1c).
#include <gtest/gtest.h>

#include <stdexcept>

#include "mem/sram_timing.h"

using apm::SramParams;
using apm::SramTiming;

TEST(SramTiming, BankMapping) {
  SramTiming t{SramParams{}};
  EXPECT_EQ(t.bank_of(0), 0u);
  EXPECT_EQ(t.bank_of(16), 1u);
  EXPECT_EQ(t.bank_of(128), 0u);   // line 8
  EXPECT_EQ(t.bank_of(4096), 0u);  // line 256
}

TEST(SramTiming, RejectsIllegalRequests) {
  SramTiming t{SramParams{}};
  EXPECT_THROW(t.check_legal(0, 0), std::invalid_argument);
  EXPECT_THROW(t.check_legal(8, 16), std::invalid_argument);
  EXPECT_THROW(t.check_legal(0, 24), std::invalid_argument);
  EXPECT_NO_THROW(t.check_legal(16, 256));
}

TEST(SramTiming, T3a_SameBankSameCycleSerializes) {
  SramTiming t{SramParams{}};
  auto a = t.schedule(0, 0, 16);
  auto b = t.schedule(0, 128, 16);
  EXPECT_EQ(a.response, 2u);
  EXPECT_EQ(b.response, 3u);
  EXPECT_EQ(a.conflict_line_cycles, 0u);
  EXPECT_EQ(b.conflict_line_cycles, 1u);
  EXPECT_EQ(t.conflict_line_cycles(), 1u);
  EXPECT_EQ(a.conflict_delay, 0u);  // Phase 3: per-request delay for the stall stack
  EXPECT_EQ(b.conflict_delay, 1u);
}

TEST(SramTiming, T3b_DifferentBanksDoNotConflict) {
  SramTiming t{SramParams{}};
  EXPECT_EQ(t.schedule(0, 0, 16).response, 2u);
  EXPECT_EQ(t.schedule(0, 16, 16).response, 2u);
  EXPECT_EQ(t.conflict_line_cycles(), 0u);
}

TEST(SramTiming, T3c_IntraRequestSerializationIsNotAConflict) {
  SramTiming t{SramParams{}};
  auto s = t.schedule(0, 0, 256);
  EXPECT_EQ(s.response, 3u);
  EXPECT_EQ(s.lines, 16u);
  EXPECT_EQ(s.conflict_line_cycles, 0u);
  EXPECT_EQ(s.conflict_delay, 0u);
}

TEST(SramTiming, T1_WritesNeverCollide) {
  SramTiming t{SramParams{}};
  EXPECT_EQ(t.schedule(34, 0, 64).response, 36u);
  EXPECT_EQ(t.schedule(38, 64, 64).response, 40u);
  EXPECT_EQ(t.schedule(42, 128, 64).response, 44u);
  EXPECT_EQ(t.schedule(46, 192, 64).response, 48u);
  EXPECT_EQ(t.conflict_line_cycles(), 0u);
}

TEST(SramTiming, T1c_ReadsBackToBack) {
  SramTiming t{SramParams{}};
  EXPECT_EQ(t.schedule(0, 0, 64).response, 2u);
  EXPECT_EQ(t.schedule(1, 64, 64).response, 3u);
  EXPECT_EQ(t.schedule(2, 128, 64).response, 4u);  // banks 0-3 free since cycle 1
  EXPECT_EQ(t.schedule(3, 192, 64).response, 5u);
  EXPECT_EQ(t.conflict_line_cycles(), 0u);
}
