// Directed Phase 1 tests. Every expected value was derived by hand in
// docs/phase1-design.md §6 before the simulator was first run. Each check
// prints "cycles: <test> <key> expected=<n> measured=<n>" (or "count:" for
// counters); scripts/phase1_exit.sh collects those lines.
#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "test/scenarios.h"

using apm::run_scenario;
using apm::ScenarioResult;

namespace {

void check(const char* kind, const ScenarioResult& r, const std::string& key,
           std::uint64_t expected) {
  const auto it = r.values.find(key);
  ASSERT_NE(it, r.values.end()) << "missing value " << key;
  std::cout << kind << ": " << testing::UnitTest::GetInstance()->current_test_info()->name()
            << " " << key << " expected=" << expected << " measured=" << it->second << "\n";
  EXPECT_EQ(it->second, expected) << key;
}

void cycles(const ScenarioResult& r, const std::string& key, std::uint64_t expected) {
  check("cycles", r, key, expected);
}

void count(const ScenarioResult& r, const std::string& key, std::uint64_t expected) {
  check("count", r, key, expected);
}

void expect_data_ok(const ScenarioResult& r) {
  EXPECT_TRUE(r.data_checked);
  EXPECT_TRUE(r.data_ok) << "destination memory differs from the functional reference";
}

}  // namespace

TEST(Phase1, T1_SingleTransfer) {
  const auto r = run_scenario("t1");
  ASSERT_TRUE(r.complete);
  cycles(r, "dma0.d0.done", 48);
  count(r, "dram.row_misses", 1);
  count(r, "dram.row_hits", 3);
  count(r, "dram.bus_busy_cycles", 16);
  count(r, "sram.conflict_line_cycles", 0);
  count(r, "dma0.d0.bursts", 4);
  expect_data_ok(r);
}

TEST(Phase1, T1b_OneBufferIsLatencyBound) {
  const auto r = run_scenario("t1b");
  ASSERT_TRUE(r.complete);
  cycles(r, "dma0.d0.done", 84);
  expect_data_ok(r);
}

TEST(Phase1, T1c_ReverseDirection) {
  const auto r = run_scenario("t1c");
  ASSERT_TRUE(r.complete);
  cycles(r, "dma0.d0.done", 48);
  count(r, "dram.bytes_written", 256);
  expect_data_ok(r);
}

TEST(Phase1, T2_TwoDmasShareOneChannel) {
  const auto r = run_scenario("t2");
  ASSERT_TRUE(r.complete);
  cycles(r, "dma0.d0.done", 64);
  cycles(r, "dma1.d0.done", 60);
  count(r, "dram.bus_busy_cycles", 32);
  count(r, "dram.row_misses", 2);
  count(r, "dram.row_hits", 6);
  count(r, "sram.conflict_line_cycles", 0);
  expect_data_ok(r);
}

TEST(Phase1, T2b_QueueBackpressure) {
  const auto r = run_scenario("t2b");
  ASSERT_TRUE(r.complete);
  cycles(r, "dma0.d0.done", 54);
  count(r, "dram.backpressure_cycles", 35);
  expect_data_ok(r);
}

TEST(Phase1, T3a_SramBankConflict) {
  const auto r = run_scenario("t3a");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgs0.r0.response", 2);
  cycles(r, "tgs1.r0.response", 3);
  count(r, "sram.conflict_line_cycles", 1);
}

TEST(Phase1, T3b_SramNoConflict) {
  const auto r = run_scenario("t3b");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgs0.r0.response", 2);
  cycles(r, "tgs1.r0.response", 2);
  count(r, "sram.conflict_line_cycles", 0);
}

TEST(Phase1, T3c_SramIntraRequestSerialization) {
  const auto r = run_scenario("t3c");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgs0.r0.response", 3);
  count(r, "sram.conflict_line_cycles", 0);
}

TEST(Phase1, T4a_RowHit) {
  const auto r = run_scenario("t4a");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r0.response", 34);
  cycles(r, "tgd0.r1.issued", 100);
  cycles(r, "tgd0.r1.response", 114);
  count(r, "dram.row_hits", 1);
}

TEST(Phase1, T4b_RowMiss) {
  const auto r = run_scenario("t4b");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r0.response", 34);
  cycles(r, "tgd0.r1.issued", 100);
  cycles(r, "tgd0.r1.response", 134);
  count(r, "dram.row_misses", 2);
}

TEST(Phase1, T4c_SameBankBackToBack) {
  const auto r = run_scenario("t4c");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r0.response", 34);
  cycles(r, "tgd0.r1.issued", 1);
  cycles(r, "tgd0.r1.response", 58);
}

TEST(Phase1, T4d_OtherBankBackToBack) {
  const auto r = run_scenario("t4d");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r0.response", 34);
  cycles(r, "tgd0.r1.issued", 1);
  cycles(r, "tgd0.r1.response", 38);
}

// Phase 5 §6: bank-address map, hand-derived in docs/phase5-design.md.
TEST(Phase5, T5a_XorMapSplitsT4c) {
  const auto r = run_scenario("t5a");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r0.response", 34);
  cycles(r, "tgd0.r1.response", 38);
}

TEST(Phase5, T5b_XorMapPermutesConflicts) {
  const auto r = run_scenario("t5b");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r0.response", 34);
  cycles(r, "tgd0.r1.response", 58);
}

TEST(Phase5, T5c_StrideOfFourRowsRowMap) {
  const auto r = run_scenario("t5c");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r1.response", 58);
  cycles(r, "tgd0.r2.response", 82);
  cycles(r, "tgd0.r3.response", 106);
}

TEST(Phase5, T5d_StrideOfFourRowsXorMap) {
  const auto r = run_scenario("t5d");
  ASSERT_TRUE(r.complete);
  cycles(r, "tgd0.r1.response", 38);
  cycles(r, "tgd0.r2.response", 42);
  cycles(r, "tgd0.r3.response", 46);
}

TEST(Phase1, F2D_DramToSramStrided) {
  const auto r = run_scenario("f2d_d2s");
  ASSERT_TRUE(r.complete);
  count(r, "dma0.d0.bursts", 8);
  expect_data_ok(r);
}

TEST(Phase1, F2D_SramToDramStrided) {
  const auto r = run_scenario("f2d_s2d");
  ASSERT_TRUE(r.complete);
  count(r, "dma0.d0.bursts", 8);
  expect_data_ok(r);
}
