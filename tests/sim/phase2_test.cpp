// Directed Phase 2 tests. Every expected value was derived by hand in
// docs/phase2-design.md §5 before the simulator was first run. Each check
// prints "cycles: <test> <key> expected=<n> measured=<n>"; the exit script
// collects those lines. Every GEMM test also checks C against a plain
// triple loop written here (independent of the model's data path).
#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "test/gemm_run.h"
#include "test/scenarios.h"

using apm::GemmRun;
using apm::GemmRunResult;

namespace {

void cycles(const std::map<std::string, std::uint64_t>& v, const std::string& key,
            std::uint64_t expected) {
  const auto it = v.find(key);
  ASSERT_NE(it, v.end()) << "missing value " << key;
  std::cout << "cycles: " << testing::UnitTest::GetInstance()->current_test_info()->name() << " "
            << key << " expected=" << expected << " measured=" << it->second << "\n";
  EXPECT_EQ(it->second, expected) << key;
}

// P2 test configuration (design note §5).
GemmRun p2(std::uint64_t m, std::uint64_t n, std::uint64_t k) {
  GemmRun r;
  r.dram.banks = 4;
  r.dram.row_bytes = 1024;
  r.dram.bus_bytes_per_cycle = 32;
  r.dram.t_hit = 10;
  r.dram.t_miss = 10;
  r.dram.queue_depth = 16;
  r.dram.burst_bytes = 64;
  r.pe.array = {4, 16};
  r.pe.kt = 64;
  r.pe.n_buf = 2;
  r.pe.n_out = 2;
  r.pe.lmem.banks = 8;
  r.pe.lmem.bank_bytes = 16;
  r.pe.lmem.t_sram = 2;
  r.pe.lmem.capacity_bytes = 16384;
  r.pe.dma.burst_bytes = 64;
  r.pe.dma.max_bursts_in_flight = 8;
  r.shape = {m, n, k};
  r.auto_layout = false;
  r.a_base = 0;
  r.b_base = 64 * 1024;
  r.c_base = 128 * 1024;
  // Deterministic operand data (a simple LCG); values do not affect timing.
  std::uint32_t x = 12345;
  auto next = [&x] {
    x = x * 1103515245u + 12345u;
    return static_cast<std::int8_t>(x >> 16);
  };
  r.a.resize(m * k);
  r.b.resize(k * n);
  for (auto& v : r.a) v = next();
  for (auto& v : r.b) v = next();
  return r;
}

GemmRunResult run_and_check(const GemmRun& run) {
  GemmRunResult r = apm::run_gemm(run);
  EXPECT_TRUE(r.complete);
  const auto& s = run.shape;
  for (std::uint64_t i = 0; i < s.m; ++i)
    for (std::uint64_t j = 0; j < s.n; ++j) {
      std::int64_t ref = 0;
      for (std::uint64_t k = 0; k < s.k; ++k)
        ref += std::int64_t{run.a[i * s.k + k]} * std::int64_t{run.b[k * s.n + j]};
      if (r.c[i * s.n + j] != ref) {
        ADD_FAILURE() << "C[" << i << "][" << j << "] = " << r.c[i * s.n + j] << ", expected " << ref;
        return r;
      }
    }
  return r;
}

}  // namespace

TEST(Phase2, T1_OneTileOneChunk) {
  const auto r = run_and_check(p2(4, 16, 64));
  cycles(r.values, "pe.s0.loaded", 52);
  cycles(r.values, "pe.s0.feed", 52);
  cycles(r.values, "pe.t0.done", 138);
  cycles(r.values, "pe.done", 158);
}

TEST(Phase2, T2_DoubleBufferedComputeBound) {
  const auto r = run_and_check(p2(4, 16, 256));
  const std::uint64_t loaded[] = {52, 92, 168, 232}, feed[] = {52, 116, 180, 244};
  for (int s = 0; s < 4; ++s) {
    cycles(r.values, "pe.s" + std::to_string(s) + ".loaded", loaded[s]);
    cycles(r.values, "pe.s" + std::to_string(s) + ".feed", feed[s]);
  }
  cycles(r.values, "pe.t0.done", 330);
  cycles(r.values, "pe.done", 350);
}

TEST(Phase2, T3_SingleBuffer) {
  auto run = p2(4, 16, 256);
  run.pe.n_buf = 1;
  const auto r = run_and_check(run);
  const std::uint64_t loaded[] = {52, 168, 284, 400}, feed[] = {52, 168, 284, 400};
  for (int s = 0; s < 4; ++s) {
    cycles(r.values, "pe.s" + std::to_string(s) + ".loaded", loaded[s]);
    cycles(r.values, "pe.s" + std::to_string(s) + ".feed", feed[s]);
  }
  cycles(r.values, "pe.t0.done", 486);
  cycles(r.values, "pe.done", 506);
}

TEST(Phase2, T4_DoubleBufferedMemoryBound) {
  auto run = p2(4, 16, 256);
  run.dram.bus_bytes_per_cycle = 16;
  const auto r = run_and_check(run);
  const std::uint64_t loaded[] = {92, 172, 252, 332}, feed[] = {92, 172, 252, 332};
  for (int s = 0; s < 4; ++s) {
    cycles(r.values, "pe.s" + std::to_string(s) + ".loaded", loaded[s]);
    cycles(r.values, "pe.s" + std::to_string(s) + ".feed", feed[s]);
  }
  cycles(r.values, "pe.t0.done", 418);
  cycles(r.values, "pe.done", 446);
}

TEST(Phase2, T5_TwoOutputTiles) {
  const auto r = run_and_check(p2(8, 16, 64));
  cycles(r.values, "pe.s0.loaded", 52);
  cycles(r.values, "pe.s1.loaded", 92);
  cycles(r.values, "pe.s0.feed", 52);
  cycles(r.values, "pe.s1.feed", 138);
  cycles(r.values, "pe.t0.done", 138);
  cycles(r.values, "pe.t0.store_done", 158);
  cycles(r.values, "pe.t1.done", 224);
  cycles(r.values, "pe.done", 244);
}

TEST(Phase2, T5b_OneOutputBuffer) {
  auto run = p2(8, 16, 64);
  run.pe.n_out = 1;
  const auto r = run_and_check(run);
  cycles(r.values, "pe.t0.store_done", 158);
  cycles(r.values, "pe.s1.feed", 158);
  cycles(r.values, "pe.t1.done", 244);
  cycles(r.values, "pe.done", 264);
}

TEST(Phase2, T6_DmaStreamsAcrossDescriptors) {
  const auto r = apm::run_scenario("p2t6");
  ASSERT_TRUE(r.complete);
  EXPECT_TRUE(r.data_checked && r.data_ok);
  cycles(r.values, "dma0.d0.done", 20);
  cycles(r.values, "dma0.d1.done", 28);
}

TEST(Phase2, T7_ExactCapacityStillRuns) {
  auto run = p2(4, 16, 64);
  run.pe.lmem.capacity_bytes = 3072;
  const auto r = run_and_check(run);
  cycles(r.values, "pe.done", 158);
}
