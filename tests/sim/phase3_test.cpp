// Directed Phase 3 tests. Expected values were derived by hand in
// docs/phase3-design.md §5 before the simulator was first run. Every run also
// enforces the in-model stall-stack assertions (src/soc/accel_run.h).
#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "soc/accel_run.h"

using apm::AccelRun;
using apm::AccelRunResult;

namespace {

void check(const char* kind, const AccelRunResult& r, const std::string& key, std::uint64_t expected) {
  const auto it = r.values.find(key);
  ASSERT_NE(it, r.values.end()) << "missing value " << key;
  std::cout << kind << ": " << testing::UnitTest::GetInstance()->current_test_info()->name() << " "
            << key << " expected=" << expected << " measured=" << it->second << "\n";
  EXPECT_EQ(it->second, expected) << key;
}

// P3 configuration (design note §5).
AccelRun p3(unsigned px, unsigned py, apm::Node sram_node, std::uint64_t m, std::uint64_t n,
            std::uint64_t k) {
  AccelRun r;
  r.cfg.dram.banks = 4;
  r.cfg.dram.row_bytes = 1024;
  r.cfg.dram.bus_bytes_per_cycle = 32;
  r.cfg.dram.t_hit = 10;
  r.cfg.dram.t_miss = 10;
  r.cfg.dram.queue_depth = 16;
  apm::GridConfig& g = r.cfg.grid;
  g.px = px;
  g.py = py;
  g.sram_node = sram_node;
  g.ring_slots = 2;
  g.sram.banks = 8;
  g.sram.bank_bytes = 16;
  g.sram.t_sram = 2;
  g.sram.capacity_bytes = 64 * 1024;
  g.gdma.max_bursts_in_flight = 8;
  g.pe.array = {4, 16};
  g.pe.kt = 64;
  g.pe.n_buf = 2;
  g.pe.n_out = 2;
  g.pe.lmem.capacity_bytes = 16 * 1024;
  g.pe.dma.max_bursts_in_flight = 8;
  g.hop = 2;
  g.link_bytes = 16;
  r.shape = {m, n, k};
  std::uint32_t x = 4242;
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

AccelRunResult run_and_check(const AccelRun& run) {
  AccelRunResult r = apm::run_accel(run);  // throws if any stall-stack assertion fails
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

void stack(const AccelRunResult& r, unsigned pe, std::uint64_t compute, std::uint64_t dram,
           std::uint64_t network, std::uint64_t dma, std::uint64_t idle, std::uint64_t conflict) {
  const std::string p = "pe" + std::to_string(pe) + ".stall.";
  check("count", r, p + "compute", compute);
  check("count", r, p + "dram", dram);
  check("count", r, p + "network", network);
  check("count", r, p + "dma", dma);
  check("count", r, p + "idle", idle);
  check("count", r, p + "sram_conflict", conflict);
}

}  // namespace

TEST(Phase3, T1_SinglePeAtSramNode) {
  const auto r = run_and_check(p3(1, 1, {0, 0}, 4, 16, 64));
  check("cycles", r, "t_end", 192);
  stack(r, 0, 86, 52, 7, 27, 20, 0);
}

TEST(Phase3, T2_OneHopAndAnIdlePe) {
  const auto r = run_and_check(p3(1, 2, {0, 1}, 4, 16, 64));
  check("cycles", r, "t_end", 272);
  stack(r, 0, 86, 52, 65, 49, 20, 0);
  stack(r, 1, 0, 0, 0, 0, 272, 0);
}

TEST(Phase3, T3_Grid2x2FunctionalAndDramBytes) {
  const auto r = run_and_check(p3(2, 2, {0, 0}, 16, 64, 128));
  check("count", r, "dram_bytes", 24576);
  check("count", r, "full_waves", 1);
}
