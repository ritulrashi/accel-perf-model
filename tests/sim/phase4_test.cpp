// Directed Phase 4 tests. Expected values were derived by hand in
// docs/phase4-design.md §7 before the simulator was first run. Every run also
// enforces the in-model assertions (stall-stack sums and the q-aware compute
// cross-check for the grid; busy/sum/DRAM-byte checks for the VU).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "compute/numerics.h"
#include "soc/accel_run.h"
#include "vec/softmax.h"
#include "vec/vu_run.h"

using apm::AccelRun;
using apm::AccelRunResult;
using apm::Precision;

namespace {

void check(const char* kind, const std::map<std::string, std::uint64_t>& v, const std::string& key,
           std::uint64_t expected) {
  const auto it = v.find(key);
  ASSERT_NE(it, v.end()) << "missing value " << key;
  std::cout << kind << ": " << testing::UnitTest::GetInstance()->current_test_info()->name() << " "
            << key << " expected=" << expected << " measured=" << it->second << "\n";
  EXPECT_EQ(it->second, expected) << key;
}

// P3 configuration (Phase 3 design note §5), grid 1x1, GEMM 4x16x64.
AccelRun p3_1x1(Precision prec) {
  AccelRun r;
  r.cfg.dram.banks = 4;
  r.cfg.dram.row_bytes = 1024;
  r.cfg.dram.bus_bytes_per_cycle = 32;
  r.cfg.dram.t_hit = 10;
  r.cfg.dram.t_miss = 10;
  r.cfg.dram.queue_depth = 16;
  apm::GridConfig& g = r.cfg.grid;
  g.px = 1;
  g.py = 1;
  g.sram_node = {0, 0};
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
  g.pe.prec = prec;
  g.hop = 2;
  g.link_bytes = 16;
  r.shape = {4, 16, 64};
  // Operands: small integers k/8 (exact in every mode) from a fixed LCG.
  std::uint32_t x = 4242;
  auto next = [&x] {
    x = x * 1103515245u + 12345u;
    return static_cast<std::int8_t>(x >> 16);
  };
  const unsigned e = apm::elem_bytes(prec);
  auto fill = [&](std::vector<std::int8_t>& i8, std::vector<std::uint8_t>& raw, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) {
      const std::int8_t v = next();
      if (prec == Precision::Int8) {
        i8.push_back(v);
        continue;
      }
      const float f = static_cast<float>(v) / 8.0f;
      const std::uint32_t w = prec == Precision::Fp16   ? apm::f32_to_fp16(f)
                              : prec == Precision::Bf16 ? apm::f32_to_bf16(f)
                                                        : apm::f32_to_bits(f);
      for (unsigned b = 0; b < e; ++b) raw.push_back(static_cast<std::uint8_t>(w >> (8 * b)));
    }
  };
  fill(r.a, r.a_bytes, 4 * 64);
  fill(r.b, r.b_bytes, 64 * 16);
  return r;
}

// Functional check of the grid run against a plain loop of the §2.2 spec.
void check_result(const AccelRun& run, const AccelRunResult& r) {
  const auto& s = run.shape;
  const Precision p = run.cfg.grid.pe.prec;
  const unsigned e = apm::elem_bytes(p);
  const std::uint64_t rows = run.m_valid ? run.m_valid : s.m;
  ASSERT_EQ(r.c.size(), rows * s.n);
  for (std::uint64_t i = 0; i < rows; ++i)
    for (std::uint64_t j = 0; j < s.n; ++j) {
      std::uint32_t want;
      if (p == Precision::Int8) {
        std::int32_t acc = 0;
        for (std::uint64_t k = 0; k < s.k; ++k) acc += std::int32_t{run.a[i * s.k + k]} * run.b[k * s.n + j];
        want = static_cast<std::uint32_t>(acc);
      } else {
        float acc = 0.0f;
        for (std::uint64_t k = 0; k < s.k; ++k) {
          const float prod = apm::decode_float(p, &run.a_bytes[(i * s.k + k) * e]) *
                             apm::decode_float(p, &run.b_bytes[(k * s.n + j) * e]);
          acc = acc + prod;
        }
        want = apm::f32_to_bits(acc);
      }
      ASSERT_EQ(static_cast<std::uint32_t>(r.c[i * s.n + j]), want) << "C[" << i << "][" << j << "]";
    }
}

void stack(const AccelRunResult& r, std::uint64_t compute, std::uint64_t dram, std::uint64_t network,
           std::uint64_t dma, std::uint64_t idle) {
  check("count", r.values, "pe0.stall.compute", compute);
  check("count", r.values, "pe0.stall.dram", dram);
  check("count", r.values, "pe0.stall.network", network);
  check("count", r.values, "pe0.stall.dma", dma);
  check("count", r.values, "pe0.stall.idle", idle);
  check("count", r.values, "pe0.stall.sram_conflict", 0);
}

}  // namespace

TEST(Phase4, T1_Fp16) {
  const AccelRun run = p3_1x1(Precision::Fp16);
  const AccelRunResult r = apm::run_accel(run);
  check_result(run, r);
  check("cycles", r.values, "t_end", 316);
  stack(r, 150, 92, 7, 47, 20);
  check("count", r.values, "dram_bytes", 2816);
}

TEST(Phase4, T2_Fp32) {
  const AccelRun run = p3_1x1(Precision::Fp32);
  const AccelRunResult r = apm::run_accel(run);
  check_result(run, r);
  check("cycles", r.values, "t_end", 564);
  stack(r, 278, 172, 7, 87, 20);
  check("count", r.values, "dram_bytes", 5376);
}

TEST(Phase4, T3_Bf16TimingEqualsFp16) {
  const AccelRun run = p3_1x1(Precision::Bf16);
  const AccelRunResult r = apm::run_accel(run);
  check_result(run, r);
  check("cycles", r.values, "t_end", 316);
  stack(r, 150, 92, 7, 47, 20);
  check("count", r.values, "dram_bytes", 2816);
}

TEST(Phase4, T4_MaskedWriteBack) {
  AccelRun run = p3_1x1(Precision::Int8);
  run.m_valid = 1;
  const AccelRunResult r = apm::run_accel(run);
  check_result(run, r);
  check("cycles", r.values, "t_end", 186);
  stack(r, 86, 52, 7, 27, 14);
  check("count", r.values, "dram_bytes", 1344);
}

namespace {

// Configuration PV (design note §7): DRAM as P2, VMEM 8 x 16 B, VU DMA M = 8,
// V = 16, t_vu = 4, two input and two output buffers.
struct VuCase {
  apm::DramParams dram;
  apm::VuConfig cfg;
  apm::VuJob job;
  std::vector<std::uint8_t> image;
};

VuCase pv(Precision prec, std::uint64_t rows, std::uint64_t n) {
  VuCase c;
  c.dram.banks = 4;
  c.dram.row_bytes = 1024;
  c.dram.bus_bytes_per_cycle = 32;
  c.dram.t_hit = 10;
  c.dram.t_miss = 10;
  c.dram.queue_depth = 16;
  c.cfg.lanes = 16;
  c.cfg.t_vu = 4;
  c.cfg.n_vbuf = 2;
  c.cfg.n_vout = 2;
  c.cfg.vmem.banks = 8;
  c.cfg.vmem.bank_bytes = 16;
  c.cfg.vmem.t_sram = 2;
  c.cfg.dma.max_bursts_in_flight = 8;
  c.job.rows = rows;
  c.job.n = n;
  c.job.spec.prec = prec;
  c.job.spec.alpha = 0.125f;
  c.job.s_base = 0;
  c.job.p_base = 1024;
  c.job.r_pack = 4;
  c.job.kt = 64;
  c.cfg.vmem.capacity_bytes = apm::vmem_required(c.cfg, c.job);
  c.image.assign(2048, 0);
  // S: row 0 all zeros, other rows j - 7*r (INT32 or FP32).
  for (std::uint64_t r = 1; r < rows; ++r)
    for (std::uint64_t j = 0; j < n; ++j) {
      const std::int32_t v = static_cast<std::int32_t>(j) - 7 * static_cast<std::int32_t>(r);
      const std::uint32_t w =
          prec == Precision::Int8 ? static_cast<std::uint32_t>(v) : apm::f32_to_bits(static_cast<float>(v));
      std::memcpy(&c.image[(r * n + j) * 4], &w, 4);
    }
  return c;
}

// The P row expected at each packed segment, from softmax_row on the same S row.
void check_p_rows(const VuCase& c, const apm::VuRunResult& r) {
  apm::SoftmaxSpec spec = c.job.spec;
  spec.lanes = c.cfg.lanes;
  const unsigned e = c.job.elem();
  const std::uint64_t seg = c.job.kt * e;
  std::vector<bool> written(c.image.size(), false);
  for (std::uint64_t row = 0; row < c.job.rows; ++row) {
    std::vector<std::uint8_t> p(c.job.n * e);
    apm::softmax_row(spec, row, &c.image[row * c.job.n * 4], c.job.n, p.data());
    for (std::uint64_t s = 0; s < c.job.chunks(); ++s) {
      const std::uint64_t at = c.job.p_base + ((row / 4) * c.job.chunks() * 4 + s * 4 + row % 4) * seg;
      for (std::uint64_t b = 0; b < seg; ++b) {
        ASSERT_EQ(r.image[at + b], p[s * seg + b]) << "row " << row << " segment " << s << " byte " << b;
        written[at + b] = true;
      }
    }
  }
  // Nothing else in the P region may change.
  for (std::uint64_t a = c.job.p_base; a < c.image.size(); ++a) {
    if (!written[a]) {
      ASSERT_EQ(r.image[a], 0) << "stray write at " << a;
    }
  }
}

void vu_stack(const apm::VuRunResult& r, std::uint64_t load, std::uint64_t busy, std::uint64_t store) {
  check("count", r.values, "vu.load", load);
  check("count", r.values, "vu.busy", busy);
  check("count", r.values, "vu.store", store);
}

}  // namespace

TEST(Phase4, V1_Fp16) {
  const VuCase c = pv(Precision::Fp16, 2, 64);
  const auto r = apm::run_vu_image(c.dram, c.cfg, c.job, c.image);
  check_p_rows(c, r);
  check("cycles", r.values, "t_end", 84);
  vu_stack(r, 20, 48, 16);
  // Row 0 of S is all zeros: every P entry is exactly 1/64 = FP16 0x2400.
  EXPECT_EQ(r.image[1024] | (r.image[1025] << 8), 0x2400);
}

TEST(Phase4, V1_Int8Fixed) {
  const VuCase c = pv(Precision::Int8, 2, 64);
  const auto r = apm::run_vu_image(c.dram, c.cfg, c.job, c.image);
  check_p_rows(c, r);
  check("cycles", r.values, "t_end", 82);
}

TEST(Phase4, V1_Fp32) {
  const VuCase c = pv(Precision::Fp32, 2, 64);
  const auto r = apm::run_vu_image(c.dram, c.cfg, c.job, c.image);
  check_p_rows(c, r);
  check("cycles", r.values, "t_end", 88);
}

TEST(Phase4, V2_TwoSegments) {
  VuCase c = pv(Precision::Fp16, 1, 128);
  for (std::uint64_t j = 0; j < 128; ++j) {  // a non-uniform row
    const std::uint32_t w = apm::f32_to_bits(static_cast<float>(j % 9) - 4.0f);
    std::memcpy(&c.image[j * 4], &w, 4);
  }
  const auto r = apm::run_vu_image(c.dram, c.cfg, c.job, c.image);
  check_p_rows(c, r);
  check("cycles", r.values, "t_end", 84);
  vu_stack(r, 28, 36, 20);
}
