// Row softmax and VU plan hand cases (Phase 4 design note §2.4, §3, §7).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vec/softmax.h"
#include "vec/vu_plan.h"
#include "vec/vu_stall.h"

using apm::Precision;
using apm::PScale;
using apm::SoftmaxSpec;

namespace {

std::vector<std::uint8_t> zeros_row(std::uint64_t n) { return std::vector<std::uint8_t>(4 * n, 0); }

}  // namespace

TEST(Softmax, ZerosRowGivesExactEighths) {
  SoftmaxSpec s;
  s.prec = Precision::Fp32;
  s.lanes = 4;
  const auto row = zeros_row(8);
  std::vector<std::uint8_t> p(8 * 4);
  EXPECT_EQ(apm::softmax_row(s, 0, row.data(), 8, p.data()), 1.0f);
  for (int j = 0; j < 8; ++j) {
    float v;
    std::memcpy(&v, &p[4 * j], 4);
    EXPECT_EQ(v, 0.125f);
  }
}

TEST(Softmax, CausalRowZeroIsOneHot) {
  SoftmaxSpec s;
  s.prec = Precision::Fp32;
  s.lanes = 4;
  s.causal = true;
  auto row = zeros_row(4);
  const float big = 5.0f;  // masked entries must not take part, even when larger
  for (int j = 1; j < 4; ++j) std::memcpy(&row[4 * j], &big, 4);
  std::vector<std::uint8_t> p(16);
  apm::softmax_row(s, 0, row.data(), 4, p.data());
  float v[4];
  std::memcpy(v, p.data(), 16);
  EXPECT_EQ(v[0], 1.0f);
  EXPECT_EQ(v[1], 0.0f);
  EXPECT_EQ(v[3], 0.0f);
}

TEST(Softmax, Fp16ZerosRow) {
  SoftmaxSpec s;
  s.prec = Precision::Fp16;
  s.lanes = 16;
  const auto row = zeros_row(64);
  std::vector<std::uint8_t> p(128);
  apm::softmax_row(s, 0, row.data(), 64, p.data());
  for (int j = 0; j < 64; ++j) EXPECT_EQ(p[2 * j] | (p[2 * j + 1] << 8), 0x2400);  // 2^-6
}

TEST(Softmax, Int8FixedAndRowScales) {
  SoftmaxSpec s;
  s.prec = Precision::Int8;
  s.lanes = 4;
  const auto row = zeros_row(4);  // INT32 zeros
  std::vector<std::uint8_t> p(4);
  s.p_scale = PScale::Fixed;
  EXPECT_EQ(apm::softmax_row(s, 0, row.data(), 4, p.data()), 1.0f / 127.0f);
  EXPECT_EQ(p[0], 32);  // RNE(0.25 * 127 = 31.75)
  s.p_scale = PScale::Row;
  EXPECT_EQ(apm::softmax_row(s, 0, row.data(), 4, p.data()), 0.25f / 127.0f);
  EXPECT_EQ(p[3], 127);  // e_j = 1 at the max
}

TEST(VuPlan, RowCycles) {
  apm::VuConfig c;
  c.lanes = 16;
  c.t_vu = 4;
  EXPECT_EQ(apm::row_cycles(c, 64), 24u);   // P4-V1
  EXPECT_EQ(apm::row_cycles(c, 128), 36u);  // P4-V2
  c.lanes = 64;
  EXPECT_EQ(apm::row_cycles(c, 1024), 60u);
}

// R = 4, Kt = 16, n = 48 (nc = 3), FP16: segments of 32 B, destination stride 128 B.
TEST(VuPlan, PackedStoreDescriptors) {
  apm::VuJob j;
  j.rows = 8;
  j.n = 48;
  j.spec.prec = Precision::Fp16;
  j.p_base = 4096;
  j.r_pack = 4;
  j.kt = 16;
  const auto d0 = apm::vu_store(j, 0, 0), d3 = apm::vu_store(j, 0, 3), d4 = apm::vu_store(j, 0, 4);
  EXPECT_EQ(d0.dst, 4096u);
  EXPECT_EQ(d3.dst, 4096u + 3 * 32);
  EXPECT_EQ(d4.dst, 4096u + (1 * 3 * 4 + 0) * 32);  // block mi = 1
  EXPECT_EQ(d0.rows, 3u);
  EXPECT_EQ(d0.row_bytes, 32u);
  EXPECT_EQ(d0.dst_stride, 128u);
  EXPECT_EQ(d0.src_stride, 32u);
}

TEST(VuPlan, CapacityBoundaryAndBytes) {
  apm::VuConfig c;
  apm::VuJob j;
  j.rows = 2;
  j.n = 64;
  j.spec.prec = Precision::Fp16;
  j.p_base = 1024;
  EXPECT_EQ(apm::vmem_required(c, j), 768u);  // 2 x 256 + 2 x 128
  c.vmem.capacity_bytes = 768;
  EXPECT_NO_THROW(apm::plan_vu(c, j));
  c.vmem.capacity_bytes = 767;
  EXPECT_THROW(apm::plan_vu(c, j), std::invalid_argument);
  EXPECT_EQ(apm::vu_dram_bytes(j), 768u);  // 2 x 64 x (4 + 2)
}

// Synthetic VU traces: P4-V1's expected timeline must classify as load 20, busy 48, store 16.
TEST(VuStall, P4V1Timeline) {
  const std::vector<apm::VuRowTrace> rows = {{20, 20, 60}, {28, 44, 84}};
  const auto n = apm::classify_vu(rows, 24, 2, -1, 84);
  EXPECT_EQ(n[0], 48u);
  EXPECT_EQ(n[1], 20u);
  EXPECT_EQ(n[2], 16u);
}

TEST(VuStall, UnexplainedIdleThrows) {
  // Row 1 loaded at 28 but started late with nothing to wait for.
  const std::vector<apm::VuRowTrace> rows = {{20, 20, 60}, {28, 50, 90}};
  EXPECT_THROW(apm::classify_vu(rows, 24, 2, -1, 90), std::logic_error);
}
