// Numerics spec hand cases (Phase 4 design note §2, §7 "Unit-level hand cases").
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "compute/numerics.h"
#include "compute/systolic.h"

using apm::bits_to_f32;
using apm::f32_to_bf16;
using apm::f32_to_fp16;

TEST(Numerics, Fp16RoundToNearestEvenHandCases) {
  EXPECT_EQ(f32_to_fp16(1.0f), 0x3C00);
  EXPECT_EQ(f32_to_fp16(65504.0f), 0x7BFF);
  EXPECT_EQ(f32_to_fp16(std::ldexp(1.0f, -24)), 0x0001);
  EXPECT_EQ(f32_to_fp16(std::ldexp(1.0f, -25)), 0x0000);         // tie to even
  EXPECT_EQ(f32_to_fp16(1.5f * std::ldexp(1.0f, -25)), 0x0001);
  EXPECT_EQ(f32_to_fp16(1.0f + std::ldexp(1.0f, -11)), 0x3C00);  // tie to even
  EXPECT_EQ(f32_to_fp16(1.0f + 3 * std::ldexp(1.0f, -11)), 0x3C02);
  EXPECT_EQ(f32_to_fp16(-2.0f), 0xC000);
  EXPECT_EQ(f32_to_fp16(65520.0f), 0x7C00);  // halfway to the next binade: rounds to infinity
}

TEST(Numerics, Fp16ExhaustiveRoundTrip) {
  for (std::uint32_t h = 0; h < 0x10000; ++h) {
    if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0) continue;  // NaN
    const float f = apm::fp16_to_f32(static_cast<std::uint16_t>(h));
    ASSERT_EQ(f32_to_fp16(f), h) << std::hex << h;
  }
}

TEST(Numerics, Bf16RoundToNearestEvenHandCases) {
  EXPECT_EQ(f32_to_bf16(1.0f + std::ldexp(1.0f, -8)), 0x3F80);  // tie to even
  EXPECT_EQ(f32_to_bf16(1.0f + 3 * std::ldexp(1.0f, -8)), 0x3F82);
  EXPECT_EQ(apm::bf16_to_f32(0x3F80), 1.0f);
  EXPECT_EQ(apm::bf16_to_f32(0xC040), -3.0f);
}

TEST(Numerics, VuExpCoefficientsFromTheFormula) {
  double fact = 1.0;
  for (int k = 0; k <= 6; ++k) {
    if (k > 0) fact *= k;
    EXPECT_EQ(apm::kVuExpCoeffBits[k], apm::f32_to_bits(static_cast<float>(std::pow(std::log(2.0), k) / fact))) << k;
  }
  EXPECT_EQ(apm::kLog2eBits, apm::f32_to_bits(static_cast<float>(1.0 / std::log(2.0))));
}

TEST(Numerics, VuExpHandCasesAndAccuracy) {
  EXPECT_EQ(apm::vu_exp(0.0f), 1.0f);
  EXPECT_EQ(apm::vu_exp(-90.0f), 0.0f);  // t = -129.8 < -125: flushed
  EXPECT_THROW(apm::vu_exp(0.5f), std::invalid_argument);
  // Accuracy (design note §10 implementation note 1): relative error at most
  // 6e-7 plus 1e-7 per unit of |t|, the rounding of t = fl(x * log2 e).
  double worst = 0;
  for (int i = 0; i <= 100000; ++i) {
    const float x = static_cast<float>(-86.0 * i / 100000.0);
    const double ref = std::exp(static_cast<double>(x));
    const double rel = std::fabs(apm::vu_exp(x) - ref) / ref;
    const double t = std::fabs(x * 1.4426950408889634);
    ASSERT_LE(rel, 6e-7 + 1e-7 * t) << "x=" << x;
    if (t <= 1.0) worst = std::max(worst, rel);
  }
  EXPECT_LE(worst, 4e-7);  // the §2.3 polynomial bound near 0
}

TEST(Numerics, LaneOrderSumHandCase) {
  const float u = std::ldexp(1.0f, -24);
  const std::vector<float> e = {1.0f, u, u, u};
  // Lanes: 1 + 2^-24 = 1 (tie to even), 2^-24 + 2^-24 = 2^-23; tree: 1 + 2^-23.
  EXPECT_EQ(apm::lane_tree_sum(e.data(), e.size(), 2), 1.0f + std::ldexp(1.0f, -23));
  // A sequential sum would give exactly 1 (the case exists to expose mutant M5).
  float seq = 0.0f;
  for (float v : e) seq += v;
  EXPECT_EQ(seq, 1.0f);
  EXPECT_THROW(apm::lane_tree_sum(e.data(), e.size(), 3), std::invalid_argument);
}

TEST(Numerics, PrecisionTable) {
  EXPECT_EQ(apm::elem_bytes(apm::Precision::Int8), 1u);
  EXPECT_EQ(apm::elem_bytes(apm::Precision::Fp16), 2u);
  EXPECT_EQ(apm::elem_bytes(apm::Precision::Bf16), 2u);
  EXPECT_EQ(apm::elem_bytes(apm::Precision::Fp32), 4u);
  EXPECT_EQ(apm::default_q(apm::Precision::Fp32), 4u);
  EXPECT_EQ(apm::parse_precision("bf16"), apm::Precision::Bf16);
  EXPECT_THROW(apm::parse_precision("fp8"), std::invalid_argument);
}

TEST(Systolic, TileLatencyWithQ) {
  EXPECT_EQ(apm::tile_latency(apm::ArrayShape{4, 16}, 64, 2), 150u);  // P4-T1: 128 + 22
  EXPECT_EQ(apm::tile_latency(apm::ArrayShape{4, 16}, 64, 4), 278u);  // P4-T2
  EXPECT_EQ(apm::tile_done(apm::ArrayShape{4, 16}, 137, 128), 287u);
}

TEST(Systolic, Fp16Accumulate) {
  apm::Accumulators acc(apm::ArrayShape{1, 1}, apm::Precision::Fp16);
  // A = [1, 2], B = [3, 4]^T -> 11.
  const std::uint16_t a[] = {f32_to_fp16(1.0f), f32_to_fp16(2.0f)};
  const std::uint16_t b[] = {f32_to_fp16(3.0f), f32_to_fp16(4.0f)};
  acc.accumulate_bytes(reinterpret_cast<const std::uint8_t*>(a), reinterpret_cast<const std::uint8_t*>(b), 2);
  EXPECT_EQ(acc.fvalues()[0], 11.0f);
}

// FP32 accumulation is two roundings, never fused (§2.2): a1*b1 = 1 + 2^-11 +
// 2^-24 rounds (tie to even) to 1 + 2^-11, which cancels acc exactly. A fused
// multiply-add would leave 2^-24.
TEST(Systolic, Fp32AccumulateIsNotFused) {
  apm::Accumulators acc(apm::ArrayShape{1, 1}, apm::Precision::Fp32);
  const float a[] = {-(1.0f + std::ldexp(1.0f, -11)), 1.0f + std::ldexp(1.0f, -12)};
  const float b[] = {1.0f, 1.0f + std::ldexp(1.0f, -12)};
  acc.accumulate_bytes(reinterpret_cast<const std::uint8_t*>(a), reinterpret_cast<const std::uint8_t*>(b), 2);
  EXPECT_EQ(acc.fvalues()[0], 0.0f);
}
