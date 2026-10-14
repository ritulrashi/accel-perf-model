// Output-stationary systolic matmul unit: timing contract and functional data
// path (Phase 2 design note §2). No SystemC here.
//
// Timing contract (the spec the Phase 5 RTL implements). A chunk of Kt
// reduction steps whose feed starts at cycle f performs
//     MAC(i, j, k)  in cycle  f + k + i + j      (row i < R, column j < C)
// because the edge reads one A column vector and one B row vector per cycle
// and skew registers delay row i by i cycles and column j by j cycles.
// After the last chunk of an output tile (feed start f_L):
//     last MAC          f_L + Kt + R + C - 3
//     drain             [f_L + Kt + R + C - 2, f_L + Kt + 2R + C - 2), one row per cycle, row R-1 first
//     tile done         f_L + Kt + 2R + C - 2
// Phase 4 (design note §1.2): a precision with q cycles per reduction step
// holds each operand vector for q cycles, so MAC(i, j, k) occupies
// [f + q*k + i + j, f + q*k + i + j + q), a chunk's feed lasts q*Kt, and every
// formula above holds with Kt replaced by the feed length q*Kt. F is unchanged.
#pragma once
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"
#include "compute/numerics.h"

namespace apm {

struct ArrayShape {
  unsigned rows = 4;  // R
  unsigned cols = 16; // C
};

// Fill skew (R + C - 2) plus drain (R): the part of a tile's latency that
// does not scale with K.
inline Cycle fill_drain(const ArrayShape& a) { return 2ull * a.rows + a.cols - 2; }

// Latency of one output tile fed without bubbles, from first feed cycle to done.
inline Cycle tile_latency(const ArrayShape& a, std::uint64_t k) { return k + fill_drain(a); }
// Phase 4: q*K + F.
inline Cycle tile_latency(const ArrayShape& a, std::uint64_t k, unsigned q) { return q * k + fill_drain(a); }

inline Cycle mac_cycle(Cycle f, unsigned i, unsigned j, std::uint64_t k) { return f + k + i + j; }
// `feed` is the feed length of one chunk: Kt for INT8, q*Kt in general.
inline Cycle drain_start(const ArrayShape& a, Cycle f_last, std::uint64_t feed) {
  return f_last + feed + a.rows + a.cols - 2;
}
inline Cycle tile_done(const ArrayShape& a, Cycle f_last, std::uint64_t feed) {
  return f_last + feed + fill_drain(a);
}

// Largest K for which an INT8 x INT8 dot product cannot overflow INT32:
// |a*b| <= 128*128 = 16384, and K * 16384 <= 2^31 - 1.
inline constexpr std::uint64_t kMaxIntK = 131071;

// The array's accumulators (one per cell): the model's functional data path.
// INT8 accumulates in INT32; the floating-point modes in FP32, in ascending k,
// without fused multiply-add (Phase 4 design note §2.2).
class Accumulators {
 public:
  explicit Accumulators(const ArrayShape& a, Precision p = Precision::Int8);

  void clear();
  // acc[i][j] += sum_k a_blk[i][k] * b_blk[k][j]; a_blk is R x kt row-major,
  // b_blk is kt x C row-major, both INT8.
  void accumulate(const std::int8_t* a_blk, const std::int8_t* b_blk, std::uint64_t kt);
  // Same, for operands stored as e-byte elements of this accumulator's precision.
  void accumulate_bytes(const std::uint8_t* a_blk, const std::uint8_t* b_blk, std::uint64_t kt);
  // R x C row-major INT32 (INT8 mode).
  const std::vector<std::int32_t>& values() const { return acc_; }
  // R x C row-major FP32 (floating-point modes).
  const std::vector<float>& fvalues() const { return facc_; }
  // R x C row-major 4-byte words (INT32 or FP32 bits), little-endian: what the drain writes.
  std::vector<std::uint8_t> bytes() const;

 private:
  ArrayShape a_;
  Precision p_;
  std::vector<std::int32_t> acc_;
  std::vector<float> facc_;
};

}  // namespace apm
