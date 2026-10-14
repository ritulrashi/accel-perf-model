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
#pragma once
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"

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

inline Cycle mac_cycle(Cycle f, unsigned i, unsigned j, std::uint64_t k) { return f + k + i + j; }
inline Cycle drain_start(const ArrayShape& a, Cycle f_last, std::uint64_t kt) {
  return f_last + kt + a.rows + a.cols - 2;
}
inline Cycle tile_done(const ArrayShape& a, Cycle f_last, std::uint64_t kt) {
  return f_last + kt + fill_drain(a);
}

// Largest K for which an INT8 x INT8 dot product cannot overflow INT32:
// |a*b| <= 128*128 = 16384, and K * 16384 <= 2^31 - 1.
inline constexpr std::uint64_t kMaxIntK = 131071;

// The array's accumulators (one per cell, INT32): the model's functional data path.
class Accumulators {
 public:
  explicit Accumulators(const ArrayShape& a);

  void clear();
  // acc[i][j] += sum_k a_blk[i][k] * b_blk[k][j]; a_blk is R x kt row-major,
  // b_blk is kt x C row-major, both INT8.
  void accumulate(const std::int8_t* a_blk, const std::int8_t* b_blk, std::uint64_t kt);
  // R x C row-major INT32.
  const std::vector<std::int32_t>& values() const { return acc_; }

 private:
  ArrayShape a_;
  std::vector<std::int32_t> acc_;
};

}  // namespace apm
