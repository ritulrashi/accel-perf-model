// Row softmax: the vector unit's functional contract (Phase 4 design note
// §2.4, §3.4). Pure C++.
//
// For one row s[0..n) (INT32 in INT8 mode, FP32 otherwise), over the
// unmasked j:
//   m   = max s_j                         (exact)
//   x_j = fl32(fl32(s_j - m) * alpha)     e_j = vu_exp(x_j); masked: e_j = 0
//   sum = lane_tree_sum(e, n, V)          r = fl32(1 / sum)
//   p_j = fl32(e_j * r)
// Encoding of P: FP16/BF16 round to nearest even; FP32 as is; INT8 fixed:
// RNE(fl32(p_j * 127)) with row scale fl32(1/127); INT8 row: RNE(fl32(e_j * 127))
// with row scale fl32(r / 127) (p_j / max p = e_j, so this is per-row
// max-scaling at no cost).
#pragma once
#include <cstdint>

#include "compute/numerics.h"

namespace apm {

enum class PScale : std::uint8_t { Fixed, Row };

struct SoftmaxSpec {
  Precision prec = Precision::Fp32;  // S is INT32 for INT8, FP32 otherwise; P is e-byte prec
  PScale p_scale = PScale::Fixed;    // INT8 only
  unsigned lanes = 16;               // V (power of two)
  float alpha = 1.0f;
  bool causal = false;
  std::uint64_t q_off = 0;  // row r is the query at position r + q_off; column j valid iff j <= r + q_off
};

// s: n 4-byte words (little-endian); p_out: n * e bytes; returns the row scale
// (INT8 modes; 1.0f for floating-point modes).
float softmax_row(const SoftmaxSpec& spec, std::uint64_t row, const std::uint8_t* s, std::uint64_t n,
                  std::uint8_t* p_out);

}  // namespace apm
