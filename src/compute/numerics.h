// Precision modes and the bit-exact numerics spec (Phase 4 design note §1-§2).
// Pure C++. The NumPy replica (python/numerics.py) and the standalone C++
// reference (tools/attn_ref.cpp) implement the same spec independently; the
// three are compared bit for bit.
//
// The whole project is built with -ffp-contract=off: every a*b + c below is
// two roundings, never a fused multiply-add (§2.2).
#pragma once
#include <cstdint>
#include <string>

namespace apm {

enum class Precision : std::uint8_t { Int8, Fp16, Bf16, Fp32 };

// Operand bytes e (§1.1).
inline unsigned elem_bytes(Precision p) {
  switch (p) {
    case Precision::Int8: return 1;
    case Precision::Fp16:
    case Precision::Bf16: return 2;
    case Precision::Fp32: return 4;
  }
  return 0;
}
// Default cycles per reduction step q = e (§1.1: fixed multiplier bits per cell).
inline unsigned default_q(Precision p) { return elem_bytes(p); }
inline bool is_float(Precision p) { return p != Precision::Int8; }

const char* precision_name(Precision p);
// "int8" | "fp16" | "bf16" | "fp32"; throws std::invalid_argument otherwise.
Precision parse_precision(const std::string& s);

float bits_to_f32(std::uint32_t b);
std::uint32_t f32_to_bits(float f);

// Exact conversions to FP32 (§2.1).
float fp16_to_f32(std::uint16_t h);
float bf16_to_f32(std::uint16_t h);
// FP32 -> FP16 / BF16, round to nearest, ties to even, FP16 subnormals included.
std::uint16_t f32_to_fp16(float f);
std::uint16_t f32_to_bf16(float f);

// Decodes one stored operand element (e bytes, little-endian) to FP32.
// INT8 is not a float mode: use the integer path for it.
float decode_float(Precision p, const std::uint8_t* src);

// The VU's exponential for x <= 0 (§2.3).
float vu_exp(float x);
// FP32 coefficients c_k = fl32((ln 2)^k / k!), k = 0..6, and fl32(log2 e).
extern const std::uint32_t kVuExpCoeffBits[7];
extern const std::uint32_t kLog2eBits;

// Lane-order sum (§2.4 step 3): lane l < V adds e[l], e[l+V], ... in order,
// then a pairwise tree w = V/2, V/4, ..., 1: lane l += lane l+w. V must be a
// power of two.
float lane_tree_sum(const float* e, std::uint64_t n, unsigned lanes);

}  // namespace apm
