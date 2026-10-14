#include "compute/numerics.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace apm {

const std::uint32_t kVuExpCoeffBits[7] = {0x3f800000, 0x3f317218, 0x3e75fdf0, 0x3d635847,
                                          0x3c1d955b, 0x3aaec3ff, 0x39218489};
const std::uint32_t kLog2eBits = 0x3fb8aa3b;

const char* precision_name(Precision p) {
  switch (p) {
    case Precision::Int8: return "int8";
    case Precision::Fp16: return "fp16";
    case Precision::Bf16: return "bf16";
    case Precision::Fp32: return "fp32";
  }
  return "?";
}

Precision parse_precision(const std::string& s) {
  if (s == "int8") return Precision::Int8;
  if (s == "fp16") return Precision::Fp16;
  if (s == "bf16") return Precision::Bf16;
  if (s == "fp32") return Precision::Fp32;
  throw std::invalid_argument("unknown precision '" + s + "' (int8, fp16, bf16, fp32)");
}

float bits_to_f32(std::uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

std::uint32_t f32_to_bits(float f) {
  std::uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

float fp16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = std::uint32_t{h & 0x8000u} << 16;
  const std::uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  if (exp == 0) {
    // Zero or subnormal: man * 2^-24, exact in FP32.
    const float v = std::ldexp(static_cast<float>(man), -24);
    return sign ? -v : v;
  }
  if (exp == 31) return bits_to_f32(sign | 0x7f800000u | (man << 13));
  return bits_to_f32(sign | ((exp - 15 + 127) << 23) | (man << 13));
}

float bf16_to_f32(std::uint16_t h) { return bits_to_f32(std::uint32_t{h} << 16); }

std::uint16_t f32_to_fp16(float f) {
  const std::uint32_t x = f32_to_bits(f);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::uint32_t ax = x & 0x7fffffffu;
  if (ax >= 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0));
  if (ax < 0x00800000u) return static_cast<std::uint16_t>(sign);  // FP32 subnormal: far below 2^-25
  const int e = static_cast<int>(ax >> 23) - 127 + 15;            // FP16 biased exponent
  const std::uint32_t mant = (ax & 0x7fffffu) | 0x800000u;        // 24 bits with the hidden bit
  if (e <= 0) {
    // FP16 subnormal: m16 = value / 2^-24 = mant >> (14 - e).
    const int shift = 14 - e;
    if (shift > 24) return static_cast<std::uint16_t>(sign);  // < 2^-25: rounds to zero
    std::uint32_t m16 = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (m16 & 1))) ++m16;
    return static_cast<std::uint16_t>(sign | m16);  // m16 == 0x400 is the smallest normal
  }
  std::uint32_t m16 = mant >> 13;  // 11 bits incl. hidden bit
  const std::uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1))) ++m16;
  // Adding the mantissa carries into the exponent when rounding overflows it.
  const std::uint32_t out = (static_cast<std::uint32_t>(e) << 10) + (m16 - 0x400u);
  if (out >= 0x7c00u) return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | out);
}

std::uint16_t f32_to_bf16(float f) {
  std::uint32_t x = f32_to_bits(f);
  if ((x & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((x >> 16) | 0x40u);  // quiet NaN
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<std::uint16_t>(x >> 16);
}

float decode_float(Precision p, const std::uint8_t* src) {
  switch (p) {
    case Precision::Fp16: return fp16_to_f32(static_cast<std::uint16_t>(src[0] | (src[1] << 8)));
    case Precision::Bf16: return bf16_to_f32(static_cast<std::uint16_t>(src[0] | (src[1] << 8)));
    case Precision::Fp32: {
      std::uint32_t b;
      std::memcpy(&b, src, 4);
      return bits_to_f32(b);
    }
    case Precision::Int8: break;
  }
  throw std::logic_error("decode_float: INT8 is not a floating-point mode");
}

float vu_exp(float x) {
  if (!(x <= 0.0f)) throw std::invalid_argument("vu_exp: argument must be <= 0");
  const float t = x * bits_to_f32(kLog2eBits);
  if (t < -125.0f) return 0.0f;  // flush: keeps every result normal
  const float n = std::nearbyint(t);  // default rounding mode: ties to even
  const float f = t - n;             // exact (Sterbenz), |f| <= 0.5
  float p = bits_to_f32(kVuExpCoeffBits[6]);
  for (int k = 5; k >= 0; --k) {
    const float pf = p * f;
    p = pf + bits_to_f32(kVuExpCoeffBits[k]);
  }
  return std::ldexp(p, static_cast<int>(n));  // exact: n >= -125, p >= 2^-0.5
}

float lane_tree_sum(const float* e, std::uint64_t n, unsigned lanes) {
  if (lanes == 0 || (lanes & (lanes - 1)) != 0) throw std::invalid_argument("lane_tree_sum: V must be a power of two");
  std::vector<float> lane(lanes, 0.0f);
  for (std::uint64_t j = 0; j < n; ++j) lane[j % lanes] += e[j];
  for (unsigned w = lanes / 2; w >= 1; w /= 2)
    for (unsigned l = 0; l < w; ++l) lane[l] += lane[l + w];
  return lane[0];
}

}  // namespace apm
