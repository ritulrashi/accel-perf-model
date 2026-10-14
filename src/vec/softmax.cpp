#include "vec/softmax.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace apm {

float softmax_row(const SoftmaxSpec& spec, std::uint64_t row, const std::uint8_t* s, std::uint64_t n,
                  std::uint8_t* p_out) {
  const bool int_mode = spec.prec == Precision::Int8;
  auto valid = [&](std::uint64_t j) { return !spec.causal || j <= row + spec.q_off; };
  auto word = [&](std::uint64_t j) {
    std::uint32_t w;
    std::memcpy(&w, s + 4 * j, 4);
    return w;
  };
  // S value j as FP32: INT32 -> FP32 is exact here (|s| < 2^24, design note §2.4).
  auto value = [&](std::uint64_t j) {
    return int_mode ? static_cast<float>(static_cast<std::int32_t>(word(j))) : bits_to_f32(word(j));
  };

  // 1. max over the unmasked entries (exact).
  bool any = false;
  float m = 0.0f;
  std::int32_t mi = std::numeric_limits<std::int32_t>::min();
  for (std::uint64_t j = 0; j < n; ++j) {
    if (!valid(j)) continue;
    if (int_mode) {
      mi = std::max(mi, static_cast<std::int32_t>(word(j)));
    } else if (!any || value(j) > m) {
      m = value(j);
    }
    any = true;
  }
  if (!any) throw std::invalid_argument("softmax_row: every entry of the row is masked");
  if (int_mode) m = static_cast<float>(mi);

  // 2. exponentials.
  std::vector<float> e(n, 0.0f);
  for (std::uint64_t j = 0; j < n; ++j) {
    if (!valid(j)) continue;
    const float d = value(j) - m;
    const float x = d * spec.alpha;
    e[j] = vu_exp(x);
  }
  // 3-4. lane-order sum, reciprocal, normalize.
  const float sum = lane_tree_sum(e.data(), n, spec.lanes);
  const float r = 1.0f / sum;

  // 5. encode.
  const unsigned eb = elem_bytes(spec.prec);
  for (std::uint64_t j = 0; j < n; ++j) {
    const float p = e[j] * r;
    std::uint8_t* dst = p_out + j * eb;
    switch (spec.prec) {
      case Precision::Fp16: {
        const std::uint16_t h = f32_to_fp16(p);
        dst[0] = static_cast<std::uint8_t>(h);
        dst[1] = static_cast<std::uint8_t>(h >> 8);
        break;
      }
      case Precision::Bf16: {
        const std::uint16_t h = f32_to_bf16(p);
        dst[0] = static_cast<std::uint8_t>(h);
        dst[1] = static_cast<std::uint8_t>(h >> 8);
        break;
      }
      case Precision::Fp32: {
        const std::uint32_t w = f32_to_bits(p);
        std::memcpy(dst, &w, 4);
        break;
      }
      case Precision::Int8: {
        const float v = spec.p_scale == PScale::Row ? e[j] : p;
        const float scaled = v * 127.0f;
        dst[0] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::nearbyint(scaled)));
        break;
      }
    }
  }
  if (!int_mode) return 1.0f;
  return spec.p_scale == PScale::Row ? r / 127.0f : 1.0f / 127.0f;
}

}  // namespace apm
