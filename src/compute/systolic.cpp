#include "compute/systolic.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace apm {

Accumulators::Accumulators(const ArrayShape& a, Precision p)
    : a_(a), p_(p), acc_(std::size_t{a.rows} * a.cols, 0), facc_(std::size_t{a.rows} * a.cols, 0.0f) {
  if (a.rows == 0 || a.cols == 0) throw std::invalid_argument("ArrayShape: empty array");
}

void Accumulators::clear() {
  std::fill(acc_.begin(), acc_.end(), 0);
  std::fill(facc_.begin(), facc_.end(), 0.0f);
}

void Accumulators::accumulate(const std::int8_t* a_blk, const std::int8_t* b_blk,
                              std::uint64_t kt) {
  for (unsigned i = 0; i < a_.rows; ++i)
    for (unsigned j = 0; j < a_.cols; ++j) {
      std::int32_t sum = 0;
      for (std::uint64_t k = 0; k < kt; ++k)
        sum += std::int32_t{a_blk[i * kt + k]} * std::int32_t{b_blk[k * a_.cols + j]};
      acc_[std::size_t{i} * a_.cols + j] += sum;
    }
}

void Accumulators::accumulate_bytes(const std::uint8_t* a_blk, const std::uint8_t* b_blk,
                                    std::uint64_t kt) {
  if (p_ == Precision::Int8) {
    accumulate(reinterpret_cast<const std::int8_t*>(a_blk), reinterpret_cast<const std::int8_t*>(b_blk), kt);
    return;
  }
  const unsigned e = elem_bytes(p_);
  std::vector<float> a(std::size_t{a_.rows} * kt), b(kt * a_.cols);
  for (std::size_t x = 0; x < a.size(); ++x) a[x] = decode_float(p_, a_blk + x * e);
  for (std::size_t x = 0; x < b.size(); ++x) b[x] = decode_float(p_, b_blk + x * e);
  // One FP32 rounding for the product (exact for FP16/BF16 inputs), one for
  // the add, in ascending k: the output-stationary order (§2.2).
  for (unsigned i = 0; i < a_.rows; ++i)
    for (unsigned j = 0; j < a_.cols; ++j) {
      float acc = facc_[std::size_t{i} * a_.cols + j];
      for (std::uint64_t k = 0; k < kt; ++k) {
        const float prod = a[i * kt + k] * b[k * a_.cols + j];
        acc = acc + prod;
      }
      facc_[std::size_t{i} * a_.cols + j] = acc;
    }
}

std::vector<std::uint8_t> Accumulators::bytes() const {
  std::vector<std::uint8_t> out(acc_.size() * 4);
  if (p_ == Precision::Int8)
    std::memcpy(out.data(), acc_.data(), out.size());
  else
    std::memcpy(out.data(), facc_.data(), out.size());
  return out;
}

}  // namespace apm
