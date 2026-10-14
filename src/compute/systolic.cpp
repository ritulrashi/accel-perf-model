#include "compute/systolic.h"

#include <algorithm>
#include <stdexcept>

namespace apm {

Accumulators::Accumulators(const ArrayShape& a) : a_(a), acc_(std::size_t{a.rows} * a.cols, 0) {
  if (a.rows == 0 || a.cols == 0) throw std::invalid_argument("ArrayShape: empty array");
}

void Accumulators::clear() { std::fill(acc_.begin(), acc_.end(), 0); }

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

}  // namespace apm
