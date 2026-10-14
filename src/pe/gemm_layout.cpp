#include "pe/gemm_layout.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace apm {

void GemmLayout::validate() const {
  auto fail = [&](const std::string& why) {
    throw std::invalid_argument("GEMM " + std::to_string(shape.m) + "x" + std::to_string(shape.n) +
                                "x" + std::to_string(shape.k) + " on a " +
                                std::to_string(array.rows) + "x" + std::to_string(array.cols) +
                                " array, Kt=" + std::to_string(kt) + ": " + why);
  };
  if (shape.m == 0 || shape.n == 0 || shape.k == 0) fail("empty GEMM");
  if (array.rows == 0 || array.cols == 0 || kt == 0) fail("empty array or chunk");
  if (shape.m % array.rows != 0) fail("M must be a multiple of R");
  if (shape.n % array.cols != 0) fail("N must be a multiple of C");
  if (shape.k % kt != 0) fail("K must be a multiple of Kt");
  if (kt % 16 != 0) fail("Kt must be a multiple of 16 (16-byte operand blocks)");
  if (array.cols % 4 != 0) fail("C must be a multiple of 4 (16-byte C rows)");
  if (prec == Precision::Int8 && shape.k > kMaxIntK)
    fail("K exceeds the INT32 accumulation bound " + std::to_string(kMaxIntK));
  if (m_valid > shape.m) fail("m_valid exceeds M");
  if (m_valid && shape.m - m_valid >= array.rows)
    fail("padding must be less than R rows (M = m_valid rounded up to a multiple of R)");
  if (a_base % 16 != 0 || b_base % 16 != 0 || c_base % 16 != 0) fail("bases must be 16-byte aligned");
}

std::uint64_t GemmLayout::a_block_addr(std::uint64_t mi, std::uint64_t c) const {
  return a_base + (mi * chunks() + c) * a_block_bytes();
}

std::uint64_t GemmLayout::b_block_addr(std::uint64_t nj, std::uint64_t c) const {
  return b_base + (nj * chunks() + c) * b_block_bytes();
}

std::uint64_t GemmLayout::c_tile_addr(std::uint64_t mi, std::uint64_t nj) const {
  return c_base + 4 * (mi * array.rows * shape.n + nj * array.cols);
}

std::uint64_t GemmLayout::tile_valid_rows(std::uint64_t mi) const {
  const std::uint64_t first = mi * array.rows;
  return std::min<std::uint64_t>(array.rows, valid_rows() - first);
}

std::vector<std::uint8_t> GemmLayout::pack_a(const std::vector<std::int8_t>& a) const {
  return pack_a_bytes(std::vector<std::uint8_t>(a.begin(), a.end()));
}

std::vector<std::uint8_t> GemmLayout::pack_b(const std::vector<std::int8_t>& b) const {
  return pack_b_bytes(std::vector<std::uint8_t>(b.begin(), b.end()));
}

std::vector<std::uint8_t> GemmLayout::pack_a_bytes(const std::vector<std::uint8_t>& a) const {
  const std::uint64_t e = elem();
  if (a.size() != shape.m * shape.k * e) throw std::invalid_argument("pack_a: wrong size");
  std::vector<std::uint8_t> out(a_packed_bytes());
  for (std::uint64_t mi = 0; mi < tiles_m(); ++mi)
    for (std::uint64_t c = 0; c < chunks(); ++c) {
      const std::uint64_t blk = a_block_addr(mi, c) - a_base;
      for (std::uint64_t i = 0; i < array.rows; ++i)
        for (std::uint64_t k = 0; k < kt; ++k)
          std::memcpy(&out[blk + (i * kt + k) * e], &a[((mi * array.rows + i) * shape.k + c * kt + k) * e], e);
    }
  return out;
}

std::vector<std::uint8_t> GemmLayout::pack_b_bytes(const std::vector<std::uint8_t>& b) const {
  const std::uint64_t e = elem();
  if (b.size() != shape.k * shape.n * e) throw std::invalid_argument("pack_b: wrong size");
  std::vector<std::uint8_t> out(b_packed_bytes());
  for (std::uint64_t nj = 0; nj < tiles_n(); ++nj)
    for (std::uint64_t c = 0; c < chunks(); ++c) {
      const std::uint64_t blk = b_block_addr(nj, c) - b_base;
      for (std::uint64_t k = 0; k < kt; ++k)
        for (std::uint64_t j = 0; j < array.cols; ++j)
          std::memcpy(&out[blk + (k * array.cols + j) * e], &b[((c * kt + k) * shape.n + nj * array.cols + j) * e], e);
    }
  return out;
}

}  // namespace apm
