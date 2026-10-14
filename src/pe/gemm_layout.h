// GEMM tiling and host-side packing (Phase 2 design note §3.1).
//
// C (M x N, INT32) = A (M x K, INT8) * B (K x N, INT8). Output tiles are R x C
// and K is split into chunks of Kt. The host packs A and B in DRAM in tile
// order so that every operand block the PE loads is one contiguous transfer:
//   A block (mi, c) = A[mi*R .. +R][c*Kt .. +Kt], R x Kt row-major,
//                     at a_base + (mi*nc + c) * R*Kt
//   B block (nj, c) = B[c*Kt .. +Kt][nj*C .. +C], Kt x C row-major,
//                     at b_base + (nj*nc + c) * Kt*C
// C is plain row-major at c_base. Output tiles are visited row-major over
// (mi, nj), each tile's chunks c = 0..nc-1 in order: step s = tile*nc + c.
#pragma once
#include <cstdint>
#include <vector>

#include "compute/systolic.h"

namespace apm {

struct GemmShape {
  std::uint64_t m = 0, n = 0, k = 0;
};

struct GemmLayout {
  GemmShape shape;
  ArrayShape array;
  std::uint64_t kt = 64;
  std::uint64_t a_base = 0, b_base = 0, c_base = 0;

  // Throws std::invalid_argument on a shape the PE cannot run (§3.1).
  void validate() const;

  std::uint64_t chunks() const { return shape.k / kt; }             // nc
  std::uint64_t tiles_m() const { return shape.m / array.rows; }
  std::uint64_t tiles_n() const { return shape.n / array.cols; }
  std::uint64_t tiles() const { return tiles_m() * tiles_n(); }
  std::uint64_t steps() const { return tiles() * chunks(); }

  std::uint64_t a_block_bytes() const { return array.rows * kt; }
  std::uint64_t b_block_bytes() const { return kt * array.cols; }
  std::uint64_t c_tile_bytes() const { return 4ull * array.rows * array.cols; }

  std::uint64_t a_block_addr(std::uint64_t mi, std::uint64_t c) const;
  std::uint64_t b_block_addr(std::uint64_t nj, std::uint64_t c) const;
  std::uint64_t c_tile_addr(std::uint64_t mi, std::uint64_t nj) const;  // top-left element

  std::uint64_t a_packed_bytes() const { return shape.m * shape.k; }
  std::uint64_t b_packed_bytes() const { return shape.k * shape.n; }
  std::uint64_t c_bytes() const { return 4 * shape.m * shape.n; }

  // Host packing: row-major A (M x K) / B (K x N) -> the packed images above.
  std::vector<std::uint8_t> pack_a(const std::vector<std::int8_t>& a) const;
  std::vector<std::uint8_t> pack_b(const std::vector<std::int8_t>& b) const;
};

}  // namespace apm
