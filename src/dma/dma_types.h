// Plain DMA descriptor and parameter types (no SystemC), shared by the DMA
// engine and pure configuration code.
#pragma once
#include <cstdint>

namespace apm {

enum class DmaDir { DramToSram, SramToDram };

struct TileDesc {
  DmaDir dir = DmaDir::DramToSram;
  std::uint64_t src = 0, dst = 0;
  std::uint32_t rows = 1;
  std::uint64_t row_bytes = 0;
  std::uint64_t src_stride = 0, dst_stride = 0;  // bytes between row starts
};

struct DmaParams {
  std::uint64_t burst_bytes = 64;
  unsigned max_bursts_in_flight = 4;  // M
};

}  // namespace apm
