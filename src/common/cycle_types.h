// Plain-C++ cycle type shared by the timing formulas and the SystemC models.
#pragma once
#include <cstdint>

namespace apm {

// The model's only unit of time (design note §0). Never converted to ns in
// any reported result.
using Cycle = std::uint64_t;

}  // namespace apm
