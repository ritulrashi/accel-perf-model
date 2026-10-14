// SystemC cannot be re-elaborated inside one process. Every harness entry
// point that builds and runs a simulation calls claim_simulation() first, so
// a second simulation in the same process fails loudly instead of misbehaving.
#pragma once
#include <stdexcept>

namespace apm {

inline void claim_simulation() {
  static bool used = false;  // one instance program-wide (inline function)
  if (used)
    throw std::logic_error("one SystemC simulation per process (run tests through ctest)");
  used = true;
}

}  // namespace apm
