// Cycle <-> sc_time bookkeeping (design note §0, §3.1).
//
// One model cycle is mapped to 1000 ps of SystemC time. Cycle c is the
// interval [c*1000, (c+1)*1000) ps. Initiators act only at a cycle boundary
// (offset 0); targets sample requests at the midpoint (offset 500). Every
// conversion is exact integer arithmetic on sc_time::value(), which counts
// in the default 1 ps resolution (checked by check_time_resolution()).
#pragma once
#include <systemc>

#include <stdexcept>

#include "common/cycle_types.h"

namespace apm {

inline constexpr std::uint64_t kPsPerCycle = 1000;
inline constexpr std::uint64_t kSampleOffsetPs = kPsPerCycle / 2;

inline void check_time_resolution() {
  if (sc_core::sc_get_time_resolution() != sc_core::sc_time(1, sc_core::SC_PS))
    throw std::logic_error("apm models require the default 1 ps SystemC time resolution");
}

inline std::uint64_t now_ps() { return sc_core::sc_time_stamp().value(); }

// The cycle the current time falls in.
inline Cycle now_cycle() { return now_ps() / kPsPerCycle; }

inline bool at_boundary() { return now_ps() % kPsPerCycle == 0; }
inline bool at_sample_point() { return now_ps() % kPsPerCycle == kSampleOffsetPs; }

// Delay from now until the boundary that starts cycle c (c must not be in the past).
inline sc_core::sc_time until_cycle(Cycle c) {
  const std::uint64_t target = c * kPsPerCycle;
  if (target < now_ps()) throw std::logic_error("until_cycle: cycle is in the past");
  return sc_core::sc_time::from_value(target - now_ps());
}

// Delay from now until the sample point of cycle c.
inline sc_core::sc_time until_sample(Cycle c) {
  const std::uint64_t target = c * kPsPerCycle + kSampleOffsetPs;
  if (target < now_ps()) throw std::logic_error("until_sample: sample point is in the past");
  return sc_core::sc_time::from_value(target - now_ps());
}

inline void require(bool ok, const char* what) {
  if (!ok) throw std::logic_error(what);
}

}  // namespace apm
