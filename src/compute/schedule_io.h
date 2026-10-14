// PE feed schedules as the model ran them (Phase 5 design note §3.1). Pure C++.
//
// One block per PE:
//   pe <id> R <R> C <C> Kt <Kt> nc <nc>
//   step <s> tile <t> chunk <c> feed <f> a <hex R*Kt bytes> b <hex Kt*C bytes>
//   tile <t> done <cycle> c <hex 4*R*C bytes>
// A blocks are R x Kt row-major, B blocks Kt x C row-major (INT8, as in LMEM);
// C is R x C row-major INT32 little-endian: the model's accumulators at drain.
#pragma once
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "common/cycle_types.h"

namespace apm {

struct SchedStep {
  std::uint64_t tile = 0, chunk = 0;
  Cycle feed = 0;
  std::vector<std::uint8_t> a, b;
};

struct SchedTile {
  std::uint64_t tile = 0;
  Cycle done = 0;
  std::vector<std::uint8_t> c;
};

struct PeSchedule {
  unsigned pe = 0, rows = 0, cols = 0;
  std::uint64_t kt = 0, chunks = 0;
  std::vector<SchedStep> steps;
  std::vector<SchedTile> tiles;
};

void write_schedule(std::ostream& out, const PeSchedule& s);
// Parses every PE block in the stream; throws std::invalid_argument on malformed input.
std::vector<PeSchedule> parse_schedules(std::istream& in);

std::string to_hex(const std::vector<std::uint8_t>& v);
std::vector<std::uint8_t> from_hex(const std::string& h);

}  // namespace apm
