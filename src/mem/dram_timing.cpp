#include "mem/dram_timing.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace apm {

void DramParams::validate() const {
  if (banks == 0 || row_bytes == 0 || bus_bytes_per_cycle == 0 || queue_depth == 0 ||
      burst_bytes == 0 || capacity_bytes == 0)
    throw std::invalid_argument("DramParams: zero-sized parameter");
  if (t_hit == 0) throw std::invalid_argument("DramParams: t_hit must be >= 1");
  if (t_miss < t_hit) throw std::invalid_argument("DramParams: t_miss must be >= t_hit");
  if (bus_bytes_per_cycle % kGranule != 0)
    throw std::invalid_argument("DramParams: bus width must be a multiple of the 16-byte granule");
  if (burst_bytes % bus_bytes_per_cycle != 0)
    throw std::invalid_argument("DramParams: burst_bytes must be a multiple of the bus width");
  if (row_bytes % burst_bytes != 0)
    throw std::invalid_argument("DramParams: row_bytes must be a multiple of burst_bytes");
}

DramTiming::DramTiming(const DramParams& p)
    : p_(p), open_row_(p.banks), bank_ready_(p.banks, 0) {
  p_.validate();
}

DramAddr DramTiming::map(std::uint64_t addr) const {
  const std::uint64_t row_index = addr / p_.row_bytes;
  return {static_cast<unsigned>(row_index % p_.banks), row_index / p_.banks};
}

void DramTiming::check_legal(std::uint64_t addr, std::uint64_t size) const {
  auto fail = [&](const char* why) {
    throw std::invalid_argument(std::string("illegal DRAM request (") + why + "): addr=" +
                                std::to_string(addr) + " size=" + std::to_string(size));
  };
  if (size == 0 || size % DramParams::kGranule != 0) fail("size not a multiple of the 16-byte granule");
  if (addr % DramParams::kGranule != 0) fail("address not 16-byte aligned");
  if (size > p_.burst_bytes) fail("larger than burst_bytes");
  if (addr / p_.row_bytes != (addr + size - 1) / p_.row_bytes) fail("crosses a row");
  if (addr + size > p_.capacity_bytes) fail("beyond capacity");
}

DramSchedule DramTiming::schedule(Cycle accept, std::uint64_t addr, std::uint64_t size) {
  check_legal(addr, size);
  const DramAddr a = map(addr);
  // Bus beats the request touches: a request narrower than the bus, or not
  // bus-aligned, still occupies whole beats (masked/partial beats). With
  // W = 16 this is size / 16 exactly, as in Phase 1 (docs/BUGS.md #8).
  const std::uint64_t w = p_.bus_bytes_per_cycle;
  const Cycle d = (addr + size - 1) / w - addr / w + 1;

  DramSchedule s{};
  s.bank = a.bank;
  s.row = a.row;
  s.hit = open_row_[a.bank] == a.row;
  s.start = std::max(accept, bank_ready_[a.bank]);
  // A miss (or an empty bank) pays the activation before the column command.
  s.col_issue = s.start + (s.hit ? 0 : p_.t_miss - p_.t_hit);
  s.data_start = std::max(s.col_issue + p_.t_hit, bus_free_);
  s.data_end = s.data_start + d;

  bus_free_ = s.data_end;
  // The bank can take its next column command once this burst's command slot
  // (D cycles) has passed, so open-row hits stream back to back on the bus.
  bank_ready_[a.bank] = s.col_issue + d;
  open_row_[a.bank] = a.row;

  (s.hit ? row_hits_ : row_misses_)++;
  bus_busy_ += d;
  return s;
}

}  // namespace apm
