#include "mem/sram_timing.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace apm {

void SramParams::validate() const {
  if (banks == 0 || bank_bytes == 0 || capacity_bytes == 0)
    throw std::invalid_argument("SramParams: zero-sized parameter");
  // t_sram >= 1 keeps every response strictly after its sample point.
  if (t_sram == 0) throw std::invalid_argument("SramParams: t_sram must be >= 1");
}

SramTiming::SramTiming(const SramParams& p) : p_(p), bank_free_(p.banks, 0) {
  p_.validate();
}

unsigned SramTiming::bank_of(std::uint64_t addr) const {
  return static_cast<unsigned>((addr / p_.bank_bytes) % p_.banks);
}

void SramTiming::check_legal(std::uint64_t addr, std::uint64_t size) const {
  auto fail = [&](const char* why) {
    throw std::invalid_argument(std::string("illegal SRAM request (") + why + "): addr=" +
                                std::to_string(addr) + " size=" + std::to_string(size));
  };
  if (size == 0 || size % p_.bank_bytes != 0) fail("size not a multiple of the bank width");
  if (addr % p_.bank_bytes != 0) fail("address not line-aligned");
  if (addr + size > p_.capacity_bytes) fail("beyond capacity");
}

SramSchedule SramTiming::schedule(Cycle accept, std::uint64_t addr, std::uint64_t size) {
  check_legal(addr, size);
  const std::uint64_t first = addr / p_.bank_bytes;
  const std::uint64_t n = size / p_.bank_bytes;
  // own[b] = earlier lines of this same request that used bank b. In an
  // otherwise idle SRAM a line would be served at accept + own[b]; anything
  // later than that is waiting for another request (a conflict).
  std::vector<std::uint64_t> own(p_.banks, 0);
  Cycle last = accept;
  std::uint64_t conflict = 0;
  for (std::uint64_t i = 0; i < n; ++i) {
    const unsigned b = static_cast<unsigned>((first + i) % p_.banks);
    const Cycle c = std::max(accept, bank_free_[b]);
    bank_free_[b] = c + 1;
    conflict += c - (accept + own[b]);
    ++own[b];
    last = std::max(last, c);
  }
  std::uint64_t max_own = 0;
  for (std::uint64_t o : own) max_own = std::max(max_own, o);
  lines_ += n;
  conflict_ += conflict;
  return {last + p_.t_sram, n, conflict, last - (accept + max_own - 1)};
}

}  // namespace apm
