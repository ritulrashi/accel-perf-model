#include "pe/lmem_alloc.h"

#include <stdexcept>

namespace apm {

LmemAllocator::LmemAllocator(std::uint64_t capacity, std::uint64_t align)
    : cap_(capacity), align_(align) {
  if (align == 0) throw std::invalid_argument("LmemAllocator: zero alignment");
}

std::uint64_t LmemAllocator::alloc(const std::string& name, std::uint64_t size) {
  if (size == 0) throw std::invalid_argument("LMEM buffer '" + name + "' has zero size");
  const std::uint64_t addr = (next_ + align_ - 1) / align_ * align_;
  if (addr + size > cap_) {
    std::string msg = "LMEM capacity exceeded: '" + name + "' needs " + std::to_string(size) +
                      " B at " + std::to_string(addr) + ", capacity " + std::to_string(cap_) +
                      " B; allocated so far:";
    for (const auto& b : bufs_)
      msg += " " + b.name + "[" + std::to_string(b.addr) + "+" + std::to_string(b.size) + "]";
    throw std::invalid_argument(msg);
  }
  bufs_.push_back({name, addr, size});
  next_ = addr + size;
  return addr;
}

}  // namespace apm
