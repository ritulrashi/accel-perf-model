// Bump allocator for a PE's local memory (Phase 2 note §2.4b). Buffers are
// carved out once, in order, line-aligned; running out of capacity is an
// error that names every buffer allocated so far.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace apm {

class LmemAllocator {
 public:
  LmemAllocator(std::uint64_t capacity, std::uint64_t align);

  struct Buffer {
    std::string name;
    std::uint64_t addr, size;
  };

  // Throws std::invalid_argument when the buffer does not fit.
  std::uint64_t alloc(const std::string& name, std::uint64_t size);
  std::uint64_t used() const { return next_; }
  const std::vector<Buffer>& buffers() const { return bufs_; }

 private:
  std::uint64_t cap_, align_, next_ = 0;
  std::vector<Buffer> bufs_;
};

}  // namespace apm
