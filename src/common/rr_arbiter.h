// Round-robin grant order over N ports (design note §3.3).
//
// At each sample point a target asks order() for the sequence in which to
// consider its pending requests, then reports which port was granted first.
// The pointer moves to the port after that one, so a port that won a tie
// goes to the back of the line for the next tie.
#pragma once
#include <stdexcept>
#include <vector>

namespace apm {

class RrArbiter {
 public:
  explicit RrArbiter(unsigned ports) : n_(ports) {
    if (ports == 0) throw std::invalid_argument("RrArbiter needs at least one port");
  }

  // Ports in priority order, starting at the pointer and wrapping.
  std::vector<unsigned> order() const {
    std::vector<unsigned> o;
    o.reserve(n_);
    for (unsigned i = 0; i < n_; ++i) o.push_back((p_ + i) % n_);
    return o;
  }

  // Call once per sample point that granted at least one request.
  void granted_first(unsigned port) {
    p_ = (port + 1) % n_;
  }

  unsigned pointer() const { return p_; }

 private:
  unsigned n_;
  unsigned p_ = 0;
};

}  // namespace apm
