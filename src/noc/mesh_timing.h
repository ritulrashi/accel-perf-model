// 2-D mesh link timing (Phase 3 design note §1.1). Pure C++.
//
// Routers at (x, y), 0 <= x < nx, 0 <= y < ny, with directed links between
// neighbours. XY routing: first along x, then along y. A packet injected at
// cycle c is pushed through its links in path order (virtual cut-through,
// unbounded router buffers):
//     t_0 = c
//     start_l      = max(t_l, link_free[l])
//     link_free[l] = start_l + ser             // link carries the packet ser cycles
//     t_{l+1}      = start_l + hop             // head reaches the next router
//     delivered    = t_H + ser   (H >= 1),   c + 1   (H = 0, local)
#pragma once
#include <cstdint>
#include <vector>

#include "common/cycle_types.h"

namespace apm {

struct MeshParams {
  unsigned nx = 1, ny = 1;
  Cycle hop = 2;                 // h: cycles per link for the head
  unsigned link_bytes = 16;      // L: bytes per cycle per link
  void validate() const;         // throws std::invalid_argument
};

struct Node {
  unsigned x = 0, y = 0;
};

class MeshTiming {
 public:
  explicit MeshTiming(const MeshParams& p);

  unsigned hops(Node a, Node b) const;
  // Directed link ids along the XY route from a to b.
  std::vector<unsigned> route(Node a, Node b) const;
  // Cycles a packet of `bytes` occupies a link (at least 1).
  Cycle ser(std::uint64_t bytes) const;
  // Reserves the route for one packet injected at `inject`; returns the
  // delivery cycle. Calls must be made in the arbitration order.
  Cycle send(Cycle inject, Node src, Node dst, Cycle ser_cycles);

  std::uint64_t link_busy(unsigned link) const { return busy_[link]; }
  std::uint64_t max_link_busy() const;
  std::uint64_t packets() const { return packets_; }
  const MeshParams& params() const { return p_; }

 private:
  unsigned link_id(Node from, unsigned dir) const;  // dir: 0 +x, 1 -x, 2 +y, 3 -y
  void check(Node n) const;

  MeshParams p_;
  std::vector<Cycle> free_;
  std::vector<std::uint64_t> busy_;
  std::uint64_t packets_ = 0;
};

}  // namespace apm
