#include "noc/mesh_timing.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace apm {

void MeshParams::validate() const {
  if (nx == 0 || ny == 0) throw std::invalid_argument("MeshParams: empty mesh");
  if (hop == 0) throw std::invalid_argument("MeshParams: hop latency must be >= 1");
  if (link_bytes == 0) throw std::invalid_argument("MeshParams: link bandwidth must be >= 1");
}

MeshTiming::MeshTiming(const MeshParams& p)
    : p_(p), free_(4ull * p.nx * p.ny, 0), busy_(4ull * p.nx * p.ny, 0) {
  p_.validate();
}

void MeshTiming::check(Node n) const {
  if (n.x >= p_.nx || n.y >= p_.ny)
    throw std::invalid_argument("mesh node (" + std::to_string(n.x) + "," + std::to_string(n.y) +
                                ") outside " + std::to_string(p_.nx) + "x" + std::to_string(p_.ny));
}

unsigned MeshTiming::link_id(Node from, unsigned dir) const {
  return (from.y * p_.nx + from.x) * 4 + dir;
}

unsigned MeshTiming::hops(Node a, Node b) const {
  return (a.x > b.x ? a.x - b.x : b.x - a.x) + (a.y > b.y ? a.y - b.y : b.y - a.y);
}

std::vector<unsigned> MeshTiming::route(Node a, Node b) const {
  check(a);
  check(b);
  std::vector<unsigned> links;
  Node n = a;
  while (n.x != b.x) {  // x first
    const bool east = b.x > n.x;
    links.push_back(link_id(n, east ? 0 : 1));
    n.x = east ? n.x + 1 : n.x - 1;
  }
  while (n.y != b.y) {  // then y
    const bool north = b.y > n.y;
    links.push_back(link_id(n, north ? 2 : 3));
    n.y = north ? n.y + 1 : n.y - 1;
  }
  return links;
}

Cycle MeshTiming::ser(std::uint64_t bytes) const {
  return std::max<Cycle>(1, (bytes + p_.link_bytes - 1) / p_.link_bytes);
}

Cycle MeshTiming::send(Cycle inject, Node src, Node dst, Cycle ser_cycles) {
  ++packets_;
  const std::vector<unsigned> path = route(src, dst);
  if (path.empty()) return inject + 1;  // local: through the router, no link
  Cycle t = inject;
  for (unsigned l : path) {
    const Cycle start = std::max(t, free_[l]);
    free_[l] = start + ser_cycles;
    busy_[l] += ser_cycles;
    t = start + p_.hop;
  }
  return t + ser_cycles;
}

std::uint64_t MeshTiming::max_link_busy() const {
  return busy_.empty() ? 0 : *std::max_element(busy_.begin(), busy_.end());
}

}  // namespace apm
