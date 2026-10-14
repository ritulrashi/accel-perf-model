// DMA engine (design note §3.6): moves 2-D tiles between DRAM and SRAM.
//
// Each descriptor is split into bursts. A burst is read from the source,
// then written to the destination. Up to M bursts are in flight; a burst
// holds one of the M buffers from its read request until its write response.
// The engine is an SC_METHOD state machine: it acts only at cycle boundaries,
// woken by a submit, a response (same cycle) or an END_REQ (next cycle,
// since END_REQ arrives at the target's mid-cycle sample point).
#pragma once
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cycle.h"

namespace apm {

enum class DmaDir { DramToSram, SramToDram };

struct TileDesc {
  DmaDir dir = DmaDir::DramToSram;
  std::uint64_t src = 0, dst = 0;
  std::uint32_t rows = 1;
  std::uint64_t row_bytes = 0;
  std::uint64_t src_stride = 0, dst_stride = 0;  // bytes between row starts
};

struct DmaParams {
  std::uint64_t burst_bytes = 64;
  unsigned max_bursts_in_flight = 4;  // M
};

struct DescResult {
  Cycle submit = 0, done = 0;
  std::uint64_t bursts = 0, bytes = 0;
};

class DmaEngine : public sc_core::sc_module {
 public:
  tlm_utils::simple_initiator_socket<DmaEngine> dram_socket;
  tlm_utils::simple_initiator_socket<DmaEngine> sram_socket;

  DmaEngine(sc_core::sc_module_name name, const DmaParams& p);

  // Before sc_start, or from a process at a cycle boundary.
  void submit(const TileDesc& d);

  const std::vector<DescResult>& results() const { return results_; }
  void report(std::map<std::string, std::uint64_t>& out) const;

 private:
  struct Burst {
    std::uint64_t src, dst;
    std::uint32_t size;
    std::vector<std::uint8_t> data;
    tlm::tlm_generic_payload rd, wr;
  };

  void act();
  void start_next_descriptor();
  void issue(tlm_utils::simple_initiator_socket<DmaEngine>& s, tlm::tlm_generic_payload& t);
  tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload& trans, tlm::tlm_phase& phase,
                                     sc_core::sc_time& delay);
  tlm_utils::simple_initiator_socket<DmaEngine>& read_socket();
  tlm_utils::simple_initiator_socket<DmaEngine>& write_socket();

  DmaParams p_;
  std::deque<std::pair<TileDesc, Cycle>> queue_;  // descriptor, submit cycle
  bool active_ = false;
  TileDesc cur_{};
  DescResult cur_result_{};
  std::vector<std::unique_ptr<Burst>> bursts_;
  std::unordered_map<tlm::tlm_generic_payload*, Burst*> burst_of_;
  std::size_t next_read_ = 0, writes_done_ = 0;
  unsigned buffers_free_ = 0;
  bool read_busy_ = false, write_busy_ = false;  // waiting for END_REQ
  std::deque<Burst*> write_q_;                   // read done, write not yet sent
  std::vector<DescResult> results_;
  sc_core::sc_event act_now_, act_next_;
};

}  // namespace apm
