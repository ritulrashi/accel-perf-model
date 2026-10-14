// DMA engine (Phase 1 design note §3.6, changed in Phase 2 note §1): moves
// 2-D tiles between DRAM and an SRAM-type memory (shared SRAM or a PE's LMEM).
//
// Every descriptor is split into bursts. Bursts of all submitted descriptors
// form one in-order stream: a burst is read from its source, then written to
// its destination. Up to M bursts are in flight; a burst holds one of the M
// buffers from its read request until its write response. A descriptor
// completes at the write response of its own last burst.
//
// Each socket carries at most one request awaiting END_REQ (TLM base
// protocol). Load reads and store writes share the DRAM socket, so occupancy
// is tracked per socket; when a write and a read want the same socket in the
// same cycle, the write goes first (it frees a buffer).
//
// The engine is an SC_METHOD state machine that acts only at cycle
// boundaries: woken by a submit, a response (same cycle) or an END_REQ (next
// cycle, since END_REQ arrives at the target's mid-cycle sample point).
#pragma once
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cycle.h"
#include "common/trace.h"
#include "common/trace_ext.h"
#include "dma/dma_types.h"

namespace apm {

struct DescResult {
  Cycle submit = 0, done = 0;
  std::uint64_t bursts = 0, bytes = 0;
  bool complete = false;
};

class DmaEngine : public sc_core::sc_module {
 public:
  tlm_utils::simple_initiator_socket<DmaEngine> dram_socket;
  tlm_utils::simple_initiator_socket<DmaEngine> sram_socket;

  // Called at the cycle a descriptor completes, from inside the TLM backward
  // path: the callee must only record and notify, never call into the DMA.
  using DoneCallback = std::function<void(std::size_t desc_id, Cycle done)>;

  DmaEngine(sc_core::sc_module_name name, const DmaParams& p);

  // Before sc_start, or from a process at a cycle boundary. Returns the
  // descriptor id (0, 1, ... in submission order).
  std::size_t submit(const TileDesc& d);
  void set_done_callback(DoneCallback cb) { on_done_ = std::move(cb); }
  // Phase 3 stall stack: when on, every burst carries a TraceExt and its
  // finished record is kept (burst_records(desc)[i] for the i-th burst).
  void set_trace(bool on) { trace_ = on; }
  const std::vector<BurstRecord>& burst_records(std::size_t desc) const { return archive_.at(desc); }

  const std::vector<DescResult>& results() const { return results_; }
  void report(std::map<std::string, std::uint64_t>& out) const;

 private:
  struct Burst {
    std::size_t desc, idx;  // descriptor id, position within the descriptor
    DmaDir dir;
    std::vector<std::uint8_t> data;
    std::vector<Segment> segs;  // trace, appended by every component on the way
    TraceExt rd_ext, wr_ext;
    tlm::tlm_generic_payload rd, wr;
    // The payloads must not free our member extensions when destroyed.
    ~Burst() {
      rd.clear_extension(&rd_ext);
      wr.clear_extension(&wr_ext);
    }
  };
  using Socket = tlm_utils::simple_initiator_socket<DmaEngine>;

  void act();
  bool& busy(const Socket& s) { return &s == &dram_socket ? dram_busy_ : sram_busy_; }
  Socket& read_socket(const Burst& b) { return b.dir == DmaDir::DramToSram ? dram_socket : sram_socket; }
  Socket& write_socket(const Burst& b) { return b.dir == DmaDir::DramToSram ? sram_socket : dram_socket; }
  void issue(Socket& s, tlm::tlm_generic_payload& t);
  tlm::tlm_sync_enum nb_transport_bw_dram(tlm::tlm_generic_payload& trans, tlm::tlm_phase& phase,
                                          sc_core::sc_time& delay);
  tlm::tlm_sync_enum nb_transport_bw_sram(tlm::tlm_generic_payload& trans, tlm::tlm_phase& phase,
                                          sc_core::sc_time& delay);
  tlm::tlm_sync_enum on_bw(Socket& s, tlm::tlm_generic_payload& trans, tlm::tlm_phase& phase);

  DmaParams p_;
  std::deque<std::unique_ptr<Burst>> to_read_;  // not yet issued, stream order
  // Issued bursts, owned here until their write response; keyed by both payloads.
  std::unordered_map<tlm::tlm_generic_payload*, Burst*> live_;
  std::unordered_map<Burst*, std::unique_ptr<Burst>> owned_;
  std::deque<Burst*> write_q_;  // read done, write not yet sent (read-response order)
  std::vector<DescResult> results_;
  std::vector<std::uint64_t> remaining_;  // per descriptor, bursts not yet written
  unsigned buffers_free_;
  bool dram_busy_ = false, sram_busy_ = false;  // waiting for END_REQ on that socket
  DoneCallback on_done_;
  bool trace_ = false;
  std::vector<std::vector<BurstRecord>> archive_;  // by descriptor, by burst index
  sc_core::sc_event act_now_, act_next_;
};

}  // namespace apm
