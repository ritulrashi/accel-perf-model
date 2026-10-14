// Common approximately-timed (AT) target behaviour shared by the DRAM channel
// and the SRAM (design note §3.1-§3.3).
//
// Protocol, per port (one initiator per port):
//   cycle c, boundary : initiator sends BEGIN_REQ          -> held as pending
//   cycle c, +500 ps  : sample point; pending requests are
//                       considered in round-robin order.
//                       accept() decides and schedules     -> END_REQ sent now
//                       a refused request stays pending
//                       (no END_REQ = backpressure) and is
//                       reconsidered at the next sample point.
//   response cycle r  : BEGIN_RESP sent at the boundary of r; the initiator
//                       must answer TLM_COMPLETED (immediate END_RESP).
//
// Data is copied between the payload and the backing store at acceptance, in
// grant order.
#pragma once
#include <systemc>
#include <tlm>
#include <tlm_utils/multi_passthrough_target_socket.h>
#include <tlm_utils/peq_with_get.h>

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/cycle.h"
#include "common/rr_arbiter.h"
#include "common/trace.h"

namespace apm {

// What a target decided for one accepted request.
struct AcceptInfo {
  Cycle response = 0;              // must be > accept cycle
  Cycle conflict = 0;              // leading cycles spent waiting for banks held by others
  StallCat cat = StallCat::Dma;    // what the rest of the residency counts as (stall stack)
};

class SampledTarget : public sc_core::sc_module {
 public:
  // Zero bound initiators is legal (a scenario may not use this target).
  tlm_utils::multi_passthrough_target_socket<SampledTarget, 32, tlm::tlm_base_protocol_types, 0,
                                             sc_core::SC_ZERO_OR_MORE_BOUND>
      socket;

  SampledTarget(sc_core::sc_module_name name, std::uint64_t capacity_bytes);

  // Backdoor access for test setup and checking (zero time, no timing effect).
  void poke(std::uint64_t addr, const std::uint8_t* src, std::size_t len);
  void peek(std::uint64_t addr, std::uint8_t* dst, std::size_t len) const;

  // Request-cycles of refused acceptance: at each sample point, the number of
  // requests left pending is added.
  std::uint64_t backpressure_cycles() const { return backpressure_; }

  virtual void report(std::map<std::string, std::uint64_t>& out) const = 0;

 protected:
  // Called once per sample point before any grant (cycle = accept cycle).
  virtual void begin_sample(Cycle) {}
  // Decide one request at its sample point. Return false to leave it pending;
  // otherwise schedule it and fill `info`. Traced bursts get the segments
  // [arrival, accept) and [accept, response) split by info.conflict.
  virtual bool accept(const tlm::tlm_generic_payload& trans, Cycle accept, AcceptInfo& info) = 0;

  void end_of_elaboration() override;

 private:
  tlm::tlm_sync_enum nb_transport_fw(int port, tlm::tlm_generic_payload& trans,
                                     tlm::tlm_phase& phase, sc_core::sc_time& delay);
  unsigned int transport_dbg(int port, tlm::tlm_generic_payload& trans);
  void sample_proc();
  void response_proc();
  void copy_data(tlm::tlm_generic_payload& trans);

  std::vector<std::uint8_t> store_;
  std::vector<tlm::tlm_generic_payload*> pending_;  // per port, at most one
  std::vector<Cycle> arrived_;                      // per port, cycle of the pending BEGIN_REQ
  std::unique_ptr<RrArbiter> arbiter_;
  sc_core::sc_event sample_ev_;
  tlm_utils::peq_with_get<tlm::tlm_generic_payload> resp_peq_;
  std::unordered_map<tlm::tlm_generic_payload*, int> port_of_;
  std::uint64_t backpressure_ = 0;
};

}  // namespace apm
