// Mesh network as a TLM interconnect between the PEs' DMA engines and the
// shared SRAM (Phase 3 design note §1.2).
//
//   PE p --BEGIN_REQ (boundary c)--> [pe_side port p]
//        <--END_REQ (sample c+0.5)--  always accepted: unbounded router buffers
//   links reserved at the sample point (round-robin over ports); the request
//   reaches the SRAM at the boundary of its delivery cycle d:
//        [mem_side port p] --BEGIN_REQ (boundary d)--> SRAM port p+1
//   SRAM --BEGIN_RESP (boundary r)--> mesh: completed at once; the reverse
//   route is reserved at sample r+0.5 and the PE gets BEGIN_RESP at the
//   boundary of that delivery cycle.
// Read requests and write acknowledgements are control packets (1 cycle on a
// link); read responses and write requests carry the burst data.
// Every traced payload gets a `network` segment from injection to delivery.
#pragma once
#include <systemc>
#include <tlm>
#include <tlm_utils/multi_passthrough_initiator_socket.h>
#include <tlm_utils/multi_passthrough_target_socket.h>
#include <tlm_utils/peq_with_get.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cycle.h"
#include "common/rr_arbiter.h"
#include "noc/mesh_timing.h"

namespace apm {

class Mesh : public sc_core::sc_module {
 public:
  tlm_utils::multi_passthrough_target_socket<Mesh, 32, tlm::tlm_base_protocol_types, 0,
                                             sc_core::SC_ZERO_OR_MORE_BOUND>
      pe_side;  // port p = PE p
  tlm_utils::multi_passthrough_initiator_socket<Mesh, 32, tlm::tlm_base_protocol_types, 0,
                                                sc_core::SC_ZERO_OR_MORE_BOUND>
      mem_side;  // port p = PE p's channel into the SRAM

  Mesh(sc_core::sc_module_name name, const MeshParams& p, Node sram_node, std::vector<Node> pe_nodes);

  const MeshTiming& timing() const { return timing_; }
  void report(std::map<std::string, std::uint64_t>& out) const;

 private:
  tlm::tlm_sync_enum nb_fw(int port, tlm::tlm_generic_payload& t, tlm::tlm_phase& ph, sc_core::sc_time& d);
  tlm::tlm_sync_enum nb_bw(int port, tlm::tlm_generic_payload& t, tlm::tlm_phase& ph, sc_core::sc_time& d);
  void end_of_elaboration() override;
  void sample_proc();
  void forward_proc();
  void deliver_proc();
  Cycle data_ser(const tlm::tlm_generic_payload& t) const { return timing_.ser(t.get_data_length()); }

  MeshTiming timing_;
  Node sram_node_;
  std::vector<Node> pe_nodes_;
  std::vector<tlm::tlm_generic_payload*> req_pending_;              // per port
  std::vector<std::vector<tlm::tlm_generic_payload*>> resp_pending_; // per port, arrival order
  std::vector<bool> fwd_busy_;                                       // per port: awaiting SRAM END_REQ
  std::unique_ptr<RrArbiter> req_arb_, resp_arb_;
  sc_core::sc_event sample_ev_;
  tlm_utils::peq_with_get<tlm::tlm_generic_payload> fwd_peq_, deliver_peq_;
  std::unordered_map<tlm::tlm_generic_payload*, int> port_of_;
};

}  // namespace apm
