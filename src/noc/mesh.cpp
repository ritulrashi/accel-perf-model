#include "noc/mesh.h"

#include "common/trace_ext.h"

namespace apm {

using tlm::tlm_generic_payload;

Mesh::Mesh(sc_core::sc_module_name name, const MeshParams& p, Node sram_node,
           std::vector<Node> pe_nodes)
    : sc_module(name),
      pe_side("pe_side"),
      mem_side("mem_side"),
      timing_(p),
      sram_node_(sram_node),
      pe_nodes_(std::move(pe_nodes)),
      fwd_peq_("fwd_peq"),
      deliver_peq_("deliver_peq") {
  timing_.route(sram_node_, sram_node_);  // validates the node
  for (Node n : pe_nodes_) timing_.route(n, n);
  pe_side.register_nb_transport_fw(this, &Mesh::nb_fw);
  mem_side.register_nb_transport_bw(this, &Mesh::nb_bw);

  SC_METHOD(sample_proc);
  sensitive << sample_ev_;
  dont_initialize();
  SC_METHOD(forward_proc);
  sensitive << fwd_peq_.get_event();
  dont_initialize();
  SC_METHOD(deliver_proc);
  sensitive << deliver_peq_.get_event();
  dont_initialize();
}

void Mesh::end_of_elaboration() {
  const std::size_t n = pe_side.size();
  require(n == pe_nodes_.size(), "Mesh: one PE-side binding per PE node required");
  require(mem_side.size() == n, "Mesh: one SRAM-side binding per PE required");
  req_pending_.assign(n, nullptr);
  resp_pending_.assign(n, {});
  fwd_busy_.assign(n, false);
  if (n > 0) {
    req_arb_ = std::make_unique<RrArbiter>(static_cast<unsigned>(n));
    resp_arb_ = std::make_unique<RrArbiter>(static_cast<unsigned>(n));
  }
}

tlm::tlm_sync_enum Mesh::nb_fw(int port, tlm_generic_payload& t, tlm::tlm_phase& ph,
                               sc_core::sc_time& d) {
  if (ph == tlm::END_RESP) return tlm::TLM_COMPLETED;
  require(ph == tlm::BEGIN_REQ && d == sc_core::SC_ZERO_TIME && at_boundary(),
          "Mesh: BEGIN_REQ must arrive at a cycle boundary with zero delay");
  require(req_pending_[port] == nullptr, "Mesh: BEGIN_REQ before END_REQ on the same port");
  req_pending_[port] = &t;
  sample_ev_.notify(until_sample(now_cycle()));
  return tlm::TLM_ACCEPTED;
}

tlm::tlm_sync_enum Mesh::nb_bw(int port, tlm_generic_payload& t, tlm::tlm_phase& ph,
                               sc_core::sc_time&) {
  if (ph == tlm::END_REQ) {
    fwd_busy_[port] = false;
    return tlm::TLM_ACCEPTED;
  }
  require(ph == tlm::BEGIN_RESP && at_boundary(), "Mesh: unexpected phase from the SRAM");
  resp_pending_[port].push_back(&t);
  sample_ev_.notify(until_sample(now_cycle()));
  return tlm::TLM_COMPLETED;
}

void Mesh::sample_proc() {
  require(at_sample_point(), "Mesh: sample off the sample point");
  const Cycle c = now_cycle();
  // Requests (PE -> SRAM). XY routes toward and away from the SRAM node never
  // share a directed link, so the order between requests and responses
  // within a cycle does not matter; within each class it is round-robin.
  bool any = false;
  unsigned first = 0;
  for (unsigned p : req_arb_->order()) {
    tlm_generic_payload* t = req_pending_[p];
    if (t == nullptr) continue;
    req_pending_[p] = nullptr;
    if (!any) first = p;
    any = true;
    const Cycle ser = t->is_write() ? data_ser(*t) : 1;
    const Cycle d = timing_.send(c, pe_nodes_[p], sram_node_, ser);
    add_trace(*t, c, d, StallCat::Network);
    port_of_[t] = static_cast<int>(p);
    fwd_peq_.notify(*t, until_cycle(d));
    tlm::tlm_phase ph = tlm::END_REQ;
    sc_core::sc_time z = sc_core::SC_ZERO_TIME;
    require(pe_side[p]->nb_transport_bw(*t, ph, z) == tlm::TLM_ACCEPTED,
            "Mesh: PE must accept END_REQ");
  }
  if (any) req_arb_->granted_first(first);

  any = false;
  for (unsigned p : resp_arb_->order()) {
    if (resp_pending_[p].empty()) continue;
    if (!any) first = p;
    any = true;
    for (tlm_generic_payload* t : resp_pending_[p]) {
      const Cycle ser = t->is_read() ? data_ser(*t) : 1;
      const Cycle d = timing_.send(c, sram_node_, pe_nodes_[p], ser);
      add_trace(*t, c, d, StallCat::Network);
      port_of_[t] = static_cast<int>(p);
      deliver_peq_.notify(*t, until_cycle(d));
    }
    resp_pending_[p].clear();
  }
  if (any) resp_arb_->granted_first(first);
}

void Mesh::forward_proc() {
  require(at_boundary(), "Mesh: forward off a boundary");
  while (tlm_generic_payload* t = fwd_peq_.get_next_transaction()) {
    const int p = port_of_.at(t);
    // Deliveries for one PE are strictly increasing in cycle (each packet on
    // a route starts at least one link-occupancy after the previous one), so
    // the SRAM has always sent END_REQ for the previous request by now.
    require(!fwd_busy_[p], "Mesh: two requests for one SRAM port in one cycle");
    fwd_busy_[p] = true;
    tlm::tlm_phase ph = tlm::BEGIN_REQ;
    sc_core::sc_time z = sc_core::SC_ZERO_TIME;
    require(mem_side[p]->nb_transport_fw(*t, ph, z) == tlm::TLM_ACCEPTED,
            "Mesh: SRAM must accept BEGIN_REQ");
  }
}

void Mesh::deliver_proc() {
  require(at_boundary(), "Mesh: delivery off a boundary");
  while (tlm_generic_payload* t = deliver_peq_.get_next_transaction()) {
    const int p = port_of_.at(t);
    port_of_.erase(t);
    tlm::tlm_phase ph = tlm::BEGIN_RESP;
    sc_core::sc_time z = sc_core::SC_ZERO_TIME;
    require(pe_side[p]->nb_transport_bw(*t, ph, z) == tlm::TLM_COMPLETED,
            "Mesh: PE must complete BEGIN_RESP");
  }
}

void Mesh::report(std::map<std::string, std::uint64_t>& out) const {
  const std::string p = std::string(basename()) + ".";
  out[p + "packets"] = timing_.packets();
  out[p + "max_link_busy"] = timing_.max_link_busy();
}

}  // namespace apm
