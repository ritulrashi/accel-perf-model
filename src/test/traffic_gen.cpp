#include "test/traffic_gen.h"

namespace apm {

TrafficGen::TrafficGen(sc_core::sc_module_name name, std::vector<TgRequest> script)
    : sc_module(name),
      socket("socket"),
      script_(std::move(script)),
      data_(script_.size()),
      results_(script_.size()) {
  for (std::size_t i = 0; i < script_.size(); ++i) {
    require(i == 0 || script_[i].at >= script_[i - 1].at, "TrafficGen: script not in cycle order");
    const TgRequest& r = script_[i];
    data_[i].assign(r.size, static_cast<std::uint8_t>(i + 1));
    payloads_.push_back(std::make_unique<tlm::tlm_generic_payload>());
    tlm::tlm_generic_payload& t = *payloads_.back();
    t.set_command(r.write ? tlm::TLM_WRITE_COMMAND : tlm::TLM_READ_COMMAND);
    t.set_address(r.addr);
    t.set_data_ptr(data_[i].data());
    t.set_data_length(r.size);
    t.set_streaming_width(r.size);
    t.set_byte_enable_ptr(nullptr);
    t.set_dmi_allowed(false);
    t.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
    index_of_[&t] = i;
  }
  socket.register_nb_transport_bw(this, &TrafficGen::nb_transport_bw);
  SC_METHOD(act);  // also runs once at time 0
  sensitive << act_next_;
}

void TrafficGen::act() {
  require(at_boundary(), "TrafficGen: acting off a cycle boundary");
  if (busy_ || next_ >= script_.size()) return;
  const Cycle c = now_cycle();
  if (script_[next_].at > c) {
    act_next_.notify(until_cycle(script_[next_].at));
    return;
  }
  results_[next_].issued = c;
  busy_ = true;
  tlm::tlm_phase phase = tlm::BEGIN_REQ;
  sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
  const tlm::tlm_sync_enum r = socket->nb_transport_fw(*payloads_[next_++], phase, delay);
  require(r == tlm::TLM_ACCEPTED, "TrafficGen: target must return TLM_ACCEPTED to BEGIN_REQ");
}

tlm::tlm_sync_enum TrafficGen::nb_transport_bw(tlm::tlm_generic_payload& trans,
                                               tlm::tlm_phase& phase, sc_core::sc_time&) {
  if (phase == tlm::END_REQ) {
    busy_ = false;
    act_next_.notify(until_cycle(now_cycle() + 1));
    return tlm::TLM_ACCEPTED;
  }
  require(phase == tlm::BEGIN_RESP, "TrafficGen: unexpected phase");
  require(trans.is_response_ok(), "TrafficGen: error response");
  TgResult& res = results_.at(index_of_.at(&trans));
  res.response = now_cycle();
  res.done = true;
  return tlm::TLM_COMPLETED;
}

void TrafficGen::report(std::map<std::string, std::uint64_t>& out) const {
  for (std::size_t i = 0; i < results_.size(); ++i) {
    const std::string p = std::string(basename()) + ".r" + std::to_string(i) + ".";
    out[p + "issued"] = results_[i].issued;
    out[p + "response"] = results_[i].response;
  }
}

}  // namespace apm
