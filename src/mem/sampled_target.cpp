#include "mem/sampled_target.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace apm {

using sc_core::sc_time;
using tlm::tlm_generic_payload;
using tlm::tlm_phase;
using tlm::tlm_sync_enum;

SampledTarget::SampledTarget(sc_core::sc_module_name name, std::uint64_t capacity_bytes)
    : sc_module(name), socket("socket"), store_(capacity_bytes, 0), resp_peq_("resp_peq") {
  socket.register_nb_transport_fw(this, &SampledTarget::nb_transport_fw);
  socket.register_transport_dbg(this, &SampledTarget::transport_dbg);

  SC_METHOD(sample_proc);
  sensitive << sample_ev_;
  dont_initialize();

  SC_METHOD(response_proc);
  sensitive << resp_peq_.get_event();
  dont_initialize();
}

void SampledTarget::end_of_elaboration() {
  pending_.assign(socket.size(), nullptr);
  if (socket.size() > 0) arbiter_ = std::make_unique<RrArbiter>(socket.size());
}

void SampledTarget::poke(std::uint64_t addr, const std::uint8_t* src, std::size_t len) {
  require(addr + len <= store_.size(), "poke beyond capacity");
  std::memcpy(store_.data() + addr, src, len);
}

void SampledTarget::peek(std::uint64_t addr, std::uint8_t* dst, std::size_t len) const {
  require(addr + len <= store_.size(), "peek beyond capacity");
  std::memcpy(dst, store_.data() + addr, len);
}

tlm_sync_enum SampledTarget::nb_transport_fw(int port, tlm_generic_payload& trans,
                                             tlm_phase& phase, sc_time& delay) {
  if (phase == tlm::END_RESP) return tlm::TLM_COMPLETED;  // not used by our initiators
  require(phase == tlm::BEGIN_REQ, "SampledTarget: unexpected phase on forward path");
  // Initiators only send at a cycle boundary with no annotated delay (§3.1).
  require(delay == sc_core::SC_ZERO_TIME, "SampledTarget: BEGIN_REQ with non-zero delay");
  require(at_boundary(), "SampledTarget: BEGIN_REQ not at a cycle boundary");
  require(pending_[port] == nullptr, "SampledTarget: BEGIN_REQ before END_REQ on the same port");
  require(trans.get_byte_enable_ptr() == nullptr, "SampledTarget: byte enables not supported");
  pending_[port] = &trans;
  // Several notifications for the same sample point collapse into one.
  sample_ev_.notify(until_sample(now_cycle()));
  return tlm::TLM_ACCEPTED;
}

unsigned int SampledTarget::transport_dbg(int, tlm_generic_payload& trans) {
  const std::uint64_t addr = trans.get_address();
  const unsigned int len = trans.get_data_length();
  if (addr + len > store_.size()) return 0;
  if (trans.is_read())
    std::memcpy(trans.get_data_ptr(), store_.data() + addr, len);
  else
    std::memcpy(store_.data() + addr, trans.get_data_ptr(), len);
  return len;
}

void SampledTarget::copy_data(tlm_generic_payload& trans) {
  const std::uint64_t addr = trans.get_address();
  const unsigned int len = trans.get_data_length();
  require(addr + len <= store_.size(), "SampledTarget: access beyond capacity");
  if (trans.is_read())
    std::memcpy(trans.get_data_ptr(), store_.data() + addr, len);
  else if (trans.is_write())
    std::memcpy(store_.data() + addr, trans.get_data_ptr(), len);
  else
    throw std::logic_error("SampledTarget: IGNORE command not supported");
  trans.set_response_status(tlm::TLM_OK_RESPONSE);
}

void SampledTarget::sample_proc() {
  require(at_sample_point(), "SampledTarget: sample process off the sample point");
  const Cycle c = now_cycle();
  begin_sample(c);

  bool granted_any = false;
  unsigned first_granted = 0;
  unsigned still_pending = 0;
  for (unsigned port : arbiter_->order()) {
    tlm_generic_payload* trans = pending_[port];
    if (trans == nullptr) continue;
    Cycle response = 0;
    if (!accept(*trans, c, response)) {
      ++still_pending;
      continue;
    }
    require(response > c, "SampledTarget: response not after acceptance");
    if (!granted_any) first_granted = port;
    granted_any = true;
    pending_[port] = nullptr;
    copy_data(*trans);
    port_of_[trans] = static_cast<int>(port);
    resp_peq_.notify(*trans, until_cycle(response));

    tlm_phase phase = tlm::END_REQ;
    sc_time delay = sc_core::SC_ZERO_TIME;
    const tlm_sync_enum r = socket[port]->nb_transport_bw(*trans, phase, delay);
    require(r == tlm::TLM_ACCEPTED, "SampledTarget: initiator must accept END_REQ");
  }
  if (granted_any) arbiter_->granted_first(first_granted);
  backpressure_ += still_pending;
  // Refused requests are reconsidered at the next sample point.
  if (still_pending > 0) sample_ev_.notify(until_sample(c + 1));
}

void SampledTarget::response_proc() {
  require(at_boundary(), "SampledTarget: response off a cycle boundary");
  while (tlm_generic_payload* trans = resp_peq_.get_next_transaction()) {
    const auto it = port_of_.find(trans);
    require(it != port_of_.end(), "SampledTarget: response for unknown transaction");
    const int port = it->second;
    port_of_.erase(it);
    tlm_phase phase = tlm::BEGIN_RESP;
    sc_time delay = sc_core::SC_ZERO_TIME;
    const tlm_sync_enum r = socket[port]->nb_transport_bw(*trans, phase, delay);
    require(r == tlm::TLM_COMPLETED, "SampledTarget: initiator must complete BEGIN_RESP");
  }
}

}  // namespace apm
