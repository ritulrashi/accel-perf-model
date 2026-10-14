#include "dma/dma_engine.h"

#include <algorithm>
#include <stdexcept>

namespace apm {

using tlm::tlm_generic_payload;

namespace {

void init_payload(tlm_generic_payload& t, tlm::tlm_command cmd, std::uint64_t addr,
                  std::uint8_t* data, std::uint32_t len) {
  t.set_command(cmd);
  t.set_address(addr);
  t.set_data_ptr(data);
  t.set_data_length(len);
  t.set_streaming_width(len);
  t.set_byte_enable_ptr(nullptr);
  t.set_dmi_allowed(false);
  t.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
}

}  // namespace

DmaEngine::DmaEngine(sc_core::sc_module_name name, const DmaParams& p)
    : sc_module(name), dram_socket("dram_socket"), sram_socket("sram_socket"), p_(p) {
  if (p_.burst_bytes == 0 || p_.max_bursts_in_flight == 0)
    throw std::invalid_argument("DmaParams: burst_bytes and max_bursts_in_flight must be >= 1");
  dram_socket.register_nb_transport_bw(this, &DmaEngine::nb_transport_bw);
  sram_socket.register_nb_transport_bw(this, &DmaEngine::nb_transport_bw);

  // Runs once at time 0 (initialization) to pick up descriptors submitted
  // before sc_start, then whenever woken.
  SC_METHOD(act);
  sensitive << act_now_ << act_next_;
}

void DmaEngine::submit(const TileDesc& d) {
  if (d.rows == 0 || d.row_bytes == 0) throw std::invalid_argument("DMA: empty descriptor");
  const bool running = sc_core::sc_is_running();
  if (running) require(at_boundary(), "DMA: submit off a cycle boundary");
  queue_.emplace_back(d, running ? now_cycle() : 0);
  if (running) act_now_.notify(sc_core::SC_ZERO_TIME);
}

tlm_utils::simple_initiator_socket<DmaEngine>& DmaEngine::read_socket() {
  return cur_.dir == DmaDir::DramToSram ? dram_socket : sram_socket;
}

tlm_utils::simple_initiator_socket<DmaEngine>& DmaEngine::write_socket() {
  return cur_.dir == DmaDir::DramToSram ? sram_socket : dram_socket;
}

void DmaEngine::start_next_descriptor() {
  const auto [d, submitted] = queue_.front();
  queue_.pop_front();
  cur_ = d;
  cur_result_ = DescResult{};
  cur_result_.submit = submitted;
  bursts_.clear();
  burst_of_.clear();
  write_q_.clear();
  next_read_ = writes_done_ = 0;
  buffers_free_ = p_.max_bursts_in_flight;

  // Split every row into bursts aligned to burst_bytes in DRAM address space,
  // so no burst crosses a DRAM row (row_bytes is a multiple of burst_bytes).
  for (std::uint32_t r = 0; r < d.rows; ++r) {
    std::uint64_t s = d.src + r * d.src_stride;
    std::uint64_t t = d.dst + r * d.dst_stride;
    std::uint64_t left = d.row_bytes;
    while (left > 0) {
      const std::uint64_t dram_addr = d.dir == DmaDir::DramToSram ? s : t;
      const std::uint64_t chunk =
          std::min(left, p_.burst_bytes - dram_addr % p_.burst_bytes);
      auto b = std::make_unique<Burst>();
      b->src = s;
      b->dst = t;
      b->size = static_cast<std::uint32_t>(chunk);
      b->data.assign(chunk, 0);
      init_payload(b->rd, tlm::TLM_READ_COMMAND, s, b->data.data(), b->size);
      init_payload(b->wr, tlm::TLM_WRITE_COMMAND, t, b->data.data(), b->size);
      burst_of_[&b->rd] = b.get();
      burst_of_[&b->wr] = b.get();
      bursts_.push_back(std::move(b));
      s += chunk;
      t += chunk;
      left -= chunk;
    }
  }
  cur_result_.bursts = bursts_.size();
  cur_result_.bytes = static_cast<std::uint64_t>(d.rows) * d.row_bytes;
  active_ = true;
}

void DmaEngine::issue(tlm_utils::simple_initiator_socket<DmaEngine>& s, tlm_generic_payload& t) {
  tlm::tlm_phase phase = tlm::BEGIN_REQ;
  sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
  const tlm::tlm_sync_enum r = s->nb_transport_fw(t, phase, delay);
  require(r == tlm::TLM_ACCEPTED, "DMA: target must return TLM_ACCEPTED to BEGIN_REQ");
}

void DmaEngine::act() {
  require(at_boundary(), "DMA: acting off a cycle boundary");
  if (!active_ && !queue_.empty()) start_next_descriptor();
  if (!active_) return;

  // Writes go out in read-response order, one per cycle per socket.
  if (!write_busy_ && !write_q_.empty()) {
    Burst* b = write_q_.front();
    write_q_.pop_front();
    write_busy_ = true;
    issue(write_socket(), b->wr);
  }
  if (!read_busy_ && buffers_free_ > 0 && next_read_ < bursts_.size()) {
    Burst* b = bursts_[next_read_++].get();
    --buffers_free_;  // held until this burst's write response
    read_busy_ = true;
    issue(read_socket(), b->rd);
  }
}

tlm::tlm_sync_enum DmaEngine::nb_transport_bw(tlm_generic_payload& trans, tlm::tlm_phase& phase,
                                              sc_core::sc_time&) {
  const bool is_read = trans.is_read();
  if (phase == tlm::END_REQ) {
    (is_read ? read_busy_ : write_busy_) = false;
    // END_REQ arrives at a sample point; the next request goes at the next boundary.
    act_next_.notify(until_cycle(now_cycle() + 1));
    return tlm::TLM_ACCEPTED;
  }
  require(phase == tlm::BEGIN_RESP, "DMA: unexpected phase on backward path");
  require(trans.is_response_ok(), "DMA: error response");
  const auto it = burst_of_.find(&trans);
  require(it != burst_of_.end(), "DMA: response for unknown burst");
  if (is_read) {
    write_q_.push_back(it->second);
  } else {
    ++buffers_free_;
    if (++writes_done_ == bursts_.size()) {
      cur_result_.done = now_cycle();
      results_.push_back(cur_result_);
      active_ = false;
    }
  }
  act_now_.notify(sc_core::SC_ZERO_TIME);
  return tlm::TLM_COMPLETED;
}

void DmaEngine::report(std::map<std::string, std::uint64_t>& out) const {
  for (std::size_t i = 0; i < results_.size(); ++i) {
    const std::string p = std::string(basename()) + ".d" + std::to_string(i) + ".";
    out[p + "submit"] = results_[i].submit;
    out[p + "done"] = results_[i].done;
    out[p + "bursts"] = results_[i].bursts;
    out[p + "bytes"] = results_[i].bytes;
  }
  out[std::string(basename()) + ".descriptors_done"] = results_.size();
}

}  // namespace apm
