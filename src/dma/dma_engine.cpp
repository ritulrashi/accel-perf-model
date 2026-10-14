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
    : sc_module(name),
      dram_socket("dram_socket"),
      sram_socket("sram_socket"),
      p_(p),
      buffers_free_(p.max_bursts_in_flight) {
  if (p_.burst_bytes == 0 || p_.max_bursts_in_flight == 0)
    throw std::invalid_argument("DmaParams: burst_bytes and max_bursts_in_flight must be >= 1");
  dram_socket.register_nb_transport_bw(this, &DmaEngine::nb_transport_bw_dram);
  sram_socket.register_nb_transport_bw(this, &DmaEngine::nb_transport_bw_sram);

  // Runs once at time 0 (initialization) to pick up descriptors submitted
  // before sc_start, then whenever woken.
  SC_METHOD(act);
  sensitive << act_now_ << act_next_;
}

std::size_t DmaEngine::submit(const TileDesc& d) {
  if (d.rows == 0 || d.row_bytes == 0) throw std::invalid_argument("DMA: empty descriptor");
  const bool running = sc_core::sc_is_running();
  if (running) require(at_boundary(), "DMA: submit off a cycle boundary");

  const std::size_t id = results_.size();
  DescResult res;
  res.submit = running ? now_cycle() : 0;
  res.bytes = static_cast<std::uint64_t>(d.rows) * d.row_bytes;

  // Split every row into bursts aligned to burst_bytes in DRAM address space,
  // so no burst crosses a DRAM row (row_bytes is a multiple of burst_bytes).
  for (std::uint32_t r = 0; r < d.rows; ++r) {
    std::uint64_t s = d.src + r * d.src_stride;
    std::uint64_t t = d.dst + r * d.dst_stride;
    std::uint64_t left = d.row_bytes;
    while (left > 0) {
      const std::uint64_t dram_addr = d.dir == DmaDir::DramToSram ? s : t;
      const std::uint64_t chunk = std::min(left, p_.burst_bytes - dram_addr % p_.burst_bytes);
      auto b = std::make_unique<Burst>();
      b->desc = id;
      b->idx = res.bursts;
      b->dir = d.dir;
      b->data.assign(chunk, 0);
      const auto len = static_cast<std::uint32_t>(chunk);
      init_payload(b->rd, tlm::TLM_READ_COMMAND, s, b->data.data(), len);
      init_payload(b->wr, tlm::TLM_WRITE_COMMAND, t, b->data.data(), len);
      if (trace_) {
        b->rd_ext.segs = b->wr_ext.segs = &b->segs;
        b->rd.set_extension(&b->rd_ext);
        b->wr.set_extension(&b->wr_ext);
      }
      to_read_.push_back(std::move(b));
      ++res.bursts;
      s += chunk;
      t += chunk;
      left -= chunk;
    }
  }
  results_.push_back(res);
  remaining_.push_back(res.bursts);
  archive_.emplace_back(trace_ ? res.bursts : 0);
  if (running) act_now_.notify(sc_core::SC_ZERO_TIME);
  return id;
}

void DmaEngine::issue(Socket& s, tlm_generic_payload& t) {
  busy(s) = true;
  tlm::tlm_phase phase = tlm::BEGIN_REQ;
  sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
  const tlm::tlm_sync_enum r = s->nb_transport_fw(t, phase, delay);
  require(r == tlm::TLM_ACCEPTED, "DMA: target must return TLM_ACCEPTED to BEGIN_REQ");
}

void DmaEngine::act() {
  require(at_boundary(), "DMA: acting off a cycle boundary");
  // Write first: if the oldest finished read's write and the next read need
  // the same socket, the write takes it. Writes leave in read-response order.
  if (!write_q_.empty() && !busy(write_socket(*write_q_.front()))) {
    Burst* b = write_q_.front();
    write_q_.pop_front();
    issue(write_socket(*b), b->wr);
  }
  const bool may_read =
      !to_read_.empty() && buffers_free_ > 0 && !busy(read_socket(*to_read_.front()));
  if (may_read) {
    std::unique_ptr<Burst> owned = std::move(to_read_.front());
    to_read_.pop_front();
    Burst* b = owned.get();
    live_[&b->rd] = b;
    live_[&b->wr] = b;
    owned_[b] = std::move(owned);
    --buffers_free_;  // held until this burst's write response
    issue(read_socket(*b), b->rd);
  }
}

tlm::tlm_sync_enum DmaEngine::nb_transport_bw_dram(tlm_generic_payload& trans,
                                                   tlm::tlm_phase& phase, sc_core::sc_time&) {
  return on_bw(dram_socket, trans, phase);
}

tlm::tlm_sync_enum DmaEngine::nb_transport_bw_sram(tlm_generic_payload& trans,
                                                   tlm::tlm_phase& phase, sc_core::sc_time&) {
  return on_bw(sram_socket, trans, phase);
}

tlm::tlm_sync_enum DmaEngine::on_bw(Socket& s, tlm_generic_payload& trans, tlm::tlm_phase& phase) {
  if (phase == tlm::END_REQ) {
    busy(s) = false;
    // END_REQ arrives at a sample point; the next request goes at the next boundary.
    act_next_.notify(until_cycle(now_cycle() + 1));
    return tlm::TLM_ACCEPTED;
  }
  require(phase == tlm::BEGIN_RESP, "DMA: unexpected phase on backward path");
  require(trans.is_response_ok(), "DMA: error response");
  const auto it = live_.find(&trans);
  require(it != live_.end(), "DMA: response for unknown burst");
  Burst* b = it->second;
  if (&trans == &b->rd) {
    write_q_.push_back(b);
  } else {
    ++buffers_free_;
    const std::size_t id = b->desc;
    if (trace_) archive_[id][b->idx] = BurstRecord{results_[id].submit, now_cycle(), std::move(b->segs)};
    live_.erase(&b->rd);
    live_.erase(&b->wr);
    owned_.erase(b);  // the burst (and its payloads) is finished
    if (--remaining_[id] == 0) {
      results_[id].done = now_cycle();
      results_[id].complete = true;
      if (on_done_) on_done_(id, results_[id].done);
    }
  }
  act_now_.notify(sc_core::SC_ZERO_TIME);
  return tlm::TLM_COMPLETED;
}

void DmaEngine::report(std::map<std::string, std::uint64_t>& out) const {
  std::uint64_t done = 0;
  for (std::size_t i = 0; i < results_.size(); ++i) {
    const std::string p = std::string(basename()) + ".d" + std::to_string(i) + ".";
    out[p + "submit"] = results_[i].submit;
    out[p + "done"] = results_[i].done;
    out[p + "bursts"] = results_[i].bursts;
    out[p + "bytes"] = results_[i].bytes;
    done += results_[i].complete ? 1 : 0;
  }
  out[std::string(basename()) + ".descriptors_done"] = done;
}

}  // namespace apm
