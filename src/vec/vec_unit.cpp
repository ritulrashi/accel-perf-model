#include "vec/vec_unit.h"

#include <algorithm>
#include <cstring>

namespace apm {

VecUnit::VecUnit(sc_core::sc_module_name name, const VuConfig& cfg, const VuJob& job)
    : sc_module(name),
      vmem("vmem", cfg.vmem),
      dma("dma", cfg.dma),
      cfg_(cfg),
      job_(job),
      bufs_(plan_vu(cfg, job)),
      t_row_(apm::row_cycles(cfg, job.n)) {
  job_.spec.lanes = cfg.lanes;
  rows_.assign(job.rows, RowRec{});
  scales_.assign(job.rows, 0.0f);
  dma.sram_socket.bind(vmem.socket);
  dma.set_done_callback([this](std::size_t id, Cycle at) { on_dma_done(id, at); });
  SC_METHOD(act);  // also runs once at time 0
  sensitive << wake_now_ << wake_timed_;
}

bool VecUnit::finished() const {
  return stores_done_ == job_.rows && (!job_.writes_scales() || scale_done_ >= 0);
}

void VecUnit::record(std::size_t id, DescInfo d) {
  if (id >= desc_info_.size()) desc_info_.resize(id + 1);
  desc_info_[id] = d;
}

void VecUnit::on_dma_done(std::size_t id, Cycle at) {
  // Inside the TLM backward path: record and wake only.
  const DescInfo& d = desc_info_.at(id);
  if (d.kind == Kind::Load) {
    rows_[d.row].loaded = static_cast<std::int64_t>(at);
  } else if (d.kind == Kind::Store) {
    rows_[d.row].store_done = static_cast<std::int64_t>(at);
    ++stores_done_;
  } else {
    scale_done_ = static_cast<std::int64_t>(at);
  }
  done_cycle_ = std::max(done_cycle_, at);
  wake_now_.notify(sc_core::SC_ZERO_TIME);
}

void VecUnit::act() {
  require(at_boundary(), "VecUnit: acting off a cycle boundary");
  const Cycle now = now_cycle();
  const unsigned e = job_.elem();

  // 1. Finish the row in the VU.
  if (computing_ && busy_until_ == now) {
    const std::uint64_t r = cur_row_;
    computing_ = false;
    rows_[r].store_submit = static_cast<std::int64_t>(now);
    record(dma.submit(vu_store(job_, bufs_.out[r % cfg_.n_vout], r)), {Kind::Store, r});
    if (r + 1 == job_.rows && job_.writes_scales()) {
      std::vector<std::uint8_t> raw(job_.scale_bytes(), 0);
      std::memcpy(raw.data(), scales_.data(), 4 * job_.rows);
      vmem.poke(bufs_.scales, raw.data(), raw.size());
      record(dma.submit(vu_scale_store(job_, bufs_.scales)), {Kind::Scales, r});
    }
  }
  require(!computing_ || busy_until_ > now, "VecUnit: missed a compute-end deadline");

  // 2. Loads into free input buffers.
  while (next_load_ < job_.rows) {
    const std::uint64_t r = next_load_;
    if (r >= cfg_.n_vbuf) {
      const std::int64_t s = rows_[r - cfg_.n_vbuf].start;
      if (s < 0 || static_cast<Cycle>(s) + t_row_ > now) break;
    }
    rows_[r].load_submit = static_cast<std::int64_t>(now);
    record(dma.submit(vu_load(job_, bufs_.in[r % cfg_.n_vbuf], r)), {Kind::Load, r});
    ++next_load_;
  }

  // 3. Start the next row.
  if (!computing_ && next_compute_ < job_.rows) {
    const std::uint64_t r = next_compute_;
    const bool loaded = rows_[r].loaded >= 0;
    const bool out_free = r < cfg_.n_vout || rows_[r - cfg_.n_vout].store_done >= 0;
    if (loaded && out_free) {
      std::vector<std::uint8_t> s(4 * job_.n), p(job_.n * e);
      vmem.peek(bufs_.in[r % cfg_.n_vbuf], s.data(), s.size());
      scales_[r] = softmax_row(job_.spec, r, s.data(), job_.n, p.data());
      vmem.poke(bufs_.out[r % cfg_.n_vout], p.data(), p.size());
      rows_[r].start = static_cast<std::int64_t>(now);
      computing_ = true;
      cur_row_ = r;
      busy_until_ = now + t_row_;
      ++next_compute_;
    }
  }
  if (computing_) wake_timed_.notify(until_cycle(busy_until_));
}

}  // namespace apm
