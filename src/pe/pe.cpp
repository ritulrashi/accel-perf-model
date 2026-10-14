#include "pe/pe.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace apm {

Pe::Pe(sc_core::sc_module_name name, const PeConfig& cfg, PeJob& job)
    : sc_module(name),
      lmem("lmem", cfg.lmem),
      dma("dma", cfg.dma),
      cfg_(cfg),
      job_(job),
      acc_(cfg.array, cfg.prec) {
  // Both LMEM capacity-check layers (Phase 2 §2.4) happen in plan_pe_buffers().
  const PeBuffers bufs = plan_pe_buffers(cfg);
  a_buf_ = bufs.a;
  b_buf_ = bufs.b;
  out_buf_ = bufs.out;

  loads_done_.assign(job_.steps(), 0);
  steps_.assign(job_.steps(), StepRec{});
  tiles_.assign(job_.tiles(), TileRec{});

  dma.sram_socket.bind(lmem.socket);
  dma.set_done_callback([this](std::size_t id, Cycle at) { on_dma_done(id, at); });

  SC_METHOD(act);  // also runs once at time 0
  sensitive << wake_now_ << wake_timed_;
}

bool Pe::operand_buffer_free(std::uint64_t step, Cycle now) const {
  if (step < cfg_.n_buf) return true;
  const std::int64_t f = steps_[step - cfg_.n_buf].feed;
  // A buffer is free once its chunk's feed has ended (Phase 2 §2.3 rule 2).
  return f >= 0 && static_cast<Cycle>(f) + cfg_.feed_cycles() <= now;
}

void Pe::issue_loads(std::uint64_t step, Cycle now) {
  const unsigned buf = static_cast<unsigned>(step % cfg_.n_buf);
  const std::uint64_t a_sz = cfg_.a_block_bytes();
  const std::uint64_t b_sz = cfg_.b_block_bytes();
  StepRec& r = steps_[step];
  r.loads_submit = static_cast<std::int64_t>(now);
  r.a_desc = dma.submit({DmaDir::DramToSram, job_.a_src(step), a_buf_[buf], 1, a_sz, a_sz, a_sz});
  record(r.a_desc, {Kind::LoadA, step});
  r.b_desc = dma.submit({DmaDir::DramToSram, job_.b_src(step), b_buf_[buf], 1, b_sz, b_sz, b_sz});
  record(r.b_desc, {Kind::LoadB, step});
}

void Pe::try_feed(Cycle now) {
  if (next_feed_ >= job_.steps() || feeding_until_ > now) return;
  const std::uint64_t s = next_feed_;
  if (loads_done_[s] < 2) return;
  const std::uint64_t t = s / job_.chunks(), c = s % job_.chunks();
  if (c == 0) {
    if (tile_in_array_) return;  // one accumulator set: wait for the previous drain
    if (t >= cfg_.n_out && tiles_[t - cfg_.n_out].store_done < 0) return;  // output buffer still busy
    acc_.clear();
    tile_in_array_ = true;
    cur_tile_ = t;
  }
  steps_[s].feed = static_cast<std::int64_t>(now);
  feeding_until_ = now + cfg_.feed_cycles();
  ++next_feed_;

  // Functional data path: the whole chunk is applied at feed start. The
  // operands cannot change before feed end because their buffer is not
  // reloaded until then.
  const unsigned buf = static_cast<unsigned>(s % cfg_.n_buf);
  std::vector<std::uint8_t> a(cfg_.a_block_bytes()), b(cfg_.b_block_bytes());
  lmem.peek(a_buf_[buf], a.data(), a.size());
  lmem.peek(b_buf_[buf], b.data(), b.size());
  acc_.accumulate_bytes(a.data(), b.data(), cfg_.kt);
  if (capture_) {
    cap_a_[s] = a;
    cap_b_[s] = b;
  }

  if (c + 1 == job_.chunks()) {  // last chunk: the drain follows (Phase 2 §2.2)
    tile_done_at_ = tile_done(cfg_.array, now, cfg_.feed_cycles());
    drain_pending_ = true;
  }
}

void Pe::finish_tile(Cycle now) {
  const std::uint64_t t = cur_tile_;
  const std::uint64_t out = out_buf_[t % cfg_.n_out];
  // The drain writes INT32 / FP32 results (little-endian, host order) into the output buffer.
  const std::vector<std::uint8_t> bytes = acc_.bytes();
  lmem.poke(out, bytes.data(), bytes.size());
  if (capture_) cap_c_[t] = bytes;
  tile_in_array_ = false;
  drain_pending_ = false;
  tiles_[t].done = static_cast<std::int64_t>(now);
  store_q_.push_back(t);
}

void Pe::try_stores(Cycle now) {
  while (!store_q_.empty() && job_.store_allowed(store_q_.front())) {
    const std::uint64_t t = store_q_.front();
    store_q_.pop_front();
    tiles_[t].store_submit = static_cast<std::int64_t>(now);
    tiles_[t].store_desc = dma.submit(job_.store(t, out_buf_[t % cfg_.n_out]));
    record(tiles_[t].store_desc, {Kind::Store, t});
  }
}

void Pe::record(std::size_t desc_id, DescInfo info) {
  if (desc_id >= desc_info_.size()) desc_info_.resize(desc_id + 1);
  desc_info_[desc_id] = info;
}

void Pe::on_dma_done(std::size_t id, Cycle at) {
  // Called from inside the TLM backward path: record and wake, nothing else.
  const DescInfo& d = desc_info_.at(id);
  if (d.kind == Kind::Store) {
    tiles_[d.index].store_done = static_cast<std::int64_t>(at);
    ++stores_done_;
    done_cycle_ = std::max(done_cycle_, at);
    job_.on_store_done(d.index, at);
  } else if (++loads_done_[d.index] == 2) {
    steps_[d.index].loaded = static_cast<std::int64_t>(at);
    job_.on_loads_done(d.index, at);
  }
  wake_now_.notify(sc_core::SC_ZERO_TIME);
}

void Pe::arm_wake(Cycle now) {
  // sc_event keeps only the earliest pending notification, so always ask for
  // the earliest future deadline; later ones are re-armed when it fires.
  Cycle next = 0;
  bool any = false;
  auto consider = [&](Cycle c) {
    if (c > now && (!any || c < next)) {
      next = c;
      any = true;
    }
  };
  consider(feeding_until_);
  if (drain_pending_) consider(tile_done_at_);
  if (any) wake_timed_.notify(until_cycle(next));
}

void Pe::act() {
  require(at_boundary(), "Pe: acting off a cycle boundary");
  const Cycle now = now_cycle();
  if (drain_pending_) {
    require(tile_done_at_ >= now, "Pe: missed a tile-done deadline");
    if (tile_done_at_ == now) finish_tile(now);
  }
  try_stores(now);
  while (next_load_ < job_.steps() && operand_buffer_free(next_load_, now) &&
         job_.operands_ready(next_load_))
    issue_loads(next_load_++, now);
  try_feed(now);
  arm_wake(now);
}

void Pe::capture_schedule(bool on) {
  if (on && cfg_.prec != Precision::Int8)
    throw std::invalid_argument("Pe: schedule capture is INT8 only (the RTL scope)");
  if (on && cfg_.cycles_per_step() != 1) throw std::invalid_argument("Pe: schedule capture needs q = 1");
  capture_ = on;
  cap_a_.assign(job_.steps(), {});
  cap_b_.assign(job_.steps(), {});
  cap_c_.assign(job_.tiles(), {});
}

PeSchedule Pe::schedule(unsigned pe_id) const {
  if (!capture_) throw std::logic_error("Pe: schedule requested without capture");
  PeSchedule s;
  s.pe = pe_id;
  s.rows = cfg_.array.rows;
  s.cols = cfg_.array.cols;
  s.kt = cfg_.kt;
  s.chunks = job_.chunks();
  for (std::uint64_t i = 0; i < steps_.size(); ++i) {
    if (steps_[i].feed < 0) throw std::logic_error("Pe: schedule of an unfinished run");
    s.steps.push_back({i / job_.chunks(), i % job_.chunks(), static_cast<Cycle>(steps_[i].feed), cap_a_[i], cap_b_[i]});
  }
  for (std::uint64_t t = 0; t < tiles_.size(); ++t)
    s.tiles.push_back({t, static_cast<Cycle>(tiles_[t].done), cap_c_[t]});
  return s;
}

void Pe::report(std::map<std::string, std::uint64_t>& out) const {
  const std::string p = std::string(basename()) + ".";
  out[p + "done"] = done_cycle_;
  out[p + "lmem_required"] = cfg_.lmem_required();
  for (std::uint64_t s = 0; s < steps_.size(); ++s) {
    out[p + "s" + std::to_string(s) + ".loaded"] = static_cast<std::uint64_t>(steps_[s].loaded);
    out[p + "s" + std::to_string(s) + ".feed"] = static_cast<std::uint64_t>(steps_[s].feed);
  }
  for (std::uint64_t t = 0; t < tiles_.size(); ++t) {
    out[p + "t" + std::to_string(t) + ".done"] = static_cast<std::uint64_t>(tiles_[t].done);
    out[p + "t" + std::to_string(t) + ".store_done"] = static_cast<std::uint64_t>(tiles_[t].store_done);
  }
}

}  // namespace apm
