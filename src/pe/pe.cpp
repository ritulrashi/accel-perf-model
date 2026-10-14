#include "pe/pe.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>


namespace apm {

Pe::Pe(sc_core::sc_module_name name, const PeConfig& cfg, const GemmLayout& job)
    : sc_module(name),
      lmem("lmem", cfg.lmem),
      dma("dma", cfg.dma),
      cfg_(cfg),
      layout_(job),
      acc_(cfg.array) {
  // Validation and both capacity-check layers (§2.4) happen in plan_pe().
  const PeBuffers bufs = plan_pe(cfg, job);
  a_buf_ = bufs.a;
  b_buf_ = bufs.b;
  out_buf_ = bufs.out;

  const std::uint64_t steps = layout_.steps(), tiles = layout_.tiles();
  loads_done_.assign(steps, 0);
  loaded_at_.assign(steps, -1);
  feed_start_.assign(steps, -1);
  tile_done_.assign(tiles, -1);
  store_done_.assign(tiles, -1);

  dma.sram_socket.bind(lmem.socket);
  dma.set_done_callback([this](std::size_t id, Cycle at) { on_dma_done(id, at); });

  SC_METHOD(act);  // also runs once at time 0
  sensitive << wake_now_ << wake_timed_;
}

bool Pe::operand_buffer_free(std::uint64_t step, Cycle now) const {
  if (step < cfg_.n_buf) return true;
  const std::int64_t f = feed_start_[step - cfg_.n_buf];
  // A buffer is free once its chunk's feed has ended (Â§2.3 rule 2).
  return f >= 0 && static_cast<Cycle>(f) + cfg_.kt <= now;
}

void Pe::issue_loads(std::uint64_t step) {
  const std::uint64_t t = step / layout_.chunks(), c = step % layout_.chunks();
  const std::uint64_t mi = t / layout_.tiles_n(), nj = t % layout_.tiles_n();
  const unsigned buf = static_cast<unsigned>(step % cfg_.n_buf);
  const std::uint64_t a_sz = layout_.a_block_bytes(), b_sz = layout_.b_block_bytes();

  record(dma.submit({DmaDir::DramToSram, layout_.a_block_addr(mi, c), a_buf_[buf], 1, a_sz, a_sz,
                     a_sz}),
         {Kind::LoadA, step});
  record(dma.submit({DmaDir::DramToSram, layout_.b_block_addr(nj, c), b_buf_[buf], 1, b_sz, b_sz,
                     b_sz}),
         {Kind::LoadB, step});
}

void Pe::try_feed(Cycle now) {
  if (next_feed_ >= layout_.steps() || feeding_until_ > now) return;
  const std::uint64_t s = next_feed_;
  if (loads_done_[s] < 2) return;
  const std::uint64_t t = s / layout_.chunks(), c = s % layout_.chunks();
  if (c == 0) {
    if (tile_in_array_) return;  // one accumulator set: wait for the previous drain
    if (t >= cfg_.n_out && store_done_[t - cfg_.n_out] < 0) return;  // output buffer still busy
    acc_.clear();
    tile_in_array_ = true;
    cur_tile_ = t;
  }
  feed_start_[s] = static_cast<std::int64_t>(now);
  feeding_until_ = now + cfg_.kt;
  ++next_feed_;

  // Functional data path: the whole chunk is applied at feed start. The
  // operands cannot change before feed end because their buffer is not
  // reloaded until then.
  const unsigned buf = static_cast<unsigned>(s % cfg_.n_buf);
  std::vector<std::int8_t> a(layout_.a_block_bytes()), b(layout_.b_block_bytes());
  lmem.peek(a_buf_[buf], reinterpret_cast<std::uint8_t*>(a.data()), a.size());
  lmem.peek(b_buf_[buf], reinterpret_cast<std::uint8_t*>(b.data()), b.size());
  acc_.accumulate(a.data(), b.data(), cfg_.kt);

  if (c + 1 == layout_.chunks()) {  // last chunk: the drain follows (§2.2)
    tile_done_at_ = tile_done(cfg_.array, now, cfg_.kt);
    drain_pending_ = true;
  }
}

void Pe::finish_tile(Cycle now) {
  const std::uint64_t t = cur_tile_;
  const std::uint64_t mi = t / layout_.tiles_n(), nj = t % layout_.tiles_n();
  const std::uint64_t out = out_buf_[t % cfg_.n_out];
  const auto& v = acc_.values();
  // The drain writes INT32 results (little-endian, host order) into the output buffer.
  std::vector<std::uint8_t> bytes(v.size() * 4);
  std::memcpy(bytes.data(), v.data(), bytes.size());
  lmem.poke(out, bytes.data(), bytes.size());
  tile_in_array_ = false;
  drain_pending_ = false;
  tile_done_[t] = static_cast<std::int64_t>(now);

  const std::uint64_t row = 4ull * cfg_.array.cols;
  record(dma.submit({DmaDir::SramToDram, out, layout_.c_tile_addr(mi, nj), cfg_.array.rows, row,
                     row, 4 * layout_.shape.n}),
         {Kind::Store, t});
}

void Pe::record(std::size_t desc_id, DescInfo info) {
  if (desc_id >= desc_info_.size()) desc_info_.resize(desc_id + 1);
  desc_info_[desc_id] = info;
}

void Pe::on_dma_done(std::size_t id, Cycle at) {
  // Called from inside the TLM backward path: record and wake, nothing else.
  const DescInfo& d = desc_info_.at(id);
  if (d.kind == Kind::Store) {
    store_done_[d.index] = static_cast<std::int64_t>(at);
    ++stores_done_;
    done_cycle_ = std::max(done_cycle_, at);
  } else if (++loads_done_[d.index] == 2) {
    loaded_at_[d.index] = static_cast<std::int64_t>(at);
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
  while (next_load_ < layout_.steps() && operand_buffer_free(next_load_, now)) issue_loads(next_load_++);
  try_feed(now);
  arm_wake(now);
}

void Pe::report(std::map<std::string, std::uint64_t>& out) const {
  const std::string p = std::string(basename()) + ".";
  out[p + "done"] = done_cycle_;
  out[p + "lmem_required"] = cfg_.lmem_required();
  for (std::uint64_t s = 0; s < layout_.steps(); ++s) {
    out[p + "s" + std::to_string(s) + ".loaded"] = static_cast<std::uint64_t>(loaded_at_[s]);
    out[p + "s" + std::to_string(s) + ".feed"] = static_cast<std::uint64_t>(feed_start_[s]);
  }
  for (std::uint64_t t = 0; t < layout_.tiles(); ++t) {
    out[p + "t" + std::to_string(t) + ".done"] = static_cast<std::uint64_t>(tile_done_[t]);
    out[p + "t" + std::to_string(t) + ".store_done"] = static_cast<std::uint64_t>(store_done_[t]);
  }
}

}  // namespace apm
