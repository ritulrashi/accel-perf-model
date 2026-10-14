#include "test/scenarios.h"

#include <systemc>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "common/cycle.h"
#include "common/sim_guard.h"
#include "dma/dma_engine.h"
#include "mem/dram_channel.h"
#include "mem/sram.h"
#include "test/traffic_gen.h"

namespace apm {

namespace {

// Generous bound: a stuck model ends with an error instead of hanging.
constexpr Cycle kMaxCycles = 1'000'000;

struct Spec {
  DramParams dram;
  SramParams sram;
  DmaParams dma;
  std::vector<TileDesc> dma_descs;               // DMA engine i runs dma_descs[i]
  std::vector<TileDesc> dma0_more;               // further descriptors for engine 0
  std::vector<std::vector<TgRequest>> tg_dram;  // bound to DRAM after the DMAs
  std::vector<std::vector<TgRequest>> tg_sram;  // bound to SRAM after the DMAs
};

TileDesc d2s(std::uint64_t src, std::uint64_t dst, std::uint64_t bytes) {
  return TileDesc{DmaDir::DramToSram, src, dst, 1, bytes, bytes, bytes};
}

TileDesc s2d(std::uint64_t src, std::uint64_t dst, std::uint64_t bytes) {
  return TileDesc{DmaDir::SramToDram, src, dst, 1, bytes, bytes, bytes};
}

TgRequest rd(Cycle at, std::uint64_t addr, std::uint32_t size) { return {at, false, addr, size}; }
TgRequest wr(Cycle at, std::uint64_t addr, std::uint32_t size) { return {at, true, addr, size}; }

const std::map<std::string, Spec>& specs() {
  static const std::map<std::string, Spec> m = [] {
    std::map<std::string, Spec> s;
    s["t1"].dma_descs = {d2s(0, 0, 256)};
    s["t1b"].dma_descs = {d2s(0, 0, 256)};
    s["t1b"].dma.max_bursts_in_flight = 1;
    s["t1c"].dma_descs = {s2d(0, 0, 256)};
    s["t2"].dma_descs = {d2s(0, 0, 256), d2s(1024, 4096, 256)};
    s["t2b"].dma_descs = {d2s(0, 0, 256)};
    s["t2b"].dram.queue_depth = 2;
    s["t3a"].tg_sram = {{wr(0, 0, 16)}, {wr(0, 128, 16)}};
    s["t3b"].tg_sram = {{wr(0, 0, 16)}, {wr(0, 16, 16)}};
    s["t3c"].tg_sram = {{wr(0, 0, 256)}};
    s["t4a"].tg_dram = {{rd(0, 0, 64), rd(100, 64, 64)}};
    s["t4b"].tg_dram = {{rd(0, 0, 64), rd(100, 4096, 64)}};
    s["t4c"].tg_dram = {{rd(0, 0, 64), rd(1, 4096, 64)}};
    s["t4d"].tg_dram = {{rd(0, 0, 64), rd(1, 1024, 64)}};
    // Phase 5 §6: the xor bank map (Phase 1 DRAM configuration otherwise).
    s["t5a"].tg_dram = {{rd(0, 0, 64), rd(1, 4096, 64)}};
    s["t5a"].dram.bank_map = BankMap::Xor;
    s["t5b"].tg_dram = {{rd(0, 0, 64), rd(1, 5120, 64)}};
    s["t5b"].dram.bank_map = BankMap::Xor;
    s["t5c"].tg_dram = {{wr(0, 0, 64), wr(1, 4096, 64), wr(2, 8192, 64), wr(3, 12288, 64)}};
    s["t5d"].tg_dram = s["t5c"].tg_dram;
    s["t5d"].dram.bank_map = BankMap::Xor;
    s["f2d_d2s"].dma_descs = {TileDesc{DmaDir::DramToSram, 0, 0, 4, 128, 1024, 256}};
    s["f2d_s2d"].dma_descs = {TileDesc{DmaDir::SramToDram, 0, 8192, 4, 128, 256, 1024}};
    // Phase 2 P2-T6: two descriptors stream back to back through one DMA
    // (P2 config: W=32, t_hit=t_miss=10, Q=16, M=8).
    Spec& t6 = s["p2t6"];
    t6.dram.bus_bytes_per_cycle = 32;
    t6.dram.t_hit = t6.dram.t_miss = 10;
    t6.dram.queue_depth = 16;
    t6.dma.max_bursts_in_flight = 8;
    t6.dma_descs = {d2s(0, 0, 256)};
    t6.dma0_more = {d2s(2048, 256, 256)};
    return s;
  }();
  return m;
}

class Top : public sc_core::sc_module {
 public:
  DramChannel dram;
  Sram sram;
  std::vector<std::unique_ptr<DmaEngine>> dmas;
  std::vector<std::unique_ptr<TrafficGen>> tgs;

  Top(sc_core::sc_module_name name, const Spec& s)
      : sc_module(name), dram("dram", s.dram), sram("sram", s.sram) {
    // Binding order = port number on each target = round-robin order (§3.3).
    for (std::size_t i = 0; i < s.dma_descs.size(); ++i) {
      dmas.push_back(std::make_unique<DmaEngine>(("dma" + std::to_string(i)).c_str(), s.dma));
      dmas.back()->dram_socket.bind(dram.socket);
      dmas.back()->sram_socket.bind(sram.socket);
      dmas.back()->submit(s.dma_descs[i]);
      if (i == 0)
        for (const auto& d : s.dma0_more) dmas.back()->submit(d);
    }
    for (std::size_t i = 0; i < s.tg_dram.size(); ++i) {
      tgs.push_back(std::make_unique<TrafficGen>(("tgd" + std::to_string(i)).c_str(), s.tg_dram[i]));
      tgs.back()->socket.bind(dram.socket);
    }
    for (std::size_t i = 0; i < s.tg_sram.size(); ++i) {
      tgs.push_back(std::make_unique<TrafficGen>(("tgs" + std::to_string(i)).c_str(), s.tg_sram[i]));
      tgs.back()->socket.bind(sram.socket);
    }
  }
};

// Initial memory contents: an address-dependent pattern, different per memory.
std::vector<std::uint8_t> pattern(std::uint64_t size, std::uint8_t salt) {
  std::vector<std::uint8_t> v(size);
  for (std::uint64_t a = 0; a < size; ++a)
    v[a] = static_cast<std::uint8_t>(((a * 2654435761ull) >> 16) ^ salt);
  return v;
}

// Functional reference (rule 4): a plain loop over each descriptor applied to
// a copy of the initial images. Shares no code with DmaEngine.
void apply_reference(const TileDesc& d, const std::vector<std::uint8_t>& dram0,
                     const std::vector<std::uint8_t>& sram0, std::vector<std::uint8_t>& dram_exp,
                     std::vector<std::uint8_t>& sram_exp) {
  const bool to_sram = d.dir == DmaDir::DramToSram;
  const std::vector<std::uint8_t>& src = to_sram ? dram0 : sram0;
  std::vector<std::uint8_t>& dst = to_sram ? sram_exp : dram_exp;
  for (std::uint64_t r = 0; r < d.rows; ++r)
    for (std::uint64_t i = 0; i < d.row_bytes; ++i)
      dst.at(d.dst + r * d.dst_stride + i) = src.at(d.src + r * d.src_stride + i);
}

}  // namespace

std::vector<std::string> scenario_names() {
  std::vector<std::string> n;
  for (const auto& [name, spec] : specs()) n.push_back(name);
  return n;
}

ScenarioResult run_scenario(const std::string& name) {
  claim_simulation();
  check_time_resolution();

  const auto it = specs().find(name);
  if (it == specs().end()) throw std::invalid_argument("unknown scenario: " + name);
  const Spec& spec = it->second;

  auto top = std::make_unique<Top>("top", spec);
  const auto dram0 = pattern(spec.dram.capacity_bytes, 0x5a);
  const auto sram0 = pattern(spec.sram.capacity_bytes, 0xc3);
  top->dram.poke(0, dram0.data(), dram0.size());
  top->sram.poke(0, sram0.data(), sram0.size());

  sc_core::sc_start(sc_core::sc_time::from_value(kMaxCycles * kPsPerCycle),
                    sc_core::SC_EXIT_ON_STARVATION);
  require(!sc_core::sc_pending_activity(), "scenario did not finish within kMaxCycles");

  ScenarioResult r;
  r.name = name;
  top->dram.report(r.values);
  top->sram.report(r.values);
  r.complete = true;
  for (const auto& d : top->dmas) {
    d->report(r.values);
    for (const auto& res : d->results()) r.complete = r.complete && res.complete;
  }
  for (const auto& t : top->tgs) {
    t->report(r.values);
    for (const auto& res : t->results()) r.complete = r.complete && res.done;
  }

  if (!spec.dma_descs.empty()) {
    auto dram_exp = dram0;
    auto sram_exp = sram0;
    for (const auto& d : spec.dma_descs) apply_reference(d, dram0, sram0, dram_exp, sram_exp);
    for (const auto& d : spec.dma0_more) apply_reference(d, dram0, sram0, dram_exp, sram_exp);
    std::vector<std::uint8_t> dram_got(dram0.size()), sram_got(sram0.size());
    top->dram.peek(0, dram_got.data(), dram_got.size());
    top->sram.peek(0, sram_got.data(), sram_got.size());
    r.data_checked = true;
    r.data_ok = dram_got == dram_exp && sram_got == sram_exp;
  }
  return r;
}

std::string ScenarioResult::json() const {
  std::ostringstream o;
  o << "{\"scenario\": \"" << name << "\", \"complete\": " << (complete ? "true" : "false")
    << ", \"data_checked\": " << (data_checked ? "true" : "false")
    << ", \"data_ok\": " << (data_ok ? "true" : "false") << ", \"values\": {";
  bool first = true;
  for (const auto& [k, v] : values) {
    o << (first ? "" : ", ") << "\"" << k << "\": " << v;
    first = false;
  }
  o << "}}\n";
  return o.str();
}

}  // namespace apm
