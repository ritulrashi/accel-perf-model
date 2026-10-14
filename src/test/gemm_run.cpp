#include "test/gemm_run.h"

#include <systemc>

#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "common/cycle.h"
#include "common/sim_guard.h"
#include "mem/dram_channel.h"

namespace apm {

namespace {

constexpr Cycle kMaxCycles = 100'000'000;

std::uint64_t round_up(std::uint64_t x, std::uint64_t to) { return (x + to - 1) / to * to; }

class GemmTop : public sc_core::sc_module {
 public:
  DramChannel dram;
  DramGemmJob job;  // declared before pe: the PE keeps a reference
  Pe pe;

  GemmTop(sc_core::sc_module_name name, const DramParams& d, const PeConfig& p, const GemmLayout& l)
      : sc_module(name), dram("dram", d), job(l), pe("pe", p, job) {
    pe.dma.dram_socket.bind(dram.socket);
  }
};

}  // namespace

GemmLayout layout_for(const GemmRun& run) {
  GemmLayout l;
  l.shape = run.shape;
  l.array = run.pe.array;
  l.kt = run.pe.kt;
  if (run.auto_layout) {
    const std::uint64_t row = run.dram.row_bytes;
    l.a_base = 0;
    l.b_base = round_up(l.a_packed_bytes(), row);
    l.c_base = round_up(l.b_base + l.b_packed_bytes(), row);
  } else {
    l.a_base = run.a_base;
    l.b_base = run.b_base;
    l.c_base = run.c_base;
  }
  return l;
}

GemmRunResult run_gemm(GemmRun run) {
  claim_simulation();
  check_time_resolution();
  const GemmLayout l = layout_for(run);
  // All configuration errors (including LMEM capacity) surface here, before
  // any SystemC module is constructed.
  plan_pe(run.pe, l);
  const std::uint64_t end = l.c_base + l.c_bytes();
  if (run.auto_layout)
    run.dram.capacity_bytes = std::max(run.dram.capacity_bytes, round_up(end, run.dram.row_bytes));
  if (end > run.dram.capacity_bytes || l.a_base + l.a_packed_bytes() > l.b_base ||
      l.b_base + l.b_packed_bytes() > l.c_base)
    throw std::invalid_argument("run_gemm: DRAM regions overlap or exceed capacity");

  auto top = std::make_unique<GemmTop>("top", run.dram, run.pe, l);
  const auto pa = l.pack_a(run.a);
  const auto pb = l.pack_b(run.b);
  top->dram.poke(l.a_base, pa.data(), pa.size());
  top->dram.poke(l.b_base, pb.data(), pb.size());

  sc_core::sc_start(sc_core::sc_time::from_value(kMaxCycles * kPsPerCycle),
                    sc_core::SC_EXIT_ON_STARVATION);
  require(!sc_core::sc_pending_activity(), "GEMM did not finish within kMaxCycles");

  GemmRunResult r;
  r.complete = top->pe.finished();
  top->dram.report(r.values);
  top->pe.lmem.report(r.values);
  top->pe.dma.report(r.values);
  top->pe.report(r.values);
  std::vector<std::uint8_t> raw(l.c_bytes());
  top->dram.peek(l.c_base, raw.data(), raw.size());
  r.c.resize(l.shape.m * l.shape.n);
  std::memcpy(r.c.data(), raw.data(), raw.size());
  return r;
}

std::string GemmRunResult::json() const {
  std::ostringstream o;
  o << "{\"complete\": " << (complete ? "true" : "false") << ", \"values\": {";
  bool first = true;
  for (const auto& [k, v] : values) {
    o << (first ? "" : ", ") << "\"" << k << "\": " << v;
    first = false;
  }
  o << "}}\n";
  return o.str();
}

}  // namespace apm
