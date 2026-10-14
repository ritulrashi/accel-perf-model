#include "vec/vu_run.h"

#include <systemc>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "common/cycle.h"
#include "common/sim_guard.h"
#include "mem/dram_channel.h"
#include "vec/vec_unit.h"

namespace apm {

namespace {

constexpr Cycle kMaxCycles = 200'000'000;

void check(bool ok, const std::string& what) {
  if (!ok) throw std::logic_error("VU assertion failed: " + what);
}

class VuTop : public sc_core::sc_module {
 public:
  DramChannel dram;
  VecUnit vu;

  VuTop(sc_core::sc_module_name name, const DramParams& d, const VuConfig& c, const VuJob& j)
      : sc_module(name), dram("dram", d), vu("vu", c, j) {
    vu.dma.dram_socket.bind(dram.socket);
  }
};

}  // namespace

VuRunResult run_vu_image(DramParams dram, const VuConfig& cfg, const VuJob& job, std::vector<std::uint8_t> image) {
  claim_simulation();
  check_time_resolution();
  plan_vu(cfg, job);  // configuration errors before any module is built
  const std::uint64_t s_end = job.s_base + job.rows * job.n * 4;
  const std::uint64_t p_end = job.p_base + job.p_region_bytes();
  if (s_end > image.size() || p_end > image.size() || job.scale_base + job.scale_bytes() > image.size())
    throw std::invalid_argument("run_vu_image: a VU region lies outside the DRAM image");
  if (image.size() % dram.row_bytes != 0)
    throw std::invalid_argument("run_vu_image: image size must be a multiple of the DRAM row");
  dram.capacity_bytes = image.size();

  auto top = std::make_unique<VuTop>("top", dram, cfg, job);
  top->dram.poke(0, image.data(), image.size());
  sc_core::sc_start(sc_core::sc_time::from_value(kMaxCycles * kPsPerCycle), sc_core::SC_EXIT_ON_STARVATION);
  require(!sc_core::sc_pending_activity(), "VU run did not finish within kMaxCycles");

  VuRunResult r;
  r.complete = top->vu.finished();
  if (!r.complete) throw std::logic_error("VU run ended incomplete (deadlock?)");
  r.t_end = top->vu.done_cycle();

  std::vector<VuRowTrace> rows;
  for (const VecUnit::RowRec& rr : top->vu.rows()) rows.push_back({rr.loaded, rr.start, rr.store_done});
  r.stalls = classify_vu(rows, top->vu.row_cycles(), cfg.n_vout, top->vu.scale_store_done(), r.t_end);
  std::uint64_t sum = 0;
  for (std::uint64_t v : r.stalls) sum += v;
  check(sum == r.t_end, "categories sum to " + std::to_string(sum) + ", T_end is " + std::to_string(r.t_end));
  const std::uint64_t busy_formula = job.rows * row_cycles(cfg, job.n);
  check(r.stalls[static_cast<std::size_t>(VuCat::Busy)] == busy_formula,
        "busy " + std::to_string(r.stalls[0]) + " != rows*T_row = " + std::to_string(busy_formula));

  top->dram.report(r.values);
  top->vu.vmem.report(r.values);
  const std::uint64_t dram_bytes = r.values.at("dram.bytes_read") + r.values.at("dram.bytes_written");
  check(dram_bytes == vu_dram_bytes(job),
        "DRAM bytes " + std::to_string(dram_bytes) + " != rows*n*(4+e) + scales = " + std::to_string(vu_dram_bytes(job)));

  r.values["t_end"] = r.t_end;
  r.values["rows"] = job.rows;
  r.values["n"] = job.n;
  r.values["t_row"] = top->vu.row_cycles();
  r.values["dram_bytes"] = dram_bytes;
  r.values["elements"] = job.rows * job.n;
  r.values["vmem_required"] = vmem_required(cfg, job);
  for (std::size_t k = 0; k < kVuCats; ++k) r.values[std::string("vu.") + vu_cat_name(static_cast<VuCat>(k))] = r.stalls[k];
  for (std::uint64_t i = 0; i < top->vu.rows().size(); ++i) {
    const VecUnit::RowRec& rr = top->vu.rows()[i];
    const std::string p = "row" + std::to_string(i) + ".";
    r.values[p + "loaded"] = static_cast<std::uint64_t>(rr.loaded);
    r.values[p + "start"] = static_cast<std::uint64_t>(rr.start);
    r.values[p + "store_done"] = static_cast<std::uint64_t>(rr.store_done);
    if (i >= 64) break;  // enough for directed tests; keeps large runs' JSON small
  }
  r.image.resize(image.size());
  top->dram.peek(0, r.image.data(), r.image.size());
  return r;
}

std::string VuRunResult::json() const {
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
