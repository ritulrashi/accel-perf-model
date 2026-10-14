#include "soc/accel_run.h"

#include <systemc>

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "common/sim_guard.h"
#include "compute/systolic.h"

namespace apm {

namespace {

constexpr Cycle kMaxCycles = 200'000'000;

std::uint64_t round_up(std::uint64_t x, std::uint64_t to) { return (x + to - 1) / to * to; }

void check(bool ok, const std::string& what) {
  if (!ok) throw std::logic_error("stall-stack assertion failed: " + what);
}

std::vector<BurstRecord> concat(const std::vector<BurstRecord>& a, const std::vector<BurstRecord>& b) {
  std::vector<BurstRecord> v = a;
  v.insert(v.end(), b.begin(), b.end());
  return v;
}

}  // namespace

GemmLayout accel_layout(const AccelRun& run) {
  GemmLayout l;
  l.shape = run.shape;
  l.array = run.cfg.grid.pe.array;
  l.kt = run.cfg.grid.pe.kt;
  l.prec = run.cfg.grid.pe.prec;
  l.m_valid = run.m_valid;
  const std::uint64_t row = run.cfg.dram.row_bytes;
  l.a_base = 0;
  l.b_base = round_up(l.a_packed_bytes(), row);
  l.c_base = round_up(l.b_base + l.b_packed_bytes(), row);
  return l;
}

namespace {

// Builds the accelerator, preloads DRAM, runs to completion and computes the
// stall stack with every §3.3 assertion. `image_bytes` > 0: read back that
// many bytes of DRAM into the result's image.
AccelRunResult simulate(AccelConfig cfg, const GemmLayout& l,
                        const std::function<void(DramChannel&)>& preload, std::uint64_t image_bytes) {
  claim_simulation();
  check_time_resolution();
  // All configuration errors surface here, before any module is built.
  const SocPlan plan(cfg.grid, l);

  auto top = std::make_unique<Accel>("top", cfg, plan);
  preload(top->dram);

  sc_core::sc_start(sc_core::sc_time::from_value(kMaxCycles * kPsPerCycle), sc_core::SC_EXIT_ON_STARVATION);
  require(!sc_core::sc_pending_activity(), "accelerator run did not finish within kMaxCycles");

  AccelRunResult r;
  r.complete = top->sched.finished();
  for (const auto& pe : top->pes) r.complete = r.complete && pe->finished();
  if (!r.complete) throw std::logic_error("accelerator run ended incomplete (deadlock?)");
  r.t_end = top->sched.t_end();

  // ---- stall stack
  const GridConfig& g = cfg.grid;
  const Cycle fd = fill_drain(g.pe.array);
  for (unsigned p = 0; p < plan.pes(); ++p) {
    const Pe& pe = *top->pes[p];
    const GridJob& job = *top->jobs[p];
    PeTrace t;
    t.kt = g.pe.feed_cycles();  // the classifier's "feed length" (q*Kt)
    t.fill_drain = fd;
    t.chunks = plan.chunks();
    t.n_out = g.pe.n_out;
    for (std::uint64_t s = 0; s < pe.step_records().size(); ++s) {
      const Pe::StepRec& sr = pe.step_records()[s];
      StepTrace st;
      st.loads_submit = sr.loads_submit;
      st.loaded = sr.loaded;
      st.feed = sr.feed;
      st.tile = s / plan.chunks();
      st.a = top->sched.stage_rec(job.set_of(s), job.a_block(s));
      st.b = top->sched.stage_rec(job.set_of(s), job.b_block(s));
      if (sr.loads_submit >= 0)
        st.load_bursts = concat(pe.dma.burst_records(sr.a_desc), pe.dma.burst_records(sr.b_desc));
      t.steps.push_back(std::move(st));
    }
    for (const Pe::TileRec& tr : pe.tile_records()) {
      TileTrace tt;
      tt.done = tr.done;
      tt.store_submit = tr.store_submit;
      tt.store_done = tr.store_done;
      if (tr.store_submit >= 0) tt.store_bursts = pe.dma.burst_records(tr.store_desc);
      t.tiles.push_back(std::move(tt));
    }
    const StallCounts n = classify_pe(t, r.t_end, p);

    std::uint64_t sum = 0;
    for (std::size_t k = 0; k < kStallCats; ++k) {
      sum += n[k];
      r.total[k] += n[k];
    }
    check(sum == r.t_end, "PE " + std::to_string(p) + ": categories sum to " + std::to_string(sum) +
                              ", T_end is " + std::to_string(r.t_end));
    const std::uint64_t compute_formula =
        plan.tiles_of(p).size() * (plan.chunks() * g.pe.feed_cycles() + fd);
    check(n[static_cast<std::size_t>(StallCat::Compute)] == compute_formula,
          "PE " + std::to_string(p) + ": compute " +
              std::to_string(n[static_cast<std::size_t>(StallCat::Compute)]) + " != tiles*(nc*q*Kt+F) = " +
              std::to_string(compute_formula));
    r.stalls.push_back(n);
  }
  std::uint64_t chip = 0;
  for (std::uint64_t v : r.total) chip += v;
  check(chip == plan.pes() * r.t_end, "chip categories sum to " + std::to_string(chip) + ", P*T_end is " +
                                          std::to_string(plan.pes() * r.t_end));

  // ---- counters and DRAM-byte cross-check
  top->dram.report(r.values);
  top->sram.report(r.values);
  top->mesh.report(r.values);
  const std::uint64_t dram_bytes = r.values.at("dram.bytes_read") + r.values.at("dram.bytes_written");
  check(dram_bytes == plan.dram_bytes_exact(), "DRAM bytes " + std::to_string(dram_bytes) +
                                                   " != plan " + std::to_string(plan.dram_bytes_exact()));
  if (plan.full_waves())
    check(plan.dram_bytes_exact() == plan.dram_bytes_formula(), "plan DRAM bytes != §2.2 formula");

  r.values["t_end"] = r.t_end;
  r.values["pes"] = plan.pes();
  r.values["ops"] = 2 * l.shape.m * l.shape.n * l.shape.k;
  r.values["useful_ops"] = 2 * l.valid_rows() * l.shape.n * l.shape.k;
  r.values["q"] = g.pe.cycles_per_step();
  r.values["elem_bytes"] = l.elem();
  r.values["dram_bytes"] = dram_bytes;
  r.values["full_waves"] = plan.full_waves() ? 1 : 0;
  for (std::size_t k = 0; k < kStallCats; ++k) {
    const std::string name = stall_name(static_cast<StallCat>(k));
    r.values["stall." + name] = r.total[k];
    for (unsigned p = 0; p < plan.pes(); ++p)
      r.values["pe" + std::to_string(p) + ".stall." + name] = r.stalls[p][k];
  }

  std::vector<std::uint8_t> raw(l.c_bytes());
  top->dram.peek(l.c_base, raw.data(), raw.size());
  r.c.resize(l.valid_rows() * l.shape.n);
  std::memcpy(r.c.data(), raw.data(), raw.size());
  if (image_bytes) {
    r.image.resize(image_bytes);
    top->dram.peek(0, r.image.data(), image_bytes);
  }
  return r;
}

}  // namespace

AccelRunResult run_accel(AccelRun run) {
  const GemmLayout l = accel_layout(run);
  l.validate();
  run.cfg.dram.capacity_bytes =
      std::max(run.cfg.dram.capacity_bytes, round_up(l.c_base + l.c_bytes(), run.cfg.dram.row_bytes));
  std::vector<std::uint8_t> pa, pb;
  if (!run.a_bytes.empty() || !run.b_bytes.empty()) {
    pa = l.pack_a_bytes(run.a_bytes);
    pb = l.pack_b_bytes(run.b_bytes);
  } else {
    if (l.prec != Precision::Int8) throw std::invalid_argument("run_accel: non-INT8 runs need a_bytes/b_bytes");
    pa = l.pack_a(run.a);
    pb = l.pack_b(run.b);
  }
  return simulate(run.cfg, l, [&](DramChannel& d) {
    d.poke(l.a_base, pa.data(), pa.size());
    d.poke(l.b_base, pb.data(), pb.size());
  }, 0);
}

AccelRunResult run_accel_image(AccelConfig cfg, const GemmLayout& l, std::vector<std::uint8_t> image) {
  l.validate();
  if (l.a_base + l.a_packed_bytes() > image.size() || l.b_base + l.b_packed_bytes() > image.size() ||
      l.c_base + l.c_bytes() > image.size())
    throw std::invalid_argument("run_accel_image: a GEMM region lies outside the DRAM image");
  if (image.size() % cfg.dram.row_bytes != 0)
    throw std::invalid_argument("run_accel_image: image size must be a multiple of the DRAM row");
  cfg.dram.capacity_bytes = image.size();
  return simulate(cfg, l, [&](DramChannel& d) { d.poke(0, image.data(), image.size()); }, image.size());
}

std::string AccelRunResult::json() const {
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
