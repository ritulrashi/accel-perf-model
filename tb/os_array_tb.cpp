// Verilator testbench for rtl/os_array.sv (Phase 5 design note §3).
//
//   os_array_tb SCHEDULE [--expect-trace FILE] [--trace-out FILE]
//
// For every PE schedule in SCHEDULE (exported by the model, or hand-written),
// drives the RTL variant with that (R, C) cycle by cycle (in cycle f+k: A
// column k and B row k of the step's blocks, with the tags; every other cycle
// a bubble) and compares three levels against the model:
//   1. tile timing: RTL first drain cycle == contract drain_start of the tile's
//      last feed, RTL done (= first drain + R) == the model's recorded tile done;
//   2. values: the R drained rows == the model's C tile, bit for bit;
//   3. every cycle: the RTL trace == contract_trace(schedule), byte for byte.
// --expect-trace (hand-written traces, one-PE schedules): the RTL trace and
// contract_trace must also both equal the file.
// Exit status 0 only if everything matches; the first mismatch is printed.
#include <verilated.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "compute/contract_trace.h"
#include "compute/schedule_io.h"
#include "rtl_variants.h"  // generated: #includes of every variant and APM_RTL_VARIANTS(X)

namespace {

using apm::Cycle;
using apm::PeSchedule;

struct Outcome {
  bool ok = true;
  std::string why;
  std::uint64_t tiles = 0, cycles = 0, macs = 0, lines = 0;
  std::vector<std::string> trace;
};

void fail(Outcome& o, const std::string& why) {
  if (o.ok) o.why = why;
  o.ok = false;
}

std::string diff_at(const std::vector<std::string>& got, const std::vector<std::string>& want, const char* gname,
                    const char* wname) {
  const std::size_t n = std::max(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    const std::string g = i < got.size() ? got[i] : "<end>";
    const std::string w = i < want.size() ? want[i] : "<end>";
    if (g != w) {
      std::ostringstream o;
      o << "trace line " << i << ":\n    " << gname << ": " << g << "\n    " << wname << ": " << w;
      return o.str();
    }
  }
  return "";
}

template <class M>
Outcome run_variant(const PeSchedule& s) {
  Outcome o;
  apm::ContractTiming timing;
  const std::vector<std::string> expect = apm::contract_trace(s, &timing);

  // Feed cycles -> (step, k).
  std::map<Cycle, std::pair<std::size_t, std::uint64_t>> feeds;
  for (std::size_t i = 0; i < s.steps.size(); ++i)
    for (std::uint64_t k = 0; k < s.kt; ++k)
      if (!feeds.emplace(s.steps[i].feed + k, std::make_pair(i, k)).second) {
        fail(o, "schedule feeds two steps in one cycle");
        return o;
      }
  Cycle end = 0;
  for (Cycle d : timing.done) end = std::max(end, d);
  end += 2;

  VerilatedContext ctx;
  M m{&ctx};
  auto clear_inputs = [&] {
    m.in_valid = 0;
    m.in_first = 0;
    m.in_last = 0;
    m.in_k = 0;
    for (unsigned i = 0; i < s.rows; ++i) m.a_col[i] = 0;
    for (unsigned j = 0; j < s.cols; ++j) m.b_row[j] = 0;
  };
  clear_inputs();
  m.rst = 1;
  m.clk = 0;
  m.eval();
  for (int r = 0; r < 2; ++r) {
    m.clk = 1;
    m.eval();
    m.clk = 0;
    m.eval();
  }
  m.rst = 0;

  struct Drain {
    Cycle at;
    unsigned row;
    std::vector<std::uint32_t> data;
  };
  std::vector<Drain> drains;
  std::vector<std::int64_t> ks(std::size_t{s.rows} * s.cols);
  for (Cycle t = 0; t <= end; ++t) {
    clear_inputs();
    const auto it = feeds.find(t);
    if (it != feeds.end()) {
      const apm::SchedStep& st = s.steps[it->second.first];
      const std::uint64_t k = it->second.second;
      m.in_valid = 1;
      m.in_first = (st.chunk == 0 && k == 0) ? 1 : 0;
      m.in_last = (st.chunk + 1 == s.chunks && k + 1 == s.kt) ? 1 : 0;
      m.in_k = static_cast<std::uint32_t>(st.chunk * s.kt + k);
      for (unsigned i = 0; i < s.rows; ++i) m.a_col[i] = st.a[i * s.kt + k];
      for (unsigned j = 0; j < s.cols; ++j) m.b_row[j] = st.b[k * s.cols + j];
    }
    m.clk = 0;
    m.eval();  // cycle t: inputs applied, combinational logic settled

    bool any = false;
    for (unsigned i = 0; i < s.rows; ++i)
      for (unsigned j = 0; j < s.cols; ++j) {
        const bool v = m.mac_valid[i][j];
        ks[std::size_t{i} * s.cols + j] = v ? static_cast<std::int64_t>(m.mac_k[i][j]) : -1;
        any = any || v;
        o.macs += v ? 1 : 0;
      }
    int drow = -1;
    if (m.drain_valid) {
      drow = static_cast<int>(m.drain_row);
      Drain d{t, static_cast<unsigned>(drow), {}};
      for (unsigned j = 0; j < s.cols; ++j) d.data.push_back(static_cast<std::uint32_t>(m.drain_data[j]));
      drains.push_back(std::move(d));
    }
    if (any || drow >= 0) o.trace.push_back(apm::trace_line(t, ks, drow));
    if (m.error) {
      fail(o, "RTL protocol error raised in cycle " + std::to_string(t));
      break;
    }
    m.clk = 1;
    m.eval();  // the edge that ends cycle t
  }
  m.final();
  o.cycles = end + 1;
  o.lines = o.trace.size();

  // Levels 1 and 2: drains come in groups of R rows, one group per tile, in order.
  if (drains.size() != std::size_t{s.rows} * s.tiles.size())
    fail(o, "level 1: RTL drained " + std::to_string(drains.size()) + " rows, expected " +
                std::to_string(std::size_t{s.rows} * s.tiles.size()));
  for (std::size_t t = 0; o.ok && t < s.tiles.size(); ++t) {
    const Drain& first = drains[t * s.rows];
    const Cycle rtl_done = first.at + s.rows;
    if (first.at != timing.drain_start.at(t))
      fail(o, "level 1: tile " + std::to_string(t) + " RTL drain starts " + std::to_string(first.at) +
                  ", contract " + std::to_string(timing.drain_start[t]));
    else if (rtl_done != s.tiles[t].done)
      fail(o, "level 1: tile " + std::to_string(t) + " RTL done " + std::to_string(rtl_done) + ", model " +
                  std::to_string(s.tiles[t].done));
    else if (timing.done.at(t) != s.tiles[t].done)
      fail(o, "level 1: tile " + std::to_string(t) + " contract done " + std::to_string(timing.done[t]) +
                  " != model's recorded done " + std::to_string(s.tiles[t].done));
    for (unsigned n = 0; o.ok && n < s.rows; ++n) {
      const Drain& d = drains[t * s.rows + n];
      if (d.at != first.at + n || d.row != s.rows - 1 - n) {
        fail(o, "level 1: tile " + std::to_string(t) + " drain cycle " + std::to_string(d.at) + " shows row " +
                    std::to_string(d.row));
        break;
      }
      for (unsigned j = 0; j < s.cols; ++j) {
        std::uint32_t want;
        std::memcpy(&want, &s.tiles[t].c[4 * (std::size_t{d.row} * s.cols + j)], 4);
        if (d.data[j] != want) {
          fail(o, "level 2: tile " + std::to_string(t) + " C[" + std::to_string(d.row) + "][" + std::to_string(j) +
                      "] RTL " + std::to_string(static_cast<std::int32_t>(d.data[j])) + ", model " +
                      std::to_string(static_cast<std::int32_t>(want)));
          break;
        }
      }
    }
  }
  o.tiles = s.tiles.size();
  // Level 3.
  if (o.ok) {
    const std::string d = diff_at(o.trace, expect, "rtl     ", "contract");
    if (!d.empty()) fail(o, "level 3: " + d);
  }
  return o;
}

Outcome run(const PeSchedule& s) {
#define APM_DISPATCH(R_, C_, CLS) \
  if (s.rows == (R_) && s.cols == (C_)) return run_variant<CLS>(s);
  APM_RTL_VARIANTS(APM_DISPATCH)
#undef APM_DISPATCH
  Outcome o;
  fail(o, "no RTL variant built for R=" + std::to_string(s.rows) + " C=" + std::to_string(s.cols));
  return o;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<std::string> v;
  std::string l;
  while (std::getline(f, l))
    if (!l.empty() && l[0] != '#') v.push_back(l);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: os_array_tb SCHEDULE [--expect-trace FILE] [--trace-out FILE]\n";
      return 2;
    }
    const std::string path = argv[1];
    std::string expect_path, trace_out;
    for (int i = 2; i + 1 < argc; i += 2) {
      const std::string k = argv[i];
      if (k == "--expect-trace") expect_path = argv[i + 1];
      else if (k == "--trace-out") trace_out = argv[i + 1];
      else throw std::runtime_error("unknown option " + k);
    }
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::vector<PeSchedule> scheds = apm::parse_schedules(f);
    if (scheds.empty()) throw std::runtime_error(path + ": no PE schedules");

    bool ok = true;
    std::string hand = "n/a";
    std::uint64_t tiles = 0, cycles = 0, macs = 0, lines = 0;
    std::ofstream tout;
    if (!trace_out.empty()) tout.open(trace_out);
    for (const PeSchedule& s : scheds) {
      Outcome o = run(s);
      tiles += o.tiles;
      cycles += o.cycles;
      macs += o.macs;
      lines += o.lines;
      if (tout.is_open())
        for (const auto& l : o.trace) tout << "pe" << s.pe << " " << l << "\n";
      if (!expect_path.empty()) {
        const std::vector<std::string> want = read_lines(expect_path);
        std::string d = diff_at(o.trace, want, "rtl ", "hand");
        const std::string dc = diff_at(apm::contract_trace(s), want, "contract", "hand    ");
        if (!d.empty() || !dc.empty()) {
          hand = "MISMATCH";
          if (o.ok) fail(o, "hand trace: " + (d.empty() ? dc : d));
        } else {
          hand = "ok";
        }
      }
      if (!o.ok) {
        ok = false;
        std::cerr << "os_array_tb: " << path << " pe " << s.pe << " (R=" << s.rows << " C=" << s.cols
                  << "): " << o.why << "\n";
      }
    }
    std::cout << "os_array_tb: file=" << path << " pes=" << scheds.size() << " tiles=" << tiles
              << " cycles=" << cycles << " macs=" << macs << " trace_lines=" << lines << " hand=" << hand
              << " result=" << (ok ? "MATCH" : "MISMATCH") << "\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "os_array_tb: " << e.what() << "\n";
    return 2;
  }
}
