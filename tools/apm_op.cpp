// Runs one op of an attention program (Phase 4 design note §4.3) on a DRAM
// image and writes the image back. Each op starts from an idle machine with
// all DRAM banks closed; the Python front end (python/attn.py) chains ops by
// passing the image along. Prints cycles, stall stacks and counters as JSON.
//
//   apm_op op=gemm    img=IN out_img=OUT M=.. N=.. K=.. R=.. C=.. Kt=.. px=.. py=..
//          a_base=.. b_base=.. c_base=.. [prec m_valid px_map py_map sram_x sram_y D nbuf nout
//          hop link sram_banks gdma_m pe_dma_m q]
//   apm_op op=softmax img=IN out_img=OUT rows=.. n=.. s_base=.. p_base=.. R=.. Kt=..
//          [prec p_scale=fixed|row causal=0|1 q_off alpha_bits scale_base V t_vu vbuf vout
//          vmem_banks vu_dma_m]
//   common DRAM keys: dram_w (16) t_hit (10) t_miss (30) dram_q (8) dram_banks (4) dram_row (1024)
//                     dram_map (row | xor; Phase 5)
//   gemm only: schedule=FILE writes every PE's feed schedule (Phase 5, INT8)
//
// A and B of a GEMM must already be packed at a_base / b_base (Phase 2 §3.1
// layout, e-byte elements); C is written row-major (valid rows) at c_base.
#include <systemc>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "soc/accel_run.h"
#include "soc/soc_plan.h"
#include "vec/vu_run.h"

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const std::vector<std::uint8_t>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
  if (!f) throw std::runtime_error("cannot write " + path);
}

}  // namespace

int sc_main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto eq = a.find('=');
      if (eq == std::string::npos) throw std::runtime_error("expected key=value, got " + a);
      kv[a.substr(0, eq)] = a.substr(eq + 1);
    }
    auto num = [&](const char* k) -> std::uint64_t {
      const auto it = kv.find(k);
      if (it == kv.end()) throw std::runtime_error(std::string("missing ") + k);
      return std::stoull(it->second, nullptr, 0);
    };
    auto opt = [&](const char* k, std::uint64_t def) { return kv.count(k) ? num(k) : def; };
    auto u = [](std::uint64_t v) { return static_cast<unsigned>(v); };
    auto str = [&](const char* k, const char* def) { return kv.count(k) ? kv.at(k) : std::string(def); };

    apm::DramParams dram;
    dram.banks = u(opt("dram_banks", 4));
    dram.row_bytes = opt("dram_row", 1024);
    dram.bus_bytes_per_cycle = u(opt("dram_w", 16));
    dram.t_hit = opt("t_hit", 10);
    dram.t_miss = opt("t_miss", 30);
    dram.queue_depth = u(opt("dram_q", 8));
    {
      const std::string m = str("dram_map", "row");
      if (m != "row" && m != "xor") throw std::runtime_error("dram_map must be row or xor");
      dram.bank_map = m == "xor" ? apm::BankMap::Xor : apm::BankMap::Row;
    }
    const apm::Precision prec = apm::parse_precision(str("prec", "int8"));
    std::vector<std::uint8_t> image = read_file(kv.at("img"));
    const std::string op = str("op", "");

    if (op == "gemm") {
      apm::AccelConfig cfg;
      cfg.dram = dram;
      if (kv.count("schedule")) cfg.schedule_path = kv.at("schedule");
      apm::GridConfig& g = cfg.grid;
      g.pe.array = {u(num("R")), u(num("C"))};
      g.pe.kt = num("Kt");
      g.pe.prec = prec;
      g.pe.q = u(opt("q", 0));
      g.px = u(num("px"));
      g.py = u(num("py"));
      g.px_map = u(opt("px_map", 0));
      g.py_map = u(opt("py_map", 0));
      g.sram_node = {u(opt("sram_x", 0)), u(opt("sram_y", 0))};
      g.ring_slots = u(opt("D", 2));
      g.pe.n_buf = u(opt("nbuf", 2));
      g.pe.n_out = u(opt("nout", 2));
      g.pe.dma.max_bursts_in_flight = u(opt("pe_dma_m", 8));
      g.gdma.max_bursts_in_flight = u(opt("gdma_m", 8));
      g.hop = opt("hop", 2);
      g.link_bytes = u(opt("link", 16));
      g.sram.banks = u(opt("sram_banks", 8));
      g.pe.lmem.capacity_bytes = g.pe.lmem_required();

      apm::GemmLayout l;
      l.shape = {num("M"), num("N"), num("K")};
      l.array = g.pe.array;
      l.kt = g.pe.kt;
      l.prec = prec;
      l.m_valid = opt("m_valid", 0);
      l.a_base = num("a_base");
      l.b_base = num("b_base");
      l.c_base = num("c_base");
      apm::GridConfig big = g;
      big.sram.capacity_bytes = ~0ull >> 1;
      g.sram.capacity_bytes = apm::SocPlan(big, l).sram_required();

      const apm::AccelRunResult r = apm::run_accel_image(cfg, l, std::move(image));
      write_file(kv.at("out_img"), r.image);
      std::cout << r.json();
      return r.complete ? 0 : 1;
    }
    if (op == "softmax") {
      apm::VuConfig cfg;
      cfg.lanes = u(opt("V", 16));
      cfg.t_vu = opt("t_vu", 4);
      cfg.n_vbuf = u(opt("vbuf", 2));
      cfg.n_vout = u(opt("vout", 2));
      cfg.vmem.banks = u(opt("vmem_banks", 8));
      cfg.dma.max_bursts_in_flight = u(opt("vu_dma_m", 8));
      apm::VuJob j;
      j.rows = num("rows");
      j.n = num("n");
      j.spec.prec = prec;
      const std::string ps = str("p_scale", "fixed");
      if (ps != "fixed" && ps != "row") throw std::runtime_error("p_scale must be fixed or row");
      j.spec.p_scale = ps == "row" ? apm::PScale::Row : apm::PScale::Fixed;
      j.spec.causal = opt("causal", 0) != 0;
      j.spec.q_off = opt("q_off", 0);
      j.spec.alpha = apm::bits_to_f32(static_cast<std::uint32_t>(opt("alpha_bits", 0x3f800000)));
      j.s_base = num("s_base");
      j.p_base = num("p_base");
      j.r_pack = u(num("R"));
      j.kt = num("Kt");
      j.scale_base = opt("scale_base", 0);
      cfg.vmem.capacity_bytes = apm::vmem_required(cfg, j);

      const apm::VuRunResult r = apm::run_vu_image(dram, cfg, j, std::move(image));
      write_file(kv.at("out_img"), r.image);
      std::cout << r.json();
      return r.complete ? 0 : 1;
    }
    throw std::runtime_error("op must be gemm or softmax");
  } catch (const std::exception& e) {
    std::cerr << "apm_op: " << e.what() << "\n";
    return 2;
  }
}
