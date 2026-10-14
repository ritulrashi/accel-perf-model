// Runs one INT8 GEMM on the Phase 3 accelerator grid; prints cycles, the
// stall stack (per PE and total) and counters as JSON. Every run enforces the
// stall-stack assertions (src/soc/accel_run.h).
//
//   apm_accel M=.. N=.. K=.. R=.. C=.. Kt=.. px=.. py=.. [key=value ...]
//             [a=A.bin b=B.bin out=C.bin | seed=N]
//
// Optional keys: sram_x, sram_y (sram_node, default 0,0), D (ring slots, 2),
// nbuf, nout (2, 2), dram_w (16), t_hit (10), t_miss (30), dram_q (8),
// hop (2), link (16), sram_banks (8), sram_bytes / lmem_bytes (0 = exactly
// what the plan needs), gdma_m, pe_dma_m (8, 8).
// Phase 4: prec (int8 | fp16 | bf16 | fp32, default int8), q (0 = e),
// m_valid (0 = M), px_map, py_map (logical mapping, 0 = physical).
// Phase 5: dram_map (row | xor), schedule=FILE (every PE's feed schedule).
// a/b files hold row-major e-byte elements of the precision; C is written as
// 4-byte words (INT32, or FP32 bits).
// Without a/b files the operands are pseudo-random from `seed` (default 1);
// operand values never affect timing.
#include <systemc>

#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "soc/accel_run.h"

namespace {

std::vector<std::uint8_t> read_bytes(const std::string& path, std::uint64_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (raw.size() != n) throw std::runtime_error(path + ": expected " + std::to_string(n) + " bytes");
  std::vector<std::uint8_t> v(n);
  std::memcpy(v.data(), raw.data(), n);
  return v;
}

// Pseudo-random operands of precision p: INT8 values, or floats in [-1, 1)
// on a 1/128 grid (representable in every floating-point mode).
std::vector<std::uint8_t> random_elems(apm::Precision p, std::uint64_t n, std::uint32_t seed) {
  const unsigned e = apm::elem_bytes(p);
  std::vector<std::uint8_t> v(n * e);
  for (std::uint64_t i = 0; i < n; ++i) {
    seed = seed * 1103515245u + 12345u;
    const auto x = static_cast<std::int8_t>(seed >> 16);
    if (p == apm::Precision::Int8) {
      v[i] = static_cast<std::uint8_t>(x);
      continue;
    }
    const float f = static_cast<float>(x) / 128.0f;
    std::uint32_t w = p == apm::Precision::Fp16   ? apm::f32_to_fp16(f)
                      : p == apm::Precision::Bf16 ? apm::f32_to_bf16(f)
                                                  : apm::f32_to_bits(f);
    for (unsigned b = 0; b < e; ++b) v[i * e + b] = static_cast<std::uint8_t>(w >> (8 * b));
  }
  return v;
}

std::vector<std::int8_t> lcg(std::uint64_t n, std::uint32_t seed) {
  std::vector<std::int8_t> v(n);
  for (auto& x : v) {
    seed = seed * 1103515245u + 12345u;
    x = static_cast<std::int8_t>(seed >> 16);
  }
  return v;
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
      return std::stoull(it->second);
    };
    auto opt = [&](const char* k, std::uint64_t def) { return kv.count(k) ? num(k) : def; };
    auto u = [](std::uint64_t v) { return static_cast<unsigned>(v); };

    apm::AccelRun run;
    run.shape = {num("M"), num("N"), num("K")};
    apm::GridConfig& g = run.cfg.grid;
    g.pe.array = {u(num("R")), u(num("C"))};
    g.pe.kt = num("Kt");
    g.px = u(num("px"));
    g.py = u(num("py"));
    g.sram_node = {u(opt("sram_x", 0)), u(opt("sram_y", 0))};
    g.ring_slots = u(opt("D", 2));
    g.pe.n_buf = u(opt("nbuf", 2));
    g.pe.n_out = u(opt("nout", 2));
    g.pe.dma.max_bursts_in_flight = u(opt("pe_dma_m", 8));
    g.gdma.max_bursts_in_flight = u(opt("gdma_m", 8));
    g.hop = opt("hop", 2);
    g.link_bytes = u(opt("link", 16));
    g.sram.banks = u(opt("sram_banks", 8));
    g.pe.prec = apm::parse_precision(kv.count("prec") ? kv.at("prec") : "int8");
    g.pe.q = u(opt("q", 0));
    g.px_map = u(opt("px_map", 0));
    g.py_map = u(opt("py_map", 0));
    run.m_valid = opt("m_valid", 0);
    if (kv.count("schedule")) run.cfg.schedule_path = kv.at("schedule");
    if (kv.count("dram_map")) run.cfg.dram.bank_map = kv.at("dram_map") == "xor" ? apm::BankMap::Xor : apm::BankMap::Row;
    if (kv.count("dram_map") && kv.at("dram_map") != "xor" && kv.at("dram_map") != "row")
      throw std::runtime_error("dram_map must be row or xor");
    run.cfg.dram.bus_bytes_per_cycle = u(opt("dram_w", 16));
    run.cfg.dram.t_hit = opt("t_hit", 10);
    run.cfg.dram.t_miss = opt("t_miss", 30);
    run.cfg.dram.queue_depth = u(opt("dram_q", 8));

    const std::uint64_t lmem = opt("lmem_bytes", 0);
    g.pe.lmem.capacity_bytes = lmem ? lmem : g.pe.lmem_required();
    const std::uint64_t sram = opt("sram_bytes", 0);
    if (sram) {
      g.sram.capacity_bytes = sram;
    } else {  // exactly what the plan needs
      apm::GridConfig big = g;
      big.sram.capacity_bytes = ~0ull >> 1;
      g.sram.capacity_bytes = apm::SocPlan(big, apm::accel_layout(run)).sram_required();
    }

    const std::uint64_t m = run.shape.m, n = run.shape.n, k = run.shape.k;
    const unsigned e = apm::elem_bytes(g.pe.prec);
    if (kv.count("a")) {
      run.a_bytes = read_bytes(kv.at("a"), m * k * e);
      run.b_bytes = read_bytes(kv.at("b"), k * n * e);
    } else if (g.pe.prec == apm::Precision::Int8) {  // Phase 3 operand stream, unchanged
      const auto seed = static_cast<std::uint32_t>(opt("seed", 1));
      run.a = lcg(m * k, seed);
      run.b = lcg(k * n, seed * 7919u + 1);
    } else {
      const auto seed = static_cast<std::uint32_t>(opt("seed", 1));
      run.a_bytes = random_elems(g.pe.prec, m * k, seed);
      run.b_bytes = random_elems(g.pe.prec, k * n, seed * 7919u + 1);
    }

    const apm::AccelRunResult r = apm::run_accel(run);
    if (kv.count("out")) {
      std::ofstream out(kv.at("out"), std::ios::binary);
      out.write(reinterpret_cast<const char*>(r.c.data()),
                static_cast<std::streamsize>(r.c.size() * sizeof(std::int32_t)));
      if (!out) throw std::runtime_error("cannot write " + kv.at("out"));
    }
    std::cout << r.json();
    return r.complete ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "apm_accel: " << e.what() << "\n";
    return 2;
  }
}
