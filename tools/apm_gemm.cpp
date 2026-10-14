// Runs one INT8 GEMM on the Phase 2 PE model and writes C (INT32, row-major,
// little-endian) to a file; prints all cycle counts and counters as JSON.
//
//   apm_gemm M=.. N=.. K=.. R=.. C=.. Kt=.. a=A.bin b=B.bin out=C.bin [key=value ...]
//
// Optional keys (defaults are the library defaults): nbuf, nout,
// lmem_bytes (0 = exactly what the buffers need), dram_w, t_hit, t_miss,
// dram_q, dma_m; schedule=FILE (Phase 5: write the PE's feed schedule).
// A.bin is M*K int8 row-major, B.bin is K*N int8 row-major.
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

#include "test/gemm_run.h"

namespace {

std::vector<std::int8_t> read_i8(const std::string& path, std::uint64_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (raw.size() != n) throw std::runtime_error(path + ": expected " + std::to_string(n) + " bytes");
  std::vector<std::int8_t> v(n);
  std::memcpy(v.data(), raw.data(), n);
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

    apm::GemmRun run;
    run.shape = {num("M"), num("N"), num("K")};
    run.pe.array = {static_cast<unsigned>(num("R")), static_cast<unsigned>(num("C"))};
    run.pe.kt = num("Kt");
    run.pe.n_buf = static_cast<unsigned>(opt("nbuf", run.pe.n_buf));
    run.pe.n_out = static_cast<unsigned>(opt("nout", run.pe.n_out));
    run.dram.bus_bytes_per_cycle = static_cast<unsigned>(opt("dram_w", run.dram.bus_bytes_per_cycle));
    run.dram.t_hit = opt("t_hit", run.dram.t_hit);
    run.dram.t_miss = opt("t_miss", run.dram.t_miss);
    run.dram.queue_depth = static_cast<unsigned>(opt("dram_q", run.dram.queue_depth));
    run.pe.dma.max_bursts_in_flight = static_cast<unsigned>(opt("dma_m", run.pe.dma.max_bursts_in_flight));
    const std::uint64_t lmem = opt("lmem_bytes", 0);
    run.pe.lmem.capacity_bytes = lmem ? lmem : run.pe.lmem_required();
    run.a = read_i8(kv.at("a"), run.shape.m * run.shape.k);
    run.b = read_i8(kv.at("b"), run.shape.k * run.shape.n);
    if (kv.count("schedule")) run.schedule_path = kv.at("schedule");

    const apm::GemmRunResult r = apm::run_gemm(run);
    std::ofstream out(kv.at("out"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(r.c.data()),
              static_cast<std::streamsize>(r.c.size() * sizeof(std::int32_t)));
    if (!out) throw std::runtime_error("cannot write " + kv.at("out"));
    std::cout << r.json();
    return r.complete ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "apm_gemm: " << e.what() << "\n";
    return 2;
  }
}
