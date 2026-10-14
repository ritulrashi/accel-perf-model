// Standalone C++ functional reference (rule 4). Shares no code with the model:
// a plain triple loop over row-major inputs.
//   INT8: accumulated in 64 bits and checked to fit INT32.
//   FP16 / BF16 / FP32 (Phase 4 design note §2.2): inputs decoded exactly to
//   FP32, acc = fl32(acc + fl32(a*b)) in ascending k, no fused multiply-add
//   (built with -ffp-contract=off). Output: FP32 bit patterns.
//
//   gemm_ref M N K A.bin B.bin C.bin [int8|fp16|bf16|fp32]
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

static std::vector<std::uint8_t> read_bytes(const char* path, std::uint64_t n) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (!f.good() && !f.eof()) { std::cerr << "gemm_ref: cannot read " << path << "\n"; std::exit(2); }
  if (raw.size() != n) { std::cerr << "gemm_ref: " << path << " has wrong size\n"; std::exit(2); }
  return std::vector<std::uint8_t>(raw.begin(), raw.end());
}

static float from_bits(std::uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

// IEEE half -> float, written out field by field (exact).
static float half_to_float(std::uint16_t h) {
  const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = static_cast<float>(m) * std::ldexp(1.0f, -24);
  else v = std::ldexp(static_cast<float>(m + 1024), e - 25);
  return s ? -v : v;
}

static float decode(const std::string& prec, const std::uint8_t* p) {
  if (prec == "fp16") return half_to_float(static_cast<std::uint16_t>(p[0] | (p[1] << 8)));
  if (prec == "bf16") return from_bits(static_cast<std::uint32_t>(p[0] | (p[1] << 8)) << 16);
  std::uint32_t b;
  std::memcpy(&b, p, 4);
  return from_bits(b);
}

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: gemm_ref M N K A.bin B.bin C.bin [int8|fp16|bf16|fp32]\n";
    return 2;
  }
  const std::uint64_t M = std::stoull(argv[1]), N = std::stoull(argv[2]), K = std::stoull(argv[3]);
  const std::string prec = argc == 8 ? argv[7] : "int8";
  const unsigned e = prec == "int8" ? 1 : prec == "fp32" ? 4 : (prec == "fp16" || prec == "bf16") ? 2 : 0;
  if (!e) { std::cerr << "gemm_ref: unknown precision " << prec << "\n"; return 2; }
  const auto A = read_bytes(argv[4], M * K * e);
  const auto B = read_bytes(argv[5], K * N * e);
  std::vector<std::uint32_t> C(M * N);
  for (std::uint64_t i = 0; i < M; ++i)
    for (std::uint64_t j = 0; j < N; ++j) {
      if (prec == "int8") {
        std::int64_t s = 0;
        for (std::uint64_t k = 0; k < K; ++k)
          s += std::int64_t{static_cast<std::int8_t>(A[i * K + k])} * std::int64_t{static_cast<std::int8_t>(B[k * N + j])};
        if (s > std::numeric_limits<std::int32_t>::max() || s < std::numeric_limits<std::int32_t>::min()) {
          std::cerr << "gemm_ref: C[" << i << "][" << j << "] overflows INT32\n";
          return 2;
        }
        C[i * N + j] = static_cast<std::uint32_t>(static_cast<std::int32_t>(s));
      } else {
        float acc = 0.0f;
        for (std::uint64_t k = 0; k < K; ++k) {
          const float prod = decode(prec, &A[(i * K + k) * e]) * decode(prec, &B[(k * N + j) * e]);
          acc = acc + prod;
        }
        std::memcpy(&C[i * N + j], &acc, 4);
      }
    }
  std::ofstream out(argv[6], std::ios::binary);
  out.write(reinterpret_cast<const char*>(C.data()), static_cast<std::streamsize>(C.size() * 4));
  return out ? 0 : 2;
}
