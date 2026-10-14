// Standalone C++ functional reference (rule 4). Shares no code with the model:
// a plain triple loop over row-major inputs, accumulated in 64 bits and
// checked to fit INT32.
//
//   gemm_ref M N K A.bin B.bin C.bin
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

static std::vector<std::int8_t> read_i8(const char* path, std::uint64_t n) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (!f.good() && !f.eof()) { std::cerr << "gemm_ref: cannot read " << path << "\n"; std::exit(2); }
  if (raw.size() != n) { std::cerr << "gemm_ref: " << path << " has wrong size\n"; std::exit(2); }
  return std::vector<std::int8_t>(raw.begin(), raw.end());
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: gemm_ref M N K A.bin B.bin C.bin\n";
    return 2;
  }
  const std::uint64_t M = std::stoull(argv[1]), N = std::stoull(argv[2]), K = std::stoull(argv[3]);
  const auto A = read_i8(argv[4], M * K);
  const auto B = read_i8(argv[5], K * N);
  std::vector<std::int32_t> C(M * N);
  for (std::uint64_t i = 0; i < M; ++i)
    for (std::uint64_t j = 0; j < N; ++j) {
      std::int64_t s = 0;
      for (std::uint64_t k = 0; k < K; ++k) s += std::int64_t{A[i * K + k]} * std::int64_t{B[k * N + j]};
      if (s > std::numeric_limits<std::int32_t>::max() || s < std::numeric_limits<std::int32_t>::min()) {
        std::cerr << "gemm_ref: C[" << i << "][" << j << "] overflows INT32\n";
        return 2;
      }
      C[i * N + j] = static_cast<std::int32_t>(s);
    }
  std::ofstream out(argv[6], std::ios::binary);
  out.write(reinterpret_cast<const char*>(C.data()), static_cast<std::streamsize>(C.size() * 4));
  return out ? 0 : 2;
}
