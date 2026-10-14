// Standalone C++ attention reference (Phase 4 design note §2, §4). Shares no
// code with the model: every operation of the numerics spec is re-implemented
// here from the design note, so that the model, this program and the NumPy
// replica (python/numerics.py) can be compared bit for bit.
//
//   attn_ref prec G N d causal q_off alpha_bits V p_scale Q.bin K.bin Vv.bin OUTDIR
//
// Q is G x d, K and Vv are N x d, all row-major e-byte elements (INT8 values
// already quantized). Writes OUTDIR/S.bin (G x N 4-byte words), P.bin (G x N
// e-byte, row-major), scales.bin (G FP32), O.bin (G x d 4-byte words: INT32
// or FP32 bits). Built with -ffp-contract=off: no fused multiply-add anywhere.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_bytes(const std::string& path, std::uint64_t n) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (raw.size() != n) {
    std::cerr << "attn_ref: " << path << " has " << raw.size() << " bytes, expected " << n << "\n";
    std::exit(2);
  }
  return std::vector<std::uint8_t>(raw.begin(), raw.end());
}

void write_bytes(const std::string& path, const void* p, std::uint64_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
  if (!f) { std::cerr << "attn_ref: cannot write " << path << "\n"; std::exit(2); }
}

std::uint32_t to_u(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }
float to_f(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

float half_to_float(std::uint16_t h) {
  const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  const float v = e == 0 ? static_cast<float>(m) * std::ldexp(1.0f, -24)
                         : std::ldexp(static_cast<float>(m + 1024), e - 25);
  return s ? -v : v;
}

// float -> half, round to nearest even, through exact double arithmetic:
// pick the two neighbouring halves and compare distances.
std::uint16_t float_to_half(float f) {
  const std::uint16_t sign = std::signbit(f) ? 0x8000 : 0;
  const double a = std::fabs(static_cast<double>(f));
  if (a >= 65520.0) return sign | 0x7c00;  // rounds to infinity
  // Quantum of the binade containing a (subnormal quantum below 2^-14).
  int exp2;
  std::frexp(a, &exp2);                       // a = m * 2^exp2, m in [0.5, 1)
  const int q = exp2 - 1 < -14 ? -24 : exp2 - 1 - 10;
  const double units = std::ldexp(a, -q);     // exact in double
  double lo = std::floor(units);
  const double frac = units - lo;
  if (frac > 0.5 || (frac == 0.5 && std::fmod(lo, 2.0) != 0.0)) lo += 1.0;
  const double v = std::ldexp(lo, q);         // the rounded magnitude
  if (v == 0.0) return sign;
  if (v >= 65520.0) return sign | 0x7c00;
  int e2;
  std::frexp(v, &e2);
  if (e2 - 1 < -14) return sign | static_cast<std::uint16_t>(std::ldexp(v, 24));  // subnormal
  const int be = e2 - 1 + 15;
  const auto man = static_cast<std::uint16_t>(std::ldexp(v, -(e2 - 1) + 10) - 1024.0);
  return sign | static_cast<std::uint16_t>(be << 10) | man;
}

std::uint16_t float_to_bf16(float f) {
  // Round to nearest even on the upper 16 bits, by distance comparison.
  const std::uint32_t b = to_u(f);
  const std::uint32_t lo = b & 0xffff0000u, hi = lo + 0x10000u;
  const double dl = std::fabs(static_cast<double>(f) - static_cast<double>(to_f(lo)));
  const double dh = std::fabs(static_cast<double>(to_f(hi)) - static_cast<double>(f));
  std::uint32_t pick = dl < dh ? lo : dh < dl ? hi : (((lo >> 16) & 1) ? hi : lo);
  return static_cast<std::uint16_t>(pick >> 16);
}

float decode(const std::string& prec, const std::uint8_t* p) {
  if (prec == "fp16") return half_to_float(static_cast<std::uint16_t>(p[0] | (p[1] << 8)));
  if (prec == "bf16") return to_f(static_cast<std::uint32_t>(p[0] | (p[1] << 8)) << 16);
  std::uint32_t b;
  std::memcpy(&b, p, 4);
  return to_f(b);
}

// §2.3: t = x*log2e; flush below -125; n = rint(t); f = t - n; Horner in FP32; ldexp.
float exp_spec(float x) {
  static const std::uint32_t c[7] = {0x3f800000, 0x3f317218, 0x3e75fdf0, 0x3d635847,
                                     0x3c1d955b, 0x3aaec3ff, 0x39218489};
  const float t = x * to_f(0x3fb8aa3b);
  if (t < -125.0f) return 0.0f;
  const float n = std::rint(t);
  const float fr = t - n;
  float p = to_f(c[6]);
  for (int k = 5; k >= 0; --k) {
    const float m = p * fr;
    p = m + to_f(c[k]);
  }
  return std::ldexp(p, static_cast<int>(n));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 14) {
    std::cerr << "usage: attn_ref prec G N d causal q_off alpha_bits V p_scale Q.bin K.bin Vv.bin OUTDIR\n";
    return 2;
  }
  const std::string prec = argv[1];
  const std::uint64_t G = std::stoull(argv[2]), N = std::stoull(argv[3]), d = std::stoull(argv[4]);
  const bool causal = std::stoull(argv[5]) != 0;
  const std::uint64_t q_off = std::stoull(argv[6]);
  const float alpha = to_f(static_cast<std::uint32_t>(std::stoull(argv[7], nullptr, 0)));
  const unsigned lanes = static_cast<unsigned>(std::stoul(argv[8]));
  const bool row_scale = std::string(argv[9]) == "row";
  const std::string outdir = argv[13];
  const bool ints = prec == "int8";
  const unsigned e = ints ? 1 : prec == "fp32" ? 4 : 2;
  const auto Q = read_bytes(argv[10], G * d * e);
  const auto K = read_bytes(argv[11], N * d * e);
  const auto Vv = read_bytes(argv[12], N * d * e);

  // S = Q K^T, ascending k.
  std::vector<std::uint32_t> S(G * N);
  for (std::uint64_t i = 0; i < G; ++i)
    for (std::uint64_t j = 0; j < N; ++j) {
      if (ints) {
        std::int64_t s = 0;
        for (std::uint64_t k = 0; k < d; ++k)
          s += std::int64_t{static_cast<std::int8_t>(Q[i * d + k])} * static_cast<std::int8_t>(K[j * d + k]);
        S[i * N + j] = static_cast<std::uint32_t>(static_cast<std::int32_t>(s));
      } else {
        float acc = 0.0f;
        for (std::uint64_t k = 0; k < d; ++k) {
          const float prod = decode(prec, &Q[(i * d + k) * e]) * decode(prec, &K[(j * d + k) * e]);
          acc = acc + prod;
        }
        S[i * N + j] = to_u(acc);
      }
    }

  // Softmax per row (§2.4).
  std::vector<std::uint8_t> P(G * N * e, 0);
  std::vector<float> scales(G);
  for (std::uint64_t i = 0; i < G; ++i) {
    auto ok = [&](std::uint64_t j) { return !causal || j <= i + q_off; };
    auto sval = [&](std::uint64_t j) {
      return ints ? static_cast<float>(static_cast<std::int32_t>(S[i * N + j])) : to_f(S[i * N + j]);
    };
    float m = -INFINITY;
    for (std::uint64_t j = 0; j < N; ++j)
      if (ok(j) && sval(j) > m) m = sval(j);
    std::vector<float> ex(N, 0.0f);
    for (std::uint64_t j = 0; j < N; ++j)
      if (ok(j)) {
        const float diff = sval(j) - m;
        ex[j] = exp_spec(diff * alpha);
      }
    std::vector<float> lane(lanes, 0.0f);
    for (std::uint64_t j = 0; j < N; ++j) lane[j % lanes] = lane[j % lanes] + ex[j];
    for (unsigned w = lanes / 2; w >= 1; w /= 2)
      for (unsigned l = 0; l < w; ++l) lane[l] = lane[l] + lane[l + w];
    const float r = 1.0f / lane[0];
    for (std::uint64_t j = 0; j < N; ++j) {
      const float p = ex[j] * r;
      std::uint8_t* dst = &P[(i * N + j) * e];
      if (prec == "fp16" || prec == "bf16") {
        const std::uint16_t h = prec == "fp16" ? float_to_half(p) : float_to_bf16(p);
        dst[0] = static_cast<std::uint8_t>(h & 0xff);
        dst[1] = static_cast<std::uint8_t>(h >> 8);
      } else if (prec == "fp32") {
        const std::uint32_t b = to_u(p);
        std::memcpy(dst, &b, 4);
      } else {
        const float v = (row_scale ? ex[j] : p) * 127.0f;
        dst[0] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::rint(v)));
      }
    }
    scales[i] = ints ? (row_scale ? r / 127.0f : 1.0f / 127.0f) : 1.0f;
  }

  // O = P V, ascending j.
  std::vector<std::uint32_t> O(G * d);
  for (std::uint64_t i = 0; i < G; ++i)
    for (std::uint64_t c = 0; c < d; ++c) {
      if (ints) {
        std::int64_t s = 0;
        for (std::uint64_t j = 0; j < N; ++j)
          s += std::int64_t{static_cast<std::int8_t>(P[i * N + j])} * static_cast<std::int8_t>(Vv[j * d + c]);
        O[i * d + c] = static_cast<std::uint32_t>(static_cast<std::int32_t>(s));
      } else {
        float acc = 0.0f;
        for (std::uint64_t j = 0; j < N; ++j) {
          const float prod = decode(prec, &P[(i * N + j) * e]) * decode(prec, &Vv[(j * d + c) * e]);
          acc = acc + prod;
        }
        O[i * d + c] = to_u(acc);
      }
    }

  write_bytes(outdir + "/S.bin", S.data(), S.size() * 4);
  write_bytes(outdir + "/P.bin", P.data(), P.size());
  write_bytes(outdir + "/scales.bin", scales.data(), scales.size() * 4);
  write_bytes(outdir + "/O.bin", O.data(), O.size() * 4);
  return 0;
}
