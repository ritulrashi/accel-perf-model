#include "compute/schedule_io.h"

#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace apm {

std::string to_hex(const std::vector<std::uint8_t>& v) {
  static const char* d = "0123456789abcdef";
  std::string s;
  s.reserve(2 * v.size());
  for (std::uint8_t b : v) {
    s.push_back(d[b >> 4]);
    s.push_back(d[b & 15]);
  }
  return s;
}

std::vector<std::uint8_t> from_hex(const std::string& h) {
  if (h.size() % 2) throw std::invalid_argument("schedule: odd-length hex field");
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    throw std::invalid_argument(std::string("schedule: bad hex digit '") + c + "'");
  };
  std::vector<std::uint8_t> v(h.size() / 2);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(nib(h[2 * i]) * 16 + nib(h[2 * i + 1]));
  return v;
}

void write_schedule(std::ostream& out, const PeSchedule& s) {
  out << "pe " << s.pe << " R " << s.rows << " C " << s.cols << " Kt " << s.kt << " nc " << s.chunks << "\n";
  for (std::size_t i = 0; i < s.steps.size(); ++i) {
    const SchedStep& st = s.steps[i];
    out << "step " << i << " tile " << st.tile << " chunk " << st.chunk << " feed " << st.feed << " a "
        << to_hex(st.a) << " b " << to_hex(st.b) << "\n";
  }
  for (const SchedTile& t : s.tiles) out << "tile " << t.tile << " done " << t.done << " c " << to_hex(t.c) << "\n";
}

std::vector<PeSchedule> parse_schedules(std::istream& in) {
  std::vector<PeSchedule> out;
  std::string line;
  std::size_t lineno = 0;
  auto fail = [&](const std::string& why) {
    throw std::invalid_argument("schedule line " + std::to_string(lineno) + ": " + why);
  };
  auto expect = [&](std::istringstream& ss, const char* word) {
    std::string w;
    if (!(ss >> w) || w != word) fail(std::string("expected '") + word + "'");
  };
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "pe") {
      PeSchedule s;
      ss >> s.pe;
      expect(ss, "R");
      ss >> s.rows;
      expect(ss, "C");
      ss >> s.cols;
      expect(ss, "Kt");
      ss >> s.kt;
      expect(ss, "nc");
      ss >> s.chunks;
      if (!ss || s.rows == 0 || s.cols == 0 || s.kt == 0 || s.chunks == 0) fail("bad pe header");
      out.push_back(std::move(s));
      continue;
    }
    if (out.empty()) fail("record before any 'pe' header");
    PeSchedule& s = out.back();
    if (kind == "step") {
      SchedStep st;
      std::uint64_t idx;
      std::string a, b;
      ss >> idx;
      expect(ss, "tile");
      ss >> st.tile;
      expect(ss, "chunk");
      ss >> st.chunk;
      expect(ss, "feed");
      ss >> st.feed;
      expect(ss, "a");
      ss >> a;
      expect(ss, "b");
      ss >> b;
      if (!ss || idx != s.steps.size()) fail("bad step record");
      st.a = from_hex(a);
      st.b = from_hex(b);
      if (st.a.size() != s.rows * s.kt || st.b.size() != s.kt * s.cols) fail("operand block size mismatch");
      s.steps.push_back(std::move(st));
    } else if (kind == "tile") {
      SchedTile t;
      std::string c;
      ss >> t.tile;
      expect(ss, "done");
      ss >> t.done;
      expect(ss, "c");
      ss >> c;
      if (!ss) fail("bad tile record");
      t.c = from_hex(c);
      if (t.c.size() != 4ull * s.rows * s.cols) fail("C tile size mismatch");
      s.tiles.push_back(std::move(t));
    } else {
      fail("unknown record '" + kind + "'");
    }
  }
  return out;
}

}  // namespace apm
