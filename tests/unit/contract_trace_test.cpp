// contract_trace and the schedule format (Phase 5 design note §3, §8): the
// model's contract expansion must reproduce the hand-written traces HT1 and
// HT2 exactly.
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "compute/contract_trace.h"
#include "compute/schedule_io.h"

namespace {

std::vector<std::string> lines_of(const std::string& path) {
  std::ifstream f(path);
  EXPECT_TRUE(f.good()) << path;
  std::vector<std::string> v;
  std::string l;
  while (std::getline(f, l))
    if (!l.empty() && l[0] != '#') v.push_back(l);
  return v;
}

apm::PeSchedule load(const std::string& path) {
  std::ifstream f(path);
  const auto s = apm::parse_schedules(f);
  EXPECT_EQ(s.size(), 1u);
  return s.at(0);
}

const std::string kDir = APM_SOURCE_DIR "/tests/rtl/";

}  // namespace

TEST(ContractTrace, HT1MatchesHandTrace) {
  apm::ContractTiming t;
  EXPECT_EQ(apm::contract_trace(load(kDir + "ht1.sched"), &t), lines_of(kDir + "ht1.trace"));
  ASSERT_EQ(t.done.size(), 1u);
  EXPECT_EQ(t.drain_start[0], 5u);
  EXPECT_EQ(t.done[0], 7u);
}

TEST(ContractTrace, HT2MatchesHandTrace) {
  apm::ContractTiming t;
  EXPECT_EQ(apm::contract_trace(load(kDir + "ht2.sched"), &t), lines_of(kDir + "ht2.trace"));
  EXPECT_EQ(t.drain_start.at(0), 7u);
  EXPECT_EQ(t.done.at(0), 9u);
}

TEST(ContractTrace, OverlappingFeedsAreRejected) {
  apm::PeSchedule s = load(kDir + "ht2.sched");
  s.steps[1].feed = 1;  // chunk 1 starts before chunk 0 has finished feeding
  EXPECT_THROW(apm::contract_trace(s), std::invalid_argument);
}

TEST(ScheduleIo, RoundTrip) {
  const apm::PeSchedule s = load(kDir + "ht2.sched");
  std::stringstream ss;
  apm::write_schedule(ss, s);
  const auto back = apm::parse_schedules(ss);
  ASSERT_EQ(back.size(), 1u);
  EXPECT_EQ(back[0].steps.size(), 2u);
  EXPECT_EQ(back[0].steps[1].feed, 3u);
  EXPECT_EQ(back[0].steps[1].b, s.steps[1].b);
  EXPECT_EQ(back[0].tiles[0].c, s.tiles[0].c);
  std::stringstream bad("pe 0 R 2 C 2 Kt 2 nc 1\nstep 0 tile 0 chunk 0 feed 0 a 0102 b 01010101\n");
  EXPECT_THROW(apm::parse_schedules(bad), std::invalid_argument);  // A block too short
}
