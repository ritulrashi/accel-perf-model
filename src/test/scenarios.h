// Directed Phase 1 scenarios (design note §6). Each one builds a small memory
// system, runs it to completion, and returns every cycle count and counter.
//
// SystemC cannot be re-elaborated inside one process, so run_scenario() may
// be called at most once per process (it throws on a second call).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace apm {

struct ScenarioResult {
  std::string name;
  bool complete = false;      // every DMA descriptor and scripted request finished
  bool data_checked = false;  // the scenario moves data that is checked
  bool data_ok = false;       // whole destination memories match the reference
  std::map<std::string, std::uint64_t> values;

  std::string json() const;  // deterministic: keys in sorted order
};

std::vector<std::string> scenario_names();
ScenarioResult run_scenario(const std::string& name);

}  // namespace apm
