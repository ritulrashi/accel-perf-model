// Runs one named directed scenario and prints its result as JSON.
//   apm_scenario --list
//   apm_scenario <name>
// Exit status: 0 complete and data correct, 1 incomplete or data mismatch,
// 2 usage error or model error.
#include <systemc>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "test/scenarios.h"

int sc_main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: apm_scenario --list | <scenario>\n";
    return 2;
  }
  const std::string arg = argv[1];
  if (arg == "--list") {
    for (const auto& n : apm::scenario_names()) std::cout << n << "\n";
    return 0;
  }
  try {
    const apm::ScenarioResult r = apm::run_scenario(arg);
    std::cout << r.json();
    return r.complete && (!r.data_checked || r.data_ok) ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "apm_scenario: " << e.what() << "\n";
    return 2;
  }
}
