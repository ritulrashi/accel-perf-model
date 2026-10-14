// GoogleTest entry point for the SystemC tests. SystemC's library provides
// main() and calls sc_main(). Each test runs one simulation, so ctest runs
// every test in its own process (gtest_discover_tests); run_scenario()
// refuses a second simulation in the same process.
#include <gtest/gtest.h>

#include <systemc>

int sc_main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
