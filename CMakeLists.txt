cmake_minimum_required(VERSION 3.24)
project(accel_perf_model LANGUAGES CXX)

# C++17 must match the SystemC build (see .devcontainer/setup.sh).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(APM_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
if(APM_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# The installed package reports 3.0.2.<release date>, so EXACT 3.0.2 would
# reject it; require the 3.0.2 prefix instead.
find_package(SystemCLanguage 3.0.2 CONFIG REQUIRED)
if(NOT SystemCLanguage_VERSION MATCHES "^3\\.0\\.2(\\.|$)")
  message(FATAL_ERROR "SystemC 3.0.2 required, found ${SystemCLanguage_VERSION}")
endif()
message(STATUS "SystemC ${SystemCLanguage_VERSION}")

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723  # v1.17.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Pure C++ timing formulas (design note §3.3-3.5); no SystemC dependency so
# they can be unit-tested in-process.
add_library(apm_timing
  src/mem/dram_timing.cpp
  src/mem/sram_timing.cpp
)
target_include_directories(apm_timing PUBLIC src)

# SystemC/TLM models and the directed-scenario harness.
add_library(apm_model
  src/mem/sampled_target.cpp
  src/mem/dram_channel.cpp
  src/mem/sram.cpp
  src/dma/dma_engine.cpp
  src/test/traffic_gen.cpp
  src/test/scenarios.cpp
)
target_link_libraries(apm_model PUBLIC apm_timing SystemC::systemc)

add_executable(apm_scenario tools/apm_scenario.cpp)
target_link_libraries(apm_scenario PRIVATE apm_model)

enable_testing()
include(GoogleTest)

add_executable(unit_tests
  tests/unit/dram_timing_test.cpp
  tests/unit/sram_timing_test.cpp
  tests/unit/rr_arbiter_test.cpp
)
target_link_libraries(unit_tests PRIVATE apm_timing GTest::gtest_main)
gtest_discover_tests(unit_tests TEST_PREFIX "unit." PROPERTIES LABELS unit)

# SystemC cannot be re-elaborated in one process. gtest_discover_tests runs
# every test case in its own process, which is exactly what we need.
add_executable(sim_tests tests/sim/main.cpp tests/sim/scenarios_test.cpp)
target_link_libraries(sim_tests PRIVATE apm_model GTest::gtest)
gtest_discover_tests(sim_tests TEST_PREFIX "sim."
  PROPERTIES LABELS sim ENVIRONMENT "SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1")

# Determinism (rule 5): the same scenario, run twice in separate processes,
# must print byte-identical JSON.
add_test(NAME determinism.t2
  COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:apm_scenario> -DSCENARIO=t2
          -P ${CMAKE_SOURCE_DIR}/tests/determinism.cmake)
set_tests_properties(determinism.t2 PROPERTIES LABELS determinism
  ENVIRONMENT "SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1")
