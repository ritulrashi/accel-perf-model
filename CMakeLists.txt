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
# Phase 4 design note §2.2: floating-point results must not depend on the
# compiler fusing a*b + c into an FMA (GCC defaults to -ffp-contract=fast in
# GNU mode). Applies to the model and to both standalone references.
add_compile_options(-ffp-contract=off)

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

# Pure C++ (no SystemC): timing formulas, systolic contract, GEMM layout and
# PE buffer planning. Unit-tested in-process.
add_library(apm_core
  src/mem/dram_timing.cpp
  src/mem/sram_timing.cpp
  src/compute/systolic.cpp
  src/pe/lmem_alloc.cpp
  src/pe/gemm_layout.cpp
  src/pe/pe_config.cpp
  src/noc/mesh_timing.cpp
  src/soc/soc_plan.cpp
  src/soc/stall.cpp
  src/compute/numerics.cpp
  src/vec/softmax.cpp
  src/vec/vu_plan.cpp
  src/vec/vu_stall.cpp
  src/compute/schedule_io.cpp
  src/compute/contract_trace.cpp
)
target_include_directories(apm_core PUBLIC src)

# SystemC/TLM models and the directed-scenario harness.
add_library(apm_model
  src/mem/sampled_target.cpp
  src/mem/dram_channel.cpp
  src/mem/sram.cpp
  src/dma/dma_engine.cpp
  src/pe/pe.cpp
  src/test/traffic_gen.cpp
  src/test/scenarios.cpp
  src/test/gemm_run.cpp
  src/noc/mesh.cpp
  src/soc/accel.cpp
  src/soc/accel_run.cpp
  src/vec/vec_unit.cpp
  src/vec/vu_run.cpp
)
target_link_libraries(apm_model PUBLIC apm_core SystemC::systemc)

add_executable(apm_scenario tools/apm_scenario.cpp)
target_link_libraries(apm_scenario PRIVATE apm_model)
add_executable(apm_gemm tools/apm_gemm.cpp)
target_link_libraries(apm_gemm PRIVATE apm_model)
add_executable(apm_accel tools/apm_accel.cpp)
target_link_libraries(apm_accel PRIVATE apm_model)
add_executable(apm_op tools/apm_op.cpp)
target_link_libraries(apm_op PRIVATE apm_model)
# Standalone functional references: deliberately link nothing from the model.
add_executable(gemm_ref tools/gemm_ref.cpp)
add_executable(attn_ref tools/attn_ref.cpp)

enable_testing()
include(GoogleTest)

add_executable(unit_tests
  tests/unit/dram_timing_test.cpp
  tests/unit/sram_timing_test.cpp
  tests/unit/rr_arbiter_test.cpp
  tests/unit/systolic_test.cpp
  tests/unit/pe_config_test.cpp
  tests/unit/mesh_timing_test.cpp
  tests/unit/stall_test.cpp
  tests/unit/soc_plan_test.cpp
  tests/unit/numerics_test.cpp
  tests/unit/softmax_test.cpp
  tests/unit/contract_trace_test.cpp
)
target_link_libraries(unit_tests PRIVATE apm_core GTest::gtest_main)
target_compile_definitions(unit_tests PRIVATE APM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(unit_tests TEST_PREFIX "unit." PROPERTIES LABELS unit)

# SystemC cannot be re-elaborated in one process. gtest_discover_tests runs
# every test case in its own process, which is exactly what we need.
add_executable(sim_tests tests/sim/main.cpp tests/sim/scenarios_test.cpp tests/sim/phase2_test.cpp
  tests/sim/phase3_test.cpp tests/sim/phase4_test.cpp)
target_link_libraries(sim_tests PRIVATE apm_model GTest::gtest)
gtest_discover_tests(sim_tests TEST_PREFIX "sim."
  PROPERTIES LABELS sim ENVIRONMENT "SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1")

# Determinism (rule 5): the same scenario, run twice in separate processes,
# must print byte-identical JSON.
add_test(NAME determinism.t2
  COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:apm_scenario> -DARGS=t2
          -P ${CMAKE_SOURCE_DIR}/tests/determinism.cmake)
# Phase 3: a 2x3 grid with contention on every resource.
add_test(NAME determinism.accel
  COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:apm_accel>
          "-DARGS=M=16;N=96;K=256;R=4;C=16;Kt=64;px=2;py=3;sram_x=1;sram_y=1;seed=3"
          -P ${CMAKE_SOURCE_DIR}/tests/determinism.cmake)
set_tests_properties(determinism.t2 determinism.accel PROPERTIES LABELS determinism
  ENVIRONMENT "SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1")

# Functional match (Phase 2 rule 4): random INT8 GEMMs, model vs NumPy vs the
# standalone C++ reference. 20 cases here; the exit script runs 200.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME functional.gemm20
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/gemm_check.py
          --apm-gemm $<TARGET_FILE:apm_gemm> --gemm-ref $<TARGET_FILE:gemm_ref> --cases 20)
set_tests_properties(functional.gemm20 PROPERTIES LABELS functional)
add_test(NAME functional.accel20
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/gemm_check.py --target accel
          --apm-accel $<TARGET_FILE:apm_accel> --gemm-ref $<TARGET_FILE:gemm_ref> --cases 20)
set_tests_properties(functional.accel20 PROPERTIES LABELS functional)
# Phase 4: precision modes on the grid (5 cases per precision; the exit
# script runs 200) and the attention program, model vs NumPy replica vs
# attn_ref (one case of each mode x precision combination; the exit script runs 80).
add_test(NAME functional.accel_prec20
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/gemm_check.py --target accel --prec all
          --apm-accel $<TARGET_FILE:apm_accel> --gemm-ref $<TARGET_FILE:gemm_ref> --cases 20)
add_test(NAME functional.attn10
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/attn_check.py
          --tools $<TARGET_FILE_DIR:apm_op> --cases 10)
set_tests_properties(functional.accel_prec20 functional.attn10 PROPERTIES LABELS functional)

# Phase 6: the parallel runner keeps output in task order; reproduction
# compares regenerated CSVs byte for byte (docs/phase6-design.md §1-§2, §5).
add_test(NAME runner.selftest COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/runner.py --selftest)
add_test(NAME runner.reproduce_selftest
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/reproduce.py --selftest)
add_test(NAME runner.jobs_gemm
  COMMAND ${CMAKE_COMMAND} "-DARGS=${Python3_EXECUTABLE};${CMAKE_SOURCE_DIR}/python/gemm_check.py;--target;accel;--prec;all;--apm-accel;$<TARGET_FILE:apm_accel>;--gemm-ref;$<TARGET_FILE:gemm_ref>;--cases;24"
          -DOUT=${CMAKE_BINARY_DIR}/jobs_gemm -P ${CMAKE_SOURCE_DIR}/tests/jobs_determinism.cmake)
add_test(NAME runner.jobs_attn
  COMMAND ${CMAKE_COMMAND} "-DARGS=${Python3_EXECUTABLE};${CMAKE_SOURCE_DIR}/python/attn_check.py;--tools;$<TARGET_FILE_DIR:apm_op>;--cases;6"
          -DOUT=${CMAKE_BINARY_DIR}/jobs_attn -P ${CMAKE_SOURCE_DIR}/tests/jobs_determinism.cmake)
set_tests_properties(runner.selftest runner.reproduce_selftest runner.jobs_gemm runner.jobs_attn
                     PROPERTIES LABELS runner ENVIRONMENT "SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1")

# Phase 5 part A: SystemVerilog array, Verilator testbench, RTL tests.
option(APM_RTL "Build the SystemVerilog array and its Verilator testbench (Phase 5)" ON)
if(APM_RTL)
  include(cmake/rtl.cmake)
endif()
