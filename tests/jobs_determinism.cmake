# Phase 6 (docs/phase6-design.md §5): runs one sweep/check script with
# --jobs 1 and --jobs 3 and requires byte-identical CSVs. ARGS is the
# script's argument list without --jobs and --csv; OUT is a scratch prefix.
foreach(j 1 3)
  execute_process(COMMAND ${ARGS} --jobs ${j} --csv ${OUT}.jobs${j}.csv
                  OUTPUT_QUIET RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "run with --jobs ${j} failed (exit ${rc}): ${ARGS}")
  endif()
endforeach()
file(READ ${OUT}.jobs1.csv a)
file(READ ${OUT}.jobs3.csv b)
if(NOT a STREQUAL b)
  message(FATAL_ERROR "--jobs 1 and --jobs 3 CSVs differ for: ${ARGS}")
endif()
string(LENGTH "${a}" n)
message(STATUS "jobs determinism: identical CSVs (${n} bytes) with --jobs 1 and --jobs 3")
