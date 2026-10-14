# Runs one model invocation twice in separate processes and requires
# byte-identical output (rule 5). EXE is the program, ARGS its argument list.
foreach(i 1 2)
  execute_process(COMMAND ${EXE} ${ARGS} OUTPUT_VARIABLE out${i} RESULT_VARIABLE rc${i})
  if(NOT rc${i} EQUAL 0)
    message(FATAL_ERROR "run ${i} of ${EXE} ${ARGS} failed (exit ${rc${i}})")
  endif()
endforeach()
if(NOT out1 STREQUAL out2)
  message(FATAL_ERROR "non-deterministic output for ${ARGS}:
--- run 1
${out1}
--- run 2
${out2}")
endif()
message(STATUS "determinism: ${ARGS} identical over 2 runs")
