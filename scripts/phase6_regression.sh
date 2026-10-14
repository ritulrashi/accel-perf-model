#!/usr/bin/env bash
# The full regression (docs/phase6-design.md §3-§4), step by step with wall
# times. CI's push job and the Codespace measurement run exactly this.
#   default (push job): release build with RTL + full CTest; ASan build
#     without RTL + full CTest; RESULTS.md number check; reproduction of all
#     committed results CSVs.
#   --full (nightly job): ASan build WITH the RTL, then also the mutation
#     checks of Phases 1-5. They write results/phaseN/mutation_check.txt in
#     this checkout, so --full is meant for CI checkouts, which are never
#     committed from.
# Writes $LOG_DIR/regression_runtime.txt (LOG_DIR defaults to build/logs) and
# one log per step. Every step runs even if an earlier one fails; the exit
# status is non-zero if any step failed.
#
# Usage: scripts/phase6_regression.sh [--full]
set -uo pipefail
cd "$(dirname "$0")/.."
export SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1

full=0
[ "${1:-}" = "--full" ] && full=1
logs=${LOG_DIR:-build/logs}
mkdir -p "$logs"
table=$logs/regression_runtime.txt
status=0

step() {  # step NAME COMMAND...
  local name=$1
  shift
  local s rc
  s=$(date +%s)
  "$@" > "$logs/$name.log" 2>&1
  rc=$?
  printf '| %-30s | %7d | %s |\n' "$name" "$(( $(date +%s) - s ))" \
    "$([ $rc -eq 0 ] && echo pass || echo "FAIL (exit $rc)")" >> "$table"
  [ $rc -eq 0 ] || status=1
}

mutation() {  # passes only if the phase's report says every mutant was caught
  "scripts/phase$1_mutation_check.sh" &&
    grep -q 'all mutants caught by their required tests: yes' "results/phase$1/mutation_check.txt"
}

{
  echo "# Regression runtime ($([ $full -eq 1 ] && echo "full / nightly" || echo "push") steps)"
  echo "host: $(uname -srm), cpus: $(nproc)"
  echo
  echo "| step | seconds | result |"
  echo "|---|---|---|"
} > "$table"
start=$(date +%s)

step release_build bash -c 'rm -rf build/release && cmake --preset release && cmake --build --preset release'
step release_ctest ctest --preset release -j"$(nproc)" --output-on-failure
if [ $full -eq 1 ]; then
  step asan_build_with_rtl bash -c 'rm -rf build/asan && cmake --preset asan && cmake --build --preset asan'
else
  step asan_build_without_rtl bash -c 'rm -rf build/asan && cmake --preset asan -DAPM_RTL=OFF && cmake --build --preset asan'
fi
step asan_ctest ctest --preset asan -j"$(nproc)" --output-on-failure
step check_results scripts/check_results.sh
step reproduce_results_csvs python3 python/reproduce.py --tools build/release
if [ $full -eq 1 ]; then
  for p in 1 2 3 4 5; do step "mutation_check_phase$p" mutation "$p"; done
fi

printf '| %-30s | %7d | %s |\n' "TOTAL" "$(( $(date +%s) - start ))" \
  "$([ $status -eq 0 ] && echo pass || echo FAIL)" >> "$table"
cat "$table"
exit $status
