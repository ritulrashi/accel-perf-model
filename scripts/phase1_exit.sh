#!/usr/bin/env bash
# Phase 1 exit criterion: build from scratch with the ASan/UBSan preset and
# the Release preset, run the whole test suite under each, and record every
# hand-derived expected-vs-measured cycle count with provenance in
# results/phase1/exit_criterion.txt.
#
# Everything is written to a temporary directory OUTSIDE the repo and moved
# into results/phase1 only at the end, so the provenance block (captured
# first) cannot see our own output and report the tree as DIRTY.
#
# Usage: scripts/phase1_exit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1

prov=$(scripts/env_report.sh)

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
logs=$stage/logs
mkdir -p "$logs"

run() {
  local preset=$1
  rm -rf "build/$preset"
  cmake --preset "$preset" > "$logs/$preset-configure.log" 2>&1
  cmake --build --preset "$preset" > "$logs/$preset-build.log" 2>&1
  # -V so the "cycles:" lines printed by passing tests reach the log.
  ctest --preset "$preset" -V > "$logs/$preset-ctest.log" 2>&1 || true
}

{
  echo "# Phase 1 exit criterion: directed memory-system tests (hand-derived cycles)"
  echo "command: scripts/phase1_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  for preset in asan release; do
    run "$preset"
    echo
    echo "## Preset: $preset"
    echo "### Expected vs measured (expected values from docs/phase1-design.md §6)"
    grep -hoE '(cycles|count): [A-Za-z0-9_]+ [A-Za-z0-9_.]+ expected=[0-9]+ measured=[0-9]+' \
      "$logs/$preset-ctest.log" | sort -u
    echo "### Mismatches"
    grep -hoE '(cycles|count): .* expected=[0-9]+ measured=[0-9]+' "$logs/$preset-ctest.log" \
      | awk '{split($4,e,"="); split($5,m,"="); if (e[2] != m[2]) {print; n++}} END {if (!n) print "none"}'
    echo "### Summary"
    grep -E 'tests passed|^ *(unit|sim|determinism) +=' "$logs/$preset-ctest.log" || true
    grep -E 'determinism: ' "$logs/$preset-ctest.log" | sed 's/^[0-9]*: //' || true
    grep -A100 'The following tests FAILED' "$logs/$preset-ctest.log" || echo "failed tests: none"
  done
} > "$stage/exit_criterion.txt"

mkdir -p results/phase1
mv "$stage/exit_criterion.txt" results/phase1/exit_criterion.txt
cat results/phase1/exit_criterion.txt
