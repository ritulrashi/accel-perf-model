#!/usr/bin/env bash
# Phase 5 exit criterion (docs/phase5-design.md §9):
#   1. both presets from scratch, whole suite: Phase 1-4 tests unchanged, RTL
#      lint (-Wall, no waivers), hand traces HT1/HT2, quick correlation, the
#      xor-map DRAM tests T5a-T5d and the unit cases;
#   2. full RTL correlation (release): 312 schedules, three levels each;
#   3. sweeps S1 (bank map; row-map rows must reproduce Phase 4) and S2
#      (equal-MAC arrangements), with plots.
# Outputs (results/phase5/): exit_criterion.txt, correlation.csv, sweep.csv,
# bank_map.png, arrangements.png. Staged outside the repo and moved in at the
# end (provenance first).
#
# Usage: scripts/phase5_exit.sh
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
  ctest --preset "$preset" -V > "$logs/$preset-ctest.log" 2>&1 || true
}

{
  echo "# Phase 5 exit criterion: RTL correlation of the compute array, bank map, sweeps"
  echo "command: scripts/phase5_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  for preset in asan release; do
    run "$preset"
    echo
    echo "## Preset: $preset"
    echo "### Expected vs measured (hand derivations: Phase 1 §6, Phase 2 §5, Phase 3 §5, Phase 4 §7, Phase 5 §6)"
    grep -hoE '(cycles|count): [A-Za-z0-9_]+ [A-Za-z0-9_.]+ expected=[0-9]+ measured=[0-9]+' \
      "$logs/$preset-ctest.log" | sort -u
    echo "### Mismatches"
    grep -hoE '(cycles|count): .* expected=[0-9]+ measured=[0-9]+' "$logs/$preset-ctest.log" \
      | awk '{split($4,e,"="); split($5,m,"="); if (e[2] != m[2]) {print; n++}} END {if (!n) print "none"}'
    echo "### RTL: lint, hand traces, quick correlation"
    grep -E 'Test +#[0-9]+: rtl\.' "$logs/$preset-ctest.log" | sed -E 's/^ *[0-9]+\/[0-9]+ //' || true
    grep -hoE 'os_array_tb: file=[^ ]*/tests/rtl/ht[12].sched .*' "$logs/$preset-ctest.log" | sed -E 's#file=[^ ]*/tests#file=tests#' || true
    grep -hoE '^[0-9]+: correlation: .*' "$logs/$preset-ctest.log" | sed -E 's/^[0-9]+: /ctest quick /' || true
    echo "### Summary"
    grep -E 'tests passed|^ *(unit|sim|determinism|functional|rtl) +=' "$logs/$preset-ctest.log" || true
    grep -A100 'The following tests FAILED' "$logs/$preset-ctest.log" || echo "failed tests: none"
  done

  echo
  echo "## RTL correlation, full program set (release build)"
  echo "command: python3 python/correlate.py --tools build/release --csv results/phase5/correlation.csv"
  python3 python/correlate.py --tools build/release --csv "$stage/correlation.csv" \
    > "$logs/correlation.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  grep -E '^correlation' "$logs/correlation.log"
  echo "non-matching schedules:"
  grep -vE 'MATCH$|^correlation' "$logs/correlation.log" || echo "  none"

  echo
  echo "## Sweeps S1 (bank map) and S2 (equal-MAC arrangements) (release build)"
  echo "command: python3 python/sweep_p5.py --tools build/release --out-dir results/phase5"
  python3 python/sweep_p5.py --tools build/release --out-dir "$stage" \
    > "$logs/sweep.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  cat "$logs/sweep.log"
} > "$stage/exit_criterion.txt"

mkdir -p results/phase5
mv "$stage/exit_criterion.txt" "$stage/correlation.csv" "$stage/sweep.csv" "$stage/bank_map.png" \
   "$stage/arrangements.png" results/phase5/
cat results/phase5/exit_criterion.txt
