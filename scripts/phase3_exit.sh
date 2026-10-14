#!/usr/bin/env bash
# Phase 3 exit criterion (docs/phase3-design.md §6):
#   1. both presets from scratch, whole suite (Phase 1-3 hand-derived tests,
#      stall stacks of P3-T1/T2, determinism, 20-case functional checks);
#   2. 100 random grid GEMMs, model vs NumPy vs the C++ reference;
#   3. the >= 100-configuration sweep: every run must pass the in-model
#      stall-stack assertions; DRAM bytes == formula; achieved <= attainable;
#      busiest link >= formula; plus roofline and stall-stack plots.
# Outputs (results/phase3/): exit_criterion.txt, functional_accel.csv,
# sweep.csv, roofline.png, stall_stacks.png. Staged outside the repo and
# moved in at the end (provenance first).
#
# Usage: scripts/phase3_exit.sh
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
  echo "# Phase 3 exit criterion: accelerator grid, stall stack, roofline"
  echo "command: scripts/phase3_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  for preset in asan release; do
    run "$preset"
    echo
    echo "## Preset: $preset"
    echo "### Expected vs measured (hand derivations: Phase 1 §6, Phase 2 §5, Phase 3 §5)"
    grep -hoE '(cycles|count): [A-Za-z0-9_]+ [A-Za-z0-9_.]+ expected=[0-9]+ measured=[0-9]+' \
      "$logs/$preset-ctest.log" | sort -u
    echo "### Mismatches"
    grep -hoE '(cycles|count): .* expected=[0-9]+ measured=[0-9]+' "$logs/$preset-ctest.log" \
      | awk '{split($4,e,"="); split($5,m,"="); if (e[2] != m[2]) {print; n++}} END {if (!n) print "none"}'
    echo "### Summary"
    grep -E 'tests passed|^ *(unit|sim|determinism|functional) +=' "$logs/$preset-ctest.log" || true
    grep -hoE 'determinism: .*' "$logs/$preset-ctest.log" | sort -u || true
    grep -hoE 'functional: target=.*' "$logs/$preset-ctest.log" | sed 's/^/ctest 20-case /' || true
    grep -A100 'The following tests FAILED' "$logs/$preset-ctest.log" || echo "failed tests: none"
  done

  echo
  echo "## Functional match on the grid, 100 cases (release build)"
  echo "command: python3 python/gemm_check.py --target accel --apm-accel build/release/apm_accel --gemm-ref build/release/gemm_ref --cases 100 --csv results/phase3/functional_accel.csv"
  python3 python/gemm_check.py --target accel --apm-accel build/release/apm_accel \
    --gemm-ref build/release/gemm_ref --cases 100 --csv "$stage/functional_accel.csv" \
    > "$logs/functional.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  grep -E '^functional:' "$logs/functional.log"
  echo "mismatching cases:"
  grep -E 'MISMATCH' "$logs/functional.log" || echo "  none"

  echo
  echo "## Sweep (release build)"
  echo "command: python3 python/sweep_p3.py --apm-accel build/release/apm_accel --out-dir results/phase3"
  s=$(date +%s)
  python3 python/sweep_p3.py --apm-accel build/release/apm_accel --out-dir "$stage" \
    > "$logs/sweep.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  echo "sweep wall time: $(( $(date +%s) - s )) s"
  cat "$logs/sweep.log"
} > "$stage/exit_criterion.txt"

mkdir -p results/phase3
mv "$stage/exit_criterion.txt" "$stage/functional_accel.csv" "$stage/sweep.csv" \
   "$stage/roofline.png" "$stage/stall_stacks.png" results/phase3/
cat results/phase3/exit_criterion.txt
