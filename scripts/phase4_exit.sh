#!/usr/bin/env bash
# Phase 4 exit criterion (docs/phase4-design.md §8):
#   1. both presets from scratch, whole suite (Phase 1-4 hand-derived tests,
#      unit cases, determinism, quick functional checks);
#   2. 200 random grid GEMMs, 50 per precision, model vs NumPy vs gemm_ref;
#   3. 80 random attention programs, model vs NumPy replica vs attn_ref;
#   4. the 54-program study on the A4 configuration: every op passes its
#      in-model assertions, DRAM bytes == formula, T_op >= formula lower
#      bounds, achieved <= attainable, every program bit-exact three ways,
#      FP64 errors; plus the breakdown, numerics and roofline plots.
# Outputs (results/phase4/): exit_criterion.txt, functional_gemm.csv,
# functional_attn.csv, study.csv, attention_breakdown.png, numerics.png,
# attention_roofline.png. Staged outside the repo and moved in at the end
# (provenance first).
#
# Usage: scripts/phase4_exit.sh
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
  echo "# Phase 4 exit criterion: precision modes, vector unit, attention workloads"
  echo "command: scripts/phase4_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  for preset in asan release; do
    run "$preset"
    echo
    echo "## Preset: $preset"
    echo "### Expected vs measured (hand derivations: Phase 1 §6, Phase 2 §5, Phase 3 §5, Phase 4 §7)"
    grep -hoE '(cycles|count): [A-Za-z0-9_]+ [A-Za-z0-9_.]+ expected=[0-9]+ measured=[0-9]+' \
      "$logs/$preset-ctest.log" | sort -u
    echo "### Mismatches"
    grep -hoE '(cycles|count): .* expected=[0-9]+ measured=[0-9]+' "$logs/$preset-ctest.log" \
      | awk '{split($4,e,"="); split($5,m,"="); if (e[2] != m[2]) {print; n++}} END {if (!n) print "none"}'
    echo "### Summary"
    grep -E 'tests passed|^ *(unit|sim|determinism|functional) +=' "$logs/$preset-ctest.log" || true
    grep -hoE 'determinism: .*' "$logs/$preset-ctest.log" | sort -u || true
    grep -hoE 'functional: target=.*' "$logs/$preset-ctest.log" | sed 's/^/ctest quick /' || true
    grep -hoE 'attention functional: .*' "$logs/$preset-ctest.log" | sed 's/^/ctest quick /' || true
    grep -A100 'The following tests FAILED' "$logs/$preset-ctest.log" || echo "failed tests: none"
  done

  echo
  echo "## GEMM functional match by precision, 200 cases (release build)"
  echo "command: python3 python/gemm_check.py --target accel --prec all --apm-accel build/release/apm_accel --gemm-ref build/release/gemm_ref --cases 200 --csv results/phase4/functional_gemm.csv"
  python3 python/gemm_check.py --target accel --prec all --apm-accel build/release/apm_accel \
    --gemm-ref build/release/gemm_ref --cases 200 --csv "$stage/functional_gemm.csv" \
    > "$logs/functional_gemm.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  grep -E '^functional:' "$logs/functional_gemm.log"
  echo "mismatching cases:"
  grep -E 'MISMATCH' "$logs/functional_gemm.log" || echo "  none"

  echo
  echo "## Attention functional match, 80 programs (release build)"
  echo "command: python3 python/attn_check.py --tools build/release --cases 80 --csv results/phase4/functional_attn.csv"
  python3 python/attn_check.py --tools build/release --cases 80 --csv "$stage/functional_attn.csv" \
    > "$logs/functional_attn.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  grep -E '^(attention functional|per combination):' "$logs/functional_attn.log"
  echo "mismatching cases:"
  grep -E 'MISMATCH' "$logs/functional_attn.log" || echo "  none"

  echo
  echo "## Study: 54 attention programs on the A4 configuration (release build)"
  echo "command: python3 python/study_p4.py --tools build/release --out-dir results/phase4"
  python3 python/study_p4.py --tools build/release --out-dir "$stage" \
    > "$logs/study.log" 2>&1 && echo "exit status: 0" || echo "exit status: $?"
  cat "$logs/study.log"
} > "$stage/exit_criterion.txt"

mkdir -p results/phase4
mv "$stage/exit_criterion.txt" "$stage/functional_gemm.csv" "$stage/functional_attn.csv" "$stage/study.csv" \
   "$stage/attention_breakdown.png" "$stage/numerics.png" "$stage/attention_roofline.png" results/phase4/
cat results/phase4/exit_criterion.txt
