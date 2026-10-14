#!/usr/bin/env bash
# Phase 6 exit criterion, Codespace part (docs/phase6-design.md §6):
#   1. the push-job regression with per-step wall times
#      (scripts/phase6_regression.sh): both builds, full CTest, RESULTS.md
#      numbers, reproduction of all 8 committed results CSVs;
#   2. the runner speedup: study_p4.py and sweep_p5.py with --jobs 1 and
#      --jobs 2, wall times, and byte-identity of their CSVs.
# The GitHub Actions part is recorded by scripts/phase6_ci_times.sh.
# Outputs (results/phase6/): exit_criterion.txt, regression_runtime.txt,
# runner_speedup.txt. Staged outside the repo and moved in at the end.
#
# Usage: scripts/phase6_exit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1

prov=$(scripts/env_report.sh)

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/logs"

LOG_DIR="$stage/logs" scripts/phase6_regression.sh > /dev/null && reg=0 || reg=$?

{
  echo "# Phase 6 regression runtime, Codespace (the CI push-job steps)"
  echo "command: scripts/phase6_regression.sh (via scripts/phase6_exit.sh)"
  echo
  echo "## Provenance"
  echo "$prov"
  echo
  cat "$stage/logs/regression_runtime.txt"
  echo
  echo "regression exit status: $reg"
} > "$stage/regression_runtime.txt"

{
  echo "# Phase 6 runner speedup: --jobs 1 vs --jobs 2"
  echo "command: scripts/phase6_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  echo
  echo "| script | jobs | wall s | CSV identical to --jobs 1 |"
  echo "|---|---|---|---|"
  for script in study_p4 sweep_p5; do
    csv=$([ $script = study_p4 ] && echo study.csv || echo sweep.csv)
    for j in 1 2; do
      d=$stage/speed_${script}_$j
      mkdir -p "$d"
      s=$(date +%s.%N)
      python3 "python/$script.py" --tools build/release --out-dir "$d" --jobs $j > "$d/log.txt" 2>&1 || true
      e=$(python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$s" "$(date +%s.%N)")
      same=$(cmp -s "$stage/speed_${script}_1/$csv" "$d/$csv" && echo yes || echo NO)
      echo "| $script.py | $j | $e | $same |"
    done
  done
} > "$stage/runner_speedup.txt"

{
  echo "# Phase 6 exit criterion: parallel runner, result reproduction, regression runtime"
  echo "command: scripts/phase6_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  echo
  echo "## Regression steps (push job, run in this Codespace)"
  cat "$stage/logs/regression_runtime.txt"
  echo "regression exit status: $reg"
  for preset in release asan; do
    echo
    echo "## CTest summary: $preset"
    grep -E 'tests passed|^ *(unit|sim|determinism|functional|rtl|runner) +=' "$stage/logs/${preset}_ctest.log" || true
    grep -A100 'The following tests FAILED' "$stage/logs/${preset}_ctest.log" || echo "failed tests: none"
  done
  echo
  echo "## RESULTS.md number check"
  cat "$stage/logs/check_results.log"
  echo
  echo "## Reproduction of the committed results CSVs"
  grep -E '^reproduce|^  ' "$stage/logs/reproduce_results_csvs.log"
  echo
  echo "## Runner speedup"
  sed -n '/^| script/,$p' "$stage/runner_speedup.txt"
} > "$stage/exit_criterion.txt"

mkdir -p results/phase6
mv "$stage/exit_criterion.txt" "$stage/regression_runtime.txt" "$stage/runner_speedup.txt" results/phase6/
cat results/phase6/exit_criterion.txt
