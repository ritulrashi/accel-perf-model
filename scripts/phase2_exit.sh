#!/usr/bin/env bash
# Phase 2 exit criterion:
#   1. build both presets from scratch and run the whole suite (Phase 1
#      regression + Phase 2 hand-derived timing tests + 20-case functional);
#   2. the full functional match: 200 random INT8 GEMMs, model vs NumPy vs the
#      standalone C++ reference (release build), per-case CSV.
# Writes results/phase2/exit_criterion.txt and results/phase2/functional.csv.
# Everything is staged outside the repo and moved in at the end, so the
# provenance block (captured first) cannot see our own output.
#
# Usage: scripts/phase2_exit.sh
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
  echo "# Phase 2 exit criterion: compute unit, double buffering, functional match"
  echo "command: scripts/phase2_exit.sh"
  echo
  echo "## Provenance"
  echo "$prov"
  for preset in asan release; do
    run "$preset"
    echo
    echo "## Preset: $preset"
    echo "### Expected vs measured (Phase 1 §6 and Phase 2 §5 hand derivations)"
    grep -hoE '(cycles|count): [A-Za-z0-9_]+ [A-Za-z0-9_.]+ expected=[0-9]+ measured=[0-9]+' \
      "$logs/$preset-ctest.log" | sort -u
    echo "### Mismatches"
    grep -hoE '(cycles|count): .* expected=[0-9]+ measured=[0-9]+' "$logs/$preset-ctest.log" \
      | awk '{split($4,e,"="); split($5,m,"="); if (e[2] != m[2]) {print; n++}} END {if (!n) print "none"}'
    echo "### Summary"
    grep -E 'tests passed|^ *(unit|sim|determinism|functional) +=' "$logs/$preset-ctest.log" || true
    grep -hoE 'functional: .*seed=.*' "$logs/$preset-ctest.log" | sed 's/^/ctest 20-case /' || true
    grep -A100 'The following tests FAILED' "$logs/$preset-ctest.log" || echo "failed tests: none"
  done

  echo
  echo "## Functional match, 200 cases (release build)"
  echo "command: python3 python/gemm_check.py --apm-gemm build/release/apm_gemm --gemm-ref build/release/gemm_ref --cases 200 --csv results/phase2/functional.csv"
  python3 python/gemm_check.py --apm-gemm build/release/apm_gemm --gemm-ref build/release/gemm_ref \
    --cases 200 --csv "$stage/functional.csv" > "$logs/functional.log" 2>&1 \
    && echo "exit status: 0" || echo "exit status: $?"
  grep -E '^functional:' "$logs/functional.log"
  echo "mismatching cases:"
  grep -E 'MISMATCH' "$logs/functional.log" || echo "  none"
  echo "case shapes covered (array RxC, Kt, nbuf, nout):"
  python3 - "$stage/functional.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for key in ("R", "C", "Kt", "nbuf", "nout"):
    vals = sorted({int(r[key]) for r in rows})
    print(f"  {key}: " + ", ".join(f"{v} ({sum(int(r[key]) == v for r in rows)} cases)" for v in vals))
print(f"  M up to {max(int(r['M']) for r in rows)}, N up to {max(int(r['N']) for r in rows)}, "
      f"K up to {max(int(r['K']) for r in rows)}")
EOF
} > "$stage/exit_criterion.txt"

mkdir -p results/phase2
mv "$stage/exit_criterion.txt" "$stage/functional.csv" results/phase2/
cat results/phase2/exit_criterion.txt
