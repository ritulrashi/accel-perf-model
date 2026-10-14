#!/usr/bin/env bash
# Can the Phase 6 checks fail? (rule 6). Copies the tracked tree to a temp
# dir, checks the unmutated copy passes, then injects one mutant at a time
# (docs/phase6-design.md §7), rebuilds, and records which tests fail. Each
# mutant must be caught by the tests the design note names for it. Nothing in
# the real tree is modified.
#
# Usage: scripts/phase6_mutation_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export SYSTEMC_DISABLE_COPYRIGHT_MESSAGE=1

prov=$(scripts/env_report.sh)

work=$(mktemp -d)
stage=$(mktemp -d)
trap 'rm -rf "$work" "$stage"' EXIT
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$work"
(cd "$work" && cmake --preset release >/dev/null && cmake --build --preset release >/dev/null)

failed_tests() {  # prints the names of failing ctest tests, one per line
  (cd "$work" && ctest --preset release -j"$(nproc)" > "$stage/ctest.log" 2>&1) || true
  sed -n '/The following tests FAILED/,$p' "$stage/ctest.log" \
    | grep -oE '[0-9]+ - [A-Za-z0-9_.]+' | sed -E 's/^[0-9]+ - //' | sort || true
  # With no failures grep exits 1; without "|| true", pipefail + set -e
  # would abort the whole script (docs/BUGS.md #1).
}

# name | file | sed expression | required failing tests (space separated)
mutants=(
  "P1|python/runner.py|s#^        for t, f in zip(tasks, futs):#        for t, f in ((fut_task[f], f) for f in as_completed(futs)):  \# MUTANT P1 completion order#|runner.selftest"
  "P2|python/runner.py|s#^                out.append((t, None, exc))#                out.append((t, None, None))  \# MUTANT P2 exception dropped#|runner.selftest"
  "P3|python/reproduce.py|s#^    if committed == regenerated:#    if len(committed.splitlines()) == len(regenerated.splitlines()):  \# MUTANT P3 line counts only#|runner.reproduce_selftest"
  "P4|python/gemm_check.py|s#^    with tempfile.TemporaryDirectory() as tmp:#    for tmp in [\"/tmp\"]:  \# MUTANT P4 shared temp dir#|runner.jobs_gemm"
)
expected_lines() { echo 1; }

{
  echo "# Phase 6 mutation check: do the tests, stall-stack assertions and functional checks detect injected bugs?"
  echo "command: scripts/phase6_mutation_check.sh"
  echo
  echo "## Provenance (of the unmutated tree the mutants were derived from)"
  echo "$prov"
  echo
  echo "## Baseline (unmutated copy)"
  base=$(failed_tests)
  total=$(grep -oE 'tests passed, [0-9]+ tests failed out of [0-9]+' "$stage/ctest.log" || true)
  echo "ctest: ${total:-no summary}"
  echo "failing tests: ${base:-none}"
  if [ -n "$base" ]; then echo "BASELINE FAILS: mutation results are meaningless"; fi
  all_caught=yes
  for m in "${mutants[@]}"; do
    IFS='|' read -r name file expr required <<< "$m"
    cp "$work/$file" "$stage/orig"
    sed -i "$expr" "$work/$file"  # basic regex: ( ? + are literals
    n=$(grep -c "MUTANT $name" "$work/$file" || true)
    echo
    echo "## $name ($file)"
    if [ "$n" -ne "$(expected_lines "$name")" ]; then
      echo "ERROR: expected $(expected_lines "$name") mutated line(s), got $n; the pattern no longer matches"
      all_caught=no
      cp "$stage/orig" "$work/$file"
      continue
    fi
    grep -n "MUTANT $name" "$work/$file"
    if ! (cd "$work" && cmake --build --preset release > "$stage/build.log" 2>&1); then
      echo "ERROR: mutant does not compile"; tail -20 "$stage/build.log"; all_caught=no
    else
      fails=$(failed_tests)
      echo "failing tests ($(echo "$fails" | grep -c . || true)):"
      echo "${fails:-  (none)}" | sed 's/^/  /'
      verdict=CAUGHT
      for t in $required; do
        grep -qx "$t" <<< "$fails" || { verdict="MISSED (required $t passed)"; all_caught=no; }
      done
      grep -hoE "(attention )?functional: .*seed=.*" "$stage/ctest.log" | sed "s/^/  3-way summary: /" || true
      echo "required: $required"
      echo "verdict: $verdict"
    fi
    cp "$stage/orig" "$work/$file"
  done
  echo
  echo "## Result"
  echo "all mutants caught by their required tests: $all_caught"
} > "$stage/mutation_check.txt"

mkdir -p results/phase6
mv "$stage/mutation_check.txt" results/phase6/mutation_check.txt
cat results/phase6/mutation_check.txt
