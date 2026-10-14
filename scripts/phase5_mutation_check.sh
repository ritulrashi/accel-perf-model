#!/usr/bin/env bash
# Can the Phase 5 checks fail? (rule 6). Copies the tracked tree to a temp
# dir, checks the unmutated copy passes, then injects one mutant at a time
# (docs/phase5-design.md §10), rebuilds, and records which tests fail. Each
# mutant must be caught by the tests the design note names for it. Nothing in
# the real tree is modified.
#
# Usage: scripts/phase5_mutation_check.sh
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
  "R1|rtl/os_array.sv|s#      assign west\[i\] = sr\[i-1\];#      assign west[i] = {in_valid, in_first, in_last, in_k, a_col[i]};  // MUTANT R1 row skew missing#|rtl.ht1 rtl.ht2 rtl.correlate_quick"
  "R2|rtl/os_array.sv|s#  assign drain_row   = RW'(drain_left - DW'(1));  // row R-1 first#  assign drain_row   = RW'(DW'(R) - drain_left);  // MUTANT R2 row 0 first#|rtl.ht1 rtl.ht2 rtl.correlate_quick"
  "R3|rtl/os_cell.sv|s#      if (v_in) acc <= first_in ? prod : acc + prod;#      if (v_in) acc <= acc + prod;  // MUTANT R3 no clear on first#|rtl.correlate_quick"
  "R4|rtl/os_array.sv|s#  assign corner_last = v_i\[R-1\]\[C-1\] \&\& l_i\[R-1\]\[C-1\];#  assign corner_last = v_i[R-1][C] \&\& l_i[R-1][C];  // MUTANT R4 drain one cycle late#|rtl.ht1 rtl.ht2 rtl.correlate_quick"
  "R5|src/compute/contract_trace.cpp|s#at(mac_cycle(st.feed, i, j, k))#at(mac_cycle(st.feed, i, 0, k))  /* MUTANT R5 contract drops j */#|unit.ContractTrace.HT1MatchesHandTrace unit.ContractTrace.HT2MatchesHandTrace rtl.ht1 rtl.correlate_quick"
  "R6|tb/os_array_tb.cpp|s#m.b_row\[j\] = st.b\[k \* s.cols + j\];#m.b_row[j] = st.b[(k == 0 ? 0 : k - 1) * s.cols + j];  // MUTANT R6 testbench B row late#|rtl.ht1 rtl.ht2 rtl.correlate_quick"
  "X1|src/mem/dram_timing.cpp|s#  const std::uint64_t bank = p_.bank_map == BankMap::Xor .*#  const std::uint64_t bank = row_index % p_.banks;  // MUTANT X1 xor map ignored#|sim.Phase5.T5a_XorMapSplitsT4c sim.Phase5.T5d_StrideOfFourRowsXorMap unit.DramTiming.XorMapHandRowsAndBijection"
  "X2|src/mem/dram_timing.cpp|s#(row_index ^ row) % p_.banks#(row_index ^ (row_index / (2 * p_.banks))) % p_.banks /* MUTANT X2 wrong xor term */#|unit.DramTiming.XorMapHandRowsAndBijection sim.Phase5.T5d_StrideOfFourRowsXorMap"
)
expected_lines() { echo 1; }

{
  echo "# Phase 5 mutation check: do the tests, stall-stack assertions and functional checks detect injected bugs?"
  echo "command: scripts/phase5_mutation_check.sh"
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

mkdir -p results/phase5
mv "$stage/mutation_check.txt" results/phase5/mutation_check.txt
cat results/phase5/mutation_check.txt
