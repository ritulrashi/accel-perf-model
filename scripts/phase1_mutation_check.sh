#!/usr/bin/env bash
# Can the Phase 1 tests fail? (rule 6: a clean result is a suspected bug in the
# comparison until shown otherwise). Copies the tracked tree to a temp dir,
# checks the unmutated copy passes, then injects one mutant at a time
# (docs/phase1-design.md §7), rebuilds, and records which tests fail. Each
# mutant must be caught by the tests the design note names for it. Nothing in
# the real tree is modified.
#
# Usage: scripts/phase1_mutation_check.sh
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
  "M1|src/mem/dram_timing.cpp|s#(s.hit ? 0 : p_.t_miss - p_.t_hit);#(p_.t_miss - p_.t_hit);  // MUTANT M1 row hit costs a miss#|sim.Phase1.T1_SingleTransfer sim.Phase1.T4a_RowHit"
  "M2|src/mem/sram_timing.cpp|s#const Cycle c = std::max(accept, bank_free_\[b\]);#const Cycle c = accept + own[b];  // MUTANT M2 no inter-request bank conflicts#|sim.Phase1.T3a_SramBankConflict"
  "M3|src/mem/dram_timing.cpp|s#s.data_start = std::max(s.col_issue + p_.t_hit, bus_free_);#s.data_start = s.col_issue + p_.t_hit;  // MUTANT M3 bus not shared#|sim.Phase1.T2_TwoDmasShareOneChannel sim.Phase1.T4d_OtherBankBackToBack"
  "M4|src/mem/dram_channel.cpp|s#if (in_queue_.size() >= timing_.params().queue_depth) return false;#if (false) return false;  // MUTANT M4 queue never refuses#|sim.Phase1.T2b_QueueBackpressure"
  "M5|src/common/rr_arbiter.h|s#p_ = (port + 1) % n_;#p_ = 0 * port;  // MUTANT M5 fixed priority, port 0 first#|sim.Phase1.T2_TwoDmasShareOneChannel"
  "M6|src/dma/dma_engine.cpp|s#^    ++buffers_free_;\$#    // MUTANT M6 (buffer freed at read response instead)#; s#write_q_.push_back(it->second);#write_q_.push_back(it->second); ++buffers_free_;  // MUTANT M6#|sim.Phase1.T1b_OneBufferIsLatencyBound"
  "M7|src/mem/dram_timing.cpp|s#s.start = std::max(accept, bank_ready_\[a.bank\]);#s.start = accept;  // MUTANT M7 no bank occupancy#|sim.Phase1.T4c_SameBankBackToBack"
  "M8|src/mem/sampled_target.cpp|/else if (trans.is_write())/{n;s#std::memcpy(store_.data() + addr, trans.get_data_ptr(), len);#(void)0;  // MUTANT M8 writes not stored#}|sim.Phase1.T1_SingleTransfer sim.Phase1.T1c_ReverseDirection sim.Phase1.F2D_DramToSramStrided sim.Phase1.F2D_SramToDramStrided"
)
expected_lines() { case $1 in M6) echo 2 ;; *) echo 1 ;; esac; }

{
  echo "# Phase 1 mutation check: do the directed tests detect injected model bugs?"
  echo "command: scripts/phase1_mutation_check.sh"
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
      echo "required: $required"
      echo "verdict: $verdict"
    fi
    cp "$stage/orig" "$work/$file"
  done
  echo
  echo "## Result"
  echo "all mutants caught by their required tests: $all_caught"
} > "$stage/mutation_check.txt"

mkdir -p results/phase1
mv "$stage/mutation_check.txt" results/phase1/mutation_check.txt
cat results/phase1/mutation_check.txt
