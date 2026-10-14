#!/usr/bin/env bash
# Can the Phase 3 checks fail? (rule 6). Copies the tracked tree to a temp
# dir, checks the unmutated copy passes, then injects one mutant at a time
# (docs/phase3-design.md §7), rebuilds, and records which tests fail. Each
# mutant must be caught by the tests the design note names for it. Nothing in
# the real tree is modified.
#
# Usage: scripts/phase3_mutation_check.sh
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
  "S1|src/noc/mesh_timing.cpp|s#const Cycle start = std::max(t, free_\[l\]);#const Cycle start = t;  // MUTANT S1 no link contention#|sim.Phase3.T2_OneHopAndAnIdlePe unit.MeshTiming.TwoPacketsShareALink"
  "S2|src/noc/mesh_timing.cpp|s#^    t = start + p_.hop;#    t = start;  // MUTANT S2 no hop latency#|sim.Phase3.T2_OneHopAndAnIdlePe unit.MeshTiming.SinglePacketThreeHops"
  "S3|src/common/trace.h|s#if (s.from <= c \&\& c < s.to) return s.cat;#if (s.from <= c \&\& c < s.to) return s.cat == StallCat::Network ? StallCat::Dma : s.cat;  // MUTANT S3 mesh charged as dma#|sim.Phase3.T2_OneHopAndAnIdlePe"
  "S4|src/soc/stall.cpp|s#constexpr StallCat kIdle = StallCat::Idle;#constexpr StallCat kIdle = StallCat::Dma;  // MUTANT S4 never idle#|sim.Phase3.T1_SinglePeAtSramNode sim.Phase3.T2_OneHopAndAnIdlePe"
  "S5|src/soc/stall.cpp|s#constexpr StallCat kTail = StallCat::Compute;#constexpr StallCat kTail = StallCat::Idle;  // MUTANT S5 tail charged as idle#|sim.Phase3.T3_Grid2x2FunctionalAndDramBytes"
  "S6|src/soc/accel.cpp|s#^      staged_\[next_q_\]\[b\].submit = static_cast<std::int64_t>(now);#      { const std::size_t id2 = gdma.submit({DmaDir::DramToSram, blk.dram_addr, blk.sram_addr, 1, blk.bytes, blk.bytes, blk.bytes}); if (id2 >= desc_.size()) desc_.resize(id2 + 1); desc_[id2] = {true, next_q_, b}; }  staged_[next_q_][b].submit = static_cast<std::int64_t>(now);  // MUTANT S6 no dedup: block staged twice#|sim.Phase3.T3_Grid2x2FunctionalAndDramBytes"
  "S7|src/soc/accel.cpp|s#consumers_left_\[q - d\] == 0;#blocks_left_[q - d] == 0;  /* MUTANT S7 slot freed when staged, not when consumed */#|functional.accel20"
  "S8|src/soc/soc_plan.cpp|s#return slot \* slot_bytes_ + t.x \* l_.a_block_bytes();#return slot * slot_bytes_ + t.y * l_.a_block_bytes();  // MUTANT S8 wrong A block#|sim.Phase3.T3_Grid2x2FunctionalAndDramBytes functional.accel20"
)
expected_lines() { echo 1; }

{
  echo "# Phase 3 mutation check: do the tests, stall-stack assertions and functional checks detect injected bugs?"
  echo "command: scripts/phase3_mutation_check.sh"
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
      grep -hoE "functional: .*seed=.*" "$stage/ctest.log" | sed "s/^/  3-way summary: /" || true
      echo "required: $required"
      echo "verdict: $verdict"
    fi
    cp "$stage/orig" "$work/$file"
  done
  echo
  echo "## Result"
  echo "all mutants caught by their required tests: $all_caught"
} > "$stage/mutation_check.txt"

mkdir -p results/phase3
mv "$stage/mutation_check.txt" results/phase3/mutation_check.txt
cat results/phase3/mutation_check.txt
