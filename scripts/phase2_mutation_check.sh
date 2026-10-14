#!/usr/bin/env bash
# Can the Phase 2 tests and the functional comparison fail? (rule 6). Copies
# the tracked tree to a temp dir, checks the unmutated copy passes, then
# injects one mutant at a time (docs/phase2-design.md §7), rebuilds, and
# records which tests fail. Each mutant must be caught by the tests the design
# note names for it. For mutants caught by the functional check, the 3-way
# summary line shows which side (model or C++ reference) was blamed. Nothing
# in the real tree is modified.
#
# Usage: scripts/phase2_mutation_check.sh
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
  "N1|src/compute/systolic.h|s#{ return 2ull \* a.rows + a.cols - 2; }#{ return a.rows + a.cols - 2; }  // MUTANT N1 no drain#|sim.Phase2.T1_OneTileOneChunk unit.Systolic.TileLatencyFormula"
  "N2|src/compute/systolic.cpp|s#acc_\[std::size_t{i} \* a_.cols + j\] += sum;#acc_[std::size_t{i} * a_.cols + j] = sum;  // MUTANT N2 overwrite#|functional.gemm20"
  "N3|src/dma/dma_engine.cpp|s#^      !to_read_.empty() && buffers_free_ > 0#      (to_read_.empty() or to_read_.front()->desc == 0 or results_[to_read_.front()->desc - 1].complete) and /* MUTANT N3 one descriptor at a time */ !to_read_.empty() \&\& buffers_free_ > 0#|sim.Phase2.T6_DmaStreamsAcrossDescriptors sim.Phase2.T2_DoubleBufferedComputeBound"
  "N4|src/pe/pe.cpp|s#^  if (loads_done_\[s\] < 2) return;#  if (false) return;  // MUTANT N4 feed ignores loads#|sim.Phase2.T4_DoubleBufferedMemoryBound functional.gemm20"
  "N5|src/pe/pe.cpp|s#return f >= 0 \&\& static_cast<Cycle>(f) + cfg_.kt <= now;#return f >= 0 \&\& static_cast<Cycle>(f) <= now;  // MUTANT N5 freed at feed start#|sim.Phase2.T3_SingleBuffer"
  "N6|src/pe/pe.cpp|s#^    if (tile_in_array_) return;.*#    // MUTANT N6 next tile does not wait for the drain#|sim.Phase2.T5_TwoOutputTiles"
  "N7|src/pe/pe.cpp|s#^    if (t >= cfg_.n_out \&\& store_done_\[t - cfg_.n_out\] < 0) return;.*#    // MUTANT N7 output buffer rule ignored#|sim.Phase2.T5b_OneOutputBuffer"
  "N8|src/pe/gemm_layout.cpp|s#const std::uint64_t blk = b_block_addr(nj, c) - b_base;#const std::uint64_t blk = b_block_addr(nj, (c + 1) % chunks()) - b_base;  // MUTANT N8 B blocks permuted#|functional.gemm20"
  "N9|tools/gemm_ref.cpp|s#for (std::uint64_t k = 0; k < K; ++k) s +=#for (std::uint64_t k = 0; k + 1 < K; ++k) s +=  /* MUTANT N9 reference drops last k */#|functional.gemm20"
)
expected_lines() { echo 1; }

{
  echo "# Phase 2 mutation check: do the timing tests and the functional comparison detect injected bugs?"
  echo "command: scripts/phase2_mutation_check.sh"
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
      grep -hoE "functional: seed=.*" "$stage/ctest.log" | sed "s/^/  3-way summary: /" || true
      echo "required: $required"
      echo "verdict: $verdict"
    fi
    cp "$stage/orig" "$work/$file"
  done
  echo
  echo "## Result"
  echo "all mutants caught by their required tests: $all_caught"
} > "$stage/mutation_check.txt"

mkdir -p results/phase2
mv "$stage/mutation_check.txt" results/phase2/mutation_check.txt
cat results/phase2/mutation_check.txt
