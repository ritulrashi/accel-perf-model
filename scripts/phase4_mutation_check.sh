#!/usr/bin/env bash
# Can the Phase 4 checks fail? (rule 6). Copies the tracked tree to a temp
# dir, checks the unmutated copy passes, then injects one mutant at a time
# (docs/phase4-design.md §9), rebuilds, and records which tests fail. Each
# mutant must be caught by the tests the design note names for it. Nothing in
# the real tree is modified.
#
# Usage: scripts/phase4_mutation_check.sh
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
  "M1|src/pe/pe_config.h|s#Cycle feed_cycles() const { return cycles_per_step() \* kt; }#Cycle feed_cycles() const { return kt; }  // MUTANT M1 feed ignores q#|sim.Phase4.T1_Fp16"
  "M2|src/pe/gemm_layout.h|s#std::uint64_t b_block_bytes() const { return kt \* array.cols \* elem(); }#std::uint64_t b_block_bytes() const { return kt * array.cols; }  // MUTANT M2 staging ignores e#|sim.Phase4.T1_Fp16 functional.accel_prec20"
  "M3|src/compute/systolic.cpp|s#const float prod = a\[i \* kt + k\] \* b\[k \* a_.cols + j\];#const float prod = 0.0f; acc = std::fma(a[i * kt + k], b[k * a_.cols + j], acc);  // MUTANT M3 fused multiply-add#|functional.accel_prec20 unit.Systolic.Fp32AccumulateIsNotFused"
  "M4|src/compute/numerics.cpp|s#if (rem > 0x1000u .*#if (rem > 0xffffu) ++m16;  // MUTANT M4 FP16 truncates (rem <= 0x1fff: never rounds up)#|unit.Numerics.Fp16RoundToNearestEvenHandCases functional.attn10"
  "M5|src/compute/numerics.cpp|s#lane\[j % lanes\] += e\[j\];#lane[0] += e[j];  // MUTANT M5 sequential sum#|unit.Numerics.LaneOrderSumHandCase functional.attn10"
  "M6|src/vec/softmax.cpp|s#auto valid = \[\&\](std::uint64_t j) { return .*#auto valid = [\&](std::uint64_t j) { (void)j; (void)row; return true; };  // MUTANT M6 mask ignored#|unit.Softmax.CausalRowZeroIsOneHot functional.attn10"
  "M7|src/vec/vu_plan.h|s#return 3 \* ((n + c.lanes - 1) / c.lanes + c.t_vu);#return 2 * ((n + c.lanes - 1) / c.lanes + c.t_vu);  // MUTANT M7 two passes#|sim.Phase4.V1_Fp16 unit.VuPlan.RowCycles"
  "M8|src/vec/vu_plan.cpp|s#const std::uint64_t dst = j.p_base + (mi \* j.chunks() \* j.r_pack + i) \* seg, stride = j.r_pack \* seg;#(void)mi; (void)i; const std::uint64_t dst = j.p_base + row * j.n * j.elem(), stride = seg;  // MUTANT M8 P row-major#|sim.Phase4.V2_TwoSegments unit.VuPlan.PackedStoreDescriptors functional.attn10"
  "M9|src/soc/soc_plan.cpp|s#static_cast<std::uint32_t>(l_.tile_valid_rows(t.mi)), row, row, 4 \* l_.shape.n};#l_.array.rows, row, row, 4 * l_.shape.n};  // MUTANT M9 write-back ignores m_valid#|sim.Phase4.T4_MaskedWriteBack unit.SocPlan.MaskedWriteBack"
  "M10|src/soc/soc_plan.cpp|s#const unsigned mx = cfg.map_x(), my = cfg.map_y();#const unsigned mx = cfg.px, my = cfg.py;  // MUTANT M10 logical mapping ignored#|unit.SocPlan.LogicalMapping1x4"
  "M11|tools/attn_ref.cpp|s#j <= i + q_off; };#j < i + q_off; };  // MUTANT M11 reference mask off by one#|functional.attn10"
)
expected_lines() { echo 1; }

{
  echo "# Phase 4 mutation check: do the tests, stall-stack assertions and functional checks detect injected bugs?"
  echo "command: scripts/phase4_mutation_check.sh"
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

mkdir -p results/phase4
mv "$stage/mutation_check.txt" results/phase4/mutation_check.txt
cat results/phase4/mutation_check.txt
