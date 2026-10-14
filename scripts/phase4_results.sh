#!/usr/bin/env bash
# Produces all official Phase 4 results on a clean checkout:
#   results/phase4/exit_criterion.txt, functional_gemm.csv, functional_attn.csv,
#     study.csv, attention_breakdown.png, numerics.png, attention_roofline.png (scripts/phase4_exit.sh)
#   results/phase4/mutation_check.txt                   (scripts/phase4_mutation_check.sh)
# Same procedure as scripts/phase1_results.sh: each script's output is moved
# out of the tree before the next one records provenance (docs/BUGS.md #3).
#
# Usage: scripts/phase4_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "phase4_results: tree is not clean before $1:" >&2
    git status --short >&2
    exit 1
  fi
}

hold=$(mktemp -d)
trap 'rm -rf "$hold"' EXIT

outputs="exit_criterion.txt functional_gemm.csv functional_attn.csv study.csv attention_breakdown.png numerics.png attention_roofline.png"

require_clean phase4_exit.sh
scripts/phase4_exit.sh > /dev/null
for f in $outputs; do mv "results/phase4/$f" "$hold/"; done
rmdir results/phase4 2>/dev/null || true

require_clean phase4_mutation_check.sh
scripts/phase4_mutation_check.sh > /dev/null
mv results/phase4/mutation_check.txt "$hold/"

mkdir -p results/phase4
for f in $outputs mutation_check.txt; do mv "$hold/$f" results/phase4/; done
grep -H '^tree:' results/phase4/*.txt
