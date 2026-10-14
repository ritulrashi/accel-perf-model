#!/usr/bin/env bash
# Produces all official Phase 3 results on a clean checkout:
#   results/phase3/exit_criterion.txt, functional_accel.csv, sweep.csv,
#     roofline.png, stall_stacks.png                    (scripts/phase3_exit.sh)
#   results/phase3/mutation_check.txt                   (scripts/phase3_mutation_check.sh)
# Same procedure as scripts/phase1_results.sh: each script's output is moved
# out of the tree before the next one records provenance (docs/BUGS.md #3).
#
# Usage: scripts/phase3_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "phase3_results: tree is not clean before $1:" >&2
    git status --short >&2
    exit 1
  fi
}

hold=$(mktemp -d)
trap 'rm -rf "$hold"' EXIT

outputs="exit_criterion.txt functional_accel.csv sweep.csv roofline.png stall_stacks.png"

require_clean phase3_exit.sh
scripts/phase3_exit.sh > /dev/null
for f in $outputs; do mv "results/phase3/$f" "$hold/"; done
rmdir results/phase3 2>/dev/null || true

require_clean phase3_mutation_check.sh
scripts/phase3_mutation_check.sh > /dev/null
mv results/phase3/mutation_check.txt "$hold/"

mkdir -p results/phase3
for f in $outputs mutation_check.txt; do mv "$hold/$f" results/phase3/; done
grep -H '^tree:' results/phase3/*.txt
