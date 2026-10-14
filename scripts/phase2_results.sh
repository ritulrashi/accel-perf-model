#!/usr/bin/env bash
# Produces all official Phase 2 results on a clean checkout:
#   results/phase2/exit_criterion.txt, functional.csv   (scripts/phase2_exit.sh)
#   results/phase2/mutation_check.txt                   (scripts/phase2_mutation_check.sh)
# Same procedure as scripts/phase1_results.sh: each script's output is moved
# out of the tree before the next one records provenance (docs/BUGS.md #3).
#
# Usage: scripts/phase2_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "phase2_results: tree is not clean before $1:" >&2
    git status --short >&2
    exit 1
  fi
}

hold=$(mktemp -d)
trap 'rm -rf "$hold"' EXIT

require_clean phase2_exit.sh
scripts/phase2_exit.sh > /dev/null
mv results/phase2/exit_criterion.txt results/phase2/functional.csv "$hold/"
rmdir results/phase2 2>/dev/null || true

require_clean phase2_mutation_check.sh
scripts/phase2_mutation_check.sh > /dev/null
mv results/phase2/mutation_check.txt "$hold/"

mkdir -p results/phase2
mv "$hold/exit_criterion.txt" "$hold/functional.csv" "$hold/mutation_check.txt" results/phase2/
grep -H '^tree:' results/phase2/*.txt
