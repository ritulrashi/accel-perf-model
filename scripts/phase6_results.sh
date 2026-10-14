#!/usr/bin/env bash
# Produces all official Phase 6 results on a clean checkout:
#   results/phase6/exit_criterion.txt, regression_runtime.txt,
#     runner_speedup.txt                                (scripts/phase6_exit.sh)
#   results/phase6/mutation_check.txt                   (scripts/phase6_mutation_check.sh)
# Same procedure as scripts/phase1_results.sh: each script's output is moved
# out of the tree before the next one records provenance (docs/BUGS.md #3).
#
# Usage: scripts/phase6_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "phase6_results: tree is not clean before $1:" >&2
    git status --short >&2
    exit 1
  fi
}

hold=$(mktemp -d)
trap 'rm -rf "$hold"' EXIT

outputs="exit_criterion.txt regression_runtime.txt runner_speedup.txt"

require_clean phase6_exit.sh
scripts/phase6_exit.sh > /dev/null
for f in $outputs; do mv "results/phase6/$f" "$hold/"; done
rmdir results/phase6 2>/dev/null || true

require_clean phase6_mutation_check.sh
scripts/phase6_mutation_check.sh > /dev/null
mv results/phase6/mutation_check.txt "$hold/"

mkdir -p results/phase6
for f in $outputs mutation_check.txt; do mv "$hold/$f" results/phase6/; done
grep -H '^tree:' results/phase6/*.txt
