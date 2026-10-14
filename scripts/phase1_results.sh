#!/usr/bin/env bash
# Produces all official Phase 1 results on a clean checkout:
#   results/phase1/exit_criterion.txt   (scripts/phase1_exit.sh)
#   results/phase1/mutation_check.txt   (scripts/phase1_mutation_check.sh)
# Each script records provenance before it writes anything. Because both
# write into results/phase1, the first one's output is moved out of the tree
# before the second one starts, and both are put back at the end. Otherwise
# the second provenance block would say DIRTY (docs/BUGS.md #3).
#
# Usage: scripts/phase1_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "phase1_results: tree is not clean before $1:" >&2
    git status --short >&2
    exit 1
  fi
}

hold=$(mktemp -d)
trap 'rm -rf "$hold"' EXIT

require_clean phase1_exit.sh
scripts/phase1_exit.sh > /dev/null
mv results/phase1/exit_criterion.txt "$hold/"
rmdir results/phase1 results 2>/dev/null || true

require_clean phase1_mutation_check.sh
scripts/phase1_mutation_check.sh > /dev/null
mv results/phase1/mutation_check.txt "$hold/"

mkdir -p results/phase1
mv "$hold/exit_criterion.txt" "$hold/mutation_check.txt" results/phase1/
grep -H '^tree:' results/phase1/*.txt
