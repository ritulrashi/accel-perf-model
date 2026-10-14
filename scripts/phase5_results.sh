#!/usr/bin/env bash
# Produces all official Phase 5 results on a clean checkout:
#   results/phase5/exit_criterion.txt, correlation.csv, sweep.csv,
#     bank_map.png, arrangements.png                    (scripts/phase5_exit.sh)
#   results/phase5/mutation_check.txt                   (scripts/phase5_mutation_check.sh)
# Same procedure as scripts/phase1_results.sh: each script's output is moved
# out of the tree before the next one records provenance (docs/BUGS.md #3).
#
# Usage: scripts/phase5_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

require_clean() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "phase5_results: tree is not clean before $1:" >&2
    git status --short >&2
    exit 1
  fi
}

hold=$(mktemp -d)
trap 'rm -rf "$hold"' EXIT

outputs="exit_criterion.txt correlation.csv sweep.csv bank_map.png arrangements.png"

require_clean phase5_exit.sh
scripts/phase5_exit.sh > /dev/null
for f in $outputs; do mv "results/phase5/$f" "$hold/"; done
rmdir results/phase5 2>/dev/null || true

require_clean phase5_mutation_check.sh
scripts/phase5_mutation_check.sh > /dev/null
mv results/phase5/mutation_check.txt "$hold/"

mkdir -p results/phase5
for f in $outputs mutation_check.txt; do mv "$hold/$f" results/phase5/; done
grep -H '^tree:' results/phase5/*.txt
