#!/usr/bin/env bash
# Rule 1: every measurement quoted in RESULTS.md must appear in a generated
# results file (scripts/check_results_numbers.py explains what is checked).
# Usage: scripts/check_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mapfile -t gen < <(find results -type f \( -name '*.txt' -o -name '*.csv' \) | sort)
python3 scripts/check_results_numbers.py RESULTS.md "${gen[@]}"
