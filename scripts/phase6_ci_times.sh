#!/usr/bin/env bash
# Records the wall time of GitHub Actions runs (docs/phase6-design.md §4):
# per job and per step, the runner label, the head commit and the conclusion,
# read from the GitHub API with `gh`. Needs an authenticated gh; runs on the
# workstation, not in the Codespace (which has no GitHub credentials).
#
# Usage: scripts/phase6_ci_times.sh OUT_FILE RUN_ID [RUN_ID ...]
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1
shift
py=$(command -v python3 || command -v python)
{
  echo "# GitHub Actions regression runtime"
  echo "command: scripts/phase6_ci_times.sh $out $*"
  echo
  echo "## Provenance (of this query; the runs record their own commit below)"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "local_commit: $(git rev-parse HEAD)"
  echo "gh: $(gh --version | head -1)"
  for run in "$@"; do
    echo
    gh run view "$run" --repo ritulrashi/accel-perf-model \
      --json databaseId,event,headSha,conclusion,createdAt,updatedAt,url,workflowName,jobs \
      | "$py" python/ci_times.py
  done
} > "$out"
cat "$out"
