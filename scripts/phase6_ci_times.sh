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
py=$(for c in python3 python; do "$c" --version > /dev/null 2>&1 && { echo "$c"; break; }; done)  # Windows: python3 may be a Store stub
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
    echo "regression steps inside the job (scripts/phase6_regression.sh table, from the job log):"
    gh run view "$run" --repo ritulrashi/accel-perf-model --log 2>/dev/null \
      | grep -oE '(host: .*cpus: [0-9]+|\| [A-Za-z0-9_]+ +\| +[0-9]+ \| [^|]+\|)$' | awk '!seen[$0]++' || true
  done
} > "$out"
cat "$out"
