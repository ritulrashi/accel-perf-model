#!/usr/bin/env python3
"""Formats one GitHub Actions run (gh run view --json ...) read from stdin:
conclusion, head commit, and per-job and per-step wall times
(docs/phase6-design.md §4). Used by scripts/phase6_ci_times.sh."""
import json
import sys
from datetime import datetime


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def main():
    r = json.load(sys.stdin)
    print(f"## Run {r['databaseId']} ({r['event']}, workflow {r['workflowName']}): {r['conclusion']}")
    print(f"url: {r['url']}")
    print(f"head_sha: {r['headSha']}")
    for j in r["jobs"]:
        jt = int((ts(j["completedAt"]) - ts(j["startedAt"])).total_seconds())
        print(f"job {j['name']}: {j['conclusion']}, wall {jt} s ({jt / 60:.1f} min)")
        print("| step | seconds | result |")
        print("|---|---|---|")
        for s in j["steps"]:
            st = int((ts(s["completedAt"]) - ts(s["startedAt"])).total_seconds()) if s.get("completedAt") else 0
            print(f"| {s['name']} | {st} | {s['conclusion']} |")


if __name__ == "__main__":
    main()
