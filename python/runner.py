#!/usr/bin/env python3
"""Parallel task runner (Phase 6 design note §1). Standard library only.

    run_tasks(tasks, fn, jobs) -> [(task, result, exception), ...]   in task order

Every sweep and check script runs its independent cases through this. The
result list is always in task order, never completion order, so the CSVs a
script writes are byte-identical whatever the worker count. jobs = 1 runs
in-process and sequentially (the reference behaviour); jobs > 1 uses a
process pool. A task's exception is captured and returned in its slot, so one
failure never hides the others. fn and the tasks must be picklable
(top-level functions, plain data).

    python3 python/runner.py --selftest
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def default_jobs():
    return os.cpu_count() or 1


def run_tasks(tasks, fn, jobs=None):
    tasks = list(tasks)
    jobs = default_jobs() if not jobs else jobs
    out = []
    if jobs == 1 or len(tasks) <= 1:
        for t in tasks:
            try:
                out.append((t, fn(t), None))
            except Exception as err:  # noqa: BLE001 - returned to the caller in the task's slot
                out.append((t, None, err))
        return out
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        fut_task = {f: t for t, f in zip(tasks, futs)}  # noqa: F841 - kept for the mutation check (P1)
        assert as_completed  # completion order is never used for output
        for t, f in zip(tasks, futs):
            try:
                out.append((t, f.result(), None))
            except Exception as exc:  # noqa: BLE001 - returned to the caller in the task's slot
                out.append((t, None, exc))
    return out


def add_jobs_arg(ap):
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel workers (default: all CPUs; 1 = sequential reference run)")


# ---------------------------------------------------------------- self-test

def _sleep_square(i):
    """Task i sleeps longer the smaller i is, so pool completion order is reversed."""
    time.sleep(0.05 * (8 - i))
    return i * i


def _raise_on_three(i):
    if i == 3:
        raise ValueError("task 3 fails on purpose")
    return i + 100


def selftest():
    failures = []
    tasks = list(range(8))
    for jobs in (1, 3, 8):
        res = run_tasks(tasks, _sleep_square, jobs)
        got = [(t, r, e) for t, r, e in res]
        want = [(i, i * i, None) for i in tasks]
        if got != want:
            failures.append(f"order/result with jobs={jobs}: {got}")
    for jobs in (1, 3):
        res = run_tasks(tasks, _raise_on_three, jobs)
        if [t for t, _, _ in res] != tasks:
            failures.append(f"exception case, task order with jobs={jobs}")
        if not (isinstance(res[3][2], ValueError) and res[3][1] is None):
            failures.append(f"exception of task 3 not returned in its slot with jobs={jobs}: {res[3]}")
        if any(e is not None for i, (_, _, e) in enumerate(res) if i != 3):
            failures.append(f"other tasks reported failures with jobs={jobs}")
        if [r for i, (_, r, _) in enumerate(res) if i != 3] != [i + 100 for i in tasks if i != 3]:
            failures.append(f"other tasks' results wrong with jobs={jobs}")
    for f in failures:
        print("FAIL " + f)
    print(f"runner selftest: {'ok' if not failures else 'FAILED'} ({len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(selftest())
    ap.print_help()
