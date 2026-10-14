#!/usr/bin/env python3
"""Result reproduction (Phase 6 design note §2).

Regenerates every committed results CSV of Phases 2-5 at HEAD, with the exact
command its phase exit script used, and byte-compares it with the committed
file. A change that silently moves any published number fails here.
Text reports (provenance, dates, wall times) and PNGs are not compared.

    python3 python/reproduce.py --tools build/release [--jobs N] [--only NAME]
    python3 python/reproduce.py --selftest

Exit status 0 only if every file is byte-identical.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def files(tools):
    """(committed CSV, command producing it, the file the command writes) per published result."""
    t = lambda name: os.path.join(tools, name)  # noqa: E731
    py = lambda script: os.path.join(ROOT, "python", script)  # noqa: E731
    return [
        ("results/phase2/functional.csv",
         [PY, py("gemm_check.py"), "--apm-gemm", t("apm_gemm"), "--gemm-ref", t("gemm_ref"), "--cases", "200",
          "--csv", "{out}/functional.csv"], "functional.csv"),
        ("results/phase3/functional_accel.csv",
         [PY, py("gemm_check.py"), "--target", "accel", "--apm-accel", t("apm_accel"), "--gemm-ref", t("gemm_ref"),
          "--cases", "100", "--csv", "{out}/functional_accel.csv"], "functional_accel.csv"),
        ("results/phase3/sweep.csv",
         [PY, py("sweep_p3.py"), "--apm-accel", t("apm_accel"), "--out-dir", "{out}"], "sweep.csv"),
        ("results/phase4/functional_gemm.csv",
         [PY, py("gemm_check.py"), "--target", "accel", "--prec", "all", "--apm-accel", t("apm_accel"),
          "--gemm-ref", t("gemm_ref"), "--cases", "200", "--csv", "{out}/functional_gemm.csv"], "functional_gemm.csv"),
        ("results/phase4/functional_attn.csv",
         [PY, py("attn_check.py"), "--tools", tools, "--cases", "80", "--csv", "{out}/functional_attn.csv"],
         "functional_attn.csv"),
        ("results/phase4/study.csv",
         [PY, py("study_p4.py"), "--tools", tools, "--out-dir", "{out}"], "study.csv"),
        ("results/phase5/correlation.csv",
         [PY, py("correlate.py"), "--tools", tools, "--csv", "{out}/correlation.csv"], "correlation.csv"),
        ("results/phase5/sweep.csv",
         [PY, py("sweep_p5.py"), "--tools", tools, "--out-dir", "{out}",
          "--phase4-study", os.path.join(ROOT, "results/phase4/study.csv")], "sweep.csv"),
    ]


def compare(committed, regenerated):
    """('IDENTICAL' | 'DIFFERS', detail) for two byte strings."""
    if committed == regenerated:
        return "IDENTICAL", f"{len(committed.splitlines())} lines"
    if committed.replace(b"\r\n", b"\n") == regenerated.replace(b"\r\n", b"\n"):
        return "DIFFERS", "line endings only (CRLF vs LF); every line's content is identical"
    a, b = committed.splitlines(), regenerated.splitlines()
    differing = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    for n, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "DIFFERS", (f"{differing} differing lines; first at line {n + 1}:\n"
                               f"    committed:   {x.decode(errors='replace')[:300]}\n"
                               f"    regenerated: {y.decode(errors='replace')[:300]}")
    return "DIFFERS", f"{differing} differing lines; line count {len(a)} committed vs {len(b)} regenerated"


def reproduce_one(entry):
    committed, cmd, produced = entry
    with tempfile.TemporaryDirectory() as out:
        argv = [c.replace("{out}", out) for c in cmd]
        start = time.time()
        p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT,
                           env=dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1"))
        wall = time.time() - start
        path = os.path.join(out, produced)
        if not os.path.exists(path):
            return committed, "MISSING", f"exit {p.returncode}: {p.stderr.strip()[-400:]}", wall, p.returncode
        with open(os.path.join(ROOT, committed), "rb") as f:
            want = f.read()
        with open(path, "rb") as f:
            got = f.read()
        status, detail = compare(want, got)
        return committed, status, detail, wall, p.returncode


def selftest():
    failures = []
    with open(os.path.join(ROOT, "results/phase4/study.csv"), "rb") as f:
        data = f.read()
    if compare(data, data)[0] != "IDENTICAL":
        failures.append("identical bytes not reported IDENTICAL")
    lines = data.splitlines(keepends=True)
    target = 7  # change one digit on line 8 of the file
    changed = bytearray(lines[target])
    pos = next(i for i, c in enumerate(changed) if chr(c).isdigit())
    changed[pos] = ord("1") if changed[pos] != ord("1") else ord("2")
    mutated = b"".join(lines[:target] + [bytes(changed)] + lines[target + 1:])
    status, detail = compare(data, mutated)
    if status != "DIFFERS" or f"line {target + 1}" not in detail:
        failures.append(f"one changed digit not found: {status} {detail}")
    if compare(data, data + b"x\n")[0] != "DIFFERS":
        failures.append("an extra line not reported")
    for f in failures:
        print("FAIL " + f)
    print(f"reproduce selftest: {'ok' if not failures else 'FAILED'} ({len(failures)} failures)")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools")
    ap.add_argument("--only", help="reproduce only committed files whose path contains this text")
    ap.add_argument("--selftest", action="store_true")
    runner.add_jobs_arg(ap)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.tools:
        ap.error("--tools is required")
    tools = os.path.abspath(args.tools)
    entries = [e for e in files(tools) if not args.only or args.only in e[0]]
    if args.jobs:  # the scripts parallelize themselves; pass the worker count through
        entries = [(c, cmd + ["--jobs", str(args.jobs)], p) for c, cmd, p in entries]
    start = time.time()
    results = [reproduce_one(e) for e in entries]  # one at a time: each script already uses every worker
    identical = 0
    for committed, status, detail, wall, rc in results:
        identical += status == "IDENTICAL"
        print(f"reproduce: {committed} {status} ({detail.splitlines()[0] if status == 'IDENTICAL' else 'see below'}) "
              f"script_exit={rc} wall_s={wall:.1f}")
        if status != "IDENTICAL":
            print("  " + detail)
    print(f"reproduce: files={len(results)} identical={identical} jobs={args.jobs or runner.default_jobs()} "
          f"wall_s={time.time() - start:.1f}")
    return 0 if identical == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
