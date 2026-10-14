#!/usr/bin/env python3
"""Functional check (Phase 2 design note §4, extended in Phase 3): random INT8
GEMMs run on the model and compared bit-exactly against two independent
references, NumPy and the standalone C++ loop (gemm_ref). A three-way
comparison also tells which side is wrong when one disagrees.

    # Phase 2: one PE straight out of DRAM
    python3 python/gemm_check.py --apm-gemm build/release/apm_gemm \
        --gemm-ref build/release/gemm_ref --cases 200 [--csv out.csv]
    # Phase 3: the accelerator grid
    python3 python/gemm_check.py --target accel --apm-accel build/release/apm_accel \
        --gemm-ref build/release/gemm_ref --cases 100 [--csv out.csv]

    # Phase 4: precision modes on the grid (design note §8.2), cases split
    # evenly over int8, fp16, bf16, fp32
    python3 python/gemm_check.py --target accel --prec all --apm-accel build/release/apm_accel         --gemm-ref build/release/gemm_ref --cases 200 [--csv out.csv]

Case i draws everything from numpy.random.default_rng([SEED, i]) (Phase 3:
[SEED, ACCEL_STREAM, i]; Phase 4: [SEED, PREC_STREAM, precision index, i]);
the seeds are fixed, so the case list is reproducible. Floating-point cases
use FP32 N(0,1) operands (10 % scaled by 1000) rounded once to the mode and
compare 4-byte words: FP32 bit patterns, accumulated per design note §2.2.
Exit status 0 only if every case matches all three ways.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numerics as nx  # noqa: E402
import runner  # noqa: E402

SEED = 20261014
ACCEL_STREAM = 3  # separate stream so Phase 2 cases stay exactly as published
PREC_STREAM = 5   # Phase 4 precision cases (Phase 3 INT8 cases stay exactly as published)


def draw_operands(rng, m, n, k):
    extreme = bool(rng.random() < 0.1)
    if extreme:  # largest accumulation magnitude: (-128) * (-128) * K
        a = np.full((m, k), -128, dtype=np.int8)
        b = np.full((k, n), -128, dtype=np.int8)
    else:
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    return extreme, a, b


def draw_case_pe(i):
    rng = np.random.default_rng([SEED, i])
    r = int(rng.choice([1, 2, 4, 8]))
    c = int(rng.choice([16, 32]))
    kt = int(rng.choice([16, 32, 64]))
    m = r * int(rng.integers(1, 5))
    n = c * int(rng.integers(1, 4))
    k = kt * int(rng.integers(1, 5))
    nbuf = int(rng.choice([1, 2]))
    nout = int(rng.choice([1, 2]))
    extreme, a, b = draw_operands(rng, m, n, k)
    p = dict(case=i, M=m, N=n, K=k, R=r, C=c, Kt=kt, nbuf=nbuf, nout=nout, extreme=extreme)
    return p, a, b


def draw_case_accel(i, rng=None):
    rng = np.random.default_rng([SEED, ACCEL_STREAM, i]) if rng is None else rng
    px, py = [(1, 1), (1, 2), (2, 2), (2, 3)][int(rng.integers(0, 4))]
    r = int(rng.choice([1, 2, 4]))
    c = int(rng.choice([4, 16]))
    kt = int(rng.choice([16, 32]))
    # Up to two waves in each direction, partial waves included.
    m = r * int(rng.integers(1, 2 * px + 1))
    n = c * int(rng.integers(1, 2 * py + 1))
    k = kt * int(rng.integers(1, 4))
    p = dict(case=i, M=m, N=n, K=k, R=r, C=c, Kt=kt, px=px, py=py,
             sram_x=int(rng.integers(0, px)), sram_y=int(rng.integers(0, py)),
             D=int(rng.choice([1, 2])), nbuf=int(rng.choice([1, 2])), nout=int(rng.choice([1, 2])))
    extreme, a, b = draw_operands(rng, m, n, k)
    p["extreme"] = extreme
    return p, a, b


def numpy_ref(a, b):
    c64 = a.astype(np.int64) @ b.astype(np.int64)
    info = np.iinfo(np.int32)
    assert c64.min() >= info.min and c64.max() <= info.max, "reference overflows INT32"
    return c64.astype(np.int32)


def draw_case_prec(prec, i):
    """Phase 4: a grid case in precision `prec` (shapes as Phase 3, own stream)."""
    rng = np.random.default_rng([SEED, PREC_STREAM, nx.PRECS.index(prec), i])
    p, a, b = draw_case_accel(i, rng)
    p["prec"] = prec
    if prec == "int8":
        return p, a, b
    large = bool(rng.random() < 0.1)
    scale = np.float32(1000.0) if large else np.float32(1.0)
    af = rng.standard_normal(a.shape, dtype=np.float32) * scale
    bf = rng.standard_normal(b.shape, dtype=np.float32) * scale
    p["extreme"] = large
    return p, nx.encode(prec, af), nx.encode(prec, bf)


def run_case_prec(i, prec, model, gemm_ref, tmp):
    p, a, b = draw_case_prec(prec, i)
    keys = ("M", "N", "K", "R", "C", "Kt", "px", "py", "sram_x", "sram_y", "D", "nbuf", "nout")
    fa, fb = os.path.join(tmp, "a.bin"), os.path.join(tmp, "b.bin")
    fc_model, fc_ref = os.path.join(tmp, "c_model.bin"), os.path.join(tmp, "c_ref.bin")
    raw_a = a.view(np.uint8) if prec == "int8" else a
    raw_b = b.view(np.uint8) if prec == "int8" else b
    raw_a.tofile(fa)
    raw_b.tofile(fb)
    env = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1")
    args = [model] + [f"{k}={p[k]}" for k in keys] + [f"prec={prec}", f"a={fa}", f"b={fb}", f"out={fc_model}"]
    pm = subprocess.run(args, capture_output=True, text=True, env=env)
    if pm.returncode != 0:
        raise RuntimeError(f"case {prec}/{i}: model failed ({pm.returncode}): {pm.stderr.strip()}")
    cycles = json.loads(pm.stdout)["values"]["t_end"]
    pr = subprocess.run([gemm_ref, str(p["M"]), str(p["N"]), str(p["K"]), fa, fb, fc_ref, prec],
                        capture_output=True, text=True)
    if pr.returncode != 0:
        raise RuntimeError(f"case {prec}/{i}: gemm_ref failed: {pr.stderr.strip()}")
    shape = (p["M"], p["N"])
    c_model = np.fromfile(fc_model, dtype="<u4").reshape(shape)
    c_cpp = np.fromfile(fc_ref, dtype="<u4").reshape(shape)
    if prec == "int8":
        c_np = numpy_ref(a, b).view(np.uint32)
    else:
        c_np = np.asarray(nx.gemm(prec, nx.decode(prec, a), nx.decode(prec, b))).view(np.uint32)
    p.update(cycles=cycles,
             model_eq_numpy=bool(np.array_equal(c_model, c_np)),
             cpp_eq_numpy=bool(np.array_equal(c_cpp, c_np)),
             model_eq_cpp=bool(np.array_equal(c_model, c_cpp)),
             model_mismatches=int(np.count_nonzero(c_model != c_np)))
    p["match"] = p["model_eq_numpy"] and p["cpp_eq_numpy"] and p["model_eq_cpp"]
    return p


def run_case(i, target, model, gemm_ref, tmp):
    if target == "pe":
        p, a, b = draw_case_pe(i)
        keys = ("M", "N", "K", "R", "C", "Kt", "nbuf", "nout")
        cycles_key = "pe.done"
    else:
        p, a, b = draw_case_accel(i)
        keys = ("M", "N", "K", "R", "C", "Kt", "px", "py", "sram_x", "sram_y", "D", "nbuf", "nout")
        cycles_key = "t_end"
    fa, fb = os.path.join(tmp, "a.bin"), os.path.join(tmp, "b.bin")
    fc_model, fc_ref = os.path.join(tmp, "c_model.bin"), os.path.join(tmp, "c_ref.bin")
    a.tofile(fa)
    b.tofile(fb)
    env = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1")
    args = [model] + [f"{k}={p[k]}" for k in keys] + [f"a={fa}", f"b={fb}", f"out={fc_model}"]
    pm = subprocess.run(args, capture_output=True, text=True, env=env)
    if pm.returncode != 0:
        raise RuntimeError(f"case {i}: model failed ({pm.returncode}): {pm.stderr.strip()}")
    cycles = json.loads(pm.stdout)["values"][cycles_key]
    pr = subprocess.run([gemm_ref, str(p["M"]), str(p["N"]), str(p["K"]), fa, fb, fc_ref],
                        capture_output=True, text=True)
    if pr.returncode != 0:
        raise RuntimeError(f"case {i}: gemm_ref failed: {pr.stderr.strip()}")
    shape = (p["M"], p["N"])
    c_model = np.fromfile(fc_model, dtype="<i4").reshape(shape)
    c_cpp = np.fromfile(fc_ref, dtype="<i4").reshape(shape)
    c_np = numpy_ref(a, b)
    p.update(cycles=cycles,
             model_eq_numpy=bool(np.array_equal(c_model, c_np)),
             cpp_eq_numpy=bool(np.array_equal(c_cpp, c_np)),
             model_eq_cpp=bool(np.array_equal(c_model, c_cpp)),
             model_mismatches=int(np.count_nonzero(c_model != c_np)))
    p["match"] = p["model_eq_numpy"] and p["cpp_eq_numpy"] and p["model_eq_cpp"]
    return p


def _case_task(t):
    """One case in its own temporary directory: cases run in parallel (python/runner.py)."""
    _, target, prec, idx, model, gemm_ref = t
    with tempfile.TemporaryDirectory() as tmp:
        if prec:
            return run_case_prec(idx, prec, model, gemm_ref, tmp)
        return run_case(idx, target, model, gemm_ref, tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("pe", "accel"), default="pe")
    ap.add_argument("--apm-gemm")
    ap.add_argument("--apm-accel")
    ap.add_argument("--gemm-ref", required=True)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--prec", choices=nx.PRECS + ("all",),
                    help="Phase 4 precision cases on the grid (default: the Phase 2/3 INT8 streams)")
    ap.add_argument("--csv")
    runner.add_jobs_arg(ap)
    args = ap.parse_args()
    if args.prec and args.target != "accel":
        ap.error("--prec needs --target accel")
    model = args.apm_gemm if args.target == "pe" else args.apm_accel
    if not model:
        ap.error(f"--target {args.target} needs --apm-{'gemm' if args.target == 'pe' else 'accel'}")

    precs = None if not args.prec else (nx.PRECS if args.prec == "all" else (args.prec,))
    tasks = [(i, args.target, precs[i % len(precs)] if precs else None, i // len(precs) if precs else i,
              model, args.gemm_ref) for i in range(args.cases)]
    rows = []
    for (i, *_), row, err in runner.run_tasks(tasks, _case_task, args.jobs):
        if err is not None:
            raise RuntimeError(f"case {i}: {err}")
        rows.append(row)
        shape = " ".join(f"{k}={row[k]}" for k in row
                         if k not in ("case", "extreme", "cycles", "model_eq_numpy", "cpp_eq_numpy",
                                      "model_eq_cpp", "model_mismatches", "match"))
        flag = "OK" if row["match"] else "MISMATCH"
        print(f"case {i}: {shape} extreme={int(row['extreme'])} cycles={row['cycles']} "
              f"model==numpy={int(row['model_eq_numpy'])} cpp==numpy={int(row['cpp_eq_numpy'])} "
              f"model==cpp={int(row['model_eq_cpp'])} {flag}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    ok = sum(r["match"] for r in rows)
    if precs:
        for pr in precs:
            rs = [r for r in rows if r["prec"] == pr]
            print(f"functional: prec={pr} cases={len(rs)} all_three_agree={sum(r['match'] for r in rs)}")
    print(f"functional: target={args.target} seed={SEED} cases={len(rows)} all_three_agree={ok} "
          f"model_wrong={sum(not r['model_eq_numpy'] for r in rows)} "
          f"cpp_ref_wrong={sum(not r['cpp_eq_numpy'] for r in rows)} "
          f"extreme_cases={sum(r['extreme'] for r in rows)}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
