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

Case i draws everything from numpy.random.default_rng([SEED, i]) (Phase 3:
[SEED, ACCEL_STREAM, i]); the seeds are fixed, so the case list is
reproducible. Exit status 0 only if every case matches all three ways.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SEED = 20261014
ACCEL_STREAM = 3  # separate stream so Phase 2 cases stay exactly as published


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


def draw_case_accel(i):
    rng = np.random.default_rng([SEED, ACCEL_STREAM, i])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("pe", "accel"), default="pe")
    ap.add_argument("--apm-gemm")
    ap.add_argument("--apm-accel")
    ap.add_argument("--gemm-ref", required=True)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--csv")
    args = ap.parse_args()
    model = args.apm_gemm if args.target == "pe" else args.apm_accel
    if not model:
        ap.error(f"--target {args.target} needs --apm-{'gemm' if args.target == 'pe' else 'accel'}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.cases):
            row = run_case(i, args.target, model, args.gemm_ref, tmp)
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
    print(f"functional: target={args.target} seed={SEED} cases={len(rows)} all_three_agree={ok} "
          f"model_wrong={sum(not r['model_eq_numpy'] for r in rows)} "
          f"cpp_ref_wrong={sum(not r['cpp_eq_numpy'] for r in rows)} "
          f"extreme_cases={sum(r['extreme'] for r in rows)}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
