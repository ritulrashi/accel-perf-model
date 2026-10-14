#!/usr/bin/env python3
"""Phase 2 functional check (design note §4): random INT8 GEMMs run on the
model (apm_gemm) and compared bit-exactly against two independent references,
NumPy and the standalone C++ loop (gemm_ref). A three-way comparison also
tells which side is wrong when one disagrees.

    python3 python/gemm_check.py --apm-gemm build/release/apm_gemm \
        --gemm-ref build/release/gemm_ref --cases 200 [--csv out.csv]

Case i draws everything from numpy.random.default_rng([SEED, i]); SEED is
fixed below, so the case list is reproducible. Exit status 0 only if every
case matches all three ways.
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


def draw_case(i):
    rng = np.random.default_rng([SEED, i])
    r = int(rng.choice([1, 2, 4, 8]))
    c = int(rng.choice([16, 32]))
    kt = int(rng.choice([16, 32, 64]))
    m = r * int(rng.integers(1, 5))
    n = c * int(rng.integers(1, 4))
    k = kt * int(rng.integers(1, 5))
    nbuf = int(rng.choice([1, 2]))
    nout = int(rng.choice([1, 2]))
    extreme = bool(rng.random() < 0.1)
    if extreme:  # largest accumulation magnitude: (-128) * (-128) * K
        a = np.full((m, k), -128, dtype=np.int8)
        b = np.full((k, n), -128, dtype=np.int8)
    else:
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    return dict(case=i, M=m, N=n, K=k, R=r, C=c, Kt=kt, nbuf=nbuf, nout=nout, extreme=extreme), a, b


def numpy_ref(a, b):
    c64 = a.astype(np.int64) @ b.astype(np.int64)
    info = np.iinfo(np.int32)
    assert c64.min() >= info.min and c64.max() <= info.max, "reference overflows INT32"
    return c64.astype(np.int32)


def run_case(i, apm_gemm, gemm_ref, tmp):
    p, a, b = draw_case(i)
    fa, fb = os.path.join(tmp, "a.bin"), os.path.join(tmp, "b.bin")
    fc_model, fc_ref = os.path.join(tmp, "c_model.bin"), os.path.join(tmp, "c_ref.bin")
    a.tofile(fa)
    b.tofile(fb)
    env = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1")
    args = [apm_gemm] + [f"{k}={p[k]}" for k in ("M", "N", "K", "R", "C", "Kt", "nbuf", "nout")]
    args += [f"a={fa}", f"b={fb}", f"out={fc_model}"]
    pm = subprocess.run(args, capture_output=True, text=True, env=env)
    if pm.returncode != 0:
        raise RuntimeError(f"case {i}: apm_gemm failed ({pm.returncode}): {pm.stderr.strip()}")
    cycles = json.loads(pm.stdout)["values"]["pe.done"]
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
    ap.add_argument("--apm-gemm", required=True)
    ap.add_argument("--gemm-ref", required=True)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.cases):
            row = run_case(i, args.apm_gemm, args.gemm_ref, tmp)
            rows.append(row)
            flag = "OK" if row["match"] else "MISMATCH"
            print(f"case {i}: M={row['M']} N={row['N']} K={row['K']} array={row['R']}x{row['C']} "
                  f"Kt={row['Kt']} nbuf={row['nbuf']} nout={row['nout']} extreme={int(row['extreme'])} "
                  f"cycles={row['cycles']} model==numpy={int(row['model_eq_numpy'])} "
                  f"cpp==numpy={int(row['cpp_eq_numpy'])} model==cpp={int(row['model_eq_cpp'])} {flag}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="
")
            w.writeheader()
            w.writerows(rows)
    ok = sum(r["match"] for r in rows)
    print(f"functional: seed={SEED} cases={len(rows)} all_three_agree={ok} "
          f"model_wrong={sum(not r['model_eq_numpy'] for r in rows)} "
          f"cpp_ref_wrong={sum(not r['cpp_eq_numpy'] for r in rows)} "
          f"extreme_cases={sum(r['extreme'] for r in rows)}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
