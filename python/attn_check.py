#!/usr/bin/env python3
"""Attention functional check (Phase 4 design note §8.3).

Random small attention programs, each run on the model and compared bit for
bit with the NumPy replica and the standalone C++ reference (attn_ref):
S, P, the INT8 row scales and O. Case i uses combination i mod 10 of
{prefill, decode} x {fp32, fp16, bf16, int8-row, int8-fixed} and draws its
hardware and shape from numpy.random.default_rng([SEED, CHECK_STREAM, i]):
grid 1x1 / 1x2 / 2x2, R in {4, 8}, C = 16, Kt in {16, 32}, d in {16, 32, 64},
L in {32, 64, 96}, T in {64, 128, 256}, g in {1, 2, 4}, V in {8, 16}; only
legal combinations (design note §4.2) are drawn.

    python3 python/attn_check.py --tools build/release --cases 80 [--csv out.csv]

Exit status 0 only if every case agrees all three ways.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn  # noqa: E402

SEED = 20261014
CHECK_STREAM = 6
COMBOS = [(mode, prec, ps) for mode in ("prefill", "decode")
          for prec, ps in (("fp32", "fixed"), ("fp16", "fixed"), ("bf16", "fixed"), ("int8", "row"), ("int8", "fixed"))]


def draw(i):
    rng = np.random.default_rng([SEED, CHECK_STREAM, i])
    mode, prec, ps = COMBOS[i % len(COMBOS)]
    while True:
        px, py = [(1, 1), (1, 2), (2, 2)][int(rng.integers(0, 3))]
        R = int(rng.choice([4, 8]))
        Kt = int(rng.choice([16, 32]))
        d = int(rng.choice([16, 32, 64]))
        L = int(rng.choice([32, 64, 96]))
        T = int(rng.choice([64, 128, 256]))
        g = int(rng.choice([1, 2, 4]))
        C = 16
        lcm_rck = R * C // math.gcd(R, C)
        lcm_rck = lcm_rck * Kt // math.gcd(lcm_rck, Kt)
        lcm_ck = C * Kt // math.gcd(C, Kt)
        ok = d % C == 0 and d % Kt == 0 and (L % lcm_rck == 0 if mode == "prefill" else T % lcm_ck == 0)
        if ok:
            break
    hw = dict(attn.HW_A4)
    hw.update(px=px, py=py, R=R, C=C, Kt=Kt,
              sram_x=int(rng.integers(0, px)), sram_y=int(rng.integers(0, py)),
              dram_w=int(rng.choice([16, 32])), t_miss=int(rng.choice([10, 30])), dram_q=8, gdma_m=8,
              sram_banks=8, link=16, V=int(rng.choice([8, 16])), vu_dma_m=8, vmem_banks=8)
    spec = dict(mode=mode, d=d, prec=prec, p_scale=ps, mapping="auto", seed=1000 + i)
    if mode == "prefill":
        spec["L"] = L
    else:
        spec.update(T=T, g=g)
    return spec, hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--cases", type=int, default=80)
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for i in range(args.cases):
        spec, hw = draw(i)
        out = attn.run_attention(spec, hw, args.tools, check=True, fp64=False)
        c = out["check"]
        row = dict(case=i, mode=spec["mode"], prec=spec["prec"], p_scale=spec["p_scale"], d=spec["d"],
                   L=spec.get("L", ""), T=spec.get("T", ""), g=spec.get("g", ""), px=hw["px"], py=hw["py"],
                   R=hw["R"], Kt=hw["Kt"], V=hw["V"], t_program=out["t_program"])
        row.update({k: int(v) for k, v in c.items()})
        rows.append(row)
        bad = [k for k, v in c.items() if not v]
        shape = f"L={spec['L']}" if spec["mode"] == "prefill" else f"T={spec['T']} g={spec['g']}"
        print(f"case {i}: {spec['mode']} {spec['prec']}/{spec['p_scale']} {shape} d={spec['d']} "
              f"grid={hw['px']}x{hw['py']} R={hw['R']} Kt={hw['Kt']} V={hw['V']} T_program={out['t_program']} "
              f"{'OK' if c['match'] else 'MISMATCH ' + ' '.join(bad)}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    ok = sum(r["match"] for r in rows)
    per = ", ".join(f"{m}/{p}/{s}={sum(r['match'] for r in rows if (r['mode'], r['prec'], r['p_scale']) == (m, p, s))}"
                    f"/{sum(1 for r in rows if (r['mode'], r['prec'], r['p_scale']) == (m, p, s))}"
                    for m, p, s in COMBOS)
    print(f"attention functional: seed={SEED} cases={len(rows)} all_three_agree={ok} "
          f"model_wrong={sum(not (r['s_model_eq_numpy'] and r['p_model_eq_numpy'] and r['o_model_eq_numpy'] and r['scales_model_eq_numpy']) for r in rows)} "
          f"cpp_ref_wrong={sum(not (r['s_cpp_eq_numpy'] and r['p_cpp_eq_numpy'] and r['o_cpp_eq_numpy'] and r['scales_cpp_eq_numpy']) for r in rows)}")
    print(f"per combination: {per}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
