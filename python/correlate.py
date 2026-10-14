#!/usr/bin/env python3
"""RTL correlation (Phase 5 design note §4).

Builds the correlation program set, has the model export every PE's feed
schedule (schedule=FILE), and runs the Verilator testbench os_array_tb on
each schedule. The testbench compares the RTL with the model at three levels
(tile timing, drained values, every cycle's per-cell MAC trace). The sets:

  hand    HT1, HT2: hand-written schedules; the RTL and contract_trace must
          also reproduce the hand-written traces
  p2dir   Phase 2 directed P2-T1..T5b (apm_gemm); tile-done cycles also
          checked against the Phase 2 hand derivations
  p2rand  the 200 Phase 2 random PE cases (python/gemm_check.py PE stream)
  p3rand  the 100 Phase 3 random grid cases (accel stream): every PE with work
  attn    A4 INT8 attention GEMM ops: prefill L=256, decode T=4096 g=8

    python3 python/correlate.py --tools build/release [--quick] [--csv out.csv]

--quick: hand, p2dir, 10 p2rand and 5 p3rand cases (the ctest subset).
Exit status 0 only if every schedule matches at every level.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn  # noqa: E402
import gemm_check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1")

# Phase 2 §5: P2 configuration and the hand-derived tile-done cycles.
P2 = dict(R=4, C=16, Kt=64, dram_w=32, t_hit=10, t_miss=10, dram_q=16, dma_m=8)
P2_DIRECTED = [
    ("P2-T1", dict(M=4, N=16, K=64, nbuf=2, nout=2), [138]),
    ("P2-T2", dict(M=4, N=16, K=256, nbuf=2, nout=2), [330]),
    ("P2-T3", dict(M=4, N=16, K=256, nbuf=1, nout=2), [486]),
    ("P2-T4", dict(M=4, N=16, K=256, nbuf=2, nout=2, dram_w=16), [418]),
    ("P2-T5", dict(M=8, N=16, K=64, nbuf=2, nout=2), [138, 224]),
    ("P2-T5b", dict(M=8, N=16, K=64, nbuf=2, nout=1), [138, 244]),
]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed ({p.returncode}): {p.stderr.strip()[-500:]}")
    return p.stdout


def tile_dones(sched_path):
    out = []
    with open(sched_path) as f:
        for line in f:
            if line.startswith("tile "):
                out.append(int(line.split()[3]))
    return out


def tb(tools, sched, expect=None):
    cmd = [os.path.join(tools, "os_array_tb"), sched] + (["--expect-trace", expect] if expect else [])
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    summary = next((l for l in p.stdout.splitlines() if l.startswith("os_array_tb:")), "")
    fields = dict(kv.split("=", 1) for kv in summary.split()[1:] if "=" in kv)
    return p.returncode, fields, p.stderr.strip()


def gemm_args(p, keys):
    return [f"{k}={p[k]}" for k in keys]


def schedules(args, work):
    """Yields (set, case, schedule path, expect-trace path or None, expected tile dones or None)."""
    for h in ("ht1", "ht2"):
        yield "hand", h, os.path.join(ROOT, "tests/rtl", h + ".sched"), os.path.join(ROOT, "tests/rtl", h + ".trace"), None
    a_path, b_path, c_path = (os.path.join(work, n) for n in ("a.bin", "b.bin", "c.bin"))
    for name, shape, dones in P2_DIRECTED:
        p = dict(P2, **shape)
        a, b = gemm_check.draw_operands(gemm_check.np.random.default_rng([7, len(name)]), p["M"], p["N"], p["K"])[1:]
        a.tofile(a_path)
        b.tofile(b_path)
        sched = os.path.join(work, f"{name}.sched")
        run([os.path.join(args.tools, "apm_gemm")] + gemm_args(p, ("M", "N", "K", "R", "C", "Kt", "nbuf", "nout",
                                                                    "dram_w", "t_hit", "t_miss", "dram_q", "dma_m"))
            + [f"a={a_path}", f"b={b_path}", f"out={c_path}", f"schedule={sched}"])
        yield "p2dir", name, sched, None, dones
    n2 = 10 if args.quick else 200
    for i in range(n2):
        p, a, b = gemm_check.draw_case_pe(i)
        a.tofile(a_path)
        b.tofile(b_path)
        sched = os.path.join(work, f"p2_{i}.sched")
        run([os.path.join(args.tools, "apm_gemm")] + gemm_args(p, ("M", "N", "K", "R", "C", "Kt", "nbuf", "nout"))
            + [f"a={a_path}", f"b={b_path}", f"out={c_path}", f"schedule={sched}"])
        yield "p2rand", str(i), sched, None, None
    n3 = 5 if args.quick else 100
    for i in range(n3):
        p, a, b = gemm_check.draw_case_accel(i)
        a.tofile(a_path)
        b.tofile(b_path)
        sched = os.path.join(work, f"p3_{i}.sched")
        run([os.path.join(args.tools, "apm_accel")]
            + gemm_args(p, ("M", "N", "K", "R", "C", "Kt", "px", "py", "sram_x", "sram_y", "D", "nbuf", "nout"))
            + [f"a={a_path}", f"b={b_path}", f"out={c_path}", f"schedule={sched}"])
        yield "p3rand", str(i), sched, None, None
    if not args.quick:
        for spec in (dict(mode="prefill", L=256, d=64, prec="int8", p_scale="row", mapping="auto", seed=0),
                     dict(mode="decode", T=4096, g=8, d=64, prec="int8", p_scale="row", mapping="auto", seed=0)):
            d = os.path.join(work, f"attn_{spec['mode']}")
            os.makedirs(d, exist_ok=True)
            attn.run_attention(spec, dict(attn.HW_A4), args.tools, check=False, fp64=False, schedule_dir=d)
            for op in ("qk", "pv"):
                yield "attn", f"{spec['mode']}_{op}", os.path.join(d, op + ".sched"), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as work:
        for set_, case, sched, expect, dones in schedules(args, work):
            rc, f, err = tb(args.tools, sched, expect)
            row = dict(set=set_, case=case, pes=f.get("pes", ""), tiles=f.get("tiles", ""), cycles=f.get("cycles", ""),
                       macs=f.get("macs", ""), trace_lines=f.get("trace_lines", ""), hand=f.get("hand", "n/a"),
                       result=f.get("result", "ERROR"), exit=rc)
            if dones is not None:
                got = tile_dones(sched)
                row["hand_dones"] = "ok" if got == dones else f"MISMATCH model {got} hand {dones}"
            else:
                row["hand_dones"] = "n/a"
            row["match"] = int(rc == 0 and row["result"] == "MATCH" and row["hand"] in ("ok", "n/a")
                               and row["hand_dones"] in ("ok", "n/a"))
            rows.append(row)
            print(f"{set_} {case}: pes={row['pes']} tiles={row['tiles']} cycles={row['cycles']} macs={row['macs']} "
                  f"hand_trace={row['hand']} hand_dones={row['hand_dones']} {row['result']}"
                  + ("" if row["match"] else f"  <-- {err}"))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    tot = lambda k: sum(int(r[k]) for r in rows if str(r[k]).isdigit())  # noqa: E731
    by_set = ", ".join(f"{s}={sum(r['match'] for r in rows if r['set'] == s)}/{sum(1 for r in rows if r['set'] == s)}"
                       for s in dict.fromkeys(r["set"] for r in rows))
    print(f"correlation: schedules={len(rows)} pe_schedules={tot('pes')} tiles={tot('tiles')} cycles={tot('cycles')} "
          f"macs={tot('macs')} trace_lines={tot('trace_lines')} all_match={sum(r['match'] for r in rows)}/{len(rows)}")
    print(f"correlation by set: {by_set}")
    return 0 if all(r["match"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
