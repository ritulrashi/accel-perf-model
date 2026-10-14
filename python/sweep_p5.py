#!/usr/bin/env python3
"""Phase 5 sweeps (design note §7, hypotheses §11).

S1  bank map: row vs xor on the Phase 4 workloads (prefill L in {256, 1024} x
    4 precisions, decode T=4096 g=1 x 4 precisions, GQA T=4096 g in {4, 8, 16}
    FP16) on the A4 configuration. Every row-map program must reproduce its
    Phase 4 study.csv cycles exactly (results/phase4/study.csv).
S2  equal-MAC arrangements (1024 MACs, xor map): 1x1 of 32x32, 2x2 of 16x16,
    4x4 of 8x8, SRAM at the corner; prefill L=1024 FP16 and INT8, decode
    T=4096 FP16 g=1 and INT8 g=8.

Every program runs through python/study_p4.run_one: in-model assertions,
DRAM bytes == formula, T_op >= formula lower bounds, achieved <= attainable,
bit-exact three ways, FP64 error. Writes sweep.csv, two plots and the
hypothesis numbers.

    python3 python/sweep_p5.py --tools build/release --out-dir DIR [--phase4-study results/phase4/study.csv]
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn  # noqa: E402
import runner  # noqa: E402
import study_p4  # noqa: E402

PRECS = ("int8", "fp16", "bf16", "fp32")
ARRANGEMENTS = [("1x1 of 32x32", dict(px=1, py=1, R=32, C=32)),
                ("2x2 of 16x16", dict(px=2, py=2, R=16, C=16)),
                ("4x4 of 8x8", dict(px=4, py=4, R=8, C=8))]
COLORS = study_p4.COLORS
SURFACE, INK, INK2, GRID = study_p4.SURFACE, study_p4.INK, study_p4.INK2, study_p4.GRID


def programs():
    out = []

    def add(sweep, label, hw_over, **spec):
        spec.setdefault("d", 64)
        spec.setdefault("p_scale", "row")
        spec.setdefault("mapping", "auto")
        spec.setdefault("seed", 0)
        hw = dict(attn.HW_A4)
        hw.update(hw_over)
        out.append(dict(sweep=sweep, arrangement=label, hw=hw, V=hw["V"], spec=spec,
                        group={"prefill": "prefill", "decode": "decode"}[spec["mode"]] if spec.get("g", 1) == 1 else "gqa"))

    for m in ("row", "xor"):
        for L in (256, 1024):
            for p in PRECS:
                add("S1", "2x2 of 16x16", dict(dram_map=m), mode="prefill", L=L, prec=p)
        for p in PRECS:
            add("S1", "2x2 of 16x16", dict(dram_map=m), mode="decode", T=4096, g=1, prec=p)
        for g in (4, 8, 16):
            add("S1", "2x2 of 16x16", dict(dram_map=m), mode="decode", T=4096, g=g, prec="fp16")
    for label, over in ARRANGEMENTS:
        o = dict(over, dram_map="xor")
        add("S2", label, o, mode="prefill", L=1024, prec="fp16")
        add("S2", label, o, mode="prefill", L=1024, prec="int8")
        add("S2", label, o, mode="decode", T=4096, g=1, prec="fp16")
        add("S2", label, o, mode="decode", T=4096, g=8, prec="int8")
    for i, s in enumerate(out):
        s["id"] = i
    return out


def phase4_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def phase4_match(row, p4):
    """The Phase 4 study row with the same workload (A4, row map)."""
    keys = ("mode", "L", "T", "g", "prec", "p_scale", "mapping", "V")
    hits = [r for r in p4 if all(str(r[k]) == str(row[k]) for k in keys) and r["group"] in ("prefill", "decode", "gqa")]
    return hits[0] if len(hits) == 1 else None


def find(rows, **kw):
    hits = [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(hits) == 1, (kw, len(hits))
    return hits[0]


def summarize(rows, failures, wall):
    print(f"sweep: programs={len(rows)} failures={len(failures)} bit_exact={sum(r['bit_exact'] for r in rows)} "
          f"phase4_reproduced={sum(1 for r in rows if r['phase4_cycles_equal'] == 1)}/"
          f"{sum(1 for r in rows if r['phase4_cycles_equal'] != '')} wall_s={wall}")
    for f in failures:
        print("FAIL " + f)
    print("per program (cycles qk / softmax / pv = T_program; formula lower bound):")
    for r in rows:
        shape = f"L={r['L']}" if r["mode"] == "prefill" else f"T={r['T']} g={r['g']}"
        print(f"  id={r['id']} {r['sweep']} map={r['dram_map']} {r['arrangement']} {r['mode']} {shape} "
              f"{r['prec']} qk={r['qk_t']} softmax={r['softmax_t']} pv={r['pv_t']} T_program={r['t_program']} "
              f"bound={r['bound_total']} rel_fro_err={r['rel_fro_err']}")
        print("      DRAM per op (row_misses / accesses, bus_util): " + " ".join(
            f"{op}={r[op + '_row_misses']}/{r[op + '_row_hits'] + r[op + '_row_misses']},{r[op + '_bus_util']}"
            for op in ("qk", "softmax", "pv"))
              + "  bounds qk/pv: " + f"{r['qk_bound']}/{r['pv_bound']}")
    s1 = [r for r in rows if r["sweep"] == "S1"]
    print("H2 xor vs row, prefill L=1024 INT8:")
    rr, rx = find(s1, dram_map="row", mode="prefill", L=1024, prec="int8"), find(s1, dram_map="xor", mode="prefill", L=1024, prec="int8")
    print(f"  qk row={rr['qk_t']} xor={rx['qk_t']} speedup={rr['qk_t'] / rx['qk_t']:.4f}; "
          f"program row={rr['t_program']} xor={rx['t_program']} speedup={rr['t_program'] / rx['t_program']:.4f}; "
          f"qk bus_util row={rr['qk_bus_util']} xor={rx['qk_bus_util']}")
    print("  all S1 workloads, xor speedup (row T / xor T):")
    for r in s1:
        if r["dram_map"] != "row":
            continue
        kw = dict(mode=r["mode"], L=r["L"], T=r["T"], g=r["g"], prec=r["prec"])
        x = find([q for q in s1 if q["dram_map"] == "xor"], **kw)
        shape = f"L={r['L']}" if r["mode"] == "prefill" else f"T={r['T']} g={r['g']}"
        print(f"    {r['mode']} {shape} {r['prec']}: row={r['t_program']} xor={x['t_program']} "
              f"speedup={r['t_program'] / x['t_program']:.4f}")
    print("H3 xor map, prefill L=1024 time ratios:")
    t = {p: find(s1, dram_map="xor", mode="prefill", L=1024, prec=p)["t_program"] for p in PRECS}
    b = {p: float(find(s1, dram_map="xor", mode="prefill", L=1024, prec=p)["bound_total"]) for p in PRECS}
    print(f"  measured fp16/int8={t['fp16'] / t['int8']:.4f} fp32/fp16={t['fp32'] / t['fp16']:.4f} "
          f"bf16/fp16={t['bf16'] / t['fp16']:.4f}")
    print(f"  bound    fp16/int8={b['fp16'] / b['int8']:.4f} fp32/fp16={b['fp32'] / b['fp16']:.4f}")
    print("H4 decode T=4096 g=1 xor/row, and GQA FP16 T(g)/T(1) per map:")
    for p in PRECS:
        r = find(s1, dram_map="row", mode="decode", T=4096, g=1, prec=p)
        x = find(s1, dram_map="xor", mode="decode", T=4096, g=1, prec=p)
        print(f"  {p}: row={r['t_program']} xor={x['t_program']} xor/row={x['t_program'] / r['t_program']:.4f}")
    for m in ("row", "xor"):
        t1 = find(s1, dram_map=m, mode="decode", T=4096, g=1, prec="fp16")["t_program"]
        for g in (4, 8, 16):
            tg = find(s1, dram_map=m, mode="decode", T=4096, g=g, prec="fp16")["t_program"]
            print(f"  {m} g={g}: T={tg} ratio={tg / t1:.4f} per_head={tg / (g * t1):.4f}")
    s2 = [r for r in rows if r["sweep"] == "S2"]
    for hyp, mode, extra in (("H5", "prefill", (("fp16", 1), ("int8", 1))), ("H6", "decode", (("fp16", 1), ("int8", 8)))):
        print(f"{hyp} equal-MAC arrangements, {mode}:")
        for prec, g in extra:
            sel = [r for r in s2 if r["mode"] == mode and r["prec"] == prec and int(r["g"] or 1) == g]
            best = min(r["t_program"] for r in sel)
            for r in sel:
                print(f"  {prec} g={g} {r['arrangement']}: T={r['t_program']} vs_best={r['t_program'] / best:.4f} "
                      f"qk={r['qk_t']} pv={r['pv_t']} softmax={r['softmax_t']} largest_qk={r['qk_largest']} "
                      f"largest_pv={r['pv_largest']}")


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_maps(rows, path):
    """Speedup of the xor map over the row map, per S1 workload (one axis, one series)."""
    plt = _plot_setup()
    s1 = [r for r in rows if r["sweep"] == "S1"]
    pairs = []
    for r in s1:
        if r["dram_map"] == "row":
            x = find([q for q in s1 if q["dram_map"] == "xor"], mode=r["mode"], L=r["L"], T=r["T"], g=r["g"], prec=r["prec"])
            shape = f"L={r['L']}" if r["mode"] == "prefill" else (f"T={r['T']}" + (f" g={r['g']}" if int(r["g"]) > 1 else ""))
            pairs.append((f"{r['mode']} {shape} {r['prec']}", r["t_program"] / x["t_program"]))
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(pairs) + 1.4), facecolor=SURFACE)
    study_p4._style(ax)
    ax.grid(False, axis="y")
    y = list(range(len(pairs)))[::-1]
    ax.barh(y, [v for _, v in pairs], height=0.6, color=COLORS[0], edgecolor=SURFACE, linewidth=2)
    for yi, (_, v) in zip(y, pairs):
        ax.annotate(f"{v:.2f}x", (v, yi), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8, color=INK2)
    ax.axvline(1.0, color=INK2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in pairs], fontsize=8, color=INK)
    ax.set_xlabel("program speedup of the xor bank map (row-map cycles / xor-map cycles)", color=INK2)
    ax.set_title("S1: XOR bank-address map vs row map (A4 config)", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_arrangements(rows, path):
    """Program cycles relative to the best arrangement, per S2 workload."""
    plt = _plot_setup()
    s2 = [r for r in rows if r["sweep"] == "S2"]
    workloads = []
    for r in s2:
        key = (r["mode"], r["prec"], r["g"])
        if key not in workloads:
            workloads.append(key)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    study_p4._style(ax)
    ax.grid(False, axis="x")
    width = 0.26
    for a, (label, _) in enumerate(ARRANGEMENTS):
        vals = []
        for mode, prec, g in workloads:
            sel = [r for r in s2 if (r["mode"], r["prec"], r["g"]) == (mode, prec, g)]
            best = min(r["t_program"] for r in sel)
            vals.append(next(r for r in sel if r["arrangement"] == label)["t_program"] / best)
        xs = [i + (a - 1) * width for i in range(len(workloads))]
        ax.bar(xs, vals, width=width - 0.02, color=COLORS[a], edgecolor=SURFACE, linewidth=2, label=label)
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels([f"{m} {p}" + (f" g={g}" if m == "decode" else "") for m, p, g in workloads], fontsize=9, color=INK)
    ax.set_ylabel("program cycles / best arrangement's", color=INK2)
    ax.set_ylim(0, None)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_title("S2: equal-MAC arrangements (1024 MACs, xor map, SRAM at the corner)", color=INK, loc="left",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--phase4-study", default="results/phase4/study.csv")
    runner.add_jobs_arg(ap)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    p4 = phase4_rows(args.phase4_study)
    start = time.time()
    rows, failures = [], []
    progs = programs()
    results = runner.run_tasks([(dict(s), args.tools, s["hw"]) for s in progs], study_p4.run_one_task, args.jobs)
    for s, (_, res, exc) in zip(progs, results):
        if exc is not None:
            failures.append(f"program {s['id']}: {exc}")
            continue
        row, fails = res
        failures += fails
        row.update(id=s["id"], sweep=s["sweep"], arrangement=s["arrangement"], dram_map=s["hw"]["dram_map"],
                   group=s["group"])
        row["phase4_cycles_equal"] = ""
        if s["hw"]["dram_map"] == "row":
            m = phase4_match(row, p4)
            if m is None:
                failures.append(f"program {s['id']}: no Phase 4 study row to compare with")
            else:
                same = all(int(m[k]) == int(row[k]) for k in ("t_program", "qk_t", "softmax_t", "pv_t"))
                row["phase4_cycles_equal"] = int(same)
                if not same:
                    failures.append(f"program {s['id']}: row-map cycles differ from Phase 4 study id {m['id']}")
        rows.append(row)
    wall = int(time.time() - start)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.out_dir, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if len(rows) == len(progs):
        plot_maps(rows, os.path.join(args.out_dir, "bank_map.png"))
        plot_arrangements(rows, os.path.join(args.out_dir, "arrangements.png"))
        summarize(rows, failures, wall)
    else:
        print(f"sweep: programs={len(progs)} ran={len(rows)} failures={len(failures)}")
        for f in failures:
            print("FAIL " + f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
