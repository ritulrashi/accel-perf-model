#!/usr/bin/env python3
"""Phase 4 studies (design note §8.4-8.6, hypotheses §10).

Runs the 54 attention programs on the A4 reference configuration. Every op
must pass its in-model assertions (apm_op exits non-zero otherwise) and every
program must satisfy the program-level PE-stack sum (python/attn.py). The
study adds, for every op:
  * DRAM bytes == the formula count (python/roofline.py),
  * T_op >= each formula lower bound (compute, DRAM, NoC, VU),
  * GEMM achieved useful ops/cycle <= attainable;
and for every program a bit-exact three-way check (model, NumPy replica,
attn_ref) plus the FP64 error of O. Writes study.csv, three plots, and the
hypothesis numbers.

    python3 python/study_p4.py --tools build/release --out-dir DIR

Sequential on purpose (Phase 6 adds the parallel runner).
"""
import argparse
import csv
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn  # noqa: E402
import numerics as nx  # noqa: E402
from roofline import ceilings, gemm_bounds, vu_bounds  # noqa: E402

PRECS = ("int8", "fp16", "bf16", "fp32")
OPS = ("qk", "softmax", "pv")
# Reference categorical palette (dataviz skill; same order as Phase 3).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PREC_COLOR = {"int8": COLORS[0], "fp16": COLORS[1], "bf16": COLORS[2], "fp32": COLORS[3], "int8-fixed": COLORS[4]}
OP_COLOR = {"qk": COLORS[0], "softmax": COLORS[1], "pv": COLORS[2]}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def studies():
    out = []

    def add(group, **spec):
        spec.setdefault("d", 64)
        spec.setdefault("p_scale", "row")
        spec.setdefault("mapping", "auto")
        spec.setdefault("seed", 0)
        out.append(dict(group=group, V=spec.pop("V", attn.HW_A4["V"]), spec=spec))

    for L in (128, 256, 512, 1024):
        for prec in PRECS:
            add("prefill", mode="prefill", L=L, prec=prec)
    for T in (512, 1024, 2048, 4096, 8192):
        for prec in PRECS:
            add("decode", mode="decode", T=T, g=1, prec=prec)
    for g in (4, 8, 16):
        for prec in ("fp16", "int8"):
            add("gqa", mode="decode", T=4096, g=g, prec=prec)
    for prec in PRECS:
        add("physical", mode="decode", T=4096, g=1, prec=prec, mapping="physical")
    for v in (16, 32, 128):
        add("vu_lanes", mode="prefill", L=1024, prec="fp16", V=v)
    for T in (512, 1024, 2048, 4096, 8192):
        add("int8_fixed", mode="decode", T=T, g=1, prec="int8", p_scale="fixed")
    for i, s in enumerate(out):
        s["id"] = i
    return out


def op_formulas(prog, hw, op):
    """Formula DRAM bytes and lower bounds of one op (roofline.py; independent of the simulator)."""
    e = nx.ELEM[prog["spec"]["prec"]]
    if op["op"] == "gemm":
        cfg = dict(M=op["M"], N=op["N"], K=op["K"], R=hw["R"], C=hw["C"], Kt=hw["Kt"], px=hw["px"], py=hw["py"],
                   px_map=op["px_map"], py_map=op["py_map"], sram_x=hw["sram_x"], sram_y=hw["sram_y"],
                   W=hw["dram_w"], L=hw["link"], e=e, q=e, m_valid=op["m_valid"])
        b = gemm_bounds(cfg)
        c = ceilings(cfg)
        return dict(dram_bytes=b["dram_bytes"], bound=max(b["compute"], b["dram"], b["noc"]),
                    compute_bound=b["compute"], dram_bound=b["dram"], noc_bound=b["noc"],
                    attainable=c["attainable"], predicted=c["predicted"], ops=c["ops"])
    scale_bytes = -(-4 * op["rows"] // 16) * 16 if (prog["spec"]["prec"] == "int8" and op["p_scale"] == "row") else 0
    b = vu_bounds(op["rows"], op["n"], e, hw["V"], hw["t_vu"], hw["dram_w"], scale_bytes)
    return dict(dram_bytes=b["dram_bytes"], bound=max(b["compute"], b["dram"]), compute_bound=b["compute"],
                dram_bound=b["dram"], noc_bound=Fraction(0))


def run_one(s, tools, failures, hw=None):
    """One study program; hw (Phase 5 sweeps) overrides the A4 configuration."""
    hw = dict(attn.HW_A4 if hw is None else hw)
    hw["V"] = s["V"]
    spec = s["spec"]
    prog = attn.lower(spec, hw)
    out = attn.run_attention(spec, hw, tools, check=True, fp64=True)
    row = dict(id=s["id"], group=s["group"], mode=spec["mode"], L=spec.get("L", ""), T=spec.get("T", ""),
               g=spec.get("g", ""), d=spec["d"], prec=spec["prec"], p_scale=spec["p_scale"] if spec["prec"] == "int8" else "",
               mapping=(f"1x{hw['px'] * hw['py']}" if (spec["mode"] == "decode" and spec["mapping"] == "auto")
                        else f"{hw['px']}x{hw['py']}"), V=hw["V"],
               t_program=out["t_program"], useful_ops=out["useful_ops"], dram_bytes=out["dram_bytes"],
               achieved=f"{out['useful_ops'] / out['t_program']:.4f}",
               ai=f"{out['useful_ops'] / out['dram_bytes']:.4f}",
               bit_exact=int(out["check"]["match"]),
               rel_fro_err=f"{out['rel_fro_err']:.4e}", max_abs_err=f"{out['max_abs_err']:.4e}")
    bound_total = Fraction(0)
    for op, res in zip(prog["ops"], out["ops"]):
        f = op_formulas(prog, hw, op)
        name = res["name"]
        row[f"{name}_t"] = res["t_end"]
        row[f"{name}_bound"] = f"{float(f['bound']):.1f}"
        row[f"{name}_dram_bytes"] = res["dram_bytes"]
        # DRAM diagnostics: row misses and bus utilization (bus busy / T_op).
        row[f"{name}_row_hits"] = res["row_hits"]
        row[f"{name}_row_misses"] = res["row_misses"]
        row[f"{name}_bus_util"] = f"{res['bus_busy'] / res['t_end']:.4f}"
        bound_total += f["bound"]
        if res["dram_bytes"] != f["dram_bytes"]:
            failures.append(f"study {s['id']} {name}: DRAM bytes {res['dram_bytes']} != formula {f['dram_bytes']}")
        if res["t_end"] < f["bound"]:
            failures.append(f"study {s['id']} {name}: T {res['t_end']} < lower bound {float(f['bound'])}")
        if op["op"] == "gemm":
            if Fraction(res["useful_ops"], res["t_end"]) > f["attainable"]:
                failures.append(f"study {s['id']} {name}: achieved > attainable")
            row[f"{name}_predicted"] = f["predicted"]
            st = res["stalls"]
            for c in attn.GEMM_CATS:
                row[f"{name}_{c}"] = st[c]
            row[f"{name}_largest"] = max(attn.GEMM_CATS, key=lambda c: st[c])
        else:
            for c in attn.VU_CATS:
                row[f"{name}_{c}"] = res["stalls"][c]
            row["softmax_busy_frac"] = f"{res['stalls']['busy'] / res['t_end']:.4f}"
    row["bound_total"] = f"{float(bound_total):.1f}"
    for c, v in out["pe_stack"].items():
        row[f"pe_{c}"] = v
    if not out["check"]["match"]:
        failures.append(f"study {s['id']}: not bit-exact three ways: "
                        + " ".join(k for k, v in out["check"].items() if not v))
    return row


def find(rows, **kw):
    hits = [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(hits) == 1, (kw, len(hits))
    return hits[0]


def summarize(rows, failures, wall):
    print(f"study: programs={len(rows)} ops={3 * len(rows)} failures={len(failures)} "
          f"bit_exact={sum(r['bit_exact'] for r in rows)} wall_s={wall}")
    for f in failures:
        print("FAIL " + f)
    print("per program (cycles qk / softmax / pv = T_program; lower bound; error):")
    for r in rows:
        shape = f"L={r['L']}" if r["mode"] == "prefill" else f"T={r['T']} g={r['g']} map={r['mapping']}"
        print(f"  id={r['id']} {r['group']} {r['mode']} {shape} {r['prec']}{('/' + r['p_scale']) if r['p_scale'] else ''} "
              f"V={r['V']} qk={r['qk_t']} softmax={r['softmax_t']} pv={r['pv_t']} T_program={r['t_program']} "
              f"bound={r['bound_total']} rel_fro_err={r['rel_fro_err']} max_abs_err={r['max_abs_err']}")
        print("      DRAM per op (row_misses / accesses, bus_util): " + " ".join(
            f"{op}={r[op + '_row_misses']}/{r[op + '_row_hits'] + r[op + '_row_misses']},{r[op + '_bus_util']}" for op in OPS))

    print("H2 prefill L=1024 time ratios:")
    t = {p: find(rows, group="prefill", L=1024, prec=p)["t_program"] for p in PRECS}
    b = {p: float(find(rows, group="prefill", L=1024, prec=p)["bound_total"]) for p in PRECS}
    print(f"  measured fp16/int8={t['fp16'] / t['int8']:.4f} fp32/fp16={t['fp32'] / t['fp16']:.4f} "
          f"bf16/fp16={t['bf16'] / t['fp16']:.4f}")
    print(f"  bound    fp16/int8={b['fp16'] / b['int8']:.4f} fp32/fp16={b['fp32'] / b['fp16']:.4f}")
    print("H3 softmax share of prefill time at L=1024:")
    for p in PRECS:
        r = find(rows, group="prefill", L=1024, prec=p)
        print(f"  {p}: softmax={r['softmax_t']} T_program={r['t_program']} share={r['softmax_t'] / r['t_program']:.4f}")
    print("H4 decode T=4096 g=1 qk op stall stacks (PE-cycles):")
    for p in PRECS:
        r = find(rows, group="decode", T=4096, prec=p)
        cats = " ".join(f"{c}={r['qk_' + c]}" for c in attn.GEMM_CATS)
        print(f"  {p}: qk={r['qk_t']} bound={r['qk_bound']} predicted={r['qk_predicted']} largest={r['qk_largest']} {cats}")
    print("H5 GQA at T=4096: T(g)/T(1), per-query-head T(g)/(g*T(1)):")
    for p in ("fp16", "int8"):
        t1 = find(rows, group="decode", T=4096, prec=p)["t_program"]
        for g in (4, 8, 16):
            tg = find(rows, group="gqa", g=g, prec=p)["t_program"]
            print(f"  {p} g={g}: T={tg} ratio={tg / t1:.4f} per_head={tg / (g * t1):.4f}")
    print("H6 physical 2x2 vs logical 1x4 mapping, decode T=4096 g=1, qk op:")
    for p in PRECS:
        lo = find(rows, group="decode", T=4096, prec=p)
        ph = find(rows, group="physical", prec=p)
        print(f"  {p}: qk logical={lo['qk_t']} physical={ph['qk_t']} speedup={ph['qk_t'] / lo['qk_t']:.4f} "
              f"program speedup={ph['t_program'] / lo['t_program']:.4f}")
    print("H7 relative Frobenius error of O vs FP64:")
    for r in rows:
        if r["group"] in ("prefill", "decode", "int8_fixed"):
            shape = f"L={r['L']}" if r["mode"] == "prefill" else f"T={r['T']}"
            print(f"  {r['mode']} {shape} {r['prec']}{('/' + r['p_scale']) if r['p_scale'] else ''}: {r['rel_fro_err']}")
    for key, grp in (("L", "prefill"), ("T", "decode")):
        for v in sorted({r[key] for r in rows if r["group"] == grp}):
            f16 = float(find(rows, group=grp, **{key: v}, prec="fp16")["rel_fro_err"])
            b16 = float(find(rows, group=grp, **{key: v}, prec="bf16")["rel_fro_err"])
            print(f"  bf16/fp16 error ratio {grp} {key}={v}: {b16 / f16:.4f}")
    for T in (512, 1024, 2048, 4096, 8192):
        fx = float(find(rows, group="int8_fixed", T=T)["rel_fro_err"])
        rw = float(find(rows, group="decode", T=T, prec="int8")["rel_fro_err"])
        print(f"  int8 decode T={T}: fixed={fx:.4e} row={rw:.4e} fixed/row={fx / rw:.4f}")
    print("H8 VU busy fraction, prefill L=1024 fp16:")
    for v in (16, 32, 64, 128):
        r = find(rows, group="vu_lanes", V=v) if v != 64 else find(rows, group="prefill", L=1024, prec="fp16")
        print(f"  V={v}: softmax={r['softmax_t']} busy={r['softmax_busy']} busy_frac={r['softmax_busy_frac']}")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_breakdown(rows, path):
    """Per-op share of program time: prefill L=1024 and decode T=4096 (g=1), each precision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [find(rows, group="prefill", L=1024, prec=p) for p in PRECS] + \
          [find(rows, group="decode", T=4096, prec=p) for p in PRECS]
    labels = [f"prefill L=1024 {r['prec']}  ({r['t_program']} cyc)" for r in sel[:4]] + \
             [f"decode T=4096 {r['prec']}  ({r['t_program']} cyc)" for r in sel[4:]]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(False, axis="y")
    y = np.arange(len(sel))[::-1]
    left = np.zeros(len(sel))
    for op in OPS:
        frac = np.array([r[f"{op}_t"] / r["t_program"] for r in sel])
        ax.barh(y, frac, left=left, height=0.62, color=OP_COLOR[op], edgecolor=SURFACE, linewidth=2,
                label={"qk": "Q·Kᵀ (GEMM)", "softmax": "softmax (VU)", "pv": "P·V (GEMM)"}[op])
        left += frac
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of program cycles", color=INK2)
    ax.set_title("Where unfused attention spends its time (A4 config, d=64)", color=INK, loc="left", fontsize=11)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_errors(rows, path):
    """Relative error of O vs FP64 against decode cache length T, one line per precision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    Ts = [512, 1024, 2048, 4096, 8192]
    series = [(p, [float(find(rows, group="decode", T=T, prec=p)["rel_fro_err"]) for T in Ts]) for p in PRECS]
    series.append(("int8-fixed", [float(find(rows, group="int8_fixed", T=T)["rel_fro_err"]) for T in Ts]))
    for name, ys in series:
        label = {"int8": "INT8, per-row P scale", "int8-fixed": "INT8, fixed P scale 1/127"}.get(name, name.upper())
        ax.plot(Ts, ys, color=PREC_COLOR[name], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(label, (Ts[-1], ys[-1]), textcoords="offset points", xytext=(8, 0), fontsize=8, color=INK2,
                    va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.set_xlim(Ts[0] / 1.3, Ts[-1] * 3.5)
    ax.set_xlabel("decode KV-cache length T", color=INK2)
    ax.set_ylabel("relative Frobenius error of O vs FP64", color=INK2)
    ax.set_title("Decode numerics by precision (g=1, d=64, N(0,1) inputs)", color=INK, loc="left", fontsize=11)
    # The band between FP32 and FP16 errors is empty: the legend goes there.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=(0.0, 0.2))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_roofline(rows, path):
    """Program-level achieved useful ops/cycle vs arithmetic intensity, prefill vs decode, by precision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style(ax)
    w = attn.HW_A4["dram_w"]
    peak = 2 * attn.HW_A4["px"] * attn.HW_A4["py"] * attn.HW_A4["R"] * attn.HW_A4["C"]
    xs = np.geomspace(0.3, 200, 200)
    for p, q in (("int8", 1), ("fp16", 2), ("fp32", 4)):
        ax.plot(xs, np.minimum(peak / q, xs * w), color=PREC_COLOR[p], linewidth=1, linestyle="--",
                label=f"roof {p.upper()}: min({peak // q}, AI·W)")
    # BF16 has exactly the FP16 timing and traffic (same e and q), so its points
    # would hide FP16's: plot FP16 and say so in the legend.
    for p in ("int8", "fp16", "fp32"):
        for mode, marker in (("prefill", "o"), ("decode", "s")):
            pts = [(float(r["ai"]), float(r["achieved"])) for r in rows
                   if r["prec"] == p and r["mode"] == mode and r["group"] in ("prefill", "decode")]
            ax.scatter(*zip(*pts), s=40, marker=marker, color=PREC_COLOR[p], edgecolors=SURFACE, linewidths=1.5,
                       label=f"{mode} {p.upper()}" + (" (= BF16)" if p == "fp16" else ""), zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("program arithmetic intensity (useful ops / DRAM byte)", color=INK2)
    ax.set_ylabel("achieved useful ops / cycle", color=INK2)
    ax.set_title("Attention programs on the roofline (A4: W=64 B/cycle)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    rows, failures = [], []
    for s in studies():
        try:
            rows.append(run_one(s, args.tools, failures))
        except Exception as ex:  # noqa: BLE001 - every failure is reported, none hidden
            failures.append(f"study {s['id']}: {ex}")
    wall = int(time.time() - start)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.out_dir, "study.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if len(rows) == len(studies()):
        plot_breakdown(rows, os.path.join(args.out_dir, "attention_breakdown.png"))
        plot_errors(rows, os.path.join(args.out_dir, "numerics.png"))
        plot_roofline(rows, os.path.join(args.out_dir, "attention_roofline.png"))
        summarize(rows, failures, wall)
    else:
        print(f"study: programs={len(studies())} ran={len(rows)} failures={len(failures)}")
        for f in failures:
            print("FAIL " + f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
