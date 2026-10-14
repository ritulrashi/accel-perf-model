#!/usr/bin/env python3
"""Phase 3 sweep (design note §6.3-6.5): runs a fixed list of >= 100
accelerator configurations, each of which must pass the in-model stall-stack
assertions (apm_accel exits non-zero otherwise), and checks every run against
the formula-only roofline (python/roofline.py):
  * DRAM bytes moved == the formula count,
  * achieved ops/cycle <= attainable (exact rational comparison),
  * the simulator's busiest mesh link was busy >= B_max / L cycles.
Writes sweep.csv (the table view), roofline.png and stall_stacks.png, and
prints a summary including the hypothesis numbers (design note §8).

    python3 python/sweep_p3.py --apm-accel build/release/apm_accel --out-dir DIR

Sequential on purpose (Phase 6 adds the parallel runner). The configuration
list is deterministic: fixed tables plus numpy default_rng([SWEEP_SEED, i]).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roofline import ceilings  # noqa: E402

SWEEP_SEED = 20261014
RANDOM_CONFIGS = 104
CATS = ["compute", "dram", "sram_conflict", "network", "dma", "idle"]
# Reference categorical palette (dataviz skill), fixed order = CATS order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REGION_COLOR = {"compute": COLORS[0], "dram": COLORS[1], "noc": COLORS[3]}
REGION_MARKER = {"compute": "o", "dram": "s", "noc": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# Memory system wide enough that the DRAM ceiling is reachable in principle:
# with t_miss=30 and D=1 a DMA needs ~33 bursts in flight to saturate W=64.
ROOF_SYSTEM = dict(sram_banks=16, gdma_m=32, dram_q=32, t_hit=10, t_miss=30, hop=2, D=2, nbuf=2, nout=2)


def table41():
    """Design note §4.1 worked examples: g=2, square arrays, M=N=2ag, K=1024."""
    out = []
    for a, w, l in [(4, 64, 64), (8, 16, 64), (8, 64, 64), (16, 64, 16), (16, 64, 64)]:
        out.append(dict(group="table41", M=4 * a, N=4 * a, K=1024, R=a, C=a, Kt=64, px=2, py=2,
                        sram_x=0, sram_y=0, W=w, L=l, **ROOF_SYSTEM))
    return out


def h5_pairs():
    """H5: same NoC-limited config with sram_node at the corner vs the centre of a 3x3 grid."""
    out = []
    for a in (4, 8, 16):
        for sx, sy, where in ((0, 0, "corner"), (1, 1, "centre")):
            out.append(dict(group=f"h5_{where}", M=6 * a, N=6 * a, K=1024, R=a, C=a, Kt=64, px=3, py=3,
                            sram_x=sx, sram_y=sy, W=64, L=16, **ROOF_SYSTEM))
    return out


def random_configs():
    out = []
    for i in range(RANDOM_CONFIGS):
        rng = np.random.default_rng([SWEEP_SEED, i])
        px, py = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 3)][int(rng.integers(0, 6))]
        r, c = [(4, 4), (4, 16), (8, 8), (16, 16)][int(rng.integers(0, 4))]
        kt = int(rng.choice([32, 64]))
        tiles_m = px * int(rng.integers(1, 3))
        tiles_n = py * int(rng.integers(1, 3))
        if rng.random() < 0.25 and tiles_m > 1:  # partial waves
            tiles_m -= 1
        if rng.random() < 0.25 and tiles_n > 1:
            tiles_n -= 1
        cfg = dict(group="random", M=r * tiles_m, N=c * tiles_n, K=kt * int(rng.choice([2, 4, 8, 16])),
                   R=r, C=c, Kt=kt, px=px, py=py,
                   sram_x=int(rng.integers(0, px)), sram_y=int(rng.integers(0, py)),
                   W=int(rng.choice([16, 32, 64])), L=int(rng.choice([16, 32, 64])),
                   hop=int(rng.choice([1, 2, 3])), sram_banks=int(rng.choice([8, 16])),
                   D=int(rng.choice([1, 2, 3])), nbuf=int(rng.choice([1, 2])), nout=int(rng.choice([1, 2])),
                   gdma_m=int(rng.choice([8, 32])), dram_q=int(rng.choice([8, 32])),
                   t_hit=10, t_miss=int(rng.choice([10, 30])))
        out.append(cfg)
    return out


def all_configs():
    cfgs = table41() + h5_pairs() + random_configs()
    for i, c in enumerate(cfgs):
        c["id"] = i
    return cfgs


ARG_KEYS = ["M", "N", "K", "R", "C", "Kt", "px", "py", "sram_x", "sram_y", "D", "nbuf", "nout",
            "hop", "sram_banks", "gdma_m", "dram_q", "t_hit", "t_miss"]


def run(apm_accel, cfg):
    args = [apm_accel] + [f"{k}={cfg[k]}" for k in ARG_KEYS]
    args += [f"dram_w={cfg['W']}", f"link={cfg['L']}", f"seed={cfg['id'] + 1}"]
    env = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1")
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apm-accel", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, failures = [], []
    for cfg in all_configs():
        rc, out, err = run(args.apm_accel, cfg)
        if rc != 0:
            failures.append(f"config {cfg['id']}: apm_accel exit {rc}: {err}")
            continue
        v = json.loads(out)["values"]
        e = ceilings(cfg)
        t_end = v["t_end"]
        achieved = Fraction(v["ops"], t_end)
        row = dict(cfg)
        row.update(t_end=t_end, ops=v["ops"], pes=v["pes"], dram_bytes=v["dram_bytes"],
                   dram_bytes_formula=e["dram_bytes"], full_waves=int(e["full_waves"]),
                   achieved=f"{float(achieved):.4f}", peak=f"{float(e['peak']):.4f}",
                   dram_ceil=f"{float(e['dram_ceil']):.4f}",
                   noc_ceil="inf" if e["noc_ceil"] is None else f"{float(e['noc_ceil']):.4f}",
                   attainable=f"{float(e['attainable']):.4f}", predicted=e["predicted"],
                   eff_vs_attainable=f"{float(achieved / e['attainable']):.4f}",
                   eff_vs_peak=f"{float(achieved / e['peak']):.4f}",
                   eff_vs_dram_ceil=f"{float(achieved / e['dram_ceil']):.4f}")
        total = v["pes"] * t_end
        for cat in CATS:
            row[f"stall_{cat}"] = v[f"stall.{cat}"]
        row["stall_sum"] = sum(v[f"stall.{cat}"] for cat in CATS)
        row["stall_sum_ok"] = int(row["stall_sum"] == total)
        noncompute = [c for c in CATS if c != "compute"]
        row["largest"] = max(CATS, key=lambda c: v[f"stall.{c}"])
        row["dominant_noncompute"] = max(noncompute, key=lambda c: v[f"stall.{c}"])
        if v["dram_bytes"] != e["dram_bytes"]:
            failures.append(f"config {cfg['id']}: DRAM bytes {v['dram_bytes']} != formula {e['dram_bytes']}")
        # Independent check of the NoC formula against the simulator: the
        # busiest link must have been busy at least B_max / L cycles.
        row["max_link_busy"] = v["mesh.max_link_busy"]
        row["noc_link_bytes"] = e["noc_link_bytes"]
        if v["mesh.max_link_busy"] * cfg["L"] < e["noc_link_bytes"]:
            failures.append(f"config {cfg['id']}: busiest link busy {v['mesh.max_link_busy']} cycles < "
                            f"formula load {e['noc_link_bytes']} B / L {cfg['L']}")
        if achieved > e["attainable"]:
            failures.append(f"config {cfg['id']}: achieved {float(achieved)} > attainable {float(e['attainable'])}")
        if not row["stall_sum_ok"]:
            failures.append(f"config {cfg['id']}: stall sum {row['stall_sum']} != P*T_end {total}")
        rows.append(row)

    with open(os.path.join(args.out_dir, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    plot_roofline(rows, os.path.join(args.out_dir, "roofline.png"))
    plot_stacks([r for r in rows if r["group"] != "random"], os.path.join(args.out_dir, "stall_stacks.png"))
    summarize(rows, failures)
    return 0 if not failures else 1


def summarize(rows, failures):
    pe_cycles = sum(r["pes"] * r["t_end"] for r in rows)
    print(f"sweep: configs={len(rows) + len([f for f in failures if 'apm_accel exit' in f])} "
          f"ran={len(rows)} failures={len(failures)} pe_cycles_classified={pe_cycles} "
          f"stall_sum_ok={sum(r['stall_sum_ok'] for r in rows)}")
    for f in failures:
        print("FAIL " + f)
    by_region = {}
    for r in rows:
        by_region.setdefault(r["predicted"], []).append(r)
    print("predicted regions: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_region.items())))
    # H2: compute-bound -> compute is the largest category; dram -> dram dominates the
    # non-compute stalls; noc -> network dominates them.
    expect = {"compute": ("largest", "compute"), "dram": ("dominant_noncompute", "dram"),
              "noc": ("dominant_noncompute", "network")}
    for region, rs in sorted(by_region.items()):
        col, want = expect[region]
        hit = sum(r[col] == want for r in rs)
        others = {}
        for r in rs:
            if r[col] != want:
                others[r[col]] = others.get(r[col], 0) + 1
        print(f"H2 {region}: {hit} of {len(rs)} have {col}={want}; others {others}")
    print("table41 / h5 rows:")
    for r in rows:
        if r["group"] == "random":
            continue
        print(f"  id={r['id']} {r['group']} a={r['R']} grid={r['px']}x{r['py']} sram=({r['sram_x']},{r['sram_y']}) "
              f"W={r['W']} L={r['L']} t_end={r['t_end']} achieved={r['achieved']} peak={r['peak']} "
              f"dram_ceil={r['dram_ceil']} noc_ceil={r['noc_ceil']} predicted={r['predicted']} "
              f"eff_vs_attainable={r['eff_vs_attainable']} eff_vs_peak={r['eff_vs_peak']} "
              f"eff_vs_dram_ceil={r['eff_vs_dram_ceil']} largest={r['largest']} "
              f"dominant_noncompute={r['dominant_noncompute']}")
    effs = sorted(float(r["eff_vs_attainable"]) for r in rows)
    print(f"efficiency vs attainable over all runs: min={effs[0]:.4f} median={effs[len(effs) // 2]:.4f} "
          f"max={effs[-1]:.4f}")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roofline(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    _style(ax)
    # Normalized roofline: x = DRAM ceiling / peak, y = achieved / peak.
    # The roof is min(1, x); NoC-limited configs sit below it by design.
    xs = [float(r["dram_ceil"]) / float(r["peak"]) for r in rows]
    lo, hi = min(xs + [0.1]) / 1.5, max(xs + [10]) * 1.5
    grid = np.geomspace(lo, hi, 200)
    ax.plot(grid, np.minimum(1, grid), color=INK, linewidth=2, label="roof: min(peak, DRAM ceiling)")
    for region in ("compute", "dram", "noc"):
        pts = [(float(r["dram_ceil"]) / float(r["peak"]), float(r["achieved"]) / float(r["peak"]))
               for r in rows if r["predicted"] == region]
        if pts:
            ax.scatter(*zip(*pts), s=36, marker=REGION_MARKER[region], color=REGION_COLOR[region],
                       edgecolors=SURFACE, linewidths=1.5, label=f"predicted {region}-bound ({len(pts)})",
                       zorder=3)
    for r in rows:
        if r["group"] == "table41":
            ax.annotate(f"a={r['R']} W={r['W']} L={r['L']}",
                        (float(r["dram_ceil"]) / float(r["peak"]), float(r["achieved"]) / float(r["peak"])),
                        textcoords="offset points", xytext=(8, 8), fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DRAM ceiling / peak  (= AI_dram x W / peak)", color=INK2)
    ax.set_ylabel("achieved / peak", color=INK2)
    ax.set_title(f"Phase 3 roofline, {len(rows)} configurations (normalized per config)", color=INK,
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_stacks(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    labels = [f"#{r['id']} {r['group']} a={r['R']} g={r['px']}x{r['py']} W={r['W']} L={r['L']} -> {r['predicted']}"
              for r in rows]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for cat, color in zip(CATS, COLORS):
        frac = np.array([r[f"stall_{cat}"] / (r["pes"] * r["t_end"]) for r in rows])
        ax.barh(y, frac, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=cat)
        left += frac
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of PE-cycles", color=INK2)
    ax.set_title("Stall stacks: design-note §4.1 configs and H5 corner/centre pairs", color=INK,
                 loc="left", fontsize=11)
    ax.legend(ncol=6, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08 - 0.4 / len(rows)),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
