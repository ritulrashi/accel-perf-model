#!/usr/bin/env python3
"""Hierarchical roofline from formulas only (Phase 3 design note §4).

Nothing here reads simulator output: every quantity is computed from the
configuration with the mapping and staging rules of design note §2, so it is
an independent hand calculation of what the model should be able to reach.

    ops         = 2*M*N*K
    peak        = 2 * Px*Py * R*C                       (ops/cycle)
    DRAM bytes  = staged A/B blocks (with wave dedup) + 4*M*N
    DRAM ceil   = ops * W / DRAM bytes                  (= AI_dram * W)
    NoC ceil    = ops * L / B_max, B_max = load bytes on the busiest mesh
                  link (XY routes from sram_node to every PE; a PE at
                  sram_node uses no link)
    attainable  = min(peak, DRAM ceil, NoC ceil); predicted region = argmin
                  (ties resolved in the order compute, dram, noc)

All arithmetic is exact (fractions.Fraction).
"""
from collections import defaultdict
from fractions import Fraction


def _ceil_div(a, b):
    return -(-a // b)


def plan(cfg):
    """Mapping counts for one configuration (design note §2.1-2.2)."""
    m, n, k = cfg["M"], cfg["N"], cfg["K"]
    r, c, kt = cfg["R"], cfg["C"], cfg["Kt"]
    px, py = cfg["px"], cfg["py"]
    tiles_m, tiles_n, nc = m // r, n // c, k // kt
    waves_m, waves_n = _ceil_div(tiles_m, px), _ceil_div(tiles_n, py)
    staged = 0
    tiles_of = defaultdict(int)  # (x, y) -> tiles
    for bm in range(waves_m):
        for bn in range(waves_n):
            rows = min(px, tiles_m - bm * px)
            cols = min(py, tiles_n - bn * py)
            staged += nc * (rows * r * kt + cols * kt * c)
            for x in range(rows):
                for y in range(cols):
                    tiles_of[(x, y)] += 1
    return {
        "tiles_of": dict(tiles_of),
        "dram_bytes": staged + 4 * m * n,
        "full_waves": tiles_m % px == 0 and tiles_n % py == 0,
    }


def busiest_link_load_bytes(cfg, tiles_of):
    """Max operand-load bytes on one directed link, XY routing from sram_node."""
    sx, sy = cfg["sram_x"], cfg["sram_y"]
    per_tile = cfg["K"] * (cfg["R"] + cfg["C"])  # every chunk loads R*Kt + Kt*C
    load = defaultdict(int)
    for (x, y), tiles in tiles_of.items():
        bytes_ = tiles * per_tile
        cx, cy = sx, sy
        while cx != x:  # x first
            nx = cx + (1 if x > cx else -1)
            load[((cx, cy), (nx, cy))] += bytes_
            cx = nx
        while cy != y:  # then y
            ny = cy + (1 if y > cy else -1)
            load[((cx, cy), (cx, ny))] += bytes_
            cy = ny
    return max(load.values()) if load else 0


def ceilings(cfg):
    p = plan(cfg)
    ops = 2 * cfg["M"] * cfg["N"] * cfg["K"]
    peak = Fraction(2 * cfg["px"] * cfg["py"] * cfg["R"] * cfg["C"])
    dram = Fraction(ops * cfg["W"], p["dram_bytes"])
    b_max = busiest_link_load_bytes(cfg, p["tiles_of"])
    noc = Fraction(ops * cfg["L"], b_max) if b_max else None
    candidates = [("compute", peak), ("dram", dram)] + ([("noc", noc)] if noc is not None else [])
    region, attainable = min(candidates, key=lambda kv: kv[1])  # min keeps the first on ties
    return {
        "ops": ops,
        "dram_bytes": p["dram_bytes"],
        "full_waves": p["full_waves"],
        "ai_dram": Fraction(ops, p["dram_bytes"]),
        "peak": peak,
        "dram_ceil": dram,
        "noc_ceil": noc,
        "noc_link_bytes": b_max,
        "attainable": attainable,
        "predicted": region,
    }


if __name__ == "__main__":
    # The five worked examples of design note §4.1 (g=2, Kt=64, K=1024).
    for a, w, l in [(4, 64, 64), (8, 16, 64), (8, 64, 64), (16, 64, 16), (16, 64, 64)]:
        cfg = dict(M=4 * a, N=4 * a, K=1024, R=a, C=a, Kt=64, px=2, py=2, sram_x=0, sram_y=0, W=w, L=l)
        e = ceilings(cfg)
        print(f"a={a} W={w} L={l}: AI={float(e['ai_dram']):.3f} peak={float(e['peak']):.1f} "
              f"dram={float(e['dram_ceil']):.1f} noc={float(e['noc_ceil']):.1f} -> {e['predicted']}")
