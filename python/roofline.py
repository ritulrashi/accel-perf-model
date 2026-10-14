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

Phase 4 (design note §6) adds optional keys, absent = Phase 3 behaviour:
  e (operand bytes), q (cycles per reduction step), m_valid (valid rows;
  ops count useful rows only, C bytes valid rows only), px_map/py_map
  (logical mapping; PE id x'*py_map + y' sits at physical node
  (id // py, id % py)). Peak = 2*P*R*C/q. It also adds the vector unit's two
  ceilings and per-op lower bounds on cycles (bounds()), used by the Phase 4
  studies: no run can finish faster than its compute, DRAM, NoC or VU time.
"""
from collections import defaultdict
from fractions import Fraction


def _ceil_div(a, b):
    return -(-a // b)


def plan(cfg):
    """Mapping counts for one configuration (design note §2.1-2.2; Phase 4 §5)."""
    m, n, k = cfg["M"], cfg["N"], cfg["K"]
    r, c, kt = cfg["R"], cfg["C"], cfg["Kt"]
    px, py = cfg["px"], cfg["py"]
    mx, my = cfg.get("px_map") or px, cfg.get("py_map") or py
    e = cfg.get("e", 1)
    m_valid = cfg.get("m_valid") or m
    tiles_m, tiles_n, nc = m // r, n // c, k // kt
    waves_m, waves_n = _ceil_div(tiles_m, mx), _ceil_div(tiles_n, my)
    staged = 0
    tiles_of = defaultdict(int)  # physical (x, y) -> tiles
    for bm in range(waves_m):
        for bn in range(waves_n):
            rows = min(mx, tiles_m - bm * mx)
            cols = min(my, tiles_n - bn * my)
            staged += nc * (rows * r * kt + cols * kt * c) * e
            for x in range(rows):
                for y in range(cols):
                    pe = x * my + y
                    tiles_of[(pe // py, pe % py)] += 1
    return {
        "tiles_of": dict(tiles_of),
        "dram_bytes": staged + 4 * m_valid * n,
        "full_waves": tiles_m % mx == 0 and tiles_n % my == 0,
    }


def busiest_link_load_bytes(cfg, tiles_of):
    """Max operand-load bytes on one directed link, XY routing from sram_node."""
    sx, sy = cfg["sram_x"], cfg["sram_y"]
    per_tile = cfg["K"] * (cfg["R"] + cfg["C"]) * cfg.get("e", 1)  # every chunk loads (R*Kt + Kt*C)*e
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
    ops = 2 * (cfg.get("m_valid") or cfg["M"]) * cfg["N"] * cfg["K"]
    peak = Fraction(2 * cfg["px"] * cfg["py"] * cfg["R"] * cfg["C"], cfg.get("q", 1))
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


def gemm_bounds(cfg):
    """Lower bounds on a GEMM op's cycles (Phase 4 §6, §10): the busiest PE's
    compute (its tiles x (nc*q*Kt + F)), DRAM bytes / W, busiest-link bytes / L."""
    p = plan(cfg)
    nc = cfg["K"] // cfg["Kt"]
    f = 2 * cfg["R"] + cfg["C"] - 2
    compute = max(p["tiles_of"].values()) * (nc * cfg.get("q", 1) * cfg["Kt"] + f)
    b_max = busiest_link_load_bytes(cfg, p["tiles_of"])
    return {"compute": Fraction(compute), "dram": Fraction(p["dram_bytes"], cfg["W"]),
            "noc": Fraction(b_max, cfg["L"]), "dram_bytes": p["dram_bytes"]}


def vu_row_cycles(n, lanes, t_vu):
    return 3 * (_ceil_div(n, lanes) + t_vu)


def vu_bounds(rows, n, e, lanes, t_vu, w, scale_bytes=0):
    """Lower bounds on a VU op's cycles: rows * T_row, and DRAM bytes / W."""
    dram_bytes = rows * n * (4 + e) + scale_bytes
    return {"compute": Fraction(rows * vu_row_cycles(n, lanes, t_vu)), "dram": Fraction(dram_bytes, w),
            "dram_bytes": dram_bytes}


def vu_ceilings(n, e, lanes, t_vu, w):
    """VU ceilings in elements/cycle: the unit (n / T_row) and DRAM (W / (4 + e))."""
    vu = Fraction(n, vu_row_cycles(n, lanes, t_vu))
    dram = Fraction(w, 4 + e)
    return {"vu_ceil": vu, "dram_ceil": dram, "attainable": min(vu, dram), "predicted": "vu" if vu < dram else "dram"}


if __name__ == "__main__":
    # The five worked examples of design note §4.1 (g=2, Kt=64, K=1024).
    for a, w, l in [(4, 64, 64), (8, 16, 64), (8, 64, 64), (16, 64, 16), (16, 64, 64)]:
        cfg = dict(M=4 * a, N=4 * a, K=1024, R=a, C=a, Kt=64, px=2, py=2, sram_x=0, sram_y=0, W=w, L=l)
        e = ceilings(cfg)
        print(f"a={a} W={w} L={l}: AI={float(e['ai_dram']):.3f} peak={float(e['peak']):.1f} "
              f"dram={float(e['dram_ceil']):.1f} noc={float(e['noc_ceil']):.1f} -> {e['predicted']}")
