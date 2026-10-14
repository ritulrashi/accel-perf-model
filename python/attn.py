#!/usr/bin/env python3
"""Attention front end (Phase 4 design note §4).

Lowers one attention workload (one KV head with g query heads) to an unfused
three-op program, runs each op on the model as its own simulation (apm_op,
DRAM image handed from op to op), and optionally checks the result:

    op 1  GEMM  S = Q K^T     A = Q (packed A), B = K^T (packed B; the KV-cache layout)
    op 2  VU    P = softmax(S, mask, alpha)   P written in the packed A layout of op 3
    op 3  GEMM  O = P V       B = V (packed B; the KV-cache layout)

    prefill: Q is L x d, K/V are L x d, causal mask, M = L
    decode:  Q is g x d (padded to R rows, m_valid = g), K/V are the T x d cache, no mask

With --check, the model's S, P, row scales and O are compared bit for bit
with the NumPy replica (python/numerics.py) and the standalone C++ reference
(attn_ref), and O is compared with an FP64 reference on the unquantized
inputs (relative Frobenius and max abs error).

    python3 python/attn.py --mode prefill --L 256 --d 64 --prec fp16 --tools build/release --check
    python3 python/attn.py --mode decode --T 4096 --d 64 --g 8 --prec int8 --p-scale row --tools build/release
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numerics as nx  # noqa: E402

SEED = 20261014
ATTN_STREAM = 4

# The A4 reference configuration (design note §8.4).
HW_A4 = dict(px=2, py=2, R=16, C=16, Kt=64, nbuf=2, nout=2, D=2, sram_x=0, sram_y=0,
             dram_w=64, t_hit=10, t_miss=30, dram_q=32, dram_banks=4, dram_row=1024,
             gdma_m=32, pe_dma_m=8, sram_banks=16, hop=2, link=64,
             V=64, t_vu=4, vbuf=2, vout=2, vu_dma_m=32, vmem_banks=16)

GEMM_HW_KEYS = ("R", "C", "Kt", "px", "py", "sram_x", "sram_y", "D", "nbuf", "nout", "hop", "link",
                "sram_banks", "gdma_m", "pe_dma_m")
VU_HW_KEYS = ("V", "t_vu", "vbuf", "vout", "vmem_banks", "vu_dma_m")
DRAM_HW_KEYS = ("dram_w", "t_hit", "t_miss", "dram_q", "dram_banks", "dram_row")
GEMM_CATS = ("compute", "dram", "sram_conflict", "network", "dma", "idle")
VU_CATS = ("busy", "load", "store")


def _round_up(x, to):
    return -(-x // to) * to


def _lcm(*xs):
    out = 1
    for x in xs:
        out = out * x // math.gcd(out, x)
    return out


# ---------------------------------------------------------------- lowering

def lower(spec, hw):
    """spec: mode, L or T, d, g, prec, p_scale, mapping ('auto' | 'physical'). Returns the program."""
    R, C, Kt = hw["R"], hw["C"], hw["Kt"]
    d, prec = spec["d"], spec["prec"]
    e = nx.ELEM[prec]
    p = hw["px"] * hw["py"]
    errors = []
    if spec["mode"] == "prefill":
        L = spec["L"]
        G, N, M, causal = L, L, L, True
        if L % _lcm(R, C, Kt):
            errors.append(f"L={L} must be a multiple of lcm(R, C, Kt) = {_lcm(R, C, Kt)}")
    elif spec["mode"] == "decode":
        T, g = spec["T"], spec["g"]
        G, N, M, causal = g, T, R, False
        if g > R:
            errors.append(f"g={g} must be <= R={R}")
        if T % _lcm(C, Kt):
            errors.append(f"T={T} must be a multiple of lcm(C, Kt) = {_lcm(C, Kt)}")
    else:
        raise ValueError("mode must be prefill or decode")
    if d % C or d % Kt:
        errors.append(f"d={d} must be a multiple of C={C} and Kt={Kt}")
    if errors:
        raise ValueError("; ".join(errors))
    if spec["mode"] == "decode" and spec.get("mapping", "auto") == "auto":
        px_map, py_map = 1, p
    else:
        px_map, py_map = 0, 0

    row = hw["dram_row"]
    sizes = [("q", M * d * e), ("kt", d * N * e), ("s", G * N * 4), ("p", M * N * e),
             ("v", N * d * e), ("o", G * d * 4), ("scales", _round_up(4 * G, 16))]
    base, at = {}, 0
    for name, size in sizes:
        base[name] = at
        at += _round_up(size, row)
    gemm_map = dict(px_map=px_map, py_map=py_map)
    ops = [
        dict(op="gemm", name="qk", M=M, N=N, K=d, m_valid=G, a_base=base["q"], b_base=base["kt"],
             c_base=base["s"], **gemm_map),
        dict(op="softmax", name="softmax", rows=G, n=N, s_base=base["s"], p_base=base["p"], R=R, Kt=Kt,
             causal=int(causal), q_off=0, p_scale=spec.get("p_scale", "fixed"), scale_base=base["scales"]),
        dict(op="gemm", name="pv", M=M, N=d, K=N, m_valid=G, a_base=base["p"], b_base=base["v"],
             c_base=base["o"], **gemm_map),
    ]
    return dict(spec=dict(spec), G=G, N=N, M=M, causal=causal, base=base, image_bytes=at, ops=ops,
                useful_ops=4 * G * N * d)


# ---------------------------------------------------------------- inputs and layouts

def make_inputs(spec, prog):
    """FP32 N(0,1) Q, K, V; their stored encodings; scales and alpha (design note §2.1, §4.4)."""
    rng = np.random.default_rng([SEED, ATTN_STREAM, spec.get("seed", 0)])
    G, N, d, prec = prog["G"], prog["N"], spec["d"], spec["prec"]
    qf = rng.standard_normal((G, d), dtype=np.float32)
    kf = rng.standard_normal((N, d), dtype=np.float32)
    vf = rng.standard_normal((N, d), dtype=np.float32)
    inv_sqrt_d = np.float32(1.0 / math.sqrt(d))
    out = dict(qf=qf, kf=kf, vf=vf)
    if prec == "int8":
        for name, x in (("q", qf), ("k", kf), ("v", vf)):
            s = np.float32(np.abs(x).max()) / np.float32(127.0)
            q = np.clip(np.rint(x / s), -127, 127).astype(np.int8)
            out[name] = q
            out["s_" + name] = np.float32(s)
        alpha = np.float32(np.float32(out["s_q"] * out["s_k"]) * inv_sqrt_d)
    else:
        for name, x in (("q", qf), ("k", kf), ("v", vf)):
            out[name] = nx.decode(prec, nx.encode(prec, x))  # the stored values, decoded
        alpha = inv_sqrt_d
    out["alpha_bits"] = int(np.array(alpha, dtype=np.float32).view(np.uint32))
    return out


def raw(prec, x):
    """Values (int8 or float32) -> stored bytes, shape + [e]."""
    return nx.encode(prec, x)


def pack_a(r, R, Kt):
    """(M, K, e) -> packed A blocks (mi, c), each R x Kt row-major."""
    M, K, e = r.shape
    return r.reshape(M // R, R, K // Kt, Kt, e).transpose(0, 2, 1, 3, 4).reshape(-1)


def unpack_a(flat, M, K, R, Kt, e):
    return flat.reshape(M // R, K // Kt, R, Kt, e).transpose(0, 2, 1, 3, 4).reshape(M, K, e)


def pack_b(r, C, Kt):
    """(K, N, e) -> packed B blocks (nj, c), each Kt x C row-major."""
    K, N, e = r.shape
    return r.reshape(K // Kt, Kt, N // C, C, e).transpose(2, 0, 1, 3, 4).reshape(-1)


def build_image(prog, inp, hw):
    spec, prec = prog["spec"], prog["spec"]["prec"]
    e, d, G, N, M = nx.ELEM[prec], spec["d"], prog["G"], prog["N"], prog["M"]
    img = np.zeros(prog["image_bytes"], dtype=np.uint8)
    q = np.zeros((M, d, e), dtype=np.uint8)
    q[:G] = raw(prec, inp["q"])  # rows G..M-1 stay zero (padding, §5.1)
    blocks = [("q", pack_a(q, hw["R"], hw["Kt"])),
              ("kt", pack_b(np.ascontiguousarray(raw(prec, inp["k"]).transpose(1, 0, 2)), hw["C"], hw["Kt"])),
              ("v", pack_b(raw(prec, inp["v"]), hw["C"], hw["Kt"]))]
    for name, data in blocks:
        img[prog["base"][name]:prog["base"][name] + data.size] = data
    return img


def extract(prog, img, hw):
    """Model outputs from the final image: S words, P bytes (valid rows, row-major), scales, O words."""
    spec, prec = prog["spec"], prog["spec"]["prec"]
    e, d, G, N, M = nx.ELEM[prec], spec["d"], prog["G"], prog["N"], prog["M"]
    b = prog["base"]
    s = img[b["s"]:b["s"] + G * N * 4].view("<u4").reshape(G, N)
    p_all = unpack_a(img[b["p"]:b["p"] + M * N * e], M, N, hw["R"], hw["Kt"], e)
    o = img[b["o"]:b["o"] + G * d * 4].view("<u4").reshape(G, d)
    scales = img[b["scales"]:b["scales"] + 4 * G].view("<f4").copy()
    return dict(s=s.copy(), p=np.ascontiguousarray(p_all[:G]), o=o.copy(), scales=scales,
                p_pad=np.ascontiguousarray(p_all[G:]))


# ---------------------------------------------------------------- running

def op_args(op, prog, hw):
    prec = prog["spec"]["prec"]
    args = [f"op={op['op']}", f"prec={prec}"] + [f"{k}={hw[k]}" for k in DRAM_HW_KEYS]
    if op["op"] == "gemm":
        args += [f"{k}={hw[k]}" for k in GEMM_HW_KEYS]
        args += [f"{k}={op[k]}" for k in ("M", "N", "K", "m_valid", "a_base", "b_base", "c_base",
                                          "px_map", "py_map")]
    else:
        args += [f"{k}={hw[k]}" for k in VU_HW_KEYS]
        args += [f"{k}={op[k]}" for k in ("rows", "n", "s_base", "p_base", "R", "Kt", "causal", "q_off",
                                          "p_scale", "scale_base")]
        args += [f"alpha_bits={prog['alpha_bits']}"]
    return args


def run_program(prog, img, hw, tools, work):
    """Runs every op in order; returns (per-op results, final image)."""
    env = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE="1")
    path_in, path_out = os.path.join(work, "img_in.bin"), os.path.join(work, "img_out.bin")
    img.tofile(path_in)
    results = []
    for op in prog["ops"]:
        cmd = [os.path.join(tools, "apm_op")] + op_args(op, prog, hw) + [f"img={path_in}", f"out_img={path_out}"]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            raise RuntimeError(f"apm_op {op['name']} failed ({p.returncode}): {p.stderr.strip()}")
        v = json.loads(p.stdout)["values"]
        res = dict(name=op["name"], op=op["op"], t_end=v["t_end"], dram_bytes=v["dram_bytes"],
                   row_hits=v["dram.row_hits"], row_misses=v["dram.row_misses"], bus_busy=v["dram.bus_busy_cycles"])
        if op["op"] == "gemm":
            res.update(pes=v["pes"], ops=v["ops"], useful_ops=v["useful_ops"],
                       max_link_busy=v["mesh.max_link_busy"],
                       stalls={c: v[f"stall.{c}"] for c in GEMM_CATS})
        else:
            res.update(elements=v["elements"], t_row=v["t_row"], stalls={c: v[f"vu.{c}"] for c in VU_CATS})
        results.append(res)
        os.replace(path_out, path_in)
    return results, np.fromfile(path_in, dtype=np.uint8)


def program_summary(prog, results, hw):
    """Totals and the program-level PE stack (§4.5): Σ = P·T_program is asserted."""
    p = hw["px"] * hw["py"]
    t_prog = sum(r["t_end"] for r in results)
    stack = {c: 0 for c in GEMM_CATS}
    stack["vu_wait"] = 0
    for r in results:
        if r["op"] == "gemm":
            assert sum(r["stalls"].values()) == r["pes"] * r["t_end"], f"{r['name']}: PE stack sum"
            for c in GEMM_CATS:
                stack[c] += r["stalls"][c]
        else:
            assert sum(r["stalls"].values()) == r["t_end"], "VU stack sum"
            stack["vu_wait"] += p * r["t_end"]
    assert sum(stack.values()) == p * t_prog, "program PE stack must sum to P * T_program"
    return dict(t_program=t_prog, pe_stack=stack, dram_bytes=sum(r["dram_bytes"] for r in results),
                useful_ops=prog["useful_ops"])


# ---------------------------------------------------------------- references

def replica(prog, inp):
    """NumPy replica of the whole program (S words, P bytes, scales, O words)."""
    spec, prec = prog["spec"], prog["spec"]["prec"]
    s = nx.gemm(prec, inp["q"], inp["k"].T)
    p_raw, scales = nx.softmax(prec, s, inp["alpha_bits"], spec_lanes(prog), causal=prog["causal"], q_off=0,
                               p_scale=spec.get("p_scale", "fixed"))
    o = nx.gemm(prec, nx.decode(prec, p_raw), inp["v"])
    return dict(s=np.asarray(s).view(np.uint32), p=p_raw, scales=scales, o=np.asarray(o).view(np.uint32))


def spec_lanes(prog):
    return prog["hw"]["V"]


def run_attn_ref(prog, inp, tools, work):
    spec, prec = prog["spec"], prog["spec"]["prec"]
    files = {}
    for name in ("q", "k", "v"):
        files[name] = os.path.join(work, f"ref_{name}.bin")
        raw(prec, inp[name]).tofile(files[name])
    out = os.path.join(work, "ref_out")
    os.makedirs(out, exist_ok=True)
    cmd = [os.path.join(tools, "attn_ref"), prec, str(prog["G"]), str(prog["N"]), str(spec["d"]),
           str(int(prog["causal"])), "0", str(inp["alpha_bits"]), str(prog["hw"]["V"]),
           spec.get("p_scale", "fixed"), files["q"], files["k"], files["v"], out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"attn_ref failed: {p.stderr.strip()}")
    G, N, d, e = prog["G"], prog["N"], spec["d"], nx.ELEM[prec]
    return dict(s=np.fromfile(os.path.join(out, "S.bin"), dtype="<u4").reshape(G, N),
                p=np.fromfile(os.path.join(out, "P.bin"), dtype=np.uint8).reshape(G, N, e),
                scales=np.fromfile(os.path.join(out, "scales.bin"), dtype="<f4"),
                o=np.fromfile(os.path.join(out, "O.bin"), dtype="<u4").reshape(G, d))


def fp64_reference(prog, inp):
    q, k, v = (inp[x].astype(np.float64) for x in ("qf", "kf", "vf"))
    s = q @ k.T / math.sqrt(prog["spec"]["d"])
    if prog["causal"]:
        s = np.where(np.arange(s.shape[1])[None, :] <= np.arange(s.shape[0])[:, None], s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    w = np.exp(s)
    return (w / w.sum(axis=1, keepdims=True)) @ v


def dequant_o(prog, inp, o_words, scales):
    prec = prog["spec"]["prec"]
    if prec != "int8":
        return o_words.view(np.float32)
    oq = o_words.view(np.int32).astype(np.float32)
    return (oq * (scales * inp["s_v"]).astype(np.float32)[:, None]).astype(np.float32)


def compare(model, rep, ref, writes_scales):
    out = {}
    for key in ("s", "p", "scales", "o"):
        m, n, c = (np.asarray(x[key]) for x in (model, rep, ref))
        if key == "scales":
            if not writes_scales:  # only INT8 with per-row P scales writes them
                m = n = c = np.zeros(1, dtype=np.float32)
            m, n, c = (np.asarray(x, dtype=np.float32).view(np.uint32) for x in (m, n, c))
        out[f"{key}_model_eq_numpy"] = bool(np.array_equal(m, n))
        out[f"{key}_cpp_eq_numpy"] = bool(np.array_equal(c, n))
        out[f"{key}_model_eq_cpp"] = bool(np.array_equal(m, c))
    out["match"] = all(v for k, v in out.items())
    return out


def run_attention(spec, hw=None, tools="build/release", check=False, fp64=True):
    hw = dict(HW_A4 if hw is None else hw)
    prog = lower(spec, hw)
    prog["hw"] = hw
    inp = make_inputs(spec, prog)
    prog["alpha_bits"] = inp["alpha_bits"]
    with tempfile.TemporaryDirectory() as work:
        img = build_image(prog, inp, hw)
        results, final = run_program(prog, img, hw, tools, work)
        model = extract(prog, final, hw)
        writes_scales = spec["prec"] == "int8" and spec.get("p_scale", "fixed") == "row"
        if spec["prec"] == "int8" and not writes_scales:  # fixed P scale: a constant, not stored
            model["scales"] = np.full(prog["G"], np.float32(1.0) / np.float32(127.0), dtype=np.float32)
        out = dict(spec=spec, ops=results, **program_summary(prog, results, hw))
        out["padding_untouched"] = bool(not model["p_pad"].any())
        if check:
            rep = replica(prog, inp)
            ref = run_attn_ref(prog, inp, tools, work)
            out["check"] = compare(model, rep, ref, writes_scales)
            out["check"]["match"] = out["check"]["match"] and out["padding_untouched"]
        if fp64:
            o = dequant_o(prog, inp, model["o"], model["scales"])
            o64 = fp64_reference(prog, inp)
            diff = o.astype(np.float64) - o64
            out["rel_fro_err"] = float(np.linalg.norm(diff) / np.linalg.norm(o64))
            out["max_abs_err"] = float(np.abs(diff).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("prefill", "decode"), required=True)
    ap.add_argument("--L", type=int)
    ap.add_argument("--T", type=int)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--g", type=int, default=1)
    ap.add_argument("--prec", choices=nx.PRECS, default="fp16")
    ap.add_argument("--p-scale", choices=("fixed", "row"), default="row")
    ap.add_argument("--mapping", choices=("auto", "physical"), default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hw", help="JSON file overriding A4 keys")
    ap.add_argument("--tools", default="build/release")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    hw = dict(HW_A4)
    if a.hw:
        with open(a.hw) as f:
            hw.update(json.load(f))
    spec = dict(mode=a.mode, d=a.d, prec=a.prec, p_scale=a.p_scale, mapping=a.mapping, seed=a.seed)
    if a.mode == "prefill":
        spec["L"] = a.L
    else:
        spec.update(T=a.T, g=a.g)
    out = run_attention(spec, hw, a.tools, check=a.check)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if (not a.check or out["check"]["match"]) else 1


if __name__ == "__main__":
    sys.exit(main())
