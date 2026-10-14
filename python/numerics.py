#!/usr/bin/env python3
"""NumPy replica of the Phase 4 numerics spec (docs/phase4-design.md §2).

An independent implementation of every floating-point operation the model
performs, written from the design note: FP16/BF16 encodings with round to
nearest even, the GEMM accumulation order (ascending k, one rounding for the
product and one for the add), the VU exponential, the lane/tree sum and the
row softmax with its P encodings. The model (C++), this module and
tools/attn_ref.cpp are compared bit for bit.

Every array operation here is on float32 arrays, so each op rounds once to
FP32; no Python-float scalar ever mixes into a float32 scalar computation.
"""
import numpy as np

PRECS = ("int8", "fp16", "bf16", "fp32")
ELEM = {"int8": 1, "fp16": 2, "bf16": 2, "fp32": 4}

# FP32 bit patterns of c_k = fl32((ln 2)^k / k!), k = 0..6, and fl32(log2 e).
VU_EXP_COEFF_BITS = (0x3F800000, 0x3F317218, 0x3E75FDF0, 0x3D635847, 0x3C1D955B, 0x3AAEC3FF, 0x39218489)
LOG2E_BITS = 0x3FB8AA3B


def f32_from_bits(b):
    return np.array(b, dtype=np.uint32).view(np.float32)


def bits_of(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


# ---------------------------------------------------------------- encodings

def bf16_encode(x):
    """float32 array -> uint16 BF16 patterns, round to nearest even (no NaN inputs)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return ((u >> 16) & 0xFFFF).astype(np.uint16)


def bf16_decode(h):
    return (np.asarray(h, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def fp16_encode(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).view(np.uint16)


def fp16_decode(h):
    return np.asarray(h, dtype=np.uint16).view(np.float16).astype(np.float32)


def encode(prec, x):
    """float32 values (or int8 values for INT8) -> raw little-endian bytes array (shape + [e])."""
    x = np.asarray(x)
    if prec == "int8":
        return x.astype(np.int8).view(np.uint8).reshape(x.shape + (1,))
    if prec == "fp16":
        w = fp16_encode(x)
    elif prec == "bf16":
        w = bf16_encode(x)
    else:
        w = np.asarray(x, dtype=np.float32).view(np.uint32)
    return w.astype(w.dtype.newbyteorder("<")).view(np.uint8).reshape(x.shape + (ELEM[prec],))


def decode(prec, raw):
    """raw bytes (shape + [e]) -> float32 (FP modes) or int32 (INT8)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    shape = raw.shape[:-1]
    if prec == "int8":
        return raw.reshape(shape).view(np.int8).astype(np.int32)
    if prec == "fp16":
        return fp16_decode(raw.view("<u2").reshape(shape))
    if prec == "bf16":
        return bf16_decode(raw.view("<u2").reshape(shape))
    return raw.view("<f4").reshape(shape).astype(np.float32)


# ---------------------------------------------------------------- GEMM

def gemm(prec, a, b):
    """a (M x K), b (K x N) decoded values -> C as 4-byte words (int32 or float32).

    INT8: exact INT32. FP modes: acc = fl32(acc + fl32(a*b)) in ascending k."""
    if prec == "int8":
        c = a.astype(np.int64) @ b.astype(np.int64)
        assert c.min() >= -(2 ** 31) and c.max() < 2 ** 31, "INT32 overflow"
        return c.astype(np.int32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for k in range(a.shape[1]):
        prod = a[:, k:k + 1] * b[k:k + 1, :]  # float32 * float32: one rounding
        acc = acc + prod
    return acc


# ---------------------------------------------------------------- VU

def vu_exp(x):
    """§2.3 for float32 x <= 0 (array)."""
    x = np.asarray(x, dtype=np.float32)
    assert np.all(x <= 0), "vu_exp: argument must be <= 0"
    t = x * f32_from_bits(LOG2E_BITS)
    flush = t < np.float32(-125.0)
    t = np.where(flush, np.float32(0.0), t)  # keep the masked lanes harmless
    n = np.rint(t)  # ties to even
    f = t - n
    c = [f32_from_bits(b) for b in VU_EXP_COEFF_BITS]
    p = np.full(x.shape, c[6], dtype=np.float32)
    for k in range(5, -1, -1):
        pf = p * f
        p = pf + c[k]
    out = np.ldexp(p, n.astype(np.int32)).astype(np.float32)
    return np.where(flush, np.float32(0.0), out)


def lane_tree_sum(e, lanes):
    """Row-wise §2.4 step 3 for a (rows x n) float32 matrix."""
    rows, n = e.shape
    width = -(-n // lanes) * lanes
    padded = np.zeros((rows, width), dtype=np.float32)
    padded[:, :n] = e
    lane = np.zeros((rows, lanes), dtype=np.float32)
    for c in range(width // lanes):
        lane = lane + padded[:, c * lanes:(c + 1) * lanes]
    w = lanes // 2
    while w >= 1:
        lane = np.concatenate([lane[:, :w] + lane[:, w:2 * w], lane[:, 2 * w:]], axis=1)
        w //= 2
    return lane[:, 0]


def softmax(prec, s, alpha_bits, lanes, causal=False, q_off=0, p_scale="fixed"):
    """§2.4 for every row of s (INT32 for INT8, float32 otherwise).

    Returns (P raw bytes rows x n x e, row scales float32)."""
    rows, n = s.shape
    alpha = f32_from_bits(alpha_bits)
    j = np.arange(n)[None, :]
    i = np.arange(rows)[:, None]
    valid = (j <= i + q_off) if causal else np.ones((rows, n), dtype=bool)
    if prec == "int8":
        mi = np.where(valid, s, np.iinfo(np.int32).min).max(axis=1)
        m = mi.astype(np.float32)
        sv = s.astype(np.float32)
    else:
        sv = np.asarray(s, dtype=np.float32)
        m = np.where(valid, sv, np.float32(-np.inf)).max(axis=1).astype(np.float32)
    d = sv - m[:, None]
    x = d * alpha
    x = np.where(valid, x, np.float32(0.0))
    e = np.where(valid, vu_exp(x), np.float32(0.0)).astype(np.float32)
    total = lane_tree_sum(e, lanes)
    r = np.float32(1.0) / total  # float32 / float32: correctly rounded
    r = r.astype(np.float32)
    p = (e * r[:, None]).astype(np.float32)
    if prec == "int8":
        v = e if p_scale == "row" else p
        q = np.rint(v * np.float32(127.0)).astype(np.int8)
        raw = q.view(np.uint8).reshape(rows, n, 1)
        scales = (r / np.float32(127.0)).astype(np.float32) if p_scale == "row" else \
            np.full(rows, np.float32(1.0) / np.float32(127.0), dtype=np.float32)
        return raw, scales
    return encode(prec, p), np.ones(rows, dtype=np.float32)
