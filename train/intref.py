"""Bit-exact integer reference of the FPGA datapath.

Every operation here has a one-to-one counterpart in rtl/llm_core.v. The RTL
testbench compares its outputs against this module, and export.py derives all
constants from the trained checkpoint.

Integer formats
  X  residual        int16, Q7.8
  N  norm output     int8, unit RMS * 2**norm_g
  q, k, v, h, a      int8, per-tensor scale
  K/V cache          int8, ring buffer of ctx positions (MQA: one K/V head)
"""
import math

import numpy as np

ALIBI_A = [64, 16, 4, 1]
EXP_LUT = np.array([round(65535 * math.exp(-u / 32)) for u in range(512)], dtype=np.int64)
I8 = 127
I16 = 32767


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def round_div(a, b):
    """Divide rounding half away from zero. b > 0. Works on ints and int64 arrays."""
    a = np.asarray(a, dtype=np.int64)
    q = (np.abs(a) + b // 2) // b
    return np.where(a < 0, -q, q)


def requant(acc, m, shift):
    """(acc * m + 2**(shift-1)) >>> shift, the rounding used after every matvec."""
    return (acc * m + (1 << (shift - 1))) >> shift


def isqrt(n):
    return math.isqrt(int(n))


def xorshift32(x):
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


class IntGPT:
    """p: dict produced by export.py (integer arrays and constants)."""

    def __init__(self, p):
        self.p = p
        c = p["cfg"]
        self.V, self.d, self.L = c["vocab"], c["d"], c["n_layer"]
        self.H, self.hd, self.ctx, self.ffn = c["n_head"], c["d"] // c["n_head"], c["ctx"], c["ffn"]
        self.norm_g = c["norm_g"]
        self.kc = np.zeros((self.L, self.ctx, self.hd), dtype=np.int64)
        self.vc = np.zeros((self.L, self.ctx, self.hd), dtype=np.int64)
        self.pos = 0
        self.rng = 1

    def reset(self):
        self.kc[:] = 0
        self.vc[:] = 0
        self.pos = 0

    # building blocks ------------------------------------------------------
    def norm(self, x):
        ss = int(np.sum(x * x))
        s = max(isqrt(ss >> 2), 1)
        return clamp(round_div(x << (self.norm_g + 2), s), -I8, I8)

    def matvec(self, w, n):
        return w.astype(np.int64) @ n.astype(np.int64)

    def attention(self, l, q):
        p = self.p
        nd = min(self.pos, self.ctx - 1) + 1
        d = np.arange(nd)
        slots = (self.pos - d) % self.ctx
        K, Vv = self.kc[l][slots], self.vc[l][slots]
        out = np.zeros(self.d, dtype=np.int64)
        for h in range(self.H):
            qh = q[h * self.hd:(h + 1) * self.hd]
            score = K @ qh
            z = requant(score, p["att_m"][l], p["att_s"][l]) - ((d * ALIBI_A[h]) >> 3)
            u = np.minimum(z.max() - z, 511)
            w = EXP_LUT[u]
            sw = int(w.sum())
            o = w @ Vv
            out[h * self.hd:(h + 1) * self.hd] = clamp(round_div(o, sw), -I8, I8)
        return out

    # one token ------------------------------------------------------------
    def step(self, tok):
        """Run the network on tok at the current position; returns int logits."""
        p = self.p
        x = p["emb"][tok].astype(np.int64) << (8 - p["emb_shift"])
        slot = self.pos % self.ctx
        for l in range(self.L):
            P = p["layers"][l]
            n = self.norm(x)
            q = clamp(requant(self.matvec(P["wq"], n), P["mq"], P["sq"]), -I8, I8)
            k = clamp(requant(self.matvec(P["wk"], n), P["mk"], P["sk"]), -I8, I8)
            v = clamp(requant(self.matvec(P["wv"], n), P["mv"], P["sv"]), -I8, I8)
            self.kc[l, slot] = k
            self.vc[l, slot] = v
            a = self.attention(l, q)
            x = clamp(x + requant(self.matvec(P["wo"], a), P["mo"], P["so"]), -I16, I16)
            n = self.norm(x)
            h = self.matvec(P["w1"], n)
            h = clamp(requant(np.maximum(h, 0), P["m1"], P["s1"]), 0, I8)
            x = clamp(x + requant(self.matvec(P["w2"], h), P["m2"], P["s2"]), -I16, I16)
        n = self.norm(x)
        self.pos += 1
        self.last_x = x
        return self.matvec(p["emb"], n)

    def sample(self, logits, mt, ucut):
        z = (logits * mt) >> 16
        u = z.max() - z
        w = np.where(u <= ucut, EXP_LUT[np.minimum(u, 511)], 0)
        sw = int(w.sum())
        self.rng = xorshift32(self.rng)
        r = self.rng % sw
        acc = np.cumsum(w)
        return int(np.argmax(acc > r))

    def logit_scale(self):
        """Real-valued logit per integer logit unit."""
        return self.p["logit_k"]

    def temp_mult(self, temperature):
        return min(int(round(self.p["logit_k"] * 32 / temperature * 65536)), 65535)
