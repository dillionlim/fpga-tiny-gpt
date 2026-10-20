"""Tiny character-level GPT whose quantized forward pass matches the FPGA datapath.

Float mode trains normally. QAT mode inserts the same rounding and clipping the
hardware applies:
  - residual stream: int16, Q7.8 (step 1/256), saturating adds
  - RMSNorm output: int8 with step 1/2**norm_g (unit-RMS values), clipped to +-127
  - weights: int8, symmetric, one scale per output row (norm gains folded in first)
  - q, k, v, mlp hidden: int8 with a per-tensor scale fixed by calibration
  - attention logits: rounded to 1/32 nat, ALiBi bias floor(dist * A / 8) / 32 nat
  - embedding: int8 with a power-of-two scale, shared with the output head
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

ALIBI_A = [64, 16, 4, 1]  # bias per position, in units of 1/256 nat (1/32 nat after >> 3)


@dataclass
class Cfg:
    vocab: int = 64
    d: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 1
    ffn: int = 256
    ctx: int = 128
    norm_g: int = 5

    @property
    def head_dim(self):
        return self.d // self.n_head

    def to_dict(self):
        return asdict(self)


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def fake_quant(x, scale, lo=-127, hi=127):
    y = torch.clamp(x / scale, lo, hi)
    return round_ste(y) * scale


def alibi_bias(ctx, n_head, device):
    """(n_head, ctx, ctx) additive bias in nats, -inf above the diagonal."""
    i = torch.arange(ctx, device=device)
    dist = (i[:, None] - i[None, :]).clamp(min=0)
    a = torch.tensor(ALIBI_A[:n_head], device=device)
    units = torch.div(dist[None] * a[:, None, None], 8, rounding_mode="floor")
    bias = -units.float() / 32.0
    causal = i[None, :] <= i[:, None]
    return bias.masked_fill(~causal[None], float("-inf"))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, hd = cfg.d, cfg.head_dim
        self.cfg = cfg
        self.g1 = nn.Parameter(torch.ones(d))
        self.g2 = nn.Parameter(torch.ones(d))
        self.wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.wk = nn.Parameter(torch.randn(cfg.n_kv_head * hd, d) / math.sqrt(d))
        self.wv = nn.Parameter(torch.randn(cfg.n_kv_head * hd, d) / math.sqrt(d))
        self.wo = nn.Parameter(torch.randn(d, d) / math.sqrt(d) / math.sqrt(2 * cfg.n_layer))
        self.w1 = nn.Parameter(torch.randn(cfg.ffn, d) / math.sqrt(d))
        self.w2 = nn.Parameter(torch.randn(d, cfg.ffn) / math.sqrt(cfg.ffn) / math.sqrt(2 * cfg.n_layer))
        for name in ["q", "k", "v", "h"]:
            self.register_buffer("s_" + name, torch.tensor(1.0))


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Parameter(torch.randn(cfg.vocab, cfg.d) * 0.5)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("emb_shift", torch.tensor(5))  # embedding scale 2**-emb_shift
        self.qat = False
        self.stats = None  # dict of lists, filled during calibration
        self._bias = None

    # quantization helpers -------------------------------------------------
    def _norm(self, x):
        rms = torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-12)
        n = x / rms
        if self.stats is not None:
            self.stats.setdefault("norm_absmax", []).append(n.abs().amax().item())
            self.stats.setdefault("norm_clip_frac", []).append(
                (n.abs() * 2 ** self.cfg.norm_g > 127).float().mean().item())
        if self.qat:
            step = 2.0 ** -self.cfg.norm_g
            n = fake_quant(n, step)
        return n

    def _resid(self, x):
        if self.stats is not None:
            self.stats.setdefault("resid_absmax", []).append(x.abs().amax().item())
        if self.qat:
            return fake_quant(x, 1 / 256, -32767, 32767)
        return x

    def _w(self, w, gain=None):
        if gain is not None:
            w = w * gain[None, :]
        if self.qat:
            s = w.detach().abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
            w = fake_quant(w, s)
        return w

    def _act(self, x, blk, name, lo=-127):
        if self.stats is not None:
            self.stats.setdefault(f"{name}{self._layer}", []).append(x.detach().abs().flatten())
        if self.qat:
            return fake_quant(x, getattr(blk, "s_" + name), lo, 127)
        return x

    def _emb_q(self):
        if self.qat:
            return fake_quant(self.emb, 2.0 ** -int(self.emb_shift))
        return self.emb

    # forward --------------------------------------------------------------
    def forward(self, idx, targets=None):
        cfg = self.cfg
        B, T = idx.shape
        H, hd, KV = cfg.n_head, cfg.head_dim, cfg.n_kv_head
        if self._bias is None or self._bias.shape[-1] < T or self._bias.device != idx.device:
            self._bias = alibi_bias(max(T, cfg.ctx), H, idx.device)
        bias = self._bias[:, :T, :T]

        E = self._emb_q()
        x = self._resid(E[idx.long()])
        for li, blk in enumerate(self.blocks):
            self._layer = li
            n = self._norm(x)
            q = self._act(n @ self._w(blk.wq, blk.g1).T, blk, "q")
            k = self._act(n @ self._w(blk.wk, blk.g1).T, blk, "k")
            v = self._act(n @ self._w(blk.wv, blk.g1).T, blk, "v")
            q = q.view(B, T, H, hd).transpose(1, 2)
            k = k.view(B, T, KV, hd).transpose(1, 2).repeat_interleave(H // KV, 1)
            v = v.view(B, T, KV, hd).transpose(1, 2).repeat_interleave(H // KV, 1)
            z = (q @ k.transpose(-1, -2)) * (32.0 / math.sqrt(hd))  # 1/32 nat units
            if self.qat:
                z = round_ste(z)
            att = torch.softmax(z / 32.0 + bias, dim=-1)
            a = (att @ v).transpose(1, 2).reshape(B, T, cfg.d)
            if self.qat:
                a = fake_quant(a, blk.s_v)
            x = self._resid(x + a @ self._w(blk.wo).T)

            n = self._norm(x)
            h = F.relu(n @ self._w(blk.w1, blk.g2).T)
            h = self._act(h, blk, "h", lo=0)
            x = self._resid(x + h @ self._w(blk.w2).T)

        n = self._norm(x)
        logits = (n @ E.T) * self.logit_scale
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, cfg.vocab), targets.reshape(-1).long())
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
