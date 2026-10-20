"""Convert a QAT checkpoint into the integer parameters used by intref.py and the RTL.

  python3 -I export.py ckpt_B.pt params_B.pkl
"""
import math
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Cfg  # noqa: E402

MULT_BITS = 16


def quant_rows(w):
    s = np.maximum(np.abs(w).max(1), 1e-8) / 127
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def mult_and_shift(m_real):
    """Pick the largest shift that keeps every multiplier below 2**MULT_BITS."""
    m_real = np.atleast_1d(m_real)
    shift = int(math.floor(math.log2((2 ** MULT_BITS - 1) / m_real.max())))
    shift = min(shift, 40)
    m = np.round(m_real * 2.0 ** shift).astype(np.int64)
    assert m.max() < 2 ** MULT_BITS and m.min() > 0 and shift >= 1, (m.max(), shift)
    return m, shift


def export(ckpt):
    cfg = Cfg(**ckpt["cfg"])
    st = {k: v.detach().cpu().double().numpy() for k, v in ckpt["state"].items()}
    hd = cfg.d // cfg.n_head
    s_n = 2.0 ** -cfg.norm_g
    emb_shift = int(st["emb_shift"])
    emb = np.clip(np.round(st["emb"] * 2 ** emb_shift), -127, 127).astype(np.int8)
    p = {"cfg": cfg.to_dict(), "emb": emb, "emb_shift": emb_shift, "layers": [],
         "att_m": [], "att_s": []}
    for l in range(cfg.n_layer):
        g = lambda n: st[f"blocks.{l}.{n}"]  # noqa: E731
        sq, sk, sv, sh = (float(g("s_" + n)) for n in "qkvh")
        P = {}
        for name, w, s_in, s_out in [
            ("q", g("wq") * g("g1")[None], s_n, sq),
            ("k", g("wk") * g("g1")[None], s_n, sk),
            ("v", g("wv") * g("g1")[None], s_n, sv),
            ("o", g("wo"), sv, 1 / 256),
            ("1", g("w1") * g("g2")[None], s_n, sh),
            ("2", g("w2"), sh, 1 / 256),
        ]:
            wq, sw = quant_rows(w)
            m, sh_ = mult_and_shift(sw * s_in / s_out)
            P["w" + name], P["m" + name], P["s" + name] = wq, m, sh_
        p["layers"].append(P)
        am, as_ = mult_and_shift(sq * sk * 32 / math.sqrt(hd))
        p["att_m"].append(int(am[0]))
        p["att_s"].append(as_)
    p["logit_k"] = 2.0 ** -emb_shift * s_n * float(st["logit_scale"])
    return p


def main():
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    assert ckpt["qat"], "export a QAT checkpoint"
    p = export(ckpt)
    with open(sys.argv[2], "wb") as f:
        pickle.dump(p, f)
    for l, P in enumerate(p["layers"]):
        print(f"layer {l}: shifts " + " ".join(f"{n}={P['s' + n]}" for n in "qkvo12")
              + f" att_m={p['att_m'][l]} att_s={p['att_s'][l]}")
    print("emb_shift", p["emb_shift"], "logit_k", p["logit_k"])


if __name__ == "__main__":
    main()
