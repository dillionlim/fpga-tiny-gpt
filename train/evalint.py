"""Check the integer reference: loss on validation text and a sample.

  python3 -I evalint.py params_C_quick.pkl
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intref import IntGPT  # noqa: E402
from vocab import decode, encode  # noqa: E402

p = pickle.load(open(sys.argv[1], "rb"))
val = np.load("/home/dillion/badge-llm/data/ts_valid.npy")
m = IntGPT(p)
k = p["logit_k"]
nll, n = 0.0, 0
t0 = time.time()
for s in range(3):
    m.reset()
    seq = val[100000 * s: 100000 * s + 301]
    for i in range(300):
        lg = m.step(int(seq[i])) * k
        lg = lg - lg.max()
        nll += np.log(np.exp(lg).sum()) - lg[seq[i + 1]]
        n += 1
print(f"int model val loss {nll / n:.4f} nats/char over {n} chars ({(time.time() - t0) / n * 1000:.1f} ms/step)")

m.reset()
m.rng = 12345
mt = m.temp_mult(0.8)
prompt = "Once upon a time"
out = []
toks = list(encode(prompt))
for t in toks[:-1]:
    m.sample(m.step(t), mt, 96)
t = toks[-1]
for _ in range(300):
    t = m.sample(m.step(t), mt, 96)
    out.append(t)
print(repr(prompt + decode(out)))
