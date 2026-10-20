"""Expected per-step logits and samples for the RTL testbench.

  python3 -I gen_vectors.py params.pkl ../rtl/gen NSTEPS
Layout of vectors.hex (32-bit words), per step: tok_in, sampled, 64 logits.
cfg.hex: seed, mt, ucut, n_prompt (steps run one at a time), n_total.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intref import IntGPT  # noqa: E402
from vocab import encode  # noqa: E402

p = pickle.load(open(sys.argv[1], "rb"))
force_prompt = int(sys.argv[4]) if len(sys.argv) > 4 else None
out, n_total = sys.argv[2], int(sys.argv[3])
m = IntGPT(p)
seed, ucut = 0xC0FFEE11, 96
m.rng = seed
mt = m.temp_mult(0.8)
prompt = list(encode("Once upon a"))
n_prompt = min(len(prompt) - 1, n_total - 1) if force_prompt is None else force_prompt
words = []
tok = prompt[0]
for i in range(n_total):
    lg = m.step(tok)
    s = m.sample(lg, mt, ucut)
    words += [tok, s] + [int(x) & 0xFFFFFF for x in lg]
    tok = prompt[i + 1] if i + 1 <= n_prompt else s
with open(f"{out}/vectors.hex", "w") as f:
    f.write("\n".join(f"{w:08x}" for w in words) + "\n")
with open(f"{out}/cfg.hex", "w") as f:
    f.write("\n".join(f"{w:08x}" for w in [seed, mt, ucut, n_prompt, n_total]) + "\n")
print(f"{n_total} steps ({n_prompt} single, {n_total - n_prompt} autoregressive), mt={mt}")
