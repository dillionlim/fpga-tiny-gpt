"""Train the tiny GPT in float, calibrate activation scales, then fine-tune with QAT.

  python3 -I train.py --arch B --steps 20000 --qat-steps 4000 --out ckpt_B.pt
  python3 -I train.py --arch A --steps 3000 --qat-steps 0      # quick sweep run
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Cfg, TinyGPT  # noqa: E402

ARCHS = {
    "A": Cfg(n_layer=2, n_kv_head=4, ffn=128),  # MHA, small MLP
    "B": Cfg(n_layer=2, n_kv_head=1, ffn=256),  # MQA, wide MLP
    "C": Cfg(n_layer=3, n_kv_head=1, ffn=128),  # MQA, deeper
}
DATA = "/home/dillion/badge-llm/data"


def get_batch(data, B, T, device):
    off = torch.randint(0, data.numel() - T - 1, (B,), device=device)
    idx = off[:, None] + torch.arange(T + 1, device=device)
    chunk = data[idx]
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, val, B, T, n_batches=40, seed=1234):
    g = torch.Generator(device=val.device).manual_seed(seed)
    model.eval()
    losses = []
    for _ in range(n_batches):
        off = torch.randint(0, val.numel() - T - 1, (B,), device=val.device, generator=g)
        idx = off[:, None] + torch.arange(T + 1, device=val.device)
        chunk = val[idx]
        _, loss = model(chunk[:, :-1], chunk[:, 1:])
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def run_phase(model, train, val, args, steps, lr, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
    warm = min(500, steps // 10)
    t0 = time.time()
    for step in range(steps + 1):
        if step % args.eval_every == 0 or step == steps:
            vl = evaluate(model, val, 256, model.cfg.ctx)
            print(f"[{label}] step {step:6d}  val {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if step == steps:
            break
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * (0.05 + 0.95 * f)
        x, y = get_batch(train, args.batch, model.cfg.ctx, train.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return vl


@torch.no_grad()
def calibrate(model, train, n_batches=8, pct=99.99):
    model.eval()
    model.stats = {}
    for _ in range(n_batches):
        x, _ = get_batch(train, 128, model.cfg.ctx, train.device)
        model(x)
    st = model.stats
    model.stats = None
    model.train()
    report = {k: float(np.max(v)) for k, v in st.items() if not isinstance(v[0], torch.Tensor)}
    report["norm_clip_frac"] = float(np.mean(st["norm_clip_frac"]))
    for li, blk in enumerate(model.blocks):
        for name in ["q", "k", "v", "h"]:
            a = torch.cat(st[f"{name}{li}"])
            hi = torch.quantile(a[torch.randperm(a.numel(), device=a.device)[:2_000_000]], pct / 100).item()
            getattr(blk, "s_" + name).fill_(max(hi, 1e-3) / 127)
            report[f"s_{name}{li}"] = hi / 127
    emax = model.emb.abs().max().item()
    shift = int(math.floor(math.log2(127 / emax)))
    model.emb_shift.fill_(min(max(shift, 0), 8))
    report["emb_absmax"] = emax
    report["emb_shift"] = int(model.emb_shift)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="B")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--qat-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    train = torch.from_numpy(np.load(f"{DATA}/ts_train_200M.npy")).to(dev)
    val = torch.from_numpy(np.load(f"{DATA}/ts_valid.npy")).to(dev)

    cfg = ARCHS[args.arch]
    model = TinyGPT(cfg).to(dev)
    print(f"arch {args.arch} {cfg.to_dict()}  params {model.n_params()}")
    run_phase(model, train, val, args, args.steps, args.lr, "float")

    if args.qat_steps:
        rep = calibrate(model, train)
        print("calibration:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
        model.qat = True
        vl = evaluate(model, val, 256, cfg.ctx)
        print(f"[qat] before fine-tune val {vl:.4f}")
        run_phase(model, train, val, args, args.qat_steps, args.lr * 0.1, "qat")
        rep = calibrate(model, train)
        print("recalibration:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
        print(f"[qat] after recalibration val {evaluate(model, val, 256, cfg.ctx):.4f}")

    if args.out:
        torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "qat": model.qat}, args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
