#!/usr/bin/env python3
"""
reset_ckpt.py — copy a finished checkpoint into a new out_dir with the
iteration counter reset, so the next stage trains with a FRESH learning-rate
schedule (warmup + decay) instead of resuming at lr≈0.

train.py's resume path reads iter_num from the checkpoint and computes the LR
from it. Resuming a finished run therefore trains at essentially zero learning
rate — the model would not move. This rewrites iter_num to 0 and best_val_loss
to 1e9, and drops the optimizer state (train.py only loads it if present) so
the second stage starts with a clean optimizer at the new learning rate.

model_args are left untouched, so max_seq_len / dim / vocab_size all carry over
exactly — which is why both stages must have been prepared with the same
tokenizer and the same max_seq_len.

Usage:
  python3 reset_ckpt.py --src out_stage_a/ckpt.pt --dst out_stage_b
"""

import argparse, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="out_stage_a/ckpt.pt")
ap.add_argument("--dst", default="out_stage_b")
a = ap.parse_args()

ckpt = torch.load(a.src, map_location="cpu")
ma = ckpt["model_args"]
print("carrying over model_args:", ma)
print(f"source iter_num: {ckpt.get('iter_num')}  "
      f"best_val_loss: {ckpt.get('best_val_loss')}")

os.makedirs(a.dst, exist_ok=True)
torch.save({"model": ckpt["model"], "model_args": ma,
            "iter_num": 0, "best_val_loss": 1e9,
            "config": ckpt.get("config", {})},
           os.path.join(a.dst, "ckpt.pt"))
print(f"\nwrote {a.dst}/ckpt.pt with iter_num=0 (optimizer state dropped)")
print(f"Now train with --init_from=resume --out_dir={a.dst} and a LOWER "
      f"learning rate.")
