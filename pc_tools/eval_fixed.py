# eval_fixed.py — R-457 fixed yardstick
# Same files, same windows, every checkpoint. Lower = better. 
# train.py's val column is a heartbeat, THIS is the verdict.
#
# Usage:
#   python3 eval_fixed.py out27m_v6/ckpt.pt
#   python3 eval_fixed.py out27m_v5/ckpt.pt out27m_v6/ckpt.pt   (compare)
#
# NOTE: paths below assume the PRETRAIN bins live in data/tok4096_pretrain
# and the FINE-TUNE shards in data/tok4096_finetune. If you have just
# swapped folders for a training run, the script finds them either way.

import sys, os
import torch, numpy as np
from model import ModelArgs, Transformer

def find(fname, dirs=("data/tok4096_pretrain", "data/tok4096",
                      "data/tok4096_finetune")):
    for d in dirs:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

FILES = [find(f) for f in
         ("data00.bin", "data01.bin", "wiki00.bin", "shard00.bin")]
FILES = [f for f in FILES if f]

DEVICE = "mps"
WINDOWS = 20          # fixed windows per file
SEQ = 512             # window length

def score(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu")
    m = Transformer(ModelArgs(**c["model_args"]))
    sd = {k.removeprefix("_orig_mod."): v for k, v in c["model"].items()}
    m.load_state_dict(sd)
    m.eval().to(DEVICE)
    print(f"== {ckpt_path}  (iter {c.get('iter_num')})")
    for f in FILES:
        a = np.memmap(f, dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(0)   # same positions every run
        L = []
        for _ in range(WINDOWS):
            i = torch.randint(0, len(a) - SEQ - 1, (1,), generator=g).item()
            x = torch.from_numpy(a[i:i+SEQ].astype(np.int64))[None].to(DEVICE)
            y = torch.from_numpy(a[i+1:i+SEQ+1].astype(np.int64))[None].to(DEVICE)
            with torch.no_grad():
                m(x, y)
                L.append(m.last_loss.item())
        print(f"  {f:44s} loss {sum(L)/len(L):.3f}")

if __name__ == "__main__":
    ckpts = sys.argv[1:]
    if not ckpts:
        print("usage: python3 eval_fixed.py <ckpt.pt> [<ckpt2.pt> ...]")
        sys.exit(1)
    if not FILES:
        print("ERROR: no eval bins found — check the data folder names.")
        sys.exit(1)
    for ck in ckpts:
        score(ck)
