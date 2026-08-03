#!/usr/bin/env python3
"""
tokenize_finetune32k.py — turn finetune32k.jsonl into shard*.bin using the
EXISTING data/tok32768.model.

Why this exists: prepare_r457.py and prepare_staged.py both call
SentencePieceTrainer unconditionally, which would OVERWRITE data/tok32768.model
and orphan every checkpoint trained against it. This script only ever reads
the tokenizer.

Format mirrors to_text() exactly (the format the 27M/ft6 trained on and the
device firmware now emits byte-for-byte):

    <mode>\\nFacts: <f1> <f2> ...\\nQuestion: <q>\\nReasoning: <r>\\nAnswer: <a>

Each example is prefixed with BOS (id 1), matching process_shard.

Usage:
    python3 tokenize_finetune32k.py
    python3 tokenize_finetune32k.py --in finetune32k.jsonl --shards 5
"""
import argparse, json, os
import numpy as np
import sentencepiece as spm

def to_text(ex):
    """Serialize one example. Mode token comes from the record; default reason."""
    mode = ex.get("mode")
    if not mode:
        q = ex.get("question", "")
        mode = "<write>" if q.strip().startswith("<write>") else "<reason>"
    # a question may already carry its mode prefix from mix_finetune; strip it
    q = ex.get("question", "").strip()
    for m in ("<reason>", "<write>"):
        if q.startswith(m):
            mode = m
            q = q[len(m):].strip()
    facts = ex.get("facts", [])
    if isinstance(facts, str):
        facts = [facts]
    fact_str = " ".join(f.strip() for f in facts)
    r = ex.get("reasoning", "").strip()
    a = ex.get("answer", "").strip()
    return (f"{mode}\nFacts: {fact_str}\nQuestion: {q}\nReasoning: {r}\nAnswer: {a}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="finetune32k.jsonl")
    ap.add_argument("--model", default="data/tok32768.model")
    ap.add_argument("--outdir", default="data/tok32768")
    ap.add_argument("--shards", type=int, default=5)
    a = ap.parse_args()

    assert os.path.exists(a.model), f"missing {a.model}"
    sp = spm.SentencePieceProcessor(model_file=a.model)
    os.makedirs(a.outdir, exist_ok=True)

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    print(f"{len(rows):,} examples from {a.inp}")

    texts = [to_text(r) for r in rows]
    print("--- first example as the model will see it ---")
    print(texts[0][:400])
    print("--- ids ---", sp.encode(texts[0])[:12])

    per = (len(texts) + a.shards - 1) // a.shards
    total = 0
    for s in range(a.shards):
        chunk = texts[s*per:(s+1)*per]
        if not chunk:
            break
        toks = []
        SPACE = sp.piece_to_id("\u2581")
        for ids in sp.encode(chunk):
            if ids and ids[0] == SPACE:
                ids = ids[1:]       # drop sentencepiece's dummy prefix:
                                    # pretrain shards don't have it either
            toks.append(1)          # BOS per example
            toks.extend(ids)
        arr = np.array(toks, dtype=np.uint16)
        p = os.path.join(a.outdir, f"shard{s:02d}.bin")
        arr.tofile(p)
        total += len(arr)
        print(f"  {p}  {len(chunk):,} examples, {len(arr):,} tokens")
    print(f"total {total:,} tokens")
    lens = [len(x) for x in sp.encode(texts[:5000])]
    lens.sort()
    print(f"example length: median {lens[len(lens)//2]}, "
          f"p99 {lens[int(len(lens)*0.99)]}, max {lens[-1]}  (seq_len 256)")

if __name__ == "__main__":
    main()
