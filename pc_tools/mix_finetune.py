#!/usr/bin/env python3
"""
mix_finetune.py — build the Phase 5 fine-tune file for the 27M.

Combines two streams and stamps each with its MODE TOKEN, so one model does
both jobs and the prefix does the switching:

  <reason>  construct.py output — the ten R-457 reasoning types, tools,
            refusals. Unchanged data; the prefix is added here so
            construct.py needs no edit (and stays byte-identical to the file
            that trained v9d, which is worth keeping).
  <write>   build_write_data.py output — narrative, grounded, refusal.
            Already carries its own "<write> ..." questions.

WHY NO SEPARATE "raw TinyStories" SLICE
The original plan reserved ~15% for raw stories to stop fine-tuning from
erasing the pretrained prose. That is already covered: write_narrative IS
TinyStories (summary -> story), and it is the majority of write.jsonl. Adding
raw text on top would spend budget teaching a skill the mix already carries.

PAIRING
--pair emits, for a share of the grounded write examples, the SAME fact set
in <reason> form too (a lookup-style question over those facts). That teaches
mode as a switch over identical input rather than as two unrelated tasks.

USAGE
  python3 mix_finetune.py --reason clean.jsonl --write write.jsonl \
      --out finetune.jsonl --reason-pct 60
"""

import argparse, json, random, re
from collections import Counter

CANON = re.compile(
    r"^The (?P<p>[a-z][a-z ]*?) of (?P<s>[a-z0-9][a-z0-9 ]*?) is "
    r"(?:about )?(?P<v>[-0-9][0-9.,e+-]*) (?P<u>.+?)\.$")

LOOKUP_Q = ["What is the {p} of {s}?", "Give the {p} of {s}.",
            "How much is the {p} of {s}?"]
LOOKUP_R = ["The facts give the {p} of {s} as {v} {u}.",
            "The facts state that the {p} of {s} is {v} {u}."]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def stamp_reason(rows):
    """Prefix every reasoning question with the <reason> mode token."""
    out = []
    for o in rows:
        o = dict(o)
        q = o.get("question", "")
        if not q.startswith("<reason>"):
            o["question"] = "<reason> " + q
        out.append(o)
    return out


def paired_reason(write_rows, rng, frac):
    """For some grounded write examples, emit the same facts as a <reason>
    lookup. Same input, two modes — that is what makes the prefix meaningful."""
    out = []
    for o in write_rows:
        if o.get("type") != "write_grounded":
            continue
        if rng.random() > frac:
            continue
        facts = o["facts"]
        target = None
        for f in facts:
            m = CANON.match(f.strip())
            if m:
                target = m.groupdict()
                break
        if not target:
            continue
        p, s, v, u = target["p"], target["s"], target["v"], target["u"]
        out.append({
            "facts": list(facts),
            "question": "<reason> " + rng.choice(LOOKUP_Q).format(p=p, s=s),
            "reasoning": rng.choice(LOOKUP_R).format(p=p, s=s, v=v, u=u)
                         + f" So the answer is {v} {u}.",
            "answer": f"{v} {u}",
            "type": "lookup",
            "tool": "lookup",
            "nf": len(facts),
            "paired": 1,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="../R-457_stageC/clean.jsonl")
    ap.add_argument("--write", default="write.jsonl")
    ap.add_argument("--out", default="finetune.jsonl")
    ap.add_argument("--reason-pct", type=int, default=60,
                    help="target share of <reason> examples in the output")
    ap.add_argument("--pair-frac", type=float, default=0.15,
                    help="share of grounded write examples that also get a "
                         "<reason> twin over the same facts")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    reason = stamp_reason(load(a.reason))
    write = load(a.write)
    print(f"reason pool: {len(reason):,}")
    print(f"write  pool: {len(write):,}")

    pairs = paired_reason(write, rng, a.pair_frac)
    print(f"paired <reason> twins: {len(pairs):,}")
    reason = reason + pairs

    # size the reason slice so the final mix hits --reason-pct, using ALL of
    # the write data (it is the scarce, newly built half)
    want_reason = int(len(write) * a.reason_pct / max(1, 100 - a.reason_pct))
    if want_reason > len(reason):
        print(f"NOTE: wanted {want_reason:,} reason examples, pool has "
              f"{len(reason):,} — using the whole pool")
        want_reason = len(reason)
    rng.shuffle(reason)
    rows = reason[:want_reason] + write
    rng.shuffle(rows)

    # sanity: every row must carry exactly one mode token, at the front
    bad = 0
    for o in rows:
        q = o.get("question", "")
        n = q.count("<reason>") + q.count("<write>")
        if n != 1 or not (q.startswith("<reason>") or q.startswith("<write>")):
            bad += 1
    if bad:
        raise SystemExit(f"ABORT: {bad} rows have a bad mode token")

    with open(a.out, "w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    mode = Counter("write" if o["question"].startswith("<write>") else "reason"
                   for o in rows)
    print(f"\nwrote {len(rows):,} examples to {a.out}")
    for k, v in mode.most_common():
        print(f"  <{k}>: {v:,} ({100*v//len(rows)}%)")
    print("  by type:")
    for k, v in Counter(o.get("type", "?") for o in rows).most_common():
        print(f"    {k}: {v:,}")


if __name__ == "__main__":
    main()
