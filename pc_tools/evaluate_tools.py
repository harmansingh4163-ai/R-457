#!/usr/bin/env python3
"""
evaluate_tools.py — evaluate an R-457 checkpoint WITH the tools active.

evaluate_r457.py lets the model guess the digits after "=", so every tool-using
type (arithmetic, counting, physics, lisp, forth) scores at a floor. This script
intercepts <calc> and <count> during generation exactly as the firmware does:
when the text ends with "=" inside an unclosed tag, it computes the result,
appends "<result></tag>", and continues generating from there.

That makes the number it reports the DEVICE's accuracy, not a lower bound.

Usage (inside llama2.c, after training):
  python3 evaluate_tools.py --ckpt out_v6/ckpt.pt --vocab-size 1024
  python3 evaluate_tools.py --ckpt out_v6/ckpt.pt --vocab-size 1024 \
      --n 200 --show 4 --no-tools     # compare against the floor
"""

import argparse, glob, json, os, re, sys
# pc_tools/ sits beside train/, where model.py & tokenizer.py live (A-3).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "train"))

# ---------------------------------------------------------------------------
# tool logic — mirrors pending_tool() in esp32_r457.ino
# ---------------------------------------------------------------------------
def pending_tool(buf):
    """If buf ends with '=' inside an unclosed <calc>/<count>, return the text
    to inject (result + closing tag). Otherwise None."""
    if not buf.endswith("="):
        return None
    open_i, tag = -1, None
    for m in re.finditer(r"<calc>", buf):
        open_i, tag = m.start(), "calc"
    for m in re.finditer(r"<count>", buf):
        if m.start() > open_i:
            open_i, tag = m.start(), "count"
    if open_i < 0:
        return None
    rest = buf[open_i:]
    if "</calc>" in rest or "</count>" in rest:
        return None                                    # already closed

    if tag == "calc":
        m = re.match(r"<calc>\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*=$", rest)
        if not m:
            return None
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "+": r = a + b
        elif op == "-": r = a - b
        elif op == "*": r = a * b
        else:
            if b == 0:
                return None
            r = a // b
        return f"{r}</calc>"

    m = re.match(r"<count>\s*([A-Za-z]+)\s*(?:,\s*([A-Za-z])\s*)?=$", rest)
    if not m:
        return None
    word, ch = m.group(1), m.group(2)
    n = word.count(ch) if ch else len(word)
    return f"{n}</count>"


def _self_test():
    cases = [
        (" ... <calc>47+2=", "49</calc>"),
        (" ... <calc>29-13=", "16</calc>"),
        (" ... <calc>6*9=", "54</calc>"),
        (" count letters in strawberry. <count>strawberry,r=", "3</count>"),
        (" <count>banana=", "6</count>"),
        (" <calc>1+1=2</calc> then <count>apple,p=", "2</count>"),
        (" <calc>12+7=19</calc> So the answer is 19.", None),
        (" <calc>7/0=", None),
        (" no tags here", None),
    ]
    bad = 0
    for text, want in cases:
        got = pending_tool(text)
        if got != want:
            bad += 1
            print(f"  FAIL {text!r} -> {got!r} (want {want!r})")
    print(f"tool self-test: {len(cases)-bad}/{len(cases)} pass")
    return bad == 0


# ---------------------------------------------------------------------------
def norm_answer(s):
    s = s.strip().lower().rstrip(".")
    s = re.sub(r"^(a|an|the)\s+", "", s)
    if "cannot" in s or "determined" in s:
        return "cannot be determined"
    return s


def build_prompt(o):
    q = o['question']
    mode = ""
    for mm in ("<reason>", "<write>"):
        if q.startswith(mm):
            mode, q = mm + "\n", q[len(mm):].strip()
            break
    if not mode:
        mode = "<reason>\n"
    return (mode
            + f"Facts: {' '.join(o['facts'])}\n"
            f"Question: {q}\n"
            f"Reasoning:")


def extract_answer(text):
    m = re.search(r"Answer:\s*(.+?)(?:\n|Facts:|$)", text, re.S)
    return m.group(1).strip().split("\n")[0] if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_v6/ckpt.pt")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--no-tools", action="store_true",
                    help="disable interception, to reproduce the floor")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    _self_test()

    import torch
    from model import ModelArgs, Transformer
    from tokenizer import Tokenizer

    device = a.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    print(f"device: {device}   tools: {'OFF' if a.no_tools else 'ON'}")

    ckpt = torch.load(a.ckpt, map_location=device)
    model = Transformer(ModelArgs(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    print(f"loaded {a.ckpt} (iter {ckpt.get('iter_num','?')}, "
          f"val loss {ckpt.get('best_val_loss', float('nan')):.4f})")

    enc = Tokenizer(f"data/tok{a.vocab_size}.model")
    max_seq = ckpt["model_args"]["max_seq_len"]

    # The tokenizer prepends a dummy space. Learn its id so injected tool
    # results start at a real token boundary — same trick the firmware uses.
    probe = enc.encode("9", bos=False, eos=False)
    space_tok = probe[0] if len(probe) == 2 else None

    def generate(prompt):
        """Greedy generation over TOKEN IDS.

        Critical: never accumulate text by decoding one token at a time —
        sentencepiece strips the leading space marker from a single-token
        decode, which silently deletes every space and corrupts the context on
        re-encode. Keep the ids, decode the whole list for the text view, and
        append tool results as ids.
        """
        ids = enc.encode(prompt, bos=True, eos=False)
        n_prompt = len(ids)
        calls = 0
        for _ in range(a.max_new):
            if len(ids) >= max_seq:
                break
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=1, temperature=0.0)
            nxt = int(out[0, -1].item())
            if nxt in (1, 2):
                break
            ids.append(nxt)

            text = enc.decode(ids[n_prompt:])
            if not a.no_tools:
                res = pending_tool(text)
                if res:
                    inj = enc.encode(res, bos=False, eos=False)
                    if space_tok is not None and inj and inj[0] == space_tok:
                        inj = inj[1:]
                    ids.extend(inj)
                    calls += 1
                    text = enc.decode(ids[n_prompt:])
            # stop once the Answer line is complete
            m = re.search(r"Answer:\s*\S+", text)
            if m and (text.endswith("\n") or "\nFacts:" in text
                      or text.rstrip().endswith(".")):
                break
        return enc.decode(ids[n_prompt:]), calls

    files = sorted(glob.glob(os.path.join(a.eval_dir, "*.prompts.jsonl")))
    if not files:
        raise SystemExit(f"no *.prompts.jsonl in {a.eval_dir}")

    summary, total_calls = {}, 0
    for pf in files:
        name = os.path.basename(pf).replace(".prompts.jsonl", "")
        exs = [json.loads(l) for l in open(pf)][:a.n]
        correct, by_type, shown = 0, {}, 0
        for o in exs:
            gen, calls = generate(build_prompt(o))
            total_calls += calls
            pred = norm_answer(extract_answer(gen))
            gold = norm_answer(o["answer"])
            ok = pred == gold
            correct += ok
            t = o.get("type", "?")
            by_type.setdefault(t, [0, 0])
            by_type[t][0] += ok
            by_type[t][1] += 1
            if shown < a.show:
                shown += 1
                print(f"\n  [{name}] {o['question']}")
                print(f"    gold: {o['answer']!r}  pred: {pred!r}  "
                      f"{'OK' if ok else 'WRONG'}  ({calls} tool call(s))")
                rm = re.search(r"Reasoning:(.*?)(?:Answer:|$)", gen, re.S)
                if rm:
                    print(f"    reasoning: {rm.group(1).strip()[:200]}")
        acc = 100.0 * correct / max(1, len(exs))
        summary[name] = acc
        print(f"\n=== {name}: {acc:.1f}%  ({correct}/{len(exs)}) ===")
        for t, (c, n) in sorted(by_type.items()):
            print(f"    {t:14} {100.0*c/n:5.1f}%  ({c}/{n})")

    print("\n================ SUMMARY ================")
    for k in sorted(summary):
        print(f"  {k:22} {summary[k]:5.1f}%")
    print(f"\n  total tool calls made: {total_calls}")
    print("  (with --no-tools these same sets score at the software FLOOR;"
          "\n   the difference is what the chip contributes)")


if __name__ == "__main__":
    main()
