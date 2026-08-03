#!/usr/bin/env python3
"""
build_write_data.py — R-457 <write> mode dataset, built WITHOUT a teacher LLM.

WHY NO QWEN
Generating 30k paragraphs with a 7B on an M4 is ~50-70 hours and produces
text in the WRONG REGISTER: Qwen writes "consequently" and "predominantly",
words that shatter into 4-5 pieces in a 4k vocab and that the base model has
never emitted. Two better sources are already on disk:

  1. NARRATIVE  — every TinyStories record carries a GPT-4-written "summary"
     field (verified: 100000/100000 in shard 0). That IS a
     (skeleton -> paragraph) corpus, 2.1M pairs, written by a far better
     model than Qwen, and perfectly in-vocab because it is the same text the
     tokenizer and the base model were trained on. Zero generation cost.

  2. GROUNDED   — build_kb.py holds the real bank facts in the canonical
     "The {property} of {subject} is {value} {unit}." shape the device
     retrieves at runtime. Paragraphs are COMPOSED from them, so every
     sentence is grounded by construction. This is the project's own
     principle (construction beats generation) applied to prose.

  3. REFUSAL    — topic asked, no facts retrieved -> say so. Same negative-
     example discipline that fixed v8's lookup invention.

SCHEMA
Reuses the EXACT four keys construct.py emits (facts / question / reasoning /
answer) so prepare_staged.py needs no changes:

  facts     the retrieved facts, or the story skeleton
  question  "<write> ..." — the mode token does the switching
  reasoning short plan (plan-then-write), or "" for narrative
  answer    the paragraph

USAGE
  python3 build_write_data.py --n-narrative 20000 --n-grounded 8000 \
      --n-refusal 2000 --out write.jsonl
  python3 build_write_data.py --out write.jsonl --check      # verify only
"""

import argparse, glob, json, os, random, re, sys

# ---------------------------------------------------------------------------
# 1. NARRATIVE — TinyStories summary -> story. Free, GPT-4 written, in-vocab.
# ---------------------------------------------------------------------------
NARRATIVE_Q = [
    "<write> Tell this as a story.",
    "<write> Write this as a story.",
    "<write> Turn this into a story.",
    "<write> Write the story.",
]

def narrative_examples(n, data_dir, max_words, tok=None, seed=0):
    """Pull (summary -> story) pairs from the TinyStories shards.

    Targets are length-filtered so facts + paragraph fit the DEVICE window
    (256 tokens), not just the training window (512)."""
    rng = random.Random(seed)
    shards = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not shards:
        print(f"WARNING: no shards in {data_dir}, narrative set will be EMPTY")
        return []
    rng.shuffle(shards)
    out, seen = [], set()
    for sh in shards:
        if len(out) >= n:
            break
        try:
            data = json.load(open(sh, encoding="utf-8"))
        except Exception as e:
            print(f"  skip {os.path.basename(sh)}: {e}")
            continue
        rng.shuffle(data)
        for rec in data:
            if len(out) >= n:
                break
            summ = (rec.get("summary") or "").strip()
            story = (rec.get("story") or "").strip()
            if not summ or not story:
                continue
            story = " ".join(story.split())          # collapse newlines
            if len(story.split()) > max_words:
                continue
            if tok is not None and len(tok.encode(story)) > max_words * 1.6:
                continue
            if summ in seen:
                continue
            seen.add(summ)
            out.append({
                "facts": [summ],
                "question": rng.choice(NARRATIVE_Q),
                "reasoning": "",
                "answer": story,
                "type": "write_narrative",
            })
        print(f"  {os.path.basename(sh)}: running total {len(out)}")
    return out

# ---------------------------------------------------------------------------
# 2. GROUNDED — compose paragraphs from the REAL bank facts.
#    Every sentence restates a given fact. Nothing is invented, so
#    check_write can verify the whole set mechanically.
# ---------------------------------------------------------------------------
CANON = re.compile(
    r"^The (?P<p>[a-z][a-z ]*?) of (?P<s>[a-z0-9][a-z0-9 ]*?) is "
    r"(?:about )?(?P<v>[-0-9][0-9.,e+-]*) (?P<u>.+?)\.$")

OPENINGS = [
    "Here are some facts about {a}.",
    "These facts describe {a}.",
    "Some things are known about {a}.",
    "This is a short description of {a}.",
]
BODY_FORMS = [
    "The {p} of {s} is {v} {u}.",
    "{S} has {ap} of {v} {u}.",
    "Its {p} is {v} {u}.",
    "For {s}, the {p} is {v} {u}.",
]
CLOSERS = [
    "Those are the facts about {a}.",
    "All of that is stated above.",
    "",
]
GROUNDED_Q = [
    "<write> Write about {a}.",
    "<write> Write a paragraph about {a}.",
    "<write> Tell me about {a}.",
    "<write> Describe {a}.",
]
# CRITICAL: these must share NO PREFIX with construct.py's lookup reasoning,
# which begins "The facts give the {p} of {s} as ...". Lookup is ~29% of the
# fine-tune mix and grounded-write ~11%, so any shared opening lets the lookup
# continuation win: the 27M produced "The facts give the melting point of ZINC
# as 660 degrees Celsius. So the answer is..." for a <write> prompt about
# aluminium. Every plan below opens on a different token.
GROUNDED_PLAN = [
    "I will use the facts about {a}.",
    "I will describe {a} from the facts.",
    "There are facts here about {a}.",
    "I can write about {a} using these facts.",
]

def load_bank_facts(kb_path):
    """Import build_kb.py and pull its canonical facts, grouped by subject."""
    d = os.path.dirname(os.path.abspath(kb_path)) or "."
    sys.path.insert(0, d)
    modname = os.path.splitext(os.path.basename(kb_path))[0]
    try:
        mod = __import__(modname)
    except Exception as e:
        print(f"WARNING: could not import {kb_path}: {e}")
        return {}
    if not hasattr(mod, "core_facts"):
        print(f"WARNING: {kb_path} has no core_facts(), grounded set EMPTY")
        return {}
    by_subj = {}
    for _, text in mod.core_facts():
        m = CANON.match(text.strip())
        if not m:
            continue
        g = m.groupdict()
        by_subj.setdefault(g["s"], {})[g["p"]] = (g["v"], g["u"])
    return by_subj

# The real bank affords only ~150 unique (subject, property-set) combinations,
# so it cannot fill an 8k slice on its own. These mirror construct.py's lookup
# vocabulary and generate canonical facts with random values. The model is
# learning to RESTATE GIVEN FACTS, not to memorise physics, so synthetic values
# are legitimate here — exactly the argument construct.py already makes.
SYN_SUBJECTS = ["copper", "aluminium", "steel", "iron", "lead", "gold",
    "silver", "tin", "zinc", "nickel", "titanium", "brass", "water", "air",
    "silicon", "germanium", "carbon", "tungsten", "capacitor", "resistor",
    "diode", "transistor", "inductor", "thermistor", "regulator", "crystal",
    "relay", "transformer", "fuse", "battery", "motor", "sensor"]

SYN_PROPERTIES = [
    ("density", "kilograms per cubic metre", 700, 21000),
    ("melting point", "degrees Celsius", 60, 3400),
    ("boiling point", "degrees Celsius", 100, 5000),
    ("resistivity", "nano ohm metres", 15, 1500),
    ("thermal conductivity", "watts per metre kelvin", 1, 430),
    ("specific heat", "joules per kilogram kelvin", 120, 4200),
    ("atomic mass", "atomic mass units", 1, 240),
    ("forward voltage", "volts", 1, 4),
    ("capacitance", "nanofarads", 1, 1000),
    ("resistance", "ohms", 10, 10000),
    ("operating voltage", "volts", 3, 48),
    ("rated current", "amps", 1, 40),
    ("tensile strength", "megapascals", 40, 1400),
]

# Materials are mass nouns ("about copper"); components are count nouns
# ("about a fuse"). The base model has real pretrained English — feeding it
# "Tell me about fuse" would fight that instead of using it.
MASS_NOUNS = {"copper", "aluminium", "steel", "iron", "lead", "gold", "silver",
    "tin", "zinc", "nickel", "titanium", "brass", "water", "air", "silicon",
    "germanium", "carbon", "tungsten"}

def art(s):
    """'copper' -> 'copper';  'fuse' -> 'a fuse';  'inductor' -> 'an inductor'"""
    if s in MASS_NOUNS:
        return s
    return ("an " if s[0] in "aeiou" else "a ") + s

def _syn_value(rng, lo, hi):
    """Value SHAPES must vary — decimals and long integers have to appear in
    the answer slot, the v8/v9 lesson."""
    r = rng.random()
    if r < 0.30:
        places = rng.choice([1, 2, 3])
        return f"{rng.uniform(lo, min(hi, lo + 500)):.{places}f}"
    if r < 0.45:
        return str(rng.randint(1000, 99999))
    return str(rng.randint(lo, hi))

def synthetic_subject(rng):
    """One subject with 1-3 canonical (property, value, unit) triples."""
    s = rng.choice(SYN_SUBJECTS)
    k = rng.choices([1, 2, 3], weights=[30, 45, 25])[0]
    props = rng.sample(SYN_PROPERTIES, k)
    return s, {p: (_syn_value(rng, lo, hi), u) for p, u, lo, hi in props}

def grounded_examples(n, by_subj, seed=0, syn_frac=0.75):
    rng = random.Random(seed + 1)
    subjects = [s for s, props in by_subj.items() if len(props) >= 1]
    if not subjects and syn_frac < 1.0:
        print("WARNING: no canonical bank facts, using synthetic only")
        syn_frac = 1.0
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        if subjects and rng.random() > syn_frac:
            s = rng.choice(subjects)
            props = list(by_subj[s].items())
        else:
            s, pd = synthetic_subject(rng)
            props = list(pd.items())
        k = min(len(props), rng.choices([1, 2, 3], weights=[30, 45, 25])[0])
        chosen = rng.sample(props, k)

        facts = [f"The {p} of {s} is {v} {u}." for p, (v, u) in chosen]
        rng.shuffle(facts)

        sents = []
        op = rng.choice(OPENINGS)
        if op:
            sents.append(op.format(s=s, a=art(s)))
        for i, (p, (v, u)) in enumerate(chosen):
            form = rng.choice(BODY_FORMS)
            # "Its X is Y" only makes sense after the subject is named
            if form.startswith("Its") and i == 0 and not op:
                form = BODY_FORMS[0]
            sents.append(form.format(p=p, s=s, S=s.capitalize(),
                                      ap=("an " if p[0] in "aeiou" else "a ") + p,
                                      v=v, u=u))
        cl = rng.choice(CLOSERS)
        if cl:
            sents.append(cl.format(s=s, a=art(s)))

        answer = " ".join(sents)
        q = rng.choice(GROUNDED_Q).format(s=s, a=art(s))
        # dedup on facts + wording: bank combos are few, so the SAME fact set
        # may recur with different phrasing (that teaches robustness), but
        # never as a byte-identical duplicate.
        sig = "|".join(sorted(facts)) + "||" + q + "||" + answer
        if sig in seen:
            continue
        seen.add(sig)

        plist = " and ".join(p for p, _ in chosen)
        out.append({
            "facts": facts,
            "question": q,
            "reasoning": rng.choice(GROUNDED_PLAN).format(s=s, a=art(s), plist=plist),
            "answer": answer,
            "type": "write_grounded",
            "nf": len(facts),
        })
    return out

# ---------------------------------------------------------------------------
# 3. REFUSAL — topic asked, the facts are about something else.
# ---------------------------------------------------------------------------
# Also kept clear of lookup's negative reasoning ("Nothing in the facts gives
# the {p} of {s}." / "The facts do not state the {p} of {s}."). Both are
# refusals so a mix-up would be mild, but distinct openings cost nothing.
REFUSALS = [
    "I do not have facts about {a}.",
    "I was not given facts about {a}.",
    "These facts are not about {a}.",
]

def refusal_examples(n, by_subj, seed=0, syn_frac=0.75):
    rng = random.Random(seed + 2)
    subjects = list(by_subj)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        if len(subjects) >= 2 and rng.random() > syn_frac:
            want, other = rng.sample(subjects, 2)
            props = list(by_subj[other].items())
        else:
            other, pd = synthetic_subject(rng)
            want = rng.choice([x for x in SYN_SUBJECTS if x != other])
            props = list(pd.items())
        k = min(len(props), rng.choices([1, 2, 3], weights=[35, 40, 25])[0])
        chosen = rng.sample(props, k)
        facts = [f"The {p} of {other} is {v} {u}." for p, (v, u) in chosen]
        sig = "|".join(sorted(facts)) + "||" + want
        if sig in seen:
            continue
        seen.add(sig)
        out.append({
            "facts": facts,
            "question": rng.choice(GROUNDED_Q).format(s=want, a=art(want)),
            "reasoning": f"The facts are about {other}, not {want}.",
            "answer": rng.choice(REFUSALS).format(s=want, a=art(want)),
            "type": "write_refusal",
            "nf": len(facts),
        })
    return out

# ---------------------------------------------------------------------------
# 4. check_write — MECHANICAL GROUNDING GATE.
#    The failure mode for <write> is asserting things the facts do not
#    support. That is checkable: every NUMBER in the paragraph must appear in
#    the facts, and the subject must be named. Built to be testable against a
#    planted error (see --plant), because a check nobody tested is a check
#    that is not running.
# ---------------------------------------------------------------------------
NUM = re.compile(r"\d+(?:\.\d+)?")

def check_write(ex):
    """Return None if clean, else a reason string."""
    ans = ex.get("answer", "")
    facts = ex.get("facts", [])
    typ = ex.get("type", "")
    if not ans or not facts:
        return "empty answer or facts"

    if typ == "write_narrative":
        # narrative is free prose; only length and non-echo are checkable
        if ans.strip() == facts[0].strip():
            return "answer merely echoes the summary"
        return None

    fact_nums = set()
    for f in facts:
        fact_nums |= set(NUM.findall(f))
    for n in NUM.findall(ans):
        if n not in fact_nums:
            return f"UNGROUNDED NUMBER {n!r} not present in the facts"

    q = ex.get("question", "")
    m = re.search(r"(?:about|Describe|me about) (?:a |an )?([a-z0-9 ]+?)\.?$", q)
    subj = m.group(1).strip().rstrip(".") if m else None

    if typ == "write_refusal":
        if not re.search(r"\bnot\b|\bdo not\b|\bNothing\b", ans):
            return "refusal example does not refuse"
        if subj and subj not in ans:
            return f"refusal does not name the topic {subj!r}"
        return None

    if typ == "write_grounded":
        if subj and subj not in ans:
            return f"paragraph never names its subject {subj!r}"
        # every fact's value must be reported
        for f in facts:
            m2 = CANON.match(f.strip())
            if m2 and m2.group("v") not in ans:
                return f"fact value {m2.group('v')!r} never stated"
        return None
    return None

def vocab_gate(examples, vocab_path, min_frac=0.90):
    """Reject paragraphs whose words are mostly NOT single tokens in the 4k
    vocab. Qwen-style register ('consequently') shatters into 4-5 pieces and
    the base model has never produced it. Free to check, and the reason
    generated data would otherwise underperform."""
    if not os.path.exists(vocab_path):
        print(f"  (no {vocab_path}, skipping vocab gate)")
        return examples, 0
    single = set()
    for line in open(vocab_path, encoding="utf-8"):
        piece = line.split("\t")[0]
        if piece.startswith("\u2581") and len(piece) > 1:
            single.add(piece[1:].lower())
    kept, dropped = [], 0
    for ex in examples:
        words = [w.strip(".,!?\"'").lower() for w in ex["answer"].split()]
        words = [w for w in words if w and not w.isdigit()]
        if not words:
            dropped += 1
            continue
        frac = sum(1 for w in words if w in single) / len(words)
        if frac < min_frac:
            dropped += 1
            continue
        kept.append(ex)
    return kept, dropped

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="write.jsonl")
    ap.add_argument("--n-narrative", type=int, default=20000)
    ap.add_argument("--n-grounded", type=int, default=8000)
    ap.add_argument("--n-refusal", type=int, default=2000)
    ap.add_argument("--data-dir", default="data/TinyStories_all_data")
    ap.add_argument("--kb", default="build_kb.py")
    ap.add_argument("--vocab", default="data/tok4096.vocab")
    ap.add_argument("--tokenizer", default="data/tok4096.model")
    ap.add_argument("--max-words", type=int, default=110,
                    help="narrative target cap; device window is 256 tokens")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true",
                    help="verify an existing --out file instead of building")
    ap.add_argument("--plant", action="store_true",
                    help="plant one ungrounded number and confirm the check "
                         "catches it (verify the verifier)")
    a = ap.parse_args()

    if a.check:
        rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        bad = 0
        for ex in rows:
            r = check_write(ex)
            if r:
                bad += 1
                if bad <= 10:
                    print(f"  [{r}] {ex.get('question','')}")
        print(f"checked {len(rows)}: {bad} failed")
        return

    tok = None
    if os.path.exists(a.tokenizer):
        try:
            import sentencepiece as spm
            tok = spm.SentencePieceProcessor(model_file=a.tokenizer)
        except Exception as e:
            print(f"  (tokenizer unavailable: {e})")

    print("narrative (TinyStories summary -> story):")
    narr = narrative_examples(a.n_narrative, a.data_dir, a.max_words, tok,
                              a.seed)
    print(f"  got {len(narr)}")

    print("grounded (composed from the real bank):")
    by_subj = load_bank_facts(a.kb)
    print(f"  {len(by_subj)} subjects with canonical facts")
    grnd = grounded_examples(a.n_grounded, by_subj, a.seed)
    print(f"  got {len(grnd)}")

    print("refusal:")
    refu = refusal_examples(a.n_refusal, by_subj, a.seed)
    print(f"  got {len(refu)}")

    rows = narr + grnd + refu

    print("vocab gate:")
    rows, dropped = vocab_gate(rows, a.vocab)
    print(f"  dropped {dropped} out-of-register examples")

    print("grounding gate (check_write):")
    clean, bad = [], 0
    for ex in rows:
        r = check_write(ex)
        if r is None:
            clean.append(ex)
        else:
            bad += 1
            if bad <= 8:
                print(f"  [{r}]")
    print(f"  rejected {bad}, kept {len(clean)}")

    if a.plant:
        print("verify the verifier:")
        victim = next(e for e in clean if e["type"] == "write_grounded")
        planted = dict(victim)
        planted["answer"] = victim["answer"] + " Its price is 4242 dollars."
        r = check_write(planted)
        print(f"  planted ungrounded number -> {r or 'NOT CAUGHT (BAD)'}")
        victim2 = next(e for e in clean if e["type"] == "write_refusal")
        planted2 = dict(victim2)
        planted2["answer"] = "Here are some facts."
        r2 = check_write(planted2)
        print(f"  planted non-refusing refusal -> {r2 or 'NOT CAUGHT (BAD)'}")

    random.Random(a.seed).shuffle(clean)
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(e["type"] for e in clean)
    print(f"\nwrote {len(clean)} examples to {a.out}")
    for k, v in c.most_common():
        print(f"  {k}: {v} ({100*v//max(1,len(clean))}%)")

if __name__ == "__main__":
    main()
