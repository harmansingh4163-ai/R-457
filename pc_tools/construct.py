#!/usr/bin/env python3
"""
construct.py v2.1 — R-457 programmatic dataset generator.

Builds guaranteed-correct reasoning examples directly from valid logical
structures. v2 added:
  - PHRASING VARIANTS (several wordings per type) so the model learns the
    logic, not one sentence template
  - --phrasing-set train|alt   (alt = different wordings, for the reworded
    held-out test)
  - --axes / --exclude-axes    (hold out a whole comparison axis, e.g. train
    without heavier/lighter, test on it)
  - --chain-lengths 3 or 3,4   (hold out chain length 4 for a length-
    generalization test)
  - irrelevant-fact injection (~1 in 6) with reasoning that notes it
  - --sort-curriculum          (order easy -> hard; note: only matters if the
    trainer reads sequentially — llama2.c samples randomly by default)

v2.1 (this file) rewrites make_lookup only:
  - fact COUNT is drawn from ONE distribution (1-4) for BOTH positive and
    negative examples, so "how many facts" carries zero signal about the
    answer. The old code built negatives with 1-2 facts and positives with
    2-4, and "single fact retrieved" became a shortcut cue for refusal —
    the false-refusal failure seen on hardware.
  - single-fact POSITIVES now exist (~21% of lookup examples), which is
    exactly the /kbn 1 hardware case.
  - the {op} negative template ("gives the X but not the Y") is only used
    when a same-subject fact is actually present, so "gives the atomic mass
    but not the atomic mass" cannot be learned.
  - the atomic-mass-vs-atomic-number value trap from v9 is kept.
  - each lookup example carries "nf": fact count, for slicing evals.

Types: transitive / syllogism / undetermined / negation / arithmetic /
counting / physics / lisp / forth / lookup.
Output: JSON Lines. Extra keys ("type","axis","k","nf") are metadata;
trainers ignore them.

Recipes (see build_datasets.sh):
  TRAIN     : --n 100000 --exclude-axes heavier --chain-lengths 3 --seed 1
  IID test  : same filters, --seed 999
  AXIS test : --axes heavier --mix 60,0,40,0 --seed 888
  LEN test  : --chain-lengths 4 --exclude-axes heavier --mix 100,0,0,0 --seed 777
  REWORDED  : same filters as TRAIN, --phrasing-set alt --seed 555
"""

import json, random, argparse

NONSENSE = [
    "glork","wug","zorb","plip","sop","dax","vimp","snod","frop","blizzle",
    "quor","frap","gleeb","wozzle","trin","splonk","zarn","blon","krit","flump",
    "drupe","klimp","vurn","sprock","nim","pol","rax","sed","mo","bo","kel",
    "dov","pim","vera","nils","tomo","blix","grud","yorp","wibble","snork",
    "plov","drak","quib","fenn","gorp","lurk","mell","narp","oont","prit",
]

# (positive, negative, superlative_max, superlative_min)
AXES = [
    ("taller","shorter","tallest","shortest"),
    ("older","younger","oldest","youngest"),
    ("bigger","smaller","biggest","smallest"),
    ("faster","slower","fastest","slowest"),
    ("heavier","lighter","heaviest","lightest"),
    ("stronger","weaker","strongest","weakest"),
    ("wider","narrower","widest","narrowest"),
    ("louder","quieter","loudest","quietest"),
    ("hotter","colder","hottest","coldest"),
    ("brighter","dimmer","brightest","dimmest"),
    ("smoother","rougher","smoothest","roughest"),
    ("sharper","duller","sharpest","dullest"),
    ("richer","poorer","richest","poorest"),
    ("deeper","shallower","deepest","shallowest"),
]

PROPS = ["blue","round","shiny","warm","glowing","soft","striped","spotted",
         "fuzzy","smooth","bright","quiet","loud"]
CLASSES = ["glork","wug","zorb","plip","dax","vimp","snod","frop","quor",
           "frap","gleeb","zarn","blon","krit","drupe","klimp","vurn",
           "sprock","yorp","blix"]

# ---------------------------------------------------------------------------
# phrasing sets — 'train' for the main dataset, 'alt' for the reworded holdout
# All fact/question phrasings keep the "X <verb> <comparative> than Y" core so
# the ruthless verifier can still machine-check them.
# ---------------------------------------------------------------------------
P = {
  "train": {
    "fact":      ["The {a} is {c} than the {b}.",
                  "A {a} is {c} than a {b}.",
                  "{A} is {c} than {b}.",
                  "It is true that the {a} is {c} than the {b}.",
                  "Remember that the {a} is {c} than the {b}.",
                  "We are told the {a} is {c} than the {b}.",
                  "The {a} happens to be {c} than the {b}.",
                  "Every {a} is {c} than every {b}."],
    "pair_q":    ["Is the {x} {c} than the {y}?",
                  "Is a {x} {c} than a {y}?",
                  "Is it true that the {x} is {c} than the {y}?",
                  "Would you say the {x} is {c} than the {y}?",
                  "Can we say the {x} is {c} than the {y}?",
                  "Is {x} {c} than {y}?"],
    "sup_q":     ["Which is the {s}?",
                  "Which one is the {s}?",
                  "Of these, which is the {s}?",
                  "Which of them is the {s}?",
                  "Who is the {s}?"],
    "reason_open": ["We know ", ""],
    "sup_reason": ["Following the order, the {top} is {c} than all the others.",
                   "Putting the comparisons in order, the {top} is {c} than all the others."],
    "undet_reason": [
      "We know how the {a} and the {b} each compare to the {m}, but the facts never compare the {a} and the {b} directly.",
      "Both the {a} and the {b} are compared to the {m}, but not to each other.",
    ],
    "universal":      ["All {x}s are {y}s.", "Every {x} is a {y}."],
    "universal_prop": ["All {x}s are {p}.", "Every {x} is {p}."],
    "neg_reason": ["All {c}s are {p}. {E} is not {p}. So {E} cannot be a {c}.",
                   "Every {c} is {p}. {E} is not {p}. So {E} is not a {c}."],
  },
  "alt": {
    "fact":      ["It is true that the {a} is {c} than the {b}.",
                  "Remember that the {a} is {c} than the {b}.",
                  "The {a} is {c} than the {b}."],
    "pair_q":    ["Is it true that the {x} is {c} than the {y}?",
                  "Would you say the {x} is {c} than the {y}?"],
    "sup_q":     ["Of these, which is the {s}?",
                  "Who is the {s}?"],
    "reason_open": ["From the facts, ", "Putting the facts together, "],
    "sup_reason": ["Walking the chain, the {top} is {c} than all the others."],
    "undet_reason": [
      "The facts tell us about the {a} and the {m}, and about the {b} and the {m}, but nothing compares the {a} and the {b}.",
    ],
    "universal":      ["Each {x} is a {y}.", "All {x}s are {y}s."],
    "universal_prop": ["Each {x} is {p}.", "All {x}s are {p}."],
    "neg_reason": ["Every {c} is {p}, but {E} is not {p}. So {E} is not a {c}."],
  },
}

def pick(ps, key, **kw):
    t = random.choice(P[ps][key])
    # {A}/{E} capitalized variants
    kw2 = dict(kw)
    for k, v in kw.items():
        kw2[k.capitalize() if len(k) == 1 else k] = v
        if len(k) == 1:
            kw2[k.upper()] = v.capitalize()
    return t.format(**kw2)

def names(k):
    return random.sample(NONSENSE, k)

def maybe_irrelevant(facts, used, reasoning_parts, ps):
    """~1 in 6: add one irrelevant property fact + a reasoning note. Only if
    room under the 4-fact cap."""
    if len(facts) >= 4 or random.random() > 1/6:
        return facts, reasoning_parts
    w = random.choice([n for n in NONSENSE if n not in used])
    prop = random.choice(PROPS)
    facts = facts + [f"The {w} is {prop}."]
    reasoning_parts = reasoning_parts + [
        f"The fact about the {w} does not matter here."]
    return facts, reasoning_parts

# ---------------------------------------------------------------------------
def make_transitive(axes, klist, ps):
    pos, neg, smax, smin = random.choice(axes)
    k = random.choice(klist)
    items = names(k)                      # items[0] > ... > items[k-1] on pos
    facts = []
    for i in range(k - 1):
        hi, lo = items[i], items[i + 1]
        if random.random() < 0.5:
            facts.append(pick(ps, "fact", a=hi, c=pos, b=lo))
        else:
            facts.append(pick(ps, "fact", a=lo, c=neg, b=hi))

    qtype = random.choice(["pair", "pair", "sup_max", "sup_min"])
    if qtype == "pair":
        i, j = sorted(random.sample(range(k), 2))
        hi, lo = items[i], items[j]
        steps = ", and ".join(
            f"the {items[t]} is {pos} than the {items[t+1]}"
            for t in range(i, j))
        opener = random.choice(P[ps]["reason_open"])
        if random.random() < 0.5:
            q = pick(ps, "pair_q", x=hi, c=pos, y=lo)
            ans, concl = "Yes", f"the {hi} is {pos} than the {lo}"
        else:
            q = pick(ps, "pair_q", x=lo, c=pos, y=hi)
            ans, concl = "No", f"the {lo} is not {pos} than the {hi}"
        parts = [f"{opener}{steps}." if opener else f"{steps[0].upper()}{steps[1:]}."]
        facts2, parts = maybe_irrelevant(facts, set(items), parts, ps)
        reasoning = " ".join(parts) + f" So {concl}."
        random.shuffle(facts2)
        return facts2, q, reasoning, ans, {"axis": pos, "k": k}
    else:
        # superlative: rebuild facts as clean positive chain (unambiguous order)
        facts = [pick(ps, "fact", a=items[i], c=pos, b=items[i+1])
                 for i in range(k - 1)]
        if qtype == "sup_max":
            q = pick(ps, "sup_q", s=smax)
            top, ans, c = items[0], items[0].capitalize(), pos
        else:
            q = pick(ps, "sup_q", s=smin)
            top, ans, c = items[-1], items[-1].capitalize(), neg
        parts = [pick(ps, "sup_reason", top=top, c=c)]
        facts2, parts = maybe_irrelevant(facts, set(items), parts, ps)
        reasoning = " ".join(parts) + f" So the {top} is the " + \
                    (smax if qtype == "sup_max" else smin) + "."
        random.shuffle(facts2)
        return facts2, q, reasoning, ans, {"axis": pos, "k": k}

def make_syllogism(ps):
    depth = random.choice([2, 2, 3])
    chain = random.sample(CLASSES, depth + 1)
    entity = random.choice([n for n in NONSENSE if n not in chain])
    E = entity.capitalize()
    facts = [pick(ps, "universal", x=chain[i], y=chain[i+1])
             for i in range(depth)]
    facts.append(f"{E} is a {chain[0]}.")

    use_property = random.random() >= 0.6 and depth <= 2
    if not use_property:
        t_idx = random.randint(1, depth)
        target = chain[t_idx]
        q = f"Is {E} a {target}?"
        ans = "Yes"
        steps = [f"{E} is a {chain[0]}"]
        for i in range(t_idx):
            steps.append(f"all {chain[i]}s are {chain[i+1]}s")
        parts = [". ".join(steps) + "."]
        facts2, parts = maybe_irrelevant(facts, set(chain) | {entity}, parts, ps)
        reasoning = " ".join(parts) + f" So {E} is a {target}."
        random.shuffle(facts2)
        return facts2, q, reasoning, ans, {}
    else:
        prop = random.choice(PROPS)
        facts.append(pick(ps, "universal_prop", x=chain[-1], p=prop))
        q = f"Is {E} {prop}?"
        ans = "Yes"
        parts = [f"{E} is a {chain[0]}, so following the chain {E} is a "
                 f"{chain[-1]}. All {chain[-1]}s are {prop}."]
        reasoning = " ".join(parts) + f" So {E} is {prop}."
        random.shuffle(facts)
        return facts, q, reasoning, ans, {}

def make_undetermined(axes, ps):
    pos, neg, smax, smin = random.choice(axes)
    a, mid, b = names(3)
    if random.random() < 0.5:
        facts = [pick(ps, "fact", a=a, c=pos, b=mid),
                 pick(ps, "fact", a=b, c=pos, b=mid)]
    else:
        facts = [pick(ps, "fact", a=a, c=neg, b=mid),
                 pick(ps, "fact", a=b, c=neg, b=mid)]
    if random.random() < 0.5:
        q = pick(ps, "pair_q", x=a, c=pos, y=b)
    else:
        q = pick(ps, "pair_q", x=b, c=pos, y=a)
    core = random.choice(P[ps]["undet_reason"]).format(a=a, b=b, m=mid)
    parts = [core]
    facts2, parts = maybe_irrelevant(facts, {a, mid, b}, parts, ps)
    reasoning = " ".join(parts) + " So it cannot be determined."
    random.shuffle(facts2)
    return facts2, q, reasoning, "Cannot be determined", {"axis": pos}

def make_negation(ps):
    cls = random.choice(CLASSES)
    prop = random.choice(PROPS)
    entity = random.choice([n for n in NONSENSE if n != cls])
    E = entity.capitalize()
    facts = [pick(ps, "universal_prop", x=cls, p=prop),
             f"{E} is not {prop}."]
    q = f"Is {E} a {cls}?"
    reasoning = random.choice(P[ps]["neg_reason"]).format(c=cls, p=prop, E=E)
    parts = [reasoning]
    facts2, parts2 = maybe_irrelevant(facts, {cls, prop, entity}, [], ps)
    if parts2:  # irrelevant note goes before the templated reasoning's "So"
        so_idx = reasoning.rfind(" So ")
        reasoning = reasoning[:so_idx] + " " + parts2[0] + reasoning[so_idx:]
    random.shuffle(facts2)
    return facts2, q, reasoning, "No", {}


# ---------------------------------------------------------------------------
# 5. ARITHMETIC via TOOL CALL  — the model emits <calc>expr=result</calc>.
#    On device, generation stops after '=' and the chip supplies the result.
# ---------------------------------------------------------------------------
COUNT_NOUNS = ["gleebs", "zorbs", "trins", "sprocks", "flumps", "krits",
               "wugs", "plips", "snods", "fraps", "blons", "vurns"]

def make_arithmetic(ps):
    holder_a, holder_b = names(2)
    A, B = holder_a.capitalize(), holder_b.capitalize()
    noun = random.choice([n for n in COUNT_NOUNS
                          if n[:-1] not in (holder_a, holder_b)])
    kind = random.choice(["add", "sub", "mul", "add", "sub"])

    if kind == "add":
        x, y = random.randint(2, 60), random.randint(2, 40)
        facts = [f"{A} has {x} {noun}.", f"{B} has {y} {noun}."]
        q = random.choice([f"How many {noun} do they have together?",
                           f"How many {noun} are there in total?"])
        expr, res = f"{x}+{y}", x + y
        lead = f"{A} has {x} {noun} and {B} has {y} {noun}."
    elif kind == "sub":
        x = random.randint(10, 90); y = random.randint(2, x - 1)
        facts = [f"{A} has {x} {noun}.", f"{A} gives {y} {noun} to {B}."]
        q = random.choice([f"How many {noun} does {A} have now?",
                           f"How many {noun} are left with {A}?"])
        expr, res = f"{x}-{y}", x - y
        lead = f"{A} started with {x} {noun} and gave away {y}."
    else:
        x, y = random.randint(2, 12), random.randint(2, 9)
        facts = [f"Each {holder_a} has {y} {noun}.", f"There are {x} {holder_a}s."]
        q = f"How many {noun} are there in total?"
        expr, res = f"{x}*{y}", x * y
        lead = f"There are {x} {holder_a}s and each has {y} {noun}."

    parts = [lead]
    facts2, parts = maybe_irrelevant(facts, {holder_a, holder_b}, parts, ps)
    reasoning = " ".join(parts) + f" <calc>{expr}={res}</calc> So the answer is {res}."
    random.shuffle(facts2)
    return facts2, q, reasoning, str(res), {"calc": expr}


# ---------------------------------------------------------------------------
# 6. COUNTING via TOOL CALL — the "how many r in strawberry" failure.
#    A tokenizer never sees individual letters, so counting them from memory is
#    guesswork. The model emits <count>strawberry,r=3</count>; on device the
#    chip runs an exact loop over the characters. The model's only job is to
#    COPY the word into the tag — which attention can do reliably.
# ---------------------------------------------------------------------------
REAL_WORDS = [
    "strawberry","banana","mississippi","bookkeeper","committee","balloon",
    "success","possession","assessment","tomorrow","parallel","necessary",
    "occurrence","embarrass","millennium","questionnaire","broccoli",
    "raspberry","butterfly","elephant","umbrella","chocolate","dinosaur",
    "telephone","alligator","kangaroo","helicopter","refrigerator","vegetable",
    "aluminium","cinnamon","pineapple","watermelon","sunflower","waterfall",
    "aardvark","abacus","acrobat","adventure","airplane","alphabet","anchor",
    "antelope","apartment","arithmetic","astronaut","avalanche","backpack",
    "bakery","bamboo","bandage","basketball","battery","beetle","bicycle",
    "biscuit","blanket","blizzard","blossom","bracelet","breakfast","bridge",
    "buffalo","bulldozer","cabbage","calendar","camera","campfire","candle",
    "canyon","caterpillar","cauliflower","ceiling","cellar","chimney",
    "cinema","circus","cliff","closet","compass","computer","concert",
    "corridor","cottage","crocodile","cucumber","cupboard","curtain",
    "daffodil","daughter","december","dessert","diamond","dictionary",
    "dolphin","doorway","dragonfly","drummer","eagle","earthquake","eggplant",
    "engine","envelope","escalator","evening","fabric","factory","feather",
    "festival","fireplace","flamingo","flashlight","forest","fountain",
    "furniture","garage","garden","giraffe","glacier","goldfish","grapefruit",
    "grasshopper","greenhouse","hamburger","hammock","harbour","hedgehog",
    "highway","hospital","hurricane","iceberg","island","jellyfish","journey",
    "jungle","kitchen","ladder","lantern","laundry","lavender","library",
    "lighthouse","lobster","magazine","mailbox","mammoth","marble","mattress",
    "meadow","mechanic","microscope","mountain","mushroom","musician",
    "narwhal","necklace","neighbour","notebook","october","octopus","orchard",
    "ostrich","oxygen","paddle","painting","pancake","panther","parachute",
    "passenger","peacock","pebble","pelican","penguin","pepper","piano",
    "picnic","pillow","pirate","planet","platypus","porcupine","postcard",
    "pottery","printer","pumpkin","puppet","pyramid","rabbit","raccoon",
    "railway","rainbow","reindeer","reptile","restaurant","rhubarb","ribbon",
    "riverbank","rocket","sandwich","satellite","scissors","scorpion",
    "seagull","seashell","shadow","shoulder","shovel","sidewalk","skeleton",
    "snowflake","spaghetti","sparrow","squirrel","stadium","staircase",
    "starfish","statue","stomach","submarine","sunrise","sweater","swimming",
    "tangerine","teacher","telescope","theatre","thunder","tomato","toolbox",
    "tortoise","tractor","treasure","triangle","trumpet","tunnel","turtle",
    "typewriter","vacuum","valley","vanilla","village","vinegar","violin",
    "volcano","vulture","waffle","walrus","wardrobe","weather","whistle",
    "windmill","window","wizard","wombat","workshop","yoghurt","zeppelin",
]

def make_counting(ps):
    word = random.choice(REAL_WORDS * 3 + NONSENSE)
    mode = random.choice(["letter", "letter", "letter", "length"])

    if mode == "letter":
        # bias toward letters that actually occur, but sometimes pick a zero
        pool = sorted(set(word))
        ch = random.choice(pool) if random.random() < 0.6 else \
             random.choice("abcdefghijklmnopqrstuvwxyz")
        n = word.count(ch)
        facts = [f"The word is {word}."]
        q = random.choice([
            f"How many times does the letter {ch} appear in it?",
            f"How many {ch}s are in the word?",
            f"How many times does {ch} appear in the word?"])
        times = "time" if n == 1 else "times"
        reasoning = (f"I will count the letters in {word}. "
                     f"<count>{word},{ch}={n}</count> "
                     f"So the letter {ch} appears {n} {times}.")
        return facts, q, reasoning, str(n), {"tool": "count"}
    else:
        n = len(word)
        facts = [f"The word is {word}."]
        q = random.choice([f"How many letters are in it?",
                           f"How long is the word?"])
        reasoning = (f"I will count the letters in {word}. "
                     f"<count>{word}={n}</count> "
                     f"So the word has {n} letters.")
        return facts, q, reasoning, str(n), {"tool": "count"}


# ---------------------------------------------------------------------------
# 7. PHYSICS / EE FORMULA SUBSTITUTION
#    The formula is GIVEN as a fact (it comes from the SD knowledge bank at
#    runtime), so the model never memorises physics — it applies a stated rule
#    to stated values and routes the arithmetic to <calc>. The "missing input"
#    variant teaches it to refuse when a required quantity was not given.
# ---------------------------------------------------------------------------
# (result, unit, a_name, a_unit, op, b_name, b_unit, formula sentence)
FORMULAS = [
    ("voltage","volts","current","amps","*","resistance","ohms",
     "Voltage equals current times resistance."),
    ("current","amps","voltage","volts","/","resistance","ohms",
     "Current equals voltage divided by resistance."),
    ("resistance","ohms","voltage","volts","/","current","amps",
     "Resistance equals voltage divided by current."),
    ("power","watts","voltage","volts","*","current","amps",
     "Power equals voltage times current."),
    ("charge","coulombs","current","amps","*","time","seconds",
     "Charge equals current times time."),
    ("energy","joules","power","watts","*","time","seconds",
     "Energy equals power times time."),
    ("total resistance","ohms","first resistance","ohms","+","second resistance","ohms",
     "Series resistance is the sum of the resistances."),
    ("total capacitance","microfarads","first capacitance","microfarads","+",
     "second capacitance","microfarads",
     "Parallel capacitance is the sum of the capacitances."),
    ("force","newtons","mass","kilograms","*","acceleration","metres per second squared",
     "Force equals mass times acceleration."),
    ("acceleration","metres per second squared","force","newtons","/","mass","kilograms",
     "Acceleration equals force divided by mass."),
    ("work","joules","force","newtons","*","distance","metres",
     "Work equals force times distance."),
    ("momentum","kilogram metres per second","mass","kilograms","*","velocity",
     "metres per second", "Momentum equals mass times velocity."),
    ("speed","metres per second","distance","metres","/","time","seconds",
     "Speed equals distance divided by time."),
    ("density","kilograms per cubic metre","mass","kilograms","/","volume",
     "cubic metres", "Density equals mass divided by volume."),
    ("pressure","pascals","force","newtons","/","area","square metres",
     "Pressure equals force divided by area."),
    ("wave speed","metres per second","frequency","hertz","*","wavelength","metres",
     "Wave speed equals frequency times wavelength."),
]

def make_physics(ps):
    (res, res_u, an, au, op, bn, bu, sentence) = random.choice(FORMULAS)

    if op == "/":                       # keep the division exact
        b = random.randint(2, 12)
        out = random.randint(2, 40)
        a = b * out
    else:
        a = random.randint(2, 40)
        b = random.randint(2, 25)
        out = a + b if op == "+" else a * b

    val_a = f"The {an} is {a} {au}."
    val_b = f"The {bn} is {b} {bu}."
    q = f"What is the {res} in {res_u}?"

    if random.random() < 0.18:          # missing-input variant
        keep, missing = (val_a, bn) if random.random() < 0.5 else (val_b, an)
        facts = [keep, sentence]
        random.shuffle(facts)
        reasoning = (f"{sentence} But the {missing} is not given. "
                     f"So it cannot be determined.")
        return facts, q, reasoning, "Cannot be determined", {"tool": "physics"}

    facts = [val_a, val_b, sentence]
    parts = [f"{sentence} The {an} is {a} {au} and the {bn} is {b} {bu}."]
    facts2, parts = maybe_irrelevant(facts, set(), parts, ps)
    reasoning = (" ".join(parts) +
                 f" <calc>{a}{op}{b}={out}</calc> So the {res} is {out} {res_u}.")
    random.shuffle(facts2)
    return facts2, q, reasoning, f"{out} {res_u}", {"tool": "physics"}


# ---------------------------------------------------------------------------
# 8/9. CODE TRACING — Lisp and Forth.
#    Chosen over Python deliberately: both have uniform, unambiguous syntax
#    (prefix and postfix), which is the same nested/sequential shape the model
#    already handles. Every step routes its arithmetic to <calc>, so the model
#    learns to TRACE, never to compute. Ground truth is produced by actually
#    evaluating the expression here, so the examples cannot be wrong.
# ---------------------------------------------------------------------------
OPS = ["+", "-", "*"]

def _apply(op, a, b):
    return a + b if op == "+" else (a - b if op == "-" else a * b)

def _pair(op):
    """operands that keep results non-negative and small"""
    if op == "-":
        a = random.randint(3, 40); b = random.randint(1, a); return a, b
    if op == "*":
        return random.randint(2, 12), random.randint(2, 9)
    return random.randint(1, 40), random.randint(1, 40)

def make_lisp(ps):
    depth = random.choice([1, 2, 2])
    steps = []
    if depth == 1:
        op = random.choice(OPS); a, b = _pair(op)
        val = _apply(op, a, b)
        expr = f"({op} {a} {b})"
        steps.append((f"({op} {a} {b})", f"{a}{op}{b}", val))
    else:
        inner_op = random.choice(OPS); ia, ib = _pair(inner_op)
        inner = _apply(inner_op, ia, ib)
        outer_op = random.choice(OPS)
        if outer_op == "-":
            outer_other = random.randint(1, max(1, inner))
            a, b, val = inner, outer_other, inner - outer_other
            expr = f"({outer_op} ({inner_op} {ia} {ib}) {outer_other})"
            second = (f"({outer_op} {inner} {outer_other})",
                      f"{inner}{outer_op}{outer_other}", val)
        else:
            outer_other = random.randint(2, 12 if outer_op == "*" else 40)
            val = _apply(outer_op, outer_other, inner)
            expr = f"({outer_op} {outer_other} ({inner_op} {ia} {ib}))"
            second = (f"({outer_op} {outer_other} {inner})",
                      f"{outer_other}{outer_op}{inner}", val)
        steps.append((f"({inner_op} {ia} {ib})", f"{ia}{inner_op}{ib}", inner))
        steps.append(second)

    facts = [f"The expression is {expr}.",
             "In this language the operator comes first and the innermost "
             "brackets are evaluated first."]
    q = random.choice(["What is the value of the expression?",
                       "What does the expression evaluate to?"])
    parts = []
    for i, (shown, ex, v) in enumerate(steps):
        lead = "First" if i == 0 else "Then"
        parts.append(f"{lead} {shown} gives <calc>{ex}={v}</calc>")
    reasoning = ". ".join(parts) + f". So the value is {steps[-1][2]}."
    random.shuffle(facts)
    return facts, q, reasoning, str(steps[-1][2]), {"tool": "lisp"}

def make_forth(ps):
    n_ops = random.choice([1, 2, 2])
    op1 = random.choice(OPS); a, b = _pair(op1)
    v1 = _apply(op1, a, b)
    toks = [str(a), str(b), op1]
    steps = [(a, b, op1, v1)]
    for _ in range(n_ops - 1):
        op = random.choice(OPS)
        if op == "-":
            c = random.randint(1, max(1, v1)); v2 = v1 - c
        elif op == "*":
            c = random.randint(2, 9); v2 = v1 * c
        else:
            c = random.randint(1, 40); v2 = v1 + c
        toks += [str(c), op]
        steps.append((v1, c, op, v2))
        v1 = v2

    program = " ".join(toks)
    facts = [f"The program is {program}.",
             "Numbers are pushed on a stack and an operator takes the top two "
             "numbers and pushes the result."]
    q = random.choice(["What is on top of the stack at the end?",
                       "What is the final value on the stack?"])
    parts = [f"Push {steps[0][0]}. Push {steps[0][1]}."]
    for i, (x, y, op, v) in enumerate(steps):
        if i > 0:
            parts.append(f"Push {y}.")
        parts.append(f"The {op} takes them and gives <calc>{x}{op}{y}={v}</calc>")
    reasoning = " ".join(p if p.endswith(".") else p + "." for p in parts) + \
                f" So the final value is {steps[-1][3]}."
    random.shuffle(facts)
    return facts, q, reasoning, str(steps[-1][3]), {"tool": "forth"}


# ---------------------------------------------------------------------------
# 10. EXTRACTIVE LOOKUP — the answer is STATED in the facts.
#    Every other type derives or computes something, which is why the model,
#    handed "The density of copper is 8960..." plus a formula, reached for the
#    formula and invented values. This type teaches the missing move: find the
#    fact that matches BOTH the property and the thing, and report it.
#    The negative cases matter most — they are what stops the invention.
#
#    The vocabulary here is deliberately REAL (copper, capacitor, resistivity)
#    rather than nonsense, because these are the words the knowledge bank
#    actually contains. Without them in the training corpus the tokenizer
#    shatters "copper" into "co-p-per" and the model has no token to reason on.
#
#    v2.1: fact COUNT is balanced across positives and negatives. The old
#    skew (negatives 1-2 facts, positives 2-4) taught "one fact -> refuse",
#    which produced the false refusal seen on hardware with /kbn 1.
# ---------------------------------------------------------------------------
MATERIALS = ["copper", "aluminium", "steel", "iron", "lead", "gold", "silver",
             "tin", "zinc", "nickel", "titanium", "brass", "water", "air",
             "silicon", "germanium", "carbon", "tungsten"]

COMPONENTS = ["capacitor", "resistor", "diode", "transistor", "inductor",
              "thermistor", "optocoupler", "regulator", "crystal", "relay",
              "transformer", "fuse", "battery", "motor", "sensor"]

# (property name, unit, low, high)
PROPERTIES = [
    ("density",              "kilograms per cubic metre", 700, 21000),
    ("melting point",        "degrees Celsius",            60,  3400),
    ("boiling point",        "degrees Celsius",           100,  5000),
    ("resistivity",          "nano ohm metres",            15,  1500),
    ("thermal conductivity", "watts per metre kelvin",      1,   430),
    ("specific heat",        "joules per kilogram kelvin", 120,  4200),
    ("atomic mass",          "atomic mass units",           1,   240),
    ("forward voltage",      "volts",                       1,     4),
    ("capacitance",          "nanofarads",                  1,  1000),
    ("resistance",           "ohms",                       10, 10000),
    ("operating voltage",    "volts",                       3,    48),
    ("rated current",        "amps",                        1,    40),
    ("tensile strength",     "megapascals",                40,  1400),
    ("expansion coefficient","parts per million per kelvin",1,    30),
]

LOOKUP_FACT_FORMS = [
    "The {p} of {s} is {v} {u}.",
    "{S} has a {p} of {v} {u}.",
    "The {p} of {s} is about {v} {u}.",
    "For {s}, the {p} is {v} {u}.",
]
LOOKUP_Q_FORMS = [
    "What is the {p} of {s}?",
    "What is the {p} of {s} in {u}?",
    "How much is the {p} of {s}?",
    "Give the {p} of {s}.",
]

# ---------------------------------------------------------------------------
# ft7: colloquial question phrasings, per property.
#
# The model must answer the same question whether it is asked in catalogue
# language ("What is the density of X?") or human language ("How heavy is X
# for its size?"). Hardware showed ft6 refusing the second even with the
# correct fact retrieved and in context.
#
# USED AT THE SAME RATE FOR POSITIVES AND NEGATIVES. Anything that appears
# more often on one side becomes a shortcut cue.
# ---------------------------------------------------------------------------
LOOKUP_PARAPHRASE = {
    "density": [
        "How heavy is {s} for its size?",
        "How dense is {s}?",
        "What does a cubic metre of {s} weigh?",
    ],
    "melting point": [
        "At what temperature does {s} melt?",
        "How hot does {s} have to get to turn liquid?",
        "When does {s} start melting?",
    ],
    "boiling point": [
        "At what temperature does {s} boil?",
        "How hot does {s} have to get to turn into vapour?",
        "When does {s} start boiling?",
    ],
    "resistivity": [
        "How well does {s} resist current?",
        "How resistive is {s}?",
        "Is {s} a good conductor, and by how much?",
    ],
    "thermal conductivity": [
        "How well does {s} conduct heat?",
        "How fast does heat travel through {s}?",
        "Is {s} good at moving heat?",
    ],
    "specific heat": [
        "How much energy does it take to warm {s}?",
        "How much heat does {s} store per kilogram?",
    ],
    "atomic mass": [
        "How much does one atom of {s} weigh?",
        "What does an atom of {s} weigh?",
    ],
    "forward voltage": [
        "How many volts does {s} drop when conducting?",
        "What voltage does {s} need to turn on?",
    ],
    "capacitance": [
        "How much charge can {s} hold?",
        "How big is the capacitance of {s}?",
    ],
    "resistance": [
        "How much does {s} resist current?",
        "How many ohms is {s}?",
    ],
    "operating voltage": [
        "What voltage does {s} run at?",
        "How many volts should {s} be given?",
    ],
    "rated current": [
        "How much current can {s} take?",
        "How many amps is {s} rated for?",
    ],
    "tensile strength": [
        "How strong is {s} before it breaks?",
        "How much pulling force can {s} take?",
    ],
    "expansion coefficient": [
        "How much does {s} expand when heated?",
        "How much does {s} grow per degree?",
    ],
}
PARAPHRASE_RATE = 0.35        # identical for positive and negative examples

def _lookup_question(prop, subject, unit):
    """Canonical form most of the time, colloquial form PARAPHRASE_RATE of it."""
    forms = LOOKUP_PARAPHRASE.get(prop)
    if forms and random.random() < PARAPHRASE_RATE:
        return random.choice(forms).format(s=subject)
    return random.choice(LOOKUP_Q_FORMS).format(p=prop, s=subject, u=unit)

LOOKUP_R_FORMS = [
    "The facts give the {p} of {s} as {v} {u}.",
    "The facts state that the {p} of {s} is {v} {u}.",
    "Looking at the facts, the {p} of {s} is {v} {u}.",
]
LOOKUP_NEG_FORMS = [
    "The facts give the {op} of {s}, but not the {p}.",
    "The facts do not state the {p} of {s}.",
    "Nothing in the facts gives the {p} of {s}.",
]

def _lookup_value(lo, hi):
    """Values must vary in SHAPE, not just magnitude. v8 used randint only, so
    every trained answer was a plain integer — and the model then read
    'atomic mass 28.085' as 14, and truncated '1448' to '144'. Decimals and
    long integers have to appear in the answer slot during training."""
    r = random.random()
    if r < 0.30:                                   # decimals
        places = random.choice([1, 2, 3])
        top = min(hi, lo + 500)
        return f"{random.uniform(lo, max(top, lo + 2)):.{places}f}"
    if r < 0.45:                                   # long integers
        return str(random.randint(1000, 99999))
    return str(random.randint(lo, hi))

def _lookup_fact(subj, prop, val, unit):
    return random.choice(LOOKUP_FACT_FORMS).format(
        p=prop, s=subj, S=subj.capitalize(), v=val, u=unit)

def make_lookup(ps):
    subjects = MATERIALS + COMPONENTS
    prop, unit, lo, hi = random.choice(PROPERTIES)
    subject = random.choice(subjects)
    others = [s for s in subjects if s != subject]
    negative = random.random() < 0.3

    # Fact COUNT is drawn from ONE distribution for both positives and
    # negatives, so count carries no signal about the answer. (Old code:
    # negatives always 1-2 facts, positives always 2-4 -> "one fact"
    # became a refusal cue, independent of whether the property matched.)
    n_facts = random.choices([1, 2, 3, 4], weights=[30, 30, 25, 15])[0]

    def other_subject():
        o = random.choice(others); others.remove(o); return o

    def other_property():
        return random.choice([p for p in PROPERTIES if p[0] != prop])

    if not negative:
        val = _lookup_value(lo, hi)
        facts = [_lookup_fact(subject, prop, val, unit)]
        trap_used = False
        while len(facts) < n_facts:
            r = random.random()
            if not trap_used and r < 0.45:
                # same subject, DIFFERENT property, short plain number —
                # the atomic-mass-vs-atomic-number trap. Kept from v9.
                p2, u2, _, _ = other_property()
                facts.append(_lookup_fact(subject, p2,
                                          str(random.randint(1, 99)), u2))
                trap_used = True
            elif r < 0.75:
                # same property, different subject -> must match SUBJECT
                facts.append(_lookup_fact(other_subject(), prop,
                                          _lookup_value(lo, hi), unit))
            else:
                # different subject AND different property
                p2, u2, l2, h2 = other_property()
                facts.append(_lookup_fact(other_subject(), p2,
                                          _lookup_value(l2, h2), u2))
        random.shuffle(facts)
        q = _lookup_question(prop, subject, unit)
        # ft7: values that are easy to truncate go through <quote>, which the
        # chip fills verbatim from the fact text (ft6 turned 103.296 into 103).
        use_quote = ("." in str(val) or len(str(val).replace(".", "")) >= 4)
        if use_quote:
            reasoning = (random.choice(LOOKUP_R_FORMS).format(
                            p=prop, s=subject, v=val, u=unit)
                         + f" <quote>{subject},{prop}={val}</quote>"
                         + f" So the answer is {val} {unit}.")
        else:
            reasoning = (random.choice(LOOKUP_R_FORMS).format(
                            p=prop, s=subject, v=val, u=unit)
                         + f" So the answer is {val} {unit}.")
        return facts, q, reasoning, f"{val} {unit}", \
               {"tool": "quote" if use_quote else "lookup", "nf": n_facts}

    # ---- negative: no fact states (subject, prop); count matches positives.
    # The {op} "gives the X but not the Y" template is ONLY used when a
    # same-subject fact is actually present, so the model can never learn
    # "gives the atomic mass but not the atomic mass".
    facts, subj_prop_named = [], None
    while len(facts) < n_facts:
        r = random.random()
        if subj_prop_named is None and r < 0.55:
            # same subject, wrong property — the classic near-miss
            p2, u2, l2, h2 = other_property()
            subj_prop_named = p2
            facts.append(_lookup_fact(subject, p2, _lookup_value(l2, h2), u2))
        elif r < 0.80:
            # right property, wrong subject
            facts.append(_lookup_fact(other_subject(), prop,
                                      _lookup_value(lo, hi), unit))
        else:
            # unrelated fact
            p2, u2, l2, h2 = other_property()
            facts.append(_lookup_fact(other_subject(), p2,
                                      _lookup_value(l2, h2), u2))
    random.shuffle(facts)
    q = _lookup_question(prop, subject, unit)
    forms = LOOKUP_NEG_FORMS if subj_prop_named else LOOKUP_NEG_FORMS[1:]
    reasoning = (random.choice(forms).format(p=prop, s=subject,
                                             op=subj_prop_named)
                 + " So it cannot be determined.")
    return facts, q, reasoning, "Cannot be determined", \
           {"tool": "lookup", "nf": n_facts}

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="constructed.jsonl")
    ap.add_argument("--mix", default="18,12,9,6,12,7,10,8,8,10",
                    help="pct: transitive,syllogism,undetermined,negation,arithmetic,counting,physics,lisp,forth,lookup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phrasing-set", choices=["train", "alt"], default="train")
    ap.add_argument("--phrasing-variants", type=int, default=0,
                    help="use only the first N wordings of each template list "
                         "(0 = all). Lower = less surface variety per pattern.")
    ap.add_argument("--axes", default="",
                    help="whitelist by positive adjective, e.g. 'heavier'")
    ap.add_argument("--exclude-axes", default="",
                    help="blacklist by positive adjective, e.g. 'heavier'")
    ap.add_argument("--chain-lengths", default="3,4",
                    help="allowed transitive chain sizes, e.g. '3' or '3,4'")
    ap.add_argument("--sort-curriculum", action="store_true",
                    help="order output easy->hard (see docs for trainer caveat)")
    a = ap.parse_args()
    if a.seed:
        random.seed(a.seed)

    if a.phrasing_variants > 0:
        for ps in P:
            for key in P[ps]:
                P[ps][key] = P[ps][key][:a.phrasing_variants]

    axes = AXES
    if a.axes:
        want = {x.strip() for x in a.axes.split(",")}
        axes = [ax for ax in AXES if ax[0] in want]
    if a.exclude_axes:
        drop = {x.strip() for x in a.exclude_axes.split(",")}
        axes = [ax for ax in axes if ax[0] not in drop]
    assert axes, "no axes left after filtering"
    klist = [int(x) for x in a.chain_lengths.split(",")]

    pcts = [int(x) for x in a.mix.split(",")]
    while len(pcts) < 10:
        pcts.append(0)          # backward compatible with shorter mixes
    assert len(pcts) == 10 and sum(pcts) == 100, \
        "mix must be 4-10 numbers summing to 100"
    order = ["transitive", "syllogism", "undetermined", "negation",
             "arithmetic", "counting", "physics", "lisp", "forth", "lookup"]
    gens = {
        "transitive":  lambda: make_transitive(axes, klist, a.phrasing_set),
        "syllogism":   lambda: make_syllogism(a.phrasing_set),
        "undetermined":lambda: make_undetermined(axes, a.phrasing_set),
        "negation":    lambda: make_negation(a.phrasing_set),
        "arithmetic":  lambda: make_arithmetic(a.phrasing_set),
        "counting":    lambda: make_counting(a.phrasing_set),
        "physics":     lambda: make_physics(a.phrasing_set),
        "lisp":        lambda: make_lisp(a.phrasing_set),
        "forth":       lambda: make_forth(a.phrasing_set),
        "lookup":      lambda: make_lookup(a.phrasing_set),
    }

    seen, out, attempts = set(), [], 0
    while len(out) < a.n:
        attempts += 1
        kind = random.choices(order, weights=pcts)[0]
        facts, q, reasoning, ans, meta = gens[kind]()
        sig = "|".join(sorted(facts)) + "||" + q
        if sig in seen:
            if attempts > a.n * 12:
                break
            continue
        seen.add(sig)
        obj = {"facts": facts, "question": q, "reasoning": reasoning,
               "answer": ans, "type": kind, **meta}
        out.append(obj)

    if a.sort_curriculum:
        rank = {"negation": 0, "counting": 1, "syllogism": 2,
                "arithmetic": 3, "forth": 4, "lisp": 5,
                "physics": 6, "transitive": 7, "undetermined": 8}
        out.sort(key=lambda o: (rank[o["type"]], len(o["facts"])))

    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(o["type"] for o in out)
    print(f"wrote {len(out)} unique examples to {a.out} "
          f"({attempts} attempts, phrasing={a.phrasing_set}, "
          f"axes={[x[0] for x in axes]}, k={klist})")
    for k in order:
        print(f"  {k}: {c[k]} ({100*c[k]//max(1,len(out))}%)")

if __name__ == "__main__":
    main()
