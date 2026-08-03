# R-457

A 27-million-parameter reasoning model that runs entirely offline on two
ESP32-S3 microcontrollers. No WiFi, no cloud, no phone. It answers questions
from a knowledge bank on an SD card, calls tools on the chip for arithmetic
it cannot do reliably in its head, refuses honestly when the facts do not
support an answer, and learns new facts you teach it at runtime.

It is slow — about 0.3 tokens per second. That is the honest headline. What
it demonstrates is not speed but that a model this small can be made to
*reason over supplied facts and know when to say no*, on hardware that costs
about $40 total.

```
> /ask What is the atomic mass of silicon?
[retrieved 2 fact(s) from the bank]

Facts: The atomic mass of silicon is 28.085 atomic mass units.
       The atomic number of silicon is 14.
Question: What is the atomic mass of silicon?
Reasoning: Looking at the facts, the atomic mass of silicon is 28.085
           atomic mass units. So the answer is 28.085 atomic mass units.
Answer: 28.085 atomic mass units

[52 tokens in 156.0s — 0.33 tok/s, conf avg 0.98 min 0.47]
```

Note the trap in that example: the atomic *number* (14) sits right next to
the atomic *mass* (28.085), and the model does not confuse them. Most of the
work in this project went into that kind of discrimination.

---

## What it does

Every capability below is verified running on the hardware, not just in
training metrics.

**Reasoning**
- Transitive chains ("A is taller than B, B is taller than C…")
- Syllogisms
- Negation
- Refusal when the question is underdetermined ("both compared to a third
  thing, but not to each other — cannot be determined")

**Tools** — the chip computes, the model routes
- `<calc>` arithmetic: the model writes `<calc>55/5=`, the firmware computes
  11 and feeds it back as forced tokens, so the KV cache stays consistent
- `<count>` letter counting: `<count>strawberry,r=` → 3

**Knowledge**
- 792-key knowledge bank on the SD card, binary-searched in ~11 reads
- Extractive lookup with distractor facts present
- Honest refusal when the retrieved facts do not contain the answer
- Synonym expansion so "how heavy" also finds "density" cards

**Generation**
- `<write>` mode: grounded paragraphs that use only the supplied numbers

**Runtime behaviour**
- `/learn` — teach it a fact; written to the SD card, used immediately
- `/refused` — every question it could not answer, logged for later teaching
- `/selftest` — runs its own canaries and prints PASS/FAIL
- Confidence flagging — prints per-answer confidence, warns when low
- Boot fingerprint — verifies the tokenizer encodes correctly at startup

---

## Measured results

Held-out accuracy, 200 examples per set, on-device tool execution:

| set | ft7b | previous (ft6) |
|---|---|---|
| axis transfer (unseen adjectives) | **85.5%** | 63.5% |
| arithmetic (`<calc>`) | 97.0% | 97.0% |
| counting (`<count>`) | 99.0% | 92.0% |
| Forth stack tracing | 98.5% | 98.0% |
| Lisp expression tracing | 100.0% | 100.0% |
| physics word problems | 99.5% | 100.0% |
| mixed in-distribution | 89.0% | 89.0% |
| reworded phrasings | 93.0% | 89.0% |
| 4-item chains | 68.5% | 55.0% |
| lookup | 61.0% | 57.5% |
| lookup (single fact) | 62.5% | 67.5% |
| lookup v2 | 57.0% | 60.0% |

**Axis transfer** is the result worth explaining. The model is trained on
comparative adjectives like "taller" and "heavier". At evaluation it is
given adjectives it has *never seen* — "sharper", "richer", "deeper" — and
must still chain them correctly. 85.5% means the transitive relation
generalised past the specific words it was taught on.

The jump from 63.5% to 85.5% is most likely explained by fixing a silent
data-corruption bug in the training data loader rather than by the training
data changes made at the same time — the two were not isolated, and the
controlled comparison was never run. See `FAILURES.md` #1; it is the most
useful thing in this repo.

**Speed:** 0.25–0.42 tok/s across the two boards. Measured, not estimated.
The UART link between boards costs about 2% of that; the rest is compute.

---

## Hardware

| part | role | cost |
|---|---|---|
| Guition JC3248W535C (ESP32-S3, 16MB flash, 8MB PSRAM) | worker — layers 0–3 | ~$15 |
| Waveshare ESP32-S3-Touch-LCD-4.3 (16MB / 8MB) | head — embedding, layers 4–7, classifier, tokenizer, SD | ~$25 |
| microSD card (any size; 512MB is plenty) | knowledge bank, learned facts, refusal log | — |
| 3 jumper wires | the link | — |

**Why two boards:** the INT4 model image is 17.2MB. A 16MB flash chip cannot
hold it alongside the firmware — the largest usable model partition is about
14MB. Splitting the layers across two boards solves this.

Two boards buy **capacity, not speed**. Generation is sequential: while one
board computes its layers the other waits. The split makes the model
*possible*, not faster.

**Wiring** (three wires, boards powered off):

```
head GPIO8  ──────→ worker GPIO18     (head TX → worker RX)
head GPIO9  ←────── worker GPIO17     (head RX ← worker TX)
head GND    ────────  worker GND
```

460800 baud, CRC + retry. The head uses GPIO 8/9 because on the Waveshare
board GPIO 17/18 are wired to the LCD panel and unavailable — 8/9 are the
I2C terminal block, used here as a plain UART.

---

## Quick start

1. Download `r457_ft7b_worker.bin` and `r457_ft7b_head.bin` from
   [Releases](../../releases).
2. Flash the model images:
   ```
   esptool.py --chip esp32s3 --port <guition-port>   write_flash 0x1F0000 r457_ft7b_worker.bin
   esptool.py --chip esp32s3 --port <waveshare-port> write_flash 0x1F0000 r457_ft7b_head.bin
   ```
3. Copy `kb.bin` to the root of the SD card, insert it in the Waveshare.
4. Open `sketches/pipeline_worker` and `sketches/pipeline_head` in the
   Arduino IDE and upload each to its board. Board settings and the two
   board-specific gotchas are in `BUILD.md` — read it, they are not obvious.
5. Wire the three jumpers, power the worker first, then the head.
6. Open the serial monitor on the head at 115200 and type `/selftest`.

Expect:

```
kb: ready, 792 keys (SD)
boot check: mode-token ok, newline ok
Ready: emb + 4 local layers of 8 total.
SELFTEST: PASS (silicon ok, aluminium ok)
```

Full build-from-source instructions — training, export, splitting — are in
`BUILD.md`.

---

## How it works

**Prompt format.** Everything the model sees is four parts:

```
<reason>
Facts: The density of aluminium is 2700 kilograms per cubic metre.
Question: What is the density of aluminium?
Reasoning: The facts give the density of aluminium as 2700 kilograms per
           cubic metre. So the answer is 2700 kilograms per cubic metre.
Answer: 2700 kilograms per cubic metre
```

`<reason>` and `<write>` are single reserved tokens that switch the model's
mode. Facts come from the SD bank, from what you typed, or from what you
have taught it with `/learn`.

**Tools.** The model is not trusted with arithmetic. It writes the *call*
and stops at the `=`; the chip computes the answer and injects it back as
forced tokens. Generation continues with a KV cache that never saw a wrong
number. A 27M model cannot divide reliably, but it can learn to ask.

**Refusal.** Roughly a third of training examples are questions the facts
do not answer. The model is taught to name what is missing and stop. This is
what makes the knowledge bank usable — a lookup system that confabulates
when it misses is worse than no lookup at all.

**The learning loop.** Ask something it cannot answer → it refuses and logs
the question to `/refused.txt` → you check `/refused` later → you teach it
with `/learn The boiling point of silicon is 3265 degrees Celsius.` → it
answers correctly from then on. Verified end to end on hardware, no
retraining involved.

---

## Repository layout

```
core/         llm_core.c/h    — inference, INT4 matmul, tokenizer, KV cache
              kb.cpp/h        — knowledge bank reader, retrieval, prompt build
sketches/     pipeline_head/  — the user-facing board (SD, tools, commands)
              pipeline_worker/— the coprocessor board
pc_tools/     construct.py    — training data generator
              build_kb.py     — knowledge bank builder
              export_model.py — checkpoint → INT4 device image
              split_image.py  — device image → two board images
              eval_fixed.py   — fixed-yardstick loss evaluation
              evaluate_tools.py — held-out capability scoring with tools
docs/         FAILURES.md, ROADMAP.md, BUILD.md
```

---

## Reading order

If you are here to build one: `BUILD.md`.

If you are here because you are training small models yourself:
**`FAILURES.md` first.** It documents a silent data-corruption bug that
taxed every training run in this project for weeks, two tokenizer bugs that
produced convincing-looking garbage, and four capability experiments that
failed with their diagnoses. That file cost far more to produce than this
one.

## Licence

MIT. Built on [llama2.c](https://github.com/karpathy/llama2.c) by Andrej
Karpathy. TinyStories dataset by Eldan & Li. Per-layer-embedding experiments
informed by [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai).
