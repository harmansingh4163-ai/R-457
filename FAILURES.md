# FAILURES.md

Everything that did not work, and why. This is the most useful file in the
repository. Most of it was expensive.

Each entry follows the same shape: what broke, how it presented, what the
actual cause was, and what it cost. The presentation matters — nearly every
bug here looked like something else first.

---

## 1. Silent data corruption on Apple Silicon (the expensive one)

**Symptom.** A fine-tune ran 4000 iterations with the loss pinned at 4.45
and never descending. The same checkpoint, evaluated on the same shard file
with a hand-written loop, scored 0.733.

**Chase.** Suspected in order: wrong shard directory, stale tokenizer,
BOS/dummy-space convention mismatch, checkpoint weights failing to load,
window alignment, batch size. All wrong. Every direct read of the data was
correct; only the training loader's output was bad.

**Cause.** In `tinystories.py`:

```python
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
```

On MPS, transferring a tensor built from a fresh `np.memmap` with
`non_blocking=True` returns garbage. Token IDs came back as values like
`23553027236040659` in a 4096-token vocabulary, and `X[1:]` did not match
`Y[:-1]` — the batches were not even internally consistent.

**Fix.** `non_blocking=False`. Two characters, twice.

**Cost.** Every 27M-era training run went through this loader. Whether the
earlier 12M models did was never checked — if you are reading this to audit
your own history, check which `tinystories.py` each run actually used rather
than assuming.

The corruption appears to have been intermittent: models still learned, just
badly, as if trained with heavy random noise injection. The two runs below
bracket the fix, but they are **not a controlled comparison** — they differ
in training data as well as in loader correctness. The control that would
isolate the loader (the older data re-run through the fixed loader) was
never performed.

| held-out set | ft6 (corrupt loader, older data) | ft7b (fixed loader, newer data) |
|---|---|---|
| axis transfer | 63.5% | 85.5% |
| transitive within axis | 33.6% | 81.6% |
| 4-item chains | 55.0% | 68.5% |
| counting | 92.0% | 99.0% |

The data changes between these two runs were paraphrased question forms and
one new tool, neither of which plausibly affects transitive chaining — which
is why the loader fix is the likely explanation. Likely, not demonstrated.

Weeks of tuning, several architecture decisions, and one abandoned model
line were all measured through this loader, so every comparison made during
that period carries an asterisk. The lesson is not "MPS is buggy" — it
is that **nothing downstream of a data loader can be trusted until the loader
itself has been verified end to end.** One assertion would have caught it:

```python
assert X.max() < vocab_size
assert (X[:, 1:] == Y[:, :-1]).all()
```

Both are now in the training loop.

---

## 2. Tokenizer: user-defined symbols shattered into text

**Symptom.** On device, the model echoed the question back before answering,
produced corrupted line boundaries, and got stuck in a loop emitting
`114/calc6=114/calc6=…` after every tool call.

**Chase.** Sampling temperature was suspected first — the loop reproduced
identically at temperature 0, which ruled that out. Adding a per-token ID
print to the firmware showed the real story.

**Cause.** Two separate bugs in the C encoder, with one shared root: it
built tokens purely by BPE merges, so it could never produce a token that
sentencepiece stores as a *whole-string user-defined symbol*.

- `<reason>` (a single reserved token, ID 9) was being encoded as the text
  pieces `<`, `re`, `ason`, `>` — four tokens. Every prompt was
  out-of-distribution at token one.
- Newline was encoded via a legacy byte-fallback offset (`byte + 3`), giving
  ID 13. In this vocabulary, reserved symbols shifted the byte block, so
  `\n` is actually ID 21. Sentencepiece stores byte pieces under the literal
  text `<0x0A>`, and the encoder never looked them up by name. Every line
  boundary in every prompt was a wrong token.

The `114/calc6=` loop was the *same* bug wearing a different mask: the
firmware injects `11</calc>` after a tool call, and the broken encoder
shattered the closing tag into mis-offset byte IDs, corrupting the context
so the model re-emitted `=` and re-triggered the tool.

**Fix.** A vocabulary-scanned special-token pre-pass, and byte fallback that
looks up `<0xNN>` pieces by name with the legacy offset only as a fallback.
Both are vocabulary-driven, so they work for any vocab size.

**Cost.** Several hours, and it invalidated the first round of on-device
evaluation. **Verification that now runs at every boot:**

```
boot check: mode-token ok, newline ok
```

The device encodes `<reason>` and `\n` at startup and reports whether each
came out as a single correct ID. Three seconds of firmware that would have
saved the whole debugging session.

---

## 3. `<quote>` tool — the model would not adopt it

**Goal.** The model truncated decimals: `103.296` became `103`. The fix
seemed obvious — add a tool that makes the chip copy the value verbatim out
of the fact text, exactly like `<calc>` and `<count>` already work.

**What happened.** After a full fine-tune, held-out accuracy on the affected
set went *down* (67.5% → 62.5%), the truncation still occurred, and the
model began emitting malformed calls on device:

```
<quote>crystal,atomic mass. So the answer is 28.085 atomic mass units.
```

Wrong subject, no `=`, no closing tag — a half-learned tool.

**Cause.** `<quote>` reached only 2,848 of 59,015 fine-tune examples (4.7%).
`<calc>` and `<count>` each hold a far larger share and were present from
the first training generation. The model learned the tool's *appearance* but
not its grammar or when to use it.

**Conclusion.** A tool needs a substantial share of the mix to be learned
properly — a rough floor of 15% based on what `<calc>` and `<count>` got.
Adding a tool cheaply, in the leftover space, produces a new failure mode
rather than a new capability.

---

## 4. Semantic retrieval — embeddings collapsed

**Goal.** Keyword retrieval misses phrasings that share no words with the
bank key. The plan: mean-pool the model's own token embeddings for each key,
store as INT8 vectors on the SD card, and match questions by dot product. No
extra training, ~400KB, under 10ms per query.

**What happened.** Rejected in ten minutes, before any firmware was written.
Nearest-neighbour probes on the built vectors:

```
'how heavy is aluminium for its size'    -> specific heat formula (0.86),
                                            specific formula (0.86),
                                            voltage divider (0.86)
'at what temperature does aluminium melt'-> frequency period (0.87),
                                            specific heat formula (0.85)
```

Every query matched everything at ~0.85. Mean-centering the vectors improved
the spread (0.39–0.73) and surfaced "temperature" correctly, but still never
retrieved an aluminium card for an aluminium question.

**Cause.** A 27M model trained on stories and templated facts never learns
that "heavy for its size" means density. Word-level embeddings encode
co-occurrence, not paraphrase equivalence, and mean-pooling discards word
order on top of that. Sentence similarity needs a model trained for it.

**What replaced it.** A hand-written synonym table (heavy→density,
melt→melting, boil→boiling…) applied as *extra* content words before
candidate generation. Dumb, deterministic, ~40 lines, and it works — the
retrieval half of the problem is solved.

**Worth keeping:** the ten-minute probe that killed this idea before it cost
an evening. Test the representation before building on it.

---

## 5. Question paraphrase — still unsolved

**Symptom.** With the correct fact retrieved and in context, and the bank
cleaned of confusing card shapes, the model still refuses a colloquial
question:

```
Facts: The density of aluminium is 2700 kilograms per cubic metre.
Question: How heavy is aluminium for its size?
Reasoning: The facts do not state the tensile strength of aluminium.
           So it cannot be determined.
```

It invents a property nobody asked about. Confidence flagging correctly
marks this as the least confident answer of any test run.

**First attempt.** Added colloquial question forms to the training data,
per property, at 35% — balanced identically across positive and negative
examples so phrasing could not become a refusal cue. It did not transfer.

**Best current diagnosis.** The paraphrase templates were instantiated over
the generic subject pool used by the data generator (motor, resistor,
sensor) and never over real element names. The model likely learned the
phrasing tied to the wrong subject distribution.

**Status:** open. This is the clearest remaining wall, and it separates
cleanly from the two layers below it — retrieval works, context is clean,
and the failure is purely in mapping the question to the property.

---

## 6. Per-layer embeddings (PLE) — a real negative result

**Hypothesis.** Keep a small dense core in fast memory and put a large
per-layer embedding table in flash, read a few rows per token. Published
work reports 28.9M stored parameters running at 9.5 tok/s this way, versus
this project's 0.3 tok/s for a dense 27M.

**Work done.** Grafted a PLE arm into the model (verified bit-identical on
dense configs), trained a 32k-vocabulary tokenizer, re-tokenized the entire
corpus, ran a 20k-iteration pretrain plus a matched dense control, then a
full fine-tune and capability evaluation. About 30 hours of compute.

**Language modelling result — PLE clearly won:**

| fixed-yardstick loss | PLE (23.4M) | dense control (14.8M) |
|---|---|---|
| stories | 1.33 | 1.71 |
| wikipedia | 6.01 | 6.64 |

**Capability result — PLE clearly lost:**

| held-out set | PLE fine-tune | dense 27M (ft6) |
|---|---|---|
| counting | 24.5% | 92.0% |
| 4-item chains | 10.5% | 55.0% |
| Lisp | 70.0% | 100.0% |
| axis transfer | 38.5% | 63.5% |
| lookup | 28.5% | 57.5% |
| physics | 99.5% | 100.0% |
| Forth | 97.0% | 98.0% |

**Reading it.** Shallow, tool-mediated tasks survived. Everything requiring
chained comparison or careful extraction collapsed. The likely cause is core
width: the PLE model had 8 layers at width 256, of which about half the
parameters were embedding — roughly 6.3M of transformer layers against 23M
in the dense model. **The flash table adds knowledge capacity, not reasoning
depth, and it cannot substitute for the latter.**

Two caveats stated honestly. First, this comparison ran entirely through the
corrupted data loader of failure #1, so the absolute numbers are suspect —
though both arms were equally affected. Second, the PLE model had more total
parameters than its control, so the language-modelling win is partly a
parameter win rather than purely an architecture win.

**Conclusion for this hardware:** loss is not capability. A model can be
measurably better at predicting text and dramatically worse at the job.

---

## 7. Dense 27M does not fit one board

The INT4 image is 17.21MB. A 16MB flash chip leaves at most ~14MB for a
model partition after bootloader, app, and NVS.

Attempted escapes, all closed:
- Larger quantization groups: `hidden_dim` is 1376 = 32 × 43, so 32 is the
  only power-of-two group size that divides it. No knob.
- Repartitioning: even squeezing the app to 1.2MB tops out around 14.9MB.
- Streaming weights from the SD card: SPI gives 1–2 MB/s, so a full
  17MB read per token is 10+ seconds per token. Physics, not engineering.

**What worked:** splitting the layers across two boards. 7.92MB + 9.29MB,
both comfortable.

---

## 8. Grammar-constrained sampling — written, reverted

**Goal.** While inside a tool tag, mask the logits so only digits, operators
and the closing tag can be sampled. Tool syntax becomes structurally
impossible to malform.

**What happened.** The vocabulary scan that builds the masks completed all
4096 IDs, then the boot never reached the next print. Reverted after several
upload cycles.

**Postscript, in fairness to the idea:** the "hang" was diagnosed during the
same session in which a *different* silent delay — SD card initialisation,
which prints nothing for several seconds — repeatedly looked like a crash.
The masking code may well have been fine. It is written and preserved; it
deserves a retry with core debug logging enabled from the start.

---

## 9. SIMD matmul — the premise was wrong

**Claim in the original backlog:** "1.5–2.5× speedup, best ratio in the whole
backlog." That number came from published ESP32-S3 LLM demos.

**What testing showed.** Two candidate optimisations, both bit-exact:

| variant | speedup |
|---|---|
| word-wide loads + hoisted offset | **0.51×** (slower) |
| algebraic hoist only, no autovectorisation | 1.07× |

The first is slower because the compiler already auto-vectorises the simple
loop, and hand-optimisation defeats it.

**Why the premise failed.** The published demos ran 260K-parameter models in
SRAM with float32 or INT8 weights. This model is INT4 nibbles in PSRAM.
ESP-DSP's SIMD operations work on INT8; unpacking nibbles to feed them costs
about what the multiply saves.

**A more useful observation.** Rough arithmetic puts the compute floor at
~140ms per token and the PSRAM bandwidth floor at ~141ms. Measured token
time is ~3000ms — **20× either floor.** That gap suggests memory latency and
cache behaviour dominate, not arithmetic throughput, and SIMD would not
address it. Any future speed work should start by measuring where the time
actually goes.

---

## 10. Non-canonical knowledge cards poisoned retrieval

**Symptom.** A false refusal on a question whose answer was the first fact in
context:

```
Facts: The density of aluminium is 2700 kilograms per cubic metre.
       Aluminium has symbol Al, atomic number 13, and atomic mass 26.982.
Question: What is the density of aluminium?
Answer: Cannot be determined
```

**Isolation.** The same question about copper — whose cards are all in the
canonical `The X of Y is Z unit.` shape — answered correctly at 0.98
confidence. Same model, same question form, opposite outcome. The variable
was card shape, not the question.

**Cause.** 36 element cards packed three properties into one sentence. The
model was trained to extract from single-fact sentences, and the
multi-property shape derailed it even when the *correct* card was also
present.

**Fix.** Two lines in the bank generator: split each summary card into three
canonical single-fact cards. Rebuild, copy to SD, done. The aluminium
question went from a false refusal to the correct answer with no retraining.

**Generalisation:** the shape of retrieved context matters as much as its
content. Scaling a knowledge bank to more cards before auditing card shape
would scale the problem.

---

## 11. Smaller lessons, each paid for

**Two boards buy capacity, not speed.** Layer splitting leaves one board
idle per token. Tensor parallelism is infeasible at UART bandwidth.
Datacentres escape this with batching; a single-user device cannot.

**The training loop's validation column is unreliable for mixed-objective
fine-tuning.** It reports whichever shard neighbourhood the evaluation
stream happens to sample — story shards score ~1.2, wikipedia shards ~4.4.
Two runs with different weights produced near-identical values at the same
step numbers, which is a data artifact, not a model measurement. All ranking
decisions now use a fixed-yardstick script: same files, same windows, every
checkpoint.

**Always save checkpoints during fine-tuning.** The default only saves on
validation improvement. A resumed run inherited a low best-validation value,
never beat it, and discarded twelve hours of training without a single
warning line.

**Surface dilution is the recurring killer.** Adding new reasoning types
without rebalancing collapses existing ones — transitive accuracy once fell
from 97% to 1.3% this way. Staged training with an explicit optimizer reset
between stages is the validated fix.

**Every verifier check must be tested against a deliberately planted error.**
Silently hollow checks — code that passes everything, including things it
should reject — recurred three separate times in this project.

**A summary written in a document is not a measurement.** Several beliefs
recorded as fact turned out to be estimates that were never verified. The
model image size, the achievable speedup, and the base checkpoint's quality
were each wrong when finally measured.
