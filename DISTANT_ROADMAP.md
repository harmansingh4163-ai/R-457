# DISTANT_ROADMAP.md

The horizon past ROADMAP.md: ideas discussed but not yet committed to,
plus the standing response to the strongest external critique. Same
rules as ROADMAP.md — honest costs, honest odds, ordered by value per
hour. Items graduate from here into ROADMAP.md; they do not ship from
here. (Session log: Aug 3, 2026.)

Estimates assume the Mac Mini M4: fine-tune ~2.5 h, evaluation ~1.5 h.

---

## Standing item 0 — the critique this file answers

The strongest fair criticism of R-457: "27M cannot natively reason;
the tools, retrieval and refusal are firmware; calling it a reasoning
model conflates host code with the network."

**Conceded:** no emergent open-ended reasoning at 27M; the memory and
split engineering is a real part of the story; "reasoning model" will
be over-read by skeptics.

**Rebutted, with evidence in this repo:**
- Refusal is learned, not a lookup-miss branch: the false-refusal
  failure (single-fact = refusal cue) was induced by training-data
  imbalance and fixed by rebalancing construct.py, not firmware.
  Heuristics do not have data-induced failure modes. FAILURES.md.
- Tool *execution* is firmware; tool *invocation* is learned: the
  model decides when to emit <calc>, composes the operands, continues
  from the result (~1,350 calls per eval run). When it broke, the bug
  was encoder-side, not tool-side.
- Axis transfer (unseen adjectives 17.6% → 41.6% with a pretrained
  base) involves no firmware at all.

**Response shipped/queued:** model-vs-firmware table for the README
(docs/README_model_vs_firmware.md) + two ablations: tools-OFF eval
(scores collapse → chip executes) and distractor-trap-ON (28.085
picked over 14 → model decides). Ablation runs still to do: ~1.5 h.

---

## Corrections carried into this roadmap

**SIMD matmul is DEAD.** Sandbox measurements: word-loads + hoisted -8
was 0.51× (defeats auto-vectorization); algebraic hoist alone 1.07×
with vectorization off. The old "1.5–2.5×" claim came from DaveBben /
eric-humane numbers that never transferred: those were 260K-param
SRAM-resident int8; ours is int4 nibbles in PSRAM, and nibble-unpack
costs what the SIMD multiply saves. The real clue: measured ~3000 ms/
token vs ~140 ms arithmetic floor AND ~141 ms PSRAM bandwidth floor —
a 20× gap over both. The bottleneck is memory latency, not compute.

Speed work therefore attacks latency, in this order:
1. PSRAM clock check (80 vs 120 MHz in the board menu) — 5 min, free
   ~1.5× if it's at 80.
2. Access-pattern audit of the matmul loop — an afternoon. Are group
   scales interleaved with their weight groups or a separate array
   (two streams = cache thrash)?
3. Dual-core split of outstanding requests — realistically 1.3–1.7×,
   real concurrency-bug risk.
4. Burst-read tiling into SRAM scratch — the fix that actually closes
   latency gaps, and a rewrite of the inner loop. Days.

---

## Tier A — built this session, awaiting hardware

### A1. On-device answer verification + citations (verify.c)
**Cost:** wiring per INTEGRATION.md, ~15 min; hardware check ~15 min.
**Odds:** high — host self-test 8/8, all planted errors caught,
-Werror clean, pure C99.

The PC verifier discipline ported to the chip: every number in the
output must be grounded in the prompt or a chip-computed tool span;
every <calc>/<count> is re-executed; refusals must assert nothing.
Grounded numbers are cited ("660 <- fact 2"). Catches the open
decimal-truncation failure LIVE (103 vs 103.296) until <quote> fixes
it. Self-test runs at boot — a hollow check can no longer hide.
Known limits (documented in verify.h): exact-string number match,
phrase-based refusal detection.

---

## Tier B — discussed improvements, not started

### B1. ft7 bundle: <quote> + rate problems + <conv> + chains 3–6
**Cost:** construct.py v3 + regen + one fine-tune + eval ≈ 5 h, plus
~20 firmware lines each for <quote> and <conv>.
**Odds:** good for <quote> (same philosophy as <count>, tokens 7–8
reserved); moderate for chains (staged-training rebalance risk —
judge per-type against ft6's table, always_save_checkpoint on).
Closes the last documented lookup failure mode.

### B2. Grammar-constrained sampling + EOS discipline
**Cost:** ~50 lines + a small stop rule; one evening.
**Odds:** high. Inside tool tags, mask logits to digits/ops/close-tag;
stop at first newline after the answer. Makes demo output clean and
tool syntax unbreakable.

### B3. Refusal logging to SD
**Cost:** small firmware.
**Odds:** high mechanically; value depends on actually using the
device. Doubles as evidence that refusal fires on semantic mismatch,
not empty retrieval.

---


### B4. fp16 KV cache
**Cost:** firmware-only, small. **Odds:** high.
Halves KV RAM; device seq 512 fits in the same 4.2MB. Also a
prerequisite for anything deeper. From MAX_MODEL.md.

### B5. gs=64 export experiment
**Cost:** one export + offline eval. **Odds:** moderate.
~1.5MB flash back (0.5625 B/param). Risk: coarser groups erode exact
decimal copying — the product. Eval before any hardware flash.

### B6. PSRAM weight-cache experiment
**Cost:** one evening of firmware. **Odds:** genuinely unknown — that
is the point.
At llm_init, memcpy ~3MB of tensors from mmapped flash into PSRAM and
swap the pointers; llm_matmul_rows does not care where a pointer
points. Physics: flash QIO 4-bit @ 80MHz vs PSRAM OPI 8-bit @ 80MHz —
2x bus width for whatever moves. Amdahl on ~6.75MB/board/token:
~1.28x best case at 3MB, ~1.7x at 5.5MB after B4 frees KV RAM.
DOUBLES AS THE BOTTLENECK PROBE: speedup tracking bytes-moved = the
matmul is bandwidth-bound (proceed to B4 + full cache); no speedup =
latency/pattern-bound (the access audit knows where to dig). Either
outcome closes a question. Sequencing: run BEFORE the audit — it is
the audit's measurement instrument.
Rejected twin: spending the same PSRAM on extra weights (~33M dense
via SD boot-load). Mechanically fine, dies on wall 3 (MAX_MODEL.md):
more bytes per token on a 0.25-0.5 tok/s device.

## Tier C — invented, uncommitted

### C1. Idle-worker retrieval
Worker scores KB embeddings for the NEXT query while the head
generates. Turns "two boards buy capacity, not speed" into a
retrieval engine. Moderate firmware; needs the semantic-retrieval
embeddings first.

### C2. Twin-12M ensemble
Both boards fit the 7.69MB 12M; run independently, compare over the
existing UART link. Agree = confident, disagree = "(unsure)". A
two-model ensemble on $16 of hardware. Firmware only. Honest cost:
a confidence feature, zero speed.

### C3. Self-authoring dataset loop
/learn + refusal logging + a /fix command → canonical training rows
on SD → periodic fine-tune on the Mac. The device collects its own
next dataset. Small firmware + one script.

### C4. Golden-stream replay canary
Seeded RNG + stored expected token streams for ~10 prompts; boot test
diffs them. Any silent regression (encoder, quant, KB) caught in 30 s.

### Rejected: self-consistency voting
3 samples × ~40 tokens at ~3 s/token ≈ 6 minutes per answer. Dead
until the latency work lands. Revisit only after a real speed win.

---

## Graduation rule

An item leaves this file when it gets a date and enters ROADMAP.md
with a measured cost. Nothing ships from the distant roadmap.
