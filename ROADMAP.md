# ROADMAP.md

What is worth doing next, with honest costs and honest odds. Ordered by
value per hour, not by appeal.

Estimates assume a Mac Mini M4: a fine-tune is ~2.5 hours, a capability
evaluation ~1.5 hours, a pretrain ~10 hours.

---

## Tier 1 — clear value, small cost

### 1. Paraphrase robustness, done properly
**Cost:** data change + one fine-tune + one evaluation ≈ 5 hours.
**Odds:** moderate. The first attempt failed (`FAILURES.md` #5).

This is the clearest remaining wall. Retrieval works, context is clean, and
the model still cannot map "how heavy is aluminium for its size" onto a
density fact sitting in front of it.

What to do differently: instantiate the paraphrase templates over the
**real subject vocabulary** — actual element and component names that appear
in the knowledge bank — rather than only the generic pool the data generator
uses. Keep the balance discipline that the first attempt got right:
identical paraphrase rates for positive and negative examples, or phrasing
becomes a refusal cue.

Verify on device with a colloquial question, not only on the held-out sets —
those use canonical phrasings and will show a false neutral.

### 2. Quiz mode
**Cost:** ~1 hour, firmware only, no training, no risk to the model.

The device picks a random card from the bank, asks *you* the question, and
string-compares your answer. Inverts the whole system for free, and makes a
much better demo than watching tokens appear at 0.3/second.

### 3. Retry grammar-constrained sampling
**Cost:** ~1 hour, firmware only. The code exists (`FAILURES.md` #8).

Enable core debug logging *first* this time, so a silent delay cannot be
mistaken for a crash. If it works it makes tool syntax structurally
unbreakable — and would suppress the malformed `<quote>` calls the shipped
model occasionally emits.

---

## Tier 2 — worth doing, larger cost

### 4. Find out where the time actually goes
**Cost:** ~3 hours of on-device instrumentation.

Measured token time is ~20× both the arithmetic floor and the PSRAM
bandwidth floor. Nobody has measured which part of a token actually costs
what. Time the matmul in isolation, then the attention, then the link
round-trip.

Until this exists, every speed optimisation is guesswork — as the SIMD
attempt demonstrated. This is the prerequisite for any serious performance
work, and it may reveal something cheap.

### 5. Knowledge bank audit and expansion
**Cost:** ~4 hours for the audit tooling; expansion is open-ended.

Failure #10 showed that card *shape* matters more than card count. Before
growing the bank, write a checker that flags any card not in the canonical
`The X of Y is Z unit.` form, and run it as part of the build.

Then expansion becomes safe: Simple Wikipedia extraction into canonical
cards could take the bank from 792 keys to tens of thousands, all on the SD
card, no retraining. The device already reads from SD in ~11 seeks
regardless of bank size — binary search does not care.

### 6. Two-key comparison — first multi-hop capability
**Cost:** data + fine-tune + evaluation ≈ 5 hours.

"Which is denser, iron or tin?" requires two retrievals and a comparison.
Every current capability is single-hop. This would be the natural next
headline after axis transfer, and it composes existing machinery rather than
adding a new tool.

Give it a real share of the training mix — see failure #3 for what happens
to capabilities that get scraps.

---

## Tier 3 — speculative, high cost

### 7. Per-layer embeddings, revisited
**Cost:** 30+ hours, and one round has already been spent.

The first attempt (`FAILURES.md` #6) produced a clean negative: better
language modelling, dramatically worse reasoning, almost certainly because
the dense core was too narrow. Two things would make a retry worthwhile:

- The corrupted data loader that affected the entire first attempt is now
  fixed. Every number in that experiment deserves an asterisk.
- The core width question was never isolated. A PLE model with a
  reasoning-sized core — width 384 or 512, table sized down to fit — is the
  experiment that was actually intended.

Do not start this without first checking that the model image fits flash.
The width and the table size trade against 14MB.

### 8. Voice input and output
**Cost:** ~8 hours plus about $5 of parts.

Text-to-speech and a small command recogniser fit comfortably on this
hardware alongside the model. It would make the device standalone. It adds
no reasoning capability, but it changes what the thing *is*.

### 9. Additional tools: unit conversion, date arithmetic
**Cost:** ~5 hours each, or ~5 hours bundled.
**Odds:** poor as scoped.

Both are mechanically easy — the chip computes, the model routes. But
failure #3 is unambiguous: a tool at ~5% of the training mix is not learned,
and leaves malformed calls behind. Doing these properly means giving each a
substantial share, which means displacing existing types, which risks the
dilution that has collapsed capabilities before.

Recommended only as part of a deliberate mix redesign, not as additions.

---

## Explicitly not doing, with reasons

**BitNet / ternary weights.** Requires quantization-aware training from
scratch and a rewritten training forward pass. Proven at 2B parameters,
unproven at 27M, where extreme quantization hurts more. Multi-week gamble
for a speedup that memory-latency analysis suggests may not materialise.

**Streaming weights from SD.** SPI gives 1–2 MB/s. A 17MB model means 10+
seconds per token. Physics.

**Speculative decoding.** Requires a draft model sharing the target's
tokenizer. Would mean training a second model and holding both in memory,
for a gain smaller than the memory cost.

**Tensor parallelism across the two boards.** UART bandwidth is three orders
of magnitude short of what per-layer activation sharding needs.

---

## The honest strategic picture

This model is at the edge of what 27M parameters do on this hardware. The
remaining gains split into three kinds:

**Cheap and real** — better data for capabilities that already almost work
(paraphrase), and firmware features that cost nothing in model quality
(quiz mode, grammar masking).

**Expensive and uncertain** — architecture changes. The PLE experiment
consumed 30 hours and returned a negative result. That result is genuinely
valuable, but it was expensive, and the next architecture experiment will be
too.

**Blocked by physics** — speed. At 0.3 tok/s this is a demonstration, not a
product. Getting to interactive speed on this hardware likely requires a
fundamentally smaller compute core, and the PLE result suggests a smaller
core cannot do this reasoning. That tension is unresolved and may not be
resolvable at this parameter count.

The most valuable single thing that could come next is not on this list: it
is someone else reproducing the results, or the failures, on their own
boards.
