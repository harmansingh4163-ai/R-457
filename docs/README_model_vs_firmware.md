# What the model does vs. what the firmware does

A fair question about any tool-using micro-model: how much is the
neural network, how much is C code? Exact split:

| Decision / work                                   | Who does it |
|---------------------------------------------------|-------------|
| Decide WHEN a step needs arithmetic               | model — emits `<calc>` mid-reasoning |
| Compose the operand expression (`13-9=`)          | model — from its own trace |
| Execute the arithmetic                            | firmware — exact, microseconds |
| Retrieve K candidate facts for a question         | firmware — key match, SD/flash |
| Decide whether a retrieved fact ANSWERS the question | model — property/subject match |
| Refuse when it doesn't                            | model — generated text, not a fallback branch |
| Copy the value into the answer                    | model (the open decimal-truncation weakness lives exactly here; `<quote>` will route it to firmware, and we say so) |
| Generalize to unseen comparative adjectives       | model — 17.6% → 41.6% with a pretrained-English base, no firmware involved |
| Verify the final answer is grounded, cite the fact| firmware — `verify.c`, after generation |

## Why refusal is learned, not a lookup-miss branch

Retrieval returns facts either way; the model reads them and decides.
The evidence is a failure only a trained model can have: an early
version learned "single fact retrieved → refuse" as a spurious cue,
because negatives were built with 1 fact and positives with 2–4. The
fix was rebalancing the training distribution (construct.py v2.1), not
editing firmware — single-fact positives went 3.5% → 68%. A C-level
heuristic cannot fail that way. Full history in FAILURES.md.

## Ablations (run these to check us)

1. **Tools OFF** — disable the injector, rerun the eval: calc/lisp/forth
   collapse. Proves the chip executes the math.
2. **Distractor trap ON** — atomic-number trap fact present at /kbn 1:
   the model answers 28.085, not 14. Proves the model chooses.

Together they separate the two claims cleanly: execution is firmware,
decision is weights.

## What we do NOT claim

No open-ended reasoning, no world knowledge beyond the bank, no
instruction-following outside the trained formats. R-457 reasons over
facts it is GIVEN, in ten trained shapes, and its every numeric answer
is machine-verified and cited on-device.
