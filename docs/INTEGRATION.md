# Wiring verify.c into the sketch — 4 edits

Sketch-folder rule applies: the folder gains exactly two files
(`verify.h`, `verify.c`) and nothing else. `vt_main.c` is a HOST test
harness — it has its own main(), so it must NOT enter the folder
(same reason build_kb.py and test_kb.cpp stay out).

## 1. Include (top of the .ino, after llm_core.h)

```c
#include "verify.h"
```

## 2. Keep the prompt visible to the verifier

`generate()` already has everything needed: `prompt` (facts+question)
and `gen` (the accumulated output buffer). At the END of `generate()`,
after the tok/s printf, add:

```c
  static char vrep[192];
  verify_rc v = verify_answer(prompt, gen, vrep, sizeof vrep);
  if      (v == VERIFY_PASS)    Serial.printf("[verified: %s]\n", vrep);
  else if (v == VERIFY_REFUSAL) Serial.printf("[verified refusal]\n");
  else                          Serial.printf("[VERIFY FAIL: %s]\n", vrep);
```

That is the whole feature: grounding check + inline citations
("660 <- fact 2") + tool recheck on every answer, ~1 ms of C.

## 3. Self-test at boot (pairs with the existing canaries)

In `setup()`, after `learn_space_token()`:

```c
  verify_selftest();   // prints 8 lines; all planted errors must be caught
```

Project rule satisfied mechanically: the checker proves itself against
planted errors on every boot, so a silent hollow check (check_count /
check_lookup history) cannot recur unseen.

## 4. Optional serial command

In the `/`-command chain (note: INSIDE the existing chain — remember the
`line[0]=='/'` dead-code lesson; this must be an `else if`, not a new
`if` block):

```c
  else if (strcmp(line, "/verify") == 0) verify_selftest();
```

## Pipeline (two-board) build

Same edits, HEAD sketch only. The head owns generation and the full
text buffer; the worker never sees text. Nothing changes on the worker.

## What to expect on real output

- /ask answers: `[verified: 660 <- fact 2]` — the citation line.
- Decimal truncation (open failure mode): now flagged live as
  `[VERIFY FAIL: UNGROUNDED number '103' not in facts]` instead of
  passing silently. Until ft7's <quote> lands, this makes the failure
  VISIBLE, which is the honest intermediate state.
- Known cosmetic: a number appearing twice cites twice. Harmless.
- Known limit: exact-string match. If a future bank stores "660.0"
  and the model says "660", that is a false FAIL. Canonical bank
  values avoid this today.
