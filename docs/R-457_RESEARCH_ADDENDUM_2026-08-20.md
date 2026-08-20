# R-457 — External Review + Deep Research Addendum — 2026-08-20

Follow-up to R-457_REVIEW_SIM_2026-08-08.md and the two 2026-08-14 research
docs (hemisphere/router, neuroscience). Scope this time: full re-read of both
repos (`R-457` and `ESP-32-s3-Story-maker-LLM` — core, sketches, toolchain,
tests, all docs), cross-check of the attached research files against the
actual code, analytical verification of the key numbers, and a new round of
literature/external research (slvDev esp32-ai PLE, NanoMind-S3, PIE/SIMD
threads, ESP32-S3 cache/optimization write-ups, MoE-at-small-scale training
stability).

## VERDICT UP FRONT

The code is honest and the attached research docs survive cross-examination.
What has NOT happened is execution of the already-identified cheap fixes —
several items from the 08-08 review (INT8 link, D-1 backport, stale
ARCHITECTURE numbers, duplicate includes, checkpoint default) are still
open in `main` while attention has moved to research questions. And the new
research changes the plan in two places: the `-O2` A/B test is now suspect
(a March 2026 atomic14 benchmark found `-Os` beating `-O2` on ESP32-S3 hot
paths due to icache pressure), and the MoE probe needs router z-loss +
load-balancing loss from iteration 0 (naive small-scale MoE training is
known to collapse).

The biggest conceptual shift: **the "20× off the floor" framing may be a
measurement artifact** — comparing against a sequential-read bandwidth
figure (~40MB/s) when the workload's actual access pattern is random
32-byte cache lines from an mmap'd flash region. If random-line reads
measure ~8MB/s, the true ceiling is ~2.1 tok/s and today's 0.33 tok/s is a
~6× gap, not 20× — a very different optimization budget. This is precisely
FAILURES.md #11's warning class: a summary figure never measured under the
workload's real access pattern.

---

## 1. What was verified (and what wasn't)

**Verified in code:**
- `llm_core.c` does what the docs claim: INT4/INT8 group dot products,
  const-pointer weights (mmap-friendly), pluggable dual-core matmul hook,
  the special-token pre-pass and `<0xNN>` byte fallback (FAILURES.md #2's
  fix is genuinely present).
- The two-board pipeline protocol is minimal and idempotent-by-construction
  (deterministic, pos-addressed `llm_layers` → identical K/V on retry).
- Story repo's `test_host.c`/`test_pipeline.c` are real bit-exactness
  harnesses, not theater.
- The router experiment's mechanism is real in `construct.py`: logic types
  use nonsense vocabulary by design, lookup uses real material vocabulary
  by design — the two vocabularies ARE disjoint by construction, for an
  unrelated reason, and that transfers to the 7-line regex.

**Still open from the 08-08 review (found unshipped in `main` today):**
- `ARCHITECTURE.md` still says "~6.75MB streamed per token" (measured layout
  cost is 7.92 worker / 9.23 head MB per board per token) and still says
  "INT8 payloads" cross the link (code sends fp32).
- Story repo `llm_core.c` is still missing the D-1 decode bounds guard
  (file sizes 18,061 vs 18,185 bytes — the diff IS the guard). Two-line
  backport, still outstanding.
- Duplicate `<SPI.h>`/`<SD.h>` includes still present in `pipeline_head.ino`.
- `train.py` `always_save_checkpoint` default — verify it's flipped; it's
  the documented 12-hour-loss footgun.

**New minor issues from this read:**
1. `pending_tool()` scans a 1024-byte `gbuf`; a long reasoning tail before a
   `<calc>` call can push the tag out of the window and silently drop a valid
   tool call. Safe at `/len 180`; not safe if maxlen is raised. Fix: restart
   accumulation on tag-open, or ring-buffer the last ~256 bytes plus any open
   tag's span.
2. `llm_sample()`'s top-p pool is `static` — single-task today, but if
tier-2
   escalation ever samples from a second task it becomes a race. Mark it
   non-reentrant or move the pool into `Sampler`.
3. `spec_ids[16]` cap in the encoder's special-token scan — fine at ~5
   symbols today; flag it so a future tool pileup doesn't silently drop
   tokens.

---

## 2. New research findings

### 2.1 The `-O2` hypothesis is now suspect

The 08-08 review's #1 action was the Arduino `-Os` vs `-O2` A/B. A March 2026
atomic14 ESP32-S3 benchmark found the opposite of the hypothesis: `-Os` beat
`-O2` on a hot path, because `-O2`'s larger code inflated icache pressure and

evicted flash-cache lines the loop needed. R-457's hot loop competes with a
17MB weight stream for a 64KB cache — instruction size is a first-class
performance variable here.

**Revised action:** still run the A/B (15 minutes), but three-way:
`-Os` vs `-O2` vs `-Os` + the matmul inner loop placed in IRAM
(`__attribute__((section(".iram")))`). IRAM placement removes the hottest
~2KB of code from the flash-cache equation entirely and is a more likely
mechanism for the NanoMind 2.1× gap than the flag itself. If the flag flip
shows nothing, conclude "cache pressure eats it" and go to IRAM, not
"compiler isn't it."

### 2.2 The MoE probe has a known landmine — and a known fix

"Does a tiny top-1 MoE train on MPS without freezing" has a documented
default failure mode: early router collapse (one expert wins, the rest never
get gradient, router logits blow up, training freezes or diverges). Bake
the mitigations in from iteration 0:
- **Router z-loss** (ST-MoE lineage): λ·mean(router_logits²), λ≈1e-3 — keeps
  probabilities in fp32-safe territory, reported to improve stability without
  quality loss.
- **Auxiliary load-balancing loss** (Switch Transformer lineage): penalize
  mismatch between each expert's probability mass share and assigned-token
  share. Standard coefficient 0.01.
- Keep the router in fp32 regardless of what gets quantized later; init
  router weights small (std ~0.01).

Probe design: `dense_ffn` vs `top1_moe_ffn` (8 experts, hidden 172 each to
match params), identical seeds, both losses in the MoE arm, 2k iterations.
Pass criterion defined BEFORE running: no loss divergence AND expert
assignment entropy stays above ~0.7 of max. The loss≠capability discipline
from PLE applies in full — the probe tests trainability, not capability.

### 2.3 The bandwidth mystery has a structural suspect

Candidates for the ~40MB/s → ~2MB/s effective gap, tightened:
1. **Scale/weight interleaving.** `qmat_attach` puts all scales before all
   packed weights per tensor, so one row's traversal ping-pongs between two
   distant flash regions — worst case for cache line utilization. A layout
   change in `export_model.py` (interleave each group's scale with its packed
   bytes) attacks this WITHOUT touching the firmware hot loop beyond pointer
   arithmetic. Testable on the host simulator first.
2. **Row-stride eviction** — consecutive rows are contiguous, so this should
   NOT be the killer; rule it out by measurement, not assumption.
3. **MMU/mmap random-access overhead** — the 40MB/s figure is a sequential
   number. The honest benchmark is *random 32-byte reads from the mmap'd
   partition*, which is the workload's actual access pattern. That number
   is the true ceiling for the current layout.

**This reorders the plan:** the first measurement is now a random-line mmap
read benchmark, not a cache-resident matmul — it decides whether the gap is
20× or ~6× before any optimization is chosen.

### 2.4 Strategic cross-check

- slvDev esp32-ai: 28.9M @ ~9.5–9.88 tok/s single-board via per-layer
  embeddings. R-457's own PLE negative (FAILURES.md #6) remains the correct
  counter-evidence that PLE doesn't transfer to reasoning tasks — press
  coverage headlines "writes stories at 9.5 tok/s," exactly the capability
  class the PLE study showed collapses under a narrow core. The two projects
  are complementary opposites: they have speed without reasoning; R-457 has
  reasoning without speed.
- MoE-at-tiny-scale has real recent literature (nanoMoE walkthroughs,
  similarity-preserving load-balancing losses reporting 36% faster
  convergence over aux-loss baselines) — the top-1 MoE FFN direction is
  supported, not a one-repo anecdote.
- Top-1 MoE FFN math, restated for the record: FFN is 8.46MB of the 17.15
  MB/token traffic; top-1 of 8 experts cuts active traffic to ~1.06MB →
  ~9.7MB/token, a ~1.7× ceiling — and unlike PLE it keeps attention and
  residual width full, dodging failure #6's diagnosed cause.

---

## 3. Revised priority stack (delta vs 08-08 review)

| # | action | change |
|---|---|---|
| 1 | Random-32B-read mmap benchmark | NEW — decides if the gap is 20× or ~6× |
| 2 | IRAM placement of matmul inner loop, then 3-way
`-Os`/`-O2`/`-Os+IRAM` A/B | REVISED — `-O2` alone now suspect |
| 3 | Interleaved scale/weight layout in `export_model.py`
(host-testable first) | NEW — attacks the two-region ping-pong |
| 4 | INT8 link quantization (3.9× payload cut, ~66ms/token, zero
risk) | UNCHANGED — still the only free win, still unshipped |
| 5 | Top-1 MoE FFN probe WITH z-loss + load-balancing from
iteration 0 | STRENGTHENED — naive version is known to fail |
| 6 | Tier-1 dual specialists (router is measured; two fine-tunes
from `ckpt_base_backup.pt`, different `--mix`) | UNCHANGED |
| 7 | Consolidation habit (CLS): interleave + refusal-weighted
replay for the next fine-tune | UNCHANGED — a rule, not a project |
| 8 | Doc/D-1/duplicate-include fixes from the 08-08 review | UNCHANGED — still open, still 30 minutes |

Known-effort order (not value order): 8 → 4 → 1 → 2 → 3 → 5 → 6 → 7.

---

## 4. New minor findings from the full re-read

(Consolidated from §1 — the three code-level items: `pending_tool()` window,
`llm_sample()` static pool reentrancy, `spec_ids[16]` cap. None block
anything today; all three deserve a comment line each so they don't become
silent failures later, per FAILURES.md #11's general lesson.)

---

## 5. Verdict

The attached research docs are unusually good: measured claims, honest
negatives, correct citations. The repos are even better: the code matches
the prose. The gap is execution of the already-known cheap items. Achieving
"something big" here does not require a new idea — it requires shipping the
measured wins (INT8 link, tier-1 specialists, the mmap benchmark, the
MoE-FFN probe) and letting the two-board setup become what it is uniquely
positioned to be: the only published ESP32-class system that reasons over
retrieved facts and refuses honestly, at whatever speed it can honestly
sustain.

## Sources
- atomic14, "I Made My ESP32-S3 Faster by Optimizing for SIZE — Not Speed"
(2026-03) — -Os beats -O2 via icache pressure
- slvDev/esp32-ai — 28.9M-param PLE LLM on ESP32-S3, ~9.5–9.88 tok/s
- ST-MoE / Switch Transformer lineage — router z-loss, load-balancing
auxiliary loss; nanoMoE and similarity-preserving router follow-ups (2025)
- Espressif ESP-IDF SPI Flash / mmap documentation (v6.x)
- Internal: R-457_REVIEW_SIM_2026-08-08.md, FAILURES.md, ARCHITECTURE.md,
llm_core.c, both pipeline sketches, both repos' tests/ and pc_tools/
