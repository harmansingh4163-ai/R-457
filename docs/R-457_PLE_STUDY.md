# R-457-PLE — Architecture Study (v2 design input, Jul 30)

Source: slvDev/esp32-ai (cloned, read). Their published result: 28.9M
stored params at 9.5 tok/s on one ESP32-S3 — ~559K dense core hot,
25M-param per-layer embedding table memory-mapped in flash (12MB at
4-bit), d_model 96, 6 layers, ple_dim 128, vocab 32768, TinyStories.

================================================================
FINDING #1 — THE ONE THAT CHANGES R-457's PLAN
================================================================
PLE's benefit SCALES WITH VOCAB. Their controlled comparison
(core-matched arms, 2 seeds each):
    vocab 32768: PLE edge +0.098 nats over matched core
    vocab  4096: PLE edge +0.025 nats; the adapters-without-table arm
                 ("ple_notable") is WORSE than baseline
Their conclusion: at realistic vocab, WHERE the params are injected is
worth ~2x params-at-the-bottom (their "fatembed" control — same table
budget spent on a wide input embedding — loses to PLE).

IMPLICATION: R-457's tok4096 was the right call for dense (shrink the
embedding, spend params on layers) and is the WRONG call for PLE (the
table is vocab x n_layers x ple_dim — big vocab IS the capacity, and
it lives in flash where it is nearly free). R-457-PLE should move to a
~32k tokenizer. Costs: retrain sentencepiece (keep character_coverage
0.9995 and the 5x R-457 weighting), re-tokenize ALL data, re-export,
firmware tokenizer swap (encoder already vocab-driven after tonight's
fix — the special-token pre-pass scans whatever vocab it gets). All
mechanical, no unknowns.

Side benefits of 32k for R-457 specifically: bank subjects and units
become single tokens; possibly better digit handling (test, don't
assume — decimals were a tok4096 failure mode).

================================================================
FINDING #2 — WHAT'S REUSABLE, FILE BY FILE
================================================================
src/model.py    Clean pluggable arms: baseline / ple / ple_notable /
                fatembed. The "ple" arm is faithful Gemma-3n-style:
                per-layer input = RMSNorm(proj(per-layer embed row))
                added via small adapters. THIS IS THE v2 MODEL FILE —
                adapt, don't rewrite.
src/train.py    Their own trainer (not llama2.c). Either port R-457's
                staged fine-tune discipline onto it, or graft the PLE
                arm into llama2.c's model.py. Decide by reading both;
                grafting keeps eval_fixed/evaluate_tools unchanged.
src/quantize.py 4-bit PTQ; RESULTS confirms the PLE gain SURVIVES
                4-bit. Reuse.
src/export.py + firmware/esp32_llm/  The flash-table mmap pattern:
                table stays in a flash partition, ~6 rows (~450B) read
                per token, 0.12ms — 0.7% of token time. Merge this
                pattern into llm_core rather than adopting their
                firmware wholesale (R-457 needs tools/KB/link/learn).
src/budget.py   Sizing calculator — use it to lay out the probe ladder.
firmware/bandwidth_bench/  Measures flash read throughput on-chip.
                RUN THIS FIRST on the Waveshare/Guition — their number
                is for their board; ours sets the table-size ceiling.
host_verify/    Their bit-exact host harness — same philosophy as
                R-457's host tests; reuse the pattern.

================================================================
FINDING #3 — THE CORE-SIZE QUESTION IS R-457's REAL EXPERIMENT
================================================================
Their 559K core writes stories. R-457 needs tools, extraction,
refusal, transitive chains — nobody knows the minimum core for that.
This is the publishable experiment ("PLE + instruction-following").

Probe ladder (30-min-probe discipline, eval_fixed + per-type accuracy
as judges, ft6's table as the dense baseline to beat):
    core ~1.5M (dim 192-224, 6L)  — slvDev-scale, probably too small
    core ~3M   (dim 256, 6-8L)    — the interesting middle
    core ~6M   (dim 320, 8L)      — safety margin
Each with a 32k-vocab PLE table sized to flash: n_layers x ple_dim x
32768. Example: 8 x 64 x 32768 = 16.8M table = ~8.4MB at 4-bit — fits
the 14MB model partition WITH the core and tokenizer. budget.py to
refine.

Speed projection (estimate, to be MEASURED): dense 27M ran 0.25-0.5
tok/s split across two boards. A ~3M core is ~9x less compute ->
roughly 2-4 tok/s on ONE board, plus SIMD (backlog #8) 1.5-2.5x ->
plausibly 4-8 tok/s interactive. Single board. That is the product.

================================================================
KEEP / CHANGE LIST FOR v2
================================================================
KEEP: staged training + reset_ckpt, eval_fixed.py, evaluate_tools.py,
  construct.py data (retokenize only), fine-tune mix ratios as starting
  point, all firmware capabilities (tools, KB, /learn, /refused,
  /selftest, confidence, boot check), the two-board rig as regression
  testbed, hardware canaries.
CHANGE: tokenizer 4096 -> ~32768; model.py gains the PLE arm; export
  writes core + table as separate sections (table at its own flash
  offset); llm_core reads table rows per token (merge slvDev pattern).
MEASURED (Jul 30, Waveshare, bandwidth_bench.ino — step 1 DONE):
  PSRAM seq 60.7 MB/s | SRAM seq 240 MB/s (351KB free) | flash random
  512B row 20.3us -> 6 rows/token = 122us. VERDICT: table cost is ~0.5%
  of a 25ms token — PLE table confirmed ~free on our silicon.
  CROSS-CHECK: dense 27M's bandwidth ceiling would be ~3.5 tok/s but we
  measured 0.25-0.4 -> the dense model was COMPUTE-bound. Therefore:
  (a) 3M-core projection ~3.4 tok/s raw, ~5-8 with SIMD, single board;
  (b) SIMD (backlog #8) is promoted to the v2 critical path — it is the
  difference between 3 tok/s and 8.

FIRST THREE CONCRETE STEPS:
  1. DONE — numbers above.
  2. Train tok32768 (same corpus recipe), retokenize pretrain data.
  3. Graft PLE arm into llama2.c model.py; pretrain the 3M-core probe;
     judge with eval_fixed before any fine-tune.
