# R-457 / 27M — Hardware Session (Jul 29, 2026)

The night the 27M reached silicon: ft6 split across both boards, pipeline
firmware compiled for Arduino for the first time, two encoder bugs found and
fixed at the root, both knowledge canaries green over the two-board link, and
the tok/s number that settles the architecture question.

STATUS: 27M TWO-BOARD PIPELINE HARDWARE-VERIFIED. DENSE 27M MEASURED
TOO SLOW TO SHIP (0.25–0.5 tok/s). PLE IS THE v2 DIRECTION.

================================================================
WHAT IS DEPLOYED RIGHT NOW
================================================================
Model:      out27m_ft6 (iter 3000), INT4 gs32 seq 256
Export:     r457_27m.bin 17.21MB full -> split_image.py 4 ->
            r457_27m_worker.bin 7.92MB (layers 0-3)
            r457_27m_head.bin   9.29MB (emb + layers 4-7 + cls + tokenizer)
Boards:     Guition  = WORKER, port ...21101, native USB
            Waveshare 4.3 = HEAD, port ...5AB01600611, CH343 bridge
Firmware:   ~/Documents/Arduino/pipeline_worker/      (5 files)
            ~/Documents/Arduino/pipeline_head_r457/   (8 files)
Link:       head TX=GPIO8 RX=GPIO9 (the I2C terminal block used as plain
            UART), worker TX=17 RX=18, GND-GND, 460800 baud.
            Zero timeouts, zero CRC errors, entire session.
KB:         flash bank, 792 keys, via kb_open (SD path removed from head).

THE GUITION'S v9d IS OVERWRITTEN. Rollback is one line:
    python3 -m esptool --chip esp32s3 --port /dev/cu.usbmodem21101 \
        write_flash 0x1F0000 r457_v9d.bin
(but re-upload esp32_storyteller with the FIXED llm_core.c first — see
THE ENCODER BUG below; the old sketch + old core is the pre-fix pair.)

================================================================
VERIFIED ON HARDWARE (all over the two-board link, temp 0)
================================================================
1. physics + tool:  55V / 5 ohm -> "<calc>55/5=" -> chip injects
   "11</calc>" -> "So the current is 11 amps." Clean single tool call.
2. SILICON CANARY:  /ask atomic mass of silicon -> 28.085 EXACT,
   six-token decimal copy, atomic-number-14 trap fact present and ignored.
3. ALUMINIUM CANARY: /ask melting point of aluminium -> 660 EXACT,
   water-melting-point trap fact present and ignored.
4. KB retrieval: 2 facts per question from the 792-key flash bank.
5. count: strawberry,r -> <count> -> injected 3 -> "Answer: 3"
6. missing-input refusal: Ohm's law, no resistance -> names the gap,
   refuses, ZERO tool calls
7. underdetermined refusal: glork/dax both vs wug -> "compared to the
   wug, but not to each other" -> Cannot be determined
8. <write> (first hardware run, mode id 10 verified): aluminium
   paragraph, both numbers grounded, no invented values
THE FULL VERIFICATION MATRIX IS GREEN — every trained capability
(calc, count, lookup x2, refusal x2, write) verified across the link.
9. SD BACKEND LIVE: CH422G latched at boot (CS/EXIO4 low, USB_SEL low,
   backlight off), then 8/9 re-muxed to the UART link — kb: SD ok
   (480 MB), 792 keys read from /kb.bin, silicon canary 28.085 FROM THE
   CARD. kb.bin traveled Mac -> phone (http.server over WiFi) -> card.
   Coexistence design (I2C-latch-then-UART on shared pins) verified.
   NOTE: expander lib prints "version: 0.1.0" but uses the demo-zip
   API — the version string is not the discriminator, the API is.
10. THREE NEW FIRMWARE FEATURES, all verified same night:
    /selftest — runs both canaries, string-checks answers, prints
      PASS/FAIL. First run: PASS.
    boot check — encodes "<reason>" and "\n" at startup, prints
      mode-token/newline ok|SHATTERED|WRONG-ID. Instant guard for the
      encoder bug class found tonight.
    confidence — logits snapshotted pre-sample, softmax p of chosen
      token; footer now prints conf avg/min, "(unsure)" below avg 0.60
      (threshold uncalibrated first guess). Observed: canaries avg
      0.98-0.99; min tracks first-token choice points (0.16-0.66).
    /learn — appends fact to /learned.txt on SD; /ask splices keyword-
      matched learned facts into the prompt. VERIFIED: copper 1085
      taught at runtime, answered correctly, bank untouched. Runtime
      teaching without the Mac. Known nit: no dedupe (doubled command
      stores the fact twice; harmless).

================================================================
THE MEASUREMENT — the number the whole arc chased
================================================================
Dense 27M across two boards: 0.25–0.5 tok/s (79 tok / 316s with tools;
42–53 tok / 119–138s without). Link cost at 460800 is ~90ms of a ~4s
token (~2%): this is COMPUTE/PSRAM-BOUND, not link-bound.

Conclusion: dense 27M on this hardware is a working TESTBED, not a
shippable interactive device. SIMD (~1.5–2.5x) lifts it to ~0.5–1 tok/s
at best. slvDev's esp32-ai runs 28.9M at 9.5 tok/s by keeping only a
~560K dense core hot and memory-mapping a 25M embedding table in flash
(Per-Layer Embeddings). Nobody has published PLE + instruction-following
+ tools + refusal. That combination is R-457's v2 architecture and the
open first-mover claim.

================================================================
THE ENCODER BUG — two symptoms, one root cause
================================================================
The C encoder (llm_core.c llm_encode) could not produce user-defined
symbol ids and used a legacy byte-fallback offset:

  SYMPTOM A: "<reason>" shattered into text pieces (63 282 2298 ...) —
  every prompt was out-of-distribution at token one. Visible as the model
  echoing "Question:" back and never settling.
  SYMPTOM B: '\n' encoded as id 13 (<0x02>) instead of 21 (<0x0A>):
  sentencepiece stores bytes as literal "<0xNN>" pieces, and tok4096's
  reserved specials shifted the byte block so the old byte+3 offset was
  wrong. Every line boundary in every prompt was a corrupted token.
  SYMPTOM C (same bug, second mask): the "114/calc6=" tool loop. The
  chip injects "11</calc>"; the broken encoder shattered "</calc>" into
  mis-offset byte ids, corrupting context after every tool call — 13
  spurious calls per generation.

FIX (llm_core.c, applied and hardware-verified):
  1. Special-token pre-pass: at first encode, scan the vocab for pieces
     matching <...> (excluding <0x..>), then match them atomically during
     encoding. Vocab-driven — correct for tok1024 AND tok4096, no
     hardcoded ids.
  2. Byte fallback looks up the "<0xNN>" piece BY NAME, falling back to
     byte+3 only if absent.
Verification: prompt ids now byte-identical to sentencepiece:
  1 4017 9 21 4062 ...  (BOS, space, <reason>, \n, 'F')

PROPAGATION: the fixed llm_core.c is copied to pipeline_worker/ and
esp32_storyteller/. THE STORYTELLER ON-DISK SKETCH CARRIES THE FIX BUT
THE 12M BOARD DOES NOT until re-uploaded. The 12M-era prompts had no
mode tokens, so the bug was latent there — but any future format with
specials would have hit it.

================================================================
HARDWARE FACTS PAID FOR THIS SESSION
================================================================
- Waveshare 4.3 USB-C goes through a CH343 UART bridge on UART0.
  Therefore: USB CDC On Boot = DISABLED for this board, or every
  Serial print goes to the void (symptom: ROM boot log, then silence).
  Guition is native USB: keep CDC ENABLED there. This is also why the
  Waveshare's esptool flash crawls at ~102 kbit/s (12 min for 9.3MB)
  while the Guition runs ~1156 kbit/s.
- Waveshare GPIO 17/18 are LCD blue-channel data lines (B2/B3) on the
  panel FPC — electrically busy and physically unreachable. The link
  uses the I2C terminal block as plain UART instead (TX=8, RX=9); the
  link protocol's CRC+retry makes the shared-block trick safe. Clean at
  460800 all session.
- The pipeline head's kb_init had its own SD-only path with the known
  placeholder pins (39/40/41/38). Replaced with the flash-bank kb_open.
  The SD backend (required for REFUSAL LOGGING later) still needs the
  CH422G expander init documented in R-457_HARDWARE.md — plain SD.begin
  can never work on this board.
- pipeline_head_r457.ino had g_m declared AFTER first use (host compile
  order hid it); declaration moved above learn_space_token.
- kb_build_prompt now emits "<reason>\n" before "Facts:" — the device
  prompt is byte-identical to to_text().
- /temp and /topp on the pipeline head set values SILENTLY (no echo).

================================================================
OPEN ITEMS
================================================================
1. Debug printf lines ({%d} per token, [i%d] per injected id) stripped
   from the head; the one-line "(tok ids: ...)" prompt fingerprint KEPT
   on purpose — it catches tokenizer mismatch instantly.
2. Commit + push all of tonight's edits to the GitHub repo (llm_core.c,
   kb.cpp, both inos, pipeline_link.h pins/baud, partitions.csv).
3. Storyteller re-upload with fixed llm_core.c before any 12M redeploy.
4. DONE — SD backend live via CH422G; refusal logging unblocked.
5. <write> mode untested on hardware.
6. EOS discipline: generation still runs past Answer into a fresh
   "Facts:" (visible in every run; /len contains it for now).

================================================================
NEXT — the v2 decision, now with data
================================================================
1. Study slvDev/esp32-ai training + quantization + firmware code (MIT).
2. Design R-457-PLE: dense core sized for the reasoning that ft6 proved
   (tools, extraction, refusal), flash-resident per-layer table for
   capacity. Target: single board, interactive tok/s.
3. This two-board rig is the REGRESSION TESTBED until PLE v1 exists —
   canaries and heldout parity run here.
4. Write-up thread: "four documented failure modes of extractive lookup"
   + axis-transfer result + the encoder root-cause story + first
   two-MCU LLM split — the Hackaday/Show HN arc is now complete enough
   to draft.
