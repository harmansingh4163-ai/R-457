# R-457 v2 Reference Notes — wladimiravila/esp32s3-distributed-ai

**Repo:** https://github.com/wladimiravila/esp32s3-distributed-ai
**Reviewed:** 2026-08-03
**Why it matters:** Same lineage as our v2 direction (slvDev/esp32-ai PLE), extended to a 56M model split across THREE ESP32-S3 N16R8 boards via ESP-NOW wireless. Trained on WikiText-103, 4-bit quantized, Gemma-style Per-Layer Embeddings.

---

## Ideas to steal for v2

### 1. KV cache on the core board (HIGHEST PRIORITY)
- Their biggest quality win. Before it, each token ran through the transformer with `seq_len=1` — attention never saw history, RoPE always at pos=0.
- Fix: KV cache in PSRAM on the core board — 1.5 MB for 256 positions. K/V appended per layer, attention over all cached positions, RoPE uses real position index. Cache resets on new-prompt message, caps at seq_len=256.
- **Action:** Check whether our worker/head split has the same seq_len=1 weakness. If yes, this is likely a bigger coherence win than any retraining. 8MB PSRAM has plenty of room for 1.5 MB.

### 2. Split-PLE across boards
- Their 50.3M-param PLE table (vocab × layers × 256 dims) is split into two 128-dim halves: PLE_A on one board (12.5 MB, 4-bit group=64), PLE_B on another (12.6 MB, group=128). Small `ple_model_proj` (~50 KB) stays local on the embedding board.
- This is how they fit a 56M model into 16MB-flash boards — directly answers our v2 partition math for going bigger than 27M on two boards.
- Note: PLE_A rows are fetched over the wireless link at runtime per token. That works at ESP-NOW latency; over our UART it would add per-token overhead — measure before copying.

### 3. ESP-NOW instead of UART
- Peer-to-peer wireless, no router. Packets ≤250 bytes with fragmentation, sequence numbers + acks. Claimed 1–5ms round-trip.
- vs our current pipeline link: GPIO 8/9 UART at 460800 baud. ESP-NOW frees the I2C terminal block and removes wiring, but adds WiFi radio power draw and protocol complexity.
- **Action:** benchmark ESP-NOW round-trip on our boards before committing; keep UART as fallback.

### 4. Web UI via Board C (nice-to-have)
- Third board runs WiFi AP + web server, streams generated text to browser via SSE. Good demo-piece pattern if we ever add a third board; not needed for the two-board product.

---

## Caveats (don't over-copy)

- **WikiText-103 training = Wikipedia-flavored free text.** No instruction-following, no tools, no KB lookup, no refusal. Their model is a different product class from R-457.
- **No tok/s numbers published.** Three-board hops per token + runtime PLE-row fetches could still be slow. Do not assume interactive speed.
- **Three boards, we have two.** Their PLE_A board doubles as decoder/WiFi. Our version would need PLE_A folded into the head board or fetched from SD/flash — repartition math required.
- Their boards are bare N16R8 devkits; ours have displays claiming GPIO 17/18 (Waveshare) — pin conflicts already documented in R-457_HARDWARE.md still apply.

---

## Files worth reading in their repo

- `firmware/common/llm.h` — distributed C inference runtime (read first)
- `firmware/common/espnow_protocol.h` — message protocol w/ fragmentation
- `src/model.py` — PLE TinyLM architecture (PyTorch)
- `src/quantize.py` — 4-bit group-wise quantization
- `src/export.py` — export to per-board binaries

---

## Proposed backlog additions

- [ ] Verify whether current 27M split runs attention at seq_len=1; if so, prototype KV cache in head-board PSRAM
- [ ] Read `llm.h` + `model.py` before starting v2 PLE implementation
- [ ] Benchmark ESP-NOW round-trip latency on Guition + Waveshare
- [ ] Work v2 partition math using Split-PLE pattern (two-board variant)

---

# Addendum — imFARSI/NanoMind-S3 (reviewed 2026-08-08)

**Repo:** https://github.com/imFARSI/NanoMind-S3
**What it is:** Karpathy stories15M (15.2M dense LLaMA-2), INT4-packed, running
**~2.96 tok/s on a SINGLE ESP32-S3** (N16R8, ESP-IDF, ~1000 lines of C).
Most directly applicable repo reviewed so far — it implements our "speed is an
access-pattern problem" thesis and hits the numbers.

## The three mechanisms (all portable to Arduino/R-457)

### 1. Flash MMU weight mapping — `main.c:127-129`
`esp_partition_mmap(part, 0, size, ESP_PARTITION_MMAP_DATA, &ptr, &handle)`
maps the 7.49 MB INT4 model partition straight into CPU address space.
Weights are read through the cache like normal memory: no SPI read calls,
no PSRAM copy, 0 MB PSRAM used for weights. This is the access-pattern fix,
already proven on this exact silicon. Callable from Arduino (ESP-IDF API is
linked in). Supersedes/complements the Cache preload plan — measure both.

### 2. Dual-core row-split matmul — `matmul.c` (186 lines, read whole file)
Persistent worker task pinned to core 1 (`xTaskCreatePinnedToCore`, prio 10),
two binary semaphores (start/done). Core 0 computes rows [0, n/2), core 1
rows [n/2, n). Our inference is single-core today — this is a near-2x on the
dominant op for ~100 lines of code. Works under Arduino (FreeRTOS present).

### 3. KV cache in PSRAM — `model.c`
3.54 MB for seq_len 256 at dim 288 / 6 layers. Independent confirmation of
our #1 backlog item with concrete sizing; their layout is worth copying.

## Calibration point for R-457
15.2M @ ~2.96 tok/s single-board implies our 27M should land ~1.5 tok/s on
ONE board with mmap + dual-core — vs ~0.3 tok/s on the current two-board
split. If that holds, the split exists only for models >single-board flash,
and the 27M product question reopens.

## Caveats
- ESP-IDF project; we are Arduino. APIs available, but partition table for
  the model region must be added to our scheme (they use a custom
  partitions.csv — read it).
- Bare devkit: no display, PSRAM bus uncontended. Our head board drives an
  LCD from the same PSRAM — expect some contention tax.
- 2.96 tok/s is their claim; not independently verified. First step is
  reproducing their benchmark on our Guition before porting anything.
- INT4 format differs (their per-row FP32 scale, nibble-packed signed
  [-8,+7]) — check against our export before assuming binary compat.

## Proposed actions
- [ ] Flash their firmware unmodified on a spare board; verify ~3 tok/s claim
- [ ] Benchmark esp_partition_mmap reads vs our current weight access path
- [ ] Prototype dual-core matmul in our runtime (self-contained change)
- [ ] Re-run the 27M single-board feasibility math if both land
