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
