# ARCHITECTURE.md

The five hardware mechanisms R-457 actually runs on, each with a
pointer into the code and an honest status line. This file exists
because two external critiques in a row recommended these mechanisms
to the project as if they were missing. They are the implementation.

---

## 1. Zero-RAM weights: esp_partition_mmap

Weights live in a raw flash partition ("model", subtype 0x40, offset
0x1F0000) and are hardware memory-mapped read-only. RAM footprint of
the parameters: 0 bytes; the MMU serves reads through the flash cache.
INT4 (group-size 32) with INT8 activations, integer matmuls.

Code: sketch `load_model()` -> `esp_partition_mmap`; `llm_core.c`
`llm_init()` parses the image in place.
Status: verified on both boards; the reason "Erase All Flash" must
stay DISABLED (it wipes the model partition).
Measured consequence: ~6.75MB streamed per token through a 64KB
cache, fully evicted every token — the project's true bottleneck
(see 5).

## 2. Layer-sequential two-board split

Partitioning is by transformer layer index, nothing conceptual:
worker = layers 0–3 (`worker_27m.bin`, 7.92MB, Guition), head =
layers 4–7 + embedding + LM head (`head_27m.bin`, 9.29MB, Waveshare).
Both images fit the 14MB model partition per board. The head drives
generation; tools and the knowledge bank live on the head so injected
tokens take the normal path and both KV caches stay consistent.

Code: `split_image.py` (cut), pipeline head/worker sketches.
Status: hardware-verified (silicon 28.085 / aluminium 660 canaries on
both boards).
Honest cost, measured: two boards buy CAPACITY, not speed — one board
idles per token; total time ≈ single-board + UART overhead.

## 3. Interconnect: CRC16-framed UART activation pipeline

Intermediate activations cross boards as INT8 payloads in frames with
0xA5 0x5A sync bytes + CRC16, at 460800 baud over plain UART. On the
Waveshare the link uses the I2C terminal block pins (GPIO 8/9) as
UART, because GPIO 17/18 are LCD data lines. No RPC layer, no mesh —
a tensor pipe.

Status: hardware-verified, coexists with the CH422G latch for SD.
Why CRC on a 20cm wire: terminal-block jumpers at 460.8k do take
noise hits; a corrupted activation is silent garbage downstream,
a failed CRC is a retry.

## 4. Dual-core matmul via FreeRTOS task notification

Each matmul's rows split across both LX7 cores: core 0 worker task
takes the upper half, core 1 the lower, synchronized with
xTaskNotifyGive/ulTaskNotifyTake — no spinlocks, no queue overhead.
Small matrices (d < 64) skip the handoff.

Code: sketch `worker_task()` / `parallel_matmul()`;
`llm_core.c: llm_matmul_rows()` is the single hot loop.
Status: in production on every build.

## 5. SIMD (Xtensa PIE): investigated and closed

Host-side experiments on the INT4 path: word-loads + hoisted
zero-point ran 0.51× (defeats compiler auto-vectorization);
algebraic hoist alone 1.07× with vectorization off. Root causes:
nibble-unpack cost ≈ the multiply it feeds, and prior-art speedups
(260K-param models at 19–32 tok/s) were SRAM-resident int8 — a
different problem.
The decisive evidence is memory-side: ~3000 ms/token vs ~140 ms
arithmetic floor AND ~170 ms QIO-flash streaming floor — a 20× gap
over both. The workload is latency-bound on flash/PSRAM, so faster
arithmetic cannot help.

Status: REOPENED (Aug 4) after external correction. (1) Host results
do not transfer: Xtensa GCC cannot auto-vectorize to PIE, so the
on-chip scalar baseline is unvectorized and manual PIE headroom is
real (EE.VMULAS.S8: 16 int8 MACs/cycle; ~7 cycles per 32 int4 MACs
incl. nibble unpack). (2) On the in-order LX7, cache stalls and
compute SERIALIZE, so compute savings pay off proportional to compute
share of token time (unmeasured; ~10-min cache-resident matmul
benchmark decides). Endgame: GDMA double-buffered tiling + PIE kernel
restores overlap and approaches the streaming floor. Scope
note: true PIE assembly was never benchmarked on-silicon; reopen only
with an on-chip PIE benchmark, and only after the memory work lands.
Clock ceilings (measured from the shipped core, qio_opi variant):
flash QIO @ 80MHz, PSRAM OPI/OCT @ 80MHz; 120MHz needs a custom
ESP-IDF core build. No free clock win exists.

---

## Sequence budget

Training seq 512; **device seq 256**. KV cache is fp32:
    KV = 2 x seq x dim x 4B x layers_per_board
       = 2 x 256 x 512 x 4 x 4 = 4.2MB   fits 8MB PSRAM
       = 2 x 512 x 512 x 4 x 4 = 8.4MB   does not
Fine-tune p99 prompt length was 218 tokens, so 256 is comfortable.
PSRAM-aware allocators: KV/logits -> SPIRAM, hot buffers -> internal
SRAM. fp16 KV (queued, DISTANT_ROADMAP B4) would make device seq 512
fit in the same 4.2MB. The binding deployment constraint has been the
14MB flash partition per board, not the KV cache.

## Where the next speed comes from

Not clocks (ceiling reached), not SIMD (closed): access pattern.
Open audit question: what turns a ~40MB/s-capable sequential flash
path into an effective ~2MB/s in `llm_matmul_rows()` — scale/weight
stream interleaving, double-touched cache lines on nibble unpack, or
prefetch-breaking row strides.
