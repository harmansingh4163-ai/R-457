# MAX_MODEL.md — how big can R-457 get on this hardware?

Prompted by an external size analysis (Aug 3, 2026). Its two-wall
framework was right; its headline architecture was arithmetic-broken
and its main PSRAM lever is one this project has already measured as
unavailable. This file is the corrected answer of record.

---

## Wall 1 — flash (bites first)

14MB model partition per board × 2 boards = 28MB.
INT4 gs32 with fp32 scales ≈ 0.625 B/param; measured all-in cost is
0.68 B/param (17.21MB image / 25.3M params, includes tokenizer, norms,
headers). Raw fit ceiling: **~41–44M params**.

## Wall 2 — PSRAM (KV cache)

KV (fp32) = 2 × seq × dim × 4B × layers_per_board.
Current deployment: dim 512, 4 layers/board, **device seq 256** ->
4.2MB. seq 512 would be 8.4MB and does not fit at fp32.
(Training seq is 512; the device cap is a deployment fact. An earlier
draft of ARCHITECTURE.md said "seq 512 at 27M" — wrong, fixed.)

**fp16 KV halves this**: seq 512 fits in the same 4.2MB with today's
model. Firmware-only change. Queued (DISTANT_ROADMAP B4).

## Wall 3 — tok/s (the one that actually binds)

Measured: dense 27M across two boards = **0.25–0.5 tok/s**, link cost
~2%; the workload is memory-latency-bound, so token time scales with
bytes streamed. A 42M dense model therefore lands at ~0.15–0.3 tok/s:
a slower testbed, not a product. The dense-27M-as-product question was
closed by measurement; a dense-42M would reopen it with a worse
number.

**The honest ceiling question is not "what fits" but "what fits at
interactive speed" — and the answer is architectural, not
dimensional: PLE** (reference point: slvDev's 28.9M at 9.5 tok/s via a
~560K hot core + flash-mapped per-layer embeddings). That is v2.

## Corrected fit-ceiling math (for the record)

The external doc's "maxed-out" spec (dim 448, hidden 1200, 12 layers,
2 kv heads, tied 4K embedding) computes to **~27M**, not its claimed
42M: ~2.1M/layer × 12 + 1.8M embedding. Reaching ~42M dense at 12
layers needs dim ≈ 560–600 — which Wall 3 makes pointless anyway.

## Constraints from the training rig (why paper specs mislead)

- **GQA is unavailable**: n_kv_heads=4 froze at val 4.7699 for 5,000+
  steps on MPS/float32; n_kv_heads=8 trains (cost: one 12-hour
  overnight to learn). Any plan whose PSRAM savings come from
  "aggressive GQA" is a plan this rig cannot train.
- **Depth risk**: 16–20 layer attempts produced NaN/never-learned
  failures at 12M scale. 12 layers is untested, not established.

## Expansion paths, honestly priced

- **fp16 KV**: real, cheap, do it. -> B4.
- **gs=64**: ~1.5MB flash back (0.5625 B/param). Risk: coarser groups
  erode exact decimal copying — the product. Offline eval first. -> B5.
- **INT2**: rejected; lookup fidelity would not survive.
- **SD layer streaming**: rejected; adds 100–200ms/layer/token to an
  already 2–4s token.
- **Third pipeline stage**: needs a third ESP32-S3. The spare original
  ESP32 has no PSRAM and cannot hold even one stage's KV. Buy, don't
  repurpose.
- **Bigger flash**: ESP32-S3 memory-maps at most 32MB of flash — the
  zero-RAM esp_partition_mmap design holds to 32MB/board and no
  further. "Up to 1GB flash" is spec trivia that breaks the design.

## Verdict

Dense ceiling on this hardware: ~41–44M by flash, ~27–30M by sanity,
**~0 by product logic** — every dense parameter added makes the device
slower where it is already too slow. The ceiling-raiser is PLE, where
capacity moves into flash-mapped embeddings and stays off the
token-time critical path.
