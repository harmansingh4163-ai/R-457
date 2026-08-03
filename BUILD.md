# BUILD.md

How to go from an empty pair of boards to a working R-457, and how to
rebuild the model from scratch if you want to change it.

Two paths:
- **Flash only** — download the prebuilt images, ~30 minutes.
- **Build from source** — train, export, split, flash. ~20 hours of compute.

---

# Part 1 — Flash only

## What you need

| item | notes |
|---|---|
| Guition JC3248W535C | ESP32-S3, 16MB flash, 8MB PSRAM — the *worker* |
| Waveshare ESP32-S3-Touch-LCD-4.3 | 16MB / 8MB — the *head* |
| microSD card, FAT32 | any size |
| 3 jumper wires | female-female depending on your headers |
| Arduino IDE 2.x with ESP32 board support | 3.x core tested |
| esptool | `pip install esptool` |

## 1. Get the images

From [Releases](../../releases):
- `r457_ft7b_worker.bin` (7.92 MB) — layers 0–3
- `r457_ft7b_head.bin` (9.29 MB) — embedding, layers 4–7, classifier, tokenizer
- `kb.bin` (100 KB) — the 792-key knowledge bank

## 2. Flash the model images

Find your ports (`ls /dev/cu.*` on macOS, Device Manager on Windows). Close
any serial monitor first.

```bash
# worker — the Guition
python3 -m esptool --chip esp32s3 --port /dev/cu.usbmodemXXXX \
    write_flash 0x1F0000 r457_ft7b_worker.bin

# head — the Waveshare
python3 -m esptool --chip esp32s3 --port /dev/cu.usbmodemYYYY \
    write_flash 0x1F0000 r457_ft7b_head.bin
```

Both must end with `Hash of data verified.`

> The Waveshare flashes at roughly a tenth the speed of the Guition — about
> 12 minutes for 9.29MB. This is not a fault. Its USB-C port goes through a
> CH343 serial bridge capped at 115200 baud, while the Guition uses the
> ESP32-S3's native USB.

## 3. Prepare the SD card

Copy `kb.bin` to the **root** of the card as exactly `kb.bin`, eject it
cleanly, and insert it into the Waveshare.

The card also stores `learned.txt` (facts you teach it) and `refused.txt`
(questions it could not answer). Both are created automatically.

## 4. Board settings in the Arduino IDE

Identical for both boards **except USB CDC On Boot**:

| setting | value |
|---|---|
| Board | ESP32S3 Dev Module |
| Flash Size | 16MB (128Mb) |
| Partition Scheme | **Custom** |
| PSRAM | OPI PSRAM |
| Flash Mode | QIO 80MHz |
| Erase All Flash Before Sketch Upload | **Disabled** |
| USB CDC On Boot | **Enabled** on the Guition, **Disabled** on the Waveshare |
| Upload Speed | 115200 |

Two of these will cost you an evening if you get them wrong:

> **Partition Scheme must be Custom.** The sketch folder contains a
> `partitions.csv` that defines the `model` data partition at `0x1F0000`.
> Any built-in scheme omits it and the firmware will not find the model.

> **Erase All Flash must stay Disabled.** Otherwise uploading the sketch
> wipes the model image you just spent 12 minutes flashing.

> **USB CDC On Boot differs per board.** The Waveshare's CH343 bridge is
> wired to UART0, so with CDC enabled every `Serial.print` goes to the
> native USB port that is not connected to anything — you see the ROM boot
> log and then silence. The Guition needs it enabled for the opposite
> reason.

## 5. Upload the sketches

- `sketches/pipeline_worker/pipeline_worker.ino` → the Guition
- `sketches/pipeline_head/pipeline_head.ino` → the Waveshare

The head sketch needs two libraries from Waveshare's demo repository, not
from the Library Manager: `ESP32_IO_Expander` and `esp-lib-utils`. The
Library Manager version boot-loops on this board. Copy them into
`~/Documents/Arduino/libraries/`.

The SD card on the Waveshare has no GPIO chip-select — it is held low by a
CH422G I2C expander. This is why the head sketch initialises the expander
before touching SPI, and why the serial log prints harmless
`IO 255 is not set as GPIO` warnings at debug level Error.

## 6. Wire the link

Boards powered off. Three wires:

```
head GPIO8  ──────→ worker GPIO18
head GPIO9  ←────── worker GPIO17
head GND    ────────  worker GND
```

Do not connect 3V3 or 5V between the boards — power each from its own USB.

On the Waveshare, GPIO 8 and 9 are the I2C terminal block, used here as a
plain UART. GPIO 17/18 are wired to the LCD panel and unavailable, which is
why the two boards use different pin numbers. Each board's
`pipeline_link.h` sets its own.

## 7. First boot

Power the **worker first**, then the head. Open the serial monitor on the
head at **115200**.

```
Pipeline HEAD board — two-chip LLM
tools ready: <calc> <count>  (space tok 4017)
boot check: mode-token ok, newline ok
kb: SD ok (480 MB)
kb: ready, 792 keys (SD)
Ready: emb + 4 local layers of 8 total.
```

If `boot check` reports `SHATTERED` or `WRONG-ID`, the tokenizer in the
flashed image does not match the firmware's encoder — re-export the
tokenizer and reflash (see failure #2 in `FAILURES.md`).

Then:

```
/selftest
```

Expect `SELFTEST: PASS (silicon ok, aluminium ok)` after about five minutes.
It is slow because it generates two full answers.

## Commands

| command | effect |
|---|---|
| `/ask <question>` | retrieve facts from the bank and answer |
| `/learn <fact>` | teach a fact, stored on SD, usable immediately |
| `/refused` | list questions it could not answer |
| `/selftest` | run the built-in canaries |
| `/temp <f>` `/topp <f>` `/len <n>` | sampling controls |

You can also type a full prompt directly, using `|` for newlines:

```
<reason>|Facts: Current equals voltage divided by resistance. The resistance
is 5 ohms. The voltage is 55 volts.|Question: What is the current in
amps?|Reasoning:
```

Expect roughly 0.3 tokens per second. A typical answer takes two minutes.

---

# Part 2 — Build from source

Only needed if you want to change the model. Requires Python 3.9+, PyTorch
with MPS or CUDA, and about 20 hours of compute.

## 1. Training data

```bash
python3 pc_tools/construct.py --n 60000 --out clean.jsonl
python3 pc_tools/mix_finetune.py --reason clean.jsonl --write write.jsonl \
        --out finetune.jsonl --reason-pct 60
```

`construct.py` generates all reasoning types with machine-checkable answers —
no teacher model is involved, so every label is correct by construction.

**Check the balance before training.** Any signal that appears more often on
positive than negative examples becomes a shortcut cue:

```bash
python3 -c "
import json, collections
r=[json.loads(l) for l in open('clean.jsonl')]
print(collections.Counter(x['type'] for x in r))"
```

## 2. Tokenizer and shards

The tokenizer is trained once over the whole corpus and then never
regenerated — several scripts in the wild will silently overwrite it, which
orphans every existing checkpoint.

```bash
python3 -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/tiny.txt', model_prefix='data/tok4096',
    model_type='bpe', vocab_size=4096, character_coverage=0.9995,
    split_digits=True, byte_fallback=True,
    allow_whitespace_only_pieces=True, normalization_rule_name='identity',
    user_defined_symbols=['<calc>','</calc>','<count>','</count>',
                          '<quote>','</quote>','<reason>','<write>'])"
```

`character_coverage=0.9995` rather than 1.0 — full coverage demands more
characters than the vocabulary can hold once the reserved symbols are
allocated.

Verify before proceeding: each reserved symbol must be a single ID, and
subjects in the knowledge bank should be single tokens.

## 3. Pretrain and fine-tune

```bash
# pretrain (~10 hours)
python3 train.py --out_dir=out_base --vocab_source=custom --vocab_size=4096 \
  --dim=512 --n_layers=8 --n_heads=8 --n_kv_heads=8 --max_seq_len=512 \
  --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=6e-4 \
  --max_iters=20000 --always_save_checkpoint=True --device=mps

# fine-tune (~2.5 hours)
python3 reset_ckpt.py --src out_base/ckpt.pt --dst out_ft
python3 train.py --out_dir=out_ft --init_from=resume --learning_rate=5e-5 \
  --max_iters=4000 --always_save_checkpoint=True --device=mps  # plus the same model flags
```

Three things that will otherwise cost you a run:

> **`--always_save_checkpoint=True` on every fine-tune.** The default saves
> only on validation improvement; a resumed run can inherit a low
> best-validation value, never beat it, and silently discard everything.

> **`--n_kv_heads=8`.** A value of 4 silently freezes training on MPS with
> float32.

> **Verify the data loader before trusting any loss.** See failure #1 —
> a silent MPS transfer bug corrupted every batch in this project for weeks.
> The training loop now asserts `X.max() < vocab_size` and
> `X[:,1:] == Y[:,:-1]`. Keep those assertions.

## 4. Judge the result

The training loop's validation column is not a quality measurement for
mixed-objective fine-tuning. Use both of these instead:

```bash
python3 pc_tools/eval_fixed.py out_ft/ckpt.pt          # same files, same windows
python3 pc_tools/evaluate_tools.py --ckpt out_ft/ckpt.pt \
        --eval-dir data/eval_all --vocab-size 4096      # per-capability accuracy
```

The second is the real verdict — it runs the tools and scores each capability
separately. Compare against the table in `README.md`.

## 5. Export, split, flash

```bash
python3 -c "from tokenizer import Tokenizer; Tokenizer('data/tok4096.model').export()"
python3 pc_tools/export_model.py out_ft/model.bin data/tok4096.bin \
        r457.bin --bits 4 --gs 32 --seq 256
python3 pc_tools/split_image.py r457.bin 4 r457_worker.bin r457_head.bin
```

> **Export the tokenizer before every flash.** Skipping it produces output
> that looks like well-formed text and is entirely wrong.

The split argument `4` is the number of layers assigned to the worker. The
head takes the rest plus the embedding, classifier, and tokenizer.

Then flash as in Part 1.

## 6. Rebuild the knowledge bank

```bash
python3 pc_tools/build_kb.py --out kb.bin
python3 pc_tools/build_kb.py --out kb.bin --query aluminium   # verify
```

Every card should be a single fact in the shape `The X of Y is Z unit.`
Multi-property summary cards derail extraction even when the correct card is
also present — see failure #10.

---

## Publishing a release

The model images are too large to keep comfortably in git history. Attach
them to a tagged release instead:

```bash
git tag v1.0 && git push origin v1.0
gh release create v1.0 r457_ft7b_worker.bin r457_ft7b_head.bin kb.bin \
   --title "R-457 v1.0 (ft7b)" \
   --notes "27M reasoning model, two-board split. Axis transfer 85.5%."
```

Keep `.bin` and `.pt` out of the repository:

```
*.bin
*.pt
data/
!partitions.csv
```

---

## Troubleshooting

| symptom | cause |
|---|---|
| ROM boot log then silence | USB CDC On Boot wrong for that board |
| `model partition mmap failed` | Partition Scheme not set to Custom |
| `boot check: SHATTERED` | tokenizer not re-exported before flashing |
| `link timeout` | TX/RX not crossed, or GND wire missing |
| repeated `crc error` | wires too long; drop `LINK_BAUD` to 115200 |
| `kb: SD.begin failed` | card not FAT32, or expander libraries from the wrong source |
| model answers with fluent nonsense | tokenizer mismatch between image and firmware |
| loss plateaus and never descends | verify the data loader before anything else |
