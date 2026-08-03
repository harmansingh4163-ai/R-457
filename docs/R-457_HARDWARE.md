# R-457 — Hardware Reference

Board-level facts that apply to BOTH models (12M r457_v9d and the 27M).
Model training and accuracy live in R-457_STATE.md and R-457_27M_STATE.md.

================================================================
INVENTORY — what exists and what each part is for
================================================================
Guition JC3248W535C (ESP32-S3, 16MB flash, 8MB PSRAM)
    THE DEPLOYED BOARD. Runs r457_v9d today. SD pins are NOT published
    anywhere findable, so no SD on this board.
    Port: /dev/cu.usbmodem21101

Waveshare ESP32-S3-Touch-LCD-4.3 (ESP32-S3-N16R8, 16MB flash, 8MB PSRAM)
    Spare today; becomes the HEAD board in the two-board 27M split, because
    it is the only board with working SD. 800x480 RGB touch LCD, unused.
    Port: /dev/cu.usbmodem5AB01600611
    NOTE: earlier docs called this "Touch-LCD-5". It is the 4.3. Pin maps
    differ across that family, so the wrong label would waste a session.

Original ESP32 (no PSRAM)
    NO ROLE. Cannot hold weights or KV cache. The only job it could take is
    tool execution (<calc>, <count>), and those run in microseconds on the
    S3 already — routing them off-chip adds serial latency and saves nothing.

512MB microSD card (SDSC, FAT)
    Verified working on the Waveshare. Not in use yet: the bank is 100KB and
    lives in flash. See FLASH BANK VS SD BANK.

Mac Mini M4, 16GB — training and toolchain. Username reu.

The two ports are visibly different, which is the cheap way to avoid flashing
the wrong board:
    ...21101          Guition   (carries the working v9d model at 0x1F0000)
    ...5AB01600611    Waveshare

================================================================
ARDUINO IDE SETTINGS
================================================================
For esp32_storyteller (the deployed 12M firmware):
    Board            ESP32S3 Dev Module
    Flash Size       16MB
    PSRAM            OPI PSRAM
    USB CDC On Boot  Enabled
    Partition Scheme CUSTOM        <-- requires partitions.csv in the sketch
    Erase All Flash  Disabled      <-- else the model at 0x1F0000 is wiped

For any OTHER sketch (probes, demos, the SD test):
    Partition Scheme must NOT be Custom, or the build fails with
        cp: .../tools/partitions/.csv: No such file or directory
    because Custom looks for a partitions.csv the probe sketch does not have.
    ALWAYS SET IT BACK TO CUSTOM before flashing esp32_storyteller again.

Sketch folder ~/Documents/Arduino/esp32_storyteller/ must contain EXACTLY
SEVEN files: esp32_storyteller.ino, kb.h, kb.cpp, kb_data.h, llm_core.c,
llm_core.h, partitions.csv. Anything else with its own main() breaks the
build. The .ino name MUST match the folder name.

================================================================
SD CARD ON THE WAVESHARE — SOLVED
================================================================
Status: verified working, NOT yet used by R-457 firmware.

## Wiring (from the Waveshare wiki, confirmed on hardware)
    ESP32-S3 GPIO11  -> MOSI
    ESP32-S3 GPIO12  -> SCK
    ESP32-S3 GPIO13  -> MISO
    CH422G   EXIO4   -> SD_CS, active low       <-- NOT a GPIO
    I2C to the CH422G: GPIO8 = SDA, GPIO9 = SCL

## The trick: SD_SS = -1
No GPIO chip-select is passed to the SD library at all. The real CS is held
LOW on the expander for the whole session (legal because the card is the only
device on that SPI bus), and SPI.setHwCs(false) stops the hardware peripheral
from driving a CS line of its own.

## Working init (from Waveshare's own 03_SD_Test demo)
    #include <esp_io_expander.hpp>
    #include <SPI.h>
    #include <SD.h>

    #define SD_MOSI 11
    #define SD_CLK  12
    #define SD_MISO 13
    #define SD_SS   -1        // no GPIO CS
    #define SD_CS    4        // CH422G EXIO4, expander pin number

    esp_expander::CH422G *ex =
        new esp_expander::CH422G(9, 8, ESP_IO_EXPANDER_I2C_CH422G_ADDRESS);
    ex->init();
    ex->begin();
    ex->enableAllIO_Output();          // CH422G IO0-7 share one direction
    ex->digitalWrite(SD_CS, LOW);      // assert CS, leave it low
    SPI.setHwCs(false);
    SPI.begin(SD_CLK, SD_MISO, SD_MOSI, SD_SS);
    SD.begin(SD_SS);

Waveshare's demo also drives LCD_BL (EXIO2) low and USB_SEL (EXIO5) low.
Backlight off saves power on a serial-only build. KEEP USB_SEL LOW — it
selects USB mode over CAN.

Other CH422G pins: EXIO1 TP_RST, EXIO2 LCD_BL, EXIO3 LCD_RST, EXIO5 USB_SEL.

## LIBRARIES — must come from the demo zip, not Library Manager
Download: files.waveshare.com/wiki/ESP32-S3-Touch-LCD-4.3/
          ESP32-S3-Touch-LCD-4.3-Demo.zip
Copy BOTH from Arduino/libraries/ into ~/Documents/Arduino/libraries/:
    ESP32_IO_Expander
    esp-lib-utils          (dependency; the expander will not build without it)

Library Manager's ESP32_IO_Expander 0.1.0 has an OLDER, INCOMPATIBLE API and
BOOT-LOOPS on this board. Symptoms and differences:
    0.1.0:      ESP_IOExpander_CH422G(port, addr, scl, sda), pinMode(4, OUTPUT)
                -> prints "version: 0.1.0" then resets forever
    demo zip:   esp_expander::CH422G(scl, sda, addr), enableAllIO_Output()
Also: passing IO_EXPANDER_PIN_NUM_4 to digitalWrite() is WRONG — that constant
is a BITMASK (0x10), and digitalWrite() wants a pin NUMBER. The symptom is
    "Pin num mask out of range, bit higher than 11 won't work"
followed by a crash.

## NEVER cp -R A LIBRARY OVER AN EXISTING ONE
`cp -R src dest/` MERGES into an existing folder. Copying the demo's
esp-lib-utils on top of a Library-Manager copy left both file layouts in
place, each defining the same symbols, and the link failed with
    multiple definition of `esp_utils_log_extract_file_name'
    multiple definition of `esp_utils_mem_gen_malloc'
Always `rm -rf` the destination first. Fingerprint of a merged library: far
more source files than the source folder has
(19 files installed vs 6 in the demo).

## Verified results
Prebuilt firmware (Demo/Firmware/SD_Test.bin, flashed at 0x0):
    Name: APPSD | Type: SDSC | Speed: 20.00 MHz | Size: 480MB
    sector_size=512, capacity=983040, bus_width=1
    mounted, wrote, renamed, read back, formatted, unmounted — all clean
Source build of Arduino/examples/03_SD_Test:
    449,094 bytes program (34%), 23,900 bytes globals (7%) — compiles clean

Throughput: 20MHz, 1-bit bus is roughly 2MB/s. Plenty for kb.bin, where a
lookup is ~log2(N)+1 seeks of a few hundred bytes each (a 100k-key bank is
about 18 reads).

WARNING: the demo FORMATS the card, and flashing SD_Test.bin at 0x0 replaces
the whole image including the partition table. Both are fine on the spare
board; never do either to the Guition while it carries the model.

================================================================
FLASH BANK VS SD BANK — the decision
================================================================
FLASH (current, working)
  + memory-mapped, effectively instant, no init, no wiring risk
  + 100KB of facts = 792 keys / 1,399 facts, compiled into kb_data.h
  - READ-ONLY. Every change means rebuild -> bin2header.py -> reflash
  - 508KB of source; competes with the model for the 16MB flash
  - practical ceiling: a few MB

SD (solved, unused)
  + 512MB ~= a million 300-byte fact cards; size stops being a constraint
  + WRITABLE — the capability flash structurally lacks
  - Waveshare only; adds ~20-50ms per lookup (irrelevant at ~1 tok/s)

SWITCH WHEN one of these lands, not before:
  1. The Simple Wikipedia fact-card factory (history, economics, business) —
     tens of thousands of cards, megabytes.
  2. REFUSAL LOGGING — refused questions written to the card, reviewed
     weekly, answered with new cards. This is the strongest reason: it is a
     capability, not a size problem, and flash cannot do it at all.
  3. Cold expert swapping, if that idea is ever revisited.

The switch is small by design: build_kb.py already emits the SD-searchable
kb.bin (sorted fixed-width key index + records, binary searched), so it is
`#define KB_ON_SD` in kb.h plus the init above — not a redesign.

CAUTION: esp32_storyteller.ino currently hardcodes placeholder SD pins
    SD_SCK 39, SD_MISO 40, SD_MOSI 41, SD_CS 38
under #ifdef KB_ON_SD. Those are WRONG for both boards and must be replaced
with the verified init above.

================================================================
DEPLOY RECIPES
================================================================
## Flash a model to the Guition (12M) — ALL THREE LINES, IN ORDER
    python3 -c "from tokenizer import Tokenizer; Tokenizer('data/tok1024.model').export()"
    python3 export_model.py out_DIR/model.bin data/tok1024.bin NAME.bin --bits 4 --gs 32 --seq 160
    python3 -m esptool --chip esp32s3 --port /dev/cu.usbmodem21101 write_flash 0x1F0000 NAME.bin
Skipping line 1 pairs new weights with a stale tokenizer; the device then
emits character-level garbage with correct sentence shape. That signature
means tokenizer mismatch, not a bad model.

## Roll back to the previous model (one line)
    python3 -m esptool --chip esp32s3 --port /dev/cu.usbmodem21101 write_flash 0x1F0000 r457_v9c.bin

## Rebuild the flash bank — ALL THREE, IN ORDER
    python3 build_kb.py --out kb.bin
    python3 bin2header.py kb.bin kb_data.h
    cp kb_data.h ~/Documents/Arduino/esp32_storyteller/
Skipping bin2header.py leaves the firmware reading the OLD blob. This also
bites host testing: kb.cpp defaults to KB_IN_FLASH, so tests read kb_data.h
rather than the kb.bin being rebuilt, and new keys appear "not to exist".

## 27M deployment (planned, not done)
    seq MUST be 256 on device, not 512:
      KV cache = 2 * seq * dim * 4 bytes * layers_per_board
               = 2 * 256 * 512 * 4 * 4 = 4.2MB   fits 8MB PSRAM
               = 2 * 512 * 512 * 4 * 4 = 8.4MB   does NOT
    export_model.py ... --bits 4 --gs 32 --seq 256, then split_image.py
    4 layers per board. Firmware constants: dim 512, hidden_dim 1376,
    vocab 4096, 4 layers/board, new tok4096.bin.
    THE BLOCKER: pipeline_head_r457.ino is host-tested 11/11 but has NEVER
    been compiled for Arduino.

================================================================
HARDWARE GOTCHAS ALREADY PAID FOR
================================================================
- Partition Scheme Custom breaks any sketch without partitions.csv; Default
  breaks esp32_storyteller's model mapping. Switch deliberately, both ways.
- A stack overflow: kb_facts_for_question put ~6KB of locals on the 8KB
  Arduino loopTask stack. /kb worked (small path), /ask crashed. Large
  buffers there are now static.
- KB_IN_FLASH was defined in the .ino, but kb.cpp is a SEPARATE COMPILATION
  UNIT and never saw it — it built the SD backend and reported "bad
  kb_data.h". Backend selection now lives in kb.h, which both include.
- `if (line[0] == '/')` swallowed every slash-command and its else printed
  the help line, making /ask, /kb and /kbn DEAD CODE. The symptom (a help
  line) pointed at caching, uploads and line endings for several rounds.
- llm_encode prepends a dummy space piece; tool injections must strip that
  token. Same effect makes a standalone sentencepiece encode of "<calc>"
  return two ids — not a tokenizer failure.
- First Arduino compile after changing a library is slow (minutes). The
  spinning "building sketch" indicator is normal; 15+ minutes is not.
- Mac: downloaded zips auto-extract, so there is no unzip step.
- zsh: pasted '#' comments are NOT comments interactively, and '<' '>' in a
  pasted placeholder are redirects. Paste command blocks comment-free with
  literal paths. Arduino C++ pasted into Terminal gives "parse error near".
- If a Mac cannot see the board's port at all, Waveshare ships a CH34x MAC
  driver; also try holding BOOT while plugging in.
