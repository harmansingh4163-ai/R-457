/*
 * pipeline_head.ino — board B: the user-facing board.
 * Holds embedding + last layers + classifier + tokenizer (head.bin).
 * Per token: embed -> ship activation to worker over UART -> worker runs
 * its layers -> finish locally -> sample. With retry on link errors.
 *
 * Sketch folder: this file + llm_core.c/.h + pipeline_link.h + partitions.csv
 */
#include <Arduino.h>
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "llm_core.h"
#include "pipeline_link.h"

// ===================== R-457 tools + knowledge bank =====================
#define KB_ENABLED 1
#if KB_ENABLED
#include <SPI.h>
#include <SD.h>
#include "kb.h"
#include <esp_io_expander.hpp>
#include <SPI.h>
#include <SD.h>

/* SD pins are hardcoded in kb_init(): SPI.begin(12, 13, 11, -1) with CS via
   the CH422G expander (EXIO4). Former placeholder #defines deleted — they
   were dead code and the exact trap R-457_HARDWARE.md warns about. */
#define KB_PATH  "/kb.bin"
static int g_kb_ready  = 0;
static int g_kb_nfacts = 2;

static void kb_init() {
  static esp_expander::CH422G *ex =
      new esp_expander::CH422G(9, 8, ESP_IO_EXPANDER_I2C_CH422G_ADDRESS);
  ex->init(); ex->begin();
  ex->enableAllIO_Output();
  ex->digitalWrite(4, LOW);   /* SD_CS asserted for the session */
  ex->digitalWrite(5, LOW);   /* USB_SEL */
  ex->digitalWrite(2, LOW);   /* backlight off (serial build) */
  SPI.setHwCs(false);
  SPI.begin(12, 13, 11, -1);
  if (!SD.begin(-1)) { Serial.println("kb: SD.begin failed"); return; }
  Serial.printf("kb: SD ok (%llu MB)\n",
                (unsigned long long)(SD.cardSize() / (1024ULL*1024ULL)));
  if (!SD.exists(KB_PATH)) { Serial.println("kb: /kb.bin not on card"); return; }
  if (!kb_open(KB_PATH))   { Serial.println("kb: bad bank file"); return; }
  g_kb_ready = 1;
  Serial.printf("kb: ready, %u keys (SD)\n", (unsigned)kb_size());
}

#endif

// The model emits "<calc>13-9=" or "<count>strawberry,r=" and stops; this chip
// computes the answer exactly and feeds it back as forced tokens, so the KV
// cache stays consistent and generation continues.
static int g_space_tok = -1;

static LLM g_m = {};

static void learn_space_token() {
  int t[8];
  int n = llm_encode(&g_m, "9", 0, 0, t, 8);
  g_space_tok = (n == 2) ? t[0] : -1;
  Serial.printf("tools ready: <calc> <count>  (space tok %d)\n", g_space_tok);
}

static const char* g_prompt = NULL;   /* facts for <quote> */

static const char* pending_tool(const char* buf) {
  static char out[40];
  int len = strlen(buf);
  if (len < 6 || buf[len - 1] != '=') return NULL;
  const char* open = NULL; const char* tag = NULL;
  for (const char* p = buf; (p = strstr(p, "<calc>"))  != NULL; p++) { open = p; tag = "calc"; }
  for (const char* p = buf; (p = strstr(p, "<count>")) != NULL; p++)
    if (!open || p > open) { open = p; tag = "count"; }
  for (const char* p = buf; (p = strstr(p, "<quote>")) != NULL; p++)
    if (!open || p > open) { open = p; tag = "quote"; }
  if (!open) return NULL;
  if (strstr(open, "</calc>") || strstr(open, "</count>") ||
      strstr(open, "</quote>")) return NULL;

  if (tag[0] == 'q') {                                  /* <quote> */
    /* model emits "<quote>subject,property=" — copy the value VERBATIM out
     * of the facts, so decimals cannot be truncated (ft6 turned 103.296
     * into 103). */
    if (!g_prompt) return NULL;
    char subj[48], prop[48];
    int i = 0; const char* q = open + 7;
    while (*q && *q != ',' && i < 47) subj[i++] = *q++;
    subj[i] = '\0';
    if (*q != ',') return NULL;
    q++; i = 0;
    while (*q && *q != '=' && i < 47) prop[i++] = *q++;
    prop[i] = '\0';
    if (!subj[0] || !prop[0]) return NULL;

    /* find a sentence naming BOTH, then take the first number after the
     * property mention */
    const char* best = NULL;
    const char* sent = g_prompt;
    while (*sent) {
      const char* end = strchr(sent, '.');
      if (!end) end = sent + strlen(sent);
      int len = (int)(end - sent);
      if (len > 0 && len < 400) {
        static char tmp[400];
        memcpy(tmp, sent, len); tmp[len] = '\0';
        const char* ps = strstr(tmp, prop);
        if (ps && strstr(tmp, subj)) {
          for (const char* z = ps; *z; z++)
            if ((*z >= '0' && *z <= '9') ||
                (*z == '-' && z[1] >= '0' && z[1] <= '9')) {
              best = sent + (z - tmp); break;
            }
          if (best) break;
        }
      }
      if (!*end) break;
      sent = end + 1;
    }
    if (!best) return NULL;
    i = 0;
    while (best[i] && i < 24 &&
           ((best[i] >= '0' && best[i] <= '9') || best[i] == '.' ||
            best[i] == '-' || best[i] == '+' || best[i] == 'e'))
      i++;
    while (i > 0 && (best[i-1] == '.' || best[i-1] == 'e')) i--;  /* trailing */
    if (i == 0) return NULL;
    snprintf(out, sizeof out, "%.*s</quote>", i, best);
    return out;
  }

  if (tag[1] == 'a') {                                  // <calc>
    long a = 0, b = 0; char op = 0;
    if (sscanf(open + 6, "%ld%c%ld", &a, &op, &b) != 3) return NULL;
    long r;
    switch (op) {
      case '+': r = a + b; break;
      case '-': r = a - b; break;
      case '*': r = a * b; break;
      case '/': if (b == 0) return NULL; r = a / b; break;
      default: return NULL;
    }
    snprintf(out, sizeof out, "%ld</calc>", r);
    return out;
  }
  char word[48]; char ch = 0; int wl = 0;               // <count>
  const char* p = open + 7;
  while (*p && *p != ',' && *p != '=' && wl < (int)sizeof(word) - 1) word[wl++] = *p++;
  word[wl] = '\0';
  if (wl == 0) return NULL;
  if (*p == ',') { ch = *(p + 1); if (ch == '=' || ch == '\0') return NULL; }
  int n = 0;
  if (ch) { for (int i = 0; i < wl; i++) if (word[i] == ch) n++; }
  else      n = wl;
  snprintf(out, sizeof out, "%d</count>", n);
  return out;
}
// ========================================================================

static Sampler g_sampler = { 0.8f, 0.9f, 0 };
static uint8_t g_buf[4096];
static float g_temp = 0.8f, g_topp = 0.9f;
static int g_maxlen = 180;
static bool g_ready = false;

static void io_write(const uint8_t* d, size_t n) { Serial1.write(d, n); }
static int io_read(uint32_t timeout_ms) {
  uint32_t t0 = millis();
  while (!Serial1.available()) {
    if (millis() - t0 >= timeout_ms) return -1;
    delayMicroseconds(50);
  }
  return Serial1.read();
}
static const LinkIO IO = { io_write, io_read };

static void* alloc_big(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  return p ? p : heap_caps_malloc(n, MALLOC_CAP_8BIT);
}

static TaskHandle_t s_worker = nullptr, s_main = nullptr;
static volatile const MMJob* s_job = nullptr;
static void mm_task(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    const MMJob* j = (const MMJob*)s_job;
    llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx,
                    j->n, j->d / 2, j->d);
    xTaskNotifyGive(s_main);
  }
}
static void parallel_matmul(const MMJob* j) {
  if (j->d < 64) {
    llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, j->d);
    return;
  }
  s_job = j; xTaskNotifyGive(s_worker);
  llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, j->d / 2);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

/* one remote round-trip with one retry; returns true on success */
static bool remote_layers(int pos) {
  for (int attempt = 0; attempt < 2; attempt++) {
    uint16_t plen = 2 + g_m.h.dim * 4;
    g_buf[0] = pos & 0xFF; g_buf[1] = pos >> 8;
    memcpy(g_buf + 2, g_m.x, g_m.h.dim * 4);
    link_send(&IO, CMD_FWD, g_buf, plen);
    uint16_t got;
    int cmd = link_recv(&IO, g_buf, sizeof(g_buf), &got, 8000);
    if (cmd == CMD_RSP && got == g_m.h.dim * 4) {
      memcpy(g_m.x, g_buf, g_m.h.dim * 4);
      return true;
    }
    Serial.printf("\n[link %s, retry]", cmd == -1 ? "timeout" : "crc error");
  }
  return false;
}

static char  gbuf[1024];
static float g_lg[4096];
static float g_conf_sum = 0, g_conf_min = 1;
static int   g_conf_n = 0, g_answer_seen = 0;

static void generate(const char* prompt) {
  g_prompt = prompt;
  static int tokens[512];
  int n = llm_encode(&g_m, prompt, 1, 0, tokens, 512);
  if (n < 1 || n >= g_m.h.seq_len) { Serial.println("(bad prompt length)"); return; }
  g_sampler.temperature = g_temp; g_sampler.topp = g_topp;
  Serial.print("\n> "); Serial.print(prompt);
  uint32_t t0 = millis(); int gen = 0, calls = 0;
  int tok = tokens[0];
  int total = min(n + g_maxlen, (int)g_m.h.seq_len);

  int glen = 0; gbuf[0] = '\0'; g_answer_seen = 0;
  g_conf_sum = 0; g_conf_min = 1; g_conf_n = 0;
  static int inject[32]; int inj_n = 0, inj_i = 0;

  for (int pos = 0; pos < total; pos++) {
    llm_embed(&g_m, tok);
    if (!remote_layers(pos)) {
      Serial.println("\nERROR: worker board not responding. Check wiring/power.");
      return;
    }
    llm_layers(&g_m, pos);
    llm_head(&g_m);

    int next;
    if (pos < n - 1)            next = tokens[pos + 1];       // prefill
    else if (inj_i < inj_n)     next = inject[inj_i++];       // tool result
    else {
      int V = g_m.h.vocab_size;
      if (V <= 4096) memcpy(g_lg, g_m.logits, sizeof(float) * V);
      next = llm_sample(&g_m, &g_sampler); gen++;
      if (V <= 4096) {
        float mx = -1e30f; for (int vi = 0; vi < V; vi++) if (g_lg[vi] > mx) mx = g_lg[vi];
        double se = 0; for (int vi = 0; vi < V; vi++) se += expf(g_lg[vi] - mx);
        float pc = (float)(expf(g_lg[next] - mx) / se);
        g_conf_sum += pc; if (pc < g_conf_min) g_conf_min = pc; g_conf_n++;
      }
    }
    if (next == 1 || next == 2) break;

    if (pos >= n - 1) {
      const char* piece = llm_decode(&g_m, tok, next);
      Serial.print(piece);
      for (const char* c = piece; *c && glen < (int)sizeof(gbuf) - 1; c++)
        gbuf[glen++] = *c;
      gbuf[glen] = '\0';

      if (g_answer_seen && strchr(piece, '\n')) break;   /* EOS discipline */
      if (strstr(gbuf, "Answer:")) g_answer_seen = 1;

      if (inj_i >= inj_n) {
        const char* res = pending_tool(gbuf);
        if (res) {
          int m = llm_encode(&g_m, res, 0, 0, inject, 32);
          int start = (m > 0 && inject[0] == g_space_tok) ? 1 : 0;
          if (m > start) {
            for (int i = start; i < m; i++) inject[i - start] = inject[i];
            inj_n = m - start; inj_i = 0; calls++;
          }
        }
      }
    }
    tok = next;
  }
  float cavg = g_conf_n ? g_conf_sum / g_conf_n : 0;
  Serial.printf("\n\n[%d tokens in %.1fs — %.2f tok/s, %d tool call(s), conf avg %.2f min %.2f]\n",
                gen, (millis()-t0)/1000.0f, gen * 1000.0f / (millis()-t0), calls, cavg, g_conf_min);
  if (g_conf_n && cavg < 0.60f) Serial.println("(unsure — low confidence answer)");
}

static void run_selftest() {
  if (!g_kb_ready) { Serial.println("selftest: kb not ready"); return; }
  Serial.println("SELFTEST (2 canaries)");
  static char sp_[900];
  kb_build_prompt("What is the atomic mass of silicon?", sp_, sizeof sp_, 2);
  generate(sp_);
  int p1 = strstr(gbuf, "28.085") != NULL;
  kb_build_prompt("What is the melting point of aluminium?", sp_, sizeof sp_, 2);
  generate(sp_);
  int p2 = strstr(gbuf, "660") != NULL;
  Serial.printf("SELFTEST: %s (silicon %s, aluminium %s)\n",
                (p1 && p2) ? "PASS" : "FAIL", p1 ? "ok" : "FAIL", p2 ? "ok" : "FAIL");
}

static void learn_fact(const char* fact) {
  /* D-5: dedup — a doubled /learn command must not store the fact twice */
  File r = SD.open("/learned.txt");
  if (r) {
    static char line[160];
    int n;
    while ((n = r.readBytesUntil('\n', line, sizeof line - 1)) > 0) {
      line[n] = '\0';
      while (n > 0 && (line[n-1] == '\r' || line[n-1] == ' ')) line[--n] = '\0';
      if (strcmp(line, fact) == 0) {
        r.close();
        Serial.println("learn: already stored, skipped");
        return;
      }
    }
    r.close();
  }
  File f = SD.open("/learned.txt", FILE_APPEND);
  if (!f) { Serial.println("learn: SD write failed"); return; }
  f.println(fact); f.close();
  Serial.println("learned -> /learned.txt");
}

static void log_refusal(const char* q) {
  if (!strstr(gbuf, "annot be determined")) return;
  if (SD.exists("/refused.txt")) {
    File r = SD.open("/refused.txt");
    if (r) {
      while (r.available()) {
        String ln = r.readStringUntil('\n'); ln.trim();
        if (ln.equals(q)) { r.close(); return; }
      }
      r.close();
    }
  }
  File f = SD.open("/refused.txt", FILE_APPEND);
  if (!f) return;
  f.println(q); f.close();
  Serial.println("(refusal logged -> /refused.txt)");
}

static int learned_facts(const char* q, char* out, int cap) {
  out[0] = 0;
  if (!SD.exists("/learned.txt")) return 0;
  File f = SD.open("/learned.txt");
  if (!f) return 0;
  char ql[200]; int qi = 0;
  for (const char* cc = q; *cc && qi < 199; cc++) ql[qi++] = tolower(*cc);
  ql[qi] = 0;
  int nf = 0, olen = 0;
  while (f.available() && nf < 2) {
    String ln = f.readStringUntil('\n'); ln.trim();
    if (!ln.length()) continue;
    String low = ln; low.toLowerCase();
    char tmp[200]; strncpy(tmp, ql, 199); tmp[199] = 0;
    int hit = 0;
    for (char* w = strtok(tmp, " ?.,"); w; w = strtok(NULL, " ?.,"))
      if (strlen(w) >= 4 && low.indexOf(w) >= 0) { hit = 1; break; }
    if (hit) { olen += snprintf(out + olen, cap - olen, " %s", ln.c_str()); nf++; }
  }
  f.close(); return nf;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nPipeline HEAD board — two-chip LLM");
  if (!psramFound()) { Serial.println("ERROR: enable PSRAM"); return; }

  s_main = xTaskGetCurrentTaskHandle();
  xTaskCreatePinnedToCore(mm_task, "mm", 4096, nullptr,
                          configMAX_PRIORITIES - 2, &s_worker, 0);
  llm_alloc_big = alloc_big;
  llm_parallel_matmul = parallel_matmul;

  const esp_partition_t* part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  const void* base = nullptr; esp_partition_mmap_handle_t h;
  if (!part || esp_partition_mmap(part, 0, part->size,
        ESP_PARTITION_MMAP_DATA, &base, &h) != ESP_OK) {
    Serial.println("ERROR: model partition mmap failed"); return;
  }
  int rc = llm_init(&g_m, (const uint8_t*)base, part->size);
  if (rc) { Serial.printf("ERROR: llm_init %d — flash head.bin here\n", rc); return; }
  if (!(g_m.h.flags & LLM_HAS_EMB) || !(g_m.h.flags & LLM_HAS_HEAD)) {
    Serial.println("ERROR: this image isn't head.bin (missing emb/head)"); return;
  }
  if (llm_tok_init(&g_m)) { Serial.println("tokenizer failed"); return; }
  g_sampler.rng = esp_random() | 1ULL;
  learn_space_token();
  { int bt[8]; int bk = llm_encode(&g_m, "<reason>", 0, 0, bt, 8);
    int nt[8]; int nk = llm_encode(&g_m, "\n", 0, 0, nt, 8);
    const char* nd = (nk >= 2) ? llm_decode(&g_m, nt[0], nt[1]) : "";
    Serial.printf("boot check: mode-token %s, newline %s\n",
                  (bk == 2) ? "ok" : "SHATTERED",
                  (nd && nd[0] == '\n') ? "ok" : "WRONG-ID"); }
#if KB_ENABLED
  kb_init();
#endif
  Serial1.begin(LINK_BAUD, SERIAL_8N1, LINK_RX_PIN, LINK_TX_PIN); /* after CH422G */
  g_ready = true;
  Serial.printf("Ready: emb + %d local layers of %d total. "
                "Type a story opening.\n", g_m.h.local_layers, g_m.h.n_layers);
}

void loop() {
  if (!g_ready) { delay(1000); return; }
  static char line[256]; static int pos = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (!pos) continue;
      line[pos] = '\0'; pos = 0;
      float fv; int iv;
      if (sscanf(line, "/temp %f", &fv) == 1) g_temp = fv;
      else if (sscanf(line, "/topp %f", &fv) == 1) g_topp = fv;
      else if (!strcmp(line, "/selftest")) run_selftest();
      else if (!strncmp(line, "/learn ", 7)) learn_fact(line + 7);
      else if (!strcmp(line, "/refused")) {
        if (!SD.exists("/refused.txt")) Serial.println("(no refusals logged)");
        else {
          File r = SD.open("/refused.txt"); int rc = 0;
          while (r && r.available()) {
            String ln = r.readStringUntil('\n'); ln.trim();
            if (ln.length()) Serial.printf("  %d. %s\n", ++rc, ln.c_str());
          }
          if (r) r.close();
          Serial.printf("(%d refused)\n", rc);
        }
      }
      else if (sscanf(line, "/len %d", &iv) == 1) g_maxlen = iv;
      else if (!strncmp(line, "/ask ", 5)) {
#if KB_ENABLED
        if (!g_kb_ready) Serial.println("(knowledge bank not available)");
        else {
          static char prompt[900];
          int nf = kb_build_prompt(line + 5, prompt, sizeof prompt, g_kb_nfacts);
          static char lbuf[300];
          if (learned_facts(line + 5, lbuf, sizeof lbuf) > 0) {
            char* qp = strstr(prompt, "\nQuestion:");
            if (qp && strlen(prompt) + strlen(lbuf) < 890) {
              memmove(qp + strlen(lbuf), qp, strlen(qp) + 1);
              memcpy(qp, lbuf, strlen(lbuf));
              nf += 1;
            }
          }
          Serial.printf("[retrieved %d fact(s) from the bank]\n", nf);
          generate(prompt);
          log_refusal(line + 5);
        }
#endif
      }
      else if (!strncmp(line, "/kb ", 4)) {
#if KB_ENABLED
        KBResult r; int nf = g_kb_ready ? kb_lookup(line + 4, &r) : 0;
        Serial.printf("%d fact(s)\n", nf);
        for (int i = 0; i < nf; i++) Serial.printf("  %s\n", r.text[i]);
#endif
      }
      else if (!strncmp(line, "/kbn ", 5)) {
#if KB_ENABLED
        g_kb_nfacts = atoi(line + 5);
        if (g_kb_nfacts < 0) g_kb_nfacts = 0;
        if (g_kb_nfacts > 3) g_kb_nfacts = 3;
        Serial.printf("bank facts per question = %d\n", g_kb_nfacts);
#endif
      }
      else {
        // '|' becomes a newline so the four-part prompt fits on one line
        for (int i = 0; line[i]; i++) if (line[i] == '|') line[i] = '\n';
        generate(line);
      }
    } else if (pos < 255) line[pos++] = c;
  }
}
