/* kb.cpp — R-457 knowledge bank reader.
 *
 * Compiles two ways:
 *   ARDUINO  -> reads from the SD card via SD.h
 *   host     -> reads from a normal file, so the same code can be tested on a
 *               PC against the exact kb.bin that build_kb.py produced.
 */

#include "kb.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* ---------------- file layer ----------------
 * Three backends. KB_IN_FLASH wins if defined: the bank is compiled into the
 * sketch as a byte array, so no SD card (and no SD pins) are needed at all.
 * That works up to a few hundred KB; big corpora still want the card.
 */
#ifdef KB_IN_FLASH
  #include <stdio.h>
  static int g_open = 0;
  #include "kb_data.h"
  static uint32_t g_pos = 0;
  static int  kbf_open(const char *)   { g_pos = 0; return 1; }
  static void kbf_close(void)          { g_pos = 0; }
  static int  kbf_seek(uint32_t off)   { if (off >= KB_BLOB_LEN) return 0;
                                         g_pos = off; return 1; }
  static int  kbf_read(void *b, int n) {
    if (g_pos + n > KB_BLOB_LEN) n = (int)(KB_BLOB_LEN - g_pos);
    if (n <= 0) return 0;
    memcpy_P(b, KB_BLOB + g_pos, n);
    g_pos += n;
    return n;
  }
#elif defined(ARDUINO)
  #include <SD.h>
  static File g_f;
  static int  g_open = 0;
  static int  kbf_open(const char *p) { g_f = SD.open(p, FILE_READ);
                                        g_open = g_f ? 1 : 0; return g_open; }
  static void kbf_close(void)          { if (g_open) { g_f.close(); g_open = 0; } }
  static int  kbf_seek(uint32_t off)   { return g_f.seek(off) ? 1 : 0; }
  static int  kbf_read(void *b, int n) { return g_f.read((uint8_t *)b, n); }
#else
  #include <stdio.h>
  static FILE *g_f = 0;
  static int   g_open = 0;
  static int  kbf_open(const char *p) { g_f = fopen(p, "rb");
                                        g_open = g_f ? 1 : 0; return g_open; }
  static void kbf_close(void)          { if (g_f) { fclose(g_f); g_f = 0; }
                                         g_open = 0; }
  static int  kbf_seek(uint32_t off)   { return fseek(g_f, off, SEEK_SET) == 0; }
  static int  kbf_read(void *b, int n) { return (int)fread(b, 1, n, g_f); }
#endif

/* ---------------- header state ---------------- */
static uint32_t g_nkeys = 0, g_idx_off = 0;

static uint32_t rd_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int kb_open(const char *path) {
#ifdef KB_IN_FLASH
  g_open = 1;
#endif
  if (!kbf_open(path)) return 0;
  uint8_t hdr[16];
  if (kbf_read(hdr, 16) != 16 || memcmp(hdr, "R457KB01", 8) != 0) {
    kbf_close();
    return 0;
  }
  g_nkeys   = rd_u32(hdr + 8);
  g_idx_off = rd_u32(hdr + 12);
  return 1;
}

void kb_close(void) { kbf_close(); g_nkeys = g_idx_off = 0; }

uint32_t kb_size(void) { return g_open ? g_nkeys : 0; }

/* normalise a query key exactly like the builder did */
static void norm_key(const char *in, char *out) {
  int n = 0;
  for (const char *c = in; *c && n < KB_KEY_BYTES; c++) {
    unsigned char ch = (unsigned char)*c;
    out[n++] = (char)tolower(ch);
  }
  out[n] = '\0';
  /* trim trailing spaces */
  while (n > 0 && out[n - 1] == ' ') out[--n] = '\0';
}

int kb_lookup(const char *key, KBResult *out) {
  out->n = 0;
  if (!g_open || g_nkeys == 0) return 0;

  char want[KB_KEY_BYTES + 1];
  norm_key(key, want);
  int wl = (int)strlen(want);
  if (wl == 0) return 0;

  long lo = 0, hi = (long)g_nkeys - 1;
  uint8_t entry[KB_IDX_ENTRY];
  while (lo <= hi) {
    long mid = (lo + hi) / 2;
    if (!kbf_seek(g_idx_off + (uint32_t)mid * KB_IDX_ENTRY)) return 0;
    if (kbf_read(entry, KB_IDX_ENTRY) != KB_IDX_ENTRY) return 0;

    char k[KB_KEY_BYTES + 1];
    memcpy(k, entry, KB_KEY_BYTES);
    k[KB_KEY_BYTES] = '\0';
    int kl = (int)strnlen(k, KB_KEY_BYTES);

    /* bytewise compare, same ordering the builder sorted with */
    int lim = kl < wl ? kl : wl;
    int cmp = memcmp(k, want, lim);
    if (cmp == 0) cmp = (kl == wl) ? 0 : (kl < wl ? -1 : 1);

    if (cmp == 0) {
      uint32_t rec_off = rd_u32(entry + KB_KEY_BYTES);
      uint16_t rec_len = rd_u16(entry + KB_KEY_BYTES + 4);
      if (rec_len > 2048) rec_len = 2048;
      static uint8_t rec[2048];
      if (!kbf_seek(rec_off)) return 0;
      int got = kbf_read(rec, rec_len);
      if (got < 2) return 0;

      int p = 1 + rec[0];                 /* skip key_len + key */
      if (p >= got) return 0;
      int nf = rec[p++];
      if (nf > KB_MAX_FACTS) nf = KB_MAX_FACTS;
      for (int i = 0; i < nf && p + 2 <= got; i++) {
        int ln = rd_u16(rec + p); p += 2;
        if (ln >= KB_TEXT_MAX) ln = KB_TEXT_MAX - 1;
        if (p + ln > got) break;
        memcpy(out->text[out->n], rec + p, ln);
        out->text[out->n][ln] = '\0';
        out->n++;
        p += ln;
      }
      return out->n;
    }
    if (cmp < 0) lo = mid + 1; else hi = mid - 1;
  }
  return 0;
}

/* ---------------- question -> facts ---------------- */

static int is_stop(const char *w) {
  static const char *S[] = {
    "the","a","an","is","are","was","were","of","in","on","to","and","or",
    "what","which","who","how","many","much","does","do","did","it","its",
    "this","that","for","with","if","then","so","be","by","at","from","not",
    "you","your","i","we","they","he","she","can","will","would","there",
    "question","answer","facts","reasoning","value","values",
    "marked","labelled","labeled","called","rated","need","needs",
    "about","one","get","give","gives","find","using","use", 0
  };
  for (int i = 0; S[i]; i++) if (strcmp(w, S[i]) == 0) return 1;
  return 0;
}


/* ---------------- synonym expansion ----------------
 * Everyday phrasings mapped onto the words the bank is actually indexed by
 * (see kb.bin key vocabulary: atomic/mass/number/melting/point/density/...).
 * Added as EXTRA content words, never replacing the originals, so exact
 * keyword retrieval keeps working exactly as before. */
struct Syn { const char *from; const char *to; };
static const Syn SYN[] = {
  /* density */
  {"heavy","density"}, {"heavier","density"}, {"heaviest","density"},
  {"weigh","density"}, {"weighs","density"}, {"weight","density"},
  {"dense","density"}, {"denser","density"},
  /* melting point */
  {"melt","melting"},  {"melts","melting"},  {"melted","melting"},
  {"liquid","melting"},{"liquefy","melting"},{"freeze","melting"},
  {"freezing","melting"},
  /* boiling point */
  {"boil","boiling"},  {"boils","boiling"},  {"vapor","boiling"},
  {"vapour","boiling"},{"evaporate","boiling"},
  /* atomic mass / number */
  {"weighs","atomic"}, {"atom","atomic"},    {"atoms","atomic"},
  {"protons","number"},{"proton","number"},
  /* thermal / electrical */
  {"conducts","conductivity"}, {"conduct","conductivity"},
  {"conducting","conductivity"},
  {"resist","resistivity"},    {"resists","resistivity"},
  {"insulator","resistivity"},
  {"hot","heat"},      {"hotter","heat"},    {"thermal","heat"},
  {"temperature","point"},
  /* misc */
  {"fast","speed"},    {"quick","speed"},    {"velocity","speed"},
  {"expands","expansion"}, {"expand","expansion"},
  {"stretch","tensile"},   {"strong","tensile"}, {"strength","tensile"},
};
static const int N_SYN = (int)(sizeof(SYN) / sizeof(SYN[0]));

int kb_facts_for_question(const char *question, char *buf, int buf_size,
                          int max_facts) {
  buf[0] = '\0';
  if (!g_open) return 0;

  /* Static, not stack: the Arduino loopTask has an 8KB stack and this chain
   * once overflowed it. Single threaded, so one shared copy is fine. */
  struct Cand { char key[68]; int nfacts; int len; int phrase; };
  static Cand cand[16];
  static char words[8][64];
  static KBResult r;
  int ncand = 0, nw = 0;

  /* 1. pull out the content words */
  char word[64]; int wl = 0;
  for (const char *c = question; ; c++) {
    unsigned char ch = (unsigned char)*c;
    if (isalnum(ch)) { if (wl < 63) word[wl++] = (char)tolower(ch); continue; }
    word[wl] = '\0';
    int numeric = wl > 0;
    for (int i = 0; i < wl; i++)
      if (!isdigit((unsigned char)word[i])) numeric = 0;
    if (wl > 0 && (wl >= 3 || numeric) && !is_stop(word) && nw < 8)
      snprintf(words[nw++], 64, "%s", word);
    wl = 0;
    if (!*c) break;
  }

  /* 1b. synonym expansion: add bank-vocabulary equivalents of everyday words.
   *     Extra words only — the originals stay, so exact matching is unchanged. */
  {
    int base = nw;
    for (int i = 0; i < base && nw < 8; i++) {
      for (int k = 0; k < N_SYN && nw < 8; k++) {
        if (strcmp(words[i], SYN[k].from) != 0) continue;
        int dup = 0;
        for (int j = 0; j < nw; j++)
          if (strcmp(words[j], SYN[k].to) == 0) { dup = 1; break; }
        if (!dup) snprintf(words[nw++], 64, "%s", SYN[k].to);
      }
    }
  }

  /* 2. every ordered PAIR of content words, then every single word.
   *    Pairs are tried in both orders and across gaps, because a question says
   *    "the melting point of aluminium" while the bank stores
   *    "aluminium melting point". */
  for (int stage = 0; stage < 2 && ncand < 16; stage++) {
    for (int i = 0; i < nw && ncand < 16; i++) {
      for (int j = 0; j < nw && ncand < 16; j++) {
        if (stage == 0 && i == j) continue;
        if (stage == 1 && j != 0) break;          /* singles: one pass only */
        char q[132];
        int numeric_single = 1;
        if (stage == 0) snprintf(q, sizeof q, "%s %s", words[i], words[j]);
        else {
          snprintf(q, sizeof q, "%s", words[i]);
          for (const char *p = q; *p; p++)
            if (!isdigit((unsigned char)*p)) { numeric_single = 0; break; }
          if (numeric_single) continue;            /* a bare number is noise */
        }
        int n = kb_lookup(q, &r);
        if (n <= 0) continue;
        int dup = 0;
        for (int k = 0; k < ncand; k++)
          if (strcmp(cand[k].key, q) == 0) { dup = 1; break; }
        if (dup) continue;
        snprintf(cand[ncand].key, sizeof cand[ncand].key, "%s", q);
        cand[ncand].nfacts = n;
        cand[ncand].len    = (int)strlen(q);
        cand[ncand].phrase = (stage == 0) ? 1 : 0;
        ncand++;
      }
    }
  }

  /* 3. rank: phrase matches first (more of the question matched), then the
   *    most specific key (fewest facts), then the longest key. */
  for (int i = 1; i < ncand; i++) {
    Cand t = cand[i]; int j = i - 1;
    while (j >= 0 && (cand[j].phrase < t.phrase ||
                     (cand[j].phrase == t.phrase &&
                      (cand[j].nfacts > t.nfacts ||
                      (cand[j].nfacts == t.nfacts && cand[j].len < t.len))))) {
      cand[j + 1] = cand[j]; j--;
    }
    cand[j + 1] = t;
  }

  int written = 0, used = 0;
  for (int i = 0; i < ncand && written < max_facts; i++) {
    int n = kb_lookup(cand[i].key, &r);
    for (int k = 0; k < n && written < max_facts; k++) {
      int need = (int)strlen(r.text[k]) + 1;
      if (used + need >= buf_size) break;
      if (used && strstr(buf, r.text[k])) continue;
      if (used) buf[used++] = ' ';
      memcpy(buf + used, r.text[k], need - 1);
      used += need - 1;
      buf[used] = '\0';
      written++;
    }
  }
  return written;
}

/* ---------------- prompt assembly ----------------
 * Turn a plain user line into R-457's four-part prompt, with facts retrieved
 * from the bank spliced in front of any facts the user supplied.
 *
 *   input : "The current is 7 amps. The resistance is 9 ohms. What is the
 *            voltage in volts?"
 *   output: "Facts: Voltage equals current times resistance. The current is 7
 *            amps. The resistance is 9 ohms.\nQuestion: What is the voltage in
 *            volts?\nReasoning:"
 *
 * Split rule: the text after the LAST '.' is the question; everything before
 * it is the user's own facts. With no '.', the whole line is the question and
 * the facts come only from the bank.
 */
int kb_build_prompt(const char *line, char *out, int out_size, int max_facts) {
  const char *split = strrchr(line, '.');
  static char facts[512]; facts[0] = '\0';
  const char *question;

  if (split && *(split + 1)) {
    int fl = (int)(split - line) + 1;
    if (fl > (int)sizeof(facts) - 1) fl = (int)sizeof(facts) - 1;
    memcpy(facts, line, fl);
    facts[fl] = '\0';
    question = split + 1;
    while (*question == ' ') question++;
  } else {
    question = line;
  }

  static char retrieved[512];
  int n = kb_facts_for_question(question, retrieved, sizeof retrieved,
                                max_facts);

  snprintf(out, out_size, "<reason>\nFacts:%s%s%s%s\nQuestion: %s\nReasoning:",
           n ? " " : "", n ? retrieved : "",
           facts[0] ? " " : "", facts[0] ? facts : "",
           question);
  return n;
}
