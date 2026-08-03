/* kb.h — R-457 knowledge bank reader (SD card, zero RAM for the bank).
 *
 * Mirrors the binary search in build_kb.py's query(). The bank stays on the
 * card; only the matched record is read. Cost is about log2(N)+1 reads.
 *
 *   kb.bin layout
 *     magic "R457KB01" | n_keys u32 | index_off u32
 *     records: key_len u8 | key | n_facts u8 | (len u16 | utf8 text)*
 *     index:   n_keys * 32 -> key[26] NUL-padded + rec_off u32 + rec_len u16,
 *              sorted bytewise
 */
#ifndef KB_H
#define KB_H

/* Backend: default is the compiled-in flash bank (no SD card).
 * Define KB_ON_SD before including to use an SD card instead. */
#if !defined(KB_ON_SD) && !defined(KB_IN_FLASH)
#define KB_ON_SD 1
#endif

#include <stdint.h>

#define KB_KEY_BYTES   26
#define KB_IDX_ENTRY   32
#define KB_MAX_FACTS    6
#define KB_TEXT_MAX   256

typedef struct {
  char text[KB_MAX_FACTS][KB_TEXT_MAX];
  int  n;
} KBResult;

/* open/close the bank. path is e.g. "/kb.bin" on SD. */
int  kb_open(const char *path);
void kb_close(void);

/* number of keys in the open bank (0 if not open) */
uint32_t kb_size(void);

/* exact key lookup (key is lowercased/truncated internally).
 * returns number of facts written into out, 0 if not found. */
int  kb_lookup(const char *key, KBResult *out);

/* Retrieval for a question: pulls candidate words out of the question,
 * looks each up, and concatenates unique facts into buf as one
 * space-separated string suitable for splicing after "Facts: ".
 * Returns the number of facts appended. */
int  kb_facts_for_question(const char *question, char *buf, int buf_size,
                           int max_facts);

/* Build a full four-part prompt from a plain question line, splicing in facts
 * retrieved from the bank. Returns how many bank facts were added. */
int  kb_build_prompt(const char *line, char *out, int out_size, int max_facts);

#endif /* KB_H */
