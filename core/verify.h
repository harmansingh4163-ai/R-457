/*
 * verify.h — R-457 on-device answer verification
 * ----------------------------------------------
 * Ports the PC verifier discipline (check_lookup / check_count grounding
 * rules) to the chip. After generation, the device re-checks its own
 * output against the prompt it was given:
 *
 *   1. GROUNDING  every number in the generated text must appear
 *                 verbatim in the prompt (facts) or inside a tool tag
 *                 whose result the chip itself computed.
 *   2. TOOL RECHECK  every closed <calc>a?b=r</calc> is re-executed and
 *                 compared; every <count>...=n</count> is re-counted.
 *                 (Injection is supposed to make these correct by
 *                 construction — this is the regression canary that
 *                 catches encoder/injection breakage like the old
 *                 "114/calc6=" loop.)
 *   3. REFUSAL SHAPE  if the output is a refusal, it must contain no
 *                 asserted numbers outside tool tags.
 *
 * Citations: for each grounded number the verifier reports WHICH fact
 * line it came from (1-based index of the "." -delimited fact in the
 * prompt), e.g.  "660 <- fact 2".
 *
 * Pure C, separate compilation unit (same rule as kb.cpp: config in the
 * header, not the .ino). Static buffers only — the 8KB loopTask stack
 * lesson applies.
 *
 * KNOWN LIMITS (honest):
 *   - Number match is exact-string after trimming a trailing '.', so
 *     "660" does not match "660.0". The bank stores one canonical form,
 *     so this is fine for /ask; it is NOT a general-text checker.
 *   - Refusal detection is phrase-based ("cannot be determined",
 *     "but not the"). New refusal templates must be added here.
 *   - It verifies GROUNDING, not truth. A grounded wrong pick (right
 *     number attached to wrong property) still passes. That is what
 *     the model is for.
 */
#ifndef R457_VERIFY_H
#define R457_VERIFY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  VERIFY_PASS      = 0,   /* all checks green                       */
  VERIFY_REFUSAL   = 1,   /* output is a well-formed refusal        */
  VERIFY_UNGROUNDED= 2,   /* a number has no source in the prompt   */
  VERIFY_TOOL_FAIL = 3,   /* a <calc>/<count> result is wrong       */
  VERIFY_EMPTY     = 4    /* nothing checkable was generated        */
} verify_rc;

/* Check `gen` (generated text) against `prompt` (facts + question).
 * Writes a human-readable one-line report (citations or the first
 * failure) into `report`. Returns a verify_rc. */
verify_rc verify_answer(const char* prompt, const char* gen,
                        char* report, int repsz);

/* Planted-error self-test — project rule: every verifier check must be
 * proven against a planted error. Prints one line per case, returns 0
 * only if every planted error is caught and every clean case passes. */
int verify_selftest(void);

#ifdef __cplusplus
}
#endif
#endif /* R457_VERIFY_H */
