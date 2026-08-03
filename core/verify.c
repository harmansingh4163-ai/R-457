/* verify.c — see verify.h. Pure C99, no Arduino includes. */
#include "verify.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#define MAX_NUM_LEN 24

/* ---------- small helpers ---------- */

static int is_numch(char c) { return (c >= '0' && c <= '9') || c == '.'; }

/* Copy the maximal number starting at s into out (trim trailing '.').
 * Returns its length in the source. */
static int grab_number(const char* s, char* out, int outsz) {
  int i = 0;
  while (is_numch(s[i]) && i < outsz - 1) { out[i] = s[i]; i++; }
  int src = i;
  while (i > 0 && out[i - 1] == '.') i--;   /* "660." -> "660" */
  out[i] = '\0';
  return src;
}

/* Does `num` appear in `hay` as a standalone number (not a substring of
 * a longer number — "14" must not match inside "28.145")? Returns the
 * 1-based fact index of the first hit (facts split on '.') or 0. */
static int find_grounded(const char* hay, const char* num) {
  int nl = (int)strlen(num);
  if (nl == 0) return 0;
  const char* p = hay;
  while ((p = strstr(p, num)) != NULL) {
    int before_ok = (p == hay) || !is_numch(p[-1]);
    char after = p[nl];
    int after_ok = !is_numch(after) ||
                   (after == '.' && !isdigit((unsigned char)p[nl + 1]));
    if (before_ok && after_ok) {
      /* count sentence terminators before the hit -> fact index */
      int fact = 1;
      for (const char* q = hay; q < p; q++)
        if (*q == '.' && !isdigit((unsigned char)q[1])) fact++;
      return fact;
    }
    p += 1;
  }
  return 0;
}

/* Is position p inside a <calc>...</calc> or <count>...</count> span? */
static int inside_tool(const char* text, const char* p) {
  const char* best_open = NULL; const char* close_tag = NULL;
  const char* q;
  for (q = text; (q = strstr(q, "<calc>"))  && q < p; q++)
    { best_open = q; close_tag = "</calc>"; }
  for (q = text; (q = strstr(q, "<count>")) && q < p; q++)
    if (!best_open || q > best_open) { best_open = q; close_tag = "</count>"; }
  if (!best_open) return 0;
  const char* close = strstr(best_open, close_tag);
  return close == NULL || p < close + (int)strlen(close_tag);
}

/* ---------- check 2: tool recheck ---------- */

static int recheck_calc(const char* gen, char* report, int repsz) {
  const char* p = gen;
  while ((p = strstr(p, "<calc>")) != NULL) {
    long a, b, r; char op;
    if (sscanf(p + 6, "%ld%c%ld=%ld", &a, &op, &b, &r) == 4) {
      long e;
      switch (op) {
        case '+': e = a + b; break;
        case '-': e = a - b; break;
        case '*': e = a * b; break;
        case '/': if (b == 0) { p++; continue; } e = a / b; break;
        default: p++; continue;
      }
      if (e != r) {
        snprintf(report, repsz, "TOOL FAIL: %ld%c%ld=%ld, chip says %ld",
                 a, op, b, r, e);
        return 0;
      }
    }
    p++;
  }
  return 1;
}

static int recheck_count(const char* gen, char* report, int repsz) {
  const char* p = gen;
  while ((p = strstr(p, "<count>")) != NULL) {
    char word[48]; char ch = 0; int wl = 0; const char* q = p + 7;
    while (*q && *q != ',' && *q != '=' && wl < (int)sizeof(word) - 1)
      word[wl++] = *q++;
    word[wl] = '\0';
    if (wl == 0) { p++; continue; }
    if (*q == ',') { ch = q[1]; q += 2; }
    if (*q != '=') { p++; continue; }
    int claimed = atoi(q + 1);
    int n = 0;
    if (ch) { for (int i = 0; i < wl; i++) if (word[i] == ch) n++; }
    else n = wl;
    if (n != claimed) {
      snprintf(report, repsz, "TOOL FAIL: count(%s%s%c)=%d, chip says %d",
               word, ch ? "," : "", ch ? ch : ' ', claimed, n);
      return 0;
    }
    p++;
  }
  return 1;
}

/* ---------- refusal detection ---------- */

static int looks_like_refusal(const char* gen) {
  /* lowercase copy, static (stack discipline) */
  static char low[1024];
  int i = 0;
  for (; gen[i] && i < (int)sizeof(low) - 1; i++)
    low[i] = (char)tolower((unsigned char)gen[i]);
  low[i] = '\0';
  return strstr(low, "cannot be determined") != NULL ||
         strstr(low, "but not the") != NULL ||
         strstr(low, "do not mention") != NULL;
}

/* ---------- main entry ---------- */

verify_rc verify_answer(const char* prompt, const char* gen,
                        char* report, int repsz) {
  report[0] = '\0';
  if (!gen || !gen[0]) { snprintf(report, repsz, "empty"); return VERIFY_EMPTY; }

  if (!recheck_calc(gen, report, repsz))  return VERIFY_TOOL_FAIL;
  if (!recheck_count(gen, report, repsz)) return VERIFY_TOOL_FAIL;

  int refusal = looks_like_refusal(gen);

  /* grounding pass over every number in gen */
  static char cites[160]; cites[0] = '\0';
  int nnum = 0, cited = 0;
  const char* p = gen;
  while (*p) {
    if (isdigit((unsigned char)*p) && (p == gen || !is_numch(p[-1]))) {
      char num[MAX_NUM_LEN];
      int adv = grab_number(p, num, sizeof num);
      if (num[0]) {
        nnum++;
        if (inside_tool(gen, p)) {
          /* chip-computed or chip-rechecked: grounded by construction */
        } else {
          int f_prompt = find_grounded(prompt, num);
          int f_gen_tool = 0;
          /* result restated after a tool tag: grounded if the same
             number exists inside a (rechecked) tool span in gen */
          if (!f_prompt) {
            const char* g = gen;
            while ((g = strstr(g, num)) != NULL) {
              if (g != p && inside_tool(gen, g)) { f_gen_tool = 1; break; }
              g++;
            }
          }
          if (!f_prompt && !f_gen_tool) {
            if (refusal) {
              snprintf(report, repsz,
                       "REFUSAL asserts number '%s'", num);
              return VERIFY_UNGROUNDED;
            }
            snprintf(report, repsz,
                     "UNGROUNDED number '%s' not in facts", num);
            return VERIFY_UNGROUNDED;
          }
          if (f_prompt && cited < 4) {
            int l = (int)strlen(cites);
            snprintf(cites + l, sizeof(cites) - l, "%s%s <- fact %d",
                     cited ? ", " : "", num, f_prompt);
            cited++;
          }
        }
      }
      p += adv;
    } else p++;
  }

  if (refusal) { snprintf(report, repsz, "refusal, no asserted numbers");
                 return VERIFY_REFUSAL; }
  if (nnum == 0) { snprintf(report, repsz, "no numbers to check");
                   return VERIFY_PASS; }
  snprintf(report, repsz, "%s", cites[0] ? cites : "grounded via tools");
  return VERIFY_PASS;
}

/* ---------- planted-error self-test ---------- */

typedef struct { const char* name; const char* prompt; const char* gen;
                 verify_rc want; } vt_case;

int verify_selftest(void) {
  static const vt_case T[] = {
    { "clean lookup",
      "Facts: The melting point of aluminium is 660 degrees Celsius. "
      "The atomic number of aluminium is 13. Question: melting point?",
      "The facts give the melting point of aluminium as 660 degrees "
      "Celsius. So the answer is 660 degrees Celsius.",
      VERIFY_PASS },
    { "PLANTED ungrounded number",
      "Facts: The melting point of aluminium is 660 degrees Celsius.",
      "So the answer is 4242 degrees Celsius.",
      VERIFY_UNGROUNDED },
    { "PLANTED partial-copy (decimal truncation)",
      "Facts: The density of X is 103.296 units.",
      "So the answer is 103 units.",
      VERIFY_UNGROUNDED },          /* '103' must not match inside 103.296 */
    { "clean calc chain",
      "Facts: The expression is (+ 3 4).",
      "First (+ 3 4) gives <calc>3+4=7</calc>. So the value is 7.",
      VERIFY_PASS },
    { "PLANTED wrong calc result",
      "Facts: The expression is (+ 3 4).",
      "First (+ 3 4) gives <calc>3+4=8</calc>. So the value is 8.",
      VERIFY_TOOL_FAIL },
    { "PLANTED wrong count",
      "Facts: none.",
      "<count>strawberry,r=5</count> So the answer is 5.",
      VERIFY_TOOL_FAIL },
    { "clean refusal",
      "Facts: The density of copper is 8960. Question: boiling point?",
      "The facts give the density of copper but not the boiling point. "
      "Cannot be determined.",
      VERIFY_REFUSAL },
    { "PLANTED refusal that asserts a number",
      "Facts: The density of copper is 8960.",
      "Cannot be determined, but it is probably 1234.",
      VERIFY_UNGROUNDED },
  };
  int fails = 0; char rep[192];
  for (unsigned i = 0; i < sizeof(T)/sizeof(T[0]); i++) {
    verify_rc got = verify_answer(T[i].prompt, T[i].gen, rep, sizeof rep);
    int ok = (got == T[i].want);
    printf("  [%s] %-38s -> %d (%s)\n", ok ? "ok" : "FAIL",
           T[i].name, (int)got, rep);
    if (!ok) fails++;
  }
  printf("verify_selftest: %s\n", fails ? "FAILED" : "all planted errors caught");
  return fails;
}
