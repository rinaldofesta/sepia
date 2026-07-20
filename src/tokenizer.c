/* SPTK v1 sidecar loader + ranked BPE + decode (tools/export_tokenizer.py
 * writes the format this reads). Token strings are stored as raw bytes
 * (the GPT-2 byte<->unicode indirection is resolved offline by the
 * exporter), so encode/decode here never touch that mapping. */
#include "tokenizer.h"
#include "unicode_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

/* Hand-coded scanner for the o200k-family pre-tokenizer pattern (llama.cpp's
 * lookahead formulation). It MUST match docs/tokenizer-pre-regex.txt
 * byte-for-byte; tokenizer_load enforces this below. */
static const char SCANNER_REGEX[] = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

static void tdie(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "sepia: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap); exit(1);
}

static void *tmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) tdie("tokenizer: out of memory");
    return p;
}

struct Tokenizer {
    uint32_t vocab, n_merges, bos, eos, n_special;
    char *regex;
    uint32_t byte_token_id[256];
    uint32_t *tok_off;      /* vocab+1 entries: token i spans [tok_off[i], tok_off[i+1]) in tok_blob */
    uint8_t *tok_blob;
    uint8_t *token_type;    /* vocab entries */
    uint64_t *pair_keys;    /* open-addressed pair map: key = (left<<32)|right, empty = UINT64_MAX */
    int32_t *pair_rank;
    int32_t *pair_merged;
    uint32_t pair_cap;
};

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

Tokenizer *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) tdie("tokenizer: cannot open %s", path);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *d = malloc((size_t)sz);
    if (!d || fread(d, 1, (size_t)sz, f) != (size_t)sz) tdie("tokenizer: short read %s", path);
    fclose(f);
    if (sz < 28 || rd_u32(d) != 0x4B545053u || rd_u32(d + 4) != 1)
        tdie("tokenizer: bad magic/version in %s", path);
    Tokenizer *t = calloc(1, sizeof *t);
    if (!t) tdie("tokenizer: oom");
    t->vocab = rd_u32(d + 8); t->n_merges = rd_u32(d + 12);
    t->bos = rd_u32(d + 16); t->eos = rd_u32(d + 20); t->n_special = rd_u32(d + 24);
    size_t off = 28;
    uint32_t rlen = rd_u32(d + off); off += 4;
    t->regex = tmalloc(rlen + 1);
    memcpy(t->regex, d + off, rlen); t->regex[rlen] = 0; off += rlen;
    if (strcmp(t->regex, SCANNER_REGEX) != 0)
        tdie("tokenizer: sidecar regex differs from the scanner's pattern; regenerate or re-port the scanner");
    for (int i = 0; i < 256; i++) { t->byte_token_id[i] = rd_u32(d + off); off += 4; }
    t->tok_off = tmalloc(((size_t)t->vocab + 1) * 4);
    for (uint32_t i = 0; i <= t->vocab; i++) { t->tok_off[i] = rd_u32(d + off); off += 4; }
    size_t blob = t->tok_off[t->vocab];
    t->tok_blob = tmalloc(blob); memcpy(t->tok_blob, d + off, blob); off += blob;
    t->token_type = tmalloc(t->vocab); memcpy(t->token_type, d + off, t->vocab); off += t->vocab;

    t->pair_cap = 1;
    while (t->pair_cap < 2 * t->n_merges) t->pair_cap <<= 1;
    t->pair_keys = tmalloc((size_t)t->pair_cap * 8);
    t->pair_rank = tmalloc((size_t)t->pair_cap * 4);
    t->pair_merged = tmalloc((size_t)t->pair_cap * 4);
    memset(t->pair_keys, 0xFF, (size_t)t->pair_cap * 8);
    for (uint32_t r = 0; r < t->n_merges; r++) {
        uint32_t l = rd_u32(d + off), rr = rd_u32(d + off + 4), m = rd_u32(d + off + 8);
        off += 12;
        uint64_t key = ((uint64_t)l << 32) | rr;
        uint32_t slot = (uint32_t)(key * 0x9E3779B97F4A7C15ull) & (t->pair_cap - 1);
        while (t->pair_keys[slot] != UINT64_MAX) slot = (slot + 1) & (t->pair_cap - 1);
        t->pair_keys[slot] = key; t->pair_rank[slot] = (int32_t)r; t->pair_merged[slot] = (int32_t)m;
    }
    off += (size_t)t->n_special * 4;   /* special ids: not needed at encode time yet */
    if (off != (size_t)sz) tdie("tokenizer: trailing bytes in %s (%zu vs %ld)", path, off, sz);
    free(d);
    return t;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    free(t->regex);
    free(t->tok_off);
    free(t->tok_blob);
    free(t->token_type);
    free(t->pair_keys);
    free(t->pair_rank);
    free(t->pair_merged);
    free(t);
}

int32_t tokenizer_bos_id(const Tokenizer *t) { return (int32_t)t->bos; }
int32_t tokenizer_eos_id(const Tokenizer *t) { return (int32_t)t->eos; }
const char *tokenizer_regex(const Tokenizer *t) { return t->regex; }

static int pair_lookup(const Tokenizer *t, uint32_t l, uint32_t r,
                       int32_t *rank, int32_t *merged) {
    uint64_t key = ((uint64_t)l << 32) | r;
    uint32_t slot = (uint32_t)(key * 0x9E3779B97F4A7C15ull) & (t->pair_cap - 1);
    while (t->pair_keys[slot] != UINT64_MAX) {
        if (t->pair_keys[slot] == key) { *rank = t->pair_rank[slot]; *merged = t->pair_merged[slot]; return 1; }
        slot = (slot + 1) & (t->pair_cap - 1);
    }
    return 0;
}

/* BPE over one pretoken piece: seed with per-byte ids, then repeatedly apply
 * the lowest-rank applicable merge (leftmost on rank ties, matching HF/GPT-2). */
int tokenizer_bpe_piece(const Tokenizer *t, const uint8_t *bytes, int n,
                        int32_t *ids, int max_ids) {
    if (n == 0) return 0;
    int32_t *sym = malloc((size_t)n * 4);
    if (!sym) tdie("tokenizer: oom");
    int m = n;
    for (int i = 0; i < n; i++) sym[i] = (int32_t)t->byte_token_id[bytes[i]];
    for (;;) {
        int best_i = -1; int32_t best_rank = INT32_MAX, best_merged = -1;
        for (int i = 0; i + 1 < m; i++) {
            int32_t rank, merged;
            if (pair_lookup(t, (uint32_t)sym[i], (uint32_t)sym[i + 1], &rank, &merged) &&
                rank < best_rank) {
                best_rank = rank; best_i = i; best_merged = merged;
            }
        }
        if (best_i < 0) break;
        sym[best_i] = best_merged;
        memmove(sym + best_i + 1, sym + best_i + 2, (size_t)(m - best_i - 2) * 4);
        m--;
    }
    if (m > max_ids) tdie("tokenizer: piece produced %d ids > cap %d", m, max_ids);
    memcpy(ids, sym, (size_t)m * 4);
    free(sym);
    return m;
}

/* ---- Pre-tokenizer scanner (SCANNER_REGEX above) ----------------------- */

static uint8_t uc_class(uint32_t cp) {
    uint32_t lo = 0, hi = uc_n_ranges;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < uc_ranges[mid].lo) hi = mid;
        else if (cp > uc_ranges[mid].hi) lo = mid + 1;
        else return uc_ranges[mid].mask;
    }
    return 0;
}

static int is_letter(int m) { return m & UC_L; }
static int is_num(int m)    { return m & UC_N; }
static int is_ws(int m)     { return m & UC_WS; }

/* Decode one UTF-8 codepoint at s[i] (i < n). On malformed/truncated input,
 * fall back to a single raw byte classified as codepoint 0xFFFD (which maps
 * to no L/M/N/WS class): the byte still flows into the piece untouched, only
 * scanner classification treats it as "no class". */
static int utf8_next(const uint8_t *s, int n, int i, uint32_t *cp) {
    uint8_t b0 = s[i];
    if (b0 < 0x80) { *cp = b0; return i + 1; }
    int len; uint32_t c;
    if ((b0 & 0xE0) == 0xC0)      { len = 2; c = b0 & 0x1F; }
    else if ((b0 & 0xF0) == 0xE0) { len = 3; c = b0 & 0x0F; }
    else if ((b0 & 0xF8) == 0xF0) { len = 4; c = b0 & 0x07; }
    else { *cp = 0xFFFD; return i + 1; }
    if (i + len > n) { *cp = 0xFFFD; return i + 1; }
    for (int k = 1; k < len; k++) {
        uint8_t bk = s[i + k];
        if ((bk & 0xC0) != 0x80) { *cp = 0xFFFD; return i + 1; }
        c = (c << 6) | (uint32_t)(bk & 0x3F);
    }
    *cp = c;
    return i + len;
}

/* 's 't 're 've 'm 'll 'd, ASCII case-insensitive; returns matched length or 0. */
static int match_contraction(const uint8_t *s, int n, int i) {
    if (i >= n || s[i] != '\'' || i + 1 >= n) return 0;
    int c1 = s[i + 1];
    int lc1 = (c1 >= 'A' && c1 <= 'Z') ? c1 + 32 : c1;
    if (lc1 == 's' || lc1 == 't' || lc1 == 'm' || lc1 == 'd') return 2;
    if (i + 2 < n) {
        int c2 = s[i + 2];
        int lc2 = (c2 >= 'A' && c2 <= 'Z') ? c2 + 32 : c2;
        if ((lc1 == 'r' && lc2 == 'e') || (lc1 == 'v' && lc2 == 'e') || (lc1 == 'l' && lc2 == 'l'))
            return 3;
    }
    return 0;
}

static int next_pretoken(const uint8_t *s, int n, int i) {
    uint32_t cp; int j, len;
    /* Branch 1+2: optional non-[\r\n L N] prefix, then letter run */
    j = utf8_next(s, n, i, &cp);
    int m0 = (int)uc_class(cp);
    int start_letters = i;
    if (cp != '\r' && cp != '\n' && !is_letter(m0) && !is_num(m0)) start_letters = j;
    /* Branches 1+2 letter runs: A*B+ | A+B*  (A = letter-or-mark not ASCII a-z,
     * B = letter-or-mark not ASCII A-Z; A∩B is every non-ASCII letter and mark,
     * so greedy A-then-B linear runs give the same extent as the regex's
     * backtracking — the B-empty case falls back to A+). */
    int a_run = 0, b_run = 0, pos = start_letters;
    while (pos < n) {
        int q = utf8_next(s, n, pos, &cp); int mm = (int)uc_class(cp);
        if (!((mm & (UC_L | UC_M)) && !(cp >= 'a' && cp <= 'z'))) break;
        a_run++; pos = q;
    }
    int after_a = pos;
    while (pos < n) {
        int q = utf8_next(s, n, pos, &cp); int mm = (int)uc_class(cp);
        if (!((mm & (UC_L | UC_M)) && !(cp >= 'A' && cp <= 'Z'))) break;
        b_run++; pos = q;
    }
    if (a_run > 0 || b_run > 0) {
        int end = (b_run > 0) ? pos : after_a;   /* A*B+ tail present, else A+ only */
        if (end > start_letters) {
            len = match_contraction(s, n, end);
            return end + len - i;
        }
    }
    /* Branch 3: \p{N}{1,3} */
    if (is_num(m0)) {
        int cnt = 1, pos3 = j;
        while (cnt < 3 && pos3 < n) {
            int q = utf8_next(s, n, pos3, &cp);
            if (!is_num((int)uc_class(cp))) break;
            cnt++; pos3 = q;
        }
        return pos3 - i;
    }
    /* Branch 4: ' '? [^\s L N]+ [\r\n/]* */
    {
        int pos4 = i; uint32_t c4;
        int q4 = utf8_next(s, n, pos4, &c4);
        if (c4 == ' ' && q4 < n) {
            uint32_t c5; utf8_next(s, n, q4, &c5);
            int m5 = (int)uc_class(c5);
            if (!is_ws(m5) && !is_letter(m5) && !is_num(m5)) pos4 = q4;
        }
        int q7 = pos4, got = 0;
        while (q7 < n) {
            uint32_t c7; int q8 = utf8_next(s, n, q7, &c7);
            int m7 = (int)uc_class(c7);
            if (is_ws(m7) || is_letter(m7) || is_num(m7)) break;
            got++; q7 = q8;
        }
        if (got > 0) {
            while (q7 < n && (s[q7] == '\r' || s[q7] == '\n' || s[q7] == '/')) q7++;
            return q7 - i;
        }
    }
    /* Branch 5: \s*[\r\n]+ (never consumes trailing non-newline ws after the
     * last newline: last_nl tracks the position right after the last \r/\n
     * seen, while pos5 may run further through trailing plain whitespace). */
    {
        int pos5 = i, last_nl = -1;
        while (pos5 < n) {
            uint32_t c5; int q = utf8_next(s, n, pos5, &c5);
            if (c5 == '\r' || c5 == '\n') { last_nl = q; pos5 = q; }
            else if (is_ws((int)uc_class(c5))) { pos5 = q; }
            else break;
        }
        if (last_nl > 0) return last_nl - i;
    }
    /* Branch 6+7: \s+(?!\S) | \s+ : yield all but the last whitespace char
     * when the run is followed by non-whitespace (give-back), else the whole
     * run. */
    {
        int pos6 = i, ws = 0, prev = i;
        while (pos6 < n) {
            uint32_t c6; int q = utf8_next(s, n, pos6, &c6);
            if (!is_ws((int)uc_class(c6))) break;
            ws++; prev = pos6; pos6 = q;
        }
        if (ws > 0) {
            if (pos6 < n && ws > 1) return prev - i;
            return pos6 - i;
        }
    }
    /* Fallback: single codepoint (unmatched by any branch). */
    return j - i;
}

int tokenizer_encode(const Tokenizer *t, const char *text, int32_t *ids, int max_ids) {
    const uint8_t *s = (const uint8_t *)text;
    int n = (int)strlen(text), i = 0, total = 0;
    while (i < n) {
        int len = next_pretoken(s, n, i);
        if (len <= 0) tdie("tokenizer: scanner made no progress at byte %d", i);
        total += tokenizer_bpe_piece(t, s + i, len, ids + total, max_ids - total);
        i += len;
    }
    return total;
}

void tokenizer_decode(const Tokenizer *t, const int32_t *ids, int n, char *buf, size_t cap) {
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        uint32_t id = (uint32_t)ids[i];
        if (id >= t->vocab) tdie("tokenizer: id %u out of range", id);
        uint32_t a = t->tok_off[id], b = t->tok_off[id + 1];
        if (w + (b - a) + 1 > cap) tdie("tokenizer: decode buffer overflow");
        memcpy(buf + w, t->tok_blob + a, b - a);
        w += b - a;
    }
    buf[w] = 0;
}
