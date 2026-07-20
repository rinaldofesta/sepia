/* SPTK v1 sidecar loader + ranked BPE + decode (tools/export_tokenizer.py
 * writes the format this reads). Token strings are stored as raw bytes
 * (the GPT-2 byte<->unicode indirection is resolved offline by the
 * exporter), so encode/decode here never touch that mapping. */
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

static void tdie(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "sepia: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap); exit(1);
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
    t->regex = malloc(rlen + 1);
    memcpy(t->regex, d + off, rlen); t->regex[rlen] = 0; off += rlen;
    for (int i = 0; i < 256; i++) { t->byte_token_id[i] = rd_u32(d + off); off += 4; }
    t->tok_off = malloc(((size_t)t->vocab + 1) * 4);
    for (uint32_t i = 0; i <= t->vocab; i++) { t->tok_off[i] = rd_u32(d + off); off += 4; }
    size_t blob = t->tok_off[t->vocab];
    t->tok_blob = malloc(blob); memcpy(t->tok_blob, d + off, blob); off += blob;
    t->token_type = malloc(t->vocab); memcpy(t->token_type, d + off, t->vocab); off += t->vocab;

    t->pair_cap = 1;
    while (t->pair_cap < 2 * t->n_merges) t->pair_cap <<= 1;
    t->pair_keys = malloc((size_t)t->pair_cap * 8);
    t->pair_rank = malloc((size_t)t->pair_cap * 4);
    t->pair_merged = malloc((size_t)t->pair_cap * 4);
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

/* Task 9 placeholder pretokenization: the whole text is one piece.
 * Task 10 replaces this with the regex scanner. */
int tokenizer_encode(const Tokenizer *t, const char *text, int32_t *ids, int max_ids) {
    return tokenizer_bpe_piece(t, (const uint8_t *)text, (int)strlen(text), ids, max_ids);
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
