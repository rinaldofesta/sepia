#ifndef SEPIA_TOKENIZER_H
#define SEPIA_TOKENIZER_H
#include <stdint.h>
#include <stddef.h>

typedef struct Tokenizer Tokenizer;

Tokenizer *tokenizer_load(const char *path);            /* dies on any format error */
void       tokenizer_free(Tokenizer *t);
int32_t    tokenizer_bos_id(const Tokenizer *t);        /* 200006 for real Inkling (== eos) */
int32_t    tokenizer_eos_id(const Tokenizer *t);
/* Encode UTF-8 text -> ids. Returns count; dies if > max_ids. Pretokenizes
 * with the o200k-family regex scanner (src/unicode_tables.h), then runs BPE
 * per pretoken piece. */
int        tokenizer_encode(const Tokenizer *t, const char *text, int32_t *ids, int max_ids);
/* Append the raw bytes of ids[0..n) into buf (cap bytes incl NUL); dies on overflow. */
void       tokenizer_decode(const Tokenizer *t, const int32_t *ids, int n, char *buf, size_t cap);
const char *tokenizer_regex(const Tokenizer *t);        /* the embedded pre-regex string */

/* BPE over one pretoken piece (bytes[0..n)): seed with per-byte ids, then
 * repeatedly apply the lowest-rank applicable merge. Task 10's pretokenizer
 * scanner calls this once per piece. */
int tokenizer_bpe_piece(const Tokenizer *t, const uint8_t *bytes, int n,
                        int32_t *ids, int max_ids);

#endif
