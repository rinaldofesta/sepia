/* GGML block-format CPU dequantization.
 * Block layouts and reference dequant order follow ggml (MIT, see NOTICE),
 * cross-checked bit-exactly against gguf-py fixtures (tools/make_quant_fixtures.py).
 * Type ids and size-table conventions shared with ds4's gguf-tools/quants.h (MIT).
 * Bit-exactness requires no FMA contraction: */
#pragma STDC FP_CONTRACT OFF

#include "quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define QK_K 256
#define QK8_0 32

static void qdie(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "sepia: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

typedef struct { int type; int64_t block; size_t bytes; } TypeSize;
static const TypeSize TYPE_SIZES[] = {
    { SEPIA_T_F32, 1, 4 }, { SEPIA_T_F16, 1, 2 }, { SEPIA_T_Q8_0, QK8_0, 34 },
    { SEPIA_T_Q4_K, QK_K, 144 }, { SEPIA_T_Q5_K, QK_K, 176 }, { SEPIA_T_Q6_K, QK_K, 210 },
    { SEPIA_T_IQ2_XS, QK_K, 74 }, { SEPIA_T_IQ3_XXS, QK_K, 98 }, { SEPIA_T_IQ4_XS, QK_K, 136 },
};

static const TypeSize *type_size(int t) {
    for (size_t i = 0; i < sizeof(TYPE_SIZES)/sizeof(TYPE_SIZES[0]); i++)
        if (TYPE_SIZES[i].type == t) return &TYPE_SIZES[i];
    return NULL;
}

int quants_supported(int t) { return type_size(t) != NULL; }
int64_t quants_block_size(int t) {
    const TypeSize *s = type_size(t);
    if (!s) qdie("quants: unsupported ggml type %d", t);
    return s->block;
}
size_t quants_type_size(int t) {
    const TypeSize *s = type_size(t);
    if (!s) qdie("quants: unsupported ggml type %d", t);
    return s->bytes;
}
size_t quants_row_bytes(int t, int64_t ne) {
    int64_t bs = quants_block_size(t);
    if (ne % bs != 0) qdie("quants: ne=%lld not a multiple of block size %lld (type %d)",
                           (long long)ne, (long long)bs, t);
    return (size_t)(ne / bs) * quants_type_size(t);
}

float quants_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static void dequant_f32(const void *src, float *dst, int64_t n) { memcpy(dst, src, (size_t)n * 4); }
static void dequant_f16(const void *src, float *dst, int64_t n) {
    const uint16_t *s = (const uint16_t *)src;
    for (int64_t i = 0; i < n; i++) dst[i] = quants_f16_to_f32(s[i]);
}

void dequantize_row(int t, const void *src, float *dst, int64_t n) {
    if (n % quants_block_size(t) != 0)
        qdie("quants: dequantize_row n=%lld not block-aligned for type %d", (long long)n, t);
    switch (t) {
    case SEPIA_T_F32: dequant_f32(src, dst, n); break;
    case SEPIA_T_F16: dequant_f16(src, dst, n); break;
    default: qdie("quants: dequant for ggml type %d not yet implemented", t);
    }
}
