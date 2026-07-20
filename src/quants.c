/* GGML block-format CPU dequantization.
 * Block layouts and reference dequant order follow ggml (MIT, see NOTICE),
 * cross-checked bit-exactly against gguf-py fixtures (tools/make_quant_fixtures.py).
 * Type ids and size-table conventions shared with ds4's gguf-tools/quants.h (MIT).
 * Bit-exactness requires no FMA contraction: */
#pragma STDC FP_CONTRACT OFF

#include "quants.h"
#include "quants_grids.h"
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

static void dequant_q8_0(const void *src, float *dst, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t b = 0; b < n / QK8_0; b++) {
        uint16_t dbits; memcpy(&dbits, p, 2);
        float d = quants_f16_to_f32(dbits);
        const int8_t *qs = (const int8_t *)(p + 2);
        for (int j = 0; j < QK8_0; j++) dst[j] = d * (float)qs[j];
        p += 34; dst += QK8_0;
    }
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void dequant_q4_k(const void *src, float *dst, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t b = 0; b < n / QK_K; b++) {
        uint16_t dbits, mbits;
        memcpy(&dbits, p, 2); memcpy(&mbits, p + 2, 2);
        const float d = quants_f16_to_f32(dbits), min = quants_f16_to_f32(mbits);
        const uint8_t *scales = p + 4;
        const uint8_t *q = p + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) dst[l]      = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) dst[l + 32] = d2 * (q[l] >> 4)  - m2;
            dst += 64; q += 32; is += 2;
        }
        p += 144;
    }
}

static void dequant_q5_k(const void *src, float *dst, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t b = 0; b < n / QK_K; b++) {
        uint16_t dbits, mbits;
        memcpy(&dbits, p, 2); memcpy(&mbits, p + 2, 2);
        const float d = quants_f16_to_f32(dbits), min = quants_f16_to_f32(mbits);
        const uint8_t *scales = p + 4;
        const uint8_t *qh = p + 16;
        const uint8_t *ql = p + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) dst[l]      = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) dst[l + 32] = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
            dst += 64; ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
        p += 176;
    }
}

static void dequant_q6_k(const void *src, float *dst, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t b = 0; b < n / QK_K; b++) {
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t *sc = (const int8_t *)(p + 192);
        uint16_t dbits; memcpy(&dbits, p + 208, 2);
        const float d = quants_f16_to_f32(dbits);
        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)(((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                int8_t q2 = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                int8_t q3 = (int8_t)(((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32);
                int8_t q4 = (int8_t)(((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32);
                dst[l]      = d * sc[is + 0] * q1;
                dst[l + 32] = d * sc[is + 2] * q2;
                dst[l + 64] = d * sc[is + 4] * q3;
                dst[l + 96] = d * sc[is + 6] * q4;
            }
            dst += 128; ql += 64; qh += 32; sc += 8;
        }
        p += 210;
    }
}

static void dequant_iq2_xs(const void *src, float *dst, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t b = 0; b < n / QK_K; b++) {
        uint16_t dbits; memcpy(&dbits, p, 2);
        const float d = quants_f16_to_f32(dbits);
        uint16_t qs[32]; memcpy(qs, p + 2, 64);
        const uint8_t *scales = p + 66;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db[2];
            db[0] = d * (0.5f + (scales[ib32] & 0xF)) * 0.25f;
            db[1] = d * (0.5f + (scales[ib32] >> 4))  * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (qs[4 * ib32 + l] & 511));
                const uint8_t signs = ksigns_iq2xs[qs[4 * ib32 + l] >> 9];
                const float dl = db[l / 2];
                for (int j = 0; j < 8; j++)
                    dst[j] = dl * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                dst += 8;
            }
        }
        p += 74;
    }
}

void dequantize_row(int t, const void *src, float *dst, int64_t n) {
    if (n % quants_block_size(t) != 0)
        qdie("quants: dequantize_row n=%lld not block-aligned for type %d", (long long)n, t);
    switch (t) {
    case SEPIA_T_F32: dequant_f32(src, dst, n); break;
    case SEPIA_T_F16: dequant_f16(src, dst, n); break;
    case SEPIA_T_Q8_0: dequant_q8_0(src, dst, n); break;
    case SEPIA_T_Q4_K: dequant_q4_k(src, dst, n); break;
    case SEPIA_T_Q5_K: dequant_q5_k(src, dst, n); break;
    case SEPIA_T_Q6_K: dequant_q6_k(src, dst, n); break;
    case SEPIA_T_IQ2_XS: dequant_iq2_xs(src, dst, n); break;
    default: qdie("quants: dequant for ggml type %d not yet implemented", t);
    }
}
