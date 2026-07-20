/* Runs quant fixtures (files under tools/fixtures/quants). Two families,
 * dispatched on the leading magic:
 *   0x58465153 "SQFX" -- dequantize_row fixtures: dequantize raw blocks,
 *     require BITWISE equality with the expected f32 payload.
 *   0x584C5153 "SQLX" -- qlinear fixtures: dequantize each weight row and
 *     double-accumulate its dot with x, require BITWISE equality with the
 *     expected f32 output vector. `qlinear_ref` below is a local
 *     reimplementation of the exact same loop as src/sepia.c's `qlinear`
 *     (dequantize_row per row + double-accumulated dot); the plan-level
 *     guarantee is that src/sepia.c's `qlinear` is a line-for-line copy, so
 *     this test never links sepia.c itself.
 * Usage:
 *   ./test_quants tools/fixtures/quants/f16.bin [more fixtures ...] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/quants.h"

static uint32_t rd_u32(FILE *f, const char *path) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "test_quants: short read in %s\n", path); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

/* Local reimplementation of src/sepia.c's qlinear: per output row, dequantize
 * the row into `scratch` (in_dim floats), then a double-accumulated dot with
 * x, rounded to float32 on write -- same numerical policy as sepia.c's
 * dotf/linear (src/sepia.c:687-696). */
static void qlinear_ref(int ggml_type, const uint8_t *wq, uint32_t out_dim, uint32_t in_dim,
                         const float *x, float *y, float *scratch) {
    size_t row_bytes = quants_row_bytes(ggml_type, (int64_t)in_dim);
    for (uint32_t o = 0; o < out_dim; o++) {
        dequantize_row(ggml_type, wq + (size_t)o * row_bytes, scratch, (int64_t)in_dim);
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)scratch[i] * (double)x[i];
        y[o] = (float)acc;
    }
}

static int run_qlinear_fixture(FILE *f, const char *path) {
    uint32_t ver = rd_u32(f, path);
    if (ver != 1) { fprintf(stderr, "test_quants: bad header %s\n", path); exit(1); }
    uint32_t ggml_type = rd_u32(f, path);
    uint32_t out_dim = rd_u32(f, path), in_dim = rd_u32(f, path);
    size_t row_bytes = quants_row_bytes((int)ggml_type, (int64_t)in_dim);
    size_t wq_bytes = row_bytes * out_dim;
    uint8_t *wq = malloc(wq_bytes ? wq_bytes : 1);
    float *x = malloc((size_t)in_dim * 4);
    float *scratch = malloc((size_t)in_dim * 4);
    float *expect = malloc((size_t)out_dim * 4), *got = malloc((size_t)out_dim * 4);
    if (!wq || !x || !scratch || !expect || !got) { fprintf(stderr, "test_quants: oom\n"); exit(1); }
    if (fread(wq, 1, wq_bytes, f) != wq_bytes ||
        fread(x, 4, in_dim, f) != in_dim ||
        fread(expect, 4, out_dim, f) != out_dim) {
        fprintf(stderr, "test_quants: truncated fixture %s\n", path); exit(1);
    }
    fclose(f);
    qlinear_ref((int)ggml_type, wq, out_dim, in_dim, x, got, scratch);
    int fails = 0;
    for (uint32_t o = 0; o < out_dim; o++) {
        if (memcmp(&got[o], &expect[o], 4) != 0) {
            if (fails < 5) {
                uint32_t gb, eb; memcpy(&gb, &got[o], 4); memcpy(&eb, &expect[o], 4);
                fprintf(stderr, "  %s elem %u: got %.9g (0x%08x) expect %.9g (0x%08x)\n",
                        path, o, got[o], gb, expect[o], eb);
            }
            fails++;
        }
    }
    printf("%-44s SQLX type %2u  %ux%u  %s\n", path, ggml_type, out_dim, in_dim, fails ? "FAIL" : "ok");
    free(wq); free(x); free(scratch); free(expect); free(got);
    return fails != 0;
}

static int run_dequant_fixture(FILE *f, const char *path) {
    uint32_t ver = rd_u32(f, path);
    if (ver != 1) { fprintf(stderr, "test_quants: bad header %s\n", path); exit(1); }
    uint32_t type = rd_u32(f, path), n_blocks = rd_u32(f, path);
    uint32_t block_elems = rd_u32(f, path), block_bytes = rd_u32(f, path);
    if ((int64_t)block_elems != quants_block_size((int)type) ||
        (size_t)block_bytes != quants_type_size((int)type)) {
        fprintf(stderr, "test_quants: %s size table mismatch (type %u: fixture %u/%u, code %lld/%zu)\n",
                path, type, block_elems, block_bytes,
                (long long)quants_block_size((int)type), quants_type_size((int)type));
        exit(1);
    }
    size_t raw_sz = (size_t)n_blocks * block_bytes;
    int64_t n = (int64_t)n_blocks * block_elems;
    void *raw = malloc(raw_sz);
    float *expect = malloc((size_t)n * 4), *got = malloc((size_t)n * 4);
    if (!raw || !expect || !got) { fprintf(stderr, "test_quants: oom\n"); exit(1); }
    if (fread(raw, 1, raw_sz, f) != raw_sz ||
        fread(expect, 4, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "test_quants: truncated fixture %s\n", path); exit(1);
    }
    fclose(f);
    dequantize_row((int)type, raw, got, n);
    int fails = 0;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&got[i], &expect[i], 4) != 0) {
            if (fails < 5) {
                uint32_t gb, eb; memcpy(&gb, &got[i], 4); memcpy(&eb, &expect[i], 4);
                fprintf(stderr, "  %s elem %lld: got %.9g (0x%08x) expect %.9g (0x%08x)\n",
                        path, (long long)i, got[i], gb, expect[i], eb);
            }
            fails++;
        }
    }
    printf("%-44s type %2u  %5u blocks  %s\n", path, type, n_blocks, fails ? "FAIL" : "ok");
    free(raw); free(expect); free(got);
    return fails != 0;
}

static int run_fixture(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "test_quants: cannot open %s\n", path); exit(1); }
    uint32_t magic = rd_u32(f, path);
    if (magic == 0x584C5153u) return run_qlinear_fixture(f, path);
    if (magic == 0x58465153u) return run_dequant_fixture(f, path);
    fprintf(stderr, "test_quants: bad header %s\n", path); exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_quants fixture.bin [...]\n"); return 2; }
    int bad = 0;
    for (int i = 1; i < argc; i++) bad |= run_fixture(argv[i]);
    if (bad) { fprintf(stderr, "test_quants: FAIL\n"); return 1; }
    printf("test_quants ok\n");
    return 0;
}
