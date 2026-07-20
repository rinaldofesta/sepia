/* Runs quant dequant fixtures (files under tools/fixtures/quants): dequantize
 * raw blocks, require BITWISE equality with the expected f32 payload. Usage:
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

static int run_fixture(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "test_quants: cannot open %s\n", path); exit(1); }
    uint32_t magic = rd_u32(f, path), ver = rd_u32(f, path);
    if (magic != 0x58465153u || ver != 1) { fprintf(stderr, "test_quants: bad header %s\n", path); exit(1); }
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

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_quants fixture.bin [...]\n"); return 2; }
    int bad = 0;
    for (int i = 1; i < argc; i++) bad |= run_fixture(argv[i]);
    if (bad) { fprintf(stderr, "test_quants: FAIL\n"); return 1; }
    printf("test_quants ok\n");
    return 0;
}
