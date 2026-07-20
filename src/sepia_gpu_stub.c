/* src/sepia_gpu_stub.c -- non-Darwin stand-in for the Metal runtime shim.
 *
 * SEPIA's GPU path (src/sepia_metal.m) is Objective-C/Metal and only
 * builds on Darwin. This stub gives every non-Darwin build the same
 * sepia_gpu_* symbols so `sepia.c` links unconditionally; each call
 * reports "no GPU support" so a `--metal` request fails cleanly via
 * sepia.c's die(), rather than the build failing to link. See
 * src/sepia_gpu.h for the full contract.
 */
#include "sepia_gpu.h"

#include <stdio.h>

int sepia_gpu_available(void) {
    return 0;
}

int sepia_gpu_init(const char *metal_dir) {
    (void)metal_dir;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

void sepia_gpu_shutdown(void) {
}

const char *sepia_gpu_device_name(void) {
    return NULL;
}

SepiaGpuBuf *sepia_gpu_wrap_mmap(void *base, size_t len) {
    (void)base;
    (void)len;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return NULL;
}

SepiaGpuBuf *sepia_gpu_alloc(size_t len, int gpu_private) {
    (void)len;
    (void)gpu_private;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return NULL;
}

void sepia_gpu_free(SepiaGpuBuf *b) {
    (void)b;
}

void *sepia_gpu_host_ptr(SepiaGpuBuf *b) {
    (void)b;
    return NULL;
}

uint64_t sepia_gpu_gpu_addr(SepiaGpuBuf *b) {
    (void)b;
    return 0;
}

int sepia_gpu_selftest_touch(SepiaGpuBuf *buf, size_t n_floats) {
    (void)buf;
    (void)n_floats;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_begin(void) {
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_flush(void) {
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_end(void) {
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_rmsnorm(SepiaGpuBuf *w, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n, float eps) {
    (void)w; (void)x; (void)y; (void)rows; (void)n; (void)eps;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_matvec(SepiaGpuBuf *w, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t out_dim, int64_t in_dim) {
    (void)w; (void)x; (void)y; (void)out_dim; (void)in_dim;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_silu_mul(SepiaGpuBuf *g, SepiaGpuBuf *u, SepiaGpuBuf *z, int64_t n) {
    (void)g; (void)u; (void)z; (void)n;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_add(SepiaGpuBuf *a, SepiaGpuBuf *b, SepiaGpuBuf *out, int64_t n) {
    (void)a; (void)b; (void)out; (void)n;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_softmax(SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n) {
    (void)x; (void)y; (void)rows; (void)n;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_sconv(SepiaGpuBuf *w, SepiaGpuBuf *hist, SepiaGpuBuf *in, SepiaGpuBuf *out,
                     int64_t C, int64_t K, int64_t T) {
    (void)w; (void)hist; (void)in; (void)out; (void)C; (void)K; (void)T;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_matvec_q(int ggml_type, SepiaGpuBuf *w, size_t w_off, SepiaGpuBuf *x, SepiaGpuBuf *y,
                        int64_t out_dim, int64_t in_dim) {
    (void)ggml_type; (void)w; (void)w_off; (void)x; (void)y; (void)out_dim; (void)in_dim;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_dequant_rows(int ggml_type, SepiaGpuBuf *raw, size_t raw_off, SepiaGpuBuf *out, int64_t n) {
    (void)ggml_type; (void)raw; (void)raw_off; (void)out; (void)n;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_rel_project(SepiaGpuBuf *r_vec, SepiaGpuBuf *rel_proj, SepiaGpuBuf *rel_logits,
                           int64_t T, int64_t H, int64_t d_rel, int64_t rel_extent) {
    (void)r_vec; (void)rel_proj; (void)rel_logits; (void)T; (void)H; (void)d_rel; (void)rel_extent;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}

int sepia_gpu_banded_attn(SepiaGpuBuf *q, SepiaGpuBuf *k, SepiaGpuBuf *v, SepiaGpuBuf *rel_logits,
                           SepiaGpuBuf *attn_out,
                           const int64_t *kv_lo, const int64_t *kv_hi, const float *tau,
                           int64_t T, int64_t H, int64_t Hkv, int64_t Dh, int64_t rel_extent,
                           int64_t q_pos_base, int64_t kv_dim, float inv_head_dim) {
    (void)q; (void)k; (void)v; (void)rel_logits; (void)attn_out;
    (void)kv_lo; (void)kv_hi; (void)tau;
    (void)T; (void)H; (void)Hkv; (void)Dh; (void)rel_extent; (void)q_pos_base; (void)kv_dim; (void)inv_head_dim;
    fprintf(stderr, "sepia: metal: not supported on this platform\n");
    return 0;
}
