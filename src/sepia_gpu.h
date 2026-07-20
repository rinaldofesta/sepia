/* src/sepia_gpu.h -- the ONLY GPU symbol surface sepia.c sees.
 * All functions return 0/NULL on failure after logging "sepia: metal: ...";
 * sepia.c decides whether that is fatal (--metal explicitly requested -> die). */
#ifndef SEPIA_GPU_H
#define SEPIA_GPU_H
#include <stdint.h>
#include <stddef.h>

int  sepia_gpu_available(void);                 /* 1 if a device exists and library compiled */
int  sepia_gpu_init(const char *metal_dir);     /* compiles the *.metal files under this dir; 1 on success */
void sepia_gpu_shutdown(void);
const char *sepia_gpu_device_name(void);

/* -------------------------------- buffers -------------------------------- */

/* Opaque; wraps an id<MTLBuffer> (and, in later tasks, an offset). Never
 * dereferenced by sepia.c -- only passed around and handed back to the
 * functions below. */
typedef struct SepiaGpuBuf SepiaGpuBuf;

/* Wraps an existing host allocation as a zero-copy Shared-storage MTLBuffer
 * (newBufferWithBytesNoCopy): no data movement, GPU and CPU see the same
 * bytes. `base` MUST be page-aligned (mmap's return value always is;
 * anything else is a caller bug, not a runtime condition -- returns NULL
 * and logs rather than asserting, so callers can probe it, e.g. the
 * --gpu-selftest alignment-violation case). `len` is rounded up to the
 * page size internally (Metal requires a page-multiple length for
 * bytesNoCopy) -- the tail bytes between `len` and the rounded length
 * belong to the underlying mapping (safe for a whole-page mmap like
 * resident.bin's) and must not be treated as valid tensor data by callers.
 * NULL on failure (misaligned base, zero length, no device, length beyond
 * the device's maxBufferLength, or Metal rejecting the wrap). */
SepiaGpuBuf *sepia_gpu_wrap_mmap(void *base, size_t len);

/* Allocates a fresh MTLBuffer: Shared storage (CPU+GPU visible via
 * host_ptr) when gpu_private is 0, Private storage (GPU-only, host_ptr
 * always NULL) when non-zero. NULL on failure. */
SepiaGpuBuf *sepia_gpu_alloc(size_t len, int gpu_private);

/* Releases the underlying MTLBuffer (a no-copy wrap never frees the host
 * memory it wraps -- that stays the caller's, e.g. resident.bin's mmap
 * outlives this call). NULL is a no-op. */
void sepia_gpu_free(SepiaGpuBuf *b);

/* [buffer contents]: the host-visible pointer for Shared buffers (for a
 * wrap_mmap buffer this is literally the wrapped `base`), NULL for
 * Private-storage buffers and NULL b. */
void *sepia_gpu_host_ptr(SepiaGpuBuf *b);

/* [buffer gpuAddress]: the buffer's device VA, for bindless tables (Task
 * 10). 0 for NULL b. */
uint64_t sepia_gpu_gpu_addr(SepiaGpuBuf *b);

/* Local-only, --gpu-selftest support: dispatches the `sepia_touch` kernel
 * (metal/ops.metal) over the first n_floats floats of buf's contents as a
 * single synchronous command buffer (encode, commit, waitUntilCompleted).
 * Not part of the stable dispatch API -- Task 3 introduces the real
 * per-op dispatch machinery (batched encoder, args structs); this exists
 * only so the selftest can prove a wrapped buffer round-trips through
 * actual GPU execution. Returns 1 on success, 0 on failure. */
int sepia_gpu_selftest_touch(SepiaGpuBuf *buf, size_t n_floats);

/* ------------------------------ batched encoding --------------------------- */
/* Mirrors ds4's begin/flush/end batching idiom (ds4_metal.m:6414-6613): one
 * open compute encoder accumulates many dispatches into a single command
 * buffer, and synchronization only happens where the caller asks for it.
 * Every sepia_gpu_<op> dispatch function below REQUIRES an active batch
 * (sepia_gpu_begin having succeeded and not yet been sepia_gpu_end'd) --
 * calling one with no active batch logs and returns 0, it does not silently
 * create a one-off command buffer.
 *
 *   sepia_gpu_begin()  -- opens a new batch. 0 if one is already open, or on
 *                         allocation failure.
 *   sepia_gpu_flush()  -- ends the current encoder, commits the command
 *                         buffer WITHOUT waiting (fire-and-forget), and
 *                         immediately opens a fresh batch so dispatching can
 *                         continue. Use this at a mid-graph point where you
 *                         want work handed to the GPU but don't need results
 *                         back yet.
 *   sepia_gpu_end()    -- ends the current encoder, commits, and BLOCKS on
 *                         waitUntilCompleted, checking the command buffer's
 *                         status. This is the only one of the three that
 *                         synchronizes -- call it once you need the results
 *                         of everything dispatched since the matching
 *                         sepia_gpu_begin(). Does not reopen a batch. */
int sepia_gpu_begin(void);
int sepia_gpu_flush(void);
int sepia_gpu_end(void);

/* --------------------------------- ops -------------------------------- */
/* All of the following require an active batch (sepia_gpu_begin()) and
 * return 1 on success, 0 on failure (invalid args, no active batch, or a
 * missing/failed pipeline). f32 in/out/accum throughout; see metal/ops.metal
 * for the exact per-kernel numerics and src/sepia.c for the CPU oracle each
 * one is compared against by --gpu-compare-tiny.
 *
 * Weight/first-operand buffer first, then the batched/main operand, then
 * the output, then dims/scalars -- kept consistent across every op below
 * (rmsnorm's `w` reused per row is the model for this ordering). */

/* y[r,i] = w[i] * (x[r,i] * rsqrt(mean_i(x[r,i]^2) + eps)), r in [0,rows). */
int sepia_gpu_rmsnorm(SepiaGpuBuf *w, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n, float eps);

/* y[o] = dot(w[o,:], x) for o in [0,out_dim); w is [out_dim,in_dim] row-major
 * and contiguous (row stride == in_dim*sizeof(float)) -- Task 3's scope is
 * plain f32 matvec only; strided/interleaved row layouts are a later task's
 * concern (see the ABI note in docs/superpowers/sdd/task-3-report.md). */
int sepia_gpu_matvec(SepiaGpuBuf *w, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t out_dim, int64_t in_dim);

/* z[i] = silu(g[i]) * u[i], silu(v) = v*sigmoid(v) -- matches silu_f/sigmoid_f
 * in src/sepia.c exactly (no accumulation, so this one is expected near
 * bit-exact rather than merely within tolerance). */
int sepia_gpu_silu_mul(SepiaGpuBuf *g, SepiaGpuBuf *u, SepiaGpuBuf *z, int64_t n);

/* out[i] = a[i] + b[i], elementwise. */
int sepia_gpu_add(SepiaGpuBuf *a, SepiaGpuBuf *b, SepiaGpuBuf *out, int64_t n);

/* Numerically stable softmax, one row at a time: y[r,:] = softmax(x[r,:]),
 * r in [0,rows) -- max-subtract, exp-sum, normalize, matching the structure
 * of attn_forward_chunk's inlined stable softmax (src/sepia.c:961-984). y
 * may alias x (in place). */
int sepia_gpu_softmax(SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n);

/* K-tap depthwise causal conv1d + residual, mirroring sconv_apply
 * (src/sepia.c:791-806) exactly: out[t,c] = in[t,c] + sum_k w[c,k]*window[t+k],
 * window = concat(hist[K-1,C], in[T,C]). Does not write an updated history
 * buffer back out (see the report's ABI notes -- Task 8's concern). */
int sepia_gpu_sconv(SepiaGpuBuf *w, SepiaGpuBuf *hist, SepiaGpuBuf *in, SepiaGpuBuf *out,
                     int64_t C, int64_t K, int64_t T);

/* ------------------------------ quantized matvec --------------------------- */
/* Task 4+: dequant-fused matvec over GGML-quantized weight rows -- the CPU
 * oracle for both is qlinear (dequantize_row per row, double-accumulated
 * dot, src/sepia.c). `ggml_type` is one of the SEPIA_T_* constants from
 * src/quants.h (this header intentionally doesn't include quants.h, to keep
 * the GPU symbol surface decoupled from the quant-format module -- callers
 * already include both, e.g. sepia.c). `w_off` is a byte offset into `w`
 * (e.g. a tensor's offset within one wrapped resident.bin buffer); the
 * weight rows starting there must be row-major [out_dim,in_dim] quantized
 * blocks, contiguous per row (row stride = (in_dim/block_size)*block_bytes
 * for the type), in_dim a multiple of the type's block size -- same
 * contract as the CPU qlinear. Returns 0 (after logging) for an unsupported
 * ggml_type, invalid args, or a dispatch failure -- callers decide whether
 * that is fatal. f32 accumulation throughout (metal/matvec_q.metal has the
 * exact per-type unpack, cross-checked bitwise against src/quants.c's
 * dequant_* functions by --gpu-quants). */
int sepia_gpu_matvec_q(int ggml_type, SepiaGpuBuf *w, size_t w_off, SepiaGpuBuf *x, SepiaGpuBuf *y,
                        int64_t out_dim, int64_t in_dim);

/* Standalone dequantization debug kernel (--gpu-quants): unpacks `n`
 * elements' worth of quantized blocks (n a multiple of the type's block
 * size -- exactly dequantize_row's own contract) starting at byte offset
 * `raw_off` into `raw`, writing `n` floats to `out`. Shares its per-type
 * unpack code with sepia_gpu_matvec_q (metal/matvec_q.metal) so the two
 * cannot drift apart -- this is what lets --gpu-quants gate the matvec
 * kernel's unpack correctness bitwise via the standalone kernel, and only
 * needs a relative-tolerance check on the dot-product/reduction itself.
 * Returns 0 (after logging) for an unsupported ggml_type, invalid args, or
 * a dispatch failure. */
int sepia_gpu_dequant_rows(int ggml_type, SepiaGpuBuf *raw, size_t raw_off, SepiaGpuBuf *out, int64_t n);

#endif
