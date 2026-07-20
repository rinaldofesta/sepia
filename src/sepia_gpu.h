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

/* --------------------------- banded attention (Task 7) --------------------- */
/* The from-scratch banded flash-attention kernel pair (metal/banded_attn.metal
 * -- no Metal reference for this op exists anywhere; llama.cpp's PR is
 * CPU+CUDA only). Mirrors attn_forward_chunk's per-(t,h) inner loop
 * (src/sepia.c) exactly: dense rel_logits recompute, content dot *
 * inv_head_dim, band-gated relative-position bias * tau, stable online
 * softmax, weighted V sum. f32 accumulators throughout (Global Constraints);
 * see .superpowers/sdd/task-7-report.md for the parallelization-shape and
 * tau-factorization design writeup.
 *
 * Buffer layouts (row-major, contiguous within a row):
 *   r_vec:      [T, H, d_rel]        -- wr's raw output, no norm applied
 *   rel_proj:   [d_rel, rel_extent]  -- shared across (t,h) within a layer
 *   rel_logits: [T, H, rel_extent]   -- kernel 1's output, kernel 2's input
 *   q:          [T, H, Dh]           -- ALREADY tau-scaled by the caller for
 *                                        global (non-sliding) layers (see the
 *                                        tau paragraph below); pass the raw
 *                                        q unchanged for sliding layers
 *                                        (tau=1 there is a no-op scale).
 *   k, v:       [cap, Hkv, Dh]       -- the FULL kv cache, cap >= q_pos_base+T;
 *                                        windowing is arithmetic (kv_lo/kv_hi
 *                                        per query token), never a separate
 *                                        storage slice.
 *   attn_out:   [T, H, Dh]           -- pre-wo concatenated output
 *
 * Tau factorization (the design choice, matching attn_forward_chunk's own
 * split rather than folding tau into rel_logits): tau multiplies BOTH q and
 * the rel-position bias in the CPU oracle (src/sepia.c: `q_scaled[d] =
 * q_vec[d] * tau` and `bias = rel_logits[distance] * tau`). This API mirrors
 * that split directly -- the CALLER pre-scales q by tau (a single
 * elementwise multiply, done once per token before upload; Task 9's real
 * integration is expected to fold this into the per-head norm kernel's
 * epilogue) and passes tau again to sepia_gpu_banded_attn, which applies it
 * only to the bias term. The alternative (pre-scaling rel_logits by tau
 * instead of q) was rejected: it would need a second, per-(t,h) tau-scaled
 * copy of rel_logits (rel_logits is otherwise position-independent within a
 * layer) and would still require gating the "distance < rel_extent" test
 * somewhere -- reproducing the CPU's own factorization is simpler and keeps
 * rel_logits itself tau-free (bit-comparable across sliding/global layers
 * and across query tokens with different tau, useful for the "compare
 * rel_logits first" debug protocol). Pass tau=1.0 uniformly for sliding
 * (local) layers, where log-scaling never applies.
 */

/* rel_logits[t,h,r] = sum_d r_vec[t,h,d] * rel_proj[d,r], r in [0,rel_extent).
 * One threadgroup per (t,h) (T*H total), f32 accum throughout -- the CPU
 * oracle's own precision pin (src/sepia.c). */
int sepia_gpu_rel_project(SepiaGpuBuf *r_vec, SepiaGpuBuf *rel_proj, SepiaGpuBuf *rel_logits,
                           int64_t T, int64_t H, int64_t d_rel, int64_t rel_extent);

/* One threadgroup per (t,h) (T*H total). Per query token t (absolute
 * position q_pos_base+t), iterates kv in [kv_lo[t],kv_hi[t]] (host-computed
 * per-token bounds into the FULL k/v cache -- sliding: kv_lo=max(0,q_pos-
 * window+1); global: kv_lo=0; kv_hi=q_pos always, matching attn_forward_
 * chunk) with GQA head mapping hk = h/(H/Hkv). f32 online-softmax
 * accumulation (running max/sum/V-accumulator); see the design doc above
 * and metal/banded_attn.metal's header comment for the simdgroup-then-
 * threadgroup reduction shape. kv_lo/kv_hi/tau are plain host arrays of
 * length T (uploaded into transient GPU buffers internally -- callers do
 * not need to manage their lifetime beyond this call); every other
 * dimension is a scalar shared across the whole (T,H) dispatch. Dh must be
 * <=128 (metal/banded_attn.metal's SEPIA_ATTN_MAX_DH -- both of this
 * model's real per-layer-type geometries use Dh=128); returns 0 (after
 * logging) otherwise. */
int sepia_gpu_banded_attn(SepiaGpuBuf *q, SepiaGpuBuf *k, SepiaGpuBuf *v, SepiaGpuBuf *rel_logits,
                           SepiaGpuBuf *attn_out,
                           const int64_t *kv_lo, const int64_t *kv_hi, const float *tau,
                           int64_t T, int64_t H, int64_t Hkv, int64_t Dh, int64_t rel_extent,
                           int64_t q_pos_base, int64_t kv_dim, float inv_head_dim);

/* ------------------------- real-model resident path (Task 9) --------------- */
/* The real model's F32 resident tensors (norms, rel_proj, router weight,
 * sconv weights) live at arbitrary byte offsets inside the ONE wrapped
 * gpu_res_buf (resident.bin, Task 2's sepia_gpu_wrap_mmap) rather than in
 * their own freshly-uploaded buffer the way the tiny-model path (Task 8)
 * uploads a fresh copy per call. These `_off` twins of the existing f32 ops
 * are identical in every way except the WEIGHT operand is bound at a
 * caller-supplied byte offset into an existing buffer instead of always 0 --
 * same kernels, same numerics, just a different `offset:` argument at the
 * Metal buffer-binding call site (metal/ops.metal and metal/banded_attn.metal
 * need no new kernel code for these). Existing (non-`_off`) call sites and
 * gates (gputest/gpucompare/gpuattn/gpuquants, the tiny --metal forward
 * path) are untouched -- these are purely additive. The `x`/`in`/`r_vec`
 * operand keeps its existing always-offset-0 contract in every case; Task
 * 9's real-model graph uses sepia_gpu_copy (below) to materialize a
 * single-row, offset-0 view whenever it needs to feed one token's row out
 * of a multi-row buffer. */
int sepia_gpu_matvec_off(SepiaGpuBuf *w, size_t w_off, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t out_dim, int64_t in_dim);
int sepia_gpu_rmsnorm_off(SepiaGpuBuf *w, size_t w_off, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n, float eps);
int sepia_gpu_sconv_off(SepiaGpuBuf *w, size_t w_off, SepiaGpuBuf *hist, SepiaGpuBuf *in, SepiaGpuBuf *out,
                         int64_t C, int64_t K, int64_t T);
int sepia_gpu_rel_project_off(SepiaGpuBuf *r_vec, SepiaGpuBuf *rel_proj, size_t rel_proj_off, SepiaGpuBuf *rel_logits,
                              int64_t T, int64_t H, int64_t d_rel, int64_t rel_extent);

/* out[i] = in[i] for i in [0,n) -- a plain elementwise GPU->GPU copy with
 * INDEPENDENT byte offsets on both sides (src_off/dst_off), used two ways in
 * the real-model graph: (a) materializing one token row of a [T,dim]
 * multi-row buffer as a standalone offset-0 operand for matvec/matvec_q
 * (whose x/y operands are always bound at offset 0), and (b) cache-write
 * (moving a layer's freshly computed K/V rows into the persistent per-layer
 * KV-cache buffer at the current position's byte offset) -- both purely
 * GPU-side data movement, no host round trip, no floating-point operation. */
int sepia_gpu_copy(SepiaGpuBuf *src, size_t src_off, SepiaGpuBuf *dst, size_t dst_off, int64_t n);

/* y[i] = x[i] * scale -- elementwise scalar multiply (e.g. the dense-MLP
 * per-layer global_scale, sec.9, applied to the FFN output; the scalar is
 * known before any router readback, so this always runs fully on GPU with
 * no host dependency, unlike the MoE branch's per-shared-expert gamma,
 * which IS readback-dependent and stays a host-side scalar accumulate,
 * see real_mlp_moe_forward_gpu's design note in src/sepia.c). */
int sepia_gpu_scale(float scale, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t n);

/* GPU-side replacement for Task 8's host-side sconv history roll (design
 * choice (b), sconv-history gap): updates `hist` [Km1,C] in place from the
 * OLD hist plus this call's `in` [T,C], reproducing sconv_apply's own tail
 * update (src/sepia.c) exactly -- pure data movement, no floating-point
 * operation, so it needs no tolerance and no host round trip (unlike Task
 * 8's tiny path, which could afford a host round trip per op at tiny scale
 * but Task 9's real model, at 66 layers x 4 sconv sites/layer, cannot).
 * T >= K-1: new hist = last K-1 rows of `in`. T < K-1: new hist = shift-keep
 * old hist rows [T,K-1) then append all of `in`. K-1 must be <= 15 (generous
 * bound for this model's K=4; dies loudly, not silently, if a future config
 * exceeds it -- see metal/ops.metal's SEPIA_SCONV_MAX_KM1). */
int sepia_gpu_sconv_hist_roll(SepiaGpuBuf *hist, SepiaGpuBuf *in, int64_t C, int64_t K, int64_t T);

#endif
