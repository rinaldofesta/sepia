// metal/ops.metal -- first real compute kernels (Task 3): rmsnorm, matvec,
// silu*mul, add, softmax, sconv. f32 in/out/accum throughout; the CPU oracle
// (src/sepia.c) accumulates reductions in double, these kernels in float --
// that gap is exactly what --gpu-compare-tiny's 2e-4 relative tolerance
// covers (docs/superpowers/plans/2026-07-20-p2-metal-streaming.md, Global
// Constraints (b)).
#include <metal_stdlib>
using namespace metal;

// A single trivial kernel so the runtime-compiled library (sepia_gpu_init,
// src/sepia_metal.m) always has something non-empty to build a pipeline
// from; also --gpu-selftest's end-to-end zero-copy proof (Task 2).
kernel void sepia_touch(device float *x [[buffer(0)]],
                         uint i [[thread_position_in_grid]]) {
    x[i] += 0.0f;
}

// ---------------------------------------------------------------------------
// Threadgroup reduction helpers shared by the row-per-threadgroup kernels
// below (rmsnorm, matvec, softmax). Ports ds4's kernel_rms_norm_fuse_impl
// idiom (norm.metal): each simdgroup partially reduces via simd_sum/simd_max,
// lane 0 of every simdgroup stashes its partial sum into threadgroup memory,
// then EVERY thread re-reduces that (<=32-entry, one slot per simdgroup)
// array with a second simd op -- so every thread in the threadgroup ends up
// holding the identical, fully-reduced value (broadcast), no separate
// "read result from thread 0" step needed. Callers must dispatch with a
// threadgroup width that is a multiple of the SIMD width (32) and at most
// 32 simdgroups (<=1024 threads), and provide >=32 floats of threadgroup
// memory at index 0.
static inline float sepia_tg_reduce_sum(float val, threadgroup float *shmem, ushort tiisg, ushort sgitg) {
    if (sgitg == 0) shmem[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    val = simd_sum(val);
    if (tiisg == 0) shmem[sgitg] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    val = shmem[tiisg];
    return simd_sum(val);
}

static inline float sepia_tg_reduce_max(float val, threadgroup float *shmem, ushort tiisg, ushort sgitg) {
    if (sgitg == 0) shmem[tiisg] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    val = simd_max(val);
    if (tiisg == 0) shmem[sgitg] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    val = shmem[tiisg];
    return simd_max(val);
}

// ---------------------------------------------------------------------------
// sepia_rmsnorm_f32: weight-fused RMSNorm, one threadgroup per row.
// out[i] = w[i] * (x[i] * rsqrt(mean(x^2) + eps)) -- same order of
// operations as rmsnorm() in src/sepia.c:710 (x*inv first, then *w); mean
// is accumulated in f32 here vs the CPU's double accumulator.
struct sepia_args_rmsnorm {
    int32_t ne0; // row length (n)
    float   eps;
};

kernel void sepia_rmsnorm_f32(
        constant sepia_args_rmsnorm &args [[buffer(0)]],
        device const float *w [[buffer(1)]],  // [ne0], shared across all rows
        device const float *x [[buffer(2)]],  // [rows, ne0]
        device       float *y [[buffer(3)]],  // [rows, ne0]; y may alias x
        threadgroup  float *shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t n = args.ne0;
    const int32_t row = (int32_t)tgpig.x;
    device const float *xr = x + (int64_t)row * n;
    device       float *yr = y + (int64_t)row * n;

    float sumf = 0.0f;
    for (int32_t i = (int32_t)tpitg.x; i < n; i += (int32_t)ntg.x) {
        float v = xr[i];
        sumf += v * v;
    }
    sumf = sepia_tg_reduce_sum(sumf, shmem, tiisg, sgitg);

    const float mean  = sumf / (float)n;
    const float scale = 1.0f / sqrt(mean + args.eps);
    for (int32_t i = (int32_t)tpitg.x; i < n; i += (int32_t)ntg.x) {
        yr[i] = w[i] * (xr[i] * scale);
    }
}

// ---------------------------------------------------------------------------
// sepia_matvec_f32: y[out] = dot(w[out,:], x), one threadgroup per output
// row. f32 accumulation (CPU's linear/qlinear/dotf accumulate in double --
// same tolerance-covered drift as rmsnorm). Task 3's scope is plain
// contiguous-row f32 matvec only (Global Constraints); Task 4+'s dequant-
// fused/interleaved-row kernels are a different family, not this one made
// generic.
struct sepia_args_matvec {
    int32_t  ne0; // in_dim (row length)
    uint64_t nb1; // W row stride in bytes (contiguous rows: ne0*sizeof(float))
};

kernel void sepia_matvec_f32(
        constant sepia_args_matvec &args [[buffer(0)]],
        device const float *w [[buffer(1)]],  // [out_dim, in_dim], row stride nb1
        device const float *x [[buffer(2)]],  // [in_dim]
        device       float *y [[buffer(3)]],  // [out_dim]
        threadgroup  float *shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t n = args.ne0;
    const int32_t row = (int32_t)tgpig.x;
    device const float *wr = (device const float *)((device const char *)w + (uint64_t)row * args.nb1);

    float sumf = 0.0f;
    for (int32_t i = (int32_t)tpitg.x; i < n; i += (int32_t)ntg.x) sumf += wr[i] * x[i];
    sumf = sepia_tg_reduce_sum(sumf, shmem, tiisg, sgitg);

    if (tpitg.x == 0) y[row] = sumf;
}

// ---------------------------------------------------------------------------
// sepia_silu_mul_f32: z[i] = silu(g[i]) * u[i], silu(v) = v*sigmoid(v) --
// matches silu_f/sigmoid_f in src/sepia.c exactly. No accumulation, so this
// one is expected near bit-exact rather than merely within tolerance.
kernel void sepia_silu_mul_f32(
        device const float *g [[buffer(0)]],
        device const float *u [[buffer(1)]],
        device       float *z [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    float gv = g[i];
    float sig = 1.0f / (1.0f + exp(-gv));
    z[i] = (gv * sig) * u[i];
}

// ---------------------------------------------------------------------------
// sepia_add_f32: out[i] = a[i] + b[i], elementwise residual add.
kernel void sepia_add_f32(
        device const float *a [[buffer(0)]],
        device const float *b [[buffer(1)]],
        device       float *out [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// sepia_softmax_f32: numerically stable softmax, one threadgroup per row --
// max-subtract, exp-sum, normalize, matching the structure of
// attn_forward_chunk's inlined stable softmax (src/sepia.c, attn_forward_chunk).
struct sepia_args_softmax {
    int32_t ne0; // row length (n)
};

kernel void sepia_softmax_f32(
        constant sepia_args_softmax &args [[buffer(0)]],
        device const float *x [[buffer(1)]], // [rows, ne0]
        device       float *y [[buffer(2)]], // [rows, ne0]; y may alias x
        threadgroup  float *shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t n = args.ne0;
    const int32_t row = (int32_t)tgpig.x;
    device const float *xr = x + (int64_t)row * n;
    device       float *yr = y + (int64_t)row * n;

    float lmax = -INFINITY;
    for (int32_t i = (int32_t)tpitg.x; i < n; i += (int32_t)ntg.x) lmax = max(lmax, xr[i]);
    float max_val = sepia_tg_reduce_max(lmax, shmem, tiisg, sgitg);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lsum = 0.0f;
    for (int32_t i = (int32_t)tpitg.x; i < n; i += (int32_t)ntg.x) {
        float e = exp(xr[i] - max_val);
        yr[i] = e;
        lsum += e;
    }
    float sum = sepia_tg_reduce_sum(lsum, shmem, tiisg, sgitg);

    const float inv_sum = 1.0f / sum;
    for (int32_t i = (int32_t)tpitg.x; i < n; i += (int32_t)ntg.x) yr[i] *= inv_sum;
}

// ---------------------------------------------------------------------------
// sepia_sconv_f32: K-tap depthwise causal conv1d + residual, mirroring
// sconv_apply (src/sepia.c) exactly. One thread per (t,c); no sequential
// dependency (unlike sconv_apply's history recurrence) because the whole
// causal window [hist || in] for this call is already resident, so every
// output position can be computed independently:
//   out[t,c] = in[t,c] + sum_k w[c,k] * window[t+k,c]
//   window = concat(hist[K-1,C], in[T,C])
// Does not write an updated history buffer (Task 3's harness runs a single
// from-scratch chunk; incremental-decode history chaining is Task 8's
// concern, see the sepia_gpu_sconv header doc in src/sepia_gpu.h).
struct sepia_args_sconv {
    int32_t C;
    int32_t K;
    int32_t T;
};

kernel void sepia_sconv_f32(
        constant sepia_args_sconv &args [[buffer(0)]],
        device const float *w    [[buffer(1)]], // [C, K]
        device const float *hist [[buffer(2)]], // [K-1, C]
        device const float *in   [[buffer(3)]], // [T, C]
        device       float *out  [[buffer(4)]], // [T, C]
        uint2 gid [[thread_position_in_grid]]) {
    const int32_t c = (int32_t)gid.x;
    const int32_t t = (int32_t)gid.y;
    const int32_t C = args.C, K = args.K, T = args.T;
    if (c >= C || t >= T) return;
    const int32_t Km1 = K - 1;

    float acc = 0.0f;
    for (int32_t k = 0; k < K; k++) {
        int32_t idx = t + k;
        float val = (idx < Km1) ? hist[idx * C + c] : in[(idx - Km1) * C + c];
        acc += w[c * K + k] * val;
    }
    out[t * C + c] = in[t * C + c] + acc;
}
