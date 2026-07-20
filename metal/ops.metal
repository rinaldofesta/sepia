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
