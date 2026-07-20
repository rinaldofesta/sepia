// metal/banded_attn.metal -- Task 7: banded flash-attention, from scratch.
// No Metal reference exists anywhere for this op (llama.cpp's PR is
// CPU+CUDA only); the reference semantics are attn_forward_chunk's
// per-(t,h) inner loop (src/sepia.c) plus the extracted ggml-cpu spec (see
// docs/superpowers/plans/2026-07-20-p2-metal-streaming.md, Task 7 and its
// "Banded attention reference spec" subsection). f32 accumulators
// throughout (Global Constraints -- fp16 VKQ accumulators overflow with
// this model's V magnitudes, llama.cpp fattn-mma-f16.cuh:695-706); all
// kv/position arithmetic is 64-bit (`long`) -- load-bearing at the
// 1M-token context this model targets.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// sepia_rel_project_f32: rel_logits[t,h,r] = sum_d r_vec[t,h,d] * rel_proj[d,r]
// -- the CPU oracle's own precision pin (src/sepia.c, attn_forward_chunk:
// "dense rel_logits recompute", GGML_PREC_F32_PEDANTIC in the ggml spec):
// f32 in/out/accum throughout. One threadgroup per (t,h) (grid = H x T),
// threads spread over the rel_extent band -- each thread's r values are
// independent dot products (no cross-thread reduction needed, unlike
// rmsnorm/matvec's row-reduce shape in ops.metal).
struct sepia_args_rel_project {
    int32_t H;
    int32_t d_rel;
    int32_t rel_extent;
};

kernel void sepia_rel_project_f32(
        constant sepia_args_rel_project &args [[buffer(0)]],
        device const float *r_vec      [[buffer(1)]], // [T,H,d_rel]
        device const float *rel_proj   [[buffer(2)]], // [d_rel,rel_extent], shared across (t,h)
        device       float *rel_logits [[buffer(3)]], // [T,H,rel_extent]
        uint3   tgpig [[threadgroup_position_in_grid]], // tgpig.x = h, tgpig.y = t
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int32_t H = args.H, d_rel = args.d_rel, rel_extent = args.rel_extent;
    const int64_t th = (int64_t)tgpig.y * (int64_t)H + (int64_t)tgpig.x; // flattened (t,h)

    device const float *rv = r_vec + th * (int64_t)d_rel;
    device       float *rl = rel_logits + th * (int64_t)rel_extent;

    for (int32_t r = (int32_t)tpitg.x; r < rel_extent; r += (int32_t)ntg.x) {
        float acc = 0.0f;
        for (int32_t d = 0; d < d_rel; d++) acc += rv[d] * rel_proj[(int64_t)d * (int64_t)rel_extent + r];
        rl[r] = acc;
    }
}

// ---------------------------------------------------------------------------
// sepia_banded_attn_f32: banded flash-attention with f32 online-softmax
// accumulation, one threadgroup per (t,h) (grid = H x T). Mirrors
// attn_forward_chunk's per-kv inner loop exactly: content = dot(q,k) *
// inv_head_dim; dist = q_pos - kv (>=0 by causality); bias = dist <
// rel_extent ? rel_logits[dist] * tau : 0 (hard zero outside the band, no
// edge-clamp -- the ggml spec's own wording); stable online softmax;
// weighted V sum.
//
// Tau/rel_logits factorization (see src/sepia_gpu.h and
// .superpowers/sdd/task-7-report.md for the full design-decision writeup):
// `q` here is ALREADY tau-scaled by the caller (host pre-applies tau to q,
// matching src/sepia.c's `q_scaled[d] = q_vec[d] * tau`), and `tau` is
// passed again here to scale ONLY the bias term (`bias = rel_logits[dist] *
// tau`) -- this is the CPU's own factorization, reproduced directly rather
// than pre-baking tau into rel_logits (which would require a second,
// per-(t,h) tau-scaled copy of rel_logits and duplicate the "distance <
// rel_extent" gating logic in two places). Pass tau=1.0 uniformly for
// sliding (local) layers, where log-scaling never applies.
//
// Parallelization shape (the "standard online-softmax-parallel" shape):
//   1. kv-strided per-thread: each thread scans a strided subset of
//      [kv_lo,kv_hi], maintaining its own running (m, l, acc[Dh]) via the
//      textbook online-softmax update.
//   2. simdgroup-reduced: the 32 lanes of a simdgroup merge their partials
//      via simd_max/simd_sum (an elementwise-rescale-then-sum identity: one
//      simd_sum per accumulator dimension, mirroring the accumulator-merge
//      shape of ds4's dsv4_misc.metal:577 -- none of its DSA control flow).
//   3. final threadgroup reduction: lane 0 of each simdgroup stashes its
//      simdgroup's combined partial into threadgroup memory; thread 0
//      sequentially folds the (small, <=32) simdgroup partials into the
//      final normalized output.
//
// A simdgroup that gets ZERO kv indices (n_kv smaller than the chosen
// threadgroup width) has every lane's running max stuck at -INFINITY; the
// `m_sg == -INFINITY` guard below is required so this doesn't compute
// exp(-inf - (-inf)) = NaN when merging that simdgroup's (empty) partial --
// with the guard, an empty simdgroup contributes a clean all-zero partial
// instead, which the final fold correctly treats as a no-op. Thread 0 always
// has at least one kv to process (kv_lo <= kv_hi always holds, by
// causality), so the fold's own running max starts finite and stays finite
// -- only the simdgroup-merge stage needs the empty-partition guard.
#define SEPIA_ATTN_MAX_DH 128

struct sepia_args_banded_attn {
    long  T;
    long  H;
    long  Hkv;
    long  Dh;
    long  rel_extent;
    long  q_pos_base;
    long  kv_dim;       // k/v row stride in elements (Hkv*Dh)
    float inv_head_dim;
};

kernel void sepia_banded_attn_f32(
        constant sepia_args_banded_attn &args [[buffer(0)]],
        device const float *q          [[buffer(1)]], // [T,H,Dh], already tau-scaled
        device const float *k          [[buffer(2)]], // [cap,Hkv,Dh], full cache (cap >= q_pos_base+T)
        device const float *v          [[buffer(3)]], // [cap,Hkv,Dh]
        device const float *rel_logits [[buffer(4)]], // [T,H,rel_extent]
        device const long  *kv_lo      [[buffer(5)]], // [T]
        device const long  *kv_hi      [[buffer(6)]], // [T], inclusive
        device const float *tau        [[buffer(7)]], // [T]
        device       float *attn_out   [[buffer(8)]], // [T,H,Dh]
        threadgroup  float *shmem_m    [[threadgroup(0)]], // [nsimdgroups]
        threadgroup  float *shmem_l    [[threadgroup(1)]], // [nsimdgroups]
        threadgroup  float *shmem_acc  [[threadgroup(2)]], // [nsimdgroups*Dh]
        uint3   tgpig [[threadgroup_position_in_grid]], // tgpig.x = h, tgpig.y = t
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const long H = args.H, Hkv = args.Hkv, Dh = args.Dh, rel_extent = args.rel_extent;
    const long t = (long)tgpig.y, h = (long)tgpig.x;
    const long group = H / Hkv;
    const long hk = h / group;
    const long q_pos = args.q_pos_base + t;
    const long kv_lo_t = kv_lo[t];
    const long kv_hi_t = kv_hi[t];
    const float tau_t = tau[t];

    device const float *q_vec = q + (t * H + h) * Dh;
    device const float *rl_row = rel_logits + (t * H + h) * rel_extent;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[SEPIA_ATTN_MAX_DH];
    for (long d = 0; d < Dh; d++) acc[d] = 0.0f;

    for (long kv = kv_lo_t + (long)tpitg.x; kv <= kv_hi_t; kv += (long)ntg.x) {
        device const float *k_vec = k + (kv * Hkv + hk) * Dh;
        device const float *v_vec = v + (kv * Hkv + hk) * Dh;

        float content = 0.0f;
        for (long d = 0; d < Dh; d++) content += q_vec[d] * k_vec[d];
        content *= args.inv_head_dim;

        long dist = q_pos - kv;
        float bias = 0.0f;
        if (dist < rel_extent) bias = rl_row[dist] * tau_t;

        float s = content + bias;
        float new_m = max(m, s);
        float corr = exp(m - new_m); // m may be -inf, new_m always finite here -> exp(-inf)=0, fine
        float p = exp(s - new_m);
        l = l * corr + p;
        for (long d = 0; d < Dh; d++) acc[d] = acc[d] * corr + p * v_vec[d];
        m = new_m;
    }

    // --- simdgroup-level merge ---
    float m_sg = simd_max(m);
    float corr = (m_sg == -INFINITY) ? 0.0f : exp(m - m_sg); // guard: see the header doc above
    float l_sg = simd_sum(l * corr);
    for (long d = 0; d < Dh; d++) acc[d] = simd_sum(acc[d] * corr);
    // acc[d] now holds the simdgroup-summed value in every lane (simd_sum broadcasts,
    // same convention as sepia_tg_reduce_sum in ops.metal).

    if (tiisg == 0) {
        shmem_m[sgitg] = m_sg;
        shmem_l[sgitg] = l_sg;
        for (long d = 0; d < Dh; d++) shmem_acc[(long)sgitg * Dh + d] = acc[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- final threadgroup reduction (thread 0 only) ---
    if (tpitg.x == 0) {
        const int32_t nsg = (int32_t)(ntg.x / 32);
        float m_f = shmem_m[0]; // simdgroup 0 always contains tpitg.x==0, which always has
        float l_f = shmem_l[0]; // >=1 valid kv (kv_lo<=kv_hi always) -- m_f starts finite and
        float acc_f[SEPIA_ATTN_MAX_DH]; // stays finite, so no empty-partition guard needed here.
        for (long d = 0; d < Dh; d++) acc_f[d] = shmem_acc[d];
        for (int32_t sg = 1; sg < nsg; sg++) {
            float m2 = shmem_m[sg], l2 = shmem_l[sg];
            float new_m = max(m_f, m2);
            float c1 = exp(m_f - new_m), c2 = exp(m2 - new_m);
            l_f = l_f * c1 + l2 * c2;
            for (long d = 0; d < Dh; d++) acc_f[d] = acc_f[d] * c1 + shmem_acc[(long)sg * Dh + d] * c2;
            m_f = new_m;
        }
        device float *out_row = attn_out + (t * H + h) * Dh;
        float inv_l = 1.0f / l_f;
        for (long d = 0; d < Dh; d++) out_row[d] = acc_f[d] * inv_l;
    }
}
