// metal/matvec_q.metal -- dequant-fused matvec kernels (Task 4): Q8_0, Q4_K.
// Row-per-threadgroup, simdgroup-reduced, f32-accumulated dot over quantized
// rows (y[out] = W_q[out,:] . x), the same shape as metal/ops.metal's
// sepia_matvec_f32 (Task 3) with the row unpacked from GGML quant blocks
// instead of read as plain floats. Row-major [out_dim,in_dim], in_dim a
// multiple of the type's block size -- identical contract to the CPU
// oracle's qlinear/qrow (src/sepia.c).
//
// Block layouts mirror src/quants.c bit-for-bit (ground truth, cross-checked
// against gguf-py fixtures): Q8_0 is f16 d + 32 x int8 qs (34B); Q4_K is a
// 144B superblock (f16 d + f16 dmin + 12B packed 6-bit scale/min pairs +
// 128B 4-bit qs), unpacked via get_scale_min_k4's nibble/byte packing --
// ported from ds4's block_q4_K (metal/moe.metal:101-106, MIT, see NOTICE),
// same field order (d, dmin, scales[12], qs[128]) as SEPIA's own layout.
//
// The unpack functions below (sepia_unpack_q8_0_block / _q4_k_block) are the
// SINGLE source of truth for both the matvec kernels and the standalone
// debug/fixture-gate kernels (sepia_dequant_rows_q8_0 / _q4_k) -- the debug
// kernels call the exact same function and copy its thread-local output to
// device memory, so the two paths cannot drift apart. MSL's half->float
// widening (used for the `d`/`dmin` scale fields below) is an exact IEEE-754
// conversion, bit-identical to src/quants.c's quants_f16_to_f32 software
// path (same real value from the same 16-bit pattern, for every class:
// zero, subnormal, normal, inf/nan) -- this is what lets --gpu-quants gate
// dequantization BITWISE rather than by tolerance (P2 plan, Global
// Constraints (b)).
#include <metal_stdlib>
using namespace metal;

// sepia_tg_reduce_sum is defined in metal/ops.metal (Task 3). Both files are
// concatenated into one translation unit by sepia_gpu_init (sepia_gpu_load_
// source sorts entries alphabetically: "matvec_q.metal" < "ops.metal", so
// THIS file's use of the helper needs a prototype here -- the concrete
// definition appearing later in the concatenated source satisfies it, same
// as any forward-declared C function in one translation unit). Kept as a
// forward declaration (not a duplicate body) so there is exactly one
// implementation to maintain.
static inline float sepia_tg_reduce_sum(float val, threadgroup float *shmem, ushort tiisg, ushort sgitg);

// ---------------------------------------------------------------------------
// Block layouts (byte-identical to src/quants.c; QK8_0=32, QK_K=256).

struct sepia_block_q8_0 {
    half d;
    char qs[32];
};

struct sepia_block_q4_k {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[128];
};

// ---------------------------------------------------------------------------
// Q8_0 unpack: bitwise port of dequant_q8_0 (src/quants.c) -- no cross-block
// state, one thread can unpack any block independently.
static inline void sepia_unpack_q8_0_block(device const sepia_block_q8_0 *blk, thread float *out /* [32] */) {
    const float d = (float)blk->d;
    for (int j = 0; j < 32; j++) out[j] = d * (float)blk->qs[j];
}

// get_scale_min_k4 (src/quants.c): unpacks one of the 8 (scale,min) 6-bit
// pairs packed into the superblock's 12-byte `scales` array. Pure integer
// bit ops -- deterministic in any language, ported verbatim.
static inline void sepia_get_scale_min_k4(int j, device const uchar *q, thread uchar &d, thread uchar &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uchar)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Q4_K unpack: bitwise port of dequant_q4_k (src/quants.c) -- same order of
// operations (d1/m1 then d2/m2 per 64-element sub-block, low nibble then
// high nibble) so the float arithmetic matches term-for-term.
static inline void sepia_unpack_q4_k_block(device const sepia_block_q4_k *blk, thread float *out /* [256] */) {
    const float d = (float)blk->d;
    const float dmin = (float)blk->dmin;
    device const uchar *scales = blk->scales;
    device const uchar *q = blk->qs;
    int is = 0;
    thread float *dst = out;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        sepia_get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * (float)sc, m1 = dmin * (float)m;
        sepia_get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * (float)sc, m2 = dmin * (float)m;
        for (int l = 0; l < 32; l++) dst[l]      = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) dst[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
        dst += 64; q += 32; is += 2;
        (void)j;
    }
}

// ---------------------------------------------------------------------------
// Matvec args: mirrors sepia_args_matvec (Task 3, metal/ops.metal) field-
// for-field -- ne0=in_dim, nb1=W row stride in bytes. Distinct type (not
// reused) because this file compiles as part of the same library but the
// row stride here is quant-block-derived, not sizeof(float)*ne0.
struct sepia_args_matvec_q {
    int32_t  ne0; // in_dim (row length in elements)
    uint64_t nb1; // W row stride in bytes: (in_dim/block_size) * block_bytes
};

// ---------------------------------------------------------------------------
// sepia_matvec_q8_0: one threadgroup per output row. Each thread strides
// over a subset of the row's Q8_0 blocks, unpacks each to a thread-local
// float[32], dots it against the matching slice of x, and accumulates into
// a per-thread f32 sum; sepia_tg_reduce_sum combines across the threadgroup
// (same reduction shape as sepia_matvec_f32).
kernel void sepia_matvec_q8_0(
        constant sepia_args_matvec_q &args [[buffer(0)]],
        device const char  *w [[buffer(1)]],  // [out_dim,in_dim] row-major q8_0 blocks, row stride nb1
        device const float *x [[buffer(2)]],  // [in_dim]
        device       float *y [[buffer(3)]],  // [out_dim]
        threadgroup  float *shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t nb = args.ne0 / 32;
    const int32_t row = (int32_t)tgpig.x;
    device const sepia_block_q8_0 *wr =
        (device const sepia_block_q8_0 *)((device const char *)w + (uint64_t)row * args.nb1);

    float sumf = 0.0f;
    float unpacked[32];
    for (int32_t b = (int32_t)tpitg.x; b < nb; b += (int32_t)ntg.x) {
        sepia_unpack_q8_0_block(wr + b, unpacked);
        device const float *xb = x + b * 32;
        for (int32_t j = 0; j < 32; j++) sumf += unpacked[j] * xb[j];
    }
    sumf = sepia_tg_reduce_sum(sumf, shmem, tiisg, sgitg);

    if (tpitg.x == 0) y[row] = sumf;
}

// ---------------------------------------------------------------------------
// sepia_matvec_q4_k: same shape as sepia_matvec_q8_0, one thread per
// superblock (256 elements) instead of per 32-element block.
kernel void sepia_matvec_q4_k(
        constant sepia_args_matvec_q &args [[buffer(0)]],
        device const char  *w [[buffer(1)]],  // [out_dim,in_dim] row-major q4_k superblocks, row stride nb1
        device const float *x [[buffer(2)]],  // [in_dim]
        device       float *y [[buffer(3)]],  // [out_dim]
        threadgroup  float *shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const int32_t nb = args.ne0 / 256;
    const int32_t row = (int32_t)tgpig.x;
    device const sepia_block_q4_k *wr =
        (device const sepia_block_q4_k *)((device const char *)w + (uint64_t)row * args.nb1);

    float sumf = 0.0f;
    float unpacked[256];
    for (int32_t b = (int32_t)tpitg.x; b < nb; b += (int32_t)ntg.x) {
        sepia_unpack_q4_k_block(wr + b, unpacked);
        device const float *xb = x + b * 256;
        for (int32_t j = 0; j < 256; j++) sumf += unpacked[j] * xb[j];
    }
    sumf = sepia_tg_reduce_sum(sumf, shmem, tiisg, sgitg);

    if (tpitg.x == 0) y[row] = sumf;
}

// ---------------------------------------------------------------------------
// Standalone debug/fixture-gate kernels (--gpu-quants): unpack N contiguous
// blocks to N*block_elems f32, no dot product. One thread per block (blocks
// are independent -- no cross-block state in either format), calling the
// exact same unpack function the matvec kernels above use, then a plain
// thread-to-device copy (no computation, so it cannot introduce drift).
kernel void sepia_dequant_rows_q8_0(
        device const char  *raw [[buffer(0)]], // [nb] contiguous q8_0 blocks (34B each)
        device       float *out [[buffer(1)]], // [nb*32]
        uint b [[thread_position_in_grid]]) {
    device const sepia_block_q8_0 *blk =
        (device const sepia_block_q8_0 *)(raw + (uint64_t)b * sizeof(sepia_block_q8_0));
    float unpacked[32];
    sepia_unpack_q8_0_block(blk, unpacked);
    device float *dst = out + (uint64_t)b * 32;
    for (int j = 0; j < 32; j++) dst[j] = unpacked[j];
}

kernel void sepia_dequant_rows_q4_k(
        device const char  *raw [[buffer(0)]], // [nb] contiguous q4_k superblocks (144B each)
        device       float *out [[buffer(1)]], // [nb*256]
        uint b [[thread_position_in_grid]]) {
    device const sepia_block_q4_k *blk =
        (device const sepia_block_q4_k *)(raw + (uint64_t)b * sizeof(sepia_block_q4_k));
    float unpacked[256];
    sepia_unpack_q4_k_block(blk, unpacked);
    device float *dst = out + (uint64_t)b * 256;
    for (int j = 0; j < 256; j++) dst[j] = unpacked[j];
}
