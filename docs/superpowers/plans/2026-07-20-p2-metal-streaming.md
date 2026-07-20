# P2 — Metal + Streaming Machinery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First real tok/s number — replace the scalar CPU path (~29-31s/token, P1) with Metal kernels and a streamed, pinned expert store, then run the PILOT routing-predictability experiment.

**Architecture:** Quantized weights never materialize to f32 in bulk again. The 14.23GB `resident.bin` mmap is wrapped zero-copy into a `MTLBuffer` (unified memory); per-layer matvecs run as dequant-fused Metal kernels reading quantized blocks directly — eliminating the ~62GiB/token f32 re-dequant that is 60% of P1's cost. Routed experts stream from the GGUF parts into slab-allocated, per-slot-mlocked GPU-visible buffers addressed by a bindless uint64 table, evicted by plain LRU (fancier policies are P3). The banded flash-attention kernel is a from-scratch MSL design against the reference semantics extracted from llama.cpp's CPU/CUDA op (the PR has NO Metal implementation). The P1-verified CPU path stays byte-for-byte intact as the oracle: the Metal gate is identical greedy token sequences on the P1-recorded prompts plus bounded activation drift — never bit-exactness (GPU f32 accumulation vs CPU double accumulation makes that gate dishonest).

**Tech Stack:** C11 + Objective-C (ARC) runtime shim + MSL kernels. No C++. Shaders runtime-compiled from `metal/*.metal` via `newLibraryWithSource:` (ds4's pattern — no metallib toolchain step). ds4 (`~/Projects/Personal/ds4`, MIT) is the port source for scaffolding and Q8_0/Q4_K kernels.

## Global Constraints

- C11 + ObjC only, no C++ anywhere (AGENTS.md). Engine C stays `-O2 -std=c11 -Wall -Wextra` warning-clean; ObjC compiles with `-fobjc-arc -Wall -Wextra`, warning-clean.
- **The CPU path is the oracle and must not change behavior**: `./sepia` (tiny-oracle self-test, token-exact 32/32+20/20) and `./sepia --real` CPU generations (P1-recorded id sequences in `docs/p1-first-tokens.md`) gate every commit.
- **Metal acceptance policy (set here, once)**: (a) tiny-model GPU forward: greedy tokens match the CPU tiny oracle exactly for 32+20 steps (expected achievable at f32; if a genuine near-tie divergence appears, STOP and escalate with logits — do not loosen silently); (b) per-op/per-layer comparisons vs CPU use max-relative-error ≤ 2e-4 on f32 activations (attention/logits) and ≤ 1e-5 on pure dequant outputs (dequant is deterministic unpacking — GPU must match CPU bitwise for the *dequantized values* since no accumulation is involved; matvec results get the relative tolerance); (c) real-model GPU generations must reproduce the exact 32-token id sequences recorded in `docs/p1-first-tokens.md` for both prompts. A divergence triggers the near-tie protocol (dump top-2 logits at the divergent step; relative gap < 1e-3 = documented near-tie, else bug).
- Banded-attention kernel: **f32 accumulators mandatory** (llama.cpp documents fp16 VKQ accumulator overflow with Inkling's large-magnitude V — `fattn-mma-f16.cuh:695-706`); all index/stride arithmetic in the kernel and its host code in 64-bit (int64 tests in the reference are load-bearing at 1M context).
- Threading: no OpenMP, no spin-waits, no polling loops — blocking `pthread_cond_wait` pools with generation counters only (colibrì lesson: active spin cost -39% GPU power budget; ds4's pools at `ds4.c:1317-1469` are the pattern).
- RAM budget: total locked+resident target ≤ ~110GB (colibrì: ~120GB triggers macOS memory compression). mlock only per-slot on the expert store, never the whole model. mlock failure degrades the cache budget with one warning, never aborts (ds4 `ds4_metal.m:8183`).
- Expert streaming I/O keeps **F_NOCACHE** (SEPIA's own measured 13.3GB/s and ~2x-buffered result, `docs/ssd-bench.md`); ds4 uses page-cache+F_RDADVISE instead — do NOT silently adopt it; Task 11 A/Bs the two on real decode before any switch.
- Makefile: fix the `LDFLAGS ?=` footgun — Metal frameworks are appended via a dedicated `METAL_LDFLAGS` in the recipe, never by overriding LDFLAGS (ledger note; ds4 uses `METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal`).
- CI (`macos-latest`, weights-free, no GPU dispatch assumed): shader **offline syntax-check** (`xcrun -sdk macosx metal -c`) and ObjC compile join `make ci`; anything needing a live `MTLDevice` or weights is local-only (`tokreal`/`configcheck` convention).
- Quant blocks stay byte-identical to the GGUF (no requant); GPU kernels read the same on-disk block layouts `src/quants.c` documents.
- Scope fence: NO learning cache / `.sepia_usage`, NO auto-pin, NO heat/hysteresis re-pin, NO PILOT *prefetch* (all P3, `docs/DESIGN.md:161-163`); NO MTP (P4). P2 ends at: Metal path + LRU-streamed experts + honest tok/s table + the PILOT *measurement* published either way.
- Commit style plain imperative; frequent commits; every commit `make ci` green.

## Context Digest (read this before any task — it replaces re-research)

**Cost anatomy of P1's 29-31s/token** (from shapes, `src/sepia.c` HEAD 9fc97d6): resident re-dequant+matmul ~60% (`real_fill_layer` materializes ~62.3GiB f32/token across 66 layers — sliding-sparse ~981MiB, global-sparse ~931MiB, dense ~2.18GiB per layer — then `linear`/`dotf` passes over it again); routed-expert stream ~38% (384 slab triples/token, fused `qlinear`, I/O itself only ~0.53s of 7.1GB cold at 13.3GB/s — CPU dequant is the cost, not disk); logits ~2% (200,058-row Q4_K `qlinear`); attention arithmetic negligible. Single scalar thread throughout; `-pthread` linked but unused in compute.

**ds4 port anchors** (`~/Projects/Personal/ds4`, MIT, credited in NOTICE):
- Runtime shader compile: `ds4_gpu_full_source()` `ds4_metal.m:3039` (concat preamble + `metal/*.metal` read at startup), `newLibraryWithSource:` `:4657`. Makefile lists `.metal` files only as rebuild triggers (`Makefile:15,208-209`).
- Zero-copy weights: `ds4_gpu_add_model_view_range()` `ds4_metal.m:1455-1598` — `newBufferWithBytesNoCopy` on page-aligned mmap views, `MTLResourceStorageModeShared`. SEPIA needs ONE view (resident.bin is 14.23GB, one mmap, page-aligned by construction).
- Batched encoding: one long-lived command buffer + open compute encoder across many dispatches (`ds4_gpu_compute_encoder`/`flush`/`end` `:706-735`, `:6414-6613`); sync only at explicit boundaries via `waitUntilCompleted` (blocking, no poll).
- Router readback without stall: `MTLSharedEvent` signal encoded mid-graph (`ds4_gpu_signal_selected_readback_ready` `:6453`), dedicated loader pthread blocks on `waitUntilSignaledValue` then preads (`ds4.c:14508` `metal_graph_selected_async_load_worker_main`).
- Expert store: slab `MTLBuffer`s ~4GiB subdivided into fixed slots + free-list (`:8274-8308`); per-slot `mlock` (`:8361-8410`); bindless per-layer `uint64_t` addr tables (expert-id → resident GPU address, 0 = miss); GPU validate kernel `kernel_stream_expert_cache_validate` `moe.metal:1441`; in-flight generation stamps prevent evicting buffers a queued command buffer still reads (`:834-912`). P2 uses this infra with **plain LRU** eviction (ds4's hotness-decayed LFU is P3-adjacent; roadmap says LRU).
- GPU-dequant matvec kernels to port/adapt: Q8_0 `kernel_mul_mv_q8_0_f32` `dense.metal:181` (+ fused gate-up-swiglu `:204`), Q4_K family in `moe.metal`; IQ2_XXS kernel + embedded codebook `moe.metal:23-92` is the **pattern** for our IQ kernels (grids as `constant` tables, ours already vendored in `src/quants_grids.h`). ds4 has NO Metal kernels for Q5_K/Q6_K/IQ2_XS/IQ3_XXS/IQ4_XS — those are new, written from our bit-verified CPU references in `src/quants.c`.
- Thread pool idiom: mutex + 2 condvars + generation counter, workers in `pthread_cond_wait`, created once (`ds4.c:1317-1469`, pread pool `ds4_metal.m:7836-8041`).
- NOT portable: all RoPE/YaRN (`dsv4_rope.metal`), compressor/indexer/DSA machinery, hyper-connections (`dsv4_hc.metal`), FP8/FP4 KV tables, `dsv4_indexed_mixed_attention_heads8` outer control flow. Reusable from `flash_attn.metal`: the online-softmax/tiling shell as reference only.

**Banded attention reference spec** (extracted from vendor/llama.cpp @ ce16fff2 — the PR is CPU+CUDA only, no Metal exists):
- Semantics (CPU `ggml-cpu/ops.cpp:8494-8739`): per (q,k): `s = dot(q,k) * scale` (scale = `1/head_dim`, applied to the QK dot ONLY, never the bias); `rel_dist = q_idx + (n_kv - n_batch) - kv_idx` in int64; `if (0 <= rel_dist < rel_extent) s += rel_logits[rel_dist, head, q_idx]` — **hard zero outside the band, no edge-clamp**; additive mask AFTER the bias; online softmax; sinks/ALiBi/softcap inert for this op.
- rel_logits is produced OUTSIDE the op by `r_proj` matmul (pinned `GGML_PREC_F32_PEDANTIC` — keep our rel projection f32); per-head RMS q/k norms and the log-N `tau` scaling also live outside the op; `tau` multiplies BOTH q and rel_logits, global (non-SWA) layers only.
- SEPIA's own CPU implementation (`attn_forward_chunk`, `src/sepia.c:871-997`) is the direct oracle: proj → sconv(K,V) → per-head RMS norm(Q,K) → cache write → per (t,h): dense rel_logits recompute (`r_vec · rel_proj` over the whole band) → content `dot*1/head_dim` + `bias*tau` if `distance < rel_extent` → stable softmax → weighted V sum → `wo`. Sliding: `kv_lo = max(0, q_pos-511)`; global: `kv_lo=0`. KV cache holds full history; windowing is arithmetic, not storage.
- Traps: fp16 accumulator overflow with Inkling's V magnitudes (use f32 accum); 64-bit strides load-bearing; support non-contiguous rel_logits or document the packed-only choice.

**SEPIA integration points** (`src/sepia.c` @ 9fc97d6): `RealModel`/`RealLayer` (manifest-resolved `QTensor`s at load, `:1699-1754`), `real_fill_layer` `:1991-2048` (the function the GPU path REPLACES), `real_expert_ffn` `:2062-2086`, `real_forward_chunk` `:2117-2156`, `mlp_moe_forward` seam (`lw->real_exps`, `:1079-1098`), `LayerCache` `:814-819` (per-layer kv_dim 2048 sliding / 1024 global), `attn_forward_chunk` `:871-997`, `qlinear`/`qrow` `:716-742`, quant layouts + grids `src/quants.c`/`src/quants_grids.h`. Tiny oracle: `tools/oracle/tiny/` (f32 safetensors, hidden 128, 6 layers — the fast dev loop).

## File Map

| Path | Status | Responsibility |
|---|---|---|
| `src/sepia_gpu.h` | new | C-side GPU API boundary (the only header sepia.c includes; compiles to no-ops without Metal) |
| `src/sepia_metal.m` | new | ObjC runtime shim: device/queue/library, buffers, encoders, kernel dispatch, expert store, loader thread |
| `metal/ops.metal` | new | rmsnorm, silu/elementwise, softmax, sconv, router top-k helpers, f32 matvec |
| `metal/matvec_q.metal` | new | dequant-fused matvec kernels: Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS, IQ4_XS (+ grids include) |
| `metal/banded_attn.metal` | new | rel-projection kernel + banded flash-attention kernel (from-scratch) |
| `metal/quants_grids_msl.h` | new, generated+committed | MSL `constant` versions of the IQ grids (generated from `src/quants_grids.h` by a tool) |
| `tools/make_msl_grids.py` | new, stdlib | converts `src/quants_grids.h` → `metal/quants_grids_msl.h` |
| `src/sepia.c` | modified | `--metal` flag, GPU forward wiring behind the existing seams, comparison/dump modes |
| `tools/test_gpu_quants.c` → mode of `sepia` | modified | GPU dequant/matvec fixture gate (local-only target) |
| `tools/pilot_routing.py` | new, stdlib | PILOT analysis over routing logs |
| `Makefile`, `.github/workflows/ci.yml` | modified | `sepia` gains Metal objects on Darwin, `METAL_LDFLAGS`, shader syntax-check in ci, local-only GPU test targets |
| `docs/p2-perf.md`, `docs/pilot-routing.md` | new | honest measurements; PILOT result published either way |

Dependency spine: 1 → 2 → 3 (dev loop exists) → {4,5,6 kernels} → 7 (attention) → 8 (tiny GPU forward gate) → 9 (real resident path) → 10 (expert store) → 11 (async overlap) → 12 (perf) → 13 (PILOT) → 14 (close-out). Tasks 4-6 are independent of 7 after 3.

---

### Task 1: Metal build scaffolding + `--metal` plumbing

**Files:** Create `src/sepia_gpu.h`, `src/sepia_metal.m` (init/teardown only), Modify `Makefile`, `.github/workflows/ci.yml` (via ci target only), `src/sepia.c` (flag)

**Interfaces (produced, consumed by all later tasks):**

```c
/* src/sepia_gpu.h — the ONLY GPU symbol surface sepia.c sees.
 * All functions return 0/NULL on failure after logging "sepia: metal: ...";
 * sepia.c decides whether that is fatal (--metal explicitly requested → die). */
#ifndef SEPIA_GPU_H
#define SEPIA_GPU_H
#include <stdint.h>
#include <stddef.h>

int  sepia_gpu_available(void);                 /* 1 if a device exists and library compiled */
int  sepia_gpu_init(const char *metal_dir);     /* compiles metal/*.metal from this dir; 1 on success */
void sepia_gpu_shutdown(void);
const char *sepia_gpu_device_name(void);
#endif
```

- `sepia_metal.m` init: `MTLCreateSystemDefaultDevice`, command queue, concatenate + `newLibraryWithSource:` over `metal/*.metal` read from `metal_dir` (ds4 pattern `ds4_metal.m:3039-4657`, minus the env-override machinery — YAGNI). A placeholder `metal/ops.metal` containing one trivial kernel (`kernel void sepia_touch(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 0.0f; }`) so the library compiles non-empty.
- Makefile: on Darwin, `sepia` links `src/sepia_metal.o` (compiled `-fobjc-arc -Wall -Wextra -O2`) with `METAL_LDFLAGS = -framework Metal -framework Foundation` appended **in the recipe** (`$(LDFLAGS) $(METAL_LDFLAGS)`) — the `?=` default stays untouched (footgun closed). Non-Darwin: a `src/sepia_gpu_stub.c` providing the same symbols returning 0 keeps the build portable.
- ci gains `shadercheck`: `xcrun -sdk macosx metal -c metal/*.metal -o /dev/null` (syntax/type check, no device needed, toolchain present on macos-latest).
- `sepia.c`: `--metal` flag parsed; for now prints device name via the API and continues on CPU (`sepia: metal: initialized (<name>)`), or dies if `--metal` given and init fails.

Steps: header+stub first with a failing compile-check, then shim, then Makefile, then `make ci` (must stay green with the ObjC object + shadercheck), commit `Add the Metal runtime scaffolding behind a --metal flag`.

### Task 2: Buffers — zero-copy resident wrap + generic buffer API

**Interfaces produced:**

```c
typedef struct SepiaGpuBuf SepiaGpuBuf;          /* opaque; wraps id<MTLBuffer>+offset */
SepiaGpuBuf *sepia_gpu_wrap_mmap(void *base, size_t len);     /* newBufferWithBytesNoCopy; base MUST be page-aligned (mmap is); NULL on failure */
SepiaGpuBuf *sepia_gpu_alloc(size_t len, int gpu_private);    /* Shared or Private storage */
void  sepia_gpu_free(SepiaGpuBuf *b);
void *sepia_gpu_host_ptr(SepiaGpuBuf *b);        /* NULL for private */
uint64_t sepia_gpu_gpu_addr(SepiaGpuBuf *b);     /* [buffer gpuAddress] for bindless tables (Task 10) */
```

TDD: a `--gpu-selftest` mode in sepia (local-only, needs a device): wrap a 4096-byte mmap'd temp file, run the touch kernel over it, verify bytes unchanged; alloc/free cycles; wrap `weights/resident.bin`'s existing mmap in `real_load` when `--metal` (assert non-NULL; 14.23GB < maxBufferLength — check and die loudly if not). Makefile target `gputest` (local-only). Commit `Add zero-copy GPU buffer wrapping for the resident weights`.

### Task 3: First kernels + the tiny-model comparison harness (the dev loop)

- `metal/ops.metal`: `sepia_rmsnorm_f32` (one threadgroup per row, simdgroup reduce, f32; weight-fused — port shape from ds4 `norm.metal` `kernel_rms_norm_fuse_impl`), `sepia_matvec_f32` (row-per-threadgroup dot, f32 accum), `sepia_silu_mul_f32`, `sepia_add_f32`, `sepia_softmax_f32` (stable, one row per threadgroup), `sepia_sconv_f32` (K=4 depthwise causal + residual, history window passed as buffer — mirror `sconv_apply` `src/sepia.c:790-805` exactly).
- Host: `sepia_gpu_dispatch_*` C functions per op (args structs with ne/nb fields, ds4 ABI convention); batched encoder (`sepia_gpu_begin/flush/end` mirroring ds4 `:6414-6613`).
- **Harness** `--gpu-compare-tiny` (local-only): loads the tiny oracle (f32 safetensors), runs each op CPU vs GPU on real tensors from a forward pass, reports max-rel-err per op. Gate: every op ≤ 2e-4 (most will be ~1e-7; the tolerance allows accumulation-order drift on long dots).
- Commit per green op group. Final commit `Add the first Metal ops with a tiny-model comparison harness`.

### Task 4: Dequant-fused matvec — Q8_0 + Q4_K (ds4 port)

- `metal/matvec_q.metal`: `sepia_matvec_q8_0` (adapt `dense.metal:181` — block layout identical to `src/quants.c` dequant_q8_0: f16 d + 32×i8, 34B), `sepia_matvec_q4_k` (adapt the moe.metal Q4_K matvec; superblock 144B, `get_scale_min_k4` logic in MSL). f32 accumulators. Row-major [out,in], in multiple of block — same contract as CPU `qlinear`.
- **Gate (bit-level on dequant, tolerance on matvec)**: extend the committed SQFX fixtures' reach — a `--gpu-quants <fixtures...>` mode dequantizes each fixture's blocks on GPU via a tiny debug kernel (`sepia_dequant_rows_<type>` — the matvec kernel's unpack factored into a callable also exposed standalone) and compares BITWISE against the fixture expected f32 (dequant has no accumulation — GPU must match exactly; MSL f16→f32 conversion is IEEE, same as CPU). Then matvec vs CPU `qlinear` on random data ≤ 2e-4 rel.
- Local-only target `gpuquants` runs all applicable fixtures. Commit `Add Q8_0 and Q4_K dequant-fused Metal matvec kernels`.

### Task 5: Dequant-fused matvec — Q5_K + Q6_K (new kernels)

From the CPU references (`dequant_q5_k`/`dequant_q6_k`, `src/quants.c` — layouts 176B/210B, the qh mask-shift and int8-scale semantics documented there). Same standalone-dequant bitwise gate + matvec tolerance gate as Task 4. STOP-and-report protocol on any bitwise dequant mismatch (element index + both bit patterns). Commit `Add Q5_K and Q6_K Metal matvec kernels`.

### Task 6: Dequant-fused matvec — IQ2_XS + IQ3_XXS + IQ4_XS (new kernels, grids in MSL)

- `tools/make_msl_grids.py` (stdlib): parses `src/quants_grids.h` (which it must NOT modify) → emits `metal/quants_grids_msl.h` with the same five tables as MSL `constant` arrays + provenance header; deterministic; committed. `#include`d by `matvec_q.metal` (the runtime source concatenation must handle the include — simplest: the shim inlines the file when concatenating, ds4-preamble style).
- Kernels follow ds4's IQ2_XXS pattern (`moe.metal:23-92` + its matvec) with our layouts: IQ2_XS 74B (u16 qs, 9-bit grid index + 7-bit sign), IQ3_XXS 98B (byte grid indices + packed scale-signs u32), IQ4_XS 136B (`kvalues_iq4nl` lookup) — exact reference semantics in `src/quants.c`.
- Same bitwise-dequant + matvec-tolerance gates. Commit `Add the IQ-family Metal matvec kernels with vendored MSL grids`.

### Task 7: Banded flash-attention kernel (from-scratch)

Two kernels in `metal/banded_attn.metal`:

1. `sepia_rel_project_f32`: per (token, head): `rel_logits[t,h,0..rel_extent) = r_vec[t,h,:] · rel_proj[:,r]` — f32 in/out/accum (reference pins this matmul to strict f32; `rel_proj` is F32 resident). One threadgroup per (t,h), threads over the band.
2. `sepia_banded_attn_f32`: one threadgroup per (t, h) for decode/short-T (T=1 dominant): loads `q_scaled` (tau pre-applied by host-side epilogue of the norm kernel on global layers — tau multiplies q AND rel_logits, matching `src/sepia.c:934-966`), iterates kv in `[kv_lo, kv_hi]` (passed as args — cache is full-history; windowing is arithmetic), per kv: `s = dot(q,k)*inv_head_dim; int64 dist = q_pos - kv; if (dist < rel_extent) s += rel_logits[dist];` — dist ≥ 0 by causality, the upper bound is the hard band cutoff (matches CPU `:960-970` and the ggml spec's zero-outside-band); online softmax with f32 M/S/acc (running max, rescale, V accumulate — the structure of ds4's `dsv4_misc.metal:577` accumulator, none of its DSA control flow); writes `attn_concat[t,h,:]`. GQA mapping `hk = h/group` in the host args. All strides/indices int64/size_t in host code; the kernel receives base offsets per dispatch so in-kernel indexing stays within one (t,h) row.
- Gate A (shapes): a synthetic harness mode compares GPU vs CPU `attn_forward_chunk` on randomized tensors at BOTH real geometries — sliding (H64/Hkv16/Dh128/window 512/rel 512) and global (H64/Hkv8/Dh128/rel 1024) — including band-edge cases: n_kv < rel_extent, n_kv == rel_extent, n_kv > rel_extent (global long-context: positions beyond the band get content-only scores), q_pos > log_scaling floor (tau ≠ 1). Tolerance ≤ 2e-4 rel on outputs.
- Gate B: tiny-model attention swap — tiny forward with ONLY attention on GPU matches CPU tiny forward ≤ 2e-4 per layer output.
- Commit `Add the banded flash-attention Metal kernel`.

### Task 8: Full tiny-model GPU forward — the token-exact gate

Wire a complete per-layer GPU graph for the tiny model (all ops from Tasks 3-7; weights are f32 → `sepia_matvec_f32`; router/top-6 on CPU host between encoder flushes for now — T=1 sync per layer is acceptable at tiny scale): `--metal` + tiny oracle runs the standard self-test through the GPU path. **Gate: `prefill 32/32, decode 20/20` — token-exact vs the same oracle the CPU gates on.** If a single position diverges: near-tie protocol (top-3 logits both sides); a genuine near-tie is an escalation to the controller (acceptance-policy decision), NOT a silent tolerance bump. Wire into local-only `gputest`. Commit `Run the tiny oracle token-exact through the full Metal forward path`.

### Task 9: Real-model resident path on GPU

- `real_load(--metal)`: wrap resident.bin (Task 2); build per-layer offset tables into the wrapped buffer (the manifest already gives offsets — `RealLayer` gains a GPU variant holding buffer+offset+type per tensor instead of arena pointers); KV cache and activations become GPU buffers (Shared).
- `real_forward_chunk(--metal)`: per layer — norm → matvec_q (wq/wk/wv/wr) → sconv → head norms (+tau) → cache write → rel-project → banded attention → wo → attn-sconv → router (CPU readback of 6 expert ids per the P1 seam; experts THIS task still run on the CPU streaming path `real_expert_ffn` — unchanged) → shared-expert matvec_q on GPU → mlp-sconv. Logits: `sepia_matvec_q4_k` over the unembed rows. `real_fill_layer`'s arena dequant is bypassed entirely in Metal mode.
- **Gate: both P1 prompts reproduce the exact recorded 32-token id sequences** (`docs/p1-first-tokens.md`), determinism double-run. Record the intermediate timing (expect the resident 60% to collapse; experts still ~11s/token CPU → expect roughly 3-5x overall).
- Commit `Run the real model's resident path on Metal`.

### Task 10: Streamed expert store on GPU

Port ds4's store shape with plain LRU: slab `MTLBuffer`s (Shared, ~4GiB each, budget flag `--expert-cache-gb`, default sized so total process stays under the 110GB ceiling: e.g. 64GB store ≈ 3400 slots ≈ 55% of all experts), fixed slot size = `idx.max_slab_bytes` rounded to page, free-list, per-slot `mlock` with degradation-on-failure (one warning, shrink budget); per-layer bindless `uint64` addr tables (gate/up/down); CPU pread (F_NOCACHE, existing fds) into slots on miss; LRU list on hit/install; in-flight generation stamps before eviction (ds4 `:834-912` pattern). GPU expert matvecs (Task 6 kernels) read via the addr tables; the weighted sum over ≤6 experts stays a small kernel (`sepia_moe_sum6` — ds4 `kernel_dsv4_moe_sum6_f32` shape).
**Gate: token sequences still exact on both prompts; cold + second-run (warm) timing recorded; report hit rate** (instrument: hits/misses per token printed with `--verbose-cache`). Commit `Add the LRU-streamed, mlocked expert store on GPU`.

### Task 11: Async overlap + I/O A/B

- `MTLSharedEvent` after the router: encode signal, commit batch without waiting; loader pthread (cond_wait pool, created once) wakes on the event, preads missing experts for the CURRENT layer into slots while the GPU proceeds with shared-expert + next encodes; graph waits on install completion only where the expert matvec is encoded (ds4's two-stage pipeline, `ds4.c:14508+`).
- A/B on real decode (32-token run each): F_NOCACHE preads (current) vs page-cache+`F_RDADVISE` (ds4's choice) — record both in the perf doc; keep the winner (constraint default: F_NOCACHE unless the A/B shows ≥10% end-to-end win).
- **Gate: sequences exact; tok/s improves or the overlap is documented as bandwidth-bound.** Commit `Overlap expert streaming with GPU compute via shared events`.

### Task 12: The first honest tok/s table

Runs (each double-run for determinism, sequences must stay exact): both P1 prompts × 32 tokens, cold (page cache purged, cache empty) and warm (immediate re-run); one 256-token generation (prompt 1) for steady-state + cache-hit curve; RSS via `/usr/bin/time -l`; per-stage breakdown (`--timing` instrumentation: resident matvecs / attention / experts-io / experts-compute / logits). `docs/p2-perf.md` in the measured-claims register: the table, the hit rate, comparison against the documented I/O ceilings (~7.5 warm / ~1.9 cold tok/s) and the 1.5-3 tok/s target, and what binds (compute vs I/O). README perf table updated from it. Commit `Record the first real tok/s measurements`.

### Task 13: PILOT routing-predictability experiment

- Instrument: `--route-log FILE` writes per (token, layer) the 6 selected expert ids + router weights (binary, small).
- Generate: ≥512 tokens across ≥4 diverse prompts (reuse P1 prompts + two longer ones) on the Metal path.
- `tools/pilot_routing.py` (stdlib): one-layer-ahead predictability à la colibrì — for each (token, layer L), predict layer L+1's top-6 as layer L's top-6; report recall@6 (fraction of L+1's actual experts present in L's set), per-layer curve, aggregate; also same-expert-consecutive-token overlap (temporal locality, informs the LRU story). GLM-5.2 baseline: 71.6% (colibrì); decision threshold for P3 PILOT prefetch: **≥ ~60% recall** (from the approved plan — cite it; this threshold enters the repo docs here for the first time).
- `docs/pilot-routing.md`: method, numbers, verdict, **published either way** (a negative result is a result — DESIGN.md:159-160 commits to this). DESIGN.md open-question line resolved with the measured number.
- Commit `Run and publish the PILOT routing-predictability experiment`.

### Task 14: Close-out

- P1 deferral honored: commit a reproducible tokenizer stress-sweep (`tools/stress_tokenizer.py`, dev-only via .venv, deterministic seed, ~8300 generated strings vs HF reference through the SPTK sidecar; recover the generator from `.superpowers/sdd/task-10-report.md` if still present, else re-derive — the corpus generator spec: random unicode mixes across the scanner's branch classes) + a one-line README claim update pointing at it.
- iobench hygiene debt (ledger): NULL-checks on `tids`/`args` mallocs, join-before-return on partial spawn failure.
- README/DESIGN roadmap tick (P2 done with the measured tok/s + PILOT verdict; P3 next), progress.md is controller-owned.
- `make ci` + full local gate suite green; commit `Close P2: Metal path measured, PILOT published, docs reconciled`.

---

## Self-Review

1. **Spec coverage** — roadmap line: "port ds4 Metal kernel architecture" → Tasks 1-3 (shim/buffers/ops) + 4 (kernel ports); "banded flash-attention kernel" → Task 7; "expert LRU + pinned store + mlock + F_NOCACHE" → Tasks 10-11; "macOS lessons (no OMP spin-wait, ~110GB RAM ceiling)" → Global Constraints + Tasks 10-11 enforcement; "first tok/s number" → Task 12; "then the PILOT routing-predictability experiment" → Task 13. P1 deferrals → Task 14. Scope fence excludes P3/P4 material explicitly.
2. **Placeholder scan** — kernel tasks specify contracts + exact reference anchors (in-repo CPU implementations that are themselves bit/token-gated, plus ds4 file:line anchors) rather than full MSL listings; each carries a hard numeric gate and a STOP protocol. This is the calibration that survived P1's Task 14. The acceptance-policy question (bit-exact impossible on GPU) is decided ONCE in Global Constraints, not per-task.
3. **Type consistency** — `SepiaGpuBuf`/`sepia_gpu_*` names used consistently across Tasks 1-11; fixture gates reuse the committed SQFX files and the P1-recorded token sequences as cross-task constants; `rel_logits`/`kv_lo`/`kv_hi`/`inv_head_dim` naming matches `src/sepia.c`'s existing conventions.

## Execution notes

- Tasks 1-8 need no real weights (tiny oracle + fixtures + synthetic); 9-13 need the local weights and a live MTLDevice — all local-only gates, CI stays weights-free with shadercheck + ObjC compile + the untouched CPU suites.
- The CPU path is never modified except at the existing `--metal` seams; any diff to CPU-side math is a review-rejectable defect.
- Worker subagents: one task per fresh subagent, controller reviews between tasks; `make ci` + "CPU oracle untouched" are the invariants every commit keeps.
