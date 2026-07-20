# SEPIA design

Written 2026-07-19, before the code. Facts below were verified against
primary sources on that date (raw `config.json`, the llama.cpp PR, both
parent repos); if reality diverges later, this doc gets corrected, not
defended.

## What SEPIA is

A vertical inference engine that runs Inkling (Thinking Machines Lab,
released 2026-07-15, Apache-2.0) on a Mac with 128GB of unified memory,
by treating RAM and SSD as one memory hierarchy. It fuses two parents:

- ds4 (antirez): one model, one engine, quantization validated against
  official-API logprobs, disk-first KV cache, OpenAI plus Anthropic
  compatible server, native agent.
- colibri (JustVugg): the model never fits in RAM and that is fine.
  Per-expert streaming with LRU plus a pinned hot store, a persistent
  learning cache of routing history with confidence-ramped auto-pin,
  router-lookahead prefetch (PILOT), MTP speculative decoding, and a
  tiny-random-model oracle gating every engine change.

## Why it should exist

1. No released Inkling quant fits 128GB (smallest is ~270GB). Streaming
   is the only local path on this machine class, and nobody has built it
   for this model.
2. llama.cpp support is a draft PR with no MTP; Inkling ships 8 MTP
   layers and the reference code claims lossless speculative decoding.
3. The effort dial (`reasoning_effort`) is a chat-template mechanism.
   A server can set it per request; none does today.
4. Tinker fine-tuning is LoRA-based. A fixed quantized base plus runtime
   adapter application gives local Tinker checkpoints without
   requantizing per fine-tune.

## Verified model facts that drive the design

From the raw `config.json` of `thinkingmachines/Inkling` (2026-07-19):

- 66 layers, hidden 6144. MoE: 256 routed experts per layer, top-6,
  2 shared experts (`shared_expert_sink: true`), per-expert intermediate
  3072. One routed expert is ~56.6M params (~16MB at pure 2-bit;
  measured 17.3-23.3MiB in the mixed UD-Q2_K_XL quant, average
  ~17.7MiB).
- Hybrid attention: 55 local layers (sliding window 512, 16 KV heads) and
  11 global layers (indices 5, 11, ..., 65; 8 KV heads). 64 query heads,
  head_dim 128.
- No RoPE. Positions come from a banded content-dependent relative
  position bias (`d_rel` 16, `rel_extent` 1024); llama.cpp added a fused
  op for it (`GGML_OP_FLASH_ATTN_EXT_BANDED`, PR #25731). Attention
  log-scaling past 128K tokens.
- Short convolutions (`use_sconv`, kernel 4) after K/V projections and on
  residual branches: per-layer conv state belongs in session state.
- Router: sigmoid plus bias (aux-loss-free), `norm_after_topk`,
  `route_scale` 8.0. `dense_mlp_idx: 2` marks early dense layers; exact
  semantics to be confirmed from modeling code during oracle work.
- 8 MTP layers in `mtp_config` (6 local, 2 global). Verified 2026-07-19
  from the checkpoint's weight map: all 160 MTP tensors live in a
  dedicated `mtp.safetensors` shard (`model.mtp.layers.0-7`, full
  transformer blocks with their own attention and sconv), separate from
  the 108 main shards. The Unsloth GGUFs do not include them at all, so
  Phase 4 sources MTP from that one shard and quantizes it in-house
  (8-bit or better, per colibri's acceptance-collapse lesson).
- Vocab 201024 (tiktoken), context 1M. Multimodal input (vision, audio),
  text output. SEPIA is text-only until further notice: a scope decision.
- BF16 checkpoint: 1.9TB, 108 shards, transformers 5.14.
- llama.cpp gotcha worth inheriting: some ops need int64 indexing; a
  256-expert MoE at this scale overflows int32.

## The memory regime on a 128GB M5 Max

Reference measurement, not a guess: colibri, on an M5 Max 128GB, streams
GLM-5.2 (744B, int4) at 2.24 tok/s with a ~75% expert hit rate, RSS
~98GB. Its report also documents three traps SEPIA inherits as rules:

- OpenMP active-spin steals the GPU's power budget on Apple Silicon
  (measured -39%). Passive waits only.
- RAM budget ~110GB is safe; ~120GB crosses into macOS memory
  compression and everything degrades. Pinned buffers get mlock'd.
- O_DIRECT-style reads (F_NOCACHE) were ~2x buffered throughput for the
  streaming workload.

Inkling budget: resident slice (attention, shared experts, routers,
embeddings: roughly 20-25B params) would be ~20-25GB at 8-bit; the
UD-Q2_K_XL source actually keeps most of it at Q5_K/Q6_K, which is why
the extracted resident.bin comes out at ~14GB (container section below).
The rest of RAM holds the pinned expert set, the per-layer LRU, and KV.
Cold decode reads ~7.1GB per token (6 experts x 64 MoE layers x
18,498,816 bytes, the measured average slab); at colibri-class hit
rates, ~1.8GB per token. Honest target: 1.5-3 tok/s before MTP and
prefetch wins.

## Container ("inkwell")

`model.inkwell/` holds `resident.bin` (embeddings, attention, shared
experts, routers, norms, output head, MTP layers), one `experts-NN.bin`
per layer (256 slabs; each expert is gate, up, down plus scales,
contiguous and 4KB-aligned so one expert is one aligned pread), and
`index.json`. Quant blocks are byte-identical to the source GGUF: no
requantization, so ported ds4/ggml kernels apply unchanged. First weight
source: Unsloth UD-Q2_K_XL (~317GB; not the smallest available quant,
chosen over the ~270GB 1-bit for output quality). Measured by the
inventory (docs/gguf-inventory-ud-q2_k_xl.md): routed experts are 95.5%
of tensor bytes; one expert's gate+up+down averages ~17.7MiB
(17.3-23.3 depending on layer); expert quant types are IQ2_XS (gate/up),
IQ3_XXS and IQ4_XS (down), so those are the dequant kernels SEPIA ports
first (gate/up on layer 65 alone are IQ3_XXS, the one exception). The
SSD benchmark (docs/ssd-bench.md, measured 2026-07-19)
settled the repack question: this machine reads 13.33 GB/s of random
15MB blocks with F_NOCACHE, and three ~5MB preads match one ~15MB pread
within 0.13%. So the plan of record is now: no repack, stream each
expert's three tensors directly from the GGUF parts via an index
sidecar; `resident.bin` extraction (small, ~14GB) stays, for mlock and
for adding the MTP sidecar later. One caveat before this is final: GGUF
tensor data is 32-byte aligned, not page-aligned, and the benchmark read
at block-aligned offsets; task 0.7 must verify F_NOCACHE throughput at
GGUF-realistic unaligned offsets before dropping the repack path for
good. The I/O ceilings at the measured average slab (18,498,816 bytes):
~7.5 tok/s warm (~1.8GB/token at 75% hit) and ~1.9 tok/s cold
(~7.1GB/token), pure I/O, before any compute overlap. docs/ssd-bench.md
quotes 8.88/2.22 for the round pre-measurement 1.5GB/6GB
parameterization: same bandwidth, different per-token estimate.

## Phase 0 (current milestone): validate before optimizing

1. Repo bootstrap: license, NOTICE, this doc, CI.
2. Kick off the ~317GB download in the background.
3. Tiny-random oracle: instantiate the real architecture at toy dims
   (seeded), greedy-generate plus one teacher-forcing pass, save tiny
   weights under `tools/oracle/` (the one weights path git tracks) and
   `ref_inkling.json`. Forces early answers on `dense_mlp_idx`, MTP
   packaging, sconv placement.
4. `sepia.c`: single-file C11 CPU engine, f32 math, no quant, no
   streaming. Default no-args run is the self-test: token-exact against
   the oracle, prefill and decode reported separately. The hardest Phase
   0 item: banded attention and sconv have no parent code to port.
5. iobench: threaded random preads at 5/15/30MB, buffered vs F_NOCACHE,
   on the actual SSD. Published table plus the repack decision.
6. Remote GGUF inventory via HTTP Range reads: exact tensor types,
   per-expert slice contiguity, MTP tensors, alignment. No download
   needed.
7. Converter v0: GGUF to inkwell, shard-by-shard, resumable, free-space
   guard, byte-validated by re-extracting sampled experts.

Exit criteria: oracle self-test token-exact (32/32), iobench table
published, inventory published, converter byte-validated, container on
disk, repo public with CI green. Live status per criterion: README's
Phase 0 checklist and docs/container.md section 6.

## Roadmap (each phase gets its own plan)

- P1 (done): CPU dequant for the container's quant types, C tokenizer
  exact vs HF `tokenizers` and `llama-tokenize`, first real greedy tokens
  from the full model cross-checked against the pinned llama.cpp
  PR-branch build on the same GGUF (text-exact both test prompts,
  id-exact on one). Full results: docs/p1-first-tokens.md.
- P2 (done): Metal kernels (banded flash attention, dequant-fused matvec,
  MoE routing) replace the scalar CPU path; routed experts served from an
  LRU-streamed, per-slot-mlocked GPU cache with async prefetch overlap;
  both P1 prompts still exact-sequence on the GPU path. Measured
  throughput ~1.39-1.57 tok/s steady-state (cold and warm) — the floor
  of, not comfortably inside, the 1.5-3 tok/s target below, and now
  compute-bound rather than I/O-bound (falls short of even the cold
  pure-I/O ceiling despite good cache hit rates). Full results:
  docs/p2-perf.md. The routing-predictability experiment (PILOT) is a
  clean negative result: mean one-layer-ahead expert recall@6 is 2.34%,
  indistinguishable from the 2.34% random-chance baseline and far below
  both the ~60% P3-viability threshold and colibrì's 71.6% GLM-5.2
  measurement — Inkling's router is not predictable one layer ahead. A
  separate same-layer consecutive-token overlap measurement (38.2%, ~16x
  chance) is real and explains this phase's own cache hit rates; it is a
  different mechanism from the one PILOT prefetch would have exploited.
  Full results: docs/pilot-routing.md.
- P3: learning cache (`.sepia_usage`), confidence-ramped auto-pin (at
  most half the expert budget, colibri's rule), heat-based re-pin with
  hysteresis — all three exploit the temporal locality PILOT confirmed is
  real, and stand independently of PILOT's own next-layer question. PILOT
  *prefetch* is dropped from scope: the P2 measurement above (2.34% vs a
  ~60% viability threshold) means predicting layer L+1's experts from
  layer L's selection has no basis in this model.
- P4: MTP speculation. MTP weights at 8-bit or better (colibri measured
  0-4% draft acceptance with a 4-bit MTP head), draft depth 1-2, and the
  draft/verify paths pinned to identical kernels (colibri's SPEC_PIN
  lesson). Hypothesis to test: in a disk-bound regime, speculation
  batches expert I/O across draft positions, so the economics differ
  from colibri's full-residency negative result. GBNF grammar-forced
  drafting for structured output.
- P5: server (OpenAI plus Anthropic wire), exact tool replay, disk-first
  sessions (at 1M context the 11 global layers alone are ~45GB of KV),
  per-request effort dial, agent integrations.
- P6: own quantizer from BF16 (shard-by-shard over 1.9TB), in-engine
  imatrix collection, asymmetric recipes vs the Unsloth baseline, NLL
  scoring against official continuations, published containers.
- P7: runtime LoRA for Tinker checkpoints; Inkling-Small as a drop-in
  when its weights release (fits in RAM, streaming tier becomes a no-op).

## Open questions

- ~~`dense_mlp_idx` semantics and MTP layer internals~~ resolved in Phase 0
  step 3: `dense_mlp_idx` is the count of leading dense layers (layers
  `[0, dense_mlp_idx)` are dense MLP, the rest MoE); MTP layer internals
  (checkpoint structure) documented, but transformers 5.14.1 has no MTP
  *forward-path* implementation at all for Inkling to read. Full detail in
  `docs/architecture-notes.md`.
- ~~Two GGUF tensors with no documented mapping yet~~ resolved in Phase 0
  step 3, from the actual modeling code (not the hypothesis below, which
  the code reading corrected): `blk.N.attn_r.weight` (6144x1024, Q8_0,
  every layer) is `self_attn.r_proj.weight` (real checkpoint name
  `attn.wr_du.weight`), the content projection feeding the banded relative
  position bias -- not `attn.rel_logits_proj.proj`, which is a real but
  separate, much smaller (`[16, rel_extent]`) tensor. `blk.N.ffn_gscale.weight`
  (scalar, every layer) is `mlp.global_scale` (dense layers) or
  `mlp.gate.global_scale` (MoE layers), gated by the real config's
  `use_global_scale` field name. Full detail in `docs/architecture-notes.md`.
- ~~Inkling routing predictability~~ resolved by the P2 experiment (Task
  13): measured mean one-layer-ahead recall@6 = **2.34%** across 708 token
  positions over 4 diverse prompts (708 * 63 layer-pairs = 44,604
  instances) -- indistinguishable from the 2.34% random-chance baseline
  for uniform 6-of-256 expert selection, and roughly 26x below the ~60%
  P3-viability threshold and roughly 31x below colibrì's 71.6% GLM-5.2
  measurement.
  Clean negative result: no one-layer-ahead structure at any of the 63
  measurable layers or in any of the 4 tested topics. A separate
  same-layer consecutive-token overlap measurement (38.2%, ~16x chance)
  explains Task 12's LRU hit rates but is not the PILOT signal. Full
  detail in `docs/pilot-routing.md`.
- Hosted-API logprob availability for P6 validation.
- SSD endurance: sustained ~1.5GB per token is real write-free read load,
  but worth documenting for users planning heavy use.
