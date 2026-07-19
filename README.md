# SEPIA

Inference engine for [Inkling](https://huggingface.co/thinkingmachines/Inkling) (975B-parameter MoE, 41B active) on Apple Silicon. One model, one machine class: a Mac with 128GB of unified memory and a fast SSD.

Status: Phase 0. Nothing generates tokens yet. This README states the plan and the math before the code, on purpose.

## The problem

The smallest published Inkling quant ([Unsloth UD-IQ1_S](https://huggingface.co/unsloth/inkling-GGUF)) is ~270GB on disk and needs ~280GB of combined memory. A 128GB Mac cannot load it. llama.cpp's Inkling support is a draft PR ([#25731](https://github.com/ggml-org/llama.cpp/pull/25731)) with no MTP speculation. Today this model does not run on the largest MacBook you can buy.

## The bet

RAM stops being a hard cutoff when experts stream from disk. Inkling routes each token to 6 of 256 experts per layer, plus 2 shared experts that are always active. At 2-bit, one routed expert is ~16MB: a single pread. The plan:

- Resident in RAM: attention, shared experts, routers, embeddings (~20-25B params, ~20-25GB at 8-bit), plus a pinned set of the experts your own workload routes to most.
- Streamed from SSD: the rest of the ~16,000 routed experts, with a per-layer LRU, a routing-history cache that persists across sessions, and router-lookahead prefetch.
- MTP speculation: Inkling ships 8 multi-token-prediction layers in the checkpoint. No local engine uses them yet.
- Effort dial per request: `reasoning_effort` is a chat-template mechanism, so a server can set it per call (low for tool glue, high for planning).
- Tinker fine-tunes: Tinker is LoRA-based, so SEPIA plans to apply adapters at runtime on one fixed quantized base. No requantization per checkpoint.

## Expected numbers, stated before writing kernels

colibri measured 2.24 tok/s on this exact machine class (MacBook Pro M5 Max, 128GB) streaming GLM-5.2 (744B) at int4 with a ~75% expert hit rate ([report](https://github.com/JustVugg/colibri/blob/main/docs/METAL-M5MAX-PERF-REPORT.md)). Inkling's regime math: a cold token reads 6 experts x ~64 MoE layers x ~16MB = ~6GB; at a ~75% hit rate that drops to ~1.5GB per token from SSD. Target: 1.5-3 tok/s before prefetch and MTP wins. If the measured numbers land below that, they get published anyway.

Open question I cannot answer yet: colibri found GLM-5.2's next-layer routing 71.6% predictable from the current layer's state, which is what makes prefetch pay. Whether Inkling routes as predictably is unknown. A Phase 2 experiment answers it either way, and a negative result ships too.

## Phase 0 (current): validate before optimizing

- [ ] Tiny-random Inkling oracle: token-exact CPU forward vs the transformers reference
- [ ] `sepia.c` CPU engine passing the oracle 32/32
- [ ] SSD microbenchmark at expert-slab sizes (buffered vs F_NOCACHE)
- [ ] Remote GGUF header inventory of the Unsloth quants (HTTP Range reads, no download)
- [ ] Converter: GGUF to per-expert streaming container, byte-validated against the source

## Lineage

SEPIA fuses two projects, with attribution in [NOTICE](NOTICE):

- [ds4](https://github.com/antirez/ds4) (antirez): vertical single-model engine, surgical asymmetric quantization, validation against official-API logprobs, KV cache as "a first-class disk citizen".
- [colibri](https://github.com/JustVugg/colibri) (JustVugg): experts streamed from disk, the learning cache, router-lookahead prefetch, the tiny-oracle discipline.

Like ds4, this project is strictly opportunistic about its model: if a better open-weights model for this machine class appears, SEPIA follows it. Inkling-Small (276B/12B active, weights not yet released) fits 128GB in RAM at 2-4 bit; the container format is designed so it drops in with the streaming tier reduced to a no-op.

Design doc: [docs/DESIGN.md](docs/DESIGN.md). Built with heavy AI assistance (Claude), with a human leading design, testing and debugging, in the ds4 tradition.

License: Apache-2.0.
