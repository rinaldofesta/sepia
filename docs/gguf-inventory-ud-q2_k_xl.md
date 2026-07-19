# GGUF inventory: Unsloth Inkling UD-Q2_K_XL

Generated 2026-07-19 by `tools/gguf_inspect.py`, entirely via HTTP Range
reads against Hugging Face -- no part of this GGUF was downloaded. Total
bytes actually fetched to produce this inventory: **42,346,295** (~40.4MB),
against 317,330,224,855 bytes (317.33GB) of file this describes. Full
machine-readable output: [gguf-inventory-ud-q2_k_xl.json](gguf-inventory-ud-q2_k_xl.json).

Reproduce with:

```
python3 tools/gguf_inspect.py --repo unsloth/inkling-GGUF \
  --file "UD-Q2_K_XL/inkling-UD-Q2_K_XL-00001-of-00008.gguf" \
  --all-parts --json --out docs/gguf-inventory-ud-q2_k_xl.json
```

## Repo id correction

README links `https://huggingface.co/unsloth/Inkling-GGUF`. The real repo
id is **`unsloth/inkling-GGUF`** (lowercase "inkling", capital "GGUF").
`unsloth/Inkling-GGUF` 307-redirects to it, so the README link still
resolves, but any tooling that calls the HF API directly (as this one
does) must use the lowercase form or it gets bounced through an extra
redirect the API itself already normalizes.

## File list

UD-Q2_K_XL is split into 8 parts under `UD-Q2_K_XL/` in the repo:

| part file | bytes | size | tensors in this part |
|---|---:|---:|---:|
| inkling-UD-Q2_K_XL-00001-of-00008.gguf | 12,986,167 | 0.01 GB | 0 (metadata-only shard) |
| inkling-UD-Q2_K_XL-00002-of-00008.gguf | 49,248,983,808 | 49.25 GB | 264 |
| inkling-UD-Q2_K_XL-00003-of-00008.gguf | 49,801,033,664 | 49.80 GB | 246 |
| inkling-UD-Q2_K_XL-00004-of-00008.gguf | 48,271,231,936 | 48.27 GB | 230 |
| inkling-UD-Q2_K_XL-00005-of-00008.gguf | 48,988,457,920 | 48.99 GB | 230 |
| inkling-UD-Q2_K_XL-00006-of-00008.gguf | 48,281,553,856 | 48.28 GB | 230 |
| inkling-UD-Q2_K_XL-00007-of-00008.gguf | 49,705,683,904 | 49.71 GB | 230 |
| inkling-UD-Q2_K_XL-00008-of-00008.gguf | 23,020,293,600 | 23.02 GB | 82 |
| **total** | **317,330,224,855** | **317.33 GB** (295.54 GiB) | **1512** |

Part 1 carries the model's full metadata (53 KV pairs, including the
201,024-token vocab and 446,189-entry BPE merge list) and zero tensors --
it's a pure header shard. Parts 2-8 each carry 3 metadata keys
(`general.architecture`, `split.no`, `split.count`) plus their slice of
the 1512 tensors and the tensor data itself. This confirms the layout
`docs/DESIGN.md` assumed for the download task: 8 GET/Range-resumable
files, no surprises.

Sibling quants also exist in the same repo at other precisions
(BF16, Q8_0, UD-IQ1_S, UD-IQ1_M, UD-Q3_K_XL, UD-Q4_K_XL) plus one
repo-root `mmproj-BF16.gguf` -- not inventoried here, out of scope for
this task.

## Quant types: overall vs. routed-expert tensors

Overall, across all 1512 tensors (317,317,138,696 bytes of tensor data;
the remaining 13,086,159 bytes across all 8 files are GGUF headers,
metadata, and alignment padding):

| ggml type | tensor count | bytes | % of tensor bytes |
|---|---:|---:|---:|
| IQ2_XS | 126 | 175,984,607,232 | 55.46% |
| IQ3_XXS | 59 | 109,131,595,776 | 34.39% |
| IQ4_XS | 7 | 17,968,398,336 | 5.66% |
| Q5_K | 261 | 9,032,736,768 | 2.85% |
| Q6_K | 199 | 3,581,706,240 | 1.13% |
| Q4_K | 1 | 694,738,944 | 0.22% |
| Q8_0 | 69 | 494,665,728 | 0.16% |
| F32 | 790 | 428,689,672 | 0.14% |

**Routed-expert tensors only** (`blk.N.ffn_gate_exps.weight`,
`blk.N.ffn_up_exps.weight`, `blk.N.ffn_down_exps.weight` -- 3 tensors per
MoE layer x 64 MoE layers = 192 tensors) account for **303,084,601,344
bytes, 95.51% of all tensor bytes**:

| ggml type | tensor count | bytes | where |
|---|---:|---:|---|
| IQ2_XS | 126 | 175,984,607,232 | `ffn_gate_exps` + `ffn_up_exps`, 63 of 64 layers each |
| IQ3_XXS | 59 | 109,131,595,776 | `ffn_down_exps` on 57 layers; `ffn_gate_exps`+`ffn_up_exps` on layer 65 only |
| IQ4_XS | 7 | 17,968,398,336 | `ffn_down_exps` on layers 40, 59, 60, 62, 63, 64, 65 |

So: **gate/up projections are IQ2_XS almost everywhere; down projection
is IQ3_XXS almost everywhere, bumped to IQ4_XS on 7 (mostly late/global)
layers; layer 65 (the last layer, also a global-attention layer) gets its
gate/up bumped to IQ3_XXS too.** This is Unsloth's importance-matrix-driven
("UD" = Unsloth Dynamic) per-tensor precision allocation, not a uniform
2-bit quant -- confirmed by `quantize.imatrix.*` metadata keys present in
part 1 (`quantize.imatrix.dataset = unsloth_calibration_inkling.txt`,
850 imatrix entries, 13,407 calibration chunks). **SEPIA's converter and
CPU dequant kernels for the expert path need IQ2_XS, IQ3_XXS, and IQ4_XS**,
not just one type.

Everything else is comparatively small and higher-precision: attention
projections and shared-expert FFNs are Q5_K/Q6_K/Q8_0, norms are F32,
`token_embd.weight` is Q5_K, `output.weight` is Q4_K.

## Dense vs. MoE layers

`inkling.dense_block_count = 2` in metadata, and tensor names confirm it
exactly: **layers 0-1 are dense** (`blk.{0,1}.ffn_gate/up/down.weight`,
no `_exps` suffix, Q5_K/Q6_K). **Layers 2-65 (64 layers) are MoE**
(`blk.N.ffn_{gate,up,down}_exps.weight`, 3D `[in, out, 256]`, plus a
router `blk.N.ffn_gate_inp.weight` shape `[6144, 258]` F32 -- 258 =
256 routed + 2 shared-expert-sink logits, matching
`shared_expert_sink: true` from `docs/DESIGN.md` -- and a router bias
`blk.N.exp_probs_b.bias` shape `[256]` F32, matching the aux-loss-free
sigmoid-plus-bias router). No layer has both dense and MoE FFN tensors.

Shared-expert tensors (`blk.N.ffn_{gate,up,down}_shexp.weight`, shape
`[in, out, 2]`, one per MoE layer) are Q5_K/Q6_K/Q8_0 -- always-active
experts get meaningfully higher precision than the routed pool.

Attention hybrid pattern from `inkling.attention.sliding_window_pattern`
(66-entry bool array) matches `docs/DESIGN.md` exactly: **55 local
(sliding-window) layers, 11 global layers at indices 5, 11, 17, 23, 29,
35, 41, 47, 53, 59, 65** (every 6th layer). `head_count_kv` per layer is
16 on local layers, 8 on global layers, also confirmed directly from
metadata.

## MTP tensors: absent

No tensor name in any of the 1512 tensors across all 8 parts matches
`mtp`/`speculat`/`draft` (case-insensitive), and no MTP-related key
appears in part 1's 53 metadata entries. **`docs/DESIGN.md` marked MTP
packaging in the GGUF as "likely bundled ... (unconfirmed)"; this
inventory resolves it: the 8 MTP layers are not present in Unsloth's
UD-Q2_K_XL conversion.** Whether they're absent from the source BF16
checkpoint entirely, or present in BF16 but dropped by Unsloth's GGUF
converter, is not answerable from this GGUF alone -- worth checking the
BF16 repo tree or `thinkingmachines/Inkling`'s original safetensors
index directly before writing off local MTP speculation for Phase 4.

## Vision/audio (mmproj) tensors: absent from this file, exist separately

No tensor name matches `vision`/`mmproj`/`image`/`audio`/`vit`/`clip`/
`whisper`. The repo tree (`unsloth/inkling-GGUF`) has exactly one
`mmproj-BF16.gguf` (183,264,288 bytes) at repo root, with **no
per-quant mmproj file** (no `mmproj-UD-Q2_K_XL.gguf`). So: UD-Q2_K_XL is
text-only as shipped; SEPIA's stated text-only scope decision costs
nothing here -- there's no quantized vision tower to skip, only a
separate always-BF16 one if multimodal is ever revisited.

## Alignment

`general.alignment = 32` in every part that carries it explicitly (parts
2-8; part 1 also has it in its full metadata set). This is GGUF's
default alignment value, stated explicitly rather than relied upon as a
default -- `tools/gguf_inspect.py` reads it from metadata with a
default-32 fallback per the GGUF spec, but Unsloth's writer sets it
either way.

## Tokenizer / vocab metadata keys

From part 1: `tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre =
"inkling"`, `tokenizer.ggml.tokens` (201,024 entries, matches
`inkling.vocab_size`), `tokenizer.ggml.token_type` (201,024 entries),
`tokenizer.ggml.merges` (446,189 BPE merge rules), `tokenizer.ggml.
bos_token_id = tokenizer.ggml.eos_token_id = 200006`,
`tokenizer.ggml.add_bos_token = False`, plus a full `tokenizer.
chat_template` (the Jinja template also visible via the HF API's `gguf.
chat_template` field). `inkling.unpadded_vocab_size = 200058` differs
from `inkling.vocab_size = 201024` -- the padding delta (966) is worth
checking against the tokenizer's real special-token count when the oracle
work reads the tokenizer, not assumed away.

## Inkling-specific metadata keys

All under the `inkling.*` namespace in part 1, cross-checked against
`docs/DESIGN.md`'s `config.json`-derived facts -- everything matches:

- `inkling.block_count = 66`, `inkling.embedding_length = 6144`,
  `inkling.dense_block_count = 2`
- MoE: `expert_count = 256`, `expert_used_count = 6`,
  `expert_shared_count = 2`, `expert_feed_forward_length = 3072`,
  `expert_gating_func = 2` (sigmoid), `expert_weights_scale = 8.0`
  (matches `route_scale: 8.0`), `feed_forward_length = 24576` (dense FFN
  width)
- Attention: `attention.head_count = 64`, `attention.head_count_kv`
  (16 local / 8 global, per-layer array), `attention.key_length =
  attention.value_length = 128`, `attention.sliding_window = 512`,
  `attention.sliding_window_pattern` (per-layer bool array),
  `attention.layer_norm_rms_epsilon ~= 1e-6`
- Relative position bias (no RoPE, matches design doc): `d_rel = 16`,
  `rel_extent = 1024`, `rel_extent_swa = 512` (a separate, shorter extent
  for sliding-window layers -- not previously documented in
  `docs/DESIGN.md`, worth folding in)
- Log-scaling past 128K context: `log_scaling_alpha = 0.1`,
  `log_scaling_n_floor = 128000`, plus `logit_scale_denom = 24.0`
  (undocumented in `docs/DESIGN.md`; needed for the oracle's logit math)
- Short convolutions: `shortconv_kernel = 4`, and per-layer tensors
  `shortconv_{attn,k,mlp,v}.weight` shape `[4, dim]` F32 -- four
  separate conv weights per layer (attention-branch, K, MLP-branch, V),
  not one
- `general.size_label = "256x23B"`, `general.quantized_by = "Unsloth"`,
  imatrix provenance under `quantize.imatrix.*`

One more per-layer tensor with no `docs/DESIGN.md` mention:
`blk.N.ffn_gscale.weight`, shape `[1]`, F32, present on **all 66**
layers (dense and MoE alike) -- a single scalar gate-scale value per
layer. And attention carries two tensors not previously named:
`attn_r.weight` (`[6144, 1024]`, Q8_0) and `attn_rel_proj.weight`
(`[512, 16]`, F32, matching `d_rel=16`) -- `attn_rel_proj` is almost
certainly the relative-position-bias projection; `attn_r` is unidentified
from tensor name alone and should be resolved by reading Inkling's
modeling code during the oracle task (Phase 0 step 3), not guessed here.

## Per-expert contiguity

GGUF stores each 3D expert tensor's dims in `ne[]` order (dim 0 =
fastest-varying / contiguous, dim 0..2 as read off the wire). For every
`_exps` tensor here, the parsed shape is `[in_or_out_dim, other_dim,
256]` -- **the expert count is `ne[2]`, the slowest-varying dimension**.
Block-quantized types (`IQ2_XS`, `IQ3_XXS`, `IQ4_XS`, all `QK_K = 256`)
group 256 *contiguous* elements along the fastest axis (`ne[0]`) into one
block; since `ne[0]` here (6144 or 3072) is always divisible by 256, no
block ever straddles a row, let alone an expert boundary. So each
expert's byte range is `[e * bytes_per_expert, (e+1) * bytes_per_expert)`
for `e` in `0..255`, fully contiguous -- confirmed by exact-integer
division below (verified against real tensor byte counts, not just
formula):

| tensor kind | shape `[ne0, ne1, n_expert]` | type | elements/expert | blocks/expert | bytes/expert |
|---|---|---|---:|---:|---:|
| `ffn_gate_exps.weight` (63/64 layers) | `[6144, 3072, 256]` | IQ2_XS | 18,874,368 | 73,728 | 5,455,872 (5.20 MiB) |
| `ffn_up_exps.weight` (63/64 layers) | `[6144, 3072, 256]` | IQ2_XS | 18,874,368 | 73,728 | 5,455,872 (5.20 MiB) |
| `ffn_down_exps.weight` (57/64 layers) | `[3072, 6144, 256]` | IQ3_XXS | 18,874,368 | 73,728 | 7,225,344 (6.89 MiB) |
| `ffn_down_exps.weight` (7/64 layers) | `[3072, 6144, 256]` | IQ4_XS | 18,874,368 | 73,728 | 10,027,008 (9.56 MiB) |

Verified: `blk.2.ffn_gate_exps.weight` is 1,396,703,232 bytes total;
`1,396,703,232 / 256 = 5,455,872.0` exactly. Same exact-division check
passes for the IQ3_XXS and IQ4_XS down-projection cases.

**Per-expert total (gate + up + down), one MoE layer:**
- 57 layers (IQ2_XS/IQ2_XS/IQ3_XXS): `5,455,872 x 2 + 7,225,344 =
  18,137,088 bytes = 17.30 MiB`
- 7 layers (IQ2_XS/IQ2_XS/IQ4_XS): `5,455,872 x 2 + 10,027,008 =
  20,938,752 bytes = 19.97 MiB`
- layer 65 (IQ3_XXS/IQ3_XXS/IQ4_XS): `7,225,344 x 2 + 10,027,008 =
  24,477,696 bytes = 23.34 MiB`

This is close to `docs/DESIGN.md`'s "~16MB" estimate per routed expert
(actual: 17.3-23.3 MiB depending on layer, average ~17.7 MiB across the
64 MoE layers) and, more importantly, **confirms the container design's
core assumption**: one expert = one contiguous byte range per tensor, so
`inkwell`'s per-expert `pread` is a real single read, not a gather across
scattered offsets, on the source GGUF's own layout.

## Answers for the download/converter tasks

- **Repo id**: `unsloth/inkling-GGUF` (not `Inkling-GGUF` as linked from
  the README; that URL still works via redirect).
- **UD-Q2_K_XL file paths + sizes**: see File list above; 8 parts,
  317,330,224,855 bytes total (317.33 GB).
- **Expert quant types**: `IQ2_XS` (gate/up, 63/64 layers), `IQ3_XXS`
  (down, 57/64 layers; also gate/up on layer 65), `IQ4_XS` (down, 7/64
  layers). SEPIA needs dequant kernels for all three, not one.
- **MTP tensors**: absent from this GGUF entirely.
- **mmproj/vision tensors**: absent from UD-Q2_K_XL; exist only as a
  separate, non-quantized `mmproj-BF16.gguf` at repo root.
- **Alignment**: 32 (GGUF default, explicit in metadata).
- **Contiguity**: confirmed -- each expert's slice of every `_exps`
  tensor is a contiguous byte range, exact arithmetic above.
