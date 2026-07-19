# Inkling architecture notes

Written 2026-07-19 for the C engine implementer (task 0.4), who has not seen
the Python modeling code and must reproduce it token-exactly from this doc,
`tools/oracle/ref_inkling.json`, and `tools/oracle/tiny/`. Every claim below
is cited to a file/class/function in `transformers==5.14.1`
(`transformers/models/inkling/{modeling,modular,configuration}_inkling.py`,
installed at `.venv/lib/python3.12/site-packages/transformers/models/inkling/`)
or to the real checkpoint's `config.json` / `model.safetensors.index.json`
(`thinkingmachines/Inkling`, fetched from the HF hub 2026-07-19). Where the
real checkpoint's field names or tensor names are given, they were fetched
directly, not guessed.

`modeling_inkling.py` is auto-generated from `modular_inkling.py`; both are
byte-identical for the text-model path used here, so citations use whichever
has the clearer line.

## 0. How the real checkpoint differs from the transformers module tree

This matters immediately because the tensor names below split into three
different vocabularies:

1. **Real checkpoint (on HF hub, `model.safetensors.index.json`)**: SGLang-
   native names, e.g. `model.llm.layers.5.attn.wq_du.weight`.
2. **transformers module attributes** (what `modeling_inkling.py` calls
   things): e.g. `model.language_model.layers.5.self_attn.q_proj.weight`.
3. **GGUF (Unsloth conversion, `docs/gguf-inventory-ud-q2_k_xl.md`)**:
   llama.cpp names, e.g. `blk.5.attn_q.weight`-style (see the mapping table
   in §9).

transformers bridges (1)->(2) via `WeightRenaming`/`WeightConverter` entries
in `transformers/conversion_mapping.py` (searched under the model_type key
`"inkling_mm_model"`). There is **no** `_checkpoint_conversion_mapping`
class attribute on `InklingPreTrainedModel` — the renaming table lives
entirely in `conversion_mapping.py`. The C engine reads (1) directly (the
real `.safetensors` shards or the GGUF), so §9's table is the one that
matters operationally; §§1-8 use transformers' names because that's the
code being described, and cross-reference (1) throughout.

## 1. Dense vs MoE layer split (`dense_mlp_idx`)

**Resolved directly from `InklingTextConfig.__post_init__`**
(`configuration_inkling.py:112-131`):

```python
if self.mlp_layer_types is None:
    dense_mlp_idx = kwargs.pop("dense_mlp_idx", 0)
    self.mlp_layer_types = [
        "dense" if i < dense_mlp_idx else "sparse" for i in range(self.num_hidden_layers)
    ]
```

`dense_mlp_idx` is the count of **leading** dense layers. With the real
`dense_mlp_idx: 2` (confirmed in the real `config.json`'s `text_config`),
layers 0 and 1 use `InklingMLP` (dense SwiGLU) and layers 2-65 use
`InklingMoE` (routed + shared experts). This matches
`docs/gguf-inventory-ud-q2_k_xl.md`'s independent finding from tensor names
(`inkling.dense_block_count = 2`, `blk.{0,1}.ffn_*` have no `_exps` suffix).

`InklingDecoderLayer.__init__` (`modeling_inkling.py:552-555`) picks the
module directly from this list: `self.mlp = InklingMoE(config) if
config.mlp_layer_types[layer_idx] == "sparse" else InklingMLP(config)`. There
is no other branch — a dense layer has **no** router, no experts, and no
shared-experts submodule at all (not zero-width, entirely absent).

Toy oracle: `dense_mlp_idx: 2`, `num_hidden_layers: 6` -> layers 0,1 dense,
layers 2-5 MoE (`InklingTextConfig.mlp_layer_types` printed as
`['dense','dense','sparse','sparse','sparse','sparse']`, verified at
construction time).

### Naming trap: `intermediate_size` vs `dense_intermediate_size`

The real `config.json`'s `text_config` has **both**
`"intermediate_size": 3072` and `"dense_intermediate_size": 24576`. The
dataclass field `intermediate_size` (default 24576, used by `InklingMLP` for
dense-layer FFN width) gets set to 3072 by ordinary dataclass construction
from the JSON, but then **unconditionally overwritten** by
`__post_init__` (`configuration_inkling.py:125-126`):

```python
if kwargs.get("dense_intermediate_size") is not None:
    self.intermediate_size = kwargs.pop("dense_intermediate_size")
```

So `self.intermediate_size` ends up 24576 (correct for dense FFN width), and
the JSON's `"intermediate_size": 3072` value is discarded — it is a decoy.
Meanwhile the true per-expert MoE width comes from a **separate** field,
`moe_intermediate_size` (dataclass default 3072, `configuration_inkling.py:92`),
which the real `config.json` does not set explicitly at all (its default
3072 already matches, by construction of the checkpoint, not by any JSON
key). **The C engine must not read `intermediate_size` for MoE width.**
Dense FFN width = `dense_intermediate_size` if present, else
`intermediate_size`. MoE per-expert width = `moe_intermediate_size` (not
`intermediate_size`), defaulting to 3072 if the checkpoint's config.json
doesn't carry that key.

## 2. Attention (`InklingAttention`, `modeling_inkling.py:185-282`)

### Per-layer-type shapes

Each layer is either `"hybrid_sliding"` (local, sliding-window) or
`"hybrid"` (global), from `config.layer_types[layer_idx]`
(`InklingAttention.__init__`, `modeling_inkling.py:190`). Which fields drive
which type:

| quantity | local (`hybrid_sliding`) field | global (`hybrid`) field | real value (local / global) |
|---|---|---|---|
| num query heads | `swa_num_attention_heads` | `num_attention_heads` | 64 / 64 (equal in the real model) |
| num KV heads | `swa_num_key_value_heads` | `num_key_value_heads` | 16 / 8 |
| head_dim | `swa_head_dim` | `head_dim` | 128 / 128 |
| rel-bias extent | `sliding_window_size` (**not** a separate field) | `rel_extent` | 512 / 1024 |
| attention window | `sliding_window_size` | none (full causal) | 512 / (unbounded, causal only) |

`layer_types` default derivation (when a checkpoint doesn't pass an explicit
`local_layer_ids`) is `"hybrid_sliding" if i in local_layer_ids else
"hybrid"` where the default `local_layer_ids = {i for i in
range(num_hidden_layers) if (i+1) % 6}` — i.e. every 6th layer (1-indexed)
is global. The real checkpoint's `config.json` passes an explicit
66-element `local_layer_ids` list rather than relying on this default, but
it produces the same pattern: global layers at 0-indexed
`{5,11,17,23,29,35,41,47,53,59,65}` (confirmed against
`docs/gguf-inventory-ud-q2_k_xl.md`'s independently-derived
`attention.sliding_window_pattern`).

GQA expansion group size = `num_heads // num_key_value_heads` (4 for real
local layers, 8 for real global layers). Applied via `repeat_kv` (torch
`repeat_interleave`-equivalent, `modeling_inkling.py:145-154`) inside the
attention interface, i.e. **after** projection, not baked into weights.

### Projections (all `nn.Linear(..., bias=False)` — real `q_bias`/`o_bias`
config fields exist but are never read; bias is unconditionally off)

```
q_proj: hidden_size -> num_heads * head_dim
k_proj: hidden_size -> num_key_value_heads * head_dim
v_proj: hidden_size -> num_key_value_heads * head_dim
r_proj: hidden_size -> num_heads * d_rel          # relative-position content projection, see §3
o_proj: num_heads * head_dim -> hidden_size
```

### Forward order (`InklingAttention.forward`, `modeling_inkling.py:217-282`)

```
q = q_proj(x)                                  # [B,T,H*Dh]
k = k_sconv(k_proj(x))                          # sconv BEFORE reshape into heads, see §5
v = v_sconv(v_proj(x))                          # sconv BEFORE reshape into heads
r = r_proj(x)                                   # [B,T,H*d_rel], -> relative_states

q = q_norm(q.view(B,T,H,Dh)).transpose(1,2)     # per-head RMSNorm, THEN transpose
k = k_norm(k.view(B,T,Hkv,Dh)).transpose(1,2)   # per-head RMSNorm, THEN transpose
v =        v.view(B,T,Hkv,Dh).transpose(1,2)    # NO norm on v

# (cache update happens here in the real code; irrelevant to a
#  from-scratch prefill+one-shot-teacher-forcing oracle run)

position_bias = rel_logits_proj(r, q_positions, kv_positions)   # see §3
# log-scaling, global layers only, see §4

attn_weights = (q @ k^T) * scaling + position_bias + causal_mask   # scaling = 1/head_dim, NOT 1/sqrt(head_dim)
attn_weights = softmax(attn_weights, dim=-1, dtype=float32)
attn_output = attn_weights @ v
attn_output = o_proj(attn_output.reshape(B,T,H*Dh))
```

Critical, easy-to-miss detail: **`self.scaling = 1.0 / self.head_dim`**
(`modeling_inkling.py:198`), not the usual `1/sqrt(head_dim)`. The code
comment explains why: q and k are RMS-normalized per head before the dot
product (`q_norm`/`k_norm`, each `InklingRMSNorm(head_dim, eps=rms_norm_eps)`,
one shared weight vector of length `head_dim` broadcast across all heads —
not a separate weight per head), so the dot-product magnitude scales
differently than raw (unnormalized) attention, and the model was
presumably trained with `1/head_dim`.

`q_norm`/`k_norm` are applied **after** the view into `[..., num_heads,
head_dim]` but **before** `transpose(1,2)` — order matters only for which
axis RMSNorm reduces over (always the last axis, `head_dim`, regardless of
transpose), so this is safe to implement as "normalize each head_dim-length
vector" without tracking the transpose at all.

Softmax is computed in float32 explicitly
(`nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)`,
`eager_attention_forward`, `modeling_inkling.py:177`) even though the oracle
runs entirely in float32 anyway, so this has no additional effect here —
noted because it would matter at real (bf16) scale.

## 3. Banded content-dependent relative position bias

No RoPE. Two learned tensors per attention layer:

- `r_proj.weight` — `nn.Linear(hidden_size, num_heads * d_rel, bias=False)`
  (`modeling_inkling.py:205`). Projects the **query-side content**
  (`hidden_states`, the layer input after `input_layernorm`) into a
  per-head, `d_rel`-dimensional vector — this is genuinely
  content-*dependent*: different query tokens produce different relative
  states, not just a lookup by distance.
- `InklingRelativeLogits.proj` — a raw `nn.Parameter` of shape `[d_rel,
  rel_extent]` (`modeling_inkling.py:129`), a **learned bank of
  distance-vs-bias profiles**, independent of content. `rel_extent` is
  `sliding_window_size` for local layers, `config.rel_extent` for global
  layers (`InklingAttention.__init__`, `modeling_inkling.py:196`) — i.e.
  this tensor's shape genuinely differs by layer type in the real
  checkpoint (`[16,512]` local vs `[16,1024]` global; the GGUF inventory
  sampled a local layer and reported `attn_rel_proj.weight` as `[512,16]`
  reversed-order, consistent).

Exact computation (`InklingRelativeLogits.forward`, `modeling_inkling.py:131-142`):

```
relative_states = r_proj(hidden_states)                      # [B,T,H*d_rel]
relative_states = relative_states.view(B, T, H, d_rel)        # per-head split

rel_logits = einsum('bthd,dr->bthr', relative_states, proj)   # [B,T,H,rel_extent]
rel_logits = rel_logits.transpose(1, 2)                       # [B,H,T,rel_extent]

distance[q,k] = query_position[q] - key_position[k]            # can be negative (future) or large (far past)
gather_index  = clamp(distance, 0, rel_extent - 1)              # [T_q, T_kv], broadcast over B,H

position_bias[b,h,q,k] = rel_logits[b,h,q, gather_index[q,k]]
position_bias[b,h,q,k] = 0.0   if distance[q,k] < 0 or distance[q,k] >= rel_extent
```

In words: for each query position and head, the model computes one
`rel_extent`-long vector of "how much do I favor each backward distance"
(from `r_proj` + the distance-bank `proj`, both content- and
head-dependent), then for each key position looks up the entry at
`clamp(query_pos - key_pos, 0, rel_extent-1)`, and zeroes the bias entirely
for keys in the future or beyond `rel_extent` in the past (this is the
"banded" part — it is a soft, additive band, distinct from the hard causal
mask which is applied separately and always). `position_bias` is added to
`attn_weights` **before** the causal/sliding-window mask and **before**
softmax (`eager_attention_forward`, `modeling_inkling.py:172-177`).

## 4. Attention log-scaling (global layers only)

`InklingAttention.forward`, `modeling_inkling.py:253-261`, gated on `not
self.is_sliding and self.config.log_scaling_n_floor is not None` — **local
(sliding-window) layers never apply this**, regardless of config:

```
effective_n = query_position + 1                              # 1-indexed absolute position
tau = 1.0 + log_scaling_alpha * log(clamp(effective_n / log_scaling_n_floor, min=1.0))
q = q.float() * tau            # scales the QUERY, in float32, before the QK matmul
position_bias = position_bias.float() * tau   # scales the position bias by the SAME tau
```

`tau == 1.0` exactly whenever `effective_n <= log_scaling_n_floor` (real
value 128000). Real value `log_scaling_alpha = 0.1`. Net effect for
positions beyond the floor: both the content term (`q·k`) and the
positional-bias term of `attn_weights` get multiplied by the same scalar
`tau(q_position)` — i.e. it rescales the *entire* pre-softmax score row for
that query position, not just one term.

**Toy-scale gap**: the oracle keeps `log_scaling_n_floor` at the real value
128000 deliberately. The toy sequence is 32 tokens total (max position 31),
so `tau == 1.0` at every position and this code path's *nontrivial* branch
(`tau != 1.0`) is never exercised by `ref_inkling.json`. The C engine's
implementation of the log-scaling formula is therefore **not** validated by
the Phase-0 oracle — implement it from this formula directly, and treat it
as an untested-by-oracle area until a real-scale (or a deliberately-lowered
`log_scaling_n_floor` second fixture) test exists.

## 5. Short convolutions (sconv)

**Four separate depthwise causal Conv1d modules per layer**, not one — this
matches the GGUF's four per-layer tensors
(`shortconv_{attn,k,mlp,v}.weight`, `docs/gguf-inventory-ud-q2_k_xl.md`) and
`InklingTextConfig.__post_init__`'s `self.number_of_conv_states = 4`
(`configuration_inkling.py:129`), one per `conv_idx` 0-3:

| `conv_idx` | module | operates on | placement |
|---|---|---|---|
| 0 | `self_attn.k_sconv` | `k_proj(x)` output, **before** reshape into heads | inside `InklingAttention.forward` |
| 1 | `self_attn.v_sconv` | `v_proj(x)` output, **before** reshape into heads | inside `InklingAttention.forward` |
| 2 | `attn_sconv` (decoder-layer level) | the **output** of the whole attention block (post-`o_proj`) | inside `InklingDecoderLayer.forward`, before the outer residual add |
| 3 | `mlp_sconv` (decoder-layer level) | the **output** of the whole MLP/MoE block | inside `InklingDecoderLayer.forward`, before the outer residual add |

There is **no** sconv on `q` and **no** sconv on the block *input* — only on
K, V, and the two block outputs. Real config field `use_sconv: true` exists
but is **never read** — `InklingAttention.__init__` and
`InklingDecoderLayer.__init__` construct all four `InklingShortConvolution`
instances unconditionally (`modeling_inkling.py:207-212, 560-565`).

### Each sconv module is its own residual block

`InklingShortConvolution.forward` (`modeling_inkling.py:500-543`):

```
residual = x
x = causal_depthwise_conv1d(x, kernel_size=conv_kernel_size)   # see below
x = x + residual
```

I.e. `k_sconv(k_proj(x))` really means `k_proj(x) + conv(k_proj(x))`, not a
plain convolution — the residual is around the conv **inside** the sconv
module, additional to (and inside of) the decoder layer's own outer
residual for `conv_idx` 2 and 3.

`InklingShortConvolution.__init__` (`modeling_inkling.py:484-498`):
`nn.Conv1d(channels, channels, kernel_size=conv_kernel_size, groups=channels,
padding=conv_kernel_size - 1, bias=False)` — **depthwise** (one filter per
channel, no cross-channel mixing) 1-D convolution, real
`sconv_kernel_size: 4`. Computation is forced to float32 internally
(`"Keep the computation in fp32"` comment, `modeling_inkling.py:508-510`;
`InklingPreTrainedModel._keep_in_fp32_modules_strict = ["attn_sconv",
"mlp_sconv", "k_sconv", "v_sconv"]`) — irrelevant for the (already-float32)
oracle, relevant if the real engine ever runs attention/FFN in reduced
precision while keeping sconv exact.

### Causal padding (full-sequence / prefill path, `causal_conv1d_fn`, `modeling_inkling.py:460-480`)

```
padding = kernel_size - 1
out = conv1d(x, weight, padding=padding, groups=channels)   # symmetric zero-pad both sides
out = out[:, :, :seq_len]                                    # keep only the first seq_len outputs
```

Padding `K-1` on both sides then truncating to the first `seq_len` outputs
is algebraically equivalent to left-only causal padding of `K-1` zeros:
output position `t` (0-indexed) is a function of input positions
`t-(K-1) .. t`, with implicit zeros for indices `< 0`. This is the standard
"pad both sides, take a prefix" trick — implement it as explicit left
zero-padding of `K-1` in the C engine; the result is identical and avoids
computing (and discarding) the last `K-1` output positions.

### Incremental decode state (`causal_conv1d_update`, `modeling_inkling.py:441-457`)

Per (layer, `conv_idx`) state = the last `conv_kernel_size - 1` input frames
to that conv, shape `[batch, channels, conv_kernel_size - 1]` where
`channels` is `num_key_value_heads * head_dim` for `conv_idx` 0/1 (K/V — so
this differs between local and global layers, matching each layer's own KV
width) or `hidden_size` for `conv_idx` 2/3 (attn-output/mlp-output, always
the model's full hidden width). Per decode step: concatenate the cached
state with the 1 new input frame -> length `conv_kernel_size` -> run the
conv with `padding=0` (produces exactly 1 output frame) -> the **last**
`conv_kernel_size - 1` frames of the concatenated input become the new
cached state (a sliding window, not an accumulating buffer).

This state lives in the model's `Cache` object; see §8.

## 6. Router pipeline (`InklingTopkRouter.forward`, `modeling_inkling.py:342-377`)

Exact order — **this is the part most likely to be subtly wrong if
re-derived instead of read**:

```
router_logits = hidden_states @ weight.T          # weight: [n_routed_experts + n_shared_experts, hidden_size], NO bias term in this linear
scores = sigmoid(router_logits)                     # "gate_activation: sigmoid" config field is never read -- sigmoid is hardcoded

routed_scores = scores[..., :-n_shared_experts]     # drop the last n_shared_experts columns (the "shared expert sink" logits)
scores_for_choice = routed_scores + e_score_correction_bias   # aux-loss-free bias, SELECTION ONLY (see below)
topk_indices = topk(scores_for_choice, num_experts_per_tok, sorted=False).indices

routed_logits = router_logits[..., :-n_shared_experts]     # RAW (pre-sigmoid, pre-bias) logits
shared_logits = router_logits[..., -n_shared_experts:]      # RAW logits for the shared-expert-sink slots
topk_logits = concat([routed_logits.gather(topk_indices), shared_logits], dim=-1)

topk_log_probs = log_sigmoid(topk_logits)
topk_weights = exp(topk_log_probs - logsumexp(topk_log_probs, dim=-1, keepdim=True))
# ^ this IS the "norm_after_topk" behavior: renormalize sigmoid outputs of
#   the SELECTED routed experts plus the shared-sink slots so they sum to 1,
#   computed only after top-k selection. The "norm_after_topk: true" config
#   field is never read -- there is no alternate "normalize-before-topk"
#   code path in this transformers version; it always normalizes after.

topk_weights = topk_weights * route_scale * global_scale     # global_scale is the per-layer scalar param (ffn_gscale, see §9)

shared_gammas = topk_weights[..., -n_shared_experts:]         # weights for the n_shared_experts shared experts
topk_weights  = topk_weights[..., :num_experts_per_tok]       # final weights for the routed experts
```

The single most important precision point: **`e_score_correction_bias` (the
aux-loss-free router bias) is used ONLY to decide which experts are
selected** (the `topk` call). It is **not** included when computing the
actual mixing weight for the selected experts — that weight comes from
`log_sigmoid`/`logsumexp` over the **raw** (bias-free) logits of the
selected experts plus the shared-sink logits. A C implementation that
folds the bias into the weight computation (not just the selection) will
select the same experts but compute different (wrong) weights.

`weight` (the router's own projection) has shape `[n_routed_experts +
n_shared_experts, hidden_size]` — real `[256+2, 6144] = [258, 6144]`,
confirmed independently by `docs/gguf-inventory-ud-q2_k_xl.md`'s
`blk.N.ffn_gate_inp.weight` shape `[6144, 258]` (GGUF reversed order).
`e_score_correction_bias` has shape `[n_routed_experts]` (real `[256]`,
matches the GGUF's `blk.N.exp_probs_b.bias`). The real on-disk name for
`e_score_correction_bias` is `mlp.gate.bias` (see §9's rename table) — the
transformers-side name is more descriptive but do not expect to find
`e_score_correction_bias` as a substring in the real checkpoint.

`shared_expert_sink: true` is a real config field but, like the others
above, is never read as a conditional — the "sink" behavior (the router
producing `n_shared_experts` extra logit columns, weighted by the same
sigmoid+logsumexp math as the routed experts, and used to gate the shared
experts) is structurally always present whenever `n_shared_experts > 0`.

## 7. MoE FFN (`InklingExperts`, `InklingSharedExperts`, `InklingMoE`)

Activation is SwiGLU (`hidden_act: "silu"`, `ACT2FN["silu"]`): `down(silu(gate(x)) * up(x))`.

### Routed experts (`InklingExperts`, `modeling_inkling.py:302-339`)

Weights, **fused**, as actual `nn.Parameter` tensors (this is the module's
real storage, saved as-is by the oracle — see the "fused vs unfused" note
below):

```
gate_up_proj: [n_routed_experts, 2 * moe_intermediate_size, hidden_size]
down_proj:    [n_routed_experts, hidden_size, moe_intermediate_size]
```

Per selected expert `e`, per token `x` (`hidden_size`-vector):

```
gate_out, up_out = split((gate_up_proj[e] @ x), moe_intermediate_size, dim=0)   # FIRST half = gate, SECOND half = up
h = silu(gate_out) * up_out
y = down_proj[e] @ h
y = y * topk_weight_for_this_(token, e)
```

Contribution across a token's `num_experts_per_tok` selected experts is a
weighted sum (`index_add_` in the reference, order-independent since it's a
sum), added into a `zeros_like(hidden_states)` accumulator
(`modeling_inkling.py:321-337`) — i.e. **only the selected experts
contribute; there is no "all experts, zero-weighted" computation** (an
important cost/precision distinction, not just an implementation detail).

**Real on-disk layout differs from the transformers in-memory layout**: the
checkpoint stores `mlp.experts.w13_weight` (gate and up **interleaved**
along the output-channel axis: pairs of consecutive rows are `(gate_i,
up_i)` for `i` in `0..moe_intermediate_size-1`, NOT the transformers
blocked layout `[gate_0..gate_n, up_0..up_n]`). transformers'
`Interleave(dim=1)` conversion op deinterleaves this on load (see §9's
`Interleave` semantics). **The C engine reading the real checkpoint
directly must de-interleave `w13_weight` itself** (split each consecutive
row-pair into (gate row, up row), then group all gate rows before all up
rows, or equivalently index gate rows at even positions and up rows at odd
positions of the *original* on-disk tensor) — it must not assume a blocked
gate-then-up layout is already on disk. `w2_weight` (down-projection) is a
straight rename, no reordering.

### Shared experts (`InklingSharedExperts`, `modeling_inkling.py:380-405`)

Weights **not fused** (three separate parameters, unlike the routed
experts):

```
gate_proj: [n_shared_experts, moe_intermediate_size, hidden_size]
up_proj:   [n_shared_experts, moe_intermediate_size, hidden_size]
down_proj: [n_shared_experts, hidden_size, moe_intermediate_size]
```

Every token goes through **every** shared expert (no routing/top-k — that's
the point of "shared"), each weighted by its own `gamma` (`shared_gammas`
from §6's router, one weight per shared expert per token):

```
for s in range(n_shared_experts):
    h = silu(gate_proj[s] @ x) * (up_proj[s] @ x)
    out += gamma[s] * (down_proj[s] @ h)
```

(the reference implementation batches this via `bmm` over all
`n_shared_experts` at once and sums; mathematically equivalent to the loop
above.) `InklingMoE.forward` (`modeling_inkling.py:418-425`) adds this
directly to the routed-experts output: `hidden_states = routed_output +
shared_experts_output` — no extra residual, no extra norm, at the
`InklingMoE` level (the decoder layer's own residual and `mlp_sconv` are
applied around the whole `InklingMoE` output, per §5).

**Real on-disk**: `mlp.shared_experts.shared_w13_weight` (interleaved
gate/up, same interleave-then-**also**-chunk pattern as above, since
target is two separate tensors — see §9) and
`mlp.shared_experts.shared_w2_weight` (plain rename to `down_proj`).

### Actual layout of the oracle's `model.safetensors` (corrected, task 0.4)

**Correction to an earlier draft of this section**, which claimed the
oracle saves `InklingExperts.gate_up_proj`/`.down_proj` and
`InklingSharedExperts.gate_proj`/`up_proj`/`down_proj` "exactly as
transformers stores them in memory" — i.e. the transformers-side names and
the in-memory (deinterleaved, three-separate-tensors-for-shared) layout.
That is wrong; verified empirically by dumping the committed
`tools/oracle/tiny/model.safetensors`'s safetensors header while building
the task 0.4 C engine. The oracle's tensor names and layout are the **real
on-disk (SGLang-native) ones from §9's rename table**, the same names used
throughout the rest of this section and never the in-memory transformers
names: `model.llm.layers.N.mlp.experts.w13_weight` /
`...w2_weight` for routed experts (fused 3D, `w13_weight` interleaved
gate/up along dim 1, exactly the "real on-disk layout" paragraph above
describes) and `model.llm.layers.N.mlp.shared_experts.shared_w13_weight` /
`...shared_w2_weight` for shared experts (**also fused and interleaved**,
`[n_shared_experts, 2*moe_intermediate_size, hidden_size]` — on disk the
shared experts' gate/up are combined into one tensor exactly like the
routed experts', not the three-separate-parameters layout that section
correctly describes for the in-memory `InklingSharedExperts` module after
loading). `model.save_pretrained()` evidently runs the rename+`Interleave`
conversion ops from §9 in reverse when writing the checkpoint back out,
rather than serializing the in-memory module tree's own attribute
names/layout as-is.

Practical upshot: the C engine's tensor lookups and the
deinterleave-by-row-index approach documented above and in §9 apply
identically to the toy oracle and the real checkpoint — there is no
toy-fixture-specific layout to special-case anywhere in the loader, which
is a stronger and simpler guarantee than the incorrect claim it replaces.
This also matches how the C engine will eventually need to slice the real
checkpoint's already-3D `ffn_{gate,up,down}_exps` GGUF tensors (per-expert
byte ranges, confirmed contiguous in `docs/gguf-inventory-ud-q2_k_xl.md`
§"Per-expert contiguity").

## 8. Norms, embedding, output head

- **Norm type**: `InklingRMSNorm` everywhere (`modeling_inkling.py:98-116`)
  — `x / sqrt(mean(x^2) + eps) * weight`, computed in float32 internally
  regardless of input dtype, cast back to input dtype at the end. Real
  `rms_norm_eps: 1e-6`, used for every RMSNorm in the model (there is no
  second eps in effect anywhere — `rms_norm_eps_moe_gate` is a real config
  field but is never read; the router has no norm of its own at all, see
  below).
- **Per-decoder-layer norms** (`InklingDecoderLayer.__init__`,
  `modeling_inkling.py:557-558`): `input_layernorm` (pre-attention) and
  `post_attention_layernorm` (pre-MLP/MoE) — standard pre-norm placement,
  each `InklingRMSNorm(hidden_size, eps=rms_norm_eps)`. Real on-disk names:
  `attn_norm` -> `input_layernorm`, `mlp_norm` -> `post_attention_layernorm`
  (§9).
- **`q_norm`/`k_norm`**: per-head `InklingRMSNorm(head_dim, eps=rms_norm_eps)`,
  see §2 — note this eps is the same `rms_norm_eps`, not a separate
  attention-specific eps.
- **No norm inside the router** (`InklingTopkRouter`) despite the real
  `rms_norm_eps_moe_gate` config field existing — that field is unread;
  the router linear operates directly on its input `hidden_states`.
- **Embedding norm**: `InklingTextModel.__init__` creates `self.embed_norm
  = InklingRMSNorm(hidden_size, eps=rms_norm_eps)`
  (`modeling_inkling.py:659`) and applies it unconditionally right after
  the embedding lookup: `inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))`
  (`modeling_inkling.py:682`). Real config field `use_embed_norm: true`
  exists but is never read as a conditional — the norm is always applied.
  **There is no separate embedding *scale*** (no multiply-by-constant on
  the embedding) — despite the `embedding_multiplier` name appearing in
  `InklingTextConfig.attribute_map` (`configuration_inkling.py:57`,
  `"embedding_multiplier": "logits_mup_width_multiplier"`), that alias
  actually controls the **output** scale, not embedding scale — see below.
- **Final norm**: `InklingTextModel.norm`, applied once after all decoder
  layers (`modeling_inkling.py:658, 718`), same `InklingRMSNorm(hidden_size,
  eps=rms_norm_eps)`. Real on-disk name: `model.llm.norm.weight`.
- **Output head / logit scale**: `InklingForConditionalGeneration.forward`
  (`modeling_inkling.py:1280`): `hidden_states = outputs[0] /
  config.text_config.logits_mup_width_multiplier` (real value 24.0)
  **before** the `lm_head` matmul. This is a *division* applied to the
  final hidden state, i.e. functionally a logit-scale (muP-style width
  correction), despite being reachable via the `embedding_multiplier` alias
  name — a naming trap worth flagging explicitly since "embedding
  multiplier" strongly suggests input-side scaling and it is not that. This
  independently confirms `docs/DESIGN.md`'s open question about the GGUF's
  `logit_scale_denom = 24.0` metadata key: same value, same role.
- **`lm_head` is NOT tied to the embedding** — separate `nn.Linear(hidden_size,
  vocab_size, bias=False)` (`modeling_inkling.py:1193`), and
  `InklingForConditionalGeneration._tied_weights_keys = {}` explicitly
  (empty, i.e. no tying) with the comment `"embed" and "unembed" are
  separate tensors in the checkpoints, never tied`
  (`modeling_inkling.py:1181-1182`). Real on-disk names: `model.llm.embed.weight`
  (input embedding) and `model.llm.unembed.weight` (renamed to `lm_head.weight`
  by the conversion table, §9) — genuinely two different tensors on disk.
- **Vocab padding**: `lm_head` produces `vocab_size` logits (real 201024,
  padded for e.g. tensor-parallel-shard divisibility), but
  `unpadded_vocab_size` (real 200058) is smaller. `InklingForConditionalGeneration.forward`
  slices the output: `if unpadded_vocab_size is not None and
  unpadded_vocab_size < logits.shape[-1]: logits = logits[..., :unpadded_vocab_size]`
  (`modeling_inkling.py:1284-1286`). So: the embedding table (`embed_tokens`,
  `nn.Embedding(vocab_size, hidden_size)`) has `vocab_size` rows — the last
  `vocab_size - unpadded_vocab_size` (real: 966) rows exist as real
  parameters (randomly initialized, technically indexable by an
  `input_ids` value in that range) but are **never produced as output**
  (logits are always truncated to `unpadded_vocab_size` columns) and the
  real tokenizer never emits ids in that range as an encoding of any real
  string — they are pure alignment padding on both the embedding and
  unembedding sides, invisible to normal use. The C engine should size its
  output buffer to `unpadded_vocab_size`, not `vocab_size`, and can ignore
  embedding rows `>= unpadded_vocab_size` entirely for any input it will
  realistically see from the real tokenizer (but must still allocate
  `vocab_size` embedding rows if loading the real checkpoint's embedding
  tensor as-is, since the tensor's on-disk shape is the padded one). Toy
  oracle mirrors this with `vocab_size: 512`, `unpadded_vocab_size: 480` (a
  proportional 6.25% gap, matching the real ~0.48% gap in spirit if not
  exact ratio — chosen large enough at toy scale to be a meaningful test,
  since the real gap is 966/201024 which would round to 0 padded ids at
  vocab 512).

## 9. Real checkpoint tensor names: full rename table + the two GGUF identifications

Source: `transformers/conversion_mapping.py`, the `"inkling_mm_model"`
entry (the table transformers itself uses to load the real checkpoint into
the module tree described in §§1-8). All patterns are regexes matched
against real on-disk safetensors keys (fetched from
`thinkingmachines/Inkling`'s `model.safetensors.index.json`, 2026-07-19),
`model.llm.layers.N.<pattern>`:

| real checkpoint (on disk) | transformers module name | notes |
|---|---|---|
| `model.llm.embed.weight` | `model.language_model.embed_tokens.weight` | input embedding, `[vocab_size, hidden_size]` |
| `model.llm.norm.weight` | `model.language_model.norm.weight` | final norm |
| `model.llm.unembed.weight` | `lm_head.weight` | output head, NOT tied to embed |
| `attn.wq_du.weight` | `self_attn.q_proj.weight` | |
| `attn.wk_dv.weight` | `self_attn.k_proj.weight` | |
| `attn.wv_dv.weight` | `self_attn.v_proj.weight` | |
| `attn.wr_du.weight` | `self_attn.r_proj.weight` | **= GGUF `attn_r.weight`, see below** |
| `attn.wo_ud.weight` | `self_attn.o_proj.weight` | |
| `attn.q_norm.weight` | `self_attn.q_norm.weight` | |
| `attn.k_norm.weight` | `self_attn.k_norm.weight` | |
| `attn.k_sconv.weight` | `self_attn.k_sconv.conv1d.weight` | |
| `attn.v_sconv.weight` | `self_attn.v_sconv.conv1d.weight` | |
| `attn.rel_logits_proj.proj` | `self_attn.rel_logits_proj.proj` | the **small** `[d_rel, rel_extent]` distance bank, NOT attn_r (see below) |
| `attn_sconv.weight` | `attn_sconv.conv1d.weight` | decoder-layer level, `conv_idx=2` |
| `mlp_sconv.weight` | `mlp_sconv.conv1d.weight` | decoder-layer level, `conv_idx=3` |
| `attn_norm.weight` | `input_layernorm.weight` | |
| `mlp_norm.weight` | `post_attention_layernorm.weight` | |
| `mlp.global_scale` (dense layers) | `mlp.global_scale` | **= GGUF `ffn_gscale.weight`, see below** |
| `mlp.gate.global_scale` (MoE layers) | `mlp.gate.global_scale` | **= GGUF `ffn_gscale.weight`, see below** |
| `mlp.w13_dn.weight` (dense layers) | `mlp.gate_proj.weight` + `mlp.up_proj.weight` | interleaved-pair -> deinterleave + chunk(dim=0), see below |
| `mlp.w2_md.weight` (dense layers) | `mlp.down_proj.weight` | plain rename |
| `mlp.gate.weight` (MoE layers) | `mlp.gate.weight` | router projection, `[n_routed+n_shared, hidden]` |
| `mlp.gate.bias` (MoE layers) | `mlp.gate.e_score_correction_bias` | aux-loss-free router bias, `[n_routed_experts]` |
| `mlp.experts.w13_weight` (MoE layers) | `mlp.experts.gate_up_proj` | interleaved -> deinterleave(dim=1), stays fused |
| `mlp.experts.w2_weight` (MoE layers) | `mlp.experts.down_proj` | plain rename |
| `mlp.shared_experts.shared_w13_weight` (MoE layers) | `mlp.shared_experts.gate_proj` + `.up_proj` | interleaved-pair -> deinterleave + chunk(dim=1) |
| `mlp.shared_experts.shared_w2_weight` (MoE layers) | `mlp.shared_experts.down_proj` | plain rename |
| `model.audio.*` | `model.audio_tower.*` | out of SEPIA's text-only scope |
| `model.visual*` | `model.vision_tower*` | out of SEPIA's text-only scope |
| `model.mtp.*` | *(not loaded)* | `InklingPreTrainedModel._keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]` — see §10 |

`N` (the layer index) is preserved as-is on both sides — **`model.llm.layers.N`
maps 1:1 to `model.language_model.layers.N`** (and to the GGUF's `blk.N`,
confirmed by `docs/gguf-inventory-ud-q2_k_xl.md`'s tensor-name-derived
per-layer facts lining up exactly with this same `N`; there is no offset or
reordering between the three naming schemes).

**Interleave semantics** (`transformers/core_model_loading.py:184-208`,
class `Interleave`): reshape the target dim from size `N` to `[N//2, 2]`
(grouping *consecutive pairs*), transpose those two axes, reshape back to
size `N`. Concretely, for a `[..., 2*d, ...]` tensor storing alternating
`(gate_0, up_0, gate_1, up_1, ..., gate_{d-1}, up_{d-1})` along that axis,
this operation produces the blocked layout `(gate_0, ..., gate_{d-1}, up_0,
..., up_{d-1})`. **This is the deinterleave the C engine must implement by
hand when reading `w13_dn.weight` / `w13_weight` / `shared_w13_weight`
directly from the real checkpoint**: row (or output-channel) `2i` on disk is
`gate_i`, row `2i+1` is `up_i` — not "first half gate, second half up" as
stored on disk (that blocked form only exists after this conversion, inside
the transformers module, not in the checkpoint file).

### Extra identification 1: `blk.N.attn_r.weight` (GGUF), shape `[6144, 1024]`

**= `self_attn.r_proj.weight`** (transformers name) **= `attn.wr_du.weight`**
(real on-disk name), the content projection described in §3:
`nn.Linear(hidden_size, num_heads * d_rel, bias=False)`. Shape check: real
`hidden_size=6144`, `num_heads=64` (both local and global, since
`num_attention_heads == swa_num_attention_heads == 64` in the real model),
`d_rel=16` -> weight shape `[num_heads*d_rel, hidden_size] = [1024, 6144]`
(`nn.Linear` weight convention is `[out_features, in_features]`). GGUF
stores 2D tensor dims in reversed (`ne[0]`=fastest/in-dim first) order,
giving `[6144, 1024]` — an exact match, and the shape is layer-type
independent (both local and global layers have `num_heads=64` in the real
model), consistent with the GGUF inventory's "every layer, same shape"
observation.

**This corrects the brief's working hypothesis.** The brief (written before
this code was read) guessed the real on-disk name was
`attn.rel_logits_proj.proj` — that name *does* exist in the real
checkpoint, but it names a **different, much smaller** tensor (the
`[d_rel, rel_extent]` distance bank from §3, e.g. `[16, 512]` on a local
layer, matching the GGUF's already-identified `attn_rel_proj.weight`
`[512, 16]`). The tensor that actually matches `attn_r`'s `[6144, 1024]`
shape is `attn.wr_du.weight` (-> `r_proj.weight`), a sibling tensor, not
`rel_logits_proj.proj`.

### Extra identification 2: `blk.N.ffn_gscale.weight` (GGUF), scalar, every layer

**= `mlp.global_scale`** on dense layers (0 and 1 in the real model) **or
`mlp.gate.global_scale`** on MoE layers (2-65) — both are literally named
`global_scale` in the transformers module tree
(`InklingMLP.global_scale`, `modeling_inkling.py:295`;
`InklingTopkRouter.global_scale`, `modeling_inkling.py:353`), each a
`nn.Parameter(torch.ones(1))`. The real `config.json`'s `use_global_scale:
true` field is the (unread-as-conditional, but namewise unambiguous) flag
for this mechanism — `global_scale` -> `gscale` is a direct
llama.cpp-converter abbreviation, and the GGUF's "one scalar per layer,
dense and MoE alike" description matches exactly (dense layers get the
`InklingMLP` version, MoE layers get the router's version — same concept,
different owning module, unified under one GGUF tensor name because both
sides are, mathematically, "the final per-layer scale on this layer's FFN
contribution").

Usage differs by layer kind, which the C engine must get right:

- **Dense layers** (`InklingMLP.forward`, `modeling_inkling.py:297-299`):
  multiplies the **FFN output**: `return down_proj(silu(gate_proj(x)) *
  up_proj(x)) * global_scale`.
- **MoE layers** (`InklingTopkRouter.forward`, §6): multiplies the
  **routing weights**, alongside `route_scale`:
  `topk_weights = topk_weights * route_scale * global_scale`, applied
  **before** the split into `shared_gammas` / final `topk_weights` — so it
  affects both the routed-expert and shared-expert contributions (via
  `shared_gammas`), not the expert FFN output directly.

## 10. MTP (documentation only — not exercised by the oracle)

**transformers 5.14.1 has no MTP module implementation for Inkling at all.**
This was verified by reading the entirety of `modeling_inkling.py` and
`modular_inkling.py`: the only MTP-related code across both files is config
plumbing (`InklingTextConfig.num_mtp_layers`, `.mtp_local_layer_ids`,
`.chain_hidden_post_norm`, `.mtp_hidden_states_first`, and the derived
properties `mtp_layer_types` / `mtp_mlp_layer_types`,
`configuration_inkling.py:106-149`) and one load-time ignore regex,
`InklingPreTrainedModel._keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]`
(`modeling_inkling.py:609`). There is no `InklingMtpLayer` class, no MTP
forward path, and `InklingConfig.__post_init__` only *reads* an
`mtp_config` block (`configuration_inkling.py:239-244`) to populate the
text config's MTP fields — it never constructs anything from them. This
means:

- **This is not a SEPIA choice or a workaround** — it's flagged per the
  team lead's instruction to report unexpected transformers behavior rather
  than silently work around it. Loading the real
  `thinkingmachines/Inkling` checkpoint via
  `InklingForConditionalGeneration.from_pretrained(...)` will *silently
  drop* all `model.mtp.*` weights (matched by the ignore regex) and produce
  a model with no speculative-decoding capability at all, even though the
  checkpoint ships MTP weights.
- The oracle's toy config passes no `mtp_config`/`num_mtp_layers` at all —
  consistent with the brief's instruction to omit MTP from the tiny
  checkpoint, and also simply necessary, since there is no MTP module for
  the toy weights to populate even if requested.
- **Checkpoint-side structure** (documentation only, from
  `model.safetensors.index.json`, real checkpoint, and consistent with
  `docs/DESIGN.md`): 8 MTP layers live in a **separate** shard,
  `mtp.safetensors`, under keys `model.mtp.layers.{0-7}.*`, **not** among
  the 108 main shards. Real `mtp_config`: `num_nextn_predict_layers: 8`,
  `chain_hidden_post_norm: false`, `local_layer_ids: [0,2,4,5,6,7]` (6 of
  the 8 MTP layers are local/SWA, 2 are global — independent of the main
  model's local/global pattern). Each MTP layer is a **full transformer
  block** with its own complete attention stack (`transformer_block.attn.*`
  — same `wq_du`/`wk_dv`/`wv_dv`/`wr_du`/`wo_ud`/`q_norm`/`k_norm`/
  `k_sconv`/`v_sconv`/`rel_logits_proj` set as a main-model layer),
  `transformer_block.attn_sconv`/`mlp_sconv`/`attn_norm`/`mlp_norm`, and
  `transformer_block.mlp.{w13_dn,w2_md,global_scale}` (MTP layers are
  **always dense MLP**, never MoE — `mtp_mlp_layer_types` property,
  `configuration_inkling.py:145-149`, unconditionally returns `["dense"] *
  num_mtp_layers`), **plus** three tensors with no main-layer analog:
  `input_proj.weight`, `embed_norm.weight`, `hidden_norm.weight` (their
  exact wiring — how a draft layer combines the previous layer's hidden
  state with a newly embedded token, the DeepSeek-MTP-style "chain" pattern
  implied by `chain_hidden_post_norm` — is not resolvable from this
  transformers version since no forward-pass code exists to read; a future
  phase implementing MTP will need to consult the original
  thinkingmachines/SGLang reference implementation directly, not
  transformers).

## 11. Cache / state objects for incremental decode

Not exercised by the oracle itself (§ "Testing" in the task report: the
oracle does one `generate()` call, which uses caching internally, plus one
**separate, uncached** teacher-forcing forward — the oracle's saved
`ref_inkling.json` doesn't expose cache internals, only final token ids).
Documented here because task 0.4's C engine will need equivalent state for
decode performance, and because understanding *what* `generate()` uses
internally explains why the oracle's two code paths (cached incremental
decode inside `generate()`, vs. one uncached full-sequence forward) are a
meaningful cross-check (§"generate() vs teacher-forcing" in the task
report).

`InklingTextModel.forward` creates `past_key_values = DynamicCache(config=self.config)`
when none is passed and `use_cache` is true (`modeling_inkling.py:684-685`).
`DynamicCache.__init__` inspects `config.layer_types` (`"hybrid"` /
`"hybrid_sliding"`, real per-layer list) and, for any layer whose type is in
`("conv", "linear_attention", "hybrid", "hybrid_sliding")`, allocates
`number_of_states = config.number_of_conv_states` (= 4, unconditionally set
in `InklingTextConfig.__post_init__`) conv-state slots for that layer
(`cache_utils.py:1637-1648`) — one slot per `conv_idx` (§5's table). Each
conv-state slot is a tensor of shape `[batch, channels, conv_kernel_size -
1]` as described in §5, lazily allocated on first use
(`DynamicLayer`/`update_conv_state`, `cache_utils.py:940-1010`).

Standard KV cache: each attention layer also gets ordinary `[batch,
num_key_value_heads, seq_len, head_dim]` key/value tensors (that layer's
own `num_key_value_heads`/`head_dim` — differs by local/global per §2).
`Cache.get_mask_sizes(query_length, layer_idx)` /
`Cache.get_query_offset(layer_idx)` (`cache_utils.py:1483-1518`) are what
`InklingAttention.forward` (`modeling_inkling.py:238-249`) queries to
determine, per call, how many KV positions exist and at what offset — the
C engine's own KV-cache bookkeeping needs the equivalent of "how many past
tokens does this specific layer have cached" and "what absolute position is
the current query at", which for Inkling's hybrid layers requires per-layer
tracking (a local/SWA layer's cache can be truncated to `sliding_window_size`
without losing correctness, since positions beyond the window can never be
attended to anyway; a global layer's cache cannot be truncated this way).
`has_previous_state(layer_idx, state_idx)` (`cache_utils.py:1453-1481`) is
what `InklingShortConvolution.forward` (`modeling_inkling.py:517-521`) uses
to pick the single-token fused-update path (`causal_conv1d_update`) versus
the full-sequence path (`causal_conv1d_fn`) — functionally, "is this a
decode step with an existing conv state, or a prefill/first call".

A generic `MtpCache(DynamicCache)` class exists in `cache_utils.py:2019`
(used by other, unrelated MTP-capable models in the transformers codebase)
but is never referenced anywhere in `modeling_inkling.py`/`modular_inkling.py`
— consistent with §10's finding that Inkling's transformers port has no MTP
forward path to use it.

## 12. Toy config used by `tools/make_oracle.py`

Every field name below is copied verbatim from the real `config.json`'s
`text_config` (fetched from `thinkingmachines/Inkling`, 2026-07-19); only
values are shrunk. See `tools/make_oracle.py`'s `build_text_config_dict()`
(and `build_vision_config_dict()` / `build_audio_config_dict()`, needed
only because `InklingForConditionalGeneration.__init__` unconditionally
constructs vision/audio towers — never exercised, since the oracle only
ever passes `input_ids`) for the authoritative, commented list — this table
is a summary, not a duplicate source of truth:

| field | real value | toy value |
|---|---:|---:|
| `hidden_size` | 6144 | 128 |
| `num_hidden_layers` | 66 | 6 |
| `vocab_size` | 201024 | 512 |
| `unpadded_vocab_size` | 200058 | 480 |
| `num_attention_heads` / `swa_num_attention_heads` | 64 / 64 | 4 / 4 |
| `num_key_value_heads` (global) | 8 | 2 |
| `swa_num_key_value_heads` (local) | 16 | 2 |
| `head_dim` / `swa_head_dim` | 128 / 128 | 16 / 16 |
| `sliding_window_size` | 512 | 8 |
| `local_layer_ids` | 55 of 66 (every-6th-excluded) | `[0,1,2,3,4]` of 6 (layer 5 global) |
| `d_rel` | 16 | 16 (kept) |
| `rel_extent` | 1024 | 16 |
| `log_scaling_n_floor` | 128000 | 128000 (kept — deliberately never triggers, §4) |
| `log_scaling_alpha` | 0.1 | 0.1 (kept) |
| `dense_mlp_idx` | 2 | 2 (kept) |
| `dense_intermediate_size` | 24576 | 256 |
| `moe_intermediate_size` (real value, not a config.json key) | 3072 | 64 |
| `sconv_kernel_size` | 4 | 4 (kept) |
| `n_routed_experts` | 256 | 8 |
| `num_experts_per_tok` | 6 | 2 |
| `n_shared_experts` | 2 | 2 (kept) |
| `route_scale` | 8.0 | 8.0 (kept) |
| `logits_mup_width_multiplier` | 24.0 | 24.0 (kept) |

All `NOTE: not read by transformers 5.14.1` fields (`gate_activation`,
`use_gate_bias`, `norm_after_topk`, `use_sconv`, `use_embed_norm`,
`shared_expert_sink`, `use_global_scale`, `q_bias`, `o_bias`,
`rms_norm_eps_moe_gate`, `final_logit_softcapping`) are passed at their real
values purely for `config.json` fidelity (so a diff against the real
checkpoint's config shows only shrunk numbers, not missing/renamed keys) —
they have zero effect on the toy model's forward pass, per §§1-8's per-field
notes.

## 13. Toy-scale gaps: what `ref_inkling.json` does NOT exercise

Explicit, not left implicit, per the brief:

- **Attention log-scaling** (§4): `tau` is always exactly 1.0 at 32 tokens.
  The nontrivial branch of the log-scaling formula is unvalidated by this
  oracle.
- **Sliding-window truncation actually mattering**: toy
  `sliding_window_size=8` with a 32-token sequence means local layers *do*
  hit the window boundary during the prompt+generation (unlike, say, a
  6-token-only test), so this much is exercised — but the KV-cache
  *memory-truncation* optimization path (§11, `DynamicCache` dropping KV
  entries beyond the window rather than just masking them) is an
  orthogonal implementation choice the C engine doesn't have to replicate;
  only the *masking* behavior (attend only to keys within the window) is
  observable in `ref_inkling.json` and must match.
- **The vocab padding gap** is exercised (unpadded_vocab_size < vocab_size,
  §8) but at a much larger relative gap than real (6.25% vs 0.48%) — purely
  because the real gap rounds to under 1 id at vocab 512; the *mechanism*
  (slice logits, don't slice embedding-lookup range) is still validated.
- **MTP**: not present in the toy checkpoint or the oracle's forward passes
  at all (§10) — this is a full scope exclusion for Phase 0, not a scale
  artifact.
- **Multimodal** (vision/audio towers): constructed (required by
  `InklingForConditionalGeneration.__init__`) but never invoked — no
  `pixel_values`/`audio_input_ids` are passed. Their forward code is
  entirely unexercised by the oracle.
- **bf16-specific behavior**: the oracle and the planned C engine are both
  float32-only (per the brief), so nothing about bf16 casting boundaries
  (e.g. `_keep_in_fp32_modules_strict` for sconv, softmax's explicit
  float32 upcast) is a meaningful test here — those boundaries only matter
  once SEPIA runs attention/FFN math in reduced precision at real scale.

## 14. transformers quirks hit while building the oracle

Per the team lead's instruction to flag rather than silently work around:

- **`scipy` is a real, undeclared-in-the-brief dependency.**
  `InklingVisionModel.__init__` -> `plan_out_scales()`
  (`modeling_inkling.py:887-946`) calls `scipy.optimize.linear_sum_assignment`
  whenever `num_hidden_layers < scales.shape[0]` (true for any small toy
  vision config with more than ~1-2 layers of "scale" resolution) — this is
  the real code path (not something SEPIA's toy config forces), it is just
  more likely to trigger at small `num_hidden_layers`. Installed via `uv
  pip install --python .venv/bin/python scipy` (now in `.venv`; not
  listed in the original brief's dependency set of
  torch+transformers+safetensors+numpy). This is a real, exercised branch
  of Inkling's actual vision-tower code, not a workaround — installing the
  dependency it legitimately needs was the correct choice rather than
  avoiding that code path.
- **`plan_out_scales()` has a real, pre-existing dtype bug at
  `temporal_patch_size=1`**: `prime_factors(1)` returns `[]`, so
  `torch.cumprod(torch.tensor([]), dim=0)` is an empty **float32** tensor
  (torch's default dtype for an empty Python-list-derived tensor), and
  `torch.cat`-ing it (even though it contributes zero rows) upcasts the
  whole `scales` tensor to float32, which later fails a `nn.Linear(...,
  size_from_scales_tensor, ...)` call expecting Python ints (`TypeError:
  empty(): argument 'size' failed to unpack the object... type must be
  tuple of ints, but got Tensor`). Confirmed independent of SEPIA's toy
  dims by calling `plan_out_scales` directly with various `(temporal,
  patch, n_layers)` triples. **Worked around by using
  `temporal_patch_size=2`** (the real model's actual value) instead of
  trying to shrink it to 1 — not a modification to the architecture under
  test, just avoiding an input value that happens to trip an unrelated,
  pre-existing bug in vision-tower code the oracle never uses anyway. Not
  filed upstream as part of this task; worth a `transformers` issue if
  vision-tower toy configs become relevant again.
- **Config field over-acceptance**: `InklingTextConfig`/`InklingVisionConfig`/
  `InklingAudioConfig` are `@strict`-decorated dataclasses
  (`huggingface_hub.dataclasses.strict`) but silently accept and store
  arbitrary extra kwargs not declared as dataclass fields (verified
  empirically: `InklingTextConfig(gate_activation="sigmoid", ...)`
  succeeds and `config.gate_activation` reads back `"sigmoid"`) — this is
  how all the `NOTE: not read by transformers 5.14.1` fields throughout
  this document get into the toy config without raising, and is also how
  the real checkpoint's config.json carries them. Not a bug, just worth
  knowing `@strict` here means "typed", not "closed to unknown keys".
- **`local_layer_ids`/attribute-map aliases are bidirectional**: passing
  `model_max_length=X` as a kwarg correctly sets
  `config.max_position_embeddings == X` (and vice versa for reading), same
  for `sconv_kernel_size` -> `conv_kernel_size` and `n_layers` ->
  `num_hidden_layers` (vision config) — verified empirically, used
  throughout the toy config to match the real checkpoint's exact key
  choices per field (some real fields use the alias name, e.g.
  `model_max_length`/`sconv_kernel_size`; others use the canonical name,
  e.g. `sliding_window_size`/`n_routed_experts` — both work, and the toy
  config matches whichever the real checkpoint actually uses, field by
  field).

## 15. C engine numerical policy (task 0.4, controller-approved)

Applies to every reduction and matmul-shaped computation in `src/sepia.c`,
stated here so later kernel work (Metal, quantization) knows the reference
bar it needs to stay within:

- **Storage**: float32 throughout — weights, activations, KV cache, sconv
  state. No bf16/f16 casting boundaries exist in this engine (the toy
  oracle and the real checkpoint's eventual dequantized activations are
  both float32 at this layer).
- **Accumulation**: every reduction (RMSNorm's mean-of-squares, every
  matmul/Linear row as a dot product, softmax's max-subtracted exp-sum, the
  MoE router's `logsumexp`, and the attention-weights-times-V weighted sum)
  accumulates in `double` and rounds to `float32` only on write. This is a
  deliberate choice, not an oversight: ATen's actual CPU reduction kernels
  for float32 tensors accumulate in `double` internally
  (`acc_type<float, DeviceType::CPU> = double`, the standard ATen
  convention for `sum`/`mean`/`norm`-shaped ops), so double accumulation in
  the C engine is the *more* faithful match to real "torch CPU float32
  semantics," not a departure from it. Matmul-shaped ops specifically
  (`nn.Linear`) run through BLAS on the torch side, which stays natively
  float32 — the C engine's double accumulation there is strictly *more*
  precise than BLAS's own summation order, not an attempt to replicate it
  bit-for-bit. What's under test against the oracle is the forward pass's
  linear-algebra *structure* (which tensor multiplies which, residual
  placement, masking, the router's selection-vs-weighting split) — not one
  specific kernel's floating-point summation order, which is exactly the
  numerical-precision-strategy latitude task 0.4's brief left to the
  implementer.
- **Validated, not just argued**: this policy was checked against a fresh
  Python forward pass over the same 32-token sequence
  (`tools/dump_activations.py --compare`), giving a max absolute logit
  difference of ~1.1e-8 at the final position (~1.5e-8 across all 32
  positions) — float32 machine-epsilon-scale noise, with prefill 32/32 and
  decode 20/20 token-exact and no argmax near-ties encountered. Approved by
  the controller reviewing task 0.4 on that evidence.
