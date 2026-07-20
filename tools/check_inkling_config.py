#!/usr/bin/env python3
"""Cross-checks docs/inkling-config.json against GGUF part 1 metadata.

Local-only (needs the real weights, not fetched in CI): loads the hand
-written config.json that the C engine's config_load (src/sepia.c:444)
reads for the real model, and asserts every field that has a GGUF-derivable
ground truth actually matches it. Fields with no GGUF source at all (router/
norm/bias toggles the real transformers module never reads, MTP scheduling)
are listed in NOT_IN_GGUF with a comment pointing at their architecture-notes.md
provenance instead -- they are documented, not asserted, because there is
nothing in the GGUF to assert them against.

Python 3 stdlib only (imports gguf_inspect, nothing else). Usage:
    python3 tools/check_inkling_config.py \
        [--config docs/inkling-config.json] \
        [--gguf weights/inkling-gguf/UD-Q2_K_XL/inkling-UD-Q2_K_XL-00001-of-00008.gguf]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_inspect import LocalSource, parse_gguf_header  # noqa: E402

# (config path in text_config, gguf key, transform applied to the raw gguf value)
CHECKS = [
    ("hidden_size", "inkling.embedding_length", int),
    ("num_hidden_layers", "inkling.block_count", int),
    ("vocab_size", "inkling.vocab_size", int),
    ("unpadded_vocab_size", "inkling.unpadded_vocab_size", int),
    ("num_attention_heads", "inkling.attention.head_count", int),
    ("head_dim", "inkling.attention.key_length", int),
    ("sliding_window_size", "inkling.attention.sliding_window", int),
    ("d_rel", "inkling.d_rel", int),
    ("rel_extent", "inkling.rel_extent", int),
    ("conv_kernel_size", "inkling.shortconv_kernel", int),
    ("intermediate_size", "inkling.feed_forward_length", int),
    ("moe_intermediate_size", "inkling.expert_feed_forward_length", int),
    ("n_routed_experts", "inkling.expert_count", int),
    ("num_experts_per_tok", "inkling.expert_used_count", int),
    ("n_shared_experts", "inkling.expert_shared_count", int),
    ("route_scale", "inkling.expert_weights_scale", float),
    ("logits_mup_width_multiplier", "inkling.logit_scale_denom", float),
    ("log_scaling_alpha", "inkling.log_scaling_alpha", lambda v: round(float(v), 6)),
    ("log_scaling_n_floor", "inkling.log_scaling_n_floor", int),
    ("max_position_embeddings", "inkling.context_length", int),
    ("bos_token_id", "tokenizer.ggml.bos_token_id", int),
]

# text_config fields with no GGUF metadata key at all -- the real transformers
# module (5.14.1) either hardcodes the behavior these fields would toggle or
# never reads them (verified in docs/architecture-notes.md). Documented here,
# not asserted, since there is nothing in the GGUF to compare them against.
NOT_IN_GGUF = {
    "gate_activation": "architecture-notes.md sec.6 (Router pipeline): sigmoid is hardcoded in InklingTopkRouter, field is never read",
    "use_gate_bias": "architecture-notes.md sec.6 (Router pipeline)",
    "norm_after_topk": "architecture-notes.md sec.6 (Router pipeline): renormalize-after-topk is unconditional",
    "shared_expert_sink": "architecture-notes.md sec.7 (MoE FFN / shared experts)",
    "use_embed_norm": "architecture-notes.md sec.8 (Norms, embedding, output head)",
    "use_global_scale": "architecture-notes.md sec.8 (Norms, embedding, output head)",
    "use_sconv": "architecture-notes.md sec.5 (Short convolutions): real config field but unconditional in the real model",
    "q_bias": "architecture-notes.md sec.2 (Attention projections: all nn.Linear(..., bias=False))",
    "o_bias": "architecture-notes.md sec.2 (Attention projections: all nn.Linear(..., bias=False))",
    "mtp_hidden_states_first": "architecture-notes.md sec.10 (MTP, documentation only -- not exercised)",
    "mtp_local_layer_ids": "architecture-notes.md sec.10 (MTP, documentation only -- not exercised)",
    "num_mtp_layers": "architecture-notes.md sec.10 (MTP, documentation only -- not exercised)",
}


def _fmt(v):
    return repr(v)


def check_scalars(tc, md, errors):
    n_ok = 0
    for path, gguf_key, transform in CHECKS:
        if path not in tc:
            errors.append(f"text_config.{path}: missing from config.json")
            continue
        if gguf_key not in md:
            errors.append(f"text_config.{path}: GGUF metadata missing key {gguf_key!r}")
            continue
        try:
            expected = transform(md[gguf_key])
        except (TypeError, ValueError) as exc:
            errors.append(f"text_config.{path}: could not transform gguf {gguf_key!r}={md[gguf_key]!r}: {exc}")
            continue
        actual = tc[path]
        if isinstance(expected, float) or isinstance(actual, float):
            ok = abs(float(actual) - float(expected)) < 1e-6
        else:
            ok = actual == expected
        if not ok:
            errors.append(
                f"text_config.{path}={_fmt(actual)} does not match {gguf_key}={_fmt(md[gguf_key])} "
                f"(transformed={_fmt(expected)})"
            )
        else:
            n_ok += 1
    return n_ok


def check_layer_pattern(tc, md, errors):
    """layer_types / local_layer_ids <-> inkling.attention.sliding_window_pattern,
    both directions."""
    n_ok = 0
    if "inkling.attention.sliding_window_pattern" not in md:
        errors.append("GGUF metadata missing key 'inkling.attention.sliding_window_pattern'")
        return n_ok
    swp = md["inkling.attention.sliding_window_pattern"]
    n = tc.get("num_hidden_layers")

    layer_types = tc.get("layer_types")
    if layer_types is None:
        errors.append("text_config.layer_types: missing from config.json")
    elif len(layer_types) != len(swp):
        errors.append(
            f"text_config.layer_types has {len(layer_types)} entries, "
            f"inkling.attention.sliding_window_pattern has {len(swp)}"
        )
    else:
        mismatches = [
            i for i in range(len(swp))
            if (layer_types[i] == "hybrid_sliding") != bool(swp[i])
        ]
        if mismatches:
            errors.append(
                f"text_config.layer_types disagrees with inkling.attention.sliding_window_pattern "
                f"at indexes {mismatches}"
            )
        else:
            n_ok += 1

    local_layer_ids = tc.get("local_layer_ids")
    if local_layer_ids is None:
        errors.append("text_config.local_layer_ids: missing from config.json")
    else:
        expected_local = {i for i, is_sliding in enumerate(swp) if is_sliding}
        actual_local = set(local_layer_ids)
        if actual_local != expected_local:
            missing = sorted(expected_local - actual_local)
            extra = sorted(actual_local - expected_local)
            errors.append(
                "text_config.local_layer_ids does not match the sliding (True) indexes of "
                f"inkling.attention.sliding_window_pattern: missing={missing} extra={extra}"
            )
        elif len(local_layer_ids) != len(actual_local):
            errors.append("text_config.local_layer_ids: contains duplicate entries")
        else:
            n_ok += 1

    return n_ok


def check_head_count_kv(tc, md, errors):
    """inkling.attention.head_count_kv: 16 on sliding layers (swa_num_key_value_heads),
    8 on global layers (num_key_value_heads)."""
    n_ok = 0
    if "inkling.attention.head_count_kv" not in md:
        errors.append("GGUF metadata missing key 'inkling.attention.head_count_kv'")
        return n_ok
    if "inkling.attention.sliding_window_pattern" not in md:
        errors.append("GGUF metadata missing key 'inkling.attention.sliding_window_pattern' "
                      "(needed to check head_count_kv per layer)")
        return n_ok
    hckv = md["inkling.attention.head_count_kv"]
    swp = md["inkling.attention.sliding_window_pattern"]
    if len(hckv) != len(swp):
        errors.append(
            f"inkling.attention.head_count_kv has {len(hckv)} entries, "
            f"inkling.attention.sliding_window_pattern has {len(swp)}"
        )
        return n_ok

    swa_kv = tc.get("swa_num_key_value_heads")
    global_kv = tc.get("num_key_value_heads")
    if swa_kv is None:
        errors.append("text_config.swa_num_key_value_heads: missing from config.json")
        return n_ok
    if global_kv is None:
        errors.append("text_config.num_key_value_heads: missing from config.json")
        return n_ok

    bad = []
    for i, (kv, is_sliding) in enumerate(zip(hckv, swp)):
        expect = swa_kv if is_sliding else global_kv
        if int(kv) != expect:
            bad.append((i, int(kv), expect))
    if bad:
        errors.append(
            "inkling.attention.head_count_kv disagrees with swa_num_key_value_heads/"
            f"num_key_value_heads at (index, actual, expected)={bad}"
        )
    else:
        n_ok += 2  # accounts for both swa_num_key_value_heads and num_key_value_heads
    return n_ok


def check_dense_block_count(tc, md, errors):
    n_ok = 0
    if "inkling.dense_block_count" not in md:
        errors.append("GGUF metadata missing key 'inkling.dense_block_count'")
        return n_ok
    mlt = tc.get("mlp_layer_types")
    if mlt is None:
        errors.append("text_config.mlp_layer_types: missing from config.json")
        return n_ok
    leading_dense = 0
    for t in mlt:
        if t != "dense":
            break
        leading_dense += 1
    gguf_dbc = int(md["inkling.dense_block_count"])
    if leading_dense != gguf_dbc:
        errors.append(
            f"text_config.mlp_layer_types has {leading_dense} leading 'dense' entries, "
            f"inkling.dense_block_count={gguf_dbc}"
        )
    elif mlt.count("dense") != leading_dense:
        errors.append(
            f"text_config.mlp_layer_types has {mlt.count('dense')} 'dense' entries total but only "
            f"{leading_dense} are leading -- dense layers must be a contiguous prefix"
        )
    else:
        n_ok += 1
    return n_ok


def check_rms_norm_eps(tc, md, errors):
    n_ok = 0
    if "inkling.attention.layer_norm_rms_epsilon" not in md:
        errors.append("GGUF metadata missing key 'inkling.attention.layer_norm_rms_epsilon'")
        return n_ok
    config_eps = tc.get("rms_norm_eps")
    if config_eps is None:
        errors.append("text_config.rms_norm_eps: missing from config.json")
        return n_ok
    gguf_eps = float(md["inkling.attention.layer_norm_rms_epsilon"])
    if abs(float(config_eps) - gguf_eps) >= 1e-12:
        errors.append(
            f"text_config.rms_norm_eps={config_eps!r} differs from "
            f"inkling.attention.layer_norm_rms_epsilon={gguf_eps!r} by "
            f">= 1e-12 (diff={abs(float(config_eps) - gguf_eps):.3e})"
        )
    else:
        n_ok += 1
    return n_ok


def check_not_in_gguf(tc, errors):
    """No GGUF ground truth to assert against -- just confirm the documented
    fields are still present (a field silently disappearing from config.json
    would otherwise go unnoticed since nothing else here touches it)."""
    n_ok = 0
    for field in NOT_IN_GGUF:
        if field not in tc:
            errors.append(f"text_config.{field}: missing from config.json (documented in NOT_IN_GGUF)")
        else:
            n_ok += 1
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="docs/inkling-config.json")
    ap.add_argument("--gguf", default="weights/inkling-gguf/UD-Q2_K_XL/"
                                      "inkling-UD-Q2_K_XL-00001-of-00008.gguf")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    tc = config.get("text_config")
    if tc is None:
        raise SystemExit(f"check_inkling_config: {args.config} missing text_config")

    src = LocalSource(args.gguf)
    hdr = parse_gguf_header(src)
    md = hdr.metadata

    errors = []
    n_ok = 0
    n_ok += check_scalars(tc, md, errors)
    n_ok += check_layer_pattern(tc, md, errors)
    n_ok += check_head_count_kv(tc, md, errors)
    n_ok += check_dense_block_count(tc, md, errors)
    n_ok += check_rms_norm_eps(tc, md, errors)
    n_ok += check_not_in_gguf(tc, errors)

    if errors:
        print(f"check_inkling_config: {len(errors)} mismatch(es) in {args.config}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"config check OK ({n_ok} fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
