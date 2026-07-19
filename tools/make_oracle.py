#!/usr/bin/env python3
"""Build SEPIA's tiny-random Inkling oracle.

This produces a toy-scale model with Inkling's exact architecture (the
native `transformers` `InklingForConditionalGeneration` class -- no
lookalike, no trust_remote_code), seeded and deterministic. Its greedy
generation plus one teacher-forcing pass become the token-exact reference
that the C engine (task 0.4) must reproduce, working only from
`tools/oracle/ref_inkling.json`, `tools/oracle/tiny/`, and
`docs/architecture-notes.md`.

Usage:
    .venv/bin/python tools/make_oracle.py              # write the committed fixtures
    .venv/bin/python tools/make_oracle.py --out-dir D   # write fixtures under D instead
    .venv/bin/python tools/make_oracle.py --check       # determinism check (see
                                                          # tools/test_oracle_determinism.sh)

See docs/architecture-notes.md for the architecture this toy config mirrors,
including which config.json fields above are structural and which are
present in the real checkpoint but never read by transformers 5.14.1's
modeling code (documented there, not silently dropped here).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ORACLE_DIR = REPO_ROOT / "tools" / "oracle"

SEED = 1234

# Fixed 12-token prompt, ids < 512 (toy vocab_size). Arbitrary but constant
# across runs -- determinism only requires it be fixed, not "realistic".
PROMPT_IDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
MAX_NEW_TOKENS = 20


# ---------------------------------------------------------------------------
# Toy config. Every key below is a real `thinkingmachines/Inkling`
# config.json field name, fetched and verified against the actual checkpoint
# (transformers 5.14.1, 2026-07-19) -- not guessed or derived from the
# dataclass defaults. Values are shrunk per the task-0.3 brief; field names
# are untouched. Comments marking "NOTE: not read by transformers 5.14.1"
# flag fields that exist in the real config.json but have no corresponding
# branch in the modeling code -- see docs/architecture-notes.md for the full
# accounting of these (this is not a SEPIA omission, it is how the shipped
# modeling code behaves).
# ---------------------------------------------------------------------------


def build_text_config_dict() -> dict:
    """Toy `InklingTextConfig` kwargs (real dims in parens)."""
    return {
        # sizes: hidden 6144->128, layers 66->6, vocab 201024->512
        "vocab_size": 512,
        # real: unpadded_vocab_size 200058 of vocab_size 201024 (966 padded
        # ids). Toy keeps a proportional gap to exercise the same
        # logits-slicing path in InklingForConditionalGeneration.forward.
        "unpadded_vocab_size": 480,
        "hidden_size": 128,
        "num_hidden_layers": 6,
        # attention: global heads use num_*, local/SWA heads use swa_*
        "num_attention_heads": 4,
        "num_key_value_heads": 2,  # global-layer KV heads (real: 8)
        "head_dim": 16,
        "q_bias": False,  # NOTE: not read -- q_proj is unconditionally bias=False
        "o_bias": False,  # NOTE: not read -- o_proj is unconditionally bias=False
        "swa_head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,  # local-layer KV heads (real: 16)
        "sliding_window_size": 8,  # real: 512
        # local_layer_ids lists the LOCAL (sliding-window) layers explicitly,
        # same field the real 66-layer config uses (55 local layers listed).
        # Layer 5 (the only id not listed) is the sole global layer here,
        # same "every 6th layer is global" pattern as the real model.
        "local_layer_ids": [0, 1, 2, 3, 4],
        # banded content-dependent relative position bias
        "d_rel": 16,  # kept at real scale per the brief
        "rel_extent": 16,  # real: 1024. Local layers ignore this and use
        # sliding_window_size as their rel_extent instead (see
        # architecture-notes.md); toy sliding_window_size=8 so local layers
        # get rel_extent=8, global layers get rel_extent=16.
        "log_scaling_n_floor": 128000,  # kept at the REAL value on purpose:
        # this guarantees log-scaling never triggers at toy scale (max
        # position here is 31 << 128000) -- an intentional, documented gap,
        # see docs/architecture-notes.md "toy-scale gaps".
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "use_embed_norm": True,  # NOTE: not read -- embed_norm is unconditional
        # short convolutions
        "use_sconv": True,  # NOTE: not read -- sconv is unconditional in
        # transformers 5.14.1 (no `if config.use_sconv` branch exists)
        "sconv_kernel_size": 4,
        # dense vs MoE FFN split
        "dense_mlp_idx": 2,  # layers [0, dense_mlp_idx) are dense MLP, the
        # rest are MoE -- resolved from InklingTextConfig.__post_init__:
        # `mlp_layer_types = ["dense" if i < dense_mlp_idx else "sparse" ...]`
        "dense_intermediate_size": 256,  # real: 24576. Always wins over the
        # `intermediate_size` key below when present (see "naming trap" in
        # docs/architecture-notes.md).
        "intermediate_size": 256,  # decoy in the real config.json (real value
        # 3072, silently discarded because dense_intermediate_size is also
        # present); set equal here so no accidental second width leaks in.
        "hidden_act": "silu",
        "moe_intermediate_size": 64,  # real: 3072. This is the field
        # InklingExperts / InklingSharedExperts actually read for per-expert
        # width -- NOT `intermediate_size`.
        # MoE routing
        "n_routed_experts": 8,  # real: 256
        "num_experts_per_tok": 2,  # real: 6
        "n_shared_experts": 2,  # kept at real value per the brief
        "shared_expert_sink": True,  # NOTE: not read as a conditional -- the
        # shared-expert-sink shape (n_shared_experts extra router logit
        # columns) is always structurally present
        "route_scale": 8.0,  # kept at real value per the brief
        "gate_activation": "sigmoid",  # NOTE: not read -- InklingTopkRouter
        # .forward() hardcodes `.sigmoid()`
        "use_gate_bias": True,  # NOTE: not read -- e_score_correction_bias is
        # unconditionally created
        "norm_after_topk": True,  # NOTE: not read -- the router always
        # normalizes after top-k selection (see architecture-notes.md)
        "use_global_scale": True,  # NOTE: not read -- the per-layer
        # `global_scale` param (ffn_gscale in the GGUF) is unconditionally
        # created on both dense (InklingMLP) and MoE (InklingTopkRouter) layers
        # output
        "logits_mup_width_multiplier": 24.0,  # divides final hidden_states
        # before lm_head -- a LOGIT scale, not an embedding scale, despite
        # the `embedding_multiplier` attribute_map alias name (see
        # architecture-notes.md).
        "final_logit_softcapping": None,  # NOTE: not read -- no softcap logic
        # exists in this modeling code
        "rms_norm_eps_moe_gate": 1e-6,  # NOTE: not read -- no separate
        # MoE-gate norm exists in this modeling code
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "pad_token_id": None,
        "bos_token_id": 1,
        "eos_token_id": 2,
        # MTP intentionally omitted: no `num_mtp_layers` / `mtp_config` kwarg
        # is passed. transformers 5.14.1's InklingForConditionalGeneration
        # has no MTP module implementation at all -- only config plumbing
        # (num_mtp_layers, mtp_local_layer_ids, mtp_layer_types property) and
        # a load-time ignore regex for `model.mtp.*` checkpoint keys. See
        # docs/architecture-notes.md "MTP" section for the checkpoint-side
        # structure (documentation only, not exercised here).
    }


def build_vision_config_dict() -> dict:
    """Toy `InklingVisionConfig` kwargs. Required by
    `InklingForConditionalGeneration.__init__` even for text-only use (the
    tower is constructed unconditionally); never exercised by our
    text-only forward pass (no `pixel_values` is ever passed)."""
    return {
        "vision_encoder_type": "hmlp",  # NOTE: not read -- there is only one encoder implementation
        "patch_size": 2,  # real: 40
        "temporal_patch_size": 2,  # real: 2, kept as-is: temporal_patch_size=1
        # hits a real (pre-existing, upstream) dtype bug in
        # `plan_out_scales` -- see docs/architecture-notes.md "transformers
        # quirks hit while building the oracle".
        "n_channels": 3,
        "hidden_size": 8,  # real: 1024
        "n_layers": 1,  # real: 4
        "num_attention_heads": 1,
        "use_vision_norm": True,  # NOTE: not read -- norm is unconditional
    }


def build_audio_config_dict() -> dict:
    """Toy `InklingAudioConfig` kwargs. Same "required but unused" story as
    vision: constructed unconditionally, never exercised (no
    `audio_input_ids` is ever passed)."""
    return {
        "n_mel_bins": 4,  # real: 80
        "mel_vocab_size": 4,  # real: 16
        "bias": False,
        "dmel_min_value": -7.0,
        "dmel_max_value": 2.0,
        "use_audio_norm": True,  # NOTE: not read -- norm is unconditional
        "audio_mode": "dmel",
    }


def check_transformers_support() -> str:
    """Verify the installed transformers has native Inkling classes.
    Returns the path used ("native") or raises if unavailable (the brief's
    fallback is trust_remote_code against the HF repo; unneeded here since
    the native classes are present -- see the task report for the version
    check)."""
    import transformers

    print(f"transformers version: {transformers.__version__}", file=sys.stderr)
    try:
        from transformers import InklingForConditionalGeneration  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "transformers lacks native Inkling classes; brief calls for "
            "upgrading the venv or falling back to trust_remote_code=True "
            "against thinkingmachines/Inkling. Neither is implemented in "
            "this script -- see the task report."
        ) from e
    return "native"


def build_model():
    """Build the toy InklingForConditionalGeneration, seeded and in float32
    on CPU. Returns (model, config)."""
    import torch
    from transformers.models.inkling.configuration_inkling import InklingConfig
    from transformers.models.inkling.modeling_inkling import InklingForConditionalGeneration

    torch.manual_seed(SEED)

    config = InklingConfig(
        text_config=build_text_config_dict(),
        vision_config=build_vision_config_dict(),
        audio_config=build_audio_config_dict(),
    )
    model = InklingForConditionalGeneration(config)
    model.eval()
    return model, config


def run_oracle(model) -> dict:
    """Greedy-generate from the fixed prompt, then run one teacher-forcing
    forward pass over the full sequence. Returns the ref dict (without
    `meta`, which the caller fills in)."""
    import torch

    prompt_ids = torch.tensor([PROMPT_IDS], dtype=torch.long)

    with torch.no_grad():
        full_ids = model.generate(
            input_ids=prompt_ids,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            # Toy weights are random: some generated token could coincide
            # with the configured eos_token_id (2) and stop generation
            # early, breaking the fixed 32-token contract. Disable early
            # stopping so we always get exactly len(PROMPT_IDS) +
            # MAX_NEW_TOKENS tokens.
            eos_token_id=None,
            pad_token_id=0,
        )

    if full_ids.shape[1] != len(PROMPT_IDS) + MAX_NEW_TOKENS:
        raise RuntimeError(
            f"generate() produced {full_ids.shape[1]} tokens, expected "
            f"{len(PROMPT_IDS) + MAX_NEW_TOKENS} (eos_token_id=None should "
            "have prevented early stopping)"
        )

    with torch.no_grad():
        tf_out = model(input_ids=full_ids, use_cache=False)
    tf_pred = tf_out.logits.argmax(dim=-1).squeeze(0).tolist()

    return {
        "prompt_ids": PROMPT_IDS,
        "full_ids": full_ids.squeeze(0).tolist(),
        "tf_pred": tf_pred,
    }


def save_oracle(model, ref: dict, out_dir: Path) -> None:
    import torch
    import transformers

    tiny_dir = out_dir / "tiny"
    tiny_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tiny_dir, safe_serialization=True)

    ref = dict(ref)
    ref["meta"] = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": SEED,
        "dtype": "float32",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "ref_inkling.json"
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2, sort_keys=True)
        f.write("\n")


def generate_all(out_dir: Path) -> dict:
    check_transformers_support()
    model, _config = build_model()
    ref = run_oracle(model)
    save_oracle(model, ref, out_dir)
    return ref


# ---------------------------------------------------------------------------
# Determinism check: run this script twice, in two separate subprocesses (a
# fresh interpreter each time, not just a repeated in-process call), and
# assert the fixtures are byte-identical.
# ---------------------------------------------------------------------------


def _run_subprocess_generate(out_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"make_oracle.py subprocess failed (exit {result.returncode})")


def check_determinism() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"

        print("running make_oracle.py (subprocess 1/2)...", file=sys.stderr)
        _run_subprocess_generate(dir_a)
        print("running make_oracle.py (subprocess 2/2)...", file=sys.stderr)
        _run_subprocess_generate(dir_b)

        files_to_compare = [
            "ref_inkling.json",
            "tiny/config.json",
            "tiny/model.safetensors",
        ]
        all_match = True
        for rel in files_to_compare:
            bytes_a = (dir_a / rel).read_bytes()
            bytes_b = (dir_b / rel).read_bytes()
            match = bytes_a == bytes_b
            status = "OK" if match else "MISMATCH"
            print(f"  {rel}: {status} ({len(bytes_a)} bytes)", file=sys.stderr)
            all_match = all_match and match

        return all_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ORACLE_DIR,
        help="Directory to write tiny/ and ref_inkling.json under (default: tools/oracle/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the generator twice in fresh subprocesses and assert byte-identical output",
    )
    args = parser.parse_args()

    if args.check:
        ok = check_determinism()
        if ok:
            print("determinism check: PASS (ref_inkling.json, config.json, model.safetensors byte-identical across two runs)")
            return 0
        else:
            print("determinism check: FAIL", file=sys.stderr)
            return 1

    ref = generate_all(args.out_dir)
    n_generated = len(ref["full_ids"]) - len(ref["prompt_ids"])
    print(
        f"wrote {args.out_dir / 'ref_inkling.json'} and {args.out_dir / 'tiny'} "
        f"({len(ref['prompt_ids'])} prompt + {n_generated} generated tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
