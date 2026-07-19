#!/usr/bin/env python3
"""Dump per-layer activations from the toy Inkling oracle for bisecting
divergence against src/sepia.c layer-by-layer (task 0.4 debugging aid).

Captures the same named checkpoints sepia.c's `--dump-acts` writes: the
embedding-norm output, each decoder layer's output, the final-norm output,
and the logits -- all for one teacher-forcing forward pass over
ref_inkling.json's full_ids (the same 32-token sequence the self-test's
prefill check uses).

Usage:
    .venv/bin/python tools/dump_activations.py                     # write dump.npz
    .venv/bin/python tools/dump_activations.py --compare sepia.bin # diff against
        `./sepia --dump-acts sepia.bin`'s raw dump, printing max abs diff per key
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_DIR = REPO_ROOT / "tools" / "oracle" / "tiny"
REF_PATH = REPO_ROOT / "tools" / "oracle" / "ref_inkling.json"


def capture_activations():
    import torch
    from transformers.models.inkling.modeling_inkling import InklingForConditionalGeneration

    ref = json.loads(REF_PATH.read_text())
    full_ids = torch.tensor([ref["full_ids"]], dtype=torch.long)

    model = InklingForConditionalGeneration.from_pretrained(TINY_DIR, dtype=torch.float32)
    model.eval()

    acts: dict[str, torch.Tensor] = {}
    hooks = []

    def save(name):
        def hook(_module, _inp, out):
            acts[name] = out.detach().clone()

        return hook

    lm = model.model.language_model
    hooks.append(lm.embed_norm.register_forward_hook(save("embed_out")))
    for i, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(save(f"layer{i}.out")))
    hooks.append(lm.norm.register_forward_hook(save("final_norm_out")))

    with torch.no_grad():
        out = model(input_ids=full_ids, use_cache=False)
    acts["logits"] = out.logits.detach().clone()

    for h in hooks:
        h.remove()
    return acts


def save_npz(acts: dict, out_path: Path) -> None:
    import numpy as np

    np.savez(out_path, **{k: v.squeeze(0).numpy() for k, v in acts.items()})
    print(f"wrote {out_path} ({len(acts)} arrays)")


def load_sepia_dump(path: Path) -> dict:
    import numpy as np

    out = {}
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            (namelen,) = struct.unpack("<I", head)
            name = f.read(namelen).decode("utf-8")
            (ndim,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            n = 1
            for d in dims:
                n *= d
            data = np.frombuffer(f.read(4 * n), dtype="<f4").reshape(dims)
            out[name] = data
    return out


def compare(acts: dict, sepia_path: Path) -> float:
    import numpy as np

    sepia = load_sepia_dump(sepia_path)
    print(f"{'key':<20} {'shape':<16} {'max_abs_diff':>14}")
    overall_max = 0.0
    for key, ref_t in acts.items():
        ref_a = ref_t.squeeze(0).numpy()
        if key not in sepia:
            print(f"{key:<20} MISSING from {sepia_path}")
            continue
        got = sepia[key]
        if got.shape != ref_a.shape:
            print(f"{key:<20} SHAPE MISMATCH ref={ref_a.shape} got={got.shape}")
            continue
        diff = float(np.max(np.abs(ref_a - got)))
        overall_max = max(overall_max, diff)
        print(f"{key:<20} {str(ref_a.shape):<16} {diff:>14.3e}")
    return overall_max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dump.npz")
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="raw binary dump from `./sepia --dump-acts FILE` to diff against",
    )
    args = ap.parse_args()

    acts = capture_activations()
    if args.compare:
        overall_max = compare(acts, args.compare)
        print(f"overall max abs diff: {overall_max:.3e}")
    else:
        save_npz(acts, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
