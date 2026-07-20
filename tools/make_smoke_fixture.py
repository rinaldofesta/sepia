#!/usr/bin/env python3
"""Builds a synthetic, weights-free fixture that exercises SEPIA's real-model
plumbing end-to-end without needing the 317GB Inkling download.

Writes a tiny single-part GGUF v3 file (packed by hand, mirroring
tools/test_make_index.py's fixture packer) holding one MoE "layer" (layer 0)
with 2 routed experts' gate/up/down tensors (Q8_0, deterministic byte-tagged
blocks) plus 3 resident tensors (two F32, one a real Q8_0 weight matrix).
Runs the REAL tools/make_index.py `build_index` and tools/extract_resident.py
`run_extract` against it -- not a reimplementation of either -- to produce
the same index.json / resident.bin / resident-manifest.json shapes `sepia
--real` consumes, then emits `smoke_expected.json`: each expert slab's exact
byte range and first-8-bytes, the 3 resident tensors' offsets, and a qlinear
expected vector (computed in float64, matching src/sepia.c's dotf/qlinear
accumulation policy) for the Q8_0 resident tensor.

`sepia --smoke <dir>` loads this same directory through the identical
manifest_load/index_load/resident_qtensor/qlinear C code paths --real uses,
byte-compares the expert slabs, and bitwise-compares the qlinear result.

Python 3 stdlib only. Usage:
    python3 tools/make_smoke_fixture.py --out /tmp/sepia-smoke
"""
import argparse
import io
import json
import os
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inspect as gi  # noqa: E402
import make_index as mi  # noqa: E402
import extract_resident as er  # noqa: E402

N_EXPERTS = 2  # synthetic layer's expert count (real deployment is 256)
GGUF_PART_NAME = "SMOKE/smoke-00001-of-00001.gguf"

# --- GGUF v3 packing (mirrors tools/test_make_index.py's build_gguf_part) ---


def _pack_string(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_metadata_kv(key, value_type, value_bytes):
    return _pack_string(key) + struct.pack("<I", value_type) + value_bytes


def _pack_tensor_info(name, dims, ggml_type, offset):
    out = _pack_string(name)
    out += struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", ggml_type)
    out += struct.pack("<Q", offset)
    return out


def _align_up(n, a):
    return n if n % a == 0 else n + (a - n % a)


def build_gguf_part(tensor_specs, alignment=32):
    """tensor_specs: [(name, dims, ggml_type, data_bytes), ...].
    Returns (file_bytes, data_start, {name: relative_offset})."""
    offsets = {}
    running = 0
    for name, _dims, _ggml_type, data in tensor_specs:
        running = _align_up(running, alignment)
        offsets[name] = running
        running += len(data)

    header = bytearray()
    header += b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensor_specs))
    header += struct.pack("<Q", 2)  # metadata_kv_count
    header += _pack_metadata_kv("general.architecture", gi.GGUF_TYPE_STRING, _pack_string("sepia-smoke"))
    header += _pack_metadata_kv("general.alignment", gi.GGUF_TYPE_UINT32, struct.pack("<I", alignment))
    for name, dims, ggml_type, _data in tensor_specs:
        header += _pack_tensor_info(name, dims, ggml_type, offsets[name])

    data_start = _align_up(len(header), alignment)
    out = bytearray(header)
    out += b"\x00" * (data_start - len(header))
    for name, _dims, _ggml_type, data in tensor_specs:
        want_pos = data_start + offsets[name]
        assert len(out) <= want_pos
        out += b"\x00" * (want_pos - len(out))
        assert len(out) == want_pos
        out += data
    return bytes(out), data_start, offsets


def _expert_tagged_bytes(n_experts, bytes_per_expert, tag_base):
    """n_experts blocks of bytes_per_expert bytes, each filled with
    ((tag_base + e) % 256) -- lets the C side's byte-compare of a pread'd
    slab's first bytes confirm the abs_offset arithmetic landed on the right
    expert's data, not just that /some/ bytes were read."""
    return b"".join(bytes([(tag_base + e) % 256]) * bytes_per_expert for e in range(n_experts))


# --- Q8_0 encode/decode (mirrors src/quants.c's dequant_q8_0 exactly) --------

QK8_0 = 32


def _pack_q8_0_block(scale, int8_values):
    assert len(int8_values) == QK8_0
    return struct.pack("<e", scale) + struct.pack("<32b", *int8_values)


def _dequant_q8_0_row(block_bytes, scale, int8_values):
    # Values are chosen (power-of-two scale, small integers) so this matches
    # C's `d * (float)qs[j]` (a float32 multiply) exactly even computed here
    # in Python's native double -- see the comment on _build_qlinear_case.
    del block_bytes
    return [scale * v for v in int8_values]


def _f32_round(x):
    """Round a Python float to its nearest float32 value, matching a C
    `(float)` cast -- a no-op for every value used in this fixture (all
    exactly representable in float32), kept for defensive clarity."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _build_q8_0_matrix(out_dim, in_dim, scale):
    """Builds an [out_dim, in_dim] Q8_0 weight matrix (in_dim must be a
    multiple of QK8_0) with a distinct, deterministic int8 pattern per row,
    returning (raw_bytes, rows_as_floats) for the qlinear reference calc."""
    assert in_dim % QK8_0 == 0
    raw = bytearray()
    rows = []
    for r in range(out_dim):
        row_vals = []
        blocks_bytes = bytearray()
        for b in range(in_dim // QK8_0):
            ints = [((j + r * 7 + b * 3) % 32) - 16 for j in range(QK8_0)]
            blocks_bytes += _pack_q8_0_block(scale, ints)
            row_vals.extend(_dequant_q8_0_row(None, scale, ints))
        raw += blocks_bytes
        rows.append(row_vals)
    return bytes(raw), rows


def _build_qlinear_case():
    """The resident Q8_0 tensor + expected qlinear vector. scale=0.5 (exact
    in fp16/fp32/double) and small integer x values keep every intermediate
    value exactly representable in float32, so the float64 reference
    computation below is bit-identical to src/sepia.c's dotf (double
    accumulate, cast to float32 on write) -- no rounding-order risk."""
    in_dim, out_dim = 32, 3
    scale = 0.5
    raw, rows = _build_q8_0_matrix(out_dim, in_dim, scale)
    x = [float(i + 1) for i in range(in_dim)]  # 1..32, exact in f32/f64
    y = []
    for row in rows:
        acc = 0.0
        for a, b in zip(row, x):
            acc += a * b
        y.append(_f32_round(acc))
    return raw, in_dim, out_dim, x, y


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="/tmp/sepia-smoke", help="output directory (recreated fresh)")
    args = p.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    shutil.rmtree(out_dir, ignore_errors=True)
    weights_dir = os.path.join(out_dir, "inkling-gguf")
    os.makedirs(weights_dir, exist_ok=True)

    # --- synthetic tensors -----------------------------------------------
    gate_data = _expert_tagged_bytes(N_EXPERTS, QK8_0 + 2, tag_base=0)   # 1 Q8_0 block/expert = 34 bytes
    up_data = _expert_tagged_bytes(N_EXPERTS, QK8_0 + 2, tag_base=50)
    down_data = _expert_tagged_bytes(N_EXPERTS, QK8_0 + 2, tag_base=100)

    f32_a = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    f32_b = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)
    q8_raw, q_in, q_out, qx, qy = _build_qlinear_case()
    Q8_TENSOR_NAME = "blk.0.attn_r.weight"

    part_specs = [
        ("token_embd_norm.weight", [4], gi.GGML_TYPE_F32, f32_a),
        ("output_norm.weight", [4], gi.GGML_TYPE_F32, f32_b),
        (Q8_TENSOR_NAME, [q_in, q_out], gi.GGML_TYPE_Q8_0, q8_raw),
        ("blk.0.ffn_gate_exps.weight", [QK8_0, 1, N_EXPERTS], gi.GGML_TYPE_Q8_0, gate_data),
        ("blk.0.ffn_up_exps.weight", [QK8_0, 1, N_EXPERTS], gi.GGML_TYPE_Q8_0, up_data),
        ("blk.0.ffn_down_exps.weight", [QK8_0, 1, N_EXPERTS], gi.GGML_TYPE_Q8_0, down_data),
    ]
    file_bytes, _data_start, _offsets = build_gguf_part(part_specs)

    part_path = os.path.join(weights_dir, GGUF_PART_NAME)
    os.makedirs(os.path.dirname(part_path), exist_ok=True)
    with open(part_path, "wb") as f:
        f.write(file_bytes)

    # --- inventory doc (ground-truth-parsed from the bytes just written,
    # mirroring tools/test_make_index.py's _inventory_from_fixture) --------
    header = gi.parse_gguf_header(io.BytesIO(file_bytes))
    inventory = {
        "repo": "sepia-smoke/synthetic",
        "quant_hint": "SMOKE",
        "parts": [{
            "part_file": GGUF_PART_NAME,
            "file_size_bytes": len(file_bytes),
            "alignment": header.alignment,
            "data_offset": header.data_offset,
        }],
        "tensors": [gi._tensor_to_dict(t, GGUF_PART_NAME) for t in header.tensors],
    }
    inventory_path = os.path.join(out_dir, "inventory.json")
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=2)

    # --- real build_index / run_extract, with N_EXPERTS patched down from
    # the production default of 256 (both make_index.py and extract_resident.py
    # share this module object, so the patch reaches extract_resident's own
    # internal build_index call too). ---------------------------------------
    mi.N_EXPERTS = N_EXPERTS

    doc, _parts = mi.build_index(weights_dir, inventory_path)
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    bin_path = os.path.join(out_dir, "resident.bin")
    manifest_path = os.path.join(out_dir, "resident-manifest.json")
    er.run_extract(weights_dir, inventory_path, bin_path, manifest_path, min_free_gb=0.001)

    # --- smoke_expected.json ------------------------------------------------
    layer0 = doc["moe_layers"]["0"]
    experts_expected = []
    for e in range(N_EXPERTS):
        entry = {"expert": e}
        for slot in ("gate", "up", "down"):
            ent = layer0[slot]["experts"][e]
            with open(part_path, "rb") as f:
                f.seek(ent["abs_offset"])
                first8 = f.read(8)
            entry[slot] = {
                "abs_offset": ent["abs_offset"],
                "nbytes": ent["nbytes"],
                "first8_hex": first8.hex(),
            }
        experts_expected.append(entry)

    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest_by_name = {t["name"]: t for t in manifest["tensors"]}
    resident_expected = [
        {"name": name, "offset": manifest_by_name[name]["offset"], "nbytes": manifest_by_name[name]["nbytes"]}
        for name in ("token_embd_norm.weight", "output_norm.weight", Q8_TENSOR_NAME)
    ]

    expected = {
        "layer": 0,
        "n_experts": N_EXPERTS,
        "experts": experts_expected,
        "resident": resident_expected,
        "qlinear": {"tensor_name": Q8_TENSOR_NAME, "x": qx, "y": qy},
    }
    expected_path = os.path.join(out_dir, "smoke_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
        f.write("\n")

    print(f"wrote smoke fixture to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
