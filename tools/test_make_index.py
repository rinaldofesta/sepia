#!/usr/bin/env python3
"""No-network unit tests for make_index.py.

The real UD-Q2_K_XL download only has one, zero-tensor part locally right
now, so a real run only exercises the inventory-fallback path. These tests
build a tiny synthetic two-part split GGUF *with real tensor data* (unlike
gguf_inspect's header-only fixtures) so the per-expert abs_offset
arithmetic, the local/inventory cross-check, and --verify's two-path
byte-compare all get exercised against actual bytes on disk.

Runnable directly: python3 tools/test_make_index.py
"""
import copy
import json
import os
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inspect as gi  # noqa: E402
import make_index as mi  # noqa: E402

N_EXPERTS = mi.N_EXPERTS  # 256, matched by the fixture tensors below


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


def _expert_tagged_bytes(n_experts, bytes_per_expert, tag_base):
    """n_experts blocks of bytes_per_expert bytes, each block filled with
    ((tag_base + e) % 256) -- lets a test byte-verify abs_offset arithmetic
    by checking which expert's tag landed at a given file offset."""
    return b"".join(bytes([(tag_base + e) % 256]) * bytes_per_expert for e in range(n_experts))


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
    header += _pack_metadata_kv("general.architecture", gi.GGUF_TYPE_STRING, _pack_string("test-arch"))
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


# Fixture: 1 MoE layer (0), gate in part A, up+down in part B, plus one
# resident tensor in each part. 256 experts, Q4_0 (block=32, type_size=18),
# dims [32, 1, 256] -> 1 block/expert -> bytes_per_expert=18 exactly.
BPE = 18
PART_A_NAME = "TEST_SPLIT/test-00001-of-00002.gguf"
PART_B_NAME = "TEST_SPLIT/test-00002-of-00002.gguf"


def _build_fixture_files():
    gate_data = _expert_tagged_bytes(N_EXPERTS, BPE, tag_base=0)
    up_data = _expert_tagged_bytes(N_EXPERTS, BPE, tag_base=1)
    down_data = _expert_tagged_bytes(N_EXPERTS, BPE, tag_base=2)
    embd_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    norm_data = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)

    part_a_specs = [
        ("token_embd.weight", [4], gi.GGML_TYPE_F32, embd_data),
        ("blk.0.ffn_gate_exps.weight", [32, 1, N_EXPERTS], gi.GGML_TYPE_Q4_0, gate_data),
    ]
    part_b_specs = [
        ("blk.0.ffn_up_exps.weight", [32, 1, N_EXPERTS], gi.GGML_TYPE_Q4_0, up_data),
        ("blk.0.ffn_down_exps.weight", [32, 1, N_EXPERTS], gi.GGML_TYPE_Q4_0, down_data),
        ("output_norm.weight", [4], gi.GGML_TYPE_F32, norm_data),
    ]
    file_a, data_start_a, offsets_a = build_gguf_part(part_a_specs)
    file_b, data_start_b, offsets_b = build_gguf_part(part_b_specs)
    return {
        PART_A_NAME: (file_a, data_start_a, offsets_a, part_a_specs),
        PART_B_NAME: (file_b, data_start_b, offsets_b, part_b_specs),
    }


def _inventory_from_fixture(fixture):
    """Parses the fixture bytes with gguf_inspect itself (ground truth) to
    build an inventory doc structurally identical to
    docs/gguf-inventory-ud-q2_k_xl.json."""
    parts, tensors = [], []
    for part_file, (file_bytes, _data_start, _offsets, _specs) in fixture.items():
        import io
        header = gi.parse_gguf_header(io.BytesIO(file_bytes))
        parts.append({
            "part_file": part_file,
            "file_size_bytes": len(file_bytes),
            "alignment": header.alignment,
            "data_offset": header.data_offset,
        })
        for t in header.tensors:
            tensors.append(gi._tensor_to_dict(t, part_file))
    return {"repo": "test/repo", "quant_hint": "TEST_SPLIT", "parts": parts, "tensors": tensors}


class MakeIndexFixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _build_fixture_files()
        self.inventory = _inventory_from_fixture(self.fixture)
        self.tmpdir = tempfile.mkdtemp()
        self.weights_dir = os.path.join(self.tmpdir, "weights")
        self.inventory_path = os.path.join(self.tmpdir, "inventory.json")
        with open(self.inventory_path, "w") as f:
            json.dump(self.inventory, f)

    def _write_local(self, part_file):
        path = os.path.join(self.weights_dir, part_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.fixture[part_file][0])
        return path

    def test_all_local_matches_inventory_and_indexes_correctly(self):
        self._write_local(PART_A_NAME)
        self._write_local(PART_B_NAME)
        doc, parts = mi.build_index(self.weights_dir, self.inventory_path)

        self.assertEqual({p["source"] for p in parts}, {"local"})
        self.assertEqual(doc["n_moe_layers_indexed"], 1)
        self.assertEqual(doc["n_expert_tensor_entries"], N_EXPERTS * 3)
        self.assertEqual(len(doc["resident_tensors"]), 2)

        gate = doc["moe_layers"]["0"]["gate"]
        self.assertEqual(gate["bytes_per_expert"], BPE)
        self.assertEqual(len(gate["experts"]), N_EXPERTS)

        # abs_offset arithmetic: data_start + tensor_offset + e * bytes_per_expert
        data_start_a = self.fixture[PART_A_NAME][1]
        offsets_a = self.fixture[PART_A_NAME][2]
        for e in (0, 1, 254, 255):
            expected = data_start_a + offsets_a["blk.0.ffn_gate_exps.weight"] + e * BPE
            self.assertEqual(gate["experts"][e]["abs_offset"], expected)
            self.assertEqual(gate["experts"][e]["nbytes"], BPE)

        # And the arithmetic actually points at the right bytes on disk:
        # expert e's slice in the gate tensor was tagged with byte value e.
        local_path = os.path.join(self.weights_dir, PART_A_NAME)
        with open(local_path, "rb") as f:
            for e in (0, 17, 255):
                f.seek(gate["experts"][e]["abs_offset"])
                data = f.read(gate["experts"][e]["nbytes"])
                self.assertEqual(data, bytes([e % 256]) * BPE)

    def test_missing_part_falls_back_to_inventory(self):
        self._write_local(PART_A_NAME)
        # part B is never written locally
        doc, parts = mi.build_index(self.weights_dir, self.inventory_path)

        by_file = {p["part_file"]: p for p in parts}
        self.assertEqual(by_file[PART_A_NAME]["source"], "local")
        self.assertEqual(by_file[PART_B_NAME]["source"], "inventory")
        self.assertIsNone(by_file[PART_B_NAME]["local_size_bytes"])

        layer0 = doc["moe_layers"]["0"]
        self.assertEqual(layer0["gate"]["source"], "local")
        self.assertEqual(layer0["up"]["source"], "inventory")
        self.assertEqual(layer0["down"]["source"], "inventory")
        # arithmetic must still be correct even from the inventory fallback
        data_start_b = self.fixture[PART_B_NAME][1]
        offsets_b = self.fixture[PART_B_NAME][2]
        expected = data_start_b + offsets_b["blk.0.ffn_up_exps.weight"] + 3 * BPE
        self.assertEqual(layer0["up"]["experts"][3]["abs_offset"], expected)

    def test_partial_download_size_mismatch_falls_back_to_inventory(self):
        path = self._write_local(PART_A_NAME)
        with open(path, "r+b") as f:
            f.truncate(100)  # simulate a still-downloading / truncated part
        doc, parts = mi.build_index(self.weights_dir, self.inventory_path)
        by_file = {p["part_file"]: p for p in parts}
        self.assertEqual(by_file[PART_A_NAME]["source"], "inventory")
        self.assertEqual(doc["moe_layers"]["0"]["gate"]["source"], "inventory")

    def test_cross_check_catches_offset_mismatch(self):
        self._write_local(PART_A_NAME)
        self._write_local(PART_B_NAME)
        bad_inventory = copy.deepcopy(self.inventory)
        for t in bad_inventory["tensors"]:
            if t["name"] == "blk.0.ffn_gate_exps.weight":
                t["offset"] += 32  # corrupt it
        bad_path = os.path.join(self.tmpdir, "bad_inventory.json")
        with open(bad_path, "w") as f:
            json.dump(bad_inventory, f)
        with self.assertRaises(mi.IndexBuildError):
            mi.build_index(self.weights_dir, bad_path)

    def test_cross_check_catches_missing_tensor(self):
        self._write_local(PART_A_NAME)
        self._write_local(PART_B_NAME)
        bad_inventory = copy.deepcopy(self.inventory)
        bad_inventory["tensors"] = [
            t for t in bad_inventory["tensors"] if t["name"] != "token_embd.weight"
        ]
        bad_path = os.path.join(self.tmpdir, "bad_inventory.json")
        with open(bad_path, "w") as f:
            json.dump(bad_inventory, f)
        with self.assertRaises(mi.IndexBuildError):
            mi.build_index(self.weights_dir, bad_path)

    def test_quant_histogram_and_resident_offsets(self):
        self._write_local(PART_A_NAME)
        self._write_local(PART_B_NAME)
        doc, _parts = mi.build_index(self.weights_dir, self.inventory_path)
        hist = doc["quant_type_histogram"]
        self.assertEqual(hist["Q4_0"]["tensor_count"], 3)
        self.assertEqual(hist["Q4_0"]["bytes"], 3 * N_EXPERTS * BPE)
        self.assertEqual(hist["F32"]["tensor_count"], 2)

        resident_by_name = {r["name"]: r for r in doc["resident_tensors"]}
        data_start_b = self.fixture[PART_B_NAME][1]
        offsets_b = self.fixture[PART_B_NAME][2]
        expected = data_start_b + offsets_b["output_norm.weight"]
        self.assertEqual(resident_by_name["output_norm.weight"]["abs_offset"], expected)
        self.assertEqual(resident_by_name["output_norm.weight"]["nbytes"], 16)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _build_fixture_files()
        self.inventory = _inventory_from_fixture(self.fixture)
        self.tmpdir = tempfile.mkdtemp()
        self.weights_dir = os.path.join(self.tmpdir, "weights")
        self.inventory_path = os.path.join(self.tmpdir, "inventory.json")
        with open(self.inventory_path, "w") as f:
            json.dump(self.inventory, f)
        for part_file in (PART_A_NAME, PART_B_NAME):
            path = os.path.join(self.weights_dir, part_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(self.fixture[part_file][0])

    def test_verify_all_local_all_ok(self):
        doc, parts = mi.build_index(self.weights_dir, self.inventory_path)
        rng = random.Random(1)
        ok = mi.run_verify(doc, parts, self.weights_dir, 30, rng)
        self.assertTrue(ok)

    def test_verify_skips_inventory_only_parts(self):
        # remove part B locally after building the fixture dir (up/down move
        # to inventory-fallback; only gate, in part A, stays verifiable)
        os.remove(os.path.join(self.weights_dir, PART_B_NAME))
        doc, parts = mi.build_index(self.weights_dir, self.inventory_path)
        rng = random.Random(2)
        # enough draws to almost certainly hit both local and inventory-only entries
        ok = mi.run_verify(doc, parts, self.weights_dir, 60, rng)
        self.assertTrue(ok)  # skips are not failures

    def test_verify_detects_corrupted_index_offset(self):
        doc, parts = mi.build_index(self.weights_dir, self.inventory_path)
        # Corrupt every cached abs_offset for one whole tensor kind (256 of
        # the 768 total candidates), so the two read paths (cached-in-doc
        # vs. fresh-parse-and-walk) disagree for that kind. A single
        # corrupted entry among 768 would only have ~48% odds of being
        # sampled in 100 draws; corrupting a whole kind makes a miss
        # astronomically unlikely (~(512/768)^100).
        for entry in doc["moe_layers"]["0"]["up"]["experts"]:
            entry["abs_offset"] += BPE
        rng = random.Random(3)
        ok = mi.run_verify(doc, parts, self.weights_dir, 100, rng)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
