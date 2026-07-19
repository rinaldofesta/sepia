#!/usr/bin/env python3
"""No-network unit tests for extract_resident.py.

The real UD-Q2_K_XL download has no resident-tensor-bearing part locally
yet (only the zero-tensor metadata part), so a real run extracts nothing.
These tests build a tiny synthetic two-part GGUF with real resident-tensor
data so the actual stream-copy, SHA256 write/re-read verification, 64-byte
alignment, idempotent rerun, and free-space guard all get exercised.

Runnable directly: python3 tools/test_extract_resident.py
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inspect as gi  # noqa: E402
import extract_resident as er  # noqa: E402


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
    offsets = {}
    running = 0
    for name, _dims, _ggml_type, data in tensor_specs:
        running = _align_up(running, alignment)
        offsets[name] = running
        running += len(data)

    header = bytearray()
    header += b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensor_specs))
    header += struct.pack("<Q", 2)
    header += _pack_metadata_kv("general.architecture", gi.GGUF_TYPE_STRING, _pack_string("test-arch"))
    header += _pack_metadata_kv("general.alignment", gi.GGUF_TYPE_UINT32, struct.pack("<I", alignment))
    for name, dims, ggml_type, _data in tensor_specs:
        header += _pack_tensor_info(name, dims, ggml_type, offsets[name])

    data_start = _align_up(len(header), alignment)
    out = bytearray(header)
    out += b"\x00" * (data_start - len(header))
    for name, _dims, _ggml_type, data in tensor_specs:
        want_pos = data_start + offsets[name]
        out += b"\x00" * (want_pos - len(out))
        out += data
    return bytes(out)


PART_A_NAME = "TEST_SPLIT/test-00001-of-00002.gguf"
PART_B_NAME = "TEST_SPLIT/test-00002-of-00002.gguf"


def _build_fixture_files():
    part_a_specs = [
        ("token_embd.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
        ("attn_norm.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)),
    ]
    part_b_specs = [
        ("output_norm.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 9.0, 10.0, 11.0, 12.0)),
        ("output.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 13.0, 14.0, 15.0, 16.0)),
    ]
    return {
        PART_A_NAME: (build_gguf_part(part_a_specs), part_a_specs),
        PART_B_NAME: (build_gguf_part(part_b_specs), part_b_specs),
    }


def _inventory_from_fixture(fixture):
    import io
    parts, tensors = [], []
    for part_file, (file_bytes, _specs) in fixture.items():
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


class ExtractResidentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _build_fixture_files()
        self.inventory = _inventory_from_fixture(self.fixture)
        self.tmpdir = tempfile.mkdtemp()
        self.weights_dir = os.path.join(self.tmpdir, "weights")
        self.inventory_path = os.path.join(self.tmpdir, "inventory.json")
        self.bin_path = os.path.join(self.tmpdir, "resident.bin")
        self.manifest_path = os.path.join(self.tmpdir, "resident-manifest.json")
        with open(self.inventory_path, "w") as f:
            json.dump(self.inventory, f)

    def _write_local(self, part_file):
        path = os.path.join(self.weights_dir, part_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.fixture[part_file][0])
        return path

    def _run(self, min_free_gb=0):
        return er.run_extract(self.weights_dir, self.inventory_path, self.bin_path,
                               self.manifest_path, min_free_gb)

    def test_extracts_all_when_all_parts_local(self):
        self._write_local(PART_A_NAME)
        self._write_local(PART_B_NAME)
        manifest, newly = self._run()

        self.assertEqual(len(newly), 4)
        self.assertEqual(manifest["n_tensors_extracted"], 4)
        self.assertEqual(manifest["pending_parts"], [])
        self.assertEqual(manifest["resident_bin_size"], os.path.getsize(self.bin_path))

        by_name = {t["name"]: t for t in manifest["tensors"]}
        for entry in by_name.values():
            self.assertEqual(entry["offset"] % er.RESIDENT_ALIGNMENT, 0)
            self.assertEqual(len(entry["sha256"]), 64)

        # bytes actually in resident.bin match the source, and the sha256
        # is the real SHA256 of those bytes (not just any 64-hex-char string)
        embd = by_name["token_embd.weight"]
        with open(self.bin_path, "rb") as f:
            f.seek(embd["offset"])
            data = f.read(embd["nbytes"])
        self.assertEqual(data, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
        self.assertEqual(hashlib.sha256(data).hexdigest(), embd["sha256"])

    def test_partial_download_extracts_available_and_defers_rest(self):
        self._write_local(PART_A_NAME)
        manifest, newly = self._run()

        self.assertEqual(len(newly), 2)
        self.assertEqual({t["name"] for t in newly}, {"token_embd.weight", "attn_norm.weight"})
        self.assertEqual(manifest["pending_parts"], [PART_B_NAME])
        self.assertEqual(manifest["n_tensors_extracted"], 2)
        self.assertEqual(manifest["n_resident_tensors_total"], 4)

    def test_idempotent_rerun_only_completes_missing_and_preserves_offsets(self):
        self._write_local(PART_A_NAME)
        manifest1, newly1 = self._run()
        self.assertEqual(len(newly1), 2)
        offsets_after_first_run = {t["name"]: t["offset"] for t in manifest1["tensors"]}

        # "download completes": part B becomes available, rerun
        self._write_local(PART_B_NAME)
        manifest2, newly2 = self._run()

        self.assertEqual(len(newly2), 2)  # only the newly available ones
        self.assertEqual({t["name"] for t in newly2}, {"output_norm.weight", "output.weight"})
        self.assertEqual(manifest2["n_tensors_extracted"], 4)
        self.assertEqual(manifest2["pending_parts"], [])

        # tensors extracted in run 1 kept the exact same resident.bin offset
        for name, offset in offsets_after_first_run.items():
            entry = next(t for t in manifest2["tensors"] if t["name"] == name)
            self.assertEqual(entry["offset"], offset)

        # a third, fully-idempotent rerun extracts nothing new
        manifest3, newly3 = self._run()
        self.assertEqual(len(newly3), 0)
        self.assertEqual(manifest3["n_tensors_extracted"], 4)

    def test_free_space_guard_aborts_before_writing(self):
        self._write_local(PART_A_NAME)
        with self.assertRaises(er.ExtractError):
            self._run(min_free_gb=10**9)  # impossible to satisfy
        self.assertFalse(os.path.exists(self.bin_path))
        self.assertFalse(os.path.exists(self.manifest_path))

    def test_write_reread_mismatch_truncates_and_raises(self):
        self._write_local(PART_A_NAME)
        with mock.patch.object(er, "_hash_range", return_value="0" * 64):
            with self.assertRaises(er.ExtractError):
                self._run()
        # the failed tensor's partial write must be truncated back off
        self.assertTrue(os.path.exists(self.bin_path))
        self.assertEqual(os.path.getsize(self.bin_path), 0)
        self.assertFalse(os.path.exists(self.manifest_path))


if __name__ == "__main__":
    unittest.main()
