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
import io
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


THREE_TENSOR_PART_NAME = "TEST_SPLIT3/test-00001-of-00001.gguf"


def _build_three_tensor_fixture():
    specs = [
        ("tensor_a.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)),
        ("tensor_b.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)),
        ("tensor_c.weight", [4], gi.GGML_TYPE_F32, struct.pack("<4f", 9.0, 10.0, 11.0, 12.0)),
    ]
    return build_gguf_part(specs)


class ManifestDurabilityTests(unittest.TestCase):
    """Reproduces the reviewer's exact scenario for the Important finding:
    a mid-run failure on the last of several tensors must not orphan the
    bytes of tensors that already succeeded and were verified before it.
    Before the fix, the manifest was only written once at the very end of
    run_extract, so failing on tensor 3 of 3 left 2 tensors' worth of
    already-verified bytes in resident.bin with no manifest entry -- a
    rerun would re-extract all 3 from scratch and permanently orphan the
    first attempt's bytes."""

    def setUp(self):
        self.file_bytes = _build_three_tensor_fixture()
        header = gi.parse_gguf_header(io.BytesIO(self.file_bytes))
        inventory = {
            "repo": "test/repo",
            "quant_hint": "TEST3",
            "parts": [{
                "part_file": THREE_TENSOR_PART_NAME,
                "file_size_bytes": len(self.file_bytes),
                "alignment": header.alignment,
                "data_offset": header.data_offset,
            }],
            "tensors": [gi._tensor_to_dict(t, THREE_TENSOR_PART_NAME) for t in header.tensors],
        }
        self.tmpdir = tempfile.mkdtemp()
        self.weights_dir = os.path.join(self.tmpdir, "weights")
        self.inventory_path = os.path.join(self.tmpdir, "inventory.json")
        self.bin_path = os.path.join(self.tmpdir, "resident.bin")
        self.manifest_path = os.path.join(self.tmpdir, "resident-manifest.json")
        with open(self.inventory_path, "w") as f:
            json.dump(inventory, f)
        local_path = os.path.join(self.weights_dir, THREE_TENSOR_PART_NAME)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(self.file_bytes)

    def _run(self, min_free_gb=0):
        return er.run_extract(self.weights_dir, self.inventory_path, self.bin_path,
                               self.manifest_path, min_free_gb)

    def test_mid_run_failure_persists_prior_tensors_and_rerun_resumes_without_orphaning(self):
        real_hash_range = er._hash_range
        calls = {"n": 0}

        def fake_hash_range(path, offset, nbytes):
            calls["n"] += 1
            if calls["n"] == 3:
                return "0" * 64  # force a write/re-read mismatch on tensor 3 of 3
            return real_hash_range(path, offset, nbytes)  # tensors 1 and 2 verify for real

        with mock.patch.object(er, "_hash_range", side_effect=fake_hash_range):
            with self.assertRaises(er.ExtractError):
                self._run()

        # After the failure: the manifest must already durably reflect the
        # 2 tensors that succeeded before tensor 3 failed -- this is the
        # fix under test. Previously this file wouldn't exist yet here.
        self.assertTrue(os.path.exists(self.manifest_path))
        with open(self.manifest_path) as f:
            manifest_after_failure = json.load(f)
        self.assertEqual(len(manifest_after_failure["tensors"]), 2)
        self.assertEqual(
            {t["name"] for t in manifest_after_failure["tensors"]},
            {"tensor_a.weight", "tensor_b.weight"},
        )
        last = manifest_after_failure["tensors"][-1]
        self.assertEqual(os.path.getsize(self.bin_path), last["offset"] + last["nbytes"])

        # Rerun (fault no longer injected): must resume, extracting only
        # the missing 3rd tensor -- not re-extracting, and not orphaning,
        # the first two.
        manifest2, newly2 = self._run()
        self.assertEqual([t["name"] for t in newly2], ["tensor_c.weight"])
        self.assertEqual(len(manifest2["tensors"]), 3)

        by_name_before = {t["name"]: t["offset"] for t in manifest_after_failure["tensors"]}
        by_name_after = {t["name"]: t["offset"] for t in manifest2["tensors"]}
        self.assertEqual(by_name_before["tensor_a.weight"], by_name_after["tensor_a.weight"])
        self.assertEqual(by_name_before["tensor_b.weight"], by_name_after["tensor_b.weight"])

        # Final resident.bin size equals the sum of manifest entries plus
        # alignment padding: walking the entries in offset order, each
        # one's offset must be the RESIDENT_ALIGNMENT-aligned end of the
        # previous one (no gaps beyond alignment padding, no overlaps),
        # and the file's total size must be exactly the last entry's end
        # -- i.e. no orphaned bytes anywhere in the file.
        ordered = sorted(manifest2["tensors"], key=lambda t: t["offset"])
        expected_pos = 0
        for entry in ordered:
            expected_pos = er._align_up(expected_pos, er.RESIDENT_ALIGNMENT)
            self.assertEqual(entry["offset"], expected_pos)
            expected_pos += entry["nbytes"]
        self.assertEqual(os.path.getsize(self.bin_path), expected_pos)
        self.assertEqual(os.path.getsize(self.bin_path), manifest2["resident_bin_size"])


class VerifyTests(unittest.TestCase):
    """--verify re-reads every manifest-recorded tensor's bytes straight
    from resident.bin and re-hashes them against the manifest's stored
    sha256 -- the trust check the P1 loader needs before mlock'ing
    resident.bin (progress.md's mlock note)."""

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
        for part_file in self.fixture:
            path = os.path.join(self.weights_dir, part_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(self.fixture[part_file][0])
        er.run_extract(self.weights_dir, self.inventory_path, self.bin_path, self.manifest_path, 0)
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

    def _verify(self):
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            rc = er.main(["--verify", "--out-bin", self.bin_path, "--out-manifest", self.manifest_path])
        return rc, stdout.getvalue()

    def test_verify_ok_after_normal_extraction(self):
        rc, out = self._verify()
        self.assertEqual(rc, 0)
        self.assertIn("verify OK 4/4", out)

    def test_verify_flags_flipped_byte_and_names_tensor(self):
        target = self.manifest["tensors"][0]
        with open(self.bin_path, "r+b") as f:
            f.seek(target["offset"])
            byte = f.read(1)
            f.seek(target["offset"])
            f.write(bytes([byte[0] ^ 0xFF]))

        rc, out = self._verify()

        self.assertEqual(rc, 1)
        self.assertIn(target["name"], out)

    def test_verify_flags_short_read_on_truncated_bin(self):
        # tensors are appended in extraction order with strictly increasing
        # offsets, so the last manifest entry is also the last region in
        # the file -- truncating one byte short of its end cuts only it.
        target = self.manifest["tensors"][-1]
        with open(self.bin_path, "r+b") as f:
            f.truncate(target["offset"] + target["nbytes"] - 1)

        rc, out = self._verify()

        self.assertEqual(rc, 1)
        self.assertIn("short read", out)
        self.assertIn(target["name"], out)


if __name__ == "__main__":
    unittest.main()
