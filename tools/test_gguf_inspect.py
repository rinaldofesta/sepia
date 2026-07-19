#!/usr/bin/env python3
"""No-network unit tests for gguf_inspect.py.

Builds a tiny synthetic GGUF v3 header by hand (struct-packed, no tensor
data section) and asserts the parser extracts names, shapes, ggml types,
offsets and metadata correctly. Runnable directly: python3 tools/test_gguf_inspect.py
"""
import io
import os
import struct
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inspect as gi  # noqa: E402


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_metadata_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    return _pack_string(key) + struct.pack("<I", value_type) + value_bytes


def build_synthetic_gguf() -> bytes:
    """Assembles a valid GGUF v3 header (magic..tensor-infos), no data
    section: the parser must never need bytes past the last tensor info,
    so a header-only file is a valid, meaningful fixture."""
    metadata_entries = [
        _pack_metadata_kv("general.architecture", gi.GGUF_TYPE_STRING, _pack_string("test-arch")),
        _pack_metadata_kv("general.alignment", gi.GGUF_TYPE_UINT32, struct.pack("<I", 32)),
        # the one split-style key
        _pack_metadata_kv("split.count", gi.GGUF_TYPE_UINT32, struct.pack("<I", 4)),
        _pack_metadata_kv("test.float", gi.GGUF_TYPE_FLOAT32, struct.pack("<f", 3.5)),
        _pack_metadata_kv("test.flag", gi.GGUF_TYPE_BOOL, struct.pack("<?", True)),
    ]

    def _pack_tensor_info(name: str, dims: list, ggml_type: int, offset: int) -> bytes:
        out = _pack_string(name)
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", offset)
        return out

    tensor_entries = [
        # F32, 8x4 -> 32 elements, block_size 1, type_size 4 -> 128 bytes
        _pack_tensor_info("token_embd.weight", [8, 4], gi.GGML_TYPE_F32, 0),
        # F16, 8x8 -> 64 elements -> 128 bytes; next offset aligned to 32 -> 128
        _pack_tensor_info("blk.0.attn_q.weight", [8, 8], gi.GGML_TYPE_F16, 128),
        # Q4_0, 3D expert tensor [in=4, out=8, n_expert=2] -> 64 elements,
        # block_size 32 -> 2 blocks * 18 bytes = 36 bytes; offset aligned -> 256
        _pack_tensor_info("blk.0.ffn_gate_exps.weight", [4, 8, 2], gi.GGML_TYPE_Q4_0, 256),
    ]

    header = b"GGUF"
    header += struct.pack("<I", 3)  # version
    header += struct.pack("<Q", len(tensor_entries))  # tensor_count
    header += struct.pack("<Q", len(metadata_entries))  # metadata_kv_count
    for entry in metadata_entries:
        header += entry
    for entry in tensor_entries:
        header += entry
    return header


class ParseGGUFHeaderTests(unittest.TestCase):
    def setUp(self):
        self.raw = build_synthetic_gguf()

    def test_version_and_counts(self):
        header = gi.parse_gguf_header(io.BytesIO(self.raw))
        self.assertEqual(header.version, 3)
        self.assertEqual(header.tensor_count, 3)
        self.assertEqual(header.metadata_kv_count, 5)

    def test_metadata_values_by_type(self):
        header = gi.parse_gguf_header(io.BytesIO(self.raw))
        self.assertEqual(header.metadata["general.architecture"], "test-arch")
        self.assertEqual(header.metadata["general.alignment"], 32)
        self.assertEqual(header.metadata["split.count"], 4)
        self.assertAlmostEqual(header.metadata["test.float"], 3.5, places=5)
        self.assertIs(header.metadata["test.flag"], True)

    def test_tensor_names_shapes_types_offsets(self):
        header = gi.parse_gguf_header(io.BytesIO(self.raw))
        by_name = {t.name: t for t in header.tensors}

        embd = by_name["token_embd.weight"]
        self.assertEqual(embd.dims, [8, 4])
        self.assertEqual(embd.ggml_type, gi.GGML_TYPE_F32)
        self.assertEqual(embd.offset, 0)
        self.assertEqual(embd.nbytes(), 128)

        attn = by_name["blk.0.attn_q.weight"]
        self.assertEqual(attn.dims, [8, 8])
        self.assertEqual(attn.ggml_type, gi.GGML_TYPE_F16)
        self.assertEqual(attn.offset, 128)
        self.assertEqual(attn.nbytes(), 128)

        experts = by_name["blk.0.ffn_gate_exps.weight"]
        self.assertEqual(experts.dims, [4, 8, 2])
        self.assertEqual(experts.ggml_type, gi.GGML_TYPE_Q4_0)
        self.assertEqual(experts.offset, 256)
        self.assertEqual(experts.nbytes(), 36)
        self.assertEqual(experts.ggml_type_name, "Q4_0")

    def test_alignment_and_data_offset(self):
        header = gi.parse_gguf_header(io.BytesIO(self.raw))
        self.assertEqual(header.alignment, 32)
        # header-only fixture: header_size == len(raw); data_offset is that,
        # rounded up to the alignment.
        expected_data_offset = ((len(self.raw) + 31) // 32) * 32
        self.assertEqual(header.data_offset, expected_data_offset)
        self.assertEqual(header.data_offset % header.alignment, 0)

    def test_bad_magic_raises(self):
        bad = b"OOPS" + self.raw[4:]
        with self.assertRaises(ValueError):
            gi.parse_gguf_header(io.BytesIO(bad))

    def test_unsupported_version_raises(self):
        bad = self.raw[:4] + struct.pack("<I", 2) + self.raw[8:]
        with self.assertRaises(ValueError):
            gi.parse_gguf_header(io.BytesIO(bad))


class LocalSourceTests(unittest.TestCase):
    def test_local_source_matches_in_memory_parse(self):
        raw = build_synthetic_gguf()
        direct = gi.parse_gguf_header(io.BytesIO(raw))

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            with gi.LocalSource(path) as src:
                via_file = gi.parse_gguf_header(src)
        finally:
            os.unlink(path)

        self.assertEqual(via_file.tensor_count, direct.tensor_count)
        self.assertEqual(via_file.metadata, direct.metadata)
        self.assertEqual([t.name for t in via_file.tensors], [t.name for t in direct.tensors])
        self.assertEqual(via_file.data_offset, direct.data_offset)
        # local mode never reads past the header either
        self.assertEqual(via_file.header_size, len(raw))


class SplitPartPathsTests(unittest.TestCase):
    def test_generates_all_sibling_parts_with_zero_padding(self):
        paths = gi.split_part_paths(
            "UD-Q2_K_XL/inkling-UD-Q2_K_XL-00001-of-00008.gguf"
        )
        self.assertEqual(len(paths), 8)
        self.assertEqual(paths[0], "UD-Q2_K_XL/inkling-UD-Q2_K_XL-00001-of-00008.gguf")
        self.assertEqual(paths[-1], "UD-Q2_K_XL/inkling-UD-Q2_K_XL-00008-of-00008.gguf")

    def test_total_override(self):
        paths = gi.split_part_paths("inkling-00001-of-00001.gguf", total_override=3)
        self.assertEqual(
            paths,
            [
                "inkling-00001-of-00003.gguf",
                "inkling-00002-of-00003.gguf",
                "inkling-00003-of-00003.gguf",
            ],
        )

    def test_non_split_filename_raises(self):
        with self.assertRaises(ValueError):
            gi.split_part_paths("mmproj-BF16.gguf")


class _FakeResponse:
    """Stands in for the object urllib.request.urlopen() returns, without
    opening a socket."""

    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    def read(self, n=-1):
        return self._body if n is None or n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class RemoteSourceRobustnessTests(unittest.TestCase):
    """Exercises the brief's explicit robustness requirements (timeout +
    one retry; clear error on an unexpected status) without any network
    access, by faking urlopen's return value / exceptions."""

    def test_retries_once_then_succeeds(self):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("simulated transient failure")
            return _FakeResponse(206, b"GGUF1234", {"Content-Range": "bytes 0-7/100"})

        with mock.patch.object(gi.urllib.request, "urlopen", side_effect=fake_urlopen):
            src = gi.RemoteSource("https://example.invalid/f.gguf", chunk_size=8, retries=1)
            data = src.read(8)

        self.assertEqual(data, b"GGUF1234")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(src.total_size, 100)

    def test_raises_clear_error_after_exhausting_retries(self):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("still failing")

        with mock.patch.object(gi.urllib.request, "urlopen", side_effect=fake_urlopen):
            src = gi.RemoteSource("https://example.invalid/f.gguf", chunk_size=8, retries=1)
            with self.assertRaises(RuntimeError):
                src.read(8)

    def test_raises_clear_error_on_unexpected_status(self):
        def fake_urlopen(req, timeout=None):
            return _FakeResponse(404, b"")

        with mock.patch.object(gi.urllib.request, "urlopen", side_effect=fake_urlopen):
            src = gi.RemoteSource("https://example.invalid/f.gguf", chunk_size=8, retries=1)
            with self.assertRaises(RuntimeError):
                src.read(8)

    def test_raises_when_server_silently_ignores_range(self):
        # First chunk (start=0) legitimately gets 206; the second chunk
        # (start=8) comes back 200, meaning the server ignored our Range
        # and would hand back bytes from the file's start again -- reading
        # blindly here would splice in the wrong bytes.
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(206, b"AAAAAAAA", {"Content-Range": "bytes 0-7/100"})
            return _FakeResponse(200, b"whole-file-from-byte-zero")

        with mock.patch.object(gi.urllib.request, "urlopen", side_effect=fake_urlopen):
            src = gi.RemoteSource("https://example.invalid/f.gguf", chunk_size=8, retries=1)
            with self.assertRaises(RuntimeError):
                src.read(16)  # needs a second chunk starting at offset 8


if __name__ == "__main__":
    unittest.main()
