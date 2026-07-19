#!/usr/bin/env python3
"""No-network unit tests for gguf_inspect.py.

Builds a tiny synthetic GGUF v3 header by hand (struct-packed, no tensor
data section) and asserts the parser extracts names, shapes, ggml types,
offsets and metadata correctly. Runnable directly: python3 tools/test_gguf_inspect.py
"""
import io
import os
import shutil
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


def _pack_tensor_info(name: str, dims: list, ggml_type: int, offset: int) -> bytes:
    out = _pack_string(name)
    out += struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", ggml_type)
    out += struct.pack("<Q", offset)
    return out


def _pack_gguf(metadata_entries: list, tensor_entries: list) -> bytes:
    header = b"GGUF"
    header += struct.pack("<I", 3)  # version
    header += struct.pack("<Q", len(tensor_entries))  # tensor_count
    header += struct.pack("<Q", len(metadata_entries))  # metadata_kv_count
    for entry in metadata_entries:
        header += entry
    for entry in tensor_entries:
        header += entry
    return header


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

    tensor_entries = [
        # F32, 8x4 -> 32 elements, block_size 1, type_size 4 -> 128 bytes
        _pack_tensor_info("token_embd.weight", [8, 4], gi.GGML_TYPE_F32, 0),
        # F16, 8x8 -> 64 elements -> 128 bytes; next offset aligned to 32 -> 128
        _pack_tensor_info("blk.0.attn_q.weight", [8, 8], gi.GGML_TYPE_F16, 128),
        # Q4_0, 3D expert tensor [in=4, out=8, n_expert=2] -> 64 elements,
        # block_size 32 -> 2 blocks * 18 bytes = 36 bytes; offset aligned -> 256
        _pack_tensor_info("blk.0.ffn_gate_exps.weight", [4, 8, 2], gi.GGML_TYPE_Q4_0, 256),
    ]

    return _pack_gguf(metadata_entries, tensor_entries)


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


class CountBoundTests(unittest.TestCase):
    """Task 0.6 review Minor (a): a declared string/array/tensor/metadata_kv
    count read off the wire drives a loop next; on a corrupted or truncated
    file that count can be enormous. These must fail fast (ValueError)
    rather than loop/read for a very long time before hitting a real EOF."""

    def _write_temp(self, raw: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(raw)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_huge_metadata_kv_count_fails_fast_not_hangs(self):
        raw = b"GGUF" + struct.pack("<I", 3)
        raw += struct.pack("<Q", 0)  # tensor_count
        raw += struct.pack("<Q", 10**15)  # metadata_kv_count: absurd, no bytes follow
        path = self._write_temp(raw)
        with gi.LocalSource(path) as src:
            with self.assertRaises(ValueError):
                gi.parse_gguf_header(src)

    def test_huge_tensor_count_fails_fast_not_hangs(self):
        raw = b"GGUF" + struct.pack("<I", 3)
        raw += struct.pack("<Q", 10**15)  # tensor_count: absurd
        raw += struct.pack("<Q", 0)  # metadata_kv_count
        path = self._write_temp(raw)
        with gi.LocalSource(path) as src:
            with self.assertRaises(ValueError):
                gi.parse_gguf_header(src)

    def test_huge_string_length_fails_fast(self):
        metadata_entries = [
            _pack_metadata_kv("k", gi.GGUF_TYPE_STRING, struct.pack("<Q", 10**15)),
        ]
        raw = _pack_gguf(metadata_entries, [])
        path = self._write_temp(raw)
        with gi.LocalSource(path) as src:
            with self.assertRaises(ValueError):
                gi.parse_gguf_header(src)

    def test_huge_array_count_fails_fast(self):
        # array of uint8 (elem_type=0... actually GGUF_TYPE_UINT8=0), count huge
        value_bytes = struct.pack("<I", gi.GGUF_TYPE_UINT8) + struct.pack("<Q", 10**15)
        metadata_entries = [_pack_metadata_kv("k", gi.GGUF_TYPE_ARRAY, value_bytes)]
        raw = _pack_gguf(metadata_entries, [])
        path = self._write_temp(raw)
        with gi.LocalSource(path) as src:
            with self.assertRaises(ValueError):
                gi.parse_gguf_header(src)

    def test_huge_n_dims_fails_fast(self):
        tensor_entries = [
            # n_dims itself is absurd; no dims data follows
            _pack_string("t") + struct.pack("<I", 2**31),
        ]
        raw = _pack_gguf([], tensor_entries)
        path = self._write_temp(raw)
        with gi.LocalSource(path) as src:
            with self.assertRaises(ValueError):
                gi.parse_gguf_header(src)

    def test_valid_small_counts_still_parse_fine(self):
        # a bound check that's too aggressive would break legitimate files;
        # the existing synthetic fixture (3 tensors, 5 metadata kvs) must
        # still parse cleanly through the same bound-checked code paths.
        header = gi.parse_gguf_header(io.BytesIO(build_synthetic_gguf()))
        self.assertEqual(header.tensor_count, 3)
        self.assertEqual(header.metadata_kv_count, 5)


class UnknownTypeCleanExitTests(unittest.TestCase):
    """Task 0.6 review Minor (b): an unrecognized ggml_type must exit
    through the same clean 'error: ...' path as any other CLI error
    (widened main() try/except), not an uncaught traceback."""

    def _write_unknown_type_file(self) -> str:
        # ggml_type 99 is not in GGML_TYPE_SIZES: TensorInfo.nbytes() raises
        # ValueError for it, but only when nbytes() is actually called --
        # i.e. during output generation, not during parsing itself.
        tensor_entries = [_pack_tensor_info("weird.weight", [4], 99, 0)]
        raw = _pack_gguf([], tensor_entries)
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(raw)
            self.addCleanup(os.unlink, f.name)
            return f.name

    def test_json_mode_reports_clean_error_and_exit_1(self):
        path = self._write_unknown_type_file()
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            rc = gi.main(["--file", path, "--local", "--json"])
        self.assertEqual(rc, 1)
        self.assertIn("error:", stderr.getvalue())
        self.assertIn("unknown ggml_type", stderr.getvalue())

    def test_human_summary_mode_reports_clean_error_and_exit_1(self):
        path = self._write_unknown_type_file()
        stderr = io.StringIO()
        with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", stderr):
            rc = gi.main(["--file", path, "--local"])
        self.assertEqual(rc, 1)
        self.assertIn("error:", stderr.getvalue())


class InspectAllPartsAndJsonDocTests(unittest.TestCase):
    """Task 0.6 review Minor (c): synthetic two-part fixture (no network)
    covering inspect_all_parts, _to_json_doc, and _summarize_metadata."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmpdir, ignore_errors=True))

        part1_tensors = [_pack_tensor_info("token_embd.weight", [4, 2], gi.GGML_TYPE_F32, 0)]
        part1_meta = [
            _pack_metadata_kv("general.architecture", gi.GGUF_TYPE_STRING, _pack_string("test-arch")),
            _pack_metadata_kv("general.alignment", gi.GGUF_TYPE_UINT32, struct.pack("<I", 32)),
        ]
        part2_tensors = [_pack_tensor_info("blk.0.attn_q.weight", [4, 4], gi.GGML_TYPE_F16, 0)]
        part2_meta = [
            _pack_metadata_kv("general.architecture", gi.GGUF_TYPE_STRING, _pack_string("test-arch")),
            _pack_metadata_kv("general.alignment", gi.GGUF_TYPE_UINT32, struct.pack("<I", 32)),
        ]

        self.part1_path = os.path.join(self.tmpdir, "model-00001-of-00002.gguf")
        self.part2_path = os.path.join(self.tmpdir, "model-00002-of-00002.gguf")
        with open(self.part1_path, "wb") as f:
            f.write(_pack_gguf(part1_meta, part1_tensors))
        with open(self.part2_path, "wb") as f:
            f.write(_pack_gguf(part2_meta, part2_tensors))

    def test_inspect_all_parts_discovers_and_orders_both_parts(self):
        results = gi.inspect_all_parts(gi._open_local, self.part1_path)
        self.assertEqual(len(results), 2)
        paths = [r[0] for r in results]
        self.assertEqual(paths, sorted(paths))
        self.assertTrue(paths[0].endswith("00001-of-00002.gguf"))
        self.assertTrue(paths[1].endswith("00002-of-00002.gguf"))
        names_by_part = {r[0]: [t.name for t in r[1].tensors] for r in results}
        self.assertEqual(names_by_part[self.part1_path], ["token_embd.weight"])
        self.assertEqual(names_by_part[self.part2_path], ["blk.0.attn_q.weight"])

    def test_to_json_doc_shape(self):
        results = gi.inspect_all_parts(gi._open_local, self.part1_path)
        doc = gi._to_json_doc(results, repo=None, quant_hint="TESTQUANT")
        self.assertEqual(doc["quant_hint"], "TESTQUANT")
        self.assertEqual(len(doc["parts"]), 2)
        self.assertEqual(len(doc["tensors"]), 2)
        by_name = {t["name"]: t for t in doc["tensors"]}
        self.assertEqual(by_name["token_embd.weight"]["ggml_type_name"], "F32")
        self.assertEqual(by_name["token_embd.weight"]["n_bytes"], 32)  # 8 elems * 4 bytes
        self.assertEqual(by_name["blk.0.attn_q.weight"]["ggml_type_name"], "F16")
        self.assertEqual(by_name["blk.0.attn_q.weight"]["part_file"], self.part2_path)
        # every part in the doc carries its own tensor_count/alignment
        part_tensor_counts = {p["part_file"]: p["tensor_count"] for p in doc["parts"]}
        self.assertEqual(part_tensor_counts[self.part1_path], 1)
        self.assertEqual(part_tensor_counts[self.part2_path], 1)

    def test_summarize_metadata_truncates_large_arrays_only(self):
        big_array = list(range(20))
        small_array = [1, 2, 3]
        metadata = {
            "general.architecture": "test-arch",
            "tokenizer.ggml.tokens": big_array,
            "small.list": small_array,
            "scalar.value": 42,
        }
        summary = gi._summarize_metadata(metadata)
        self.assertEqual(summary["general.architecture"], "test-arch")
        self.assertEqual(summary["small.list"], small_array)
        self.assertEqual(summary["scalar.value"], 42)
        self.assertEqual(summary["tokenizer.ggml.tokens"], {
            "truncated": True,
            "count": 20,
            "sample": big_array[:gi._METADATA_ARRAY_SAMPLE],
        })


if __name__ == "__main__":
    unittest.main()
