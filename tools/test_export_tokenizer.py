#!/usr/bin/env python3
"""Unit tests for export_tokenizer.py against a synthetic GGUF (stdlib only)."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_tokenizer as ex

def pack_kv_str(key, val):
    kb, vb = key.encode(), val.encode()
    return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) +
            struct.pack("<Q", len(vb)) + vb)

def pack_kv_u32(key, val):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", val)

def pack_kv_bool(key, val):
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 7) + struct.pack("<?", val)

def pack_kv_str_array(key, vals):
    kb = key.encode()
    out = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9)      # array
    out += struct.pack("<I", 8) + struct.pack("<Q", len(vals))        # of string
    for v in vals:
        vb = v.encode()
        out += struct.pack("<Q", len(vb)) + vb
    return out

def pack_kv_i32_array(key, vals):
    kb = key.encode()
    out = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9)
    out += struct.pack("<I", 5) + struct.pack("<Q", len(vals))        # of int32
    out += struct.pack(f"<{len(vals)}i", *vals)
    return out

B2U = ex.bytes_to_unicode()
H = B2U[ord("h")]; I = B2U[ord("i")]; SP = B2U[ord(" ")]
# build() requires every one of the 256 raw bytes to have its own single-char
# token in the vocab (the real GPT-2 byte-level alphabet always does); h, i,
# and space are already covered by the first three tokens below, so the
# filler list covers the other 253 bytes to make this a valid fixture.
_FILLER_BYTES = [b for b in range(256) if b not in (ord("h"), ord("i"), ord(" "))]
TOKENS = [H, I, SP, H + I, SP + H, "<|x|>"] + [B2U[b] for b in _FILLER_BYTES]
TTYPES = [1, 1, 1, 1, 1, 3] + [1] * len(_FILLER_BYTES)
MERGES = [f"{H} {I}", f"{SP} {H}"]

def synth_gguf(path):
    kvs = (pack_kv_str("tokenizer.ggml.model", "gpt2") +
           pack_kv_str("tokenizer.ggml.pre", "inkling") +
           pack_kv_str_array("tokenizer.ggml.tokens", TOKENS) +
           pack_kv_i32_array("tokenizer.ggml.token_type", TTYPES) +
           pack_kv_str_array("tokenizer.ggml.merges", MERGES) +
           pack_kv_u32("tokenizer.ggml.bos_token_id", 5) +
           pack_kv_u32("tokenizer.ggml.eos_token_id", 5) +
           pack_kv_bool("tokenizer.ggml.add_bos_token", False) +
           pack_kv_u32("general.alignment", 32))
    hdr = b"GGUF" + struct.pack("<IQQ", 3, 0, 9)
    path.write_bytes(hdr + kvs)

class ExportTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gguf = Path(self.tmp.name) / "toy.gguf"
        synth_gguf(self.gguf)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        with ex.LocalSource(str(self.gguf)) as src:
            hdr = ex.parse_gguf_header(src)
        return ex.build(hdr.metadata, "REGEX")

    def test_header_counts(self):
        data = self.build()
        magic, ver, vocab, n_merges, bos, eos, n_special = struct.unpack_from("<7I", data, 0)
        self.assertEqual((magic, ver), (ex.MAGIC, 1))
        self.assertEqual((vocab, n_merges, bos, eos, n_special),
                         (len(TOKENS), 2, 5, 5, 1))

    def test_merge_triples_resolved(self):
        data = self.build()
        # walk: 28B header + regex + byte_token_id + offsets + blob + token_type
        off = 28
        (rlen,) = struct.unpack_from("<I", data, off); off += 4 + rlen
        off += 256 * 4
        offs = struct.unpack_from(f"<{len(TOKENS)+1}I", data, off); off += (len(TOKENS) + 1) * 4
        off += offs[-1] + len(TOKENS)
        l0, r0, m0 = struct.unpack_from("<3I", data, off)
        self.assertEqual((l0, r0, m0), (0, 1, 3))   # "h"+"i" -> "hi"

    def test_control_token_bytes_are_literal(self):
        data = self.build()
        off = 28
        (rlen,) = struct.unpack_from("<I", data, off); off += 4 + rlen
        off += 256 * 4
        offs = struct.unpack_from(f"<{len(TOKENS)+1}I", data, off); off += (len(TOKENS) + 1) * 4
        blob = data[off: off + offs[-1]]
        self.assertEqual(blob[offs[5]:offs[6]], b"<|x|>")

    def test_missing_key_dies(self):
        with self.assertRaises(SystemExit):
            ex.build({"tokenizer.ggml.model": "gpt2"}, "R")

if __name__ == "__main__":
    unittest.main()
