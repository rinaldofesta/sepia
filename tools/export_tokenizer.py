#!/usr/bin/env python3
"""Export the Inkling tokenizer from GGUF part 1 metadata into a compact
binary sidecar (weights/tokenizer.bin, "SPTK" v1) for the C tokenizer.

Python 3 stdlib only (imported nothing beyond gguf_inspect). Usage:
    python3 tools/export_tokenizer.py \
        [--gguf weights/inkling-gguf/UD-Q2_K_XL/inkling-UD-Q2_K_XL-00001-of-00008.gguf] \
        [--regex-file docs/tokenizer-pre-regex.txt] [--out weights/tokenizer.bin]

Everything is resolved offline so the C side never touches the GPT-2
byte<->unicode indirection: token strings become raw bytes, merge rules
become (left_id, right_id, merged_id) triples ranked by position.
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_inspect import LocalSource, parse_gguf_header  # noqa: E402

MAGIC = 0x4B545053  # "SPTK"
VERSION = 1

def bytes_to_unicode():
    # GPT-2's byte<->unicode bijection, verbatim semantics.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def token_to_bytes(tok, char_to_byte, token_type):
    if token_type != 1:
        # control/user-defined tokens are literal strings (e.g. <|message_user|>)
        return tok.encode("utf-8")
    out = bytearray()
    for ch in tok:
        if ch not in char_to_byte:
            raise SystemExit(f"export_tokenizer: normal token {tok!r} has char {ch!r} "
                             f"outside the byte-level alphabet")
        out.append(char_to_byte[ch])
    return bytes(out)

def build(md, regex_text):
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"):
        if key not in md:
            raise SystemExit(f"export_tokenizer: GGUF metadata missing {key}")
    if md.get("tokenizer.ggml.model") != "gpt2":
        raise SystemExit(f"export_tokenizer: unexpected tokenizer.ggml.model "
                         f"{md.get('tokenizer.ggml.model')!r} (expected 'gpt2')")
    tokens = md["tokenizer.ggml.tokens"]
    merges = md["tokenizer.ggml.merges"]
    ttypes = md["tokenizer.ggml.token_type"]
    bos = int(md["tokenizer.ggml.bos_token_id"])
    eos = int(md["tokenizer.ggml.eos_token_id"])
    if len(tokens) != len(ttypes):
        raise SystemExit("export_tokenizer: tokens/token_type length mismatch")

    byte_to_char = bytes_to_unicode()
    char_to_byte = {c: b for b, c in byte_to_char.items()}

    tok_id = {t: i for i, t in enumerate(tokens)}
    if len(tok_id) != len(tokens):
        raise SystemExit("export_tokenizer: duplicate token strings in vocab")

    blob = bytearray()
    offs = [0]
    for i, t in enumerate(tokens):
        blob += token_to_bytes(t, char_to_byte, int(ttypes[i]))
        offs.append(len(blob))

    byte_token_id = []
    for b in range(256):
        ch = byte_to_char[b]
        if ch not in tok_id:
            raise SystemExit(f"export_tokenizer: single-byte token for byte {b:#x} missing from vocab")
        byte_token_id.append(tok_id[ch])

    triples = []
    for rank, m in enumerate(merges):
        parts = m.split(" ")
        if len(parts) != 2:
            raise SystemExit(f"export_tokenizer: merge {rank} not two space-separated parts: {m!r}")
        left, right = parts
        merged = left + right
        try:
            triples.append((tok_id[left], tok_id[right], tok_id[merged]))
        except KeyError as e:
            raise SystemExit(f"export_tokenizer: merge {rank} references unknown token {e}")

    specials = [i for i, tt in enumerate(ttypes) if int(tt) != 1]

    out = bytearray()
    out += struct.pack("<7I", MAGIC, VERSION, len(tokens), len(triples), bos, eos, len(specials))
    rx = regex_text.encode("utf-8")
    out += struct.pack("<I", len(rx)) + rx
    out += struct.pack(f"<{256}I", *byte_token_id)
    out += struct.pack(f"<{len(offs)}I", *offs)
    out += bytes(blob)
    out += bytes(min(int(t), 255) for t in ttypes)
    for l, r, mg in triples:
        out += struct.pack("<3I", l, r, mg)
    out += struct.pack(f"<{len(specials)}I", *specials)
    return bytes(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="weights/inkling-gguf/UD-Q2_K_XL/"
                                      "inkling-UD-Q2_K_XL-00001-of-00008.gguf")
    ap.add_argument("--regex-file", default="docs/tokenizer-pre-regex.txt")
    ap.add_argument("--out", default="weights/tokenizer.bin")
    args = ap.parse_args()

    regex_text = Path(args.regex_file).read_text().strip()
    if not regex_text:
        raise SystemExit(f"export_tokenizer: {args.regex_file} is empty")
    src = LocalSource(args.gguf)
    hdr = parse_gguf_header(src)
    data = build(hdr.metadata, regex_text)
    Path(args.out).write_bytes(data)
    print(f"export_tokenizer: wrote {args.out} ({len(data)} bytes, "
          f"{hdr.metadata['tokenizer.ggml.tokens'].__len__()} tokens)")

if __name__ == "__main__":
    main()
