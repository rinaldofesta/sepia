#!/usr/bin/env python3
"""Build tokenizer test fixtures.

Dev-only (.venv/bin/python tools/make_tokenizer_fixtures.py). Requires the
`tokenizers` package (in .venv) and, for the real fixtures, weights/tokenizer.bin
(Task 7) — the reference tokenizer is constructed from the SAME vocab/merges/
regex the C tokenizer reads, so fixtures and engine share one source of truth.
"""
import json
import struct
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tools" / "fixtures" / "tokenizer"
REGEX = (REPO / "docs" / "tokenizer-pre-regex.txt").read_text().strip()

CASES = [
    "Hello, world!",
    "hello",
    " hello",
    "  leading spaces",
    "trailing spaces  ",
    "The capital of France is",
    "don't we'll I'm you've it's CAN'T",
    "MixedCASE Words And ALLCAPS",
    "x = f(y) + 3;  // comment",
    "def main():\n    return 0\n",
    "1 22 333 4444 55555 1234567890",
    "3.14159, -42, 1e-8",
    "tabs\tand\nnewlines\r\nmixed \n\n done",
    "punct!!! ??? ... --- ///path/to/file",
    "città perché naïve Übermensch açúcar",
    "日本語のテキスト 中文文本 한국어",
    "русский текст ελληνικά عربى",
    "emoji 🙂🚀 and 👍🏽 skin tone",
    "ﬁ ligature, ½ fraction, № sign",
    "a",
    " ",
    "\n",
    "",
    "word" * 50,
    "\xa0nbsp and \u2028line sep",
]

def gpt2_byte_alphabet():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def read_sptk(path):
    d = Path(path).read_bytes()
    magic, ver, vocab, n_merges, bos, eos, n_special = struct.unpack_from("<7I", d, 0)
    assert magic == 0x4B545053 and ver == 1, "bad tokenizer.bin"
    off = 28
    (rlen,) = struct.unpack_from("<I", d, off); off += 4 + rlen
    off += 256 * 4
    offs = struct.unpack_from(f"<{vocab+1}I", d, off); off += (vocab + 1) * 4
    blob = d[off: off + offs[-1]]; off += offs[-1]
    ttypes = d[off: off + vocab]; off += vocab
    merges = [struct.unpack_from("<3I", d, off + 12 * i)[:2] for i in range(n_merges)]
    toks_raw = [blob[offs[i]:offs[i+1]] for i in range(vocab)]
    return toks_raw, ttypes, merges, bos, eos

def build_reference(toks_raw, ttypes, merge_id_pairs):
    b2u = gpt2_byte_alphabet()
    def to_char_space(i):
        if ttypes[i] != 1:
            return toks_raw[i].decode("utf-8")
        return "".join(b2u[b] for b in toks_raw[i])
    vocab = {to_char_space(i): i for i in range(len(toks_raw))}
    merges = [(to_char_space(l), to_char_space(r)) for l, r in merge_id_pairs]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok

def emit_cases(tok, path, cases):
    out = []
    for text in cases:
        ids = tok.encode(text).ids
        assert tok.decode(ids, skip_special_tokens=False) == text or text == "", \
            f"roundtrip failed for {text!r}"
        out.append({"text": text, "ids": ids})
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path} ({len(out)} cases)")

def make_mini():
    """Tiny self-contained SPTK + cases: 256 byte tokens + a few merges."""
    b2u = gpt2_byte_alphabet()
    toks = [b2u[b] for b in range(256)]
    # explicit merge list building "hello" and " the" and "ing" chains:
    merges = [
        (b2u[ord("h")], b2u[ord("e")]),                     # he
        (b2u[ord("h")] + b2u[ord("e")], b2u[ord("l")]),     # hel
        (b2u[ord("h")] + b2u[ord("e")] + b2u[ord("l")], b2u[ord("l")]),   # hell
        (b2u[ord("h")] + b2u[ord("e")] + b2u[ord("l")] + b2u[ord("l")], b2u[ord("o")]),  # hello
        (b2u[ord(" ")], b2u[ord("t")]),                     # ' t'
        (b2u[ord(" ")] + b2u[ord("t")], b2u[ord("h")]),     # ' th'
        (b2u[ord(" ")] + b2u[ord("t")] + b2u[ord("h")], b2u[ord("e")]),   # ' the'
        (b2u[ord("i")], b2u[ord("n")]),                     # in
        (b2u[ord("i")] + b2u[ord("n")], b2u[ord("g")]),     # ing
    ]
    for l, r in merges:
        toks.append(l + r)
    toks.append("<|end|>")
    ttypes = [1] * (len(toks) - 1) + [3]
    tok_id = {t: i for i, t in enumerate(toks)}
    u2b = {c: b for b, c in b2u.items()}

    blob = bytearray(); offs = [0]
    for i, t in enumerate(toks):
        raw = t.encode() if ttypes[i] != 1 else bytes(u2b[c] for c in t)
        blob += raw; offs.append(len(blob))
    byte_token_id = [tok_id[b2u[b]] for b in range(256)]
    triples = [(tok_id[l], tok_id[r], tok_id[l + r]) for l, r in merges]
    specials = [len(toks) - 1]

    out = bytearray()
    out += struct.pack("<7I", 0x4B545053, 1, len(toks), len(triples),
                       specials[0], specials[0], len(specials))
    rx = REGEX.encode(); out += struct.pack("<I", len(rx)) + rx
    out += struct.pack("<256I", *byte_token_id)
    out += struct.pack(f"<{len(offs)}I", *offs)
    out += bytes(blob)
    out += bytes(ttypes)
    for t in triples:
        out += struct.pack("<3I", *t)
    out += struct.pack(f"<{len(specials)}I", *specials)
    (FIX / "mini.bin").write_bytes(out)
    print(f"wrote {FIX/'mini.bin'} ({len(out)} bytes)")

    ref = build_reference([blob[offs[i]:offs[i+1]] for i in range(len(toks))],
                          bytes(ttypes), [(l, r) for l, r, _ in triples])
    emit_cases(ref, FIX / "mini_cases.json", [
        "hello", " the thing", "hello hello", "in the end", "ing",
        "The", "  hello\n", "xyz", "héllo",
    ])

def make_real():
    toks_raw, ttypes, merges, bos, eos = read_sptk(REPO / "weights" / "tokenizer.bin")
    ref = build_reference(toks_raw, ttypes, merges)
    emit_cases(ref, FIX / "real_cases.json", CASES)

if __name__ == "__main__":
    FIX.mkdir(parents=True, exist_ok=True)
    make_mini()
    make_real()
