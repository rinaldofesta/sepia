#!/usr/bin/env python3
"""Generate bit-exact dequant fixtures for tools/test_quants.c.

Dev-only tool (NOT stdlib-only; excluded from CI):
    uvx --from gguf --with numpy python tools/make_quant_fixtures.py
Writes tools/fixtures/quants/<type>.bin and qlinear_q8_0.bin (Task 13).
Reference implementation: gguf-py's gguf.quants (independent of ggml C code).
"""
import struct
from pathlib import Path

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T
from gguf.quants import dequantize, quantize

OUT_DIR = Path(__file__).resolve().parent / "fixtures" / "quants"
MAGIC = 0x58465153  # "SQFX"
N_BLOCKS = 64
SEED = 20260720

QUANTIZABLE = [T.Q8_0]                        # the only container type gguf-py can quantize
RAW_BYTES = {  # qtype -> byte offsets of the f16 scale fields to patch in each block
    T.Q4_K: (0, 2), T.Q5_K: (0, 2), T.Q6_K: (208,),
    T.IQ2_XS: (0,), T.IQ3_XXS: (0,), T.IQ4_XS: (0,),
}

def finite_f16_bits(rng, n):
    # positive normal f16: exponent field 1..30 (31 would be Inf/NaN), random mantissa
    exp = rng.integers(0x01, 0x1F, size=n, dtype=np.uint16)
    mant = rng.integers(0, 0x400, size=n, dtype=np.uint16)
    return (exp << np.uint16(10)) | mant

def write_fixture(qtype, raw, expect):
    blk, tsz = GGML_QUANT_SIZES[qtype]
    n_blocks = len(raw) // tsz
    assert len(raw) == n_blocks * tsz and expect.size == n_blocks * blk
    assert expect.dtype == np.float32
    assert np.isfinite(expect).all(), f"{qtype.name}: non-finite expected values"
    path = OUT_DIR / f"{qtype.name.lower()}.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, 1, int(qtype), n_blocks, blk, tsz))
        f.write(raw)
        f.write(expect.astype("<f4").tobytes())
    print(f"{path}  {n_blocks} blocks, {path.stat().st_size} bytes")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # f16 fixture: every one of 64 blocks is one f16 value; include denormals + 0
    f16_bits = np.concatenate([
        np.array([0x0000, 0x8000, 0x0001, 0x83FF, 0x3C00, 0xBC00, 0x7BFF], dtype=np.uint16),
        finite_f16_bits(rng, N_BLOCKS - 7),
    ])
    expect = f16_bits.view(np.float16).astype(np.float32)
    path = OUT_DIR / "f16.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<6I", MAGIC, 1, int(T.F16), N_BLOCKS, 1, 2))
        f.write(f16_bits.astype("<u2").tobytes())
        f.write(expect.astype("<f4").tobytes())
    print(f"{path}  {N_BLOCKS} blocks, {path.stat().st_size} bytes")

    for qt in QUANTIZABLE:
        blk, tsz = GGML_QUANT_SIZES[qt]
        data = rng.standard_normal(N_BLOCKS * blk, dtype=np.float32) * 3.0
        raw = quantize(data, qt).tobytes()
        expect = dequantize(np.frombuffer(raw, dtype=np.uint8), qt).astype(np.float32)
        write_fixture(qt, raw, expect)

    for qt, patch_offs in RAW_BYTES.items():
        blk, tsz = GGML_QUANT_SIZES[qt]
        raw = rng.integers(0, 256, size=(N_BLOCKS, tsz), dtype=np.uint8)
        for off in patch_offs:
            d = finite_f16_bits(rng, N_BLOCKS).view(np.uint8).reshape(N_BLOCKS, 2)
            raw[:, off:off + 2] = d
        raw = raw.reshape(-1).tobytes()
        expect = dequantize(np.frombuffer(raw, dtype=np.uint8), qt).astype(np.float32)
        write_fixture(qt, raw, expect)

    make_qlinear_fixture(rng)   # defined in Task 13; stub `pass` until then

def make_qlinear_fixture(rng):
    pass

if __name__ == "__main__":
    main()
