#!/usr/bin/env python3
"""GGUF header inspector.

Parses GGUF v3 headers (magic, version, metadata, tensor infos) from a
local file or, remotely, via HTTP Range reads against Hugging Face --
without ever downloading tensor data. Split-GGUF aware: a set of sibling
`*-NNNNN-of-MMMMM.gguf` parts can be inspected and aggregated in one call.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
Quant type table cross-checked against llama.cpp's gguf-py/gguf/constants.py
(GGMLQuantizationType, GGML_QUANT_SIZES) on 2026-07-19.

Python 3 stdlib only -- this module is imported by later offline tools
(the GGUF-to-inkwell converter), so keep it dependency-free.
"""
import argparse
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSION = 3
DEFAULT_ALIGNMENT = 32

# --- GGUF metadata value type ids (ggml gguf.h) -----------------------------
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# struct format + byte size for each fixed-width scalar metadata value type
_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: ("<B", 1),
    GGUF_TYPE_INT8: ("<b", 1),
    GGUF_TYPE_UINT16: ("<H", 2),
    GGUF_TYPE_INT16: ("<h", 2),
    GGUF_TYPE_UINT32: ("<I", 4),
    GGUF_TYPE_INT32: ("<i", 4),
    GGUF_TYPE_FLOAT32: ("<f", 4),
    GGUF_TYPE_BOOL: ("<?", 1),
    GGUF_TYPE_UINT64: ("<Q", 8),
    GGUF_TYPE_INT64: ("<q", 8),
    GGUF_TYPE_FLOAT64: ("<d", 8),
}

# --- ggml tensor type ids (ggml.h enum ggml_type / gguf-py GGMLQuantizationType)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40
GGML_TYPE_Q1_0 = 41
GGML_TYPE_Q2_0 = 42

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}

# ggml_type id -> (block_size in elements, bytes per block). QK_K = 256.
GGML_TYPE_SIZES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
    27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54),
    35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18),
}


# --- byte sources: local file or remote HTTP Range, both expose read(n)/tell()

class LocalSource:
    """Reads a GGUF header sequentially from a local file."""

    def __init__(self, path):
        self.path = path
        self.total_size = os.path.getsize(path)
        self._f = open(path, "rb")
        self._pos = 0
        self.fetched_bytes = 0

    def read(self, n):
        data = self._f.read(n)
        if len(data) != n:
            raise ValueError(
                f"unexpected EOF reading {self.path}: wanted {n} bytes at "
                f"offset {self._pos}, got {len(data)}"
            )
        self._pos += len(data)
        self.fetched_bytes += len(data)
        return data

    def tell(self):
        return self._pos

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _parse_content_range_total(header_value):
    # format: "bytes start-end/total" (total may be "*" if unknown)
    tail = header_value.rsplit("/", 1)[-1].strip()
    return int(tail) if tail.isdigit() else None


class RemoteSource:
    """Reads a GGUF header via progressive HTTP Range requests.

    Fetches in `chunk_size` chunks starting at offset 0, only as many as
    the parser actually asks for -- tensor data (which starts well after
    the header) is never requested.
    """

    def __init__(self, url, chunk_size=4 * 1024 * 1024, timeout=30.0, retries=1):
        self.url = url
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.retries = retries
        self.total_size = None
        self.fetched_bytes = 0
        self._buf = bytearray()
        self._pos = 0

    def _fetch_range(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        attempts = self.retries + 1
        last_error = None
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    if status not in (200, 206):
                        raise RuntimeError(
                            f"unexpected HTTP status {status} for {self.url} "
                            f"(requested bytes={start}-{end})"
                        )
                    if status == 200 and start != 0:
                        # Server ignored our Range header and is about to hand
                        # back the whole object from byte 0: reading `expected`
                        # bytes here would silently return the WRONG slice.
                        raise RuntimeError(
                            f"{self.url} returned 200 (ignored Range: "
                            f"bytes={start}-{end}); server does not support "
                            "partial content, refusing to continue"
                        )
                    content_range = resp.headers.get("Content-Range")
                    if content_range and self.total_size is None:
                        self.total_size = _parse_content_range_total(content_range)
                    expected = end - start + 1
                    data = resp.read(expected)
                    self.fetched_bytes += len(data)
                    return data
            except OSError as exc:  # covers urllib.error.URLError, timeouts
                last_error = exc
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"failed to fetch bytes={start}-{end} from {self.url} "
                        f"after {attempts} attempt(s): {exc}"
                    ) from exc
                continue
        raise RuntimeError(f"failed to fetch bytes={start}-{end} from {self.url}: {last_error}")

    def _ensure(self, n):
        if self.total_size is not None:
            n = min(n, self.total_size)
        while len(self._buf) < n:
            start = len(self._buf)
            end = start + self.chunk_size - 1
            if self.total_size is not None:
                end = min(end, self.total_size - 1)
            chunk = self._fetch_range(start, end)
            if not chunk:
                raise RuntimeError(
                    f"empty response fetching bytes={start}-{end} from {self.url}; "
                    "file may be shorter than the GGUF header requires"
                )
            self._buf.extend(chunk)

    def read(self, n):
        self._ensure(self._pos + n)
        if self._pos + n > len(self._buf):
            raise ValueError(
                f"reached end of available data ({len(self._buf)} bytes) while "
                f"trying to read {n} bytes at offset {self._pos} from {self.url}"
            )
        data = bytes(self._buf[self._pos:self._pos + n])
        self._pos += n
        return data

    def tell(self):
        return self._pos

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def build_resolve_url(repo, file_path, revision="main"):
    quoted = "/".join(urllib.parse.quote(seg) for seg in file_path.split("/"))
    return f"https://huggingface.co/{repo}/resolve/{revision}/{quoted}"


# --- GGUF header parsing -----------------------------------------------------

def _read_exact(src, n):
    data = src.read(n)
    if len(data) != n:
        raise ValueError(f"short read: wanted {n} bytes, got {len(data)}")
    return data


def _read_u32(src):
    return struct.unpack("<I", _read_exact(src, 4))[0]


def _read_u64(src):
    return struct.unpack("<Q", _read_exact(src, 8))[0]


def _check_count_bound(src, count, min_bytes_per_item, what):
    """A declared count/length read straight off the wire (string length,
    array/tensor/metadata-kv count, dim count) drives a Python loop or a
    single big read next. On a corrupted or truncated file that count can
    be an enormous garbage value; looping or reading that many times before
    hitting a real EOF is a hang, not a fast failure. Bounds `count`
    against what the source's own known total size could possibly still
    hold (every item needs at least `min_bytes_per_item` bytes), and fails
    loudly instead. A no-op if the source doesn't know its total size yet
    (some RemoteSource calls before the first successful fetch)."""
    total_size = getattr(src, "total_size", None)
    if total_size is None:
        return
    remaining = max(total_size - src.tell(), 0)
    max_possible = remaining // min_bytes_per_item if min_bytes_per_item else remaining
    if count > max_possible:
        raise ValueError(
            f"declared {what}={count} would need at least "
            f"{count * min_bytes_per_item:,} bytes but only {remaining:,} remain in "
            f"the {total_size:,}-byte source -- refusing to loop that many times "
            "(likely a corrupted or truncated file)"
        )


def read_gguf_string(src):
    length = _read_u64(src)
    _check_count_bound(src, length, 1, "string length")
    raw = _read_exact(src, length)
    return raw.decode("utf-8")


def _min_bytes_for_value_type(value_type):
    if value_type == GGUF_TYPE_STRING:
        return 8  # at least the string's own length prefix
    if value_type == GGUF_TYPE_ARRAY:
        return 12  # at least elem_type(4) + count(8)
    fmt_size = _SCALAR_FORMATS.get(value_type)
    return fmt_size[1] if fmt_size else 1


def read_metadata_value(src, value_type):
    if value_type == GGUF_TYPE_STRING:
        return read_gguf_string(src)
    if value_type == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(src)
        count = _read_u64(src)
        _check_count_bound(src, count, _min_bytes_for_value_type(elem_type), "array count")
        return [read_metadata_value(src, elem_type) for _ in range(count)]
    fmt_size = _SCALAR_FORMATS.get(value_type)
    if fmt_size is None:
        raise ValueError(f"unknown GGUF metadata value type id {value_type}")
    fmt, size = fmt_size
    return struct.unpack(fmt, _read_exact(src, size))[0]


def _align_up(offset, alignment):
    if not alignment:
        alignment = DEFAULT_ALIGNMENT
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: list
    ggml_type: int
    offset: int

    @property
    def ggml_type_name(self):
        return GGML_TYPE_NAMES.get(self.ggml_type, f"UNKNOWN({self.ggml_type})")

    @property
    def n_elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    def nbytes(self):
        sizes = GGML_TYPE_SIZES.get(self.ggml_type)
        if sizes is None:
            raise ValueError(f"unknown ggml_type {self.ggml_type} for tensor {self.name!r}")
        block_size, type_size = sizes
        n = self.n_elements
        if n % block_size != 0:
            raise ValueError(
                f"tensor {self.name!r}: element count {n} is not a multiple of "
                f"block size {block_size} for type {self.ggml_type_name}"
            )
        return (n // block_size) * type_size


@dataclass
class GGUFHeader:
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata: dict
    tensors: list
    alignment: int
    data_offset: int  # tensor data start, relative to the file's own byte 0
    header_size: int  # bytes consumed by magic..last tensor info (pre-padding)


def parse_gguf_header(src):
    """Parses a GGUF v3 header from `src` (anything exposing read(n)/tell()).
    Never reads past the last tensor info -- tensor data is untouched."""
    magic = _read_exact(src, 4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file: magic bytes were {magic!r}, expected b'GGUF'")
    version = _read_u32(src)
    if version != SUPPORTED_VERSION:
        raise ValueError(f"unsupported GGUF version {version} (this tool only reads v3)")
    tensor_count = _read_u64(src)
    metadata_kv_count = _read_u64(src)
    # A metadata kv needs at least an 8-byte key-length prefix + 4-byte value
    # type id; a tensor info needs at least an 8-byte name-length prefix +
    # 4-byte n_dims + 4-byte ggml_type + 8-byte offset (dims may be empty).
    _check_count_bound(src, metadata_kv_count, 12, "metadata_kv_count")
    _check_count_bound(src, tensor_count, 24, "tensor_count")

    metadata = {}
    for _ in range(metadata_kv_count):
        key = read_gguf_string(src)
        value_type = _read_u32(src)
        metadata[key] = read_metadata_value(src, value_type)

    tensors = []
    for _ in range(tensor_count):
        name = read_gguf_string(src)
        n_dims = _read_u32(src)
        _check_count_bound(src, n_dims, 8, "tensor n_dims")
        dims = [_read_u64(src) for _ in range(n_dims)]
        ggml_type = _read_u32(src)
        offset = _read_u64(src)
        tensors.append(TensorInfo(name=name, n_dims=n_dims, dims=dims, ggml_type=ggml_type, offset=offset))

    header_size = src.tell()
    alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    data_offset = _align_up(header_size, alignment)

    return GGUFHeader(
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        data_offset=data_offset,
        header_size=header_size,
    )


# --- split-GGUF discovery -----------------------------------------------------

_SPLIT_RE = re.compile(r"^(?P<prefix>.*)-(?P<index>\d+)-of-(?P<total>\d+)\.gguf$")


def split_part_paths(file_path, total_override=None):
    """Given one split-GGUF part path, returns the paths of all parts
    (1..N), preserving the directory and the zero-padding width."""
    dirname, _, basename = file_path.rpartition("/")
    m = _SPLIT_RE.match(basename)
    if not m:
        raise ValueError(
            f"{file_path!r} does not look like a split GGUF part "
            "(expected a name like *-NNNNN-of-MMMMM.gguf)"
        )
    width = len(m.group("index"))
    total = int(total_override) if total_override is not None else int(m.group("total"))
    prefix = m.group("prefix")
    names = [f"{prefix}-{i:0{width}d}-of-{total:0{width}d}.gguf" for i in range(1, total + 1)]
    return [f"{dirname}/{n}" if dirname else n for n in names]


# --- CLI ----------------------------------------------------------------------

def _tensor_to_dict(tensor, part_file):
    return {
        "part_file": part_file,
        "name": tensor.name,
        "shape": tensor.dims,
        "ggml_type": tensor.ggml_type,
        "ggml_type_name": tensor.ggml_type_name,
        "n_bytes": tensor.nbytes(),
        "offset": tensor.offset,
    }


_METADATA_ARRAY_SAMPLE = 8


def _summarize_metadata(metadata):
    """Full metadata dict, but large arrays (e.g. the ~200k-entry tokenizer
    vocab) are truncated to a sample so the JSON stays reviewable."""
    out = {}
    for key, value in metadata.items():
        if isinstance(value, list) and len(value) > _METADATA_ARRAY_SAMPLE:
            out[key] = {
                "truncated": True,
                "count": len(value),
                "sample": value[:_METADATA_ARRAY_SAMPLE],
            }
        else:
            out[key] = value
    return out


def _open_local(path):
    return LocalSource(path)


def _open_remote(repo, path, chunk_size, timeout, retries):
    return RemoteSource(build_resolve_url(repo, path), chunk_size=chunk_size, timeout=timeout, retries=retries)


def inspect_one(opener, path):
    with opener(path) as src:
        header = parse_gguf_header(src)
        return path, header, src.fetched_bytes, src.total_size


def inspect_all_parts(opener, base_path):
    """Iterates every split sibling of base_path. Returns a list of
    (path, header, bytes_fetched, file_size) tuples, one per part."""
    first_path, first_header, first_fetched, first_size = inspect_one(opener, base_path)
    total_from_meta = first_header.metadata.get("split.count")
    all_paths = split_part_paths(base_path, total_override=total_from_meta)

    results = [(first_path, first_header, first_fetched, first_size)]
    for path in all_paths:
        if path == first_path:
            continue
        results.append(inspect_one(opener, path))
    # keep numeric order even though base_path may not have been part 1
    results.sort(key=lambda r: r[0])
    return results


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", help="Hugging Face repo id, e.g. unsloth/inkling-GGUF (remote mode)")
    p.add_argument("--file", required=True, help="repo-relative path (remote mode) or filesystem path (--local)")
    p.add_argument("--local", action="store_true", help="treat --file as a local filesystem path")
    p.add_argument("--all-parts", action="store_true", help="discover and aggregate every split sibling of --file")
    p.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024, help="bytes per HTTP Range request (default 4MB)")
    p.add_argument("--timeout", type=float, default=30.0, help="socket timeout in seconds (default 30)")
    p.add_argument("--retries", type=int, default=1, help="retries per HTTP request after the first attempt (default 1)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a human summary")
    p.add_argument("--out", help="write JSON to this path (implies --json)")
    return p


def _print_human_summary(results, repo_or_local):
    total_fetched = sum(r[2] for r in results)
    total_file_bytes = sum(r[3] or 0 for r in results)
    print(f"source: {repo_or_local}")
    print(f"parts inspected: {len(results)}")
    print(f"bytes fetched (header reads only): {total_fetched:,}")
    if total_file_bytes:
        print(f"total file bytes across parts: {total_file_bytes:,}")
    for path, header, fetched, size in results:
        print(f"\n== {path} ==")
        print(f"  version={header.version} tensor_count={header.tensor_count} "
              f"metadata_kv_count={header.metadata_kv_count} alignment={header.alignment} "
              f"data_offset={header.data_offset}")
        print(f"  bytes fetched for this part's header: {fetched:,}"
              + (f" (of {size:,} total file bytes)" if size else ""))
        for t in header.tensors:
            print(f"    {t.name}  shape={t.dims}  type={t.ggml_type_name}  "
                  f"n_bytes={t.nbytes()}  offset={t.offset}")


def _to_json_doc(results, repo, quant_hint):
    parts = []
    tensors = []
    for path, header, fetched, size in results:
        parts.append({
            "part_file": path,
            "file_size_bytes": size,
            "version": header.version,
            "tensor_count": header.tensor_count,
            "metadata_kv_count": header.metadata_kv_count,
            "alignment": header.alignment,
            "data_offset": header.data_offset,
            "header_bytes_fetched": fetched,
            "metadata": _summarize_metadata(header.metadata),
        })
        for t in header.tensors:
            tensors.append(_tensor_to_dict(t, path))
    return {
        "repo": repo,
        "quant_hint": quant_hint,
        "parts": parts,
        "tensors": tensors,
    }


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    if args.local:
        opener = _open_local
        source_label = args.file
    else:
        if not args.repo:
            print("error: --repo is required unless --local is given", file=sys.stderr)
            return 2
        opener = lambda path: _open_remote(args.repo, path, args.chunk_size, args.timeout, args.retries)  # noqa: E731
        source_label = f"{args.repo}:{args.file}"

    # One try/except covers both parsing (inspect_*) and output generation
    # (_to_json_doc / _print_human_summary): both call tensor.nbytes(),
    # which raises ValueError for an unrecognized ggml_type, and that error
    # deserves the same clean "error: ..." exit as a parse failure, not an
    # uncaught traceback.
    try:
        if args.all_parts:
            results = inspect_all_parts(opener, args.file)
        else:
            results = [inspect_one(opener, args.file)]

        if args.json or args.out:
            quant_hint = args.file.split("/")[0] if "/" in args.file else None
            doc = _to_json_doc(results, args.repo if not args.local else None, quant_hint)
            text = json.dumps(doc, indent=2)
            if args.out:
                with open(args.out, "w") as f:
                    f.write(text)
                    f.write("\n")
                print(f"wrote {args.out}", file=sys.stderr)
            else:
                print(text)
        else:
            _print_human_summary(results, source_label)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
