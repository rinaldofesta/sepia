#!/usr/bin/env python3
"""Extracts every resident (non-expert) tensor from the Inkling UD-Q2_K_XL
GGUF parts into a single `weights/resident.bin` plus a JSON manifest.

Per `docs/DESIGN.md`'s container plan, `resident.bin` (embeddings,
attention, shared experts, routers, norms, output head; ~14GB) is the one
piece SEPIA still extracts rather than streaming in place -- it's what gets
mlock'd, and it's where the future MTP sidecar attaches. Everything else
(the 256-expert routed pool, 95.5% of tensor bytes) stays in the GGUF parts
and is addressed via `tools/make_index.py`'s sidecar instead.

Reuses `tools/make_index.py`'s resident-tensor listing (which itself reuses
`gguf_inspect.py`'s parser) instead of re-deriving the local/inventory
merge and cross-check logic a second time.

The 317GB download may still be running: this tool extracts whatever
resident tensors live in *fully downloaded* parts, records the rest as
`pending_parts` in the manifest, and exits 0 -- a clean, expected state,
not an error. Rerunning later is idempotent: already-extracted tensors
(tracked by name in the existing manifest) are skipped; only newly
available ones get appended. `resident.bin` is append-only -- an entry's
offset never changes once written, so nothing already extracted is ever
rewritten or reordered.

Every tensor's bytes are SHA256'd while streaming from the source GGUF
part, then the just-written region of resident.bin is re-read and
re-hashed; the two digests must match before the tensor is added to the
manifest, or the tool aborts loudly and truncates resident.bin back to
its state before that tensor (so a re-run always finds a clean, fully
verified file to resume from). The manifest itself is persisted
(atomically: written to a temp file, then renamed into place) right after
*each* tensor is successfully verified, not just once at the end -- so a
crash or failure partway through a run (e.g. tensor 3 of 3) still leaves a
durable, accurate manifest for everything extracted before it. Without
this, a mid-run failure would leave already-verified bytes physically in
resident.bin with no manifest entry pointing at them; a rerun would
re-extract those tensors from scratch, appending fresh copies and
permanently orphaning the originals.

Python 3 stdlib only.

Usage:
    python3 tools/extract_resident.py
    python3 tools/extract_resident.py --min-free-gb 100
    python3 tools/extract_resident.py --verify
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_index as mi  # noqa: E402

DEFAULT_WEIGHTS_DIR = mi.DEFAULT_WEIGHTS_DIR
DEFAULT_INVENTORY = mi.DEFAULT_INVENTORY
DEFAULT_OUT_BIN = "weights/resident.bin"
DEFAULT_OUT_MANIFEST = "weights/resident-manifest.json"
DEFAULT_MIN_FREE_GB = 50
RESIDENT_ALIGNMENT = 64
CHUNK_BYTES = 4 * 1024 * 1024


class ExtractError(Exception):
    """Raised for any condition that should abort loudly: a write/re-read
    SHA256 mismatch, insufficient free space, or a manifest tensor whose
    recorded shape no longer matches the current index."""


def _align_up(n, a):
    return n if n % a == 0 else n + (a - n % a)


def _check_free_space(target_dir, min_free_gb):
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        raise ExtractError(
            f"only {free_gb:.1f}GB free on the filesystem holding {target_dir!r}, "
            f"below the --min-free-gb guard ({min_free_gb}GB); aborting before writing anything"
        )
    return free_gb


def _load_manifest(path):
    if not os.path.exists(path):
        return {"tensors": []}
    with open(path) as f:
        return json.load(f)


def _write_manifest_atomic(manifest_path, manifest):
    """Writes manifest_path atomically (temp file + os.replace) so a crash
    mid-write never leaves a corrupt/partial JSON file -- and so repeated
    calls, one per successfully extracted tensor, are each a durable
    checkpoint an interrupted run can resume from."""
    directory = os.path.dirname(os.path.abspath(manifest_path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".resident-manifest-", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, manifest_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _hash_range(path, offset, nbytes):
    hasher = hashlib.sha256()
    remaining = nbytes
    with open(path, "rb") as f:
        f.seek(offset)
        while remaining > 0:
            chunk = f.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ExtractError(f"unexpected EOF reading {path!r} at offset {f.tell()}")
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest()


def _extract_one(tensor, weights_dir, bin_path):
    """Streams one resident tensor's bytes from its source GGUF part into
    resident.bin (appended, RESIDENT_ALIGNMENT-aligned), SHA256ing during
    the write, then re-reads the written range and re-hashes it. Returns
    the manifest entry, or raises ExtractError and leaves resident.bin
    truncated back to its pre-call length on any failure."""
    pre_len = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
    start_pos = _align_up(pre_len, RESIDENT_ALIGNMENT)
    source_path = os.path.join(weights_dir, tensor["part_file"])

    try:
        with open(source_path, "rb") as src:
            src.seek(tensor["abs_offset"])
            with open(bin_path, "r+b" if os.path.exists(bin_path) else "w+b") as dst:
                dst.seek(pre_len)
                dst.write(b"\x00" * (start_pos - pre_len))  # alignment padding
                hasher = hashlib.sha256()
                remaining = tensor["nbytes"]
                while remaining > 0:
                    chunk = src.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ExtractError(
                            f"unexpected EOF reading {tensor['name']!r} from {source_path!r}"
                        )
                    hasher.update(chunk)
                    dst.write(chunk)
                    remaining -= len(chunk)
                dst.flush()
                os.fsync(dst.fileno())
        write_sha256 = hasher.hexdigest()

        reread_sha256 = _hash_range(bin_path, start_pos, tensor["nbytes"])
        if write_sha256 != reread_sha256:
            raise ExtractError(
                f"{tensor['name']!r}: SHA256 mismatch between write ({write_sha256}) "
                f"and re-read ({reread_sha256}) -- resident.bin write may be corrupt"
            )
    except ExtractError:
        os.truncate(bin_path, pre_len)
        raise

    return {
        "name": tensor["name"],
        "ggml_type": tensor["ggml_type"],
        "ggml_type_name": tensor["ggml_type_name"],
        "shape": tensor["shape"],
        "offset": start_pos,
        "nbytes": tensor["nbytes"],
        "sha256": write_sha256,
        "source_part_file": tensor["part_file"],
        "source_abs_offset": tensor["abs_offset"],
    }


def run_extract(weights_dir, inventory_path, bin_path, manifest_path, min_free_gb):
    doc, _parts = mi.build_index(weights_dir, inventory_path)
    resident = doc["resident_tensors"]

    _check_free_space(os.path.dirname(os.path.abspath(bin_path)) or ".", min_free_gb)

    manifest = _load_manifest(manifest_path)
    done_tensors = list(manifest.get("tensors", []))
    done_by_name = {t["name"]: t for t in done_tensors}

    current_by_name = {t["name"]: t for t in resident}
    for name, prior in done_by_name.items():
        cur = current_by_name.get(name)
        if cur is None:
            print(f"warning: {name!r} is in the existing manifest but not in the current "
                  "index; leaving it in place", file=sys.stderr)
            continue
        if (prior["shape"], prior["ggml_type"], prior["nbytes"]) != (cur["shape"], cur["ggml_type"], cur["nbytes"]):
            raise ExtractError(
                f"{name!r}: already-extracted manifest entry disagrees with the current "
                f"index (shape/type/nbytes changed) -- refusing to proceed idempotently"
            )

    newly_extracted = []
    pending_parts = set()

    def _current_manifest_doc():
        all_tensors = done_tensors + newly_extracted
        return {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "resident_bin_path": bin_path,
            "resident_bin_size": os.path.getsize(bin_path) if os.path.exists(bin_path) else 0,
            "alignment": RESIDENT_ALIGNMENT,
            "n_resident_tensors_total": len(resident),
            "n_tensors_extracted": len(all_tensors),
            "tensors": all_tensors,
            "pending_parts": sorted(pending_parts),
        }

    for tensor in resident:
        if tensor["name"] in done_by_name:
            continue
        if tensor["source"] != "local":
            pending_parts.add(tensor["part_file"])
            continue
        _check_free_space(os.path.dirname(os.path.abspath(bin_path)) or ".", min_free_gb)
        entry = _extract_one(tensor, weights_dir, bin_path)
        newly_extracted.append(entry)
        # Persisted right after this one tensor is durably verified on disk,
        # not just once at the end -- see the module docstring for why.
        _write_manifest_atomic(manifest_path, _current_manifest_doc())

    # Final write: covers the 0-new-tensors case (pending_parts/generated_at
    # still need refreshing) and is otherwise a cheap, idempotent re-write
    # of what the last incremental write already persisted.
    final_manifest = _current_manifest_doc()
    _write_manifest_atomic(manifest_path, final_manifest)

    return final_manifest, newly_extracted


def _print_summary(manifest, newly_extracted):
    print(f"resident.bin: {manifest['resident_bin_size']:,} bytes "
          f"({manifest['n_tensors_extracted']}/{manifest['n_resident_tensors_total']} tensors extracted)")
    print(f"newly extracted this run: {len(newly_extracted)}")
    if manifest["pending_parts"]:
        print(f"pending parts (not yet fully downloaded, {len(manifest['pending_parts'])}):")
        for p in manifest["pending_parts"]:
            print(f"  {p}")
    else:
        print("pending parts: none -- resident extraction is complete")


def verify(manifest_path, bin_path):
    """Re-reads every manifest-recorded tensor's bytes straight from
    bin_path and re-hashes them against the manifest's stored sha256 --
    the trust check the P1 loader needs before it mlocks resident.bin.
    Does not extract or modify anything. Prints one line per tensor that
    fails (short read, or a sha256 mismatch) plus a summary line, and
    returns 0 if every tensor matched or 1 if any did not."""
    if not os.path.exists(manifest_path):
        raise ExtractError(f"no manifest at {manifest_path!r} -- nothing to verify")
    if not os.path.exists(bin_path):
        raise ExtractError(f"no resident.bin at {bin_path!r} -- nothing to verify")
    with open(manifest_path) as f:
        manifest = json.load(f)
    tensors = manifest.get("tensors", [])

    bad = 0
    with open(bin_path, "rb") as f:
        for t in tensors:
            f.seek(t["offset"])
            data = f.read(t["nbytes"])
            if len(data) != t["nbytes"]:
                print(f"verify FAIL {t['name']}: short read ({len(data)}/{t['nbytes']} bytes)")
                bad += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest != t["sha256"]:
                print(f"verify FAIL {t['name']}: sha256 {digest} != manifest {t['sha256']}")
                bad += 1

    n = len(tensors)
    if bad:
        print(f"verify: {bad}/{n} tensors FAILED")
        return 1
    print(f"verify OK {n}/{n}")
    return 0


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--inventory", default=DEFAULT_INVENTORY)
    p.add_argument("--out-bin", default=DEFAULT_OUT_BIN)
    p.add_argument("--out-manifest", default=DEFAULT_OUT_MANIFEST)
    p.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                    help=f"abort before writing if free disk space is below this (default {DEFAULT_MIN_FREE_GB}GB)")
    p.add_argument("--verify", action="store_true",
                    help="re-hash every manifest tensor's bytes in --out-bin against its "
                         "recorded sha256 and exit 0/1; does not extract anything")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.verify:
        try:
            return verify(args.out_manifest, args.out_bin)
        except ExtractError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        manifest, newly_extracted = run_extract(
            args.weights_dir, args.inventory, args.out_bin, args.out_manifest, args.min_free_gb
        )
    except (mi.IndexBuildError, ExtractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_summary(manifest, newly_extracted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
