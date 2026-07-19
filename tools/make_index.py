#!/usr/bin/env python3
"""Builds the SEPIA GGUF index sidecar for Inkling UD-Q2_K_XL.

Per `docs/ssd-bench.md`'s no-repack verdict, SEPIA streams every expert's
gate/up/down tensors directly from the source GGUF parts -- no converter, no
`experts-NN.bin` slabs. This tool builds the sidecar that makes that
consumable: for every one of the 64 MoE layers' 256 routed experts, the
exact `(part_file, abs_offset, nbytes, ggml_type)` of its gate/up/down
slices, computed from `docs/gguf-inventory-ud-q2_k_xl.md`'s confirmed
per-expert contiguity (`abs_offset = part_data_start + tensor.offset +
e * bytes_per_expert`); plus the resident (non-expert) tensor list and a
header block (alignment, part sizes, quant histogram, provenance).

Reuses `gguf_inspect.py`'s GGUF parser (no duplicated parsing logic). Works
from local part files where the 317GB download has completed them, and
falls back to the committed inventory JSON (`docs/gguf-inventory-ud-q2_k_xl
.json`, authoritative for expected values) for parts not yet downloaded --
each part is tagged with its source ("local" | "inventory") in the output.
Where a part is available both ways, the local parse is cross-checked
against the inventory and any mismatch (name/type/offset) is a loud
failure, not a silent divergence.

The index JSON this tool produces (default
`weights/inkling-ud-q2_k_xl.sepia-index.json`) is derived data and stays
untracked; this tool is what's committed.

Python 3 stdlib only.

Usage:
    python3 tools/make_index.py
    python3 tools/make_index.py --verify 20
    python3 tools/make_index.py --out /tmp/index.json --no-write-if-exists
"""
import argparse
import datetime
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inspect as gi  # noqa: E402

N_EXPERTS = 256
DEFAULT_WEIGHTS_DIR = "weights/inkling-gguf"
DEFAULT_INVENTORY = "docs/gguf-inventory-ud-q2_k_xl.json"
DEFAULT_OUT = "weights/inkling-ud-q2_k_xl.sepia-index.json"

_EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


class IndexBuildError(Exception):
    """Raised for any condition that should abort loudly: an inventory/local
    mismatch, a missing inventory, or a verification failure."""


# --- loading the two tensor sources -----------------------------------------

def load_inventory(path):
    if not os.path.exists(path):
        raise IndexBuildError(
            f"inventory JSON not found at {path!r} -- it is the required "
            "fallback/cross-check source for parts not yet downloaded "
            "(regenerate with tools/gguf_inspect.py --all-parts --json)"
        )
    with open(path) as f:
        doc = json.load(f)
    for key in ("repo", "parts", "tensors"):
        if key not in doc:
            raise IndexBuildError(f"inventory JSON {path!r} is missing required key {key!r}")
    return doc


def _inventory_tensors_by_part(inventory):
    by_part = {}
    for t in inventory["tensors"]:
        by_part.setdefault(t["part_file"], []).append(t)
    return by_part


def _parse_local_part(local_path):
    """Parses one local GGUF part's header, returning gguf_inspect's
    GGUFHeader. Any parse failure propagates as-is (loud, not swallowed)."""
    with gi.LocalSource(local_path) as src:
        return gi.parse_gguf_header(src)


def _local_tensor_dicts(header, part_file):
    # Reuses gguf_inspect's own tensor->dict shape (name/shape/ggml_type/
    # ggml_type_name/n_bytes/offset/part_file) so local and inventory
    # tensors are directly comparable without a second schema.
    return [gi._tensor_to_dict(t, part_file) for t in header.tensors]


def _cross_check(local_tensors, inventory_tensors, part_file):
    """Fails loudly on any divergence between a locally-parsed part and the
    inventory's record of the same part: same tensor names, and matching
    ggml_type/offset/shape for each. The inventory is authoritative for
    expected values (docs/gguf-inventory-ud-q2_k_xl.md), so any mismatch
    here means either the download is corrupt or the inventory is stale --
    both need a human, not a silent fallback."""
    local_by_name = {t["name"]: t for t in local_tensors}
    inv_by_name = {t["name"]: t for t in inventory_tensors}
    local_names, inv_names = set(local_by_name), set(inv_by_name)
    if local_names != inv_names:
        only_local = sorted(local_names - inv_names)
        only_inv = sorted(inv_names - local_names)
        raise IndexBuildError(
            f"tensor set mismatch for {part_file!r} between local parse and "
            f"inventory: only in local={only_local[:5]}"
            f"{'...' if len(only_local) > 5 else ''}, "
            f"only in inventory={only_inv[:5]}{'...' if len(only_inv) > 5 else ''}"
        )
    mismatches = []
    for name, lt in local_by_name.items():
        it = inv_by_name[name]
        if (lt["ggml_type"], lt["offset"], lt["shape"]) != (it["ggml_type"], it["offset"], it["shape"]):
            mismatches.append(
                f"{name}: local(type={lt['ggml_type']}, offset={lt['offset']}, shape={lt['shape']}) "
                f"!= inventory(type={it['ggml_type']}, offset={it['offset']}, shape={it['shape']})"
            )
    if mismatches:
        raise IndexBuildError(
            f"local/inventory mismatch for {part_file!r} ({len(mismatches)} tensor(s)): "
            + "; ".join(mismatches[:5])
            + ("; ..." if len(mismatches) > 5 else "")
        )


def resolve_parts(weights_dir, inventory):
    """Returns an ordered list of part records:
    {part_file, source ("local"|"inventory"), expected_size_bytes,
     local_size_bytes, data_start, alignment, tensors}.

    A part counts as "local" only if a file exists at its expected path AND
    its size exactly matches the inventory's expected size -- a partial
    download (which the download client keeps hidden under a *.incomplete
    name until it completes, but checked here defensively regardless)
    should fall back to inventory, not be parsed half-written."""
    inv_tensors_by_part = _inventory_tensors_by_part(inventory)
    parts = []
    for inv_part in inventory["parts"]:
        part_file = inv_part["part_file"]
        expected_size = inv_part["file_size_bytes"]
        local_path = os.path.join(weights_dir, part_file)
        local_size = os.path.getsize(local_path) if os.path.exists(local_path) else None

        inv_tensors = inv_tensors_by_part.get(part_file, [])

        if local_size is not None and local_size == expected_size:
            header = _parse_local_part(local_path)
            local_tensors = _local_tensor_dicts(header, part_file)
            _cross_check(local_tensors, inv_tensors, part_file)
            parts.append({
                "part_file": part_file,
                "source": "local",
                "expected_size_bytes": expected_size,
                "local_size_bytes": local_size,
                "data_start": header.data_offset,
                "alignment": header.alignment,
                "tensors": local_tensors,
            })
        else:
            if local_size is not None:
                print(
                    f"warning: {local_path!r} exists but size {local_size:,} != "
                    f"expected {expected_size:,}; treating as not-yet-downloaded "
                    "(falling back to inventory)",
                    file=sys.stderr,
                )
            parts.append({
                "part_file": part_file,
                "source": "inventory",
                "expected_size_bytes": expected_size,
                "local_size_bytes": local_size,
                "data_start": inv_part["data_offset"],
                "alignment": inv_part["alignment"],
                "tensors": inv_tensors,
            })
    return parts


# --- assembling the index doc -----------------------------------------------

def _bytes_per_expert(tensor_dict):
    n_bytes, shape = tensor_dict["n_bytes"], tensor_dict["shape"]
    if len(shape) != 3 or shape[-1] != N_EXPERTS:
        raise IndexBuildError(
            f"{tensor_dict['name']!r} does not look like a {N_EXPERTS}-expert tensor "
            f"(shape={shape})"
        )
    if n_bytes % N_EXPERTS != 0:
        raise IndexBuildError(
            f"{tensor_dict['name']!r}: n_bytes={n_bytes} is not an exact multiple of "
            f"{N_EXPERTS} experts"
        )
    return n_bytes // N_EXPERTS


_REQUIRED_EXPERT_KINDS = {"gate", "up", "down"}


def _build_moe_layers(parts):
    """Returns {layer: {"gate"|"up"|"down": tensor_block}}, where each
    tensor_block carries the per-tensor arithmetic (bytes_per_expert etc.)
    the C loader can verify against, plus the 256 per-expert entries."""
    layers = {}
    for part in parts:
        for t in part["tensors"]:
            m = _EXPERT_TENSOR_RE.match(t["name"])
            if not m:
                continue
            layer, kind = int(m.group(1)), m.group(2)
            bpe = _bytes_per_expert(t)
            data_start = part["data_start"]
            experts = [
                {
                    "part_file": t["part_file"],
                    "abs_offset": data_start + t["offset"] + e * bpe,
                    "nbytes": bpe,
                    "ggml_type": t["ggml_type"],
                }
                for e in range(N_EXPERTS)
            ]
            layers.setdefault(layer, {})[kind] = {
                "tensor_name": t["name"],
                "part_file": t["part_file"],
                "source": part["source"],
                "ggml_type": t["ggml_type"],
                "ggml_type_name": t["ggml_type_name"],
                "data_start": data_start,
                "tensor_offset": t["offset"],
                "n_bytes_total": t["n_bytes"],
                "n_expert": N_EXPERTS,
                "bytes_per_expert": bpe,
                "experts": experts,
            }

    # A MoE layer must have all three of gate/up/down or none at all -- a
    # layer with only 1 or 2 of them is a corrupted inventory/local part,
    # not something a partial index entry should paper over silently.
    for layer, kinds in layers.items():
        missing = _REQUIRED_EXPERT_KINDS - kinds.keys()
        if missing:
            raise IndexBuildError(
                f"MoE layer {layer} is missing expert tensor kind(s) {sorted(missing)} "
                f"(found {sorted(kinds.keys())}) -- refusing to emit a partial layer"
            )
    return layers


def _build_resident_tensors(parts):
    resident = []
    for part in parts:
        for t in part["tensors"]:
            if _EXPERT_TENSOR_RE.match(t["name"]):
                continue
            resident.append({
                "name": t["name"],
                "part_file": t["part_file"],
                "source": part["source"],
                "abs_offset": part["data_start"] + t["offset"],
                "nbytes": t["n_bytes"],
                "ggml_type": t["ggml_type"],
                "ggml_type_name": t["ggml_type_name"],
                "shape": t["shape"],
            })
    return resident


def _quant_histogram(parts):
    hist = {}
    for part in parts:
        for t in part["tensors"]:
            entry = hist.setdefault(t["ggml_type_name"], {"tensor_count": 0, "bytes": 0})
            entry["tensor_count"] += 1
            entry["bytes"] += t["n_bytes"]
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]["bytes"]))


def build_index(weights_dir, inventory_path):
    inventory = load_inventory(inventory_path)
    parts = resolve_parts(weights_dir, inventory)

    alignments = {p["alignment"] for p in parts}
    if len(alignments) != 1:
        raise IndexBuildError(f"parts disagree on alignment: {sorted(alignments)}")
    alignment = alignments.pop()

    moe_layers = _build_moe_layers(parts)
    resident_tensors = _build_resident_tensors(parts)

    n_layers = len(moe_layers)
    n_experts_indexed = sum(len(kinds.get(k, {}).get("experts", []))
                             for kinds in moe_layers.values() for k in ("gate", "up", "down"))

    doc = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_repo": inventory["repo"],
        "quant": inventory.get("quant_hint"),
        "alignment": alignment,
        "n_experts_per_layer": N_EXPERTS,
        "n_moe_layers_indexed": n_layers,
        "n_expert_tensor_entries": n_experts_indexed,
        "parts": [
            {
                "part_file": p["part_file"],
                "source": p["source"],
                "expected_size_bytes": p["expected_size_bytes"],
                "local_size_bytes": p["local_size_bytes"],
                "data_start": p["data_start"],
                "alignment": p["alignment"],
            }
            for p in parts
        ],
        "quant_type_histogram": _quant_histogram(parts),
        "resident_tensors": resident_tensors,
        "moe_layers": {str(layer): kinds for layer, kinds in sorted(moe_layers.items())},
    }
    return doc, parts


# --- --verify N --------------------------------------------------------------

def _fresh_lookup(local_path, tensor_name, e):
    """Independent code path #2 for --verify: re-parses the part's header
    fresh from disk right now (not the cached index doc) and walks its
    tensor list looking for tensor_name, recomputing abs_offset/nbytes from
    scratch. Returns (abs_offset, nbytes) or None if not found."""
    header = _parse_local_part(local_path)
    for t in header.tensors:
        if t.name == tensor_name:
            bpe = t.nbytes() // N_EXPERTS
            return header.data_offset + t.offset + e * bpe, bpe
    return None


def run_verify(doc, parts, weights_dir, n, rng):
    parts_by_file = {p["part_file"]: p for p in parts}
    candidates = []
    for layer_str, kinds in doc["moe_layers"].items():
        for kind, block in kinds.items():
            for e, entry in enumerate(block["experts"]):
                candidates.append((block["tensor_name"], e, entry, parts_by_file[entry["part_file"]]))

    if not candidates:
        print("verify: no MoE expert tensors in the index (nothing to verify)")
        return True

    picks = [candidates[rng.randrange(len(candidates))] for _ in range(n)]

    ok = skipped = failed = 0
    failures = []
    for tensor_name, e, entry, part in picks:
        if part["source"] != "local" or part["local_size_bytes"] != part["expected_size_bytes"]:
            skipped += 1
            continue
        local_path = os.path.join(weights_dir, part["part_file"])
        try:
            direct_offset, direct_nbytes = entry["abs_offset"], entry["nbytes"]
            fresh = _fresh_lookup(local_path, tensor_name, e)
            if fresh is None:
                raise IndexBuildError(f"{tensor_name!r} not found walking a fresh parse of {local_path!r}")
            fresh_offset, fresh_nbytes = fresh
            if (direct_offset, direct_nbytes) != (fresh_offset, fresh_nbytes):
                raise IndexBuildError(
                    f"{tensor_name!r} expert {e}: direct-arithmetic "
                    f"(offset={direct_offset}, nbytes={direct_nbytes}) != fresh-walk "
                    f"(offset={fresh_offset}, nbytes={fresh_nbytes})"
                )
            if direct_offset + direct_nbytes > part["local_size_bytes"]:
                raise IndexBuildError(
                    f"{tensor_name!r} expert {e}: slice [{direct_offset}, "
                    f"{direct_offset + direct_nbytes}) crosses part EOF ({part['local_size_bytes']})"
                )
            with open(local_path, "rb") as f:
                f.seek(direct_offset)
                data_a = f.read(direct_nbytes)
                f.seek(fresh_offset)
                data_b = f.read(fresh_nbytes)
            if data_a != data_b or len(data_a) != direct_nbytes:
                raise IndexBuildError(
                    f"{tensor_name!r} expert {e}: byte-compare across the two read paths failed"
                )
            ok += 1
        except IndexBuildError as exc:
            failed += 1
            failures.append(str(exc))

    print(f"verify: OK {ok}/{n} (skipped {skipped} not-yet-downloaded, {failed} failed)")
    if failures:
        for msg in failures[:10]:
            print(f"  FAIL: {msg}", file=sys.stderr)
        return False
    return True


# --- CLI ----------------------------------------------------------------------

def _print_summary(doc, parts):
    print(f"source_repo={doc['source_repo']} quant={doc['quant']} alignment={doc['alignment']}")
    for p in doc["parts"]:
        size_note = f"{p['local_size_bytes']:,}" if p["local_size_bytes"] is not None else "not present"
        print(f"  {p['part_file']}: source={p['source']} local_size={size_note} "
              f"expected={p['expected_size_bytes']:,}")
    local_n = sum(1 for p in doc["parts"] if p["source"] == "local")
    print(f"parts: {local_n} local, {len(doc['parts']) - local_n} inventory-fallback")
    print(f"MoE layers indexed: {doc['n_moe_layers_indexed']} "
          f"(expert tensor entries: {doc['n_expert_tensor_entries']:,})")
    print(f"resident tensors: {len(doc['resident_tensors']):,}")
    for name, entry in doc["quant_type_histogram"].items():
        print(f"  {name}: {entry['tensor_count']} tensors, {entry['bytes']:,} bytes")


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR,
                    help=f"local GGUF parts directory (default {DEFAULT_WEIGHTS_DIR})")
    p.add_argument("--inventory", default=DEFAULT_INVENTORY,
                    help=f"inventory JSON path (default {DEFAULT_INVENTORY})")
    p.add_argument("--out", default=DEFAULT_OUT,
                    help=f"index JSON output path (default {DEFAULT_OUT})")
    p.add_argument("--no-write", action="store_true", help="build (and optionally verify) but don't write --out")
    p.add_argument("--verify", type=int, default=0, metavar="N",
                    help="byte-verify N random experts whose parts are fully downloaded")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for --verify (default: unseeded)")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    try:
        doc, parts = build_index(args.weights_dir, args.inventory)
    except IndexBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(doc, parts)

    if not args.no_write:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")

    if args.verify:
        rng = random.Random(args.seed)
        ok = run_verify(doc, parts, args.weights_dir, args.verify, rng)
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
