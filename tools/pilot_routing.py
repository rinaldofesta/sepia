#!/usr/bin/env python3
"""PILOT routing-predictability analysis (SEPIA P2 Task 13).

Reads one or more --route-log binary files written by `sepia --metal --real
--route-log FILE` (see the RouteLogRecord comment in src/sepia.c) and
computes two independent signals over the recorded MoE routing decisions:

  1. One-layer-ahead recall@6 -- the PILOT/colibri signal. For each (token,
     layer L) that has a routing record, treat layer L's 6 selected expert
     ids as a PREDICTION of layer L+1's 6 selected ids for the SAME token
     (only when L+1 also has a record for that token; dense layers have no
     router and are naturally absent, so the pair is skipped there and at
     the last layer, which has no L+1 at all). recall@6 = |predicted n
     actual| / 6. Reported as a per-layer curve (mean over all tokens with
     that pair) and one aggregate mean over every (token, layer-pair)
     instance across every input file.

  2. Same-layer consecutive-token overlap -- the cache/LRU temporal-locality
     signal. NOT the same measurement as (1): this asks whether token t's
     expert set at a FIXED layer L overlaps with token t+1's expert set at
     that SAME layer, which is what actually drives an LRU cache's hit rate
     (distinct from next-layer predictability). For each layer L and each
     token t where L has records at both t and t+1 *within the same input
     file* (token positions are never compared across different prompts/
     files -- each file's own generation restarts its token_idx at 0),
     overlap = |ids(L,t) n ids(L,t+1)| / 6.

Stdlib only (struct + argparse), no numpy/torch -- this is just reading
fixed-size binary records and counting/averaging. Deterministic: every file
is read once into an in-memory dict and every summation iterates in sorted
(layer, token) order regardless of on-disk record order, so re-running
against the same input file(s) reproduces bit-identical output.

Record format (56 bytes, native byte order -- produced and consumed on the
same machine, see src/sepia.c's RouteLogRecord comment for the authoritative
description):
    int32 token_idx, int32 layer_idx, int32 expert_ids[6], float expert_w[6]
"""
import argparse
import struct
import sys

RECORD_FMT = "<2i6i6f"
RECORD_SIZE = struct.calcsize(RECORD_FMT)
assert RECORD_SIZE == 56, f"unexpected RouteLogRecord size {RECORD_SIZE} (expected 56)"


def read_route_log(path):
    """Parses one route-log file.

    Returns (by_layer, n_records) where by_layer is
    {layer_idx: {token_idx: (expert_ids_tuple6, weights_tuple6)}}.
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % RECORD_SIZE != 0:
        raise ValueError(
            f"{path}: file size {len(data)} bytes is not a multiple of the "
            f"{RECORD_SIZE}-byte record -- truncated or wrong-format file")
    n = len(data) // RECORD_SIZE
    by_layer = {}
    for i in range(n):
        chunk = data[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
        fields = struct.unpack(RECORD_FMT, chunk)
        token_idx, layer_idx = fields[0], fields[1]
        expert_ids = fields[2:8]
        weights = fields[8:14]
        by_layer.setdefault(layer_idx, {})[token_idx] = (expert_ids, weights)
    return by_layer, n


def layer_ahead_recall(by_layer):
    """One-layer-ahead recall@6 within a single file's by_layer dict.

    Returns (per_layer, overall_mean, overall_n) where per_layer is
    {L: (mean_recall_at_L, n_token_pairs)} -- keyed by the PREDICTING layer L
    (predicting L+1).
    """
    per_layer = {}
    total_sum, total_n = 0.0, 0
    for L in sorted(by_layer.keys()):
        if (L + 1) not in by_layer:
            continue  # last layer, or the (architecturally absent here) case
                      # of a dense layer immediately following a sparse one
        cur, nxt = by_layer[L], by_layer[L + 1]
        s, n = 0.0, 0
        for t in sorted(cur.keys()):
            if t not in nxt:
                continue
            predicted = set(cur[t][0])
            actual = set(nxt[t][0])
            s += len(predicted & actual) / 6.0
            n += 1
        if n == 0:
            continue
        per_layer[L] = (s / n, n)
        total_sum += s
        total_n += n
    overall_mean = total_sum / total_n if total_n else float("nan")
    return per_layer, overall_mean, total_n


def temporal_locality(by_layer):
    """Same-layer consecutive-token expert-set overlap within a single file.

    Returns (per_layer, overall_mean, overall_n), same shape as
    layer_ahead_recall, keyed by layer L.
    """
    per_layer = {}
    total_sum, total_n = 0.0, 0
    for L in sorted(by_layer.keys()):
        toks = by_layer[L]
        s, n = 0.0, 0
        for t in sorted(toks.keys()):
            if (t + 1) not in toks:
                continue
            a = set(toks[t][0])
            b = set(toks[t + 1][0])
            s += len(a & b) / 6.0
            n += 1
        if n == 0:
            continue
        per_layer[L] = (s / n, n)
        total_sum += s
        total_n += n
    overall_mean = total_sum / total_n if total_n else float("nan")
    return per_layer, overall_mean, total_n


def combine_per_layer(per_file_results):
    """Combines several (per_layer, overall_mean, overall_n) results -- one
    per input file -- into a single weighted-by-n aggregate. Never merges
    the raw per-file dicts (that would let file A's token 5 collide with
    file B's token 5); only the already-computed per-file scalar
    means+counts are combined here, weighted by sample count."""
    sum_map, n_map = {}, {}
    total_sum, total_n = 0.0, 0
    for per_layer, _, _ in per_file_results:
        for L, (mean, n) in per_layer.items():
            sum_map[L] = sum_map.get(L, 0.0) + mean * n
            n_map[L] = n_map.get(L, 0) + n
            total_sum += mean * n
            total_n += n
    combined_per_layer = {L: (sum_map[L] / n_map[L], n_map[L]) for L in sorted(sum_map)}
    combined_overall = total_sum / total_n if total_n else float("nan")
    return combined_per_layer, combined_overall, total_n


def format_table(per_layer, label):
    lines = [f"| Layer L | N | {label} |", "|---:|---:|---:|"]
    for L in sorted(per_layer.keys()):
        mean, n = per_layer[L]
        lines.append(f"| {L} | {n} | {mean:.4f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route_logs", nargs="+", help="one or more --route-log binary files")
    ap.add_argument("--markdown", action="store_true",
                     help="also emit full per-layer curves as markdown tables (for docs/pilot-routing.md)")
    args = ap.parse_args(argv)

    per_file = []  # (path, by_layer, n_records)
    for path in args.route_logs:
        by_layer, n_records = read_route_log(path)
        per_file.append((path, by_layer, n_records))

    print("=== Per-file summary ===")
    total_records = 0
    total_tokens = 0
    for path, by_layer, n_records in per_file:
        all_tokens = set()
        for toks in by_layer.values():
            all_tokens.update(toks.keys())
        n_tokens = len(all_tokens)
        n_layers = len(by_layer)
        print(f"{path}: {n_records} records, {n_tokens} distinct token positions, "
              f"{n_layers} distinct layers (min={min(by_layer)}, max={max(by_layer)})")
        total_records += n_records
        total_tokens += n_tokens
    print(f"TOTAL: {total_records} records across {len(per_file)} file(s), "
          f"{total_tokens} token positions (sum over files)")

    la_results = [layer_ahead_recall(by_layer) for _, by_layer, _ in per_file]
    tl_results = [temporal_locality(by_layer) for _, by_layer, _ in per_file]

    la_per_layer, la_overall_mean, la_overall_n = combine_per_layer(la_results)
    tl_per_layer, tl_overall_mean, tl_overall_n = combine_per_layer(tl_results)

    print()
    print("=== One-layer-ahead recall@6 (PILOT/colibri signal) ===")
    print(f"overall mean recall@6 = {la_overall_mean:.4f}  (n={la_overall_n} (token, layer-pair) instances)")
    if la_per_layer:
        best_L = max(la_per_layer, key=lambda L: la_per_layer[L][0])
        worst_L = min(la_per_layer, key=lambda L: la_per_layer[L][0])
        print(f"per-layer min = {la_per_layer[worst_L][0]:.4f} (L={worst_L}), "
              f"max = {la_per_layer[best_L][0]:.4f} (L={best_L})")

    print()
    print("=== Same-layer consecutive-token overlap (temporal-locality / LRU signal) ===")
    print(f"overall mean overlap@6 = {tl_overall_mean:.4f}  (n={tl_overall_n} (token, layer) instances)")
    if tl_per_layer:
        best_L = max(tl_per_layer, key=lambda L: tl_per_layer[L][0])
        worst_L = min(tl_per_layer, key=lambda L: tl_per_layer[L][0])
        print(f"per-layer min = {tl_per_layer[worst_L][0]:.4f} (L={worst_L}), "
              f"max = {tl_per_layer[best_L][0]:.4f} (L={best_L})")

    if args.markdown:
        print()
        print("=== Markdown: one-layer-ahead recall@6 per layer ===")
        print(format_table(la_per_layer, "recall@6"))
        print()
        print("=== Markdown: same-layer consecutive-token overlap per layer ===")
        print(format_table(tl_per_layer, "overlap@6"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
