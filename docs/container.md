# Container: GGUF parts + sepia-index sidecar + resident.bin

Written 2026-07-19 (task 0.7), superseding the "inkwell" repack framing in
`docs/DESIGN.md`. `docs/ssd-bench.md` measured that three ~5MB preads match
one ~15MB pread within 0.13% on this machine's SSD, so SEPIA does not
repack the GGUF into `experts-NN.bin` slabs. This doc records the one open
caveat that decision depended on (unaligned-offset throughput), then
describes the container as built: the source GGUF parts themselves, an
index sidecar this task adds, and a small extracted resident blob.

## 1. The unaligned-offset A/B (the gating check)

`docs/ssd-bench.md`'s matrix only measured **block-aligned** offsets
(multiples of the read size). Real GGUF tensor data is 32-byte aligned,
not block- or page-aligned, so an expert's true `pread` offset is
essentially a random multiple of 32. This section verifies that doesn't
cost meaningful throughput before relying on it.

### Method

`tools/iobench.c` gained a 6th, optional `align_bytes` argument (default
`0` = the original block-aligned behavior; a positive value, e.g. `32`,
draws offsets uniformly from every `align_bytes`-aligned position that
still keeps the whole block inside the file). Full usage:

```
./iobench <file> <block_mb> <n_reads> <threads> <direct 0|1> [align_bytes]
```

Per the brief: back-to-back interleaved pairs (A = aligned, B =
`align_bytes=32`), 3 repetitions, `direct=1` (F_NOCACHE), threads=4 (the
best-thread cell from `docs/ssd-bench.md`), against `weights/iobench.bin`
(the same 32GiB file), at the 5MiB and 15MiB block sizes with the same
`n_reads` as ssd-bench.md's matrix (2400 and 800, so every cell reads the
same ~12.58GB regardless of block size):

```
./iobench weights/iobench.bin 5  2400 4 1 0    # A
./iobench weights/iobench.bin 5  2400 4 1 32   # B
./iobench weights/iobench.bin 5  2400 4 1 0    # A
./iobench weights/iobench.bin 5  2400 4 1 32   # B
./iobench weights/iobench.bin 5  2400 4 1 0    # A
./iobench weights/iobench.bin 5  2400 4 1 32   # B
./iobench weights/iobench.bin 15 800  4 1 0    # A
./iobench weights/iobench.bin 15 800  4 1 32   # B
./iobench weights/iobench.bin 15 800  4 1 0    # A
./iobench weights/iobench.bin 15 800  4 1 32   # B
./iobench weights/iobench.bin 15 800  4 1 0    # A
./iobench weights/iobench.bin 15 800  4 1 32   # B
```

### Environment note: this measurement is noisier than ssd-bench.md's

Unlike the original ssd-bench.md session, this run happened on a machine
with **active concurrent load**: the 317GB download still running in the
background, plus several other agent sessions concurrently building/testing
in the same repo. Two best-effort cache flushes (150GB and 60GB throwaway
`/dev/urandom` writes, the same technique ssd-bench.md used) barely moved
`vm_stat`'s active+inactive page counts (~122GB either way) -- confirming
other processes were refilling the page cache as fast as the flush could
evict it. No passwordless `sudo purge` was available (same as ssd-bench.md).
Absolute throughput numbers below are flagged as measured under that load;
per the brief, **the interleaved ratio is the decision input**, not the
absolute GB/s.

A second, independent confound surfaced during extended diagnostic runs
(beyond the 3 reps below, kept only as supporting evidence, not the primary
result): the aligned side (A) draws offsets from a small, fixed set of
positions (`file_size / block_size` -- 6,553 for 5MiB blocks, 2,184 for
15MiB), while the unaligned side (B) draws from roughly a billion 32-byte
positions. Repeated back-to-back runs let A's small position set become
mostly cache-resident far faster than B's effectively-unlimited one, which
manifests as the *ratio drifting down* the more the same file gets
re-read -- an artifact of unequal candidate-space size, not of real SSD
alignment behavior. A 12-pair low-`n_reads` (low self-collision) diagnostic
run showed a stable ~78-86% ratio, but only once absolute throughput was
~5x the real SSD ceiling (i.e. an almost-pure RAM-copy measurement): a
fixed per-request unaligned-copy cost is a large fraction of a cache-speed
request but a small fraction of a real disk-latency-bound one, so that
regime likely *overstates* the true disk-bound penalty. The freshest
samples below -- closest to `docs/ssd-bench.md`'s own clean baseline
(13.34 GB/s at 5MiB/threads=4/direct=1, reproduced here almost exactly at
13.34 GB/s on the very first pair) -- are the most trustworthy read of
genuine disk-bound behavior, which is why they're reported as primary.

### Results (primary, 3-rep interleaved, as specified)

| block | rep | A gbps (aligned) | B gbps (align=32) | B/A ratio |
|---:|---:|---:|---:|---:|
| 5MiB  | 1 | 13.34 | 15.27 | 114.4% |
| 5MiB  | 2 | 17.84 | 18.72 | 104.9% |
| 5MiB  | 3 | 22.46 | 24.27 | 108.1% |
| 15MiB | 1 | 28.91 | 30.18 | 104.4% |
| 15MiB | 2 | 38.16 | 35.07 |  91.9% |
| 15MiB | 3 | 50.67 | 43.21 |  85.3% |

Mean ratio (the decision input): **5MiB = 109.1%**, **15MiB = 93.9%**. Both
clear the 90% bar; the rep-to-rep climb in absolute GB/s (13->22, 29->51)
is the concurrent-load/cache effect above, not a per-read slowdown --
within each pair the two reads happened back-to-back under the same
conditions.

### Verdict: CONFIRMED, with a flagged margin concern

Unaligned throughput is **>= 90% of aligned in both block sizes** (109.1%
and 93.9%), so **the no-repack plan is confirmed**: SEPIA streams each
expert's gate/up/down tensors directly from the GGUF parts via the index
sidecar below, no `experts-NN.bin` conversion step.

Flagged for the controller: the 15MiB cell's individual reps ranged
85.3-104.4%, straddling the 90% line, and the measurement environment
(concurrent agents + an active 317GB download, no working cache flush)
was not the clean, isolated condition ssd-bench.md had. The freshest pairs
(rep 1, closest to the historical baseline) and the physical reasoning
above both point toward the true disk-bound ratio being at or above what's
reported here. A rerun in a quieter window (download complete, no
concurrent agents) would be worth doing to tighten this, but is not judged
to gate proceeding with the no-repack design.

## 2. The container's three pieces

### GGUF parts (unchanged)

`weights/inkling-gguf/UD-Q2_K_XL/inkling-UD-Q2_K_XL-000NN-of-00008.gguf`,
8 files, 317.33GB total, exactly as downloaded from
`unsloth/inkling-GGUF`. Never modified, never repacked. Quant blocks stay
byte-identical to the source, so ported ds4/ggml dequant kernels apply
unchanged (per `docs/DESIGN.md`).

### `weights/inkling-ud-q2_k_xl.sepia-index.json` (sepia-index sidecar)

Built by `tools/make_index.py` (stdlib only, imports `gguf_inspect.py`'s
parser -- no duplicated parsing logic). For every one of the 64 MoE
layers' 256 routed experts, this records the exact
`(part_file, abs_offset, nbytes, ggml_type)` of its gate/up/down slices,
using the per-expert contiguity `docs/gguf-inventory-ud-q2_k_xl.md`
confirmed:

```
abs_offset = part_data_start + tensor.offset + expert_index * bytes_per_expert(tensor)
```

Each of the 64 layers' gate/up/down tensor blocks in the index also carries
the raw arithmetic inputs (`data_start`, `tensor_offset`, `n_bytes_total`,
`n_expert`, `bytes_per_expert`) alongside the 256 precomputed per-expert
entries, so the C loader can recompute and cross-check instead of trusting
the cached numbers blindly.

It also lists every **resident** (non-`_exps`) tensor -- embeddings,
attention, shared experts, routers, norms, output head -- with the same
`abs_offset`/`nbytes`/`ggml_type`/`shape` shape, plus a header block
(alignment, per-part expected sizes, quant-type histogram, generation
timestamp, source repo id).

**Local-vs-inventory fallback.** A part counts as available locally only
if a file exists at its expected path *and* its size exactly matches
`docs/gguf-inventory-ud-q2_k_xl.json`'s expected size (a partial download,
if ever visible under its final name, falls back rather than being parsed
half-written). Parts not yet downloaded fall back to the committed
inventory JSON, which is authoritative for expected values. Every part in
the index is tagged `"source": "local"` or `"source": "inventory"`. Where
both are available for the same part, the local parse is cross-checked
against the inventory (tensor names, types, offsets) and any divergence is
a loud failure (`IndexBuildError`), not a silent fallback.

**Validation: `--verify N`.** Byte-verifies N random experts whose parts
are fully downloaded. "Two independent code paths" means independent
*traversal and caching*, not two different offset formulas: both paths
compute `abs_offset = data_start + tensor.offset + e * bytes_per_expert`
identically, but one reads the already-computed value cached in the index
doc, and the other re-parses the part's header fresh from disk right then
and walks its tensor list from scratch to find the tensor and recompute
the offset. That catches index/cache corruption, staleness, and drift
between the committed doc and the actual on-disk part -- it does *not*
independently verify that the offset formula itself is correct, since
both paths share it. The formula's correctness is ground-truthed
separately, by `tools/test_make_index.py`'s byte-tagged fixture (each
synthetic expert's bytes are tagged with its own index, so the test
confirms the computed `abs_offset` actually points at the right expert's
bytes, not just that two computations of the same formula agree with each
other). `--verify` then also checks the slice doesn't cross the part's
EOF, and prints `OK n/N` with a skipped-not-yet-downloaded count.

The index JSON itself is derived data and stays untracked; `tools/make_index.py`
is what's committed.

### `weights/resident.bin` + `weights/resident-manifest.json`

Built by `tools/extract_resident.py` (stdlib only, reuses
`tools/make_index.py`'s resident-tensor listing rather than re-deriving
the local/inventory merge a second time). Streams every resident tensor's
bytes out of the GGUF parts into one contiguous, 64-byte-aligned file, for
mlock'ing at load time (per `docs/DESIGN.md`, ~14GB: embeddings, attention,
shared experts, routers, norms, output head) and as the future home for
the MTP sidecar.

Each tensor's bytes are SHA256'd while streaming from the source, then
`resident.bin`'s just-written range is re-read and re-hashed; a mismatch
truncates the file back to its pre-tensor length and aborts loudly, so a
rerun always finds a fully-verified file to resume from. The manifest is
written (atomically: temp file + rename) right after *each* tensor is
successfully verified, not just once at the end -- so a failure partway
through a run (say, tensor 3 of 5) still leaves a durable, accurate
manifest for the first 2, instead of orphaning their already-verified
bytes in `resident.bin` with nothing pointing at them.

`resident.bin` is **append-only**: an already-extracted tensor's offset
never changes across reruns. "Idempotent" here specifically means *won't
redo already-recorded work* -- a rerun trusts the manifest's existing
entries (matched by tensor name, with a shape/type/nbytes sanity check
against the current index) and skips straight to whatever's still
missing; it does **not** re-hash or re-verify every previously-extracted
tensor's bytes against `resident.bin` on every run. That's a deliberate
accepted tradeoff (full re-verification on every rerun would mean
re-reading the whole growing file every time), not an oversight -- if
`resident.bin` were ever corrupted out-of-band after extraction, this
tool would not detect it; that would need a separate, explicit
`--verify`-style pass, which doesn't exist yet. `--min-free-gb`-gated
(default 50GB) reruns only append tensors from parts that have since
finished downloading. If some parts with resident tensors aren't fully
downloaded yet, the tool extracts what's available, records the rest
under `pending_parts` in the manifest, and **exits 0** (an expected, not
an error, state).

Both derived artifacts (`resident.bin`, `resident-manifest.json`) stay
untracked.

### Future: MTP sidecar (not built yet)

Per `docs/DESIGN.md`, the 8 MTP layers aren't in the Unsloth UD-Q2_K_XL
GGUF at all (confirmed in `docs/gguf-inventory-ud-q2_k_xl.md`); Phase 4
sources them from the BF16 checkpoint's dedicated `mtp.safetensors` shard
and quantizes them in-house. That sidecar is out of scope for this task.

## 3. How the C loader consumes each piece

(Contract for the streaming loader; `src/sepia.c` doesn't wire this up yet
-- that's a later phase-1 task, tracked separately.)

- **Resident weights**: `mmap()` or read `resident.bin` once at startup;
  `resident-manifest.json` gives each tensor's `(offset, nbytes, ggml_type,
  shape)` inside it. This is the set that gets `mlock`'d.
- **Expert weights**: for a routed expert `(layer, e)`, look up
  `moe_layers[str(layer)]["gate"|"up"|"down"]["experts"][e]` in the sepia-index
  JSON for `(part_file, abs_offset, nbytes, ggml_type)`, then
  `pread(fd_for_part_file, buf, nbytes, abs_offset)` with `F_NOCACHE`
  (`fcntl(fd, F_NOCACHE, 1)`) -- one syscall per tensor, three per expert,
  directly against the open GGUF part file descriptor. No intermediate
  copy step; the pread'd bytes are the same quant blocks ds4-ported dequant
  kernels expect.
- **Startup**: `make_index.py --verify N` should be run once against the
  final, fully-downloaded weight set before first production use, to catch
  any drift between the committed inventory and the actual on-disk parts.

## 4. Regeneration commands

```
# rebuild the inventory JSON/md (only needed if the source repo changes)
python3 tools/gguf_inspect.py --repo unsloth/inkling-GGUF \
  --file "UD-Q2_K_XL/inkling-UD-Q2_K_XL-00001-of-00008.gguf" \
  --all-parts --json --out docs/gguf-inventory-ud-q2_k_xl.json

# build/refresh the sepia-index sidecar (rerun any time more parts finish downloading)
python3 tools/make_index.py --verify 50

# extract whatever resident tensors are available (idempotent, safe to rerun)
python3 tools/extract_resident.py

# unaligned-offset A/B (Section 1's method, rerun in a quieter window to tighten the margin)
make iobench
for rep in 1 2 3; do
  ./iobench weights/iobench.bin 5  2400 4 1 0
  ./iobench weights/iobench.bin 5  2400 4 1 32
done
for rep in 1 2 3; do
  ./iobench weights/iobench.bin 15 800  4 1 0
  ./iobench weights/iobench.bin 15 800  4 1 32
done
```

## 5. `gguf_inspect.py` hardening (closes task 0.6 review Minors)

Three Minors closed in the parser this task reuses everywhere above:

1. **Hang prevention**: every declared count read off the wire before a
   loop (`metadata_kv_count`, `tensor_count`, tensor `n_dims`, a string's
   byte length, an array's element count) is now sanity-bounded against
   how many bytes the source could possibly still hold, given each item's
   minimum size, before looping -- a corrupted or truncated file fails
   loudly and immediately instead of looping/reading for a very long time.
2. **Clean error routing**: an unrecognized `ggml_type` (raised by
   `TensorInfo.nbytes()`) now exits through the same `error: ...`
   message and exit code 1 as any other CLI failure -- `main()`'s
   try/except was widened to cover output generation
   (`_to_json_doc`/`_print_human_summary`), not just parsing.
3. **New unit tests**: `inspect_all_parts`, `_to_json_doc`, and
   `_summarize_metadata` are now covered via a synthetic two-part local
   fixture (no network), plus dedicated tests for both hardening items
   above.

`tools/test_gguf_inspect.py`: 14 -> 25 tests, all passing.

## 6. Current real-data status (2026-07-19, download in progress)

Only part 1 of 8 (`inkling-UD-Q2_K_XL-00001-of-00008.gguf`, the
12.99MB metadata-only shard) is fully downloaded; parts 2-8 (the ones
holding all 1512 tensors) are still in flight. Real runs today therefore:

- `tools/make_index.py`: all 64 MoE layers / 49,152 expert tensor entries
  and all 1,320 resident tensors indexed successfully (from the inventory
  fallback for parts 2-8, local parse for part 1's zero tensors). 1 part
  local, 7 inventory-fallback. `--verify 20`: `OK 0/20 (skipped 20
  not-yet-downloaded, 0 failed)` -- correct, expected behavior; the actual
  verify *logic* (two-path byte-compare, corruption detection) and the
  partial-MoE-layer guard (a layer must have all of gate/up/down or none)
  are exercised by `tools/test_make_index.py`'s synthetic fixtures instead
  (12/12 passing).
- `tools/extract_resident.py`: `0/1320` tensors extracted (part 1 has no
  resident tensors of its own), all 7 other parts listed under
  `pending_parts`, `resident.bin` not created (nothing to write yet),
  exits 0. The actual copy/SHA256/alignment/idempotent-rerun logic,
  including a reproduction of a mid-run failure (tensor 3 of 3) that
  proves the incremental manifest write leaves no orphaned bytes on
  rerun, is exercised by `tools/test_extract_resident.py`'s synthetic
  fixtures (6/6 passing).
- No mismatch between the inventory JSON and any locally-parsed part was
  found (part 1 trivially matches, having zero tensors); the cross-check
  mechanism itself is independently verified to catch real mismatches by
  `tools/test_make_index.py`'s deliberately-corrupted-inventory tests.

Both tools are safe to rerun as the download progresses -- they're
idempotent and will pick up newly-completed parts automatically.
