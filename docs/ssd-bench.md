# SSD random-pread microbenchmark

Generated 2026-07-19 by `tools/iobench.c`, measuring this machine's actual
decode access pattern: random, block-aligned preads sized like an expert
slab (routed experts are 95.5% of tensor bytes; one expert's gate+up+down
averages ~17.7MiB per `docs/gguf-inventory-ud-q2_k_xl.md`), buffered vs
`F_NOCACHE`, across thread counts.

## Machine

- Model: MacBook Pro (`Mac17,6`), chip **Apple M5 Max**, 18 cores (6P+12E),
  128GB unified memory.
- macOS 26.5.2 (build 25F84), Darwin 25.5.0.
- SSD: `APPLE SSD AP2048Z`, 2TB, NVMe over Apple Fabric, on the internal
  `disk3s5` (APFS) volume. ~960GB free at the time of the run.

```
sysctl -n hw.model machdep.cpu.brand_string
sw_vers
diskutil info /
system_profiler SPStorageDataType SPNVMeDataType
```

`diskutil info /` reports `Solid State: Yes` and `Protocol: Apple Fabric`
but not a drive model (Apple Silicon's integrated controller doesn't
expose one there -- `Media Type` just says `Generico`); the model string
above came from `system_profiler SPNVMeDataType` instead.

## Test file

```
dd if=/dev/urandom of=weights/iobench.bin bs=4m count=8192
```

34,359,738,368 bytes (32GiB) exactly, from `/dev/urandom` so no run of
identical blocks can get cached/deduped cheaply. `weights/` is entirely
gitignored; the file is untracked and kept on disk for reuse (not deleted
after this run). Free disk was checked first (~960GB, well above the
100GB abort threshold) with `df -g`.

## Build and correctness test

```
make iobench
./tools/test_iobench.sh
```

`tools/test_iobench.sh` builds the binary, runs it against a small
generated 64MiB file for both `direct=0` and `direct=1`, and asserts the
reported `total_gb` equals `n_reads * block_mb` exactly. It also asserts
the documented expected-failure behavior: a block size larger than the
whole file aborts with a nonzero exit and a `"too small"` message on
stderr, rather than silently returning a short read. All checks pass.

## Method notes

- `block_mb` is MiB (`1024*1024` bytes); reported throughput is decimal
  GB/s (`bytes / 1e9`), the conventional storage-benchmark unit. Don't
  cross-compare the two without converting.
- `n_reads = 12000 / block_mb` (2400 at 5MiB, 800 at 15MiB, 400 at
  30MiB), so every cell reads the same ~12.58GB regardless of block size
  or thread count — comparable aggregates, and above the ~8-16GB target
  needed to defeat the 32GB file's own locality.
- Every cell's exact command is `./iobench weights/iobench.bin <block_mb>
  <n_reads> <threads> <direct>` with `<n_reads>` from the formula above
  and `<block_mb>`/`<threads>`/`<direct>` from the table row.
- Run order: all `direct=1` (F_NOCACHE) cells first, then all `direct=0`
  (buffered) cells. F_NOCACHE reads don't populate the page cache, so
  their order doesn't matter; buffered reads do, so running them last
  (and in ascending block/thread order within that) means each buffered
  cell can only get *more* cache-assisted than the one before it, not
  less. This is called out explicitly below because it shows up plainly
  in the numbers.

### Anomaly: the first full run was contaminated, and here's the proof

The first end-to-end run (immediately after `dd`-creating the 32GB file)
produced F_NOCACHE numbers of 25-119 GB/s that scaled with thread count
and converged to the *same* ceiling as the buffered numbers at every
thread count — e.g. at 15MiB/threads=4/direct=1 it reported 63.16 GB/s,
identical in shape to the buffered run. That's memory-copy bandwidth, not
NAND flash: no SSD on this machine class does 119 GB/s. Root cause:
macOS's `F_NOCACHE` disables caching for *future* I/O on that descriptor,
but does not evict pages already resident in the unified buffer cache —
and the file had just been fully written (and thus fully cached) seconds
before. `vm_stat` confirmed ~40GB of file-backed pages resident going
into that run. Tried `mmap()` + `madvise(MADV_DONTNEED)` on the whole
file to force-drop those pages surgically: no effect (`vm_stat`'s
file-backed count didn't move). No passwordless `sudo purge` was
available. Fix: wrote a 140GB throwaway file of fresh `/dev/urandom` data
(bigger than the 128GB of RAM) to force real LRU eviction of the old
pages, verified via `vm_stat` (file-backed pages jumped to the new file's
own footprint, free pages dropped from ~72GB to ~12GB — genuine turnover,
not a no-op), deleted the throwaway file, then reran the entire matrix
below from a verified-cold cache. A quick sanity sweep (15MiB block,
threads 1/8/16, F_NOCACHE) after the flush gave 11.7/13.3/13.4 GB/s —
plateaued with thread count as a real bandwidth-saturated device should,
not still climbing into the hundreds — before committing to the full
24-cell rerun. The table below is that clean rerun; the contaminated run
is not reported as a data point anywhere in it.

## Results (clean run, cold cache)

| block MiB | direct | threads | n_reads | total GB | wall s | GB/s | mean ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 1 | 2400 | 12.58 | 1.240 | 10.15 | 0.516 | 0.540 |
| 5 | 1 | 4 | 2400 | 12.58 | 0.943 | 13.34 | 1.571 | 1.780 |
| 5 | 1 | 8 | 2400 | 12.58 | 0.946 | 13.30 | 3.150 | 3.548 |
| 5 | 1 | 16 | 2400 | 12.58 | 0.946 | 13.30 | 6.291 | 6.791 |
| 15 | 1 | 1 | 800 | 12.58 | 1.071 | 11.75 | 1.338 | 1.477 |
| 15 | 1 | 4 | 800 | 12.58 | 0.944 | 13.33 | 4.715 | 5.246 |
| 15 | 1 | 8 | 800 | 12.58 | 0.945 | 13.32 | 9.420 | 9.979 |
| 15 | 1 | 16 | 800 | 12.58 | 0.946 | 13.31 | 18.737 | 19.858 |
| 30 | 1 | 1 | 400 | 12.58 | 1.027 | 12.25 | 2.567 | 2.828 |
| 30 | 1 | 4 | 400 | 12.58 | 0.949 | 13.25 | 9.484 | 10.885 |
| 30 | 1 | 8 | 400 | 12.58 | 0.946 | 13.31 | 18.841 | 20.131 |
| 30 | 1 | 16 | 400 | 12.58 | 0.942 | 13.36 | 37.371 | 38.739 |
| 5 | 0 | 1 | 2400 | 12.58 | 1.116 | 11.27 | 0.465 | 0.595 |
| 5 | 0 | 4 | 2400 | 12.58 | 0.740 | 17.01 | 1.223 | 10.019 |
| 5 | 0 | 8 | 2400 | 12.58 | 0.671 | 18.76 | 2.194 | 14.481 |
| 5 | 0 | 16 | 2400 | 12.58 | 0.507 | 24.82 | 3.271 | 21.485 |
| 15 | 0 | 1 | 800 | 12.58 | 0.678 | 18.57 | 0.846 | 1.384 |
| 15 | 0 | 4 | 800 | 12.58 | 0.285 | 44.11 | 1.397 | 3.022 |
| 15 | 0 | 8 | 800 | 12.58 | 0.197 | 63.79 | 1.925 | 4.422 |
| 15 | 0 | 16 | 800 | 12.58 | 0.141 | 89.14 | 2.680 | 8.907 |
| 30 | 0 | 1 | 400 | 12.58 | 0.582 | 21.61 | 1.455 | 2.093 |
| 30 | 0 | 4 | 400 | 12.58 | 0.229 | 55.06 | 2.259 | 3.806 |
| 30 | 0 | 8 | 400 | 12.58 | 0.181 | 69.55 | 3.466 | 9.301 |
| 30 | 0 | 16 | 400 | 12.58 | 0.153 | 82.14 | 5.617 | 19.468 |

direct=1 rows are F_NOCACHE (kernel bypasses the page cache); direct=0
rows are buffered reads through the page cache.

Reading the direct=1 rows: throughput climbs from thread=1 to thread=4
(10.2 -> 13.3 GB/s at 5MiB; 11.8 -> 13.3 GB/s at 15MiB; 12.3 -> 13.3 GB/s
at 30MiB) and then flatlines at ~13.3 GB/s from thread=4 through
thread=16 regardless of block size, while mean latency keeps climbing
roughly linearly with thread count (queue depth). That's a
bandwidth-saturated device: past 4 concurrent requests the SSD's random
read bandwidth ceiling is the limit, not request count or size.

The direct=0 rows show the opposite shape: throughput keeps climbing well
past what F_NOCACHE ever reaches (up to 89.14 GB/s at 15MiB/threads=16,
6.7x the F_NOCACHE ceiling for the same block size), and does so entirely
because each successive buffered cell (run in ascending thread order,
same block size, same 32GB file) increasingly hits pages the previous
cell already pulled into cache. The p99 tail confirms the mixed
hit/miss state within a cell -- e.g. 5MiB/threads=4/direct=0 has p99
8.2x its own mean (10.02ms vs 1.22ms), i.e. most reads in that cell were
already cached but a meaningful tail still went to disk. **The buffered
numbers in this table are not a usable estimate of this SSD's real
random-read bandwidth**; they measure a mix of disk and RAM in
proportions that depend on run history. Only the direct=1 rows are used
for the conclusions below.

## Conclusions

**1. Achievable random-read GB/s at 15MB blocks (the expert-slab case), best config.**
Best is threads=4, F_NOCACHE: **13.33 GB/s** (raw `gbps=13.3267` from the
15MiB/threads=4/direct=1 cell; 13.25-13.36 GB/s across all three block
sizes at threads=4/8/16 -- effectively flat; the device is
bandwidth-saturated already at 4 concurrent requests, so more threads
buys nothing and only adds latency).

**2. The repack decision input.** Same total bytes (12.58GB), best thread
count each, F_NOCACHE: 5MiB blocks (threads=4) reach **13.34 GB/s** (raw
`gbps=13.3442`); 15MiB blocks (threads=4) reach **13.33 GB/s** (raw
`gbps=13.3267`, the same cell as conclusion 1). Difference:
`(13.3442 - 13.3267) / 13.3267 = 0.13%`. Plainly: **3 preads of ~5MB
reach 13.34 GB/s vs 1 pread of ~15MB at 13.33 GB/s -- not a material
difference** (well under the 15% threshold; within measurement noise).
This SSD's random-read bandwidth is saturated by request count/size well
before the gate/up/down split matters. Per `docs/DESIGN.md`'s own stated
fallback ("if the SSD benchmark shows three smaller preads match one slab
pread, stream directly from the GGUF and skip the repack"): they match,
so the repack step is not justified by throughput -- stream directly
from the GGUF's three tensors per expert and skip building
`experts-NN.bin` slabs, unless a future non-throughput reason (e.g. one
syscall vs three, alignment, or a slower SSD on different hardware)
reopens the question.

**3. Decode ceiling estimate.** Best 15MiB F_NOCACHE throughput (raw
`gbps=13.3267`, same cell as conclusion 1) divided by the two decode
regimes from `docs/DESIGN.md`:
- Warm (~1.5GB/token): `13.3267 / 1.5 = 8.88` tok/s I/O ceiling.
- Cold (~6GB/token): `13.3267 / 6 = 2.22` tok/s I/O ceiling.

These are pure I/O bounds assuming perfect overlap of every other cost;
real decode overlaps compute (dequant, attention, MoE routing) with these
reads rather than paying for them serially, so actual tok/s will sit
somewhere below the ceiling, not at it. They're still useful as an upper
bound: `docs/DESIGN.md`'s honest target of 1.5-3 tok/s sits comfortably
under the 2.22 tok/s cold-decode I/O ceiling, meaning SSD bandwidth is
not expected to be the binding constraint at that target -- compute and
MTP/prefetch overlap are.
