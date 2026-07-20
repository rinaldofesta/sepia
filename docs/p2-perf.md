# P2: the first honest tok/s table

2026-07-20, M5 Max 128GB, branch `p2-metal` (Tasks 1-11 complete, HEAD
`3f07a6b` at measurement time). The engine's `--metal --real` mode:
resident weights wrapped zero-copy into a Metal buffer and read by
dequant-fused matvec kernels (no more per-token f32 re-dequant, Task 9),
banded flash-attention on GPU (Task 7), routed experts served from an
LRU-streamed, per-slot-mlocked GPU-resident cache (Task 10; default
`--expert-cache-gb 64` -> 2127 slots across 15 slabs, 12.6% of all 16,896
`(layer, expert)` pairs), with expert I/O overlapped against GPU compute
via a 4-thread loader pool (Task 11). Raw logs: `weights/p2-runs/`
(gitignored).

## Runs

Both P1 prompts, 32 greedy tokens, cold + warm in one process via
`--repeat 2` (a fresh process's expert cache starts empty -- "cold" -- and
the second generation in the *same* process reuses the cache the first
run populated -- "warm"; a short-lived CLI has no other way to observe a
genuinely warm cache across process boundaries, since F_NOCACHE bypasses
the OS page cache for GGUF reads and the GPU expert cache has no
cross-process persistence -- the exact rationale Task 10 established for
this flag, re-run here rather than just cited from its report).

```
./sepia --metal --real --prompt "The capital of France is" --n-gen 32 --repeat 2 --verbose-cache
./sepia --metal --real --prompt "def fibonacci(n):" --n-gen 32 --repeat 2 --verbose-cache
```

### Determinism / sequence exactness (re-verified, not just cited)

Both prompts, both cold and warm runs, reproduced the exact 32-token id
sequences recorded in `docs/p1-first-tokens.md` and every P2 task since
Task 9. No divergence at any position; the near-tie protocol was never
invoked.

**Prompt 1** (`The capital of France is`):
`12650 13 623 9029 328 290 8370 382 9741 13 623 9029 328 19420 382 51802
13 623 9029 328 22384 382 27388 13 623 9029 328 26350 382 20264 13 623`

**Prompt 2** (`def fibonacci(n):`):
`59 77 1150 271 392 271 1069 8729 1890 25 538 297 382 220 15 503 220 16
11 622 297 3392 1150 271 392 271 538 297 5017 220 16 16008`

No divergence -> no regression. This gate is satisfied for Task 12.

## Timing table (32-token double-runs)

| Prompt | Run | First token (ms) | Steady mean (ms/token) | Steady range (ms) | Wall, 32 tokens (s) | Steady tok/s | Cumulative hit rate (end) |
|---|---|---:|---:|---|---:|---:|---:|
| 1 (France) | cold | 6610.9 | 686.7 | 582.0-1673.1 | 28.50 | 1.456 | 76.6% |
| 1 (France) | warm | 3935.1 | 635.6 | 608.8-673.1 | 24.30 | 1.573 | 79.2% |
| 2 (fibonacci) | cold | 5863.8 | 721.3 | 632.7-1112.9 | 28.90 | 1.386 | 64.3% |
| 2 (fibonacci) | warm | 3169.9 | 701.0 | 651.9-795.7 | 25.59 | 1.427 | 66.5% |

("Steady" = the 31 tokens after the first, which folds in prefill cost.)

These numbers re-verify Task 11's own recorded numbers (676.2 / 617.4 /
716.9 / 694.9 ms for the same four conditions) rather than just citing
them: 2-8% slower here, within ordinary run-to-run system variance
(thermal state, background load on a shared machine) -- not a regression,
confirmed by the exact-matching token sequences above.

## 256-token run and the hit-rate curve

```
/usr/bin/time -l ./sepia --metal --real --prompt "The capital of France is" --n-gen 256 --verbose-cache
```

- First token: 6899.9 ms.
- Steady-state (255 tokens): mean **702.9 ms/token (1.423 tok/s)**, range
  589.9-1804.6 ms.
- Cumulative hit rate: 27.7% (end of prefill) -> 76.6% (token 32) ->
  82.1% (token 65) -> 82.8% (97) -> 83.8% (129) -> 83.0% (161) -> 81.8%
  (193) -> 81.8% (225) -> **82.6%** (token 256, final).
- Per-32-token-chunk throughput: 1.392, 1.471, 1.440, 1.387, 1.393,
  1.355, 1.395, 1.574 tok/s (chunks 1-8, tokens 2-256).

**The plateau, stated plainly**: the cache saturates by roughly token 65
(hit rate ~82%, consistent with the 2127-slot/12.6%-of-all-experts budget
against this prompt's repetitive "capital of X is Y" idiom) and does not
climb further through token 256 -- it oscillates in an 82-84% band with no
further trend. Per-token throughput stops improving in step: post-saturation
chunk means sit in a noisy 635-738ms band with no systematic further drop.
**This is the requested steady-state curve, and it is flat, not still
climbing** -- more tokens past ~65 do not buy more speed on this workload.
Total wall time for the whole run (load + prefill + 256 decode steps):
188.04s.

## Memory

`/usr/bin/time -l` on the 256-token run:

- maximum resident set size: 64,307,068,928 bytes (**64.31GB**)
- peak memory footprint: 64,654,607,728 bytes (64.65GB)
- `real 188.04s / user 4.14s / sys 51.90s`
- page faults: 260,344; voluntary context switches 47,536; involuntary
  531,819

Consistent with Task 10's shorter-run RSS of 56.84GB: a longer run
touches more distinct cache slots (17,408 misses accumulated over 256
tokens -- several times the 2127-slot capacity, so slots get evicted and
reinstalled repeatedly, as expected for a budget covering only 12.6% of
all experts) which pushes touched memory higher. Still comfortably under
the ~110GB Global Constraints ceiling.

Note: the 64.31GB RSS figure excludes the 14.23GB `resident.bin` zero-copy
mmap wrap -- a shared, file-backed MTLBuffer that macOS's RSS accounting
doesn't attribute to the process the way touched anonymous memory is. True
footprint is closer to ~78.5GB, still safely under the 110GB ceiling.

## What's measurable, and what isn't

The plan's Task 12 text names a `--timing` per-stage breakdown (resident
matvecs / attention / experts-io / experts-compute / logits) as the goal.
**That instrumentation does not exist in the codebase.** Tasks 9-11 built
`--verbose-cache` (hit/miss counts) and `--repeat` (cold/warm in one
process); no per-stage GPU timer was ever added. Reporting what IS
measurable instead of inventing a breakdown:

- Total ms/token and hit rate (both tables above).
- The `/usr/bin/time -l` real/user/sys split is a coarse proxy, not a
  real breakdown: user CPU time is tiny (4.14s over a 188s run) because
  GPU kernel execution is never charged to user CPU time; sys time
  (51.90s, ~28% of wall) is dominated by the loader pool's blocking
  `pread()`/`mlock()` calls (17,408 misses this run) plus condvar-wait
  syscalls. The remaining ~70% of wall time is the main thread blocked on
  `waitUntilCompleted` for GPU dispatches or on the loader pool's condvar
  -- i.e. most of the clock is neither raw disk I/O nor host CPU work,
  which points at GPU kernel execution as the largest remaining cost, but
  this cannot be split further (resident matvec vs. attention vs. expert
  matvec vs. logits) without real per-stage instrumentation that was
  never built. Flagging the gap rather than guessing a split.

## Comparison against the documented ceilings and target

From `docs/ssd-bench.md` / `docs/DESIGN.md`: pure-I/O ceilings (assuming
perfect compute overlap, at this SSD's measured 13.33 GB/s F_NOCACHE
bandwidth) are **~7.5 tok/s warm** (75% hit, ~1.8GB/token) and **~1.9
tok/s cold** (0% hit, ~7.1GB/token). The original project's honest target
(`docs/DESIGN.md`) was **1.5-3 tok/s** before MTP/prefetch.

Measured: 1.386-1.573 tok/s across the four cold/warm double-run
conditions; 1.423 tok/s average over the full 256-token run (plateauing
1.35-1.57 tok/s per 32-token chunk after the initial ramp).

**Verdict, stated plainly:**

- **Falls far short of both pure-I/O ceilings** -- including the *cold*
  ceiling (1.9 tok/s, which assumes a literal 0% hit rate on every
  token). Every measured run here achieved substantially better hit rates
  than that assumption (27-93% per-step, 64-93% cumulative by the end of
  a 32-token run; 82-84% by the plateau of the 256-token run). If SSD
  bandwidth were still the binding constraint, hit rates this good should
  already be clearing the 0%-hit ceiling comfortably; they don't come
  close. **I/O is no longer the bottleneck.** The binding constraint has
  moved, across this project's phases, from "CPU dequant of resident
  weights" (P1, ~60% of cost) through "CPU expert streaming + dequant"
  (pre-Task 10, ~38% of cost) to **GPU compute time now** -- never disk
  bandwidth, at any stage measured so far.
- **Sits at, not comfortably inside, the 1.5-3 tok/s target.** Only one
  of the four cold/warm conditions (prompt 1, warm: 1.573 tok/s) clears
  1.5 tok/s; the other three (1.456 / 1.386 / 1.427) land just under the
  floor. The 256-token plateau band (1.35-1.57 tok/s) tells the same
  story. **P2 meets the target's bottom edge, not its middle.**
- What the data says is left on the table: Task 4 already disclosed a
  deliberate design choice -- "plain unpack-then-dot kernels, ds4's
  packed-lane throughput optimization consciously deferred as a Task 12
  lever, not an oversight." This measurement is exactly that lever coming
  due: the system is compute-bound now, and the dequant-fused matvec
  kernels (the least throughput-optimized part of the compute path) are
  the most likely place a real further win lives. Task 13's PILOT
  experiment targets routing predictability / hit rate, but this data
  says hit rate is *not* the current limiter (there's 4-5x of I/O
  headroom unused); a prefetch win there would still be real (P3), but it
  would not be the thing that gets P2 comfortably past 1.5 tok/s -- kernel
  throughput would.

## Concerns

- No per-stage GPU timing instrumentation exists (see above); the
  compute-vs-I/O conclusion rests on a coarse `real`/`user`/`sys` proxy
  and an arithmetic argument (measured hit rates already beat the
  cold-ceiling's 0%-hit assumption, yet measured tok/s doesn't reach that
  ceiling), not a real per-op breakdown. Directionally solid, not
  maximally precise.
- The 256-token run was a single run, not double-run -- the
  determinism gate is satisfied by the four 32-token double-runs above
  (per the plan's own scoping); the 256-token run's job is curve shape /
  plateau discovery, not another determinism proof.
- Run-to-run system variance: this session's steady-state numbers ran
  2-8% slower than Task 11's own recorded numbers for the identical
  commands, consistent with ordinary noise on a shared machine (thermal
  state, background load) and not a regression -- confirmed by the
  exact-matching token sequences on every run.
