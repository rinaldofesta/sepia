# P2 Task 13: the PILOT routing-predictability experiment

2026-07-21, M5 Max 128GB, branch `p2-metal` (Tasks 1-12 complete, HEAD
`c991817` at measurement time). Question: does Inkling's MoE router pick a
predictable set of experts one layer ahead of time, the way colibrì
measured on GLM-5.2 (71.6% one-layer-ahead recall)? This is the P2
close-out experiment; the result is published either way, per the
project's own stated commitment (`docs/DESIGN.md`'s open-questions
section, resolved below).

**This is a distinct signal from Task 12's cache hit rate.** Task 12
measured whether an expert *happened to still be resident* in the LRU
cache (`docs/p2-perf.md`, 65-95% hit rate depending on cache warmth) --
that is a fact about the cache's history, not about the router's
next-layer behavior. This experiment asks a different question entirely:
given layer L's chosen 6 experts, how well do they predict layer L+1's
chosen 6, for the *same token*? The two are reported together below only
because the second (same-layer, consecutive-token) analysis this task
also runs *does* speak to the cache-hit-rate story -- see "Temporal
locality" below.

## Method

### Instrumentation: `--route-log FILE`

`src/sepia.c`'s `real_mlp_moe_forward_gpu` (the `--metal --real` GPU MoE
path) appends one fixed-size binary record per (token, sparse-layer)
selection when `--route-log FILE` is given. Format (see the
`RouteLogRecord` comment in `src/sepia.c` for the authoritative version),
56 bytes, native byte order, no header/versioning -- an internal
diagnostic file, not a public format:

```c
typedef struct {
    int32_t token_idx;      /* absolute generation position (prompt then generated) */
    int32_t layer_idx;      /* 0..65; only sparse (MoE) layers ever write a record --
                              * layers 0-1 are dense (no router), so records only exist
                              * for layers 2..65 (64 sparse layers) */
    int32_t expert_ids[6];  /* selected routed-expert ids, in router-weight order */
    float   expert_w[6];    /* matching post-route mixing weights (route_scale *
                              * global_scale * renormalized softmax weight) */
} RouteLogRecord;
```

The write happens immediately after `moe_route_select` computes the
selection and before any expert-cache streaming/eviction work runs, so it
is a pure observer: it cannot perturb the routing decision or the
generated token sequence.

`--route-log` requires `--metal` (only the GPU MoE path is instrumented
for this task, since PILOT data collection happens exclusively via
`--metal --real`); CPU real mode is untouched.

**Inertness gate.** Two checks:
- Direct A/B: `The capital of France is` generated for 8 tokens with and
  without `--route-log` produced byte-identical `(token id, decoded text)`
  sequences.
- At the scale actually used for data collection (150-token runs, below):
  prompt 1's first 32 generated ids exactly match the sequence recorded in
  `docs/p1-first-tokens.md` (`12650 13 623 9029 328 290 8370 382 9741 13
  623 9029 328 19420 382 51802 13 623 9029 328 22384 382 27388 13 623 9029
  328 26350 382 20264 13 623`), and prompt 2's ids match the same
  document's recorded prefix through at least 30 tokens (`59 77 1150 271
  392 271 1069 8729 1890 25 538 297 382 220 15 503 220 16 11 622 297 3392
  1150 271 392 271 538 297 5017 220 ...`). No divergence from the
  established P1/P2 baseline at any position checked.

### Data generation

Four prompts, `--metal --real --route-log <file> --n-gen 150` each (the
two P1 prompts, reused verbatim, plus two new prompts chosen for topic
diversity -- one narrative/prose, one technical/structured, both longer
than the P1 pair so the combined dataset clears the 512-token floor with
margin):

| # | Prompt | Type | Prompt tokens | Generated | Total token positions logged |
|---|---|---|---:|---:|---:|
| 1 | `The capital of France is` | factual (P1) | 5 | 150 | 155 |
| 2 | `def fibonacci(n):` | code (P1) | 4 | 150 | 154 |
| 3 | `The lighthouse keeper had not seen another ship in three months, but tonight the horizon flickered with something unfamiliar. She climbed the spiral stairs two at a time, her lantern swinging wildly, and when she reached the top she saw` | narrative/prose | 45 | 150 | 195 |
| 4 | `The HTTP POST /v1/orders endpoint accepts a JSON request body with the following fields: customer_id (integer, required), items (array of objects, required), and discount_code (string, optional). On success it returns HTTP 201 with a response body containing:` | technical/structured | 54 | 150 | 204 |

**Total: 708 token positions logged, 64 sparse layers each = 45,312
routing records** -- well over the >=512-token floor. Commands run (raw
logs, gitignored per the `weights/p1-runs`/`weights/p2-runs` convention:
`weights/pilot-runs/`):

```
./sepia --metal --real --prompt "The capital of France is" --n-gen 150 \
  --route-log weights/pilot-runs/prompt1.route --verbose-cache
./sepia --metal --real --prompt "def fibonacci(n):" --n-gen 150 \
  --route-log weights/pilot-runs/prompt2.route --verbose-cache
./sepia --metal --real --prompt "The lighthouse keeper had not seen another ship in three months, but tonight the horizon flickered with something unfamiliar. She climbed the spiral stairs two at a time, her lantern swinging wildly, and when she reached the top she saw" --n-gen 150 \
  --route-log weights/pilot-runs/prompt3.route --verbose-cache
./sepia --metal --real --prompt "The HTTP POST /v1/orders endpoint accepts a JSON request body with the following fields: customer_id (integer, required), items (array of objects, required), and discount_code (string, optional). On success it returns HTTP 201 with a response body containing:" --n-gen 150 \
  --route-log weights/pilot-runs/prompt4.route --verbose-cache
```

### Analysis: `tools/pilot_routing.py`

Stdlib only (`struct` + `argparse`); reads the fixed-size binary records
and computes two independent metrics:

1. **One-layer-ahead recall@6** (the PILOT/colibrì signal): for each
   (token, layer L) with a record, treat layer L's 6 selected expert ids
   as a *prediction* of layer L+1's 6 selected ids for the *same token*
   (only when L+1 also has a record for that token -- the last layer has
   no L+1 and is skipped). `recall@6 = |predicted n actual| / 6`. Reported
   per-layer (mean over all tokens at that L) and as one aggregate mean
   over every (token, layer-pair) instance, across all four files
   (per-file means combined weighted by sample count, never by merging
   raw per-file token dicts -- token indices restart at 0 in every file
   and must never be compared across prompts).
2. **Same-layer consecutive-token overlap** (the temporal-locality / LRU
   signal, a genuinely different question): for each layer L and each
   token t where L has records at both t and t+1 *within the same file*,
   `overlap@6 = |ids(L,t) n ids(L,t+1)| / 6`.

Run:

```
python3 tools/pilot_routing.py \
  weights/pilot-runs/prompt1.route weights/pilot-runs/prompt2.route \
  weights/pilot-runs/prompt3.route weights/pilot-runs/prompt4.route \
  --markdown
```

**Determinism**: run twice against the identical four input files;
`diff` of the two full outputs (including every per-layer row) was empty.

**Random-chance baseline** (for calibration, not measured -- simple
combinatorics): for two independent uniformly-random 6-subsets of 256
experts, `E[|A n B|] = 6*6/256 = 0.1406`, so `E[recall@6] = 6/256 =
0.02344` (2.34%). This is the number a router with *zero* one-layer-ahead
structure would produce purely by chance, given 256 routed experts and
top-6 selection.

## Results

### Per-file summary

| File | Records | Distinct token positions | Layers |
|---|---:|---:|---|
| `prompt1.route` | 9,920 | 155 | 2-65 (64 layers) |
| `prompt2.route` | 9,856 | 154 | 2-65 (64 layers) |
| `prompt3.route` | 12,480 | 195 | 2-65 (64 layers) |
| `prompt4.route` | 13,056 | 204 | 2-65 (64 layers) |
| **Total** | **45,312** | **708** | |

Layers 0-1 never appear (dense, `dense_mlp_idx=2` -- confirmed empirically
here, matching `docs/architecture-notes.md`'s documented semantics); every
sparse layer 2-65 has a record for every logged token, with no gaps.

### One-layer-ahead recall@6 (PILOT/colibrì signal)

**Overall: mean recall@6 = 0.0234 (2.34%), n = 44,604 (token, layer-pair)
instances.** Per-layer range: min 0.0113 (L=18) to max 0.0499 (L=41) --
every one of the 63 layer-pairs (L=2..64, predicting L+1) sits within
roughly 1-5%, tightly clustered around the 2.34% random-chance baseline
with no layer showing a meaningfully elevated signal.

Per-prompt breakdown (same computation, one file at a time -- confirms the
result is not an artifact of one topic):

| Prompt | Mean recall@6 |
|---|---:|
| 1 (France, factual) | 0.0205 |
| 2 (fibonacci, code) | 0.0246 |
| 3 (lighthouse, narrative) | 0.0242 |
| 4 (HTTP API, technical) | 0.0239 |

All four land in a narrow 2.0-2.5% band, indistinguishable from chance
(2.34%) and from each other. Topic diversity does not surface any
one-layer-ahead structure.

Full per-layer curve (all 63 predicting layers, N=708 tokens each,
combined across all four prompts):

| Layer L | N | recall@6 |
|---:|---:|---:|
| 2 | 708 | 0.0233 |
| 3 | 708 | 0.0245 |
| 4 | 708 | 0.0153 |
| 5 | 708 | 0.0252 |
| 6 | 708 | 0.0195 |
| 7 | 708 | 0.0169 |
| 8 | 708 | 0.0228 |
| 9 | 708 | 0.0233 |
| 10 | 708 | 0.0280 |
| 11 | 708 | 0.0148 |
| 12 | 708 | 0.0167 |
| 13 | 708 | 0.0377 |
| 14 | 708 | 0.0259 |
| 15 | 708 | 0.0320 |
| 16 | 708 | 0.0308 |
| 17 | 708 | 0.0266 |
| 18 | 708 | 0.0113 |
| 19 | 708 | 0.0158 |
| 20 | 708 | 0.0141 |
| 21 | 708 | 0.0238 |
| 22 | 708 | 0.0247 |
| 23 | 708 | 0.0278 |
| 24 | 708 | 0.0153 |
| 25 | 708 | 0.0167 |
| 26 | 708 | 0.0129 |
| 27 | 708 | 0.0160 |
| 28 | 708 | 0.0188 |
| 29 | 708 | 0.0202 |
| 30 | 708 | 0.0261 |
| 31 | 708 | 0.0318 |
| 32 | 708 | 0.0207 |
| 33 | 708 | 0.0306 |
| 34 | 708 | 0.0167 |
| 35 | 708 | 0.0148 |
| 36 | 708 | 0.0132 |
| 37 | 708 | 0.0245 |
| 38 | 708 | 0.0193 |
| 39 | 708 | 0.0191 |
| 40 | 708 | 0.0238 |
| 41 | 708 | 0.0499 |
| 42 | 708 | 0.0282 |
| 43 | 708 | 0.0191 |
| 44 | 708 | 0.0313 |
| 45 | 708 | 0.0341 |
| 46 | 708 | 0.0247 |
| 47 | 708 | 0.0207 |
| 48 | 708 | 0.0355 |
| 49 | 708 | 0.0273 |
| 50 | 708 | 0.0313 |
| 51 | 708 | 0.0391 |
| 52 | 708 | 0.0212 |
| 53 | 708 | 0.0292 |
| 54 | 708 | 0.0179 |
| 55 | 708 | 0.0228 |
| 56 | 708 | 0.0231 |
| 57 | 708 | 0.0184 |
| 58 | 708 | 0.0162 |
| 59 | 708 | 0.0174 |
| 60 | 708 | 0.0207 |
| 61 | 708 | 0.0299 |
| 62 | 708 | 0.0485 |
| 63 | 708 | 0.0122 |
| 64 | 708 | 0.0122 |

(Layer 65 has no successor and is excluded, per the method above.)

### Temporal locality (same-layer, consecutive-token overlap -- the LRU signal)

**Overall: mean overlap@6 = 0.3822 (38.2%), n = 45,056 (token, layer)
instances.** This is ~16x the 2.34% chance baseline -- a real, substantial
signal that a token's chosen experts at a fixed layer strongly resemble
the *previous* token's chosen experts at that *same* layer. Per-layer
range: min 0.1226 (L=2, the very first sparse layer) to max 0.5765 (L=65,
the last layer). The curve rises fairly monotonically with depth (roughly
12% at L=2 to 35-45% through the middle layers to 58% at L=65), consistent
with -- and a plausible mechanistic explanation for -- Task 12's measured
65-95% LRU cache hit rates (`docs/p2-perf.md`): consecutive tokens really
do re-use much of the same expert set at a given layer, which is exactly
what an LRU eviction policy exploits. This is *not* the PILOT signal
(next-layer prediction); it is the reason the existing cache already
works reasonably well without any prefetch.

Per-prompt breakdown:

| Prompt | Mean overlap@6 |
|---|---:|
| 1 (France, factual) | 0.3722 |
| 2 (fibonacci, code) | 0.3948 |
| 3 (lighthouse, narrative) | 0.4150 |
| 4 (HTTP API, technical) | 0.3488 |

Full per-layer curve (all 64 sparse layers, N=704 consecutive-token pairs
each, combined across all four prompts):

| Layer L | N | overlap@6 |
|---:|---:|---:|
| 2 | 704 | 0.1226 |
| 3 | 704 | 0.2048 |
| 4 | 704 | 0.2481 |
| 5 | 704 | 0.2259 |
| 6 | 704 | 0.2479 |
| 7 | 704 | 0.2609 |
| 8 | 704 | 0.3291 |
| 9 | 704 | 0.3118 |
| 10 | 704 | 0.2940 |
| 11 | 704 | 0.3835 |
| 12 | 704 | 0.3651 |
| 13 | 704 | 0.3899 |
| 14 | 704 | 0.3464 |
| 15 | 704 | 0.3277 |
| 16 | 704 | 0.3497 |
| 17 | 704 | 0.3868 |
| 18 | 704 | 0.3504 |
| 19 | 704 | 0.3655 |
| 20 | 704 | 0.3741 |
| 21 | 704 | 0.3428 |
| 22 | 704 | 0.3693 |
| 23 | 704 | 0.4418 |
| 24 | 704 | 0.3613 |
| 25 | 704 | 0.3350 |
| 26 | 704 | 0.3880 |
| 27 | 704 | 0.3665 |
| 28 | 704 | 0.3802 |
| 29 | 704 | 0.3932 |
| 30 | 704 | 0.4091 |
| 31 | 704 | 0.4124 |
| 32 | 704 | 0.4285 |
| 33 | 704 | 0.4560 |
| 34 | 704 | 0.4943 |
| 35 | 704 | 0.4777 |
| 36 | 704 | 0.4164 |
| 37 | 704 | 0.4458 |
| 38 | 704 | 0.4612 |
| 39 | 704 | 0.4420 |
| 40 | 704 | 0.4697 |
| 41 | 704 | 0.4339 |
| 42 | 704 | 0.4458 |
| 43 | 704 | 0.3857 |
| 44 | 704 | 0.5024 |
| 45 | 704 | 0.4465 |
| 46 | 704 | 0.4138 |
| 47 | 704 | 0.3771 |
| 48 | 704 | 0.4186 |
| 49 | 704 | 0.3866 |
| 50 | 704 | 0.4098 |
| 51 | 704 | 0.4427 |
| 52 | 704 | 0.3459 |
| 53 | 704 | 0.3958 |
| 54 | 704 | 0.3369 |
| 55 | 704 | 0.3435 |
| 56 | 704 | 0.3606 |
| 57 | 704 | 0.3930 |
| 58 | 704 | 0.3795 |
| 59 | 704 | 0.3617 |
| 60 | 704 | 0.4242 |
| 61 | 704 | 0.4595 |
| 62 | 704 | 0.3840 |
| 63 | 704 | 0.3793 |
| 64 | 704 | 0.4818 |
| 65 | 704 | 0.5765 |

## Comparison against the thresholds

- **colibrì's GLM-5.2 baseline: 71.6% one-layer-ahead recall** (vs 41.3%
  naive), from `~/.claude/plans/i-want-to-create-synthetic-newell.md`
  (the approved external plan), line 44: "PILOT: run the real router one
  layer ahead on stale post-attention state -> 71.6% recall on GLM-5.2 (vs
  41.3% naive)".
- **P3-viability threshold: >= ~60% recall.** Source: the same approved
  external plan, line 100 -- "P3: ... PILOT prefetch (if P2 experiment
  shows >=~60% recall) ..." -- and this plan document's own Task 13
  section, which restates the threshold and instructs it be cited here as
  its first entry into the repo's own docs.
- **Measured on Inkling: 2.34% mean recall@6**, indistinguishable from the
  2.34% random-chance baseline for uniform 6-of-256 selection.

**Inkling clears neither threshold, by a wide margin.** 2.34% is roughly
30x below the ~60% P3-viability floor and roughly 31x below colibrì's
71.6% GLM-5.2 measurement -- not a marginal miss, a complete absence of
the effect. The per-layer curve (above) shows no layer, and the per-prompt
breakdown shows no topic, coming anywhere close to either threshold; the
result is not sensitive to which 63-layer slice or which prompt is
examined.

## Verdict

**Inkling's MoE router shows no one-layer-ahead routing predictability
beyond random chance.** Measured mean recall@6 = 2.34%, matching the
2.34% expected value for two independent uniform-random 6-of-256 expert
selections, across all four tested prompts (factual, code, narrative,
technical) and across every one of the 63 measurable layer-pairs. This is
a clean negative result, not an ambiguous one: the signal colibrì found on
GLM-5.2 (71.6%) simply is not present in this architecture's router at
one-layer lookahead, at least not on this sample.

**Consequence for P3**: per the plan's own decision rule (>=~60% recall to
justify PILOT prefetch), **this result does not justify building PILOT
prefetch for P3.** The scope-fence note already excluded PILOT prefetch
from P2 pending this measurement (`docs/DESIGN.md`, roadmap); this
measurement means P3 should not spend effort on it either, absent some
other justification not covered by this experiment (e.g. a different
lookahead distance, a different staleness point in the router's inputs,
or a much larger sample revealing a weak but real effect this sample size
cannot detect -- none of which this experiment was scoped to test).

**What P3 *should* still take from this task**: the temporal-locality
result (38.2% mean same-layer consecutive-token overlap, ~16x chance,
rising with depth to 58% at the last layer) is real, substantial, and
already-exploited by the existing LRU cache (Task 10) -- it is the
mechanistic explanation for Task 12's measured 65-95% hit rates, not a new
opportunity, since the LRU policy already captures temporal locality by
construction. This experiment does not identify a route to a fundamentally
different cache strategy; it explains why the current one already works.

Published either way, as the project's design doc committed to -- this is
the "either way" case: a negative result, reported without hedging.

## Reproduction

```
make ci                                     # weights-free gates, still green
python3 tools/pilot_routing.py \
  weights/pilot-runs/prompt1.route weights/pilot-runs/prompt2.route \
  weights/pilot-runs/prompt3.route weights/pilot-runs/prompt4.route \
  --markdown
```

`weights/pilot-runs/*.route` are gitignored (same convention as
`weights/p1-runs/` and `weights/p2-runs/`); regenerate with the four
`--route-log` commands above against the real 317GB weights.

## Concerns

- **Sample size and lookahead scope**: 708 token positions across 4
  prompts is enough to establish that the effect, if any, is far below
  the decision threshold -- it is not enough to rule out a small but
  non-zero effect at a much larger sample, nor does it test any lookahead
  distance other than exactly one layer, nor any router-input staleness
  point other than the one this codebase's forward pass naturally
  produces (colibrì's own method used "stale post-attention state";
  this measurement uses the router's normal, non-stale inputs since
  SEPIA's forward pass has no separate stale-state mechanism to test
  against). A qualitatively different result at another lookahead
  distance or staleness point cannot be ruled out by this experiment.
- **No cross-model comparison run here**: colibrì's 71.6% is quoted from
  the approved external plan, not re-measured against GLM-5.2 on this
  machine -- the comparison is against a cited number, not a fresh
  apples-to-apples run under identical tooling.
- **Route-log's inertness check is a spot-check, not a full re-run**: the
  8-token direct A/B and the 30-32 token cross-check against
  `docs/p1-first-tokens.md` are strong evidence but not an independent
  double-run of all 708 tokens with and without the flag (that would cost
  another ~8 minutes of GPU time for marginal additional confidence given
  the write is provably a pure observer -- see `src/sepia.c`'s comment at
  the write call site).
