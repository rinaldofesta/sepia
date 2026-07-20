# P1: first real-weight tokens, cross-checked

2026-07-20, M5 Max 128GB, branch `p1-cpu-dequant`. The engine's `--real` mode
(scalar CPU path: per-layer resident dequant into an f32 arena + per-token
expert streaming from the GGUF parts) generating greedily from the full
UD-Q2_K_XL Inkling weights (317GB, 8 parts, no repack).

## Runs

Each prompt was run twice, back to back, 32 greedy tokens per run
(`./sepia --real --prompt "..." --n-gen 32`). Raw logs: `weights/p1-runs/`
(gitignored).

### Prompt 1: `The capital of France is`

Continuation (all 32 tokens):

```
 Paris. The capital of the UK is London. The capital of Russia is Moscow.
 The capital of Italy is Rome. The capital of Spain is Madrid. The
```

Token ids: `12650 13 623 9029 328 290 8370 382 9741 13 623 9029 328 19420 382
51802 13 623 9029 328 22384 382 27388 13 623 9029 328 26350 382 20264 13 623`

### Prompt 2: `def fibonacci(n):`

Continuation (32 tokens; the model greedily continues as a Jupyter-notebook
JSON fragment, a plausible base-completion regime for this prompt):

```
\n",
    "    # Base case: if n is 0 or 1, return n\n",
    "    if n <= 1:\
```

Token ids: `59 77 1150 271 392 271 1069 8729 1890 25 538 297 382 220 15 503
220 16 11 622 297 3392 1150 271 392 271 538 297 5017 220 16 16008`

## Determinism

Both prompts: the two independent runs produced identical 32-token id
sequences (compared record-by-record from the run logs, newline-bearing
tokens included). Exact-match, no tolerance.

## Timing and memory (informational — P1 is correctness-only by design)

- First token (includes prefill + first full layer-dequant pass):
  135.6s (prompt 1) / 106.9s (prompt 2).
- Steady-state decode: mean 31.0s/token (prompt 1), 28.9s/token (prompt 2).
  The cost is dominated by re-dequantizing each layer's resident weights
  every token on a single scalar thread; the plan accepted minutes/token
  for P1, and this is the lever P2's Metal path replaces.
- Peak RSS: 15.85GB (14.23GB resident.bin mmap + arena + KV/scratch).
- Real-model load: 0.17s (resident.bin mmap'd, 8 part fds, F_NOCACHE,
  index/manifest parsed, slab shapes validated).

## llama.cpp cross-check

Reference build: the pinned Inkling draft-PR checkout (`vendor/PIN`, commit
`ce16fff2ad1d1ddfb39365fedfb4fcfd6db178f1`), CPU-only
(`-DGGML_METAL=OFF -DLLAMA_CURL=OFF`). Generations via **`llama-completion`**
`-n 32 --temp 0 --top-k 1 -ctk f32 -ctv f32 -no-cnv` (raw completion; at this
pin `llama-cli` is an interactive-first UI, and without `-no-cnv` the
completion tool auto-enters conversation mode and aborts on Inkling's custom
chat template).

### Tokenizer

`llama-tokenize` vs SEPIA's tokenizer on 7 diverse cases (ASCII, accents,
CJK, contractions, digit runs, mixed whitespace): **7/7 exact id-sequence
match**, no BOS prepended on either side (`add_bos_token=false`). Combined
with the committed fixture gates (mini 9/9, real-vocab 25/25 vs HF
`tokenizers`, plus an ~8300-string stress sweep with zero mismatches), the
three implementations agree exactly on everything tested.

### Greedy tokens

- **Prompt 1: id-exact.** Identical 32-token text, and SEPIA's
  `prompt_ids + generated_ids` are exactly the canonical tokenization of
  that shared text (verified via `llama-tokenize` round-trip) — the two
  engines walked the same id path.
- **Prompt 2: text-exact.** Identical 32-token text. A direct id dump is
  not extractable from `llama-completion` at this pin (its verbose log
  carries no per-step sample ids), and the canonical-retokenization proof
  does not apply here because the continuation is escape-heavy (it ends in
  a bare `\`, and greedy id sequences are not canonical at such
  boundaries). Identical text over 32 consecutive greedy steps with
  proven-identical tokenizers and identical prompt tokenization
  (llama.cpp's `n_past` starts at 4 = our prompt token count) leaves no
  plausible divergent-id path; recorded as text-exact rather than claiming
  more than was measured.

No divergence appeared at any position on either prompt, so the near-tie
protocol (top-2 logit comparison at the first divergent step) was never
invoked.

## Verdict

P1's acceptance line — "first real greedy tokens on big Inkling, correctness
only, cross-checked against a llama.cpp PR-branch build on the same GGUF" —
is met: deterministic generations on both prompts, byte-identical
continuations vs the reference, id-exact where id-level evidence exists.
