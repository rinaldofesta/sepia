CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDFLAGS ?= -pthread -lm

UNAME_S := $(shell uname -s)

# GPU runtime: an Objective-C/Metal shim on Darwin, a symbol-compatible
# no-op stub everywhere else (src/sepia_gpu.h is the shared contract).
# METAL_LDFLAGS is appended in the `sepia` recipe below, never merged into
# LDFLAGS itself -- that would silently push `-framework ...` onto every
# other target's link line and defeat the `?=` default for non-Darwin/non-
# Metal builds.
ifeq ($(UNAME_S),Darwin)
OBJCFLAGS ?= -O2 -fobjc-arc -Wall -Wextra
METAL_LDFLAGS = -framework Metal -framework Foundation
METAL_SRCS := $(wildcard metal/*.metal)
GPU_OBJ := src/sepia_metal.o
else
GPU_OBJ := src/sepia_gpu_stub.o
endif

# Targets grow with the phases:
#   sepia    (0.4)  CPU reference engine
#   iobench  (0.5)  SSD microbenchmark
#   test     (0.4)  oracle self-test
ci: pycheck tooltests sepia test_quants test_tokenizer shadercheck
	./sepia
	./test_quants tools/fixtures/quants/f16.bin tools/fixtures/quants/q8_0.bin tools/fixtures/quants/q4_k.bin tools/fixtures/quants/q5_k.bin tools/fixtures/quants/q6_k.bin tools/fixtures/quants/iq2_xs.bin tools/fixtures/quants/iq3_xxs.bin tools/fixtures/quants/iq4_xs.bin tools/fixtures/quants/qlinear_q8_0.bin
	./test_tokenizer tools/fixtures/tokenizer/mini.bin tools/fixtures/tokenizer/mini_cases.json
	@echo "ci ok"

# Tool test suites. Excluded on purpose: tools/test_oracle_determinism.sh
# (needs the torch venv, absent in CI; re-proven locally whenever the
# oracle regenerates). `sepia` is a prerequisite (not just built later by
# `ci`'s own list) so `make tooltests` alone always has a fresh binary for
# the --smoke line below.
tooltests: iobench sepia
	python3 tools/test_gguf_inspect.py
	python3 tools/test_make_index.py
	python3 tools/test_extract_resident.py
	python3 tools/test_export_tokenizer.py
	bash tools/test_iobench.sh
	bash tools/test_sepia_malformed.sh
	python3 tools/make_smoke_fixture.py --out "$${TMPDIR:-/tmp}/sepia-smoke" && ./sepia --smoke "$${TMPDIR:-/tmp}/sepia-smoke"
	@echo "tooltests ok"

pycheck:
	@if ls tools/*.py >/dev/null 2>&1; then \
		python3 -m py_compile tools/*.py && echo "pycheck ok"; \
	else \
		echo "pycheck: no python tools yet"; \
	fi

sepia: src/sepia.c src/quants.c src/quants.h src/tokenizer.c src/tokenizer.h src/sepia_gpu.h $(GPU_OBJ)
	$(CC) $(CFLAGS) -o sepia src/sepia.c src/quants.c src/tokenizer.c $(GPU_OBJ) $(LDFLAGS) $(METAL_LDFLAGS)

# .metal files are listed only as a rebuild trigger for the shim object --
# they are read at runtime by sepia_gpu_init(), never compiled in here.
# quants.h is a real #include (Task 4's matvec_q dispatch switches on its
# SEPIA_T_* type ids), so it belongs in the prerequisite list too.
src/sepia_metal.o: src/sepia_metal.m src/sepia_gpu.h src/quants.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o src/sepia_metal.o src/sepia_metal.m

src/sepia_gpu_stub.o: src/sepia_gpu_stub.c src/sepia_gpu.h
	$(CC) $(CFLAGS) -c -o src/sepia_gpu_stub.o src/sepia_gpu_stub.c

# Offline Metal shader syntax/type check -- no device needed, so it runs in
# CI (macos-latest ships the toolchain). Compiles the SAME translation unit
# sepia_gpu_load_source builds at runtime: metal/*.metal concatenated in
# sorted order (plain glob order in sh is already lexicographic, matching
# that loader's `sortedArrayUsingSelector:@selector(compare:)`) into one temp
# file, then `metal -c` on that single file -- not per-file, since some
# files (e.g. matvec_q.metal) reference `constant` tables defined in another
# file (00_quants_grids.metal) and only compile standalone thanks to their
# own `extern constant` forward declarations; checking the concatenated TU
# is what actually matches what ships. Non-Darwin: no Metal toolchain
# exists, so this is a no-op rather than a hard failure. Darwin machines
# with only Command Line Tools (no Xcode.app) have the Metal framework but
# not the offline `metal` compiler -- `xcrun -f metal` detects that so this
# skips gracefully instead of hard-failing `make ci`; runtime compilation
# via `--metal` is unaffected either way.
shadercheck:
ifeq ($(UNAME_S),Darwin)
	@if xcrun -f metal >/dev/null 2>&1; then \
		tmp="$${TMPDIR:-/tmp}/sepia_shadercheck_$$$$.metal"; \
		rm -f "$$tmp"; \
		for f in metal/*.metal; do \
			echo "shadercheck: concatenating $$f"; \
			cat "$$f" >> "$$tmp"; \
		done; \
		xcrun -sdk macosx metal -c "$$tmp" -o /dev/null; rc=$$?; \
		rm -f "$$tmp"; \
		if [ "$$rc" -ne 0 ]; then exit 1; fi; \
		echo "shadercheck ok"; \
	else \
		echo "shadercheck: skipped (no offline Metal compiler; runtime compile covered by --metal)"; \
	fi
else
	@echo "shadercheck: skipped (not Darwin)"
endif

test: sepia
	./sepia

# iobench needs a large local test file (weights/iobench.bin); ci does not
# run it, only compiles it via this target on demand.
iobench: tools/iobench.c
	$(CC) $(CFLAGS) -o iobench tools/iobench.c $(LDFLAGS)

test_quants: tools/test_quants.c src/quants.c src/quants.h
	$(CC) $(CFLAGS) -o test_quants tools/test_quants.c src/quants.c $(LDFLAGS)

test_tokenizer: tools/test_tokenizer.c src/tokenizer.c src/tokenizer.h src/unicode_tables.h
	$(CC) $(CFLAGS) -o test_tokenizer tools/test_tokenizer.c src/tokenizer.c $(LDFLAGS)

# local-only: needs weights/tokenizer.bin (see tools/export_tokenizer.py)
tokreal: test_tokenizer
	./test_tokenizer weights/tokenizer.bin tools/fixtures/tokenizer/real_cases.json

# local-only: verifies docs/inkling-config.json against GGUF part 1 metadata
# (needs weights/inkling-gguf/..., not fetched in ci)
configcheck:
	python3 tools/check_inkling_config.py

# local-only: needs a live Metal device (--gpu-selftest dies cleanly without
# one); exercises the zero-copy buffer API end-to-end, then (Task 8) runs the
# plain tiny-oracle self-test through the full Metal forward path and
# requires the same token-exact prefill 32/32 + decode 20/20 the CPU oracle
# gates on. Not in ci -- ci stays weights-free and device-free per the
# Global Constraints.
gputest: sepia
	./sepia --metal --gpu-selftest
	./sepia --metal

# local-only: needs a live Metal device and the committed tiny oracle fixture
# (tools/oracle/tiny/, already required by the plain self-test) -- the Task 3
# dev-loop gate. Drives one real T=32 prefill on the CPU, replays every
# captured rmsnorm/matvec/silu_mul/add/softmax/sconv instance on the GPU, and
# dies loudly if any op kind's worst instance exceeds the 2e-4 relative-error
# tolerance. Not in ci for the same reason gputest isn't (needs a live device).
gpucompare: sepia
	./sepia --metal --gpu-compare-tiny

# local-only: needs a live Metal device; the Task 4/5/6 gate for the
# Q8_0/Q4_K/Q5_K/Q6_K/IQ2_XS/IQ3_XXS/IQ4_XS dequant-fused matvec kernels --
# bitwise dequant vs the committed SQFX fixtures, plus a tolerance-checked
# matvec vs CPU qlinear. Not in ci for the same reason gputest/gpucompare
# aren't (needs a live device).
gpuquants: sepia
	./sepia --metal --gpu-quants tools/fixtures/quants/q8_0.bin tools/fixtures/quants/q4_k.bin tools/fixtures/quants/q5_k.bin tools/fixtures/quants/q6_k.bin tools/fixtures/quants/iq2_xs.bin tools/fixtures/quants/iq3_xxs.bin tools/fixtures/quants/iq4_xs.bin

# local-only: needs a live Metal device; Task 7 Gate A for the banded
# flash-attention kernels (sepia_gpu_rel_project + sepia_gpu_banded_attn) --
# synthetic randomized tensors at both real per-layer-type geometries
# (sliding H64/Hkv16/Dh128/window512/rel512; global H64/Hkv8/Dh128/rel1024)
# compared against the REAL attn_forward_chunk CPU oracle across the
# n_kv/tau edge cases (band cutoff live, log-scaling floor). Gate B (the
# tiny-model attention-swap comparison) folds into gpucompare instead, since
# it reuses --gpu-compare-tiny's captured tiny-oracle forward pass. Not in
# ci for the same reason gputest/gpucompare/gpuquants aren't (needs a live
# device).
gpuattn: sepia
	./sepia --metal --gpu-compare-attn

# local-only: needs a live Metal device AND the real 317GB UD-Q2_K_XL weights
# under weights/ -- Task 9/10/11's real-model resident-path gate. Reproduces
# the exact 32-token id sequences P1 recorded on the CPU path (docs/
# p1-first-tokens.md) for both prompts, on the --metal GPU path (quantized
# resident weights read straight off gpu_res_buf; routed experts stream
# through the GPU-resident LRU expert cache, Task 10, now overlapped with
# GPU compute via a loader thread pool, Task 11). --repeat 2 runs each
# prompt twice in the SAME process (run 1 cold, run 2 warm -- the cache is
# empty at process start and stays populated across the repeat) and proves
# double-run determinism at the same time; --verbose-cache reports the
# hit/miss counts Task 10's gate asks for. --expert-io-mode {nocache|
# pagecache} (Task 11) selects the routed-expert pread path for the A/B --
# not exercised by this default invocation (F_NOCACHE). Not in ci (needs
# weights + a live device, same reason gputest/gpucompare/gpuattn/gpuquants
# aren't).
realmetal: sepia
	./sepia --metal --real --prompt "The capital of France is" --n-gen 32 --repeat 2 --verbose-cache
	./sepia --metal --real --prompt "def fibonacci(n):" --n-gen 32 --repeat 2 --verbose-cache

.PHONY: ci pycheck tooltests sepia test iobench test_quants test_tokenizer tokreal configcheck shadercheck gputest gpucompare gpuquants gpuattn realmetal
