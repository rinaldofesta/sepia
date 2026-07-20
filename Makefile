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
src/sepia_metal.o: src/sepia_metal.m src/sepia_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o src/sepia_metal.o src/sepia_metal.m

src/sepia_gpu_stub.o: src/sepia_gpu_stub.c src/sepia_gpu.h
	$(CC) $(CFLAGS) -c -o src/sepia_gpu_stub.o src/sepia_gpu_stub.c

# Offline Metal shader syntax/type check -- no device needed, so it runs in
# CI (macos-latest ships the toolchain). `metal -c` refuses multiple inputs
# sharing one -o (mirrors clang -c), hence the per-file loop. Non-Darwin:
# no Metal toolchain exists, so this is a no-op rather than a hard failure.
# Darwin machines with only Command Line Tools (no Xcode.app) have the Metal
# framework but not the offline `metal` compiler -- `xcrun -f metal` detects
# that so this skips gracefully instead of hard-failing `make ci`; runtime
# compilation via `--metal` is unaffected either way.
shadercheck:
ifeq ($(UNAME_S),Darwin)
	@if xcrun -f metal >/dev/null 2>&1; then \
		for f in metal/*.metal; do \
			echo "shadercheck: $$f"; \
			xcrun -sdk macosx metal -c "$$f" -o /dev/null || exit 1; \
		done; \
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

.PHONY: ci pycheck tooltests sepia test iobench test_quants test_tokenizer tokreal configcheck shadercheck
