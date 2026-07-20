CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDFLAGS ?= -pthread -lm

# Targets grow with the phases:
#   sepia    (0.4)  CPU reference engine
#   iobench  (0.5)  SSD microbenchmark
#   test     (0.4)  oracle self-test
ci: pycheck tooltests sepia test_quants
	./sepia
	./test_quants tools/fixtures/quants/f16.bin tools/fixtures/quants/q8_0.bin tools/fixtures/quants/q4_k.bin
	@echo "ci ok"

# Tool test suites. Excluded on purpose: tools/test_oracle_determinism.sh
# (needs the torch venv, absent in CI; re-proven locally whenever the
# oracle regenerates).
tooltests: iobench
	python3 tools/test_gguf_inspect.py
	python3 tools/test_make_index.py
	python3 tools/test_extract_resident.py
	bash tools/test_iobench.sh
	bash tools/test_sepia_malformed.sh
	@echo "tooltests ok"

pycheck:
	@if ls tools/*.py >/dev/null 2>&1; then \
		python3 -m py_compile tools/*.py && echo "pycheck ok"; \
	else \
		echo "pycheck: no python tools yet"; \
	fi

sepia: src/sepia.c
	$(CC) $(CFLAGS) -o sepia src/sepia.c $(LDFLAGS)

test: sepia
	./sepia

# iobench needs a large local test file (weights/iobench.bin); ci does not
# run it, only compiles it via this target on demand.
iobench: tools/iobench.c
	$(CC) $(CFLAGS) -o iobench tools/iobench.c $(LDFLAGS)

test_quants: tools/test_quants.c src/quants.c src/quants.h
	$(CC) $(CFLAGS) -o test_quants tools/test_quants.c src/quants.c $(LDFLAGS)

.PHONY: ci pycheck tooltests sepia test iobench test_quants
