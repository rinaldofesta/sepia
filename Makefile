CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDFLAGS ?= -pthread -lm

# Targets grow with the phases:
#   sepia    (0.4)  CPU reference engine
#   iobench  (0.5)  SSD microbenchmark
#   test     (0.4)  oracle self-test
ci: pycheck sepia
	./sepia
	@echo "ci ok"

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

.PHONY: ci pycheck sepia test iobench
