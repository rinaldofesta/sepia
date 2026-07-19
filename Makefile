CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra

# Targets grow with the phases:
#   sepia    (0.4)  CPU reference engine
#   iobench  (0.5)  SSD microbenchmark
#   test     (0.4)  oracle self-test
ci: pycheck
	@echo "ci ok"

pycheck:
	@if ls tools/*.py >/dev/null 2>&1; then \
		python3 -m py_compile tools/*.py && echo "pycheck ok"; \
	else \
		echo "pycheck: no python tools yet"; \
	fi

.PHONY: ci pycheck
