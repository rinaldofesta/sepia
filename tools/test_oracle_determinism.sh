#!/usr/bin/env bash
# Determinism gate for tools/make_oracle.py: runs the generator twice in
# fresh subprocesses and asserts ref_inkling.json, config.json, and
# model.safetensors come out byte-identical. Runnable with no arguments.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python tools/make_oracle.py --check
