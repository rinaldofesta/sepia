#!/usr/bin/env bash
# Regression test for the safetensors bounds check in src/sepia.c's
# st_find() (reviewer-found gap on task 0.4): a hand-crafted header can have
# data_offsets that are internally self-consistent (o1-o0 matches the
# declared shape) but still point past EOF. Before the fix this silently
# read out-of-mapping memory or segfaulted; it must now die loudly with a
# clear message and exit 1.
#
# Uses SEPIA_CONFIG_PATH / SEPIA_WEIGHTS_PATH / SEPIA_REF_PATH (env
# overrides on the committed fixture paths, added alongside the fix) to
# point a built ./sepia at a malformed copy of the tiny checkpoint without
# touching the Makefile or the committed fixtures themselves.
set -euo pipefail
cd "$(dirname "$0")/.."

make sepia >/dev/null
BIN=./sepia

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

cp tools/oracle/tiny/config.json "$tmpdir/config.json"

# Corrupt one tensor's data_offsets in the safetensors header: shift both
# offsets by the same huge delta, so o1-o0 (the byte-length-vs-shape check
# that already existed) still matches the declared shape exactly, but the
# resulting pointer lands far past the file's actual data region -- the
# specific case the byte-length check alone does not catch.
python3 - "tools/oracle/tiny/model.safetensors" "$tmpdir/model.safetensors" <<'PY'
import json
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    raw = f.read()

header_len = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8 : 8 + header_len])

name = "model.llm.embed.weight"
o0, o1 = header[name]["data_offsets"]
delta = 10_000_000_000  # self-consistent (o1-o0 unchanged), but far past any real file size
header[name]["data_offsets"] = [o0 + delta, o1 + delta]

new_header = json.dumps(header).encode("utf-8")
data = raw[8 + header_len :]
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(new_header)))
    f.write(new_header)
    f.write(data)
PY

set +e
out=$(SEPIA_CONFIG_PATH="$tmpdir/config.json" SEPIA_WEIGHTS_PATH="$tmpdir/model.safetensors" "$BIN" 2>&1)
status=$?
set -e

echo "$out"

if [ "$status" -eq 0 ]; then
    echo "FAIL: malformed checkpoint was accepted (exit 0)"
    exit 1
fi
if [ "$status" -ge 128 ]; then
    echo "FAIL: process crashed (exit $status, signal $((status - 128))) instead of failing loudly"
    exit 1
fi
if [ "$status" -ne 1 ]; then
    echo "FAIL: expected sepia's die() exit code 1, got $status"
    exit 1
fi
if ! echo "$out" | grep -q "fall outside"; then
    echo "FAIL: expected the data_offsets bounds-check message, got: $out"
    exit 1
fi
echo "ok: malformed data_offsets rejected loudly (exit 1, diagnostic message), not a crash"

# Sanity check that the env-var override itself is transparent: pointed at
# the real, unmodified fixtures, the self-test must still pass exactly as
# it does with no override at all.
out2=$(SEPIA_CONFIG_PATH="tools/oracle/tiny/config.json" \
       SEPIA_WEIGHTS_PATH="tools/oracle/tiny/model.safetensors" \
       SEPIA_REF_PATH="tools/oracle/ref_inkling.json" \
       "$BIN")
echo "$out2"
if ! echo "$out2" | grep -q "^prefill 32/32$" || ! echo "$out2" | grep -q "^decode 20/20$"; then
    echo "FAIL: env-path override changed self-test behavior against the real fixtures"
    exit 1
fi
echo "ok: env-var path override is transparent against the real fixtures"

echo "test_sepia_malformed: all checks passed"
