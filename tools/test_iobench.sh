#!/usr/bin/env bash
# Correctness test for tools/iobench.c: builds it, runs it against a small
# generated file, and checks the reported bytes match n_reads * block_bytes
# exactly. Also asserts the documented expected-failure behavior: a file
# smaller than one block is a loud, nonzero-exit error, not a short read.
set -euo pipefail
cd "$(dirname "$0")/.."

make iobench >/dev/null
BIN=./iobench

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

small_file="$tmpdir/small.bin"
dd if=/dev/urandom of="$small_file" bs=1m count=64 >/dev/null 2>&1

check_bytes() {
    local direct=$1
    local align_bytes=${2:-}
    local block_mb=4 n_reads=40 threads=4
    local out
    if [ -n "$align_bytes" ]; then
        out=$("$BIN" "$small_file" "$block_mb" "$n_reads" "$threads" "$direct" "$align_bytes")
    else
        out=$("$BIN" "$small_file" "$block_mb" "$n_reads" "$threads" "$direct")
    fi
    echo "$out"
    local expect_gb
    expect_gb=$(python3 -c "print(f'{$n_reads*$block_mb*1024*1024/1e9:.4f}')")
    local got_gb
    got_gb=$(echo "$out" | sed -n 's/.*total_gb=\([0-9.]*\).*/\1/p')
    if [ "$got_gb" != "$expect_gb" ]; then
        echo "FAIL: direct=$direct align_bytes=$align_bytes expected total_gb=$expect_gb got $got_gb"
        exit 1
    fi
    echo "ok: direct=$direct align_bytes=${align_bytes:-0 (default)} total_gb=$got_gb matches $n_reads x ${block_mb}MiB"
}

check_bytes 0
check_bytes 1

# align_bytes: task 0.7's unaligned-offset mode. Omitted (tested above) must
# behave exactly like align_bytes=0 (block-aligned, the original behavior);
# a positive value must still read exactly n_reads * block_mb total bytes,
# just from GGUF-realistic unaligned starting offsets.
check_bytes 1 0
check_bytes 1 32

# Expected-failure: block bigger than the whole file must abort loudly
# (nonzero exit, message on stderr), not silently short-read.
tiny_file="$tmpdir/tiny.bin"
dd if=/dev/urandom of="$tiny_file" bs=1m count=1 >/dev/null 2>&1
if err=$("$BIN" "$tiny_file" 4 8 2 0 2>&1); then
    echo "FAIL: expected nonzero exit for block larger than file"
    exit 1
fi
if ! echo "$err" | grep -q "too small"; then
    echo "FAIL: expected 'too small' message, got: $err"
    exit 1
fi
echo "ok: block-larger-than-file fails loudly as documented"

echo "test_iobench: all checks passed"
