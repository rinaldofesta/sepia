/* src/sepia_gpu.h -- the ONLY GPU symbol surface sepia.c sees.
 * All functions return 0/NULL on failure after logging "sepia: metal: ...";
 * sepia.c decides whether that is fatal (--metal explicitly requested -> die). */
#ifndef SEPIA_GPU_H
#define SEPIA_GPU_H
#include <stdint.h>
#include <stddef.h>

int  sepia_gpu_available(void);                 /* 1 if a device exists and library compiled */
int  sepia_gpu_init(const char *metal_dir);     /* compiles the *.metal files under this dir; 1 on success */
void sepia_gpu_shutdown(void);
const char *sepia_gpu_device_name(void);

/* -------------------------------- buffers -------------------------------- */

/* Opaque; wraps an id<MTLBuffer> (and, in later tasks, an offset). Never
 * dereferenced by sepia.c -- only passed around and handed back to the
 * functions below. */
typedef struct SepiaGpuBuf SepiaGpuBuf;

/* Wraps an existing host allocation as a zero-copy Shared-storage MTLBuffer
 * (newBufferWithBytesNoCopy): no data movement, GPU and CPU see the same
 * bytes. `base` MUST be page-aligned (mmap's return value always is;
 * anything else is a caller bug, not a runtime condition -- returns NULL
 * and logs rather than asserting, so callers can probe it, e.g. the
 * --gpu-selftest alignment-violation case). `len` is rounded up to the
 * page size internally (Metal requires a page-multiple length for
 * bytesNoCopy) -- the tail bytes between `len` and the rounded length
 * belong to the underlying mapping (safe for a whole-page mmap like
 * resident.bin's) and must not be treated as valid tensor data by callers.
 * NULL on failure (misaligned base, zero length, no device, length beyond
 * the device's maxBufferLength, or Metal rejecting the wrap). */
SepiaGpuBuf *sepia_gpu_wrap_mmap(void *base, size_t len);

/* Allocates a fresh MTLBuffer: Shared storage (CPU+GPU visible via
 * host_ptr) when gpu_private is 0, Private storage (GPU-only, host_ptr
 * always NULL) when non-zero. NULL on failure. */
SepiaGpuBuf *sepia_gpu_alloc(size_t len, int gpu_private);

/* Releases the underlying MTLBuffer (a no-copy wrap never frees the host
 * memory it wraps -- that stays the caller's, e.g. resident.bin's mmap
 * outlives this call). NULL is a no-op. */
void sepia_gpu_free(SepiaGpuBuf *b);

/* [buffer contents]: the host-visible pointer for Shared buffers (for a
 * wrap_mmap buffer this is literally the wrapped `base`), NULL for
 * Private-storage buffers and NULL b. */
void *sepia_gpu_host_ptr(SepiaGpuBuf *b);

/* [buffer gpuAddress]: the buffer's device VA, for bindless tables (Task
 * 10). 0 for NULL b. */
uint64_t sepia_gpu_gpu_addr(SepiaGpuBuf *b);

/* Local-only, --gpu-selftest support: dispatches the `sepia_touch` kernel
 * (metal/ops.metal) over the first n_floats floats of buf's contents as a
 * single synchronous command buffer (encode, commit, waitUntilCompleted).
 * Not part of the stable dispatch API -- Task 3 introduces the real
 * per-op dispatch machinery (batched encoder, args structs); this exists
 * only so the selftest can prove a wrapped buffer round-trips through
 * actual GPU execution. Returns 1 on success, 0 on failure. */
int sepia_gpu_selftest_touch(SepiaGpuBuf *buf, size_t n_floats);

#endif
