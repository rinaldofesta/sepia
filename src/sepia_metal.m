/* src/sepia_metal.m -- Objective-C Metal runtime shim.
 *
 * Task 1 scope: device/queue/library init and teardown only. Later tasks
 * (buffers, encoders, kernel dispatch, the expert store) build on this.
 *
 * Shader sources under metal_dir (its *.metal files) are concatenated and
 * handed to -newLibraryWithSource: at startup -- no metallib toolchain
 * step, no env-var source overrides, no pipeline-cache dictionary (all
 * YAGNI for this task). This mirrors ds4's runtime source-concatenation
 * pattern (ds4_metal.m:3039-3110 for the concat-and-compile shape,
 * :4561-4660 for the device/queue/newLibraryWithSource init) with the
 * env-override machinery and multi-file bookkeeping stripped out -- SEPIA
 * ports the pattern, not the code. See NOTICE.
 *
 * Every function returns 0/NULL on failure after logging
 * "sepia: metal: ..." to stderr; it never calls exit() -- sepia.c decides
 * whether a failure is fatal (a bare `--metal` request dies, later modes
 * may choose to fall back).
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "sepia_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_library;
static char g_device_name[256];
static int g_available;

/* Concatenates every *.metal file directly under metal_dir into one
 * translation unit for -newLibraryWithSource:. Sorted for a deterministic
 * build. Returns nil (after logging) if the directory can't be read, a
 * file can't be read, or no .metal files are found. */
static NSString *sepia_gpu_load_source(const char *metal_dir) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *dir = [fm stringWithFileSystemRepresentation:metal_dir
                                                      length:strlen(metal_dir)];

    NSError *list_error = nil;
    NSArray<NSString *> *entries = [fm contentsOfDirectoryAtPath:dir error:&list_error];
    if (!entries) {
        fprintf(stderr, "sepia: metal: cannot list %s: %s\n", metal_dir,
                list_error.localizedDescription.UTF8String);
        return nil;
    }

    NSArray<NSString *> *sorted = [entries sortedArrayUsingSelector:@selector(compare:)];
    NSMutableString *source = [NSMutableString string];
    int found = 0;
    for (NSString *name in sorted) {
        if (![name.pathExtension isEqualToString:@"metal"]) {
            continue;
        }
        NSString *path = [dir stringByAppendingPathComponent:name];
        NSError *read_error = nil;
        NSString *text = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&read_error];
        if (!text) {
            fprintf(stderr, "sepia: metal: cannot read %s: %s\n", path.UTF8String,
                    read_error.localizedDescription.UTF8String);
            return nil;
        }
        [source appendFormat:@"// -- %@ --\n%@\n", name, text];
        found = 1;
    }

    if (!found) {
        fprintf(stderr, "sepia: metal: no .metal sources found in %s\n", metal_dir);
        return nil;
    }
    return source;
}

int sepia_gpu_available(void) {
    return g_available;
}

int sepia_gpu_init(const char *metal_dir) {
    if (g_available) {
        return 1;
    }
    if (!metal_dir || !metal_dir[0]) {
        fprintf(stderr, "sepia: metal: init requires a metal_dir\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "sepia: metal: no Metal device available\n");
            return 0;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "sepia: metal: failed to create a command queue\n");
            return 0;
        }

        NSString *source = sepia_gpu_load_source(metal_dir);
        if (!source) {
            return 0;
        }

        NSError *compile_error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                       options:options
                                                         error:&compile_error];
        if (!library) {
            fprintf(stderr, "sepia: metal: shader compilation failed: %s\n",
                    compile_error.localizedDescription.UTF8String);
            return 0;
        }

        g_device = device;
        g_queue = queue;
        g_library = library;
        strncpy(g_device_name, device.name.UTF8String, sizeof(g_device_name) - 1);
        g_device_name[sizeof(g_device_name) - 1] = '\0';
        g_available = 1;
    }
    return 1;
}

void sepia_gpu_shutdown(void) {
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_device_name[0] = '\0';
    g_available = 0;
}

const char *sepia_gpu_device_name(void) {
    return g_available ? g_device_name : NULL;
}

/* --------------------------------- buffers -------------------------------- */

/* SepiaGpuBuf is a one-object-per-allocation heap node holding a strong
 * ARC reference to the id<MTLBuffer>. calloc (not malloc) is load-bearing:
 * ARC treats every store to a __strong field as retain-new/release-old, so
 * the field's initial value must be nil (calloc's zeroed memory) rather
 * than malloc's garbage -- otherwise the very first assignment below would
 * ask ARC to release an arbitrary bit pattern. sepia_gpu_free mirrors this
 * by nil-ing the field (an ARC release) before the raw free(), rather than
 * freeing out from under a live strong reference. */
struct SepiaGpuBuf {
    id<MTLBuffer> buffer;
};

static size_t sepia_gpu_round_up_page(size_t len, size_t page) {
    return (len + page - 1) & ~(page - 1);
}

SepiaGpuBuf *sepia_gpu_wrap_mmap(void *base, size_t len) {
    if (!g_available || !g_device) {
        fprintf(stderr, "sepia: metal: wrap_mmap: GPU not initialized\n");
        return NULL;
    }
    if (!base || len == 0) {
        fprintf(stderr, "sepia: metal: wrap_mmap: invalid base/len\n");
        return NULL;
    }

    const size_t page = (size_t)getpagesize();
    if (((uintptr_t)base % page) != 0) {
        fprintf(stderr, "sepia: metal: wrap_mmap: base %p is not page-aligned (page=%zu)\n",
                base, page);
        return NULL;
    }

    /* Metal requires a page-multiple length for bytesNoCopy. The tail
     * bytes between len and the rounded length belong to the caller's
     * mapping (a whole-page mmap, e.g. resident.bin, always has them) --
     * see sepia_gpu.h's contract comment. */
    size_t rounded_len = sepia_gpu_round_up_page(len, page);

    @autoreleasepool {
        uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
        if ((uint64_t)rounded_len > max_buffer) {
            fprintf(stderr,
                    "sepia: metal: wrap_mmap: length %.2f GB exceeds this device's "
                    "maxBufferLength (%.2f GB)\n",
                    (double)rounded_len / 1e9, (double)max_buffer / 1e9);
            return NULL;
        }

        id<MTLBuffer> mtl_buf = [g_device newBufferWithBytesNoCopy:base
                                                             length:rounded_len
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
        if (!mtl_buf) {
            fprintf(stderr, "sepia: metal: wrap_mmap: newBufferWithBytesNoCopy failed "
                            "(base=%p, len=%zu)\n", base, rounded_len);
            return NULL;
        }

        struct SepiaGpuBuf *b = calloc(1, sizeof(*b));
        if (!b) {
            fprintf(stderr, "sepia: metal: wrap_mmap: out of memory\n");
            return NULL;
        }
        b->buffer = mtl_buf;
        return b;
    }
}

SepiaGpuBuf *sepia_gpu_alloc(size_t len, int gpu_private) {
    if (!g_available || !g_device) {
        fprintf(stderr, "sepia: metal: alloc: GPU not initialized\n");
        return NULL;
    }
    if (len == 0) {
        fprintf(stderr, "sepia: metal: alloc: zero length\n");
        return NULL;
    }

    @autoreleasepool {
        uint64_t max_buffer = (uint64_t)[g_device maxBufferLength];
        if ((uint64_t)len > max_buffer) {
            fprintf(stderr,
                    "sepia: metal: alloc: length %.2f GB exceeds this device's "
                    "maxBufferLength (%.2f GB)\n",
                    (double)len / 1e9, (double)max_buffer / 1e9);
            return NULL;
        }

        MTLResourceOptions opts = gpu_private ? MTLResourceStorageModePrivate
                                               : MTLResourceStorageModeShared;
        id<MTLBuffer> mtl_buf = [g_device newBufferWithLength:len options:opts];
        if (!mtl_buf) {
            fprintf(stderr, "sepia: metal: alloc: newBufferWithLength failed (len=%zu, %s)\n",
                    len, gpu_private ? "private" : "shared");
            return NULL;
        }

        struct SepiaGpuBuf *b = calloc(1, sizeof(*b));
        if (!b) {
            fprintf(stderr, "sepia: metal: alloc: out of memory\n");
            return NULL;
        }
        b->buffer = mtl_buf;
        return b;
    }
}

void sepia_gpu_free(SepiaGpuBuf *b) {
    if (!b) return;
    b->buffer = nil; /* ARC release; the raw free() below can't do this itself */
    free(b);
}

void *sepia_gpu_host_ptr(SepiaGpuBuf *b) {
    if (!b || !b->buffer) return NULL;
    return [b->buffer contents]; /* Metal returns nil here for Private storage */
}

uint64_t sepia_gpu_gpu_addr(SepiaGpuBuf *b) {
    if (!b || !b->buffer) return 0;
    return (uint64_t)[b->buffer gpuAddress];
}

int sepia_gpu_selftest_touch(SepiaGpuBuf *buf, size_t n_floats) {
    if (!g_available || !g_device || !g_queue || !g_library) {
        fprintf(stderr, "sepia: metal: selftest_touch: GPU not initialized\n");
        return 0;
    }
    if (!buf || !buf->buffer || n_floats == 0) {
        fprintf(stderr, "sepia: metal: selftest_touch: invalid buf/n_floats\n");
        return 0;
    }

    @autoreleasepool {
        id<MTLFunction> fn = [g_library newFunctionWithName:@"sepia_touch"];
        if (!fn) {
            fprintf(stderr, "sepia: metal: selftest_touch: sepia_touch function not found\n");
            return 0;
        }

        NSError *error = nil;
        id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn
                                                                                   error:&error];
        if (!pso) {
            fprintf(stderr, "sepia: metal: selftest_touch: pipeline creation failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 0;
        }

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf->buffer offset:0 atIndex:0];

        NSUInteger width = pso.threadExecutionWidth;
        if (width > (NSUInteger)n_floats) width = (NSUInteger)n_floats;
        MTLSize grid = MTLSizeMake((NSUInteger)n_floats, 1, 1);
        MTLSize tg = MTLSizeMake(width, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "sepia: metal: selftest_touch: command buffer failed: %s\n",
                    cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown error");
            return 0;
        }
    }
    return 1;
}
