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
#include "quants.h"

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

/* Batched encoding state (Task 3) -- see the sepia_gpu_begin/flush/end
 * definitions below for the full contract. Declared here (rather than next
 * to those functions) so sepia_gpu_shutdown, above, can reset them too.
 *
 * Thread-safety: this entire module is single-thread-only -- g_device,
 * g_queue, g_library, g_batch_cb, g_batch_enc, and g_pso_cache are plain
 * globals with no locking, and every sepia_gpu_* entry point assumes it is
 * called from the one thread that owns them. Task 11's planned loader
 * pthread must never touch any of this state or call into this file: it
 * only preads expert bytes into host memory and signals a shared event --
 * all Metal API calls (encoding, PSO lookups, buffer/queue access) stay on
 * the single thread that already does so today. */
static id<MTLCommandBuffer> g_batch_cb;
static id<MTLComputeCommandEncoder> g_batch_enc;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pso_cache;

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
    g_batch_cb = nil;
    g_batch_enc = nil;
    g_pso_cache = nil;
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

/* ------------------------------ batched encoding --------------------------- */
/* g_batch_cb/g_batch_enc mirror ds4's g_batch_cb/g_batch_enc (ds4_metal.m:47-
 * 48): one open command buffer + one open compute encoder that dispatch
 * calls append to, until flush (commit, no wait, reopen) or end (commit,
 * wait, don't reopen). g_pso_cache avoids rebuilding a MTLComputePipelineState
 * per dispatch call -- the comparison harness calls the same handful of
 * kernels hundreds of times over a captured instance list. (All three are
 * declared at the top of the file, with the other globals, so
 * sepia_gpu_shutdown can reset them.) */

static id<MTLComputeCommandEncoder> sepia_gpu_encoder(void) {
    if (!g_batch_cb) return nil;
    if (!g_batch_enc) g_batch_enc = [g_batch_cb computeCommandEncoder];
    return g_batch_enc;
}

static id<MTLComputePipelineState> sepia_gpu_pso(NSString *name) {
    if (!g_pso_cache) g_pso_cache = [NSMutableDictionary new];
    id<MTLComputePipelineState> pso = g_pso_cache[name];
    if (pso) return pso;

    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "sepia: metal: kernel function \"%s\" not found\n", name.UTF8String);
        return nil;
    }
    NSError *error = nil;
    pso = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
        fprintf(stderr, "sepia: metal: pipeline creation failed for \"%s\": %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    g_pso_cache[name] = pso;
    return pso;
}

/* Smallest threadgroup width (a power of two, multiple of the 32-wide SIMD
 * group, capped at 1024) that lets a strided i+=ntg.x loop cover n elements
 * without excessive over-subscription for the row-reduction kernels
 * (rmsnorm/matvec/softmax). */
static NSUInteger sepia_gpu_reduce_width(int64_t n) {
    NSUInteger w = 32;
    while (w < (NSUInteger)n && w < 1024) w *= 2;
    return w;
}

int sepia_gpu_begin(void) {
    if (!g_available || !g_queue) {
        fprintf(stderr, "sepia: metal: begin: GPU not initialized\n");
        return 0;
    }
    if (g_batch_cb) {
        fprintf(stderr, "sepia: metal: begin: a batch is already active\n");
        return 0;
    }
    @autoreleasepool {
        g_batch_cb = [g_queue commandBuffer];
    }
    if (!g_batch_cb) {
        fprintf(stderr, "sepia: metal: begin: failed to create a command buffer\n");
        return 0;
    }
    return 1;
}

int sepia_gpu_flush(void) {
    if (!g_batch_cb) {
        fprintf(stderr, "sepia: metal: flush: no active batch\n");
        return 0;
    }
    @autoreleasepool {
        if (g_batch_enc) {
            [g_batch_enc endEncoding];
            g_batch_enc = nil;
        }
        [g_batch_cb commit];
        g_batch_cb = [g_queue commandBuffer];
    }
    if (!g_batch_cb) {
        fprintf(stderr, "sepia: metal: flush: failed to open the next batch\n");
        return 0;
    }
    return 1;
}

int sepia_gpu_end(void) {
    if (!g_batch_cb) {
        fprintf(stderr, "sepia: metal: end: no active batch\n");
        return 0;
    }
    id<MTLCommandBuffer> cb;
    @autoreleasepool {
        if (g_batch_enc) {
            [g_batch_enc endEncoding];
            g_batch_enc = nil;
        }
        cb = g_batch_cb;
        g_batch_cb = nil;
        [cb commit];
    }
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "sepia: metal: end: command buffer failed: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown error");
        return 0;
    }
    return 1;
}

/* --------------------------------- op args ---------------------------- */
/* Mirror the `constant` structs in metal/ops.metal field-for-field (same
 * types, same order -- ds4's ne/nb ABI convention, ds4_metal.m:3108-3204,
 * scaled down to what Task 3's contiguous f32 ops actually need: no
 * per-dim nb0..nb3 generality yet, since every buffer here is a plain
 * row-major [rows,n] or [out,in] layout. Task 4+'s strided/interleaved
 * quantized layouts are expected to grow these, not replace them). */
typedef struct {
    int32_t ne0;
    float   eps;
} sepia_args_rmsnorm;

typedef struct {
    int32_t  ne0;
    uint64_t nb1;
} sepia_args_matvec;

typedef struct {
    int32_t ne0;
} sepia_args_softmax;

typedef struct {
    int32_t C;
    int32_t K;
    int32_t T;
} sepia_args_sconv;

/* --------------------------------- ops ---------------------------------- */

int sepia_gpu_rmsnorm(SepiaGpuBuf *w, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n, float eps) {
    if (!w || !w->buffer || !x || !x->buffer || !y || !y->buffer || rows <= 0 || n <= 0) {
        fprintf(stderr, "sepia: metal: rmsnorm: invalid arguments\n");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: rmsnorm: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(@"sepia_rmsnorm_f32");
        if (!pso) return 0;

        sepia_args_rmsnorm args = { (int32_t)n, eps };
        [enc setComputePipelineState:pso];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:w->buffer offset:0 atIndex:1];
        [enc setBuffer:x->buffer offset:0 atIndex:2];
        [enc setBuffer:y->buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];

        NSUInteger tw = sepia_gpu_reduce_width(n);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    }
    return 1;
}

int sepia_gpu_matvec(SepiaGpuBuf *w, SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t out_dim, int64_t in_dim) {
    if (!w || !w->buffer || !x || !x->buffer || !y || !y->buffer || out_dim <= 0 || in_dim <= 0) {
        fprintf(stderr, "sepia: metal: matvec: invalid arguments\n");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: matvec: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(@"sepia_matvec_f32");
        if (!pso) return 0;

        sepia_args_matvec args = { (int32_t)in_dim, (uint64_t)(in_dim * (int64_t)sizeof(float)) };
        [enc setComputePipelineState:pso];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:w->buffer offset:0 atIndex:1];
        [enc setBuffer:x->buffer offset:0 atIndex:2];
        [enc setBuffer:y->buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];

        NSUInteger tw = sepia_gpu_reduce_width(in_dim);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)out_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    }
    return 1;
}

int sepia_gpu_silu_mul(SepiaGpuBuf *g, SepiaGpuBuf *u, SepiaGpuBuf *z, int64_t n) {
    if (!g || !g->buffer || !u || !u->buffer || !z || !z->buffer || n <= 0) {
        fprintf(stderr, "sepia: metal: silu_mul: invalid arguments\n");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: silu_mul: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(@"sepia_silu_mul_f32");
        if (!pso) return 0;

        [enc setComputePipelineState:pso];
        [enc setBuffer:g->buffer offset:0 atIndex:0];
        [enc setBuffer:u->buffer offset:0 atIndex:1];
        [enc setBuffer:z->buffer offset:0 atIndex:2];

        NSUInteger width = pso.threadExecutionWidth;
        if (width > (NSUInteger)n) width = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    return 1;
}

int sepia_gpu_add(SepiaGpuBuf *a, SepiaGpuBuf *b, SepiaGpuBuf *out, int64_t n) {
    if (!a || !a->buffer || !b || !b->buffer || !out || !out->buffer || n <= 0) {
        fprintf(stderr, "sepia: metal: add: invalid arguments\n");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: add: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(@"sepia_add_f32");
        if (!pso) return 0;

        [enc setComputePipelineState:pso];
        [enc setBuffer:a->buffer offset:0 atIndex:0];
        [enc setBuffer:b->buffer offset:0 atIndex:1];
        [enc setBuffer:out->buffer offset:0 atIndex:2];

        NSUInteger width = pso.threadExecutionWidth;
        if (width > (NSUInteger)n) width = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    return 1;
}

int sepia_gpu_softmax(SepiaGpuBuf *x, SepiaGpuBuf *y, int64_t rows, int64_t n) {
    if (!x || !x->buffer || !y || !y->buffer || rows <= 0 || n <= 0) {
        fprintf(stderr, "sepia: metal: softmax: invalid arguments\n");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: softmax: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(@"sepia_softmax_f32");
        if (!pso) return 0;

        sepia_args_softmax args = { (int32_t)n };
        [enc setComputePipelineState:pso];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:x->buffer offset:0 atIndex:1];
        [enc setBuffer:y->buffer offset:0 atIndex:2];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];

        NSUInteger tw = sepia_gpu_reduce_width(n);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    }
    return 1;
}

int sepia_gpu_sconv(SepiaGpuBuf *w, SepiaGpuBuf *hist, SepiaGpuBuf *in, SepiaGpuBuf *out,
                     int64_t C, int64_t K, int64_t T) {
    if (!w || !w->buffer || !hist || !hist->buffer || !in || !in->buffer || !out || !out->buffer ||
        C <= 0 || K <= 0 || T <= 0) {
        fprintf(stderr, "sepia: metal: sconv: invalid arguments\n");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: sconv: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(@"sepia_sconv_f32");
        if (!pso) return 0;

        sepia_args_sconv args = { (int32_t)C, (int32_t)K, (int32_t)T };
        [enc setComputePipelineState:pso];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:w->buffer offset:0 atIndex:1];
        [enc setBuffer:hist->buffer offset:0 atIndex:2];
        [enc setBuffer:in->buffer offset:0 atIndex:3];
        [enc setBuffer:out->buffer offset:0 atIndex:4];

        NSUInteger tgx = MIN((NSUInteger)C, (NSUInteger)32);
        NSUInteger tgy = MIN((NSUInteger)T, (NSUInteger)32);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)C, (NSUInteger)T, 1)
             threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
    }
    return 1;
}

/* --------------------------- quantized matvec (Task 4) --------------------- */
/* Mirrors sepia_args_matvec_q in metal/matvec_q.metal field-for-field: ne0
 * is in_dim, nb1 is the row stride in bytes -- same ne/nb ABI shape as
 * Task 3's sepia_args_matvec, just for quantized rows whose stride is
 * quant-block-derived rather than sizeof(float)*ne0. */
typedef struct {
    int32_t  ne0;
    uint64_t nb1;
} sepia_args_matvec_q;

int sepia_gpu_matvec_q(int ggml_type, SepiaGpuBuf *w, size_t w_off, SepiaGpuBuf *x, SepiaGpuBuf *y,
                        int64_t out_dim, int64_t in_dim) {
    if (!w || !w->buffer || !x || !x->buffer || !y || !y->buffer || out_dim <= 0 || in_dim <= 0) {
        fprintf(stderr, "sepia: metal: matvec_q: invalid arguments\n");
        return 0;
    }

    NSString *pso_name;
    int64_t block;
    size_t block_bytes;
    switch (ggml_type) {
    case SEPIA_T_Q8_0: pso_name = @"sepia_matvec_q8_0"; block = 32;  block_bytes = 34;  break;
    case SEPIA_T_Q4_K: pso_name = @"sepia_matvec_q4_k"; block = 256; block_bytes = 144; break;
    case SEPIA_T_Q5_K: pso_name = @"sepia_matvec_q5_k"; block = 256; block_bytes = 176; break;
    case SEPIA_T_Q6_K: pso_name = @"sepia_matvec_q6_k"; block = 256; block_bytes = 210; break;
    default:
        fprintf(stderr, "sepia: metal: matvec_q: no GPU kernel for ggml type %d\n", ggml_type);
        return 0;
    }
    if (in_dim % block != 0) {
        fprintf(stderr,
                "sepia: metal: matvec_q: in_dim %lld not a multiple of block size %lld (type %d)\n",
                (long long)in_dim, (long long)block, ggml_type);
        return 0;
    }

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: matvec_q: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(pso_name);
        if (!pso) return 0;

        uint64_t nb1 = (uint64_t)(in_dim / block) * (uint64_t)block_bytes;
        sepia_args_matvec_q args = { (int32_t)in_dim, nb1 };
        [enc setComputePipelineState:pso];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:w->buffer offset:w_off atIndex:1];
        [enc setBuffer:x->buffer offset:0 atIndex:2];
        [enc setBuffer:y->buffer offset:0 atIndex:3];
        [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];

        NSUInteger tw = sepia_gpu_reduce_width(in_dim / block);
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)out_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    }
    return 1;
}

int sepia_gpu_dequant_rows(int ggml_type, SepiaGpuBuf *raw, size_t raw_off, SepiaGpuBuf *out, int64_t n) {
    if (!raw || !raw->buffer || !out || !out->buffer || n <= 0) {
        fprintf(stderr, "sepia: metal: dequant_rows: invalid arguments\n");
        return 0;
    }

    NSString *pso_name;
    int64_t block;
    switch (ggml_type) {
    case SEPIA_T_Q8_0: pso_name = @"sepia_dequant_rows_q8_0"; block = 32;  break;
    case SEPIA_T_Q4_K: pso_name = @"sepia_dequant_rows_q4_k"; block = 256; break;
    case SEPIA_T_Q5_K: pso_name = @"sepia_dequant_rows_q5_k"; block = 256; break;
    case SEPIA_T_Q6_K: pso_name = @"sepia_dequant_rows_q6_k"; block = 256; break;
    default:
        fprintf(stderr, "sepia: metal: dequant_rows: no GPU kernel for ggml type %d\n", ggml_type);
        return 0;
    }
    if (n % block != 0) {
        fprintf(stderr, "sepia: metal: dequant_rows: n %lld not a multiple of block size %lld (type %d)\n",
                (long long)n, (long long)block, ggml_type);
        return 0;
    }

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = sepia_gpu_encoder();
        if (!enc) {
            fprintf(stderr, "sepia: metal: dequant_rows: no active batch (call sepia_gpu_begin first)\n");
            return 0;
        }
        id<MTLComputePipelineState> pso = sepia_gpu_pso(pso_name);
        if (!pso) return 0;

        [enc setComputePipelineState:pso];
        [enc setBuffer:raw->buffer offset:raw_off atIndex:0];
        [enc setBuffer:out->buffer offset:0 atIndex:1];

        int64_t nb = n / block;
        NSUInteger width = pso.threadExecutionWidth;
        if (width > (NSUInteger)nb) width = (NSUInteger)nb;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)nb, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    return 1;
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
