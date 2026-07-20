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

#include <stdio.h>
#include <string.h>

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
