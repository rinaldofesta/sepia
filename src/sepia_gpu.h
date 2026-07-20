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
#endif
