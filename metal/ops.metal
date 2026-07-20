// metal/ops.metal -- Task 1 scaffolding placeholder.
//
// A single trivial kernel so the runtime-compiled library (sepia_gpu_init,
// src/sepia_metal.m) has something non-empty to build a pipeline from.
// Real ops (rmsnorm, matvec, softmax, sconv, ...) arrive in Task 3+.
#include <metal_stdlib>
using namespace metal;

kernel void sepia_touch(device float *x [[buffer(0)]],
                         uint i [[thread_position_in_grid]]) {
    x[i] += 0.0f;
}
