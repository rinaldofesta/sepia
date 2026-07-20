#ifndef SEPIA_QUANTS_H
#define SEPIA_QUANTS_H
#include <stdint.h>
#include <stddef.h>

enum {
    SEPIA_T_F32 = 0, SEPIA_T_F16 = 1, SEPIA_T_Q8_0 = 8,
    SEPIA_T_Q4_K = 12, SEPIA_T_Q5_K = 13, SEPIA_T_Q6_K = 14,
    SEPIA_T_IQ2_XS = 17, SEPIA_T_IQ3_XXS = 18, SEPIA_T_IQ4_XS = 23,
};

int64_t quants_block_size(int ggml_type);
size_t  quants_type_size(int ggml_type);
int     quants_supported(int ggml_type);
size_t  quants_row_bytes(int ggml_type, int64_t ne);
float   quants_f16_to_f32(uint16_t bits);
void    dequantize_row(int ggml_type, const void *src, float *dst, int64_t n);

#endif
