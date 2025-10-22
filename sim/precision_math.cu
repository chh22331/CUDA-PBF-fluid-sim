#include "precision_math.cuh"

namespace sim {

__global__ void KPackFloatToHalf(const float* src, __half* dst, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void KUnpackHalfToFloat(const __half* src, float* dst, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = __half2float(src[i]);
    }
}

void PackFloatToHalf(const float* src, __half* dst, uint32_t N, cudaStream_t s) {
    if (!src || !dst || N == 0) return;
    uint32_t blocks = (N + 255u) / 256u;
    KPackFloatToHalf<<<blocks, 256, 0, s>>>(src, dst, N);
}

void UnpackHalfToFloat(const __half* src, float* dst, uint32_t N, cudaStream_t s) {
    if (!src || !dst || N == 0) return;
    uint32_t blocks = (N + 255u) / 256u;
    KUnpackHalfToFloat<<<blocks, 256, 0, s>>>(src, dst, N);
}

} // namespace sim