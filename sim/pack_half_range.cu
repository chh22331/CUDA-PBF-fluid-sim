#include "precision_traits.cuh"
#include "device_buffers.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void kPackRange(const float4* __restrict__ src,
    sim::Half4* __restrict__ dst,
    uint32_t begin, uint32_t count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint32_t i = begin + tid;
    float4 v = src[i];
    sim::Half4 h;
    h.x = __float2half(v.x);
    h.y = __float2half(v.y);
    h.z = __float2half(v.z);
    h.w = __float2half(v.w);
    dst[i] = h;
}

static void launchPack(const float4* src, sim::Half4* dst,
    uint32_t begin, uint32_t count, cudaStream_t s) {
    if (!src || !dst || count == 0) return;
    const uint32_t block = 256;
    const uint32_t grid = (count + block - 1) / block;
    kPackRange << <grid, block, 0, s >> > (src, dst, begin, count);
}

extern "C" void LaunchPackRangeToHalfPos(const float4* src, sim::Half4* dst,
    uint32_t begin, uint32_t count, cudaStream_t s) {
    launchPack(src, dst, begin, count, s);
}
extern "C" void LaunchPackRangeToHalfVel(const float4* src, sim::Half4* dst,
    uint32_t begin, uint32_t count, cudaStream_t s) {
    launchPack(src, dst, begin, count, s);
}
extern "C" void LaunchPackRangeToHalfPosPred(const float4* src, sim::Half4* dst,
    uint32_t begin, uint32_t count, cudaStream_t s) {
    launchPack(src, dst, begin, count, s);
}