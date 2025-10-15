#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_buffers.cuh"

namespace sim {

    __global__ void kPackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float4 v = src[i];
        dst[i].x = __float2half(v.x);
        dst[i].y = __float2half(v.y);
        dst[i].z = __float2half(v.z);
        dst[i].w = __float2half(v.w);
    }

    __global__ void kUnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        Half4 h = src[i];
        dst[i] = make_float4(
            __half2float(h.x),
            __half2float(h.y),
            __half2float(h.z),
            __half2float(h.w));
    }

    void PackFloat4ToHalf4(const float4* src, Half4* dst, uint32_t N, cudaStream_t s) {
        if (!src || !dst || N == 0) return;
        dim3 t(256);
        dim3 b((N + t.x - 1u) / t.x);
        kPackFloat4ToHalf4 << <b, t, 0, s >> > (src, dst, N);
    }

    void UnpackHalf4ToFloat4(const Half4* src, float4* dst, uint32_t N, cudaStream_t s) {
        if (!src || !dst || N == 0) return;
        dim3 t(256);
        dim3 b((N + t.x - 1u) / t.x);
        kUnpackHalf4ToFloat4 << <b, t, 0, s >> > (src, dst, N);
    }

} // namespace sim