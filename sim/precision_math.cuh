#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {

// 仅提供通用累加策略（纯头文件，不含 CUDA 核调用语法，避免 MSVC 编译 .cpp 时出现 <<< >>> / __global__ 语法错误）
struct HalfAccum {
    __device__ static inline __half add(__half a, __half b) {
    #if __CUDA_ARCH__ >= 530
        return __hadd(a, b);
    #else
        return __float2half(__half2float(a) + __half2float(b));
    #endif
    }
    __device__ static inline __half addFloat(__half a, float f) {
    #if __CUDA_ARCH__ >= 530
        return __hadd(a, __float2half(f));
    #else
        return __float2half(__half2float(a) + f);
    #endif
    }
    __device__ static inline float toFloat(__half h) { return __half2float(h); }
};

struct MixedAccum {
    __device__ static inline float add(float a, float b) { return a + b; }
    __device__ static inline float addFloat(float a, float f) { return a + f; }
    __device__ static inline float toFloat(float v) { return v; }
};

struct LambdaComputePolicy { bool useHalfCompute; bool forceFp32Accum; };

// CUDA kernel 原型只在 NVCC 下可见，避免 MSVC 语法错误
#ifdef __CUDACC__
__global__ void KPackFloatToHalf(const float* src, __half* dst, uint32_t N);
__global__ void KUnpackHalfToFloat(const __half* src, float* dst, uint32_t N);
#endif

// 主机侧（MSVC 可编译）包装函数声明；实现放在 precision_math.cu
void PackFloatToHalf(const float* src, __half* dst, uint32_t N, cudaStream_t s);
void UnpackHalfToFloat(const __half* src, float* dst, uint32_t N, cudaStream_t s);

} // namespace sim