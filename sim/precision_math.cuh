#pragma once
#include <cuda_fp16.h>

namespace sim {

    struct HalfAccum {
        // 累加器以 __half 形式（无 Kahan）
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
        __device__ static inline float toFloat(__half h) {
            return __half2float(h);
        }
    };

    struct MixedAccum {
        // FP32 累加（对应 forceFp32Accumulate=true）
        __device__ static inline float add(float a, float b) { return a + b; }
        __device__ static inline float addFloat(float a, float f) { return a + f; }
        __device__ static inline float toFloat(float v) { return v; }
    };
 
    struct LambdaComputePolicy {
        bool useHalfCompute;      // 阶段要求 FP16
        bool forceFp32Accum;      // 是否强制 FP32 累加
    };

} // namespace sim