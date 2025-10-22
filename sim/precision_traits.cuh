#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "parameters.h"
#include "device_buffers.cuh"

namespace sim {

    struct DevicePrecisionView {
        uint8_t useHalfPos;
        uint8_t useHalfVel;
        uint8_t useHalfPosPred;
        uint8_t useHalfGeneric; // 为后续邻居等阶段统一只读入口预留
        uint8_t useHalfLambda; // 新增
        uint8_t useHalfDensity; // 新增
        uint8_t useHalfAux; // 新增
        const Half4* d_pos_h4;
        const Half4* d_vel_h4;
        const Half4* d_pos_pred_h4;
        const __half* d_lambda_h; // 新增
        const __half* d_density_h; // 新增
        const __half* d_aux_h; // 新增
        //预留：后续可加入 lambda/density half 指针
    };

#ifndef SIM_HALF_CONVERT_DEFINED
#define SIM_HALF_CONVERT_DEFINED
    __host__ __device__ inline __half float_to_half(float v) { return __float2half(v); }
    __host__ __device__ inline float  half_to_float(__half h) { return __half2float(h); }
#endif

    // 仅声明，避免多重定义
    extern __device__ __constant__ DevicePrecisionView g_precisionView;

    // 主机更新
    bool UpdateDevicePrecisionView(const DeviceBuffers& bufs, const SimPrecision& pr);

    // 半精聚合策略
    enum class AccumMode : uint8_t {
        ForceFP32 = 0,
        MixedFP16Promote = 1
    };

    struct PrecisionAccum {
        __device__ static inline AccumMode mode(bool forceFp32) {
            return forceFp32 ? AccumMode::ForceFP32 : AccumMode::MixedFP16Promote;
        }
        __device__ static inline float add(float a, float b, AccumMode) {
            return a + b;
        }
        __device__ static inline float fma(float x, float y, float acc, AccumMode) {
#if __CUDA_ARCH__ >= 530
            return __fmaf_rn(x, y, acc);
#else
            return x * y + acc;
#endif
        }
    };

    struct PrecisionTraits {
        __device__ static inline float4 loadPos(const float4* d_pos_fp32,
            const Half4* d_pos_h4,
            uint32_t i) {
            if (g_precisionView.useHalfPos && d_pos_h4) {
                Half4 h = d_pos_h4[i];
                return make_float4(__half2float(h.x),
                    __half2float(h.y),
                    __half2float(h.z),
                    __half2float(h.w));
            }
            return d_pos_fp32[i];
        }
        __device__ static inline float4 loadPosPred(const float4* d_pos_pred_fp32,
            const Half4* d_pos_pred_h4,
            uint32_t i) {
            if (g_precisionView.useHalfPosPred && d_pos_pred_h4) {
                Half4 h = d_pos_pred_h4[i];
                return make_float4(__half2float(h.x),
                    __half2float(h.y),
                    __half2float(h.z),
                    __half2float(h.w));
            }
            return d_pos_pred_fp32[i];
        }
        __device__ static inline float4 loadVel(const float4* d_vel_fp32,
            const Half4* d_vel_h4,
            uint32_t i) {
            if (g_precisionView.useHalfVel && d_vel_h4) {
                Half4 h = d_vel_h4[i];
                return make_float4(__half2float(h.x),
                    __half2float(h.y),
                    __half2float(h.z),
                    __half2float(h.w));
            }
            return d_vel_fp32[i];
        }
        __device__ static inline float loadLambda(const float* d_lambda_fp32,
            const __half* d_lambda_h,
            uint32_t i) {
            if (g_precisionView.useHalfLambda && d_lambda_h) return __half2float(d_lambda_h[i]);
            return d_lambda_fp32[i];
        }
        __device__ static inline float loadDensity(const float* d_density_fp32,
            const __half* d_density_h,
            uint32_t i) {
            if (g_precisionView.useHalfDensity && d_density_h) return __half2float(d_density_h[i]);
            return d_density_fp32[i];
        }
        __device__ static inline float loadAux(const float* d_aux_fp32,
            const __half* d_aux_h,
            uint32_t i) {
            if (g_precisionView.useHalfAux && d_aux_h) return __half2float(d_aux_h[i]);
            return d_aux_fp32[i];
        }
        __device__ static inline void storeVel(float4* d_vel_fp32,
            uint32_t i,
            const float3& v) {
            float4 dst = d_vel_fp32[i];
            dst.x = v.x; dst.y = v.y; dst.z = v.z;
            d_vel_fp32[i] = dst;
        }
    };

} // namespace sim