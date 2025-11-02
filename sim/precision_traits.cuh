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
        uint8_t useHalfRender; // 新增：渲染半精位置
        uint8_t nativeHalfActive; // 新增：原生 half4 主存储激活
        const Half4* d_pos_h4;
        const Half4* d_vel_h4;
        const Half4* d_pos_pred_h4;
        const __half* d_lambda_h; // 新增
        const __half* d_density_h; // 新增
        const __half* d_aux_h; // 新增
        const Half4* d_render_pos_h4; // 新增
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
        // ===== Loads =====
        __device__ static inline float4 loadPos(const float4* d_pos_fp32,
            const Half4* d_pos_h4,
            uint32_t i) {
            if ((g_precisionView.nativeHalfActive || g_precisionView.useHalfPos) && d_pos_h4) {
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
            if ((g_precisionView.nativeHalfActive || g_precisionView.useHalfPosPred) && d_pos_pred_h4) {
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
            if ((g_precisionView.nativeHalfActive || g_precisionView.useHalfVel) && d_vel_h4) {
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
        __device__ static inline float4 loadPosForRender(const float4* d_pos_fp32, const Half4* d_render_h4, uint32_t i){
            if (g_precisionView.useHalfRender && d_render_h4){ Half4 h = d_render_h4[i]; return make_float4(__half2float(h.x),__half2float(h.y),__half2float(h.z),__half2float(h.w)); }
            return d_pos_fp32[i];
        }

        // ===== Stores =====
        __device__ static inline void storePos(float4* d_pos_fp32, Half4* d_pos_h4, uint32_t i, const float3& p) {
            if ((g_precisionView.nativeHalfActive || g_precisionView.useHalfPos) && g_precisionView.d_pos_h4) {
                Half4 h; h.x = __float2half(p.x); h.y = __float2half(p.y); h.z = __float2half(p.z); h.w = __float2half(1.0f); ((Half4*)g_precisionView.d_pos_h4)[i] = h; return;
            }
            float4 v = d_pos_fp32[i]; v.x = p.x; v.y = p.y; v.z = p.z; d_pos_fp32[i] = v;
        }
        __device__ static inline void storePosPred(float4* d_pos_pred_fp32, Half4* d_pos_pred_h4, uint32_t i, const float3& p, float w) {
            if ((g_precisionView.nativeHalfActive || g_precisionView.useHalfPosPred) && g_precisionView.d_pos_pred_h4) {
                Half4 h; h.x = __float2half(p.x); h.y = __float2half(p.y); h.z = __float2half(p.z); h.w = __float2half(w); ((Half4*)g_precisionView.d_pos_pred_h4)[i] = h; return;
            }
            d_pos_pred_fp32[i] = make_float4(p.x, p.y, p.z, w);
        }
        __device__ static inline void storeVel(float4* d_vel_fp32, Half4* d_vel_h4, uint32_t i, const float3& v3) {
            if ((g_precisionView.nativeHalfActive || g_precisionView.useHalfVel) && g_precisionView.d_vel_h4) {
                Half4 h; h.x = __float2half(v3.x); h.y = __float2half(v3.y); h.z = __float2half(v3.z); h.w = __float2half(0.0f); ((Half4*)g_precisionView.d_vel_h4)[i] = h; return;
            }
            float4 dst = d_vel_fp32[i]; dst.x = v3.x; dst.y = v3.y; dst.z = v3.z; d_vel_fp32[i] = dst;
        }
        // Legacy overloads (maintain existing calls)
        __device__ static inline void storeVel(float4* d_vel_fp32, uint32_t i, const float3& v3) {
            storeVel(d_vel_fp32, (Half4*)nullptr, i, v3);
        } 
        __device__ static inline void storePos(float4* d_pos_fp32, uint32_t i, const float3& p) {
            storePos(d_pos_fp32, (Half4*)nullptr, i, p);
        } 
    };

} // namespace sim