#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include "parameters.h"
#include "device_buffers.cuh"
#include "precision_traits.cuh"
#include "device_globals.cuh"

// 更稳健：局部显式 load，避免依赖 g_precisionView 阶段标志
__device__ inline float4 load_pos_pred_direct(const float4* pos_pred_fp32, const sim::Half4* pos_pred_h4, uint32_t i) {
    if (pos_pred_h4) {
        sim::Half4 h = pos_pred_h4[i];
        return make_float4(__half2float(h.x), __half2float(h.y), __half2float(h.z), __half2float(h.w));
    }
    return pos_pred_fp32[i];
}
__device__ inline float4 load_vel_direct(const float4* vel_fp32, const sim::Half4* vel_h4, uint32_t i) {
    if (vel_h4) {
        sim::Half4 h = vel_h4[i];
        return make_float4(__half2float(h.x), __half2float(h.y), __half2float(h.z), __half2float(h.w));
    }
    return vel_fp32[i];
}

// 与 FP32 版本严格统一：分别处理 min / max（不使用 else-if）
__global__ void KBoundaryHalf(
    float4* __restrict__ pos_pred_fp32,
    float4* __restrict__ vel_fp32,
    sim::Half4* __restrict__ pos_pred_h4,
    sim::Half4* __restrict__ vel_h4,
    sim::GridBounds grid,
    float restitution,
    uint32_t totalCount,
    uint32_t ghostCount,
    uint8_t ghostClampEnable)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCount) return;

    const uint32_t fluidCount = (ghostCount <= totalCount) ? (totalCount - ghostCount) : totalCount;
    const bool isGhost = (i >= fluidCount);
    if (isGhost && !ghostClampEnable) return;

    // 直接加载（不依赖 g_precisionView）
    float4 p4 = load_pos_pred_direct(pos_pred_fp32, pos_pred_h4, i);
    float4 v4 = load_vel_direct(vel_fp32, vel_h4, i);
    float3 p = make_float3(p4.x, p4.y, p4.z);
    float3 v = make_float3(v4.x, v4.y, v4.z);

    const float e = fminf(fmaxf(restitution, 0.0f), 1.0f);
    bool clamped = false;

    // 下边界
    if (p.x < grid.mins.x) { p.x = grid.mins.x; v.x *= -e; clamped = true; }
    if (p.y < grid.mins.y) { p.y = grid.mins.y; v.y *= -e; clamped = true; }
    if (p.z < grid.mins.z) { p.z = grid.mins.z; v.z *= -e; clamped = true; }
    // 上边界
    if (p.x > grid.maxs.x) { p.x = grid.maxs.x; v.x *= -e; clamped = true; }
    if (p.y > grid.maxs.y) { p.y = grid.maxs.y; v.y *= -e; clamped = true; }
    if (p.z > grid.maxs.z) { p.z = grid.maxs.z; v.z *= -e; clamped = true; }

    // 写 pos_pred (双写：half + fp32，保证镜像一致)
    {
        if (pos_pred_h4) {
            sim::Half4 h;
            h.x = __float2half(p.x);
            h.y = __float2half(p.y);
            h.z = __float2half(p.z);
            h.w = __float2half(1.0f);
            pos_pred_h4[i] = h;
        }
        pos_pred_fp32[i] = make_float4(p.x, p.y, p.z, 1.0f);
    }

    // 写 vel (双写，且明确 w=0)
    {
        if (vel_h4) {
            sim::Half4 h;
            h.x = __float2half(v.x);
            h.y = __float2half(v.y);
            h.z = __float2half(v.z);
            h.w = __float2half(0.0f);
            vel_h4[i] = h;
        }
        vel_fp32[i] = make_float4(v.x, v.y, v.z, 0.0f);
    }

    // 可选统计：只统计流体粒子
    // if (clamped && !isGhost) atomicAdd(&sim::g_boundaryClampCountHalf, 1u);
}

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

extern "C" void LaunchBoundaryHalf(
    float4* pos_pred,
    float4* vel,
    const sim::Half4* pos_pred_h4_const,
    const sim::Half4* vel_h4_const,
    sim::GridBounds grid,
    float restitution,
    uint32_t totalCount,
    uint32_t ghostCount,
    bool /*forceFp32Accum*/,      // 保留，占位
    uint8_t ghostClampEnable,
    cudaStream_t s)
{
    if (totalCount == 0) return;
    // 去 const：我们需要写入镜像
    sim::Half4* pos_h4 = const_cast<sim::Half4*>(pos_pred_h4_const);
    sim::Half4* v_h4   = const_cast<sim::Half4*>(vel_h4_const);

    KBoundaryHalf<<<gridFor(totalCount), 256, 0, s>>>(
        pos_pred,
        vel,
        pos_h4,
        v_h4,
        grid,
        restitution,
        totalCount,
        ghostCount,
        ghostClampEnable);
}