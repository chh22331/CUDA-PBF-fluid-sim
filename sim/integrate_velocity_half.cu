#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "parameters.h"

using namespace sim;

// Half4 在 device_buffers.cuh 已定义
// 半精算术策略：
//  - forceFp32Accum = true  -> 仍用 float 计算（安全路径）
//  - forceFp32Accum = false -> 使用 __half / __hfma/__hadd 等 (如果架构不支持半精FMA，回退 float 转换)

__device__ inline float4 make_out(float nx, float ny, float nz, float w) {
    return make_float4(nx, ny, nz, w);
}

// ============== Integrate (Half Arithmetic) ============= //
__global__ void KIntegratePredHalf(
    const Half4* __restrict__ pos_h4,
    const Half4* __restrict__ vel_h4,
    float4* __restrict__ pos_pred_out,
    float3 gravity,
    float dt,
    uint32_t N,
    int forceFp32Accum,
    int semiImplicit) // 0=显式;1=半隐式
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Half4 hp = pos_h4[i];
    Half4 hv = vel_h4[i];

    if (forceFp32Accum) {
        float px = __half2float(hp.x);
        float py = __half2float(hp.y);
        float pz = __half2float(hp.z);
        float vx = __half2float(hv.x);
        float vy = __half2float(hv.y);
        float vz = __half2float(hv.z);

        if (semiImplicit) {
            vx += gravity.x * dt;
            vy += gravity.y * dt;
            vz += gravity.z * dt;
            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;
        }
        else {
            px += vx * dt + gravity.x * dt;
            py += vy * dt + gravity.y * dt;
            pz += vz * dt + gravity.z * dt;
        }

        pos_pred_out[i] = make_out(px, py, pz, __half2float(hp.w));
    }
    else {
#if __CUDA_ARCH__ >= 530
        __half hdt = __float2half(dt);
        __half gx = __float2half(gravity.x);
        __half gy = __float2half(gravity.y);
        __half gz = __float2half(gravity.z);

        __half vx = hv.x;
        __half vy = hv.y;
        __half vz = hv.z;

        if (semiImplicit) {
            vx = __hadd(vx, __hmul(gx, hdt));
            vy = __hadd(vy, __hmul(gy, hdt));
            vz = __hadd(vz, __hmul(gz, hdt));
            __half dx = __hmul(vx, hdt);
            __half dy = __hmul(vy, hdt);
            __half dz = __hmul(vz, hdt);
            dx = __hadd(dx, hp.x);
            dy = __hadd(dy, hp.y);
            dz = __hadd(dz, hp.z);
            pos_pred_out[i] = make_out(__half2float(dx), __half2float(dy), __half2float(dz), __half2float(hp.w));
        }
        else {
            __half dx = __hmul(vx, hdt);
            __half dy = __hmul(vy, hdt);
            __half dz = __hmul(vz, hdt);
            dx = __hadd(dx, __hmul(gx, hdt));
            dy = __hadd(dy, __hmul(gy, hdt));
            dz = __hadd(dz, __hmul(gz, hdt));
            dx = __hadd(dx, hp.x);
            dy = __hadd(dy, hp.y);
            dz = __hadd(dz, hp.z);
            pos_pred_out[i] = make_out(__half2float(dx), __half2float(dy), __half2float(dz), __half2float(hp.w));
        }
#else
        float px = __half2float(hp.x);
        float py = __half2float(hp.y);
        float pz = __half2float(hp.z);
        float vx = __half2float(hv.x);
        float vy = __half2float(hv.y);
        float vz = __half2float(hv.z);

        if (semiImplicit) {
            vx += gravity.x * dt;
            vy += gravity.y * dt;
            vz += gravity.z * dt;
            px += vx * dt; py += vy * dt; pz += vz * dt;
        }
        else {
            px += vx * dt + gravity.x * dt;
            py += vy * dt + gravity.y * dt;
            pz += vz * dt + gravity.z * dt;
        }
        pos_pred_out[i] = make_out(px, py, pz, __half2float(hp.w));
#endif
    }
}

// 说明：原文件中还定义了一个 extern "C" LaunchVelocityHalf，与 sim/velocity_half.cu 中改进版同名但签名不同。
// 为消除 LNK2005 重复定义错误，已移除该旧版本实现，请统一使用 sim/velocity_half.cu 中的
// LaunchVelocityHalf(float4*, sim::Half4*, const sim::Half4*, const sim::Half4*, float, uint32_t, bool, cudaStream_t)
// 若需要旧的精简签名，可在其他 TU 中新增一个包装函数调用改进版。

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

extern "C" void LaunchIntegratePredHalf(
    const Half4* pos_h4,
    const Half4* vel_h4,
    float4* pos_pred_out,
    float3 gravity,
    float dt,
    uint32_t N,
    bool forceFp32Accum,
    bool semiImplicit,        // 新增：半隐式开关
    cudaStream_t s)
{
    if (!pos_h4 || !vel_h4 || N == 0) return;
    KIntegratePredHalf<<<gridFor(N), 256, 0, s>>>(
        pos_h4, vel_h4, pos_pred_out,
        gravity, dt, N,
        forceFp32Accum ? 1 : 0,
        semiImplicit ? 1 : 0);
}