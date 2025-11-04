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
            // v(t+dt) = v + g*dt
            vx += gravity.x * dt;
            vy += gravity.y * dt;
            vz += gravity.z * dt;
            // x(t+dt) = x + v(t+dt)*dt
            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;
        }
        else {
            // 显式原逻辑：x += v*dt + g*dt
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
            // v += g*dt
            vx = __hadd(vx, __hmul(gx, hdt));
            vy = __hadd(vy, __hmul(gy, hdt));
            vz = __hadd(vz, __hmul(gz, hdt));
            // x += v*dt
            __half dx = __hmul(vx, hdt);
            __half dy = __hmul(vy, hdt);
            __half dz = __hmul(vz, hdt);
            dx = __hadd(dx, hp.x);
            dy = __hadd(dy, hp.y);
            dz = __hadd(dz, hp.z);
            pos_pred_out[i] = make_out(__half2float(dx), __half2float(dy), __half2float(dz), __half2float(hp.w));
        }
        else {
            // 显式：x += v*dt + g*dt
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

// ============== Velocity (Half Arithmetic) ============= //
// 由 (pos_pred - pos) * inv_dt 推导速度
__global__ void KVelocityHalf(
    float4* __restrict__ vel_out,   // FP32 输出
    const Half4* __restrict__ pos_h4,
    const Half4* __restrict__ pos_pred_h4,
    float inv_dt,
    uint32_t N,
    int forceFp32Accum)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Half4 p0 = pos_h4[i];
    Half4 p1 = pos_pred_h4[i];

    if (forceFp32Accum) {
        float x0 = __half2float(p0.x), y0 = __half2float(p0.y), z0 = __half2float(p0.z);
        float x1 = __half2float(p1.x), y1 = __half2float(p1.y), z1 = __half2float(p1.z);
        float vx = (x1 - x0) * inv_dt;
        float vy = (y1 - y0) * inv_dt;
        float vz = (z1 - z0) * inv_dt;
        float4 v = vel_out[i];
        v.x = vx; v.y = vy; v.z = vz;
        vel_out[i] = v;
    }
    else {
#if __CUDA_ARCH__ >= 530
        __half hinv = __float2half(inv_dt);
        __half dx = __hsub(p1.x, p0.x);
        __half dy = __hsub(p1.y, p0.y);
        __half dz = __hsub(p1.z, p0.z);

        dx = __hmul(dx, hinv);
        dy = __hmul(dy, hinv);
        dz = __hmul(dz, hinv);

        float4 v = vel_out[i];
        v.x = __half2float(dx);
        v.y = __half2float(dy);
        v.z = __half2float(dz);
        vel_out[i] = v;
#else
        float x0 = __half2float(p0.x), y0 = __half2float(p0.y), z0 = __half2float(p0.z);
        float x1 = __half2float(p1.x), y1 = __half2float(p1.y), z1 = __half2float(p1.z);
        float vx = (x1 - x0) * inv_dt;
        float vy = (y1 - y0) * inv_dt;
        float vz = (z1 - z0) * inv_dt;
        float4 v = vel_out[i];
        v.x = vx; v.y = vy; v.z = vz;
        vel_out[i] = v;
#endif
    }
}

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
    KIntegratePredHalf << <gridFor(N), 256, 0, s >> > (
        pos_h4, vel_h4, pos_pred_out,
        gravity, dt, N,
        forceFp32Accum ? 1 : 0,
        semiImplicit ? 1 : 0);
}

extern "C" void LaunchVelocityHalf(
    float4* vel_out,
    const Half4* pos_h4,
    const Half4* pos_pred_h4,
    float inv_dt,
    uint32_t N,
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (!pos_h4 || !pos_pred_h4 || !vel_out || N == 0) return;
    KVelocityHalf << <gridFor(N), 256, 0, s >> > (
        vel_out, pos_h4, pos_pred_h4,
        inv_dt, N,
        forceFp32Accum ? 1 : 0);
}