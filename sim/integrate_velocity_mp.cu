#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "device_buffers.cuh"
#include "parameters.h"

extern "C" {

    // IntegratePred (Mixed Precision read, FP32 compute)
    __global__ void KIntegratePredMP(const float4* __restrict__ d_pos_fp32,
        const float4* __restrict__ d_vel_fp32,
        float4* __restrict__ d_pos_pred_fp32,
        const sim::Half4* __restrict__ d_pos_h4,
        const sim::Half4* __restrict__ d_vel_h4,
        float3 gravity,
        float dt,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 p  = sim::PrecisionTraits::loadPos(d_pos_fp32, d_pos_h4, i);
        float4 v4 = sim::PrecisionTraits::loadVel(d_vel_fp32, d_vel_h4, i);

        float3 v   = make_float3(v4.x, v4.y, v4.z);
        float3 pos = make_float3(p.x, p.y, p.z);

        pos.x += v.x * dt + gravity.x * dt;
        pos.y += v.y * dt + gravity.y * dt;
        pos.z += v.z * dt + gravity.z * dt;

        d_pos_pred_fp32[i] = make_float4(pos.x, pos.y, pos.z, p.w);
    }

    __global__ void KVelocityMP(float4* __restrict__ d_vel_fp32,
        const float4* __restrict__ d_pos_fp32,
        const float4* __restrict__ d_pos_pred_fp32,
        const sim::Half4* __restrict__ d_pos_h4,
        const sim::Half4* __restrict__ d_pos_pred_h4,
        float inv_dt,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 p0 = sim::PrecisionTraits::loadPos(d_pos_fp32, d_pos_h4, i);
        float4 p1 = sim::PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, i);

        float3 v;
        v.x = (p1.x - p0.x) * inv_dt;
        v.y = (p1.y - p0.y) * inv_dt;
        v.z = (p1.z - p0.z) * inv_dt;

        sim::PrecisionTraits::storeVel(d_vel_fp32, i, v);
    }

    void LaunchIntegratePredMP(const float4* d_pos,
        const float4* d_vel,
        float4* d_pos_pred,
        const sim::Half4* d_pos_h4,
        const sim::Half4* d_vel_h4,
        float3 gravity,
        float dt,
        uint32_t N,
        cudaStream_t s)
    {
        if (N == 0) return;
        uint32_t threads = 256;
        uint32_t blocks  = (N + threads - 1) / threads;
        KIntegratePredMP<<<blocks, threads, 0, s>>>(d_pos, d_vel, d_pos_pred,
            d_pos_h4, d_vel_h4,
            gravity, dt, N);
    }

    void LaunchVelocityMP(float4* d_vel,
        const float4* d_pos,
        const float4* d_pos_pred,
        const sim::Half4* d_pos_h4,
        const sim::Half4* d_pos_pred_h4,
        float inv_dt,
        uint32_t N,
        cudaStream_t s)
    {
        if (N == 0) return;
        uint32_t threads = 256;
        uint32_t blocks  = (N + threads - 1) / threads;
        KVelocityMP<<<blocks, threads, 0, s>>>(d_vel,
            d_pos,
            d_pos_pred,
            d_pos_h4,
            d_pos_pred_h4,
            inv_dt,
            N);
    }

} // extern "C"