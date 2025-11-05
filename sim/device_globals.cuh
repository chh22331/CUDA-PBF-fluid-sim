#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {

    extern __device__ float4* g_pos;
    extern __device__ float4* g_vel;
    extern __device__ float4* g_pos_pred;
    extern __device__ float4* g_delta;
    extern __device__ float*  g_lambda;
    extern __device__ uint32_t g_ghostCount;

    struct Half4;
    extern __device__ Half4* g_pos_h4;
    extern __device__ Half4* g_vel_h4;
    extern __device__ Half4* g_pos_pred_h4;

    struct DeviceBuffers;
    void BindDeviceGlobals(float4* d_pos_curr,
                           float4* d_vel_ptr,
                           float4* d_pos_next,
                           float4* d_delta_ptr,
                           float*  d_lambda_ptr);

    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs);
    void UploadGhostCount(uint32_t ghostCount);

    // 新增：仅重新绑定位置（内部 ping-pong 简化用，不改 vel/lambda）
    inline void RebindPositionGlobals(float4* d_pos_curr, float4* d_pos_next, Half4* h_curr, Half4* h_next) {
        cudaMemcpyToSymbol(g_pos,      &d_pos_curr, sizeof(float4*));
        cudaMemcpyToSymbol(g_pos_pred, &d_pos_next, sizeof(float4*));
        if (h_curr && h_next) {
            cudaMemcpyToSymbol(g_pos_h4,      &h_curr, sizeof(Half4*));
            cudaMemcpyToSymbol(g_pos_pred_h4, &h_next, sizeof(Half4*));
        } else {
            Half4* nullH = nullptr;
            cudaMemcpyToSymbol(g_pos_h4,      &nullH, sizeof(Half4*));
            cudaMemcpyToSymbol(g_pos_pred_h4, &nullH, sizeof(Half4*));
        }
    }

} // namespace sim