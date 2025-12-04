#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {

    // Device-side global pointers used by CUDA kernels.
    extern __device__ float4* g_pos;        // current position buffer (float4)
    extern __device__ float4* g_vel;        // current velocity buffer (float4)
    extern __device__ float4* g_pos_pred;   // predicted/next position buffer (float4)
    extern __device__ float4* g_delta;      // position correction delta (float4)
    extern __device__ float*  g_lambda;     // constraint lambda scalars

    // Extra state
    extern __device__ uint32_t g_ghostCount; // number of ghost particles (device-side)

    // Optional half-precision aliases (defined elsewhere).
    struct Half4;
    extern __device__ Half4* g_pos_h4;
    extern __device__ Half4* g_vel_h4;
    extern __device__ Half4* g_pos_pred_h4;

    // Forward declaration: host-side buffer aggregator (defined in device_buffers.cuh).
    struct DeviceBuffers;

    // Bind basic device globals from raw pointers.
    // Copies host-side pointer values into device-side symbols with cudaMemcpyToSymbol.
    void BindDeviceGlobals(float4* d_pos_curr,
                           float4* d_vel_ptr,
                           float4* d_pos_next,
                           float4* d_delta_ptr,
                           float*  d_lambda_ptr);

    // Bind device globals from a DeviceBuffers instance (preferred entry point).
    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs);

    // Upload ghost count to device symbol.
    void UploadGhostCount(uint32_t ghostCount);

    // Rebind only position globals (and optional Half4 variants).
    // This is useful for internal ping-pong swaps without touching vel/lambda/delta.
    inline void RebindPositionGlobals(float4* d_pos_curr,
                                      float4* d_pos_next,
                                      Half4*  h_curr,
                                      Half4*  h_next) {
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