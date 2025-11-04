#include "device_globals.cuh"
#include "device_buffers.cuh"

namespace sim {
    __device__ float4* g_pos = nullptr;
    __device__ float4* g_vel = nullptr;
    __device__ float4* g_pos_pred = nullptr;
    __device__ float4* g_delta = nullptr;
    __device__ float* g_lambda = nullptr;
    __device__ uint32_t g_ghostCount =0;
    __device__ Half4* g_pos_h4 = nullptr;
    __device__ Half4* g_vel_h4 = nullptr;
    __device__ Half4* g_pos_pred_h4 = nullptr;

    void BindDeviceGlobals(float4* d_pos_curr,
        float4* d_vel_ptr,
        float4* d_pos_next,
        float4* d_delta_ptr,
        float* d_lambda_ptr) {
        cudaMemcpyToSymbol(g_pos, &d_pos_curr, sizeof(float4*));
        cudaMemcpyToSymbol(g_vel, &d_vel_ptr, sizeof(float4*));
        cudaMemcpyToSymbol(g_pos_pred, &d_pos_next, sizeof(float4*));
        cudaMemcpyToSymbol(g_delta, &d_delta_ptr, sizeof(float4*));
        cudaMemcpyToSymbol(g_lambda, &d_lambda_ptr, sizeof(float*));
    }

    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs) {
        BindDeviceGlobals(bufs.d_pos_curr, bufs.d_vel, bufs.d_pos_next, bufs.d_delta, bufs.d_lambda);
        if (bufs.usePosHalf && bufs.d_pos_curr_h4 && bufs.d_pos_next_h4) {
            cudaMemcpyToSymbol(g_pos_h4, &bufs.d_pos_curr_h4, sizeof(Half4*));
            cudaMemcpyToSymbol(g_pos_pred_h4, &bufs.d_pos_next_h4, sizeof(Half4*));
        } else {
            Half4* nullH = nullptr;
            cudaMemcpyToSymbol(g_pos_h4, &nullH, sizeof(Half4*));
            cudaMemcpyToSymbol(g_pos_pred_h4, &nullH, sizeof(Half4*));
        }
        if (bufs.useVelHalf && bufs.d_vel_h4) {
            cudaMemcpyToSymbol(g_vel_h4, &bufs.d_vel_h4, sizeof(Half4*));
        } else {
            Half4* nullH = nullptr;
            cudaMemcpyToSymbol(g_vel_h4, &nullH, sizeof(Half4*));
        }
    }

    void UploadGhostCount(uint32_t ghostCount) {
        uint32_t h = ghostCount; //通过临时变量确保地址类型匹配
        cudaMemcpyToSymbol(g_ghostCount, &h, sizeof(uint32_t));
    }

} // namespace sim