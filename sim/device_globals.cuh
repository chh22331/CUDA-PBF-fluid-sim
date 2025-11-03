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

    // 边界钳制统计（已存在定义）
    extern __device__ uint32_t g_boundaryClampCountFp32;
    extern __device__ uint32_t g_boundaryClampCountHalf;

    // 新增：XSPH 诊断统计
    extern __device__ uint32_t g_xsphNaNCount;
    extern __device__ uint32_t g_xsphAnomalyCount;

    void BindDeviceGlobals(float4* d_pos_curr,
        float4* d_vel_curr,
        float4* d_pos_next,
        float4* d_delta,
        float*  d_lambda);

    struct DeviceBuffers;
    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs);

    void UploadGhostCount(uint32_t ghostCount);

    uint32_t ReadAndResetBoundaryClampCounts(uint32_t* halfCountOut);

    // 新增：读取并复位 XSPH 诊断计数
    uint32_t ReadAndResetXsphDiag(uint32_t* anomalyCountOut);

} // namespace sim