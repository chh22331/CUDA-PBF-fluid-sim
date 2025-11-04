#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {

    // 全局设备侧指针（仅存放地址，不存放整块数组）
    // 注意：这些不是 constant memory，仅普通全局，读取不会隐式广播。
    extern __device__ float4* g_pos;
    extern __device__ float4* g_vel;
    extern __device__ float4* g_pos_pred;
    extern __device__ float4* g_delta;
    extern __device__ float* g_lambda;
    extern __device__ uint32_t g_ghostCount; // 新增：幽灵粒子计数（用于核内跳过积分等）

    // 新增：原生 half 主存储指针（在 nativeHalfActive 时使用）
    struct Half4; // 前置声明
    extern __device__ Half4* g_pos_h4;
    extern __device__ Half4* g_vel_h4;
    extern __device__ Half4* g_pos_pred_h4;

    // 绑定（Host 调用）：把当前缓冲区指针写入上述符号。
    // 只在 allocate 后或 ping-pong swap 后调用一次即可。
    void BindDeviceGlobals(float4* d_pos_curr,
        float4* d_vel_curr,
        float4* d_pos_next,
        float4* d_delta,
        float* d_lambda);

    // 简化辅助：从 DeviceBuffers 直接绑定
    struct DeviceBuffers; // 前置声明
    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs);

    // 新增：绑定幽灵粒子数
    void UploadGhostCount(uint32_t ghostCount);

} // namespace sim