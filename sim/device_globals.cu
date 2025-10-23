#include "device_globals.cuh"
#include "device_buffers.cuh"

namespace sim {

    // 定义设备侧符号指针（初始为 nullptr）
    __device__ float4* g_pos = nullptr;
    __device__ float4* g_vel = nullptr;
    __device__ float4* g_pos_pred = nullptr;
    __device__ float4* g_delta = nullptr;
    __device__ float* g_lambda = nullptr;
    // 新增：幽灵粒子计数
    __device__ uint32_t g_ghostCount = 0;

    static void CopyPtrToSymbol(float4* const* hPtr, float4** symbol) {
        // 这里的 symbol 是编译期常量符号地址；cudaMemcpyToSymbol 目的参数必须是符号名，不是运行时值
        // 由于需对不同类型调用，实际分开展示。此函数仅用于统一写法时的概念示例。
    }

    void BindDeviceGlobals(float4* d_pos_curr,
        float4* d_vel_curr,
        float4* d_pos_next,
        float4* d_delta_ptr,
        float* d_lambda_ptr) {
        // 每次执行仅拷贝几个指针 (5 * 8 字节)，极低开销
        cudaMemcpyToSymbol(g_pos, &d_pos_curr, sizeof(float4*));
        cudaMemcpyToSymbol(g_vel, &d_vel_curr, sizeof(float4*));
        cudaMemcpyToSymbol(g_pos_pred, &d_pos_next, sizeof(float4*));
        cudaMemcpyToSymbol(g_delta, &d_delta_ptr, sizeof(float4*));
        cudaMemcpyToSymbol(g_lambda, &d_lambda_ptr, sizeof(float*));
    }

    void BindDeviceGlobalsFrom(const DeviceBuffers& bufs) {
        BindDeviceGlobals(bufs.d_pos_curr,
            bufs.d_vel_curr,
            bufs.d_pos_next,
            bufs.d_delta,
            bufs.d_lambda);
    }

    // 新增：上传幽灵粒子数
    void UploadGhostCount(uint32_t ghostCount) {
        cudaMemcpyToSymbol(g_ghostCount, &ghostCount, sizeof(uint32_t));
    }

} // namespace sim