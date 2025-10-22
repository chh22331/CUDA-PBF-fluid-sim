#pragma once
#include <cuda_runtime.h>

namespace sim {

// 设备侧统一位置指针表（方案 C）
struct SimPosDeviceTable {
    float4* curr;
    float4* next;
};

// 在头文件仅做“声明”，避免多重定义；真正的定义放到 .cu
extern __device__ __constant__ SimPosDeviceTable g_simPosConst;

// 上传 Host 缓冲指针到设备常量（在 .cu 中实现，避免 MSVC 直接编译含 __constant__ 定义导致问题）
void UploadSimPosTableConst(const float4* curr, const float4* next);

// 设备端交换占位（常量内存仍需 Host 侧刷新）
__global__ void SimPosSwapKernelConst(float4* newCurr, float4* newNext);

} // namespace sim