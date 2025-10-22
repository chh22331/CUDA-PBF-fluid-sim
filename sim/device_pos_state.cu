#include "device_pos_state.cuh"

namespace sim {

// 真正的常量符号定义（仅此一处）
__device__ __constant__ SimPosDeviceTable g_simPosConst;

// Host 侧更新常量表
void UploadSimPosTableConst(const float4* curr, const float4* next) {
    SimPosDeviceTable h{ const_cast<float4*>(curr), const_cast<float4*>(next) };
    cudaMemcpyToSymbol(g_simPosConst, &h, sizeof(h));
}

// 设备端占位 kernel（目前不做写操作，只用于潜在的同步/标记）
__global__ void SimPosSwapKernelConst(float4* newCurr, float4* newNext) {
    (void)newCurr; (void)newNext;
}

} // namespace sim