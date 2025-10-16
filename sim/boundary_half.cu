#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "parameters.h"
#include "device_buffers.cuh"

using namespace sim;

// 半精算术边界：仅对位置修正 + 速度反弹 (rest>0)
// forceFp32 -> 所有计算转 float
__global__ void KBoundaryHalf(
    float4* __restrict__ pos_pred,
    float4* __restrict__ vel,
    const Half4* __restrict__ pos_pred_h4,
    const Half4* __restrict__ vel_h4,
    sim::GridBounds grid,
    float restitution,
    uint32_t N,
    int forceFp32Accum)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p4 = pos_pred[i];
    float4 v4 = vel[i];

#if __CUDA_ARCH__ >= 530
    // 将半精度加载与转换整个放在同一条件编译块，避免 host 侧看到使用却看不到声明导致的编译错误
    if (pos_pred_h4 && vel_h4 && !forceFp32Accum) {
        const Half4 hp = pos_pred_h4[i];
        const Half4 hv = vel_h4[i];
        p4.x = __half2float(hp.x);
        p4.y = __half2float(hp.y);
        p4.z = __half2float(hp.z);
        v4.x = __half2float(hv.x);
        v4.y = __half2float(hv.y);
        v4.z = __half2float(hv.z);
    }
#else
    // 编译为主机或较老架构时忽略半精路径（保持 p4/v4 原始 FP32 数据）
    (void)pos_pred_h4;
    (void)vel_h4;
#endif

    float3 mins = grid.mins;
    float3 maxs = grid.maxs;

    auto clampAxis = [&](float& x, float& vx, float lo, float hi) {
        if (x < lo) {
            x = lo;
            if (restitution > 0.f) vx = -vx * restitution;
            else vx = 0.f;
        }
        else if (x > hi) {
            x = hi;
            if (restitution > 0.f) vx = -vx * restitution;
            else vx = 0.f;
        }
    };

    clampAxis(p4.x, v4.x, mins.x, maxs.x);
    clampAxis(p4.y, v4.y, mins.y, maxs.y);
    clampAxis(p4.z, v4.z, mins.z, maxs.z);

    pos_pred[i] = p4;
    vel[i] = v4;
}

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

extern "C" void LaunchBoundaryHalf(
    float4* pos_pred,
    float4* vel,
    const sim::Half4* pos_pred_h4,
    const sim::Half4* vel_h4,
    sim::GridBounds grid,
    float restitution,
    uint32_t N,
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (N == 0) return;
    KBoundaryHalf<<<gridFor(N), 256, 0, s>>>(
        pos_pred, vel, pos_pred_h4, vel_h4,
        grid, restitution, N, forceFp32Accum ? 1 : 0);
}