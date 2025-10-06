#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "emit_params.h"

// 补齐 float3 叉积
__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

// 设备侧哈希与随机数（避免设备 lambda 依赖）
__host__ __device__ inline uint32_t wanghash(uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2d;
    x = x ^ (x >> 15);
    return x;
}
__host__ __device__ inline float rand01(uint32_t s) {
    return float(wanghash(s) & 0x00FFFFFFu) / float(0x01000000);
}

// 设备常量 & 主机侧缓存（用于参数更新）
__device__ __constant__ sim::EmitParams gEmitParamsConst;
static sim::EmitParams gEmitParamsHost{};

// 旧核函数：按参数逐一传入
extern "C" __global__
void RecycleToNozzleKernel(float4* pos, float4* pos_pred, float4* vel,
    sim::GridBounds grid, float3 nozzlePos, float3 nozzleDir,
    float nozzleRadius, float initSpeed, float dt, float recycleY,
    uint32_t N)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; // 越界保护

    // 读取
    float4 Pp = pos_pred[i];
    if (Pp.y > recycleY) return;

    // 构建与 nozzleDir 正交基
    float3 z = normalize(nozzleDir);
    float3 tmp = fabsf(z.x) > 0.5f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 x = normalize(cross(tmp, z));
    float3 y = cross(z, x);

    // 圆盘均匀采样
    float u = rand01(i * 2u + 1u);
    float v = rand01(i * 2u + 2u);
    float r = nozzleRadius * sqrtf(u);
    float a = 6.2831853f * v;
    float2 d = make_float2(r * cosf(a), r * sinf(a));

    float3 newPos = nozzlePos + x * d.x + y * d.y;
    float3 vinit = z * initSpeed;

    // 设置 pos、pos_pred 与 vel
    pos[i] = make_float4(newPos.x, newPos.y, newPos.z, 1.0f);
    pos_pred[i] = make_float4(newPos.x + vinit.x * dt,
        newPos.y + vinit.y * dt,
        newPos.z + vinit.z * dt, 1.0f);
    vel[i] = make_float4(vinit.x, vinit.y, vinit.z, 0.0f);
}

// 新核函数：多了 enabled 参数，便于“禁用但仍捕获到图节点”
extern "C" __global__
void RecycleToNozzleKernelConst(float4* pos, float4* pos_pred, float4* vel,
    sim::GridBounds grid, float3 nozzlePos, float3 nozzleDir,
    float nozzleRadius, float initSpeed, float dt, float recycleY,
    uint32_t N, int enabled)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (!enabled) return;

    // 读取
    float4 Pp = pos_pred[i];
    if (Pp.y > recycleY) return;

    // 构建与 nozzleDir 正交基
    float3 z = normalize(nozzleDir);
    float3 tmp = fabsf(z.x) > 0.5f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 x = normalize(cross(tmp, z));
    float3 y = cross(z, x);

    // 圆盘均匀采样
    float u = rand01(i * 2u + 1u);
    float v = rand01(i * 2u + 2u);
    float r = nozzleRadius * sqrtf(u);
    float a = 6.2831853f * v;
    float2 d = make_float2(r * cosf(a), r * sinf(a));

    float3 newPos = nozzlePos + x * d.x + y * d.y;
    float3 vinit = z * initSpeed;

    // 设置 pos、pos_pred 与 vel
    pos[i] = make_float4(newPos.x, newPos.y, newPos.z, 1.0f);
    pos_pred[i] = make_float4(newPos.x + vinit.x * dt,
        newPos.y + vinit.y * dt,
        newPos.z + vinit.z * dt, 1.0f);
    vel[i] = make_float4(vinit.x, vinit.y, vinit.z, 0.0f);
}

// 旧的 host 封装（未被当前工程使用，保留）
extern "C" void LaunchRecycleToNozzle(float4* pos, float4* pos_pred, float4* vel,
    sim::GridBounds grid, float3 nozzlePos, float3 nozzleDir,
    float nozzleRadius, float initSpeed, float dt, float recycleY,
    uint32_t N, cudaStream_t s)
{
    dim3 bs(256), gs((N + bs.x - 1) / bs.x);
    RecycleToNozzleKernel << <gs, bs, 0, s >> > (pos, pos_pred, vel, grid, nozzlePos, nozzleDir,
        nozzleRadius, initSpeed, dt, recycleY, N);
}

// 新的 host 封装：使用主机侧缓存的喷口参数，且总会提交一次核调用（enabled 设备侧早退）
extern "C" void LaunchRecycleToNozzleConst(float4* pos, float4* pos_pred, float4* vel,
    sim::GridBounds grid, float dt, uint32_t N, int enabled, cudaStream_t s)
{
    const sim::EmitParams ep = gEmitParamsHost; // 捕获当前主机缓存
    dim3 bs(256), gs((N + bs.x - 1) / bs.x);
    RecycleToNozzleKernelConst << <gs, bs, 0, s >> > (
        pos, pos_pred, vel,
        grid,
        ep.nozzlePos, ep.nozzleDir, ep.nozzleRadius, ep.nozzleSpeed,
        dt, ep.recycleY, N, enabled
        );
}

// 设备常量更新 + 主机侧缓存
extern "C" void SetEmitParamsAsync(const sim::EmitParams* h, cudaStream_t s)
{
    if (!h) return;
    gEmitParamsHost = *h;
    cudaMemcpyToSymbolAsync(gEmitParamsConst, h, sizeof(sim::EmitParams), 0, cudaMemcpyHostToDevice, s);
}

// 向外导出用于 Graph 节点匹配/更新的核函数地址
extern "C" void* GetRecycleKernelPtr()
{
    // 必须与上面 LaunchRecycleToNozzleConst 中实际提交到 Graph 的核一致
    return (void*)&RecycleToNozzleKernelConst;
}