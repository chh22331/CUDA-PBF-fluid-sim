#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "parameters.h"
#include "device_buffers.cuh"
#include "precision_stage.h"

using namespace sim;

// Half 算术 Velocity 更新：v = (pos_pred - pos_curr) * inv_dt
// 支持 forceFp32Accumulate 与可配置 guardMax 钳制（来自 debug.velocityNaNGuardMax）。
// 若提供 vel_out_h4 非空，则同步写半精镜像，避免后续再 pack。

namespace {
__global__ void KVelocityHalf(
 float4* __restrict__ vel_out,
 Half4* __restrict__ vel_out_h4,
 const Half4* __restrict__ pos_curr_h4,
 const Half4* __restrict__ pos_pred_h4,
 float inv_dt,
 uint32_t N,
 int forceFp32,
 float guardMax)
{
 uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= N) return;

 Half4 pc = pos_curr_h4[i];
 Half4 pp = pos_pred_h4[i];

 float vx, vy, vz;

#if __CUDA_ARCH__ >=530
 if (!forceFp32) {
 __half2 c01 = __halves2half2(pc.x, pc.y);
 __half2 p01 = __halves2half2(pp.x, pp.y);
 __half2 c2w = __halves2half2(pc.z, pc.w);
 __half2 p2w = __halves2half2(pp.z, pp.w);
 __half2 inv2 = __halves2half2(__float2half(inv_dt), __float2half(inv_dt));
 __half2 d01 = __hmul2(__hsub2(p01, c01), inv2);
 __half2 d2w = __hmul2(__hsub2(p2w, c2w), inv2);
 vx = __half2float(__low2half(d01));
 vy = __half2float(__high2half(d01));
 vz = __half2float(__low2half(d2w));
 } else
#endif
 {
 float cx = __half2float(pc.x), cy = __half2float(pc.y), cz = __half2float(pc.z);
 float px = __half2float(pp.x), py = __half2float(pp.y), pz = __half2float(pp.z);
 vx = (px - cx) * inv_dt;
 vy = (py - cy) * inv_dt;
 vz = (pz - cz) * inv_dt;
 }

 float maxAbs = fmaxf(fmaxf(fabsf(vx), fabsf(vy)), fabsf(vz));
 if (!(maxAbs <= guardMax) || isnan(vx) || isnan(vy) || isnan(vz)) {
 vx = vy = vz =0.f; // 异常回退
 }
 vel_out[i] = make_float4(vx, vy, vz,0.f);

 if (vel_out_h4) {
 vel_out_h4[i].x = __float2half(vx);
 vel_out_h4[i].y = __float2half(vy);
 vel_out_h4[i].z = __float2half(vz);
 vel_out_h4[i].w = __float2half(0.f);
 }
}

inline dim3 GridFor(uint32_t N) { return dim3((N +255u) /256u); }
} // namespace

extern "C" void LaunchVelocityHalf(
 float4* vel_out_fp32,
 sim::Half4* vel_out_h4,
 const sim::Half4* pos_curr_h4,
 const sim::Half4* pos_pred_h4,
 float inv_dt,
 uint32_t N,
 bool forceFp32Accumulate,
 float guardMax,
 cudaStream_t s)
{
 if (!vel_out_fp32 || !pos_curr_h4 || !pos_pred_h4 || N ==0) return;
 dim3 bs(256); dim3 gs = GridFor(N);
 KVelocityHalf<<<gs, bs,0, s>>>(vel_out_fp32, vel_out_h4, (const Half4*)pos_curr_h4, (const Half4*)pos_pred_h4, inv_dt, N, forceFp32Accumulate ?1 :0, guardMax);
}
