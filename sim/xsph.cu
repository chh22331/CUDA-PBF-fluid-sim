#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"
#include "device_globals.cuh" // 保留：如后续再启用诊断可使用

// 可选：启用简单 dv 限幅（不依赖外部设备符号）
#ifndef XSPH_SIMPLE_LIMIT
#define XSPH_SIMPLE_LIMIT1
#endif

namespace {

__device__ inline float3 xsphClampDv(const float3& dv, float h, float dt){
#if XSPH_SIMPLE_LIMIT
 float maxAbs = fmaxf(fmaxf(fabsf(dv.x), fabsf(dv.y)), fabsf(dv.z));
 float limit =5.0f * h / fmaxf(1e-6f, dt);
 if (maxAbs > limit){
 float s = limit / maxAbs;
 return make_float3(dv.x * s, dv.y * s, dv.z * s);
 }
#endif
 return dv;
}

__global__ void KXSPH(
 float4* __restrict__ vel_out,
 const float4* __restrict__ vel_in,
 const float4* __restrict__ pos_pred,
 const uint32_t* __restrict__ indicesSorted,
 const uint32_t* __restrict__ keysSorted,
 const uint32_t* __restrict__ cellStart,
 const uint32_t* __restrict__ cellEnd,
 sim::DeviceParams dp,
 uint32_t N)
{
 uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
 if (sortedIdx >= N) return;

 uint32_t pid = indicesSorted[sortedIdx];
 uint32_t ghostCount = dp.ghostCount;
 uint32_t fluidCount = (ghostCount <= N)? (N - ghostCount): N;
 bool isGhost = (pid >= fluidCount);
 if (isGhost && !dp.ghostContribXsph){ vel_out[pid] = vel_in[pid]; return; }

 const sim::GridBounds& grid = dp.grid;
 const sim::KernelCoeffs& kc = dp.kernel;
 if (dp.xsph_c <=0.f){ vel_out[pid] = vel_in[pid]; return; }

 float3 pi = to_float3(pos_pred[pid]);
 float3 vi = to_float3(vel_in[pid]);

 //还原当前粒子所在 cell
 uint32_t key = keysSorted[sortedIdx];
 int3 ci;
 ci.x = int(key % (uint32_t)grid.dim.x);
 uint32_t key_div_x = key / (uint32_t)grid.dim.x;
 ci.y = int(key_div_x % (uint32_t)grid.dim.y);
 ci.z = int(key_div_x / (uint32_t)grid.dim.y);

 float3 dv_sum = make_float3(0.f,0.f,0.f);
 int neighborCount =0;
 const float h = kc.h;
 const float cs = grid.cellSize;
 int reach = (cs >0.f) ? max(1, int(ceilf(h / cs))) :1;

 for (int dz=-reach; dz<=reach; ++dz){
 for (int dy=-reach; dy<=reach; ++dy){
 for (int dx=-reach; dx<=reach; ++dx){
 int3 cc = make_int3(ci.x+dx, ci.y+dy, ci.z+dz);
 if (!sim::inBounds(cc, grid.dim)) continue;
 uint32_t cidx = sim::linIdx(cc, grid.dim);
 uint32_t beg = cellStart[cidx];
 uint32_t end = cellEnd[cidx];
 if (beg ==0xFFFFFFFFu || beg >= end) continue;
 for (uint32_t k = beg; k < end; ++k){
 uint32_t j = indicesSorted[k];
 bool jGhost = (j >= fluidCount);
 if (jGhost && !dp.ghostContribXsph) continue;
 if (j == pid) continue;
 float3 pj = to_float3(pos_pred[j]);
 float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
 float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
 if (r2 > kc.h2) continue;
 float hr2 = kc.h2 - r2;
 float w = kc.poly6 * hr2 * hr2 * hr2;
 float3 vj = to_float3(vel_in[j]);
 dv_sum.x += (vj.x - vi.x) * w;
 dv_sum.y += (vj.y - vi.y) * w;
 dv_sum.z += (vj.z - vi.z) * w;
 ++neighborCount;
 }
 }
 }
 }

 float gate =1.f;
 if (dp.pbf.xsph_gate_enable){
 int nMin = max(0, dp.pbf.xsph_n_min);
 int nMax = max(nMin+1, dp.pbf.xsph_n_max);
 float t = (float(neighborCount) - float(nMin)) / float(nMax - nMin);
 gate = fminf(1.f, fmaxf(0.f, t));
 }

 // dv_sum -> clamp
 dv_sum = xsphClampDv(dv_sum, kc.h, dp.dt);
 float mass_over_rest = (dp.restDensity >0.f) ? (dp.particleMass / dp.restDensity) :0.f;
 float c = dp.xsph_c * gate * mass_over_rest;
 float3 vnew = make_float3(vi.x + c * dv_sum.x, vi.y + c * dv_sum.y, vi.z + c * dv_sum.z);
 vel_out[pid] = make_float4(vnew.x, vnew.y, vnew.z,0.0f);
}

__global__ void KXSPHCompact(
 float4* __restrict__ vel_out,
 const float4* __restrict__ vel_in,
 const float4* __restrict__ pos_pred,
 const uint32_t* __restrict__ indicesSorted,
 const uint32_t* __restrict__ keysSorted,
 const uint32_t* __restrict__ uniqueKeys,
 const uint32_t* __restrict__ offsets,
 const uint32_t* __restrict__ compactCount,
 sim::DeviceParams dp,
 uint32_t N)
{
 uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
 if (sortedIdx >= N) return;
 uint32_t pid = indicesSorted[sortedIdx];
 uint32_t ghostCount = dp.ghostCount;
 uint32_t fluidCount = (ghostCount <= N)? (N - ghostCount): N;
 bool isGhost = (pid >= fluidCount);
 if (isGhost && !dp.ghostContribXsph){ vel_out[pid] = vel_in[pid]; return; }

 const sim::KernelCoeffs& kc = dp.kernel;
 if (dp.xsph_c <=0.f){ vel_out[pid] = vel_in[pid]; return; }

 float3 pi = to_float3(pos_pred[pid]);
 float3 vi = to_float3(vel_in[pid]);

 //还原 cell
 uint32_t key = keysSorted[sortedIdx];
 int3 ci;
 ci.x = int(key % (uint32_t)dp.grid.dim.x);
 uint32_t key_div_x = key / (uint32_t)dp.grid.dim.x;
 ci.y = int(key_div_x % (uint32_t)dp.grid.dim.y);
 ci.z = int(key_div_x / (uint32_t)dp.grid.dim.y);

 uint32_t M = *compactCount;
 float3 dv_sum = make_float3(0,0,0);
 int neighborCount =0;

 const float h = kc.h;
 const float cs = dp.grid.cellSize;
 int reach = (cs >0.f) ? max(1, int(ceilf(h / cs))) :1;

 for (int dz=-reach; dz<=reach; ++dz){
 for (int dy=-reach; dy<=reach; ++dy){
 for (int dx=-reach; dx<=reach; ++dx){
 int3 cc = make_int3(ci.x+dx, ci.y+dy, ci.z+dz);
 if (!sim::inBounds(cc, dp.grid.dim)) continue;
 uint32_t cidx = sim::linIdx(cc, dp.grid.dim);
 uint32_t beg=0,end=0;
 if(!sim::compact_cell_range(uniqueKeys, offsets, M, cidx, beg, end)) continue;
 for (uint32_t k=beg; k<end; ++k){
 uint32_t j = indicesSorted[k];
 bool jGhost = (j >= fluidCount);
 if (jGhost && !dp.ghostContribXsph) continue;
 if (j == pid) continue;
 float3 pj = to_float3(pos_pred[j]);
 float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
 float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
 if (r2 > kc.h2) continue;
 float hr2 = kc.h2 - r2;
 float w = kc.poly6 * hr2 * hr2 * hr2;
 float3 vj = to_float3(vel_in[j]);
 dv_sum.x += (vj.x - vi.x) * w;
 dv_sum.y += (vj.y - vi.y) * w;
 dv_sum.z += (vj.z - vi.z) * w;
 ++neighborCount;
 }
 }
 }
 }

 float gate =1.f;
 if (dp.pbf.xsph_gate_enable){
 int nMin = max(0, dp.pbf.xsph_n_min);
 int nMax = max(nMin+1, dp.pbf.xsph_n_max);
 float t = (float(neighborCount) - float(nMin)) / float(nMax - nMin);
 gate = fminf(1.f, fmaxf(0.f, t));
 }

 dv_sum = xsphClampDv(dv_sum, kc.h, dp.dt);
 float mass_over_rest = (dp.restDensity >0.f) ? (dp.particleMass / dp.restDensity) :0.f;
 float c = dp.xsph_c * gate * mass_over_rest;
 float3 vnew = make_float3(vi.x + c * dv_sum.x, vi.y + c * dv_sum.y, vi.z + c * dv_sum.z);
 vel_out[pid] = make_float4(vnew.x, vnew.y, vnew.z,0.0f);
}

} // anon

extern "C" void LaunchXSPH(
 float4* vel_out,
 const float4* vel_in,
 const float4* pos_pred,
 const uint32_t* indicesSorted,
 const uint32_t* keysSorted,
 const uint32_t* cellStart,
 const uint32_t* cellEnd,
 sim::DeviceParams dp,
 uint32_t N,
 cudaStream_t s)
{
 if (N ==0 || dp.xsph_c <=0.f) return;
 const int BS=256; dim3 b(BS), g((N+BS-1)/BS);
 KXSPH<<<g,b,0,s>>>(vel_out, vel_in, pos_pred, indicesSorted, keysSorted, cellStart, cellEnd, dp, N);
}

extern "C" void LaunchXSPHCompact(
 float4* vel_out,
 const float4* vel_in,
 const float4* pos_pred,
 const uint32_t* indicesSorted,
 const uint32_t* keysSorted,
 const uint32_t* uniqueKeys,
 const uint32_t* offsets,
 const uint32_t* compactCount,
 sim::DeviceParams dp,
 uint32_t N,
 cudaStream_t s)
{
 if (N ==0 || dp.xsph_c <=0.f) return;
 const int BS=256; dim3 b(BS), g((N+BS-1)/BS);
 KXSPHCompact<<<g,b,0,s>>>(vel_out, vel_in, pos_pred, indicesSorted, keysSorted, uniqueKeys, offsets, compactCount, dp, N);
}