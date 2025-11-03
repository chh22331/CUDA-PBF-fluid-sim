#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include "parameters.h"
#include "device_buffers.cuh"
#include "precision_stage.h"

#ifdef __CUDACC__
namespace {
__device__ inline void atomicAddULL(unsigned long long* addr, unsigned long long v){ atomicAdd(addr, v); }
__device__ inline void atomicMaxDouble(double* addr, double v){ unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr); unsigned long long old = *ull; double oldF = __longlong_as_double(old); while (oldF < v){ unsigned long long assumed = old; unsigned long long desired = __double_as_longlong(v); old = atomicCAS(ull, assumed, desired); oldF = __longlong_as_double(old); if (assumed == old) break; } }

__global__ void KVelocityExplodeDiag(const float4* __restrict__ vel,
 const float4* __restrict__ pos,
 const float4* __restrict__ pos_pred,
 const sim::Half4* __restrict__ pos_h4,
 uint32_t N,
 float inv_dt,
 float h,
 int stride,
 float kLimit,
 double* sumSpeed,
 double* sumRelErr,
 double* sumPosQuant,
 double* maxSpeed,
 double* maxRelErr,
 unsigned long long* overLimit,
 unsigned long long* samples)
{
 uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= N) return;
 if (stride <=0) stride =1;
 if ((int)i % stride !=0) return;
 float3 v = make_float3(vel[i].x, vel[i].y, vel[i].z);
 float speed = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
 float3 pc = make_float3(pos[i].x, pos[i].y, pos[i].z);
 float3 pp = make_float3(pos_pred[i].x, pos_pred[i].y, pos_pred[i].z);
 float3 disp = make_float3(pp.x - pc.x, pp.y - pc.y, pp.z - pc.z);
 float3 vRef = make_float3(disp.x * inv_dt, disp.y * inv_dt, disp.z * inv_dt);
 float refSpeed = sqrtf(vRef.x*vRef.x + vRef.y*vRef.y + vRef.z*vRef.z) +1e-9f;
 float3 dv = make_float3(v.x - vRef.x, v.y - vRef.y, v.z - vRef.z);
 float relErr = sqrtf(dv.x*dv.x + dv.y*dv.y + dv.z*dv.z) / refSpeed;
 float posQuant =0.f;
 if (pos_h4){ float3 ph = make_float3(__half2float(pos_h4[i].x), __half2float(pos_h4[i].y), __half2float(pos_h4[i].z)); posQuant = fabsf(ph.x - pc.x) + fabsf(ph.y - pc.y) + fabsf(ph.z - pc.z); }
 atomicAdd(sumSpeed, (double)speed); atomicAdd(sumRelErr, (double)relErr); atomicAdd(sumPosQuant, (double)posQuant); atomicMaxDouble(maxSpeed, (double)speed); atomicMaxDouble(maxRelErr, (double)relErr);
 float speedLimit = kLimit * h * inv_dt; if (speed > speedLimit) atomicAddULL(overLimit,1ull); atomicAddULL(samples,1ull);
}
} // namespace
#endif

extern "C" void LaunchVelocityExplodeDiag(const float4* vel,
 const float4* pos,
 const float4* pos_pred,
 const sim::Half4* pos_h4,
 uint32_t N,
 float inv_dt,
 float h,
 int stride,
 float kLimit,
 double* sumSpeed,
 double* sumRelErr,
 double* sumPosQuant,
 double* maxSpeed,
 double* maxRelErr,
 unsigned long long* overLimit,
 unsigned long long* samples,
 cudaStream_t s)
{
#ifndef __CUDACC__
 (void)vel;(void)pos;(void)pos_pred;(void)pos_h4;(void)N;(void)inv_dt;(void)h;(void)stride;(void)kLimit;(void)sumSpeed;(void)sumRelErr;(void)sumPosQuant;(void)maxSpeed;(void)maxRelErr;(void)overLimit;(void)samples;(void)s; return;
#else
 if (!vel || !pos || !pos_pred || N==0) return; dim3 bs(256); dim3 gs((N + bs.x -1)/bs.x);
 KVelocityExplodeDiag<<<gs, bs,0, s>>>(vel, pos, pos_pred, pos_h4, N, inv_dt, h, stride, kLimit, sumSpeed, sumRelErr, sumPosQuant, maxSpeed, maxRelErr, overLimit, samples);
#endif
}
