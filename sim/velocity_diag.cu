#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "parameters.h"

extern "C" __global__ void KVelocityDiag(const float4* prev, const float4* curr, uint32_t N, double* sumPrev, double* sumDv, int* hasNaN){
 uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= N) return;
 float4 vp = prev[i];
 float4 vn = curr[i];
 float3 p = make_float3(vp.x, vp.y, vp.z);
 float3 n = make_float3(vn.x, vn.y, vn.z);
 float l1Prev = fabsf(p.x) + fabsf(p.y) + fabsf(p.z);
 float3 dv = make_float3(n.x - p.x, n.y - p.y, n.z - p.z);
 float l1Dv = fabsf(dv.x) + fabsf(dv.y) + fabsf(dv.z);
 if (isnan(n.x) || isnan(n.y) || isnan(n.z)) atomicAdd(hasNaN,1);
 atomicAdd(sumPrev, (double)l1Prev);
 atomicAdd(sumDv, (double)l1Dv);
}

extern "C" void LaunchVelocityDiag(const float4* prev, const float4* curr, uint32_t N, double* sumPrev, double* sumDv, int* hasNaN, cudaStream_t s){
 if (!prev || !curr || N==0) return;
 dim3 bs(256); dim3 gs((N + bs.x -1)/bs.x);
 KVelocityDiag<<<gs, bs,0, s>>>(prev, curr, N, sumPrev, sumDv, hasNaN);
}
