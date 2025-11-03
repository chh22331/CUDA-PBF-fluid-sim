#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

extern "C" __global__ void KXSphDiagKernel(const float4* velPrev, const float4* velNew, uint32_t N, double* outSumPrev, double* outSumDv, int* outNaN){
 __shared__ double sPrev[128];
 __shared__ double sDv[128];
 __shared__ int sNaN[128];
 int tid = threadIdx.x; sPrev[tid]=0.0; sDv[tid]=0.0; sNaN[tid]=0;
 uint32_t i = blockIdx.x * blockDim.x + tid;
 if(i < N){
 float4 a = velPrev[i];
 float4 b = velNew[i];
 float dvx = b.x - a.x; float dvy = b.y - a.y; float dvz = b.z - a.z;
 double av = fabs((double)a.x) + fabs((double)a.y) + fabs((double)a.z) +1e-20;
 double dv = fabs((double)dvx) + fabs((double)dvy) + fabs((double)dvz);
 sPrev[tid] = av; sDv[tid] = dv;
 if(!isfinite(b.x) || !isfinite(b.y) || !isfinite(b.z)) sNaN[tid]=1;
 }
 __syncthreads();
 for(int ofs = blockDim.x/2; ofs>0; ofs >>=1){
 if(tid < ofs){ sPrev[tid]+=sPrev[tid+ofs]; sDv[tid]+=sDv[tid+ofs]; sNaN[tid]|=sNaN[tid+ofs]; }
 __syncthreads();
 }
 if(tid==0){ atomicAdd(outSumPrev, sPrev[0]); atomicAdd(outSumDv, sDv[0]); if(sNaN[0]) atomicExch(outNaN,1); }
}

extern "C" void LaunchXSphDiag(const float4* velPrev, const float4* velNew, uint32_t N, double* sumPrev, double* sumDv, int* hasNaN, cudaStream_t s){
 if(N==0) return;
 int BS =128; int GS = (N + BS -1)/BS;
 KXSphDiagKernel<<<GS,BS,0,s>>>(velPrev, velNew, N, sumPrev, sumDv, hasNaN);
}
