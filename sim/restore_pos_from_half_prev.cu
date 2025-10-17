#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"

extern "C" __global__ void K_RestorePosFromHalfPrev(const sim::Half4* prev, float4* pos, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    sim::Half4 hp = prev[i];
    pos[i] = make_float4(sim::half_to_float(hp.x),
        sim::half_to_float(hp.y),
        sim::half_to_float(hp.z),
        sim::half_to_float(hp.w)); // 保留 w
}

extern "C" void LaunchRestorePosFromHalfPrev(const sim::Half4* prev, float4* pos, uint32_t N, cudaStream_t s) {
    if (!prev || !pos || N == 0) return;
    const uint32_t tpb = 256;
    const uint32_t blocks = (N + tpb - 1) / tpb;
    K_RestorePosFromHalfPrev << <blocks, tpb, 0, s >> > (prev, pos, N);
}