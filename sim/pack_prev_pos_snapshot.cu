#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
extern "C" __global__ void K_PackFloat4ToHalf4(const float4 * src, sim::Half4 * dst, uint32_t N) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float4 v = src[i];
	sim::Half4 h;
	h.x = sim::float_to_half(v.x);
	h.y = sim::float_to_half(v.y);
	h.z = sim::float_to_half(v.z);
	h.w = sim::float_to_half(v.w); // 保留 w（可用于粒子属性/颜色索引）
	dst[i] = h;
}

extern "C" void LaunchPackFloat4ToHalf4(const float4 * src, sim::Half4 * dst, uint32_t N, cudaStream_t s) {
	if (!src || !dst || N == 0) return;
	uint32_t tpb = 256;
	uint32_t blocks = (N + tpb - 1) / tpb;
	K_PackFloat4ToHalf4 << <blocks, tpb, 0, s >> > (src, dst, N);
}