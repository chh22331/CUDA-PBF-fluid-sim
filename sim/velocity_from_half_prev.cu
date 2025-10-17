#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "../engine/core/console.h"

extern "C" __global__ void K_VelocityFromHalfPrev(float4 * vel,
	const sim::Half4 * prev_pos_h4,
	const float4 * pos_pred,
	float inv_dt,
	uint32_t N)
	{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	sim::Half4 hp = prev_pos_h4[i];
	float3 pPrev = make_float3(sim::half_to_float(hp.x),
		sim::half_to_float(hp.y),
		sim::half_to_float(hp.z));
	float4 c = pos_pred[i];
	float3 pCurr = make_float3(c.x, c.y, c.z);
	float3 v = make_float3((pCurr.x - pPrev.x) * inv_dt,
		(pCurr.y - pPrev.y) * inv_dt,
		(pCurr.z - pPrev.z) * inv_dt);
	vel[i] = make_float4(v.x, v.y, v.z, 0.f);
	}
extern "C" void LaunchVelocityFromHalfPrev(float4 * vel_out,
	const sim::Half4 * prev_pos_h4,
	const float4 * pos_pred,
	float inv_dt,
	uint32_t N,
	cudaStream_t s)
	 {
	if (!vel_out || !prev_pos_h4 || !pos_pred || N == 0) return;
	uint32_t tpb = 256;
	uint32_t blocks = (N + tpb - 1) / tpb;
	K_VelocityFromHalfPrev << <blocks, tpb, 0, s >> > (vel_out, prev_pos_h4, pos_pred, inv_dt, N);
	}