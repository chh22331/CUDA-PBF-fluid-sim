#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "parameters.h"
#include "device_buffers.cuh"

using namespace sim;

extern "C" {

    // ========== 公共工具 ========== //
    __device__ inline void decodeCell(uint32_t key, int3 dim, int& x, int& y, int& z) {
        int xy = dim.x * dim.y;
        z = key / xy;
        uint32_t r = key - uint32_t(z) * xy;
        y = int(r / dim.x);
        x = int(r - uint32_t(y) * dim.x);
    }

    __device__ inline uint32_t makeKey(int x, int y, int z, int3 dim) {
        return (uint32_t)((z * dim.y + y) * dim.x + x);
    }

    // 二分查 uniqueKeys（升序），找不到返回 -1
    __device__ inline int binSearchKey(const uint32_t* uniqueKeys, uint32_t M, uint32_t target) {
        uint32_t lo = 0, hi = M;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            uint32_t v = uniqueKeys[mid];
            if (v < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < M && uniqueKeys[lo] == target) return (int)lo;
        return -1;
    }

    __device__ inline float W_poly6(const KernelCoeffs& kc, float r2) {
        if (r2 >= kc.h2) return 0.f;
        float t = kc.h2 - r2;
        return kc.poly6 * t * t * t;
    }

    __device__ inline float3 grad_spiky(const KernelCoeffs& kc, float3 rij, float r) {
        if (r <= 1e-8f || r >= kc.h) return make_float3(0, 0, 0);
        float t = (kc.h - r);
        float s = kc.spiky * t * t;
        float invr = 1.0f / r;
        return make_float3(s * rij.x * invr, s * rij.y * invr, s * rij.z * invr);
    }

    // ========== Lambda (Compact MP) 每个线程处理排序后条目 k = [0,N) ========== //
    __global__ void KLambdaCompactMP(
        float* __restrict__ d_lambda,
        const float4* __restrict__ d_pos_pred_fp32,
        const Half4* __restrict__ d_pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ keysSorted,
        const uint32_t* __restrict__ uniqueKeys,
        const uint32_t* __restrict__ offsets,     // 长度 >= M+1
        const uint32_t* __restrict__ compactCount,
        DeviceParams dp,
        uint32_t N)
    {
        uint32_t kSelf = blockIdx.x * blockDim.x + threadIdx.x;
        if (kSelf >= N) return;

        const KernelCoeffs kc = dp.kernel;
        const float h2 = kc.h2;
        const float mass = dp.particleMass;
        const float restRho = dp.restDensity;
        const float invRest = (restRho > 0.f) ? (1.f / restRho) : 0.f;

        uint32_t pid = indicesSorted[kSelf];
        float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pid);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);
        uint32_t keySelf = keysSorted[kSelf];

        int3 dim = dp.grid.dim;
        int cx, cy, cz;
        decodeCell(keySelf, dim, cx, cy, cz);

        uint32_t M = *compactCount;

        float rho = 0.f;
        float3 grad_i = make_float3(0, 0, 0);
        float sumGrad2 = 0.f;

        for (int dz = -1; dz <= 1; ++dz) {
            int z = cz + dz;
            if (z < 0 || z >= dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int y = cy + dy;
                if (y < 0 || y >= dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int x = cx + dx;
                    if (x < 0 || x >= dim.x) continue;
                    uint32_t nKey = makeKey(x, y, z, dim);
                    int idxCell = binSearchKey(uniqueKeys, M, nKey);
                    if (idxCell < 0) continue;
                    uint32_t start = offsets[idxCell];
                    uint32_t end = offsets[idxCell + 1];
                    for (uint32_t k = start; k < end; ++k) {
                        uint32_t pjId = indicesSorted[k];
                        float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pjId);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 >= h2) continue;
                        rho += mass * W_poly6(kc, r2);

                        if (pjId != pid) {
                            float r = sqrtf(r2);
                            float3 g = grad_spiky(kc, rij, r);
                            float scale = -mass * invRest;
                            g.x *= scale; g.y *= scale; g.z *= scale;
                            sumGrad2 += g.x * g.x + g.y * g.y + g.z * g.z;
                            grad_i.x -= g.x; grad_i.y -= g.y; grad_i.z -= g.z;
                        }
                    }
                }
            }
        }
        sumGrad2 += grad_i.x * grad_i.x + grad_i.y * grad_i.y + grad_i.z * grad_i.z;
        float C = (restRho > 0.f) ? (rho * invRest - 1.f) : 0.f;
        float denom = sumGrad2 + dp.pbf.lambda_denom_eps;
        float lambda = (denom > 0.f) ? (-C / denom) : 0.f;
        if (dp.pbf.enable_lambda_clamp) {
            float L = dp.pbf.lambda_max_abs;
            lambda = fminf(fmaxf(lambda, -L), L);
        }
        d_lambda[pid] = lambda;
    }

    // ========== Delta Apply (Compact MP) ========== //
    __global__ void KDeltaApplyCompactMP(
        float4* __restrict__ d_pos_pred,
        float4* __restrict__ d_delta,
        const float* __restrict__ d_lambda,
        const float4* __restrict__ d_pos_pred_fp32,
        const Half4* __restrict__ d_pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ keysSorted,
        const uint32_t* __restrict__ uniqueKeys,
        const uint32_t* __restrict__ offsets,
        const uint32_t* __restrict__ compactCount,
        DeviceParams dp,
        uint32_t N)
    {
        uint32_t kSelf = blockIdx.x * blockDim.x + threadIdx.x;
        if (kSelf >= N) return;

        uint32_t pid = indicesSorted[kSelf];
        float lambda_i = d_lambda[pid];

        float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pid);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

        uint32_t keySelf = keysSorted[kSelf];
        int3 dim = dp.grid.dim;
        int cx, cy, cz;
        decodeCell(keySelf, dim, cx, cy, cz);

        uint32_t M = *compactCount;

        const KernelCoeffs kc = dp.kernel;
        const float h2 = kc.h2;
        const float mass = dp.particleMass;
        const float invRest = (dp.restDensity > 0.f) ? (1.f / dp.restDensity) : 0.f;

        float3 disp = make_float3(0, 0, 0);

        for (int dz = -1; dz <= 1; ++dz) {
            int z = cz + dz; if (z < 0 || z >= dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int y = cy + dy; if (y < 0 || y >= dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int x = cx + dx; if (x < 0 || x >= dim.x) continue;
                    uint32_t nKey = makeKey(x, y, z, dim);
                    int idxCell = binSearchKey(uniqueKeys, M, nKey);
                    if (idxCell < 0) continue;
                    uint32_t start = offsets[idxCell];
                    uint32_t end = offsets[idxCell + 1];
                    for (uint32_t k = start; k < end; ++k) {
                        uint32_t pjId = indicesSorted[k];
                        if (pjId == pid) continue;
                        float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pjId);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 >= h2) continue;
                        float r = sqrtf(r2);
                        float3 g = grad_spiky(kc, rij, r);
                        float lambda_j = d_lambda[pjId];
                        float scale = -(lambda_i + lambda_j) * mass * invRest;
                        disp.x += scale * g.x;
                        disp.y += scale * g.y;
                        disp.z += scale * g.z;
                    }
                }
            }
        }

        if (dp.pbf.enable_relax) {
            float w = dp.pbf.relax_omega;
            disp.x *= w; disp.y *= w; disp.z *= w;
        }
        if (dp.pbf.enable_disp_clamp && dp.kernel.h > 0.f) {
            float maxDisp = dp.pbf.disp_clamp_max_h * dp.kernel.h;
            float d2 = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
            float d = sqrtf(d2);
            if (d > maxDisp && d > 1e-12f) {
                float s = maxDisp / d;
                disp.x *= s; disp.y *= s; disp.z *= s;
            }
        }

        float4 outP = d_pos_pred[pid];
        outP.x += disp.x; outP.y += disp.y; outP.z += disp.z;
        d_pos_pred[pid] = outP;
        if (d_delta) d_delta[pid] = make_float4(disp.x, disp.y, disp.z, 0.f);
    }

    // ========== XSPH (Compact MP) ========== //
    __global__ void KXSPHCompactMP(
        float4* __restrict__ vel_out,
        const float4* __restrict__ vel_in_fp32,
        const Half4* __restrict__ vel_in_h4,
        const float4* __restrict__ pos_pred_fp32,
        const Half4* __restrict__ pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ keysSorted,
        const uint32_t* __restrict__ uniqueKeys,
        const uint32_t* __restrict__ offsets,
        const uint32_t* __restrict__ compactCount,
        DeviceParams dp,
        uint32_t N)
    {
        uint32_t kSelf = blockIdx.x * blockDim.x + threadIdx.x;
        if (kSelf >= N) return;
        uint32_t pid = indicesSorted[kSelf];

        float4 vi4 = PrecisionTraits::loadVel(vel_in_fp32, vel_in_h4, pid);
        if (dp.xsph_c <= 0.f) {
            vel_out[pid] = vi4;
            return;
        }

        float4 pi4 = PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, pid);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);
        float3 vi = make_float3(vi4.x, vi4.y, vi4.z);

        uint32_t keySelf = keysSorted[kSelf];
        int3 dim = dp.grid.dim;
        int cx, cy, cz;
        decodeCell(keySelf, dim, cx, cy, cz);

        uint32_t M = *compactCount;

        const KernelCoeffs kc = dp.kernel;
        const float h2 = kc.h2;
        const float mass = dp.particleMass;
        const float invRest = (dp.restDensity > 0.f) ? (1.f / dp.restDensity) : 0.f;

        float3 acc = make_float3(0, 0, 0);

        for (int dz = -1; dz <= 1; ++dz) {
            int z = cz + dz; if (z < 0 || z >= dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int y = cy + dy; if (y < 0 || y >= dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int x = cx + dx; if (x < 0 || x >= dim.x) continue;
                    uint32_t nKey = makeKey(x, y, z, dim);
                    int idxCell = binSearchKey(uniqueKeys, M, nKey);
                    if (idxCell < 0) continue;
                    uint32_t start = offsets[idxCell];
                    uint32_t end = offsets[idxCell + 1];
                    for (uint32_t k = start; k < end; ++k) {
                        uint32_t pjId = indicesSorted[k];
                        if (pjId == pid) continue;
                        float4 pj4 = PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, pjId);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 >= h2) continue;
                        float4 vj4 = PrecisionTraits::loadVel(vel_in_fp32, vel_in_h4, pjId);
                        float3 vj = make_float3(vj4.x, vj4.y, vj4.z);
                        float w = W_poly6(kc, r2);
                        float scale = dp.xsph_c * mass * invRest * w;
                        acc.x += (vj.x - vi.x) * scale;
                        acc.y += (vj.y - vi.y) * scale;
                        acc.z += (vj.z - vi.z) * scale;
                    }
                }
            }
        }

        vi4.x += acc.x; vi4.y += acc.y; vi4.z += acc.z;
        vel_out[pid] = vi4;
    }

    // ========== Host Launch 封装 ========== //
    static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

    void LaunchLambdaCompactMP(
        float* lambda,
        const float4* pos_pred_fp32,
        const Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,
        const uint32_t* uniqueKeys,
        const uint32_t* offsets,
        const uint32_t* compactCount,
        DeviceParams dp,
        uint32_t N,
        cudaStream_t s)
    {
        if (N == 0) return;
        KLambdaCompactMP << <gridFor(N), 256, 0, s >> > (lambda,
            pos_pred_fp32, pos_pred_h4,
            indicesSorted, keysSorted,
            uniqueKeys, offsets, compactCount,
            dp, N);
    }

    void LaunchDeltaApplyCompactMP(
        float4* pos_pred,
        float4* delta,
        const float* lambda,
        const float4* pos_pred_fp32,
        const Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,
        const uint32_t* uniqueKeys,
        const uint32_t* offsets,
        const uint32_t* compactCount,
        DeviceParams dp,
        uint32_t N,
        cudaStream_t s)
    {
        if (N == 0) return;
        KDeltaApplyCompactMP << <gridFor(N), 256, 0, s >> > (
            pos_pred, delta, lambda,
            pos_pred_fp32, pos_pred_h4,
            indicesSorted, keysSorted,
            uniqueKeys, offsets, compactCount,
            dp, N);
    }

    void LaunchXSPHCompactMP(
        float4* vel_out,
        const float4* vel_in_fp32,
        const Half4* vel_in_h4,
        const float4* pos_pred_fp32,
        const Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,
        const uint32_t* uniqueKeys,
        const uint32_t* offsets,
        const uint32_t* compactCount,
        DeviceParams dp,
        uint32_t N,
        cudaStream_t s)
    {
        if (N == 0) return;
        KXSPHCompactMP << <gridFor(N), 256, 0, s >> > (
            vel_out,
            vel_in_fp32, vel_in_h4,
            pos_pred_fp32, pos_pred_h4,
            indicesSorted, keysSorted,
            uniqueKeys, offsets, compactCount,
            dp, N);
    }

} // extern "C"