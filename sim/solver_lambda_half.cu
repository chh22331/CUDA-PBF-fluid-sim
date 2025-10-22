#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "precision_math.cuh"
#include "parameters.h"

using namespace sim;

// 通用内联（梯度/核函数保持 FP32）
__device__ inline float W_poly6_f(const KernelCoeffs& kc, float r2) {
    if (r2 >= kc.h2) return 0.f;
    float t = kc.h2 - r2; return kc.poly6 * t * t * t;
}
__device__ inline float3 grad_spiky_f(const KernelCoeffs& kc, float3 rij, float r) {
    if (r <= 1e-8f || r >= kc.h) return make_float3(0, 0, 0);
    float t = (kc.h - r);
    float s = kc.spiky * t * t;
    float invr = 1.0f / r;
    return make_float3(s * rij.x * invr, s * rij.y * invr, s * rij.z * invr);
}

static __device__ inline void decodeKey(uint32_t key, int3 dim, int& cx, int& cy, int& cz) {
    int xy = dim.x * dim.y;
    cz = int(key / xy);
    uint32_t rem = key - uint32_t(cz) * xy;
    cy = int(rem / dim.x);
    cx = int(rem - uint32_t(cy) * dim.x);
}

__global__ void KLambdaWarmStart(float* __restrict__ d_lambda,
    uint32_t existingCount,
    uint32_t totalCount,
    float decay)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalCount) return;
    if (i < existingCount) {
        float v = d_lambda[i];
        d_lambda[i] = v * decay; // 衰减保留
    }
    else {
        d_lambda[i] = 0.f;       // 新发射粒子
    }
}

extern "C" void LaunchLambdaWarmStart(float* lambda,
    uint32_t existingCount,
    uint32_t totalCount,
    float decay,
    cudaStream_t s)
{
    if (!lambda || totalCount == 0) return;
    uint32_t blocks = (totalCount + 255u) / 256u;
    KLambdaWarmStart << <blocks, 256, 0, s >> > (lambda, existingCount, totalCount, decay);
}

// λ计算：读取上一迭代 λ 支持半精镜像（若启用）
__global__ void KLambdaHalfFloatAccum(
    float* __restrict__ d_lambda,
    const float4* __restrict__ d_pos_pred_fp32,
    const Half4* __restrict__ d_pos_pred_h4,
    const float* __restrict__ d_lambda_fp32,
    const __half* __restrict__ d_lambda_h,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    DeviceParams dp,
    uint32_t N)
{
    uint32_t is = blockIdx.x * blockDim.x + threadIdx.x;
    if (is >= N) return;

    uint32_t pid = indicesSorted[is];

    // 读上一迭代 / 上一帧 λ（用于 XPBD）
    float lambda_prev = PrecisionTraits::loadLambda(d_lambda_fp32, d_lambda_h, pid);

    float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pid);
    float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

    uint32_t key = keysSorted[is];
    int3 dim = dp.grid.dim;
    int cx, cy, cz;
    decodeKey(key, dim, cx, cy, cz);

    const KernelCoeffs kc = dp.kernel;
    const float h2 = kc.h2;
    const float mass = dp.particleMass;
    const float rest = dp.restDensity;
    const float invRest = rest > 0.f ? 1.f / rest : 0.f;

    float rho = 0.f;
    float sumGrad2 = 0.f;
    float3 grad_i = make_float3(0.f, 0.f, 0.f);

    for (int dz = -1; dz <= 1; ++dz) {
        int z = cz + dz; if (z < 0 || z >= dim.z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int y = cy + dy; if (y < 0 || y >= dim.y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int x = cx + dx; if (x < 0 || x >= dim.x) continue;
                uint32_t nk = (uint32_t)((z * dim.y + y) * dim.x + x);
                uint32_t s = cellStart[nk];
                uint32_t e = cellEnd[nk];
                if (s == 0xFFFFFFFFu || e == 0xFFFFFFFFu || e <= s) continue;
                for (uint32_t k = s; k < e; ++k) {
                    uint32_t pj = indicesSorted[k];
                    float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pj);
                    float3 pjv = make_float3(pj4.x, pj4.y, pj4.z);
                    float3 rij = make_float3(pi.x - pjv.x, pi.y - pjv.y, pi.z - pjv.z);
                    float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                    if (r2 >= h2) continue;

                    float w = mass * W_poly6_f(kc, r2);
                    rho += w;

                    if (pj != pid) {
                        float r = sqrtf(r2);
                        float3 g = grad_spiky_f(kc, rij, r);
                        float scale = -mass * invRest;
                        g.x *= scale; g.y *= scale; g.z *= scale;
                        // 邻居梯度平方
                        sumGrad2 += g.x * g.x + g.y * g.y + g.z * g.z;
                        // 自身梯度累积
                        grad_i.x -= g.x; grad_i.y -= g.y; grad_i.z -= g.z;
                    }
                }
            }
        }
    }
    // 自身梯度平方
    float grad_i_sq = grad_i.x * grad_i.x + grad_i.y * grad_i.y + grad_i.z * grad_i.z;
    sumGrad2 += grad_i_sq;

    float C = rest > 0.f ? (rho * invRest - 1.f) : 0.f;
    float denom = sumGrad2 + dp.pbf.lambda_denom_eps;

    // XPBD 合规项
    float lambda = 0.f;
    if (dp.pbf.xpbd_enable && dp.pbf.compliance > 0.f && dp.dt > 0.f) {
        float alpha = dp.pbf.compliance / (dp.dt * dp.dt);
        float denom_x = denom + alpha;
        float numer_x = C + alpha * lambda_prev;
        lambda = (denom_x > 0.f) ? (-numer_x / denom_x) : 0.f;
    }
    else {
        lambda = (denom > 0.f) ? (-C / denom) : 0.f;
    }

    if (dp.pbf.enable_lambda_clamp) {
        float L = dp.pbf.lambda_max_abs;
        lambda = fminf(fmaxf(lambda, -L), L);
    }
    d_lambda[pid] = lambda;
}

__global__ void KDeltaApplyHalfGeneric(
    float4* __restrict__ d_pos_pred,
    float4* __restrict__ d_delta,
    const float* __restrict__ d_lambda_fp32,
    const __half* __restrict__ d_lambda_h,
    const float4* __restrict__ d_pos_pred_fp32,
    const Half4* __restrict__ d_pos_pred_h4,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    DeviceParams dp,
    uint32_t N)
{
    uint32_t iSorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (iSorted >= N) return;
    uint32_t pid = indicesSorted[iSorted];

    float lambda_i = PrecisionTraits::loadLambda(d_lambda_fp32, d_lambda_h, pid);

    float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pid);
    float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

    uint32_t key = keysSorted[iSorted];
    int3 dim = dp.grid.dim;
    int xy = dim.x * dim.y;
    int cz = int(key / xy);
    uint32_t rem = key - uint32_t(cz) * xy;
    int cy = int(rem / dim.x);
    int cx = int(rem - uint32_t(cy) * dim.x);

    const KernelCoeffs kc = dp.kernel;
    const float h2 = kc.h2;
    const float mass = dp.particleMass;
    const float invRest = dp.restDensity > 0.f ? 1.0f / dp.restDensity : 0.f;

    float3 disp = make_float3(0, 0, 0);

    for (int dz = -1; dz <= 1; ++dz) {
        int z = cz + dz; if (z < 0 || z >= dim.z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int y = cy + dy; if (y < 0 || y >= dim.y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int x = cx + dx; if (x < 0 || x >= dim.x) continue;
                uint32_t nKey = (uint32_t)((z * dim.y + y) * dim.x + x);
                uint32_t s = cellStart[nKey];
                uint32_t e = cellEnd[nKey];
                if (s == 0xFFFFFFFFu || e == 0xFFFFFFFFu || e <= s) continue;
                for (uint32_t k = s; k < e; ++k) {
                    uint32_t pj = indicesSorted[k];
                    if (pj == pid) continue;
                    float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pj);
                    float3 pjv = make_float3(pj4.x, pj4.y, pj4.z);
                    float3 rij = make_float3(pi.x - pjv.x, pi.y - pjv.y, pi.z - pjv.z);
                    float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                    if (r2 >= h2) continue;
                    float r = sqrtf(r2);
                    float3 g = grad_spiky_f(kc, rij, r);
                    float lambda_j = PrecisionTraits::loadLambda(d_lambda_fp32, d_lambda_h, pj);
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

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

extern "C" void LaunchLambdaHalf(
    float* lambda,
    const float4* pos_pred_fp32,
    const Half4* pos_pred_h4,
    const float* lambda_fp32,
    const __half* lambda_h,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    DeviceParams dp,
    uint32_t N,
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (N == 0) return;
    // 忽略 forceFp32Accum（全部使用 float 累加）
    (void)forceFp32Accum;
    KLambdaHalfFloatAccum << <gridFor(N), 256, 0, s >> > (lambda, pos_pred_fp32, pos_pred_h4, lambda_fp32, lambda_h,
        indicesSorted, keysSorted, cellStart, cellEnd,
        dp, N);
}

extern "C" void LaunchDeltaApplyHalf(
    float4* pos_pred,
    float4* delta,
    const float* lambda_fp32,
    const __half* lambda_h,
    const float4* pos_pred_fp32,
    const Half4* pos_pred_h4,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    DeviceParams dp,
    uint32_t N,
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (N == 0) return;
    bool halfAccum = !forceFp32Accum;
    KDeltaApplyHalfGeneric << <gridFor(N), 256, 0, s >> > (
        pos_pred, delta, lambda_fp32, lambda_h,
        pos_pred_fp32, pos_pred_h4,
        indicesSorted, keysSorted, cellStart, cellEnd,
        dp, N, halfAccum);
}