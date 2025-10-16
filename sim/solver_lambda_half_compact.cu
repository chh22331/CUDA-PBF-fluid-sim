#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "precision_math.cuh"
#include "parameters.h"

using namespace sim;

__device__ inline float W_poly6_f(const KernelCoeffs& kc, float r2) {
    if (r2 >= kc.h2) return 0.f;
    float t = kc.h2 - r2; return kc.poly6 * t * t * t;
}
__device__ inline float3 grad_spiky_f(const KernelCoeffs& kc, float3 rij, float r) {
    if (r <= 1e-8f || r >= kc.h) return make_float3(0, 0, 0);
    float t = kc.h - r;
    float s = kc.spiky * t * t;
    float invr = 1.f / r;
    return make_float3(s * rij.x * invr, s * rij.y * invr, s * rij.z * invr);
}
__device__ inline int binSearchKey(const uint32_t* keys, uint32_t M, uint32_t target) {
    uint32_t lo = 0, hi = M;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint32_t v = keys[mid];
        if (v < target) lo = mid + 1; else hi = mid;
    }
    return (lo < M && keys[lo] == target) ? (int)lo : -1;
}
__device__ inline void decode(uint32_t key, int3 d, int& x, int& y, int& z) {
    int xy = d.x * d.y; z = key / xy;
    uint32_t r = key - uint32_t(z) * xy;
    y = r / d.x;
    x = r - uint32_t(y) * d.x;
}

template<typename DummyA, typename DummyB>
__global__ void KLambdaHalfCompact(
    float* __restrict__ d_lambda,
    const float4* __restrict__ d_pos_pred_fp32,
    const Half4* __restrict__ d_pos_pred_h4,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ uniqueKeys,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ compactCount,
    DeviceParams dp,
    uint32_t N,
    bool halfAccum) // 已废弃, 仅为保持接口; 实际始终用 FP32
{
    (void)halfAccum;

    uint32_t kSelf = blockIdx.x * blockDim.x + threadIdx.x;
    if (kSelf >= N) return;
    uint32_t pid = indicesSorted[kSelf];
    float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, pid);
    float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

    uint32_t keySelf = keysSorted[kSelf];
    int3 dim = dp.grid.dim;
    int cx, cy, cz;
    decode(keySelf, dim, cx, cy, cz);

    uint32_t M = *compactCount;
    const KernelCoeffs kc = dp.kernel;
    const float h2 = kc.h2;
    const float mass = dp.particleMass;
    const float rest = dp.restDensity;
    const float invRest = rest > 0.f ? 1.f / rest : 0.f;

    float rho_f = 0.f;
    float sumG_f = 0.f;
    float3 grad_i = make_float3(0.f, 0.f, 0.f);

    for (int dz = -1; dz <= 1; ++dz) {
        int z = cz + dz; if (z < 0 || z >= dim.z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int y = cy + dy; if (y < 0 || y >= dim.y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int x = cx + dx; if (x < 0 || x >= dim.x) continue;
                uint32_t nk = (uint32_t)((z * dim.y + y) * dim.x + x);
                int idx = binSearchKey(uniqueKeys, M, nk);
                if (idx < 0) continue;
                uint32_t s = offsets[idx];
                uint32_t e = offsets[idx + 1];
                for (uint32_t k = s; k < e; ++k) {
                    uint32_t q = indicesSorted[k];
                    float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, q);
                    float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                    float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                    float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                    if (r2 >= h2) continue;
                    float w = mass * W_poly6_f(kc, r2);
                    rho_f += w;
                    if (q != pid) {
                        float r = sqrtf(r2);
                        float3 g = grad_spiky_f(kc, rij, r);
                        float scale = -mass * invRest;
                        g.x *= scale; g.y *= scale; g.z *= scale;
                        float mag2 = g.x * g.x + g.y * g.y + g.z * g.z;
                        sumG_f += mag2;
                        grad_i.x -= g.x; grad_i.y -= g.y; grad_i.z -= g.z;
                    }
                }
            }
        }
    }
    // self term
    float gi2 = grad_i.x * grad_i.x + grad_i.y * grad_i.y + grad_i.z * grad_i.z;
    sumG_f += gi2;

    float rho = rho_f;
    float sumG = sumG_f;
    float C = rest > 0.f ? (rho * (1.f / rest) - 1.f) : 0.f;
    float denom = sumG + dp.pbf.lambda_denom_eps;
    float lambda = (denom > 0.f) ? (-C / denom) : 0.f;
    if (dp.pbf.enable_lambda_clamp) {
        float L = dp.pbf.lambda_max_abs;
        lambda = fminf(fmaxf(lambda, -L), L);
    }
    d_lambda[pid] = lambda;
}

template<typename DummyA, typename DummyB>
__global__ void KDeltaApplyHalfCompact(
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
    int cx, cy, cz; decode(keySelf, dim, cx, cy, cz);
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
                uint32_t nk = (uint32_t)((z * dim.y + y) * dim.x + x);
                int idx = binSearchKey(uniqueKeys, M, nk);
                if (idx < 0) continue;
                uint32_t s = offsets[idx], e = offsets[idx + 1];
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
                    float lambda_j = d_lambda[pj];
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
        float md = dp.pbf.disp_clamp_max_h * dp.kernel.h;
        float d2 = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
        float d = sqrtf(d2);
        if (d > md && d > 1e-12f) {
            float s = md / d;
            disp.x *= s; disp.y *= s; disp.z *= s;
        }
    }
    float4 outP = d_pos_pred[pid];
    outP.x += disp.x; outP.y += disp.y; outP.z += disp.z;
    d_pos_pred[pid] = outP;
    if (d_delta) d_delta[pid] = make_float4(disp.x, disp.y, disp.z, 0.f);
}

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

extern "C" void LaunchLambdaCompactHalf(
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
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (N == 0) return;
    bool halfAccum = !forceFp32Accum; // 现在忽略
    KLambdaHalfCompact<float, float> << <gridFor(N), 256, 0, s >> > (
        lambda, pos_pred_fp32, pos_pred_h4,
        indicesSorted, keysSorted,
        uniqueKeys, offsets, compactCount,
        dp, N, halfAccum);
}

extern "C" void LaunchDeltaApplyCompactHalf(
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
    bool forceFp32Accum,
    cudaStream_t s)
{
    (void)forceFp32Accum;
    if (N == 0) return;
    KDeltaApplyHalfCompact<float, float> << <gridFor(N), 256, 0, s >> > (
        pos_pred, delta, lambda,
        pos_pred_fp32, pos_pred_h4,
        indicesSorted, keysSorted,
        uniqueKeys, offsets, compactCount,
        dp, N);
}