#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "parameters.h"
#include "device_buffers.cuh"

extern "C" {

    using namespace sim;

    // ========== 通用辅助 ========== //
    struct NeighborIterCtx {
        GridBounds grid;
        float invCell;
        __device__ inline uint3 cellOf(float3 p) const {
            float3 rel = make_float3(p.x - grid.mins.x, p.y - grid.mins.y, p.z - grid.mins.z);
            int cx = (int)floorf(rel.x * invCell);
            int cy = (int)floorf(rel.y * invCell);
            int cz = (int)floorf(rel.z * invCell);
            cx = max(0, min(grid.dim.x - 1, cx));
            cy = max(0, min(grid.dim.y - 1, cy));
            cz = max(0, min(grid.dim.z - 1, cz));
            return make_uint3((unsigned)cx, (unsigned)cy, (unsigned)cz);
        }
        __device__ inline uint32_t hash(int x, int y, int z) const {
            return (uint32_t)((z * grid.dim.y + y) * grid.dim.x + x);
        }
    };

    __device__ inline float W_poly6(const KernelCoeffs& kc, float r2) {
        if (r2 >= kc.h2) return 0.f;
        float t = kc.h2 - r2;
        // 预计算 kc.poly6 = 315 / (64 pi h^9)
        return kc.poly6 * t * t * t;
    }

    __device__ inline float3 grad_spiky(const KernelCoeffs& kc, float3 rij, float r) {
        if (r <= 1e-8f || r >= kc.h) return make_float3(0, 0, 0);
        float t = (kc.h - r);
        float s = kc.spiky * t * t; // kc.spiky = -45/(pi h^6)，此处乘方向 (rij / r)
        float invr = 1.0f / r;
        return make_float3(s * rij.x * invr, s * rij.y * invr, s * rij.z * invr);
    }

    // ========== Lambda 计算（Dense Grid）========== //
    __global__ void KLambdaMPDense(
        float* __restrict__ d_lambda,
        const float4* __restrict__ d_pos_pred_fp32,
        const Half4* __restrict__ d_pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ cellStart,
        const uint32_t* __restrict__ cellEnd,
        DeviceParams dp,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        // 读取粒子 i 的位置（直接用 half 镜像或 fp32 原始）
        float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, i);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

        NeighborIterCtx ctx{ dp.grid, 1.0f / dp.grid.cellSize };
        uint3 ci = ctx.cellOf(pi);

        const KernelCoeffs kc = dp.kernel;
        const float h2 = kc.h2;
        const float restRho = dp.restDensity;
        const float mass = dp.particleMass;
        const float invRest = (restRho > 0.f) ? (1.0f / restRho) : 0.f;

        float rho = 0.f;
        float3 grad_i = make_float3(0, 0, 0);
        float sumGrad2 = 0.f;

        // 遍历 27 邻居
        for (int dz = -1; dz <= 1; ++dz) {
            int cz = (int)ci.z + dz;
            if (cz < 0 || cz >= dp.grid.dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int cy = (int)ci.y + dy;
                if (cy < 0 || cy >= dp.grid.dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int cx = (int)ci.x + dx;
                    if (cx < 0 || cx >= dp.grid.dim.x) continue;
                    uint32_t h = ctx.hash(cx, cy, cz);
                    uint32_t s = cellStart[h];
                    uint32_t e = cellEnd[h];
                    if (s == 0xFFFFFFFFu || e == 0xFFFFFFFFu || e <= s) continue;
                    for (uint32_t k = s; k < e; ++k) {
                        uint32_t j = indicesSorted[k];
                        if (j >= N) continue;
                        float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 >= h2) continue;
                        rho += mass * W_poly6(kc, r2);

                        if (j != i) {
                            float r = sqrtf(r2);
                            float3 g = grad_spiky(kc, rij, r);
                            // ∂C_i/∂x_j = -(m/restRho) * g
                            float scale = -mass * invRest;
                            g.x *= scale; g.y *= scale; g.z *= scale;
                            sumGrad2 += g.x * g.x + g.y * g.y + g.z * g.z;
                            grad_i.x -= g.x;
                            grad_i.y -= g.y;
                            grad_i.z -= g.z;
                        }
                    }
                }
            }
        }
        // 自身梯度贡献
        sumGrad2 += grad_i.x * grad_i.x + grad_i.y * grad_i.y + grad_i.z * grad_i.z;

        // 约束
        float C = (restRho > 0.f) ? (rho * invRest - 1.0f) : 0.f;
        float denom = sumGrad2 + dp.pbf.lambda_denom_eps;
        float lambda = (denom > 0.f) ? (-C / denom) : 0.f;

        if (dp.pbf.enable_lambda_clamp) {
            float absMax = dp.pbf.lambda_max_abs;
            if (lambda > absMax) lambda = absMax;
            if (lambda < -absMax) lambda = -absMax;
        }
        d_lambda[i] = lambda;
    }

    // ========== Delta Apply（Dense Grid）========== //
    __global__ void KDeltaApplyMPDense(
        float4* __restrict__ d_pos_pred,     // 写回仍 FP32
        float4* __restrict__ d_delta,        // 输出位移（供后续聚合/调试，可复用现有缓冲）
        const float* __restrict__ d_lambda,
        const float4* __restrict__ d_pos_pred_fp32,
        const Half4* __restrict__ d_pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ cellStart,
        const uint32_t* __restrict__ cellEnd,
        DeviceParams dp,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 pi4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, i);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

        NeighborIterCtx ctx{ dp.grid, 1.0f / dp.grid.cellSize };
        uint3 ci = ctx.cellOf(pi);

        const KernelCoeffs kc = dp.kernel;
        const float h2 = kc.h2;
        const float mass = dp.particleMass;
        const float invRest = (dp.restDensity > 0.f) ? (1.0f / dp.restDensity) : 0.f;

        float lambda_i = d_lambda[i];
        float3 disp = make_float3(0, 0, 0);

        for (int dz = -1; dz <= 1; ++dz) {
            int cz = (int)ci.z + dz;
            if (cz < 0 || cz >= dp.grid.dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int cy = (int)ci.y + dy;
                if (cy < 0 || cy >= dp.grid.dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int cx = (int)ci.x + dx;
                    if (cx < 0 || cx >= dp.grid.dim.x) continue;
                    uint32_t h = ctx.hash(cx, cy, cz);
                    uint32_t s = cellStart[h];
                    uint32_t e = cellEnd[h];
                    if (s == 0xFFFFFFFFu || e == 0xFFFFFFFFu || e <= s) continue;
                    for (uint32_t k = s; k < e; ++k) {
                        uint32_t j = indicesSorted[k];
                        if (j >= N || j == i) continue;
                        float4 pj4 = PrecisionTraits::loadPosPred(d_pos_pred_fp32, d_pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 >= h2) continue;
                        float r = sqrtf(r2);
                        float3 g = grad_spiky(kc, rij, r); // 与 Lambda 中保持一致
                        float lambda_j = d_lambda[j];
                        // 经典 PBF Δp_i += -(λ_i + λ_j) * (m/ρ0) * g
                        float scale = -(lambda_i + lambda_j) * mass * invRest;
                        disp.x += scale * g.x;
                        disp.y += scale * g.y;
                        disp.z += scale * g.z;
                    }
                }
            }
        }

        // 放松（Jacobi 迭代 ω）
        if (dp.pbf.enable_relax) {
            float w = dp.pbf.relax_omega;
            disp.x *= w; disp.y *= w; disp.z *= w;
        }

        // 位移钳制（保持与 pbf 调参一致）
        if (dp.pbf.enable_disp_clamp && dp.kernel.h > 0.f) {
            float maxDisp = dp.pbf.disp_clamp_max_h * dp.kernel.h;
            float d2 = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
            float d = sqrtf(d2);
            if (d > maxDisp && d > 1e-12f) {
                float s = maxDisp / d;
                disp.x *= s; disp.y *= s; disp.z *= s;
            }
        }

        // 写回 pos_pred
        float4 outP = d_pos_pred[i];
        outP.x += disp.x; outP.y += disp.y; outP.z += disp.z;
        d_pos_pred[i] = outP;

        if (d_delta) {
            d_delta[i] = make_float4(disp.x, disp.y, disp.z, 0.f);
        }
    }

    // ========== XSPH（Dense Grid）========== //
    __global__ void KXSPHMPDense(
        float4* __restrict__ vel_out,
        const float4* __restrict__ vel_in_fp32,
        const Half4* __restrict__ vel_in_h4,
        const float4* __restrict__ pos_pred_fp32,
        const Half4* __restrict__ pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ cellStart,
        const uint32_t* __restrict__ cellEnd,
        DeviceParams dp,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        if (dp.xsph_c <= 0.f) {
            // 直接拷贝
            float4 v4 = PrecisionTraits::loadVel(vel_in_fp32, vel_in_h4, i);
            vel_out[i] = v4;
            return;
        }

        float4 pi4 = PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);
        float4 vi4 = PrecisionTraits::loadVel(vel_in_fp32, vel_in_h4, i);
        float3 vi = make_float3(vi4.x, vi4.y, vi4.z);

        NeighborIterCtx ctx{ dp.grid, 1.0f / dp.grid.cellSize };
        uint3 ci = ctx.cellOf(pi);
        const KernelCoeffs kc = dp.kernel;
        const float h2 = kc.h2;
        const float mass = dp.particleMass;
        const float invRest = (dp.restDensity > 0.f) ? (1.0f / dp.restDensity) : 0.f;

        float3 acc = make_float3(0, 0, 0);

        for (int dz = -1; dz <= 1; ++dz) {
            int cz = (int)ci.z + dz;
            if (cz < 0 || cz >= dp.grid.dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int cy = (int)ci.y + dy;
                if (cy < 0 || cy >= dp.grid.dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int cx = (int)ci.x + dx;
                    if (cx < 0 || cx >= dp.grid.dim.x) continue;
                    uint32_t h = ctx.hash(cx, cy, cz);
                    uint32_t s = cellStart[h];
                    uint32_t e = cellEnd[h];
                    if (s == 0xFFFFFFFFu || e == 0xFFFFFFFFu || e <= s) continue;
                    for (uint32_t k = s; k < e; ++k) {
                        uint32_t j = indicesSorted[k];
                        if (j >= N || j == i) continue;
                        float4 pj4 = PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 >= h2) continue;
                        float4 vj4 = PrecisionTraits::loadVel(vel_in_fp32, vel_in_h4, j);
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
        float4 outV = vi4;
        outV.x += acc.x;
        outV.y += acc.y;
        outV.z += acc.z;
        vel_out[i] = outV;
    }

    // ========== 各阶段 Host Launch 封装（Dense）========== //
    static inline uint32_t LaunchGrid(uint32_t N) {
        const uint32_t t = 256;
        return (N + t - 1) / t;
    }

    void LaunchLambdaMP(
        float* lambda,
        const float4* pos_pred_fp32,
        const Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted, // 未用（保留以对齐原签名）
        const uint32_t* cellStart,
        const uint32_t* cellEnd,
        DeviceParams dp,
        uint32_t N,
        cudaStream_t s)
    {
        (void)keysSorted;
        if (N == 0) return;
        KLambdaMPDense << <LaunchGrid(N), 256, 0, s >> > (
            lambda, pos_pred_fp32, pos_pred_h4,
            indicesSorted, cellStart, cellEnd,
            dp, N);
    }

    void LaunchDeltaApplyMP(
        float4* pos_pred,
        float4* delta,
        const float* lambda,
        const float4* pos_pred_fp32,
        const Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,
        const uint32_t* cellStart,
        const uint32_t* cellEnd,
        DeviceParams dp,
        uint32_t N,
        cudaStream_t s)
    {
        (void)keysSorted;
        if (N == 0) return;
        KDeltaApplyMPDense << <LaunchGrid(N), 256, 0, s >> > (
            pos_pred, delta, lambda,
            pos_pred_fp32, pos_pred_h4,
            indicesSorted, cellStart, cellEnd,
            dp, N);
    }

    void LaunchXSPHMP(
        float4* vel_out,
        const float4* vel_in_fp32,
        const Half4* vel_in_h4,
        const float4* pos_pred_fp32,
        const Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,
        const uint32_t* cellStart,
        const uint32_t* cellEnd,
        DeviceParams dp,
        uint32_t N,
        cudaStream_t s)
    {
        (void)keysSorted;
        if (N == 0) return;
        KXSPHMPDense << <LaunchGrid(N), 256, 0, s >> > (
            vel_out, vel_in_fp32, vel_in_h4,
            pos_pred_fp32, pos_pred_h4,
            indicesSorted, cellStart, cellEnd,
            dp, N);
    }

    // 预留：Compact (hashed) 版本占位（后续 M3 实现真正的稀疏 key 访问）
    // void LaunchLambdaCompactMP(...);
    // void LaunchDeltaApplyCompactMP(...);
    // void LaunchXSPHCompactMP(...);

} // extern "C"