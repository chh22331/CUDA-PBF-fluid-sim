#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "precision_traits.cuh"
#include "parameters.h"
#include "device_buffers.cuh"
#include "device_globals.cuh" // 新增：XSPH诊断符号
namespace sim { extern __device__ uint32_t g_xsphNaNCount; extern __device__ uint32_t g_xsphAnomalyCount; }

using namespace sim;

__device__ inline float W_poly6_x(const KernelCoeffs& kc, float r2) {
 if (r2 >= kc.h2) return 0.f;
 float t = kc.h2 - r2;
 return kc.poly6 * t * t * t;
}

// ============ 公共辅助（Compact 版需要） ============ //
__device__ inline void decodeCell(uint32_t key, int3 dim, int& x, int& y, int& z) {
    int xy = dim.x * dim.y;
    z = int(key / xy);
    uint32_t r = key - uint32_t(z) * xy;
    y = int(r / dim.x);
    x = int(r - uint32_t(y) * dim.x);
}

__device__ inline uint32_t makeKey(int x, int y, int z, int3 dim) {
    return (uint32_t)((z * dim.y + y) * dim.x + x);
}

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

// ============ Dense 半精算术 XSPH ============ //
__global__ void KXSPHHalfDense(
    float4* __restrict__ vel_out,
    const Half4* __restrict__ vel_in_h4,
    const Half4* __restrict__ pos_pred_h4,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    DeviceParams dp,
    uint32_t N,
    int forceFp32Accum)
{
    uint32_t iSorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (iSorted >= N) return;
    uint32_t pid = indicesSorted[iSorted];

    // ===== Ghost filtering =====
    uint32_t ghostCount = dp.ghostCount;
    uint32_t fluidCount = (ghostCount <= N) ? (N - ghostCount) : N;
    bool isGhost = (pid >= fluidCount);
    if (isGhost && !dp.ghostContribXsph) {
        // passthrough (copy input half velocity to fp32 out)
        Half4 vi_h = vel_in_h4[pid];
        vel_out[pid] = make_float4(__half2float(vi_h.x), __half2float(vi_h.y), __half2float(vi_h.z), __half2float(vi_h.w));
        return;
    }

    Half4 vi_h = vel_in_h4[pid];
    Half4 pi_h = pos_pred_h4[pid];

    if (dp.xsph_c <=0.f) {
        vel_out[pid] = make_float4(__half2float(vi_h.x), __half2float(vi_h.y), __half2float(vi_h.z), __half2float(vi_h.w));
        return;
    }

    float3 pi = make_float3(__half2float(pi_h.x), __half2float(pi_h.y), __half2float(pi_h.z));
    float3 vi = make_float3(__half2float(vi_h.x), __half2float(vi_h.y), __half2float(vi_h.z));

#if __CUDA_ARCH__ >=530
    __half ax = __float2half(0.f);
    __half ay = __float2half(0.f);
    __half az = __float2half(0.f);
#endif
    float3 acc_f = make_float3(0,0,0);
    int neighborCount =0;

    const KernelCoeffs kc = dp.kernel;
    const float h2 = kc.h2;
    const float mass = dp.particleMass;
    const float invRest = (dp.restDensity >0.f) ? (1.f / dp.restDensity) :0.f;
    const float coeff = dp.xsph_c * mass * invRest; // same as mass_over_rest * xsph_c

    // decode cell of particle i
    int3 dim = dp.grid.dim;
    float invCell =1.0f / dp.grid.cellSize;
    int cx = (int)floorf((pi.x - dp.grid.mins.x) * invCell);
    int cy = (int)floorf((pi.y - dp.grid.mins.y) * invCell);
    int cz = (int)floorf((pi.z - dp.grid.mins.z) * invCell);
    cx = max(0, min(dim.x -1, cx));
    cy = max(0, min(dim.y -1, cy));
    cz = max(0, min(dim.z -1, cz));

    for (int dz = -1; dz <=1; ++dz) {
        int z = cz + dz; if (z <0 || z >= dim.z) continue;
        for (int dy = -1; dy <=1; ++dy) {
            int y = cy + dy; if (y <0 || y >= dim.y) continue;
            for (int dx = -1; dx <=1; ++dx) {
                int x = cx + dx; if (x <0 || x >= dim.x) continue;
                uint32_t nKey = (uint32_t)((z * dim.y + y) * dim.x + x);
                uint32_t s = cellStart[nKey];
                uint32_t e = cellEnd[nKey];
                if (s ==0xFFFFFFFFu || e ==0xFFFFFFFFu || e <= s) continue;
                for (uint32_t k = s; k < e; ++k) {
                    uint32_t pj = indicesSorted[k];
                    bool jGhost = (pj >= fluidCount);
                    if (jGhost && !dp.ghostContribXsph) continue;
                    if (pj == pid) continue;
                    Half4 pj_h = pos_pred_h4[pj];
                    float3 pjv = make_float3(__half2float(pj_h.x), __half2float(pj_h.y), __half2float(pj_h.z));
                    float3 rij = make_float3(pi.x - pjv.x, pi.y - pjv.y, pi.z - pjv.z);
                    float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                    if (r2 >= h2) continue;
                    Half4 vj_h = vel_in_h4[pj];
                    float3 vj = make_float3(__half2float(vj_h.x), __half2float(vj_h.y), __half2float(vj_h.z));
                    float w = W_poly6_x(kc, r2);
                    float scale = coeff * w;
                    float dxv = (vj.x - vi.x) * scale;
                    float dyv = (vj.y - vi.y) * scale;
                    float dzv = (vj.z - vi.z) * scale;
                    if (forceFp32Accum) {
                        acc_f.x += dxv; acc_f.y += dyv; acc_f.z += dzv;
                    } else {
#if __CUDA_ARCH__ >=530
                        ax = __hadd(ax, __float2half(dxv));
                        ay = __hadd(ay, __float2half(dyv));
                        az = __hadd(az, __float2half(dzv));
#else
                        acc_f.x += dxv; acc_f.y += dyv; acc_f.z += dzv;
#endif
                    }
                    ++neighborCount;
                }
            }
        }
    }

    float3 accFinal;
    if (forceFp32Accum) {
        accFinal = acc_f;
    } else {
#if __CUDA_ARCH__ >=530
        accFinal = make_float3(__half2float(ax), __half2float(ay), __half2float(az));
#else
        accFinal = acc_f;
#endif
    }

    // Neighbor gating identical to FP32 path
    float gate =1.0f;
    if (dp.pbf.xsph_gate_enable) {
        int nMin = max(0, dp.pbf.xsph_n_min);
        int nMax = max(nMin +1, dp.pbf.xsph_n_max);
        float t = (float(neighborCount) - float(nMin)) / float(nMax - nMin);
        gate = fminf(1.f, fmaxf(0.f, t));
    }

    float3 dvApplied = accFinal;
    float maxAbs = fmaxf(fmaxf(fabsf(dvApplied.x), fabsf(dvApplied.y)), fabsf(dvApplied.z));
    if (!isGhost) {
    if (!(maxAbs <=1e6f) || isnan(dvApplied.x) || isnan(dvApplied.y) || isnan(dvApplied.z)) {
 atomicAdd(&g_xsphNaNCount,1u);
 dvApplied = make_float3(0,0,0);
 } else if (maxAbs >5.0f * kc.h / fmaxf(1e-6f, dp.dt)) {
 atomicAdd(&g_xsphAnomalyCount,1u);
 float scale = (5.0f * kc.h / fmaxf(1e-6f, dp.dt)) / maxAbs;
 dvApplied.x *= scale; dvApplied.y *= scale; dvApplied.z *= scale;
 }
 }
 accFinal = dvApplied;

    float4 outV;
    outV.x = vi.x + gate * accFinal.x;
    outV.y = vi.y + gate * accFinal.y;
    outV.z = vi.z + gate * accFinal.z;
    outV.w = __half2float(vi_h.w);
    vel_out[pid] = outV;
}

// ============ Compact 半精算术 XSPH（真正使用稀疏段） ============ //
__global__ void KXSPHHalfCompact(
    float4* __restrict__ vel_out,
    const Half4* __restrict__ vel_in_h4,
    const Half4* __restrict__ pos_pred_h4,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ uniqueKeys,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ compactCount,
    DeviceParams dp,
    uint32_t N,
    int forceFp32Accum)
{
    uint32_t kSelf = blockIdx.x * blockDim.x + threadIdx.x;
    if (kSelf >= N) return;

    uint32_t pid = indicesSorted[kSelf];
    uint32_t ghostCount = dp.ghostCount;
    uint32_t fluidCount = (ghostCount <= N) ? (N - ghostCount) : N;
    bool isGhost = (pid >= fluidCount);
    if (isGhost && !dp.ghostContribXsph) {
        Half4 vi_h_passthrough = vel_in_h4[pid];
        vel_out[pid] = make_float4(__half2float(vi_h_passthrough.x), __half2float(vi_h_passthrough.y), __half2float(vi_h_passthrough.z), __half2float(vi_h_passthrough.w));
        return;
    }

    Half4 vi_h = vel_in_h4[pid];
    Half4 pi_h = pos_pred_h4[pid];

    if (dp.xsph_c <=0.f) {
        vel_out[pid] = make_float4(__half2float(vi_h.x), __half2float(vi_h.y), __half2float(vi_h.z), __half2float(vi_h.w));
        return;
    }

    float3 pi = make_float3(__half2float(pi_h.x), __half2float(pi_h.y), __half2float(pi_h.z));
    float3 vi = make_float3(__half2float(vi_h.x), __half2float(vi_h.y), __half2float(vi_h.z));

#if __CUDA_ARCH__ >=530
    __half ax = __float2half(0.f);
    __half ay = __float2half(0.f);
    __half az = __float2half(0.f);
#endif
    float3 acc_f = make_float3(0,0,0);
    int neighborCount =0;

    uint32_t keySelf = keysSorted[kSelf];
    int3 dim = dp.grid.dim;
    int cx, cy, cz;
    decodeCell(keySelf, dim, cx, cy, cz);

    uint32_t M = *compactCount;

    const KernelCoeffs kc = dp.kernel;
    const float h2 = kc.h2;
    const float mass = dp.particleMass;
    const float invRest = (dp.restDensity >0.f) ? (1.f / dp.restDensity) :0.f;
    const float coeff = dp.xsph_c * mass * invRest;

    for (int dz = -1; dz <=1; ++dz) {
        int z = cz + dz; if (z <0 || z >= dim.z) continue;
        for (int dy = -1; dy <=1; ++dy) {
            int y = cy + dy; if (y <0 || y >= dim.y) continue;
            for (int dx = -1; dx <=1; ++dx) {
                int x = cx + dx; if (x <0 || x >= dim.x) continue;
                uint32_t nKey = makeKey(x, y, z, dim);
                int idxCell = binSearchKey(uniqueKeys, M, nKey);
                if (idxCell <0) continue;
                uint32_t s = offsets[idxCell];
                uint32_t e = offsets[idxCell +1];
                for (uint32_t k = s; k < e; ++k) {
                    uint32_t pj = indicesSorted[k];
                    bool jGhost = (pj >= fluidCount);
                    if (jGhost && !dp.ghostContribXsph) continue;
                    if (pj == pid) continue;
                    Half4 pj_h = pos_pred_h4[pj];
                    float3 pjv = make_float3(__half2float(pj_h.x), __half2float(pj_h.y), __half2float(pj_h.z));
                    float3 rij = make_float3(pi.x - pjv.x, pi.y - pjv.y, pi.z - pjv.z);
                    float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                    if (r2 >= h2) continue;
                    Half4 vj_h = vel_in_h4[pj];
                    float3 vj = make_float3(__half2float(vj_h.x), __half2float(vj_h.y), __half2float(vj_h.z));
                    float w = W_poly6_x(kc, r2);
                    float scale = coeff * w;
                    float dxv = (vj.x - vi.x) * scale;
                    float dyv = (vj.y - vi.y) * scale;
                    float dzv = (vj.z - vi.z) * scale;
                    if (forceFp32Accum) {
                        acc_f.x += dxv; acc_f.y += dyv; acc_f.z += dzv;
                    } else {
#if __CUDA_ARCH__ >=530
                        ax = __hadd(ax, __float2half(dxv));
                        ay = __hadd(ay, __float2half(dyv));
                        az = __hadd(az, __float2half(dzv));
#else
                        acc_f.x += dxv; acc_f.y += dyv; acc_f.z += dzv;
#endif
                    }
                    ++neighborCount;
                }
            }
        }
    }

    float3 accFinal;
    if (forceFp32Accum) {
        accFinal = acc_f;
    } else {
#if __CUDA_ARCH__ >=530
        accFinal = make_float3(__half2float(ax), __half2float(ay), __half2float(az));
#else
        accFinal = acc_f;
#endif
    }

    float gate =1.0f;
    if (dp.pbf.xsph_gate_enable) {
        int nMin = max(0, dp.pbf.xsph_n_min);
        int nMax = max(nMin +1, dp.pbf.xsph_n_max);
        float t = (float(neighborCount) - float(nMin)) / float(nMax - nMin);
        gate = fminf(1.f, fmaxf(0.f, t));
    }

    float3 dvApplied = accFinal;
    float maxAbs = fmaxf(fmaxf(fabsf(dvApplied.x), fabsf(dvApplied.y)), fabsf(dvApplied.z));
    if (!isGhost) {
    if (!(maxAbs <=1e6f) || isnan(dvApplied.x) || isnan(dvApplied.y) || isnan(dvApplied.z)) {
 atomicAdd(&g_xsphNaNCount,1u);
 dvApplied = make_float3(0,0,0);
 } else if (maxAbs >5.0f * kc.h / fmaxf(1e-6f, dp.dt)) {
 atomicAdd(&g_xsphAnomalyCount,1u);
 float scale = (5.0f * kc.h / fmaxf(1e-6f, dp.dt)) / maxAbs;
 dvApplied.x *= scale; dvApplied.y *= scale; dvApplied.z *= scale;
 }
 }
 accFinal = dvApplied;

    float4 outV;
    outV.x = vi.x + gate * accFinal.x;
    outV.y = vi.y + gate * accFinal.y;
    outV.z = vi.z + gate * accFinal.z;
    outV.w = __half2float(vi_h.w);
    vel_out[pid] = outV;
}

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

// ============ Host Launch ============ //
extern "C" void LaunchXSPHHalfDense(
    float4* vel_out,
    const Half4* vel_in_h4,
    const Half4* pos_pred_h4,
    const uint32_t* indicesSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    DeviceParams dp,
    uint32_t N,
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (!vel_in_h4 || !pos_pred_h4 || N == 0) return;
    KXSPHHalfDense<<<gridFor(N), 256, 0, s>>>(
        vel_out, vel_in_h4, pos_pred_h4,
        indicesSorted, cellStart, cellEnd,
        dp, N, forceFp32Accum ? 1 : 0);
}

extern "C" void LaunchXSPHHalfCompact(
    float4* vel_out,
    const Half4* vel_in_h4,
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
    if (!vel_in_h4 || !pos_pred_h4 || N == 0) return;
    if (!uniqueKeys || !offsets || !compactCount) return;
    KXSPHHalfCompact<<<gridFor(N), 256, 0, s>>>(
        vel_out, vel_in_h4, pos_pred_h4,
        indicesSorted, keysSorted,
        uniqueKeys, offsets, compactCount,
        dp, N, forceFp32Accum ? 1 : 0);
}