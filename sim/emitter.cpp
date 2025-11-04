#include "emitter.h"
#include "logging.h" // Ϊ CUDA_CHECK

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            sim::Log(sim::LogChannel::Error, "CUDA %s (%d)", cudaGetErrorString(_err), (int)_err); \
        } \
    } while (0)
#endif

#include <algorithm>

namespace sim {
    float4* Emitter::s_h_pos = nullptr;
    float4* Emitter::s_h_vel = nullptr;
    uint32_t Emitter::s_h_cap = 0;

    void Emitter::EnsureHostCapacity(uint32_t want) {
        if (s_h_cap >= want) return;
        if (s_h_pos) { cudaFreeHost(s_h_pos); s_h_pos = nullptr; }
        if (s_h_vel) { cudaFreeHost(s_h_vel); s_h_vel = nullptr; }
        if (want == 0) { s_h_cap = 0; return; }
        CUDA_CHECK(cudaMallocHost((void**)&s_h_pos, sizeof(float4) * want));
        CUDA_CHECK(cudaMallocHost((void**)&s_h_vel, sizeof(float4) * want));
        s_h_cap = want;
    }

    void Emitter::GenerateNozzleLattice(uint32_t emit,
                                        const EmitParams& ep,
                                        const SimParams& sp,
                                        uint64_t frameIndex,
                                        std::vector<float2>& poissonPts,
                                        float4* h_pos,
                                        float4* h_vel) {
        auto normalize3 = [](float3 v) {
            float n2 = v.x * v.x + v.y * v.y + v.z * v.z;
            if (n2 <= 1e-20f) return make_float3(0, -1, 0);
            float invn = 1.0f / sqrtf(n2);
            return make_float3(v.x * invn, v.y * invn, v.z * invn);
        };
        auto cross3 = [](float3 a, float3 b) {
            return make_float3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
        };
        auto add3 = [](float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); };
        auto mul3 = [](float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); };

        const float3 dir = normalize3(ep.nozzleDir);
        float3 ref = fabsf(dir.y) < 0.99f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        float3 u = normalize3(cross3(ref, dir));
        float3 v = cross3(dir, u);

        std::mt19937 rng(0xBADC0DEu ^ (uint32_t)frameIndex ^ (uint32_t)sp.numParticles);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float h = sp.kernel.h;
        const float R = (ep.nozzleRadius > 0.f) ? ep.nozzleRadius : (h * 1.5f);
        const float minSpacing = (((1e-6f) > (((console::Instance().sim.poisson_min_spacing_factor_h) * h))) ? (1e-6f) : (((console::Instance().sim.poisson_min_spacing_factor_h) * h)));
        poissonPts.clear();
        poisson_disk_in_circle(R, minSpacing, (int)emit, rng, poissonPts);
        const uint32_t nPoisson = (uint32_t)poissonPts.size();
        const uint32_t nRemain = emit - nPoisson;

        for (uint32_t i = 0; i < nPoisson; ++i) {
            const float2 xy = poissonPts[i];
            float3 offset = add3(mul3(u, xy.x), mul3(v, xy.y));
            float3 p0 = add3(ep.nozzlePos, offset);
            float3 vel0 = mul3(dir, (ep.nozzleSpeed > 0.f ? ep.nozzleSpeed : 1.0f));
            h_pos[i] = make_float4(p0.x, p0.y, p0.z, 1.0f);
            h_vel[i] = make_float4(vel0.x, vel0.y, vel0.z, 0.0f);
        }
        for (uint32_t i = 0; i < nRemain; ++i) {
            const float u0 = U(rng);
            const float u1 = U(rng);
            const float r = R * sqrtf(u0);
            const float th = 6.28318530718f * u1;
            float3 offset = add3(mul3(u, r * cosf(th)), mul3(v, r * sinf(th)));
            float3 p0 = add3(ep.nozzlePos, offset);
            float3 vel0 = mul3(dir, (ep.nozzleSpeed > 0.f ? ep.nozzleSpeed : 1.0f));
            const uint32_t idx = nPoisson + i;
            h_pos[idx] = make_float4(p0.x, p0.y, p0.z, 1.0f);
            h_vel[idx] = make_float4(vel0.x, vel0.y, vel0.z, 0.0f);
        }
    }

    void Emitter::ApplyJitter(uint32_t emit,
                              float4* h_pos,
                              const SimParams& sp,
                              const console::RuntimeConsole& cc,
                              uint64_t frameIndex) {
        if (!cc.sim.emit_jitter_enable || emit == 0) return;
        const float h_use = (sp.kernel.h > 0.f) ? sp.kernel.h : 1.f;
        const float amp = cc.sim.emit_jitter_scale_h * h_use;
        if (amp <= 0.f) return;
        std::mt19937 jrng(cc.sim.emit_jitter_seed ^ (uint32_t)frameIndex);
        std::uniform_real_distribution<float> J(-1.0f, 1.0f);
        for (uint32_t i = 0; i < emit; ++i) {
            float ox, oy, oz;
            for (;;) {
                ox = J(jrng); oy = J(jrng); oz = J(jrng);
                if (ox * ox + oy * oy + oz * oz <= 1.0f) break;
            }
            ox *= amp; oy *= amp; oz *= amp;
            float4 p = h_pos[i];
            p.x = fminf(fmaxf(p.x + ox, sp.grid.mins.x), sp.grid.maxs.x);
            p.y = fminf(fmaxf(p.y + oy, sp.grid.mins.y), sp.grid.maxs.y);
            p.z = fminf(fmaxf(p.z + oz, sp.grid.mins.z), sp.grid.maxs.z);
            h_pos[i] = p;
        }
    }

    uint32_t Emitter::EmitFaucet(DeviceBuffers& buffers,
                                 SimParams& simParams,
                                 const console::RuntimeConsole& cc,
                                 const EmitParams& ep,
                                 uint64_t frameIndex,
                                 cudaStream_t stream) {
        if (!cc.sim.faucetFillEnable) return 0;
        if (simParams.numParticles >= simParams.maxParticles) return 0;
        const uint32_t canEmit = simParams.maxParticles - simParams.numParticles;
        const uint32_t emit = std::min<uint32_t>((uint32_t)cc.sim.emitPerStep, canEmit);
        if (emit == 0) return 0;
        EnsureHostCapacity(emit);
        std::vector<float2> poissonPts; poissonPts.reserve(emit);
        GenerateNozzleLattice(emit, ep, simParams, frameIndex, poissonPts, s_h_pos, s_h_vel);
        ApplyJitter(emit, s_h_pos, simParams, cc, frameIndex);
        const uint32_t begin = simParams.numParticles;
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_pos_curr + begin, s_h_pos, sizeof(float4) * emit, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_vel + begin, s_h_vel, sizeof(float4) * emit, cudaMemcpyHostToDevice, stream));
        simParams.numParticles += emit;
        return emit;
    }
}
