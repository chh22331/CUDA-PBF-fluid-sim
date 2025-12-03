#include "audio_forces.h"
#include <cuda_runtime.h>

namespace sim {

__device__ __constant__ AudioFrameData g_audioFrameConst;
__device__ __constant__ AudioForceParams g_audioForceParams;

namespace {

__device__ inline float hashFloat(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    return (float)(x & 0x00FFFFFF) / float(0x01000000);
}

__global__ void KApplyAudioForces(float4* posPred, uint32_t numParticles) {
    const AudioForceParams params = g_audioForceParams;
    if (!params.enabled || params.keyCount == 0 || params.invDomainWidth <= 0.0f) return;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float4 p = posPred[idx];

    float norm = (p.x - params.domainMinX) * params.invDomainWidth;
    norm = fminf(fmaxf(norm, 0.0f), 0.9999f);
    int key = (int)(norm * params.keyCount);
    key = (key < 0) ? 0 : ((key >= (int)params.keyCount) ? (params.keyCount - 1) : key);
    float intensity = fmaxf(0.0f, g_audioFrameConst.keyIntensities[key]);

    float surfaceBlend = 0.0f;
    if (params.surfaceFalloff > 1e-3f) {
        float start = params.surfaceY - params.surfaceFalloff;
        surfaceBlend = (p.y - start) / params.surfaceFalloff;
        surfaceBlend = fminf(fmaxf(surfaceBlend, 0.0f), 1.0f);
    }
    else if (p.y >= params.surfaceY) {
        surfaceBlend = 1.0f;
    }

    if (surfaceBlend <= 0.0f) return;

    constexpr float kMinIntensity = 1e-4f;
    float beatStrength = (g_audioFrameConst.isBeat ? g_audioFrameConst.beatStrength : 0.0f);
    float down = 0.0f;
    bool hasImpulse = false;
    if (intensity > kMinIntensity) {
        down += intensity * params.baseStrength;
        hasImpulse = true;
    }
    if (beatStrength > 0.0f) {
        down += beatStrength * params.beatImpulseStrength;
        hasImpulse = true;
    }
    down *= surfaceBlend;
    float dt = (params.dt > 0.0f) ? params.dt : 0.016f;
    if (hasImpulse && down > 0.0f) {
        p.y -= down * dt;
    }

    float globalEnergy = g_audioFrameConst.globalEnergy * params.turbulenceScale * params.globalEnergyScale;
    globalEnergy = fmaxf(globalEnergy, 0.0f);
    if (!hasImpulse && globalEnergy <= 0.0f) {
        return;
    }

    float h = hashFloat(idx * 9781u ^ g_audioFrameConst.frameSeed);
    float lateral = (h - 0.5f) * params.lateralScale * (0.5f + 0.5f * intensity);
    p.x += lateral * dt;
    float h2 = hashFloat((idx + 1337u) * 7411u ^ (g_audioFrameConst.frameSeed * 17u + 3u));
    float lateralZ = (h2 - 0.5f) * params.lateralScale * 0.5f * (intensity + globalEnergy);
    p.z += lateralZ * dt;

    posPred[idx] = p;
}

} // namespace

void UploadAudioFrameData(const AudioFrameData& data, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(g_audioFrameConst, &data, sizeof(AudioFrameData), 0, cudaMemcpyHostToDevice, stream);
}

void UploadAudioForceParams(const AudioForceParams& params, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(g_audioForceParams, &params, sizeof(AudioForceParams), 0, cudaMemcpyHostToDevice, stream);
}

void LaunchAudioForces(float4* posPred, uint32_t numParticles, cudaStream_t stream) {
    if (!posPred || numParticles == 0) return;
    const uint32_t BS = 256;
    dim3 block(BS), grid((numParticles + BS - 1) / BS);
    KApplyAudioForces<<<grid, block, 0, stream>>>(posPred, numParticles);
}

} // namespace sim
