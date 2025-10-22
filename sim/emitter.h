#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "emit_params.h"
#include "device_buffers.cuh"
#include "parameters.h"
#include "poisson_disk.h"
#include "math_utils.h"
#include "logging.h"
#include "../engine/core/console.h"

namespace sim {
    // 负责喷口（faucet）粒子发射逻辑，隔离出 step() 中的复杂随机播种代码。
    class Emitter {
    public:
        // 执行一次发射。返回本帧实际新增粒子数。
        // 要求：simParams.numParticles 已为当前活跃数；不超过 maxParticles。
        static uint32_t EmitFaucet(DeviceBuffers& buffers,
                                   SimParams& simParams,
                                   const console::RuntimeConsole& cc,
                                   const EmitParams& ep,
                                   uint64_t frameIndex,
                                   cudaStream_t stream);
    private:
        // 复用的 host pinned 缓冲（避免频繁分配释放）。
        static float4* s_h_pos;
        static float4* s_h_vel;
        static uint32_t s_h_cap;

        static void EnsureHostCapacity(uint32_t want);
        static void GenerateNozzleLattice(uint32_t emit,
                                          const EmitParams& ep,
                                          const SimParams& sp,
                                          uint64_t frameIndex,
                                          std::vector<float2>& poissonPts,
                                          float4* h_pos,
                                          float4* h_vel);
        static void ApplyJitter(uint32_t emit,
                                float4* h_pos,
                                const SimParams& sp,
                                const console::RuntimeConsole& cc,
                                uint64_t frameIndex);
    };
}
