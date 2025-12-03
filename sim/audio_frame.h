#pragma once
#include <cstdint>

namespace sim {

// Shared constants/structures between CPU audio analysis and CUDA kernels.
static constexpr uint32_t kAudioKeyCount = 88u;

struct AudioFrameData {
    float    keyIntensities[kAudioKeyCount] = {};
    float    globalEnergy = 0.0f;
    float    beatStrength = 0.0f;
    uint32_t isBeat = 0;
    uint32_t frameSeed = 0;
    float    globalEnergyEma = 0.0f;
    float    reserved[2] = {};
};

} // namespace sim

