#pragma once
#include <cuda_runtime.h>
#include "audio_frame.h"
#include "parameters.h"

namespace sim {

void UploadAudioFrameData(const AudioFrameData& data, cudaStream_t stream);
void UploadAudioForceParams(const AudioForceParams& params, cudaStream_t stream);
void LaunchAudioForces(float4* posPred, uint32_t numParticles, cudaStream_t stream);

} // namespace sim

