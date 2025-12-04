#pragma once
#include <vector>
#include <random>
#include <cuda_runtime.h>

namespace sim {
void poisson_disk_in_circle(float R, float rMin, int target,
                            std::mt19937& rng,
                            std::vector<float2>& outPts);
}
